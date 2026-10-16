"""带前置修正的 OCR HTTP 服务（FastAPI）。

接口
----
``GET  /health``        引擎是否就绪 + 说明
``GET  /version``       服务与引擎版本
``POST /jobs``          提交 OCR 任务 → 202 ``{"job_id": ...}``
``GET  /jobs/{id}``     取结果 → ``{"status": "queued|running|done|error", ...}``
``POST /prepass``       只要前置修正：返回修正后的 PDF（base64）+ 逐页报告

设计取舍
--------
* **异步任务制**：跨网络 + 扫描件 PDF 往往几十秒起，阻塞式请求容易被中间设备掐断。
* **单工作线程**：GPU 只有一块显存时，串行跑比并发抢显存稳。
* **Bearer token 鉴权**：服务常挂在组网 / 内网里，不能裸奔。

启动::

    python -m ocr_table_prepass.service            # 等价于下面的 uvicorn 命令
    uvicorn ocr_table_prepass.service:app --host 0.0.0.0 --port 8766

``--host 0.0.0.0`` 别省：省掉它 uvicorn 默认只绑 ``127.0.0.1``，本机 curl 通、
从别的机器一律连不上（现象是 curl 直接 "Couldn't connect"）。
"""

from __future__ import annotations

import base64
import binascii
import os
import queue
import tempfile
import threading
import time
import uuid
from typing import Optional

from . import paddlex_fix
from ._compat import env_flag, env_number, env_text
from .engine import EngineError, build_engine, sniff_kind
from .orient import OrientationPrepass

# 多表格单元格错配的上游修复（PaddleX PR #4756，未被合并）。
# 放在所有推理相关的构造之前——补丁必须早于流水线建立。
print(f"[paddlex-fix] {paddlex_fix.apply()}")

SERVICE_VERSION = "0.1.0"
MAX_UPLOAD_MB = int(env_number("MAX_UPLOAD_MB", 80))
MAX_PAGES = int(env_number("MAX_PAGES", 50))
DEFAULT_DPI = int(env_number("DPI", 200))
JOB_KEEP = 500


class _Auth:
    """鉴权配置按需读取——这样 ``.env`` 在服务启动后才补上也能生效。"""

    @staticmethod
    def token() -> str:
        return env_text("TOKEN")


def _check_token(authorization: Optional[str]) -> None:
    from fastapi import HTTPException

    token = _Auth.token()
    if not token:
        return                     # 未配置 token：仅限本机 / 可信网络调试
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="token 无效")


def _create_app():
    """延迟建 app：没装 FastAPI 时 import 本模块只报一句人话，而不是崩在顶层。"""

    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class JobRequest(BaseModel):
        name: str
        data: str                      # base64
        # 用 Optional[...] 而不是 ``int | None``：Pydantic 会在运行时求值注解，
        # 而 ``X | None`` 这种写法在 Python 3.9 上求值会直接抛 TypeError。
        dpi: Optional[int] = None
        max_pages: Optional[int] = None

    class PrepassRequest(BaseModel):
        name: str
        data: str                      # base64
        fragment_mode: Optional[str] = None

    app = FastAPI(title="ocr-table-prepass", version=SERVICE_VERSION)
    engine = build_engine()
    prepass = OrientationPrepass()
    state: dict = {"warmed": False}
    jobs: dict = {}
    lock = threading.Lock()
    work: "queue.Queue[str]" = queue.Queue()

    # ----------------------------------------------------------------- 任务 #

    def _update(job_id: str, **fields) -> None:
        with lock:
            job = jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _trim_jobs() -> None:
        with lock:
            if len(jobs) <= JOB_KEEP:
                return
            done = sorted((jid for jid, j in jobs.items()
                           if j.get("status") in ("done", "error")),
                          key=lambda jid: jobs[jid].get("finished_at", 0))
            for jid in done[: max(0, len(jobs) - JOB_KEEP)]:
                jobs.pop(jid, None)

    def _run_job(job_id: str) -> None:
        with lock:
            job = dict(jobs.get(job_id) or {})
        if not job:
            return
        _update(job_id, status="running")
        started = time.time()

        ok, why = engine.available()
        if not ok:
            _update(job_id, status="error", error=f"引擎不可用：{why}",
                    elapsed=round(time.time() - started, 2))
            return
        try:
            text, meta = engine.run(job["name"], job["_data"], job["dpi"], job["max_pages"])
        except EngineError as exc:
            _update(job_id, status="error", error=str(exc),
                    elapsed=round(time.time() - started, 2))
            return
        except Exception as exc:  # noqa: BLE001
            _update(job_id, status="error", error=f"{type(exc).__name__}: {exc}",
                    elapsed=round(time.time() - started, 2))
            return

        _update(job_id, status="done", text=text, pages=meta.get("pages", 0),
                engine=engine.name, meta=meta, elapsed=round(time.time() - started, 2))

    def _worker() -> None:
        """单线程串行消费任务队列。"""

        while True:
            job_id = work.get()
            try:
                _run_job(job_id)
            except Exception as exc:  # noqa: BLE001 - 工作线程绝不能死
                _update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                work.task_done()
                _trim_jobs()

    def _public(job: dict) -> dict:
        """对外视图：不暴露内部字段 ``_data``（原始字节可能几十 MB）。"""

        return {k: v for k, v in job.items() if not k.startswith("_")}

    def _decode(data: str) -> bytes:
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="data 不是合法 base64") from exc

    # ---------------------------------------------------------------- 路由 #

    @app.get("/health")
    def health() -> dict:
        ok, why = engine.available()
        return {"ok": ok, "service": SERVICE_VERSION, "engine": engine.name,
                "engine_version": engine.version, "note": why,
                "warming": not state["warmed"],
                "disabled_modules": getattr(engine, "disabled", []),
                "supported": ["pdf", "image"], "max_pages": MAX_PAGES,
                "default_dpi": DEFAULT_DPI,
                "prepass": {"orientation": prepass.enabled(),
                            "upright_dpi": prepass.upright_dpi,
                            "fragment_fix": env_flag("FRAGMENT_FIX", True)}}

    @app.get("/version")
    def version() -> dict:
        return {"service": SERVICE_VERSION, "engine": engine.name,
                "engine_version": engine.version}

    @app.post("/jobs", status_code=202)
    def submit(req: JobRequest, authorization: Optional[str] = Header(default=None)) -> dict:
        _check_token(authorization)
        data = _decode(req.data)
        limit = MAX_UPLOAD_MB * 1024 * 1024
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限")

        kind = sniff_kind(req.name, data)
        if kind not in ("pdf", "image"):
            raise HTTPException(status_code=415,
                                detail=f"本服务只做 OCR（pdf / 图片），收到 {kind}")

        job_id = uuid.uuid4().hex
        with lock:
            jobs[job_id] = {
                "job_id": job_id, "status": "queued", "name": req.name, "kind": kind,
                "bytes": len(data), "dpi": req.dpi or DEFAULT_DPI,
                "max_pages": min(req.max_pages or MAX_PAGES, MAX_PAGES),
                "created_at": time.time(), "_data": data,
            }
        work.put(job_id)
        return {"job_id": job_id, "status": "queued", "queue_len": work.qsize()}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, authorization: Optional[str] = Header(default=None)) -> dict:
        _check_token(authorization)
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="任务不存在或已过期")
            view = _public(job)
        if view.get("status") in ("done", "error"):
            view.setdefault("finished_at", view.get("created_at"))
        return view

    @app.post("/prepass")
    def do_prepass(req: PrepassRequest,
                   authorization: Optional[str] = Header(default=None)) -> dict:
        """只做前置修正：返回修正后的 PDF 与逐页报告。

        给已经有一套 OCR 流水线、只想把这一层插进去的人用——拿到 ``pdf`` 直接喂给
        自己的引擎即可。
        """

        _check_token(authorization)
        data = _decode(req.data)
        fd, src = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        try:
            with open(src, "wb") as fh:
                fh.write(data)
            report = prepass.apply(src, dpi=DEFAULT_DPI,
                                   fragment_mode=req.fragment_mode or "rotated")
            try:
                with open(report.path, "rb") as fh:
                    fixed = fh.read()
            finally:
                report.cleanup()
            return {"name": req.name, "report": report.as_dict(),
                    "pdf": base64.b64encode(fixed).decode("ascii"),
                    "bytes": len(fixed)}
        finally:
            try:
                os.unlink(src)
            except OSError:
                pass

    def _warmup() -> None:
        """提前把模型权重加载进显存。

        不做这一步，第一次 ``/health`` 或第一个任务要等模型初始化完（几十秒），
        容易让调用方误判成"服务没起来"。
        """

        try:
            ok, why = engine.available()
            print(f"[warmup] engine={engine.name} ok={ok} {why}".rstrip())
            if prepass.enabled():
                ready = prepass._classifier.available()  # noqa: SLF001 - 预热同一份模型
                print(f"[warmup] orientation model ok={ready}")
        except Exception as exc:  # noqa: BLE001 - 预热失败不能让服务起不来
            print(f"[warmup] 失败：{type(exc).__name__}: {exc}")
        finally:
            state["warmed"] = True

    threading.Thread(target=_warmup, name="prepass-warmup", daemon=True).start()
    threading.Thread(target=_worker, name="prepass-worker", daemon=True).start()
    return app


try:
    app = _create_app()
except ImportError as _exc:      # pragma: no cover - 只有没装 FastAPI 时走到
    app = None
    print(f"[service] 没有装 FastAPI，HTTP 服务不可用：{_exc}\n"
          "          库本身仍可用：from ocr_table_prepass import prepare_pdf")


def main() -> int:
    if app is None:
        return 2
    import uvicorn

    uvicorn.run(app, host=env_text("HOST", "0.0.0.0"),
                port=int(env_number("PORT", 8766)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
