"""``ocr_table_prepass.service`` 的调用方：把 PDF / 图片交给远端 OCR，拿回 Markdown。

只用标准库（``urllib``），不需要 requests。也可以当模板抄：协议就四个 HTTP 端点。

``OcrClient`` 会自动带上 ``/prepass``，给只想用修正层的场景用：

    client = OcrClient("http://127.0.0.1:8766")
    fixed, report = client.prepass_file("scan.pdf", "scan.fixed.pdf")
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from ._compat import env_number, env_text, warn_once

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 超时预算：默认值必须跟页数走。固定 300 秒和 50 页的上限自相矛盾——
# 按实测约 6 秒/页，50 页光推理就要 300 秒，再算上上传必然超时。
PER_PAGE_BUDGET = 12.0     # 秒/页，给实测值留一倍余量
UPLOAD_BUDGET = 120.0      # 上传 + 排队

# Tailscale 等组网的 100.64.0.0/10 是「非全局」但**不是** is_private，必须单独列。
MESH_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_PRIVATE_SUFFIXES = (".local", ".ts.net", ".lan", ".internal", ".corp", ".localdomain")


@dataclass
class OcrResult:
    """一次 OCR 调用的结果；失败时 ``error`` 一定有值。"""

    text: str = ""
    pages: int = 0
    engine: str = ""
    elapsed: float = 0.0
    cached: bool = False
    meta: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


def budget_for(max_pages: int) -> float:
    return max(300.0, max_pages * PER_PAGE_BUDGET + UPLOAD_BUDGET)


# --------------------------------------------------------------------------- #
# 数据边界（可选）：内部材料只允许发往内网 / 组网地址
# --------------------------------------------------------------------------- #


def _ip_is_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 4 and ip in MESH_NETWORK:
        return True
    return ip.is_private or ip.is_loopback or ip.is_link_local


def endpoint_is_private(url: str) -> Tuple[bool, str]:
    """把地址判定为「内网/组网」还是「公网」。

    主机名会先解析再判断，这样 MagicDNS 名字（``xxx.ts.net``）也能正确识别。
    """

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False, "地址里没有主机名"
    if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES):
        return True, "内网/组网域名"
    try:
        addrs = sorted({info[4][0] for info in socket.getaddrinfo(host, None)})
    except OSError:
        return False, f"无法解析主机名 {host}"
    if addrs and all(_ip_is_private(a) for a in addrs):
        return True, f"内网/组网地址（{', '.join(addrs)}）"
    return False, f"公网地址（{', '.join(addrs)}）"


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class OcrClient:
    """四个端点的薄封装：``/health``、``/version``、``/jobs``、``/prepass``。"""

    def __init__(self, url: str = "", token: str = "", *, timeout: float = 0.0,
                 dpi: int = 0, max_pages: int = 0, cache_dir: Optional[str] = None,
                 allow_public: bool = True) -> None:
        self.url = (url or env_text("URL")).rstrip("/")
        self.token = token or env_text("TOKEN")
        self.dpi = int(dpi or env_number("DPI", 200))
        self.max_pages = int(max_pages or env_number("MAX_PAGES", 50))
        self.timeout = float(timeout or env_number("TIMEOUT", budget_for(self.max_pages)))
        # 结果缓存：默认**关**（库不该偷偷往你的目录里写东西）。
        # 传 cache_dir=... 或设 TABLE_PREPASS_CACHE=<目录> 打开；设成 off / 0 就是关。
        raw_cache = cache_dir if cache_dir is not None else env_text("CACHE")
        if (raw_cache or "").strip().lower() in ("", "off", "0", "none", "no"):
            self.cache_dir = None
        else:
            self.cache_dir = Path(raw_cache)
        self.allow_public = allow_public

    # -- 底层 HTTP --------------------------------------------------------- #

    def _request(self, path: str, method: str = "GET", payload: Optional[dict] = None,
                 timeout: float = 10.0) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers: dict = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, headers=headers,
                                     method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _budget_check(self) -> str:
        """返回拒绝原因，空串表示放行。"""

        if not self.url:
            return "未配置服务地址（TABLE_PREPASS_URL）"
        if self.allow_public:
            return ""
        allowed, why = endpoint_is_private(self.url)
        if allowed:
            return ""
        return (f"数据边界：拒绝把材料发往{why}（{self.url}）；"
                "确需外发请显式打开 allow_public")

    # -- 端点 -------------------------------------------------------------- #

    def health(self, timeout: float = 5.0) -> dict:
        """探一次服务；任何失败都返回 ``{"ok": False, "note": ...}``，不抛异常。"""

        if not self.url:
            return {"ok": False, "note": "未配置服务地址"}
        try:
            return self._request("/health", timeout=timeout)
        except urllib.error.HTTPError as exc:
            return {"ok": False, "note": f"HTTP {exc.code}"}
        except Exception as exc:  # noqa: BLE001 - 探测不该把调用方拖下水
            return {"ok": False, "note": f"{type(exc).__name__}: {exc}"}

    def signature(self, max_age: float = 900.0) -> str:
        """引擎指纹：版本变了 → 已缓存的解析结果要重算。

        **服务在不在线不能影响这个值。** 否则远端一关机，指纹就变，整库会被重解析。
        所以探测失败时退化成稳定的 ``ocr:on``。
        """

        memo = getattr(self, "_sig", None)
        now = time.time()
        if memo and memo[1] == self.url and now - memo[0] < max_age:
            return memo[2]
        info = self.health(timeout=3.0)
        value = (f"ocr:{info.get('engine', '?')}@{info.get('engine_version', '?')}"
                 if info.get("ok") else "ocr:on")
        self._sig = (now, self.url, value)      # type: ignore[attr-defined]
        return value

    # -- 结果缓存 ---------------------------------------------------------- #

    def _cache_path(self, data: bytes) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(data).hexdigest()
        return self.cache_dir / f"{digest}-{self.dpi}-{self.max_pages}.json"

    def _read_cache(self, path: Path, sig: str) -> Optional[OcrResult]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if payload.get("sig") != sig or not payload.get("text"):
            return None
        return OcrResult(text=payload["text"], pages=int(payload.get("pages") or 0),
                         engine=str(payload.get("engine") or ""),
                         elapsed=float(payload.get("elapsed") or 0.0),
                         meta=payload.get("meta") or {}, cached=True)

    def _write_cache(self, path: Path, sig: str, result: OcrResult) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "sig": sig, "engine": result.engine, "pages": result.pages,
                "elapsed": result.elapsed, "meta": result.meta, "text": result.text,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # 缓存写不进去不影响本次结果

    # -- 主流程 ------------------------------------------------------------ #

    def ocr_bytes(self, name: str, data: bytes, *, use_cache: bool = True) -> OcrResult:
        """把一份文件交给远端 OCR，返回文本或失败原因。永不抛异常。"""

        blocked = self._budget_check()
        if blocked:
            return OcrResult(error=blocked)

        sig = self.signature()
        cache_path = self._cache_path(data)
        if use_cache and cache_path is not None:
            hit = self._read_cache(cache_path, sig)
            if hit is not None:
                return hit

        started = time.time()
        try:
            job = self._request("/jobs", "POST", {
                "name": name,
                "data": base64.b64encode(data).decode("ascii"),
                "dpi": self.dpi,
                "max_pages": self.max_pages,
            }, timeout=30.0)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")[:200]
            return OcrResult(error=f"提交失败 HTTP {exc.code}：{body}")
        except Exception as exc:  # noqa: BLE001
            return OcrResult(error=f"提交失败 {type(exc).__name__}: {exc}")

        job_id = job.get("job_id")
        if not job_id:
            return OcrResult(error=f"服务未返回 job_id：{str(job)[:200]}")

        deadline = started + self.timeout
        delay, misses = 1.0, 0
        while time.time() < deadline:
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)
            try:
                view = self._request(f"/jobs/{job_id}", timeout=15.0)
            except Exception:  # noqa: BLE001 - 轮询失败不算致命，继续等
                misses += 1
                if misses >= 5:
                    return OcrResult(error="连续 5 次取结果失败，放弃")
                continue
            misses = 0
            status = view.get("status")
            if status == "done":
                result = OcrResult(text=view.get("text") or "",
                                   pages=int(view.get("pages") or 0),
                                   engine=str(view.get("engine") or ""),
                                   elapsed=float(view.get("elapsed") or 0.0),
                                   meta=view.get("meta") or {})
                if use_cache and cache_path is not None:
                    self._write_cache(cache_path, sig, result)
                return result
            if status == "error":
                return OcrResult(error=f"服务端报错：{str(view.get('error') or '未知')[:200]}")
        return OcrResult(error=f"等待超时（>{self.timeout:.0f}s）")

    def ocr_file(self, path, *, use_cache: bool = True) -> OcrResult:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return OcrResult(error=f"无法读取文件：{exc}")
        return self.ocr_bytes(path.name, data, use_cache=use_cache)

    def prepass_bytes(self, name: str, data: bytes,
                      fragment_mode: str = "rotated") -> Tuple[bytes, dict]:
        """只要前置修正层：返回 ``(修正后的 PDF 字节, 报告)``；失败时抛异常。"""

        blocked = self._budget_check()
        if blocked:
            raise RuntimeError(blocked)
        payload = self._request("/prepass", "POST", {
            "name": name, "data": base64.b64encode(data).decode("ascii"),
            "fragment_mode": fragment_mode,
        }, timeout=self.timeout)
        return base64.b64decode(payload["pdf"]), payload.get("report") or {}

    def prepass_file(self, path, out_path: Optional[str] = None,
                     fragment_mode: str = "rotated") -> Tuple[bytes, dict]:
        """只要前置修正层：把结果写盘（给了 ``out_path`` 时），并返回字节与报告。"""

        path = Path(path)
        fixed, report = self.prepass_bytes(path.name, path.read_bytes(), fragment_mode)
        if out_path:
            Path(out_path).write_bytes(fixed)
        return fixed, report


def default_client() -> OcrClient:
    """按环境变量建一个客户端（``TABLE_PREPASS_URL`` / ``_TOKEN`` …）。"""

    if not env_text("URL"):
        warn_once("no-url", "[client] 未配置 TABLE_PREPASS_URL，客户端暂时用不了")
    return OcrClient()
