"""OCR 引擎适配层：把「文件字节 → 文本 / Markdown」抽象成可插拔实现。

服务只依赖 ``Engine`` 这一个接口，换引擎不动 API、也不动调用方代码。两个实现：

* ``PaddleStructureEngine`` —— PaddleOCR 3.x 的 PP-StructureV3，GPU 跑，中文 + 表格 +
  版面还原最强，输出 Markdown，是这个项目的首选；
* ``RapidOCREngine`` —— ``rapidocr-onnxruntime``，CPU 兜底，CUDA 装不动时保证服务可用
  （只有逐行文本，没有版面与表格结构）。

选择顺序由 ``TABLE_PREPASS_ENGINE`` 决定（``paddle`` / ``rapid`` / ``auto``）。
``auto`` 会优先用 Paddle（能 import 就用），否则退回 Rapid。

Paddle 这条路会把 :mod:`ocr_table_prepass.orient` 的前置修正接进流水线：
**先转正页面、擦掉跨页残条，再交给 PP-StructureV3**。
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from ._compat import env_text
from .orient import OrientationPrepass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTS = {".pdf"}


def _default_cache_dirs() -> None:
    """把模型权重默认落到当前目录，而不是 C 盘的用户目录。

    PaddleX 默认写在 ``~/.paddlex``（Windows 即 ``C:\\Users\\<你>\\.paddlex``），
    HuggingFace 系走 ``~/.cache/huggingface``——动辄几个 GB 的权重塞进系统盘很烦人。
    用 ``setdefault`` 让你显式设过的环境变量优先；必须在 import paddlex / paddleocr
    **之前**执行才生效，所以放在模块顶层。
    """

    here = Path.cwd()
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(here / ".paddlex"))
    os.environ.setdefault("HF_HOME", str(here / ".hf_cache"))


_default_cache_dirs()


# --------------------------------------------------------------------------- #
# 子模块开关
# --------------------------------------------------------------------------- #

# PP-StructureV3 默认把版面、表格、公式、图表、印章全跑一遍。财务 / 尽调材料里公式和
# 曲线图基本用不上，而它们既占显存又拖慢推理（6GB 卡上尤其明显）。
# 用 ``TABLE_PREPASS_DISABLE=formula,chart,seal`` 按需关掉。
_MODULE_KWARGS = {
    "orientation": "use_doc_orientation_classify",
    "unwarping": "use_doc_unwarping",
    "textline": "use_textline_orientation",
    "table": "use_table_recognition",
    "formula": "use_formula_recognition",
    "chart": "use_chart_recognition",
    "seal": "use_seal_recognition",
}


def disabled_modules() -> list:
    """读 ``TABLE_PREPASS_DISABLE``，只保留认识的模块名（容忍中文逗号与多余空格）。"""

    raw = (env_text("DISABLE") or "").replace("，", ",")
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return sorted(name for name in wanted if name in _MODULE_KWARGS)


# 默认必须打开的开关。Paddle 这几个的默认值是 False，但对表格识别是错的：
# 整页方向分类（use_doc_orientation_classify）在流水线内部也会做方向纠正，表格单元格
# 能因此拿到更干净的裁剪；真正"把表头列序摆正"的是我们自己的前置转正（见 orient.py）。
# 想关掉做对照：TABLE_PREPASS_DISABLE=orientation
_DEFAULT_ENABLE = {"use_doc_orientation_classify": True}


class EngineError(RuntimeError):
    """引擎不可用或推理失败；消息会原样回给调用方，所以要写人话。"""


class Engine:
    """引擎接口。``run`` 收到原始字节，返回 ``(文本, 元信息)``。"""

    name = "base"
    version = "?"

    def available(self) -> tuple:
        """能否用：``(True, "")`` 或 ``(False, 原因)``。用于 /health 与自动降级判断。"""

        raise NotImplementedError

    def run(self, filename: str, data: bytes, dpi: int, max_pages: int) -> tuple:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 共用小工具
# --------------------------------------------------------------------------- #


def _write_temp(filename: str, data: bytes) -> str:
    """把上传的字节落到临时文件——两个引擎都要真实路径而不是内存流。"""

    fd, path = tempfile.mkstemp(suffix=Path(filename).suffix.lower() or ".bin")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _render_pdf_pages(path: str, dpi: int, max_pages: int) -> list:
    """把 PDF 逐页渲染成 PNG 字节；需要 PyMuPDF。"""

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise EngineError("需要 PyMuPDF 才能渲染 PDF（pip install pymupdf）") from exc

    out: list = []
    doc = fitz.open(path)
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            if len(out) >= max_pages:
                break
            out.append(page.get_pixmap(matrix=matrix, alpha=False).tobytes("png"))
    finally:
        doc.close()
    return out


def _limit_pdf_pages(path: str, max_pages: int) -> tuple:
    """按页数上限裁剪 PDF → ``(要处理的路径, 是否发生了截断)``。

    PaddleOCR 的 ``predict`` 会处理**整份** PDF，不受 ``max_pages`` 约束——不裁的话
    100 页的文件会一路跑到底：既超出调用方的时间预算，也白占显存。裁剪需要 PyMuPDF；
    没有它时按原文件处理并打印告警（宁可慢，也别静默截断）。
    """

    if max_pages <= 0 or not path.lower().endswith(".pdf"):
        return path, False
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[engine] 没有 PyMuPDF，无法按页数上限裁剪")
        return path, False
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[engine] 打开 PDF 失败，按原文件处理：{type(exc).__name__}: {exc}")
        return path, False
    try:
        if doc.page_count <= max_pages:
            return path, False
        out = fitz.open()
        try:
            out.insert_pdf(doc, from_page=0, to_page=max_pages - 1)
            fd, clipped = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            out.save(clipped)
            print(f"[engine] 共 {doc.page_count} 页，按上限只解析前 {max_pages} 页")
            return clipped, True
        finally:
            out.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[engine] 裁剪页数失败，按原文件处理：{type(exc).__name__}: {exc}")
        return path, False
    finally:
        doc.close()


# 前置修正层：进程内共享一个实例（方向模型只加载一次）。
_PREPASS = OrientationPrepass()


# --------------------------------------------------------------------------- #
# PaddleOCR 3.x / PP-StructureV3
# --------------------------------------------------------------------------- #


class PaddleStructureEngine(Engine):
    """PaddleOCR 3.x 的 PP-StructureV3 版面解析流水线（GPU）。"""

    name = "pp-structure-v3"

    def __init__(self) -> None:
        self._pipeline = None
        self._init_error = ""
        self.disabled: list = []

    def _ensure(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._init_error:
            raise EngineError(self._init_error)
        try:
            import paddleocr
        except ImportError as exc:
            self._init_error = "未安装 paddleocr（pip install paddleocr）"
            raise EngineError(self._init_error) from exc
        self.version = getattr(paddleocr, "__version__", "?")
        if not hasattr(paddleocr, "PPStructureV3"):
            self._init_error = (f"paddleocr {self.version} 里没有 PPStructureV3，"
                                "需要 3.x；请确认安装的是 3.x 版本")
            raise EngineError(self._init_error)
        try:
            pipeline, error = self._construct(paddleocr)
            if pipeline is None:
                raise error
            self._pipeline = pipeline
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001 - 模型下载 / 显存 / CUDA 都可能在这里炸
            hint = ""
            if "additional dependencies" in str(exc) or "DependencyError" in type(exc).__name__:
                hint = ("；缺 paddlex 的 ocr 依赖，补装："
                        'python -m pip install "paddlex[ocr]"')
            self._init_error = f"PPStructureV3 初始化失败：{type(exc).__name__}: {exc}{hint}"
            raise EngineError(self._init_error) from exc
        return self._pipeline

    def _construct(self, paddleocr) -> tuple:
        """带子模块开关构造流水线；参数不被接受就退回默认，绝不让引擎起不来。"""

        kwargs = dict(_DEFAULT_ENABLE)
        wanted = disabled_modules()
        kwargs.update({_MODULE_KWARGS[name]: False for name in wanted})   # 显式禁用优先
        try:
            self.disabled = wanted
            return paddleocr.PPStructureV3(**kwargs), None
        except (TypeError, ValueError) as exc:
            # 不同小版本对参数的处理不一样：paddlex 抛的是 ValueError（Unknown argument），
            # Python 原生是 TypeError。两种都要兜住——一个可选开关失效，
            # 绝不能让整条 OCR 链路失效。
            print(f"[engine] 本机版本不接受 {sorted(kwargs)}，改用默认配置：{exc}")
            self.disabled = []
        except Exception as exc:  # noqa: BLE001 - OOM / 模型下载失败等
            self.disabled = []
            return None, exc
        try:
            return paddleocr.PPStructureV3(), None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    def available(self) -> tuple:
        try:
            self._ensure()
            return True, ""
        except EngineError as exc:
            return False, str(exc)

    def run(self, filename: str, data: bytes, dpi: int, max_pages: int) -> tuple:
        pipeline = self._ensure()
        original = _write_temp(filename, data)
        limited = original
        report = None
        out_dir = tempfile.mkdtemp(prefix="ppstructure_")
        try:
            limited, truncated = _limit_pdf_pages(original, max_pages)
            # 前置修正：歪页转正 + 顶部跨页残条（详见 orient.py / fragment.py）
            report = _PREPASS.apply(limited, dpi=dpi)
            result = pipeline.predict(input=report.path or limited)
            markdown = _collect_paddle_markdown(result, out_dir)
            meta = {"pages": _count_paddle_pages(result, markdown), "dpi": dpi,
                    "max_pages": max_pages}
            if truncated:
                meta["truncated"] = True
            if report.rotated:
                meta["rotated_pages"] = sorted(report.rotated)
            if report.strips:
                meta["fragment_strips"] = sorted(report.strips)
            if report.notes:
                meta["prepass_notes"] = list(report.notes)
            return markdown, meta
        finally:
            if report is not None:
                report.cleanup()
            for extra in {limited} - {original, ""}:
                _safe_unlink(extra)
            _safe_unlink(original)
            shutil.rmtree(out_dir, ignore_errors=True)


_PAGE_KEYS = ("page_index", "page_id", "page_no", "page_number")


def _item_page_index(item, fallback: int) -> int:
    """尽量取出这一页在 PDF 里的真实页号；拿不到就退回序号。

    顺序错乱是真实踩过的坑：源 PDF 页序正常（第 6 页是 1.1 定义、第 15 页是 12.2），
    但整份输出里 1.1 跑到了 76% 的位置。所以每页都打上标记，顺序对不对一眼可见。
    """

    data = getattr(item, "json", None)
    for source in (data.get("res") if isinstance(data, dict) else None, data, item):
        if not isinstance(source, dict) and source is not item:
            continue
        for key in _PAGE_KEYS:
            try:
                value = (source.get(key) if isinstance(source, dict)
                         else getattr(source, key, None))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, int) and value >= 0:
                return value + 1          # Paddle 这边按 0 起算
    return fallback


def _natural_key(path: Path) -> list:
    """按文件名里的数字排，别用字典序。

    ``sorted()`` 会把 ``page_10.md`` 排在 ``page_2.md`` 前面——页数一多，输出顺序就整个
    错乱。
    """

    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)]


def _collect_paddle_markdown(result, out_dir: str) -> str:
    """从 PP-StructureV3 的结果对象里取 Markdown。

    PaddleOCR 3.x 各小版本的结果对象属性略有差异，这里按「先直接读、再落盘读」的顺序
    多路尝试，避免绑死某一个版本的 API。装好引擎后第一次跑通时，控制台会打印实际走到的
    分支，便于按真实版本收敛。
    """

    chunks: list = []
    seen: list = []
    items = result if isinstance(result, (list, tuple)) else [result]
    for index, item in enumerate(items, start=1):
        text = _paddle_item_markdown(item)
        if text:
            page = _item_page_index(item, index)
            seen.append(page)
            chunks.append(f"<!-- page {page} -->\n{text}")
    joined = "\n\n".join(c.strip() for c in chunks if c and c.strip())
    if joined:
        print(f"[engine] 结果对象直读成功，共 {len(chunks)} 段，页号 {seen[:25]}")
        return joined

    # 直读失败：让流水线把 Markdown 落盘，再把 .md 读回来。
    for item in items:
        saver = getattr(item, "save_to_markdown", None)
        if callable(saver):
            try:
                saver(save_path=out_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] save_to_markdown 失败：{type(exc).__name__}: {exc}")
    md_files = sorted(Path(out_dir).rglob("*.md"), key=_natural_key)
    joined = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in md_files)
    if joined.strip():
        print(f"[engine] 落盘读取成功，共 {len(md_files)} 个 .md，顺序 "
              f"{[p.name for p in md_files][:20]}")
    return joined.strip()


def _paddle_item_markdown(item) -> str:
    for attr in ("markdown", "md", "text"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    for method in ("to_markdown", "get_markdown"):
        func = getattr(item, method, None)
        if callable(func):
            try:
                value = func()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, str) and value.strip():
                return value
    data = getattr(item, "json", None)
    if isinstance(data, dict):
        return _markdown_from_layout_json(data)
    return ""


def _markdown_from_layout_json(data: dict) -> str:
    """兜底：从版面 JSON 里按阅读顺序拼纯文本（丢版面，但至少不空手而归）。"""

    lines: list = []

    def walk(node) -> None:
        if isinstance(node, dict):
            content = node.get("content") or node.get("text")
            if isinstance(content, str) and content.strip():
                lines.append(content.strip())
            for key in ("layout_det_res", "overall_ocr_res", "parsing_res_list",
                        "table_res_list", "res", "boxes"):
                if key in node:
                    walk(node[key])
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(data)
    return "\n".join(lines)


def _count_paddle_pages(result, markdown: str) -> int:
    items = result if isinstance(result, (list, tuple)) else [result]
    return len(items) if items else (1 if markdown.strip() else 0)


# --------------------------------------------------------------------------- #
# RapidOCR（CPU 兜底）
# --------------------------------------------------------------------------- #


class RapidOCREngine(Engine):
    """rapidocr-onnxruntime：CPU 也能跑，用来兜 CUDA 装不动的局面。"""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None
        self._init_error = ""

    def _ensure(self):
        if self._engine is not None:
            return self._engine
        if self._init_error:
            raise EngineError(self._init_error)
        module = None
        for candidate in ("rapidocr_onnxruntime", "rapidocr"):
            try:
                module = __import__(candidate, fromlist=["RapidOCR"])
                break
            except ImportError:
                continue
        if module is None:
            self._init_error = "未安装 RapidOCR（pip install rapidocr-onnxruntime）"
            raise EngineError(self._init_error)
        self.version = getattr(module, "__version__", "?")
        try:
            self._engine = module.RapidOCR()
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"RapidOCR 初始化失败：{type(exc).__name__}: {exc}"
            raise EngineError(self._init_error) from exc
        return self._engine

    def available(self) -> tuple:
        try:
            self._ensure()
            return True, ""
        except EngineError as exc:
            return False, str(exc)

    def run(self, filename: str, data: bytes, dpi: int, max_pages: int) -> tuple:
        engine = self._ensure()
        pages: list = []
        if Path(filename).suffix.lower() in PDF_EXTS:
            src = _write_temp(filename, data)
            try:
                pages = _render_pdf_pages(src, dpi, max_pages)
            finally:
                _safe_unlink(src)
        else:
            pages = [data]

        texts: list = []
        for index, blob in enumerate(pages, start=1):
            img = _write_temp(f"page{index}.png", blob)
            try:
                result, _elapsed = engine(img)
            finally:
                _safe_unlink(img)
            lines = [row[1] for row in (result or []) if len(row) > 1 and row[1]]
            if lines:
                texts.append(f"<!-- page {index} -->\n" + "\n".join(lines))
        return "\n\n".join(texts).strip(), {"pages": len(pages), "dpi": dpi,
                                            "max_pages": max_pages}


# --------------------------------------------------------------------------- #
# 选择与自检
# --------------------------------------------------------------------------- #


def build_engine(preferred: str = "") -> Engine:
    """按配置挑一个引擎；``auto`` 优先 Paddle，装不上就退 Rapid。"""

    choice = (preferred or env_text("ENGINE", "auto")).strip().lower()
    if choice == "paddle":
        return PaddleStructureEngine()
    if choice in ("rapid", "rapidocr", "cpu"):
        return RapidOCREngine()
    return PaddleStructureEngine() if _paddle_importable() else RapidOCREngine()


def _paddle_importable() -> bool:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


def sniff_kind(filename: str, data: bytes) -> str:
    """粗略判断上传内容类型，供上层做前置校验。"""

    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTS or data[:4] == b"%PDF":
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if zipfile.is_zipfile(io.BytesIO(data[:4096])) or re.match(rb"^PK\x03\x04", data[:4]):
        return "office-zip"
    return ext.lstrip(".") or "unknown"


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
