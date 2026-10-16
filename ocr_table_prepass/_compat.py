"""可选依赖与环境变量的统一入口。

两条原则贯穿整个包：

1. **可选依赖降级，不抛异常。** numpy / PyMuPDF / paddleocr 任何一个缺失，都只是
   "这一层修正不生效"，主流程照常跑——OCR 结果差一点，比整个任务失败好得多。
2. **环境变量双前缀。** 新前缀 ``TABLE_PREPASS_*`` 优先；如果没设，就去看这个库被
   抽出来之前用的旧前缀 ``COMPS_OCR_*``。已经在跑旧部署的人不用改任何配置。
"""

from __future__ import annotations

import os
from typing import Any, Optional

ENV_PREFIX = "TABLE_PREPASS_"
LEGACY_PREFIX = "COMPS_OCR_"

_WARNED: dict = {}


def warn_once(key: str, message: str) -> None:
    """同一条降级提示只打一次，避免逐页刷屏。"""

    if key in _WARNED:
        return
    _WARNED[key] = True
    print(message)


def env_raw(name: str) -> Optional[str]:
    """取环境变量：``TABLE_PREPASS_<name>`` → ``COMPS_OCR_<name>``，都没有则 None。"""

    value = os.environ.get(ENV_PREFIX + name)
    if value is None or not value.strip():
        value = os.environ.get(LEGACY_PREFIX + name)
    if value is None or not value.strip():
        return None
    return value.strip()


def env_text(name: str, default: str = "") -> str:
    return env_raw(name) or default


def env_flag(name: str, default: bool = True) -> bool:
    raw = env_raw(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def env_number(name: str, default: float) -> float:
    raw = env_raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def find_numpy():
    """拿到 numpy 模块；没装就返回 None（并只提示一次）。"""

    try:
        import numpy as np
    except Exception:  # noqa: BLE001 - 缺包 / 环境损坏都当"没有"
        warn_once("numpy", "[prepass] 没有 numpy，跨页残条检测跳过")
        return None
    return np


def import_pymupdf():
    """拿到 PyMuPDF 模块；没装就返回 None。

    包名有两个：新的 ``pymupdf``，和沿用很久的 ``fitz``。两个都试。
    """

    try:
        import pymupdf  # type: ignore[import-not-found]

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore[import-not-found]

        return fitz
    except ImportError:
        warn_once("pymupdf", "[prepass] 没有 PyMuPDF，页面转正与残条擦除跳过")
        return None


def cleanup_temp(path: str) -> None:
    """删临时文件；删不掉也不叫什么大事。"""

    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def describe(value: Any, limit: int = 160) -> str:
    """把任意值压成一行短摘要，日志用。"""

    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")
