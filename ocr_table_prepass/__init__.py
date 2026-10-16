"""ocr-table-prepass —— 表格 OCR 的前置修正层。

三层互相独立、都能单独用：

* :mod:`ocr_table_prepass.orient` —— 逐页方向转正（在流水线**之前**纠正坐标系）
* :mod:`ocr_table_prepass.fragment` —— 跨页残条检测与擦除
* :mod:`ocr_table_prepass.paddlex_fix` —— 多表格单元格错配的运行时补丁

最常用的一行：:func:`prepare_pdf`。
"""

from .client import OcrClient, OcrResult, endpoint_is_private
from .fragment import (
    erase_fragment_strip,
    find_fragment_strip,
    find_horizontal_lines,
    fragment_fix_enabled,
)
from .orient import (
    ANGLES,
    PROBE_DPI,
    UPRIGHT_DPI,
    OrientationPrepass,
    OrientationReport,
    PageOrientationClassifier,
    prepare_pdf,
)

__version__ = "0.1.0"

__all__ = [
    "ANGLES",
    "PROBE_DPI",
    "UPRIGHT_DPI",
    "OcrClient",
    "OcrResult",
    "OrientationPrepass",
    "OrientationReport",
    "PageOrientationClassifier",
    "__version__",
    "endpoint_is_private",
    "erase_fragment_strip",
    "find_fragment_strip",
    "find_horizontal_lines",
    "fragment_fix_enabled",
    "prepare_pdf",
]
