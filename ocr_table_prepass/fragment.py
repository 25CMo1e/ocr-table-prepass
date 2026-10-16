"""跨页残条（cross-page fragment row）的检测与擦除。

病灶
====
表格跨页时，本页顶端往往只剩上一页切走的**行尾**——很薄的一条续行，带内几乎没字
（真实样张里是「动）。」这种尾巴）。PP-StructureV3 的行界在这种几何下会切错：把下面
几行的行界切到行**内部**，于是同一张表里两家公司的内容互相串行、公司名被拼在一起。

真实案例（43 页并购协议，同一张表跨第 33/34 页）：

* 110 / 120 档渲染：第 34 页两家公司的内容互相串行；
* 100 / 132 / 144 档渲染：各有 1~2 行错开。

修法
====
把残条带**上方那条表格边框线**擦掉（填白），整页行界立刻恢复正确——
110 / 120 / 144 三档 + 整份 43 页文档端到端都验证过。

判据刻意收紧，宁可不修也不误修
==============================

同时满足两条才算命中：

* 页面最上面两条长横线的间距 ≤ 20 磅（``gap_pt``）：残条带很薄；
* 这条带子里的暗像素占比 ≤ 5%（``max_dark``）：带内基本没字。

对照数据：本页正常表头行是 26.8 磅、带内 9.2% 有字 —— 两条都不满足，不会被误判。
即便误擦，对照实验（人为擦掉第 33 页上边框）显示表格结构不受影响。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ._compat import env_flag, env_number, find_numpy, import_pymupdf, warn_once

DEFAULT_GAP_PT = 20.0        # 残条带最大高度（磅）
DEFAULT_MAX_DARK = 0.05      # 残条带内暗像素占比上限
DEFAULT_DARK_BELOW = 150     # 灰度低于它算"墨"
DEFAULT_LINE_FRAC = 0.30     # 一条横线至少要占页宽的多少比例，才算表格边框线
DEFAULT_LINE_GAP = 2         # 线内允许的断口（像素）——扫描件线条常有小缺口
DEFAULT_LINE_ROW_GAP = 2     # 相邻几行并作同一条线（像素）
DEFAULT_MARGIN_PT = 3.0      # 擦除矩形在左右两侧外扩多少（磅）
DEFAULT_PAD_PT = 1.2         # 擦除矩形在上下两侧外扩多少（磅）


def fragment_fix_enabled() -> bool:
    """``TABLE_PREPASS_FRAGMENT_FIX=0`` 可关闭残条修复；默认开。"""

    return env_flag("FRAGMENT_FIX", True)


def gap_limit_pt() -> float:
    """残条带的高度上限（磅），``TABLE_PREPASS_FRAGMENT_GAP_PT`` 可调。"""

    return env_number("FRAGMENT_GAP_PT", DEFAULT_GAP_PT)


def max_dark_ratio() -> float:
    """带内暗像素占比上限，``TABLE_PREPASS_FRAGMENT_MAX_DARK`` 可调。"""

    return env_number("FRAGMENT_MAX_DARK", DEFAULT_MAX_DARK)


# --------------------------------------------------------------------------- #
# 页面图 → 灰度数组
# --------------------------------------------------------------------------- #


def as_gray(image: Any):
    """把各种来源的页面图统一成二维灰度数组（float，0~255）。

    支持：PyMuPDF Pixmap（``samples`` / ``width`` / ``height`` / ``n`` / ``stride``）、
    PIL ``Image``、以及已经是 numpy 数组的灰度图（``H×W``）或彩色图（``H×W×3/4``）。
    认不出来的类型直接抛 ``TypeError``——调用方自己决定是跳过还是报错。
    """

    np = find_numpy()
    if np is None:
        raise RuntimeError("需要 numpy 才能做像素级检测")
    if image is None:
        raise TypeError("页面图为空")

    if hasattr(image, "samples") and hasattr(image, "width") and hasattr(image, "height"):
        width = int(image.width)
        height = int(image.height)
        channels = int(getattr(image, "n", 3) or 3)
        stride = int(getattr(image, "stride", 0) or width * channels)
        raw = np.frombuffer(image.samples, dtype=np.uint8)
        raw = raw[: height * stride].reshape(height, stride)
        array = raw[:, : width * channels].reshape(height, width, channels)
    elif hasattr(image, "convert"):                     # PIL.Image
        array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(image)                       # 已经是数组

    if array.ndim == 3:
        depth = array.shape[2]
        array = array[..., : min(3, depth)].mean(axis=2)
    elif array.ndim != 2:
        raise TypeError(f"看不出这是哪种页面图：shape={getattr(array, 'shape', None)}")

    array = array.astype("float32", copy=False)
    # 有些来源（PIL 的 float 模式、自己归一化过的数组）取值范围是 0~1。
    if array.size and float(array.max()) <= 1.5:
        array = array * 255.0
    return array


# --------------------------------------------------------------------------- #
# 长横线
# --------------------------------------------------------------------------- #


def find_horizontal_lines(dark, *, min_frac: float = DEFAULT_LINE_FRAC,
                          gap: int = DEFAULT_LINE_GAP,
                          row_gap: int = DEFAULT_LINE_ROW_GAP
                          ) -> List[Tuple[int, int, int, int]]:
    """在二维布尔数组里找「长横线」，返回 ``[(y0, y1, x0, x1), ...]``（像素坐标）。

    行内允许 ≤``gap`` 像素的断口（扫描件线条常有小缺口）；相邻行（间隔 ≤``row_gap``）
    并作同一条线。只为残条判据服务，故意写得保守：一条线至少要有页宽的 ``min_frac``。
    """

    height, width = dark.shape
    min_len = max(8, int(round(width * min_frac)))
    lines: List[Tuple[int, int, int, int]] = []
    for y in range(height):
        if int(dark[y].sum()) < min_len * 0.4:      # 先粗筛，省掉逐像素扫描
            continue
        row = dark[y]
        best_len = best_start = 0
        start, gap_run = -1, 0
        for x in range(width):
            if row[x]:
                if start < 0:
                    start = x
                gap_run = 0
                if x - start + 1 > best_len:
                    best_len, best_start = x - start + 1, start
            else:
                gap_run += 1
                if gap_run > gap:
                    start = -1
        if best_len < min_len:
            continue
        x0, x1 = best_start, best_start + best_len - 1
        if lines and y - lines[-1][1] <= row_gap:
            ly0, _ly1, lx0, lx1 = lines[-1]
            lines[-1] = (ly0, y, min(lx0, x0), max(lx1, x1))
        else:
            lines.append((y, y, x0, x1))
    return lines


def find_fragment_strip(image: Any, *, zoom: float = 1.0,
                        gap_pt: Optional[float] = None,
                        max_dark: Optional[float] = None,
                        dark_below: float = DEFAULT_DARK_BELOW,
                        margin_pt: float = DEFAULT_MARGIN_PT,
                        pad_pt: float = DEFAULT_PAD_PT,
                        respect_env: bool = True
                        ) -> Optional[Tuple[float, float, float, float]]:
    """页面顶部若有「跨页残条」，返回要擦掉的矩形 ``(x0, y0, x1, y1)``（**磅**）。

    ``image`` 是页面图（PyMuPDF Pixmap / PIL Image / numpy 数组都行），``zoom`` 是
    它相对 72 DPI 的渲染倍率 —— 像素坐标除以 ``zoom`` 才换算成磅。

    任何前提不满足 / 任何异常都返回 ``None``：**不修可以，改错不行。**
    """

    if respect_env and not fragment_fix_enabled():
        return None
    np = find_numpy()
    if np is None:
        warn_once("numpy", "[prepass] 没有 numpy，跨页残条检测跳过")
        return None
    if zoom <= 0:
        return None
    gap_limit = gap_limit_pt() if gap_pt is None else float(gap_pt)
    dark_limit = max_dark_ratio() if max_dark is None else float(max_dark)
    try:
        gray = as_gray(image)
        dark = gray < dark_below
        lines = find_horizontal_lines(dark)
        if len(lines) < 2:
            return None
        (y0a, y1a, x0a, x1a) = lines[0]
        (y0b, _y1b, _x0b, _x1b) = lines[1]
        if (y0b - y1a) / zoom > gap_limit:
            return None                                  # 离得太远：正常表格行，不是残条
        band = dark[y1a + 1:y0b]
        if band.size and float(band.mean()) > dark_limit:
            return None                                  # 带内字太多：那是数据行
        height, width = dark.shape
        return (max(0.0, x0a / zoom - margin_pt),
                y0a / zoom - pad_pt,
                min(width / zoom, x1a / zoom + margin_pt),
                y1a / zoom + pad_pt)
    except Exception:  # noqa: BLE001 - 探测层绝不把主流程拖下水
        return None


def erase_fragment_strip(page: Any, strip: Sequence[float],
                         *, fill: Tuple[float, float, float] = (1, 1, 1)) -> bool:
    """把残条带上方那条边框线用白色矩形盖掉；成功返回 True。

    ``page`` 是 PyMuPDF 的 Page（要有 ``draw_rect``）。这一步本身出错也不该影响 OCR：
    返回 False，调用方原样保留即可（等同没开这个功能）。
    """

    fitz = import_pymupdf()
    if fitz is None or page is None:
        return False
    try:
        page.draw_rect(fitz.Rect(*[float(v) for v in strip]), color=None, fill=fill)
        return True
    except Exception as exc:  # noqa: BLE001
        warn_once("erase", f"[prepass] 擦除残条边框失败，按原样保留"
                           f"（{type(exc).__name__}: {exc}）")
        return False
