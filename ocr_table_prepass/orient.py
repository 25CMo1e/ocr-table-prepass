"""逐页方向转正：在版面/表格结构识别**之前**，把横躺的页面转正。

为什么必须在流水线之前做
========================
实测：PP-StructureV3 内部的方向模块**是在工作的**（一张 180° 倒置的正文页，41 行里
只有 3 个标点不同，其余逐字读对——没有方向纠正是做不到的），但它纠正后的坐标系
**没有传给表格结构那一步**：表格的单元格顺序仍按**原始朝向**计算。

症状很隐蔽：整页旋转 90° 的扫描页上，一张 9 列表格的表头会整列错位（「序号」跑到
第 7 列）。把那张图手动转正后再跑，表头与原文列序一字不差。所以我们自己先转正，
让"原始朝向"就是正的。

模型
====
PaddleOCR 3.x 的 ``DocImgOrientationClassification``（文档图像方向分类），四分类
0 / 90 / 180 / 270。这个模型在 PP-StructureV3 内部也在用，区别只是我们**单独调用它**、
并且**在流水线之外完成转正**。

两个实打实踩过的细节
====================

1. **predict 必须传文件路径。** 它的入参是可迭代对象，直接喂 ``bytes`` 会被当成
   "一批输入"，逐字节跑一遍，输出直接爆掉。
2. **重建页的渲染 DPI 要低（110），页面尺寸要用真实物理尺寸。** 同一张会被判错的
   横躺表格做四个对照，只有"流水线把页面放大约 1.2~1.44 倍"这一档判对：

   ===========  ======  ======  ========  ========
   对照档位      1.20x   1.44x   1:1       0.54x
   判定          ✅      ✅      ❌        ❌
   ===========  ======  ======  ========  ========

   放大相当于一次低通滤波，把源扫描件（264 DPI JPEG）的振铃抹掉；1:1 或缩小会把
   振铃原样带进去。所以按 110 DPI 渲染位图、页面写成真实物理尺寸，流水线按它固定的
   144 DPI 渲染即得约 1.31x —— 落在 1.20~1.44 这个窗口里。

   页面尺寸的单位是**点**（1/72 英寸）：像素数要除以渲染倍率才是物理尺寸。直接拿像素
   当点会让页面放大 zoom 倍（实测 A4 变 A2），流水线再渲染时超出它的 4000px 上限被
   强制降采样，那一页的精度和尺度就跟别的页对不上。

``TABLE_PREPASS_ORIENT=0`` 可以整层关掉（做对照实验时用）。
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._compat import cleanup_temp, env_flag, env_number, import_pymupdf, warn_once
from .fragment import erase_fragment_strip, find_fragment_strip

# 逐页判角度用的低分辨率：够用且快。
PROBE_DPI = 100
# 重建转正页用的渲染 DPI。刻意低于流水线自己的渲染 DPI —— 见模块开头第二点。
UPRIGHT_DPI = 110
# 方向模型只会给出这四个值。
ANGLES = (0, 90, 180, 270)


@dataclass
class OrientationReport:
    """一次前置修正的结果。``path`` 才是要送进 OCR 的那个文件。"""

    source: str = ""
    path: str = ""
    pages: int = 0
    angles: List[int] = field(default_factory=list)      # 逐页判定角度，-1 = 没判出来
    rotated: Dict[int, int] = field(default_factory=dict)  # 页码(1 起) -> 角度
    strips: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    fragment_mode: str = "rotated"
    elapsed: float = 0.0
    temporary: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """有没有真的改动过页面（没改动时 ``path`` 就是原文件）。"""

        return bool(self.path) and self.path != self.source

    def as_dict(self) -> dict:
        return {
            "source": os.path.basename(self.source),
            "pages": self.pages,
            "angles": self.angles,
            "rotated_pages": {str(k): v for k, v in sorted(self.rotated.items())},
            "fragment_strips": {str(k): [round(v, 1) for v in rect]
                                for k, rect in sorted(self.strips.items())},
            "fragment_mode": self.fragment_mode,
            "changed": self.changed,
            "elapsed": round(self.elapsed, 2),
            "notes": list(self.notes),
        }

    def cleanup(self) -> None:
        """删掉这次生成的临时 PDF（``path`` 不是原文件、且确实是临时文件时）。"""

        if self.temporary and self.changed:
            cleanup_temp(self.path)


class PageOrientationClassifier:
    """单独调用文档图像方向分类模型：给一张页面图，返回它现在旋转了多少度。"""

    def __init__(self, model: Any = None) -> None:
        self._model = model
        self._tmp = ""
        self._tried = False
        self._error = ""

    # -- 生命周期 ---------------------------------------------------------- #

    def _ensure(self) -> bool:
        """懒加载模型；任何一步失败都只是"这一层不生效"。"""

        if self._model is not None:
            return True
        if self._tried:
            return False
        self._tried = True
        try:
            from paddleocr import DocImgOrientationClassification
        except ImportError as exc:
            self._error = f"没有 paddleocr（{exc}）"
            warn_once("orient-model", f"[prepass] 方向模型不可用：{self._error}，跳过转正")
            return False
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            warn_once("orient-model", f"[prepass] 方向模型不可用：{self._error}，跳过转正")
            return False
        try:
            self._model = DocImgOrientationClassification()
        except Exception as exc:  # noqa: BLE001 - 显存 / 权重下载都可能在这里炸
            self._error = f"{type(exc).__name__}: {exc}"
            warn_once("orient-model", f"[prepass] 方向模型不可用：{self._error}，跳过转正")
            return False
        return True

    def available(self) -> bool:
        return self._ensure()

    @property
    def error(self) -> str:
        return self._error

    # -- 判角度 ------------------------------------------------------------ #

    def angle_of(self, png: bytes) -> int:
        """这张页面图当前旋转了多少度（0 / 90 / 180 / 270）；判不出来返回 -1。

        注意必须传**文件路径**给 predict：它的参数是可迭代的，直接给 bytes 会被当成
        "一批输入"逐字节跑（真机上踩过，输出直接爆掉）。
        """

        if not self._ensure():
            return -1
        try:
            if not self._tmp:
                fd, self._tmp = tempfile.mkstemp(suffix=".png")
                os.close(fd)
            with open(self._tmp, "wb") as fh:
                fh.write(png)
            results = self._model.predict(self._tmp)
        except Exception:  # noqa: BLE001
            return -1
        if not isinstance(results, (list, tuple)):
            try:
                results = list(results)
            except TypeError:
                return -1
        if not results:
            return -1
        data = getattr(results[0], "json", None) or {}
        names = (data.get("res") or {}).get("label_names") or []
        try:
            angle = int(str(names[0]))
        except (IndexError, ValueError, TypeError):
            return -1
        return angle if angle in ANGLES else -1

    def close(self) -> None:
        if self._tmp:
            cleanup_temp(self._tmp)
            self._tmp = ""


class OrientationPrepass:
    """把 ``apply()`` 挂在 OCR 之前：歪页转正，顺手修掉顶部的跨页残条。"""

    def __init__(self, classifier: Optional[PageOrientationClassifier] = None,
                 probe_dpi: Optional[float] = None,
                 upright_dpi: Optional[float] = None) -> None:
        self._classifier = classifier or PageOrientationClassifier()
        self.probe_dpi = float(probe_dpi or env_number("PROBE_DPI", PROBE_DPI))
        self.upright_dpi = float(upright_dpi or env_number("UPRIGHT_DPI", UPRIGHT_DPI))

    def enabled(self) -> bool:
        """``TABLE_PREPASS_ORIENT=0`` 关掉整层（做 A/B 对照用）。"""

        return env_flag("ORIENT", True)

    # -- 主入口 ------------------------------------------------------------ #

    def apply(self, pdf_path: str, *, dpi: int = 200, out_path: Optional[str] = None,
              fragment_mode: str = "rotated") -> OrientationReport:
        """转正 ``pdf_path`` 里的歪页 → 返回报告（``report.path`` 是要送进 OCR 的文件）。

        ``fragment_mode``：

        * ``"rotated"``（默认）：只对**被转正重建**的页做残条修复。真机端到端验证过的
          就是这条路径。
        * ``"all"``：每一页都查一遍残条。页不需要转正、但同样出现"顶端残条导致行界
          错切"时用它（检测在低分辨率渲染图上做，擦除直接盖在原页面上）。
        * ``"off"``：只转正，不修残条。

        没有歪页（且 ``fragment_mode != "all"``）时**原样返回**，一点开销都不多花。

        ``dpi`` 只是留一个和调用方一致的参数位（也是 ``apply`` 的调用方最容易顺手传的
        那个值）；**重建页实际用的是 ``upright_dpi``**，理由见上面的对照说明。
        """

        report = OrientationReport(source=pdf_path, path=pdf_path,
                                   fragment_mode=fragment_mode)
        started = time.time()
        if not self.enabled():
            report.notes.append("TABLE_PREPASS_ORIENT=0，未做方向转正")
            report.elapsed = time.time() - started
            return report

        fitz = import_pymupdf()
        if fitz is None:
            report.notes.append("缺少 PyMuPDF，未做方向转正")
            report.elapsed = time.time() - started
            return report
        if not self._classifier.available():
            report.notes.append(f"方向模型不可用（{self._classifier.error or '未知原因'}），未做方向转正")
            report.elapsed = time.time() - started
            return report

        mode = (fragment_mode or "rotated").lower()
        src = fitz.open(pdf_path)
        try:
            probe_zoom = self.probe_dpi / 72.0
            probe = fitz.Matrix(probe_zoom, probe_zoom)
            bad: Dict[int, int] = {}
            angles: List[int] = []
            for index, page in enumerate(src):
                angle = self._classifier.angle_of(
                    page.get_pixmap(matrix=probe, alpha=False).tobytes("png"))
                angles.append(angle)
                if angle > 0:
                    bad[index] = angle
            report.pages = len(angles)
            report.angles = angles
            probe_spent = time.time() - started
            # 逐页把判定结果打出来，判断得准不准，看这一行就知道：
            # 位置就是页码（从第 1 页起），? 表示这一页没判出来。
            print("[prepass] 逐页角度（按页序，第 1 页起）: "
                  + " ".join("?" if a < 0 else str(a) for a in angles))

            if not bad:
                print(f"[prepass] 逐页判完 {len(angles)} 页，都是正的（耗时 {probe_spent:.1f}s）")
            else:
                report.rotated = {i + 1: a for i, a in bad.items()}
                print(f"[prepass] 逐页判完 {len(angles)} 页（耗时 {probe_spent:.1f}s）："
                      f"第 {list(report.rotated)} 页旋转 {list(report.rotated.values())}°，转正后重建")

            if not bad and mode != "all":
                report.elapsed = time.time() - started
                return report

            out = fitz.open()
            try:
                zoom = self.upright_dpi / 72.0
                for index, page in enumerate(src):
                    strip = None
                    if index in bad:
                        pix = self._upright(page, bad[index], zoom)
                        # 页面尺寸用真实物理尺寸（像素 ÷ 渲染倍率），见模块开头。
                        w_pt, h_pt = pix.width / zoom, pix.height / zoom
                        new_page = out.new_page(width=w_pt, height=h_pt)
                        new_page.insert_image(fitz.Rect(0, 0, w_pt, h_pt),
                                              stream=pix.tobytes("png"))
                        if mode != "off":
                            strip = find_fragment_strip(pix, zoom=zoom)
                    else:
                        out.insert_pdf(src, from_page=index, to_page=index)
                        if mode == "all":
                            new_page = out[out.page_count - 1]
                            strip = find_fragment_strip(
                                page.get_pixmap(matrix=probe, alpha=False),
                                zoom=probe_zoom)
                    if strip is None:
                        continue
                    report.strips[index + 1] = strip
                    if erase_fragment_strip(new_page, strip):
                        print(f"[prepass] 第 {index + 1} 页顶部有跨页残条，"
                              f"已擦除其上边框线 y={strip[1]:.1f}~{strip[3]:.1f}pt")

                if not report.rotated and not report.strips:
                    report.elapsed = time.time() - started
                    return report                      # 什么都没改，别多写一个文件
                if out_path:
                    dst = out_path
                else:
                    fd, dst = tempfile.mkstemp(suffix=".pdf")
                    os.close(fd)
                    report.temporary = True
                out.save(dst)
            finally:
                out.close()
            report.path = dst
            report.elapsed = time.time() - started
            return report
        finally:
            src.close()

    # -- 单页转正 ---------------------------------------------------------- #

    def _upright(self, page: Any, angle: int, zoom: float) -> Any:
        """按角度转正，并自检：转完再判一次，确认模型说的是 0。"""

        fitz = import_pymupdf()
        for correction in ((360 - angle) % 360, angle):
            if not correction:
                continue
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom).prerotate(correction), alpha=False)
            if self._classifier.angle_of(pix.tobytes("png")) == 0:
                return pix
        print(f"[prepass] 第 {page.number + 1} 页两个方向都试过仍判不出 0，保留原样")
        return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)


_DEFAULT = OrientationPrepass()


def prepare_pdf(pdf_path: str, *, dpi: int = 200, out_path: Optional[str] = None,
                fragment_mode: str = "rotated",
                prepass: Optional[OrientationPrepass] = None) -> OrientationReport:
    """模块级快捷方式：用进程内共享的预处理器跑一次。"""

    return (prepass or _DEFAULT).apply(pdf_path, dpi=dpi, out_path=out_path,
                                       fragment_mode=fragment_mode)
