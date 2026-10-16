"""真 API 的集成测试：用**真的 PyMuPDF** 造 PDF、真的渲染像素、真的画矩形。

上面几套测试用假模块换掉了 ``sys.modules``，能测"逻辑对不对"；这一套测"真 API 调得对
不对"——比如重建页的单位是点不是像素、``draw_rect`` 的坐标原点在左上角、PNG 渲染倍率
换算。**没装 PyMuPDF 时整份跳过**（方向模型仍然用假的，本机不需要 GPU）。
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

try:                 # 新版包名
    import pymupdf as fitz
except ImportError:  # pragma: no cover - 取决于环境
    try:
        import fitz
    except ImportError:
        fitz = None  # type: ignore[assignment]

from ocr_table_prepass import OrientationPrepass, PageOrientationClassifier
from ocr_table_prepass import fragment

A4_W, A4_H = 595.32, 841.92
DPI = 110.0
# 两条横线的间距（磅）：110 DPI 下约 13 像素，正是"跨页残条"那种薄带子。
FRAGMENT_GAP_PT = 8.5


def _png_size(path: str) -> tuple:
    """读 PNG 的 IHDR 拿宽高——不用任何图像库。"""

    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


class FakeOrientationModel:
    """假的方向模型：看到横的图就说"转了 90°"。

    判据来自**真实渲染出来的像素**（PNG 宽 > 高），所以这一套测试里 prerotate 真的被执行了。
    """

    def predict(self, path):
        width, height = _png_size(path)
        label = 90 if width > height else 0

        class _Result:
            json = {"res": {"label_names": [label], "scores": [0.99]}}

        return [_Result()]


def build_pdf(path: str, *, pages: int = 3, landscape: tuple = (),
              fragment_pages: tuple = ()) -> None:
    """造一份测试 PDF：正常页 + 可选的横躺页 + 可选的"顶端两根横线"。"""

    doc = fitz.open()
    try:
        for index in range(pages):
            if index in landscape:
                page = doc.new_page(width=A4_H, height=A4_W)
            else:
                page = doc.new_page(width=A4_W, height=A4_H)
            page.insert_text((72, 160), f"page {index + 1} body text", fontsize=11)
            if index in fragment_pages:
                # 带内刻意不放字：这才是"残条"，有字就是正常数据行了。
                y = 40.0
                for offset in (0.0, FRAGMENT_GAP_PT):
                    page.draw_line(fitz.Point(60, y + offset),
                                   fitz.Point(page.rect.width - 60, y + offset),
                                   width=1)
        doc.save(path)
    finally:
        doc.close()


def dark_of(page) -> "object":
    """把一页渲染成灰度图并二值化，用来看"还有几根横线"。"""

    zoom = DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return fragment.as_gray(pix) < 150


@unittest.skipIf(fitz is None, "需要真的 PyMuPDF 才能跑这一套")
class TestRealPdfIntegration(unittest.TestCase):
    def _prepass(self) -> OrientationPrepass:
        return OrientationPrepass(
            classifier=PageOrientationClassifier(model=FakeOrientationModel()))

    def test_rotated_page_is_rebuilt_at_true_physical_size(self) -> None:
        """重建页必须是真实物理尺寸（点），不是像素——否则 A4 会变成 A2。"""

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "scan.pdf"
            out = Path(tmp) / "fixed.pdf"
            build_pdf(str(src), landscape=(1,))

            report = self._prepass().apply(str(src), out_path=str(out))
            self.assertEqual(report.angles, [0, 90, 0])
            self.assertEqual(sorted(report.rotated), [2])
            self.assertTrue(report.changed)

            fixed = fitz.open(str(out))
            try:
                self.assertEqual(fixed.page_count, 3, "页数必须保持不变")
                # 横躺页转正后是 A4 竖版（不是 A2：把像素当点会得到约 909×1286）
                self.assertAlmostEqual(fixed[1].rect.width, A4_W, delta=1.0)
                self.assertAlmostEqual(fixed[1].rect.height, A4_H, delta=1.0)
                # 没被判歪的页原样拷过去，尺寸不变
                self.assertAlmostEqual(fixed[0].rect.width, A4_W, delta=0.5)
                self.assertAlmostEqual(fixed[0].rect.height, A4_H, delta=0.5)
                # 没被动过的页保留原有文字层；被转正的页是位图重建，没有文字层（预期行为）
                self.assertIn("page 1 body text", fixed[0].get_text())
                self.assertIn("page 3 body text", fixed[2].get_text())
                self.assertEqual(fixed[1].get_text().strip(), "")
                self.assertTrue(fixed[1].get_images(), "转正页应当是一张重建出来的位图")
            finally:
                fixed.close()

    def test_fragment_band_is_erased_on_a_real_page(self) -> None:
        """顶端的残条带：真渲染 → 真检测 → 真画白矩形，擦完少一根横线。"""

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "scan.pdf"
            out = Path(tmp) / "fixed.pdf"
            build_pdf(str(src), fragment_pages=(0,))

            before = fitz.open(str(src))
            try:
                lines_before = fragment.find_horizontal_lines(dark_of(before[0]))
            finally:
                before.close()
            self.assertGreaterEqual(len(lines_before), 2, "测试用的 PDF 本身就该有两根横线")

            report = self._prepass().apply(str(src), out_path=str(out),
                                           fragment_mode="all")
            self.assertEqual(sorted(report.strips), [1])
            self.assertEqual(report.rotated, {}, "这份文档没有歪页")

            after = fitz.open(str(out))
            try:
                lines_after = fragment.find_horizontal_lines(dark_of(after[0]))
            finally:
                after.close()
            self.assertEqual(len(lines_after), len(lines_before) - 1,
                             "残条带上方的边框线应当被白色矩形盖掉")

    def test_a_quiet_document_is_left_completely_alone(self) -> None:
        """没有歪页、没有残条：不重写文件，也不多花开销。"""

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "scan.pdf"
            build_pdf(str(src))
            digest = src.read_bytes()

            report = self._prepass().apply(str(src))
            self.assertFalse(report.changed)
            self.assertEqual(report.path, str(src))
            self.assertEqual(report.angles, [0, 0, 0])
            self.assertEqual(src.read_bytes(), digest, "原文件不该被动过")

    def test_default_mode_only_touches_rotated_pages(self) -> None:
        """默认 ``rotated``：同一份文档里，横躺页修了，正常页连碰都不碰。"""

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "scan.pdf"
            out = Path(tmp) / "fixed.pdf"
            build_pdf(str(src), landscape=(1,), fragment_pages=(0, 2))

            report = self._prepass().apply(str(src), out_path=str(out))
            self.assertEqual(sorted(report.rotated), [2])
            # 第 1、3 页没被转正，所以即使顶端有残条也不动手（默认范围就是"只修转正页"）
            self.assertEqual(report.strips, {})


if __name__ == "__main__":
    unittest.main()
