"""跨页残条的检测与擦除（``ocr_table_prepass.fragment``，不碰 GPU、不需要 PDF）。"""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import numpy as np

from ocr_table_prepass import fragment


def pix(width: int, height: int, lines, text_pixels=None) -> types.SimpleNamespace:
    """构造一个假 pixmap：白底 + 指定横线 + 稀疏文字像素。"""

    gray = np.full((height, width), 255, dtype=np.uint8)
    for y0, y1, x0, x1 in lines:
        gray[y0:y1 + 1, x0:x1 + 1] = 40
    for y, xs in (text_pixels or []):
        for x in xs:
            gray[y, x] = 40
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return types.SimpleNamespace(width=width, height=height, n=3,
                                 stride=width * 3, samples=rgb.tobytes())


class TestFragmentStrip(unittest.TestCase):
    """实测病灶：跨页表格在本页顶端只剩薄薄一条续行（"动）。"这类行尾）时，结构模型会把
    下面几行的行界切到行内部，两家公司的内容互相串行。擦掉残条带上方的表格边框线即可
    恢复（110/120/144 三档 + 整份 43 页文档端到端都验证过）。
    """

    def test_fragment_band_detected(self) -> None:
        """两线相距 13px（≤20pt）、带内只有零星文字 → 命中，返回擦除矩形。"""

        image = pix(400, 100, [(20, 20, 10, 390), (33, 33, 10, 390)],
                    text_pixels=[(26, [200, 204, 208, 212])])
        strip = fragment.find_fragment_strip(image, zoom=1.0)
        self.assertIsNotNone(strip)
        x0, y0, x1, y1 = strip
        self.assertLessEqual(x0, 10)
        self.assertGreaterEqual(x1, 390)
        self.assertAlmostEqual(y0, 18.8, places=1)
        self.assertAlmostEqual(y1, 21.2, places=1)

    def test_normal_header_row_rejected(self) -> None:
        """两线相距 27px（>20pt）：正常首行（实测 26.8 磅的表头），不动。"""

        image = pix(400, 100, [(20, 20, 10, 390), (47, 47, 10, 390)])
        self.assertIsNone(fragment.find_fragment_strip(image, zoom=1.0))

    def test_dense_band_rejected(self) -> None:
        """间距够近但带内布满文字（实测 9.2% 有字的数据行）：不动。"""

        text = [(y, list(range(20, 380, 7))) for y in range(23, 31)]
        image = pix(400, 100, [(20, 20, 10, 390), (33, 33, 10, 390)],
                    text_pixels=text)
        self.assertIsNone(fragment.find_fragment_strip(image, zoom=1.0))

    def test_missing_lines_rejected(self) -> None:
        one = pix(400, 100, [(20, 20, 10, 390)])
        blank = pix(400, 100, [])
        self.assertIsNone(fragment.find_fragment_strip(one, zoom=1.0))
        self.assertIsNone(fragment.find_fragment_strip(blank, zoom=1.0))

    def test_gap_limit_env(self) -> None:
        """阈值可调：把上限收到 10 磅，13 磅的残条就不再命中。"""

        image = pix(400, 100, [(20, 20, 10, 390), (33, 33, 10, 390)])
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_FRAGMENT_GAP_PT": "10"},
                             clear=False):
            self.assertIsNone(fragment.find_fragment_strip(image, zoom=1.0))

    def test_switch_off(self) -> None:
        image = pix(400, 100, [(20, 20, 10, 390), (33, 33, 10, 390)])
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_FRAGMENT_FIX": "0"},
                             clear=False):
            self.assertIsNone(fragment.find_fragment_strip(image, zoom=1.0))

    def test_legacy_env_prefix_still_works(self) -> None:
        """抽库之前用的前缀是 ``COMPS_OCR_*``：老部署不该因为改名就失效。"""

        image = pix(400, 100, [(20, 20, 10, 390), (33, 33, 10, 390)])
        with mock.patch.dict(os.environ, {"COMPS_OCR_FRAGMENT_FIX": "0"},
                             clear=False):
            self.assertIsNone(fragment.find_fragment_strip(image, zoom=1.0))

    def test_zoom_scales_back_to_points(self) -> None:
        """同样的几何在 2 倍分辨率下要换算出同样的磅值：13px @2x 仍是 6.5pt。"""

        image = pix(800, 200, [(40, 40, 20, 780), (66, 66, 20, 780)])
        strip = fragment.find_fragment_strip(image, zoom=2.0)
        self.assertIsNotNone(strip)
        _, y0, _, y1 = strip
        self.assertAlmostEqual(y0, 18.8, places=1)
        self.assertAlmostEqual(y1, 21.2, places=1)

    def test_foreign_objects_never_raise(self) -> None:
        """认不出的对象一律安全返回 None——绝不能让修正层把主流程拖下水。"""

        for weird in (object(), types.SimpleNamespace(), 42, None, np.zeros(3)):
            self.assertIsNone(fragment.find_fragment_strip(weird, zoom=1.0))

    def test_line_scanner_tolerates_small_gaps(self) -> None:
        """扫描件的线条常有小缺口：2px 断口算同一条线。"""

        dark = np.zeros((40, 400), dtype=bool)
        dark[20, 10:200] = True
        dark[20, 202:390] = True            # 2px 断口
        lines = fragment.find_horizontal_lines(dark)
        self.assertEqual(len(lines), 1)
        self.assertGreaterEqual(lines[0][3], 389)


class TestAsGray(unittest.TestCase):
    """页面图 → 灰度数组：各家渲染库给的形态不一样，都得吃得下。"""

    def test_pymupdf_pixmap_like(self) -> None:
        gray = fragment.as_gray(pix(10, 4, [(1, 1, 0, 9)]))
        self.assertEqual(gray.shape, (4, 10))
        self.assertLess(gray[1, 5], 150)
        self.assertGreater(gray[0, 5], 200)

    def test_plain_2d_array(self) -> None:
        array = np.full((5, 7), 255, dtype=np.uint8)
        self.assertEqual(fragment.as_gray(array).shape, (5, 7))

    def test_rgb_array(self) -> None:
        array = np.zeros((5, 7, 3), dtype=np.uint8)
        array[..., 0] = 30
        self.assertAlmostEqual(float(fragment.as_gray(array)[0, 0]), 10.0, places=1)

    def test_normalised_float_array(self) -> None:
        """0~1 的浮点图（有些渲染管线这么给）要还原到 0~255，否则"整页全黑"。"""

        array = np.ones((4, 4), dtype="float32")
        self.assertGreater(float(fragment.as_gray(array).max()), 200)

    def test_pil_like_image(self) -> None:
        """只要有 ``convert('L')`` 就当图像处理（PIL / Pillow 形态）。"""

        class FakeImage:
            def __init__(self) -> None:
                self.mode = ""

            def convert(self, mode):
                self.mode = mode
                return np.full((3, 3), 200, dtype=np.uint8)

        gray = fragment.as_gray(FakeImage())
        self.assertEqual(gray.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
