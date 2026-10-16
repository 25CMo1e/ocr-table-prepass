"""送进 OCR 流水线之前先把横躺的页转正（``ocr_table_prepass.orient``）。

实测病灶：PP-StructureV3 内部的方向模块会工作（180° 倒置的正文页能逐字读对），但它
纠正后的坐标系没有传给表格结构那一步——横躺页上的表格表头会整列错位。只要"原始朝向"
是正的，表头就是对的。
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ocr_table_prepass import orient
from tests.fakes import (FakeDoc, FakeNewPage, fake_orientation_classifier,
                         fake_pymupdf, gray_with_fragment)


def _paddleocr_module() -> types.ModuleType:
    module = types.ModuleType("paddleocr")
    module.DocImgOrientationClassification = fake_orientation_classifier()
    return module


class TestOrientationPrepass(unittest.TestCase):
    def _apply(self, angles, env=None, gray=None, fragment_mode="rotated", **kwargs):
        built = FakeDoc(angles, gray)
        modules = {"pymupdf": fake_pymupdf(angles, built), "paddleocr": _paddleocr_module()}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, env or {}, clear=False):
            report = orient.OrientationPrepass(**kwargs).apply(
                "C:/fake.pdf", fragment_mode=fragment_mode)
        if report.changed and report.temporary:
            self.addCleanup(report.cleanup)
        return report, built

    def test_upright_document_is_left_alone(self) -> None:
        """没有歪页时原样返回，一点开销都不多花，也不多写一个文件。"""

        report, _ = self._apply((0, 0, 0))
        self.assertEqual(report.path, "C:/fake.pdf")
        self.assertEqual(report.rotated, {})
        self.assertEqual(report.angles, [0, 0, 0])
        self.assertFalse(report.changed)
        self.assertFalse(report.temporary)

    def test_rotated_page_triggers_a_rebuild(self) -> None:
        report, _ = self._apply((0, 90, 0))
        self.assertEqual(sorted(report.rotated), [2], "应报出第 2 页（1 起算）")
        self.assertEqual(report.rotated[2], 90)
        self.assertTrue(report.changed)
        self.assertTrue(Path(report.path).exists())

    def test_180_and_270_are_handled(self) -> None:
        report, _ = self._apply((180, 270, 0))
        self.assertEqual(report.rotated, {1: 180, 2: 270})

    def test_unknown_page_is_reported_not_fixed(self) -> None:
        """判不出来（-1）的页不动它，只在报告里留一个 ``-1``。"""

        module = types.ModuleType("paddleocr")

        class _Broken:
            def predict(self, path):
                raise RuntimeError("boom")

        module.DocImgOrientationClassification = _Broken
        with mock.patch.dict(sys.modules, {"pymupdf": fake_pymupdf((0, 90)),
                                           "paddleocr": module}):
            report = orient.OrientationPrepass().apply("C:/fake.pdf")
        self.assertEqual(report.angles, [-1, -1])
        self.assertEqual(report.rotated, {})
        self.assertFalse(report.changed)

    def test_rebuilt_page_keeps_true_physical_size(self) -> None:
        """重建页按**真实物理尺寸**建，配合降低的渲染 DPI 让流水线放大约 1.3 倍。

        实测（同一张会判错的横躺表格，多个对照）：流水线放大约 1.2~1.44 倍时判对，
        1:1 或缩小则判错。所以渲染 DPI 用 UPRIGHT_DPI 而不是调用方给的 dpi，
        页面尺寸仍取真实物理尺寸（像素 ÷ 渲染倍率）。
        """

        report, doc = self._apply((0, 90))
        self.assertTrue(report.changed)
        width, height = doc.page_size
        zoom = orient.UPRIGHT_DPI / 72.0              # 假 pixmap 固定 595x842
        self.assertAlmostEqual(width, 595 / zoom, places=3)
        self.assertAlmostEqual(height, 842 / zoom, places=3)
        # 流水线按固定 144 DPI 渲染：放大倍数 = 144 / UPRIGHT_DPI，应落在 1.2~1.44
        raster = (595 / zoom) / 72 * 144
        self.assertAlmostEqual(raster / 595, 144 / orient.UPRIGHT_DPI, places=3)
        self.assertGreaterEqual(raster / 595, 1.2)
        self.assertLessEqual(raster / 595, 1.44)

    def test_upright_dpi_is_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_UPRIGHT_DPI": "144"},
                             clear=False):
            prepass = orient.OrientationPrepass()
        self.assertAlmostEqual(prepass.upright_dpi, 144.0)

    def test_can_be_switched_off(self) -> None:
        report, _ = self._apply((0, 90), env={"TABLE_PREPASS_ORIENT": "0"})
        self.assertEqual(report.path, "C:/fake.pdf")
        self.assertEqual(report.rotated, {})
        self.assertTrue(report.notes, "关掉时应在报告里说明原因")

    def test_legacy_env_prefix_can_switch_it_off_too(self) -> None:
        report, _ = self._apply((0, 90), env={"COMPS_OCR_ORIENT": "0"})
        self.assertFalse(report.changed)

    def test_missing_model_is_harmless(self) -> None:
        """方向模型不可用时只是"这一层不生效"，不能影响 OCR 本身。"""

        with mock.patch.dict(sys.modules, {"pymupdf": fake_pymupdf((0, 90)),
                                           "paddleocr": None}):
            report = orient.OrientationPrepass().apply("C:/fake.pdf")
        self.assertEqual(report.path, "C:/fake.pdf")
        self.assertFalse(report.changed)
        self.assertTrue(any("方向模型不可用" in note for note in report.notes))

    def test_missing_pymupdf_is_harmless(self) -> None:
        with mock.patch.dict(sys.modules, {"pymupdf": None, "fitz": None}):
            report = orient.OrientationPrepass().apply("C:/fake.pdf")
        self.assertFalse(report.changed)
        self.assertTrue(any("PyMuPDF" in note for note in report.notes))

    def test_predict_receives_a_path_not_bytes(self) -> None:
        """predict 的入参是可迭代对象，喂 bytes 会被当成"一批输入"逐字节跑。"""

        seen: list = []

        class _Classifier:
            def predict(self, value):
                seen.append(value)
                raise AssertionError("这一层只检查传进去的是什么")

        classifier = orient.PageOrientationClassifier(model=_Classifier())
        self.assertEqual(classifier.angle_of(b"\x89PNG\r\n\x1a\n"),
                         -1, "判不出来时要返回 -1，而不是抛出去")
        self.assertTrue(seen and isinstance(seen[0], str), f"应传路径，实际 {seen!r}")
        self.assertTrue(Path(seen[0]).exists(), "传过去的路径必须真的写好了图")
        classifier.close()

    def test_report_serialises(self) -> None:
        report, _ = self._apply((0, 90))
        payload = report.as_dict()
        self.assertEqual(payload["rotated_pages"], {"2": 90})
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["pages"], 2)

    def test_page_that_cannot_be_corrected_keeps_its_original_render(self) -> None:
        """两个方向都试过仍判不出 0：保留原样那一版，别反复折腾。"""

        class _Stubborn:
            """永远说"这一页歪了 90°"——于是自检永远过不去。"""

            def predict(self, path):
                class _Result:
                    json = {"res": {"label_names": [90], "scores": [0.9]}}

                return [_Result()]

        classifier = orient.PageOrientationClassifier(model=_Stubborn())
        built = FakeDoc((0,))
        modules = {"pymupdf": fake_pymupdf((0,), built), "paddleocr": _paddleocr_module()}
        with mock.patch.dict(sys.modules, modules):
            report = orient.OrientationPrepass(classifier=classifier).apply("C:/fake.pdf")
        self.addCleanup(report.cleanup)
        self.assertEqual(sorted(report.rotated), [1])
        # 兜底那一版是"原样渲染"：没有 prerotate，所以 tag 还是 angle:0
        self.assertEqual(built.pages_out[0].images[0][1], b"angle:0")


class TestFragmentOnRebuiltPages(unittest.TestCase):
    """残条修复搭在"转正重建"这条路径上：转正页是重新栅格化的，正好顺手修。"""

    def _apply(self, angles, gray, **kwargs):
        built = FakeDoc(angles, gray)
        modules = {"pymupdf": fake_pymupdf(angles, built), "paddleocr": _paddleocr_module()}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, {}, clear=False):
            report = orient.OrientationPrepass().apply("C:/fake.pdf", **kwargs)
        if report.changed and report.temporary:
            self.addCleanup(report.cleanup)
        return report, built

    def test_strip_erased_on_the_rotated_page(self) -> None:
        report, doc = self._apply((0, 90), gray_with_fragment())
        self.assertEqual(sorted(report.rotated), [2])
        self.assertEqual(sorted(report.strips), [2], "只在被转正的那一页上修残条")
        rect = report.strips[2]
        self.assertLess(rect[1], rect[3])
        drawn = doc.pages_out[1].rects
        self.assertEqual(len(drawn), 1, "擦除矩形应真的画到了重建页上")
        self.assertEqual(drawn[0][1].get("fill"), (1, 1, 1))
        self.assertEqual(report.as_dict()["fragment_strips"]["2"][1], round(rect[1], 1))

    def test_no_fragment_geometry_means_no_erase(self) -> None:
        """正常表头（换算成磅仍相距 26 磅，实测第 33 页就是 26.8 磅）不该被误擦。"""

        gray = gray_with_fragment(top=20, bottom=60)      # 40px ÷ 1.53 ≈ 26pt
        report, doc = self._apply((0, 90), gray)
        self.assertEqual(report.rotated, {2: 90})
        self.assertEqual(report.strips, {})
        self.assertEqual(doc.pages_out[1].rects, [])

    def test_mode_off_skips_the_strip(self) -> None:
        report, doc = self._apply((0, 90), gray_with_fragment(), fragment_mode="off")
        self.assertEqual(report.strips, {})
        self.assertEqual(doc.pages_out[1].rects, [])

    def test_mode_all_also_fixes_upright_pages(self) -> None:
        """文档整体是正的、但顶端同样有残条时，用 ``mode='all'`` 也能修。"""

        report, doc = self._apply((0, 0), gray_with_fragment(), fragment_mode="all")
        self.assertEqual(report.rotated, {})
        self.assertEqual(sorted(report.strips), [1, 2])
        self.assertTrue(report.changed, "只擦残条也应该产出新文件")
        self.assertEqual(len(doc.pages_out[0].rects), 1)

    def test_mode_all_on_a_clean_document_changes_nothing(self) -> None:
        report, _ = self._apply((0, 0), None, fragment_mode="all")
        self.assertFalse(report.changed)
        self.assertEqual(report.path, "C:/fake.pdf")

    def test_erase_failure_does_not_break_the_rebuild(self) -> None:
        """擦除本身出错也只是"这一层不生效"：重建照常完成，绝不能把 OCR 拖下水。"""

        class _ExplodingPage(FakeNewPage):
            def draw_rect(self, *args, **kwargs):
                raise RuntimeError("boom")

        class _Doc(FakeDoc):
            def new_page(self, width=0, height=0):
                self.page_size = (width, height)
                page = _ExplodingPage()
                self.pages_out.append(page)
                return page

        built = _Doc((0, 90), gray_with_fragment())
        modules = {"pymupdf": fake_pymupdf((0, 90), built), "paddleocr": _paddleocr_module()}
        with mock.patch.dict(sys.modules, modules):
            report = orient.OrientationPrepass().apply("C:/fake.pdf")
        self.addCleanup(report.cleanup)
        self.assertEqual(sorted(report.rotated), [2])
        self.assertTrue(report.changed)
        self.assertTrue(Path(report.path).exists())
        self.assertEqual(sorted(report.strips), [2], "擦除失败也要如实记进报告")

    def test_out_path_is_honoured_and_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "fixed.pdf")
            built = FakeDoc((0, 90))
            modules = {"pymupdf": fake_pymupdf((0, 90), built),
                       "paddleocr": _paddleocr_module()}
            with mock.patch.dict(sys.modules, modules):
                report = orient.OrientationPrepass().apply("C:/fake.pdf", out_path=target)
            self.assertEqual(report.path, target)
            self.assertFalse(report.temporary)
            report.cleanup()
            self.assertTrue(Path(target).exists(), "显式指定的输出文件不该被当成临时文件删掉")


if __name__ == "__main__":
    unittest.main()
