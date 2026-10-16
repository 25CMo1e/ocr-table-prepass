"""引擎适配层：页数上限、页序、子模块开关，以及"前置修正接进流水线"这条路。

不装 paddleocr、不碰 GPU——用假的 paddleocr / PyMuPDF 换掉 ``sys.modules``。
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ocr_table_prepass import engine as engines
from ocr_table_prepass import paddlex_fix
from tests.fakes import (FakeDoc, FakePipeline, fake_limit_fitz, fake_orientation_classifier,
                         fake_paddleocr, fake_pymupdf, gray_with_fragment)

np = __import__("numpy")


class TestPaddlexMultiTableFix(unittest.TestCase):
    """上游多表格单元格错配的运行时补丁（PaddleX PR #4756，未被合并）。

    现象：多张表格的裁剪被放进同一批次，NMS 打乱顺序后，原实现按「扁平顺序 +
    box_nums 切片」分组，于是单元格被分给错误的表格/列。单表不触发。
    """

    def test_reordered_batch_is_regrouped_by_batch_index(self) -> None:
        # 两张表，分别 3 / 2 个框；但返回顺序被打乱成「表2 的 2 个框在前」
        boxes = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
        box_nums = np.array([3, 2])
        batch_inds = np.array([1, 1, 0, 0, 0])
        got = paddlex_fix._group_by_batch_index([boxes, box_nums, batch_inds])  # noqa: SLF001
        self.assertEqual([len(g["boxes"]) for g in got], [3, 2])
        np.testing.assert_array_equal(got[0]["boxes"], boxes[[2, 3, 4]])   # 表1 的框
        np.testing.assert_array_equal(got[1]["boxes"], boxes[[0, 1]])      # 表2 的框

    def test_in_order_batch_still_regroups_identically(self) -> None:
        boxes = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
        got = paddlex_fix._group_by_batch_index(                           # noqa: SLF001
            [boxes, np.array([3, 2]), np.array([0, 0, 0, 1, 1])])
        np.testing.assert_array_equal(got[0]["boxes"], boxes[:3])
        np.testing.assert_array_equal(got[1]["boxes"], boxes[3:])

    def test_without_batch_inds_falls_back(self) -> None:
        """拿不到 batch_inds 就返回 None，交回原实现——那种情况下原实现本来就对。"""

        boxes = np.zeros((5, 4), dtype=np.float32)
        self.assertIsNone(                                               # noqa: SLF001
            paddlex_fix._group_by_batch_index([boxes, np.array([3, 2])]))
        # 三维浮点数组是分割 mask，不能被误当成 batch_inds
        masks = np.zeros((5, 10, 10), dtype=np.float32)
        self.assertIsNone(                                               # noqa: SLF001
            paddlex_fix._group_by_batch_index([boxes, np.array([3, 2]), masks]))

    def test_switch_and_missing_paddlex_are_harmless(self) -> None:
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_PADDLEX_FIX": "0"}, clear=False):
            self.assertFalse(paddlex_fix.enabled())
            self.assertIn("未打补丁", paddlex_fix.apply())
        # 本机没装 paddlex：apply 只能返回说明，绝不能抛异常
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_PADDLEX_FIX": "1"}, clear=False):
            self.assertTrue(paddlex_fix.enabled())
            self.assertIsInstance(paddlex_fix.apply(), str)

    def test_legacy_env_prefix_still_works(self) -> None:
        with mock.patch.dict(os.environ, {"COMPS_OCR_PADDLEX_FIX": "0"}, clear=False):
            self.assertFalse(paddlex_fix.enabled())

    def test_apply_is_idempotent_and_reversible(self) -> None:
        class _Pred:
            def _format_output(self, pred):
                return "original"

        pkg = types.ModuleType("paddlex")
        pkg.inference = types.ModuleType("paddlex.inference")
        pkg.inference.models = types.ModuleType("paddlex.inference.models")
        pkg.inference.models.object_detection = types.ModuleType(
            "paddlex.inference.models.object_detection")
        pkg.inference.models.object_detection.predictor = types.ModuleType(
            "paddlex.inference.models.object_detection.predictor")
        pkg.inference.models.object_detection.predictor.DetRunnerPredictor = _Pred
        with mock.patch.dict(sys.modules, {
                "paddlex": pkg,
                "paddlex.inference": pkg.inference,
                "paddlex.inference.models": pkg.inference.models,
                "paddlex.inference.models.object_detection":
                    pkg.inference.models.object_detection,
                "paddlex.inference.models.object_detection.predictor":
                    pkg.inference.models.object_detection.predictor}), \
                mock.patch.dict(os.environ, {"TABLE_PREPASS_PADDLEX_FIX": "1"},
                                clear=False):
            before = _Pred._format_output
            self.assertIn("已打补丁", paddlex_fix.apply())
            self.assertIsNot(_Pred._format_output, before, "方法应被替换")
            self.assertTrue(getattr(_Pred._format_output, "_prepass_patched", False))
            self.assertIn("已打过补丁", paddlex_fix.apply())        # 幂等
            self.assertIn("已撤回", paddlex_fix.revert())
            self.assertIs(_Pred._format_output, before, "方法应被还原")
            self.assertEqual(_Pred()._format_output(None), "original")


class TestDisabledModulesParsing(unittest.TestCase):
    def test_splits_and_drops_unknown_names(self) -> None:
        raw = " formula , chart ，seal, 没这个 "
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": raw}):
            self.assertEqual(engines.disabled_modules(), ["chart", "formula", "seal"])

    def test_empty_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": ""}):
            self.assertEqual(engines.disabled_modules(), [])


class TestPageLimit(unittest.TestCase):
    """页数上限必须真的执行。

    真实漏洞：引擎收了 max_pages 却没用，而 PaddleOCR 的 predict 会处理整份 PDF，
    于是 100 页的文件会一路跑到底——既超出调用方的时间预算，也白占显存。
    """

    def _src(self, name: str = "doc.pdf") -> str:
        path = Path(tempfile.mkdtemp(prefix="pagelimit-")) / name
        path.write_bytes(b"%PDF-1.4 original")
        return str(path)

    def test_slices_when_over_the_cap(self) -> None:
        src = self._src()
        with mock.patch.dict(sys.modules, {"fitz": fake_limit_fitz(120)}):
            path, truncated = engines._limit_pdf_pages(src, 50)  # noqa: SLF001
        self.assertTrue(truncated)
        self.assertNotEqual(path, src)
        self.assertTrue(Path(path).exists(), "裁剪结果应当落盘")
        Path(path).unlink()

    def test_keeps_original_when_within_the_cap(self) -> None:
        src = self._src()
        with mock.patch.dict(sys.modules, {"fitz": fake_limit_fitz(10)}):
            path, truncated = engines._limit_pdf_pages(src, 50)  # noqa: SLF001
        self.assertFalse(truncated)
        self.assertEqual(path, src)

    def test_without_pymupdf_it_warns_and_keeps_the_original(self) -> None:
        """宁可慢，也不要静默截断。"""

        src = self._src()
        with mock.patch.dict(sys.modules, {"fitz": None, "pymupdf": None}):
            path, truncated = engines._limit_pdf_pages(src, 50)  # noqa: SLF001
        self.assertFalse(truncated)
        self.assertEqual(path, src)

    def test_non_pdf_is_left_alone(self) -> None:
        src = self._src("photo.png")
        path, truncated = engines._limit_pdf_pages(src, 50)  # noqa: SLF001
        self.assertFalse(truncated)
        self.assertEqual(path, src)


class _Page:
    """假的结果条目，带 Paddle 风格的 page_index。"""

    def __init__(self, markdown: str, page=None) -> None:
        self.markdown = markdown
        self.json = {"page_index": page} if page is not None else {}


class TestPageOrdering(unittest.TestCase):
    """页序不能靠运气。

    真实事故：源 PDF 页序正常（第 6 页是 1.1 定义、第 15 页是 12.2），但整份输出里
    1.1 跑到了 76% 的位置。这里锁住两条：页号取真实值、每页打标记。
    """

    def test_natural_sort_beats_lexicographic(self) -> None:
        names = ["x_page_1.md", "x_page_10.md", "x_page_2.md"]
        got = [p.name for p in sorted(map(Path, names), key=engines._natural_key)]  # noqa: SLF001
        self.assertEqual(got, ["x_page_1.md", "x_page_2.md", "x_page_10.md"],
                         "字典序会把 page_10 排到 page_2 前面")

    def test_page_index_prefers_the_real_page_number(self) -> None:
        self.assertEqual(engines._item_page_index(_Page("x", 4), 1), 5)  # noqa: SLF001
        self.assertEqual(engines._item_page_index(_Page("x"), 7), 7)     # noqa: SLF001

    def test_every_page_gets_a_marker(self) -> None:
        out = engines._collect_paddle_markdown(  # noqa: SLF001
            [_Page("甲", 0), _Page("乙", 1)], tempfile.mkdtemp())
        self.assertIn("<!-- page 1 -->", out)
        self.assertIn("<!-- page 2 -->", out)
        self.assertLess(out.index("page 1"), out.index("page 2"))


class TestPipelineConstruction(unittest.TestCase):
    def _build(self, record: list, fail_on_kwargs: bool = False):
        module = fake_paddleocr(record, fail_on_kwargs)
        with mock.patch.dict(sys.modules, {"paddleocr": module}):
            engine = engines.PaddleStructureEngine()
            engine._ensure()  # noqa: SLF001
            return engine

    def test_toggles_are_passed_through(self) -> None:
        record: list = []
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": "formula,chart"}):
            engine = self._build(record)
        self.assertEqual(record, [{"use_doc_orientation_classify": True,
                                   "use_formula_recognition": False,
                                   "use_chart_recognition": False}])
        self.assertEqual(engine.disabled, ["chart", "formula"])

    def test_unknown_kwargs_fall_back_to_defaults(self) -> None:
        """参数名对不上时宁可全开，也不能让引擎起不来。"""

        record: list = []
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": "formula"}):
            engine = self._build(record, fail_on_kwargs=True)
        self.assertEqual(record, [{"use_doc_orientation_classify": True,
                                   "use_formula_recognition": False}, {}])
        self.assertEqual(engine.disabled, [])
        self.assertIsInstance(engine._pipeline, FakePipeline)  # noqa: SLF001

    def test_valueerror_is_tolerated_too(self) -> None:
        """paddlex 对陌生参数抛的是 ValueError（不是 Python 原生的 TypeError）。

        真实事故：只兜了 TypeError，于是 ``Unknown argument`` 直接把引擎带下去，
        远程 OCR 整个不可用。
        """

        record: list = []
        module = fake_paddleocr(record, fail_error=ValueError(
            "Unknown argument: use_doc_orientation_classify"))
        with mock.patch.dict(sys.modules, {"paddleocr": module}), \
                mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": ""}):
            engine = engines.PaddleStructureEngine()
            ok, why = engine.available()
        self.assertTrue(ok, why)
        self.assertEqual(record, [{"use_doc_orientation_classify": True}, {}])
        self.assertIsInstance(engine._pipeline, FakePipeline)  # noqa: SLF001

    def test_page_orientation_is_on_by_default(self) -> None:
        """整页方向分类默认打开（表格单元格能因此拿到更干净的裁剪）。"""

        record: list = []
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": ""}):
            self._build(record)
        self.assertEqual(record, [{"use_doc_orientation_classify": True}])

    def test_page_orientation_can_be_disabled_for_an_ab_test(self) -> None:
        record: list = []
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": "orientation"}):
            self._build(record)
        self.assertFalse(record[-1]["use_doc_orientation_classify"])

    def test_constructor_failure_is_reported_as_engine_error(self) -> None:
        module = fake_paddleocr([])

        def boom(**_kwargs):
            raise RuntimeError("CUDA out of memory")

        module.PPStructureV3 = boom
        with mock.patch.dict(sys.modules, {"paddleocr": module}), \
                mock.patch.dict(os.environ, {"TABLE_PREPASS_DISABLE": ""}):
            engine = engines.PaddleStructureEngine()
            ok, why = engine.available()
        self.assertFalse(ok)
        self.assertIn("CUDA out of memory", why)


class TestPrepassIsWiredIntoThePipeline(unittest.TestCase):
    """真正的"插进去"就在这里：送进 predict 的必须是**转正之后**的那份 PDF。"""

    def _run(self, angles, gray=None):
        seen: list = []

        class _RecordingPipeline:
            def predict(self, input=None):
                seen.append(input)
                return []

        record: list = []
        module = fake_paddleocr(record)
        module.DocImgOrientationClassification = fake_orientation_classifier()
        module.PPStructureV3 = lambda **kwargs: (record.append(kwargs), _RecordingPipeline())[1]
        built = FakeDoc(angles, gray)
        # 用一份全新的前置修正器：模块里那个是进程内共享的，测试之间不能互相污染。
        fresh = engines.OrientationPrepass()
        with mock.patch.dict(sys.modules, {"paddleocr": module,
                                           "pymupdf": fake_pymupdf(angles, built)}), \
                mock.patch.dict(os.environ, {"TABLE_PREPASS_ORIENT": "1"}, clear=False), \
                mock.patch.object(engines, "_PREPASS", fresh):
            engine = engines.PaddleStructureEngine()
            engine._ensure()  # noqa: SLF001
            text, meta = engine.run("scan.pdf", b"%PDF-1.4 fake", 200, 50)
        return seen, meta, text

    def test_rotated_page_is_fixed_before_predict(self) -> None:
        seen, meta, _ = self._run((0, 90))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].endswith(".pdf"))
        self.assertEqual(meta["rotated_pages"], [2])

    def test_the_prepass_temp_file_is_cleaned_up(self) -> None:
        seen, _meta, _text = self._run((0, 90))
        self.assertFalse(Path(seen[0]).exists(), "交给流水线的那份临时文件用完要删掉")

    def test_upright_document_is_handed_over_untouched(self) -> None:
        seen, meta, _ = self._run((0, 0))
        self.assertNotIn("rotated_pages", meta)
        self.assertTrue(seen[0].endswith(".pdf"))

    def test_fragment_page_is_reported_in_meta(self) -> None:
        _seen, meta, _text = self._run((0, 90), gray_with_fragment())
        self.assertEqual(meta["fragment_strips"], [2])


if __name__ == "__main__":
    unittest.main()
