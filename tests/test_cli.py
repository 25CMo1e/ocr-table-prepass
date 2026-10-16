"""命令行：``fix`` / ``ocr`` / ``health`` 三个子命令的基本行为。"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ocr_table_prepass import __main__ as cli
from ocr_table_prepass import orient
from ocr_table_prepass.__main__ import main
from tests.fakes import (FakeDoc, fake_orientation_classifier, fake_pymupdf,
                         gray_with_fragment)


def _modules(angles, doc, gray=None):
    module = types.ModuleType("paddleocr")
    module.DocImgOrientationClassification = fake_orientation_classifier()
    return {"pymupdf": fake_pymupdf(angles, doc, gray), "paddleocr": module}


class TestFixCommand(unittest.TestCase):
    def test_writes_the_fixed_pdf_and_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixed.pdf"
            report_path = Path(tmp) / "report.json"
            doc = FakeDoc((0, 90), gray_with_fragment())
            with mock.patch.dict(sys.modules, _modules((0, 90), doc)):
                code = main(["fix", "scan.pdf", "-o", str(target),
                             "--report", str(report_path)])
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["rotated_pages"], {"2": 90})
            self.assertEqual(sorted(payload["fragment_strips"]), ["2"])

    def test_without_output_it_leaves_no_temp_file_behind(self) -> None:
        doc = FakeDoc((0, 90))
        created: list = []

        class _Spy:
            def __init__(self) -> None:
                self._inner = orient.OrientationPrepass()

            def apply(self, *args, **kwargs):
                report = self._inner.apply(*args, **kwargs)
                created.append(report.path)
                return report

        with mock.patch.dict(sys.modules, _modules((0, 90), doc)), \
                mock.patch.object(cli, "OrientationPrepass", _Spy):
            code = main(["fix", "scan.pdf"])
        self.assertEqual(code, 0)
        self.assertTrue(created)
        self.assertFalse(Path(created[0]).exists(), "临时文件应在收尾时删掉")

    def test_unchanged_document_can_fail_the_command(self) -> None:
        for args, expected in ((["fix", "scan.pdf"], 0),
                               (["fix", "scan.pdf", "--fail-if-unchanged"], 1)):
            doc = FakeDoc((0, 0))
            with mock.patch.dict(sys.modules, _modules((0, 0), doc)):
                self.assertEqual(main(args), expected)

    def test_mode_all_is_accepted(self) -> None:
        doc = FakeDoc((0, 0), gray_with_fragment())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixed.pdf"
            with mock.patch.dict(sys.modules, _modules((0, 0), doc,
                                                       gray_with_fragment())):
                code = main(["fix", "scan.pdf", "-o", str(target), "--mode", "all"])
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())


class TestRemoteCommands(unittest.TestCase):
    def test_health_reports_failure_when_nothing_is_listening(self) -> None:
        """端口 1 上不可能有服务：命令应当给出结论并返回 1，而不是抛异常。"""

        self.assertEqual(main(["health", "--url", "http://127.0.0.1:1"]), 1)

    def test_ocr_without_a_service_is_reported(self) -> None:
        self.assertEqual(main(["ocr", "/tmp/nope.pdf", "--url", "http://127.0.0.1:1"]), 1)


if __name__ == "__main__":
    unittest.main()
