"""HTTP 客户端（只用标准库）：把假的服务端响应喂进去，检查四条路径都对。"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ocr_table_prepass import client as client_mod
from ocr_table_prepass.client import OcrClient, endpoint_is_private


class FakeTransport(OcrClient):
    """把 ``_request`` 换成一张写死的应答表；顺手记下每个请求，供断言用。"""

    def __init__(self, replies: dict, **kwargs) -> None:
        super().__init__("http://127.0.0.1:8766", **kwargs)
        self.replies = replies
        self.calls: list = []

    def _request(self, path, method="GET", payload=None, timeout=10.0):
        self.calls.append((method, path, payload))
        reply = self.replies.get((method, path))
        if reply is None:
            reply = self.replies.get(path)
        if callable(reply):
            return reply(payload)
        if reply is None:
            raise AssertionError(f"没有准备这条应答：{method} {path}")
        return json.loads(json.dumps(reply))


class TestEndpointBoundary(unittest.TestCase):
    """数据边界：内部材料只允许发往内网 / 组网地址（默认不拦，可显式打开）。"""

    def test_private_and_mesh_addresses(self) -> None:
        for url in ("http://127.0.0.1:8766", "http://192.168.1.23:8766",
                    "http://10.0.0.5:8766", "http://100.101.102.103:8766",
                    "http://ocr-win:8766".replace("ocr-win", "box.ts.net")):
            allowed, why = endpoint_is_private(url)
            self.assertTrue(allowed, f"{url} 应判为内网/组网（{why}）")

    def test_public_address_is_flagged(self) -> None:
        allowed, why = endpoint_is_private("http://8.8.8.8:8766")
        self.assertFalse(allowed)
        self.assertIn("公网", why)

    def test_budget_scales_with_the_page_cap(self) -> None:
        """固定 300 秒和 50 页的上限自相矛盾：50 页光推理就要 300 秒。"""

        self.assertEqual(client_mod.budget_for(50), 720.0)
        self.assertEqual(client_mod.budget_for(1), 300.0)


class TestOcrClient(unittest.TestCase):
    def test_done_path(self) -> None:
        transport = FakeTransport({
            ("POST", "/jobs"): {"job_id": "j1", "status": "queued"},
            "/jobs/j1": {"status": "done", "text": "| 甲 | 乙 |", "pages": 2,
                         "engine": "pp-structure-v3", "elapsed": 7.5,
                         "meta": {"rotated_pages": [2]}},
        })
        with mock.patch.object(client_mod.time, "sleep", lambda _s: None):
            result = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "| 甲 | 乙 |")
        self.assertEqual(result.meta["rotated_pages"], [2])
        posted = [c for c in transport.calls if c[1] == "/jobs"][0][2]
        self.assertEqual(base64.b64decode(posted["data"]), b"%PDF-1.4")

    def test_error_status_is_reported(self) -> None:
        transport = FakeTransport({
            ("POST", "/jobs"): {"job_id": "j1"},
            "/jobs/j1": {"status": "error", "error": "引擎不可用"},
        })
        with mock.patch.object(client_mod.time, "sleep", lambda _s: None):
            result = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
        self.assertFalse(result.ok)
        self.assertIn("引擎不可用", result.error)

    def test_missing_job_id_is_reported(self) -> None:
        transport = FakeTransport({("POST", "/jobs"): {"status": "queued"}})
        result = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
        self.assertIn("未返回 job_id", result.error)

    def test_cache_hit_skips_the_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replies = {
                "/health": {"ok": True, "engine": "pp-structure-v3",
                            "engine_version": "3.7.0"},
                ("POST", "/jobs"): {"job_id": "j1"},
                "/jobs/j1": {"status": "done", "text": "结果", "pages": 1,
                             "engine": "pp-structure-v3"},
            }
            transport = FakeTransport(replies, cache_dir=tmp)
            with mock.patch.object(client_mod.time, "sleep", lambda _s: None):
                first = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
                calls_after_first = len(transport.calls)
                second = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(second.text, "结果")
            self.assertEqual(len(transport.calls), calls_after_first,
                             "缓存命中时不该再发任何请求")

    def test_signature_falls_back_to_a_stable_value_when_offline(self) -> None:
        """服务在不在线不能影响指纹：否则远端一关机，整库会被重解析。"""

        transport = FakeTransport({"/health": {"ok": False, "note": "连不上"}})
        self.assertEqual(transport.signature(), "ocr:on")

    def test_signature_uses_the_engine_version(self) -> None:
        transport = FakeTransport({"/health": {"ok": True, "engine": "rapidocr",
                                              "engine_version": "1.3.0"}})
        self.assertEqual(transport.signature(), "ocr:rapidocr@1.3.0")

    def test_public_endpoint_can_be_blocked(self) -> None:
        transport = OcrClient("http://8.8.8.8:8766", allow_public=False)
        result = transport.ocr_bytes("a.pdf", b"%PDF-1.4")
        self.assertIn("数据边界", result.error)

    def test_no_url_is_reported_instead_of_raising(self) -> None:
        with mock.patch.dict(os.environ, {"TABLE_PREPASS_URL": "",
                                          "COMPS_OCR_URL": ""}, clear=False):
            result = OcrClient().ocr_bytes("a.pdf", b"%PDF-1.4")
        self.assertFalse(result.ok)
        self.assertIn("未配置服务地址", result.error)

    def test_missing_file_is_reported(self) -> None:
        transport = FakeTransport({}, cache_dir=None)
        result = transport.ocr_file("/definitely/not/here.pdf")
        self.assertFalse(result.ok)
        self.assertIn("无法读取文件", result.error)


class TestPrepassOverHttp(unittest.TestCase):
    def test_prepass_round_trip(self) -> None:
        payload = base64.b64encode(b"%PDF-1.4 fixed").decode("ascii")
        transport = FakeTransport({
            ("POST", "/prepass"): {"name": "a.pdf", "pdf": payload, "bytes": 13,
                                   "report": {"rotated_pages": {"2": 90},
                                              "fragment_strips": {"2": [10, 20, 500, 21]}}},
        })
        fixed, report = transport.prepass_bytes("a.pdf", b"%PDF-1.4", "all")
        self.assertEqual(fixed, b"%PDF-1.4 fixed")
        self.assertEqual(report["rotated_pages"], {"2": 90})
        sent = transport.calls[0][2]
        self.assertEqual(sent["fragment_mode"], "all")

    def test_prepass_file_writes_the_result(self) -> None:
        payload = base64.b64encode(b"%PDF-1.4 fixed").decode("ascii")
        transport = FakeTransport({
            ("POST", "/prepass"): {"pdf": payload, "report": {}}})
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.pdf"
            source.write_bytes(b"%PDF-1.4")
            target = Path(tmp) / "a.fixed.pdf"
            fixed, _report = transport.prepass_file(source, target)
            self.assertEqual(target.read_bytes(), b"%PDF-1.4 fixed")
            self.assertEqual(fixed, b"%PDF-1.4 fixed")

    def test_prepass_respects_the_boundary_guard(self) -> None:
        transport = OcrClient("http://8.8.8.8:8766", allow_public=False)
        with self.assertRaises(RuntimeError):
            transport.prepass_bytes("a.pdf", b"%PDF-1.4")


class TestServiceModule(unittest.TestCase):
    def test_import_is_safe_without_fastapi(self) -> None:
        """没装 FastAPI 时 import 本模块只提示一句，不能崩在顶层。"""

        from ocr_table_prepass import service

        if service.app is None:
            self.assertEqual(service.main(), 2)
        else:                                   # pragma: no cover - 装了 FastAPI 才走到
            self.assertTrue(hasattr(service.app, "routes"))


if __name__ == "__main__":
    unittest.main()
