"""Tests for the Ornn collector. Parsing tests come from the starter; failure tests are new."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collectors" / "ornn"))

from collect_ornn import Collector, normalize  # noqa: E402


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class NormalizeTests(unittest.TestCase):
    def test_history(self):
        payload = {"success": True, "gpu_type": "H100 SXM", "data": [
            {"timestamp": "2026-11-10T00:00:00.000Z", "index_value": 2.5},
            {"timestamp": "2026-11-11T00:00:00.000Z", "index_value": 2.6},
        ]}
        rows = normalize("gpu_history", payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["instrument"], "H100 SXM")
        self.assertEqual(rows[0]["value_text"], "2.5")
        self.assertEqual(rows[0]["observation_kind"], "daily_settlement")

    def test_daily_gpu(self):
        payload = {"success": True, "date": "2026-11-11T00:00:00Z", "data": [
            {"gpu_type": "B200", "region": "global", "index_value": 6.85}
        ]}
        rows = normalize("gpu_daily", payload)
        self.assertEqual(rows[0]["source_time"], payload["date"])
        self.assertEqual(rows[0]["region"], "global")

    def test_current_gpu_is_not_daily(self):
        payload = {"success": True, "data": {"gpu_name": "H200", "index_value": 3.1,
            "last_updated": "2026-11-11T16:02:00Z"}}
        rows = normalize("gpu_spot", payload)
        self.assertEqual(rows[0]["observation_kind"], "current_price")
        self.assertEqual(rows[0]["source_time"], "2026-11-11T16:02:00Z")

    def test_token_index(self):
        payload = {"success": True, "data": [{"date": "2026-11-11", "lab": "anthropic", "indexPerMtok": 1.2}]}
        rows = normalize("otpi", payload)
        self.assertEqual(rows[0]["unit"], "USD/million_tokens")
        self.assertEqual(rows[0]["instrument"], "anthropic")

    def test_unexpected_shape_rejected(self):
        with self.assertRaises(ValueError):
            normalize("gpu_spot", {"data": []})

    def test_missing_price_rejected(self):
        with self.assertRaises(ValueError):
            normalize("otpi", {"data": [{"date": "2026-11-11", "lab": "anthropic"}]})


class ArchiveTests(unittest.TestCase):
    def test_archive_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = Collector(Path(temp_dir))
            content = b'{"success":true,"data":[]}'
            _, raw_path, digest = collector._archive("ornn", "test", content, "https://example.com")
            self.assertEqual((Path(temp_dir) / raw_path).read_bytes(), content)
            self.assertEqual(len(digest), 64)
            collector.close()

    def test_dedupe_key_supports_revisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = Collector(Path(temp_dir))
            fetch_values = [2.50, 2.50, 2.55]

            def fake_open(_request, timeout):
                value = fetch_values.pop(0)
                payload = {"success": True, "data": [
                    {"date": "2026-11-11T00:00:00Z", "gpu_type": "H100 SXM", "index_value": value}
                ]}
                return FakeResponse(json.dumps(payload).encode("utf-8"))

            with patch("collect_ornn.urlopen", side_effect=fake_open):
                for _ in range(3):
                    collector.retrieve("gpu_daily", "/api/daily-index/all")
            n_fetches = collector.conn.execute("SELECT COUNT(*) FROM fetches").fetchone()[0]
            n_obs = collector.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            self.assertEqual(n_fetches, 3)
            self.assertEqual(n_obs, 2)
            collector.close()



class FailureTests(unittest.TestCase):
    def test_changed_shape_still_archives_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = Collector(Path(temp_dir))
            body = b'{"success": true, "data": "a shape we did not expect"}'
            with patch("collect_ornn.urlopen", return_value=FakeResponse(body)):
                result = collector.retrieve("gpu_daily", "/api/daily-index/all")
            self.assertIsNone(result)
            self.assertEqual(collector.errors, 1)
            status, raw_file = collector.conn.execute("SELECT status, raw_file FROM fetches").fetchone()
            self.assertEqual(status, "parse_error")
            self.assertEqual((Path(temp_dir) / raw_file).read_bytes(), body)
            collector.close()

    def test_http_failure_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = Collector(Path(temp_dir))
            error = HTTPError("https://example.com", 404, "Not Found", {}, io.BytesIO(b'{"error":"nope"}'))
            with patch("collect_ornn.urlopen", side_effect=error):
                result = collector.retrieve("gpu_daily", "/api/daily-index/all")
            self.assertIsNone(result)
            status, problem, raw_file = collector.conn.execute(
                "SELECT status, problem, raw_file FROM fetches"
            ).fetchone()
            self.assertEqual(status, "failed")
            self.assertEqual(problem, "http_404")
            self.assertIsNotNone(raw_file)
            collector.close()


if __name__ == "__main__":
    unittest.main()
