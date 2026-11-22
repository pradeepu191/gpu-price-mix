"""Tests for flattening snapshots into the panel table."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import build_panel  # noqa: E402


class PanelTests(unittest.TestCase):
    def test_row_has_price_per_gpu_and_time(self):
        offer = {"id": 5, "machine_id": 9, "host_id": 3, "gpu_name": "H100 SXM",
                 "num_gpus": 4, "dph_base": 8.0, "rentable": False, "gpu_ids": [11, 12, 13, 14],
                 "public_ipaddr": "1.2.3.4"}
        row = build_panel.offer_row("20261119T020347Z", "on-demand", offer)
        self.assertAlmostEqual(row["price_per_gpu"], 2.0)
        self.assertEqual(row["snapshot_utc"], "2026-11-19T02:03:47+00:00")
        self.assertEqual(row["gpu_ids"], "11 12 13 14")
        self.assertNotIn("public_ipaddr", row)

    def test_missing_fields_become_blank(self):
        row = build_panel.offer_row("20261119T020347Z", "bid", {"id": 1, "num_gpus": 1, "dph_base": 1.0})
        self.assertIsNone(row["reliability"])


class RunFilterTests(unittest.TestCase):
    def test_full_only_keeps_runs_with_finished_lookups(self):
        import json
        import tempfile
        from pathlib import Path as P
        import vast_data
        rows = [
            {"market_type": "on-demand", "status": "complete", "run_id": "A"},
            {"market_type": "on-demand", "status": "complete", "run_id": "B", "lookup_status": "complete"},
            {"market_type": "on-demand", "status": "complete", "run_id": "C", "lookup_status": "partial"},
            {"market_type": "on-demand", "status": "failed", "run_id": "D", "lookup_status": "complete"},
            {"market_type": "bid", "status": "complete", "run_id": "E", "lookup_status": "complete"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = P(temp_dir) / "manifest"
            manifest.mkdir()
            with open(manifest / "vast_runs.jsonl", "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            self.assertEqual(vast_data.complete_runs(temp_dir, "on-demand"), ["A", "B", "C"])
            self.assertEqual(vast_data.complete_runs(temp_dir, "on-demand", full_only=True), ["B"])
            self.assertEqual(vast_data.latest_complete_run(temp_dir, "on-demand", full_only=True), "B")


if __name__ == "__main__":
    unittest.main()
