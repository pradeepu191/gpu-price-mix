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


if __name__ == "__main__":
    unittest.main()
