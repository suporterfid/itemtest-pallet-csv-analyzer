"""Regression tests for structured-mode KPI calculations."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    calculate_expected_epc_stats,
    compile_structured_kpis,
)


class TestStructuredMetrics(unittest.TestCase):
    """Ensure structured-mode KPIs match the specification."""

    def setUp(self) -> None:  # noqa: D401 - standard unittest setup description
        """Create representative data for structured layout analysis."""

        self.summary = pd.DataFrame(
            {
                "EPC": ["AAA111", "FFF222", "EEE333"],
                "EPC_suffix3": ["111", "222", "333"],
                "total_reads": [5, 2, 1],
                "expected_epc": [True, True, False],
            }
        )
        timestamps = pd.date_range("2024-01-01", periods=6, freq="s")
        self.raw_df = pd.DataFrame(
            {
                "EPC": ["AAA111", "AAA111", "AAA111", "FFF222", "FFF222", "EEE333"],
                "RSSI": [-50.2, -52.1, -51.0, -60.0, -61.2, -65.5],
                "Antenna": [1, 1, 2, 2, 3, 1],
                "Timestamp": timestamps,
            }
        )
        self.ant_counts = pd.DataFrame(
            {
                "Antenna": [1, 2, 3],
                "total_reads": [3, 2, 1],
                "rssi_avg": [-53.5667, -60.6, -61.2],
                "participation_pct": [50.0, 33.3333333333, 16.6666666667],
            }
        )
        self.positions_df = pd.DataFrame(
            [
                {"Row": "1", "Face": "Front", "Suffix": "111", "Read": True, "total_reads": 5},
                {"Row": "1", "Face": "Front", "Suffix": "222", "Read": True, "total_reads": 2},
                {"Row": "1", "Face": "Left", "Suffix": "555", "Read": False, "total_reads": 0},
            ]
        )
        self.expected_full = {"AAA111"}
        self.expected_suffixes = {"222", "555"}

    def test_expected_epc_stats_and_coverage(self) -> None:
        """Coverage and missing tokens should respect full and suffix sets."""

        stats = calculate_expected_epc_stats(
            self.summary,
            expected_full=self.expected_full,
            expected_suffixes=self.expected_suffixes,
        )

        self.assertEqual(stats["total_expected"], 3)
        self.assertEqual(stats["found_expected"], 2)
        self.assertIn("555", stats["missing_suffix"])
        self.assertAlmostEqual(stats["coverage_rate"], 2 / 3 * 100, places=4)

    def test_compile_structured_kpis_produces_expected_values(self) -> None:
        """Aggregated KPIs should reflect redundancy, balance, and layout coverage."""

        metrics = compile_structured_kpis(
            self.summary,
            self.raw_df,
            self.ant_counts,
            expected_full=self.expected_full,
            expected_suffixes=self.expected_suffixes,
            positions_df=self.positions_df,
        )

        self.assertAlmostEqual(metrics["coverage_rate"], 2 / 3 * 100, places=2)
        self.assertEqual(metrics["expected_total"], 3)
        self.assertEqual(metrics["expected_found"], 2)
        self.assertAlmostEqual(metrics["tag_read_redundancy"], 3.5, places=2)

        proportions = np.array([0.5, 2 / 6, 1 / 6])
        expected_balance = float(proportions.std(ddof=0) * 100)
        self.assertAlmostEqual(metrics["antenna_balance"], expected_balance, places=6)

        # RSSI stability: std deviation of mean RSSI per antenna
        rssi_means = (
            self.raw_df.dropna(subset=["RSSI", "Antenna"]).groupby("Antenna")["RSSI"].mean()
        )
        expected_rssi_stability = float(rssi_means.std(ddof=0))
        self.assertAlmostEqual(metrics["rssi_stability_index"], expected_rssi_stability, places=6)

        self.assertIsNotNone(metrics["top_performer_antenna"])
        self.assertEqual(metrics["top_performer_antenna"]["antenna"], 1)

        face_coverage = metrics["layout_face_coverage"]
        self.assertIsInstance(face_coverage, pd.DataFrame)
        self.assertFalse(face_coverage.empty)
        front_row = face_coverage.loc[face_coverage["Face"] == "Front"].iloc[0]
        self.assertEqual(int(front_row["total_positions"]), 2)
        self.assertEqual(int(front_row["read_positions"]), 2)

        self.assertIn("Left - Row 1 (555)", metrics["missing_position_labels"])


if __name__ == "__main__":
    unittest.main()
