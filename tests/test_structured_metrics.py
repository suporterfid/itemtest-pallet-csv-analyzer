"""Regression tests for structured-mode KPI calculations."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    calculate_expected_epc_stats,
    compile_structured_kpis,
    calculate_mode_performance,
)


class TestStructuredMetrics(unittest.TestCase):
    """Ensure structured-mode KPIs match the specification."""

    def setUp(self) -> None:  # noqa: D401 - standard unittest setup description
        """Create representative data for structured layout analysis."""

        self.summary = pd.DataFrame(
            {
                "EPC": [
                    "AAA111000000000000000000",
                    "FFF222000000000000000000",
                    "EEE333000000000000000000",
                    "WRONG555000000000000000000",
                    "NOP777000000000000000000",
                    "OUT888000000000000000000",
                ],
                "EPC_suffix3": ["111", "222", "333", "555", "777", "888"],
                "total_reads": [5, 2, 1, 20, 4, 3],
                "expected_epc": [True, True, False, False, False, False],
                "pallet_position": [
                    "Front - Row 1",
                    "Front - Row 1",
                    "—",
                    "Left - Row 1",
                    "—",
                    "—",
                ],
            }
        )
        total_reads = int(self.summary["total_reads"].sum())
        timestamps = pd.date_range("2024-01-01", periods=total_reads, freq="s")
        freq_cycle = [915.25, 915.5, 915.75, 916.0]
        rows: list[dict[str, object]] = []
        idx = 0
        for epc, reads in zip(self.summary["EPC"], self.summary["total_reads"]):
            for _ in range(int(reads)):
                rows.append(
                    {
                        "EPC": epc,
                        "RSSI": -50.0 - (idx % 7),
                        "Antenna": (idx % 3) + 1,
                        "Timestamp": timestamps[idx],
                        "Frequency": freq_cycle[idx % len(freq_cycle)],
                    }
                )
                idx += 1
        self.raw_df = pd.DataFrame(rows)
        ant_group = self.raw_df.groupby("Antenna")
        self.ant_counts = ant_group.size().reset_index(name="total_reads")
        self.ant_counts["rssi_avg"] = ant_group["RSSI"].mean().reset_index(drop=True)
        total_reads_by_ant = float(self.ant_counts["total_reads"].sum())
        self.ant_counts["participation_pct"] = (
            self.ant_counts["total_reads"].astype(float) / total_reads_by_ant * 100
        )
        self.positions_df = pd.DataFrame(
            [
                {
                    "Row": "1",
                    "Face": "Front",
                    "Suffix": "111",
                    "Read": True,
                    "total_reads": 5,
                    "PositionLabel": "Front - Row 1",
                    "ExpectedToken": "AAA111000000000000000000",
                    "ExpectedEPC": "AAA111000000000000000000",
                },
                {
                    "Row": "1",
                    "Face": "Front",
                    "Suffix": "222",
                    "Read": True,
                    "total_reads": 2,
                    "PositionLabel": "Front - Row 1",
                    "ExpectedToken": "222",
                    "ExpectedEPC": None,
                },
                {
                    "Row": "1",
                    "Face": "Left",
                    "Suffix": "555",
                    "Read": True,
                    "total_reads": 20,
                    "PositionLabel": "Left - Row 1",
                    "ExpectedToken": "FFF555000000000000000000",
                    "ExpectedEPC": "FFF555000000000000000000",
                },
                {
                    "Row": "2",
                    "Face": "Rear",
                    "Suffix": "999",
                    "Read": False,
                    "total_reads": 0,
                    "PositionLabel": "Rear - Row 2",
                    "ExpectedToken": "999",
                    "ExpectedEPC": None,
                },
            ]
        )
        self.expected_full = {
            "AAA111000000000000000000",
            "FFF555000000000000000000",
        }
        self.expected_suffixes = {"222", "999"}

    def test_expected_epc_stats_and_coverage(self) -> None:
        """Coverage and missing tokens should respect full and suffix sets."""

        stats = calculate_expected_epc_stats(
            self.summary,
            expected_full=self.expected_full,
            expected_suffixes=self.expected_suffixes,
        )

        self.assertEqual(stats["total_expected"], 4)
        self.assertEqual(stats["found_expected"], 2)
        self.assertIn("FFF555000000000000000000", stats["missing_full"])
        self.assertIn("999", stats["missing_suffix"])
        self.assertAlmostEqual(stats["coverage_rate"], 2 / 4 * 100, places=4)

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

        self.assertAlmostEqual(metrics["coverage_rate"], 2 / 4 * 100, places=2)
        self.assertEqual(metrics["expected_total"], 4)
        self.assertEqual(metrics["expected_found"], 2)
        self.assertAlmostEqual(metrics["tag_read_redundancy"], 3.5, places=2)

        proportions = self.ant_counts["total_reads"].astype(float) / self.ant_counts["total_reads"].sum()
        expected_balance = float(proportions.std(ddof=0) * 100)
        self.assertAlmostEqual(metrics["antenna_balance"], expected_balance, places=6)

        # RSSI stability: std deviation of mean RSSI per antenna
        rssi_means = (
            self.raw_df.dropna(subset=["RSSI", "Antenna"]).groupby("Antenna")["RSSI"].mean()
        )
        expected_rssi_stability = float(rssi_means.std(ddof=0))
        self.assertAlmostEqual(metrics["rssi_stability_index"], expected_rssi_stability, places=6)

        self.assertIsNotNone(metrics["top_performer_antenna"])
        self.assertIn(metrics["top_performer_antenna"]["antenna"], self.ant_counts["Antenna"].tolist())

        face_coverage = metrics["layout_face_coverage"]
        self.assertIsInstance(face_coverage, pd.DataFrame)
        self.assertFalse(face_coverage.empty)
        front_row = face_coverage.loc[face_coverage["Face"] == "Front"].iloc[0]
        self.assertEqual(int(front_row["total_positions"]), 2)
        self.assertEqual(int(front_row["read_positions"]), 2)

        self.assertIn("Rear - Row 2 (999)", metrics["missing_position_labels"])

        # New diagnostics
        self.assertEqual(metrics["read_hotspots_count"], 1)
        hotspots_df = metrics["read_hotspots"]
        self.assertEqual(hotspots_df.iloc[0]["EPC"], "WRONG555000000000000000000")
        expected_threshold = float(np.mean(self.summary["total_reads"]) + 2 * np.std(self.summary["total_reads"], ddof=0))
        self.assertAlmostEqual(metrics["read_hotspots_threshold"], expected_threshold)

        freq_usage = metrics["frequency_usage"]
        self.assertEqual(metrics["frequency_unique_count"], len(freq_usage))
        self.assertTrue(set(freq_usage["frequency_mhz"]) <= {915.25, 915.5, 915.75, 916.0})

        location_errors = metrics["location_errors"]
        self.assertEqual(metrics["location_error_count"], 1)
        location_row = location_errors.iloc[0]
        self.assertEqual(location_row["ExpectedEPC"], "FFF555000000000000000000")
        self.assertIn("Left - Row 1", location_row["ExpectedPosition"])

        reads_by_face = metrics["reads_by_face"]
        left_face = reads_by_face.loc[reads_by_face["Face"] == "Left"].iloc[0]
        self.assertEqual(int(left_face["total_reads"]), 20)

    def test_calculate_mode_performance_with_metadata(self) -> None:
        """Mode performance should derive read rates when metadata provides ModeIndex."""

        metadata = {"ModeIndex": 5}
        indicator = calculate_mode_performance(metadata, self.summary, self.raw_df)

        self.assertEqual(indicator["mode_index"], 5)

        duration_seconds = (
            self.raw_df["Timestamp"].max() - self.raw_df["Timestamp"].min()
        ).total_seconds()
        expected_reads_per_second = self.raw_df.shape[0] / duration_seconds
        expected_reads_per_minute = expected_reads_per_second * 60.0
        expected_epcs_per_minute = self.summary.shape[0] / (duration_seconds / 60.0)

        self.assertAlmostEqual(
            indicator["reads_per_second"], expected_reads_per_second, places=6
        )
        self.assertAlmostEqual(
            indicator["reads_per_minute"], expected_reads_per_minute, places=6
        )
        self.assertAlmostEqual(
            indicator["epcs_per_minute"], expected_epcs_per_minute, places=6
        )
        description = indicator["description"]
        self.assertIsInstance(description, str)
        self.assertIn("ModeIndex 5", description)
        self.assertIn("leituras/s", description)

    def test_calculate_mode_performance_without_mode_index(self) -> None:
        """Helper should return empty indicators when ModeIndex metadata is missing."""

        indicator = calculate_mode_performance({}, self.summary, self.raw_df)

        self.assertIsNone(indicator["mode_index"])
        self.assertIsNone(indicator["reads_per_second"])
        self.assertIsNone(indicator["reads_per_minute"])
        self.assertIsNone(indicator["epcs_per_minute"])
        self.assertIsNone(indicator["description"])


if __name__ == "__main__":
    unittest.main()
