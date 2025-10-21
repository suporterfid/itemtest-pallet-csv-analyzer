# -*- coding: utf-8 -*-
"""Tests ensuring the textual summary is generated correctly."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.itemtest_analyzer import (
    compose_summary_text,
    build_arg_parser,
    generate_consolidated_summary,
)


class TestComposeSummaryText(unittest.TestCase):
    """Ensure the textual summary highlights essential information."""

    def test_summary_includes_hostname_from_metadata(self) -> None:
        metadata = {"Hostname": "192.168.68.100"}
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142"],
                "total_reads": [12],
                "first_time": [pd.Timestamp("2025-01-01T10:00:00Z")],
                "last_time": [pd.Timestamp("2025-01-01T10:15:00Z")],
                "expected_epc": [True],
            }
        )
        ant_counts = pd.DataFrame(
            {
                "Antenna": [1],
                "total_reads": [12],
                "participation_pct": [100.0],
                "rssi_avg": [-53.0],
            }
        )

        summary_text = compose_summary_text(
            Path("dummy.csv"), metadata, summary, ant_counts, positions_df=None
        )

        self.assertIn("- Hostname: 192.168.68.100", summary_text)


class TestCliSummaryFlag(unittest.TestCase):
    """Validate the CLI parser wiring for the consolidated summary flag."""

    def test_summary_flag_defaults_to_false(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "--input",
            "input_dir",
            "--output",
            "output_dir",
        ])
        self.assertFalse(args.summary)

    def test_summary_flag_enabled(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--input",
                "input_dir",
                "--output",
                "output_dir",
                "--summary",
            ]
        )
        self.assertTrue(args.summary)


class TestGenerateConsolidatedSummary(unittest.TestCase):
    """Ensure the consolidated executive workbook is generated correctly."""

    def test_generate_summary_workbook(self) -> None:
        records = [
            {
                "file": "structured.csv",
                "mode": "structured",
                "hostname": "192.168.0.10",
                "layout_used": True,
                "total_epcs": 10,
                "total_reads": 120,
                "expected_detected": 8,
                "unexpected_detected": 2,
                "coverage_rate": 80.0,
                "expected_total": 10,
                "expected_found": 8,
                "tag_read_redundancy": 1.5,
                "antenna_balance": 4.2,
                "rssi_stability_index": 1.1,
                "top_performer": "1 (55.0% of reads), 120 reads",
                "layout_total_positions": 40,
                "layout_read_positions": 32,
                "layout_overall_coverage": 80.0,
                "first_read": "2025-01-01 10:00:00",
                "last_read": "2025-01-01 10:05:00",
                "excel_report": "structured.xlsx",
                "summary_log": "structured.txt",
            },
            {
                "file": "continuous.csv",
                "mode": "continuous",
                "hostname": "192.168.0.11",
                "layout_used": False,
                "total_epcs": 20,
                "total_reads": 180,
                "expected_detected": 0,
                "unexpected_detected": 20,
                "average_dwell_seconds": 2.5,
                "throughput_per_minute": 30.0,
                "read_continuity_rate": 82.0,
                "session_duration_seconds": 600.0,
                "session_active_seconds": 520.0,
                "concurrency_peak": 7,
                "concurrency_average": 3.4,
                "concurrency_peak_time": "2025-01-02 12:05:00",
                "dominant_antenna": 3,
                "alerts_count": 1,
                "analysis_window_seconds": 2.0,
                "epcs_per_minute_mean": 28.0,
                "epcs_per_minute_peak": 36,
                "epcs_per_minute_peak_time": "2025-01-02 12:06:00",
                "first_read": "2025-01-02 12:00:00",
                "last_read": "2025-01-02 12:10:00",
                "excel_report": "continuous.xlsx",
                "summary_log": "continuous.txt",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = generate_consolidated_summary(records, Path(tmp_dir))
            self.assertIsNotNone(out_path)
            assert out_path is not None
            self.assertTrue(out_path.exists())

            per_file_df = pd.read_excel(out_path, sheet_name="Detalhes_Por_Arquivo")
            overview_df = pd.read_excel(out_path, sheet_name="Resumo_Geral")

        self.assertEqual(per_file_df.shape[0], 2)
        self.assertEqual(int(per_file_df["total_reads"].sum()), 300)
        modes = set(per_file_df["mode"].str.lower())
        self.assertIn("structured", modes)
        self.assertIn("continuous", modes)

        overall_row = overview_df.loc[overview_df["mode"].str.lower() == "overall"]
        self.assertEqual(int(overall_row.iloc[0]["total_reads"]), 300)
        self.assertAlmostEqual(float(overall_row.iloc[0]["avg_throughput_per_minute"]), 30.0, places=2)

    def test_generate_summary_with_empty_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_consolidated_summary([], Path(tmp_dir))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
