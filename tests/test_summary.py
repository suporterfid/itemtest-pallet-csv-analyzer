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

    def test_summary_highlights_continuous_new_metrics(self) -> None:
        metadata: dict = {}
        summary = pd.DataFrame(
            {
                "EPC": ["ABC"],
                "total_reads": [5],
                "first_time": [pd.Timestamp("2025-01-01T10:00:00")],
                "last_time": [pd.Timestamp("2025-01-01T10:05:00")],
            }
        )
        ant_counts = pd.DataFrame({"Antenna": [1], "total_reads": [5]})
        continuous_details = {
            "average_dwell_seconds": 3.2,
            "tag_dwell_time_max": 6.4,
            "read_continuity_rate": 75.0,
            "total_events": 4,
            "session_start": pd.Timestamp("2025-01-01T10:00:00"),
            "session_end_with_grace": pd.Timestamp("2025-01-01T10:06:00"),
            "session_duration_seconds": 360.0,
            "session_active_seconds": 270.0,
            "dominant_antenna": 1,
            "throughput_per_minute": 12.5,
            "session_throughput": 18.0,
            "epcs_per_minute_mean": 10.0,
            "concurrency_average": 2.0,
            "congestion_index": 0.5,
            "epcs_per_minute_peak": 12,
            "epcs_per_minute_peak_time": pd.Timestamp("2025-01-01T10:03:00"),
            "concurrency_peak": 3,
            "concurrency_peak_time": pd.Timestamp("2025-01-01T10:02:00"),
            "inactive_periods_count": 2,
            "inactive_total_seconds": 48.0,
            "inactive_longest_seconds": 30.0,
            "global_rssi_avg": -50.5,
            "global_rssi_std": 1.8,
        }

        summary_text = compose_summary_text(
            Path("continuous.csv"),
            metadata,
            summary,
            ant_counts,
            positions_df=None,
            analysis_mode="continuous",
            continuous_details=continuous_details,
        )

        self.assertIn("Tempo máximo de permanência", summary_text)
        self.assertIn("Throughput da sessão (leituras/min)", summary_text)
        self.assertIn("Índice de congestão", summary_text)
        self.assertIn("Períodos inativos (>5× janela)", summary_text)
        self.assertIn("RSSI médio global", summary_text)


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
                "session_throughput": 44.0,
                "read_continuity_rate": 82.0,
                "session_duration_seconds": 600.0,
                "session_active_seconds": 520.0,
                "tag_dwell_time_max": 7.5,
                "concurrency_peak": 7,
                "concurrency_average": 3.4,
                "concurrency_peak_time": "2025-01-02 12:05:00",
                "dominant_antenna": 3,
                "inactive_periods_count": 3,
                "inactive_total_seconds": 90.0,
                "inactive_longest_seconds": 40.0,
                "congestion_index": 0.35,
                "global_rssi_avg": -48.2,
                "global_rssi_std": 2.1,
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
        self.assertIn("session_throughput", per_file_df.columns)
        self.assertIn("tag_dwell_time_max", per_file_df.columns)
        self.assertIn("inactive_periods_count", per_file_df.columns)
        self.assertIn("global_rssi_avg", per_file_df.columns)

        overall_row = overview_df.loc[overview_df["mode"].str.lower() == "overall"]
        self.assertEqual(int(overall_row.iloc[0]["total_reads"]), 300)
        self.assertAlmostEqual(float(overall_row.iloc[0]["avg_throughput_per_minute"]), 30.0, places=2)

    def test_generate_summary_with_empty_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_consolidated_summary([], Path(tmp_dir))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
