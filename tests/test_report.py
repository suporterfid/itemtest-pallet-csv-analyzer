# -*- coding: utf-8 -*-
"""Tests ensuring the metadata worksheet preserves the hostname entry."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.report import write_excel


class TestReportWorkbookStructure(unittest.TestCase):
    """Validate workbook structure and executive indicators."""

    def test_workbook_uses_mandated_sheet_names(self) -> None:
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142", "000000000000000000000001"],
                "total_reads": [12, 8],
            }
        )
        unexpected = pd.DataFrame(
            {
                "EPC": ["FFFFFFFFFFFFFFFFFFFFFFFF"],
                "total_reads": [3],
            }
        )
        ant_counts = pd.DataFrame(
            {
                "Antenna": [1, 2],
                "total_reads": [12, 11],
                "participation_pct": [52.2, 47.8],
            }
        )
        metadata = {"Hostname": "192.168.68.100", "Session": 1}
        positions = pd.DataFrame(
            {
                "Position": ["A1"],
                "EPC": ["303132333435363738394142"],
                "Reads": [12],
            }
        )
        structured_metrics = {
            "coverage_rate": 96.0,
            "expected_total": 50,
            "expected_found": 48,
            "tag_read_redundancy": 2.1,
            "antenna_balance": 4.3,
            "rssi_stability_index": 1.2,
            "top_performer_antenna": {
                "antenna": 1,
                "participation_pct": 52.2,
                "total_reads": 12,
            },
        }
        continuous_metrics = {
            "average_dwell_seconds": 1.8,
            "throughput_per_minute": 37.5,
            "session_throughput": 58.2,
            "read_continuity_rate": 83.4,
            "session_duration_seconds": 420.0,
            "session_active_seconds": 380.0,
            "concurrency_average": 2.6,
            "concurrency_peak": 5,
            "concurrency_peak_time": pd.Timestamp("2025-01-01 12:03:00"),
            "epcs_per_minute_mean": 33.0,
            "epcs_per_minute_peak": 58,
            "epcs_per_minute_peak_time": pd.Timestamp("2025-01-01 12:04:00"),
            "dominant_antenna": 2,
            "tag_dwell_time_max": 4.5,
            "inactive_periods_count": 2,
            "inactive_total_seconds": 45.0,
            "inactive_longest_seconds": 30.0,
            "congestion_index": 0.75,
            "global_rssi_avg": -51.2,
            "global_rssi_std": 1.4,
            "inactive_periods": pd.DataFrame(
                [
                    {
                        "start_time": pd.Timestamp("2025-01-01 12:10:00"),
                        "end_time": pd.Timestamp("2025-01-01 12:12:00"),
                        "duration_seconds": 120.0,
                        "gap_seconds": 130.0,
                        "gap_multiplier": 6.5,
                    }
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.xlsx"
            write_excel(
                str(out_path),
                summary,
                unexpected,
                ant_counts,
                metadata,
                positions_df=positions,
                structured_metrics=structured_metrics,
                continuous_metrics=continuous_metrics,
            )

            with pd.ExcelFile(out_path) as workbook:
                sheet_names = set(workbook.sheet_names)
        expected_names = {
            "Resumo_por_EPC",
            "EPCs_inesperados",
            "Leituras_por_Antena",
            "Metadata",
            "Indicadores_Executivos",
            "Posicoes_Pallet",
            "Structured_KPIs",
            "Fluxo_Contínuo",
        }
        self.assertTrue(
            expected_names.issubset(sheet_names),
            f"Workbook sheets missing: {expected_names - sheet_names}",
        )

    def test_executive_sheet_includes_structured_metrics(self) -> None:
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142", "000000000000000000000001"],
                "total_reads": [12, 8],
            }
        )
        ant_counts = pd.DataFrame({"Antenna": [1], "total_reads": [20]})
        metadata = {"Hostname": "reader", "Session": 1}
        structured_metrics = {
            "coverage_rate": 95.0,
            "expected_total": 100,
            "expected_found": 95,
            "tag_read_redundancy": 2.4,
            "antenna_balance": 3.2,
            "rssi_stability_index": 1.25,
            "top_performer_antenna": {
                "antenna": 3,
                "participation_pct": 57.8,
                "total_reads": 120,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.xlsx"
            write_excel(
                str(out_path),
                summary,
                summary.iloc[0:0].copy(),
                ant_counts,
                metadata,
                structured_metrics=structured_metrics,
            )

            exec_df = pd.read_excel(out_path, sheet_name="Indicadores_Executivos")

        exec_values = exec_df.set_index("Indicator")["Value"].to_dict()

        self.assertEqual(exec_values.get("Coverage rate"), "95.00% (95/100)")
        self.assertEqual(exec_values.get("Tag read redundancy"), "2.40×")
        self.assertEqual(exec_values.get("Antenna balance (σ)"), "3.20%")
        self.assertEqual(exec_values.get("RSSI stability index (σ)"), "1.25 dBm")
        self.assertEqual(
            exec_values.get("Top performer antenna"),
            "3 (57.8% of reads), 120 reads",
        )

    def test_executive_sheet_includes_continuous_metrics(self) -> None:
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142", "000000000000000000000001"],
                "total_reads": [12, 9],
            }
        )
        ant_counts = pd.DataFrame({"Antenna": [1, 2], "total_reads": [12, 9]})
        metadata = {"Hostname": "reader", "Session": 2}
        continuous_metrics = {
            "average_dwell_seconds": 12.34,
            "throughput_per_minute": 45.6,
            "session_throughput": 72.5,
            "read_continuity_rate": 88.9,
            "session_duration_seconds": 300.0,
            "session_active_seconds": 240.0,
            "concurrency_average": 3.5,
            "concurrency_peak": 7,
            "concurrency_peak_time": pd.Timestamp("2025-01-02 08:15:00"),
            "epcs_per_minute_mean": 42.1,
            "epcs_per_minute_peak": 60,
            "epcs_per_minute_peak_time": pd.Timestamp("2025-01-02 08:17:00"),
            "dominant_antenna": 4,
            "tag_dwell_time_max": 18.9,
            "inactive_periods_count": 4,
            "inactive_total_seconds": 55.0,
            "inactive_longest_seconds": 20.0,
            "congestion_index": 0.42,
            "global_rssi_avg": -49.8,
            "global_rssi_std": 1.25,
            "inactive_periods": pd.DataFrame(
                [
                    {
                        "start_time": pd.Timestamp("2025-01-02 08:05:00"),
                        "end_time": pd.Timestamp("2025-01-02 08:06:30"),
                        "duration_seconds": 90.0,
                        "gap_seconds": 100.0,
                        "gap_multiplier": 5.0,
                    }
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.xlsx"
            write_excel(
                str(out_path),
                summary,
                summary.iloc[0:0].copy(),
                ant_counts,
                metadata,
                continuous_metrics=continuous_metrics,
            )

            exec_df = pd.read_excel(out_path, sheet_name="Indicadores_Executivos")
            flow_sheet = pd.read_excel(out_path, sheet_name="Fluxo_Contínuo", header=None)

        exec_values = exec_df.set_index("Indicator")["Value"]

        self.assertAlmostEqual(float(exec_values["Average dwell time (s)"]), 12.34, places=2)
        self.assertAlmostEqual(float(exec_values["Maximum dwell time (s)"]), 18.9, places=1)
        self.assertAlmostEqual(
            float(exec_values["Throughput (distinct EPCs/min)"]), 45.6, places=2
        )
        self.assertAlmostEqual(
            float(exec_values["Session throughput (reads/min)"]), 72.5, places=1
        )
        self.assertAlmostEqual(
            float(exec_values["Read continuity rate (%)"]), 88.9, places=1
        )
        self.assertAlmostEqual(float(exec_values["Average active EPCs/min"]), 42.1, places=1)
        self.assertAlmostEqual(
            float(exec_values["Congestion index (reads/s)"]), 0.42, places=2
        )
        self.assertAlmostEqual(
            float(exec_values["Total inactive time (s)"]), 55.0, places=1
        )
        self.assertAlmostEqual(
            float(exec_values["Longest inactive period (s)"]), 20.0, places=1
        )
        self.assertAlmostEqual(
            float(exec_values["Global RSSI mean (dBm)"]), -49.8, places=1
        )
        self.assertAlmostEqual(
            float(exec_values["Global RSSI std (dBm)"]), 1.25, places=2
        )
        self.assertEqual(
            str(exec_values["Peak active EPCs/min"]), "60 at 2025-01-02 08:17:00"
        )
        self.assertEqual(
            str(exec_values["Peak concurrent EPCs"]), "7 at 2025-01-02 08:15:00"
        )
        self.assertEqual(
            int(float(exec_values["Inactive periods (>5× window)"])), 4
        )

        first_column = flow_sheet.iloc[:, 0].astype(str).tolist()
        self.assertIn("Maximum dwell time (s)", first_column)
        self.assertIn("Session throughput (reads/min)", first_column)
        self.assertIn("Inactive periods (>5× window)", first_column)
        self.assertTrue(any("start_time" in str(value) for value in first_column))
        self.assertEqual(str(exec_values["Dominant antenna"]), "4")


class TestReportMetadataSheet(unittest.TestCase):
    """Ensure the generated workbook includes essential metadata."""

    def test_metadata_sheet_includes_hostname(self) -> None:
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142"],
                "total_reads": [12],
            }
        )
        unexpected = summary.iloc[0:0].copy()
        ant_counts = pd.DataFrame({"Antenna": [1], "total_reads": [12]})
        metadata = {"Hostname": "192.168.68.100", "Session": 1}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.xlsx"
            write_excel(str(out_path), summary, unexpected, ant_counts, metadata)

            md_df = pd.read_excel(out_path, sheet_name="Metadata")

        mask = (md_df["Key"] == "Hostname") & (md_df["Value"] == "192.168.68.100")
        self.assertTrue(mask.any(), "Metadata sheet did not list the expected Hostname entry.")


if __name__ == "__main__":
    unittest.main()
