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
        hotspots_df = pd.DataFrame(
            [
                {
                    "EPC": "WRONG555",
                    "EPC_suffix3": "555",
                    "total_reads": 20,
                    "expected_epc": False,
                    "pallet_position": "Left - Row 1",
                    "z_score": 2.5,
                }
            ]
        )
        frequency_df = pd.DataFrame(
            [
                {"frequency_mhz": 915.25, "read_count": 10, "participation_pct": 50.0},
                {"frequency_mhz": 915.75, "read_count": 10, "participation_pct": 50.0},
            ]
        )
        location_df = pd.DataFrame(
            [
                {
                    "EPC": "WRONG555",
                    "EPC_suffix3": "555",
                    "total_reads": 20,
                    "ExpectedEPC": "FFF555",
                    "ExpectedPosition": "Left - Row 1",
                    "ObservedPosition": "Left - Row 1",
                }
            ]
        )
        reads_face_df = pd.DataFrame(
            [
                {
                    "Face": "Left",
                    "total_positions": 10,
                    "positions_with_reads": 9,
                    "total_reads": 100,
                    "participation_pct": 60.0,
                },
                {
                    "Face": "Front",
                    "total_positions": 8,
                    "positions_with_reads": 7,
                    "total_reads": 40,
                    "participation_pct": 24.0,
                },
            ]
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
            "read_hotspots": hotspots_df,
            "read_hotspots_count": 1,
            "read_hotspots_threshold": 18.0,
            "frequency_usage": frequency_df,
            "frequency_unique_count": len(frequency_df),
            "location_errors": location_df,
            "location_error_count": len(location_df),
            "reads_by_face": reads_face_df,
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
            "global_rssi_avg": -47.3,
            "global_rssi_std": 3.2,
            "rssi_noise_flag": False,
            "rssi_noise_indicator": "Estabilidade de RSSI dentro do esperado (σ=3.20 dBm; 6.5 leituras/EPC)",
            "rssi_noise_reads_per_epc": 6.5,
            "top_performer_antenna": {
                "antenna": 3,
                "participation_pct": 57.8,
                "total_reads": 120,
            },
            "read_hotspots": pd.DataFrame(
                [
                    {
                        "EPC": "WRONG555",
                        "EPC_suffix3": "555",
                        "total_reads": 25,
                        "expected_epc": False,
                        "pallet_position": "Left - Row 1",
                        "z_score": 2.3,
                    }
                ]
            ),
            "read_hotspots_count": 1,
            "read_hotspots_threshold": 20.0,
            "frequency_usage": pd.DataFrame(
                [
                    {"frequency_mhz": 915.25, "read_count": 8, "participation_pct": 40.0},
                    {"frequency_mhz": 915.5, "read_count": 12, "participation_pct": 60.0},
                ]
            ),
            "frequency_unique_count": 2,
            "location_errors": pd.DataFrame(
                [
                    {
                        "EPC": "WRONG555",
                        "EPC_suffix3": "555",
                        "total_reads": 25,
                        "ExpectedEPC": "FFF555",
                        "ExpectedPosition": "Left - Row 1",
                        "ObservedPosition": "Left - Row 1",
                    }
                ]
            ),
            "location_error_count": 1,
            "reads_by_face": pd.DataFrame(
                [
                    {
                        "Face": "Left",
                        "total_positions": 12,
                        "positions_with_reads": 10,
                        "total_reads": 140,
                        "participation_pct": 70.0,
                    }
                ]
            ),
        }

        positions_df = pd.DataFrame(
            [
                {
                    "Row": "1",
                    "Face": "Front",
                    "Suffix": "111",
                    "Read": True,
                    "total_reads": 10,
                    "PositionLabel": "Front - Row 1",
                    "ExpectedToken": "111",
                    "ExpectedEPC": None,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.xlsx"
            write_excel(
                str(out_path),
                summary,
                summary.iloc[0:0].copy(),
                ant_counts,
                metadata,
                positions_df=positions_df,
                structured_metrics=structured_metrics,
            )

            exec_df = pd.read_excel(out_path, sheet_name="Indicadores_Executivos")
            structured_sheet = pd.read_excel(out_path, sheet_name="Structured_KPIs", header=None)
            positions_sheet = pd.read_excel(out_path, sheet_name="Posicoes_Pallet", header=None)

        exec_values = exec_df.set_index("Indicator")["Value"].to_dict()

        self.assertEqual(exec_values.get("Coverage rate"), "95.00% (95/100)")
        self.assertEqual(exec_values.get("Tag read redundancy"), "2.40×")
        self.assertEqual(exec_values.get("Antenna balance (σ)"), "3.20%")
        self.assertEqual(exec_values.get("RSSI stability index (σ)"), "1.25 dBm")
        self.assertAlmostEqual(
            float(exec_values.get("Global RSSI mean (dBm)")), -47.3, places=2
        )
        self.assertAlmostEqual(
            float(exec_values.get("Global RSSI std (dBm)")), 3.2, places=2
        )
        self.assertEqual(
            exec_values.get("RSSI noise indicator"),
            "Estabilidade de RSSI dentro do esperado (σ=3.20 dBm; 6.5 leituras/EPC)",
        )
        self.assertEqual(
            exec_values.get("Top performer antenna"),
            "3 (57.8% of reads), 120 reads",
        )

        structured_values = structured_sheet.astype(str).fillna("")
        flattened = structured_values.values.ravel().tolist()
        self.assertIn("Read hotspots", flattened)
        self.assertIn("Frequency (MHz)", flattened)
        self.assertIn("Expected EPC", flattened)
        self.assertIn("RSSI noise indicator", flattened)
        self.assertIn("Global RSSI mean (dBm)", flattened)

        positions_values = positions_sheet.astype(str).fillna("")
        self.assertIn("Participation (%)", positions_values.values)

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
            "rssi_noise_flag": True,
            "rssi_noise_indicator": "Variação elevada sem ganho de EPCs (σ=1.25 dBm; 15.0 leituras/EPC)",
            "rssi_noise_reads_per_epc": 15.0,
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
            str(exec_values["RSSI noise indicator"]),
            "Variação elevada sem ganho de EPCs (σ=1.25 dBm; 15.0 leituras/EPC)",
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
        self.assertIn("RSSI noise indicator", first_column)


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
