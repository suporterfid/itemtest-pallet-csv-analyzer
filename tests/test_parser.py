# -*- coding: utf-8 -*-
"""Regression tests for the ItemTest CSV parser."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.parser import read_itemtest_csv


class TestParserSample(unittest.TestCase):
    """Validate CSV parsing with locale-specific delimiters and decimals."""

    def test_sample_file_preserves_epc_and_metrics(self) -> None:
        sample_path = Path("samples/Sample_ItemTest.csv")
        df, _ = read_itemtest_csv(sample_path)

        # The sample file must produce valid EPC data (non-empty and hexadecimal).
        self.assertFalse(df.empty, "The DataFrame should not be empty.")
        self.assertTrue(df["EPC"].str.len().gt(0).all(), "There are empty EPCs.")
        self.assertTrue(
            df["EPC"].str.fullmatch(r"[0-9A-Fa-f]{24,}").all(),
            "There are EPCs with invalid formatting.",
        )

        # Core metrics must remain numeric and contain actual values.
        for column in ["RSSI", "Frequency", "PhaseAngle", "DopplerFrequency"]:
            self.assertIn(column, df.columns, f"Column {column} is missing from the DataFrame.")
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[column]),
                f"Column {column} is no longer numeric.",
            )

        self.assertTrue(
            df["RSSI"].notna().any(), "Column RSSI does not contain valid numeric values.",
        )
        self.assertTrue(
            df["Frequency"].notna().any(),
            "Column Frequency does not contain valid numeric values.",
        )

    def test_metadata_hostname_populated_from_reader_name(self) -> None:
        sample_path = Path("samples/Sample_ItemTest.csv")
        _, metadata = read_itemtest_csv(sample_path)

        self.assertEqual(
            metadata.get("Hostname"),
            "192.168.68.100",
            "Hostname should be populated using ReaderName as a fallback.",
        )

    def test_metadata_hostname_fallback_from_dataframe_column(self) -> None:
        csv_content = """// Automated test
// Timestamp, EPC, Antenna, Hostname
2025-01-01T00:00:00Z;303132333435363738394142;1;10.0.0.10
"""
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".csv", encoding="utf-8"
            ) as handle:
                handle.write(csv_content)
                tmp_path = Path(handle.name)

            _, metadata = read_itemtest_csv(tmp_path)

            self.assertEqual(
                metadata.get("Hostname"),
                "10.0.0.10",
                "Hostname should fall back to the CSV column value.",
            )
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
