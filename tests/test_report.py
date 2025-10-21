# -*- coding: utf-8 -*-
"""Tests ensuring the metadata worksheet preserves the hostname entry."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.report import write_excel


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
