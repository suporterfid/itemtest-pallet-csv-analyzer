# -*- coding: utf-8 -*-
"""Testes para garantir que a planilha de metadata preserve o hostname."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.report import write_excel


class TestReportMetadataSheet(unittest.TestCase):
    """Valida que o Excel gerado apresenta os metadados essenciais."""

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
            out_path = Path(tmp_dir) / "relatorio.xlsx"
            write_excel(str(out_path), summary, unexpected, ant_counts, metadata)

            md_df = pd.read_excel(out_path, sheet_name="Metadata")

        mask = (md_df["Chave"] == "Hostname") & (md_df["Valor"] == "192.168.68.100")
        self.assertTrue(mask.any(), "Planilha Metadata não listou o Hostname esperado.")


if __name__ == "__main__":
    unittest.main()
