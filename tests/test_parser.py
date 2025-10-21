# -*- coding: utf-8 -*-
"""Testes de regressão para o parser de CSV do ItemTest."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.parser import read_itemtest_csv


class TestParserSample(unittest.TestCase):
    """Valida parsing de CSV com delimitador e decimal locais."""

    def test_sample_file_preserves_epc_and_metrics(self) -> None:
        sample_path = Path("samples/Teste_Exemplo_ItemTest.csv")
        df, _ = read_itemtest_csv(sample_path)

        # O arquivo deve gerar dados de EPC válidos (não vazios e hexadecimais).
        self.assertFalse(df.empty, "O DataFrame não deveria estar vazio.")
        self.assertTrue(df["EPC"].str.len().gt(0).all(), "Existem EPCs vazios.")
        self.assertTrue(
            df["EPC"].str.fullmatch(r"[0-9A-Fa-f]{24,}").all(),
            "Há EPCs com formato inválido.",
        )

        # As métricas principais devem ser numéricas e conter dados reais.
        for column in ["RSSI", "Frequency", "PhaseAngle", "DopplerFrequency"]:
            self.assertIn(column, df.columns, f"Coluna {column} ausente no DataFrame.")
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[column]),
                f"Coluna {column} deixou de ser numérica.",
            )

        self.assertTrue(
            df["RSSI"].notna().any(), "Coluna RSSI não contém valores numéricos válidos.",
        )
        self.assertTrue(
            df["Frequency"].notna().any(),
            "Coluna Frequency não contém valores numéricos válidos.",
        )

    def test_metadata_hostname_populated_from_reader_name(self) -> None:
        sample_path = Path("samples/Teste_Exemplo_ItemTest.csv")
        _, metadata = read_itemtest_csv(sample_path)

        self.assertEqual(
            metadata.get("Hostname"),
            "192.168.68.100",
            "Hostname deveria aproveitar ReaderName como fallback.",
        )

    def test_metadata_hostname_fallback_from_dataframe_column(self) -> None:
        csv_content = """// Teste automático
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
                "Hostname deveria ser preenchido a partir da coluna do CSV.",
            )
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
