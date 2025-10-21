# -*- coding: utf-8 -*-
"""Testes de regressão para o parser de CSV do ItemTest."""

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
