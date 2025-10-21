# -*- coding: utf-8 -*-
"""Testes para validar a geração do resumo textual."""

from pathlib import Path
import unittest

import pandas as pd

from src.analisar_itemtest import compose_summary_text


class TestComposeSummaryText(unittest.TestCase):
    """Garante que o resumo textual apresenta informações essenciais."""

    def test_summary_includes_hostname_from_metadata(self) -> None:
        metadata = {"Hostname": "192.168.68.100"}
        summary = pd.DataFrame(
            {
                "EPC": ["303132333435363738394142"],
                "total_reads": [12],
                "first_time": [pd.Timestamp("2025-01-01T10:00:00Z")],
                "last_time": [pd.Timestamp("2025-01-01T10:15:00Z")],
                "EPC_esperado": [True],
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


if __name__ == "__main__":
    unittest.main()
