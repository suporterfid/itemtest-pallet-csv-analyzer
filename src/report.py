# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from pathlib import Path

def write_excel(
    out_path: str,
    summary_epc: pd.DataFrame,
    unexpected: pd.DataFrame,
    ant_counts: pd.DataFrame,
    metadata: dict,
    positions_df: pd.DataFrame|None = None
):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary_epc.to_excel(writer, index=False, sheet_name="Resumo_por_EPC")
        unexpected_df = unexpected
        if unexpected_df is None:
            unexpected_df = pd.DataFrame(columns=summary_epc.columns)
        unexpected_df.to_excel(writer, index=False, sheet_name="EPCs_inesperados")
        ant_counts.to_excel(writer, index=False, sheet_name="Leituras_por_Antena")
        if positions_df is not None:
            positions_df.to_excel(writer, index=False, sheet_name="Posicoes_Pallet")
        if metadata:
            md_df = pd.DataFrame(list(metadata.items()), columns=["Chave","Valor"])
            md_df.to_excel(writer, index=False, sheet_name="Metadata")
