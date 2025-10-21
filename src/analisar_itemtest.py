# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

from .parser import read_itemtest_csv
from .metrics import summarize_by_epc, summarize_by_antenna
from .plots import plot_reads_by_epc, plot_reads_by_antenna, boxplot_rssi_by_antenna
from .report import write_excel
from .pallet_layout import read_layout, build_expected_sets, map_position_by_suffix

def process_file(csv_path: Path, out_dir: Path, layout_df: pd.DataFrame|None):
    df, metadata = read_itemtest_csv(str(csv_path))

    # métricas
    summary = summarize_by_epc(df)
    ant_counts = summarize_by_antenna(df)

    # se layout presente, anotar posição por sufixo
    positions_df = None
    if layout_df is not None:
        pos_map = map_position_by_suffix(layout_df)
        summary["Posicao_Pallet"] = summary["EPC_suffix3"].map(pos_map).fillna("—")
        # construir tabela de cobertura por posição
        positions = []
        for _,row in layout_df.iterrows():
            linha = str(row["Linha"]).strip()
            for face in ["Traseira","Lateral_Esquerda","Lateral_Direita","Frente"]:
                for token in row[face]:
                    suf = token[-3:].upper() if len(token)>=3 else token.upper()
                    found = summary.loc[summary["EPC_suffix3"]==suf, "total_reads"].sum()
                    positions.append({"Linha":linha,"Face":face,"Sufixo":suf,"Lido": bool(found),"Total_leituras": int(found)})
        positions_df = pd.DataFrame(positions)

    # identificar EPCs inesperados? Se layout ausente, pula. Se presente, usa sets.
    unexpected = pd.DataFrame()
    if layout_df is not None:
        sets = build_expected_sets(layout_df)
        expected_suf = sets["expected_suffixes"]
        expected_full = sets["expected_full"]
        def is_expected(row):
            epc = str(row["EPC"]).upper()
            suf = str(row["EPC_suffix3"]).upper()
            return (epc in expected_full) or (suf in expected_suf)
        summary["expected_suffix"] = summary.apply(is_expected, axis=1)
        unexpected = summary[~summary["expected_suffix"]].copy()

    # gráficos
    fig_dir = out_dir/"graficos"/csv_path.stem
    plot_reads_by_epc(summary, str(fig_dir), title=f"Leituras por EPC — {csv_path.name}")
    plot_reads_by_antenna(ant_counts, str(fig_dir), title=f"Leituras por Antena — {csv_path.name}")
    boxplot_rssi_by_antenna(df, str(fig_dir), title=f"RSSI por Antena — {csv_path.name}")

    # excel
    excel_out = out_dir/f"{csv_path.stem}_resultado.xlsx"
    write_excel(str(excel_out), summary, unexpected, ant_counts, metadata, positions_df=positions_df)
    return excel_out

def main():
    ap = argparse.ArgumentParser(description="Impinj ItemTest RFID Analyzer (com referência opcional de pallet)")
    ap.add_argument("--input", required=True, help="Pasta contendo CSVs do ItemTest")
    ap.add_argument("--output", required=True, help="Pasta para salvar resultados")
    ap.add_argument("--layout", required=False, help="Arquivo de layout do pallet (CSV/XLSX/MD)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    layout_df = None
    if args.layout:
        layout_df = read_layout(args.layout)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"Nenhum CSV encontrado em {in_dir}")
        sys.exit(1)

    results = []
    for f in csv_files:
        print(f"Processando {f.name} ...")
        res = process_file(f, out_dir, layout_df)
        results.append(res)
    print("Concluído. Arquivos gerados:")
    for r in results:
        print(" -", r)

if __name__ == "__main__":
    main()
