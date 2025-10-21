# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
import re
from typing import Iterable
import pandas as pd

from .parser import read_itemtest_csv
from .metrics import summarize_by_epc, summarize_by_antenna
from .plots import plot_reads_by_epc, plot_reads_by_antenna, boxplot_rssi_by_antenna
from .report import write_excel
from .pallet_layout import read_layout, build_expected_sets, map_position_by_suffix

HEX_EPC_PATTERN = re.compile(r"^[0-9A-F]{24,}$", re.IGNORECASE)

def _tokenize_expected_source(text: str) -> list[str]:
    """Split raw text into individual EPC or suffix tokens."""
    tokens: list[str] = []
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        for part in re.split(r"[\s,;]+", clean):
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens

def _build_expected_sets(tokens: Iterable[str]) -> dict[str, set[str]]:
    """Return normalized sets of expected EPCs (full and suffix) from tokens."""
    suffixes: set[str] = set()
    full: set[str] = set()
    for token in tokens:
        cleaned = token.strip()
        if not cleaned:
            continue
        upper_token = cleaned.upper()
        if HEX_EPC_PATTERN.match(upper_token):
            full.add(upper_token)
        elif len(upper_token) >= 3:
            suffixes.add(upper_token[-3:])
    return {"expected_suffixes": suffixes, "expected_full": full}

def load_expected_tokens(source: str | None) -> dict[str, set[str]]:
    """Load expected EPC/suffix tokens from a file path or inline string."""
    if not source:
        return {"expected_suffixes": set(), "expected_full": set()}
    candidate = Path(source)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        tokens = _tokenize_expected_source(text)
    else:
        tokens = _tokenize_expected_source(source)
    return _build_expected_sets(tokens)

def process_file(
    csv_path: Path,
    out_dir: Path,
    layout_df: pd.DataFrame | None,
    expected_registry: dict[str, set[str]] | None = None,
):
    df, metadata = read_itemtest_csv(str(csv_path))

    # métricas
    summary = summarize_by_epc(df)
    ant_counts = summarize_by_antenna(df)

    expected_suffixes: set[str] = set()
    expected_full: set[str] = set()
    if expected_registry:
        expected_suffixes.update(expected_registry.get("expected_suffixes", set()))
        expected_full.update(expected_registry.get("expected_full", set()))

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

        sets = build_expected_sets(layout_df)
        expected_suffixes.update(sets["expected_suffixes"])
        expected_full.update(sets["expected_full"])

    # identificar EPCs esperados/inesperados combinando layout e lista configurada
    epc_upper = summary["EPC"].astype(str).str.upper()
    suffix_upper = summary["EPC_suffix3"].astype(str).str.upper()
    if expected_full or expected_suffixes:
        mask_expected = epc_upper.isin(expected_full) | suffix_upper.isin(expected_suffixes)
    else:
        mask_expected = pd.Series(True, index=summary.index)
    summary["EPC_esperado"] = mask_expected
    summary["Status_EPC"] = summary["EPC_esperado"].map({True: "Esperado", False: "Inesperado"})
    unexpected = summary[~mask_expected].copy()

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
    ap.add_argument("--expected", required=False, help="Arquivo ou lista de EPCs/sufixos esperados para uso sem layout")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    layout_df = None
    if args.layout:
        layout_df = read_layout(args.layout)
    try:
        expected_registry = load_expected_tokens(args.expected)
    except Exception as exc:
        print(f"Erro ao carregar lista de EPCs esperados: {exc}")
        sys.exit(1)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"Nenhum CSV encontrado em {in_dir}")
        sys.exit(1)

    results = []
    for f in csv_files:
        print(f"Processando {f.name} ...")
        res = process_file(f, out_dir, layout_df, expected_registry=expected_registry)
        results.append(res)
    print("Concluído. Arquivos gerados:")
    for r in results:
        print(" -", r)

if __name__ == "__main__":
    main()
