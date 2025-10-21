# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import pandas as pd
import re

ROW_COLUMN = "Row"
FACE_COLUMNS = ["Rear", "Left_Side", "Right_Side", "Front"]


def read_layout(path: str) -> pd.DataFrame:
    """Read a pallet reference layout (CSV, XLSX or Markdown) into a DataFrame.

    The returned DataFrame contains the columns ``Row`` plus the face columns defined
    in :data:`FACE_COLUMNS`. Each cell may contain multiple EPC identifiers or
    suffix tokens separated by ``/``, ``,`` or ``;``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".csv",):
        df = pd.read_csv(p)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif suffix in (".md", ".markdown"):
        # tentativa simples: extrair linhas de tabela Markdown ou CSV-like
        txt = p.read_text(encoding="utf-8", errors="ignore")
        # tente encontrar bloco de tabela por linhas com separadores de |
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]
        if table_lines:
            # remover separadores de alinhamento
            clean = [ln for ln in table_lines if not set(ln.replace("|","").strip()).issubset(set("-: "))]
            # split por |
            rows = [[c.strip(" `") for c in ln.split("|")] for ln in clean]
            # remover colunas vazias iniciais/finais comuns
            rows = [r[1:-1] if r and r[0]=="" and r[-1]=="" else r for r in rows]
            header = rows[0]
            data = rows[1:]
            df = pd.DataFrame(data, columns=header)
        else:
            # fallback: tentar como CSV
            from io import StringIO
            df = pd.read_csv(StringIO(txt))
    else:
        raise ValueError(f"Unsupported layout format: {suffix}")

    # Normalise column names to English canonical values
    rename_map = {
        "Linha": ROW_COLUMN,
        "line": ROW_COLUMN,
        "linha": ROW_COLUMN,
        "altura": ROW_COLUMN,
        "Row": ROW_COLUMN,
        "row": ROW_COLUMN,
        "Traseira": "Rear",
        "Fundo": "Rear",
        "Back": "Rear",
        "Rear": "Rear",
        "Lateral_Esquerda": "Left_Side",
        "Esquerda": "Left_Side",
        "Left": "Left_Side",
        "Left Side": "Left_Side",
        "Lateral_Direita": "Right_Side",
        "Direita": "Right_Side",
        "Right": "Right_Side",
        "Right Side": "Right_Side",
        "Frente": "Front",
        "Front": "Front",
    }
    df.columns = [rename_map.get(c.strip(), c.strip()) for c in df.columns]
    required_columns = [ROW_COLUMN, *FACE_COLUMNS]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Layout file is missing required columns: {missing}. Present columns: {list(df.columns)}"
        )

    # Normalise cell contents into token lists (suffixes or full EPCs)
    def split_vals(val):
        if pd.isna(val):
            return []
        s = str(val).strip()
        parts = re.split(r"[\/,;]\s*|\s+\+\s+", s)
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    for col in FACE_COLUMNS:
        df[col] = df[col].apply(split_vals)

    return df

def build_expected_sets(df_layout: pd.DataFrame) -> dict:
    """Return expected EPC suffix and full-code sets derived from the layout."""

    suffixes: set[str] = set()
    full: set[str] = set()
    hex_pattern = re.compile(r"^[0-9A-Fa-f]{24,}$")
    for col in FACE_COLUMNS:
        for values in df_layout[col]:
            for token in values:
                t = token.strip()
                if hex_pattern.match(t):
                    full.add(t.upper())
                elif len(t) >= 3:
                    suffixes.add(t[-3:].upper())
    return {"expected_suffixes": suffixes, "expected_full": full}

def map_position_by_suffix(df_layout: pd.DataFrame) -> dict[str, str]:
    """Return a mapping from EPC suffix (3 chars) to layout position description."""

    positions: dict[str, str] = {}
    for _, row in df_layout.iterrows():
        row_label = str(row[ROW_COLUMN]).strip()
        for face in FACE_COLUMNS:
            for token in row[face]:
                suffix = token[-3:].upper() if len(token) >= 3 else token.upper()
                positions[suffix] = f"{face.replace('_', ' ')} - Row {row_label}"
    return positions


__all__ = ["ROW_COLUMN", "FACE_COLUMNS", "read_layout", "build_expected_sets", "map_position_by_suffix"]
