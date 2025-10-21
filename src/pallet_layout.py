# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import pandas as pd
import re

def read_layout(path: str) -> pd.DataFrame:
    """
    Lê um arquivo de referência de layout do pallet (CSV, XLSX ou Markdown .md)
    e retorna um DataFrame com colunas: Linha, Traseira, Lateral_Esquerda, Lateral_Direita, Frente
    Cada célula pode conter múltiplos itens separados por '/', ',' ou ';'.
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
        raise ValueError(f"Formato de layout não suportado: {suffix}")

    # padronizar nomes de colunas
    rename_map = {
        "Linha":"Linha", "line":"Linha", "altura":"Linha",
        "Traseira":"Traseira", "Fundo":"Traseira", "Back":"Traseira",
        "Lateral_Esquerda":"Lateral_Esquerda", "Esquerda":"Lateral_Esquerda", "Left":"Lateral_Esquerda",
        "Lateral_Direita":"Lateral_Direita", "Direita":"Lateral_Direita", "Right":"Lateral_Direita",
        "Frente":"Frente", "Front":"Frente"
    }
    df.columns = [rename_map.get(c.strip(), c.strip()) for c in df.columns]
    needed = ["Linha","Traseira","Lateral_Esquerda","Lateral_Direita","Frente"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no layout: {missing}. Presentes: {list(df.columns)}")

    # normalizar células em listas de valores (sufixos ou EPCs completos)
    def split_vals(val):
        if pd.isna(val):
            return []
        s = str(val).strip()
        # separadores comuns
        parts = re.split(r"[\/,;]\s*|\s+\+\s+", s)
        # remover strings vazias
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    for col in ["Traseira","Lateral_Esquerda","Lateral_Direita","Frente"]:
        df[col] = df[col].apply(split_vals)

    # garantir ordenação da linha do topo para base? Mantemos conforme arquivo.
    return df

def build_expected_sets(df_layout: pd.DataFrame) -> dict:
    """
    A partir do layout, retorna dois conjuntos:
      - expected_suffixes: sufixos de 3 chars
      - expected_full: EPCs completos (hex longos)
    """
    import re
    suffixes=set(); full=set()
    HEX_EPC_MIN24 = re.compile(r"^[0-9A-Fa-f]{24,}$")
    for col in ["Traseira","Lateral_Esquerda","Lateral_Direita","Frente"]:
        for arr in df_layout[col]:
            for token in arr:
                t = token.strip()
                if HEX_EPC_MIN24.match(t):
                    full.add(t.upper())
                elif len(t)>=3:
                    suffixes.add(t[-3:].upper())
    return {"expected_suffixes": suffixes, "expected_full": full}

def map_position_by_suffix(df_layout: pd.DataFrame) -> dict[str, str]:
    """
    Retorna um dicionário: sufixo(3) -> posição "Face - Linha X"
    Se houver múltiplas entradas iguais, a última prevalece.
    """
    pos = {}
    for _,row in df_layout.iterrows():
        linha = str(row["Linha"]).strip()
        for face in ["Traseira","Lateral_Esquerda","Lateral_Direita","Frente"]:
            for token in row[face]:
                suf = token[-3:].upper() if len(token)>=3 else token.upper()
                pos[suf] = f"{face} - Linha {linha}"
    return pos
