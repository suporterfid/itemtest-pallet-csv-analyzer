# -*- coding: utf-8 -*-
import re, io
from pathlib import Path
import pandas as pd

IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HEX_EPC_MIN24 = re.compile(r"^[0-9A-Fa-f]{24,}$")

def read_itemtest_csv(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Lê CSV exportado pelo Impinj ItemTest com cabeçalhos comentados começando por //
    Retorna (df, metadata_dict)
    """
    raw_lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = None
    meta_lines = []
    for i, line in enumerate(raw_lines):
        if line.strip().startswith("//"):
            meta_lines.append(line.strip()[2:].strip())
            if "Timestamp" in line and "EPC" in line and "Antenna" in line:
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError(f"Não foi possível localizar o cabeçalho de dados em {path}")
    # parse metadata key=value from meta_lines except the last header line
    metadata = {}
    for ml in meta_lines[:-1] if len(meta_lines) > 1 else meta_lines:
        parts = [p.strip() for p in ml.split(",") if p.strip()]
        for token in parts:
            if "=" in token:
                k, v = token.split("=", 1)
                metadata[k.strip()] = v.strip()
            else:
                metadata[token] = True

    # build CSV string removing the '//' of header
    csv_str = "\n".join(raw_lines[header_idx:])
    if csv_str.startswith("//"):
        csv_str = csv_str[2:].lstrip()
    df = pd.read_csv(io.StringIO(csv_str))

    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # coerce numeric
    for col in ["RSSI","Antenna","Frequency","PhaseAngle","DopplerFrequency","CRHandle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # clean EPCs: remove IP-like and keep only long hex EPCs
    if "EPC" not in df.columns:
        raise RuntimeError("Coluna EPC ausente no CSV.")
    df["EPC"] = df["EPC"].astype(str).str.strip()
    df = df[~df["EPC"].str.match(IPV4_RE, na=False)]
    df = df[df["EPC"].str.match(HEX_EPC_MIN24, na=False)]

    return df, metadata

def suffix3(epc: str) -> str|None:
    if not isinstance(epc, str):
        return None
    epc = epc.strip()
    return epc[-3:] if len(epc) >= 3 else epc
