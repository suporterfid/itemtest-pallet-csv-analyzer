# -*- coding: utf-8 -*-
import re, io
from pathlib import Path
import pandas as pd

IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HEX_EPC_MIN24 = re.compile(r"^[0-9A-Fa-f]{24,}$")

HOSTNAME_ALIASES = (
    "ReaderName",
    "ReaderHostname",
    "Reader Hostname",
    "Reader Host",
    "ReaderAddress",
    "Reader Address",
    "ReaderIP",
    "Reader IP",
    "Reader",
)

def read_itemtest_csv(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Lê CSV exportado pelo Impinj ItemTest com cabeçalhos comentados começando por //
    Retorna (df, metadata_dict)
    """
    raw_lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = None
    meta_lines: list[str] = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        is_comment = stripped.startswith("//")
        content = stripped[2:].strip() if is_comment else stripped
        if is_comment or ("=" in content and header_idx is None):
            if content:
                meta_lines.append(content)
        if "Timestamp" in content and "EPC" in content and "Antenna" in content:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Não foi possível localizar o cabeçalho de dados em {path}")
    # parse metadata key=value from meta_lines except the last header line
    if meta_lines and "Timestamp" in meta_lines[-1] and "EPC" in meta_lines[-1]:
        meta_lines = meta_lines[:-1]

    parsed_pairs: dict[str, str] = {}
    for ml in meta_lines:
        if not ml:
            continue
        tokens = [token.strip() for token in ml.split(",") if token.strip()]
        current_key: str | None = None
        current_parts: list[str] = []
        for token in tokens:
            if "=" in token:
                key_candidate, value_part = token.split("=", 1)
                normalized_key = key_candidate.strip()
                if normalized_key and any(ch.isalpha() for ch in normalized_key):
                    if current_key is not None:
                        parsed_pairs[current_key] = ",".join(part for part in current_parts if part)
                    current_key = normalized_key
                    current_parts = [value_part.strip()]
                    continue
            if current_key is not None:
                current_parts.append(token)
        if current_key is not None:
            parsed_pairs[current_key] = ",".join(part for part in current_parts if part)
            current_key = None

    metadata: dict[str, object] = {k: v for k, v in parsed_pairs.items()}

    if "AntennaIDs" in parsed_pairs:
        antennas = [int(match) for match in re.findall(r"-?\d+", parsed_pairs["AntennaIDs"])]
        metadata["AntennaIDs"] = antennas

    if "ModeIndex" in parsed_pairs:
        mode_value = parsed_pairs["ModeIndex"].strip()
        try:
            metadata["ModeIndex"] = int(mode_value)
        except ValueError:
            metadata["ModeIndex"] = mode_value

    if "Session" in parsed_pairs:
        session_value = parsed_pairs["Session"].strip()
        try:
            metadata["Session"] = int(session_value)
        except ValueError:
            metadata["Session"] = session_value

    if "InventoryMode" in parsed_pairs:
        metadata["InventoryMode"] = parsed_pairs["InventoryMode"].strip()

    if "Hostname" in parsed_pairs:
        metadata["Hostname"] = parsed_pairs["Hostname"].strip()

    if "PowersInDbm" in parsed_pairs:
        power_pairs: dict[int, float] = {}
        power_pattern = re.compile(
            r"(-?\d+)\s*=>\s*([-+]?\d+(?:[.,]\d+)?)(?=(?:\s*,\s*-?\d+\s*=>)|$)"
        )
        for ant, power in power_pattern.findall(parsed_pairs["PowersInDbm"]):
            try:
                antenna_id = int(ant)
                power_value = float(power.replace(",", "."))
            except ValueError:
                continue
            power_pairs[antenna_id] = power_value
        if power_pairs:
            metadata["PowersInDbm"] = power_pairs
        else:
            metadata["PowersInDbm"] = parsed_pairs["PowersInDbm"].strip()

    # build CSV string removing the '//' of header
    csv_lines = raw_lines[header_idx:]
    if not csv_lines:
        raise RuntimeError("Arquivo CSV não possui linhas de dados após o cabeçalho.")

    header_line = csv_lines[0]
    if header_line.startswith("//"):
        header_line = header_line[2:].lstrip()

    # o arquivo de exemplo mistura cabeçalho com vírgulas e dados com ponto e vírgula;
    # alinhe o cabeçalho ao delimitador detectado nas primeiras linhas de dados
    data_line = next(
        (line for line in csv_lines[1:] if line and not line.lstrip().startswith("//")),
        "",
    )
    if ";" in data_line and ";" not in header_line:
        normalized_header = header_line.replace(", ", ";").replace(",", ";")
    else:
        normalized_header = header_line

    csv_str = "\n".join([normalized_header, *csv_lines[1:]])

    # detect delimiter automatically while supporting decimal commas from ItemTest exports
    read_buffer = io.StringIO(csv_str)
    try:
        df = pd.read_csv(read_buffer, sep=None, engine="python", decimal=",")
    except (pd.errors.ParserError, ValueError):
        # fall back to semicolon-separated parsing (common in PT-BR locale exports)
        df = pd.read_csv(io.StringIO(csv_str), sep=";", engine="python", decimal=",")

    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # coerce numeric
    for col in ["RSSI", "Antenna", "Frequency", "PhaseAngle", "DopplerFrequency", "CRHandle"]:
        if col in df.columns:
            series = df[col]
            if series.dtype == object:
                series = series.astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # clean EPCs: remove IP-like and keep only long hex EPCs
    if "EPC" not in df.columns:
        raise RuntimeError("Coluna EPC ausente no CSV.")
    df["EPC"] = df["EPC"].astype(str).str.strip()
    df = df[~df["EPC"].str.match(IPV4_RE, na=False)]
    df = df[df["EPC"].str.match(HEX_EPC_MIN24, na=False)]

    def _normalize_hostname(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    hostname_value = _normalize_hostname(metadata.get("Hostname"))
    if hostname_value:
        metadata["Hostname"] = hostname_value
    else:
        metadata.pop("Hostname", None)
        for alias in HOSTNAME_ALIASES:
            alias_candidate = _normalize_hostname(parsed_pairs.get(alias))
            if alias_candidate:
                metadata["Hostname"] = alias_candidate
                hostname_value = alias_candidate
                break
    if not hostname_value and "Hostname" in df.columns:
        for raw_value in df["Hostname"]:
            candidate = _normalize_hostname(raw_value)
            if candidate:
                metadata["Hostname"] = candidate
                hostname_value = candidate
                break

    return df, metadata

def suffix3(epc: str) -> str|None:
    if not isinstance(epc, str):
        return None
    epc = epc.strip()
    return epc[-3:] if len(epc) >= 3 else epc
