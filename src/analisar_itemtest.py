# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

if __package__ in (None, ""):
    package_path = Path(__file__).resolve().parent
    project_root = package_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = package_path.name

from .parser import read_itemtest_csv
from .metrics import summarize_by_epc, summarize_by_antenna
from .plots import (
    plot_reads_by_epc,
    plot_reads_by_antenna,
    boxplot_rssi_by_antenna,
    plot_active_epcs_over_time,
    plot_antenna_heatmap,
)
from .report import write_excel
from .pallet_layout import read_layout, build_expected_sets, map_position_by_suffix
from .continuous_mode import analyze_continuous_flow

HEX_EPC_PATTERN = re.compile(r"^[0-9A-F]{24,}$", re.IGNORECASE)


def _configure_logging() -> tuple[logging.Logger, Path]:
    """Configure file and console logging for the CLI module."""

    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{date_str}_{Path(__file__).stem}.log"

    logger = logging.getLogger("itemtest.analisar")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


LOGGER, LOG_FILE_PATH = _configure_logging()


def _positive_float(value: str) -> float:
    """Return a positive float parsed from *value* or raise ``ArgumentTypeError``."""

    try:
        converted = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Valor da janela deve ser numérico em segundos."
        ) from exc
    if converted <= 0:
        raise argparse.ArgumentTypeError("Valor da janela deve ser positivo.")
    return converted


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


def _build_continuous_alerts(
    anomalous_epcs: Iterable[str] | None,
    inconsistency_flags: dict[str, Iterable[str]] | None,
) -> list[str]:
    """Return formatted alert strings for continuous-mode findings."""

    alerts: list[str] = []
    if anomalous_epcs:
        anomalous_list = [str(epc) for epc in anomalous_epcs if epc]
        if anomalous_list:
            sample = ", ".join(anomalous_list[:5])
            suffix = " ..." if len(anomalous_list) > 5 else ""
            alerts.append(
                f"EPCs com permanência atípica ({len(anomalous_list)}): {sample}{suffix}"
            )

    flag_labels = {
        "epcs_only_top_antennas": "EPCs concentrados apenas em antenas superiores",
        "epcs_sem_antena": "EPCs sem antena identificada",
    }
    for key, values in (inconsistency_flags or {}).items():
        if not values:
            continue
        formatted_values = [str(value) for value in values if value]
        if not formatted_values:
            continue
        sample = ", ".join(formatted_values[:5])
        suffix = " ..." if len(formatted_values) > 5 else ""
        label = flag_labels.get(key, str(key))
        alerts.append(f"{label} ({len(formatted_values)}): {sample}{suffix}")

    return alerts

def compose_summary_text(
    csv_path: Path,
    metadata: dict,
    summary: pd.DataFrame,
    ant_counts: pd.DataFrame,
    positions_df: pd.DataFrame | None,
    *,
    analysis_mode: str = "structured",
    continuous_details: dict | None = None,
) -> str:
    """Compose a human-readable summary with optional continuous-mode insights."""

    total_epcs = int(summary.shape[0]) if summary is not None else 0
    total_reads = int(summary["total_reads"].sum()) if not summary.empty else 0

    expected_count = None
    unexpected_count = None
    if "EPC_esperado" in summary.columns:
        expected_count = int(summary["EPC_esperado"].sum())
        unexpected_count = int((~summary["EPC_esperado"]).sum())

    def _format_timestamp(value) -> str | None:
        if value is None:
            return None
        try:
            ts = pd.to_datetime(value)
        except Exception:
            return str(value)
        if pd.isna(ts):
            return None
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        except (TypeError, AttributeError):
            pass
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    first_seen = (
        _format_timestamp(summary["first_time"].min())
        if "first_time" in summary.columns
        else None
    )
    last_seen = (
        _format_timestamp(summary["last_time"].max())
        if "last_time" in summary.columns
        else None
    )

    metadata_lines: list[str] = []
    hostname = metadata.get("Hostname")
    if hostname:
        metadata_lines.append(f"- Hostname: {hostname}")
    mode_index = metadata.get("ModeIndex")
    if mode_index is not None:
        metadata_lines.append(f"- ModeIndex: {mode_index}")
    session = metadata.get("Session")
    if session is not None:
        metadata_lines.append(f"- Sessão: {session}")
    inventory_mode = metadata.get("InventoryMode")
    if inventory_mode:
        metadata_lines.append(f"- InventoryMode: {inventory_mode}")
    antennas = metadata.get("AntennaIDs")
    if antennas:
        antenna_list = ", ".join(str(a) for a in antennas)
        metadata_lines.append(f"- Antenas declaradas: {antenna_list}")
    powers = metadata.get("PowersInDbm")
    if isinstance(powers, dict) and powers:
        formatted_powers = []
        for ant_id, power in sorted(powers.items()):
            if isinstance(power, (int, float)):
                formatted_powers.append(f"Antena {ant_id}: {power:.1f} dBm")
            else:
                formatted_powers.append(f"Antena {ant_id}: {power}")
        metadata_lines.append("- Potências declaradas: " + "; ".join(formatted_powers))
    elif powers:
        metadata_lines.append(f"- Potências declaradas: {powers}")
    if not metadata_lines:
        metadata_lines.append("- Nenhum metadado relevante encontrado.")

    mode_label = "Contínuo" if analysis_mode == "continuous" else "Estruturado"
    general_lines: list[str] = [f"- Modo de análise: {mode_label}"]
    if expected_count is not None and unexpected_count is not None:
        general_lines.append(
            f"- EPCs únicos: {total_epcs} (esperados: {expected_count}, inesperados: {unexpected_count})"
        )
    else:
        general_lines.append(f"- EPCs únicos: {total_epcs}")
    general_lines.append(f"- Leituras totais: {total_reads}")
    if first_seen and last_seen:
        general_lines.append(f"- Janela de leitura: {first_seen} → {last_seen}")
    elif first_seen:
        general_lines.append(f"- Primeira leitura registrada em {first_seen}")
    elif last_seen:
        general_lines.append(f"- Última leitura registrada em {last_seen}")

    continuous_lines: list[str] = []
    if analysis_mode == "continuous":
        details = continuous_details or {}
        average_dwell = details.get("average_dwell_seconds")
        if average_dwell is not None and not pd.isna(average_dwell):
            continuous_lines.append(
                f"- Tempo médio de permanência: {float(average_dwell):.2f} s"
            )
        total_events = details.get("total_events")
        if total_events is not None and not pd.isna(total_events):
            continuous_lines.append(
                f"- Eventos de entrada/saída detectados: {int(total_events)}"
            )
        dominant = details.get("dominant_antenna")
        if dominant is not None and not (
            isinstance(dominant, float) and pd.isna(dominant)
        ) and str(dominant) != "":
            try:
                dom_display = int(dominant)
            except (TypeError, ValueError):
                dom_display = dominant
            continuous_lines.append(f"- Antena dominante: {dom_display}")
        peak_value = details.get("epcs_per_minute_peak")
        if peak_value is not None and not pd.isna(peak_value):
            peak_time = details.get("epcs_per_minute_peak_time")
            if peak_time is not None:
                try:
                    peak_ts = pd.to_datetime(peak_time)
                    if pd.isna(peak_ts):
                        raise ValueError
                    peak_label = peak_ts.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    peak_label = str(peak_time)
                peak_repr = f"{int(peak_value)} às {peak_label}"
            else:
                peak_repr = str(int(peak_value))
            continuous_lines.append(
                f"- Pico de EPCs ativos/min: {peak_repr}"
            )
        alerts = details.get("alerts")
        if alerts is None:
            alerts = _build_continuous_alerts(
                details.get("anomalous_epcs"),
                details.get("inconsistency_flags"),
            )
        alerts = [alert for alert in alerts if alert]
        if alerts:
            for alert in alerts:
                continuous_lines.append(f"- Alerta: {alert}")
        else:
            continuous_lines.append("- Nenhum alerta identificado no modo contínuo.")

    antenna_lines: list[str] = []
    if ant_counts is not None and not ant_counts.empty:
        for row in ant_counts.itertuples(index=False):
            antenna_id = getattr(row, "Antenna", "?")
            reads = getattr(row, "total_reads", 0)
            line = f"- Antena {antenna_id}: {reads} leituras"
            participation = getattr(row, "participation_pct", None)
            if participation is not None and not pd.isna(participation):
                line += f" ({participation:.1f}%)"
            rssi_avg = getattr(row, "rssi_avg", None)
            if rssi_avg is not None and not pd.isna(rssi_avg):
                line += f", RSSI médio {rssi_avg:.1f} dBm"
            antenna_lines.append(line)
    else:
        antenna_lines.append("- Nenhuma leitura agregada por antena disponível.")

    layout_lines: list[str] = []
    if positions_df is None:
        layout_lines.append("- Layout não fornecido.")
    elif positions_df.empty:
        layout_lines.append("- Layout fornecido, mas sem posições definidas.")
    else:
        total_positions = int(len(positions_df))
        read_positions = int(positions_df["Lido"].sum())
        coverage_pct = (read_positions / total_positions * 100) if total_positions else 0.0
        layout_lines.append(
            f"- Cobertura do layout: {read_positions} de {total_positions} posições ({coverage_pct:.1f}%)"
        )
        missing = positions_df[~positions_df["Lido"]]
        if not missing.empty:
            missing_records = missing[["Face", "Linha", "Sufixo"]].drop_duplicates()
            descriptors: list[str] = []
            for row in missing_records.itertuples(index=False):
                face = getattr(row, "Face")
                line = getattr(row, "Linha")
                suffix = getattr(row, "Sufixo")
                descriptors.append(f"{face} - Linha {line} ({suffix})")
                if len(descriptors) == 5:
                    break
            extra = " ..." if len(missing_records) > 5 else ""
            layout_lines.append(
                f"- Posições sem leitura ({len(missing_records)}): " + "; ".join(descriptors) + extra
            )
        else:
            layout_lines.append("- Todas as posições do layout foram cobertas pelas leituras.")

    sections = [
        ("Metadados principais", metadata_lines),
        ("Indicadores gerais", general_lines),
    ]
    if analysis_mode == "continuous":
        if not continuous_lines:
            continuous_lines = ["- Nenhum indicador adicional disponível."]
        sections.append(("Indicadores modo contínuo", continuous_lines))
    sections.extend(
        [
            ("Leituras por antena", antenna_lines),
            ("Cobertura do layout", layout_lines),
        ]
    )

    header = f"Resumo ItemTest — {csv_path.name}"
    divider = "=" * len(header)
    lines: list[str] = [header, divider]
    for title, content in sections:
        lines.append(title + ":")
        lines.extend(content)
        lines.append("")
    return "\n".join(lines).rstrip()

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

    # resumo textual
    summary_text = compose_summary_text(
        csv_path,
        metadata,
        summary,
        ant_counts,
        positions_df,
        analysis_mode="structured",
    )
    LOGGER.info("\n%s", summary_text)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_log = log_dir / f"{csv_path.stem}_resumo.txt"
    summary_log.write_text(summary_text + "\n", encoding="utf-8")
    LOGGER.info("Resumo salvo em: %s", summary_log)

    # gráficos
    fig_dir = out_dir/"graficos"/csv_path.stem
    plot_reads_by_epc(summary, str(fig_dir), title=f"Leituras por EPC — {csv_path.name}")
    plot_reads_by_antenna(ant_counts, str(fig_dir), title=f"Leituras por Antena — {csv_path.name}")
    boxplot_rssi_by_antenna(df, str(fig_dir), title=f"RSSI por Antena — {csv_path.name}")

    # excel
    excel_out = out_dir/f"{csv_path.stem}_resultado.xlsx"
    write_excel(str(excel_out), summary, unexpected, ant_counts, metadata, positions_df=positions_df)
    return excel_out


def process_continuous_file(
    csv_path: Path,
    out_dir: Path,
    window_seconds: float,
    expected_registry: dict[str, set[str]] | None = None,
) -> Path:
    """Process a CSV file using the continuous flow analysis pipeline."""

    LOGGER.info("Processando (modo contínuo) %s ...", csv_path.name)
    df, metadata = read_itemtest_csv(str(csv_path))
    result = analyze_continuous_flow(df, window_seconds)

    summary = result.per_epc_summary.copy()
    if summary.empty:
        summary = pd.DataFrame(
            columns=[
                "EPC",
                "first_time",
                "last_time",
                "duration_present",
                "total_reads",
                "read_events",
                "antenna_distribution",
                "initial_antenna",
                "final_antenna",
                "direction_estimate",
            ]
        )
    if "EPC" not in summary.columns:
        summary["EPC"] = pd.Series(dtype=str)
    summary["EPC"] = summary["EPC"].astype(str)
    summary["EPC_suffix3"] = summary["EPC"].str[-3:].str.upper()
    if "total_reads" not in summary.columns:
        summary["total_reads"] = pd.Series(dtype=int)

    ant_counts = summarize_by_antenna(df)

    expected_suffixes: set[str] = set()
    expected_full: set[str] = set()
    if expected_registry:
        expected_suffixes.update(expected_registry.get("expected_suffixes", set()))
        expected_full.update(expected_registry.get("expected_full", set()))

    if expected_full or expected_suffixes:
        epc_upper = summary["EPC"].str.upper()
        suffix_upper = summary["EPC_suffix3"].str.upper()
        mask_expected = epc_upper.isin(expected_full) | suffix_upper.isin(expected_suffixes)
    else:
        mask_expected = pd.Series(True, index=summary.index)

    summary["EPC_esperado"] = mask_expected
    summary["Status_EPC"] = summary["EPC_esperado"].map({True: "Esperado", False: "Inesperado"})
    unexpected = summary.loc[~mask_expected].copy()

    dominant_antenna = None
    if ant_counts is not None and not ant_counts.empty:
        try:
            idx = ant_counts["total_reads"].astype(float).idxmax()
            candidate = ant_counts.loc[idx, "Antenna"]
            if pd.notna(candidate):
                try:
                    dominant_antenna = int(candidate)
                except (TypeError, ValueError):
                    dominant_antenna = candidate
        except ValueError:
            dominant_antenna = None

    total_events = int(result.epc_timeline.shape[0]) if not result.epc_timeline.empty else 0
    alerts = _build_continuous_alerts(result.anomalous_epcs, result.inconsistency_flags)

    peak_value = None
    peak_time = None
    if not result.epcs_per_minute.empty:
        peak_value = int(result.epcs_per_minute.max())
        peak_time = result.epcs_per_minute.idxmax()

    continuous_details: dict[str, object] = {
        "average_dwell_seconds": result.average_dwell_seconds,
        "total_events": total_events,
        "dominant_antenna": dominant_antenna,
        "alerts": alerts,
        "anomalous_epcs": result.anomalous_epcs,
        "inconsistency_flags": result.inconsistency_flags,
    }
    if peak_value is not None:
        continuous_details["epcs_per_minute_peak"] = peak_value
    if peak_time is not None:
        continuous_details["epcs_per_minute_peak_time"] = peak_time

    summary_text = compose_summary_text(
        csv_path,
        metadata,
        summary,
        ant_counts,
        positions_df=None,
        analysis_mode="continuous",
        continuous_details=continuous_details,
    )
    LOGGER.info("\n%s", summary_text)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_log = log_dir / f"{csv_path.stem}_resumo_continuo.txt"
    summary_log.write_text(summary_text + "\n", encoding="utf-8")
    LOGGER.info("Resumo contínuo salvo em: %s", summary_log)

    alerts_path = log_dir / f"{csv_path.stem}_alertas_continuo.txt"
    alerts_to_write = alerts if alerts else ["Nenhum alerta gerado."]
    alerts_path.write_text("\n".join(alerts_to_write) + "\n", encoding="utf-8")
    LOGGER.info("Alertas contínuos salvos em: %s", alerts_path)

    epcs_per_minute_path = log_dir / f"{csv_path.stem}_epcs_por_minuto.csv"
    epcs_per_minute_df = result.epcs_per_minute.rename_axis("minute").reset_index()
    if epcs_per_minute_df.empty:
        epcs_per_minute_df = pd.DataFrame(columns=["minute", "unique_epcs"])
    else:
        epcs_per_minute_df["minute"] = pd.to_datetime(
            epcs_per_minute_df["minute"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    epcs_per_minute_df.to_csv(epcs_per_minute_path, index=False, encoding="utf-8")
    LOGGER.info("EPCs/minuto registrados em: %s", epcs_per_minute_path)

    timeline_excel = result.epc_timeline.copy()
    timeline_log_path = log_dir / f"{csv_path.stem}_timeline_continuo.csv"
    timeline_log = timeline_excel.copy()
    if timeline_log.empty:
        timeline_log = pd.DataFrame(
            columns=[
                "EPC",
                "event_index",
                "entry_time",
                "exit_time",
                "duration_seconds",
                "read_count",
            ]
        )
    else:
        for col in ("entry_time", "exit_time"):
            if col in timeline_log.columns:
                timeline_log[col] = pd.to_datetime(
                    timeline_log[col], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S")
    timeline_log.to_csv(timeline_log_path, index=False, encoding="utf-8")
    LOGGER.info("Timeline contínua exportada em: %s", timeline_log_path)

    fig_dir = out_dir / "graficos" / f"{csv_path.stem}_continuo"
    plot_reads_by_epc(summary, str(fig_dir), title=f"Leituras por EPC — {csv_path.name} (contínuo)")
    plot_reads_by_antenna(
        ant_counts,
        str(fig_dir),
        title=f"Leituras por Antena — {csv_path.name} (contínuo)",
    )
    boxplot_rssi_by_antenna(
        df,
        str(fig_dir),
        title=f"RSSI por Antena — {csv_path.name} (contínuo)",
    )
    plot_active_epcs_over_time(
        result.epcs_per_minute,
        str(fig_dir),
        title=f"EPCs ativos ao longo do tempo — {csv_path.name}",
    )
    plot_antenna_heatmap(
        summary,
        str(fig_dir),
        title=f"Mapa de calor por antena — {csv_path.name}",
    )

    excel_out = out_dir / f"{csv_path.stem}_resultado_continuo.xlsx"
    write_excel(
        str(excel_out),
        summary,
        unexpected,
        ant_counts,
        metadata,
        positions_df=None,
        continuous_timeline=timeline_excel,
        continuous_metrics=continuous_details,
    )
    LOGGER.info("Relatório Excel contínuo salvo em: %s", excel_out)
    return excel_out

def main():
    ap = argparse.ArgumentParser(description="Impinj ItemTest RFID Analyzer (com referência opcional de pallet)")
    ap.add_argument("--input", required=True, help="Pasta contendo CSVs do ItemTest")
    ap.add_argument("--output", required=True, help="Pasta para salvar resultados")
    ap.add_argument("--layout", required=False, help="Arquivo de layout do pallet (CSV/XLSX/MD)")
    ap.add_argument("--expected", required=False, help="Arquivo ou lista de EPCs/sufixos esperados para uso sem layout")
    ap.add_argument(
        "--mode",
        choices=("structured", "continuous"),
        help="Força o modo de análise (padrão: estruturado com layout, contínuo sem layout)",
    )
    ap.add_argument(
        "--window",
        type=_positive_float,
        default=2.0,
        help="Janela de tempo em segundos para agrupamento no modo contínuo (default: 2.0)",
    )
    args = ap.parse_args()

    LOGGER.info("Arquivo de log configurado em: %s", LOG_FILE_PATH)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_path = Path(args.layout) if args.layout else None
    window_seconds = float(args.window)

    if args.mode:
        effective_mode = args.mode
        LOGGER.info("Modo selecionado via parâmetro: %s", effective_mode)
    else:
        effective_mode = "structured" if layout_path else "continuous"
        LOGGER.info(
            "Modo não informado explicitamente; inferido como %s com base no layout.",
            effective_mode,
        )

    LOGGER.info(
        "Layout: %s | Lista de EPCs esperados: %s | Janela modo contínuo: %.2f s",
        layout_path if layout_path else "(não fornecido)",
        args.expected if args.expected else "(não fornecida)",
        window_seconds,
    )

    layout_df = None
    if layout_path and effective_mode == "structured":
        layout_df = read_layout(str(layout_path))
    elif layout_path and effective_mode == "continuous":
        LOGGER.warning(
            "Layout fornecido (%s) não será utilizado no modo contínuo.",
            layout_path,
        )
    try:
        expected_registry = load_expected_tokens(args.expected)
    except Exception as exc:
        LOGGER.error("Erro ao carregar lista de EPCs esperados: %s", exc)
        sys.exit(1)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        LOGGER.error("Nenhum CSV encontrado em %s", in_dir)
        sys.exit(1)

    results: list[Path] = []
    if effective_mode == "continuous":
        for f in csv_files:
            try:
                res = process_continuous_file(
                    f,
                    out_dir,
                    window_seconds=window_seconds,
                    expected_registry=expected_registry,
                )
            except Exception as exc:  # pragma: no cover - capture informative tracebacks
                LOGGER.exception(
                    "Falha ao processar %s em modo contínuo: %s",
                    f.name,
                    exc,
                )
                sys.exit(1)
            results.append(res)
    else:
        for f in csv_files:
            LOGGER.info("Processando %s ...", f.name)
            res = process_file(f, out_dir, layout_df, expected_registry=expected_registry)
            results.append(res)
    LOGGER.info("Concluído. Arquivos gerados:")
    for r in results:
        LOGGER.info(" - %s", r)

if __name__ == "__main__":
    main()
