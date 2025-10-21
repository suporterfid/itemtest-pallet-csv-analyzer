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
from .metrics import summarize_by_epc, summarize_by_antenna, compile_structured_kpis
from .plots import (
    plot_reads_by_epc,
    plot_reads_by_antenna,
    boxplot_rssi_by_antenna,
    plot_active_epcs_over_time,
    plot_antenna_heatmap,
)
from .report import write_excel
from .pallet_layout import (
    ROW_COLUMN,
    FACE_COLUMNS,
    read_layout,
    build_expected_sets,
    map_position_by_suffix,
)
from .continuous_mode import analyze_continuous_flow
from .analisar_itemtest import compose_summary_text, build_continuous_alerts

HEX_EPC_PATTERN = re.compile(r"^[0-9A-F]{24,}$", re.IGNORECASE)


def _configure_logging() -> tuple[logging.Logger, Path]:
    """Configure file and console logging for the CLI module."""

    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{date_str}_{Path(__file__).stem}.log"

    logger = logging.getLogger("itemtest.analyzer")
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
            "Window size must be a numeric value in seconds."
        ) from exc
    if converted <= 0:
        raise argparse.ArgumentTypeError("Window size must be positive.")
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


def process_file(
    csv_path: Path,
    out_dir: Path,
    layout_df: pd.DataFrame | None,
    expected_registry: dict[str, set[str]] | None = None,
):
    df, metadata = read_itemtest_csv(str(csv_path))

    summary = summarize_by_epc(df)
    ant_counts = summarize_by_antenna(df)

    expected_suffixes: set[str] = set()
    expected_full: set[str] = set()
    if expected_registry:
        expected_suffixes.update(expected_registry.get("expected_suffixes", set()))
        expected_full.update(expected_registry.get("expected_full", set()))

    positions_df = None
    if layout_df is not None:
        position_map = map_position_by_suffix(layout_df)
        summary["pallet_position"] = summary["EPC_suffix3"].map(position_map).fillna("—")

        coverage_records: list[dict[str, object]] = []
        for _, row in layout_df.iterrows():
            row_label = str(row[ROW_COLUMN]).strip()
            for face in FACE_COLUMNS:
                face_label = face.replace("_", " ")
                for token in row[face]:
                    suffix = token[-3:].upper() if len(token) >= 3 else token.upper()
                    total_reads_for_suffix = summary.loc[
                        summary["EPC_suffix3"] == suffix, "total_reads"
                    ].sum()
                    coverage_records.append(
                        {
                            ROW_COLUMN: row_label,
                            "Face": face_label,
                            "Suffix": suffix,
                            "Read": bool(total_reads_for_suffix),
                            "total_reads": int(total_reads_for_suffix),
                        }
                    )
        positions_df = pd.DataFrame(coverage_records)

        sets = build_expected_sets(layout_df)
        expected_suffixes.update(sets["expected_suffixes"])
        expected_full.update(sets["expected_full"])

    epc_upper = summary["EPC"].astype(str).str.upper()
    suffix_upper = summary["EPC_suffix3"].astype(str).str.upper()
    if expected_full or expected_suffixes:
        mask_expected = epc_upper.isin(expected_full) | suffix_upper.isin(expected_suffixes)
    else:
        mask_expected = pd.Series(True, index=summary.index)
    summary["expected_epc"] = mask_expected
    summary["epc_status"] = summary["expected_epc"].map({True: "Expected", False: "Unexpected"})
    unexpected = summary[~mask_expected].copy()

    structured_metrics = compile_structured_kpis(
        summary,
        df,
        ant_counts,
        expected_full=expected_full,
        expected_suffixes=expected_suffixes,
        positions_df=positions_df,
    )

    summary_text = compose_summary_text(
        csv_path,
        metadata,
        summary,
        ant_counts,
        positions_df,
        analysis_mode="structured",
        structured_metrics=structured_metrics,
    )
    LOGGER.info("\n%s", summary_text)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_log = log_dir / f"{csv_path.stem}_summary.txt"
    summary_log.write_text(summary_text + "\n", encoding="utf-8")
    LOGGER.info("Summary saved to: %s", summary_log)

    figures_dir = out_dir / "graficos" / csv_path.stem
    plot_reads_by_epc(summary, str(figures_dir), title=f"Reads by EPC — {csv_path.name}")
    plot_reads_by_antenna(
        ant_counts,
        str(figures_dir),
        title=f"Reads by Antenna — {csv_path.name}",
    )
    boxplot_rssi_by_antenna(
        df,
        str(figures_dir),
        title=f"RSSI by Antenna — {csv_path.name}",
    )

    LOGGER.info("Gráficos salvos em: %s", figures_dir)

    excel_out = out_dir / f"{csv_path.stem}_result.xlsx"
    write_excel(
        str(excel_out),
        summary,
        unexpected,
        ant_counts,
        metadata,
        positions_df=positions_df,
        structured_metrics=structured_metrics,
    )
    return excel_out


def process_continuous_file(
    csv_path: Path,
    out_dir: Path,
    window_seconds: float,
    expected_registry: dict[str, set[str]] | None = None,
) -> Path:
    """Process a CSV file using the continuous flow analysis pipeline."""

    LOGGER.info("Processing (continuous mode) %s ...", csv_path.name)
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

    summary["expected_epc"] = mask_expected
    summary["epc_status"] = summary["expected_epc"].map({True: "Expected", False: "Unexpected"})
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
    alerts = build_continuous_alerts(result.anomalous_epcs, result.inconsistency_flags)

    peak_value = None
    peak_time = None
    mean_epcs = None
    if not result.epcs_per_minute.empty:
        peak_value = int(result.epcs_per_minute.max())
        peak_time = result.epcs_per_minute.idxmax()
        mean_epcs = float(result.epcs_per_minute.mean())

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
    if mean_epcs is not None:
        continuous_details["epcs_per_minute_mean"] = mean_epcs

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

    alerts_path = log_dir / f"{csv_path.stem}_alertas_continuos.txt"
    alerts_to_write = alerts if alerts else ["Nenhum alerta gerado."]
    alerts_path.write_text("\n".join(alerts_to_write) + "\n", encoding="utf-8")
    LOGGER.info("Alertas contínuos salvos em: %s", alerts_path)

    epcs_per_minute_path = log_dir / f"{csv_path.stem}_epcs_por_minuto.csv"
    epcs_per_minute_df = result.epcs_per_minute.rename_axis("minute").reset_index()
    epcs_per_minute_df.columns = ["minuto", "epcs_unicos"]
    if epcs_per_minute_df.empty:
        epcs_per_minute_df = pd.DataFrame(columns=["minuto", "epcs_unicos"])
    else:
        epcs_per_minute_df["minuto"] = pd.to_datetime(
            epcs_per_minute_df["minuto"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    epcs_per_minute_df.to_csv(epcs_per_minute_path, index=False, encoding="utf-8")
    LOGGER.info("EPCs por minuto registrados em: %s", epcs_per_minute_path)

    timeline_excel = result.epc_timeline.copy()
    timeline_log_path = log_dir / f"{csv_path.stem}_fluxo_continuo.csv"
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
    LOGGER.info("Fluxo contínuo exportado para: %s", timeline_log_path)

    fig_dir = out_dir / "graficos" / f"{csv_path.stem}_continuous"
    plot_reads_by_epc(summary, str(fig_dir), title=f"Reads by EPC — {csv_path.name} (continuous)")
    plot_reads_by_antenna(
        ant_counts,
        str(fig_dir),
        title=f"Reads by Antenna — {csv_path.name} (continuous)",
    )
    boxplot_rssi_by_antenna(
        df,
        str(fig_dir),
        title=f"RSSI by Antenna — {csv_path.name} (continuous)",
    )
    plot_active_epcs_over_time(
        result.epcs_per_minute,
        str(fig_dir),
        title=f"Active EPCs over time — {csv_path.name}",
    )
    plot_antenna_heatmap(
        summary,
        str(fig_dir),
        title=f"Antenna heatmap — {csv_path.name}",
    )

    LOGGER.info("Gráficos (modo contínuo) salvos em: %s", fig_dir)

    excel_out = out_dir / f"{csv_path.stem}_continuous_result.xlsx"
    write_excel(
        str(excel_out),
        summary,
        unexpected,
        ant_counts,
        metadata,
        positions_df=None,
        structured_metrics=None,
        continuous_timeline=timeline_excel,
        continuous_metrics=continuous_details,
        continuous_epcs_per_minute=result.epcs_per_minute,
    )
    LOGGER.info("Continuous Excel report saved to: %s", excel_out)
    return excel_out


def orchestrate_processing(
    csv_files: Iterable[Path],
    *,
    mode: str,
    out_dir: Path,
    layout_df: pd.DataFrame | None,
    expected_registry: dict[str, set[str]] | None,
    window_seconds: float,
) -> list[Path]:
    """Dispatch the CSV files to the appropriate processing pipeline."""

    normalized_mode = mode.lower()
    if normalized_mode not in {"structured", "continuous"}:
        raise ValueError(f"Unsupported analysis mode: {mode}")

    results: list[Path] = []
    if normalized_mode == "continuous":
        for csv_path in csv_files:
            try:
                result = process_continuous_file(
                    csv_path,
                    out_dir,
                    window_seconds=window_seconds,
                    expected_registry=expected_registry,
                )
            except Exception as exc:  # pragma: no cover - log then abort similarly to CLI flow
                LOGGER.exception(
                    "Failed to process %s in continuous mode: %s",
                    csv_path.name,
                    exc,
                )
                raise SystemExit(1) from exc
            results.append(result)
    else:
        for csv_path in csv_files:
            LOGGER.info("Processing %s ...", csv_path.name)
            result = process_file(
                csv_path,
                out_dir,
                layout_df,
                expected_registry=expected_registry,
            )
            results.append(result)

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Impinj ItemTest RFID Analyzer (with optional pallet reference)"
    )
    ap.add_argument("--input", required=True, help="Folder containing ItemTest CSV exports")
    ap.add_argument("--output", required=True, help="Folder where the results will be stored")
    ap.add_argument(
        "--layout",
        required=False,
        help="Pallet layout file (CSV/XLSX/MD)",
    )
    ap.add_argument(
        "--expected",
        required=False,
        help="File path or inline list with expected EPCs/suffixes for operation without layout",
    )
    ap.add_argument(
        "--mode",
        choices=("structured", "continuous"),
        help="Forces the analysis mode (default: structured with layout, continuous without layout)",
    )
    ap.add_argument(
        "--window",
        type=_positive_float,
        default=2.0,
        help="Window size in seconds for continuous mode grouping (default: 2.0)",
    )
    args = ap.parse_args()

    LOGGER.info("Log file configured at: %s", LOG_FILE_PATH)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout_path = Path(args.layout) if args.layout else None
    window_seconds = float(args.window)

    if args.mode:
        effective_mode = args.mode
        LOGGER.info("Mode selected via CLI parameter: %s", effective_mode)
    else:
        effective_mode = "structured" if layout_path else "continuous"
        LOGGER.info(
            "Mode not explicitly provided; inferred as %s based on layout availability.",
            effective_mode,
        )

    LOGGER.info(
        "Parameters summary → input: %s | output: %s | mode: %s | window: %.2fs | layout: %s | expected list: %s",
        str(in_dir),
        str(out_dir),
        effective_mode,
        window_seconds,
        str(layout_path) if layout_path else "(not provided)",
        args.expected if args.expected else "(not provided)",
    )

    layout_df = None
    if layout_path and effective_mode == "structured":
        layout_df = read_layout(str(layout_path))
    elif layout_path and effective_mode == "continuous":
        LOGGER.warning(
            "Provided layout (%s) will not be used in continuous mode.",
            layout_path,
        )
    try:
        expected_registry = load_expected_tokens(args.expected)
    except Exception as exc:
        LOGGER.error("Failed to load expected EPC list: %s", exc)
        sys.exit(1)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        LOGGER.error("No CSV files found in %s", in_dir)
        sys.exit(1)

    results = orchestrate_processing(
        csv_files,
        mode=effective_mode,
        out_dir=out_dir,
        layout_df=layout_df,
        expected_registry=expected_registry,
        window_seconds=window_seconds,
    )
    LOGGER.info("Completed. Generated files:")
    for r in results:
        LOGGER.info(" - %s", r)

if __name__ == "__main__":
    main()
