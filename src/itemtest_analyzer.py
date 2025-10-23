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

SUMMARY_PER_FILE_SHEET = "Detalhes_Por_Arquivo"
SUMMARY_OVERVIEW_SHEET = "Resumo_Geral"
SUMMARY_FILE_NAME = "executive_summary.xlsx"

if __package__ in (None, ""):
    package_path = Path(__file__).resolve().parent
    project_root = package_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = package_path.name

from .parser import read_itemtest_csv
from .metrics import (
    summarize_by_epc,
    summarize_by_antenna,
    compile_structured_kpis,
)
from .plots import (
    plot_reads_by_epc,
    plot_reads_by_antenna,
    boxplot_rssi_by_antenna,
    plot_rssi_vs_frequency,
    plot_active_epcs_over_time,
    plot_antenna_heatmap,
    plot_pallet_heatmap,
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


def _clean_float(value: object) -> float | None:
    """Return ``value`` as ``float`` or ``None`` when it cannot be converted."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _clean_int(value: object) -> int | None:
    """Return ``value`` as ``int`` or ``None`` when conversion is not possible."""

    if value is None:
        return None
    if isinstance(value, (int,)):
        return int(value)
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_timestamp_str(value: object) -> str | None:
    """Return a human-readable timestamp string or ``None`` when unavailable."""

    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return str(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _format_top_performer_label(info: object) -> str | None:
    """Return a descriptive label for the structured-mode top performer."""

    if not isinstance(info, dict):
        return None
    antenna = info.get("antenna")
    if antenna is None:
        return None
    try:
        antenna_label = str(int(antenna))
    except (TypeError, ValueError):
        antenna_label = str(antenna)
    participation = info.get("participation_pct")
    if participation is not None and not pd.isna(participation):
        antenna_label += f" ({float(participation):.1f}% of reads)"
    total_reads = info.get("total_reads")
    if total_reads is not None and not pd.isna(total_reads):
        try:
            total_reads_int = int(total_reads)
        except (TypeError, ValueError):
            total_reads_int = None
        if total_reads_int is not None:
            antenna_label += f", {total_reads_int} reads"
    return antenna_label


SUMMARY_COLUMN_ORDER = [
    "file",
    "mode",
    "hostname",
    "layout_used",
    "total_epcs",
    "total_reads",
    "expected_detected",
    "unexpected_detected",
    "coverage_rate",
    "expected_total",
    "expected_found",
    "tag_read_redundancy",
    "antenna_balance",
    "rssi_stability_index",
    "top_performer",
    "average_dwell_seconds",
    "throughput_per_minute",
    "session_throughput",
    "read_continuity_rate",
    "session_duration_seconds",
    "session_active_seconds",
    "tag_dwell_time_max",
    "concurrency_peak",
    "concurrency_average",
    "concurrency_peak_time",
    "dominant_antenna",
    "inactive_periods_count",
    "inactive_total_seconds",
    "inactive_longest_seconds",
    "congestion_index",
    "global_rssi_avg",
    "global_rssi_std",
    "rssi_noise_flag",
    "rssi_noise_indicator",
    "rssi_noise_reads_per_epc",
    "alerts_count",
    "analysis_window_seconds",
    "layout_total_positions",
    "layout_read_positions",
    "layout_overall_coverage",
    "epcs_per_minute_mean",
    "epcs_per_minute_peak",
    "epcs_per_minute_peak_time",
    "first_read",
    "last_read",
    "excel_report",
    "summary_log",
]


def _register_structured_summary(
    summary_records: list[dict[str, object]] | None,
    *,
    csv_path: Path,
    excel_out: Path,
    summary_df: pd.DataFrame,
    metadata: dict[str, object],
    structured_metrics: dict[str, object] | None,
    summary_log: Path,
    layout_applied: bool,
) -> None:
    """Append structured-mode metrics to the consolidated summary registry."""

    if summary_records is None:
        return

    record: dict[str, object] = {
        "file": csv_path.name,
        "mode": "structured",
        "hostname": metadata.get("Hostname"),
        "layout_used": bool(layout_applied),
        "excel_report": str(excel_out),
        "summary_log": str(summary_log),
    }

    if summary_df is None or summary_df.empty:
        record.update(
            {
                "total_epcs": 0,
                "total_reads": 0,
                "expected_detected": 0,
                "unexpected_detected": 0,
                "first_read": None,
                "last_read": None,
            }
        )
    else:
        total_epcs = None
        if "EPC" in summary_df.columns:
            try:
                total_epcs = int(summary_df["EPC"].nunique())
            except Exception:
                total_epcs = None
        if total_epcs is None:
            total_epcs = int(summary_df.shape[0])
        record["total_epcs"] = total_epcs
        if "total_reads" in summary_df.columns:
            try:
                record["total_reads"] = int(summary_df["total_reads"].sum())
            except Exception:
                record["total_reads"] = None
        expected_series = summary_df.get("expected_epc")
        if expected_series is not None:
            expected_bool = expected_series.fillna(False).astype(bool)
            record["expected_detected"] = int(expected_bool.sum())
            record["unexpected_detected"] = int((~expected_bool).sum())
        record["first_read"] = _format_timestamp_str(summary_df.get("first_time", pd.Series(dtype="datetime64[ns]")).min())
        record["last_read"] = _format_timestamp_str(summary_df.get("last_time", pd.Series(dtype="datetime64[ns]")).max())

    metrics = structured_metrics or {}
    record["coverage_rate"] = _clean_float(metrics.get("coverage_rate"))
    record["expected_total"] = _clean_int(metrics.get("expected_total"))
    record["expected_found"] = _clean_int(metrics.get("expected_found"))
    record["tag_read_redundancy"] = _clean_float(metrics.get("tag_read_redundancy"))
    record["antenna_balance"] = _clean_float(metrics.get("antenna_balance"))
    record["rssi_stability_index"] = _clean_float(metrics.get("rssi_stability_index"))
    record["global_rssi_avg"] = _clean_float(metrics.get("global_rssi_avg"))
    record["global_rssi_std"] = _clean_float(metrics.get("global_rssi_std"))
    record["rssi_noise_flag"] = metrics.get("rssi_noise_flag")
    record["rssi_noise_indicator"] = metrics.get("rssi_noise_indicator")
    record["rssi_noise_reads_per_epc"] = _clean_float(
        metrics.get("rssi_noise_reads_per_epc")
    )
    record["layout_total_positions"] = _clean_int(metrics.get("layout_total_positions"))
    record["layout_read_positions"] = _clean_int(metrics.get("layout_read_positions"))
    record["layout_overall_coverage"] = _clean_float(metrics.get("layout_overall_coverage"))
    record["top_performer"] = _format_top_performer_label(metrics.get("top_performer_antenna"))

    summary_records.append(record)


def _register_continuous_summary(
    summary_records: list[dict[str, object]] | None,
    *,
    csv_path: Path,
    excel_out: Path,
    summary_df: pd.DataFrame,
    metadata: dict[str, object],
    continuous_details: dict[str, object],
    summary_log: Path,
    alerts: list[str] | None,
    window_seconds: float,
) -> None:
    """Append continuous-mode metrics to the consolidated summary registry."""

    if summary_records is None:
        return

    record: dict[str, object] = {
        "file": csv_path.name,
        "mode": "continuous",
        "hostname": metadata.get("Hostname"),
        "layout_used": False,
        "excel_report": str(excel_out),
        "summary_log": str(summary_log),
        "analysis_window_seconds": _clean_float(window_seconds),
    }

    if summary_df is None or summary_df.empty:
        record["total_epcs"] = 0
        record["total_reads"] = 0
        record["expected_detected"] = 0
        record["unexpected_detected"] = 0
        record["first_read"] = _format_timestamp_str(continuous_details.get("session_start"))
        record["last_read"] = _format_timestamp_str(continuous_details.get("session_end_with_grace") or continuous_details.get("session_end"))
    else:
        if "EPC" in summary_df.columns:
            try:
                record["total_epcs"] = int(summary_df["EPC"].nunique())
            except Exception:
                record["total_epcs"] = int(summary_df.shape[0])
        else:
            record["total_epcs"] = int(summary_df.shape[0])
        if "total_reads" in summary_df.columns:
            try:
                record["total_reads"] = int(summary_df["total_reads"].sum())
            except Exception:
                record["total_reads"] = None
        expected_series = summary_df.get("expected_epc")
        if expected_series is not None:
            expected_bool = expected_series.fillna(False).astype(bool)
            record["expected_detected"] = int(expected_bool.sum())
            record["unexpected_detected"] = int((~expected_bool).sum())
        record["first_read"] = _format_timestamp_str(summary_df.get("first_time", pd.Series(dtype="datetime64[ns]")).min())
        record["last_read"] = _format_timestamp_str(summary_df.get("last_time", pd.Series(dtype="datetime64[ns]")).max())

    record["average_dwell_seconds"] = _clean_float(continuous_details.get("average_dwell_seconds"))
    record["throughput_per_minute"] = _clean_float(continuous_details.get("throughput_per_minute"))
    record["session_throughput"] = _clean_float(continuous_details.get("session_throughput"))
    record["read_continuity_rate"] = _clean_float(continuous_details.get("read_continuity_rate"))
    record["session_duration_seconds"] = _clean_float(continuous_details.get("session_duration_seconds"))
    record["session_active_seconds"] = _clean_float(continuous_details.get("session_active_seconds"))
    record["tag_dwell_time_max"] = _clean_float(continuous_details.get("tag_dwell_time_max"))
    record["concurrency_peak"] = _clean_int(continuous_details.get("concurrency_peak"))
    record["concurrency_average"] = _clean_float(continuous_details.get("concurrency_average"))
    record["concurrency_peak_time"] = _format_timestamp_str(continuous_details.get("concurrency_peak_time"))
    record["dominant_antenna"] = _clean_int(continuous_details.get("dominant_antenna"))
    record["inactive_periods_count"] = _clean_int(continuous_details.get("inactive_periods_count"))
    record["inactive_total_seconds"] = _clean_float(continuous_details.get("inactive_total_seconds"))
    record["inactive_longest_seconds"] = _clean_float(continuous_details.get("inactive_longest_seconds"))
    record["congestion_index"] = _clean_float(continuous_details.get("congestion_index"))
    record["global_rssi_avg"] = _clean_float(continuous_details.get("global_rssi_avg"))
    record["global_rssi_std"] = _clean_float(continuous_details.get("global_rssi_std"))
    record["rssi_noise_flag"] = continuous_details.get("rssi_noise_flag")
    record["rssi_noise_indicator"] = continuous_details.get("rssi_noise_indicator")
    record["rssi_noise_reads_per_epc"] = _clean_float(
        continuous_details.get("rssi_noise_reads_per_epc")
    )
    record["alerts_count"] = len(alerts or [])
    record["epcs_per_minute_mean"] = _clean_float(continuous_details.get("epcs_per_minute_mean"))
    record["epcs_per_minute_peak"] = _clean_int(continuous_details.get("epcs_per_minute_peak"))
    record["epcs_per_minute_peak_time"] = _format_timestamp_str(
        continuous_details.get("epcs_per_minute_peak_time")
    )

    summary_records.append(record)


def _build_overview_table(per_file_df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated overview metrics grouped by analysis mode."""

    if per_file_df.empty:
        return pd.DataFrame(columns=["mode", "files"])

    agg_spec: dict[str, tuple[str, str]] = {"files": ("file", "count")}
    if "total_epcs" in per_file_df.columns:
        agg_spec["total_epcs"] = ("total_epcs", "sum")
    if "total_reads" in per_file_df.columns:
        agg_spec["total_reads"] = ("total_reads", "sum")
    if "expected_detected" in per_file_df.columns:
        agg_spec["expected_detected"] = ("expected_detected", "sum")
    if "unexpected_detected" in per_file_df.columns:
        agg_spec["unexpected_detected"] = ("unexpected_detected", "sum")
    if "coverage_rate" in per_file_df.columns:
        agg_spec["avg_coverage_rate"] = ("coverage_rate", "mean")
    if "tag_read_redundancy" in per_file_df.columns:
        agg_spec["avg_tag_redundancy"] = ("tag_read_redundancy", "mean")
    if "antenna_balance" in per_file_df.columns:
        agg_spec["avg_antenna_balance"] = ("antenna_balance", "mean")
    if "average_dwell_seconds" in per_file_df.columns:
        agg_spec["avg_dwell_seconds"] = ("average_dwell_seconds", "mean")
    if "throughput_per_minute" in per_file_df.columns:
        agg_spec["avg_throughput_per_minute"] = ("throughput_per_minute", "mean")
    if "session_throughput" in per_file_df.columns:
        agg_spec["avg_session_throughput"] = ("session_throughput", "mean")
    if "read_continuity_rate" in per_file_df.columns:
        agg_spec["avg_read_continuity"] = ("read_continuity_rate", "mean")
    if "session_duration_seconds" in per_file_df.columns:
        agg_spec["avg_session_duration"] = ("session_duration_seconds", "mean")
    if "layout_overall_coverage" in per_file_df.columns:
        agg_spec["avg_layout_coverage"] = ("layout_overall_coverage", "mean")
    if "concurrency_peak" in per_file_df.columns:
        agg_spec["max_concurrency_peak"] = ("concurrency_peak", "max")
    if "concurrency_average" in per_file_df.columns:
        agg_spec["avg_concurrency"] = ("concurrency_average", "mean")
    if "tag_dwell_time_max" in per_file_df.columns:
        agg_spec["max_tag_dwell_time"] = ("tag_dwell_time_max", "max")
    if "inactive_periods_count" in per_file_df.columns:
        agg_spec["total_inactive_periods"] = ("inactive_periods_count", "sum")
    if "inactive_total_seconds" in per_file_df.columns:
        agg_spec["sum_inactive_seconds"] = ("inactive_total_seconds", "sum")
    if "inactive_longest_seconds" in per_file_df.columns:
        agg_spec["max_inactive_seconds"] = ("inactive_longest_seconds", "max")
    if "congestion_index" in per_file_df.columns:
        agg_spec["avg_congestion_index"] = ("congestion_index", "mean")
    if "global_rssi_avg" in per_file_df.columns:
        agg_spec["avg_global_rssi"] = ("global_rssi_avg", "mean")
    if "global_rssi_std" in per_file_df.columns:
        agg_spec["avg_global_rssi_std"] = ("global_rssi_std", "mean")

    overview = (
        per_file_df.groupby("mode", dropna=False).agg(**agg_spec).reset_index()
    )

    float_cols = overview.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_cols):
        overview[float_cols] = overview[float_cols].round(2)

    overall_row: dict[str, object] = {"mode": "overall", "files": int(per_file_df["file"].count())}
    if "total_epcs" in per_file_df.columns:
        overall_row["total_epcs"] = float(per_file_df["total_epcs"].sum())
    if "total_reads" in per_file_df.columns:
        overall_row["total_reads"] = float(per_file_df["total_reads"].sum())
    if "expected_detected" in per_file_df.columns:
        overall_row["expected_detected"] = float(per_file_df["expected_detected"].sum())
    if "unexpected_detected" in per_file_df.columns:
        overall_row["unexpected_detected"] = float(per_file_df["unexpected_detected"].sum())
    if "coverage_rate" in per_file_df.columns:
        overall_row["avg_coverage_rate"] = _clean_float(per_file_df["coverage_rate"].mean())
    if "tag_read_redundancy" in per_file_df.columns:
        overall_row["avg_tag_redundancy"] = _clean_float(per_file_df["tag_read_redundancy"].mean())
    if "antenna_balance" in per_file_df.columns:
        overall_row["avg_antenna_balance"] = _clean_float(per_file_df["antenna_balance"].mean())
    if "average_dwell_seconds" in per_file_df.columns:
        overall_row["avg_dwell_seconds"] = _clean_float(per_file_df["average_dwell_seconds"].mean())
    if "throughput_per_minute" in per_file_df.columns:
        overall_row["avg_throughput_per_minute"] = _clean_float(per_file_df["throughput_per_minute"].mean())
    if "session_throughput" in per_file_df.columns:
        overall_row["avg_session_throughput"] = _clean_float(per_file_df["session_throughput"].mean())
    if "read_continuity_rate" in per_file_df.columns:
        overall_row["avg_read_continuity"] = _clean_float(per_file_df["read_continuity_rate"].mean())
    if "session_duration_seconds" in per_file_df.columns:
        overall_row["avg_session_duration"] = _clean_float(per_file_df["session_duration_seconds"].mean())
    if "layout_overall_coverage" in per_file_df.columns:
        overall_row["avg_layout_coverage"] = _clean_float(per_file_df["layout_overall_coverage"].mean())
    if "concurrency_peak" in per_file_df.columns:
        overall_row["max_concurrency_peak"] = _clean_int(per_file_df["concurrency_peak"].max())
    if "concurrency_average" in per_file_df.columns:
        overall_row["avg_concurrency"] = _clean_float(per_file_df["concurrency_average"].mean())
    if "tag_dwell_time_max" in per_file_df.columns:
        overall_row["max_tag_dwell_time"] = _clean_float(per_file_df["tag_dwell_time_max"].max())
    if "inactive_periods_count" in per_file_df.columns:
        overall_row["total_inactive_periods"] = _clean_float(per_file_df["inactive_periods_count"].sum())
    if "inactive_total_seconds" in per_file_df.columns:
        overall_row["sum_inactive_seconds"] = _clean_float(per_file_df["inactive_total_seconds"].sum())
    if "inactive_longest_seconds" in per_file_df.columns:
        overall_row["max_inactive_seconds"] = _clean_float(per_file_df["inactive_longest_seconds"].max())
    if "congestion_index" in per_file_df.columns:
        overall_row["avg_congestion_index"] = _clean_float(per_file_df["congestion_index"].mean())
    if "global_rssi_avg" in per_file_df.columns:
        overall_row["avg_global_rssi"] = _clean_float(per_file_df["global_rssi_avg"].mean())
    if "global_rssi_std" in per_file_df.columns:
        overall_row["avg_global_rssi_std"] = _clean_float(per_file_df["global_rssi_std"].mean())

    overview = pd.concat([overview, pd.DataFrame([overall_row])], ignore_index=True)

    return overview


def generate_consolidated_summary(
    summary_records: list[dict[str, object]] | None, out_dir: Path
) -> Path | None:
    """Persist consolidated executive metrics to an Excel workbook."""

    if not summary_records:
        LOGGER.info("No per-file metrics captured; skipping consolidated summary generation.")
        return None

    per_file_df = pd.DataFrame(summary_records)
    if per_file_df.empty:
        LOGGER.info("Per-file registry is empty; skipping consolidated summary generation.")
        return None

    column_order = [col for col in SUMMARY_COLUMN_ORDER if col in per_file_df.columns]
    column_order.extend([col for col in per_file_df.columns if col not in column_order])
    per_file_df = per_file_df[column_order]

    if "layout_used" in per_file_df.columns:
        per_file_df["layout_used"] = per_file_df["layout_used"].fillna(False).astype(bool)

    if "rssi_noise_flag" in per_file_df.columns:
        per_file_df["rssi_noise_flag"] = per_file_df["rssi_noise_flag"].apply(
            lambda value: None if pd.isna(value) else bool(value)
        )

    float_candidates = [
        "coverage_rate",
        "tag_read_redundancy",
        "antenna_balance",
        "rssi_stability_index",
        "average_dwell_seconds",
        "throughput_per_minute",
        "session_throughput",
        "read_continuity_rate",
        "session_duration_seconds",
        "session_active_seconds",
        "tag_dwell_time_max",
        "concurrency_average",
        "layout_overall_coverage",
        "epcs_per_minute_mean",
        "analysis_window_seconds",
        "inactive_total_seconds",
        "inactive_longest_seconds",
        "congestion_index",
        "global_rssi_avg",
        "global_rssi_std",
        "rssi_noise_reads_per_epc",
    ]
    for column in float_candidates:
        if column in per_file_df.columns:
            per_file_df[column] = pd.to_numeric(per_file_df[column], errors="coerce")

    overview_df = _build_overview_table(per_file_df)

    output_path = out_dir / SUMMARY_FILE_NAME
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        per_file_df.to_excel(writer, index=False, sheet_name=SUMMARY_PER_FILE_SHEET)
        overview_df.to_excel(writer, index=False, sheet_name=SUMMARY_OVERVIEW_SHEET)

    LOGGER.info("Executive summary workbook saved to: %s", output_path)
    return output_path


def process_file(
    csv_path: Path,
    out_dir: Path,
    layout_df: pd.DataFrame | None,
    expected_registry: dict[str, set[str]] | None = None,
    summary_records: list[dict[str, object]] | None = None,
):
    df, metadata = read_itemtest_csv(str(csv_path))

    summary = summarize_by_epc(df)
    ant_counts = summarize_by_antenna(df)

    figures_dir = out_dir / "graficos" / csv_path.stem
    figures_dir.mkdir(parents=True, exist_ok=True)

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
                    token_value = str(token).strip()
                    suffix = (
                        token_value[-3:].upper()
                        if len(token_value) >= 3
                        else token_value.upper()
                    )
                    total_reads_for_suffix = summary.loc[
                        summary["EPC_suffix3"] == suffix, "total_reads"
                    ].sum()
                    expected_token = token_value.upper()
                    expected_epc = (
                        expected_token
                        if expected_token and HEX_EPC_PATTERN.match(expected_token)
                        else None
                    )
                    position_label = f"{face_label} - Row {row_label}"
                    coverage_records.append(
                        {
                            ROW_COLUMN: row_label,
                            "Face": face_label,
                            "Suffix": suffix,
                            "Read": bool(total_reads_for_suffix),
                            "total_reads": int(total_reads_for_suffix),
                            "PositionLabel": position_label,
                            "ExpectedToken": expected_token,
                            "ExpectedEPC": expected_epc,
                        }
                    )
        positions_df = pd.DataFrame(coverage_records)

        plot_pallet_heatmap(
            positions_df,
            str(figures_dir),
            title=f"Pallet heatmap — {csv_path.name}",
        )

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
    plot_rssi_vs_frequency(
        df,
        str(figures_dir),
        title=f"RSSI vs Frequency — {csv_path.name}",
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
    _register_structured_summary(
        summary_records,
        csv_path=csv_path,
        excel_out=excel_out,
        summary_df=summary,
        metadata=metadata,
        structured_metrics=structured_metrics,
        summary_log=summary_log,
        layout_applied=layout_df is not None,
    )
    return excel_out


def process_continuous_file(
    csv_path: Path,
    out_dir: Path,
    window_seconds: float,
    expected_registry: dict[str, set[str]] | None = None,
    summary_records: list[dict[str, object]] | None = None,
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
        "read_continuity_rate": result.read_continuity_rate,
        "throughput_per_minute": result.throughput_per_minute,
        "session_throughput": result.session_throughput,
        "session_start": result.session_start,
        "session_end": result.session_end,
        "session_end_with_grace": result.session_end_with_grace,
        "session_duration_seconds": result.session_duration_seconds,
        "session_active_seconds": result.session_active_seconds,
        "tag_dwell_time_max": result.tag_dwell_time_max,
        "inactive_periods": result.inactive_periods,
        "inactive_periods_count": result.inactive_periods_count,
        "inactive_total_seconds": result.inactive_total_seconds,
        "inactive_longest_seconds": result.inactive_longest_seconds,
        "congestion_index": result.congestion_index,
        "global_rssi_avg": result.global_rssi_avg,
        "global_rssi_std": result.global_rssi_std,
        "concurrency_peak": result.concurrency_peak,
        "concurrency_peak_time": result.concurrency_peak_time,
        "concurrency_average": result.concurrency_average,
        "concurrency_timeline": result.concurrency_timeline,
    }
    continuous_details.update(
        {
            "rssi_noise_flag": result.rssi_noise_flag,
            "rssi_noise_indicator": result.rssi_noise_indicator,
            "rssi_noise_reads_per_epc": result.rssi_noise_reads_per_epc,
        }
    )
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

    concurrency_log_path = log_dir / f"{csv_path.stem}_concurrency_timeline.csv"
    concurrency_log = result.concurrency_timeline.copy()
    if concurrency_log.empty:
        concurrency_log = pd.DataFrame(
            columns=["timestamp", "change", "active_epcs", "duration_seconds"]
        )
    else:
        if "timestamp" in concurrency_log.columns:
            concurrency_log["timestamp"] = pd.to_datetime(
                concurrency_log["timestamp"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
    concurrency_log.to_csv(concurrency_log_path, index=False, encoding="utf-8")
    LOGGER.info("Linha do tempo de simultaneidade exportada para: %s", concurrency_log_path)

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
    plot_rssi_vs_frequency(
        df,
        str(fig_dir),
        title=f"RSSI vs Frequency — {csv_path.name} (continuous)",
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
    _register_continuous_summary(
        summary_records,
        csv_path=csv_path,
        excel_out=excel_out,
        summary_df=summary,
        metadata=metadata,
        continuous_details=continuous_details,
        summary_log=summary_log,
        alerts=alerts,
        window_seconds=window_seconds,
    )
    return excel_out


def orchestrate_processing(
    csv_files: Iterable[Path],
    *,
    mode: str,
    out_dir: Path,
    layout_df: pd.DataFrame | None,
    expected_registry: dict[str, set[str]] | None,
    window_seconds: float,
    summary_records: list[dict[str, object]] | None = None,
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
                    summary_records=summary_records,
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
                summary_records=summary_records,
            )
            results.append(result)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Return an argument parser configured with all CLI options."""

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
    ap.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Generate an executive summary workbook consolidating KPIs across all processed "
            "CSV files in the selected mode."
        ),
    )
    return ap


def main():
    ap = build_arg_parser()
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

    summary_records: list[dict[str, object]] | None = [] if args.summary else None

    results = orchestrate_processing(
        csv_files,
        mode=effective_mode,
        out_dir=out_dir,
        layout_df=layout_df,
        expected_registry=expected_registry,
        window_seconds=window_seconds,
        summary_records=summary_records,
    )
    LOGGER.info("Completed. Generated files:")
    for r in results:
        LOGGER.info(" - %s", r)

    if summary_records is not None:
        summary_path = generate_consolidated_summary(summary_records, out_dir)
        if summary_path is not None:
            LOGGER.info("Consolidated executive summary available at: %s", summary_path)

if __name__ == "__main__":
    main()
