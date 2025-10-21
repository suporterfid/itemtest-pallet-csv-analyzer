# -*- coding: utf-8 -*-
"""Utility functions for calculating metrics used by the analyzer."""

from __future__ import annotations

from collections import Counter
from typing import Collection

import numpy as np
import pandas as pd

from .parser import suffix3


def antenna_mode(series: pd.Series):
    """Return the most frequent antenna identifier for *series*.

    Parameters
    ----------
    series:
        Series containing antenna identifiers for a single EPC.

    Returns
    -------
    int | None
        The most frequent antenna as an integer, or ``None`` when unavailable.
    """

    s = series.dropna().astype(int)
    if s.empty:
        return None
    counts = Counter(s)
    maxc = max(counts.values())
    # Choose the lowest antenna ID when there is a tie
    return min([a for a, c in counts.items() if c == maxc])


def summarize_by_epc(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ItemTest reads by EPC and expose convenience columns."""

    g = df.groupby("EPC", as_index=False)
    summary = g.agg(
        total_reads=("EPC", "count"),
        rssi_avg=("RSSI", "mean"),
        rssi_best=("RSSI", "max"),
        rssi_worst=("RSSI", "min"),
        first_time=("Timestamp", "min"),
        last_time=("Timestamp", "max"),
    )
    # Determine first/last antenna based on timestamp ordering
    first_rows = (
        df.sort_values(["EPC", "Timestamp"])
        .groupby("EPC", as_index=False)
        .first()[["EPC", "Antenna"]]
        .rename(columns={"Antenna": "antenna_first"})
    )
    last_rows = (
        df.sort_values(["EPC", "Timestamp"])
        .groupby("EPC", as_index=False)
        .last()[["EPC", "Antenna"]]
        .rename(columns={"Antenna": "antenna_last"})
    )
    summary = summary.merge(first_rows, on="EPC", how="left").merge(
        last_rows, on="EPC", how="left"
    )
    # Most frequent antenna per EPC
    modes = (
        df.groupby("EPC")["Antenna"]
        .apply(antenna_mode)
        .reset_index(name="antenna_mode")
    )
    summary = summary.merge(modes, on="EPC", how="left")
    # EPC suffix for matching layout positions
    summary["EPC_suffix3"] = summary["EPC"].astype(str).apply(suffix3).str.upper()
    return summary


def summarize_by_antenna(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise read counts and RSSI statistics per antenna."""

    ant = (
        df.dropna(subset=["Antenna"])
        .groupby("Antenna")
        .agg(total_reads=("EPC", "count"), rssi_avg=("RSSI", "mean"))
        .reset_index()
        .sort_values("Antenna")
    )
    total_reads = int(ant["total_reads"].sum()) if not ant.empty else 0
    if total_reads > 0:
        ant["participation_pct"] = ant["total_reads"].astype(float) / total_reads * 100
    else:
        ant["participation_pct"] = 0.0
    return ant


def _extract_rssi_series(raw_df: pd.DataFrame | None) -> pd.Series:
    """Return a numeric RSSI series extracted from ``raw_df``."""

    if raw_df is None or raw_df.empty or "RSSI" not in raw_df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(raw_df["RSSI"], errors="coerce").dropna()
    return series if not series.empty else pd.Series(dtype=float)


def calculate_global_rssi_average(raw_df: pd.DataFrame | None) -> float:
    """Return the global RSSI average in dBm for ``raw_df``."""

    series = _extract_rssi_series(raw_df)
    if series.empty:
        return float("nan")
    return float(series.mean())


def calculate_global_rssi_std(raw_df: pd.DataFrame | None) -> float:
    """Return the global RSSI population standard deviation for ``raw_df``."""

    series = _extract_rssi_series(raw_df)
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0))


def compile_global_rssi_metrics(
    raw_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None = None,
    *,
    std_threshold: float = 4.0,
    redundancy_threshold: float = 10.0,
) -> dict[str, object]:
    """Return aggregated RSSI statistics and a noise indicator.

    Parameters
    ----------
    raw_df:
        Raw ItemTest readings containing an ``RSSI`` column.
    summary_df:
        Optional per-EPC summary with ``total_reads`` to estimate read redundancy.
    std_threshold:
        Minimum population standard deviation (in dBm) that signals high variance.
    redundancy_threshold:
        Minimum average reads per EPC required to flag potential noise.
    """

    avg = calculate_global_rssi_average(raw_df)
    std = calculate_global_rssi_std(raw_df)

    total_reads: float | None = None
    unique_epcs: float | None = None

    if summary_df is not None and not summary_df.empty:
        if "EPC" in summary_df.columns:
            try:
                unique_epcs = float(summary_df["EPC"].nunique())
            except Exception:  # pragma: no cover - defensive fallback
                unique_epcs = float(summary_df.shape[0])
        else:
            unique_epcs = float(summary_df.shape[0])
        if "total_reads" in summary_df.columns:
            total_reads = float(
                pd.to_numeric(summary_df["total_reads"], errors="coerce").sum()
            )
        else:
            total_reads = float(summary_df.shape[0])
    elif raw_df is not None and not raw_df.empty:
        if "EPC" in raw_df.columns:
            try:
                unique_epcs = float(raw_df["EPC"].nunique())
            except Exception:  # pragma: no cover - defensive fallback
                unique_epcs = None
        total_reads = float(raw_df.shape[0])

    reads_per_epc = float("nan")
    if unique_epcs is not None and unique_epcs > 0 and total_reads is not None:
        reads_per_epc = float(total_reads) / float(unique_epcs)

    noise_flag: bool | None = None
    indicator: str | None = None

    if not np.isnan(std):
        if not np.isnan(reads_per_epc):
            noise_flag = bool(std >= std_threshold and reads_per_epc >= redundancy_threshold)
            if noise_flag:
                indicator = (
                    f"Variação elevada sem ganho de EPCs (σ={std:.2f} dBm; "
                    f"{reads_per_epc:.1f} leituras/EPC)"
                )
            else:
                indicator = (
                    f"Estabilidade de RSSI dentro do esperado (σ={std:.2f} dBm; "
                    f"{reads_per_epc:.1f} leituras/EPC)"
                )
        else:
            indicator = f"Variação de RSSI calculada (σ={std:.2f} dBm)"

    return {
        "global_rssi_avg": avg,
        "global_rssi_std": std,
        "rssi_noise_flag": noise_flag,
        "rssi_noise_indicator": indicator,
        "rssi_noise_reads_per_epc": reads_per_epc,
    }


def _normalise_expected_sets(
    expected_full: Collection[str] | None,
    expected_suffixes: Collection[str] | None,
) -> tuple[set[str], set[str]]:
    """Return normalised uppercase sets without duplicate suffix entries."""

    full_set = {
        str(value).strip().upper()
        for value in expected_full or []
        if str(value).strip()
    }
    suffix_set = {
        str(value).strip().upper()[-3:]
        for value in expected_suffixes or []
        if str(value).strip()
    }
    # Avoid counting the suffix twice when the full EPC is provided.
    full_suffixes = {
        value[-3:].upper()
        for value in full_set
        if isinstance(value, str) and len(value) >= 3
    }
    suffix_only = {suffix for suffix in suffix_set if suffix not in full_suffixes}
    return full_set, suffix_only


def _observed_epc_tokens(summary_df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Extract unique observed EPCs and suffix tokens from ``summary_df``."""

    if summary_df is None or summary_df.empty or "EPC" not in summary_df.columns:
        return set(), set()
    observed_full = (
        summary_df["EPC"].dropna().astype(str).str.upper().str.strip().tolist()
    )
    full_set = {token for token in observed_full if token}
    if "EPC_suffix3" in summary_df.columns:
        suffix_series = summary_df["EPC_suffix3"].dropna().astype(str)
        suffix_set = {token.upper()[-3:] for token in suffix_series if token}
    else:
        suffix_set = {
            token[-3:]
            for token in observed_full
            if isinstance(token, str) and len(token) >= 3
        }
    return full_set, suffix_set


def calculate_expected_epc_stats(
    summary_df: pd.DataFrame,
    *,
    expected_full: Collection[str] | None = None,
    expected_suffixes: Collection[str] | None = None,
) -> dict[str, object]:
    """Return coverage and missing counts for expected EPC tokens."""

    expected_full_set, expected_suffix_set = _normalise_expected_sets(
        expected_full, expected_suffixes
    )
    observed_full, observed_suffix = _observed_epc_tokens(summary_df)

    missing_full = expected_full_set - observed_full
    missing_suffix = expected_suffix_set - observed_suffix
    total_expected = len(expected_full_set) + len(expected_suffix_set)
    found_expected = total_expected - len(missing_full) - len(missing_suffix)

    if total_expected > 0:
        coverage_rate = found_expected / total_expected * 100
    else:
        coverage_rate = np.nan

    return {
        "total_expected": total_expected,
        "found_expected": found_expected,
        "missing_full": missing_full,
        "missing_suffix": missing_suffix,
        "coverage_rate": coverage_rate,
    }


def calculate_coverage_rate(
    summary_df: pd.DataFrame,
    expected_full: Collection[str] | None,
    expected_suffixes: Collection[str] | None,
) -> float:
    """Return the coverage percentage for expected tags."""

    stats = calculate_expected_epc_stats(
        summary_df, expected_full=expected_full, expected_suffixes=expected_suffixes
    )
    return float(stats["coverage_rate"]) if stats["coverage_rate"] is not None else np.nan


def calculate_tag_read_redundancy(
    summary_df: pd.DataFrame,
    *,
    expected_only: bool = False,
) -> float:
    """Compute the average number of reads per EPC."""

    if summary_df is None or summary_df.empty or "total_reads" not in summary_df.columns:
        return np.nan
    if expected_only and "expected_epc" in summary_df.columns:
        working_df = summary_df.loc[summary_df["expected_epc"]]
    else:
        working_df = summary_df
    total_reads = float(working_df.get("total_reads", pd.Series(dtype=float)).sum())
    unique_epcs = int(working_df.shape[0])
    if unique_epcs == 0:
        return np.nan
    return total_reads / unique_epcs


def calculate_antenna_balance(ant_counts: pd.DataFrame) -> float:
    """Return the standard deviation (percentage) of antenna participation."""

    if ant_counts is None or ant_counts.empty or "total_reads" not in ant_counts.columns:
        return np.nan
    total_reads = ant_counts["total_reads"].astype(float).sum()
    if total_reads <= 0:
        return np.nan
    proportions = ant_counts["total_reads"].astype(float) / total_reads
    if proportions.empty:
        return np.nan
    return float(proportions.std(ddof=0) * 100)


def calculate_rssi_stability_index(df: pd.DataFrame) -> float:
    """Return the standard deviation of mean RSSI per antenna."""

    if df is None or df.empty or "RSSI" not in df.columns or "Antenna" not in df.columns:
        return np.nan
    clean = df.dropna(subset=["RSSI", "Antenna"])
    if clean.empty:
        return np.nan
    means = clean.groupby("Antenna")["RSSI"].mean()
    if means.empty:
        return np.nan
    if len(means) == 1:
        return 0.0
    return float(means.std(ddof=0))


def calculate_layout_face_coverage(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Return coverage breakdown by pallet face."""

    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=["Face", "total_positions", "read_positions", "coverage_pct", "total_reads"])
    grouped = (
        positions_df.groupby("Face", dropna=False)
        .agg(
            total_positions=("Suffix", "count"),
            read_positions=("Read", "sum"),
            total_reads=("total_reads", "sum"),
        )
        .reset_index()
    )
    grouped["read_positions"] = grouped["read_positions"].astype(int)
    grouped["coverage_pct"] = grouped.apply(
        lambda row: (row["read_positions"] / row["total_positions"] * 100)
        if row["total_positions"]
        else 0.0,
        axis=1,
    )
    return grouped.sort_values("Face").reset_index(drop=True)


def calculate_layout_row_coverage(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Return coverage breakdown by pallet row and face."""

    if positions_df is None or positions_df.empty:
        return pd.DataFrame(
            columns=["Face", "Row", "total_positions", "read_positions", "coverage_pct", "total_reads"]
        )
    grouped = (
        positions_df.groupby(["Face", "Row"], dropna=False)
        .agg(
            total_positions=("Suffix", "count"),
            read_positions=("Read", "sum"),
            total_reads=("total_reads", "sum"),
        )
        .reset_index()
    )
    grouped["read_positions"] = grouped["read_positions"].astype(int)
    grouped["coverage_pct"] = grouped.apply(
        lambda row: (row["read_positions"] / row["total_positions"] * 100)
        if row["total_positions"]
        else 0.0,
        axis=1,
    )
    return grouped.sort_values(["Row", "Face"]).reset_index(drop=True)


def detect_read_hotspots(
    summary_df: pd.DataFrame, *, std_multiplier: float = 2.0
) -> tuple[pd.DataFrame, float]:
    """Return EPCs whose read volume exceeds ``mean + std_multiplier × std``.

    Parameters
    ----------
    summary_df:
        Aggregated per-EPC summary produced by :func:`summarize_by_epc`.
    std_multiplier:
        Multiplier applied to the population standard deviation to compute the
        hotspot threshold. Defaults to ``2.0`` (≈95% confidence interval).

    Returns
    -------
    tuple[pd.DataFrame, float]
        A tuple with the table of hotspots (sorted by ``total_reads`` in
        descending order) and the numeric threshold used for detection. When no
        hotspot is found the DataFrame is empty and the threshold is ``np.nan``.
    """

    base_columns = ["EPC", "EPC_suffix3", "total_reads", "expected_epc", "pallet_position"]
    result_columns = [column for column in base_columns if column in (summary_df.columns if summary_df is not None else [])]
    result_columns.append("z_score")

    empty = pd.DataFrame(columns=result_columns)
    if (
        summary_df is None
        or summary_df.empty
        or "total_reads" not in summary_df.columns
        or std_multiplier <= 0
    ):
        return empty, float("nan")

    working = summary_df.copy()
    working["total_reads"] = pd.to_numeric(working["total_reads"], errors="coerce")
    working = working.dropna(subset=["total_reads"])
    if working.empty:
        return empty, float("nan")

    mean_reads = float(working["total_reads"].mean())
    std_reads = float(working["total_reads"].std(ddof=0))
    if std_reads <= 0 or np.isnan(std_reads):
        return empty, float("nan")

    threshold = mean_reads + std_multiplier * std_reads
    working["z_score"] = (working["total_reads"] - mean_reads) / std_reads
    hotspots = working.loc[working["total_reads"] >= threshold].copy()
    if hotspots.empty:
        return empty, threshold

    hotspots = hotspots.sort_values("total_reads", ascending=False)
    hotspots = hotspots[result_columns]
    return hotspots.reset_index(drop=True), float(threshold)


def calculate_frequency_usage(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return the unique frequency channels observed in ``raw_df``.

    The resulting DataFrame exposes the ``frequency_mhz`` column alongside the
    absolute ``read_count`` and the relative participation in percentage.
    """

    columns = ["frequency_mhz", "read_count", "participation_pct"]
    if raw_df is None or raw_df.empty or "Frequency" not in raw_df.columns:
        return pd.DataFrame(columns=columns)

    freq_series = pd.to_numeric(raw_df["Frequency"], errors="coerce").dropna()
    if freq_series.empty:
        return pd.DataFrame(columns=columns)

    counts = freq_series.round(3).value_counts()
    total_reads = float(counts.sum())
    usage_df = (
        counts.rename_axis("frequency_mhz")
        .reset_index(name="read_count")
        .sort_values("frequency_mhz")
        .reset_index(drop=True)
    )
    if total_reads > 0:
        usage_df["participation_pct"] = usage_df["read_count"].astype(float) / total_reads * 100
    else:
        usage_df["participation_pct"] = 0.0
    return usage_df


def detect_location_errors(
    summary_df: pd.DataFrame, positions_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Return EPCs whose suffix matches a position with a different expected EPC."""

    columns = [
        "EPC",
        "EPC_suffix3",
        "total_reads",
        "ExpectedEPC",
        "ExpectedPosition",
        "ObservedPosition",
    ]
    if (
        summary_df is None
        or summary_df.empty
        or positions_df is None
        or positions_df.empty
        or "ExpectedEPC" not in positions_df.columns
    ):
        return pd.DataFrame(columns=columns)

    expected_positions = positions_df.dropna(subset=["ExpectedEPC"]).copy()
    expected_positions["ExpectedEPC"] = expected_positions["ExpectedEPC"].astype(str).str.upper()
    expected_positions = expected_positions.loc[expected_positions["ExpectedEPC"].str.len() > 0]
    if expected_positions.empty:
        return pd.DataFrame(columns=columns)

    expected_positions["Suffix"] = expected_positions["Suffix"].astype(str).str.upper()
    if "PositionLabel" in expected_positions.columns:
        expected_positions["PositionLabel"] = expected_positions["PositionLabel"].astype(str)
    else:
        expected_positions["PositionLabel"] = expected_positions.apply(
            lambda row: f"{row.get('Face', 'Unknown')} - Row {row.get('Row', '?')}",
            axis=1,
        )

    grouped = expected_positions.groupby("Suffix").agg(
        expected_epcs=("ExpectedEPC", lambda values: sorted({str(v) for v in values if str(v).strip()})),
        positions=(
            "PositionLabel",
            lambda values: sorted({str(v) for v in values if str(v).strip()}),
        ),
    )

    if grouped.empty:
        return pd.DataFrame(columns=columns)

    observed = summary_df.copy()
    observed["EPC"] = observed["EPC"].astype(str).str.upper()
    observed["EPC_suffix3"] = observed["EPC_suffix3"].astype(str).str.upper()
    observed["total_reads"] = pd.to_numeric(observed.get("total_reads"), errors="coerce")
    if "pallet_position" in observed.columns:
        observed_position_col = "pallet_position"
    else:
        observed_position_col = None

    errors: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        suffix = getattr(row, "EPC_suffix3", None)
        if not suffix or suffix not in grouped.index:
            continue
        expected_epcs = grouped.loc[suffix, "expected_epcs"]
        if not expected_epcs:
            continue
        epc_value = getattr(row, "EPC", "")
        if epc_value in expected_epcs:
            continue
        position_list = grouped.loc[suffix, "positions"]
        observed_position = getattr(row, observed_position_col) if observed_position_col else None
        errors.append(
            {
                "EPC": epc_value,
                "EPC_suffix3": suffix,
                "total_reads": getattr(row, "total_reads", np.nan),
                "ExpectedEPC": "; ".join(expected_epcs),
                "ExpectedPosition": "; ".join(position_list),
                "ObservedPosition": observed_position,
            }
        )

    if not errors:
        return pd.DataFrame(columns=columns)

    errors_df = pd.DataFrame(errors)
    if "total_reads" in errors_df.columns:
        errors_df = errors_df.sort_values(
            by=["total_reads", "EPC"], ascending=[False, True]
        )
    return errors_df.reset_index(drop=True)


def calculate_read_distribution_by_face(positions_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return aggregated read counts per pallet face."""

    columns = [
        "Face",
        "total_positions",
        "positions_with_reads",
        "total_reads",
        "participation_pct",
    ]
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=columns)

    working = positions_df.copy()
    working["Face"] = working["Face"].fillna("Unknown")
    working["Read"] = working["Read"].astype(bool)

    grouped = (
        working.groupby("Face", dropna=False)
        .agg(
            total_positions=("Suffix", "count"),
            positions_with_reads=("Read", "sum"),
            total_reads=("total_reads", "sum"),
        )
        .reset_index()
    )
    grouped["positions_with_reads"] = grouped["positions_with_reads"].astype(int)
    total_reads = grouped["total_reads"].astype(float).sum()
    if total_reads > 0:
        grouped["participation_pct"] = grouped["total_reads"].astype(float) / total_reads * 100
    else:
        grouped["participation_pct"] = 0.0
    return grouped.sort_values("total_reads", ascending=False).reset_index(drop=True)


def compile_structured_kpis(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ant_counts: pd.DataFrame,
    *,
    expected_full: Collection[str] | None = None,
    expected_suffixes: Collection[str] | None = None,
    positions_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Aggregate KPI values for structured-mode analyses."""

    stats = calculate_expected_epc_stats(
        summary_df, expected_full=expected_full, expected_suffixes=expected_suffixes
    )
    redundancy = calculate_tag_read_redundancy(
        summary_df, expected_only=bool(expected_full or expected_suffixes)
    )
    balance = calculate_antenna_balance(ant_counts)
    rssi_stability = calculate_rssi_stability_index(raw_df)
    global_rssi_metrics = compile_global_rssi_metrics(raw_df, summary_df)

    top_performer: dict[str, object] | None = None
    if ant_counts is not None and not ant_counts.empty:
        candidate = ant_counts.sort_values("total_reads", ascending=False).iloc[0]
        top_performer = {
            "antenna": candidate.get("Antenna"),
            "total_reads": int(candidate.get("total_reads", 0))
            if not pd.isna(candidate.get("total_reads"))
            else None,
        }
        participation = candidate.get("participation_pct")
        if participation is not None and not pd.isna(participation):
            top_performer["participation_pct"] = float(participation)

    face_coverage = calculate_layout_face_coverage(positions_df) if positions_df is not None else pd.DataFrame()
    row_coverage = calculate_layout_row_coverage(positions_df) if positions_df is not None else pd.DataFrame()
    hotspots_df, hotspots_threshold = detect_read_hotspots(summary_df)
    frequency_usage = calculate_frequency_usage(raw_df)
    location_errors = detect_location_errors(summary_df, positions_df)
    face_distribution = calculate_read_distribution_by_face(positions_df)
    layout_total_positions = int(positions_df.shape[0]) if positions_df is not None else None
    layout_read_positions = (
        int(positions_df["Read"].sum()) if positions_df is not None and "Read" in positions_df.columns else None
    )
    layout_overall_coverage = (
        layout_read_positions / layout_total_positions * 100
        if layout_total_positions and layout_read_positions is not None
        else np.nan
    )
    missing_positions = (
        positions_df.loc[~positions_df["Read"]].copy()
        if positions_df is not None and "Read" in positions_df.columns
        else pd.DataFrame(columns=getattr(positions_df, "columns", []))
    )

    missing_labels: list[str] = []
    if not missing_positions.empty:
        for row in missing_positions[["Face", "Row", "Suffix"]].drop_duplicates().itertuples(index=False):
            face = getattr(row, "Face", "?")
            row_label = getattr(row, "Row", "?")
            suffix = getattr(row, "Suffix", "?")
            missing_labels.append(f"{face} - Row {row_label} ({suffix})")

    result = {
        "coverage_rate": stats["coverage_rate"],
        "expected_total": stats["total_expected"],
        "expected_found": stats["found_expected"],
        "missing_expected_full": sorted(stats["missing_full"]),
        "missing_expected_suffix": sorted(stats["missing_suffix"]),
        "tag_read_redundancy": redundancy,
        "antenna_balance": balance,
        "rssi_stability_index": rssi_stability,
        "top_performer_antenna": top_performer,
        "layout_face_coverage": face_coverage,
        "layout_row_coverage": row_coverage,
        "layout_total_positions": layout_total_positions,
        "layout_read_positions": layout_read_positions,
        "layout_overall_coverage": layout_overall_coverage,
        "missing_position_labels": missing_labels,
        "missing_positions_table": missing_positions,
        "read_hotspots": hotspots_df,
        "read_hotspots_threshold": hotspots_threshold,
        "read_hotspots_count": int(hotspots_df.shape[0]) if hotspots_df is not None else 0,
        "frequency_usage": frequency_usage,
        "frequency_unique_count": int(frequency_usage.shape[0]) if frequency_usage is not None else 0,
        "location_errors": location_errors,
        "location_error_count": int(location_errors.shape[0]) if location_errors is not None else 0,
        "reads_by_face": face_distribution,
    }
    result.update(global_rssi_metrics)
    return result


__all__ = [
    "antenna_mode",
    "summarize_by_epc",
    "summarize_by_antenna",
    "calculate_expected_epc_stats",
    "calculate_coverage_rate",
    "calculate_tag_read_redundancy",
    "calculate_antenna_balance",
    "calculate_rssi_stability_index",
    "calculate_global_rssi_average",
    "calculate_global_rssi_std",
    "compile_global_rssi_metrics",
    "calculate_layout_face_coverage",
    "calculate_layout_row_coverage",
    "detect_read_hotspots",
    "calculate_frequency_usage",
    "detect_location_errors",
    "calculate_read_distribution_by_face",
    "compile_structured_kpis",
]
