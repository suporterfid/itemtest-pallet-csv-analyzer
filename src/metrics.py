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

    return {
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
    }


__all__ = [
    "antenna_mode",
    "summarize_by_epc",
    "summarize_by_antenna",
    "calculate_expected_epc_stats",
    "calculate_coverage_rate",
    "calculate_tag_read_redundancy",
    "calculate_antenna_balance",
    "calculate_rssi_stability_index",
    "calculate_layout_face_coverage",
    "calculate_layout_row_coverage",
    "compile_structured_kpis",
]
