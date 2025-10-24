# -*- coding: utf-8 -*-
"""Utility functions for calculating metrics used by the analyzer."""

from __future__ import annotations

from collections import Counter
from typing import Collection, Iterable

import numpy as np
import pandas as pd

from .parser import suffix3

LOGISTICS_PREFIX = "331A"


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

    reads_per_epc_value: float | None
    if np.isnan(reads_per_epc):
        reads_per_epc_value = None
    else:
        reads_per_epc_value = float(reads_per_epc)

    return {
        "global_rssi_avg": avg,
        "global_rssi_std": std,
        "rssi_noise_flag": noise_flag,
        "rssi_noise_indicator": indicator,
        "rssi_noise_reads_per_epc": reads_per_epc_value,
    }


def calculate_mode_performance(
    metadata: dict[str, object] | None,
    summary_df: pd.DataFrame | None,
    raw_df: pd.DataFrame | None,
) -> dict[str, object]:
    """Return reading-rate indicators associated with the ItemTest ``ModeIndex``.

    The helper inspects the ``ModeIndex`` entry from ``metadata`` and estimates the
    observation window using the earliest and latest timestamps available from the
    raw reads (or, as a fallback, from the aggregated per-EPC summary). It then
    derives read throughput metrics such as ``reads_per_second`` and
    ``epcs_per_minute``. When the ``ModeIndex`` metadata is absent the function
    returns an empty indicator while keeping the output structure predictable.

    Parameters
    ----------
    metadata:
        Dictionary returned by :func:`parser.read_itemtest_csv` containing the
        ItemTest metadata comment headers.
    summary_df:
        Optional per-EPC aggregation produced by :func:`summarize_by_epc`.
    raw_df:
        Raw ItemTest reads as returned by :func:`parser.read_itemtest_csv`.

    Returns
    -------
    dict[str, object]
        Dictionary containing the normalised ``mode_index`` value along with the
        computed rates and a human-readable ``description`` ready to be surfaced
        in reports.
    """

    result: dict[str, object] = {
        "mode_index": None,
        "reads_per_second": None,
        "reads_per_minute": None,
        "epcs_per_minute": None,
        "observation_seconds": None,
        "description": None,
    }

    if not metadata or not isinstance(metadata, dict):
        return result

    raw_mode = metadata.get("ModeIndex")
    if raw_mode is None:
        return result
    try:
        if pd.isna(raw_mode):
            return result
    except Exception:  # pragma: no cover - defensive fallback
        pass

    try:
        mode_index: int | str = int(raw_mode)
    except (TypeError, ValueError):
        mode_index = raw_mode

    try:
        if pd.isna(mode_index):
            return result
    except Exception:  # pragma: no cover - defensive fallback
        pass

    result["mode_index"] = mode_index

    total_reads: float | None = None
    if raw_df is not None and not raw_df.empty:
        total_reads = float(raw_df.shape[0])
    elif summary_df is not None and not summary_df.empty:
        if "total_reads" in summary_df.columns:
            total_reads = float(
                pd.to_numeric(summary_df["total_reads"], errors="coerce").sum()
            )
        else:
            total_reads = float(summary_df.shape[0])

    unique_epcs: float | None = None
    if summary_df is not None and not summary_df.empty:
        if "EPC" in summary_df.columns:
            try:
                unique_epcs = float(summary_df["EPC"].nunique())
            except Exception:  # pragma: no cover - defensive fallback
                unique_epcs = float(summary_df.shape[0])
        else:
            unique_epcs = float(summary_df.shape[0])
    elif raw_df is not None and not raw_df.empty and "EPC" in raw_df.columns:
        try:
            unique_epcs = float(raw_df["EPC"].nunique())
        except Exception:  # pragma: no cover - defensive fallback
            unique_epcs = None

    start_time = None
    end_time = None
    if raw_df is not None and not raw_df.empty and "Timestamp" in raw_df.columns:
        ts_series = pd.to_datetime(raw_df["Timestamp"], errors="coerce").dropna()
        if not ts_series.empty:
            start_time = ts_series.min()
            end_time = ts_series.max()

    if (start_time is None or end_time is None) and summary_df is not None and not summary_df.empty:
        if "first_time" in summary_df.columns:
            first_series = pd.to_datetime(summary_df["first_time"], errors="coerce").dropna()
            if not first_series.empty:
                candidate = first_series.min()
                start_time = candidate if start_time is None else min(start_time, candidate)
        if "last_time" in summary_df.columns:
            last_series = pd.to_datetime(summary_df["last_time"], errors="coerce").dropna()
            if not last_series.empty:
                candidate = last_series.max()
                end_time = candidate if end_time is None else max(end_time, candidate)

    duration_seconds: float | None = None
    if start_time is not None and end_time is not None:
        try:
            delta = pd.to_datetime(end_time) - pd.to_datetime(start_time)
            seconds = float(delta.total_seconds())
        except Exception:  # pragma: no cover - defensive fallback
            seconds = None
        if seconds is not None and seconds > 0:
            duration_seconds = seconds

    if duration_seconds is None:
        result["observation_seconds"] = None
    else:
        result["observation_seconds"] = duration_seconds

    reads_per_second: float | None = None
    if duration_seconds is not None and total_reads is not None:
        reads_per_second = total_reads / duration_seconds if duration_seconds > 0 else None
        if reads_per_second is not None:
            result["reads_per_second"] = reads_per_second
            result["reads_per_minute"] = reads_per_second * 60.0

    epcs_per_minute: float | None = None
    if duration_seconds is not None and duration_seconds > 0 and unique_epcs is not None:
        minutes = duration_seconds / 60.0
        if minutes > 0:
            epcs_per_minute = unique_epcs / minutes
            result["epcs_per_minute"] = epcs_per_minute

    description_parts: list[str] = []
    if mode_index is not None and str(mode_index) != "":
        description_parts.append(f"ModeIndex {mode_index}")

    rate_segments: list[str] = []
    rps_value = result.get("reads_per_second")
    if rps_value is not None:
        rate_segments.append(f"{float(rps_value):.2f} leituras/s")
    rpm_value = result.get("reads_per_minute")
    if rpm_value is not None:
        rate_segments.append(f"{float(rpm_value):.2f} leituras/min")
    epm_value = result.get("epcs_per_minute")
    if epm_value is not None:
        rate_segments.append(f"{float(epm_value):.2f} EPCs/min")

    if rate_segments:
        prefix = description_parts[0] if description_parts else "Indicador de modo"
        result["description"] = f"{prefix} — {', '.join(rate_segments)}"
    elif description_parts:
        result["description"] = description_parts[0]

    return result


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


def filter_logistics_epcs(
    summary_df: pd.DataFrame,
    *,
    prefix: str = LOGISTICS_PREFIX,
) -> pd.DataFrame:
    """Return a copy of ``summary_df`` filtered to logistics EPCs."""

    if summary_df is None or summary_df.empty or "EPC" not in summary_df.columns:
        return pd.DataFrame(columns=getattr(summary_df, "columns", []))
    working = summary_df.copy()
    mask = working["EPC"].astype(str).str.upper().str.startswith(prefix.upper())
    return working.loc[mask].reset_index(drop=True)


def calculate_logistics_duplicate_rate(logistics_df: pd.DataFrame) -> float | None:
    """Return the average duplicate read count per logistics EPC."""

    if logistics_df is None or logistics_df.empty:
        return None
    if "total_reads" not in logistics_df.columns:
        return None
    reads = pd.to_numeric(logistics_df["total_reads"], errors="coerce").dropna()
    if reads.empty:
        return None
    duplicates = reads - 1.0
    duplicates = duplicates.clip(lower=0.0)
    return float(duplicates.mean())


def calculate_logistics_cycle_time(logistics_df: pd.DataFrame) -> float | None:
    """Return the average dwell time in seconds per logistics EPC."""

    if logistics_df is None or logistics_df.empty:
        return None
    if "duration_present" not in logistics_df.columns:
        return None
    durations = pd.to_numeric(logistics_df["duration_present"], errors="coerce").dropna()
    if durations.empty:
        return None
    return float(durations.mean())


def summarize_logistics_attempts(
    logistics_df: pd.DataFrame,
    *,
    attempt_windows: Iterable[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return attempt-level success indicators for logistics EPCs."""

    attempts: list[dict[str, object]] = []
    logistics_df = logistics_df.copy() if logistics_df is not None else pd.DataFrame()
    logistics_df["first_time"] = pd.to_datetime(
        logistics_df.get("first_time"), errors="coerce"
    )
    logistics_df["last_time"] = pd.to_datetime(
        logistics_df.get("last_time"), errors="coerce"
    )

    def _count_attempt(
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        label: str,
    ) -> dict[str, object]:
        if start is not None and pd.isna(start):
            start = None
        if end is not None and pd.isna(end):
            end = None
        mask = pd.Series(True, index=logistics_df.index)
        if start is not None:
            mask &= logistics_df["last_time"].ge(start) | logistics_df["last_time"].isna()
        if end is not None:
            mask &= logistics_df["first_time"].le(end) | logistics_df["first_time"].isna()
        subset = logistics_df.loc[mask]
        tote_count = int(subset.shape[0]) if not subset.empty else 0
        duplicates = calculate_logistics_duplicate_rate(subset)
        return {
            "attempt_id": label,
            "start_time": start,
            "end_time": end,
            "logistics_epcs": tote_count,
            "successful": bool(tote_count > 0),
            "duplicate_reads_avg": duplicates,
        }

    if attempt_windows:
        for idx, window in enumerate(attempt_windows, start=1):
            if not isinstance(window, dict):
                continue
            start_raw = window.get("start") or window.get("start_time")
            end_raw = window.get("end") or window.get("end_time")
            start = pd.to_datetime(start_raw, errors="coerce") if start_raw is not None else None
            end = pd.to_datetime(end_raw, errors="coerce") if end_raw is not None else None
            label = str(window.get("label") or f"Attempt {idx}")
            attempts.append(_count_attempt(start, end, label))
    else:
        if logistics_df.empty:
            start = end = None
        else:
            start = logistics_df["first_time"].min()
            end = logistics_df["last_time"].max()
        attempts.append(_count_attempt(start, end, "Attempt 1"))

    attempts_df = pd.DataFrame(attempts)
    total_attempts = int(attempts_df.shape[0]) if not attempts_df.empty else 0
    successful_attempts = int(attempts_df["successful"].sum()) if not attempts_df.empty else 0
    success_rate = (
        float(successful_attempts) / float(total_attempts) * 100.0
        if total_attempts
        else None
    )
    return {
        "attempts": attempts_df,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "success_rate_pct": success_rate,
    }


def calculate_logistics_spatial_coverage(
    positions_df: pd.DataFrame | None,
    *,
    prefix: str = LOGISTICS_PREFIX,
) -> tuple[float | None, pd.DataFrame]:
    """Return coverage percentage and mask for logistics zones."""

    if positions_df is None or positions_df.empty:
        return None, pd.DataFrame()
    working = positions_df.copy()
    for column in ("ExpectedToken", "ExpectedEPC"):
        if column not in working.columns:
            working[column] = ""
    mask = (
        working["ExpectedToken"].astype(str).str.upper().str.startswith(prefix.upper())
        | working["ExpectedEPC"].astype(str).str.upper().str.startswith(prefix.upper())
    )
    logistics_positions = working.loc[mask]
    if logistics_positions.empty:
        return None, pd.DataFrame(columns=working.columns)
    logistics_positions = logistics_positions.reset_index(drop=True)
    total_positions = int(logistics_positions.shape[0])
    read_positions = int(
        pd.to_numeric(logistics_positions.get("Read"), errors="coerce").fillna(0).astype(int).sum()
    )
    coverage_pct = None
    if total_positions > 0:
        coverage_pct = read_positions / total_positions * 100.0
    return coverage_pct, logistics_positions


def calculate_reader_uptime_from_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    """Return uptime indicators derived from ItemTest metadata."""

    uptime_seconds = None
    scheduled_seconds = None
    if isinstance(metadata, dict):
        raw_uptime = metadata.get("ReaderUptimeSeconds")
        raw_scheduled = metadata.get("ScheduledSessionSeconds") or metadata.get(
            "SessionDurationSeconds"
        )
        if raw_uptime is not None:
            try:
                uptime_seconds = float(raw_uptime)
            except (TypeError, ValueError):
                uptime_seconds = None
            if uptime_seconds is not None and pd.isna(uptime_seconds):
                uptime_seconds = None
        if raw_scheduled is not None:
            try:
                scheduled_seconds = float(raw_scheduled)
            except (TypeError, ValueError):
                scheduled_seconds = None
            if scheduled_seconds is not None and pd.isna(scheduled_seconds):
                scheduled_seconds = None

    uptime_pct = None
    if (
        uptime_seconds is not None
        and scheduled_seconds is not None
        and scheduled_seconds > 0
    ):
        uptime_pct = uptime_seconds / scheduled_seconds * 100.0

    return {
        "uptime_seconds": uptime_seconds,
        "scheduled_seconds": scheduled_seconds,
        "uptime_pct": uptime_pct,
    }


def _collect_expected_logistics_tokens(
    *,
    coverage_mask: pd.DataFrame | None,
    positions_df: pd.DataFrame | None,
    expected_full: Collection[str] | None,
    prefix: str,
) -> set[str]:
    """Return a normalised set of expected logistics EPCs."""

    prefix_upper = prefix.upper()
    expected_tokens: set[str] = set()

    def _ingest(values: Iterable | pd.Series | None) -> None:
        if values is None:
            return
        if isinstance(values, pd.Series):
            iterable = values.tolist()
        else:
            iterable = values
        for value in iterable:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, (list, tuple, set)):
                _ingest(value)
                continue
            text = str(value).strip()
            if not text:
                continue
            token = text.upper()
            if token.startswith(prefix_upper):
                expected_tokens.add(token)

    if coverage_mask is not None and not coverage_mask.empty:
        for column in ("ExpectedEPC", "ExpectedToken"):
            if column in coverage_mask.columns:
                _ingest(coverage_mask[column])

    if not expected_tokens and positions_df is not None and not positions_df.empty:
        for column in ("ExpectedEPC", "ExpectedToken"):
            if column in positions_df.columns:
                _ingest(positions_df[column])

    if expected_full:
        _ingest(expected_full)

    return expected_tokens


def compile_logistics_kpis(
    summary_df: pd.DataFrame,
    metadata: dict[str, object] | None,
    *,
    positions_df: pd.DataFrame | None = None,
    continuous_details: dict[str, object] | None = None,
    attempt_windows: Iterable[dict[str, object]] | None = None,
    expected_full: Collection[str] | None = None,
    prefix: str = LOGISTICS_PREFIX,
) -> dict[str, object]:
    """Aggregate logistics KPIs combining structured and continuous artefacts."""

    logistics_df = filter_logistics_epcs(summary_df, prefix=prefix)
    total_totes = int(logistics_df.shape[0]) if not logistics_df.empty else 0

    duplicate_rate = calculate_logistics_duplicate_rate(logistics_df)
    cycle_time = calculate_logistics_cycle_time(logistics_df)

    attempts_info = summarize_logistics_attempts(
        logistics_df, attempt_windows=attempt_windows
    )
    success_rate = attempts_info.get("success_rate_pct")
    failure_rate = None
    if success_rate is not None:
        failure_rate = max(0.0, 100.0 - float(success_rate))

    coverage_pct, coverage_mask = calculate_logistics_spatial_coverage(
        positions_df, prefix=prefix
    )

    observed_tokens: set[str] = set()
    if not logistics_df.empty and "EPC" in logistics_df.columns:
        observed_tokens = set(
            logistics_df["EPC"].dropna().astype(str).str.upper().tolist()
        )
    observed_count = int(len(observed_tokens))

    expected_tokens = _collect_expected_logistics_tokens(
        coverage_mask=coverage_mask,
        positions_df=positions_df,
        expected_full=expected_full,
        prefix=prefix,
    )
    expected_count = int(len(expected_tokens))

    read_rate_pct: float | None = None
    if expected_count > 0:
        read_rate_pct = observed_count / expected_count * 100.0

    missed_tokens: list[str] = []
    if expected_tokens:
        missed_tokens = sorted(expected_tokens.difference(observed_tokens))

    logistics_concurrency_peak = None
    logistics_concurrency_average = None
    logistics_concurrency_peak_time = None
    logistics_cycle_time_avg = cycle_time
    per_tote_summary: pd.DataFrame | None = None
    concurrency_timeline: pd.DataFrame | None = None

    if continuous_details:
        peak_value = continuous_details.get("logistics_concurrency_peak")
        if peak_value is not None and not pd.isna(peak_value):
            try:
                logistics_concurrency_peak = int(peak_value)
            except (TypeError, ValueError):
                logistics_concurrency_peak = None
        avg_value = continuous_details.get("logistics_concurrency_average")
        if avg_value is not None and not pd.isna(avg_value):
            logistics_concurrency_average = float(avg_value)
        peak_time_value = continuous_details.get("logistics_concurrency_peak_time")
        if peak_time_value is not None:
            try:
                peak_time_candidate = pd.to_datetime(peak_time_value)
                if not pd.isna(peak_time_candidate):
                    logistics_concurrency_peak_time = peak_time_candidate
            except Exception:  # pragma: no cover - defensive conversion
                logistics_concurrency_peak_time = None
        cycle_value = continuous_details.get("logistics_cycle_time_average")
        if cycle_value is not None and not pd.isna(cycle_value):
            logistics_cycle_time_avg = float(cycle_value)
        per_tote_candidate = continuous_details.get("logistics_per_tote_summary")
        if isinstance(per_tote_candidate, pd.DataFrame):
            per_tote_summary = per_tote_candidate.copy()
        elif per_tote_candidate is not None:
            per_tote_summary = pd.DataFrame(per_tote_candidate)
        timeline_candidate = continuous_details.get("logistics_concurrency_timeline")
        if isinstance(timeline_candidate, pd.DataFrame):
            concurrency_timeline = timeline_candidate.copy()
        elif timeline_candidate is not None:
            concurrency_timeline = pd.DataFrame(timeline_candidate)

    uptime_info = calculate_reader_uptime_from_metadata(metadata)

    result = {
        "prefix": prefix,
        "total_logistics_epcs": total_totes,
        "duplicate_reads_per_tote": duplicate_rate,
        "tote_cycle_time_seconds": logistics_cycle_time_avg,
        "attempt_success_rate_pct": success_rate,
        "attempt_failure_rate_pct": failure_rate,
        "attempts_table": attempts_info.get("attempts"),
        "total_attempts": attempts_info.get("total_attempts"),
        "successful_attempts": attempts_info.get("successful_attempts"),
        "coverage_pct": coverage_pct,
        "coverage_table": coverage_mask,
        "expected_logistics_epcs": sorted(expected_tokens),
        "expected_logistics_epcs_count": expected_count,
        "observed_logistics_epcs_count": observed_count,
        "logistics_read_rate_pct": read_rate_pct,
        "missed_logistics_epcs": missed_tokens,
        "missed_logistics_epcs_count": len(missed_tokens),
        "concurrent_capacity": logistics_concurrency_peak,
        "concurrent_capacity_avg": logistics_concurrency_average,
        "concurrent_capacity_time": logistics_concurrency_peak_time,
        "logistics_per_tote_summary": per_tote_summary,
        "logistics_concurrency_timeline": concurrency_timeline,
        "reader_uptime_pct": uptime_info.get("uptime_pct"),
        "reader_uptime_seconds": uptime_info.get("uptime_seconds"),
        "scheduled_session_seconds": uptime_info.get("scheduled_seconds"),
    }

    return result


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
    "calculate_mode_performance",
    "calculate_layout_face_coverage",
    "calculate_layout_row_coverage",
    "detect_read_hotspots",
    "calculate_frequency_usage",
    "detect_location_errors",
    "calculate_read_distribution_by_face",
    "compile_structured_kpis",
    "filter_logistics_epcs",
    "calculate_logistics_duplicate_rate",
    "calculate_logistics_cycle_time",
    "summarize_logistics_attempts",
    "calculate_logistics_spatial_coverage",
    "calculate_reader_uptime_from_metadata",
    "compile_logistics_kpis",
]
