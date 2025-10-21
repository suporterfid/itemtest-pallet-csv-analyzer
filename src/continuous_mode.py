# -*- coding: utf-8 -*-
"""Utilities for analysing continuous RFID reading streams."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable

import pandas as pd

LOGGER = logging.getLogger("itemtest.continuous")


@dataclass(slots=True)
class ContinuousFlowResult:
    """Container for artefacts produced by :func:`analyze_continuous_flow`."""

    per_epc_summary: pd.DataFrame
    epc_timeline: pd.DataFrame
    epcs_per_minute: pd.Series
    average_dwell_seconds: float | None
    anomalous_epcs: list[str]
    inconsistency_flags: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the analysis result."""

        return {
            "per_epc_summary": self.per_epc_summary.to_dict(orient="records"),
            "epc_timeline": self.epc_timeline.to_dict(orient="records"),
            "epcs_per_minute": self.epcs_per_minute.to_dict(),
            "average_dwell_seconds": self.average_dwell_seconds,
            "anomalous_epcs": list(self.anomalous_epcs),
            "inconsistency_flags": {
                key: list(values) for key, values in self.inconsistency_flags.items()
            },
        }


def analyze_continuous_flow(
    df: pd.DataFrame,
    window_seconds: float,
) -> ContinuousFlowResult:
    """Analyse continuous reads identifying dwell intervals and directional hints.

    Parameters
    ----------
    df:
        DataFrame containing at least the columns ``EPC``, ``Timestamp`` and
        ``Antenna``.
    window_seconds:
        Maximum allowed gap (in seconds) between successive reads of the same EPC
        to consider that it is still present in the read field.

    Returns
    -------
    ContinuousFlowResult
        Object with per-EPC statistics, timeline intervals, activity per minute,
        anomalous dwell detections and inconsistency flags for alerting.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive for continuous analysis.")

    if df.empty:
        empty_frame = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")
        return ContinuousFlowResult(
            per_epc_summary=empty_frame,
            epc_timeline=empty_frame,
            epcs_per_minute=empty_series,
            average_dwell_seconds=None,
            anomalous_epcs=[],
            inconsistency_flags={},
        )

    required_columns = {"EPC", "Timestamp", "Antenna"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "DataFrame for continuous analysis is missing required columns: "
            + ", ".join(sorted(missing))
        )

    window_td = pd.to_timedelta(window_seconds, unit="s")

    working = df.copy()
    working["Timestamp"] = pd.to_datetime(working["Timestamp"], errors="coerce")
    invalid_ts = working["Timestamp"].isna()
    if invalid_ts.any():
        LOGGER.warning(
            "Ignoring %s reads with invalid Timestamp during continuous analysis.",
            int(invalid_ts.sum()),
        )
        working = working.loc[~invalid_ts]

    if working.empty:
        empty_frame = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")
        return ContinuousFlowResult(
            per_epc_summary=empty_frame,
            epc_timeline=empty_frame,
            epcs_per_minute=empty_series,
            average_dwell_seconds=None,
            anomalous_epcs=[],
            inconsistency_flags={"invalid_data": ["No valid timestamps"]},
        )

    working = working.sort_values(["EPC", "Timestamp", "Antenna"]).reset_index(drop=True)

    per_epc_records: list[dict[str, Any]] = []
    timeline_records: list[dict[str, Any]] = []
    per_epc_antennas: dict[str, set[int]] = {}

    for epc, group in working.groupby("EPC", sort=False):
        sorted_group = group.sort_values("Timestamp")

        time_diffs = sorted_group["Timestamp"].diff().dt.total_seconds()
        # ``True`` when a new interval starts because the gap exceeded the window
        new_interval = time_diffs.gt(window_seconds).cumsum()

        interval_groups = sorted_group.groupby(new_interval)
        intervals = interval_groups["Timestamp"].agg(["first", "last"])
        intervals = intervals.rename(columns={"first": "entry", "last": "exit"})
        intervals["exit"] = intervals["exit"] + window_td
        intervals["duration"] = (
            intervals["exit"] - intervals["entry"]
        ).dt.total_seconds()
        interval_read_counts = interval_groups.size()

        antenna_numeric = pd.to_numeric(sorted_group["Antenna"], errors="coerce")
        invalid_antenna_count = int(antenna_numeric.isna().sum())
        if invalid_antenna_count:
            LOGGER.debug(
                "EPC %s has %s reads with invalid antenna IDs removed from distribution.",
                epc,
                invalid_antenna_count,
            )
        antenna_series = antenna_numeric.dropna().astype(int, copy=False)
        per_epc_antennas[epc] = set(antenna_series.tolist())

        antenna_distribution = (
            antenna_series.value_counts(normalize=True).sort_index() * 100
            if not antenna_series.empty
            else pd.Series(dtype="float64")
        )

        duration_present = float(intervals["duration"].sum()) if not intervals.empty else 0.0
        read_events = int(intervals.shape[0]) if not intervals.empty else 0
        first_seen = intervals["entry"].iloc[0] if not intervals.empty else None
        last_seen = intervals["exit"].iloc[-1] - window_td if not intervals.empty else None

        initial_antenna = _first_valid_antenna(sorted_group["Antenna"].tolist())
        final_antenna = _first_valid_antenna(reversed(sorted_group["Antenna"].tolist()))
        direction_estimate = _infer_direction(sorted_group["Antenna"].tolist())

        for event_idx, row in enumerate(
            intervals[["entry", "exit", "duration"]].itertuples(index=True, name="Interval"),
            start=1,
        ):
            entry = row.entry
            exit_ = row.exit
            duration = row.duration
            label = row.Index
            timeline_records.append(
                {
                    "EPC": epc,
                    "event_index": event_idx,
                    "entry_time": entry,
                    "exit_time": exit_,
                    "duration_seconds": float(duration),
                    "read_count": int(interval_read_counts.get(label, 0)),
                }
            )

        per_epc_records.append(
            {
                "EPC": epc,
                "first_time": first_seen,
                "last_time": last_seen,
                "duration_present": duration_present,
                "total_reads": int(sorted_group.shape[0]),
                "read_events": read_events,
                "antenna_distribution": (
                    antenna_distribution.round(2).to_dict() if not antenna_distribution.empty else {}
                ),
                "initial_antenna": initial_antenna,
                "final_antenna": final_antenna,
                "direction_estimate": direction_estimate,
            }
        )

    per_epc_summary = pd.DataFrame(per_epc_records)
    if not per_epc_summary.empty:
        per_epc_summary = per_epc_summary.sort_values(
            by=["first_time", "EPC"], na_position="last"
        ).reset_index(drop=True)

    epc_timeline = pd.DataFrame(timeline_records)
    if not epc_timeline.empty:
        epc_timeline = epc_timeline.sort_values(
            by=["entry_time", "EPC", "event_index"]
        ).reset_index(drop=True)

    epcs_per_minute = (
        working.assign(_minute=working["Timestamp"].dt.floor("T"))
        .groupby("_minute")["EPC"]
        .nunique()
    )
    epcs_per_minute.name = "unique_epcs"

    average_dwell_seconds = (
        float(per_epc_summary["duration_present"].mean())
        if not per_epc_summary.empty
        else None
    )

    anomalous_epcs = _detect_anomalous_durations(per_epc_summary, window_seconds)
    inconsistency_flags = _detect_inconsistencies(per_epc_summary, per_epc_antennas)

    return ContinuousFlowResult(
        per_epc_summary=per_epc_summary,
        epc_timeline=epc_timeline,
        epcs_per_minute=epcs_per_minute,
        average_dwell_seconds=average_dwell_seconds,
        anomalous_epcs=anomalous_epcs,
        inconsistency_flags=inconsistency_flags,
    )


def _first_valid_antenna(values: Iterable[Any]) -> int | None:
    """Return the first non-null antenna value cast to ``int`` if possible."""

    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                continue
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return None


def _infer_direction(sequence: Iterable[Any]) -> str:
    """Infer movement direction based on antenna sequence trends."""

    cleaned = [
        _first_valid_antenna([value])
        for value in sequence
    ]
    cleaned = [value for value in cleaned if value is not None]
    if len(cleaned) < 2:
        return "unknown"

    deltas = [b - a for a, b in zip(cleaned, cleaned[1:])]
    positives = sum(delta > 0 for delta in deltas)
    negatives = sum(delta < 0 for delta in deltas)

    if positives and not negatives:
        return "forward"
    if negatives and not positives:
        return "reverse"
    if positives and negatives:
        return "mixed"
    return "static"


def _detect_anomalous_durations(
    per_epc_summary: pd.DataFrame, window_seconds: float
) -> list[str]:
    """Return EPCs whose dwell time deviates strongly from the sample."""

    if per_epc_summary.empty or "duration_present" not in per_epc_summary:
        return []

    durations = per_epc_summary["duration_present"].astype(float)
    if durations.empty:
        return []

    q1 = durations.quantile(0.25)
    q3 = durations.quantile(0.75)
    iqr = q3 - q1

    if iqr > 0:
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
    else:
        mean_val = durations.mean()
        lower = max(0.0, mean_val - window_seconds)
        upper = mean_val + (2 * window_seconds)

    mask = (durations < lower) | (durations > upper)
    anomalous = per_epc_summary.loc[mask, "EPC"].astype(str).tolist()
    if anomalous:
        LOGGER.info(
            "Detected %s EPC(s) with anomalous dwell time (bounds %.2fs - %.2fs)",
            len(anomalous),
            lower,
            upper,
        )
    return anomalous


def _detect_inconsistencies(
    per_epc_summary: pd.DataFrame, per_epc_antennas: dict[str, set[int]]
) -> dict[str, list[str]]:
    """Return a dictionary with lists of EPCs triggering inconsistency flags."""

    if per_epc_summary.empty:
        return {}

    inconsistency_flags: dict[str, list[str]] = {
        "epcs_only_top_antennas": [],
        "epcs_without_antenna": [],
    }

    all_antennas = sorted({a for antennas in per_epc_antennas.values() for a in antennas})
    if all_antennas:
        cutoff_index = max(0, len(all_antennas) - max(1, len(all_antennas) // 2))
        top_antennas = set(all_antennas[cutoff_index:])
    else:
        top_antennas = set()

    for epc, antennas in per_epc_antennas.items():
        if not antennas:
            inconsistency_flags["epcs_without_antenna"].append(epc)
            continue
        if top_antennas and antennas.issubset(top_antennas) and len(top_antennas) < len(all_antennas):
            inconsistency_flags["epcs_only_top_antennas"].append(epc)

    return {key: values for key, values in inconsistency_flags.items() if values}


__all__ = [
    "ContinuousFlowResult",
    "analyze_continuous_flow",
]

