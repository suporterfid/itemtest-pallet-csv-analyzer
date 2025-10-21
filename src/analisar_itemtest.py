# -*- coding: utf-8 -*-
"""Utilities for composing textual reports for ItemTest analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .pallet_layout import ROW_COLUMN


def build_continuous_alerts(
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
                f"EPCs with atypical dwell time ({len(anomalous_list)}): {sample}{suffix}"
            )

    flag_labels = {
        "epcs_only_top_antennas": "EPCs restricted to upper antennas",
        "epcs_without_antenna": "EPCs without an identified antenna",
        "invalid_data": "Invalid data encountered",
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
    if "expected_epc" in summary.columns:
        expected_count = int(summary["expected_epc"].sum())
        unexpected_count = int((~summary["expected_epc"]).sum())

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
        metadata_lines.append(f"- Session: {session}")
    inventory_mode = metadata.get("InventoryMode")
    if inventory_mode:
        metadata_lines.append(f"- InventoryMode: {inventory_mode}")
    antennas = metadata.get("AntennaIDs")
    if antennas:
        antenna_list = ", ".join(str(a) for a in antennas)
        metadata_lines.append(f"- Declared antennas: {antenna_list}")
    powers = metadata.get("PowersInDbm")
    if isinstance(powers, dict) and powers:
        formatted_powers = []
        for ant_id, power in sorted(powers.items()):
            if isinstance(power, (int, float)):
                formatted_powers.append(f"Antenna {ant_id}: {power:.1f} dBm")
            else:
                formatted_powers.append(f"Antenna {ant_id}: {power}")
        metadata_lines.append("- Declared powers: " + "; ".join(formatted_powers))
    elif powers:
        metadata_lines.append(f"- Declared powers: {powers}")
    if not metadata_lines:
        metadata_lines.append("- No relevant metadata found.")

    mode_label = "Continuous" if analysis_mode == "continuous" else "Structured"
    general_lines: list[str] = [f"- Analysis mode: {mode_label}"]
    if expected_count is not None and unexpected_count is not None:
        general_lines.append(
            f"- Unique EPCs: {total_epcs} (expected: {expected_count}, unexpected: {unexpected_count})"
        )
    else:
        general_lines.append(f"- Unique EPCs: {total_epcs}")
    general_lines.append(f"- Total reads: {total_reads}")
    if first_seen and last_seen:
        general_lines.append(f"- Reading window: {first_seen} → {last_seen}")
    elif first_seen:
        general_lines.append(f"- First read recorded at {first_seen}")
    elif last_seen:
        general_lines.append(f"- Last read recorded at {last_seen}")

    continuous_lines: list[str] = []
    if analysis_mode == "continuous":
        details = continuous_details or {}
        average_dwell = details.get("average_dwell_seconds")
        if average_dwell is not None and not pd.isna(average_dwell):
            continuous_lines.append(
                f"- Tempo médio de permanência: {float(average_dwell):.2f} s"
            )
        else:
            continuous_lines.append("- Tempo médio de permanência: não disponível")

        total_events = details.get("total_events")
        if total_events is not None and not pd.isna(total_events):
            continuous_lines.append(
                f"- Eventos de entrada/saída detectados: {int(total_events)}"
            )
        else:
            continuous_lines.append("- Eventos de entrada/saída detectados: não disponível")

        dominant = details.get("dominant_antenna")
        if dominant is not None and not (
            isinstance(dominant, float) and pd.isna(dominant)
        ) and str(dominant) != "":
            try:
                dom_display = int(dominant)
            except (TypeError, ValueError):
                dom_display = dominant
            continuous_lines.append(f"- Antena dominante: {dom_display}")
        else:
            continuous_lines.append("- Antena dominante: não identificada")

        mean_epcs = details.get("epcs_per_minute_mean")
        if mean_epcs is not None and not pd.isna(mean_epcs):
            continuous_lines.append(
                f"- EPCs ativos (média por minuto): {float(mean_epcs):.2f}"
            )

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
                peak_repr = f"{int(peak_value)} at {peak_label}"
            else:
                peak_repr = str(int(peak_value))
            continuous_lines.append(
                f"- Pico de EPCs ativos por minuto: {peak_repr}"
            )

        alerts = details.get("alerts")
        if alerts is None:
            alerts = build_continuous_alerts(
                details.get("anomalous_epcs"),
                details.get("inconsistency_flags"),
            )
        alerts = [alert for alert in alerts if alert]
        if alerts:
            for alert in alerts:
                continuous_lines.append(f"- Alerta: {alert}")
        else:
            continuous_lines.append("- Nenhum alerta identificado para o modo contínuo.")

    antenna_lines: list[str] = []
    if ant_counts is not None and not ant_counts.empty:
        for row in ant_counts.itertuples(index=False):
            antenna_id = getattr(row, "Antenna", "?")
            reads = getattr(row, "total_reads", 0)
            line = f"- Antenna {antenna_id}: {reads} reads"
            participation = getattr(row, "participation_pct", None)
            if participation is not None and not pd.isna(participation):
                line += f" ({participation:.1f}%)"
            rssi_avg = getattr(row, "rssi_avg", None)
            if rssi_avg is not None and not pd.isna(rssi_avg):
                line += f", average RSSI {rssi_avg:.1f} dBm"
            antenna_lines.append(line)
    else:
        antenna_lines.append("- No antenna aggregate reads available.")

    layout_lines: list[str] = []
    if positions_df is None:
        layout_lines.append("- Layout not provided.")
    elif positions_df.empty:
        layout_lines.append("- Layout provided but no positions defined.")
    else:
        total_positions = int(len(positions_df))
        read_positions = int(positions_df["Read"].sum())
        coverage_pct = (read_positions / total_positions * 100) if total_positions else 0.0
        layout_lines.append(
            f"- Layout coverage: {read_positions} of {total_positions} positions ({coverage_pct:.1f}%)"
        )
        missing = positions_df[~positions_df["Read"]]
        if not missing.empty:
            missing_records = missing[["Face", ROW_COLUMN, "Suffix"]].drop_duplicates()
            descriptors: list[str] = []
            for row in missing_records.itertuples(index=False):
                face = getattr(row, "Face")
                row_label = getattr(row, ROW_COLUMN)
                suffix = getattr(row, "Suffix")
                descriptors.append(f"{face} - Row {row_label} ({suffix})")
                if len(descriptors) == 5:
                    break
            extra = " ..." if len(missing_records) > 5 else ""
            layout_lines.append(
                "- Positions without reads ("
                + f"{len(missing_records)}): "
                + "; ".join(descriptors)
                + extra
            )
        else:
            layout_lines.append("- All layout positions were covered by reads.")

    sections = [
        ("Key metadata", metadata_lines),
        ("General indicators", general_lines),
    ]
    if analysis_mode == "continuous":
        if not continuous_lines:
            continuous_lines = ["- No additional indicators available."]
        sections.append(("Continuous mode indicators", continuous_lines))
    sections.extend(
        [
            ("Reads by antenna", antenna_lines),
            ("Layout coverage", layout_lines),
        ]
    )

    header = f"ItemTest Summary — {csv_path.name}"
    divider = "=" * len(header)
    lines: list[str] = [header, divider]
    for title, content in sections:
        lines.append(title + ":")
        lines.extend(content)
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["compose_summary_text", "build_continuous_alerts"]
