"""Tests for continuous-mode analytics metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.continuous_mode import analyze_continuous_flow
from src.metrics import calculate_mode_performance


@pytest.fixture
def sample_continuous_dataframe() -> pd.DataFrame:
    """Return a simple continuous-mode dataset with overlapping EPC activity."""

    return pd.DataFrame(
        [
            {"EPC": "EPC1", "Timestamp": "2024-01-01T10:00:00", "Antenna": 1, "RSSI": -50},
            {"EPC": "EPC1", "Timestamp": "2024-01-01T10:00:01", "Antenna": 1, "RSSI": -51},
            {"EPC": "EPC1", "Timestamp": "2024-01-01T10:00:03", "Antenna": 2, "RSSI": -52},
            {"EPC": "EPC2", "Timestamp": "2024-01-01T10:00:02", "Antenna": 3, "RSSI": -55},
            {"EPC": "EPC2", "Timestamp": "2024-01-01T10:00:04", "Antenna": 3, "RSSI": -56},
        ]
    )


def test_analyze_continuous_flow_metrics(sample_continuous_dataframe: pd.DataFrame) -> None:
    """Ensure key metrics are computed for overlapping tag intervals."""

    result = analyze_continuous_flow(sample_continuous_dataframe, window_seconds=2)

    assert "rssi_std" in result.per_epc_summary.columns
    epc1_row = result.per_epc_summary.set_index("EPC").loc["EPC1"]
    assert pytest.approx(epc1_row["rssi_std"], rel=1e-6) == pytest.approx(0.816496580927726, rel=1e-6)

    assert not result.concurrency_timeline.empty
    assert int(result.concurrency_peak or 0) == 2
    assert pytest.approx(result.concurrency_average or 0.0, rel=1e-6) == pytest.approx(1.5, rel=1e-6)

    assert pytest.approx(result.read_continuity_rate or 0.0, rel=1e-6) == pytest.approx(100.0, rel=1e-6)
    assert pytest.approx(result.throughput_per_minute or 0.0, rel=1e-6) == pytest.approx(20.0, rel=1e-6)
    assert pytest.approx(result.session_throughput or 0.0, rel=1e-6) == pytest.approx(50.0, rel=1e-6)
    assert not result.reads_per_minute.empty
    assert int(result.reads_per_minute.iloc[0]) == 5
    assert int(result.reads_per_minute.sum()) == 5

    assert result.session_duration_seconds == pytest.approx(6.0, rel=1e-6)
    assert result.session_active_seconds == pytest.approx(6.0, rel=1e-6)
    assert result.tag_dwell_time_max == pytest.approx(5.0, rel=1e-6)
    assert result.inactive_periods_count == 0
    assert result.inactive_periods.empty
    assert result.inactive_total_seconds == pytest.approx(0.0, rel=1e-6)
    assert result.inactive_longest_seconds is None
    assert pytest.approx(result.congestion_index or 0.0, rel=1e-6) == pytest.approx(5 / 6, rel=1e-6)
    assert pytest.approx(result.global_rssi_avg or 0.0, rel=1e-6) == pytest.approx(-52.8, rel=1e-6)
    assert pytest.approx(result.global_rssi_std or 0.0, rel=1e-6) == pytest.approx(2.315167380558045, rel=1e-6)


def test_analyze_continuous_flow_sparse_reads() -> None:
    """Validate metrics when reads are sparse and broken into multiple intervals."""

    sparse_df = pd.DataFrame(
        [
            {"EPC": "TAG1", "Timestamp": "2024-01-01T08:00:00", "Antenna": 1, "RSSI": -40},
            {"EPC": "TAG1", "Timestamp": "2024-01-01T08:00:14", "Antenna": 1, "RSSI": -41},
        ]
    )

    result = analyze_continuous_flow(sparse_df, window_seconds=2)

    assert len(result.epc_timeline) == 2
    assert int(result.per_epc_summary.iloc[0]["read_events"]) == 2
    assert pytest.approx(result.read_continuity_rate or 0.0, rel=1e-6) == pytest.approx(25.0, rel=1e-6)
    assert pytest.approx(result.concurrency_average or 0.0, rel=1e-6) == pytest.approx(0.25, rel=1e-6)
    assert result.session_duration_seconds == pytest.approx(16.0, rel=1e-6)
    assert result.session_active_seconds == pytest.approx(4.0, rel=1e-6)
    assert result.tag_dwell_time_max == pytest.approx(2.0, rel=1e-6)
    assert result.inactive_periods_count == 1
    assert not result.inactive_periods.empty
    first_idle = result.inactive_periods.iloc[0]
    assert pytest.approx(first_idle["duration_seconds"], rel=1e-6) == pytest.approx(12.0, rel=1e-6)
    assert pytest.approx(first_idle["gap_multiplier"], rel=1e-6) == pytest.approx(7.0, rel=1e-6)
    assert pytest.approx(result.inactive_total_seconds or 0.0, rel=1e-6) == pytest.approx(12.0, rel=1e-6)
    assert pytest.approx(result.congestion_index or 0.0, rel=1e-6) == pytest.approx(0.5, rel=1e-6)
    assert pytest.approx(result.session_throughput or 0.0, rel=1e-6) == pytest.approx(7.5, rel=1e-6)
    assert not result.reads_per_minute.empty
    assert int(result.reads_per_minute.iloc[0]) == 2
    assert int(result.reads_per_minute.sum()) == 2
    assert pytest.approx(result.global_rssi_avg or 0.0, rel=1e-6) == pytest.approx(-40.5, rel=1e-6)
    assert pytest.approx(result.global_rssi_std or 0.0, rel=1e-6) == pytest.approx(0.5, rel=1e-6)


def test_calculate_mode_performance_continuous(sample_continuous_dataframe: pd.DataFrame) -> None:
    """Mode performance helper should compute rates for continuous-mode reads."""

    metadata = {"ModeIndex": "7"}
    result = analyze_continuous_flow(sample_continuous_dataframe, window_seconds=2)
    indicator = calculate_mode_performance(
        metadata,
        result.per_epc_summary,
        sample_continuous_dataframe,
    )

    assert indicator["mode_index"] == 7
    assert indicator["description"] and "ModeIndex 7" in indicator["description"]

    duration_seconds = (
        pd.to_datetime(sample_continuous_dataframe["Timestamp"]).max()
        - pd.to_datetime(sample_continuous_dataframe["Timestamp"]).min()
    ).total_seconds()
    expected_reads_per_second = sample_continuous_dataframe.shape[0] / duration_seconds
    expected_epcs_per_minute = (
        sample_continuous_dataframe["EPC"].nunique() / (duration_seconds / 60.0)
    )

    assert pytest.approx(indicator["reads_per_second"], rel=1e-6) == pytest.approx(
        expected_reads_per_second, rel=1e-6
    )
    assert pytest.approx(indicator["reads_per_minute"], rel=1e-6) == pytest.approx(
        expected_reads_per_second * 60.0, rel=1e-6
    )
    assert pytest.approx(indicator["epcs_per_minute"], rel=1e-6) == pytest.approx(
        expected_epcs_per_minute, rel=1e-6
    )


def test_calculate_mode_performance_continuous_without_metadata(
    sample_continuous_dataframe: pd.DataFrame,
) -> None:
    """When ModeIndex is missing the helper should return empty indicators."""

    result = analyze_continuous_flow(sample_continuous_dataframe, window_seconds=2)
    indicator = calculate_mode_performance(
        {},
        result.per_epc_summary,
        sample_continuous_dataframe,
    )

    assert indicator["mode_index"] is None
    assert indicator["reads_per_second"] is None
    assert indicator["reads_per_minute"] is None
    assert indicator["epcs_per_minute"] is None
    assert indicator["description"] is None
