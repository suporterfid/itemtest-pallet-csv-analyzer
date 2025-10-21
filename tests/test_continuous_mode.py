"""Tests for continuous-mode analytics metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.continuous_mode import analyze_continuous_flow


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

    assert result.session_duration_seconds == pytest.approx(6.0, rel=1e-6)
    assert result.session_active_seconds == pytest.approx(6.0, rel=1e-6)


def test_analyze_continuous_flow_sparse_reads() -> None:
    """Validate metrics when reads are sparse and broken into multiple intervals."""

    sparse_df = pd.DataFrame(
        [
            {"EPC": "TAG1", "Timestamp": "2024-01-01T08:00:00", "Antenna": 1, "RSSI": -40},
            {"EPC": "TAG1", "Timestamp": "2024-01-01T08:00:10", "Antenna": 1, "RSSI": -41},
        ]
    )

    result = analyze_continuous_flow(sparse_df, window_seconds=2)

    assert len(result.epc_timeline) == 2
    assert int(result.per_epc_summary.iloc[0]["read_events"]) == 2
    assert pytest.approx(result.read_continuity_rate or 0.0, rel=1e-6) == pytest.approx(33.3333333, rel=1e-6)
    assert pytest.approx(result.concurrency_average or 0.0, rel=1e-6) == pytest.approx(1 / 3, rel=1e-6)
    assert result.session_duration_seconds == pytest.approx(12.0, rel=1e-6)
    assert result.session_active_seconds == pytest.approx(4.0, rel=1e-6)
