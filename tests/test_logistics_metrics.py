# -*- coding: utf-8 -*-
"""Unit tests covering logistics KPI helper functions."""

import pandas as pd
import pytest

from src.metrics import compile_logistics_kpis


def test_compile_logistics_kpis_with_metadata_and_continuous_details() -> None:
    summary = pd.DataFrame(
        {
            "EPC": [
                "331A00000000000000000001",
                "331A00000000000000000002",
                "AAABBB000000000000000003",
            ],
            "total_reads": [12, 8, 4],
            "duration_present": [30.0, 50.0, 12.0],
            "first_time": [
                pd.Timestamp("2025-01-01 12:00:00"),
                pd.Timestamp("2025-01-01 12:01:00"),
                pd.Timestamp("2025-01-01 12:02:00"),
            ],
            "last_time": [
                pd.Timestamp("2025-01-01 12:05:00"),
                pd.Timestamp("2025-01-01 12:03:30"),
                pd.Timestamp("2025-01-01 12:02:30"),
            ],
        }
    )
    metadata = {"ReaderUptimeSeconds": 5400, "ScheduledSessionSeconds": 6000}
    positions_df = pd.DataFrame(
        {
            "Face": ["Left", "Front", "Right"],
            "ExpectedToken": [
                "331A00000000000000000001",
                "331A00000000000000000002",
                "998877000000000000000000",
            ],
            "Read": [True, False, True],
        }
    )
    continuous_details = {
        "logistics_concurrency_peak": 3,
        "logistics_concurrency_peak_time": pd.Timestamp("2025-01-01 12:02:00"),
        "logistics_concurrency_average": 1.75,
        "logistics_cycle_time_average": 40.0,
        "logistics_per_tote_summary": pd.DataFrame(
            {
                "EPC": ["331A00000000000000000001"],
                "duration_present": [30.0],
            }
        ),
        "logistics_concurrency_timeline": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01 12:02:00")],
                "active_epcs": [2],
                "duration_seconds": [60.0],
            }
        ),
    }

    result = compile_logistics_kpis(
        summary,
        metadata,
        positions_df=positions_df,
        continuous_details=continuous_details,
    )

    assert result["total_logistics_epcs"] == 2
    assert result["attempt_success_rate_pct"] == 100.0
    assert result["attempt_failure_rate_pct"] == 0.0
    assert result["duplicate_reads_per_tote"] == ((12 - 1) + (8 - 1)) / 2
    assert result["coverage_pct"] == 50.0
    assert result["concurrent_capacity"] == 3
    assert result["concurrent_capacity_avg"] == 1.75
    assert result["reader_uptime_pct"] == pytest.approx(90.0)
    assert isinstance(result["coverage_table"], pd.DataFrame)
    assert not result["coverage_table"].empty


def test_logistics_kpis_without_logistics_epcs() -> None:
    summary = pd.DataFrame({"EPC": ["ABC123"], "total_reads": [5]})
    result = compile_logistics_kpis(summary, {})
    assert result["total_logistics_epcs"] == 0
    assert result["attempt_success_rate_pct"] == 0.0
    assert result["attempt_failure_rate_pct"] == 100.0
