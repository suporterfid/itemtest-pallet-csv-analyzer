# -*- coding: utf-8 -*-
"""Validate generation of the RSSI vs Frequency scatter plot artifacts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # pragma: no cover - ensure headless backend for matplotlib

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from src import itemtest_analyzer


@pytest.fixture
def _stub_write_excel(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Avoid creating Excel workbooks during plot generation tests."""

    def _noop_write_excel(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(itemtest_analyzer, "write_excel", _noop_write_excel)
    return _noop_write_excel


@pytest.fixture
def structured_dataframe() -> pd.DataFrame:
    """Return a minimal structured-mode dataset with valid frequency readings."""

    timestamps = pd.date_range("2025-01-01T12:00:00Z", periods=4, freq="s")
    return pd.DataFrame(
        {
            "EPC": [
                "303132333435363738394142",
                "303132333435363738394142",
                "303132333435363738394143",
                "303132333435363738394144",
            ],
            "Timestamp": timestamps,
            "Antenna": [1, 1, 2, 2],
            "RSSI": [-51.2, -50.4, -60.1, -59.5],
            "Frequency": [915.25, 915.5, 915.75, 0.0],
        }
    )


@pytest.fixture
def continuous_dataframe() -> pd.DataFrame:
    """Return a small dataset compatible with the continuous analysis pipeline."""

    timestamps = pd.to_datetime(
        [
            "2025-01-01T08:00:00Z",
            "2025-01-01T08:00:01Z",
            "2025-01-01T08:00:03Z",
            "2025-01-01T08:00:04Z",
        ]
    )
    return pd.DataFrame(
        {
            "EPC": [
                "303132333435363738394145",
                "303132333435363738394145",
                "303132333435363738394146",
                "303132333435363738394146",
            ],
            "Timestamp": timestamps,
            "Antenna": [1, 1, 2, 2],
            "RSSI": [-55.0, -54.5, -60.0, -59.0],
            "Frequency": [915.25, 915.5, 915.5, 915.75],
        }
    )


def test_process_file_generates_rssi_frequency_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    structured_dataframe: pd.DataFrame,
    _stub_write_excel: Callable[..., None],
) -> None:
    """Ensure the structured pipeline persists the RSSI vs Frequency chart."""

    csv_path = tmp_path / "structured.csv"
    csv_path.write_text("placeholder", encoding="utf-8")

    metadata = {"Hostname": "test-reader"}

    def _fake_reader(path: str):
        assert path == str(csv_path)
        return structured_dataframe.copy(deep=True), metadata.copy()

    monkeypatch.setattr(itemtest_analyzer, "read_itemtest_csv", _fake_reader)

    itemtest_analyzer.process_file(csv_path, tmp_path, layout_df=None)

    expected_plot = tmp_path / "graficos" / "structured" / "rssi_vs_frequency.png"
    assert expected_plot.exists(), "Scatter plot for structured mode was not created."
    assert expected_plot.stat().st_size > 0


def test_process_file_generates_pallet_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    structured_dataframe: pd.DataFrame,
    _stub_write_excel: Callable[..., None],
) -> None:
    """Ensure the structured pipeline exports the pallet layout heatmap."""

    csv_path = tmp_path / "structured.csv"
    csv_path.write_text("placeholder", encoding="utf-8")

    layout_df = pd.DataFrame(
        [
            {
                "Row": "1",
                "Rear": ["143"],
                "Left_Side": ["144"],
                "Right_Side": ["145"],
                "Front": ["142"],
            }
        ],
        columns=["Row", "Rear", "Left_Side", "Right_Side", "Front"],
    )

    metadata = {"Hostname": "test-reader"}

    def _fake_reader(path: str):
        assert path == str(csv_path)
        return structured_dataframe.copy(deep=True), metadata.copy()

    monkeypatch.setattr(itemtest_analyzer, "read_itemtest_csv", _fake_reader)

    itemtest_analyzer.process_file(csv_path, tmp_path, layout_df=layout_df)

    expected_plot = tmp_path / "graficos" / "structured" / "pallet_heatmap.png"
    assert expected_plot.exists(), "Pallet heatmap was not created."
    assert expected_plot.stat().st_size > 0


def test_process_continuous_file_generates_rssi_frequency_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuous_dataframe: pd.DataFrame,
    _stub_write_excel: Callable[..., None],
) -> None:
    """Ensure the continuous pipeline persists the RSSI vs Frequency chart."""

    csv_path = tmp_path / "continuous.csv"
    csv_path.write_text("placeholder", encoding="utf-8")

    metadata = {"Hostname": "test-reader"}

    def _fake_reader(path: str):
        assert path == str(csv_path)
        return continuous_dataframe.copy(deep=True), metadata.copy()

    monkeypatch.setattr(itemtest_analyzer, "read_itemtest_csv", _fake_reader)

    itemtest_analyzer.process_continuous_file(
        csv_path,
        tmp_path,
        window_seconds=2.0,
    )

    expected_plot = (
        tmp_path
        / "graficos"
        / f"{csv_path.stem}_continuous"
        / "rssi_vs_frequency.png"
    )
    assert expected_plot.exists(), "Scatter plot for continuous mode was not created."
    assert expected_plot.stat().st_size > 0

    throughput_plot = (
        tmp_path
        / "graficos"
        / f"{csv_path.stem}_continuous"
        / "throughput_per_minute.png"
    )
    assert throughput_plot.exists(), "Throughput per minute plot was not created."
    assert throughput_plot.stat().st_size > 0
