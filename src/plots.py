# -*- coding: utf-8 -*-
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_reads_by_epc(
    df_summary: pd.DataFrame,
    outdir: str,
    title: str = "Reads by EPC",
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ordered = df_summary.sort_values("total_reads", ascending=False)
    plt.figure()
    plt.bar(ordered["EPC_suffix3"].astype(str), ordered["total_reads"].astype(int))
    plt.title(title)
    plt.xlabel("EPC (suffix 3)")
    plt.ylabel("Total reads")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out / "reads_by_epc.png")
    plt.close()

def plot_reads_by_antenna(
    df_ant: pd.DataFrame,
    outdir: str,
    title: str = "Reads by Antenna",
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(df_ant["Antenna"].astype(int).astype(str), df_ant["total_reads"].astype(int))
    plt.title(title)
    plt.xlabel("Antenna")
    plt.ylabel("Total reads")
    plt.tight_layout()
    plt.savefig(out / "reads_by_antenna.png")
    plt.close()

def boxplot_rssi_by_antenna(
    df: pd.DataFrame,
    outdir: str,
    title: str = "RSSI distribution by antenna",
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if "RSSI" not in df.columns or df["RSSI"].dropna().empty:
        return
    antennas = sorted(df["Antenna"].dropna().unique())
    data = [df.loc[df["Antenna"] == a, "RSSI"].dropna() for a in antennas]
    labels = [str(int(a)) for a in antennas]
    if not any(len(values) > 0 for values in data):
        return
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel("Antenna")
    plt.ylabel("RSSI (dBm)")
    plt.tight_layout()
    plt.savefig(out / "rssi_by_antenna_boxplot.png")
    plt.close()

def plot_active_epcs_over_time(
    epcs_per_minute: pd.Series,
    outdir: str,
    title: str = "Active EPCs per minute",
):
    """Plot the number of active EPCs over time for continuous mode analysis."""

    if epcs_per_minute is None or epcs_per_minute.empty:
        return

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    series = epcs_per_minute.sort_index()
    times = pd.to_datetime(series.index)
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert(None)

    plt.figure()
    plt.plot(times.to_pydatetime(), series.astype(float), marker="o")
    plt.title(title)
    plt.xlabel("Time (min)")
    plt.ylabel("Active unique EPCs")
    plt.grid(axis="y", alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig(out / "active_epcs_over_time.png")
    plt.close()


def plot_antenna_heatmap(
    per_epc_summary: pd.DataFrame,
    outdir: str,
    title: str = "Antenna participation heatmap",
):
    """Plot a heatmap showing antenna participation per EPC for continuous mode."""

    if per_epc_summary is None or per_epc_summary.empty:
        return
    if "antenna_distribution" not in per_epc_summary.columns:
        return

    records: list[dict] = []
    labels: list[str] = []
    for row in per_epc_summary.itertuples(index=False):
        distribution = getattr(row, "antenna_distribution", None)
        if isinstance(distribution, dict) and distribution:
            normalized = {}
            for key, value in distribution.items():
                try:
                    normalized_key = int(key)
                except (TypeError, ValueError):
                    try:
                        normalized_key = int(float(key))
                    except (TypeError, ValueError):
                        normalized_key = str(key)
                try:
                    normalized[normalized_key] = float(value)
                except (TypeError, ValueError):
                    continue
            records.append(normalized)
        else:
            records.append({})
        labels.append(str(getattr(row, "EPC", "")))

    if not any(record for record in records):
        return

    matrix = pd.DataFrame(records, index=labels).fillna(0.0)
    if matrix.empty:
        return

    def _sort_key(value: object) -> tuple[int, str]:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return (0, f"{int(value):04d}")
        text = str(value)
        if text.isdigit():
            return (0, f"{int(text):04d}")
        return (1, text)

    sorted_columns = sorted(matrix.columns, key=_sort_key)
    matrix = matrix.reindex(columns=sorted_columns)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, max(4, len(matrix) * 0.25)))
    data = matrix.to_numpy(dtype=float)
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Participation (%)")
    x_labels = [str(col) for col in matrix.columns]
    suffix_labels = [label[-4:] if len(label) > 4 else label for label in matrix.index]
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(suffix_labels)), suffix_labels)
    plt.xlabel("Antenna")
    plt.ylabel("EPC (suffix)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / "antenna_heatmap.png")
    plt.close()
