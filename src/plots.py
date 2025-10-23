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
    plt.boxplot(data, tick_labels=labels)
    plt.title(title)
    plt.xlabel("Antenna")
    plt.ylabel("RSSI (dBm)")
    plt.tight_layout()
    plt.savefig(out / "rssi_by_antenna_boxplot.png")
    plt.close()


def plot_rssi_vs_frequency(
    df: pd.DataFrame,
    outdir: str,
    title: str = "RSSI vs Frequency",
) -> None:
    """Plot a scatter chart showing RSSI distribution across frequency channels.

    Parameters
    ----------
    df:
        Raw ItemTest reads including ``RSSI`` and ``Frequency`` columns.
    outdir:
        Destination directory where the ``rssi_vs_frequency.png`` artifact
        should be written. The directory is created when it does not exist.
    title:
        Custom chart title displayed above the scatter plot.
    """

    if df is None or df.empty:
        return
    if "RSSI" not in df.columns or "Frequency" not in df.columns:
        return

    working = pd.DataFrame(
        {
            "frequency": pd.to_numeric(df["Frequency"], errors="coerce"),
            "rssi": pd.to_numeric(df["RSSI"], errors="coerce"),
        }
    ).dropna()
    working = working.loc[working["frequency"] > 0]
    if working.empty:
        return

    freq_abs_max = float(working["frequency"].abs().max())
    scale_factor = 1.0
    if freq_abs_max >= 1_000_000:
        scale_factor = 1_000_000.0
    elif freq_abs_max >= 1_000:
        scale_factor = 1_000.0
    working["frequency_mhz"] = working["frequency"] / scale_factor

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        working["frequency_mhz"],
        working["rssi"],
        s=18,
        alpha=0.6,
        color="#1f77b4",
    )
    plt.title(title)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("RSSI (dBm)")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    x_min = float(working["frequency_mhz"].min())
    x_max = float(working["frequency_mhz"].max())
    if x_max > x_min:
        margin = max((x_max - x_min) * 0.05, 0.1)
        plt.xlim(x_min - margin, x_max + margin)

    y_min = float(working["rssi"].min())
    y_max = float(working["rssi"].max())
    if y_max > y_min:
        y_margin = max((y_max - y_min) * 0.05, 1.0)
        plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.savefig(out / "rssi_vs_frequency.png", dpi=150)
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


def plot_pallet_heatmap(
    positions_df: pd.DataFrame | None,
    outdir: str,
    title: str = "Pallet read density heatmap",
) -> None:
    """Plot a pallet heatmap using layout rows/faces against read totals.

    Parameters
    ----------
    positions_df:
        DataFrame containing at least the columns ``Row``, ``Face``, and
        ``total_reads`` produced by the structured-layout coverage logic. Each
        row represents a pallet position (row/face) with accumulated reads.
    outdir:
        Destination directory that will store the ``pallet_heatmap.png`` file.
    title:
        Custom chart title shown above the heatmap.
    """

    if positions_df is None or positions_df.empty:
        return

    required_columns = {"Row", "Face", "total_reads"}
    if not required_columns.issubset(positions_df.columns):
        return

    working = positions_df.copy()
    working["Row"] = working["Row"].astype(str)
    working["Face"] = working["Face"].astype(str)
    working["total_reads"] = pd.to_numeric(
        working["total_reads"], errors="coerce"
    ).fillna(0.0)

    if working.empty:
        return

    # Preserve the layout order as defined in the original dataframe
    def _unique_preserve_order(series: pd.Series) -> list[str]:
        seen: list[str] = []
        for item in series:
            label = str(item)
            if label not in seen:
                seen.append(label)
        return seen

    row_labels = _unique_preserve_order(working["Row"])
    face_labels = _unique_preserve_order(working["Face"])

    pivot = (
        working.pivot_table(
            index="Row",
            columns="Face",
            values="total_reads",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=row_labels, columns=face_labels)
        .astype(float)
    )

    if pivot.empty:
        return

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = pivot.to_numpy()
    n_rows, n_cols = data.shape
    fig_width = max(6.0, n_cols * 1.2)
    fig_height = max(4.0, n_rows * 0.7)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, aspect="auto", cmap="YlGn")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Total reads")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(face_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Face")
    ax.set_ylabel("Row")
    ax.set_title(title)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = data[row_idx, col_idx]
            if pd.isna(value):
                continue
            normalized = im.norm(value)
            text_color = "white" if normalized > 0.6 else "black"
            if float(value).is_integer():
                label = f"{int(value)}"
            else:
                label = f"{value:.1f}"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out / "pallet_heatmap.png", dpi=150)
    plt.close(fig)
