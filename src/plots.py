# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_reads_by_epc(df_summary: pd.DataFrame, outdir: str, title: str = "Leituras por EPC"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    d = df_summary.sort_values("total_reads", ascending=False)
    plt.figure()
    plt.bar(d["EPC_suffix3"].astype(str), d["total_reads"].astype(int))
    plt.title(title)
    plt.xlabel("EPC (sufixo 3)")
    plt.ylabel("Total de leituras")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out/"leituras_por_epc.png")
    plt.close()

def plot_reads_by_antenna(df_ant: pd.DataFrame, outdir: str, title: str = "Leituras por Antena"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(df_ant["Antenna"].astype(int).astype(str), df_ant["total_reads"].astype(int))
    plt.title(title)
    plt.xlabel("Antena")
    plt.ylabel("Total de leituras")
    plt.tight_layout()
    plt.savefig(out/"leituras_por_antena.png")
    plt.close()

def boxplot_rssi_by_antenna(df: pd.DataFrame, outdir: str, title: str = "Distribuição de RSSI por Antena"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    if "RSSI" not in df.columns or df["RSSI"].dropna().empty:
        return
    data = [df.loc[df["Antenna"]==a, "RSSI"].dropna() for a in sorted(df["Antenna"].dropna().unique())]
    labels = [str(int(a)) for a in sorted(df["Antenna"].dropna().unique())]
    if not any(len(x)>0 for x in data):
        return
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel("Antena")
    plt.ylabel("RSSI (dBm)")
    plt.tight_layout()
    plt.savefig(out/"rssi_por_antena_boxplot.png")
    plt.close()
