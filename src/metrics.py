# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from collections import Counter
from .parser import suffix3

def antenna_mode(series: pd.Series):
    s = series.dropna().astype(int)
    if s.empty:
        return None
    counts = Counter(s)
    maxc = max(counts.values())
    # menor ID em caso de empate
    return min([a for a,c in counts.items() if c==maxc])

def summarize_by_epc(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("EPC", as_index=False)
    summary = g.agg(
        total_reads=("EPC","count"),
        rssi_avg=("RSSI","mean"),
        rssi_best=("RSSI","max"),
        rssi_worst=("RSSI","min"),
        first_time=("Timestamp","min"),
        last_time=("Timestamp","max"),
    )
    # primeira/última antena por tempo
    first_rows = df.sort_values(["EPC","Timestamp"]).groupby("EPC", as_index=False).first()[["EPC","Antenna"]].rename(columns={"Antenna":"antenna_first"})
    last_rows  = df.sort_values(["EPC","Timestamp"]).groupby("EPC", as_index=False).last()[["EPC","Antenna"]].rename(columns={"Antenna":"antenna_last"})
    summary = summary.merge(first_rows, on="EPC", how="left").merge(last_rows, on="EPC", how="left")
    # antena mais frequente
    modes = df.groupby("EPC")["Antenna"].apply(antenna_mode).reset_index(name="antenna_mode")
    summary = summary.merge(modes, on="EPC", how="left")
    # sufixo
    summary["EPC_suffix3"] = summary["EPC"].astype(str).apply(suffix3).str.upper()
    return summary

def summarize_by_antenna(df: pd.DataFrame) -> pd.DataFrame:
    ant = (
        df.dropna(subset=["Antenna"])
          .groupby("Antenna")
          .agg(total_reads=("EPC", "count"), rssi_avg=("RSSI", "mean"))
          .reset_index()
          .sort_values("Antenna")
    )
    total_reads = int(ant["total_reads"].sum()) if not ant.empty else 0
    if total_reads > 0:
        ant["participation_pct"] = (
            ant["total_reads"].astype(float) / total_reads * 100
        )
    else:
        ant["participation_pct"] = 0.0
    return ant
