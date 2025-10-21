# 📗 GUIDE_METRICS.md — RFID Metrics Reference for *ItemTestAnalyzer*

## 🧭 Overview

This guide defines **all Key Performance Indicators (KPIs)** and analytical metrics used by the *ItemTestAnalyzer* project.
It covers both **Structured Mode** (controlled pallet tests) and **Continuous Mode** (real-world dock unloading).

Each metric includes:

* Definition
* Formula or calculation method
* Scope of application
* Purpose (Executive / Technical / Diagnostic)

---

## 🎯 1. Common Metrics (All Modes)

| Metric                   | Definition                                         | Formula / Calculation                                   | Purpose    |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------------- | ---------- |
| **TotalDistinctEPCs**    | Number of unique EPCs read during the test/session | `count_unique(EPC)`                                     | Executive  |
| **TotalReads**           | Total number of read events (rows in CSV)          | `len(df)`                                               | Technical  |
| **AverageRSSI**          | Mean signal strength of all reads                  | `mean(RSSI)`                                            | Technical  |
| **RSSI_StdDev**          | RSSI stability indicator (variation)               | `std(RSSI)`                                             | Diagnostic |
| **BestRSSI**             | Highest RSSI (closest to 0) per EPC                | `max(RSSI per EPC)`                                     | Technical  |
| **WorstRSSI**            | Lowest RSSI (most negative) per EPC                | `min(RSSI per EPC)`                                     | Technical  |
| **AntennaParticipation** | % of total reads per antenna                       | `(reads_by_antenna / total_reads) × 100`                | Technical  |
| **TopPerformerAntenna**  | Antenna with highest contribution                  | `argmax(reads_by_antenna)`                              | Executive  |
| **AntennaBalance**       | Balance deviation between antennas                 | `std(reads_by_antenna / total_reads)`                   | Diagnostic |
| **ModePerformance**      | Performance by RF mode (ModeIndex)                 | Compare `CoverageRate` or `TotalReads` across ModeIndex | Technical  |
| **NoiseIndicator**       | Potential interference or ghost reads              | High RSSI variance without EPC increase                 | Diagnostic |

---

## 🧩 2. Structured Mode Metrics

| Metric                     | Definition                                       | Formula / Calculation                      | Purpose    |
| -------------------------- | ------------------------------------------------ | ------------------------------------------ | ---------- |
| **CoverageRate**           | % of expected tags that were successfully read   | `(unique_EPCs_read / expected_EPCs) × 100` | Executive  |
| **PositionalCoverage**     | Coverage by line and face (based on layout file) | Count of EPCs found per grid position      | Executive  |
| **ReadRedundancy**         | Average times each EPC was read                  | `total_reads / unique_EPCs_read`           | Diagnostic |
| **RSSI_Stability_Index**   | RSSI variation per antenna                       | `std(RSSI_grouped_by_antenna)`             | Technical  |
| **ReadDistributionByFace** | Relative read count per pallet face              | Group by `Face` column in layout           | Executive  |
| **MissedTags**             | Expected EPCs not detected                       | `expected_set - read_set`                  | Diagnostic |
| **TagHotspots**            | EPCs with abnormal read density                  | `reads_per_EPC > mean + 2×std`             | Diagnostic |
| **FrequencyUsage**         | Frequency channels used in session               | `unique(Frequency)`                        | Technical  |
| **TagLocationError**       | EPCs read in position different from reference   | Cross-compare layout mapping               | Diagnostic |

---

## ⚙️ 3. Continuous Mode Metrics

| Metric                  | Definition                                   | Formula / Calculation                        | Purpose    |
| ----------------------- | -------------------------------------------- | -------------------------------------------- | ---------- |
| **TagDwellTimeAvg**     | Average time a tag remained detectable       | `mean(duration_present_per_EPC)`             | Executive  |
| **TagDwellTimeMax**     | Longest dwell duration observed              | `max(duration_present_per_EPC)`              | Diagnostic |
| **DurationPresent**     | Total time EPC was seen in field             | `last_timestamp - first_timestamp (per EPC)` | Technical  |
| **ReadEvents**          | Number of entry/exit events per EPC          | `count_transitions(EPC_active_state)`        | Technical  |
| **ThroughputPerMinute** | Unique EPCs detected per minute              | `unique_EPCs / duration_minutes`             | Executive  |
| **ReadContinuityRate**  | % of time with EPCs active                   | `(active_seconds / total_seconds) × 100`     | Executive  |
| **ConcurrentTagsPeak**  | Max EPCs simultaneously detected             | `max(active_EPCs_per_second)`                | Technical  |
| **ActiveConcurrency**   | Average simultaneous EPCs                    | `mean(active_EPCs_per_second)`               | Technical  |
| **DirectionEstimate**   | Movement direction inferred by antenna order | `first_antenna → last_antenna`               | Technical  |
| **RSSI_Variability**    | Signal variation per EPC                     | `std(RSSI_per_EPC)`                          | Diagnostic |
| **AntennaDominance**    | Antenna with most reads per EPC              | `mode(Antenna_per_EPC)`                      | Technical  |
| **SessionDuration**     | Total duration of reading activity           | `max(Timestamp) - min(Timestamp)`            | Executive  |
| **SessionThroughput**   | Distinct EPCs ÷ session time                 | `unique_EPCs / session_duration_minutes`     | Executive  |
| **InactivePeriods**     | Gaps without tag reads                       | Detect intervals > 5×window with zero reads  | Diagnostic |
| **CongestionIndex**     | EPC overlap ratio (read density)             | `total_reads / active_duration_seconds`      | Technical  |

---

## 📊 4. Executive KPI Dashboard (Suggested)

| KPI                              | Description                          | Source                 |
| -------------------------------- | ------------------------------------ | ---------------------- |
| **EPCs Distinct**                | Total EPCs identified during session | `TotalDistinctEPCs`    |
| **RSSI Mean (dBm)**              | Average signal strength              | `AverageRSSI`          |
| **RSSI Stability (σ)**           | Consistency of readings              | `RSSI_StdDev`          |
| **Coverage (%)**                 | % of expected EPCs read              | `CoverageRate`         |
| **Avg Dwell (s)**                | Average tag presence time            | `TagDwellTimeAvg`      |
| **Throughput (/min)**            | Distinct tags detected per minute    | `ThroughputPerMinute`  |
| **Concurrent Tags (max)**        | Peak simultaneous EPCs               | `ConcurrentTagsPeak`   |
| **Antenna 1–4 Contribution (%)** | Share of reads by antenna            | `AntennaParticipation` |
| **Antenna Balance**              | Variance across antennas             | `AntennaBalance`       |
| **Noise Index**                  | RSSI noise proxy                     | `RSSI_Variability`     |

---

## 🧮 5. Derived Relationships

| Relationship             | Formula                                                 | Insight                               |
| ------------------------ | ------------------------------------------------------- | ------------------------------------- |
| **Efficiency Index**     | `(CoverageRate × (100 - AntennaBalance)) / RSSI_StdDev` | Measures overall system efficiency    |
| **Flow Stability Index** | `(1 - RSSI_Variability/100) × ReadContinuityRate`       | Measures stability of continuous flow |
| **Redundancy Index**     | `TagReadRedundancy / TotalDistinctEPCs`                 | Quantifies reading repetition         |
| **Operational Load**     | `ConcurrentTagsPeak × AvgDwell / SessionDuration`       | Indicates potential reader saturation |
| **Reader Health Score**  | Weighted composite: RSSI stability + balance + coverage | General condition of reader setup     |

---

## 🔍 6. Data Integrity Checks

| Check                   | Description                       | Handling                   |
| ----------------------- | --------------------------------- | -------------------------- |
| Invalid EPC             | Non-hex EPCs or short EPCs        | Filter out                 |
| IP in EPC               | Example `192.168.68.100`          | Ignore                     |
| Empty RSSI or Timestamp | Log warning, skip line            | Logged as “Incomplete row” |
| Duplicate Timestamp/EPC | Count but flag in summary         | “Duplicate read” count     |
| Missing Frequency       | Fill NaN or flag as N/A           | Diagnostic alert           |
| ModeIndex Mismatch      | Conflicting modes in same session | Log “Mixed Mode” warning   |

---

## 🧰 7. Implementation Notes for Developers

* All metrics should be implemented as **pure functions** within `metrics.py`.
  Each function must accept a `pandas.DataFrame` and return a scalar or Series.
* Calculations must be **vectorized** for performance.
* Metrics must be **mode-aware** — only relevant metrics should appear in each mode’s report.
* Aggregated executive KPIs (for Power BI or Streamlit) must be generated in `report.py`.
* Use descriptive names and maintain **unit consistency** (RSSI in dBm, time in seconds, rate in %).
* Each metric should include docstrings:

  ```python
  def calculate_coverage_rate(df: pd.DataFrame, expected_epcs: list[str]) -> float:
      """
      Calculates tag coverage percentage for structured tests.
      Args:
          df: DataFrame with EPC readings
          expected_epcs: list of expected EPC identifiers
      Returns:
          float: percentage of expected EPCs read
      """
  ```

---

## 🧱 8. Example Calculation Flow (Continuous Mode)

1. Load CSV → clean → convert timestamps.
2. Group by EPC → aggregate per time window (2s default).
3. Compute dwell times → detect entry/exit events.
4. Aggregate global KPIs: throughput, dwell mean, concurrency.
5. Generate executive and technical sheets:

   * `Fluxo_Contínuo`
   * `Indicadores_Executivos`
   * `Resumo_por_EPC`

---

## 📈 9. Graph Recommendations

| Graph Type                   | Purpose                         | Mode       |
| ---------------------------- | ------------------------------- | ---------- |
| Bar — Reads per Antenna      | Visualize antenna contribution  | Both       |
| Boxplot — RSSI per Antenna   | Identify noise or imbalance     | Both       |
| Heatmap — Layout Coverage    | Visualize pallet face coverage  | Structured |
| Line — EPCs Active Over Time | Show tag flow rate              | Continuous |
| Line — Throughput (EPC/min)  | Monitor process efficiency      | Continuous |
| Scatter — RSSI × Frequency   | Detect interference patterns    | Both       |
| Bubble — Dwell vs RSSI       | Correlate tag distance and time | Continuous |

---

## 🧠 10. Executive Summary Generation Template

The report module should produce a concise summary like:

> **Session Summary — Dock Test (2025-10-20)**
> Reader: `192.168.68.100` (ModeIndex 1002, DualTarget)
> EPCs detected: **1,280**
> Avg RSSI: **–53.2 dBm**
> Avg Dwell: **3.4 s**
> Coverage: **98.1%** (structured mode)
> Peak concurrency: **87 EPCs**
> Antenna #3 contributed **47%** of total reads.
> No anomalies detected in RSSI or frequency distribution.

---

## 📘 11. Maintenance Recommendations

* Maintain separation of **data ingestion**, **metric computation**, and **report generation** layers.
* For versioning, add metric version tags (`v1.0`, `v1.1`) as column suffixes when formulas evolve.
* Validate against at least one **reference test CSV** per mode before merging new logic.
* Keep historical KPI definitions archived for auditability.


