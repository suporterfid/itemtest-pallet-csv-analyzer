# 🤖 AGENTS.md — AI Development Guide for *ItemTestAnalyzer*

## 🧭 Overview

This repository implements a **Python-based analytics application** that processes CSV exports from **Impinj ItemTest** RFID reading sessions.

It supports **two operation modes**:

1. **Structured Mode (`--mode structured`)**
   Used for controlled RFID tests with **known tag layouts** (e.g., pallets or totes).
   Uses an optional **layout reference file** (CSV/XLSX/Markdown) to correlate EPCs with physical pallet positions and calculate coverage, reliability, and antenna performance metrics.

2. **Continuous Mode (`--mode continuous`)**
   Designed for **real-world operations**, such as unloading multiple pallets from a trailer in a distribution center (dock).
   In this mode, tags are **not known in advance**, and the system detects **EPC entry, dwell, and exit events** over time to analyze throughput, congestion, and antenna performance dynamically.

This document defines how **AI agents (Codex, Assistants, or Responses API)** can safely extend or refactor this repository while maintaining functional parity and analytical integrity.

---

## 🧩 Project Structure

```
ItemTestAnalyzer/
├── src/
│   ├── itemtest_analyzer.py     ← Main CLI entry point
│   ├── parser.py                ← Parses Impinj ItemTest CSV logs
│   ├── metrics.py               ← Computes EPC, antenna, and KPI statistics
│   ├── pallet_layout.py         ← Handles optional pallet layout reference
│   ├── continuous_mode.py       ← Handles continuous stream and dwell analysis
│   ├── plots.py                 ← Generates static and temporal charts
│   └── report.py                ← Builds Excel reports and textual summaries
├── samples/
│   ├── Sample_ItemTest.csv
│   └── Sample_Pallet_Layout.xlsx
├── output/
│   ├── figures/
│   └── logs/
├── requirements.txt
├── run.bat
└── README.md
```

---

## ⚙️ Agent Responsibilities

### 🎯 Primary Goals

AI agents collaborating on this repository may:

1. **Enhance structured-mode analytics** — improve coverage calculations, positional mapping, and antenna balance metrics.
2. **Expand continuous-mode analysis** — detect tag dwell times, directionality, congestion peaks, and tag concurrency.
3. **Extend reporting** — include new executive KPIs, statistical summaries, and Power BI–ready exports.
4. **Automate workflows** — batch-process CSV folders and merge reports across sessions.
5. **Integrate visualization** — add or extend dashboards using Streamlit, Plotly, or Power BI connectors.

---

### 🧱 Must Preserve

* CLI entrypoint in `src/itemtest_analyzer.py`
* Parameters:

  ```bash
  python src/itemtest_analyzer.py --input <input_dir> --output <output_dir> [--layout <layout_file>] [--mode <structured|continuous>] [--window <seconds>]
  ```
* Windows 11 compatibility (UTF-8, spaces in paths)
* Backward compatibility with Impinj ItemTest CSV format
* Consistent Excel outputs across modes (`Resumo_por_EPC`, `Leituras_por_Antena`, etc.)

---

## 🧠 System Behavior Summary

### Step 1 — CSV Parsing

* Ignore IP-like EPCs (`192.168.x.x`)
* Retain only valid hex EPCs (`[0-9A-F]{24,}`)
* Extract metadata:

  * `AntennaIDs`, `ModeIndex`, `PowersInDbm`, `Session`, `InventoryMode`, `Hostname`
* Convert timestamps and numeric columns properly (`RSSI`, `Frequency`, etc.)

---

### Step 2 — Structured Mode (with layout reference)

* Correlate EPCs (or suffixes) with pallet positions.
* Generate layout coverage and reliability maps.
* Compute:

  * **CoverageRate** = % of expected EPCs read
  * **AntennaBalance** = deviation between antennas
  * **RSSI Stability Index** (std deviation per antenna)
  * **TopPerformerAntenna** (most reads)
  * **TagReadRedundancy** (avg. repeated reads per EPC)
* Generate `Posicoes_Pallet` sheet and visual heatmap of read density.

---

### Step 3 — Continuous Mode (dock unloading / unknown EPCs)

When `--mode continuous` or no layout is provided:

1. **Temporal Grouping**

   * Aggregate readings by EPC and configurable window (`--window`, default 2 s).
   * Detect EPC **entry** and **exit** events based on gaps > window.

2. **Dynamic Metrics per EPC**

   | Field                  | Description                       |
   | ---------------------- | --------------------------------- |
   | `duration_present`     | Total time EPC was detected       |
   | `read_events`          | Number of entry/exit cycles       |
   | `antenna_distribution` | % reads per antenna               |
   | `direction_estimate`   | Inferred motion (e.g., Ant1→Ant3) |
   | `RSSI_variability`     | Std dev of RSSI over time         |
   | `active_concurrency`   | Avg. simultaneous EPCs active     |
   | `TagDwellTimeAvg`      | Mean dwell time in field          |
   | `TagDwellTimeMax`      | Max dwell time per EPC            |

3. **Aggregated Insights**

   * EPCs per minute (throughput)
   * Average dwell time
   * RSSI mean and stability
   * Antenna dominance
   * Concurrency peak (max EPCs simultaneously detected)
   * Session performance metrics (duration, throughput, RSSI trend)

4. **Detect Anomalies**

   * EPCs lingering excessively (possible obstructions)
   * Inconsistent antenna reads (upper antennas only, etc.)
   * Frequency drift or interference (`RSSI` fluctuation + frequency span)

---

## 📊 Output Reports

### Excel Sheets

| Sheet                    | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `Resumo_por_EPC`         | Detailed EPC-level metrics (reads, RSSI, dwell) |
| `Leituras_por_Antena`    | Antenna performance breakdown                   |
| `Fluxo_Contínuo`         | Entry/exit timing (continuous mode only)        |
| `Indicadores_Executivos` | Key performance indicators (aggregated)         |
| `Metadata`               | Reader configuration and session info           |
| `Posicoes_Pallet`        | Physical coverage (structured mode only)        |

### Charts

* Reads per EPC
* Reads per Antenna
* RSSI Boxplot by Antenna
* EPCs Active over Time (continuous mode)
* Heatmap (antenna coverage or pallet layout)
* Throughput Timeline (EPCs/min)
* RSSI vs Frequency scatter plot (noise detection)

### Text Summary Example

> “Between 14:00 and 14:15, reader `192.168.68.100` detected **1,280 unique EPCs**,
> mean dwell time **3.4 s**, average RSSI **–53.2 dBm**,
> with **Antenna 3** responsible for **47% of total reads**.”

---

## 🧰 Agent Action Map

| Capability             | Module                 | Purpose                                            |
| ---------------------- | ---------------------- | -------------------------------------------------- |
| Add new metrics        | `metrics.py`           | Extend both modes with additional KPIs             |
| Add dwell analysis     | `continuous_mode.py`   | Implement EPC presence tracking and timing logic   |
| Add advanced charts    | `plots.py`             | Generate time-based and heatmap visuals            |
| Extend Excel reporting | `report.py`            | Add KPIs and summaries for executives              |
| Expand CLI             | `itemtest_analyzer.py` | Support `--mode`, `--window`, `--summary`          |
| Update parsing rules   | `parser.py`            | Enhance metadata extraction and timestamp handling |
| Improve layout mapping | `pallet_layout.py`     | Multi-pallet, 3D mapping, or position weighting    |

---

## 🧠 Suggested AI Tasks

| Task                               | Objective                                            | Tools                       |
| ---------------------------------- | ---------------------------------------------------- | --------------------------- |
| `agent.extend_kpis()`              | Add CoverageRate, RSSI_StdDev, TagDwellTimeAvg, etc. | `pandas`, `numpy`           |
| `agent.add_noise_analysis()`       | Detect high RSSI variance and frequency anomalies    | `numpy`, `matplotlib`       |
| `agent.build_executive_summary()`  | Create Excel sheet with KPIs for management          | `xlsxwriter`, `pandas`      |
| `agent.add_concurrency_metric()`   | Track peak concurrent tags                           | `pandas`, `datetime`        |
| `agent.generate_throughput_plot()` | Plot EPCs per minute                                 | `matplotlib`, `seaborn`     |
| `agent.add_streamlit_dashboard()`  | Visual dashboard for structured/continuous modes     | `streamlit`, `plotly`       |
| `agent.create_pdf_report()`        | Export executive PDF with summarized metrics         | `reportlab`                 |
| `agent.merge_sessions()`           | Combine multiple CSVs into unified analytics         | `glob`, `pathlib`, `pandas` |

---

## 🧩 Agent Development Rules

1. **Preserve CLI parameters** — `--input`, `--output`, `--layout`, `--mode`, `--window`, `--summary`.
2. **Maintain backward compatibility** with current CSV/Excel structure.
3. **Use Pathlib for file paths** (no hardcoded strings).
4. **Document functions** with purpose, params, and return values.
5. **Include logging** for missing or malformed EPCs, empty timestamps, and antenna mismatches.
6. **Follow PEP8 and UTF-8 standards.**
7. **Keep metrics modular** — define each KPI as an independent function for composability.

---

## 🧠 Future Extensions (AI Roadmap)

| Feature                 | Description                                       |
| ----------------------- | ------------------------------------------------- |
| 📊 Power BI connector   | Generate standardized KPI exports for dashboards  |
| 🧾 PDF Reports          | Branded summaries for executives                  |
| 🧠 Predictive Analytics | Identify blind zones and RF interference patterns |
| 🧮 Comparative Mode     | Compare multiple sessions or readers              |
| 🌐 Streamlit App        | Real-time upload, visualization, and export       |
| 🧱 Docker Support       | Containerized version for production use          |

---

## 🧩 Prompt Template for AI Agents

```markdown
You are an AI engineer contributing to the repository ItemTestAnalyzer.
Implement a new feature or enhancement following these requirements:

- Support both `--mode structured` and `--mode continuous`.
- Preserve CLI parameters (`--input`, `--output`, `--layout`, `--mode`, `--window`, `--summary`).
- Compute or visualize additional metrics such as CoverageRate, TagDwellTimeAvg, RSSI_StdDev, and ThroughputPerMinute.
- Maintain Excel output consistency and backward compatibility.

Use Python 3.11+, follow PEP8, and ensure UTF-8 compliance on Windows 11.
Provide full updated module code with docstrings.
```

---

## 🧠 Summary

The *ItemTestAnalyzer* is a **dual-mode analytics platform** for RFID performance evaluation:

* **Structured Mode:** validation and layout-based testing.
* **Continuous Mode:** real-time behavior analysis for dock operations.

The repository is designed for **AI-assisted evolution**, enabling agents to deliver executive-level insight and technical depth — bridging engineering data with operational intelligence.

---


