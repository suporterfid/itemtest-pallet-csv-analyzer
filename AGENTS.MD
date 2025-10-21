# 🤖 AGENTS.md — AI Development Guide for *ItemTestAnalyzer*

## 🧭 Overview

This repository implements a **Python-based data analysis application** that processes CSV exports from **Impinj ItemTest** RFID reading tests.

The system supports **two operation modes**:

1. **Structured Mode (`--mode structured`)**
   Used for controlled RFID reading tests of totes or pallets where tags (EPCs) and their physical positions are known.
   Optionally uses a **pallet layout reference file** (CSV/XLSX/Markdown) for coverage mapping.

2. **Continuous Mode (`--mode continuous`)**
   Designed for **real-world operations**, such as unloading multiple pallets from a trailer at a distribution dock.
   In this mode, tags are **not known in advance**, and the system analyzes **continuous reading streams**, detecting EPC entries and exits dynamically.

This document provides **guidelines for AI agents** (Codex, Responses API, or Assistants) to safely extend or refactor the system.

---

## 🧩 Project Structure

```
ItemTestAnalyzer/
├── src/
│   ├── analisar_itemtest.py     ← Main CLI entry point
│   ├── parser.py                ← Parses Impinj ItemTest CSV logs
│   ├── metrics.py               ← Computes EPC and antenna statistics
│   ├── pallet_layout.py         ← Handles optional pallet layout reference
│   ├── continuous_mode.py       ← Handles continuous stream analysis (dock unloading)
│   ├── plots.py                 ← Generates charts using matplotlib
│   └── report.py                ← Builds Excel reports with xlsxwriter
├── samples/
│   ├── Teste_Estruturado.csv
│   ├── Teste_Continuo.csv
│   └── Layout_Pallet_Exemplo.xlsx
├── output/
│   ├── graficos/
│   └── logs/
├── requirements.txt
├── run.bat
└── README.md
```

---

## ⚙️ Agent Responsibilities

### 🎯 Primary Goals

An AI agent working on this repository may:

1. **Enhance structured-mode analytics** — improve tag position correlation, layout validation, or coverage metrics.
2. **Develop continuous-mode logic** — detect tag flow, presence duration, and antenna-based movement patterns.
3. **Extend reporting** — add temporal charts, pallet flow dashboards, or executive summaries.
4. **Automate workflows** — multi-session batch processing or integration with Power BI.
5. **Support visualization** — implement interactive dashboards using Streamlit or FastAPI.

---

### 🧱 Must Preserve

* CLI entrypoint in `src/analisar_itemtest.py`
* Parameter usage:

  ```bash
  python src/analisar_itemtest.py --input <input_dir> --output <output_dir> [--layout <layout_file>] [--mode <structured|continuous>] [--window <seconds>]
  ```
* Windows 11 compatibility (UTF-8, spaces in paths)
* Backward compatibility with previous CSV format (Impinj ItemTest)
* Excel output consistency across modes (`Resumo_por_EPC`, `Leituras_por_Antena`, etc.)

---

## 🧠 System Behavior Summary

### Step 1 — Parse ItemTest CSV

* Ignore IP-like EPCs (`192.168.x.x`)
* Keep only valid hexadecimal EPCs (`[0-9A-F]{24,}`)
* Extract metadata:

  * `AntennaIDs`, `ModeIndex`, `PowersInDbm`, `Session`, `InventoryMode`, `Hostname`

---

### Step 2 — Structured Mode (with layout)

* Associate EPCs or suffixes with physical pallet positions.
* Generate sheet `Posicoes_Pallet` and calculate coverage rate.
* Mark missing tags and compute read distribution by side and height.

---

### Step 3 — Continuous Mode (dock unloading)

When `--mode continuous` is active or no layout file is provided:

1. **Group readings** by EPC and time window (configurable via `--window`, default = 2 seconds).
2. **Detect presence intervals:**

   * If an EPC stops appearing for longer than the window, mark as **“exited field”**.
3. **Compute metrics:**

   | Field                  | Description                     |
   | ---------------------- | ------------------------------- |
   | `duration_present`     | Total time EPC was detected     |
   | `read_events`          | Number of entry/exit events     |
   | `antenna_distribution` | Percentage of reads per antenna |
   | `direction_estimate`   | Approximate movement direction  |
4. **Generate summaries:**

   * EPCs per minute
   * Average dwell time
   * Antenna with highest detection rate
   * EPCs lingering too long (potential jam or read anomaly)

---

### Step 4 — Output Reports

**Excel outputs:**

* `Resumo_por_EPC` — all EPC stats
* `Leituras_por_Antena` — per-antenna performance
* `Fluxo_Contínuo` — (only in continuous mode) EPC entry/exit timeline
* `Metadata` — reader/session info
* `Posicoes_Pallet` — (only if layout available)

**Charts:**

* Reads per EPC
* Reads per Antenna
* RSSI Boxplot by Antenna
* EPCs Active over Time (continuous mode)
* Heatmap of antenna coverage (continuous mode)

**Text summary (auto-generated):**

> “During dock unloading from 14:00–14:15,
> reader `192.168.68.100` detected 1,280 unique EPCs
> average dwell: 3.4s, mean RSSI: –53.2 dBm,
> main antenna: 3 (base left).”

---

## 🧰 Agent Action Map

| Agent Capability      | Module                 | Purpose                                          |
| --------------------- | ---------------------- | ------------------------------------------------ |
| Add new metrics       | `metrics.py`           | Extend structured mode calculations              |
| Add flow detection    | `continuous_mode.py`   | Implement EPC presence tracking and timing logic |
| Add new chart type    | `plots.py`             | Visualize movement, heatmaps, or trends          |
| Modify Excel layout   | `report.py`            | Add new worksheets or format summaries           |
| Add CLI options       | `analisar_itemtest.py` | Extend with `--mode` or `--window`               |
| Update parsing        | `parser.py`            | Adjust CSV header recognition or metadata logic  |
| Expand layout mapping | `pallet_layout.py`     | Handle multi-pallet or dynamic layouts           |

---

## 🧠 Suggested AI Tasks

| Task                                 | Goal                                                         | Tools                   |
| ------------------------------------ | ------------------------------------------------------------ | ----------------------- |
| `agent.add_continuous_mode()`        | Implement dynamic presence detection and time-based grouping | `pandas`, `numpy`       |
| `agent.calculate_dwell_time()`       | Add per-EPC dwell time and movement direction                | `datetime`, `pandas`    |
| `agent.extend_structured_coverage()` | Cross-analyze layout positions with read frequency           | `pandas`, `numpy`       |
| `agent.generate_flow_heatmap()`      | Visualize dock throughput over time                          | `matplotlib`, `seaborn` |
| `agent.add_window_parameter()`       | Add `--window` CLI argument for time threshold tuning        | `argparse`              |
| `agent.add_streamlit_ui()`           | Add interactive web dashboard for EPC activity visualization | `streamlit`, `plotly`   |
| `agent.create_pdf_summary()`         | Export executive PDF summaries with top metrics              | `reportlab`             |

---

## 🧩 Agent Development Rules

1. **Do not modify** core CLI parameters (`--input`, `--output`, `--layout`, `--mode`, `--window`).
2. **Preserve backward compatibility** with structured mode outputs.
3. **Use Pathlib for all file paths.**
4. **Ensure time-based grouping is parameterized** (window in seconds).
5. **Always log anomalies** — duplicate EPCs, missing timestamps, or inconsistent antennas.
6. **Document new functions** with purpose, parameters, and return values.
7. **Follow PEP8 and UTF-8** coding standards.

---

## 🧠 Future Extensions (AI Roadmap)

| Feature                  | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| 🔄 Multi-session summary | Aggregate several CSVs into one report                 |
| ⏱ Real-time monitor      | Live dashboard for continuous mode (via Streamlit)     |
| 🧭 Antenna calibration   | Calculate gain balance and coverage bias               |
| 🧮 Power BI export       | Generate standardized CSV for BI ingestion             |
| 🧾 PDF report            | Auto-generate executive summaries per session          |
| 🧠 Predictive analytics  | Identify “blind spots” using tag dwell and RSSI trends |

---

## 🧩 Prompt Template for AI Agents

```markdown
You are an AI software engineer contributing to the repository ItemTestAnalyzer.
Implement a feature according to the following guidelines:

- Mode support: structured and continuous.
- CLI must preserve arguments (--input, --output, --layout, --mode, --window).
- In continuous mode, analyze tag entry/exit and dwell times.
- In structured mode, maintain pallet layout correlation.

Use Python 3.11+, follow PEP8, and ensure UTF-8 compatibility on Windows 11.
Return full file code for any module you modify.
```

---

## 🧠 Summary

This repository serves as a **dual-mode platform** for RFID read analysis:

* **Structured Mode** for controlled pallet tests;
* **Continuous Mode** for live field scenarios such as dock unload monitoring.

It’s engineered for modular, AI-assisted evolution — enabling both human developers and AI agents to extend it toward full **traceability, performance insight, and automation** in real-world RFID deployments.


