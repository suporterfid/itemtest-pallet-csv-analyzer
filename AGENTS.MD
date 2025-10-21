# 🤖 AGENTS.md — AI Development Guide for *ItemTestAnalyzer*

## 🧭 Overview

This repository implements a **Python-based data analysis application** that processes CSV exports from **Impinj ItemTest** RFID reading tests.

It can optionally use a **pallet layout reference file** (CSV/XLSX/Markdown) to associate each tag (EPC) to its physical position on the pallet, enabling positional coverage analysis.

This document provides **guidelines for AI agents** (such as GPT Codex, OpenAI Assistants, or Responses API workflows) to extend, refactor, or automate this repository safely and consistently.

---

## 🧩 Project Structure

```
ItemTestAnalyzer/
├── src/
│   ├── analisar_itemtest.py     ← Main CLI entry point
│   ├── parser.py                ← Parses Impinj ItemTest CSV logs
│   ├── metrics.py               ← Computes EPC and antenna statistics
│   ├── pallet_layout.py         ← Handles optional pallet layout reference
│   ├── plots.py                 ← Generates charts using matplotlib
│   └── report.py                ← Builds Excel reports with xlsxwriter
├── samples/
│   ├── Teste_Exemplo_ItemTest.csv
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

An AI Agent working on this repository may:

1. **Enhance analysis logic** — improve statistical accuracy, add correlation between antennas, or refine filtering of invalid EPCs.
2. **Extend reporting** — include new summary sheets, visual dashboards, or automated PDF generation.
3. **Automate workflows** — batch processing, parameter presets, or integration with dashboards (e.g. Power BI, Streamlit).
4. **Support UI integration** — implement optional web UI using Streamlit or FastAPI.

### 🧱 Must Preserve

* CLI entrypoint in `src/analisar_itemtest.py`
* Parameter usage:

  ```bash
  python src/analisar_itemtest.py --input <input_dir> --output <output_dir> [--layout <layout_file>]
  ```
* Compatibility with Windows 11 (UTF-8, paths with spaces)
* Support for **CSV**, **XLSX**, and **Markdown** layout references
* Columns and report structure in Excel output (`Resumo_por_EPC`, `Leituras_por_Antena`, etc.)

---

## 🧠 System Behavior Summary

### Step 1 — Parse ItemTest CSV

* Ignore IP-like EPCs (`192.168.x.x`)
* Keep only valid hex EPCs (`[0-9A-F]{24,}`)
* Extract metadata:

  * `AntennaIDs`, `ModeIndex`, `PowersInDbm`, `Session`, `InventoryMode`, `Hostname`

### Step 2 — Calculate Metrics

| Field                  | Description                          |
| ---------------------- | ------------------------------------ |
| total_reads            | Total number of EPC reads            |
| rssi_avg               | Average RSSI                         |
| rssi_best              | Best RSSI (closest to 0)             |
| rssi_worst             | Worst RSSI (most negative)           |
| antenna_first          | Antenna that first read the tag      |
| antenna_last           | Antenna that last read the tag       |
| antenna_mode           | Antenna that read the tag most often |
| first_time / last_time | Timestamps for first/last reads      |

### Step 3 — Optional Pallet Reference

* If a layout file is provided (`--layout`):

  * Associate EPCs or suffixes to physical positions.
  * Generate sheet `Posicoes_Pallet` in Excel with coverage summary.
* If no layout file is provided:

  * Skip positional analysis, proceed with generic metrics.

### Step 4 — Outputs

* Excel report with multiple tabs
* PNG charts:

  * Reads per EPC
  * Reads per Antenna
  * RSSI boxplot by Antenna
* Console log of processed files

---

## 🧰 Agent Action Map

| Agent Capability      | File / Module          | Purpose                                         |
| --------------------- | ---------------------- | ----------------------------------------------- |
| Add new metric or KPI | `metrics.py`           | Define new columns or aggregates                |
| Add new chart type    | `plots.py`             | Implement extra matplotlib or Plotly visuals    |
| Modify Excel layout   | `report.py`            | Add tabs, conditional formatting, summaries     |
| Add new CLI option    | `analisar_itemtest.py` | Extend argparse with new parameter              |
| Update parsing rules  | `parser.py`            | Adjust metadata extraction or EPC validation    |
| Expand layout logic   | `pallet_layout.py`     | Support new reference schemas or 3D positioning |

---

## 🧠 Suggested AI Tasks

| Task                           | Goal                                                               | Tools                       |
| ------------------------------ | ------------------------------------------------------------------ | --------------------------- |
| `agent.refactor_metrics()`     | Add new statistical columns (e.g. RSSI stddev, antenna coverage %) | `pandas`, `numpy`           |
| `agent.generate_heatmap()`     | Create pallet heatmap visualization from layout file               | `matplotlib`, `seaborn`     |
| `agent.create_pdf_report()`    | Auto-export reports to PDF with summary                            | `matplotlib`, `reportlab`   |
| `agent.add_streamlit_ui()`     | Add simple dashboard to run and view results                       | `streamlit`, `plotly`       |
| `agent.validate_layout_file()` | Verify consistency between layout and EPCs                         | built-in Python, `pandas`   |
| `agent.run_batch_analysis()`   | Run multiple folders sequentially and merge outputs                | `pathlib`, `glob`, `pandas` |

---

## 🧩 Agent Development Rules

1. **Never break CLI arguments** — `--input`, `--output`, `--layout` are mandatory interface points.
2. **Preserve file I/O separation** — no in-place edits inside `/samples` or `/src`.
3. **Never hardcode paths** — all inputs/outputs must use `Pathlib`.
4. **Preserve compatibility with Windows 11 (UTF-8)**.
5. **Ensure safe error handling** — use clear `try/except` with friendly messages.
6. **Comment all new functions** using docstrings describing purpose and parameters.
7. **Follow code style:** PEP8, type hints, 4-space indentation, UTF-8 headers.

---

## 🧠 Future Extensions (AI Roadmap)

| Feature                    | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| 🔄 Multi-reader comparison | Compare results from multiple readers (by Hostname)    |
| 📈 Trend analysis          | Correlate RSSI distributions across tests              |
| 🗺️ Spatial mapping        | Generate pallet heatmap (3D position → read strength)  |
| 🧮 Statistical export      | CSV aggregation for Power BI dashboards                |
| 🌐 Streamlit dashboard     | Upload CSV + layout → visualize coverage interactively |
| 🧾 PDF report              | Generate automatic executive summary in PDF format     |

---

## 🧩 Prompt Template for Agents

Example instruction for a Codex/Responses agent to extend this project:

```markdown
You are an AI software engineer working on the repository ItemTestAnalyzer.
Your task is to extend the application by implementing the following feature:
- [Describe new metric, visualization, or integration].

Use Python 3.11+, follow PEP8, and ensure compatibility with Windows 11.
Do not modify CLI argument names. Keep structure under /src.
Output only modified and new files with complete code content.
```

---

## 🧠 Summary

This repository serves as a **foundation for AI-assisted data analysis of RFID test results**.
It’s designed to be modular, extensible, and maintainable — empowering both engineers and AI agents to evolve it into a more complete analytics platform.

