# 🧾 CHANGELOG — ItemTestAnalyzer

## Unreleased

### ✨ Features
- Added an RSSI vs Frequency scatter plot that is generated for structured and
  continuous analyses, filtering invalid readings to highlight RF channel
  anomalies alongside existing chart artifacts.
- Introduced a `--summary` CLI flag that aggregates per-file KPIs into an
  `executive_summary.xlsx` workbook for quick executive reviews of batch
  processing runs.
- Standardized the Excel report structure to use the mandated Portuguese sheet
  names (`Resumo_por_EPC`, `Leituras_por_Antena`, `Fluxo_Contínuo`,
  `Indicadores_Executivos`, `Metadata`, `Posicoes_Pallet`, and `EPCs_inesperados`)
  while enriching the executive dashboard with KPIs from both structured and
  continuous analyses.
- Expanded continuous-mode analysis with RSSI variability per EPC, concurrency timelines, read continuity, throughput, and session duration KPIs surfaced across summaries, Excel (Fluxo_Contínuo/Indicadores_Executivos), and log exports.
- Added structured-mode diagnostics for read hotspots, frequency usage, location mismatches, and face-level read distribution, exposing them through metrics, Excel sheets (Structured_KPIs/Posicoes_Pallet) and textual summaries.
- Added continuous-mode KPIs for TagDwellTimeMax, session throughput (reads/min), idle gaps (>5× janela), congestion index, and global RSSI statistics, wiring them through Excel (`Fluxo_Contínuo`/`Indicadores_Executivos`), textual summaries, and the consolidated executive workbook.
- Added regression tests covering continuous-mode metric calculations, including overlapping and sparse read scenarios, to prevent future regressions.
- Implemented structured-mode KPI calculations (coverage rate, antenna balance, RSSI stability, redundancy) and exposed them in the `Structured_KPIs` worksheet and textual summaries.
- Added the `participation_pct` percentage column to the antenna summary, exposing how much each antenna contributes to the total number of reads.
- Introduced configurable expected EPC/suffix lists (via `--expected`) to classify records without a pallet layout, flagging each EPC as `Esperado` or `Inesperado` and always generating the `EPCs_inesperados` worksheet.
- Generated a structured textual summary combining metadata, per-antenna statistics, and layout coverage, registrando-o nos arquivos de log sob `output/logs/` para cada CSV processado.
- Configured centralized logging for the CLI to persist INFO/ERROR entries in `output/logs/<data>_itemtest_analyzer.log` while mirroring messages to the console.
- Metadata parsing agora replica o `Hostname` a partir de `ReaderName` ou da primeira linha válida do CSV, garantindo que o IP do leitor apareça no resumo textual e na planilha `Metadata`.

### 🐞 Fixes
- Corrigido o bootstrap do CLI para aceitar tanto `python src/itemtest_analyzer.py` quanto `python -m src.itemtest_analyzer`, mantendo a compatibilidade com o `run.bat` no Windows.
- Ajustado o parser de CSV para detectar automaticamente delimitadores (`;`) e decimais com vírgula, preservando EPCs e métricas numéricas ao importar testes do ItemTest.

## v0.1 — First AI-generated Release (2025-10-21)

### Overview
Initial release of the **ItemTestAnalyzer** project — a Python-based tool for automated analysis of RFID reading tests from **Impinj ItemTest** software.

This version was entirely generated from an AI Agent specification authored by **Alexandre Vieira dos Santos (Joe)** and assembled by GPT-5.

---

### ✨ Features
- ✅ **Automatic parsing** of ItemTest CSV files with commented headers (`//` lines).  
- ✅ **Metadata extraction**: AntennaIDs, RF Mode, Power per antenna, Session, Inventory Mode, Hostname.  
- ✅ **Filtering rules**:
  - Removes IPs in `EPC` (e.g., `192.168.68.100`);
  - Keeps only valid hexadecimal EPCs ≥ 24 characters.
- ✅ **Computation of metrics per EPC**:
  - total_reads, rssi_avg, rssi_best, rssi_worst
  - antenna_first, antenna_last, antenna_mode
  - timestamps of first/last read
- ✅ **Metrics per Antenna**:
  - total_reads and average RSSI
- ✅ **Optional pallet layout reference** (`--layout`) supporting CSV/XLSX/Markdown.
  - Associates EPCs or suffixes with physical positions.
  - Generates sheet `Posicoes_Pallet` in Excel.
- ✅ **Excel report generation** (via `xlsxwriter`)
  - Tabs: Resumo_por_EPC, Leituras_por_Antena, Metadata, and optionally Posicoes_Pallet.
- ✅ **Automatic chart generation**:
  - Reads per EPC (bar)
  - Reads per Antenna (bar)
  - RSSI per Antenna (boxplot)
- ✅ **UTF-8 + Windows 11 compatibility**
- ✅ **CLI and Batch Execution** (`run.bat`)

### 🐞 Fixes
- Normalized metadata parsing in `read_itemtest_csv`, keeping antenna lists, session, mode index, hostname and power-per-antenna
  pairs consistent even when header lines are split across commas or newlines.

---

### 🧩 Architecture
- Modular code organized under `/src`
- Each functional aspect isolated in its own module:
  - `parser.py` → CSV parsing  
  - `metrics.py` → data aggregation  
  - `pallet_layout.py` → layout reference logic  
  - `plots.py` → visualization  
  - `report.py` → Excel export  
  - `itemtest_analyzer.py` → CLI orchestrator

---

### 🧠 Agent-Ready Design
- `AGENTS.md` defines how AI agents can safely extend the project.  
- Code written with clear docstrings and PEP8 compliance for easy refactoring.  
- Structure supports CI/CD or local execution via Python/PowerShell/Batch.  

---

### 🧰 Known Limitations
- No Power BI export yet (CSV only).
- No Streamlit visualization layer (planned for v0.2).
- Layout cross-validation assumes unique suffixes (3-char overlap not handled).

---

### 📅 Next Milestone (v0.2)
Planned additions:
- Streamlit interactive dashboard.
- Pallet heatmap generation.
- Enhanced metadata parsing (including AntennaID correlation logic).
- Power BI connector (CSV summary export).
- Option for consolidated multi-file Excel summary.

---
