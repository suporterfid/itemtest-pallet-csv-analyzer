# ItemTestAnalyzer

ItemTestAnalyzer processes **Impinj ItemTest** CSV exports and produces KPI-rich reports for both controlled pallet experiments and free-flow operational reads. The CLI automatically prepares Excel workbooks, plots, and textual summaries that highlight tag coverage, antenna behaviour, dwell time, and throughput trends across the analysed sessions.

## Key Features
- **Structured mode** – combine CSV reads with a pallet layout (CSV/XLSX/Markdown) to compute coverage, redundancy, antenna balance, RSSI stability, and layout heatmaps. Expected EPCs can come from the layout and/or a custom list.
- **Continuous mode** – detect EPC entry/exit cycles, dwell time, throughput per minute, concurrency peaks, antenna direction hints, and anomaly/alert flags when tags are not known in advance.
- **Rich reporting** – generate per-file Excel workbooks, PNG charts, human-readable summaries, CSV exports, and an optional consolidated executive summary workbook.
- **Expected EPC list support** – accept inline tokens or external files with full EPCs/suffixes to classify expected vs. unexpected tags even without a pallet layout.

## Installation
1. Install Python 3.11 or newer.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. (Optional) On Windows, use `run.bat` as a shortcut once Python is installed.

## Input Artifacts
- **ItemTest CSV exports** – the tool recognises ItemTest `//` metadata headers, normalises delimiters, coerces numeric columns, and discards invalid/IP-like EPCs.
- **Pallet layouts (optional)** – provide a CSV, XLSX, or Markdown file mapping rows and faces to EPC codes or suffixes. Layouts determine expected EPCs, pallet positions, and coverage breakdowns.
- **Expected list (optional)** – pass a text/CSV file or an inline string with EPCs or 3-character suffixes. Tokens support commas, semicolons, spaces, and `#` comments.

Sample files live in [`samples/`](samples/): `Sample_ItemTest.csv`, `Sample_Pallet_Layout.xlsx`, and `Layout_Pallet_Exemplo.csv`.

## CLI Usage
### Basic commands
Structured mode with a pallet layout:
```bash
python src/itemtest_analyzer.py --input "C:\RFID\Tests" --output "C:\RFID\Results" --layout "C:\RFID\Pallets\Layout.xlsx"
```

Continuous analysis without a layout (mode inferred automatically):
```bash
python src/itemtest_analyzer.py --input "C:\RFID\DockReads" --output "C:\RFID\Results"
```

### Forcing a mode
Use `--mode structured` to require layout-style metrics even if a layout is not provided (supply expected EPCs via `--expected`). Use `--mode continuous` to run the dwell/concurrency pipeline even when a layout file is present.

### Expected EPC sources
Provide a file path:
```bash
python src/itemtest_analyzer.py --input ./csv --output ./results --expected ./expected_epcs.txt
```

…or inline tokens (quotes recommended):
```bash
python src/itemtest_analyzer.py --input ./csv --output ./results --mode continuous --expected "300833B2DDD9014000000000, # comments allowed\n123"
```

### Continuous window tuning
Adjust the dwell grouping window (seconds) for continuous mode:
```bash
python src/itemtest_analyzer.py --input ./csv --output ./results --mode continuous --window 3.5
```

### Executive summary across files
Add `--summary` to export `executive_summary.xlsx` with per-file metrics and aggregated overviews:
```bash
python src/itemtest_analyzer.py --input ./csv --output ./results --mode structured --summary
```

### CLI option reference
| Option | Description |
| --- | --- |
| `--input <dir>` | Directory containing ItemTest CSV exports (required). |
| `--output <dir>` | Destination folder for reports (required). |
| `--layout <file>` | Pallet layout file (CSV/XLSX/Markdown). Enables structured KPIs. |
| `--mode {structured,continuous}` | Force a pipeline. Defaults to structured when a layout is provided, continuous otherwise. |
| `--window <seconds>` | Continuous-mode dwell grouping window (default `2.0`). |
| `--expected <path|string>` | Expected EPC list (file path or inline tokens). Applies to both modes. |
| `--summary` | Produce a consolidated executive workbook summarising all processed CSVs. |

## Outputs
For each CSV file, the tool writes artefacts under `--output`:
- **Excel report** (`*_result.xlsx` or `*_continuous_result.xlsx`) containing:
  | Sheet | Description |
  | --- | --- |
  | `Resumo_por_EPC` | Per-EPC metrics (reads, RSSI, dwell, antenna usage). |
  | `EPCs_inesperados` | EPCs flagged as unexpected based on layout/expected lists. |
  | `Leituras_por_Antena` | Read counts and participation per antenna. |
  | `Posicoes_Pallet` | Layout coverage with read totals (structured mode). |
  | `Structured_KPIs` | Coverage, redundancy, balance, RSSI stability, missing tags, and layout coverage tables (structured mode). |
  | `Fluxo_Contínuo` | Continuous metrics, alerts, EPC/minute counts, concurrency timeline, and dwell events (continuous mode). |
  | `Indicadores_Executivos` | Executive KPI snapshot (always present). |
  | `Metadata` | Parsed ItemTest metadata (reader, session, powers, etc.). |
- **PNG charts** under `<output>/graficos/<file_stem>/` (or `<file_stem>_continuous/`), including reads per EPC, reads per antenna, RSSI boxplots, EPC activity over time, and antenna participation heatmaps (continuous mode).
- **Logs and summaries** under `<output>/logs/`, such as structured summaries, continuous summaries, alerts, EPC-per-minute CSVs, and concurrency timelines. The CLI also stores a global log at `output/logs/<date>_itemtest_analyzer.log` relative to the project root.

When `--summary` is enabled, `executive_summary.xlsx` is saved to the output directory. It contains:
- `Detalhes_Por_Arquivo` – one row per processed CSV with key metrics, file paths, and alert counts.
- `Resumo_Geral` – aggregated averages/totals grouped by analysis mode plus an overall row.

## Sample Workflow
Process the provided sample data in structured mode:
```bash
python src/itemtest_analyzer.py --input samples --output output/sample_run --layout samples/Sample_Pallet_Layout.xlsx --summary
```
Inspect the generated Excel workbooks, PNG charts, and the consolidated `executive_summary.xlsx` in `output/sample_run/`.

## Additional Notes
- EPCs resembling IPv4 addresses and non-hexadecimal codes are automatically ignored.
- Layout-derived expected EPCs are merged with tokens provided via `--expected`.
- Continuous reports surface alerts for atypical dwell times or suspicious antenna coverage to help pinpoint congestion or blind zones.
- All outputs use UTF-8 encoding and are compatible with Windows paths that contain spaces.
