# ItemTestAnalyzer

Automates the analysis of **Impinj ItemTest** CSV exports with optional **pallet layout**
references (CSV/XLSX/Markdown).

## Requirements
- Python 3.11+
- `pip install -r requirements.txt`

## CLI Usage
```
python src/itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Results" --layout "C:\RFID\Pallets\Layout.xlsx"
```
Without a layout reference:
```
python src/itemtest_analyzer.py --input "C:\RFID\Tests\CSV" --output "C:\RFID\Results"
```

## Outputs
- One `.xlsx` workbook per processed CSV file
- Charts stored in `output/figures/<file-name>/`
- Text summaries and logs under `output/logs/`

## Notes
- Rows where the `EPC` resembles an IP (e.g., `192.168.68.100`) are ignored.
- Invalid EPCs (non-hex/short) are removed.
- When a **layout** is provided, the report includes a `Pallet_Positions` worksheet and
  highlights expected suffix matches.
