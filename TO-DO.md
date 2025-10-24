# 🧭 TO-DO — Next Development Steps

## 🎯 Short-Term Goals (v0.2)

| ID | Task | Priority | Responsible | Notes |
|----|-------|-----------|--------------|-------|
| 01 | Add Streamlit dashboard for local visualization | 🔥 High | AI Agent / Developer | Simple upload UI with EPC stats + coverage charts |
| 02 | Implement pallet heatmap & RSSI/Frequency scatter generator | ✅ Done | AI Agent | Structured run now saves `pallet_heatmap.png` and RSSI vs Frequency scatter charts |
| 03 | Validate layout file content consistency | ⚡ Medium | Developer | Warn if tags in layout are missing from CSVs |
| 04 | Add RSSI standard deviation per EPC | ✅ Done | Developer | Continuous-mode summary now includes `rssi_std` per EPC |
| 05 | Export consolidated CSV for Power BI | ⚡ Medium | Developer | `--summary` workbook now consolidates KPIs; add CSV export for Power BI ingestion |
| 06 | Implement summary PDF generation | 🧩 Medium | AI Agent | Use reportlab or matplotlib to create executive report |
| 07 | Add logging with timestamps and error capture | ✅ Done | Developer | Logging centralizado grava INFO/ERROR em `output/logs/<data>_itemtest_analyzer.log` |
| 08 | Enhance CLI with `--merge` option | 🧩 Medium | Developer | Merge multiple test summaries into one Excel |
| 09 | Improve metadata parsing robustness | ✅ Done | Developer | Completed: parser now normalizes composite header metadata fields |
| 10 | Unit tests for all core modules | ⚙️ Low | Developer | Use pytest with sample data |
| 11 | Add participation percentage per antenna | ✅ Done | AI Agent | `participation_pct` now available in antenna summary and Excel report |
| 12 | Allow expected EPC list without layout | ✅ Done | AI Agent | New `--expected` option loads EPC/suffix presets and flags unexpected tags |
| 13 | Generate per-file textual summary | ✅ Done | AI Agent | Prints metadata, per-antenna stats, and layout coverage in the logs |
| 14 | Garantir execução do CLI como módulo ou script | ✅ Done | Developer | Ajustado bootstrap em `itemtest_analyzer.py` para configurar `sys.path` automaticamente |
| 15 | Corrigir parser para CSV com `;` e decimal `,` | ✅ Done | AI Agent | Parser ajustado e teste de regressão cobrindo EPCs e métricas |
| 16 | Garantir fallback de Hostname no parser | ✅ Done | AI Agent | Hostname passa a usar `ReaderName` ou dados da coluna para alimentar relatórios |
| 17 | Expand structured-mode KPIs in metrics/report | ✅ Done | AI Agent | Coverage, redundancy, balance, RSSI stability, read hotspots, frequency usage, location errors, and face distribution now surfaced across summaries and Excel |
| 18 | Align Excel workbook sheet naming with executive KPIs | ✅ Done | AI Agent | Tabs renamed to mandated Portuguese titles and executive dashboard merges structured/continuous metrics |
| 19 | Expand continuous-mode KPIs (dwell max, idle gaps, congestion) | ✅ Done | AI Agent | Session throughput, inactivity detection, congestion index, and global RSSI propagated to reports/summary |
| 20 | Consolidate global RSSI KPIs and noise detector | ✅ Done | AI Agent | Funções compartilhadas calculam média/desvio, indicador de ruído e agora exportam leituras/EPC para planilhas, sumários e modo contínuo |
| 21 | Implement eight requested KPIs across SPEC, GUIDE, and reports | 🔥 High | AI Agent / Developer | Ensure KPI definitions align across SPEC.md, GUIDE_METRICS.md, and Excel/text outputs with regression coverage |
| 22 | Update reporting templates for new KPI set | ⚡ Medium | Developer | Refresh Excel/summary templates so new KPIs display in Indicadores_Executivos and structured KPI tabs |
| 23 | Validate EPC header 331A handling in parser/export | ⚡ Medium | Developer | Confirm parser normalizes 331A headers and reports surface EPC fields without data loss |

---

## 🧠 Long-Term Enhancements (v0.3+)

- 📈 Power BI integration template.
- 🌐 FastAPI or Streamlit microservice for web execution.
- 🧮 Machine learning module for tag visibility prediction (based on RSSI/antennas).
- 🧾 PDF report with branded layout and pallet diagram.
- 🧱 Dockerfile for cross-platform packaging.
- 🧑‍💻 PyInstaller build to create a Windows `.exe` bundle.
- 🔄 Integration with versioned layout database (to track pallet design changes).

---

## 🧩 AI Agent Collaboration Guidelines

When working on this repository via AI automation:

1. **Follow PEP8** and preserve existing CLI arguments (`--input`, `--output`, `--layout`, `--mode`, `--window`, `--expected`, `--summary`).
2. **Keep all functional code inside `/src`** — no logic in root directory.
3. **Document all functions** with concise docstrings.
4. **Preserve UTF-8 encoding** and Windows path compatibility.
5. **Avoid destructive refactors** — prefer additive commits.
6. **Use AGENTS.md** as the reference for module responsibilities.
7. **Update CHANGELOG.md** and **TO-DO.md** after each major agent commit.

---

## 🧾 Template for Future Changelog Entries

```markdown
## v0.X — [Release Name] (YYYY-MM-DD)
### ✨ Features
- [list new features]

### 🐞 Fixes
- [list bugfixes]

### 🧩 Improvements
- [list code improvements or AI agent refactors]
