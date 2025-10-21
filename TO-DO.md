# 🧭 TO-DO — Next Development Steps

## 🎯 Short-Term Goals (v0.2)

| ID | Task | Priority | Responsible | Notes |
|----|-------|-----------|--------------|-------|
| 01 | Add Streamlit dashboard for local visualization | 🔥 High | AI Agent / Developer | Simple upload UI with EPC stats + coverage charts |
| 02 | Implement pallet heatmap generator | 🔥 High | AI Agent | Use matplotlib/seaborn 2D grid with color by RSSI mean |
| 03 | Validate layout file content consistency | ⚡ Medium | Developer | Warn if tags in layout are missing from CSVs |
| 04 | Add RSSI standard deviation per EPC | ⚡ Medium | Developer | New column in `metrics.py` |
| 05 | Export consolidated CSV for Power BI | ⚡ Medium | Developer | Combine summaries across all CSVs |
| 06 | Implement summary PDF generation | 🧩 Medium | AI Agent | Use reportlab or matplotlib to create executive report |
| 07 | Add logging with timestamps and error capture | ✅ Done | Developer | Logging centralizado grava INFO/ERROR em `output/logs/<data>_analisar_itemtest.log` |
| 08 | Enhance CLI with `--merge` option | 🧩 Medium | Developer | Merge multiple test summaries into one Excel |
| 09 | Improve metadata parsing robustness | ✅ Done | Developer | Completed: parser now normalizes composite header metadata fields |
| 10 | Unit tests for all core modules | ⚙️ Low | Developer | Use pytest with sample data |
| 11 | Add participation percentage per antenna | ✅ Done | AI Agent | `participation_pct` now available in antenna summary and Excel report |
| 12 | Allow expected EPC list without layout | ✅ Done | AI Agent | New `--expected` option loads EPC/suffix presets and flags unexpected tags |
| 13 | Generate per-file textual summary | ✅ Done | AI Agent | Prints metadata, per-antenna stats, and layout coverage in the logs |
| 14 | Garantir execução do CLI como módulo ou script | ✅ Done | Developer | Ajustado bootstrap em `analisar_itemtest.py` para configurar `sys.path` automaticamente |
| 15 | Corrigir parser para CSV com `;` e decimal `,` | ✅ Done | AI Agent | Parser ajustado e teste de regressão cobrindo EPCs e métricas |
| 16 | Garantir fallback de Hostname no parser | ✅ Done | AI Agent | Hostname passa a usar `ReaderName` ou dados da coluna para alimentar relatórios |

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

1. **Follow PEP8** and preserve existing CLI arguments (`--input`, `--output`, `--layout`).
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
