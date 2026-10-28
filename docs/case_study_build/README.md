# Case-study build

How `docs/final_case_study.{docx,pdf}` were produced from `docs/final_case_study.md`.
Nothing here changes code or experimental evidence.

| File | Purpose |
|---|---|
| `report_figures.py` | Draws `fig_effects.png` and `fig_response.png` from `reports/phase5/results.json` (committed evidence) |
| `check_numbers.py` | Recomputes every quantitative claim in the report from committed result files and checks that it appears verbatim |
| `build_docx.js` | Converts the Markdown to DOCX (Node.js, `docx` npm package 9.5.1) |
| `export_pdf.ps1` | Exports the DOCX to PDF with Microsoft Word and prints the page count |

From the repository root:

```bash
python docs/case_study_build/report_figures.py
python docs/case_study_build/check_numbers.py
node docs/case_study_build/build_docx.js
powershell -ExecutionPolicy Bypass -File docs/case_study_build/export_pdf.ps1
```

`build_docx.js` needs the `docx` package (`npm install docx@9.5.1` in any
directory on `NODE_PATH`). `export_pdf.ps1` needs Microsoft Word on Windows.
