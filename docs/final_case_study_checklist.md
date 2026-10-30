# Case-Study Submission Checklist

For `docs/final_case_study.{md,docx,pdf}`. Items marked **You** need your
personal action or confirmation before submission.

## Verified by the build

| Item | Status |
|---|---|
| Format | IEEE conference layout: two 3.5-inch columns, US Letter, Times New Roman 10 pt, Roman-numeral sections, `TABLE I` / `Fig. 1` labels, IEEE-style references. Built to the template's specifications, not from IEEE's official template file |
| Page count | 4 pages, counted from the rendered PDF; limit is 8 |
| Three independent operating points | 6, 14, and 22 processes; 1,000 workloads each (Table II) |
| Numbers match committed evidence | 38/38 claims recomputed from result files and found verbatim (`docs/case_study_build/check_numbers.py`) |
| Figures from committed data | `docs/case_study_build/report_figures.py` reads `reports/phase5/results.json`; no values hand-entered |
| Repository link | https://github.com/ujvall/schedX/commit/e848cf2 resolves (HTTP 200); repository public |
| No kernel/hardware claims | stated as simulator-only in the abstract and limitations |
| References | 12 sources checked against Crossref, OpenAlex, JMLR, JSTOR, or Open Library (details below) |

## You must act on or confirm

1. **AI-generated content.** This text, including its plainer rewording, was
   written by an AI assistant. The course limits AI-generated content to under
   10%, so **rewording by AI does not bring it within that limit**. Use it as a
   fact base: rewrite it in your own words and make sure you can explain and
   defend every sentence. No detector score can be predicted or promised.
2. **AI-use disclosure (Section IX).** Check what your course or university
   requires, then edit the disclosure. Keep it factual: the code, tests, and
   documentation were developed with substantial AI assistance. Fill in the
   bracketed note with the work and review you personally did.
3. **Placeholders** on page 1: course name and code, instructor, university,
   and submission date.
4. **Contribution statement.** Confirm "individual project by Ujval Sai" and
   that `cremebits` is your Git identity.
5. **Format and page-limit rules.** If your course requires IEEE's official
   Word template or A4 paper, tell me or paste the text into it. If your course
   follows IEEE strictly, remove the tagline under the title. At 4 pages the
   report is within the limit either way.
6. **Plagiarism check.** Run the institution's tool on your final rewritten
   version.

## Reference verification notes

| Ref. | Verified via | Note for you |
|---|---|---|
| [1] Silberschatz, Galvin, Gagne (Wiley 2018) | Open Library (title, authors, publisher, ISBN) | Edition number not given by the registry; cited by ISBN |
| [2] Kleinrock (1967), JACM | Crossref DOI | **Check** that its content supports "queueing analyses of time-shared systems … quantum size affects delay" (read the abstract) |
| [3] Schrage (1968), Operations Research | Crossref DOI | Supports SRTF/SRPT optimality |
| [4] Pedregosa et al. (2011), JMLR | jmlr.org metadata (vol. 12, pp. 2825–2830) | — |
| [5] Breiman (2001), Machine Learning | Crossref DOI | — |
| [6] Kaplan & Meier (1958), JASA | Crossref DOI | — |
| [7] Efron (1979), Annals of Statistics | Crossref DOI | — |
| [8] Good (2005), Springer | Crossref DOI + OpenAlex (author) | — |
| [9] Holm (1979), Scand. J. Statistics | OpenAlex + JSTOR stable page | No issue number or Crossref DOI; cited by JSTOR URL |
| [10] Wilcoxon (1945), Biometrics Bulletin | Crossref DOI | **End page 83 not confirmed** (registries list start page 80 only); check on JSTOR |
| [11] Pratt (1959), JASA | Crossref DOI | — |
| [12] Negi & Kumar (2005), TENCON | Crossref DOI | Cited only as an example of ML applied to Linux process scheduling (supported by its title) |

## Writing-style note

No humanization tool or academic-editing skill was used. The wording was
simplified for readability, not to avoid AI detection, and uses "we" only for
the project's method. Section IX states the AI assistance.
