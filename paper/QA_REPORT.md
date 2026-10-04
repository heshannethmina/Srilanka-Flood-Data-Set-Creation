# Manuscript verification — 24 September 2026

- Compiled with installed MiKTeX using `pdflatex -interaction=nonstopmode
  -halt-on-error`; reference resolution completed across successive builds.
- Final PDF: **6 US Letter pages**, standard IEEEtran conference class, 10-point
  body text, 6 figures, 2 tables and 18 cited bibliography entries.
- Final log: **no overfull boxes or undefined citations/references**. Routine
  underfull-box notices from justified text remain; rendered inspection found no
  associated clipping or overlap.
- Rendered all final pages at 150 dpi and inspected every page. Tables, figures
  and equations are legible, contained within the columns and free of overlapping
  elements. The sixth page contains the complete bibliography.
- Confirmed all **19 PDF font resources are embedded**, including those inside
  figure XObjects.
- Checked nine DNN rows: CSV AP/ECE values match the historical result ledger;
  all nine AP values are present in the manuscript table.
- Verified every citation resolves to a bibliography item, labels are unique,
  and every float and equation cross-reference resolves.
- Verified the episode figure's annual counts sum to **1,464**, drawn from the
  **1,469**-entry catalogue with five post-2024 entries excluded.
- Confirmed the active figure directory contains five generated vector PDFs;
  the sixth figure is the single-column TikZ architecture in the manuscript.
- Confirmed no figure placeholder or unresolved `??` remains. The supplied
  author metadata is present in the manuscript.

The delivery copy in `../output/pdf/sri_lanka_dnn_paper.pdf` is byte-identical
to `main.pdf` (the path is relative to the repository root's `paper` folder).
These are document and evidence-consistency checks, not a replication of model
training. The missing-artifact and submission limitations remain documented in
`REVISION_NOTES.md` and the manuscript.
