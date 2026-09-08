# IEEE DNN research manuscript

**Terrain-Conditioned Deep Neural Networks for Next-Day High-Flow Prediction in
Sri Lanka** — revised 8 September 2026.

This is a seven-page IEEE conference-format research draft focused on MMF-Net's
neural representation, attention, static conditioning, objective and radar
fusion. The original manuscript is preserved in `archive/original-2026-09-08/`.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Editable IEEE source; author fields remain to be completed |
| `main.pdf` | Compiled manuscript |
| `make_figures.py` | Reproducibly builds the five vector data figures |
| `figures/` | Five data figures; architecture is drawn in LaTeX |
| `data/` | Recorded result CSV, Natural Earth outline and catalogue count audit |
| `REVISION_NOTES.md` | Substantive corrections and remaining experimental work |
| `SOURCES_VERIFIED.md` | Primary reference links and verification limits |

## Build

With Python, Matplotlib, NumPy and MiKTeX or TeX Live installed:

```bash
python paper/make_figures.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

No network access or model training is needed to regenerate the figures. The
country outline is vendored locally with provenance in `data/README.md`.
Scientific plots are vector PDFs with embedded fonts; the architecture is TikZ.

The local revision used the Codex bundled Python runtime with plotting packages
in the ignored `tmp/paper_python` directory, and the user's installed MiKTeX.
An ordinary Python installation with Matplotlib and NumPy can run the same
script without those local paths.

## Evidence boundaries

The model scores come from the hand-compiled `docs/RESULTS.md` snapshot, not a
new training run. Their values were retained. Older baseline and graph results
predate the main model's data cutoff and are explicitly separated.

The paper does not claim observed inundation prediction, operational readiness,
statistical significance, a controlled graph improvement or proven pretrained
radar gains. Exact model-eligible counts, prediction-level graphs and confidence
intervals require the missing panel/run artifacts. The catalogue plot is an
actual count of the supplied event file, not a substitute for those artifacts.

Author details and final experimental evidence remain necessary before
submission. See `REVISION_NOTES.md` for the specific evidence gaps. No target
venue or page limit has been selected; standard IEEE conference layout is used.
