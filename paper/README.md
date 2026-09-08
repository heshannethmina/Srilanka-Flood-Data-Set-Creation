# Paper

IEEE conference-format manuscript built from the **model 2 (MMF-Net)** results of
2026-08-10/11 and the baseline / protocol runs of 2026-08-02.

```bash
python paper/make_figures.py          # regenerate figures/*.pdf
cd paper && pdflatex main && pdflatex main
```

Compiles clean with MiKTeX/TeX Live: 7 pages, no undefined references, no
overfull boxes above 10 pt.

## What is where

| File | Contents |
|---|---|
| `main.tex` | the manuscript (IEEEtran, `conference` option) |
| `make_figures.py` | generates the four data figures as vector PDFs |
| `figures/*.pdf` | generated — do not edit by hand |

## Figures

| # | Figure | Source |
|---|---|---|
| 1 | Study-area map | **PLACEHOLDER** |
| 2 | MMF-Net architecture | TikZ, inline in `main.tex` |
| 3 | Ablation ladder PR-AUC | `make_figures.py :: fig_ladder` |
| 4 | Seed-noise comparison | `make_figures.py :: fig_seed_noise` |
| 5 | Leakage divergence | `make_figures.py :: fig_leakage` |
| 6 | Operating points | `make_figures.py :: fig_operating_points` |
| 7 | Reliability diagram | **PLACEHOLDER** |

The figure palette is an Okabe–Ito subset validated colourblind-safe (worst
adjacent CVD $\Delta E$ 11.0 deutan; normal-vision floor 24.2), and colour is
never the only encoding — every series is also direct-labelled.

## The two placeholders

Both are boxed and red-flagged in the compiled PDF so they cannot be missed.

**Fig. 1 — study-area map.** Needs `data/processed/nodes.csv` and `edges.csv`
plus a basin polygon layer (GADM level 2, or the Survey Department basin layer).
Draw the 51 nodes coloured by climatic zone and the 35 directed flow edges.

**Fig. 7 — reliability diagram.** Needs `runs/N5_bce_temporal_preds.npz`, which
is produced by the Kaggle run and is not in the repository. Use
`floodlib.metrics.reliability_curve` with 15 equal-count bins; overlay the
uncalibrated focal rung N4 for contrast.

## Numbers

Every figure in `make_figures.py` hard-codes its values so the paper builds
without `runs/`. The same values appear in `docs/RESULTS.md`. **If a run is
repeated, update both.**

## Before submitting

- [ ] Fill the author block in `main.tex` (bracketed placeholders are brace-wrapped
      because a line starting with `[` directly after `\\` parses as an optional
      length argument — keep the braces).
- [ ] Replace Fig. 1 and Fig. 7.
- [ ] Fill the repository URL in the Reproducibility section.
- [ ] Check the venue's page limit — the manuscript is currently 7 pages including
      references.
- [ ] Re-read §VII Limitations against the current state of the code; it states
      the missing embargo, the reanalysis-derived labels, and the unresolved
      river-topology question honestly, and those claims should stay accurate.
