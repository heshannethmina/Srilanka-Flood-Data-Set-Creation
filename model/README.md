# `model/` — model creation

This directory builds and evaluates the **TF-STGNN**. It is deliberately kept
separate from [`../scripts/`](../scripts/), which builds the **datasets**. The two
have no shared entry point: rebuilding the data and rebuilding the model are
always separate acts, and neither can silently trigger the other.

| Directory | Job | Consumes | Produces |
|---|---|---|---|
| [`../scripts/`](../scripts/) | dataset creation | public APIs (Open-Meteo, NASA POWER, Planetary Computer) | `flood_dataset.parquet`, graph files, SAR frames |
| `model/` | model creation | those two published Kaggle datasets | trained weights, metrics JSON |

## Running it — Kaggle

[`kaggle_run.py`](kaggle_run.py) is the only entry point and it **only runs on
Kaggle**. It exits immediately anywhere else.

**Notebook setup**

1. New notebook → **Add Input** → attach both:
   - `uom230429e/sri-lanka-flood-tabular-graph-2003-2025` (required by every stage)
   - `uom230429e/flood-data-set` (required by the `sar` stage only)
2. Settings → **Accelerator: GPU** (T4 or P100), **Internet: On** (for the clone).

**Cell 1 — get the code**

```python
!git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
```

**Cell 2 — run a stage**

```python
!python /kaggle/working/repo/model/kaggle_run.py --stage baselines
```

## Save & Run All

[`../notebooks/tfstgnn_kaggle.ipynb`](../notebooks/tfstgnn_kaggle.ipynb) is a
ready-made notebook for batch execution: import it on Kaggle
(**File → Import Notebook**), attach the two datasets, set the accelerator to GPU
and internet to On, then **Save & Run All (Commit)**.

Two properties make that safe to leave unattended:

- **A time budget.** Kaggle kills a GPU session at ~9 h and a killed session
  saves nothing. Stages run cheapest-and-most-decisive first, and any stage that
  will not fit inside `--time-budget-hours` (default 8) is skipped with the exact
  command to finish it in a second session.
- **Per-stage fault isolation.** A stage that crashes is logged and the run
  continues; four finished stages are never lost to the fifth one failing. The
  process still exits non-zero, so a failure is visible rather than silent. The
  ladder isolates each preset individually too, so losing M5 does not lose M0–M4.

## Stages

Kaggle GPU sessions cap at ~9–12 hours and the full set exceeds that; each stage
writes its own JSON, so a session that dies partway loses only the stage in
flight.

| Stage | What it does | Answers | Rough cost |
|---|---|---|---|
| `baselines` | persistence · climatology · discharge-percentile rule · gradient-boosted trees | the bar to clear | ~10 min |
| `ladder` | M0 → M5, one change per step | RQ1, RQ3, RQ4 | 2–4 h |
| `leakage` | M3 under a random split | **RQ2** | ~30 min |
| `spatial` | M3 with the Gin basin held out | spatial generalisation | ~30 min |
| `sar` | M6_scalars then M6_cnn | **RQ5** | 3–5 h |
| `all` | all of the above, in order | — | likely over one session |

Useful flags: `--time-budget-hours` (default 8), `--epochs` (default 60, early
stopping usually fires sooner), `--seeds` (M6 ensemble size, default 5),
`--image-px` (256 by default; 512 is the native frame size but quadruples
activation memory), `--batch-size` (M6_cnn only).

### Revisiting one ladder step

`--presets` narrows the ladder and `--ladder-seeds` overrides the seed count for
whatever it selects, so a single step can be re-run as an ensemble without
repeating the other five:

```python
!python /kaggle/working/repo/model/kaggle_run.py \
    --stage ladder --presets M2,M3 --ladder-seeds 5
```

Overridden runs are written under a `_s<N>` suffix — `M2_temporal_s5.json`
alongside `M2_temporal.json` — because the point of the re-run is to compare
against the original, and overwriting it would destroy the comparison. The
summary table labels them `M2 x5` so the two are never confused.

## Output

Everything lands in `/kaggle/working/runs`:

- `<preset>_<protocol>.json` — every metric plus the full model and training
  config, so a run is reproducible from its own output
- `<preset>_<protocol>_preds.npz` — test probabilities with day / node / event
  indices, so figures can be regenerated without retraining

A summary table prints at the end of each invocation. Hit **Save Version** to
keep `/kaggle/working` as downloadable notebook output.

## What is in here

| File | Contents |
|---|---|
| [`kaggle_run.py`](kaggle_run.py) | the Kaggle entry point — stages, environment checks, summary table |
| [`tfstgnn/config.py`](tfstgnn/config.py) | feature lists, `ModelConfig`, `TrainConfig`, the M0–M6 presets |
| [`tfstgnn/data.py`](tfstgnn/data.py) | long panel → dense `[T, N, F]`, train-only normalisation, split masks, batcher |
| [`tfstgnn/graph.py`](tfstgnn/graph.py) | relational `edge_index` + 4-dim edge attributes |
| [`tfstgnn/modules.py`](tfstgnn/modules.py) | GRU + attention pooling, FiLM, ResNet-18 SAR stem, relational GATv2 |
| [`tfstgnn/model.py`](tfstgnn/model.py) | the assembled network and its six heads |
| [`tfstgnn/losses.py`](tfstgnn/losses.py) | focal × `label_confidence` + Huber |
| [`tfstgnn/metrics.py`](tfstgnn/metrics.py) | PR-AUC, Brier decomposition, ECE, POD/FAR/CSI, event lead time |
| [`tfstgnn/calibrate.py`](tfstgnn/calibrate.py) | temperature scaling, isotonic (PAVA) |
| [`tfstgnn/train.py`](tfstgnn/train.py) | training loop, ensembling, calibration |
| [`tfstgnn/baselines.py`](tfstgnn/baselines.py) | the four reference models |
| [`tfstgnn/sar.py`](tfstgnn/sar.py) | SAR index join, causal frame map, LRU frame store |

Design rationale for each of these is in [`../docs/MODEL.md`](../docs/MODEL.md).

## Dependencies

Kaggle's default image already has everything: `torch`, `pandas`, `numpy`,
`pyarrow`, `lightgbm`, `pillow`. There is **no PyTorch Geometric** requirement —
with 51 nodes and 239 edges the relational GATv2 is written directly against
torch scatter ops.

## Status

First full run on the real data: 2026-08-02, Kaggle T4, 5.07 h, all five stages,
none skipped or failed. Headline: no model beats the gradient-boosted-tree
baseline on PR-AUC (best 0.742 vs 0.850), but every model from M1 up beats every
baseline on pre-onset event detection, which is the metric §7.8 pre-registers as
the one that matters.

Two caveats attach to those numbers. **M2 leads on event detection (0.408) but is
a single seed**, while M5 and M6 are 5-seed ensembles — that comparison is not yet
fair, and `--presets M2,M3 --ladder-seeds 5` is the run that fixes it. **RQ1
(directed flow edges) remains unresolved**: M2→M3 falls on every test metric but
rises on validation, at n=1 each.
