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

## Stages

Run them one at a time. Kaggle GPU sessions cap at ~9–12 hours and `--stage all`
can exceed that; each stage writes its own JSON, so a session that dies partway
loses only the stage in flight.

| Stage | What it does | Answers | Rough cost |
|---|---|---|---|
| `baselines` | persistence · climatology · discharge-percentile rule · gradient-boosted trees | the bar to clear | ~10 min |
| `ladder` | M0 → M5, one change per step | RQ1, RQ3, RQ4 | 2–4 h |
| `leakage` | M3 under a random split | **RQ2** | ~30 min |
| `spatial` | M3 with the Gin basin held out | spatial generalisation | ~30 min |
| `sar` | M6_scalars then M6_cnn | **RQ5** | 3–5 h |
| `all` | all of the above, in order | — | likely over one session |

Useful flags: `--epochs` (default 60, early stopping usually fires sooner),
`--seeds` (ensemble size, default 5), `--image-px` (256 by default; 512 is the
native frame size but quadruples activation memory), `--batch-size` (M6_cnn only).

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

The code is validated end to end against a synthetic panel with the real schema
— every preset, both vision modes, all three protocols, all four baselines.
**No model has been trained on the real data yet.** This directory contains no
results and none should be quoted from it.
