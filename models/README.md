# `models/` — model creation

This directory builds and evaluates **two model families** against the same data
and the same metrics. It is deliberately kept separate from
[`../scripts/`](../scripts/), which builds the **datasets**. The two have no
shared entry point: rebuilding the data and rebuilding the model are always
separate acts, and neither can silently trigger the other.

| Directory | Job | Consumes | Produces |
|---|---|---|---|
| [`../scripts/`](../scripts/) | dataset creation | public APIs (Open-Meteo, NASA POWER, Planetary Computer) | `flood_dataset.parquet`, graph files, SAR frames |
| `models/` | model creation | those two published Kaggle datasets | trained weights, metrics JSON |

## The two models

| | **Model 1 — TF-STGNN** | **Model 2 — MMF-Net** |
|---|---|---|
| Package | [`model1/`](model1/) | [`model2/`](model2/) |
| Temporal encoder | 2-layer GRU + attention pooling | transformer over the 14-day window, one token per day |
| Input layer | features fed as raw scalars | **per-feature periodic (PLR) numerical embeddings** |
| Cross-feature mixing | one linear layer | optional **attention across the 33 channels** |
| Spatial structure | relational GATv2 over the 51-node river graph | **none — per-node by design** |
| Terrain | FiLM conditioning | FiLM conditioning |
| SAR imagery | ResNet-18, embedding **concatenated** | ResNet-18 **pretrained on labelled chips**, fused through a **learned gate** |
| Ladder | M0 → M6 | N0 → N6 |

**Why a second model.** Model 1's first full run produced one uncomfortable
result: the gradient-boosted-tree baseline reaches PR-AUC 0.850 while the best
network reaches 0.742. That is the well-documented failure of neural networks on
tabular data, and the published fix is a better input layer rather than a bigger
network — per-feature *embeddings* instead of raw scalars (Gorishniy et al.,
NeurIPS 2022). Model 2 is that fix, plus two follow-ups model 1 left open:

- **Is the graph earning its place?** Model 1 compares message passing against a
  weak per-node floor (M0, a plain GRU). Model 2 is a strong per-node network,
  so M5 vs N5 is the comparison that actually tests the graph.
- **Was the SAR branch useless, or just fused badly?** M6_cnn returned nothing
  (val event PR-AUC 0.2988 vs 0.3007 without imagery). It asked a randomly
  initialised ResNet-18 to learn what water looks like from a target with a 1.9%
  positive rate. [`model2/pretrain_sar.py`](model2/pretrain_sar.py) trains that
  same encoder first on the 3,489 labelled flood/dry chips in
  `image_manifest.csv` — a **40% positive rate** — and only then attaches it,
  through a gate that can ignore the ~92% of node-days with no fresh frame.

Both families import [`floodlib/`](floodlib/) for the data, the loss, the
metrics and the training loop, so a difference between an M-row and an N-row is
a difference of architecture and never of evaluation protocol.

## Running it — Kaggle

[`kaggle_run.py`](kaggle_run.py) is the only entry point and it **only runs on
Kaggle**. It exits immediately anywhere else.

**Notebook setup**

1. New notebook → **Add Input** → attach both:
   - `uom230429e/sri-lanka-flood-tabular-graph-2003-2025` (required by every stage)
   - `uom230429e/flood-data-set` (required by the SAR stages only)
2. Settings → **Accelerator: GPU** (T4 or P100), **Internet: On** (for the clone).

**Cell 1 — get the code**

```python
!git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
```

**Cell 2 — run a stage**

```python
!python /kaggle/working/repo/models/kaggle_run.py --stage ladder2
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
  process still exits non-zero, so a failure is visible rather than silent. Each
  ladder isolates its presets individually too, so losing N5 does not lose N0–N4.

## Stages

Kaggle GPU sessions cap at ~9–12 hours and the full set exceeds that; each stage
writes its own JSON, so a session that dies partway loses only the stage in
flight. `--stage all` runs them in this order:

| Stage | What it does | Answers | Rough cost |
|---|---|---|---|
| `baselines` | persistence · climatology · discharge-percentile rule · gradient-boosted trees | the bar to clear | ~10 min |
| `ladder2` | **N0 → N5**, model 2, one change per step | RQ6, RQ7, and the tree gap | 2–3 h |
| `sar_pretrain` | trains the SAR encoder on the labelled chips | can a CNN see flooding at all? | ~25 min |
| `ladder` | M0 → M5, model 1, one change per step | RQ1, RQ3, RQ4 | 2–4 h |
| `leakage` | M3 under a random split | **RQ2** | ~30 min |
| `spatial` | M3 with the Gin basin held out | spatial generalisation | ~30 min |
| `sar2` | N6_scalars then N6_gated, using the pretrained encoder | **RQ8** | 3–4 h |
| `sar` | M6_scalars then M6_cnn | **RQ5** | 3–5 h |

Model 1's ladder has already been run once, which is why model 2's comes first
by default: closing the gap to the tree baseline is the open question, and
`sar_pretrain` is cheap and unblocks `sar2`.

Useful flags: `--time-budget-hours` (default 8), `--epochs` (default 60, early
stopping usually fires sooner), `--seeds` (M6/N6 ensemble size, default 5),
`--pretrain-epochs` (default 25), `--image-px` (256 by default; 512 is the native
frame size but quadruples activation memory), `--batch-size` (the CNN presets
only).

### Revisiting one ladder step

`--presets` narrows model 1's ladder, `--presets2` narrows model 2's, and
`--ladder-seeds` overrides the seed count for whatever they select, so a single
step can be re-run as an ensemble without repeating the others:

```python
!python /kaggle/working/repo/models/kaggle_run.py \
    --stage ladder --presets M2,M3 --ladder-seeds 5
```

Overridden runs are written under a `_s<N>` suffix — `M2_temporal_s5.json`
alongside `M2_temporal.json` — because the point of the re-run is to compare
against the original, and overwriting it would destroy the comparison. The
summary table labels them `M2 x5` so the two are never confused.

### The SAR encoder, standalone

```python
!python -m model2.pretrain_sar --out runs/sar_encoder.pt --epochs 25
```

Writes `sar_encoder.pt` plus a `.json` with the validation AP per epoch. Read
that number before reading anything downstream: **a chip-level AP near the 40%
base rate means the encoder learned nothing visual**, and N6_gated should not
then be expected to help. Chips falling in the main experiment's test block are
dropped by joining to the panel's own `split_temporal` column, so pretraining
cannot leak into the reported test scores.

## Output

Everything lands in `/kaggle/working/runs`:

- `<preset>_<protocol>.json` — every metric plus the full model and training
  config, so a run is reproducible from its own output
- `<preset>_<protocol>_preds.npz` — test probabilities with day / node / event
  indices, so figures can be regenerated without retraining
- `sar_encoder.pt` / `.json` — the pretrained encoder and its training curve

A summary table prints at the end of each invocation, grouped baselines → model 1
→ model 2. Hit **Save Version** to keep `/kaggle/working` as downloadable
notebook output. Offline, [`report.py`](report.py) renders the same table from a
downloaded `runs/` directory.

## What is in here

**Shared** — [`floodlib/`](floodlib/), the data and the evaluation:

| File | Contents |
|---|---|
| [`floodlib/schema.py`](floodlib/schema.py) | the column contract: feature lists, targets, `BaseModelConfig` |
| [`floodlib/traincfg.py`](floodlib/traincfg.py) | `TrainConfig`, `Preset` |
| [`floodlib/data.py`](floodlib/data.py) | long panel → dense `[T, N, F]`, train-only normalisation, split masks, batcher |
| [`floodlib/graph.py`](floodlib/graph.py) | relational `edge_index` + 4-dim edge attributes (model 1 only) |
| [`floodlib/blocks.py`](floodlib/blocks.py) | layers both families use: GRU encoder, FiLM, ResNet-18 SAR stem |
| [`floodlib/engine.py`](floodlib/engine.py) | the training loop, ensembling, calibration, thresholding |
| [`floodlib/losses.py`](floodlib/losses.py) | focal × `label_confidence` + Huber |
| [`floodlib/metrics.py`](floodlib/metrics.py) | PR-AUC, Brier decomposition, ECE, POD/FAR/CSI, event lead time |
| [`floodlib/calibrate.py`](floodlib/calibrate.py) | temperature scaling, isotonic (PAVA) |
| [`floodlib/sar.py`](floodlib/sar.py) | SAR index join, causal frame map, LRU frame store |
| [`floodlib/baselines.py`](floodlib/baselines.py) | the four reference models |

**Model 1** — [`model1/`](model1/):

| File | Contents |
|---|---|
| [`model1/config.py`](model1/config.py) | `ModelConfig`, the M0–M6 presets |
| [`model1/modules.py`](model1/modules.py) | the relational GATv2 |
| [`model1/model.py`](model1/model.py) | the assembled network and its six heads |
| [`model1/train.py`](model1/train.py) | preset table + model factory, onto the shared engine |

**Model 2** — [`model2/`](model2/):

| File | Contents |
|---|---|
| [`model2/config.py`](model2/config.py) | `MMFConfig`, the N0–N6 presets |
| [`model2/modules.py`](model2/modules.py) | numerical embeddings, temporal + feature transformers, gated SAR fusion |
| [`model2/model.py`](model2/model.py) | the assembled network and its six heads |
| [`model2/train.py`](model2/train.py) | preset table + model factory, onto the shared engine |
| [`model2/pretrain_sar.py`](model2/pretrain_sar.py) | chip-level SAR encoder pretraining |

Design rationale is in [`../docs/MODEL.md`](../docs/MODEL.md).

## Dependencies

Kaggle's default image already has everything: `torch`, `pandas`, `numpy`,
`pyarrow`, `lightgbm`, `pillow`. There is **no PyTorch Geometric** requirement —
with 51 nodes and 239 edges the relational GATv2 is written directly against
torch scatter ops.

## Status

**Model 1** — first full run on the real data: 2026-08-02, Kaggle T4, 5.07 h, all
five stages, none skipped or failed. Headline: no model beats the
gradient-boosted-tree baseline on PR-AUC (best 0.742 vs 0.850), but every model
from M1 up beats every baseline on pre-onset event detection, which is the metric
§7.8 pre-registers as the one that matters.

Two caveats attach to those numbers. **M2 leads on event detection (0.408) but is
a single seed**, while M5 and M6 are 5-seed ensembles — that comparison is not yet
fair, and `--presets M2,M3 --ladder-seeds 5` is the run that fixes it. **RQ1
(directed flow edges) remains unresolved**: M2→M3 falls on every test metric but
rises on validation, at n=1 each.

**Model 2** — first full run 2026-08-10, Kaggle T4: `ladder2` 2.45 h, `sar2`
6.62 h, `sar_pretrain` failed (fixed since — the manifest is not in the Kaggle
dataset, only in the repo).

| Preset | Change | PR-AUC | ev.det | FAR | ECE | params |
|---|---|---|---|---|---|---|
| N0 | linear feature embeddings | 0.6083 | 0.290 | 0.412 | 0.0024 | 551k |
| N1 | **+ periodic (PLR) embeddings** | **0.7605** | 0.420 | 0.325 | 0.0053 | 555k |
| N2 | + cross-feature attention | 0.7738 | **0.423** | 0.313 | 0.0056 | 693k |
| N3 | + FiLM terrain | **0.8269** | 0.197 | 0.110 | 0.0032 | 711k |
| N4 | + focal × confidence | 0.7592 | 0.220 | 0.138 | 0.0525 | 711k |
| N5 ×5 | + ensemble + temperature | 0.7846 | 0.208 | 0.145 | 0.0043 | 711k |
| N6_scalars ×5 | + SAR scalars | 0.7284 | 0.397 | 0.344 | 0.0055 | 719k |
| N6_gated ×5 | + gated SAR CNN | **0.8310** | 0.217 | 0.107 | 0.0066 | 11,967k |

Three findings:

1. **Periodic numerical embeddings are the single largest gain in the project.**
   N0 → N1 changes nothing but the input layer and moves PR-AUC +0.152 and event
   detection +0.130. The tabular-embedding hypothesis holds.
2. **The gap to gradient-boosted trees is largely closed.** Model 1's best was
   0.7421 against the GBT's 0.8496 — 0.108 behind. N3 reaches 0.8269 and clears
   the discharge-percentile rule (0.8164) outright, at 711k parameters.
3. **The focal rung is a regression, not a gain.** N3 → N4 costs 0.068 PR-AUC and
   makes calibration 16× worse (ECE 0.0032 → 0.0525); the ensemble in N5 recovers
   only part of it. `N5_bce` and `N6_gated_bce` apply the ensemble and calibrator
   to the loss that was actually working, and have not been run yet.

**Do not read the `ev.det` column down the ladder** — the thresholds differ
(FAR ranges 0.107 to 0.412), so it compares operating points rather than models.
Use [`rethreshold.py`](rethreshold.py) to force a common false-alarm rate first.

Confirmed by this run: imagery covers **9 of 51 nodes**, 2,578 frames.
