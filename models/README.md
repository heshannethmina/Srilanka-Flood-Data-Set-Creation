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

## The three models

| | **Model 1 — TF-STGNN** | **Model 2 — MMF-Net** | **Model 3 — STG-Former** |
|---|---|---|---|
| Package | [`model1/`](model1/) | [`model2/`](model2/) | [`model3/`](model3/) |
| Temporal encoder | 2-layer GRU + attention pooling | transformer over the 14-day window, one token per day | model 2's, unchanged |
| Input layer | features fed as raw scalars | **per-feature periodic (PLR) numerical embeddings** | model 2's, unchanged |
| Cross-feature mixing | one linear layer | optional **attention across the 33 channels** | model 2's, unchanged |
| Spatial structure | relational GATv2 over the 51-node river graph | **none — per-node by design** | model 1's GATv2, unchanged |
| Terrain | FiLM conditioning | FiLM conditioning | FiLM conditioning |
| SAR imagery | ResNet-18, embedding **concatenated** | ResNet-18 **pretrained on labelled chips**, fused through a **learned gate** | model 2's gate; **off every default rung** |
| Ladder | M0 → M6 | N0 → N6 | P0 → P4 |

Model 3 imports both halves rather than reimplementing them, so "model 3 minus
model 2 equals the graph" stays true when either is edited.

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

All three families import [`floodlib/`](floodlib/) for the data, the loss, the
metrics and the training loop, so a difference between an M-row, an N-row and a
P-row is a difference of architecture and never of evaluation protocol.

**Why a third model.** Models 1 and 2 between them left RQ1 not merely
unresolved but *unanswerable*. The comparison being read as "the graph does not
help" — N5_bce 0.8355 without a graph against M5 0.7421 with one — varies three
things at once: the encoder, the loss (M5 sits on the focal rung that model 2
later measured at −0.0509 PR-AUC), and the data extent (model 1's 2026-08-02 run
predates `truncate_after="2024-12-31"`). Model 3 fills the empty cell of the
2×2 — model 2's input layer *with* model 1's graph — so the graph is the only
moving part:

| | weak input layer | PLR numerical embeddings |
|---|---|---|
| **no graph** | N0 — 0.6083 | N5_bce — 0.8355 |
| **relational graph** | M5 — 0.7421 (focal loss) | **model 3** |

Its `P3 − P0_x5` contrast holds encoder, loss, panel, seed count and calibrator
fixed. That interval is the RQ1 answer; nothing already run is.

## Model 4 — ready-to-run follow-up

[The Kaggle notebook](../notebooks/tfstgnn_kaggle.ipynb) now runs **Hydro-TEM**,
a graph-free temporal neural ensemble, alongside a freshly trained Model 2
architecture control and matched baselines through `kaggle_run.py --stage model4`.
It follows the existing clone-and-run procedure: push the repository, attach
the tabular dataset, enable GPU and Internet, and Run All without editing code.
Outputs go to `/kaggle/working/runs/model4` and are packaged in `runs.zip`.
See [Model 4's design and evaluation contract](../docs/MODEL4.md).
Model 4 has no measured result yet. The older CLI stages below remain available.

## Running Models 1–4 — Kaggle

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

## Earlier CLI batch workflow (Models 1–3)

The shared CLI supports the Models 1–3 batch with `--stage all`. The current
[`notebook`](../notebooks/tfstgnn_kaggle.ipynb) uses the same CLI with
`--stage model4` for its dedicated experiment. The following stage descriptions
apply to the older batch schedule.

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
| `ladder3` | **P0 → P3**, model 3, one change per step | **RQ1** | ~4 h |
| `onset` | `P4_onset` and its paired graph-free control | early warning | ~3 h |
| `ladder2` | **N0 → N5**, model 2, one change per step | RQ6, RQ7, and the tree gap | 2–3 h |
| `sar_pretrain` | trains the SAR encoder on the labelled chips | can a CNN see flooding at all? | ~25 min |
| `ladder` | M0 → M5, model 1, one change per step | RQ1, RQ3, RQ4 | 2–4 h |
| `leakage` | M3 under a random split | **RQ2** | ~30 min |
| `spatial` | M3 with the Gin basin held out | spatial generalisation | ~30 min |
| `sar2` | N6_scalars then N6_gated, using the pretrained encoder | **RQ8** | 3–4 h |
| `sar` | M6_scalars then M6_cnn | **RQ5** | 3–5 h |

Both earlier ladders have already been run, which is why model 3's comes first
by default: RQ1 is the open question and `ladder3` is the only stage that can
answer it.

One run belongs beside it and is a rung of model 1's ladder rather than a stage:

```python
!python /kaggle/working/repo/models/kaggle_run.py \
    --stage ladder --presets M5_bce --ladder-seeds 5      # ~40 min
```

Model 1's M4/M5/M6 all sit on the focal loss that model 2 showed costs +0.0509
PR-AUC, so every published M-vs-N comparison is confounded by it. `M5_bce` is
what makes a three-family results table legitimate.

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
- `<preset>_<protocol>_diag.json` — the training diagnostics (below)
- `<preset>_<protocol>_preds.npz` — test probabilities with day / node / event
  indices, so figures can be regenerated without retraining
- `sar_encoder.pt` / `.json` — the pretrained encoder and its training curve

A summary table prints at the end of each invocation, grouped baselines → model 1
→ model 2. Hit **Save Version** to keep `/kaggle/working` as downloadable
notebook output. Offline, [`results_doc.py`](results_doc.py) writes the full
results document — headline table with seed error bars, the pre-declared
contrasts with paired confidence intervals, and the matched-FAR early-warning
table — straight into [`../docs/RESULTS.md`](../docs/RESULTS.md):

```bash
python models/results_doc.py runs/ --out docs/RESULTS.md
```

Run it after every Kaggle session and commit the result, so the paper's numbers
and the repository's numbers cannot drift apart. [`report.py`](report.py) renders
just the summary table from a
downloaded `runs/` directory.

### Reading the diagnostics

A results table says *what* a model scored, never why. Every run also writes
`_diag.json`, and [`diagnose.py`](diagnose.py) turns it into decisions:

```bash
python models/diagnose.py runs/                 # every run
python models/diagnose.py runs/N3_temporal_diag.json --episodes
```

| Section | The question it answers |
|---|---|
| seed spread | is the gap between two rungs bigger than the noise between seeds? |
| early stopping | did patience fire while the model was still improving? |
| gradients | is `grad_clip` doing the optimiser's job (`lr` too high) or nothing? |
| per-head loss | are the auxiliary heads learning, or just consuming capacity? |
| episodes | are the misses **threshold** misses (signal present, ranked too low) or **blind** ones (no signal at all)? These have opposite fixes. |
| by severity | is detection *worse* on the largest floods? |
| by zone / position | is failure concentrated upstream (features/lookback) or at outlets (routing)? |
| SAR gate | did the gated imagery branch ever actually open? |

Each section ends in a verdict rather than a number. The verdicts flag a
condition; they do not promise the fix will work.

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

**Model 3** — [`model3/`](model3/):

| File | Contents |
|---|---|
| [`model3/config.py`](model3/config.py) | `STGFConfig`, the P0–P4 presets, `LADDER`, `ONSET_WEIGHTS` |
| [`model3/model.py`](model3/model.py) | the assembly — model 2's encoder, model 1's GATv2, both imported |
| [`model3/train.py`](model3/train.py) | preset table + model factory, onto the shared engine |

Model 3 owns no layers of its own. That is deliberate: it is a *comparison*, and
a private copy of `TemporalTransformer` or `RelationalGATv2` could drift from the
original and quietly invalidate it.

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

**Model 2** — Kaggle T4, 2026-08-10: `ladder2` 2.45 h, `sar2` 6.62 h; 2026-08-11:
`N5_bce` 1.61 h.

| Preset | Change | PR-AUC | ev.det | FAR | ECE | params |
|---|---|---|---|---|---|---|
| N0 | linear feature embeddings | 0.6083 | 0.290 | 0.412 | 0.0024 | 551k |
| N1 | **+ periodic (PLR) embeddings** | **0.7605** | 0.420 | 0.325 | 0.0053 | 555k |
| N2 | + cross-feature attention | 0.7738 | **0.423** | 0.313 | 0.0056 | 693k |
| N3 | + FiLM terrain | 0.8269 | 0.197 | 0.110 | 0.0032 | 711k |
| N4 | + focal × confidence | 0.7592 | 0.220 | 0.138 | 0.0525 | 711k |
| N5 ×5 | + ensemble + temperature, on focal | 0.7846 | 0.208 | 0.145 | 0.0043 | 711k |
| **N5_bce ×5** | **the same, on BCE** | **0.8355** | 0.220 | 0.120 | **0.0016** | 711k |
| N6_scalars ×5 | + SAR scalars | 0.7284 | 0.397 | 0.344 | 0.0055 | 719k |
| N6_gated ×5 | + gated SAR CNN | 0.8310 | 0.217 | 0.107 | 0.0066 | 11,967k |

`N5_bce` is the headline model: best PR-AUC, ECE, Brier (0.0080), CSI (0.561) and
F1 (0.719) in the project, at 711k parameters. Four findings:

1. **Periodic numerical embeddings are the single largest gain in the project.**
   N0 → N1 changes nothing but the input layer and moves PR-AUC +0.152 and event
   detection +0.130. The tabular-embedding hypothesis holds.
2. **The focal loss is a defect, not a cost.** N5 → N5_bce changes *only* the
   loss — same architecture, same 5 seeds, same temperature scaling — and gains
   **+0.0509 PR-AUC** while cutting ECE to a third. Model 1's whole upper ladder
   (M4, M5, M6) sits on the same rung, so its published numbers are understated
   and `M5_bce` is worth running.
3. **The SAR branch adds nothing.** N5_bce (711k, no imagery) beats N6_gated
   (11,967k, imagery) outright. What looked like a gain from vision was the BCE
   loss recovering what focal had destroyed. Imagery has now lost twice on the
   merits, and it only ever reached 9 of the 51 nodes.
4. **The gap to gradient-boosted trees is 0.0141** (0.8355 vs 0.8496). Model 1
   lost by 0.1075, so 87% of it is closed, and N5_bce clears the operational
   discharge-percentile rule (0.8164) by a clear margin.

Two caveats. **The top three rows may not be separable**: best validation episode
PR-AUC varied 0.3375–0.3856 across N5_bce's five seeds, a range of 0.048, so
0.8355 / 0.8310 / 0.8269 needs the per-seed error bar in `_diag.json` before any
ordering is claimed. And **do not read the `ev.det` column down the ladder** —
the thresholds differ (FAR spans 0.107 to 0.412), so it compares operating points
rather than models; use [`rethreshold.py`](rethreshold.py) first.

From the gradient logging: the clip binds on ~30% of steps early, dips to 4–10%
after warmup, then climbs to **44–54% past epoch 40**, and seed 3 exhausted the
60-epoch budget without early stopping. `grad_clip=2.0` and `epochs=80` are the
cheap things left to try.

**`sar_pretrain` has never completed.** Three bugs, all fixed but not yet re-run:
the manifest is in the repo rather than the Kaggle dataset; the join used
`target_date` when Sentinel-1's acquisition almost never lands on it (48 of 2,578
frames matched); and `image_dataset.csv`'s own `label` column silently shadowed
the manifest's in the merge.

Confirmed by these runs: imagery covers **9 of 51 nodes**, 2,578 frames.
