# TF-STGNN — baseline architecture

Implementation of the model specified in [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)
§7.7, plus the six baselines of §7.8 and the incremental M0 → M6 ladder of §8.

This document describes **model 1**. A second family, **model 2 (MMF-Net)**, is a
graph-free multimodal transformer built to answer what model 1's results left
open — see the module map at the end and
[models/README.md](../models/README.md) for the comparison.

Code lives in [models/](../models/) — deliberately separate from
[scripts/](../scripts/), which creates the datasets. The two have no shared entry
point, so a data rebuild and a model rebuild are always separate acts.

This document is the **design rationale**. The **runbook** is
[models/README.md](../models/README.md).

## Where it runs

Kaggle, via [models/kaggle_run.py](../models/kaggle_run.py) — the only entry point,
and it exits immediately anywhere else. Both published datasets are attached to
the notebook as inputs; nothing is downloaded and nothing is uploaded.

```python
!git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
!python /kaggle/working/repo/models/kaggle_run.py --stage ladder
```

Kaggle's default image already carries `torch`, `pandas`, `numpy`, `pyarrow`,
`lightgbm` and `pillow`. There is no PyTorch Geometric requirement: the graph is
51 nodes and 239 edges, so the relational GATv2 is written directly against torch
scatter ops in [modules.py](../models/floodlib/modules.py).

Each run writes `runs/<preset>_<protocol>.json` (all metrics, both configs) and
`..._preds.npz` (test probabilities, labels, day/node/event indices) so figures
can be regenerated without retraining.

## The ladder

| Preset | Change from the previous step | Answers |
|---|---|---|
| `M0` | GRU only — no graph, no terrain, BCE | the floor |
| `M1` | + FiLM terrain conditioning | RQ4 |
| `M2` | + spatial edges | — |
| `M3` | + directed flow edges as a separate relation | **RQ1** |
| `M4` | + focal loss × `label_confidence` | class imbalance |
| `M5` | + 5-seed ensemble, temperature scaling | **RQ3** |
| `M6_scalars` | + the four per-frame SAR scalars | **RQ5** |
| `M6_cnn` | + ResNet-18 SAR branch | **RQ5** |

Each step changes exactly one thing, so every ablation row in §7.9 is a diff
against its predecessor rather than a separately tuned model.

## Design decisions worth knowing

**Self-loops are a third relation.** Flow edges alone leave every headwater node
with no incoming edge, so a flow-only ablation would silently zero those nodes.
Relation 2 is a self-loop with `[1, 0, 0, 0]` attributes.

**Attention is normalised across relations jointly.** A node's softmax runs over
all its incoming edges regardless of type, so the model *learns* how much to
weight its upstream parent against its spatial neighbours instead of having that
ratio fixed by the architecture. The parameters producing the messages stay
per-relation, which is what §7.2 requires.

**FiLM starts at identity.** γ is parameterised as `1 + Δγ` with the final layer
zero-initialised, so at step 0 M1 is exactly M0. Any gain is attributable to the
conditioning rather than to a different initialisation.

**Early stopping uses event-level PR-AUC.** The validation block is
hydrologically quiet (0.81 % positive vs 2.28 % in test, §11.4), which makes
node-day PR-AUC a noisy selection signal. `metrics.event_pr_auc` scores one
positive per episode (max probability in the 7 days before onset) against one
negative per flood-free node-week.

**SAR frames are attached forward in time, never backward.** A frame acquired on
day *d* becomes visible on day *d* and stays visible — with a growing
`age_in_days` — until the next pass or 60 days, whichever comes first. Attaching
it to earlier days would leak an observation that had not yet been made.

**The CNN batch is packed, not dense.** Only 9 of 51 nodes have imagery. Frames
are gathered into a `[K, 2, P, P]` tensor and scattered back by flat index, so
the CNN runs 5.7× fewer forward passes and a 512 px batch costs ~150 MB instead
of 3.4 GB.

**Calibration and thresholds are fitted on validation, then frozen.** Fitting
either on test would reintroduce exactly the leakage this project exists to
eliminate.

**Plain accuracy is not implemented anywhere.** At a 1.9 % positive rate, "never
floods" scores 98.1 %.

## Module map

Shared — `floodlib/` owns the data and the evaluation, so a difference between
the two families is a difference of architecture and never of protocol:

| File | Contents |
|---|---|
| [kaggle_run.py](../models/kaggle_run.py) | the Kaggle entry point — stages, environment checks, summary table |
| [report.py](../models/report.py) | offline results table from a downloaded `runs/` |
| [floodlib/schema.py](../models/floodlib/schema.py) | the column contract: feature lists, targets, `BaseModelConfig` |
| [floodlib/traincfg.py](../models/floodlib/traincfg.py) | `TrainConfig`, `Preset` |
| [floodlib/data.py](../models/floodlib/data.py) | long panel → dense `[T, N, F]`, train-only normalisation, split masks, `SnapshotBatcher` |
| [floodlib/graph.py](../models/floodlib/graph.py) | relational `edge_index` + 4-dim edge attributes |
| [floodlib/blocks.py](../models/floodlib/blocks.py) | layers both families use: GRU + attention pooling, FiLM, ResNet-18 SAR stem |
| [floodlib/engine.py](../models/floodlib/engine.py) | the training loop both families run through — ensembling, calibration, thresholding |
| [floodlib/losses.py](../models/floodlib/losses.py) | focal × `label_confidence` + Huber, multi-head weighting |
| [floodlib/metrics.py](../models/floodlib/metrics.py) | PR-AUC, ROC-AUC, Brier decomposition, ECE, POD/FAR/CSI, event lead time |
| [floodlib/calibrate.py](../models/floodlib/calibrate.py) | temperature scaling, isotonic (PAVA, no sklearn) |
| [floodlib/baselines.py](../models/floodlib/baselines.py) | persistence, climatology, discharge-percentile rule, GBT |
| [floodlib/sar.py](../models/floodlib/sar.py) | SAR index join, causal frame map, LRU frame store |

Model 1 — the relational graph network described above:

| File | Contents |
|---|---|
| [model1/config.py](../models/model1/config.py) | `ModelConfig`, the M0–M6 presets |
| [model1/modules.py](../models/model1/modules.py) | the relational GATv2 |
| [model1/model.py](../models/model1/model.py) | the assembled network and its six heads |
| [model1/train.py](../models/model1/train.py) | preset table + model factory |

Model 2 — a graph-free multimodal transformer, built to test whether the tree
baseline can be beaten by a better *input layer* and whether the SAR branch was
useless or merely fused badly:

| File | Contents |
|---|---|
| [model2/config.py](../models/model2/config.py) | `MMFConfig`, the N0–N6 presets |
| [model2/modules.py](../models/model2/modules.py) | periodic numerical embeddings, temporal + cross-feature transformers, gated SAR fusion |
| [model2/model.py](../models/model2/model.py) | the assembled network and its six heads |
| [model2/train.py](../models/model2/train.py) | preset table + model factory |
| [model2/pretrain_sar.py](../models/model2/pretrain_sar.py) | chip-level SAR encoder pretraining on `image_manifest.csv` |

## Status

**Model 1** has been run end to end on the real data (2026-08-02, Kaggle T4,
5.07 h, all five stages). Results and their caveats are in
[models/README.md](../models/README.md); the two open items are the single-seed
M2 confound and RQ1.

**Model 2** has been run on the real data (2026-08-10/11, Kaggle T4, ~11 h across
`ladder2`, `sar2` and `N5_bce`). Its headline `N5_bce` reaches PR-AUC **0.8355**
against model 1's best of 0.7421 and the gradient-boosted-tree baseline's 0.8496
— and it does so **without a graph**, which is the comparison model 1 could not
make because its only per-node reference was a plain GRU.

Two results carry back to model 1. The focal loss is a **defect rather than a
cost**: swapping it for BCE, changing nothing else, gains +0.0509 PR-AUC and cuts
ECE to a third, and M4/M5/M6 all sit on that same rung. And the SAR branch adds
nothing once the loss is fixed — a 711k-parameter model with no imagery beats an
11,967k-parameter one with it.

Numbers, caveats and the per-seed spread are in
[models/README.md](../models/README.md). `sar_pretrain` has not yet completed, so
RQ8 (transferring a pretrained SAR encoder) remains formally untested.
