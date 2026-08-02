# TF-STGNN — baseline architecture

Implementation of the model specified in [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)
§7.7, plus the six baselines of §7.8 and the incremental M0 → M6 ladder of §8.

Code lives in [model/](../model/) — deliberately separate from
[scripts/](../scripts/), which creates the datasets. The two have no shared entry
point, so a data rebuild and a model rebuild are always separate acts.

This document is the **design rationale**. The **runbook** is
[model/README.md](../model/README.md).

## Where it runs

Kaggle, via [model/kaggle_run.py](../model/kaggle_run.py) — the only entry point,
and it exits immediately anywhere else. Both published datasets are attached to
the notebook as inputs; nothing is downloaded and nothing is uploaded.

```python
!git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
!python /kaggle/working/repo/model/kaggle_run.py --stage ladder
```

Kaggle's default image already carries `torch`, `pandas`, `numpy`, `pyarrow`,
`lightgbm` and `pillow`. There is no PyTorch Geometric requirement: the graph is
51 nodes and 239 edges, so the relational GATv2 is written directly against torch
scatter ops in [modules.py](../model/tfstgnn/modules.py).

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

| File | Contents |
|---|---|
| [kaggle_run.py](../model/kaggle_run.py) | the Kaggle entry point — stages, environment checks, summary table |
| [config.py](../model/tfstgnn/config.py) | feature lists, `ModelConfig`, `TrainConfig`, the M0–M6 presets |
| [data.py](../model/tfstgnn/data.py) | long panel → dense `[T, N, F]`, train-only normalisation, split masks, `SnapshotBatcher` |
| [graph.py](../model/tfstgnn/graph.py) | relational `edge_index` + 4-dim edge attributes |
| [modules.py](../model/tfstgnn/modules.py) | GRU + attention pooling, FiLM, ResNet-18 SAR stem, relational GATv2 |
| [model.py](../model/tfstgnn/model.py) | the assembled network and its six heads |
| [losses.py](../model/tfstgnn/losses.py) | focal × `label_confidence` + Huber, multi-head weighting |
| [metrics.py](../model/tfstgnn/metrics.py) | PR-AUC, ROC-AUC, Brier decomposition, ECE, POD/FAR/CSI, event lead time |
| [calibrate.py](../model/tfstgnn/calibrate.py) | temperature scaling, isotonic (PAVA, no sklearn) |
| [train.py](../model/tfstgnn/train.py) | training loop, ensembling, calibration |
| [baselines.py](../model/tfstgnn/baselines.py) | persistence, climatology, discharge-percentile rule, GBT |
| [sar.py](../model/tfstgnn/sar.py) | SAR index join, causal frame map, LRU frame store |

## Status

The code is validated end to end — every preset, both vision modes, all three
protocols and all four baselines have been run against a synthetic panel with
the real schema. **No model has been trained on the real data yet**, so this
document contains no results and none should be quoted from it.
