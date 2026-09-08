# Results

> **Status: pre-model-3 snapshot, hand-compiled 2026-09-08.** Every number below
> is from the runs of 2026-08-02 (model 1) and 2026-08-10/11 (model 2). Once
> `--stage ladder3` has run, regenerate this file from the run directory and let
> the generated version replace it entirely:
>
> ```bash
> python models/results_doc.py runs/ --out docs/RESULTS.md
> ```
>
> The generator recomputes §1–§4 with paired confidence intervals that this
> hand-written snapshot cannot have. Treat the orderings here as provisional.

## 1. Every model, one table

Temporal protocol, test block. Baselines in italics.

| model | PR-AUC | ECE | Brier | POD | FAR | CSI | ev.det | lead | params |
|---|---|---|---|---|---|---|---|---|---|
| _persistence_ | 0.590 | — | — | 0.769 | 0.231 | — | 0.223 | 6.30 | — |
| _climatology_ | 0.108 | — | — | 0.556 | 0.882 | — | 0.515 | — | — |
| _discharge_pctl_ | 0.816 | — | — | 0.769 | 0.231 | — | 0.223 | — | — |
| _**gbt (lightgbm)**_ | **0.8496** | 0.003 | 0.0076 | 0.634 | 0.080 | 0.601 | 0.161 | 6.30 | — |
| **MODEL 1 — TF-STGNN** | | | | | | | | | |
| M0 GRU only | 0.598 | 0.004 | 0.0137 | 0.540 | 0.441 | 0.379 | 0.262 | 5.66 | — |
| M1 +FiLM | 0.703 | 0.006 | 0.0121 | 0.579 | 0.310 | 0.459 | 0.324 | 4.60 | — |
| M2 +spatial | 0.715 | 0.007 | 0.0113 | 0.621 | 0.315 | 0.483 | **0.408** | 4.44 | — |
| M3 +flow | 0.704 | 0.010 | 0.0135 | 0.621 | 0.342 | 0.469 | 0.380 | 4.67 | — |
| M4 +focal | 0.709 | 0.076 | 0.0229 | 0.634 | 0.336 | 0.480 | 0.330 | 4.97 | — |
| M5 ×5 | 0.742 | 0.009 | 0.0119 | 0.632 | 0.282 | 0.506 | 0.344 | 4.76 | — |
| M6_scalars ×5 | 0.732 | 0.009 | 0.0119 | 0.600 | 0.279 | 0.487 | 0.324 | 4.70 | — |
| M6_cnn ×5 | 0.726 | 0.017 | 0.0154 | 0.664 | 0.338 | 0.496 | 0.366 | 4.88 | 11,800k |
| **MODEL 2 — MMF-Net** | | | | | | | | | |
| N0 linear embeds | 0.6083 | 0.0024 | — | — | 0.412 | — | 0.290 | — | 551k |
| N1 **+PLR embeds** | 0.7605 | 0.0053 | — | — | 0.325 | — | 0.420 | — | 555k |
| N2 +feature attn | 0.7738 | 0.0056 | — | — | 0.313 | — | **0.423** | — | 693k |
| N3 +FiLM | 0.8269 | 0.0032 | — | — | 0.110 | — | 0.197 | — | 711k |
| N4 +focal | 0.7592 | 0.0525 | — | — | 0.138 | — | 0.220 | — | 711k |
| N5 ×5 | 0.7846 | 0.0043 | — | — | 0.145 | — | 0.208 | — | 711k |
| **N5_bce ×5** | **0.8355** | **0.0016** | **0.0080** | — | 0.120 | **0.561** | 0.220 | — | 711k |
| N6_scalars ×5 | 0.7284 | 0.0055 | — | — | 0.344 | — | 0.397 | — | 719k |
| N6_gated ×5 | 0.8310 | 0.0066 | — | — | 0.107 | — | 0.217 | — | 11,967k |
| **MODEL 3 — STG-Former** | | | | | | | | | |
| P0 … P4_onset | _not yet run_ | | | | | | | | 711k / 980k |

Other protocols, **not comparable to the rows above** and never to be quoted
beside them without the caveat: M3 [basin holdout] 0.840, M3 [random split]
0.905. The basin split still trains on 2003–2025 and keeps temporal leakage, so
0.840 measures spatial transfer only.

**Two things this table cannot tell you, by construction:**

1. **The `ev.det` column is not readable down the page.** FAR spans 0.080 to
   0.441, so those rows sit at different operating points. N2 → N3 looks like
   event detection halving (0.423 → 0.197); its FAR also fell 3× and its PR-AUC
   *rose*. Run `models/rethreshold.py runs/ --far 0.231` before believing any
   ordering in that column.
2. **The top model-2 rows are probably not separable.** N5_bce, N6_gated and N3
   are 0.8355 / 0.8310 / 0.8269 — gaps of 0.0045 and 0.0086 — while N5_bce's own
   five seeds spread over 0.048 in best-val episode PR-AUC. Differences below
   roughly 0.02 PR-AUC should not be claimed as orderings.

## 2. Where each research question stands

| RQ | Question | Status | What settles it |
|---|---|---|---|
| **RQ1** | Do directed flow edges earn their place? | **UNRESOLVED, and currently unanswerable** | model 3 `P2 − P1` and `P3 − P0_x5` |
| **RQ2** | Does a random split leak? | **RESOLVED — strongest result** | done; see §3 |
| RQ3 | Ensemble + calibration | PARTIAL — ECE 0.073 → 0.0088, but ev.det stays below M2's | matched-FAR table |
| RQ4 | FiLM terrain conditioning | POSITIVE on model 1 (+0.105); ambiguous on model 2 | N2 → N3 at matched FAR |
| RQ5/RQ8 | Does SAR imagery help? | **NULL, twice** | closed unless `P5_sar` is run |
| RQ6 | Periodic numerical embeddings | **POSITIVE — largest single gain in the project** (+0.152) | single seeds; size not exact value |
| RQ7 | Cross-feature attention | Weak positive (+0.0133) | inside seed noise; do not claim |

### Why RQ1 is not merely unresolved but currently *unanswerable*

The comparison everyone reads as "the graph doesn't help" is N5_bce (0.8355, no
graph) against M5 (0.7421, graph). That comparison varies **three** things at
once:

| | M5 | N5_bce |
|---|---|---|
| encoder | GRU on raw scalars | transformer on PLR embeddings |
| loss | `focal_conf` — measured at −0.0509 PR-AUC | `bce` |
| panel | 2026-08-02 run, **predates** `truncate_after="2024-12-31"` | truncated |

Model 3 exists to remove all three. Its `P3 − P0_x5` contrast holds encoder,
loss, panel, seed count and calibrator fixed and varies only whether messages
pass along the river. That interval is the RQ1 answer, and nothing else in the
project is.

## 3. The findings that carry a paper

Ranked by how much of the contribution they carry. Note that none of the top
three is an architecture result — that matters for how the paper is framed.

**1. Leakage selects a different model, it does not merely inflate the score.**
Under a random split, PR-AUC rises 0.704 → 0.905 (+29% relative) while pre-onset
event detection **collapses 0.380 → 0.089**. The inflated model has learned to
memorise neighbouring days and never warns before onset. PR-AUC inflation from
random splits is already known in the literature; the *ev.det collapse* is the
novel part, and it converts a methodological caution into a measured operational
failure. Lead with this.
_Caveat: n=1 seed per arm. The effect is far too large to be seed noise; the
exact numbers are not defensible to three decimals._

**2. The target is a persistence ceiling, not a contest.** `target_flood_1d` is
tomorrow's discharge above the 98th percentile, and discharge is strongly
autocorrelated — so `discharge_pctl` alone scores 0.816 and the GBT 0.8496.
Model 2 closed 87% of the neural gap and it bought nothing scientific, because
even winning would prove only that a network matched a percentile rule. The
useful reframing: report **onset detection at a capped FAR** as the headline,
where every baseline sits at 0.16–0.22 and every model from M1 up beats them.
This is what `P4_onset` and `--eval-head 3` were added for.

**3. Focal loss is a defect on this task, not a cost.** N5 → N5_bce changes only
the loss and gains +0.0509 PR-AUC with ECE cut to a third. This is a clean,
publishable negative result about a technique that is near-universal in
imbalanced remote-sensing work. It also means model 1's entire published upper
ladder is understated — `M5_bce` is the run that quantifies by how much.

**4. Periodic numerical embeddings beat depth.** N0 → N1 changes only the input
layer and moves PR-AUC +0.152, the largest single-change gain anywhere in the
project — larger than FiLM's +0.105. Supports Gorishniy et al. (NeurIPS 2022) on
a real hydrological panel.

**5. SAR imagery does not help at a 1-day horizon.** Two independent nulls
(M6_cnn; N6_gated at 17× the parameters scoring *below* N5_bce).
_Caveat: this is a coverage result — 9 of 51 nodes, ~12-day revisit against a
1-day horizon — not a claim about SAR for flood mapping in general._

## 4. Decisions, and the rule for each

| Decision | Rule | Current state |
|---|---|---|
| Does model 3 keep the graph? | `P3 − P0_x5` CI excludes 0 | **unknown** — run `--stage ladder3` |
| Is the assembly correct? | `P0 − N3` = 0.0000 exactly | must be checked **before** P1–P3 are read |
| Do flow edges beat spatial alone? | `P2 − P1` CI excludes 0 | unknown |
| Which model is the headline? | best PR-AUC whose CI clears the runner-up | N5_bce provisionally; not separable from N6_gated/N3 |
| Is any ordering claimable? | \|Δ\| > 2 sd of seed spread (≈0.042 for N5_bce) | most current gaps fail this |
| Report SAR at all? | only if `P5_sar` beats P3 | two nulls; default is to drop it |
| Report `ev.det` down a column? | never — use matched FAR | not yet done |

**The next run, in order.** Nothing else is worth GPU time until these land:

```bash
# 1. free, no GPU: makes the ev.det column readable at last
python models/rethreshold.py runs/ --far 0.231

# 2. ~4 h: the RQ1 answer. P0 first — it is an assembly check, and if it
#    misses N3's 0.8269 the rest of the ladder is not worth running.
python models/kaggle_run.py --stage ladder3

# 3. ~40 min: removes the loss confound from every M-vs-N comparison
python models/kaggle_run.py --stage ladder --presets M5_bce --ladder-seeds 5

# 4. ~3 h: the early-warning headline, run as a pair or not at all
python models/kaggle_run.py --stage onset

# 5. regenerate this document with real confidence intervals
python models/results_doc.py runs/ --out docs/RESULTS.md
```

Cheap and worth folding into a later sweep, from the model 2 gradient logs: the
clip binds on 44–54% of steps past epoch 40 and seed 3 never early-stopped
inside 60 epochs, so `--grad-clip 2.0 --epochs 80` is the obvious sweep. Do it
**after** the ladder, not during — model 3's `P0`/`P0_x5` rungs are controls that
reproduce model 2, and they cannot do that under a changed optimiser schedule.

## 5. Paper skeleton this evidence supports

The architecture is the vehicle, not the contribution — a transformer encoder
plus a relational GNN is a combination of known parts and a reviewer will say
so. The contribution is the **evaluation protocol and what it reveals**:

1. **Claim** — on autocorrelated hydrological panels, standard practice
   (random splits, focal loss, PR-AUC on a next-day threshold target) produces
   models that score well and fail operationally.
2. **Evidence** — the ev.det collapse under random splits (§3.1); the focal-loss
   regression (§3.3); the persistence ceiling (§3.2).
3. **Remedy** — block-temporal splits, BCE, and onset-at-capped-FAR as the
   reported metric.
4. **Under that protocol**, a fair architecture comparison becomes possible for
   the first time — three families, one encoder difference and one graph
   difference, each with a paired interval (§2).
5. **Negative results reported, not buried** — SAR, focal loss, and whichever of
   RQ1/RQ7 the intervals fail to resolve.

A paper whose headline is "our GNN scores 0.84" competes with a percentile rule
that scores 0.816 and loses to a LightGBM that scores 0.8496. A paper whose
headline is "here is why flood-prediction benchmarks overstate skill, measured
three ways on a real basin network" does not have that problem.

## 6. Known limitations to state before a reviewer finds them

- **No embargo between splits.** The 14-day lookback window is taken by absolute
  day index, so test days in the first 13 days of 2021 read features from
  December 2020 (the val block). Causal and not label leakage, but state it.
- **The validation block is quieter than train and test** — 0.81% positive vs
  2.03% / 2.28%. Real hydrology, but it makes early stopping and the frozen
  threshold optimistic-biased. This is why `event_pr_auc` is the selection
  signal rather than node-day PR-AUC.
- **Same nodes in train and test.** The temporal protocol tests the same 51
  gauges at later dates. Spatial transfer is a separate protocol (Gin holdout, 3
  nodes) that only model 1's M3 has run.
- **`sar_pretrain` has never completed** — three bugs, all fixed, none re-run —
  so RQ8 is formally untested regardless of the two SAR nulls.
- **Model 1's numbers predate the `truncate_after` policy**, so M-rows and N-rows
  differ in data extent as well as loss. `M5_bce` fixes both at once.
- **`RelationalGATv2` allocates parameters per relation whether or not that
  relation has edges**, so `P1` (spatial only) reports the same 980k as `P2`
  despite ~67k of them never receiving a gradient. Subtract before making an
  efficiency argument.
