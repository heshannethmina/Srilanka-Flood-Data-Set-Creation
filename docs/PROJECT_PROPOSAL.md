# Project Proposal Pack

> **What this file is.** A complete, self-contained source document for generating
> a formal project proposal (research proposal, thesis proposal, or funding
> application). Every number here was **measured** from the artifacts in this
> repository; nothing is estimated unless explicitly labelled "estimate".
>
> **Instructions for the AI generating the proposal:**
> 1. Do **not** invent results. No model has been trained. Write results sections
>    as *planned experiments with success criteria*, never as achieved numbers.
> 2. Use §16 (Fact Sheet) verbatim for all quantities.
> 3. Keep the framing *terrain-aware and hydrology-grounded*, **not**
>    "physics-guided" — there are no conservation constraints in the loss and no
>    assimilated in-situ gauge data. Do not upgrade this claim.
> 4. The known problems in §11 and §12 are **deliberate content**. A proposal that
>    states its own weaknesses and plans for them is stronger than one that hides
>    them. Do not delete them.
> 5. Companion document: `docs/PROJECT_BRIEF.md` holds the deeper dataset
>    documentation. This file is the proposal; that one is the technical annex.

---

## 1. Title & abstract

**Title.** Terrain-Aware Multimodal Spatiotemporal Graph Neural Network for
Probabilistic Flood Early Warning in Sri Lanka.

**Short title.** TF-STGNN: Calibrated Multimodal Flood Forecasting for Sri Lankan
River Basins.

**Abstract (≈200 words).**
Sri Lanka experiences recurrent monsoon-driven riverine flooding that displaces
hundreds of thousands of people, yet regional machine-learning flood forecasting
is undermined by three methodological failures: river-basin topology is ignored
in favour of Euclidean proximity, evaluation uses random splits that leak
spatially and temporally autocorrelated flood days between training and test
sets, and outputs are uncalibrated point predictions unsuitable for issuing
warnings. This project delivers (i) a reproducible, leakage-controlled, 23-year
daily spatiotemporal benchmark spanning 51 hydrologically anchored monitoring
nodes across 16 Sri Lankan river basins, built entirely from public, key-free
reanalysis sources and validated against 11 documented historical flood
episodes; and (ii) a multimodal spatiotemporal graph neural network that fuses a
14-day meteorological and hydrological sequence, static terrain descriptors,
Sentinel-1 SAR imagery, and a directed upstream→downstream river graph into a
**calibrated** 24-hour-ahead flood probability. The model is evaluated under
three mutually exclusive leakage-controlled protocols — temporal block, unseen
spatial basin, and event-level grouping — using PR-AUC, Brier score, Expected
Calibration Error and event-level lead time. The dataset, code and trained
models will be released publicly.

**Keywords.** flood early warning; graph neural networks; spatiotemporal
forecasting; probability calibration; Sentinel-1 SAR; multimodal fusion; data
leakage; Sri Lanka; GloFAS.

---

## 2. Problem statement and motivation

### 2.1 The operational problem

Sri Lanka's flood regime is bimodal and severe. The south-west monsoon
(May–September) drives flooding in the wet-zone basins — Kelani, Kalu, Gin,
Nilwala — where steep, short, fast-responding catchments concentrate runoff into
densely populated valleys within hours. The north-east monsoon (December–
February) floods the dry-zone east and north — Batticaloa, Gal Oya, Malwathu
Oya. Major episodes in 2003, 2011, 2016, 2017, 2021, 2024 and the November 2025
Cyclone Ditwah event each displaced large populations.

Effective early warning requires three things a typical ML flood model does not
provide: **lead time** (a forecast, not a nowcast), **spatial specificity** (which
node, not which country), and **calibrated confidence** (a probability an
official can act on, where "70%" means 70%).

### 2.2 The research problem

Three failures recur in the flood-ML literature and jointly make published
accuracy figures unreliable:

- Spatial models treat monitoring points as an unstructured point cloud or a
  regular grid, discarding the single most informative physical constraint
  available — that water flows downhill along a known channel network.
- Spatiotemporal data are split randomly. Because a flood is a multi-day,
  multi-node correlated event, random splitting places days from the *same*
  flood in both training and test sets. Reported skill is then partly memory.
- Outputs are thresholded point predictions. Without calibration, a model that
  ranks events well can still be systematically overconfident, which in an
  early-warning setting produces either alarm fatigue or missed disasters.

### 2.3 Why Sri Lanka specifically

No ML-ready Sri Lankan flood benchmark exists. Published national-scale work
relies on small collections of documented events (tens of samples). This project
constructs a benchmark two orders of magnitude larger (409,350 supervised
samples) from public data, making rigorous, leakage-controlled evaluation
possible for the first time in this region.

---

## 3. Research gaps and the response to each

| Gap | Statement | Key literature | Response in this project |
|---|---|---|---|
| **G1** | Spatial neural models neglect physical hydrological topology, using Euclidean proximity or grids instead of directed river routing and terrain gradient. | FloodGNN-GRU (Cambridge Env. Data Science, 2024); *Explicit Water Balance Constraints for Trustworthy GNN Flood Forecasting* (MDPI Appl. Sci., 2026) — the "topology paradox". | A 51-node graph with **35 directed upstream→downstream flow edges** kept as a *separate relation* from **204 distance-decayed spatial k-NN edges**, consumed by a relational GNN with per-relation parameters. Flow edges are never symmetrised. Edge attributes carry elevation drop. |
| **G2** | Random cross-validation on autocorrelated spatiotemporal data inflates reported skill. | Roberts et al., *Ecography*; *Choosing blocks for spatial cross-validation* (Frontiers in Remote Sensing). | Random splitting is made structurally impossible: three block protocols ship as dataset columns (temporal, whole-basin holdout, event-level groups). All thresholds, climatologies and percentiles are fitted on the training period only. A dedicated experiment **quantifies** the inflation. |
| **G3** | Deterministic, uncalibrated outputs are unusable for warning. | *Flood Uncertainty Estimation Using Deep Ensembles* (MDPI Water, 2022); *Real-Time Probabilistic Flood Forecasting* (MDPI Water, 2020). | Probability is the primary output. Temperature/isotonic calibration fitted on the validation block only; 5-seed deep ensemble for uncertainty; Brier, ECE and reliability diagrams are mandatory reported metrics; ambiguous labels down-weighted by `label_confidence`. |
| **G4** | Weak fusion of dynamic meteorology with static terrain and remote sensing. | *Intelligent flood forecasting and warning: a survey*. | Three-stream architecture: terrain **modulates** the temporal encoder via FiLM (not concatenation), and a Sentinel-1 SAR CNN branch joins 1:1 to the tabular record through a shared `tabular_key`. |

---

## 4. Research questions and hypotheses

| # | Research question | Hypothesis | How it is tested |
|---|---|---|---|
| **RQ1** | Does encoding directed river topology improve 24-hour flood probability over spatial proximity alone? | **H1** Adding directed flow edges improves test PR-AUC over a spatial-only graph, with the largest gain at downstream nodes that have upstream parents. | Ablation M2 (spatial-only) vs M3 (flow + spatial); per-position breakdown (upstream/mid/outlet). |
| **RQ2** | How much do random splits inflate reported skill on this data? | **H2** A random node-day split inflates PR-AUC materially relative to the temporal block split, for an identical model. | Same architecture, same hyperparameters, two split protocols. Single headline number. |
| **RQ3** | Can a deep model produce calibrated flood probabilities under ~1.9% class imbalance? | **H3** Post-hoc temperature scaling plus a 5-seed ensemble reduces ECE substantially versus the raw network, without degrading PR-AUC. | ECE / Brier / reliability diagrams before and after calibration on the test block. |
| **RQ4** | Does static terrain conditioning outperform feature concatenation? | **H4** FiLM conditioning on elevation and drainage proxy beats concatenation, especially for nodes whose terrain differs most from the training mean. | Ablation: none vs concat vs FiLM. |
| **RQ5** | Does Sentinel-1 SAR add predictive value beyond rainfall and discharge? | **H5 (null-leaning)** At a 12-day revisit, SAR adds *severity and state* information but little *lead time*; scalar SAR features capture most of the gain that a CNN would. | Three-way comparison on the imaged sites: no image · SAR scalars · SAR CNN. **A negative result here is a publishable finding and must be reported honestly.** |

---

## 5. Aim and objectives

**Aim.** To build and rigorously evaluate a multimodal spatiotemporal graph
neural network that issues calibrated 24-hour-ahead flood probabilities for Sri
Lankan river basins, under evaluation protocols that eliminate spatiotemporal
data leakage.

| # | Objective | Measurable success criterion |
|---|---|---|
| **O1** | Publish a reproducible, leakage-controlled Sri Lankan flood benchmark. | Dataset rebuilds end-to-end from scripts with no API key; ≥400,000 valid supervised samples; three split protocols shipped as columns. **Status: achieved** (409,350 samples). |
| **O2** | Validate the discharge-derived labels against documented floods. | ≥8 of 11 documented episodes detected at the primary threshold, with misses characterised rather than hidden. **Status: achieved** (8/11 high, 9/11 moderate). |
| **O3** | Construct and publish an aligned Sentinel-1 SAR image branch. | ≥2,000 frames across ≥9 nodes, 1:1 joinable to the tabular record, with a documented index. **Status: in progress.** |
| **O4** | Implement and train the TF-STGNN. | Model trains to convergence; beats all six baselines on test PR-AUC under the temporal split. |
| **O5** | Deliver calibrated probabilities. | Post-calibration ECE below the uncalibrated baseline; reliability diagram within tolerance across the probability range; Brier decomposition reported. |
| **O6** | Quantify the leakage effect (the G2 contribution). | A single reported figure: PR-AUC under random split minus PR-AUC under temporal split, same model. |
| **O7** | Release code, dataset and trained weights. | Public repository + archived dataset (GitHub Release / Zenodo DOI) with dataset card and data dictionary. |

---

## 6. Novelty and contributions

1. **The first public, ML-ready, leakage-controlled Sri Lankan flood benchmark.**
   51 nodes × 16 basins × daily 2003–2025; 410,931 node-days; built from
   key-free public sources and fully rebuildable.
2. **Labels grounded in operational hydrology, not annotation.** Return-period
   exceedance of GloFAS discharge against per-node thresholds fitted on training
   years only — the same definition GloFAS itself uses operationally.
3. **Leakage control as a shipped dataset artifact**, not a paper claim: three
   precomputed protocols, plus a measured quantification of what random
   splitting would have bought.
4. **Explicit multi-relational hydrological topology** — directed flow edges held
   separate from spatial edges, with elevation-drop edge attributes.
5. **Zero-cost multimodal alignment.** The hydrological label engine
   auto-labels the SAR imagery through a shared key, eliminating manual
   annotation for the vision branch.
6. **Two diagnostic findings that improve the field's practice** (see §11):
   a documented GloFAS grid-snapping failure mode that explains specific missed
   events, and a measured quantification of Sentinel-1's revisit ceiling over
   Sri Lanka that bounds what any SAR-based early-warning claim can promise.

---

## 7. Methodology

### 7.1 Data sources (all public, no API key)

| Modality | Source | Product / resolution |
|---|---|---|
| Precipitation (primary) | Open-Meteo Archive API | ERA5-Land, 0.1° |
| Temperature, humidity, wind, radiation, soil wetness | NASA POWER daily point API | MERRA-2 / GEOS, ~0.5° |
| River discharge | Open-Meteo Flood API | Copernicus **GloFAS** reanalysis, ~0.05° |
| Elevation | Open-Meteo | Copernicus DEM |
| SAR imagery | Microsoft Planetary Computer STAC | Sentinel-1 RTC (γ⁰, IW), 10 m |
| River topology | Expert-built (`scripts/common/nodes.py`) | 51 nodes, `downstream_of` links |

All acquisition is programmatic, cached, resumable and rate-limit aware.

### 7.2 The node network and graph

51 monitoring nodes across 16 basins, each carrying basin, latitude/longitude,
hydrological position (upstream / mid / downstream / outlet), climate zone
(wet / dry / intermediate) and a `downstream_of` parent link.

- **Flow edges (35):** directed parent→child, weight 1.0, haversine distance.
- **Spatial edges (204):** k = 4 nearest neighbours, bidirectional, weight
  `exp(−d / 40 km)`.
- **Edge attributes:** `[weight, distance_km/100, Δelevation/100, is_flow]`.

### 7.3 Features (67 columns; 33 dynamic + 2 static used by the model)

Meteorology; soil wetness and anomalies; engineered antecedent rainfall
(trailing sums over 2/3/5/7/10/15/30 days, 3- and 7-day maxima, an Antecedent
Precipitation Index with k = 0.90, wet-day counts); discharge dynamics (rise,
trailing means, day-of-year climatology, anomaly, z-score, empirical
percentile); static terrain (elevation, drainage-area proxy).

**Leakage control:** every climatology, threshold and empirical percentile is
fitted on `date ≤ 2017-12-31` only.

### 7.4 Labels and targets

Flood state at node *v*, day *t* is `Q_v(t) ≥ q_v^(0.98)`, where the threshold is
the 98th percentile of that node's **training-period** discharge distribution
(~2-year return period). Moderate (90th) and severe (99.5th) variants are also
provided. `label_confidence` down-weights days within ±15% of the threshold.

Targets are strictly causal and computed within node boundaries:
`target_flood_h = max(flood_state[t+1 … t+h])` for h ∈ {1, 2, 3}; plus onset
targets, next-day discharge, and 3-day maximum discharge z-score.

**Primary target: `target_flood_1d`** — 24-hour-ahead flood probability.

### 7.5 Evaluation protocols (choose one per experiment; never mix)

| Protocol | Definition | Valid samples |
|---|---|---|
| Temporal | train ≤ 2017 · val 2018–2020 · test 2021+ | 277,899 / 55,896 / 75,555 |
| Spatial | whole **Gin** basin held out | 377,330 train / 32,020 test |
| Event | GroupKFold over 1,469 flood episodes | — |

### 7.6 Sentinel-1 SAR branch — 9 sites, three basins

Three basins, each sampled as an **upstream → mid → outlet transect**, so the
imagery visualises a flood wave propagating downstream — the same structure the
directed flow edges encode.

| Basin | Nodes | Places |
|---|---|---|
| Kelani | KEL_HAN, KEL_KAD, KEL_COL | Hanwella, Kaduwela, Kelani mouth (Colombo) |
| Kalu | KAL_RAT, KAL_BUL, KAL_KLT | Ratnapura, Bulathsinhala, Kalutara mouth |
| Nilwala | NIL_PIT, NIL_AKU, NIL_MAT | Pitabeddara, Akuressa, Matara mouth |

**Specification.** Every Sentinel-1 IW pass 2015–2025 at a fixed 5 km × 5 km
footprint per site, resampled to 512 × 512 px (~10 m/px). VV and VH converted to
dB, clipped to [−30, +5]. Water rule: VV < −17 dB. Per-frame scalars:
`water_fraction`, `vv_mean`, `vh_mean`, `valid_fraction`.

**Fixed footprint is essential:** a scattered-chip design gives the CNN a
different scene every sample, so it cannot learn *change*. Pinning the location
turns each site into a dry → flood → recede sequence.

**Build environment.** The fetch is network-bound (~2,500 frames × 2 remote COG
reads). It runs on **Kaggle** via `scripts/kaggle/kaggle_build_image_dataset.py`,
which is self-contained — it re-derives the labels from the same public APIs, so
no local upload is required, and stores output as 8-bit PNG (quantisation step
0.137 dB, an order of magnitude below SAR speckle, so effectively lossless).

### 7.7 Model — TF-STGNN

**Inputs per day:** `X[B, 51, 14, 33]` dynamic; `S[51, 9]` static (elevation,
log drainage proxy, zone one-hot, position one-hot); `I[51, 2, 512, 512]` SAR
with a presence flag and age-in-days; `edge_index[2, 239]` with 4-dim edge
attributes; a `valid_sample` mask.

**Normalisation:** z-scored using training-period statistics only; `log1p`
applied first to heavy-tailed columns (precipitation accumulations, discharge).

| Stream | Design | Output |
|---|---|---|
| 1. Temporal | 2-layer GRU (H = 128, dropout 0.2) over the 14-day window, then learned temporal-attention pooling. Attention weights double as an interpretability figure showing the rainfall→discharge lag per basin. | `[B, 51, 128]` |
| 2. Static | MLP 9 → 64 → 256 producing (γ, β); FiLM modulation `h ← γ ⊙ h + β`, then LayerNorm. Terrain *rescales sensitivity to rainfall* rather than being diluted among 33 dynamic channels. | modulated `h` |
| 3. Vision | ResNet-18 with a 2-channel stem (no pretrained weights) → 64-d, masked to zero for nodes without imagery. Cheaper alternative to ablate first: append the four scalar SAR features as extra dynamic channels. | `[51, 66]` |
| Fusion | concat → Linear 194 → 128 → GELU → LayerNorm | `z_in` |
| 4. Graph | 2 × edge-type-conditioned GATv2 (4 heads × 32), **separate parameters for flow and spatial relations**, residual + LayerNorm. Two layers matches the 5-hop longest chain without oversmoothing a 51-node graph. | `z` |
| Head | concat(`z_in`, `z`) → 256 → 128 → six heads: t+1 (primary), t+2, t+3, onset, next-day discharge, 3-day max z-score. The skip path lets an out-of-distribution basin bypass the graph — important for the Gin holdout. | probabilities |

**Approximate size (estimate):** ~0.3 M parameters without the vision branch,
~11.5 M with it — trainable on a single GPU in minutes per seed, which supports
the operational-deployability argument.

**Loss.**
`L = Σ_h w_h · mean(mask · label_confidence · Focal(ŷ_h, y_h; α≈0.75, γ=2)) + 0.2 · Huber(regression heads)`
with `w = {1d: 1.0, 2d: 0.3, 3d: 0.3, onset: 0.5}`.

**Training.** AdamW, lr 1e-3, weight decay 1e-4, cosine schedule with 5-epoch
warmup, batch = 32 full-graph day snapshots, gradient clip 1.0, ~60 epochs,
early stopping on **validation event-level PR-AUC** (patience 10).

**Calibration.** Temperature scaling (or isotonic) fitted on the validation block
only, then frozen. Uncertainty from a 5-seed deep ensemble; MC dropout as a
secondary estimate.

### 7.8 Baselines (identical splits and `valid_sample` filter)

1. Persistence — `p̂ = flood_state(t)`
2. Per-node day-of-year climatology
3. **Discharge-percentile rule** tuned on validation — effectively "what GloFAS
   already tells you"; this is the real bar to clear
4. **Gradient-boosted trees** (XGBoost/LightGBM) on flattened features — often
   dominant on tabular hydrology; must not be omitted
5. Per-node LSTM/GRU, no graph — isolates the value of G1
6. GNN with spatial edges only — the direct G1 ablation
7. Full TF-STGNN

### 7.9 Ablation grid

| Axis | Variants |
|---|---|
| Graph | none · spatial-only · flow-only · flow + spatial |
| Static fusion | none · concat · FiLM |
| Lookback | 7 · 14 · 30 days |
| Loss | BCE · weighted BCE · focal · focal × `label_confidence` |
| Calibration | raw · temperature · isotonic · ensemble |
| Imagery (imaged sites) | none · SAR scalars · SAR CNN |
| Split protocol | random (diagnostic only) · temporal · basin · event |

### 7.10 Metrics

- **Ranking:** PR-AUC (headline), ROC-AUC.
- **Calibration:** Brier score with reliability/resolution/uncertainty
  decomposition, ECE (15 bins), reliability diagrams.
- **Operational:** POD, FAR, CSI, F1 at validation-selected thresholds;
  precision–recall at fixed false-alarm budgets (e.g. FAR ≤ 20%).
- **Event-level:** proportion of the 1,469 episodes detected with ≥ 1 day lead
  time, and mean lead time on detected events. **This is the headline
  operational metric** and is not equivalent to node-day recall.
- Plain accuracy is never reported — predicting "no flood" always scores 98.1%.

---

## 8. Work plan

Model build order is deliberately incremental so that a working artifact exists
at every stage and each research question maps to one step.

| WP | Work package | Weeks | Output |
|---|---|---|---|
| **WP1** | Dataset consolidation. Reconcile documentation with the current build; decide the 2025 ragged-edge policy; freeze v1.0. | 1 | Frozen `flood_dataset.parquet`, updated dataset card |
| **WP2** | **Coordinate re-snapping study** (see §11.1). Probe candidate offsets per node against the flood API; report before/after; decide whether to re-snap and rebuild. | 1–2 | Re-snapping report; possibly dataset v1.1 |
| **WP3** | SAR image branch build on Kaggle: 9 sites, ~2,500 frames, published as a Kaggle Dataset. | 2 | `image_dataset.csv` + frames |
| **WP4** | Baselines 1–4 under all three protocols. | 2 | Baseline results table |
| **WP5** | TF-STGNN incremental build M0 → M6 (see below). | 4 | Trained models, ablation table |
| **WP6** | Calibration and uncertainty: ensembles, temperature/isotonic, reliability analysis. | 2 | Calibration results (RQ3) |
| **WP7** | Leakage quantification experiment (RQ2), figures, write-up, public release. | 2 | Paper/thesis draft, Zenodo release |

**Total: ~14–16 weeks.**

**Incremental model build (WP5):**

| Step | Configuration | Answers |
|---|---|---|
| M0 | GRU only, no graph, no static, BCE | establishes the floor |
| M1 | + FiLM static conditioning | RQ4 |
| M2 | + spatial edges | — |
| M3 | + directed flow edges (relational) | **RQ1** |
| M4 | + focal loss × `label_confidence` | class imbalance |
| M5 | + deep ensemble & temperature scaling | **RQ3** |
| M6 | + SAR scalars, then SAR CNN | **RQ5** |

---

## 9. Deliverables

1. **Dataset v1.0** — `flood_dataset.parquet` (410,931 node-days × 67 columns),
   graph files, events, thresholds, dataset card, data dictionary.
2. **SAR image dataset** — ~2,500 fixed-footprint frames over 9 nodes with a
   joinable index, published as a Kaggle Dataset.
3. **Reproducible pipeline** — the `scripts/` tree, runnable end-to-end with no
   API key, plus the Kaggle build notebook.
4. **Trained models** — TF-STGNN weights for M0–M6 and all baselines.
5. **Results package** — baseline table, ablation grid, calibration analysis,
   leakage-quantification figure, event-level lead-time analysis.
6. **Manuscript / thesis chapters** and a public release with DOI.

---

## 10. Resources required

| Resource | Detail | Cost |
|---|---|---|
| Data | ERA5-Land, NASA POWER, GloFAS, Copernicus DEM, Sentinel-1 (Planetary Computer) | **Free, no API key** |
| Image build compute | Kaggle notebook, internet enabled, 12 h sessions, resumable | Free tier |
| Training compute | Single GPU; model is small (≈0.3–11.5 M parameters) | Kaggle / Colab free tier sufficient |
| Storage | ~40 MB tabular, ~1.4 GB imagery at 512 px (~350 MB at 256 px) | Negligible |
| Software | Python, pandas, PyArrow, PyTorch, PyTorch Geometric, rasterio, pystac-client, planetary-computer | Open source |

**The entire project is executable at zero monetary cost.** This is itself a
contribution: it makes the work reproducible in a low-resource research setting,
which matters for the region the project serves.

---

## 11. Findings that shape the methodology

These emerged from diagnostic work already completed and are **deliberate
proposal content**.

### 11.1 GloFAS grid snapping places several nodes on tributaries, not main stems

Open-Meteo snaps each coordinate to the nearest GloFAS river cell. For several
nodes that cell is a minor tributary. Measured mean discharge:

| Node | Place | Mean Q (m³/s) | Assessment |
|---|---|---|---|
| KAL_KLT | Kalutara (Kalu mouth) | 208.6 | correct main stem |
| KEL_KAD | Kaduwela (Kelani) | 163.5 | correct main stem |
| KAL_RAT | Ratnapura (Kalu) | 46.9 | plausible |
| KEL_COL | **Kelani mouth, Colombo** | **2.2** | mis-snapped |
| NIL_PIT / NIL_AKU / NIL_MAT | **entire Nilwala chain** | **1.8 / 1.1 / 0.3** | mis-snapped |

**Consequence.** Labels remain *internally* valid — `flood_state` is a per-node
percentile, i.e. "unusually high flow for this cell", and every node retains its
~1.9% positive rate. But at affected nodes the label tracks a tributary rather
than the main river. **This is the most probable explanation for the three
missed south-western events (2003, 2014, 2017) in the validation report** — a
more specific and defensible account than the generic "GloFAS under-represents
flashy catchments".

**Planned action (WP2).** Probe candidate coordinate offsets per node against
the flood API, select the cell with the largest mean discharge consistent with
the node's hydrological position, report before/after, and decide on a rebuild.
Reporting this diagnostic is a contribution in its own right: any study using
Open-Meteo/GloFAS point queries is exposed to the same failure mode.

### 11.2 Sentinel-1 revisit over Sri Lanka is capped at ~12 days

Measured at Hanwella, 2015–2025:

| Collection | Unique acquisition dates | Descending | Ascending |
|---|---|---|---|
| `sentinel-1-rtc` | **263** | 259 | **4** |
| `sentinel-1-grd` | 255 | 270 | 4 |

Sri Lanka is served by essentially **one descending track**, so the orbit's
12-day exact-repeat cycle *is* the sampling rate; only 4 ascending passes
occurred in 11 years. Switching collections gains nothing. Daily or 2-day SAR is
unobtainable from Sentinel-1 at this latitude.

**Consequence for the design.** Lead time must come from the daily
hydro-meteorological branch; imagery contributes *observed inundation state*,
*river morphology* and *severity*, not forecast frequency. The architecture
therefore feeds the image embedding together with an **age-in-days** feature and
a presence mask. Any claim that SAR provides the early warning would be false,
and the proposal must not make it. (Copernicus EMS rapid mapping operates under
the same constraint.)

### 11.3 Flood-positive SAR frames are scarce, which dictates the vision target

Joining existing frames to labels at Hanwella: **6 of 263 frames** fall on a
flood day. The arithmetic is unavoidable — 78 flood days occurred in the period,
but a 12-day revisit samples only ~6.6% of days, while floods last 3–5 days.

Scaled to 9 sites: ~2,500 frames but only **~50–55 flood-positive** ones.

**Consequence.** Binary flood classification on imagery alone is not viable.
Planned mitigations, in priority order:
1. **`discharge_pctl` regression** — every frame becomes a usable sample.
2. **`water_fraction` change detection** against a per-site dry baseline —
   requires no labels at all.
3. **`flood_moderate`** instead of `flood_high` — roughly 4× the positives.
4. Reserve `flood_state` for a qualitative case study (the November 2025 Ditwah
   pre-event frame is the strongest single example).
Site selection across **three basins** rather than one river also maximises the
number of *independent* flood events among the positives.

### 11.4 The validation block is hydrologically quiet

Positive rate for `target_flood_1d`: train 2.03%, **validation 0.81%**, test
2.28%. Real hydrology, but it biases model selection. **Mitigation:** select on
event-level cross-validation rather than the validation block alone, and report
this asymmetry explicitly.

### 11.5 The 2025 record is ragged

Only 3 of 51 nodes have 2025 data (extended to capture Cyclone Ditwah).
**Decision required in WP1:** truncate all model experiments at 2024-12-31 and
retain 2025 for the image branch only (**recommended**), or mask the ragged edge
explicitly.

---

## 12. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Proxy labels (discharge exceedance) diverge from real inundation, especially urban drainage failure in Colombo. | High | Medium | State as a scope boundary; cross-check with SAR `water_fraction`; roadmap item for change-detection labels. |
| R2 | GloFAS mis-snapping degrades labels at some nodes (§11.1). | **Confirmed** | Medium | WP2 re-snapping study; report both versions. |
| R3 | Gradient-boosted trees match or beat the GNN. | Medium | Medium | Report honestly; the contribution is then the benchmark, the leakage quantification and the calibration, which stand regardless. |
| R4 | SAR branch adds no measurable value (§11.3). | **Medium-high** | Low | Pre-registered as H5 with a null-leaning hypothesis; a negative result is a legitimate finding. |
| R5 | Extreme class imbalance destabilises training. | Medium | Medium | Focal loss, class weighting, `label_confidence` weighting, PR-based selection. |
| R6 | Kaggle session limits interrupt the image build. | Medium | Low | Builder is resumable via `RESUME_FROM`; 12 h commits. |
| R7 | Public APIs change or rate-limit. | Low | High | All raw responses cached to parquet; the cache is the reproducibility guarantee. |
| R8 | Gin-basin holdout is too small or unrepresentative. | Low | Medium | 32,020 valid samples across 4 nodes; report alongside event-level CV rather than alone. |
| R9 | Overfitting to the quiet validation block (§11.4). | Medium | Medium | Event-level CV for model selection. |

---

## 13. Limitations (state plainly in the proposal)

1. **Reanalysis, not gauges.** ERA5-Land and GloFAS are model products; skill
   degrades for small flashy catchments and reservoir-regulated rivers — and
   several nodes in this network (Victoria, Udawalawe, Inginiyagala) are
   dam-controlled.
2. **Proxy labels.** Discharge exceedance ≠ mapped inundation extent.
3. **Coarse meteorology.** NASA POWER at ~0.5° means adjacent upland/lowland
   nodes can share a grid cell; ERA5-Land at 0.1° partially mitigates this.
4. **Expert-approximate node placement**, snapped to reanalysis grid cells, not
   surveyed gauge locations.
5. **~1.9% positive rate** demands PR-based evaluation throughout.
6. **Imagery is spatially thin** — 9 of 51 nodes — and supports a case study,
   not a network-wide claim.
7. **No hydraulic or conservation constraints** in the loss: the work is
   *terrain-aware*, not *physics-guided*.
8. **Ragged 2025 coverage** (§11.5).

---

## 14. Expected outcomes and impact

**Scientific.** A benchmark that makes leakage-free flood-ML evaluation possible
for Sri Lanka; a quantified measure of how much random splitting inflates
reported skill; evidence on whether directed river topology helps; a documented
GloFAS snapping failure mode relevant to anyone using point-queried reanalysis
discharge.

**Practical.** A calibrated, node-level 24-hour flood probability, small enough
to run on commodity hardware, built entirely from free data — deployable by a
national agency without licensing or infrastructure barriers.

**Open science.** Dataset, code and weights released publicly, with a
reproducible pipeline requiring no API key and no paid compute.

---

## 15. Future work (roadmap to v2)

1. Complete the SAR branch and add change-detection inundation masks as
   independent observed labels.
2. Cross-validate labels against World Bank / UNU / UNESCO-IHP-WINS flood
   polygons.
3. Ingest Sri Lanka **Irrigation Department gauge and discharge records** — the
   single upgrade that would justify the term *physics-guided* and enable
   water-balance constraints in the loss.
4. Sub-daily (3–6 h) targets via hourly IMERG/GPM, turning a daily model into an
   operational nowcast.
5. Learned routing lags on flow edges (travel time upstream→downstream).
6. Optional daily 2-D rainfall-raster modality to complement the 12-day SAR.

---

## 16. Fact sheet — measured values (use these verbatim; invent nothing)

**Dataset**
- 410,931 node-days × 67 columns · 409,350 valid supervised samples
- 51 nodes · 16 basins · 2003-01-01 → 2025-12-31 · 8,036 days/node (2003–2024)
- 2025 present for 3 nodes only (KEL_HAN, KEL_KAD, KEL_COL)
- Graph: 239 edges = 35 flow + 204 spatial (k = 4, exp(−d/40 km))
- Flood events: 1,469 (mean 5.39 days, median 3, max 45)
- Per-node 98th-percentile thresholds: 1.4 → 632.1 m³/s (median 15.8)

**Class balance (valid samples)**
- `flood_state` 1.907% · `target_flood_1d` 1.907% · `target_flood_2d` 2.295% ·
  `target_flood_3d` 2.654% · `target_onset_1d` 0.388%

**Splits (valid samples)**
- Temporal: train 277,899 (2.028% pos) · val 55,896 (0.810%) · test 75,555 (2.275%)
- Basin holdout (Gin): 377,330 train / 32,020 test

**Label validation:** 8/11 documented episodes detected at the 98th-percentile
threshold, 9/11 at the 90th. Misses: May 2003, Jun 2014 (moderate only), May 2017 —
all pre-2011 south-western flash floods.

**SAR branch**
- 9 nodes across Kelani / Kalu / Nilwala, upstream→outlet transects
- ~260 frames per site → ~2,340–2,500 total, 2015–2025
- Frame: 512 × 512 px, 5 km footprint, VV + VH in dB clipped to [−30, +5]
- Storage: ~1.04 MB/frame as float16 `.npy`; ~0.58 MB as PNG; ~1.4 GB total
- Revisit: median gap 12 days; 263 unique dates at Hanwella over 11 years;
  259 descending vs 4 ascending
- Expected flood-positive frames: ~50–55 of ~2,500 (~2%)

**Node discharge (mean, m³/s):** KAL_KLT 208.6 · KEL_KAD 163.5 · KEL_RUW 63.9 ·
KAL_RAT 46.9 · KEL_AVI 35.2 · KEL_KIT 20.7 · KAL_BAL 10.9 · KEL_HAN 8.0 ·
KEL_UP 6.4 · KAL_BUL 2.3 · KEL_COL 2.2 · KAL_AGA 1.8 · NIL_PIT 1.8 ·
NIL_AKU 1.1 · NIL_URA 0.9 · NIL_MAT 0.3

---

## 17. References

**G1 — hydrological topology**
- *FloodGNN-GRU: a spatio-temporal graph neural network for flood prediction*, Cambridge Environmental Data Science, 2024.
- *Explicit Water Balance Constraints for Trustworthy Graph Neural Network Flood Forecasting*, MDPI Applied Sciences, 2026.

**G2 — leakage and cross-validation**
- Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure*, Ecography.
- *Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study*, Frontiers in Remote Sensing.

**G3 — uncertainty and calibration**
- *Flood Uncertainty Estimation Using Deep Ensembles*, MDPI Water, 2022.
- *Real-Time Probabilistic Flood Forecasting Using Multiple Machine Learning Methods*, MDPI Water, 2020.

**G4 — multimodal fusion**
- *Intelligent flood forecasting and warning: a survey*, OAE Intelligent Robotics.

**Regional**
- *Geospatial assessment of a severe flood event in the Nilwala River basin, Sri Lanka*, Sustainable Water Resources Management (Springer), 2024.

**Data products requiring citation**
- Copernicus Emergency Management Service — GloFAS global river discharge reanalysis.
- ECMWF ERA5-Land reanalysis (via Open-Meteo Archive API).
- NASA POWER daily point data (MERRA-2 / GEOS).
- Copernicus Sentinel-1 RTC (γ⁰) via Microsoft Planetary Computer.
- Copernicus DEM.

---

## 18. Suggested proposal structure for the generating AI

1. Title page, abstract, keywords (§1)
2. Introduction and motivation (§2) — include a Sri Lanka basin map
3. Literature review and research gaps (§3)
4. Research questions, hypotheses, objectives (§4, §5)
5. Novelty and contributions (§6)
6. Methodology (§7) — include the pipeline flow diagram from `scripts/README.md`
   and the architecture diagram from §7.7
7. Preliminary work and diagnostic findings (§11) — this section is what
   distinguishes this proposal from a speculative one: the dataset already
   exists and has already been interrogated
8. Work plan, timeline, deliverables (§8, §9)
9. Resources (§10)
10. Risks and mitigations (§12)
11. Limitations (§13)
12. Expected outcomes and impact (§14)
13. Future work (§15)
14. References (§17)

**Figures to generate:** basin/node map with directed flow edges; per-node
threshold distribution; class-balance and split-size bars; a Kelani discharge
series with threshold and event shading; a SAR dry→flood→recede contact sheet;
the architecture block diagram; a reliability-diagram template.
