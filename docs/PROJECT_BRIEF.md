# Project Brief — Complete Handoff Pack

> **Purpose of this file.** Everything an external writer (human or AI) needs to
> produce a full technical report / thesis chapter / paper on this project,
> without inventing anything. Every number below was measured from the actual
> artifacts in this repository on **2026-08-01**. If a number is not in this
> file, it has not been measured — do not invent it.

---

## 0. Ground rules for whoever writes the report

1. **Do not invent model results.** No model has been trained yet. This project
   is currently a *dataset + benchmark construction* effort with a specified
   model design. Any "results" section must be written as *planned
   experiments*, not achieved numbers.
2. **Use the measured numbers in §3 and §7 verbatim.** Older documents in this
   repo (`README.md`, `docs/DATASET_CARD.md`) quote a superseded build
   (409,836 rows, 2003–2024). The current build is **410,931 rows** and extends
   to 2025 for three nodes. See §3.6 for the exact discrepancy list.
3. **Honest framing:** the work is *terrain-aware and hydrology-grounded*, not
   *physics-guided* in the strict sense (no assimilated in-situ gauge data, no
   conservation constraints in the loss). Do not upgrade this claim.
4. **Labels are a discharge-exceedance proxy**, not surveyed inundation extent.
   Say so wherever labels are described.

---

## 1. Title, one-paragraph summary, and contribution claims

**Title:** Terrain-Aware Multimodal Spatiotemporal Graph Neural Network for
Probabilistic Flood Early Warning in Sri Lanka.

**Summary.** Sri Lanka suffers recurrent, deadly monsoon-driven riverine
flooding, but regional flood-ML work is undermined by three methodological
failures: river-basin topology is ignored, evaluation uses random splits that
leak spatially and temporally correlated flood days between train and test, and
outputs are uncalibrated point predictions unfit for early warning. This project
builds a reproducible, leakage-controlled, 23-year daily spatiotemporal dataset
over 51 hydrologically anchored nodes in 16 Sri Lankan river basins, fusing
(i) a directed river-flow + distance-decayed spatial graph, (ii) daily
reanalysis meteorology and soil wetness, (iii) Copernicus GloFAS river
discharge, (iv) static terrain descriptors, and (v) a fixed-footprint Sentinel-1
SAR image time series, and specifies a node-level probabilistic
spatiotemporal GNN evaluated under strict temporal, spatial-basin, and
event-level holdouts with calibration-sensitive metrics.

**Claimed contributions (defensible as of today):**

- **C1 — A public, rebuildable Sri Lankan flood benchmark.** 51 nodes × 16
  basins × daily 2003–2025; 410,931 node-days; 409,350 valid supervised
  samples. Built entirely from **no-API-key public sources**, fully resumable,
  reproducible from scripts in this repo.
- **C2 — Physically grounded labels, not synthetic ones.** Flood state =
  GloFAS discharge ≥ per-node return-period threshold, thresholds estimated on
  the **train period only**. Validated against 11 independently documented Sri
  Lankan flood episodes (8/11 fire at the primary threshold, 9/11 at moderate)
  — misses reported, not hidden.
- **C3 — Leakage control as a first-class dataset artifact.** Three
  pre-computed, mutually exclusive evaluation protocols shipped as columns
  (`split_temporal`, `split_basin_holdout`, `event_id` for GroupKFold), plus
  train-only climatology/threshold/percentile fitting.
- **C4 — Explicit hydrological topology.** A 51-node graph with **35 directed
  upstream→downstream flow edges** and **204 distance-decayed spatial k-NN
  edges** (k=4, exp(−d/40 km)), so flow direction is encoded, not learned from
  proximity alone.
- **C5 — A 1:1-aligned multimodal vision branch.** Sentinel-1 SAR frames carry a
  `tabular_key = node_id + "_" + date` that joins directly to the tabular panel,
  so imagery inherits labels from the hydrological label engine at zero extra
  annotation cost.

---

## 2. Research gaps → what this project does about each

| # | Gap | Evidence in literature | Concrete design response in this project |
|---|---|---|---|
| **G1** | Spatial models ignore physical river topology (use Euclidean grids / plain k-NN) | FloodGNN-GRU (Cambridge Env. Data Science, 2024); "Explicit Water Balance Constraints for Trustworthy GNN Flood Forecasting" (MDPI Appl. Sci., 2026) — the *topology paradox* | `scripts/tabular/04_graph.py` builds a **directed** flow adjacency from a hand-curated `downstream_of` field for all 51 nodes (35 flow edges), kept as a **separate edge type** from the 204 spatial k-NN edges so a relational/multi-edge GNN can weight them independently. Static terrain (elevation, drainage proxy) is attached per node. |
| **G2** | Random splits leak across autocorrelated flood days | Roberts et al., *Ecography* (CV for temporal/spatial/hierarchical structure); "Choosing blocks for spatial cross-validation" (Frontiers in Remote Sensing) | Random splitting is **structurally impossible** in this dataset: three block protocols are precomputed (temporal 2017/2020 cut, whole-**Gin**-basin holdout, event-level `event_id` groups). All thresholds, day-of-year climatologies and empirical percentiles are fit on `date ≤ 2017-12-31` only (`02_features.py:77-94`, `140-146`). |
| **G3** | Uncalibrated, deterministic outputs unusable for warning | "Flood Uncertainty Estimation Using Deep Ensembles" (MDPI Water, 2022); "Real-Time Probabilistic Flood Forecasting Using Multiple ML Methods" (MDPI Water, 2020) | Target is an explicit **probability** P(flood at t+1); evaluation protocol mandates **Brier score, reliability diagrams and Expected Calibration Error** alongside PR-AUC; post-hoc temperature/isotonic calibration fitted on the val block only; deep-ensemble / MC-dropout variance as the uncertainty estimator. `label_confidence` (0.5–1.0) down-weights days inside a ±15% band around the threshold, so ambiguous labels do not force overconfidence. |
| **G4** | Weak dynamic–static and tabular–imagery fusion | "Intelligent flood forecasting and warning: a survey" | Three-stream architecture (§6): temporal encoder over 33 dynamic features × k-day lookback, static terrain embedding injected via FiLM-style conditioning, and a SAR CNN branch over 2-channel dB chips joined 1:1 by `tabular_key`. |

---

## 3. The dataset — measured facts

### 3.1 Headline numbers (measured from `data/processed/flood_dataset.parquet`)

| Quantity | Value |
|---|---|
| Rows (node-days) | **410,931** |
| Columns | **67** |
| Valid supervised samples (`valid_sample == 1`) | **409,350** |
| Nodes | **51** |
| Basins | **16** |
| Date span | **2003-01-01 → 2025-12-31** |
| Days per node (full-coverage nodes) | 8,036 (2003–2024) |
| Nodes with 2025 coverage | **3 only** — `KEL_HAN`, `KEL_KAD`, `KEL_COL` (365 days each = 1,095 rows) |
| Flood-state prevalence (98th-pctl, valid rows) | **1.907%** |
| Positive rate `target_flood_1d` | **1.907%** |
| Positive rate `target_flood_2d` | **2.295%** |
| Positive rate `target_flood_3d` | **2.654%** |
| Positive rate `target_onset_1d` | **0.388%** |
| Flood events (merged episodes) | **1,469** |
| Event duration (days) | mean 5.39, median 3, IQR 2–6, max 45 |
| Graph edges | **239** total = **35 flow** + **204 spatial** |
| Per-node `thr_high` (98th pctl discharge) | min 1.4, median 15.8, max 632.1 m³/s |

### 3.2 Split sizes (valid samples only)

| Protocol | Partition | N | Positive rate (`target_flood_1d`) |
|---|---|---|---|
| `split_temporal` | train (≤ 2017-12-31) | **277,899** | 2.028% |
| | val (2018–2020) | **55,896** | 0.810% |
| | test (2021-01-01 →) | **75,555** | 2.275% |
| `split_basin_holdout` | train (15 basins) | **377,330** | — |
| | test (**Gin** basin, 3 nodes) | **32,020** | — |
| `event_id` | 1,469 groups for GroupKFold | — | — |

> Note for the report: the **val block (2018–2020) is genuinely quieter** than
> train and test (0.81% vs 2.03%/2.28%). This is real hydrology, not a bug — it
> means early stopping on val PR-AUC is optimistic-biased and should be
> discussed as a limitation / motivation for event-level CV.

### 3.3 Node network (51 nodes, 16 basins)

Defined in `scripts/common/nodes.py` as tuples of
`(node_id, name, basin, lat, lon, position, downstream_of, zone)`.
`position ∈ {upstream, mid, downstream, outlet}`; `zone ∈ {wet, dry, intermediate}`.

| Basin | Nodes | Node IDs (upstream → outlet) |
|---|---|---|
| Kelani | 7 | KEL_UP (Norwood), KEL_KIT (Kitulgala), KEL_RUW (Ruwanwella), KEL_HAN (Hanwella), KEL_AVI (Avissawella), KEL_KAD (Kaduwela), KEL_COL (Kelani mouth, Colombo) |
| Kalu | 5 | KAL_BAL (Balangoda), KAL_RAT (Ratnapura), KAL_BUL (Bulathsinhala), KAL_AGA (Agalawatta), KAL_KLT (Kalutara mouth) |
| Mahaweli | 7 | MAH_NUW (Nuwara Eliya), MAH_PER (Peradeniya/Kandy), MAH_KAN (Kandy city), MAH_VIC (Victoria reservoir), MAH_MHY (Mahiyanganaya), MAH_MAN (Manampitiya), MAH_POL (Polonnaruwa) |
| Gin | 4 | GIN_NEL (Neluwa), GIN_THA (Thawalama), GIN_BAD (Baddegama), GIN_GAL (Galle mouth) |
| Nilwala | 4 | NIL_PIT (Pitabeddara), NIL_URA (Urubokka), NIL_AKU (Akuressa), NIL_MAT (Matara mouth) |
| AttanagaluOya | 4 | ATT_NIT (Nittambuwa), ATT_VEY (Veyangoda), ATT_GAM (Gampaha), ATT_JAE (Ja-Ela mouth) |
| Batticaloa | 3 | BAT_VAL (Valachchenai), BAT_ERA (Eravur), BAT_BAT (Batticaloa lagoon) |
| DeduruOya | 3 | DED_KUR (Kurunegala), DED_WAR (Wariyapola), DED_CHI (Chilaw mouth) |
| Walawe | 3 | WAL_EMB (Embilipitiya), WAL_UDA (Udawalawe reservoir), WAL_AMB (Ambalantota mouth) |
| GalOya | 2 | GAL_ING (Inginiyagala reservoir), GAL_AMP (Ampara) |
| MahaOya | 2 | MHO_GIR (Giriulla), MHO_MOU (Kochchikade mouth) |
| MalwathuOya | 2 | MAL_ANU (Anuradhapura), MAL_MAN (Mannar mouth) |
| ColomboMetro | 2 | COL_KOT (Kotte/Diyawanna), COL_KOL (Kolonnawa) |
| KirindiOya | 1 | KIR_TIS (Tissamaharama) |
| KalaOya | 1 | KLA_ELA (Eluwankulama) |
| YanOya | 1 | YAN_HOR (Horowpothana) |

Coordinates are hand-placed from gazetteer / river-course knowledge, then
**snapped by the APIs to their nearest reanalysis grid cell**; the snapped
lat/lon and the Copernicus-DEM elevation returned by the API are stored in
`data/processed/nodes_meta.csv` and merged into `nodes.csv`.

### 3.4 Graph construction (`scripts/tabular/04_graph.py`)

- **Flow edges (35).** For every node with a non-null `downstream_of` parent, one
  **directed** edge `parent → node`, `weight = 1.0`, `distance_km` = haversine
  distance. Encodes upstream→downstream routing.
- **Spatial edges (204).** For each node, its **k = 4** nearest neighbours by
  haversine distance, emitted in both directions, with
  `weight = exp(−d / 40 km)` (distance decay, `spatial_scale_km = 40`).
- Duplicate `(src, dst, edge_type)` triples are dropped.
- Both types live in one `edges.csv` with an `edge_type` column, so the model can
  either (a) run two separate message-passing streams, or (b) use a relational
  GNN (R-GCN / heterogeneous GAT) with edge-type embeddings. **Recommend (b)**
  and ablate against (a) and against a flow-edges-only / spatial-edges-only graph.

### 3.5 Feature inventory (67 columns; 33 dynamic + 2 static used by the reference loader)

**Identifiers & static context:** `node_id`, `date`, `basin`, `zone`,
`position`, `elevation_m` (Copernicus DEM, m), `drainage_proxy_q` (mean
train-period discharge, m³/s — a drainage-area proxy), `log_drainage_proxy`.

**Meteorology (daily).** `precipitation_sum` (**primary**, ERA5-Land 0.1° via
Open-Meteo, falls back to NASA POWER where missing) [mm]; `precip_era5` (raw
ERA5-Land) [mm]; `precipitation_sum_power` (NASA POWER 0.5°, kept for
comparison) [mm]; `temperature_2m_mean/max/min` [°C]; `relative_humidity_2m`
[%]; `windspeed_10m_mean`, `windspeed_10m_max` [m/s]; `shortwave_radiation`
[kWh/m²/day].

**Soil wetness (NASA POWER / MERRA-2, 0–1 saturation fraction).**
`soil_wet_top` (GWETTOP), `soil_wet_root` (GWETROOT), `soil_wet_profile`
(GWETPROF), and each one's `_anom` (anomaly vs train-period mean).

**Engineered antecedent rainfall.** `precip_sum_{2,3,5,7,10,15,30}d` (trailing
accumulations, mm); `precip_max_{3,7}d`; `api_k090` (Antecedent Precipitation
Index, exponential decay k = 0.90, recursive `acc = 0.9·acc + P_t`);
`wetdays_7d` (count of days > 1 mm in trailing 7 days).

**Hydrology (GloFAS discharge).** `river_discharge` / `discharge` [m³/s];
`log_discharge` = log1p(Q); `discharge_rise_1d` (Q_t − Q_{t−1}),
`discharge_rise_3d`; `discharge_mean_3d`, `discharge_mean_7d`;
`q_clim_mean`, `q_clim_std` (day-of-year climatology, **train only**, 15-day
centred circular smoothing); `discharge_anom` = Q − q_clim_mean;
`discharge_zscore` = anom / q_clim_std; `discharge_pctl` (empirical percentile
against the sorted **train** discharge distribution, 0–1).

**Labels & thresholds.** `flood_moderate/high/severe` (int8), corresponding
`thr_moderate/high/severe` [m³/s], `flood_state` (= `flood_high`, the primary
binary state), `label_confidence` (float 0.5–1.0; 0 if discharge missing).

**Targets.** `target_flood_1d/2d/3d` (1 if `flood_state` occurs within the next
24/48/72 h; **primary = `target_flood_1d`**); `target_onset_1d/2d/3d` (1 if a
*new* flood begins in the window, i.e. flood ahead **and** not currently
flooding); `target_next1d_discharge` [m³/s, regression];
`target_next3d_max_zscore` (max discharge z-score over t+1…t+3, regression).

**Bookkeeping.** `event_id` (NA on ~98.1% of rows — non-flood days),
`split_temporal`, `split_basin_holdout`, `valid_sample`.

**`valid_sample` definition** (`03_labels_splits.py:112-119`): row index within
node ≥ 30 (30-day feature warm-up) **AND** `target_flood_3d` not null **AND**
`river_discharge` not null. **Always filter on `valid_sample == 1` for
training and evaluation.**

**Missingness (valid rows):** everything except `event_id` (98.1% NA by design)
is ≥ 99.96% complete; `discharge_rise_3d` 0.04% missing, `discharge_rise_1d` and
all targets 0.01%.

### 3.6 Known discrepancies between docs and the current build (say this in the report, or fix the docs first)

| Claim in `README.md` / `DATASET_CARD.md` | Current measured reality |
|---|---|
| "2003-01-01 → 2024-12-31" | 2003-01-01 → **2025-12-31**, but 2025 exists for **only 3 Kelani nodes** (added to cover Cyclone Ditwah for the image branch) |
| "409,836 node-day rows / 408,255 valid" | **410,931 rows / 409,350 valid** |
| "test 2021–2024" | test is `date > 2020-12-31`, i.e. **2021–2025** in practice (2025 rows only for 3 nodes) |
| "10 documented flood episodes" | **11** (Nov 2025 Cyclone Ditwah added) |
| Branch B = 128×128 scattered chips, `image_index.csv` | Superseded by **Branch B v2**: fixed-footprint 512×512 time series (§5). The old `data/images/` chips have been deleted; `image_index.csv` no longer exists. `image_manifest.csv` (3,489 rows) is retained as a v1 artifact. |

The 2025 partial coverage is an **unbalanced panel**. For the report: either
(a) truncate to 2024-12-31 for all model experiments and keep 2025 only for the
image branch (**recommended, simplest to defend**), or (b) explicitly document
the ragged edge and mask it in the tensor builder.

---

## 4. Labelling methodology and its validation

### 4.1 Definition

For node *v*, let Q_v(t) be daily mean GloFAS river discharge. Let
q_v^(p) be the *p*-th percentile of {Q_v(t) : t ≤ 2017-12-31}. Then

- `flood_moderate` = 1[Q ≥ q^(0.90)]  (~frequent high flow)
- `flood_high`     = 1[Q ≥ q^(0.98)]  (**primary**; ~2-year return period)
- `flood_severe`   = 1[Q ≥ q^(0.995)] (~5–10-year return period)

This is the **return-period-exceedance** definition used operationally by
GloFAS itself. Thresholds are per node (they range 1.4 → 632.1 m³/s across the
network), which is what makes a small wet-zone tributary and the lower Mahaweli
comparable on one scale.

`label_confidence` = 0.5 + 0.5·clip(|Q − q^(0.98)| / (0.15·q^(0.98)), 0, 1),
and 0 where discharge is missing — i.e. days within ±15% of the threshold are
down-weighted to 0.5. Use it as a per-sample loss weight.

### 4.2 Targets are strictly causal

Targets are computed **per node, within node boundaries only**, from *future*
days: `target_flood_h = max(flood_state[t+1 … t+h])`. No feature at time t uses
information after t. The 30-day warm-up in `valid_sample` guarantees the
30-day rolling features are fully populated.

### 4.3 Event construction

Consecutive `flood_state == 1` days per node are merged into episodes with a
**2-day gap tolerance**; each episode gets `event_id = <node>_<YYYYMMDD of
start>`. 1,469 episodes total. These are the groups for GroupKFold.

### 4.4 Validation against documented Sri Lankan floods (`docs/validation_report.md`)

Detection window ±4 days (downstream peaks lag rainfall). "Fired@high" counts
node-days triggering the 98th-pctl threshold among affected-basin nodes.

| Event | Basins | Fired@high | Fired@moderate | Max pctl | Verdict |
|---|---|---|---|---|---|
| May 2003 SW floods (Ratnapura/Matara) | Kalu, Nilwala, Gin | 0/13 | 0/13 | 0.829 | ⚠️ missed |
| Jan 2011 Eastern floods | Batticaloa, GalOya, Mahaweli | 12/12 | 12/12 | 1.000 | ✅ |
| Feb 2011 Eastern/NC floods | Batticaloa, GalOya, Mahaweli | 12/12 | 12/12 | 1.000 | ✅ |
| Jun 2014 Kelani/Kalu floods | Kelani, Kalu | 0/12 | 1/12 | 0.918 | ◧ moderate only |
| May 2016 Kelani floods (Aranayake) | Kelani, AttanagaluOya | 11/11 | 11/11 | 1.000 | ✅ |
| May 2017 SW floods (Kalu/Gin/Nilwala) | Kalu, Gin, Nilwala | 0/13 | 0/13 | 0.896 | ⚠️ missed |
| May 2018 SW floods | Kelani, Kalu, AttanagaluOya | 1/16 | 6/16 | 0.981 | ✅ |
| Dec 2019 floods | Kelani, Kalu, Gin | 1/16 | 16/16 | 0.990 | ✅ |
| Nov 2021 Western floods | Kelani, Kalu, AttanagaluOya | 5/16 | 14/16 | 0.997 | ✅ |
| Jun 2024 SW / Western floods | Kelani, Kalu, Nilwala, Gin | 14/20 | 19/20 | 0.994 | ✅ |
| Nov 2025 Cyclone Ditwah floods | Kelani, Kalu, AttanagaluOya | 3/3 | 3/3 | 1.000 | ✅ |

**Detection: 8/11 at the high threshold, 9/11 at moderate.** Misses concentrate
in **pre-2011 south-western flash floods (2003, 2014, 2017)**, where GloFAS
reanalysis under-represents small, fast-responding wet-zone catchments.
Post-2011 events are captured reliably. This is a documented, well-known GloFAS
limitation — report it explicitly as a label-quality bound on achievable recall
for early-period wet-zone events.

### 4.5 Per-basin flood-day rates (from the validation report)

| Basin | Flood-days | Node-days | Rate % | | Basin | Flood-days | Node-days | Rate % |
|---|---|---|---|---|---|---|---|---|
| Mahaweli | 1099 | 56252 | 1.95 | | MahaOya | 320 | 16072 | 1.99 |
| Kelani | 1096 | 57347 | 1.91 | | ColomboMetro | 308 | 16072 | 1.92 |
| Kalu | 717 | 40180 | 1.78 | | MalwathuOya | 306 | 16072 | 1.90 |
| Gin | 665 | 32144 | 2.07 | | GalOya | 266 | 16072 | 1.66 |
| Nilwala | 628 | 32144 | 1.95 | | KalaOya | 164 | 8036 | 2.04 |
| AttanagaluOya | 608 | 32144 | 1.89 | | KirindiOya | 156 | 8036 | 1.94 |
| DeduruOya | 465 | 24108 | 1.93 | | YanOya | 155 | 8036 | 1.93 |
| Walawe | 463 | 24108 | 1.92 | | Batticaloa | 391 | 24108 | 1.62 |

The near-constant ~1.9% is *by construction* (98th percentile per node); the
deviations reflect changed post-2017 flow regimes relative to the train-period
thresholds — dry-zone basins (Batticaloa 1.62%, GalOya 1.66%) came in lower,
Gin (2.07%) and KalaOya (2.04%) higher.

---

## 5. Modality 4 — the Sentinel-1 SAR image branch

### 5.1 Why SAR

C-band radar penetrates the monsoon cloud cover that blinds optical sensors
exactly when floods happen. Open water is specular → very low VV backscatter;
flooded vegetation can be *bright* via double-bounce. Both signatures are
usable.

### 5.2 Branch B **v1** (superseded, retained as an artifact)

`scripts/imagery/06_image_manifest.py` → `07_fetch_images_mpc.py`. Sampled
scattered 128×128 chips at nodes on selected dates: **3,489 chip requests** —
1,398 positive (699 at flood peak + 699 **pre-event**, ~3 days before onset —
the "predict before it floods" case) and 2,091 dry negatives; positive severity
~956 high / ~442 severe. `data/processed/image_manifest.csv` still exists
(421 KB). The fetched `.npy` chips and `image_index.csv` have been **deleted**;
`docs/IMAGE_DATASET.md` still documents v1 and is out of date.

### 5.3 Branch B **v2** (current) — fixed-location SAR time series

Builder: **`scripts/imagery/10_build_image_dataset.py`**.

Rationale: scattered chips give the CNN a different footprint every sample, so
it cannot learn *change*. v2 pins three flood-prone Kelani sites forming a
**downstream transect** and pulls **every Sentinel-1 pass** at the *same*
footprint, producing clean dry → flood → recede sequences.

| Setting | Value |
|---|---|
| Sites | `KEL_HAN` Hanwella (6.902, 80.082), `KEL_KAD` Kaduwela (6.933, 79.983), `KEL_COL` Kelani mouth / Colombo (6.970, 79.873) |
| Window | 2015-01-01 → 2025-12-31 (Sentinel-1 operational from Oct 2014) |
| Footprint | 5,000 m per side, resampled to **512 × 512** px (≈10 m/px native) |
| Source | Microsoft Planetary Computer STAC, collection **`sentinel-1-rtc`** (Radiometric Terrain Corrected γ⁰, UTM COGs), **anonymous, no account** |
| Bands | VV (+ VH where dual-pol), converted to dB, clipped to **[−30, +5] dB**, nodata → −30 |
| Mode filter | IW (`sar:instrument_mode = IW`); one frame per calendar date, dual-pol preferred |
| Storage | `data/timeseries/<site>/<YYYY-MM-DD>.npy` — float16 `[2, 512, 512]` (VV, VH) — plus a viewable `_rgb.png` false-colour (R = VV, G = VH, B = VV−VH) |
| Water rule | pixel counts as open water if **VV < −17 dB** |
| Derived per-frame scalars | `water_fraction`, `vv_mean`, `vh_mean`, `valid_fraction` (nodata floor excluded) |
| Index | `data/processed/image_dataset.csv` — one row per (site, satellite pass) |

Index columns, grouped: **(A) identity** `image_id, site_id, site_name, basin,
lat, lon, array_path, png_path`; **(B) acquisition** `date, scene_id, platform,
orbit_state, rel_orbit, footprint_m, px, has_vh`; **(C) label (joined from
Branch A on `tabular_key`)** `tabular_key, label (= flood_state),
flood_within_3d (= target_flood_3d), severity, discharge, discharge_pctl,
event_id`; **(D) SAR features** `water_fraction, vv_mean, vh_mean,
valid_fraction`; **(E) split** `split_temporal`.

**Cyclone Ditwah (Nov 2025)** is the largest event in the window. The 26 Nov
satellite pass is a genuine **pre-event frame**: same-day `label = 0` but
`flood_within_3d = 1` — the single most valuable sample in the vision branch for
demonstrating lead time.

### 5.4 v2 build status — **INCOMPLETE** (measured 2026-08-01)

| Site | Frames on disk | Shapes | Date range |
|---|---|---|---|
| KEL_HAN | **263** | 260 × (2,512,512) + **3 × (2,256,256)** | 2015-02-16 → 2025-12-20 |
| KEL_KAD | **55** | 52 × (2,512,512) + **3 × (2,256,256)** | 2015-02-16 → 2018-02-12 |
| KEL_COL | **3** | **3 × (2,256,256)** | 2016-05-05 → 2016-05-29 |

`data/processed/image_dataset.csv` **has not been written yet** (the builder
writes it only after a full pass completes). Outstanding work:

1. Delete the **9 leftover 256×256 frames** (3 per site, dates 2016-05-05 /
   -22 / -29) left over from the older `09_timeseries_sample.py` run, so they
   refetch at 512 px. Rule: delete any frame whose shape ≠ (2, 512, 512).
2. Re-run `python scripts/imagery/10_build_image_dataset.py` (**resumable** — it
   skips frames whose `.npy` already exists) until KEL_KAD and KEL_COL reach
   parity with KEL_HAN (expect ~260 frames each; ~780 total).
3. Verify counts / label balance and confirm the Ditwah 26-Nov-2025 frame shows
   elevated `water_fraction`.
4. Update `docs/IMAGE_DATASET.md` (still describes v1), `scripts/README.md`,
   and add `data/timeseries/` to `.gitignore`. None of the v2 work is committed
   to git yet.

**For the report:** describe the vision branch as *designed and partially
materialised* — 321 frames of an expected ~780 — and either present it as
future work or restrict image experiments to KEL_HAN, which is essentially
complete (263 frames, 2015–2025).

---

## 6. The DNN problem formulation and proposed architecture

### 6.1 Formal statement

Given a directed, multi-relational graph G = (V, E) with |V| = N = 51 and
E = E_flow ∪ E_spatial (|E_flow| = 35 directed, |E_spatial| = 204 weighted by
exp(−d/40 km)), and for each node v and day t:

- dynamic sequence **X**_v,t−k+1:t ∈ ℝ^{k × F_d}, F_d = 33 (§3.5), lookback
  k (default 14, ablate 7 / 14 / 30);
- static terrain vector **s**_v ∈ ℝ^{F_s}, F_s = 2 (`elevation_m`,
  `log_drainage_proxy`) — optionally + one-hot `zone` (3) and `position` (4);
- optional SAR frame **I**_v,t ∈ ℝ^{2 × 512 × 512} (VV, VH in dB) from the
  nearest Sentinel-1 pass ≤ t, with an age-in-days feature;

learn f_θ producing a **calibrated probability**

  p̂_v,t = P(Y_v,t+1 = 1 | X, s, I, G),  Y_v,t+1 = 1[Q_v(t+1) ≥ q_v^(0.98)]

with auxiliary heads for t+2, t+3 (`target_flood_2d/3d`), onset
(`target_onset_1d`), and the two regression targets
(`target_next1d_discharge`, `target_next3d_max_zscore`).

This is **node-level probabilistic spatiotemporal classification on a fixed
graph with a rolling temporal window** — i.e. transductive in node identity,
inductive in time (temporal split) and, for the Gin holdout, inductive in
node identity too.

### 6.2 Proposed architecture (three streams + fusion head)

```
 dynamic X[k,33] ──► temporal encoder (GRU / TCN / temporal-attention)  ──┐
                                                                         │
 static s[F_s]   ──► MLP ──► FiLM (γ, β) modulating the temporal state ───┤
                                                                         ├─► spatial
 SAR I[2,512,512]──► light CNN (ResNet-ish, 2-ch stem) ──► pooled emb ────┘    message
                                                                              passing
                                                                                 │
        relational GNN over E_flow (directed) + E_spatial (weighted)  ◄──────────┘
                                                                                 │
                                                     ┌───────────────────────────┤
                                            heads:   t+1 (primary) · t+2 · t+3 · onset · Q_{t+1} · z-max
```

- **Temporal encoder:** per-node GRU (or dilated TCN) over the k-day window →
  h_v,t ∈ ℝ^H. Shared weights across nodes.
- **Static conditioning (addresses G4):** FiLM — h ← γ(s) ⊙ h + β(s) — so
  terrain *modulates* the response to rainfall rather than being concatenated
  and washed out. Ablate against plain concatenation.
- **Spatial layer (addresses G1):** relational GNN with **separate parameters
  per edge type**; flow edges carry direction (upstream→downstream only),
  spatial edges are symmetric and weighted. Suggested: 2 layers of R-GAT /
  edge-type-conditioned GraphSAGE. Optionally add a learned lag on flow edges
  (routing delay) as an extension.
- **Vision stream (addresses G4):** small CNN on the 2-channel dB chip; only 3
  of 51 nodes have imagery, so fuse with a **presence mask + zero embedding**
  for imageless nodes, and report the image ablation on the KEL_HAN transect
  only. Cheap alternative worth reporting: feed the scalar `water_fraction`,
  `vv_mean`, `vh_mean`, `valid_fraction` as extra dynamic features (no CNN) —
  a strong, honest baseline for "does SAR add anything?".
- **Output head:** sigmoid per horizon + regression heads; multi-task loss.

### 6.3 Loss

L = Σ_h w_h · FocalBCE(p̂_h, y_h; α, γ) · c  +  λ_reg · Huber(regression heads)

where `c = label_confidence` (per-sample weight, 0.5 in the ±15% ambiguity
band), focal γ ≈ 2, α tuned to the ~1.9% positive rate. Report class-weighted
BCE vs focal as an ablation.

### 6.4 Calibration (addresses G3)

1. Train the base model.
2. Fit **temperature scaling** (or isotonic regression) on the **val block
   only** (2018–2020) — never on test.
3. Uncertainty: **deep ensemble** of M = 5 seeds (recommended; the MDPI Water
   2022 precedent) and/or MC dropout; report predictive mean and variance.
4. Report reliability diagrams, **ECE**, **Brier score**, and Brier
   decomposition (reliability / resolution / uncertainty) per split.

### 6.5 Metrics — mandatory reporting set

- **Ranking:** PR-AUC (**headline**, given 1.9% positives), ROC-AUC.
- **Calibration:** Brier score, ECE (15 bins), reliability diagram.
- **Operational:** POD / recall, FAR, CSI (critical success index), and F1 at
  thresholds chosen on the val block; **precision–recall at fixed
  false-alarm budgets** (e.g. FAR ≤ 20%), which is what a warning agency
  actually cares about.
- **Event-level:** fraction of the 1,469 events detected with ≥ 1 day of lead
  time; mean lead time on detected events. This is the most report-worthy
  metric and is not the same as node-day recall.
- Never report plain accuracy (98.1% by predicting "no flood" always).

### 6.6 Baselines to beat (all must use the same splits and `valid_sample` filter)

1. **Persistence** — p̂ = flood_state(t).
2. **Climatology** — per-node, per-day-of-year positive rate from train.
3. **Discharge-percentile rule** — threshold on `discharge_pctl(t)` tuned on val
   (this is essentially "what GloFAS already tells you"; beating it is the real
   bar).
4. **Gradient-boosted trees** (XGBoost / LightGBM) on flattened node-day
   features — historically very strong on tabular hydrology; **do not omit it**.
5. **LSTM / GRU per node, no graph** — isolates the value of G1.
6. **GNN with spatial edges only** (no flow edges) — the direct G1 ablation.
7. **Full model** — flow + spatial + static FiLM (+ SAR where available).

### 6.7 Ablation grid to report

| Axis | Variants |
|---|---|
| Graph | none · spatial-only · flow-only · flow + spatial (relational) |
| Static fusion | none · concat · FiLM |
| Lookback k | 7 · 14 · 30 days |
| Loss | BCE · class-weighted BCE · focal · focal × `label_confidence` |
| Calibration | raw · temperature · isotonic · deep ensemble |
| Imagery (KEL_HAN only) | none · SAR scalars · SAR CNN |
| Split protocol | temporal · Gin-basin holdout · event GroupKFold |

**Headline experiment for G2:** train the *same* model under a **random**
node-day split and under `split_temporal`, and report the inflation in PR-AUC.
That single comparison is the strongest empirical evidence for Gap 2 and costs
one extra training run.

---

## 7. Pipeline, reproducibility, and file inventory

### 7.1 Execution order

```bash
pip install -r requirements.txt
python scripts/tabular/01_download.py     # raw reanalysis, resumable, no API key
python scripts/tabular/run_all.py         # runs 02 → 05
# optional vision branch (v2):
pip install pystac-client planetary-computer rasterio pyproj pillow
python scripts/imagery/10_build_image_dataset.py   # resumable
```

| Stage | Script | Produces |
|---|---|---|
| shared | `scripts/common/config.py` | dates, API URLs, POWER parameter map, severity percentiles, horizons, split dates, holdout basin |
| shared | `scripts/common/nodes.py` | the 51 nodes + `downstream_of` topology → `nodes.csv` |
| 01 | `tabular/01_download.py` | `data/raw/<node>_{weather,flood,precip}.parquet`, `<node>_meta.json`, `nodes_meta.csv` |
| 02 | `tabular/02_features.py` | `panel_features.parquet` (34 MB), `thresholds.json` |
| 03 | `tabular/03_labels_splits.py` | **`flood_dataset.parquet`** (38 MB), `events.csv` |
| 04 | `tabular/04_graph.py` | `nodes.csv`, `edges.csv` |
| 05 | `tabular/05_validate.py` | `docs/validation_report.md` |
| 06/07 | `imagery/06_image_manifest.py`, `07_fetch_images_mpc.py` | v1 chip manifest / chips (**superseded**) |
| 09 | `imagery/09_timeseries_sample.py` | v2 precursor — contact sheets `data/timeseries/<site>_contact.png` |
| 10 | `imagery/10_build_image_dataset.py` | `data/timeseries/<site>/*.npy` + `_rgb.png`, `image_dataset.csv` |
| loaders | `loaders/torch_loader.py` | `X[T,N,F]`, `y[T,N]`, `mask`, `edge_index`, `edge_weight` for PyTorch/PyG |
| loaders | `loaders/image_loader.py` | pairs chips with tabular rows via `tabular_key` |
| viz | `viz/preview_chips.py`, `viz/08_export_png.py` | **viewing only — never train on the 8-bit PNGs** |

### 7.2 Data sources

| Modality | Source | Backend / resolution | Key needed |
|---|---|---|---|
| Precipitation (primary) | Open-Meteo Archive API | **ERA5-Land, 0.1°** | No |
| Precip / temp / humidity / wind / radiation / soil wetness | **NASA POWER** daily point API | MERRA-2 / GEOS, ~0.5° | No |
| River discharge | Open-Meteo **Flood** API | Copernicus **GloFAS** reanalysis, ~0.05° | No |
| Elevation | Copernicus DEM via Open-Meteo | grid-cell | No |
| SAR imagery | Microsoft Planetary Computer STAC | **Sentinel-1 RTC** (γ⁰, IW, 10 m) | No (anonymous) |
| River topology | Hand-built (`scripts/common/nodes.py`) | expert placement | — |

All downloads are **cached, resumable, rate-limit-aware** and require **no API
key** — a genuine reproducibility contribution worth stating.

Licensing: derived from CC-BY / free-access reanalysis products. Attribution to
ECMWF/Copernicus, GloFAS, NASA POWER, Open-Meteo and Microsoft Planetary
Computer is required. Repo licence: see `LICENSE`.

### 7.3 What is / isn't in git

Tracked (small): `nodes.csv`, `nodes_meta.csv`, `edges.csv`, `events.csv`,
`thresholds.json`, `image_manifest.csv`, `sample_preview.csv`, all scripts and
docs. Not tracked (heavy, rebuildable): `flood_dataset.csv` (~250 MB),
`flood_dataset.parquet`, `panel_features.parquet`, `data/raw/*`,
`data/timeseries/*`. Distribution of the prebuilt archive is intended via
GitHub Releases / Zenodo DOI.

---

## 8. Limitations — state these plainly

1. **Reanalysis, not gauges.** ERA5-Land rainfall and GloFAS discharge are model
   products. Skill degrades for small flashy catchments and reservoir-regulated
   rivers (Mahaweli/Victoria, Walawe/Udawalawe, Gal Oya/Inginiyagala are all
   dam-controlled nodes in this network). Directly visible in §4.4: the three
   missed events are all pre-2011 wet-zone flash floods.
2. **Proxy labels.** Discharge exceedance ≠ mapped inundation. Urban drainage
   failure (relevant to the two ColomboMetro nodes) can flood without a basin
   discharge spike, and vice versa.
3. **Coarse meteorology.** NASA POWER is ~0.5°, so nearby upland/lowland nodes
   can share a grid cell; discharge at ~0.05° resolves basins more finely. The
   ERA5-Land 0.1° precipitation partially mitigates this.
4. **Node placement is expert-approximate** and snapped to reanalysis grid
   cells — not surveyed gauge locations.
5. **Class imbalance ~1.9%** — requires focal/weighted loss and PR-based
   metrics; accuracy and ROC-AUC alone are misleading.
6. **Val block is anomalously quiet** (0.81% positives vs 2.03/2.28%) — model
   selection on it is biased; prefer event-level CV for selection.
7. **Ragged 2025 edge** — only 3 of 51 nodes have 2025 data.
8. **Imagery is spatially thin** — 3 Kelani sites, and only KEL_HAN is
   near-complete. The vision branch supports a *case study*, not a
   network-wide claim. Sentinel-1's ~12-day revisit also means **lead time comes
   from the hydro-meteorological branch, not the images**; imagery contributes
   current inundation state, river morphology and severity sharpening. Say this
   explicitly — it is the honest version of "multimodal".
9. **No hydraulic/conservation constraints** in the loss — hence
   *terrain-aware*, not *physics-guided*.

---

## 9. Roadmap (v2 → v3)

1. Finish the Branch B v2 fetch (§5.4) and publish `image_dataset.csv`.
2. Sentinel-1 change-detection flood masks → **observed inundation-extent**
   labels for major events, as an independent cross-check on the discharge
   proxy.
3. World Bank / UNU / UNESCO-IHP-WINS flood polygons as external label
   validation.
4. Sri Lanka **Irrigation Department** gauge & discharge records — the single
   upgrade that would justify the term *physics-guided* and enable
   water-balance / flow-continuity constraints.
5. Sub-daily (3–6 h) targets via hourly IMERG/GPM ingestion — the step that
   turns a 24-h-ahead daily model into an operational nowcast.
6. Learned routing lags on flow edges (travel time upstream→downstream).

---

## 10. Reference list (as cited in the feasibility study)

**G1 — topology**
- *FloodGNN-GRU: a spatio-temporal graph neural network for flood prediction*, Cambridge Environmental Data Science, 2024.
- *Explicit Water Balance Constraints for Trustworthy Graph Neural Network Flood Forecasting*, MDPI Applied Sciences, 2026.

**G2 — leakage / cross-validation**
- Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure*, Ecography.
- *Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study*, Frontiers in Remote Sensing.

**G3 — uncertainty & calibration**
- *Flood Uncertainty Estimation Using Deep Ensembles*, MDPI Water, 2022.
- *Real-Time Probabilistic Flood Forecasting Using Multiple Machine Learning Methods*, MDPI Water, 2020.

**G4 — multimodal fusion**
- *Intelligent flood forecasting and warning: a survey*, OAE Intelligent Robotics / IEEE survey.

**Data / product citations required**
- Copernicus Emergency Management Service — **GloFAS** global river discharge reanalysis.
- ECMWF **ERA5-Land** hourly/daily reanalysis (via Open-Meteo Archive API).
- **NASA POWER** (Prediction Of Worldwide Energy Resources) daily point data, MERRA-2/GEOS.
- **Copernicus Sentinel-1** RTC (γ⁰) via Microsoft Planetary Computer.
- **Copernicus DEM** elevation.

---

## 11. Suggested report outline

1. Introduction — Sri Lankan flood context, cost, why early warning.
2. Related work & the four gaps (§2).
3. Study area & node network (§3.3) — include a basin map.
4. Data sources and acquisition pipeline (§7.1–7.2) — include the Mermaid flow diagram in `scripts/README.md`.
5. Feature engineering (§3.5).
6. Label construction & validation (§4) — the 11-event table is the credibility anchor.
7. Graph construction (§3.4).
8. Multimodal SAR branch (§5) — present as case study + partial build.
9. Problem formulation & architecture (§6.1–6.3).
10. Evaluation protocol: splits, baselines, ablations, calibration, metrics (§6.4–6.7).
11. Expected results / experimental plan — **no fabricated numbers**.
12. Limitations (§8) and roadmap (§9).
13. Reproducibility statement (§7.3) and references (§10).

**Figures worth generating:** basin/node map with flow edges; per-node
threshold distribution (1.4 → 632.1 m³/s); class-balance and split-size bars;
a Kelani discharge time series with threshold and event shading; the KEL_HAN
SAR contact sheet (`data/timeseries/KEL_HAN_contact.png`) showing dry → flood →
recede; a reliability-diagram template.
