# Dataset Card — Sri Lanka Multimodal Spatiotemporal Flood Dataset (v1)

## Motivation
Sri Lanka experiences recurrent, deadly monsoon floods (2003, 2011, 2016, 2017…).
No single ready-made ML-ready Sri Lankan flood dataset exists. This dataset was
constructed to enable **probabilistic flood early-warning** research with a
**spatiotemporal graph neural network**, with a deliberate emphasis on
**leakage-free, event-based evaluation** — the aspect most often done wrong in
flood-ML papers.

## Composition
- **Instances:** node-day records. A node is a fixed point on a river basin; each
  day carries meteorological + hydrological features and future-looking flood targets.
- **Nodes:** 51 across 16 basins, spanning wet-zone (SW-monsoon) and dry-zone
  (NE-monsoon) regimes and upstream→outlet positions.
- **Span:** 2003-01-01 to 2024-12-31, daily.
- **Size:** 409,836 node-days (408,255 valid supervised samples) — two orders of
  magnitude above the "handful of World Bank flood events" that motivated
  building this in the first place.
- **Modalities:** rainfall & antecedent-rainfall indices, temperature, wind, ET₀,
  radiation, (optional) soil moisture & humidity, river discharge & derived
  hydrology, static terrain (elevation, drainage proxy), and graph topology.

## Collection process
- **Weather/soil:** Open-Meteo Archive API (ECMWF ERA5 / ERA5-Land reanalysis).
- **Discharge:** Open-Meteo Flood API (Copernicus **GloFAS** reanalysis).
- **Elevation:** Copernicus DEM via Open-Meteo.
- All retrieved programmatically (`scripts/01_download.py`), UTC-aligned, cached
  to parquet, fully resumable. No API key required.

## Labeling
Flood state = **river discharge ≥ per-node return-period threshold** (90/98/99.5th
percentile of the training-period discharge distribution → moderate/high/severe).
The primary binary state uses the 98th-percentile (~2-year) threshold, the
standard return-period-exceedance definition used operationally by GloFAS.

**Prediction targets** are strictly future-looking (t+1…t+3 days) and computed
per node without crossing node boundaries, so there is no temporal leakage into
the target. `label_confidence` down-weights days near the threshold.

### Label validation
`scripts/05_validate.py` cross-checks flood_state against **10 independently
documented flood episodes** (2003–2024). Results in `docs/validation_report.md`.

## Recommended use & splits
Use `valid_sample == 1`. Choose **one** evaluation protocol:
1. **Temporal generalization** — train ≤2017, val 2018–2020, test 2021–2024.
2. **Spatial generalization** — hold out the **Gin** basin (`split_basin_holdout`).
3. **Event-level CV** — GroupKFold on `event_id` (+ time-block negatives).

Never split node-days randomly: days within one flood are highly correlated and
would leak between train and test, inflating scores.

## Distribution & licensing
Derived from CC-BY / free-access reanalysis products (see
`docs/SOURCES_AND_LICENSE.md`). Attribution to ECMWF/Copernicus, GloFAS, and
Open-Meteo is required. Redistribution of derived features is permitted with
attribution.

## Known limitations & biases
- **Reanalysis, not gauges.** ERA5-Land rainfall and GloFAS discharge are
  model products; skill degrades for small flashy catchments and reservoir-
  regulated rivers (e.g. parts of Mahaweli/Walawe with major dams).
- **Proxy labels.** Discharge exceedance ≠ mapped inundation. A cell can flood
  locally (urban drainage failure) without a basin-scale discharge spike, and
  vice-versa. Treat labels as hydrological flood-hazard, not surveyed extent.
- **Class imbalance.** Flood days are rare (~2% by construction); use
  class weighting / focal loss and report PR-AUC, not just accuracy/ROC-AUC.
- **Node placement** is expert-approximate and snapped to reanalysis grid cells.

## Roadmap to v2 (stated honestly)
1. **Sentinel-1 SAR** change-detection flood masks (Google Earth Engine) to add
   observed inundation-extent labels for major events.
2. **World Bank / UNU / UNESCO-IHP-WINS** flood polygons as independent label
   cross-checks.
3. **Irrigation Department gauge/discharge** to upgrade from *terrain-aware* to
   genuinely *physics-guided* (conservation & flow-continuity constraints).
4. Sub-daily (3–6 h) targets once hourly IMERG/GPM ingestion is added.
