# `scripts/` — Pipeline Guide

Every script that builds the Sri Lanka multimodal flood dataset lives here. This
file explains **what each script does, the order to run them, and what each one
produces**.

The dataset has two branches that share the same node network and flood labels:

- **Branch A — Tabular / graph** (rainfall, soil, river discharge → flood labels & a spatiotemporal graph). This is the *label engine*.
- **Branch B — Imagery** (Sentinel-1 SAR radar chips), labeled automatically by Branch A.

All outputs land in `data/raw/` (cached API downloads) and `data/processed/`
(analysis-ready tables). Nothing here needs an API key.

---

## Quick start

```bash
pip install numpy pandas pyarrow requests            # Branch A (tabular)

# ---- Branch A: build the tabular dataset + graph ----
python scripts/01_download.py        # download raw reanalysis (slow, cached, resumable)
python scripts/run_all.py            # runs steps 02 -> 05 in order

# ---- Branch B: build the SAR image dataset (optional) ----
pip install pystac-client planetary-computer rasterio pyproj pillow matplotlib
python scripts/06_image_manifest.py  # decide which chips to fetch
python scripts/07_fetch_images_mpc.py  # download them (network-heavy, resumable)
```

After Branch A you have `data/processed/flood_dataset.parquet` — the core,
analysis-ready dataset. Everything else builds on it.

---

## Execution order at a glance

```mermaid
flowchart TD
    cfg[config.py + nodes.py<br/><i>shared config</i>] --> s1
    s1[01_download.py<br/>raw weather / discharge / precip] --> s2
    s2[02_features.py<br/>panel_features.parquet] --> s3
    s3[03_labels_splits.py<br/><b>flood_dataset.parquet</b> + events.csv] --> s4
    s3 --> s5[05_validate.py<br/>validation_report.md]
    s4[04_graph.py<br/>nodes.csv + edges.csv] --> tl
    s3 --> s6[06_image_manifest.py<br/>image_manifest.csv]
    s6 --> s7[07_fetch_images_mpc.py<br/>data/images/*.npy + image_index.csv]
    s7 --> il[image_loader.py]
    s7 --> pv[preview_chips.py / 08_export_png.py<br/><i>viewing only</i>]
    s4 --> tl[torch_loader.py]
    s3 --> tl
    s3 --> s9[09_timeseries_sample.py<br/>fixed-location Kelani series]
    style s3 fill:#1f6feb,color:#fff
    style s4 fill:#1f6feb,color:#fff
```

`run_all.py` covers steps **02 → 05**. Steps 01, 06, 07 are run individually
because they hit external APIs and are slow/resumable.

---

## Branch A — tabular dataset & graph

| Order | Script | What it does | Run | Produces |
|---|---|---|---|---|
| — | **`config.py`** | Central settings: date range, API URLs, feature/parameter maps, severity percentiles, prediction horizons, train/val/test split dates, holdout basin. *Imported by everything; not run directly.* | *(imported)* | — |
| — | **`nodes.py`** | Defines the 50+ river-monitoring **nodes** (id, basin, lat/lon, hydrological position, `downstream_of` topology, climate zone). | `python scripts/nodes.py` | `data/processed/nodes.csv` |
| 1 | **`01_download.py`** | Downloads real reanalysis per node from Open-Meteo / NASA POWER (both free, no key): daily weather + soil wetness, GloFAS river discharge, high-res ERA5 precip. Serial + rate-limit-aware + **resumable** (skips cached feeds). | `python scripts/01_download.py [all\|discharge\|weather\|precip]` | `data/raw/<node>_*.parquet`, `nodes_meta.csv` |
| 2 | **`02_features.py`** | Merges the raw feeds into a per-node daily panel and engineers features: antecedent precip (rolling sums, API index), discharge dynamics (rise, anomaly, z-score, percentile), soil-wetness anomalies, static terrain. Computes per-node return-period **thresholds** on train years only (no leakage). | `python scripts/02_features.py` | `panel_features.parquet`, `thresholds.json` |
| 3 | **`03_labels_splits.py`** | Turns features into the **final dataset**: flood state (discharge ≥ threshold), multi-horizon targets (flood within 1/2/3 days, onset, next-day discharge), merges flood **events**, adds temporal & basin-holdout splits, and a `valid_sample` flag. | `python scripts/03_labels_splits.py` | **`flood_dataset.parquet`**, `events.csv` |
| 4 | **`04_graph.py`** | Builds the graph for the spatiotemporal GNN: directed **flow** edges (upstream→downstream) + distance-decayed **spatial** kNN edges. | `python scripts/04_graph.py` | `nodes.csv`, `edges.csv` |
| 5 | **`05_validate.py`** | Sanity/EDA report: checks the discharge-derived labels fire during **documented historical Sri Lankan floods**, plus class balance, split sizes, per-basin counts, missingness. | `python scripts/05_validate.py` | `docs/validation_report.md` |
| — | **`run_all.py`** | Convenience runner for steps **02 → 05** (after `01_download.py` has cached raw data). | `python scripts/run_all.py` | all of the above |

## Branch B — Sentinel-1 SAR imagery

Requires: `pip install pystac-client planetary-computer rasterio pyproj pillow matplotlib`

| Order | Script | What it does | Run | Produces |
|---|---|---|---|---|
| 6 | **`06_image_manifest.py`** | Uses the flood labels to decide **which** satellite chips to fetch and how they're labeled — flood-peak (label 1), pre-event antecedent (label 1, the "predict before it floods" case), and dry negatives (label 0). Downloads no imagery. | `python scripts/06_image_manifest.py` | `image_manifest.csv` |
| 7 | **`07_fetch_images_mpc.py`** | Fetches each manifest chip from Microsoft Planetary Computer (`sentinel-1-rtc`), reads a window around the node, converts VV/VH to dB, saves a 2-channel chip. **Resumable**, retries MPC throttles. Knobs: `MAX_CHIPS`, `ONLY_LABEL`, `SHUFFLE`, `IMG_WORKERS`. | `python scripts/07_fetch_images_mpc.py` | `data/images/<id>.npy` (float16 `[2,128,128]`), `image_index.csv` |
| 9* | **`09_timeseries_sample.py`** | **Fixed-location time series** (new approach): pins flood-prone Kelani coordinates and pulls every Sentinel-1 pass across a date window at the *same* footprint, so you see dry→flood→recede. Bigger/sharper frames + a dated contact sheet per site. Knobs: `START`, `END`, `WIN_M`, `OUT_PX`, `MAX_FRAMES`. | `python scripts/09_timeseries_sample.py` | `data/timeseries/<site>/*.npy`, `*_rgb.png`, `<site>_contact.png` |

\* Independent of step 8 — it's an alternative imaging strategy, not a pipeline stage.

## Loaders (feed a model)

| Script | What it does | Run |
|---|---|---|
| **`image_loader.py`** | Pairs each fetched SAR chip with its matching tabular feature row (via `tabular_key`) → aligned arrays for a two-branch fusion model. NumPy only. | `python scripts/image_loader.py` (sanity check) |
| **`torch_loader.py`** | Reference PyTorch/PyG loader: turns `flood_dataset.parquet` + graph into `X[T,N,F]` node-time tensors, targets, mask, and `edge_index`. Torch only needed if you call `build_torch_geometric()`. | `python scripts/torch_loader.py` (sanity check) |

## Viewers / utilities (never train on these)

| Script | What it does | Run | Produces |
|---|---|---|---|
| **`preview_chips.py`** | Renders sample grids of fetched chips (grayscale VV + false-color) for a quick look. | `python scripts/preview_chips.py` | `docs/chips_*.png` |
| **`08_export_png.py`** | Exports **every** `.npy` chip to viewable PNGs (VV / VH / false-color), sorted into `flood/`, `dry/`. 8-bit lossy — viewing only. Knobs: `OUT`, `SCALE`, `MODE`. | `python scripts/08_export_png.py` | `data/images_png/` |

> ⚠️ **PNGs are a viewing copy only.** The model trains on the raw float16 dB
> `.npy` arrays (real radar measurements incl. negative values); 8-bit PNGs are
> lossy-normalized. See `docs/IMAGE_DATASET.md`.

---

## Key outputs reference

| File | Description |
|---|---|
| `data/processed/flood_dataset.parquet` | **The core dataset** — one row per node-day with features, labels, targets, splits. |
| `data/processed/nodes.csv` / `edges.csv` | Graph structure for the GNN. |
| `data/processed/events.csv` | Merged flood episodes (start/end/peak). |
| `data/processed/thresholds.json` | Per-node return-period discharge thresholds. |
| `data/processed/image_manifest.csv` / `image_index.csv` | Requested vs. successfully fetched SAR chips. |
| `data/images/*.npy` | SAR chips (Branch B training data). |
| `docs/validation_report.md` | Label validation vs documented floods + EDA. |

## Notes

- **Resumable:** `01_download.py` and `07_fetch_images_mpc.py` skip work already
  cached — safe to re-run to fill gaps after an interruption or rate-limit.
- **No leakage:** thresholds, climatology, and percentiles are all fit on the
  **train** period (`config.TRAIN_END`) only.
- **Re-run downstream after config changes:** editing `config.py` (dates,
  thresholds, splits) or `nodes.py` means re-running steps 02 → 05 (`run_all.py`).
