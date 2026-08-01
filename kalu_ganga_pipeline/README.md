# Kalu Ganga Flood Data Pipeline

A **self-contained** pipeline to collect tabular and Sentinel-1 SAR image flood data for the **Kalu Ganga** river basin, Sri Lanka.

## Nodes covered (5 monitoring points)

| Node ID    | Location           | Position   | Lat     | Lon    |
|------------|--------------------|------------|---------|--------|
| `KAL_BAL`  | Balangoda          | upstream   | 6.652   | 80.698 |
| `KAL_RAT`  | Ratnapura          | mid        | 6.683   | 80.400 |
| `KAL_BUL`  | Bulathsinhala      | mid        | 6.660   | 80.190 |
| `KAL_AGA`  | Agalawatta         | mid        | 6.535   | 80.150 |
| `KAL_KLT`  | Kalutara (mouth)   | outlet     | 6.583   | 79.960 |

River flows: **Balangoda → Ratnapura → Bulathsinhala → Agalawatta → Kalutara**

---

## Quick Start

```bash
cd kalu_ganga_pipeline

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw data (slow, resumable)
python 01_download.py

# 3. Build tabular dataset + graph + validation (fast)
python run_tabular.py

# 4. Plan satellite image requests (fast)
python 06_image_manifest.py

# 5. Fetch Sentinel-1 SAR chips (slow, resumable)
python 07_fetch_images.py
```

---

## File Structure

```
kalu_ganga_pipeline/
├── config.py              Central settings (dates, thresholds, splits, API URLs)
├── nodes.py               5 Kalu Ganga monitoring nodes + river topology
├── requirements.txt       Python dependencies
│
├── 01_download.py         Download weather, discharge, precipitation (free APIs)
├── 02_features.py         Engineer tabular features per node-day
├── 03_labels_splits.py    Flood labels, events, train/val/test splits
├── 04_graph.py            Build river-flow + spatial kNN graph
├── 05_validate.py         Validate labels vs documented flood events
├── run_tabular.py         Run steps 02-05 in order (convenience runner)
│
├── 06_image_manifest.py   Plan which SAR chips to fetch
├── 07_fetch_images.py     Fetch Sentinel-1 SAR chips from MS Planetary Computer
│
├── image_loader.py        Pair SAR chips with tabular features (NumPy arrays)
│
└── data/
    ├── raw/               Cached API downloads per node
    └── processed/         Analysis-ready outputs (see below)
```

---

## Outputs

### Tabular (Branch A)

| File | Format | Description |
|------|--------|-------------|
| `data/raw/<node>_weather.parquet` | Parquet | NASA POWER daily weather + soil wetness |
| `data/raw/<node>_flood.parquet`   | Parquet | GloFAS river discharge |
| `data/raw/<node>_precip.parquet`  | Parquet | ERA5 high-res precipitation |
| `data/processed/panel_features.parquet` | Parquet | Engineered features, all nodes |
| `data/processed/thresholds.json`  | JSON   | Per-node flood discharge thresholds |
| `data/processed/flood_dataset.parquet` | Parquet | **Core dataset** (node × day) |
| `data/processed/events.csv`       | CSV    | Merged flood episodes |
| `data/processed/nodes.csv`        | CSV    | Node metadata |
| `data/processed/edges.csv`        | CSV    | Graph edges (flow + spatial kNN) |
| `data/processed/validation_report.md` | Markdown | Label validation vs real floods |

### Imagery (Branch B)

| File | Format | Description |
|------|--------|-------------|
| `data/processed/image_manifest.csv` | CSV | Chip requests (what to fetch) |
| `data/processed/image_index.csv`    | CSV | Successfully fetched chips index |
| `data/images/<id>.npy`             | NumPy | SAR chip — float16 `[2, 128, 128]` |

### SAR Chip Format

```
Shape   : [2, 128, 128]   float16
Channel 0 : VV polarization (dB, clipped to [-30, +5])
Channel 1 : VH polarization (dB, clipped to [-30, +5])
Footprint : ~2.56 km × 2.56 km window around the river node
```

---

## Optional Environment Knobs (Step 7)

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_CHIPS` | (all) | Fetch only first N chips |
| `ONLY_LABEL` | (both) | `0` = dry only, `1` = flood only |
| `SHUFFLE` | `0` | Set `1` to shuffle manifest first |
| `IMG_WORKERS` | `3` | Parallel download threads |

```bash
# Quick smoke-test (10 chips only)
MAX_CHIPS=10 python 07_fetch_images.py
```

## View the images

After `07_fetch_images.py` has created `.npy` chips in `data/images/`, use
the viewer scripts in `viz/`:

```bash
pip install -r requirements.txt
python viz/preview_chips.py
python viz/08_export_png.py
```

- `viz/preview_chips.py` writes small contact-sheet previews to `data/preview/`.
- `viz/08_export_png.py` exports every `.npy` chip to viewable PNGs in `data/png_images/`.
- The `.npy` files are the real radar data; PNGs are for inspection only.

---

## Data Sources (all free, no API key)

| Source | What |
|--------|------|
| NASA POWER | Daily weather, soil wetness (2003-2025) |
| Open-Meteo Archive (ERA5) | High-res precipitation |
| Open-Meteo Flood (GloFAS) | River discharge |
| Microsoft Planetary Computer | Sentinel-1 RTC SAR imagery (2015-2025) |

---

## Notes

- **Resumable:** Steps 01 and 07 skip already-cached data — safe to interrupt and restart.
- **No leakage:** Flood thresholds, climatology, and percentiles are all fitted on the **train** period (`TRAIN_END = 2017-12-31`) only.
- **Run from this folder:** All scripts use `os.path.dirname(__file__)` as root, so always run as `python <script>.py` from inside `kalu_ganga_pipeline/`.
