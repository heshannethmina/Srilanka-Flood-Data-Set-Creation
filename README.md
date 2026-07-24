# Sri Lanka Multimodal Spatiotemporal Flood Dataset (v1)

A **reproducible, event-based, leakage-aware** dataset for probabilistic flood
early-warning research over Sri Lanka's major river basins, built entirely from
**public, no-API-key** reanalysis sources.

> **Title framing (honest):** The current build is *terrain- and hydrology-aware*.
> It uses **GloFAS river discharge** as both a feature and the basis for flood
> labels, so it is physically grounded. It is **not** a fully "physics-guided"
> model dataset in the strict sense (no assimilated gauge/discharge from the
> Irrigation Department). Recommended title:
> **"Terrain-Aware Multimodal Spatiotemporal Graph Neural Network for
> Probabilistic Flood Early Warning in Sri Lanka."**

---

## What this is

- **Spatial units (nodes):** 51 monitoring points across 16 flood-prone basins
  (Kelani, Kalu, Gin, Nilwala, Attanagalu Oya, Maha Oya, Deduru Oya, Mahaweli,
  Gal Oya, Batticaloa, Walawe, Kirindi Oya, Malwathu Oya, Kala Oya, Yan Oya, Colombo metro).
- **Time:** daily, **2003-01-01 → 2024-12-31** (8,036 days/node).
- **Scale:** **409,836 node-day rows**; **408,255 valid supervised samples**.
- **Graph:** directed upstream→downstream *flow* edges + spatial *k-NN* edges.
- **Targets:** flood within **24h / 48h / 72h** (classification), flood onset,
  next-day discharge and 3-day max discharge z-score (regression).

## Data sources (all public, no key)

| Modality | Source | Backend |
|---|---|---|
| Rainfall, temperature, humidity, wind, radiation, **soil wetness** | **NASA POWER** daily API | MERRA-2 / GEOS assimilation |
| River discharge (flood signal) | **Open-Meteo Flood API** | Copernicus **GloFAS** reanalysis |
| Elevation | NASA POWER grid metadata | POWER/SRTM grid |
| Terrain topology | Hand-built basin graph (`scripts/nodes.py`) | — |

See [`docs/SOURCES_AND_LICENSE.md`](docs/SOURCES_AND_LICENSE.md).

## How the labels work (physically grounded, not synthetic)

Flood state at a node/day = **GloFAS river discharge ≥ a per-node
return-period threshold**, where thresholds are the 90th / 98th / 99.5th
discharge percentiles estimated **on the training period only** (moderate /
high / severe). This is the same return-period-exceedance definition used by
operational systems such as GloFAS. Labels are validated against **10
documented historical flood episodes** (2003–2024) in
[`docs/validation_report.md`](docs/validation_report.md).

## Leakage control (the core scientific safeguard)

Pixels/days from the same flood are highly correlated, so we **never** split
randomly. Three split schemes are provided as columns:

- `split_temporal` — train ≤ 2017 · val 2018–2020 · test 2021–2024 (unseen recent events).
- `split_basin_holdout` — the entire **Gin** basin is held out for spatial generalization.
- `event_id` — merged flood episodes for GroupKFold / event-level CV.

Climatologies, thresholds and anomalies are all fit on the **train period only**.

## Getting the data

To keep the repository lightweight, the **heavy generated files are not committed**
(the full `flood_dataset.csv` alone is ~250 MB, plus ~70 MB of SAR chips). The repo
ships the **code, docs, graph structure, labels, thresholds and image index** — everything
needed to rebuild the full dataset locally, or to inspect the structure directly.

- **Tracked (small):** `nodes.csv`, `edges.csv`, `events.csv`, `thresholds.json`,
  `image_index.csv`, `image_manifest.csv`, `sample_preview.csv`.
- **Rebuild locally (large):** run the pipeline below to regenerate
  `flood_dataset.parquet`, the compact/full CSVs, and fetch the Sentinel-1 chips.
- **Prebuilt archive:** the full parquet + chip archive is distributed via the repo's
  **GitHub Releases** (or a Zenodo DOI) — see the Releases tab.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/01_download.py        # caches raw data (resumable, ~public API)
python scripts/run_all.py            # features + labels + graph + validation
```

Outputs land in `data/processed/`:

| File | Contents |
|---|---|
| `flood_dataset.parquet` | **main** analysis-ready panel (one row per node-day) |
| `nodes.csv`, `edges.csv` | graph structure |
| `events.csv` | flood episodes (event-level CV) |
| `thresholds.json` | per-node discharge return-period thresholds |
| `nodes_meta.csv` | snapped coordinates + elevation |

See [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) and
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for full column definitions.

## Branch B — Sentinel-1 SAR imagery (multimodal extension)

An optional **vision branch**: Sentinel-1 radar chips per node, auto-labeled by
the discharge-based flood labels (they share a `tabular_key`, so image and tabular
samples align 1:1 for a fusion model). Fetched anonymously from Microsoft
Planetary Computer — no account needed.

```bash
python scripts/06_image_manifest.py        # build chip request list (3,489 chips)
python scripts/07_fetch_images_mpc.py      # fetch Sentinel-1 chips (resumable)
python scripts/image_loader.py             # pair chips with tabular features
```
Full details: [docs/IMAGE_DATASET.md](docs/IMAGE_DATASET.md).

## Use it in PyTorch

```bash
python scripts/torch_loader.py           # sanity check
# build_torch_geometric() returns X[T,N,F], y[T,N], edge_index, edge_weight
```

## Limitations

- Discharge and weather are **reanalysis / assimilation** products (GloFAS,
  NASA POWER/MERRA-2), not raw gauges; GloFAS discharge is skillful for
  large/medium basins but less so for small, flashy, or heavily regulated
  (reservoir) catchments.
- NASA POWER meteorology is on a ~0.5° grid, so nearby lowland/upland nodes can
  share a grid cell; discharge (GloFAS, ~0.05°) resolves the basins more finely.
- Labels are a **discharge-exceedance proxy** for flooding, not surveyed
  inundation extent. For inundation-extent labels, fuse Sentinel-1 SAR change
  detection (see the roadmap in the dataset card).
