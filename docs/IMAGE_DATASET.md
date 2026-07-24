# Branch B — Sentinel-1 SAR Image Dataset

The vision-branch dataset. Each sample is a small **Sentinel-1 radar chip** of a
river node, **labeled automatically** by the discharge-derived flood labels from
the tabular dataset (Branch A). SAR is used because radar sees through the
monsoon cloud cover that blinds optical sensors during floods.

## Why it links cleanly to the tabular dataset

Every chip carries a `tabular_key = node_id + "_" + date`. That key matches a row
in `flood_dataset.parquet`, so the image and tabular samples align **1:1** for a
fusion model — no separate labelling effort was needed. The tabular dataset is
the *label engine* for the imagery.

## Source

- **Microsoft Planetary Computer** STAC, collection **`sentinel-1-rtc`**
  (Radiometric Terrain Corrected, γ⁰, UTM COGs) — accessed anonymously (no account).
- Bands: **VV** and **VH** backscatter, converted to **dB** and clipped to
  `[-30, 5]` dB (nodata → −30).
- Coverage: Sentinel-1 operational from **Oct 2014**, so image samples span
  2015–2024 (a subset of the tabular record).

## Files

| File | Contents |
|---|---|
| `data/processed/image_manifest.csv` | 3,489 chip **requests** (what to fetch, with labels) — produced by `06_image_manifest.py`. |
| `data/images/<sample_id>.npy` | fetched chip, `float16` array `[2, 128, 128]` = (VV, VH) dB. |
| `data/processed/image_index.csv` | one row per **successfully fetched** chip (the actual image dataset). |

### `image_manifest.csv` / `image_index.csv` columns
| Column | Description |
|---|---|
| `sample_id` | Chip id (`S000123`), also the `.npy` filename. |
| `node_id`, `lat`, `lon`, `basin` | Location. |
| `target_date` | Requested date; `actual_date` (index) is the real satellite pass used. |
| `label` | 1 = flood, 0 = dry. |
| `severity` | none / moderate / high / severe (from discharge return period). |
| `purpose` | `flood_peak` (at discharge peak), `pre_event` (image ~3 d BEFORE onset — the "predict before it floods" case), `dry_negative`. |
| `event_id` | Links to `events.csv` for the flood episode. |
| `tabular_key` | Join key to `flood_dataset.parquet` (`node_id_date`). |
| `scene_id` | Sentinel-1 scene actually used (index only). |

## Sample composition (manifest)

- **1,398 positive** (699 flood-peak + 699 pre-event) · **2,091 dry negatives**.
- Positive severity: ~956 high, ~442 severe.

## Fetch it

```bash
pip install pystac-client planetary-computer rasterio pyproj
python scripts/07_fetch_images_mpc.py            # full manifest (resumable, cached)

# knobs (env vars):
MAX_CHIPS=600        # only the first N manifest rows
ONLY_LABEL=0         # fetch only dry (0) or only flood (1) chips
SHUFFLE=1            # random balanced subset instead of manifest order
IMG_WORKERS=3        # keep <=3 — MPC throttles anonymous signing above that
```
- **Resumable**: existing `.npy` chips are skipped; re-run to fill gaps.
- `no_scene` rows = no Sentinel-1 pass within ±12 days (rare, sparse-coverage nodes/dates).
- The manifest lists **all positives first, then negatives** — for a balanced
  subset use `SHUFFLE=1`, or fetch each class with `ONLY_LABEL=1` / `ONLY_LABEL=0`.

## Load it (pairs image + tabular for a fusion model)

```bash
python scripts/image_loader.py     # sanity check
```
`image_loader.load_pairs()` returns, per fetched chip: the `[2,128,128]` SAR
tensor, its label/severity, **and** the matching tabular feature row — ready to
feed a two-branch (GNN + CNN) fusion network.

## Honest notes

- **Lead time comes from Branch A (rainfall/discharge)**, not the images:
  Sentinel-1 revisits ~12 days, so it cannot refresh hourly. Images add
  *current inundation state*, *river morphology*, and sharpen *severity* — the
  `pre_event` chips let the model associate pre-flood radar signatures with
  imminent floods, but the hours-ahead signal is hydro-meteorological.
- SAR backscatter over water is low (specular reflection); flooded vegetation can
  appear bright (double-bounce). A water-fraction feature per chip is a good
  future addition (thresholding VV dB).
- For a truly *live* map, pair this with geostationary (Himawari/INSAT) cloud/rain
  nowcasts or river CCTV — see the fusion design discussion.
