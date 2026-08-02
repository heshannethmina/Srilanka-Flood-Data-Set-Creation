# Kaggle metadata — "Sri Lanka Flood Sentinel-1 SAR 2015–2025"

Copy-paste pack for `kaggle.com/datasets/uom230429e/flood-data-set`.
Each section below maps to one field in the Kaggle dataset editor.

Measured facts used here (from the build run):
- 2,578 frames across 9 sites, 2015-01-01 → 2025-12-31
- 5,164 files = 2,578 full-res PNG + 2,578 lite PNG + 8 metadata files
- 1.76 GB, 60 unique columns (34 in `image_dataset.csv`, 26 in `tabular_context.csv`)

---

## 1. Title

```
Sri Lanka Flood Sentinel-1 SAR 2015-2025
```

## 2. Subtitle

```
2,578 co-registered SAR frames over 9 river sites, labelled with GloFAS flood state
```

## 3. Description (About this dataset)

```
# Sri Lanka Flood — Sentinel-1 SAR Image Dataset (2015–2025)

A ready-to-train, label-joined Synthetic Aperture Radar (SAR) image dataset for
flood detection and short-horizon flood forecasting over three flood-prone river
basins in south-western Sri Lanka.

Every Sentinel-1 pass over 9 fixed river locations between 2015-01-01 and
2025-12-31 has been rendered as a fixed-footprint, fixed-resolution 8-bit PNG and
joined to a daily hydrological label derived from GloFAS river discharge
reanalysis. Because the footprint, projection and pixel grid are identical for
every frame at a given site, the frames form a true co-registered time series —
you can stack them, difference them, or feed them to a temporal model without any
further alignment work.

## What's inside

| | |
|---|---|
| Frames | 2,578 PNG (512 × 512 px, 5 km × 5 km footprint, ~10 m/px) |
| Sites | 9, across 3 basins (Kelani, Kalu, Nilwala) |
| Period | 2015-01-01 → 2025-12-31 (Sentinel-1 operational era) |
| Revisit | ~12 days (one dominant descending track over Sri Lanka) |
| Bands | VV, VH (Sentinel-1 RTC gamma-nought, dB) + derived VV−VH ratio |
| Labels | flood_state, flood_within_3d, severity, discharge percentile |
| Size | 1.76 GB (plus a 256 px "lite" copy for quick download) |

## The 9 sites

Each basin is sampled as an upstream → mid → outlet transect, so the dataset
captures a flood wave travelling downstream — useful for graph or sequence models
that reason about routing between stations.

| Site ID | Location | Basin | Position | Lat | Lon |
|---|---|---|---|---|---|
| KEL_HAN | Hanwella | Kelani | mid | 6.902 | 80.082 |
| KEL_KAD | Kaduwela | Kelani | mid | 6.933 | 79.983 |
| KEL_COL | Colombo (Kelani mouth) | Kelani | outlet | 6.970 | 79.873 |
| KAL_RAT | Ratnapura | Kalu | mid | 6.683 | 80.400 |
| KAL_BUL | Bulathsinhala | Kalu | mid | 6.660 | 80.190 |
| KAL_KLT | Kalutara (Kalu mouth) | Kalu | outlet | 6.583 | 79.960 |
| NIL_PIT | Pitabeddara | Nilwala | upstream | 6.300 | 80.552 |
| NIL_AKU | Akuressa | Nilwala | mid | 6.101 | 80.481 |
| NIL_MAT | Matara (Nilwala mouth) | Nilwala | outlet | 5.949 | 80.550 |

## Why SAR

Optical satellites are useless during a Sri Lankan flood — the flood happens
because of the monsoon, and the monsoon means cloud. C-band radar penetrates
cloud and works at night, so it images the flood while the flood is happening.
Open water appears very dark in SAR (specular reflection away from the sensor),
which makes water extraction a largely threshold-based problem; flooded
vegetation appears bright in VH (double-bounce off trunks under standing water).

## PNG encoding — how to get decibels back

The PNGs are quantised, not lossy. The step is 0.137 dB, roughly an order of
magnitude below Sentinel-1's own speckle noise (~1–2 dB), so no usable radiometric
information is lost.

```python
import numpy as np
from PIL import Image

a = np.asarray(Image.open("frames/KEL_HAN/2016-05-22.png")).astype(np.float32)
vv_db    = a[..., 0] / 255 * 35 - 30    # R channel, range -30 .. +5 dB
vh_db    = a[..., 1] / 255 * 35 - 30    # G channel, range -30 .. +5 dB
ratio_db = a[..., 2] / 255 * 20         # B channel, range   0 .. 20 dB (derived)
```

Train on R and G — those are the two real polarisations. B is a derived
VV−VH ratio, provided free for convenience and for visual inspection.

## Labels

Labels come from GloFAS v4 river discharge reanalysis (via the Open-Meteo Flood
API), converted to per-site return-period exceedance:

- `label` (= flood_state) — discharge ≥ the 98th percentile of that site's
  2003–2017 distribution
- `flood_within_3d` — a flood occurs at t+1 … t+3. **Use this one** if you want a
  workable positive rate; the same-day `label` is positive on only ~2% of frames
  because a 12-day revisit rarely coincides with a 1–3 day flood peak.
- `severity` — none / moderate / high / severe (90th / 98th / 99.5th percentile)
- `discharge_pctl` — continuous [0,1] percentile of the training distribution.
  A regression target on this is far better conditioned than binary
  classification given the class imbalance.

**Thresholds were fitted on the 2003–2017 training window only**, never on the
full record, so the labels themselves carry no test-period leakage.

## How to split (please read)

**Do not split randomly.** Neighbouring frames at one site are strongly
autocorrelated and sites within a basin flood together, so a random split leaks
the answer and will hand you a beautiful, meaningless PR-AUC.

Use the provided `split_temporal` column — train ≤ 2017, val 2018–2020,
test 2021+ — or hold out an entire basin for a spatial-generalisation test.

## Known limitations

1. **Revisit ceiling.** Sri Lanka is covered by a single descending Sentinel-1
   track (259 of 263 passes at Hanwella are descending). ~12-day repeat is a hard
   physical limit — daily or 2-day SAR over this region does not exist. Frames are
   snapshots, not a dense time series.
2. **Class imbalance.** Only ~2% of frames land on a flood day. Same-day binary
   classification is the hardest framing available here; prefer `flood_within_3d`,
   `discharge_pctl` regression, or `water_fraction` change detection.
3. **GloFAS grid snapping.** Open-Meteo snaps each coordinate to the nearest
   GloFAS river cell, and for some sites that cell is a tributary rather than the
   main stem — KEL_COL reports 2.2 m³/s at the Kelani mouth, and the Nilwala chain
   reports 0.3–1.8 m³/s. Labels remain internally valid (flood_state is a *per-site*
   percentile, i.e. "unusually high flow for this cell") but at those sites they
   track a tributary rather than the main river. The imagery is unaffected: the
   5 km footprint is centred on the real town and contains the real river.
4. **VH availability.** A small number of scenes are single-polarisation; the
   `has_vh` column flags them and their G channel is filled at the −30 dB floor.

## Provenance and reproducibility

Built end to end by the public notebook
[dataset creation flood](https://www.kaggle.com/code/uom230429e/dataset-creation-flood).
Every input is a free, keyless public API, so the build is fully reproducible:

- **Imagery** — Sentinel-1 RTC (Radiometrically Terrain Corrected γ⁰), queried
  through the Microsoft Planetary Computer STAC API and read as windowed COG
  reads (only the 5 km footprint is downloaded, not whole scenes).
- **Discharge** — GloFAS reanalysis via the Open-Meteo Flood API.
- **Precipitation** — ERA5-Land via the Open-Meteo Archive API.

## Citation

If this dataset is useful, please cite it and credit the underlying providers:
Copernicus Sentinel-1 data (2015–2025), Copernicus Emergency Management Service
(GloFAS), and ECMWF ERA5-Land.
```

## 4. Tags

Pick these from Kaggle's tag picker (all exist):

```
earth and nature, geospatial analysis, remote sensing, satellite imagery,
water bodies, climate change, disaster, computer vision, image, time series analysis,
asia, environment
```

Kaggle's usability score wants ≥ 3 tags; 8–12 is the sweet spot.

## 5. License

```
CC BY 4.0 (Attribution 4.0 International)
```

Correct choice: Copernicus Sentinel data is free and open with an attribution
requirement, and Open-Meteo publishes under CC BY 4.0. Do **not** pick CC0 — you
cannot waive the attribution the Copernicus licence requires.

## 6. Expected update frequency

```
Annually
```

(New Sentinel-1 passes accumulate at ~30/site/year; a yearly rebuild is the
realistic cadence.)

## 7. Coverage

| Field | Value |
|---|---|
| Temporal Coverage Start Date | `2015-01-01` |
| Temporal Coverage End Date | `2025-12-31` |
| Geospatial Coverage | `Sri Lanka — Western, Sabaragamuwa and Southern Provinces (Kelani, Kalu and Nilwala river basins); bounding box 5.92°N–7.00°N, 79.85°E–80.58°E` |

## 8. Provenance

**Sources**

```
Sentinel-1 RTC (gamma-nought) via Microsoft Planetary Computer STAC API — https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
GloFAS v4 river discharge reanalysis via Open-Meteo Flood API — https://open-meteo.com/en/docs/flood-api
ERA5-Land precipitation via Open-Meteo Archive API — https://open-meteo.com/en/docs/historical-weather-api
```

**Collection Methodology**

```
For each of 9 river sites, every Sentinel-1 IW-mode RTC scene intersecting a
±0.02° box around the site between 2015-01-01 and 2025-12-31 was located via the
Planetary Computer STAC API. Where several scenes shared a calendar date, the
dual-polarisation scene was preferred and one frame per date was retained.

For each retained scene, a 5,000 m x 5,000 m window centred on the site was read
directly from the cloud-optimised GeoTIFF in the scene's native UTM projection
and resampled to 512 x 512 px. Linear gamma-nought backscatter was converted to
decibels (10*log10), clipped to [-30, +5] dB and quantised to 8-bit, with VV in
the red channel, VH in green, and the derived VV-VH ratio (clipped to [0, 20] dB)
in blue. Quantisation step is 0.137 dB, below Sentinel-1's speckle noise floor.

Per-frame summary statistics (water_fraction at a -17 dB VV threshold, vv_mean,
vh_mean, valid_fraction) were computed over valid pixels only.

Labels were derived independently: daily GloFAS river discharge for 2003-2025 was
fetched per site, and return-period thresholds (90th / 98th / 99.5th percentile)
were fitted on the 2003-2017 training window only. flood_state marks discharge at
or above the 98th percentile; forward-looking targets at 1 and 3 days and an onset
flag were computed strictly causally within each site. Frames were joined to
labels on a site_id + date key.
```

## 9. Authors

| Field | Value |
|---|---|
| Author Name | `<your full name>` |
| Bio | `Undergraduate researcher, University of Moratuwa. Building a terrain-aware multimodal spatiotemporal graph neural network for probabilistic flood early warning in Sri Lanka.` |

## 10. File descriptions

**`sar_flood_ds/`** — the full-resolution dataset. Train from this one.

**`sar_flood_ds/frames/`** — 2,578 PNG frames in 9 per-site subfolders, named
`frames/<SITE_ID>/<YYYY-MM-DD>.png`. 512 × 512 px, 8-bit RGB. R = VV dB,
G = VH dB, B = VV−VH dB; decoding formulas in `README.txt`. Every frame at a given
site shares an identical 5 km footprint and pixel grid, so the frames are
co-registered and directly stackable.

**`sar_flood_ds/image_dataset.csv`** — the index. One row per frame (2,578 rows,
34 columns): file path, acquisition metadata, SAR summary statistics, the joined
flood labels, and the temporal split assignment. Start here.

**`sar_flood_ds/tabular_context.csv`** — daily hydrology for all 9 sites over
2003-01-01 → 2025-12-31 (~75,000 rows, 26 columns), including days with no
satellite pass. Provides antecedent rainfall and discharge history for the tabular
branch of a multimodal model, and lets you build lookback windows ending on any
frame date.

**`sar_flood_ds/thresholds.json`** — per-site discharge thresholds (m³/s) at the
90th / 98th / 99.5th percentile plus the training-period mean, all fitted on
2003–2017 only. Needed to reproduce or re-derive the labels.

**`sar_flood_ds/README.txt`** — PNG ↔ decibel decoding formulas, label
definitions, and the splitting rule, in plain text next to the data.

**`sar_flood_lite/`** — a 256 × 256 px copy of everything above (~10× smaller).
For quick prototyping, or for downloading to a laptop when the full 1.76 GB is
impractical. Identical CSVs and identical file naming.

## 11. Column descriptors — `image_dataset.csv` (34 columns)

| Column | Description |
|---|---|
| `image_id` | Unique frame identifier, `<site_id>_<date>` |
| `site_id` | Site code, e.g. KEL_HAN. Prefix is the basin: KEL / KAL / NIL |
| `site_name` | Human-readable location name |
| `basin` | River basin: Kelani, Kalu or Nilwala |
| `lat` | Site latitude, decimal degrees (WGS84), frame centre |
| `lon` | Site longitude, decimal degrees (WGS84), frame centre |
| `png_path` | Path to the frame, relative to the dataset root |
| `npy_path` | Path to the float16 dB array; empty in this release |
| `date` | Satellite acquisition date, YYYY-MM-DD (UTC) |
| `scene_id` | Source Sentinel-1 RTC scene identifier on Planetary Computer |
| `platform` | Satellite: sentinel-1a or sentinel-1b |
| `orbit_state` | Orbit direction: ascending or descending (~98% descending) |
| `rel_orbit` | Relative orbit (track) number — constant track means constant geometry |
| `footprint_m` | Ground footprint side length in metres (5000 for all frames) |
| `px` | Image side length in pixels (512 for all frames) |
| `has_vh` | True if the source scene was dual-polarisation. If False, the G channel is filled at the −30 dB floor and `vh_mean` is null |
| `tabular_key` | Join key into `tabular_context.csv` (`<site_id>_<date>`) |
| `label` | **Primary binary label.** 1 if GloFAS discharge that day is at or above the site's 98th-percentile threshold. Positive on ~2% of frames |
| `flood_within_3d` | 1 if a flood occurs on any of t+1 … t+3. **Recommended classification target** — more positives than `label` and a genuine forecasting task |
| `target_flood_1d` | 1 if a flood occurs at t+1 (next-day nowcast target) |
| `target_onset_1d` | 1 if a flood *begins* at t+1 (t is dry, t+1 is flooded). Very rare — for onset-detection studies |
| `severity` | Ordinal class: none / moderate / high / severe (90th / 98th / 99.5th percentile) |
| `label_confidence` | 0.5–1.0. Low near the decision threshold (within ±15%), 1.0 far from it. Use for sample weighting or to exclude ambiguous days |
| `discharge` | GloFAS river discharge, m³/s. Absolute values are not comparable across sites (see limitation 3) |
| `discharge_pctl` | Discharge as a percentile [0,1] of that site's 2003–2017 distribution. **Cross-site comparable — the best regression target in this file** |
| `precipitation_sum` | Same-day ERA5-Land precipitation, mm |
| `precip_sum_3d` | Rolling 3-day precipitation total, mm |
| `precip_sum_7d` | Rolling 7-day precipitation total, mm |
| `api_k090` | Antecedent Precipitation Index, decay k = 0.90. A proxy for catchment wetness / soil saturation |
| `water_fraction` | Share of valid pixels with VV < −17 dB, i.e. the open-water fraction of the footprint. **The main image-derived signal** — a strong weak-supervision and change-detection target |
| `vv_mean` | Mean VV backscatter over valid pixels, dB |
| `vh_mean` | Mean VH backscatter over valid pixels, dB. Null when `has_vh` is False |
| `valid_fraction` | Share of pixels with real data (not the nodata floor). Frames below ~0.9 are partially outside the scene footprint — consider filtering |
| `split_temporal` | Recommended split: train (≤ 2017), val (2018–2020), test (2021+). **Never split randomly** |

## 12. Column descriptors — `tabular_context.csv` (26 columns)

| Column | Description |
|---|---|
| `date` | Calendar date, YYYY-MM-DD. Every day 2003-01-01 → 2025-12-31, pass or no pass |
| `node_id` | Site code, matching `site_id` in `image_dataset.csv` |
| `site_name` | Human-readable location name |
| `basin` | River basin: Kelani, Kalu or Nilwala |
| `lat` | Site latitude, decimal degrees (WGS84) |
| `lon` | Site longitude, decimal degrees (WGS84) |
| `river_discharge` | GloFAS daily mean river discharge, m³/s |
| `precipitation_sum` | ERA5-Land daily precipitation total, mm |
| `flood_moderate` | 1 if discharge ≥ the site's 90th-percentile training threshold |
| `flood_high` | 1 if discharge ≥ the 98th-percentile threshold |
| `flood_severe` | 1 if discharge ≥ the 99.5th-percentile threshold |
| `flood_state` | Alias of `flood_high` — the primary binary flood label |
| `discharge_pctl` | Empirical percentile [0,1] against the 2003–2017 training distribution |
| `label_confidence` | 0.5–1.0; low within ±15% of the primary threshold |
| `target_flood_1d` | Flood occurs at t+1 |
| `target_flood_2d` | Flood occurs on any of t+1 … t+2 |
| `target_flood_3d` | Flood occurs on any of t+1 … t+3 |
| `target_onset_1d` | Flood begins at t+1 (t dry, t+1 flooded) |
| `precip_sum_3d` | Rolling 3-day precipitation total, mm |
| `precip_sum_7d` | Rolling 7-day precipitation total, mm |
| `precip_sum_15d` | Rolling 15-day precipitation total, mm |
| `precip_sum_30d` | Rolling 30-day precipitation total, mm |
| `api_k090` | Antecedent Precipitation Index, decay k = 0.90 |
| `severity` | Ordinal class: none / moderate / high / severe |
| `split_temporal` | train (≤ 2017) / val (2018–2020) / test (2021+) |
| `tabular_key` | Join key to `image_dataset.csv` (`<node_id>_<date>`) |

All rolling and forward-looking columns are computed **within a single site**, so
no value ever mixes data across sites.

## 13. Starter code to paste into the description or a linked notebook

```python
import pandas as pd, numpy as np
from PIL import Image

ROOT = "/kaggle/input/flood-data-set/sar_flood_ds"
df = pd.read_csv(f"{ROOT}/image_dataset.csv")

train = df[df.split_temporal == "train"]
val   = df[df.split_temporal == "val"]
test  = df[df.split_temporal == "test"]

def load_vv_vh(row):
    a = np.asarray(Image.open(f"{ROOT}/{row.png_path}")).astype(np.float32)
    return a[..., 0] / 255 * 35 - 30, a[..., 1] / 255 * 35 - 30

vv, vh = load_vv_vh(df.iloc[0])
print(vv.shape, vv.min().round(1), vv.mean().round(1), vv.max().round(1))
```
