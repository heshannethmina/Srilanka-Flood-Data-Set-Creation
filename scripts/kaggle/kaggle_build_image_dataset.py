"""
KAGGLE NOTEBOOK — build the Sri Lanka Sentinel-1 SAR flood image dataset (9 sites).

Why Kaggle: the fetch is network-bound (~2,300 frames x 2 remote COG reads).
Kaggle has fast internet; your PC does not. This script is SELF-CONTAINED — it
needs NO upload from your machine. It re-derives the flood labels from the same
public APIs the repo uses, so labels match `flood_dataset.parquet` exactly.

HOW TO USE
  1. New Notebook on kaggle.com.
  2. Settings -> Internet: ON  (needs phone verification, one time).
  3. Settings -> Accelerator: None (this is I/O bound, GPU is wasted).
  4. Paste this whole file into ONE cell (or split on the `# %%` markers).
  5. Run. Then "Save Version" -> "Save & Run All (Commit)" for the full 12 h budget.
  6. Output -> "New Dataset" so you can train from it without ever downloading.

RESUMING ACROSS SESSIONS (12 h limit)
  Add your previous run's output as an input dataset, then set
  RESUME_FROM = "/kaggle/input/<your-dataset-slug>". Existing frames are copied
  in and skipped. The fetch is fully resumable.

OUTPUT (in /kaggle/working/sar_flood_ds/)
  frames/<SITE>/<YYYY-MM-DD>.png   RGB PNG, 8-bit  <- the training data
  image_dataset.csv                one row per frame: labels, splits, SAR features
  tabular_context.csv              daily discharge/precip/labels per site (fusion branch)
  thresholds.json                  per-site return-period discharge thresholds
  README.txt                       PNG <-> dB decoding formulas
  sar_flood_lite.zip               optional 256 px copy, small enough to download

PNG ENCODING (lossless enough: 0.137 dB/step, far below SAR speckle ~1-2 dB)
  R = VV  dB in [-30, +5]   ->  u8 ;  dB = R/255*35 - 30
  G = VH  dB in [-30, +5]   ->  u8 ;  dB = G/255*35 - 30
  B = VV-VH  dB in [0, 20]  ->  u8 ;  dB = B/255*20
Train on R,G (the two real polarisations). B is a derived ratio channel, free
for the model to use and handy for visual inspection.
"""

# %% ---------------------------------------------------------------- install
# Kaggle images ship rasterio/pyproj already; the STAC clients are the additions.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pystac-client", "planetary-computer", "rasterio", "pyproj"], check=True)

# %% ---------------------------------------------------------------- config
import os, io, json, time, math, shutil, warnings
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

warnings.filterwarnings("ignore")

# --- the 9 sites -------------------------------------------------------------
# All are existing nodes from scripts/common/nodes.py — the node_id MUST match,
# or the label join produces nulls. Three basins x a downstream transect each
# (upstream -> mid -> outlet), so the set shows a flood wave travelling down a
# river, which is exactly the story the directed flow edges encode.
#
# mean_q / thr_high are the GloFAS values measured from flood_dataset.parquet;
# flood_days = days at flood_state in 2015-2025 (the Sentinel-1 era).
SITES = [
    # id         name                      basin      lat     lon      pos       mean_q  thr_high  flood_days
    ("KEL_HAN", "Hanwella",               "Kelani",   6.902, 80.082),  # mid       8.0     21.8      78
    ("KEL_KAD", "Kaduwela",               "Kelani",   6.933, 79.983),  # mid     163.5    505.8      83
    ("KEL_COL", "Colombo (Kelani mouth)", "Kelani",   6.970, 79.873),  # outlet    2.2      8.6      80
    ("KAL_RAT", "Ratnapura",              "Kalu",     6.683, 80.400),  # mid      46.9    157.5      48
    ("KAL_BUL", "Bulathsinhala",          "Kalu",     6.660, 80.190),  # mid       2.3      6.7      59
    ("KAL_KLT", "Kalutara (Kalu mouth)",  "Kalu",     6.583, 79.960),  # outlet  208.6    632.1      71
    ("NIL_PIT", "Pitabeddara (upper)",    "Nilwala",  6.300, 80.552),  # upstream  1.8      8.7      67
    ("NIL_AKU", "Akuressa",               "Nilwala",  6.101, 80.481),  # mid       1.1      3.8     111
    ("NIL_MAT", "Matara (Nilwala mouth)", "Nilwala",  5.949, 80.550),  # outlet    0.3      1.4      78
]
# KNOWN ISSUE — GloFAS grid snapping. Open-Meteo snaps each coordinate to the
# nearest GloFAS river cell, and for some nodes that cell is a small tributary
# rather than the main stem: KEL_COL (2.2 m3/s at the Kelani mouth) and the whole
# Nilwala chain (0.3-1.8 m3/s) are far below the real channel flow. The labels
# stay internally valid — flood_state is a PER-NODE percentile, i.e. "unusually
# high flow for this cell" — but at those nodes it tracks a tributary, not the
# main river. This is the most likely reason the 2003/2014/2017 south-western
# events are missed in docs/validation_report.md. The SAR imagery is unaffected:
# the 5 km footprint is centred on the real town and contains the real river.

START, END = "2015-01-01", "2025-12-31"   # Sentinel-1 operational from Oct 2014
TAB_START = "2003-01-01"                  # labels need the full record for thresholds
TRAIN_END, VAL_END = "2017-12-31", "2020-12-31"

WIN_M   = 5000.0     # footprint side, metres
OUT_PX  = 512        # pixels per side (~10 m/px native at 5 km)
WATER_DB = -17.0     # VV below this counts as open water
WORKERS = 8          # Kaggle handles more than a home line; MPC throttles above ~12
DB_LO, DB_HI = -30.0, 5.0
SEV_PCTL = {"moderate": 0.90, "high": 0.98, "severe": 0.995}

SAVE_NPY  = False    # also write float16 [2,PX,PX] .npy (4x the disk, exact dB)
MAKE_LITE = True     # also emit a 256 px zip you can actually download
LITE_PX   = 256

OUT   = "/kaggle/working/sar_flood_ds"
FRAMES = os.path.join(OUT, "frames")
RESUME_FROM = None   # e.g. "/kaggle/input/srilanka-sar-flood-v1" to continue a previous run
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

os.makedirs(FRAMES, exist_ok=True)

# GDAL tuning for remote COG reads — without these every read re-lists the bucket.
# DO NOT set CPL_VSIL_CURL_ALLOWED_EXTENSIONS here: Planetary Computer signed URLs
# end in "?st=...&sig=..." rather than ".tif", so an extension allow-list makes
# GDAL refuse every single asset (symptom: ok=0, every frame errors on /vsicurl/).
os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "100000000",
})

# carry over anything already fetched in a previous session
if RESUME_FROM and os.path.isdir(os.path.join(RESUME_FROM, "frames")):
    print("Resuming from", RESUME_FROM)
    for site_id, *_ in SITES:
        src = os.path.join(RESUME_FROM, "frames", site_id)
        if os.path.isdir(src):
            dst = os.path.join(FRAMES, site_id)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if not os.path.exists(os.path.join(dst, f)):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("  carried over:",
          sum(len(os.listdir(os.path.join(FRAMES, d))) for d in os.listdir(FRAMES)), "files")


# %% ------------------------------------------- STEP 1: labels (Branch A, mini)
# Rebuilds the label engine for these 9 sites only: GloFAS discharge -> per-node
# return-period thresholds fitted on TRAIN YEARS ONLY -> flood state + targets.
FLOOD_URL   = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get(url, params, tries=10):
    last = None
    for a in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            if "Hourly API request limit" in r.text:
                time.sleep(95); continue
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(3 * (a + 1), 45))
    raise RuntimeError(f"API failed: {last}")


def build_tabular():
    frames, thresholds = [], {}
    for site_id, name, basin, lat, lon in SITES:
        q = pd.DataFrame(_get(FLOOD_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": TAB_START, "end_date": END,
            "daily": "river_discharge", "timezone": "UTC"})["daily"]
        ).rename(columns={"time": "date"})
        p = pd.DataFrame(_get(ARCHIVE_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": TAB_START, "end_date": END,
            "daily": "precipitation_sum", "timezone": "UTC"})["daily"]
        ).rename(columns={"time": "date"})

        d = q.merge(p, on="date", how="left")
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").reset_index(drop=True)
        d["river_discharge"] = pd.to_numeric(d["river_discharge"], errors="coerce")
        d["precipitation_sum"] = pd.to_numeric(d["precipitation_sum"], errors="coerce")

        # --- thresholds from the TRAIN period only (no leakage) ---
        tm = d["date"] <= pd.Timestamp(TRAIN_END)
        qt = d.loc[tm, "river_discharge"].dropna()
        thr = {s: float(qt.quantile(pc)) for s, pc in SEV_PCTL.items()}
        thr["q_mean_train"] = float(qt.mean())
        thresholds[site_id] = thr

        for s in SEV_PCTL:
            d[f"flood_{s}"] = (d["river_discharge"] >= thr[s]).astype("int8")
        d["flood_state"] = d["flood_high"]

        # empirical percentile against the train distribution
        qs = np.sort(qt.to_numpy())
        d["discharge_pctl"] = np.searchsorted(
            qs, d["river_discharge"].to_numpy(), side="right") / max(len(qs), 1)

        # label confidence: 0.5 inside a +/-15% band around the primary threshold
        rel = (d["river_discharge"] - thr["high"]).abs() / max(thr["high"], 1e-9)
        d["label_confidence"] = np.where(
            d["river_discharge"].isna(), 0.0, 0.5 + 0.5 * np.clip(rel / 0.15, 0, 1))

        # forward-looking targets (strictly causal, within this site only)
        st = d["flood_state"].to_numpy()
        n = len(st)
        for h in (1, 2, 3):
            fut = np.full(n, np.nan)
            for i in range(n - 1):
                fut[i] = st[i + 1: min(i + h, n - 1) + 1].max()
            d[f"target_flood_{h}d"] = fut
        d["target_onset_1d"] = np.where(
            (d["target_flood_1d"] == 1) & (st == 0), 1.0,
            np.where(d["target_flood_1d"].isna(), np.nan, 0.0))

        # antecedent rainfall (a few useful ones for a fusion model)
        pr = d["precipitation_sum"].fillna(0.0)
        for w in (3, 7, 15, 30):
            d[f"precip_sum_{w}d"] = pr.rolling(w, min_periods=1).sum()
        acc, api = 0.0, np.zeros(len(pr))
        for i, v in enumerate(pr.to_numpy()):
            acc = acc * 0.90 + v
            api[i] = acc
        d["api_k090"] = api

        d["severity"] = np.select(
            [d["flood_severe"] == 1, d["flood_high"] == 1, d["flood_moderate"] == 1],
            ["severe", "high", "moderate"], default="none")
        d["split_temporal"] = np.select(
            [d["date"] <= pd.Timestamp(TRAIN_END), d["date"] <= pd.Timestamp(VAL_END)],
            ["train", "val"], default="test")
        d["node_id"], d["site_name"], d["basin"] = site_id, name, basin
        d["lat"], d["lon"] = lat, lon
        d["tabular_key"] = site_id + "_" + d["date"].dt.strftime("%Y-%m-%d")
        frames.append(d)
        print(f"  {site_id}: {len(d)} days | thr_high={thr['high']:.1f} m3/s | "
              f"flood days {int(d.flood_state.sum())}")
        time.sleep(1.0)

    tab = pd.concat(frames, ignore_index=True)
    tab.to_csv(os.path.join(OUT, "tabular_context.csv"), index=False)
    with open(os.path.join(OUT, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    return tab


print("=== STEP 1: labels from GloFAS + ERA5-Land ===")
TAB = build_tabular()
print(f"tabular_context.csv: {len(TAB)} node-days\n")


# %% ------------------------------------------- STEP 2: find every Sentinel-1 pass
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

# No sign_inplace here: signed hrefs expire (~30 min) and a full run is longer.
# We sign each asset lazily, right before reading it.
CATALOG = pystac_client.Client.open(STAC_URL)


def search_year(lon, lat, y0, y1, tries=5):
    d = 0.02
    for a in range(tries):
        try:
            s = CATALOG.search(collections=["sentinel-1-rtc"],
                               bbox=[lon - d, lat - d, lon + d, lat + d],
                               datetime=f"{y0}/{y1}",
                               query={"sar:instrument_mode": {"eq": "IW"}})
            return [it for it in s.items() if "vv" in it.assets]
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))
    return []


def list_passes(lon, lat):
    """One frame per calendar date; prefer the dual-pol (VH-bearing) scene."""
    items = []
    for yr in range(pd.Timestamp(START).year, pd.Timestamp(END).year + 1):
        y0 = max(pd.Timestamp(START), pd.Timestamp(f"{yr}-01-01")).strftime("%Y-%m-%d")
        y1 = min(pd.Timestamp(END),   pd.Timestamp(f"{yr}-12-31")).strftime("%Y-%m-%d")
        items.extend(search_year(lon, lat, y0, y1))
    items.sort(key=lambda it: (pd.Timestamp(it.datetime), "vh" not in it.assets))
    seen, out = set(), []
    for it in items:
        day = pd.Timestamp(it.datetime).strftime("%Y-%m-%d")
        if day not in seen:
            seen.add(day)
            out.append((day, it))
    return out


print("=== STEP 2: STAC search ===")
TASKS = []
for site_id, name, basin, lat, lon in SITES:
    os.makedirs(os.path.join(FRAMES, site_id), exist_ok=True)
    passes = list_passes(lon, lat)
    print(f"  {site_id}: {len(passes)} Sentinel-1 passes")
    for day, item in passes:
        TASKS.append((site_id, name, basin, lat, lon, day, item))
print(f"total {len(TASKS)} frames to build\n")


# %% ------------------------------------------- STEP 3: fetch pixels -> PNG
def to_db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(x <= 0, np.nan, x))
    return np.clip(np.where(np.isfinite(db), db, DB_LO), DB_LO, DB_HI)


def read_band(href, lon, lat, px):
    with rasterio.open(pc.sign(href)) as ds:      # sign lazily -> token never stale
        tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        half = WIN_M / 2.0
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=0, out_shape=(px, px))
    return to_db(arr.astype("float32"))


def q8(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1).__mul__(255).round().astype(np.uint8)


def sar_features(vv, vh, has_vh):
    valid = vv > (DB_LO + 0.1)                    # exclude the nodata floor
    if valid.sum() == 0:
        return dict(water_fraction=np.nan, vv_mean=np.nan, vh_mean=np.nan, valid_fraction=0.0)
    v = vv[valid]
    return dict(water_fraction=round(float((v < WATER_DB).mean()), 4),
                vv_mean=round(float(v.mean()), 2),
                vh_mean=round(float(vh[valid].mean()), 2) if has_vh else np.nan,
                valid_fraction=round(float(valid.mean()), 4))


def build_frame(t):
    site_id, name, basin, lat, lon, day, item = t
    sdir = os.path.join(FRAMES, site_id)
    png = os.path.join(sdir, f"{day}.png")
    npy = os.path.join(sdir, f"{day}.npy")
    has_vh = "vh" in item.assets

    if os.path.exists(png):                        # resumable
        a = np.asarray(Image.open(png)).astype(np.float32)
        vv = a[..., 0] / 255 * 35 - 30
        vh = a[..., 1] / 255 * 35 - 30
    else:
        last = None
        for a in range(4):
            try:
                vv = read_band(item.assets["vv"].href, lon, lat, OUT_PX)
                vh = read_band(item.assets["vh"].href, lon, lat, OUT_PX) if has_vh \
                    else np.full_like(vv, DB_LO)
                rgb = np.dstack([q8(vv, DB_LO, DB_HI), q8(vh, DB_LO, DB_HI),
                                 q8(vv - vh, 0.0, 20.0)])
                Image.fromarray(rgb).save(png, optimize=True)
                if SAVE_NPY:
                    np.save(npy, np.stack([vv, vh]).astype("float16"))
                break
            except Exception as e:
                last = f"{type(e).__name__}: {e}"      # full message, never truncated
                time.sleep(2 * (a + 1))
        else:
            print(f"    ! {site_id} {day}: {last}", flush=True)
            return None

    p = item.properties
    row = dict(image_id=f"{site_id}_{day}", site_id=site_id, site_name=name, basin=basin,
               lat=lat, lon=lon,
               png_path=f"frames/{site_id}/{day}.png",
               npy_path=f"frames/{site_id}/{day}.npy" if SAVE_NPY else "",
               date=day, scene_id=item.id, platform=p.get("platform"),
               orbit_state=p.get("sat:orbit_state"), rel_orbit=p.get("sat:relative_orbit"),
               footprint_m=int(WIN_M), px=OUT_PX, has_vh=has_vh,
               tabular_key=f"{site_id}_{day}")
    row.update(sar_features(vv, vh, has_vh))
    return row


# ---- SMOKE TEST: prove one read works before burning two hours on 2,500 -------
# If this raises, the full run would have produced ok=0. Fix here, not after.
print("=== SMOKE TEST: one frame ===")
_sid, _n, _b, _la, _lo, _day, _item = TASKS[0]
try:
    _vv = read_band(_item.assets["vv"].href, _lo, _la, 64)
    print(f"  OK {_sid} {_day}  shape={_vv.shape}  "
          f"dB min/mean/max = {_vv.min():.1f}/{_vv.mean():.1f}/{_vv.max():.1f}")
except Exception as e:
    print(f"  FAILED -> {type(e).__name__}: {e}")
    print("  trying the whole-item signing fallback ...")
    _signed = pc.sign(_item)
    _vv = read_band(_signed.assets["vv"].href, _lo, _la, 64)
    print(f"  fallback OK, shape={_vv.shape} — set SIGN_WHOLE_ITEM = True")
    raise SystemExit("Stop here and switch to the fallback before the full run.")

print("=== STEP 3: fetching frames ===")
rows, done = [], 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(build_frame, t) for t in TASKS]
    for fut in as_completed(futs):
        r = fut.result()
        done += 1
        if r:
            rows.append(r)
        if done % 50 == 0 or done == len(TASKS):
            el = time.time() - t0
            eta = el / done * (len(TASKS) - done)
            print(f"    {done}/{len(TASKS)}  ok={len(rows)}  "
                  f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
print(f"built {len(rows)} frames in {(time.time()-t0)/60:.1f} min\n")


# %% ------------------------------------------- STEP 4: join labels -> index CSV
img = pd.DataFrame(rows).sort_values(["site_id", "date"]).reset_index(drop=True)

keep = TAB[["tabular_key", "flood_state", "target_flood_1d", "target_flood_3d",
            "target_onset_1d", "severity", "river_discharge", "discharge_pctl",
            "label_confidence", "precipitation_sum", "precip_sum_3d", "precip_sum_7d",
            "api_k090", "split_temporal"]].rename(
    columns={"flood_state": "label", "target_flood_3d": "flood_within_3d",
             "river_discharge": "discharge"})
img = img.merge(keep, on="tabular_key", how="left")

COLS = ["image_id", "site_id", "site_name", "basin", "lat", "lon", "png_path", "npy_path",
        "date", "scene_id", "platform", "orbit_state", "rel_orbit",
        "footprint_m", "px", "has_vh", "tabular_key",
        "label", "flood_within_3d", "target_flood_1d", "target_onset_1d", "severity",
        "label_confidence", "discharge", "discharge_pctl",
        "precipitation_sum", "precip_sum_3d", "precip_sum_7d", "api_k090",
        "water_fraction", "vv_mean", "vh_mean", "valid_fraction", "split_temporal"]
img = img[[c for c in COLS if c in img.columns]]
img.to_csv(os.path.join(OUT, "image_dataset.csv"), index=False)

with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write(f"""Sri Lanka Sentinel-1 SAR flood image dataset
sites={len(SITES)} frames={len(img)} window={START}..{END} footprint={WIN_M:.0f}m px={OUT_PX}

PNG decoding (uint8 -> dB):
  VV_dB     = R/255*35 - 30      (range -30..+5)
  VH_dB     = G/255*35 - 30      (range -30..+5)
  VV-VH_dB  = B/255*20           (range 0..20, derived)
Quantisation step 0.137 dB, well below SAR speckle (~1-2 dB).
Train on R and G; B is a free derived ratio channel.

label            = flood_state  (GloFAS discharge >= 98th pctl of 2003-2017)
flood_within_3d  = flood occurs within t+1..t+3   <- best positive-rich target
severity         = none/moderate/high/severe
water_fraction   = share of valid pixels with VV < {WATER_DB} dB
split_temporal   = train (<=2017) / val (2018-2020) / test (2021+)
Never split randomly: use split_temporal or hold out a whole basin.
""")

print("=== SUMMARY ===")
print(f"frames            : {len(img)}")
print(f"per site          : {img.site_id.value_counts().to_dict()}")
print(f"label=1 (flood)   : {int((img.label == 1).sum())}")
print(f"flood_within_3d=1 : {int((img.flood_within_3d == 1).sum())}")
print(f"severity          : {img.severity.value_counts().to_dict()}")
print(f"split             : {img.split_temporal.value_counts().to_dict()}")
print(f"missing label     : {int(img.label.isna().sum())}")
sz = sum(os.path.getsize(os.path.join(dp, f))
         for dp, _, fs in os.walk(OUT) for f in fs)
print(f"total size        : {sz/1e9:.2f} GB")


# %% ------------------------------------------- STEP 5: small downloadable copy
if MAKE_LITE:
    print(f"\n=== STEP 5: {LITE_PX}px lite copy ===")
    lite = f"/kaggle/working/lite{LITE_PX}"
    for site_id, *_ in SITES:
        os.makedirs(os.path.join(lite, "frames", site_id), exist_ok=True)
    for _, r in img.iterrows():
        src = os.path.join(OUT, r["png_path"])
        dst = os.path.join(lite, r["png_path"])
        if os.path.exists(src) and not os.path.exists(dst):
            Image.open(src).resize((LITE_PX, LITE_PX), Image.BILINEAR).save(dst, optimize=True)
    for f in ("image_dataset.csv", "tabular_context.csv", "thresholds.json", "README.txt"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(lite, f))
    shutil.make_archive("/kaggle/working/sar_flood_lite", "zip", lite)
    shutil.rmtree(lite)
    z = os.path.getsize("/kaggle/working/sar_flood_lite.zip")
    print(f"sar_flood_lite.zip: {z/1e6:.0f} MB  <- download this one")


# %% ------------------------------------------- STEP 6: sanity check
import matplotlib.pyplot as plt

flood = img[img.label == 1].sort_values("water_fraction", ascending=False).head(3)
dry = img[(img.label == 0) & (img.site_id.isin(flood.site_id))].head(3)
show = pd.concat([flood, dry])
if len(show):
    fig, ax = plt.subplots(1, len(show), figsize=(4 * len(show), 4.4))
    ax = np.atleast_1d(ax)
    for a, (_, r) in zip(ax, show.iterrows()):
        a.imshow(Image.open(os.path.join(OUT, r["png_path"])))
        a.set_title(f"{r.site_id} {r.date}\nlabel={r.label} water={r.water_fraction}", fontsize=9)
        a.axis("off")
    plt.tight_layout(); plt.show()

print("\nDone. Save Version -> Save & Run All, then Output -> New Dataset.")
