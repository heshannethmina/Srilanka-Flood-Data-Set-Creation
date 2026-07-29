"""
Step 10 - Build the fixed-location Sentinel-1 IMAGE DATASET (Branch B v2).

Pins a few flood-prone Kelani sites and pulls EVERY Sentinel-1 pass across the
window at the same footprint, so each site is a clean dry->flood->recede time
series. One row per (site, satellite pass) is written to a single index CSV;
the raw pixels live in per-site .npy frames it points at.

Two data sources (both already used elsewhere, no API key):
  * Microsoft Planetary Computer STAC 'sentinel-1-rtc'  -> pixels + acquisition metadata
  * data/processed/flood_dataset.parquet (Branch A)      -> labels, discharge, splits

Outputs:
  data/timeseries/<site>/<YYYY-MM-DD>.npy       float16 [2,PX,PX] (VV,VH) dB  (real data)
  data/timeseries/<site>/<YYYY-MM-DD>_rgb.png   viewable false-color frame
  data/processed/image_dataset.csv              the dataset index (one row per frame)

Columns (A identity · B acquisition · C label · D SAR features · E split):
  image_id, site_id, site_name, basin, lat, lon, array_path, png_path,
  date, scene_id, platform, orbit_state, rel_orbit, footprint_m, px, has_vh,
  tabular_key, label, flood_within_3d, severity, discharge, discharge_pctl, event_id,
  water_fraction, vv_mean, vh_mean, valid_fraction,
  split_temporal

Env knobs:
  START=2015-01-01  END=2025-12-31   window to sweep
  WIN_M=5000        footprint metres
  OUT_PX=512        pixels per side (~10 m/px native at 5 km)
  WATER_DB=-17      VV dB threshold below which a pixel counts as open water
  WORKERS=4         parallel COG reads (MPC throttles high concurrency)

Needs: pip install pystac-client planetary-computer rasterio pyproj pillow
Run:   python scripts/imagery/10_build_image_dataset.py
"""
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import config as C

warnings.filterwarnings("ignore")

# 3 flood-prone Kelani sites forming a downstream transect (from nodes.py).
SITES = [
    ("KEL_HAN", "Hanwella",              "Kelani", 6.902, 80.082),
    ("KEL_KAD", "Kaduwela",              "Kelani", 6.933, 79.983),
    ("KEL_COL", "Colombo (Kelani mouth)", "Kelani", 6.970, 79.873),
]

START = os.environ.get("START", "2015-01-01")
END = os.environ.get("END", "2025-12-31")
WIN_M = float(os.environ.get("WIN_M", "5000"))
OUT_PX = int(os.environ.get("OUT_PX", "512"))
WATER_DB = float(os.environ.get("WATER_DB", "-17"))
WORKERS = int(os.environ.get("WORKERS", "4"))

DB_LO, DB_HI = -30.0, 5.0
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TS_DIR = os.path.join(C.ROOT, "data", "timeseries")
OUT_CSV = os.path.join(C.PROC_DIR, "image_dataset.csv")


def _lazy_imports():
    global pystac_client, planetary_computer, rasterio, from_bounds, Transformer
    import pystac_client, planetary_computer, rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer


def to_db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(x <= 0, np.nan, x))
    db = np.where(np.isfinite(db), db, DB_LO)
    return np.clip(db, DB_LO, DB_HI)


def read_band(signed_href, lon, lat):
    with rasterio.open(signed_href) as ds:
        tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        half = WIN_M / 2.0
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=0,
                      out_shape=(OUT_PX, OUT_PX))
    return to_db(arr.astype("float32"))


def norm(a, lo, hi):
    x = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).round().astype(np.uint8)


def save_png(vv, vh, path):
    rgb = np.dstack([norm(vv, -25, 0), norm(vh, -30, -5), norm(vv - vh, 0, 15)])
    Image.fromarray(rgb).save(path)


def _search_year(catalog, lon, lat, y0, y1):
    """One year-chunk STAC query, retried (MPC/Azure returns transient 504s)."""
    d = 0.02
    for attempt in range(5):
        try:
            search = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=[lon - d, lat - d, lon + d, lat + d],
                datetime=f"{y0}/{y1}",
                query={"sar:instrument_mode": {"eq": "IW"}},
            )
            return [it for it in search.items() if "vv" in it.assets]
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    return []


def list_passes(catalog, lon, lat):
    """All Sentinel-1 IW passes intersecting the point in [START,END], one per date.
    Queried year-by-year so a single huge request can't time out."""
    items = []
    for yr in range(pd.Timestamp(START).year, pd.Timestamp(END).year + 1):
        y0 = max(pd.Timestamp(START), pd.Timestamp(f"{yr}-01-01")).strftime("%Y-%m-%d")
        y1 = min(pd.Timestamp(END), pd.Timestamp(f"{yr}-12-31")).strftime("%Y-%m-%d")
        items.extend(_search_year(catalog, lon, lat, y0, y1))
    # prefer dual-pol (has vh) when two passes share a date, else nearest first
    items.sort(key=lambda it: (pd.Timestamp(it.datetime), "vh" not in it.assets))
    seen, out = set(), []
    for it in items:
        day = pd.Timestamp(it.datetime).strftime("%Y-%m-%d")
        if day in seen:
            continue
        seen.add(day)
        out.append((day, it))
    return out


def sar_features(vv, vh, has_vh):
    """Scalar SAR features from the chip. Excludes the nodata floor (-30 dB)."""
    valid = vv > (DB_LO + 0.1)
    vf = float(valid.mean())
    if valid.sum() == 0:
        return dict(water_fraction=np.nan, vv_mean=np.nan, vh_mean=np.nan,
                    valid_fraction=0.0)
    vvv = vv[valid]
    return dict(
        water_fraction=round(float((vvv < WATER_DB).mean()), 4),
        vv_mean=round(float(vvv.mean()), 2),
        vh_mean=round(float(vh[valid].mean()), 2) if has_vh else np.nan,
        valid_fraction=round(vf, 4),
    )


def get_array(site_id, day, item, lon, lat, sdir):
    """Load the chip from disk if cached, else fetch VV(+VH) from MPC. Returns (vv,vh,has_vh)."""
    npy = os.path.join(sdir, f"{day}.npy")
    png = os.path.join(sdir, f"{day}_rgb.png")
    has_vh = "vh" in item.assets
    if os.path.exists(npy):
        chip = np.load(npy).astype(np.float32)
        return chip[0], chip[1], has_vh
    last = None
    for attempt in range(4):
        try:
            vv = read_band(item.assets["vv"].href, lon, lat)
            vh = read_band(item.assets["vh"].href, lon, lat) if has_vh else np.zeros_like(vv)
            np.save(npy, np.stack([vv, vh]).astype("float16"))
            save_png(vv, vh, png)
            return vv, vh, has_vh
        except Exception as e:
            last = str(e)[:70]
            time.sleep(2 * (attempt + 1))
    print(f"    ! {site_id} {day} failed: {last}")
    return None, None, has_vh


def severity_str(r):
    if r.get("flood_severe") == 1:
        return "severe"
    if r.get("flood_high") == 1:
        return "high"
    if r.get("flood_moderate") == 1:
        return "moderate"
    return "none"


def main():
    _lazy_imports()
    os.makedirs(TS_DIR, exist_ok=True)
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace)

    print(f"Window {START}..{END}  footprint {WIN_M:.0f} m @ {OUT_PX}px  "
          f"water<{WATER_DB}dB  workers={WORKERS}")

    # 1) collect every (site, pass) task via cheap STAC queries
    tasks = []   # (site_id, name, basin, lat, lon, day, item, sdir)
    for site_id, name, basin, lat, lon in SITES:
        sdir = os.path.join(TS_DIR, site_id)
        os.makedirs(sdir, exist_ok=True)
        passes = list_passes(catalog, lon, lat)
        print(f"  [{site_id}] {name}: {len(passes)} Sentinel-1 passes")
        for day, item in passes:
            tasks.append((site_id, name, basin, lat, lon, day, item, sdir))

    # 2) fetch/load pixels in parallel, build A+B+D rows
    rows = []
    done = [0]

    def work(t):
        site_id, name, basin, lat, lon, day, item, sdir = t
        vv, vh, has_vh = get_array(site_id, day, item, lon, lat, sdir)
        if vv is None:
            return None
        p = item.properties
        row = dict(
            image_id=f"{site_id}_{day}", site_id=site_id, site_name=name, basin=basin,
            lat=lat, lon=lon,
            array_path=os.path.relpath(os.path.join(sdir, f"{day}.npy"), C.ROOT).replace("\\", "/"),
            png_path=os.path.relpath(os.path.join(sdir, f"{day}_rgb.png"), C.ROOT).replace("\\", "/"),
            date=day, scene_id=item.id, platform=p.get("platform"),
            orbit_state=p.get("sat:orbit_state"), rel_orbit=p.get("sat:relative_orbit"),
            footprint_m=int(WIN_M), px=OUT_PX, has_vh=has_vh,
            tabular_key=f"{site_id}_{day}",
        )
        row.update(sar_features(vv, vh, has_vh))
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            done[0] += 1
            if r:
                rows.append(r)
            if done[0] % 100 == 0 or done[0] == len(tasks):
                print(f"    {done[0]}/{len(tasks)} frames processed", flush=True)

    img = pd.DataFrame(rows).sort_values(["site_id", "date"]).reset_index(drop=True)

    # 3) join Branch-A labels/discharge/split (C + E) on tabular_key
    tab = pd.read_parquet(os.path.join(C.PROC_DIR, "flood_dataset.parquet"))
    tab["date"] = pd.to_datetime(tab["date"]).dt.strftime("%Y-%m-%d")
    tab["tabular_key"] = tab["node_id"].astype(str) + "_" + tab["date"]
    tab["severity"] = tab.apply(severity_str, axis=1)
    keep = tab[["tabular_key", "flood_state", "target_flood_3d", "severity",
                "river_discharge", "discharge_pctl", "event_id", "split_temporal"]].rename(
        columns={"flood_state": "label", "target_flood_3d": "flood_within_3d",
                 "river_discharge": "discharge"})
    img = img.merge(keep, on="tabular_key", how="left")

    cols = ["image_id", "site_id", "site_name", "basin", "lat", "lon",
            "array_path", "png_path",
            "date", "scene_id", "platform", "orbit_state", "rel_orbit",
            "footprint_m", "px", "has_vh",
            "tabular_key", "label", "flood_within_3d", "severity",
            "discharge", "discharge_pctl", "event_id",
            "water_fraction", "vv_mean", "vh_mean", "valid_fraction",
            "split_temporal"]
    img = img[cols]
    img.to_csv(OUT_CSV, index=False)

    npos = int((img["label"] == 1).sum())
    print(f"\nimage_dataset.csv: {len(img)} frames -> {OUT_CSV}")
    print(f"  flood (label=1): {npos}  |  dry: {len(img)-npos}  |  "
          f"missing label: {int(img['label'].isna().sum())}")
    print(f"  per site: {img['site_id'].value_counts().to_dict()}")
    print(f"  frames dir: {TS_DIR}")


if __name__ == "__main__":
    main()
