"""
Step 7 - Fetch Sentinel-1 SAR chips for each manifest row (Branch B imagery).

Source: Microsoft Planetary Computer (STAC), collection 'sentinel-1-rtc'.
No account required -- asset URLs are signed anonymously via planetary_computer.

For each manifest row we:
  1. search MPC for S1 IW-mode scenes intersecting the node within +/- window,
  2. pick the scene nearest the target_date that has VV (+ VH),
  3. read a small window (default ~2.56 km) around the node from the VV/VH COGs,
  4. convert to dB, normalize, resize to OUT_PX, save as a 2-channel .npy chip.

Outputs:
  data/images/<sample_id>.npy         float16 [2, OUT_PX, OUT_PX] (VV, VH)
  data/processed/image_index.csv       one row per successfully fetched chip

Env knobs:
  MAX_CHIPS=N     only fetch the first N manifest rows (for a quick subset)
  IMG_WORKERS=K   parallel workers (default 4)

Run:  python 07_fetch_images_mpc.py
"""
import os
import sys
import math
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

warnings.filterwarnings("ignore")

OUT_PX = 128            # output chip size (pixels)
WIN_M = 2560.0         # window footprint in metres (~2.56 km)
SEARCH_DAYS = 12       # +/- days to look for a satellite pass (S1 revisit ~12d)
DB_LO, DB_HI = -30.0, 5.0   # SAR backscatter dB clip range; nodata -> DB_LO
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
IMG_DIR = os.path.join(C.ROOT, "data", "images")


def _lazy_imports():
    global pystac_client, planetary_computer, rasterio, Window, from_bounds, Transformer
    import pystac_client, planetary_computer, rasterio
    from rasterio.windows import Window, from_bounds
    from pyproj import Transformer


def to_db(x):
    """Linear power -> dB, clipped to [DB_LO, DB_HI]; nodata/invalid -> DB_LO."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(x <= 0, np.nan, x))
    db = np.where(np.isfinite(db), db, DB_LO)
    return np.clip(db, DB_LO, DB_HI)


def read_chip(signed_href, lon, lat):
    """Read a WIN_M window around (lon,lat) from a signed S1 COG; return dB array."""
    with rasterio.open(signed_href) as ds:
        tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        half = WIN_M / 2.0
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=0,
                      out_shape=(OUT_PX, OUT_PX))
    return to_db(arr.astype("float32"))


def fetch_one(row, catalog):
    """Retry wrapper around _fetch_one_try to survive MPC signing throttles."""
    import time
    sid = row["sample_id"]
    out = os.path.join(IMG_DIR, f"{sid}.npy")
    if os.path.exists(out):
        return sid, "cached", None
    last = None
    for attempt in range(5):
        try:
            return _fetch_one_try(row, catalog, out)
        except Exception as e:
            last = str(e)[:60]
            time.sleep(2 * (attempt + 1) + (hash(sid) % 3))   # backoff + jitter
    return sid, f"err:{last}", None


def _fetch_one_try(row, catalog, out):
    sid = row["sample_id"]
    lon, lat = float(row["lon"]), float(row["lat"])
    d = 0.02
    tgt = pd.Timestamp(row["target_date"])
    w0 = (tgt - pd.Timedelta(days=SEARCH_DAYS)).date()
    w1 = (tgt + pd.Timedelta(days=SEARCH_DAYS)).date()
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=[lon - d, lat - d, lon + d, lat + d],
        datetime=f"{w0}/{w1}",
        query={"sar:instrument_mode": {"eq": "IW"}},
    )
    items = list(search.items())
    if not items:
        return sid, "no_scene", None
    tgt = pd.Timestamp(row["target_date"]).tz_localize("UTC")
    items.sort(key=lambda it: abs(pd.Timestamp(it.datetime) - tgt))
    item = None
    for it in items:
        if "vv" in it.assets:
            item = it
            break
    if item is None:
        return sid, "no_vv", None
    vv = read_chip(item.assets["vv"].href, lon, lat)
    vh = (read_chip(item.assets["vh"].href, lon, lat)
          if "vh" in item.assets else np.zeros_like(vv))
    chip = np.stack([vv, vh]).astype("float16")
    np.save(out, chip)
    return sid, "ok", {"sample_id": sid, "scene_id": item.id,
                       "actual_date": pd.Timestamp(item.datetime).date(),
                       "path": os.path.relpath(out, C.ROOT),
                       "node_id": row["node_id"], "label": row["label"],
                       "severity": row["severity"], "purpose": row["purpose"],
                       "tabular_key": row["tabular_key"]}


def main():
    _lazy_imports()
    os.makedirs(IMG_DIR, exist_ok=True)
    man = pd.read_csv(os.path.join(C.PROC_DIR, "image_manifest.csv"))
    only = os.environ.get("ONLY_LABEL", "")          # "0" or "1" to fetch one class
    if only != "":
        man = man[man["label"] == int(only)]
    if os.environ.get("SHUFFLE", "0") == "1":        # balanced random subset
        man = man.sample(frac=1, random_state=7).reset_index(drop=True)
    limit = int(os.environ.get("MAX_CHIPS", "0"))
    if limit > 0:
        man = man.head(limit)
    workers = int(os.environ.get("IMG_WORKERS", "3"))   # MPC throttles high concurrency
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    records, stats = [], {}
    print(f"Fetching {len(man)} chips with {workers} workers -> {IMG_DIR}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, r, catalog): r["sample_id"]
                for _, r in man.iterrows()}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sid, status, rec = fut.result()
            except Exception as e:
                sid, status, rec = futs[fut], f"err:{str(e)[:40]}", None
            stats[status.split(':')[0]] = stats.get(status.split(':')[0], 0) + 1
            if rec:
                records.append(rec)
            if done % 25 == 0 or done == len(man):
                print(f"  [{done}/{len(man)}] {stats}", flush=True)

    if records:
        idx = pd.DataFrame(records)
        idxp = os.path.join(C.PROC_DIR, "image_index.csv")
        if os.path.exists(idxp):
            old = pd.read_csv(idxp)
            idx = pd.concat([old, idx]).drop_duplicates("sample_id", keep="last")
        idx.to_csv(idxp, index=False)
        print(f"\nSaved {len(records)} new chips. image_index.csv now {len(idx)} rows.")
    print(f"Status summary: {stats}")


if __name__ == "__main__":
    main()
