"""
Step 9 (sample) - Fixed-location Sentinel-1 TIME SERIES over the Kelani river.

Instead of scattered one-off chips (07_fetch_images_mpc.py), this pins a few
FLOOD-PRONE coordinates and pulls every Sentinel-1 pass across a date window at
the *same* footprint. Because the frame never moves you get a clean
    dry  ->  flooding  ->  receding
sequence you can flip through -- much clearer and re-runnable for any window.

Default window aims at the May-2016 Colombo/Kelani flood so water actually
rises and falls in the series.

Outputs (per site <ID>):
  data/timeseries/<ID>/<YYYY-MM-DD>.npy      float16 [2,OUT_PX,OUT_PX] (VV,VH) dB
  data/timeseries/<ID>/<YYYY-MM-DD>_rgb.png  viewable false-color frame
  data/timeseries/<ID>_contact.png           dated grid of the whole series

Env knobs:
  START=2016-04-15  END=2016-07-15   date window to sweep
  WIN_M=6000        footprint metres (bigger = more river/floodplain context)
  OUT_PX=256        output pixels (sharper than the 128px training chips)
  MAX_FRAMES=0      cap frames per site (0 = all passes in the window)

Needs: pip install pystac-client planetary-computer rasterio pyproj matplotlib
Run:   python scripts/09_timeseries_sample.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

warnings.filterwarnings("ignore")

# 3 flood-prone Kelani sites (from nodes.py). id, name, lat, lon
SITES = [
    ("KEL_HAN", "Hanwella",              6.902, 80.082),
    ("KEL_KAD", "Kaduwela",              6.933, 79.983),
    ("KEL_COL", "Colombo (Kelani mouth)", 6.970, 79.873),
]

START = os.environ.get("START", "2016-04-15")
END = os.environ.get("END", "2016-07-15")
WIN_M = float(os.environ.get("WIN_M", "6000"))
OUT_PX = int(os.environ.get("OUT_PX", "256"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))

DB_LO, DB_HI = -30.0, 5.0
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TS_DIR = os.path.join(C.ROOT, "data", "timeseries")


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


def read_chip(signed_href, lon, lat):
    with rasterio.open(signed_href) as ds:
        tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        half = WIN_M / 2.0
        win = from_bounds(cx - half, cy - half, cx + half, cy + half, ds.transform)
        arr = ds.read(1, window=win, boundless=True, fill_value=0,
                      out_shape=(OUT_PX, OUT_PX))
    return to_db(arr.astype("float32"))


def norm(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def falsecolor(vv, vh):
    """VV / VH / VV-VH -> RGB. blue-ish = smooth water, green = flooded veg."""
    return np.dstack([norm(vv, -25, 0), norm(vh, -30, -5), norm(vv - vh, 0, 15)])


def list_passes(catalog, lon, lat):
    """All Sentinel-1 IW passes intersecting the point in [START,END], one per date."""
    d = 0.02
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=[lon - d, lat - d, lon + d, lat + d],
        datetime=f"{START}/{END}",
        query={"sar:instrument_mode": {"eq": "IW"}},
    )
    items = [it for it in search.items() if "vv" in it.assets]
    items.sort(key=lambda it: pd.Timestamp(it.datetime))
    seen, out = set(), []
    for it in items:
        day = pd.Timestamp(it.datetime).strftime("%Y-%m-%d")
        if day in seen:
            continue
        seen.add(day)
        out.append((day, it))
    if MAX_FRAMES > 0:
        out = out[:MAX_FRAMES]
    return out


def fetch_frame(it, lon, lat):
    """Read one pass with a small retry loop (MPC throttles anonymous signing)."""
    for attempt in range(4):
        try:
            vv = read_chip(it.assets["vv"].href, lon, lat)
            vh = (read_chip(it.assets["vh"].href, lon, lat)
                  if "vh" in it.assets else np.zeros_like(vv))
            return vv, vh
        except Exception as e:
            last = str(e)[:60]
            time.sleep(2 * (attempt + 1))
    print(f"      ! failed: {last}")
    return None, None


def contact_sheet(site_id, name, frames):
    """frames: list of (day, rgb). Dated grid so the flood pulse is visible."""
    n = len(frames)
    if n == 0:
        return
    ncol = min(6, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (day, rgb) in zip(np.ravel(axes), frames):
        ax.imshow(rgb)
        ax.set_title(day, fontsize=8)
    fig.suptitle(f"{site_id}  {name} - Sentinel-1 time series "
                 f"(blue-ish = water)  {START} .. {END}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(TS_DIR, f"{site_id}_contact.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    contact sheet -> {out}")


def main():
    _lazy_imports()
    os.makedirs(TS_DIR, exist_ok=True)
    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace)

    print(f"Window {START} .. {END}   footprint {WIN_M:.0f} m @ {OUT_PX}px")
    for site_id, name, lat, lon in SITES:
        print(f"\n[{site_id}] {name} ({lat},{lon})")
        sdir = os.path.join(TS_DIR, site_id)
        os.makedirs(sdir, exist_ok=True)
        passes = list_passes(catalog, lon, lat)
        print(f"  {len(passes)} Sentinel-1 passes found")
        frames = []
        for day, it in passes:
            npy = os.path.join(sdir, f"{day}.npy")
            png = os.path.join(sdir, f"{day}_rgb.png")
            if os.path.exists(npy) and os.path.exists(png):
                chip = np.load(npy).astype(np.float32)
                rgb = falsecolor(chip[0], chip[1])
            else:
                vv, vh = fetch_frame(it, lon, lat)
                if vv is None:
                    continue
                np.save(npy, np.stack([vv, vh]).astype("float16"))
                rgb = falsecolor(vv, vh)
                plt.imsave(png, rgb)
            frames.append((day, rgb))
            print(f"    {day}  ok")
        contact_sheet(site_id, name, frames)

    print(f"\ndone -> {TS_DIR}  (open the *_contact.png files)")


if __name__ == "__main__":
    main()
