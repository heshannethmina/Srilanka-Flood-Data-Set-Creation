"""
Step 1 - Download raw reanalysis data for all 5 Kalu Ganga nodes.

Sources (both public, no API key):
  * NASA POWER Archive API  -> ERA5-Land backed daily weather + soil wetness
  * Open-Meteo Flood API    -> GloFAS reanalysis daily river discharge
  * Open-Meteo Archive API  -> high-res ERA5 precipitation

Design:
  * Runs SERIALLY with polite pacing (parallel bursts trip the rate limiter)
  * Caches each feed independently - safe to stop and restart
  * Fully resumable: re-running skips whatever is already cached

Usage:
  python 01_download.py          # all feeds
  python 01_download.py discharge
  python 01_download.py weather
  python 01_download.py precip
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import requests

# ---- local imports -----------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
from nodes import as_records


def _get(url, params, heavy=False):
    last = None
    for attempt in range(C.MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=C.REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            body = r.text[:160]
            last = f"HTTP {r.status_code}: {body}"
            if "Hourly API request limit" in body:
                print(f"  Rate-limited. Waiting 95s ...", flush=True)
                time.sleep(95)
                continue
            base = 3
        except requests.RequestException as e:
            last = str(e)
            base = 4
        wait = min(base * (attempt + 1) * (2 if heavy else 1), 60)
        time.sleep(wait)
    raise RuntimeError(f"Failed after {C.MAX_RETRIES} retries: {last}")


def fetch_daily_weather(lat, lon):
    """NASA POWER daily meteorology (open, no key, includes soil wetness)."""
    j = _get(C.POWER_URL, {
        "parameters": ",".join(C.POWER_PARAMS.keys()),
        "community":  "AG",
        "longitude":  lon, "latitude": lat,
        "start":      C.START_DATE.replace("-", ""),
        "end":        C.END_DATE.replace("-", ""),
        "format":     "JSON",
    })
    p = j["properties"]["parameter"]
    frames = {dst: pd.Series(p[src], name=dst) for src, dst in C.POWER_PARAMS.items()}
    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    df = df.replace(C.POWER_FILL, np.nan)
    coords = j.get("geometry", {}).get("coordinates", [lon, lat, np.nan])
    meta = {
        "snap_lat": coords[1], "snap_lon": coords[0],
        "elevation_m": coords[2] if len(coords) > 2 else np.nan,
    }
    return df, meta


def fetch_precip_era5(lat, lon):
    """High-resolution (0.1deg) ERA5-Land daily precipitation from Open-Meteo."""
    j = _get(C.ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "daily": "precipitation_sum", "timezone": "UTC",
    })
    df = pd.DataFrame(j["daily"]).rename(
        columns={"time": "date", "precipitation_sum": "precip_era5"})
    return df


def fetch_discharge(lat, lon):
    """GloFAS reanalysis river discharge."""
    j = _get(C.FLOOD_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "daily": ",".join(C.FLOOD_VARS), "timezone": "UTC",
    })
    return pd.DataFrame(j["daily"]).rename(columns={"time": "date"})


def raw_path(nid, suffix):
    return os.path.join(C.RAW_DIR, f"{nid}_{suffix}")


# ── Phase runners ─────────────────────────────────────────────────────────────

def phase_discharge(nodes):
    print("=== Phase D: river discharge (GloFAS / flood-api) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]
        fpath = raw_path(nid, "flood.parquet")
        if os.path.exists(fpath):
            print(f"  [D {i}/{len(nodes)}] {nid}  CACHED", flush=True)
            continue
        df = fetch_discharge(nd["lat"], nd["lon"])
        df.to_parquet(fpath, index=False)
        print(f"  [D {i}/{len(nodes)}] {nid}  {len(df)} days downloaded", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def phase_weather(nodes):
    print("=== Phase W: daily weather + soil wetness (NASA POWER) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]
        wpath = raw_path(nid, "weather.parquet")
        mpath = raw_path(nid, "meta.json")
        if os.path.exists(wpath) and os.path.exists(mpath):
            print(f"  [W {i}/{len(nodes)}] {nid}  CACHED", flush=True)
            continue
        wdf, meta = fetch_daily_weather(nd["lat"], nd["lon"])
        wdf.to_parquet(wpath, index=False)
        meta.update({"node_id": nid,
                     **{k: nd[k] for k in ("name", "basin", "lat", "lon",
                                           "position", "downstream_of", "zone")}})
        with open(mpath, "w") as f:
            json.dump(meta, f)
        print(f"  [W {i}/{len(nodes)}] {nid}  elev={meta['elevation_m']}m", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def phase_precip(nodes):
    print("=== Phase P: high-res ERA5 precipitation (Open-Meteo archive) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]
        ppath = raw_path(nid, "precip.parquet")
        if os.path.exists(ppath):
            print(f"  [P {i}/{len(nodes)}] {nid}  CACHED", flush=True)
            continue
        df = fetch_precip_era5(nd["lat"], nd["lon"])
        df.to_parquet(ppath, index=False)
        print(f"  [P {i}/{len(nodes)}] {nid}  {len(df)} days downloaded", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def write_meta_csv(nodes):
    rows = []
    for nd in nodes:
        mpath = raw_path(nd["node_id"], "meta.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                rows.append(json.load(f))
    if rows:
        os.makedirs(C.PROC_DIR, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(C.PROC_DIR, "nodes_meta.csv"), index=False)
    print(f"  nodes_meta.csv: {len(rows)} nodes", flush=True)


def main():
    os.makedirs(C.RAW_DIR, exist_ok=True)
    nodes = as_records()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"\nKalu Ganga download  stage='{stage}'  nodes={len(nodes)}"
          f"  {C.START_DATE} -> {C.END_DATE}\n")
    if stage in ("all", "discharge"):
        phase_discharge(nodes)
    if stage in ("all", "weather"):
        phase_weather(nodes)
        write_meta_csv(nodes)
    if stage in ("all", "precip"):
        phase_precip(nodes)
    print(f"\nDownload '{stage}' complete.  Re-run to fill any gaps.")


if __name__ == "__main__":
    main()
