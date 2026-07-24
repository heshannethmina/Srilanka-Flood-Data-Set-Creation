"""
Step 1 - Download real reanalysis data for every node (serial + resilient).

Sources (both public, no API key):
  * Open-Meteo Archive API  -> ERA5-Land backed daily weather (+ optional hourly soil/humidity)
  * Open-Meteo Flood API    -> GloFAS reanalysis daily river discharge

Design for the free tier's minutely/hourly quotas:
  * runs SERIALLY with polite pacing (parallel bursts trip the rate limiter)
  * caches each feed independently, so essential feeds survive if a later one fails
  * Phase 1: daily weather + discharge for ALL nodes  (cheap, essential)
  * Phase 2: hourly soil moisture + humidity          (expensive, best-effort)
  * fully resumable: re-running skips whatever is already cached
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
            # hourly-quota block clears only next clock hour -> wait it out in ~90s steps
            if "Hourly API request limit" in body:
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
        "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": C.START_DATE.replace("-", ""),
        "end": C.END_DATE.replace("-", ""),
        "format": "JSON",
    })
    p = j["properties"]["parameter"]
    frames = {}
    for src, dst in C.POWER_PARAMS.items():
        s = pd.Series(p[src], name=dst)
        frames[dst] = s
    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    # POWER uses -999 as fill; replace with NaN
    df = df.replace(C.POWER_FILL, np.nan)
    coords = j.get("geometry", {}).get("coordinates", [lon, lat, np.nan])
    meta = {"snap_lat": coords[1], "snap_lon": coords[0],
            "elevation_m": coords[2] if len(coords) > 2 else np.nan}
    return df, meta


def fetch_precip_era5(lat, lon):
    """High-resolution (0.1deg) ERA5-Land daily precipitation from Open-Meteo.
    Single-variable request -> light enough to stay within the free-tier quota."""
    j = _get(C.ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "daily": "precipitation_sum", "timezone": "UTC",
    })
    df = pd.DataFrame(j["daily"]).rename(
        columns={"time": "date", "precipitation_sum": "precip_era5"})
    return df


def fetch_discharge(lat, lon):
    j = _get(C.FLOOD_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": C.START_DATE, "end_date": C.END_DATE,
        "daily": ",".join(C.FLOOD_VARS), "timezone": "UTC",
    })
    return pd.DataFrame(j["daily"]).rename(columns={"time": "date"})


def p(nid, suffix):
    return os.path.join(C.RAW_DIR, f"{nid}_{suffix}")


def phase_discharge(nodes):
    """Fetch GloFAS discharge for all nodes (flood-api; separate quota)."""
    print("=== Phase D: river discharge (flood-api) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]; fpath = p(nid, "flood.parquet")
        if os.path.exists(fpath):
            print(f"[PD {i}/{len(nodes)}] {nid} cached", flush=True); continue
        fetch_discharge(nd["lat"], nd["lon"]).to_parquet(fpath, index=False)
        print(f"[PD {i}/{len(nodes)}] {nid} downloaded", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def phase_precip(nodes):
    """Fetch high-res ERA5 daily precipitation for all nodes (archive-api)."""
    print("=== Phase P: high-res ERA5 precipitation (archive-api) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]; ppath = p(nid, "precip.parquet")
        if os.path.exists(ppath):
            print(f"[PP {i}/{len(nodes)}] {nid} cached", flush=True); continue
        fetch_precip_era5(nd["lat"], nd["lon"]).to_parquet(ppath, index=False)
        print(f"[PP {i}/{len(nodes)}] {nid} downloaded", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def phase_weather(nodes):
    """Fetch daily weather + meta for all nodes (archive-api)."""
    print("=== Phase W: daily weather (archive-api) ===", flush=True)
    for i, nd in enumerate(nodes, 1):
        nid = nd["node_id"]
        wpath, mpath = p(nid, "weather.parquet"), p(nid, "meta.json")
        if os.path.exists(wpath) and os.path.exists(mpath):
            print(f"[PW {i}/{len(nodes)}] {nid} cached", flush=True); continue
        wdf, meta = fetch_daily_weather(nd["lat"], nd["lon"])
        wdf.to_parquet(wpath, index=False)
        meta.update({"node_id": nid, **{k: nd[k] for k in
                    ("name", "basin", "lat", "lon", "position", "downstream_of", "zone")}})
        with open(mpath, "w") as f:
            json.dump(meta, f)
        print(f"[PW {i}/{len(nodes)}] {nid} downloaded (elev {meta['elevation_m']}m)", flush=True)
        time.sleep(C.SLEEP_BETWEEN)


def phase1_essential(nodes):
    phase_discharge(nodes)
    phase_weather(nodes)


def write_meta_csv(nodes):
    rows = []
    for nd in nodes:
        mpath = p(nd["node_id"], "meta.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                rows.append(json.load(f))
    if rows:
        os.makedirs(C.PROC_DIR, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(C.PROC_DIR, "nodes_meta.csv"), index=False)
    print(f"nodes_meta.csv written with {len(rows)} nodes", flush=True)


def main():
    os.makedirs(C.RAW_DIR, exist_ok=True)
    nodes = as_records()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "discharge"):
        phase_discharge(nodes)
    if stage in ("all", "weather"):
        phase_weather(nodes)
        write_meta_csv(nodes)
    if stage in ("all", "precip"):
        phase_precip(nodes)
    print(f"\nDownload stage '{stage}' complete (re-run to resume any gaps).", flush=True)


if __name__ == "__main__":
    main()
