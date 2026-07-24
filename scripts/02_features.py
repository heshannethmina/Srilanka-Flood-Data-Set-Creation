"""
Step 2 - Assemble the per-node daily panel and engineer features.

Reads cached raw parquet from data/raw, merges weather + soil/humidity +
discharge per node, and computes dynamic (time-varying) and static features.

Output: data/processed/panel_features.parquet   (long format: one row per node-day)
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from nodes import as_records


def load_node_frame(nid):
    w = pd.read_parquet(os.path.join(C.RAW_DIR, f"{nid}_weather.parquet"))
    f = pd.read_parquet(os.path.join(C.RAW_DIR, f"{nid}_flood.parquet"))
    df = w.merge(f, on="date", how="left")
    # hourly soil moisture / humidity is optional (best-effort feed)
    hpath = os.path.join(C.RAW_DIR, f"{nid}_hourly.parquet")
    if os.path.exists(hpath):
        h = pd.read_parquet(hpath)
        df = df.merge(h, on="date", how="left")
    # high-res ERA5 precipitation (preferred over POWER's coarse 0.5deg precip)
    ppath = os.path.join(C.RAW_DIR, f"{nid}_precip.parquet")
    if os.path.exists(ppath):
        pr = pd.read_parquet(ppath)
        df = df.merge(pr, on="date", how="left")
        df = df.rename(columns={"precipitation_sum": "precipitation_sum_power"})
        df["precipitation_sum"] = pd.to_numeric(df["precip_era5"], errors="coerce")
        # fall back to POWER precip where ERA5 is missing
        df["precipitation_sum"] = df["precipitation_sum"].fillna(
            pd.to_numeric(df["precipitation_sum_power"], errors="coerce"))
    df["date"] = pd.to_datetime(df["date"])
    df["river_discharge"] = pd.to_numeric(df.get("river_discharge"), errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def antecedent_precip(df):
    p = df["precipitation_sum"].fillna(0.0)
    for win in (2, 3, 5, 7, 10, 15, 30):
        df[f"precip_sum_{win}d"] = p.rolling(win, min_periods=1).sum()
    df["precip_max_3d"] = p.rolling(3, min_periods=1).max()
    df["precip_max_7d"] = p.rolling(7, min_periods=1).max()
    # Antecedent Precipitation Index (exponential decay, k=0.90)
    k = 0.90
    api = np.zeros(len(p))
    acc = 0.0
    pv = p.to_numpy()
    for i in range(len(pv)):
        acc = acc * k + pv[i]
        api[i] = acc
    df["api_k090"] = api
    # wet-day count in trailing window
    df["wetdays_7d"] = (p > 1.0).rolling(7, min_periods=1).sum()
    return df


def discharge_features(df):
    q = df["river_discharge"].astype(float)
    df["discharge"] = q
    df["log_discharge"] = np.log1p(q.clip(lower=0))
    # day-to-day rise (flood dynamics)
    df["discharge_rise_1d"] = q.diff(1)
    df["discharge_rise_3d"] = q - q.shift(3)
    df["discharge_mean_3d"] = q.rolling(3, min_periods=1).mean()
    df["discharge_mean_7d"] = q.rolling(7, min_periods=1).mean()

    # Seasonal (day-of-year) climatology from TRAIN period only to avoid leakage
    train_mask = df["date"] <= pd.Timestamp(C.TRAIN_END)
    doy = df["date"].dt.dayofyear
    clim = (df.loc[train_mask]
              .assign(doy=doy[train_mask])
              .groupby("doy")["river_discharge"].agg(["mean", "std"]))
    # smooth climatology with circular rolling mean
    clim = clim.reindex(range(1, 367)).astype(float).interpolate().bfill().ffill()
    clim["mean_s"] = clim["mean"].rolling(15, min_periods=1, center=True).mean()
    clim["std_s"] = clim["std"].rolling(15, min_periods=1, center=True).mean().replace(0, np.nan)
    df["_doy"] = doy
    df["q_clim_mean"] = df["_doy"].map(clim["mean_s"])
    df["q_clim_std"] = df["_doy"].map(clim["std_s"])
    df["discharge_anom"] = df["discharge"] - df["q_clim_mean"]
    df["discharge_zscore"] = df["discharge_anom"] / df["q_clim_std"].replace(0, np.nan)
    # empirical percentile rank vs TRAIN distribution
    train_q = df.loc[train_mask, "river_discharge"].dropna().to_numpy()
    train_q_sorted = np.sort(train_q)
    df["discharge_pctl"] = np.searchsorted(train_q_sorted, q.to_numpy(), side="right") / max(len(train_q_sorted), 1)
    df.drop(columns=["_doy"], inplace=True)
    return df, train_q_sorted


def build():
    nodes = {n["node_id"]: n for n in as_records()}
    meta_path = os.path.join(C.PROC_DIR, "nodes_meta.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

    frames = []
    thresholds = {}
    for nid, nd in nodes.items():
        wpath = os.path.join(C.RAW_DIR, f"{nid}_weather.parquet")
        fpath = os.path.join(C.RAW_DIR, f"{nid}_flood.parquet")
        if not (os.path.exists(wpath) and os.path.exists(fpath)):
            print(f"  WARN: missing weather/flood raw for {nid}, skipping")
            continue
        df = load_node_frame(nid)
        if df["river_discharge"].notna().sum() < 365:
            print(f"  WARN: {nid} has no usable discharge, skipping")
            continue
        df["node_id"] = nid
        df["basin"] = nd["basin"]
        df["zone"] = nd["zone"]
        df["position"] = nd["position"]

        # fill short gaps in POWER meteorology / soil wetness
        for col in ["soil_wet_top", "soil_wet_root", "soil_wet_profile",
                    "relative_humidity_2m", "shortwave_radiation",
                    "windspeed_10m_mean", "windspeed_10m_max",
                    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]:
            if col in df:
                df[col] = df[col].interpolate(limit=5).ffill().bfill()

        df = antecedent_precip(df)
        df, train_q_sorted = discharge_features(df)

        # soil-wetness anomaly vs train climatology (simple mean)
        tm = df["date"] <= pd.Timestamp(C.TRAIN_END)
        for sc in ["soil_wet_top", "soil_wet_root", "soil_wet_profile"]:
            if sc in df:
                mu = df.loc[tm, sc].mean()
                df[f"{sc}_anom"] = df[sc] - mu

        # per-node return-period discharge thresholds (train distribution)
        qtrain = df.loc[tm, "river_discharge"].dropna()
        thr = {sev: float(qtrain.quantile(p)) for sev, p in C.SEVERITY_PERCENTILES.items()}
        # annual-maxima based ~return level for validation/reporting
        ann_max = df.loc[tm].assign(y=df.loc[tm, "date"].dt.year).groupby("y")["river_discharge"].max()
        thr["annmax_median"] = float(ann_max.median()) if len(ann_max) else np.nan
        thr["q_mean_train"] = float(qtrain.mean())
        thresholds[nid] = thr

        frames.append(df)
        print(f"  {nid}: {len(df)} rows  thr(high)={thr['high']:.1f} m3/s")

    panel = pd.concat(frames, ignore_index=True)

    # static terrain features from meta (snapped elevation) + derived proxies
    if meta is not None:
        elev = meta.set_index("node_id")["elevation_m"].to_dict()
        panel["elevation_m"] = panel["node_id"].map(elev)
    # drainage-area proxy = mean train discharge (larger river => larger area)
    qmean = {nid: thresholds[nid]["q_mean_train"] for nid in thresholds}
    panel["drainage_proxy_q"] = panel["node_id"].map(qmean)
    panel["log_drainage_proxy"] = np.log1p(panel["drainage_proxy_q"].fillna(0))

    os.makedirs(C.PROC_DIR, exist_ok=True)
    panel.to_parquet(os.path.join(C.PROC_DIR, "panel_features.parquet"), index=False)
    with open(os.path.join(C.PROC_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nPanel: {panel.shape[0]} node-days x {panel.shape[1]} cols "
          f"-> data/processed/panel_features.parquet")
    return panel


if __name__ == "__main__":
    build()
