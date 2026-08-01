"""
Loader that pairs Kalu Ganga Sentinel-1 SAR chips (Branch B) with their
matching tabular feature rows (Branch A), ready for a fusion model.

Each sample:
  image   : float32 [2, 128, 128]  (VV, VH dB)
  tabular : float32 [F]             (matching flood_dataset.parquet row)
  label   : int (0/1)
  severity: int (0=none, 1=moderate, 2=high, 3=severe)

Usage:
  from loaders import load_pairs
  d = load_pairs()
  # d["image"]   : (N, 2, 128, 128)
  # d["tabular"] : (N, F)
  # d["label"]   : (N,)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))         # kalu_ganga_pipeline/
PROC = os.path.join(ROOT, "data", "processed")

SEV_MAP = {"none": 0, "moderate": 1, "high": 2, "severe": 3}

TAB_FEATURES = [
    "precipitation_sum", "precip_sum_3d", "precip_sum_7d", "precip_sum_30d",
    "api_k090", "soil_wet_top", "soil_wet_root", "soil_wet_profile",
    "temperature_2m_mean", "relative_humidity_2m", "windspeed_10m_max",
    "discharge", "log_discharge", "discharge_rise_3d", "discharge_anom",
    "discharge_zscore", "discharge_pctl", "elevation_m", "log_drainage_proxy",
]


def load_index():
    idx = pd.read_csv(os.path.join(PROC, "image_index.csv"))
    idx["severity_id"] = idx["severity"].map(SEV_MAP).fillna(0).astype(int)
    return idx


def load_pairs(feature_cols=None):
    """Return aligned arrays for all successfully fetched chips."""
    feature_cols = feature_cols or TAB_FEATURES
    idx = load_index()
    tab = pd.read_parquet(os.path.join(PROC, "flood_dataset.parquet"))
    tab["tabular_key"] = (tab["node_id"].astype(str) + "_"
                          + pd.to_datetime(tab["date"]).dt.strftime("%Y-%m-%d"))
    tab = tab.set_index("tabular_key")
    feats = [c for c in feature_cols if c in tab.columns]

    images, tabular, labels, sev, keys = [], [], [], [], []
    for _, r in idx.iterrows():
        p = os.path.join(ROOT, r["path"])
        if not os.path.exists(p):
            continue
        key = r["tabular_key"]
        if key not in tab.index:
            continue
        row = tab.loc[key, feats]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        images.append(np.load(p).astype(np.float32))
        tabular.append(row.to_numpy(np.float32))
        labels.append(int(r["label"]))
        sev.append(int(r["severity_id"]))
        keys.append(key)

    empty_img = np.empty((0, 2, 128, 128), np.float32)
    empty_tab = np.empty((0, len(feats)),  np.float32)
    return {
        "image":    np.stack(images)  if images  else empty_img,
        "tabular":  np.stack(tabular) if tabular else empty_tab,
        "label":    np.array(labels,  np.int64),
        "severity": np.array(sev,     np.int64),
        "keys":     keys,
        "features": feats,
    }


if __name__ == "__main__":
    d = load_pairs()
    print(f"Paired samples : {len(d['label'])}")
    if len(d["label"]):
        print(f"  image tensor  : {d['image'].shape}  (N, 2, 128, 128)")
        print(f"  tabular tensor: {d['tabular'].shape}  features={len(d['features'])}")
        print(f"  positives     : {int(d['label'].sum())} / {len(d['label'])}")
        print(f"  severity dist : {np.bincount(d['severity'], minlength=4).tolist()}"
              f"  (none/mod/high/severe)")
