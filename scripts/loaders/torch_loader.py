"""
Example loader that turns the materialized dataset into spatiotemporal graph
tensors for a PyTorch / PyTorch-Geometric flood-forecasting model.

This is a reference implementation showing the intended data contract; it does
not require torch to be installed unless you actually call build_torch_geometric().

Sample = one time step t. Model input = a window of the previous `lookback`
days of node features over the full graph; target = flood within h days ahead.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.abspath(os.path.join(HERE, "..", "..", "data", "processed"))

# Dynamic feature columns fed to the model (edit as needed)
DYNAMIC_FEATURES = [
    "precipitation_sum", "precip_sum_2d", "precip_sum_3d",
    "precip_sum_5d", "precip_sum_7d", "precip_sum_15d", "precip_sum_30d",
    "precip_max_3d", "precip_max_7d", "api_k090", "wetdays_7d",
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m", "windspeed_10m_mean", "windspeed_10m_max",
    "shortwave_radiation",
    "soil_wet_top", "soil_wet_root", "soil_wet_profile",
    "soil_wet_top_anom", "soil_wet_root_anom", "soil_wet_profile_anom",
    "discharge", "log_discharge", "discharge_rise_1d", "discharge_rise_3d",
    "discharge_mean_3d", "discharge_mean_7d", "discharge_anom",
    "discharge_zscore", "discharge_pctl",
]
OPTIONAL_FEATURES = []  # soil wetness now comes from NASA POWER (always present)
STATIC_FEATURES = ["elevation_m", "log_drainage_proxy"]
TARGET = "target_flood_1d"   # 24h-ahead flood probability


def load_frames():
    df = pd.read_parquet(os.path.join(PROC, "flood_dataset.parquet"))
    nodes = pd.read_csv(os.path.join(PROC, "nodes.csv"))
    edges = pd.read_csv(os.path.join(PROC, "edges.csv"))
    return df, nodes, edges


def to_node_time_tensor(df, features):
    """Return X[T, N, F], target[T, N], dates, node_ids."""
    df = df.sort_values(["date", "node_id"])
    node_ids = sorted(df["node_id"].unique())
    dates = sorted(df["date"].unique())
    nidx = {n: i for i, n in enumerate(node_ids)}
    didx = {d: i for i, d in enumerate(dates)}
    feats = [f for f in features if f in df.columns]
    X = np.full((len(dates), len(node_ids), len(feats)), np.nan, np.float32)
    y = np.full((len(dates), len(node_ids)), np.nan, np.float32)
    m = np.zeros((len(dates), len(node_ids)), np.float32)
    fi = {f: k for k, f in enumerate(feats)}
    arr = df[["date", "node_id", TARGET, "valid_sample"] + feats].to_numpy(object)
    cols = ["date", "node_id", TARGET, "valid_sample"] + feats
    ci = {c: k for k, c in enumerate(cols)}
    for row in arr:
        t = didx[row[ci["date"]]]; n = nidx[row[ci["node_id"]]]
        for f in feats:
            X[t, n, fi[f]] = row[ci[f]]
        y[t, n] = row[ci[TARGET]]
        m[t, n] = row[ci["valid_sample"]]
    return X, y, m, dates, node_ids, feats


def build_edge_index(edges, node_ids, edge_type=None):
    nidx = {n: i for i, n in enumerate(node_ids)}
    e = edges if edge_type is None else edges[edges["edge_type"] == edge_type]
    src = [nidx[s] for s in e["src"] if s in nidx]
    dst = [nidx[d] for d in e["dst"] if d in nidx]
    import numpy as np
    return np.array([src, dst], dtype=np.int64), e["weight"].to_numpy(np.float32)


def build_torch_geometric(lookback=14, horizon_target="target_flood_1d"):
    import torch
    df, nodes, edges = load_frames()
    feats = DYNAMIC_FEATURES + [f for f in OPTIONAL_FEATURES if f in df.columns]
    X, y, m, dates, node_ids, used = to_node_time_tensor(df, feats)
    edge_index, edge_w = build_edge_index(edges, node_ids)
    print(f"X {X.shape} (T,N,F)  edges {edge_index.shape[1]}  features {len(used)}")
    return {
        "X": torch.tensor(np.nan_to_num(X)),
        "y": torch.tensor(np.nan_to_num(y)),
        "mask": torch.tensor(m),
        "edge_index": torch.tensor(edge_index),
        "edge_weight": torch.tensor(edge_w),
        "dates": dates, "node_ids": node_ids, "features": used,
        "lookback": lookback,
    }


if __name__ == "__main__":
    df, nodes, edges = load_frames()
    print(f"Loaded flood_dataset: {df.shape}, nodes {len(nodes)}, edges {len(edges)}")
    print("Dynamic features present:",
          sum(f in df.columns for f in DYNAMIC_FEATURES), "/", len(DYNAMIC_FEATURES))
    print("Optional (soil/humidity) present:",
          sum(f in df.columns for f in OPTIONAL_FEATURES), "/", len(OPTIONAL_FEATURES))
