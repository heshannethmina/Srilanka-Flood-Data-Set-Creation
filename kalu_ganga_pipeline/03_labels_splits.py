"""
Step 3 - Build flood labels, multi-horizon prediction targets, flood events,
and leakage-free train/val/test splits for the Kalu Ganga basin.

Flood state = GloFAS river discharge >= per-node return-period threshold
(estimated on the TRAIN period only - no data leakage).

Output: data/processed/flood_dataset.parquet
        data/processed/events.csv
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C


def add_flood_state(df, thresholds):
    df = df.sort_values(["node_id", "date"]).reset_index(drop=True)
    for sev in C.SEVERITY_PERCENTILES:
        thr = df["node_id"].map(lambda n: thresholds[n][sev])
        df[f"flood_{sev}"] = (df["river_discharge"] >= thr).astype("int8")
        df[f"thr_{sev}"]   = thr.astype("float32")
    df["flood_state"] = df[f"flood_{C.PRIMARY_SEVERITY}"]
    # label confidence: ramp 0.5 -> 1.0 outside a ±15% ambiguity band
    thr_p = df["node_id"].map(lambda n: thresholds[n][C.PRIMARY_SEVERITY])
    rel  = (df["river_discharge"] - thr_p).abs() / thr_p.replace(0, np.nan)
    conf = 0.5 + 0.5 * np.clip(rel / 0.15, 0, 1)
    df["label_confidence"] = np.where(df["river_discharge"].isna(), 0.0, conf).astype("float32")
    return df


def add_horizon_targets(df):
    parts = []
    for nid, g in df.groupby("node_id", sort=False):
        g     = g.sort_values("date").copy()
        state = g["flood_state"].to_numpy()
        n     = len(state)
        for h in C.FLOOD_HORIZONS_DAYS:
            fut = np.full(n, np.nan)
            for i in range(n):
                j0, j1 = i + 1, min(i + h, n - 1)
                if j0 <= n - 1:
                    fut[i] = state[j0:j1 + 1].max() if j1 >= j0 else np.nan
            g[f"target_flood_{h}d"] = fut
            g[f"target_onset_{h}d"] = np.where(
                (fut == 1) & (state == 0), 1.0,
                np.where(np.isnan(fut), np.nan, 0.0))
        z = g["discharge_zscore"].to_numpy()
        g["target_next1d_discharge"] = np.roll(g["river_discharge"].to_numpy(), -1)
        g.iloc[-1, g.columns.get_loc("target_next1d_discharge")] = np.nan
        fut_z = np.full(n, np.nan)
        for i in range(n):
            j1 = min(i + 3, n - 1)
            if i + 1 <= n - 1:
                fut_z[i] = np.nanmax(z[i + 1:j1 + 1])
        g["target_next3d_max_zscore"] = fut_z
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_events(df):
    """Merge consecutive flood_state days per node into events (gap tolerance 2d)."""
    events = []
    df["event_id"] = pd.NA
    for nid, g in df.groupby("node_id", sort=False):
        g   = g.sort_values("date")
        idx = g.index[g["flood_state"] == 1].tolist()
        if not idx:
            continue
        basin  = g["basin"].iloc[0]
        blocks = []
        cur    = [idx[0]]
        for k in range(1, len(idx)):
            if (g.loc[idx[k], "date"] - g.loc[idx[k - 1], "date"]).days <= 2:
                cur.append(idx[k])
            else:
                blocks.append(cur); cur = [idx[k]]
        blocks.append(cur)
        for b in blocks:
            start = g.loc[b[0], "date"]; end = g.loc[b[-1], "date"]
            peak  = g.loc[b, "river_discharge"].max()
            eid   = f"{nid}_{start.strftime('%Y%m%d')}"
            df.loc[b, "event_id"] = eid
            events.append({"event_id": eid, "node_id": nid, "basin": basin,
                           "start": start.date(), "end": end.date(),
                           "duration_days": (end - start).days + 1,
                           "peak_discharge": round(float(peak), 2)})
    return df, pd.DataFrame(events)


def add_splits(df):
    d = df["date"]
    df["split_temporal"] = np.select(
        [d <= pd.Timestamp(C.TRAIN_END), d <= pd.Timestamp(C.VAL_END)],
        ["train", "val"], default="test")
    return df


def add_validity(df):
    df   = df.sort_values(["node_id", "date"]).reset_index(drop=True)
    warm = df.groupby("node_id").cumcount() >= 30
    tgt_ok  = df[f"target_flood_{max(C.FLOOD_HORIZONS_DAYS)}d"].notna()
    disc_ok = df["river_discharge"].notna()
    df["valid_sample"] = (warm & tgt_ok & disc_ok).astype("int8")
    return df


def main():
    panel = pd.read_parquet(os.path.join(C.PROC_DIR, "panel_features.parquet"))
    with open(os.path.join(C.PROC_DIR, "thresholds.json")) as f:
        thresholds = json.load(f)

    df = add_flood_state(panel, thresholds)
    df = add_horizon_targets(df)
    df, events = add_events(df)
    df = add_splits(df)
    df = add_validity(df)

    out = os.path.join(C.PROC_DIR, "flood_dataset.parquet")
    df.to_parquet(out, index=False)
    events.to_csv(os.path.join(C.PROC_DIR, "events.csv"), index=False)

    v = df[df["valid_sample"] == 1]
    print(f"Final dataset: {df.shape[0]} node-days  ({v.shape[0]} valid samples)"
          f"  x {df.shape[1]} cols")
    print(f"Flood events : {len(events)}")
    print(f"Positive rate (24h target, valid) = {v['target_flood_1d'].mean():.4f}")
    print(f"Splits (valid): {v['split_temporal'].value_counts().to_dict()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
