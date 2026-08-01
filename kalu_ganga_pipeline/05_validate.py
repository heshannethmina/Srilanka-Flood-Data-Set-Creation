"""
Step 5 - Validate discharge-based flood labels against documented Kalu Ganga
flood events, and produce a concise EDA report.

Output: data/processed/validation_report.md
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

# Documented Kalu Ganga / SW-zone flood events (public records: DMC / ReliefWeb / news)
KNOWN_EVENTS = [
    ("May 2003 Ratnapura floods",              "2003-05-15", "2003-05-22", ["Kalu"]),
    ("Jun 2014 Kalu floods",                   "2014-06-01", "2014-06-10", ["Kalu"]),
    ("May 2016 SW monsoon floods",             "2016-05-14", "2016-05-22", ["Kalu"]),
    ("May 2017 SW floods (Ratnapura/Kalutara)","2017-05-25", "2017-06-02", ["Kalu"]),
    ("May 2018 SW floods",                     "2018-05-19", "2018-05-28", ["Kalu"]),
    ("Dec 2019 Kalu floods",                   "2019-12-01", "2019-12-12", ["Kalu"]),
    ("Nov 2021 Western / Kalu floods",         "2021-11-08", "2021-11-16", ["Kalu"]),
    ("Jun 2024 SW / Kalu floods",              "2024-06-01", "2024-06-12", ["Kalu"]),
]


def main():
    df = pd.read_parquet(os.path.join(C.PROC_DIR, "flood_dataset.parquet"))
    df["date"] = pd.to_datetime(df["date"])

    lines = []
    def w(s=""): lines.append(s)

    w("# Validation & EDA Report — Kalu Ganga Flood Dataset\n")
    w(f"Nodes: {sorted(df['node_id'].unique())}  |  "
      f"{df['date'].min().date()} -> {df['date'].max().date()}\n")

    # ── overall stats ──────────────────────────────────────────────────────────
    v = df[df["valid_sample"] == 1]
    w("## 1. Dataset size & class balance\n")
    w(f"- Total node-days      : **{len(df):,}**")
    w(f"- Valid supervised rows: **{len(v):,}**")
    w(f"- Flood-state prevalence ('{C.PRIMARY_SEVERITY}'): **{df['flood_state'].mean()*100:.2f}%**")
    for h in C.FLOOD_HORIZONS_DAYS:
        w(f"- Positive rate target_flood_{h}d (valid): **{v[f'target_flood_{h}d'].mean()*100:.2f}%**")
    w("")

    # ── split sizes ───────────────────────────────────────────────────────────
    w("## 2. Split sizes (valid samples)\n")
    for k, val in v["split_temporal"].value_counts().sort_index().items():
        w(f"  - {k}: {val:,}")
    w("")

    # ── per-node stats ────────────────────────────────────────────────────────
    w("## 3. Per-node flood statistics\n")
    w("| Node | Position | Flood-days | Node-days | Rate % | Thr_high m3/s |")
    w("|---|---|---|---|---|---|")
    thr_path = os.path.join(C.PROC_DIR, "thresholds.json")
    import json
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for nid, g in df.groupby("node_id"):
        fd   = int(g["flood_state"].sum())
        nd   = len(g)
        rate = fd / nd * 100
        thr  = thresholds.get(nid, {}).get("high", float("nan"))
        pos  = g["position"].iloc[0]
        w(f"| {nid} | {pos} | {fd} | {nd} | {rate:.2f} | {thr:.1f} |")
    w("")

    # ── documented event detection ────────────────────────────────────────────
    w("## 4. Label validation vs documented Kalu Ganga flood events\n")
    w("Detection window widened to ±4 days (downstream discharge lags rain).\n")
    w("| Event | Fired@high | Fired@moderate | Max pctl | Detected |")
    w("|---|---|---|---|---|")
    hits_high = hits_mod = 0
    for name, s, e, basins in KNOWN_EVENTS:
        s   = pd.Timestamp(s) - pd.Timedelta(days=4)
        e   = pd.Timestamp(e) + pd.Timedelta(days=4)
        sub = df[(df["date"] >= s) & (df["date"] <= e) & (df["basin"].isin(basins))]
        if sub.empty:
            w(f"| {name} | (no data) | – | – | – |"); continue
        fh  = sub[sub["flood_high"]     == 1]["node_id"].nunique()
        fm  = sub[sub["flood_moderate"] == 1]["node_id"].nunique()
        tot = sub["node_id"].nunique()
        maxp = sub["discharge_pctl"].max()
        if fh > 0: hits_high += 1
        if fm > 0: hits_mod  += 1
        flag = "✅" if fh > 0 else ("◧" if fm > 0 else "⚠️")
        w(f"| {name} | {fh}/{tot} | {fm}/{tot} | {maxp:.3f} | {flag} |")
    w(f"\n**Detection: {hits_high}/{len(KNOWN_EVENTS)} at high threshold, "
      f"{hits_mod}/{len(KNOWN_EVENTS)} at moderate threshold.**\n")

    # ── missingness ───────────────────────────────────────────────────────────
    w("## 5. Feature missingness (top columns with gaps)\n")
    miss = (df.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False).head(10)
    if len(miss):
        w("| Column | % missing |")
        w("|---|---|")
        for c, m in miss.items():
            w(f"| {c} | {m} |")
    else:
        w("No missing values detected.")

    out = os.path.join(C.PROC_DIR, "validation_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Validation: {hits_high}/{len(KNOWN_EVENTS)} events detected -> {out}")


if __name__ == "__main__":
    main()
