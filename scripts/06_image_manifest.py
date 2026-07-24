"""
Step 6 - Build the IMAGE SAMPLING MANIFEST for the vision branch (Branch B).

Uses the discharge-derived labels already in flood_dataset.parquet to specify
exactly which satellite chips to fetch and how they are labeled. No imagery is
downloaded here -- this only produces the request list that a fetcher
(07_fetch_images_*.py) consumes.

Each manifest row = one chip request:
  node_id, lat, lon, basin, target_date, window_start/end, label, severity,
  purpose, event_id.

Sample types (purpose):
  flood_peak   -> image nearest the discharge peak of a flood event   (label 1)
  pre_event    -> image ~lead_days BEFORE onset (antecedent state,
                  label = will-flood, for the "predict before" framing) (label 1)
  dry_negative -> image on a calm, low-discharge day                  (label 0)

Output: data/processed/image_manifest.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

S1_START = "2015-01-01"     # Sentinel-1 operational coverage
CHIP_HALF_KM = 3.2          # half-size of the chip footprint (=> ~6.4 km box)
WINDOW_DAYS = 6             # +/- days to search for an actual satellite pass
LEAD_DAYS = 3              # pre-event lead time for the antecedent chip
NEG_PER_POS = 1.5          # negatives per positive
RNG = np.random.default_rng(42)


def severity_of(row):
    if row["flood_severe"] == 1:
        return "severe"
    if row["flood_high"] == 1:
        return "high"
    if row["flood_moderate"] == 1:
        return "moderate"
    return "none"


def main():
    df = pd.read_parquet(os.path.join(C.PROC_DIR, "flood_dataset.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    nodes = pd.read_csv(os.path.join(C.PROC_DIR, "nodes.csv")).set_index("node_id")
    df = df[df["date"] >= pd.Timestamp(S1_START)].copy()

    rows = []
    sid = 0

    # ---- positives: one chip at each event's discharge peak (+ a pre-event chip) ----
    flood = df[df["event_id"].notna()].copy()
    for eid, g in flood.groupby("event_id"):
        g = g.sort_values("date")
        peak = g.loc[g["river_discharge"].idxmax()]
        nid = peak["node_id"]
        lat, lon = nodes.loc[nid, "lat"], nodes.loc[nid, "lon"]
        sev = severity_of(peak)
        # flood-peak positive
        rows.append(dict(sample_id=f"S{sid:06d}", node_id=nid, lat=lat, lon=lon,
            basin=peak["basin"], target_date=peak["date"].date(),
            window_start=(peak["date"] - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end=(peak["date"] + pd.Timedelta(days=WINDOW_DAYS)).date(),
            label=1, severity=sev, purpose="flood_peak", event_id=eid,
            chip_half_km=CHIP_HALF_KM)); sid += 1
        # pre-event antecedent positive (image BEFORE the flood -> prediction framing)
        onset = g["date"].min() - pd.Timedelta(days=LEAD_DAYS)
        rows.append(dict(sample_id=f"S{sid:06d}", node_id=nid, lat=lat, lon=lon,
            basin=peak["basin"], target_date=onset.date(),
            window_start=(onset - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end=(onset + pd.Timedelta(days=2)).date(),
            label=1, severity=sev, purpose="pre_event", event_id=eid,
            chip_half_km=CHIP_HALF_KM)); sid += 1

    n_pos = len(rows)

    # ---- negatives: calm low-discharge days, spread across nodes/time ----
    dry = df[(df["flood_state"] == 0) & (df["discharge_pctl"] < 0.5)].copy()
    n_neg = int(n_pos * NEG_PER_POS)
    # stratify by node so every node contributes negatives
    per_node = max(1, n_neg // df["node_id"].nunique())
    neg_pick = (dry.sample(frac=1, random_state=42)      # shuffle
                   .groupby("node_id", group_keys=False)
                   .head(per_node))                       # cap per node, keeps columns
    neg_pick = neg_pick.sample(min(n_neg, len(neg_pick)), random_state=42)
    for _, r in neg_pick.iterrows():
        nid = r["node_id"]
        lat, lon = nodes.loc[nid, "lat"], nodes.loc[nid, "lon"]
        d = r["date"]
        rows.append(dict(sample_id=f"S{sid:06d}", node_id=nid, lat=lat, lon=lon,
            basin=r["basin"], target_date=d.date(),
            window_start=(d - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end=(d + pd.Timedelta(days=WINDOW_DAYS)).date(),
            label=0, severity="none", purpose="dry_negative", event_id="",
            chip_half_km=CHIP_HALF_KM)); sid += 1

    man = pd.DataFrame(rows)
    # attach the matching tabular-sample key so the two branches align 1:1
    man["tabular_key"] = man["node_id"].astype(str) + "_" + man["target_date"].astype(str)
    out = os.path.join(C.PROC_DIR, "image_manifest.csv")
    man.to_csv(out, index=False)
    print(f"Image manifest: {len(man)} chip requests "
          f"({(man.label==1).sum()} positive / {(man.label==0).sum()} negative)")
    print(f"  purposes: {man.purpose.value_counts().to_dict()}")
    print(f"  severity (positives): {man[man.label==1].severity.value_counts().to_dict()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
