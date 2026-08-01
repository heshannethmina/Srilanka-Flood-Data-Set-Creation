"""
Step 6 - Build the IMAGE SAMPLING MANIFEST for the Sentinel-1 branch.

Uses flood_dataset.parquet (Branch A labels) to decide which satellite chips
to request and how they are labeled. No imagery is downloaded here.

Each manifest row = one chip request:
  node_id, lat, lon, basin, target_date, window_start/end,
  label, severity, purpose, event_id, chip_half_km

Sample types (purpose):
  flood_peak   -> chip nearest the discharge peak of a flood event  (label 1)
  pre_event    -> chip LEAD_DAYS before flood onset                  (label 1)
  dry_negative -> chip on a calm, low-discharge day                  (label 0)

Output: data/processed/image_manifest.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

S1_START    = "2015-01-01"   # Sentinel-1 operational start
CHIP_HALF_KM = 3.2           # half-size => ~6.4 km chip footprint
WINDOW_DAYS  = 6             # +/- days search window for satellite pass
LEAD_DAYS    = 3             # pre-event lead time for antecedent chip
NEG_PER_POS  = 1.5           # negatives per positive sample
RNG          = np.random.default_rng(42)


def severity_of(row):
    if row["flood_severe"]   == 1: return "severe"
    if row["flood_high"]     == 1: return "high"
    if row["flood_moderate"] == 1: return "moderate"
    return "none"


def main():
    df = pd.read_parquet(os.path.join(C.PROC_DIR, "flood_dataset.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    # Only Sentinel-1 era
    df = df[df["date"] >= pd.Timestamp(S1_START)].copy()

    nodes = pd.read_csv(os.path.join(C.PROC_DIR, "nodes.csv")).set_index("node_id")

    rows = []
    sid  = 0

    # ── positives: one chip at each event's discharge peak + a pre-event chip ──
    flood = df[df["event_id"].notna()].copy()
    for eid, g in flood.groupby("event_id"):
        g    = g.sort_values("date")
        peak = g.loc[g["river_discharge"].idxmax()]
        nid  = peak["node_id"]
        lat, lon = nodes.loc[nid, "lat"], nodes.loc[nid, "lon"]
        sev  = severity_of(peak)
        # flood-peak chip
        rows.append(dict(
            sample_id=f"KAL{sid:05d}", node_id=nid, lat=lat, lon=lon,
            basin=peak["basin"], target_date=peak["date"].date(),
            window_start=(peak["date"] - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end  =(peak["date"] + pd.Timedelta(days=WINDOW_DAYS)).date(),
            label=1, severity=sev, purpose="flood_peak", event_id=eid,
            chip_half_km=CHIP_HALF_KM)); sid += 1
        # pre-event antecedent chip
        onset = g["date"].min() - pd.Timedelta(days=LEAD_DAYS)
        rows.append(dict(
            sample_id=f"KAL{sid:05d}", node_id=nid, lat=lat, lon=lon,
            basin=peak["basin"], target_date=onset.date(),
            window_start=(onset - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end  =(onset + pd.Timedelta(days=2)).date(),
            label=1, severity=sev, purpose="pre_event", event_id=eid,
            chip_half_km=CHIP_HALF_KM)); sid += 1

    n_pos = len(rows)

    # ── negatives: calm low-discharge days spread across nodes ──────────────
    dry   = df[(df["flood_state"] == 0) & (df["discharge_pctl"] < 0.5)].copy()
    n_neg = int(n_pos * NEG_PER_POS)
    per_node = max(1, n_neg // df["node_id"].nunique())
    neg_pick = (dry.sample(frac=1, random_state=42)
                   .groupby("node_id", group_keys=False)
                   .head(per_node))
    neg_pick = neg_pick.sample(min(n_neg, len(neg_pick)), random_state=42)
    for _, r in neg_pick.iterrows():
        nid      = r["node_id"]
        lat, lon = nodes.loc[nid, "lat"], nodes.loc[nid, "lon"]
        d        = r["date"]
        rows.append(dict(
            sample_id=f"KAL{sid:05d}", node_id=nid, lat=lat, lon=lon,
            basin=r["basin"], target_date=d.date(),
            window_start=(d - pd.Timedelta(days=WINDOW_DAYS)).date(),
            window_end  =(d + pd.Timedelta(days=WINDOW_DAYS)).date(),
            label=0, severity="none", purpose="dry_negative", event_id="",
            chip_half_km=CHIP_HALF_KM)); sid += 1

    man = pd.DataFrame(rows)
    man["tabular_key"] = (man["node_id"].astype(str) + "_"
                          + man["target_date"].astype(str))
    out = os.path.join(C.PROC_DIR, "image_manifest.csv")
    man.to_csv(out, index=False)
    print(f"Image manifest: {len(man)} chip requests")
    print(f"  positive (label=1): {(man.label==1).sum()}"
          f"  negative (label=0): {(man.label==0).sum()}")
    print(f"  purposes: {man.purpose.value_counts().to_dict()}")
    print(f"  severity (positives): "
          f"{man[man.label==1].severity.value_counts().to_dict()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
