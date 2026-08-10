"""Compare models at a **matched false-alarm ratio**, offline, from `*_preds.npz`.

    python models/rethreshold.py runs/ --far 0.231
    python models/rethreshold.py runs/ --far 0.231 --csv matched.csv

Why this is necessary
---------------------
Each run picks its own decision threshold on validation, so the models in a
results table are not sitting at the same operating point. The first model 2
ladder made that unmissable: N2 reports event detection 0.423 at FAR 0.313 and
N3 reports 0.197 at FAR 0.110. Read as a column, N3 looks like a catastrophic
regression in early warning. It is not — N3 is simply operating far more
conservatively, and its threshold-free PR-AUC went *up* (0.774 -> 0.827). The
ev.det column is comparing operating points, not models.

This script removes that confound: it re-derives each model's threshold so that
every model raises false alarms at the same rate, then reports what each one
detects there. An operator picks a false-alarm budget first and asks what they
get for it; this is that question.

Caveat, stated plainly: the threshold is fitted on the test predictions, so
these numbers are a **like-for-like diagnostic, not a held-out result**. The
headline table stays the one in `report.py`, whose thresholds come from
validation only. Quote this alongside it, never instead of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floodlib.metrics import contingency, episode_outcomes, evaluate   # noqa: E402


def threshold_for_far(y: np.ndarray, p: np.ndarray, target_far: float,
                      n_grid: int = 512) -> float:
    """The threshold whose false-alarm ratio is closest to `target_far`.

    Searched over probability quantiles rather than a uniform grid: after
    calibration the probabilities pile up near zero, and a uniform grid spends
    almost all its points in a region no operator would ever threshold at.
    """
    qs = np.unique(np.quantile(p, np.linspace(0.50, 0.99999, n_grid)))
    best_t, best_gap = 0.5, np.inf
    for t in qs:
        far = contingency(y, p, float(t)).get("far", np.nan)
        if not np.isfinite(far):
            continue
        gap = abs(far - target_far)
        if gap < best_gap:
            best_t, best_gap = float(t), gap
    return best_t


def collect(run_dir: str, protocol: Optional[str] = "temporal") -> List[Dict]:
    """Every model in `run_dir` that saved its test predictions."""
    rows: List[Dict] = []
    for npz_path in sorted(glob.glob(os.path.join(run_dir, "*_preds.npz"))):
        stem = os.path.basename(npz_path).replace("_preds.npz", "")
        meta_path = os.path.join(run_dir, stem + ".json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if protocol and meta.get("protocol") != protocol:
            continue
        d = np.load(npz_path)
        n = meta.get("train_config", {}).get("n_seeds", 1)
        rows.append({
            "model": meta["preset"] + (f" x{n}" if n and n > 1 else ""),
            "family": meta.get("family", ""),
            "protocol": meta.get("protocol", "?"),
            "y": d["test_y"], "p": d["test_prob"], "day": d["test_day"],
            "node": d["test_node"], "event": d["test_event"],
            "own_threshold": meta.get("threshold"),
            "own": meta["test"],
        })
    return rows


def render(rows: List[Dict], target_far: float) -> str:
    head = (f"{'Model':<16s}{'FAR':>7s}{'POD':>7s}{'CSI':>7s}"
            f"{'missed':>8s}{'f.alarm':>8s}"
            f"{'early':>7s}{'late':>6s}{'never':>7s}"
            f"{'early%':>8s}{'never%':>8s}{'lead':>7s}")
    rule = "=" * len(head)
    out = [rule,
           f"MATCHED AT FAR = {target_far:.3f}  ({rows[0]['matched'].get('n_events', 0)} test episodes)".center(len(head)),
           rule, head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: -(r["episodes"].get("warned_early_rate") or 0)):
        m, e = r["matched"], r["episodes"]
        out.append(
            f"{r['model'][:15]:<16s}{m.get('far', np.nan):>7.3f}"
            f"{m.get('pod', np.nan):>7.3f}{m.get('csi', np.nan):>7.3f}"
            f"{m.get('fn', 0):>8d}{m.get('fp', 0):>8d}"
            f"{e.get('warned_early', 0):>7d}{e.get('warned_late', 0):>6d}"
            f"{e.get('never_warned', 0):>7d}"
            f"{e.get('warned_early_rate', np.nan):>8.3f}"
            f"{e.get('never_warned_rate', np.nan):>8.3f}"
            f"{e.get('episode_mean_lead_days', np.nan):>7.2f}")
    out.append(rule)
    out += [
        "Every model is forced to the same false-alarm rate, so these columns",
        "compare models rather than operating points.",
        "",
        "  missed/f.alarm  node-day false negatives / false positives",
        "  early           episodes with an alarm in onset-7 .. onset-1  (= ev.det)",
        "  late            no pre-onset alarm, but one during the episode",
        "  never           no alarm at all from onset-7 to the episode's end",
        "",
        "'late' is a lead-time failure; 'never' is a forecasting failure. ev.det",
        "alone cannot tell them apart, and node-day POD hides both -- it is",
        "dominated by days inside an ongoing flood, which autocorrelation makes",
        "nearly free to predict.",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="runs")
    ap.add_argument("--far", type=float, default=0.231,
                    help="target false-alarm ratio; 0.231 is the gradient-boosted "
                         "tree baseline's, which makes the comparison the one that "
                         "matters")
    ap.add_argument("--protocol", default="temporal")
    ap.add_argument("--csv", help="also write the table as CSV")
    a = ap.parse_args()

    rows = collect(a.run_dir, a.protocol)
    if not rows:
        raise SystemExit(
            f"no *_preds.npz with a matching .json under {a.run_dir!r} — unzip "
            "runs.zip from the Kaggle notebook output first")

    for r in rows:
        r["thr"] = threshold_for_far(r["y"], r["p"], a.far)
        r["matched"] = evaluate(r["y"], r["p"], r["thr"],
                                r["event"], r["day"], r["node"])
        r["episodes"] = episode_outcomes(r["p"], r["event"], r["day"], r["node"],
                                         r["thr"])
    print(render(rows, a.far))

    if a.csv:
        import csv
        matched_keys = ["far", "pod", "csi", "tp", "fp", "fn", "tn"]
        episode_keys = ["n_events", "warned_early", "warned_late", "never_warned",
                        "warned_early_rate", "never_warned_rate",
                        "episode_mean_lead_days"]
        keys = ["model", "family", "protocol", "thr"] + matched_keys + episode_keys
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({"model": r["model"], "family": r["family"],
                            "protocol": r["protocol"], "thr": r["thr"],
                            **{k: r["matched"].get(k) for k in matched_keys},
                            **{k: r["episodes"].get(k) for k in episode_keys}})
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
