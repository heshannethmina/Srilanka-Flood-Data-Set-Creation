"""Render a results table from the JSON a run leaves in `runs/`.

Separate from `kaggle_run.py` on purpose: that only runs on Kaggle, and a table
is something you want offline, after downloading `runs.zip`, while writing up.

    python model/report.py runs/                    # every result found
    python model/report.py runs/ --protocol temporal # one protocol
    python model/report.py runs/ --csv table.csv     # for a paper

Files it understands, all written by `kaggle_run.py`:
  baselines_<protocol>.json   several baselines in one file
  <preset>_<protocol>.json    one model, e.g. M2_temporal.json
  <preset>_<protocol>_s5.json a re-run at an overridden seed count
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

#: (json key, header, width, decimals). Order here is the column order.
COLUMNS = [
    ("pr_auc", "PR-AUC", 8, 4), ("roc_auc", "ROC-AUC", 8, 4),
    ("brier", "Brier", 9, 5), ("ece", "ECE", 7, 4),
    ("pod", "POD", 6, 3), ("far", "FAR", 6, 3), ("csi", "CSI", 6, 3),
    ("event_detection_rate", "ev.det", 7, 3), ("mean_lead_days", "lead", 6, 2),
]


def collect(run_dir: str, protocol: Optional[str] = None) -> List[Dict]:
    """One row per model found, tagged with the protocol it was evaluated under."""
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if "baseline" in os.path.basename(path):
            # A baselines file holds several models keyed by name.
            proto = os.path.basename(path).replace("baselines_", "").replace(".json", "")
            for name, v in d.items():
                rows.append({"model": v.get("baseline", name), "protocol": proto,
                             "group": "baselines", **v["test"]})
        else:
            n = d.get("train_config", {}).get("n_seeds", 1)
            # A 5-seed run and a 1-seed run of the same preset are different
            # models; the label has to say so or the table silently conflates them.
            label = d["preset"] + (f" x{n}" if n and n > 1 else "")
            rows.append({"model": label, "protocol": d["protocol"],
                         "group": "model", **d["test"]})
    if protocol:
        rows = [r for r in rows if r["protocol"] == protocol]
    return rows


def render(rows: List[Dict], title: str = "TF-STGNN EVALUATION SUMMARY") -> str:
    header = f"{'Model':<20s}{'Protocol':<11s}" + "".join(
        f"{h:>{w}s}" for _, h, w, _ in COLUMNS)
    rule = "=" * len(header)
    out = [rule, title.center(len(header)), rule, header, "-" * len(header)]

    # Baselines first: they are the bar, so they belong above what clears it.
    for group in ("baselines", "model"):
        block = [r for r in rows if r["group"] == group]
        if not block:
            continue
        if group == "model" and out[-1] != "-" * len(header):
            out.append("")
        for r in sorted(block, key=lambda r: (r["protocol"], r["model"])):
            line = f"{r['model'][:19]:<20s}{r['protocol'][:10]:<11s}"
            for key, _, w, dec in COLUMNS:
                v = r.get(key)
                # A baseline with no event metrics must read as absent, not zero.
                line += f"{'--':>{w}s}" if v is None else f"{v:>{w}.{dec}f}"
            out.append(line)
    out.append(rule)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="runs")
    ap.add_argument("--protocol", help="temporal | basin | random")
    ap.add_argument("--csv", help="also write the table as CSV")
    ap.add_argument("--title", default="TF-STGNN EVALUATION SUMMARY")
    a = ap.parse_args()

    rows = collect(a.run_dir, a.protocol)
    if not rows:
        raise SystemExit(f"no result JSON under {a.run_dir!r} — download "
                         "runs.zip from the Kaggle notebook output and unzip it")
    print(render(rows, a.title))

    if a.csv:
        import csv
        keys = ["model", "protocol"] + [k for k, _, _, _ in COLUMNS]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
