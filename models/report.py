"""Render a results table from the JSON a run leaves in `runs/`.

Separate from `kaggle_run.py` on purpose: that only runs on Kaggle, and a table
is something you want offline, after downloading `runs.zip`, while writing up.

    python models/report.py runs/                     # every result found
    python models/report.py runs/ --protocol temporal # one protocol
    python models/report.py runs/ --csv table.csv     # for a paper

Files it understands, all written by `kaggle_run.py`:
  baselines_<protocol>.json   several baselines in one file
  <preset>_<protocol>.json    one model, e.g. M2_temporal.json or N5_temporal.json
  <preset>_<protocol>_s5.json a re-run at an overridden seed count

Rows are grouped baselines → model 1 → model 2 → model 3, because the baselines
are the bar and the three families have to be read against it before they are
read against each other.

This prints *what each model scored*. For the document you make decisions and
paper claims from — seed error bars, paired confidence intervals on the
differences, and event detection at a matched false-alarm ratio — use
`results_doc.py` instead.
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

#: Raw contingency counts, shown only with `--counts`. Off by default so the
#: headline table stays the width it has been quoted at; `missed` and `f.alarm`
#: are the two an operator actually reads.
COUNT_COLUMNS = [
    ("fn", "missed", 8), ("fp", "f.alarm", 9),
    ("tp", "hits", 7), ("tn", "nulls", 9),
]

#: Print order. A group absent from the run directory is simply skipped.
#: ASCII only: this table is printed to Kaggle logs and to Windows consoles,
#: and a cp1252 terminal turns an em dash into a replacement character.
GROUPS = [("baselines", "BASELINES"), ("model1", "MODEL 1: TF-STGNN"),
          ("model2", "MODEL 2: MMF-Net"), ("model3", "MODEL 3: STG-Former")]

#: Short names for `TrainConfig.eval_head`, mirroring `floodlib.schema.CLS_HEADS`.
#: Duplicated rather than imported on purpose: this file is run against a
#: downloaded `runs/` directory from wherever it happens to be, with no
#: guarantee that `models/` is on the path.
_HEADS = ["flood_1d", "flood_2d", "flood_3d", "onset_1d"]

#: Model-name column width. Wide enough for the longest preset plus a
#: seed count and an eval-head marker ("P4_onset_ctrl x5 [onset_1d]"),
#: because a truncated marker is worse than none -- it hides that the row
#: is scored on a different target.
_MODEL_W = 30


def _group_of(d: Dict) -> str:
    """Which family a result belongs to.

    `family` has been written into every result since the two-model split; the
    preset-prefix fallback keeps JSON from the earlier single-model runs
    readable rather than silently ungrouped.
    """
    fam = d.get("family", "")
    if fam:
        if "STG-Former" in fam:
            return "model3"
        return "model2" if "MMF" in fam else "model1"
    p = str(d.get("preset", ""))
    if p.startswith("P"):
        return "model3"
    return "model2" if p.startswith("N") else "model1"


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
                             "group": "baselines", "n_params": None, **v["test"]})
        elif "preset" in d and "test" in d:
            n = d.get("train_config", {}).get("n_seeds", 1)
            # A 5-seed run and a 1-seed run of the same preset are different
            # models; the label has to say so or the table silently conflates them.
            label = d["preset"] + (f" x{n}" if n and n > 1 else "")
            # A row scored on onset and a row scored on flood_1d share the
            # PR-AUC column but are not comparable in it -- onset is far sparser
            # and its numbers are much smaller. Mark it in the label so the two
            # cannot be read down the same column by accident.
            eh = d.get("train_config", {}).get("eval_head", 0)
            if eh:
                label += f" [{_HEADS[eh]}]"
            rows.append({"model": label, "protocol": d["protocol"],
                         "group": _group_of(d),
                         "n_params": d.get("n_params"), **d["test"]})
    if protocol:
        rows = [r for r in rows if r["protocol"] == protocol]
    return rows


def _fmt_params(v) -> str:
    return "      --" if not v else f"{v / 1e3:>7.0f}k"


def render(rows: List[Dict], title: str = "FLOOD EARLY-WARNING EVALUATION SUMMARY",
           show_params: bool = True, show_counts: bool = False) -> str:
    counts = COUNT_COLUMNS if show_counts else []
    width = (_MODEL_W + 11 + sum(w for _, _, w, _ in COLUMNS)
             + sum(w for _, _, w in counts) + (8 if show_params else 0))
    header = (f"{'Model':<{_MODEL_W}s}{'Protocol':<11s}"
              + "".join(f"{h:>{w}s}" for _, h, w, _ in COLUMNS)
              + "".join(f"{h:>{w}s}" for _, h, w in counts)
              + (f"{'params':>8s}" if show_params else ""))
    rule = "=" * width
    out = [rule, title.center(width), rule, header, "-" * width]

    # Baselines first: they are the bar, so they belong above what clears it.
    for group, heading in GROUPS:
        block = [r for r in rows if r["group"] == group]
        if not block:
            continue
        if out[-1] != "-" * width:
            out.append("")
        out.append(f"-- {heading} " + "-" * max(width - len(heading) - 4, 0))
        for r in sorted(block, key=lambda r: (r["protocol"], r["model"])):
            line = (f"{r['model'][:_MODEL_W - 1]:<{_MODEL_W}s}"
                    f"{r['protocol'][:10]:<11s}")
            for key, _, w, dec in COLUMNS:
                v = r.get(key)
                # A baseline with no event metrics must read as absent, not zero.
                line += f"{'--':>{w}s}" if v is None else f"{v:>{w}.{dec}f}"
            for key, _, w in counts:
                v = r.get(key)
                line += f"{'--':>{w}s}" if v is None else f"{int(v):>{w}d}"
            if show_params:
                line += _fmt_params(r.get("n_params"))
            out.append(line)
    out.append(rule)
    if show_counts:
        out.append("missed/f.alarm are node-day counts and flatter the model: most "
                   "hits are days")
        out.append("inside an ongoing flood. Use rethreshold.py for the "
                   "episode-level breakdown.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="runs")
    ap.add_argument("--protocol", help="temporal | basin | random")
    ap.add_argument("--csv", help="also write the table as CSV")
    ap.add_argument("--title", default="FLOOD EARLY-WARNING EVALUATION SUMMARY")
    ap.add_argument("--no-params", action="store_true",
                    help="drop the parameter-count column")
    ap.add_argument("--counts", action="store_true",
                    help="add the raw contingency counts: missed floods (FN), "
                         "false alarms (FP), hits (TP), correct nulls (TN)")
    a = ap.parse_args()

    rows = collect(a.run_dir, a.protocol)
    if not rows:
        raise SystemExit(f"no result JSON under {a.run_dir!r} — download "
                         "runs.zip from the Kaggle notebook output and unzip it")
    print(render(rows, a.title, show_params=not a.no_params, show_counts=a.counts))

    if a.csv:
        import csv
        keys = (["model", "protocol", "group", "n_params"]
                + [k for k, _, _, _ in COLUMNS] + [k for k, _, _ in COUNT_COLUMNS])
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
