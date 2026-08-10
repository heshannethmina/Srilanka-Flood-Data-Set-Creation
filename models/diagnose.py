"""Read `*_diag.json` and say what to change next.

    python models/diagnose.py runs/                 # every run found
    python models/diagnose.py runs/N3_temporal_diag.json
    python models/diagnose.py runs/ --episodes      # also list the worst misses

Each section ends in a verdict rather than a number, because the point of the
diagnostics is to pick the next experiment, not to add rows to a table. The
verdicts are deliberately conservative: they flag a condition, they do not claim
the fix will work.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def _load(path: str) -> List[Dict]:
    if os.path.isfile(path):
        return [json.load(open(path))]
    out = []
    for p in sorted(glob.glob(os.path.join(path, "*_diag.json"))):
        try:
            out.append(json.load(open(p)))
        except json.JSONDecodeError:
            print(f"[warn] unreadable: {p}")
    return out


def _fmt(v, dec=4):
    return "  --" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{dec}f}"


# ------------------------------------------------------------------ sections

def seeds(d: Dict) -> List[str]:
    s = d.get("seed_spread") or {}
    if not isinstance(s, dict) or "pr_auc" not in s:
        return []
    pr = s["pr_auc"]
    out = [f"  seed spread (n={pr['n_seeds']}):"]
    for k, v in s.items():
        out.append(f"    {k:22s} mean {v['mean']:.4f}  sd {v['std']:.4f}  "
                   f"range [{v['min']:.4f}, {v['max']:.4f}]")
    if pr["n_seeds"] > 1:
        # Two rungs separated by less than roughly two standard deviations of
        # seed noise are not distinguishable by this experiment.
        out.append(f"    -> differences below ~{2 * pr['std']:.4f} PR-AUC are "
                   f"inside seed noise; do not claim them")
    else:
        out.append("    -> single seed: no error bar, so no rung comparison is "
                   "safe yet")
    return out


def stopping(d: Dict) -> List[str]:
    s = d.get("stopping") or {}
    if not isinstance(s, dict) or "epochs_run" not in s:
        return []
    out = [f"  early stopping: ran {s['epochs_run']} epochs, best at "
           f"{s['best_epoch']} (+{s['epochs_after_best']} after)"]
    out.append(f"    selection slope (episode PR-AUC) {s['val_event_pr_auc_slope_tail']:+.5f} "
               f"/epoch | node-day {s['val_pr_auc_slope_tail']:+.5f}")
    if s.get("still_improving_on_node_day"):
        out.append("    -> STOPPED EARLY on a noisier signal: node-day PR-AUC was "
                   "still rising while")
        out.append("       episode PR-AUC had flattened. Try patience=15-20, or "
                   "select on a blend.")
    elif s.get("stopped_on_patience"):
        out.append("    -> patience fired with both metrics flat: converged, "
                   "more epochs will not help")
    else:
        out.append("    -> ran to the epoch budget; if both slopes are positive, "
                   "raise `epochs`")
    return out


def optimisation(d: Dict) -> List[str]:
    o = d.get("optimisation") or {}
    if not isinstance(o, dict) or "grad_norm_mean" not in o:
        return []
    out = [f"  gradients: |g| mean {o['grad_norm_mean']:.3f} "
           f"(first epoch {o['grad_norm_first_epoch']:.3f}, "
           f"last {o['grad_norm_last_epoch']:.3f}), clip={o['grad_clip']}"]
    out.append(f"    clipped on {o['clip_fraction_mean']:.1%} of steps "
               f"({o['clip_fraction_first_epoch']:.1%} in epoch 0)")
    out.append(f"    train loss {o['train_loss_first']:.5f} -> {o['train_loss_last']:.5f}")
    if o.get("clip_is_binding"):
        out.append("    -> the clip is doing the optimiser's job on most steps: "
                   "lower `lr` or raise `grad_clip`")
    elif o.get("clip_is_noop"):
        out.append("    -> the clip almost never fires; it is not protecting "
                   "anything")
    out.append("    NOTE loss magnitudes are not comparable across loss "
               "functions, so an identical")
    out.append("         `lr` is not an identical step size between a BCE rung "
               "and a focal one.")
    return out


def heads(d: Dict) -> List[str]:
    h = d.get("head_balance") or {}
    if not isinstance(h, dict) or not h:
        return []
    base = {k: v for k, v in h.items() if not k.endswith("__reduction")}
    red = {k[:-11]: v for k, v in h.items() if k.endswith("__reduction")}
    out = ["  per-head loss (mean over the last fifth of training):"]
    for k, v in sorted(base.items(), key=lambda kv: -kv[1]):
        out.append(f"    {k:22s} {v:.5f}   reduced by {red.get(k, float('nan')):.5f} "
                   f"since epoch 0")
    weak = [k for k, v in red.items() if np.isfinite(v) and v <= 0]
    if weak:
        out.append(f"    -> {', '.join(weak)} did not improve at all: they are "
                   "consuming capacity")
        out.append("       for nothing. Try dropping their weight in "
                   "`head_weights` to 0.")
    return out


def episodes(d: Dict, show: int = 0) -> List[str]:
    eps = d.get("episodes")
    if not isinstance(eps, list) or not eps:
        return []
    n = len(eps)
    c = {k: sum(e["outcome"] == k for e in eps) for k in ("early", "late", "never")}
    out = [f"  episodes (n={n}): early {c['early']} ({c['early']/n:.1%})  "
           f"late {c['late']} ({c['late']/n:.1%})  never {c['never']} "
           f"({c['never']/n:.1%})"]

    missed = [e for e in eps if e["outcome"] != "early"]
    if missed:
        mp = np.array([e.get("max_prob_pre_onset", np.nan) for e in missed], float)
        mp = mp[np.isfinite(mp)]
        if mp.size:
            near = float((mp >= 0.5 * d.get("threshold", 0.5)).mean())
            out.append(f"    of the {len(missed)} not warned early, best pre-onset "
                       f"probability: median {np.median(mp):.4f}, "
                       f"90th pct {np.quantile(mp, 0.9):.4f}")
            out.append(f"    {near:.1%} of them got at least half the decision "
                       f"threshold ({d.get('threshold', 0.5):.4f})")
            if near > 0.5:
                out.append("    -> mostly THRESHOLD misses, not blind ones: the "
                           "signal was there but ranked")
                out.append("       too low. Attack the threshold/calibration or "
                           "the lead-time weighting.")
            else:
                out.append("    -> mostly BLIND misses: no pre-onset signal at "
                           "all. Attack features or")
                out.append("       lookback, not the loss.")

    sev = d.get("by_severity")
    if isinstance(sev, list) and sev:
        out.append("    detection by peak-discharge quartile:")
        for q in sev:
            out.append(f"      Q{q['quartile']} (n={q['n']:4d})  early {q['early']:.1%}"
                       f"  late {q['late']:.1%}  never {q['never']:.1%}")
        top, bot = sev[-1], sev[0]
        if top["early"] < bot["early"]:
            out.append("    -> WORSE on the largest floods than the smallest. "
                       "That is the wrong way round")
            out.append("       for an early-warning system and is the finding to "
                       "chase first.")

    if show:
        worst = sorted([e for e in eps if e["outcome"] == "never"],
                       key=lambda e: -(e.get("peak_discharge") or 0))[:show]
        if worst:
            out.append(f"    worst never-warned episodes:")
            for e in worst:
                out.append(f"      {str(e.get('event_id'))[:24]:24s} "
                           f"node {str(e.get('node_id')):10s} "
                           f"dur {e.get('duration_days')}  "
                           f"peak {_fmt(e.get('peak_discharge'), 1)}  "
                           f"max p {_fmt(e.get('max_prob_pre_onset'))}")
    return out


def geography(d: Dict) -> List[str]:
    out: List[str] = []
    for by in ("zone", "position"):
        g = d.get(f"by_{by}")
        if not isinstance(g, list) or not g:
            continue
        out.append(f"  by {by}:")
        for r in sorted(g, key=lambda r: r["csi"]):
            out.append(f"    {str(r[by])[:14]:14s} nodes {r['n_nodes']:3d}  "
                       f"pos {r['n_pos']:6d}  POD {r['pod']:.3f}  "
                       f"FAR {r['far']:.3f}  CSI {r['csi']:.3f}")
        worst, best = g[0], g[-1]
        gap = max(x["csi"] for x in g) - min(x["csi"] for x in g)
        if gap > 0.15:
            lo = min(g, key=lambda r: r["csi"])
            out.append(f"    -> CSI varies by {gap:.2f} across {by}; weakest is "
                       f"'{lo[by]}'. A single global")
            out.append(f"       model may be the wrong shape — consider "
                       f"conditioning on {by}.")
    nodes = d.get("nodes")
    if isinstance(nodes, list) and nodes:
        ranked = sorted([n for n in nodes if n.get("n_pos", 0) >= 10],
                        key=lambda n: n["csi"])[:5]
        if ranked:
            out.append("  weakest nodes (>=10 positives):")
            for n in ranked:
                out.append(f"    {str(n.get('node_id')):10s} "
                           f"{str(n.get('zone', '')):13s} "
                           f"{str(n.get('position', '')):11s} "
                           f"pos {n['n_pos']:4d}  CSI {n['csi']:.3f}  "
                           f"missed {n['fn']:4d}")
    return out


def gate(d: Dict) -> List[str]:
    g = d.get("sar_gate_mean")
    if g is None:
        return []
    out = [f"  SAR fusion gate: mean opening {g:.4f} on node-days with a frame"]
    if g < 0.10:
        out.append("    -> the gate stayed essentially shut. The imagery branch "
                   "is not contributing;")
        out.append("       any metric difference is noise or capacity, not "
                   "vision.")
    elif g > 0.5:
        out.append("    -> the gate is wide open; the model is leaning on the "
                   "imagery where it exists")
    return out


# --------------------------------------------------------------------- main

def report(d: Dict, show_episodes: int = 0) -> str:
    n = d.get("per_seed") or []
    head = (f"{d.get('family', '?')} · {d.get('preset', '?')} "
            f"[{d.get('protocol', '?')}] · {len(n)} seed(s) · "
            f"threshold {d.get('threshold', float('nan')):.4f}")
    lines = ["=" * 78, head, "=" * 78]
    for fn in (seeds, stopping, optimisation, heads):
        block = fn(d)
        if block:
            lines += block + [""]
    for fn in (lambda x: episodes(x, show_episodes), geography, gate):
        block = fn(d)
        if block:
            lines += block + [""]
    return "\n".join(lines).rstrip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="runs",
                    help="a runs/ directory or a single *_diag.json")
    ap.add_argument("--episodes", type=int, nargs="?", const=8, default=0,
                    help="also list the N worst never-warned episodes")
    a = ap.parse_args()

    ds = _load(a.path)
    if not ds:
        raise SystemExit(
            f"no *_diag.json under {a.path!r}. They are written next to each "
            "result JSON by runs from this version onward — earlier runs have "
            "only the metrics.")
    for d in ds:
        print(report(d, a.episodes))
        print()


if __name__ == "__main__":
    main()
