"""Post-run diagnostics: the evidence needed to decide what to change next.

A results table says *what* a model scored. None of it says why, or what to do
about it. These builders produce the things that actually drive the next
decision, and every one of them exists because a specific question came up and
could not be answered from the numbers already being saved:

`episode_table`
    Which flood episodes were missed, and how bad were they? A model that only
    misses marginal events is in a completely different position from one that
    misses the severe ones, and the headline `event_detection_rate` cannot tell
    the two apart.
`node_table`
    Where does it fail — headwaters or outlets, wet zone or dry? Failure
    concentrated upstream is a lookback or feature problem; concentrated at
    outlets it is a routing problem. Different fixes.
`seed_spread`
    Are the differences between ladder rungs bigger than the noise between
    seeds? N3 (0.8269) and N6_gated (0.8310) are currently being compared with
    no idea whether that gap is real.
`stopping_report`
    Did early stopping fire while the model was still improving? N3 stopped at
    epoch 50 with validation PR-AUC still climbing, because patience watches
    *episode* PR-AUC, which is much noisier.
`optimisation_report`
    Was the gradient being clipped constantly (learning rate too high) or never
    (the clip is a no-op)? And are the losses on comparable scales? Focal runs
    at a twentieth of BCE's magnitude, so a fixed learning rate does not mean a
    fixed step size — which would make the N3 → N4 comparison partly an
    optimisation confound rather than a pure loss-shape effect.

Everything here is pure NumPy over arrays the run already has in memory, and
every caller wraps it in a try/except: a bug in a diagnostic must never destroy
a training run that has already finished.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import average_precision, contingency


# --------------------------------------------------------------- per-episode

def episode_table(p: np.ndarray, y: np.ndarray, event: np.ndarray,
                  day: np.ndarray, node: np.ndarray, threshold: float,
                  node_ids: Optional[Sequence[str]] = None,
                  event_ids: Optional[Sequence[str]] = None,
                  events_meta: Optional[Dict[str, Dict]] = None,
                  max_lead: int = 7) -> List[Dict]:
    """One row per flood episode: what the model saw, and when.

    `events_meta` maps an original `event_id` to whatever `events.csv` holds for
    it (duration, peak discharge, basin), so a miss can be read against how big
    the flood actually was.
    """
    p, y, event, day, node = (np.asarray(a).ravel() for a in (p, y, event, day, node))
    alarms: Dict[int, np.ndarray] = {}
    for nd in np.unique(node):
        sel = node == nd
        alarms[int(nd)] = np.sort(day[sel][p[sel] >= threshold])

    rows: List[Dict] = []
    for e in np.unique(event[event >= 0]):
        sel = event == e
        d = day[sel]
        onset, end = int(d.min()), int(d.max())
        nd = int(node[sel][0])
        a = alarms.get(nd, np.empty(0))
        pre = a[(a >= onset - max_lead) & (a <= onset - 1)]
        during = a[(a >= onset) & (a <= end)]

        # The model's best pre-onset probability is the diagnostic number: a
        # miss at 0.48 against a 0.50 threshold is a calibration problem, a miss
        # at 0.01 means the signal was never there at all.
        w = (node == nd) & (day >= onset - max_lead) & (day < onset)
        max_pre = float(p[w].max()) if w.any() else float("nan")

        code = int(e)
        row = {
            "event_code": code,
            "event_id": (event_ids[code] if event_ids and code < len(event_ids)
                         else None),
            "node": nd,
            "node_id": node_ids[nd] if node_ids and nd < len(node_ids) else None,
            "onset_day": onset,
            "duration_days": int(sel.sum()),
            "outcome": ("early" if pre.size else
                        "late" if during.size else "never"),
            "lead_days": int(onset - pre.min()) if pre.size else None,
            "max_prob_pre_onset": max_pre,
            "max_prob_during": float(p[sel].max()),
        }
        if events_meta and row["event_id"] in events_meta:
            row.update({k: v for k, v in events_meta[row["event_id"]].items()
                        if k not in row})
        rows.append(row)
    return rows


def missed_by_severity(rows: List[Dict], key: str = "peak_discharge",
                       n_bins: int = 4) -> List[Dict]:
    """Detection rate by magnitude quartile — are the misses the big ones?"""
    vals = np.array([r.get(key) if r.get(key) is not None else np.nan
                     for r in rows], dtype=float)
    if not np.isfinite(vals).any():
        return []
    edges = np.nanquantile(vals[np.isfinite(vals)],
                           np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (vals >= lo) & (vals <= hi if i == n_bins - 1 else vals < hi)
        block = [r for r, s in zip(rows, sel) if s]
        if not block:
            continue
        out.append({
            "quartile": i + 1, "lo": float(lo), "hi": float(hi),
            "n": len(block),
            "early": sum(r["outcome"] == "early" for r in block) / len(block),
            "late": sum(r["outcome"] == "late" for r in block) / len(block),
            "never": sum(r["outcome"] == "never" for r in block) / len(block),
        })
    return out


# ------------------------------------------------------------------ per-node

def node_table(p: np.ndarray, y: np.ndarray, node: np.ndarray, threshold: float,
               node_ids: Optional[Sequence[str]] = None,
               node_meta: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    """One row per river node, so failure can be located geographically."""
    p, y, node = (np.asarray(a).ravel() for a in (p, y, node))
    rows: List[Dict] = []
    for nd in np.unique(node):
        sel = node == nd
        yn, pn = y[sel], p[sel]
        c = contingency(yn, pn, threshold)
        nid = node_ids[int(nd)] if node_ids and int(nd) < len(node_ids) else None
        row = {"node": int(nd), "node_id": nid, "n": int(sel.sum()),
               "n_pos": int(yn.sum()),
               "pr_auc": average_precision(yn, pn) if yn.sum() else float("nan"),
               "pod": c["pod"], "far": c["far"], "csi": c["csi"],
               "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]}
        if node_meta and nid in node_meta:
            row.update(node_meta[nid])
        rows.append(row)
    return rows


def group_breakdown(rows: List[Dict], by: str) -> List[Dict]:
    """Aggregate a node table by `zone`, `position` or `basin`."""
    keys = sorted({str(r.get(by)) for r in rows if r.get(by) is not None})
    out = []
    for k in keys:
        block = [r for r in rows if str(r.get(by)) == k]
        tp = sum(r["tp"] for r in block)
        fp = sum(r["fp"] for r in block)
        fn = sum(r["fn"] for r in block)
        out.append({by: k, "n_nodes": len(block), "n_pos": sum(r["n_pos"] for r in block),
                    "pod": tp / max(tp + fn, 1), "far": fp / max(tp + fp, 1),
                    "csi": tp / max(tp + fp + fn, 1)})
    return out


# ------------------------------------------------------- training-time health

def seed_spread(per_seed: List[Dict], keys: Sequence[str] =
                ("pr_auc", "event_detection_rate", "ece")) -> Dict[str, Dict]:
    """Mean / std / min / max across seeds, and the ensemble's gain over the mean.

    Without this a 0.004 difference between two ladder rungs is uninterpretable.
    """
    out: Dict[str, Dict] = {}
    for k in keys:
        vals = np.array([s.get(k, np.nan) for s in per_seed], dtype=float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        out[k] = {"n_seeds": int(vals.size), "mean": float(vals.mean()),
                  "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                  "min": float(vals.min()), "max": float(vals.max())}
    return out


def stopping_report(history: List[Dict], patience: int) -> Dict:
    """Did training stop because it converged, or because patience ran out?

    `select` is the episode-level PR-AUC that early stopping watches; `val_pr_auc`
    is the node-day one. When the first plateaus and the second is still rising,
    stopping was driven by the noisier signal and the run was cut short.
    """
    if not history:
        return {}
    sel = np.array([h.get("val_event_pr_auc", np.nan) for h in history], float)
    node = np.array([h.get("val_pr_auc", np.nan) for h in history], float)
    n = len(history)
    best_ep = int(np.nanargmax(sel)) if np.isfinite(sel).any() else -1
    tail = max(n // 5, 3)
    def slope(a):
        a = a[-tail:]
        ok = np.isfinite(a)
        return float(np.polyfit(np.arange(ok.sum()), a[ok], 1)[0]) if ok.sum() > 1 else 0.0
    return {
        "epochs_run": n,
        "best_epoch": best_ep,
        "epochs_after_best": n - 1 - best_ep,
        "stopped_on_patience": (n - 1 - best_ep) >= patience,
        "val_event_pr_auc_slope_tail": slope(sel),
        "val_pr_auc_slope_tail": slope(node),
        # The case worth catching: selection metric flat or falling while the
        # node-day metric still climbs -> the run was ended by the noisier signal.
        "still_improving_on_node_day": slope(node) > 1e-4 and slope(sel) <= 1e-4,
    }


def optimisation_report(history: List[Dict], grad_clip: float) -> Dict:
    """Learning-rate and gradient health, and the loss-scale comparison.

    `clip_fraction` near 1.0 means the clip is doing the optimiser's job and the
    learning rate is too high; near 0.0 it is a no-op. `loss_scale` matters when
    comparing rungs: focal loss is roughly twenty times smaller in magnitude than
    BCE here, so an identical `lr` is not an identical step size.
    """
    if not history:
        return {}
    gn = np.array([h.get("grad_norm_mean", np.nan) for h in history], float)
    cf = np.array([h.get("clip_fraction", np.nan) for h in history], float)
    tl = np.array([h.get("train_loss", np.nan) for h in history], float)
    f = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    return {"grad_clip": grad_clip,
            "grad_norm_mean": f(gn),
            "grad_norm_first_epoch": float(gn[0]) if gn.size else float("nan"),
            "grad_norm_last_epoch": float(gn[-1]) if gn.size else float("nan"),
            "clip_fraction_mean": f(cf),
            "clip_fraction_first_epoch": float(cf[0]) if cf.size else float("nan"),
            "train_loss_first": float(tl[0]) if tl.size else float("nan"),
            "train_loss_last": float(tl[-1]) if tl.size else float("nan"),
            "clip_is_binding": f(cf) > 0.5,
            "clip_is_noop": f(cf) < 0.01}


def head_balance(history: List[Dict], cls_heads: Sequence[str]) -> Dict[str, float]:
    """Mean per-head classification loss over the last fifth of training.

    Read against `head_weights`: a head whose loss barely moves is contributing
    nothing but is still consuming capacity and gradient.
    """
    if not history:
        return {}
    tail = history[max(len(history) * 4 // 5, 0):]
    out: Dict[str, float] = {}
    for i, h in enumerate(cls_heads):
        vals = [ep["per_head"][i] for ep in tail
                if isinstance(ep.get("per_head"), list) and i < len(ep["per_head"])]
        if vals:
            out[h] = float(np.mean(vals))
    first = history[0].get("per_head")
    if isinstance(first, list):
        for i, h in enumerate(cls_heads):
            if i < len(first) and h in out:
                out[h + "__reduction"] = float(first[i] - out[h])
    return out
