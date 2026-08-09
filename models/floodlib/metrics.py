"""Evaluation metrics (PROJECT_PROPOSAL.md §7.10).

Pure NumPy — no scikit-learn requirement, so the metrics run identically in a
bare Kaggle image. Plain accuracy is deliberately not implemented: at a 1.9%
positive rate, "never flood" scores 98.1%.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ------------------------------------------------------------------- ranking

def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """PR-AUC by the step-wise (interpolation-free) definition."""
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    n_pos = y.sum()
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------- calibration

def brier_decomposition(y: np.ndarray, p: np.ndarray, bins: int = 15) -> Dict[str, float]:
    """Murphy decomposition: Brier = reliability − resolution + uncertainty."""
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    brier = float(np.mean((p - y) ** 2))
    base = y.mean()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel = res = 0.0
    for b in range(bins):
        sel = idx == b
        n = sel.sum()
        if n == 0:
            continue
        p_bar, o_bar = p[sel].mean(), y[sel].mean()
        rel += n * (p_bar - o_bar) ** 2
        res += n * (o_bar - base) ** 2
    n_all = len(y)
    return {"brier": brier, "reliability": rel / n_all, "resolution": res / n_all,
            "uncertainty": float(base * (1 - base))}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 15):
    """(bin_centre, mean_predicted, observed_frequency, count) for the diagram."""
    y = np.asarray(y, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        rows.append(((edges[b] + edges[b + 1]) / 2,
                     float(p[sel].mean()) if sel.sum() else np.nan,
                     float(y[sel].mean()) if sel.sum() else np.nan,
                     int(sel.sum())))
    return np.array(rows)


# --------------------------------------------------------------- operational

def contingency(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    yhat = (np.asarray(p).ravel() >= threshold).astype(np.int64)
    y = np.asarray(y).ravel().astype(np.int64)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    pod = tp / max(tp + fn, 1)                       # hit rate / recall
    far = fp / max(tp + fp, 1)                       # false alarm ratio
    csi = tp / max(tp + fp + fn, 1)                  # critical success index
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * pod / max(prec + pod, 1e-12)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pod": pod, "far": far, "csi": csi, "precision": prec, "f1": f1}


def best_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "f1",
                   max_far: Optional[float] = None, n: int = 200) -> float:
    """Threshold search on the *validation* split only — never on test."""
    grid = np.quantile(np.asarray(p).ravel(), np.linspace(0.5, 0.9999, n))
    best, best_score = 0.5, -np.inf
    for t in np.unique(grid):
        c = contingency(y, p, t)
        if max_far is not None and c["far"] > max_far:
            continue
        s = c[criterion]
        if s > best_score:
            best, best_score = float(t), s
    return best


# -------------------------------------------------------------- event-level

def event_metrics(p: np.ndarray, event: np.ndarray, day: np.ndarray, node: np.ndarray,
                  threshold: float, max_lead: int = 7) -> Dict[str, float]:
    """Detection rate and mean lead time over flood episodes.

    An episode counts as detected if the model exceeded `threshold` on any day
    from `onset − max_lead` to `onset − 1`; lead time is onset minus the first
    such day. This is **not** node-day recall — an episode detected once counts
    once, which is what an operator experiences.
    """
    p, event, day, node = (np.asarray(a).ravel() for a in (p, event, day, node))
    alarms: Dict[int, np.ndarray] = {}
    for nd in np.unique(node):
        sel = node == nd
        alarms[int(nd)] = np.sort(day[sel][p[sel] >= threshold])

    detected, leads = 0, []
    ev_ids = np.unique(event[event >= 0])
    for e in ev_ids:
        sel = event == e
        onset = int(day[sel].min())
        nd = int(node[sel][0])
        a = alarms.get(nd, np.empty(0))
        w = a[(a >= onset - max_lead) & (a <= onset - 1)]
        if w.size:
            detected += 1
            leads.append(onset - int(w.min()))
    n_ev = len(ev_ids)
    return {"n_events": n_ev,
            "event_detection_rate": detected / max(n_ev, 1),
            "mean_lead_days": float(np.mean(leads)) if leads else float("nan"),
            "median_lead_days": float(np.median(leads)) if leads else float("nan")}


def event_pr_auc(y: np.ndarray, p: np.ndarray, event: np.ndarray,
                 day: np.ndarray, node: np.ndarray, week_len: int = 7) -> float:
    """PR-AUC where the unit is an *episode*, not a node-day.

    Positives: one per episode, scored by the model's maximum probability in the
    7 days before onset. Negatives: one per flood-free node-week, scored by its
    maximum probability. Used for early stopping, because node-day PR-AUC on a
    hydrologically quiet validation block (§11.4) is a noisy selection signal.
    """
    p, event, day, node, y = (np.asarray(a).ravel() for a in (p, event, day, node, y))
    scores, labels = [], []
    for e in np.unique(event[event >= 0]):
        sel = event == e
        onset, nd = int(day[sel].min()), int(node[sel][0])
        w = (node == nd) & (day >= onset - week_len) & (day < onset)
        if w.any():
            scores.append(p[w].max())
            labels.append(1)
    week = day // week_len
    key = node.astype(np.int64) * 10_000 + week
    has_event = np.zeros(len(p), bool)
    for k in np.unique(key[event >= 0]):
        has_event |= key == k
    for k in np.unique(key[~has_event]):
        sel = key == k
        scores.append(p[sel].max())
        labels.append(0)
    if not scores or sum(labels) == 0:
        return float("nan")
    return average_precision(np.array(labels), np.array(scores))


# ---------------------------------------------------------------- aggregate

def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5,
             event: Optional[np.ndarray] = None, day: Optional[np.ndarray] = None,
             node: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {
        "pr_auc": average_precision(y, p),
        "roc_auc": roc_auc(y, p),
        "ece": expected_calibration_error(y, p),
        "n": int(len(np.asarray(y).ravel())),
        "pos_rate": float(np.asarray(y).ravel().mean()),
    }
    out.update(brier_decomposition(y, p))
    out.update(contingency(y, p, threshold))
    if event is not None and day is not None and node is not None:
        out.update(event_metrics(p, event, day, node, threshold))
        out["event_pr_auc"] = event_pr_auc(y, p, event, day, node)
    return out
