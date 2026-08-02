"""Post-hoc probability calibration (PROJECT_PROPOSAL.md §7.7 "Calibration").

Fitted on the **validation block only**, then frozen. Fitting on test would be
the same leakage the project exists to eliminate.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Single scalar T minimising validation NLL of sigmoid(logit / T)."""
    lg = torch.as_tensor(np.asarray(logits, np.float32).ravel())
    tg = torch.as_tensor(np.asarray(y, np.float32).ravel())
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / log_t.exp(), tg)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Isotonic regression via pool-adjacent-violators (no sklearn needed)."""
    p = np.asarray(p, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    order = np.argsort(p, kind="mergesort")
    xs, ys = p[order], y[order]

    # Pool adjacent violators: merge any block whose mean falls below its
    # predecessor's, until the block means are non-decreasing.
    vals: list[float] = []
    wts: list[float] = []
    starts: list[float] = []
    for k in range(len(ys)):
        vals.append(float(ys[k]))
        wts.append(1.0)
        starts.append(float(xs[k]))
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            starts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    knots_x = np.array(starts)
    knots_y = np.array(vals)

    def apply(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, np.float64).ravel()
        j = np.searchsorted(knots_x, q, side="right") - 1
        j = np.clip(j, 0, len(knots_y) - 1)
        return np.clip(knots_y[j], 0.0, 1.0)

    return apply


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64) / max(T, 1e-3)))
