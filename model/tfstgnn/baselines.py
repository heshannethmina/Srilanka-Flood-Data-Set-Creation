"""Baselines 1–4 (PROJECT_PROPOSAL.md §7.8).

Every baseline is scored on **exactly** the sample set the network sees: same
protocol, same `valid_sample` filter, same lookback warm-up cut. Anything less
and the comparison is not a comparison.

    python -m tfstgnn.baselines --protocol temporal --out runs

Baseline 3 (the discharge-percentile rule) is the real bar: it is what GloFAS
already tells an operator for free. Baseline 4 (gradient-boosted trees) is the
one most likely to beat the GNN on tabular hydrology, and R3 in the risk
register says to report that honestly if it happens.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .calibrate import fit_isotonic
from .data import build_panel, load_panel
from .metrics import best_threshold, evaluate


def _dense(df: pd.DataFrame, node_ids, dates, col: str, fill=np.nan) -> np.ndarray:
    nmap = {n: i for i, n in enumerate(node_ids)}
    dmap = {d: i for i, d in enumerate(dates)}
    out = np.full((len(dates), len(node_ids)), fill, np.float32)
    t = df["date"].values.astype("datetime64[D]")
    ok = pd.Series(t).isin(dates).to_numpy() & df["node_id"].isin(nmap).to_numpy()
    sub = df[ok]
    ti = pd.Series(sub["date"].values.astype("datetime64[D]")).map(dmap).to_numpy(np.int64)
    ni = sub["node_id"].map(nmap).to_numpy(np.int64)
    out[ti, ni] = sub[col].to_numpy(np.float32)
    return out


def _sample_mask(panel, protocol: str, split: str, lookback: int) -> np.ndarray:
    code = {"train": 0, "val": 1, "test": 2}[split]
    m = (panel.splits[protocol] == code) & (panel.valid > 0)
    m[: lookback - 1] = False
    return m


def _flat(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return arr[mask]


def run_baselines(protocol: str = "temporal", root: Optional[str] = None,
                  lookback: int = 14, out_dir: Optional[str] = None,
                  gbt_lags=(0, 1, 3, 7, 14), max_far: Optional[float] = None) -> Dict:
    df, nodes, _ = load_panel(root)
    panel = build_panel(df, nodes)
    node_ids, dates = panel.node_ids, panel.dates
    T, N = panel.valid.shape

    tr_m = _sample_mask(panel, protocol, "train", lookback)
    va_m = _sample_mask(panel, protocol, "val", lookback)
    te_m = _sample_mask(panel, protocol, "test", lookback)

    y = panel.Y[:, :, 0]
    day = np.repeat(np.arange(T)[:, None], N, axis=1)
    node = np.repeat(np.arange(N)[None, :], T, axis=0)

    pctl_raw = _dense(df, node_ids, dates, "discharge_pctl")
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    doy = np.repeat(doy[:, None], N, axis=1)

    def score(name: str, s_val: np.ndarray, s_test: np.ndarray,
              calibrate: bool = True) -> Dict:
        yv, yt = _flat(y, va_m), _flat(y, te_m)
        pv = np.nan_to_num(_flat(s_val, va_m), nan=0.0)
        pt = np.nan_to_num(_flat(s_test, te_m), nan=0.0)
        if calibrate:
            cal = fit_isotonic(pv, yv)
            pv_c, pt_c = cal(pv), cal(pt)
        else:
            pv_c, pt_c = np.clip(pv, 0, 1), np.clip(pt, 0, 1)
        thr = best_threshold(yv, pv_c, "f1", max_far=max_far)
        return {"baseline": name, "threshold": float(thr),
                "test": evaluate(yt, pt_c, thr, _flat(panel.event, te_m),
                                 _flat(day, te_m), _flat(node, te_m)),
                "val": evaluate(yv, pv_c, thr)}

    results = {}

    # 1 — persistence: today's flood state is tomorrow's forecast
    results["persistence"] = score("persistence", panel.flood_state,
                                   panel.flood_state, calibrate=True)

    # 2 — per-node day-of-year climatology, fitted on the training mask only
    clim = np.zeros((367, N), np.float32)
    for n in range(N):
        sel = tr_m[:, n]
        if sel.sum() == 0:
            continue
        s = pd.Series(y[sel, n]).groupby(doy[sel, n]).mean()
        # ±7-day circular smoothing so rare day-of-years are not 0/1 spikes
        full = np.full(367, np.nan)
        full[s.index.to_numpy()] = s.to_numpy()
        pad = np.concatenate([full[-7:], full, full[:7]])
        sm = pd.Series(pad).rolling(15, center=True, min_periods=1).mean().to_numpy()
        clim[:, n] = sm[7:-7]
    clim = np.nan_to_num(clim, nan=float(y[tr_m].mean()))
    clim_pred = clim[doy, node]
    results["climatology"] = score("climatology", clim_pred, clim_pred, calibrate=False)

    # 3 — discharge-percentile rule ("what GloFAS already tells you")
    results["discharge_pctl"] = score("discharge_pctl", pctl_raw, pctl_raw)

    # 4 — gradient-boosted trees on the flattened lookback window
    gbt = _run_gbt(panel, tr_m, va_m, te_m, y, gbt_lags)
    if gbt is not None:
        pv_full = np.full((T, N), np.nan, np.float32)
        pt_full = np.full((T, N), np.nan, np.float32)
        pv_full[gbt["val_mask"]] = gbt["val"]
        pt_full[gbt["test_mask"]] = gbt["test"]
        # score() evaluates on va_m/te_m; the lag trim can only drop the very
        # first `maxlag` days, which the lookback warm-up already excluded.
        va_m &= gbt["val_mask"]
        te_m &= gbt["test_mask"]
        results["gbt"] = score(f"gbt[{gbt['impl']}]", pv_full, pt_full)

    print(f"\n=== BASELINES ({protocol}) ===")
    print(f"{'baseline':22s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>8s} "
          f"{'ECE':>7s} {'POD':>6s} {'FAR':>6s} {'CSI':>6s}")
    for k, v in results.items():
        t = v["test"]
        print(f"{v['baseline']:22s} {t['pr_auc']:8.4f} {t['roc_auc']:8.4f} "
              f"{t['brier']:8.5f} {t['ece']:7.4f} {t['pod']:6.3f} "
              f"{t['far']:6.3f} {t['csi']:6.3f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"baselines_{protocol}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[baselines] wrote {path}")
    return results


def _run_gbt(panel, tr_m, va_m, te_m, y, lags) -> Optional[Dict]:
    """Flatten X at a handful of lags into a per-sample feature vector."""
    T, N, F = panel.X.shape
    maxlag = max(lags)

    def design(mask: np.ndarray):
        m = mask.copy()
        m[:maxlag] = False
        t_i, n_i = np.nonzero(m)
        cols = [panel.X[t_i - L, n_i] for L in lags]
        cols.append(panel.S[n_i])
        return np.concatenate(cols, axis=1).astype(np.float32), y[t_i, n_i], m

    Xtr, ytr, _ = design(tr_m)
    Xva, yva, mva = design(va_m)
    Xte, yte, mte = design(te_m)
    print(f"[gbt] design matrix {Xtr.shape} "
          f"({Xtr.nbytes / 1e6:.0f} MB), {ytr.mean():.3%} positive")

    impl, model = None, None
    try:
        import lightgbm as lgb
        impl = "lightgbm"
        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            is_unbalance=True, verbosity=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    except ImportError:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            impl = "sklearn-histgb"
            model = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.1,
                class_weight="balanced")
            model.fit(Xtr, ytr)
        except ImportError:
            print("[gbt] skipped — install lightgbm or scikit-learn")
            return None

    return {"impl": impl,
            "val": model.predict_proba(Xva)[:, 1].astype(np.float32),
            "test": model.predict_proba(Xte)[:, 1].astype(np.float32),
            "val_mask": mva, "test_mask": mte}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baselines 1-4.")
    ap.add_argument("--protocol", default="temporal",
                    choices=["temporal", "basin", "random"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--max-far", type=float, default=None)
    a = ap.parse_args()
    run_baselines(a.protocol, a.root, a.lookback, a.out, max_far=a.max_far)


if __name__ == "__main__":
    main()
