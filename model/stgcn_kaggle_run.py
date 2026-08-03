"""Kaggle runner for the STGCN baseline architecture.

Can be run on Kaggle GPU instances or locally:

    python model/stgcn_kaggle_run.py --protocol temporal --epochs 60
    python model/stgcn_kaggle_run.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stgcn.config import STGCNModelConfig, STGCNTrainConfig
from stgcn.train import run_stgcn


def find_data_root() -> Optional[str]:
    """Find input dataset root under /kaggle/input or local data directory."""
    candidates = [
        "/kaggle/input/sri-lanka-flood-tabular-graph-2003-2025",
        os.path.abspath(os.path.join(HERE, "..", "data", "processed")),
    ]
    hits = sorted(glob.glob("/kaggle/input/**/flood_dataset.parquet", recursive=True), key=len)
    if hits:
        candidates.insert(0, os.path.dirname(hits[0]))

    for c in candidates:
        if os.path.exists(os.path.join(c, "flood_dataset.parquet")):
            return c
    return None


def run_stage(stage: str, out_dir: str, data_root: Optional[str] = None, epochs: int = 60) -> Dict:
    mcfg = STGCNModelConfig()

    if stage == "temporal":
        tcfg = STGCNTrainConfig(protocol="temporal", epochs=epochs)
    elif stage == "basin":
        tcfg = STGCNTrainConfig(protocol="basin", epochs=epochs)
    elif stage == "random":
        tcfg = STGCNTrainConfig(protocol="random", epochs=epochs)
    elif stage == "ensemble":
        tcfg = STGCNTrainConfig(protocol="temporal", epochs=epochs, n_seeds=5, calibration="temperature")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"\n=======================================================")
    print(f"   STGCN STAGE: {stage.upper()} (protocol={tcfg.protocol})")
    print(f"=======================================================")

    results = run_stgcn(mcfg, tcfg, root=data_root, out_dir=out_dir)
    return results


def print_summary_table(results_list: list[Dict]) -> None:
    print("\n" + "=" * 80)
    print("                      STGCN BASELINE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Stage / Protocol':20s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>8s} {'ECE':>7s} {'POD':>6s} {'FAR':>6s} {'CSI':>6s}")
    print("-" * 80)

    for r in results_list:
        p = r["protocol"]
        t = r["test"]
        print(f"{p:20s} {t['pr_auc']:8.4f} {t['roc_auc']:8.4f} {t['brier']:8.5f} {t['ece']:7.4f} {t['pod']:6.3f} {t['far']:6.3f} {t['csi']:6.3f}")

    print("=" * 80 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="STGCN Baseline Kaggle Runner")
    ap.add_argument("--stage", default="temporal", choices=["temporal", "basin", "random", "ensemble", "all"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="/kaggle/working/runs" if os.path.exists("/kaggle/input") else "runs")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    data_root = a.root or find_data_root()
    if not data_root:
        print("[error] Could not locate flood_dataset.parquet.")
        sys.exit(1)

    print(f"[stgcn_kaggle_run] Using data root: {data_root}")
    print(f"[stgcn_kaggle_run] Output directory: {a.out}")

    stages = ["temporal", "basin", "ensemble"] if a.stage == "all" else [a.stage]
    results_list = []

    for s in stages:
        res = run_stage(s, out_dir=a.out, data_root=data_root, epochs=a.epochs)
        results_list.append(res)

    print_summary_table(results_list)


if __name__ == "__main__":
    main()
