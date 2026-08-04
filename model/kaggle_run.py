"""Kaggle runner for the TF-STGNN model — **this file only runs on Kaggle**.

Dataset creation lives in `scripts/`. This directory is model creation only; the
two never share an entry point, so a rebuild of the data and a rebuild of the
model are always separate acts.

Attach both datasets to the notebook, turn the GPU on, then:

    !git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
    !python /kaggle/working/repo/model/kaggle_run.py --stage all

Stages (run them separately if you are near a session limit):

    baselines   the four reference models of §7.8              ~10 min
    ladder      M0 → M5, the incremental build of §8           ~2-4 h
    leakage     M3 under a random split — the RQ2 number       ~30 min
    spatial     M3 with the Gin basin held out                 ~30 min
    sar         M6_scalars then M6_cnn — needs the SAR dataset ~3-5 h
    all         every stage above, in that order

Everything lands in /kaggle/working/runs as JSON, which survives "Save Version"
and can be downloaded as notebook output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RUNS = "/kaggle/working/runs"

ORDER = ["baselines", "ladder", "leakage", "spatial", "sar"]

#: Rough hours per stage, used only to decide whether one still fits in the
#: session budget. Cheap-and-decisive stages run first so that a session which
#: runs out of time has still produced the results that matter most.
ESTIMATE_H = {"baselines": 0.2, "ladder": 3.5, "leakage": 0.5,
              "spatial": 0.5, "sar": 4.0}


# --------------------------------------------------------------- environment

def require_kaggle() -> None:
    if not os.path.isdir("/kaggle/input"):
        sys.exit(
            "This script only runs on Kaggle (/kaggle/input not found).\n"
            "It is the model-creation entry point for a Kaggle notebook; there "
            "is no local mode by design."
        )


def find_input(filename: str, hint: str) -> Optional[str]:
    """Locate a mounted dataset by a file it must contain.

    Searched recursively and at any depth: Kaggle names each mount after the
    dataset slug, slugs change when a dataset is re-published, and the mount
    layout itself varies (`/kaggle/input/<slug>/` in the classic form,
    `/kaggle/input/datasets/<owner>/<slug>/<version>/` in the newer one).
    Hunting for the file is the only durable way to find it.
    """
    hits = sorted(glob.glob(f"/kaggle/input/**/{filename}", recursive=True),
                  key=len)          # shallowest match wins
    if hits:
        return os.path.dirname(hits[0])
    print(f"[warn] {filename} not found under /kaggle/input — {hint}")
    return None


def show_inputs() -> None:
    """Print what is actually mounted, so a miss is diagnosable from the log."""
    found = [p for p in glob.glob("/kaggle/input/**/*", recursive=True)
             if os.path.isfile(p)]
    if not found:
        print("[input] /kaggle/input holds no files — attach the datasets via "
              "'Add Input' in the notebook's right-hand panel")
        return
    print(f"[input] {len(found)} files mounted; a sample:")
    for p in sorted(found, key=len)[:12]:
        print(f"        {p}")


def report_env() -> None:
    try:
        import torch
        gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "CPU only — enable the GPU accelerator in notebook settings")
        print(f"[env] torch {torch.__version__} | {gpu}")
    except ImportError:
        sys.exit("[env] torch is not installed in this Kaggle image")


# ------------------------------------------------------------------- stages

def stage_baselines(root: str) -> Dict:
    from tfstgnn.baselines import run_baselines
    return {"baselines_temporal": run_baselines("temporal", root, out_dir=RUNS)}


def stage_ladder(root: str, epochs: int, presets: List[str],
                 seeds: Optional[int]) -> Dict:
    """Run the ladder steps in `presets`.

    `seeds` overrides the seed count for every step; left as None each preset
    uses the count its own config declares (1 for M0-M4, 5 for M5). Overriding
    it writes under a `_s<N>` suffix so a re-run cannot clobber the canonical
    result it is meant to be compared against.
    """
    from tfstgnn.train import run
    tag = f"s{seeds}" if seeds is not None else None
    out = {}
    for preset in presets:
        try:
            out[preset] = run(preset=preset, protocol="temporal", root=root,
                              out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                              tag=tag)
        except Exception:
            # The ladder is the longest stage; losing M5 should not also lose
            # M0-M4, which are already written to disk.
            print(f"[ladder] {preset} FAILED — continuing with the next step")
            traceback.print_exc()
    return out


def stage_leakage(root: str, epochs: int) -> Dict:
    """RQ2. Identical model, identical hyperparameters, one protocol changed."""
    from tfstgnn.train import run
    return {"M3_random": run(preset="M3", protocol="random", root=root,
                             out_dir=RUNS, epochs=epochs, n_seeds=1)}


def stage_spatial(root: str, epochs: int) -> Dict:
    from tfstgnn.train import run
    return {"M3_basin": run(preset="M3", protocol="basin", root=root,
                            out_dir=RUNS, epochs=epochs, n_seeds=1)}


def stage_sar(root: str, sar_root: Optional[str], epochs: int,
              seeds: Optional[int], image_px: int, batch_size: int) -> Dict:
    from tfstgnn.train import run
    if sar_root is None:
        print("[sar] skipped — attach uom230429e/flood-data-set to the notebook")
        return {}
    out = {"M6_scalars": run(preset="M6_scalars", protocol="temporal", root=root,
                             out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                             sar_root=sar_root)}
    out["M6_cnn"] = run(preset="M6_cnn", protocol="temporal", root=root,
                        out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                        sar_root=sar_root, image_px=image_px,
                        batch_size=batch_size)
    return out


# ------------------------------------------------------------------ summary

def summarise() -> None:
    """One table over everything in RUNS, so a finished session reads at a glance."""
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if "baseline" in os.path.basename(path):
            for name, v in d.items():
                rows.append({"model": v.get("baseline", name), **v["test"]})
        else:
            # Seed count belongs in the label: a 5-seed M2 and a 1-seed M2 are
            # different models, and comparing them against 5-seed M5/M6 is the
            # whole point of re-running a step.
            n = d.get("train_config", {}).get("n_seeds", 1)
            seeds = f" x{n}" if n and n > 1 else ""
            rows.append({"model": f"{d['preset']}{seeds} [{d['protocol']}]",
                         **d["test"]})

    if not rows:
        return
    print("\n" + "=" * 96)
    print(f"{'model':24s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>9s} {'ECE':>7s} "
          f"{'POD':>6s} {'FAR':>6s} {'CSI':>6s} {'ev.det':>7s} {'lead':>6s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['model'][:24]:24s} {r.get('pr_auc', float('nan')):8.4f} "
              f"{r.get('roc_auc', float('nan')):8.4f} {r.get('brier', float('nan')):9.5f} "
              f"{r.get('ece', float('nan')):7.4f} {r.get('pod', float('nan')):6.3f} "
              f"{r.get('far', float('nan')):6.3f} {r.get('csi', float('nan')):6.3f} "
              f"{r.get('event_detection_rate', float('nan')):7.3f} "
              f"{r.get('mean_lead_days', float('nan')):6.2f}")
    print("=" * 96)
    print("Judge these on ev.det at a comparable FAR, not on PR-AUC: the target is "
          "tomorrow's discharge over the 98th percentile and discharge is strongly "
          "autocorrelated, so discharge_pctl already ranks near the ceiling.")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TF-STGNN model on Kaggle.")
    ap.add_argument("--stage", default="all", choices=["all"] + ORDER)
    ap.add_argument("--time-budget-hours", type=float, default=8.0,
                    help="stop starting new stages past this; Kaggle kills a GPU "
                         "session at ~9 h, and a killed session saves nothing")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5,
                    help="ensemble size for M6 in the sar stage (§7.7 uses 5)")
    ap.add_argument("--presets", default="M0,M1,M2,M3,M4,M5",
                    help="which ladder steps to run, comma-separated; narrow it "
                         "to revisit one step without re-running the whole ladder")
    ap.add_argument("--ladder-seeds", type=int, default=None,
                    help="override the seed count for every preset in --presets. "
                         "Use it to re-run a single-seed step as an ensemble, e.g. "
                         "--presets M2,M3 --ladder-seeds 5. Results are written "
                         "under a _s<N> suffix and never overwrite the originals.")
    ap.add_argument("--image-px", type=int, default=256,
                    help="SAR frame size for M6_cnn; 256 quarters the activation "
                         "memory versus 512 and still resolves the channel")
    ap.add_argument("--batch-size", type=int, default=8, help="M6_cnn only")
    a = ap.parse_args()

    from tfstgnn.config import PRESETS
    presets = [p.strip() for p in a.presets.split(",") if p.strip()]
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        sys.exit(f"[fatal] unknown preset(s) {unknown}; "
                 f"choose from {list(PRESETS)}")

    require_kaggle()
    report_env()
    os.makedirs(RUNS, exist_ok=True)

    root = find_input("flood_dataset.parquet",
                      "attach uom230429e/sri-lanka-flood-tabular-graph-2003-2025")
    if root is None:
        show_inputs()
        sys.exit("[fatal] the tabular dataset is required by every stage")
    sar_root = find_input("image_dataset.csv",
                          "the sar stage will be skipped")
    print(f"[data] tabular {root}\n[data] sar     {sar_root}")

    stages = ORDER if a.stage == "all" else [a.stage]
    started = time.time()
    done: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []

    for name in stages:
        elapsed_h = (time.time() - started) / 3600.0
        est = ESTIMATE_H.get(name, 1.0)
        if len(stages) > 1 and elapsed_h + est > a.time_budget_hours:
            print(f"\n[skip] {name}: ~{est:.1f} h needed, only "
                  f"{a.time_budget_hours - elapsed_h:.1f} h of budget left. "
                  f"Run it in a fresh session:\n"
                  f"       !python {os.path.abspath(__file__)} --stage {name}")
            skipped.append(name)
            continue

        t0 = time.time()
        print(f"\n{'#' * 70}\n# stage: {name}\n{'#' * 70}")
        try:
            if name == "baselines":
                stage_baselines(root)
            elif name == "ladder":
                stage_ladder(root, a.epochs, presets, a.ladder_seeds)
            elif name == "leakage":
                stage_leakage(root, a.epochs)
            elif name == "spatial":
                stage_spatial(root, a.epochs)
            elif name == "sar":
                stage_sar(root, sar_root, a.epochs, a.seeds, a.image_px,
                          a.batch_size)
            done.append(name)
            print(f"[stage {name}] done in {(time.time() - t0) / 60:.1f} min")
        except Exception:
            # A batch run must not lose four finished stages to the fifth one
            # crashing — record it, keep going, and surface it in the summary.
            failed.append(name)
            print(f"[stage {name}] FAILED after {(time.time() - t0) / 60:.1f} min")
            traceback.print_exc()

    summarise()
    print(f"\n[session] {(time.time() - started) / 3600.0:.2f} h · "
          f"done {done or '-'} · skipped {skipped or '-'} · failed {failed or '-'}")
    print(f"Results in {RUNS} — 'Save Version' keeps them as notebook output.")
    if failed:
        sys.exit(f"[fatal] {len(failed)} stage(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
