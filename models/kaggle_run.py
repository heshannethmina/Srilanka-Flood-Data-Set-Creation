"""Kaggle runner for both model families — **this file only runs on Kaggle**.

Dataset creation lives in `scripts/`. This directory is model creation only; the
two never share an entry point, so a rebuild of the data and a rebuild of the
model are always separate acts.

Attach both datasets to the notebook, turn the GPU on, then:

    !git clone -q https://github.com/heshannethmina/Srilanka-Flood-Data-Set-Creation /kaggle/working/repo
    !python /kaggle/working/repo/models/kaggle_run.py --stage ladder2

Stages (run them separately if you are near a session limit):

    baselines     the four reference models of §7.8                  ~10 min
    ladder3       P0 -> P3, model 3 — the RQ1 answer                 ~4 h
    onset         P4_onset + its paired control — early warning      ~3 h
    ladder2       N0 -> N5, model 2's transformer ladder             ~2-3 h
    sar_pretrain  train the SAR encoder on the labelled chips        ~25 min
    ladder        M0 -> M5, model 1's graph ladder                   ~2-4 h
    leakage       M3 under a random split — the RQ2 number           ~30 min
    spatial       M3 with the Gin basin held out                     ~30 min
    sar2          N6_scalars + N6_gated, using the pretrained encoder ~3-4 h
    sar           M6_scalars + M6_cnn — model 1's SAR ablation       ~3-5 h
    all           every stage above, in that order

`all` runs them in decreasing order of value per GPU-hour, and stops starting
new stages once `--time-budget-hours` is gone, printing the command to resume
with. Both earlier ladders have already been run, so `ladder3` leads: it is the
only stage that can answer RQ1, because it is the only one that varies the graph
with the encoder, the loss and the panel all held fixed.

One run belongs with it and is not a stage of its own, because it is a rung of
model 1's ladder:

    --stage ladder --presets M5_bce --ladder-seeds 5     ~40 min

Model 1's M4/M5/M6 all sit on the focal loss that model 2 showed costs +0.0509
PR-AUC, so every published M-vs-N comparison is confounded. `M5_bce` is what
makes the three-family results table legitimate. Run it alongside `ladder3`.

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
sys.path.insert(0, HERE)          # makes floodlib, model1..model3 importable

RUNS = "/kaggle/working/runs"
ENCODER = os.path.join(RUNS, "sar_encoder.pt")

ORDER = ["baselines", "ladder3", "onset", "ladder2", "sar_pretrain", "ladder",
         "leakage", "spatial", "sar2", "sar"]

#: Rough hours per stage, used only to decide whether one still fits in the
#: session budget. Cheap-and-decisive stages run first so that a session which
#: runs out of time has still produced the results that matter most.
ESTIMATE_H = {"baselines": 0.2, "ladder3": 4.0, "onset": 3.0, "ladder2": 2.5,
              "sar_pretrain": 0.4, "ladder": 3.5, "leakage": 0.5, "spatial": 0.5,
              "sar2": 3.5, "sar": 4.0}


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
    """Print the GPU, and refuse to start if its architecture is unsupported.

    Kaggle hands out both T4 (sm_75) and P100 (sm_60), and its PyTorch build
    dropped sm_60. On a P100 every CUDA kernel fails with "no kernel image is
    available for execution on the device" — but only once the first tensor
    reaches the GPU, which is after the dataset load. Checking the arch list up
    front turns a wasted session into a ten-second error with the fix in it.
    """
    try:
        import torch
    except ImportError:
        sys.exit("[env] torch is not installed in this Kaggle image")

    if not torch.cuda.is_available():
        print(f"[env] torch {torch.__version__} | CPU only — enable the GPU "
              "accelerator in notebook settings")
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = list(torch.cuda.get_arch_list())
    print(f"[env] torch {torch.__version__} | {name} ({arch})")
    if arch not in supported:
        sys.exit(
            f"[fatal] {name} is {arch}, which this PyTorch build does not "
            f"support (it has: {' '.join(supported)}).\n"
            "        Every CUDA kernel would fail with 'no kernel image is "
            "available for execution on the device'.\n"
            "        Fix: notebook settings -> Accelerator -> GPU T4 x2, then "
            "re-run. The P100 is sm_60 and no\n"
            "        longer supported by Kaggle's torch build.")


# ------------------------------------------------------------------- stages

def stage_baselines(root: str) -> Dict:
    from floodlib.baselines import run_baselines
    return {"baselines_temporal": run_baselines("temporal", root, out_dir=RUNS)}


def _ladder(run_fn, root: str, epochs: int, presets: List[str],
            seeds: Optional[int], label: str, overrides: Optional[Dict] = None,
            **kw) -> Dict:
    """Run `presets` in order, isolating each step's failure from the rest.

    `seeds` overrides the seed count for every step; left as None each preset
    uses the count its own config declares. Overriding it writes under a `_s<N>`
    suffix so a re-run cannot clobber the canonical result it is meant to be
    compared against.
    """
    tover, mover = overrides or ({}, {})
    # An explicit seed override still needs its own suffix; when hyperparameters
    # are also overridden, engine.run derives the tag from them instead.
    tag = f"s{seeds}" if seeds is not None and not (tover or mover) else None
    out = {}
    for preset in presets:
        try:
            out[preset] = run_fn(preset=preset, protocol="temporal", root=root,
                                 out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                                 tag=tag, train_overrides=tover, **mover, **kw)
        except Exception:
            # The ladder is the longest stage; losing the last rung should not
            # also lose the earlier ones, which are already written to disk.
            print(f"[{label}] {preset} FAILED — continuing with the next step")
            traceback.print_exc()
    return out


def stage_ladder(root: str, epochs: int, presets: List[str],
                 seeds: Optional[int], overrides=None) -> Dict:
    from model1.train import run
    return _ladder(run, root, epochs, presets, seeds, "ladder", overrides)


def stage_ladder2(root: str, epochs: int, presets: List[str],
                  seeds: Optional[int], overrides=None) -> Dict:
    from model2.train import run
    return _ladder(run, root, epochs, presets, seeds, "ladder2", overrides)


def stage_ladder3(root: str, epochs: int, presets: List[str],
                  seeds: Optional[int], overrides=None) -> Dict:
    """Model 3 — the only ladder that can attribute anything to the graph.

    P0 runs first and is an assembly check, not a result: it is byte-identical
    to model 2's N3, so a score far from 0.8269 means the wiring is wrong and
    P1-P3 are not worth the GPU time. Check it before reading anything below it.
    """
    from model3.train import run
    return _ladder(run, root, epochs, presets, seeds, "ladder3", overrides)


def stage_onset(root: str, epochs: int, seeds: Optional[int],
                overrides=None) -> Dict:
    """The early-warning pair, scored on `target_onset_1d` rather than flood_1d.

    Run as a pair or not at all: an onset number without its graph-free control
    says nothing, which is exactly how model 1's M2 ev.det result became
    unreadable.
    """
    from model3.config import ONSET_RUNGS
    from model3.train import run
    return _ladder(run, root, epochs, ONSET_RUNGS, seeds, "onset", overrides)


def stage_leakage(root: str, epochs: int) -> Dict:
    """RQ2. Identical model, identical hyperparameters, one protocol changed."""
    from model1.train import run
    return {"M3_random": run(preset="M3", protocol="random", root=root,
                             out_dir=RUNS, epochs=epochs, n_seeds=1)}


def stage_spatial(root: str, epochs: int) -> Dict:
    from model1.train import run
    return {"M3_basin": run(preset="M3", protocol="basin", root=root,
                            out_dir=RUNS, epochs=epochs, n_seeds=1)}


def stage_sar_pretrain(root: str, sar_root: Optional[str], epochs: int,
                       image_px: int) -> Dict:
    """Train the SAR encoder on the labelled flood/dry chips (model 2, RQ8)."""
    from model2.pretrain_sar import pretrain
    if sar_root is None:
        print("[sar_pretrain] skipped — attach uom230429e/flood-data-set")
        return {}
    return {"sar_encoder": pretrain(root=root, sar_root=sar_root,
                                    out_path=ENCODER, epochs=epochs,
                                    px=image_px)}


def stage_sar(root: str, sar_root: Optional[str], epochs: int,
              seeds: Optional[int], image_px: int, batch_size: int) -> Dict:
    from model1.train import run
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


def stage_sar2(root: str, sar_root: Optional[str], epochs: int,
               seeds: Optional[int], image_px: int, batch_size: int) -> Dict:
    from model2.train import run
    if sar_root is None:
        print("[sar2] skipped — attach uom230429e/flood-data-set to the notebook")
        return {}
    ckpt = ENCODER if os.path.exists(ENCODER) else None
    if ckpt is None:
        # Not fatal: N6_gated still trains the encoder from scratch. But the
        # point of the stage is the transfer, so say so rather than let a null
        # result be read as evidence against pretraining.
        print("[sar2] WARNING no pretrained encoder at "
              f"{ENCODER} — run --stage sar_pretrain first, or N6_gated tests "
              "the gate alone rather than gate + transfer")
    out = {"N6_scalars": run(preset="N6_scalars", protocol="temporal", root=root,
                             out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                             sar_root=sar_root)}
    out["N6_gated"] = run(preset="N6_gated", protocol="temporal", root=root,
                          out_dir=RUNS, epochs=epochs, n_seeds=seeds,
                          sar_root=sar_root, image_px=image_px,
                          batch_size=batch_size, sar_pretrained=ckpt)
    return out


# ------------------------------------------------------------------ summary

def summarise() -> None:
    """One table over everything in RUNS, so a finished session reads at a glance."""
    try:
        from report import collect, render
    except ImportError:
        return
    rows = collect(RUNS)
    if not rows:
        return
    print("\n" + render(rows))
    print("Judge these on ev.det at a comparable FAR, not on PR-AUC: the target is "
          "tomorrow's discharge over the 98th percentile and discharge is strongly "
          "autocorrelated, so discharge_pctl already ranks near the ceiling.")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the flood early-warning models on Kaggle.")
    ap.add_argument("--stage", default="all", choices=["all"] + ORDER)
    ap.add_argument("--time-budget-hours", type=float, default=8.0,
                    help="stop starting new stages past this; Kaggle kills a GPU "
                         "session at ~9 h, and a killed session saves nothing")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5,
                    help="ensemble size for the M6/N6 SAR presets (§7.7 uses 5)")
    ap.add_argument("--presets", default="M0,M1,M2,M3,M4,M5",
                    help="model 1 ladder steps, comma-separated; narrow it to "
                         "revisit one step without re-running the whole ladder")
    ap.add_argument("--presets2", default="N0,N1,N2,N3,N4,N5",
                    help="model 2 ladder steps, comma-separated")
    ap.add_argument("--presets3", default=None,
                    help="model 3 ladder steps, comma-separated; defaults to "
                         "model3.config.LADDER (P0,P1,P2,P0_x5,P3)")
    ap.add_argument("--ladder-seeds", type=int, default=None,
                    help="override the seed count for every preset in --presets "
                         "and --presets2. Use it to re-run a single-seed step as "
                         "an ensemble, e.g. --presets M2,M3 --ladder-seeds 5. "
                         "Results are written under a _s<N> suffix and never "
                         "overwrite the originals.")
    ap.add_argument("--image-px", type=int, default=256,
                    help="SAR frame size for the CNN presets; 256 quarters the "
                         "activation memory versus 512 and still resolves the channel")
    ap.add_argument("--pretrain-epochs", type=int, default=25,
                    help="epochs for the sar_pretrain stage")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="M6_cnn / N6_gated only")
    # The same hyperparameter override flags every train.py accepts, so a sweep
    # is a sequence of command lines rather than a sequence of config edits.
    from floodlib import engine
    engine.add_override_args(ap)
    a = ap.parse_args()
    overrides = engine.overrides_from_args(a)

    from model1.config import PRESETS as P1
    from model2.config import PRESETS as P2
    from model3.config import LADDER as L3
    from model3.config import PRESETS as P3
    presets = [p.strip() for p in a.presets.split(",") if p.strip()]
    presets2 = [p.strip() for p in a.presets2.split(",") if p.strip()]
    presets3 = ([p.strip() for p in a.presets3.split(",") if p.strip()]
                if a.presets3 else list(L3))
    for names, table, flag in ((presets, P1, "--presets"),
                               (presets2, P2, "--presets2"),
                               (presets3, P3, "--presets3")):
        unknown = [p for p in names if p not in table]
        if unknown:
            sys.exit(f"[fatal] unknown {flag} preset(s) {unknown}; "
                     f"choose from {list(table)}")

    require_kaggle()
    report_env()
    os.makedirs(RUNS, exist_ok=True)

    root = find_input("flood_dataset.parquet",
                      "attach uom230429e/sri-lanka-flood-tabular-graph-2003-2025")
    if root is None:
        show_inputs()
        sys.exit("[fatal] the tabular dataset is required by every stage")
    sar_root = find_input("image_dataset.csv",
                          "the sar stages will be skipped")
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
                stage_ladder(root, a.epochs, presets, a.ladder_seeds, overrides)
            elif name == "ladder2":
                stage_ladder2(root, a.epochs, presets2, a.ladder_seeds, overrides)
            elif name == "ladder3":
                stage_ladder3(root, a.epochs, presets3, a.ladder_seeds, overrides)
            elif name == "onset":
                stage_onset(root, a.epochs, a.ladder_seeds, overrides)
            elif name == "leakage":
                stage_leakage(root, a.epochs)
            elif name == "spatial":
                stage_spatial(root, a.epochs)
            elif name == "sar_pretrain":
                stage_sar_pretrain(root, sar_root, a.pretrain_epochs, a.image_px)
            elif name == "sar":
                stage_sar(root, sar_root, a.epochs, a.seeds, a.image_px,
                          a.batch_size)
            elif name == "sar2":
                stage_sar2(root, sar_root, a.epochs, a.seeds, a.image_px,
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
