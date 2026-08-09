"""Model-agnostic training / evaluation engine.

Every model family runs through this one file. A family supplies a `build_model`
callable and a preset table; everything else — the split protocol, the loss, the
early-stopping criterion, the ensembling, the calibration, the threshold search
and the metrics — is fixed here.

That is deliberate and it is the reason the engine exists at all. If `model1`
and `model2` each owned a copy of this loop, a drift in either one (a different
early-stopping metric, a threshold fitted on a different split) would silently
turn the comparison between them into a comparison of evaluation protocols. The
proposal's central claim is about leakage control; the code has to be at least
as careful as the claim.

Enforced here rather than left to the operator: normalisation and thresholds
come from train/val only, the decision threshold is chosen on validation, and
the calibrator is fitted on validation and frozen before test is ever touched.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .calibrate import apply_temperature, fit_isotonic, fit_temperature
from .data import Panel, SnapshotBatcher, build_panel, load_panel
from .graph import build_graph
from .losses import MultiHeadLoss
from .metrics import best_threshold, evaluate
from .schema import BaseModelConfig
from .traincfg import Preset, TrainConfig

#: A family's factory: (model config, graph or None) → an nn.Module whose
#: forward signature is (x, s, img, img_pos, img_mask, img_age).
ModelBuilder = Callable[[BaseModelConfig, object], nn.Module]


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------- predictions

def _image_inputs(b: Dict, store, px: int, device: torch.device):
    """Packed SAR inputs for the CNN branch, or Nones when it is disabled."""
    if store is None or "frame" not in b:
        return None, None, None, None
    img, pos, mask = store.batch(b["frame"], px)
    return (torch.as_tensor(img, device=device),
            torch.as_tensor(pos, device=device),
            torch.as_tensor(mask, device=device),
            torch.as_tensor(b["age"], device=device))


@torch.no_grad()
def collect(model: nn.Module, batcher: SnapshotBatcher, S: torch.Tensor,
            device: torch.device, px: int, store=None) -> Dict[str, np.ndarray]:
    """Run the model over a split and flatten to 1-D arrays of valid node-days."""
    model.eval()
    logits, ys, days, nodes, events = [], [], [], [], []
    for b in batcher:
        x = torch.as_tensor(b["x"], device=device)
        img, ipos, imask, age = _image_inputs(b, store, px, device)
        out = model(x, S, img, ipos, imask, age)
        m = b["mask"] > 0                                   # [B, N]
        lg = out["logits"][..., 0].detach().cpu().numpy()
        logits.append(lg[m])
        ys.append(b["y"][..., 0][m])
        events.append(b["event"][m])
        d = np.repeat(b["days"][:, None], m.shape[1], axis=1)
        n = np.repeat(np.arange(m.shape[1])[None, :], m.shape[0], axis=0)
        days.append(d[m])
        nodes.append(n[m])
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return {"logit": cat(logits), "y": cat(ys), "day": cat(days),
            "node": cat(nodes), "event": cat(events)}


# ------------------------------------------------------------------ training

def train_one(build_model: ModelBuilder, panel: Panel, mcfg: BaseModelConfig,
              tcfg: TrainConfig, graph, device: torch.device, verbose: bool = True,
              frame_map=None, store=None
              ) -> Tuple[nn.Module, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    set_seed(tcfg.seed)
    model = build_model(mcfg, graph).to(device)
    S = torch.as_tensor(panel.S, device=device)
    px = mcfg.image_px

    mk = lambda split, sh: SnapshotBatcher(panel, tcfg.protocol, split, mcfg.lookback,
                                           tcfg.batch_size, shuffle=sh,
                                           frame_map=frame_map)
    tr, va, te = mk("train", True), mk("val", False), mk("test", False)
    if verbose:
        print(f"[train] params {n_params(model):,} | "
              f"train {tr.n_samples:,} ({tr.pos_rate:.3%} pos) · "
              f"val {va.n_samples:,} ({va.pos_rate:.3%}) · "
              f"test {te.n_samples:,} ({te.pos_rate:.3%})")

    if tcfg.loss == "wbce" and tcfg.pos_weight is None:
        tcfg = replace(tcfg, pos_weight=(1 - tr.pos_rate) / max(tr.pos_rate, 1e-6))

    crit = MultiHeadLoss(tcfg, panel.cls_heads).to(device)
    # Frozen pretrained encoders contribute no gradients; handing their tensors
    # to AdamW would still allocate optimiser state for them.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    def lr_at(ep: int) -> float:
        if ep < tcfg.warmup_epochs:
            return (ep + 1) / max(tcfg.warmup_epochs, 1)
        prog = (ep - tcfg.warmup_epochs) / max(tcfg.epochs - tcfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best_score, best_state, bad = -np.inf, None, 0
    for ep in range(tcfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in tr:
            x = torch.as_tensor(b["x"], device=device)
            img, ipos, imask, age = _image_inputs(b, store, px, device)
            out = model(x, S, img, ipos, imask, age)
            loss = crit(out,
                        torch.as_tensor(b["y"], device=device),
                        torch.as_tensor(b["r"], device=device),
                        torch.as_tensor(b["mask"], device=device),
                        torch.as_tensor(b["conf"], device=device))["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        vp = collect(model, va, S, device, px, store)
        vm = evaluate(vp["y"], 1 / (1 + np.exp(-vp["logit"])), 0.5,
                      vp["event"], vp["day"], vp["node"])
        score = vm.get("event_pr_auc", np.nan)
        if not np.isfinite(score):
            score = vm["pr_auc"]

        if verbose and (ep % tcfg.log_every == 0):
            print(f"  ep {ep:3d}  loss {tot / max(nb,1):.5f}  "
                  f"val PR-AUC {vm['pr_auc']:.4f}  event PR-AUC {score:.4f}  "
                  f"ECE {vm['ece']:.4f}  ({time.time()-t0:.1f}s)")

        if score > best_score + 1e-5:
            best_score, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best {best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, collect(model, va, S, device, px, store),
            collect(model, te, S, device, px, store))


# ------------------------------------------------------------------ pipeline

def run(preset: str, presets: Dict[str, Preset], build_model: ModelBuilder, *,
        family: str = "model", protocol: Optional[str] = None,
        root: Optional[str] = None, out_dir: Optional[str] = None,
        epochs: Optional[int] = None, n_seeds: Optional[int] = None,
        device_spec: str = "auto", max_far: Optional[float] = None,
        verbose: bool = True, sar_root: Optional[str] = None,
        image_px: Optional[int] = None, batch_size: Optional[int] = None,
        tag: Optional[str] = None, **model_overrides) -> Dict:
    """Train `preset` from `presets` and write its result JSON.

    `model_overrides` are applied to the model config after the preset, which is
    how a family exposes its own knobs (e.g. model2's pretrained SAR encoder
    path) without the engine needing to know they exist.
    """
    p = presets[preset]
    mcfg, tcfg = p.model, p.train
    if protocol:
        tcfg = replace(tcfg, protocol=protocol)
    if epochs is not None:
        tcfg = replace(tcfg, epochs=epochs, warmup_epochs=min(tcfg.warmup_epochs, epochs))
    if n_seeds is not None:
        tcfg = replace(tcfg, n_seeds=n_seeds)
    if batch_size is not None:
        tcfg = replace(tcfg, batch_size=batch_size)
    if image_px is not None:
        mcfg = replace(mcfg, image_px=image_px)
    overrides = {k: v for k, v in model_overrides.items() if v is not None}
    if overrides:
        mcfg = replace(mcfg, **overrides)

    device = resolve_device(device_spec)
    print(f"[run] {family} preset {preset} ({p.answers}) | "
          f"protocol {tcfg.protocol} | {device}")

    df, nodes, edges = load_panel(root)
    panel = build_panel(df, nodes, truncate_after=tcfg.truncate_after)
    graph = build_graph(edges, nodes, panel.node_ids, mcfg.graph_mode)

    # ---- SAR branch ------------------------------------------------------
    frame_map = store = None
    if mcfg.vision_mode == "scalars":
        from .sar import attach_sar_scalars
        attach_sar_scalars(panel, sar_root)
        mcfg = replace(mcfg, n_dynamic=panel.X.shape[-1])
    elif mcfg.vision_mode == "cnn":
        from .sar import FrameStore, build_frame_map, load_index
        idx, sroot = load_index(sar_root)
        frame_map = build_frame_map(idx, panel.node_ids, panel.dates)
        store = FrameStore(idx, sroot, px=mcfg.image_px)
        print(f"[sar] {len(store)} frames, imagery at "
              f"{int((frame_map['frame'] >= 0).any(0).sum())} nodes")

    print(f"[data] X{panel.X.shape}  {graph.summary()}")

    val_logits: List[np.ndarray] = []
    test_logits: List[np.ndarray] = []
    val_ref = test_ref = None
    params = 0
    for k in range(max(tcfg.n_seeds, 1)):
        seed_cfg = replace(tcfg, seed=tcfg.seed + k)
        if verbose and tcfg.n_seeds > 1:
            print(f"[seed {k+1}/{tcfg.n_seeds}]")
        model, vp, tp = train_one(build_model, panel, mcfg, seed_cfg, graph,
                                  device, verbose, frame_map, store)
        params = n_params(model)
        del model                       # a 5-seed CNN ensemble will not fit otherwise
        val_logits.append(vp["logit"])
        test_logits.append(tp["logit"])
        val_ref, test_ref = vp, tp

    # Deep ensemble: average in probability space, then re-logit for calibration.
    def ens(ls: List[np.ndarray]) -> np.ndarray:
        pr = np.mean([1 / (1 + np.exp(-l)) for l in ls], axis=0)
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return np.log(pr / (1 - pr))

    v_logit, t_logit = ens(val_logits), ens(test_logits)
    v_prob_raw = 1 / (1 + np.exp(-v_logit))
    t_prob_raw = 1 / (1 + np.exp(-t_logit))

    # ---- calibration and thresholding: validation only -------------------
    temperature, cal = 1.0, None
    if tcfg.calibration == "temperature":
        temperature = fit_temperature(v_logit, val_ref["y"])
        v_prob = apply_temperature(v_logit, temperature)
        t_prob = apply_temperature(t_logit, temperature)
    elif tcfg.calibration == "isotonic":
        cal = fit_isotonic(v_prob_raw, val_ref["y"])
        v_prob, t_prob = cal(v_prob_raw), cal(t_prob_raw)
    else:
        v_prob, t_prob = v_prob_raw, t_prob_raw

    thr = best_threshold(val_ref["y"], v_prob, "f1", max_far=max_far)

    results = {
        "preset": preset, "family": family, "protocol": tcfg.protocol,
        "n_params": params,
        "temperature": temperature, "threshold": thr,
        "val": evaluate(val_ref["y"], v_prob, thr,
                        val_ref["event"], val_ref["day"], val_ref["node"]),
        "test": evaluate(test_ref["y"], t_prob, thr,
                         test_ref["event"], test_ref["day"], test_ref["node"]),
        "test_uncalibrated": evaluate(test_ref["y"], t_prob_raw, thr),
        "model_config": asdict(mcfg), "train_config": asdict(tcfg),
    }

    print("\n=== TEST ===")
    for k in ("pr_auc", "roc_auc", "brier", "ece", "pod", "far", "csi", "f1",
              "event_detection_rate", "mean_lead_days"):
        v = results["test"].get(k)
        if v is not None:
            print(f"  {k:22s} {v:.4f}")
    print(f"  {'ECE (uncalibrated)':22s} {results['test_uncalibrated']['ece']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # `tag` keeps a variant run from overwriting the canonical one — re-running
        # a single-seed ladder step as an ensemble must not destroy the n=1 result
        # it is meant to be compared against.
        stem = f"{preset}_{tcfg.protocol}" + (f"_{tag}" if tag else "")
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        np.savez_compressed(
            os.path.join(out_dir, f"{stem}_preds.npz"),
            test_prob=t_prob, test_y=test_ref["y"], test_day=test_ref["day"],
            test_node=test_ref["node"], test_event=test_ref["event"])
        print(f"[run] wrote {out_dir}/{stem}.json")
    return results


# --------------------------------------------------------------------- CLI

def add_common_args(ap) -> None:
    """The argparse flags every family's `train.py` accepts identically."""
    ap.add_argument("--protocol", default=None,
                    choices=["temporal", "basin", "event", "random"])
    ap.add_argument("--root", default=None, help="dir holding flood_dataset.parquet")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-far", type=float, default=None,
                    help="cap false-alarm ratio when picking the threshold")
    ap.add_argument("--sar-root", default=None, help="dir holding image_dataset.csv")
    ap.add_argument("--image-px", type=int, default=None,
                    help="downscale SAR frames (256 halves VRAM vs 512)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename, so a variant run does "
                         "not overwrite the canonical one")
    ap.add_argument("--quiet", action="store_true")
