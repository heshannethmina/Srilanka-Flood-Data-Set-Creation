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
              ) -> Tuple[nn.Module, Dict[str, np.ndarray], Dict[str, np.ndarray],
                         List[Dict]]:
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
    history: List[Dict] = []
    for ep in range(tcfg.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        cls_tot = reg_tot = 0.0
        head_tot: Optional[np.ndarray] = None
        gn_sum, gn_max, clipped = 0.0, 0.0, 0
        for b in tr:
            x = torch.as_tensor(b["x"], device=device)
            img, ipos, imask, age = _image_inputs(b, store, px, device)
            out = model(x, S, img, ipos, imask, age)
            parts = crit(out,
                         torch.as_tensor(b["y"], device=device),
                         torch.as_tensor(b["r"], device=device),
                         torch.as_tensor(b["mask"], device=device),
                         torch.as_tensor(b["conf"], device=device))
            loss = parts["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip_grad_norm_ returns the total norm *before* clipping; it was
            # being thrown away, and it is the cheapest signal there is for
            # whether the learning rate is sane.
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                      tcfg.grad_clip))
            opt.step()
            gn_sum += gn
            gn_max = max(gn_max, gn)
            clipped += int(gn > tcfg.grad_clip)
            tot += float(loss.detach())
            cls_tot += float(parts["cls"])
            reg_tot += float(parts["reg"])
            ph = parts["per_head"].detach().cpu().numpy()
            head_tot = ph if head_tot is None else head_tot + ph
            nb += 1
        sched.step()

        vp = collect(model, va, S, device, px, store)
        vm = evaluate(vp["y"], 1 / (1 + np.exp(-vp["logit"])), 0.5,
                      vp["event"], vp["day"], vp["node"])
        score = vm.get("event_pr_auc", np.nan)
        if not np.isfinite(score):
            score = vm["pr_auc"]

        d = max(nb, 1)
        history.append({
            "epoch": ep, "train_loss": tot / d,
            "train_cls": cls_tot / d, "train_reg": reg_tot / d,
            "per_head": (head_tot / d).tolist() if head_tot is not None else None,
            "grad_norm_mean": gn_sum / d, "grad_norm_max": gn_max,
            "clip_fraction": clipped / d,
            "lr": float(opt.param_groups[0]["lr"]),
            "val_pr_auc": vm["pr_auc"], "val_event_pr_auc": float(score),
            "val_ece": vm["ece"], "val_brier": vm.get("brier"),
            "val_pod": vm.get("pod"), "val_far": vm.get("far"),
            "seconds": time.time() - t0,
        })

        if verbose and (ep % tcfg.log_every == 0):
            print(f"  ep {ep:3d}  loss {tot / d:.5f}  "
                  f"val PR-AUC {vm['pr_auc']:.4f}  event PR-AUC {score:.4f}  "
                  f"ECE {vm['ece']:.4f}  |grad| {gn_sum / d:.2f} "
                  f"clip {clipped / d:.0%}  ({time.time()-t0:.1f}s)")

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
            collect(model, te, S, device, px, store), history)


# ------------------------------------------------------------------ pipeline

def run(preset: str, presets: Dict[str, Preset], build_model: ModelBuilder, *,
        family: str = "model", protocol: Optional[str] = None,
        root: Optional[str] = None, out_dir: Optional[str] = None,
        epochs: Optional[int] = None, n_seeds: Optional[int] = None,
        device_spec: str = "auto", max_far: Optional[float] = None,
        verbose: bool = True, sar_root: Optional[str] = None,
        image_px: Optional[int] = None, batch_size: Optional[int] = None,
        tag: Optional[str] = None, train_overrides: Optional[Dict] = None,
        **model_overrides) -> Dict:
    """Train `preset` from `presets` and write its result JSON.

    `train_overrides` and `model_overrides` are applied on top of the preset, so
    a hyperparameter sweep is a sequence of command lines rather than a sequence
    of edits to `config.py` — which matters because on Kaggle every edit costs a
    commit, a push and a re-clone.

    When either is non-empty and no explicit `tag` is given, one is derived from
    the overrides so two points of a sweep cannot overwrite each other.
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

    tover = {k: v for k, v in (train_overrides or {}).items() if v is not None}
    if tover:
        tcfg = replace(tcfg, **tover)
    # A knob one family has and the other does not (model2's `d_model`, say) is
    # dropped with a warning rather than raising: a sweep script shared between
    # the two families should not die on the first irrelevant flag.
    mover = {k: v for k, v in model_overrides.items() if v is not None}
    unknown = [k for k in mover if not hasattr(mcfg, k)]
    if unknown and verbose:
        print(f"[run] ignoring {unknown}: not a field of "
              f"{type(mcfg).__name__}")
    mover = {k: v for k, v in mover.items() if hasattr(mcfg, k)}
    if mover:
        mcfg = replace(mcfg, **mover)
    if tover or mover:
        shown = ", ".join(f"{k}={v}" for k, v in sorted({**tover, **mover}.items()))
        if tag is None:
            tag = auto_tag({**tover, **mover})
        if verbose:
            print(f"[run] overrides: {shown}  ->  tag '{tag}'")

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
    histories: List[List[Dict]] = []
    per_seed: List[Dict] = []
    gate_stats: List[float] = []
    val_ref = test_ref = None
    params = 0
    for k in range(max(tcfg.n_seeds, 1)):
        seed_cfg = replace(tcfg, seed=tcfg.seed + k)
        if verbose and tcfg.n_seeds > 1:
            print(f"[seed {k+1}/{tcfg.n_seeds}]")
        model, vp, tp, hist = train_one(build_model, panel, mcfg, seed_cfg, graph,
                                        device, verbose, frame_map, store)
        params = n_params(model)
        # A gated multimodal branch that never opens its gate contributes
        # nothing; without this the only evidence would be a metric difference
        # too small to attribute.
        gate = getattr(getattr(model, "sar", None), "last_gate_mean", None)
        if gate is not None:
            gate_stats.append(float(gate))
        del model                       # a 5-seed CNN ensemble will not fit otherwise
        histories.append(hist)
        # Each seed scored on its own, so the spread between seeds is knowable
        # and a small gap between two ladder rungs can be judged against it.
        per_seed.append(evaluate(tp["y"], 1 / (1 + np.exp(-tp["logit"])), 0.5,
                                 tp["event"], tp["day"], tp["node"]))
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

    diag = build_diagnostics(results, histories, per_seed, panel, nodes,
                             t_prob, test_ref, thr, tcfg, gate_stats, root, verbose)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # `tag` keeps a variant run from overwriting the canonical one — re-running
        # a single-seed ladder step as an ensemble must not destroy the n=1 result
        # it is meant to be compared against.
        stem = f"{preset}_{tcfg.protocol}" + (f"_{tag}" if tag else "")
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        with open(os.path.join(out_dir, f"{stem}_diag.json"), "w") as f:
            json.dump(diag, f, indent=2, default=float)
        np.savez_compressed(
            os.path.join(out_dir, f"{stem}_preds.npz"),
            test_prob=t_prob, test_y=test_ref["y"], test_day=test_ref["day"],
            test_node=test_ref["node"], test_event=test_ref["event"])
        print(f"[run] wrote {out_dir}/{stem}.json and {stem}_diag.json")
    return results


# ---------------------------------------------------------------- diagnostics

def build_diagnostics(results: Dict, histories: List[List[Dict]],
                      per_seed: List[Dict], panel: Panel, nodes,
                      t_prob: np.ndarray, test_ref: Dict, thr: float,
                      tcfg: TrainConfig, gate_stats: List[float],
                      root: Optional[str] = None, verbose: bool = True) -> Dict:
    """Assemble everything needed to decide what to change next.

    Wrapped section by section: a bug in a diagnostic must never destroy a run
    that has already finished training, so each block degrades to an error string
    rather than raising. Six hours of GPU time is not worth a KeyError.
    """
    from . import diagnostics as dg

    diag: Dict = {"preset": results["preset"], "family": results["family"],
                  "protocol": results["protocol"], "threshold": thr,
                  "history": histories, "per_seed": per_seed}

    def attempt(name, fn):
        try:
            diag[name] = fn()
        except Exception as exc:                       # noqa: BLE001
            diag[name] = {"error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"[diag] {name} failed: {type(exc).__name__}: {exc}")

    h0 = histories[0] if histories else []
    attempt("seed_spread", lambda: dg.seed_spread(per_seed))
    attempt("stopping", lambda: dg.stopping_report(h0, tcfg.patience))
    attempt("optimisation", lambda: dg.optimisation_report(h0, tcfg.grad_clip))
    attempt("head_balance", lambda: dg.head_balance(h0, panel.cls_heads))

    # events.csv and nodes.csv carry the context that turns "missed" into
    # "missed a severe flood on an upstream node".
    meta_ev: Dict[str, Dict] = {}
    try:
        meta_ev = _events_meta(root)
    except Exception as exc:                           # noqa: BLE001
        if verbose:
            print(f"[diag] events.csv not joined ({type(exc).__name__}) — "
                  "episodes will have no severity")
    meta_nd: Dict[str, Dict] = {}
    try:
        cols = [c for c in ("basin", "zone", "position", "elevation_m")
                if c in nodes.columns]
        meta_nd = nodes.set_index("node_id")[cols].to_dict("index")
    except Exception:                                  # noqa: BLE001
        pass

    attempt("episodes", lambda: dg.episode_table(
        t_prob, test_ref["y"], test_ref["event"], test_ref["day"],
        test_ref["node"], thr, panel.node_ids, panel.event_ids, meta_ev))
    attempt("by_severity", lambda: dg.missed_by_severity(diag.get("episodes", [])))
    attempt("nodes", lambda: dg.node_table(
        t_prob, test_ref["y"], test_ref["node"], thr, panel.node_ids, meta_nd))
    for by in ("zone", "position", "basin"):
        attempt(f"by_{by}", lambda by=by: dg.group_breakdown(diag.get("nodes", []), by))

    if gate_stats:
        diag["sar_gate_mean"] = float(np.mean(gate_stats))
        diag["sar_gate_per_seed"] = gate_stats
    return diag


def _events_meta(root: Optional[str] = None) -> Dict[str, Dict]:
    """`event_id` → magnitude columns from the dataset's own events table.

    `root` is threaded through from the caller rather than re-resolved: on
    Kaggle the mount path is not one `resolve_root` knows about, and silently
    losing the severity join would gut the most useful diagnostic there is.
    """
    import pandas as pd
    from .data import resolve_root
    path = os.path.join(resolve_root(root), "events.csv")
    ev = pd.read_csv(path)
    cols = [c for c in ("duration_days", "peak_discharge", "basin", "severity")
            if c in ev.columns]
    return ev.set_index("event_id")[cols].to_dict("index")


# --------------------------------------------------------------------- CLI

#: Flags that override `TrainConfig`. Every family shares this config, so all of
#: them always apply.
TRAIN_KNOBS = ("loss", "lr", "weight_decay", "grad_clip", "patience",
               "focal_alpha", "focal_gamma", "reg_weight", "calibration",
               "warmup_epochs", "head_weights")

#: Flags that override a model config. Which of these a family actually has
#: differs — `run` drops the rest with a warning.
MODEL_KNOBS = ("dropout", "lookback", "d_model", "n_layers", "n_heads",
               "ff_mult", "d_emb", "n_freq", "freq_sigma", "feature_attn",
               "fusion_hidden", "fusion_out", "head_hidden", "head_hidden2",
               "gru_hidden", "gru_layers", "gat_layers", "gat_heads")


def auto_tag(overrides: Dict, max_len: int = 48) -> str:
    """A short filesystem-safe slug naming a sweep point.

    Two runs of the same preset with different hyperparameters are different
    models and must not share `runs/<preset>_<protocol>.json`. Deriving the tag
    from the overrides means a sweep cannot silently overwrite itself.
    """
    parts = []
    for k, v in sorted(overrides.items()):
        if isinstance(v, dict):
            v = "-".join(f"{a}{b:g}" for a, b in sorted(v.items()))
        elif isinstance(v, bool):
            v = "on" if v else "off"
        elif isinstance(v, float):
            v = f"{v:g}"
        short = "".join(w[0] for w in k.split("_")) if "_" in k else k[:6]
        parts.append(f"{short}{v}")
    slug = "_".join(parts).replace(".", "p").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    return slug[:max_len]


def parse_head_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
    """`"onset_1d=1.0,flood_2d=0.1"` -> a full `head_weights` dict.

    Names may omit the `target_` prefix. Unlisted heads keep their default, so a
    sweep can move one head without restating the other three.
    """
    if not spec:
        return None
    out = dict(TrainConfig().head_weights)
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k = k.strip()
        k = k if k.startswith("target_") else f"target_{k}"
        if k not in out:
            raise SystemExit(f"[fatal] unknown head '{k}'; "
                             f"choose from {sorted(out)}")
        out[k] = float(v)
    return out


def overrides_from_args(a) -> Tuple[Dict, Dict]:
    """Split parsed argparse flags into (train overrides, model overrides)."""
    train = {k: getattr(a, k, None) for k in TRAIN_KNOBS if k != "head_weights"}
    train["head_weights"] = parse_head_weights(getattr(a, "head_weights", None))
    model = {k: getattr(a, k, None) for k in MODEL_KNOBS}
    return ({k: v for k, v in train.items() if v is not None},
            {k: v for k, v in model.items() if v is not None})


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
                         "not overwrite the canonical one. Derived from the "
                         "override flags below when they are used and this is not")
    ap.add_argument("--quiet", action="store_true")
    add_override_args(ap)


def add_override_args(ap) -> None:
    """Hyperparameter override flags, shared by every entry point.

    Separate from `add_common_args` so `kaggle_run.py`, which has its own
    parser, offers exactly the same knobs by exactly the same names.
    """
    g = ap.add_argument_group(
        "hyperparameter overrides",
        "Applied on top of the preset. Any of these makes the run a distinct "
        "sweep point, written under an auto-derived tag so nothing is "
        "overwritten. Unset flags keep the preset's value.")
    g.add_argument("--loss", default=None,
                   choices=["bce", "wbce", "focal", "focal_conf"])
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--grad-clip", type=float, default=None)
    g.add_argument("--patience", type=int, default=None)
    g.add_argument("--warmup-epochs", type=int, default=None)
    g.add_argument("--focal-alpha", type=float, default=None)
    g.add_argument("--focal-gamma", type=float, default=None)
    g.add_argument("--reg-weight", type=float, default=None,
                   help="weight on the two auxiliary regression heads")
    g.add_argument("--head-weights", default=None,
                   help="e.g. 'onset_1d=1.0,flood_2d=0.1'; unlisted heads keep "
                        "their default")
    g.add_argument("--calibration", default=None,
                   choices=["none", "temperature", "isotonic"])
    g.add_argument("--dropout", type=float, default=None)
    g.add_argument("--lookback", type=int, default=None)
    # Architecture width/depth. A family without one of these ignores it.
    g.add_argument("--d-model", type=int, default=None, help="model2 only")
    g.add_argument("--n-layers", type=int, default=None, help="model2 only")
    g.add_argument("--n-heads", type=int, default=None, help="model2 only")
    g.add_argument("--d-emb", type=int, default=None, help="model2 only")
    g.add_argument("--n-freq", type=int, default=None, help="model2 only")
    g.add_argument("--gru-hidden", type=int, default=None, help="model1 only")
    g.add_argument("--gat-layers", type=int, default=None, help="model1 only")
