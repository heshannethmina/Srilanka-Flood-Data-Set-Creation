"""Training and evaluation loop for the STGCN baseline architecture."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    from tfstgnn.calibrate import apply_temperature, fit_isotonic, fit_temperature
    from tfstgnn.data import Panel, SnapshotBatcher, build_panel, load_panel
    from tfstgnn.graph import build_graph
    from tfstgnn.losses import MultiHeadLoss
    from tfstgnn.metrics import best_threshold, evaluate
except (ImportError, ValueError):
    from ..tfstgnn.calibrate import apply_temperature, fit_isotonic, fit_temperature
    from ..tfstgnn.data import Panel, SnapshotBatcher, build_panel, load_panel
    from ..tfstgnn.graph import build_graph
    from ..tfstgnn.losses import MultiHeadLoss
    from ..tfstgnn.metrics import best_threshold, evaluate

try:
    from stgcn.config import STGCNModelConfig, STGCNTrainConfig
    from stgcn.stgcn_model import STGCNModel
except (ImportError, ValueError):
    from .config import STGCNModelConfig, STGCNTrainConfig
    from .stgcn_model import STGCNModel



def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect(model: STGCNModel, batcher: SnapshotBatcher, S: torch.Tensor,
            edge_index: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Run model over a split and return predictions and targets for valid node-days."""
    model.eval()
    logits, ys, days, nodes, events = [], [], [], [], []
    for b in batcher:
        x = torch.as_tensor(b["x"], device=device)
        out = model(x, S, edge_index)
        m = b["mask"] > 0
        lg = out["logits"][..., 0].detach().cpu().numpy()
        logits.append(lg[m])
        ys.append(b["y"][..., 0][m])
        events.append(b["event"][m])
        d = np.repeat(b["days"][:, None], m.shape[1], axis=1)
        n = np.repeat(np.arange(m.shape[1])[None, :], m.shape[0], axis=0)
        days.append(d[m])
        nodes.append(n[m])

    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return {
        "logit": cat(logits),
        "y": cat(ys),
        "day": cat(days),
        "node": cat(nodes),
        "event": cat(events),
    }


def train_one(panel: Panel, mcfg: STGCNModelConfig, tcfg: STGCNTrainConfig,
              graph, device: torch.device, verbose: bool = True
              ) -> Tuple[STGCNModel, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    set_seed(tcfg.seed)
    edge_index = torch.as_tensor(graph.edge_index, device=device)
    model = STGCNModel(mcfg, n_nodes=panel.S.shape[0]).to(device)
    S = torch.as_tensor(panel.S, device=device)

    mk = lambda split, sh: SnapshotBatcher(panel, tcfg.protocol, split, mcfg.lookback,
                                           tcfg.batch_size, shuffle=sh)
    tr, va, te = mk("train", True), mk("val", False), mk("test", False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[STGCN train] params {n_params:,} | "
              f"train {tr.n_samples:,} ({tr.pos_rate:.3%} pos) · "
              f"val {va.n_samples:,} ({va.pos_rate:.3%}) · "
              f"test {te.n_samples:,} ({te.pos_rate:.3%})")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    def lr_sched(epoch: int) -> float:
        if epoch < tcfg.warmup_epochs:
            return (epoch + 1) / max(1, tcfg.warmup_epochs)
        progress = (epoch - tcfg.warmup_epochs) / max(1, tcfg.epochs - tcfg.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_sched)
    criterion = MultiHeadLoss(tcfg, panel.cls_heads)

    best_score = -1.0
    best_weights = None
    stagnant = 0

    for epoch in range(tcfg.epochs):
        model.train()
        total_loss, total_cls, total_reg, n_batches = 0.0, 0.0, 0.0, 0
        for b in tr:
            x = torch.as_tensor(b["x"], device=device)
            y = torch.as_tensor(b["y"], device=device)
            r = torch.as_tensor(b["r"], device=device)
            m = torch.as_tensor(b["mask"], device=device)
            c = torch.as_tensor(b["conf"], device=device)

            opt.zero_grad()
            out = model(x, S, edge_index)
            loss_dict = criterion(out, y, r, m, c)
            loss = loss_dict["loss"]
            loss.backward()

            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()

            total_loss += float(loss.detach())
            total_cls += float(loss_dict["cls"].detach())
            total_reg += float(loss_dict["reg"].detach())
            n_batches += 1


        sched.step()

        # Validation check
        val_pred = collect(model, va, S, edge_index, device)
        val_p = 1.0 / (1.0 + np.exp(-val_pred["logit"]))
        ev_metrics = evaluate(val_pred["y"], val_p, 0.5, val_pred["event"],
                              val_pred["day"], val_pred["node"])
        val_score = ev_metrics[tcfg.select_metric]

        if val_score > best_score:
            best_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1

        if verbose and (epoch % tcfg.log_every == 0 or epoch == tcfg.epochs - 1):
            print(f"  epoch {epoch:2d}/{tcfg.epochs} | "
                  f"loss {total_loss/n_batches:6.4f} (cls {total_cls/n_batches:6.4f} reg {total_reg/n_batches:6.4f}) | "
                  f"val {tcfg.select_metric} {val_score:6.4f} (best {best_score:6.4f})")

        if stagnant >= tcfg.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (patience {tcfg.patience})")
            break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    val_res = collect(model, va, S, edge_index, device)
    test_res = collect(model, te, S, edge_index, device)
    return model, val_res, test_res


def run_stgcn(
    mcfg: Optional[STGCNModelConfig] = None,
    tcfg: Optional[STGCNTrainConfig] = None,
    root: Optional[str] = None,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    mcfg = mcfg or STGCNModelConfig()
    tcfg = tcfg or STGCNTrainConfig()
    device = resolve_device(tcfg.device)

    df, nodes, edges = load_panel(root)
    panel = build_panel(df, nodes, truncate_after=tcfg.truncate_after)
    graph = build_graph(edges, nodes, panel.node_ids, mode="both")

    t0 = time.time()

    val_logits_list, val_ys_list = [], []
    test_logits_list, test_ys_list = [], []

    for seed_idx in range(tcfg.n_seeds):
        cfg_seed = replace_seed(tcfg, tcfg.seed + seed_idx)
        if verbose and tcfg.n_seeds > 1:
            print(f"\n--- Seed {seed_idx + 1}/{tcfg.n_seeds} (seed={cfg_seed.seed}) ---")
        model, va_res, te_res = train_one(panel, mcfg, cfg_seed, graph, device, verbose=verbose)
        val_logits_list.append(va_res["logit"])
        val_ys_list.append(va_res["y"])
        test_logits_list.append(te_res["logit"])
        test_ys_list.append(te_res["y"])

    # Average logits over seeds
    val_logit_avg = np.mean(val_logits_list, axis=0)
    test_logit_avg = np.mean(test_logits_list, axis=0)
    y_val, y_test = val_ys_list[0], test_ys_list[0]

    # Convert logits to probabilities
    val_prob = 1.0 / (1.0 + np.exp(-val_logit_avg))
    test_prob = 1.0 / (1.0 + np.exp(-test_logit_avg))

    # Calibration fitting on validation set
    if tcfg.calibration == "temperature":
        T_opt = fit_temperature(val_logit_avg, y_val)
        val_prob = apply_temperature(val_logit_avg, T_opt)
        test_prob = apply_temperature(test_logit_avg, T_opt)
        if verbose:
            print(f"[STGCN calibrate] temperature T={T_opt:.3f}")
    elif tcfg.calibration == "isotonic":
        cal = fit_isotonic(val_prob, y_val)
        val_prob, test_prob = cal(val_prob), cal(test_prob)
        if verbose:
            print("[STGCN calibrate] isotonic regression")

    # Optimal threshold selection on validation set
    best_thr = best_threshold(y_val, val_prob, criterion="f1")


    # Final evaluation
    val_eval = evaluate(y_val, val_prob, best_thr, va_res["event"], va_res["day"], va_res["node"])
    test_eval = evaluate(y_test, test_prob, best_thr, te_res["event"], te_res["day"], te_res["node"])

    elapsed = time.time() - t0

    results = {
        "model": "STGCN",
        "protocol": tcfg.protocol,
        "elapsed_sec": round(elapsed, 2),
        "threshold": float(best_thr),
        "mcfg": asdict(mcfg),
        "tcfg": asdict(tcfg),
        "val": val_eval,
        "test": test_eval,
    }

    if verbose:
        print(f"\n=== STGCN RESULTS ({tcfg.protocol}) ===")
        print(f"Elapsed: {elapsed:.1f}s | Optimal Threshold: {best_thr:.4f}")
        t = test_eval
        print(f"{'Model':10s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>8s} "
              f"{'ECE':>7s} {'POD':>6s} {'FAR':>6s} {'CSI':>6s}")
        print(f"{'STGCN':10s} {t['pr_auc']:8.4f} {t['roc_auc']:8.4f} "
              f"{t['brier']:8.5f} {t['ece']:7.4f} {t['pod']:6.3f} "
              f"{t['far']:6.3f} {t['csi']:6.3f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"stgcn_{tcfg.protocol}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[STGCN] Results written to {out_path}")

    return results


def replace_seed(tcfg: STGCNTrainConfig, new_seed: int) -> STGCNTrainConfig:
    from dataclasses import replace
    return replace(tcfg, seed=new_seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate STGCN baseline architecture.")
    ap.add_argument("--protocol", default="temporal", choices=["temporal", "basin", "random"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    mcfg = STGCNModelConfig()
    tcfg = STGCNTrainConfig(protocol=a.protocol, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
    run_stgcn(mcfg, tcfg, root=a.root, out_dir=a.out)


if __name__ == "__main__":
    main()
