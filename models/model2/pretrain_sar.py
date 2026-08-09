"""Pretrain the SAR encoder on the labelled flood/dry chips, then transfer it.

    python -m model2.pretrain_sar --out runs/sar_encoder.pt --epochs 25
    python -m model2.train --preset N6_gated --sar-pretrained runs/sar_encoder.pt

Why this exists
---------------
Model 1's SAR CNN branch (M6_cnn) returned nothing: validation event PR-AUC
0.2988 against 0.3007 for the same model without imagery, at twenty times the
parameters and seven times the training time. That is the expected outcome, not
a surprising one. A ResNet-18 initialised from scratch was being asked to learn
a visual representation of flooding from a few thousand frames while the only
gradient reaching it came through a target with a 1.9% positive rate. There is
not enough signal in that setting to learn what water looks like.

`data/processed/image_manifest.csv` is a far better teacher for exactly that
question, and it was going unused: 3,489 chips over all 51 nodes, labelled
flood/dry at a **40% positive rate**, with a graded severity column. Training
the encoder on that task first, and only then attaching it, separates "can a CNN
see flooding in Sentinel-1?" from "does seeing flooding help predict tomorrow's
discharge?" — two questions M6_cnn had confounded into one null result.

Leakage
-------
Pretraining labels are derived from flood events, so a chip that falls in the
main experiment's **test** block must never be trained on. Rather than assume
where that block starts, this script joins each chip to the panel's own
`split_temporal` column and keeps only rows the main experiment calls train or
val. Chips that cannot be matched to a panel row are dropped, and every count is
printed so a silent join failure is visible in the log rather than showing up
later as an inexplicably good test score.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from floodlib.blocks import SarCNN
from floodlib.data import PARQUET_NAME, resolve_root
from floodlib.engine import resolve_device, set_seed
from floodlib.metrics import average_precision, roc_auc
from floodlib.sar import FrameStore, load_index

MANIFEST = "image_manifest.csv"


# ------------------------------------------------------------------- labelling

def label_frames(idx: pd.DataFrame, root: Optional[str] = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Attach a flood/dry label and a train/val split to every usable SAR frame.

    Returns one row per matched frame with `frame_pos` (its positional index into
    `idx`, which is what `FrameStore.get` takes), `label` and `split`.
    """
    data_root = resolve_root(root)
    man_path = os.path.join(data_root, MANIFEST)
    if not os.path.exists(man_path):
        raise FileNotFoundError(
            f"{MANIFEST} not found in {data_root}. It ships with the tabular "
            "dataset and carries the chip labels this script trains on.")

    man = pd.read_csv(man_path)
    man["target_date"] = pd.to_datetime(man["target_date"])
    man = (man[["node_id", "target_date", "label", "severity"]]
           .drop_duplicates(subset=["node_id", "target_date"]))

    frames = idx.reset_index(drop=True).copy()
    frames["frame_pos"] = np.arange(len(frames), dtype=np.int64)
    frames["date"] = pd.to_datetime(frames["date"])

    merged = frames.merge(man, how="inner",
                          left_on=["site_id", "date"],
                          right_on=["node_id", "target_date"])
    if verbose:
        print(f"[pretrain] {len(frames)} frames · {len(man)} labelled chips · "
              f"{len(merged)} matched on (site_id, date)")
    if merged.empty:
        raise RuntimeError(
            "no frame matched the manifest on (site_id, date). Check that "
            "image_dataset.csv's site_id uses the same node codes as "
            "image_manifest.csv's node_id.")

    # ---- split from the panel, so the main experiment's test block is safe --
    panel_df = pd.read_parquet(os.path.join(data_root, PARQUET_NAME),
                               columns=["node_id", "date", "split_temporal"])
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    merged = merged.merge(panel_df, how="left", on=["node_id", "date"])

    before = len(merged)
    merged = merged[merged["split_temporal"].isin(["train", "val"])].copy()
    merged = merged.rename(columns={"split_temporal": "split"})
    merged["label"] = merged["label"].astype(np.float32)
    if verbose:
        print(f"[pretrain] dropped {before - len(merged)} chips in the main test "
              f"block or unmatched to the panel; {len(merged)} remain")
        print(f"[pretrain] train {int((merged['split'] == 'train').sum())} · "
              f"val {int((merged['split'] == 'val').sum())} · "
              f"positive rate {merged['label'].mean():.1%}")
    if merged.empty:
        raise RuntimeError("every labelled chip fell outside train/val")
    return merged.reset_index(drop=True)


# ------------------------------------------------------------------- batching

def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flips and quarter turns. A flood chip is still a flood chip mirrored, and
    SAR imagery has no canonical up, so these are label-preserving by
    construction — unlike brightness jitter, which would corrupt the dB scale
    the water/land contrast lives on."""
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    if rng.random() < 0.5:
        x = x[:, ::-1, :]
    k = int(rng.integers(4))
    if k:
        x = np.rot90(x, k, axes=(1, 2))
    return np.ascontiguousarray(x)


class ChipBatcher:
    def __init__(self, table: pd.DataFrame, store: FrameStore,
                 batch_size: int = 32, shuffle: bool = False,
                 augment: bool = False, seed: int = 0):
        self.pos = table["frame_pos"].to_numpy(np.int64)
        self.y = table["label"].to_numpy(np.float32)
        self.store = store
        self.bs, self.shuffle, self.augment = batch_size, shuffle, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.pos) / self.bs))

    @property
    def pos_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else float("nan")

    def __iter__(self):
        order = np.arange(len(self.pos))
        if self.shuffle:
            self.rng.shuffle(order)
        for i in range(0, len(order), self.bs):
            sel = order[i: i + self.bs]
            imgs = [self.store.get(int(self.pos[j])) for j in sel]
            if self.augment:
                imgs = [_augment(a, self.rng) for a in imgs]
            yield np.stack(imgs), self.y[sel]


# ------------------------------------------------------------------- training

class ChipClassifier(nn.Module):
    """The transferable encoder plus a throwaway 1-logit head.

    Only `encoder` is saved for transfer; `out` exists to create the gradient
    and is discarded, which is why `load_pretrained_encoder` tolerates a head
    that does not match.
    """

    def __init__(self, in_ch: int = 2, feat_dim: int = 64):
        super().__init__()
        self.encoder = SarCNN(in_ch, feat_dim)
        self.out = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.gelu(self.encoder(x))).squeeze(-1)


@torch.no_grad()
def _score(model: nn.Module, batcher: ChipBatcher,
           device: torch.device) -> Tuple[float, float]:
    model.eval()
    ps, ys = [], []
    for xb, yb in batcher:
        p = torch.sigmoid(model(torch.as_tensor(xb, device=device)))
        ps.append(p.detach().cpu().numpy())
        ys.append(yb)
    if not ps:
        return float("nan"), float("nan")
    p, y = np.concatenate(ps), np.concatenate(ys)
    return average_precision(y, p), roc_auc(y, p)


def pretrain(root: Optional[str] = None, sar_root: Optional[str] = None,
             out_path: str = "runs/sar_encoder.pt", epochs: int = 25,
             batch_size: int = 32, lr: float = 3e-4, px: int = 256,
             feat_dim: int = 64, patience: int = 6, seed: int = 0,
             device_spec: str = "auto", verbose: bool = True) -> Dict:
    device = resolve_device(device_spec)
    set_seed(seed)

    idx, sroot = load_index(sar_root)
    table = label_frames(idx, root, verbose)
    store = FrameStore(idx, sroot, px=px, cache_size=1024)

    tr_tab = table[table["split"] == "train"]
    va_tab = table[table["split"] == "val"]
    tr = ChipBatcher(tr_tab, store, batch_size, shuffle=True, augment=True, seed=seed)
    va = ChipBatcher(va_tab, store, batch_size, shuffle=False, augment=False)

    model = ChipClassifier(2, feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # 40% positives — near enough to balanced that plain BCE is the right choice
    # here, in deliberate contrast to the 1.9% downstream task.
    crit = nn.BCEWithLogitsLoss()

    if verbose:
        print(f"[pretrain] {device} · {len(tr_tab)} train / {len(va_tab)} val chips "
              f"at {px}px · encoder {sum(p.numel() for p in model.encoder.parameters()):,} params")

    best_ap, best_state, bad, history = -np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for xb, yb in tr:
            logit = model(torch.as_tensor(xb, device=device))
            loss = crit(logit, torch.as_tensor(yb, device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        ap, auc = _score(model, va, device)
        history.append({"epoch": ep, "loss": tot / max(nb, 1),
                        "val_ap": ap, "val_auc": auc})
        if verbose:
            print(f"  ep {ep:3d}  loss {tot / max(nb,1):.4f}  "
                  f"val AP {ap:.4f}  val ROC-AUC {auc:.4f}  ({time.time()-t0:.1f}s)")

        if np.isfinite(ap) and ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val AP {best_ap:.4f})")
                break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta = {"val_ap": float(best_ap), "px": px, "feat_dim": feat_dim,
            "n_train": int(len(tr_tab)), "n_val": int(len(va_tab)),
            "pos_rate_train": tr.pos_rate, "history": history}
    torch.save({"encoder": best_state, "meta": meta}, out_path)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    if verbose:
        print(f"[pretrain] best val AP {best_ap:.4f} → {out_path}")
        print("  A chip-level AP near the 40% base rate means the encoder learned "
              "nothing visual, and N6_gated should not be expected to help.")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pretrain the SAR encoder on labelled flood/dry chips.")
    ap.add_argument("--root", default=None, help="dir holding flood_dataset.parquet")
    ap.add_argument("--sar-root", default=None, help="dir holding image_dataset.csv")
    ap.add_argument("--out", default="runs/sar_encoder.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-px", type=int, default=256)
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    pretrain(a.root, a.sar_root, a.out, a.epochs, a.batch_size, a.lr,
             a.image_px, a.feat_dim, a.patience, a.seed, a.device, not a.quiet)


if __name__ == "__main__":
    main()
