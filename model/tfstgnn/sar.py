"""Sentinel-1 branch — joins the SAR image dataset to the tabular panel.

Source: kaggle.com/datasets/uom230429e/flood-data-set (1.76 GB, 2,578 frames,
9 of the 51 nodes). Frames are 8-bit PNG; R = VV dB, G = VH dB, B = VV−VH dB,
each channel a linear quantisation of the stated dB range.

Two things this module gets right and a naive join would not:

1. **Causality.** Sentinel-1 revisits every ~12 days (§11.2), so most node-days
   have no frame. Each day is given the most recent frame *at or before* that
   day, never a future one, together with its age in days.
2. **Presence.** 42 of the 51 nodes have no imagery at all. They carry a zero
   embedding and a zero presence flag rather than being dropped, so the graph
   stays whole.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import SAR_SCALARS

VV_RANGE = (-30.0, 5.0)
VH_RANGE = (-30.0, 5.0)
MAX_AGE_DAYS = 60          # older than this is treated as "no observation"


def resolve_sar_root(root: Optional[str] = None) -> str:
    candidates = [c for c in [
        root,
        os.environ.get("SAR_DATA_ROOT"),
        "/kaggle/input/flood-data-set/sar_flood_ds",
        "/kaggle/input/flood-data-set/sar_flood_lite",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                     "data", "sar_flood_ds")),
    ] if c]
    for c in candidates:
        if os.path.exists(os.path.join(c, "image_dataset.csv")):
            return c
    raise FileNotFoundError(
        "image_dataset.csv not found in any of: " + ", ".join(candidates) +
        "\nOn Kaggle add the dataset uom230429e/flood-data-set; locally set "
        "SAR_DATA_ROOT.")


def load_index(root: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    root = resolve_sar_root(root)
    idx = pd.read_csv(os.path.join(root, "image_dataset.csv"))
    idx["date"] = pd.to_datetime(idx["date"])
    return idx, root


# ------------------------------------------------------- frame → node-day map

def build_frame_map(idx: pd.DataFrame, node_ids: Sequence[str],
                    dates: np.ndarray, max_age: int = MAX_AGE_DAYS
                    ) -> Dict[str, np.ndarray]:
    """For every (day, node): index of the newest frame at or before that day.

    Returns `frame`  [T, N] int32 (-1 = none) and `age` [T, N] float32 (days).
    """
    T, N = len(dates), len(node_ids)
    nmap = {n: i for i, n in enumerate(node_ids)}
    dates = np.asarray(dates).astype("datetime64[D]")

    # A frame is attached to the first panel day at or *after* its acquisition,
    # never before: attaching it earlier would let the model see an observation
    # that had not yet been made. On a daily panel this is the exact date.
    frame = np.full((T, N), -1, np.int32)
    acq = idx["date"].to_numpy().astype("datetime64[D]")
    slot = np.searchsorted(dates, acq, side="left")
    for pos, (site, d) in enumerate(zip(idx["site_id"], slot)):
        n = nmap.get(site)
        if n is not None and 0 <= d < T:
            frame[d, n] = pos

    # forward-fill: carry the last observation forward, tracking its age
    age = np.full((T, N), np.inf, np.float32)
    last = np.full(N, -1, np.int32)
    last_day = np.full(N, -1, np.int64)
    for t in range(T):
        new = frame[t] >= 0
        last = np.where(new, frame[t], last)
        last_day = np.where(new, t, last_day)
        frame[t] = last
        age[t] = np.where(last >= 0, t - last_day, np.inf)

    stale = age > max_age
    frame[stale] = -1
    age[stale] = 0.0
    return {"frame": frame, "age": np.nan_to_num(age, posinf=0.0)}


# ------------------------------------------------------ mode 1: scalar branch

def sar_scalar_channels(idx: pd.DataFrame, node_ids: Sequence[str],
                        dates: np.ndarray, fmap: Dict[str, np.ndarray]
                        ) -> Tuple[np.ndarray, List[str]]:
    """[T, N, 6] — the four per-frame scalars plus presence and normalised age.

    Standardised with statistics over the frames themselves (which are
    observations, not labels, so this is not a leakage path); absent node-days
    get 0, which after standardisation is the observed mean.
    """
    T, N = len(dates), len(node_ids)
    cols = [c for c in SAR_SCALARS if c in idx.columns]
    vals = idx[cols].to_numpy(np.float32)
    mu = np.nanmean(vals, axis=0, keepdims=True)
    sd = np.nanstd(vals, axis=0, keepdims=True) + 1e-6
    vals = np.nan_to_num((vals - mu) / sd, nan=0.0)

    frame, age = fmap["frame"], fmap["age"]
    out = np.zeros((T, N, len(cols) + 2), np.float32)
    has = frame >= 0
    flat = frame[has]
    out[..., : len(cols)][has] = vals[flat]
    out[..., len(cols)] = has.astype(np.float32)
    out[..., len(cols) + 1] = age / 30.0
    return out, cols + ["sar_present", "sar_age_30d"]


def attach_sar_scalars(panel, sar_root: Optional[str] = None):
    """Widen `panel.X` with the SAR scalar channels. Returns the new channel names."""
    idx, _ = load_index(sar_root)
    fmap = build_frame_map(idx, panel.node_ids, panel.dates)
    extra, names = sar_scalar_channels(idx, panel.node_ids, panel.dates, fmap)
    panel.X = np.concatenate([panel.X, extra], axis=-1)
    panel.features = list(panel.features) + names
    n_imaged = int((fmap["frame"] >= 0).any(axis=0).sum())
    print(f"[sar] +{len(names)} channels; imagery present at "
          f"{n_imaged}/{len(panel.node_ids)} nodes")
    return names


# --------------------------------------------------------- mode 2: CNN branch

class FrameStore:
    """Lazy PNG → float32 [2, P, P] dB loader with an LRU cache.

    Caching decoded frames matters: at a 12-day revisit the same frame is reused
    for ~12 consecutive days, so the hit rate is high and disk I/O stops being
    the bottleneck of the CNN ablation.
    """

    def __init__(self, idx: pd.DataFrame, root: str, px: Optional[int] = None,
                 cache_size: int = 512):
        from PIL import Image  # noqa: F401  (fail early with a clear message)
        self.idx = idx.reset_index(drop=True)
        self.root = root
        self.px = px
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.cache_size = cache_size

    def __len__(self) -> int:
        return len(self.idx)

    def get(self, i: int) -> np.ndarray:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        from PIL import Image
        path = os.path.join(self.root, self.idx.loc[i, "png_path"])
        img = Image.open(path)
        if self.px and img.size[0] != self.px:
            img = img.resize((self.px, self.px), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        vv = a[..., 0] / 255.0 * (VV_RANGE[1] - VV_RANGE[0]) + VV_RANGE[0]
        vh = a[..., 1] / 255.0 * (VH_RANGE[1] - VH_RANGE[0]) + VH_RANGE[0]
        # scale to roughly unit range; dB in [-30, 5] → [-1, 1]-ish
        out = np.stack([(vv + 12.5) / 17.5, (vh + 12.5) / 17.5]).astype(np.float32)
        self.cache[i] = out
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return out

    def batch(self, frame_idx: np.ndarray, px: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """frame_idx [B, N] → (packed images [K, 2, px, px], pos [K], mask [B, N]).

        Only the K node-days that actually have a frame are materialised. With
        9 imaged nodes out of 51 that is a 5.7x saving in both memory and CNN
        compute — at 512 px the dense form would be a 3.4 GB batch tensor.
        """
        B, N = frame_idx.shape
        mask = (frame_idx >= 0).astype(np.float32)
        pos = np.flatnonzero(frame_idx.reshape(-1) >= 0).astype(np.int64)
        flat = frame_idx.reshape(-1)
        imgs = (np.stack([self.get(int(flat[p])) for p in pos])
                if pos.size else np.zeros((0, 2, px, px), np.float32))
        return imgs, pos, mask
