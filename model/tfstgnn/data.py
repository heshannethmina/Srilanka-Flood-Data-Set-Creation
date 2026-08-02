"""Dataset assembly: the 410k-row long panel → dense spatiotemporal tensors.

The whole panel is small (8,036 days x 51 nodes x 33 features = ~54 MB float32),
so it is held dense in memory and sliced per batch. No DataLoader, no shuffling
of individual rows: **a sample is one day t, i.e. a full-graph snapshot** with a
`lookback`-day history, exactly as the model consumes it.

Leakage control is enforced structurally here:
  * normalisation statistics are fitted on the training mask only;
  * targets are read from the dataset's own causal columns;
  * the two derived regression targets are built by forward-shifting *within*
    each node's own column of the dense array, so no value ever crosses nodes.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    CLS_HEADS, DYNAMIC_FEATURES, LOG1P_FEATURES, POSITIONS, REG_HEADS,
    STATIC_CONTINUOUS, ZONES, Protocol,
)

PARQUET_NAME = "flood_dataset.parquet"


# --------------------------------------------------------------------- loading

def resolve_root(root: Optional[str] = None) -> str:
    """Find the directory holding flood_dataset.parquet.

    Checks, in order: the explicit argument, $FLOOD_DATA_ROOT, the standard
    Kaggle input mount, and the repo's own data/processed.
    """
    candidates: List[str] = []
    if root:
        candidates.append(root)
    if os.environ.get("FLOOD_DATA_ROOT"):
        candidates.append(os.environ["FLOOD_DATA_ROOT"])
    candidates += [
        "/kaggle/input/sri-lanka-flood-tabular-graph-2003-2025",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                     "data", "processed")),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, PARQUET_NAME)):
            return c
    raise FileNotFoundError(
        f"{PARQUET_NAME} not found in any of: {candidates}\n"
        "Download the tabular dataset (38 MB) from "
        "kaggle.com/datasets/uom230429e/sri-lanka-flood-tabular-graph-2003-2025 "
        "into data/processed/, or set FLOOD_DATA_ROOT."
    )


def load_panel(root: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = resolve_root(root)
    df = pd.read_parquet(os.path.join(root, PARQUET_NAME))
    nodes = pd.read_csv(os.path.join(root, "nodes.csv"))
    edges = pd.read_csv(os.path.join(root, "edges.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df, nodes, edges


# ------------------------------------------------------------------ containers

@dataclass
class Panel:
    """Dense arrays. T = days, N = nodes, F = dynamic features."""
    X: np.ndarray                 # [T, N, F] float32, normalised, nan→0
    S: np.ndarray                 # [N, 9]    float32 static terrain
    Y: np.ndarray                 # [T, N, H] float32 classification targets
    R: np.ndarray                 # [T, N, 2] float32 regression targets
    valid: np.ndarray             # [T, N] float32 valid_sample AND target present
    conf: np.ndarray              # [T, N] float32 label_confidence
    event: np.ndarray             # [T, N] int32, event code or -1
    flood_state: np.ndarray       # [T, N] float32 (for the persistence baseline)
    dates: np.ndarray             # [T] datetime64[D]
    node_ids: List[str]
    basins: List[str]
    features: List[str]
    cls_heads: List[str]
    reg_heads: List[str]
    splits: Dict[str, np.ndarray]  # protocol → [T, N] int8: 0 train, 1 val, 2 test, -1 unused
    norm: Dict[str, np.ndarray]    # {'mean','std'} used, kept for inference

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.X.shape


# ----------------------------------------------------------------- dense build

def _dense(df: pd.DataFrame, t_idx: np.ndarray, n_idx: np.ndarray,
           cols: Sequence[str], T: int, N: int, fill=np.nan) -> np.ndarray:
    out = np.full((T, N, len(cols)), fill, dtype=np.float32)
    out[t_idx, n_idx, :] = df[list(cols)].to_numpy(dtype=np.float32, na_value=np.nan)
    return out


def _forward_max(a: np.ndarray, h: int) -> np.ndarray:
    """max over t+1 … t+h along axis 0, NaN-propagating at the ragged tail."""
    T = a.shape[0]
    stack = np.full((h,) + a.shape, np.nan, dtype=np.float32)
    for k in range(1, h + 1):
        stack[k - 1, : T - k] = a[k:]
    with warnings.catch_warnings():          # all-NaN at the ragged tail is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(stack, axis=0)


def _shift_back(a: np.ndarray, k: int) -> np.ndarray:
    """value at t+k."""
    out = np.full_like(a, np.nan)
    out[: a.shape[0] - k] = a[k:]
    return out


def build_panel(
    df: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    truncate_after: Optional[str] = "2024-12-31",
    features: Sequence[str] = tuple(DYNAMIC_FEATURES),
) -> Panel:
    if truncate_after:
        df = df[df["date"] <= pd.Timestamp(truncate_after)]

    feats = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"[data] WARNING {len(missing)} dynamic features absent: {missing}")

    node_ids = sorted(df["node_id"].unique())
    dates = np.sort(df["date"].unique())
    N, T = len(node_ids), len(dates)
    nmap = {n: i for i, n in enumerate(node_ids)}
    dmap = {d: i for i, d in enumerate(dates)}

    df = df.sort_values(["date", "node_id"], kind="stable")
    t_idx = df["date"].map(dmap).to_numpy(np.int64)
    n_idx = df["node_id"].map(nmap).to_numpy(np.int64)

    # ---- dynamic features -------------------------------------------------
    X = _dense(df, t_idx, n_idx, feats, T, N)
    log_cols = [i for i, f in enumerate(feats) if f in LOG1P_FEATURES]
    if log_cols:
        sub = X[:, :, log_cols]
        X[:, :, log_cols] = np.sign(sub) * np.log1p(np.abs(sub))

    # ---- classification targets ------------------------------------------
    Y = _dense(df, t_idx, n_idx, [h for h in CLS_HEADS if h in df.columns], T, N)
    cls_heads = [h for h in CLS_HEADS if h in df.columns]

    # ---- derived regression targets (causal, within-node) -----------------
    pctl = _dense(df, t_idx, n_idx, ["discharge_pctl"], T, N)[:, :, 0]
    zsc = _dense(df, t_idx, n_idx, ["discharge_zscore"], T, N)[:, :, 0]
    R = np.stack([_shift_back(pctl, 1), _forward_max(zsc, 3)], axis=-1).astype(np.float32)

    # ---- masks and weights ------------------------------------------------
    valid = _dense(df, t_idx, n_idx, ["valid_sample"], T, N)[:, :, 0]
    valid = np.nan_to_num(valid, nan=0.0)
    valid = valid * (~np.isnan(Y[:, :, 0])).astype(np.float32)   # primary target present
    conf = np.nan_to_num(_dense(df, t_idx, n_idx, ["label_confidence"], T, N)[:, :, 0],
                         nan=1.0)
    flood_state = np.nan_to_num(
        _dense(df, t_idx, n_idx, ["flood_state"], T, N)[:, :, 0], nan=0.0)

    # ---- event codes ------------------------------------------------------
    ev = np.full((T, N), -1, dtype=np.int32)
    if "event_id" in df.columns:
        codes, _ = pd.factorize(df["event_id"], use_na_sentinel=True)
        ev[t_idx, n_idx] = codes.astype(np.int32)

    # ---- static terrain ---------------------------------------------------
    nd = nodes.set_index("node_id").reindex(node_ids)
    cont = nd[[c for c in STATIC_CONTINUOUS if c in nd.columns]].to_numpy(np.float32)
    if "log_drainage_proxy" not in nd.columns:
        # derive from the panel if the nodes table predates the column
        lp = df.groupby("node_id")["log_drainage_proxy"].first().reindex(node_ids)
        cont = np.column_stack([cont, lp.to_numpy(np.float32)])
    zone = np.stack([(nd["zone"] == z).to_numpy(np.float32) for z in ZONES], axis=1)
    pos = np.stack([(nd["position"] == p).to_numpy(np.float32) for p in POSITIONS], axis=1)
    S = np.concatenate([np.nan_to_num(cont), zone, pos], axis=1).astype(np.float32)

    # ---- split protocols --------------------------------------------------
    splits = _build_splits(df, t_idx, n_idx, T, N, dates)

    # ---- normalisation: TRAIN MASK ONLY -----------------------------------
    train_mask = (splits["temporal"] == 0) & (valid > 0)
    mean, std = _fit_norm(X, train_mask)
    X = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    s_mean = S[:, :cont.shape[1]].mean(0, keepdims=True)
    s_std = S[:, :cont.shape[1]].std(0, keepdims=True) + 1e-6
    S[:, :cont.shape[1]] = (S[:, :cont.shape[1]] - s_mean) / s_std

    return Panel(
        X=X, S=S, Y=np.nan_to_num(Y, nan=0.0), R=np.nan_to_num(R, nan=0.0),
        valid=valid.astype(np.float32), conf=conf.astype(np.float32), event=ev,
        flood_state=flood_state,
        dates=dates.astype("datetime64[D]"), node_ids=node_ids,
        basins=nd["basin"].tolist(), features=feats,
        cls_heads=cls_heads, reg_heads=list(REG_HEADS),
        splits=splits, norm={"mean": mean, "std": std},
    )


def _fit_norm(X: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sel = X[train_mask]                       # [n_train_rows, F]
    mean = np.nanmean(sel, axis=0, keepdims=True)
    std = np.nanstd(sel, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32)[None], std.astype(np.float32)[None]


def _build_splits(df, t_idx, n_idx, T, N, dates) -> Dict[str, np.ndarray]:
    code = {"train": 0, "val": 1, "test": 2}
    out: Dict[str, np.ndarray] = {}

    tmp = np.full((T, N), -1, dtype=np.int8)
    tmp[t_idx, n_idx] = df["split_temporal"].map(code).fillna(-1).to_numpy(np.int8)
    out["temporal"] = tmp

    if "split_basin_holdout" in df.columns:
        b = np.full((T, N), -1, dtype=np.int8)
        b[t_idx, n_idx] = df["split_basin_holdout"].map(code).fillna(-1).to_numpy(np.int8)
        # No val column ships with the basin protocol: carve 2018–2020 out of train.
        years = np.asarray(dates).astype("datetime64[Y]").astype(int) + 1970
        val_days = ((years >= 2018) & (years <= 2020))[:, None]
        b[val_days & (b == 0)] = 1
        out["basin"] = b

    # Diagnostic-only random split (RQ2). Seeded so the number is reproducible.
    rng = np.random.default_rng(1234)
    r = rng.random((T, N))
    rand = np.where(r < 0.70, 0, np.where(r < 0.85, 1, 2)).astype(np.int8)
    rand[out["temporal"] == -1] = -1
    out["random"] = rand
    return out


# ------------------------------------------------------------------- batching

def event_folds(panel: Panel, n_splits: int = 5, seed: int = 0) -> List[np.ndarray]:
    """GroupKFold over flood episodes (§7.5, protocol 3).

    Every node-day belongs to a group: its `event_id` when inside an episode,
    otherwise its calendar month. Background days are grouped by month rather
    than left individual, so serial autocorrelation cannot leak either.
    """
    T, N = panel.valid.shape
    months = (panel.dates.astype("datetime64[M]").astype(int))[:, None].repeat(N, 1)
    group = np.where(panel.event >= 0, panel.event, -(months + 1000)).astype(np.int64)
    uniq = np.unique(group[panel.valid > 0])
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_splits)
    return [np.isin(group, c) & (panel.valid > 0) for c in chunks]


class SnapshotBatcher:
    """Yields batches of full-graph day snapshots.

    A day t is usable if t >= lookback-1 and at least one node is both valid and
    assigned to the requested split. Nodes outside the split are masked out of
    the loss rather than dropped, so the graph stays whole — which is the point
    of the basin-holdout protocol.
    """

    def __init__(self, panel: Panel, protocol: Protocol, split: str,
                 lookback: int = 14, batch_size: int = 32, shuffle: bool = False,
                 mask_override: Optional[np.ndarray] = None,
                 frame_map: Optional[Dict[str, np.ndarray]] = None):
        self.p = panel
        self.lookback = lookback
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frame_map = frame_map
        code = {"train": 0, "val": 1, "test": 2}[split]
        sel = panel.splits[protocol] == code if mask_override is None else mask_override
        self.node_mask = (sel & (panel.valid > 0)).astype(np.float32)   # [T, N]
        usable = self.node_mask.sum(1) > 0
        usable[: lookback - 1] = False
        self.days = np.flatnonzero(usable)

    def __len__(self) -> int:
        return int(np.ceil(len(self.days) / self.batch_size))

    @property
    def n_samples(self) -> int:
        return int(self.node_mask[self.days].sum())

    @property
    def pos_rate(self) -> float:
        m = self.node_mask[self.days]
        return float((self.p.Y[self.days, :, 0] * m).sum() / max(m.sum(), 1))

    def __iter__(self):
        days = self.days.copy()
        if self.shuffle:
            np.random.shuffle(days)
        L = self.lookback
        for i in range(0, len(days), self.batch_size):
            d = days[i: i + self.batch_size]
            win = d[:, None] - np.arange(L - 1, -1, -1)[None, :]     # [B, L]
            batch = {
                "x": self.p.X[win],                                  # [B, L, N, F]
                "y": self.p.Y[d],                                    # [B, N, H]
                "r": self.p.R[d],                                    # [B, N, 2]
                "mask": self.node_mask[d],                           # [B, N]
                "conf": self.p.conf[d],                              # [B, N]
                "event": self.p.event[d],                            # [B, N]
                "days": d,
            }
            if self.frame_map is not None:
                batch["frame"] = self.frame_map["frame"][d]          # [B, N] int32
                batch["age"] = self.frame_map["age"][d]              # [B, N] float32
            yield batch


def save_norm(panel: Panel, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"features": panel.features,
                   "mean": panel.norm["mean"].ravel().tolist(),
                   "std": panel.norm["std"].ravel().tolist()}, f, indent=2)
