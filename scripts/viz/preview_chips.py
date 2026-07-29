"""Render grids of Sentinel-1 SAR chips to viewable PNGs."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
DOCS = os.path.join(ROOT, "docs")

idx = pd.read_csv(os.path.join(PROC, "image_index.csv"))


def load(r):
    return np.load(os.path.join(ROOT, r["path"])).astype(np.float32)  # [2,128,128]


def grid(df, title, fname, ncol=6):
    n = len(df)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.35 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, r) in zip(np.ravel(axes), df.iterrows()):
        vv = load(r)[0]
        ax.imshow(vv, cmap="gray", vmin=-25, vmax=0)
        tag = "FLOOD" if r["label"] == 1 else "dry"
        ax.set_title(f"{r['node_id']} {tag}\n{r.get('severity','')} {r.get('purpose','')}",
                     fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(DOCS, fname)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("wrote", out)


def falsecolor(df, title, fname, ncol=6):
    """VV / VH / VV-VH ratio -> RGB so water & flooded veg stand out."""
    n = len(df)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.35 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")

    def norm(a, lo, hi):
        return np.clip((a - lo) / (hi - lo), 0, 1)

    for ax, (_, r) in zip(np.ravel(axes), df.iterrows()):
        c = load(r)
        vv, vh = c[0], c[1]
        rgb = np.dstack([norm(vv, -25, 0), norm(vh, -30, -5),
                         norm(vv - vh, 0, 15)])
        ax.imshow(rgb)
        tag = "FLOOD" if r["label"] == 1 else "dry"
        ax.set_title(f"{r['node_id']} {tag}\n{r.get('severity','')}", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(DOCS, fname)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("wrote", out)


# 1) big grid: 24 flood + 24 dry
flood = idx[idx["label"] == 1].sample(24, random_state=1)
dry = idx[idx["label"] == 0].sample(24, random_state=1)
grid(flood, "48 FLOOD chips (Sentinel-1 VV, dB) — dark = water/wet",
     "chips_flood.png")
grid(dry, "48 DRY chips (Sentinel-1 VV, dB) — river confined to channel",
     "chips_dry.png")

# 2) false-color mix
mix = pd.concat([idx[idx["label"] == 1].sample(18, random_state=3),
                 idx[idx["label"] == 0].sample(18, random_state=3)])
falsecolor(mix, "False-color (VV / VH / VV-VH) — blue-ish = smooth water, "
           "green = flooded vegetation", "chips_falsecolor.png")
