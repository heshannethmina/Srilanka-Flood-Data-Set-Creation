"""Render sample Sentinel-1 SAR chips to PNGs for quick visual inspection."""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROC = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "data", "preview")


def load_chip(row):
    return np.load(os.path.join(ROOT, row["path"])).astype(np.float32)


def render_grid(df, title, filename, ncol=6):
    n = len(df)
    nrow = max(1, (n + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.35 * nrow))
    axes = np.ravel(axes)
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, df.iterrows()):
        chip = load_chip(row)
        vv, vh = chip[0], chip[1]
        rgb = np.dstack([
            np.clip((vv + 25) / 25, 0, 1),
            np.clip((vh + 30) / 25, 0, 1),
            np.clip((vv - vh) / 15, 0, 1),
        ])
        ax.imshow(rgb)
        tag = "FLOOD" if int(row["label"]) == 1 else "dry"
        ax.set_title(f"{row['node_id']} {tag}\n{row.get('severity', '')} {row.get('purpose', '')}", fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def main():
    idx_path = os.path.join(PROC, "image_index.csv")
    if not os.path.exists(idx_path):
        print("image_index.csv not found. Run 07_fetch_images.py first.")
        return

    idx = pd.read_csv(idx_path)
    if idx.empty:
        print("image_index.csv is empty.")
        return

    flood = idx[idx["label"] == 1]
    dry = idx[idx["label"] == 0]

    if len(flood) == 0 and len(dry) == 0:
        print("No chips available to preview.")
        return

    if len(flood) > 0:
        render_grid(
            flood.sample(min(12, len(flood)), random_state=1),
            "Kalu Ganga flood chips (false color)",
            "chips_flood.png",
        )
    if len(dry) > 0:
        render_grid(
            dry.sample(min(12, len(dry)), random_state=1),
            "Kalu Ganga dry chips (false color)",
            "chips_dry.png",
        )


if __name__ == "__main__":
    main()