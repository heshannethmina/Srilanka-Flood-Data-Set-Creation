"""Export every Sentinel-1 SAR chip (.npy) to viewable PNG files.

This creates inspection copies only. The raw float16 [2,128,128] .npy chips
in data/images/ remain the real dataset.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
IMG_DIR = os.path.join(ROOT, "data", "images")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, os.environ.get("OUT", os.path.join("data", "png_images")))
SCALE = int(os.environ.get("SCALE", "2"))
MODE = os.environ.get("MODE", "all").lower()


def norm(a, lo, hi):
    x = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).round().astype(np.uint8)


def save(arr_u8, path):
    img = Image.fromarray(arr_u8)
    if SCALE != 1:
        img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    img.save(path)


def load_labels():
    meta = {}
    idx_path = os.path.join(PROC, "image_index.csv")
    if os.path.exists(idx_path):
        idx = pd.read_csv(idx_path)
        for _, row in idx.iterrows():
            sub = "flood" if int(row["label"]) == 1 else "dry"
            stem = f"{row['sample_id']}_{row['node_id']}_{row.get('severity', '')}".strip("_")
            meta[str(row["sample_id"])] = (sub, stem)
    return meta


def main():
    meta = load_labels()
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".npy"))
    if not files:
        print("no .npy chips found in", IMG_DIR)
        return

    for sub in ("flood", "dry", "unlabeled"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    n = 0
    for file_name in files:
        sid = file_name[:-4]
        chip = np.load(os.path.join(IMG_DIR, file_name)).astype(np.float32)
        vv, vh = chip[0], chip[1]

        sub, stem = meta.get(sid, ("unlabeled", sid))
        base = os.path.join(OUT, sub, stem)

        if MODE in ("all", "vv"):
            save(norm(vv, -25, 0), base + "_vv.png")
        if MODE == "all":
            save(norm(vh, -30, -5), base + "_vh.png")
        if MODE in ("all", "rgb"):
            rgb = np.dstack([
                norm(vv, -25, 0),
                norm(vh, -30, -5),
                norm(vv - vh, 0, 15),
            ])
            save(rgb, base + "_rgb.png")

        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(files)} chips exported...")

    print(f"done: {n} chips -> {OUT}")
    print("  flood/ , dry/ , unlabeled/ subfolders; _vv/_vh/_rgb PNGs per chip")


if __name__ == "__main__":
    main()