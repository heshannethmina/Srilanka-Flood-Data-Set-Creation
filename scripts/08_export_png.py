"""
Export every Sentinel-1 SAR chip (.npy) to a viewable PNG folder.

This is a VIEWING copy only. The raw float16 [2,128,128] .npy chips in
data/images/ remain the real dataset the model trains on (see docs/IMAGE_DATASET.md).
PNGs are 8-bit and lossy-normalized — do NOT train on them.

For each chip we write three PNGs so both radar channels are visible:
  <id>_vv.png     VV backscatter, grayscale   (dark = smooth water / wet)
  <id>_vh.png     VH backscatter, grayscale
  <id>_rgb.png    false color (VV / VH / VV-VH) — blue-ish = water, green = flooded veg

Chips are sorted into flood/ and dry/ subfolders using image_index.csv labels.

Usage:
    python scripts/08_export_png.py
Env knobs:
    OUT=data/images_png     output folder
    SCALE=2                 upscale factor (nearest) for easier viewing; 1 = native 128px
    MODE=all                which PNGs: all | rgb | vv
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
IMG_DIR = os.path.join(ROOT, "data", "images")
PROC = os.path.join(ROOT, "data", "processed")

OUT = os.path.join(ROOT, os.environ.get("OUT", os.path.join("data", "images_png")))
SCALE = int(os.environ.get("SCALE", "2"))
MODE = os.environ.get("MODE", "all").lower()


def norm(a, lo, hi):
    """dB array -> 0..255 uint8, clipped."""
    x = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).round().astype(np.uint8)


def save(arr_u8, path):
    im = Image.fromarray(arr_u8)
    if SCALE != 1:
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
    im.save(path)


def load_labels():
    """sample_id -> (subfolder, filename-stem) using the index when available."""
    meta = {}
    fp = os.path.join(PROC, "image_index.csv")
    if os.path.exists(fp):
        idx = pd.read_csv(fp)
        for _, r in idx.iterrows():
            sub = "flood" if int(r["label"]) == 1 else "dry"
            stem = f"{r['sample_id']}_{r['node_id']}_{r.get('severity','')}".strip("_")
            meta[str(r["sample_id"])] = (sub, stem)
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
    for f in files:
        sid = f[:-4]
        chip = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)  # [2,128,128]
        vv, vh = chip[0], chip[1]

        sub, stem = meta.get(sid, ("unlabeled", sid))
        base = os.path.join(OUT, sub, stem)

        if MODE in ("all", "vv"):
            save(norm(vv, -25, 0), base + "_vv.png")
        if MODE == "all":
            save(norm(vh, -30, -5), base + "_vh.png")
        if MODE in ("all", "rgb"):
            rgb = np.dstack([norm(vv, -25, 0), norm(vh, -30, -5),
                             norm(vv - vh, 0, 15)])
            save(rgb, base + "_rgb.png")
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(files)} chips exported...")

    print(f"done: {n} chips -> {OUT}")
    print("  flood/ , dry/ , unlabeled/  subfolders; _vv/_vh/_rgb PNGs per chip")


if __name__ == "__main__":
    main()
