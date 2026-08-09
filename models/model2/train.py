"""Model 2 (MMF-Net) entry point.

    python -m model2.train --preset N5
    python -m model2.train --preset N6_gated --sar-pretrained runs/sar_encoder.pt

Shares `floodlib.engine` with model 1, so both families are trained, thresholded,
calibrated and scored by the same code.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

from floodlib import engine
from floodlib.schema import BaseModelConfig

from .config import FAMILY, PRESETS, MMFConfig
from .model import MMFNet


def build_model(mcfg: BaseModelConfig, graph=None) -> MMFNet:
    assert isinstance(mcfg, MMFConfig), "model2 requires a model2 MMFConfig"
    return MMFNet(mcfg)


def run(preset: str = "N5", protocol: Optional[str] = None, root: Optional[str] = None,
        out_dir: Optional[str] = None, epochs: Optional[int] = None,
        n_seeds: Optional[int] = None, device_spec: str = "auto",
        max_far: Optional[float] = None, verbose: bool = True,
        sar_root: Optional[str] = None, image_px: Optional[int] = None,
        batch_size: Optional[int] = None, tag: Optional[str] = None,
        sar_pretrained: Optional[str] = None) -> Dict:
    return engine.run(
        preset, PRESETS, build_model, family=FAMILY, protocol=protocol, root=root,
        out_dir=out_dir, epochs=epochs, n_seeds=n_seeds, device_spec=device_spec,
        max_far=max_far, verbose=verbose, sar_root=sar_root, image_px=image_px,
        batch_size=batch_size, tag=tag, sar_pretrained=sar_pretrained)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MMF-Net flood model.")
    ap.add_argument("--preset", default="N5", choices=list(PRESETS))
    ap.add_argument("--sar-pretrained", default=None,
                    help="encoder checkpoint from model2.pretrain_sar")
    engine.add_common_args(ap)
    a = ap.parse_args()
    run(a.preset, a.protocol, a.root, a.out, a.epochs, a.seeds, a.device,
        a.max_far, not a.quiet, a.sar_root, a.image_px, a.batch_size, a.tag,
        a.sar_pretrained)


if __name__ == "__main__":
    main()
