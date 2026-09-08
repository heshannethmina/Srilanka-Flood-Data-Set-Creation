"""Model 3 (STG-Former) entry point.

    python -m model3.train --preset P0          # the assembly check, run first
    python -m model3.train --preset P3
    python -m model3.train --preset P4_onset

Shares `floodlib.engine` with models 1 and 2, so all three families are trained,
thresholded, calibrated and scored by the same code — which is the only reason
their rows belong in one table.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

from floodlib import engine
from floodlib.schema import BaseModelConfig

from .config import FAMILY, PRESETS, STGFConfig
from .model import STGFormer


def build_model(mcfg: BaseModelConfig, graph=None) -> STGFormer:
    assert isinstance(mcfg, STGFConfig), "model3 requires a model3 STGFConfig"
    return STGFormer(mcfg, graph)


def run(preset: str = "P3", protocol: Optional[str] = None, root: Optional[str] = None,
        out_dir: Optional[str] = None, epochs: Optional[int] = None,
        n_seeds: Optional[int] = None, device_spec: str = "auto",
        max_far: Optional[float] = None, verbose: bool = True,
        sar_root: Optional[str] = None, image_px: Optional[int] = None,
        batch_size: Optional[int] = None, tag: Optional[str] = None,
        sar_pretrained: Optional[str] = None,
        train_overrides: Optional[Dict] = None, **model_overrides) -> Dict:
    return engine.run(
        preset, PRESETS, build_model, family=FAMILY, protocol=protocol, root=root,
        out_dir=out_dir, epochs=epochs, n_seeds=n_seeds, device_spec=device_spec,
        max_far=max_far, verbose=verbose, sar_root=sar_root, image_px=image_px,
        batch_size=batch_size, tag=tag, train_overrides=train_overrides,
        sar_pretrained=sar_pretrained, **model_overrides)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the STG-Former flood model.")
    ap.add_argument("--preset", default="P3", choices=list(PRESETS))
    ap.add_argument("--sar-pretrained", default=None,
                    help="encoder checkpoint from model2.pretrain_sar (P5_sar only)")
    engine.add_common_args(ap)
    a = ap.parse_args()
    tover, mover = engine.overrides_from_args(a)
    run(a.preset, a.protocol, a.root, a.out, a.epochs, a.seeds, a.device,
        a.max_far, not a.quiet, a.sar_root, a.image_px, a.batch_size, a.tag,
        a.sar_pretrained, train_overrides=tover, **mover)


if __name__ == "__main__":
    main()
