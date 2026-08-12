"""Model 1 (TF-STGNN) entry point.

    python -m model1.train --preset M3
    python -m model1.train --preset M5 --protocol temporal --out runs
    python -m model1.train --preset M3 --protocol random   # RQ2 diagnostic only

The training loop itself is `floodlib.engine`, shared with model 2. This file
supplies only the two things that are specific to this family: its preset table
and how to build its network.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

from floodlib import engine
from floodlib.schema import BaseModelConfig

from .config import FAMILY, PRESETS, ModelConfig
from .model import TFSTGNN


def build_model(mcfg: BaseModelConfig, graph) -> TFSTGNN:
    assert isinstance(mcfg, ModelConfig), "model1 requires a model1 ModelConfig"
    return TFSTGNN(mcfg, graph)


def run(preset: str = "M3", protocol: Optional[str] = None, root: Optional[str] = None,
        out_dir: Optional[str] = None, epochs: Optional[int] = None,
        n_seeds: Optional[int] = None, device_spec: str = "auto",
        max_far: Optional[float] = None, verbose: bool = True,
        sar_root: Optional[str] = None, image_px: Optional[int] = None,
        batch_size: Optional[int] = None, tag: Optional[str] = None,
        train_overrides: Optional[Dict] = None, **model_overrides) -> Dict:
    return engine.run(
        preset, PRESETS, build_model, family=FAMILY, protocol=protocol, root=root,
        out_dir=out_dir, epochs=epochs, n_seeds=n_seeds, device_spec=device_spec,
        max_far=max_far, verbose=verbose, sar_root=sar_root, image_px=image_px,
        batch_size=batch_size, tag=tag, train_overrides=train_overrides,
        **model_overrides)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the TF-STGNN flood model.")
    ap.add_argument("--preset", default="M3", choices=list(PRESETS))
    engine.add_common_args(ap)
    a = ap.parse_args()
    tover, mover = engine.overrides_from_args(a)
    run(a.preset, a.protocol, a.root, a.out, a.epochs, a.seeds, a.device,
        a.max_far, not a.quiet, a.sar_root, a.image_px, a.batch_size, a.tag,
        train_overrides=tover, **mover)


if __name__ == "__main__":
    main()
