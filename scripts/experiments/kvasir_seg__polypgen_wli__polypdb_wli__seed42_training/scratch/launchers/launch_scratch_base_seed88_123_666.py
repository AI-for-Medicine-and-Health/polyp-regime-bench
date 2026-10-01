#!/usr/bin/env python3
"""Queue Scratch base training after pretrained aug5x seed123/666 passes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[2]
ROOT = TRAINING_ROOT.parents[2]
SHARED_LAUNCHER = TRAINING_ROOT / "pretrain/launch_seed123_seed666_base_aug3x.py"
SCRATCH_TRAINER = TRAINING_ROOT / "scratch/common/train_one.py"
spec = importlib.util.spec_from_file_location("scratch_shared_scheduler", SHARED_LAUNCHER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load shared scheduler: {SHARED_LAUNCHER}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.TRAINER = SCRATCH_TRAINER
module.CONDITIONS = ("scratch_base",)
module.SEEDS = (88, 123, 666)


if __name__ == "__main__":
    raise SystemExit(module.main())
