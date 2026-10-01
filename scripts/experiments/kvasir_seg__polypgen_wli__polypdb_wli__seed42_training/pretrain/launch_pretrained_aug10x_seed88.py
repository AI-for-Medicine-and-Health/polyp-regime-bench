#!/usr/bin/env python3
"""Launch the 10x-augmentation seed-88 experiment.

The scheduling implementation is shared with the completed 3x experiment;
only the condition and output root are overridden here so the two runs cannot
overwrite one another.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parents[3]
SHARED_LAUNCHER = SCRIPT_ROOT / "launch_pretrained_aug3x_seed88.py"
spec = importlib.util.spec_from_file_location("launch_pretrained_aug10x_shared_scheduler", SHARED_LAUNCHER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load shared scheduler: {SHARED_LAUNCHER}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.CONDITION = "pretrained_aug10x"
module.RUN_ROOT = ROOT / "runs/training/kvasir_seg__polypgen_wli__polypdb_wli__seed42/pretrained_aug10x/seed88"

if __name__ == "__main__":
    raise SystemExit(module.main())
