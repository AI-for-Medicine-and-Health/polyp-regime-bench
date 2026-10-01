#!/usr/bin/env python3
"""Launch pretrained_aug5x for seed123 and seed666 using the shared scheduler."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parents[3]
SHARED_LAUNCHER = SCRIPT_ROOT / "launch_seed123_seed666_base_aug3x.py"
spec = importlib.util.spec_from_file_location("launch_pretrained_aug5x_seed123_seed666_shared", SHARED_LAUNCHER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load shared scheduler: {SHARED_LAUNCHER}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.CONDITIONS = ("pretrained_aug5x",)
module.SEEDS = (123, 666)

if __name__ == "__main__":
    raise SystemExit(module.main())
