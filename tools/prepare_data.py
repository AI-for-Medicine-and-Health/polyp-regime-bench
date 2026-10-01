#!/usr/bin/env python3
"""Check source dataset locations and rebuild the study's local data views."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
SCRIPT_DIR = ROOT / "scripts/experiments/three_dataset_wli_seed42"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--base", action="store_true")
    group.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    required = [
        ROOT / "datasets/raw/kvasir_seg/images",
        ROOT / "datasets/raw/kvasir_seg/masks",
        *[ROOT / f"datasets/raw/polypgen/data_C{n}" for n in range(1, 7)],
        ROOT / "datasets/raw/polypdb/PolypDB/PolypDB_center_wise",
    ]
    required += [ROOT / "datasets/manifests/splits/kvasir_seg.json",
                 ROOT / "datasets/manifests/splits/polypgen.json",
                 ROOT / "datasets/manifests/polypdb_centerwise_deduplicated.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing officially sourced input files or manifests:\n" + "\n".join(missing))
    if args.check:
        print("Input paths exist. Compare their contents with the dataset card before materializing.")
    elif args.base:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "materialize_splits.py")], cwd=ROOT, check=True)
    else:
        base = ROOT / "datasets/materialized/active" / VERSION
        if not base.exists():
            raise SystemExit("Build the base split first with --base")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "materialize_incremental_augmentations.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
