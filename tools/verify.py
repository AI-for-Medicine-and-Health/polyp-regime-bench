#!/usr/bin/env python3
"""Verify a downloaded released checkpoint against the published SHA-256 index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    index = ROOT / "models/checkpoint_index.jsonl"
    if not index.is_file():
        raise FileNotFoundError(f"Download the model index first: {index}")
    matches = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    matches = [row for row in matches if all(row[key] == value for key, value in (
        ("condition", args.condition), ("seed", args.seed), ("dataset", args.dataset), ("model", args.model)))]
    if len(matches) != 1:
        raise SystemExit(f"Expected one indexed checkpoint; found {len(matches)}")
    row = matches[0]
    path = ROOT / "models" / row["path_in_repo"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
        raise SystemExit(f"Integrity check failed: {path}")
    print(f"PASS {path}")


if __name__ == "__main__":
    main()
