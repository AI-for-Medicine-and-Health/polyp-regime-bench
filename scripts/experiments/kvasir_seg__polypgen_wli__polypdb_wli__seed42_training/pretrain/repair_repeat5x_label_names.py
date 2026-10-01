#!/usr/bin/env python3
"""Repair repeat5x label basenames to match their paired materialized images.

The previous materializer derived image and label output stems independently.
This repair preserves the old hardlinks, adds correctly named hardlinks, and
updates repeat manifests to point at the image-matched label files.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42__repeat5x"
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    manifest_root = ROOT / "datasets/manifests/experiments" / VERSION
    data_root = ROOT / "datasets/materialized/active" / VERSION
    summary = {}
    for dataset in DATASETS:
        manifest_path = manifest_root / f"{dataset}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label_dir = data_root / dataset / "train/labels_detection"
        seen_targets: set[str] = set()
        repaired = 0
        for record in manifest["records"]:
            if record["split"] != "train":
                continue
            image_rel = Path(record["image"])
            old_label_rel = Path(record["label"])
            target_rel = image_rel.parent.parent / "labels_detection" / f"{image_rel.stem}.txt"
            target_rel_str = target_rel.as_posix()
            if target_rel_str in seen_targets:
                raise RuntimeError(f"duplicate image-derived label target: {target_rel_str}")
            seen_targets.add(target_rel_str)
            old_label = ROOT / old_label_rel
            target_label = ROOT / target_rel
            if not old_label.is_file():
                raise FileNotFoundError(old_label)
            if target_label.exists():
                if not target_label.is_file() or sha256(target_label) != sha256(old_label):
                    raise RuntimeError(f"conflicting existing label target: {target_label}")
            else:
                try:
                    os.link(old_label, target_label)
                except OSError:
                    shutil.copy2(old_label, target_label)
            if old_label_rel.as_posix() != target_rel_str:
                record["previous_unpaired_label"] = old_label_rel.as_posix()
                record["label"] = target_rel_str
                repaired += 1

        images = list((data_root / dataset / "train/images").iterdir())
        missing = [image.name for image in images if not (label_dir / f"{image.stem}.txt").is_file()]
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} train images still lack matching labels; examples={missing[:5]}")
        atomic_json(manifest_path, manifest)
        summary[dataset] = {"train_images": len(images), "paired_labels": len(images), "repaired_manifest_records": repaired}

    version_manifest_path = manifest_root / "version_manifest.json"
    version_manifest = json.loads(version_manifest_path.read_text(encoding="utf-8"))
    version_manifest["label_pairing_repair"] = {
        "fixed_at": datetime.now(timezone.utc).isoformat(),
        "rule": "train labels use the paired materialized image stem plus .txt",
        "validation": "every train image has a same-stem labels_detection file",
        "datasets": summary,
        "legacy_mismatched_label_files_preserved": True,
    }
    atomic_json(version_manifest_path, version_manifest)
    print(json.dumps({"status": "PASS", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
