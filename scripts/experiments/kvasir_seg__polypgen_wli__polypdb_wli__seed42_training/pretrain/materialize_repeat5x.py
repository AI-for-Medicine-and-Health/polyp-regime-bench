#!/usr/bin/env python3
"""Materialize a 5x byte-identical repeat control from the seed-42 train split."""
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
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
VERSION = BASE + "__repeat5x"
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MANIFESTS = ROOT / "datasets/manifests/experiments"
MATERIALIZED = ROOT / "datasets/materialized/active"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    source_manifest_root = MANIFESTS / BASE
    source_data_root = MATERIALIZED / BASE
    output_manifest_root = MANIFESTS / VERSION
    output_data_root = MATERIALIZED / VERSION
    if output_manifest_root.exists() or output_data_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_manifest_root} or {output_data_root}")

    output_manifest_root.parent.mkdir(parents=True, exist_ok=True)
    output_data_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_stage = Path(tempfile.mkdtemp(prefix=f".{VERSION}.", dir=output_manifest_root.parent))
    data_stage = Path(tempfile.mkdtemp(prefix=f".{VERSION}.", dir=output_data_root.parent))
    methods: Counter[str] = Counter()
    version_counts: dict[str, dict[str, int]] = {}
    try:
        for dataset in DATASETS:
            source_json = json.loads((source_manifest_root / f"{dataset}.json").read_text(encoding="utf-8"))
            source_dataset_root = source_data_root / dataset
            target_dataset_root = data_stage / dataset
            target_dataset_root.mkdir(parents=True, exist_ok=True)
            records_out: list[dict] = []
            source_counts = Counter(record["split"] for record in source_json["records"])

            for split in ("train", "val", "test"):
                if split != "train":
                    (target_dataset_root / split).symlink_to(
                        os.path.relpath(source_dataset_root / split, target_dataset_root), target_is_directory=True
                    )

            for record in source_json["records"]:
                if record["split"] != "train":
                    unchanged = dict(record)
                    for key in ("image", "mask", "label"):
                        unchanged[key] = rel((ROOT / record[key]).resolve())
                    records_out.append(unchanged)
                    continue

                src_image = (ROOT / record["image"]).resolve()
                src_mask = (ROOT / record["mask"]).resolve()
                src_label = (ROOT / record["label"]).resolve()
                if not all(path.is_file() for path in (src_image, src_mask, src_label)):
                    raise FileNotFoundError(f"incomplete source pair/label for {record['external_id']}")
                if sha256(src_image) != record["image_sha256"] or sha256(src_mask) != record["mask_sha256"]:
                    raise ValueError(f"base manifest hash mismatch: {record['external_id']}")

                for repeat_index in range(5):
                    suffix = f"__rep{repeat_index:02d}"
                    image_name = f"{src_image.stem}{suffix}{src_image.suffix}"
                    mask_name = f"{src_mask.stem}{suffix}{src_mask.suffix}"
                    # Detection loaders locate a label by the image basename.
                    # Some source manifests carry dataset/center prefixes only
                    # on labels, so derive both output names from the image.
                    label_name = f"{src_image.stem}{suffix}.txt"
                    out_image = target_dataset_root / "train/images" / image_name
                    out_mask = target_dataset_root / "train/masks" / mask_name
                    out_label = target_dataset_root / "train/labels_detection" / label_name
                    methods[link_or_copy(src_image, out_image)] += 1
                    methods[link_or_copy(src_mask, out_mask)] += 1
                    methods[link_or_copy(src_label, out_label)] += 1
                    repeated = dict(record)
                    repeated.update({
                        "external_id": f"{record['external_id']}__rep{repeat_index:02d}",
                        "parent_external_id": record["external_id"],
                        "repeat_index": repeat_index,
                        "image": rel(output_data_root / dataset / "train/images" / image_name),
                        "mask": rel(output_data_root / dataset / "train/masks" / mask_name),
                        "label": rel(output_data_root / dataset / "train/labels_detection" / label_name),
                    })
                    records_out.append(repeated)

            target_dataset_root.mkdir(parents=True, exist_ok=True)
            yaml = (
                f"path: {(output_data_root / dataset).as_posix()}\n"
                "train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: polyp\n"
            )
            (target_dataset_root / "dataset.yaml").write_text(yaml, encoding="utf-8")
            counts = Counter(record["split"] for record in records_out)
            expected = {"train": source_counts["train"] * 5, "val": source_counts["val"], "test": source_counts["test"]}
            if dict(counts) != expected:
                raise AssertionError(f"split counts mismatch for {dataset}: got {dict(counts)}, expected {expected}")
            for split in ("train", "val", "test"):
                image_dir = target_dataset_root / split / "images"
                if len(list(image_dir.iterdir())) != expected[split]:
                    raise AssertionError(f"image count mismatch: {dataset}/{split}")
                label_dir = target_dataset_root / split / "labels_detection"
                missing = [image.name for image in image_dir.iterdir() if not (label_dir / f"{image.stem}.txt").is_file()]
                if missing:
                    raise AssertionError(f"unpaired image labels for {dataset}/{split}: {missing[:5]}")
            manifest = {
                **source_json,
                "version": VERSION,
                "source_version": BASE,
                "source_manifest_sha256": sha256(source_manifest_root / f"{dataset}.json"),
                "records": records_out,
                "counts": dict(counts),
                "repeat_control": {
                    "repeat_factor": 5,
                    "unique_training_images": source_counts["train"],
                    "byte_identical_repeats": True,
                    "augmentation_added": False,
                    "repeat_index_range": [0, 4],
                },
            }
            write_json(manifest_stage / f"{dataset}.json", manifest)
            version_counts[dataset] = dict(counts)

        base_version_manifest = json.loads((source_manifest_root / "version_manifest.json").read_text(encoding="utf-8"))
        version_manifest = {
            **base_version_manifest,
            "version": VERSION,
            "manifest_root": f"datasets/manifests/experiments/{VERSION}",
            "materialized_root": f"datasets/materialized/active/{VERSION}",
            "counts": version_counts,
            "repeat_control": {
                "repeat_factor": 5,
                "unique_training_images_per_dataset": {dataset: base_version_manifest["counts"][dataset]["train"] for dataset in DATASETS},
                "val_test": "symlinked unchanged from the base version",
                "augmentation_added": False,
                "file_strategy": dict(methods),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(manifest_stage / "version_manifest.json", version_manifest)
        (manifest_stage / "README.md").write_text(
            f"# {VERSION}\n\nMatched exposure control: each original train image, mask, and detection label is repeated five times byte-for-byte; no synthetic views are created. Validation and test splits are unchanged symlinks to `{BASE}`.\n",
            encoding="utf-8",
        )
        manifest_stage.rename(output_manifest_root)
        data_stage.rename(output_data_root)
    except Exception:
        shutil.rmtree(manifest_stage, ignore_errors=True)
        shutil.rmtree(data_stage, ignore_errors=True)
        raise

    print(json.dumps({"status": "PASS", "version": VERSION, "counts": version_counts, "file_strategy": dict(methods)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
