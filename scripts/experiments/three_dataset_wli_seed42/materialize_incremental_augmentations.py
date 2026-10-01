#!/usr/bin/env python3
"""Create nested 3x, 5x and 10x training views for the WLI seed-42 version.

The versions are intentionally nested:

* aug3x = original + aug01..aug02
* aug5x = aug3x + aug03..aug04
* aug10x = aug5x + aug05..aug09

Only train images/masks/labels are augmented.  Validation and test directories
are relative symlinks to the immutable base version, so their contents and
checksums remain exactly unchanged without wasting disk space.

This script is specific to the current three-dataset WLI-aligned version.  It
does not read or modify the historical CVC-inclusive augmentation directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover
    raise SystemExit("albumentations is required; activate the AI-Exp1 environment") from exc


ROOT = Path(__file__).resolve().parents[3]
BASE_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
MANIFEST_BASE = ROOT / "datasets/manifests/experiments" / BASE_VERSION
MATERIALIZED_BASE = ROOT / "datasets/materialized/active" / BASE_VERSION
SEED = 42
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
VERSION_STEPS = (
    ("aug3x", 3, 2, None),
    ("aug5x", 5, 4, "aug3x"),
    ("aug10x", 10, 9, "aug5x"),
)
JPEG_QUALITY = 95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def materialized_path(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def link_directory(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)


def reuse_file(source: Path, target: Path) -> str:
    """Reuse a file without copying bytes; fall back to a normal copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    resolved = source.resolve()
    try:
        os.link(resolved, target)
        return "hardlink"
    except OSError:
        shutil.copy2(resolved, target)
        return "copy"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise RuntimeError(f"failed to write image: {path}")


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), binary):
        raise RuntimeError(f"failed to write mask: {path}")


def boxes_from_mask(mask_path: Path) -> tuple[list[list[float]], int, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    height, width = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[list[float]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component])
        if area <= 0:
            continue
        boxes.append([
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ])
    return boxes, width, height


def write_yolo(path: Path, mask_path: Path) -> int:
    boxes, width, height = boxes_from_mask(mask_path)
    rows = [
        f"0 {xc:.8f} {yc:.8f} {box_width:.8f} {box_height:.8f}"
        for xc, yc, box_width, box_height in boxes
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    return len(boxes)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key not in {"matrix", "bbox_matrix"}}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def build_transform(item_seed: int) -> A.Compose:
    """Frozen image/mask transform family; no mosaic to keep one parent per pair."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-45, 45),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                    A.MedianBlur(blur_limit=5),
                ],
                p=0.25,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20),
                    A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8)),
                ],
                p=0.35,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.02, 0.08),
                hole_width_range=(0.02, 0.08),
                p=0.15,
            ),
        ],
        seed=item_seed,
        save_applied_params=True,
    )


def output_paths(dataset_root: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        dataset_root / "train/images" / f"{stem}.jpg",
        dataset_root / "train/masks" / f"{stem}.png",
        dataset_root / "train/labels_detection" / f"{stem}.txt",
    )


def make_original_record(
    dataset: str,
    base_record: dict[str, Any],
    dataset_root: Path,
    index: int,
) -> dict[str, Any]:
    stem = f"{index:06d}__orig__{safe_id(base_record['external_id'])}"
    image_target, mask_target, label_target = output_paths(dataset_root, stem)
    image_source = materialized_path(base_record["image"])
    mask_source = materialized_path(base_record["mask"])
    reuse_file(image_source, image_target)
    reuse_file(mask_source, mask_target)
    box_count = write_yolo(label_target, mask_target)
    image = cv2.imread(str(image_target), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_target), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"invalid original pair: {image_target} / {mask_target}")
    return {
        "output_image": f"train/images/{image_target.name}",
        "output_mask": f"train/masks/{mask_target.name}",
        "output_label": f"train/labels_detection/{label_target.name}",
        "source_id": base_record["external_id"],
        "source_image": base_record["source_image"],
        "source_mask": base_record["source_mask"],
        "source_image_sha256": base_record["image_sha256"],
        "source_mask_sha256": base_record["mask_sha256"],
        "variant": "original",
        "variant_index": 0,
        "seed": None,
        "parent_version": BASE_VERSION,
        "materialization": "hardlink_from_base_train",
        "applied_transforms": [],
        "input_box_count": int(base_record["box_count"]),
        "output_box_count": box_count,
        "output_image_sha256": sha256(image_target),
        "output_mask_sha256": sha256(mask_target),
        "output_label_sha256": sha256(label_target),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "labels_valid": True,
    }


def make_augmented_record(
    dataset: str,
    base_record: dict[str, Any],
    dataset_root: Path,
    index: int,
    variant_index: int,
) -> dict[str, Any]:
    source_image = materialized_path(base_record["image"])
    source_mask = materialized_path(base_record["mask"])
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"invalid source pair: {source_image} / {source_mask}")

    item_seed = SEED * 1_000_000 + (DATASETS.index(dataset) + 1) * 100_000 + index * 10 + variant_index
    transformed = build_transform(item_seed)(image=image, mask=mask)
    output_image = transformed["image"]
    output_mask = transformed["mask"]
    stem = f"{index:06d}__aug{variant_index:02d}__{safe_id(base_record['external_id'])}"
    image_target, mask_target, label_target = output_paths(dataset_root, stem)
    write_image(image_target, output_image)
    write_mask(mask_target, output_mask)
    output_box_count = write_yolo(label_target, mask_target)
    check_image = cv2.imread(str(image_target), cv2.IMREAD_COLOR)
    check_mask = cv2.imread(str(mask_target), cv2.IMREAD_GRAYSCALE)
    if check_image is None or check_mask is None or check_image.shape[:2] != check_mask.shape[:2]:
        raise RuntimeError(f"invalid generated pair: {image_target} / {mask_target}")
    return {
        "output_image": f"train/images/{image_target.name}",
        "output_mask": f"train/masks/{mask_target.name}",
        "output_label": f"train/labels_detection/{label_target.name}",
        "source_id": base_record["external_id"],
        "source_image": base_record["source_image"],
        "source_mask": base_record["source_mask"],
        "source_image_sha256": base_record["image_sha256"],
        "source_mask_sha256": base_record["mask_sha256"],
        "variant": f"aug{variant_index:02d}",
        "variant_index": variant_index,
        "seed": item_seed,
        "parent_version": BASE_VERSION,
        "materialization": "generated",
        "applied_transforms": json_safe(transformed.get("applied_transforms", [])),
        "input_box_count": int(base_record["box_count"]),
        "output_box_count": output_box_count,
        "output_image_sha256": sha256(image_target),
        "output_mask_sha256": sha256(mask_target),
        "output_label_sha256": sha256(label_target),
        "width": int(check_image.shape[1]),
        "height": int(check_image.shape[0]),
        "labels_valid": True,
    }


def clone_previous_records(
    dataset: str,
    previous_version: str,
    previous_records: list[dict[str, Any]],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    previous_root = ROOT / "datasets/materialized/active" / previous_version / dataset
    cloned: list[dict[str, Any]] = []
    for record in previous_records:
        for key in ("output_image", "output_mask", "output_label"):
            source = previous_root / record[key]
            target = dataset_root / record[key]
            reuse_file(source, target)
        cloned_record = dict(record)
        cloned_record["parent_version"] = previous_version
        cloned_record["materialization"] = "hardlink_from_parent_version"
        cloned_record["inherited"] = True
        cloned.append(cloned_record)
    return cloned


def load_base_records(dataset: str) -> list[dict[str, Any]]:
    payload = load_json(MANIFEST_BASE / f"{dataset}.json")
    return sorted(
        [record for record in payload["records"] if record["split"] == "train"],
        key=lambda record: record["external_id"],
    )


def materialize_version(tag: str, total_multiplier: int, variant_count: int, parent_tag: str | None) -> dict[str, Any]:
    version = f"{BASE_VERSION}__{tag}"
    manifest_root = ROOT / "datasets/manifests/experiments" / version
    materialized_root = ROOT / "datasets/materialized/active" / version
    if manifest_root.exists() or materialized_root.exists():
        raise FileExistsError(f"version already exists; refusing overwrite: {version}")
    manifest_root.mkdir(parents=True)
    materialized_root.mkdir(parents=True)

    summaries: dict[str, Any] = {}
    total_generated = 0
    total_inherited = 0
    for dataset in DATASETS:
        dataset_root = materialized_root / dataset
        dataset_root.mkdir(parents=True)
        base_records = load_base_records(dataset)
        source_by_id = {record["external_id"]: record for record in base_records}
        if len(source_by_id) != len(base_records):
            raise RuntimeError(f"duplicate base train external_id in {dataset}")

        if parent_tag is None:
            records = [make_original_record(dataset, record, dataset_root, index) for index, record in enumerate(base_records)]
            inherited_count = 0
            first_variant = 1
        else:
            parent_version = f"{BASE_VERSION}__{parent_tag}"
            previous_payload = load_json(
                ROOT / "datasets/manifests/experiments" / parent_version / f"{dataset}_augmentation.json"
            )
            records = clone_previous_records(dataset, parent_version, previous_payload["records"], dataset_root)
            inherited_count = len(records)
            first_variant = max(int(record["variant_index"]) for record in records) + 1
            if first_variant != (3 if tag == "aug5x" else 5):
                raise RuntimeError(f"unexpected nested variant boundary for {dataset}: {first_variant}")

        generated_records = []
        for variant_index in range(first_variant, variant_count + 1):
            for index, base_record in enumerate(base_records):
                generated_records.append(
                    make_augmented_record(dataset, base_record, dataset_root, index, variant_index)
                )
        records.extend(generated_records)

        for split in ("val", "test"):
            link_directory(MATERIALIZED_BASE / dataset / split, dataset_root / split)
        (dataset_root / "dataset.yaml").write_text(
            f"path: {dataset_root}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: polyp\n",
            encoding="utf-8",
        )

        counts = Counter(record["variant"] for record in records)
        payload = {
            "schema_version": 2,
            "version": version,
            "dataset": dataset,
            "seed": SEED,
            "base_version": BASE_VERSION,
            "parent_version": f"{BASE_VERSION}__{parent_tag}" if parent_tag else BASE_VERSION,
            "total_multiplier": total_multiplier,
            "variants_per_original": variant_count,
            "source_train_images": len(base_records),
            "total_train_images": len(records),
            "variant_counts": dict(sorted(counts.items())),
            "generated_in_this_version": len(generated_records),
            "inherited_from_parent": inherited_count,
            "val_augmented": False,
            "test_augmented": False,
            "evaluation_materialization": "relative symlinks to base version val/test directories",
            "transform_protocol": "WLI incremental paired image-mask transforms; no mosaic; mask interpolation=nearest",
            "records": records,
        }
        write_json(manifest_root / f"{dataset}_augmentation.json", payload)
        summaries[dataset] = {
            "source_train_images": len(base_records),
            "total_train_images": len(records),
            "generated_in_this_version": len(generated_records),
            "inherited_from_parent": inherited_count,
            "variant_counts": dict(sorted(counts.items())),
        }
        total_generated += len(generated_records)
        total_inherited += inherited_count

    version_manifest = {
        "version": version,
        "base_version": BASE_VERSION,
        "parent_version": f"{BASE_VERSION}__{parent_tag}" if parent_tag else BASE_VERSION,
        "seed": SEED,
        "total_multiplier": total_multiplier,
        "variants_per_original": variant_count,
        "new_variants_in_this_version": ([1, 2] if tag == "aug3x" else [3, 4] if tag == "aug5x" else [5, 6, 7, 8, 9]),
        "datasets": list(DATASETS),
        "manifest_root": str(manifest_root.relative_to(ROOT)),
        "materialized_root": str(materialized_root.relative_to(ROOT)),
        "split_source": str(MANIFEST_BASE.relative_to(ROOT)),
        "reuse_policy": "3x generated from base; 5x hardlinks all 3x files and adds aug03-aug04; 10x hardlinks all 5x files and adds aug05-aug09",
        "val_test_policy": "not augmented; split directories are relative symlinks to base version",
        "summaries": summaries,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_root / "version_manifest.json", version_manifest)
    write_json(manifest_root / "augmentation_manifest.json", {
        "version": version,
        "base_version": BASE_VERSION,
        "parent_version": version_manifest["parent_version"],
        "total_multiplier": total_multiplier,
        "new_variants_in_this_version": version_manifest["new_variants_in_this_version"],
        "generated_in_this_version": total_generated,
        "inherited_from_parent": total_inherited,
        "datasets": summaries,
    })
    (manifest_root / "README.md").write_text(
        (
            f"# {version}\n\n"
            f"Nested augmentation view for `{BASE_VERSION}`.\n\n"
            f"- Train: original samples plus variants `aug01` through `aug{variant_count:02d}` ({total_multiplier}x total).\n"
            f"- This step adds: {', '.join(f'aug{i:02d}' for i in version_manifest['new_variants_in_this_version'])}.\n"
            f"- Validation and test are not augmented; their directories are symlinks to the base version.\n"
            f"- Images and masks are transformed together; masks use nearest-neighbor interpolation.\n"
            f"- No mosaic is used, so every output has one auditable parent image.\n"
            f"- CVC-ClinicDB is excluded.\n"
        ),
        encoding="utf-8",
    )
    (materialized_root / "README.md").write_text(
        f"# {version}\n\nMaterialized nested training view. Train files are generated or hardlinked from the parent version; val/test are relative symlinks to `{BASE_VERSION}`.\n",
        encoding="utf-8",
    )
    return version_manifest


def main() -> int:
    if not MANIFEST_BASE.is_dir() or not MATERIALIZED_BASE.is_dir():
        raise SystemExit(f"base version is missing: {BASE_VERSION}")
    manifests = []
    for tag, multiplier, variants, parent in VERSION_STEPS:
        print(f"creating {BASE_VERSION}__{tag}", flush=True)
        manifests.append(materialize_version(tag, multiplier, variants, parent))
    print(json.dumps({"status": "PASS", "versions": [item["version"] for item in manifests]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
