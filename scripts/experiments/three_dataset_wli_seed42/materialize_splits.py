#!/usr/bin/env python3
"""Create the three-dataset WLI-aligned seed-42 dataset version."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / VERSION
MATERIALIZED_ROOT = ROOT / "datasets/materialized/active" / VERSION
SEED = 42

SOURCE_MANIFESTS = {
    "kvasir_seg": ROOT / "datasets/manifests/splits/kvasir_seg.json",
    "polypgen_wli": ROOT / "datasets/manifests/splits/polypgen.json",
    "polypdb_wli": ROOT / "datasets/manifests/polypdb_centerwise_deduplicated.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_verify(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        if not target.is_symlink() or target.resolve() != source:
            raise RuntimeError(f"Existing target does not match source: {target}")
        return
    target.symlink_to(os.path.relpath(source, target.parent))


def source_path(dataset: str, relative_or_project_path: str) -> Path:
    if relative_or_project_path.startswith("datasets/"):
        return ROOT / relative_or_project_path
    if dataset == "kvasir_seg":
        return ROOT / "datasets/raw/kvasir_seg" / relative_or_project_path
    if dataset == "polypgen_wli":
        return ROOT / "datasets/raw/polypgen" / relative_or_project_path
    raise RuntimeError(f"Unsupported relative source path: {dataset} {relative_or_project_path}")


def mask_boxes(mask_path: Path) -> tuple[list[list[float]], int, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    height, width = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[list[float]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component].tolist()
        if area <= 0:
            continue
        boxes.append([
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ])
    return boxes, width, height


def parse_polypdb(record: dict[str, Any]) -> tuple[str, str, str]:
    parts = Path(record["image_path"]).parts
    marker = parts.index("PolypDB_center_wise")
    center, modality = parts[marker + 1], parts[marker + 2]
    return center, modality, Path(record["image_path"]).stem


def allocate_center_counts(size: int) -> dict[str, int]:
    """Return train/val/test counts with exact 80/10/10 allocation."""
    fractions = {"train": 0.80, "val": 0.10, "test": 0.10}
    counts = {split: int(size * fraction) for split, fraction in fractions.items()}
    for split in counts:
        if counts[split] == 0 and size >= 3:
            counts[split] = 1
    while sum(counts.values()) < size:
        split = max(fractions, key=lambda name: (size * fractions[name] - counts[name], fractions[name]))
        counts[split] += 1
    while sum(counts.values()) > size:
        candidates = [name for name in counts if counts[name] > 1]
        split = min(candidates, key=lambda name: (size * fractions[name] - counts[name], fractions[name]))
        counts[split] -= 1
    return counts


def split_polypdb(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        center, modality, _ = parse_polypdb(record)
        if modality != "WLI":
            raise RuntimeError("Non-WLI record reached PolypDB WLI split")
        groups[center].append(record)
    result: list[dict[str, Any]] = []
    allocation: dict[str, dict[str, int]] = {}
    for center_index, center in enumerate(sorted(groups)):
        items = list(groups[center])
        random.Random(SEED + center_index).shuffle(items)
        counts = allocate_center_counts(len(items))
        allocation[center] = counts
        cursor = 0
        for split in ("train", "val", "test"):
            for item in items[cursor:cursor + counts[split]]:
                result.append({**item, "split": split, "center": center, "modality": "WLI"})
            cursor += counts[split]
    return result, allocation


def load_dataset_records(dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = SOURCE_MANIFESTS[dataset]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "polypdb_wli":
        records = []
        for item in payload["records"]:
            center, modality, _ = parse_polypdb(item)
            if modality == "WLI":
                records.append(item)
        records, allocation = split_polypdb(records)
        return records, {"allocation": allocation, "filtered_from": len(payload["records"]), "filter_count": len(records)}

    records = []
    for split, items in payload["records"].items():
        for item in items:
            normalized = {**item, "split": split}
            if dataset == "polypgen_wli":
                normalized["modality"] = "WLI"
                normalized["wli_filter_method"] = "official_endocv_wli_training_subset; sequence data excluded"
            records.append(normalized)
    return records, {"allocation": None, "filtered_from": len(records), "filter_count": len(records)}


def materialize_dataset(dataset: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_root = MATERIALIZED_ROOT / dataset
    output_records: list[dict[str, Any]] = []
    image_hashes: dict[str, set[str]] = defaultdict(set)
    label_hashes: dict[str, set[str]] = defaultdict(set)
    for item in sorted(records, key=lambda row: (row["split"], row.get("image_id", row.get("sample_id", "")))):
        if dataset == "polypdb_wli":
            center, modality, stem = parse_polypdb(item)
            external_id = f"{center}__{modality}__{stem}"
            image_source = ROOT / item["image_path"]
            mask_source = ROOT / item["mask_path"]
            image_sha = item["image_sha256"]
            mask_sha = item["mask_sha256"]
        else:
            center = item.get("center")
            modality = "WLI" if dataset == "polypgen_wli" else "WLI-like"
            image_id = item.get("scoped_image_id", item["image_id"]).replace("/", "__")
            external_id = image_id
            image_source = source_path("kvasir_seg" if dataset == "kvasir_seg" else "polypgen_wli", item["image_path"])
            mask_source = source_path("kvasir_seg" if dataset == "kvasir_seg" else "polypgen_wli", item["mask_path"])
            image_sha = item["sha256"]
            mask_sha = sha256(mask_source)

        if not image_source.is_file() or not mask_source.is_file():
            raise FileNotFoundError(f"Missing pair: {image_source} / {mask_source}")
        with Image.open(image_source) as image:
            image_width, image_height = image.size
            image.verify()
        boxes, mask_width, mask_height = mask_boxes(mask_source)
        if (image_width, image_height) != (mask_width, mask_height):
            raise RuntimeError(f"Image/mask size mismatch: {image_source} vs {mask_source}")

        image_target = dataset_root / item["split"] / "images" / f"{external_id}{image_source.suffix.lower()}"
        mask_target = dataset_root / item["split"] / "masks" / f"{external_id}{mask_source.suffix.lower()}"
        label_target = dataset_root / item["split"] / "labels_detection" / f"{external_id}.txt"
        link_or_verify(image_source, image_target)
        link_or_verify(mask_source, mask_target)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_text = "".join(f"0 {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}\n" for xc, yc, width, height in boxes)
        if label_target.exists() and label_target.read_text(encoding="utf-8") != label_text:
            raise RuntimeError(f"Existing label differs: {label_target}")
        if not label_target.exists():
            label_target.write_text(label_text, encoding="utf-8")
        image_hashes[item["split"]].add(image_sha)
        label_hashes[item["split"]].add(sha256(label_target))
        output_records.append({
            "external_id": external_id,
            "dataset": dataset,
            "split": item["split"],
            "center": center,
            "modality": modality,
            "source_image": item["image_path"],
            "source_mask": item["mask_path"],
            "image": str(image_target.relative_to(ROOT)),
            "mask": str(mask_target.relative_to(ROOT)),
            "label": str(label_target.relative_to(ROOT)),
            "image_sha256": image_sha,
            "mask_sha256": mask_sha,
            "width": image_width,
            "height": image_height,
            "box_count": len(boxes),
            "group_id": item.get("group_id"),
        })

    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset.yaml").write_text(
        f"path: {dataset_root}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: polyp\n",
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        "records": output_records,
        "counts": Counter(item["split"] for item in output_records),
        "cross_split_image_hash_overlap": sorted(set.intersection(*(image_hashes[split] for split in ("train", "val", "test")))) if all(image_hashes[split] for split in ("train", "val", "test")) else [],
    }


def main() -> int:
    loaded: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
        dataset: load_dataset_records(dataset) for dataset in SOURCE_MANIFESTS
    }
    materialized: dict[str, dict[str, Any]] = {}
    for dataset, (records, info) in loaded.items():
        result = materialize_dataset(dataset, records)
        materialized[dataset] = {**result, "source_info": info}

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_counts: dict[str, dict[str, int]] = {}
    dataset_audits: dict[str, Any] = {}
    for dataset, result in materialized.items():
        records = result.pop("records")
        manifest_counts[dataset] = {key: int(value) for key, value in result["counts"].items()}
        payload = {
            "version": VERSION,
            "dataset": dataset,
            "seed": SEED,
            "records": records,
            "counts": manifest_counts[dataset],
            "source_manifest": str(SOURCE_MANIFESTS[dataset].relative_to(ROOT)),
            "source_manifest_sha256": sha256(SOURCE_MANIFESTS[dataset]),
            "filter": (
                {"type": "none", "reason": "all Kvasir-SEG records retained", "modality_label": "WLI-like"}
                if dataset == "kvasir_seg" else
                {"type": "official_subset", "reason": "official EndoCV WLI single-frame subset; sequence data excluded", "modality": "WLI"}
                if dataset == "polypgen_wli" else
                {"type": "field_filter", "field": "modality", "keep": ["WLI"], "source_count": result["source_info"]["filtered_from"]}
            ),
        }
        write_json(MANIFEST_ROOT / f"{dataset}.json", payload)
        center_split = Counter((str(record.get("center") or "NONE"), record["split"]) for record in records)
        dataset_audits[dataset] = {
            "counts": manifest_counts[dataset],
            "source_manifest": payload["source_manifest"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
            "center_split_counts": {"|".join(key): value for key, value in sorted(center_split.items())},
            "record_count": len(records),
        }
        if dataset == "polypgen_wli":
            write_json(MANIFEST_ROOT / "polypgen_wli_filtered_ids.json", {
                "filter_method": "official_endocv_wli_training_subset",
                "sequence_data_excluded": True,
                "wli_image_ids": [r["external_id"] for r in records],
            })
        if dataset == "polypdb_wli":
            write_json(MANIFEST_ROOT / "polypdb_wli_filtered_ids.json", {
                "filter_field": "modality",
                "kept_modalities": ["WLI"],
                "wli_image_ids": [r["external_id"] for r in records],
            })

    all_hashes = {
        dataset: {split: {r["image_sha256"] for r in json.loads((MANIFEST_ROOT / f"{dataset}.json").read_text())["records"] if r["split"] == split} for split in ("train", "val", "test")}
        for dataset in materialized
    }
    split_audit = {
        "version": VERSION,
        "seed": SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_audits,
        "cross_split_image_sha256_overlap": {
            dataset: {f"{a}|{b}": sorted(all_hashes[dataset][a] & all_hashes[dataset][b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
            for dataset in all_hashes
        },
        "polypgen_protocol": "official WLI single-frame subset; C1-C5 train/val inherited from seed-42 center split; C6 fixed test",
        "polypdb_protocol": "WLI-only; center-stratified 80/10/10 train/val/test; seed=42",
        "materialization": "relative symlinks to raw images/masks plus generated YOLO detection labels",
    }
    write_json(MANIFEST_ROOT / "split_audit.json", split_audit)
    write_json(MANIFEST_ROOT / "version_manifest.json", {
        "version": VERSION,
        "seed": SEED,
        "datasets": list(SOURCE_MANIFESTS),
        "manifest_root": str(MANIFEST_ROOT.relative_to(ROOT)),
        "materialized_root": str(MATERIALIZED_ROOT.relative_to(ROOT)),
        "counts": manifest_counts,
        "rules": {
            "kvasir_seg": "existing seed-42 split 700/200/100; all records retained",
            "polypgen_wli": "official WLI single-frame subset; C1-C5 train/val, C6 test",
            "polypdb_wli": "modality=WLI, center-stratified 80/10/10, seed=42",
        },
    })
    (MANIFEST_ROOT / "README.md").write_text(
        """# Three-dataset WLI-aligned seed-42 version

- Kvasir-SEG: all 1000 records, existing seed-42 700/200/100 split.
- PolypGen: official WLI single-frame subset; C1-C5 train/val and C6 fixed test.
- PolypDB: modality=WLI only; center-stratified 80/10/10 train/val/test with seed=42.
- CVC-ClinicDB is excluded.
- Materialized images and masks are symlinks; detection labels are generated from masks.
""",
        encoding="utf-8",
    )
    (MATERIALIZED_ROOT / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (MATERIALIZED_ROOT / "README.md").write_text(
        f"# {VERSION}\n\nThree-dataset WLI-aligned materialized view, generated with seed=42. Images and masks are relative symlinks to the project raw data; YOLO detection labels are generated from masks. CVC is excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "version": VERSION, "counts": manifest_counts, "manifest_root": str(MANIFEST_ROOT), "materialized_root": str(MATERIALIZED_ROOT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
