#!/usr/bin/env python3
"""Classify test-set errors of validation-selected aug5x single models for S3."""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

os.environ.setdefault("VOTING_CONDITION", "pretrained_aug5x")
os.environ.setdefault(
    "VOTING_DATA_VERSION", "kvasir_seg__polypgen_wli__polypdb_wli__seed42__aug5x"
)

import run_voting_analysis as voting


SEEDS = (88, 123, 666)
CONFIDENCE = 0.30
MATCH_IOU = 0.50
NEAR_MISS_IOU = 0.10
INFERENCE_ROOT = voting.ROOT / "runs/inference" / voting.BASE_DATA_VERSION
OUTPUT = voting.ROOT / "reports" / voting.BASE_DATA_VERSION / "figures/supplementary/S3_error_audit.json"
ERROR_TYPES = ("pure_miss", "near_miss", "duplicate", "other_fp")


def classify_image(ground_truth: list[list[float]], predictions: list[dict]) -> Counter:
    """Match at IoU .5, then pair each low-overlap FN/FP once as a near miss."""
    counts: Counter = Counter()
    matched_gt: set[int] = set()
    unmatched_predictions: list[tuple[int, dict]] = []
    operating = sorted((p for p in predictions if p["score"] >= CONFIDENCE),
                       key=lambda p: p["score"], reverse=True)
    for prediction_index, prediction in enumerate(operating):
        overlaps = [voting.box_iou(prediction["box"], box) if index not in matched_gt else -1.0
                    for index, box in enumerate(ground_truth)]
        if overlaps and max(overlaps) >= MATCH_IOU:
            matched_gt.add(max(range(len(overlaps)), key=lambda index: overlaps[index]))
            counts["tp"] += 1
        else:
            unmatched_predictions.append((prediction_index, prediction))

    unmatched_gt = set(range(len(ground_truth))) - matched_gt
    near_pairs = []
    for prediction_index, prediction in unmatched_predictions:
        for ground_truth_index in unmatched_gt:
            overlap = voting.box_iou(prediction["box"], ground_truth[ground_truth_index])
            if NEAR_MISS_IOU <= overlap < MATCH_IOU:
                near_pairs.append((overlap, prediction_index, ground_truth_index))
    paired_predictions: set[int] = set()
    paired_gt: set[int] = set()
    for _, prediction_index, ground_truth_index in sorted(near_pairs, reverse=True):
        if prediction_index in paired_predictions or ground_truth_index in paired_gt:
            continue
        paired_predictions.add(prediction_index)
        paired_gt.add(ground_truth_index)
        counts["near_miss"] += 1

    for prediction_index, prediction in unmatched_predictions:
        if prediction_index in paired_predictions:
            continue
        best_overlap = max((voting.box_iou(prediction["box"], box) for box in ground_truth), default=0.0)
        counts["duplicate" if best_overlap >= MATCH_IOU else "other_fp"] += 1
    counts["pure_miss"] += len(unmatched_gt - paired_gt)
    counts["fp"] = counts["near_miss"] + counts["duplicate"] + counts["other_fp"]
    counts["fn"] = counts["near_miss"] + counts["pure_miss"]
    return counts


def main() -> None:
    rows = []
    for seed in SEEDS:
        voting.SEED = seed
        voting.INFER_ROOT = INFERENCE_ROOT / f"consolidated_pretrained_aug5x_seed{seed}_voting"
        analysis = json.loads((voting.INFER_ROOT / "voting_analysis.json").read_text(encoding="utf-8"))
        for dataset in voting.DATASETS:
            dataset_analysis = analysis["datasets"][dataset]
            single = next(row for row in dataset_analysis["results"] if row["method"] == "single_best")
            model = single["params"]["model"]
            ground_truth = voting.load_ground_truth(dataset, "test")
            predictions = voting.load_predictions(dataset, "test", model)
            counts: Counter = Counter()
            images_by_type: dict[str, list[str]] = defaultdict(list)
            for image_id, boxes in ground_truth.items():
                image_counts = classify_image(boxes, predictions[image_id])
                counts.update(image_counts)
                for kind in ERROR_TYPES:
                    if image_counts[kind]:
                        images_by_type[kind].append(image_id)
            expected = single["test_metrics"]
            if any(counts[kind] != expected[kind] for kind in ("tp", "fp", "fn")):
                raise AssertionError(f"S3 counts differ from existing voting analysis: {seed} {dataset}")
            rows.append({
                "seed": seed, "dataset": dataset, "validation_selected_model": model,
                "test_images": len(ground_truth), "ground_truth_boxes": sum(map(len, ground_truth.values())),
                "counts": {kind: counts[kind] for kind in ("tp", "fp", "fn", *ERROR_TYPES)},
                "images_by_error_type": dict(images_by_type),
                "image_counts": {kind: len(images_by_type[kind]) for kind in ERROR_TYPES},
            })

    summaries = {}
    for dataset in voting.DATASETS:
        chosen = [row for row in rows if row["dataset"] == dataset]
        keys = ("tp", "fp", "fn", *ERROR_TYPES)
        summaries[dataset] = {
            "test_images_per_seed": chosen[0]["test_images"],
            "ground_truth_boxes_per_seed": chosen[0]["ground_truth_boxes"],
            "selected_models_by_seed": {str(row["seed"]): row["validation_selected_model"] for row in chosen},
            "mean_counts_per_seed": {kind: statistics.mean(row["counts"][kind] for row in chosen) for kind in keys},
            "sample_sd_across_seeds": {kind: statistics.stdev(row["counts"][kind] for row in chosen) for kind in keys},
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition": "pretrained_aug5x", "split": "test", "seeds": list(SEEDS),
        "selection": "best single model per dataset and seed selected on validation mAP50-95",
        "confidence_threshold": CONFIDENCE, "match_iou": MATCH_IOU,
        "near_miss_iou_range": [NEAR_MISS_IOU, MATCH_IOU],
        "definitions": {
            "pure_miss": "unmatched GT after one-to-one near-miss pairing",
            "near_miss": "one-to-one prediction/GT pair with IoU in [0.10, 0.50); counts as one FP and one FN under standard evaluation",
            "duplicate": "unmatched prediction with IoU >= 0.50 to a GT already matched by another prediction",
            "other_fp": "remaining unmatched prediction; its visual cause is not inferred automatically",
        },
        "note": "The same test images recur for three model seeds; pooled counts are not independent cases.",
        "per_seed": rows, "summary": summaries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
