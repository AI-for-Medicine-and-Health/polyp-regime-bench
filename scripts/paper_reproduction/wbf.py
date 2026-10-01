#!/usr/bin/env python3
"""Replay the paper's frozen WBF selection and fuse exported box caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_CANDIDATES_PER_MODEL_IMAGE = 50

def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def get_choice(selection, condition, seed, dataset):
    try:
        return selection["choices"][condition][str(seed)][dataset]
    except KeyError as exc:
        raise SystemExit(f"No frozen WBF choice for {condition}/{seed}/{dataset}") from exc


def show_grid(choice):
    print("fusion_iou\tskip_box_thr\tval_mAP50:95\tselected")
    selected = choice["selected_parameters"]
    for row in choice["grid"]:
        p = (row["fusion_iou"], row["skip_box_thr"])
        mark = "*" if p == (selected["fusion_iou"], selected["skip_box_thr"]) else ""
        print(f"{p[0]:.2f}\t{p[1]:.3f}\t{row['metrics']['mAP50_95']:.6f}\t{mark}")


def load_ground_truth(root: Path, dataset: str):
    folder = root / "datasets/materialized/active/kvasir_seg__polypgen_wli__polypdb_wli__seed42" / dataset / "val"
    labels = folder / "labels"
    if not labels.exists():
        labels = folder / "labels_detection"
    output = {}
    for path in labels.glob("*.txt"):
        boxes = []
        for line in path.read_text().splitlines():
            cls, xc, yc, width, height = map(float, line.split())
            boxes.append([xc-width/2, yc-height/2, xc+width/2, yc+height/2])
        output[path.stem] = boxes
    if not output:
        raise SystemExit(f"No validation labels found under {labels}; materialize the paper split first")
    return output


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def ap101(scores, flags, n_gt):
    if not scores or not n_gt:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    recall, precision = [], []
    for i in order:
        tp += flags[i]
        fp += 1 - flags[i]
        recall.append(tp / n_gt)
        precision.append(tp / max(tp + fp, 1))
    return sum(max((p for r, p in zip(recall, precision) if r >= threshold), default=0.0)
               for threshold in [i / 100 for i in range(101)]) / 101


def evaluate(predictions, ground_truth):
    aps = []
    n_gt = sum(map(len, ground_truth.values()))
    for threshold in [i / 100 for i in range(50, 96, 5)]:
        all_scores, all_flags = [], []
        for image_id, gt in ground_truth.items():
            used = set()
            rows = sorted(predictions.get(image_id, []), key=lambda x: x["score"], reverse=True)
            for pred in rows:
                overlaps = [box_iou(pred["box"], box) if j not in used else -1 for j, box in enumerate(gt)]
                idx = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
                hit = idx >= 0 and overlaps[idx] >= threshold
                all_scores.append(pred["score"])
                all_flags.append(int(hit))
                if hit:
                    used.add(idx)
        aps.append(ap101(all_scores, all_flags, n_gt))
    return sum(aps) / len(aps)


def tune(choice, root: Path, condition: str, seed: int, dataset: str):
    from ensemble_boxes import weighted_boxes_fusion

    root = root.resolve()
    grouped_models = []
    for model in choice["selected_models"]:
        cache_path = root / choice["validation_cache_provenance"][model]["path"]
        digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        expected = choice["validation_cache_provenance"][model]["sha256"]
        if digest != expected:
            raise SystemExit(f"Validation cache hash mismatch for {model}: {digest}")
        grouped = {}
        for pred in read_json(cache_path)["predictions"]:
            grouped.setdefault(pred["image_id"], []).append(pred)
        for image_id in grouped:
            grouped[image_id] = sorted(grouped[image_id], key=lambda p: p["score"], reverse=True)[:MAX_CANDIDATES_PER_MODEL_IMAGE]
        grouped_models.append(grouped)
    gt = load_ground_truth(root, dataset)
    rows = []
    for iou in (0.2, 0.5, 0.7):
        for floor in (0.001, 0.02):
            fused = {}
            for image_id in gt:
                boxes, scores, labels = [], [], []
                for model in grouped_models:
                    dets = model.get(image_id, [])
                    boxes.append([d["box"] for d in dets])
                    scores.append([d["score"] for d in dets])
                    labels.append([0] * len(dets))
                bx, sc, _ = weighted_boxes_fusion(boxes, scores, labels, weights=[1.0]*len(boxes),
                    iou_thr=iou, skip_box_thr=floor, conf_type="avg")
                fused[image_id] = [{"box": list(map(float, b)), "score": float(s)} for b, s in zip(bx, sc)]
            rows.append({"fusion_iou": iou, "skip_box_thr": floor,
                         "validation_mAP50_95": evaluate(fused, gt)})
    winner = max(rows, key=lambda x: x["validation_mAP50_95"])
    print(json.dumps({"condition": condition, "seed": seed, "dataset": dataset,
        "selected_models": choice["selected_models"], "grid": rows, "selected_parameters": {
            "fusion_iou": winner["fusion_iou"], "skip_box_thr": winner["skip_box_thr"]}}, indent=2))


def fuse(choice, root: Path, condition: str, seed: int, dataset: str, split: str, output: Path):
    from ensemble_boxes import weighted_boxes_fusion

    records = []
    root = root.resolve()
    for model in choice["selected_models"]:
        rel = Path(choice["validation_cache_provenance"][model]["path"])
        cache = root / rel
        if split == "test":
            cache = cache.parent.parent / "test" / cache.name
        if not cache.is_file():
            raise SystemExit(f"Missing prediction cache: {cache}")
        records.append((model, read_json(cache)))

    # Keep image ordering from the first model cache and fail on incomplete inputs.
    by_model = []
    image_ids = []
    seen = set()
    for (model, data), model_record in zip(records, choice["selected_models"]):
        provenance = choice["validation_cache_provenance"][model]
        if split == "val":
            digest = hashlib.sha256((root / Path(provenance["path"])).read_bytes()).hexdigest()
            if digest != provenance["sha256"]:
                raise SystemExit(f"Validation cache hash mismatch for {model}: {digest}")
        grouped = {}
        for pred in data["predictions"]:
            grouped.setdefault(pred["image_id"], []).append(pred)
            if pred["image_id"] not in seen:
                image_ids.append(pred["image_id"])
                seen.add(pred["image_id"])
        for image_id in grouped:
            grouped[image_id] = sorted(grouped[image_id], key=lambda p: p["score"], reverse=True)[:MAX_CANDIDATES_PER_MODEL_IMAGE]
        by_model.append(grouped)

    params = choice["selected_parameters"]
    predictions = []
    for image_id in image_ids:
        boxes_list, scores_list, labels_list = [], [], []
        for cache in by_model:
            detections = cache.get(image_id, [])
            boxes_list.append([p["box"] for p in detections])
            scores_list.append([p["score"] for p in detections])
            labels_list.append([0] * len(detections))
        boxes, scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[1.0] * len(records), iou_thr=params["fusion_iou"],
            skip_box_thr=params["skip_box_thr"], conf_type="avg",
        )
        predictions.append({"image_id": image_id, "predictions": [
            {"box": [float(x) for x in box], "score": float(score)}
            for box, score in zip(boxes, scores)
        ]})

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "condition": condition, "dataset": dataset, "split": split, "seed": seed,
        "method": "equal_weight_wbf_val_tuned", "selected_models": [m for m, _ in records],
        "parameters": params, "coordinate_format": "xyxy_normalized_to_image_width_height",
        "predictions": predictions,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(predictions)} images to {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True,
                    help="wbf_val_tuned_all_conditions_ap001/validation_selection.json")
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="Project root containing the runs/inference cache tree")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, choices=(88, 123, 666), required=True)
    ap.add_argument("--dataset", choices=("kvasir_seg", "polypgen_wli", "polypdb_wli"), required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--show-grid", action="store_true",
                    help="Print the six recorded validation trials and frozen winner")
    ap.add_argument("--tune", action="store_true",
                    help="Recompute the validation grid from raw caches and materialized YOLO labels")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    choice = get_choice(read_json(args.selection), args.condition, args.seed, args.dataset)
    if args.show_grid:
        show_grid(choice)
    if args.tune:
        tune(choice, args.root, args.condition, args.seed, args.dataset)
    if args.output:
        fuse(choice, args.root, args.condition, args.seed, args.dataset, args.split, args.output)
    elif not (args.show_grid or args.tune):
        ap.error("provide --show-grid, --tune, and/or --output")


if __name__ == "__main__":
    main()
