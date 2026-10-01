#!/usr/bin/env python3
"""Export and evaluate voting rules for the current seed-88 experiment.

The validation split is used to select every voting rule and its parameters.
The test split is read only after those choices are frozen.  This file is
specific to the current Kvasir/PolypGen-WLI/PolypDB-WLI experiment and never
references the historical CVC-inclusive experiments.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ROOT = SCRIPT_ROOT.parents[3]
BASE_DATA_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
CONDITION = os.environ.get("VOTING_CONDITION", "pretrained_base")
SEED = int(os.environ.get("VOTING_SEED", "88"))
DATA_VERSION = os.environ.get(
    "VOTING_DATA_VERSION",
    BASE_DATA_VERSION + ({"pretrained_aug3x": "__aug3x", "pretrained_aug5x": "__aug5x"}.get(CONDITION, "")),
)
DATA_ROOT = ROOT / "datasets/materialized/active" / DATA_VERSION
TRAIN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / CONDITION / f"seed{SEED}"
INFER_ROOT = ROOT / "runs/inference" / BASE_DATA_VERSION / os.environ.get(
    "VOTING_OUTPUT_NAME", f"{CONDITION}_seed{SEED}_voting"
)
CACHE_ROOT = Path(os.environ.get("VOTING_CACHE_ROOT", str(INFER_ROOT)))
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s",
    "rtdetr_l", "rtdetr_x",
)
WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "yolo11_s": "yolo11s.pt", "yolov5_s": "yolov5su.pt", "yolov8_s": "yolov8s.pt",
    "yolov9_s": "yolov9s.pt", "yolov3_tinyu": "yolov3-tinyu.pt",
    "yolov3_sppu": "yolov3-sppu.pt", "yolov10_s": "yolov10s.pt",
    "yolo12_s": "yolo12s.pt", "yolo26_s": "yolo26s.pt",
    "rtdetr_l": "rtdetr-l.pt", "rtdetr_x": "rtdetr-x.pt",
}
IMAGE_SIZE = 640
# Raw exports retain every candidate above the export floor.  Fusion analysis
# uses the highest-scoring candidates per model/image to keep the validation
# grid tractable on RT-DETR outputs (which can contain 150 candidates/image).
MAX_CANDIDATES_PER_MODEL_IMAGE = 50
IOU_THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(0.50, 0.951, 0.05))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(dataset: str, split: str) -> list[Path]:
    root = DATA_ROOT / dataset / split / "images"
    paths = sorted(path for path in root.iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"no images found: {root}")
    return paths


def checkpoint_path(dataset: str, model: str) -> Path:
    path = TRAIN_ROOT / dataset / model / ("best.pt" if model == "fasterrcnn_resnet50_fpn" else "weights/best.pt")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def normalize_box(box: list[float] | np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)), max(0.0, min(1.0, y2 / height)),
    ]


def write_predictions(output: Path, dataset: str, model: str, split: str, checkpoint: Path,
                      by_id: dict[str, list[dict[str, Any]]], score_floor: float, batch: int) -> None:
    detections = [
        {"image_id": image_id, "box": item["box"], "score": float(item["score"])}
        for image_id in sorted(by_id) for item in by_id[image_id]
    ]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATA_VERSION,
        "dataset": dataset, "model": model, "split": split, "seed": SEED,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "inference_batch": batch,
        "coordinate_format": "xyxy_normalized_to_image_width_height",
        "candidate_score_floor": score_floor,
        "image_count": len(by_id), "detection_count": len(detections),
        "predictions": detections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_ultralytics(dataset: str, model_name: str, split: str, checkpoint: Path,
                       device: str, batch: int, score_floor: float) -> int:
    import torch
    from ultralytics import RTDETR, YOLO

    paths = image_paths(dataset, split)
    by_id: dict[str, list[dict[str, Any]]] = {path.stem: [] for path in paths}
    model = RTDETR(str(checkpoint)) if model_name.startswith("rtdetr_") else YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in paths], imgsz=IMAGE_SIZE, batch=batch,
        conf=score_floor, device=device, stream=True, verbose=False, save=False,
    )
    for result in results:
        image_id = Path(result.path).stem
        height, width = result.orig_shape
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        by_id[image_id] = [
            {"box": normalize_box(box, int(width), int(height)), "score": float(score)}
            for box, score in zip(boxes, scores) if float(score) >= score_floor
        ]
    output = INFER_ROOT / "raw" / dataset / split / f"{model_name}.json"
    write_predictions(output, dataset, model_name, split, checkpoint, by_id, score_floor, batch)
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return sum(len(items) for items in by_id.values())


def build_fasterrcnn(device: str):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE,
        box_detections_per_img=300,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    torch_device = f"cuda:{device}" if str(device).isdigit() else device
    return model.to(torch_device)


def export_fasterrcnn(dataset: str, model_name: str, split: str, checkpoint: Path,
                      device: str, batch: int, score_floor: float) -> int:
    import torch

    torch_device = f"cuda:{device}" if str(device).isdigit() else device
    model = build_fasterrcnn(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    paths = image_paths(dataset, split)
    by_id: dict[str, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for start in range(0, len(paths), batch):
            batch_paths = paths[start:start + batch]
            tensors = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read image: {path}")
                resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
                tensors.append(tensor.to(torch_device, non_blocking=True))
            outputs = model(tensors)
            for path, result in zip(batch_paths, outputs):
                boxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                by_id[path.stem] = [
                    {"box": normalize_box(box, IMAGE_SIZE, IMAGE_SIZE), "score": float(score)}
                    for box, score in zip(boxes, scores) if float(score) >= score_floor
                ]
    output = INFER_ROOT / "raw" / dataset / split / f"{model_name}.json"
    write_predictions(output, dataset, model_name, split, checkpoint, by_id, score_floor, batch)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return sum(len(items) for items in by_id.values())


def export_all(device: str, batch: int, score_floor: float, force: bool, delay_seconds: float,
               model_names: tuple[str, ...] = MODELS) -> None:
    last_model = None
    for dataset in DATASETS:
        for model in model_names:
            checkpoint = checkpoint_path(dataset, model)
            needs_export = force or any(
                not (INFER_ROOT / "raw" / dataset / split / f"{model}.json").is_file()
                for split in ("val", "test")
            )
            if needs_export and last_model is not None and model != last_model and delay_seconds > 0:
                print(json.dumps({
                    "event": "model_switch_delay", "previous_model": last_model,
                    "next_model": model, "seconds": delay_seconds,
                }), flush=True)
                time.sleep(delay_seconds)
            if needs_export:
                last_model = model
            for split in ("val", "test"):
                output = INFER_ROOT / "raw" / dataset / split / f"{model}.json"
                if output.is_file() and not force:
                    try:
                        cached = json.loads(output.read_text(encoding="utf-8"))
                        valid_cache = (
                            cached.get("dataset_version") == DATA_VERSION
                            and cached.get("dataset") == dataset
                            and cached.get("model") == model
                            and cached.get("split") == split
                            and cached.get("seed") == SEED
                            and cached.get("checkpoint_sha256") == sha256(checkpoint)
                            and cached.get("inference_batch") == batch
                        )
                    except (OSError, json.JSONDecodeError):
                        valid_cache = False
                    if valid_cache:
                        print(json.dumps({"event": "skip_prediction", "dataset": dataset, "model": model, "split": split}), flush=True)
                        continue
                cached_path = CACHE_ROOT / "raw" / dataset / split / f"{model}.json"
                if cached_path.is_file() and not force:
                    try:
                        cached = json.loads(cached_path.read_text(encoding="utf-8"))
                        valid_cache = (
                            cached.get("dataset_version") == DATA_VERSION
                            and cached.get("dataset") == dataset
                            and cached.get("model") == model
                            and cached.get("split") == split
                            and cached.get("seed") == SEED
                            and cached.get("checkpoint_sha256") == sha256(checkpoint)
                            and cached.get("inference_batch") == batch
                        )
                    except (OSError, json.JSONDecodeError):
                        valid_cache = False
                    if valid_cache:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cached_path, output)
                        print(json.dumps({"event": "reuse_prediction_cache", "dataset": dataset, "model": model, "split": split}), flush=True)
                        continue
                print(json.dumps({"event": "export_prediction", "dataset": dataset, "model": model, "split": split}), flush=True)
                if model == "fasterrcnn_resnet50_fpn":
                    count = export_fasterrcnn(dataset, model, split, checkpoint, device, batch, score_floor)
                else:
                    count = export_ultralytics(dataset, model, split, checkpoint, device, batch, score_floor)
                print(json.dumps({"event": "prediction_complete", "dataset": dataset, "model": model, "split": split, "detections": count}), flush=True)


def load_ground_truth(dataset: str, split: str) -> dict[str, list[list[float]]]:
    output: dict[str, list[list[float]]] = {}
    for image in image_paths(dataset, split):
        label = DATA_ROOT / dataset / split / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            label = DATA_ROOT / dataset / split / "labels_detection" / f"{image.stem}.txt"
        boxes: list[list[float]] = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"invalid label: {label}")
                _, xc, yc, width, height = (float(value) for value in values)
                boxes.append([xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2])
        output[image.stem] = boxes
    return output


def load_predictions(dataset: str, split: str, model: str) -> dict[str, list[dict[str, Any]]]:
    path = INFER_ROOT / "raw" / dataset / split / f"{model}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {image.stem: [] for image in image_paths(dataset, split)}
    for item in payload["predictions"]:
        output[item["image_id"]].append({
            "box": [float(value) for value in item["box"]],
            "score": float(item["score"]), "model": model,
        })
    for image_id in output:
        output[image_id].sort(key=lambda item: item["score"], reverse=True)
        output[image_id] = output[image_id][:MAX_CANDIDATES_PER_MODEL_IMAGE]
    return output


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def match_flags(predictions: list[dict[str, Any]], ground_truth: list[list[float]], threshold: float) -> tuple[list[float], list[int]]:
    used = [False] * len(ground_truth)
    scores: list[float] = []
    flags: list[int] = []
    for item in sorted(predictions, key=lambda value: value["score"], reverse=True):
        scores.append(float(item["score"]))
        overlaps = [box_iou(item["box"], box) if not used[index] else -1.0 for index, box in enumerate(ground_truth)]
        if not overlaps:
            flags.append(0)
            continue
        index = int(np.argmax(overlaps))
        hit = overlaps[index] >= threshold
        flags.append(int(hit))
        if hit:
            used[index] = True
    return scores, flags


def average_precision(scores: list[float], flags: list[int], total_gt: int) -> float:
    if total_gt <= 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    true_positive = np.asarray(flags, dtype=float)[order]
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / total_gt
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive) + np.cumsum(false_positive), 1e-12)
    values = [float(np.max(precision[recall >= threshold])) if np.any(recall >= threshold) else 0.0 for threshold in np.linspace(0.0, 1.0, 101)]
    return float(np.mean(values))


def compute_metrics(predictions: dict[str, list[dict[str, Any]]], ground_truth: dict[str, list[list[float]]], operating_confidence: float) -> dict[str, float | int]:
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    ap_values = []
    for threshold in IOU_THRESHOLDS:
        scores: list[float] = []
        flags: list[int] = []
        for image_id in sorted(ground_truth):
            image_scores, image_flags = match_flags(predictions.get(image_id, []), ground_truth[image_id], threshold)
            scores.extend(image_scores)
            flags.extend(image_flags)
        ap_values.append(average_precision(scores, flags, total_gt))
    tp = fp = 0
    for image_id, gt in ground_truth.items():
        operating = [item for item in predictions.get(image_id, []) if item["score"] >= operating_confidence]
        _, flags = match_flags(operating, gt, 0.50)
        tp += sum(flags)
        fp += len(flags) - sum(flags)
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "mAP50": float(ap_values[0]), "mAP50_95": float(np.mean(ap_values)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "fp_per_image": fp / max(len(ground_truth), 1), "tp": tp, "fp": fp, "fn": fn,
    }


def cluster_predictions(model_predictions: dict[str, list[dict[str, Any]]], model_weights: dict[str, float],
                        cluster_iou: float, score_floor: float, min_models: int | None,
                        use_median: bool = False) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for model, items in model_predictions.items():
        for item in items:
            score = float(item["score"])
            if score < score_floor:
                continue
            best_index = None
            best_overlap = cluster_iou
            for index, cluster in enumerate(clusters):
                overlap = max(box_iou(item["box"], other) for other in cluster["boxes"])
                if overlap >= best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is None:
                clusters.append({"boxes": [item["box"]], "scores": [score], "models": [model]})
            else:
                cluster = clusters[best_index]
                cluster["boxes"].append(item["box"])
                cluster["scores"].append(score)
                cluster["models"].append(model)
    output = []
    for cluster in clusters:
        unique_models = sorted(set(cluster["models"]))
        if min_models is not None and len(unique_models) < min_models:
            continue
        boxes = np.asarray(cluster["boxes"], dtype=float)
        scores = np.asarray(cluster["scores"], dtype=float)
        weights = np.asarray([model_weights[model] * score for model, score in zip(cluster["models"], scores)], dtype=float)
        if use_median:
            box = np.median(boxes, axis=0)
        else:
            box = np.average(boxes, axis=0, weights=np.maximum(weights, 1e-12))
        score = float(np.average(scores, weights=np.asarray([model_weights[m] for m in cluster["models"]], dtype=float)))
        output.append({"box": box.tolist(), "score": score, "support": len(unique_models)})
    return sorted(output, key=lambda item: item["score"], reverse=True)


def fuse_dataset(predictions: dict[str, dict[str, list[dict[str, Any]]]], method: str,
                 params: dict[str, Any], model_weights: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    image_ids = sorted(next(iter(predictions.values())))
    if method == "single_best":
        model = params["model"]
        return {image_id: predictions[model][image_id] for image_id in image_ids}
    min_models = params.get("min_models")
    if method == "strict_consensus_vote":
        min_models = max(int(min_models), math.ceil(len(MODELS) / 2))
    return {
        image_id: cluster_predictions(
            {model: predictions[model][image_id] for model in MODELS}, model_weights,
            float(params["cluster_iou"]), float(params["score_floor"]), min_models,
            use_median=method == "majority_vote" or method == "strict_consensus_vote",
        )
        for image_id in image_ids
    }


def best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row["metrics"]["mAP50_95"]), float(row["metrics"]["recall"]), -float(row["metrics"]["fp_per_image"])))


def model_val_weights(dataset: str, val_predictions: dict[str, dict[str, list[dict[str, Any]]]], ground_truth: dict[str, list[list[float]]]) -> dict[str, float]:
    scores = {}
    for model in MODELS:
        scores[model] = compute_metrics(val_predictions[model], ground_truth, 0.30)["mAP50_95"]
    mean_score = max(float(np.mean(list(scores.values()))), 1e-6)
    return {model: max(score, 1e-6) / mean_score for model, score in scores.items()}


def analyze_dataset(dataset: str) -> dict[str, Any]:
    val_gt = load_ground_truth(dataset, "val")
    test_gt = load_ground_truth(dataset, "test")
    val = {model: load_predictions(dataset, "val", model) for model in MODELS}
    test = {model: load_predictions(dataset, "test", model) for model in MODELS}
    equal_weights = {model: 1.0 for model in MODELS}
    weighted = model_val_weights(dataset, val, val_gt)
    methods = {
        "equal_soft_vote": {"weights": equal_weights, "min_values": [None]},
        "val_weighted_soft_vote": {"weights": weighted, "min_values": [None]},
        "majority_vote": {"weights": equal_weights, "min_values": [2, 3, 4, 5, 6]},
        "strict_consensus_vote": {"weights": equal_weights, "min_values": [6, 7, 8, 9, 10, 11, 12]},
    }
    selections: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []

    single_rows = []
    for model in MODELS:
        metrics = compute_metrics(val[model], val_gt, 0.30)
        single_rows.append({"model": model, "metrics": metrics})
    selected_single = max(single_rows, key=lambda row: (float(row["metrics"]["mAP50_95"]), float(row["metrics"]["recall"])))
    single_params = {"model": selected_single["model"], "selection": "best validation mAP50-95"}
    single_test_metrics = compute_metrics(test[selected_single["model"]], test_gt, 0.30)
    result_rows.append({"method": "single_best", "params": single_params, "val_metrics": selected_single["metrics"], "test_metrics": single_test_metrics})
    selections["single_best"] = {"selected_model": selected_single["model"], "val_candidates": single_rows}

    for method, spec in methods.items():
        candidates = []
        fused_cache: dict[tuple[float, float, int | None], dict[str, list[dict[str, Any]]]] = {}
        for cluster_iou in (0.30, 0.50, 0.70):
            for score_floor in (0.01, 0.05, 0.10, 0.20, 0.30):
                for min_models in spec["min_values"]:
                    cache_key = (cluster_iou, score_floor, min_models)
                    params = {"cluster_iou": cluster_iou, "score_floor": score_floor, "operating_confidence": 0.30}
                    if min_models is not None:
                        params["min_models"] = min_models
                    fused = fused_cache.get(cache_key)
                    if fused is None:
                        fused = fuse_dataset(val, method, params, spec["weights"])
                        fused_cache[cache_key] = fused
                    for operating in (0.10, 0.20, 0.30, 0.40, 0.50):
                        params = {"cluster_iou": cluster_iou, "score_floor": score_floor, "operating_confidence": operating}
                        if min_models is not None:
                            params["min_models"] = min_models
                        metrics = compute_metrics(fused, val_gt, operating)
                        candidates.append({"params": params, "metrics": metrics})
        selected = best_candidate(candidates)
        frozen = dict(selected["params"])
        test_fused = fuse_dataset(test, method, frozen, spec["weights"])
        test_metrics = compute_metrics(test_fused, test_gt, float(frozen["operating_confidence"]))
        result_rows.append({"method": method, "params": frozen, "val_metrics": selected["metrics"], "test_metrics": test_metrics})
        selections[method] = {"selected": selected, "candidate_count": len(candidates), "model_weights": spec["weights"]}

    return {"dataset": dataset, "ground_truth_counts": {"val_images": len(val_gt), "val_boxes": sum(map(len, val_gt.values())), "test_images": len(test_gt), "test_boxes": sum(map(len, test_gt.values()))}, "results": result_rows, "selections": selections}


def write_analysis() -> None:
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATA_VERSION,
        "training_protocol": f"{CONDITION}/seed{SEED}, best validation checkpoint, 25 epochs",
        "selection_rule": "all voting parameters selected on validation mAP50-95; test evaluated only after selection",
        "datasets": {},
    }
    for dataset in DATASETS:
        print(json.dumps({"event": "analyze_dataset_started", "dataset": dataset}), flush=True)
    with ProcessPoolExecutor(
        max_workers=min(3, len(DATASETS)),
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        output["datasets"] = dict(zip(DATASETS, pool.map(analyze_dataset, DATASETS)))
    INFER_ROOT.mkdir(parents=True, exist_ok=True)
    (INFER_ROOT / "voting_analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for dataset, item in output["datasets"].items():
        for result in item["results"]:
            rows.append({
                "dataset": dataset, "method": result["method"],
                "params": json.dumps(result["params"], sort_keys=True),
                "val_mAP50_95": result["val_metrics"]["mAP50_95"], "val_mAP50": result["val_metrics"]["mAP50"],
                "val_precision": result["val_metrics"]["precision"], "val_recall": result["val_metrics"]["recall"],
                "test_mAP50_95": result["test_metrics"]["mAP50_95"], "test_mAP50": result["test_metrics"]["mAP50"],
                "test_precision": result["test_metrics"]["precision"], "test_recall": result["test_metrics"]["recall"],
                "test_f1": result["test_metrics"]["f1"], "test_fp_per_image": result["test_metrics"]["fp_per_image"],
            })
    with (INFER_ROOT / "voting_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--models", help="comma-separated subset of models to export; analysis always uses the full ensemble")
    args = parser.parse_args()
    if not args.export and not args.analyze:
        parser.error("choose --export and/or --analyze")
    if args.export:
        model_names = MODELS
        if args.models:
            model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
            unknown = sorted(set(model_names) - set(MODELS))
            if unknown:
                parser.error(f"unknown model(s): {', '.join(unknown)}")
            if not model_names:
                parser.error("--models must select at least one model")
        export_all(args.device, args.batch, args.score_floor, args.force, args.delay_seconds, model_names)
    if args.analyze:
        write_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
