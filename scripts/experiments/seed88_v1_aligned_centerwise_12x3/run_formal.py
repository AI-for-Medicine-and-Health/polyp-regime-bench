#!/usr/bin/env python3
"""Run the formal seed-88, 3-dataset, 12-model experiment.

Ultralytics models use the frozen YOLO views and the same no-online-
augmentation training settings as the validated pilot. Faster R-CNN uses a
Torchvision adapter with the same optimizer, learning rate, weight decay, seed,
epoch budget, validation-based checkpoint selection, and a documented batch
override. Runs are isolated and never overwrite an existing run directory.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[3]
DATA_VERSION = os.environ.get("GI_DATA_VERSION", "v1_aligned_centerwise_split42_aug42_10x")
RUN_GROUP = os.environ.get("GI_RUN_GROUP", "seed88_v1_aligned_centerwise_12x3")
EXPERIMENT_ID = os.environ.get("GI_EXPERIMENT_ID", "seed88_v1_aligned_centerwise_12x3")
AUGMENTATION_PROTOCOL = os.environ.get("GI_AUGMENTATION_PROTOCOL", "v1_10x")
AUGMENTATION_ENABLED = os.environ.get("GI_AUGMENTATION_ENABLED", "1") == "1"
DATASETS = ("kvasir_seg", "polypgen", "cvc_clinicdb")
MODELS = (
    "fasterrcnn_resnet50_fpn",
    "yolo11_s",
    "yolov5_s",
    "yolov8_s",
    "yolov9_s",
    "yolov3_tinyu",
    "yolov3_sppu",
    "yolov10_s",
    "yolo12_s",
    "yolo26_s",
    "rtdetr_l",
    "rtdetr_x",
)
WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "yolo11_s": "yolo11s.pt",
    "yolov5_s": "yolov5su.pt",
    "yolov8_s": "yolov8s.pt",
    "yolov9_s": "yolov9s.pt",
    "yolov3_tinyu": "yolov3-tinyu.pt",
    "yolov3_sppu": "yolov3-sppu.pt",
    "yolov10_s": "yolov10s.pt",
    "yolo12_s": "yolo12s.pt",
    "yolo26_s": "yolo26s.pt",
    "rtdetr_l": "rtdetr-l.pt",
    "rtdetr_x": "rtdetr-x.pt",
}
DATA_ROOT = ROOT / "datasets/materialized/active" / DATA_VERSION / "splits"
# Faster R-CNN reads its training images directly. For the no-augmentation
# baseline the training view is the regular split tree, not the augmented tree.
AUG_ROOT = (
    ROOT / "datasets/materialized/active" / DATA_VERSION / "augmented"
    if AUGMENTATION_ENABLED
    else DATA_ROOT
)
RUN_ROOT = ROOT / "runs/training" / RUN_GROUP
REPORT_ROOT = ROOT / "runs/trial" / RUN_GROUP
SUMMARY_PATH = REPORT_ROOT / "formal_results.json"
RUNTIME_ROOT = ROOT / "runs/trial" / RUN_GROUP / "runtime"
SUMMARY_LOCK_PATH = RUNTIME_ROOT / "formal_results.lock"
SEED = 88
IMAGE_SIZE = 640
EPOCHS = 30
LR0 = 1e-4
WEIGHT_DECAY = 5e-4
FRCNN_BATCH_OVERRIDE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_id(dataset: str, model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_{RUN_GROUP}_{dataset}_{model}"


def yaml_for(dataset: str) -> Path:
    path = DATA_ROOT / dataset / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_best_metrics(results_csv: Path) -> dict[str, float | int]:
    with results_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty training metrics: {results_csv}")
    map_key = next(key for key in rows[0] if "mAP50-95(B)" in key)
    map50_key = next(key for key in rows[0] if "mAP50(B)" in key and "95" not in key)
    best = max(rows, key=lambda row: float(row[map_key]))
    return {
        "best_epoch": int(float(best["epoch"])) + 1,
        "best_val_map50": float(best[map50_key]),
        "best_val_map50_95": float(best[map_key]),
    }


def metric_row(metrics: Any, prefix: str) -> dict[str, float]:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    return {
        f"{prefix}_map50": float(metrics.box.map50),
        f"{prefix}_map50_95": float(metrics.box.map),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def make_config(dataset: str, model: str, yaml_path: Path, device: str, epochs: int, batch: int) -> dict[str, Any]:
    import torch

    weight = ROOT / "assets/pretrained_weights" / WEIGHTS[model]
    return {
        "protocol": "seed88_v1_aligned_centerwise_12x3",
        "dataset": dataset,
        "model": model,
        "framework": "torchvision" if model == "fasterrcnn_resnet50_fpn" else "ultralytics",
        "seed": SEED,
        "split_seed": 42,
        "split_id": "v1_aligned_centerwise",
        "split_manifest": f"datasets/manifests/experiments/seed88_v1_aligned_centerwise_12x3/{dataset}.json",
        "dataset_version": DATA_VERSION,
        "augmentation_seed": 42,
        "augmentation_protocol": AUGMENTATION_PROTOCOL,
        "augmentation_enabled": AUGMENTATION_ENABLED,
        "data_yaml": str(yaml_path.resolve()),
        "weight": str(weight),
        "weight_sha256": sha256(weight),
        "epochs": epochs,
        "batch": batch,
        "imgsz": IMAGE_SIZE,
        "optimizer": "AdamW",
        "initial_lr": LR0,
        "weight_decay": WEIGHT_DECAY,
        "online_augmentation": False,
        "pretrained": True,
        "deterministic": True,
        "device": device,
        "physical_device": os.environ.get("GI_PHYSICAL_DEVICE", device),
        "workers": 4,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def snapshot_run(run_dir: Path, config: dict[str, Any], yaml_path: Path, command: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "dataset.yaml").write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        (ROOT / config["split_manifest"]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    augmentation_manifest = AUG_ROOT / config["dataset"] / "augmentation_manifest.json"
    if augmentation_manifest.is_file():
        augmentation_text = augmentation_manifest.read_text(encoding="utf-8")
    else:
        version_manifest = ROOT / "datasets/materialized/active" / DATA_VERSION / "version_manifest.json"
        if not version_manifest.is_file():
            raise FileNotFoundError(f"missing augmentation/version manifest: {augmentation_manifest}")
        augmentation_text = version_manifest.read_text(encoding="utf-8")
    (run_dir / "augmentation_manifest.json").write_text(augmentation_text, encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({**config, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def ap_from_matches(scores: list[float], true_positive: list[int], total_gt: int) -> float:
    if not scores or total_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(true_positive, dtype=np.float64)[order]
    fp = 1.0 - tp
    recall = np.cumsum(tp) / total_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall_grid = np.linspace(0.0, 1.0, 101)
    values = []
    for threshold in recall_grid:
        values.append(float(np.max(precision[recall >= threshold])) if np.any(recall >= threshold) else 0.0)
    return float(np.mean(values))


def detection_metrics(
    ground_truth: dict[str, np.ndarray], predictions: dict[str, list[tuple[float, list[float]]]]
) -> dict[str, float | int]:
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    thresholds = np.arange(0.50, 0.951, 0.05)
    aps = []
    for threshold in thresholds:
        scores: list[float] = []
        matches: list[int] = []
        used = {key: np.zeros(len(value), dtype=bool) for key, value in ground_truth.items()}
        for image_id, items in predictions.items():
            items = sorted(items, key=lambda item: item[0], reverse=True)
            gt = ground_truth.get(image_id, np.zeros((0, 4), dtype=np.float64))
            for score, box in items:
                scores.append(score)
                if not len(gt):
                    matches.append(0)
                    continue
                overlaps = iou_matrix(np.asarray([box]), gt)[0]
                index = int(np.argmax(overlaps))
                hit = overlaps[index] >= threshold and not used[image_id][index]
                matches.append(int(hit))
                if hit:
                    used[image_id][index] = True
        aps.append(ap_from_matches(scores, matches, total_gt))

    operating_scores: list[float] = []
    operating_matches: list[int] = []
    used = {key: np.zeros(len(value), dtype=bool) for key, value in ground_truth.items()}
    for image_id, items in predictions.items():
        gt = ground_truth.get(image_id, np.zeros((0, 4), dtype=np.float64))
        for score, box in sorted(items, key=lambda item: item[0], reverse=True):
            if score < 0.05:
                continue
            operating_scores.append(score)
            if not len(gt):
                operating_matches.append(0)
                continue
            overlaps = iou_matrix(np.asarray([box]), gt)[0]
            index = int(np.argmax(overlaps))
            hit = overlaps[index] >= 0.50 and not used[image_id][index]
            operating_matches.append(int(hit))
            if hit:
                used[image_id][index] = True
    tp = int(sum(operating_matches))
    fp = len(operating_matches) - tp
    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "map50": float(aps[0]),
        "map50_95": float(np.mean(aps)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class YoloDetectionDataset:
    def __init__(self, root: Path, split: str, size: int = IMAGE_SIZE):
        self.image_dir = root / split / "images"
        self.label_dir = root / split / "labels"
        self.paths = sorted(self.image_dir.iterdir())
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        height, width = image.shape[:2]
        resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = __import__("torch").from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        boxes = []
        label_path = self.label_dir / f"{path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"invalid YOLO label: {label_path}")
            _, xc, yc, bw, bh = (float(value) for value in values)
            boxes.append([
                (xc - bw / 2) * self.size,
                (yc - bh / 2) * self.size,
                (xc + bw / 2) * self.size,
                (yc + bh / 2) * self.size,
            ])
        return tensor, np.asarray(boxes, dtype=np.float32).reshape(-1, 4), path.stem


def collate_detection(batch):
    images, targets, ids = zip(*batch)
    return list(images), list(targets), list(ids)


def build_fasterrcnn(device: str):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE, box_detections_per_img=300
    )
    state = torch.load(ROOT / "assets/pretrained_weights" / WEIGHTS["fasterrcnn_resnet50_fpn"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model.to(device)


def evaluate_fasterrcnn(model, loader, device: str) -> dict[str, float | int]:
    import torch

    model.eval()
    ground_truth: dict[str, np.ndarray] = {}
    predictions: dict[str, list[tuple[float, list[float]]]] = defaultdict(list)
    with torch.inference_mode():
        for images, targets, ids in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for output, target, image_id in zip(outputs, targets, ids):
                ground_truth[image_id] = target
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                predictions[image_id].extend((float(score), box.tolist()) for box, score in zip(boxes, scores))
    return detection_metrics(ground_truth, predictions)


def run_fasterrcnn(
    dataset: str, run_dir: Path, config: dict[str, Any], epochs: int, batch: int, workers: int, device: str,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    train_set = YoloDetectionDataset(AUG_ROOT / dataset, "train")
    val_set = YoloDetectionDataset(DATA_ROOT / dataset, "val")
    test_set = YoloDetectionDataset(DATA_ROOT / dataset, "test")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=collate_detection, pin_memory=True, generator=generator, persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate_detection, pin_memory=True, persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate_detection, pin_memory=True, persistent_workers=workers > 0)
    model = build_fasterrcnn(device)
    start_epoch = 0
    if resume_from is not None:
        resume_payload = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state"], strict=True)
        start_epoch = int(resume_payload["epoch"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    metrics_path = run_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "map50", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn"]
    best = -1.0
    best_epoch = None
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        prior_best = torch.load(best_path, map_location="cpu", weights_only=False)
        best = float(prior_best["metrics"]["map50_95"])
        best_epoch = int(prior_best["epoch"])
    if resume_from is None:
        metrics_mode = "w"
    else:
        metrics_mode = "a"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if metrics_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch + 1, epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0
            for images, targets, _ in train_loader:
                images = [image.to(device, non_blocking=True) for image in images]
                target_dicts = []
                for boxes in targets:
                    target_dicts.append({
                        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                        "labels": torch.ones(len(boxes), dtype=torch.int64, device=device),
                    })
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                    loss = sum(model(images, target_dicts).values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Faster R-CNN loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu())
                batches += 1
            val = evaluate_fasterrcnn(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), **val}
            writer.writerow(row)
            handle.flush()
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": val, "config": config}, run_dir / "last.pt")
            if val["map50_95"] > best:
                best = float(val["map50_95"])
                best_epoch = epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": val, "config": config}, run_dir / "best.pt")
            print(json.dumps({"event": "epoch_complete", "dataset": dataset, "model": config["model"], "epoch": epoch, "val_map50_95": val["map50_95"]}), flush=True)
    best_payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"], strict=True)
    test = evaluate_fasterrcnn(model, test_loader, device)
    return {
        "best_epoch": best_epoch,
        "best_val_map50": best_payload["metrics"]["map50"],
        "best_val_map50_95": best_payload["metrics"]["map50_95"],
        **{f"test_{key}": value for key, value in test.items() if key in {"map50", "map50_95", "precision", "recall", "f1"}},
    }


def run_ultralytics(
    dataset: str, run_dir: Path, config: dict[str, Any], epochs: int, batch: int, workers: int, device: str
) -> dict[str, Any]:
    import albumentations as A
    import torch
    from ultralytics import RTDETR, YOLO

    weight = ROOT / "assets/pretrained_weights" / WEIGHTS[config["model"]]
    model = RTDETR(str(weight)) if config["model"].startswith("rtdetr_") else YOLO(str(weight))
    model.train(
        data=config["data_yaml"], imgsz=IMAGE_SIZE, batch=batch, epochs=epochs, patience=epochs,
        optimizer="AdamW", lr0=LR0, lrf=1.0, cos_lr=False, warmup_epochs=0.0,
        warmup_momentum=0.0, warmup_bias_lr=0.0, weight_decay=WEIGHT_DECAY,
        device=device, seed=SEED, deterministic=True, project=str(run_dir.parent), name=run_dir.name,
        exist_ok=True, workers=workers, amp=True, pretrained=True, single_cls=True,
        verbose=True, plots=False, save_period=-1, val=True,
        augmentations=[A.NoOp(p=0.0)], auto_augment=None, erasing=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
    )
    metrics_path = run_dir / "results.csv"
    best = run_dir / "weights/best.pt"
    if not metrics_path.exists() or not best.exists():
        raise RuntimeError(f"Ultralytics outputs missing for {config['model']}: {run_dir}")
    shutil.copy2(metrics_path, run_dir / "metrics.csv")
    val_metrics = read_best_metrics(metrics_path)
    best_model = RTDETR(str(best)) if config["model"].startswith("rtdetr_") else YOLO(str(best))
    test = best_model.val(
        data=config["data_yaml"], split="test", imgsz=IMAGE_SIZE, batch=batch,
        device=device, workers=workers, plots=False, augment=False, verbose=False,
    )
    result = {
        **val_metrics,
        **metric_row(test, "test"),
    }
    del model, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_one(dataset: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    yaml_path = yaml_for(dataset)
    batch = FRCNN_BATCH_OVERRIDE if model == "fasterrcnn_resnet50_fpn" and args.batch is None else (args.batch or 16)
    run_id = now_id(dataset, model)
    run_dir = RUN_ROOT / dataset / model / run_id
    config = make_config(dataset, model, yaml_path, args.device, args.epochs, batch)
    snapshot_run(run_dir, config, yaml_path, " ".join(os.sys.argv))
    started = time.time()
    print(json.dumps({"event": "formal_train_start", "dataset": dataset, "model": model, "run_dir": str(run_dir)}), flush=True)
    try:
        details = (
            run_fasterrcnn(dataset, run_dir, config, args.epochs, batch, args.workers, f"cuda:{args.device}" if args.device.isdigit() else args.device)
            if model == "fasterrcnn_resnet50_fpn"
            else run_ultralytics(dataset, run_dir, config, args.epochs, batch, args.workers, args.device)
        )
        best = run_dir / "weights/best.pt" if model != "fasterrcnn_resnet50_fpn" else run_dir / "best.pt"
        result = {
            **config, **details, "status": "completed", "runtime_seconds": round(time.time() - started, 3),
            "run_dir": str(run_dir), "best_checkpoint": str(best),
            "best_checkpoint_sha256": sha256(best), "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        result = {
            **config, "status": "failed", "runtime_seconds": round(time.time() - started, 3),
            "run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "formal_run_complete", "dataset": dataset, "model": model, "status": result["status"], "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return result


def write_summary(results: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = []
        if SUMMARY_PATH.exists():
            existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")).get("results", [])
        merged = existing + [
            row for row in results
            if row.get("run_dir") not in {old.get("run_dir") for old in existing}
        ]
        payload = json.dumps({
            "schema_version": 1,
            "experiment": EXPERIMENT_ID,
            "seed": SEED,
            "epochs": EPOCHS,
            "results": merged,
        }, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
        temporary = SUMMARY_PATH.with_name(f".{SUMMARY_PATH.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, SUMMARY_PATH)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    models = MODELS if args.model == "all" else (args.model,)
    for dataset in datasets:
        yaml_for(dataset)
    if args.check_only:
        print(json.dumps({"status": "PASS", "datasets": datasets, "models": models, "formal_training_started": False}))
        return 0
    results = []
    for dataset in datasets:
        for model in models:
            result = run_one(dataset, model, args)
            results.append(result)
            write_summary(results)
            if result["status"] != "completed":
                return 1
    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
