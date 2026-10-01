#!/usr/bin/env python3
"""Train one model from random initialization on the base split."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[2]
ROOT = TRAINING_ROOT.parents[2]
PRETRAIN_TRAINER = TRAINING_ROOT / "pretrain/common/train_one.py"
BASE_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_VERSION
MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_VERSION
RUNS_ROOT = ROOT / "runs/training" / BASE_VERSION / "scratch_base"
ARCH_ROOT = TRAINING_ROOT / "scratch/configs/architectures"
EPOCHS = 25
IMAGE_SIZE = 640
LR0 = 1e-4
WEIGHT_DECAY = 5e-4
FRCNN_BATCH = 4
ULTRALYTICS_BATCH = 16
WORKERS = 2
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s",
    "rtdetr_l", "rtdetr_x",
)
ARCHITECTURES = {
    "yolo11_s": "yolo11s.yaml", "yolov5_s": "yolov5s.yaml", "yolov8_s": "yolov8s.yaml",
    "yolov9_s": "yolov9s.yaml", "yolov3_tinyu": "yolov3-tiny.yaml",
    "yolov3_sppu": "yolov3-spp.yaml", "yolov10_s": "yolov10s.yaml",
    "yolo12_s": "yolo12s.yaml", "yolo26_s": "yolo26s.yaml",
    "rtdetr_l": "rtdetr-l.yaml", "rtdetr_x": "rtdetr-x.yaml",
}


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location("pretrain_train_one_for_scratch", PRETRAIN_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared training helpers: {PRETRAIN_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_shared(module, seed: int) -> None:
    module.DATA_VERSION = BASE_VERSION
    module.DATA_ROOT = DATA_ROOT
    module.AUG_ROOT = DATA_ROOT
    module.YAML_ROOT = DATA_ROOT
    module.MANIFEST_ROOT = MANIFEST_ROOT
    module.VERSION_MANIFEST_ROOT = MANIFEST_ROOT
    module.RUN_ROOT = RUNS_ROOT / f"seed{seed}"
    module.SEED = seed


def build_scratch_fasterrcnn(device: str):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE,
        box_detections_per_img=300,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model.to(device)


def run_scratch_ultralytics(core, dataset: str, run_dir: Path, config: dict, epochs: int, batch: int, workers: int, device: str):
    import albumentations as A
    import torch
    import yaml
    from ultralytics import RTDETR, YOLO

    architecture = ARCH_ROOT / ARCHITECTURES[config["model"]]
    if not architecture.is_file():
        raise FileNotFoundError(
            f"missing cached architecture {architecture}; run prepare_architectures.py before Scratch training"
        )
    run_architecture = run_dir / architecture.name
    run_architecture.write_text(architecture.read_text(encoding="utf-8"), encoding="utf-8")
    model_cls = RTDETR if config["model"].startswith("rtdetr_") else YOLO
    model = model_cls(str(architecture))
    model.train(
        data=config["data_yaml"], imgsz=IMAGE_SIZE, batch=batch, epochs=epochs, patience=epochs,
        optimizer="AdamW", lr0=LR0, lrf=1.0, cos_lr=False, warmup_epochs=0.0,
        warmup_momentum=0.0, warmup_bias_lr=0.0, weight_decay=WEIGHT_DECAY,
        device=device, seed=config["seed"], deterministic=True, project=str(run_dir.parent), name=run_dir.name,
        exist_ok=True, workers=workers, amp=True, pretrained=False, single_cls=True,
        verbose=True, plots=False, save_period=-1, val=True,
        augmentations=[A.NoOp(p=0.0)], auto_augment=None, erasing=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
    )
    metrics_path = run_dir / "results.csv"
    best = run_dir / "weights/best.pt"
    if not metrics_path.exists() or not best.exists():
        raise RuntimeError(f"Scratch Ultralytics outputs missing for {config['model']}: {run_dir}")
    import shutil
    shutil.copy2(metrics_path, run_dir / "metrics.csv")
    val_metrics = core.read_best_metrics(metrics_path)
    best_model = model_cls(str(best))
    test = best_model.val(
        data=config["data_yaml"], split="test", imgsz=IMAGE_SIZE, batch=batch,
        device=device, workers=workers, plots=False, augment=False, verbose=False,
    )
    result = {**val_metrics, **core.metric_row(test, "test")}
    del model, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def make_config(dataset: str, model: str, seed: int, device: str, run_dir: Path) -> dict:
    manifest = MANIFEST_ROOT / f"{dataset}.json"
    yaml_path = DATA_ROOT / dataset / "dataset.yaml"
    architecture = ARCH_ROOT / ARCHITECTURES.get(model, "fasterrcnn_resnet50_fpn.yaml")
    result = {
        "experiment": f"{BASE_VERSION}__scratch_base__seed{seed}",
        "dataset": dataset, "model": model,
        "framework": "torchvision" if model == "fasterrcnn_resnet50_fpn" else "ultralytics",
        "initialization": "random_architecture_initialization", "pretrained": False,
        "seed": seed, "split_seed": 42, "dataset_version": BASE_VERSION,
        "split_manifest": str(manifest.relative_to(ROOT)), "split_manifest_sha256": sha256(manifest),
        "data_yaml": str(yaml_path.resolve()), "augmentation_condition": "scratch_base",
        "architecture_yaml": str(architecture.relative_to(ROOT)) if model != "fasterrcnn_resnet50_fpn" else None,
        "architecture_sha256": sha256(architecture) if model != "fasterrcnn_resnet50_fpn" else None,
        "epochs": EPOCHS, "imgsz": IMAGE_SIZE,
        "batch": FRCNN_BATCH if model == "fasterrcnn_resnet50_fpn" else ULTRALYTICS_BATCH,
        "workers": WORKERS, "optimizer": "AdamW", "initial_lr": LR0,
        "weight_decay": WEIGHT_DECAY, "online_augmentation": False, "amp": True,
        "deterministic": True, "device": device,
        "physical_device": os.environ.get("GI_PHYSICAL_DEVICE", device), "run_dir": str(run_dir),
    }
    return result


def snapshot(run_dir: Path, config: dict, manifest: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "dataset.yaml").write_text(Path(config["data_yaml"]).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "version_manifest.json").write_text(
        (MANIFEST_ROOT / "version_manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({**config, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--condition", default="scratch_base", choices=("scratch_base",))
    parser.add_argument("--seed", type=int, required=True, choices=(88, 123, 666))
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.epochs != EPOCHS:
        raise SystemExit(f"this protocol requires epochs={EPOCHS}")
    for required in (DATA_ROOT / args.dataset / "dataset.yaml", MANIFEST_ROOT / f"{args.dataset}.json", MANIFEST_ROOT / "version_manifest.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.model != "fasterrcnn_resnet50_fpn" and not (ARCH_ROOT / ARCHITECTURES[args.model]).is_file():
        raise FileNotFoundError(ARCH_ROOT / ARCHITECTURES[args.model])
    shared = load_pretrain_module()
    configure_shared(shared, args.seed)
    shared.ensure_label_aliases(args.dataset)
    run_dir = RUNS_ROOT / f"seed{args.seed}" / args.dataset / args.model
    if args.check_only:
        print(json.dumps({"status": "PASS", "dataset": args.dataset, "model": args.model, "seed": args.seed, "pretrained": False, "run_dir": str(run_dir)}))
        return 0
    existing = [path for path in run_dir.iterdir() if path.name != "train.log"] if run_dir.exists() else []
    if existing:
        status_path = run_dir / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
            print(json.dumps({"status": "SKIP_COMPLETED", "run_dir": str(run_dir)}), flush=True)
            return 0
        raise RuntimeError(f"model output directory is not empty: {run_dir}")
    core = shared.load_core()
    core.build_fasterrcnn = build_scratch_fasterrcnn
    core.run_ultralytics = lambda dataset, run_dir, config, epochs, batch, workers, device: run_scratch_ultralytics(
        core, dataset, run_dir, config, epochs, batch, workers, device
    )
    config = make_config(args.dataset, args.model, args.seed, args.device, run_dir)
    snapshot(run_dir, config, MANIFEST_ROOT / f"{args.dataset}.json")
    started = time.time()
    print(json.dumps({"event": "scratch_train_start", **config}, ensure_ascii=False), flush=True)
    try:
        if args.model == "fasterrcnn_resnet50_fpn":
            details = core.run_fasterrcnn(args.dataset, run_dir, config, EPOCHS, FRCNN_BATCH, args.workers, f"cuda:{args.device}")
            best = run_dir / "best.pt"
        else:
            details = core.run_ultralytics(args.dataset, run_dir, config, EPOCHS, ULTRALYTICS_BATCH, args.workers, args.device)
            best = run_dir / "weights/best.pt"
        result = {
            **config, **details, "status": "completed", "runtime_seconds": round(time.time() - started, 3),
            "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        code = 0
    except Exception as exc:
        result = {
            **config, "status": "failed", "runtime_seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        code = 1
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "scratch_train_complete", "dataset": args.dataset, "model": args.model, "seed": args.seed, "status": result["status"], "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
