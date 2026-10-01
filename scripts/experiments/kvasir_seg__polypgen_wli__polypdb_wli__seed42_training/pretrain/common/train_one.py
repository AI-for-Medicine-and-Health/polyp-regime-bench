#!/usr/bin/env python3
"""Train one model on one current WLI-aligned dataset.

This entry point deliberately imports the validated training implementation
for the model mechanics, but supplies the current dataset version, output
layout, seed, epoch budget, and manifests locally. It never references the
historical CVC-inclusive experiment directories.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ROOT = SCRIPT_ROOT.parents[3]
CORE_PATH = ROOT / "scripts/experiments/kvasir_seg__polypgen_wli__polypdb_wli__seed42_training/pretrain/common/training_core.py"
BASE_DATA_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
AUG3X_DATA_VERSION = BASE_DATA_VERSION + "__aug3x"
AUG5X_DATA_VERSION = BASE_DATA_VERSION + "__aug5x"
AUG10X_DATA_VERSION = BASE_DATA_VERSION + "__aug10x"
REPEAT5X_DATA_VERSION = BASE_DATA_VERSION + "__repeat5x"
DATA_VERSION = BASE_DATA_VERSION
DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
AUG_ROOT = DATA_ROOT
YAML_ROOT = DATA_ROOT
MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
VERSION_MANIFEST_ROOT = MANIFEST_ROOT
RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_base" / "seed88"
SEED = 88
EPOCHS = 25
IMAGE_SIZE = 640
LR0 = 1e-4
WEIGHT_DECAY = 5e-4
WORKERS = 2
ULTRALYTICS_BATCH = 16
FRCNN_BATCH = 4
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


def load_core():
    spec = importlib.util.spec_from_file_location("validated_training_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA_VERSION = DATA_VERSION
    module.DATA_ROOT = DATA_ROOT
    module.AUG_ROOT = AUG_ROOT
    module.SEED = SEED
    module.EPOCHS = EPOCHS
    module.IMAGE_SIZE = IMAGE_SIZE
    module.LR0 = LR0
    module.WEIGHT_DECAY = WEIGHT_DECAY
    module.FRCNN_BATCH_OVERRIDE = FRCNN_BATCH
    module.DATASETS = DATASETS
    module.MODELS = MODELS
    module.WEIGHTS = WEIGHTS
    return module


def configure_condition(condition: str) -> None:
    global DATA_VERSION, DATA_ROOT, AUG_ROOT, YAML_ROOT, MANIFEST_ROOT, VERSION_MANIFEST_ROOT, RUN_ROOT
    if condition == "pretrained_base":
        DATA_VERSION = BASE_DATA_VERSION
        DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
        AUG_ROOT = DATA_ROOT
        YAML_ROOT = DATA_ROOT
        MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
        VERSION_MANIFEST_ROOT = MANIFEST_ROOT
        RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_base" / "seed88"
    elif condition == "pretrained_aug3x":
        DATA_VERSION = AUG3X_DATA_VERSION
        DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
        AUG_ROOT = ROOT / "datasets/materialized/active" / AUG3X_DATA_VERSION
        YAML_ROOT = AUG_ROOT
        MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
        VERSION_MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / AUG3X_DATA_VERSION
        RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_aug3x" / "seed88"
    elif condition == "pretrained_aug10x":
        DATA_VERSION = AUG10X_DATA_VERSION
        DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
        AUG_ROOT = ROOT / "datasets/materialized/active" / AUG10X_DATA_VERSION
        YAML_ROOT = AUG_ROOT
        MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
        VERSION_MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / AUG10X_DATA_VERSION
        RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_aug10x" / "seed88"
    elif condition == "pretrained_aug5x":
        DATA_VERSION = AUG5X_DATA_VERSION
        DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
        AUG_ROOT = ROOT / "datasets/materialized/active" / AUG5X_DATA_VERSION
        YAML_ROOT = AUG_ROOT
        MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
        VERSION_MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / AUG5X_DATA_VERSION
        RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_aug5x" / "seed88"
    elif condition == "pretrained_repeat5x":
        DATA_VERSION = REPEAT5X_DATA_VERSION
        DATA_ROOT = ROOT / "datasets/materialized/active" / BASE_DATA_VERSION
        AUG_ROOT = ROOT / "datasets/materialized/active" / REPEAT5X_DATA_VERSION
        YAML_ROOT = AUG_ROOT
        MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / BASE_DATA_VERSION
        VERSION_MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / REPEAT5X_DATA_VERSION
        RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / "pretrained_repeat5x" / "seed88"
    else:
        raise ValueError(f"unsupported condition: {condition}")


def configure_seed(seed: int) -> None:
    global SEED, RUN_ROOT
    if seed not in (88, 123, 666):
        raise ValueError(f"unsupported training seed: {seed}")
    SEED = seed
    condition = {
        AUG3X_DATA_VERSION: "pretrained_aug3x",
        AUG5X_DATA_VERSION: "pretrained_aug5x",
        AUG10X_DATA_VERSION: "pretrained_aug10x",
        REPEAT5X_DATA_VERSION: "pretrained_repeat5x",
    }.get(DATA_VERSION, "pretrained_base")
    RUN_ROOT = ROOT / "runs/training" / BASE_DATA_VERSION / condition / f"seed{SEED}"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_label_aliases(dataset: str) -> None:
    """Ultralytics expects labels beside images under a directory named labels."""
    dataset_root = AUG_ROOT / dataset
    for split in ("train", "val", "test"):
        alias = dataset_root / split / "labels"
        target = Path("labels_detection")
        if alias.is_symlink():
            if alias.resolve() != (alias.parent / target).resolve():
                raise RuntimeError(f"wrong labels alias: {alias}")
            continue
        if alias.exists():
            raise RuntimeError(f"unexpected existing labels path: {alias}")
        if not (alias.parent / target).is_dir():
            raise FileNotFoundError(alias.parent / target)
        alias.symlink_to(target, target_is_directory=True)


def config_for(dataset: str, model: str, device: str, run_dir: Path) -> dict:
    weight = ROOT / "assets/pretrained_weights" / WEIGHTS[model]
    manifest = MANIFEST_ROOT / f"{dataset}.json"
    yaml_path = YAML_ROOT / dataset / "dataset.yaml"
    condition = {
        AUG3X_DATA_VERSION: "pretrained_aug3x",
        AUG5X_DATA_VERSION: "pretrained_aug5x",
        AUG10X_DATA_VERSION: "pretrained_aug10x",
        REPEAT5X_DATA_VERSION: "pretrained_repeat5x",
    }.get(DATA_VERSION, "pretrained_base")
    return {
        "experiment": f"{DATA_VERSION}__{condition}__seed{SEED}",
        "dataset": dataset,
        "model": model,
        "framework": "torchvision" if model == "fasterrcnn_resnet50_fpn" else "ultralytics",
        "initialization": "public_pretrained",
        "pretrained": True,
        "seed": SEED,
        "split_seed": 42,
        "dataset_version": DATA_VERSION,
        "split_manifest": str(manifest.relative_to(ROOT)),
        "split_manifest_sha256": sha256(manifest),
        "data_yaml": str(yaml_path.resolve()),
        "augmentation_condition": condition,
        "weight": str(weight.resolve()),
        "weight_sha256": sha256(weight),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": FRCNN_BATCH if model == "fasterrcnn_resnet50_fpn" else ULTRALYTICS_BATCH,
        "workers": WORKERS,
        "optimizer": "AdamW",
        "initial_lr": LR0,
        "weight_decay": WEIGHT_DECAY,
        "online_augmentation": False,
        "amp": True,
        "deterministic": True,
        "device": device,
        "physical_device": os.environ.get("GI_PHYSICAL_DEVICE", device),
        "run_dir": str(run_dir),
    }


def snapshot(run_dir: Path, config: dict) -> None:
    # The scheduler pre-creates the model directory and opens train.log before
    # this process starts. All other files are still required to be new.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "dataset.yaml").write_text(Path(config["data_yaml"]).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        (ROOT / config["split_manifest"]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "version_manifest.json").write_text(
        (VERSION_MANIFEST_ROOT / "version_manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({**config, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()}, indent=2)
        + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--condition", default="pretrained_base", choices=("pretrained_base", "pretrained_aug3x", "pretrained_aug5x", "pretrained_aug10x", "pretrained_repeat5x"))
    parser.add_argument("--seed", type=int, default=88, choices=(88, 123, 666))
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="resume Faster R-CNN model weights from a saved last.pt; optimizer state is reinitialized")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_condition(args.condition)
    configure_seed(args.seed)
    batch_size = args.batch or (FRCNN_BATCH if args.model == "fasterrcnn_resnet50_fpn" else ULTRALYTICS_BATCH)
    if batch_size < 1:
        raise SystemExit("batch must be a positive integer")
    if args.epochs < 1 or args.epochs > EPOCHS:
        raise SystemExit(f"epochs must be between 1 and {EPOCHS}")
    for required in (YAML_ROOT / args.dataset / "dataset.yaml", ROOT / "assets/pretrained_weights" / WEIGHTS[args.model]):
        if not required.is_file():
            raise FileNotFoundError(required)
    ensure_label_aliases(args.dataset)
    run_dir = RUN_ROOT / args.dataset / args.model
    if args.check_only:
        print(json.dumps({"status": "PASS", "dataset": args.dataset, "model": args.model, "run_dir": str(run_dir)}))
        return 0
    # The coordinator creates train.log before spawning this process. That
    # log is not a training artifact and must not block the first run.
    existing = [path for path in run_dir.iterdir() if path.name != "train.log"]
    previous_config = None
    if args.resume_checkpoint is not None:
        checkpoint = args.resume_checkpoint.resolve()
        expected_checkpoint = (run_dir / "last.pt").resolve()
        if args.model != "fasterrcnn_resnet50_fpn" or checkpoint != expected_checkpoint:
            raise SystemExit("--resume-checkpoint is supported only for this run's Faster R-CNN last.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        previous_status = run_dir / "status.json"
        if previous_status.is_file():
            previous_config = json.loads(previous_status.read_text(encoding="utf-8"))
    if existing:
        status_path = run_dir / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
            print(json.dumps({"status": "SKIP_COMPLETED", "run_dir": str(run_dir)}), flush=True)
            return 0
        if args.resume_checkpoint is None:
            raise RuntimeError(f"model output directory is not empty: {run_dir}")

    core = load_core()
    config = config_for(args.dataset, args.model, args.device, run_dir)
    config["epochs"] = args.epochs
    config["batch"] = batch_size
    if args.resume_checkpoint is not None:
        checkpoint_payload = __import__("torch").load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        config["resume_checkpoint"] = str(args.resume_checkpoint.resolve())
        config["resume_from_epoch"] = int(checkpoint_payload["epoch"])
        config["previous_batch"] = (previous_config or {}).get("batch")
        config["optimizer_state_resumed"] = False
    snapshot(run_dir, config)
    started = time.time()
    print(json.dumps({"event": "train_start", **config}, ensure_ascii=False), flush=True)
    try:
        if args.model == "fasterrcnn_resnet50_fpn":
            details = core.run_fasterrcnn(
                args.dataset, run_dir, config, args.epochs, batch_size, args.workers, f"cuda:{args.device}",
                resume_from=args.resume_checkpoint,
            )
            best = run_dir / "best.pt"
        else:
            details = core.run_ultralytics(
                args.dataset, run_dir, config, args.epochs, batch_size, args.workers, args.device
            )
            best = run_dir / "weights/best.pt"
        result = {
            **config, **details, "status": "completed",
            "runtime_seconds": round(time.time() - started, 3),
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": sha256(best),
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
    print(json.dumps({"event": "train_complete", "dataset": args.dataset, "model": args.model, "status": result["status"], "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
