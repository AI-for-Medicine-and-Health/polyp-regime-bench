#!/usr/bin/env python3
"""Materialize model-definition YAMLs for true random-initialized training.

The public checkpoints are inspected only to recover their architecture
definition. Their parameter tensors are never copied into Scratch models.
The resulting YAML files are cached under scratch/configs/architectures so
training processes do not need to touch the public checkpoint files.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "scripts/experiments/kvasir_seg__polypgen_wli__polypdb_wli__seed42_training/scratch/configs/architectures"
WEIGHTS = ROOT / "assets/pretrained_weights"
MODELS = {
    "yolo11_s": ("yolo11s.yaml", "yolo11s.pt"),
    "yolov5_s": ("yolov5s.yaml", "yolov5su.pt"),
    "yolov8_s": ("yolov8s.yaml", "yolov8s.pt"),
    "yolov9_s": ("yolov9s.yaml", "yolov9s.pt"),
    "yolov3_tinyu": ("yolov3-tiny.yaml", "yolov3-tinyu.pt"),
    "yolov3_sppu": ("yolov3-spp.yaml", "yolov3-sppu.pt"),
    "yolov10_s": ("yolov10s.yaml", "yolov10s.pt"),
    "yolo12_s": ("yolo12s.yaml", "yolo12s.pt"),
    "yolo26_s": ("yolo26s.yaml", "yolo26s.pt"),
    "rtdetr_l": ("rtdetr-l.yaml", "rtdetr-l.pt"),
    "rtdetr_x": ("rtdetr-x.yaml", "rtdetr-x.pt"),
}


def main() -> int:
    from ultralytics import RTDETR, YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    for model_name, (yaml_name, weight_name) in MODELS.items():
        output = OUT / yaml_name
        if output.is_file():
            print(f"exists {model_name}: {output}")
            continue
        weight = WEIGHTS / weight_name
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model_cls = RTDETR if model_name.startswith("rtdetr_") else YOLO
        checkpoint = model_cls(str(weight))
        architecture = copy.deepcopy(checkpoint.model.yaml)
        architecture.pop("yaml_file", None)
        output.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
        del checkpoint
        print(f"created {model_name}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
