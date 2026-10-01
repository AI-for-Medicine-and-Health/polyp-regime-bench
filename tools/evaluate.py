#!/usr/bin/env python3
"""Evaluate a released Ultralytics checkpoint on a prepared local split."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml from a locally materialized split")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError("The checkpoint and dataset.yaml must both exist")
    architecture = (args.checkpoint.parent.parent.name if args.checkpoint.parent.name == "weights"
                    else args.checkpoint.parent.name)
    if architecture.startswith("rtdetr_"):
        from ultralytics import RTDETR
        model = RTDETR(str(args.checkpoint))
    elif architecture.startswith(("yolo", "yolov")):
        from ultralytics import YOLO
        model = YOLO(str(args.checkpoint))
    else:
        raise SystemExit("Use the original Faster R-CNN evaluation code for fasterrcnn_resnet50_fpn")
    kwargs = {"data": str(args.data), "split": args.split, "imgsz": args.imgsz,
              "project": "runs/quickstart_eval", "name": architecture, "exist_ok": False}
    if args.device is not None:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)
    print({"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)})


if __name__ == "__main__":
    main()
