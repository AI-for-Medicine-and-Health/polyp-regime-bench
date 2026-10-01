#!/usr/bin/env python3
"""Run a released Ultralytics YOLO or RT-DETR checkpoint on local images."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="image, video, or directory of images")
    parser.add_argument("--output", type=Path, default=Path("runs/quickstart"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.source.exists():
        raise FileNotFoundError("The checkpoint and source must both exist")
    architecture = (args.checkpoint.parent.parent.name if args.checkpoint.parent.name == "weights"
                    else args.checkpoint.parent.name)
    if architecture.startswith("rtdetr_"):
        from ultralytics import RTDETR
        model = RTDETR(str(args.checkpoint))
    elif architecture.startswith(("yolo", "yolov")):
        from ultralytics import YOLO
        model = YOLO(str(args.checkpoint))
    else:
        raise SystemExit("This quick-start handles YOLO/RT-DETR checkpoints; use the original Faster R-CNN evaluator for fasterrcnn_resnet50_fpn")
    kwargs = {"source": str(args.source), "conf": args.conf, "imgsz": args.imgsz,
              "project": str(args.output), "name": architecture, "exist_ok": False, "save": True}
    if args.device is not None:
        kwargs["device"] = args.device
    model.predict(**kwargs)


if __name__ == "__main__":
    main()
