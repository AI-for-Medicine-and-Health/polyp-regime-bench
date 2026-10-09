#!/usr/bin/env python3
"""Download benchmark manifests or a selected released checkpoint from the Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = "AI-for-Medicine-and-Health/polyp-regime-bench-data"
MODEL_REPO = "AI-for-Medicine-and-Health/polyp-regime-bench-models"
CONDITIONS = ("scratch_base", "scratch_100ep", "pretrained_base", "pretrained_aug3x", "pretrained_aug5x", "pretrained_aug10x", "pretrained_repeat5x")
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = ("fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s", "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s", "rtdetr_l", "rtdetr_x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("manifests", help="download data split and augmentation manifests")
    model = sub.add_parser("model", help="download one checkpoint and its checksum index")
    model.add_argument("--condition", choices=CONDITIONS, required=True)
    model.add_argument("--seed", type=int, choices=(88, 123, 666), required=True)
    model.add_argument("--dataset", choices=DATASETS, required=True)
    model.add_argument("--model", choices=MODELS, required=True)
    predictions = sub.add_parser("predictions", help="download cached predictions for a condition and seed")
    predictions.add_argument("--condition", choices=CONDITIONS, required=True)
    predictions.add_argument("--seed", type=int, choices=(88, 123, 666), required=True)
    args = parser.parse_args()
    if args.action == "manifests":
        target = ROOT / "datasets"
        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=target,
                          allow_patterns=["manifests/**", "README.md"])
    elif args.action == "model":
        target = ROOT / "models"
        checkpoint = f"checkpoints/{args.condition}/seed{args.seed}/{args.dataset}/{args.model}/best.pt"
        snapshot_download(repo_id=MODEL_REPO, local_dir=target,
                          allow_patterns=[checkpoint, "checkpoint_index.jsonl", "README.md"])
        if not (target / checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint was not found in {MODEL_REPO}: {checkpoint}")
    else:
        target = ROOT / "models"
        pattern = f"predictions/consolidated_{args.condition}_seed{args.seed}_voting/**"
        snapshot_download(repo_id=MODEL_REPO, local_dir=target,
                          allow_patterns=[pattern, "prediction_index.jsonl", "README.md"])
    print(f"Downloaded to {target}")


if __name__ == "__main__":
    main()
