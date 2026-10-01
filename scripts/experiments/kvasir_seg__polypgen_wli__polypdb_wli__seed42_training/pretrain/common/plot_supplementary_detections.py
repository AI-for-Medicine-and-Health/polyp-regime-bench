#!/usr/bin/env python3
"""Render reproducible, zoomable examples for supplementary figures S1/S2.

The six test images are selected by ground-truth box area quartiles, without
examining predictions. Model and vote choices use validation results only.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VOTING_CONDITION", "pretrained_aug5x")
os.environ.setdefault(
    "VOTING_DATA_VERSION", "kvasir_seg__polypgen_wli__polypdb_wli__seed42__aug5x"
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

import run_voting_analysis as voting


ROOT = voting.ROOT
VERSION = voting.BASE_DATA_VERSION
SEED = 88
DATASETS = voting.DATASETS
CONDITIONS = (
    "scratch_base",
    "pretrained_base",
    "pretrained_aug5x",
    "pretrained_aug10x",
    "pretrained_repeat5x",
)
INFERENCE_ROOT = ROOT / "runs/inference" / VERSION
MANIFEST_ROOT = ROOT / "datasets/manifests/experiments" / VERSION
OUTPUT_ROOT = ROOT / "reports" / VERSION / "figures" / "supplementary"
MAX_IMAGE_EDGE = 1000
COLORS = {"ground_truth": "#3de36d", "prediction": "#ff4d6d"}
MODEL_LABELS = {
    "fasterrcnn_resnet50_fpn": "Faster R-CNN\nResNet-50-FPN",
    "yolo11_s": "YOLO11-S", "yolov5_s": "YOLOv5-S", "yolov8_s": "YOLOv8-S",
    "yolov9_s": "YOLOv9-S", "yolov3_tinyu": "YOLOv3-TinyU",
    "yolov3_sppu": "YOLOv3-SPPU", "yolov10_s": "YOLOv10-S",
    "yolo12_s": "YOLO12-S", "yolo26_s": "YOLO26-S",
    "rtdetr_l": "RT-DETR-L", "rtdetr_x": "RT-DETR-X",
}
CONDITION_LABELS = {
    "scratch_base": "Scratch",
    "pretrained_base": "Pretrained",
    "pretrained_aug5x": "Pretrained + 5× aug",
    "pretrained_aug10x": "Pretrained + 10× aug",
    "pretrained_repeat5x": "Pretrained + 5× repeat",
}


def load_test_records(dataset: str) -> list[dict]:
    manifest = json.loads((MANIFEST_ROOT / f"{dataset}.json").read_text(encoding="utf-8"))
    return [record for record in manifest["records"] if record["split"] == "test"]


def gt_boxes(record: dict) -> list[list[float]]:
    path = ROOT / record["label"]
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"invalid detection label: {path}")
        _, xc, yc, w, h = map(float, values)
        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return boxes


def select_examples() -> dict[str, list[dict]]:
    selected = {}
    for dataset in DATASETS:
        positives = []
        for record in load_test_records(dataset):
            boxes = gt_boxes(record)
            if boxes:
                area = max((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
                positives.append((area, str(record["external_id"]), record))
        positives.sort(key=lambda item: (item[0], item[1]))
        if len(positives) < 2:
            raise RuntimeError(f"fewer than two positive test images in {dataset}")
        selected[dataset] = [positives[round(q * (len(positives) - 1))][2] for q in (0.25, 0.75)]
    return selected


def load_images(selected: dict[str, list[dict]]) -> dict[str, Image.Image]:
    output = {}
    for records in selected.values():
        for record in records:
            path = ROOT / record["image"]
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
                output[record["external_id"]] = image.copy()
    return output


def raw_predictions(condition: str, dataset: str, model: str, ids: set[str]) -> dict[str, list[dict]]:
    path = INFERENCE_ROOT / f"consolidated_{condition}_seed{SEED}_voting" / "raw" / dataset / "test" / f"{model}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["dataset"] != dataset or payload["model"] != model or payload["split"] != "test" or payload["seed"] != SEED:
        raise ValueError(f"incompatible prediction export: {path}")
    output = {image_id: [] for image_id in ids}
    for item in payload["predictions"]:
        if item["image_id"] in output:
            output[item["image_id"]].append({"box": item["box"], "score": float(item["score"]), "model": model})
    for image_id in output:
        output[image_id].sort(key=lambda item: item["score"], reverse=True)
        output[image_id] = output[image_id][:voting.MAX_CANDIDATES_PER_MODEL_IMAGE]
    return output


def draw_cell(ax, image: Image.Image, ground_truth: list[list[float]], predictions: list[dict]) -> None:
    ax.imshow(image)
    width, height = image.size
    for box in ground_truth:
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                               fill=False, linewidth=2.5, edgecolor=COLORS["ground_truth"]))
    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        ax.add_patch(Rectangle((x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
                               fill=False, linewidth=2.0, edgecolor=COLORS["prediction"]))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


def make_canvas(rows: list[str], selected: dict[str, list[dict]], images: dict[str, Image.Image]):
    fig, axes = plt.subplots(len(rows), 6, figsize=(23, 3 * len(rows) + 2), dpi=180)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.955, bottom=0.03, wspace=0.02, hspace=0.08)
    columns = [(dataset, record) for dataset in DATASETS for record in selected[dataset]]
    for col, (dataset, record) in enumerate(columns):
        label = {"kvasir_seg": "Kvasir-SEG", "polypgen_wli": "PolypGen WLI", "polypdb_wli": "PolypDB WLI"}[dataset]
        area_label = "Q1 area" if col % 2 == 0 else "Q3 area"
        axes[0, col].set_title(f"{label}\n{area_label}", fontsize=12, pad=9)
    for row, title in enumerate(rows):
        pos = axes[row, 0].get_position()
        fig.text(0.108, (pos.y0 + pos.y1) / 2, title,
                 ha="right", va="center", fontsize=11)
    return fig, axes, columns


def save_figure(fig, stem: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure_s1(selected: dict[str, list[dict]], images: dict[str, Image.Image]) -> None:
    rows = [MODEL_LABELS[model] for model in voting.MODELS]
    fig, axes, columns = make_canvas(rows, selected, images)
    fig.suptitle("S1  Pretrained + 5x augmentation: direct detections by 12 models (seed 88)\n"
                 "Green = ground truth; red = predictions at confidence >= 0.30", fontsize=18, y=0.995)
    fig.text(0.5, 0.008, "The same six test images are used in every row; Q1/Q3 refer to ground-truth maximum box-area quartiles.",
             ha="center", va="bottom", fontsize=10)
    for dataset in DATASETS:
        ids = {record["external_id"] for record in selected[dataset]}
        for row, model in enumerate(voting.MODELS):
            preds = raw_predictions("pretrained_aug5x", dataset, model, ids)
            for col, (ds, record) in enumerate(columns):
                if ds != dataset:
                    continue
                image_id = record["external_id"]
                draw_cell(axes[row, col], images[image_id], gt_boxes(record),
                          [item for item in preds[image_id] if item["score"] >= 0.30])
    save_figure(fig, "S1_aug5x_12_models_seed88")


def select_vote(analysis: dict) -> dict:
    eligible = [row for row in analysis["results"] if row["method"] != "single_best"]
    return max(eligible, key=lambda row: (row["val_metrics"]["mAP50_95"],
                                          row["val_metrics"]["recall"],
                                          -row["val_metrics"]["fp_per_image"]))


def fused_predictions(method: str, params: dict, weights: dict[str, float],
                      predictions: dict[str, list[dict]]) -> list[dict]:
    min_models = params.get("min_models")
    if method == "strict_consensus_vote":
        min_models = max(int(min_models), math.ceil(len(voting.MODELS) / 2))
    fused = voting.cluster_predictions(predictions, weights, float(params["cluster_iou"]),
                                       float(params["score_floor"]), min_models,
                                       use_median=method in ("majority_vote", "strict_consensus_vote"))
    return [item for item in fused if item["score"] >= float(params["operating_confidence"])]


def figure_s2(selected: dict[str, list[dict]], images: dict[str, Image.Image]) -> dict:
    rows = [f"{CONDITION_LABELS[condition]}\nsingle" if kind == "single" else f"{CONDITION_LABELS[condition]}\nvote"
            for condition in CONDITIONS for kind in ("single", "vote")]
    fig, axes, columns = make_canvas(rows, selected, images)
    fig.suptitle("S2  Same images and fixed single-model architecture across training conditions (seed 88)\n"
                 "Green = ground truth; red = predictions at the evaluation operating confidence", fontsize=18, y=0.995)
    fig.text(0.5, 0.008, "Single-model architecture is fixed per dataset. Vote rule and threshold are selected on validation; see metadata.",
             ha="center", va="bottom", fontsize=10)
    metadata = {}
    aug5_dir = INFERENCE_ROOT / f"consolidated_pretrained_aug5x_seed{SEED}_voting"
    aug5_analysis = json.loads((aug5_dir / "voting_analysis.json").read_text(encoding="utf-8"))
    for dataset in DATASETS:
        fixed_model = aug5_analysis["datasets"][dataset]["selections"]["single_best"]["selected_model"]
        ids = {record["external_id"] for record in selected[dataset]}
        metadata[dataset] = {"fixed_single_model": fixed_model, "votes": {}}
        for ci, condition in enumerate(CONDITIONS):
            directory = INFERENCE_ROOT / f"consolidated_{condition}_seed{SEED}_voting"
            analysis = json.loads((directory / "voting_analysis.json").read_text(encoding="utf-8"))["datasets"][dataset]
            vote = select_vote(analysis)
            method, params = vote["method"], vote["params"]
            weights = analysis["selections"][method]["model_weights"]
            metadata[dataset]["votes"][condition] = {"method": method, "params": params}
            by_model = {model: raw_predictions(condition, dataset, model, ids) for model in voting.MODELS}
            for col, (ds, record) in enumerate(columns):
                if ds != dataset:
                    continue
                image_id = record["external_id"]
                truth = gt_boxes(record)
                single = [item for item in by_model[fixed_model][image_id] if item["score"] >= 0.30]
                vote_preds = fused_predictions(method, params, weights,
                                               {model: by_model[model][image_id] for model in voting.MODELS})
                draw_cell(axes[2 * ci, col], images[image_id], truth, single)
                draw_cell(axes[2 * ci + 1, col], images[image_id], truth, vote_preds)
            print(f"S2 {dataset} {condition}: single={fixed_model} vote={method}", flush=True)
    save_figure(fig, "S2_conditions_single_vs_vote_seed88")
    return metadata


def main() -> None:
    selected = select_examples()
    images = load_images(selected)
    figure_s1(selected, images)
    metadata = figure_s2(selected, images)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "single_operating_confidence": 0.30,
        "selected_image_rule": "ground-truth maximum box area nearest Q1 and Q3 within each test split",
        "selected_images": {
            dataset: [{"image_id": r["external_id"], "image": r["image"], "gt_boxes": len(gt_boxes(r))}
                      for r in records] for dataset, records in selected.items()
        },
        "conditions": list(CONDITIONS),
        "s2_selection": metadata,
    }
    (OUTPUT_ROOT / "S1_S2_selection_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved figures and metadata to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
