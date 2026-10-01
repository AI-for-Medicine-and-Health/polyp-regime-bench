#!/usr/bin/env python3
"""Validation-selected, CRV-inspired box-confidence verification pilot.

For each majority-vote candidate, retrieve lesion-free training patches using a
frozen ImageNet ResNet-18 descriptor, replace the candidate box with each
matched patch, and re-query the validation-selected frozen detector. A robust
positive response drop weighted by descriptor-space appearance change is added
to the candidate score. Alpha is selected on validation mAP50-95 only; the
test split is processed afterward with alpha frozen.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().with_name("run_voting_analysis.py")
ROOT = SCRIPT.parents[5]
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
DATA_VERSION = BASE + "__aug5x"
DATA_ROOT = ROOT / "datasets/materialized/active" / DATA_VERSION
OUTPUT_ROOT = ROOT / "runs/inference" / BASE / "crv_pretrained_aug5x"
REFERENCE_ROOT = OUTPUT_ROOT / "normal_reference_banks"
TOP_K = 3
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
IMAGE_SIZE = 640
MAX_REFERENCES = 4000


def load_voting_module(seed: int, output_name: str):
    os.environ["VOTING_CONDITION"] = "pretrained_aug5x"
    os.environ["VOTING_SEED"] = str(seed)
    os.environ["VOTING_OUTPUT_NAME"] = output_name
    spec = importlib.util.spec_from_file_location("voting_analysis_crv", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load voting module: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenPatchEncoder:
    def __init__(self, device: str):
        import torch
        import torch.nn.functional as F
        from torchvision.models import ResNet18_Weights, resnet18

        self.torch = torch
        self.F = F
        torch.set_num_threads(2)
        self.device = f"cuda:{device}" if str(device).isdigit() else device
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def encode(self, bgr_patches: list[np.ndarray], batch: int = 64) -> np.ndarray:
        if not bgr_patches:
            return np.empty((0, 768), dtype=np.float32)
        output = []
        with self.torch.inference_mode():
            for start in range(0, len(bgr_patches), batch):
                chunk = bgr_patches[start:start + batch]
                arr = np.stack([
                    cv2.cvtColor(cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    for p in chunk
                ])
                x = self.torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 255.0
                x = (x - self.mean) / self.std
                m = self.model
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                x = m.layer1(x)
                f2 = m.layer2(x)
                f3 = m.layer3(f2)
                v2 = self.F.adaptive_avg_pool2d(f2, 1).flatten(1)
                v3 = self.F.adaptive_avg_pool2d(f3, 1).flatten(1)
                desc = self.F.normalize(self.torch.cat([v2, v3], dim=1), dim=1)
                output.append(desc.cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0)


def load_or_build_reference_bank(dataset: str, encoder: FrozenPatchEncoder, cap: int = MAX_REFERENCES):
    cache = REFERENCE_ROOT / f"{dataset}_resnet18_layer2_layer3.npz"
    if cache.is_file():
        payload = np.load(cache)
        return payload["features"], payload["patches"]

    image_root = DATA_ROOT / dataset / "train" / "images"
    mask_root = DATA_ROOT / dataset / "train" / "masks"
    image_paths = sorted(p for p in image_root.iterdir() if p.is_file() and "__orig__" in p.stem)
    if not image_paths:
        raise RuntimeError(f"no original training images in {image_root}")

    patches: list[np.ndarray] = []
    for path in image_paths:
        mask_path = mask_root / f"{path.stem}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = image.shape[:2]
        scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        # Exclude the lesion and a 12-pixel context margin from normal references.
        lesion = (mask >= 128).astype(np.uint8)
        lesion = cv2.dilate(lesion, np.ones((25, 25), np.uint8), iterations=1)
        ph, pw = 64, 64
        for y in range(0, image.shape[0] - ph + 1, ph):
            for x in range(0, image.shape[1] - pw + 1, pw):
                patch = image[y:y + ph, x:x + pw]
                if float((lesion[y:y + ph, x:x + pw] > 0).mean()) > 0.01:
                    continue
                if float(patch.mean()) < 18.0:
                    continue
                patches.append(patch.copy())
                if len(patches) >= cap:
                    break
            if len(patches) >= cap:
                break
        if len(patches) >= cap:
            break
    if len(patches) < 50:
        raise RuntimeError(f"too few mask-excluded normal patches for {dataset}: {len(patches)}")

    features = encoder.encode(patches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, patches=np.stack(patches))
    return features, np.stack(patches)


def resize_keep_aspect(image: np.ndarray, width: int = IMAGE_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, IMAGE_SIZE / h)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def make_counterfactual(image: np.ndarray, box: list[float], donor: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(x1 + 1, min(w, int(round(box[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(box[3] * h))))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    donor = cv2.resize(donor, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out = image.copy()
    roi = out[y1:y2, x1:x2]
    # Smooth box-edge transitions to avoid a hard pasted rectangle.
    alpha = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    fade = min(12, max(2, min(alpha.shape) // 5))
    alpha[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    alpha[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    alpha[:, :fade] *= np.linspace(0, 1, fade, dtype=np.float32)[None, :]
    alpha[:, -fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[None, :]
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, fade / 2))[:, :, None]
    out[y1:y2, x1:x2] = (roi.astype(np.float32) * (1.0 - alpha) + donor.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


def verifier_original_score(verifier_preds: list[dict[str, Any]], box: list[float], iou_fn) -> float:
    matches = [float(p["score"]) for p in verifier_preds if iou_fn(p["box"], box) >= 0.50]
    return max(matches, default=0.0)


def query_counterfactual_scores(model, records, batch: int, device: str, score_floor: float, iou_fn):
    import torch

    scores = []
    with torch.inference_mode():
        for start in range(0, len(records), batch):
            chunk = records[start:start + batch]
            images = [record["image"] for record in chunk]
            outputs = model.predict(
                source=images, imgsz=IMAGE_SIZE, batch=min(batch, len(images)),
                conf=score_floor, device=device, stream=False, verbose=False, save=False,
            )
            for record, result in zip(chunk, outputs):
                if result.boxes is None:
                    scores.append(0.0)
                    continue
                height, width = result.orig_shape
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                candidates = [
                    (float(score), [float(box[0] / width), float(box[1] / height), float(box[2] / width), float(box[3] / height)])
                    for box, score in zip(boxes, confs)
                ]
                matched = [score for score, other in candidates if iou_fn(other, record["box"]) >= 0.50]
                scores.append(max(matched, default=0.0))
    return scores


def run_split(va, dataset: str, split: str, predictions, verifier: str, params: dict[str, Any],
              bank_features: np.ndarray, bank_patches: np.ndarray, encoder: FrozenPatchEncoder,
              model, device: str, batch: int, score_floor: float):
    import torch

    gt = va.load_ground_truth(dataset, split)
    verifier_predictions = predictions[verifier]
    fused = va.fuse_dataset(predictions, "majority_vote", params, {m: 1.0 for m in va.MODELS})
    evidence_predictions: dict[str, list[dict[str, Any]]] = {}
    image_paths = {p.stem: p for p in va.image_paths(dataset, split)}
    bank = torch.from_numpy(bank_features).to(encoder.device)
    all_images = sorted(gt)

    for image_id in all_images:
        path = image_paths[image_id]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        candidates = fused.get(image_id, [])
        if not candidates:
            evidence_predictions[image_id] = []
            continue

        crops = []
        usable = []
        h, w = image.shape[:2]
        for index, item in enumerate(candidates):
            x1 = max(0, min(w - 1, int(item["box"][0] * w)))
            y1 = max(0, min(h - 1, int(item["box"][1] * h)))
            x2 = max(x1 + 1, min(w, int(np.ceil(item["box"][2] * w))))
            y2 = max(y1 + 1, min(h, int(np.ceil(item["box"][3] * h))))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                continue
            orig_response = verifier_original_score(verifier_predictions.get(image_id, []), item["box"], va.box_iou)
            if orig_response <= score_floor:
                continue
            crops.append(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA))
            usable.append((index, item, orig_response))

        evidence = {i: [] for i in range(len(candidates))}
        if crops:
            query_features = encoder.encode(crops, batch=64)
            similarities = query_features @ bank_features.T
            distances = 1.0 - similarities
            pending = []
            meta = []
            appearance_raw: dict[int, list[tuple[float, int, float]]] = {candidate_idx: [] for candidate_idx, _, _ in usable}
            for query_idx, (candidate_idx, item, orig_response) in enumerate(usable):
                nearest = np.argpartition(similarities[query_idx], -TOP_K)[-TOP_K:]
                nearest = nearest[np.argsort(similarities[query_idx, nearest])[::-1]]
                query_patch = crops[query_idx].astype(np.float32)
                for ref_idx in nearest:
                    donor_patch = bank_patches[int(ref_idx)]
                    cf = make_counterfactual(image, item["box"], donor_patch)
                    if cf is None:
                        continue
                    raw_change = float(np.abs(query_patch - donor_patch.astype(np.float32)).mean() / 255.0)
                    appearance_raw[candidate_idx].append((raw_change, int(ref_idx), orig_response))
                    pending.append({"image": cf, "box": item["box"]})
                    meta.append((candidate_idx, orig_response, raw_change))
                    if len(pending) >= batch:
                        cf_scores = query_counterfactual_scores(model, pending, batch, device, score_floor, va.box_iou)
                        for (cand_idx, orig, app), cf_score in zip(meta, cf_scores):
                            evidence[cand_idx].append((orig, app, cf_score))
                        pending.clear(); meta.clear()
            if pending:
                cf_scores = query_counterfactual_scores(model, pending, batch, device, score_floor, va.box_iou)
                for (candidate_idx, orig, app), cf_score in zip(meta, cf_scores):
                    evidence[candidate_idx].append((orig, app, cf_score))

            # Per-image robust normalization mirrors CRV's normalization of
            # appearance change before combining it with positive response drop.
            all_changes = [change for values in appearance_raw.values() for change, _, _ in values]
            q50 = float(np.quantile(all_changes, 0.50)) if all_changes else 0.0
            q995 = float(np.quantile(all_changes, 0.995)) if all_changes else 1.0
            scale = max(q995 - q50, 1e-6)
            for candidate_idx, values in evidence.items():
                weighted_drops = [
                    max(0.0, orig - cf) * float(np.clip((change - q50) / scale, 0.0, 1.0))
                    for orig, change, cf in values
                ]
                evidence[candidate_idx] = statistics.median(weighted_drops) if weighted_drops else []

        evidence_predictions[image_id] = [
            {"box": item["box"], "score": float(item["score"]), "verification": float(evidence[i]) if isinstance(evidence[i], (int, float)) else 0.0, "support": int(item.get("support", 0))}
            for i, item in enumerate(candidates)
        ]

    # Note: candidate batches can cross image/candidate boundaries. This strict
    # assertion catches any accidental mismatch between raw detector outputs
    # and response records before scoring.
    if set(evidence_predictions) != set(gt):
        raise RuntimeError("prediction/ground-truth image IDs do not match")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return evidence_predictions, gt


def apply_alpha(predictions: dict[str, list[dict[str, Any]]], alpha: float):
    return {
        image_id: [{"box": item["box"], "score": float(np.clip(item["score"] + alpha * item["verification"], 0.0, 1.0))} for item in items]
        for image_id, items in predictions.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(88, 123, 666), required=True)
    parser.add_argument("--dataset", choices=("kvasir_seg", "polypgen_wli", "polypdb_wli"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--score-floor", type=float, default=0.001)
    args = parser.parse_args()

    output_name = f"consolidated_pretrained_aug5x_seed{args.seed}_voting"
    va = load_voting_module(args.seed, output_name)
    summary_path = va.INFER_ROOT / "voting_analysis.json"
    baseline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_summary = baseline_summary["datasets"][args.dataset]
    method_rows = {x["method"]: x for x in dataset_summary["results"]}
    majority_params = method_rows["majority_vote"]["params"]
    verifier = method_rows["single_best"]["params"]["model"]

    encoder = FrozenPatchEncoder(args.device)
    bank_features, bank_patches = load_or_build_reference_bank(args.dataset, encoder)
    model_predictions = {m: va.load_predictions(args.dataset, "val", m) for m in va.MODELS}
    # Counterfactual verification is evaluated on the same fused candidate
    # pool for both baseline and CRV; all selection uses validation only.
    from ultralytics import RTDETR, YOLO
    checkpoint = va.checkpoint_path(args.dataset, verifier)
    model = RTDETR(str(checkpoint)) if verifier.startswith("rtdetr_") else YOLO(str(checkpoint))
    val_predictions, val_gt = run_split(va, args.dataset, "val", model_predictions, verifier, majority_params,
                                        bank_features, bank_patches, encoder, model, args.device, args.batch, args.score_floor)
    del model
    gc.collect()
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    val_rows = []
    for alpha in ALPHAS:
        adjusted = apply_alpha(val_predictions, alpha)
        ap = va.compute_metrics(adjusted, val_gt, float(majority_params["operating_confidence"]))
        val_rows.append({"alpha": alpha, "metrics": ap})
    selected = max(val_rows, key=lambda x: (x["metrics"]["mAP50_95"], x["metrics"]["f1"], -x["alpha"]))
    alpha = float(selected["alpha"])

    # Only after alpha is frozen on validation do we load test predictions,
    # synthesize test counterfactuals, and evaluate test labels once.
    test_model_predictions = {m: va.load_predictions(args.dataset, "test", m) for m in va.MODELS}
    model = RTDETR(str(checkpoint)) if verifier.startswith("rtdetr_") else YOLO(str(checkpoint))
    test_predictions, test_gt = run_split(va, args.dataset, "test", test_model_predictions, verifier, majority_params,
                                          bank_features, bank_patches, encoder, model, args.device, args.batch, args.score_floor)
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    baseline_test = va.compute_metrics(va.fuse_dataset(test_model_predictions, "majority_vote", majority_params,
                                                       {m: 1.0 for m in va.MODELS}), test_gt,
                                       float(majority_params["operating_confidence"]))
    crv_test = va.compute_metrics(apply_alpha(test_predictions, alpha), test_gt,
                                  float(majority_params["operating_confidence"]))
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATA_VERSION,
        "training_condition": "pretrained_aug5x",
        "seed": args.seed,
        "dataset": args.dataset,
        "method": "majority-vote boxes reweighted by median positive same-detector counterfactual response drop times normalized appearance change",
        "normal_reference_source": "original training images only; 64x64 patches with segmentation masks dilated by 12px and lesion occupancy <=1%",
        "top_k_references": TOP_K,
        "verifier_model_selected_on_validation": verifier,
        "majority_vote_params_frozen_from_validation": majority_params,
        "alpha_candidates": list(ALPHAS),
        "validation_alpha_selection": "max mAP50-95, tie-break by F1, then smaller alpha",
        "selected_alpha": alpha,
        "validation_candidates": val_rows,
        "test_baseline_majority_vote": baseline_test,
        "test_crv_reweighted_majority_vote": crv_test,
        "test_delta_mAP50_95": float(crv_test["mAP50_95"] - baseline_test["mAP50_95"]),
        "test_delta_f1": float(crv_test["f1"] - baseline_test["f1"]),
        "test_delta_fp_per_image": float(crv_test["fp_per_image"] - baseline_test["fp_per_image"]),
    }
    out_dir = OUTPUT_ROOT / f"seed{args.seed}" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "crv_vote_pilot.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "pilot_complete", "seed": args.seed, "dataset": args.dataset,
                      "verifier": verifier, "alpha": alpha,
                      "val_map50_95": selected["metrics"]["mAP50_95"],
                      "test_baseline_map50_95": baseline_test["mAP50_95"],
                      "test_crv_map50_95": crv_test["mAP50_95"], "output": str(target)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
