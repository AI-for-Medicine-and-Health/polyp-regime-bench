#!/usr/bin/env python3
"""Select six models on validation; fuse boxes supported by >=3 models at score>0.2.

Validation and test are separate execution phases. All selections are written to
disk before any worker reads test labels or test predictions. Reuses GPU exports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import run_voting_analysis as base

ROOT = Path(__file__).resolve().parents[5]
VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
CONDITION = "pretrained_aug5x"
RUN_ROOT = ROOT / "runs/inference" / VERSION
REPORT_ROOT = ROOT / "reports" / VERSION
CONFIDENCE = 0.20
MIN_MODELS = 3
IOU_GRID = (0.50, 0.60, 0.70, 0.80)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def configure(seed):
    base.DATA_ROOT = ROOT / "datasets/materialized/active" / (VERSION + "__aug5x")
    base.INFER_ROOT = RUN_ROOT / f"consolidated_{CONDITION}_seed{seed}_voting"


def checked_ground_truth(dataset, split):
    for path in base.image_paths(dataset, split):
        directory = base.DATA_ROOT / dataset / split
        if not any((directory / name / (path.stem + ".txt")).is_file()
                   for name in ("labels", "labels_detection")):
            raise FileNotFoundError(f"Missing label: {dataset}/{split}/{path.stem}")
    return base.load_ground_truth(dataset, split)


def load_cached(dataset, split, model, seed, gt):
    path = base.INFER_ROOT / "raw" / dataset / split / (model + ".json")
    raw = path.read_bytes()
    data = json.loads(raw)
    expected = {"dataset": dataset, "split": split, "model": model, "seed": seed,
                "dataset_version": VERSION + "__aug5x", "image_count": len(gt)}
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Cache metadata mismatch {path}: {key}={data.get(key)!r}, expected {value!r}")
    if data["candidate_score_floor"] >= CONFIDENCE:
        raise ValueError(f"Insufficient export score floor: {path}")
    predictions = {image_id: [] for image_id in gt}
    for detection in data["predictions"]:
        box, score = detection["box"], float(detection["score"])
        if not np.isfinite([*box, score]).all() or box[2] < box[0] or box[3] < box[1]:
            raise ValueError(f"Invalid detection: {path}")
        predictions[detection["image_id"]].append({"box": box, "score": score})
    for detections in predictions.values():
        detections.sort(key=lambda row: row["score"], reverse=True)
    provenance = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
                  "checkpoint_sha256": data["checkpoint_sha256"], "export_score_floor": data["candidate_score_floor"]}
    return predictions, provenance


def filter_predictions(predictions):
    # Strictly greater, not >=. No top-k limit for the requested confidence gate.
    return {image_id: [row for row in rows if row["score"] > CONFIDENCE]
            for image_id, rows in predictions.items()}


def fuse_image(by_model, cluster_iou):
    # Preserve the existing model-ordered greedy, maximum-member-IoU clustering.
    # Collapse same-model duplicates before coordinate/score fusion: one vote and
    # one (highest-confidence) representative box per model in each cluster.
    clusters = []
    for model in base.MODELS:
        for row in sorted(by_model.get(model, []), key=lambda r: r["score"], reverse=True):
            if row["score"] <= CONFIDENCE:
                continue
            best_index, best_iou = None, cluster_iou
            for index, cluster in enumerate(clusters):
                overlap = max(base.box_iou(row["box"], item["box"]) for _, item in cluster)
                if overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index is None:
                clusters.append([(model, row)])
            else:
                clusters[best_index].append((model, row))
    output = []
    for cluster in clusters:
        representatives = {}
        for model, row in cluster:
            if model not in representatives or row["score"] > representatives[model]["score"]:
                representatives[model] = row
        if len(representatives) < MIN_MODELS:
            continue
        models = sorted(representatives)
        output.append({
            "box": np.median([representatives[m]["box"] for m in models], axis=0).tolist(),
            "score": float(np.mean([representatives[m]["score"] for m in models])),
            "support": len(models), "supporting_models": models,
            "member_boxes": [representatives[m]["box"] for m in models],
            "member_scores": [representatives[m]["score"] for m in models],
        })
    return sorted(output, key=lambda row: row["score"], reverse=True)


def fuse_dataset(predictions, cluster_iou):
    ids = sorted(next(iter(predictions.values())))
    return {image_id: fuse_image({m: rows[image_id] for m, rows in predictions.items()}, cluster_iou)
            for image_id in ids}


def select_validation(job):
    seed, dataset, output = job
    configure(seed)
    gt = checked_ground_truth(dataset, "val")
    ranking, predictions, provenance = [], {}, {}
    for model in base.MODELS:
        full, provenance[model] = load_cached(dataset, "val", model, seed, gt)
        # Match the old single_best ranking convention (low-floor, top-50 AP).
        ranked = {image_id: rows[:base.MAX_CANDIDATES_PER_MODEL_IMAGE] for image_id, rows in full.items()}
        metrics = base.compute_metrics(ranked, gt, CONFIDENCE)
        ranking.append({"model": model, "val_metrics_for_ranking": metrics})
        predictions[model] = filter_predictions(full)
    ranking.sort(key=lambda r: (-r["val_metrics_for_ranking"]["mAP50_95"],
                                -r["val_metrics_for_ranking"]["recall"], r["model"]))
    selected_models = [row["model"] for row in ranking[:6]]
    chosen_predictions = {m: predictions[m] for m in base.MODELS if m in selected_models}
    candidates = []
    for iou in IOU_GRID:
        fused = fuse_dataset(chosen_predictions, iou)
        candidates.append({"cluster_iou": iou, "val_metrics": base.compute_metrics(fused, gt, CONFIDENCE)})
    selected = max(candidates, key=lambda row: (row["val_metrics"]["mAP50_95"],
                    row["val_metrics"]["f1"], row["val_metrics"]["recall"], -row["cluster_iou"]))
    result = {"seed": seed, "dataset": dataset, "condition": CONDITION,
              "selected_models": selected_models, "single_model": selected_models[0],
              "cluster_iou": selected["cluster_iou"], "input_confidence_strict_gt": CONFIDENCE,
              "min_distinct_models": MIN_MODELS, "model_ranking": ranking,
              "iou_candidates": candidates, "val_fusion_metrics": selected["val_metrics"],
              "val_single_metrics": base.compute_metrics(predictions[selected_models[0]], gt, CONFIDENCE),
              "val_images": len(gt), "val_boxes": sum(map(len, gt.values())), "val_cache_provenance": provenance}
    directory = Path(output) / f"seed{seed}" / dataset
    write_json(directory / "selection.json", result)
    write_json(directory / "val_fused_predictions.json", fuse_dataset(chosen_predictions, result["cluster_iou"]))
    return result


def evaluate_test(job):
    selection, output = job
    seed, dataset = selection["seed"], selection["dataset"]
    configure(seed)
    gt = checked_ground_truth(dataset, "test")
    predictions, provenance = {}, {}
    for model in selection["selected_models"]:
        full, provenance[model] = load_cached(dataset, "test", model, seed, gt)
        if provenance[model]["checkpoint_sha256"] != selection["val_cache_provenance"][model]["checkpoint_sha256"]:
            raise ValueError(f"Validation/test checkpoint mismatch: {seed}/{dataset}/{model}")
        predictions[model] = filter_predictions(full)
    fused = fuse_dataset(predictions, selection["cluster_iou"])
    result = {**selection, "test_images": len(gt), "test_boxes": sum(map(len, gt.values())),
              "test_fusion_metrics": base.compute_metrics(fused, gt, CONFIDENCE),
              "test_single_metrics": base.compute_metrics(predictions[selection["single_model"]], gt, CONFIDENCE),
              "test_cache_provenance": provenance,
              "support_histogram": dict(Counter(row["support"] for rows in fused.values() for row in rows))}
    result["test_delta_vs_single"] = {key: result["test_fusion_metrics"][key] - result["test_single_metrics"][key]
                                      for key in ("mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image")}
    directory = Path(output) / f"seed{seed}" / dataset
    write_json(directory / "test_fused_predictions.json", fused)
    write_json(directory / "result.json", result)
    return result


def build_report(results, output, elapsed):
    def mean_sd(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"
    lines = ["# Aug5x：验证集 Top-6 模型，至少 3 票的硬投票融合", "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}。后处理耗时：{elapsed:.1f} 秒。", "",
        "## 协议", "",
        "- 每个数据集、每个 seed 独立按验证集 mAP50–95 选择 12 个模型中的前 6 个。单模型对照为同一排名第一名。",
        "- 排名沿用旧脚本的低分数导出缓存、每模型每图 top-50 候选和 AP 实现。实际融合与单模型对照保留所有 score > 0.2 的缓存框，不做 top-50 截断。",
        "- 至少 3 个不同模型支持才保留；每模型在同一簇只算一票，并只用该模型最高分框参与融合。",
        "- 框坐标取各模型代表框的逐坐标中位数；输出分数取它们的算术平均。没有额外的输出分数搜索或 NMS。",
        "- 沿用旧脚本的固定模型顺序贪心聚类，候选框与簇内任一成员达到 IoU 阈值即可加入；不要求簇内所有框两两达到该阈值。",
        "- 聚类 IoU 从 0.50/0.60/0.70/0.80 中按验证集 mAP50–95 选择，并列依次比较 F1、Recall、较低 IoU。所有 9 组选择落盘后才加载测试数据。",
        "- P/R/F1 在匹配 IoU=0.50 下计算；mAP50–95 使用 0.50:0.05:0.95 和旧脚本的 101 点插值实现。",
        "- 本表 AP 对两种方法都基于已过滤 score>0.2 的输出，不能直接与旧报告低分数完整候选上的 AP 混比。",
        "- 复用既有 GPU 预测缓存，本轮 CPU 并行后处理，不重新推理或训练。", "",
        "## 测试结果（三个训练 seed 均值 ± 样本标准差）", "",
        "| 数据集 | 方法 | mAP50 | mAP50–95 | Precision | Recall | F1 | FP/image |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    metrics = ("mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image")
    for dataset in base.DATASETS:
        group = [r for r in results if r["dataset"] == dataset]
        for label, key in (("最佳单模型，score>0.2", "test_single_metrics"),
                           ("Top-6 / ≥3票，score>0.2", "test_fusion_metrics")):
            lines.append(f"| {dataset} | {label} | " + " | ".join(mean_sd([r[key][m] for r in group]) for m in metrics) + " |")
    lines += ["", "## 配对差值（融合 − 同 seed 最佳单模型）", "",
              "| 数据集 | Δ mAP50–95 | Δ Precision | Δ Recall | Δ F1 | Δ FP/image |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        group = [r for r in results if r["dataset"] == dataset]
        lines.append(f"| {dataset} | " + " | ".join(mean_sd([r["test_delta_vs_single"][m] for r in group])
                     for m in metrics[1:]) + " |")
    lines += ["", "## 验证集选出的模型与 IoU", "",
              "| 数据集 | Seed | 六模型（按验证集排名） | 聚类 IoU | Val mAP50–95：单模型→融合 |",
              "|---|---:|---|---:|---:|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['seed']} | {', '.join(r['selected_models'])} | {r['cluster_iou']:.2f} | "
                     f"{r['val_single_metrics']['mAP50_95']:.4f} → {r['val_fusion_metrics']['mAP50_95']:.4f} |")
    lines += ["", "## 各 seed 的测试指标", "",
              "| 数据集 | Seed | 方法 | mAP50–95 | Precision | Recall | F1 | TP | FP | FN |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        for label, key in (("single", "test_single_metrics"), ("vote", "test_fusion_metrics")):
            v = r[key]
            lines.append(f"| {r['dataset']} | {r['seed']} | {label} | " + " | ".join(f"{v[m]:.4f}" for m in metrics[1:5])
                         + f" | {v['tp']} | {v['fp']} | {v['fn']} |")
    lines += ["", "## 结论与适用范围", ""]
    for dataset in base.DATASETS:
        group = [r for r in results if r["dataset"] == dataset]
        delta = {m: statistics.mean(r["test_delta_vs_single"][m] for r in group) for m in metrics}
        wins = sum(r["test_delta_vs_single"]["mAP50_95"] > 0 for r in group)
        lines.append(f"- {dataset}：平均 ΔmAP50–95={delta['mAP50_95']:+.4f}，ΔP={delta['precision']:+.4f}，"
                     f"ΔR={delta['recall']:+.4f}，ΔF1={delta['f1']:+.4f}；mAP50–95 在 {wins}/{len(group)} 个 seed 上提高。")
    lines += ["- 三个 seed 是固定数据划分下的训练随机性重复，不构成统计显著性或跨中心普遍性证明。",
              "- 这是用户指定的 3-of-6 共识筛选；3/6 为半数支持，并非严格多数。不同模型的置信度尚未校准，统一阈值不代表等同的置信可靠性。",
              "- 现有原始预测已经过各模型推理框架的导出处理。本实验只分析该缓存中的候选框。", "",
              "## 产物与复现", "",
              f"- 脚本：`{Path(__file__).relative_to(ROOT)}`",
              f"- 本次完整结果目录：`{output.relative_to(ROOT)}`",
              "- `selection_all.json`：测试前冻结的排名、模型组合、IoU、验证集指标及输入缓存 SHA256。",
              "- `results.json` / `summary.csv`：全部测试结果。",
              "- `seed<seed>/<dataset>/test_fused_predictions.json`：融合框、支持数、支持模型、成员框及分数。",
              "- 同目录保存 `selection.json`、`val_fused_predictions.json` 和 `result.json`。", ""]
    return "\n".join(lines)


def self_test():
    a, b, c = base.MODELS[:3]
    box = [0.1, 0.1, 0.3, 0.3]
    d = lambda s: {"box": box, "score": s}
    assert not fuse_image({a: [d(.8), d(.7), d(.6)]}, .5)
    assert not fuse_image({a: [d(.8)], b: [d(.7)], c: [d(.2)]}, .5)
    fused = fuse_image({a: [d(.8), d(.7)], b: [d(.6)], c: [d(.4)]}, .5)
    assert len(fused) == 1 and fused[0]["support"] == 3
    assert len(fused[0]["member_scores"]) == 3 and abs(fused[0]["score"] - .6) < 1e-12
    assert np.allclose(fused[0]["box"], box)
    assert not fuse_image({a: [d(.8)], b: [d(.7)], c: [{"box": [.6, .6, .8, .8], "score": .9}]}, .5)
    assert filter_predictions({"x": [d(.2), d(.20001)]}) == {"x": [d(.20001)]}
    gt = {"x": [box]}
    metrics = base.compute_metrics({"x": fused}, gt, CONFIDENCE)
    assert metrics["tp"] == 1 and metrics["fp"] == 0 and metrics["fn"] == 0 and metrics["mAP50_95"] == 1
    print("PASS: distinct votes, strict confidence, representative fusion, spatial matching, metric sanity", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--output-name", default="top6_hard_vote_conf020_min3_aug5x")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if Path(args.output_name).name != args.output_name or args.workers < 1:
        parser.error("output-name must be a directory name; workers must be positive")
    output = RUN_ROOT / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    selections, results = [], []
    jobs = [(seed, dataset, str(output)) for seed in (88, 123, 666) for dataset in base.DATASETS]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for future in as_completed([pool.submit(select_validation, job) for job in jobs]):
            row = future.result()
            selections.append(row)
            print(json.dumps({"phase": "validation_complete", "seed": row["seed"], "dataset": row["dataset"],
                              "models": row["selected_models"], "iou": row["cluster_iou"]}), flush=True)
        selections.sort(key=lambda r: (r["dataset"], r["seed"]))
        write_json(output / "selection_all.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selections": selections})
        print("ALL SELECTIONS FROZEN; starting test evaluation", flush=True)
        for future in as_completed([pool.submit(evaluate_test, (selection, str(output))) for selection in selections]):
            row = future.result()
            results.append(row)
            print(json.dumps({"phase": "test_complete", "seed": row["seed"], "dataset": row["dataset"],
                              "delta": row["test_delta_vs_single"]}), flush=True)
    results.sort(key=lambda r: (r["dataset"], r["seed"]))
    elapsed = time.monotonic() - started
    write_json(output / "results.json", {"schema_version": 1, "elapsed_seconds": elapsed,
               "script_sha256": base.sha256(Path(__file__)), "evaluator_sha256": base.sha256(Path(base.__file__)),
               "results": results})
    csv_rows = []
    for r in results:
        for method, key in (("single_best_conf_gt_020", "test_single_metrics"), ("top6_hard_vote_3", "test_fusion_metrics")):
            csv_rows.append({"dataset": r["dataset"], "seed": r["seed"], "method": method,
                             "cluster_iou": r["cluster_iou"] if method == "top6_hard_vote_3" else "",
                             "models": ";".join(r["selected_models"]) if method == "top6_hard_vote_3" else r["single_model"], **r[key]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = REPORT_ROOT / (args.output_name + ".md")
    with report.open("x", encoding="utf-8") as handle:
        handle.write(build_report(results, output, elapsed))
    print(json.dumps({"phase": "done", "elapsed_seconds": elapsed, "report": str(report), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
