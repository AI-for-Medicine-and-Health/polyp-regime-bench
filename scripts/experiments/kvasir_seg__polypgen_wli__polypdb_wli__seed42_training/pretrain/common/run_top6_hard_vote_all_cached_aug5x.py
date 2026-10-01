#!/usr/bin/env python3
"""Remove the 0.2 score gate, using ALL cached boxes and frozen top-six/IoU.

This removes post-export confidence constraints, not the original detector's
export floor/max_det/NMS. Nine workers run three seeds times three datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import run_top6_hard_vote_aug5x as previous

base = previous.base
SOURCE = previous.RUN_ROOT / "top6_hard_vote_conf020_min3_aug5x"
OUTPUT_NAME = "top6_hard_vote_all_cached_min3_aug5x"
METRICS = ("mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image")


def fuse_image(by_model, threshold):
    """Exact previous greedy clustering, accelerated using pairwise NumPy IoU.

All input boxes participate. Model order, within-model score order, last-cluster
tie-breaking, distinct votes and representative selection match the old method.
"""
    members = [(model, row) for model in base.MODELS
               for row in sorted(by_model.get(model, []), key=lambda r: r["score"], reverse=True)]
    if not members:
        return []
    boxes = np.asarray([row["box"] for _, row in members], dtype=np.float64)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    ix = np.maximum(0, np.minimum(boxes[:, None, 2], boxes[None, :, 2])
                    - np.maximum(boxes[:, None, 0], boxes[None, :, 0]))
    iy = np.maximum(0, np.minimum(boxes[:, None, 3], boxes[None, :, 3])
                    - np.maximum(boxes[:, None, 1], boxes[None, :, 1]))
    intersection = ix * iy
    union = area[:, None] + area[None, :] - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    assignments = np.empty(len(members), dtype=np.int64)
    representatives = []
    for index, (model, row) in enumerate(members):
        best_overlap = np.full(len(representatives), -1.0)
        np.maximum.at(best_overlap, assignments[:index], overlap[index, :index])
        best = float(np.max(best_overlap)) if best_overlap.size else -1.0
        if best >= threshold:
            cluster_index = int(np.flatnonzero(best_overlap == best)[-1])
        else:
            cluster_index = len(representatives)
            representatives.append({})
        assignments[index] = cluster_index
        cluster = representatives[cluster_index]
        if model not in cluster or row["score"] > cluster[model]["score"]:
            cluster[model] = row
    fused = []
    for cluster in representatives:
        if len(cluster) < 3:
            continue
        models = sorted(cluster)
        fused.append({"box": np.median([cluster[m]["box"] for m in models], axis=0).tolist(),
                      "score": float(np.mean([cluster[m]["score"] for m in models])),
                      "support": len(models), "supporting_models": models,
                      "member_boxes": [cluster[m]["box"] for m in models],
                      "member_scores": [cluster[m]["score"] for m in models]})
    return sorted(fused, key=lambda row: row["score"], reverse=True)


def fuse_dataset(predictions, iou):
    return {image_id: fuse_image({m: rows[image_id] for m, rows in predictions.items()}, iou)
            for image_id in sorted(next(iter(predictions.values())))}


def worker(job):
    old, output = job
    seed, dataset = old["seed"], old["dataset"]
    previous.configure(seed)
    directory = Path(output) / f"seed{seed}" / dataset
    result = {"seed": seed, "dataset": dataset, "condition": previous.CONDITION,
              "selected_models": old["selected_models"], "single_model": old["single_model"],
              "cluster_iou": old["cluster_iou"], "min_distinct_models": 3,
              "additional_input_score_filter": None, "additional_output_score_filter": None,
              "additional_top_k": None, "splits": {}}
    for split in ("val", "test"):
        gt = previous.checked_ground_truth(dataset, split)
        predictions, provenance = {}, {}
        for model in old["selected_models"]:
            predictions[model], provenance[model] = previous.load_cached(dataset, split, model, seed, gt)
            expected = old[f"{split}_cache_provenance"][model]
            if provenance[model]["sha256"] != expected["sha256"]:
                raise ValueError(f"Cache changed since 0.2 experiment: {seed}/{dataset}/{split}/{model}")
        # Reproduce the old gated result, verifying the vectorized implementation.
        gated = {m: previous.filter_predictions(rows) for m, rows in predictions.items()}
        gated_fused = fuse_dataset(gated, old["cluster_iou"])
        gated_metrics = base.compute_metrics(gated_fused, gt, 0.0)
        for key, value in old[f"{split}_fusion_metrics"].items():
            if abs(gated_metrics[key] - value) > 1e-12:
                raise AssertionError(f"0.2 baseline mismatch: {seed}/{dataset}/{split}/{key}")
        all_fused = fuse_dataset(predictions, old["cluster_iou"])
        metrics = base.compute_metrics(all_fused, gt, 0.0)
        single = base.compute_metrics(predictions[old["single_model"]], gt, 0.0)
        result["splits"][split] = {
            "images": len(gt), "ground_truth_boxes": sum(map(len, gt.values())),
            "old_conf020_fusion_metrics": gated_metrics, "all_cached_fusion_metrics": metrics,
            "all_cached_single_metrics": single,
            "delta_vs_conf020": {key: metrics[key] - gated_metrics[key] for key in METRICS},
            "input_boxes_by_model": {m: sum(map(len, rows.values())) for m, rows in predictions.items()},
            "input_boxes_conf020_by_model": {m: sum(map(len, rows.values())) for m, rows in gated.items()},
            "fused_boxes": sum(map(len, all_fused.values())),
            "support_histogram": dict(Counter(row["support"] for rows in all_fused.values() for row in rows)),
            "cache_provenance": provenance,
        }
        previous.write_json(directory / (split + "_fused_predictions.json"), all_fused)
    previous.write_json(directory / "result.json", result)
    return result


def report(results, output, elapsed):
    def average_sd(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
    lines = ["# Aug5x：Top-6 / 至少 3 票，取消额外置信度筛选", "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}；9 个并行进程；后处理耗时 {elapsed:.1f} 秒。", "",
        "## 实验协议", "",
        "- 固定上轮验证集选出的六模型和聚类 IoU；唯一实验变量是取消输入 score>0.2 筛选。",
        "- 使用每个模型的全部缓存框，不作 top-50 或其他 top-k 截断；融合后不加置信度门槛，P/R/F1 统计全部融合输出。",
        "- 缓存原始导出阈值为 0.001，仍受原推理的 max_det/模型查询数/NMS 等约束。因此本实验的‘全部’是全部导出缓存框，不是网络所有未后处理候选。",
        "- 至少 3 个不同模型支持；每模型最高分代表框参与中位坐标融合，分数取代表框分数均值，聚类方式与上轮完全一致。",
        "- 向量化 IoU 加速后的实现已逐组复算 score>0.2 的原方案，验证/测试各项指标均与原结果一致。",
        "- 前六模型沿用先前验证集低分数、top-50 的 AP 排名，不因本次观察测试结果而更换。",
        "- 不重训、不重推 GPU；原始缓存 SHA256 与上轮逐个核对一致。", "",
        "## 测试集结果（三 seed 均值 ± 样本标准差）", "",
        "| 数据集 | 方法 | mAP50 | mAP50–95 | Precision | Recall | F1 | FP/image |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    methods = (("上轮：score>0.2", "old_conf020_fusion_metrics"),
               ("本轮：全部缓存框", "all_cached_fusion_metrics"),
               ("最佳单模型：全部缓存框", "all_cached_single_metrics"))
    for dataset in base.DATASETS:
        group = [r["splits"]["test"] for r in results if r["dataset"] == dataset]
        for label, key in methods:
            lines.append(f"| {dataset} | {label} | " + " | ".join(average_sd([r[key][m] for r in group]) for m in METRICS) + " |")
    lines += ["", "## 各 seed：取消阈值前后", "",
              "| 数据集 | Seed | IoU | Recall 前→后 | Precision 前→后 | mAP50–95 前→后 | FP 前→后 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a, b = r["splits"]["test"]["old_conf020_fusion_metrics"], r["splits"]["test"]["all_cached_fusion_metrics"]
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['cluster_iou']:.2f} | "
                     + " | ".join(f"{a[m]:.4f} → {b[m]:.4f}" for m in ("recall", "precision", "mAP50_95"))
                     + f" | {a['fp']} → {b['fp']} |")
    lines += ["", "## 结论与限制", ""]
    for dataset in base.DATASETS:
        group = [r["splits"]["test"]["delta_vs_conf020"] for r in results if r["dataset"] == dataset]
        lines.append(f"- {dataset}：ΔRecall={statistics.mean(r['recall'] for r in group):+.4f}，"
                     f"ΔPrecision={statistics.mean(r['precision'] for r in group):+.4f}，"
                     f"ΔF1={statistics.mean(r['f1'] for r in group):+.4f}，"
                     f"ΔmAP50–95={statistics.mean(r['mAP50_95'] for r in group):+.4f}。")
    lines += ["- 放入低分框会改变贪心聚类归属和融合坐标，旧融合输出不一定是新输出的子集；因此 Recall 并非数学上保证单调增加。",
              "- 与上轮比较的是同一算法在不同输入门槛下的实际输出；AP 变化包含候选召回、排序和定位的共同影响。",
              "- 不根据测试结果调 IoU、模型组合或阈值。三个 seed 为固定划分上的训练重复。", "",
              "## 文件", "",
              f"- 脚本：`{Path(__file__).relative_to(previous.ROOT)}`",
              f"- 输出目录：`{output.relative_to(previous.ROOT)}`",
              "- `frozen_config.json` 保存上轮冻结的模型、IoU 与完整参照；`results.json`、`summary.csv` 保存指标。",
              "- `seed<seed>/<dataset>/val_fused_predictions.json` 和 `test_fused_predictions.json` 保存全部融合框及支持模型。", ""]
    return "\n".join(lines)


def self_test():
    # Differential tests against the original scalar implementation with its
    # score gate disabled; include zero-score, duplicate and spatially distinct boxes.
    old_threshold = previous.CONFIDENCE
    previous.CONFIDENCE = -float("inf")
    rng = np.random.default_rng(42)
    try:
        for trial in range(12):
            data = {}
            for model in base.MODELS[:6]:
                rows = []
                for _ in range(18):
                    xy = rng.uniform(0, .65, 2)
                    wh = rng.uniform(.05, .35, 2)
                    rows.append({"box": [*xy, *(xy + wh)], "score": float(rng.choice([0., .001, .2, .5, .9]))})
                rows += [{"box": [.1, .1, .3, .3], "score": .001}] * 2
                data[model] = rows
            for threshold in (.3, .5, .6, .7, .8):
                assert fuse_image(data, threshold) == previous.fuse_image(data, threshold)
        empty = {m: [] for m in base.MODELS[:6]}
        assert fuse_image(empty, .5) == []
        data = {m: [{"box": [.1, .1, .3, .3], "score": 0.0}] for m in base.MODELS[:3]}
        assert fuse_image(data, .5)[0]["support"] == 3
    finally:
        previous.CONFIDENCE = old_threshold
    print("PASS: 60 differential clustering checks, empty input and zero-score voting", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.workers < 1 or Path(args.output_name).name != args.output_name:
        parser.error("workers must be positive; output-name must be a directory name")
    started = time.monotonic()
    original = (SOURCE / "results.json").read_bytes()
    reference = json.loads(original)["results"]
    if len(reference) != 9:
        raise ValueError("Expected 9 baseline groups")
    output = previous.RUN_ROOT / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    previous.write_json(output / "frozen_config.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(original).hexdigest(), "reference_results": reference,
        "workers": min(args.workers, 9), "confidence_filter": None, "top_k": None})
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 9)) as pool:
        futures = [pool.submit(worker, (r, str(output))) for r in reference]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "complete", "seed": result["seed"], "dataset": result["dataset"],
                              "test_delta": result["splits"]["test"]["delta_vs_conf020"]}), flush=True)
    results.sort(key=lambda r: (r["dataset"], r["seed"]))
    elapsed = time.monotonic() - started
    previous.write_json(output / "results.json", {"elapsed_seconds": elapsed,
        "script_sha256": base.sha256(Path(__file__)), "results": results})
    rows = []
    for r in results:
        for split in ("val", "test"):
            for key in ("old_conf020_fusion_metrics", "all_cached_fusion_metrics", "all_cached_single_metrics"):
                rows.append({"seed": r["seed"], "dataset": r["dataset"], "split": split, "method": key,
                             **r["splits"][split][key]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report_path = previous.REPORT_ROOT / (args.output_name + ".md")
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report(results, output, elapsed))
    print(json.dumps({"event": "all_done", "seconds": elapsed, "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
