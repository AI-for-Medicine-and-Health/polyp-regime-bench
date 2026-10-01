#!/usr/bin/env python3
"""Evaluate the frozen top-six, three-vote aug5x ensemble at input score >0.1."""
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

import run_top6_hard_vote_all_cached_aug5x as full

prior = full.previous
base = full.base
THRESHOLD = 0.10
REFERENCE = full.SOURCE / "results.json"
OUTPUT_NAME = "top6_hard_vote_conf010_min3_aug5x"


def keep_above(predictions, confidence):
    return {image_id: [box for box in boxes if box["score"] > confidence]
            for image_id, boxes in predictions.items()}


def worker(job):
    frozen, directory = job
    seed, dataset = frozen["seed"], frozen["dataset"]
    prior.configure(seed)
    location = Path(directory) / f"seed{seed}" / dataset
    result = {"seed": seed, "dataset": dataset, "condition": "pretrained_aug5x",
              "selected_models": frozen["selected_models"], "single_model": frozen["single_model"],
              "cluster_iou": frozen["cluster_iou"], "input_score_strict_gt": THRESHOLD,
              "min_distinct_models": 3, "output_score_gate": None, "splits": {}}
    for split in ("val", "test"):
        gt = prior.checked_ground_truth(dataset, split)
        cached, provenance = {}, {}
        for model in frozen["selected_models"]:
            cached[model], provenance[model] = prior.load_cached(dataset, split, model, seed, gt)
            if provenance[model]["sha256"] != frozen[f"{split}_cache_provenance"][model]["sha256"]:
                raise ValueError(f"Raw prediction cache changed: {seed}/{dataset}/{split}/{model}")
        gated_020 = {model: keep_above(rows, 0.20) for model, rows in cached.items()}
        metrics_020 = base.compute_metrics(full.fuse_dataset(gated_020, frozen["cluster_iou"]), gt, 0.0)
        for metric, expected in frozen[f"{split}_fusion_metrics"].items():
            if abs(metrics_020[metric] - expected) > 1e-12:
                raise AssertionError(f"0.2 baseline mismatch: {seed}/{dataset}/{split}/{metric}")
        gated_010 = {model: keep_above(rows, THRESHOLD) for model, rows in cached.items()}
        fused = full.fuse_dataset(gated_010, frozen["cluster_iou"])
        metrics_010 = base.compute_metrics(fused, gt, 0.0)
        single_010 = base.compute_metrics(gated_010[frozen["single_model"]], gt, 0.0)
        result["splits"][split] = {
            "images": len(gt), "ground_truth_boxes": sum(map(len, gt.values())),
            "hard_vote_conf010": metrics_010, "hard_vote_conf020": metrics_020,
            "single_best_conf010": single_010,
            "delta_010_minus_020": {m: metrics_010[m] - metrics_020[m] for m in full.METRICS},
            "input_box_counts": {m: sum(map(len, rows.values())) for m, rows in gated_010.items()},
            "fused_box_count": sum(map(len, fused.values())),
            "support_histogram": dict(Counter(p["support"] for boxes in fused.values() for p in boxes)),
            "raw_cache_provenance": provenance,
        }
        prior.write_json(location / f"{split}_fused_predictions.json", fused)
    prior.write_json(location / "result.json", result)
    return result


def make_report(results, output, elapsed):
    def mean_sd(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
    lines = ["# Aug5x：六模型、至少三票，输入置信度严格大于 0.1", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}；9 个并行进程；耗时 {elapsed:.1f} 秒。", "",
             "固定原实验在验证集选出的六模型与聚类 IoU。每个模型只保留 `score > 0.1` 的缓存框；至少 3 个不同模型支持后取代表框坐标中位数、分数均值。融合后不再筛选。",
             "使用既有 GPU 推理缓存做 CPU 后处理，没有重新推理。缓存本身的导出下限是 0.001。",
             "冻结的模型组合和 IoU 来自最初 score>0.2 的验证实验；本轮 0.1 由用户预先指定，不按测试结果搜索。",
             "mAP50–95 按当前条件输出框的完整分数排序计算；P/R/F1 按全部输出框、匹配 IoU=0.5 计算。", "",
             "## 测试指标（3 seed 均值 ± 样本标准差）", "",
             "| 数据集 | 条件 | mAP50–95 | Precision | Recall | F1 | FP/image |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        groups = [r["splits"]["test"] for r in results if r["dataset"] == dataset]
        for label, key in (("三票融合 >0.2", "hard_vote_conf020"), ("三票融合 >0.1", "hard_vote_conf010"),
                           ("最佳单模型 >0.1", "single_best_conf010")):
            lines.append(f"| {dataset} | {label} | " + " | ".join(mean_sd([r[key][m] for r in groups])
                         for m in ("mAP50_95", "precision", "recall", "f1", "fp_per_image")) + " |")
    lines += ["", "## 0.1 相对 0.2 的测试差值", "",
              "| 数据集 | ΔmAP50–95 | ΔPrecision | ΔRecall | ΔF1 | ΔFP/image |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        groups = [r["splits"]["test"]["delta_010_minus_020"] for r in results if r["dataset"] == dataset]
        lines.append(f"| {dataset} | " + " | ".join(mean_sd([r[m] for r in groups])
                     for m in ("mAP50_95", "precision", "recall", "f1", "fp_per_image")) + " |")
    lines += ["", "## 各 seed 的实际测试值", "",
              "| 数据集 | seed | IoU | P：0.2→0.1 | R：0.2→0.1 | F1：0.2→0.1 | FP：0.2→0.1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        old, new = (r["splits"]["test"][key] for key in ("hard_vote_conf020", "hard_vote_conf010"))
        lines.append(f"| {r['dataset']} | {r['seed']} | {r['cluster_iou']:.2f} | "
                     + " | ".join(f"{old[m]:.4f}→{new[m]:.4f}" for m in ("precision", "recall", "f1"))
                     + f" | {old['fp']}→{new['fp']} |")
    lines += ["", "## 限制和复现", "",
              "- 原始缓存仍由模型推理时的导出分数和后处理限制；本轮仅改变额外的输入阈值。",
              "- 同一训练 seed 的 0.2 基线由本轮复算，验证集与测试集各指标均与原报告完全一致。",
              "- 3 个 seed 共用固定数据划分；不能据此宣称统计显著或跨中心普遍性。",
              f"- 脚本：`{Path(__file__).relative_to(prior.ROOT)}`",
              f"- 完整结果：`{output.relative_to(prior.ROOT)}`；含 `frozen_config.json`、`results.json`、`summary.csv` 和逐图融合框。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    args = parser.parse_args()
    if args.workers < 1 or Path(args.output_name).name != args.output_name:
        parser.error("workers must be positive and output-name a single directory name")
    raw = REFERENCE.read_bytes()
    frozen = json.loads(raw)["results"]
    if len(frozen) != 9:
        raise ValueError("Expected nine frozen seed/dataset groups")
    output = prior.RUN_ROOT / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    prior.write_json(output / "frozen_config.json", {
        "reference": str(REFERENCE.relative_to(prior.ROOT)),
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_models_and_iou": [{k: r[k] for k in ("seed", "dataset", "selected_models", "cluster_iou")}
                                  for r in frozen],
        "new_input_threshold_strict_gt": THRESHOLD, "min_distinct_models": 3})
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 9)) as pool:
        for future in as_completed([pool.submit(worker, (r, str(output))) for r in frozen]):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "finished", "seed": result["seed"], "dataset": result["dataset"],
                              "delta": result["splits"]["test"]["delta_010_minus_020"]}), flush=True)
    results.sort(key=lambda r: (r["dataset"], r["seed"]))
    elapsed = time.monotonic() - started
    prior.write_json(output / "results.json", {"script_sha256": base.sha256(Path(__file__)),
        "elapsed_seconds": elapsed, "results": results})
    rows = []
    for result in results:
        for split in ("val", "test"):
            for method in ("hard_vote_conf020", "hard_vote_conf010", "single_best_conf010"):
                rows.append({"dataset": result["dataset"], "seed": result["seed"],
                             "split": split, "method": method, **result["splits"][split][method]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report_path = prior.REPORT_ROOT / f"{args.output_name}.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(make_report(results, output, elapsed))
    print(json.dumps({"event": "done", "elapsed_seconds": elapsed, "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
