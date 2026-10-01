#!/usr/bin/env python3
"""Validation-selected IoU for 12-model hard voting at 5/6/7 votes.

Each validation configuration has its own process. Test predictions and labels
are opened only after all validation choices have been persisted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import run_top6_hard_vote_all_cached_aug5x as fusion

prior = fusion.previous
base = fusion.base
ROOT = prior.ROOT
NAME = "top12_hard_vote_min5_6_7_aug5x"
OUT = prior.RUN_ROOT / NAME
REPORT = prior.REPORT_ROOT / (NAME + ".md")
TOP6_010 = prior.RUN_ROOT / "top6_hard_vote_conf010_min3_aug5x" / "results.json"
TOP6_020 = prior.RUN_ROOT / "top6_hard_vote_conf020_min3_aug5x" / "results.json"
SEEDS = (88, 123, 666)
VOTES = (5, 6, 7)
CONFIDENCES = (.1, .2)
IOUS = (.3, .4, .5, .6, .7, .8)
METRICS = ("mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split(seed, dataset, split):
    prior.configure(seed)
    gt = prior.checked_ground_truth(dataset, split)
    predictions, provenance = {}, {}
    for model in base.MODELS:
        predictions[model], provenance[model] = prior.load_cached(dataset, split, model, seed, gt)
    return gt, predictions, provenance


def fuse_vote(predictions, cluster_iou, votes):
    # The existing vectorized fusion emits all clusters with >=3 supporters.
    # Apply the requested >=5/6/7 decision to the exact same clusters.
    clusters = fusion.fuse_dataset(predictions, cluster_iou)
    return {image_id: [row for row in rows if row["support"] >= votes]
            for image_id, rows in clusters.items()}


def run_job(job):
    phase, seed, dataset, confidence, iou, votes, expected, out = job
    gt, predictions, provenance = load_split(seed, dataset, phase)
    for model in base.MODELS:
        if model not in expected or provenance[model]["sha256"] != expected[model]["sha256"]:
            raise ValueError(f"Unexpected cache for {seed}/{dataset}/{phase}/{model}")
    filtered = {model: {image_id: [row for row in rows if row["score"] > confidence]
                        for image_id, rows in by_image.items()}
                for model, by_image in predictions.items()}
    fused = fuse_vote(filtered, iou, votes)
    metrics = base.compute_metrics(fused, gt, 0.0)
    result = {"phase": phase, "seed": seed, "dataset": dataset, "confidence": confidence,
              "cluster_iou": iou, "min_votes": votes, "process_id": os.getpid(),
              "images": len(gt), "gt_boxes": sum(map(len, gt.values())),
              "fused_boxes": sum(map(len, fused.values())), **metrics}
    if phase == "test":
        directory = Path(out) / f"seed{seed}" / dataset
        prior.write_json(directory / f"conf{int(confidence*100):02d}_min{votes}_test_predictions.json", fused)
    return result


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["seed"], row["confidence"], row["min_votes"])].append(row)
    selected = []
    for key, group in sorted(groups.items()):
        if len(group) != len(IOUS):
            raise AssertionError(f"Incomplete validation IoU grid: {key}")
        # One frozen IoU per confidence/vote variant: AP, then F1, recall, lower IoU.
        best = max(group, key=lambda r: (r["mAP50_95"], r["f1"], r["recall"], -r["cluster_iou"]))
        selected.append(best)
    if len(selected) != len(SEEDS) * len(base.DATASETS) * len(CONFIDENCES) * len(VOTES):
        raise AssertionError("Wrong number of frozen selections")
    return selected


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row_index, conf in enumerate(CONFIDENCES):
        for ax, dataset in zip(axes[row_index], base.DATASETS):
            values = np.asarray([[statistics.mean(r["mAP50_95"] for r in rows
                         if r["dataset"] == dataset and r["confidence"] == conf
                         and r["min_votes"] == votes and r["cluster_iou"] == iou)
                         for iou in IOUS] for votes in VOTES])
            im = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap="viridis")
            ax.set_title(f"{dataset}, score > {conf}")
            ax.set_xticks(range(len(IOUS)), [f"{x:.1f}" for x in IOUS])
            ax.set_yticks(range(len(VOTES)), VOTES)
            ax.set_xlabel("Clustering IoU")
            ax.set_ylabel("Minimum distinct votes")
            for y in range(len(VOTES)):
                for x in range(len(IOUS)):
                    ax.text(x, y, f"{values[y,x]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if values[y,x] < .55 else "black")
    fig.colorbar(im, ax=axes, label="Validation mAP50–95 (three-seed mean)", shrink=.7)
    figure = OUT / "validation_map_heatmap.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)


def make_report(selected, tested, reference_010, reference_020, elapsed, workers):
    def fmt(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"

    def metric_row(dataset, label, group):
        return f"| {dataset} | {label} | " + " | ".join(fmt([r[m] for r in group]) for m in METRICS) + " |"

    by_test = {(r["dataset"], r["seed"], r["confidence"], r["min_votes"]): r for r in tested}
    lines = ["# 五倍增强：12 模型硬投票，至少 5/6/7 票", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}；验证 {len(SEEDS)*len(base.DATASETS)*len(CONFIDENCES)*len(VOTES)*len(IOUS)} 个独立进程，最多同时 {workers} 个；总耗时 {elapsed:.1f} 秒。", "",
             "## 实验协议", "",
             "- 每个数据集、训练 seed 使用全部 12 个预训练 + 5× 增强模型。每个模型每簇最多一票；同模型只保留最高分代表框，中位数融合坐标，代表框分数取平均。",
             "- 分别要求至少 5、6、7 个不同模型支持；6/12 是半数支持，7/12 才是严格多数。输入仅保留 score > 0.1 或 > 0.2 的缓存框，输出不再按分数筛选。",
             "- 聚类 IoU 网格为 0.3–0.8，步长 0.1。对每个置信度和最低票数组合，按验证集 mAP50–95 选 IoU；并列依次比 F1、Recall、较低 IoU。所有选择冻结落盘后才加载测试集。",
             "- 每个‘数据集 × seed × 输入阈值 × 最低票数 × IoU’配置独立进程。P/R/F1 使用 IoU=0.50 匹配和全部融合框；AP 使用 0.50:0.05:0.95 的 101 点插值。",
             "- 复用原有 GPU 推理缓存，本轮只做 CPU 后处理。缓存的导出分数下限为 0.001，原模型 NMS、max_det 等限制仍然存在。",
             "- 对照为此前的 Top-6/≥3 票固定 >0.1、>0.2 结果，以及同一验证集排名第一的单模型；Top-6 的 IoU 是此前验证集选定的，和本轮 12 模型分别调优，属于方法级而非单一变量比较。", "",
             "## 验证集参数曲面", "",
             f"![验证集 mAP50–95 热力图](../../runs/inference/{prior.VERSION}/{NAME}/validation_map_heatmap.png)", "",
             "## 测试集结果（三训练 seed 均值 ± 标准差）", "",
             "| 数据集 | 方法 | mAP50 | mAP50–95 | Precision | Recall | F1 | FP/图 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        old010 = [r for r in reference_010 if r["dataset"] == dataset]
        old020 = [r for r in reference_020 if r["dataset"] == dataset]
        lines.append(metric_row(dataset, "最佳单模型 >0.1", [r["splits"]["test"]["single_best_conf010"] for r in old010]))
        lines.append(metric_row(dataset, "Top-6/≥3票 >0.1", [r["splits"]["test"]["hard_vote_conf010"] for r in old010]))
        lines.append(metric_row(dataset, "最佳单模型 >0.2", [r["test_single_metrics"] for r in old020]))
        lines.append(metric_row(dataset, "Top-6/≥3票 >0.2", [r["test_fusion_metrics"] for r in old020]))
        for confidence in CONFIDENCES:
            for votes in VOTES:
                group = [r for r in tested if r["dataset"] == dataset and r["confidence"] == confidence and r["min_votes"] == votes]
                lines.append(metric_row(dataset, f"12模型/≥{votes}票 >{confidence}", group))
    lines += ["", "## 核心对照：12 模型 ≥6 票 − Top-6 ≥3 票（同输入阈值，逐 seed 配对均值）", "",
              "| 数据集 | 输入阈值 | ΔmAP50–95 | ΔPrecision | ΔRecall | ΔF1 | ΔFP/图 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        for confidence, reference in ((.1, reference_010), (.2, reference_020)):
            differences = []
            for seed in SEEDS:
                new = by_test[(dataset, seed, confidence, 6)]
                old_row = next(r for r in reference if r["dataset"] == dataset and r["seed"] == seed)
                old = (old_row["splits"]["test"]["hard_vote_conf010"] if confidence == .1
                       else old_row["test_fusion_metrics"])
                differences.append({m: new[m] - old[m] for m in METRICS})
            lines.append(f"| {dataset} | >{confidence:.1f} | " + " | ".join(
                f"{statistics.mean(r[m] for r in differences):+.4f}" for m in METRICS[1:]) + " |")
    lines += ["", "12 模型 ≥6 票在三个数据集、两个输入阈值下均提高 Precision 和 F1、减少 FP/图；同时均降低 Recall 和 mAP50–95。若主目标是检测 AP，这一方案不优于现有 Top-6/≥3 票；若重视少误报和工作点 F1，则值得保留为备选。", "",
              "## 验证集选出的 IoU 与逐 seed 测试结果", "",
              "| 数据集 | Seed | 输入阈值 | 最低票数 | IoU | Val mAP50–95 | Test mAP50–95 | Test P/R/F1 | Test FP/图 |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in selected:
        test = by_test[(row["dataset"], row["seed"], row["confidence"], row["min_votes"])]
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['confidence']:.1f} | {row['min_votes']} | "
                     f"{row['cluster_iou']:.1f} | {row['mAP50_95']:.4f} | {test['mAP50_95']:.4f} | "
                     f"{test['precision']:.3f}/{test['recall']:.3f}/{test['f1']:.3f} | {test['fp_per_image']:.3f} |")
    lines += ["", "## 解释与限制", "",
              "- 增加票数可能减少假阳性，也可能丢掉难例；表中多个工作点是探索性比较，不能从测试集挑最好一行作为无偏最终性能。",
              "- 12 个模型的错误存在相关性；投票数不能解释为独立证据数。贪心聚类只要求候选框与簇中某个成员满足 IoU，并非簇内两两匹配。",
              "- 三 seed 是固定划分上的训练重复，不是三个独立中心。该测试集已在之前实验中反复查看，论文定论需要新的独立留出中心。", "",
              "## 产物", "",
              f"- 脚本：`{Path(__file__).relative_to(ROOT)}`",
              f"- 数据目录：`{OUT.relative_to(ROOT)}`；含 `validation_grid.csv`、`frozen_selections.json`、`test_selected.csv`、逐图融合预测和输入缓存校验信息。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.workers <= 100:
        parser.error("workers must be 1..100")
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Freeze cache digests and model roster before grid execution. These reads
    # are validation-only; test caches and labels are not touched here.
    val_sources = {}
    for seed in SEEDS:
        for dataset in base.DATASETS:
            _, _, provenance = load_split(seed, dataset, "val")
            val_sources[(seed, dataset)] = provenance
    prior.write_json(OUT / "protocol.json", {"models": base.MODELS, "seeds": SEEDS,
        "datasets": base.DATASETS, "min_votes": VOTES, "confidence_strict_gt": CONFIDENCES,
        "cluster_ious": IOUS, "max_workers": args.workers,
        "validation_cache_provenance": [
            {"seed": seed, "dataset": dataset, "models": val_sources[(seed, dataset)]}
            for seed in SEEDS for dataset in base.DATASETS]})
    jobs = [("val", seed, dataset, confidence, iou, votes, val_sources[(seed, dataset)], str(OUT))
            for seed in SEEDS for dataset in base.DATASETS for confidence in CONFIDENCES
            for votes in VOTES for iou in IOUS]
    rows = []
    with mp.get_context("fork").Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_job, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 18 == 0:
                print(json.dumps({"phase": "validation", "done": len(rows), "total": len(jobs),
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    if len(rows) != len(jobs) or len({r["process_id"] for r in rows}) != len(rows):
        raise AssertionError("Each validation configuration needs a separate process")
    rows.sort(key=lambda r: (r["dataset"], r["seed"], r["confidence"], r["min_votes"], r["cluster_iou"]))
    write_csv(OUT / "validation_grid.csv", rows)
    prior.write_json(OUT / "validation_grid.json", rows)
    selected = select(rows)
    prior.write_json(OUT / "frozen_selections.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": selected})
    print("VALIDATION CHOICES FROZEN; starting test cache reads", flush=True)
    test_sources = {}
    for seed in SEEDS:
        for dataset in base.DATASETS:
            _, _, provenance = load_split(seed, dataset, "test")
            for model in base.MODELS:
                if provenance[model]["checkpoint_sha256"] != val_sources[(seed, dataset)][model]["checkpoint_sha256"]:
                    raise ValueError(f"Val/test checkpoint mismatch: {seed}/{dataset}/{model}")
            test_sources[(seed, dataset)] = provenance
    prior.write_json(OUT / "test_cache_provenance.json", [
        {"seed": seed, "dataset": dataset, "models": test_sources[(seed, dataset)]}
        for seed in SEEDS for dataset in base.DATASETS])
    test_jobs = [("test", r["seed"], r["dataset"], r["confidence"], r["cluster_iou"],
                  r["min_votes"], test_sources[(r["seed"], r["dataset"])], str(OUT)) for r in selected]
    tested = []
    with mp.get_context("fork").Pool(processes=min(args.workers, len(test_jobs)), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_job, test_jobs, chunksize=1):
            tested.append(row)
            if len(tested) % 9 == 0:
                print(json.dumps({"phase": "test", "done": len(tested), "total": len(test_jobs)}), flush=True)
    tested.sort(key=lambda r: (r["dataset"], r["seed"], r["confidence"], r["min_votes"]))
    prior.write_json(OUT / "test_selected.json", tested)
    write_csv(OUT / "test_selected.csv", tested)
    make_figure(rows)
    old010 = json.loads(TOP6_010.read_text())["results"]
    old020 = json.loads(TOP6_020.read_text())["results"]
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(make_report(selected, tested, old010, old020, time.monotonic()-started, args.workers))
    print(json.dumps({"phase": "done", "report": str(REPORT),
                      "seconds": round(time.monotonic()-started, 1)}), flush=True)


if __name__ == "__main__":
    main()
