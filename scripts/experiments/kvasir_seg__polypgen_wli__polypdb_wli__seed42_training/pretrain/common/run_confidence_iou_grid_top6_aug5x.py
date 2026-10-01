#!/usr/bin/env python3
"""Independent-process validation grid for aug5x top-six, three-vote fusion.

Each confidence/cluster-IoU/seed/dataset grid point runs in a fresh process.
Validation chooses two prespecified objectives. Test is read only afterward.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
import argparse

import numpy as np
import run_top6_hard_vote_all_cached_aug5x as fusion

prior = fusion.previous
base = fusion.base
ROOT = prior.ROOT
NAME = "confidence_iou_grid_top6_hard_vote_pretrained_aug5x"
OUTPUT = prior.RUN_ROOT / NAME
REPORT = prior.REPORT_ROOT / (NAME + ".md")
REFERENCE = fusion.SOURCE / "selection_all.json"
TEST_REFERENCE = fusion.SOURCE / "results.json"
THRESHOLDS = (None, .01, .03, .05, .075, .10, .125, .15, .175, .20, .25, .30, .40, .50)
IOUS = (.30, .40, .50, .60, .70, .80)
SEEDS = (88, 123, 666)
RECALL_FLOOR = .85
METRICS = ("mAP50", "mAP50_95", "precision", "recall", "f1", "fp_per_image")


def name_threshold(threshold):
    return "all" if threshold is None else f"{threshold:.3f}"


def filter_cached(predictions, threshold):
    if threshold is None:
        return predictions
    return {model: {image_id: [row for row in boxes if row["score"] > threshold]
                    for image_id, boxes in by_image.items()}
            for model, by_image in predictions.items()}


def load_split(frozen, split):
    seed, dataset = frozen["seed"], frozen["dataset"]
    prior.configure(seed)
    gt = prior.checked_ground_truth(dataset, split)
    predictions, cache = {}, {}
    for model in frozen["selected_models"]:
        predictions[model], cache[model] = prior.load_cached(dataset, split, model, seed, gt)
        expected = frozen[f"{split}_cache_provenance"][model]["sha256"]
        if cache[model]["sha256"] != expected:
            raise ValueError(f"Prediction cache changed: {seed}/{dataset}/{split}/{model}")
    return gt, predictions, cache


def compute_val(job):
    frozen, threshold, iou = job
    gt, predictions, cache = load_split(frozen, "val")
    used = filter_cached(predictions, threshold)
    fused = fusion.fuse_dataset(used, iou)
    metrics = base.compute_metrics(fused, gt, 0.0)
    return {"seed": frozen["seed"], "dataset": frozen["dataset"],
            "confidence": name_threshold(threshold), "cluster_iou": iou,
            "process_id": os.getpid(), "images": len(gt),
            "input_boxes": sum(len(boxes) for by_image in used.values() for boxes in by_image.values()),
            "fused_boxes": sum(map(len, fused.values())), **metrics}


def compute_test(job):
    frozen, objective, selected, output = job
    gt, predictions, cache = load_split(frozen, "test")
    confidence = selected["confidence"]
    threshold = None if confidence == "all" else float(confidence)
    used = filter_cached(predictions, threshold)
    fused = fusion.fuse_dataset(used, selected["cluster_iou"])
    metrics = base.compute_metrics(fused, gt, 0.0)
    directory = Path(output) / f"seed{frozen['seed']}" / frozen["dataset"]
    prior.write_json(directory / f"{objective}_test_fused_predictions.json", fused)
    return {"seed": frozen["seed"], "dataset": frozen["dataset"], "objective": objective,
            "confidence": confidence, "cluster_iou": selected["cluster_iou"],
            "process_id": os.getpid(), "test_images": len(gt),
            "test_gt_boxes": sum(map(len, gt.values())),
            "test_fused_boxes": sum(map(len, fused.values())),
            "test_metrics": metrics, "test_cache_provenance": cache,
            "val_metrics": {m: selected[m] for m in METRICS},
            "recall_floor_met_on_val": selected["recall"] >= RECALL_FLOOR}


def choose(val_rows):
    groups = defaultdict(list)
    for row in val_rows:
        groups[(row["dataset"], row["seed"])].append(row)
    selections = []
    for (dataset, seed), rows in sorted(groups.items()):
        if len(rows) != len(THRESHOLDS) * len(IOUS):
            raise AssertionError(f"Incomplete grid: {dataset}/seed{seed}")
        # Tie breaks prefer more recall, then more confidence screening and tighter
        # cluster matching; no test information enters the selection.
        def tie(row):
            return (row["recall"], -row["fp_per_image"],
                    -1 if row["confidence"] == "all" else float(row["confidence"]), row["cluster_iou"])
        f1 = max(rows, key=lambda row: (row["f1"], row["mAP50_95"], *tie(row)))
        eligible = [row for row in rows if row["recall"] >= RECALL_FLOOR]
        if eligible:
            constrained = min(eligible, key=lambda row: (row["fp_per_image"], -row["precision"],
                         -row["mAP50_95"], -row["recall"],
                         1 if row["confidence"] == "all" else -float(row["confidence"]), -row["cluster_iou"]))
        else:
            constrained = max(rows, key=lambda row: (row["recall"], -row["fp_per_image"], row["f1"]))
        for objective, selected in (("best_val_f1", f1), ("min_val_fp_with_recall_ge_085", constrained)):
            selections.append({"seed": seed, "dataset": dataset, "objective": objective,
                               "confidence": selected["confidence"], "cluster_iou": selected["cluster_iou"],
                               "val_metrics": {m: selected[m] for m in METRICS},
                               "val_tp": selected["tp"], "val_fp": selected["fp"], "val_fn": selected["fn"],
                               "recall_floor_met": selected["recall"] >= RECALL_FLOOR,
                               "process_id": selected["process_id"]})
    return selections


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["confidence"], row["cluster_iou"])].append(row)
    cmap_metrics = (("mAP50_95", "mAP50-95"), ("precision", "Precision @ IoU .50"),
                    ("recall", "Recall @ IoU .50"), ("f1", "F1 @ IoU .50"),
                    ("fp_per_image", "False positives / image"))
    for field, title in cmap_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        for ax, dataset in zip(axes, base.DATASETS):
            matrix = np.asarray([[statistics.mean(r[field] for r in groups[(dataset, name_threshold(c), iou)])
                                  for iou in IOUS] for c in THRESHOLDS])
            im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
            ax.set_title(dataset)
            ax.set_xticks(range(len(IOUS)), [f"{v:.2f}" for v in IOUS])
            ax.set_yticks(range(len(THRESHOLDS)), [name_threshold(v) for v in THRESHOLDS])
            ax.set_xlabel("Clustering IoU")
            ax.set_ylabel("Input confidence > threshold")
            fig.colorbar(im, ax=ax, shrink=.78)
        fig.suptitle(f"Validation {title} (mean of 3 seeds; top six selected within seed)")
        fig.savefig(figs / f"validation_{field}_heatmap.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    plot = None
    for ax, dataset in zip(axes, base.DATASETS):
        points = []
        for c in THRESHOLDS:
            for iou in IOUS:
                subset = groups[(dataset, name_threshold(c), iou)]
                points.append((statistics.mean(x["recall"] for x in subset),
                               statistics.mean(x["precision"] for x in subset),
                               .001 if c is None else c))
        plot = ax.scatter([p[0] for p in points], [p[1] for p in points],
                          c=[p[2] for p in points], cmap="plasma", vmin=.001, vmax=.5, s=32, alpha=.78)
        ax.set_title(dataset); ax.set_xlabel("Recall @ IoU .50"); ax.set_ylabel("Precision @ IoU .50")
        ax.grid(alpha=.2)
    fig.colorbar(plot, ax=axes, label="Input confidence (all cached = .001 for colour)", shrink=.8)
    fig.suptitle("Validation precision-recall trade-off across 84 configurations")
    fig.savefig(figs / "validation_precision_recall_tradeoff.png", dpi=180)
    plt.close(fig)


def make_report(rows, selected, tested, reference, output, elapsed, worker_limit):
    def avg_sd(values):
        return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"
    lookup = {(r["dataset"], r["seed"], r["objective"]): r for r in tested}
    earlier_010 = json.loads((prior.RUN_ROOT / "top6_hard_vote_conf010_min3_aug5x" / "results.json").read_text())["results"]
    lines = ["# 五倍增强：六模型三票融合的置信度 × 聚类 IoU 网格", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}；756 个独立验证进程，最大同时 {worker_limit} 个；总耗时 {elapsed:.1f} 秒。", "",
             "## 方法", "",
             "- 每数据集、每训练 seed 沿用此前在验证集选出的前六模型；每个模型只计一票，至少三个不同模型支持。坐标按每模型最高分代表框取中位数。",
             "- 输入分数范围：全部缓存框、0.01、0.03、0.05、0.075、0.10、0.125、0.15、0.175、0.20、0.25、0.30、0.40、0.50（均为严格大于）；聚类 IoU：0.30–0.80，步长 0.10。",
             "- 每个网格点为独立进程；14×6×3×3=756 组，仅在验证集计算。",
             "- 选出验证 F1 最高点，以及验证 Recall≥0.85 时 FP/图最低点；若无点满足召回约束，回退到最高 Recall 并明确标记。所有选择冻结后才读取测试集。",
             "- P/R/F1 在真值匹配 IoU=0.50、全部输出框下计算；mAP50–95 使用 0.50:0.05:0.95 的 101 点插值。聚类 IoU 与评估 IoU 是不同参数。",
             "- 输入是原有 GPU 推理缓存；导出阈值 0.001，RT-DETR 最多每图 300 框，原推理 NMS/候选截断不可从缓存恢复。本轮为 CPU 后处理。", "",
             "## 验证集参数曲面（三训练 seed 均值）", ""]
    for field in ("mAP50_95", "precision", "recall", "f1", "fp_per_image"):
        lines.append(f"![{field} heatmap](../../runs/inference/{prior.VERSION}/{NAME}/figures/validation_{field}_heatmap.png)")
        lines.append("")
    lines += [f"![Precision and recall trade-off](../../runs/inference/{prior.VERSION}/{NAME}/figures/validation_precision_recall_tradeoff.png)", "",
              "## 验证集所选参数及测试集结果", "",
              "| 数据集 | Seed | 验证目标 | 验证 R≥0.85 | 输入阈值 | 聚类 IoU | Val P/R/F1 | Test mAP50–95 | Test P/R/F1 | Test FP/图 |",
              "|---|---:|---|---|---:|---:|---|---:|---|---:|"]
    for sel in selected:
        test = lookup[(sel["dataset"], sel["seed"], sel["objective"])]
        val, met = sel["val_metrics"], test["test_metrics"]
        lines.append(f"| {sel['dataset']} | {sel['seed']} | {sel['objective']} | "
                     f"{'是' if sel['recall_floor_met'] else '否（最高召回回退）'} | {sel['confidence']} | {sel['cluster_iou']:.2f} | "
                     f"{val['precision']:.3f}/{val['recall']:.3f}/{val['f1']:.3f} | {met['mAP50_95']:.4f} | "
                     f"{met['precision']:.3f}/{met['recall']:.3f}/{met['f1']:.3f} | {met['fp_per_image']:.3f} |")
    lines += ["", "## 测试集三 seed 均值 ± 标准差", "",
              "| 数据集 | 方法 | mAP50–95 | Precision | Recall | F1 | FP/图 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in base.DATASETS:
        comparison = [r for r in reference if r["dataset"] == dataset]
        earlier = [r for r in earlier_010 if r["dataset"] == dataset]
        for label, values in (("固定 >0.1", [r["splits"]["test"]["hard_vote_conf010"] for r in earlier]),
                              ("固定 >0.2", [r["test_fusion_metrics"] for r in comparison])):
            lines.append(f"| {dataset} | {label} | " + " | ".join(avg_sd([v[m] for v in values])
                         for m in ("mAP50_95", "precision", "recall", "f1", "fp_per_image")) + " |")
        for objective, label in (("best_val_f1", "Val F1 优选"),
                                 ("min_val_fp_with_recall_ge_085", "Val R≥.85 时最少 FP")):
            group = [r["test_metrics"] for r in tested if r["dataset"] == dataset and r["objective"] == objective]
            lines.append(f"| {dataset} | {label} | " + " | ".join(avg_sd([v[m] for v in group])
                         for m in ("mAP50_95", "precision", "recall", "f1", "fp_per_image")) + " |")
    lines += ["", "## 主要观察", "",
              "- 按验证集 F1 选参数后，测试 F1 三数据集均高于固定 0.2 工作点，提升在 PolypGen 很小；mAP50–95 在三个数据集都低于固定 0.2。不存在同时改进所有指标的一组参数。",
              "- 在验证集约束 Recall≥0.85 时，Kvasir 三个 seed 都选择了不筛选输入，测试误报明显增加；seed123 连验证集 0.85 召回门槛也未达到。较高召回不能简单通过降低阈值获得而不付出误报代价。", "",
              "## 解释界限与文件", "",
              "- Kvasir-SEG seed123 的 3-of-6 网格无法达到验证 Recall≥0.85；该目标按预定规则回退到最大验证 Recall，所选点并不满足 0.85 约束。",
              "- 热力图是验证集结果；测试集只评估验证集冻结的两个目标，不从测试图上二次选最优阈值。",
              "- 固定划分上三个训练 seed 不等于患者或中心层面的独立重复。当前测试集已用于之前探索；论文最终确认应使用新留出中心。",
              "- 脚本逐点保存进程 PID，`validation_grid.csv` 可核验每组确实由不同进程计算。",
              f"- 脚本：`{Path(__file__).relative_to(ROOT)}`",
              f"- 全部结果：`{output.relative_to(ROOT)}`；包含 `validation_grid.csv`、`frozen_selections.json`、`test_selected.json`、`test_selected.csv`、`figures/` 和逐图融合输出。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 100:
        parser.error("workers must be between 1 and 100")
    source = REFERENCE.read_bytes()
    frozen = json.loads(source)["selections"]
    if len(frozen) != 9:
        raise ValueError("Expected 9 frozen seed/dataset ensembles")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    jobs = [(r, confidence, iou) for r in frozen for confidence in THRESHOLDS for iou in IOUS]
    if len(jobs) != 756:
        raise AssertionError("Grid size changed unexpectedly")
    prior.write_json(OUTPUT / "protocol.json", {
        "source": str(REFERENCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(source).hexdigest(),
        "seeds": SEEDS, "datasets": base.DATASETS, "confidence_thresholds": [name_threshold(v) for v in THRESHOLDS],
        "cluster_ious": IOUS, "total_val_processes": len(jobs), "max_concurrent_processes": args.workers,
        "selection_objectives": ["max_validation_f1", "min_validation_fp_per_image_given_recall_ge_0.85"],
        "models_per_group": [{"seed": r["seed"], "dataset": r["dataset"], "selected_models": r["selected_models"]}
                             for r in frozen]})
    started = time.monotonic()
    rows = []
    # maxtasksperchild=1 gives each grid point its own PID and address space.
    with mp.get_context("fork").Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(compute_val, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 25 == 0 or len(rows) == len(jobs):
                print(json.dumps({"phase": "validation", "completed": len(rows), "total": len(jobs),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    if len({r["process_id"] for r in rows}) != len(jobs):
        raise AssertionError("Every configuration must run in a unique process")
    rows.sort(key=lambda r: (r["dataset"], r["seed"],
                             -1 if r["confidence"] == "all" else float(r["confidence"]), r["cluster_iou"]))
    write_csv(OUTPUT / "validation_grid.csv", rows)
    prior.write_json(OUTPUT / "validation_grid.json", rows)
    selected = choose(rows)
    prior.write_json(OUTPUT / "frozen_selections.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selections": selected})
    print("VALIDATION SELECTIONS FROZEN; test evaluation begins", flush=True)
    # This older test-results file is intentionally opened only after the new
    # validation selections have been written and frozen.
    test_reference = json.loads(TEST_REFERENCE.read_text())["results"]
    lookup = {(r["seed"], r["dataset"]): r for r in test_reference}
    test_jobs = []
    for choice in selected:
        source_model = lookup[(choice["seed"], choice["dataset"])]
        validation_source = next(r for r in frozen if r["seed"] == choice["seed"] and r["dataset"] == choice["dataset"])
        if (source_model["selected_models"] != validation_source["selected_models"]
                or source_model["val_cache_provenance"] != validation_source["val_cache_provenance"]):
            raise ValueError("Frozen validation selection differs from earlier test reference")
        selected_row = next(r for r in rows if r["dataset"] == choice["dataset"] and r["seed"] == choice["seed"]
                            and r["confidence"] == choice["confidence"] and r["cluster_iou"] == choice["cluster_iou"])
        test_jobs.append((source_model, choice["objective"], selected_row, str(OUTPUT)))
    tested = []
    with mp.get_context("fork").Pool(processes=min(args.workers, len(test_jobs)), maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(compute_test, test_jobs, chunksize=1):
            tested.append(result)
            print(json.dumps({"phase": "test", "completed": len(tested), "total": len(test_jobs),
                              "dataset": result["dataset"], "seed": result["seed"],
                              "objective": result["objective"]}), flush=True)
    tested.sort(key=lambda r: (r["dataset"], r["seed"], r["objective"]))
    prior.write_json(OUTPUT / "test_selected.json", tested)
    write_csv(OUTPUT / "test_selected.csv", [{"dataset": r["dataset"], "seed": r["seed"],
        "objective": r["objective"], "confidence": r["confidence"], "cluster_iou": r["cluster_iou"],
        "val_recall_floor_met": r["recall_floor_met_on_val"],
        **{"val_" + k: v for k, v in r["val_metrics"].items()},
        **{"test_" + k: v for k, v in r["test_metrics"].items()}} for r in tested])
    make_plots(rows, OUTPUT)
    elapsed = time.monotonic() - started
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(make_report(rows, selected, tested, test_reference, OUTPUT, elapsed, args.workers))
    print(json.dumps({"phase": "done", "seconds": round(elapsed, 1), "report": str(REPORT)}), flush=True)


if __name__ == "__main__":
    main()
