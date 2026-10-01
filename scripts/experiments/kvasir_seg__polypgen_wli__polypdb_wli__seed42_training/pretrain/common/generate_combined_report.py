#!/usr/bin/env python3
"""Create the cross-condition, cross-seed training and voting report."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
TRAIN_ROOT = ROOT / "runs/training" / BASE
INFER_ROOT = ROOT / "runs/inference" / BASE
REPORT_ROOT = ROOT / "reports" / BASE
CONDITIONS = ("scratch_base", "pretrained_base", "pretrained_aug3x", "pretrained_aug5x")
SEEDS = (88, 123, 666)
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
DATASET_LABELS = {"kvasir_seg": "Kvasir-SEG", "polypgen_wli": "PolypGen-WLI", "polypdb_wli": "PolypDB-WLI"}
METRICS = (
    ("best_epoch", "Best epoch"), ("best_val_map50", "Val mAP50"),
    ("best_val_map50_95", "Val mAP50-95"), ("test_map50", "Test mAP50"),
    ("test_map50_95", "Test mAP50-95"), ("test_precision", "Test precision"),
    ("test_recall", "Test recall"), ("test_f1", "Test F1"),
)
VOTE_METRICS = (
    ("val_map50", "Val mAP50", "val_metrics", "mAP50"),
    ("val_map50_95", "Val mAP50-95", "val_metrics", "mAP50_95"),
    ("val_precision", "Val precision", "val_metrics", "precision"),
    ("val_recall", "Val recall", "val_metrics", "recall"),
    ("val_f1", "Val F1", "val_metrics", "f1"),
    ("val_fp_per_image", "Val FP/image", "val_metrics", "fp_per_image"),
    ("test_map50", "Test mAP50", "test_metrics", "mAP50"),
    ("test_map50_95", "Test mAP50-95", "test_metrics", "mAP50_95"),
    ("test_precision", "Test precision", "test_metrics", "precision"),
    ("test_recall", "Test recall", "test_metrics", "recall"),
    ("test_f1", "Test F1", "test_metrics", "f1"),
    ("test_fp_per_image", "Test FP/image", "test_metrics", "fp_per_image"),
)
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s", "rtdetr_l", "rtdetr_x",
)
METHODS = ("single_best", "equal_soft_vote", "val_weighted_soft_vote", "majority_vote", "strict_consensus_vote")


def fmt(values: list[float], digits: int = 4) -> str:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows: dict[tuple[str, str, str, int], dict] = {}
    for condition in CONDITIONS:
        for dataset in DATASETS:
            for model in MODELS:
                for seed in SEEDS:
                    path = TRAIN_ROOT / condition / f"seed{seed}" / dataset / model / "test_metrics.json"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    item = load(path)
                    if item.get("status") != "completed":
                        raise RuntimeError(f"training task incomplete: {path}")
                    rows[(condition, dataset, model, seed)] = item

    votes: dict[tuple[str, int, str], dict] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            output = f"consolidated_{condition}_seed{seed}_voting"
            path = INFER_ROOT / output / "voting_analysis.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = load(path)
            for dataset in DATASETS:
                votes[(condition, seed, dataset)] = payload["datasets"][dataset]

    lines = [
        "# Kvasir-SEG / PolypGen-WLI / PolypDB-WLI：四种训练条件综合报告",
        "",
        "## 1. 实验范围与统计口径",
        "",
        "- 条件：scratch（从头训练）、pretrained base、pretrained 3× 离线增强、pretrained 5× 离线增强。",
        "- 数据集划分 seed=42；训练 seed=88、123、666；CVC-ClinicDB 未纳入。每个条件共 108 个训练任务（3 数据集 × 12 模型 × 3 seeds）。",
        "- 各训练指标表为同一模型跨3个训练 seed 的均值 ± 样本标准差；均值和标准差不是跨架构统计。",
        "- 汇总表先在每个 seed 内对12个模型取均值，再对3个 seed 计算均值 ± 样本标准差。",
        "- 融合配置只依据验证集选择，测试集只用于冻结配置后的最终评估；融合指标取3个 seed 的均值 ± 样本标准差。",
        "- 所有实验均训练25 epochs、输入640、AdamW（lr=1e-4，weight decay=5e-4）、在线增强关闭；scratch 使用随机初始化，其余条件从公开预训练权重初始化。增强倍数表示训练视图总倍数。",
        "- 预测由 GPU 执行。曾尝试两张 GPU 共12模型并发（每卡6个），但受其他作业动态占用显存影响发生 OOM；失败任务分批续跑，最多每卡2个模型并发。单模型先试 batch=128，OOM 时回退到32；此前已完成的 seed88 预测缓存沿用 batch=8。不同 seed/模型实际 batch 和并发可能不同，原始评估指标不受批次大小影响。",
        "- 原始训练指标来自各模板的 `test_metrics.json`；融合指标来自独立预测和投票评估脚本，数值可能因评估实现/置信度协议不同而与训练框架单模型指标略有差异。",
        "",
        "## 2. 训练完整性",
        "",
        "| 条件 | 计划任务 | 完成任务 | 未完成 |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        count = sum(key[0] == condition for key in rows)
        lines.append(f"| `{condition}` | 108 | {count} | {108-count} |")

    lines += ["", "## 3. 数据集级结果总览", "", "此处每个 seed 先对12个模型的 mAP50-95 求平均，再在3个 seed 间汇总。", ""]
    lines += ["| 条件 | 数据集 | Val mAP50-95 | Test mAP50-95 | Test precision | Test recall | Test F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for dataset in DATASETS:
            seed_metrics = []
            for seed in SEEDS:
                seed_items = [rows[(condition, dataset, model, seed)] for model in MODELS]
                seed_metrics.append({
                    "val": statistics.mean(item["best_val_map50_95"] for item in seed_items),
                    **{key: statistics.mean(item[key] for item in seed_items) for key in ("test_map50_95", "test_precision", "test_recall", "test_f1")},
                })
            lines.append(
                f"| `{condition}` | {DATASET_LABELS[dataset]} | {fmt([x['val'] for x in seed_metrics])} | "
                f"{fmt([x['test_map50_95'] for x in seed_metrics])} | {fmt([x['test_precision'] for x in seed_metrics])} | "
                f"{fmt([x['test_recall'] for x in seed_metrics])} | {fmt([x['test_f1'] for x in seed_metrics])} |"
            )

    lines += ["", "## 4. 各条件、数据集和模型的完整训练指标", "", "每格均为3个训练 seed 的均值 ± 样本标准差；epoch 同样按连续值汇总。", ""]
    for condition in CONDITIONS:
        lines += [f"### {condition}", ""]
        for dataset in DATASETS:
            lines += [f"#### {DATASET_LABELS[dataset]}", "", "| 模型 | Best epoch | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test precision | Test recall | Test F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            model_means = {}
            for model in MODELS:
                items = [rows[(condition, dataset, model, seed)] for seed in SEEDS]
                model_means[model] = statistics.mean(item["best_val_map50_95"] for item in items)
            for model in sorted(MODELS, key=lambda m: model_means[m], reverse=True):
                items = [rows[(condition, dataset, model, seed)] for seed in SEEDS]
                vals = [fmt([float(item[key]) for item in items], 2 if key == "best_epoch" else 4) for key, _ in METRICS]
                lines.append(f"| `{model}` | " + " | ".join(vals) + " |")
            lines.append("")

    lines += ["## 5. 验证集调参后的投票/融合结果", "", "方法：`single_best`、等权软投票、验证集加权软投票、majority vote、strict consensus vote。每个 seed/数据集分别在验证集上选择配置；delta 是同一 seed 对 `single_best` 的测试 mAP50-95 差值。", ""]
    for condition in CONDITIONS:
        lines += [f"### {condition}", ""]
        for dataset in DATASETS:
            lines += [f"#### {DATASET_LABELS[dataset]}", "", "| 方法 | Val mAP50 | Val mAP50-95 | Val precision | Val recall | Val F1 | Val FP/image | Test mAP50 | Test mAP50-95 | Δ Test mAP50-95 | Test precision | Test recall | Test F1 | Test FP/image |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            by_seed = {}
            for seed in SEEDS:
                result_list = votes[(condition, seed, dataset)]["results"]
                by_seed[seed] = {result["method"]: result for result in result_list}
            for method in METHODS:
                result_by_seed = [by_seed[seed][method] for seed in SEEDS]
                vals = []
                for key, _, split_key, metric_key in VOTE_METRICS:
                    vals.append(fmt([float(result[split_key][metric_key]) for result in result_by_seed]))
                deltas = []
                for seed in SEEDS:
                    baseline = by_seed[seed]["single_best"]["test_metrics"]["mAP50_95"]
                    current = by_seed[seed][method]["test_metrics"]["mAP50_95"]
                    deltas.append(float(current - baseline))
                lines.append(f"| `{method}` | " + " | ".join(vals[:8] + [fmt(deltas)] + vals[8:]) + " |")
            lines += ["", "各 seed 在验证集选出的融合参数：", "", "| Seed | 方法 | 参数（JSON） |", "|---:|---|---|"]
            for seed in SEEDS:
                for method in METHODS:
                    params = json.dumps(by_seed[seed][method]["params"], ensure_ascii=False, sort_keys=True)
                    lines.append(f"| {seed} | `{method}` | `{params}` |")
            lines.append("")

    lines += ["## 6. 解读注意事项", "", "- 以验证集选择规则作为模型选择依据；测试集最优项只做描述，不能反向选择模型或阈值。",
              "- 融合是否优于单模型应以跨 seed 的测试 mAP50-95 变化和 precision/recall/F1 一并判断，不应只看单次 seed 的某一项指标。",
              "- 本报告反映固定划分（seed42）下的三次训练随机种子重复，不等同于跨数据划分或外部中心验证。", "",
              "## 7. 产物路径", "",
              f"- 训练结果：`runs/training/{BASE}/<condition>/seed<seed>/<dataset>/<model>/test_metrics.json`",
              f"- 投票结果：`runs/inference/{BASE}/consolidated_<condition>_seed<seed>_voting/voting_analysis.json`",
              ""]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    target = REPORT_ROOT / "combined_scratch_pretrained_aug3x_aug5x_seed88_123_666.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
