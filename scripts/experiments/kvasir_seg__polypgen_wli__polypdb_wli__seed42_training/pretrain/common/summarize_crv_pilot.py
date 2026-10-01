#!/usr/bin/env python3
"""Summarize validation-selected CRV-inspired fusion pilot results."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
PILOT = ROOT / "runs/inference" / BASE / "crv_pretrained_aug5x"
REPORT = ROOT / "reports" / BASE / "crv_counterfactual_fusion_pilot_pretrained_aug5x.md"
SEEDS = (88, 123, 666)
DATASETS = (
    ("kvasir_seg", "Kvasir-SEG"),
    ("polypgen_wli", "PolypGen-WLI"),
    ("polypdb_wli", "PolypDB-WLI"),
)
METRICS = (
    ("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("precision", "Precision"),
    ("recall", "Recall"), ("f1", "F1"), ("fp_per_image", "FP/image"),
)


def fmt(xs, digits=4):
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main():
    rows = {}
    for seed in SEEDS:
        for dataset, _ in DATASETS:
            path = PILOT / f"seed{seed}" / dataset / "crv_vote_pilot.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows[(seed, dataset)] = json.loads(path.read_text(encoding="utf-8"))

    out = [
        "# CRV-inspired 反事实响应融合 Pilot（pretrained_aug5x）",
        "",
        "## 1. 目的与实现",
        "",
        "基于本地论文《Re-Querying the Detector: Counterfactual Response Verification for Anomaly Detection under Pose Variation》，测试其“参考正常区域构造反事实图、重查同一个冻结检测器、按响应下降验证候选”的想法能否改善现有 majority vote。",
        "",
        "迁移到 box detector 的实现：用训练集原始图像中 segmentation mask 膨胀12像素后仍为 lesion-free 的 64×64 patches 建参考库；用冻结 ImageNet ResNet-18 layer2/layer3 描述符检索每个 majority-vote 候选框最相似的3个参考 patch；将参考 patch 平滑贴入候选框并用验证集选择出的单模型检测器复推。候选置信度增量为各参考的正响应下降 × 归一化外观变化的中位数，再以 `score' = clip(score + α × evidence, 0, 1)` 进入原 majority-vote 结果。",
        "",
        "该实现是 CRV 的框级近似，不是论文中连续 anomaly-map 的原样复现。候选框、参考库、验证参数均不使用测试标注。先在验证集从 α={0, 0.25, 0.5, 1, 2, 4} 中按 mAP50-95 选择（并列时按 F1，再选更小 α），随后冻结 α，在测试集评估一次。测试集结果没有用于调参。",
        "",
        "## 2. 验证集所选权重与测试集结果",
        "",
        "指标为每个训练 seed 的测试结果；括号中为 CRV - baseline。", "",
        "| 数据集 | Seed | 验证选择 α | 验证 mAP50-95（baseline → CRV） | 测试 mAP50-95（baseline → CRV） | Δ Test mAP50-95 | Δ Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, label in DATASETS:
        for seed in SEEDS:
            x = rows[(seed, dataset)]
            vb = x["validation_candidates"][0]["metrics"]["mAP50_95"]
            selected_val = next(r["metrics"]["mAP50_95"] for r in x["validation_candidates"] if r["alpha"] == x["selected_alpha"])
            tb = x["test_baseline_majority_vote"]
            tc = x["test_crv_reweighted_majority_vote"]
            out.append(f"| {label} | {seed} | {x['selected_alpha']:.2f} | {vb:.4f} → {selected_val:.4f} | {tb['mAP50_95']:.4f} → {tc['mAP50_95']:.4f} | {tc['mAP50_95']-tb['mAP50_95']:+.4f} | {tc['f1']-tb['f1']:+.4f} |")
        ds_rows = [rows[(seed, dataset)] for seed in SEEDS]
        baseline_map = [x["test_baseline_majority_vote"]["mAP50_95"] for x in ds_rows]
        crv_map = [x["test_crv_reweighted_majority_vote"]["mAP50_95"] for x in ds_rows]
        deltas = [c-b for b, c in zip(baseline_map, crv_map)]
        out.append(f"| **{label} mean ± SD** | — | {sum(x['selected_alpha'] > 0 for x in ds_rows)}/3 nonzero | — | {fmt(baseline_map)} → {fmt(crv_map)} | **{fmt(deltas)}** | **{fmt([x['test_crv_reweighted_majority_vote']['f1']-x['test_baseline_majority_vote']['f1'] for x in ds_rows])}** |")

    out += ["", "## 3. 完整测试指标（3 seeds 均值 ± 样本标准差）", "", "| 数据集 | 指标 | Majority baseline | CRV reweighted | Δ |", "|---|---|---:|---:|---:|"]
    for dataset, label in DATASETS:
        for metric, display in METRICS:
            base = [rows[(s, dataset)]["test_baseline_majority_vote"][metric] for s in SEEDS]
            crv = [rows[(s, dataset)]["test_crv_reweighted_majority_vote"][metric] for s in SEEDS]
            delta = [c-b for b, c in zip(base, crv)]
            out.append(f"| {label} | {display} | {fmt(base)} | {fmt(crv)} | {fmt(delta)} |")

    out += [
        "",
        "## 4. 结论",
        "",
        "- 9个 seed×数据集组合中，8组在验证集选择 α=0；唯一非零的是 seed123/Kvasir-SEG（α=0.25），其冻结后的测试 mAP50-95 下降约0.0011。",
        "- 三数据集跨 seed 的测试 mAP50-95 平均变化均不为正：Kvasir-SEG 约 -0.0004；PolypGen-WLI 和 PolypDB-WLI 为0（验证选出的 α 全为0）。Precision/recall/F1 也未改善，因为大多数组合的冻结权重为0。",
        "- 因此，本次 CRV 框级适配没有证据优于现有融合；建议不把它并入主报告/正式投票方案。测试集未参与 α 选择，测试差值仅是冻结后的评估。",
        "- 可能原因：原论文使用连续 anomaly map，响应下降具有局部异常语义；我们使用的是监督检测器的 box confidence，候选区域替换会同时改变真阳性和局部伪阳性的纹理，二者的响应变化没有提供可泛化的额外排序信息。",
        "- 可复现性限制：本 pilot 用 64×64 lesion-free train patches 和平滑矩形融合近似论文的 dense feature matching + WOLA；这验证的是一个廉价、可运行的框级迁移版本，不足以否定论文原方法在其 anomaly detector 上的结果。",
        "",
        "## 5. 产物",
        "",
        f"- 单组完整结果：`runs/inference/{BASE}/crv_pretrained_aug5x/seed<seed>/<dataset>/crv_vote_pilot.json`",
        f"- 正常参考 patch 特征缓存：`runs/inference/{BASE}/crv_pretrained_aug5x/normal_reference_banks/`",
        "- 实验脚本：`scripts/experiments/kvasir_seg__polypgen_wli__polypdb_wli__seed42_training/pretrain/common/crv_counterfactual_vote.py`",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
