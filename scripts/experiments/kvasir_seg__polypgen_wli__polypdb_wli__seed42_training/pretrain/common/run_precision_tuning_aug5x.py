#!/usr/bin/env python3
"""Validation-only precision-oriented fusion search over cached aug5x outputs."""
from __future__ import annotations

import concurrent.futures
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
OUT = Path(__file__).resolve().parents[5] / "runs/inference" / BASE / "precision_tuning_pretrained_aug5x"
SEEDS = (88, 123, 666)
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
RECALL_FLOORS = (0.75, 0.80)


def worker(job: tuple[int, str]) -> dict[str, Any]:
    seed, dataset = job
    os.environ.update({
        "VOTING_CONDITION": "pretrained_aug5x",
        "VOTING_SEED": str(seed),
        "VOTING_OUTPUT_NAME": f"consolidated_pretrained_aug5x_seed{seed}_voting",
    })
    module_path = HERE / "run_voting_analysis.py"
    spec = importlib.util.spec_from_file_location("voting_impl", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gt = mod.load_ground_truth(dataset, "val")
    preds = {model: mod.load_predictions(dataset, "val", model) for model in mod.MODELS}
    test_gt_path = mod.DATA_ROOT / dataset / "test" / "labels"
    if not test_gt_path.exists():
        raise FileNotFoundError(test_gt_path)

    # A cluster is generated once per geometry/support/floor setting; confidence
    # thresholds are then evaluated cheaply. This is cached-prediction CPU work.
    candidates: list[dict[str, Any]] = []
    for iou in (0.50, 0.60, 0.70, 0.80):
        for score_floor in (0.01, 0.05, 0.10, 0.20):
            for support in range(3, 13):
                fused: dict[str, list[dict[str, Any]]] = {}
                for image_id in sorted(gt):
                    per_model = {m: preds[m].get(image_id, []) for m in mod.MODELS}
                    fused[image_id] = mod.cluster_predictions(
                        per_model, {m: 1.0 for m in mod.MODELS}, iou, score_floor,
                        support, use_median=True,
                    )
                for conf in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                    tp = fp = 0
                    for image_id, boxes in gt.items():
                        operating = [p for p in fused[image_id] if p["score"] >= conf]
                        _, flags = mod.match_flags(operating, boxes, 0.50)
                        tp += sum(flags)
                        fp += len(flags) - sum(flags)
                    n_gt = sum(map(len, gt.values()))
                    recall = tp / max(n_gt, 1)
                    precision = tp / max(tp + fp, 1)
                    f05 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-12)
                    candidates.append({
                        "params": {"cluster_iou": iou, "score_floor": score_floor,
                                   "min_models": support, "operating_confidence": conf},
                        "precision": precision, "recall": recall, "f0_5": f05,
                        "tp": tp, "fp": fp, "fn": n_gt - tp,
                    })

    selected: dict[str, Any] = {}
    for floor in RECALL_FLOORS:
        eligible = [r for r in candidates if r["recall"] >= floor]
        # Precision first, then F0.5, then recall; choose simplest support as final tie-break.
        best = max(eligible or candidates, key=lambda r: (
            r["precision"] if eligible else r["f0_5"], r["f0_5"], r["recall"],
            r["params"]["min_models"],
        ))
        selected[f"recall_at_least_{floor:.2f}"] = {**best, "recall_constraint_met": bool(eligible)}

    # The test split is opened only now, after validation selections are frozen.
    test_gt = mod.load_ground_truth(dataset, "test")
    test_preds = {model: mod.load_predictions(dataset, "test", model) for model in mod.MODELS}
    tested = {}
    for name, choice in selected.items():
        p = choice["params"]
        fused = {
            image_id: mod.cluster_predictions(
                {m: test_preds[m].get(image_id, []) for m in mod.MODELS},
                {m: 1.0 for m in mod.MODELS}, p["cluster_iou"], p["score_floor"],
                p["min_models"], use_median=True,
            ) for image_id in sorted(test_gt)
        }
        metrics = mod.compute_metrics(fused, test_gt, p["operating_confidence"])
        tested[name] = {"params": p, "val_metrics": {k: choice[k] for k in ("precision", "recall", "f0_5", "tp", "fp", "fn")},
                        "test_metrics": metrics, "recall_constraint_met_on_val": choice["recall_constraint_met"]}

    return {"condition": "pretrained_aug5x", "seed": seed, "dataset": dataset,
            "val_candidate_count": len(candidates), "selected": tested,
            "top_val_candidates_by_precision_at_recall_075": [
                r for r in sorted((x for x in candidates if x["recall"] >= 0.75),
                                  key=lambda x: (x["precision"], x["f0_5"]), reverse=True)[:20]
            ]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, dataset) for seed in SEEDS for dataset in DATASETS]
    results = []
    # Nine independent CPU workers (one per seed/dataset) avoid oversubscription;
    # raw detector inference is not repeated.
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            out_file = OUT / f"seed{job[0]}" / job[1] / "precision_tuning.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"event": "finished", "seed": job[0], "dataset": job[1],
                              "selected": {k: v["test_metrics"]["precision"] for k, v in result["selected"].items()}},
                             ensure_ascii=False), flush=True)

    results.sort(key=lambda x: (x["dataset"], x["seed"]))
    (OUT / "precision_tuning_all.json").write_text(json.dumps({"schema_version": 1, "selection":
        "maximize validation precision subject to recall >= 0.75 or 0.80; test evaluated after freeze",
        "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for result in results:
        for objective, item in result["selected"].items():
            rows.append({"seed": result["seed"], "dataset": result["dataset"], "objective": objective,
                         **item["params"], "val_precision": item["val_metrics"]["precision"],
                         "val_recall": item["val_metrics"]["recall"], "val_f0_5": item["val_metrics"]["f0_5"],
                         "test_precision": item["test_metrics"]["precision"], "test_recall": item["test_metrics"]["recall"],
                         "test_f1": item["test_metrics"]["f1"], "test_map50_95": item["test_metrics"]["mAP50_95"],
                         "test_fp_per_image": item["test_metrics"]["fp_per_image"],
                         "recall_constraint_met_on_val": item["recall_constraint_met_on_val"]})
    with (OUT / "precision_tuning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"event": "all_done", "output": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
