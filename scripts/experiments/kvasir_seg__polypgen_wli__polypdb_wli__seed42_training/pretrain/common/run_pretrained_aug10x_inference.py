#!/usr/bin/env python3
"""Run resumable raw inference and parallel voting analysis for trained conditions.

The worker queue is restartable: valid per-model prediction JSONs are reused,
and only checkpoints whose training status is completed are scheduled.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
BASE = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
CONDITION = "pretrained_aug10x"
DATA_VERSION = BASE + "__aug10x"
SEEDS = (88, 123, 666)
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s",
    "rtdetr_l", "rtdetr_x",
)
TRAIN_ROOT = ROOT / "runs/training" / BASE / CONDITION
INFERENCE_ROOT = ROOT / "runs/inference" / BASE
VOTER = Path(__file__).resolve().with_name("run_voting_analysis.py")
SCRIPT = Path(__file__).resolve()


def output_root(seed: int) -> Path:
    return INFERENCE_ROOT / f"consolidated_{CONDITION}_seed{seed}_voting"


def configure_condition(condition: str) -> None:
    global CONDITION, DATA_VERSION, TRAIN_ROOT
    CONDITION = condition
    suffix = {"pretrained_aug10x": "__aug10x", "pretrained_repeat5x": "__repeat5x"}[condition]
    DATA_VERSION = BASE + suffix
    TRAIN_ROOT = ROOT / "runs/training" / BASE / CONDITION


def status_path(seed: int, dataset: str, model: str) -> Path:
    return TRAIN_ROOT / f"seed{seed}" / dataset / model / "status.json"


def checkpoint_path(seed: int, dataset: str, model: str) -> Path:
    run = TRAIN_ROOT / f"seed{seed}" / dataset / model
    return run / ("best.pt" if model == "fasterrcnn_resnet50_fpn" else "weights/best.pt")


def completed(seed: int, dataset: str, model: str) -> bool:
    try:
        state = json.loads(status_path(seed, dataset, model).read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return state == "completed" and checkpoint_path(seed, dataset, model).is_file()


def load_voter(seed: int):
    os.environ.update({
        "VOTING_CONDITION": CONDITION,
        "VOTING_SEED": str(seed),
        "VOTING_DATA_VERSION": DATA_VERSION,
        "VOTING_OUTPUT_NAME": f"consolidated_{CONDITION}_seed{seed}_voting",
    })
    module_name = f"voting_{CONDITION}_seed{seed}"
    spec = importlib.util.spec_from_file_location(module_name, VOTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import voting implementation: {VOTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def write_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    INFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    target = INFERENCE_ROOT / f"{CONDITION}_inference_scheduler_status.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(target)


def prepare_outputs() -> None:
    for seed in SEEDS:
        out = output_root(seed)
        (out / "worker_logs").mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            for split in ("val", "test"):
                (out / "raw" / dataset / split).mkdir(parents=True, exist_ok=True)


def run_worker(args: argparse.Namespace) -> int:
    os.environ["VOTING_CONDITION"] = CONDITION
    os.environ["VOTING_SEED"] = str(args.seed)
    os.environ["VOTING_DATA_VERSION"] = DATA_VERSION
    os.environ["VOTING_OUTPUT_NAME"] = f"consolidated_{CONDITION}_seed{args.seed}_voting"
    module = load_voter(args.seed)
    module.DATASETS = (args.dataset,)
    module.export_all(
        device="0", batch=args.batch, score_floor=0.001, force=args.force,
        delay_seconds=0.0, model_names=(args.model,),
    )
    return 0


def run_analysis(seed: int) -> None:
    module = load_voter(seed)
    module.DATASETS = DATASETS
    module.write_analysis()


def run_analysis_subprocess(seed: int) -> int:
    env = os.environ.copy()
    env.update({
        "VOTING_CONDITION": CONDITION,
        "VOTING_SEED": str(seed),
        "VOTING_DATA_VERSION": DATA_VERSION,
        "VOTING_OUTPUT_NAME": f"consolidated_{CONDITION}_seed{seed}_voting",
    })
    log_path = output_root(seed) / "worker_logs" / "analysis.log"
    command = [sys.executable, str(SCRIPT), "--analysis-worker-seed", str(seed), "--condition", CONDITION]
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("pretrained_aug10x", "pretrained_repeat5x"),
                        default="pretrained_aug10x")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--analysis-worker-seed", type=int, choices=SEEDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", choices=("0", "1"), help="physical GPU for this worker")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers-per-gpu", type=int, default=None,
                        help="set the same worker count on both GPUs (compatibility option)")
    parser.add_argument("--workers-gpu0", type=int, default=3)
    parser.add_argument("--workers-gpu1", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="recompute every prediction, ignoring caches")
    args = parser.parse_args()
    configure_condition(args.condition)
    if args.workers_per_gpu is not None:
        args.workers_gpu0 = args.workers_per_gpu
        args.workers_gpu1 = args.workers_per_gpu
    if args.analysis_worker_seed is not None:
        if args.worker:
            parser.error("--analysis-worker-seed cannot be combined with --worker")
        run_analysis(args.analysis_worker_seed)
        return 0
    if args.worker:
        if args.seed is None or args.dataset is None or args.model is None or args.device is None:
            parser.error("--worker requires --seed, --dataset, --model, and --device")
        return run_worker(args)

    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    if not 1 <= args.workers_gpu0 <= 4 or not 1 <= args.workers_gpu1 <= 4:
        parser.error("per-GPU worker counts must be between 1 and 4")
    prepare_outputs()

    queue = [
        (seed, dataset, model)
        for seed in SEEDS for dataset in DATASETS for model in MODELS
        if completed(seed, dataset, model)
    ]
    initial_completed_count = len(queue)
    active: dict[str, list[dict[str, Any]]] = {"0": [], "1": []}
    finished: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "condition": CONDITION, "batch": args.batch,
        "workers_per_gpu": {"0": args.workers_gpu0, "1": args.workers_gpu1},
        "force_recompute": args.force,
        "task_count_at_start": initial_completed_count,
        "queued": len(queue), "finished": [], "failed": [],
        "status": "running",
    }
    write_state(state)
    next_gpu = "0"
    slots = {"0": args.workers_gpu0, "1": args.workers_gpu1}

    def launch(device: str, task: tuple[int, str, str]) -> None:
        seed, dataset, model = task
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": device,
            "GI_PHYSICAL_DEVICE": device,
            "PYTHONUNBUFFERED": "1",
            "VOTING_CONDITION": CONDITION,
            "VOTING_SEED": str(seed),
            "VOTING_DATA_VERSION": DATA_VERSION,
            "VOTING_OUTPUT_NAME": f"consolidated_{CONDITION}_seed{seed}_voting",
        })
        log_path = output_root(seed) / "worker_logs" / f"{dataset}__{model}__gpu{device}.log"
        log = log_path.open("a", encoding="utf-8")
        command = [sys.executable, str(SCRIPT), "--worker", "--seed", str(seed),
                   "--dataset", dataset, "--model", model, "--device", device,
                   "--batch", str(args.batch), "--condition", CONDITION]
        if args.force:
            command.append("--force")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active[device].append({
            "seed": seed, "dataset": dataset, "model": model,
            "process": process, "log": log, "log_path": str(log_path),
        })
        emit({
            "event": "model_started", "seed": seed, "dataset": dataset,
            "model": model, "gpu": device, "batch": args.batch,
            "active_gpu0": len(active["0"]), "active_gpu1": len(active["1"]),
            "queued": len(queue),
        })

    # Fill each GPU to the configured number of concurrent model workers.
    for device in ("0", "1"):
        while queue and len(active[device]) < slots[device]:
            task = queue.pop(0)
            launch(device, task)

    while queue or active["0"] or active["1"]:
        for device in ("0", "1"):
            remaining = []
            for item in active[device]:
                code = item["process"].poll()
                if code is None:
                    remaining.append(item)
                    continue
                item["log"].close()
                result = {
                    "seed": item["seed"], "dataset": item["dataset"],
                    "model": item["model"], "gpu": device, "returncode": code,
                    "status": "completed" if code == 0 else "failed",
                    "log": item["log_path"],
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                finished.append(result)
                emit({"event": "model_finished", **result})
                if code != 0:
                    failures.append(result)
            active[device] = remaining
            while queue and len(active[device]) < slots[device]:
                task = queue.pop(0)
                launch(device, task)

        state.update({
            "queued": len(queue), "finished": finished, "failed": failures,
            "active_gpu0": len(active["0"]), "active_gpu1": len(active["1"]),
        })
        write_state(state)
        time.sleep(2.0)

    analyses: dict[str, str] = {}
    eligible_for_analysis: list[int] = []
    for seed in SEEDS:
        all_trained = all(completed(seed, dataset, model) for dataset in DATASETS for model in MODELS)
        all_predictions = all(
            (output_root(seed) / "raw" / dataset / split / f"{model}.json").is_file()
            for dataset in DATASETS for split in ("val", "test") for model in MODELS
        )
        if all_trained and all_predictions:
            eligible_for_analysis.append(seed)
        else:
            analyses[str(seed)] = "deferred_until_all_checkpoints_and_predictions_exist"

    if eligible_for_analysis:
        with ThreadPoolExecutor(max_workers=len(eligible_for_analysis)) as pool:
            futures = {
                pool.submit(run_analysis_subprocess, seed): seed
                for seed in eligible_for_analysis
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    code = future.result()
                    analyses[str(seed)] = "complete" if code == 0 else f"failed: subprocess_exit_{code}"
                    if code != 0:
                        failures.append({"seed": seed, "task": "analysis", "status": "failed", "returncode": code})
                except Exception as exc:
                    analyses[str(seed)] = f"failed: {type(exc).__name__}: {exc}"
                    failures.append({"seed": seed, "task": "analysis", "status": "failed", "error": str(exc)})

    for seed in SEEDS:
        seed_state = {
            "seed": seed, "condition": CONDITION, "dataset_version": DATA_VERSION,
            "output_dir": output_root(seed).relative_to(ROOT).as_posix(),
            "completed_checkpoints_by_dataset": {
                dataset: [model for model in MODELS if completed(seed, dataset, model)]
                for dataset in DATASETS
            },
            "missing_checkpoints": [
                {"dataset": dataset, "model": model}
                for dataset in DATASETS for model in MODELS
                if not completed(seed, dataset, model)
            ],
            "analysis_status": analyses[str(seed)],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (output_root(seed) / "inference_status.json").write_text(
            json.dumps(seed_state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    state.update({
        "queued": 0, "finished": finished, "failed": failures,
        "analysis_status_by_seed": analyses,
        "status": "complete" if not failures else "partial_with_failures",
    })
    write_state(state)
    emit({"event": "inference_queue_complete", "status": state["status"],
          "completed_models": sum(x["status"] == "completed" for x in finished),
          "failed_models": len(failures), "analysis_status_by_seed": analyses})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
