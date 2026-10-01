#!/usr/bin/env python3
"""Persistent scheduler for the four remaining seed experiments.

Runs 144 tasks:
  pretrained_base/{seed123,seed666}
  pretrained_aug3x/{seed123,seed666}

The scheduler owns nine slots per physical GPU and launches at most one new
task every 120 seconds.  It archives interrupted partial run directories on
startup so a server reboot or service restart can safely requeue them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parents[3]
TRAINER = SCRIPT_ROOT / "common/train_one.py"
BASE_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
RUNS_ROOT = ROOT / "runs/training" / BASE_VERSION
CONDITIONS = ("pretrained_base", "pretrained_aug3x")
SEEDS = (123, 666)
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s",
    "rtdetr_l", "rtdetr_x",
)
EPOCHS = 25
INTERVAL = 120.0
SLOTS_PER_GPU = 9


def task_dir(condition: str, seed: int, dataset: str, model: str) -> Path:
    return RUNS_ROOT / condition / f"seed{seed}" / dataset / model


def task_status(condition: str, seed: int, dataset: str, model: str) -> str | None:
    path = task_dir(condition, seed, dataset, model) / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return "invalid"


def archive_partial(condition: str, seed: int, dataset: str, model: str, label: str) -> None:
    source = task_dir(condition, seed, dataset, model)
    if not source.exists():
        return
    if source.is_dir() and not any(source.iterdir()):
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = RUNS_ROOT / condition / f"seed{seed}" / f"_interrupted_{label}_{stamp}" / dataset / model
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(archive))


def reconcile_interrupted() -> list[dict]:
    archived = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            for dataset in DATASETS:
                for model in MODELS:
                    status = task_status(condition, seed, dataset, model)
                    path = task_dir(condition, seed, dataset, model)
                    if status in {"running", "failed", "invalid"}:
                        archive_partial(condition, seed, dataset, model, status)
                        archived.append({"condition": condition, "seed": seed, "dataset": dataset, "model": model, "reason": status})
                    elif status is None and path.is_dir() and any(path.iterdir()):
                        archive_partial(condition, seed, dataset, model, "untracked")
                        archived.append({"condition": condition, "seed": seed, "dataset": dataset, "model": model, "reason": "untracked"})
    return archived


def start_task(condition: str, seed: int, dataset: str, model: str, device: str, workers: int):
    run_dir = task_dir(condition, seed, dataset, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable, str(TRAINER), "--condition", condition, "--seed", str(seed),
        "--dataset", dataset, "--model", model, "--device", "0",
        "--epochs", str(EPOCHS), "--workers", str(workers),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": device,
        "GI_PHYSICAL_DEVICE": device,
        "PYTHONUNBUFFERED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    log.write(json.dumps({
        "event": "task_submitted", "condition": condition, "seed": seed,
        "dataset": dataset, "model": model, "physical_device": device,
        "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False) + "\n")
    log.flush()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log, log_path


def all_tasks() -> list[tuple[str, int, str, str]]:
    return [
        (condition, seed, dataset, model)
        for condition in CONDITIONS for seed in SEEDS
        for dataset in DATASETS for model in MODELS
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--launch-interval", type=float, default=INTERVAL)
    parser.add_argument("--gpu0-slots", type=int, default=SLOTS_PER_GPU)
    parser.add_argument("--gpu1-slots", type=int, default=SLOTS_PER_GPU)
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--scheduler-log-name", default=None)
    parser.add_argument("--immediate-refill", action="store_true", help="after the initial ramp fills both GPUs, immediately refill each released slot on that GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = all_tasks()
    if args.dry_run:
        print(json.dumps({
            "status": "PASS", "task_count": len(tasks), "conditions": CONDITIONS,
            "seeds": SEEDS, "slots_per_gpu": [args.gpu0_slots, args.gpu1_slots],
            "interval_seconds": args.launch_interval,
        }, ensure_ascii=False, indent=2))
        return 0

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    archived = reconcile_interrupted()
    queue = []
    skipped = []
    for task in tasks:
        if task_status(*task) == "completed":
            skipped.append({"condition": task[0], "seed": task[1], "dataset": task[2], "model": task[3]})
        else:
            queue.append(task)

    default_tag = "_".join(CONDITIONS) + "_" + "_".join(f"seed{s}" for s in SEEDS)
    summary_name = args.summary_name or (
        "seed123_seed666_base_aug3x_scheduler_summary.json"
        if CONDITIONS == ("pretrained_base", "pretrained_aug3x") and SEEDS == (123, 666)
        else f"{default_tag}_scheduler_summary.json"
    )
    scheduler_log_name = args.scheduler_log_name or summary_name.replace("_summary.json", ".log")
    scheduler_log = RUNS_ROOT / scheduler_log_name
    active: dict[str, list[dict]] = {"0": [], "1": []}
    finished: list[dict] = []
    preferred_device = "0"
    next_launch = time.monotonic()
    ramp_filled_once = False
    with scheduler_log.open("a", encoding="utf-8") as log:
        start_event = {
            "event": "scheduler_start", "tasks": len(queue), "skipped": len(skipped),
            "archived_interrupted": archived, "epochs": EPOCHS,
            "interval_seconds": args.launch_interval,
            "slots_per_gpu": {"0": args.gpu0_slots, "1": args.gpu1_slots},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        log.write(json.dumps(start_event, ensure_ascii=False) + "\n"); log.flush()
        print(json.dumps(start_event, ensure_ascii=False), flush=True)

        while queue or active["0"] or active["1"]:
            released_devices = []
            for device in ("0", "1"):
                remaining = []
                for item in active[device]:
                    code = item["process"].poll()
                    if code is None:
                        remaining.append(item)
                        continue
                    item["log"].write(json.dumps({
                        "event": "task_process_exit", "returncode": code,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
                    item["log"].flush(); item["log"].close()
                    result = {
                        "condition": item["condition"], "seed": item["seed"],
                        "dataset": item["dataset"], "model": item["model"],
                        "physical_device": device, "returncode": code,
                        "status": "completed" if code == 0 else "failed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    finished.append(result)
                    released_devices.append(device)
                    event = {"event": "task_finished", **result}
                    log.write(json.dumps(event, ensure_ascii=False) + "\n"); log.flush()
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                active[device] = remaining

            # During the initial ramp, preserve the configured launch interval.
            # Once both GPUs have reached capacity, a completed slot is refilled
            # immediately on that same GPU instead of waiting another interval.
            if args.immediate_refill and ramp_filled_once and queue and released_devices:
                for device in released_devices:
                    if not queue:
                        break
                    condition, seed, dataset, model = queue.pop(0)
                    process, output, log_path = start_task(condition, seed, dataset, model, device, args.workers)
                    active[device].append({
                        "condition": condition, "seed": seed, "dataset": dataset, "model": model,
                        "process": process, "log": output, "log_path": log_path,
                    })
                    ramp_filled_once = (
                        len(active["0"]) >= args.gpu0_slots
                        and len(active["1"]) >= args.gpu1_slots
                    ) or ramp_filled_once
                    event = {
                        "event": "task_started", "condition": condition, "seed": seed,
                        "dataset": dataset, "model": model, "physical_device": device,
                        "active_gpu0": len(active["0"]), "active_gpu1": len(active["1"]),
                        "remaining": len(queue), "refill": "immediate",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    log.write(json.dumps(event, ensure_ascii=False) + "\n"); log.flush()
                    print(json.dumps(event, ensure_ascii=False), flush=True)

            if queue and time.monotonic() >= next_launch:
                capacities = {"0": args.gpu0_slots - len(active["0"]), "1": args.gpu1_slots - len(active["1"])}
                available = [d for d in ("0", "1") if capacities[d] > 0]
                if available:
                    if preferred_device in available and capacities[preferred_device] == max(capacities[d] for d in available):
                        device = preferred_device
                    else:
                        device = max(available, key=lambda d: capacities[d])
                    condition, seed, dataset, model = queue.pop(0)
                    process, output, log_path = start_task(condition, seed, dataset, model, device, args.workers)
                    active[device].append({
                        "condition": condition, "seed": seed, "dataset": dataset, "model": model,
                        "process": process, "log": output, "log_path": log_path,
                    })
                    ramp_filled_once = (
                        len(active["0"]) >= args.gpu0_slots
                        and len(active["1"]) >= args.gpu1_slots
                    ) or ramp_filled_once
                    event = {
                        "event": "task_started", "condition": condition, "seed": seed,
                        "dataset": dataset, "model": model, "physical_device": device,
                    "active_gpu0": len(active["0"]), "active_gpu1": len(active["1"]),
                    "remaining": len(queue), "refill": "ramp",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    log.write(json.dumps(event, ensure_ascii=False) + "\n"); log.flush()
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    preferred_device = "1" if device == "0" else "0"
                    next_launch = time.monotonic() + args.launch_interval
            time.sleep(2.0)

        failed = [item for item in finished if item["status"] != "completed"]
        summary = {
            "status": "PASS" if not failed and len(finished) + len(skipped) == len(tasks) else "FAILED",
            "task_count": len(tasks), "skipped_completed": skipped,
            "archived_interrupted": archived, "finished": finished,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (RUNS_ROOT / summary_name).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log.write(json.dumps({"event": "scheduler_complete", **summary}, ensure_ascii=False) + "\n"); log.flush()
        print(json.dumps({"event": "scheduler_complete", **summary}, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
