#!/usr/bin/env python3
"""Stagger 36 seed-88 pretrained-base jobs across two GPUs.

At most six jobs run on each physical GPU. A new job is submitted every 120
seconds, including after a slot becomes available, so CUDA initialization is
not burst-launched for all twelve concurrent slots.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parents[3]
TRAINER = SCRIPT_ROOT / "common/train_one.py"
RUN_ROOT = ROOT / "runs/training/kvasir_seg__polypgen_wli__polypdb_wli__seed42/pretrained_base/seed88"
DATASETS = ("kvasir_seg", "polypgen_wli", "polypdb_wli")
MODELS = (
    "fasterrcnn_resnet50_fpn", "yolo11_s", "yolov5_s", "yolov8_s", "yolov9_s",
    "yolov3_tinyu", "yolov3_sppu", "yolov10_s", "yolo12_s", "yolo26_s",
    "rtdetr_l", "rtdetr_x",
)
EPOCHS = 25
INTERVAL = 120.0
SLOTS_PER_GPU = 6


def task_status(dataset: str, model: str) -> str | None:
    path = RUN_ROOT / dataset / model / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return None


def start_task(dataset: str, model: str, device: str, workers: int) -> tuple[subprocess.Popen, object, Path]:
    run_dir = RUN_ROOT / dataset / model
    log_path = run_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable, str(TRAINER), "--dataset", dataset, "--model", model,
        "--device", "0", "--epochs", str(EPOCHS), "--workers", str(workers),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": device,
        "GI_PHYSICAL_DEVICE": device,
        "PYTHONUNBUFFERED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    log.write(json.dumps({
        "event": "task_submitted", "dataset": dataset, "model": model,
        "physical_device": device, "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False) + "\n")
    log.flush()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log, log_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--launch-interval", type=float, default=INTERVAL)
    parser.add_argument("--gpu0-slots", type=int, default=SLOTS_PER_GPU)
    parser.add_argument("--gpu1-slots", type=int, default=SLOTS_PER_GPU)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = [(dataset, model) for dataset in DATASETS for model in MODELS]
    if args.dry_run:
        print(json.dumps({"status": "PASS", "task_count": len(tasks), "tasks": tasks, "interval_seconds": args.launch_interval}, ensure_ascii=False, indent=2))
        return 0

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    coordinator_log = RUN_ROOT / "scheduler.log"
    active: dict[str, list[dict]] = {"0": [], "1": []}
    finished: list[dict] = []
    queue: list[tuple[str, str]] = []
    skipped: list[dict] = []
    for dataset, model in tasks:
        status = task_status(dataset, model)
        if status == "completed":
            skipped.append({"dataset": dataset, "model": model, "status": "skipped_completed"})
        else:
            queue.append((dataset, model))

    next_launch = time.monotonic()
    preferred_device = "0"
    with coordinator_log.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "scheduler_start", "tasks": len(queue), "skipped": len(skipped), "epochs": EPOCHS, "interval_seconds": args.launch_interval, "slots_per_gpu": {"0": args.gpu0_slots, "1": args.gpu1_slots}, "started_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
        log.flush()
        while queue or active["0"] or active["1"]:
            for device in ("0", "1"):
                remaining = []
                for item in active[device]:
                    code = item["process"].poll()
                    if code is None:
                        remaining.append(item)
                        continue
                    item["log"].write(json.dumps({"event": "task_process_exit", "returncode": code, "finished_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    item["log"].flush()
                    item["log"].close()
                    result = {"dataset": item["dataset"], "model": item["model"], "physical_device": device, "returncode": code, "status": "completed" if code == 0 else "failed", "finished_at": datetime.now(timezone.utc).isoformat()}
                    finished.append(result)
                    log.write(json.dumps({"event": "task_finished", **result}, ensure_ascii=False) + "\n")
                    log.flush()
                    print(json.dumps({"event": "task_finished", **result}, ensure_ascii=False), flush=True)
                active[device] = remaining

            now = time.monotonic()
            if queue and now >= next_launch:
                capacities = {"0": args.gpu0_slots - len(active["0"]), "1": args.gpu1_slots - len(active["1"])}
                available = [device for device in ("0", "1") if capacities[device] > 0]
                if available:
                    if preferred_device in available and capacities[preferred_device] == max(capacities[d] for d in available):
                        device = preferred_device
                    else:
                        device = max(available, key=lambda d: capacities[d])
                    dataset, model = queue.pop(0)
                    process, output, log_path = start_task(dataset, model, device, args.workers)
                    active[device].append({"dataset": dataset, "model": model, "process": process, "log": output, "log_path": log_path})
                    event = {"event": "task_started", "dataset": dataset, "model": model, "physical_device": device, "active_gpu0": len(active["0"]), "active_gpu1": len(active["1"]), "remaining": len(queue), "started_at": datetime.now(timezone.utc).isoformat()}
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    log.flush()
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    preferred_device = "1" if device == "0" else "0"
                    next_launch = time.monotonic() + args.launch_interval
            time.sleep(2.0)

        summary = {"status": "PASS" if all(item["status"] == "completed" for item in finished) else "FAILED", "epochs": EPOCHS, "task_count": len(tasks), "skipped_completed": skipped, "finished": finished, "completed_at": datetime.now(timezone.utc).isoformat()}
        (RUN_ROOT / "scheduler_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.write(json.dumps({"event": "scheduler_complete", **summary}, ensure_ascii=False) + "\n")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
