#!/usr/bin/env python3
"""Retry the failed aug10x seed123 PolypDB Faster R-CNN task safely.

The main aug10x scheduler has no queued work left, so this process waits for
one of its 9-per-GPU slots to become free before starting the retry. This
avoids interrupting or overfilling the existing scheduler.
"""
from __future__ import annotations

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
BASE_VERSION = "kvasir_seg__polypgen_wli__polypdb_wli__seed42"
RUNS_ROOT = ROOT / "runs/training" / BASE_VERSION
TASK_ROOT = RUNS_ROOT / "pretrained_aug10x/seed123/polypdb_wli/fasterrcnn_resnet50_fpn"
SEED_ROOT = RUNS_ROOT / "pretrained_aug10x/seed123"
MAIN_LOG = RUNS_ROOT / "seed123_seed666_aug10x_scheduler.log"
SUMMARY = RUNS_ROOT / "seed123_seed666_aug10x_retry_summary.json"
TRAINER = SCRIPT_ROOT / "common/train_one.py"
CONDITION = "pretrained_aug10x"
SEED = 123
DATASET = "polypdb_wli"
MODEL = "fasterrcnn_resnet50_fpn"
MAX_SLOTS = 9


def read_scheduler_state() -> tuple[dict[str, int], int]:
    active: dict[tuple[str, int, str, str], str] = {}
    remaining = 10**9
    if MAIN_LOG.is_file():
        for line in MAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (event.get("condition"), event.get("seed"), event.get("dataset"), event.get("model"))
            event_name = event.get("event")
            if event_name == "task_started" and event.get("physical_device") in {"0", "1"}:
                active[key] = str(event["physical_device"])
                remaining = int(event.get("remaining", remaining))
            elif event_name == "task_finished":
                active.pop(key, None)
    counts = {"0": 0, "1": 0}
    for device in active.values():
        counts[device] += 1
    return counts, remaining


def archive_failed_output() -> Path | None:
    if not TASK_ROOT.exists():
        return None
    status_path = TASK_ROOT / "status.json"
    if not status_path.is_file():
        raise RuntimeError(f"retry target has no status.json: {TASK_ROOT}")
    status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
    if status == "completed":
        return None
    if status != "failed":
        raise RuntimeError(f"retry target is not failed: status={status!r}, path={TASK_ROOT}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = SEED_ROOT / f"_retry_archive_{stamp}" / DATASET / MODEL
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(TASK_ROOT), str(archive))
    return archive


def main() -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    archived = archive_failed_output()
    if archived is None and TASK_ROOT.exists():
        status = json.loads((TASK_ROOT / "status.json").read_text(encoding="utf-8")).get("status")
        if status == "completed":
            SUMMARY.write_text(json.dumps({"status": "SKIP_COMPLETED", "task": str(TASK_ROOT)}, indent=2) + "\n", encoding="utf-8")
            return 0

    selected_device = None
    while selected_device is None:
        counts, remaining = read_scheduler_state()
        if remaining <= 0:
            available = [device for device in ("0", "1") if counts[device] < MAX_SLOTS]
            if available:
                selected_device = min(available, key=lambda device: counts[device])
                break
        print(json.dumps({"event": "retry_wait", "active_slots": counts, "main_queue_remaining": remaining}), flush=True)
        time.sleep(30)

    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = TASK_ROOT / "train.log"
    log = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable, str(TRAINER), "--condition", CONDITION, "--seed", str(SEED),
        "--dataset", DATASET, "--model", MODEL, "--device", "0", "--epochs", "25", "--workers", "2",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": selected_device,
        "GI_PHYSICAL_DEVICE": selected_device,
        "PYTHONUNBUFFERED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    log.write(json.dumps({
        "event": "retry_task_submitted", "condition": CONDITION, "seed": SEED,
        "dataset": DATASET, "model": MODEL, "physical_device": selected_device,
        "archived_failed_output": str(archived) if archived else None,
        "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False) + "\n")
    log.flush()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = process.wait()
    log.close()
    result = {
        "status": "completed" if code == 0 else "failed", "returncode": code,
        "condition": CONDITION, "seed": SEED, "dataset": DATASET, "model": MODEL,
        "physical_device": selected_device, "archived_failed_output": str(archived) if archived else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
