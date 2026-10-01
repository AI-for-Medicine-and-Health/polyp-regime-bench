#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source /apps/python/etc/profile.d/conda.sh
conda activate AI-Exp1
cd "${ROOT}"

LOG="${ROOT}/runs/inference/kvasir_seg__polypgen_wli__polypdb_wli__seed42/pretrained_repeat5x_inference_scheduler.log"
mkdir -p "$(dirname "${LOG}")"
exec python "${SCRIPT_DIR}/common/run_pretrained_aug10x_inference.py" \
  --condition pretrained_repeat5x --batch 128 --workers-gpu0 3 --workers-gpu1 4 2>&1 | tee -a "${LOG}"
