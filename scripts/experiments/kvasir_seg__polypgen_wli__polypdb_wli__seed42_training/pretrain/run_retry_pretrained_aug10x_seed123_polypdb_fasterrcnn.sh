#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source /apps/python/etc/profile.d/conda.sh
conda activate AI-Exp1
cd "${ROOT}"
exec python "${SCRIPT_DIR}/retry_pretrained_aug10x_seed123_polypdb_fasterrcnn.py"
