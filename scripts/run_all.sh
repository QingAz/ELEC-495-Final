#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (so paths like configs/... work)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/figures results/checkpoints results/logs

{
  echo "=== RUN ALL START: $(date) ==="
  echo "PWD: $(pwd)"
  echo "Python: $(python --version 2>&1)"
  echo ""

  python -m src.data  --config configs/base.yaml
  python -m src.eval  --config configs/base.yaml --mode dt_sweep --dt_config configs/dt_sweep.yaml
  python -m src.train --config configs/base.yaml --model resnet
  python -m src.eval  --config configs/base.yaml --model resnet
  python -m src.train --config configs/base.yaml --model neuralode
  python -m src.eval  --config configs/base.yaml --model neuralode
  python -m src.plots --config configs/base.yaml

  echo ""
  echo "=== RUN ALL END: $(date) ==="
} 2>&1 | tee results/logs/run_all.log
