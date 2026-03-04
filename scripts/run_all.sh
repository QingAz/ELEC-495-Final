#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/base.yaml
python -m src.eval --config configs/base.yaml --dt-config configs/dt_sweep.yaml
