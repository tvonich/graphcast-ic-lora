#!/bin/bash
# Run the example LoRA-only optimization on one GPU.
#
# Usage:  bash run_lora_only.sh [path/to/config.json]
#
# Relative paths (config, params/, stats/, data/, outputs/) are resolved from the
# repository root. Override the weight/statistics locations with
# GRAPHCAST_PARAMS_PATH and GRAPHCAST_STATS_PATH if they live elsewhere.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

export j_path="$(realpath "${1:-configs/lora_only_2017-01-01T12.json}")"
export GRAPHCAST_PARAMS_PATH="${GRAPHCAST_PARAMS_PATH:-params/params_GraphCast_small.npz}"
export GRAPHCAST_STATS_PATH="${GRAPHCAST_STATS_PATH:-stats}"
# Put the bundled (modified) graphcast/ package ahead of any installed copy.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

python -u src/optimize_ic_lora.py
