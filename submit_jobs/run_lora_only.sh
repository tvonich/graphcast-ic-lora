#!/bin/bash -l
# =============================================================================
# GraphCast LoRA-only optimization: run script (plain bash or PBS)
#
# Optimizes a rank-2 LoRA correction to the GraphCast_small processor weights so
# that the forecast started at 2017-01-01 12 UTC fits ERA5 over a 10-day window
# (configs/lora_only_2017-01-01T12.json). One GPU; about 25 minutes on an A100.
#
# USAGE (from the repository root):
#   bash submit_jobs/run_lora_only.sh [path/to/config.json]   # any machine with a GPU
#   mkdir -p logs && qsub submit_jobs/run_lora_only.sh        # PBS (e.g. NCAR Derecho)
#   qsub -v RUN_CONFIG=path/to/config.json submit_jobs/run_lora_only.sh
#
# With qsub, submit from the repository root (PBS runs a copy of this script, so
# the repository is found through the submission directory) and set your project
# code in the #PBS -A line below or with qsub -A. PBS logs go to logs/.
#
# Paths: a config path is taken relative to the current directory (with qsub,
# the submission directory). Relative paths inside the config (dataset_file_path,
# save_root) and the default params/ and stats/ locations are taken relative to
# the repository root.
# =============================================================================

#PBS -N graphcast_ic_lora
#PBS -A YOUR_PROJECT_CODE_HERE
#PBS -l select=1:ncpus=4:ngpus=1:mem=120GB:gpu_type=a100
#PBS -l walltime=00:30:00
#PBS -q develop
#PBS -o logs
#PBS -j oe

set -eo pipefail

# Locate the repository: next to this script, or (under qsub, where this script
# runs from PBS's spool directory) the submission directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HERE/../batch_modules/optimize_ic_lora.py" ]; then
  REPO="$(cd "$HERE/.." && pwd)"
else
  REPO="${PBS_O_WORKDIR:-}"
  [ -n "$REPO" ] && cd "$REPO"
fi
[ -f "$REPO/batch_modules/optimize_ic_lora.py" ] || {
  echo "Repository not found (with qsub, submit from the repository root)." >&2; exit 1; }

# =============================================================================
# USER CONFIGURATION (edit here, or set these variables in the environment)
# =============================================================================
# Run configuration: first argument, else RUN_CONFIG, else the example.
RUN_CONFIG="${1:-${RUN_CONFIG:-$REPO/configs/lora_only_2017-01-01T12.json}}"
# Conda environment created from environment.yml (name or path).
# Set CONDA_ENV="" to use the currently active Python environment as is.
CONDA_ENV="${CONDA_ENV-graphcast-ic-lora}"
# GraphCast weights and normalization statistics (see README, Setup step 2).
export GRAPHCAST_PARAMS_PATH="${GRAPHCAST_PARAMS_PATH:-params/params_GraphCast_small.npz}"
export GRAPHCAST_STATS_PATH="${GRAPHCAST_STATS_PATH:-stats}"
# =============================================================================

# Resolve the config path before changing directory.
CONFIG="$(realpath "$RUN_CONFIG")"
[ -f "$CONFIG" ] || { echo "Config file not found: $CONFIG" >&2; exit 1; }
cd "$REPO"

if [ -n "$CONDA_ENV" ]; then
  # On NCAR systems 'module load conda' provides conda; elsewhere conda must be on PATH.
  module load conda 2>/dev/null || true
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi
set -u

# Stop if JAX cannot see a GPU (it would otherwise fall back to the CPU and run
# for days). The usual cause is a missing CUDA 12 toolkit; see environment.yml.
python -c 'import jax, sys; sys.exit(0 if jax.default_backend() == "gpu" else 1)' || {
  echo "JAX found no GPU; see the CUDA note at the top of environment.yml." >&2; exit 1; }

export j_path="$CONFIG"
# Put the bundled (modified) graphcast/ package ahead of any installed copy.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

start_time=$(date +%s)
python -u batch_modules/optimize_ic_lora.py
elapsed=$(( $(date +%s) - start_time ))
echo "Total time: $((elapsed / 60)) min $((elapsed % 60)) s"
