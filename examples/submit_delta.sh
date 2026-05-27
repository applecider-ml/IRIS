#!/bin/bash
#SBATCH --job-name=IRIS
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/iris_%j.out
#SBATCH --error=logs/iris_%j.err

# ============================================================
# IRIS Pipeline — Delta Job Submission
# ============================================================
#
# Usage:
#   sbatch examples/submit_delta.sh
#
#   # With custom data directory:
#   sbatch --export=DATA_DIR=/path/to/data examples/submit_delta.sh
#
#   # Skip QC (faster):
#   sbatch --export=SKIP_QC=1 examples/submit_delta.sh
#
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/projects/bcrv/asasli/IRIS}"
DATA_DIR="${DATA_DIR:-/projects/bcrv/asasli/later/Spectra}"
SCALE_METHOD="${SCALE_METHOD:-both}"
SKIP_QC="${SKIP_QC:-0}"
SKIP_FETCH="${SKIP_FETCH:-0}"
SKIP_GATHER="${SKIP_GATHER:-0}"

# ── Setup ─────────────────────────────────────────────────────
cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================"
echo "IRIS Pipeline — $(date)"
echo "============================================"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "Data dir:   $DATA_DIR"
echo "Scale:      $SCALE_METHOD"
echo "============================================"

# Load modules (Delta-specific)
module purge
module load python/3.12.7
module list

# Activate venv if exists, otherwise use pip install
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[setup] Installing IRIS package..."
    pip install --quiet -e .
fi

# ── Build arguments ───────────────────────────────────────────
ARGS="--data-dir $DATA_DIR --scale-method $SCALE_METHOD"

if [ "$SKIP_QC" = "1" ]; then
    ARGS="$ARGS --skip-qc"
fi
if [ "$SKIP_FETCH" = "1" ]; then
    ARGS="$ARGS --skip-fetch"
fi
if [ "$SKIP_GATHER" = "1" ]; then
    ARGS="$ARGS --skip-gather"
fi

# ── Run ───────────────────────────────────────────────────────
echo ""
echo "Running: python run_pipeline.py $ARGS"
echo ""

python run_pipeline.py $ARGS

echo ""
echo "============================================"
echo "IRIS Pipeline complete — $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"
