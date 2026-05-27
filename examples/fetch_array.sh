#!/bin/bash
#SBATCH --job-name=iris_array
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/iris_%A_%a.out
#SBATCH --error=logs/iris_%A_%a.err

# ============================================================
# IRIS — Parallel fetch with SLURM array jobs
# ============================================================
#
# Splits an ID list across N array tasks, each fetching its
# own slice into a separate output directory.
#
# Usage:
#   # Set your token
#   export FRITZ_TOKEN="your-token-here"
#
#   # Submit 10 parallel fetch jobs
#   sbatch --export=ALL examples/fetch_array.sh
#
#   # Custom number of slices and ID file
#   sbatch --export=ALL --array=0-4 examples/fetch_array.sh
#
#   # Custom output location
#   sbatch --export=ALL,OUTPUT_BASE=/scratch/mydata examples/fetch_array.sh
#
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/projects/bcrv/asasli/IRIS}"
ID_FILE="${ID_FILE:-}"
OUTPUT_BASE="${OUTPUT_BASE:-fritz_data}"
MAX_WORKERS="${MAX_WORKERS:-1}"
REQUEST_DELAY="${REQUEST_DELAY:-3}"

# Number of slices = size of the SLURM array
N_SLICES="${SLURM_ARRAY_TASK_COUNT:-10}"

# ── Setup ─────────────────────────────────────────────────────
cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================"
echo "IRIS Array Fetch — $(date)"
echo "============================================"
echo "Node:        $(hostname)"
echo "Array task:  ${SLURM_ARRAY_TASK_ID} / ${N_SLICES}"
echo "Output:      ${OUTPUT_BASE}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# Load modules
module purge
module load python/3.12.7

# Activate venv or install
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    pip install --quiet -e .
fi

# ── Build command ─────────────────────────────────────────────
CMD="python examples/fetch_spectra.py"
CMD="$CMD --output-dir ${OUTPUT_BASE}_${SLURM_ARRAY_TASK_ID}"
CMD="$CMD --slice-index $SLURM_ARRAY_TASK_ID"
CMD="$CMD --n-slices $N_SLICES"
CMD="$CMD --max-workers $MAX_WORKERS"
CMD="$CMD --request-delay $REQUEST_DELAY"

if [ -n "$ID_FILE" ]; then
    CMD="$CMD --ids-file $ID_FILE"
fi

echo ""
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "============================================"
echo "IRIS Array Fetch complete — $(date)"
echo "Task ${SLURM_ARRAY_TASK_ID} done."
echo "============================================"
