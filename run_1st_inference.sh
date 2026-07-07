#!/bin/bash
#SBATCH --job-name=inference_1st
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --time=1440
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Runs the dummy-inference benchmark for the 1st solution only, with more CPU.
#   8 CPUs x 16 GB = 128 GB total host RAM.
# The solution's inference.py writes inference_metrics.json / .txt into its folder.
#
# Submit with:  sbatch run_1st_inference.sh
# (or run directly on a node:  bash run_1st_inference.sh)

source ~/anaconda3/etc/profile.d/conda.sh

# Resolve the main folder so paths work regardless of how the script is
# launched. Under sbatch the script is copied to Slurm's spool dir, so
# ${BASH_SOURCE[0]} is useless there — prefer $SLURM_SUBMIT_DIR (the dir you
# ran sbatch from) and fall back to the script's own dir for plain `bash` runs.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  MAIN_DIR="$SLURM_SUBMIT_DIR"
else
  MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$MAIN_DIR" || { echo "!!! Cannot cd to $MAIN_DIR"; exit 1; }
mkdir -p logs

sol="1st_solution"
env_name="jax_env"
args=""

echo "==================================================================="
echo ">>> [$(date '+%F %T')] Running inference for ${sol}"
echo "==================================================================="

if ! conda activate "$env_name" 2>/dev/null; then
  echo "!!! Could not activate conda env '${env_name}' for ${sol}."
  exit 1
fi

# --- GPU memory behaviour (JAX/XLA) ---
# Don't preallocate the whole GPU, cap the usable fraction, and use the async
# allocator to avoid fragmentation-driven OOMs. These are also set as defaults
# inside inference.py; exporting here lets you override without editing code.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Run inside the solution folder so inference_metrics.json is written there
# and any relative paths resolve correctly.
(
  cd "$MAIN_DIR/$sol" && srun python inference.py ${args}
)
rc=$?

conda deactivate

if [ $rc -eq 0 ]; then
  echo ">>> Finished ${sol}  ->  ${sol}/inference_metrics.json"
else
  echo "!!! ${sol} exited with code ${rc}"
fi

echo "==================================================================="
echo ">>> 1st solution inference complete (status: ${rc})"
echo "==================================================================="
exit $((rc))
