#!/bin/bash
#SBATCH --job-name=inference_top5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Runs the dummy-inference benchmark for the top-5 solutions sequentially.
# Each solution's inference.py writes inference_metrics.json / .txt into its own folder.
#
# Submit with:  sbatch run_top5_inference.sh
# (or run directly on a node:  bash run_top5_inference.sh)

source ~/anaconda3/etc/profile.d/conda.sh

# Resolve the main folder (this script's directory) so paths work regardless
# of where sbatch is launched from.
MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MAIN_DIR"
mkdir -p logs

# Solutions in rank order.
SOLUTIONS=(1st_solution 2nd_solution 3rd_solution 4th_solution 5th_solution)

# Conda environment per solution. The 1st solution is JAX/Keras; the rest are
# PyTorch. Adjust the names to match your environments (e.g. create a jax_env,
# or set every entry to pytorch_env if a single env has all dependencies).
declare -A ENVS=(
  [1st_solution]=jax_env
  [2nd_solution]=pytorch_env
  [3rd_solution]=pytorch_env
  [4th_solution]=pytorch_env
  [5th_solution]=pytorch_env
)

# Optional per-solution CLI args (3rd/4th/5th accept argparse flags). Leave
# blank to use each script's defaults.
declare -A ARGS=(
  [1st_solution]=""
  [2nd_solution]=""
  [3rd_solution]=""
  [4th_solution]=""
  [5th_solution]=""
)

overall_status=0

for sol in "${SOLUTIONS[@]}"; do
  echo "==================================================================="
  echo ">>> [$(date '+%F %T')] Running inference for ${sol}"
  echo "==================================================================="

  env_name="${ENVS[$sol]:-pytorch_env}"
  if ! conda activate "$env_name" 2>/dev/null; then
    echo "!!! Could not activate conda env '${env_name}' for ${sol} — skipping."
    overall_status=1
    continue
  fi

  # Run inside the solution folder so inference_metrics.json is written there
  # and any relative paths resolve correctly.
  (
    cd "$MAIN_DIR/$sol" && srun python inference.py ${ARGS[$sol]}
  )
  rc=$?

  conda deactivate

  if [ $rc -eq 0 ]; then
    echo ">>> Finished ${sol}  ->  ${sol}/inference_metrics.json"
  else
    echo "!!! ${sol} exited with code ${rc}"
    overall_status=1
  fi
done

echo "==================================================================="
echo ">>> All top-5 inference runs complete (overall status: ${overall_status})"
echo "==================================================================="
exit $overall_status
