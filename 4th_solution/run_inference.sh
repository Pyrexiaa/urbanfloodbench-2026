#!/bin/bash
#SBATCH --job-name=inference_4th_solution
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_env

srun python inference.py

