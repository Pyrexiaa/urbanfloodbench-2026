# Top 5 Solutions — Inference Guide

This guide walks you through setting up the correct conda environments and running inference for all 5 solutions.

---

## Overview

| Solution | Folder | Environment | Framework |
|---|---|---|---|
| 1st Solution | `1st_solution/` | `jax_env` | JAX |
| 2nd – 5th Solution | `2nd_solution/` … `5th_solution/` | `pytorch_env` | PyTorch (GPU) |

---

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- CUDA 12.8 compatible GPU (required for PyTorch solutions)
- SLURM workload manager (for `sbatch` job submission)

---

## Environment Setup

### 1st Solution — JAX Environment

```bash
# Create the environment
conda create --name jax_env python=3.11 -y
conda activate jax_env

# Install dependencies
cd 1st_solution
pip install -r requirements.txt
```

### 2nd–5th Solutions — PyTorch Environment

```bash
# Create the environment
conda create --name pytorch_env python=3.11 -y
conda activate pytorch_env

# Install PyTorch with CUDA 12.8 GPU support first
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies (run from any of the solution folders)
cd 2nd_solution
pip install -r requirements.txt
```

> **Note:** The PyTorch GPU installation step must be run **before** `pip install -r requirements.txt` to ensure the correct CUDA-enabled build is used.

---

## Running Inference

Each solution contains a `run_inference.sh` bash script that is submitted via SLURM's `sbatch` command.

### Steps

1. Activate the appropriate conda environment for the solution you want to run.
2. Navigate into the solution's folder.
3. Submit the job with `sbatch`.

### 1st Solution (JAX)

```bash
conda activate jax_env
cd 1st_solution
sbatch run_inference.sh
```

### 2nd Solution (PyTorch)

```bash
conda activate pytorch_env
cd 2nd_solution
sbatch run_inference.sh
```

### 3rd Solution (PyTorch)

```bash
conda activate pytorch_env
cd 3rd_solution
sbatch run_inference.sh
```

### 4th Solution (PyTorch)

```bash
conda activate pytorch_env
cd 4th_solution
sbatch run_inference.sh
```

### 5th Solution (PyTorch)

```bash
conda activate pytorch_env
cd 5th_solution
sbatch run_inference.sh
```

---

## Monitoring Jobs

After submitting with `sbatch`, you can monitor your job with standard SLURM commands:

```bash
# Check job status
squeue -u $USER

# View job output logs (replace <job_id> with the ID returned by sbatch)
cat slurm-<job_id>.out

# Cancel a job if needed
scancel <job_id>
```

---

## Quick Reference

```
# JAX (1st solution only)
conda activate jax_env

# PyTorch (2nd–5th solutions)
conda activate pytorch_env

# Submit inference job from inside solution folder
sbatch run_inference.sh
```