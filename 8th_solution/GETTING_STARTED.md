# Getting Started

This guide walks you through setting up the environment, preparing data, training models, and running inference.

---

## 1. Environment Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate urbanfloodnet
pip install torch-geometric
```

### Option B: Pip

```bash
pip install -r requirements.txt
```

> **GPU users**: The default `requirements.txt` installs PyTorch with CUDA 11.8. If you need a different CUDA version, install PyTorch manually first following [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

---

## 2. Data Setup

Download the competition data from Kaggle and unzip it so the folder structure looks like this:

```
UrbanFloodNet/
└── data/
    ├── Model_1/
    │   ├── train/
    │   │   ├── 1d_nodes_static.csv
    │   │   ├── 2d_nodes_static.csv
    │   │   ├── 1d_edges_static.csv
    │   │   ├── 2d_edges_static.csv
    │   │   ├── 1d_edge_index.csv
    │   │   ├── 2d_edge_index.csv
    │   │   ├── 1d2d_connections.csv
    │   │   ├── event_0/
    │   │   │   ├── 1d_nodes_dynamic_all.csv
    │   │   │   └── 2d_nodes_dynamic_all.csv
    │   │   ├── event_1/
    │   │   └── ...
    │   └── test/
    │       ├── event_0/
    │       └── ...
    └── Model_2/
        ├── train/  (same structure)
        └── test/   (same structure)
```

### Enriched features

The released checkpoints were trained with enriched static features (NLCD rasters + shapefile-derived attributes). These features are **required to reproduce competition results** — loading a checkpoint without them will fail due to input dimension mismatch. If training from scratch, the pipeline will fall back gracefully to the base features if these files are absent.

**NLCD raster features:**

Pre-clipped rasters are available on the [NLCD Raster Data release](https://github.com/AriMarkowitz/UrbanFloodNet/releases/tag/v1.0-rasters). Download and place them in the expected directories, then run the extraction script:

```bash
# Download rasters
mkdir -p "data/Model_1/Model1 Rasters" "data/Model_2/Model2 Rasters"
gh release download v1.0-rasters --repo AriMarkowitz/UrbanFloodNet --pattern "Model_1_*" --dir "data/Model_1/Model1 Rasters"
gh release download v1.0-rasters --repo AriMarkowitz/UrbanFloodNet --pattern "Model_2_*" --dir "data/Model_2/Model2 Rasters"

# Extract features at 2D node locations
python scripts/extract_raster_features.py
# Creates: data/Model_{1,2}/train/2d_nodes_raster_features.csv
```

**Shapefile-derived 1D node features:**
```bash
python scripts/scrape_shp_files.py
# Creates: data/Model_{1,2}/train/1d_nodes_static_expanded.csv
```

---

## 3. Configuration

Edit `configs/data.yaml` before training:

```yaml
selected_model: "Model_1"    # Which model to train/evaluate
data_folder: "data"          # Path to the data directory
max_events: -1               # -1 = all events; set lower for quick testing
validation_split: 0.2        # 80/20 train/val split
test_split: 0.0
random_seed: 42
```

You can also override the model selection at runtime with an environment variable:

```bash
SELECTED_MODEL=Model_2 python src/train.py
```

---

## 4. Training

### Train a single model with default curriculum

```bash
# Train Model_1 (32 epochs, curriculum: h=1 -> h=128)
SELECTED_MODEL=Model_1 python src/train.py

# Train Model_2 (62 epochs, curriculum: h=1 -> h=256)
SELECTED_MODEL=Model_2 python src/train.py
```

### Train with GPU + mixed precision

```bash
CUDA_VISIBLE_DEVICES=0 SELECTED_MODEL=Model_1 python src/train.py --mixed-precision
```

### Custom curriculum

Override the built-in schedule with `--curriculum "horizon:epochs,..."`:

```bash
# Quick test: 1 epoch at h=1
SELECTED_MODEL=Model_1 python src/train.py --curriculum "1:1" --no-val

# Custom ramp: 2 epochs each at h=1,4,16,64
SELECTED_MODEL=Model_1 python src/train.py --curriculum "1:2,4:2,16:2,64:2"

# Aggressive schedule for Model_2
SELECTED_MODEL=Model_2 python src/train.py --curriculum "1:4,4:4,16:4,64:4,128:4,256:4"
```

### Resume interrupted training

```bash
SELECTED_MODEL=Model_1 python src/train.py --resume checkpoints/latest
```

### Transfer learning (Model_1 -> Model_2)

Train Model_1 first, then warm-start Model_2 from those weights:

```bash
SELECTED_MODEL=Model_1 python src/train.py --mixed-precision
SELECTED_MODEL=Model_2 python src/train.py --mixed-precision --pretrain checkpoints/latest
```

### Fine-tune on all data (train + val)

After initial training, fine-tune at the max horizon using all available data:

```bash
SELECTED_MODEL=Model_1 python src/train.py \
    --resume checkpoints/latest \
    --train-split all \
    --no-val \
    --epochs 4 \
    --lr 3e-4 \
    --max-h 128
```

### Key training flags

| Flag | Description |
|------|-------------|
| `--mixed-precision` | Enable fp16 training (recommended for GPU) |
| `--curriculum "h:n,..."` | Custom curriculum schedule |
| `--no-val` | Skip validation (faster training) |
| `--resume DIR` | Resume from checkpoint directory |
| `--pretrain DIR` | Transfer learning from another model's weights |
| `--train-split all` | Train on all data (for final fine-tuning) |
| `--epochs N` | Override epoch count (extra epochs when used with `--resume`) |
| `--lr FLOAT` | Override learning rate |
| `--max-h N` | Override max rollout horizon |
| `--batch-size N` | Override batch size |
| `--no-mirror-latest` | Don't copy checkpoints to `checkpoints/latest/` |
| `--keep-short-events` | Include events shorter than current horizon |

---

## 5. Inference

After training both models, run inference to generate a Kaggle submission:

```bash
python src/autoregressive_inference.py \
    --checkpoint-dir checkpoints \
    --output submission.csv \
    --sample path/to/sample_submission.csv
```

The script automatically loads both Model_1 and Model_2 from `checkpoints/latest/` and combines their predictions.

### Inference flags

| Flag | Description |
|------|-------------|
| `--checkpoint-dir DIR` | Base checkpoint directory (default: `checkpoints`) |
| `--model1-dir DIR` | Override checkpoint dir for Model_1 |
| `--model2-dir DIR` | Override checkpoint dir for Model_2 |
| `--model1-ckpt FILE` | Use a specific `.pt` file for Model_1 |
| `--model2-ckpt FILE` | Use a specific `.pt` file for Model_2 |
| `--select POLICY` | Checkpoint selection: `val_loss` (default) or `latest_epoch` |
| `--output FILE` | Output CSV path (default: `submission.csv`) |
| `--sample FILE` | Path to `sample_submission.csv` |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, or `mps` |
| `--max-events N` | Limit events processed (for testing) |

---

## 6. Pipeline Scripts

For convenience, shell scripts in `run/` chain training and inference together. These are not required — you can always run `train.py` and `autoregressive_inference.py` directly.

```bash
# Full pipeline: train both models -> inference -> submit
bash run/pipeline.sh [GPU_ID] [Model_1|Model_2|all]

# Fine-tune on train+val -> inference -> submit
bash run/pipeline_finetune_submit.sh [GPU_ID] [Model_1|Model_2|all]

# Transfer learning: train Model_1, warm-start Model_2 -> inference
bash run/pipeline_transfer.sh [GPU_ID]

# Inference only (requires existing checkpoints)
bash run/pipeline_inference.sh [GPU_ID] [all] [--select val_loss]
```

SLURM users can submit these as jobs via `slurm/submit_slurm.sh`.

---

## Quick Smoke Test

To verify everything works before a full training run:

```bash
# 1. Test data loading for Model_1
SELECTED_MODEL=Model_1 python -c "
import sys; sys.path.insert(0, 'src')
from data import get_recurrent_dataloader, get_model_config
data = get_model_config()
dl = get_recurrent_dataloader(history_len=10, forecast_len=4, batch_size=2, shuffle=True, split='train')
batch = next(iter(dl))
print('Model_1 data OK:', {k: v.shape for k, v in batch.items() if hasattr(v, 'shape')})
"

# 2. Train 1 epoch at h=1 with no validation
SELECTED_MODEL=Model_1 python src/train.py --curriculum "1:1" --no-val --no-mirror-latest
```

---

## Checkpoints

Training saves checkpoints to `checkpoints/<run_name>/` and mirrors the latest to `checkpoints/latest/`. The inference script looks in `checkpoints/latest/` by default.

```
checkpoints/
├── latest/
│   ├── Model_1_best.pt
│   ├── Model_1_normalizers.pkl
│   ├── Model_2_best.pt
│   └── Model_2_normalizers.pkl
└── Model_1_20260316_184603/
    ├── Model_1_epoch_001.pt
    ├── Model_1_epoch_002.pt
    ├── Model_1_best.pt
    └── ...
```

---

## Logging

Training logs to [Weights & Biases](https://wandb.ai). Set `WANDB_MODE=disabled` to skip:

```bash
WANDB_MODE=disabled SELECTED_MODEL=Model_1 python src/train.py --curriculum "1:1" --no-val
```

---

## Reproduce Competition Results

Pre-trained checkpoints for our top Kaggle submissions are available on the [GitHub Releases page](https://github.com/AriMarkowitz/UrbanFloodNet/releases/tag/v1.0-top3).

### 1. Download data

Download the competition data from the [UrbanFloodBench Kaggle competition](https://www.kaggle.com/competitions/urban-flood-modelling) and set up the `data/` directory as described in [Data Setup](#2-data-setup) above.

### 2. Download checkpoints

```bash
mkdir -p checkpoints/released
cd checkpoints/released
gh release download v1.0-top3 --repo AriMarkowitz/UrbanFloodNet
cd ../..
```

### 3. Run inference

```bash
python src/autoregressive_inference.py \
    --model1-ckpt checkpoints/released/Model_1_epoch_032.pt \
    --model1-dir checkpoints/released \
    --model2-ckpt checkpoints/released/Model_2_ep66_h256.pt \
    --model2-dir checkpoints/released \
    --output submission.csv \
    --sample data/Model_1/test/sample_submission.csv
```

The `--model{1,2}-dir` flags point to the directory containing the corresponding `*_normalizers.pkl` files, which are required for denormalization.

Alternatively, copy the checkpoints and normalizers into `checkpoints/latest/` and use the pipeline scripts:

```bash
bash run/pipeline_inference.sh
```

> **Note:** The pipeline scripts (`run/pipeline*.sh`) include an automatic Kaggle submission step at the end. Set `SKIP_SUBMIT=1` to disable it, or remove the `--yes` flag to require confirmation before submitting.

### Available checkpoints

See the [Releases page](https://github.com/AriMarkowitz/UrbanFloodNet/releases/tag/v1.0-top3) for download links.

| File | Model | Epoch | Forecast Horizon | Public Score |
|------|-------|-------|-------------------|--------------|
| `Model_2_ep66_h256.pt` | Model 2 | 66 | 256 | 0.0227 |
| `Model_2_ep61_h192.pt` | Model 2 | 61 | 192 | 0.0237 |
| `Model_2_ep58_h192.pt` | Model 2 | 58 | 192 | 0.0244 |
| `Model_1_epoch_032.pt` | Model 1 | 32 | 128 | — |
