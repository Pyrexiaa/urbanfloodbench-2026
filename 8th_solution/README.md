# UrbanFloodNet

A heterogeneous graph neural network for autoregressive flood prediction across coupled 1D/2D hydraulic systems. Two separate models (Model_1, Model_2) are trained on different simulation domains; their predictions are combined for submission to the [UrbanFloodBench Kaggle competition](https://www.kaggle.com/competitions/urban-flood-modelling).

**8th place on the final leaderboard.**

- **Heterogeneous GNN** with two node types (1D channels, 2D floodplain cells) and up to 10 directed edge types, encoding the full coupled hydraulic topology
- **Autoregressive rollout** — GRU-based recurrent cell predicts water levels step-by-step (up to 500 steps), using each prediction as input to the next
- **Curriculum learning** progressively extends rollout horizon (h=1 to h=256) across training to stabilize long-horizon gradients
- **Topology-aware message passing** with static coupling weights (elevation, slope, pipe geometry) gated by dynamic hidden states
- **Dual model design** — separate Model_1 (3.7K nodes) and Model_2 (4.5K nodes + global context node) trained on different simulation domains, combined at inference
- **NLCD-enriched features** — land cover (one-hot), impervious surface %, and tree canopy cover extracted from MRLC rasters at each 2D node

**[Getting Started](GETTING_STARTED.md)** — setup, training, and inference instructions.

**[Architecture](ARCHITECTURE.md)** — full technical details on the model, data, and training.

## Architecture Overview

UrbanFloodNet separates static structure from dynamic processes. A single static heterogeneous graph encodes time-invariant topology (elevation, slopes, connectivity), while water levels and rainfall are fed as dynamic time series through a recurrent cell at each timestep.

1. **Static Graph**: A `HeteroData` object with two node types (`oneD`, `twoD`) and six directed edge types. Built once and reused across all timesteps. Model_2 adds a `global` context node with four additional edge types.

2. **Recurrent Cell** (`HeteroTransportCell`): At each timestep, performs heterogeneous message passing over the static graph, combines messages with dynamic inputs (water level + rainfall), and updates per-node hidden states via GRU.

3. **Autoregressive Rollout**: Given 10 timesteps of history (teacher-forced warm start), the model predicts future timesteps by feeding each prediction forward as input to the next step.

4. **Prediction Heads**: Per-node-type MLPs decode hidden states into predicted water level deltas (`LayerNorm -> Linear -> ReLU -> Linear`).

## Repository Structure

```
UrbanFloodNet/
├── src/
│   ├── model.py                     # Model architecture (MP modules, GRU cell, AR model)
│   ├── data.py                      # Graph construction, datasets, dataloaders, rainfall features
│   ├── data_config.py               # Loads configs/data.yaml, exposes module-level constants
│   ├── data_lazy.py                 # Lazy data initialization with disk caching + parallelized preprocessing
│   ├── normalization.py             # Feature normalization (streaming min-max, meanstd, log transforms)
│   ├── train.py                     # Training loop with curriculum learning
│   └── autoregressive_inference.py  # Inference: loads both models, runs AR rollout, writes Kaggle CSV
├── configs/
│   └── data.yaml                    # Data configuration (model selection, splits, paths)
├── run/                             # Pipeline shell scripts (train -> inference -> submit)
├── slurm/                           # SLURM job scripts for HPC clusters
├── scripts/
│   ├── extract_raster_features.py   # Extract NLCD land cover + impervious surface features
│   └── scrape_shp_files.py          # Parse shapefile geometry for 1D node enrichment
├── kaggle/
│   └── submit_to_kaggle.py          # Kaggle submission helper
├── data/                            # (gitignored) Model_1/ and Model_2/ train/test data
├── checkpoints/                     # (gitignored) Saved model weights + normalizers
├── environment.yml                  # Conda environment spec
└── requirements.txt                 # Pip requirements
```

## Data Sources

- **Core data** (node/edge CSVs, event time series): Provided by the Kaggle competition.
- **NLCD raster features**: Downloaded from the [MRLC Viewer](https://www.mrlc.gov/viewer/) — NLCD 2024 Land Cover, 2024 Fractional Impervious Surface, and 2023 Tree Canopy Cover. Extracted at 2D node locations using `scripts/extract_raster_features.py`.

## Data Model

### Static Graph (`HeteroData`)

```python
data["oneD"].x_static           # [N_1d, 14] static features for 1D channel nodes
data["twoD"].x_static           # [N_2d, 28] static features for 2D floodplain cells
data["oneD", "oneDedge", "oneD"].edge_index         # 1D->1D edge connectivity
data["oneD", "oneDedge", "oneD"].edge_attr_static   # Static edge features (11 cols)
# ... similar for all 6 edge types (+ 4 global edges for Model_2)
```

### Static Node Features

**1D nodes (14 features)**:
`position_x`, `position_y`, `depth`, `invert_elevation`, `surface_elevation`, `base_area`,
`NodeType_Boundary`, `NodeType_External`, `NodeType_Junction`, `NodeType_Start`,
`has_drop_inlet`, `ConnectUS`, `ConnectDS`, `channel_2d_elev_diff`

- Node type flags and connectivity from shapefile-derived `1d_nodes_static_expanded.csv`
- `channel_2d_elev_diff` = connected 2D elevation - channel invert elevation (engineered feature)

**2D nodes (28 features)**:
`position_x`, `position_y`, `area`, `roughness`, `min_elevation`, `elevation`,
`curvature`, `flow_accumulation`, `aspect_sin`, `aspect_cos`,
16 NLCD land cover one-hot columns (`lc_11`..`lc_95`), `nlcd_fct_imp`, `nlcd_tcc_2023`

- Aspect encoded as (sin, cos) to handle circularity; flat/undefined sentinel -> (0, 0)
- NLCD features from raster extraction (land cover type, impervious %, tree canopy %)

### Static Edge Features

**1D edges (11 features)**: `relative_position_x`, `relative_position_y`, `length`, `diameter`, `shape`, `roughness`, `slope`, `USEnLoss`, `DSExLoss`, `USBFLoss`, `DSBFLoss`

**2D edges (5 features)**: `relative_position_x`, `relative_position_y`, `face_length`, `length`, `slope`

**Cross-type edges**: Model_1: zero placeholder (dim=1); Model_2: `[distance, elev_diff]` (dim=2)

### Dynamic Time Series

```python
y_hist_1d:       [B, H, N_1d, 1]                # Historical water levels (1D)
y_hist_2d:       [B, H, N_2d, 1]                # Historical water levels (2D)
rain_hist_2d:    [B, H, N_2d, RAIN_N_CHANNELS]  # Augmented rainfall history (8 channels)
rain_future_2d:  [B, T, N_2d, RAIN_N_CHANNELS]  # Future rainfall forecast
```

**Rainfall channels** (8 total): raw rainfall, cumulative mean, rolling sums over 6/12/24/36 steps (normalized by global training max), sin/cos absolute temporal encoding.

## Training

Training uses **curriculum learning** that gradually increases the autoregressive rollout horizon. Model_1 trains for 32 epochs (h=1 to h=128), Model_2 for 62 epochs (h=1 to h=256). Custom schedules are supported via `--curriculum`.

**Loss**: Equal-weight average of 1D and 2D MSE on water levels normalized by Kaggle sigma — `sqrt(MSE_norm) == NRMSE` directly.

**Stability**: Gradient clipping (max_norm=1.0), log-linear LR decay (1e-3 to ~3.16e-5), mixed precision, gradient checkpointing, LayerNorm on hidden states/messages/inputs, early stopping at max horizon.

See [GETTING_STARTED.md](GETTING_STARTED.md) for training commands and all CLI flags.

## Inference

Inference runs both Model_1 and Model_2 sequentially: loads checkpoints from `checkpoints/latest/`, warm-starts on 10 timesteps of history, autoregressively predicts remaining steps, denormalizes, and writes a Kaggle-format CSV.

See [GETTING_STARTED.md](GETTING_STARTED.md) for inference commands and flags.
