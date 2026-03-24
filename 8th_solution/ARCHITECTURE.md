# UrbanFloodNet Architecture

UrbanFloodNet is a heterogeneous graph neural network that autoregressively predicts water levels across a coupled 1D/2D flood hydraulic system. Two separate models (Model_1, Model_2) are trained on different simulation domains and their predictions are combined for the Kaggle submission.

- **Static heterogeneous graph** (built once) encodes topology, geometry, and land cover; **dynamic time series** (water levels + rainfall) are injected at each recurrent step
- **Two message-passing module types**: `StaticDynamicEdgeMP` (gated static+dynamic edges for same-type and cross-type connections) and `GATv2CrossTypeMP` (multi-head attention for global/cross-type edges)
- **GRU recurrent cell** (`HeteroTransportCell`) updates per-node hidden states each timestep from aggregated messages + dynamic inputs
- **Curriculum training** ramps rollout horizon from h=1 to h=128 (Model_1) or h=256 (Model_2), with continued training at progressively longer horizons
- **Normalization by Kaggle sigma** makes training loss directly interpretable as NRMSE in competition metric space
- **694K parameters** (Model_2) — compact enough for single-GPU training on L40S with mixed precision and gradient checkpointing

---

## Problem Formulation

Given 10 timesteps of history (water levels + rainfall), predict n timesteps (between 70 adn 500) of water level at every 1d and 2d node in the graph. Rainfall is used as an external forcing during autoregressive rollout.

**Inputs per timestep:**
- 1D nodes (channels): `[water_level, rain_channels(8), water_level_connected_2d]` — shape `[N_1d, 10]`
  - `rain_channels(8)` = rainfall features from the connected 2D node (gathered via `rain_1d_index`)
  - `water_level_connected_2d` = water level of each channel's connected 2D cell
- 2D nodes (floodplain cells): `[water_level, rain_channels(8)]` — shape `[N_2d, 9]`
- Static node/edge features: precomputed at graph construction, fixed across time

**Rainfall channels** (RAIN_N_CHANNELS=8):
| Channel | Description |
|---------|-------------|
| 0 | Raw normalized rainfall |
| 1 | Mean rainfall since event start (cumsum / (t+1)) |
| 2 | 6-step rolling sum, normalized by global training max |
| 3 | 12-step rolling sum, normalized by global training max |
| 4 | 24-step rolling sum, normalized by global training max |
| 5 | 36-step rolling sum, normalized by global training max |
| 6 | sin(t_abs / T_max) — absolute temporal encoding |
| 7 | cos(t_abs / T_max) — absolute temporal encoding |

**Static node features:**
- 1D nodes (14 features): `position_x`, `position_y`, `depth`, `invert_elevation`, `surface_elevation`, `base_area`, `NodeType_Boundary`, `NodeType_External`, `NodeType_Junction`, `NodeType_Start`, `has_drop_inlet`, `ConnectUS`, `ConnectDS`, `channel_2d_elev_diff`
  - Node type flags derived from shapefile geometry (`1d_nodes_static_expanded.csv`)
  - `channel_2d_elev_diff` = connected 2D cell elevation - channel invert elevation; captures how deeply incised the channel is relative to its floodplain (strong RMSE predictor, r=0.58)
- 2D nodes (28 features): `position_x`, `position_y`, `area`, `roughness`, `min_elevation`, `elevation`, `curvature`, `flow_accumulation`, `aspect_sin`, `aspect_cos`, 16 NLCD land cover one-hot columns (`lc_11`..`lc_95`), `nlcd_fct_imp`, `nlcd_tcc_2023`
  - Aspect encoded as (sin, cos) to handle circularity; aspect=-1 sentinel (flat/undefined) -> (0, 0)
  - NLCD features from raster extraction: land cover class (one-hot), fractional impervious surface %, tree canopy cover %. Extracted from https://www.mrlc.gov/viewer/.

**Static edge features:**
- 1D edges (11 features): `relative_position_x`, `relative_position_y`, `length`, `diameter`, `shape`, `roughness`, `slope`, `USEnLoss`, `DSExLoss`, `USBFLoss`, `DSBFLoss`
- 2D edges (5 features): `relative_position_x`, `relative_position_y`, `face_length`, `length`, `slope`

**Outputs per timestep:**
- `water_level` at every node (1D and 2D separately), in normalized units (denormalized to meters at inference)

---

## Graph Structure

The computational graph is a static `HeteroData` object with two node types and six directed edge types (plus four global edges for Model_2).

### Node Types

| Type | Count | Role |
|------|-------|------|
| `oneD` | 17 (Model_1) / 197 (Model_2) | 1D hydraulic channels |
| `twoD` | 3716 (Model_1)/ 4299 (Model_2) | 2D floodplain cells |
| `global` | 1 (Model_2 only) | Basin-wide context aggregator |

### Edge Types

| Edge | Direction | Module (Model_1) | Module (Model_2) | Purpose |
|------|-----------|-----------------|-----------------|---------|
| `oneDedge` | 1D -> 1D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Downstream channel flow |
| `oneDedgeRev` | 1D <- 1D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Backwater / reverse flow |
| `twoDedge` | 2D -> 2D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Floodplain propagation |
| `twoDedgeRev` | 2D <- 2D | `StaticDynamicEdgeMP` | `StaticDynamicEdgeMP` | Bidirectional inundation |
| `twoDoneD` | 2D -> 1D | `GATv2CrossTypeMP` | `StaticDynamicEdgeMP` | Drainage into channels |
| `oneDtwoD` | 1D -> 2D | `GATv2CrossTypeMP` | `StaticDynamicEdgeMP` | Channel overflow onto floodplain |
| `oneDglobal` | 1D -> global | — | `GATv2CrossTypeMP` | 1D state -> basin summary |
| `globaloneD` | global -> 1D | — | `GATv2CrossTypeMP` | Basin summary -> 1D nodes |
| `twoDglobal` | 2D -> global | — | `GATv2CrossTypeMP` | 2D state -> basin summary |
| `globaltwoD` | global -> 2D | — | `GATv2CrossTypeMP` | Basin summary -> 2D nodes |

**Cross-type edge features:**
- **Model_1**: Zero placeholder (dim=1) — GATv2 attention learns purely from hidden states
- **Model_2**: `[distance, elev_diff]` (dim=2, z-scored) — `elev_diff = 2D_elevation - 1D_invert_elevation`; positive = deeply incised channel. For `oneDtwoD` the sign is flipped (directional). Allows `StaticDynamicEdgeMP.base_weight` to learn static suppression of deeply incised connections.

**Global edges** (Model_2 only): Zero placeholder features (dim=1), all use `GATv2CrossTypeMP`.

Directional edge features (`relative_position_x`, `relative_position_y`, `slope`) are negated for reverse edges at graph construction time, encoding directional asymmetry.

---

## Model Architecture

### Top Level: `FloodAutoregressiveHeteroModel`

```
FloodAutoregressiveHeteroModel
├── cell: HeteroTransportCell      <- single RNN step
└── heads: ModuleDict
    ├── oneD: LayerNorm -> Linear(h_dim_1d -> head_hidden) -> ReLU -> Linear(head_hidden -> 1)
    └── twoD: LayerNorm -> Linear(h_dim_2d -> head_hidden) -> ReLU -> Linear(head_hidden -> 1)
```

The model runs `forward_unroll()`, which:
1. **Warm start** (H=10 steps): teacher-force the cell with true historical water levels to build up hidden state
2. **Autoregressive rollout** (T steps, up to forecast_len): predict -> feed prediction as input -> repeat

### HeteroTransportCell — One Timestep

For each timestep the cell performs:

| Step | Operation | Code | Output Shape |
|------|-----------|------|--------------|
| 1 | Dynamic projection | `dyn_emb[nt] = dyn_norm[nt](dyn_proj[nt](x_dyn_t[nt]))` | `[N_nt, msg_dim]` |
| 2 | Heterogeneous message passing | `msg[nt] = HeteroConv(all_edge_types)(h_t, static_graph)` | `[N_nt, msg_dim]` |
| 3 | GRU update | `h_next[nt] = h_norm[nt](GRUCell(cat(dyn_emb, msg), h_t[nt]))` | `[N_nt, h_dim_nt]` |
| 4 | Output head | `pred[nt] = head[nt](h_next[nt])` | `[N_nt, 1]` |

**Extra 1D hops** (Model_2, `num_1d_extra_hops=4`): After the initial message passing, `oneD->oneD` edges run 4 additional hops. Each hop projects normalized messages to h_dim and adds them to a proxy hidden state, then re-runs the 1D MP modules. This was added specifically for Model_2, whose 190-node 1D channel network is much larger and more spatially extended than Model_1's — a single message-passing step has limited receptive field, so multiple hops allow distant 1D nodes to influence each other within a single timestep, better capturing how water levels propagate along the channel network.

**Global context node** (Model_2 only): A single virtual node connected to all 1D and 2D nodes via dedicated edge types. This provides a complementary mechanism to the extra hops — rather than propagating information hop-by-hop along the graph, the global node aggregates and broadcasts state across the entire domain in one step, helping the model capture basin-wide dependencies in Model_2's larger spatial domain.

**Hidden state dimensions:**

| Parameter | Model_1 | Model_2 |
|-----------|---------|---------|
| `h_dim` (GRU hidden) | 96 (all types) | `oneD`: 192, `twoD`: 96, `global`: 32 |
| `msg_dim` | 64 | 64 |
| `hidden_dim` (1D homo edges) | 64 | 64 |
| `hidden_dim` (2D homo edges) | 128 | 128 |
| `hidden_dim` (cross-type edges) | 32 | 32 |
| `hidden_dim` (global edges) | — | 32 |
| `num_1d_extra_hops` | 0 | 4 |

**Normalization:** LayerNorm on hidden state (`h_norm`), aggregated messages (`msg_norm`), and dynamic projections (`dyn_norm`). Prevents magnitude explosion across long rollout horizons.

### Message Passing Modules

#### `StaticDynamicEdgeMP` (homogeneous edges, and cross-type edges in Model_2)

Computes a gated message from source to destination:

```
edge_emb  = MLP( [edge_static || src_static || dst_static] )
base_wt   = softplus( w^T edge_emb )          <- static coupling strength
dyn_gate  = sigmoid( MLP( [h_src || h_dst] ) ) <- dynamic temporal gate
payload   = MLP( h_src )                       <- message content

message   = (base_wt * dyn_gate) * payload
```

The `softplus` ensures positive coupling; the `sigmoid` gate lets the model suppress messages when the hydraulic system is inactive. Supports asymmetric h_dim for source/destination node types (`h_dim_src`, `h_dim_dst`).

#### `GATv2CrossTypeMP` (cross-type edges in Model_1; global edges in Model_2)

Wraps `torch_geometric.nn.GATv2Conv` with 4 attention heads, followed by a two-layer feed-forward network (FFN):

```
attn_out     = GATv2Conv(
                   (h_src, h_dst),
                   in_channels  = (h_dim_src, h_dim_dst),
                   out_channels = msg_dim // heads,
                   heads        = 4,
                   concat       = True,
               )                                   # [N_dst, msg_dim]

message[dst] = FFN( attn_out )                     # Linear(msg_dim->hidden_dim)->ReLU->Linear(hidden_dim->msg_dim)
```

Attention is computed jointly over source and destination hidden states, allowing the model to learn which source nodes most influence each destination. The FFN adds nonlinear expressivity that a single GATv2Conv layer lacks.

**Model_2 uses `StaticDynamicEdgeMP` for 1D<->2D cross-type edges instead**, because GATv2 cannot suppress deeply incised channels (15-34m elevation gap) using only hidden states. The `[distance, elev_diff]` edge features allow `base_weight` to learn static suppression directly from geometry. GATv2 is still used for global node edges.

---

## Normalization

### Water Level Normalization

Water levels are normalized using **meanstd normalization scaled by the Kaggle competition sigma**:

```
y_norm = (y - mean) / kaggle_sigma
```

| Model | Node Type | Kaggle sigma (m) |
|-------|-----------|-----------------|
| Model_1 | 1D | 16.878 |
| Model_1 | 2D | 14.379 |
| Model_2 | 1D | 3.192 |
| Model_2 | 2D | 2.727 |

This means `sqrt(MSE_norm) == NRMSE` directly in the competition metric space, so training loss is directly interpretable as normalized RMSE.

### Static Feature Normalization

- Streaming min-max normalization to [0, 1]
- Log transform applied to heavy-tailed features (skewness > 2.0): signed log `sign(x) * log1p(abs(x))`
- **Unified normalization** across 1D/2D for shared spatial axes: `position_x`, `position_y`, and all elevation columns share joint min/max so spatial relationships are consistent
- **Unified edge normalization** for shared features: `relative_position_x`, `relative_position_y`, `slope` use joint min/max across 1D and 2D edges

### Dynamic Feature Normalization

- Water level: meanstd with Kaggle sigma (as above)
- Rainfall: streaming min-max with optional log transform
- Rolling sum features: normalized by global max per window computed over training events

### Data Leakage Prevention

- Events are split 80/20 train/val before normalization
- All dynamic normalization statistics are computed on training events only
- Rainfall rolling-sum maxes computed on training events only

---

## Loss Function

Equal-weight average of 1D and 2D MSE:

```
loss = (MSE_1d + MSE_2d) / 2
```

Since water levels are normalized by `kaggle_sigma`, the loss directly equals `(NRMSE_1d^2 + NRMSE_2d^2) / 2`, and `sqrt(loss)` approximates the competition metric.

---

## Training

### Curriculum Learning

Training gradually increases the rollout horizon to prevent gradient explosion from backpropagating through many steps from epoch 1.

**Model_1** (4 epochs per stage, 32 total):

| Epochs | max_h |
|--------|-------|
| 1-4 | 1 |
| 5-8 | 2 |
| 9-12 | 4 |
| 13-16 | 8 |
| 17-20 | 16 |
| 21-24 | 32 |
| 25-28 | 64 |
| 29-32 | 128 |

**Model_2** (6 epochs at h=1, then 4 epochs per stage, 62 total):

| Epochs | max_h |
|--------|-------|
| 1-6 | 1 |
| 7-10 | 2 |
| 11-14 | 4 |
| 15-18 | 6 |
| 19-22 | 8 |
| 23-26 | 12 |
| 27-30 | 16 |
| 31-34 | 24 |
| 35-38 | 32 |
| 39-42 | 48 |
| 43-46 | 64 |
| 47-50 | 96 |
| 51-54 | 128 |
| 55-58 | 196 |
| 59-62 | 256 |

### Stability Measures

- **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)` every step
- **Optional Log-linear LR decay**: `lr_init=1e-3 -> lr_final=3.16e-5` over all epochs (1.5 decades)
- **Mixed precision**: `torch.amp.autocast('cuda')` + `GradScaler` (via `--mixed-precision` flag)
- **Gradient checkpointing**: Auto-enabled with `--mixed-precision`; recomputes activations during backward to avoid storing the full rollout graph in memory
- **Early stopping**: Patience=5 epochs, active only at `max_h == forecast_len`
- **LayerNorm everywhere**: On hidden state, messages, and dynamic inputs — prevents magnitude explosion over long rollout horizons

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `history_len` | 10 |
| `forecast_len` | Model_1: 128; Model_2: 256 |
| `batch_size` | 24 |
| `epochs` | Model_1: 32; Model_2: 62 |
| `lr` | 1e-3 (constant) |
| `msg_dim` | 64 |
| `dropout` | 0.0 |
| `optimizer` | Adam |
| `loss` | MSE (equal weight 1D + 2D) |

**Model-specific dimensions:**

| | Model_1 | Model_2 |
|---|---------|---------|
| `h_dim` (oneD) | 96 | 192 |
| `h_dim` (twoD) | 96 | 96 |
| `h_dim` (global) | — | 32 |
| `hidden_dim` (1D edges) | 64 | 64 |
| `hidden_dim` (2D edges) | 128 | 128 |
| `hidden_dim` (cross-type) | 32 | 32 |
| `num_1d_extra_hops` | 0 | 4 |

Model_2 gives 1D nodes a larger hidden size (192 vs 96) because its ~190-node channel network is the hard bottleneck. Cross-type edges use hidden_dim=32 because there are only ~170 connections — a smaller MLP avoids overfitting. 2D homogeneous edges use 128 because the 2D mesh has thousands of edges.

---

## Inference

Inference is fully autoregressive over all prediction steps with rainfall forcing (no teacher forcing on outputs). The `autoregressive_inference.py` script:

1. Iterates over both Model_1 and Model_2, reloading data and normalizers for each
2. Loads checkpoint from `checkpoints/latest/` (selected by `--select` policy or explicit `--model{1,2}-ckpt`)
3. Loads model-specific normalizers from `checkpoints/latest/Model_{id}_normalizers.pkl`
4. For each test event: runs `autoregressive_rollout_both()` with B=1
5. Denormalizes predictions: `y = y_norm * sigma + mean`, then clamps `water_level >= 0`
6. Matches predictions to `sample_submission.csv` format (key: model_id, event_id, node_type, node_id, step_idx)
7. Writes combined Kaggle submission CSV

Checkpoint selection supports two automatic policies, or explicit checkpoint paths via `--model1-ckpt`/`--model2-ckpt`:
- `val_loss` (default): Scan all `.pt` files, pick lowest `val_loss` in metadata
- `latest_epoch`: Use the highest-numbered epoch checkpoint

---

## Data Layout

```
data/{SELECTED_MODEL}/
├── train/
│   ├── 1d_nodes_static.csv             # Static node features for 1D nodes (7 base features)
│   ├── 1d_nodes_static_expanded.csv    # + shapefile-derived features (node types, connectivity)
│   ├── 2d_nodes_static.csv             # Static node features for 2D nodes (10 features)
│   ├── 2d_nodes_raster_features.csv    # NLCD: land cover, impervious %, tree canopy %
│   ├── 1d_edges_static.csv             # Static edge features for 1D->1D edges
│   ├── 2d_edges_static.csv             # Static edge features for 2D->2D edges
│   ├── 1d_edge_index.csv              # Connectivity for 1D->1D edges
│   ├── 2d_edge_index.csv              # Connectivity for 2D->2D edges
│   ├── 1d2d_connections.csv           # Connectivity for cross-type edges
│   └── event_{i}/
│       ├── 1d_nodes_dynamic_all.csv    # Time series: water_level per 1D node
│       └── 2d_nodes_dynamic_all.csv    # Time series: water_level + rainfall per 2D node
├── test/
│   └── event_{i}/
│       ├── 1d_nodes_dynamic_all.csv
│       └── 2d_nodes_dynamic_all.csv
└── {Model} Rasters/                    # NLCD source rasters (LndCov, FctImp, TCC 2023)
```

**Data sources:**
- Core data (CSVs, events): Kaggle competition
- NLCD rasters: [MRLC Viewer](https://www.mrlc.gov/viewer/) — Land Cover 2024, Fractional Impervious Surface 2024, Tree Canopy Cover 2023

Events are split 80/20 train/val by `configs/data.yaml` (seed=42, no held-out test set). During the fine-tune stage, all events (train+val) are used.

Preprocessed data is cached to `data/{SELECTED_MODEL}/.cache/` for fast subsequent loads.

---

## Configuration

Data configuration lives in `configs/data.yaml` and is loaded by `src/data_config.py`:

```yaml
selected_model: "Model_1"     # or "Model_2"; overridable via SELECTED_MODEL env var
data_folder: "data"
max_events: -1                 # -1 = all events
validation_split: 0.2
test_split: 0.0
random_seed: 42
```

All downstream code imports from `data_config` as before (`from data_config import SELECTED_MODEL, ...`).

---

## Pipeline

```
run/pipeline.sh [GPU_ID] [Model_1|Model_2|all]
  Stage 1: Training          (src/train.py, models sequentially, Model_2 first)
  Stage 2: Inference         (src/autoregressive_inference.py, both models combined)
  Stage 3: Architecture snapshot (run/snapshot_arch.sh)
  Stage 4: Kaggle submission (kaggle/submit_to_kaggle.py)

run/pipeline_finetune_submit.sh [GPU_ID] [Model_1|Model_2|all]
  Stage 1: Fine-tune on train+val (--no-val, --max-h 64, reduced LR)
  Stage 2-4: same as above

run/pipeline_transfer.sh [GPU_ID]
  Stage 1a: Train Model_1 from scratch
  Stage 1b: Train Model_2 warm-started from Model_1 weights (--pretrain)
  Stage 2-4: Inference, snapshot, Kaggle submission

run/pipeline_inference.sh [GPU_ID] [all] [--select POLICY]
  Stage 1: Inference only (no training)
  Stage 2: Kaggle submission
```

SLURM jobs: `sbatch slurm/submit_slurm.sh` (training) and `sbatch slurm/submit_inference_slurm.sh` (inference+submit).

### Training Order

When running both models (`pipeline.sh all`), **Model_2 trains first**. This is intentional — Model_2 is harder (higher-resolution domain, stricter Kaggle metric) and we want early visibility into its performance during a run.

---

## Feature Engineering Scripts

- **`scripts/extract_raster_features.py`**: Extracts NLCD land cover, fractional impervious surface, and tree canopy cover (2023) from raster data downloaded via [MRLC Viewer](https://www.mrlc.gov/viewer/). Land cover is one-hot encoded against a fixed 16-class NLCD vocabulary.
- **`scripts/scrape_shp_files.py`**: Parses shapefile geometry to derive expanded 1D node features (node types, connectivity flags).

