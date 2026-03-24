# Solution Write-Up: Urban Flood Forecasting in Coupled 1D-2D Hydraulic Systems

**Public Leaderboard Score: 0.0178 (Standardized RMSE)**

---

## 1. Solution Overview

Our approach is a **heterogeneous ensemble** that combines gradient boosting trees (CatBoost, LightGBM, XGBoost) with deep learning models (Pre-Norm Transformers, BiGRU with attention, GCN-augmented networks). The key insight driving our pipeline design is that the four subproblems — defined by (model_id, node_type) — have fundamentally different characteristics and benefit from distinct modeling strategies:

| Submodel | Strategy | Prediction Mode |
|----------|----------|-----------------|
| M1 NT1 (1D drainage) | Seed-blended CatBoost + LightGBM | Batch (non-autoregressive) |
| M1 NT2 (2D surface) | Seed-blended CatBoost + LightGBM | Batch (non-autoregressive) |
| M2 NT2 (2D surface) | Seed-averaged CatBoost + LightGBM + XGBoost | Batch (non-autoregressive) |
| M2 NT1 (1D drainage) | Weighted blend: 50% boosting + 25% DL main + 25% DL optional | Batch (non-autoregressive) |

The solution achieves diversity through: (i) different model families (gradient boosting vs. deep learning), (ii) different random seeds, (iii) different neural architectures (Transformer, BiGRU, GCN), and (iv) different target transformations (delta, ratio, raw).

---

## 2. Feature Engineering

Feature engineering is the backbone of our boosting models and also informs the tabular inputs to deep learning models. We construct features across four categories: **rainfall dynamics**, **warmup-derived hydraulic state**, **graph-structural properties**, and **cross-feature interactions**.

### 2.1 Rainfall Features

Since rainfall is the only dynamic input available beyond the 10 warmup timesteps, we extract a rich temporal representation:

- **Instantaneous**: current rainfall depth
- **Lagged values**: `rain_lag_{1,2,3}` and extended lags `dodepch1k_lag_{1..499}` (up to 499 past steps)
- **Rolling aggregations**: `rain_roll_{3,6,12,24}` (cumulative sums over sliding windows)
- **Cumulative sum**: `rain_cumsum` (total accumulated rainfall up to timestep *t*)
- **Exponential moving averages**: `rain_ema_{10,20,50,100}` at multiple decay rates
- **Trend indicators**: `rain_delta` (first difference), `rain_accel` (second difference), `rain_phase_rising`, `rain_phase_falling`, `rain_roc_smooth`
- **Future lookahead**: `rain_future_{1,3,5,10}` (known future rainfall at specific horizons), `rain_future_sum{5,10}`, `rain_future_max{5,10}`, `rain_remaining` (total event rainfall minus cumulative), `rain_future_trend`
- **Duration and dry spells**: `rain_duration` (consecutive wet timesteps), `dry_spell` (consecutive dry timesteps)
- **Timestep duration**: actual time intervals from `timesteps.csv` to handle variable-length steps

Two distinctive features for M2 NT1 deserve special mention:
- **`giga_fft`** — a weighted sum of the 499 extended rain lags, where weights are the Pearson correlation of each lag with the target water level, normalized to sum to 1. This massive dimensionality reduction (499 features → 1) preserves the dominant signal while avoiding the curse of dimensionality. Quantile-thresholded variants (`giga_fft_0.8`, `giga_fft_0.9`, `giga_fft_0.95`, `giga_fft_0.99`) use only lags whose correlation exceeds the corresponding quantile, further sharpening the response. Collectively, these approximate a **data-driven unit hydrograph**.
- **`ridge_p`** — prediction from a Ridge regression fitted on all 499 rain lag columns, predicting raw water level. This single meta-feature captures the best linear rainfall-to-level mapping and serves as a strong baseline signal for the boosting models.

### 2.2 Warmup-Derived Hydraulic State Features

The first 10 timesteps provide initial conditions. We extract:

- **Per-timestep water levels**: `water_level_0` through `water_level_9` (individual warmup values)
- **Summary statistics**: `warmup_mean`, `warmup_std`, `warmup_min`, `warmup_max`, `warmup_last` (= `water_level_9`), `warmup_range`
- **Trend indicators**: `warmup_last_delta` (difference between last two warmup steps)
- **Derived ratios** (1D nodes): `fill_ratio` (how full the node is relative to its depth), `head_above_invert`, `headroom_to_surface`, `head_above_invert_over_depth`, `depth_x_fill`
- **Neighbor warmup aggregation**: mean, min, max of neighbor nodes' warmup water levels using the graph adjacency structure
- **Dynamic warmup features**: edge flows and velocities during warmup (mean, std, last), inlet flow statistics

### 2.3 Graph-Structural Features

We leverage the full coupled 1D-2D graph topology:

- **Node degree and connectivity**: `n_upstream`, `n_downstream`, `upstream_total` (transitive count of all upstream nodes)
- **Centrality measures**: PageRank and degree centrality computed via NetworkX
- **Neighbor aggregations**: mean, min, max of static features (elevation, area, roughness) from immediate neighbors
- **2-hop features**: `hop2_elev_{mean,min,max,std}` — aggregated features from 2-hop neighborhood, providing broader spatial context
- **1D-2D coupling features**: elevation difference between coupled 1D and 2D nodes (`coupling_elev_diff`), surface elevation at the connected 2D node
- **Pipe capacity** (1D only): Manning's formula-based flow capacity estimates (`pipe_cap_total_in`, `pipe_cap_total_out`, `pipe_cap_max`, `pipe_cap_mean`), computed from diameter, roughness, and slope of connected edges
- **Elevation gradients**: difference between node elevation and mean neighbor elevation
- **Distance to drain** (2D nodes only): Euclidean distance to the nearest 1D drain node (`dist_nearest_drain`, `dist_2nd_drain`, `dist_3rd_drain`), computed via scipy cKDTree — captures proximity to the underground drainage system which strongly influences surface water removal

### 2.4 Cross-Feature Interactions

Multiplicative interactions between hydraulic state and rainfall:

- `upstream_x_rain_cumsum`, `inlet_x_rain_cumsum`, `fill_x_rain_cumsum`, `cap_x_rain_cumsum`
- `upstream_x_rain_ema_50`, `wl9_minus_hop2_elev_min`, `wl9_minus_hop2_elev_mean`

These capture the physical intuition that flood risk depends on both the incoming rainfall load and the current state of the drainage infrastructure.

---

## 3. Modeling Approaches

### 3.1 Gradient Boosting Foundation (M1 NT1, M1 NT2)

**Architecture**: Seed-blended CatBoost + LightGBM ensemble with fixed 50/50 weighting.

**Key design decision — batch (non-autoregressive) prediction**: Unlike traditional autoregressive approaches for time-series forecasting, we predict each timestep independently using only static features, rainfall features (past + future), and warmup-derived features. This design choice was motivated by:
1. Avoiding error accumulation from autoregressive rollout
2. Enabling seed blending (multiple models with different random seeds), which requires consistent inputs across seeds
3. Leveraging the rich future rainfall information provided in the test set

**Seed blending**: For each submodel, we train 3 seeds x 2 model types (CatBoost + LightGBM) = 6 models. Final prediction is the arithmetic mean of all 6 predictions. This significantly reduces variance without computational overhead at inference time.

**Validation**: Event-level train/validation split (53 train, 15 validation events per model), with early stopping.

**Hyperparameters**:

| Parameter | M1 NT1 (CatBoost) | M1 NT1 (LightGBM) | M1 NT2 / M2 NT2 (CatBoost) | M1 NT2 / M2 NT2 (LightGBM) |
|-----------|-------------------|-------------------|---------------------------|---------------------------|
| Iterations/Estimators | 10,000 | 10,000 | 5,000 | 5,000 |
| Max depth | 8 | 8 | 8 | 8 |
| Learning rate | 0.05 | 0.05 | 0.15 | 0.15 |
| Regularization | l2_leaf_reg=5 | reg_lambda=1 | l2_leaf_reg=5 | reg_lambda=1-3 |
| Subsampling | bagging_temp=0.8 | subsample=0.8 | bagging_temp=0.8 | subsample=0.8 |

**Validation Standardized RMSE** (50/50 ensemble):
- M1 NT1: 0.0236 (Std = 0.0014)
- M1 NT2: 0.0658 (Std = 0.0046)

### 3.2 Gradient Boosting for M2 NT2

**Architecture**: CatBoost + LightGBM + XGBoost triple ensemble with weighted averaging.

**Ensemble weights**: 0.35 * CatBoost + 0.20 * LightGBM + 0.45 * XGBoost.

**Target**: Raw water level (no delta transformation, as `USE_TARGET_DELTA = False`). M2 NT2 nodes exhibit smoother dynamics that are better captured directly.

**Seed averaging**: 4 independent seeds (56, 57, 58, 59), each producing the full triple ensemble. Final prediction averages across seeds.

**Hyperparameters** (all three models): max_depth=8, learning_rate=0.16, CatBoost iterations=12,000, LightGBM n_estimators=20,000, XGBoost n_estimators=12,000, reg_lambda=3.

**Validation Standardized RMSE**: 0.1172 (Std = 0.0430)

### 3.3 Gradient Boosting for M2 NT1

**Architecture**: CatBoost + LightGBM + XGBoost ensemble with heterogeneous target transformations.

**Key innovation — mixed target transformations**: Different models predict different transformations of the target:
- CatBoost: delta target (`water_level - water_level_9`), converted back by adding `water_level_9`
- XGBoost: delta target, converted back by adding `water_level_9`
- LightGBM: ratio target (`water_level / water_level_9`), converted back by multiplying by `water_level_9`

This diversity in target representation provides complementary views: delta captures absolute changes from the warmup state, while ratio captures proportional changes.

**Ensemble weights**: 0.4 * CatBoost + 0.4 * XGBoost + 0.2 * LightGBM.

**Extended features specific to M2 NT1**:
- 499 extended rain lags (`dodepch1k_lag_1..499`)
- Ridge regression-derived `giga_fft` feature (data-driven unit hydrograph approximation)
- Auxiliary LightGBM-predicted inlet flow (`pred_inlet_flow`) as an additional feature

**Seed averaging**: 6 seeds (56-61).

**Hyperparameters**: CatBoost iterations=50,000, LightGBM n_estimators=34,000, XGBoost n_estimators=50,000, max_depth=8-9, learning_rate=0.02, l2_leaf_reg=15.

### 3.4 Deep Learning Models for M2 NT1

M2 NT1 is the most challenging subproblem, so we invested heavily in deep learning to provide complementary predictions to boosting.

#### 3.4.1 Architecture: FloodDLModel

All DL models share a common modular architecture with configurable components:

```
Input: [Tabular features] + [Rain sequence] + [Optional: GCN embeddings]
                |                    |                     |
         TabularEncoder      SequenceEncoder          GCN Embeddings
         (2-layer MLP,       (configurable:            (2-layer GCN,
          256 hidden)         Transformer/BiGRU/        64→32 dims,
                              LSTM/CNN)                 self-supervised)
                |                    |                     |
                └────────── Fusion (concatenation) ────────┘
                                     |
                              [Optional: SCSE]
                              (channel + spatial
                               squeeze-excitation)
                                     |
                               Prediction Head
                              (2-layer MLP, 256 hidden → 1)
```

**Rain sequence encoder**: The key temporal input is a per-timestep rainfall window — a 3-channel sequence of shape `(seq_len, 3)` where:
- Channel 0: rainfall depth
- Channel 1: cumulative rainfall
- Channel 2: rainfall intensity peak

The sequence covers `10 (warmup) + past_window + future_window + 1` timesteps, providing both historical and future rainfall context. During warmup steps, channel 0 carries water level values instead of rainfall, giving the encoder direct access to the hydraulic initial condition.

#### 3.4.2 Model Variants

**Main DL Ensemble** (3 models, averaged):

| Model | Sequence Encoder | Extras | Past/Future Window | Epochs | LR | Val Std RMSE |
|-------|-----------------|--------|-------------------|--------|-----|-------------|
| N2_transformer_scse | Pre-Norm Transformer (4 heads, 2 layers, ff_mult=2) | SCSE attention block | 120/10 | 40 | 1.5e-4 | 0.3275 |
| N5_big_tf_w200 | Pre-Norm Transformer (4 heads, 2 layers, ff_mult=2) | Larger window | 200/10 | 40 | 1.5e-4 | 0.3283 |
| N3_transformer_gcn | Pre-Norm Transformer (4 heads, 2 layers, ff_mult=2) | 2-layer GCN embeddings (64→32) | 120/10 | 40 | 1.5e-4 | 0.3449 |

**Optional DL Ensemble** (3 models, averaged — used for additional blending):

| Model | Sequence Encoder | Past/Future Window | Val Std RMSE |
|-------|-----------------|-------------------|-------------|
| E6_transformer | Post-LN Transformer (d=64, 4 heads, 2 layers) | 120/10 | 0.1088 |
| E7_bigru_attention | BiGRU with multiplicative attention (96 hidden, 2 layers) | 120/10 | 0.1098 |
| E8_bigru_gcn | BiGRU (96 hidden, 2 layers) + self-supervised GCN (32d) | 120/10 | 0.1130 |

**Single DL models** (used individually for blending): N2, N3, N9 (N9_scse_gcn: BiGRU + SCSE + GCN variant).

#### 3.4.3 Training Details

- **Optimizer**: AdamW (weight_decay=1e-5)
- **Scheduler**: Cosine annealing LR
- **Batch size**: 1024-2048 (training), 16384 (inference)
- **Target**: Delta target (`water_level - water_level_9`)
- **Loss**: MSE
- **Refit strategy**: 5-fold GroupKFold cross-validation (grouped by event_id) on the full training data. Each fold trains to its own best epoch via early stopping (patience=6-8). Final prediction is the arithmetic mean across all 5 folds, reducing variance from event-specific overfitting.
- **GCN preprocessing**: Self-supervised learning — the GCN learns to reconstruct node features from graph-aggregated neighbors. The learned 32-dimensional embeddings are concatenated with tabular features.

#### 3.4.4 SCSE Attention Block

The Squeeze-and-Channel-Squeeze-and-Excitation (SCSE) block, adapted from computer vision, applies dual attention on the fused feature vector:
- **Channel squeeze-excitation**: learns channel-wise importance weights via global average → FC → ReLU → FC → Sigmoid
- **Spatial squeeze-excitation**: learns feature-wise importance via 1x1 convolution → Sigmoid
- The two attention maps are **summed** (not concatenated), preserving dimensionality while enhancing informative features

---

## 4. Final Ensemble Assembly

### 4.1 Best Submission: hz.parquet (Score: 0.0178)

| Submodel | Components | Blending |
|----------|-----------|---------|
| (M1, NT1) | Foundation boosting (Section 3.1) | Seed-blended 50/50 CB+LGB |
| (M1, NT2) | Foundation boosting (Section 3.1) | Seed-blended 50/50 CB+LGB |
| (M2, NT2) | 4 boosting seeds (Section 3.2) | Average of 4 seed runs |
| (M2, NT1) | 6 boosting seeds + DL main + DL optional | 50% boosting avg + 25% DL main ensemble + 25% avg(N2, N3, N9) |

### 4.2 Diversity Analysis

The ensemble leverages multiple axes of diversity:

1. **Model family diversity**: Gradient boosting (CatBoost, LightGBM, XGBoost) vs. deep learning (Transformer, BiGRU, GCN)
2. **Target transformation diversity**: Delta, ratio, and raw targets across different boosting models
3. **Architectural diversity**: Pre-Norm Transformers, BiGRU with attention, GCN-augmented models, SCSE attention
4. **Seed diversity**: Multiple random seeds within each model family
5. **Temporal context diversity**: Different rain window sizes (120 vs. 200 past steps)

---

## 5. Key Design Decisions and Insights

### 5.1 Batch vs. Autoregressive Prediction

We deliberately chose **non-autoregressive (batch) prediction** for all boosting models. While autoregressive approaches are natural for time-series, they suffer from error accumulation — especially critical for long forecast horizons in this competition. By treating each timestep as an independent prediction conditioned on known inputs (rainfall + warmup state + static features), we:
- Eliminated error propagation across timesteps
- Enabled seed blending (which requires consistent inputs)
- Could leverage future rainfall information directly

### 5.2 Submodel-Specific Strategies

Different subproblems warranted different approaches:
- **M1 (both node types)**: Relatively well-behaved dynamics; boosting alone was sufficient
- **M2 NT2**: Additional XGBoost improved diversity; `USE_TARGET_DELTA=False` worked better for smoother 2D surface dynamics
- **M2 NT1**: Most challenging — underground drainage dynamics are nonlinear and complex. Required the full hybrid boosting + DL approach with 6 boosting seeds and 6+ DL models for adequate coverage

### 5.3 Graph-Aware Features vs. Graph Neural Networks

We used the graph structure at two levels:
1. **Feature engineering level** (all models): Multi-hop aggregations, centrality, pipe capacity, neighbor warmup — these capture spatial context as tabular features accessible to all model types
2. **GCN embedding level** (DL models): Self-supervised 2-layer GCN produces 32-dim node embeddings that encode topological information. These embeddings are concatenated with tabular features before the prediction head.

This dual approach ensures that even non-graph-native models (boosting trees) benefit from spatial structure.

### 5.4 Rich Rainfall Representation

With rainfall being the primary dynamic driver available at test time, we invested heavily in its representation:
- Past lags (up to 499 steps) capture the unit hydrograph response
- Future lookahead features leverage the known future rainfall
- Multi-scale aggregations (rolling sums, EMAs) capture different temporal scales of the rainfall-runoff process
- The `giga_fft` feature (Ridge-weighted lag combination) approximates a learned unit hydrograph, improving predictions for M2 NT1

### 5.5 Delta Target and Warmup Utilization

Predicting `water_level - water_level_9` (the change from the last warmup step) rather than the absolute water level significantly improves performance for M2 NT1. This:
- Centers predictions around the initial condition
- Allows the model to focus on learning the *change* driven by rainfall
- Reduces the dynamic range of the target variable

For M2 NT2 and the M1 foundation (both NT1 and NT2), raw water level targets performed better. The foundation models benefit from predicting absolute values because the batch (non-AR) approach with seed blending requires stable targets that don't depend on autoregressive state.

---

## 6. Computational Environment

- **Hardware**: Kaggle GPU notebooks (NVIDIA T4 / P100)
- **Libraries**: PyTorch, CatBoost, LightGBM, XGBoost, scikit-learn, NetworkX, pandas, NumPy, SciPy
- **Training time**: ~12 hours total across all models
  - Foundation boosting (M1 + M2 NT2): ~3 hours
  - M2 NT1 boosting (6 seeds): ~4 hours
  - M2 NT2 boosting (4 seeds): ~2 hours
  - DL models: ~3 hours (across all variants)

---

## 7. What Didn't Work

- **Autoregressive prediction with water level lags**: Introduced significant error accumulation for long horizons; batch prediction without WL lags was consistently better
- **Spectral graph embeddings (Laplacian eigenvectors)**: Provided marginal improvement and were not included in the final submission
- **Numerical embeddings (rtdl_num_embeddings)**: Tested but disabled in all final models
- **Deeper transformers (4+ layers)**: Overfitting on the relatively small per-node training data
- **Shared models across urban models**: M1 and M2 have sufficiently different characteristics that separate models always outperformed shared ones
- **Pure deep learning without boosting**: DL alone was competitive for M2 NT1 but the hybrid approach always won

---

## 8. Reproducibility

All code is provided as Jupyter notebooks with fixed random seeds. The pipeline is fully reproducible:

1. Run `notebooks/boosting_m1_nt1_nt2.ipynb` (Step 1: M1 NT1 + M1 NT2)
2. Run `notebooks/boosting_m2_nt1_seed_{56..61}.ipynb` (Step 2: M2 NT1 boosting, 6 seeds)
3. Run `notebooks/boosting_m2_nt2_seed_{56..59}.ipynb` (Step 3: M2 NT2 boosting, 4 seeds)
4. Run `notebooks/dl_m2_nt1_main_ensemble.ipynb` (Step 4: M2 NT1 DL main)
5. Run `notebooks/dl_m2_nt1_{n2,n3,n9}_*.ipynb` (Step 5: M2 NT1 DL single models)
6. Run `python scripts/make_submission_hz.py` (Step 6: Assembly)

Assembly scripts support configurable paths and blend weights via command-line arguments.
