# Solution Write-up: Heterogeneous GNN with Gradient Chain Training and Gradient-Boosted Residual Correction for Urban Flood Forecasting

## 1. Abstract

We achieve 4th place (SRMSE 0.0135) in the UrbanFloodBench challenge for autoregressive (AR) water level prediction on coupled 1D–2D urban drainage networks, within 8.9% of the winning entry (0.0124). Our solution is a two-stage pipeline: a heterogeneous GNN (HeteroFloodGNNv11, 1.72M parameters) followed by XGBoost GPU residual correction.

**The core finding is that the GNN's 4-layer message passing leaves substantial spatial information on the table, and a gradient-boosted second stage with multi-hop neighbor features recovers it.** XGBoost residual correction reduces LB SRMSE from an estimated 0.0165 (GNN-only, not submitted independently) to 0.0135 — a larger improvement than any single GNN training technique.

Three techniques drive the GNN stage: (1) **per-node normalization** that recovers dynamic 2D signal otherwise suppressed 6.2x by global normalization (up to 23.3% validation improvement, architecture-agnostic), (2) **gradient chain with truncated BPTT** (separate 1D/2D chains, k_1d=32, k_2d=16) combined with trajectory loss that jointly reduce validation SRMSE by 19.7%, and (3) **SRMSE-aligned node weighting** that mitigates loss-metric misalignment (-7% validation improvement).

We also report extensive negative results from 90+ experiments. The most important lesson: training tricks that improve in-sample validation frequently degrade leaderboard performance in AR models — a finding we document with specific failure cases (noise injection: val -5.1%, LB +74%).

## 2. Problem Formulation

### 2.1 Task Description

The UrbanFloodBench competition requires predicting water levels across a coupled 1D–2D urban drainage network given rainfall forcing. The system consists of two interconnected sub-models:

- **Model 1 (M1)**: 17 nodes in the 1D drainage network and 3,716 nodes on the 2D surface mesh. M1 contributes approximately 3% of the total evaluation metric.
- **Model 2 (M2)**: 198 nodes in the 1D drainage network and 4,299 nodes on the 2D surface mesh. M2 dominates the evaluation, contributing approximately 97% of the total SRMSE.

Each simulation event spans up to 445 timesteps. The first 10 timesteps serve as spin-up with ground truth inputs; the remaining timesteps (up to 435) must be predicted autoregressively.

### 2.2 Evaluation Metric

The competition uses Standardized Root Mean Square Error (SRMSE, hereafter used throughout), a competition-specific metric defined as SRMSE = 0.5 * (RMSE_1d / std_1d + RMSE_2d / std_2d), where RMSE is computed over predicted vs. true water levels and std is the global standard deviation of water level values across all training events. Unlike standard hydrological metrics (NSE, KGE), SRMSE assigns equal weight to the 1D and 2D sub-networks regardless of node count, implicitly prioritizing balanced performance across the coupled system. Critically, the global standard deviation (1D: 3.19, 2D: 2.73) differs substantially from the per-node mean standard deviation (1D: 2.26, 2D: 0.44), creating a 6.2x discrepancy for 2D nodes that has profound implications for training strategy.

### 2.3 Training Data

The training set comprises 69 rainfall events with complete ground truth water levels. Midway through the competition, the organizers released a corrected version of the dataset that fixed a node index mapping issue in the 1D dynamic data. Our solution uses exclusively the corrected dataset.

No held-out validation set was used in the final pipeline; all 69 events were used for training (full-data training), with the public leaderboard (5 submissions/day) serving as the sole evaluation signal. This decision was motivated by two observations: (1) switching from 59 hold-out events to 69 full-data events improved LB by 5.4% (0.0312 → 0.0295) with no other changes, and (2) during development, validation SRMSE and LB SRMSE moved directionally consistently across 15+ submissions (no rank inversions). We acknowledge that this approach limits offline reproducibility and would not be acceptable in a pure research setting; leave-one-event-out CV should be conducted for the journal paper.

### 2.4 Baseline and Organizer Model

The competition is organized by the authors of DUALFloodGNN (arXiv:2512.23964). Our early experiments with a DUALFloodGNN-inspired re-implementation (v12) achieved competitive 2D performance (0.1012) but poor 1D performance (0.2603), which we attribute to the edge count imbalance between 1D pipes (197 edges) and 2D mesh (17,715 edges) in the M2 graph. This motivated our development of a heterogeneous architecture with type-specific encoders, decoders, and separate 1D/2D training strategies.

## 3. Model Architecture

### 3.1 Encoder-Processor-Decoder Framework

HeteroFloodGNNv11 follows the encoder-processor-decoder paradigm established by MeshGraphNets (Pfaff et al., 2021) and GraphCast (Lam et al., 2023), building on the broader framework of learning physical simulations with graph networks (Sanchez-Gonzalez et al., 2020). We adapt this paradigm for heterogeneous flood networks. The architecture processes 1D drainage nodes and 2D surface nodes through separate type-specific pathways while exchanging information via coupling edges.

**Encoder.** Type-specific 2-layer multi-layer perceptrons (MLPs) encode node features into a shared hidden space of dimension 128. Edge features for each edge type (1D pipe, 2D mesh, 1D–2D coupling) are similarly encoded. Input noise (std=0.02) is applied during training for regularization.

**Processor.** Four layers of NodeEdgeConv perform message passing on the heterogeneous graph. Each processor layer consists of:
1. Node-to-edge aggregation: source and target node embeddings are concatenated with the current edge embedding and passed through an edge update MLP with a residual connection.
2. Edge-to-node aggregation: updated edge messages are aggregated to target nodes via the convolutional operator, with residual connections and layer normalization (Ba et al., 2016).

The processor updates both node and edge embeddings at each layer, enabling the model to learn refined spatial relationships. With 4 layers and the graph topology of M2, the effective receptive field covers approximately 4 hops — a deliberate design choice, as we found that spatial information beyond 3 hops provides diminishing returns (Section 5.3).

**Decoder.** Type-specific 2-layer MLPs decode processed node embeddings into predictions:
- 1D nodes: 2 channels (water level delta + inlet flow absolute value)
- 2D nodes: 1 channel (water level delta)
- 1D pipe edges: 2 channels (flow + velocity, absolute values)

The distinction between delta predictions (water level) and absolute predictions (flow, velocity) reflects the flux/state paradigm from the Saint-Venant equations governing shallow water flow: water level is a conserved state variable that evolves via temporal integration of fluxes, while flow and velocity are instantaneous flux quantities determined by the current hydraulic state. The model mirrors this structure — state variables are predicted as deltas and accumulated, while flux variables are predicted as absolute values and overwritten at each step.

### 3.2 Per-Node Normalization

The single most impactful architectural decision was replacing global normalization with per-node normalization for water level deltas. Under global normalization, a single standard deviation (1D: 3.19, 2D: 2.73) is used to normalize all node deltas. However, the global std reflects both inter-node baseline differences (due to elevation and hydraulic geometry) and dynamic fluctuations. Individual 2D nodes exhibit much lower dynamic variance (mean per-node std: 0.44), meaning their normalized deltas are compressed by a factor of 6.2x (2.73 / 0.44) relative to their true dynamic range.

In hydraulic terms, per-node normalization is analogous to working with water level anomalies relative to each node's reference level — standard practice in flood modeling, where baseline water depths vary greatly across the network due to pipe inverts, surface elevation, and Manning's roughness. Global normalization inadvertently buries the dynamic signal of low-variance surface nodes beneath the inter-node static variability.

Per-node normalization stores node-specific mean and standard deviation statistics (precomputed from training data) and uses them to normalize model outputs:

```
delta_raw = model_output * per_node_std_i
```

This change improved validation SRMSE by 9.8% to 23.3% across all architecture variants tested, with the strongest effect on HeteroFloodGNNv11 (-23.3%, from 0.1371 to 0.1052). Importantly, this improvement is architecture-agnostic: it produced consistent gains on v4c (-21.3%), v10 (-9.8%), and v6 GRU (-4.9%).

An important corollary is that **delta clipping becomes destructive** under per-node normalization. Clipping bounds computed from the pooled delta distribution (p0.5–p99.5) systematically truncate predictions for high-variance nodes. For example, Node 1464 (per-node std=3.90) had 40.9% of its timesteps clipped, losing up to 80% of the predicted delta magnitude. Removing delta clipping improved validation SRMSE by 12.6% to 13.6%.

## 4. Training Strategy

### 4.1 Gradient Chain with Truncated BPTT

Standard autoregressive GNN training treats each timestep independently: the model predicts a delta, the delta is applied to update the state, and the updated state is **detached** from the computational graph before being fed back as input. This breaks the gradient flow across timesteps, preventing the model from learning to minimize cumulative drift.

Our gradient chain approach maintains differentiability across the autoregressive rollout by replacing the detached state tensor with a differentiable copy at each step. Specifically, after computing the water level delta and updating the current state, we inject the new 1D water levels back into the graph input tensor as differentiable tensors, enabling gradients to flow from future timesteps back through the chain of predictions.

For 1D nodes (198 nodes), the full gradient chain spans k_1d=32 steps before truncation. For 2D nodes (4,299 nodes), memory constraints limit the chain to k_2d=16 steps. These truncation lengths were primarily determined by GPU memory (32 GB), not by physical timescale analysis, and introduce gradient bias (Tallec & Ollivier, 2017); whether k_2d=16 is sufficient to capture the relevant CFL-governed information propagation speed on the 2D mesh remains an open question. Empirically, extending k_1d beyond 32 destabilized training (Section 8.2), while k_2d=16 with high learning rate yielded a 2.8% improvement over k_2d=8. A critical implementation detail is that the 1D and 2D chains must be truncated **separately** (v30): an earlier implementation (v26–v28) inadvertently truncated the 1D chain at the same interval as 2D (every 8 steps), limiting 1D gradient flow to 8 steps. Correcting this — separate TBPTT with k_1d=32 and k_2d=16 — yielded a 4.2% validation improvement.

The training loss combines two terms:
- **Delta loss**: Per-timestep MSE between predicted and true normalized deltas
- **Trajectory loss** (weight=0.3): MSE of cumulative water level error over the rollout window, providing a direct gradient signal for drift correction

In a controlled experiment starting from the v11c baseline with per-node normalization (val 0.1186), adding the 1D gradient chain reduced validation SRMSE by 19.7% to 0.0952, with particularly strong improvement on 1D nodes (-24.8%). Note that the 0.1186 baseline (v11c) differs from the 0.1052 in Table 1 (v11b) because v11c incorporates a validation bug fix (water_volume zeroing in the validation loop) that was absent in v11b, resulting in a higher corrected baseline. Ablation studies (5 epochs, r=32) confirmed that the combination of gradient chain and trajectory loss is essential: gradient chain alone (w_traj=0) achieves 3.7% 1D improvement; trajectory loss alone (detached, ba0) achieves 4.1%; but the combination (w_traj=0.3 with chain) achieves 12.5% — a superlinear interaction where the chain enables the trajectory loss to propagate meaningful multi-step gradients.

### 4.2 SRMSE-Aligned Loss

The SRMSE metric weights each node inversely by the global standard deviation, meaning high-variance nodes (those with large absolute water level changes) contribute disproportionately. Standard MSE loss, applied after per-node normalization, treats all nodes equally — creating a fundamental loss-metric misalignment.

We mitigate this by introducing per-node loss weights proportional to the square of the ratio of per-node std to global std:

```
w_i = (per_node_std_i / global_std)^2
```

This weighting is applied to both delta loss and trajectory loss. The SRMSE-aligned loss reduced validation SRMSE by 7% (from 0.0526 to 0.0489) in the final full-data training pipeline (v76). An earlier, simpler formulation (v28) using w_i = sigma_i / mean(sigma) achieved 5.7% improvement during the hold-out training phase; the squared formulation in v76 better approximates the SRMSE metric's quadratic dependence on per-node standard deviation.

### 4.3 Iterative Refinement

Rather than training a single model end-to-end, we employ iterative refinement: each training iteration starts from the previous best checkpoint and re-trains with the same hyperparameters. Each iteration follows a two-phase schedule:

- **Phase 1**: rollout=32, lr=2e-5, 15 epochs (fast convergence on short sequences)
- **Phase 2**: rollout=64, lr=4e-6, 15 epochs (fine-tuning on longer sequences, best model selected here)

After Phase 2, a polish stage (lr=1.2e-6 → 2.4e-7) extracts remaining gains. This iterative approach produced steady 1–3% improvements per iteration over 12 cycles (v56 → v68), reaching diminishing returns at v76.

### 4.4 Additional Training Techniques

- **Step-zone weighting** (v64/v76): Loss weights of 1.0x for t<100, 1.5x for t=100–199, 2.5x for t=200–299, and 5.0x for t≥300, focusing optimization on the challenging late-phase predictions where error accumulation is most severe.
- **Degree-1 node boost** (v66): Degree-1 nodes (48.5% of all nodes) contribute 58% of MSE; a 1.5x loss multiplier on these nodes improved convergence.
- **Full-data training** (v56+): Using all 69 training events without hold-out, relying on LB for evaluation. This provided a 17% increase in training data at no cost, given the availability of the leaderboard.
- **Mixed rollout training** (v75): Phase 1 with rollout=32 for fast convergence, Phase 2 with rollout=400 to expose the model to near-full-length autoregressive dynamics during training.

## 5. Inference Pipeline

### 5.1 Autoregressive Rollout and Seed Ensemble

At inference time, the model performs a fully autoregressive rollout: starting from the spin-up state at t=10, each timestep prediction feeds back as input to the next. We average predictions from 3 independently trained models (seeds 42, 123, 777) with identical architecture and hyperparameters. Seed ensemble reduces LB SRMSE by 0.0002 (0.0267 → 0.0265) — a modest improvement that is nonetheless consistent.

No delta clipping is applied during inference, as per-node normalization renders pooled clip bounds destructive (Section 3.2). No test-time adaptation or online correction is used; all corrections are pre-computed from training data.

**Model 1 (M1).** The same HeteroFloodGNNv11 architecture was adapted for M1 (17 1D + 3,716 2D nodes). All M2 training techniques — gradient chain, node-weighted loss, separate TBPTT, trajectory loss — were transferred to M1 (v35), achieving validation SRMSE of 0.0040. M1 was subsequently trained on all 69 events (v36, full-data, val=0.0038). M1 contributes approximately 3% of total SRMSE (computed as M1's SRMSE contribution relative to total across M1+M2), so no XGBoost residual correction was applied to M1; its GNN predictions with zone bias are used directly in the final submission.

### 5.2 Zone-Aware Bias Correction

The autoregressive rollout introduces time-dependent systematic bias: slight overprediction in early steps (0–100) and significant underprediction in late steps (200+). This bias is structural — the same pattern appears consistently across all 69 training events.

Zone bias correction operates as follows:
1. Run full AR rollout on all 69 training events using the final checkpoint
2. Compute per-node residual (predicted − true) at each timestep
3. Partition timesteps into 30 temporal zones (~14 steps each)
4. Average residual per node per zone = zone bias
5. At inference, subtract zone bias from predicted **water levels** (not deltas)

This distinction is critical: per-step **delta** correction (subtracting bias from each step's predicted delta) caused catastrophic failure (+129% LB degradation) because small per-step corrections accumulate through hundreds of autoregressive steps, generating massive drift. Zone bias correction operates on the final water levels with coarse temporal grouping, avoiding AR feedback amplification.

Zone bias improved LB by 0.0006 (0.0265 → 0.0259), making it 3x more effective than seed ensemble.

### 5.3 XGBoost Residual Correction

The most impactful component of our inference pipeline is a second-stage XGBoost model that learns to correct residuals from the zone-bias-corrected GNN predictions (i.e., XGBoost operates on the output of Stage 1, not raw GNN predictions). Conceptually, the XGBoost stage is analogous to additional spatial aggregation beyond the GNN's 4-hop receptive field: while not performing message passing in the formal sense, it captures multi-hop spatial context through explicit neighbor features.

**Feature engineering (37 features per node type for v9b).** Features fall into five categories:

| Category | Features | Description |
|----------|----------|-------------|
| Node static | degree, is_coupled, static features | Graph topology |
| Temporal | step_ratio, zone_id, raw_step, cum_rain, rain_now, rain_lag10/30, steps_since_peak | Temporal position and rainfall state |
| Prediction | pred_val, pred_delta (vs initial), pred_over_nodestd | GNN output features |
| Event-level | total_rain, peak_rain, peak_position, rain_duration, late_rain, event_T | Event-level rainfall statistics |
| Spatial (most important) | 1-hop/2-hop/3-hop neighbor mean_pred, std_pred + coupled pred, laplacian, rain_x_pred_delta, pred_ratio_nb1, pred_minus_nb2 | Multi-hop spatial context |

The spatial features — particularly the 3-hop neighbor aggregations — are the most important feature group, as they extend the effective spatial receptive field well beyond the GNN's 4 layers.

**Model configuration:**
- XGBoost GPU (Chen & Guestrin, 2016; `tree_method: gpu_hist, device: cuda`) on RTX 5090
- Separate 1D and 2D models; 2D further split into 3 std-based buckets (low/mid/high) in v9b
- Learning rate 0.01, max_leaves 127 (1D) / 63 (2D), max_depth 10
- 180,000 boosting rounds with early stopping at 1,000
- 5-fold cross-validation with event-level splits

**Progression:** Early versions used LightGBM (Ke et al., 2017); we switched to XGBoost GPU at v6 for CUDA acceleration.

| Version | OOF SRMSE (out-of-fold) | LB | Key Change |
|---------|-----------|-----|-----------|
| v2 (LightGBM) | 0.0398 | 0.0198 | Baseline: 23 features |
| v4 (LightGBM) | 0.0361 | 0.0186 | +1-hop neighbor features |
| v5-nw (LightGBM) | 0.0322 | 0.0165 | +2-hop, 5000 rounds |
| v6 (XGBoost GPU) | 0.0306 | 0.0157 | +3-hop, lr=0.01, 30000 rounds |
| **v9b (XGBoost GPU)** | **0.0071 (in-sample)** | **0.0135** | +std-bucket 2D, 180K rounds, +4 features |

The v9b cross-validation was interrupted after 2 of 5 folds due to time constraints. In-sample evaluation on all 69 training events yields a mean SRMSE of 0.0071 (88.8% reduction from the GNN-only baseline of 0.0631), though this figure includes training data and thus overstates generalization performance. On the leaderboard, the XGBoost stage reduces SRMSE from 0.0165 (v76 GNN + zone bias) to 0.0135, an 18.2% improvement.

## 6. Experimental Analysis

### Table 1: Cumulative Ablation (GNN Stage)

Note: This table reflects the forward development order (each row adds one technique to the previous best). It is not a backward-elimination ablation; interaction effects between components are not isolated. The Delta column shows the relative improvement from the immediately preceding row.

| Component | Val SRMSE | Delta | Cumulative |
|-----------|-----------|-------|------------|
| Baseline (v11, global norm) | 0.1371 | — | — |
| + Per-node normalization (v11b) | 0.1052 | -23.3% | -23.3% |
| + Gradient chain, 1D only (v23) | 0.0952 | -9.5% | -30.6% |
| + 2D gradient chain (v26) | 0.0824 | -13.4% | -39.9% |
| + Node-weighted loss (v28) | 0.0777 | -5.7% | -43.3% |
| + Separate TBPTT (v30) | 0.0726 | -6.6% | -47.0% |
| + Extended k_2d=16 (v32c) | 0.0699 | -3.7% | -49.0% |
| + Iterative refinement × 12 (v68) | 0.0526 | -24.7% | -61.6% |
| + SRMSE-aligned loss (v76) | 0.0489 | -7.0% | -64.3% |

### Table 2: XGBoost Residual Correction Progression

| Version | CV SRMSE | LB | Features | Learner |
|---------|----------|-----|----------|---------|
| GNN only (v68 + 3-zone bias) | — | 0.0259 | — | — |
| GNN only (v76 + 30-zone bias) | — | 0.0165* (est.) | — | — |
| v2 | 0.0398 (OOF) | 0.0198 | 23 | LightGBM CPU |
| v4 | 0.0361 (OOF) | 0.0186 | +1-hop spatial | LightGBM CPU |
| v5-nw | 0.0322 (OOF) | 0.0165 | +2-hop, 5K rounds | LightGBM CPU |
| v6 | 0.0306 (OOF) | 0.0157 | +3-hop, 30K rounds | XGBoost GPU |
| **v9b** | **0.0071 (in-sample)** | **0.0135** | **+std-bucket, 180K rounds, +4 feat** | **XGBoost GPU** |

\* v76 GNN-only LB is estimated from the v68→v76 validation improvement ratio; this configuration was not submitted independently to the leaderboard. v9b CV was incomplete (2/5 folds); the in-sample figure overstates generalization. Based on the OOF/LB ratio observed for v6 (0.0306 / 0.0157 = 1.95), the v9b OOF SRMSE can be estimated at approximately 0.026.

### Table 3: XGBoost v9b Top-15 Feature Importance

| Rank | 1D Feature | Gain | 2D Feature | Gain |
|------|-----------|------|-----------|------|
| 1 | is_coupled | 1.5 | laplacian | 3.1 |
| 2 | static_5 | 0.8 | event_T | 2.9 |
| 3 | static_0 | 0.7 | pred_ratio_nb1 | 2.7 |
| 4 | late_rain | 0.7 | pred_minus_nb2 | 2.5 |
| 5 | static_1 | 0.6 | pred_delta | 2.3 |
| 6 | node_std | 0.5 | is_coupled | 2.0 |
| 7 | static_4 | 0.5 | node_std | 1.9 |
| 8 | static_3 | 0.5 | static_4 | 1.7 |
| 9 | raw_step | 0.4 | cum_rain | 1.6 |
| 10 | coupled_2d | 0.4 | pred_val | 1.6 |
| 11 | pred_ratio_nb1 | 0.4 | nb1_std | 1.5 |
| 12 | steps_since_peak | 0.4 | pred_over_nodestd | 1.5 |
| 13 | degree | 0.4 | static_0 | 1.5 |
| 14 | cum_rain | 0.4 | late_rain | 1.4 |
| 15 | rain_duration | 0.4 | nb2_std | 1.3 |

Note: Static features represent physical node attributes provided by the organizers (1D: pipe length, diameter, invert elevation, Manning's coefficient, etc.; 2D: ground elevation, area, imperviousness, etc.). The gain values reported are from the final v9b model trained on all data (not fold-specific). The 1D model exhibits relatively flat importance across features (max gain 1.5), while the 2D model shows stronger differentiation. For 2D, spatial-derived features dominate: laplacian (pred − nb1_mean), pred_ratio_nb1, pred_minus_nb2, and nb1_std/nb2_std collectively account for the top importance positions. The is_coupled feature ranks highly for both 1D and 2D, reflecting the importance of the 1D–2D interface for residual correction.

## 7. Failure Mode Analysis

### 7.1 Low-Variance 2D Nodes

The primary remaining bottleneck is low-variance 2D nodes (per-node std < 0.156). These nodes contribute 28.3% of total error but achieve only 27.4% improvement from XGBoost correction — the lowest improvement rate across all std buckets. By contrast, high-variance nodes (std > 0.688) contribute 10.7% of error and achieve 52.0% improvement. The fundamental issue is that SRMSE divides by per-node std: even small absolute errors on low-variance nodes produce large standardized errors.

### 7.2 Late-Phase Error Accumulation

Steps 200–299 exhibit the highest SRMSE (0.162–0.174), driven by error accumulation through 200+ autoregressive steps. Twenty 2D nodes show late/early error ratios exceeding 2.5x. The gradient chain (k_1d=32, k_2d=16) mitigates this for the first ~32/16 steps of each TBPTT window but cannot fully prevent drift over 435 total steps.

### 7.3 1D Single-Step Error

Root-cause analysis (EDA v54) revealed that 1D teacher-forcing single-step error is 2–5x that of 2D, indicating an intrinsic architectural limitation. Delta error autocorrelation of 0.79 for 1D nodes means errors are systematic — each step's error correlates strongly with the previous step's — leading to persistent drift rather than random walk behavior.

### 7.4 Recession-Phase 1D

During the recession phase (step 200+), 1D correction improvement drops to 10.2% compared to 20.3% at peak. The worst 3 events (the longest event class, T=445) account for 8% of total SRMSE, suggesting that the longest events are disproportionately difficult.

## 8. What Didn't Work

We conducted over 90 experiments. The following is a representative selection of approaches that failed to improve leaderboard performance, organized by category. We report these with the same rigor as our positive results, as understanding failure modes is essential for future work on GNN-based flood simulation.

### 8.1 Architecture Variants

| Approach | Result | Hypothesis and Explanation |
|----------|--------|---------------------------|
| DUALFloodGNN-inspired (v12) | 2D=0.1012 (competitive), 1D=0.2603 | Originally designed for 2D mesh; the heterogeneous 1D–2D setting with extreme edge count imbalance (197 vs 17,715) requires type-specific treatment |
| GRU temporal memory (v6/v6b) | val=0.1450 (from scratch), 0.0811 (TL) | Quality-diversity tradeoff: weak models are diverse but unusable; strong fine-tuned models converge to same predictions (corr=0.89) |
| Hidden dim 256 (v49) | Training crashed (CUDA error) | Unstable at higher capacity; only completed 1 iteration (val=0.1553 vs v76 baseline 0.0489) |
| Temporal Bundling K=4 (v16/v29) | 2D=0.0460 (good), 1D=0.1166 (bad) | Reduces AR steps 4x but 1D predictions degrade; useful only as ensemble diversity source |
| Global pooling (v52b/v92) | PLATEAU+FROZEN at baseline 0.0489 | 1D has only 198 nodes; direct message passing already captures global context |
| Fourier Neural Operator (v7/v7b) | Failed to converge | FNO assumes regular grids; irregular unstructured mesh destroys spectral representation |
| Dual decoder (v93) | val=0.0501 (worse than 0.0489 baseline) | Separate decoders for early/late phases add parameters without useful inductive bias |
| Graph-GPS Transformer (v94) | val=0.0889 (1.8x worse) | Graph transformer (Rampasek et al., 2022) with positional encodings; massive overhead without benefit on this graph topology |

### 8.2 Training Strategies

| Approach | Result | Hypothesis and Explanation |
|----------|--------|---------------------------|
| Noise injection (v27, std=0.1) | LB +74% (catastrophic) | Creates train/inference distribution shift; model learns to denoise rather than predict |
| Noise injection (v55, std=0.1, numpy-side) | Diverged | Same distribution shift, more severe |
| Noise injection (v55d, std=0.02) | val -0.3% | Marginal; noise too small to help, any larger is harmful |
| Extended rollout r=128 (v55c, v23c) | Destabilized | 12 events dropped; gradient instability at long horizons |
| Event-length weighting (v33/v34) | No improvement | Event weight sqrt(T/median_T) does not help beyond base training |
| Physics-informed loss (v5) | Validation degraded | Adding mass conservation constraints as auxiliary loss terms destabilized the learned representations; unlike in DUALFloodGNN where physics losses are integrated from the start, retrofitting them onto a pre-trained data-driven model proved counterproductive |
| Physical features (v47/v47b) | Zero effect | GNN already captures implicit physical information; explicit features are redundant |
| Feature exclusion for zero-valued test vars (v48) | 1D 2.3x worse | Inlet flow information is critical even though it is model-predicted, not ground truth |
| k_1d=64 gradient chain (v32, v36) | 1D destabilized | Long 1D chains amplify gradient noise; k_1d=32 is the stability limit |
| Anti-drift penalty (v77) | val=0.0491 (marginal) | Mean-residual penalty + EMA hurts per-node accuracy more than it helps bias |
| High learning rate (v79, lr=1e-5, 5–10x polish lr) | val +6.5% | Destabilizes learned representations |
| Push-forward training (v89) | val=0.0482 (Δ=-0.0004, AUTO-STOP) | Full-rollout training with extended gradient chain; negligible gain, PLATEAU detected |

### 8.3 Post-Processing and Ensemble

| Approach | Result | Hypothesis and Explanation |
|----------|--------|---------------------------|
| Per-step delta bias correction | LB +129% | Small per-step corrections accumulate through hundreds of autoregressive steps, generating massive drift (max ±3.99 raw WL) |
| SRMSE-weighted XGBoost (1/std²) | OOF +7.4% | Over-emphasizes low-variance nodes; corrections are less reliable on these nodes |
| Correction clipping (p95–p99.9) | All worse | Large corrections are valid; node 434 improves from 0.73 to 0.12 with full correction |
| Global shrinkage (alpha < 1) | All worse | Uniformly reducing correction magnitude is never helpful |
| Temporal smoothing (MA/EMA) | Max -0.64% | Lag-1 autocorrelation=0.97 suggests potential, but effect is negligible in practice |
| Per-node shrinkage | OOF -7.0% but overfit risk | 69 events × 4,497 parameters; per-node shrinkage factors cannot generalize |
| AR features in XGBoost (prev_residual) | OOF +256% | Train/test leakage: training uses GT residuals, inference uses XGBoost-predicted residuals |
| LightGBM GPU (OpenCL) | 3x slower than CPU | OpenCL implementation inferior to CPU on RTX 5090; XGBoost CUDA is the only viable GPU option |
| Heun inference (2nd-order Runge-Kutta) | +10.5% worse | Model's learned deltas are implicitly calibrated for Euler-step integration; Heun's half-step evaluations query the learned force field outside its training distribution |
| SWA (stochastic weight averaging) | Worse than best single | Weight-space averaging is inferior to prediction-space ensemble for this task |
| Weak model ensemble (v25+v23b) | LB worsened | Models weaker than best single destroy ensemble quality regardless of diversity |
| Same-architecture seed ensemble >3 | LB worsened | High inter-model correlation (>0.99) means additional seeds add noise, not signal |

### 8.4 Key Lessons from Negative Results

1. **Val improvement ≠ LB improvement.** Training tricks (noise, node weighting) often improve in-sample validation while degrading generalization (v27: val -5.1%, LB +74%).
2. **Quality-diversity tradeoff in ensembles.** Weak models provide diversity but their inclusion always degrades the ensemble. Strong models converge to similar predictions, limiting ensemble benefit. The sweet spot (seed ensemble of 3 identical models) provides modest but reliable improvement.
3. **AR feedback amplifies corrections.** Any correction applied at the delta level (per-step) accumulates through hundreds of AR steps. Only corrections applied to absolute water levels with coarse temporal grouping are safe.
4. **Explicit physics features are redundant.** Adding hydraulic gradient and Manning's equation terms provides zero improvement, suggesting these relationships are already captured by the combination of static node features (which include pipe properties) and the GNN's learned representations.

## 9. Discussion

### 9.1 Two-Stage Architecture

The most surprising finding is the magnitude of improvement from XGBoost residual correction (18.2% LB improvement over the v76 GNN baseline). The GNN, despite its sophisticated gradient chain training, has a fundamental receptive field limitation: 4 message passing layers cover approximately 4 hops. The XGBoost stage extends effective spatial context to 3 hops beyond the GNN's frontier through precomputed neighbor statistics, complementing the GNN's learned representations with explicit spatial aggregation via gradient boosting.

This two-stage NN + GBDT residual correction pattern has precedent in other domains: LSTM-XGBoost hybrids for air quality forecasting (Li et al., 2025), CNN-BiGRU-XGBoost for wind power prediction, and physics-ML hybrids in hydrology where machine learning corrects residuals from process-based models. However, GNN + GBDT specifically — where the GBDT stage explicitly extends the GNN's spatial receptive field via multi-hop neighbor features — appears to be novel in the flood modeling literature. The spatial features consistently dominated XGBoost feature importance rankings, confirming that the GNN's spatial representation is the primary bottleneck.

### 9.2 Per-Node Normalization as a General Principle

Per-node normalization produced architecture-agnostic improvements of 9.8%–23.3% across four different model variants. This technique should be applicable to any graph-based physical simulation where node-level variance differs substantially from global variance — a common scenario in heterogeneous networks where different physical domains (pipes vs. surface) operate at different scales.

### 9.3 Gradient Chain Training

The gradient chain with TBPTT achieved a 19.7% improvement by enabling the model to optimize directly for multi-step prediction accuracy rather than single-step delta accuracy. The separate treatment of 1D and 2D gradient chains (different TBPTT windows) was critical, as the two node types have different memory requirements and stability characteristics.

The iterative refinement strategy (12 cycles, each starting from the previous best checkpoint with a fresh optimizer) produced steady 1–3% improvements per cycle, with diminishing returns after cycle 8. We hypothesize that re-initializing the optimizer state (resetting momentum and adaptive learning rate statistics) helps escape saddle points in the highly non-convex multi-step loss landscape, though we lack direct evidence to distinguish this from simple learning rate schedule effects.

### 9.4 Limitations

Our approach has several limitations:
1. **No held-out validation.** Full-data training prevents offline evaluation; all tuning relied on the 5-submission/day leaderboard.
2. **XGBoost is non-differentiable.** The two stages cannot be jointly optimized.
3. **Low-variance 2D nodes remain challenging.** The SRMSE metric's division by per-node std amplifies errors on these nodes, and neither the GNN nor XGBoost effectively addresses this.
4. **Computational cost.** The full training pipeline (GNN: ~150 GPU-hours for 3 seeds × 12 iterations; XGBoost: ~20 GPU-hours for 180K rounds with 5-fold CV; bias/feature computation: ~10 hours) totals approximately 180 GPU-hours on an RTX 5090. Inference for a single event takes approximately 30 seconds (GNN rollout) plus 5 seconds (XGBoost correction).
5. **Generalization to other networks.** Our approach was developed and evaluated on a single urban drainage network. Whether the techniques — particularly per-node normalization and the XGBoost spatial features — transfer to networks with different topologies, scales, or hydraulic regimes remains untested. Broader evaluation across diverse urban catchments, as advocated by recent reviews of deep learning for flood mapping (Bentivoglio et al., 2022), would strengthen the claims.

## 10. Conclusion

The single most important takeaway from this work: **a GNN's learned spatial representation is incomplete, and a simple gradient-boosted second stage with explicit multi-hop features can recover substantial residual structure.** This hybrid GNN + GBDT paradigm achieved LB 0.0135, reducing the gap to the winning solution to 8.9%.

Three techniques were essential: per-node normalization (up to 23.3% improvement, architecture-agnostic), gradient chain TBPTT with trajectory loss (19.7% joint improvement), and SRMSE-aligned loss weighting (-7%). Each addresses a distinct failure mode — signal compression, gradient truncation, and loss-metric misalignment — and their effects are largely additive.

Equally important is what did not work. Over 90 experiments revealed that training tricks improving in-sample validation frequently harm LB in AR models; per-step delta corrections are catastrophic in long-horizon AR rollout; and explicit physical features add nothing when static node attributes already encode the relevant physics. These negative results should inform future GNN-based flood simulation efforts.

The gap between our solution (LB 0.0135) and the winning entry (0.0124) — 8.9% — suggests room for improvement, potentially through larger GNN receptive fields, learnable residual correction, or end-to-end differentiable two-stage training. We hope this detailed account of both successes and failures contributes to the broader understanding of GNN-based flood simulation.

## References

- Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421.
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. (2021). Learning mesh-based simulation with graph networks. *ICLR 2021*.
- Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., & Battaglia, P. (2020). Learning to simulate complex physics with graph networks. *ICML 2020*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. KDD*, 785–794.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- Williams, R. J., & Peng, J. (1990). An efficient gradient-based algorithm for on-line training of recurrent network trajectories. *Neural Computation*, 2(4), 490–501.
- Tallec, C., & Ollivier, Y. (2017). Unbiasing truncated backpropagation through time. *arXiv:1705.08209*.
- Rampasek, L., Galkin, M., Dwivedi, V. P., et al. (2022). Recipe for a general, powerful, scalable graph transformer. *NeurIPS 2022*.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
- Bentivoglio, R., Isufi, E., Jonkman, S. N., & Taormina, R. (2022). Deep learning methods for flood mapping: a review of existing applications and future research directions. *Hydrology and Earth System Sciences*, 26(16), 4345–4378.
- [Competition organizers]. DUALFloodGNN. arXiv:2512.23964. [Full citation to be completed with organizer input.]

---

## Figures

The following figures should accompany this write-up:

1. **`figures/architecture.png`** — HeteroFloodGNNv11 encoder-processor-decoder architecture diagram. Shows separate 1D/2D encoder pathways, 4-layer NodeEdgeConv processor with edge updates and residual connections, type-specific decoders (1D: 2ch node + 2ch edge, 2D: 1ch node), and 1D–2D coupling edges. Annotate hidden_dim=128, 1.72M parameters.

2. **`figures/pipeline.png`** — End-to-end inference pipeline: GNN 3-seed ensemble → zone-aware bias correction (30 zones) → XGBoost residual correction (v9b, 37 features). Show data flow from rainfall input through spin-up, AR rollout, ensemble averaging, bias subtraction, and XGBoost correction to final submission. Annotate LB improvement at each stage: 0.0259 → 0.0135.

3. **`figures/ablation_waterfall.png`** — Waterfall chart showing cumulative validation SRMSE improvement from baseline (0.1371) through each major component: per-node norm (-23.3%), gradient chain 1D (-9.5%), gradient chain 2D (-13.4%), node-weighted loss (-5.7%), separate TBPTT (-6.6%), extended k_2d (-3.7%), iterative refinement (-24.7%), SRMSE-aligned loss (-7.0%), ending at 0.0489.

4. **`figures/training_progression.png`** — Validation SRMSE over training versions v11 through v76. X-axis: version identifier, Y-axis: validation SRMSE. Annotate major breakthroughs (per-node norm, gradient chain, SRMSE-aligned loss). Include inset showing diminishing returns in iterative refinement (v56–v68).

5. **`figures/xgb_feature_importance.png`** — Horizontal bar chart of top-15 XGBoost features by gain, split into 1D (left panel) and 2D (right panel). Color-code by feature category (static, temporal, prediction, event-level, spatial). Spatial features should visually dominate.

6. **`figures/zone_bias_pattern.png`** — Heatmap or line plot showing temporal bias structure across 30 zones for representative nodes. X-axis: zone (1–30), Y-axis: mean bias (predicted − true). Show overpredict in early zones, underpredict in late zones. Include separate traces for 1D and 2D nodes, annotated with mean absolute bias values (1D: 0.099, 2D: 0.073 for late zone).
