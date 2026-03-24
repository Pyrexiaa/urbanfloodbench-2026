# FloodGraphFlow-XGB

Solution for the Kaggle competition 

[**UrbanFloodBench: Flood Modelling**](https://www.kaggle.com/competitions/urban-flood-modelling/overview)

<img width="400" height="233" alt="Gemini_Generated_Image_vr2b04vr2b04vr2b" src="https://github.com/user-attachments/assets/c47540cb-5d99-4e67-86e4-10e874e2c8fc" />

---
A graph-aware XGBoost pipeline for urban flood prediction using hydraulic context features, flow proxies, and mass-balance approximations.

Developed by team **NSEOverflow**
- Brandon Low — [@zxKyouma](https://github.com/zxKyouma)
- Daphne Chu — [@D-Keii](https://github.com/D-Keii)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate flood
```

## Dataset
Download the [UrbanFloodBench dataset](https://drive.google.com/file/d/18XT8NlOfWOnR64mi6faKOR2mVzqpaLSA/view) and the [model checkpoints](https://drive.google.com/drive/u/1/folders/1BIUuIi7HLhT46s47kIhYu0hA4vBXFsmG)

After downloading:
1. Extract the archive
2. Place the extracted folders in the project root (`./`) 
3. Run the metadata setup script:
   ```bash
   bash scripts/populate_model_metadata.sh
   ```
4. Place the model checkpoints in `/saved_models`

## Repository Structure
Your directory should look like this:
```
FloodGraphFlow-XGB
...
├── saved_models
│   ├── model1_best.pkl
│   └── model2_best.pkl
├── Models
│   ├── Model_1
│   │   ├── train
│   │   │   ├── events.csv
│   │   │   ├── events_split_seed42
│   │   │   │   ├── train_split.csv
│   │   │   │   └── val_split.csv
│   │   │   └── events_hardholdout_seed42
│   │   ├── test
│   │   │   └── events.csv
│   │   ├── processed
│   │   │   └── csv_features_stats.yaml
│   │   └── model1_node_pca.joblib
│   └── Model_2
...
└── README.md
```

## Model Training

The models for Cities 1 and 2 are trained with the following commands:
```
# Train Model_1
bash scripts/train_model1_best.sh

# Train Model_2
bash scripts/train_model2_best.sh
```

## Inference
Model inference can be done with the following commands:
```
# Model_1 test predictions
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model1_best.yaml \
    --backend xgboost_cpu \
    --load_model_path saved_models/model1_best.pkl \
    --dump_test_predictions predictions/model1_test_predictions.parquet

# Model_2 test predictions
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model2_best.yaml \
    --backend xgboost_cpu \
    --load_model_path saved_models/model2_best.pkl \
    --dump_test_predictions predictions/model2_test_predictions.parquet

# Merge for submission
python scripts/merge_xgb_submission.py \
    --sample sample_submission.parquet \
    --model1 predictions/model1_test_predictions.parquet \
    --model2 predictions/model2_test_predictions.parquet \
    --output submissions/floodgraphflow_xgb_submission.parquet
```
*Note: `sample_submission.parquet` should be downloaded from the [competition website](https://www.kaggle.com/competitions/urban-flood-modelling/data?select=sample_submission.parquet).*

## Approach

We use a **graph-aware stacked XGBoost pipeline** rather than a single end-to-end sequence model.

Flood behavior in this task is **not purely local**: each node depends on *upstream inflow*, *downstream blockage*, *storage*, and *boundary conditions*. A single flat regressor struggled to represent these different hydraulic regimes.

By combining **graph-derived features**, **physics-inspired hydraulic proxy features**, **auxiliary flow surrogates (`qnet`, `qin`, `qout`)**, and a **two-stage regime-aware predictor**, the model captures propagation, retention, and delayed drainage more reliably than local rainfall and water-level features alone.

---

### Model Overview

At a high level, the model works in **four stages**.

**Stage 1 — Graph-aware feature construction**

> We build node-level temporal features from rainfall, water level, and static network attributes, then augment them with **graph features**, **boundary indicators**, and **mass-deficit storage proxies** that encode coarse hydraulic structure.

**Stage 2 — Auxiliary flow prediction**

> We train **Stage-A models** to predict latent hydraulic quantities such as **net flow**, **inflow**, and **outflow**. Their **out-of-fold predictions** are fed back into the feature set.

**Stage 3 — Regime-aware prediction**

> A **two-stage XGBoost predictor** combines **regime classification** with **conditional regression**, allowing the model to treat calm, rising, and storage-dominated states differently.

**Stage 4 — Final submission assembly**

> The full pipeline is trained separately for **`Model_1`** and **`Model_2`**, and their predictions are merged into the final submission.

---
### Preprocessing
- **Normalization:** z-score normalization applied to all static and dynamic node/edge features.

- **Stabilization of heavy-tailed features:**
  - Clipping of extreme values
  - `log1p` / `asinh` transforms for hydraulic ratios and interaction terms

- **Edge-case handling:**
  - Dedicated treatment of zero-area endpoint nodes to maintain stable feature distributions

---

### Hyperparameters

All hyperparameters were tuned using **Optuna**.

- **Main regressor:** 800 trees, learning rate 0.03, max depth 8  
- **Regime classifier:** 600 trees, learning rate 0.03, max depth 6  
- **Event settings:** quantile = 0.88, horizon = 24 

---

### Feature Engineering

The final model used a total of 262 features.

The strongest feature families in the final submission were:

#### 1. Graph-propagated water-level context

> These features summarize nearby hydraulic state over the drainage graph rather than using only the local node.

- **`fe_graph_pulse`**
- **`fe_graph_hop2_features`**
- **`fe_level_imbalance_features`**

#### 2. Stage-A net-flow stack (`qnet`)

> We train an auxiliary model to predict net flow, then feed those predictions back into the main model.

- **`qnet_stack`**
- **`qnet_phys_baseline_feature`**
- **`fe_qhat_graph2`**
- **`fe_qhat_graph2_hop2`**

#### 3. Stage-A inflow / outflow stack (`qin`, `qout`)

> These features expose directional transport structure that is hard to recover from local predictors alone.

- **`qinout_stack`**

#### 4. Basin mass-deficit and storage proxy features

> These are the main mass-balance-like engineered features.

- **`fe_basin_mass_deficit_features`**

A detailed list of all included features is included in `FEATURES.md`.

---

### Model Ablations
| Model | Addition | City 2 Score |
|---|---|---:|
| A | Baseline XGBoost | 0.203494 |
| B | A + pruned feature set + graph / qhat graph-neighbor features | 0.141594 |
| C | B + auxiliary `peak_within_24` target | 0.138205 |
| D | C + basin / storage mass-deficit framing | 0.084896 |
| E | D + node priors + downstream lockup + subcatchment mass-deficit | 0.079190 |
| F | E + `twi_spi` + multiscale mass mismatch + `HAND` proxy features | 0.077998 |
| G | F + phase-MoE pilot | 0.077271 |
| H | G + pruneA regime cleanup | 0.076822 |
| I | H + edge-aware downstream features | 0.076526 |
| J | I + node drop priors | 0.075713 |
| K | J + drain-regime priors | 0.074033 |
| L | K + endpoint boundary features | 0.074011 |
| M | L + upstream historical EMA features | 0.065236 |
| N | M + `qin/qout/qnet` historical EMA features | 0.056407 |
| O | N + surcharge expert + deep-storage expert | 0.051369 |
| P | O + split directional `qnet` history EMA | 0.048904 |

*Note: M was used as the final model, as models N, O, and P performed worse on the public Kaggle leaderboard
