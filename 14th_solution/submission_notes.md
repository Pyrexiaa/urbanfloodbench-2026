# Submission Notes

## Final Rank

- Final private rank: `14th`

## Final Relevant Variants

- `v2a`
  - Warmup-aligned autoregressive LightGBM
  - Public LB: `0.0390`

- `v2g`
  - `v2a` + `1D-2D` coupling features on all `2D`
  - Public LB: `0.0389`

- `v2h`
  - `v2a` + conservative `1D-2D` coupling only on `Model 2 / 2D`
  - Public LB: `0.0387`

## Trained Model Availability

I do not currently have separate serialized trained model files saved locally from the Kaggle notebook runs.

What is available:

- the final source code variants
- Kaggle notebook metadata
- experiment notes and postmortem
- submission files / public leaderboard results

## Reproducibility Notes

The code is written for Kaggle notebook execution with:

- competition source: `urban-flood-modelling`
- dataset source: `attiqueansari/urbanfloodbench`

The final code path is in:

- `code/lgbm_auto_v2h.py`

Related baseline and ablation references:

- `code/lgbm_auto_v2a.py`
- `code/lgbm_auto_v2g.py`
