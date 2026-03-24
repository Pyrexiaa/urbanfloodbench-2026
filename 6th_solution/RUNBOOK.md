# UrbanFloodBench (UrbanFlood): 

This is the current full-train path for the best Kaggle submission.

## Reproduce From Scratch

### 1) Train the rich full stack

```bash
bash scripts/run_full_train_rich_ema_noclamp_and_predict.sh
bash scripts/train_member_m2_stage5_after_rich_from_full_rich.sh
bash scripts/train_member_m2_stage4_after_stage5_from_full_rich.sh
bash scripts/train_member_m2_stage7_after_stage4_nodewise_graphpost2_fullw1_from_full_rich.sh
```

Expected rich artifact:
- `runs/$(readlink runs/_last_full_rich)/submission_m2_stage7_after_stage4_nodewise_graphpost2_fullw1.parquet`

### 2) Train the nodewise-chain full stack

```bash
bash scripts/run_full_train_nodewise_chain_ema_noclamp_and_predict.sh
bash scripts/train_member_m2_stage7_global_nodewise_graphpost2_fullw1_from_full_nodewise_chain.sh
```

Expected nodewise-chain artifacts:
- `runs/$(readlink runs/_last_full_nodewise_chain)/baseline_a_full.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/baseline_b_full.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/resid1d_m2_stage1_graphpost2_nodewise_ema0999_noclamp_e246.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/resid1d_m2_stage7_rich_nodewise_ema0999_noclamp_s1_e284.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/resid2d_m2_nodewise_ema0999_noclamp_e260.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/resid2d_m1_ema0999_noclamp_e594.pt`
- `runs/$(readlink runs/_last_full_nodewise_chain)/resid1d_m1_stage1_ema0999_noclamp_e116.pt`

### 3) Train the full `v3cpl + inletv12` targeted stack

This step reuses the rich and nodewise-chain runs above.

```bash
BASE_RUN="runs/$(readlink runs/_last_full_nodewise_chain)" \
RICH_RUN="runs/$(readlink runs/_last_full_rich)" \
bash scripts/run_full_train_v3cpl_inletv12_targeted_and_blend.sh
```

Expected artifact:
- `runs/$(readlink runs/_last_full_v3cpl_inletv12_targeted)/submission_blend_rich50__v3cpl_inletv12_b50.parquet`


### 4) Train the full gate-on-top submission

This step reuses:
- the nodewise-chain run from step 2
- the `v3cpl + inletv12` run from step 3

```bash
BEST_OLDBASE_DIR="runs/$(readlink runs/_last_full_nodewise_chain)" \
BEST_RUN_DIR="runs/$(readlink runs/_last_full_v3cpl_inletv12_targeted)" \
bash scripts/run_full_train_gate_best_vs_g_and_predict.sh
```

Final submit parquet:
- `runs/$(readlink runs/_last_full_gate_best_vs_g)/submission_gate_best_vs_g.parquet`

## Canonical Artifacts

- Rich run:
  - `runs/full_rich_ema_seed42_20260226_230903`
- Nodewise-chain run:
  - `runs/full_nodewise_chain_ema_seed42_20260307_1130`
- `v3cpl + inletv12` run:
  - `runs/full_v3cpl_inletv12_targeted_seed42_20260309_144425`
- Gate run:
  - `runs/full_gate_best_vs_g_seed42_gate_best_vs_g_port_20260311`

