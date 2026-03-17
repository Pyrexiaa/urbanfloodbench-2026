"""v9b final model + OOF check + test submission. CVスキップ版."""
import os, sys, pickle, time, numpy as np, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_model_config, load_event_data, list_events
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events
from src.model_lstm1d import build_adjacency
from run_train_m2_v11c_multistep import MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM
from run_inference_ensemble import predict_event, build_submission
from residual_correction_v9b import (apply_correction, build_neighbor_maps,
    get_feat_names_1d, get_feat_names_2d, SPIN_UP)
import xgboost as xgb
import pandas as pd

device = "cuda"
BASE = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
cache_dir = os.path.join(BASE, "Models", "checkpoints")

config = load_model_config(data_dir, MODEL_ID)
with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
    norm_stats = pickle.load(f)
with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
    per_node_stats = pickle.load(f)
std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")

_, _, degree_1d = build_adjacency(config.edge_index_1d, config.num_1d_nodes)
conn = config.connections_1d2d
coupled_set = set(conn[:, 0])
is_coupled = np.array([1 if i in coupled_set else 0 for i in range(config.num_1d_nodes)])
node_feats = {"degree": degree_1d, "is_coupled": is_coupled}
sd_1d = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
sd_2d = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0
coupled_2d_set = set(conn[:, 1])
(nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
 map_1d_to_2d, map_2d_to_1d) = build_neighbor_maps(config)
fn_1d = get_feat_names_1d(sd_1d)
fn_2d = get_feat_names_2d(sd_2d)

# Load v9b cache
print("Loading v9b cache...")
with open(os.path.join(cache_dir, "lgbm_v9b_step1_cache.pkl"), "rb") as f:
    cache = pickle.load(f)
X_1d = cache["X_1d"]; y_1d = cache["y_1d"]
X_2d = cache["X_2d"]; y_2d = cache["y_2d"]
per_event_pred = cache["per_event_pred"]
del cache; gc.collect()
assert X_1d.shape[1] == len(fn_1d), f"1D mismatch: {X_1d.shape[1]} vs {len(fn_1d)}"
assert X_2d.shape[1] == len(fn_2d), f"2D mismatch: {X_2d.shape[1]} vs {len(fn_2d)}"
print(f"  X_1d={X_1d.shape}, X_2d={X_2d.shape}")

# Median rounds from previous CV
ROUNDS_1D = 124217
ROUNDS_2D = 179919

params_1d = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.01, "max_leaves": 127, "max_depth": 10,
    "min_child_weight": 50, "colsample_bytree": 0.8, "subsample": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "tree_method": "gpu_hist", "device": "cuda", "verbosity": 0,
}
params_2d = params_1d.copy()
params_2d["max_leaves"] = 63

# === Step 4: Final model ===
print(f"\n--- Step 4: Final model (1D={ROUNDS_1D}, 2D={ROUNDS_2D}) ---")
t0 = time.time()
dt_1d = xgb.DMatrix(X_1d, label=y_1d, feature_names=fn_1d)
print(f"  1D DMatrix OK ({time.time()-t0:.0f}s)")
fm1d = xgb.train(params_1d, dt_1d, num_boost_round=ROUNDS_1D)
print(f"  1D trained ({time.time()-t0:.0f}s)")
del dt_1d; gc.collect()

dt_2d = xgb.DMatrix(X_2d, label=y_2d, feature_names=fn_2d)
print(f"  2D DMatrix OK ({time.time()-t0:.0f}s)")
fm2d = xgb.train(params_2d, dt_2d, num_boost_round=ROUNDS_2D)
print(f"  2D trained ({time.time()-t0:.0f}s)")
del dt_2d; gc.collect()

save_path = os.path.join(cache_dir, "lgbm_residual_v9b.pkl")
with open(save_path, "wb") as f:
    pickle.dump({"final_model_1d": fm1d, "final_model_2d": fm2d,
                 "feat_names_1d": fn_1d, "feat_names_2d": fn_2d}, f)
print(f"  Saved: {save_path}")

# Feature importance
imp1 = fm1d.get_score(importance_type="gain")
top5_1d = sorted(imp1.items(), key=lambda x: -x[1])[:5]
print(f"  1D top5: {', '.join(f'{n}={v:.0f}' for n,v in top5_1d)}")
imp2 = fm2d.get_score(importance_type="gain")
top5_2d = sorted(imp2.items(), key=lambda x: -x[1])[:5]
print(f"  2D top5: {', '.join(f'{n}={v:.0f}' for n,v in top5_2d)}")

# OOF sanity check (in-sample, 10 events)
train_ids_arr = list_events(data_dir, MODEL_ID, "train")
train_ids = train_ids_arr.tolist() if hasattr(train_ids_arr, "tolist") else list(train_ids_arr)
print(f"\n--- OOF sanity check (in-sample, 10 events) ---")
raw_s2, corr_s2 = [], []
for eid in train_ids[:10]:
    ed = per_event_pred[eid]
    ec = load_event_data(data_dir, MODEL_ID, eid, config)
    c1, c2 = apply_correction(ed["pred_1d"], ed["pred_2d"], ec, config, node_feats,
        fm1d, fm2d, fn_1d, fn_2d, sd_1d, sd_2d, coupled_2d_set,
        nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
        map_1d_to_2d, map_2d_to_1d, per_node_stats=per_node_stats)
    rr2 = np.sqrt(np.mean((ed["pred_2d"] - ed["gt_2d"])**2, axis=0))
    rc2 = np.sqrt(np.mean((c2 - ed["gt_2d"])**2, axis=0))
    raw_s2.append(np.mean(rr2 / std_2d)); corr_s2.append(np.mean(rc2 / std_2d))
r2, c2v = np.mean(raw_s2), np.mean(corr_s2)
print(f"  2D SRMSE (10 events): {r2:.4f} -> {c2v:.4f} ({(1-c2v/r2)*100:.1f}%)")
if c2v >= r2:
    print("  WARNING: correction worsening! Aborting.")
    sys.exit(1)

# === Step 5: Test submission ===
del X_1d, y_1d, X_2d, y_2d, per_event_pred; gc.collect()

import torch
ckpt_path = os.path.join(cache_dir, "best_model_2_v76_aligned_r400.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = HeteroFloodGNNv11(
    hidden_dim=128, num_processor_layers=4, noise_std=0.0,
    coupling_edge_dim=ckpt.get("coupling_edge_dim", COUPLING_EDGE_DIM),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"\n--- Step 5: Test submission ---")
test_ids_arr = list_events(data_dir, MODEL_ID, "test")
test_ids = test_ids_arr.tolist() if hasattr(test_ids_arr, "tolist") else list(test_ids_arr)

all_preds = []
t0 = time.time()
for i, eid in enumerate(test_ids):
    with torch.no_grad():
        event = load_event_data(data_dir, MODEL_ID, eid, config, split="test")
        preds = predict_event(model, config, event, norm_stats, device,
                              spin_up=SPIN_UP, future_rain_steps=FUTURE_RAIN_STEPS,
                              coupling_features=True, is_v11=True, per_node_stats=per_node_stats)
    p1d = np.array(preds["pred_1d"]); p2d = np.array(preds["pred_2d"])
    ec = load_event_data(data_dir, MODEL_ID, eid, config, split="test")
    c1, c2 = apply_correction(p1d, p2d, ec, config, node_feats, fm1d, fm2d, fn_1d, fn_2d,
        sd_1d, sd_2d, coupled_2d_set,
        nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
        map_1d_to_2d, map_2d_to_1d, per_node_stats=per_node_stats)
    all_preds.append((MODEL_ID, eid, {"pred_1d": c1, "pred_2d": c2}))
    torch.cuda.empty_cache()
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(test_ids)} ({time.time()-t0:.1f}s)")
print(f"  Done in {time.time()-t0:.1f}s")

best_sub = pd.read_parquet(os.path.join(BASE, "Dataset_Rerelease", "submission_v76_3seed_30zone.parquet"))
m1_rows = best_sub[best_sub["model_id"] == 1].copy()
m2 = build_submission(all_preds)
final = pd.concat([m1_rows, m2], ignore_index=True)
final = final.sort_values(["model_id", "event_id", "node_type", "node_id"]).reset_index(drop=True)
final["row_id"] = range(len(final))
out = os.path.join(BASE, "Dataset_Rerelease", "submission_v76_lgbm_v9b.parquet")
final.to_parquet(out, index=False)
print(f"  {len(final)} rows -> {out}")
print("DONE!")
