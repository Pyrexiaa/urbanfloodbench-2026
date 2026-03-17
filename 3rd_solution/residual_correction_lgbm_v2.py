"""Residual Correction v2: 1D+2D, 強化特徴量, 2000ラウンド, test submission生成.

v1からの改善:
- late_rain, peak_position, rain_duration, early_rain, mid_rain 追加
- 2Dノードのresidual補正追加
- num_boost_round 500→2000
- step_stride 5→3 (データ量増加)
- test推論 + submission parquet生成
"""
import os, sys, pickle, time
import numpy as np
import torch
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import load_model_config, load_event_data, build_graph_at_timestep, list_events
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events
from src.model_lstm1d import build_adjacency
from run_train_m2_v11c_multistep import (
    mask_2d_edges_only, MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM,
)

device = "cuda"
data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
cache_dir = os.path.join(BASE, "Models", "checkpoints")
SPIN_UP = 10
STEP_STRIDE_1D = 3
STEP_STRIDE_2D = 10  # 2Dは4299ノードなので粗くサンプル
N_ZONES = 10


def rollout_event_residuals(model, event, config, norm_stats, per_node_stats):
    """AR rollout returning per-step, per-node residual + predicted values."""
    max_t = event.nodes_1d_dynamic.shape[0] - 1
    n_1d = event.nodes_1d_dynamic.shape[1]
    n_2d = event.nodes_2d_dynamic.shape[1]

    pn_std_1d = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_2d = torch.tensor(per_node_stats["2d_wl_std"], dtype=torch.float32, device=device)

    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    dyn_e1d = event.edges_1d_dynamic.copy()
    orig_1d = event.nodes_1d_dynamic
    orig_2d = event.nodes_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = dyn_e1d
    dyn_2d[:, :, 2] = 0.0

    cur_1d = torch.tensor(dyn_1d[SPIN_UP, :, 0], dtype=torch.float32, device=device)
    cur_2d = torch.tensor(dyn_2d[SPIN_UP, :, 1], dtype=torch.float32, device=device)

    T = max_t - SPIN_UP
    res_1d = np.zeros((T, n_1d), dtype=np.float32)
    res_2d = np.zeros((T, n_2d), dtype=np.float32)
    pred_1d_all = np.zeros((T, n_1d), dtype=np.float32)
    pred_2d_all = np.zeros((T, n_2d), dtype=np.float32)
    rain_per_step = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for step in range(T):
            t = SPIN_UP + step
            if t + 1 > max_t:
                break

            if step > 0:
                dyn_1d[t, :, 0] = cur_1d.cpu().numpy()
                dyn_2d[t, :, 1] = cur_2d.cpu().numpy()
            mask_2d_edges_only(event, t)
            if t > 0:
                mask_2d_edges_only(event, t - 1)

            g = build_graph_at_timestep(
                config, event, t, prev_t=t-1, norm_stats=norm_stats,
                future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                per_node_stats=per_node_stats,
            ).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(g)

            p1d = out["1d"].float()
            delta_1d = p1d[:, 0] * pn_std_1d
            cur_1d = cur_1d + delta_1d

            p2d = out["2d"].float()
            delta_2d = p2d[:, 0] * pn_std_2d
            cur_2d = cur_2d + delta_2d

            if t + 1 < dyn_1d.shape[0]:
                dyn_1d[t+1, :, 1] = p1d[:, 1].cpu().numpy()
            if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                ep = out["1d_edge"].float().cpu().numpy()
                dyn_e1d[t+1, :, 0] = ep[:, 0]
                dyn_e1d[t+1, :, 1] = ep[:, 1]

            c1d = cur_1d.cpu().numpy()
            c2d = cur_2d.cpu().numpy()
            pred_1d_all[step] = c1d
            pred_2d_all[step] = c2d
            res_1d[step] = c1d - orig_1d[t+1, :, 0]
            res_2d[step] = c2d - orig_2d[t+1, :, 1]
            rain_per_step[step] = np.mean(orig_2d[t, :, 0])

    event.nodes_1d_dynamic = orig_1d
    event.nodes_2d_dynamic = orig_2d

    return {
        "T": T,
        "res_1d": res_1d[:T],
        "res_2d": res_2d[:T],
        "pred_1d": pred_1d_all[:T],
        "pred_2d": pred_2d_all[:T],
        "rain": rain_per_step[:T],
    }


def rollout_event_test(model, event, config, norm_stats, per_node_stats):
    """Test event rollout (no GT available) — returns predictions only."""
    max_t = event.num_timesteps - 1  # consistent with predict_event
    n_1d = config.num_1d_nodes
    n_2d = config.num_2d_nodes

    pn_std_1d = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_2d = torch.tensor(per_node_stats["2d_wl_std"], dtype=torch.float32, device=device)

    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    dyn_e1d = event.edges_1d_dynamic.copy()
    event_1d_orig = event.nodes_1d_dynamic
    event_2d_orig = event.nodes_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = dyn_e1d
    dyn_2d[:, :, 2] = 0.0

    cur_1d = torch.tensor(dyn_1d[SPIN_UP, :, 0], dtype=torch.float32, device=device)
    cur_2d = torch.tensor(dyn_2d[SPIN_UP, :, 1], dtype=torch.float32, device=device)

    T = max_t - SPIN_UP
    pred_1d_all = np.zeros((T, n_1d), dtype=np.float32)
    pred_2d_all = np.zeros((T, n_2d), dtype=np.float32)
    rain_per_step = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for step in range(T):
            t = SPIN_UP + step
            if t + 1 > max_t:
                break

            if step > 0:
                dyn_1d[t, :, 0] = cur_1d.cpu().numpy()
                dyn_2d[t, :, 1] = cur_2d.cpu().numpy()
            mask_2d_edges_only(event, t)
            if t > 0:
                mask_2d_edges_only(event, t - 1)

            g = build_graph_at_timestep(
                config, event, t, prev_t=t-1, norm_stats=norm_stats,
                future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                per_node_stats=per_node_stats,
            ).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(g)

            p1d = out["1d"].float()
            delta_1d = p1d[:, 0] * pn_std_1d
            cur_1d = cur_1d + delta_1d

            p2d = out["2d"].float()
            delta_2d = p2d[:, 0] * pn_std_2d
            cur_2d = cur_2d + delta_2d

            if t + 1 < dyn_1d.shape[0]:
                dyn_1d[t+1, :, 1] = p1d[:, 1].cpu().numpy()
            if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                ep = out["1d_edge"].float().cpu().numpy()
                dyn_e1d[t+1, :, 0] = ep[:, 0]
                dyn_e1d[t+1, :, 1] = ep[:, 1]

            pred_1d_all[step] = cur_1d.cpu().numpy()
            pred_2d_all[step] = cur_2d.cpu().numpy()
            rain_per_step[step] = np.mean(event_2d_orig[t, :, 0])

    event.nodes_1d_dynamic = event_1d_orig
    event.nodes_2d_dynamic = event_2d_orig

    return {
        "T": T,
        "pred_1d": pred_1d_all[:T],
        "pred_2d": pred_2d_all[:T],
        "rain": rain_per_step[:T],
    }


def compute_rain_features(event, T):
    """イベントの降雨特徴量を計算."""
    rain = event.nodes_2d_dynamic[:, :, 0].mean(axis=1)  # [T_full]
    total_rain = float(rain.sum())
    peak_rain = float(rain.max())
    peak_position = int(np.argmax(rain))
    rain_duration = int(np.sum(rain > 0.01))

    # R1で効果的だった特徴量
    T_full = len(rain)
    early_rain = float(rain[:min(100, T_full)].sum())
    mid_rain = float(rain[min(100, T_full):min(200, T_full)].sum())
    late_rain = float(rain[min(200, T_full):].sum())

    cum_rain = np.cumsum(rain[SPIN_UP:SPIN_UP + T])
    return {
        "total_rain": total_rain,
        "peak_rain": peak_rain,
        "peak_position": peak_position,
        "rain_duration": rain_duration,
        "early_rain": early_rain,
        "mid_rain": mid_rain,
        "late_rain": late_rain,
        "rain_full": rain,
        "cum_rain": cum_rain,
    }


def build_features_1d(event, rollout, config, node_feats, rain_feats, step_stride):
    """1Dノードの特徴量行列を構築."""
    n_1d = config.num_1d_nodes
    T = rollout["T"]
    static_dim = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
    wl_init = event.nodes_1d_dynamic[SPIN_UP, :, 0]
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]

    steps = list(range(0, T, step_stride))
    n_steps = len(steps)

    # 特徴量次元数
    n_feat = (2 + static_dim + 15)  # degree, coupled, static(6), 3 step, 4 rain, 2 pred, 4 event-level

    X = np.zeros((n_steps * n_1d, n_feat), dtype=np.float32)
    idx = 0
    for si, step in enumerate(steps):
        rain_now = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rain_l10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rain_now
        rain_l30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rain_now

        for ni in range(n_1d):
            row = []
            row.append(node_feats["degree"][ni])
            row.append(node_feats["is_coupled"][ni])
            if static_dim > 0:
                for si2 in range(static_dim):
                    row.append(config.nodes_1d_static[ni, si2])
            row.append(step / max(T - 1, 1))             # step_ratio
            row.append(step // max(T // N_ZONES, 1))      # step_zone
            row.append(step)                               # raw step (NEW)
            row.append(cum_rain[step] if step < len(cum_rain) else cum_rain[-1])
            row.append(rain_now)
            row.append(rain_l10)
            row.append(rain_l30)                           # NEW: 30-step rain avg
            row.append(rollout["pred_1d"][step, ni])
            row.append(rollout["pred_1d"][step, ni] - wl_init[ni])
            row.append(rain_feats["total_rain"])
            row.append(rain_feats["peak_rain"])
            row.append(rain_feats["peak_position"])        # NEW
            row.append(rain_feats["rain_duration"])         # NEW
            row.append(rain_feats["late_rain"])             # NEW (R1最重要)
            row.append(T)

            X[idx] = row
            idx += 1

    return X[:idx]


def build_features_2d(event, rollout, config, rain_feats, step_stride):
    """2Dノードの特徴量行列を構築."""
    n_2d = config.num_2d_nodes
    T = rollout["T"]
    static_dim = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0
    wl_init = event.nodes_2d_dynamic[SPIN_UP, :, 1]
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]

    # 2D coupling情報
    conn = config.connections_1d2d
    coupled_2d = set(conn[:, 1])

    steps = list(range(0, T, step_stride))
    n_steps = len(steps)

    # 2Dは静的特徴 + step + rain + pred
    n_feat = (1 + static_dim + 13)
    X = np.zeros((n_steps * n_2d, n_feat), dtype=np.float32)
    idx = 0
    for si, step in enumerate(steps):
        rain_now = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rain_l10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rain_now
        rain_l30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rain_now

        for ni in range(n_2d):
            row = []
            row.append(1 if ni in coupled_2d else 0)
            if static_dim > 0:
                for si2 in range(static_dim):
                    row.append(config.nodes_2d_static[ni, si2])
            row.append(step / max(T - 1, 1))
            row.append(step // max(T // N_ZONES, 1))
            row.append(step)
            row.append(cum_rain[step] if step < len(cum_rain) else cum_rain[-1])
            row.append(rain_now)
            row.append(rain_l10)
            row.append(rain_l30)
            row.append(rollout["pred_2d"][step, ni])
            row.append(rollout["pred_2d"][step, ni] - wl_init[ni])
            row.append(rain_feats["total_rain"])
            row.append(rain_feats["peak_rain"])
            row.append(rain_feats["late_rain"])
            row.append(T)

            X[idx] = row
            idx += 1

    return X[:idx]


def get_feat_names_1d(static_dim):
    names = ["degree", "is_coupled"]
    for si in range(static_dim):
        names.append(f"static_{si}")
    names.extend(["step_ratio", "step_zone", "raw_step",
                   "cum_rain", "rain_now", "rain_last10", "rain_last30",
                   "pred_val", "pred_delta",
                   "total_rain", "peak_rain", "peak_position", "rain_duration",
                   "late_rain", "event_T"])
    return names


def get_feat_names_2d(static_dim):
    names = ["is_coupled"]
    for si in range(static_dim):
        names.append(f"static_{si}")
    names.extend(["step_ratio", "step_zone", "raw_step",
                   "cum_rain", "rain_now", "rain_last10", "rain_last30",
                   "pred_val", "pred_delta",
                   "total_rain", "peak_rain", "late_rain", "event_T"])
    return names


def main():
    print("=" * 70)
    print("  Residual Correction v2: 1D+2D, enhanced features, test submission")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    # Load v76 model
    ckpt_path = os.path.join(cache_dir, "best_model_2_v76_aligned_r400.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=ckpt.get("coupling_edge_dim", COUPLING_EDGE_DIM),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model: v76, val={ckpt.get('best_val', '?')}")

    # Node features
    _, _, degree_1d = build_adjacency(config.edge_index_1d, config.num_1d_nodes)
    conn = config.connections_1d2d
    coupled_set = set(conn[:, 0])
    is_coupled = np.array([1 if i in coupled_set else 0 for i in range(config.num_1d_nodes)])
    node_feats = {"degree": degree_1d, "is_coupled": is_coupled}

    n_1d = config.num_1d_nodes
    n_2d = config.num_2d_nodes
    static_dim_1d = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
    static_dim_2d = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0

    train_ids_arr = list_events(data_dir, MODEL_ID, "train")
    train_ids = train_ids_arr.tolist() if hasattr(train_ids_arr, 'tolist') else list(train_ids_arr)

    # ===== Step 1: Compute train residuals =====
    print("\n--- Step 1: Train rollout ---")
    train_rollouts = {}
    train_rain_feats = {}
    t0 = time.time()
    for i, eid in enumerate(train_ids):
        event = load_event_data(data_dir, MODEL_ID, eid, config)
        r = rollout_event_residuals(model, event, config, norm_stats, per_node_stats)
        rf = compute_rain_features(event, r["T"])
        train_rollouts[eid] = r
        train_rain_feats[eid] = rf
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(train_ids)} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ===== Step 2: Build 1D features =====
    print("\n--- Step 2: Building features ---")
    X_1d_list = []
    y_1d_list = []
    event_row_ranges_1d = []  # (start, end) per event
    cursor = 0

    for eid in train_ids:
        event = load_event_data(data_dir, MODEL_ID, eid, config)
        r = train_rollouts[eid]
        rf = train_rain_feats[eid]
        X = build_features_1d(event, r, config, node_feats, rf, STEP_STRIDE_1D)
        T = r["T"]
        steps = list(range(0, T, STEP_STRIDE_1D))
        y = np.array([r["res_1d"][s, ni] for s in steps for ni in range(n_1d)], dtype=np.float32)
        X_1d_list.append(X)
        y_1d_list.append(y)
        event_row_ranges_1d.append((cursor, cursor + len(y)))
        cursor += len(y)

    X_1d = np.concatenate(X_1d_list)
    y_1d = np.concatenate(y_1d_list)
    print(f"  1D: X={X_1d.shape}, y={y_1d.shape}")

    # Build 2D features
    X_2d_list = []
    y_2d_list = []
    event_row_ranges_2d = []
    cursor = 0

    for eid in train_ids:
        event = load_event_data(data_dir, MODEL_ID, eid, config)
        r = train_rollouts[eid]
        rf = train_rain_feats[eid]
        X = build_features_2d(event, r, config, rf, STEP_STRIDE_2D)
        T = r["T"]
        steps = list(range(0, T, STEP_STRIDE_2D))
        y = np.array([r["res_2d"][s, ni] for s in steps for ni in range(n_2d)], dtype=np.float32)
        X_2d_list.append(X)
        y_2d_list.append(y)
        event_row_ranges_2d.append((cursor, cursor + len(y)))
        cursor += len(y)

    X_2d = np.concatenate(X_2d_list)
    y_2d = np.concatenate(y_2d_list)
    print(f"  2D: X={X_2d.shape}, y={y_2d.shape}")

    # ===== Step 3: LightGBM CV =====
    print("\n--- Step 3: LightGBM Training ---")
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    params_1d = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 10,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }
    params_2d = params_1d.copy()
    params_2d["num_leaves"] = 63  # 2Dはノード数多いので控えめに

    feat_names_1d = get_feat_names_1d(static_dim_1d)
    feat_names_2d = get_feat_names_2d(static_dim_2d)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- 1D CV ---
    print("\n  === 1D LightGBM ===")
    oof_1d = np.zeros(len(y_1d), dtype=np.float32)
    models_1d = []

    for fold, (tr_ev, va_ev) in enumerate(kf.split(train_ids)):
        tr_rows = []
        va_rows = []
        for ei in tr_ev:
            s, e = event_row_ranges_1d[ei]
            tr_rows.extend(range(s, e))
        for ei in va_ev:
            s, e = event_row_ranges_1d[ei]
            va_rows.extend(range(s, e))

        dtrain = lgb.Dataset(X_1d[tr_rows], y_1d[tr_rows], feature_name=feat_names_1d)
        dval = lgb.Dataset(X_1d[va_rows], y_1d[va_rows], feature_name=feat_names_1d, reference=dtrain)

        bst = lgb.train(
            params_1d, dtrain, num_boost_round=2000,
            valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = bst.predict(X_1d[va_rows])
        oof_1d[va_rows] = pred
        models_1d.append(bst)

        rmse_raw = np.sqrt(np.mean(y_1d[va_rows]**2))
        rmse_corr = np.sqrt(np.mean((y_1d[va_rows] - pred)**2))
        print(f"    Fold {fold}: raw={rmse_raw:.4f} -> corr={rmse_corr:.4f} "
              f"({(1-rmse_corr/rmse_raw)*100:.1f}%) rounds={bst.best_iteration}")

    rmse_raw_1d = np.sqrt(np.mean(y_1d**2))
    rmse_corr_1d = np.sqrt(np.mean((y_1d - oof_1d)**2))
    print(f"  1D OOF: raw={rmse_raw_1d:.4f} -> corr={rmse_corr_1d:.4f} ({(1-rmse_corr_1d/rmse_raw_1d)*100:.1f}%)")

    # --- 2D CV ---
    print("\n  === 2D LightGBM ===")
    oof_2d = np.zeros(len(y_2d), dtype=np.float32)
    models_2d = []

    for fold, (tr_ev, va_ev) in enumerate(kf.split(train_ids)):
        tr_rows = []
        va_rows = []
        for ei in tr_ev:
            s, e = event_row_ranges_2d[ei]
            tr_rows.extend(range(s, e))
        for ei in va_ev:
            s, e = event_row_ranges_2d[ei]
            va_rows.extend(range(s, e))

        dtrain = lgb.Dataset(X_2d[tr_rows], y_2d[tr_rows], feature_name=feat_names_2d)
        dval = lgb.Dataset(X_2d[va_rows], y_2d[va_rows], feature_name=feat_names_2d, reference=dtrain)

        bst = lgb.train(
            params_2d, dtrain, num_boost_round=2000,
            valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = bst.predict(X_2d[va_rows])
        oof_2d[va_rows] = pred
        models_2d.append(bst)

        rmse_raw = np.sqrt(np.mean(y_2d[va_rows]**2))
        rmse_corr = np.sqrt(np.mean((y_2d[va_rows] - pred)**2))
        print(f"    Fold {fold}: raw={rmse_raw:.4f} -> corr={rmse_corr:.4f} "
              f"({(1-rmse_corr/rmse_raw)*100:.1f}%) rounds={bst.best_iteration}")

    rmse_raw_2d = np.sqrt(np.mean(y_2d**2))
    rmse_corr_2d = np.sqrt(np.mean((y_2d - oof_2d)**2))
    print(f"  2D OOF: raw={rmse_raw_2d:.4f} -> corr={rmse_corr_2d:.4f} ({(1-rmse_corr_2d/rmse_raw_2d)*100:.1f}%)")

    # ===== SRMSE Impact =====
    print("\n--- SRMSE Impact (OOF) ---")
    # Compute per-event SRMSE
    srmse_raw_1d_list = []
    srmse_corr_1d_list = []
    srmse_raw_2d_list = []
    srmse_corr_2d_list = []

    for ei, eid in enumerate(train_ids):
        r = train_rollouts[eid]
        T = r["T"]

        # 1D
        steps_1d = list(range(0, T, STEP_STRIDE_1D))
        n_steps_1d = len(steps_1d)
        s1, e1 = event_row_ranges_1d[ei]
        raw_1d = y_1d[s1:e1].reshape(n_steps_1d, n_1d)
        corr_1d = (y_1d[s1:e1] - oof_1d[s1:e1]).reshape(n_steps_1d, n_1d)
        srmse_raw_1d_list.append(np.mean(np.sqrt(np.mean(raw_1d**2, axis=0)) / std_1d))
        srmse_corr_1d_list.append(np.mean(np.sqrt(np.mean(corr_1d**2, axis=0)) / std_1d))

        # 2D
        steps_2d = list(range(0, T, STEP_STRIDE_2D))
        n_steps_2d = len(steps_2d)
        s2, e2 = event_row_ranges_2d[ei]
        raw_2d = y_2d[s2:e2].reshape(n_steps_2d, n_2d)
        corr_2d = (y_2d[s2:e2] - oof_2d[s2:e2]).reshape(n_steps_2d, n_2d)
        srmse_raw_2d_list.append(np.mean(np.sqrt(np.mean(raw_2d**2, axis=0)) / std_2d))
        srmse_corr_2d_list.append(np.mean(np.sqrt(np.mean(corr_2d**2, axis=0)) / std_2d))

    m_raw_1d = np.mean(srmse_raw_1d_list)
    m_corr_1d = np.mean(srmse_corr_1d_list)
    m_raw_2d = np.mean(srmse_raw_2d_list)
    m_corr_2d = np.mean(srmse_corr_2d_list)
    m_raw_total = (m_raw_1d + m_raw_2d) / 2
    m_corr_total = (m_corr_1d + m_corr_2d) / 2

    print(f"  1D SRMSE: {m_raw_1d:.4f} -> {m_corr_1d:.4f} ({(1-m_corr_1d/m_raw_1d)*100:.1f}%)")
    print(f"  2D SRMSE: {m_raw_2d:.4f} -> {m_corr_2d:.4f} ({(1-m_corr_2d/m_raw_2d)*100:.1f}%)")
    print(f"  Total:    {m_raw_total:.4f} -> {m_corr_total:.4f} ({(1-m_corr_total/m_raw_total)*100:.1f}%)")

    # Feature importance
    print("\n--- Feature Importance (1D, last fold) ---")
    imp = models_1d[-1].feature_importance(importance_type="gain")
    for name, val in sorted(zip(feat_names_1d, imp), key=lambda x: -x[1])[:10]:
        print(f"  {name:20s}: {val:.0f}")

    # ===== Step 4: Train final models on ALL data =====
    print("\n--- Step 4: Training final models on all data ---")
    dtrain_1d_full = lgb.Dataset(X_1d, y_1d, feature_name=feat_names_1d)
    # Use median best_iteration from CV
    best_rounds_1d = int(np.median([m.best_iteration for m in models_1d]))
    final_model_1d = lgb.train(params_1d, dtrain_1d_full, num_boost_round=best_rounds_1d)
    print(f"  1D final model: {best_rounds_1d} rounds")

    dtrain_2d_full = lgb.Dataset(X_2d, y_2d, feature_name=feat_names_2d)
    best_rounds_2d = int(np.median([m.best_iteration for m in models_2d]))
    final_model_2d = lgb.train(params_2d, dtrain_2d_full, num_boost_round=best_rounds_2d)
    print(f"  2D final model: {best_rounds_2d} rounds")

    # ===== Step 5: Test inference + correction =====
    print("\n--- Step 5: Test inference with correction ---")
    test_ids_arr = list_events(data_dir, MODEL_ID, "test")
    test_ids = test_ids_arr.tolist() if hasattr(test_ids_arr, 'tolist') else list(test_ids_arr)
    print(f"  Test events: {len(test_ids)}")

    coupled_2d_set = set(conn[:, 1])
    all_predictions = []  # list of (model_id, event_id, {"pred_1d": ..., "pred_2d": ...})
    t0 = time.time()
    for i, eid in enumerate(test_ids):
        event = load_event_data(data_dir, MODEL_ID, eid, config, split="test")
        r = rollout_event_test(model, event, config, norm_stats, per_node_stats)
        rf = compute_rain_features(event, r["T"])
        T = r["T"]
        rain = rf["rain_full"]
        wl_init_1d = event.nodes_1d_dynamic[SPIN_UP, :, 0]
        wl_init_2d = event.nodes_2d_dynamic[SPIN_UP, :, 1]

        # 1D correction: vectorized per-step
        pred_1d_corrected = r["pred_1d"].copy()
        for step in range(T):
            rain_now = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
            rain_l10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rain_now
            rain_l30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rain_now
            cum_r = rf["cum_rain"][step] if step < len(rf["cum_rain"]) else rf["cum_rain"][-1]

            X_step = np.zeros((n_1d, len(feat_names_1d)), dtype=np.float32)
            # Vectorized: set all node features at once
            X_step[:, 0] = node_feats["degree"]
            X_step[:, 1] = node_feats["is_coupled"]
            if static_dim_1d > 0:
                X_step[:, 2:2+static_dim_1d] = config.nodes_1d_static
            base = 2 + static_dim_1d
            X_step[:, base] = step / max(T - 1, 1)
            X_step[:, base+1] = step // max(T // N_ZONES, 1)
            X_step[:, base+2] = step
            X_step[:, base+3] = cum_r
            X_step[:, base+4] = rain_now
            X_step[:, base+5] = rain_l10
            X_step[:, base+6] = rain_l30
            X_step[:, base+7] = r["pred_1d"][step]
            X_step[:, base+8] = r["pred_1d"][step] - wl_init_1d
            X_step[:, base+9] = rf["total_rain"]
            X_step[:, base+10] = rf["peak_rain"]
            X_step[:, base+11] = rf["peak_position"]
            X_step[:, base+12] = rf["rain_duration"]
            X_step[:, base+13] = rf["late_rain"]
            X_step[:, base+14] = T

            correction = final_model_1d.predict(X_step)
            pred_1d_corrected[step] -= correction

        # 2D correction: vectorized per-step
        pred_2d_corrected = r["pred_2d"].copy()
        is_coupled_2d = np.array([1 if ni in coupled_2d_set else 0 for ni in range(n_2d)], dtype=np.float32)
        for step in range(T):
            rain_now = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
            rain_l10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rain_now
            rain_l30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rain_now
            cum_r = rf["cum_rain"][step] if step < len(rf["cum_rain"]) else rf["cum_rain"][-1]

            X_step_2d = np.zeros((n_2d, len(feat_names_2d)), dtype=np.float32)
            X_step_2d[:, 0] = is_coupled_2d
            if static_dim_2d > 0:
                X_step_2d[:, 1:1+static_dim_2d] = config.nodes_2d_static
            base2 = 1 + static_dim_2d
            X_step_2d[:, base2] = step / max(T - 1, 1)
            X_step_2d[:, base2+1] = step // max(T // N_ZONES, 1)
            X_step_2d[:, base2+2] = step
            X_step_2d[:, base2+3] = cum_r
            X_step_2d[:, base2+4] = rain_now
            X_step_2d[:, base2+5] = rain_l10
            X_step_2d[:, base2+6] = rain_l30
            X_step_2d[:, base2+7] = r["pred_2d"][step]
            X_step_2d[:, base2+8] = r["pred_2d"][step] - wl_init_2d
            X_step_2d[:, base2+9] = rf["total_rain"]
            X_step_2d[:, base2+10] = rf["peak_rain"]
            X_step_2d[:, base2+11] = rf["late_rain"]
            X_step_2d[:, base2+12] = T

            correction_2d = final_model_2d.predict(X_step_2d)
            pred_2d_corrected[step] -= correction_2d

        all_predictions.append((MODEL_ID, eid, {
            "pred_1d": pred_1d_corrected,
            "pred_2d": pred_2d_corrected,
        }))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(test_ids)} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ===== Step 6: Build submission using build_submission format =====
    print("\n--- Step 6: Building submission ---")
    from run_inference_ensemble import build_submission

    # Load best existing submission (contains M1 predictions)
    best_sub_path = os.path.join(BASE, "Dataset_Rerelease", "submission_v76_3seed_30zone.parquet")
    best_sub = pd.read_parquet(best_sub_path)
    print(f"  Base submission: {best_sub_path} ({len(best_sub)} rows)")

    # Build M2 submission
    m2_sub = build_submission(all_predictions)

    # Replace M2 rows in best submission
    # model_id == 2 rows
    m1_mask = best_sub["model_id"] == 1
    m2_mask = best_sub["model_id"] == 2

    # M2 from LightGBM correction
    m2_new = m2_sub[m2_sub["model_id"] == MODEL_ID].copy()
    m2_new = m2_new.sort_values(["event_id", "node_type", "node_id"]).reset_index(drop=True)

    # Build final submission: keep M1 from best, replace M2
    m1_rows = best_sub[m1_mask].copy()
    final_sub = pd.concat([m1_rows, m2_new], ignore_index=True)
    final_sub = final_sub.sort_values(["model_id", "event_id", "node_type", "node_id"]).reset_index(drop=True)
    final_sub["row_id"] = range(len(final_sub))

    out_path = os.path.join(BASE, "Dataset_Rerelease", "submission_v76_lgbm_corrected.parquet")
    final_sub.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(final_sub)} rows)")
    print(f"  Submission saved successfully")

    # Save models
    save_path = os.path.join(cache_dir, "lgbm_residual_v2.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "models_1d": models_1d,
            "models_2d": models_2d,
            "final_model_1d": final_model_1d,
            "final_model_2d": final_model_2d,
            "feat_names_1d": feat_names_1d,
            "feat_names_2d": feat_names_2d,
            "params_1d": params_1d,
            "params_2d": params_2d,
            "srmse_1d": (m_raw_1d, m_corr_1d),
            "srmse_2d": (m_raw_2d, m_corr_2d),
        }, f)
    print(f"  Saved models: {save_path}")

    print(f"\n{'=' * 70}")
    print(f"  Summary:")
    print(f"  1D SRMSE: {m_raw_1d:.4f} -> {m_corr_1d:.4f} ({(1-m_corr_1d/m_raw_1d)*100:.1f}%)")
    print(f"  2D SRMSE: {m_raw_2d:.4f} -> {m_corr_2d:.4f} ({(1-m_corr_2d/m_raw_2d)*100:.1f}%)")
    print(f"  Total:    {m_raw_total:.4f} -> {m_corr_total:.4f} ({(1-m_corr_total/m_raw_total)*100:.1f}%)")
    print(f"  Submissions: {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
