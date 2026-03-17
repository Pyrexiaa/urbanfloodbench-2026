"""v10: v9b + 時間方向特徴量 (pred_step_diff, nb1_step_diff, pred_step_diff2). CVスキップ."""
import os, sys, pickle, time, numpy as np, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_model_config, load_event_data, list_events
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events
from src.model_lstm1d import build_adjacency
from run_train_m2_v11c_multistep import MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM
from run_inference_ensemble import predict_event, build_submission
from residual_correction_v9b import (build_neighbor_maps,
    compute_nb_feats_1d, compute_nb_feats_2d, SPIN_UP, N_ZONES)
import xgboost as xgb
import pandas as pd

device = "cuda"
BASE = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
cache_dir = os.path.join(BASE, "Models", "checkpoints")
STEP_STRIDE_1D = 2
STEP_STRIDE_2D = 5
ROUNDS_1D = 124217
ROUNDS_2D = 179919


def get_feat_names_1d_v10(static_dim):
    names = ["degree", "is_coupled"]
    for si in range(static_dim):
        names.append(f"static_{si}")
    names.extend(["step_ratio", "step_zone", "raw_step",
                   "cum_rain", "rain_now", "rain_last10", "rain_last30",
                   "pred_val", "pred_delta",
                   "total_rain", "peak_rain", "peak_position", "rain_duration",
                   "late_rain", "event_T",
                   "nb1_mean", "nb1_std", "coupled_2d",
                   "nb2_mean", "nb2_std",
                   "nb3_mean", "nb3_std",
                   "laplacian", "steps_since_peak", "node_std",
                   "rain_x_pred_delta", "pred_ratio_nb1", "pred_minus_nb2", "pred_over_nodestd",
                   # v10: 時間方向特徴量
                   "pred_step_diff", "nb1_step_diff", "pred_step_diff2"])
    return names


def get_feat_names_2d_v10(static_dim):
    names = ["is_coupled"]
    for si in range(static_dim):
        names.append(f"static_{si}")
    names.extend(["step_ratio", "step_zone", "raw_step",
                   "cum_rain", "rain_now", "rain_last10", "rain_last30",
                   "pred_val", "pred_delta",
                   "total_rain", "peak_rain", "late_rain", "event_T",
                   "nb1_mean", "nb1_std", "coupled_1d",
                   "nb2_mean", "nb2_std",
                   "nb3_mean", "nb3_std",
                   "laplacian", "steps_since_peak", "node_std",
                   "rain_x_pred_delta", "pred_ratio_nb1", "pred_minus_nb2", "pred_over_nodestd",
                   # v10: 時間方向特徴量
                   "pred_step_diff", "nb1_step_diff", "pred_step_diff2"])
    return names


def build_features_1d_v10(rollout, config, node_feats, rain_feats, step_stride,
                           nb1, nb2, nb3, map_1d_to_2d, per_node_stats=None,
                           wl_init=None):
    """v9b + 時間方向3特徴量."""
    n_1d = config.num_1d_nodes
    T = rollout["T"]
    sd = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]
    peak_pos = rain_feats["peak_position"]
    nstd_1d = per_node_stats["1d_wl_std"] if per_node_stats is not None else np.ones(n_1d)

    steps = list(range(0, T, step_stride))
    n_feat = 2 + sd + 15 + 7 + 7 + 3  # v9b + 3 temporal
    X = np.zeros((len(steps) * n_1d, n_feat), dtype=np.float32)

    # precompute nb1_mean for all steps (for temporal diff)
    pred_1d = rollout["pred_1d"]  # (T, n_1d)
    pred_2d = rollout["pred_2d"]  # (T, n_2d)
    nb1_means_cache = {}

    idx = 0
    prev_step = None
    for si, step in enumerate(steps):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_1d(pred_1d[step], pred_2d[step], nb1, nb2, nb3, map_1d_to_2d)
        nb1_means_cache[step] = nbf[:, 0].copy()

        # temporal: pred[t] - pred[t-1], nb1_mean[t] - nb1_mean[t-1], pred[t] - pred[t-2]
        if step > 0:
            pred_diff = pred_1d[step] - pred_1d[step - 1]  # per node
        else:
            pred_diff = np.zeros(n_1d, dtype=np.float32)

        if step >= 2:
            pred_diff2 = pred_1d[step] - pred_1d[step - 2]
        else:
            pred_diff2 = np.zeros(n_1d, dtype=np.float32)

        # nb1 step diff: need nb1_mean at step-1
        if prev_step is not None and (step - 1) in nb1_means_cache:
            nb1_diff = nb1_means_cache[step] - nb1_means_cache[step - 1]
        elif step > 0:
            # step-1 not in cache (stride > 1), compute directly
            nb1_diff = nb1_means_cache[step] - compute_nb_feats_1d(
                pred_1d[step-1], pred_2d[step-1], nb1, nb2, nb3, map_1d_to_2d)[:, 0]
        else:
            nb1_diff = np.zeros(n_1d, dtype=np.float32)

        for ni in range(n_1d):
            pv = pred_1d[step, ni]
            lap = pv - nbf[ni, 0] if nbf[ni, 0] != 0 else 0.0
            pd = pv - wl_init[ni]
            row = [node_feats["degree"][ni], node_feats["is_coupled"][ni]]
            if sd > 0:
                row.extend(config.nodes_1d_static[ni].tolist())
            row.extend([step / max(T-1, 1), step // max(T // N_ZONES, 1), step,
                        cr, rn, rl10, rl30, pv, pd,
                        rain_feats["total_rain"], rain_feats["peak_rain"],
                        rain_feats["peak_position"], rain_feats["rain_duration"],
                        rain_feats["late_rain"], T])
            row.extend(nbf[ni].tolist())
            nb1_mean = nbf[ni, 0]
            nb2_mean = nbf[ni, 3]
            row.extend([
                lap, ssp, nstd_1d[ni],
                rn * pd,
                pv / (nb1_mean + 1e-8) if nb1_mean != 0 else 1.0,
                pv - nb2_mean,
                pv / (nstd_1d[ni] + 1e-8),
                # v10 temporal
                pred_diff[ni], nb1_diff[ni], pred_diff2[ni],
            ])
            X[idx] = row
            idx += 1
        prev_step = step
    return X[:idx]


def build_features_2d_v10(rollout, config, rain_feats, step_stride,
                           nb1, nb2, nb3, map_2d_to_1d, per_node_stats=None,
                           wl_init=None):
    """v9b + 時間方向3特徴量."""
    n_2d = config.num_2d_nodes
    T = rollout["T"]
    sd = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]
    coupled_2d = set(config.connections_1d2d[:, 1])
    peak_pos = rain_feats["peak_position"]
    nstd_2d = per_node_stats["2d_wl_std"] if per_node_stats is not None else np.ones(n_2d)

    steps = list(range(0, T, step_stride))
    n_feat = 1 + sd + 13 + 7 + 7 + 3  # v9b + 3 temporal
    X = np.zeros((len(steps) * n_2d, n_feat), dtype=np.float32)

    pred_1d = rollout["pred_1d"]
    pred_2d = rollout["pred_2d"]
    nb1_means_cache = {}

    idx = 0
    for si, step in enumerate(steps):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_2d(pred_2d[step], pred_1d[step], nb1, nb2, nb3, map_2d_to_1d)
        nb1_means_cache[step] = nbf[:, 0].copy()

        if step > 0:
            pred_diff = pred_2d[step] - pred_2d[step - 1]
        else:
            pred_diff = np.zeros(n_2d, dtype=np.float32)

        if step >= 2:
            pred_diff2 = pred_2d[step] - pred_2d[step - 2]
        else:
            pred_diff2 = np.zeros(n_2d, dtype=np.float32)

        if step > 0:
            if (step - 1) in nb1_means_cache:
                nb1_diff = nb1_means_cache[step] - nb1_means_cache[step - 1]
            else:
                nb1_diff = nb1_means_cache[step] - compute_nb_feats_2d(
                    pred_2d[step-1], pred_1d[step-1], nb1, nb2, nb3, map_2d_to_1d)[:, 0]
        else:
            nb1_diff = np.zeros(n_2d, dtype=np.float32)

        for ni in range(n_2d):
            pv = pred_2d[step, ni]
            lap = pv - nbf[ni, 0] if nbf[ni, 0] != 0 else 0.0
            pd = pv - wl_init[ni]
            row = [1 if ni in coupled_2d else 0]
            if sd > 0:
                row.extend(config.nodes_2d_static[ni].tolist())
            row.extend([step / max(T-1, 1), step // max(T // N_ZONES, 1), step,
                        cr, rn, rl10, rl30, pv, pd,
                        rain_feats["total_rain"], rain_feats["peak_rain"],
                        rain_feats["late_rain"], T])
            row.extend(nbf[ni].tolist())
            nb1_mean = nbf[ni, 0]
            nb2_mean = nbf[ni, 3]
            row.extend([
                lap, ssp, nstd_2d[ni],
                rn * pd,
                pv / (nb1_mean + 1e-8) if nb1_mean != 0 else 1.0,
                pv - nb2_mean,
                pv / (nstd_2d[ni] + 1e-8),
                # v10 temporal
                pred_diff[ni], nb1_diff[ni], pred_diff2[ni],
            ])
            X[idx] = row
            idx += 1
    return X[:idx]


def apply_correction_v10(pred_1d, pred_2d, event, config, node_feats,
                          model_1d, model_2d, fn_1d, fn_2d,
                          sd_1d, sd_2d, coupled_2d_set,
                          nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
                          map_1d_to_2d, map_2d_to_1d, per_node_stats=None):
    """v9b apply_correction + 時間方向3特徴量."""
    import xgboost as xgb
    T = pred_1d.shape[0]
    n_1d = config.num_1d_nodes
    n_2d = config.num_2d_nodes
    rain = event.nodes_2d_dynamic[:, :, 0].mean(axis=1)
    total_rain = float(rain.sum())
    peak_rain = float(rain.max())
    peak_pos = int(np.argmax(rain))
    rain_dur = int(np.sum(rain > 0.01))
    T_full = len(rain)
    late_rain = float(rain[min(200, T_full):].sum())
    cum_rain = np.cumsum(rain[SPIN_UP:SPIN_UP + T])
    wl_init_1d = event.nodes_1d_dynamic[SPIN_UP - 1, :, 0]
    wl_init_2d = event.nodes_2d_dynamic[SPIN_UP - 1, :, 1]
    is_c2d = np.array([1 if ni in coupled_2d_set else 0 for ni in range(n_2d)], dtype=np.float32)
    nstd_1d = per_node_stats["1d_wl_std"] if per_node_stats is not None else np.ones(n_1d)
    nstd_2d = per_node_stats["2d_wl_std"] if per_node_stats is not None else np.ones(n_2d)

    # 1D
    X_all_1d = np.zeros((T * n_1d, len(fn_1d)), dtype=np.float32)
    prev_nb1m_1d = np.zeros(n_1d, dtype=np.float32)
    for step in range(T):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_1d(pred_1d[step], pred_2d[step], nb1_1d, nb2_1d, nb3_1d, map_1d_to_2d)
        s = step * n_1d
        X_all_1d[s:s+n_1d, 0] = node_feats["degree"]
        X_all_1d[s:s+n_1d, 1] = node_feats["is_coupled"]
        b = 2
        if sd_1d > 0:
            X_all_1d[s:s+n_1d, b:b+sd_1d] = config.nodes_1d_static; b += sd_1d
        X_all_1d[s:s+n_1d, b] = step / max(T-1, 1)
        X_all_1d[s:s+n_1d, b+1] = step // max(T // N_ZONES, 1)
        X_all_1d[s:s+n_1d, b+2] = step; X_all_1d[s:s+n_1d, b+3] = cr
        X_all_1d[s:s+n_1d, b+4] = rn; X_all_1d[s:s+n_1d, b+5] = rl10; X_all_1d[s:s+n_1d, b+6] = rl30
        X_all_1d[s:s+n_1d, b+7] = pred_1d[step]; X_all_1d[s:s+n_1d, b+8] = pred_1d[step] - wl_init_1d
        X_all_1d[s:s+n_1d, b+9] = total_rain; X_all_1d[s:s+n_1d, b+10] = peak_rain
        X_all_1d[s:s+n_1d, b+11] = peak_pos; X_all_1d[s:s+n_1d, b+12] = rain_dur
        X_all_1d[s:s+n_1d, b+13] = late_rain; X_all_1d[s:s+n_1d, b+14] = T
        X_all_1d[s:s+n_1d, b+15:b+22] = nbf
        lap_1d = pred_1d[step] - nbf[:, 0]
        pd_1d = pred_1d[step] - wl_init_1d
        nb1m = nbf[:, 0]; nb2m = nbf[:, 3]
        X_all_1d[s:s+n_1d, b+22] = lap_1d
        X_all_1d[s:s+n_1d, b+23] = ssp
        X_all_1d[s:s+n_1d, b+24] = nstd_1d
        X_all_1d[s:s+n_1d, b+25] = rn * pd_1d
        safe_nb1 = np.where(nb1m != 0, nb1m, 1e-8)
        X_all_1d[s:s+n_1d, b+26] = np.where(nb1m != 0, pred_1d[step] / safe_nb1, 1.0)
        X_all_1d[s:s+n_1d, b+27] = pred_1d[step] - nb2m
        X_all_1d[s:s+n_1d, b+28] = pred_1d[step] / (nstd_1d + 1e-8)
        # v10 temporal
        X_all_1d[s:s+n_1d, b+29] = pred_1d[step] - pred_1d[step-1] if step > 0 else 0
        X_all_1d[s:s+n_1d, b+30] = nb1m - prev_nb1m_1d if step > 0 else 0
        X_all_1d[s:s+n_1d, b+31] = pred_1d[step] - pred_1d[step-2] if step >= 2 else 0
        prev_nb1m_1d = nb1m.copy()

    preds_1d = model_1d.predict(xgb.DMatrix(X_all_1d, feature_names=fn_1d))
    corr_1d = pred_1d.copy()
    for step in range(T):
        corr_1d[step] -= preds_1d[step * n_1d:(step + 1) * n_1d]

    # 2D
    X_all_2d = np.zeros((T * n_2d, len(fn_2d)), dtype=np.float32)
    prev_nb1m_2d = np.zeros(n_2d, dtype=np.float32)
    for step in range(T):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_2d(pred_2d[step], pred_1d[step], nb1_2d, nb2_2d, nb3_2d, map_2d_to_1d)
        s = step * n_2d
        X_all_2d[s:s+n_2d, 0] = is_c2d
        b = 1
        if sd_2d > 0:
            X_all_2d[s:s+n_2d, b:b+sd_2d] = config.nodes_2d_static; b += sd_2d
        X_all_2d[s:s+n_2d, b] = step / max(T-1, 1)
        X_all_2d[s:s+n_2d, b+1] = step // max(T // N_ZONES, 1)
        X_all_2d[s:s+n_2d, b+2] = step; X_all_2d[s:s+n_2d, b+3] = cr
        X_all_2d[s:s+n_2d, b+4] = rn; X_all_2d[s:s+n_2d, b+5] = rl10; X_all_2d[s:s+n_2d, b+6] = rl30
        X_all_2d[s:s+n_2d, b+7] = pred_2d[step]; X_all_2d[s:s+n_2d, b+8] = pred_2d[step] - wl_init_2d
        X_all_2d[s:s+n_2d, b+9] = total_rain; X_all_2d[s:s+n_2d, b+10] = peak_rain
        X_all_2d[s:s+n_2d, b+11] = late_rain; X_all_2d[s:s+n_2d, b+12] = T
        X_all_2d[s:s+n_2d, b+13:b+20] = nbf
        lap_2d = pred_2d[step] - nbf[:, 0]
        pd_2d = pred_2d[step] - wl_init_2d
        nb1m = nbf[:, 0]; nb2m = nbf[:, 3]
        X_all_2d[s:s+n_2d, b+20] = lap_2d
        X_all_2d[s:s+n_2d, b+21] = ssp
        X_all_2d[s:s+n_2d, b+22] = nstd_2d
        X_all_2d[s:s+n_2d, b+23] = rn * pd_2d
        safe_nb1 = np.where(nb1m != 0, nb1m, 1e-8)
        X_all_2d[s:s+n_2d, b+24] = np.where(nb1m != 0, pred_2d[step] / safe_nb1, 1.0)
        X_all_2d[s:s+n_2d, b+25] = pred_2d[step] - nb2m
        X_all_2d[s:s+n_2d, b+26] = pred_2d[step] / (nstd_2d + 1e-8)
        # v10 temporal
        X_all_2d[s:s+n_2d, b+27] = pred_2d[step] - pred_2d[step-1] if step > 0 else 0
        X_all_2d[s:s+n_2d, b+28] = nb1m - prev_nb1m_2d if step > 0 else 0
        X_all_2d[s:s+n_2d, b+29] = pred_2d[step] - pred_2d[step-2] if step >= 2 else 0
        prev_nb1m_2d = nb1m.copy()

    preds_2d = model_2d.predict(xgb.DMatrix(X_all_2d, feature_names=fn_2d))
    corr_2d = pred_2d.copy()
    for step in range(T):
        corr_2d[step] -= preds_2d[step * n_2d:(step + 1) * n_2d]

    return corr_1d, corr_2d


def main():
    from residual_correction_lgbm_v2 import compute_rain_features

    config = load_model_config(data_dir, MODEL_ID)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
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
    n_1d, n_2d = config.num_1d_nodes, config.num_2d_nodes

    print("Building neighbor maps...")
    (nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
     map_1d_to_2d, map_2d_to_1d) = build_neighbor_maps(config)

    fn_1d = get_feat_names_1d_v10(sd_1d)
    fn_2d = get_feat_names_2d_v10(sd_2d)
    print(f"Features: 1D={len(fn_1d)}, 2D={len(fn_2d)}")

    train_ids_arr = list_events(data_dir, MODEL_ID, "train")
    train_ids = train_ids_arr.tolist() if hasattr(train_ids_arr, "tolist") else list(train_ids_arr)

    # === Step 1: per_event_predから特徴量生成 ===
    print("\n--- Step 1: Feature generation from per_event_pred ---")
    with open(os.path.join(cache_dir, "per_event_pred_v8.pkl"), "rb") as f:
        per_event_pred = pickle.load(f)
    print(f"  per_event_pred: {len(per_event_pred)} events")

    X_1d_list, y_1d_list = [], []
    X_2d_list, y_2d_list = [], []
    ev_ranges_1d, ev_ranges_2d = [], []
    c1d, c2d_c = 0, 0
    t0 = time.time()
    for i, eid in enumerate(train_ids):
        ed = per_event_pred[eid]
        T = ed["pred_1d"].shape[0]
        res_1d = ed["pred_1d"] - ed["gt_1d"]  # pred - gt (v8と同じ符号)
        res_2d = ed["pred_2d"] - ed["gt_2d"]
        r = {"T": T, "res_1d": res_1d, "res_2d": res_2d,
             "pred_1d": ed["pred_1d"], "pred_2d": ed["pred_2d"]}

        event = load_event_data(data_dir, MODEL_ID, eid, config)
        rf = compute_rain_features(event, T)
        wl_init_1d = event.nodes_1d_dynamic[SPIN_UP, :, 0]
        wl_init_2d = event.nodes_2d_dynamic[SPIN_UP, :, 1]

        X1 = build_features_1d_v10(r, config, node_feats, rf, STEP_STRIDE_1D,
                                    nb1_1d, nb2_1d, nb3_1d, map_1d_to_2d,
                                    per_node_stats=per_node_stats, wl_init=wl_init_1d)
        steps_1d = list(range(0, T, STEP_STRIDE_1D))
        y1 = np.array([res_1d[s, ni] for s in steps_1d for ni in range(n_1d)], dtype=np.float32)
        X_1d_list.append(X1); y_1d_list.append(y1)
        ev_ranges_1d.append((c1d, c1d + len(y1))); c1d += len(y1)

        event2 = load_event_data(data_dir, MODEL_ID, eid, config)
        X2 = build_features_2d_v10(r, config, rf, STEP_STRIDE_2D,
                                    nb1_2d, nb2_2d, nb3_2d, map_2d_to_1d,
                                    per_node_stats=per_node_stats, wl_init=wl_init_2d)
        steps_2d = list(range(0, T, STEP_STRIDE_2D))
        y2 = np.array([res_2d[s, ni] for s in steps_2d for ni in range(n_2d)], dtype=np.float32)
        X_2d_list.append(X2); y_2d_list.append(y2)
        ev_ranges_2d.append((c2d_c, c2d_c + len(y2))); c2d_c += len(y2)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(train_ids)} ({time.time()-t0:.1f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")

    X_1d = np.concatenate(X_1d_list); y_1d = np.concatenate(y_1d_list)
    X_2d = np.concatenate(X_2d_list); y_2d = np.concatenate(y_2d_list)
    print(f"  X_1d={X_1d.shape}, X_2d={X_2d.shape}")

    # 列数assert
    assert X_1d.shape[1] == len(fn_1d), f"1D mismatch: {X_1d.shape[1]} vs {len(fn_1d)}"
    assert X_2d.shape[1] == len(fn_2d), f"2D mismatch: {X_2d.shape[1]} vs {len(fn_2d)}"
    print(f"  Assert OK: 1D={X_1d.shape[1]}=={len(fn_1d)}, 2D={X_2d.shape[1]}=={len(fn_2d)}")

    # === Step 4: Final model ===
    params_1d = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.01, "max_leaves": 127, "max_depth": 10,
        "min_child_weight": 50, "colsample_bytree": 0.8, "subsample": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "tree_method": "gpu_hist", "device": "cuda", "verbosity": 0,
    }
    params_2d = params_1d.copy()
    params_2d["max_leaves"] = 63

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

    # Feature importance
    imp1 = fm1d.get_score(importance_type="gain")
    top10_1d = sorted(imp1.items(), key=lambda x: -x[1])[:10]
    print(f"  1D top10: {', '.join(f'{n}={v:.0f}' for n,v in top10_1d)}")
    imp2 = fm2d.get_score(importance_type="gain")
    top10_2d = sorted(imp2.items(), key=lambda x: -x[1])[:10]
    print(f"  2D top10: {', '.join(f'{n}={v:.0f}' for n,v in top10_2d)}")

    # OOF sanity check (in-sample, 10 events)
    print(f"\n--- OOF sanity check (in-sample, 10 events) ---")
    raw_s2, corr_s2 = [], []
    for eid in train_ids[:10]:
        ed = per_event_pred[eid]
        ec = load_event_data(data_dir, MODEL_ID, eid, config)
        c1, c2 = apply_correction_v10(ed["pred_1d"], ed["pred_2d"], ec, config, node_feats,
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
        c1, c2 = apply_correction_v10(p1d, p2d, ec, config, node_feats, fm1d, fm2d, fn_1d, fn_2d,
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
    out = os.path.join(BASE, "Dataset_Rerelease", "submission_v76_lgbm_v10.parquet")
    final.to_parquet(out, index=False)
    print(f"  {len(final)} rows -> {out}")
    print("DONE!")


if __name__ == "__main__":
    main()
