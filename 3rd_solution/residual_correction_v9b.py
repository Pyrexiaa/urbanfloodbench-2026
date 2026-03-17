"""Residual Correction v9b: v8コピー + バケット別2D + 新特徴量 + 180K rounds.

v8からの変更:
- rain_delta削除 (gain最下位)
- 追加: rain_x_pred_delta, pred_ratio_nb1, pred_minus_nb2, pred_over_nodestd
- バケット別2Dモデル (low/mid/high std)
- 180000 rounds, early_stopping=1000
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
# torch is imported lazily in Step 5 only

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import load_model_config, load_event_data, list_events
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events
from src.model_lstm1d import build_adjacency
from run_train_m2_v11c_multistep import (
    MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM,
)
from residual_correction_lgbm_v2 import (
    rollout_event_residuals, compute_rain_features,
)
from run_inference_ensemble import predict_event

device = "cuda"
data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
cache_dir = os.path.join(BASE, "Models", "checkpoints")
SPIN_UP = 10
STEP_STRIDE_1D = 2   # v5: 3→2
STEP_STRIDE_2D = 5    # v5: 10→5
N_ZONES = 10
NUM_BOOST_ROUND = 180000
EARLY_STOPPING = 1000

# バケット定義 (EDA H3: 特徴量の効き方がバケットで異なる)
STD_BOUNDARIES = [0.156, 0.377]
BUCKET_NAMES = ["low_std", "mid_std", "high_std"]


def build_neighbor_maps(config):
    """1-hop, 2-hop, 3-hop隣接 + coupling maps."""
    ei_1d = config.edge_index_1d
    n_1d = config.num_1d_nodes
    nb1_1d = [set() for _ in range(n_1d)]
    for j in range(ei_1d.shape[1]):
        src, tgt = ei_1d[0, j], ei_1d[1, j]
        nb1_1d[src].add(tgt)
        nb1_1d[tgt].add(src)

    nb2_1d = [set() for _ in range(n_1d)]
    for ni in range(n_1d):
        for nb in nb1_1d[ni]:
            for nb2 in nb1_1d[nb]:
                if nb2 != ni and nb2 not in nb1_1d[ni]:
                    nb2_1d[ni].add(nb2)

    nb3_1d = [set() for _ in range(n_1d)]
    for ni in range(n_1d):
        for nb2 in nb2_1d[ni]:
            for nb3 in nb1_1d[nb2]:
                if nb3 != ni and nb3 not in nb1_1d[ni] and nb3 not in nb2_1d[ni]:
                    nb3_1d[ni].add(nb3)

    ei_2d = config.edge_index_2d
    n_2d = config.num_2d_nodes
    nb1_2d = [set() for _ in range(n_2d)]
    for j in range(ei_2d.shape[1]):
        src, tgt = ei_2d[0, j], ei_2d[1, j]
        nb1_2d[src].add(tgt)
        nb1_2d[tgt].add(src)

    nb2_2d = [set() for _ in range(n_2d)]
    for ni in range(n_2d):
        for nb in nb1_2d[ni]:
            for nb2 in nb1_2d[nb]:
                if nb2 != ni and nb2 not in nb1_2d[ni]:
                    nb2_2d[ni].add(nb2)

    nb3_2d = [set() for _ in range(n_2d)]
    for ni in range(n_2d):
        for nb2 in nb2_2d[ni]:
            for nb3 in nb1_2d[nb2]:
                if nb3 != ni and nb3 not in nb1_2d[ni] and nb3 not in nb2_2d[ni]:
                    nb3_2d[ni].add(nb3)

    nb1_1d = [list(s) for s in nb1_1d]
    nb2_1d = [list(s) for s in nb2_1d]
    nb3_1d = [list(s) for s in nb3_1d]
    nb1_2d = [list(s) for s in nb1_2d]
    nb2_2d = [list(s) for s in nb2_2d]
    nb3_2d = [list(s) for s in nb3_2d]

    conn = config.connections_1d2d
    map_1d_to_2d = {}
    map_2d_to_1d = {}
    for row in conn:
        n1d, n2d = int(row[0]), int(row[1])
        map_1d_to_2d.setdefault(n1d, []).append(n2d)
        map_2d_to_1d.setdefault(n2d, []).append(n1d)

    return (nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
            map_1d_to_2d, map_2d_to_1d)


def compute_nb_feats_1d(pred_1d_step, pred_2d_step,
                         nb1, nb2, nb3, map_1d_to_2d):
    """[n_1d, 7]: nb1_mean, nb1_std, coupled_2d, nb2_mean, nb2_std, nb3_mean, nb3_std."""
    n = len(pred_1d_step)
    feats = np.zeros((n, 7), dtype=np.float32)
    for ni in range(n):
        if nb1[ni]:
            p = pred_1d_step[nb1[ni]]
            feats[ni, 0] = p.mean()
            feats[ni, 1] = p.std() if len(nb1[ni]) > 1 else 0.0
        c2d = map_1d_to_2d.get(ni)
        if c2d:
            feats[ni, 2] = pred_2d_step[c2d].mean()
        if nb2[ni]:
            p = pred_1d_step[nb2[ni]]
            feats[ni, 3] = p.mean()
            feats[ni, 4] = p.std() if len(nb2[ni]) > 1 else 0.0
        if nb3[ni]:
            p = pred_1d_step[nb3[ni]]
            feats[ni, 5] = p.mean()
            feats[ni, 6] = p.std() if len(nb3[ni]) > 1 else 0.0
    return feats


def compute_nb_feats_2d(pred_2d_step, pred_1d_step,
                         nb1, nb2, nb3, map_2d_to_1d):
    """[n_2d, 7]: nb1_mean, nb1_std, coupled_1d, nb2_mean, nb2_std, nb3_mean, nb3_std."""
    n = len(pred_2d_step)
    feats = np.zeros((n, 7), dtype=np.float32)
    for ni in range(n):
        if nb1[ni]:
            p = pred_2d_step[nb1[ni]]
            feats[ni, 0] = p.mean()
            feats[ni, 1] = p.std() if len(nb1[ni]) > 1 else 0.0
        c1d = map_2d_to_1d.get(ni)
        if c1d:
            feats[ni, 2] = pred_1d_step[c1d].mean()
        if nb2[ni]:
            p = pred_2d_step[nb2[ni]]
            feats[ni, 3] = p.mean()
            feats[ni, 4] = p.std() if len(nb2[ni]) > 1 else 0.0
        if nb3[ni]:
            p = pred_2d_step[nb3[ni]]
            feats[ni, 5] = p.mean()
            feats[ni, 6] = p.std() if len(nb3[ni]) > 1 else 0.0
    return feats


def get_feat_names_1d(static_dim):
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
                   # v8 features (rain_delta削除) + v9新特徴量
                   "laplacian", "steps_since_peak", "node_std",
                   "rain_x_pred_delta", "pred_ratio_nb1", "pred_minus_nb2", "pred_over_nodestd"])
    return names


def get_feat_names_2d(static_dim):
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
                   # v8 features (rain_delta削除) + v9新特徴量
                   "laplacian", "steps_since_peak", "node_std",
                   "rain_x_pred_delta", "pred_ratio_nb1", "pred_minus_nb2", "pred_over_nodestd"])
    return names


def build_features_1d(event, rollout, config, node_feats, rain_feats, step_stride,
                       nb1, nb2, nb3, map_1d_to_2d, per_node_stats=None):
    n_1d = config.num_1d_nodes
    T = rollout["T"]
    sd = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
    wl_init = event.nodes_1d_dynamic[SPIN_UP, :, 0]
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]
    peak_pos = rain_feats["peak_position"]
    # node_std for 1D
    nstd_1d = per_node_stats["1d_wl_std"] if per_node_stats is not None else np.ones(n_1d)

    steps = list(range(0, T, step_stride))
    n_feat = 2 + sd + 15 + 7 + 7  # rain_delta削除(-1), 新4特徴量追加(+4) = net +3 vs v8
    X = np.zeros((len(steps) * n_1d, n_feat), dtype=np.float32)

    idx = 0
    for step in steps:
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        # steps_since_peak
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_1d(rollout["pred_1d"][step], rollout["pred_2d"][step],
                                   nb1, nb2, nb3, map_1d_to_2d)
        for ni in range(n_1d):
            pv = rollout["pred_1d"][step, ni]
            # laplacian = pred - nb1_mean
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
            # v9: rain_delta削除, laplacian + ssp + node_std + 4新特徴量
            nb1_mean = nbf[ni, 0]
            nb2_mean = nbf[ni, 3]
            row.extend([
                lap, ssp, nstd_1d[ni],
                rn * pd,  # rain_x_pred_delta
                pv / (nb1_mean + 1e-8) if nb1_mean != 0 else 1.0,  # pred_ratio_nb1
                pv - nb2_mean,  # pred_minus_nb2
                pv / (nstd_1d[ni] + 1e-8),  # pred_over_nodestd
            ])
            X[idx] = row
            idx += 1
    return X[:idx]


def build_features_2d(event, rollout, config, rain_feats, step_stride,
                       nb1, nb2, nb3, map_2d_to_1d, per_node_stats=None):
    n_2d = config.num_2d_nodes
    T = rollout["T"]
    sd = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0
    wl_init = event.nodes_2d_dynamic[SPIN_UP, :, 1]
    rain = rain_feats["rain_full"]
    cum_rain = rain_feats["cum_rain"]
    coupled_2d = set(config.connections_1d2d[:, 1])
    peak_pos = rain_feats["peak_position"]
    # node_std for 2D
    nstd_2d = per_node_stats["2d_wl_std"] if per_node_stats is not None else np.ones(n_2d)

    steps = list(range(0, T, step_stride))
    n_feat = 1 + sd + 13 + 7 + 7  # rain_delta削除(-1), 新4特徴量追加(+4) = net +3 vs v8
    X = np.zeros((len(steps) * n_2d, n_feat), dtype=np.float32)

    idx = 0
    for step in steps:
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        # steps_since_peak
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_2d(rollout["pred_2d"][step], rollout["pred_1d"][step],
                                   nb1, nb2, nb3, map_2d_to_1d)
        for ni in range(n_2d):
            pv = rollout["pred_2d"][step, ni]
            # laplacian = pred - nb1_mean
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
            # v9: rain_delta削除, laplacian + ssp + node_std + 4新特徴量
            nb1_mean = nbf[ni, 0]
            nb2_mean = nbf[ni, 3]
            row.extend([
                lap, ssp, nstd_2d[ni],
                rn * pd,  # rain_x_pred_delta
                pv / (nb1_mean + 1e-8) if nb1_mean != 0 else 1.0,  # pred_ratio_nb1
                pv - nb2_mean,  # pred_minus_nb2
                pv / (nstd_2d[ni] + 1e-8),  # pred_over_nodestd
            ])
            X[idx] = row
            idx += 1
    return X[:idx]


def apply_correction(pred_1d, pred_2d, event, config, node_feats,
                      model_1d, model_2d, fn_1d, fn_2d,
                      sd_1d, sd_2d, coupled_2d_set,
                      nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
                      map_1d_to_2d, map_2d_to_1d, per_node_stats=None):
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
    # v8: node_std
    nstd_1d = per_node_stats["1d_wl_std"] if per_node_stats is not None else np.ones(n_1d)
    nstd_2d = per_node_stats["2d_wl_std"] if per_node_stats is not None else np.ones(n_2d)

    # バッチ化: 全ステップ分を一括でDMatrix作成→predict1回
    # 1D: (T * n_1d, n_feat)
    X_all_1d = np.zeros((T * n_1d, len(fn_1d)), dtype=np.float32)
    for step in range(T):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        # v8: rain_delta, steps_since_peak
        rn_prev = rain[SPIN_UP + step - 1] if SPIN_UP + step - 1 >= 0 and step > 0 else rn
        rain_delta = rn - rn_prev
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_1d(pred_1d[step], pred_2d[step],
                                   nb1_1d, nb2_1d, nb3_1d, map_1d_to_2d)
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
        # v9: laplacian, ssp, node_std + 4新特徴量 (rain_delta削除)
        lap_1d = pred_1d[step] - nbf[:, 0]
        pd_1d = pred_1d[step] - wl_init_1d
        nb1m = nbf[:, 0]
        nb2m = nbf[:, 3]
        X_all_1d[s:s+n_1d, b+22] = lap_1d
        X_all_1d[s:s+n_1d, b+23] = ssp
        X_all_1d[s:s+n_1d, b+24] = nstd_1d
        X_all_1d[s:s+n_1d, b+25] = rn * pd_1d  # rain_x_pred_delta
        safe_nb1 = np.where(nb1m != 0, nb1m, 1e-8)
        X_all_1d[s:s+n_1d, b+26] = np.where(nb1m != 0, pred_1d[step] / safe_nb1, 1.0)  # pred_ratio_nb1
        X_all_1d[s:s+n_1d, b+27] = pred_1d[step] - nb2m  # pred_minus_nb2
        X_all_1d[s:s+n_1d, b+28] = pred_1d[step] / (nstd_1d + 1e-8)  # pred_over_nodestd

    preds_1d = model_1d.predict(xgb.DMatrix(X_all_1d, feature_names=fn_1d))
    corr_1d = pred_1d.copy()
    for step in range(T):
        corr_1d[step] -= preds_1d[step * n_1d:(step + 1) * n_1d]

    # 2D: (T * n_2d, n_feat)
    X_all_2d = np.zeros((T * n_2d, len(fn_2d)), dtype=np.float32)
    for step in range(T):
        rn = rain[SPIN_UP + step] if SPIN_UP + step < len(rain) else 0
        rl10 = rain[max(0, SPIN_UP+step-10):SPIN_UP+step].mean() if step > 0 else rn
        rl30 = rain[max(0, SPIN_UP+step-30):SPIN_UP+step].mean() if step > 0 else rn
        cr = cum_rain[step] if step < len(cum_rain) else cum_rain[-1]
        # v8: rain_delta, steps_since_peak
        rn_prev = rain[SPIN_UP + step - 1] if SPIN_UP + step - 1 >= 0 and step > 0 else rn
        rain_delta = rn - rn_prev
        ssp = max(0, step - peak_pos)
        nbf = compute_nb_feats_2d(pred_2d[step], pred_1d[step],
                                   nb1_2d, nb2_2d, nb3_2d, map_2d_to_1d)
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
        # v9: laplacian, ssp, node_std + 4新特徴量 (rain_delta削除)
        lap_2d = pred_2d[step] - nbf[:, 0]
        pd_2d = pred_2d[step] - wl_init_2d
        nb1m = nbf[:, 0]
        nb2m = nbf[:, 3]
        X_all_2d[s:s+n_2d, b+20] = lap_2d
        X_all_2d[s:s+n_2d, b+21] = ssp
        X_all_2d[s:s+n_2d, b+22] = nstd_2d
        X_all_2d[s:s+n_2d, b+23] = rn * pd_2d  # rain_x_pred_delta
        safe_nb1 = np.where(nb1m != 0, nb1m, 1e-8)
        X_all_2d[s:s+n_2d, b+24] = np.where(nb1m != 0, pred_2d[step] / safe_nb1, 1.0)  # pred_ratio_nb1
        X_all_2d[s:s+n_2d, b+25] = pred_2d[step] - nb2m  # pred_minus_nb2
        X_all_2d[s:s+n_2d, b+26] = pred_2d[step] / (nstd_2d + 1e-8)  # pred_over_nodestd

    preds_2d = model_2d.predict(xgb.DMatrix(X_all_2d, feature_names=fn_2d))
    corr_2d = pred_2d.copy()
    for step in range(T):
        corr_2d[step] -= preds_2d[step * n_2d:(step + 1) * n_2d]

    return corr_1d, corr_2d


def main():
    print("=" * 70)
    print("  v9b: v8 base + bucket 2D + new feats + 180K rounds")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    ckpt_path = os.path.join(cache_dir, "best_model_2_v76_aligned_r400.pt")
    # GNN modelはStep 5でだけロード (VRAM節約)
    model = None
    print(f"  Model: v76 (deferred load for Step 5)")

    _, _, degree_1d = build_adjacency(config.edge_index_1d, config.num_1d_nodes)
    conn = config.connections_1d2d
    coupled_set = set(conn[:, 0])
    is_coupled = np.array([1 if i in coupled_set else 0 for i in range(config.num_1d_nodes)])
    node_feats = {"degree": degree_1d, "is_coupled": is_coupled}

    n_1d = config.num_1d_nodes
    n_2d = config.num_2d_nodes
    sd_1d = config.nodes_1d_static.shape[1] if config.nodes_1d_static is not None else 0
    sd_2d = config.nodes_2d_static.shape[1] if config.nodes_2d_static is not None else 0
    coupled_2d_set = set(conn[:, 1])

    print("  Building neighbor maps (1/2/3-hop)...")
    (nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
     map_1d_to_2d, map_2d_to_1d) = build_neighbor_maps(config)
    print(f"  1D: 1h={np.mean([len(n) for n in nb1_1d]):.1f}, "
          f"2h={np.mean([len(n) for n in nb2_1d]):.1f}, "
          f"3h={np.mean([len(n) for n in nb3_1d]):.1f}")
    print(f"  2D: 1h={np.mean([len(n) for n in nb1_2d]):.1f}, "
          f"2h={np.mean([len(n) for n in nb2_2d]):.1f}, "
          f"3h={np.mean([len(n) for n in nb3_2d]):.1f}")

    fn_1d = get_feat_names_1d(sd_1d)
    fn_2d = get_feat_names_2d(sd_2d)
    print(f"  Features: 1D={len(fn_1d)}, 2D={len(fn_2d)}")

    train_ids_arr = list_events(data_dir, MODEL_ID, "train")
    train_ids = train_ids_arr.tolist() if hasattr(train_ids_arr, 'tolist') else list(train_ids_arr)

    # ===== Step 1: Train rollout + features (with cache) =====
    # v8 cache has different features than v6, but rollout/per_event_pred are reusable
    cache_file = os.path.join(cache_dir, "lgbm_v9b_step1_cache.pkl")
    v8_cache_file = os.path.join(cache_dir, "lgbm_v8_step1_cache.pkl")
    v6_cache_file = os.path.join(cache_dir, "lgbm_v6_step1_cache.pkl")
    if os.path.exists(cache_file):
        print("\n--- Step 1: Loading v8 cache ---")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        X_1d = cache["X_1d"]; y_1d = cache["y_1d"]
        X_2d = cache["X_2d"]; y_2d = cache["y_2d"]
        ev_ranges_1d = cache["ev_ranges_1d"]; ev_ranges_2d = cache["ev_ranges_2d"]
        per_event_pred = cache["per_event_pred"]
        print(f"  1D: X={X_1d.shape}, 2D: X={X_2d.shape}")
    elif True:
        # per_event_predは分離キャッシュから読む (メモリ節約: 0.6GB vs 3.1GB)
        pep_file = os.path.join(cache_dir, "per_event_pred_v8.pkl")
        if os.path.exists(pep_file):
            print(f"\n--- Step 1: Loading per_event_pred (split), regenerating v9b features ---")
            with open(pep_file, "rb") as f:
                per_event_pred = pickle.load(f)
        elif os.path.exists(v8_cache_file):
            print(f"\n--- Step 1: Reusing v8 cache, regenerating v9b features ---")
            with open(v8_cache_file, "rb") as f:
                v6_cache = pickle.load(f)
            per_event_pred = v6_cache["per_event_pred"]
            del v6_cache
        elif os.path.exists(v6_cache_file):
            print(f"\n--- Step 1: Reusing v6 cache, regenerating v9b features ---")
            with open(v6_cache_file, "rb") as f:
                v6_cache = pickle.load(f)
            per_event_pred = v6_cache["per_event_pred"]
            del v6_cache
        else:
            per_event_pred = {}
        # per_event_predからres計算 + 特徴量構築 (GNN rollout不要)
        X_1d_list, y_1d_list = [], []
        X_2d_list, y_2d_list = [], []
        ev_ranges_1d, ev_ranges_2d = [], []
        c1d, c2d_c = 0, 0

        t0 = time.time()
        for i, eid in enumerate(train_ids):
            ed = per_event_pred[eid]
            T = ed["pred_1d"].shape[0]
            res_1d = ed["pred_1d"] - ed["gt_1d"]  # residual = pred - gt (v8と同じ符号)
            res_2d = ed["pred_2d"] - ed["gt_2d"]
            r = {"T": T, "res_1d": res_1d, "res_2d": res_2d,
                 "pred_1d": ed["pred_1d"], "pred_2d": ed["pred_2d"]}

            event = load_event_data(data_dir, MODEL_ID, eid, config)
            rf = compute_rain_features(event, T)

            event2 = load_event_data(data_dir, MODEL_ID, eid, config)
            X1 = build_features_1d(event2, r, config, node_feats, rf, STEP_STRIDE_1D,
                                    nb1_1d, nb2_1d, nb3_1d, map_1d_to_2d,
                                    per_node_stats=per_node_stats)
            steps_1d = list(range(0, T, STEP_STRIDE_1D))
            y1 = np.array([res_1d[s, ni] for s in steps_1d for ni in range(n_1d)], dtype=np.float32)
            X_1d_list.append(X1); y_1d_list.append(y1)
            ev_ranges_1d.append((c1d, c1d + len(y1))); c1d += len(y1)

            event3 = load_event_data(data_dir, MODEL_ID, eid, config)
            X2 = build_features_2d(event3, r, config, rf, STEP_STRIDE_2D,
                                    nb1_2d, nb2_2d, nb3_2d, map_2d_to_1d,
                                    per_node_stats=per_node_stats)
            steps_2d = list(range(0, T, STEP_STRIDE_2D))
            y2 = np.array([res_2d[s, ni] for s in steps_2d for ni in range(n_2d)], dtype=np.float32)
            X_2d_list.append(X2); y_2d_list.append(y2)
            ev_ranges_2d.append((c2d_c, c2d_c + len(y2))); c2d_c += len(y2)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(train_ids)} ({time.time()-t0:.1f}s)")
        print(f"  Features done in {time.time()-t0:.1f}s (no GNN rollout)")

        X_1d = np.concatenate(X_1d_list); y_1d = np.concatenate(y_1d_list)
        X_2d = np.concatenate(X_2d_list); y_2d = np.concatenate(y_2d_list)
        print(f"  1D: X={X_1d.shape}, 2D: X={X_2d.shape}")

        print("  Saving v8 Step 1 cache...")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X_1d": X_1d, "y_1d": y_1d, "X_2d": X_2d, "y_2d": y_2d,
                "ev_ranges_1d": ev_ranges_1d, "ev_ranges_2d": ev_ranges_2d,
                "per_event_pred": per_event_pred,
            }, f)
        print(f"  Cache saved: {cache_file}")
    else:
        print("\n--- Step 1: Train rollout + features (full) ---")
        X_1d_list, y_1d_list = [], []
        X_2d_list, y_2d_list = [], []
        ev_ranges_1d, ev_ranges_2d = [], []
        c1d, c2d_c = 0, 0
        per_event_pred = {}

        t0 = time.time()
        for i, eid in enumerate(train_ids):
            event = load_event_data(data_dir, MODEL_ID, eid, config)
            r = rollout_event_residuals(model, event, config, norm_stats, per_node_stats)
            rf = compute_rain_features(event, r["T"])
            T = r["T"]

            event2 = load_event_data(data_dir, MODEL_ID, eid, config)
            X1 = build_features_1d(event2, r, config, node_feats, rf, STEP_STRIDE_1D,
                                    nb1_1d, nb2_1d, nb3_1d, map_1d_to_2d,
                                    per_node_stats=per_node_stats)
            steps_1d = list(range(0, T, STEP_STRIDE_1D))
            y1 = np.array([r["res_1d"][s, ni] for s in steps_1d for ni in range(n_1d)], dtype=np.float32)
            X_1d_list.append(X1); y_1d_list.append(y1)
            ev_ranges_1d.append((c1d, c1d + len(y1))); c1d += len(y1)

            event3 = load_event_data(data_dir, MODEL_ID, eid, config)
            X2 = build_features_2d(event3, r, config, rf, STEP_STRIDE_2D,
                                    nb1_2d, nb2_2d, nb3_2d, map_2d_to_1d,
                                    per_node_stats=per_node_stats)
            steps_2d = list(range(0, T, STEP_STRIDE_2D))
            y2 = np.array([r["res_2d"][s, ni] for s in steps_2d for ni in range(n_2d)], dtype=np.float32)
            X_2d_list.append(X2); y_2d_list.append(y2)
            ev_ranges_2d.append((c2d_c, c2d_c + len(y2))); c2d_c += len(y2)

            event4 = load_event_data(data_dir, MODEL_ID, eid, config)
            preds = predict_event(model, config, event4, norm_stats, device,
                                  spin_up=SPIN_UP, future_rain_steps=FUTURE_RAIN_STEPS,
                                  coupling_features=True, is_v11=True, per_node_stats=per_node_stats)
            p1d = np.array(preds["pred_1d"]); p2d = np.array(preds["pred_2d"])
            event5 = load_event_data(data_dir, MODEL_ID, eid, config)
            Tp = p1d.shape[0]
            per_event_pred[eid] = {
                "pred_1d": p1d, "pred_2d": p2d,
                "gt_1d": event5.nodes_1d_dynamic[SPIN_UP:SPIN_UP+Tp, :, 0],
                "gt_2d": event5.nodes_2d_dynamic[SPIN_UP:SPIN_UP+Tp, :, 1],
            }
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(train_ids)} ({time.time()-t0:.1f}s)")
        print(f"  Done in {time.time()-t0:.1f}s")

        X_1d = np.concatenate(X_1d_list); y_1d = np.concatenate(y_1d_list)
        X_2d = np.concatenate(X_2d_list); y_2d = np.concatenate(y_2d_list)
        print(f"  1D: X={X_1d.shape}, 2D: X={X_2d.shape}")

        print("  Saving Step 1 cache...")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X_1d": X_1d, "y_1d": y_1d, "X_2d": X_2d, "y_2d": y_2d,
                "ev_ranges_1d": ev_ranges_1d, "ev_ranges_2d": ev_ranges_2d,
                "per_event_pred": per_event_pred,
            }, f)
        print(f"  Cache saved: {cache_file}")

    # 列数整合チェック (v8キャッシュ混入防止)
    assert X_1d.shape[1] == len(fn_1d), f"1D列数不一致: X={X_1d.shape[1]} vs fn={len(fn_1d)}"
    assert X_2d.shape[1] == len(fn_2d), f"2D列数不一致: X={X_2d.shape[1]} vs fn={len(fn_2d)}"

    # ===== Step 2: 5-Fold CV (XGBoost GPU) =====
    print(f"\n--- Step 2: 5-Fold CV (XGBoost GPU, lr=0.01, rounds={NUM_BOOST_ROUND}) ---")
    import xgboost as xgb
    from sklearn.model_selection import KFold

    params_1d = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_leaves": 127, "max_depth": 10,
        "min_child_weight": 50, "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "tree_method": "gpu_hist", "device": "cuda",
        "verbosity": 0,
    }
    params_2d = params_1d.copy()
    params_2d["max_leaves"] = 63

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_corrected = {}
    fold_models = {}
    cv_cache_path = os.path.join(cache_dir, "lgbm_v9b_cv_cache.pkl")

    # fold checkpointがあれば復元
    start_fold = 0
    if os.path.exists(cv_cache_path):
        with open(cv_cache_path, "rb") as f:
            cv_ckpt = pickle.load(f)
        start_fold = cv_ckpt["next_fold"]
        oof_corrected = cv_ckpt["oof_corrected"]
        fold_models = cv_ckpt["fold_models"]
        print(f"  ** Resuming from fold {start_fold} (cache: {cv_cache_path}) **")

    for fold, (tr_ev, va_ev) in enumerate(kf.split(train_ids)):
        if fold < start_fold:
            continue

        t_fold = time.time()
        tr_1d = []; va_1d = []
        for ei in tr_ev:
            s, e = ev_ranges_1d[ei]; tr_1d.extend(range(s, e))
        for ei in va_ev:
            s, e = ev_ranges_1d[ei]; va_1d.extend(range(s, e))

        dt = xgb.DMatrix(X_1d[tr_1d], label=y_1d[tr_1d], feature_names=fn_1d)
        dv = xgb.DMatrix(X_1d[va_1d], label=y_1d[va_1d], feature_names=fn_1d)
        m1d = xgb.train(params_1d, dt, num_boost_round=NUM_BOOST_ROUND,
                        evals=[(dv, "val")], early_stopping_rounds=EARLY_STOPPING,
                        verbose_eval=False)
        print(f"  Fold {fold}: 1D rounds={m1d.best_iteration}")

        tr_2d = []; va_2d = []
        for ei in tr_ev:
            s, e = ev_ranges_2d[ei]; tr_2d.extend(range(s, e))
        for ei in va_ev:
            s, e = ev_ranges_2d[ei]; va_2d.extend(range(s, e))

        dt2 = xgb.DMatrix(X_2d[tr_2d], label=y_2d[tr_2d], feature_names=fn_2d)
        dv2 = xgb.DMatrix(X_2d[va_2d], label=y_2d[va_2d], feature_names=fn_2d)
        m2d = xgb.train(params_2d, dt2, num_boost_round=NUM_BOOST_ROUND,
                        evals=[(dv2, "val")], early_stopping_rounds=EARLY_STOPPING,
                        verbose_eval=False)
        print(f"          2D rounds={m2d.best_iteration} ({time.time()-t_fold:.0f}s)")

        # OOF correction
        va_eids = [train_ids[ei] for ei in va_ev]
        for eid in va_eids:
            ed = per_event_pred[eid]
            ec = load_event_data(data_dir, MODEL_ID, eid, config)
            c1, c2 = apply_correction(
                ed["pred_1d"], ed["pred_2d"], ec, config, node_feats,
                m1d, m2d, fn_1d, fn_2d, sd_1d, sd_2d, coupled_2d_set,
                nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
                map_1d_to_2d, map_2d_to_1d, per_node_stats=per_node_stats)
            oof_corrected[eid] = {"pred_1d": c1, "pred_2d": c2}

        fold_models[fold] = {"m1d_rounds": m1d.best_iteration, "m2d_rounds": m2d.best_iteration}
        with open(cv_cache_path, "wb") as f:
            pickle.dump({"next_fold": fold + 1, "oof_corrected": oof_corrected,
                         "fold_models": fold_models}, f)
        print(f"  ** Fold {fold} checkpoint saved **")

    # ===== Step 3: OOF SRMSE =====
    oof_eids = [eid for eid in train_ids if eid in oof_corrected]
    print(f"\n--- Step 3: OOF SRMSE ({len(oof_eids)}/{len(train_ids)} events) ---")
    raw_s1, raw_s2, corr_s1, corr_s2 = [], [], [], []
    for eid in oof_eids:
        ed = per_event_pred[eid]; oc = oof_corrected[eid]
        rr1 = np.sqrt(np.mean((ed["pred_1d"] - ed["gt_1d"])**2, axis=0))
        rc1 = np.sqrt(np.mean((oc["pred_1d"] - ed["gt_1d"])**2, axis=0))
        rr2 = np.sqrt(np.mean((ed["pred_2d"] - ed["gt_2d"])**2, axis=0))
        rc2 = np.sqrt(np.mean((oc["pred_2d"] - ed["gt_2d"])**2, axis=0))
        raw_s1.append(np.mean(rr1 / std_1d)); corr_s1.append(np.mean(rc1 / std_1d))
        raw_s2.append(np.mean(rr2 / std_2d)); corr_s2.append(np.mean(rc2 / std_2d))

    r1, c1 = np.mean(raw_s1), np.mean(corr_s1)
    r2, c2 = np.mean(raw_s2), np.mean(corr_s2)
    rt, ct = (r1+r2)/2, (c1+c2)/2
    print(f"  1D SRMSE: {r1:.4f} -> {c1:.4f} ({(1-c1/r1)*100:.1f}%)")
    print(f"  2D SRMSE: {r2:.4f} -> {c2:.4f} ({(1-c2/r2)*100:.1f}%)")
    print(f"  Total:    {rt:.4f} -> {ct:.4f} ({(1-ct/rt)*100:.1f}%)")

    # ===== Step 4: Final model =====
    print(f"\n--- Step 4: Final model (XGBoost GPU) ---")
    rounds_1d = int(np.median([v["m1d_rounds"] for v in fold_models.values()]))
    rounds_2d = int(np.median([v["m2d_rounds"] for v in fold_models.values()]))
    dt_full_1d = xgb.DMatrix(X_1d, label=y_1d, feature_names=fn_1d)
    fm1d = xgb.train(params_1d, dt_full_1d, num_boost_round=rounds_1d)
    dt_full_2d = xgb.DMatrix(X_2d, label=y_2d, feature_names=fn_2d)
    fm2d = xgb.train(params_2d, dt_full_2d, num_boost_round=rounds_2d)
    print(f"  1D: {rounds_1d} rounds, 2D: {rounds_2d} rounds")

    save_path = os.path.join(cache_dir, "lgbm_residual_v9b.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"final_model_1d": fm1d, "final_model_2d": fm2d,
                      "feat_names_1d": fn_1d, "feat_names_2d": fn_2d}, f)
    print(f"  Saved: {save_path}")

    print(f"\n  1D top features:")
    imp = fm1d.get_score(importance_type="gain")
    for name, val in sorted(imp.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:20s}: {val:>12.0f}")
    print(f"\n  2D top features:")
    imp2 = fm2d.get_score(importance_type="gain")
    for name, val in sorted(imp2.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:20s}: {val:>12.0f}")

    # ===== Step 5: Test submission =====
    # GNN modelをここでロード (Step 1-4ではGPU不要だった)
    import torch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=ckpt.get("coupling_edge_dim", COUPLING_EDGE_DIM),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n--- Step 5: Test submission ---")
    test_ids_arr = list_events(data_dir, MODEL_ID, "test")
    test_ids = test_ids_arr.tolist() if hasattr(test_ids_arr, 'tolist') else list(test_ids_arr)
    from run_inference_ensemble import build_submission

    all_preds = []
    t0 = time.time()
    for i, eid in enumerate(test_ids):
        event = load_event_data(data_dir, MODEL_ID, eid, config, split="test")
        preds = predict_event(model, config, event, norm_stats, device,
                              spin_up=SPIN_UP, future_rain_steps=FUTURE_RAIN_STEPS,
                              coupling_features=True, is_v11=True, per_node_stats=per_node_stats)
        p1d = np.array(preds["pred_1d"]); p2d = np.array(preds["pred_2d"])

        ec = load_event_data(data_dir, MODEL_ID, eid, config, split="test")
        c1, c2 = apply_correction(
            p1d, p2d, ec, config, node_feats, fm1d, fm2d, fn_1d, fn_2d,
            sd_1d, sd_2d, coupled_2d_set,
            nb1_1d, nb2_1d, nb3_1d, nb1_2d, nb2_2d, nb3_2d,
            map_1d_to_2d, map_2d_to_1d, per_node_stats=per_node_stats)
        all_preds.append((MODEL_ID, eid, {"pred_1d": c1, "pred_2d": c2}))

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

    # Blend with v5
    v5 = pd.read_parquet(os.path.join(BASE, "Dataset_Rerelease", "submission_v76_lgbm_v5_nw.parquet"))
    pred_cols = [c for c in final.columns if c.startswith("prediction_")]
    m2_mask = final["model_id"] == 2
    blend = final.copy()
    for col in pred_cols:
        blend.loc[m2_mask, col] = 0.5 * final.loc[m2_mask, col] + 0.5 * v5.loc[m2_mask, col]
    out_bl = os.path.join(BASE, "Dataset_Rerelease", "submission_v76_lgbm_v9b_blend_v5.parquet")
    blend.to_parquet(out_bl, index=False)
    print(f"  Blend v9b+v5: {len(blend)} rows -> {out_bl}")

    print(f"\n{'=' * 70}")
    print(f"  OOF: {ct:.4f}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
