"""UrbanFloodBench Ensemble推論.

複数seedのModel_2チェックポイントをアンサンブルし、
Model_1(v1) + Model_2(ensemble)でsubmission生成。

Usage: python run_inference_ensemble.py --m2_tags s42,s123,s777
       python run_inference_ensemble.py --m2_tags focused  # single model
"""
import sys
import os
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_model_config, load_event_data, build_graph_at_timestep,
    compute_normalization_stats, list_events,
)
from src.model import HeteroFloodGNN, HeteroFloodGNNv2, HeteroFloodGNNv3, HeteroFloodGNNv4, HeteroFloodGNNv5, HeteroFloodGNNv6, HeteroFloodGNNv10, HeteroFloodGNNv11, HeteroFloodGNNv11_TB, HeteroFloodGNNv11_GRU, HeteroFloodGNNv11_GRU2, HeteroFloodGNNv11_GAT, HeteroFloodGNNv15, HeteroFloodGNNv18, HeteroFloodGNNv20, HeteroFloodGNNv21, HeteroFloodGNNv45, HeteroFloodGNNv97
from src.model_v12 import DUALFloodGNNv12
from src.fno_model import FloodFNO, GridInterpolator


def predict_event(
    model: torch.nn.Module,
    config,
    event,
    norm_stats: dict,
    device: str,
    spin_up: int = 10,
    delta_stats: dict | None = None,
    future_rain_steps: int = 0,
    coupling_features: bool = False,
    is_v6: bool = False,
    is_v10: bool = False,
    is_v11: bool = False,
    is_v12: bool = False,
    is_v15: bool = False,
    use_prev_t2: bool = False,
    is_gru: bool = False,
    is_tb: bool = False,
    tb_K: int = 4,
    per_node_stats: dict | None = None,
    denorm_stats: dict | None = None,
    manning_features: bool = False,
    delta_bias_corr: dict | None = None,
    use_heun: bool = False,
    recession_clip: bool = False,
) -> dict:
    """1イベントの全ノード水位を自己回帰予測 (bf16).

    delta_stats: delta clipping + WL bounds (Model_2 error explosion対策).
      keys: delta_clip_1d, delta_clip_2d, wl_bounds_1d, wl_bounds_2d
    is_v6: True → GRU hidden state管理 + step_ratio + spin-up warmup
    is_v10: True → 2-dim output (wl_delta + aux_delta), auxiliary feedback
    is_v11: True → flux/state paradigm (wl delta + inlet abs + edge abs)
    is_v12: True → DUALFloodGNN (node delta + edge delta feedback)
    per_node_stats: graph入力正規化用 (wl_mean, wl_std)
    denorm_stats: 出力非正規化用. Noneならper_node_statsを使用. v13はdelta_stdを渡す。
    per_node_stats: v11b per-node normalization stats. Model outputs normalized
      delta → denormalize: delta_raw = delta_norm * per_node_std
    """
    model.eval()
    max_t = event.num_timesteps
    last_gt_t = spin_up - 1
    pred_steps = max_t - spin_up

    # 非正規化用std: denorm_stats優先、なければper_node_stats
    _ds = denorm_stats if denorm_stats is not None else per_node_stats

    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    e1d_dyn = event.edges_1d_dynamic.copy()
    e2d_dyn = event.edges_2d_dynamic.copy()
    current_wl_1d = dyn_1d[last_gt_t, :, 0].copy()
    current_wl_2d = dyn_2d[last_gt_t, :, 1].copy()

    # delta clipping / WL bounds (Noneなら無制限)
    if delta_stats is not None:
        dc_1d = delta_stats["delta_clip_1d"]  # (low, high)
        dc_2d = delta_stats["delta_clip_2d"]
        wb_1d = delta_stats["wl_bounds_1d"]   # (low, high)
        wb_2d = delta_stats["wl_bounds_2d"]
    else:
        dc_1d = dc_2d = wb_1d = wb_2d = None

    pred_1d_all = []
    pred_2d_all = []

    orig_1d = event.nodes_1d_dynamic
    orig_2d = event.nodes_2d_dynamic
    orig_e1d = event.edges_1d_dynamic
    orig_e2d = event.edges_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = e1d_dyn
    event.edges_2d_dynamic = e2d_dyn

    # テスト条件: t>=spin_upで利用不可な特徴量をゼロ化
    # v10: inlet_flow/water_volumeは予測→フィードバック
    # v11: inlet_flow + 1Dエッジは予測→上書き、water_volumeはゼロ化
    # v12: 1D/2Dエッジは予測→delta累積フィードバック
    for t in range(spin_up, max_t):
        if not is_v10 and not is_v11:
            dyn_1d[t, :, 1] = 0.0    # inlet_flow (v10/v11が予測、他はゼロ)
        if not is_v10:
            dyn_2d[t, :, 2] = 0.0    # water_volume (v10のみ予測)
        if not is_v11 and not is_v12:
            if t < e1d_dyn.shape[0]:
                e1d_dyn[t, :, :] = 0.0  # 1Dエッジ (v11/v12が予測)
        if not is_v12:
            if t < e2d_dyn.shape[0]:
                e2d_dyn[t, :, :] = 0.0  # 2Dエッジ (v12のみ予測)

    # v6: GRU hidden state initialization
    h_1d, h_2d = None, None
    # v19 (GRU): 1D GRU hidden state
    h_1d_gru = None
    # v10: auxiliary variable tracking (inlet_flow, water_volume)
    cur_inlet = dyn_1d[last_gt_t, :, 1].copy() if is_v10 else None
    cur_volume = dyn_2d[last_gt_t, :, 2].copy() if is_v10 else None
    # v11: no special state init needed (flux overwrite handled in loop)

    with torch.no_grad():
        # v19 (GRU): spin-up warmup — run through GT steps to build h_1d
        if is_gru:
            gru_warmup_start = max(0, spin_up - 8)
            for t in range(gru_warmup_start, spin_up):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    graph = build_graph_at_timestep(
                        config, event, t, prev_t=max(t - 1, 0), norm_stats=norm_stats,
                        future_rain_steps=future_rain_steps,
                        coupling_features=coupling_features,
                        per_node_stats=per_node_stats,
                        manning_features=manning_features,
                    ).to(device)
                    _, h_1d_gru = model(graph, h_1d=h_1d_gru)

        # v6: spin-up warmup — run GRU through GT steps to build hidden state
        if is_v6:
            for t in range(spin_up):
                sr = t / max(max_t - 1, 1)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    graph = build_graph_at_timestep(
                        config, event, t, prev_t=max(t - 1, 0), norm_stats=norm_stats,
                        future_rain_steps=future_rain_steps,
                        coupling_features=coupling_features,
                        step_ratio=sr,
                        per_node_stats=per_node_stats,
                        manning_features=manning_features,
                    ).to(device)
                    _, h_1d, h_2d = model(graph, h_1d, h_2d)

        # ---- TB (Temporal Bundling) K-step bundle rollout ----
        if is_tb:
            K = tb_K
            for bundle_start in range(0, pred_steps, K):
                t = last_gt_t + bundle_start
                # Write current state into dyn arrays
                if bundle_start > 0:
                    dyn_1d[t, :, 0] = current_wl_1d
                    dyn_2d[t, :, 1] = current_wl_2d

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    graph = build_graph_at_timestep(
                        config, event, t, prev_t=t - 1, norm_stats=norm_stats,
                        future_rain_steps=future_rain_steps,
                        coupling_features=coupling_features,
                        per_node_stats=per_node_stats,
                        manning_features=manning_features,
                    ).to(device)
                    out = model(graph)

                out_1d_np = out["1d"].float().cpu().numpy()  # [N_1d, K*2]
                out_2d_np = out["2d"].float().cpu().numpy()  # [N_2d, K]

                # Apply K sub-step deltas sequentially
                steps_in_bundle = min(K, pred_steps - bundle_start)
                for k in range(steps_in_bundle):
                    delta_1d = out_1d_np[:, 2 * k]      # wl_delta for sub-step k
                    delta_2d = out_2d_np[:, k]           # wl_delta for sub-step k
                    # Per-node denormalization
                    if _ds is not None:
                        delta_1d = delta_1d * _ds["1d_wl_std"]
                        delta_2d = delta_2d * _ds["2d_wl_std"]
                    # Delta clipping
                    if dc_1d is not None:
                        delta_1d = np.clip(delta_1d, dc_1d[0], dc_1d[1])
                    if dc_2d is not None:
                        delta_2d = np.clip(delta_2d, dc_2d[0], dc_2d[1])

                    next_wl_1d = current_wl_1d + delta_1d
                    next_wl_2d = current_wl_2d + delta_2d
                    if wb_1d is not None:
                        next_wl_1d = np.clip(next_wl_1d, wb_1d[0], wb_1d[1])
                    if wb_2d is not None:
                        next_wl_2d = np.clip(next_wl_2d, wb_2d[0], wb_2d[1])

                    pred_1d_all.append(next_wl_1d.copy())
                    pred_2d_all.append(next_wl_2d.copy())
                    current_wl_1d = next_wl_1d
                    current_wl_2d = next_wl_2d

                    # Write intermediate state for next bundle's graph construction
                    next_t = t + k + 1
                    if next_t < dyn_1d.shape[0]:
                        dyn_1d[next_t, :, 0] = current_wl_1d
                        dyn_2d[next_t, :, 1] = current_wl_2d
                        # Flux overwrite: inlet_flow
                        dyn_1d[next_t, :, 1] = out_1d_np[:, 2 * k + 1]
                    # Edge flux overwrite
                    if "1d_edge" in out and next_t < e1d_dyn.shape[0]:
                        edge_np = out["1d_edge"].float().cpu().numpy()
                        e1d_dyn[next_t, :, 0] = edge_np[:, 2 * k]
                        e1d_dyn[next_t, :, 1] = edge_np[:, 2 * k + 1]

        # ---- Standard single-step rollout ----
        else:
            for step in range(pred_steps):
                t = last_gt_t + step
                if step > 0:
                    dyn_1d[t, :, 0] = current_wl_1d
                    dyn_2d[t, :, 1] = current_wl_2d
                    # v10: 補助変数もフィードバック (delta累積)
                    if is_v10:
                        dyn_1d[t, :, 1] = cur_inlet
                        dyn_2d[t, :, 2] = cur_volume
                    # v11: inlet_flow + 1D edge は前stepで上書き済み (下記参照)

                sr = t / max(max_t - 1, 1) if is_v6 else None
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    graph = build_graph_at_timestep(
                        config, event, t, prev_t=t - 1, norm_stats=norm_stats,
                        future_rain_steps=future_rain_steps,
                        coupling_features=coupling_features,
                        step_ratio=sr,
                        per_node_stats=per_node_stats,
                        prev_t2=t - 2 if (is_v15 or use_prev_t2) else None,
                        hydraulic_gradients=is_v15,
                        manning_features=manning_features,
                    ).to(device)
                    if is_v6:
                        out, h_1d, h_2d = model(graph, h_1d, h_2d)
                    elif is_gru:
                        out, h_1d_gru = model(graph, h_1d=h_1d_gru)
                    else:
                        out = model(graph)

                # v5 dual output対応: {"nodes": {...}, "edges": {...}}
                if isinstance(out, dict) and "nodes" in out:
                    out = out["nodes"]

                # v12: DUALFloodGNN (node delta + edge delta feedback)
                if is_v12:
                    out_1d_np = out["1d"].float().cpu().numpy()
                    out_2d_np = out["2d"].float().cpu().numpy()
                    delta_1d = out_1d_np[:, 0]
                    delta_2d = out_2d_np[:, 0]
                    # Edge delta feedback: 累積して次step入力
                    next_t = t + 1
                    if "1d_edge" in out and next_t < e1d_dyn.shape[0]:
                        ep = out["1d_edge"].float().cpu().numpy()
                        e1d_dyn[next_t, :, 0] = e1d_dyn[t, :, 0] + ep[:, 0]
                        e1d_dyn[next_t, :, 1] = e1d_dyn[t, :, 1] + ep[:, 1]
                    if "2d_edge" in out and next_t < e2d_dyn.shape[0]:
                        ep = out["2d_edge"].float().cpu().numpy()
                        e2d_dyn[next_t, :, 0] = e2d_dyn[t, :, 0] + ep[:, 0]
                        e2d_dyn[next_t, :, 1] = e2d_dyn[t, :, 1] + ep[:, 1]
                # v11/v11b: flux/state paradigm
                elif is_v11:
                    out_1d_np = out["1d"].float().cpu().numpy()
                    out_2d_np = out["2d"].float().cpu().numpy()
                    delta_1d = out_1d_np[:, 0]
                    delta_2d = out_2d_np[:, 0]
                    # v11b/v13 per-node denormalization: model outputs normalized delta
                    if _ds is not None:
                        delta_1d = delta_1d * _ds["1d_wl_std"]
                        delta_2d = delta_2d * _ds["2d_wl_std"]
                    # Heun's corrector: predict→correct, average slopes for 2nd-order accuracy
                    if use_heun:
                        k1_1d, k1_2d = delta_1d.copy(), delta_2d.copy()
                        nt = t + 1
                        if nt < dyn_1d.shape[0]:
                            # Tentative Euler step + predictor flux for corrector input
                            dyn_1d[nt, :, 0] = current_wl_1d + k1_1d
                            dyn_2d[nt, :, 1] = current_wl_2d + k1_2d
                            dyn_1d[nt, :, 1] = out_1d_np[:, 1]
                            if "1d_edge" in out and nt < e1d_dyn.shape[0]:
                                ep = out["1d_edge"].float().cpu().numpy()
                                e1d_dyn[nt, :, 0] = ep[:, 0]
                                e1d_dyn[nt, :, 1] = ep[:, 1]
                            # Corrector: evaluate model at t+1 with tentative state
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                gc = build_graph_at_timestep(
                                    config, event, nt, prev_t=t,
                                    norm_stats=norm_stats,
                                    future_rain_steps=future_rain_steps,
                                    coupling_features=coupling_features,
                                    per_node_stats=per_node_stats,
                                    prev_t2=t - 1 if (is_v15 or use_prev_t2) else None,
                                    hydraulic_gradients=is_v15,
                                    manning_features=manning_features,
                                ).to(device)
                                out = model(gc)
                            # Replace out_1d_np so flux writes below use corrector output
                            out_1d_np = out["1d"].float().cpu().numpy()
                            k2_1d = out_1d_np[:, 0]
                            k2_2d = out["2d"].float().cpu().numpy()[:, 0]
                            if _ds is not None:
                                k2_1d = k2_1d * _ds["1d_wl_std"]
                                k2_2d = k2_2d * _ds["2d_wl_std"]
                            delta_1d = (k1_1d + k2_1d) / 2.0
                            delta_2d = (k1_2d + k2_2d) / 2.0
                    # Flux overwrite: inlet_flow (次step入力) — corrector output if Heun
                    next_t = t + 1
                    if next_t < dyn_1d.shape[0]:
                        dyn_1d[next_t, :, 1] = out_1d_np[:, 1]  # absolute overwrite
                    # Flux overwrite: 1D edge flow/velocity (次step入力)
                    if "1d_edge" in out and next_t < e1d_dyn.shape[0]:
                        edge_pred = out["1d_edge"].float().cpu().numpy()
                        e1d_dyn[next_t, :, 0] = edge_pred[:, 0]
                        e1d_dyn[next_t, :, 1] = edge_pred[:, 1]
                # v10/v10b: 2-dim output → [:, 0] = wl_delta, [:, 1] = aux_delta
                elif is_v10:
                    out_1d_np = out["1d"].float().cpu().numpy()
                    out_2d_np = out["2d"].float().cpu().numpy()
                    delta_1d = out_1d_np[:, 0]
                    delta_2d = out_2d_np[:, 0]
                    # v10b: per-node WL delta denormalization (column 0 only)
                    if _ds is not None:
                        delta_1d = delta_1d * _ds["1d_wl_std"]
                        delta_2d = delta_2d * _ds["2d_wl_std"]
                    cur_inlet = cur_inlet + out_1d_np[:, 1]  # inlet_flow delta累積
                    cur_volume = cur_volume + out_2d_np[:, 1]  # water_volume delta累積
                else:
                    delta_1d = out["1d"].squeeze(-1).float().cpu().numpy()
                    delta_2d = out["2d"].squeeze(-1).float().cpu().numpy()
                    # v4cb etc: per-node denormalization for non-v11 models
                    if _ds is not None:
                        delta_1d = delta_1d * _ds["1d_wl_std"]
                        delta_2d = delta_2d * _ds["2d_wl_std"]
                    # Heun's corrector for basic models (v4c etc)
                    if use_heun:
                        k1_1d, k1_2d = delta_1d.copy(), delta_2d.copy()
                        nt = t + 1
                        if nt < dyn_1d.shape[0]:
                            dyn_1d[nt, :, 0] = current_wl_1d + k1_1d
                            dyn_2d[nt, :, 1] = current_wl_2d + k1_2d
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                gc = build_graph_at_timestep(
                                    config, event, nt, prev_t=t,
                                    norm_stats=norm_stats,
                                    future_rain_steps=future_rain_steps,
                                    coupling_features=coupling_features,
                                    per_node_stats=per_node_stats,
                                    prev_t2=t - 1 if (is_v15 or use_prev_t2) else None,
                                    hydraulic_gradients=is_v15,
                                    manning_features=manning_features,
                                ).to(device)
                                out_c = model(gc)
                            k2_1d = out_c["1d"].squeeze(-1).float().cpu().numpy()
                            k2_2d = out_c["2d"].squeeze(-1).float().cpu().numpy()
                            if _ds is not None:
                                k2_1d = k2_1d * _ds["1d_wl_std"]
                                k2_2d = k2_2d * _ds["2d_wl_std"]
                            delta_1d = (k1_1d + k2_1d) / 2.0
                            delta_2d = (k1_2d + k2_2d) / 2.0

                # Per-node delta bias correction (AR drift compensation)
                if delta_bias_corr is not None:
                    delta_1d = delta_1d - delta_bias_corr["delta_bias_1d"]
                    delta_2d = delta_2d - delta_bias_corr["delta_bias_2d"]

                # Recession physics clip: rain=0 → delta_1d <= 0 (排水のみ)
                if recession_clip and t >= spin_up + 10:
                    rain_2d = dyn_2d[t, :, 0]  # [N_2d] GT rainfall
                    conn = config.connections_1d2d  # [N_conn, 2] = (node_1d, node_2d)
                    coupled_rain = np.zeros(config.num_1d_nodes, dtype=np.float32)
                    for node_1d, node_2d in conn:
                        coupled_rain[node_1d] += rain_2d[node_2d]
                    # rain=0 のノードのみ delta を 0以下にクリップ
                    no_rain_mask = coupled_rain == 0.0
                    delta_1d = np.where(no_rain_mask & (delta_1d > 0), 0.0, delta_1d)

                # Delta clipping: P0.5/P99.5で制限 (error explosion抑制)
                if dc_1d is not None:
                    delta_1d = np.clip(delta_1d, dc_1d[0], dc_1d[1])
                if dc_2d is not None:
                    delta_2d = np.clip(delta_2d, dc_2d[0], dc_2d[1])

                next_wl_1d = current_wl_1d + delta_1d
                next_wl_2d = current_wl_2d + delta_2d

                # WL bounds: train min-margin ~ max+margin
                if wb_1d is not None:
                    next_wl_1d = np.clip(next_wl_1d, wb_1d[0], wb_1d[1])
                if wb_2d is not None:
                    next_wl_2d = np.clip(next_wl_2d, wb_2d[0], wb_2d[1])

                pred_1d_all.append(next_wl_1d.copy())
                pred_2d_all.append(next_wl_2d.copy())

                current_wl_1d = next_wl_1d
                current_wl_2d = next_wl_2d

    event.nodes_1d_dynamic = orig_1d
    event.nodes_2d_dynamic = orig_2d
    event.edges_1d_dynamic = orig_e1d
    event.edges_2d_dynamic = orig_e2d

    return {
        "pred_1d": np.stack(pred_1d_all),
        "pred_2d": np.stack(pred_2d_all),
        "start_timestep": spin_up + 1,
    }


def predict_event_fno(
    model: FloodFNO,
    config,
    event,
    norm_stats: dict,
    grid_interp: GridInterpolator,
    device: str,
    spin_up: int = 10,
    future_rain_steps: int = 10,
    use_mask_channel: bool = False,
) -> dict:
    """FNO v7/v7b autoregressive prediction for one event.

    use_mask_channel: If True (v7b), append valid_mask as extra input channel
                      and skip zero-masking in mesh_to_grid.
    """
    model.eval()
    max_t = event.num_timesteps
    last_gt_t = spin_up - 1
    pred_steps = max_t - spin_up

    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    e1d_dyn = event.edges_1d_dynamic.copy()
    e2d_dyn = event.edges_2d_dynamic.copy()
    current_wl_1d = dyn_1d[last_gt_t, :, 0].copy()
    current_wl_2d = dyn_2d[last_gt_t, :, 1].copy()

    orig_1d = event.nodes_1d_dynamic
    orig_2d = event.nodes_2d_dynamic
    orig_e1d = event.edges_1d_dynamic
    orig_e2d = event.edges_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = e1d_dyn
    event.edges_2d_dynamic = e2d_dyn

    # Precompute mask channel for v7b
    mask_ch = grid_interp.get_mask_channel() if use_mask_channel else None  # (1, H, W)

    # Zero out test-unavailable features
    for t in range(spin_up, max_t):
        dyn_1d[t, :, 1] = 0.0
        dyn_2d[t, :, 2] = 0.0
        if t < e1d_dyn.shape[0]:
            e1d_dyn[t, :, :] = 0.0
        if t < e2d_dyn.shape[0]:
            e2d_dyn[t, :, :] = 0.0

    pred_1d_all, pred_2d_all = [], []

    with torch.no_grad():
        for step in range(pred_steps):
            t = last_gt_t + step
            if step > 0:
                dyn_1d[t, :, 0] = current_wl_1d
                dyn_2d[t, :, 1] = current_wl_2d

            graph = build_graph_at_timestep(
                config, event, t, prev_t=t - 1, norm_stats=norm_stats,
                inject_rainfall=True, extra_features=True,
                future_rain_steps=future_rain_steps,
            )
            x_2d = graph["2d"].x.to(device)
            x_1d = graph["1d"].x.to(device)

            # v7b: no masking + mask channel, v7: standard masking
            grid_2d = grid_interp.mesh_to_grid(x_2d, apply_mask=not use_mask_channel).unsqueeze(0)
            if mask_ch is not None:
                grid_2d = torch.cat([grid_2d, mask_ch.unsqueeze(0)], dim=1)

            out = model(grid_2d, x_1d)
            delta_2d = grid_interp.grid_to_mesh(out["2d_grid"].squeeze(0))
            delta_1d = out["1d"]

            d1 = delta_1d.squeeze().cpu().numpy()
            d2 = delta_2d.squeeze().cpu().numpy()

            current_wl_1d = current_wl_1d + d1
            current_wl_2d = current_wl_2d + d2

            pred_1d_all.append(current_wl_1d.copy())
            pred_2d_all.append(current_wl_2d.copy())

    event.nodes_1d_dynamic = orig_1d
    event.nodes_2d_dynamic = orig_2d
    event.edges_1d_dynamic = orig_e1d
    event.edges_2d_dynamic = orig_e2d

    return {
        "pred_1d": np.stack(pred_1d_all),
        "pred_2d": np.stack(pred_2d_all),
        "start_timestep": spin_up + 1,
    }


def predict_event_stepwise(
    models: list[torch.nn.Module],
    config,
    event,
    norm_stats: dict,
    device: str,
    per_node_stats: dict,
    spin_up: int = 10,
    future_rain_steps: int = 0,
    recession_clip: bool = False,
) -> dict:
    """Step-wise ensemble: 各stepで全モデルのdeltaを平均してフィードバック.

    AR誤差蓄積率を sigma/sqrt(K) に削減。全モデルがv11系であることを前提。
    """
    K = len(models)
    for m in models:
        m.eval()
    max_t = event.num_timesteps
    last_gt_t = spin_up - 1
    n_1d = config.num_1d_nodes
    n_2d = config.num_2d_nodes

    pn_std_1d = per_node_stats["1d_wl_std"]
    pn_std_2d = per_node_stats["2d_wl_std"]

    # 共有状態
    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    dyn_e1d = event.edges_1d_dynamic.copy()
    dyn_e2d = event.edges_2d_dynamic.copy()

    orig_1d = event.nodes_1d_dynamic
    orig_2d = event.nodes_2d_dynamic
    orig_e1d = event.edges_1d_dynamic
    orig_e2d = event.edges_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = dyn_e1d
    event.edges_2d_dynamic = dyn_e2d

    # 未知チャンネルをゼロ化
    for t in range(spin_up, max_t):
        dyn_2d[t, :, 2] = 0.0
        if t < dyn_e2d.shape[0]:
            dyn_e2d[t, :, :] = 0.0

    current_wl_1d = dyn_1d[last_gt_t, :, 0].copy()
    current_wl_2d = dyn_2d[last_gt_t, :, 1].copy()

    pred_1d_all = []
    pred_2d_all = []

    with torch.no_grad():
        for t in range(spin_up, max_t):
            # 共有状態を更新
            dyn_1d[t, :, 0] = current_wl_1d
            dyn_2d[t, :, 1] = current_wl_2d

            # 各モデルの delta を収集
            delta_1d_sum = np.zeros(n_1d, dtype=np.float64)
            delta_2d_sum = np.zeros(n_2d, dtype=np.float64)
            inlet_sum = np.zeros(n_1d, dtype=np.float64)
            edge_sum = None

            for model in models:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    g = build_graph_at_timestep(
                        config, event, t, prev_t=t - 1,
                        norm_stats=norm_stats,
                        future_rain_steps=future_rain_steps,
                        coupling_features=True,
                        per_node_stats=per_node_stats,
                    ).to(device)
                    out = model(g)

                out_1d = out["1d"].float().cpu().numpy()
                out_2d = out["2d"].float().cpu().numpy()
                delta_1d_sum += out_1d[:, 0] * pn_std_1d
                delta_2d_sum += out_2d[:, 0] * pn_std_2d
                inlet_sum += out_1d[:, 1]
                if "1d_edge" in out:
                    ep = out["1d_edge"].float().cpu().numpy()
                    if edge_sum is None:
                        edge_sum = ep.astype(np.float64)
                    else:
                        edge_sum += ep

            # 平均 delta
            delta_1d = (delta_1d_sum / K).astype(np.float32)
            delta_2d = (delta_2d_sum / K).astype(np.float32)

            # Recession clip
            if recession_clip and t >= spin_up + 10:
                rain_2d = dyn_2d[t, :, 0]
                conn = config.connections_1d2d
                coupled_rain = np.zeros(n_1d, dtype=np.float32)
                for node_1d, node_2d in conn:
                    coupled_rain[node_1d] += rain_2d[node_2d]
                no_rain_mask = coupled_rain == 0.0
                delta_1d = np.where(no_rain_mask & (delta_1d > 0), 0.0, delta_1d)

            current_wl_1d = current_wl_1d + delta_1d
            current_wl_2d = current_wl_2d + delta_2d
            pred_1d_all.append(current_wl_1d.copy())
            pred_2d_all.append(current_wl_2d.copy())

            # 共有 flux feedback (平均)
            next_t = t + 1
            if next_t < max_t:
                dyn_1d[next_t, :, 1] = (inlet_sum / K).astype(np.float32)
            if edge_sum is not None and next_t < dyn_e1d.shape[0]:
                avg_edge = (edge_sum / K).astype(np.float32)
                dyn_e1d[next_t, :, 0] = avg_edge[:, 0]
                dyn_e1d[next_t, :, 1] = avg_edge[:, 1]

    event.nodes_1d_dynamic = orig_1d
    event.nodes_2d_dynamic = orig_2d
    event.edges_1d_dynamic = orig_e1d
    event.edges_2d_dynamic = orig_e2d

    return {
        "pred_1d": np.stack(pred_1d_all),
        "pred_2d": np.stack(pred_2d_all),
        "start_timestep": spin_up + 1,
    }


def build_submission(predictions: list) -> pd.DataFrame:
    """submission DataFrame構築 (node_type → node_id → timestep 順).

    sample_submissionの行順序: model → event → node_type → node_id → timestep
    """
    dfs = []
    for model_id, event_id, preds in predictions:
        pred_1d = preds["pred_1d"]  # (T, N_1d)
        pred_2d = preds["pred_2d"]  # (T, N_2d)
        T_pred, N_1d = pred_1d.shape
        _, N_2d = pred_2d.shape

        # 1D: node_id → timestep 順 (node_idでrepeat, ts内でtile)
        nid_1d = np.repeat(np.arange(N_1d), T_pred)
        ts_1d = np.tile(np.arange(T_pred), N_1d)
        # pred_1d[t, n] → node順に並べ替え: pred_1d.T.flatten() = [n0_t0,n0_t1,...,n1_t0,...]
        df_1d = pd.DataFrame({
            "model_id": model_id,
            "event_id": event_id,
            "node_type": 1,
            "node_id": nid_1d,
            "water_level": pred_1d.T.flatten(),
        })

        # 2D: 同様
        nid_2d = np.repeat(np.arange(N_2d), T_pred)
        ts_2d = np.tile(np.arange(T_pred), N_2d)
        df_2d = pd.DataFrame({
            "model_id": model_id,
            "event_id": event_id,
            "node_type": 2,
            "node_id": nid_2d,
            "water_level": pred_2d.T.flatten(),
        })

        # node_type=1が先、node_type=2が後 (既にそうなっている)
        dfs.append(df_1d)
        dfs.append(df_2d)

    result = pd.concat(dfs, ignore_index=True)
    result.insert(0, "row_id", range(len(result)))
    return result


def load_model(ckpt_path: str, hidden_dim: int, num_layers: int, device: str,
               use_v2: bool = False) -> torch.nn.Module:
    """チェックポイントからモデル読み込み. v1/v2/v3/v4/v4c/v11b/v11b_tb自動検出."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_version = "v1"
    coupling_edge_dim = 0
    per_node_norm = False
    tb_K = 4  # temporal bundling K (default)
    if use_v2:
        model_version = "v2"
    if isinstance(ckpt, dict):
        ver_flag = ckpt.get("model_version", "")
        coupling_edge_dim = ckpt.get("coupling_edge_dim", 0)
        per_node_norm = ckpt.get("per_node_norm", False)
        tb_K = ckpt.get("K", 4)
        # v97: auto-detect hidden_dim/num_layers from checkpoint
        hidden_dim = ckpt.get("hidden_dim", hidden_dim)
        num_layers = ckpt.get("num_processor_layers", num_layers)
        if ver_flag == "v99_history":
            model_version = "v99"
        elif ver_flag == "v97":
            model_version = "v97"
        elif ver_flag == "v11_GAT":
            model_version = "v11_GAT"
        elif ver_flag == "v21":
            model_version = "v21"
        elif ver_flag == "v20":
            model_version = "v20"
        elif ver_flag == "v11c_gru2":
            model_version = "v11c_gru2"
        elif ver_flag == "v11c_gru":
            model_version = "v11c_gru"
        elif ver_flag in ("v11b_tb", "v11c_tb"):
            model_version = "v11b_tb"  # same arch, different training
        elif ver_flag == "v11c":
            model_version = "v11b"  # v11c = same arch as v11b, water_volume zeroed training
        elif ver_flag == "v15":
            model_version = "v15"
        elif ver_flag == "v18":
            model_version = "v18"
        elif ver_flag == "v17":
            model_version = "v17"  # same arch as v11, per-node norm, from-scratch seed
        elif ver_flag == "v13":
            model_version = "v13"  # same arch as v11, delta normalization
        elif ver_flag == "v12":
            model_version = "v12"
        elif ver_flag == "v11b":
            model_version = "v11b"  # same arch as v11, per-node normalization
        elif ver_flag == "v11":
            model_version = "v11"
        elif ver_flag == "v10b":
            model_version = "v10b"  # same arch as v10, per-node WL normalization
        elif ver_flag == "v10":
            model_version = "v10"
        elif ver_flag == "v6b":
            model_version = "v6b"  # same arch as v6, per-node normalization
        elif ver_flag == "v6":
            model_version = "v6"
        elif ver_flag == "v5":
            model_version = "v5"
        elif ver_flag == "v4cb":
            model_version = "v4cb"  # same arch as v4c, per-node normalization
        elif ver_flag in ("v4c", "v4"):
            model_version = "v4c" if coupling_edge_dim > 0 else "v4"
        elif ver_flag == "v3":
            model_version = "v3"
        elif ver_flag == "v2":
            model_version = "v2"

    if model_version == "v99":
        # v99: v45 arch (v11 + t-2 history), same as v11 but NODE_DIMS expanded (15/21)
        model = HeteroFloodGNNv45(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v97":
        # v97: multi-step K=5, 256dim/6layer, uses TB inference path
        model = HeteroFloodGNNv97(K=tb_K, hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                   coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v21":
        # v21: v11 + Manning equilibrium fill ratio (1D input 14 dims)
        model = HeteroFloodGNNv21(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v20":
        # v20: v11 + upstream_inlet_flow_sum (1D input 14 dims)
        model = HeteroFloodGNNv20(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v11_GAT":
        num_heads = ckpt.get("num_heads", 4) if isinstance(ckpt, dict) else 4
        model = HeteroFloodGNNv11_GAT(hidden_dim=hidden_dim, num_processor_layers=num_layers,
                                      num_heads=num_heads, noise_std=0.0,
                                      coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v11c_gru2":
        model = HeteroFloodGNNv11_GRU2(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                       coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v11c_gru":
        model = HeteroFloodGNNv11_GRU(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                      coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v11b_tb":
        model = HeteroFloodGNNv11_TB(K=tb_K, hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                     coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v15":
        model = HeteroFloodGNNv15(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v18":
        model = HeteroFloodGNNv18(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v17":
        # v17: same arch as v11, per-node norm, from-scratch seed averaging
        model = HeteroFloodGNNv11(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v13":
        # v13: same arch as v11, uses delta_std for normalization (not wl_std)
        model = HeteroFloodGNNv11(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v12":
        model = DUALFloodGNNv12(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version in ("v11", "v11b"):
        model = HeteroFloodGNNv11(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version in ("v10", "v10b"):
        model = HeteroFloodGNNv10(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                  coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version in ("v6", "v6b"):
        model = HeteroFloodGNNv6(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                 coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v5":
        model = HeteroFloodGNNv5(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0).to(device)
    elif model_version in ("v4c", "v4cb"):
        model = HeteroFloodGNNv4(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0,
                                 coupling_edge_dim=coupling_edge_dim).to(device)
    elif model_version == "v4":
        model = HeteroFloodGNNv4(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0).to(device)
    elif model_version == "v3":
        model = HeteroFloodGNNv3(hidden_dim=hidden_dim, num_processor_layers=num_layers, noise_std=0.0).to(device)
    elif model_version == "v2":
        model = HeteroFloodGNNv2(hidden_dim=hidden_dim, num_processor_layers=6).to(device)
    else:
        model = HeteroFloodGNN(hidden_dim=hidden_dim, num_processor_layers=num_layers).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        val = ckpt.get("best_val", "?")
        seed = ckpt.get("seed", "?")
        pn_str = " +per_node_norm" if per_node_norm else ""
        print(f"  Loaded {model_version} (seed={seed}, val={val:.4f}{pn_str}): {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded {model_version}: {ckpt_path}")
    # Attach per_node_norm flag for inference use
    model._per_node_norm = per_node_norm
    # TB: temporal bundling K value
    model._tb_K = tb_K if model_version in ("v11b_tb", "v97") else 0
    # GRU: v11c_gru / v11c_gru2 flag (both use forward(data, h_1d=None) -> (out, h_1d))
    model._is_gru = (model_version in ("v11c_gru", "v11c_gru2"))
    # v13: delta normalization (uses delta_std instead of wl_std for denorm)
    delta_norm = False
    if isinstance(ckpt, dict):
        delta_norm = ckpt.get("delta_norm", False)
    model._delta_norm = delta_norm
    # v21: Manning features flag
    manning = False
    if isinstance(ckpt, dict):
        manning = ckpt.get("manning_features", False)
    model._manning_features = manning
    # v99: prev_t2 history input flag
    prev_t2_flag = False
    if isinstance(ckpt, dict):
        prev_t2_flag = ckpt.get("prev_t2", False)
    model._prev_t2 = prev_t2_flag
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Dataset_Rerelease/Models")
    parser.add_argument("--checkpoint_dir", type=str, default="Models/checkpoints")
    parser.add_argument("--output", type=str, default="submission_ensemble.parquet")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--m2_tags", type=str, default="pushforward,pf_s123",
                        help="Model_2 checkpoint tags (comma-sep: pushforward,pf_s123)")
    parser.add_argument("--m1_tag", type=str, default=None,
                        help="Model_1 checkpoint tag (e.g., tc_m1)")
    parser.add_argument("--m1_v1", action="store_true",
                        help="Force use Model_1 v1 (ignore pushforward)")
    parser.add_argument("--m2_weights", type=str, default=None,
                        help="Model_2 ensemble weights (comma-sep, e.g. '0.3,0.7'). Default: equal weights")
    parser.add_argument("--no_clip", action="store_true",
                        help="Disable delta clipping for Model_2")
    parser.add_argument("--fno_2d", type=str, default=None,
                        help="FNO v7 tag for hybrid M2 2D prediction (e.g. v7_r32). "
                             "Uses GNN for 1D, FNO for 2D.")
    parser.add_argument("--hybrid_1d", type=str, default=None,
                        help="Separate M2 model tag for 1D predictions (e.g. v11_r32). "
                             "Main M2 ensemble used for 2D only.")
    parser.add_argument("--bias_corr", type=str, default=None,
                        help="Path to delta_bias_correction pickle for AR drift compensation")
    parser.add_argument("--heun", action="store_true",
                        help="Use Heun's method (predictor-corrector) for 2nd-order AR integration")
    parser.add_argument("--zone_bias", type=str, default=None,
                        help="Path to bias_stats pickle for zone-aware bias correction (post-processing)")
    parser.add_argument("--recession_clip", action="store_true",
                        help="Rain=0 recession constraint: clip 1D delta to <=0 (drainage only)")
    parser.add_argument("--stepwise", action="store_true",
                        help="Step-wise ensemble: average deltas at each step (reduces AR error accumulation)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    all_predictions = []
    tags = [t.strip() for t in args.m2_tags.split(",")]

    # Model_2 weighted ensemble
    m2_weights: list[float] | None = None
    if args.m2_weights:
        m2_weights = [float(w) for w in args.m2_weights.split(",")]
        assert len(m2_weights) == len(tags), f"weights ({len(m2_weights)}) must match tags ({len(tags)})"
        wsum = sum(m2_weights)
        m2_weights = [w / wsum for w in m2_weights]  # normalize
        print(f"Model_2 weighted ensemble: {list(zip(tags, m2_weights))}")

    # Model_2用 delta clipping統計 (error explosion対策)
    delta_stats_path = os.path.join(args.checkpoint_dir, "delta_stats_model_2.pkl")
    delta_stats_m2 = None
    if os.path.exists(delta_stats_path) and not args.no_clip:
        with open(delta_stats_path, "rb") as f:
            delta_stats_m2 = pickle.load(f)
        print(f"Loaded delta_stats for Model_2 clipping")
        print(f"  delta_clip_1d: {delta_stats_m2['delta_clip_1d']}")
        print(f"  delta_clip_2d: {delta_stats_m2['delta_clip_2d']}")
        print(f"  wl_bounds_1d:  {delta_stats_m2['wl_bounds_1d']}")
        print(f"  wl_bounds_2d:  {delta_stats_m2['wl_bounds_2d']}")

    # Model_2用 per-node normalization stats (v11b)
    pn_stats_path = os.path.join(args.checkpoint_dir, "per_node_stats_model_2.pkl")
    per_node_stats_m2 = None
    if os.path.exists(pn_stats_path):
        with open(pn_stats_path, "rb") as f:
            per_node_stats_m2 = pickle.load(f)
        print(f"Loaded per_node_stats for Model_2 (v11b per-node denorm)")

    # Model_2用 per-node delta normalization stats (v13)
    pn_delta_stats_path = os.path.join(args.checkpoint_dir, "per_node_delta_stats_model_2.pkl")
    per_node_delta_stats_m2 = None
    if os.path.exists(pn_delta_stats_path):
        with open(pn_delta_stats_path, "rb") as f:
            per_node_delta_stats_m2 = pickle.load(f)
        print(f"Loaded per_node_delta_stats for Model_2 (v13 delta norm)")

    # Model_1用 per-node normalization stats (v4cb)
    pn_stats_m1_path = os.path.join(args.checkpoint_dir, "per_node_stats_model_1.pkl")
    per_node_stats_m1 = None
    if os.path.exists(pn_stats_m1_path):
        with open(pn_stats_m1_path, "rb") as f:
            per_node_stats_m1 = pickle.load(f)
        print(f"Loaded per_node_stats for Model_1 (v4cb per-node denorm)")

    # Delta bias correction for AR drift compensation (Model_2 only)
    delta_bias_corr_m2 = None
    if args.bias_corr:
        with open(args.bias_corr, "rb") as f:
            delta_bias_corr_m2 = pickle.load(f)
        print(f"Loaded delta bias correction: "
              f"1D mean={delta_bias_corr_m2['delta_bias_1d'].mean():.5f}, "
              f"2D mean={delta_bias_corr_m2['delta_bias_2d'].mean():.5f}")

    # Zone-aware bias correction (post-processing on predicted WL)
    zone_bias_stats = None
    if args.zone_bias:
        with open(args.zone_bias, "rb") as f:
            zone_bias_stats = pickle.load(f)
        print(f"Loaded zone bias stats: {len(zone_bias_stats['zone_bias_1d'])} zones")
        for zi, bounds in enumerate(zone_bias_stats["zone_bounds"]):
            zb1 = zone_bias_stats["zone_bias_1d"].get(zi)
            zb2 = zone_bias_stats["zone_bias_2d"].get(zi)
            if zb1 is not None:
                print(f"  Zone {zi} (step {bounds[0]}-{bounds[1]}): "
                      f"1D |mean|={np.abs(zb1).mean():.5f}, 2D |mean|={np.abs(zb2).mean():.5f}")

    for model_id in [1, 2]:
        print(f"\n=== Inference Model_{model_id} ===")

        config = load_model_config(args.data_dir, model_id, split="train")

        # norm_statsキャッシュ (v2優先: 修正データ用)
        cache_v2 = os.path.join(args.checkpoint_dir, f"norm_stats_model_{model_id}_v2.pkl")
        cache_path = os.path.join(args.checkpoint_dir, f"norm_stats_model_{model_id}.pkl")
        if os.path.exists(cache_v2):
            with open(cache_v2, "rb") as f:
                norm_stats = pickle.load(f)
            print(f"  Loaded cached norm_stats (v2 corrected)")
        elif os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                norm_stats = pickle.load(f)
            print(f"  Loaded cached norm_stats (old)")
        else:
            norm_stats = compute_normalization_stats(args.data_dir, model_id, config)

        # Model_1: testcond / pushforward / v1
        if model_id == 1:
            if args.m1_tag:
                m1_path = os.path.join(args.checkpoint_dir, f"best_model_1_{args.m1_tag}.pt")
                models = [load_model(m1_path, args.hidden_dim, args.num_layers, device)]
            else:
                pf_path = os.path.join(args.checkpoint_dir, "best_model_1_pushforward.pt")
                v1_path = os.path.join(args.checkpoint_dir, "best_model_1.pt")
                if os.path.exists(pf_path) and not args.m1_v1:
                    models = [load_model(pf_path, args.hidden_dim, args.num_layers, device)]
                else:
                    models = [load_model(v1_path, args.hidden_dim, args.num_layers, device)]
        else:
            # Model_2: 複数チェックポイント
            models = []
            for tag in tags:
                if tag == "focused":
                    ckpt_name = "best_model_2_focused.pt"
                elif tag == "pushforward":
                    ckpt_name = "best_model_2_pushforward.pt"
                elif tag == "extended":
                    ckpt_name = "best_model_2_extended.pt"
                elif tag.startswith("ext_r"):
                    ckpt_name = f"best_model_2_{tag}.pt"
                elif tag.startswith("pf_"):
                    ckpt_name = f"best_model_2_{tag}.pt"
                else:
                    ckpt_name = f"best_model_2_{tag}.pt"
                ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
                if os.path.exists(ckpt_path):
                    models.append(load_model(ckpt_path, args.hidden_dim, args.num_layers, device))
                else:
                    print(f"  WARNING: {ckpt_path} not found, skipping")

            if not models:
                # fallback to v1
                ckpt_path = os.path.join(args.checkpoint_dir, "best_model_2.pt")
                models = [load_model(ckpt_path, args.hidden_dim, args.num_layers, device)]

            print(f"  Ensemble: {len(models)} models")

        # Load hybrid_1d model for M2 1D prediction (e.g. v11 for 1D, main ensemble for 2D)
        hybrid_1d_model = None
        if model_id == 2 and args.hybrid_1d:
            h1d_path = os.path.join(args.checkpoint_dir, f"best_model_2_{args.hybrid_1d}.pt")
            if os.path.exists(h1d_path):
                hybrid_1d_model = load_model(h1d_path, args.hidden_dim, args.num_layers, device)
                hybrid_1d_model.eval()
                print(f"  Hybrid 1D: loaded {args.hybrid_1d} -> 1D predictions from this model")
                print(f"  -> Main ensemble used for 2D only")
            else:
                print(f"  WARNING: hybrid_1d checkpoint {h1d_path} not found")

        # Load FNO model for hybrid M2 2D prediction
        fno_model = None
        fno_grid_interp = None
        fno_use_mask = False
        if model_id == 2 and args.fno_2d:
            fno_path = os.path.join(args.checkpoint_dir, f"best_model_2_{args.fno_2d}.pt")
            if os.path.exists(fno_path):
                fno_ckpt = torch.load(fno_path, map_location=device, weights_only=False)
                fno_use_mask = fno_ckpt.get("use_mask_channel", False)
                # Detect FNO dims from checkpoint
                sample_graph = build_graph_at_timestep(
                    config, load_event_data(args.data_dir, model_id,
                                            list_events(args.data_dir, model_id, "train")[0],
                                            config),
                    t=1, norm_stats=norm_stats,
                    inject_rainfall=True, extra_features=True,
                    future_rain_steps=10,
                )
                in_2d = sample_graph["2d"].x.shape[1]
                if fno_use_mask:
                    in_2d += 1  # v7b: +1 for valid mask channel
                in_1d = sample_graph["1d"].x.shape[1]
                fno_model = FloodFNO(
                    in_2d=in_2d, in_1d=in_1d,
                    fno_width=fno_ckpt.get("fno_width", 64),
                    fno_modes=fno_ckpt.get("fno_modes", 12),
                    fno_layers=fno_ckpt.get("fno_layers", 4),
                    noise_std=0.0,
                ).to(device)
                fno_model.load_state_dict(fno_ckpt["model_state_dict"])
                fno_model.eval()
                coords_2d = config.nodes_2d_static[:, :2]
                fno_grid_interp = GridInterpolator(
                    coords_2d, grid_size=fno_ckpt.get("grid_size", 64), device=device,
                )
                ver = fno_ckpt.get("model_version", "v7_fno")
                mask_str = " +mask" if fno_use_mask else ""
                print(f"  FNO hybrid: loaded {args.fno_2d} ({ver}{mask_str}, val={fno_ckpt.get('best_val', '?'):.4f})")
                print(f"  -> GNN for 1D, FNO for 2D")
            else:
                print(f"  WARNING: FNO checkpoint {fno_path} not found")

        test_event_ids = list_events(args.data_dir, model_id, "test")
        print(f"  Test events: {len(test_event_ids)} events")

        for eid in test_event_ids:
            t0 = time.time()
            event = load_event_data(args.data_dir, model_id, eid, config, split="test")

            # Step-wise ensemble for Model_2 (all v11)
            _use_stepwise = model_id == 2 and args.stepwise and len(models) > 1
            if _use_stepwise:
                pn_sw = per_node_stats_m2 if per_node_stats_m2 is not None else None
                # future_rain_steps: チェックポイントから取得 (全モデル同一前提)
                _frs_sw = 10  # v11c default
                final_preds = predict_event_stepwise(
                    models, config, event, norm_stats, device,
                    per_node_stats=pn_sw,
                    future_rain_steps=_frs_sw,
                    recession_clip=args.recession_clip,
                )

            # 各モデルで予測 (Model_2のみdelta clipping適用)
            ds = delta_stats_m2 if model_id == 2 else None
            preds_list = []
            if not _use_stepwise:
              for model in models:
                # v4/v4c/v5/v6はfuture_rain_steps=10で推論
                frs = 10 if isinstance(model, (HeteroFloodGNNv4, HeteroFloodGNNv5, HeteroFloodGNNv6, HeteroFloodGNNv10, DUALFloodGNNv12)) else 0
                # v4c/v6/v10: coupling_features=True (水頭差をedge featureに追加)
                cf = hasattr(model, "coupling_edge_dim") and model.coupling_edge_dim > 0
                # TB: temporal bundling (v11b_tb) — must check BEFORE v11
                tb_flag = isinstance(model, (HeteroFloodGNNv11_TB, HeteroFloodGNNv97))
                tb_K_val = model.K if tb_flag else 4
                # v6: GRU hidden state管理 + step_ratio + spin-up warmup
                v6_flag = isinstance(model, HeteroFloodGNNv6)
                # v10: 2-dim output + auxiliary feedback (v11はv4継承なのでv10にはマッチしない)
                v10_flag = isinstance(model, HeteroFloodGNNv10)
                # v11: flux/state paradigm (wl delta + inlet abs + edge abs)
                # TB inherits v11 but uses its own rollout, so v11_flag stays True for feature zeroing
                v11_flag = isinstance(model, HeteroFloodGNNv11)
                # v19 (GRU): 1D GRU hidden state管理
                gru_flag = getattr(model, "_is_gru", False)
                # v12: DUALFloodGNN (node delta + edge delta feedback)
                v12_flag = isinstance(model, DUALFloodGNNv12)
                # v15/v18: extended input features (prev_t2, hydraulic_gradients)
                v15_flag = isinstance(model, (HeteroFloodGNNv15, HeteroFloodGNNv18))
                # v99/v45: prev_t2 history only (no hydraulic_gradients)
                prev_t2_flag = getattr(model, "_prev_t2", False)
                # per-node normalization: M2=v11b/v4cb/v10b, M1=v4cb
                # per_node_stats: graph入力用 (wl_mean/wl_std)
                # denorm_stats: 出力非正規化用 (v13のみdelta_std)
                if model_id == 2 and getattr(model, "_per_node_norm", False):
                    pn = per_node_stats_m2
                elif model_id == 1 and getattr(model, "_per_node_norm", False):
                    pn = per_node_stats_m1
                else:
                    pn = None
                # v13: delta normalization → 出力非正規化にdelta_stdを使用
                dn = None
                if model_id == 2 and getattr(model, "_delta_norm", False) and per_node_delta_stats_m2 is not None:
                    dn = {
                        "1d_wl_std": per_node_delta_stats_m2["1d_delta_std"],
                        "2d_wl_std": per_node_delta_stats_m2["2d_delta_std"],
                    }
                manning_flag = getattr(model, "_manning_features", False)
                # Delta bias correction: Model_2 only (single-model inference)
                dbc = delta_bias_corr_m2 if model_id == 2 else None
                preds = predict_event(model, config, event, norm_stats, device,
                                      delta_stats=ds, future_rain_steps=frs,
                                      coupling_features=cf, is_v6=v6_flag,
                                      is_v10=v10_flag, is_v11=v11_flag,
                                      is_v12=v12_flag, is_v15=v15_flag,
                                      use_prev_t2=prev_t2_flag,
                                      is_gru=gru_flag,
                                      is_tb=tb_flag, tb_K=tb_K_val,
                                      per_node_stats=pn, denorm_stats=dn,
                                      manning_features=manning_flag,
                                      delta_bias_corr=dbc,
                                      use_heun=args.heun,
                                      recession_clip=args.recession_clip)
                preds_list.append(preds)

            # アンサンブル (weighted or equal average) — stepwiseの場合はスキップ
            if not _use_stepwise:
                if len(preds_list) == 1:
                    final_preds = preds_list[0]
                elif model_id == 2 and m2_weights is not None:
                    w_1d = sum(w * p["pred_1d"] for w, p in zip(m2_weights, preds_list))
                    w_2d = sum(w * p["pred_2d"] for w, p in zip(m2_weights, preds_list))
                    final_preds = {
                        "pred_1d": w_1d,
                        "pred_2d": w_2d,
                        "start_timestep": preds_list[0]["start_timestep"],
                    }
                else:
                    final_preds = {
                        "pred_1d": np.mean([p["pred_1d"] for p in preds_list], axis=0),
                        "pred_2d": np.mean([p["pred_2d"] for p in preds_list], axis=0),
                        "start_timestep": preds_list[0]["start_timestep"],
                    }

            # Hybrid FNO: replace 2D predictions with FNO output
            if model_id == 2 and fno_model is not None:
                fno_preds = predict_event_fno(
                    fno_model, config, event, norm_stats,
                    fno_grid_interp, device, future_rain_steps=10,
                    use_mask_channel=fno_use_mask,
                )
                final_preds["pred_2d"] = fno_preds["pred_2d"]

            # Hybrid 1D: replace 1D predictions with separate model output
            if model_id == 2 and hybrid_1d_model is not None:
                frs_h = 10 if isinstance(hybrid_1d_model, (HeteroFloodGNNv4, HeteroFloodGNNv5, HeteroFloodGNNv6, HeteroFloodGNNv10, DUALFloodGNNv12)) else 0
                cf_h = hasattr(hybrid_1d_model, "coupling_edge_dim") and hybrid_1d_model.coupling_edge_dim > 0
                pn_h = per_node_stats_m2 if getattr(hybrid_1d_model, "_per_node_norm", False) else None
                h1d_preds = predict_event(
                    hybrid_1d_model, config, event, norm_stats, device,
                    delta_stats=ds, future_rain_steps=frs_h,
                    coupling_features=cf_h,
                    is_v6=isinstance(hybrid_1d_model, HeteroFloodGNNv6),
                    is_v10=isinstance(hybrid_1d_model, HeteroFloodGNNv10),
                    is_v11=isinstance(hybrid_1d_model, HeteroFloodGNNv11),
                    is_v12=isinstance(hybrid_1d_model, DUALFloodGNNv12),
                    is_v15=isinstance(hybrid_1d_model, (HeteroFloodGNNv15, HeteroFloodGNNv18)),
                    use_prev_t2=getattr(hybrid_1d_model, "_prev_t2", False),
                    per_node_stats=pn_h,
                    manning_features=getattr(hybrid_1d_model, "_manning_features", False),
                    recession_clip=args.recession_clip,
                )
                final_preds["pred_1d"] = h1d_preds["pred_1d"]

            # Zone-aware bias correction (Model_2 only, post-processing)
            if model_id == 2 and zone_bias_stats is not None:
                pred_1d = final_preds["pred_1d"]  # (T, N_1d)
                pred_2d = final_preds["pred_2d"]  # (T, N_2d)
                T = pred_1d.shape[0]
                zb1d = zone_bias_stats["zone_bias_1d"]
                zb2d = zone_bias_stats["zone_bias_2d"]
                zbounds = zone_bias_stats["zone_bounds"]
                is_rain_cond = zone_bias_stats.get("rain_conditioned", False)
                spin_up = 10
                for t in range(T):
                    zi = next(
                        (i for i, (lo, hi) in enumerate(zbounds) if lo <= t < hi),
                        len(zbounds) - 1,
                    )
                    if is_rain_cond:
                        abs_t = spin_up + t
                        # 1D: per-node coupled rain
                        rain_2d_t = event.nodes_2d_dynamic[abs_t, :, 0] if abs_t < event.nodes_2d_dynamic.shape[0] else np.zeros(event.nodes_2d_dynamic.shape[1])
                        conn = config.connections_1d2d
                        coupled = np.zeros(config.num_1d_nodes, dtype=np.float32)
                        counts = np.zeros(config.num_1d_nodes, dtype=np.float32)
                        for n1d, n2d in conn:
                            coupled[n1d] += rain_2d_t[n2d]
                            counts[n1d] += 1
                        mask_c = counts > 0
                        coupled[mask_c] /= counts[mask_c]
                        no_rain_1d = coupled < 1e-6
                        # Apply rain-conditioned bias per node
                        r0_1d = zone_bias_stats["rain0_bias_1d"]
                        r1_1d = zone_bias_stats["rain1_bias_1d"]
                        if zi in r0_1d:
                            pred_1d[t, no_rain_1d] -= r0_1d[zi][no_rain_1d]
                            pred_1d[t, ~no_rain_1d] -= r1_1d[zi][~no_rain_1d]
                        # 2D: direct rain per node
                        no_rain_2d = rain_2d_t < 1e-6
                        r0_2d = zone_bias_stats["rain0_bias_2d"]
                        r1_2d = zone_bias_stats["rain1_bias_2d"]
                        if zi in r0_2d:
                            pred_2d[t, no_rain_2d] -= r0_2d[zi][no_rain_2d]
                            pred_2d[t, ~no_rain_2d] -= r1_2d[zi][~no_rain_2d]
                    else:
                        if zi in zb1d:
                            pred_1d[t] -= zb1d[zi]
                        if zi in zb2d:
                            pred_2d[t] -= zb2d[zi]

            elapsed = time.time() - t0
            hybrid_flags = []
            if model_id == 2 and fno_model is not None:
                hybrid_flags.append("fno-2d")
            if model_id == 2 and hybrid_1d_model is not None:
                hybrid_flags.append("hybrid-1d")
            if model_id == 2 and zone_bias_stats is not None:
                hybrid_flags.append("zone-bias")
            suffix = f" [{'+'.join(hybrid_flags)}]" if hybrid_flags else ""
            print(
                f"  Event {eid}: {event.num_timesteps} steps, "
                f"pred 1D={final_preds['pred_1d'].shape} 2D={final_preds['pred_2d'].shape}, "
                f"{elapsed:.1f}s{suffix}"
            )
            all_predictions.append((model_id, eid, final_preds))

    # Build submission
    print("\nBuilding submission...")
    t0 = time.time()
    df_sub = build_submission(all_predictions)
    print(f"  Shape: {df_sub.shape}, built in {time.time()-t0:.1f}s")

    output_path = os.path.join(os.path.dirname(args.data_dir), args.output)
    if args.output.endswith(".parquet"):
        df_sub.to_parquet(output_path, index=False)
    else:
        df_sub.to_csv(output_path, index=False)

    # CSV同時出力は無効化 (parquetのみ)

    print(f"  Saved to {output_path}")
    print(f"  Total rows: {len(df_sub):,}")


if __name__ == "__main__":
    main()
