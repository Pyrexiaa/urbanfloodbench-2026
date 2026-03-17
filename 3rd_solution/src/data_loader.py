"""データローダー: CSV → PyG HeteroData グラフ構築.

各イベントのグラフ構造:
  Node types: '1d' (drainage), '2d' (surface)
  Edge types: ('1d','pipe','1d'), ('2d','surface','2d'),
              ('1d','coupling','2d'), ('2d','coupling','1d')
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """モデル（Model_1/Model_2）の静的構成情報."""
    model_id: int
    num_1d_nodes: int
    num_2d_nodes: int
    num_1d_edges: int
    num_2d_edges: int
    num_1d2d_connections: int
    timestep_interval: int  # seconds

    # Static features (loaded once)
    nodes_1d_static: np.ndarray = field(repr=False, default=None)  # [N_1d, F_1d_static]
    nodes_2d_static: np.ndarray = field(repr=False, default=None)  # [N_2d, F_2d_static]
    edges_1d_static: np.ndarray = field(repr=False, default=None)  # [E_1d, F_1d_edge_static]
    edges_2d_static: np.ndarray = field(repr=False, default=None)  # [E_2d, F_2d_edge_static]

    # Graph connectivity
    edge_index_1d: np.ndarray = field(repr=False, default=None)    # [2, E_1d]
    edge_index_2d: np.ndarray = field(repr=False, default=None)    # [2, E_2d]
    connections_1d2d: np.ndarray = field(repr=False, default=None)  # [N_conn, 2] (node_1d, node_2d)

    # Normalization stats (computed from train)
    norm_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class EventData:
    """1イベントの全時系列データ."""
    model_id: int
    event_id: int
    num_timesteps: int

    # Dynamic features [T, N, F]
    nodes_1d_dynamic: np.ndarray = field(repr=False, default=None)  # [T, N_1d, 2] (water_level, inlet_flow)
    nodes_2d_dynamic: np.ndarray = field(repr=False, default=None)  # [T, N_2d, 3] (rainfall, water_level, water_volume)
    edges_1d_dynamic: np.ndarray = field(repr=False, default=None)  # [T, E_1d, 2] (flow, velocity)
    edges_2d_dynamic: np.ndarray = field(repr=False, default=None)  # [T, E_2d, 2] (flow, velocity)

    # Test用: rainfallのみの時系列 (spin-up後)
    rainfall_only: np.ndarray = field(repr=False, default=None)     # [T_predict, N_2d]
    spin_up_steps: int = 10


def load_model_config(data_dir: str, model_id: int, split: str = "train") -> ModelConfig:
    """モデルの静的設定と特徴量を読み込む."""
    model_dir = os.path.join(data_dir, f"Model_{model_id}", split)

    # Dataset summary
    summary = pd.read_csv(os.path.join(model_dir, "dataset_summary.csv"))
    row = summary.iloc[0]

    config = ModelConfig(
        model_id=model_id,
        num_1d_nodes=int(row["num_1d_nodes"]),
        num_2d_nodes=int(row["num_2d_nodes"]),
        num_1d_edges=int(row["num_1d_edges"]),
        num_2d_edges=int(row["num_2d_edges"]),
        num_1d2d_connections=int(row["num_1d2d_connections"]),
        timestep_interval=int(row["timestep_interval"]),
    )

    # Static node features (drop node_idx column)
    df_1d = pd.read_csv(os.path.join(model_dir, "1d_nodes_static.csv"))
    config.nodes_1d_static = df_1d.drop(columns=["node_idx"]).values.astype(np.float32)

    df_2d = pd.read_csv(os.path.join(model_dir, "2d_nodes_static.csv"))
    config.nodes_2d_static = df_2d.drop(columns=["node_idx"]).values.astype(np.float32)

    # Static edge features (drop edge_idx column)
    df_e1d = pd.read_csv(os.path.join(model_dir, "1d_edges_static.csv"))
    config.edges_1d_static = df_e1d.drop(columns=["edge_idx"]).values.astype(np.float32)

    df_e2d = pd.read_csv(os.path.join(model_dir, "2d_edges_static.csv"))
    config.edges_2d_static = df_e2d.drop(columns=["edge_idx"]).values.astype(np.float32)

    # Edge connectivity
    df_ei_1d = pd.read_csv(os.path.join(model_dir, "1d_edge_index.csv"))
    config.edge_index_1d = df_ei_1d[["from_node", "to_node"]].values.T.astype(np.int64)

    df_ei_2d = pd.read_csv(os.path.join(model_dir, "2d_edge_index.csv"))
    config.edge_index_2d = df_ei_2d[["from_node", "to_node"]].values.T.astype(np.int64)

    # 1D-2D connections
    df_conn = pd.read_csv(os.path.join(model_dir, "1d2d_connections.csv"))
    config.connections_1d2d = df_conn[["node_1d", "node_2d"]].values.astype(np.int64)

    return config


def load_event_data(
    data_dir: str,
    model_id: int,
    event_id: int,
    config: ModelConfig,
    split: str = "train",
) -> EventData:
    """1イベントの動的データを読み込み、[T, N, F]テンソルに変換."""
    event_dir = os.path.join(data_dir, f"Model_{model_id}", split, f"event_{event_id}")

    # Timesteps
    df_ts = pd.read_csv(os.path.join(event_dir, "timesteps.csv"))
    num_timesteps = len(df_ts)

    event = EventData(
        model_id=model_id,
        event_id=event_id,
        num_timesteps=num_timesteps,
    )

    # 1D nodes dynamic — ベクトル化読み込み (NaN→0で埋める)
    df = pd.read_csv(os.path.join(event_dir, "1d_nodes_dynamic_all.csv"))
    df = df.sort_values(["timestep", "node_idx"]).fillna(0.0)
    available_ts = df["timestep"].nunique()
    event.nodes_1d_dynamic = (
        df[["water_level", "inlet_flow"]]
        .values.astype(np.float32)
        .reshape(available_ts, config.num_1d_nodes, 2)
    )
    # spin-up: テストでは最初10ステップのみ有効な水位
    event.spin_up_steps = 10

    # 2D nodes dynamic — ベクトル化読み込み
    df = pd.read_csv(os.path.join(event_dir, "2d_nodes_dynamic_all.csv"))
    df = df.sort_values(["timestep", "node_idx"])
    available_ts_2d = df["timestep"].nunique()
    # rainfallは全timestepで有効、water_level/volumeはNaN可能
    rainfall_arr = (
        df["rainfall"].fillna(0.0).values.astype(np.float32)
        .reshape(available_ts_2d, config.num_2d_nodes)
    )
    wl_arr = (
        df["water_level"].fillna(0.0).values.astype(np.float32)
        .reshape(available_ts_2d, config.num_2d_nodes)
    )
    wv_arr = (
        df["water_volume"].fillna(0.0).values.astype(np.float32)
        .reshape(available_ts_2d, config.num_2d_nodes)
    )
    event.nodes_2d_dynamic = np.stack([rainfall_arr, wl_arr, wv_arr], axis=-1)

    # 全timestepのrainfallを保存 (推論時に使用)
    event.rainfall_only = rainfall_arr

    # 1D edges dynamic — ベクトル化読み込み
    df = pd.read_csv(os.path.join(event_dir, "1d_edges_dynamic_all.csv"))
    df = df.sort_values(["timestep", "edge_idx"]).fillna(0.0)
    available_ts_e1d = df["timestep"].nunique()
    event.edges_1d_dynamic = (
        df[["flow", "velocity"]]
        .values.astype(np.float32)
        .reshape(available_ts_e1d, config.num_1d_edges, 2)
    )

    # 2D edges dynamic — ベクトル化読み込み
    df = pd.read_csv(os.path.join(event_dir, "2d_edges_dynamic_all.csv"))
    df = df.sort_values(["timestep", "edge_idx"]).fillna(0.0)
    available_ts_e2d = df["timestep"].nunique()
    event.edges_2d_dynamic = (
        df[["flow", "velocity"]]
        .values.astype(np.float32)
        .reshape(available_ts_e2d, config.num_2d_edges, 2)
    )

    return event


def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score正規化. NaN/Infを0で埋める."""
    result = (arr - mean) / std
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def compute_quick_norm_stats(config: ModelConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """静的特徴量からnorm statsを素早く計算 (mean, std).

    動的特徴量は後でcompute_normalization_statsで計算可能。
    ここでは静的特徴量のみ。
    """
    stats = {}
    for name, arr in [
        ("nodes_1d_static", config.nodes_1d_static),
        ("nodes_2d_static", config.nodes_2d_static),
        ("edges_1d_static", config.edges_1d_static),
        ("edges_2d_static", config.edges_2d_static),
    ]:
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0) + 1e-8
        stats[name] = (mean.astype(np.float32), std.astype(np.float32))
    return stats


def _compute_coupled_rainfall(
    config: ModelConfig,
    event: EventData,
    t: int,
) -> np.ndarray:
    """1Dノードに接続された2Dノードの平均rainfallを計算. [N_1d, 1]"""
    conn = config.connections_1d2d  # [N_conn, 2] = (node_1d, node_2d)
    rainfall = event.nodes_2d_dynamic[t, :, 0]  # [N_2d]
    coupled_rain = np.zeros(config.num_1d_nodes, dtype=np.float32)
    counts = np.zeros(config.num_1d_nodes, dtype=np.float32)
    for node_1d, node_2d in conn:
        coupled_rain[node_1d] += rainfall[node_2d]
        counts[node_1d] += 1.0
    # 接続なしノードは0のまま
    mask = counts > 0
    coupled_rain[mask] /= counts[mask]
    return coupled_rain.reshape(-1, 1)


def _manning_fill_ratio(flow: np.ndarray, diameter: np.ndarray,
                        roughness: np.ndarray, slope: np.ndarray,
                        n_iter: int = 25) -> np.ndarray:
    """Vectorized Manning bisection: edge flow → pipe fill ratio (0~1).

    Manning's equation for circular pipe:
      Q = (1/n) * A(h) * R(h)^(2/3) * |S|^(1/2)
    where θ = 2*arccos(1 - 2h/D), A = D²/8*(θ - sinθ), P = Dθ/2, R = A/P.
    Bisection to find h given |Q|, then return h/D.
    """
    E = len(flow)
    Q = np.abs(flow)
    D = np.maximum(diameter, 1e-6)
    n = np.maximum(roughness, 1e-6)
    S = np.maximum(np.abs(slope), 1e-10)
    valid = (Q > 1e-10) & (diameter > 1e-6) & (roughness > 1e-6) & (np.abs(slope) > 1e-10)

    h_low = np.zeros(E, dtype=np.float64)
    h_high = D.astype(np.float64)

    for _ in range(n_iter):
        h = (h_low + h_high) * 0.5
        ratio = np.clip(1.0 - 2.0 * h / D, -1.0, 1.0)
        theta = 2.0 * np.arccos(ratio)
        A = (D ** 2 / 8.0) * (theta - np.sin(theta))
        P = D * theta * 0.5
        R = np.where(P > 1e-10, A / P, 0.0)
        Q_m = (1.0 / n) * A * np.power(np.maximum(R, 0.0), 2.0 / 3.0) * np.sqrt(S)
        h_low = np.where((Q_m < Q) & valid, h, h_low)
        h_high = np.where((Q_m >= Q) & valid, h, h_high)

    result = np.where(valid, (h_low + h_high) / (2.0 * D), 0.0)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _compute_manning_node_feature(config: 'ModelConfig', event: 'EventData',
                                  t: int) -> np.ndarray:
    """各1Dノードの Manning equilibrium fill ratio を計算.

    edge static cols (after drop edge_idx):
      [0]=rel_pos_x, [1]=rel_pos_y, [2]=length, [3]=diameter,
      [4]=shape, [5]=roughness, [6]=slope
    edge dynamic: [0]=flow, [1]=velocity

    Returns: [N_1d, 1] fill ratio (0~1).
    """
    es = config.edges_1d_static
    diameter = es[:, 3]
    roughness = es[:, 5]
    slope = es[:, 6]
    flow = event.edges_1d_dynamic[t, :, 0]

    fill = _manning_fill_ratio(flow, diameter, roughness, slope)  # [E]

    # Aggregate to nodes: max fill ratio of adjacent edges
    ei = config.edge_index_1d  # [2, E]
    node_fill = np.zeros(config.num_1d_nodes, dtype=np.float32)
    for e in range(ei.shape[1]):
        s, d = ei[0, e], ei[1, e]
        node_fill[s] = max(node_fill[s], fill[e])
        node_fill[d] = max(node_fill[d], fill[e])

    return node_fill.reshape(-1, 1)


def build_graph_at_timestep(
    config: ModelConfig,
    event: EventData,
    t: int,
    prev_t: Optional[int] = None,
    norm_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    inject_rainfall: bool = False,
    extra_features: bool = False,
    future_rain_steps: int = 0,
    coupling_features: bool = False,
    step_ratio: Optional[float] = None,  # v6: normalized timestep position (0~1), None=不使用
    per_node_stats: Optional[Dict[str, np.ndarray]] = None,  # per-node WL normalization stats
    prev_t2: Optional[int] = None,  # v15: t-2 timestep for 3-step temporal context
    hydraulic_gradients: bool = False,  # v15: Δdepth edge features (independent of extra_features)
    upstream_flow_feature: bool = False,  # v20: upstream inlet flow sum
    manning_features: bool = False,  # v21: Manning equilibrium fill ratio
    storage_feature: bool = False,  # v47: base_area * max(0, wl - invert_elev) for 1D nodes
    elapsed_ratio: Optional[float] = None,  # v47: step/total_steps for all nodes (None=不使用)
    hydraulic_gradient_norm: bool = False,  # v47: (wl_src - wl_dst)/edge_length for 1D edges
    skip_test_zero_vars: bool = False,  # v48: テスト時ゼロ変数(inlet_flow, water_volume, edge dynamics)を入力から除外
) -> HeteroData:
    """時刻tのスナップショットからPyG HeteroDataグラフを構築.

    Node features = [static_features, dynamic_t, (dynamic_t-1)]
    Edge features = [static_features, dynamic_t]
    norm_stats: 事前計算された正規化パラメータ (mean, std)
    inject_rainfall: Trueなら1Dノードに接続2Dの降雨情報を追加 (1D: 10→12)
    coupling_features: Trueなら1D-2D coupling edgeに水頭差を追加 (edge_dim=2)
    extra_features: Trueなら累積降雨量+デルタWL追加 (1D: 10→14, 2D: 15→19)
    future_rain_steps: >0なら未来k stepの降雨統計を追加 (1D: +3, 2D: +3)
      テストで全降雨が既知なので合法的に使用可能。
    per_node_stats: per-node WL正規化統計量 (compute_per_node_dynamic_stats()の出力)。
      指定時、水位(WL)列のみper-node Z-scoreで正規化し、他の動的特徴はグローバル正規化を維持。
    """
    data = HeteroData()

    # 正規化関数
    def norm(arr: np.ndarray, key: str) -> np.ndarray:
        if norm_stats is not None and key in norm_stats:
            mean, std = norm_stats[key]
            return _normalize(arr, mean, std)
        return np.nan_to_num(arr, nan=0.0)

    prev_t_val = prev_t if (prev_t is not None and prev_t >= 0) else t

    # --- 1D Nodes ---
    static_1d = norm(config.nodes_1d_static, "nodes_1d_static")  # [N_1d, 6]
    dyn_1d_t = event.nodes_1d_dynamic[t]  # [N_1d, 2] = [water_level, inlet_flow]
    dyn_1d_prev = event.nodes_1d_dynamic[prev_t_val]
    # Per-node WL normalization: WL列のみper-node、他はグローバル
    if per_node_stats is not None:
        pn_mean_1d = per_node_stats["1d_wl_mean"]  # [N_1d]
        pn_std_1d = per_node_stats["1d_wl_std"]    # [N_1d]
        # WL: per-node Z-score
        wl_t_norm = ((dyn_1d_t[:, 0] - pn_mean_1d) / pn_std_1d).reshape(-1, 1)
        wl_prev_norm = ((dyn_1d_prev[:, 0] - pn_mean_1d) / pn_std_1d).reshape(-1, 1)
        if skip_test_zero_vars:
            # v48: inlet_flowを除外 (テスト時ゼロになる変数)
            dyn_1d_t_norm = wl_t_norm
            dyn_1d_prev_norm = wl_prev_norm
        else:
            # inlet_flow: グローバル正規化 (norm_stats使用)
            if norm_stats is not None and "nodes_1d_dynamic" in norm_stats:
                gm, gs = norm_stats["nodes_1d_dynamic"]
                inlet_t_norm = ((dyn_1d_t[:, 1] - gm[1]) / gs[1]).reshape(-1, 1)
                inlet_prev_norm = ((dyn_1d_prev[:, 1] - gm[1]) / gs[1]).reshape(-1, 1)
            else:
                inlet_t_norm = dyn_1d_t[:, 1:2]
                inlet_prev_norm = dyn_1d_prev[:, 1:2]
            dyn_1d_t_norm = np.concatenate([wl_t_norm, inlet_t_norm], axis=1)
            dyn_1d_prev_norm = np.concatenate([wl_prev_norm, inlet_prev_norm], axis=1)
    else:
        dyn_1d_t_norm = norm(dyn_1d_t, "nodes_1d_dynamic")
        dyn_1d_prev_norm = norm(dyn_1d_prev, "nodes_1d_dynamic")
    features_1d = [static_1d, dyn_1d_t_norm, dyn_1d_prev_norm]

    # Rainfall injection: 1Dノードに接続2Dノードの降雨を追加
    if inject_rainfall:
        rain_t = _compute_coupled_rainfall(config, event, t)
        rain_prev_t = t - 1 if prev_t is None else prev_t
        rain_prev = _compute_coupled_rainfall(config, event, max(rain_prev_t, 0))
        features_1d.append(rain_t)       # [N_1d, 1]
        features_1d.append(rain_prev)    # [N_1d, 1]

    # Extra features: 累積降雨量 + デルタWL (テスト条件で有効)
    if extra_features:
        # デルタWL: wl_t - wl_{t-1} (速度情報, テスト条件でも利用可)
        delta_wl_1d = (dyn_1d_t[:, 0] - dyn_1d_prev[:, 0]).reshape(-1, 1)
        features_1d.append(delta_wl_1d)  # [N_1d, 1]

        # 累積降雨量: 接続2Dノードの降雨合計 (テスト条件で利用可能な唯一のforcing)
        cumul_rain_2d = np.sum(event.nodes_2d_dynamic[:t + 1, :, 0], axis=0)  # [N_2d]
        # 1Dノード: 接続2Dノードの平均累積降雨量
        conn = config.connections_1d2d
        cumul_rain_1d = np.zeros(config.num_1d_nodes, dtype=np.float32)
        counts = np.zeros(config.num_1d_nodes, dtype=np.float32)
        for n1d, n2d in conn:
            cumul_rain_1d[n1d] += cumul_rain_2d[n2d]
            counts[n1d] += 1.0
        mask = counts > 0
        cumul_rain_1d[mask] /= counts[mask]
        # スケーリング: 100mm程度で正規化 (典型的なイベント総降雨量)
        cumul_rain_1d_scaled = (cumul_rain_1d / 50.0).reshape(-1, 1)
        features_1d.append(cumul_rain_1d_scaled)  # [N_1d, 1]

        # 現在の降雨量 (inject_rainfallと同様だが別実装)
        rain_now = _compute_coupled_rainfall(config, event, t)  # [N_1d, 1]
        features_1d.append(rain_now)

        # タイムステップ正規化値 (rollout位置情報)
        t_feat = np.full((config.num_1d_nodes, 1), t / 400.0, dtype=np.float32)
        features_1d.append(t_feat)  # [N_1d, 1]

    # Future rainfall features: テストで全降雨既知なので合法
    if future_rain_steps > 0:
        T_max = event.nodes_2d_dynamic.shape[0]
        rain_all = event.nodes_2d_dynamic[:, :, 0]  # [T, N_2d]
        t_end = min(t + future_rain_steps, T_max)
        future_rain = rain_all[t + 1:t_end + 1] if t + 1 < T_max else np.zeros((0, rain_all.shape[1]))
        remaining_rain = rain_all[t + 1:] if t + 1 < T_max else np.zeros((0, rain_all.shape[1]))

        # 2D: future max, future sum, remaining sum (ノード別)
        if len(future_rain) > 0:
            fr_max_2d = future_rain.max(axis=0).reshape(-1, 1)
            fr_sum_2d = future_rain.sum(axis=0).reshape(-1, 1) / max(future_rain_steps, 1)
        else:
            fr_max_2d = np.zeros((rain_all.shape[1], 1), dtype=np.float32)
            fr_sum_2d = np.zeros((rain_all.shape[1], 1), dtype=np.float32)
        rem_sum_2d = (remaining_rain.sum(axis=0).reshape(-1, 1) / 50.0 if len(remaining_rain) > 0
                      else np.zeros((rain_all.shape[1], 1), dtype=np.float32))

        # 1D: 接続2Dノードの平均
        conn = config.connections_1d2d
        fr_max_1d = np.zeros((config.num_1d_nodes, 1), dtype=np.float32)
        fr_sum_1d = np.zeros((config.num_1d_nodes, 1), dtype=np.float32)
        rem_sum_1d = np.zeros((config.num_1d_nodes, 1), dtype=np.float32)
        for n1d, n2d in conn:
            fr_max_1d[n1d] = fr_max_2d[n2d]
            fr_sum_1d[n1d] = fr_sum_2d[n2d]
            rem_sum_1d[n1d] = rem_sum_2d[n2d]

        features_1d.extend([fr_max_1d, fr_sum_1d, rem_sum_1d])

    # v20: upstream inlet flow sum (explicit flag)
    if upstream_flow_feature:
        inlet_t = event.nodes_1d_dynamic[t, :, 1]
        inv_elev = config.nodes_1d_static[:, 3]
        ei_1d = config.edge_index_1d
        upstream_sum = np.zeros(config.num_1d_nodes, dtype=np.float32)
        for e in range(ei_1d.shape[1]):
            s, d = ei_1d[0, e], ei_1d[1, e]
            if inv_elev[s] > inv_elev[d]:
                upstream_sum[d] += inlet_t[s]
            elif inv_elev[d] > inv_elev[s]:
                upstream_sum[s] += inlet_t[d]
            else:
                upstream_sum[d] += inlet_t[s]
                upstream_sum[s] += inlet_t[d]
        if norm_stats is not None and "nodes_1d_dynamic" in norm_stats:
            _, gs = norm_stats["nodes_1d_dynamic"]
            upstream_sum_norm = (upstream_sum / max(gs[1], 1e-8)).reshape(-1, 1)
        else:
            upstream_sum_norm = upstream_sum.reshape(-1, 1)
        features_1d.append(upstream_sum_norm)

    # v21: Manning equilibrium fill ratio — 物理的均衡水深
    # edge flow + pipe geometry (diameter, roughness, slope) → Manning逆算 → fill ratio
    if manning_features:
        manning_fill = _compute_manning_node_feature(config, event, t)  # [N_1d, 1]
        features_1d.append(manning_fill)

    # v47: storage = base_area * max(0, wl - invert_elevation) — 物理的貯水量
    # テスト時も予測WLから計算可能。static col: [3]=invert_elevation, [5]=base_area (after drop node_idx)
    if storage_feature:
        wl_raw = event.nodes_1d_dynamic[t, :, 0]  # [N_1d]
        inv_elev = config.nodes_1d_static[:, 3]     # invert_elevation
        base_area = config.nodes_1d_static[:, 5]    # base_area
        water_depth = np.maximum(wl_raw - inv_elev, 0.0)
        storage = (base_area * water_depth).astype(np.float32)
        # log(1+storage) で正規化 (典型値 0~1000 → 0~7 のオーダー)
        storage_norm = np.log1p(storage).reshape(-1, 1) / 5.0
        features_1d.append(storage_norm)

    # v47: elapsed_ratio — 降雨パターン位相情報
    if elapsed_ratio is not None:
        er_1d = np.full((config.num_1d_nodes, 1), elapsed_ratio, dtype=np.float32)
        features_1d.append(er_1d)

    # v15: t-2 dynamic features (3-step temporal context, GNN-SWS)
    # Appended AFTER future_rain so first 13 dims match v11b for weight transfer
    if prev_t2 is not None:
        prev_t2_val = max(0, prev_t2)
        dyn_1d_prev2 = event.nodes_1d_dynamic[prev_t2_val]
        if per_node_stats is not None:
            wl_prev2_norm = ((dyn_1d_prev2[:, 0] - per_node_stats["1d_wl_mean"])
                             / per_node_stats["1d_wl_std"]).reshape(-1, 1)
            if norm_stats is not None and "nodes_1d_dynamic" in norm_stats:
                gm, gs = norm_stats["nodes_1d_dynamic"]
                inlet_prev2_norm = ((dyn_1d_prev2[:, 1] - gm[1]) / gs[1]).reshape(-1, 1)
            else:
                inlet_prev2_norm = dyn_1d_prev2[:, 1:2]
            dyn_1d_prev2_norm = np.concatenate([wl_prev2_norm, inlet_prev2_norm], axis=1)
        else:
            dyn_1d_prev2_norm = norm(dyn_1d_prev2, "nodes_1d_dynamic")
        features_1d.append(dyn_1d_prev2_norm)

    # v6: step_ratio (normalized timestep position) for GRU temporal encoding
    if step_ratio is not None:
        sr_1d = np.full((config.num_1d_nodes, 1), step_ratio, dtype=np.float32)
        features_1d.append(sr_1d)

    x_1d = np.concatenate(features_1d, axis=1)
    data["1d"].x = torch.tensor(x_1d, dtype=torch.float32)

    # --- 2D Nodes ---
    static_2d = norm(config.nodes_2d_static, "nodes_2d_static")  # [N_2d, 9]
    dyn_2d_t = event.nodes_2d_dynamic[t]  # [N_2d, 3] = [rainfall, water_level, water_volume]
    dyn_2d_prev = event.nodes_2d_dynamic[prev_t_val]
    # Per-node WL normalization: WL列のみper-node、rainfall/volumeはグローバル
    if per_node_stats is not None:
        pn_mean_2d = per_node_stats["2d_wl_mean"]  # [N_2d]
        pn_std_2d = per_node_stats["2d_wl_std"]    # [N_2d]
        if norm_stats is not None and "nodes_2d_dynamic" in norm_stats:
            gm, gs = norm_stats["nodes_2d_dynamic"]
            # rainfall: グローバル
            rain_t_norm = ((dyn_2d_t[:, 0] - gm[0]) / gs[0]).reshape(-1, 1)
            rain_prev_norm = ((dyn_2d_prev[:, 0] - gm[0]) / gs[0]).reshape(-1, 1)
        else:
            rain_t_norm = dyn_2d_t[:, 0:1]
            rain_prev_norm = dyn_2d_prev[:, 0:1]
        # WL: per-node
        wl_t_norm = ((dyn_2d_t[:, 1] - pn_mean_2d) / pn_std_2d).reshape(-1, 1)
        wl_prev_norm = ((dyn_2d_prev[:, 1] - pn_mean_2d) / pn_std_2d).reshape(-1, 1)
        if skip_test_zero_vars:
            # v48: water_volumeを除外 (テスト時ゼロになる変数)
            dyn_2d_t_norm = np.concatenate([rain_t_norm, wl_t_norm], axis=1)
            dyn_2d_prev_norm = np.concatenate([rain_prev_norm, wl_prev_norm], axis=1)
        else:
            if norm_stats is not None and "nodes_2d_dynamic" in norm_stats:
                gm, gs = norm_stats["nodes_2d_dynamic"]
                vol_t_norm = ((dyn_2d_t[:, 2] - gm[2]) / gs[2]).reshape(-1, 1)
                vol_prev_norm = ((dyn_2d_prev[:, 2] - gm[2]) / gs[2]).reshape(-1, 1)
            else:
                vol_t_norm = dyn_2d_t[:, 2:3]
                vol_prev_norm = dyn_2d_prev[:, 2:3]
            dyn_2d_t_norm = np.concatenate([rain_t_norm, wl_t_norm, vol_t_norm], axis=1)
            dyn_2d_prev_norm = np.concatenate([rain_prev_norm, wl_prev_norm, vol_prev_norm], axis=1)
    else:
        dyn_2d_t_norm = norm(dyn_2d_t, "nodes_2d_dynamic")
        dyn_2d_prev_norm = norm(dyn_2d_prev, "nodes_2d_dynamic")
    features_2d = [static_2d, dyn_2d_t_norm, dyn_2d_prev_norm]

    # Extra features for 2D
    if extra_features:
        # デルタWL
        delta_wl_2d = (dyn_2d_t[:, 1] - dyn_2d_prev[:, 1]).reshape(-1, 1)
        features_2d.append(delta_wl_2d)  # [N_2d, 1]

        # 累積降雨量
        cumul_rain_2d_arr = np.sum(event.nodes_2d_dynamic[:t + 1, :, 0], axis=0)
        cumul_rain_2d_scaled = (cumul_rain_2d_arr / 50.0).reshape(-1, 1)
        features_2d.append(cumul_rain_2d_scaled)  # [N_2d, 1]

        # デルタ降雨量 (降雨の変化率)
        delta_rain = (dyn_2d_t[:, 0] - dyn_2d_prev[:, 0]).reshape(-1, 1)
        features_2d.append(delta_rain)  # [N_2d, 1]

        # タイムステップ正規化値
        t_feat = np.full((config.num_2d_nodes, 1), t / 400.0, dtype=np.float32)
        features_2d.append(t_feat)  # [N_2d, 1]

    # Future rainfall for 2D nodes
    if future_rain_steps > 0:
        features_2d.extend([fr_max_2d, fr_sum_2d, rem_sum_2d])

    # v47: elapsed_ratio for 2D nodes
    if elapsed_ratio is not None:
        er_2d = np.full((config.num_2d_nodes, 1), elapsed_ratio, dtype=np.float32)
        features_2d.append(er_2d)

    # v15: t-2 dynamic features for 2D (3-step temporal context)
    if prev_t2 is not None:
        prev_t2_val = max(0, prev_t2)
        dyn_2d_prev2 = event.nodes_2d_dynamic[prev_t2_val]
        if per_node_stats is not None:
            if norm_stats is not None and "nodes_2d_dynamic" in norm_stats:
                gm, gs = norm_stats["nodes_2d_dynamic"]
                rain_prev2_norm = ((dyn_2d_prev2[:, 0] - gm[0]) / gs[0]).reshape(-1, 1)
                vol_prev2_norm = ((dyn_2d_prev2[:, 2] - gm[2]) / gs[2]).reshape(-1, 1)
            else:
                rain_prev2_norm = dyn_2d_prev2[:, 0:1]
                vol_prev2_norm = dyn_2d_prev2[:, 2:3]
            wl_prev2_norm = ((dyn_2d_prev2[:, 1] - per_node_stats["2d_wl_mean"])
                             / per_node_stats["2d_wl_std"]).reshape(-1, 1)
            dyn_2d_prev2_norm = np.concatenate([rain_prev2_norm, wl_prev2_norm, vol_prev2_norm], axis=1)
        else:
            dyn_2d_prev2_norm = norm(dyn_2d_prev2, "nodes_2d_dynamic")
        features_2d.append(dyn_2d_prev2_norm)

    # v6: step_ratio for 2D nodes
    if step_ratio is not None:
        sr_2d = np.full((config.num_2d_nodes, 1), step_ratio, dtype=np.float32)
        features_2d.append(sr_2d)

    x_2d = np.concatenate(features_2d, axis=1)
    data["2d"].x = torch.tensor(x_2d, dtype=torch.float32)

    # --- 1D Edges (pipe) ---
    data["1d", "pipe", "1d"].edge_index = torch.tensor(config.edge_index_1d, dtype=torch.long)
    static_e1d = norm(config.edges_1d_static, "edges_1d_static")  # [E_1d, 7]
    if skip_test_zero_vars:
        # v48: edge dynamics (flow, velocity) を除外
        edge_parts_1d = [static_e1d]
    else:
        if t < event.edges_1d_dynamic.shape[0]:
            dyn_e1d = norm(event.edges_1d_dynamic[t], "edges_1d_dynamic")
        else:
            dyn_e1d = np.zeros((config.num_1d_edges, 2), dtype=np.float32)
        edge_parts_1d = [static_e1d, dyn_e1d]

    # 水力勾配エッジ特徴: wl_source - wl_target (テスト条件でも利用可)
    if extra_features:
        src_1d, tgt_1d = config.edge_index_1d[0], config.edge_index_1d[1]
        wl_1d_t = event.nodes_1d_dynamic[t, :, 0]  # [N_1d] current WL
        wl_1d_prev = event.nodes_1d_dynamic[prev_t_val, :, 0]
        hgrad_1d = (wl_1d_t[src_1d] - wl_1d_t[tgt_1d]).reshape(-1, 1)  # [E_1d, 1]
        hgrad_1d_prev = (wl_1d_prev[src_1d] - wl_1d_prev[tgt_1d]).reshape(-1, 1)
        edge_parts_1d.append(hgrad_1d)
        edge_parts_1d.append(hgrad_1d_prev)

    # v15: hydraulic gradients for pipe edges (standalone, not overlapping with extra_features)
    if hydraulic_gradients and not extra_features:
        src_1d, tgt_1d = config.edge_index_1d[0], config.edge_index_1d[1]
        wl_1d_t = event.nodes_1d_dynamic[t, :, 0]
        wl_1d_prev = event.nodes_1d_dynamic[prev_t_val, :, 0]
        hgrad_1d = (wl_1d_t[src_1d] - wl_1d_t[tgt_1d]).reshape(-1, 1)
        hgrad_1d_prev = (wl_1d_prev[src_1d] - wl_1d_prev[tgt_1d]).reshape(-1, 1)
        edge_parts_1d.append(hgrad_1d)
        edge_parts_1d.append(hgrad_1d_prev)

    # v47: normalized hydraulic gradient = (wl_src - wl_dst) / edge_length
    # パイプ内流れ方向と強度の代理変数。テスト時も予測WLから計算可能。
    # edge_static col: [2]=length (after drop edge_idx)
    if hydraulic_gradient_norm:
        src_1d, tgt_1d = config.edge_index_1d[0], config.edge_index_1d[1]
        wl_1d_t = event.nodes_1d_dynamic[t, :, 0]
        edge_length = config.edges_1d_static[:, 2]  # length
        wl_diff = wl_1d_t[src_1d] - wl_1d_t[tgt_1d]
        hgrad_norm = (wl_diff / np.maximum(edge_length, 1.0)).astype(np.float32)
        # 典型値 ~0.001-0.01 → ×100 でスケーリング
        hgrad_norm_scaled = (hgrad_norm * 100.0).reshape(-1, 1)
        edge_parts_1d.append(hgrad_norm_scaled)

    edge_attr_1d = np.concatenate(edge_parts_1d, axis=1)
    data["1d", "pipe", "1d"].edge_attr = torch.tensor(edge_attr_1d, dtype=torch.float32)

    # --- 2D Edges (surface) ---
    data["2d", "surface", "2d"].edge_index = torch.tensor(config.edge_index_2d, dtype=torch.long)
    static_e2d = norm(config.edges_2d_static, "edges_2d_static")  # [E_2d, 5]
    if skip_test_zero_vars:
        # v48: edge dynamics (flow, velocity) を除外
        edge_parts_2d = [static_e2d]
    else:
        if t < event.edges_2d_dynamic.shape[0]:
            dyn_e2d = norm(event.edges_2d_dynamic[t], "edges_2d_dynamic")
        else:
            dyn_e2d = np.zeros((config.num_2d_edges, 2), dtype=np.float32)
        edge_parts_2d = [static_e2d, dyn_e2d]

    # 水力勾配エッジ特徴: wl_source - wl_target
    if extra_features:
        src_2d, tgt_2d = config.edge_index_2d[0], config.edge_index_2d[1]
        wl_2d_t = event.nodes_2d_dynamic[t, :, 1]
        wl_2d_prev = event.nodes_2d_dynamic[prev_t_val, :, 1]
        hgrad_2d = (wl_2d_t[src_2d] - wl_2d_t[tgt_2d]).reshape(-1, 1)
        hgrad_2d_prev = (wl_2d_prev[src_2d] - wl_2d_prev[tgt_2d]).reshape(-1, 1)
        edge_parts_2d.append(hgrad_2d)
        edge_parts_2d.append(hgrad_2d_prev)

    # v15: hydraulic gradients for surface edges (standalone)
    if hydraulic_gradients and not extra_features:
        src_2d, tgt_2d = config.edge_index_2d[0], config.edge_index_2d[1]
        wl_2d_t = event.nodes_2d_dynamic[t, :, 1]
        wl_2d_prev = event.nodes_2d_dynamic[prev_t_val, :, 1]
        hgrad_2d = (wl_2d_t[src_2d] - wl_2d_t[tgt_2d]).reshape(-1, 1)
        hgrad_2d_prev = (wl_2d_prev[src_2d] - wl_2d_prev[tgt_2d]).reshape(-1, 1)
        edge_parts_2d.append(hgrad_2d)
        edge_parts_2d.append(hgrad_2d_prev)

    edge_attr_2d = np.concatenate(edge_parts_2d, axis=1)
    data["2d", "surface", "2d"].edge_attr = torch.tensor(edge_attr_2d, dtype=torch.float32)

    # --- 1D-2D Coupling edges (bidirectional) ---
    conn = config.connections_1d2d  # [N_conn, 2] = (node_1d, node_2d)
    # 1D → 2D
    data["1d", "coupling", "2d"].edge_index = torch.tensor(
        conn.T, dtype=torch.long
    )
    # 2D → 1D (reverse)
    data["2d", "coupling", "1d"].edge_index = torch.tensor(
        conn[:, [1, 0]].T, dtype=torch.long
    )

    # Coupling edge features: 水頭差 (hydraulic head gradient)
    if coupling_features:
        wl_1d_t = event.nodes_1d_dynamic[t, :, 0]        # [N_1d]
        wl_2d_t = event.nodes_2d_dynamic[t, :, 1]        # [N_2d]
        wl_1d_prev = event.nodes_1d_dynamic[prev_t_val, :, 0]
        wl_2d_prev = event.nodes_2d_dynamic[prev_t_val, :, 1]
        node_1d_ids = conn[:, 0]  # [N_conn]
        node_2d_ids = conn[:, 1]  # [N_conn]

        # 1D→2D: 水頭差 = wl_1d - wl_2d (正=1Dの方が高い=排水方向)
        wl_diff_12 = (wl_1d_t[node_1d_ids] - wl_2d_t[node_2d_ids]).reshape(-1, 1)
        wl_diff_12_prev = (wl_1d_prev[node_1d_ids] - wl_2d_prev[node_2d_ids]).reshape(-1, 1)
        data["1d", "coupling", "2d"].edge_attr = torch.tensor(
            np.concatenate([wl_diff_12, wl_diff_12_prev], axis=1), dtype=torch.float32
        )

        # 2D→1D: 水頭差 = wl_2d - wl_1d (符号反転)
        wl_diff_21 = (wl_2d_t[node_2d_ids] - wl_1d_t[node_1d_ids]).reshape(-1, 1)
        wl_diff_21_prev = (wl_2d_prev[node_2d_ids] - wl_1d_prev[node_1d_ids]).reshape(-1, 1)
        data["2d", "coupling", "1d"].edge_attr = torch.tensor(
            np.concatenate([wl_diff_21, wl_diff_21_prev], axis=1), dtype=torch.float32
        )

    return data


def compute_normalization_stats(
    data_dir: str,
    model_id: int,
    config: ModelConfig,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """全学習イベントから正規化統計量 (mean, std) を計算.

    Returns dict with keys like 'nodes_1d_static', 'nodes_1d_dynamic', etc.
    """
    model_dir = os.path.join(data_dir, f"Model_{model_id}", "train")
    events = sorted([d for d in os.listdir(model_dir) if d.startswith("event_")])

    # Static features: 全イベントで同じなので直接計算
    stats = {}
    stats["nodes_1d_static"] = (
        np.mean(config.nodes_1d_static, axis=0),
        np.std(config.nodes_1d_static, axis=0) + 1e-8,
    )
    stats["nodes_2d_static"] = (
        np.mean(config.nodes_2d_static, axis=0),
        np.std(config.nodes_2d_static, axis=0) + 1e-8,
    )
    stats["edges_1d_static"] = (
        np.mean(config.edges_1d_static, axis=0),
        np.std(config.edges_1d_static, axis=0) + 1e-8,
    )
    stats["edges_2d_static"] = (
        np.mean(config.edges_2d_static, axis=0),
        np.std(config.edges_2d_static, axis=0) + 1e-8,
    )

    # Dynamic features: 全イベントを集約
    all_1d_dyn = []
    all_2d_dyn = []
    all_e1d_dyn = []
    all_e2d_dyn = []

    for event_name in events:
        eid = int(event_name.split("_")[1])
        ev = load_event_data(data_dir, model_id, eid, config, split="train")
        all_1d_dyn.append(ev.nodes_1d_dynamic.reshape(-1, 2))
        all_2d_dyn.append(ev.nodes_2d_dynamic.reshape(-1, 3))
        all_e1d_dyn.append(ev.edges_1d_dynamic.reshape(-1, 2))
        all_e2d_dyn.append(ev.edges_2d_dynamic.reshape(-1, 2))

    cat_1d = np.concatenate(all_1d_dyn, axis=0)
    stats["nodes_1d_dynamic"] = (np.mean(cat_1d, axis=0), np.std(cat_1d, axis=0) + 1e-8)

    cat_2d = np.concatenate(all_2d_dyn, axis=0)
    stats["nodes_2d_dynamic"] = (np.mean(cat_2d, axis=0), np.std(cat_2d, axis=0) + 1e-8)

    cat_e1d = np.concatenate(all_e1d_dyn, axis=0)
    stats["edges_1d_dynamic"] = (np.mean(cat_e1d, axis=0), np.std(cat_e1d, axis=0) + 1e-8)

    cat_e2d = np.concatenate(all_e2d_dyn, axis=0)
    stats["edges_2d_dynamic"] = (np.mean(cat_e2d, axis=0), np.std(cat_e2d, axis=0) + 1e-8)

    return stats


def compute_per_node_dynamic_stats(
    events: list,
    config: "ModelConfig",
    min_std: float = 0.01,
) -> Dict[str, np.ndarray]:
    """全訓練イベントからper-nodeのmean/stdを計算.

    per-node正規化: 各ノードの全イベント×全タイムステップでの統計量。
    グローバル正規化と比べて、ノードごとの変動幅に適応した正規化が可能。
    信号対ノイズ比が劇的に改善する (delta予測精度に直結)。

    Args:
        events: EventDataのリスト (訓練データ)
        config: ModelConfig
        min_std: std下限 (変動しないノードの0除算防止)

    Returns:
        dict with keys:
          "1d_wl_mean": [N_1d]  1D水位ノード別平均
          "1d_wl_std":  [N_1d]  1D水位ノード別std (min_std保証)
          "2d_wl_mean": [N_2d]  2D水位ノード別平均
          "2d_wl_std":  [N_2d]  2D水位ノード別std (min_std保証)
    """
    # 1D water_level: axis=0 of dynamic dim=0
    all_1d_wl = []  # list of [T, N_1d]
    all_2d_wl = []  # list of [T, N_2d]
    for ev in events:
        all_1d_wl.append(ev.nodes_1d_dynamic[:, :, 0])  # water_level
        all_2d_wl.append(ev.nodes_2d_dynamic[:, :, 1])   # water_level

    # Concat along time axis: [total_T, N]
    cat_1d = np.concatenate(all_1d_wl, axis=0)  # [total_T, N_1d]
    cat_2d = np.concatenate(all_2d_wl, axis=0)  # [total_T, N_2d]

    return {
        "1d_wl_mean": np.mean(cat_1d, axis=0).astype(np.float32),
        "1d_wl_std": np.maximum(np.std(cat_1d, axis=0), min_std).astype(np.float32),
        "2d_wl_mean": np.mean(cat_2d, axis=0).astype(np.float32),
        "2d_wl_std": np.maximum(np.std(cat_2d, axis=0), min_std).astype(np.float32),
    }


def list_events(data_dir: str, model_id: int, split: str = "train") -> List[int]:
    """利用可能なイベントIDのリストを返す."""
    model_dir = os.path.join(data_dir, f"Model_{model_id}", split)
    events = [d for d in os.listdir(model_dir) if d.startswith("event_")]
    return sorted([int(e.split("_")[1]) for e in events])
