# #####
# data.py
# #####

# urbanflood/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch


Split = Literal["train", "test"]


@dataclass(frozen=True)
class GraphData:
    model_id: int
    n_1d: int
    n_2d: int
    n_total: int

    node_static_1d: torch.Tensor  # [N1, F1]
    node_static_2d: torch.Tensor  # [N2, F2]

    edge_index_1d: torch.LongTensor  # [2, E1]
    edge_attr_1d: torch.Tensor  # [E1, F1e]
    edge_index_2d: torch.LongTensor  # [2, E2]
    edge_attr_2d: torch.Tensor  # [E2, F2e]
    edge_index_c: torch.LongTensor  # [2, Ec]
    edge_attr_c: torch.Tensor  # [Ec, Fce]

    # 2D cell areas (for rain->volume feature), raw units (ft^2)
    area_2d: torch.Tensor  # [N2]

    # 1D->2D connectivity for rainfall proxy
    conn_src_1d: torch.LongTensor  # [Nc]
    conn_dst_2d: torch.LongTensor  # [Nc]
    conn_count_1d: torch.Tensor  # [N1], float

    # Optional static offset to convert depth->head. Raw units (ft).
    # Convention: [invert_elevation for 1D nodes, bed elevation for 2D nodes]
    head_offset: torch.Tensor  # [N_total]


@dataclass(frozen=True)
class EventData:
    model_id: int
    event_id: int
    split: Split
    timesteps: int  # T

    y_1d: torch.Tensor  # [T, N1] (NaNs in test after warmup)
    y_2d: torch.Tensor  # [T, N2] (NaNs in test after warmup)
    rain_2d: torch.Tensor  # [T, N2]
    inlet_1d: torch.Tensor | None = None  # [T, N1]
    volume_2d: torch.Tensor | None = None  # [T, N2]


def _zscore(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean.squeeze(0), std.squeeze(0)


def _fill_nan_with_median(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    if x.ndim != 2:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isfinite(col)
        if mask.all():
            continue
        if mask.any():
            fill = float(np.median(col[mask]))
        else:
            fill = 0.0
        col = np.where(mask, col, fill)
        x[:, j] = col
    return x


def _read_csv_sorted(path: Path, sort_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")
    return df.reset_index(drop=True)


def _build_directed_edges_from_static(
    *,
    edge_index_df: pd.DataFrame,
    edge_static_df: pd.DataFrame,
    feature_cols: list[str],
    flip_cols: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (src, dst, edge_features) for directed edges, where the reverse edge
    is created by flipping the sign of specified columns (e.g., dx, dy, slope).
    """
    if len(edge_index_df) != len(edge_static_df):
        raise ValueError("edge_index and edge_static row counts differ")

    edge_static = edge_static_df[feature_cols].to_numpy(np.float32)
    edge_static_rev = edge_static.copy()
    for col in flip_cols:
        if col not in feature_cols:
            continue
        j = feature_cols.index(col)
        edge_static_rev[:, j] *= -1.0

    src = edge_index_df["from_node"].to_numpy(np.int64)
    dst = edge_index_df["to_node"].to_numpy(np.int64)

    src_dir = np.concatenate([src, dst], axis=0)
    dst_dir = np.concatenate([dst, src], axis=0)
    feat_dir = np.concatenate([edge_static, edge_static_rev], axis=0)
    return src_dir, dst_dir, feat_dir


def load_graph(model_root: Path, *, model_id: int, split_for_static: Split = "train") -> GraphData:
    """
    Load static graph for a given model. Static files exist in both train/ and test/,
    but are identical; `split_for_static` selects which folder to read from.
    """
    model_root = Path(model_root)
    base = model_root / f"Model_{model_id}" / split_for_static

    nodes_1d_df = _read_csv_sorted(base / "1d_nodes_static.csv", ["node_idx"])
    nodes_2d_df = _read_csv_sorted(base / "2d_nodes_static.csv", ["node_idx"])

    n_1d = int(nodes_1d_df.shape[0])
    n_2d = int(nodes_2d_df.shape[0])
    n_total = n_1d + n_2d

    node_1d_cols = [c for c in nodes_1d_df.columns if c != "node_idx"]
    node_2d_cols = [c for c in nodes_2d_df.columns if c != "node_idx"]

    node_1d = nodes_1d_df[node_1d_cols].to_numpy(np.float32)
    node_2d = nodes_2d_df[node_2d_cols].to_numpy(np.float32)
    node_1d = _fill_nan_with_median(node_1d)
    node_2d = _fill_nan_with_median(node_2d)

    if "flow_accumulation" in node_2d_cols:
        j = node_2d_cols.index("flow_accumulation")
        node_2d[:, j] = np.log1p(np.maximum(node_2d[:, j], 0.0))

    node_1d_z, _, _ = _zscore(node_1d)
    node_2d_z, _, _ = _zscore(node_2d)

    area_2d = nodes_2d_df["area"].to_numpy(np.float32) if "area" in nodes_2d_df.columns else np.ones(n_2d, np.float32)

    # Head offsets (raw, unnormalized).
    inv = nodes_1d_df["invert_elevation"].to_numpy(np.float32) if "invert_elevation" in nodes_1d_df.columns else np.zeros(n_1d, np.float32)
    if "min_elevation" in nodes_2d_df.columns:
        bed = nodes_2d_df["min_elevation"].to_numpy(np.float32)
        if "elevation" in nodes_2d_df.columns:
            elev = nodes_2d_df["elevation"].to_numpy(np.float32)
            bed = np.where(np.isfinite(bed), bed, elev)
        else:
            bed = np.nan_to_num(bed, nan=0.0, posinf=0.0, neginf=0.0)
    elif "elevation" in nodes_2d_df.columns:
        bed = nodes_2d_df["elevation"].to_numpy(np.float32)
    else:
        bed = np.zeros(n_2d, np.float32)
    inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    bed = np.nan_to_num(bed, nan=0.0, posinf=0.0, neginf=0.0)
    head_offset = np.concatenate([inv, bed], axis=0).astype(np.float32, copy=False)

    # 1D edges
    e1_idx = _read_csv_sorted(base / "1d_edge_index.csv", ["edge_idx"])
    e1_static = _read_csv_sorted(base / "1d_edges_static.csv", ["edge_idx"])
    e1_cols = [c for c in e1_static.columns if c != "edge_idx"]
    e1_src, e1_dst, e1_attr = _build_directed_edges_from_static(
        edge_index_df=e1_idx,
        edge_static_df=e1_static,
        feature_cols=e1_cols,
        flip_cols={"relative_position_x", "relative_position_y", "slope"},
    )
    e1_attr = _fill_nan_with_median(e1_attr)
    e1_attr_z, _, _ = _zscore(e1_attr)

    # 2D edges
    e2_idx = _read_csv_sorted(base / "2d_edge_index.csv", ["edge_idx"])
    e2_static = _read_csv_sorted(base / "2d_edges_static.csv", ["edge_idx"])
    e2_cols = [c for c in e2_static.columns if c != "edge_idx"]
    e2_src, e2_dst, e2_attr = _build_directed_edges_from_static(
        edge_index_df=e2_idx,
        edge_static_df=e2_static,
        feature_cols=e2_cols,
        flip_cols={"relative_position_x", "relative_position_y", "slope"},
    )
    e2_attr = _fill_nan_with_median(e2_attr)
    e2_attr_z, _, _ = _zscore(e2_attr)

    # Coupling edges + 1D rainfall proxy mapping
    conn_df = _read_csv_sorted(base / "1d2d_connections.csv", ["connection_idx"])
    if not {"node_1d", "node_2d"}.issubset(set(conn_df.columns)):
        raise ValueError("1d2d_connections.csv missing expected columns")
    conn_src_1d = conn_df["node_1d"].to_numpy(np.int64)
    conn_dst_2d = conn_df["node_2d"].to_numpy(np.int64)

    # Directed coupling edges (1d->2d_uid and reverse)
    x1 = nodes_1d_df["position_x"].to_numpy(np.float32)
    y1 = nodes_1d_df["position_y"].to_numpy(np.float32)
    x2 = nodes_2d_df["position_x"].to_numpy(np.float32)
    y2 = nodes_2d_df["position_y"].to_numpy(np.float32)

    dx_12 = x2[conn_dst_2d] - x1[conn_src_1d]
    dy_12 = y2[conn_dst_2d] - y1[conn_src_1d]
    dist = np.sqrt(dx_12**2 + dy_12**2 + 1e-12)
    a2 = area_2d[conn_dst_2d]

    c_feat_fwd = np.stack([dx_12, dy_12, dist, a2], axis=1).astype(np.float32)
    c_feat_rev = c_feat_fwd.copy()
    c_feat_rev[:, 0] *= -1.0
    c_feat_rev[:, 1] *= -1.0

    c_attr = np.concatenate([c_feat_fwd, c_feat_rev], axis=0)
    c_attr = _fill_nan_with_median(c_attr)
    c_attr_z, _, _ = _zscore(c_attr)

    c_src = np.concatenate([conn_src_1d, n_1d + conn_dst_2d], axis=0)
    c_dst = np.concatenate([n_1d + conn_dst_2d, conn_src_1d], axis=0)

    conn_count = np.bincount(conn_src_1d, minlength=n_1d).astype(np.float32)
    conn_count = np.maximum(conn_count, 1.0)

    return GraphData(
        model_id=int(model_id),
        n_1d=n_1d,
        n_2d=n_2d,
        n_total=n_total,
        node_static_1d=torch.from_numpy(node_1d_z),
        node_static_2d=torch.from_numpy(node_2d_z),
        edge_index_1d=torch.from_numpy(np.stack([e1_src, e1_dst], axis=0)).long(),
        edge_attr_1d=torch.from_numpy(e1_attr_z),
        edge_index_2d=torch.from_numpy(np.stack([e2_src, e2_dst], axis=0)).long(),
        edge_attr_2d=torch.from_numpy(e2_attr_z),
        edge_index_c=torch.from_numpy(np.stack([c_src, c_dst], axis=0)).long(),
        edge_attr_c=torch.from_numpy(c_attr_z),
        area_2d=torch.from_numpy(area_2d.astype(np.float32)),
        conn_src_1d=torch.from_numpy(conn_src_1d).long(),
        conn_dst_2d=torch.from_numpy(conn_dst_2d).long(),
        conn_count_1d=torch.from_numpy(conn_count),
        head_offset=torch.from_numpy(head_offset),
    )


def list_events(model_root: Path, *, model_id: int, split: Split) -> list[int]:
    base = Path(model_root) / f"Model_{model_id}" / split
    event_ids: list[int] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("event_"):
            continue
        try:
            event_ids.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    event_ids.sort()
    return event_ids


def _load_event_from_csvs(
    *,
    model_root: Path,
    model_id: int,
    split: Split,
    event_id: int,
    n_1d: int,
    n_2d: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = Path(model_root) / f"Model_{model_id}" / split / f"event_{event_id}"
    t_df = pd.read_csv(base / "timesteps.csv")
    t = int(t_df.shape[0])

    # 1D nodes
    df1 = pd.read_csv(
        base / "1d_nodes_dynamic_all.csv",
        usecols=["timestep", "node_idx", "water_level", "inlet_flow"],
        dtype={"timestep": np.float32, "node_idx": np.int32, "water_level": np.float32, "inlet_flow": np.float32},
    )
    if df1.shape[0] != t * n_1d:
        raise ValueError(f"unexpected 1D rows: got {df1.shape[0]} expected {t*n_1d}")
    node_idx_1 = df1["node_idx"].to_numpy(np.int64)
    if not np.array_equal(node_idx_1[:n_1d], np.arange(n_1d, dtype=np.int64)):
        # Fallback to a stable sort if ordering isn't timestep-major then node_idx.
        df1 = df1.sort_values(["timestep", "node_idx"], kind="mergesort").reset_index(drop=True)
        node_idx_1 = df1["node_idx"].to_numpy(np.int64)
    y1 = df1["water_level"].to_numpy(np.float32).reshape(t, n_1d)
    q1 = df1["inlet_flow"].to_numpy(np.float32).reshape(t, n_1d)

    # 2D nodes
    df2 = pd.read_csv(
        base / "2d_nodes_dynamic_all.csv",
        usecols=["timestep", "node_idx", "rainfall", "water_level", "water_volume"],
        dtype={"timestep": np.int32, "node_idx": np.int32, "rainfall": np.float32, "water_level": np.float32, "water_volume": np.float32},
    )
    if df2.shape[0] != t * n_2d:
        raise ValueError(f"unexpected 2D rows: got {df2.shape[0]} expected {t*n_2d}")
    node_idx_2 = df2["node_idx"].to_numpy(np.int64)
    if not np.array_equal(node_idx_2[:n_2d], np.arange(n_2d, dtype=np.int64)):
        df2 = df2.sort_values(["timestep", "node_idx"], kind="mergesort").reset_index(drop=True)
        node_idx_2 = df2["node_idx"].to_numpy(np.int64)
    rain2 = df2["rainfall"].to_numpy(np.float32).reshape(t, n_2d)
    y2 = df2["water_level"].to_numpy(np.float32).reshape(t, n_2d)
    v2 = df2["water_volume"].to_numpy(np.float32).reshape(t, n_2d)

    return t, y1, y2, rain2, q1, v2


def load_event(
    model_root: Path,
    *,
    graph: GraphData,
    split: Split,
    event_id: int,
    cache_dir: Path | None = None,
) -> EventData:
    model_root = Path(model_root)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"model_{graph.model_id}" / split / f"event_{event_id}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            z = np.load(cache_path)
            if {"t", "y1", "y2", "r2", "q1", "v2"}.issubset(set(z.files)):
                t = int(z["t"])
                y1 = z["y1"].astype(np.float32)
                y2 = z["y2"].astype(np.float32)
                r2 = z["r2"].astype(np.float32)
                q1 = z["q1"].astype(np.float32)
                v2 = z["v2"].astype(np.float32)
                return EventData(
                    model_id=graph.model_id,
                    event_id=int(event_id),
                    split=split,
                    timesteps=t,
                    y_1d=torch.from_numpy(y1),
                    y_2d=torch.from_numpy(y2),
                    rain_2d=torch.from_numpy(r2),
                    inlet_1d=torch.from_numpy(q1),
                    volume_2d=torch.from_numpy(v2),
                )

    t, y1, y2, r2, q1, v2 = _load_event_from_csvs(
        model_root=model_root,
        model_id=graph.model_id,
        split=split,
        event_id=event_id,
        n_1d=graph.n_1d,
        n_2d=graph.n_2d,
    )
    if cache_path is not None:
        np.savez(cache_path, t=np.int32(t), y1=y1, y2=y2, r2=r2, q1=q1, v2=v2)

    return EventData(
        model_id=graph.model_id,
        event_id=int(event_id),
        split=split,
        timesteps=t,
        y_1d=torch.from_numpy(y1),
        y_2d=torch.from_numpy(y2),
        rain_2d=torch.from_numpy(r2),
        inlet_1d=torch.from_numpy(q1),
        volume_2d=torch.from_numpy(v2),
    )


def split_event_ids(
    event_ids: list[int],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1)")
    rng = np.random.default_rng(int(seed))
    ids = list(event_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(val_ratio))))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def rain_proxy_1d(
    graph: GraphData,
    rain_2d_t: torch.Tensor,  # [N2]
    *,
    to_volume: bool,
) -> torch.Tensor:
    """
    Compute a 1D rainfall proxy from connected 2D rainfall for a single timestep.
    - to_volume=False: mean rainfall depth over connected 2D cells
    - to_volume=True: sum of (rain_ft * area) over connected 2D cells
    """
    if rain_2d_t.ndim != 1 or rain_2d_t.shape[0] != graph.n_2d:
        raise ValueError("rain_2d_t must have shape [N2]")

    device = rain_2d_t.device
    dtype = rain_2d_t.dtype

    conn_src = graph.conn_src_1d.to(device=device)
    conn_dst = graph.conn_dst_2d.to(device=device)

    vals = rain_2d_t.index_select(0, conn_dst)  # [Nc]
    if to_volume:
        area = graph.area_2d.to(device=device, dtype=dtype).index_select(0, conn_dst)
        vals = (vals / 12.0) * area

    out = torch.zeros(graph.n_1d, device=device, dtype=dtype)
    out.index_add_(0, conn_src, vals)
    if not to_volume:
        out = out / graph.conn_count_1d.to(device=device, dtype=dtype)
    return out
