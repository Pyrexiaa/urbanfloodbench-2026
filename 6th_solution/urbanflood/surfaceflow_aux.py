from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urbanflood.baseline import (
    RegimeARKXConfig,
    aggregate_2d_to_1d_mean,
    fit_regime_arkx_exo_per_node,
    rollout_regime_arkx_exo,
)


@dataclass(frozen=True)
class SurfaceFlowAuxCfg:
    model_id: int = 2
    warmup: int = 10
    k: int = 10
    ridge: float = 1e-3
    bins: tuple[float, ...] = (0.0, 0.03, 0.05)
    equalize_events: bool = True


def load_edge_flow_2d(
    *,
    model_root: Path,
    model_id: int,
    split: str,
    event_id: int,
    n_edges: int,
    cache_dir: Path | None = None,
) -> np.ndarray:
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"model_{int(model_id)}" / str(split) / f"event_{int(event_id)}_edgeflow2d.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            z = np.load(cache_path)
            if "q" in z.files:
                q = z["q"].astype(np.float32)
                if q.ndim == 2 and q.shape[1] == int(n_edges):
                    return q

    base = Path(model_root) / f"Model_{int(model_id)}" / str(split) / f"event_{int(event_id)}"
    t_df = pd.read_csv(base / "timesteps.csv")
    t = int(t_df.shape[0])
    df = pd.read_csv(
        base / "2d_edges_dynamic_all.csv",
        usecols=["timestep", "edge_idx", "flow"],
        dtype={"timestep": np.int32, "edge_idx": np.int32, "flow": np.float32},
    )
    if int(df.shape[0]) != int(t * n_edges):
        raise ValueError(f"unexpected 2D edge rows: got {df.shape[0]} expected {t*n_edges}")
    edge_idx = df["edge_idx"].to_numpy(np.int64)
    if not np.array_equal(edge_idx[:n_edges], np.arange(n_edges, dtype=np.int64)):
        df = df.sort_values(["timestep", "edge_idx"], kind="mergesort").reset_index(drop=True)
    q = df["flow"].to_numpy(np.float32).reshape(t, n_edges)
    if cache_path is not None:
        np.savez(cache_path, q=q)
    return q


def load_surface_edge_weight(
    *,
    model_root: Path,
    model_id: int,
    split_for_static: str = "train",
) -> np.ndarray:
    base = Path(model_root) / f"Model_{int(model_id)}" / str(split_for_static)
    df = pd.read_csv(base / "2d_edges_static.csv", usecols=["edge_idx", "face_length", "length"])
    df = df.sort_values(["edge_idx"], kind="mergesort").reset_index(drop=True)
    face = df["face_length"].to_numpy(np.float32)
    length = np.maximum(df["length"].to_numpy(np.float32), 1e-3)
    return (face / length).astype(np.float32, copy=False)


def build_surface_flow_exo(
    *,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    bed_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
) -> np.ndarray:
    y2 = np.asarray(y2_2d, dtype=np.float32)
    rain = np.asarray(rain_2d, dtype=np.float32)
    bed = np.asarray(bed_2d, dtype=np.float32)
    if y2.ndim != 2 or rain.ndim != 2:
        raise ValueError("y2_2d and rain_2d must be [T, N2]")
    if y2.shape != rain.shape:
        raise ValueError("y2_2d and rain_2d must align")
    if bed.shape != (int(y2.shape[1]),):
        raise ValueError("bed_2d must be [N2]")

    head = (y2 + bed[None, :]).astype(np.float32, copy=False)
    head_from = head[:, edge_from]
    head_to = head[:, edge_to]
    dhead = (head_from - head_to).astype(np.float32, copy=False)
    dhead_prev = np.empty_like(dhead)
    dhead_prev[0] = dhead[0]
    dhead_prev[1:] = dhead[:-1]
    ddhead = (dhead - dhead_prev).astype(np.float32, copy=False)

    depth_from = y2[:, edge_from]
    depth_to = y2[:, edge_to]
    rain_from = rain[:, edge_from]
    rain_to = rain[:, edge_to]
    exo = np.stack(
        [
            depth_from,
            depth_to,
            dhead,
            np.abs(dhead).astype(np.float32, copy=False),
            ddhead,
            rain_from,
            rain_to,
        ],
        axis=-1,
    )
    return exo.astype(np.float32, copy=False)


def predict_surface_flow_from_ckpt(
    *,
    ckpt: dict,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    bed_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    q_edge_init: np.ndarray,
    warmup: int,
) -> np.ndarray:
    if str(ckpt.get("kind", "")) != "aux_surfaceflow_m2_2d":
        raise ValueError("surfaceflow ckpt must be kind aux_surfaceflow_m2_2d")
    exo = build_surface_flow_exo(
        y2_2d=y2_2d,
        rain_2d=rain_2d,
        bed_2d=bed_2d,
        edge_from=edge_from,
        edge_to=edge_to,
    )
    max_delta_raw = ckpt.get("max_delta", None)
    if max_delta_raw is None:
        max_delta = None
    elif isinstance(max_delta_raw, np.ndarray):
        max_delta = max_delta_raw.astype(np.float32, copy=False)
    elif hasattr(max_delta_raw, "numpy"):
        max_delta = max_delta_raw.numpy().astype(np.float32, copy=False)
    else:
        max_delta = float(max_delta_raw)
    return rollout_regime_arkx_exo(
        w=ckpt["w"].numpy() if hasattr(ckpt["w"], "numpy") else np.asarray(ckpt["w"], dtype=np.float32),
        bins=tuple(float(x) for x in ckpt["bins"]),
        y_init=np.asarray(q_edge_init, dtype=np.float32),
        rain=np.asarray(rain_2d, dtype=np.float32),
        exo=exo,
        warmup=int(warmup),
        max_delta=max_delta,
    )


def aggregate_surface_edge_flows_to_nodes(
    *,
    q_edge: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    n_2d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_edge, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != int(edge_from.shape[0]) or q.shape[1] != int(edge_to.shape[0]):
        raise ValueError("q_edge must be [T, E] aligned to edge_from/edge_to")
    inflow = np.zeros((int(q.shape[0]), int(n_2d)), dtype=np.float32)
    outflow = np.zeros_like(inflow)
    for j, (u, v) in enumerate(zip(edge_from.tolist(), edge_to.tolist(), strict=True)):
        qj = q[:, j]
        pos = np.maximum(qj, 0.0).astype(np.float32, copy=False)
        neg = np.maximum(-qj, 0.0).astype(np.float32, copy=False)
        inflow[:, v] += pos
        outflow[:, u] += pos
        inflow[:, u] += neg
        outflow[:, v] += neg
    net = (inflow - outflow).astype(np.float32, copy=False)
    total = (inflow + outflow).astype(np.float32, copy=False)
    return inflow, outflow, net, total


def aggregate_surface_node_flux_to_1d(
    *,
    surface_node_flux: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
) -> np.ndarray:
    return aggregate_2d_to_1d_mean(
        surface_node_flux,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    ).astype(np.float32, copy=False)


def build_surface_flow_1d_features(
    *,
    q_edge: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    n_2d: int,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
) -> np.ndarray:
    inflow_2d, outflow_2d, net_2d, total_2d = aggregate_surface_edge_flows_to_nodes(
        q_edge=q_edge,
        edge_from=edge_from,
        edge_to=edge_to,
        n_2d=n_2d,
    )
    inflow_1d = aggregate_surface_node_flux_to_1d(
        surface_node_flux=inflow_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    outflow_1d = aggregate_surface_node_flux_to_1d(
        surface_node_flux=outflow_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    net_1d = aggregate_surface_node_flux_to_1d(
        surface_node_flux=net_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    total_1d = aggregate_surface_node_flux_to_1d(
        surface_node_flux=total_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    return np.stack([inflow_1d, outflow_1d, net_1d, total_1d], axis=-1).astype(np.float32, copy=False)


def build_coupled_surface_slot_index(
    *,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    edge_weight: np.ndarray,
    n_1d: int,
    slots_per_node: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    conn_src = np.asarray(conn_src_1d, dtype=np.int64)
    conn_dst = np.asarray(conn_dst_2d, dtype=np.int64)
    edge_from = np.asarray(edge_from, dtype=np.int64)
    edge_to = np.asarray(edge_to, dtype=np.int64)
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    n1 = int(n_1d)
    k = int(slots_per_node)
    if conn_src.ndim != 1 or conn_dst.ndim != 1 or conn_src.shape != conn_dst.shape:
        raise ValueError("conn_src_1d/conn_dst_2d must be 1D arrays of the same shape")
    if edge_from.ndim != 1 or edge_to.ndim != 1 or edge_from.shape != edge_to.shape:
        raise ValueError("edge_from/edge_to must be 1D arrays of the same shape")
    if edge_weight.shape != edge_from.shape:
        raise ValueError("edge_weight must align with edge_from/edge_to")
    if n1 < 1 or k < 1:
        raise ValueError("n_1d and slots_per_node must be >= 1")

    edge_slots = np.full((n1, k), -1, dtype=np.int64)
    edge_sign = np.zeros((n1, k), dtype=np.float32)
    incident_from: dict[int, np.ndarray] = {}
    incident_to: dict[int, np.ndarray] = {}
    for node2d in np.unique(conn_dst):
        incident_from[int(node2d)] = np.flatnonzero(edge_from == int(node2d))
        incident_to[int(node2d)] = np.flatnonzero(edge_to == int(node2d))

    for node1 in range(n1):
        cell_ids = conn_dst[conn_src == node1]
        if cell_ids.size == 0:
            continue
        candidates: list[tuple[float, int, float]] = []
        seen: set[int] = set()
        for cell in cell_ids.tolist():
            for eidx in incident_to.get(int(cell), np.empty((0,), dtype=np.int64)).tolist():
                if int(eidx) in seen:
                    continue
                seen.add(int(eidx))
                candidates.append((float(edge_weight[eidx]), int(eidx), 1.0))
            for eidx in incident_from.get(int(cell), np.empty((0,), dtype=np.int64)).tolist():
                if int(eidx) in seen:
                    continue
                seen.add(int(eidx))
                candidates.append((float(edge_weight[eidx]), int(eidx), -1.0))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        for slot, (_, eidx, sign) in enumerate(candidates[:k]):
            edge_slots[node1, slot] = int(eidx)
            edge_sign[node1, slot] = float(sign)
    return edge_slots, edge_sign


def build_coupled_surface_slot_features(
    *,
    q_edge: np.ndarray,
    edge_slots: np.ndarray,
    edge_sign: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_edge, dtype=np.float32)
    slots = np.asarray(edge_slots, dtype=np.int64)
    signs = np.asarray(edge_sign, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError("q_edge must be [T, E]")
    if slots.ndim != 2 or signs.shape != slots.shape:
        raise ValueError("edge_slots/edge_sign must be [N, K] with the same shape")

    T, _ = q.shape
    N, K = slots.shape
    feat = np.zeros((T, N, int(3 * K)), dtype=np.float32)
    for slot in range(K):
        eidx = slots[:, slot]
        sgn = signs[:, slot]
        valid = eidx >= 0
        if not np.any(valid):
            continue
        series = np.zeros((T, N), dtype=np.float32)
        series[:, valid] = q[:, eidx[valid]] * sgn[valid][None, :]
        prev = np.empty_like(series)
        prev[0] = series[0]
        prev[1:] = series[:-1]
        delta = (series - prev).astype(np.float32, copy=False)
        base = int(3 * slot)
        feat[:, :, base + 0] = series
        feat[:, :, base + 1] = prev
        feat[:, :, base + 2] = delta
    return feat


def build_coupled_neighbor_node_index(
    *,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    edge_weight: np.ndarray,
    n_1d: int,
    slots_per_node: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    conn_src = np.asarray(conn_src_1d, dtype=np.int64)
    conn_dst = np.asarray(conn_dst_2d, dtype=np.int64)
    edge_from = np.asarray(edge_from, dtype=np.int64)
    edge_to = np.asarray(edge_to, dtype=np.int64)
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    n1 = int(n_1d)
    k = int(slots_per_node)
    if conn_src.ndim != 1 or conn_dst.ndim != 1 or conn_src.shape != conn_dst.shape:
        raise ValueError("conn_src_1d/conn_dst_2d must be 1D arrays of the same shape")
    if edge_from.ndim != 1 or edge_to.ndim != 1 or edge_from.shape != edge_to.shape:
        raise ValueError("edge_from/edge_to must be 1D arrays of the same shape")
    if edge_weight.shape != edge_from.shape:
        raise ValueError("edge_weight must align with edge_from/edge_to")
    if n1 < 1 or k < 1:
        raise ValueError("n_1d and slots_per_node must be >= 1")

    center_cell = np.full((n1,), -1, dtype=np.int64)
    node_slots = np.full((n1, k), -1, dtype=np.int64)
    nbr_from: dict[int, list[tuple[float, int]]] = {}
    for eidx, (u, v) in enumerate(zip(edge_from.tolist(), edge_to.tolist(), strict=True)):
        nbr_from.setdefault(int(u), []).append((float(edge_weight[eidx]), int(v)))
        nbr_from.setdefault(int(v), []).append((float(edge_weight[eidx]), int(u)))

    for node1 in range(n1):
        cell_ids = conn_dst[conn_src == node1]
        if cell_ids.size == 0:
            continue
        center = int(cell_ids[0])
        center_cell[node1] = center
        candidates: list[tuple[float, int]] = []
        seen: set[int] = {center}
        for cell in cell_ids.tolist():
            for w, nbr in nbr_from.get(int(cell), []):
                if int(nbr) in seen:
                    continue
                seen.add(int(nbr))
                candidates.append((float(w), int(nbr)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for slot, (_, nbr) in enumerate(candidates[:k]):
            node_slots[node1, slot] = int(nbr)
    return center_cell, node_slots


def build_coupled_neighbor_node_features(
    *,
    y2_2d: np.ndarray,
    bed_2d: np.ndarray,
    center_cell: np.ndarray,
    neighbor_slots: np.ndarray,
) -> np.ndarray:
    y2 = np.asarray(y2_2d, dtype=np.float32)
    bed = np.asarray(bed_2d, dtype=np.float32)
    center = np.asarray(center_cell, dtype=np.int64)
    nbr = np.asarray(neighbor_slots, dtype=np.int64)
    if y2.ndim != 2:
        raise ValueError("y2_2d must be [T, N2]")
    if bed.shape != (int(y2.shape[1]),):
        raise ValueError("bed_2d must be [N2]")
    if center.shape != (int(nbr.shape[0]),):
        raise ValueError("center_cell must be [N1] aligned to neighbor_slots")
    if nbr.ndim != 2:
        raise ValueError("neighbor_slots must be [N1, K]")

    T, _ = y2.shape
    N, K = nbr.shape
    feat = np.zeros((T, N, int(4 * K)), dtype=np.float32)
    center_valid = center >= 0
    center_head = np.zeros((T, N), dtype=np.float32)
    if np.any(center_valid):
        center_head[:, center_valid] = y2[:, center[center_valid]]

    for slot in range(K):
        idx = nbr[:, slot]
        valid = idx >= 0
        if not np.any(valid):
            continue
        head = np.zeros((T, N), dtype=np.float32)
        head[:, valid] = y2[:, idx[valid]]
        depth = np.zeros((T, N), dtype=np.float32)
        depth[:, valid] = np.maximum(head[:, valid] - bed[idx[valid]][None, :], 0.0).astype(np.float32, copy=False)
        depth_prev = np.empty_like(depth)
        depth_prev[0] = depth[0]
        depth_prev[1:] = depth[:-1]
        gap = (head - center_head).astype(np.float32, copy=False)
        base = int(4 * slot)
        feat[:, :, base + 0] = depth
        feat[:, :, base + 1] = depth_prev
        feat[:, :, base + 2] = (depth - depth_prev).astype(np.float32, copy=False)
        feat[:, :, base + 3] = gap
    return feat


def fit_surface_flow_aux(
    *,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    exo_sequences: list[np.ndarray],
    cfg: SurfaceFlowAuxCfg,
) -> tuple[np.ndarray, np.ndarray]:
    return fit_regime_arkx_exo_per_node(
        sequences,
        exo_sequences=exo_sequences,
        cfg=RegimeARKXConfig(
            k=int(cfg.k),
            ridge=float(cfg.ridge),
            bins=tuple(float(x) for x in cfg.bins),
            equalize_events=bool(cfg.equalize_events),
        ),
    )


def build_payload(
    *,
    cfg: SurfaceFlowAuxCfg,
    split: dict,
    w: np.ndarray,
    counts: np.ndarray,
    edge_count: int,
    exo_dim: int,
    max_delta: float | np.ndarray | None = None,
    metrics: dict | None = None,
) -> dict:
    payload = {
        "kind": "aux_surfaceflow_m2_2d",
        "cfg": asdict(cfg),
        "split": split,
        "exo_kind": "surface_2d_edge_state_v1",
        "exo_dim": int(exo_dim),
        "edge_count": int(edge_count),
        "bins": [float(x) for x in cfg.bins],
        "w": w,
        "regime_step_counts": counts,
    }
    if max_delta is not None:
        payload["max_delta"] = max_delta
    if metrics:
        payload["metrics"] = metrics
    return payload
