from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urbanflood.baseline import (
    RegimeARKXConfig,
    aggregate_2d_to_1d_mean,
    aggregate_2d_to_1d_sum,
    build_coupled_1d_exo,
    fit_regime_arkx_exo_per_node,
    rollout_regime_arkx_exo,
)


@dataclass(frozen=True)
class EdgeFlowAuxCfg:
    model_id: int = 2
    warmup: int = 10
    k: int = 10
    ridge: float = 1e-3
    bins: tuple[float, ...] = (0.0, 0.03, 0.05)
    equalize_events: bool = True


def load_edge_flow_1d(
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
        cache_path = Path(cache_dir) / f"model_{int(model_id)}" / str(split) / f"event_{int(event_id)}_edgeflow1d.npz"
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
        base / "1d_edges_dynamic_all.csv",
        usecols=["timestep", "edge_idx", "flow"],
        dtype={"timestep": np.int32, "edge_idx": np.int32, "flow": np.float32},
    )
    if int(df.shape[0]) != int(t * n_edges):
        raise ValueError(f"unexpected 1D edge rows: got {df.shape[0]} expected {t*n_edges}")
    edge_idx = df["edge_idx"].to_numpy(np.int64)
    if not np.array_equal(edge_idx[:n_edges], np.arange(n_edges, dtype=np.int64)):
        df = df.sort_values(["timestep", "edge_idx"], kind="mergesort").reset_index(drop=True)
    q = df["flow"].to_numpy(np.float32).reshape(t, n_edges)
    if cache_path is not None:
        np.savez(cache_path, q=q)
    return q


def build_edge_flow_exo(
    *,
    y1_1d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
) -> np.ndarray:
    y1 = np.asarray(y1_1d, dtype=np.float32)
    rain = np.asarray(rain_2d, dtype=np.float32)
    if y1.ndim != 2:
        raise ValueError("y1_1d must be [T, N1]")
    if rain.ndim != 2 or rain.shape[0] != y1.shape[0]:
        raise ValueError("rain_2d must be [T, N2] aligned to y1_1d")
    T, _ = y1.shape

    rain_mean_1d = aggregate_2d_to_1d_mean(
        rain,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=int(y1.shape[1]),
    ).astype(np.float32, copy=False)
    rain_vol_2d = ((rain / 12.0) * area_2d[None, :]).astype(np.float32, copy=False)
    rain_vol_1d = aggregate_2d_to_1d_sum(
        rain_vol_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=int(y1.shape[1]),
    ).astype(np.float32, copy=False)

    y_from = y1[:, edge_from]
    y_to = y1[:, edge_to]
    dhead = (y_from - y_to).astype(np.float32, copy=False)
    dhead_prev = np.empty_like(dhead)
    dhead_prev[0] = dhead[0]
    dhead_prev[1:] = dhead[:-1]
    ddhead = (dhead - dhead_prev).astype(np.float32, copy=False)

    rain_from = rain_mean_1d[:, edge_from]
    rain_to = rain_mean_1d[:, edge_to]
    rvol_from = rain_vol_1d[:, edge_from]
    rvol_to = rain_vol_1d[:, edge_to]

    exo = np.stack(
        [
            y_from,
            y_to,
            dhead,
            np.abs(dhead).astype(np.float32, copy=False),
            ddhead,
            rain_from,
            rain_to,
            rvol_from,
            rvol_to,
        ],
        axis=-1,
    )
    if exo.shape[:2] != (T, int(edge_from.shape[0])):
        raise RuntimeError(f"unexpected edge exo shape: {exo.shape}")
    return exo.astype(np.float32, copy=False)


def predict_edge_flow_from_ckpt(
    *,
    ckpt: dict,
    y1_1d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    q_edge_init: np.ndarray,
    warmup: int,
) -> np.ndarray:
    if str(ckpt.get("kind", "")) != "aux_edgeflow_m2_1d":
        raise ValueError("edgeflow ckpt must be kind aux_edgeflow_m2_1d")
    exo = build_edge_flow_exo(
        y1_1d=y1_1d,
        rain_2d=rain_2d,
        area_2d=area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        edge_from=edge_from,
        edge_to=edge_to,
    )
    max_delta_raw = ckpt.get("max_delta", None)
    if max_delta_raw is None:
        max_delta = None
    elif isinstance(max_delta_raw, np.ndarray):
        max_delta = max_delta_raw.astype(np.float32, copy=False)
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


def aggregate_edge_flows_to_nodes(
    *,
    q_edge: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    n_1d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q_edge, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != int(edge_from.shape[0]) or q.shape[1] != int(edge_to.shape[0]):
        raise ValueError("q_edge must be [T, E] aligned to edge_from/edge_to")
    inflow = np.zeros((int(q.shape[0]), int(n_1d)), dtype=np.float32)
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


def build_edge_slot_index(
    *,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    edge_weight: np.ndarray,
    n_1d: int,
    slots_per_dir: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    edge_from = np.asarray(edge_from, dtype=np.int64)
    edge_to = np.asarray(edge_to, dtype=np.int64)
    edge_weight = np.asarray(edge_weight, dtype=np.float32)
    if edge_from.ndim != 1 or edge_to.ndim != 1 or edge_from.shape != edge_to.shape:
        raise ValueError("edge_from/edge_to must be 1D arrays of the same shape")
    if edge_weight.shape != edge_from.shape:
        raise ValueError("edge_weight must align with edge_from/edge_to")
    n1 = int(n_1d)
    k = int(slots_per_dir)
    if n1 < 1 or k < 1:
        raise ValueError("n_1d and slots_per_dir must be >= 1")

    in_slots = np.full((n1, k), -1, dtype=np.int64)
    out_slots = np.full((n1, k), -1, dtype=np.int64)
    for node in range(n1):
        idx_in = np.flatnonzero(edge_to == node)
        if idx_in.size:
            idx_in = idx_in[np.argsort(edge_weight[idx_in])[::-1]]
            in_slots[node, : min(k, idx_in.size)] = idx_in[:k]
        idx_out = np.flatnonzero(edge_from == node)
        if idx_out.size:
            idx_out = idx_out[np.argsort(edge_weight[idx_out])[::-1]]
            out_slots[node, : min(k, idx_out.size)] = idx_out[:k]
    return in_slots, out_slots


def build_edge_flow_slot_features(
    *,
    q_edge: np.ndarray,
    edge_in_slots: np.ndarray,
    edge_out_slots: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_edge, dtype=np.float32)
    in_slots = np.asarray(edge_in_slots, dtype=np.int64)
    out_slots = np.asarray(edge_out_slots, dtype=np.int64)
    if q.ndim != 2:
        raise ValueError("q_edge must be [T, E]")
    if in_slots.ndim != 2 or out_slots.ndim != 2 or in_slots.shape != out_slots.shape:
        raise ValueError("edge_in_slots/edge_out_slots must be [N, K] with the same shape")

    T, E = q.shape
    N, K = in_slots.shape
    feat = np.zeros((T, N, int(6 * K)), dtype=np.float32)

    def _write_block(slot_idx: np.ndarray, *, offset: int, sign: float) -> None:
        for slot in range(K):
            eidx = slot_idx[:, slot]
            valid = eidx >= 0
            if not np.any(valid):
                continue
            series = np.zeros((T, N), dtype=np.float32)
            series[:, valid] = sign * q[:, eidx[valid]]
            prev = np.empty_like(series)
            prev[0] = series[0]
            prev[1:] = series[:-1]
            delta = (series - prev).astype(np.float32, copy=False)
            base = int(offset + 3 * slot)
            feat[..., base] = series
            feat[..., base + 1] = prev
            feat[..., base + 2] = delta

    _write_block(in_slots, offset=0, sign=1.0)
    _write_block(out_slots, offset=3 * K, sign=-1.0)
    return feat.astype(np.float32, copy=False)


def fit_edge_flow_aux(
    *,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    exo_sequences: list[np.ndarray],
    cfg: EdgeFlowAuxCfg,
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
    cfg: EdgeFlowAuxCfg,
    split: dict,
    w: np.ndarray,
    counts: np.ndarray,
    edge_count: int,
    exo_dim: int,
    max_delta: float | np.ndarray | None = None,
    metrics: dict | None = None,
) -> dict:
    payload = {
        "kind": "aux_edgeflow_m2_1d",
        "cfg": asdict(cfg),
        "split": split,
        "edge_count": int(edge_count),
        "exo_kind": "node_head_rain_v1",
        "exo_dim": int(exo_dim),
        "bins": [float(x) for x in cfg.bins],
        "w": w,
        "regime_step_counts": counts,
    }
    if max_delta is not None:
        payload["max_delta"] = max_delta
    if metrics:
        payload["metrics"] = metrics
    return payload
