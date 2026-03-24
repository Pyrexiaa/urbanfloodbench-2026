# #####
# baseline.py
# #####

# urbanflood/baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


def rain_scalar_from_2d(rain_2d: np.ndarray) -> np.ndarray:
    """
    Convert rain_2d[T, N2] into a single rain time series rain[T].

    In this dataset rainfall is effectively spatially uniform, but we use a mean
    for robustness if there are tiny per-cell numeric differences.
    """
    rain_2d = np.asarray(rain_2d, dtype=np.float64)
    if rain_2d.ndim != 2:
        raise ValueError(f"rain_2d must be [T, N2], got shape={rain_2d.shape}")
    return rain_2d.mean(axis=1)


def aggregate_2d_to_1d_mean(
    values_2d: np.ndarray,  # [T, N2]
    *,
    conn_src_1d: np.ndarray,  # [Nc]
    conn_dst_2d: np.ndarray,  # [Nc]
    n_1d: int,
) -> np.ndarray:
    """
    Aggregate a 2D node time series onto 1D nodes by averaging over connected 2D nodes.

    Intended for using predicted 2D state as an exogenous driver for 1D residual
    features (coupled dynamics).
    """
    values_2d = np.asarray(values_2d, dtype=np.float64)
    if values_2d.ndim != 2:
        raise ValueError(f"values_2d must be [T, N2], got shape={values_2d.shape}")
    T, N2 = values_2d.shape

    conn_src_1d = np.asarray(conn_src_1d, dtype=np.int64)
    conn_dst_2d = np.asarray(conn_dst_2d, dtype=np.int64)
    if conn_src_1d.ndim != 1 or conn_dst_2d.ndim != 1 or conn_src_1d.shape != conn_dst_2d.shape:
        raise ValueError("conn_src_1d/conn_dst_2d must be 1D arrays of the same shape")
    if conn_src_1d.size == 0:
        return np.zeros((T, int(n_1d)), dtype=np.float64)
    if (conn_src_1d < 0).any() or (conn_src_1d >= int(n_1d)).any():
        raise ValueError("conn_src_1d indices out of range")
    if (conn_dst_2d < 0).any() or (conn_dst_2d >= int(N2)).any():
        raise ValueError("conn_dst_2d indices out of range")

    out = np.zeros((T, int(n_1d)), dtype=np.float64)
    vals = values_2d[:, conn_dst_2d]  # [T, Nc]
    # Nc is relatively small; loop over connections with vector ops per connection.
    for j, s in enumerate(conn_src_1d.tolist()):
        out[:, s] += vals[:, j]

    deg = np.bincount(conn_src_1d, minlength=int(n_1d)).astype(np.float64)
    out = out / np.maximum(deg[None, :], 1.0)
    return out


def aggregate_2d_to_1d_sum(
    values_2d: np.ndarray,  # [T, N2]
    *,
    conn_src_1d: np.ndarray,  # [Nc]
    conn_dst_2d: np.ndarray,  # [Nc]
    n_1d: int,
) -> np.ndarray:
    """
    Aggregate a 2D node time series onto 1D nodes by summing over connected 2D nodes.
    """
    values_2d = np.asarray(values_2d, dtype=np.float64)
    if values_2d.ndim != 2:
        raise ValueError(f"values_2d must be [T, N2], got shape={values_2d.shape}")
    T, N2 = values_2d.shape

    conn_src_1d = np.asarray(conn_src_1d, dtype=np.int64)
    conn_dst_2d = np.asarray(conn_dst_2d, dtype=np.int64)
    if conn_src_1d.ndim != 1 or conn_dst_2d.ndim != 1 or conn_src_1d.shape != conn_dst_2d.shape:
        raise ValueError("conn_src_1d/conn_dst_2d must be 1D arrays of the same shape")
    if conn_src_1d.size == 0:
        return np.zeros((T, int(n_1d)), dtype=np.float64)
    if (conn_src_1d < 0).any() or (conn_src_1d >= int(n_1d)).any():
        raise ValueError("conn_src_1d indices out of range")
    if (conn_dst_2d < 0).any() or (conn_dst_2d >= int(N2)).any():
        raise ValueError("conn_dst_2d indices out of range")

    out = np.zeros((T, int(n_1d)), dtype=np.float64)
    vals = values_2d[:, conn_dst_2d]  # [T, Nc]
    for j, s in enumerate(conn_src_1d.tolist()):
        out[:, s] += vals[:, j]
    return out


def connected_area_1d(
    area_2d: np.ndarray,
    *,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
) -> np.ndarray:
    """
    Sum connected 2D cell areas for each 1D node.
    """
    area_2d = np.asarray(area_2d, dtype=np.float64)
    if area_2d.ndim != 1:
        raise ValueError(f"area_2d must be [N2], got shape={area_2d.shape}")
    conn_src_1d = np.asarray(conn_src_1d, dtype=np.int64)
    conn_dst_2d = np.asarray(conn_dst_2d, dtype=np.int64)
    if conn_src_1d.ndim != 1 or conn_dst_2d.ndim != 1 or conn_src_1d.shape != conn_dst_2d.shape:
        raise ValueError("conn_src_1d/conn_dst_2d must be 1D arrays of the same shape")
    if conn_src_1d.size == 0:
        return np.zeros((int(n_1d),), dtype=np.float64)
    if (conn_src_1d < 0).any() or (conn_src_1d >= int(n_1d)).any():
        raise ValueError("conn_src_1d indices out of range")
    if (conn_dst_2d < 0).any() or (conn_dst_2d >= int(area_2d.shape[0])).any():
        raise ValueError("conn_dst_2d indices out of range")
    out = np.zeros((int(n_1d),), dtype=np.float64)
    np.add.at(out, conn_src_1d, area_2d[conn_dst_2d])
    return out


def build_coupled_1d_exo(
    *,
    y2_2d: np.ndarray,  # [T, N2]
    rain_2d: np.ndarray,  # [T, N2]
    area_2d: np.ndarray,  # [N2]
    conn_src_1d: np.ndarray,  # [Nc]
    conn_dst_2d: np.ndarray,  # [Nc]
    n_1d: int,
) -> np.ndarray:
    """
    Build node-wise exogenous features for 1D rollout from the coupled 2D state.

    Feature order:
      0: connected 2D mean level at time t
      1: connected 2D mean level at time t-1
      2: connected 2D level delta (t - (t-1))
      3: connected 2D mean rainfall depth at time t
      4: connected 2D rainfall volume proxy at time t
      5: cumulative connected rainfall volume proxy through time t
    """
    y2_2d = np.asarray(y2_2d, dtype=np.float64)
    rain_2d = np.asarray(rain_2d, dtype=np.float64)
    area_2d = np.asarray(area_2d, dtype=np.float64)
    if y2_2d.ndim != 2 or rain_2d.ndim != 2:
        raise ValueError("y2_2d and rain_2d must be [T, N2]")
    if y2_2d.shape != rain_2d.shape:
        raise ValueError("y2_2d and rain_2d must have the same shape")
    T, N2 = y2_2d.shape
    if area_2d.shape != (N2,):
        raise ValueError(f"area_2d must be shape ({N2},), got {area_2d.shape}")

    cpl_level = aggregate_2d_to_1d_mean(
        y2_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    cpl_prev = np.empty_like(cpl_level)
    cpl_prev[0] = cpl_level[0]
    cpl_prev[1:] = cpl_level[:-1]
    cpl_delta = cpl_level - cpl_prev

    rain_mean = aggregate_2d_to_1d_mean(
        rain_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    rain_volume_2d = (rain_2d / 12.0) * area_2d[None, :]
    rain_vol = aggregate_2d_to_1d_sum(
        rain_volume_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    rain_vol_cum = np.cumsum(rain_vol, axis=0)

    exo = np.stack([cpl_level, cpl_prev, cpl_delta, rain_mean, rain_vol, rain_vol_cum], axis=-1)
    if exo.shape != (T, int(n_1d), 6):
        raise RuntimeError(f"unexpected exo shape: {exo.shape}")
    return exo.astype(np.float32, copy=False)


def build_coupled_local_1d_exo(
    *,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
    center_cell: np.ndarray,
    neighbor_slots: np.ndarray,
) -> np.ndarray:
    base = build_coupled_1d_exo(
        y2_2d=y2_2d,
        rain_2d=rain_2d,
        area_2d=area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    ).astype(np.float64, copy=False)
    y2 = np.asarray(y2_2d, dtype=np.float64)
    rain = np.asarray(rain_2d, dtype=np.float64)
    center = np.asarray(center_cell, dtype=np.int64)
    nbr = np.asarray(neighbor_slots, dtype=np.int64)
    if center.shape != (int(n_1d),):
        raise ValueError("center_cell must be [N1]")
    if nbr.ndim != 2 or nbr.shape[0] != int(n_1d):
        raise ValueError("neighbor_slots must be [N1, K]")
    T, n2 = y2.shape
    center_safe = np.clip(center, 0, n2 - 1)
    center_level = y2[:, center_safe]
    center_mask = (center >= 0).astype(np.float64, copy=False)
    center_level *= center_mask[None, :]

    if int(nbr.shape[1]) <= 0:
        extra = np.zeros((T, int(n_1d), 8), dtype=np.float64)
        return np.concatenate([base, extra], axis=2).astype(np.float32, copy=False)

    nbr_safe = np.clip(nbr, 0, n2 - 1)
    nbr_mask = nbr >= 0
    nbr_count = np.maximum(nbr_mask.sum(axis=1, keepdims=False), 1).astype(np.float64, copy=False)

    nbr_level = y2[:, nbr_safe] * nbr_mask[None, :, :]
    nbr_rain = rain[:, nbr_safe] * nbr_mask[None, :, :]

    nbr_mean = nbr_level.sum(axis=2) / nbr_count[None, :]
    nbr_prev = np.empty_like(nbr_mean)
    nbr_prev[0] = nbr_mean[0]
    nbr_prev[1:] = nbr_mean[:-1]
    nbr_delta = nbr_mean - nbr_prev

    nbr_level_masked = np.where(nbr_mask[None, :, :], nbr_level, -1e30)
    nbr_max = nbr_level_masked.max(axis=2)
    nbr_max[:, nbr_count <= 0.0] = 0.0

    gap = nbr_level - center_level[:, :, None]
    gap_pos = np.maximum(gap, 0.0)
    gap_mean_pos = gap_pos.sum(axis=2) / nbr_count[None, :]
    gap_pos_masked = np.where(nbr_mask[None, :, :], gap_pos, -1e30)
    gap_max_pos = gap_pos_masked.max(axis=2)
    gap_max_pos[:, nbr_count <= 0.0] = 0.0

    rain_mean = nbr_rain.sum(axis=2) / nbr_count[None, :]
    rain_masked = np.where(nbr_mask[None, :, :], nbr_rain, -1e30)
    rain_max = rain_masked.max(axis=2)
    rain_max[:, nbr_count <= 0.0] = 0.0

    extra = np.stack([nbr_mean, nbr_prev, nbr_delta, nbr_max, gap_mean_pos, gap_max_pos, rain_mean, rain_max], axis=2)
    return np.concatenate([base, extra], axis=2).astype(np.float32, copy=False)


def build_inlet_augmented_exo(
    coupled_exo: np.ndarray,  # [T, N, D]
    inlet_1d: np.ndarray,  # [T, N]
) -> np.ndarray:
    """
    Augment coupled 1D exogenous features with inlet-flow state:
      - inlet flow at time t
      - inlet flow at time t-1
      - inlet flow delta (t - (t-1))
    """
    coupled_exo = np.asarray(coupled_exo, dtype=np.float64)
    inlet_1d = np.asarray(inlet_1d, dtype=np.float64)
    if coupled_exo.ndim != 3:
        raise ValueError("coupled_exo must be [T, N, D]")
    if inlet_1d.ndim != 2 or inlet_1d.shape[:2] != coupled_exo.shape[:2]:
        raise ValueError("inlet_1d must be [T, N] aligned to coupled_exo")

    q_prev = np.empty_like(inlet_1d)
    q_prev[0] = inlet_1d[0]
    q_prev[1:] = inlet_1d[:-1]
    q_delta = inlet_1d - q_prev

    aug = np.concatenate([coupled_exo, inlet_1d[..., None], q_prev[..., None], q_delta[..., None]], axis=2)
    return aug.astype(np.float32, copy=False)


def build_storage_augmented_exo(
    coupled_exo: np.ndarray,  # [T, N, D]
    volagg_1d: np.ndarray,  # [T, N]
    conn_area_1d: np.ndarray,  # [N]
) -> np.ndarray:
    """
    Augment coupled 1D exogenous features with connected 2D storage state:
      - signed log1p(volume) at time t
      - signed log1p(volume) at time t-1
      - delta signed log-volume
      - area-normalized storage depth at time t
      - area-normalized storage depth at time t-1
      - delta normalized storage depth
    """
    coupled_exo = np.asarray(coupled_exo, dtype=np.float64)
    volagg_1d = np.asarray(volagg_1d, dtype=np.float64)
    conn_area_1d = np.asarray(conn_area_1d, dtype=np.float64)
    if coupled_exo.ndim != 3:
        raise ValueError("coupled_exo must be [T, N, D]")
    if volagg_1d.ndim != 2 or volagg_1d.shape[:2] != coupled_exo.shape[:2]:
        raise ValueError("volagg_1d must be [T, N] aligned to coupled_exo")
    if conn_area_1d.ndim != 1 or conn_area_1d.shape[0] != coupled_exo.shape[1]:
        raise ValueError("conn_area_1d must be [N] aligned to coupled_exo")

    v_prev = np.empty_like(volagg_1d)
    v_prev[0] = volagg_1d[0]
    v_prev[1:] = volagg_1d[:-1]
    v_log = np.sign(volagg_1d) * np.log1p(np.abs(volagg_1d))
    v_log_prev = np.sign(v_prev) * np.log1p(np.abs(v_prev))
    v_log_delta = v_log - v_log_prev

    area = np.maximum(conn_area_1d, 1.0)
    d = volagg_1d / area[None, :]
    d_prev = v_prev / area[None, :]
    d_delta = d - d_prev

    aug = np.concatenate(
        [
            coupled_exo,
            v_log[..., None],
            v_log_prev[..., None],
            v_log_delta[..., None],
            d[..., None],
            d_prev[..., None],
            d_delta[..., None],
        ],
        axis=2,
    )
    return aug.astype(np.float32, copy=False)


def build_inlet_storage_augmented_exo(
    coupled_exo: np.ndarray,  # [T, N, D]
    inlet_1d: np.ndarray,  # [T, N]
    volagg_1d: np.ndarray,  # [T, N]
    conn_area_1d: np.ndarray,  # [N]
) -> np.ndarray:
    """
    Augment coupled 1D exogenous features with both inlet-flow and storage state.
    """
    exo = build_inlet_augmented_exo(coupled_exo, inlet_1d)
    storage_only = build_storage_augmented_exo(
        np.zeros((exo.shape[0], exo.shape[1], 0), dtype=np.float32),
        volagg_1d,
        conn_area_1d,
    )
    return np.concatenate([exo, storage_only], axis=2).astype(np.float32, copy=False)


def _solve_batch_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.zeros_like(b, dtype=np.float64)
        for i in range(int(A.shape[0])):
            try:
                out[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                out[i] = np.linalg.lstsq(A[i], b[i], rcond=1e-8)[0]
        return out


@dataclass(frozen=True)
class AR1XConfig:
    ridge: float = 1e-4
    ridge_scale: bool = False


@dataclass(frozen=True)
class RegimeARKXConfig:
    k: int = 10
    ridge: float = 1e-3
    # Bins applied to rain[t+1] in inches. With right=True digitize:
    # reg0: r<=bins[0], reg1: bins[0]<r<=bins[1], ..., regR: r>bins[-1]
    bins: tuple[float, ...] = (0.0, 0.03, 0.05)
    # If True, weight each event equally in the normal equations (approx LB weighting).
    equalize_events: bool = False
    # If True, scale ridge per-node by trace(X^T X)/F to make regularization more scale-aware.
    ridge_scale: bool = False


def _symmetrize_inplace(A: np.ndarray) -> None:
    # A: [..., F, F]
    F = int(A.shape[-1])
    for i in range(F):
        for j in range(i):
            A[..., i, j] = A[..., j, i]


def fit_ar1x_per_node(
    sequences: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    cfg: AR1XConfig,
) -> np.ndarray:
    """
    Fit AR(1)+exogenous per node:

      y[t+1] = w0 + w1*y[t] + w2*rain[t+1] + w3*cum_rain[t+1]

    Returns:
      w[N, 4] float32
    """
    sequences = list(sequences)
    if not sequences:
        raise ValueError("no sequences provided")

    y0, _ = sequences[0]
    y0 = np.asarray(y0)
    if y0.ndim != 2:
        raise ValueError("y must be [T, N]")
    N = int(y0.shape[1])

    # Normal equations for each node.
    # A: [N, 4, 4], b: [N, 4]
    A = np.zeros((N, 4, 4), dtype=np.float64)
    b = np.zeros((N, 4), dtype=np.float64)

    for y, rain in sequences:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != N:
            raise ValueError("inconsistent y shapes across sequences")

        if rain.ndim == 2:
            r = rain_scalar_from_2d(rain)
        else:
            r = np.asarray(rain, dtype=np.float64)
        if r.ndim != 1 or r.shape[0] != y.shape[0]:
            raise ValueError("rain must be [T] aligned to y")

        T = int(y.shape[0])
        if T < 2:
            continue

        y_t = y[:-1]  # [T-1, N]
        y_next = y[1:]  # [T-1, N]
        r_feat = r[1:]  # [T-1]
        cum = np.cumsum(r)
        c_feat = cum[1:]  # [T-1]

        S = int(T - 1)
        # Scalars
        sum_r = float(r_feat.sum())
        sum_c = float(c_feat.sum())
        sum_r2 = float((r_feat * r_feat).sum())
        sum_c2 = float((c_feat * c_feat).sum())
        sum_rc = float((r_feat * c_feat).sum())

        # Per-node
        sum_y = y_t.sum(axis=0)  # [N]
        sum_y2 = (y_t * y_t).sum(axis=0)  # [N]
        sum_yn = y_next.sum(axis=0)  # [N]
        sum_yyn = (y_t * y_next).sum(axis=0)  # [N]
        sum_r_y = (y_t * r_feat[:, None]).sum(axis=0)  # [N]
        sum_c_y = (y_t * c_feat[:, None]).sum(axis=0)  # [N]
        sum_r_yn = (y_next * r_feat[:, None]).sum(axis=0)  # [N]
        sum_c_yn = (y_next * c_feat[:, None]).sum(axis=0)  # [N]

        # A blocks
        A[:, 0, 0] += S
        A[:, 0, 1] += sum_y
        A[:, 0, 2] += sum_r
        A[:, 0, 3] += sum_c

        A[:, 1, 1] += sum_y2
        A[:, 1, 2] += sum_r_y
        A[:, 1, 3] += sum_c_y

        A[:, 2, 2] += sum_r2
        A[:, 2, 3] += sum_rc

        A[:, 3, 3] += sum_c2

        # b
        b[:, 0] += sum_yn
        b[:, 1] += sum_yyn
        b[:, 2] += sum_r_yn
        b[:, 3] += sum_c_yn

    _symmetrize_inplace(A)

    ridge = float(cfg.ridge)
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")

    if bool(cfg.ridge_scale):
        F = 4
        scale = np.trace(A, axis1=1, axis2=2) / float(F)  # [N]
        scale = np.maximum(scale, 1e-12)
        A = A + (ridge * scale)[:, None, None] * np.eye(F, dtype=np.float64)[None, :, :]
    else:
        A = A + ridge * np.eye(4, dtype=np.float64)[None, :, :]

    w = _solve_batch_linear(A, b)  # [N, 4]
    if not np.isfinite(w).all():
        raise FloatingPointError("non-finite AR1X weights")
    return w.astype(np.float32, copy=False)


def fit_regime_arkx_per_node(
    sequences: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    cfg: RegimeARKXConfig,
    min_regime_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-node multi-regime AR(k)+exogenous:

      y[t+1] = w0 + sum_{j=0..k-1} w_{j+1}*y[t-j] + w_r*rain[t+1] + w_c*cum_rain[t+1]

    Regime selected by rain[t+1] bins.

    Returns:
      w[R, N, 3+k] float32 with feature order:
        [1, y[t], ..., y[t-k+1], rain[t+1], cum_rain[t+1]]
      regime_step_counts[R] int64
    """
    sequences = list(sequences)
    if not sequences:
        raise ValueError("no sequences provided")

    k = int(cfg.k)
    if k < 1:
        raise ValueError("cfg.k must be >= 1")

    y0, _ = sequences[0]
    y0 = np.asarray(y0)
    if y0.ndim != 2:
        raise ValueError("y must be [T, N]")
    N = int(y0.shape[1])

    bins = np.asarray(cfg.bins, dtype=np.float64)
    if bins.ndim != 1 or bins.size < 1:
        raise ValueError("cfg.bins must contain at least 1 threshold")
    R = int(bins.size + 1)
    F = int(1 + k + 2)

    A = np.zeros((R, N, F, F), dtype=np.float64)
    b = np.zeros((R, N, F), dtype=np.float64)
    counts = np.zeros((R,), dtype=np.int64)

    for y, rain in sequences:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != N:
            raise ValueError("inconsistent y shapes across sequences")

        if rain.ndim == 2:
            r = rain_scalar_from_2d(rain)
        else:
            r = np.asarray(rain, dtype=np.float64)
        if r.ndim != 1 or r.shape[0] != y.shape[0]:
            raise ValueError("rain must be [T] aligned to y")

        T = int(y.shape[0])
        if T < (k + 1):
            continue

        S = int(T - k)
        w_event = (1.0 / float(max(1, S))) if bool(cfg.equalize_events) else 1.0

        y_next = y[k:]  # [S, N]  (t+1)
        r_feat = r[k:]  # [S]     (rain[t+1])
        cum = np.cumsum(r)
        c_feat = cum[k:]  # [S]   (cum_rain[t+1])

        # Lag features [S, N, k] ordered as [y[t], y[t-1], ..., y[t-k+1]].
        lags = np.empty((S, N, k), dtype=np.float64)
        for j in range(k):
            lags[..., j] = y[(k - 1 - j) : (T - 1 - j)]

        # Design matrix X[S, N, F]
        X = np.empty((S, N, F), dtype=np.float64)
        X[..., 0] = 1.0
        X[..., 1 : 1 + k] = lags
        X[..., 1 + k] = r_feat[:, None]
        X[..., 2 + k] = c_feat[:, None]

        g = np.digitize(r_feat, bins, right=True)  # [S], values 0..R-1
        for reg in range(R):
            mask = g == reg
            if not mask.any():
                continue
            Xr = X[mask]  # [Sreg, N, F]
            yn = y_next[mask]  # [Sreg, N]
            Sreg = int(Xr.shape[0])
            counts[reg] += Sreg

            A[reg] += w_event * np.einsum("snf,sng->nfg", Xr, Xr, optimize=True)
            b[reg] += w_event * np.einsum("snf,sn->nf", Xr, yn, optimize=True)

    _symmetrize_inplace(A)

    ridge = float(cfg.ridge)
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    fallback_reg = int(np.argmax(counts)) if counts.size else 0

    w = np.zeros((R, N, F), dtype=np.float64)
    eye0 = np.eye(F, dtype=np.float64)
    eye = eye0[None, :, :]
    for reg in range(R):
        if int(counts[reg]) < int(min_regime_steps):
            continue
        if bool(cfg.ridge_scale):
            scale = np.trace(A[reg], axis1=1, axis2=2) / float(F)  # [N]
            scale = np.maximum(scale, 1e-12)
            Ar = A[reg] + (ridge * scale)[:, None, None] * eye0[None, :, :]
        else:
            Ar = A[reg] + ridge * eye
        w[reg] = _solve_batch_linear(Ar, b[reg])

    # Fallback for extremely rare regimes: copy weights from the most common regime.
    if int(counts[fallback_reg]) >= int(min_regime_steps):
        for reg in range(R):
            if int(counts[reg]) < int(min_regime_steps):
                w[reg] = w[fallback_reg]

    if not np.isfinite(w).all():
        raise FloatingPointError("non-finite RegimeARKX weights")
    return w.astype(np.float32, copy=False), counts


def fit_regime_arkx_exo_per_node(
    sequences: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    exo_sequences: Iterable[np.ndarray],
    regime_signal_sequences: Iterable[np.ndarray] | None = None,
    cfg: RegimeARKXConfig,
    min_regime_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-node multi-regime AR(k)+exogenous with node-wise exogenous features:

      y[t+1] = w0 + sum_j w_{j+1} * y[t-j] + w_r * rain[t+1] + w_c * cum_rain[t+1] + sum_d u_d * exo_d[t+1]

    `exo_sequences` must align 1:1 with `sequences`, with shape [T, N, D].
    If `regime_signal_sequences` is provided, regimes are selected from those
    node-wise signals instead of rainfall. Each item must be [T] or [T, N].
    """
    sequences = list(sequences)
    exo_sequences = list(exo_sequences)
    if not sequences:
        raise ValueError("no sequences provided")
    if len(sequences) != len(exo_sequences):
        raise ValueError("sequences and exo_sequences must have the same length")
    if regime_signal_sequences is not None:
        regime_signal_sequences = list(regime_signal_sequences)
        if len(sequences) != len(regime_signal_sequences):
            raise ValueError("sequences and regime_signal_sequences must have the same length")

    k = int(cfg.k)
    if k < 1:
        raise ValueError("cfg.k must be >= 1")

    y0, _ = sequences[0]
    y0 = np.asarray(y0)
    if y0.ndim != 2:
        raise ValueError("y must be [T, N]")
    N = int(y0.shape[1])

    exo0 = np.asarray(exo_sequences[0], dtype=np.float64)
    if exo0.ndim != 3 or exo0.shape[1] != N:
        raise ValueError("exo must be [T, N, D] aligned to y")
    D = int(exo0.shape[2])

    bins = np.asarray(cfg.bins, dtype=np.float64)
    if bins.ndim != 1 or bins.size < 1:
        raise ValueError("cfg.bins must contain at least 1 threshold")
    R = int(bins.size + 1)
    F = int(1 + k + 2 + D)

    A = np.zeros((R, N, F, F), dtype=np.float64)
    b = np.zeros((R, N, F), dtype=np.float64)
    counts = np.zeros((R,), dtype=np.int64)
    counts_node = np.zeros((R, N), dtype=np.int64)

    if regime_signal_sequences is None:
        regime_iter = [None] * len(sequences)
    else:
        regime_iter = regime_signal_sequences

    for (y, rain), exo, regime_sig in zip(sequences, exo_sequences, regime_iter, strict=True):
        y = np.asarray(y, dtype=np.float64)
        exo = np.asarray(exo, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != N:
            raise ValueError("inconsistent y shapes across sequences")
        if exo.ndim != 3 or exo.shape[:2] != y.shape or exo.shape[2] != D:
            raise ValueError("exo must be [T, N, D] aligned to y")

        if rain.ndim == 2:
            r = rain_scalar_from_2d(rain)
        else:
            r = np.asarray(rain, dtype=np.float64)
        if r.ndim != 1 or r.shape[0] != y.shape[0]:
            raise ValueError("rain must be [T] aligned to y")

        T = int(y.shape[0])
        if T < (k + 1):
            continue

        S = int(T - k)
        w_event = (1.0 / float(max(1, S))) if bool(cfg.equalize_events) else 1.0

        y_next = y[k:]
        r_feat = r[k:]
        cum = np.cumsum(r)
        c_feat = cum[k:]

        lags = np.empty((S, N, k), dtype=np.float64)
        for j in range(k):
            lags[..., j] = y[(k - 1 - j) : (T - 1 - j)]

        X = np.empty((S, N, F), dtype=np.float64)
        X[..., 0] = 1.0
        X[..., 1 : 1 + k] = lags
        X[..., 1 + k] = r_feat[:, None]
        X[..., 2 + k] = c_feat[:, None]
        X[..., 3 + k :] = exo[k:]

        if regime_sig is None:
            g = np.digitize(r_feat, bins, right=True)
            for reg in range(R):
                mask = g == reg
                if not mask.any():
                    continue
                Xr = X[mask]
                yn = y_next[mask]
                Sreg = int(Xr.shape[0])
                counts[reg] += Sreg
                counts_node[reg] += Sreg
                A[reg] += w_event * np.einsum("snf,sng->nfg", Xr, Xr, optimize=True)
                b[reg] += w_event * np.einsum("snf,sn->nf", Xr, yn, optimize=True)
            continue

        regime_sig = np.asarray(regime_sig, dtype=np.float64)
        if regime_sig.ndim == 1:
            if regime_sig.shape[0] != T:
                raise ValueError("regime_signal must be [T] aligned to y")
            regime_step = np.broadcast_to(regime_sig[k:, None], (S, N))
        elif regime_sig.ndim == 2:
            if regime_sig.shape != (T, N):
                raise ValueError("regime_signal must be [T, N] aligned to y")
            regime_step = regime_sig[k:]
        else:
            raise ValueError("regime_signal must be [T] or [T, N]")

        g = np.digitize(regime_step, bins, right=True)
        for reg in range(R):
            mask = (g == reg).astype(np.float64, copy=False)
            if not mask.any():
                continue
            counts[reg] += int(mask.sum())
            counts_node[reg] += mask.sum(axis=0).astype(np.int64, copy=False)
            A[reg] += w_event * np.einsum("sn,snf,sng->nfg", mask, X, X, optimize=True)
            b[reg] += w_event * np.einsum("sn,snf,sn->nf", mask, X, y_next, optimize=True)

    _symmetrize_inplace(A)

    ridge = float(cfg.ridge)
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    w = np.zeros((R, N, F), dtype=np.float64)
    eye0 = np.eye(F, dtype=np.float64)
    solved = counts_node >= int(min_regime_steps)
    for reg in range(R):
        idx = np.flatnonzero(solved[reg])
        if idx.size == 0:
            continue
        if bool(cfg.ridge_scale):
            scale = np.trace(A[reg, idx], axis1=1, axis2=2) / float(F)
            scale = np.maximum(scale, 1e-12)
            Ar = A[reg, idx] + (ridge * scale)[:, None, None] * eye0[None, :, :]
        else:
            Ar = A[reg, idx] + ridge * eye0[None, :, :]
        w[reg, idx] = _solve_batch_linear(Ar, b[reg, idx])

    fallback_reg_node = np.argmax(counts_node, axis=0)
    for reg in range(R):
        miss_idx = np.flatnonzero(~solved[reg])
        if miss_idx.size == 0:
            continue
        w[reg, miss_idx] = w[fallback_reg_node[miss_idx], miss_idx]

    if not np.isfinite(w).all():
        raise FloatingPointError("non-finite RegimeARKX+exo weights")
    return w.astype(np.float32, copy=False), counts


def rollout_ar1x(
    *,
    w: np.ndarray,  # [N, 4]
    y_init: np.ndarray,  # [T, N] (may contain NaNs after warmup in test)
    rain: np.ndarray,  # [T] or [T, N2]
    warmup: int = 10,
) -> np.ndarray:
    """Autoregressive rollout using AR(1)+exogenous weights."""
    y_init = np.asarray(y_init, dtype=np.float64)
    if y_init.ndim != 2:
        raise ValueError("y_init must be [T, N]")
    T, N = y_init.shape
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (N, 4):
        raise ValueError(f"w must be [N, 4], got {w.shape}")

    if rain.ndim == 2:
        r = rain_scalar_from_2d(rain)
    else:
        r = np.asarray(rain, dtype=np.float64)
    if r.shape != (T,):
        raise ValueError("rain must be [T] aligned to y_init")
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    if warmup < 1:
        raise ValueError("warmup must be >= 1 for AR1 rollout")

    cum = np.cumsum(r)
    y = np.empty((T, N), dtype=np.float64)
    y[:warmup] = y_init[:warmup]
    if not np.isfinite(y[:warmup]).all():
        raise ValueError("non-finite warmup values")

    for t in range(warmup - 1, T - 1):
        r_next = float(r[t + 1])
        c_next = float(cum[t + 1])
        y_t = y[t]
        y[t + 1] = w[:, 0] + w[:, 1] * y_t + w[:, 2] * r_next + w[:, 3] * c_next

    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite rollout")
    return y.astype(np.float32, copy=False)


def rollout_regime_arkx(
    *,
    w: np.ndarray,  # [R, N, 3+k]
    bins: tuple[float, ...],
    y_init: np.ndarray,  # [T, N]
    rain: np.ndarray,  # [T] or [T, N2]
    warmup: int = 10,
    delta_scale: float = 1.0,
    max_delta: float | np.ndarray | None = None,
) -> np.ndarray:
    """Autoregressive rollout using regime AR(k)+exogenous weights (k inferred from w)."""
    y_init = np.asarray(y_init, dtype=np.float64)
    if y_init.ndim != 2:
        raise ValueError("y_init must be [T, N]")
    T, N = y_init.shape

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 3 or w.shape[1] != N:
        raise ValueError(f"w must be [R, N, F], got {w.shape}")
    R = int(w.shape[0])
    F = int(w.shape[2])
    k = int(F - 3)
    if k < 1:
        raise ValueError("w has invalid feature dimension")

    bins_arr = np.asarray(bins, dtype=np.float64)
    if bins_arr.ndim != 1 or bins_arr.size < 1:
        raise ValueError("bins must contain at least 1 threshold")

    if rain.ndim == 2:
        r = rain_scalar_from_2d(rain)
    else:
        r = np.asarray(rain, dtype=np.float64)
    if r.shape != (T,):
        raise ValueError("rain must be [T] aligned to y_init")
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    if warmup < k:
        raise ValueError(f"warmup must be >= k={k} for AR{k} rollout")

    delta_scale_f = float(delta_scale)
    if not np.isfinite(delta_scale_f) or delta_scale_f <= 0.0 or delta_scale_f > 1.0:
        raise ValueError("delta_scale must be finite and in (0, 1]")

    if max_delta is not None:
        md_arr = np.asarray(max_delta, dtype=np.float64)
        if md_arr.ndim == 0:
            md = float(md_arr)
            if md <= 0.0:
                raise ValueError("max_delta must be > 0 if provided")
            md_node = None
        else:
            if md_arr.shape != (N,):
                raise ValueError(f"max_delta array must be shape (N,) == ({N},), got {md_arr.shape}")
            if not np.isfinite(md_arr).all():
                raise ValueError("max_delta array must be finite")
            if (md_arr <= 0.0).any():
                raise ValueError("max_delta array must be > 0 for all nodes")
            md = None
            md_node = md_arr
    else:
        md = None
        md_node = None

    cum = np.cumsum(r)
    y = np.empty((T, N), dtype=np.float64)
    y[:warmup] = y_init[:warmup]
    if not np.isfinite(y[:warmup]).all():
        raise ValueError("non-finite warmup values")

    # reg is determined by rain[t+1]
    for t in range(warmup - 1, T - 1):
        r_next = float(r[t + 1])
        c_next = float(cum[t + 1])
        reg = int(np.digitize([r_next], bins_arr, right=True)[0])
        if reg < 0 or reg >= R:
            raise ValueError("regime index out of range")
        ww = w[reg]  # [N, F]

        pred = ww[:, 0].copy()
        for j in range(k):
            pred += ww[:, 1 + j] * y[t - j]
        pred += ww[:, 1 + k] * r_next + ww[:, 2 + k] * c_next

        if delta_scale_f != 1.0:
            pred = y[t] + delta_scale_f * (pred - y[t])

        if md is not None:
            d = pred - y[t]
            d = np.clip(d, -md, md)
            pred = y[t] + d
        elif md_node is not None:
            d = pred - y[t]
            d = np.clip(d, -md_node, md_node)
            pred = y[t] + d

        y[t + 1] = pred

    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite rollout")
    return y.astype(np.float32, copy=False)


def rollout_regime_arkx_exo(
    *,
    w: np.ndarray,  # [R, N, 3+k+D]
    bins: tuple[float, ...],
    y_init: np.ndarray,  # [T, N]
    rain: np.ndarray,  # [T] or [T, N2]
    exo: np.ndarray,  # [T, N, D]
    regime_signal: np.ndarray | None = None,  # [T] or [T, N]
    warmup: int = 10,
    delta_scale: float = 1.0,
    max_delta: float | np.ndarray | None = None,
) -> np.ndarray:
    """Autoregressive rollout using regime AR(k)+X with node-wise exogenous features."""
    y_init = np.asarray(y_init, dtype=np.float64)
    if y_init.ndim != 2:
        raise ValueError("y_init must be [T, N]")
    T, N = y_init.shape

    exo = np.asarray(exo, dtype=np.float64)
    if exo.ndim != 3 or exo.shape[:2] != (T, N):
        raise ValueError(f"exo must be [T, N, D], got {exo.shape}")
    D = int(exo.shape[2])

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 3 or w.shape[1] != N:
        raise ValueError(f"w must be [R, N, F], got {w.shape}")
    R = int(w.shape[0])
    F = int(w.shape[2])
    k = int(F - 3 - D)
    if k < 1:
        raise ValueError("w has invalid feature dimension for exo rollout")

    bins_arr = np.asarray(bins, dtype=np.float64)
    if bins_arr.ndim != 1 or bins_arr.size < 1:
        raise ValueError("bins must contain at least 1 threshold")

    if rain.ndim == 2:
        r = rain_scalar_from_2d(rain)
    else:
        r = np.asarray(rain, dtype=np.float64)
    if r.shape != (T,):
        raise ValueError("rain must be [T] aligned to y_init")
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    if warmup < k:
        raise ValueError(f"warmup must be >= k={k} for AR{k} rollout")

    delta_scale_f = float(delta_scale)
    if not np.isfinite(delta_scale_f) or delta_scale_f <= 0.0 or delta_scale_f > 1.0:
        raise ValueError("delta_scale must be finite and in (0, 1]")

    if max_delta is not None:
        md_arr = np.asarray(max_delta, dtype=np.float64)
        if md_arr.ndim == 0:
            md = float(md_arr)
            if md <= 0.0:
                raise ValueError("max_delta must be > 0 if provided")
            md_node = None
        else:
            if md_arr.shape != (N,):
                raise ValueError(f"max_delta array must be shape (N,) == ({N},), got {md_arr.shape}")
            if not np.isfinite(md_arr).all():
                raise ValueError("max_delta array must be finite")
            if (md_arr <= 0.0).any():
                raise ValueError("max_delta array must be > 0 for all nodes")
            md = None
            md_node = md_arr
    else:
        md = None
        md_node = None

    cum = np.cumsum(r)
    y = np.empty((T, N), dtype=np.float64)
    y[:warmup] = y_init[:warmup]
    if not np.isfinite(y[:warmup]).all():
        raise ValueError("non-finite warmup values")

    if regime_signal is not None:
        regime_signal = np.asarray(regime_signal, dtype=np.float64)
        if regime_signal.ndim == 1:
            if regime_signal.shape != (T,):
                raise ValueError("regime_signal must be [T] aligned to y_init")
        elif regime_signal.ndim == 2:
            if regime_signal.shape != (T, N):
                raise ValueError(f"regime_signal must be [T, N] == ({T}, {N}), got {regime_signal.shape}")
        else:
            raise ValueError("regime_signal must be [T] or [T, N]")

    for t in range(warmup - 1, T - 1):
        r_next = float(r[t + 1])
        c_next = float(cum[t + 1])
        if regime_signal is None:
            reg_idx = np.full((N,), int(np.digitize([r_next], bins_arr, right=True)[0]), dtype=np.int64)
        elif regime_signal.ndim == 1:
            reg_idx = np.full((N,), int(np.digitize([float(regime_signal[t + 1])], bins_arr, right=True)[0]), dtype=np.int64)
        else:
            reg_idx = np.digitize(regime_signal[t + 1], bins_arr, right=True).astype(np.int64, copy=False)
        if (reg_idx < 0).any() or (reg_idx >= R).any():
            raise ValueError("regime index out of range")
        ww = w[reg_idx, np.arange(N)]

        pred = ww[:, 0].copy()
        for j in range(k):
            pred += ww[:, 1 + j] * y[t - j]
        pred += ww[:, 1 + k] * r_next + ww[:, 2 + k] * c_next
        pred += (ww[:, 3 + k :] * exo[t + 1]).sum(axis=1)

        if delta_scale_f != 1.0:
            pred = y[t] + delta_scale_f * (pred - y[t])

        if md is not None:
            d = pred - y[t]
            d = np.clip(d, -md, md)
            pred = y[t] + d
        elif md_node is not None:
            d = pred - y[t]
            d = np.clip(d, -md_node, md_node)
            pred = y[t] + d

        y[t + 1] = pred

    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite rollout")
    return y.astype(np.float32, copy=False)


def rollout_regime_arkx_ensemble(
    *,
    models: list[tuple[np.ndarray, tuple[float, ...]]],
    weights: list[float] | None = None,
    y_init: np.ndarray,  # [T, N]
    rain: np.ndarray,  # [T] or [T, N2]
    warmup: int = 10,
) -> np.ndarray:
    """
    Step-wise ensemble rollout for multiple regime AR(k)+X models.

    At each timestep, each model predicts y[t+1] from the shared ensemble state,
    and the ensemble state is updated using the weighted mean prediction.
    """
    if not models:
        raise ValueError("models list must be non-empty")

    y_init = np.asarray(y_init, dtype=np.float64)
    if y_init.ndim != 2:
        raise ValueError("y_init must be [T, N]")
    T, N = y_init.shape

    if rain.ndim == 2:
        r = rain_scalar_from_2d(rain)
    else:
        r = np.asarray(rain, dtype=np.float64)
    if r.shape != (T,):
        raise ValueError("rain must be [T] aligned to y_init")
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    # Normalize model inputs + validate shapes
    w_list: list[np.ndarray] = []
    bins_list: list[np.ndarray] = []
    k_list: list[int] = []
    k_req = 1
    for ww, bb in models:
        w = np.asarray(ww, dtype=np.float64)
        if w.ndim != 3 or w.shape[1] != N:
            raise ValueError("each w must be [R, N, F]")
        F = int(w.shape[2])
        k = int(F - 3)
        if k < 1:
            raise ValueError("invalid feature dimension for regime_arkx")
        k_req = max(k_req, k)
        b = np.asarray(bb, dtype=np.float64)
        if b.ndim != 1 or b.size < 1:
            raise ValueError("bins must contain at least 1 threshold")
        w_list.append(w)
        bins_list.append(b)
        k_list.append(k)

    if warmup < k_req:
        raise ValueError(f"warmup must be >= {k_req} for the largest k in the ensemble")

    if weights is None:
        wts = np.full((len(w_list),), 1.0 / float(len(w_list)), dtype=np.float64)
    else:
        if len(weights) != len(w_list):
            raise ValueError("weights length must match models length")
        wts = np.asarray(weights, dtype=np.float64)
        if wts.ndim != 1 or wts.shape[0] != len(w_list):
            raise ValueError("weights must be a 1D list aligned to models")
        if not np.isfinite(wts).all() or (wts < 0.0).any():
            raise ValueError("weights must be finite and >= 0")
        s = float(wts.sum())
        if s <= 0.0:
            raise ValueError("weights must sum to > 0")
        wts = wts / s

    cum = np.cumsum(r)
    y = np.empty((T, N), dtype=np.float64)
    y[:warmup] = y_init[:warmup]
    if not np.isfinite(y[:warmup]).all():
        raise ValueError("non-finite warmup values")

    for t in range(warmup - 1, T - 1):
        r_next = float(r[t + 1])
        c_next = float(cum[t + 1])

        pred_sum = np.zeros((N,), dtype=np.float64)
        for wt, ww, bb, k in zip(wts.tolist(), w_list, bins_list, k_list):
            reg = int(np.digitize([r_next], bb, right=True)[0])
            if reg < 0 or reg >= int(ww.shape[0]):
                raise ValueError("regime index out of range")
            wreg = ww[reg]  # [N, F]

            pred = wreg[:, 0].copy()
            for j in range(k):
                pred += wreg[:, 1 + j] * y[t - j]
            pred += wreg[:, 1 + k] * r_next + wreg[:, 2 + k] * c_next
            pred_sum += float(wt) * pred

        y[t + 1] = pred_sum

    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite rollout")
    return y.astype(np.float32, copy=False)


def rollout_regime_arkx_ensemble_md(
    *,
    models: list[tuple[np.ndarray, tuple[float, ...], float | None]],
    weights: list[float] | None = None,
    y_init: np.ndarray,  # [T, N]
    rain: np.ndarray,  # [T] or [T, N2]
    warmup: int = 10,
) -> np.ndarray:
    """
    Step-wise ensemble rollout for multiple regime AR(k)+X models with optional max-delta clamps.

    Each model is a tuple: (w, bins, max_delta) where max_delta is a scalar clamp
    applied to (pred - y[t]) before ensembling.
    """
    if not models:
        raise ValueError("models list must be non-empty")

    y_init = np.asarray(y_init, dtype=np.float64)
    if y_init.ndim != 2:
        raise ValueError("y_init must be [T, N]")
    T, N = y_init.shape

    if rain.ndim == 2:
        r = rain_scalar_from_2d(rain)
    else:
        r = np.asarray(rain, dtype=np.float64)
    if r.shape != (T,):
        raise ValueError("rain must be [T] aligned to y_init")
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    w_list: list[np.ndarray] = []
    bins_list: list[np.ndarray] = []
    md_list: list[float | None] = []
    k_list: list[int] = []
    k_req = 1
    for ww, bb, md in models:
        w = np.asarray(ww, dtype=np.float64)
        if w.ndim != 3 or w.shape[1] != N:
            raise ValueError("each w must be [R, N, F]")
        k = int(w.shape[2] - 3)
        if k < 1:
            raise ValueError("invalid feature dimension for regime_arkx")
        k_req = max(k_req, k)
        b = np.asarray(bb, dtype=np.float64)
        if b.ndim != 1 or b.size < 1:
            raise ValueError("bins must contain at least 1 threshold")
        if md is None:
            md_f = None
        else:
            md_f = float(md)
            if not np.isfinite(md_f) or md_f <= 0.0:
                raise ValueError("max_delta must be finite and >0 when provided")
        w_list.append(w)
        bins_list.append(b)
        md_list.append(md_f)
        k_list.append(k)

    if warmup < k_req:
        raise ValueError(f"warmup must be >= {k_req} for the largest k in the ensemble")

    if weights is None:
        wts = np.full((len(w_list),), 1.0 / float(len(w_list)), dtype=np.float64)
    else:
        if len(weights) != len(w_list):
            raise ValueError("weights length must match models length")
        wts = np.asarray(weights, dtype=np.float64)
        if wts.ndim != 1 or wts.shape[0] != len(w_list):
            raise ValueError("weights must be a 1D list aligned to models")
        if not np.isfinite(wts).all() or (wts < 0.0).any():
            raise ValueError("weights must be finite and >= 0")
        s = float(wts.sum())
        if s <= 0.0:
            raise ValueError("weights must sum to > 0")
        wts = wts / s

    cum = np.cumsum(r)
    y = np.empty((T, N), dtype=np.float64)
    y[:warmup] = y_init[:warmup]
    if not np.isfinite(y[:warmup]).all():
        raise ValueError("non-finite warmup values")

    for t in range(warmup - 1, T - 1):
        r_next = float(r[t + 1])
        c_next = float(cum[t + 1])

        pred_sum = np.zeros((N,), dtype=np.float64)
        y_t = y[t]
        for wt, ww, bb, md, k in zip(wts.tolist(), w_list, bins_list, md_list, k_list):
            reg = int(np.digitize([r_next], bb, right=True)[0])
            if reg < 0 or reg >= int(ww.shape[0]):
                raise ValueError("regime index out of range")
            wreg = ww[reg]  # [N, F]

            pred = wreg[:, 0].copy()
            for j in range(k):
                pred += wreg[:, 1 + j] * y[t - j]
            pred += wreg[:, 1 + k] * r_next + wreg[:, 2 + k] * c_next

            if md is not None:
                d = pred - y_t
                d = np.clip(d, -md, md)
                pred = y_t + d

            pred_sum += float(wt) * pred

        y[t + 1] = pred_sum

    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite rollout")
    return y.astype(np.float32, copy=False)


def predict_model2_from_baseline_ckpts(
    ckpts: list[dict],
    *,
    graph2,
    mixed_mode: str,
    alpha_1d: float,
    alpha_2d: float,
    y1_init: np.ndarray,  # [T, N1]
    y2_init: np.ndarray,  # [T, N2]
    rain_2d: np.ndarray,  # [T, N2]
    q1_init: np.ndarray | None = None,  # [T, N1]
    vagg_init: np.ndarray | None = None,  # [T, N1]
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict Model-2 (1D and 2D) baselines from one or two baseline checkpoints.

    Supported:
    - single ckpt: kind "regime_arkx", "split_1d2d", "split_1d2d_coupled", "split_1d2d_coupled_inlet", "split_1d2d_coupled_local_inlet", "split_1d2d_coupled_local_storage_inlet", "split_1d2d_coupled_storage_inlet", or "split_1d2d_coupled_storage_inlet_statebin"
    - two ckpts with mixed_mode="weighted_split_ns": one non-split regime_arkx + one split baseline
    """
    if not ckpts:
        raise ValueError("ckpts must be non-empty")

    def _parse_max_delta(md_raw):
        if md_raw is None:
            return None
        if torch.is_tensor(md_raw):
            if md_raw.ndim == 0 or int(md_raw.numel()) == 1:
                md_f = float(md_raw.item())
                return None if md_f <= 0.0 else md_f
            return md_raw.numpy()
        md_f = float(md_raw)
        return None if md_f <= 0.0 else md_f

    if len(ckpts) == 1 or str(mixed_mode) == "single":
        ckpt = ckpts[0]
        kind2 = str(ckpt["model_2"].get("kind", ""))
        if kind2 == "split_1d2d":
            p1 = ckpt["model_2"]["parts"]["1d"]
            p2 = ckpt["model_2"]["parts"]["2d"]
            w1d = p1["w"].numpy()
            w2d = p2["w"].numpy()
            b1 = tuple(float(x) for x in p1["bins"])
            b2 = tuple(float(x) for x in p2["bins"])
            md = _parse_max_delta(p1.get("max_delta", 0.0))
            y1_pred = rollout_regime_arkx(w=w1d, bins=b1, y_init=y1_init, rain=rain_2d, warmup=warmup, max_delta=md)
            y2_pred = rollout_regime_arkx(w=w2d, bins=b2, y_init=y2_init, rain=rain_2d, warmup=warmup)
            return y1_pred, y2_pred
        if kind2 == "split_1d2d_coupled":
            p1 = ckpt["model_2"]["parts"]["1d"]
            p2 = ckpt["model_2"]["parts"]["2d"]
            w1d = p1["w"].numpy()
            w2d = p2["w"].numpy()
            b1 = tuple(float(x) for x in p1["bins"])
            b2 = tuple(float(x) for x in p2["bins"])
            md = _parse_max_delta(p1.get("max_delta", 0.0))
            y2_pred = rollout_regime_arkx(w=w2d, bins=b2, y_init=y2_init, rain=rain_2d, warmup=warmup)
            exo1d = build_coupled_1d_exo(
                y2_2d=y2_pred,
                rain_2d=rain_2d,
                area_2d=graph2.area_2d.cpu().numpy(),
                conn_src_1d=graph2.conn_src_1d.cpu().numpy(),
                conn_dst_2d=graph2.conn_dst_2d.cpu().numpy(),
                n_1d=int(graph2.n_1d),
            )
            y1_pred = rollout_regime_arkx_exo(w=w1d, bins=b1, y_init=y1_init, rain=rain_2d, exo=exo1d, warmup=warmup, max_delta=md)
            return y1_pred, y2_pred
        if kind2 == "split_1d2d_coupled_inlet":
            if q1_init is None:
                raise ValueError("q1_init is required for split_1d2d_coupled_inlet")
            p1 = ckpt["model_2"]["parts"]["1d"]
            p2 = ckpt["model_2"]["parts"]["2d"]
            y2_pred = rollout_regime_arkx(
                w=p2["w"].numpy(),
                bins=tuple(float(x) for x in p2["bins"]),
                y_init=y2_init,
                rain=rain_2d,
                warmup=warmup,
            )
            coupled_exo = build_coupled_1d_exo(
                y2_2d=y2_pred,
                rain_2d=rain_2d,
                area_2d=graph2.area_2d.cpu().numpy(),
                conn_src_1d=graph2.conn_src_1d.cpu().numpy(),
                conn_dst_2d=graph2.conn_dst_2d.cpu().numpy(),
                n_1d=int(graph2.n_1d),
            )
            q_model = p1["q_model"]
            q_pred = rollout_regime_arkx_exo(
                w=q_model["w"].numpy(),
                bins=tuple(float(x) for x in q_model["bins"]),
                y_init=q1_init,
                rain=rain_2d,
                exo=coupled_exo,
                warmup=warmup,
            )
            y_model = p1["y_model"]
            md = _parse_max_delta(y_model.get("max_delta", 0.0))
            y1_pred = rollout_regime_arkx_exo(
                w=y_model["w"].numpy(),
                bins=tuple(float(x) for x in y_model["bins"]),
                y_init=y1_init,
                rain=rain_2d,
                exo=build_inlet_augmented_exo(coupled_exo, q_pred),
                warmup=warmup,
                max_delta=md,
            )
            return y1_pred, y2_pred
        if kind2 in {"split_1d2d_coupled_local_inlet", "split_1d2d_coupled_local_inlet_statebin", "split_1d2d_coupled_local_storage_inlet"}:
            if q1_init is None:
                raise ValueError(f"q1_init is required for {kind2}")
            p1 = ckpt["model_2"]["parts"]["1d"]
            p2 = ckpt["model_2"]["parts"]["2d"]
            center_cell = p1["center_cell"].cpu().numpy() if torch.is_tensor(p1["center_cell"]) else np.asarray(p1["center_cell"], dtype=np.int64)
            neighbor_slots = (
                p1["neighbor_slots"].cpu().numpy() if torch.is_tensor(p1["neighbor_slots"]) else np.asarray(p1["neighbor_slots"], dtype=np.int64)
            )
            y2_pred = rollout_regime_arkx(
                w=p2["w"].numpy(),
                bins=tuple(float(x) for x in p2["bins"]),
                y_init=y2_init,
                rain=rain_2d,
                warmup=warmup,
            )
            local_exo = build_coupled_local_1d_exo(
                y2_2d=y2_pred,
                rain_2d=rain_2d,
                area_2d=graph2.area_2d.cpu().numpy(),
                conn_src_1d=graph2.conn_src_1d.cpu().numpy(),
                conn_dst_2d=graph2.conn_dst_2d.cpu().numpy(),
                n_1d=int(graph2.n_1d),
                center_cell=center_cell,
                neighbor_slots=neighbor_slots,
            )
            conn_area = None
            v_pred = None
            if kind2 == "split_1d2d_coupled_local_storage_inlet":
                if vagg_init is None:
                    raise ValueError(f"vagg_init is required for {kind2}")
                area_2d = graph2.area_2d.cpu().numpy()
                conn_src_1d = graph2.conn_src_1d.cpu().numpy()
                conn_dst_2d = graph2.conn_dst_2d.cpu().numpy()
                n_1d = int(graph2.n_1d)
                conn_area = connected_area_1d(
                    area_2d,
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=n_1d,
                )
                v_model = p1["v_model"]
                vmd_raw = v_model.get("max_delta", None)
                if torch.is_tensor(vmd_raw):
                    vmd = vmd_raw.numpy()
                else:
                    vmd = vmd_raw
                v_pred = rollout_regime_arkx_exo(
                    w=v_model["w"].numpy(),
                    bins=tuple(float(x) for x in v_model["bins"]),
                    y_init=vagg_init,
                    rain=rain_2d,
                    exo=local_exo,
                    warmup=warmup,
                    max_delta=vmd,
                )
            q_model = p1["q_model"]
            q_pred = rollout_regime_arkx_exo(
                w=q_model["w"].numpy(),
                bins=tuple(float(x) for x in q_model["bins"]),
                y_init=q1_init,
                rain=rain_2d,
                exo=build_storage_augmented_exo(local_exo, v_pred, conn_area) if v_pred is not None and conn_area is not None else local_exo,
                warmup=warmup,
            )
            y_model = p1["y_model"]
            md = _parse_max_delta(y_model.get("max_delta", 0.0))
            regime_signal = None
            if kind2 == "split_1d2d_coupled_local_inlet_statebin":
                regime_src = str(y_model.get("regime_source", ""))
                if regime_src != "inlet_positive":
                    raise ValueError(f"unsupported y_model regime_source: {regime_src}")
                regime_signal = np.maximum(q_pred, 0.0).astype(np.float32, copy=False)
            y1_pred = rollout_regime_arkx_exo(
                w=y_model["w"].numpy(),
                bins=tuple(float(x) for x in y_model["bins"]),
                y_init=y1_init,
                rain=rain_2d,
                exo=(
                    build_inlet_storage_augmented_exo(local_exo, q_pred, v_pred, conn_area)
                    if v_pred is not None and conn_area is not None
                    else build_inlet_augmented_exo(local_exo, q_pred)
                ),
                warmup=warmup,
                regime_signal=regime_signal,
                max_delta=md,
            )
            return y1_pred, y2_pred
        if kind2 in {"split_1d2d_coupled_storage_inlet", "split_1d2d_coupled_storage_inlet_statebin"}:
            if q1_init is None:
                raise ValueError(f"q1_init is required for {kind2}")
            if vagg_init is None:
                raise ValueError(f"vagg_init is required for {kind2}")
            p1 = ckpt["model_2"]["parts"]["1d"]
            p2 = ckpt["model_2"]["parts"]["2d"]
            y2_pred = rollout_regime_arkx(
                w=p2["w"].numpy(),
                bins=tuple(float(x) for x in p2["bins"]),
                y_init=y2_init,
                rain=rain_2d,
                warmup=warmup,
            )
            area_2d = graph2.area_2d.cpu().numpy()
            conn_src_1d = graph2.conn_src_1d.cpu().numpy()
            conn_dst_2d = graph2.conn_dst_2d.cpu().numpy()
            n_1d = int(graph2.n_1d)
            coupled_exo = build_coupled_1d_exo(
                y2_2d=y2_pred,
                rain_2d=rain_2d,
                area_2d=area_2d,
                conn_src_1d=conn_src_1d,
                conn_dst_2d=conn_dst_2d,
                n_1d=n_1d,
            )
            conn_area = connected_area_1d(
                area_2d,
                conn_src_1d=conn_src_1d,
                conn_dst_2d=conn_dst_2d,
                n_1d=n_1d,
            )
            v_model = p1["v_model"]
            vmd_raw = v_model.get("max_delta", None)
            if torch.is_tensor(vmd_raw):
                vmd = vmd_raw.numpy()
            else:
                vmd = vmd_raw
            v_pred = rollout_regime_arkx_exo(
                w=v_model["w"].numpy(),
                bins=tuple(float(x) for x in v_model["bins"]),
                y_init=vagg_init,
                rain=rain_2d,
                exo=coupled_exo,
                warmup=warmup,
                max_delta=vmd,
            )
            q_model = p1["q_model"]
            q_pred = rollout_regime_arkx_exo(
                w=q_model["w"].numpy(),
                bins=tuple(float(x) for x in q_model["bins"]),
                y_init=q1_init,
                rain=rain_2d,
                exo=build_storage_augmented_exo(coupled_exo, v_pred, conn_area),
                warmup=warmup,
            )
            y_model = p1["y_model"]
            md = _parse_max_delta(y_model.get("max_delta", 0.0))
            regime_signal = None
            if kind2 == "split_1d2d_coupled_storage_inlet_statebin":
                regime_src = str(y_model.get("regime_source", ""))
                if regime_src != "storage_depth":
                    raise ValueError(f"unsupported y_model regime_source: {regime_src}")
                regime_signal = (v_pred / np.maximum(conn_area[None, :], 1.0)).astype(np.float32, copy=False)
            y1_pred = rollout_regime_arkx_exo(
                w=y_model["w"].numpy(),
                bins=tuple(float(x) for x in y_model["bins"]),
                y_init=y1_init,
                rain=rain_2d,
                exo=build_inlet_storage_augmented_exo(coupled_exo, q_pred, v_pred, conn_area),
                warmup=warmup,
                regime_signal=regime_signal,
                max_delta=md,
            )
            return y1_pred, y2_pred

        # Non-split baseline
        w_full = ckpt["model_2"]["w"].numpy()
        bins_full = tuple(float(x) for x in ckpt["model_2"]["bins"])
        y_init = np.concatenate([y1_init, y2_init], axis=1)
        y_pred = rollout_regime_arkx(w=w_full, bins=bins_full, y_init=y_init, rain=rain_2d, warmup=warmup)
        n1 = int(graph2.n_1d)
        return y_pred[:, :n1], y_pred[:, n1:]

    mode = str(mixed_mode)
    if mode != "weighted_split_ns":
        raise ValueError(f"unsupported mixed_mode: {mode}")
    if len(ckpts) != 2:
        raise ValueError("weighted_split_ns requires exactly 2 checkpoints")

    ckpt_split = None
    ckpt_ns = None
    for ckpt in ckpts:
        kk = str(ckpt["model_2"].get("kind", ""))
        if kk.startswith("split_1d2d"):
            ckpt_split = ckpt
        elif kk == "regime_arkx":
            ckpt_ns = ckpt
    if ckpt_split is None or ckpt_ns is None:
        raise ValueError("weighted_split_ns requires one split baseline and one regime_arkx checkpoint")

    a1 = float(alpha_1d)
    a2 = float(alpha_2d)
    if not np.isfinite(a1) or not np.isfinite(a2) or a1 < 0.0 or a1 > 1.0 or a2 < 0.0 or a2 > 1.0:
        raise ValueError("alpha_1d/alpha_2d must be finite and in [0, 1]")

    n1 = int(graph2.n_1d)
    w_full = ckpt_ns["model_2"]["w"].numpy()
    b_ns = tuple(float(x) for x in ckpt_ns["model_2"]["bins"])
    w_ns1 = w_full[:, :n1, :]
    w_ns2 = w_full[:, n1:, :]

    split_kind = str(ckpt_split["model_2"].get("kind", ""))
    if split_kind == "split_1d2d":
        p1 = ckpt_split["model_2"]["parts"]["1d"]
        p2 = ckpt_split["model_2"]["parts"]["2d"]
        w_split1 = p1["w"].numpy()
        w_split2 = p2["w"].numpy()
        b1 = tuple(float(x) for x in p1["bins"])
        b2 = tuple(float(x) for x in p2["bins"])

        md_raw = p1.get("max_delta", 0.0)
        if torch.is_tensor(md_raw):
            md_f = float(md_raw.item())
        else:
            md_f = float(md_raw)
        md = None if md_f <= 0.0 else md_f

        y1_pred = rollout_regime_arkx_ensemble_md(
            models=[(w_ns1, b_ns, None), (w_split1, b1, md)],
            weights=[1.0 - a1, a1],
            y_init=y1_init,
            rain=rain_2d,
            warmup=warmup,
        )
        y2_pred = rollout_regime_arkx_ensemble(
            models=[(w_ns2, b_ns), (w_split2, b2)],
            weights=[1.0 - a2, a2],
            y_init=y2_init,
            rain=rain_2d,
            warmup=warmup,
        )
        return y1_pred, y2_pred

    y1_ns = rollout_regime_arkx(w=w_ns1, bins=b_ns, y_init=y1_init, rain=rain_2d, warmup=warmup)
    y2_ns = rollout_regime_arkx(w=w_ns2, bins=b_ns, y_init=y2_init, rain=rain_2d, warmup=warmup)
    y1_split, y2_split = predict_model2_from_baseline_ckpts(
        [ckpt_split],
        graph2=graph2,
        mixed_mode="single",
        alpha_1d=alpha_1d,
        alpha_2d=alpha_2d,
        y1_init=y1_init,
        y2_init=y2_init,
        rain_2d=rain_2d,
        q1_init=q1_init,
        vagg_init=vagg_init,
        warmup=warmup,
    )
    return ((1.0 - a1) * y1_ns + a1 * y1_split).astype(np.float32, copy=False), (
        (1.0 - a2) * y2_ns + a2 * y2_split
    ).astype(np.float32, copy=False)
