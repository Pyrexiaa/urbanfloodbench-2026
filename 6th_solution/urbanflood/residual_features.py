# #####
# residual_features.py
# #####

# urbanflood/residual_features.py
from __future__ import annotations

import numpy as np
import torch

from urbanflood.baseline import aggregate_2d_to_1d_sum, rain_scalar_from_2d


def neighbor_mean_time_major(
    y: np.ndarray,  # [T, N]
    *,
    src: np.ndarray,  # [E]
    dst: np.ndarray,  # [E]
) -> np.ndarray:
    """
    Incoming-neighbor mean for each node at each timestep (time-major).

    This is intentionally simple (loop over edges) since the 1D pipe graph is small.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y must be [T, N]")
    T, N = y.shape

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError("src/dst must be 1D arrays of same shape")
    if src.size == 0:
        return y.copy()
    if (src < 0).any() or (src >= N).any() or (dst < 0).any() or (dst >= N).any():
        raise ValueError("edge indices out of range")

    out = np.zeros((T, N), dtype=np.float32)
    for s, d in zip(src.tolist(), dst.tolist()):
        out[:, d] += y[:, s]

    deg = np.bincount(dst, minlength=N).astype(np.float32)
    mean = out / np.maximum(deg[None, :], 1.0)
    if (deg == 0).any():
        mean[:, deg == 0] = y[:, deg == 0]
    return mean


def neighbor_mean_time_major_torch(
    x: torch.Tensor,  # [T, N] or [T, N, F]
    *,
    edge_index: torch.LongTensor,  # [2, E]
    deg_inv: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    """
    Vectorized incoming-neighbor mean (time-major) for torch tensors.

    Works on CPU or CUDA. For nodes with zero indegree, the output falls back
    to the input (identity), matching `neighbor_mean_time_major` behavior.
    """
    if x.ndim == 2:
        x3 = x.unsqueeze(-1)
        squeeze = True
    else:
        x3 = x
        squeeze = False
    if x3.ndim != 3:
        raise ValueError("x must be [T, N] or [T, N, F]")
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must be [2, E]")

    T, N, F = x3.shape
    src = edge_index[0]
    dst = edge_index[1]
    if src.ndim != 1 or dst.ndim != 1 or int(src.shape[0]) != int(dst.shape[0]):
        raise ValueError("edge_index must be [2, E] with 1D rows")

    out = torch.zeros((T, N, F), device=x3.device, dtype=x3.dtype)
    src2 = src.to(device=x3.device)
    dst2 = dst.to(device=x3.device)
    out.index_add_(1, dst2, x3.index_select(1, src2))

    deg = torch.bincount(dst2, minlength=int(N)).to(torch.float32)
    if deg_inv is None:
        deg_inv2 = (1.0 / torch.clamp(deg, min=1.0)).to(device=x3.device)
    else:
        if deg_inv.ndim != 1 or int(deg_inv.shape[0]) != int(N):
            raise ValueError("deg_inv must be [N]")
        deg_inv2 = deg_inv.to(device=x3.device, dtype=torch.float32)

    out = out * deg_inv2.to(dtype=x3.dtype)[None, :, None]
    if bool((deg == 0).any()):
        mask0 = deg == 0
        out[:, mask0, :] = x3[:, mask0, :]

    if squeeze:
        return out.squeeze(-1)
    return out


def neighbor_weighted_posneg_mean_time_major(
    y: np.ndarray,  # [T, N]
    *,
    src: np.ndarray,  # [E]
    dst: np.ndarray,  # [E]
    w: np.ndarray,  # [E]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted incoming positive/negative head-difference means for each node:

      dh = y[src] - y[dst]
      pos_mean[dst] = sum(w * relu(dh)) / sum(w)
      neg_mean[dst] = sum(w * relu(-dh)) / sum(w)
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y must be [T, N]")
    T, N = y.shape

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError("src/dst must be 1D arrays of same shape")
    if w.shape != src.shape:
        raise ValueError("w must align to src/dst")
    if src.size == 0:
        z = np.zeros((T, N), dtype=np.float32)
        return z, z
    if (src < 0).any() or (src >= N).any() or (dst < 0).any() or (dst >= N).any():
        raise ValueError("edge indices out of range")

    pos = np.zeros((T, N), dtype=np.float32)
    neg = np.zeros((T, N), dtype=np.float32)
    for s, d, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
        if not np.isfinite(ww) or ww == 0.0:
            continue
        dh = y[:, s] - y[:, d]
        pos[:, d] += ww * np.maximum(dh, 0.0)
        neg[:, d] += ww * np.maximum(-dh, 0.0)

    wsum = np.bincount(dst, weights=w.astype(np.float64), minlength=N).astype(np.float32)
    denom = np.maximum(wsum[None, :], 1e-6)
    return pos / denom, neg / denom


def neighbor_weighted_posneg_summax_time_major(
    y: np.ndarray,  # [T, N]
    *,
    src: np.ndarray,  # [E]
    dst: np.ndarray,  # [E]
    w: np.ndarray,  # [E]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted incoming positive/negative head-difference sums and maxes for each node:

      dh = y[src] - y[dst]
      pos_sum[dst] = sum(w * relu(dh))
      neg_sum[dst] = sum(w * relu(-dh))
      pos_max[dst] = max(w * relu(dh))
      neg_max[dst] = max(w * relu(-dh))
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y must be [T, N]")
    T, N = y.shape

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError("src/dst must be 1D arrays of same shape")
    if w.shape != src.shape:
        raise ValueError("w must align to src/dst")
    if src.size == 0:
        z = np.zeros((T, N), dtype=np.float32)
        return z, z, z, z
    if (src < 0).any() or (src >= N).any() or (dst < 0).any() or (dst >= N).any():
        raise ValueError("edge indices out of range")

    pos_sum = np.zeros((T, N), dtype=np.float32)
    neg_sum = np.zeros((T, N), dtype=np.float32)
    pos_max = np.zeros((T, N), dtype=np.float32)
    neg_max = np.zeros((T, N), dtype=np.float32)

    for s, d, ww in zip(src.tolist(), dst.tolist(), w.tolist()):
        if not np.isfinite(ww) or ww == 0.0:
            continue
        dh = y[:, s] - y[:, d]
        wpos = ww * np.maximum(dh, 0.0)
        wneg = ww * np.maximum(-dh, 0.0)
        pos_sum[:, d] += wpos
        neg_sum[:, d] += wneg
        pos_max[:, d] = np.maximum(pos_max[:, d], wpos)
        neg_max[:, d] = np.maximum(neg_max[:, d], wneg)

    return pos_sum, neg_sum, pos_max, neg_max


def augment_dyn_features_2d_v2_nbrmean(
    x: torch.Tensor,  # [T, N2, 5]
    *,
    edge_index: torch.LongTensor,
    edge_deg_inv: torch.Tensor | None = None,
    include_diff: bool = True,
) -> torch.Tensor:
    """
    Augment base 2D dynamic features with neighbor-mean coupling terms.

    Base layout (5):
      [depth2, ddepth2, rain, cum, t_norm]

    Adds:
      nbr_mean(depth2), nbr_mean(ddepth2)          (+2)
      depth2 - nbr_mean(depth2), ddepth2 - ...     (+2, optional)
    """
    if x.ndim != 3 or int(x.shape[-1]) != 5:
        raise ValueError("x must be [T, N2, 5]")
    base = x[..., :2]  # [T, N2, 2]
    nbr = neighbor_mean_time_major_torch(base, edge_index=edge_index, deg_inv=edge_deg_inv)  # [T, N2, 2]
    parts = [x, nbr]
    if bool(include_diff):
        parts.append(base - nbr)
    return torch.cat(parts, dim=-1)


def build_static_features_m2_1d(graph2) -> torch.Tensor:
    """
    Static augmentation for Model 2 1D nodes (version 1).

    Deterministic features:
    - z-scored 1D node static features
    - mean incoming/outgoing 1D edge attributes (directed)
    - connected 2D node static features (deg 0/1 in this dataset)
    - coupling edge attributes (dx, dy, dist, area) for the connected cell
    """
    node1 = graph2.node_static_1d.float().cpu()
    n1 = int(graph2.n_1d)
    parts = [node1]

    # 1D in/out edge means
    src = graph2.edge_index_1d[0].cpu()
    dst = graph2.edge_index_1d[1].cpu()
    eattr = graph2.edge_attr_1d.float().cpu()
    F = int(eattr.shape[1])

    in_sum = torch.zeros((n1, F), dtype=torch.float32)
    in_sum.index_add_(0, dst, eattr)
    in_deg = torch.bincount(dst, minlength=n1).to(torch.float32).unsqueeze(1)
    in_mean = in_sum / torch.clamp(in_deg, min=1.0)
    parts.append(in_mean)

    out_sum = torch.zeros((n1, F), dtype=torch.float32)
    out_sum.index_add_(0, src, eattr)
    out_deg = torch.bincount(src, minlength=n1).to(torch.float32).unsqueeze(1)
    out_mean = out_sum / torch.clamp(out_deg, min=1.0)
    parts.append(out_mean)

    # Connected 2D static
    conn_src = graph2.conn_src_1d.cpu()
    conn_dst = graph2.conn_dst_2d.cpu()
    n2s = int(graph2.node_static_2d.shape[1])
    conn_map = torch.full((n1,), -1, dtype=torch.long)
    conn_map[conn_src] = conn_dst
    conn_static = torch.zeros((n1, n2s), dtype=torch.float32)
    mask = conn_map >= 0
    if bool(mask.any()):
        conn_static[mask] = graph2.node_static_2d.float().cpu().index_select(0, conn_map[mask])
    parts.append(conn_static)

    # Coupling edge attrs (first Nc rows are 1D->2D)
    nce = int(graph2.edge_attr_c.shape[1])
    Nc = int(conn_src.numel())
    c_attr_fwd = graph2.edge_attr_c[:Nc].float().cpu()
    conn_edge = torch.zeros((n1, nce), dtype=torch.float32)
    conn_edge.index_copy_(0, conn_src, c_attr_fwd)
    parts.append(conn_edge)

    return torch.cat(parts, dim=1)


def build_static_features_m2_1d_v2(graph2) -> torch.Tensor:
    """
    Static augmentation for Model 2 1D nodes (version 2).

    Adds pipe-direction boundary indicators:
    - indeg_norm, outdeg_norm, deg_norm
    - is_source (indeg==0), is_sink (outdeg==0)
    """
    base = build_static_features_m2_1d(graph2)

    n1 = int(graph2.n_1d)
    E1 = int(graph2.edge_index_1d.shape[1] // 2)  # first half are original pipe directions
    src = graph2.edge_index_1d[0, :E1].cpu()
    dst = graph2.edge_index_1d[1, :E1].cpu()

    indeg = torch.bincount(dst, minlength=n1).to(torch.float32)
    outdeg = torch.bincount(src, minlength=n1).to(torch.float32)
    deg = indeg + outdeg

    indeg_n = indeg / torch.clamp(indeg.max(), min=1.0)
    outdeg_n = outdeg / torch.clamp(outdeg.max(), min=1.0)
    deg_n = deg / torch.clamp(deg.max(), min=1.0)
    is_source = (indeg == 0).to(torch.float32)
    is_sink = (outdeg == 0).to(torch.float32)

    extra = torch.stack([indeg_n, outdeg_n, deg_n, is_source, is_sink], dim=1)  # [N1, 5]
    return torch.cat([base, extra], dim=1)


def build_dyn_features_1d_v1(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1] 0/1
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 1), used in stages 1-3.

    Feature layout:
      [depth1, ddepth1, depth2, ddepth2, gap, gap_pos, gap_neg, rain, cum, t_norm]
    + incoming neighbor aggregates (if nbr_src/dst provided):
      [nbr_depth1, nbr_ddepth1, nbr_gap]
    """
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    if y1_base.shape != y2_agg.shape:
        raise ValueError("y1_base and y2_agg must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    r = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    cum = np.cumsum(r).astype(np.float32, copy=False)

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    if inv.shape != (N,) or bed.shape != (N,):
        raise ValueError("invert/bed shapes mismatch")

    has = np.asarray(has_conn_1d, dtype=np.float32)
    if has.shape != (N,):
        raise ValueError("has_conn_1d shape mismatch")
    has = np.where(has > 0.0, 1.0, 0.0)

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    Tp = int(idx.shape[0])

    y1_t = y1_base[idx]
    y1_tm1 = y1_base[idx - 1]
    dy1 = y1_t - y1_tm1

    y2_t = y2_agg[idx] * has[None, :]
    y2_tm1 = y2_agg[idx - 1] * has[None, :]
    dy2 = y2_t - y2_tm1

    depth1_full = (y1_base - inv[None, :]).astype(np.float32, copy=False)
    depth1 = depth1_full[idx]
    ddepth1 = dy1
    depth2 = y2_t - bed[None, :]
    ddepth2 = dy2

    gap = (y2_t - y1_t) * has[None, :]
    gpos = np.maximum(gap, 0.0)
    gneg = np.maximum(-gap, 0.0)

    rain_t = r[idx][:, None].repeat(N, axis=1)
    cum_t = cum[idx][:, None].repeat(N, axis=1)
    t_norm = (idx.astype(np.float32) / float(max(1, T - 1)))[:, None].repeat(N, axis=1)

    use_nbr = (nbr_src is not None) or (nbr_dst is not None)
    if use_nbr:
        if nbr_src is None or nbr_dst is None:
            raise ValueError("nbr_src/nbr_dst must both be provided or both be None")
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_src, dst=nbr_dst)[idx]
        nbr_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_src, dst=nbr_dst)[idx]
        nbr_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_src, dst=nbr_dst)[idx]
        dyn_dim = 10 + 3
    else:
        nbr_depth1 = None
        nbr_ddepth1 = None
        nbr_gap = None
        dyn_dim = 10

    X = np.empty((Tp, N, int(dyn_dim)), dtype=np.float32)
    pos = 0
    X[..., pos] = depth1
    pos += 1
    X[..., pos] = ddepth1
    pos += 1
    X[..., pos] = depth2
    pos += 1
    X[..., pos] = ddepth2
    pos += 1
    X[..., pos] = gap
    pos += 1
    X[..., pos] = gpos
    pos += 1
    X[..., pos] = gneg
    pos += 1
    X[..., pos] = rain_t
    pos += 1
    X[..., pos] = cum_t
    pos += 1
    X[..., pos] = t_norm
    pos += 1

    if use_nbr:
        X[..., pos] = nbr_depth1
        pos += 1
        X[..., pos] = nbr_ddepth1
        pos += 1
        X[..., pos] = nbr_gap
        pos += 1

    if int(pos) != int(dyn_dim):
        raise RuntimeError("internal dyn feature dim mismatch")
    return torch.from_numpy(X)


def build_dyn_features_1d_v3(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1] (connected cell head)
    y2_knn_mean: np.ndarray,  # [T, N1] (depth)
    y2_knn_max: np.ndarray,  # [T, N1] (depth)
    rain_2d: np.ndarray,  # [T, N2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1] 0/1
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 3), used in stage-6 specialist booster.

    Base layout (14):
      [depth1, ddepth1, depth2, ddepth2, gap, gap_pos, gap_neg,
       y2m, dy2m, y2x, dy2x, rain, cum, t_norm]
    + optional rain_lags
    + optional neighbor aggregates (incoming, and outgoing if use_out_nbr):
      [nbr_depth1, nbr_ddepth1, nbr_gap] (+ outgoing variants)
    """
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != y2m.shape or y1_base.shape != y2x.shape:
        raise ValueError("y1_base/y2_agg/y2_knn_mean/y2_knn_max must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    r = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    cum = np.cumsum(r).astype(np.float32, copy=False)

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    if inv.shape != (N,) or bed.shape != (N,) or has.shape != (N,):
        raise ValueError("invert/bed/has shapes mismatch")
    has = np.where(has > 0.0, 1.0, 0.0)

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    Tp = int(idx.shape[0])

    y1_t = y1_base[idx]
    y1_tm1 = y1_base[idx - 1]
    dy1 = y1_t - y1_tm1

    y2_t = y2_agg[idx] * has[None, :]
    y2_tm1 = y2_agg[idx - 1] * has[None, :]
    dy2 = y2_t - y2_tm1

    depth1_full = (y1_base - inv[None, :]).astype(np.float32, copy=False)
    depth1 = depth1_full[idx]
    ddepth1 = dy1
    depth2 = y2_t - bed[None, :]
    ddepth2 = dy2

    gap = (y2_t - y1_t) * has[None, :]
    gpos = np.maximum(gap, 0.0)
    gneg = np.maximum(-gap, 0.0)

    y2m_t = y2m[idx]
    dy2m = y2m_t - y2m[idx - 1]
    y2x_t = y2x[idx]
    dy2x = y2x_t - y2x[idx - 1]

    rain_lags = int(rain_lags)
    if rain_lags < 0:
        raise ValueError("rain_lags must be >= 0")
    if rain_lags and int(warmup) < int(rain_lags):
        raise ValueError("warmup must be >= rain_lags")

    if rain_lags:
        r_lags = []
        for j in range(1, int(rain_lags) + 1):
            r_lags.append(r[idx - j])
        r_lags = np.stack(r_lags, axis=1).astype(np.float32, copy=False)  # [T', L]
    else:
        r_lags = None

    use_out_nbr = bool(use_out_nbr)
    use_nbr_in = (nbr_src is not None) or (nbr_dst is not None)
    if use_out_nbr and not use_nbr_in:
        raise ValueError("use_out_nbr=True requires nbr_src/nbr_dst")

    if use_nbr_in:
        if nbr_src is None or nbr_dst is None:
            raise ValueError("nbr_src/nbr_dst must both be provided or both be None")
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_src, dst=nbr_dst)[idx]
        nbr_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_src, dst=nbr_dst)[idx]
        nbr_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_src, dst=nbr_dst)[idx]
    else:
        nbr_depth1 = None
        nbr_ddepth1 = None
        nbr_gap = None

    if use_out_nbr:
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_out_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_dst, dst=nbr_src)[idx]
    else:
        nbr_out_depth1 = None
        nbr_out_ddepth1 = None
        nbr_out_gap = None

    base_dim = 14
    if r_lags is not None:
        base_dim += int(rain_lags)
    if use_nbr_in:
        base_dim += 3
    if use_out_nbr:
        base_dim += 3

    rain_t = r[idx][:, None].repeat(N, axis=1)
    cum_t = cum[idx][:, None].repeat(N, axis=1)
    t_norm = (idx.astype(np.float32) / float(max(1, T - 1)))[:, None].repeat(N, axis=1)

    X = np.empty((Tp, N, int(base_dim)), dtype=np.float32)
    pos = 0
    for arr in (
        depth1,
        ddepth1,
        depth2,
        ddepth2,
        gap,
        gpos,
        gneg,
        y2m_t,
        dy2m,
        y2x_t,
        dy2x,
        rain_t,
        cum_t,
        t_norm,
    ):
        X[..., pos] = arr
        pos += 1

    if r_lags is not None:
        for j in range(int(rain_lags)):
            X[..., pos] = r_lags[:, j][:, None].repeat(N, axis=1)
            pos += 1

    if use_nbr_in:
        X[..., pos] = nbr_depth1
        pos += 1
        X[..., pos] = nbr_ddepth1
        pos += 1
        X[..., pos] = nbr_gap
        pos += 1

    if use_out_nbr:
        X[..., pos] = nbr_out_depth1
        pos += 1
        X[..., pos] = nbr_out_ddepth1
        pos += 1
        X[..., pos] = nbr_out_gap
        pos += 1

    if int(pos) != int(base_dim):
        raise RuntimeError("internal dyn feature dim mismatch (v3)")
    return torch.from_numpy(X)


def build_dyn_features_1d_v8(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1] (connected cell)
    y2_knn_mean: np.ndarray,  # [T, N1] (depth)
    y2_knn_max: np.ndarray,  # [T, N1] (depth)
    rain_2d: np.ndarray,  # [T, N2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1] 0/1
    lap_src: np.ndarray,  # [E] (directed, original pipe direction)
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 8), stage-4 booster.

    Layout (base):
      [depth1, ddepth1, depth2, ddepth2, gap, gap_pos, gap_neg,
       y2m, dy2m, y2x, dy2x,
       in_h_pos, in_h_neg, in_d_pos, in_d_neg,
       out_h_pos, out_h_neg, out_d_pos, out_d_neg,
       rain, cum, t_norm]
    + optional rain_lags / neighbor aggregates (in+out)
    """
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != y2m.shape or y1_base.shape != y2x.shape:
        raise ValueError("y1_base/y2_agg/y2_knn_mean/y2_knn_max must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    r = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    cum = np.cumsum(r).astype(np.float32, copy=False)

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    has = np.where(has > 0.0, 1.0, 0.0)

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    Tp = int(idx.shape[0])

    y1_t = y1_base[idx]
    y1_tm1 = y1_base[idx - 1]
    dy1 = y1_t - y1_tm1

    y2_t = y2_agg[idx] * has[None, :]
    y2_tm1 = y2_agg[idx - 1] * has[None, :]
    dy2 = y2_t - y2_tm1

    depth1_full = (y1_base - inv[None, :]).astype(np.float32, copy=False)
    depth1 = depth1_full[idx]
    ddepth1 = dy1
    depth2 = y2_t - bed[None, :]
    ddepth2 = dy2

    gap = (y2_t - y1_t) * has[None, :]
    gpos = np.maximum(gap, 0.0)
    gneg = np.maximum(-gap, 0.0)

    y2m_t = y2m[idx]
    dy2m = y2m_t - y2m[idx - 1]
    y2x_t = y2x[idx]
    dy2x = y2x_t - y2x[idx - 1]

    lap_src = np.asarray(lap_src, dtype=np.int64)
    lap_dst = np.asarray(lap_dst, dtype=np.int64)
    lap_w = np.asarray(lap_w, dtype=np.float32)
    if lap_src.shape != lap_dst.shape or lap_src.shape != lap_w.shape:
        raise ValueError("lap_src/lap_dst/lap_w must have same shape")

    in_h_pos_full, in_h_neg_full = neighbor_weighted_posneg_mean_time_major(y1_base, src=lap_src, dst=lap_dst, w=lap_w)
    in_d_pos_full, in_d_neg_full = neighbor_weighted_posneg_mean_time_major(
        depth1_full, src=lap_src, dst=lap_dst, w=lap_w
    )

    # Outgoing: reverse edges, then swap pos/neg (since dh flips sign).
    out_h_pos_rev, out_h_neg_rev = neighbor_weighted_posneg_mean_time_major(y1_base, src=lap_dst, dst=lap_src, w=lap_w)
    out_d_pos_rev, out_d_neg_rev = neighbor_weighted_posneg_mean_time_major(
        depth1_full, src=lap_dst, dst=lap_src, w=lap_w
    )
    out_h_pos_full = out_h_neg_rev
    out_h_neg_full = out_h_pos_rev
    out_d_pos_full = out_d_neg_rev
    out_d_neg_full = out_d_pos_rev

    in_h_pos = in_h_pos_full[idx]
    in_h_neg = in_h_neg_full[idx]
    in_d_pos = in_d_pos_full[idx]
    in_d_neg = in_d_neg_full[idx]
    out_h_pos = out_h_pos_full[idx]
    out_h_neg = out_h_neg_full[idx]
    out_d_pos = out_d_pos_full[idx]
    out_d_neg = out_d_neg_full[idx]

    rain_lags = int(rain_lags)
    if rain_lags < 0:
        raise ValueError("rain_lags must be >= 0")
    if rain_lags and int(warmup) < int(rain_lags):
        raise ValueError("warmup must be >= rain_lags")

    if rain_lags:
        r_lags = []
        for j in range(1, int(rain_lags) + 1):
            r_lags.append(r[idx - j])
        r_lags = np.stack(r_lags, axis=1).astype(np.float32, copy=False)  # [T', L]
    else:
        r_lags = None

    use_out_nbr = bool(use_out_nbr)
    use_nbr_in = (nbr_src is not None) or (nbr_dst is not None)
    if use_out_nbr and not use_nbr_in:
        raise ValueError("use_out_nbr=True requires nbr_src/nbr_dst")

    if use_nbr_in:
        if nbr_src is None or nbr_dst is None:
            raise ValueError("nbr_src/nbr_dst must both be provided or both be None")
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_src, dst=nbr_dst)[idx]
        nbr_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_src, dst=nbr_dst)[idx]
        nbr_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_src, dst=nbr_dst)[idx]
    else:
        nbr_depth1 = None
        nbr_ddepth1 = None
        nbr_gap = None

    if use_out_nbr:
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_out_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_dst, dst=nbr_src)[idx]
    else:
        nbr_out_depth1 = None
        nbr_out_ddepth1 = None
        nbr_out_gap = None

    # Base dims: 22
    base_dim = 22
    if rain_lags:
        base_dim += int(rain_lags)
    if use_nbr_in:
        base_dim += 3
    if use_out_nbr:
        base_dim += 3

    rain_t = r[idx][:, None].repeat(N, axis=1)
    cum_t = cum[idx][:, None].repeat(N, axis=1)
    t_norm = (idx.astype(np.float32) / float(max(1, T - 1)))[:, None].repeat(N, axis=1)

    X = np.empty((Tp, N, int(base_dim)), dtype=np.float32)
    pos = 0
    for arr in (
        depth1,
        ddepth1,
        depth2,
        ddepth2,
        gap,
        gpos,
        gneg,
        y2m_t,
        dy2m,
        y2x_t,
        dy2x,
        in_h_pos,
        in_h_neg,
        in_d_pos,
        in_d_neg,
        out_h_pos,
        out_h_neg,
        out_d_pos,
        out_d_neg,
        rain_t,
        cum_t,
        t_norm,
    ):
        X[..., pos] = arr
        pos += 1

    if r_lags is not None:
        for j in range(int(rain_lags)):
            X[..., pos] = r_lags[:, j][:, None].repeat(N, axis=1)
            pos += 1

    if use_nbr_in:
        X[..., pos] = nbr_depth1
        pos += 1
        X[..., pos] = nbr_ddepth1
        pos += 1
        X[..., pos] = nbr_gap
        pos += 1

    if use_out_nbr:
        X[..., pos] = nbr_out_depth1
        pos += 1
        X[..., pos] = nbr_out_ddepth1
        pos += 1
        X[..., pos] = nbr_out_gap
        pos += 1

    if int(pos) != int(base_dim):
        raise RuntimeError("internal dyn feature dim mismatch (v8)")
    return torch.from_numpy(X)


def build_dyn_features_1d_v9(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1] (depth)
    y2_knn_max: np.ndarray,  # [T, N1] (depth)
    rain_2d: np.ndarray,  # [T, N2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E] (directed)
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 9), stage-5 confluence booster.

    Extends v8 with:
    - weighted in/out gradient sums and maxes
    - surcharge hinge: relu(depth1 - diam_max)
    - static pipe-weight sums (in/out) normalized, broadcast over time
    """
    # Start with the same core computations as v8.
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != y2m.shape or y1_base.shape != y2x.shape:
        raise ValueError("y1_base/y2_agg/y2_knn_mean/y2_knn_max must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    r = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    cum = np.cumsum(r).astype(np.float32, copy=False)

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    has = np.where(has > 0.0, 1.0, 0.0)

    diam = np.asarray(diam_max_1d, dtype=np.float32)
    if diam.shape != (N,):
        raise ValueError("diam_max_1d shape mismatch")
    diam = np.nan_to_num(diam, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    Tp = int(idx.shape[0])

    y1_t = y1_base[idx]
    y1_tm1 = y1_base[idx - 1]
    dy1 = y1_t - y1_tm1

    y2_t = y2_agg[idx] * has[None, :]
    y2_tm1 = y2_agg[idx - 1] * has[None, :]
    dy2 = y2_t - y2_tm1

    depth1_full = (y1_base - inv[None, :]).astype(np.float32, copy=False)
    depth1 = depth1_full[idx]
    ddepth1 = dy1
    depth2 = y2_t - bed[None, :]
    ddepth2 = dy2

    gap = (y2_t - y1_t) * has[None, :]
    gpos = np.maximum(gap, 0.0)
    gneg = np.maximum(-gap, 0.0)

    y2m_t = y2m[idx]
    dy2m = y2m_t - y2m[idx - 1]
    y2x_t = y2x[idx]
    dy2x = y2x_t - y2x[idx - 1]

    lap_src = np.asarray(lap_src, dtype=np.int64)
    lap_dst = np.asarray(lap_dst, dtype=np.int64)
    lap_w = np.asarray(lap_w, dtype=np.float32)
    if lap_src.shape != lap_dst.shape or lap_src.shape != lap_w.shape:
        raise ValueError("lap_src/lap_dst/lap_w must have same shape")

    # Static weight sums (normalized)
    wsum_in = np.bincount(lap_dst, weights=lap_w.astype(np.float64), minlength=N).astype(np.float32)
    wsum_out = np.bincount(lap_src, weights=lap_w.astype(np.float64), minlength=N).astype(np.float32)
    wsum_in_n = (wsum_in / float(max(1e-6, float(wsum_in.max())))).astype(np.float32, copy=False)
    wsum_out_n = (wsum_out / float(max(1e-6, float(wsum_out.max())))).astype(np.float32, copy=False)

    # Means
    in_h_pos_full, in_h_neg_full = neighbor_weighted_posneg_mean_time_major(y1_base, src=lap_src, dst=lap_dst, w=lap_w)
    in_d_pos_full, in_d_neg_full = neighbor_weighted_posneg_mean_time_major(
        depth1_full, src=lap_src, dst=lap_dst, w=lap_w
    )
    out_h_pos_rev, out_h_neg_rev = neighbor_weighted_posneg_mean_time_major(y1_base, src=lap_dst, dst=lap_src, w=lap_w)
    out_d_pos_rev, out_d_neg_rev = neighbor_weighted_posneg_mean_time_major(
        depth1_full, src=lap_dst, dst=lap_src, w=lap_w
    )
    out_h_pos_full = out_h_neg_rev
    out_h_neg_full = out_h_pos_rev
    out_d_pos_full = out_d_neg_rev
    out_d_neg_full = out_d_pos_rev

    # Sums + maxes (junction strength)
    in_h_pos_sum_full, in_h_neg_sum_full, in_h_pos_max_full, in_h_neg_max_full = neighbor_weighted_posneg_summax_time_major(
        y1_base, src=lap_src, dst=lap_dst, w=lap_w
    )
    in_d_pos_sum_full, in_d_neg_sum_full, _, _ = neighbor_weighted_posneg_summax_time_major(
        depth1_full, src=lap_src, dst=lap_dst, w=lap_w
    )
    out_h_pos_sum_rev, out_h_neg_sum_rev, out_h_pos_max_rev, out_h_neg_max_rev = neighbor_weighted_posneg_summax_time_major(
        y1_base, src=lap_dst, dst=lap_src, w=lap_w
    )
    out_d_pos_sum_rev, out_d_neg_sum_rev, _, _ = neighbor_weighted_posneg_summax_time_major(
        depth1_full, src=lap_dst, dst=lap_src, w=lap_w
    )
    out_h_pos_sum_full = out_h_neg_sum_rev
    out_h_neg_sum_full = out_h_pos_sum_rev
    out_d_pos_sum_full = out_d_neg_sum_rev
    out_d_neg_sum_full = out_d_pos_sum_rev
    out_h_pos_max_full = out_h_neg_max_rev
    out_h_neg_max_full = out_h_pos_max_rev

    in_h_pos_m = in_h_pos_full[idx]
    in_h_neg_m = in_h_neg_full[idx]
    in_d_pos_m = in_d_pos_full[idx]
    in_d_neg_m = in_d_neg_full[idx]
    out_h_pos_m = out_h_pos_full[idx]
    out_h_neg_m = out_h_neg_full[idx]
    out_d_pos_m = out_d_pos_full[idx]
    out_d_neg_m = out_d_neg_full[idx]

    in_h_pos_s = in_h_pos_sum_full[idx]
    in_h_neg_s = in_h_neg_sum_full[idx]
    in_d_pos_s = in_d_pos_sum_full[idx]
    in_d_neg_s = in_d_neg_sum_full[idx]
    out_h_pos_s = out_h_pos_sum_full[idx]
    out_h_neg_s = out_h_neg_sum_full[idx]
    out_d_pos_s = out_d_pos_sum_full[idx]
    out_d_neg_s = out_d_neg_sum_full[idx]

    in_h_pos_max = in_h_pos_max_full[idx]
    in_h_neg_max = in_h_neg_max_full[idx]
    out_h_pos_max = out_h_pos_max_full[idx]
    out_h_neg_max = out_h_neg_max_full[idx]

    surcharge_full = np.maximum(depth1_full - diam[None, :], 0.0).astype(np.float32, copy=False)
    surcharge = surcharge_full[idx]
    dsurcharge = surcharge - surcharge_full[idx - 1]

    rain_lags = int(rain_lags)
    if rain_lags < 0:
        raise ValueError("rain_lags must be >= 0")
    if rain_lags and int(warmup) < int(rain_lags):
        raise ValueError("warmup must be >= rain_lags")
    if rain_lags:
        r_lags = []
        for j in range(1, int(rain_lags) + 1):
            r_lags.append(r[idx - j])
        r_lags = np.stack(r_lags, axis=1).astype(np.float32, copy=False)
    else:
        r_lags = None

    use_out_nbr = bool(use_out_nbr)
    use_nbr_in = (nbr_src is not None) or (nbr_dst is not None)
    if use_out_nbr and not use_nbr_in:
        raise ValueError("use_out_nbr=True requires nbr_src/nbr_dst")

    if use_nbr_in:
        if nbr_src is None or nbr_dst is None:
            raise ValueError("nbr_src/nbr_dst must both be provided or both be None")
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_src, dst=nbr_dst)[idx]
        nbr_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_src, dst=nbr_dst)[idx]
        nbr_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_src, dst=nbr_dst)[idx]
    else:
        nbr_depth1 = None
        nbr_ddepth1 = None
        nbr_gap = None

    if use_out_nbr:
        ddepth1_full2 = np.diff(depth1_full, axis=0, prepend=depth1_full[:1])
        gap_full = ((y2_agg * has[None, :]) - y1_base) * has[None, :]
        nbr_out_depth1 = neighbor_mean_time_major(depth1_full, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_ddepth1 = neighbor_mean_time_major(ddepth1_full2, src=nbr_dst, dst=nbr_src)[idx]
        nbr_out_gap = neighbor_mean_time_major(gap_full.astype(np.float32, copy=False), src=nbr_dst, dst=nbr_src)[idx]
    else:
        nbr_out_depth1 = None
        nbr_out_ddepth1 = None
        nbr_out_gap = None

    # Base dims: 38
    base_dim = 38
    if rain_lags:
        base_dim += int(rain_lags)
    if use_nbr_in:
        base_dim += 3
    if use_out_nbr:
        base_dim += 3

    rain_t = r[idx][:, None].repeat(N, axis=1)
    cum_t = cum[idx][:, None].repeat(N, axis=1)
    t_norm = (idx.astype(np.float32) / float(max(1, T - 1)))[:, None].repeat(N, axis=1)

    X = np.empty((Tp, N, int(base_dim)), dtype=np.float32)
    pos = 0
    for arr in (
        depth1,
        ddepth1,
        depth2,
        ddepth2,
        gap,
        gpos,
        gneg,
        y2m_t,
        dy2m,
        y2x_t,
        dy2x,
        in_h_pos_m,
        in_h_neg_m,
        in_d_pos_m,
        in_d_neg_m,
        out_h_pos_m,
        out_h_neg_m,
        out_d_pos_m,
        out_d_neg_m,
        in_h_pos_s,
        in_h_neg_s,
        in_d_pos_s,
        in_d_neg_s,
        out_h_pos_s,
        out_h_neg_s,
        out_d_pos_s,
        out_d_neg_s,
        in_h_pos_max,
        in_h_neg_max,
        out_h_pos_max,
        out_h_neg_max,
        surcharge,
        dsurcharge,
        wsum_in_n[None, :].repeat(Tp, axis=0),
        wsum_out_n[None, :].repeat(Tp, axis=0),
        rain_t,
        cum_t,
        t_norm,
    ):
        X[..., pos] = arr
        pos += 1

    if r_lags is not None:
        for j in range(int(rain_lags)):
            X[..., pos] = r_lags[:, j][:, None].repeat(N, axis=1)
            pos += 1

    if use_nbr_in:
        X[..., pos] = nbr_depth1
        pos += 1
        X[..., pos] = nbr_ddepth1
        pos += 1
        X[..., pos] = nbr_gap
        pos += 1

    if use_out_nbr:
        X[..., pos] = nbr_out_depth1
        pos += 1
        X[..., pos] = nbr_out_ddepth1
        pos += 1
        X[..., pos] = nbr_out_gap
        pos += 1

    if int(pos) != int(base_dim):
        raise RuntimeError("internal dyn feature dim mismatch (v9)")
    return torch.from_numpy(X)


def build_dyn_features_1d_v10(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 10).

    Extends v9 with warmup-window summary features, broadcast across the
    forecast horizon so later stages can condition on the observed pre-rollout
    hydraulic state instead of relying only on the current timestep features.
    """
    x9 = build_dyn_features_1d_v9(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )

    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    has = np.where(has > 0.0, 1.0, 0.0).astype(np.float32, copy=False)
    diam = np.asarray(diam_max_1d, dtype=np.float32)

    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    Tp = int(T - warmup)
    y2_masked = y2_agg * has[None, :]
    depth1_full = (y1_base - inv[None, :]).astype(np.float32, copy=False)
    depth2_full = (y2_masked - bed[None, :]).astype(np.float32, copy=False)
    gap_full = ((y2_masked - y1_base) * has[None, :]).astype(np.float32, copy=False)
    surcharge_full = np.maximum(depth1_full - diam[None, :], 0.0).astype(np.float32, copy=False)

    warm_last = int(warmup - 1)
    warm_depth1_last = depth1_full[warm_last]
    warm_depth1_mean = depth1_full[:warmup].mean(axis=0).astype(np.float32, copy=False)
    warm_depth1_slope = (depth1_full[warm_last] - depth1_full[0]).astype(np.float32, copy=False)
    warm_depth2_last = depth2_full[warm_last]
    warm_gap_last = gap_full[warm_last]
    warm_gap_absmax = np.max(np.abs(gap_full[:warmup]), axis=0).astype(np.float32, copy=False)
    warm_surcharge_last = surcharge_full[warm_last]
    warm_y2m_last = y2m[warm_last]
    warm_y2x_last = y2x[warm_last]

    rain = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    warm_rain_sum = np.float32(rain[:warmup].sum())
    warm_rain_last = np.float32(rain[warm_last])

    ctx = np.empty((Tp, N, 11), dtype=np.float32)
    ctx[..., 0] = warm_depth1_last[None, :]
    ctx[..., 1] = warm_depth1_mean[None, :]
    ctx[..., 2] = warm_depth1_slope[None, :]
    ctx[..., 3] = warm_depth2_last[None, :]
    ctx[..., 4] = warm_gap_last[None, :]
    ctx[..., 5] = warm_gap_absmax[None, :]
    ctx[..., 6] = warm_surcharge_last[None, :]
    ctx[..., 7] = warm_y2m_last[None, :]
    ctx[..., 8] = warm_y2x_last[None, :]
    ctx[..., 9] = warm_rain_sum
    ctx[..., 10] = warm_rain_last
    return torch.cat([x9, torch.from_numpy(ctx)], dim=-1)


def build_dyn_features_1d_v11(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_1d: np.ndarray,  # [T, N1] (only warmup is used)
    vol2_agg: np.ndarray,  # [T, N1] (only warmup is used)
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 11).

    Extends v10 with warmup-only hydraulic hidden-state summaries that are
    provided in the competition files but were previously unused:
      - 1D inlet flow
      - connected 2D water volume
    """
    x10 = build_dyn_features_1d_v10(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )

    inlet_1d = np.asarray(inlet_1d, dtype=np.float32)
    vol2_agg = np.asarray(vol2_agg, dtype=np.float32)
    if inlet_1d.ndim != 2 or vol2_agg.ndim != 2:
        raise ValueError("inlet_1d and vol2_agg must be [T, N1]")
    if inlet_1d.shape != vol2_agg.shape or inlet_1d.shape != y1_base.shape:
        raise ValueError("inlet_1d/vol2_agg must align with y1_base")
    T, N = inlet_1d.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    Tp = int(T - warmup)
    warm_last = int(warmup - 1)

    q_last = inlet_1d[warm_last].astype(np.float32, copy=False)
    q_mean = np.nanmean(inlet_1d[:warmup], axis=0).astype(np.float32, copy=False)
    q_slope = (inlet_1d[warm_last] - inlet_1d[0]).astype(np.float32, copy=False)
    q_absmax = np.nanmax(np.abs(inlet_1d[:warmup]), axis=0).astype(np.float32, copy=False)

    v_last = vol2_agg[warm_last].astype(np.float32, copy=False)
    v_mean = np.nanmean(vol2_agg[:warmup], axis=0).astype(np.float32, copy=False)
    v_slope = (vol2_agg[warm_last] - vol2_agg[0]).astype(np.float32, copy=False)

    ctx = np.empty((Tp, N, 7), dtype=np.float32)
    ctx[..., 0] = q_last[None, :]
    ctx[..., 1] = q_mean[None, :]
    ctx[..., 2] = q_slope[None, :]
    ctx[..., 3] = q_absmax[None, :]
    ctx[..., 4] = v_last[None, :]
    ctx[..., 5] = v_mean[None, :]
    ctx[..., 6] = v_slope[None, :]
    return torch.cat([x10, torch.from_numpy(ctx)], dim=-1)


def build_dyn_features_1d_v12(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 12).

    Extends `v9` with a predicted inlet-flow trajectory available at test time.
    This uses an auxiliary inlet-flow baseline model rather than future ground
    truth inlet flow.
    """
    x9 = build_dyn_features_1d_v9(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )

    q = np.asarray(inlet_pred_1d, dtype=np.float32)
    if q.ndim != 2 or q.shape != y1_base.shape:
        raise ValueError("inlet_pred_1d must be [T, N1] aligned to y1_base")
    T, N = q.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    q_t = q[idx]
    q_tm1 = q[idx - 1]
    dq = q_t - q_tm1
    q_pos = np.maximum(q_t, 0.0).astype(np.float32, copy=False)
    q_neg = np.maximum(-q_t, 0.0).astype(np.float32, copy=False)
    q_feat = np.stack([q_t, q_tm1, dq, q_pos, q_neg], axis=-1).astype(np.float32, copy=False)
    return torch.cat([x9, torch.from_numpy(q_feat)], dim=-1)


def build_dyn_features_1d_v13(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    base_area_1d: np.ndarray,  # [N1]
    conn_area_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
    dt_seconds: float = 300.0,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 13).

    Extends `v12` with simple storage / continuity proxies:
      - 1D storage proxy from base area
      - connected 2D storage proxy from cell area
      - inlet-flow volume-transfer proxy over one timestep
      - continuity-style imbalances
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )

    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    q = np.asarray(inlet_pred_1d, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != q.shape:
        raise ValueError("y1_base/y2_agg/inlet_pred_1d must align")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    base_area = np.asarray(base_area_1d, dtype=np.float32)
    conn_area = np.asarray(conn_area_1d, dtype=np.float32)
    if inv.shape != (N,) or bed.shape != (N,) or has.shape != (N,) or base_area.shape != (N,) or conn_area.shape != (N,):
        raise ValueError("static proxy shapes mismatch")
    has = np.where(has > 0.0, 1.0, 0.0).astype(np.float32, copy=False)
    base_area = np.maximum(base_area, 0.0).astype(np.float32, copy=False)
    conn_area = np.maximum(conn_area, 0.0).astype(np.float32, copy=False)

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    depth1 = np.maximum(y1_base - inv[None, :], 0.0).astype(np.float32, copy=False)
    depth2 = np.maximum((y2_agg * has[None, :]) - bed[None, :], 0.0).astype(np.float32, copy=False)

    vol1 = (depth1 * base_area[None, :]).astype(np.float32, copy=False)
    vol2 = (depth2 * conn_area[None, :]).astype(np.float32, copy=False)
    dvol1 = (vol1[idx] - vol1[idx - 1]).astype(np.float32, copy=False)
    dvol2 = (vol2[idx] - vol2[idx - 1]).astype(np.float32, copy=False)

    qdt = (q[idx] * np.float32(dt_seconds)).astype(np.float32, copy=False)
    imb1 = (dvol1 - qdt).astype(np.float32, copy=False)
    imb2 = (dvol2 + qdt).astype(np.float32, copy=False)
    vol_feat = np.stack([vol1[idx], dvol1, vol2[idx], dvol2, qdt, imb1, imb2], axis=-1).astype(np.float32, copy=False)
    return torch.cat([x12, torch.from_numpy(vol_feat)], dim=-1)


def build_dyn_features_1d_v14(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    edgeflow_node_feats: np.ndarray,  # [T, N1, 4]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 14).

    Extends `v12` with predicted incident 1D pipe-flow aggregates per node:
      - inflow
      - outflow
      - net inflow
      - total |flow|
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    qf = np.asarray(edgeflow_node_feats, dtype=np.float32)
    if qf.ndim != 3 or qf.shape[:2] != y1_base.shape or qf.shape[2] != 4:
        raise ValueError("edgeflow_node_feats must be [T, N1, 4] aligned to y1_base")
    T = int(qf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    return torch.cat([x12, torch.from_numpy(qf[idx].astype(np.float32, copy=False))], dim=-1)


def build_dyn_features_1d_v15(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    volagg_pred_1d: np.ndarray,  # [T, N1]
    conn_area_1d: np.ndarray,  # [N1]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 15).

    Extends `v12` with forecast-time predicted coupled 2D storage aggregates:
      - log1p(volume)
      - previous log1p(volume)
      - delta log-volume
      - volume / connected area
      - delta(volume / connected area)
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    v = np.asarray(volagg_pred_1d, dtype=np.float32)
    if v.ndim != 2 or v.shape != y1_base.shape:
        raise ValueError("volagg_pred_1d must be [T, N1] aligned to y1_base")
    area = np.asarray(conn_area_1d, dtype=np.float32)
    if area.shape != (y1_base.shape[1],):
        raise ValueError("conn_area_1d must be [N1]")
    T = int(v.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    area_safe = np.maximum(area, 1.0).astype(np.float32, copy=False)
    vlog = (np.sign(v) * np.log1p(np.abs(v))).astype(np.float32, copy=False)
    depth_eq = (v / area_safe[None, :]).astype(np.float32, copy=False)
    feat = np.stack(
        [
            vlog[idx],
            vlog[idx - 1],
            (vlog[idx] - vlog[idx - 1]).astype(np.float32, copy=False),
            depth_eq[idx],
            (depth_eq[idx] - depth_eq[idx - 1]).astype(np.float32, copy=False),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return torch.cat([x12, torch.from_numpy(feat)], dim=-1)


def build_dyn_features_1d_v16(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    y1_aux_base: np.ndarray,  # [T, N1]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 16).

    Extends `v12` with an alternative baseline-side 1D trajectory:
      - auxiliary baseline at time t
      - auxiliary baseline at time t-1
      - auxiliary baseline delta
      - disagreement vs current stacked base at time t
      - disagreement at time t-1
      - disagreement delta
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    y1a = np.asarray(y1_aux_base, dtype=np.float32)
    if y1a.ndim != 2 or y1a.shape != y1_base.shape:
        raise ValueError("y1_aux_base must be [T, N1] aligned to y1_base")
    T = int(y1a.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    y1a_t = y1a[idx]
    y1a_tm1 = y1a[idx - 1]
    dy1a = (y1a_t - y1a_tm1).astype(np.float32, copy=False)
    diff_t = (y1a_t - y1_base[idx]).astype(np.float32, copy=False)
    diff_tm1 = (y1a_tm1 - y1_base[idx - 1]).astype(np.float32, copy=False)
    ddiff = (diff_t - diff_tm1).astype(np.float32, copy=False)
    feat = np.stack([y1a_t, y1a_tm1, dy1a, diff_t, diff_tm1, ddiff], axis=-1).astype(np.float32, copy=False)
    return torch.cat([x12, torch.from_numpy(feat)], dim=-1)


def build_dyn_features_1d_v17(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    edgeflow_node_feats: np.ndarray,  # [T, N1, 6*K]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 17).

    Extends `v12` with top-K incident predicted 1D edge-flow slots. Slots are
    ordered separately for incoming and outgoing pipes by static conductance,
    and each slot contributes:
      - oriented flow at time t   (positive = into the node)
      - oriented flow at time t-1
      - delta oriented flow
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    qf = np.asarray(edgeflow_node_feats, dtype=np.float32)
    if qf.ndim != 3 or qf.shape[:2] != y1_base.shape or int(qf.shape[2]) < 6:
        raise ValueError("edgeflow_node_feats must be [T, N1, 6*K] aligned to y1_base")
    T = int(qf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    return torch.cat([x12, torch.from_numpy(qf[idx].astype(np.float32, copy=False))], dim=-1)


def build_dyn_features_1d_v18(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    surfaceflow_node_feats: np.ndarray,  # [T, N1, 4]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 18).

    Extends `v12` with forecast-time predicted local surface-transport signals
    derived from 2D edge flow near the coupled 2D cells:
      - incoming surface flow
      - outgoing surface flow
      - net surface flow
      - total |surface flow|
    For each signal we include time t, time t-1, and delta.
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    sf = np.asarray(surfaceflow_node_feats, dtype=np.float32)
    if sf.ndim != 3 or sf.shape[:2] != y1_base.shape or sf.shape[2] != 4:
        raise ValueError("surfaceflow_node_feats must be [T, N1, 4] aligned to y1_base")
    T = int(sf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    sf_t = sf[idx]
    sf_tm1 = sf[idx - 1]
    dsf = (sf_t - sf_tm1).astype(np.float32, copy=False)
    feat = np.concatenate([sf_t, sf_tm1, dsf], axis=-1).astype(np.float32, copy=False)
    return torch.cat([x12, torch.from_numpy(feat)], dim=-1)


def build_dyn_features_1d_v19(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    surfaceflow_slot_feats: np.ndarray,  # [T, N1, 3*K]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 19).

    Extends `v12` with local slot-based surface transport around the coupled
    2D inlet cell. Each slot is an incident 2D edge around the connected 2D
    cell, ordered by static conductance, and oriented so positive means flow
    into the inlet cell.
    """
    x12 = build_dyn_features_1d_v12(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    sf = np.asarray(surfaceflow_slot_feats, dtype=np.float32)
    if sf.ndim != 3 or sf.shape[:2] != y1_base.shape or int(sf.shape[2]) < 3:
        raise ValueError("surfaceflow_slot_feats must be [T, N1, 3*K] aligned to y1_base")
    T = int(sf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    return torch.cat([x12, torch.from_numpy(sf[idx].astype(np.float32, copy=False))], dim=-1)


def build_dyn_features_1d_v20(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    surfaceflow_node_feats: np.ndarray,  # [T, N1, 4]
    surfaceflow_slot_feats: np.ndarray,  # [T, N1, 3*K]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 20).

    Combines the coarse local surface-transport aggregates from `v18` with the
    slot-based local surface-edge transport from `v19`.
    """
    x18 = build_dyn_features_1d_v18(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        surfaceflow_node_feats=surfaceflow_node_feats,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    sf = np.asarray(surfaceflow_slot_feats, dtype=np.float32)
    if sf.ndim != 3 or sf.shape[:2] != y1_base.shape or int(sf.shape[2]) < 3:
        raise ValueError("surfaceflow_slot_feats must be [T, N1, 3*K] aligned to y1_base")
    T = int(sf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    return torch.cat([x18, torch.from_numpy(sf[idx].astype(np.float32, copy=False))], dim=-1)


def build_dyn_features_1d_v21(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1]
    y2_knn_max: np.ndarray,  # [T, N1]
    rain_2d: np.ndarray,  # [T, N2]
    inlet_pred_1d: np.ndarray,  # [T, N1]
    surfaceflow_slot_feats: np.ndarray,  # [T, N1, 3*K]
    local2d_node_feats: np.ndarray,  # [T, N1, 4*K2]
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    lap_src: np.ndarray,  # [E]
    lap_dst: np.ndarray,  # [E]
    lap_w: np.ndarray,  # [E]
    diam_max_1d: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 1D residual (version 21).

    Extends `v19` with topological neighboring 2D-cell state slots around the
    coupled inlet cell. This adds local storage / head-gradient context on top
    of the local surface-edge transport slots.
    """
    x19 = build_dyn_features_1d_v19(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        surfaceflow_slot_feats=surfaceflow_slot_feats,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    lf = np.asarray(local2d_node_feats, dtype=np.float32)
    if lf.ndim != 3 or lf.shape[:2] != y1_base.shape or int(lf.shape[2]) < 4:
        raise ValueError("local2d_node_feats must be [T, N1, 4*K] aligned to y1_base")
    T = int(lf.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    return torch.cat([x19, torch.from_numpy(lf[idx].astype(np.float32, copy=False))], dim=-1)


def build_dyn_features_1d_v22(
    *,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    y2_knn_mean: np.ndarray,
    y2_knn_max: np.ndarray,
    rain_2d: np.ndarray,
    inlet_pred_1d: np.ndarray,
    surfaceflow_slot_feats: np.ndarray,
    y1_aux_base: np.ndarray,
    invert_1d: np.ndarray,
    bed_agg_1d: np.ndarray,
    has_conn_1d: np.ndarray,
    lap_src: np.ndarray,
    lap_dst: np.ndarray,
    lap_w: np.ndarray,
    diam_max_1d: np.ndarray,
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    x19 = build_dyn_features_1d_v19(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        rain_2d=rain_2d,
        inlet_pred_1d=inlet_pred_1d,
        surfaceflow_slot_feats=surfaceflow_slot_feats,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
        use_out_nbr=use_out_nbr,
        rain_lags=rain_lags,
    )
    y1a = np.asarray(y1_aux_base, dtype=np.float32)
    if y1a.ndim != 2 or y1a.shape != y1_base.shape:
        raise ValueError("y1_aux_base must be [T, N1] aligned to y1_base")
    T = int(y1a.shape[0])
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")
    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    y1a_t = y1a[idx]
    y1a_tm1 = y1a[idx - 1]
    dy1a = (y1a_t - y1a_tm1).astype(np.float32, copy=False)
    diff_t = (y1a_t - y1_base[idx]).astype(np.float32, copy=False)
    diff_tm1 = (y1a_tm1 - y1_base[idx - 1]).astype(np.float32, copy=False)
    ddiff = (diff_t - diff_tm1).astype(np.float32, copy=False)
    feat = np.stack([y1a_t, y1a_tm1, dy1a, diff_t, diff_tm1, ddiff], axis=-1).astype(np.float32, copy=False)
    return torch.cat([x19, torch.from_numpy(feat)], dim=-1)


def build_warm_context_1d_v1(
    *,
    y1_base: np.ndarray,  # [T, N1]
    y2_agg: np.ndarray,  # [T, N1]
    y2_knn_mean: np.ndarray,  # [T, N1] (depth)
    y2_knn_max: np.ndarray,  # [T, N1] (depth)
    invert_1d: np.ndarray,  # [N1]
    bed_agg_1d: np.ndarray,  # [N1]
    has_conn_1d: np.ndarray,  # [N1]
    inlet_1d: np.ndarray | None,  # [T, N1]
    vol2_agg: np.ndarray | None,  # [T, N1]
    warmup: int,
) -> torch.Tensor:
    """
    Warmup-only context for Model2 1D residuals.

    Unlike `v10` / `v11`, this is not broadcast across forecast timesteps.
    It preserves the full observed warmup trajectory and is intended to seed
    the GRU hidden state through `warm_ctx -> h0`.

    Per warmup step, channels are:
      [depth1, depth2, gap, y2_knn_mean_depth, y2_knn_max_depth, inlet_flow, log1p(volume)]
    """
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != y2m.shape or y1_base.shape != y2x.shape:
        raise ValueError("y1_base/y2_agg/y2_knn_mean/y2_knn_max must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    if inv.shape != (N,) or bed.shape != (N,) or has.shape != (N,):
        raise ValueError("invert/bed/has shapes mismatch")
    has = np.where(has > 0.0, 1.0, 0.0).astype(np.float32, copy=False)

    y2_masked = (y2_agg[:warmup] * has[None, :]).astype(np.float32, copy=False)
    depth1 = (y1_base[:warmup] - inv[None, :]).astype(np.float32, copy=False)
    depth2 = (y2_masked - bed[None, :]).astype(np.float32, copy=False)
    gap = ((y2_masked - y1_base[:warmup]) * has[None, :]).astype(np.float32, copy=False)

    if inlet_1d is None:
        q = np.zeros((int(warmup), N), dtype=np.float32)
    else:
        q = np.asarray(inlet_1d, dtype=np.float32)
        if q.shape != y1_base.shape:
            raise ValueError("inlet_1d must align with y1_base")
        q = np.nan_to_num(q[:warmup], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if vol2_agg is None:
        v_log = np.zeros((int(warmup), N), dtype=np.float32)
    else:
        vol = np.asarray(vol2_agg, dtype=np.float32)
        if vol.shape != y1_base.shape:
            raise ValueError("vol2_agg must align with y1_base")
        vol = np.nan_to_num(vol[:warmup], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        v_log = (np.sign(vol) * np.log1p(np.abs(vol))).astype(np.float32, copy=False)

    ctx = np.stack(
        [
            depth1,
            depth2,
            gap,
            y2m[:warmup].astype(np.float32, copy=False),
            y2x[:warmup].astype(np.float32, copy=False),
            q,
            v_log,
        ],
        axis=-1,
    )  # [warmup, N, C]
    ctx = np.transpose(ctx, (1, 0, 2)).reshape(N, -1).astype(np.float32, copy=False)
    return torch.from_numpy(ctx)


def build_warm_context_1d_v2(
    *,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    y2_knn_mean: np.ndarray,
    y2_knn_max: np.ndarray,
    invert_1d: np.ndarray,
    bed_agg_1d: np.ndarray,
    has_conn_1d: np.ndarray,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    warmup: int,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
) -> torch.Tensor:
    ctx1 = build_warm_context_1d_v1(
        y1_base=y1_base,
        y2_agg=y2_agg,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        invert_1d=invert_1d,
        bed_agg_1d=bed_agg_1d,
        has_conn_1d=has_conn_1d,
        inlet_1d=inlet_1d,
        vol2_agg=vol2_agg,
        warmup=warmup,
    ).numpy()
    extra_parts: list[np.ndarray] = []
    for feat in (edgeflow_node_feats, surfaceflow_slot_feats, local2d_node_feats):
        if feat is None:
            continue
        arr = np.asarray(feat, dtype=np.float32)
        if arr.ndim != 3 or int(arr.shape[0]) < int(warmup):
            raise ValueError("warm context local feature arrays must be [T, N1, C] with T >= warmup")
        warm = arr[:warmup].astype(np.float32, copy=False)
        extra_parts.append(np.mean(warm, axis=0).astype(np.float32, copy=False))
        extra_parts.append(warm[-1].astype(np.float32, copy=False))
    if not extra_parts:
        return torch.from_numpy(ctx1.astype(np.float32, copy=False))
    ctx = np.concatenate([ctx1, *extra_parts], axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(ctx)


def build_warm_sequence_1d_v1(
    *,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    y2_knn_mean: np.ndarray,
    y2_knn_max: np.ndarray,
    invert_1d: np.ndarray,
    bed_agg_1d: np.ndarray,
    has_conn_1d: np.ndarray,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    warmup: int,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
) -> torch.Tensor:
    y1_base = np.asarray(y1_base, dtype=np.float32)
    y2_agg = np.asarray(y2_agg, dtype=np.float32)
    y2m = np.asarray(y2_knn_mean, dtype=np.float32)
    y2x = np.asarray(y2_knn_max, dtype=np.float32)
    if y1_base.shape != y2_agg.shape or y1_base.shape != y2m.shape or y1_base.shape != y2x.shape:
        raise ValueError("y1_base/y2_agg/y2_knn_mean/y2_knn_max must have same shape")
    T, N = y1_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    inv = np.asarray(invert_1d, dtype=np.float32)
    bed = np.asarray(bed_agg_1d, dtype=np.float32)
    has = np.asarray(has_conn_1d, dtype=np.float32)
    if inv.shape != (N,) or bed.shape != (N,) or has.shape != (N,):
        raise ValueError("invert/bed/has shapes mismatch")
    has = np.where(has > 0.0, 1.0, 0.0).astype(np.float32, copy=False)

    y2_masked = (y2_agg[:warmup] * has[None, :]).astype(np.float32, copy=False)
    depth1 = (y1_base[:warmup] - inv[None, :]).astype(np.float32, copy=False)
    depth2 = (y2_masked - bed[None, :]).astype(np.float32, copy=False)
    gap = ((y2_masked - y1_base[:warmup]) * has[None, :]).astype(np.float32, copy=False)

    if inlet_1d is None:
        q = np.zeros((int(warmup), N), dtype=np.float32)
    else:
        q = np.asarray(inlet_1d, dtype=np.float32)
        if q.shape != y1_base.shape:
            raise ValueError("inlet_1d must align with y1_base")
        q = np.nan_to_num(q[:warmup], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if vol2_agg is None:
        v_log = np.zeros((int(warmup), N), dtype=np.float32)
    else:
        vol = np.asarray(vol2_agg, dtype=np.float32)
        if vol.shape != y1_base.shape:
            raise ValueError("vol2_agg must align with y1_base")
        vol = np.nan_to_num(vol[:warmup], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        v_log = (np.sign(vol) * np.log1p(np.abs(vol))).astype(np.float32, copy=False)

    parts = [
        depth1[..., None],
        depth2[..., None],
        gap[..., None],
        y2m[:warmup].astype(np.float32, copy=False)[..., None],
        y2x[:warmup].astype(np.float32, copy=False)[..., None],
        q[..., None],
        v_log[..., None],
    ]
    for feat in (edgeflow_node_feats, surfaceflow_slot_feats, local2d_node_feats):
        if feat is None:
            continue
        arr = np.asarray(feat, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[:2] != y1_base.shape:
            raise ValueError("local warm sequence feature arrays must be [T, N1, C] aligned to y1_base")
        parts.append(arr[:warmup].astype(np.float32, copy=False))
    ctx = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    return torch.from_numpy(ctx)


def build_dyn_features_2d(
    *,
    y2_base: np.ndarray,  # [T, N2]
    rain_2d: np.ndarray,  # [T, N2]
    bed_2d: np.ndarray,  # [N2]
    warmup: int,
) -> torch.Tensor:
    """
    Dynamic features for Model2 2D residual (minimal).

    Layout:
      [depth2, ddepth2, rain, cum, t_norm]
    """
    y2_base = np.asarray(y2_base, dtype=np.float32)
    if y2_base.ndim != 2:
        raise ValueError("y2_base must be [T, N2]")
    T, N2 = y2_base.shape
    if T <= warmup:
        raise ValueError("sequence shorter than warmup")

    r = rain_scalar_from_2d(rain_2d).astype(np.float32, copy=False)
    cum = np.cumsum(r).astype(np.float32, copy=False)

    bed = np.asarray(bed_2d, dtype=np.float32)
    if bed.shape != (N2,):
        raise ValueError("bed_2d must be [N2]")

    idx = np.arange(int(warmup), int(T), dtype=np.int64)
    Tp = int(idx.shape[0])

    y_t = y2_base[idx]
    y_tm1 = y2_base[idx - 1]
    dy = y_t - y_tm1

    depth2 = y_t - bed[None, :]
    ddepth2 = dy

    rain_t = r[idx][:, None].repeat(N2, axis=1)
    cum_t = cum[idx][:, None].repeat(N2, axis=1)
    t_norm = (idx.astype(np.float32) / float(max(1, T - 1)))[:, None].repeat(N2, axis=1)

    X = np.empty((Tp, N2, 5), dtype=np.float32)
    X[..., 0] = depth2
    X[..., 1] = ddepth2
    X[..., 2] = rain_t
    X[..., 3] = cum_t
    X[..., 4] = t_norm
    return torch.from_numpy(X)
