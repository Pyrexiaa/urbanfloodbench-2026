from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from urbanflood.baseline import aggregate_2d_to_1d_mean, rollout_ar1x
from urbanflood.blend_gate_train import (
    BlendGateGRU,
    BlendGateGroupedMLP,
    BlendGateLinear,
    BlendGateMLP,
    BlendGateWarmMLP,
    _build_graph_context,
    _build_stack_bundle,
    _load_torch,
    _predict_stack_event,
    build_pipe_neighbor_node_features,
)
from urbanflood.data import load_event, load_graph, list_events
from urbanflood.metric import STD_DEV_DICT
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import build_static_features_m2_1d, build_static_features_m2_1d_v2
from urbanflood.residual_predict import _apply_resid1d_v1, _apply_resid2d, _schema, _write_event_rows
from urbanflood.residual_train import _pick_node_static
from urbanflood.surfaceflow_aux import build_coupled_neighbor_node_features


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-ckpt", type=str, required=True)
    p.add_argument("--m1-baseline-ckpt", type=str, required=True)
    p.add_argument("--m1-resid2d-ckpt", type=str, default="")
    p.add_argument("--m1-resid1d-ckpt", type=str, nargs="*", default=[])
    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--amp-bf16", action="store_true")
    return p.parse_args()


def _build_gate_model(*, ck: dict, dyn_dim: int, static_dim: int, warm_seq_dim: int, warmup: int, device: torch.device) -> torch.nn.Module:
    arch = str(ck.get("arch", "mlp"))
    if arch == "warm_mlp":
        model = BlendGateWarmMLP(
            dyn_dim=dyn_dim,
            static_dim=static_dim,
            warm_seq_dim=warm_seq_dim,
            warmup=warmup,
            hidden_dim=int(ck.get("hidden_dim", 16)),
            dropout=float(ck.get("dropout", 0.0)),
            correction_ft=float(ck.get("correction_ft", 0.0)),
        )
    elif arch == "gru":
        model = BlendGateGRU(
            dyn_dim=dyn_dim,
            static_dim=static_dim,
            hidden_dim=int(ck.get("hidden_dim", 16)),
            dropout=float(ck.get("dropout", 0.0)),
            correction_ft=float(ck.get("correction_ft", 0.0)),
        )
    elif arch == "group_mlp":
        model = BlendGateGroupedMLP(
            dyn_dim=dyn_dim,
            static_dim=static_dim,
            hidden_dim=int(ck.get("hidden_dim", 16)),
            dropout=float(ck.get("dropout", 0.0)),
            correction_ft=float(ck.get("correction_ft", 0.0)),
            n_groups=3,
        )
    elif arch == "linear":
        model = BlendGateLinear(
            dyn_dim=dyn_dim,
            static_dim=static_dim,
            correction_ft=float(ck.get("correction_ft", 0.0)),
        )
    else:
        model = BlendGateMLP(
            dyn_dim=dyn_dim,
            static_dim=static_dim,
            hidden_dim=int(ck.get("hidden_dim", 16)),
            dropout=float(ck.get("dropout", 0.0)),
            correction_ft=float(ck.get("correction_ft", 0.0)),
        )
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


def _ctx_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _build_gate_inputs(
    *,
    event_id: int,
    graph2,
    graph_ctx: dict,
    best_stack,
    alt_stack,
    model_root: Path,
    cache_dir: Path | None,
    warmup: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    warm_ctx_version: int,
    warm_seq_version: int,
    future_local2d_version: int,
    future_local1d_version: int,
    scope_mode: str,
    scope_q_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y1_obs, y2_obs, y1_best, y2_best, q_pred, rain_agg = _predict_stack_event(
        stack=best_stack,
        event_id=event_id,
        split="test",
        graph2=graph2,
        graph_ctx=graph_ctx,
        model_root=model_root,
        cache_dir=cache_dir,
        warmup=warmup,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    _y1_obs_alt, _y2_obs_alt, y1_alt, y2_alt, q_pred_alt, _rain_agg_alt = _predict_stack_event(
        stack=alt_stack,
        event_id=event_id,
        split="test",
        graph2=graph2,
        graph_ctx=graph_ctx,
        model_root=model_root,
        cache_dir=cache_dir,
        warmup=warmup,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    best_tail = y1_best[warmup:].astype(np.float32, copy=False)
    alt_tail = y1_alt[warmup:].astype(np.float32, copy=False)
    delta_best = np.zeros_like(best_tail)
    delta_best[1:] = best_tail[1:] - best_tail[:-1]
    delta_alt = np.zeros_like(alt_tail)
    delta_alt[1:] = alt_tail[1:] - alt_tail[:-1]
    gap = alt_tail - best_tail
    delta_gap = delta_alt - delta_best
    fill_best = np.clip((best_tail - graph_ctx["inv1"][None, :]) / graph_ctx["depth"][None, :], 0.0, 2.0)
    fill_alt = np.clip((alt_tail - graph_ctx["inv1"][None, :]) / graph_ctx["depth"][None, :], 0.0, 2.0)
    q_pred_tail = np.zeros_like(best_tail) if q_pred is None else q_pred[warmup:].astype(np.float32, copy=False)
    q_pred_alt_tail = np.zeros_like(best_tail) if q_pred_alt is None else q_pred_alt[warmup:].astype(np.float32, copy=False)
    q_gap = q_pred_alt_tail - q_pred_tail

    special_static = ((graph_ctx["static_gate"][:, 0] > 0.5) | (graph_ctx["static_gate"][:, 1] > 0.5)).astype(np.float32, copy=False)
    if str(scope_mode) == "none":
        scope = np.ones_like(best_tail, dtype=np.float32)
    elif str(scope_mode) == "source_ba0":
        scope = np.broadcast_to(special_static[None, :], best_tail.shape).astype(np.float32, copy=False)
    elif str(scope_mode) == "source_ba0_qpos":
        q_mask = (q_pred_tail > float(scope_q_thresh)).astype(np.float32, copy=False)
        scope = (np.broadcast_to(special_static[None, :], best_tail.shape) * q_mask).astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported scope_mode: {scope_mode}")

    y2_best_depth = (y2_best - graph_ctx["bed2"][None, :]).astype(np.float32, copy=False)
    y2_alt_depth = (y2_alt - graph_ctx["bed2"][None, :]).astype(np.float32, copy=False)
    y2_best_depth_agg = aggregate_2d_to_1d_mean(
        y2_best_depth,
        conn_src_1d=graph_ctx["conn_src"],
        conn_dst_2d=graph_ctx["conn_dst"],
        n_1d=int(graph_ctx["n1"]),
    )[warmup:].astype(np.float32, copy=False)
    y2_alt_depth_agg = aggregate_2d_to_1d_mean(
        y2_alt_depth,
        conn_src_1d=graph_ctx["conn_src"],
        conn_dst_2d=graph_ctx["conn_dst"],
        n_1d=int(graph_ctx["n1"]),
    )[warmup:].astype(np.float32, copy=False)
    y2_depth_gap = y2_alt_depth_agg - y2_best_depth_agg
    rain_tail = rain_agg[warmup:].astype(np.float32, copy=False)
    rain_prev = np.zeros_like(rain_tail)
    rain_prev[1:] = rain_tail[:-1]
    rain_delta = rain_tail - rain_prev
    rain_cum = np.cumsum(rain_tail, axis=0).astype(np.float32, copy=False)
    rain_cum_scale = np.maximum(np.max(rain_cum, axis=0, keepdims=True), 1e-6)
    rain_cum_norm = (rain_cum / rain_cum_scale).astype(np.float32, copy=False)
    time_frac = np.linspace(0.0, 1.0, num=int(best_tail.shape[0]), dtype=np.float32)[:, None]
    time_frac = np.repeat(time_frac, int(best_tail.shape[1]), axis=1)

    dyn_parts = [
        np.stack(
            [
                best_tail,
                delta_best,
                alt_tail,
                delta_alt,
                gap,
                delta_gap,
                q_pred_tail,
                fill_best,
                q_pred_alt_tail,
                q_gap,
                fill_alt,
                y2_best_depth_agg,
                y2_alt_depth_agg,
                y2_depth_gap,
                rain_tail,
                rain_prev,
                rain_delta,
                rain_cum_norm,
                time_frac,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
    ]
    if int(future_local2d_version) >= 1:
        local_best = build_coupled_neighbor_node_features(
            y2_2d=y2_best,
            bed_2d=graph_ctx["bed2"],
            center_cell=graph_ctx["local2d_center_cell"],
            neighbor_slots=graph_ctx["local2d_neighbor_slots"],
        )[warmup:].astype(np.float32, copy=False)
        local_alt = build_coupled_neighbor_node_features(
            y2_2d=y2_alt,
            bed_2d=graph_ctx["bed2"],
            center_cell=graph_ctx["local2d_center_cell"],
            neighbor_slots=graph_ctx["local2d_neighbor_slots"],
        )[warmup:].astype(np.float32, copy=False)
        dyn_parts.extend([local_best, local_alt, (local_alt - local_best).astype(np.float32, copy=False)])
    if int(future_local1d_version) >= 1:
        pipe_best = build_pipe_neighbor_node_features(
            y1_1d=y1_best,
            edge_from=graph_ctx["masks"]["src_dir"],
            edge_to=graph_ctx["masks"]["dst_dir"],
            edge_weight=graph_ctx["lap_w_dir"],
            n_1d=int(graph_ctx["n1"]),
        )[warmup:].astype(np.float32, copy=False)
        pipe_alt = build_pipe_neighbor_node_features(
            y1_1d=y1_alt,
            edge_from=graph_ctx["masks"]["src_dir"],
            edge_to=graph_ctx["masks"]["dst_dir"],
            edge_weight=graph_ctx["lap_w_dir"],
            n_1d=int(graph_ctx["n1"]),
        )[warmup:].astype(np.float32, copy=False)
        dyn_parts.extend([pipe_best, pipe_alt, (pipe_alt - pipe_best).astype(np.float32, copy=False)])
    dyn = np.concatenate(dyn_parts, axis=-1).astype(np.float32, copy=False)

    if int(warm_ctx_version) > 0 or int(warm_seq_version) > 0:
        raise ValueError("blend_gate_predict currently supports gates with warm_ctx_version=0 and warm_seq_version=0 only")

    return (
        torch.from_numpy(dyn),
        None,
        best_tail,
        alt_tail,
        y2_best[warmup:].astype(np.float32, copy=False),
        scope,
    )


def main() -> None:
    args = _parse_args()
    model_root = Path(args.model_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp_bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16

    gate_ck = _load_torch(Path(args.gate_ckpt))
    if str(gate_ck.get("kind", "")) != "blend_gate_m2_1d":
        raise ValueError("gate_ckpt must be kind blend_gate_m2_1d")
    warmup = int(gate_ck.get("warmup", 10))

    graph2 = load_graph(model_root, model_id=2, split_for_static="train")
    graph_ctx = _build_graph_context(model_root=model_root, graph2=graph2, device=device)
    best_meta = gate_ck["best_stack"]
    alt_meta = gate_ck["alt_stack"]
    best_stack = _build_stack_bundle(
        baseline_paths=list(best_meta["baseline_ckpts"]),
        resid2d_path=str(best_meta["resid2d_ckpt"]),
        resid1d_paths=list(best_meta["resid1d_ckpt"]),
        graph2=graph2,
        graph_ctx=graph_ctx,
        device=device,
        model_root=model_root,
    )
    alt_stack = _build_stack_bundle(
        baseline_paths=list(alt_meta["baseline_ckpts"]),
        resid2d_path=str(alt_meta["resid2d_ckpt"]),
        resid1d_paths=list(alt_meta["resid1d_ckpt"]),
        graph2=graph2,
        graph_ctx=graph_ctx,
        device=device,
        model_root=model_root,
    )

    sample_event = int(sorted(list_events(model_root, model_id=2, split="test"))[0])
    x_dyn_ex, warm_seq_ex, _best_ex, _alt_ex, _y2_ex, _scope_ex = _build_gate_inputs(
        event_id=sample_event,
        graph2=graph2,
        graph_ctx=graph_ctx,
        best_stack=best_stack,
        alt_stack=alt_stack,
        model_root=model_root,
        cache_dir=cache_dir,
        warmup=warmup,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        warm_ctx_version=int(gate_ck.get("warm_ctx_version", 0) or 0),
        warm_seq_version=int(gate_ck.get("warm_seq_version", 0) or 0),
        future_local2d_version=int(gate_ck.get("future_local2d_version", 0) or 0),
        future_local1d_version=int(gate_ck.get("future_local1d_version", 0) or 0),
        scope_mode=str(gate_ck.get("scope_mode", "none") or "none"),
        scope_q_thresh=float(gate_ck.get("scope_q_thresh", 0.0) or 0.0),
    )
    static_gate = torch.from_numpy(graph_ctx["static_gate"]).to(device)
    warm_seq_dim = 0 if warm_seq_ex is None else int(warm_seq_ex.shape[-1])
    gate_model = _build_gate_model(
        ck=gate_ck,
        dyn_dim=int(x_dyn_ex.shape[-1]),
        static_dim=int(static_gate.shape[1]),
        warm_seq_dim=warm_seq_dim,
        warmup=warmup,
        device=device,
    )
    group_idx = torch.zeros((int(static_gate.shape[0]),), dtype=torch.long, device=device)
    source_mask = static_gate[:, 0] > 0.5
    ba0_mask = static_gate[:, 1] > 0.5
    group_idx[source_mask] = 1
    group_idx[ba0_mask] = 2

    # Model 1 setup
    graph1 = load_graph(model_root, model_id=1, split_for_static="train")
    baseline_m1 = _load_torch(Path(args.m1_baseline_ckpt))
    w1 = baseline_m1["model_1"]["w"].numpy()
    warmup_m1 = int(baseline_m1["cfg"]["warmup"])
    if warmup_m1 != warmup:
        raise ValueError("Model1 and gate warmup must match")
    inv1_m1 = graph1.head_offset[: graph1.n_1d].cpu().numpy().astype(np.float32, copy=False)
    bed2_m1 = graph1.head_offset[graph1.n_1d :].cpu().numpy().astype(np.float32, copy=False)
    conn_src_m1 = graph1.conn_src_1d.cpu().numpy()
    conn_dst_m1 = graph1.conn_dst_2d.cpu().numpy()
    has_conn_m1 = (np.bincount(conn_src_m1, minlength=int(graph1.n_1d)) > 0).astype(np.float32)
    bed_agg_m1 = aggregate_2d_to_1d_mean(
        bed2_m1[None, :],
        conn_src_1d=conn_src_m1,
        conn_dst_2d=conn_dst_m1,
        n_1d=int(graph1.n_1d),
    )[0].astype(np.float32, copy=False)
    node_static_m1_2d = graph1.node_static_2d.to(device)
    m1_resid2d = None
    m1_edge_index_2d = None
    m1_edge_deg_inv_2d = None
    if str(args.m1_resid2d_ckpt):
        ck = _load_torch(Path(args.m1_resid2d_ckpt))
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        m1_resid2d = ResidualNodeGRU(n_nodes=int(graph1.n_2d), cfg=mcfg).to(device)
        m1_resid2d.load_state_dict(ck["state_dict"])
        m1_resid2d.eval()
        need_edges = (len(m1_resid2d.graph_mix) > 0) or (len(getattr(m1_resid2d, "graph_mix_post", ())) > 0) or (int(getattr(mcfg, "dyn_dim", 5)) in (7, 9))
        if need_edges:
            m1_edge_index_2d = graph1.edge_index_2d.to(device)
            deg = torch.bincount(graph1.edge_index_2d[1].cpu(), minlength=int(graph1.n_2d)).clamp(min=1).to(torch.float32)
            m1_edge_deg_inv_2d = (1.0 / deg).to(device)

    m1_stages: list[tuple[ResidualNodeGRU, torch.Tensor, bool]] = []
    m1_nbr_src = graph1.edge_index_1d[0].cpu().numpy().astype(np.int64, copy=False)
    m1_nbr_dst = graph1.edge_index_1d[1].cpu().numpy().astype(np.int64, copy=False)
    node_static_m1_1d_raw = graph1.node_static_1d.to(device)
    node_static_m1_1d_aug1 = build_static_features_m2_1d(graph1).to(device)
    node_static_m1_1d_aug2 = build_static_features_m2_1d_v2(graph1).to(device)
    for path in list(args.m1_resid1d_ckpt):
        ck = _load_torch(Path(path))
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        model = ResidualNodeGRU(n_nodes=int(graph1.n_1d), cfg=mcfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        use_nbr = int(mcfg.dyn_dim) == 13 or bool((ck.get("cfg", {}) or {}).get("use_nbr_feats", False))
        node_static = _pick_node_static(
            expected_dim=int(mcfg.static_dim),
            raw=node_static_m1_1d_raw,
            aug1=node_static_m1_1d_aug1,
            aug2=node_static_m1_1d_aug2,
        )
        m1_stages.append((model, node_static, use_nbr))

    events1 = sorted(list_events(model_root, model_id=1, split="test"))
    events2 = sorted(list_events(model_root, model_id=2, split="test"))
    if int(args.max_events) > 0:
        events1 = events1[: int(args.max_events)]
        events2 = events2[: int(args.max_events)]

    import pyarrow.parquet as pq

    row_base = 0
    with pq.ParquetWriter(out_path, _schema(), compression="snappy") as writer:
        for eid in tqdm(events1, desc="predict model 1 (gate submission)", leave=False):
            ev = load_event(model_root, graph=graph1, split="test", event_id=eid, cache_dir=cache_dir)
            y1 = ev.y_1d.numpy().astype(np.float32, copy=False)
            y2 = ev.y_2d.numpy().astype(np.float32, copy=False)
            rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
            y_init = np.concatenate([y1, y2], axis=1)
            y_pred = rollout_ar1x(w=w1, y_init=y_init, rain=rain, warmup=warmup)
            y1_base = y_pred[:, : graph1.n_1d]
            y2_base = y_pred[:, graph1.n_1d :]
            if m1_resid2d is not None:
                y2_pred = _apply_resid2d(
                    resid2d=m1_resid2d,
                    node_static_2d=node_static_m1_2d,
                    y2_base=y2_base,
                    rain_2d=rain,
                    bed2=bed2_m1,
                    warmup=warmup,
                    edge_index_2d=m1_edge_index_2d,
                    edge_deg_inv_2d=m1_edge_deg_inv_2d,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            else:
                y2_pred = y2_base
            y1_pred = y1_base.astype(np.float32, copy=True)
            if m1_stages:
                y2_agg = aggregate_2d_to_1d_mean(
                    y2_pred,
                    conn_src_1d=conn_src_m1,
                    conn_dst_2d=conn_dst_m1,
                    n_1d=int(graph1.n_1d),
                ).astype(np.float32, copy=False)
                for model, node_static, use_nbr in m1_stages:
                    y1_pred = _apply_resid1d_v1(
                        model=model,
                        node_static_1d=node_static,
                        y1_base=y1_pred,
                        y2_agg=y2_agg,
                        rain_2d=rain,
                        inv1=inv1_m1,
                        bed_agg=bed_agg_m1,
                        has_conn=has_conn_m1,
                        nbr_src=m1_nbr_src if use_nbr else None,
                        nbr_dst=m1_nbr_dst if use_nbr else None,
                        warmup=warmup,
                        device=device,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                    )
            row_base = _write_event_rows(writer=writer, row_base=row_base, model_id=1, event_id=eid, y1=y1_pred[warmup:], y2=y2_pred[warmup:])

        for eid in tqdm(events2, desc="predict model 2 (gate submission)", leave=False):
            x_dyn, warm_seq, best_tail, alt_tail, y2_tail, scope = _build_gate_inputs(
                event_id=eid,
                graph2=graph2,
                graph_ctx=graph_ctx,
                best_stack=best_stack,
                alt_stack=alt_stack,
                model_root=model_root,
                cache_dir=cache_dir,
                warmup=warmup,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                warm_ctx_version=int(gate_ck.get("warm_ctx_version", 0) or 0),
                warm_seq_version=int(gate_ck.get("warm_seq_version", 0) or 0),
                future_local2d_version=int(gate_ck.get("future_local2d_version", 0) or 0),
                future_local1d_version=int(gate_ck.get("future_local1d_version", 0) or 0),
                scope_mode=str(gate_ck.get("scope_mode", "none") or "none"),
                scope_q_thresh=float(gate_ck.get("scope_q_thresh", 0.0) or 0.0),
            )
            x_gpu = x_dyn.to(device)
            with torch.no_grad():
                if str(gate_ck.get("arch", "mlp")) == "warm_mlp":
                    gate, corr = gate_model(x_gpu, static_gate, warm_seq.to(device) if warm_seq is not None else None)
                elif str(gate_ck.get("arch", "mlp")) == "group_mlp":
                    gate, corr = gate_model(x_gpu, static_gate, group_idx)
                else:
                    gate, corr = gate_model(x_gpu, static_gate)
            best_gpu = torch.from_numpy(best_tail).to(device)
            alt_gpu = torch.from_numpy(alt_tail).to(device)
            scope_gpu = torch.from_numpy(scope).to(device)
            y1_gate = best_gpu + scope_gpu.float() * (gate.float() * (alt_gpu - best_gpu) + corr.float())
            row_base = _write_event_rows(
                writer=writer,
                row_base=row_base,
                model_id=2,
                event_id=eid,
                y1=y1_gate.detach().float().cpu().numpy(),
                y2=y2_tail,
            )

    print(f"wrote gated submission parquet: {out_path}")


if __name__ == "__main__":
    main()
