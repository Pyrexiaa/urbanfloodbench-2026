# urbanflood/eval_stack.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from urbanflood.baseline import aggregate_2d_to_1d_mean, aggregate_2d_to_1d_sum, predict_model2_from_baseline_ckpts, rollout_ar1x
from urbanflood.data import load_event, load_graph
from urbanflood.edgeflow_aux import build_edge_slot_index, load_edge_flow_1d
from urbanflood.metric import EventScore, score_event_from_full_grids, score_overall
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import build_dyn_features_1d_v1
from urbanflood.residual_train import (
    _apply_resid1d_ckpt,
    _apply_resid2d,
    _build_aux_baseline_views,
    _build_aux_traj_views,
    _build_coupling_views,
    _build_edgeflow_pred_views,
    _build_edgeflow_pred_slot_views,
    _build_inlet_pred_views,
    _build_surfaceflow_pred_views,
    _build_surfaceflow_pred_slot_views,
    _build_volagg_pred_views,
    _knn_2d_idx,
    _load_torch,
    _expert_group_idx_topo3,
    _mask_from_cfg,
    _mask_inputs,
    _ordered_pre_models,
    _pick_node_static,
    _pipe_weights,
)
from urbanflood.residual_features import build_static_features_m2_1d, build_static_features_m2_1d_v2
from urbanflood.surfaceflow_aux import (
    build_coupled_neighbor_node_features,
    build_coupled_neighbor_node_index,
    build_coupled_surface_slot_index,
    load_edge_flow_2d,
    load_surface_edge_weight,
)
from urbanflood.utils import seed_everything


def _apply_resid1d_v1(
    *,
    model: ResidualNodeGRU,
    node_static_1d: torch.Tensor,
    y1_base: np.ndarray,  # [T,N1]
    y2_agg: np.ndarray,  # [T,N1]
    rain_2d: np.ndarray,  # [T,N2]
    inv1: np.ndarray,  # [N1]
    bed_agg: np.ndarray,  # [N1]
    has_conn: np.ndarray,  # [N1]
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    warmup: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    y1 = np.asarray(y1_base, dtype=np.float32).copy()
    x = build_dyn_features_1d_v1(
        y1_base=y1_base,
        y2_agg=y2_agg,
        rain_2d=rain_2d,
        invert_1d=inv1,
        bed_agg_1d=bed_agg,
        has_conn_1d=has_conn,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        warmup=warmup,
    )
    x = x.to(device, dtype=amp_dtype) if amp_enabled else x.to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            r = model(x, node_static_1d).detach().float().cpu().numpy()
    y1[warmup:] = y1[warmup:] + r
    return y1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument(
        "--split-from",
        type=str,
        default="",
        help="Optional baseline checkpoint to source the train/val split from (useful for evaluating full-train artifacts).",
    )
    p.add_argument("--mixed-mode", type=str, default="weighted_split_ns", choices=["single", "weighted_split_ns"])
    p.add_argument("--alpha-1d", type=float, default=0.9)
    p.add_argument("--alpha-2d", type=float, default=0.5)

    p.add_argument("--m1-resid2d-ckpt", type=str, default="", help="Optional Model1 2D residual ckpt.")
    p.add_argument(
        "--m1-resid1d-ckpt",
        type=str,
        nargs="*",
        default=[],
        help="Optional Model1 1D residual ckpts (applied in order).",
    )

    p.add_argument("--resid2d-ckpt", type=str, default="", help="Optional Model2 2D residual ckpt.")
    p.add_argument(
        "--resid1d-ckpt",
        type=str,
        nargs="*",
        default=[],
        help="Optional Model2 1D residual ckpts (applied in order).",
    )

    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp-bf16", dest="amp_bf16", action="store_true")
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false")
    p.set_defaults(amp_bf16=True)
    p.add_argument("--max-events", type=int, default=0, help="Optional limit per model (smoke).")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    seed_everything(int(a.seed))
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16
    amp_enabled = bool(a.amp_bf16)

    model_root = Path(a.model_root)
    cache_dir = Path(a.cache_dir) if a.cache_dir else None

    baseline_ckpts = [_load_torch(Path(p)) for p in a.baseline_ckpt]
    warmup = int(baseline_ckpts[0]["cfg"]["warmup"])
    for c in baseline_ckpts[1:]:
        if int(c["cfg"]["warmup"]) != warmup:
            raise ValueError("all baseline checkpoints must share warmup")

    if str(a.split_from):
        split_ckpt = _load_torch(Path(a.split_from))
        if int(split_ckpt["cfg"]["warmup"]) != warmup:
            raise ValueError("--split-from warmup must match --baseline-ckpt warmup")
        split = split_ckpt.get("split", None)
        if split is None:
            raise ValueError("--split-from baseline ckpt missing split")
    else:
        split = baseline_ckpts[0].get("split", None)
        if split is None:
            raise ValueError("baseline ckpt missing split")
        for c in baseline_ckpts[1:]:
            if c.get("split", None) != split:
                raise ValueError("all baseline checkpoints must share split (or pass --split-from)")

    # --- Model 1 setup ---
    graph1 = load_graph(model_root, model_id=1, split_for_static="train")
    w1 = baseline_ckpts[0]["model_1"]["w"].cpu().numpy().astype(np.float32, copy=False)
    n1_m1 = int(graph1.n_1d)
    n2_m1 = int(graph1.n_2d)
    head_off1 = graph1.head_offset.cpu().numpy().astype(np.float32, copy=False)
    inv1_m1 = head_off1[:n1_m1].astype(np.float32, copy=False)
    bed2_m1 = head_off1[n1_m1:].astype(np.float32, copy=False)
    conn_src_m1 = graph1.conn_src_1d.cpu().numpy()
    conn_dst_m1 = graph1.conn_dst_2d.cpu().numpy()
    has_conn_m1 = (np.bincount(conn_src_m1, minlength=n1_m1) > 0).astype(np.float32, copy=False)
    bed_agg_m1 = aggregate_2d_to_1d_mean(
        bed2_m1[None, :],
        conn_src_1d=conn_src_m1,
        conn_dst_2d=conn_dst_m1,
        n_1d=n1_m1,
    )[0].astype(np.float32, copy=False)

    node_static_m1_2d = graph1.node_static_2d.float().to(device)
    node_static_m1_1d_raw = graph1.node_static_1d.float().to(device)
    node_static_m1_1d_aug1 = build_static_features_m2_1d(graph1).to(device)
    node_static_m1_1d_aug2 = build_static_features_m2_1d_v2(graph1).to(device)
    m1_nbr_src = graph1.edge_index_1d[0].cpu().numpy()
    m1_nbr_dst = graph1.edge_index_1d[1].cpu().numpy()

    m1_resid2d = None
    m1_edge_index_2d = None
    m1_edge_deg_inv_2d = None
    if str(a.m1_resid2d_ckpt):
        ck = _load_torch(Path(a.m1_resid2d_ckpt))
        if str(ck.get("kind", "")) != "residual_m1_2d":
            raise ValueError("--m1-resid2d-ckpt must be kind residual_m1_2d")
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        m1_resid2d = ResidualNodeGRU(n_nodes=n2_m1, cfg=mcfg).to(device)
        m1_resid2d.load_state_dict(ck["state_dict"])
        m1_resid2d.eval()
        need_edges = (len(m1_resid2d.graph_mix) > 0) or (len(getattr(m1_resid2d, "graph_mix_post", ())) > 0) or (int(getattr(mcfg, "dyn_dim", 5)) in (7, 9))
        if need_edges:
            m1_edge_index_2d = graph1.edge_index_2d.to(device)
            deg = torch.bincount(graph1.edge_index_2d[1].cpu(), minlength=int(n2_m1)).clamp(min=1).to(torch.float32)
            m1_edge_deg_inv_2d = (1.0 / deg).to(device)

    m1_stages: list[tuple[ResidualNodeGRU, torch.Tensor, bool]] = []
    for pth in a.m1_resid1d_ckpt:
        ck = _load_torch(Path(pth))
        if str(ck.get("kind", "")) != "residual_m1_1d":
            raise ValueError("--m1-resid1d-ckpt must be kind residual_m1_1d")
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        if int(mcfg.dyn_dim) not in (10, 13):
            raise ValueError(f"unsupported Model1 1D dyn_dim={mcfg.dyn_dim} (expected 10 or 13)")
        use_nbr = int(mcfg.dyn_dim) == 13
        m = ResidualNodeGRU(n_nodes=n1_m1, cfg=mcfg).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        node_static = _pick_node_static(
            expected_dim=int(mcfg.static_dim),
            raw=node_static_m1_1d_raw,
            aug1=node_static_m1_1d_aug1,
            aug2=node_static_m1_1d_aug2,
        )
        m1_stages.append((m, node_static, use_nbr))

    # --- Model 2 setup ---
    graph2 = load_graph(model_root, model_id=2, split_for_static="train")
    n1 = int(graph2.n_1d)
    n2 = int(graph2.n_2d)

    head_off = graph2.head_offset.cpu().numpy().astype(np.float32, copy=False)
    bed2 = head_off[n1:].astype(np.float32, copy=False)
    conn_src = graph2.conn_src_1d.cpu().numpy()
    conn_dst = graph2.conn_dst_2d.cpu().numpy()

    # Static features for residual models.
    node_static_2d = graph2.node_static_2d.float().to(device)
    node_static_1d_raw = graph2.node_static_1d.float().to(device)
    node_static_1d_aug1 = build_static_features_m2_1d(graph2).to(device)
    node_static_1d_aug2 = build_static_features_m2_1d_v2(graph2).to(device)

    # Masks/graph metadata for residual stages.
    masks = _mask_inputs(model_root=model_root, graph2=graph2)
    lap_w_dir, diam_max_1d = _pipe_weights(model_root=model_root, graph2=graph2, masks=masks)
    edgeflow_in_slots, edgeflow_out_slots = build_edge_slot_index(
        edge_from=masks["src_dir"],
        edge_to=masks["dst_dir"],
        edge_weight=lap_w_dir,
        n_1d=int(n1),
        slots_per_dir=2,
    )
    surface_edge_weight = load_surface_edge_weight(model_root=model_root, model_id=2, split_for_static="train")
    src2_full = graph2.edge_index_2d[0].cpu().numpy().astype(np.int64, copy=False)
    dst2_full = graph2.edge_index_2d[1].cpu().numpy().astype(np.int64, copy=False)
    e2_dir = int(src2_full.shape[0] // 2)
    surface_slot_edges, surface_slot_sign = build_coupled_surface_slot_index(
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        edge_from=src2_full[:e2_dir],
        edge_to=dst2_full[:e2_dir],
        edge_weight=surface_edge_weight,
        n_1d=int(n1),
        slots_per_node=6,
    )
    local2d_center_cell, local2d_neighbor_slots = build_coupled_neighbor_node_index(
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        edge_from=src2_full[:e2_dir],
        edge_to=dst2_full[:e2_dir],
        edge_weight=surface_edge_weight,
        n_1d=int(n1),
        slots_per_node=4,
    )
    edge_index_full = graph2.edge_index_1d.to(device)
    E1 = int(graph2.edge_index_1d.shape[1] // 2)
    edge_index_dir = edge_index_full[:, :E1]

    deg_full = torch.bincount(graph2.edge_index_1d[1].cpu(), minlength=int(n1)).clamp(min=1).to(torch.float32)
    edge_deg_inv_full = (1.0 / deg_full).to(device)
    deg_dir = torch.bincount(graph2.edge_index_1d[1, :E1].cpu(), minlength=int(n1)).clamp(min=1).to(torch.float32)
    edge_deg_inv_dir = (1.0 / deg_dir).to(device)

    # Weighted neighbor mean for graph mixing (pipe conductance proxy). Only used if a ckpt was trained with it.
    w_dir = torch.from_numpy(lap_w_dir.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    if int(w_dir.shape[0]) != int(E1):
        raise ValueError("lap_w_dir length mismatch vs pipe-direction edge count")
    edge_weight_dir = w_dir
    edge_weight_full = torch.cat([w_dir, w_dir], dim=0)
    d_full = torch.zeros((int(n1),), device=device, dtype=torch.float32)
    d_full.index_add_(0, edge_index_full[1], edge_weight_full)
    edge_deg_inv_full_w = (1.0 / d_full.clamp(min=1.0)).to(device)
    d_dir = torch.zeros((int(n1),), device=device, dtype=torch.float32)
    d_dir.index_add_(0, edge_index_dir[1], edge_weight_dir)
    edge_deg_inv_dir_w = (1.0 / d_dir.clamp(min=1.0)).to(device)

    def _pick_edges(cfg_like: dict) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor | None]:
        use_dir = str(cfg_like.get("graph_mix_edge", "full")) == "dir"
        use_w = bool(cfg_like.get("graph_mix_weighted", False))
        if use_dir:
            if use_w:
                return edge_index_dir, edge_deg_inv_dir_w, edge_weight_dir
            return edge_index_dir, edge_deg_inv_dir, None
        if use_w:
            return edge_index_full, edge_deg_inv_full_w, edge_weight_full
        return edge_index_full, edge_deg_inv_full, None

    # Optional Model2 residuals.
    resid2d = None
    m2_edge_index_2d = None
    m2_edge_deg_inv_2d = None
    if str(a.resid2d_ckpt):
        ck2d = _load_torch(Path(a.resid2d_ckpt))
        if str(ck2d.get("kind", "")) != "residual_m2_2d":
            raise ValueError(f"unsupported resid2d kind: {ck2d.get('kind')}")
        resid2d_dyn_feat = str((ck2d.get("cfg", {}) or {}).get("dyn_feat", ""))
        m2d_cfg = ResidualGRUConfig(**ck2d["model_cfg"])
        resid2d = ResidualNodeGRU(n_nodes=n2, cfg=m2d_cfg).to(device)
        resid2d.load_state_dict(ck2d["state_dict"])
        resid2d.eval()
        need_edges = (len(resid2d.graph_mix) > 0) or (len(getattr(resid2d, "graph_mix_post", ())) > 0) or (int(getattr(m2d_cfg, "dyn_dim", 5)) in (7, 9))
        if need_edges:
            m2_edge_index_2d = graph2.edge_index_2d.to(device)
            deg = torch.bincount(graph2.edge_index_2d[1].cpu(), minlength=int(n2)).clamp(min=1).to(torch.float32)
            m2_edge_deg_inv_2d = (1.0 / deg).to(device)
    else:
        resid2d_dyn_feat = ""

    stages: list[tuple[dict, ResidualNodeGRU, dict, torch.Tensor, np.ndarray | None]] = []
    need_knn_k = 0
    coupling_paths: set[str] = set()
    inlet_ckpt_paths: set[str] = set()
    aux_baseline_ckpt_paths: set[str] = set()
    aux_pre_residual_ckpt_paths: set[str] = set()
    edgeflow_ckpt_paths: set[str] = set()
    surfaceflow_ckpt_paths: set[str] = set()
    volagg_ckpt_paths: set[str] = set()
    for p in a.resid1d_ckpt:
        ck = _load_torch(Path(p))
        if str(ck.get("kind", "")) != "residual_m2_1d":
            raise ValueError(f"unsupported 1D residual kind: {ck.get('kind')}")
        cfg = ck.get("cfg", {}) or {}
        dv = int(cfg.get("dyn_feat_version", 1) or 1)
        if dv in (3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            need_knn_k = max(need_knn_k, int(cfg.get("y2_knn_k", 0) or 0))
        if int(cfg.get("warm_ctx_version", 0) or 0) in (1, 2) or int(cfg.get("warm_seq_version", 0) or 0) in (1,):
            need_knn_k = max(need_knn_k, int(cfg.get("y2_knn_k", 0) or 0))
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        m = ResidualNodeGRU(n_nodes=n1, cfg=mcfg).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()

        node_static = _pick_node_static(
            expected_dim=int(mcfg.static_dim),
            raw=node_static_1d_raw,
            aug1=node_static_1d_aug1,
            aug2=node_static_1d_aug2,
        )
        node_mask = _mask_from_cfg(cfg, masks=masks, n1=n1)
        coupling_path = str(cfg.get("resid2d_ckpt_coupling", "") or "")
        if coupling_path:
            w_agg = float(cfg.get("resid2d_coupling_blend", 0.0) or 0.0)
            w_knn = float(cfg.get("resid2d_coupling_blend_knn", 0.0) or 0.0)
            if w_agg > 0.0 or w_knn > 0.0:
                coupling_paths.add(coupling_path)
        inlet_path = str(cfg.get("inlet_ckpt", "") or "")
        if inlet_path:
            inlet_ckpt_paths.add(inlet_path)
        aux_baseline_path = str(cfg.get("aux_baseline_ckpt", "") or "")
        if aux_baseline_path:
            aux_baseline_ckpt_paths.add(aux_baseline_path)
        for aux_pre_path in tuple(str(x) for x in (cfg.get("aux_pre_residual_ckpts", ()) or ())):
            if aux_pre_path:
                aux_pre_residual_ckpt_paths.add(aux_pre_path)
        edgeflow_path = str(cfg.get("edgeflow_ckpt", "") or "")
        if edgeflow_path:
            edgeflow_ckpt_paths.add(edgeflow_path)
        surfaceflow_path = str(cfg.get("surfaceflow_ckpt", "") or "")
        if surfaceflow_path:
            surfaceflow_ckpt_paths.add(surfaceflow_path)
        volagg_path = str(cfg.get("volagg_ckpt", "") or "")
        if volagg_path:
            volagg_ckpt_paths.add(volagg_path)
        stages.append((ck, m, cfg, node_static, node_mask))

    knn_idx = _knn_2d_idx(model_root=model_root, k=int(need_knn_k), graph2=graph2) if int(need_knn_k) > 0 else None
    coupling_resid2d_models: dict[str, tuple[ResidualNodeGRU, str]] = {}
    for path in sorted(coupling_paths):
        ck_cpl = _load_torch(Path(path))
        if str(ck_cpl.get("kind", "")) != "residual_m2_2d":
            raise ValueError(f"unsupported coupling 2D residual kind: {ck_cpl.get('kind')}")
        dyn_feat_cpl = str((ck_cpl.get("cfg", {}) or {}).get("dyn_feat", ""))
        mcfg_cpl = ResidualGRUConfig(**ck_cpl["model_cfg"])
        m_cpl = ResidualNodeGRU(n_nodes=n2, cfg=mcfg_cpl).to(device)
        m_cpl.load_state_dict(ck_cpl["state_dict"])
        m_cpl.eval()
        coupling_resid2d_models[path] = (m_cpl, dyn_feat_cpl)

    inlet_ckpts: dict[str, dict] = {}
    for path in sorted(inlet_ckpt_paths):
        ck_in = _load_torch(Path(path))
        if str((ck_in.get("model_2", {}) or {}).get("kind", "")) not in {
            "split_1d2d_coupled_inlet",
            "split_1d2d_coupled_storage_inlet",
            "aux_inlet_m2_1d_surface_slots",
        }:
            raise ValueError(f"unsupported inlet baseline kind: {path}")
        inlet_ckpts[path] = ck_in

    aux_baseline_ckpts: dict[str, dict] = {}
    for path in sorted(aux_baseline_ckpt_paths):
        ck_aux = _load_torch(Path(path))
        if str(ck_aux.get("kind", "")) != "baseline_arx":
            raise ValueError(f"unsupported aux baseline kind: {path}")
        aux_baseline_ckpts[path] = ck_aux

    aux_pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]] = {}
    for path in sorted(aux_pre_residual_ckpt_paths):
        ck_pre = _load_torch(Path(path))
        if str(ck_pre.get("kind", "")) != "residual_m2_1d":
            raise ValueError(f"unsupported aux pre-residual kind: {path}")
        cfg_pre = ck_pre.get("cfg", {}) or {}
        mcfg_pre = ResidualGRUConfig(**ck_pre["model_cfg"])
        m_pre = ResidualNodeGRU(n_nodes=n1, cfg=mcfg_pre).to(device)
        m_pre.load_state_dict(ck_pre["state_dict"])
        m_pre.eval()
        aux_pre_model_map[path] = (ck_pre, m_pre, cfg_pre)

    edgeflow_ckpts: dict[str, dict] = {}
    for path in sorted(edgeflow_ckpt_paths):
        ck_q = _load_torch(Path(path))
        if str(ck_q.get("kind", "")) != "aux_edgeflow_m2_1d":
            raise ValueError(f"unsupported edgeflow aux kind: {path}")
        edgeflow_ckpts[path] = ck_q

    surfaceflow_ckpts: dict[str, dict] = {}
    for path in sorted(surfaceflow_ckpt_paths):
        ck_q = _load_torch(Path(path))
        if str(ck_q.get("kind", "")) != "aux_surfaceflow_m2_2d":
            raise ValueError(f"unsupported surfaceflow aux kind: {path}")
        surfaceflow_ckpts[path] = ck_q

    volagg_ckpts: dict[str, dict] = {}
    for path in sorted(volagg_ckpt_paths):
        ck_v = _load_torch(Path(path))
        if str(ck_v.get("kind", "")) != "aux_volagg_m2_1d":
            raise ValueError(f"unsupported volagg aux kind: {path}")
        volagg_ckpts[path] = ck_v

    # --- Evaluate ---
    val1 = list(split["model_1"]["val"])
    val2 = list(split["model_2"]["val"])
    if int(a.max_events) > 0:
        val1 = val1[: int(a.max_events)]
        val2 = val2[: int(a.max_events)]

    scores: list[EventScore] = []

    # Model 1 proxy (baseline + optional residuals).
    for eid in tqdm(val1, desc="eval model 1 (val)", leave=False):
        ev = load_event(model_root, graph=graph1, split="train", event_id=eid, cache_dir=cache_dir)
        y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
        y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
        y_init = np.concatenate([y1_true, y2_true], axis=1)
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

        if m1_stages:
            y2_agg = aggregate_2d_to_1d_mean(
                y2_pred,
                conn_src_1d=conn_src_m1,
                conn_dst_2d=conn_dst_m1,
                n_1d=n1_m1,
            ).astype(np.float32, copy=False)
            y1_pred = y1_base.astype(np.float32, copy=True)
            for m, node_static_1d, use_nbr in m1_stages:
                y1_pred = _apply_resid1d_v1(
                    model=m,
                    node_static_1d=node_static_1d,
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
        else:
            y1_pred = y1_base
        scores.append(
            score_event_from_full_grids(
                model_id=1,
                event_id=int(eid),
                y_true_1d=y1_true,
                y_pred_1d=y1_pred,
                y_true_2d=y2_true,
                y_pred_2d=y2_pred,
                warmup=warmup,
            )
        )

    # Model 2 residual stack proxy.
    inv1 = head_off[:n1].astype(np.float32, copy=False)
    bed_agg = aggregate_2d_to_1d_mean(bed2[None, :], conn_src_1d=conn_src, conn_dst_2d=conn_dst, n_1d=n1)[0].astype(
        np.float32, copy=False
    )
    conn_area_1d = aggregate_2d_to_1d_sum(
        graph2.area_2d.cpu().numpy().astype(np.float32, copy=False)[None, :],
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        n_1d=n1,
    )[0].astype(np.float32, copy=False)
    has_conn = (np.bincount(conn_src, minlength=n1) > 0).astype(np.float32, copy=False)

    for eid in tqdm(val2, desc="eval model 2 (val)", leave=False):
        ev = load_event(model_root, graph=graph2, split="train", event_id=eid, cache_dir=cache_dir)
        y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
        y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
        inlet_1d_ev = ev.inlet_1d.numpy().astype(np.float32, copy=False) if ev.inlet_1d is not None else None
        edgeflow_1d_ev = None
        if edgeflow_ckpts:
            edgeflow_1d_ev = load_edge_flow_1d(
                model_root=model_root,
                model_id=2,
                split="train",
                event_id=eid,
                n_edges=int(graph2.edge_index_1d.shape[1] // 2),
                cache_dir=cache_dir,
            )
        surfaceflow_2d_ev = None
        if surfaceflow_ckpts:
            surfaceflow_2d_ev = load_edge_flow_2d(
                model_root=model_root,
                model_id=2,
                split="train",
                event_id=eid,
                n_edges=int(graph2.edge_index_2d.shape[1] // 2),
                cache_dir=cache_dir,
            )
        vol2_agg_ev = None
        if ev.volume_2d is not None:
            vol2_agg_ev = aggregate_2d_to_1d_sum(
                ev.volume_2d.numpy().astype(np.float32, copy=False),
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_1d=n1,
            ).astype(np.float32, copy=False)
        volagg_init_ev = vol2_agg_ev

        y1_base, y2_base = predict_model2_from_baseline_ckpts(
            baseline_ckpts,
            graph2=graph2,
            mixed_mode=str(a.mixed_mode),
            alpha_1d=float(a.alpha_1d),
            alpha_2d=float(a.alpha_2d),
            y1_init=y1_true,
            y2_init=y2_true,
            rain_2d=rain,
            q1_init=(ev.inlet_1d.numpy().astype(np.float32, copy=False) if ev.inlet_1d is not None else None),
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )

        if resid2d is not None:
            y2_corr = _apply_resid2d(
                resid2d=resid2d,
                node_static_2d=node_static_2d,
                y2_base=y2_base,
                rain_2d=rain,
                bed2=bed2,
                warmup=warmup,
                dyn_feat=resid2d_dyn_feat,
                y1_base=y1_base,
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_2d=n2,
                edge_index_2d=m2_edge_index_2d,
                edge_deg_inv_2d=m2_edge_deg_inv_2d,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        else:
            y2_corr = y2_base

        y2_corr_coupling: dict[str, np.ndarray] = {}
        for path, (model2d_cpl, dyn_feat_cpl) in coupling_resid2d_models.items():
            y2_corr_coupling[path] = _apply_resid2d(
                resid2d=model2d_cpl,
                node_static_2d=node_static_2d,
                y2_base=y2_base,
                rain_2d=rain,
                bed2=bed2,
                warmup=warmup,
                dyn_feat=dyn_feat_cpl,
                y1_base=y1_base,
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_2d=n2,
                edge_index_2d=m2_edge_index_2d,
                edge_deg_inv_2d=m2_edge_deg_inv_2d,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        y1_feat = y1_base.astype(np.float32, copy=True)
        for ck, m, cfg, node_static, node_mask in stages:
            use_nbr = bool(cfg.get("use_nbr_feats", False)) or bool(cfg.get("use_out_nbr_feats", False))
            pipe_dir = bool(cfg.get("nbr_pipe_dir", False))
            if use_nbr:
                src = masks["src_dir"] if pipe_dir else masks["src_full"]
                dst = masks["dst_dir"] if pipe_dir else masks["dst_full"]
            else:
                src = None
                dst = None

            coupling_path = str(cfg.get("resid2d_ckpt_coupling", "") or "")
            y2_agg, y2_knn_mean, y2_knn_max = _build_coupling_views(
                cfg_like=cfg,
                y2_main=y2_corr,
                y2_alt=y2_corr_coupling.get(coupling_path),
                bed2=bed2,
                conn_src=conn_src,
                conn_dst=conn_dst,
                n_1d=n1,
                knn_idx=knn_idx,
            )
            inlet_pred = _build_inlet_pred_views(
                cfg_like=cfg,
                inlet_ckpts=inlet_ckpts,
                surfaceflow_ckpts=surfaceflow_ckpts,
                graph2=graph2,
                y2_main=y2_corr,
                y2_alt=y2_corr_coupling.get(coupling_path),
                rain_2d=rain,
                q1_init=inlet_1d_ev,
                volagg_init=volagg_init_ev,
                q2edge_init=surfaceflow_2d_ev,
                warmup=warmup,
                surface_slot_edges=surface_slot_edges,
                surface_slot_sign=surface_slot_sign,
                center_cell=local2d_center_cell,
                neighbor_slots=local2d_neighbor_slots,
            )
            aux_y1_base = _build_aux_traj_views(
                cfg_like=cfg,
                aux_baseline_ckpts=aux_baseline_ckpts,
                aux_pre_model_map=aux_pre_model_map,
                graph2=graph2,
                y1_init=y1_true,
                y2_init=y2_true,
                rain_2d=rain,
                q1_init=inlet_1d_ev,
                vagg_init=volagg_init_ev,
                q2edge_init=surfaceflow_2d_ev,
                qedge_init=edgeflow_1d_ev,
                warmup=warmup,
                node_static_raw=node_static_1d_raw,
                node_static_aug1=node_static_1d_aug1,
                node_static_aug2=node_static_1d_aug2,
                masks=masks,
                y2_corr=y2_corr,
                y2_corr_coupling=y2_corr_coupling,
                bed2=bed2,
                conn_src=conn_src,
                conn_dst=conn_dst,
                n1=n1,
                knn_idx=knn_idx,
                inlet_ckpts=inlet_ckpts,
                edgeflow_ckpts=edgeflow_ckpts,
                surfaceflow_ckpts=surfaceflow_ckpts,
                volagg_ckpts=volagg_ckpts,
                surface_slot_edges=surface_slot_edges,
                surface_slot_sign=surface_slot_sign,
                local2d_center_cell=local2d_center_cell,
                local2d_neighbor_slots=local2d_neighbor_slots,
                inv1=inv1,
                bed_agg=bed_agg,
                has_conn=has_conn,
                lap_w_dir=lap_w_dir,
                diam_max_1d=diam_max_1d,
                conn_area_1d=conn_area_1d,
                vol2_agg_ev=vol2_agg_ev,
                edgeflow_in_slots=edgeflow_in_slots,
                edgeflow_out_slots=edgeflow_out_slots,
                edge_index_1d_full=edge_index_full,
                edge_deg_inv_1d_full=edge_deg_inv_full_w,
                edge_weight_1d_full=edge_weight_full,
                edge_index_1d_dir=edge_index_dir,
                edge_deg_inv_1d_dir=edge_deg_inv_dir_w,
                edge_weight_1d_dir=edge_weight_dir,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if int(cfg.get("dyn_feat_version", 1) or 1) == 17:
                edgeflow_pred = _build_edgeflow_pred_slot_views(
                    cfg_like=cfg,
                    edgeflow_ckpts=edgeflow_ckpts,
                    graph2=graph2,
                    y1_1d=y1_feat,
                    rain_2d=rain,
                    qedge_init=edgeflow_1d_ev,
                    warmup=warmup,
                    edge_in_slots=edgeflow_in_slots,
                    edge_out_slots=edgeflow_out_slots,
                )
            else:
                edgeflow_pred = _build_edgeflow_pred_views(
                    cfg_like=cfg,
                    edgeflow_ckpts=edgeflow_ckpts,
                    graph2=graph2,
                    y1_1d=y1_feat,
                    rain_2d=rain,
                    qedge_init=edgeflow_1d_ev,
                    warmup=warmup,
                )
            volagg_pred = _build_volagg_pred_views(
                cfg_like=cfg,
                volagg_ckpts=volagg_ckpts,
                graph2=graph2,
                y2_main=y2_corr,
                y2_alt=y2_corr_coupling.get(coupling_path),
                rain_2d=rain,
                vagg_init=volagg_init_ev,
                warmup=warmup,
            )
            surfaceflow_pred = _build_surfaceflow_pred_views(
                cfg_like=cfg,
                surfaceflow_ckpts=surfaceflow_ckpts,
                graph2=graph2,
                y2_main=y2_corr,
                y2_alt=y2_corr_coupling.get(coupling_path),
                rain_2d=rain,
                q2edge_init=surfaceflow_2d_ev,
                warmup=warmup,
            )
            if int(cfg.get("dyn_feat_version", 1) or 1) in (19, 20, 21, 22):
                surfaceflow_slot_pred = _build_surfaceflow_pred_slot_views(
                    cfg_like=cfg,
                    surfaceflow_ckpts=surfaceflow_ckpts,
                    graph2=graph2,
                    y2_main=y2_corr,
                    y2_alt=y2_corr_coupling.get(coupling_path),
                    rain_2d=rain,
                    q2edge_init=surfaceflow_2d_ev,
                    warmup=warmup,
                    surface_slot_edges=surface_slot_edges,
                    surface_slot_sign=surface_slot_sign,
                )
            else:
                surfaceflow_slot_pred = None
            if int(cfg.get("dyn_feat_version", 1) or 1) == 21:
                blend_local = float(cfg.get("resid2d_coupling_blend", 0.0) or 0.0)
                y2_local_src = y2_corr
                y2_alt_local = y2_corr_coupling.get(coupling_path)
                if y2_alt_local is not None and blend_local > 0.0:
                    y2_local_src = ((1.0 - blend_local) * y2_corr + blend_local * y2_alt_local).astype(np.float32, copy=False)
                local2d_node_pred = build_coupled_neighbor_node_features(
                    y2_2d=y2_local_src,
                    bed_2d=bed2,
                    center_cell=local2d_center_cell,
                    neighbor_slots=local2d_neighbor_slots,
                )
            else:
                local2d_node_pred = None
            eidx_1d, einv_1d, ew_1d = _pick_edges(cfg)
            y1_feat = _apply_resid1d_ckpt(
                ckpt=ck,
                model=m,
                node_static=node_static,
                cfg=cfg,
                node_mask=node_mask,
                expert_group_idx=_expert_group_idx_topo3(masks=masks, n1=n1) if bool(cfg.get("expert_group_topo3", False)) else None,
                y1_feat=y1_feat,
                y2_agg=y2_agg,
                rain_2d=rain,
                warmup=warmup,
                inv1=inv1,
                bed_agg=bed_agg,
                has_conn=has_conn,
                nbr_src=src,
                nbr_dst=dst,
                y2_knn_mean=y2_knn_mean,
                y2_knn_max=y2_knn_max,
                lap_src=masks["src_dir"],
                lap_dst=masks["dst_dir"],
                lap_w=lap_w_dir,
                diam_max_1d=diam_max_1d,
                base_area_1d=masks.get("base_area", None),
                conn_area_1d=conn_area_1d,
                inlet_1d=inlet_1d_ev,
                vol2_agg=vol2_agg_ev,
                    inlet_pred_1d=inlet_pred,
                    aux_y1_base=aux_y1_base,
                    edgeflow_node_feats=edgeflow_pred,
                    surfaceflow_node_feats=surfaceflow_pred,
                    surfaceflow_slot_feats=surfaceflow_slot_pred,
                    local2d_node_feats=local2d_node_pred,
                    volagg_pred_1d=volagg_pred,
                    edge_index_1d=eidx_1d,
                edge_deg_inv_1d=einv_1d,
                edge_weight_1d=ew_1d,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        scores.append(
            score_event_from_full_grids(
                model_id=2,
                event_id=int(eid),
                y_true_1d=y1_true,
                y_pred_1d=y1_feat,
                y_true_2d=y2_true,
                y_pred_2d=y2_corr,
                warmup=warmup,
            )
        )

    s1 = float(np.mean([x.score for x in scores if x.model_id == 1]))
    s2 = float(np.mean([x.score for x in scores if x.model_id == 2]))
    overall = float(score_overall(scores))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    m1_1d = _mean([x.score_1d for x in scores if x.model_id == 1])
    m1_2d = _mean([x.score_2d for x in scores if x.model_id == 1])
    m2_1d = _mean([x.score_1d for x in scores if x.model_id == 2])
    m2_2d = _mean([x.score_2d for x in scores if x.model_id == 2])

    print(f"model1 mean={s1:.9f} (1d={m1_1d:.9f} 2d={m1_2d:.9f})  n_events={len(val1)}")
    print(f"model2 mean={s2:.9f} (1d={m2_1d:.9f} 2d={m2_2d:.9f})  n_events={len(val2)}")
    print(f"overall={overall:.9f}")


if __name__ == "__main__":
    main()
