from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from urbanflood.baseline import aggregate_2d_to_1d_mean, aggregate_2d_to_1d_sum, predict_model2_from_baseline_ckpts
from urbanflood.data import load_event, load_graph
from urbanflood.edgeflow_aux import build_edge_flow_slot_features, build_edge_slot_index, load_edge_flow_1d
from urbanflood.metric import STD_DEV_DICT
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import build_static_features_m2_1d, build_static_features_m2_1d_v2
from urbanflood.residual_train import (
    _apply_resid1d_ckpt,
    _apply_resid2d,
    _build_aux_traj_views,
    _build_coupling_views,
    _build_edgeflow_pred_slot_views,
    _build_edgeflow_pred_views,
    _expert_group_idx_topo3,
    _build_inlet_pred_views,
    _build_surfaceflow_pred_slot_views,
    _build_surfaceflow_pred_views,
    _build_volagg_pred_views,
    _knn_2d_idx,
    _load_torch,
    _mask_from_cfg,
    _mask_inputs,
    _pick_node_static,
    _pipe_weights,
)
from urbanflood.surfaceflow_aux import (
    build_coupled_neighbor_node_features,
    build_coupled_neighbor_node_index,
    build_coupled_surface_slot_features,
    build_coupled_surface_slot_index,
    load_edge_flow_2d,
    load_surface_edge_weight,
)
from urbanflood.utils import seed_everything


@dataclass
class StackBundle:
    baseline_ckpts: list[dict]
    resid2d: ResidualNodeGRU
    resid2d_dyn_feat: str
    edge_index_2d: torch.LongTensor | None
    edge_deg_inv_2d: torch.Tensor | None
    stages: list[tuple[dict, ResidualNodeGRU, dict, torch.Tensor, np.ndarray | None]]
    knn_idx: np.ndarray | None
    coupling_resid2d_models: dict[str, tuple[ResidualNodeGRU, str]]
    inlet_ckpts: dict[str, dict]
    aux_baseline_ckpts: dict[str, dict]
    aux_pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]]
    edgeflow_ckpts: dict[str, dict]
    surfaceflow_ckpts: dict[str, dict]
    volagg_ckpts: dict[str, dict]


class BlendGateMLP(nn.Module):
    def __init__(self, dyn_dim: int, static_dim: int, hidden_dim: int, dropout: float, correction_ft: float = 0.0) -> None:
        super().__init__()
        self.correction_ft = float(correction_ft)
        in_dim = int(dyn_dim) + int(static_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        self.corr_head = nn.Linear(int(hidden_dim), 1) if self.correction_ft > 0.0 else None
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.0)
            if self.corr_head is not None:
                self.corr_head.bias.zero_()

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dyn = x_dyn.float()
        static_expand = x_static.float().unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        h = self.trunk(torch.cat([x_dyn, static_expand], dim=-1))
        gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        if self.corr_head is None:
            corr = torch.zeros_like(gate)
        else:
            corr = torch.tanh(self.corr_head(h).squeeze(-1)) * self.correction_ft
        return gate, corr


class BlendGateWarmMLP(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        warm_seq_dim: int,
        warmup: int,
        hidden_dim: int,
        dropout: float,
        correction_ft: float = 0.0,
    ) -> None:
        super().__init__()
        self.correction_ft = float(correction_ft)
        self.warmup = int(warmup)
        warm_in = int(warm_seq_dim * warmup)
        self.warm_proj = nn.Sequential(
            nn.LayerNorm(warm_in),
            nn.Linear(warm_in, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        in_dim = int(dyn_dim) + int(static_dim) + int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        self.corr_head = nn.Linear(int(hidden_dim), 1) if self.correction_ft > 0.0 else None
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.0)
            if self.corr_head is not None:
                self.corr_head.bias.zero_()

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor, warm_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dyn = x_dyn.float()
        static_expand = x_static.float().unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        warm_flat = warm_seq.float().reshape(int(warm_seq.shape[0]), -1)
        warm_ctx = self.warm_proj(warm_flat)
        warm_expand = warm_ctx.unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        h = self.trunk(torch.cat([x_dyn, static_expand, warm_expand], dim=-1))
        gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        if self.corr_head is None:
            corr = torch.zeros_like(gate)
        else:
            corr = torch.tanh(self.corr_head(h).squeeze(-1)) * self.correction_ft
        return gate, corr


class BlendGateLinear(nn.Module):
    def __init__(self, dyn_dim: int, static_dim: int, correction_ft: float = 0.0) -> None:
        super().__init__()
        self.correction_ft = float(correction_ft)
        in_dim = int(dyn_dim) + int(static_dim)
        self.gate_head = nn.Linear(in_dim, 1)
        self.corr_head = nn.Linear(in_dim, 1) if self.correction_ft > 0.0 else None
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.0)
            if self.corr_head is not None:
                self.corr_head.bias.zero_()

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dyn = x_dyn.float()
        static_expand = x_static.float().unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        x = torch.cat([x_dyn, static_expand], dim=-1)
        gate = torch.sigmoid(self.gate_head(x).squeeze(-1))
        if self.corr_head is None:
            corr = torch.zeros_like(gate)
        else:
            corr = torch.tanh(self.corr_head(x).squeeze(-1)) * self.correction_ft
        return gate, corr


class BlendGateGroupedMLP(nn.Module):
    def __init__(self, dyn_dim: int, static_dim: int, hidden_dim: int, dropout: float, correction_ft: float = 0.0, n_groups: int = 3) -> None:
        super().__init__()
        self.correction_ft = float(correction_ft)
        self.n_groups = int(n_groups)
        in_dim = int(dyn_dim) + int(static_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gate_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(self.n_groups)])
        self.corr_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(self.n_groups)]) if self.correction_ft > 0.0 else None
        with torch.no_grad():
            for head in self.gate_heads:
                head.bias.fill_(-2.0)
            if self.corr_heads is not None:
                for head in self.corr_heads:
                    head.bias.zero_()

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor, group_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dyn = x_dyn.float()
        static_expand = x_static.float().unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        h = self.trunk(torch.cat([x_dyn, static_expand], dim=-1))
        gate_logits = torch.stack([head(h).squeeze(-1) for head in self.gate_heads], dim=-1)
        gather_idx = group_idx.long().view(1, -1, 1).expand(int(h.shape[0]), -1, 1)
        gate = torch.sigmoid(torch.gather(gate_logits, dim=-1, index=gather_idx).squeeze(-1))
        if self.corr_heads is None:
            corr = torch.zeros_like(gate)
        else:
            corr_raw = torch.stack([head(h).squeeze(-1) for head in self.corr_heads], dim=-1)
            corr = torch.tanh(torch.gather(corr_raw, dim=-1, index=gather_idx).squeeze(-1)) * self.correction_ft
        return gate, corr


class BlendGateGRU(nn.Module):
    def __init__(self, dyn_dim: int, static_dim: int, hidden_dim: int, dropout: float, correction_ft: float = 0.0) -> None:
        super().__init__()
        self.correction_ft = float(correction_ft)
        self.gru = nn.GRU(
            input_size=int(dyn_dim) + int(static_dim),
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.trunk = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gate_head = nn.Linear(int(hidden_dim), 1)
        self.corr_head = nn.Linear(int(hidden_dim), 1) if self.correction_ft > 0.0 else None
        with torch.no_grad():
            self.gate_head.bias.fill_(-2.0)
            if self.corr_head is not None:
                self.corr_head.bias.zero_()

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dyn = x_dyn.float()
        static_expand = x_static.float().unsqueeze(0).expand(int(x_dyn.shape[0]), -1, -1)
        x = torch.cat([x_dyn, static_expand], dim=-1).permute(1, 0, 2)
        h, _ = self.gru(x)
        z = self.trunk(h)
        gate = torch.sigmoid(self.gate_head(z).squeeze(-1).permute(1, 0))
        if self.corr_head is None:
            corr = torch.zeros_like(gate)
        else:
            corr = torch.tanh(self.corr_head(z).squeeze(-1).permute(1, 0)) * self.correction_ft
        return gate, corr


def build_pipe_neighbor_node_features(
    *,
    y1_1d: np.ndarray,
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    edge_weight: np.ndarray,
    n_1d: int,
) -> np.ndarray:
    y = np.asarray(y1_1d, dtype=np.float32)
    src = np.asarray(edge_from, dtype=np.int64)
    dst = np.asarray(edge_to, dtype=np.int64)
    w = np.asarray(edge_weight, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != int(n_1d):
        raise ValueError("y1_1d must be [T, N1]")
    if src.shape != dst.shape or src.shape != w.shape:
        raise ValueError("edge arrays must have the same shape")

    T, n1 = y.shape
    if int(n1) != int(n_1d):
        raise ValueError("n_1d mismatch")

    up_den = np.bincount(dst, weights=w, minlength=n1).astype(np.float32, copy=False)
    down_den = np.bincount(src, weights=w, minlength=n1).astype(np.float32, copy=False)

    up_mean = np.zeros((T, n1), dtype=np.float32)
    down_mean = np.zeros((T, n1), dtype=np.float32)
    for t in range(T):
        up_num = np.bincount(dst, weights=w * y[t, src], minlength=n1).astype(np.float32, copy=False)
        down_num = np.bincount(src, weights=w * y[t, dst], minlength=n1).astype(np.float32, copy=False)
        up_mean[t] = up_num / np.maximum(up_den, 1e-6)
        down_mean[t] = down_num / np.maximum(down_den, 1e-6)

    up_prev = np.empty_like(up_mean)
    up_prev[0] = up_mean[0]
    up_prev[1:] = up_mean[:-1]
    down_prev = np.empty_like(down_mean)
    down_prev[0] = down_mean[0]
    down_prev[1:] = down_mean[:-1]
    up_delta = up_mean - up_prev
    down_delta = down_mean - down_prev

    feats = np.stack(
        [
            up_mean,
            down_mean,
            up_mean - y,
            down_mean - y,
            up_delta,
            down_delta,
            up_mean - down_mean,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return feats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--best-baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--best-resid2d-ckpt", type=str, required=True)
    p.add_argument("--best-resid1d-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--alt-baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--alt-resid2d-ckpt", type=str, required=True)
    p.add_argument("--alt-resid1d-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--split-from", type=str, default="")
    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--arch", type=str, default="mlp", choices=["linear", "mlp", "group_mlp", "gru", "warm_mlp"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-source-scale", type=float, default=0.0)
    p.add_argument("--loss-ba0-scale", type=float, default=0.0)
    p.add_argument("--loss-special-scale", type=float, default=0.0)
    p.add_argument("--loss-q-scale", type=float, default=0.25)
    p.add_argument("--correction-ft", type=float, default=0.0)
    p.add_argument("--correction-penalty", type=float, default=0.0)
    p.add_argument("--route-loss-scale", type=float, default=0.0)
    p.add_argument("--route-temp", type=float, default=0.02)
    p.add_argument("--warm-ctx-version", type=int, default=0)
    p.add_argument("--warm-seq-version", type=int, default=0)
    p.add_argument("--future-local2d-version", type=int, default=0)
    p.add_argument("--future-local1d-version", type=int, default=0)
    p.add_argument("--scope-mode", type=str, default="none", choices=["none", "source_ba0", "source_ba0_qpos"])
    p.add_argument("--scope-q-thresh", type=float, default=0.0)
    p.add_argument("--amp-bf16", dest="amp_bf16", action="store_true")
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false")
    p.set_defaults(amp_bf16=True)
    p.add_argument("--m1-mean", type=float, default=0.001453301)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _build_graph_context(*, model_root: Path, graph2, device: torch.device) -> dict:
    n1 = int(graph2.n_1d)
    n2 = int(graph2.n_2d)
    head_off = graph2.head_offset.cpu().numpy().astype(np.float32, copy=False)
    inv1 = head_off[:n1].astype(np.float32, copy=False)
    bed2 = head_off[n1:].astype(np.float32, copy=False)
    conn_src = graph2.conn_src_1d.cpu().numpy()
    conn_dst = graph2.conn_dst_2d.cpu().numpy()

    node_static_2d = graph2.node_static_2d.float().to(device)
    node_static_1d_raw = graph2.node_static_1d.float().to(device)
    node_static_1d_aug1 = build_static_features_m2_1d(graph2).to(device)
    node_static_1d_aug2 = build_static_features_m2_1d_v2(graph2).to(device)

    masks = _mask_inputs(model_root=model_root, graph2=graph2)
    lap_w_dir, diam_max_1d = _pipe_weights(model_root=model_root, graph2=graph2, masks=masks)

    edgeflow_in_slots, edgeflow_out_slots = build_edge_slot_index(
        edge_from=masks["src_dir"],
        edge_to=masks["dst_dir"],
        edge_weight=lap_w_dir,
        n_1d=n1,
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
        n_1d=n1,
        slots_per_node=6,
    )
    local2d_center_cell, local2d_neighbor_slots = build_coupled_neighbor_node_index(
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        edge_from=src2_full[:e2_dir],
        edge_to=dst2_full[:e2_dir],
        edge_weight=surface_edge_weight,
        n_1d=n1,
        slots_per_node=4,
    )

    edge_index_full = graph2.edge_index_1d.to(device)
    edge_count_dir = int(graph2.edge_index_1d.shape[1] // 2)
    edge_index_dir = edge_index_full[:, :edge_count_dir]
    deg_full = torch.bincount(graph2.edge_index_1d[1].cpu(), minlength=n1).clamp(min=1).to(torch.float32)
    edge_deg_inv_full = (1.0 / deg_full).to(device)
    deg_dir = torch.bincount(graph2.edge_index_1d[1, :edge_count_dir].cpu(), minlength=n1).clamp(min=1).to(torch.float32)
    edge_deg_inv_dir = (1.0 / deg_dir).to(device)
    edge_weight_dir = torch.from_numpy(lap_w_dir.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
    edge_weight_full = torch.cat([edge_weight_dir, edge_weight_dir], dim=0)
    d_full = torch.zeros((n1,), device=device, dtype=torch.float32)
    d_full.index_add_(0, edge_index_full[1], edge_weight_full)
    edge_deg_inv_full_w = (1.0 / d_full.clamp(min=1.0)).to(device)
    d_dir = torch.zeros((n1,), device=device, dtype=torch.float32)
    d_dir.index_add_(0, edge_index_dir[1], edge_weight_dir)
    edge_deg_inv_dir_w = (1.0 / d_dir.clamp(min=1.0)).to(device)

    deg_conn = np.bincount(conn_src, minlength=n1).astype(np.float32, copy=False)
    has_conn = (deg_conn > 0).astype(np.float32, copy=False)
    conn_area_1d = aggregate_2d_to_1d_sum(
        graph2.area_2d.cpu().numpy().astype(np.float32, copy=False)[None, :],
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        n_1d=n1,
    )[0].astype(np.float32, copy=False)
    bed_agg = np.zeros((n1,), dtype=np.float32)
    for src, dst in zip(conn_src.tolist(), conn_dst.tolist()):
        bed_agg[src] += bed2[dst]
    bed_agg = bed_agg / np.maximum(deg_conn, 1.0)

    base_area = np.asarray(masks["base_area"], dtype=np.float32)
    depth = np.maximum(np.asarray(masks["depth"], dtype=np.float32), 1e-3)
    static_gate = np.stack(
        [
            np.asarray(masks["mask_source"], dtype=np.float32),
            np.asarray(masks["mask_ba0"], dtype=np.float32),
            (np.asarray(masks["indeg"], dtype=np.int64) >= 2).astype(np.float32),
            np.clip(np.asarray(masks["indeg"], dtype=np.float32) / 4.0, 0.0, 1.0),
            np.clip(base_area / max(float(np.nanmax(base_area)), 1.0), 0.0, 1.0),
            np.clip(depth / max(float(np.nanmax(depth)), 1.0), 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    expert_group_idx_topo3 = torch.from_numpy(_expert_group_idx_topo3(masks=masks, n1=n1)).to(device=device, dtype=torch.long)

    return {
        "n1": n1,
        "n2": n2,
        "inv1": inv1,
        "bed2": bed2,
        "conn_src": conn_src,
        "conn_dst": conn_dst,
        "node_static_2d": node_static_2d,
        "node_static_1d_raw": node_static_1d_raw,
        "node_static_1d_aug1": node_static_1d_aug1,
        "node_static_1d_aug2": node_static_1d_aug2,
        "masks": masks,
        "lap_w_dir": lap_w_dir,
        "diam_max_1d": diam_max_1d,
        "edgeflow_in_slots": edgeflow_in_slots,
        "edgeflow_out_slots": edgeflow_out_slots,
        "surface_slot_edges": surface_slot_edges,
        "surface_slot_sign": surface_slot_sign,
        "local2d_center_cell": local2d_center_cell,
        "local2d_neighbor_slots": local2d_neighbor_slots,
        "edge_index_full": edge_index_full,
        "edge_deg_inv_full": edge_deg_inv_full,
        "edge_deg_inv_full_w": edge_deg_inv_full_w,
        "edge_weight_full": edge_weight_full,
        "edge_index_dir": edge_index_dir,
        "edge_deg_inv_dir": edge_deg_inv_dir,
        "edge_deg_inv_dir_w": edge_deg_inv_dir_w,
        "edge_weight_dir": edge_weight_dir,
        "has_conn": has_conn,
        "conn_area_1d": conn_area_1d,
        "bed_agg": bed_agg,
        "static_gate": static_gate,
        "expert_group_idx_topo3": expert_group_idx_topo3,
        "depth": depth,
    }


def _build_stack_bundle(
    *,
    baseline_paths: list[str],
    resid2d_path: str,
    resid1d_paths: list[str],
    graph2,
    graph_ctx: dict,
    device: torch.device,
    model_root: Path,
) -> StackBundle:
    baseline_ckpts = [_load_torch(Path(path)) for path in baseline_paths]
    ck2d = _load_torch(Path(resid2d_path))
    m2d_cfg = ResidualGRUConfig(**ck2d["model_cfg"])
    resid2d = ResidualNodeGRU(n_nodes=int(graph2.n_2d), cfg=m2d_cfg).to(device)
    resid2d.load_state_dict(ck2d["state_dict"])
    resid2d.eval()
    resid2d_dyn_feat = str((ck2d.get("cfg", {}) or {}).get("dyn_feat", ""))
    need_edges = (len(resid2d.graph_mix) > 0) or (len(getattr(resid2d, "graph_mix_post", ())) > 0) or (int(getattr(m2d_cfg, "dyn_dim", 5)) in (7, 9))
    edge_index_2d = graph2.edge_index_2d.to(device) if need_edges else None
    if need_edges:
        deg = torch.bincount(graph2.edge_index_2d[1].cpu(), minlength=int(graph2.n_2d)).clamp(min=1).to(torch.float32)
        edge_deg_inv_2d = (1.0 / deg).to(device)
    else:
        edge_deg_inv_2d = None

    stages: list[tuple[dict, ResidualNodeGRU, dict, torch.Tensor, np.ndarray | None]] = []
    need_knn_k = 0
    coupling_paths: set[str] = set()
    inlet_paths: set[str] = set()
    aux_baseline_paths: set[str] = set()
    aux_pre_paths: set[str] = set()
    edgeflow_paths: set[str] = set()
    surfaceflow_paths: set[str] = set()
    volagg_paths: set[str] = set()
    for path in resid1d_paths:
        ck = _load_torch(Path(path))
        cfg = ck.get("cfg", {}) or {}
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        model = ResidualNodeGRU(n_nodes=int(graph2.n_1d), cfg=mcfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        node_static = _pick_node_static(
            expected_dim=int(mcfg.static_dim),
            raw=graph_ctx["node_static_1d_raw"],
            aug1=graph_ctx["node_static_1d_aug1"],
            aug2=graph_ctx["node_static_1d_aug2"],
        )
        node_mask = _mask_from_cfg(cfg, masks=graph_ctx["masks"], n1=int(graph2.n_1d))
        stages.append((ck, model, cfg, node_static, node_mask))

        dyn_ver = int(cfg.get("dyn_feat_version", 1) or 1)
        if dyn_ver in (3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            need_knn_k = max(need_knn_k, int(cfg.get("y2_knn_k", 0) or 0))
        if int(cfg.get("warm_ctx_version", 0) or 0) in (1,):
            need_knn_k = max(need_knn_k, int(cfg.get("y2_knn_k", 0) or 0))
        coupling_path = str(cfg.get("resid2d_ckpt_coupling", "") or "")
        if coupling_path and (float(cfg.get("resid2d_coupling_blend", 0.0) or 0.0) > 0.0 or float(cfg.get("resid2d_coupling_blend_knn", 0.0) or 0.0) > 0.0):
            coupling_paths.add(coupling_path)
        for key, bag in (
            ("inlet_ckpt", inlet_paths),
            ("aux_baseline_ckpt", aux_baseline_paths),
            ("edgeflow_ckpt", edgeflow_paths),
            ("surfaceflow_ckpt", surfaceflow_paths),
            ("volagg_ckpt", volagg_paths),
        ):
            value = str(cfg.get(key, "") or "")
            if value:
                bag.add(value)
        for aux_path in tuple(str(x) for x in (cfg.get("aux_pre_residual_ckpts", ()) or ())):
            if aux_path:
                aux_pre_paths.add(aux_path)

    knn_idx = _knn_2d_idx(model_root=model_root, k=need_knn_k, graph2=graph2) if need_knn_k > 0 else None

    coupling_resid2d_models: dict[str, tuple[ResidualNodeGRU, str]] = {}
    for path in sorted(coupling_paths):
        ck = _load_torch(Path(path))
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        model = ResidualNodeGRU(n_nodes=int(graph2.n_2d), cfg=mcfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        coupling_resid2d_models[path] = (model, str((ck.get("cfg", {}) or {}).get("dyn_feat", "")))

    inlet_ckpts = {path: _load_torch(Path(path)) for path in sorted(inlet_paths)}
    aux_baseline_ckpts = {path: _load_torch(Path(path)) for path in sorted(aux_baseline_paths)}
    aux_pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]] = {}
    for path in sorted(aux_pre_paths):
        ck = _load_torch(Path(path))
        cfg = ck.get("cfg", {}) or {}
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        model = ResidualNodeGRU(n_nodes=int(graph2.n_1d), cfg=mcfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        aux_pre_model_map[path] = (ck, model, cfg)

    edgeflow_ckpts = {path: _load_torch(Path(path)) for path in sorted(edgeflow_paths)}
    surfaceflow_ckpts = {path: _load_torch(Path(path)) for path in sorted(surfaceflow_paths)}
    volagg_ckpts = {path: _load_torch(Path(path)) for path in sorted(volagg_paths)}

    return StackBundle(
        baseline_ckpts=baseline_ckpts,
        resid2d=resid2d,
        resid2d_dyn_feat=resid2d_dyn_feat,
        edge_index_2d=edge_index_2d,
        edge_deg_inv_2d=edge_deg_inv_2d,
        stages=stages,
        knn_idx=knn_idx,
        coupling_resid2d_models=coupling_resid2d_models,
        inlet_ckpts=inlet_ckpts,
        aux_baseline_ckpts=aux_baseline_ckpts,
        aux_pre_model_map=aux_pre_model_map,
        edgeflow_ckpts=edgeflow_ckpts,
        surfaceflow_ckpts=surfaceflow_ckpts,
        volagg_ckpts=volagg_ckpts,
    )


def _predict_stack_event(
    *,
    stack: StackBundle,
    event_id: int,
    split: str = "train",
    graph2,
    graph_ctx: dict,
    model_root: Path,
    cache_dir: Path | None,
    warmup: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    n1 = int(graph2.n_1d)
    n2 = int(graph2.n_2d)
    ev = load_event(model_root, graph=graph2, split=str(split), event_id=event_id, cache_dir=cache_dir)
    y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
    y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
    rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
    inlet_1d_ev = ev.inlet_1d.numpy().astype(np.float32, copy=False) if ev.inlet_1d is not None else None
    edgeflow_1d_ev = None
    if stack.edgeflow_ckpts:
        edgeflow_1d_ev = load_edge_flow_1d(
            model_root=model_root,
            model_id=2,
            split=str(split),
            event_id=event_id,
            n_edges=int(graph2.edge_index_1d.shape[1] // 2),
            cache_dir=cache_dir,
        )
    surfaceflow_2d_ev = None
    if stack.surfaceflow_ckpts:
        surfaceflow_2d_ev = load_edge_flow_2d(
            model_root=model_root,
            model_id=2,
            split=str(split),
            event_id=event_id,
            n_edges=int(graph2.edge_index_2d.shape[1] // 2),
            cache_dir=cache_dir,
        )
    vol2_agg_ev = None
    if ev.volume_2d is not None:
        vol2_agg_ev = aggregate_2d_to_1d_sum(
            ev.volume_2d.numpy().astype(np.float32, copy=False),
            conn_src_1d=graph_ctx["conn_src"],
            conn_dst_2d=graph_ctx["conn_dst"],
            n_1d=n1,
        ).astype(np.float32, copy=False)
    volagg_init_ev = vol2_agg_ev

    first_cfg = stack.stages[0][2]
    y1_base, y2_base = predict_model2_from_baseline_ckpts(
        stack.baseline_ckpts,
        graph2=graph2,
        mixed_mode=str(first_cfg.get("mixed_mode", "weighted_split_ns")),
        alpha_1d=float(first_cfg.get("alpha_1d", 0.9)),
        alpha_2d=float(first_cfg.get("alpha_2d", 0.5)),
        y1_init=y1_true,
        y2_init=y2_true,
        rain_2d=rain,
        q1_init=inlet_1d_ev,
        vagg_init=volagg_init_ev,
        warmup=warmup,
    )

    y2_corr = _apply_resid2d(
        resid2d=stack.resid2d,
        node_static_2d=graph_ctx["node_static_2d"],
        y2_base=y2_base,
        rain_2d=rain,
        bed2=graph_ctx["bed2"],
        warmup=warmup,
        dyn_feat=stack.resid2d_dyn_feat,
        y1_base=y1_base,
        conn_src_1d=graph_ctx["conn_src"],
        conn_dst_2d=graph_ctx["conn_dst"],
        n_2d=n2,
        edge_index_2d=stack.edge_index_2d,
        edge_deg_inv_2d=stack.edge_deg_inv_2d,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    y2_corr_coupling: dict[str, np.ndarray] = {}
    for path, (model2d_cpl, dyn_feat_cpl) in stack.coupling_resid2d_models.items():
        y2_corr_coupling[path] = _apply_resid2d(
            resid2d=model2d_cpl,
            node_static_2d=graph_ctx["node_static_2d"],
            y2_base=y2_base,
            rain_2d=rain,
            bed2=graph_ctx["bed2"],
            warmup=warmup,
            dyn_feat=dyn_feat_cpl,
            y1_base=y1_base,
            conn_src_1d=graph_ctx["conn_src"],
            conn_dst_2d=graph_ctx["conn_dst"],
            n_2d=n2,
            edge_index_2d=stack.edge_index_2d,
            edge_deg_inv_2d=stack.edge_deg_inv_2d,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

    y1_feat = y1_base.astype(np.float32, copy=True)
    inlet_pred_final = None
    for ck, model, cfg, node_static, node_mask in stack.stages:
        use_nbr = bool(cfg.get("use_nbr_feats", False)) or bool(cfg.get("use_out_nbr_feats", False))
        pipe_dir = bool(cfg.get("nbr_pipe_dir", False))
        if use_nbr:
            nbr_src = graph_ctx["masks"]["src_dir"] if pipe_dir else graph_ctx["masks"]["src_full"]
            nbr_dst = graph_ctx["masks"]["dst_dir"] if pipe_dir else graph_ctx["masks"]["dst_full"]
        else:
            nbr_src = None
            nbr_dst = None

        coupling_path = str(cfg.get("resid2d_ckpt_coupling", "") or "")
        y2_agg, y2_knn_mean, y2_knn_max = _build_coupling_views(
            cfg_like=cfg,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            bed2=graph_ctx["bed2"],
            conn_src=graph_ctx["conn_src"],
            conn_dst=graph_ctx["conn_dst"],
            n_1d=n1,
            knn_idx=stack.knn_idx,
        )
        inlet_pred = _build_inlet_pred_views(
            cfg_like=cfg,
            inlet_ckpts=stack.inlet_ckpts,
            surfaceflow_ckpts=stack.surfaceflow_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            q1_init=inlet_1d_ev,
            volagg_init=volagg_init_ev,
            q2edge_init=surfaceflow_2d_ev,
            warmup=warmup,
            surface_slot_edges=graph_ctx["surface_slot_edges"],
            surface_slot_sign=graph_ctx["surface_slot_sign"],
            center_cell=graph_ctx["local2d_center_cell"],
            neighbor_slots=graph_ctx["local2d_neighbor_slots"],
        )
        inlet_pred_final = inlet_pred
        aux_y1_base = _build_aux_traj_views(
            cfg_like=cfg,
            aux_baseline_ckpts=stack.aux_baseline_ckpts,
            aux_pre_model_map=stack.aux_pre_model_map,
            graph2=graph2,
            y1_init=y1_true,
            y2_init=y2_true,
            rain_2d=rain,
            q1_init=inlet_1d_ev,
            vagg_init=volagg_init_ev,
            q2edge_init=surfaceflow_2d_ev,
            qedge_init=edgeflow_1d_ev,
            warmup=warmup,
            node_static_raw=graph_ctx["node_static_1d_raw"],
            node_static_aug1=graph_ctx["node_static_1d_aug1"],
            node_static_aug2=graph_ctx["node_static_1d_aug2"],
            masks=graph_ctx["masks"],
            y2_corr=y2_corr,
            y2_corr_coupling=y2_corr_coupling,
            bed2=graph_ctx["bed2"],
            conn_src=graph_ctx["conn_src"],
            conn_dst=graph_ctx["conn_dst"],
            n1=n1,
            knn_idx=stack.knn_idx,
            inlet_ckpts=stack.inlet_ckpts,
            edgeflow_ckpts=stack.edgeflow_ckpts,
            surfaceflow_ckpts=stack.surfaceflow_ckpts,
            volagg_ckpts=stack.volagg_ckpts,
            surface_slot_edges=graph_ctx["surface_slot_edges"],
            surface_slot_sign=graph_ctx["surface_slot_sign"],
            local2d_center_cell=graph_ctx["local2d_center_cell"],
            local2d_neighbor_slots=graph_ctx["local2d_neighbor_slots"],
            inv1=graph_ctx["inv1"],
            bed_agg=graph_ctx["bed_agg"],
            has_conn=graph_ctx["has_conn"],
            lap_w_dir=graph_ctx["lap_w_dir"],
            diam_max_1d=graph_ctx["diam_max_1d"],
            conn_area_1d=graph_ctx["conn_area_1d"],
            vol2_agg_ev=vol2_agg_ev,
            edgeflow_in_slots=graph_ctx["edgeflow_in_slots"],
            edgeflow_out_slots=graph_ctx["edgeflow_out_slots"],
            edge_index_1d_full=graph_ctx["edge_index_full"],
            edge_deg_inv_1d_full=graph_ctx["edge_deg_inv_full_w"],
            edge_weight_1d_full=graph_ctx["edge_weight_full"],
            edge_index_1d_dir=graph_ctx["edge_index_dir"],
            edge_deg_inv_1d_dir=graph_ctx["edge_deg_inv_dir_w"],
            edge_weight_1d_dir=graph_ctx["edge_weight_dir"],
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        if int(cfg.get("dyn_feat_version", 1) or 1) == 17:
            edgeflow_pred = _build_edgeflow_pred_slot_views(
                cfg_like=cfg,
                edgeflow_ckpts=stack.edgeflow_ckpts,
                graph2=graph2,
                y1_1d=y1_feat,
                rain_2d=rain,
                qedge_init=edgeflow_1d_ev,
                warmup=warmup,
                edge_in_slots=graph_ctx["edgeflow_in_slots"],
                edge_out_slots=graph_ctx["edgeflow_out_slots"],
            )
        else:
            edgeflow_pred = _build_edgeflow_pred_views(
                cfg_like=cfg,
                edgeflow_ckpts=stack.edgeflow_ckpts,
                graph2=graph2,
                y1_1d=y1_feat,
                rain_2d=rain,
                qedge_init=edgeflow_1d_ev,
                warmup=warmup,
            )
        volagg_pred = _build_volagg_pred_views(
            cfg_like=cfg,
            volagg_ckpts=stack.volagg_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )
        surfaceflow_pred = _build_surfaceflow_pred_views(
            cfg_like=cfg,
            surfaceflow_ckpts=stack.surfaceflow_ckpts,
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
                surfaceflow_ckpts=stack.surfaceflow_ckpts,
                graph2=graph2,
                y2_main=y2_corr,
                y2_alt=y2_corr_coupling.get(coupling_path),
                rain_2d=rain,
                q2edge_init=surfaceflow_2d_ev,
                warmup=warmup,
                surface_slot_edges=graph_ctx["surface_slot_edges"],
                surface_slot_sign=graph_ctx["surface_slot_sign"],
            )
        else:
            surfaceflow_slot_pred = None

        if int(cfg.get("dyn_feat_version", 1) or 1) == 21:
            blend_local = float(cfg.get("resid2d_coupling_blend", 0.0) or 0.0)
            y2_local = y2_corr
            y2_alt_local = y2_corr_coupling.get(coupling_path)
            if y2_alt_local is not None and blend_local > 0.0:
                y2_local = ((1.0 - blend_local) * y2_corr + blend_local * y2_alt_local).astype(np.float32, copy=False)
            local2d_node_pred = build_coupled_neighbor_node_features(
                y2_2d=y2_local,
                bed_2d=graph_ctx["bed2"],
                center_cell=graph_ctx["local2d_center_cell"],
                neighbor_slots=graph_ctx["local2d_neighbor_slots"],
            )
        else:
            local2d_node_pred = None

        if str(cfg.get("graph_mix_edge", "full")) == "dir":
            edge_index_1d = graph_ctx["edge_index_dir"]
            edge_deg_inv_1d = graph_ctx["edge_deg_inv_dir_w"] if bool(cfg.get("graph_mix_weighted", False)) else graph_ctx["edge_deg_inv_dir"]
            edge_weight_1d = graph_ctx["edge_weight_dir"] if bool(cfg.get("graph_mix_weighted", False)) else None
        else:
            edge_index_1d = graph_ctx["edge_index_full"]
            edge_deg_inv_1d = graph_ctx["edge_deg_inv_full_w"] if bool(cfg.get("graph_mix_weighted", False)) else graph_ctx["edge_deg_inv_full"]
            edge_weight_1d = graph_ctx["edge_weight_full"] if bool(cfg.get("graph_mix_weighted", False)) else None

        y1_feat = _apply_resid1d_ckpt(
            ckpt=ck,
            model=model,
            node_static=node_static,
            cfg=cfg,
            expert_group_idx=graph_ctx["expert_group_idx_topo3"] if bool(cfg.get("expert_group_topo3", False)) and int(cfg.get("expert_heads", 1) or 1) > 1 else None,
            node_mask=node_mask,
            y1_feat=y1_feat,
            y2_agg=y2_agg,
            rain_2d=rain,
            warmup=warmup,
            inv1=graph_ctx["inv1"],
            bed_agg=graph_ctx["bed_agg"],
            has_conn=graph_ctx["has_conn"],
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            lap_src=graph_ctx["masks"]["src_dir"],
            lap_dst=graph_ctx["masks"]["dst_dir"],
            lap_w=graph_ctx["lap_w_dir"],
            diam_max_1d=graph_ctx["diam_max_1d"],
            base_area_1d=graph_ctx["masks"].get("base_area", None),
            conn_area_1d=graph_ctx["conn_area_1d"],
            inlet_1d=inlet_1d_ev,
            vol2_agg=vol2_agg_ev,
            inlet_pred_1d=inlet_pred,
            aux_y1_base=aux_y1_base,
            edgeflow_node_feats=edgeflow_pred,
            surfaceflow_node_feats=surfaceflow_pred,
            surfaceflow_slot_feats=surfaceflow_slot_pred,
            local2d_node_feats=local2d_node_pred,
            volagg_pred_1d=volagg_pred,
            edge_index_1d=edge_index_1d,
            edge_deg_inv_1d=edge_deg_inv_1d,
            edge_weight_1d=edge_weight_1d,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

    rain_agg_1d = aggregate_2d_to_1d_mean(
        rain,
        conn_src_1d=graph_ctx["conn_src"],
        conn_dst_2d=graph_ctx["conn_dst"],
        n_1d=n1,
    ).astype(np.float32, copy=False)

    return y1_true, y2_true, y1_feat, y2_corr, inlet_pred_final, rain_agg_1d


def _build_example(
    *,
    event_id: int,
    graph2,
    graph_ctx: dict,
    best_stack: StackBundle,
    alt_stack: StackBundle,
    model_root: Path,
    cache_dir: Path | None,
    warmup: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    loss_source_scale: float,
    loss_ba0_scale: float,
    loss_special_scale: float,
    loss_q_scale: float,
    route_temp: float,
    warm_ctx_version: int,
    warm_seq_version: int,
    future_local2d_version: int,
    future_local1d_version: int,
    scope_mode: str,
    scope_q_thresh: float,
) -> dict:
    y1_true, y2_true, y1_best, y2_best, q_pred, rain_agg = _predict_stack_event(
        stack=best_stack,
        event_id=event_id,
        split="train",
        graph2=graph2,
        graph_ctx=graph_ctx,
        model_root=model_root,
        cache_dir=cache_dir,
        warmup=warmup,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    _y1_true_alt, _y2_true_alt, y1_alt, y2_alt, q_pred_alt, _rain_agg_alt = _predict_stack_event(
        stack=alt_stack,
        event_id=event_id,
        split="train",
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
    true_tail = y1_true[warmup:].astype(np.float32, copy=False)
    delta_best = np.zeros_like(best_tail)
    delta_best[1:] = best_tail[1:] - best_tail[:-1]
    delta_alt = np.zeros_like(alt_tail)
    delta_alt[1:] = alt_tail[1:] - alt_tail[:-1]
    gap = alt_tail - best_tail
    delta_gap = delta_alt - delta_best
    fill_best = np.clip((best_tail - graph_ctx["inv1"][None, :]) / graph_ctx["depth"][None, :], 0.0, 2.0)
    fill_alt = np.clip((alt_tail - graph_ctx["inv1"][None, :]) / graph_ctx["depth"][None, :], 0.0, 2.0)
    if q_pred is None:
        q_pred_tail = np.zeros_like(best_tail)
    else:
        q_pred_tail = q_pred[warmup:].astype(np.float32, copy=False)
    if q_pred_alt is None:
        q_pred_alt_tail = np.zeros_like(best_tail)
    else:
        q_pred_alt_tail = q_pred_alt[warmup:].astype(np.float32, copy=False)
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

    if int(warm_ctx_version) >= 1 and warmup > 0:
        warm_true = y1_true[:warmup].astype(np.float32, copy=False)
        warm_best = y1_best[:warmup].astype(np.float32, copy=False)
        warm_alt = y1_alt[:warmup].astype(np.float32, copy=False)
        warm_err_best = np.abs(warm_best - warm_true).astype(np.float32, copy=False)
        warm_err_alt = np.abs(warm_alt - warm_true).astype(np.float32, copy=False)
        warm_true_prev = warm_true[-2] if warmup >= 2 else warm_true[-1]
        warm_y2_depth = (y2_true[:warmup] - graph_ctx["bed2"][None, :]).astype(np.float32, copy=False)
        warm_y2_depth_agg = aggregate_2d_to_1d_mean(
            warm_y2_depth,
            conn_src_1d=graph_ctx["conn_src"],
            conn_dst_2d=graph_ctx["conn_dst"],
            n_1d=int(graph_ctx["n1"]),
        ).astype(np.float32, copy=False)
        warm_rain = rain_agg[:warmup].astype(np.float32, copy=False)
        ctx = np.stack(
            [
                warm_true[-1],
                (warm_true[-1] - warm_true_prev).astype(np.float32, copy=False),
                warm_true.mean(axis=0).astype(np.float32, copy=False),
                np.max(warm_true, axis=0).astype(np.float32, copy=False),
                warm_err_best.mean(axis=0).astype(np.float32, copy=False),
                warm_err_alt.mean(axis=0).astype(np.float32, copy=False),
                (warm_err_alt.mean(axis=0) - warm_err_best.mean(axis=0)).astype(np.float32, copy=False),
                warm_err_best[-1],
                warm_err_alt[-1],
                (warm_err_alt[-1] - warm_err_best[-1]).astype(np.float32, copy=False),
                (warm_best[-1] - warm_true[-1]).astype(np.float32, copy=False),
                (warm_alt[-1] - warm_true[-1]).astype(np.float32, copy=False),
                warm_y2_depth_agg.mean(axis=0).astype(np.float32, copy=False),
                warm_y2_depth_agg[-1],
                (warm_y2_depth_agg[-1] - (warm_y2_depth_agg[-2] if warmup >= 2 else warm_y2_depth_agg[-1])).astype(np.float32, copy=False),
                warm_rain.mean(axis=0).astype(np.float32, copy=False),
                warm_rain[-1],
                (warm_rain[-1] - (warm_rain[-2] if warmup >= 2 else warm_rain[-1])).astype(np.float32, copy=False),
                np.sum(warm_rain, axis=0).astype(np.float32, copy=False),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        dyn = np.concatenate(
            [
                dyn,
                np.broadcast_to(ctx[None, :, :], (int(dyn.shape[0]), int(dyn.shape[1]), int(ctx.shape[-1]))),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    if int(warm_ctx_version) >= 2 and warmup > 0:
        ev = load_event(model_root, graph=graph2, split="train", event_id=event_id, cache_dir=cache_dir)
        ctx_parts: list[np.ndarray] = []

        def _ctx_cols(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 1:
                return arr[:, None]
            return arr

        if ev.inlet_1d is not None:
            q_true = ev.inlet_1d.numpy().astype(np.float32, copy=False)[:warmup]
            q_true_last = q_true[-1]
            q_true_prev = q_true[-2] if warmup >= 2 else q_true_last
            q_true_pos = np.maximum(q_true, 0.0).astype(np.float32, copy=False)
            ctx_parts.extend(
                [
                    _ctx_cols(q_true.mean(axis=0).astype(np.float32, copy=False)),
                    _ctx_cols(q_true_last.astype(np.float32, copy=False)),
                    _ctx_cols((q_true_last - q_true_prev).astype(np.float32, copy=False)),
                    _ctx_cols(q_true_pos.mean(axis=0).astype(np.float32, copy=False)),
                    _ctx_cols(q_true_pos[-1].astype(np.float32, copy=False)),
                ]
            )
        if ev.volume_2d is not None:
            vagg = aggregate_2d_to_1d_sum(
                ev.volume_2d.numpy().astype(np.float32, copy=False)[:warmup],
                conn_src_1d=graph_ctx["conn_src"],
                conn_dst_2d=graph_ctx["conn_dst"],
                n_1d=int(graph_ctx["n1"]),
            ).astype(np.float32, copy=False)
            vagg_last = vagg[-1]
            vagg_prev = vagg[-2] if warmup >= 2 else vagg_last
            ctx_parts.extend(
                [
                    _ctx_cols(vagg.mean(axis=0).astype(np.float32, copy=False)),
                    _ctx_cols(vagg_last.astype(np.float32, copy=False)),
                    _ctx_cols((vagg_last - vagg_prev).astype(np.float32, copy=False)),
                ]
            )
        q2edge_true = load_edge_flow_2d(
            model_root=model_root,
            model_id=2,
            split="train",
            event_id=event_id,
            n_edges=int(graph2.edge_index_2d.shape[1] // 2),
            cache_dir=cache_dir,
        )[:warmup]
        surface_slot_true = build_coupled_surface_slot_features(
            q_edge=q2edge_true,
            edge_slots=graph_ctx["surface_slot_edges"],
            edge_sign=graph_ctx["surface_slot_sign"],
        ).astype(np.float32, copy=False)
        local2d_true = build_coupled_neighbor_node_features(
            y2_2d=y2_true[:warmup].astype(np.float32, copy=False),
            bed_2d=graph_ctx["bed2"],
            center_cell=graph_ctx["local2d_center_cell"],
            neighbor_slots=graph_ctx["local2d_neighbor_slots"],
        ).astype(np.float32, copy=False)
        ctx_parts.extend(
            [
                _ctx_cols(surface_slot_true.mean(axis=0).astype(np.float32, copy=False)),
                _ctx_cols(surface_slot_true[-1].astype(np.float32, copy=False)),
                _ctx_cols(local2d_true.mean(axis=0).astype(np.float32, copy=False)),
                _ctx_cols(local2d_true[-1].astype(np.float32, copy=False)),
            ]
        )
        if ctx_parts:
            ctx2 = np.concatenate(ctx_parts, axis=-1).astype(np.float32, copy=False)
            dyn = np.concatenate(
                [
                    dyn,
                    np.broadcast_to(ctx2[None, :, :], (int(dyn.shape[0]), int(dyn.shape[1]), int(ctx2.shape[-1]))),
                ],
                axis=-1,
            ).astype(np.float32, copy=False)

    if int(warm_ctx_version) >= 3 and warmup > 0:
        q1edge_true = load_edge_flow_1d(
            model_root=model_root,
            model_id=2,
            split="train",
            event_id=event_id,
            n_edges=int(graph2.edge_index_1d.shape[1] // 2),
            cache_dir=cache_dir,
        )[:warmup]
        edge_slot_true = build_edge_flow_slot_features(
            q_edge=q1edge_true,
            edge_in_slots=graph_ctx["edgeflow_in_slots"],
            edge_out_slots=graph_ctx["edgeflow_out_slots"],
        ).astype(np.float32, copy=False)
        ctx3 = np.concatenate(
            [
                edge_slot_true.mean(axis=0).astype(np.float32, copy=False),
                edge_slot_true[-1].astype(np.float32, copy=False),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        dyn = np.concatenate(
            [
                dyn,
                np.broadcast_to(ctx3[None, :, :], (int(dyn.shape[0]), int(dyn.shape[1]), int(ctx3.shape[-1]))),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    warm_seq = None
    if int(warm_seq_version) >= 1 and warmup > 0:
        ev = load_event(model_root, graph=graph2, split="train", event_id=event_id, cache_dir=cache_dir)
        depth1_true = (y1_true[:warmup] - graph_ctx["inv1"][None, :]).astype(np.float32, copy=False)
        depth2_true = aggregate_2d_to_1d_mean(
            (y2_true[:warmup] - graph_ctx["bed2"][None, :]).astype(np.float32, copy=False),
            conn_src_1d=graph_ctx["conn_src"],
            conn_dst_2d=graph_ctx["conn_dst"],
            n_1d=int(graph_ctx["n1"]),
        ).astype(np.float32, copy=False)
        rain_true = rain_agg[:warmup].astype(np.float32, copy=False)
        if ev.inlet_1d is None:
            inlet_true = np.zeros_like(depth1_true, dtype=np.float32)
        else:
            inlet_true = ev.inlet_1d.numpy().astype(np.float32, copy=False)[:warmup]
        if ev.volume_2d is None:
            volagg_true = np.zeros_like(depth1_true, dtype=np.float32)
        else:
            volagg_true = aggregate_2d_to_1d_sum(
                ev.volume_2d.numpy().astype(np.float32, copy=False)[:warmup],
                conn_src_1d=graph_ctx["conn_src"],
                conn_dst_2d=graph_ctx["conn_dst"],
                n_1d=int(graph_ctx["n1"]),
            ).astype(np.float32, copy=False)
        volagg_true = (np.sign(volagg_true) * np.log1p(np.abs(volagg_true))).astype(np.float32, copy=False)

        q2edge_true = load_edge_flow_2d(
            model_root=model_root,
            model_id=2,
            split="train",
            event_id=event_id,
            n_edges=int(graph2.edge_index_2d.shape[1] // 2),
            cache_dir=cache_dir,
        )[:warmup]
        surface_slot_true = build_coupled_surface_slot_features(
            q_edge=q2edge_true,
            edge_slots=graph_ctx["surface_slot_edges"],
            edge_sign=graph_ctx["surface_slot_sign"],
        ).astype(np.float32, copy=False)
        surface_series = surface_slot_true[:, :, 0::3]
        surface_pos = np.sum(np.maximum(surface_series, 0.0), axis=-1).astype(np.float32, copy=False)
        surface_neg = np.sum(np.maximum(-surface_series, 0.0), axis=-1).astype(np.float32, copy=False)
        surface_abs = np.sum(np.abs(surface_series), axis=-1).astype(np.float32, copy=False)

        q1edge_true = load_edge_flow_1d(
            model_root=model_root,
            model_id=2,
            split="train",
            event_id=event_id,
            n_edges=int(graph2.edge_index_1d.shape[1] // 2),
            cache_dir=cache_dir,
        )[:warmup]
        edge_slot_true = build_edge_flow_slot_features(
            q_edge=q1edge_true,
            edge_in_slots=graph_ctx["edgeflow_in_slots"],
            edge_out_slots=graph_ctx["edgeflow_out_slots"],
        ).astype(np.float32, copy=False)
        edge_series = edge_slot_true[:, :, 0::3]
        edge_pos = np.sum(np.maximum(edge_series, 0.0), axis=-1).astype(np.float32, copy=False)
        edge_neg = np.sum(np.maximum(-edge_series, 0.0), axis=-1).astype(np.float32, copy=False)
        edge_abs = np.sum(np.abs(edge_series), axis=-1).astype(np.float32, copy=False)

        warm_seq = np.stack(
            [
                depth1_true,
                depth2_true,
                inlet_true,
                volagg_true,
                rain_true,
                surface_pos,
                surface_neg,
                surface_abs,
                edge_pos,
                edge_neg,
                edge_abs,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        warm_seq = np.transpose(warm_seq, (1, 0, 2)).astype(np.float32, copy=False)

    loss_w = np.ones_like(best_tail, dtype=np.float32)
    if float(loss_source_scale) > 0.0:
        loss_w += float(loss_source_scale) * graph_ctx["static_gate"][None, :, 0]
    if float(loss_ba0_scale) > 0.0:
        loss_w += float(loss_ba0_scale) * graph_ctx["static_gate"][None, :, 1]
    if float(loss_special_scale) > 0.0:
        q_scale = max(float(loss_q_scale), 1e-6)
        q_pos = np.clip(q_pred_tail / q_scale, 0.0, 2.0)
        special = ((graph_ctx["static_gate"][None, :, 0] > 0.5) | (graph_ctx["static_gate"][None, :, 1] > 0.5)).astype(np.float32, copy=False)
        loss_w += float(loss_special_scale) * special * q_pos

    route_scale = max(float(route_temp), 1e-6)
    route_margin = np.abs(best_tail - true_tail) - np.abs(alt_tail - true_tail)
    route_logits = np.clip(route_margin / route_scale, -40.0, 40.0)
    route_target = 1.0 / (1.0 + np.exp(-route_logits))
    route_w = np.clip(np.abs(route_margin) / route_scale, 0.0, 2.0).astype(np.float32, copy=False)

    return {
        "event_id": int(event_id),
        "x_dyn": torch.from_numpy(dyn),
        "best_tail": torch.from_numpy(best_tail),
        "alt_tail": torch.from_numpy(alt_tail),
        "y_true": torch.from_numpy(true_tail),
        "loss_w": torch.from_numpy(loss_w),
        "route_target": torch.from_numpy(route_target.astype(np.float32, copy=False)),
        "route_w": torch.from_numpy(route_w),
        "scope": torch.from_numpy(scope),
        "warm_seq": None if warm_seq is None else torch.from_numpy(warm_seq),
    }


def main() -> None:
    args = _parse_args()
    seed_everything(int(args.seed))
    device = torch.device("cuda")
    amp_enabled = bool(args.amp_bf16)
    amp_dtype = torch.bfloat16
    model_root = Path(args.model_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")

    split_ckpt_path = Path(args.split_from) if str(args.split_from) else Path(args.best_baseline_ckpt[0])
    split_ckpt = _load_torch(split_ckpt_path)
    split = split_ckpt.get("split", None)
    if split is None:
        raise ValueError("split checkpoint missing split")
    warmup = int(split_ckpt["cfg"]["warmup"])

    graph2 = load_graph(model_root, model_id=2, split_for_static="train")
    graph_ctx = _build_graph_context(model_root=model_root, graph2=graph2, device=device)

    best_stack = _build_stack_bundle(
        baseline_paths=list(args.best_baseline_ckpt),
        resid2d_path=str(args.best_resid2d_ckpt),
        resid1d_paths=list(args.best_resid1d_ckpt),
        graph2=graph2,
        graph_ctx=graph_ctx,
        device=device,
        model_root=model_root,
    )
    alt_stack = _build_stack_bundle(
        baseline_paths=list(args.alt_baseline_ckpt),
        resid2d_path=str(args.alt_resid2d_ckpt),
        resid1d_paths=list(args.alt_resid1d_ckpt),
        graph2=graph2,
        graph_ctx=graph_ctx,
        device=device,
        model_root=model_root,
    )

    train_ids = list(split["model_2"]["train"])
    val_ids = list(split["model_2"]["val"])
    if not train_ids:
        raise ValueError("gate training requires at least one train event")
    full_train_no_val = len(val_ids) == 0

    t0 = time.time()
    train_cache = {
        event_id: _build_example(
            event_id=event_id,
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
            loss_source_scale=float(args.loss_source_scale),
            loss_ba0_scale=float(args.loss_ba0_scale),
            loss_special_scale=float(args.loss_special_scale),
            loss_q_scale=float(args.loss_q_scale),
            route_temp=float(args.route_temp),
            warm_ctx_version=int(args.warm_ctx_version),
            warm_seq_version=int(args.warm_seq_version),
            future_local2d_version=int(args.future_local2d_version),
            future_local1d_version=int(args.future_local1d_version),
            scope_mode=str(args.scope_mode),
            scope_q_thresh=float(args.scope_q_thresh),
        )
        for event_id in tqdm(train_ids, desc="prep train (blend-gate)", leave=False)
    }
    val_cache = {
        event_id: _build_example(
            event_id=event_id,
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
            loss_source_scale=float(args.loss_source_scale),
            loss_ba0_scale=float(args.loss_ba0_scale),
            loss_special_scale=float(args.loss_special_scale),
            loss_q_scale=float(args.loss_q_scale),
            route_temp=float(args.route_temp),
            warm_ctx_version=int(args.warm_ctx_version),
            warm_seq_version=int(args.warm_seq_version),
            future_local2d_version=int(args.future_local2d_version),
            future_local1d_version=int(args.future_local1d_version),
            scope_mode=str(args.scope_mode),
            scope_q_thresh=float(args.scope_q_thresh),
        )
        for event_id in tqdm(val_ids, desc="prep val (blend-gate)", leave=False)
    }
    print(f"prep: cached {len(train_cache)} train + {len(val_cache)} val events in {time.time()-t0:.1f}s")

    static_gate = torch.from_numpy(graph_ctx["static_gate"]).to(device)
    group_idx = torch.zeros((int(static_gate.shape[0]),), dtype=torch.long, device=device)
    source_mask = static_gate[:, 0] > 0.5
    ba0_mask = static_gate[:, 1] > 0.5
    group_idx[source_mask] = 1
    group_idx[ba0_mask] = 2

    for cache in (train_cache, val_cache):
        for ex in cache.values():
            ex["x_gpu"] = ex["x_dyn"].to(device)
            ex["best_gpu"] = ex["best_tail"].to(device)
            ex["alt_gpu"] = ex["alt_tail"].to(device)
            ex["true_gpu"] = ex["y_true"].to(device)
            ex["loss_w_gpu"] = ex["loss_w"].to(device)
            ex["route_target_gpu"] = ex["route_target"].to(device)
            ex["route_w_gpu"] = ex["route_w"].to(device)
            ex["scope_gpu"] = ex["scope"].to(device)
            ex["warm_seq_gpu"] = None if ex["warm_seq"] is None else ex["warm_seq"].to(device)

    dyn_dim = int(next(iter(train_cache.values()))["x_dyn"].shape[-1]) if train_cache else int(next(iter(val_cache.values()))["x_dyn"].shape[-1])
    warm_seq_dim = 0
    if train_cache:
        warm_seq = next(iter(train_cache.values()))["warm_seq"]
    else:
        warm_seq = next(iter(val_cache.values()))["warm_seq"]
    if warm_seq is not None:
        warm_seq_dim = int(warm_seq.shape[-1])

    if str(args.arch) == "warm_mlp":
        if warm_seq_dim <= 0:
            raise ValueError("arch=warm_mlp requires warm_seq_version > 0")
        model = BlendGateWarmMLP(
            dyn_dim=dyn_dim,
            static_dim=int(static_gate.shape[1]),
            warm_seq_dim=warm_seq_dim,
            warmup=warmup,
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            correction_ft=float(args.correction_ft),
        ).to(device)
    elif str(args.arch) == "gru":
        model = BlendGateGRU(
            dyn_dim=dyn_dim,
            static_dim=int(static_gate.shape[1]),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            correction_ft=float(args.correction_ft),
        ).to(device)
    elif str(args.arch) == "group_mlp":
        model = BlendGateGroupedMLP(
            dyn_dim=dyn_dim,
            static_dim=int(static_gate.shape[1]),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            correction_ft=float(args.correction_ft),
            n_groups=3,
        ).to(device)
    elif str(args.arch) == "linear":
        model = BlendGateLinear(
            dyn_dim=dyn_dim,
            static_dim=int(static_gate.shape[1]),
            correction_ft=float(args.correction_ft),
        ).to(device)
    else:
        model = BlendGateMLP(
            dyn_dim=dyn_dim,
            static_dim=int(static_gate.shape[1]),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            correction_ft=float(args.correction_ft),
        ).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    std_1d = float(STD_DEV_DICT[(2, 1)])
    eps = 1e-12

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    def save_ckpt(*, state_dict: dict[str, torch.Tensor], best_val_m2: float, best_epoch_idx: int) -> None:
        payload = {
            "kind": "blend_gate_m2_1d",
            "best_stack": {
                "baseline_ckpts": list(args.best_baseline_ckpt),
                "resid2d_ckpt": str(args.best_resid2d_ckpt),
                "resid1d_ckpt": list(args.best_resid1d_ckpt),
            },
            "alt_stack": {
                "baseline_ckpts": list(args.alt_baseline_ckpt),
                "resid2d_ckpt": str(args.alt_resid2d_ckpt),
                "resid1d_ckpt": list(args.alt_resid1d_ckpt),
            },
            "split_from": str(split_ckpt_path),
            "seed": int(args.seed),
            "warmup": int(warmup),
            "hidden_dim": int(args.hidden_dim),
            "arch": str(args.arch),
            "dropout": float(args.dropout),
            "correction_ft": float(args.correction_ft),
            "correction_penalty": float(args.correction_penalty),
            "route_loss_scale": float(args.route_loss_scale),
            "route_temp": float(args.route_temp),
            "warm_ctx_version": int(args.warm_ctx_version),
            "warm_seq_version": int(args.warm_seq_version),
            "future_local2d_version": int(args.future_local2d_version),
            "future_local1d_version": int(args.future_local1d_version),
            "scope_mode": str(args.scope_mode),
            "scope_q_thresh": float(args.scope_q_thresh),
            "full_train_no_val": bool(full_train_no_val),
            "train_event_count": int(len(train_ids)),
            "val_event_count": int(len(val_ids)),
            "best_val_m2": float(best_val_m2),
            "best_epoch": int(best_epoch_idx),
            "state_dict": state_dict,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        meta_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2) + "\n")

    rng = np.random.default_rng(int(args.seed))
    for epoch in range(int(args.epochs)):
        model.train()
        train_order = list(train_cache.keys())
        rng.shuffle(train_order)
        losses = []
        t_epoch = time.time()
        for event_id in train_order:
            ex = train_cache[event_id]
            x_dyn = ex["x_gpu"]
            best_tail = ex["best_gpu"]
            alt_tail = ex["alt_gpu"]
            y_true = ex["true_gpu"]
            loss_w = ex["loss_w_gpu"]
            route_target = ex["route_target_gpu"]
            route_w = ex["route_w_gpu"]
            scope = ex["scope_gpu"]
            warm_seq = ex["warm_seq_gpu"]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                if str(args.arch) == "warm_mlp":
                    gate, corr = model(x_dyn, static_gate, warm_seq)
                elif str(args.arch) == "group_mlp":
                    gate, corr = model(x_dyn, static_gate, group_idx)
                else:
                    gate, corr = model(x_dyn, static_gate)
            y_hat = best_tail + scope.float() * (gate.float() * (alt_tail - best_tail) + corr.float())
            err = (y_hat - y_true) / std_1d
            mse_node = torch.sum((err**2) * loss_w, dim=0) / torch.clamp(torch.sum(loss_w, dim=0), min=eps)
            rmse_node = torch.sqrt(mse_node + eps)
            loss = rmse_node.mean()
            if float(args.correction_penalty) > 0.0 and float(args.correction_ft) > 0.0:
                corr_pen = torch.mean((corr.float() / max(float(args.correction_ft), 1e-6)) ** 2)
                loss = loss + float(args.correction_penalty) * corr_pen
            if float(args.route_loss_scale) > 0.0:
                gate_clamped = gate.float().clamp(min=1e-5, max=1.0 - 1e-5)
                route_weight = route_w * loss_w * scope.float()
                route_bce = F.binary_cross_entropy(gate_clamped, route_target.float(), reduction="none")
                route_den = torch.clamp(torch.sum(route_weight), min=eps)
                route_loss = torch.sum(route_bce * route_weight) / route_den
                loss = loss + float(args.route_loss_scale) * route_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        if val_cache:
            model.eval()
            val_1d_scores = []
            with torch.no_grad():
                for ex in val_cache.values():
                    if str(args.arch) == "warm_mlp":
                        gate, corr = model(ex["x_gpu"], static_gate, ex["warm_seq_gpu"])
                    elif str(args.arch) == "group_mlp":
                        gate, corr = model(ex["x_gpu"], static_gate, group_idx)
                    else:
                        gate, corr = model(ex["x_gpu"], static_gate)
                    y_hat = ex["best_gpu"] + ex["scope_gpu"].float() * (gate.float() * (ex["alt_gpu"] - ex["best_gpu"]) + corr.float())
                    err = (y_hat - ex["true_gpu"]) / std_1d
                    val_1d_scores.append(float(torch.sqrt(torch.mean(err**2, dim=0) + eps).mean().cpu().item()))
            val_1d = float(np.mean(val_1d_scores)) if val_1d_scores else float("nan")
            val_2d = 0.011078185
            val_m2 = 0.5 * (val_1d + val_2d)
            overall = 0.5 * (float(args.m1_mean) + val_m2)

            if val_m2 < best_val:
                best_val = val_m2
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_ckpt(state_dict=best_state, best_val_m2=best_val, best_epoch_idx=best_epoch)

            print(
                f"epoch {epoch+1:03d}/{int(args.epochs)} "
                f"train={train_loss:.6f} "
                f"val_m2={val_m2:.6f} "
                f"val_1d={val_1d:.6f} "
                f"val_2d={val_2d:.6f} "
                f"overall={overall:.6f} "
                f"dt={time.time()-t_epoch:.1f}s"
            )
        else:
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_ckpt(state_dict=best_state, best_val_m2=float("nan"), best_epoch_idx=best_epoch)
            print(
                f"epoch {epoch+1:03d}/{int(args.epochs)} "
                f"train={train_loss:.6f} "
                f"full_train_no_val=1 "
                f"dt={time.time()-t_epoch:.1f}s"
            )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    if val_cache:
        print(f"best_val_m2={best_val:.9f} best_epoch={best_epoch} overall={(float(args.m1_mean) + best_val) / 2.0:.9f}")
    else:
        print(f"saved_full_train_gate_epoch={best_epoch}")


if __name__ == "__main__":
    main()
