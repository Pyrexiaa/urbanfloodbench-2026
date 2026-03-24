# #####
# residual_train.py
# #####

# urbanflood/residual_train.py
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from urbanflood.baseline import (
    aggregate_2d_to_1d_mean,
    aggregate_2d_to_1d_sum,
    build_coupled_1d_exo,
    build_storage_augmented_exo,
    connected_area_1d,
    predict_model2_from_baseline_ckpts,
    rollout_regime_arkx_exo,
)
from urbanflood.data import load_event, load_graph
from urbanflood.edgeflow_aux import (
    aggregate_edge_flows_to_nodes,
    build_edge_flow_slot_features,
    build_edge_slot_index,
    load_edge_flow_1d,
    predict_edge_flow_from_ckpt,
)
from urbanflood.inlet_aux import predict_inlet_from_ckpt
from urbanflood.metric import STD_DEV_DICT
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import (
    augment_dyn_features_2d_v2_nbrmean,
    build_dyn_features_1d_v1,
    build_dyn_features_1d_v3,
    build_dyn_features_1d_v8,
    build_dyn_features_1d_v9,
    build_dyn_features_1d_v10,
    build_dyn_features_1d_v11,
    build_dyn_features_1d_v12,
    build_dyn_features_1d_v13,
    build_dyn_features_1d_v14,
    build_dyn_features_1d_v15,
    build_dyn_features_1d_v16,
    build_dyn_features_1d_v17,
    build_dyn_features_1d_v18,
    build_dyn_features_1d_v19,
    build_dyn_features_1d_v20,
    build_dyn_features_1d_v21,
    build_dyn_features_1d_v22,
    build_warm_context_1d_v1,
    build_warm_context_1d_v2,
    build_warm_sequence_1d_v1,
    build_dyn_features_2d,
    build_static_features_m2_1d,
    build_static_features_m2_1d_v2,
)
from urbanflood.surfaceflow_aux import (
    build_coupled_neighbor_node_features,
    build_coupled_neighbor_node_index,
    build_coupled_surface_slot_features,
    build_coupled_surface_slot_index,
    build_surface_flow_1d_features,
    load_edge_flow_2d,
    load_surface_edge_weight,
    predict_surface_flow_from_ckpt,
)
from urbanflood.volagg_aux import predict_volagg_from_ckpt
from urbanflood.utils import seed_everything


@dataclass(frozen=True)
class Residual1DTrainCfg:
    baseline_ckpts: tuple[str, ...]
    mixed_mode: str = "weighted_split_ns"
    alpha_1d: float = 0.9
    alpha_2d: float = 0.5
    resid2d_ckpt: str = ""
    resid2d_ckpt_coupling: str = ""
    resid2d_coupling_blend: float = 0.0
    resid2d_coupling_blend_knn: float = 0.0
    inlet_ckpt: str = ""
    aux_baseline_ckpt: str = ""
    aux_pre_residual_ckpts: tuple[str, ...] = ()
    edgeflow_ckpt: str = ""
    surfaceflow_ckpt: str = ""
    volagg_ckpt: str = ""
    pre_residual_ckpts: tuple[str, ...] = ()

    model_root: str = "Models"
    cache_dir: str = ".cache/urbanflood"
    out_path: str = "runs/resid_m2_1d.pt"

    seed: int = 42
    stage: int = 1
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    node_emb_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    nodewise_head: bool = False
    expert_heads: int = 1
    expert_gate_hidden: int = 0
    expert_gate_dropout: float = 0.0
    expert_group_topo3: bool = False
    graph_mix_k: int = 0
    graph_mix_post_k: int = 0
    graph_mix_dropout: float = 0.0
    graph_mix_edge: str = "full"  # "full" (undirected) or "dir" (pipe-direction only)
    graph_mix_weighted: bool = False
    ema_decay: float = 0.0
    clamp: float = 1.0
    clamp_mode: str = "hard"
    zero_init_out: bool = False

    cache_on_gpu: bool = True
    amp_bf16: bool = True
    max_events: int = 0

    # Feature + mask config (kept compatible with older ckpts).
    dyn_feat_version: int = 1
    warm_ctx_version: int = 0
    warm_seq_version: int = 0
    y2_knn_k: int = 0
    rain_lags: int = 0
    use_nbr_feats: bool = True
    use_out_nbr_feats: bool = False
    nbr_pipe_dir: bool = False
    use_static_aug: bool = True
    static_aug_version: int = 1

    mask_base_area_zero: bool = False
    mask_pipe_source: bool = False
    mask_pipe_indeg_ge: int = 0
    mask_node_list: tuple[int, ...] = ()
    mask_mode: str = "or"

    loss_node_weight_source: float = 0.0
    loss_node_weight_base_area_zero: float = 0.0
    loss_node_weight_indeg_ge: int = 0
    loss_node_weight_indeg_scale: float = 0.0
    loss_time_weight_fill: float = 0.0
    loss_time_fill_clip: float = 1.0
    loss_time_weight_inlet_pos: float = 0.0
    loss_time_inlet_pos_scale: float = 0.25


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument(
        "--mixed-mode",
        type=str,
        default="weighted_split_ns",
        choices=["single", "weighted_split_ns"],
    )
    p.add_argument("--alpha-1d", type=float, default=0.9)
    p.add_argument("--alpha-2d", type=float, default=0.5)
    p.add_argument("--resid2d-ckpt", type=str, required=True, help="Model2 2D residual ckpt (used for 1D coupling feats).")
    p.add_argument(
        "--resid2d-coupling-ckpt",
        type=str,
        default="",
        help="Optional secondary Model2 2D residual ckpt used only to blend 1D coupling features.",
    )
    p.add_argument(
        "--resid2d-coupling-blend",
        type=float,
        default=0.0,
        help="Blend weight in [0,1] for the secondary 2D residual when building 1D aggregate coupling features.",
    )
    p.add_argument(
        "--resid2d-coupling-blend-knn",
        type=float,
        default=0.0,
        help="Optional separate blend weight in [0,1] for KNN-derived 2D coupling features.",
    )
    p.add_argument(
        "--inlet-ckpt",
        type=str,
        default="",
        help="Optional split baseline ckpt with an inlet-flow model used to build Model2 1D inlet exogenous features.",
    )
    p.add_argument(
        "--aux-baseline-ckpt",
        type=str,
        default="",
        help="Optional single baseline ckpt used to build an alternative Model2 1D baseline trajectory feature.",
    )
    p.add_argument(
        "--aux-pre-residual-ckpt",
        type=str,
        nargs="*",
        default=[],
        help="Optional 1D residual stack applied on top of --aux-baseline-ckpt to build an alternative trajectory feature.",
    )
    p.add_argument(
        "--edgeflow-ckpt",
        type=str,
        default="",
        help="Optional split edge-flow aux ckpt used to build Model2 1D pipe-flow aggregate features.",
    )
    p.add_argument(
        "--surfaceflow-ckpt",
        type=str,
        default="",
        help="Optional split surface-flow aux ckpt used to build Model2 1D local surface-transport features.",
    )
    p.add_argument(
        "--volagg-ckpt",
        type=str,
        default="",
        help="Optional split coupled-volume aux ckpt used to build Model2 1D storage aggregate features.",
    )
    p.add_argument("--pre-residual-ckpt", type=str, nargs="*", default=[], help="Optional pre-1D residual ckpts (stack).")

    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0, help="0 uses stage default.")
    p.add_argument("--lr", type=float, default=0.0, help="0 uses stage default.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--node-emb-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dyn-feat-version", type=int, default=0, help="0 uses the stage default.")
    p.add_argument("--warm-ctx-version", type=int, default=0, help="0 disables; >0 enables warmup hidden-state context.")
    p.add_argument("--warm-seq-version", type=int, default=0, help="0 disables; >0 enables learned warmup sequence encoding.")
    p.add_argument("--y2-knn-k", type=int, default=-1, help="-1 uses the stage default.")
    p.add_argument("--rain-lags", type=int, default=-1, help="-1 uses the stage default.")
    p.add_argument("--static-aug-version", type=int, default=0, help="0 uses the stage default.")
    p.add_argument("--use-out-nbr-feats", dest="use_out_nbr_feats", action="store_true")
    p.add_argument("--no-use-out-nbr-feats", dest="use_out_nbr_feats", action="store_false")
    p.set_defaults(use_out_nbr_feats=None)
    p.add_argument("--nbr-pipe-dir", dest="nbr_pipe_dir", action="store_true")
    p.add_argument("--no-nbr-pipe-dir", dest="nbr_pipe_dir", action="store_false")
    p.set_defaults(nbr_pipe_dir=None)
    p.add_argument("--bidirectional", dest="bidirectional", action="store_true")
    p.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    p.set_defaults(bidirectional=False)
    p.add_argument(
        "--nodewise-head",
        dest="nodewise_head",
        action="store_true",
        help="Enable per-node linear readout head (extra capacity; can help small graphs; may overfit).",
    )
    p.add_argument("--no-nodewise-head", dest="nodewise_head", action="store_false")
    p.set_defaults(nodewise_head=False)
    p.add_argument("--expert-heads", type=int, default=1, help="Number of mixture readout experts for the residual head.")
    p.add_argument("--expert-gate-hidden", type=int, default=0, help="Hidden size for the expert gate MLP; 0 uses a single linear gate.")
    p.add_argument("--expert-gate-dropout", type=float, default=0.0, help="Dropout inside the expert gate MLP.")
    p.add_argument(
        "--expert-group-topo3",
        dest="expert_group_topo3",
        action="store_true",
        help="Use deterministic 3-way topology routing for expert heads: default / source-or-base-area-zero / indeg>=2 non-source.",
    )
    p.add_argument("--no-expert-group-topo3", dest="expert_group_topo3", action="store_false")
    p.set_defaults(expert_group_topo3=False)
    p.add_argument(
        "--clamp",
        type=float,
        default=None,
        help="Optional override for residual output clamp (feet). 0 disables clamping.",
    )
    p.add_argument(
        "--clamp-mode",
        type=str,
        default=None,
        choices=["hard", "tanh"],
        help="Optional override for clamp mode (defaults to stage preset).",
    )
    p.add_argument("--zero-init-out", dest="zero_init_out", action="store_true", help="Initialize the residual readout to zero so the model starts from the incoming base trajectory.")
    p.add_argument("--no-zero-init-out", dest="zero_init_out", action="store_false")
    p.set_defaults(zero_init_out=False)
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional EMA decay in (0, 1). 0 disables EMA. When enabled, validation is reported using EMA weights.",
    )
    p.add_argument("--graph-mix-k", type=int, default=0, help="Optional learned 1D graph mixing steps (0 disables).")
    p.add_argument(
        "--graph-mix-post-k",
        type=int,
        default=0,
        help="Optional learned 1D graph mixing steps after the GRU (0 disables).",
    )
    p.add_argument("--graph-mix-dropout", type=float, default=0.0, help="Dropout for graph mixing blocks.")
    p.add_argument(
        "--graph-mix-edge",
        type=str,
        default="full",
        choices=["full", "dir"],
        help='Which 1D edge_index to use for graph mixing: "full" uses undirected (both directions), "dir" uses pipe-direction only.',
    )
    p.add_argument(
        "--graph-mix-weighted",
        dest="graph_mix_weighted",
        action="store_true",
        help="Use a weighted neighbor mean in graph mixing based on pipe conductance proxy (diam^2/(rough*len)).",
    )
    p.add_argument("--no-graph-mix-weighted", dest="graph_mix_weighted", action="store_false")
    p.set_defaults(graph_mix_weighted=False)
    p.add_argument("--cache-on-gpu", dest="cache_on_gpu", action="store_true")
    p.add_argument("--no-cache-on-gpu", dest="cache_on_gpu", action="store_false")
    p.set_defaults(cache_on_gpu=True)
    p.add_argument("--amp-bf16", dest="amp_bf16", action="store_true")
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false")
    p.set_defaults(amp_bf16=True)
    p.add_argument("--max-events", type=int, default=0, help="Optional limit on number of events per split (smoke).")
    p.add_argument("--mask-base-area-zero", dest="mask_base_area_zero", action="store_true")
    p.add_argument("--no-mask-base-area-zero", dest="mask_base_area_zero", action="store_false")
    p.set_defaults(mask_base_area_zero=None)
    p.add_argument("--mask-pipe-source", dest="mask_pipe_source", action="store_true")
    p.add_argument("--no-mask-pipe-source", dest="mask_pipe_source", action="store_false")
    p.set_defaults(mask_pipe_source=None)
    p.add_argument("--mask-pipe-indeg-ge", type=int, default=-1)
    p.add_argument("--mask-mode", type=str, default="", choices=["", "or", "and"])
    p.add_argument("--loss-node-weight-source", type=float, default=0.0)
    p.add_argument("--loss-node-weight-base-area-zero", type=float, default=0.0)
    p.add_argument("--loss-node-weight-indeg-ge", type=int, default=0)
    p.add_argument("--loss-node-weight-indeg-scale", type=float, default=0.0)
    p.add_argument("--loss-time-weight-fill", type=float, default=0.0)
    p.add_argument("--loss-time-fill-clip", type=float, default=1.0)
    p.add_argument("--loss-time-weight-inlet-pos", type=float, default=0.0)
    p.add_argument("--loss-time-inlet-pos-scale", type=float, default=0.25)
    return p.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _stage_defaults(stage: int) -> dict:
    s = int(stage)
    if s == 1:
        return dict(dyn_feat_version=1, static_aug_version=1, use_out_nbr_feats=False, nbr_pipe_dir=False, y2_knn_k=0, rain_lags=0, clamp=1.0, clamp_mode="hard", dropout=0.0, lr=1e-3)
    if s == 2:
        return dict(dyn_feat_version=1, static_aug_version=1, use_out_nbr_feats=False, nbr_pipe_dir=False, y2_knn_k=0, rain_lags=0, clamp=0.5, clamp_mode="hard", dropout=0.0, lr=1e-3)
    if s == 3:
        return dict(dyn_feat_version=1, static_aug_version=1, use_out_nbr_feats=False, nbr_pipe_dir=False, y2_knn_k=0, rain_lags=0, clamp=0.25, clamp_mode="hard", dropout=0.0, lr=1e-3)
    if s == 4:
        return dict(
            dyn_feat_version=8,
            static_aug_version=2,
            use_out_nbr_feats=True,
            nbr_pipe_dir=True,
            y2_knn_k=32,
            rain_lags=8,
            clamp=1.0,
            clamp_mode="tanh",
            dropout=0.1,
            lr=3e-4,
            mask_base_area_zero=True,
            mask_pipe_source=True,
            mask_pipe_indeg_ge=0,
            mask_mode="or",
        )
    if s == 5:
        return dict(
            dyn_feat_version=9,
            static_aug_version=2,
            use_out_nbr_feats=True,
            nbr_pipe_dir=True,
            y2_knn_k=32,
            rain_lags=8,
            clamp=1.0,
            clamp_mode="tanh",
            dropout=0.1,
            lr=3e-4,
            mask_base_area_zero=False,
            mask_pipe_source=False,
            mask_pipe_indeg_ge=2,
            mask_mode="or",
        )
    if s == 6:
        # Specialist booster trained on a fixed set of hard nodes (derived offline).
        node_list = (
            0,
            1,
            2,
            3,
            6,
            10,
            15,
            23,
            24,
            27,
            28,
            29,
            31,
            32,
            35,
            37,
            41,
            42,
            47,
            48,
            50,
            51,
            52,
            54,
            61,
            63,
            65,
            67,
            69,
            70,
            71,
            80,
            83,
            85,
            87,
            89,
            95,
            99,
            101,
            108,
            112,
            125,
            126,
            127,
            130,
            134,
            197,
        )
        return dict(
            dyn_feat_version=3,
            static_aug_version=2,
            use_out_nbr_feats=True,
            nbr_pipe_dir=True,
            y2_knn_k=32,
            rain_lags=8,
            clamp=0.5,
            clamp_mode="tanh",
            dropout=0.1,
            lr=3e-4,
            mask_node_list=node_list,
            mask_mode="or",
        )
    if s == 7:
        # Global small-clamp booster.
        return dict(
            dyn_feat_version=9,
            static_aug_version=2,
            use_out_nbr_feats=True,
            nbr_pipe_dir=True,
            y2_knn_k=32,
            rain_lags=8,
            clamp=0.25,
            clamp_mode="tanh",
            dropout=0.1,
            lr=3e-4,
            mask_node_list=(),
            mask_mode="or",
        )
    raise ValueError(f"unknown stage: {stage}")


def _mask_inputs(*, model_root: Path, graph2) -> dict:
    """Precompute masks + degrees for Model2 1D nodes."""
    n1 = int(graph2.n_1d)

    # Pipe-direction edges only (original direction, not duplicated reverse).
    src_full = graph2.edge_index_1d[0].cpu().numpy()
    dst_full = graph2.edge_index_1d[1].cpu().numpy()
    E1 = int(src_full.shape[0] // 2)
    src_dir = src_full[:E1]
    dst_dir = dst_full[:E1]

    indeg = np.bincount(dst_dir, minlength=n1).astype(np.int64, copy=False)
    outdeg = np.bincount(src_dir, minlength=n1).astype(np.int64, copy=False)
    mask_source = indeg == 0
    mask_sink = outdeg == 0

    # base_area==0 from raw static CSV (z-scored features lose the exact zeros).
    mask_ba0 = None
    base_area = None
    depth = None
    surface_elevation = None
    try:
        import pandas as pd

        df = pd.read_csv(model_root / "Model_2" / "train" / "1d_nodes_static.csv").sort_values("node_idx").reset_index(drop=True)
        cols = {c.lower(): c for c in df.columns}
        col = cols.get("base_area", cols.get("base area", None))
        if col is not None:
            ba = df[col].to_numpy(np.float32)
            ba = np.nan_to_num(ba, nan=0.0, posinf=0.0, neginf=0.0)
            base_area = ba.astype(np.float32, copy=False)
            mask_ba0 = ba <= 0.0
        c_depth = cols.get("depth", None)
        if c_depth is not None:
            d = df[c_depth].to_numpy(np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            depth = d.astype(np.float32, copy=False)
        c_surf = cols.get("surface_elevation", cols.get("surface elevation", None))
        if c_surf is not None:
            se = df[c_surf].to_numpy(np.float32)
            se = np.nan_to_num(se, nan=0.0, posinf=0.0, neginf=0.0)
            surface_elevation = se.astype(np.float32, copy=False)
    except Exception:
        mask_ba0 = None
        base_area = None
        depth = None
        surface_elevation = None

    return dict(
        src_full=src_full,
        dst_full=dst_full,
        src_dir=src_dir,
        dst_dir=dst_dir,
        indeg=indeg,
        outdeg=outdeg,
        mask_source=mask_source,
        mask_sink=mask_sink,
        mask_ba0=mask_ba0,
        base_area=base_area,
        depth=depth,
        surface_elevation=surface_elevation,
    )


def _mask_from_cfg(cfg: dict, *, masks: dict, n1: int) -> np.ndarray | None:
    use_ba0 = bool(cfg.get("mask_base_area_zero", False))
    use_source = bool(cfg.get("mask_pipe_source", False))
    indeg_ge = int(cfg.get("mask_pipe_indeg_ge", 0))
    node_list = tuple(int(x) for x in (cfg.get("mask_node_list", ()) or ()))
    mode = str(cfg.get("mask_mode", "or"))

    parts: list[np.ndarray] = []
    if node_list:
        idx = np.asarray(node_list, dtype=np.int64)
        if (idx < 0).any() or (idx >= int(n1)).any():
            raise ValueError("mask_node_list has out-of-range indices for Model2 1D nodes")
        m = np.zeros((int(n1),), dtype=np.bool_)
        m[idx] = True
        parts.append(m)
    if use_ba0:
        mb = masks.get("mask_ba0", None)
        if mb is None:
            raise ValueError("mask_base_area_zero requires base_area column in 1d_nodes_static.csv")
        parts.append(np.asarray(mb, dtype=np.bool_))
    if use_source:
        parts.append(np.asarray(masks["mask_source"], dtype=np.bool_))
    if indeg_ge > 0:
        parts.append(np.asarray(masks["indeg"], dtype=np.int64) >= indeg_ge)
    if not parts:
        return None
    m = parts[0].copy()
    if mode == "and":
        for p in parts[1:]:
            m &= p
    else:
        for p in parts[1:]:
            m |= p
    if m.shape != (n1,):
        raise RuntimeError("internal mask shape mismatch")
    if int(m.sum()) <= 0:
        raise ValueError("configured mask is empty")
    return m


def _expert_group_idx_topo3(*, masks: dict, n1: int) -> np.ndarray:
    group_idx = np.zeros((int(n1),), dtype=np.int64)
    mask_source = np.asarray(masks.get("mask_source", np.zeros((n1,), dtype=np.float32)), dtype=np.float32) > 0.0
    mask_ba0_raw = masks.get("mask_ba0", None)
    mask_ba0 = np.asarray(mask_ba0_raw, dtype=np.float32) > 0.0 if mask_ba0_raw is not None else np.zeros((int(n1),), dtype=bool)
    mask_src_ba0 = np.logical_or(mask_source, mask_ba0)
    indeg = np.asarray(masks.get("indeg", np.zeros((n1,), dtype=np.int64)), dtype=np.int64)
    mask_indeg = np.logical_and(indeg >= 2, np.logical_not(mask_src_ba0))
    group_idx[mask_src_ba0] = 1
    group_idx[mask_indeg] = 2
    return group_idx


def _pipe_weights(*, model_root: Path, graph2, masks: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Conductance proxy per pipe-direction edge + max incident diameter per node.
    Used by dyn-feat v8/v9.
    """
    n1 = int(graph2.n_1d)
    src_dir = masks["src_dir"]
    dst_dir = masks["dst_dir"]
    E = int(src_dir.shape[0])

    lap_w = np.ones((E,), dtype=np.float32)
    diam_max = np.zeros((n1,), dtype=np.float32)
    try:
        import pandas as pd

        df = pd.read_csv(model_root / "Model_2" / "train" / "1d_edges_static.csv").sort_values("edge_idx").reset_index(drop=True)
        cols = {c.lower(): c for c in df.columns}
        c_len = cols.get("length", None)
        c_d = cols.get("diameter", None)
        c_r = cols.get("roughness", None)
        if c_len is None or c_d is None or c_r is None:
            raise KeyError("missing length/diameter/roughness")
        length = df[c_len].to_numpy(np.float32)
        diam = df[c_d].to_numpy(np.float32)
        rough = df[c_r].to_numpy(np.float32)
        w = (diam * diam) / (np.maximum(rough, 1e-6) * np.maximum(length, 1e-6))
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if w.shape[0] != E:
            raise ValueError("edge static count mismatch vs edge_index (pipe-direction)")
        lap_w = w

        dmax = np.zeros((n1,), dtype=np.float32)
        np.maximum.at(dmax, src_dir, diam.astype(np.float32, copy=False))
        np.maximum.at(dmax, dst_dir, diam.astype(np.float32, copy=False))
        diam_max = np.nan_to_num(dmax, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    except Exception:
        # Fallbacks already initialized.
        pass
    return lap_w, diam_max


def _knn_2d_idx(*, model_root: Path, k: int, graph2) -> np.ndarray:
    """1D->2D KNN mapping based on planar distance (Model2 only)."""
    import pandas as pd

    base = model_root / "Model_2" / "train"
    n1_df = pd.read_csv(base / "1d_nodes_static.csv").sort_values("node_idx").reset_index(drop=True)
    n2_df = pd.read_csv(base / "2d_nodes_static.csv").sort_values("node_idx").reset_index(drop=True)
    x1 = n1_df["position_x"].to_numpy(np.float32)
    y1 = n1_df["position_y"].to_numpy(np.float32)
    x2 = n2_df["position_x"].to_numpy(np.float32)
    y2 = n2_df["position_y"].to_numpy(np.float32)
    if x1.shape[0] != int(graph2.n_1d) or x2.shape[0] != int(graph2.n_2d):
        raise ValueError("static CSV node counts do not match loaded graph")

    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]
    d2 = dx * dx + dy * dy
    kk = int(k)
    part = np.argpartition(d2, kth=min(kk - 1, d2.shape[1] - 1), axis=1)[:, :kk]
    row = np.arange(d2.shape[0])[:, None]
    part_sorted = part[row, np.argsort(d2[row, part], axis=1)]
    return part_sorted.astype(np.int64, copy=False)


def _pick_node_static(
    *,
    expected_dim: int,
    raw: torch.Tensor,
    aug1: torch.Tensor,
    aug2: torch.Tensor,
) -> torch.Tensor:
    if int(raw.shape[1]) == int(expected_dim):
        return raw
    if int(aug1.shape[1]) == int(expected_dim):
        return aug1
    if int(aug2.shape[1]) == int(expected_dim):
        return aug2
    raise ValueError(f"could not match node_static dim={expected_dim} (raw={raw.shape[1]} aug1={aug1.shape[1]} aug2={aug2.shape[1]})")


def _build_dyn_1d(
    *,
    dyn_ver: int,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    rain_2d: np.ndarray,
    warmup: int,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    # v8/v9 extras
    y2_knn_mean: np.ndarray | None,
    y2_knn_max: np.ndarray | None,
    lap_src: np.ndarray | None,
    lap_dst: np.ndarray | None,
    lap_w: np.ndarray | None,
    diam_max_1d: np.ndarray | None,
    base_area_1d: np.ndarray | None,
    conn_area_1d: np.ndarray | None,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    inlet_pred_1d: np.ndarray | None,
    aux_y1_base: np.ndarray | None,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
    volagg_pred_1d: np.ndarray | None,
    use_out_nbr: bool,
    rain_lags: int,
) -> torch.Tensor:
    dv = int(dyn_ver)
    if dv == 1:
        return build_dyn_features_1d_v1(
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
    if dv == 3:
        if y2_knn_mean is None or y2_knn_max is None:
            raise ValueError("dyn_ver=3 requires y2_knn_*")
        return build_dyn_features_1d_v3(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 8:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None:
            raise ValueError("dyn_ver=8 requires y2_knn_* and lap_*")
        return build_dyn_features_1d_v8(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 9:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=9 requires y2_knn_*, lap_*, diam_max_1d")
        return build_dyn_features_1d_v9(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 10:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=10 requires y2_knn_*, lap_*, diam_max_1d")
        return build_dyn_features_1d_v10(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 11:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=11 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_1d is None or vol2_agg is None:
            raise ValueError("dyn_ver=11 requires inlet_1d and vol2_agg")
        return build_dyn_features_1d_v11(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 12:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=12 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=12 requires inlet_pred_1d")
        return build_dyn_features_1d_v12(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 13:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=13 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=13 requires inlet_pred_1d")
        if base_area_1d is None or conn_area_1d is None:
            raise ValueError("dyn_ver=13 requires base_area_1d and conn_area_1d")
        return build_dyn_features_1d_v13(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            base_area_1d=base_area_1d,
            conn_area_1d=conn_area_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 14:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=14 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=14 requires inlet_pred_1d")
        if edgeflow_node_feats is None:
            raise ValueError("dyn_ver=14 requires edgeflow_node_feats")
        return build_dyn_features_1d_v14(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            edgeflow_node_feats=edgeflow_node_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 15:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=15 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=15 requires inlet_pred_1d")
        if volagg_pred_1d is None:
            raise ValueError("dyn_ver=15 requires volagg_pred_1d")
        if conn_area_1d is None:
            raise ValueError("dyn_ver=15 requires conn_area_1d")
        return build_dyn_features_1d_v15(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            volagg_pred_1d=volagg_pred_1d,
            conn_area_1d=conn_area_1d,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 16:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=16 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=16 requires inlet_pred_1d")
        if aux_y1_base is None:
            raise ValueError("dyn_ver=16 requires aux_y1_base")
        return build_dyn_features_1d_v16(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            y1_aux_base=aux_y1_base,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 17:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=17 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=17 requires inlet_pred_1d")
        if edgeflow_node_feats is None:
            raise ValueError("dyn_ver=17 requires edgeflow_node_feats")
        return build_dyn_features_1d_v17(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            edgeflow_node_feats=edgeflow_node_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 18:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=18 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=18 requires inlet_pred_1d")
        if surfaceflow_node_feats is None:
            raise ValueError("dyn_ver=18 requires surfaceflow_node_feats")
        return build_dyn_features_1d_v18(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            surfaceflow_node_feats=surfaceflow_node_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 19:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=19 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=19 requires inlet_pred_1d")
        if surfaceflow_slot_feats is None:
            raise ValueError("dyn_ver=19 requires surfaceflow_slot_feats")
        return build_dyn_features_1d_v19(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 20:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=20 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=20 requires inlet_pred_1d")
        if surfaceflow_node_feats is None:
            raise ValueError("dyn_ver=20 requires surfaceflow_node_feats")
        if surfaceflow_slot_feats is None:
            raise ValueError("dyn_ver=20 requires surfaceflow_slot_feats")
        return build_dyn_features_1d_v20(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            surfaceflow_node_feats=surfaceflow_node_feats,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 21:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=21 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=21 requires inlet_pred_1d")
        if surfaceflow_slot_feats is None:
            raise ValueError("dyn_ver=21 requires surfaceflow_slot_feats")
        if local2d_node_feats is None:
            raise ValueError("dyn_ver=21 requires local2d_node_feats")
        return build_dyn_features_1d_v21(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            local2d_node_feats=local2d_node_feats,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    if dv == 22:
        if y2_knn_mean is None or y2_knn_max is None or lap_src is None or lap_dst is None or lap_w is None or diam_max_1d is None:
            raise ValueError("dyn_ver=22 requires y2_knn_*, lap_*, diam_max_1d")
        if inlet_pred_1d is None:
            raise ValueError("dyn_ver=22 requires inlet_pred_1d")
        if surfaceflow_slot_feats is None:
            raise ValueError("dyn_ver=22 requires surfaceflow_slot_feats")
        if aux_y1_base is None:
            raise ValueError("dyn_ver=22 requires aux_y1_base")
        return build_dyn_features_1d_v22(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            rain_2d=rain_2d,
            inlet_pred_1d=inlet_pred_1d,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            y1_aux_base=aux_y1_base,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            lap_src=lap_src,
            lap_dst=lap_dst,
            lap_w=lap_w,
            diam_max_1d=diam_max_1d,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
            use_out_nbr=bool(use_out_nbr),
            rain_lags=int(rain_lags),
        )
    raise ValueError(f"unsupported dyn_feat_version: {dv}")


def _build_warm_ctx_1d(
    *,
    warm_ctx_ver: int,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    warmup: int,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    y2_knn_mean: np.ndarray | None,
    y2_knn_max: np.ndarray | None,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
) -> torch.Tensor | None:
    wv = int(warm_ctx_ver)
    if wv <= 0:
        return None
    if wv == 1:
        if y2_knn_mean is None or y2_knn_max is None:
            raise ValueError("warm_ctx_version=1 requires y2_knn_mean/y2_knn_max")
        return build_warm_context_1d_v1(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            warmup=warmup,
        )
    if wv == 2:
        if y2_knn_mean is None or y2_knn_max is None:
            raise ValueError("warm_ctx_version=2 requires y2_knn_mean/y2_knn_max")
        return build_warm_context_1d_v2(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            warmup=warmup,
            edgeflow_node_feats=edgeflow_node_feats,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            local2d_node_feats=local2d_node_feats,
        )
    raise ValueError(f"unsupported warm_ctx_version: {wv}")


def _build_warm_seq_1d(
    *,
    warm_seq_ver: int,
    y1_base: np.ndarray,
    y2_agg: np.ndarray,
    warmup: int,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    y2_knn_mean: np.ndarray | None,
    y2_knn_max: np.ndarray | None,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
) -> torch.Tensor | None:
    sv = int(warm_seq_ver)
    if sv <= 0:
        return None
    if sv == 1:
        if y2_knn_mean is None or y2_knn_max is None:
            raise ValueError("warm_seq_version=1 requires y2_knn_mean/y2_knn_max")
        return build_warm_sequence_1d_v1(
            y1_base=y1_base,
            y2_agg=y2_agg,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            warmup=warmup,
            edgeflow_node_feats=edgeflow_node_feats,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            local2d_node_feats=local2d_node_feats,
        )
    raise ValueError(f"unsupported warm_seq_version: {sv}")


def _predict_inlet_1d_from_ckpt(
    *,
    ckpt: dict,
    graph2,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    q1_init: np.ndarray,
    volagg_init: np.ndarray | None,
    warmup: int,
    surface_slot_feats: np.ndarray | None = None,
    local2d_node_feats: np.ndarray | None = None,
) -> np.ndarray:
    kind2 = str((ckpt.get("model_2", {}) or {}).get("kind", ""))
    if kind2 == "aux_inlet_m2_1d_surface_slots":
        return predict_inlet_from_ckpt(
            ckpt=ckpt,
            y2_2d=y2_2d,
            rain_2d=rain_2d,
            area_2d=graph2.area_2d.cpu().numpy().astype(np.float32, copy=False),
            conn_src_1d=graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False),
            conn_dst_2d=graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False),
            n_1d=int(graph2.n_1d),
            q1_init=q1_init,
            warmup=warmup,
            surface_slot_feats=surface_slot_feats,
            local2d_node_feats=local2d_node_feats,
        )
    if kind2 not in {"split_1d2d_coupled_inlet", "split_1d2d_coupled_storage_inlet"}:
        raise ValueError("inlet_ckpt must be split_1d2d_coupled_inlet, split_1d2d_coupled_storage_inlet, or aux_inlet_m2_1d_surface_slots")
    area_2d = graph2.area_2d.cpu().numpy()
    conn_src_1d = graph2.conn_src_1d.cpu().numpy()
    conn_dst_2d = graph2.conn_dst_2d.cpu().numpy()
    exo = build_coupled_1d_exo(
        y2_2d=y2_2d,
        rain_2d=rain_2d,
        area_2d=area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=int(graph2.n_1d),
    )
    p1 = ckpt["model_2"]["parts"]["1d"]
    if kind2 == "split_1d2d_coupled_inlet":
        q_model = p1["q_model"]
        return rollout_regime_arkx_exo(
            w=q_model["w"].numpy(),
            bins=tuple(float(x) for x in q_model["bins"]),
            y_init=q1_init,
            rain=rain_2d,
            exo=exo,
            warmup=warmup,
        )
    if volagg_init is None:
        raise ValueError("split_1d2d_coupled_storage_inlet inlet prediction requires volagg_init")
    conn_area_1d = connected_area_1d(
        area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=int(graph2.n_1d),
    ).astype(np.float32, copy=False)
    v_model = p1["v_model"]
    vmd_raw = v_model.get("max_delta", None)
    if torch.is_tensor(vmd_raw):
        vmd = vmd_raw.numpy()
    else:
        vmd = vmd_raw
    v_pred = rollout_regime_arkx_exo(
        w=v_model["w"].numpy(),
        bins=tuple(float(x) for x in v_model["bins"]),
        y_init=volagg_init,
        rain=rain_2d,
        exo=exo,
        warmup=warmup,
        max_delta=vmd,
    )
    q_model = p1["q_model"]
    return rollout_regime_arkx_exo(
        w=q_model["w"].numpy(),
        bins=tuple(float(x) for x in q_model["bins"]),
        y_init=q1_init,
        rain=rain_2d,
        exo=build_storage_augmented_exo(exo, v_pred, conn_area_1d),
        warmup=warmup,
    )


def _apply_resid2d(
    *,
    resid2d: ResidualNodeGRU,
    node_static_2d: torch.Tensor,  # [N2,F]
    y2_base: np.ndarray,  # [T,N2]
    rain_2d: np.ndarray,  # [T,N2]
    bed2: np.ndarray,  # [N2]
    warmup: int,
    dyn_feat: str = "",
    y1_base: np.ndarray | None = None,
    conn_src_1d: np.ndarray | None = None,
    conn_dst_2d: np.ndarray | None = None,
    n_2d: int | None = None,
    edge_index_2d: torch.LongTensor | None = None,
    edge_deg_inv_2d: torch.Tensor | None = None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    def _agg_1d_to_2d_mean(y1: np.ndarray) -> np.ndarray:
        if conn_src_1d is None or conn_dst_2d is None or n_2d is None:
            raise ValueError("v3cpl requires conn_src_1d, conn_dst_2d, and n_2d")
        y1 = np.asarray(y1, dtype=np.float32)
        if y1.ndim != 2:
            raise ValueError("y1_base must be [T, N1] for v3cpl")
        out = np.zeros((int(y1.shape[0]), int(n_2d)), dtype=np.float32)
        vals = y1[:, conn_src_1d]
        for j, d in enumerate(conn_dst_2d.tolist()):
            out[:, d] += vals[:, j]
        deg2 = np.bincount(conn_dst_2d, minlength=int(n_2d)).astype(np.float32, copy=False)
        out = out / np.maximum(deg2[None, :], 1.0)
        return out

    y2 = np.asarray(y2_base, dtype=np.float32).copy()
    x2 = build_dyn_features_2d(y2_base=y2, rain_2d=rain_2d, bed_2d=bed2, warmup=warmup)  # [T',N2,5]
    x2 = x2.to(device, dtype=amp_dtype) if amp_enabled else x2.to(device)
    dyn_dim = int(getattr(resid2d.cfg, "dyn_dim", 5))
    dyn_feat_key = str(dyn_feat or "")
    if dyn_feat_key == "v3cpl":
        if y1_base is None:
            raise ValueError("v3cpl requires y1_base")
        T = int(y2.shape[0])
        idx = np.arange(int(warmup), int(T), dtype=np.int64)
        y1_to_2 = _agg_1d_to_2d_mean(y1_base)
        deg2 = np.bincount(conn_dst_2d, minlength=int(n_2d)).astype(np.float32, copy=False)
        has_conn2 = (deg2 > 0).astype(np.float32, copy=False)
        diff = (y1_to_2 - y2) * has_conn2[None, :]
        diff_t = diff[idx]
        diff_tm1 = diff[idx - 1]
        ddiff = diff_t - diff_tm1
        cpl = np.stack([diff_t, ddiff], axis=-1).astype(np.float32, copy=False)
        x2 = torch.cat([x2, torch.from_numpy(cpl).to(device=x2.device, dtype=x2.dtype)], dim=-1)
    elif dyn_feat_key == "v2nbr" or dyn_dim in (7, 9):
        if edge_index_2d is None:
            raise ValueError("edge_index_2d is required for 2D dyn_dim in (7,9)")
        x2 = augment_dyn_features_2d_v2_nbrmean(
            x2,
            edge_index=edge_index_2d,
            edge_deg_inv=edge_deg_inv_2d,
            include_diff=dyn_dim == 9,
        )
    elif dyn_dim != 5:
        raise ValueError(f"unsupported 2D dyn_dim={dyn_dim} (expected 5,7,9)")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            r = (
                resid2d(x2, node_static_2d, edge_index=edge_index_2d, edge_deg_inv=edge_deg_inv_2d)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
    y2[warmup:] = y2[warmup:] + r
    return y2


def _build_coupling_views(
    *,
    cfg_like: dict,
    y2_main: np.ndarray,
    y2_alt: np.ndarray | None,
    bed2: np.ndarray,
    conn_src: np.ndarray,
    conn_dst: np.ndarray,
    n_1d: int,
    knn_idx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    blend = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
    blend_knn = float(cfg_like.get("resid2d_coupling_blend_knn", 0.0) or 0.0)
    if not (0.0 <= blend <= 1.0):
        raise ValueError("resid2d_coupling_blend must be in [0,1]")
    if not (0.0 <= blend_knn <= 1.0):
        raise ValueError("resid2d_coupling_blend_knn must be in [0,1]")

    if y2_alt is not None and blend > 0.0:
        y2_agg_src = ((1.0 - blend) * y2_main + blend * y2_alt).astype(np.float32, copy=False)
    else:
        y2_agg_src = y2_main

    y2_agg = aggregate_2d_to_1d_mean(
        y2_agg_src,
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        n_1d=n_1d,
    ).astype(np.float32, copy=False)

    if knn_idx is None:
        return y2_agg, None, None

    if y2_alt is not None and blend_knn > 0.0:
        y2_knn_src = ((1.0 - blend_knn) * y2_main + blend_knn * y2_alt).astype(np.float32, copy=False)
    else:
        y2_knn_src = y2_main

    y2_depth = (y2_knn_src - bed2[None, :]).astype(np.float32, copy=False)
    y2_knn = y2_depth[:, knn_idx]
    y2_knn_mean = y2_knn.mean(axis=2).astype(np.float32, copy=False)
    y2_knn_max = y2_knn.max(axis=2).astype(np.float32, copy=False)
    return y2_agg, y2_knn_mean, y2_knn_max


def _build_inlet_pred_views(
    *,
    cfg_like: dict,
    inlet_ckpts: dict[str, dict],
    surfaceflow_ckpts: dict[str, dict],
    graph2,
    y2_main: np.ndarray,
    y2_alt: np.ndarray | None,
    rain_2d: np.ndarray,
    q1_init: np.ndarray | None,
    volagg_init: np.ndarray | None,
    q2edge_init: np.ndarray | None,
    warmup: int,
    surface_slot_edges: np.ndarray | None = None,
    surface_slot_sign: np.ndarray | None = None,
    center_cell: np.ndarray | None = None,
    neighbor_slots: np.ndarray | None = None,
) -> np.ndarray | None:
    path = str(cfg_like.get("inlet_ckpt", "") or "")
    if not path:
        return None
    if q1_init is None:
        raise ValueError("inlet_ckpt requires q1_init / inlet_1d warmup data")
    ckpt = inlet_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded inlet_ckpt: {path}")
    blend = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
    if not (0.0 <= blend <= 1.0):
        raise ValueError("resid2d_coupling_blend must be in [0,1]")
    if y2_alt is not None and blend > 0.0:
        y2_src = ((1.0 - blend) * y2_main + blend * y2_alt).astype(np.float32, copy=False)
    else:
        y2_src = y2_main
    kind2 = str((ckpt.get("model_2", {}) or {}).get("kind", ""))
    surface_slot_feats = None
    local2d_node_feats = None
    if kind2 == "aux_inlet_m2_1d_surface_slots":
        if q2edge_init is None:
            raise ValueError("aux_inlet_m2_1d_surface_slots requires q2edge_init / warmup 2D edge flow data")
        if surface_slot_edges is None or surface_slot_sign is None:
            raise ValueError("aux_inlet_m2_1d_surface_slots requires surface slot indices")
        surface_slot_feats = _build_surfaceflow_pred_slot_views(
            cfg_like=cfg_like,
            surfaceflow_ckpts=surfaceflow_ckpts,
            graph2=graph2,
            y2_main=y2_main,
            y2_alt=y2_alt,
            rain_2d=rain_2d,
            q2edge_init=q2edge_init,
            warmup=warmup,
            surface_slot_edges=surface_slot_edges,
            surface_slot_sign=surface_slot_sign,
        )
        if bool(ckpt.get("use_local2d_slots", False)):
            if center_cell is None or neighbor_slots is None:
                raise ValueError("aux_inlet_m2_1d_surface_slots with local2d slots requires local 2D slot indices")
            local2d_node_feats = build_coupled_neighbor_node_features(
                y2_2d=y2_src,
                bed_2d=graph2.head_offset[graph2.n_1d :].cpu().numpy().astype(np.float32, copy=False),
                center_cell=center_cell,
                neighbor_slots=neighbor_slots,
            )
    return _predict_inlet_1d_from_ckpt(
        ckpt=ckpt,
        graph2=graph2,
        y2_2d=y2_src,
        rain_2d=rain_2d,
        q1_init=q1_init,
        volagg_init=volagg_init,
        warmup=warmup,
        surface_slot_feats=surface_slot_feats,
        local2d_node_feats=local2d_node_feats,
    )


def _build_edgeflow_pred_views(
    *,
    cfg_like: dict,
    edgeflow_ckpts: dict[str, dict],
    graph2,
    y1_1d: np.ndarray,
    rain_2d: np.ndarray,
    qedge_init: np.ndarray | None,
    warmup: int,
) -> np.ndarray | None:
    path = str(cfg_like.get("edgeflow_ckpt", "") or "")
    if not path:
        return None
    if qedge_init is None:
        raise ValueError("edgeflow_ckpt requires qedge_init / warmup 1D edge flow data")
    ckpt = edgeflow_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded edgeflow_ckpt: {path}")
    src_full = graph2.edge_index_1d[0].cpu().numpy().astype(np.int64, copy=False)
    dst_full = graph2.edge_index_1d[1].cpu().numpy().astype(np.int64, copy=False)
    e1 = int(src_full.shape[0] // 2)
    edge_from = src_full[:e1]
    edge_to = dst_full[:e1]
    q_pred = predict_edge_flow_from_ckpt(
        ckpt=ckpt,
        y1_1d=y1_1d,
        rain_2d=rain_2d,
        area_2d=graph2.area_2d.cpu().numpy().astype(np.float32, copy=False),
        conn_src_1d=graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False),
        conn_dst_2d=graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False),
        edge_from=edge_from,
        edge_to=edge_to,
        q_edge_init=qedge_init,
        warmup=warmup,
    )
    inflow, outflow, net, total = aggregate_edge_flows_to_nodes(
        q_edge=q_pred,
        edge_from=edge_from,
        edge_to=edge_to,
        n_1d=int(graph2.n_1d),
    )
    return np.stack([inflow, outflow, net, total], axis=-1).astype(np.float32, copy=False)


def _build_edgeflow_pred_slot_views(
    *,
    cfg_like: dict,
    edgeflow_ckpts: dict[str, dict],
    graph2,
    y1_1d: np.ndarray,
    rain_2d: np.ndarray,
    qedge_init: np.ndarray | None,
    warmup: int,
    edge_in_slots: np.ndarray,
    edge_out_slots: np.ndarray,
) -> np.ndarray | None:
    path = str(cfg_like.get("edgeflow_ckpt", "") or "")
    if not path:
        return None
    if qedge_init is None:
        raise ValueError("edgeflow_ckpt requires qedge_init / warmup 1D edge flow data")
    ckpt = edgeflow_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded edgeflow_ckpt: {path}")
    src_full = graph2.edge_index_1d[0].cpu().numpy().astype(np.int64, copy=False)
    dst_full = graph2.edge_index_1d[1].cpu().numpy().astype(np.int64, copy=False)
    e1 = int(src_full.shape[0] // 2)
    edge_from = src_full[:e1]
    edge_to = dst_full[:e1]
    q_pred = predict_edge_flow_from_ckpt(
        ckpt=ckpt,
        y1_1d=y1_1d,
        rain_2d=rain_2d,
        area_2d=graph2.area_2d.cpu().numpy().astype(np.float32, copy=False),
        conn_src_1d=graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False),
        conn_dst_2d=graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False),
        edge_from=edge_from,
        edge_to=edge_to,
        q_edge_init=qedge_init,
        warmup=warmup,
    )
    return build_edge_flow_slot_features(
        q_edge=q_pred,
        edge_in_slots=edge_in_slots,
        edge_out_slots=edge_out_slots,
    )


def _build_volagg_pred_views(
    *,
    cfg_like: dict,
    volagg_ckpts: dict[str, dict],
    graph2,
    y2_main: np.ndarray,
    y2_alt: np.ndarray | None,
    rain_2d: np.ndarray,
    vagg_init: np.ndarray | None,
    warmup: int,
) -> np.ndarray | None:
    path = str(cfg_like.get("volagg_ckpt", "") or "")
    if not path:
        return None
    if vagg_init is None:
        raise ValueError("volagg_ckpt requires vagg_init / warmup coupled 2D volume data")
    ckpt = volagg_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded volagg_ckpt: {path}")
    blend = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
    if not (0.0 <= blend <= 1.0):
        raise ValueError("resid2d_coupling_blend must be in [0,1]")
    if y2_alt is not None and blend > 0.0:
        y2_src = ((1.0 - blend) * y2_main + blend * y2_alt).astype(np.float32, copy=False)
    else:
        y2_src = y2_main
    return predict_volagg_from_ckpt(
        ckpt=ckpt,
        y2_2d=y2_src,
        rain_2d=rain_2d,
        area_2d=graph2.area_2d.cpu().numpy().astype(np.float32, copy=False),
        conn_src_1d=graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False),
        conn_dst_2d=graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False),
        n_1d=int(graph2.n_1d),
        vagg_init=vagg_init,
        warmup=warmup,
    )


def _build_surfaceflow_pred_views(
    *,
    cfg_like: dict,
    surfaceflow_ckpts: dict[str, dict],
    graph2,
    y2_main: np.ndarray,
    y2_alt: np.ndarray | None,
    rain_2d: np.ndarray,
    q2edge_init: np.ndarray | None,
    warmup: int,
) -> np.ndarray | None:
    path = str(cfg_like.get("surfaceflow_ckpt", "") or "")
    if not path:
        return None
    if q2edge_init is None:
        raise ValueError("surfaceflow_ckpt requires q2edge_init / warmup 2D edge flow data")
    ckpt = surfaceflow_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded surfaceflow_ckpt: {path}")
    blend = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
    if not (0.0 <= blend <= 1.0):
        raise ValueError("resid2d_coupling_blend must be in [0,1]")
    if y2_alt is not None and blend > 0.0:
        y2_src = ((1.0 - blend) * y2_main + blend * y2_alt).astype(np.float32, copy=False)
    else:
        y2_src = y2_main
    src_full = graph2.edge_index_2d[0].cpu().numpy().astype(np.int64, copy=False)
    dst_full = graph2.edge_index_2d[1].cpu().numpy().astype(np.int64, copy=False)
    e2 = int(src_full.shape[0] // 2)
    edge_from = src_full[:e2]
    edge_to = dst_full[:e2]
    q_pred = predict_surface_flow_from_ckpt(
        ckpt=ckpt,
        y2_2d=y2_src,
        rain_2d=rain_2d,
        bed_2d=graph2.head_offset[graph2.n_1d :].cpu().numpy().astype(np.float32, copy=False),
        edge_from=edge_from,
        edge_to=edge_to,
        q_edge_init=q2edge_init,
        warmup=warmup,
    )
    return build_surface_flow_1d_features(
        q_edge=q_pred,
        edge_from=edge_from,
        edge_to=edge_to,
        n_2d=int(graph2.n_2d),
        conn_src_1d=graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False),
        conn_dst_2d=graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False),
        n_1d=int(graph2.n_1d),
    )


def _build_surfaceflow_pred_slot_views(
    *,
    cfg_like: dict,
    surfaceflow_ckpts: dict[str, dict],
    graph2,
    y2_main: np.ndarray,
    y2_alt: np.ndarray | None,
    rain_2d: np.ndarray,
    q2edge_init: np.ndarray | None,
    warmup: int,
    surface_slot_edges: np.ndarray,
    surface_slot_sign: np.ndarray,
) -> np.ndarray | None:
    path = str(cfg_like.get("surfaceflow_ckpt", "") or "")
    if not path:
        return None
    if q2edge_init is None:
        raise ValueError("surfaceflow_ckpt requires q2edge_init / warmup 2D edge flow data")
    ckpt = surfaceflow_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded surfaceflow_ckpt: {path}")
    blend = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
    if not (0.0 <= blend <= 1.0):
        raise ValueError("resid2d_coupling_blend must be in [0,1]")
    if y2_alt is not None and blend > 0.0:
        y2_src = ((1.0 - blend) * y2_main + blend * y2_alt).astype(np.float32, copy=False)
    else:
        y2_src = y2_main
    src_full = graph2.edge_index_2d[0].cpu().numpy().astype(np.int64, copy=False)
    dst_full = graph2.edge_index_2d[1].cpu().numpy().astype(np.int64, copy=False)
    e2 = int(src_full.shape[0] // 2)
    edge_from = src_full[:e2]
    edge_to = dst_full[:e2]
    q_pred = predict_surface_flow_from_ckpt(
        ckpt=ckpt,
        y2_2d=y2_src,
        rain_2d=rain_2d,
        bed_2d=graph2.head_offset[graph2.n_1d :].cpu().numpy().astype(np.float32, copy=False),
        edge_from=edge_from,
        edge_to=edge_to,
        q_edge_init=q2edge_init,
        warmup=warmup,
    )
    return build_coupled_surface_slot_features(
        q_edge=q_pred,
        edge_slots=surface_slot_edges,
        edge_sign=surface_slot_sign,
    )


def _build_aux_baseline_views(
    *,
    cfg_like: dict,
    aux_baseline_ckpts: dict[str, dict],
    graph2,
    y1_init: np.ndarray,
    y2_init: np.ndarray,
    rain_2d: np.ndarray,
    q1_init: np.ndarray | None,
    vagg_init: np.ndarray | None,
    warmup: int,
) -> np.ndarray | None:
    path = str(cfg_like.get("aux_baseline_ckpt", "") or "")
    if not path:
        return None
    ckpt = aux_baseline_ckpts.get(path)
    if ckpt is None:
        raise ValueError(f"missing loaded aux_baseline_ckpt: {path}")
    y1_aux, _ = predict_model2_from_baseline_ckpts(
        [ckpt],
        graph2=graph2,
        mixed_mode="single",
        alpha_1d=1.0,
        alpha_2d=1.0,
        y1_init=y1_init,
        y2_init=y2_init,
        rain_2d=rain_2d,
        q1_init=q1_init,
        vagg_init=vagg_init,
        warmup=warmup,
    )
    return y1_aux.astype(np.float32, copy=False)


def _ordered_pre_models(*, cfg_like: dict, pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]]) -> list[tuple[dict, ResidualNodeGRU, dict]]:
    paths = tuple(str(x) for x in (cfg_like.get("aux_pre_residual_ckpts", ()) or ()))
    if not paths:
        return []
    out: list[tuple[dict, ResidualNodeGRU, dict]] = []
    for path in paths:
        item = pre_model_map.get(path)
        if item is None:
            raise ValueError(f"missing loaded aux-pre-residual ckpt: {path}")
        out.append(item)
    return out


def _rollout_pre_residual_stack(
    *,
    pre_models: list[tuple[dict, ResidualNodeGRU, dict]],
    y1_base_init: np.ndarray,
    node_static_raw: torch.Tensor,
    node_static_aug1: torch.Tensor,
    node_static_aug2: torch.Tensor | None,
    masks: dict,
    y2_corr: np.ndarray,
    y2_corr_coupling: dict[str, np.ndarray],
    bed2: np.ndarray,
    conn_src: np.ndarray,
    conn_dst: np.ndarray,
    n1: int,
    knn_idx: np.ndarray | None,
    inlet_ckpts: dict[str, dict],
    aux_baseline_ckpts: dict[str, dict],
    edgeflow_ckpts: dict[str, dict],
    surfaceflow_ckpts: dict[str, dict],
    volagg_ckpts: dict[str, dict],
    graph2,
    rain: np.ndarray,
    y2_init: np.ndarray,
    inlet_1d_ev: np.ndarray | None,
    vol2_agg_ev: np.ndarray | None,
    volagg_init_ev: np.ndarray | None,
    surfaceflow_2d_ev: np.ndarray | None,
    edgeflow_1d_ev: np.ndarray | None,
    warmup: int,
    surface_slot_edges: np.ndarray | None,
    surface_slot_sign: np.ndarray | None,
    local2d_center_cell: np.ndarray | None,
    local2d_neighbor_slots: np.ndarray | None,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    lap_w_dir: np.ndarray,
    diam_max_1d: np.ndarray,
    conn_area_1d: np.ndarray | None,
    edgeflow_in_slots: np.ndarray | None,
    edgeflow_out_slots: np.ndarray | None,
    edge_index_1d_full: torch.LongTensor,
    edge_deg_inv_1d_full: torch.Tensor,
    edge_weight_1d_full: torch.Tensor,
    edge_index_1d_dir: torch.LongTensor,
    edge_deg_inv_1d_dir: torch.Tensor,
    edge_weight_1d_dir: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    y1_feat = y1_base_init.astype(np.float32, copy=True)
    for pre, m, pcfg in pre_models:
        mcfg = ResidualGRUConfig(**pre["model_cfg"])
        node_static_p = _pick_node_static(
            expected_dim=int(mcfg.static_dim),
            raw=node_static_raw,
            aug1=node_static_aug1,
            aug2=node_static_aug2,
        )

        use_nbr = bool(pcfg.get("use_nbr_feats", False)) or bool(pcfg.get("use_out_nbr_feats", False))
        pipe_dir = bool(pcfg.get("nbr_pipe_dir", False))
        if use_nbr:
            src = masks["src_dir"] if pipe_dir else masks["src_full"]
            dst = masks["dst_dir"] if pipe_dir else masks["dst_full"]
        else:
            src = None
            dst = None

        pre_mask = _mask_from_cfg(pcfg, masks=masks, n1=n1)
        coupling_path = str(pcfg.get("resid2d_ckpt_coupling", "") or "")
        y2_agg_p, y2_knn_mean_p, y2_knn_max_p = _build_coupling_views(
            cfg_like=pcfg,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            bed2=bed2,
            conn_src=conn_src,
            conn_dst=conn_dst,
            n_1d=n1,
            knn_idx=knn_idx,
        )
        inlet_pred_p = _build_inlet_pred_views(
            cfg_like=pcfg,
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
        aux_y1_base_p = _build_aux_baseline_views(
            cfg_like=pcfg,
            aux_baseline_ckpts=aux_baseline_ckpts,
            graph2=graph2,
            y1_init=y1_base_init,
            y2_init=y2_init,
            rain_2d=rain,
            q1_init=inlet_1d_ev,
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )
        if int(pcfg.get("dyn_feat_version", 1) or 1) == 17:
            edgeflow_pred_p = _build_edgeflow_pred_slot_views(
                cfg_like=pcfg,
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
            edgeflow_pred_p = _build_edgeflow_pred_views(
                cfg_like=pcfg,
                edgeflow_ckpts=edgeflow_ckpts,
                graph2=graph2,
                y1_1d=y1_feat,
                rain_2d=rain,
                qedge_init=edgeflow_1d_ev,
                warmup=warmup,
            )
        volagg_pred_p = _build_volagg_pred_views(
            cfg_like=pcfg,
            volagg_ckpts=volagg_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )
        surfaceflow_pred_p = _build_surfaceflow_pred_views(
            cfg_like=pcfg,
            surfaceflow_ckpts=surfaceflow_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            q2edge_init=surfaceflow_2d_ev,
            warmup=warmup,
        )
        if int(pcfg.get("dyn_feat_version", 1) or 1) in (19, 20, 21, 22):
            surfaceflow_slot_pred_p = _build_surfaceflow_pred_slot_views(
                cfg_like=pcfg,
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
            surfaceflow_slot_pred_p = None
        if int(pcfg.get("dyn_feat_version", 1) or 1) == 21:
            blend_local = float(pcfg.get("resid2d_coupling_blend", 0.0) or 0.0)
            y2_local_src_p = y2_corr
            y2_alt_p = y2_corr_coupling.get(coupling_path)
            if y2_alt_p is not None and blend_local > 0.0:
                y2_local_src_p = ((1.0 - blend_local) * y2_corr + blend_local * y2_alt_p).astype(np.float32, copy=False)
            local2d_node_pred_p = build_coupled_neighbor_node_features(
                y2_2d=y2_local_src_p,
                bed_2d=bed2,
                center_cell=local2d_center_cell,
                neighbor_slots=local2d_neighbor_slots,
            )
        else:
            local2d_node_pred_p = None
        if str(pcfg.get("graph_mix_edge", "full") or "full") == "dir":
            eidx_pre = edge_index_1d_dir
            einv_pre = edge_deg_inv_1d_dir
            ew_pre = edge_weight_1d_dir
        else:
            eidx_pre = edge_index_1d_full
            einv_pre = edge_deg_inv_1d_full
            ew_pre = edge_weight_1d_full
        y1_feat = _apply_resid1d_ckpt(
            ckpt=pre,
            model=m,
            node_static=node_static_p,
            cfg=pcfg,
            node_mask=pre_mask,
            expert_group_idx=_expert_group_idx_topo3(masks=masks, n1=n1) if bool(pcfg.get("expert_group_topo3", False)) else None,
            y1_feat=y1_feat,
            y2_agg=y2_agg_p,
            rain_2d=rain,
            warmup=warmup,
            inv1=inv1,
            bed_agg=bed_agg,
            has_conn=has_conn,
            nbr_src=src,
            nbr_dst=dst,
            y2_knn_mean=y2_knn_mean_p,
            y2_knn_max=y2_knn_max_p,
            lap_src=masks["src_dir"],
            lap_dst=masks["dst_dir"],
            lap_w=lap_w_dir,
            diam_max_1d=diam_max_1d,
            base_area_1d=masks.get("base_area", None),
            conn_area_1d=conn_area_1d,
            inlet_1d=inlet_1d_ev,
            vol2_agg=vol2_agg_ev,
            inlet_pred_1d=inlet_pred_p,
            aux_y1_base=aux_y1_base_p,
            edgeflow_node_feats=edgeflow_pred_p,
            surfaceflow_node_feats=surfaceflow_pred_p,
            surfaceflow_slot_feats=surfaceflow_slot_pred_p,
            local2d_node_feats=local2d_node_pred_p,
            volagg_pred_1d=volagg_pred_p,
            edge_index_1d=eidx_pre,
            edge_deg_inv_1d=einv_pre,
            edge_weight_1d=ew_pre,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
    return y1_feat.astype(np.float32, copy=False)


def _build_aux_traj_views(
    *,
    cfg_like: dict,
    aux_baseline_ckpts: dict[str, dict],
    aux_pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]],
    graph2,
    y1_init: np.ndarray,
    y2_init: np.ndarray,
    rain_2d: np.ndarray,
    q1_init: np.ndarray | None,
    vagg_init: np.ndarray | None,
    q2edge_init: np.ndarray | None,
    qedge_init: np.ndarray | None,
    warmup: int,
    node_static_raw: torch.Tensor,
    node_static_aug1: torch.Tensor,
    node_static_aug2: torch.Tensor | None,
    masks: dict,
    y2_corr: np.ndarray,
    y2_corr_coupling: dict[str, np.ndarray],
    bed2: np.ndarray,
    conn_src: np.ndarray,
    conn_dst: np.ndarray,
    n1: int,
    knn_idx: np.ndarray | None,
    inlet_ckpts: dict[str, dict],
    edgeflow_ckpts: dict[str, dict],
    surfaceflow_ckpts: dict[str, dict],
    volagg_ckpts: dict[str, dict],
    surface_slot_edges: np.ndarray | None,
    surface_slot_sign: np.ndarray | None,
    local2d_center_cell: np.ndarray | None,
    local2d_neighbor_slots: np.ndarray | None,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    lap_w_dir: np.ndarray,
    diam_max_1d: np.ndarray,
    conn_area_1d: np.ndarray | None,
    vol2_agg_ev: np.ndarray | None,
    edgeflow_in_slots: np.ndarray | None,
    edgeflow_out_slots: np.ndarray | None,
    edge_index_1d_full: torch.LongTensor,
    edge_deg_inv_1d_full: torch.Tensor,
    edge_weight_1d_full: torch.Tensor,
    edge_index_1d_dir: torch.LongTensor,
    edge_deg_inv_1d_dir: torch.Tensor,
    edge_weight_1d_dir: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray | None:
    y1_aux = _build_aux_baseline_views(
        cfg_like=cfg_like,
        aux_baseline_ckpts=aux_baseline_ckpts,
        graph2=graph2,
        y1_init=y1_init,
        y2_init=y2_init,
        rain_2d=rain_2d,
        q1_init=q1_init,
        vagg_init=vagg_init,
        warmup=warmup,
    )
    if y1_aux is None:
        return None
    aux_pre_models = _ordered_pre_models(cfg_like=cfg_like, pre_model_map=aux_pre_model_map)
    if not aux_pre_models:
        return y1_aux
    return _rollout_pre_residual_stack(
        pre_models=aux_pre_models,
        y1_base_init=y1_aux,
        node_static_raw=node_static_raw,
        node_static_aug1=node_static_aug1,
        node_static_aug2=node_static_aug2,
        masks=masks,
        y2_corr=y2_corr,
        y2_corr_coupling=y2_corr_coupling,
        bed2=bed2,
        conn_src=conn_src,
        conn_dst=conn_dst,
        n1=n1,
        knn_idx=knn_idx,
        inlet_ckpts=inlet_ckpts,
        aux_baseline_ckpts=aux_baseline_ckpts,
        edgeflow_ckpts=edgeflow_ckpts,
        surfaceflow_ckpts=surfaceflow_ckpts,
        volagg_ckpts=volagg_ckpts,
        graph2=graph2,
        rain=rain_2d,
        y2_init=y2_init,
        inlet_1d_ev=q1_init,
        vol2_agg_ev=vol2_agg_ev,
        volagg_init_ev=vagg_init,
        surfaceflow_2d_ev=q2edge_init,
        edgeflow_1d_ev=qedge_init,
        warmup=warmup,
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
        edgeflow_in_slots=edgeflow_in_slots,
        edgeflow_out_slots=edgeflow_out_slots,
        edge_index_1d_full=edge_index_1d_full,
        edge_deg_inv_1d_full=edge_deg_inv_1d_full,
        edge_weight_1d_full=edge_weight_1d_full,
        edge_index_1d_dir=edge_index_1d_dir,
        edge_deg_inv_1d_dir=edge_deg_inv_1d_dir,
        edge_weight_1d_dir=edge_weight_1d_dir,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )


def _apply_resid1d_ckpt(
    *,
    ckpt: dict,
    model: ResidualNodeGRU,
    node_static: torch.Tensor,
    cfg: dict,
    node_mask: np.ndarray | None,
    expert_group_idx: np.ndarray | None,
    y1_feat: np.ndarray,  # [T,N1]
    y2_agg: np.ndarray,  # [T,N1]
    rain_2d: np.ndarray,  # [T,N2]
    warmup: int,
    inv1: np.ndarray,
    bed_agg: np.ndarray,
    has_conn: np.ndarray,
    nbr_src: np.ndarray | None,
    nbr_dst: np.ndarray | None,
    y2_knn_mean: np.ndarray | None,
    y2_knn_max: np.ndarray | None,
    lap_src: np.ndarray | None,
    lap_dst: np.ndarray | None,
    lap_w: np.ndarray | None,
    diam_max_1d: np.ndarray | None,
    base_area_1d: np.ndarray | None,
    conn_area_1d: np.ndarray | None,
    inlet_1d: np.ndarray | None,
    vol2_agg: np.ndarray | None,
    inlet_pred_1d: np.ndarray | None,
    aux_y1_base: np.ndarray | None,
    edgeflow_node_feats: np.ndarray | None,
    surfaceflow_node_feats: np.ndarray | None,
    surfaceflow_slot_feats: np.ndarray | None,
    local2d_node_feats: np.ndarray | None,
    volagg_pred_1d: np.ndarray | None,
    edge_index_1d: torch.LongTensor | None,
    edge_deg_inv_1d: torch.Tensor | None,
    edge_weight_1d: torch.Tensor | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    dyn_ver = int(cfg.get("dyn_feat_version", 1) or 1)
    use_out = bool(cfg.get("use_out_nbr_feats", False))
    rain_lags = int(cfg.get("rain_lags", 0))

    x = _build_dyn_1d(
        dyn_ver=dyn_ver,
        y1_base=y1_feat,
        y2_agg=y2_agg,
        rain_2d=rain_2d,
        warmup=warmup,
        inv1=inv1,
        bed_agg=bed_agg,
        has_conn=has_conn,
        nbr_src=nbr_src,
        nbr_dst=nbr_dst,
        y2_knn_mean=y2_knn_mean,
        y2_knn_max=y2_knn_max,
        lap_src=lap_src,
        lap_dst=lap_dst,
        lap_w=lap_w,
        diam_max_1d=diam_max_1d,
        base_area_1d=base_area_1d,
        conn_area_1d=conn_area_1d,
        inlet_1d=inlet_1d,
        vol2_agg=vol2_agg,
        inlet_pred_1d=inlet_pred_1d,
        aux_y1_base=aux_y1_base,
        edgeflow_node_feats=edgeflow_node_feats,
        surfaceflow_node_feats=surfaceflow_node_feats,
        surfaceflow_slot_feats=surfaceflow_slot_feats,
        local2d_node_feats=local2d_node_feats,
        volagg_pred_1d=volagg_pred_1d,
        use_out_nbr=use_out,
        rain_lags=rain_lags,
    )
    x = x.to(device, dtype=amp_dtype) if amp_enabled else x.to(device)
    with torch.no_grad():
        warm_ctx = _build_warm_ctx_1d(
            warm_ctx_ver=int(cfg.get("warm_ctx_version", 0) or 0),
            y1_base=y1_feat,
            y2_agg=y2_agg,
            warmup=warmup,
            inv1=inv1,
            bed_agg=bed_agg,
            has_conn=has_conn,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            edgeflow_node_feats=edgeflow_node_feats,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            local2d_node_feats=local2d_node_feats,
        )
        warm_seq = _build_warm_seq_1d(
            warm_seq_ver=int(cfg.get("warm_seq_version", 0) or 0),
            y1_base=y1_feat,
            y2_agg=y2_agg,
            warmup=warmup,
            inv1=inv1,
            bed_agg=bed_agg,
            has_conn=has_conn,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            inlet_1d=inlet_1d,
            vol2_agg=vol2_agg,
            edgeflow_node_feats=edgeflow_node_feats,
            surfaceflow_slot_feats=surfaceflow_slot_feats,
            local2d_node_feats=local2d_node_feats,
        )
        if warm_ctx is not None:
            warm_ctx = warm_ctx.to(device, dtype=amp_dtype) if amp_enabled else warm_ctx.to(device)
        if warm_seq is not None:
            warm_seq = warm_seq.to(device, dtype=amp_dtype) if amp_enabled else warm_seq.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            expert_group_idx_t = None if expert_group_idx is None else torch.from_numpy(expert_group_idx.astype(np.int64, copy=False)).to(device=x.device)
            r = (
                model(
                    x,
                    node_static,
                    warm_ctx=warm_ctx,
                    warm_seq=warm_seq,
                    expert_group_idx=expert_group_idx_t,
                    edge_index=edge_index_1d,
                    edge_deg_inv=edge_deg_inv_1d,
                    edge_weight=edge_weight_1d,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )
    if node_mask is not None:
        r = r * node_mask[None, :].astype(np.float32, copy=False)
    y1 = np.asarray(y1_feat, dtype=np.float32).copy()
    y1[warmup:] = y1[warmup:] + r
    return y1


def main() -> None:
    a = _parse_args()
    stage = int(a.stage)
    dflt = _stage_defaults(stage)
    epochs = int(a.epochs) if int(a.epochs) > 0 else 400
    lr = float(a.lr) if float(a.lr) > 0.0 else float(dflt["lr"])
    clamp = float(a.clamp) if a.clamp is not None else float(dflt["clamp"])
    clamp_mode = str(a.clamp_mode) if a.clamp_mode is not None else str(dflt["clamp_mode"])

    cfg = Residual1DTrainCfg(
        baseline_ckpts=tuple(str(x) for x in a.baseline_ckpt),
        mixed_mode=str(a.mixed_mode),
        alpha_1d=float(a.alpha_1d),
        alpha_2d=float(a.alpha_2d),
        resid2d_ckpt=str(a.resid2d_ckpt),
        resid2d_ckpt_coupling=str(a.resid2d_coupling_ckpt),
        resid2d_coupling_blend=float(a.resid2d_coupling_blend),
        resid2d_coupling_blend_knn=float(a.resid2d_coupling_blend_knn),
        inlet_ckpt=str(a.inlet_ckpt),
        aux_baseline_ckpt=str(a.aux_baseline_ckpt),
        aux_pre_residual_ckpts=tuple(str(x) for x in (a.aux_pre_residual_ckpt or [])),
        edgeflow_ckpt=str(a.edgeflow_ckpt),
        surfaceflow_ckpt=str(a.surfaceflow_ckpt),
        volagg_ckpt=str(a.volagg_ckpt),
        pre_residual_ckpts=tuple(str(x) for x in (a.pre_residual_ckpt or [])),
        model_root=str(a.model_root),
        cache_dir=str(a.cache_dir),
        out_path=str(a.out),
        seed=int(a.seed),
        stage=stage,
        epochs=epochs,
        lr=lr,
        weight_decay=float(a.weight_decay),
        grad_clip=float(a.grad_clip),
        ema_decay=float(a.ema_decay),
        node_emb_dim=int(a.node_emb_dim),
        hidden_dim=int(a.hidden_dim),
        num_layers=int(a.num_layers),
        dropout=float(dflt["dropout"]),
        bidirectional=bool(a.bidirectional),
        nodewise_head=bool(a.nodewise_head),
        expert_heads=int(a.expert_heads),
        expert_gate_hidden=int(a.expert_gate_hidden),
        expert_gate_dropout=float(a.expert_gate_dropout),
        expert_group_topo3=bool(a.expert_group_topo3),
        graph_mix_k=int(a.graph_mix_k),
        graph_mix_post_k=int(a.graph_mix_post_k),
        graph_mix_dropout=float(a.graph_mix_dropout),
        graph_mix_edge=str(a.graph_mix_edge),
        graph_mix_weighted=bool(a.graph_mix_weighted),
        clamp=clamp,
        clamp_mode=clamp_mode,
        zero_init_out=bool(a.zero_init_out),
        cache_on_gpu=bool(a.cache_on_gpu),
        amp_bf16=bool(a.amp_bf16),
        max_events=int(a.max_events),
        dyn_feat_version=int(a.dyn_feat_version) if int(a.dyn_feat_version) > 0 else int(dflt["dyn_feat_version"]),
        warm_ctx_version=int(a.warm_ctx_version),
        warm_seq_version=int(a.warm_seq_version),
        y2_knn_k=int(a.y2_knn_k) if int(a.y2_knn_k) >= 0 else int(dflt.get("y2_knn_k", 0)),
        rain_lags=int(a.rain_lags) if int(a.rain_lags) >= 0 else int(dflt.get("rain_lags", 0)),
        use_nbr_feats=True,
        use_out_nbr_feats=bool(a.use_out_nbr_feats) if a.use_out_nbr_feats is not None else bool(dflt.get("use_out_nbr_feats", False)),
        nbr_pipe_dir=bool(a.nbr_pipe_dir) if a.nbr_pipe_dir is not None else bool(dflt.get("nbr_pipe_dir", False)),
        use_static_aug=True,
        static_aug_version=int(a.static_aug_version) if int(a.static_aug_version) > 0 else int(dflt["static_aug_version"]),
        mask_base_area_zero=bool(a.mask_base_area_zero) if a.mask_base_area_zero is not None else bool(dflt.get("mask_base_area_zero", False)),
        mask_pipe_source=bool(a.mask_pipe_source) if a.mask_pipe_source is not None else bool(dflt.get("mask_pipe_source", False)),
        mask_pipe_indeg_ge=int(a.mask_pipe_indeg_ge) if int(a.mask_pipe_indeg_ge) >= 0 else int(dflt.get("mask_pipe_indeg_ge", 0)),
        mask_node_list=tuple(int(x) for x in (dflt.get("mask_node_list", ()) or ())),
        mask_mode=str(a.mask_mode) if str(a.mask_mode) else str(dflt.get("mask_mode", "or")),
        loss_node_weight_source=float(a.loss_node_weight_source),
        loss_node_weight_base_area_zero=float(a.loss_node_weight_base_area_zero),
        loss_node_weight_indeg_ge=int(a.loss_node_weight_indeg_ge),
        loss_node_weight_indeg_scale=float(a.loss_node_weight_indeg_scale),
        loss_time_weight_fill=float(a.loss_time_weight_fill),
        loss_time_fill_clip=float(a.loss_time_fill_clip),
        loss_time_weight_inlet_pos=float(a.loss_time_weight_inlet_pos),
        loss_time_inlet_pos_scale=float(a.loss_time_inlet_pos_scale),
    )

    seed_everything(cfg.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16
    amp_enabled = bool(cfg.amp_bf16)
    ema_decay = float(cfg.ema_decay)
    use_ema = 0.0 < ema_decay < 1.0
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("--ema-decay must be in [0, 1)")

    model_root = Path(cfg.model_root)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")

    # Baseline ckpts must share split/warmup.
    ckpts = [_load_torch(Path(p)) for p in cfg.baseline_ckpts]
    warmup = int(ckpts[0]["cfg"]["warmup"])
    split = ckpts[0].get("split", None)
    if split is None:
        raise ValueError("baseline ckpt missing split")
    for c in ckpts[1:]:
        if int(c["cfg"]["warmup"]) != warmup:
            raise ValueError("all baseline checkpoints must share warmup")
        if c.get("split", None) != split:
            raise ValueError("all baseline checkpoints must share split")

    train_ids = list(split["model_2"]["train"])
    val_ids = list(split["model_2"]["val"])
    if cfg.max_events > 0:
        train_ids = train_ids[: cfg.max_events]
        val_ids = val_ids[: cfg.max_events]

    graph2 = load_graph(model_root, model_id=2, split_for_static="train")
    n1 = int(graph2.n_1d)
    n2 = int(graph2.n_2d)

    # Raw static elevations.
    head_off = graph2.head_offset.cpu().numpy().astype(np.float32, copy=False)
    inv1 = head_off[:n1].astype(np.float32, copy=False)
    bed2 = head_off[n1:].astype(np.float32, copy=False)
    conn_src = graph2.conn_src_1d.cpu().numpy()
    conn_dst = graph2.conn_dst_2d.cpu().numpy()
    deg = np.bincount(conn_src, minlength=n1).astype(np.float32, copy=False)
    has_conn = (deg > 0).astype(np.float32, copy=False)
    bed_agg = np.zeros((n1,), dtype=np.float32)
    for s, d in zip(conn_src.tolist(), conn_dst.tolist()):
        bed_agg[s] += bed2[d]
    bed_agg = bed_agg / np.maximum(deg, 1.0)
    conn_area_1d = aggregate_2d_to_1d_sum(
        graph2.area_2d.cpu().numpy().astype(np.float32, copy=False)[None, :],
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        n_1d=n1,
    )[0].astype(np.float32, copy=False)

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

    # 1D edge_index options for graph mixing:
    # - full: undirected (both directions)
    # - dir:  pipe-direction only (first half)
    edge_index_full = graph2.edge_index_1d.to(device)
    E1 = int(graph2.edge_index_1d.shape[1] // 2)
    edge_index_dir = edge_index_full[:, :E1]

    deg_full = torch.bincount(graph2.edge_index_1d[1].cpu(), minlength=int(n1)).clamp(min=1).to(torch.float32)
    edge_deg_inv_full = (1.0 / deg_full).to(device)
    deg_dir = torch.bincount(graph2.edge_index_1d[1, :E1].cpu(), minlength=int(n1)).clamp(min=1).to(torch.float32)
    edge_deg_inv_dir = (1.0 / deg_dir).to(device)

    # Optional weighted neighbor mean in graph mixing (pipe conductance proxy).
    need_weighted = bool(cfg.graph_mix_weighted)
    need_dir = str(cfg.graph_mix_edge) == "dir"

    # KNN mapping if needed by this stage or any pre-residual stage.
    need_knn_k = int(cfg.y2_knn_k)
    for p in cfg.pre_residual_ckpts:
        pre = _load_torch(Path(p))
        pre_cfg = pre.get("cfg", {}) or {}
        need_weighted = need_weighted or bool(pre_cfg.get("graph_mix_weighted", False))
        need_dir = need_dir or str(pre_cfg.get("graph_mix_edge", "full")) == "dir"
        dv = int(pre_cfg.get("dyn_feat_version", 1) or 1)
        if dv in (3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
            need_knn_k = max(need_knn_k, int(pre_cfg.get("y2_knn_k", 0) or 0))
        if int(pre_cfg.get("warm_ctx_version", 0) or 0) in (1, 2) or int(pre_cfg.get("warm_seq_version", 0) or 0) in (1,):
            need_knn_k = max(need_knn_k, int(pre_cfg.get("y2_knn_k", 0) or 0))
    if int(cfg.warm_ctx_version) in (1, 2) or int(cfg.warm_seq_version) in (1,):
        need_knn_k = max(need_knn_k, int(cfg.y2_knn_k))
    knn_idx = _knn_2d_idx(model_root=model_root, k=int(need_knn_k), graph2=graph2) if int(need_knn_k) > 0 else None

    edge_weight_full = None
    edge_deg_inv_full_w = None
    edge_weight_dir = None
    edge_deg_inv_dir_w = None
    if need_weighted:
        # lap_w_dir is for pipe-direction edges only (length E1). Duplicate for the reverse half.
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

    edge_index_1d_full = edge_index_full
    edge_deg_inv_1d_full = edge_deg_inv_full_w if edge_deg_inv_full_w is not None else edge_deg_inv_full
    edge_weight_1d_full = edge_weight_full
    edge_index_1d_dir = edge_index_dir
    edge_deg_inv_1d_dir = edge_deg_inv_dir_w if edge_deg_inv_dir_w is not None else edge_deg_inv_dir
    edge_weight_1d_dir = edge_weight_dir

    def _pick_edges(cfg_like: dict) -> tuple[torch.LongTensor | None, torch.Tensor | None, torch.Tensor | None]:
        k_pre = int(cfg_like.get("graph_mix_k", 0) or 0)
        k_post = int(cfg_like.get("graph_mix_post_k", 0) or 0)
        if k_pre <= 0 and k_post <= 0:
            return None, None, None
        use_dir = str(cfg_like.get("graph_mix_edge", "full")) == "dir"
        use_w = bool(cfg_like.get("graph_mix_weighted", False))
        if use_dir:
            if use_w:
                return edge_index_dir, edge_deg_inv_dir_w, edge_weight_dir
            return edge_index_dir, edge_deg_inv_dir, None
        if use_w:
            return edge_index_full, edge_deg_inv_full_w, edge_weight_full
        return edge_index_full, edge_deg_inv_full, None

    edge_index_1d, edge_deg_inv_1d, edge_weight_1d = _pick_edges(asdict(cfg))

    # Node-static options (computed once).
    node_static_raw = graph2.node_static_1d.float().to(device)
    node_static_aug1 = build_static_features_m2_1d(graph2).to(device)
    node_static_aug2 = build_static_features_m2_1d_v2(graph2).to(device)

    # Load 2D residual (used to build coupling features).
    ck2d = _load_torch(Path(cfg.resid2d_ckpt))
    if str(ck2d.get("kind", "")) != "residual_m2_2d":
        raise ValueError("resid2d-ckpt must be kind residual_m2_2d")
    m2d_dyn_feat = str((ck2d.get("cfg", {}) or {}).get("dyn_feat", ""))
    m2d_cfg = ResidualGRUConfig(**ck2d["model_cfg"])
    m2d = ResidualNodeGRU(n_nodes=n2, cfg=m2d_cfg).to(device)
    m2d.load_state_dict(ck2d["state_dict"])
    m2d.eval()
    node_static_2d = graph2.node_static_2d.float().to(device)
    edge_index_2d = None
    edge_deg_inv_2d = None
    need_edges_2d = (
        (int(getattr(m2d_cfg, "graph_mix_k", 0) or 0) > 0)
        or (int(getattr(m2d_cfg, "graph_mix_post_k", 0) or 0) > 0)
        or (int(getattr(m2d_cfg, "dyn_dim", 5)) in (7, 9))
    )
    if need_edges_2d:
        edge_index_2d = graph2.edge_index_2d.to(device)
        deg = torch.bincount(graph2.edge_index_2d[1].cpu(), minlength=int(n2)).clamp(min=1).to(torch.float32)
        edge_deg_inv_2d = (1.0 / deg).to(device)

    # Load pre-1D residual models (applied sequentially to the baseline).
    pre_models: list[tuple[dict, ResidualNodeGRU, dict]] = []
    for p in cfg.pre_residual_ckpts:
        pre = _load_torch(Path(p))
        if str(pre.get("kind", "")) != "residual_m2_1d":
            raise ValueError(f"pre-residual kind unsupported: {pre.get('kind')}")
        pcfg = pre.get("cfg", {}) or {}
        mcfg = ResidualGRUConfig(**pre["model_cfg"])
        m = ResidualNodeGRU(n_nodes=n1, cfg=mcfg).to(device)
        m.load_state_dict(pre["state_dict"])
        m.eval()
        pre_models.append((pre, m, pcfg))

    coupling_resid2d_models: dict[str, tuple[ResidualNodeGRU, str]] = {}
    coupling_paths: set[str] = set()

    def _maybe_add_coupling_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("resid2d_ckpt_coupling", "") or "")
        if not path:
            return
        w_agg = float(cfg_like.get("resid2d_coupling_blend", 0.0) or 0.0)
        w_knn = float(cfg_like.get("resid2d_coupling_blend_knn", 0.0) or 0.0)
        if w_agg > 0.0 or w_knn > 0.0:
            coupling_paths.add(path)

    _maybe_add_coupling_path(asdict(cfg))
    for _, _, pcfg in pre_models:
        _maybe_add_coupling_path(pcfg)

    for path in sorted(coupling_paths):
        ck = _load_torch(Path(path))
        if str(ck.get("kind", "")) != "residual_m2_2d":
            raise ValueError(f"resid2d-coupling-ckpt must be kind residual_m2_2d: {path}")
        dyn_feat_cpl = str((ck.get("cfg", {}) or {}).get("dyn_feat", ""))
        mcfg = ResidualGRUConfig(**ck["model_cfg"])
        m = ResidualNodeGRU(n_nodes=n2, cfg=mcfg).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        coupling_resid2d_models[path] = (m, dyn_feat_cpl)

    inlet_ckpt_paths: set[str] = set()
    aux_baseline_ckpt_paths: set[str] = set()
    aux_pre_residual_ckpt_paths: set[str] = set()
    edgeflow_ckpt_paths: set[str] = set()
    surfaceflow_ckpt_paths: set[str] = set()
    volagg_ckpt_paths: set[str] = set()

    def _maybe_add_inlet_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("inlet_ckpt", "") or "")
        if path:
            inlet_ckpt_paths.add(path)

    def _maybe_add_aux_baseline_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("aux_baseline_ckpt", "") or "")
        if path:
            aux_baseline_ckpt_paths.add(path)

    def _maybe_add_aux_pre_residual_paths(cfg_like: dict) -> None:
        for path in tuple(str(x) for x in (cfg_like.get("aux_pre_residual_ckpts", ()) or ())):
            if path:
                aux_pre_residual_ckpt_paths.add(path)

    def _maybe_add_edgeflow_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("edgeflow_ckpt", "") or "")
        if path:
            edgeflow_ckpt_paths.add(path)

    def _maybe_add_surfaceflow_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("surfaceflow_ckpt", "") or "")
        if path:
            surfaceflow_ckpt_paths.add(path)

    def _maybe_add_volagg_path(cfg_like: dict) -> None:
        path = str(cfg_like.get("volagg_ckpt", "") or "")
        if path:
            volagg_ckpt_paths.add(path)

    _maybe_add_inlet_path(asdict(cfg))
    _maybe_add_aux_baseline_path(asdict(cfg))
    _maybe_add_aux_pre_residual_paths(asdict(cfg))
    _maybe_add_edgeflow_path(asdict(cfg))
    _maybe_add_surfaceflow_path(asdict(cfg))
    _maybe_add_volagg_path(asdict(cfg))
    for _, _, pcfg in pre_models:
        _maybe_add_inlet_path(pcfg)
        _maybe_add_aux_baseline_path(pcfg)
        _maybe_add_aux_pre_residual_paths(pcfg)
        _maybe_add_edgeflow_path(pcfg)
        _maybe_add_surfaceflow_path(pcfg)
        _maybe_add_volagg_path(pcfg)

    inlet_ckpts: dict[str, dict] = {}
    for path in sorted(inlet_ckpt_paths):
        ck = _load_torch(Path(path))
        kind = str((ck.get("model_2", {}) or {}).get("kind", ""))
        if kind not in {"split_1d2d_coupled_inlet", "split_1d2d_coupled_storage_inlet", "aux_inlet_m2_1d_surface_slots"}:
            raise ValueError(
                f"inlet-ckpt must be model_2 kind split_1d2d_coupled_inlet, "
                f"split_1d2d_coupled_storage_inlet, or aux_inlet_m2_1d_surface_slots: {path}"
            )
        inlet_ckpts[path] = ck

    aux_baseline_ckpts: dict[str, dict] = {}
    for path in sorted(aux_baseline_ckpt_paths):
        ck = _load_torch(Path(path))
        if str(ck.get("kind", "")) != "baseline_arx":
            raise ValueError(f"aux-baseline-ckpt must be kind baseline_arx: {path}")
        aux_baseline_ckpts[path] = ck

    aux_pre_model_map: dict[str, tuple[dict, ResidualNodeGRU, dict]] = {}
    for path in sorted(aux_pre_residual_ckpt_paths):
        pre = _load_torch(Path(path))
        if str(pre.get("kind", "")) != "residual_m2_1d":
            raise ValueError(f"aux-pre-residual-ckpt must be kind residual_m2_1d: {path}")
        pcfg = pre.get("cfg", {}) or {}
        mcfg = ResidualGRUConfig(**pre["model_cfg"])
        m = ResidualNodeGRU(n_nodes=n1, cfg=mcfg).to(device)
        m.load_state_dict(pre["state_dict"])
        m.eval()
        aux_pre_model_map[path] = (pre, m, pcfg)

    edgeflow_ckpts: dict[str, dict] = {}
    for path in sorted(edgeflow_ckpt_paths):
        ck = _load_torch(Path(path))
        if str(ck.get("kind", "")) != "aux_edgeflow_m2_1d":
            raise ValueError(f"edgeflow-ckpt must be kind aux_edgeflow_m2_1d: {path}")
        edgeflow_ckpts[path] = ck

    surfaceflow_ckpts: dict[str, dict] = {}
    for path in sorted(surfaceflow_ckpt_paths):
        ck = _load_torch(Path(path))
        if str(ck.get("kind", "")) != "aux_surfaceflow_m2_2d":
            raise ValueError(f"surfaceflow-ckpt must be kind aux_surfaceflow_m2_2d: {path}")
        surfaceflow_ckpts[path] = ck

    volagg_ckpts: dict[str, dict] = {}
    for path in sorted(volagg_ckpt_paths):
        ck = _load_torch(Path(path))
        if str(ck.get("kind", "")) != "aux_volagg_m2_1d":
            raise ValueError(f"volagg-ckpt must be kind aux_volagg_m2_1d: {path}")
        volagg_ckpts[path] = ck

    # Current-stage mask (for loss and inference-time application).
    mask_curr = _mask_from_cfg(asdict(cfg), masks=masks, n1=n1)
    mask_curr_gpu = None if mask_curr is None else torch.from_numpy(mask_curr.astype(np.bool_, copy=False)).to(device)
    expert_group_idx = None
    expert_group_idx_gpu = None
    if bool(cfg.expert_group_topo3):
        if int(cfg.expert_heads) != 3:
            raise ValueError("--expert-group-topo3 requires --expert-heads 3")
        expert_group_idx = _expert_group_idx_topo3(masks=masks, n1=n1)
        expert_group_idx_gpu = torch.from_numpy(expert_group_idx).to(device=device, dtype=torch.long)

    loss_node_weight = np.ones((n1,), dtype=np.float32)
    if float(cfg.loss_node_weight_source) > 0.0:
        loss_node_weight += float(cfg.loss_node_weight_source) * np.asarray(masks["mask_source"], dtype=np.float32)
    if float(cfg.loss_node_weight_base_area_zero) > 0.0 and masks.get("mask_ba0", None) is not None:
        loss_node_weight += float(cfg.loss_node_weight_base_area_zero) * np.asarray(masks["mask_ba0"], dtype=np.float32)
    if float(cfg.loss_node_weight_indeg_scale) > 0.0 and int(cfg.loss_node_weight_indeg_ge) > 0:
        loss_node_weight += float(cfg.loss_node_weight_indeg_scale) * (
            np.asarray(masks["indeg"], dtype=np.int64) >= int(cfg.loss_node_weight_indeg_ge)
        ).astype(np.float32, copy=False)
    use_loss_node_weight = bool(np.max(np.abs(loss_node_weight - 1.0)) > 1e-8)
    loss_node_weight_gpu = torch.from_numpy(loss_node_weight).to(device) if use_loss_node_weight else None

    depth_1d = masks.get("depth", None)
    if depth_1d is None:
        depth_1d = np.ones((n1,), dtype=np.float32)
    depth_1d = np.maximum(np.asarray(depth_1d, dtype=np.float32), 1e-3)

    std_1d = float(STD_DEV_DICT[(2, 1)])
    std_2d = float(STD_DEV_DICT[(2, 2)])
    eps = 1e-12

    def build_example(event_id: int) -> dict:
        ev = load_event(model_root, graph=graph2, split="train", event_id=event_id, cache_dir=cache_dir)
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
                event_id=event_id,
                n_edges=int(graph2.edge_index_1d.shape[1] // 2),
                cache_dir=cache_dir,
            )
        surfaceflow_2d_ev = None
        if surfaceflow_ckpts:
            surfaceflow_2d_ev = load_edge_flow_2d(
                model_root=model_root,
                model_id=2,
                split="train",
                event_id=event_id,
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
            ckpts,
            graph2=graph2,
            mixed_mode=str(cfg.mixed_mode),
            alpha_1d=float(cfg.alpha_1d),
            alpha_2d=float(cfg.alpha_2d),
            y1_init=y1_true,
            y2_init=y2_true,
            rain_2d=rain,
            q1_init=(ev.inlet_1d.numpy().astype(np.float32, copy=False) if ev.inlet_1d is not None else None),
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )

        # Correct 2D baseline first, then build 1D coupling features from corrected 2D.
        y2_corr = _apply_resid2d(
            resid2d=m2d,
            node_static_2d=node_static_2d,
            y2_base=y2_base,
            rain_2d=rain,
            bed2=bed2,
            warmup=warmup,
            dyn_feat=m2d_dyn_feat,
            y1_base=y1_base,
            conn_src_1d=conn_src,
            conn_dst_2d=conn_dst,
            n_2d=n2,
            edge_index_2d=edge_index_2d,
            edge_deg_inv_2d=edge_deg_inv_2d,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        # Cache the 2D score for this event (it does not depend on the current 1D stage weights).
        err2 = (y2_corr[warmup:] - y2_true[warmup:]) / std_2d
        score2d = float(np.mean(np.sqrt(np.mean(err2**2, axis=0) + eps)))

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
                edge_index_2d=edge_index_2d,
                edge_deg_inv_2d=edge_deg_inv_2d,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        # Apply stacked pre-residuals to build the correct base for this stage.
        y1_feat = _rollout_pre_residual_stack(
            pre_models=pre_models,
            y1_base_init=y1_base,
            node_static_raw=node_static_raw,
            node_static_aug1=node_static_aug1,
            node_static_aug2=node_static_aug2,
            masks=masks,
            y2_corr=y2_corr,
            y2_corr_coupling=y2_corr_coupling,
            bed2=bed2,
            conn_src=conn_src,
            conn_dst=conn_dst,
            n1=n1,
            knn_idx=knn_idx,
            inlet_ckpts=inlet_ckpts,
            aux_baseline_ckpts=aux_baseline_ckpts,
            edgeflow_ckpts=edgeflow_ckpts,
            surfaceflow_ckpts=surfaceflow_ckpts,
            volagg_ckpts=volagg_ckpts,
            graph2=graph2,
            rain=rain,
            y2_init=y2_true,
            inlet_1d_ev=inlet_1d_ev,
            vol2_agg_ev=vol2_agg_ev,
            volagg_init_ev=volagg_init_ev,
            surfaceflow_2d_ev=surfaceflow_2d_ev,
            edgeflow_1d_ev=edgeflow_1d_ev,
            warmup=warmup,
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
            edgeflow_in_slots=edgeflow_in_slots,
            edgeflow_out_slots=edgeflow_out_slots,
            edge_index_1d_full=edge_index_1d_full,
            edge_deg_inv_1d_full=edge_deg_inv_1d_full,
            edge_weight_1d_full=edge_weight_1d_full,
            edge_index_1d_dir=edge_index_1d_dir,
            edge_deg_inv_1d_dir=edge_deg_inv_1d_dir,
            edge_weight_1d_dir=edge_weight_1d_dir,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        # Build this stage's features + targets.
        use_nbr = bool(cfg.use_nbr_feats) or bool(cfg.use_out_nbr_feats)
        if use_nbr:
            src = masks["src_dir"] if bool(cfg.nbr_pipe_dir) else masks["src_full"]
            dst = masks["dst_dir"] if bool(cfg.nbr_pipe_dir) else masks["dst_full"]
        else:
            src = None
            dst = None

        coupling_path = str(cfg.resid2d_ckpt_coupling or "")
        y2_agg, y2_knn_mean, y2_knn_max = _build_coupling_views(
            cfg_like=asdict(cfg),
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            bed2=bed2,
            conn_src=conn_src,
            conn_dst=conn_dst,
            n_1d=n1,
            knn_idx=knn_idx,
        )
        inlet_pred = _build_inlet_pred_views(
            cfg_like=asdict(cfg),
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
            cfg_like=asdict(cfg),
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
            node_static_raw=node_static_raw,
            node_static_aug1=node_static_aug1,
            node_static_aug2=node_static_aug2,
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
            edge_index_1d_full=edge_index_1d_full,
            edge_deg_inv_1d_full=edge_deg_inv_1d_full,
            edge_weight_1d_full=edge_weight_1d_full,
            edge_index_1d_dir=edge_index_1d_dir,
            edge_deg_inv_1d_dir=edge_deg_inv_1d_dir,
            edge_weight_1d_dir=edge_weight_1d_dir,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        if int(cfg.dyn_feat_version) == 17:
            edgeflow_pred = _build_edgeflow_pred_slot_views(
                cfg_like=asdict(cfg),
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
                cfg_like=asdict(cfg),
                edgeflow_ckpts=edgeflow_ckpts,
                graph2=graph2,
                y1_1d=y1_feat,
                rain_2d=rain,
                qedge_init=edgeflow_1d_ev,
                warmup=warmup,
            )
        volagg_pred = _build_volagg_pred_views(
            cfg_like=asdict(cfg),
            volagg_ckpts=volagg_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            vagg_init=volagg_init_ev,
            warmup=warmup,
        )
        surfaceflow_pred = _build_surfaceflow_pred_views(
            cfg_like=asdict(cfg),
            surfaceflow_ckpts=surfaceflow_ckpts,
            graph2=graph2,
            y2_main=y2_corr,
            y2_alt=y2_corr_coupling.get(coupling_path),
            rain_2d=rain,
            q2edge_init=surfaceflow_2d_ev,
            warmup=warmup,
        )
        if int(cfg.dyn_feat_version) in (19, 20, 21, 22):
            surfaceflow_slot_pred = _build_surfaceflow_pred_slot_views(
                cfg_like=asdict(cfg),
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
        if int(cfg.dyn_feat_version) == 21:
            blend_local = float(cfg.resid2d_coupling_blend or 0.0)
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

        x_dyn = _build_dyn_1d(
            dyn_ver=int(cfg.dyn_feat_version),
            y1_base=y1_feat,
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
            use_out_nbr=bool(cfg.use_out_nbr_feats),
            rain_lags=int(cfg.rain_lags),
        )
        warm_ctx = _build_warm_ctx_1d(
            warm_ctx_ver=int(cfg.warm_ctx_version),
            y1_base=y1_feat,
            y2_agg=y2_agg,
            warmup=warmup,
            inv1=inv1,
            bed_agg=bed_agg,
            has_conn=has_conn,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            inlet_1d=inlet_1d_ev,
            vol2_agg=vol2_agg_ev,
            edgeflow_node_feats=edgeflow_pred,
            surfaceflow_slot_feats=surfaceflow_slot_pred,
            local2d_node_feats=local2d_node_pred,
        )
        warm_seq = _build_warm_seq_1d(
            warm_seq_ver=int(cfg.warm_seq_version),
            y1_base=y1_feat,
            y2_agg=y2_agg,
            warmup=warmup,
            inv1=inv1,
            bed_agg=bed_agg,
            has_conn=has_conn,
            y2_knn_mean=y2_knn_mean,
            y2_knn_max=y2_knn_max,
            inlet_1d=inlet_1d_ev,
            vol2_agg=vol2_agg_ev,
            edgeflow_node_feats=edgeflow_pred,
            surfaceflow_slot_feats=surfaceflow_slot_pred,
            local2d_node_feats=local2d_node_pred,
        )

        base_tail = y1_feat[warmup:].astype(np.float32, copy=False)
        y_res = (y1_true[warmup:] - base_tail).astype(np.float32, copy=False)
        loss_w = None
        if float(cfg.loss_time_weight_fill) > 0.0 or float(cfg.loss_time_weight_inlet_pos) > 0.0:
            loss_w_np = np.ones_like(base_tail, dtype=np.float32)
            if float(cfg.loss_time_weight_fill) > 0.0:
                fill_clip = max(float(cfg.loss_time_fill_clip), 1e-6)
                fill = (y1_true[warmup:] - inv1[None, :]) / depth_1d[None, :]
                fill = np.clip(fill, 0.0, fill_clip) / fill_clip
                loss_w_np += float(cfg.loss_time_weight_fill) * fill.astype(np.float32, copy=False)
            if float(cfg.loss_time_weight_inlet_pos) > 0.0 and inlet_1d_ev is not None:
                q_scale = max(float(cfg.loss_time_inlet_pos_scale), 1e-6)
                q_pos = np.clip(inlet_1d_ev[warmup:] / q_scale, 0.0, 2.0)
                loss_w_np += float(cfg.loss_time_weight_inlet_pos) * q_pos.astype(np.float32, copy=False)
            loss_w = torch.from_numpy(loss_w_np)
        return {
            "event_id": int(event_id),
            "x_dyn": x_dyn,
            "warm_ctx": warm_ctx,
            "warm_seq": warm_seq,
            "y_res": torch.from_numpy(y_res),
            "base_tail": base_tail,
            "loss_w": loss_w,
            "score2d": float(score2d),
        }

    t_prep0 = time.time()
    train_cache = {eid: build_example(eid) for eid in tqdm(train_ids, desc="prep train (m2-1d)", leave=False)}
    val_cache = {eid: build_example(eid) for eid in tqdm(val_ids, desc="prep val (m2-1d)", leave=False)}
    print(f"prep: cached {len(train_cache)} train + {len(val_cache)} val events in {time.time()-t_prep0:.1f}s")
    val_2d_const = float(np.mean([ex["score2d"] for ex in val_cache.values()])) if val_cache else float("nan")

    # Cache train tensors on GPU (optional).
    if bool(cfg.cache_on_gpu):
        for ex in train_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(
                device, non_blocking=True
            )
            wc = ex.get("warm_ctx", None)
            if wc is not None:
                ex["warm_ctx_gpu"] = wc.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else wc.to(
                    device, non_blocking=True
                )
            ws = ex.get("warm_seq", None)
            if ws is not None:
                ex["warm_seq_gpu"] = ws.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else ws.to(
                    device, non_blocking=True
                )
            ex["y_gpu"] = ex["y_res"].to(device, non_blocking=True)
            lw = ex.get("loss_w", None)
            if lw is not None:
                ex["loss_w_gpu"] = lw.to(device, non_blocking=True)
        for ex in val_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(
                device, non_blocking=True
            )
            wc = ex.get("warm_ctx", None)
            if wc is not None:
                ex["warm_ctx_gpu"] = wc.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else wc.to(
                    device, non_blocking=True
                )
            ws = ex.get("warm_seq", None)
            if ws is not None:
                ex["warm_seq_gpu"] = ws.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else ws.to(
                    device, non_blocking=True
                )
        print("cached train features on GPU")

    # Model for this stage.
    node_static_train = node_static_aug2 if int(cfg.static_aug_version) == 2 else node_static_aug1
    mcfg = ResidualGRUConfig(
        dyn_dim=int(train_cache[train_ids[0]]["x_dyn"].shape[-1]) if train_ids else int(cfg.dyn_feat_version),
        static_dim=int(node_static_train.shape[1]),
        warm_ctx_dim=int(train_cache[train_ids[0]]["warm_ctx"].shape[-1]) if (train_ids and train_cache[train_ids[0]].get("warm_ctx", None) is not None) else 0,
        warm_seq_dim=int(train_cache[train_ids[0]]["warm_seq"].shape[-1]) if (train_ids and train_cache[train_ids[0]].get("warm_seq", None) is not None) else 0,
        node_emb_dim=int(cfg.node_emb_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        bidirectional=bool(cfg.bidirectional),
        nodewise_head=bool(cfg.nodewise_head),
        expert_heads=int(cfg.expert_heads),
        expert_gate_hidden=int(cfg.expert_gate_hidden),
        expert_gate_dropout=float(cfg.expert_gate_dropout),
        graph_mix_k=int(cfg.graph_mix_k),
        graph_mix_post_k=int(cfg.graph_mix_post_k),
        graph_mix_dropout=float(cfg.graph_mix_dropout),
        clamp=float(cfg.clamp),
        clamp_mode=str(cfg.clamp_mode),
        zero_init_out=bool(cfg.zero_init_out),
    )
    model = ResidualNodeGRU(n_nodes=n1, cfg=mcfg).to(device)
    ema_model = None
    if use_ema:
        ema_model = ResidualNodeGRU(n_nodes=n1, cfg=mcfg).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    opt = AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    def save_ckpt(*, state_dict: dict[str, torch.Tensor], best_val_m2: float, best_epoch: int) -> None:
        payload = {
            "kind": "residual_m2_1d",
            "baseline_ckpts": [str(x) for x in cfg.baseline_ckpts],
            "cfg": asdict(cfg),
            "model_cfg": asdict(mcfg),
            "feat_base_dim": int(mcfg.dyn_dim),
            "warmup": int(warmup),
            "best_val_m2": float(best_val_m2),
            "best_epoch": int(best_epoch),
            "state_dict": state_dict,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        meta_path.write_text(json.dumps({"best_val_m2": best_val_m2, "best_epoch": best_epoch, "cfg": payload["cfg"]}, indent=2) + "\n")

    rng = np.random.default_rng(int(cfg.seed))
    for epoch in range(int(cfg.epochs)):
        model.train()
        ids = list(train_cache.keys())
        rng.shuffle(ids)
        losses = []
        t1 = time.time()
        for eid in ids:
            ex = train_cache[eid]
            if bool(cfg.cache_on_gpu):
                x_dyn = ex["x_gpu"]
                warm_ctx = ex.get("warm_ctx_gpu", None)
                warm_seq = ex.get("warm_seq_gpu", None)
                y_res = ex["y_gpu"]
                loss_w = ex.get("loss_w_gpu", None)
            else:
                x = ex["x_dyn"]
                x_dyn = x.to(device, dtype=amp_dtype) if amp_enabled else x.to(device)
                wc = ex.get("warm_ctx", None)
                warm_ctx = (wc.to(device, dtype=amp_dtype) if amp_enabled else wc.to(device)) if wc is not None else None
                ws = ex.get("warm_seq", None)
                warm_seq = (ws.to(device, dtype=amp_dtype) if amp_enabled else ws.to(device)) if ws is not None else None
                y_res = ex["y_res"].to(device)
                lw = ex.get("loss_w", None)
                loss_w = lw.to(device) if lw is not None else None

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred = model(
                    x_dyn,
                    node_static_train,
                    warm_ctx=warm_ctx,
                    warm_seq=warm_seq,
                    expert_group_idx=expert_group_idx_gpu,
                    edge_index=edge_index_1d,
                    edge_deg_inv=edge_deg_inv_1d,
                    edge_weight=edge_weight_1d,
                )  # [T', N1]
            err = (pred.float() - y_res) / std_1d
            if loss_w is not None:
                mse_node = torch.sum((err**2) * loss_w, dim=0) / torch.clamp(torch.sum(loss_w, dim=0), min=eps)
            else:
                mse_node = torch.mean(err**2, dim=0)
            rmse_node = torch.sqrt(mse_node + eps)
            if mask_curr_gpu is not None:
                rmse_use = rmse_node[mask_curr_gpu]
                node_w_use = loss_node_weight_gpu[mask_curr_gpu] if loss_node_weight_gpu is not None else None
            else:
                rmse_use = rmse_node
                node_w_use = loss_node_weight_gpu
            if node_w_use is not None:
                loss = torch.sum(rmse_use * node_w_use) / torch.clamp(torch.sum(node_w_use), min=eps)
            else:
                loss = rmse_use.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.grad_clip) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=True):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        if val_cache:
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            vals_1d = []
            vals_m2 = []
            with torch.no_grad():
                for ex in val_cache.values():
                    x_dyn = ex["x_gpu"] if bool(cfg.cache_on_gpu) else ex["x_dyn"].to(device)
                    warm_ctx = ex.get("warm_ctx_gpu", None) if bool(cfg.cache_on_gpu) else ex.get("warm_ctx", None)
                    warm_seq = ex.get("warm_seq_gpu", None) if bool(cfg.cache_on_gpu) else ex.get("warm_seq", None)
                    if warm_ctx is not None and (not bool(cfg.cache_on_gpu)):
                        warm_ctx = warm_ctx.to(device, dtype=amp_dtype) if amp_enabled else warm_ctx.to(device)
                    if warm_seq is not None and (not bool(cfg.cache_on_gpu)):
                        warm_seq = warm_seq.to(device, dtype=amp_dtype) if amp_enabled else warm_seq.to(device)
                    base_tail = ex["base_tail"]
                    y_res_np = ex["y_res"].numpy()
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        pred = (
                            eval_model(
                                x_dyn,
                                node_static_train,
                                warm_ctx=warm_ctx,
                                warm_seq=warm_seq,
                                expert_group_idx=expert_group_idx_gpu,
                                edge_index=edge_index_1d,
                                edge_deg_inv=edge_deg_inv_1d,
                                edge_weight=edge_weight_1d,
                            )
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                    if mask_curr is not None:
                        pred = pred * mask_curr[None, :].astype(np.float32, copy=False)
                    y_pred = base_tail + pred
                    y_true = base_tail + y_res_np
                    err = (y_pred - y_true) / std_1d
                    rmse1d = float(np.mean(np.sqrt(np.mean(err**2, axis=0) + eps)))
                    vals_1d.append(rmse1d)
                    vals_m2.append(0.5 * (rmse1d + float(ex["score2d"])))
            val_1d = float(np.mean(vals_1d)) if vals_1d else float("nan")
            val_m2 = float(np.mean(vals_m2)) if vals_m2 else float("nan")
            dt = time.time() - t1
            if ema_model is not None:
                print(
                    f"epoch {epoch+1:03d}/{cfg.epochs} "
                    f"train={train_loss:.6f} "
                    f"val_m2_ema={val_m2:.6f} "
                    f"val_1d_ema={val_1d:.6f} "
                    f"val_2d={val_2d_const:.6f} "
                    f"dt={dt:.1f}s"
                )
            else:
                print(
                    f"epoch {epoch+1:03d}/{cfg.epochs} "
                    f"train={train_loss:.6f} "
                    f"val_m2={val_m2:.6f} "
                    f"val_1d={val_1d:.6f} "
                    f"val_2d={val_2d_const:.6f} "
                    f"dt={dt:.1f}s"
                )
            if np.isfinite(val_m2) and val_m2 < best_val:
                best_val = float(val_m2)
                best_epoch = int(epoch + 1)
                src = ema_model if ema_model is not None else model
                best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
                save_ckpt(state_dict=best_state, best_val_m2=best_val, best_epoch=best_epoch)
        else:
            dt = time.time() - t1
            print(f"epoch {epoch+1:03d}/{cfg.epochs} train={train_loss:.6f} dt={dt:.1f}s")

    if best_state is None:
        src = ema_model if ema_model is not None else model
        best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
        best_epoch = int(cfg.epochs)
        best_val = float("nan")
    save_ckpt(state_dict=best_state, best_val_m2=best_val, best_epoch=best_epoch)
    print(f"saved 1D residual stage{cfg.stage} to {out_path}")


if __name__ == "__main__":
    main()
