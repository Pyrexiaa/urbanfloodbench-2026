from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from urbanflood.baseline import build_coupled_1d_exo, predict_model2_from_baseline_ckpts
from urbanflood.data import load_event, load_graph, list_events, split_event_ids
from urbanflood.inlet_aux import InletAuxCfg, build_inlet_local_surface_exo, build_payload, fit_inlet_aux, predict_inlet_from_ckpt
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_train import _apply_resid2d
from urbanflood.surfaceflow_aux import (
    build_coupled_neighbor_node_features,
    build_coupled_neighbor_node_index,
    build_coupled_surface_slot_features,
    build_coupled_surface_slot_index,
    load_edge_flow_2d,
    load_surface_edge_weight,
    predict_surface_flow_from_ckpt,
)
from urbanflood.utils import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=int, default=2, choices=[2])
    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, default="runs/inlet_aux_m2_surface_slots.pt")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-from", type=str, default="")
    p.add_argument("--baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument("--mixed-mode", type=str, default="weighted_split_ns", choices=["single", "weighted_split_ns"])
    p.add_argument("--alpha-1d", type=float, default=0.9)
    p.add_argument("--alpha-2d", type=float, default=0.5)
    p.add_argument("--resid2d-ckpt", type=str, required=True)
    p.add_argument("--resid2d-coupling-ckpt", type=str, default="")
    p.add_argument("--resid2d-coupling-blend", type=float, default=0.0)
    p.add_argument("--surfaceflow-ckpt", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--bins", type=float, nargs="+", default=[0.0, 0.03, 0.05])
    p.add_argument("--equalize-events", dest="equalize_events", action="store_true")
    p.add_argument("--no-equalize-events", dest="equalize_events", action="store_false")
    p.set_defaults(equalize_events=True)
    p.add_argument("--surface-slots-per-node", type=int, default=6)
    p.add_argument("--use-local2d-slots", action="store_true")
    p.add_argument("--neighbor-slots-per-node", type=int, default=4)
    p.add_argument(
        "--max-delta-quantile",
        type=float,
        default=0.995,
        help="Quantile in (0,1] of |Δq_inlet| per node used as rollout clamp. 0 disables.",
    )
    return p.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _split_ids(ids: list[int], *, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if val_ratio <= 0.0:
        return sorted(ids), []
    return split_event_ids(ids, val_ratio=val_ratio, seed=seed)


def _corr_mean(a: np.ndarray, b: np.ndarray) -> float:
    vals: list[float] = []
    for j in range(int(a.shape[1])):
        x = np.asarray(a[:, j], dtype=np.float64)
        y = np.asarray(b[:, j], dtype=np.float64)
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        c = np.corrcoef(x, y)[0, 1]
        if np.isfinite(c):
            vals.append(float(c))
    return float(np.mean(vals)) if vals else float("nan")


def _load_resid2d_model(path: Path, *, n_nodes: int, device: torch.device) -> tuple[ResidualNodeGRU, str]:
    ck = _load_torch(path)
    if str(ck.get("kind", "")) != "residual_m2_2d":
        raise ValueError(f"resid2d ckpt must be kind residual_m2_2d: {path}")
    dyn_feat = str((ck.get("cfg", {}) or {}).get("dyn_feat", ""))
    mcfg = ResidualGRUConfig(**ck["model_cfg"])
    model = ResidualNodeGRU(n_nodes=int(n_nodes), cfg=mcfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, dyn_feat


def main() -> None:
    a = parse_args()
    cfg = InletAuxCfg(
        model_id=int(a.model_id),
        warmup=int(a.warmup),
        k=int(a.k),
        ridge=float(a.ridge),
        bins=tuple(float(x) for x in a.bins),
        equalize_events=bool(a.equalize_events),
        use_surface_slots=True,
        use_local2d_slots=bool(a.use_local2d_slots),
    )
    seed_everything(int(a.seed))

    model_root = Path(a.model_root)
    cache_dir = Path(a.cache_dir) if a.cache_dir else None
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_ckpts = [_load_torch(Path(p)) for p in a.baseline_ckpt]
    warmup = int(cfg.warmup)
    for ck in baseline_ckpts[1:]:
        if int(ck["cfg"]["warmup"]) != warmup:
            raise ValueError("baseline_ckpt warmup mismatch")

    if a.split_from:
        ck = _load_torch(Path(a.split_from))
        split = ck.get("split", None)
        if split is None:
            raise ValueError("--split-from ckpt missing 'split'")
        train_ids = list(split["model_2"]["train"])
        val_ids = list(split["model_2"]["val"])
    else:
        ids = list_events(model_root, model_id=int(cfg.model_id), split="train")
        train_ids, val_ids = _split_ids(ids, val_ratio=float(a.val_ratio), seed=int(a.seed))
        split = {"model_2": {"train": train_ids, "val": val_ids}}

    graph = load_graph(model_root, model_id=int(cfg.model_id), split_for_static="train")
    n1 = int(graph.n_1d)
    n2 = int(graph.n_2d)
    area_2d = graph.area_2d.cpu().numpy().astype(np.float32, copy=False)
    conn_src = graph.conn_src_1d.cpu().numpy().astype(np.int64, copy=False)
    conn_dst = graph.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False)
    bed_2d = graph.head_offset[graph.n_1d :].cpu().numpy().astype(np.float32, copy=False)
    src2_full = graph.edge_index_2d[0].cpu().numpy().astype(np.int64, copy=False)
    dst2_full = graph.edge_index_2d[1].cpu().numpy().astype(np.int64, copy=False)
    e2 = int(src2_full.shape[0] // 2)
    edge_from = src2_full[:e2]
    edge_to = dst2_full[:e2]
    edge_weight = load_surface_edge_weight(model_root=model_root, model_id=int(cfg.model_id), split_for_static="train")

    surface_slot_edges, surface_slot_sign = build_coupled_surface_slot_index(
        conn_src_1d=conn_src,
        conn_dst_2d=conn_dst,
        edge_from=edge_from,
        edge_to=edge_to,
        edge_weight=edge_weight,
        n_1d=n1,
        slots_per_node=int(a.surface_slots_per_node),
    )
    if bool(cfg.use_local2d_slots):
        center_cell, neighbor_slots = build_coupled_neighbor_node_index(
            conn_src_1d=conn_src,
            conn_dst_2d=conn_dst,
            edge_from=edge_from,
            edge_to=edge_to,
            edge_weight=edge_weight,
            n_1d=n1,
            slots_per_node=int(a.neighbor_slots_per_node),
        )
    else:
        center_cell = None
        neighbor_slots = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    node_static_2d = graph.node_static_2d.to(device)
    edge_index_2d = graph.edge_index_2d.to(device)
    deg2 = torch.bincount(graph.edge_index_2d[1].cpu(), minlength=int(n2)).clamp(min=1).to(torch.float32)
    edge_deg_inv_2d = (1.0 / deg2).to(device)

    resid2d_main, resid2d_dyn = _load_resid2d_model(Path(a.resid2d_ckpt), n_nodes=n2, device=device)
    resid2d_cpl = None
    resid2d_cpl_dyn = ""
    cpl_blend = float(a.resid2d_coupling_blend)
    if a.resid2d_coupling_ckpt and cpl_blend > 0.0:
        resid2d_cpl, resid2d_cpl_dyn = _load_resid2d_model(Path(a.resid2d_coupling_ckpt), n_nodes=n2, device=device)

    surfaceflow_ckpt = _load_torch(Path(a.surfaceflow_ckpt))
    if str(surfaceflow_ckpt.get("kind", "")) != "aux_surfaceflow_m2_2d":
        raise ValueError("--surfaceflow-ckpt must be kind aux_surfaceflow_m2_2d")

    def _build_event_inputs(eid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        ev = load_event(model_root, graph=graph, split="train", event_id=eid, cache_dir=cache_dir)
        if ev.inlet_1d is None:
            raise ValueError("inlet_1d is required for inlet aux")
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
        y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
        y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
        q1_true = ev.inlet_1d.numpy().astype(np.float32, copy=False)
        q2edge_true = load_edge_flow_2d(
            model_root=model_root,
            model_id=int(cfg.model_id),
            split="train",
            event_id=eid,
            n_edges=e2,
            cache_dir=cache_dir,
        )
        y1_base, y2_base = predict_model2_from_baseline_ckpts(
            baseline_ckpts,
            graph2=graph,
            mixed_mode=str(a.mixed_mode),
            alpha_1d=float(a.alpha_1d),
            alpha_2d=float(a.alpha_2d),
            y1_init=y1_true,
            y2_init=y2_true,
            rain_2d=rain,
            warmup=warmup,
        )
        y2_main = _apply_resid2d(
            resid2d=resid2d_main,
            node_static_2d=node_static_2d,
            y2_base=y2_base,
            rain_2d=rain,
            bed2=bed_2d,
            warmup=warmup,
            dyn_feat=resid2d_dyn,
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
        if resid2d_cpl is not None:
            y2_alt = _apply_resid2d(
                resid2d=resid2d_cpl,
                node_static_2d=node_static_2d,
                y2_base=y2_base,
                rain_2d=rain,
                bed2=bed_2d,
                warmup=warmup,
                dyn_feat=resid2d_cpl_dyn,
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
            y2_src = ((1.0 - cpl_blend) * y2_main + cpl_blend * y2_alt).astype(np.float32, copy=False)
        else:
            y2_src = y2_main

        q2edge_pred = predict_surface_flow_from_ckpt(
            ckpt=surfaceflow_ckpt,
            y2_2d=y2_src,
            rain_2d=rain,
            bed_2d=bed_2d,
            edge_from=edge_from,
            edge_to=edge_to,
            q_edge_init=q2edge_true,
            warmup=warmup,
        )
        surface_slot_feats = build_coupled_surface_slot_features(
            q_edge=q2edge_pred,
            edge_slots=surface_slot_edges,
            edge_sign=surface_slot_sign,
        )
        if bool(cfg.use_local2d_slots):
            local2d_feats = build_coupled_neighbor_node_features(
                y2_2d=y2_src,
                bed_2d=bed_2d,
                center_cell=center_cell,
                neighbor_slots=neighbor_slots,
            )
        else:
            local2d_feats = None

        coupled_exo = build_coupled_1d_exo(
            y2_2d=y2_src,
            rain_2d=rain,
            area_2d=area_2d,
            conn_src_1d=conn_src,
            conn_dst_2d=conn_dst,
            n_1d=n1,
        )
        exo = build_inlet_local_surface_exo(
            coupled_exo=coupled_exo,
            surface_slot_feats=surface_slot_feats,
            local2d_node_feats=local2d_feats,
            use_surface_slots=bool(cfg.use_surface_slots),
            use_local2d_slots=bool(cfg.use_local2d_slots),
        )
        return q1_true, rain, y2_src, surface_slot_feats, local2d_feats, exo

    train_sequences: list[tuple[np.ndarray, np.ndarray]] = []
    train_exo: list[np.ndarray] = []
    delta_samples: list[np.ndarray] = []

    for eid in tqdm(train_ids, desc="load inlet aux train", leave=False):
        q_true, rain, _, _, _, exo = _build_event_inputs(int(eid))
        train_sequences.append((q_true, rain))
        train_exo.append(exo)
        dq = np.diff(q_true, axis=0)
        if dq.size:
            delta_samples.append(np.abs(dq).astype(np.float32, copy=False))

    w, counts = fit_inlet_aux(sequences=train_sequences, exo_sequences=train_exo, cfg=cfg)

    max_delta = None
    qv = float(a.max_delta_quantile)
    if qv > 0.0:
        if not delta_samples:
            raise ValueError("cannot estimate max_delta without train deltas")
        dq_all = np.concatenate(delta_samples, axis=0)
        max_delta = np.quantile(dq_all, q=min(max(qv, 0.0), 1.0), axis=0).astype(np.float32, copy=False)
        max_delta = np.maximum(max_delta, 1e-3).astype(np.float32, copy=False)

    metrics = {}
    payload_tmp = build_payload(
        cfg=cfg,
        split=split,
        w=w,
        counts=counts,
        exo_dim=int(train_exo[0].shape[2]),
        node_count=n1,
        source_baseline_ckpts=[str(x) for x in a.baseline_ckpt],
        source_resid2d_ckpt=str(a.resid2d_ckpt),
        source_resid2d_coupling_ckpt=str(a.resid2d_coupling_ckpt or ""),
        source_resid2d_coupling_blend=float(a.resid2d_coupling_blend),
        source_surfaceflow_ckpt=str(a.surfaceflow_ckpt),
        max_delta=max_delta,
    )

    if val_ids:
        rmses: list[float] = []
        corrs: list[float] = []
        for eid in tqdm(val_ids, desc="eval inlet aux val", leave=False):
            q_true, rain, y2_src, surface_slot_feats, local2d_feats, _ = _build_event_inputs(int(eid))
            q_pred = predict_inlet_from_ckpt(
                ckpt=payload_tmp,
                y2_2d=y2_src,
                rain_2d=rain,
                area_2d=area_2d,
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_1d=n1,
                q1_init=q_true,
                warmup=warmup,
                surface_slot_feats=surface_slot_feats,
                local2d_node_feats=local2d_feats,
            )
            err = q_pred[warmup:] - q_true[warmup:]
            rmses.append(float(np.mean(np.sqrt(np.mean(err**2, axis=0)))))
            corrs.append(_corr_mean(q_pred[warmup:], q_true[warmup:]))
        metrics = {
            "val_rmse": float(np.mean(rmses)),
            "val_corr": float(np.mean(corrs)),
        }
        print(f"[val] inlet_aux rmse={metrics['val_rmse']:.6f} corr={metrics['val_corr']:.6f}")

    payload = build_payload(
        cfg=cfg,
        split=split,
        w=w,
        counts=counts,
        exo_dim=int(train_exo[0].shape[2]),
        node_count=n1,
        source_baseline_ckpts=[str(x) for x in a.baseline_ckpt],
        source_resid2d_ckpt=str(a.resid2d_ckpt),
        source_resid2d_coupling_ckpt=str(a.resid2d_coupling_ckpt or ""),
        source_resid2d_coupling_blend=float(a.resid2d_coupling_blend),
        source_surfaceflow_ckpt=str(a.surfaceflow_ckpt),
        max_delta=max_delta,
        metrics=metrics,
    )
    torch.save(payload, out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "cfg": asdict(cfg),
                "metrics": metrics,
                "exo_dim": int(train_exo[0].shape[2]),
                "node_count": n1,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved inlet aux to {out_path}")


if __name__ == "__main__":
    main()
