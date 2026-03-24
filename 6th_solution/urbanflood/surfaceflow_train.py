from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from urbanflood.data import load_event, load_graph, list_events, split_event_ids
from urbanflood.surfaceflow_aux import (
    SurfaceFlowAuxCfg,
    build_payload,
    build_surface_flow_1d_features,
    build_surface_flow_exo,
    fit_surface_flow_aux,
    load_edge_flow_2d,
    predict_surface_flow_from_ckpt,
)
from urbanflood.utils import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=int, default=2, choices=[2])
    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, default="runs/surfaceflow_aux_m2.pt")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-from", type=str, default="")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--bins", type=float, nargs="+", default=[0.0, 0.03, 0.05])
    p.add_argument("--equalize-events", dest="equalize_events", action="store_true")
    p.add_argument("--no-equalize-events", dest="equalize_events", action="store_false")
    p.set_defaults(equalize_events=True)
    p.add_argument(
        "--max-delta-quantile",
        type=float,
        default=0.995,
        help="Quantile in (0,1] of |Δq| per 2D edge used as rollout clamp. 0 disables.",
    )
    return p.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
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


def main() -> None:
    a = parse_args()
    cfg = SurfaceFlowAuxCfg(
        model_id=int(a.model_id),
        warmup=int(a.warmup),
        k=int(a.k),
        ridge=float(a.ridge),
        bins=tuple(float(x) for x in a.bins),
        equalize_events=bool(a.equalize_events),
    )
    seed_everything(int(a.seed))

    model_root = Path(a.model_root)
    cache_dir = Path(a.cache_dir) if a.cache_dir else None
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    e2 = int(graph.edge_index_2d.shape[1] // 2)
    edge_from = graph.edge_index_2d[0, :e2].cpu().numpy().astype(np.int64, copy=False)
    edge_to = graph.edge_index_2d[1, :e2].cpu().numpy().astype(np.int64, copy=False)
    conn_src = graph.conn_src_1d.cpu().numpy().astype(np.int64, copy=False)
    conn_dst = graph.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False)
    bed_2d = graph.head_offset[graph.n_1d :].cpu().numpy().astype(np.float32, copy=False)

    train_sequences: list[tuple[np.ndarray, np.ndarray]] = []
    train_exo: list[np.ndarray] = []
    delta_samples: list[np.ndarray] = []

    for eid in tqdm(train_ids, desc="load surfaceflow train", leave=False):
        ev = load_event(model_root, graph=graph, split="train", event_id=eid, cache_dir=cache_dir)
        q_edge = load_edge_flow_2d(
            model_root=model_root,
            model_id=int(cfg.model_id),
            split="train",
            event_id=eid,
            n_edges=e2,
            cache_dir=cache_dir,
        )
        y2 = ev.y_2d.numpy().astype(np.float32, copy=False)
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)
        exo = build_surface_flow_exo(
            y2_2d=y2,
            rain_2d=rain,
            bed_2d=bed_2d,
            edge_from=edge_from,
            edge_to=edge_to,
        )
        train_sequences.append((q_edge, rain))
        train_exo.append(exo)
        dq = np.diff(q_edge, axis=0)
        if dq.size:
            delta_samples.append(np.abs(dq).astype(np.float32, copy=False))

    w, counts = fit_surface_flow_aux(sequences=train_sequences, exo_sequences=train_exo, cfg=cfg)

    max_delta = None
    qv = float(a.max_delta_quantile)
    if qv > 0.0:
        if not delta_samples:
            raise ValueError("cannot estimate max_delta without train deltas")
        dq_all = np.concatenate(delta_samples, axis=0)
        max_delta = np.quantile(dq_all, q=min(max(qv, 0.0), 1.0), axis=0).astype(np.float32, copy=False)
        max_delta = np.maximum(max_delta, 1e-3).astype(np.float32, copy=False)

    metrics = {}
    if val_ids:
        rmse_edges: list[float] = []
        corr_edges: list[float] = []
        rmse_nodes: list[float] = []
        corr_nodes: list[float] = []
        payload_tmp = build_payload(
            cfg=cfg,
            split=split,
            w=w,
            counts=counts,
            edge_count=e2,
            exo_dim=int(train_exo[0].shape[2]),
            max_delta=max_delta,
        )
        for eid in tqdm(val_ids, desc="eval surfaceflow val", leave=False):
            ev = load_event(model_root, graph=graph, split="train", event_id=eid, cache_dir=cache_dir)
            q_true = load_edge_flow_2d(
                model_root=model_root,
                model_id=int(cfg.model_id),
                split="train",
                event_id=eid,
                n_edges=e2,
                cache_dir=cache_dir,
            )
            q_pred = predict_surface_flow_from_ckpt(
                ckpt=payload_tmp,
                y2_2d=ev.y_2d.numpy().astype(np.float32, copy=False),
                rain_2d=ev.rain_2d.numpy().astype(np.float32, copy=False),
                bed_2d=bed_2d,
                edge_from=edge_from,
                edge_to=edge_to,
                q_edge_init=q_true,
                warmup=int(cfg.warmup),
            )
            err = q_pred[int(cfg.warmup) :] - q_true[int(cfg.warmup) :]
            rmse_edges.append(float(np.mean(np.sqrt(np.mean(err**2, axis=0)))))
            corr_edges.append(_corr_mean(q_pred[int(cfg.warmup) :], q_true[int(cfg.warmup) :]))

            node_true = build_surface_flow_1d_features(
                q_edge=q_true,
                edge_from=edge_from,
                edge_to=edge_to,
                n_2d=n2,
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_1d=n1,
            )[int(cfg.warmup) :].reshape(-1, 4 * n1)
            node_pred = build_surface_flow_1d_features(
                q_edge=q_pred,
                edge_from=edge_from,
                edge_to=edge_to,
                n_2d=n2,
                conn_src_1d=conn_src,
                conn_dst_2d=conn_dst,
                n_1d=n1,
            )[int(cfg.warmup) :].reshape(-1, 4 * n1)
            rmse_nodes.append(float(np.mean(np.sqrt(np.mean((node_pred - node_true) ** 2, axis=0)))))
            corr_nodes.append(_corr_mean(node_pred, node_true))

        metrics = {
            "val_edge_rmse": float(np.mean(rmse_edges)),
            "val_edge_corr": float(np.mean(corr_edges)),
            "val_nodeagg_rmse": float(np.mean(rmse_nodes)),
            "val_nodeagg_corr": float(np.mean(corr_nodes)),
        }
        print(
            "[val] surfaceflow_aux "
            f"edge_rmse={metrics['val_edge_rmse']:.6f} "
            f"edge_corr={metrics['val_edge_corr']:.6f} "
            f"nodeagg_rmse={metrics['val_nodeagg_rmse']:.6f} "
            f"nodeagg_corr={metrics['val_nodeagg_corr']:.6f}"
        )

    payload = build_payload(
        cfg=cfg,
        split=split,
        w=w,
        counts=counts,
        edge_count=e2,
        exo_dim=int(train_exo[0].shape[2]),
        max_delta=max_delta,
        metrics=metrics,
    )
    torch.save(payload, out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "cfg": asdict(cfg),
                "metrics": metrics,
                "edge_count": e2,
                "exo_dim": int(train_exo[0].shape[2]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved surfaceflow aux to {out_path}")


if __name__ == "__main__":
    main()
