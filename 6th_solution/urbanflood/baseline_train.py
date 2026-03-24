# #####
# baseline_train.py
# #####

# urbanflood/baseline_train.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from urbanflood.baseline import (
    AR1XConfig,
    RegimeARKXConfig,
    aggregate_2d_to_1d_sum,
    build_coupled_1d_exo,
    build_coupled_local_1d_exo,
    build_inlet_augmented_exo,
    build_inlet_storage_augmented_exo,
    build_storage_augmented_exo,
    connected_area_1d,
    fit_ar1x_per_node,
    fit_regime_arkx_exo_per_node,
    fit_regime_arkx_per_node,
    rollout_ar1x,
    rollout_regime_arkx_exo,
    rollout_regime_arkx,
)
from urbanflood.data import load_event, load_graph, list_events, split_event_ids
from urbanflood.metric import score_event_from_full_grids
from urbanflood.utils import seed_everything


@dataclass(frozen=True)
class BaselineTrainCfg:
    model_root: str = "Models"
    cache_dir: str = ".cache/urbanflood"
    out_path: str = "runs/baseline.pt"

    preset: str = "a"  # "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i"
    warmup: int = 10

    seed: int = 42
    val_ratio: float = 0.1
    split_from: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="a", choices=["a", "b", "c", "d", "e", "f", "g", "h", "i"], help="Baseline preset: a=non-split, b=split, c=split+coupled-1d, d=split+coupled-1d+inlet, e=split+coupled-1d+storage+inlet, f=e plus storage-state-conditioned 1D regimes, g=d plus local coupled 2D neighborhood features, h=g plus inlet-state-conditioned 1D regimes, i=g plus storage state in both inlet and 1D passes.")
    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, default="runs/baseline.pt")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1, help="0 disables validation split (full-train).")
    p.add_argument(
        "--split-from",
        type=str,
        default="",
        help="Optional path to an existing baseline ckpt whose split will be reused (recommended for preset b).",
    )
    return p.parse_args()


def _split_ids(ids: list[int], *, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if val_ratio <= 0.0:
        return sorted(ids), []
    return split_event_ids(ids, val_ratio=val_ratio, seed=seed)


def _build_coupled_neighbor_slots(*, model_root: Path, graph2, slots_per_node: int = 4) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    n1 = int(graph2.n_1d)
    n2 = int(graph2.n_2d)
    base = Path(model_root) / "Model_2" / "train"
    df = pd.read_csv(base / "2d_edges_static.csv", usecols=["edge_idx", "face_length", "length"])
    df = df.sort_values(["edge_idx"], kind="mergesort").reset_index(drop=True)
    face = df["face_length"].to_numpy(np.float32)
    length = np.maximum(df["length"].to_numpy(np.float32), 1e-3)
    w = (face / length).astype(np.float32, copy=False)

    src_full = graph2.edge_index_2d[0].cpu().numpy().astype(np.int64, copy=False)
    dst_full = graph2.edge_index_2d[1].cpu().numpy().astype(np.int64, copy=False)
    e_dir = int(src_full.shape[0] // 2)
    src = src_full[:e_dir]
    dst = dst_full[:e_dir]
    if w.shape[0] != e_dir:
        raise ValueError("2D edge static count mismatch")

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n2)]
    for s, d, ww in zip(src.tolist(), dst.tolist(), w.tolist(), strict=True):
        adj[s].append((d, ww))
        adj[d].append((s, ww))

    center = np.full((n1,), -1, dtype=np.int64)
    conn_src = graph2.conn_src_1d.cpu().numpy().astype(np.int64, copy=False)
    conn_dst = graph2.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False)
    center[conn_src] = conn_dst

    slots = np.full((n1, int(slots_per_node)), -1, dtype=np.int64)
    for node_idx in range(n1):
        c = int(center[node_idx])
        if c < 0:
            continue
        nbrs = sorted(adj[c], key=lambda x: (-float(x[1]), int(x[0])))
        used: set[int] = set()
        j = 0
        for nbr, _ in nbrs:
            if nbr == c or nbr in used:
                continue
            slots[node_idx, j] = int(nbr)
            used.add(int(nbr))
            j += 1
            if j >= int(slots_per_node):
                break
    return center, slots


def _strict_increasing_bins(values: np.ndarray, *, probs: tuple[float, ...], floor: float = 0.0, eps: float = 1e-4) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    pos = arr[arr > floor]
    if pos.size == 0:
        return (float(floor), float(floor + 0.02), float(floor + 0.10))

    qs = np.quantile(pos, probs).astype(np.float64, copy=False)
    bins: list[float] = [float(floor)]
    last = float(floor)
    for q in qs.tolist():
        qf = max(float(q), last + float(eps))
        bins.append(qf)
        last = qf
    return tuple(bins)


def _load_sequences(
    *,
    model_root: Path,
    cache_dir: Path | None,
    model_id: int,
    event_ids: list[int],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    graph = load_graph(model_root, model_id=model_id, split_for_static="train")
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    for eid in tqdm(event_ids, desc=f"load model {model_id} train events", leave=False):
        ev = load_event(model_root, graph=graph, split="train", event_id=eid, cache_dir=cache_dir)
        y = np.concatenate([ev.y_1d.numpy(), ev.y_2d.numpy()], axis=1)  # [T, N]
        rain = ev.rain_2d.numpy()  # [T, N2]
        if not np.isfinite(y).all():
            raise ValueError(f"non-finite y in train event: model={model_id} event={eid}")
        if not np.isfinite(rain).all():
            raise ValueError(f"non-finite rain in train event: model={model_id} event={eid}")
        seqs.append((y, rain))
    return seqs, graph.n_1d, graph.n_2d


def _eval_model(
    *,
    model_root: Path,
    cache_dir: Path | None,
    model_id: int,
    val_ids: list[int],
    warmup: int,
    w1: np.ndarray,
    model2_kind: str,
    model2_payload: dict,
) -> tuple[float, float, float]:
    graph = load_graph(model_root, model_id=model_id, split_for_static="train")
    scores = []
    s1d = []
    s2d = []

    for eid in tqdm(val_ids, desc=f"eval model {model_id}", leave=False):
        ev = load_event(model_root, graph=graph, split="train", event_id=eid, cache_dir=cache_dir)
        y1_true = ev.y_1d.numpy()
        y2_true = ev.y_2d.numpy()
        rain = ev.rain_2d.numpy()

        if model_id == 1:
            y_init = np.concatenate([y1_true, y2_true], axis=1)
            y_pred = rollout_ar1x(w=w1, y_init=y_init, rain=rain, warmup=warmup)
            y1_pred = y_pred[:, : graph.n_1d]
            y2_pred = y_pred[:, graph.n_1d :]
        else:
            if str(model2_kind) == "regime_arkx":
                w2 = model2_payload["w"]
                bins2 = tuple(float(x) for x in model2_payload["bins"])
                y_init = np.concatenate([y1_true, y2_true], axis=1)
                y_pred = rollout_regime_arkx(w=w2, bins=bins2, y_init=y_init, rain=rain, warmup=warmup)
                y1_pred = y_pred[:, : graph.n_1d]
                y2_pred = y_pred[:, graph.n_1d :]
            elif str(model2_kind) == "split_1d2d":
                p1 = model2_payload["parts"]["1d"]
                p2 = model2_payload["parts"]["2d"]
                w1d = p1["w"]
                w2d = p2["w"]
                b1 = tuple(float(x) for x in p1["bins"])
                b2 = tuple(float(x) for x in p2["bins"])
                md = float(p1.get("max_delta", 0.0))
                md_arg = None if md <= 0.0 else md
                y1_pred = rollout_regime_arkx(w=w1d, bins=b1, y_init=y1_true, rain=rain, warmup=warmup, max_delta=md_arg)
                y2_pred = rollout_regime_arkx(w=w2d, bins=b2, y_init=y2_true, rain=rain, warmup=warmup)
            elif str(model2_kind) == "split_1d2d_coupled":
                p1 = model2_payload["parts"]["1d"]
                p2 = model2_payload["parts"]["2d"]
                w1d = p1["w"]
                w2d = p2["w"]
                b1 = tuple(float(x) for x in p1["bins"])
                b2 = tuple(float(x) for x in p2["bins"])
                md = float(p1.get("max_delta", 0.0))
                md_arg = None if md <= 0.0 else md
                y2_pred = rollout_regime_arkx(w=w2d, bins=b2, y_init=y2_true, rain=rain, warmup=warmup)
                exo1d = build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=graph.area_2d.numpy(),
                    conn_src_1d=graph.conn_src_1d.numpy(),
                    conn_dst_2d=graph.conn_dst_2d.numpy(),
                    n_1d=graph.n_1d,
                )
                y1_pred = rollout_regime_arkx_exo(w=w1d, bins=b1, y_init=y1_true, rain=rain, exo=exo1d, warmup=warmup, max_delta=md_arg)
            elif str(model2_kind) == "split_1d2d_coupled_inlet":
                p1 = model2_payload["parts"]["1d"]
                p2 = model2_payload["parts"]["2d"]
                y2_pred = rollout_regime_arkx(
                    w=p2["w"],
                    bins=tuple(float(x) for x in p2["bins"]),
                    y_init=y2_true,
                    rain=rain,
                    warmup=warmup,
                )
                coupled_exo = build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=graph.area_2d.numpy(),
                    conn_src_1d=graph.conn_src_1d.numpy(),
                    conn_dst_2d=graph.conn_dst_2d.numpy(),
                    n_1d=graph.n_1d,
                )
                q1_true = ev.inlet_1d.numpy() if ev.inlet_1d is not None else None
                if q1_true is None:
                    raise ValueError("inlet_1d is required for split_1d2d_coupled_inlet evaluation")
                q_model = p1["q_model"]
                q_pred = rollout_regime_arkx_exo(
                    w=q_model["w"],
                    bins=tuple(float(x) for x in q_model["bins"]),
                    y_init=q1_true,
                    rain=rain,
                    exo=coupled_exo,
                    warmup=warmup,
                )
                y_model = p1["y_model"]
                md = float(y_model.get("max_delta", 0.0))
                md_arg = None if md <= 0.0 else md
                y1_pred = rollout_regime_arkx_exo(
                    w=y_model["w"],
                    bins=tuple(float(x) for x in y_model["bins"]),
                    y_init=y1_true,
                    rain=rain,
                    exo=build_inlet_augmented_exo(coupled_exo, q_pred),
                    warmup=warmup,
                    max_delta=md_arg,
                )
            elif str(model2_kind) in {
                "split_1d2d_coupled_local_inlet",
                "split_1d2d_coupled_local_inlet_statebin",
                "split_1d2d_coupled_local_storage_inlet",
            }:
                p1 = model2_payload["parts"]["1d"]
                p2 = model2_payload["parts"]["2d"]
                center_cell = p1["center_cell"].cpu().numpy() if torch.is_tensor(p1["center_cell"]) else np.asarray(p1["center_cell"], dtype=np.int64)
                neighbor_slots = (
                    p1["neighbor_slots"].cpu().numpy() if torch.is_tensor(p1["neighbor_slots"]) else np.asarray(p1["neighbor_slots"], dtype=np.int64)
                )
                y2_pred = rollout_regime_arkx(
                    w=p2["w"],
                    bins=tuple(float(x) for x in p2["bins"]),
                    y_init=y2_true,
                    rain=rain,
                    warmup=warmup,
                )
                q1_true = ev.inlet_1d.numpy() if ev.inlet_1d is not None else None
                if q1_true is None:
                    raise ValueError(f"inlet_1d is required for {model2_kind} evaluation")
                conn_area_1d = None
                v_pred = None
                local_exo = build_coupled_local_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=graph.area_2d.numpy(),
                    conn_src_1d=graph.conn_src_1d.numpy(),
                    conn_dst_2d=graph.conn_dst_2d.numpy(),
                    n_1d=graph.n_1d,
                    center_cell=center_cell,
                    neighbor_slots=neighbor_slots,
                )
                if str(model2_kind) == "split_1d2d_coupled_local_storage_inlet":
                    if ev.volume_2d is None:
                        raise ValueError(f"volume_2d is required for {model2_kind} evaluation")
                    area_2d = graph.area_2d.numpy()
                    conn_src_1d = graph.conn_src_1d.numpy()
                    conn_dst_2d = graph.conn_dst_2d.numpy()
                    conn_area_1d = connected_area_1d(
                        area_2d,
                        conn_src_1d=conn_src_1d,
                        conn_dst_2d=conn_dst_2d,
                        n_1d=graph.n_1d,
                    ).astype(np.float32, copy=False)
                    v1_true = aggregate_2d_to_1d_sum(
                        ev.volume_2d.numpy().astype(np.float32, copy=False),
                        conn_src_1d=conn_src_1d,
                        conn_dst_2d=conn_dst_2d,
                        n_1d=graph.n_1d,
                    ).astype(np.float32, copy=False)
                    v_model = p1["v_model"]
                    vmd_raw = v_model.get("max_delta", None)
                    if torch.is_tensor(vmd_raw):
                        vmd_arg = vmd_raw.numpy()
                    else:
                        vmd_arg = vmd_raw
                    v_pred = rollout_regime_arkx_exo(
                        w=v_model["w"],
                        bins=tuple(float(x) for x in v_model["bins"]),
                        y_init=v1_true,
                        rain=rain,
                        exo=local_exo,
                        warmup=warmup,
                        max_delta=vmd_arg,
                    )
                q_model = p1["q_model"]
                q_pred = rollout_regime_arkx_exo(
                    w=q_model["w"],
                    bins=tuple(float(x) for x in q_model["bins"]),
                    y_init=q1_true,
                    rain=rain,
                    exo=build_storage_augmented_exo(local_exo, v_pred, conn_area_1d) if v_pred is not None and conn_area_1d is not None else local_exo,
                    warmup=warmup,
                )
                y_model = p1["y_model"]
                md = float(y_model.get("max_delta", 0.0))
                md_arg = None if md <= 0.0 else md
                regime_signal = None
                if str(model2_kind) == "split_1d2d_coupled_local_inlet_statebin":
                    regime_src = str(y_model.get("regime_source", ""))
                    if regime_src != "inlet_positive":
                        raise ValueError(f"unsupported y_model regime_source: {regime_src}")
                    regime_signal = np.maximum(q_pred, 0.0).astype(np.float32, copy=False)
                y1_pred = rollout_regime_arkx_exo(
                    w=y_model["w"],
                    bins=tuple(float(x) for x in y_model["bins"]),
                    y_init=y1_true,
                    rain=rain,
                    exo=(
                        build_inlet_storage_augmented_exo(local_exo, q_pred, v_pred, conn_area_1d)
                        if v_pred is not None and conn_area_1d is not None
                        else build_inlet_augmented_exo(local_exo, q_pred)
                    ),
                    warmup=warmup,
                    regime_signal=regime_signal,
                    max_delta=md_arg,
                )
            elif str(model2_kind) in {"split_1d2d_coupled_storage_inlet", "split_1d2d_coupled_storage_inlet_statebin"}:
                p1 = model2_payload["parts"]["1d"]
                p2 = model2_payload["parts"]["2d"]
                y2_pred = rollout_regime_arkx(
                    w=p2["w"],
                    bins=tuple(float(x) for x in p2["bins"]),
                    y_init=y2_true,
                    rain=rain,
                    warmup=warmup,
                )
                area_2d = graph.area_2d.numpy()
                conn_src_1d = graph.conn_src_1d.numpy()
                conn_dst_2d = graph.conn_dst_2d.numpy()
                conn_area_1d = connected_area_1d(
                    area_2d,
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=graph.n_1d,
                ).astype(np.float32, copy=False)
                coupled_exo = build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=area_2d,
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=graph.n_1d,
                )
                q1_true = ev.inlet_1d.numpy() if ev.inlet_1d is not None else None
                if q1_true is None:
                    raise ValueError("inlet_1d is required for split_1d2d_coupled_storage_inlet evaluation")
                if ev.volume_2d is None:
                    raise ValueError("volume_2d is required for split_1d2d_coupled_storage_inlet evaluation")
                v1_true = aggregate_2d_to_1d_sum(
                    ev.volume_2d.numpy().astype(np.float32, copy=False),
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=graph.n_1d,
                ).astype(np.float32, copy=False)
                v_model = p1["v_model"]
                vmd_raw = v_model.get("max_delta", None)
                if torch.is_tensor(vmd_raw):
                    vmd_arg = vmd_raw.numpy()
                else:
                    vmd_arg = vmd_raw
                v_pred = rollout_regime_arkx_exo(
                    w=v_model["w"],
                    bins=tuple(float(x) for x in v_model["bins"]),
                    y_init=v1_true,
                    rain=rain,
                    exo=coupled_exo,
                    warmup=warmup,
                    max_delta=vmd_arg,
                )
                q_model = p1["q_model"]
                q_pred = rollout_regime_arkx_exo(
                    w=q_model["w"],
                    bins=tuple(float(x) for x in q_model["bins"]),
                    y_init=q1_true,
                    rain=rain,
                    exo=build_storage_augmented_exo(coupled_exo, v_pred, conn_area_1d),
                    warmup=warmup,
                )
                y_model = p1["y_model"]
                md_raw = y_model.get("max_delta", 0.0)
                if torch.is_tensor(md_raw):
                    md_arg = md_raw.numpy()
                else:
                    md_val = float(md_raw)
                    md_arg = None if md_val <= 0.0 else md_val
                regime_signal = None
                if str(model2_kind) == "split_1d2d_coupled_storage_inlet_statebin":
                    regime_src = str(y_model.get("regime_source", ""))
                    if regime_src != "storage_depth":
                        raise ValueError(f"unsupported y_model regime_source: {regime_src}")
                    regime_signal = (v_pred / np.maximum(conn_area_1d[None, :], 1.0)).astype(np.float32, copy=False)
                y1_pred = rollout_regime_arkx_exo(
                    w=y_model["w"],
                    bins=tuple(float(x) for x in y_model["bins"]),
                    y_init=y1_true,
                    rain=rain,
                    exo=build_inlet_storage_augmented_exo(coupled_exo, q_pred, v_pred, conn_area_1d),
                    warmup=warmup,
                    regime_signal=regime_signal,
                    max_delta=md_arg,
                )
            else:
                raise ValueError(f"unknown model2_kind: {model2_kind}")

        es = score_event_from_full_grids(
            model_id=model_id,
            event_id=eid,
            y_true_1d=y1_true,
            y_pred_1d=y1_pred,
            y_true_2d=y2_true,
            y_pred_2d=y2_pred,
            warmup=warmup,
        )
        scores.append(es.score)
        s1d.append(es.score_1d)
        s2d.append(es.score_2d)

    return float(np.mean(scores)), float(np.mean(s1d)), float(np.mean(s2d))


def main() -> None:
    a = parse_args()
    cfg = BaselineTrainCfg(
        model_root=str(a.model_root),
        cache_dir=str(a.cache_dir),
        out_path=str(a.out),
        preset=str(a.preset),
        warmup=int(a.warmup),
        seed=int(a.seed),
        val_ratio=float(a.val_ratio),
        split_from=str(a.split_from),
    )

    seed_everything(cfg.seed)
    model_root = Path(cfg.model_root)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Split handling: either reuse from an existing ckpt or create fresh.
    if cfg.split_from:
        ck = torch.load(cfg.split_from, map_location="cpu")
        split = ck.get("split", None)
        if split is None:
            raise ValueError("--split-from ckpt missing 'split'")
        train1 = list(split["model_1"]["train"])
        val1 = list(split["model_1"]["val"])
        train2 = list(split["model_2"]["train"])
        val2 = list(split["model_2"]["val"])
    else:
        ids1 = list_events(model_root, model_id=1, split="train")
        ids2 = list_events(model_root, model_id=2, split="train")
        train1, val1 = _split_ids(ids1, val_ratio=cfg.val_ratio, seed=cfg.seed)
        train2, val2 = _split_ids(ids2, val_ratio=cfg.val_ratio, seed=cfg.seed)
        split = {"model_1": {"train": train1, "val": val1}, "model_2": {"train": train2, "val": val2}}

    # Load train sequences
    seqs1, n1_1d, n1_2d = _load_sequences(model_root=model_root, cache_dir=cache_dir, model_id=1, event_ids=train1)
    seqs2, n2_1d, n2_2d = _load_sequences(model_root=model_root, cache_dir=cache_dir, model_id=2, event_ids=train2)

    # Model 1 baseline (AR1X, same for presets)
    w1 = fit_ar1x_per_node(seqs1, cfg=AR1XConfig(ridge=1e-4))

    # Model 2 baseline
    preset = str(cfg.preset)
    if preset == "a":
        # Non-split AR10X with 4 regimes (bins 0.0/0.03/0.05).
        w2, counts2 = fit_regime_arkx_per_node(
            seqs2,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05)),
        )
        model2_kind = "regime_arkx"
        model2_payload = {
            "kind": model2_kind,
            "k": 10,
            "bins": [0.0, 0.03, 0.05],
            "regime_step_counts": torch.from_numpy(counts2),
            "w": torch.from_numpy(w2),
        }
    elif preset == "b":
        # Split Model2: 1D equalized (clamped), 2D tuned regimes.
        seqs2_1d = [(y[:, :n2_1d], rain) for (y, rain) in seqs2]
        seqs2_2d = [(y[:, n2_1d:], rain) for (y, rain) in seqs2]

        w2_2d, counts2_2d = fit_regime_arkx_per_node(
            seqs2_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05, 0.16)),
        )
        w2_1d, counts2_1d = fit_regime_arkx_per_node(
            seqs2_1d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        model2_kind = "split_1d2d"
        model2_payload = {
            "kind": model2_kind,
            "n_1d": int(n2_1d),
            "n_2d": int(n2_2d),
            "parts": {
                "1d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05],
                    "equalize_events": True,
                    "max_delta": float(0.4),
                    "regime_step_counts": torch.from_numpy(counts2_1d),
                    "w": torch.from_numpy(w2_1d),
                },
                "2d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05, 0.16],
                    "equalize_events": False,
                    "regime_step_counts": torch.from_numpy(counts2_2d),
                    "w": torch.from_numpy(w2_2d),
                },
            },
        }
    elif preset == "c":
        seqs2_1d = [(y[:, :n2_1d], rain) for (y, rain) in seqs2]
        seqs2_2d = [(y[:, n2_1d:], rain) for (y, rain) in seqs2]
        graph2 = load_graph(model_root, model_id=2, split_for_static="train")

        w2_2d, counts2_2d = fit_regime_arkx_per_node(
            seqs2_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05, 0.16)),
        )

        exo2_1d: list[np.ndarray] = []
        for (y2_true, rain) in tqdm(seqs2_2d, desc="build model 2 1d coupled exo", leave=False):
            y2_pred = rollout_regime_arkx(w=w2_2d, bins=(0.0, 0.03, 0.05, 0.16), y_init=y2_true, rain=rain, warmup=cfg.warmup)
            exo2_1d.append(
                build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=graph2.area_2d.numpy(),
                    conn_src_1d=graph2.conn_src_1d.numpy(),
                    conn_dst_2d=graph2.conn_dst_2d.numpy(),
                    n_1d=n2_1d,
                )
            )

        w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
            seqs2_1d,
            exo_sequences=exo2_1d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        model2_kind = "split_1d2d_coupled"
        model2_payload = {
            "kind": model2_kind,
            "n_1d": int(n2_1d),
            "n_2d": int(n2_2d),
            "parts": {
                "1d": {
                    "kind": "regime_arkx_exo",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05],
                    "equalize_events": True,
                    "max_delta": float(0.4),
                    "exo_kind": "coupled_2d_state_v1",
                    "exo_dim": int(exo2_1d[0].shape[2]),
                    "regime_step_counts": torch.from_numpy(counts2_1d),
                    "w": torch.from_numpy(w2_1d),
                },
                "2d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05, 0.16],
                    "equalize_events": False,
                    "regime_step_counts": torch.from_numpy(counts2_2d),
                    "w": torch.from_numpy(w2_2d),
                },
            },
        }
    elif preset == "d":
        seqs2_1d = [(y[:, :n2_1d], rain) for (y, rain) in seqs2]
        seqs2_2d = [(y[:, n2_1d:], rain) for (y, rain) in seqs2]
        graph2 = load_graph(model_root, model_id=2, split_for_static="train")
        train_events2 = [load_event(model_root, graph=graph2, split="train", event_id=eid, cache_dir=cache_dir) for eid in train2]
        inlet2 = [ev.inlet_1d.numpy() if ev.inlet_1d is not None else None for ev in train_events2]
        if any(x is None for x in inlet2):
            raise ValueError("preset d requires inlet_1d in training events")

        w2_2d, counts2_2d = fit_regime_arkx_per_node(
            seqs2_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05, 0.16)),
        )

        coupled_exo_2d: list[np.ndarray] = []
        for (y2_true, rain) in tqdm(seqs2_2d, desc="build model 2 coupled exo", leave=False):
            y2_pred = rollout_regime_arkx(w=w2_2d, bins=(0.0, 0.03, 0.05, 0.16), y_init=y2_true, rain=rain, warmup=cfg.warmup)
            coupled_exo_2d.append(
                build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=graph2.area_2d.numpy(),
                    conn_src_1d=graph2.conn_src_1d.numpy(),
                    conn_dst_2d=graph2.conn_dst_2d.numpy(),
                    n_1d=n2_1d,
                )
            )

        q2_1d = [np.asarray(x, dtype=np.float32) for x in inlet2]
        wq_1d, countsq_1d = fit_regime_arkx_exo_per_node(
            [(q, rain) for q, (_, rain) in zip(q2_1d, seqs2_1d, strict=True)],
            exo_sequences=coupled_exo_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        y1_exo_2d: list[np.ndarray] = []
        for q_true, coupled_exo, (_, rain) in zip(q2_1d, coupled_exo_2d, seqs2_1d, strict=True):
            q_pred = rollout_regime_arkx_exo(
                w=wq_1d,
                bins=(0.0, 0.03, 0.05),
                y_init=q_true,
                rain=rain,
                exo=coupled_exo,
                warmup=cfg.warmup,
            )
            y1_exo_2d.append(build_inlet_augmented_exo(coupled_exo, q_pred))

        w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
            seqs2_1d,
            exo_sequences=y1_exo_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        model2_kind = "split_1d2d_coupled_inlet"
        model2_payload = {
            "kind": model2_kind,
            "n_1d": int(n2_1d),
            "n_2d": int(n2_2d),
            "parts": {
                "1d": {
                    "kind": "regime_arkx_exo_three_pass" if preset == "i" else "regime_arkx_exo_two_pass",
                    "exo_kind": "coupled_2d_state_v1_plus_inlet_v1",
                    "q_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [0.0, 0.03, 0.05],
                        "equalize_events": True,
                        "exo_dim": int(coupled_exo_2d[0].shape[2]),
                        "regime_step_counts": torch.from_numpy(countsq_1d),
                        "w": torch.from_numpy(wq_1d),
                    },
                    "y_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [0.0, 0.03, 0.05],
                        "equalize_events": True,
                        "exo_dim": int(y1_exo_2d[0].shape[2]),
                        "max_delta": float(0.4),
                        "regime_step_counts": torch.from_numpy(counts2_1d),
                        "w": torch.from_numpy(w2_1d),
                    },
                },
                "2d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05, 0.16],
                    "equalize_events": False,
                    "regime_step_counts": torch.from_numpy(counts2_2d),
                    "w": torch.from_numpy(w2_2d),
                },
            },
        }
    elif preset in {"g", "h", "i"}:
        seqs2_1d = [(y[:, :n2_1d], rain) for (y, rain) in seqs2]
        seqs2_2d = [(y[:, n2_1d:], rain) for (y, rain) in seqs2]
        graph2 = load_graph(model_root, model_id=2, split_for_static="train")
        train_events2 = [load_event(model_root, graph=graph2, split="train", event_id=eid, cache_dir=cache_dir) for eid in train2]
        inlet2 = [ev.inlet_1d.numpy() if ev.inlet_1d is not None else None for ev in train_events2]
        volume2 = [ev.volume_2d.numpy() if ev.volume_2d is not None else None for ev in train_events2]
        if any(x is None for x in inlet2):
            raise ValueError(f"preset {preset} requires inlet_1d in training events")
        if preset == "i" and any(x is None for x in volume2):
            raise ValueError("preset i requires volume_2d in training events")

        center_cell, neighbor_slots = _build_coupled_neighbor_slots(model_root=model_root, graph2=graph2, slots_per_node=4)
        area_2d = graph2.area_2d.numpy().astype(np.float32, copy=False)
        conn_src_1d = graph2.conn_src_1d.numpy()
        conn_dst_2d = graph2.conn_dst_2d.numpy()
        conn_area_1d = connected_area_1d(
            area_2d,
            conn_src_1d=conn_src_1d,
            conn_dst_2d=conn_dst_2d,
            n_1d=n2_1d,
        ).astype(np.float32, copy=False)

        w2_2d, counts2_2d = fit_regime_arkx_per_node(
            seqs2_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05, 0.16)),
        )

        local_exo_2d: list[np.ndarray] = []
        for (y2_true, rain) in tqdm(seqs2_2d, desc="build model 2 local coupled exo", leave=False):
            y2_pred = rollout_regime_arkx(w=w2_2d, bins=(0.0, 0.03, 0.05, 0.16), y_init=y2_true, rain=rain, warmup=cfg.warmup)
            local_exo_2d.append(
                build_coupled_local_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=area_2d,
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=n2_1d,
                    center_cell=center_cell,
                    neighbor_slots=neighbor_slots,
                )
            )

        q2_1d = [np.asarray(x, dtype=np.float32) for x in inlet2]
        q_exo_1d: list[np.ndarray] = list(local_exo_2d)
        y1_exo_2d: list[np.ndarray] = []
        if preset == "i":
            vagg2_1d = [
                aggregate_2d_to_1d_sum(
                    np.asarray(v, dtype=np.float32),
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=n2_1d,
                ).astype(np.float32, copy=False)
                for v in volume2
            ]
            wv_1d, countsv_1d = fit_regime_arkx_exo_per_node(
                [(v, rain) for v, (_, rain) in zip(vagg2_1d, seqs2_1d, strict=True)],
                exo_sequences=local_exo_2d,
                cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
            )
            dv_all = np.concatenate([np.abs(np.diff(v, axis=0)).astype(np.float32, copy=False) for v in vagg2_1d], axis=0)
            v_max_delta = np.maximum(np.quantile(dv_all, q=0.995, axis=0).astype(np.float32, copy=False), 1e-3)
            vagg_pred_1d: list[np.ndarray] = []
            for v_true, local_exo, (_, rain) in zip(vagg2_1d, local_exo_2d, seqs2_1d, strict=True):
                vagg_pred_1d.append(
                    rollout_regime_arkx_exo(
                        w=wv_1d,
                        bins=(0.0, 0.03, 0.05),
                        y_init=v_true,
                        rain=rain,
                        exo=local_exo,
                        warmup=cfg.warmup,
                        max_delta=v_max_delta,
                    )
                )
            q_exo_1d = [
                build_storage_augmented_exo(local_exo, v_pred, conn_area_1d)
                for local_exo, v_pred in zip(local_exo_2d, vagg_pred_1d, strict=True)
            ]
        wq_1d, countsq_1d = fit_regime_arkx_exo_per_node(
            [(q, rain) for q, (_, rain) in zip(q2_1d, seqs2_1d, strict=True)],
            exo_sequences=q_exo_1d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-2, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        for idx_ev, (q_true, local_exo, (_, rain)) in enumerate(zip(q2_1d, local_exo_2d, seqs2_1d, strict=True)):
            q_pred = rollout_regime_arkx_exo(
                w=wq_1d,
                bins=(0.0, 0.03, 0.05),
                y_init=q_true,
                rain=rain,
                exo=q_exo_1d[idx_ev],
                warmup=cfg.warmup,
            )
            if preset == "i":
                y1_exo_2d.append(build_inlet_storage_augmented_exo(local_exo, q_pred, vagg_pred_1d[idx_ev], conn_area_1d))
            else:
                y1_exo_2d.append(build_inlet_augmented_exo(local_exo, q_pred))

        if preset == "h":
            inlet_state_bins = _strict_increasing_bins(
                np.concatenate([np.maximum(q[cfg.warmup :], 0.0) for q in q2_1d], axis=0),
                probs=(0.50, 0.90),
                floor=0.0,
            )
            q_pred_1d: list[np.ndarray] = []
            for q_true, local_exo, (_, rain) in zip(q2_1d, local_exo_2d, seqs2_1d, strict=True):
                q_pred_1d.append(
                    rollout_regime_arkx_exo(
                        w=wq_1d,
                        bins=(0.0, 0.03, 0.05),
                        y_init=q_true,
                        rain=rain,
                        exo=local_exo,
                        warmup=cfg.warmup,
                    )
                )
            w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
                seqs2_1d,
                exo_sequences=y1_exo_2d,
                regime_signal_sequences=[np.maximum(q_pred, 0.0).astype(np.float32, copy=False) for q_pred in q_pred_1d],
                cfg=RegimeARKXConfig(k=10, ridge=1e-2, bins=inlet_state_bins, equalize_events=True),
            )
            model2_kind = "split_1d2d_coupled_local_inlet_statebin"
        else:
            inlet_state_bins = (0.0, 0.03, 0.05)
            w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
                seqs2_1d,
                exo_sequences=y1_exo_2d,
                cfg=RegimeARKXConfig(k=10, ridge=1e-2, bins=(0.0, 0.03, 0.05), equalize_events=True),
            )
            model2_kind = "split_1d2d_coupled_local_storage_inlet" if preset == "i" else "split_1d2d_coupled_local_inlet"
        model2_payload = {
            "kind": model2_kind,
            "n_1d": int(n2_1d),
            "n_2d": int(n2_2d),
            "parts": {
                "1d": {
                    "kind": "regime_arkx_exo_two_pass",
                    "exo_kind": "coupled_2d_local_state_v1_plus_storage_v1_plus_inlet_v1" if preset == "i" else "coupled_2d_local_state_v1_plus_inlet_v1",
                    "center_cell": torch.from_numpy(center_cell),
                    "neighbor_slots": torch.from_numpy(neighbor_slots),
                    "q_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [0.0, 0.03, 0.05],
                        "equalize_events": True,
                        "exo_dim": int(q_exo_1d[0].shape[2]),
                        "regime_step_counts": torch.from_numpy(countsq_1d),
                        "w": torch.from_numpy(wq_1d),
                    },
                    "y_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [float(x) for x in inlet_state_bins],
                        "equalize_events": True,
                        "exo_dim": int(y1_exo_2d[0].shape[2]),
                        "max_delta": float(0.4),
                        **({"regime_source": "inlet_positive"} if preset == "h" else {}),
                        "regime_step_counts": torch.from_numpy(counts2_1d),
                        "w": torch.from_numpy(w2_1d),
                    },
                    **(
                        {
                            "v_model": {
                                "kind": "regime_arkx_exo",
                                "k": 10,
                                "bins": [0.0, 0.03, 0.05],
                                "equalize_events": True,
                                "exo_dim": int(local_exo_2d[0].shape[2]),
                                "max_delta": torch.from_numpy(v_max_delta),
                                "regime_step_counts": torch.from_numpy(countsv_1d),
                                "w": torch.from_numpy(wv_1d),
                            }
                        }
                        if preset == "i"
                        else {}
                    ),
                },
                "2d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05, 0.16],
                    "equalize_events": False,
                    "regime_step_counts": torch.from_numpy(counts2_2d),
                    "w": torch.from_numpy(w2_2d),
                },
            },
        }
    elif preset in {"e", "f"}:
        seqs2_1d = [(y[:, :n2_1d], rain) for (y, rain) in seqs2]
        seqs2_2d = [(y[:, n2_1d:], rain) for (y, rain) in seqs2]
        graph2 = load_graph(model_root, model_id=2, split_for_static="train")
        train_events2 = [load_event(model_root, graph=graph2, split="train", event_id=eid, cache_dir=cache_dir) for eid in train2]
        inlet2 = [ev.inlet_1d.numpy() if ev.inlet_1d is not None else None for ev in train_events2]
        volume2 = [ev.volume_2d.numpy() if ev.volume_2d is not None else None for ev in train_events2]
        if any(x is None for x in inlet2):
            raise ValueError("preset e requires inlet_1d in training events")
        if any(x is None for x in volume2):
            raise ValueError("preset e requires volume_2d in training events")

        area_2d = graph2.area_2d.numpy().astype(np.float32, copy=False)
        conn_src_1d = graph2.conn_src_1d.numpy()
        conn_dst_2d = graph2.conn_dst_2d.numpy()
        conn_area_1d = connected_area_1d(
            area_2d,
            conn_src_1d=conn_src_1d,
            conn_dst_2d=conn_dst_2d,
            n_1d=n2_1d,
        ).astype(np.float32, copy=False)

        w2_2d, counts2_2d = fit_regime_arkx_per_node(
            seqs2_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05, 0.16)),
        )

        coupled_exo_2d: list[np.ndarray] = []
        for (y2_true, rain) in tqdm(seqs2_2d, desc="build model 2 coupled exo", leave=False):
            y2_pred = rollout_regime_arkx(w=w2_2d, bins=(0.0, 0.03, 0.05, 0.16), y_init=y2_true, rain=rain, warmup=cfg.warmup)
            coupled_exo_2d.append(
                build_coupled_1d_exo(
                    y2_2d=y2_pred,
                    rain_2d=rain,
                    area_2d=area_2d,
                    conn_src_1d=conn_src_1d,
                    conn_dst_2d=conn_dst_2d,
                    n_1d=n2_1d,
                )
            )

        vagg2_1d = [
            aggregate_2d_to_1d_sum(
                np.asarray(v, dtype=np.float32),
                conn_src_1d=conn_src_1d,
                conn_dst_2d=conn_dst_2d,
                n_1d=n2_1d,
            ).astype(np.float32, copy=False)
            for v in volume2
        ]
        wv_1d, countsv_1d = fit_regime_arkx_exo_per_node(
            [(v, rain) for v, (_, rain) in zip(vagg2_1d, seqs2_1d, strict=True)],
            exo_sequences=coupled_exo_2d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )
        dv_all = np.concatenate([np.abs(np.diff(v, axis=0)).astype(np.float32, copy=False) for v in vagg2_1d], axis=0)
        v_max_delta = np.maximum(np.quantile(dv_all, q=0.995, axis=0).astype(np.float32, copy=False), 1e-3)

        vagg_pred_1d: list[np.ndarray] = []
        for v_true, coupled_exo, (_, rain) in zip(vagg2_1d, coupled_exo_2d, seqs2_1d, strict=True):
            v_pred = rollout_regime_arkx_exo(
                w=wv_1d,
                bins=(0.0, 0.03, 0.05),
                y_init=v_true,
                rain=rain,
                exo=coupled_exo,
                warmup=cfg.warmup,
                max_delta=v_max_delta,
            )
            vagg_pred_1d.append(v_pred)

        q2_1d = [np.asarray(x, dtype=np.float32) for x in inlet2]
        q_exo_1d = [build_storage_augmented_exo(coupled_exo, v_pred, conn_area_1d) for coupled_exo, v_pred in zip(coupled_exo_2d, vagg_pred_1d, strict=True)]
        wq_1d, countsq_1d = fit_regime_arkx_exo_per_node(
            [(q, rain) for q, (_, rain) in zip(q2_1d, seqs2_1d, strict=True)],
            exo_sequences=q_exo_1d,
            cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
        )

        y1_exo_2d: list[np.ndarray] = []
        storage_depth_pred_1d: list[np.ndarray] = []
        for q_true, v_pred, coupled_exo, (_, rain) in zip(q2_1d, vagg_pred_1d, coupled_exo_2d, seqs2_1d, strict=True):
            q_pred = rollout_regime_arkx_exo(
                w=wq_1d,
                bins=(0.0, 0.03, 0.05),
                y_init=q_true,
                rain=rain,
                exo=build_storage_augmented_exo(coupled_exo, v_pred, conn_area_1d),
                warmup=cfg.warmup,
            )
            y1_exo_2d.append(build_inlet_storage_augmented_exo(coupled_exo, q_pred, v_pred, conn_area_1d))
            storage_depth_pred_1d.append((v_pred / np.maximum(conn_area_1d[None, :], 1.0)).astype(np.float32, copy=False))

        dy_all = np.concatenate([np.abs(np.diff(y1, axis=0)).astype(np.float32, copy=False) for y1, _ in seqs2_1d], axis=0)
        y_max_delta = np.maximum(np.quantile(dy_all, q=0.995, axis=0).astype(np.float32, copy=False), 1e-3)

        if preset == "f":
            y_state_bins = _strict_increasing_bins(
                np.concatenate([d[cfg.warmup :] for d in storage_depth_pred_1d], axis=0),
                probs=(0.50, 0.90),
                floor=0.0,
            )
            w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
                seqs2_1d,
                exo_sequences=y1_exo_2d,
                regime_signal_sequences=storage_depth_pred_1d,
                cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=y_state_bins, equalize_events=True),
            )
            model2_kind = "split_1d2d_coupled_storage_inlet_statebin"
        else:
            y_state_bins = (0.0, 0.03, 0.05)
            w2_1d, counts2_1d = fit_regime_arkx_exo_per_node(
                seqs2_1d,
                exo_sequences=y1_exo_2d,
                cfg=RegimeARKXConfig(k=10, ridge=1e-3, bins=(0.0, 0.03, 0.05), equalize_events=True),
            )
            model2_kind = "split_1d2d_coupled_storage_inlet"

        model2_payload = {
            "kind": model2_kind,
            "n_1d": int(n2_1d),
            "n_2d": int(n2_2d),
            "parts": {
                "1d": {
                    "kind": "regime_arkx_exo_three_pass",
                    "exo_kind": "coupled_2d_state_v1_plus_storage_v1_plus_inlet_v1",
                    "v_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [0.0, 0.03, 0.05],
                        "equalize_events": True,
                        "exo_dim": int(coupled_exo_2d[0].shape[2]),
                        "max_delta": torch.from_numpy(v_max_delta),
                        "regime_step_counts": torch.from_numpy(countsv_1d),
                        "w": torch.from_numpy(wv_1d),
                    },
                    "q_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [0.0, 0.03, 0.05],
                        "equalize_events": True,
                        "exo_dim": int(q_exo_1d[0].shape[2]),
                        "regime_step_counts": torch.from_numpy(countsq_1d),
                        "w": torch.from_numpy(wq_1d),
                    },
                    "y_model": {
                        "kind": "regime_arkx_exo",
                        "k": 10,
                        "bins": [float(x) for x in y_state_bins],
                        "equalize_events": True,
                        "exo_dim": int(y1_exo_2d[0].shape[2]),
                        "max_delta": torch.from_numpy(y_max_delta),
                        **({"regime_source": "storage_depth"} if preset == "f" else {}),
                        "regime_step_counts": torch.from_numpy(counts2_1d),
                        "w": torch.from_numpy(w2_1d),
                    },
                },
                "2d": {
                    "kind": "regime_arkx",
                    "k": 10,
                    "bins": [0.0, 0.03, 0.05, 0.16],
                    "equalize_events": False,
                    "regime_step_counts": torch.from_numpy(counts2_2d),
                    "w": torch.from_numpy(w2_2d),
                },
            },
        }
    else:
        raise ValueError(f"unknown preset: {preset}")

    payload = {
        "kind": "baseline_arx",
        "cfg": asdict(cfg),
        "split": split,
        "model_1": {"n_1d": int(n1_1d), "n_2d": int(n1_2d), "w": torch.from_numpy(w1)},
        "model_2": model2_payload,
    }

    # Optional validation
    if val1:
        m1, m1_1d, m1_2d = _eval_model(
            model_root=model_root,
            cache_dir=cache_dir,
            model_id=1,
            val_ids=val1,
            warmup=cfg.warmup,
            w1=w1,
            model2_kind=model2_kind,
            model2_payload={"w": None, "bins": []},
        )
        print(f"[val] model 1: mean={m1:.6f} mean1d={m1_1d:.6f} mean2d={m1_2d:.6f}")
    else:
        m1 = float("nan")

    if val2:
        m2, m2_1d, m2_2d = _eval_model(
            model_root=model_root,
            cache_dir=cache_dir,
            model_id=2,
            val_ids=val2,
            warmup=cfg.warmup,
            w1=w1,
            model2_kind=model2_kind,
            model2_payload={
                "w": (payload["model_2"]["w"].numpy() if model2_kind == "regime_arkx" else None),
                "bins": payload["model_2"].get("bins", []),
                "parts": payload["model_2"].get("parts", None),
            }
            if model2_kind in {"split_1d2d", "split_1d2d_coupled", "split_1d2d_coupled_inlet", "split_1d2d_coupled_local_inlet", "split_1d2d_coupled_local_inlet_statebin", "split_1d2d_coupled_local_storage_inlet", "split_1d2d_coupled_storage_inlet", "split_1d2d_coupled_storage_inlet_statebin"}
            else {"w": payload["model_2"]["w"].numpy(), "bins": payload["model_2"]["bins"]},
        )
        print(f"[val] model 2: mean={m2:.6f} mean1d={m2_1d:.6f} mean2d={m2_2d:.6f}")
    else:
        m2 = float("nan")

    if np.isfinite(m1) and np.isfinite(m2):
        print(f"[val] overall proxy = {0.5 * (m1 + m2):.6f}")

    torch.save(payload, out_path)
    out_path.with_suffix(".json").write_text(json.dumps({"cfg": payload["cfg"], "model2_kind": model2_kind}, indent=2) + "\n")
    print(f"saved baseline to {out_path}")


if __name__ == "__main__":
    main()
