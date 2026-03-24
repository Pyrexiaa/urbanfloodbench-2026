# #####
# residual_train_2d.py
# #####

# urbanflood/residual_train_2d.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from urbanflood.baseline import aggregate_2d_to_1d_sum, predict_model2_from_baseline_ckpts, rollout_ar1x
from urbanflood.data import load_event, load_graph
from urbanflood.metric import STD_DEV_DICT
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import augment_dyn_features_2d_v2_nbrmean, build_dyn_features_2d
from urbanflood.utils import seed_everything


@dataclass(frozen=True)
class Residual2DTrainCfg:
    baseline_ckpts: tuple[str, ...]
    model_id: int = 2
    mixed_mode: str = "weighted_split_ns"
    alpha_1d: float = 0.9
    alpha_2d: float = 0.5
    dyn_feat: str = "v1"

    model_root: str = "Models"
    cache_dir: str = ".cache/urbanflood"
    out_path: str = "runs/resid_m2_2d.pt"

    seed: int = 42
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    node_emb_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    nodewise_head: bool = False
    graph_mix_k: int = 0
    graph_mix_post_k: int = 0
    graph_mix_dropout: float = 0.0
    ema_decay: float = 0.0
    clamp: float = 0.25
    clamp_mode: str = "hard"
    node_chunk: int = 0
    time_chunk: int = 0

    cache_on_gpu: bool = True
    amp_bf16: bool = True
    max_events: int = 0  # 0 = no limit (use all from split)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=int, default=2, choices=[1, 2], help="Which model to train the 2D residual for.")
    p.add_argument("--baseline-ckpt", type=str, nargs="+", required=True)
    p.add_argument(
        "--mixed-mode",
        type=str,
        default="weighted_split_ns",
        choices=["single", "weighted_split_ns"],
        help="How to combine baseline ckpts for the Model2 base forecast.",
    )
    p.add_argument("--alpha-1d", type=float, default=0.9)
    p.add_argument("--alpha-2d", type=float, default=0.5)
    p.add_argument(
        "--dyn-feat",
        type=str,
        default="v1",
        choices=["v1", "v2nbr", "v3cpl"],
        help=(
            "Dynamic feature set for the 2D residual. "
            "v2nbr adds neighbor-mean coupling terms for depth/ddepth. "
            "v3cpl adds sparse 1D→2D coupling features from baseline 1D head mapped to connected 2D nodes."
        ),
    )

    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, default="runs/resid_m2_2d.pt")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--node-emb-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--nodewise-head",
        dest="nodewise_head",
        action="store_true",
        help="Enable per-node linear readout head (extra capacity; can help small graphs; may overfit).",
    )
    p.add_argument("--no-nodewise-head", dest="nodewise_head", action="store_false")
    p.set_defaults(nodewise_head=False)
    p.add_argument("--graph-mix-k", type=int, default=0, help="Optional learned graph mixing steps (0 disables).")
    p.add_argument(
        "--graph-mix-post-k",
        type=int,
        default=0,
        help="Optional learned graph mixing steps applied after the GRU on the hidden sequence (0 disables).",
    )
    p.add_argument("--graph-mix-dropout", type=float, default=0.0, help="Dropout for graph mixing blocks.")
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional EMA decay in (0, 1). 0 disables EMA. When enabled, validation is reported using EMA weights.",
    )
    p.add_argument("--clamp", type=float, default=0.25)
    p.add_argument("--clamp-mode", type=str, default="hard", choices=["hard", "tanh"])
    p.add_argument(
        "--node-chunk",
        type=int,
        default=0,
        help="Optional node-chunk size for training/eval (reduces VRAM by splitting the node batch). 0 disables.",
    )
    p.add_argument(
        "--time-chunk",
        type=int,
        default=0,
        help="Optional time-chunk size for truncated BPTT (reduces VRAM by splitting the time dimension). 0 disables.",
    )

    p.add_argument("--cache-on-gpu", dest="cache_on_gpu", action="store_true")
    p.add_argument("--no-cache-on-gpu", dest="cache_on_gpu", action="store_false")
    p.set_defaults(cache_on_gpu=True)
    p.add_argument("--amp-bf16", dest="amp_bf16", action="store_true", help="Use bf16 autocast (no GradScaler).")
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false", help="Disable bf16 autocast.")
    p.set_defaults(amp_bf16=True)
    p.add_argument("--max-events", type=int, default=0, help="Optional limit on number of events per split (smoke).")
    return p.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.0
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    a = _parse_args()
    cfg = Residual2DTrainCfg(
        model_id=int(a.model_id),
        baseline_ckpts=tuple(str(x) for x in a.baseline_ckpt),
        mixed_mode=str(a.mixed_mode),
        alpha_1d=float(a.alpha_1d),
        alpha_2d=float(a.alpha_2d),
        dyn_feat=str(a.dyn_feat),
        model_root=str(a.model_root),
        cache_dir=str(a.cache_dir),
        out_path=str(a.out),
        seed=int(a.seed),
        epochs=int(a.epochs),
        lr=float(a.lr),
        weight_decay=float(a.weight_decay),
        grad_clip=float(a.grad_clip),
        node_emb_dim=int(a.node_emb_dim),
        hidden_dim=int(a.hidden_dim),
        num_layers=int(a.num_layers),
        dropout=float(a.dropout),
        nodewise_head=bool(a.nodewise_head),
        graph_mix_k=int(a.graph_mix_k),
        graph_mix_post_k=int(a.graph_mix_post_k),
        graph_mix_dropout=float(a.graph_mix_dropout),
        ema_decay=float(a.ema_decay),
        clamp=float(a.clamp),
        clamp_mode=str(a.clamp_mode),
        node_chunk=int(a.node_chunk),
        time_chunk=int(a.time_chunk),
        cache_on_gpu=bool(a.cache_on_gpu),
        amp_bf16=bool(a.amp_bf16),
        max_events=int(a.max_events),
    )
    ema_decay = float(cfg.ema_decay)
    use_ema = 0.0 < ema_decay < 1.0
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("--ema-decay must be in [0, 1)")
    if int(cfg.node_chunk) < 0:
        raise ValueError("--node-chunk must be >= 0")
    if int(cfg.node_chunk) > 0 and (int(cfg.graph_mix_k) > 0 or int(cfg.graph_mix_post_k) > 0):
        raise ValueError("--node-chunk is incompatible with graph mixing (--graph-mix-k/--graph-mix-post-k > 0)")
    if int(cfg.time_chunk) < 0:
        raise ValueError("--time-chunk must be >= 0")
    if int(cfg.time_chunk) == 1:
        raise ValueError("--time-chunk=1 is not supported (use 0 to disable)")
    if int(cfg.time_chunk) > 0 and int(cfg.node_chunk) > 0:
        raise ValueError("--time-chunk is incompatible with --node-chunk (pick one)")

    seed_everything(cfg.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16
    amp_enabled = bool(cfg.amp_bf16)

    model_root = Path(cfg.model_root)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")

    ckpts = [_load_torch(Path(p)) for p in cfg.baseline_ckpts]
    warmup = int(ckpts[0]["cfg"]["warmup"])
    split = ckpts[0].get("split", None)
    if split is None:
        raise ValueError("baseline ckpt missing split; train baselines with --val-ratio>0 or use baseline_train preset output")
    for c in ckpts[1:]:
        if int(c["cfg"]["warmup"]) != warmup:
            raise ValueError("all baseline checkpoints must share warmup")
        if c.get("split", None) != split:
            raise ValueError("all baseline checkpoints must share the same split (train/val events)")

    model_id = int(cfg.model_id)
    split_key = f"model_{model_id}"
    train_ids = list(split[split_key]["train"])
    val_ids = list(split[split_key]["val"])
    if cfg.max_events > 0:
        train_ids = train_ids[: cfg.max_events]
        val_ids = val_ids[: cfg.max_events]

    graph = load_graph(model_root, model_id=model_id, split_for_static="train")
    n1 = int(graph.n_1d)
    n2 = int(graph.n_2d)

    head_off = graph.head_offset.cpu().numpy().astype(np.float32, copy=False)
    bed2 = head_off[n1:].astype(np.float32, copy=False)
    conn_src_1d = graph.conn_src_1d.cpu().numpy().astype(np.int64, copy=False)
    conn_dst_2d = graph.conn_dst_2d.cpu().numpy().astype(np.int64, copy=False)
    deg2 = np.bincount(conn_dst_2d, minlength=int(n2)).astype(np.float32, copy=False)
    has_conn2 = (deg2 > 0).astype(np.float32, copy=False)
    deg2_safe = np.maximum(deg2, 1.0).astype(np.float32, copy=False)

    node_static_2d = graph.node_static_2d.float().to(device)
    edge_deg_inv_cpu = None
    if str(cfg.dyn_feat) == "v2nbr":
        deg = torch.bincount(graph.edge_index_2d[1], minlength=int(n2)).clamp(min=1).to(torch.float32)
        edge_deg_inv_cpu = 1.0 / deg

    std_2d = float(STD_DEV_DICT[(model_id, 2)])
    eps = 1e-12

    w1 = None
    if int(model_id) == 1:
        w1 = ckpts[0]["model_1"]["w"].cpu().numpy().astype(np.float32, copy=False)

    def agg_1d_to_2d_mean(y1: np.ndarray) -> np.ndarray:
        y1 = np.asarray(y1, dtype=np.float32)
        if y1.ndim != 2 or int(y1.shape[1]) != int(n1):
            raise ValueError("y1 must be [T, N1]")
        T = int(y1.shape[0])
        out = np.zeros((T, int(n2)), dtype=np.float32)
        vals = y1[:, conn_src_1d]  # [T, Nc]
        for j, d in enumerate(conn_dst_2d.tolist()):
            out[:, d] += vals[:, j]
        out = out / deg2_safe[None, :]
        return out

    def build_example(event_id: int) -> dict:
        ev = load_event(model_root, graph=graph, split="train", event_id=event_id, cache_dir=cache_dir)
        y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
        y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)

        if int(model_id) == 2:
            y1_base, y2_base = predict_model2_from_baseline_ckpts(
                ckpts,
                graph2=graph,
                mixed_mode=str(cfg.mixed_mode),
                alpha_1d=float(cfg.alpha_1d),
                alpha_2d=float(cfg.alpha_2d),
                y1_init=y1_true,
                y2_init=y2_true,
                rain_2d=rain,
                q1_init=(ev.inlet_1d.numpy().astype(np.float32, copy=False) if ev.inlet_1d is not None else None),
                vagg_init=(
                    aggregate_2d_to_1d_sum(
                        ev.volume_2d.numpy().astype(np.float32, copy=False),
                        conn_src_1d=graph.conn_src_1d.numpy(),
                        conn_dst_2d=graph.conn_dst_2d.numpy(),
                        n_1d=int(graph.n_1d),
                    ).astype(np.float32, copy=False)
                    if ev.volume_2d is not None
                    else None
                ),
                warmup=warmup,
            )
        else:
            # Model 1 baseline is a single AR1X over (1D+2D) nodes.
            assert w1 is not None
            y_init = np.concatenate([y1_true, y2_true], axis=1)
            y_pred = rollout_ar1x(w=w1, y_init=y_init, rain=rain, warmup=warmup)
            y2_base = y_pred[:, n1:]
            y1_base = None

        x_dyn = build_dyn_features_2d(y2_base=y2_base, rain_2d=rain, bed_2d=bed2, warmup=warmup)  # [T', N2, 5]
        if str(cfg.dyn_feat) == "v3cpl":
            if int(model_id) != 2 or y1_base is None:
                raise ValueError("dyn-feat v3cpl is only supported for model_id=2")
            T = int(y2_base.shape[0])
            idx = np.arange(int(warmup), int(T), dtype=np.int64)
            # Mean connected 1D head mapped to 2D nodes (sparse; only ~174 nodes have any connection).
            y1_to_2 = agg_1d_to_2d_mean(y1_base)  # [T, N2]
            diff = (y1_to_2 - y2_base) * has_conn2[None, :]  # [T, N2], 0 for non-connected nodes
            diff_t = diff[idx]
            diff_tm1 = diff[idx - 1]
            ddiff = diff_t - diff_tm1
            cpl = np.stack([diff_t, ddiff], axis=-1).astype(np.float32, copy=False)  # [T', N2, 2]
            x_dyn = torch.cat([x_dyn, torch.from_numpy(cpl)], dim=-1)
        if str(cfg.dyn_feat) == "v2nbr":
            x_dyn = augment_dyn_features_2d_v2_nbrmean(
                x_dyn,
                edge_index=graph.edge_index_2d,
                edge_deg_inv=edge_deg_inv_cpu,
                include_diff=True,
            )
        y_res = (y2_true[warmup:] - y2_base[warmup:]).astype(np.float32, copy=False)
        return {
            "event_id": int(event_id),
            "x_dyn": x_dyn,
            "y_res": torch.from_numpy(y_res),
        }

    t0 = time.time()
    train_cache = {eid: build_example(eid) for eid in tqdm(train_ids, desc="prep train (m2-2d)", leave=False)}
    val_cache = {eid: build_example(eid) for eid in tqdm(val_ids, desc="prep val (m2-2d)", leave=False)}
    print(f"prep: cached {len(train_cache)} train + {len(val_cache)} val events in {time.time()-t0:.1f}s")

    if bool(cfg.cache_on_gpu):
        for ex in train_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(
                device, non_blocking=True
            )
            ex["y_gpu"] = ex["y_res"].to(device, non_blocking=True)
        for ex in val_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(
                device, non_blocking=True
            )
            # y_res kept on CPU for val to reduce VRAM pressure
        print("cached train features on GPU")

    dyn_dim = 5
    if train_cache:
        dyn_dim = int(next(iter(train_cache.values()))["x_dyn"].shape[-1])

    mcfg = ResidualGRUConfig(
        dyn_dim=int(dyn_dim),
        static_dim=int(graph.node_static_2d.shape[1]),
        node_emb_dim=int(cfg.node_emb_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        bidirectional=False,
        nodewise_head=bool(cfg.nodewise_head),
        graph_mix_k=int(cfg.graph_mix_k),
        graph_mix_post_k=int(cfg.graph_mix_post_k),
        graph_mix_dropout=float(cfg.graph_mix_dropout),
        clamp=float(cfg.clamp),
        clamp_mode=str(cfg.clamp_mode),
    )
    model = ResidualNodeGRU(n_nodes=n2, cfg=mcfg).to(device)
    ema_model = None
    if use_ema:
        ema_model = ResidualNodeGRU(n_nodes=n2, cfg=mcfg).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    opt = AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    edge_index_2d = None
    edge_deg_inv_2d = None
    if int(cfg.graph_mix_k) > 0 or int(cfg.graph_mix_post_k) > 0:
        edge_index_2d = graph.edge_index_2d.to(device)
        deg = torch.bincount(graph.edge_index_2d[1].cpu(), minlength=int(n2)).clamp(min=1).to(torch.float32)
        edge_deg_inv_2d = (1.0 / deg).to(device)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    def save_ckpt(*, state_dict: dict[str, torch.Tensor], best_val_2d: float, best_epoch: int) -> None:
        payload = {
            "kind": f"residual_m{model_id}_2d",
            "baseline_ckpts": [str(x) for x in cfg.baseline_ckpts],
            "cfg": asdict(cfg),
            "model_cfg": asdict(mcfg),
            "feat_base_dim": int(mcfg.dyn_dim),
            "warmup": int(warmup),
            "best_val_2d": float(best_val_2d),
            "best_epoch": int(best_epoch),
            "state_dict": state_dict,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        meta_path.write_text(json.dumps({"best_val_2d": best_val_2d, "best_epoch": best_epoch, "cfg": payload["cfg"]}, indent=2) + "\n")

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
                y_res = ex["y_gpu"]
            else:
                x = ex["x_dyn"]
                x_dyn = x.to(device, dtype=amp_dtype) if amp_enabled else x.to(device)
                y_res = ex["y_res"].to(device)

            opt.zero_grad(set_to_none=True)
            if int(cfg.time_chunk) > 0:
                t_chunk = int(cfg.time_chunk)
                t_total = int(x_dyn.shape[0])
                h0 = None
                loss_acc = 0.0
                for s in range(0, t_total, t_chunk):
                    e = min(t_total, s + t_chunk)
                    x_c = x_dyn[s:e]
                    y_c = y_res[s:e]
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        pred_c, h_T = model(
                            x_c,
                            node_static_2d,
                            h0=h0,
                            edge_index=edge_index_2d,
                            edge_deg_inv=edge_deg_inv_2d,
                            return_h=True,
                        )  # [Tc, N2]
                    # Truncated BPTT: stop gradients across time chunks.
                    h0 = h_T.detach()
                    err = (pred_c.float() - y_c) / std_2d
                    # Use per-node MSE for streaming additivity (close proxy to RMSE objective).
                    mse_node = torch.mean(err**2, dim=0)  # [N2]
                    loss_c = mse_node.mean()
                    w = float(e - s) / float(max(1, t_total))
                    (loss_c * w).backward()
                    loss_acc += float(loss_c.detach().cpu().item()) * w
                loss_val = float(loss_acc)
            elif int(cfg.node_chunk) > 0:
                node_chunk = int(cfg.node_chunk)
                n_total = int(n2)
                loss_sum = 0.0
                for s in range(0, n_total, node_chunk):
                    e = min(n_total, s + node_chunk)
                    x_c = x_dyn[:, s:e, :]
                    y_c = y_res[:, s:e]
                    ns_c = node_static_2d[s:e]
                    node_ids = torch.arange(s, e, device=device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        pred_c = model(
                            x_c,
                            ns_c,
                            node_ids=node_ids,
                            edge_index=edge_index_2d,
                            edge_deg_inv=edge_deg_inv_2d,
                        )  # [T', Nc]
                    err = (pred_c.float() - y_c) / std_2d
                    rmse_node = torch.sqrt(torch.mean(err**2, dim=0) + eps)  # [Nc]
                    loss = rmse_node.sum() / float(n_total)
                    loss.backward()
                    loss_sum += float(loss.detach().cpu().item())
                loss_val = float(loss_sum)
            else:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    pred = model(x_dyn, node_static_2d, edge_index=edge_index_2d, edge_deg_inv=edge_deg_inv_2d)  # [T', N2]
                err = (pred.float() - y_res) / std_2d
                rmse_node = torch.sqrt(torch.mean(err**2, dim=0) + eps)
                loss = rmse_node.mean()
                loss.backward()
                loss_val = float(loss.detach().cpu().item())

            if float(cfg.grad_clip) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
            opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=True):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            losses.append(loss_val)

        train_loss = float(np.mean(losses)) if losses else float("nan")

        if val_cache:
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            vals = []
            with torch.no_grad():
                for ex in val_cache.values():
                    x_dyn = ex["x_gpu"] if bool(cfg.cache_on_gpu) else ex["x_dyn"].to(device)
                    y_true = ex["y_res"].numpy()  # [T', N2]
                    if int(cfg.time_chunk) > 0:
                        t_chunk = int(cfg.time_chunk)
                        t_total = int(y_true.shape[0])
                        pred = np.empty((t_total, int(n2)), dtype=np.float32)
                        h0 = None
                        for s in range(0, t_total, t_chunk):
                            e = min(t_total, s + t_chunk)
                            x_c = x_dyn[s:e]
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                                pred_c, h_T = eval_model(
                                    x_c,
                                    node_static_2d,
                                    h0=h0,
                                    edge_index=edge_index_2d,
                                    edge_deg_inv=edge_deg_inv_2d,
                                    return_h=True,
                                )
                            pred[s:e] = pred_c.detach().float().cpu().numpy()
                            h0 = h_T.detach()
                    elif int(cfg.node_chunk) > 0:
                        node_chunk = int(cfg.node_chunk)
                        n_total = int(n2)
                        pred = np.empty((int(y_true.shape[0]), n_total), dtype=np.float32)
                        for s in range(0, n_total, node_chunk):
                            e = min(n_total, s + node_chunk)
                            x_c = x_dyn[:, s:e, :]
                            ns_c = node_static_2d[s:e]
                            node_ids = torch.arange(s, e, device=device)
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                                pred_c = (
                                    eval_model(
                                        x_c,
                                        ns_c,
                                        node_ids=node_ids,
                                        edge_index=edge_index_2d,
                                        edge_deg_inv=edge_deg_inv_2d,
                                    )
                                    .detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                )
                            pred[:, s:e] = pred_c
                    else:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                            pred = (
                                eval_model(x_dyn, node_static_2d, edge_index=edge_index_2d, edge_deg_inv=edge_deg_inv_2d)
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                            )
                    err = (pred - y_true) / std_2d
                    rmse = float(np.mean(np.sqrt(np.mean(err**2, axis=0) + eps)))
                    vals.append(rmse)
            val_2d = float(np.mean(vals)) if vals else float("nan")
            dt = time.time() - t1
            if ema_model is not None:
                print(f"epoch {epoch+1:03d}/{cfg.epochs} train={train_loss:.6f} val_2d_ema={val_2d:.6f} dt={dt:.1f}s")
            else:
                print(f"epoch {epoch+1:03d}/{cfg.epochs} train={train_loss:.6f} val_2d={val_2d:.6f} dt={dt:.1f}s")
            if np.isfinite(val_2d) and val_2d < best_val:
                best_val = float(val_2d)
                best_epoch = int(epoch + 1)
                src = ema_model if ema_model is not None else model
                best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
                save_ckpt(state_dict=best_state, best_val_2d=best_val, best_epoch=best_epoch)
        else:
            dt = time.time() - t1
            print(f"epoch {epoch+1:03d}/{cfg.epochs} train={train_loss:.6f} dt={dt:.1f}s")

    if best_state is None:
        src = ema_model if ema_model is not None else model
        best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
        best_epoch = int(cfg.epochs)
        best_val = float("nan")
    save_ckpt(state_dict=best_state, best_val_2d=best_val, best_epoch=best_epoch)
    print(f"saved 2D residual to {out_path}")


if __name__ == "__main__":
    main()
