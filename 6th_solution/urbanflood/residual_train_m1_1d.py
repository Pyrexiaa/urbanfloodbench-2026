# #####
# residual_train_m1_1d.py
# #####

# urbanflood/residual_train_m1_1d.py
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

from urbanflood.baseline import aggregate_2d_to_1d_mean, rollout_ar1x
from urbanflood.data import load_event, load_graph
from urbanflood.metric import STD_DEV_DICT
from urbanflood.residual import ResidualGRUConfig, ResidualNodeGRU
from urbanflood.residual_features import (
    augment_dyn_features_2d_v2_nbrmean,
    build_dyn_features_1d_v1,
    build_dyn_features_2d,
    build_static_features_m2_1d,
    build_static_features_m2_1d_v2,
)
from urbanflood.utils import seed_everything


@dataclass(frozen=True)
class ResidualM11DTrainCfg:
    baseline_ckpt: str
    resid2d_ckpt: str = ""
    pre_residual_ckpts: tuple[str, ...] = ()
    stage: int = 1

    model_root: str = "Models"
    cache_dir: str = ".cache/urbanflood"
    out_path: str = "runs/resid_m1_1d.pt"

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
    bidirectional: bool = False
    ema_decay: float = 0.0
    clamp: float = 1.0
    clamp_mode: str = "hard"

    static_aug_version: int = 1  # 0=raw, 1=aug1, 2=aug2
    cache_on_gpu: bool = True
    amp_bf16: bool = True
    max_events: int = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-ckpt", type=str, required=True)
    p.add_argument("--resid2d-ckpt", type=str, default="", help="Optional Model1 2D residual ckpt for coupling feats.")
    p.add_argument("--pre-residual-ckpt", type=str, nargs="*", default=[], help="Optional pre-Model1 1D residual ckpts (stack).")
    p.add_argument("--stage", type=int, default=1, help="Stage index for logging (1=base residual, 2+=boosters).")

    p.add_argument("--model-root", type=str, default="Models")
    p.add_argument("--cache-dir", type=str, default=".cache/urbanflood")
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional EMA decay in (0, 1). 0 disables EMA. When enabled, validation is reported using EMA weights.",
    )

    p.add_argument("--node-emb-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--nodewise-head", dest="nodewise_head", action="store_true")
    p.add_argument("--no-nodewise-head", dest="nodewise_head", action="store_false")
    p.set_defaults(nodewise_head=False)
    p.add_argument("--bidirectional", dest="bidirectional", action="store_true")
    p.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    p.set_defaults(bidirectional=False)
    p.add_argument("--clamp", type=float, default=1.0)
    p.add_argument("--clamp-mode", type=str, default="hard", choices=["hard", "tanh"])

    p.add_argument("--static-aug-version", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--cache-on-gpu", dest="cache_on_gpu", action="store_true")
    p.add_argument("--no-cache-on-gpu", dest="cache_on_gpu", action="store_false")
    p.set_defaults(cache_on_gpu=True)
    p.add_argument("--amp-bf16", dest="amp_bf16", action="store_true")
    p.add_argument("--no-amp-bf16", dest="amp_bf16", action="store_false")
    p.set_defaults(amp_bf16=True)
    p.add_argument("--max-events", type=int, default=0, help="Optional limit on number of events per split (smoke).")
    return p.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    raise ValueError(
        f"could not match node_static dim={expected_dim} (raw={raw.shape[1]} aug1={aug1.shape[1]} aug2={aug2.shape[1]})"
    )


def _apply_resid2d(
    *,
    resid2d: ResidualNodeGRU,
    node_static_2d: torch.Tensor,
    y2_base: np.ndarray,
    rain_2d: np.ndarray,
    bed2: np.ndarray,
    warmup: int,
    edge_index_2d: torch.LongTensor | None = None,
    edge_deg_inv_2d: torch.Tensor | None = None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    y2 = np.asarray(y2_base, dtype=np.float32).copy()
    x2 = build_dyn_features_2d(y2_base=y2, rain_2d=rain_2d, bed_2d=bed2, warmup=warmup)
    x2 = x2.to(device, dtype=amp_dtype) if amp_enabled else x2.to(device)
    dyn_dim = int(getattr(resid2d.cfg, "dyn_dim", 5))
    if dyn_dim in (7, 9):
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


def main() -> None:
    a = _parse_args()
    cfg = ResidualM11DTrainCfg(
        baseline_ckpt=str(a.baseline_ckpt),
        resid2d_ckpt=str(a.resid2d_ckpt),
        pre_residual_ckpts=tuple(str(x) for x in (a.pre_residual_ckpt or [])),
        stage=int(a.stage),
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
        bidirectional=bool(a.bidirectional),
        ema_decay=float(a.ema_decay),
        clamp=float(a.clamp),
        clamp_mode=str(a.clamp_mode),
        static_aug_version=int(a.static_aug_version),
        cache_on_gpu=bool(a.cache_on_gpu),
        amp_bf16=bool(a.amp_bf16),
        max_events=int(a.max_events),
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

    base = _load_torch(Path(cfg.baseline_ckpt))
    warmup = int(base["cfg"]["warmup"])
    split = base.get("split", None)
    if split is None:
        raise ValueError("baseline ckpt missing split")
    train_ids = list(split["model_1"]["train"])
    val_ids = list(split["model_1"]["val"])
    if cfg.max_events > 0:
        train_ids = train_ids[: cfg.max_events]
        val_ids = val_ids[: cfg.max_events]

    graph1 = load_graph(model_root, model_id=1, split_for_static="train")
    n1 = int(graph1.n_1d)
    n2 = int(graph1.n_2d)

    head_off = graph1.head_offset.cpu().numpy().astype(np.float32, copy=False)
    inv1 = head_off[:n1].astype(np.float32, copy=False)
    bed2 = head_off[n1:].astype(np.float32, copy=False)

    conn_src = graph1.conn_src_1d.cpu().numpy()
    conn_dst = graph1.conn_dst_2d.cpu().numpy()
    deg = np.bincount(conn_src, minlength=n1).astype(np.float32, copy=False)
    has_conn = (deg > 0).astype(np.float32, copy=False)
    bed_agg = aggregate_2d_to_1d_mean(bed2[None, :], conn_src_1d=conn_src, conn_dst_2d=conn_dst, n_1d=n1)[0].astype(
        np.float32, copy=False
    )

    nbr_src = graph1.edge_index_1d[0].cpu().numpy()
    nbr_dst = graph1.edge_index_1d[1].cpu().numpy()

    std_1d = float(STD_DEV_DICT[(1, 1)])
    std_2d = float(STD_DEV_DICT[(1, 2)])
    eps = 1e-12

    w = base["model_1"]["w"].cpu().numpy().astype(np.float32, copy=False)

    node_static_raw = graph1.node_static_1d.float().to(device)
    node_static_aug1 = build_static_features_m2_1d(graph1).to(device)
    node_static_aug2 = build_static_features_m2_1d_v2(graph1).to(device)

    if int(cfg.static_aug_version) == 2:
        node_static_train = node_static_aug2
    elif int(cfg.static_aug_version) == 1:
        node_static_train = node_static_aug1
    else:
        node_static_train = node_static_raw

    resid2d = None
    node_static_2d = graph1.node_static_2d.float().to(device)
    edge_index_2d = None
    edge_deg_inv_2d = None
    if str(cfg.resid2d_ckpt):
        ck2d = _load_torch(Path(cfg.resid2d_ckpt))
        if str(ck2d.get("kind", "")) != "residual_m1_2d":
            raise ValueError("--resid2d-ckpt must be kind residual_m1_2d")
        mcfg2d = ResidualGRUConfig(**ck2d["model_cfg"])
        resid2d = ResidualNodeGRU(n_nodes=n2, cfg=mcfg2d).to(device)
        resid2d.load_state_dict(ck2d["state_dict"])
        resid2d.eval()
        need_edges = (len(resid2d.graph_mix) > 0) or (int(getattr(mcfg2d, "dyn_dim", 5)) in (7, 9))
        if need_edges:
            edge_index_2d = graph1.edge_index_2d.to(device)
            deg = torch.bincount(graph1.edge_index_2d[1].cpu(), minlength=int(n2)).clamp(min=1).to(torch.float32)
            edge_deg_inv_2d = (1.0 / deg).to(device)

    # Optional pre-1D residuals (stacked sequentially onto the AR1X base).
    pre_models: list[tuple[ResidualNodeGRU, torch.Tensor, bool]] = []
    for p in cfg.pre_residual_ckpts:
        pre = _load_torch(Path(p))
        if str(pre.get("kind", "")) != "residual_m1_1d":
            raise ValueError(f"pre-residual kind unsupported: {pre.get('kind')}")
        mcfg_pre = ResidualGRUConfig(**pre["model_cfg"])
        if int(mcfg_pre.dyn_dim) not in (10, 13):
            raise ValueError(f"unsupported pre-residual dyn_dim={mcfg_pre.dyn_dim} (expected 10 or 13)")
        use_nbr = int(mcfg_pre.dyn_dim) == 13
        m = ResidualNodeGRU(n_nodes=n1, cfg=mcfg_pre).to(device)
        m.load_state_dict(pre["state_dict"])
        m.eval()
        node_static_pre = _pick_node_static(
            expected_dim=int(mcfg_pre.static_dim),
            raw=node_static_raw,
            aug1=node_static_aug1,
            aug2=node_static_aug2,
        )
        pre_models.append((m, node_static_pre, use_nbr))

    def build_example(event_id: int) -> dict:
        ev = load_event(model_root, graph=graph1, split="train", event_id=event_id, cache_dir=cache_dir)
        y1_true = ev.y_1d.numpy().astype(np.float32, copy=False)
        y2_true = ev.y_2d.numpy().astype(np.float32, copy=False)
        rain = ev.rain_2d.numpy().astype(np.float32, copy=False)

        y_init = np.concatenate([y1_true, y2_true], axis=1)
        y_pred = rollout_ar1x(w=w, y_init=y_init, rain=rain, warmup=warmup)
        y1_base = y_pred[:, :n1]
        y2_base = y_pred[:, n1:]

        if resid2d is not None:
            y2_used = _apply_resid2d(
                resid2d=resid2d,
                node_static_2d=node_static_2d,
                y2_base=y2_base,
                rain_2d=rain,
                bed2=bed2,
                warmup=warmup,
                edge_index_2d=edge_index_2d,
                edge_deg_inv_2d=edge_deg_inv_2d,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        else:
            y2_used = y2_base

        err2 = (y2_used[warmup:] - y2_true[warmup:]) / std_2d
        score2d = float(np.mean(np.sqrt(np.mean(err2**2, axis=0) + eps)))

        y2_agg = aggregate_2d_to_1d_mean(
            y2_used,
            conn_src_1d=conn_src,
            conn_dst_2d=conn_dst,
            n_1d=n1,
        ).astype(np.float32, copy=False)

        # Apply stacked pre-residuals to build the correct base for this stage.
        y1_feat = y1_base.astype(np.float32, copy=True)
        for m, node_static_pre, use_nbr in pre_models:
            y1_feat = _apply_resid1d_v1(
                model=m,
                node_static_1d=node_static_pre,
                y1_base=y1_feat,
                y2_agg=y2_agg,
                rain_2d=rain,
                inv1=inv1,
                bed_agg=bed_agg,
                has_conn=has_conn,
                nbr_src=nbr_src if use_nbr else None,
                nbr_dst=nbr_dst if use_nbr else None,
                warmup=warmup,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        x_dyn = build_dyn_features_1d_v1(
            y1_base=y1_feat,
            y2_agg=y2_agg,
            rain_2d=rain,
            invert_1d=inv1,
            bed_agg_1d=bed_agg,
            has_conn_1d=has_conn,
            nbr_src=nbr_src,
            nbr_dst=nbr_dst,
            warmup=warmup,
        )

        base_tail = y1_feat[warmup:].astype(np.float32, copy=False)
        y_res = (y1_true[warmup:] - base_tail).astype(np.float32, copy=False)
        return {
            "event_id": int(event_id),
            "x_dyn": x_dyn,
            "y_res": torch.from_numpy(y_res),
            "base_tail": base_tail,
            "score2d": float(score2d),
        }

    t0 = time.time()
    train_cache = {eid: build_example(eid) for eid in tqdm(train_ids, desc="prep train (m1-1d)", leave=False)}
    val_cache = {eid: build_example(eid) for eid in tqdm(val_ids, desc="prep val (m1-1d)", leave=False)}
    print(f"prep: cached {len(train_cache)} train + {len(val_cache)} val events in {time.time()-t0:.1f}s")
    val_2d_const = float(np.mean([ex["score2d"] for ex in val_cache.values()])) if val_cache else float("nan")

    if bool(cfg.cache_on_gpu):
        for ex in train_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(device, non_blocking=True)
            ex["y_gpu"] = ex["y_res"].to(device, non_blocking=True)
        for ex in val_cache.values():
            x = ex["x_dyn"]
            ex["x_gpu"] = x.to(device, non_blocking=True, dtype=amp_dtype) if amp_enabled else x.to(device, non_blocking=True)
        print("cached train features on GPU")

    mcfg = ResidualGRUConfig(
        dyn_dim=int(train_cache[train_ids[0]]["x_dyn"].shape[-1]) if train_ids else 1,
        static_dim=int(node_static_train.shape[1]),
        node_emb_dim=int(cfg.node_emb_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        bidirectional=bool(cfg.bidirectional),
        nodewise_head=bool(cfg.nodewise_head),
        clamp=float(cfg.clamp),
        clamp_mode=str(cfg.clamp_mode),
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

    def save_ckpt(*, state_dict: dict[str, torch.Tensor], best_val_m1: float, best_epoch: int) -> None:
        payload = {
            "kind": "residual_m1_1d",
            "baseline_ckpt": str(cfg.baseline_ckpt),
            "resid2d_ckpt": str(cfg.resid2d_ckpt),
            "pre_residual_ckpts": [str(x) for x in cfg.pre_residual_ckpts],
            "cfg": asdict(cfg),
            "model_cfg": asdict(mcfg),
            "feat_base_dim": int(mcfg.dyn_dim),
            "warmup": int(warmup),
            "best_val_m1": float(best_val_m1),
            "best_epoch": int(best_epoch),
            "state_dict": state_dict,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        meta_path.write_text(json.dumps({"best_val_m1": best_val_m1, "best_epoch": best_epoch, "cfg": payload["cfg"]}, indent=2) + "\n")

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

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x_dyn, node_static_train)
            err = (pred.float() - y_res) / std_1d
            rmse_node = torch.sqrt(torch.mean(err**2, dim=0) + eps)
            loss = rmse_node.mean()

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
            vals_m1 = []
            with torch.no_grad():
                for ex in val_cache.values():
                    x_dyn = ex["x_gpu"] if bool(cfg.cache_on_gpu) else ex["x_dyn"].to(device)
                    base_tail = ex["base_tail"]
                    y_res_np = ex["y_res"].numpy()
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        pred = eval_model(x_dyn, node_static_train).detach().float().cpu().numpy()
                    y_pred = base_tail + pred
                    y_true = base_tail + y_res_np
                    err = (y_pred - y_true) / std_1d
                    rmse1d = float(np.mean(np.sqrt(np.mean(err**2, axis=0) + eps)))
                    vals_1d.append(rmse1d)
                    vals_m1.append(0.5 * (rmse1d + float(ex["score2d"])))
            val_1d = float(np.mean(vals_1d)) if vals_1d else float("nan")
            val_m1 = float(np.mean(vals_m1)) if vals_m1 else float("nan")
            dt = time.time() - t1
            if ema_model is not None:
                print(
                    f"epoch {epoch+1:03d}/{cfg.epochs} "
                    f"train={train_loss:.6f} "
                    f"val_m1_ema={val_m1:.6f} "
                    f"val_1d_ema={val_1d:.6f} "
                    f"val_2d={val_2d_const:.6f} "
                    f"dt={dt:.1f}s"
                )
            else:
                print(
                    f"epoch {epoch+1:03d}/{cfg.epochs} "
                    f"train={train_loss:.6f} "
                    f"val_m1={val_m1:.6f} "
                    f"val_1d={val_1d:.6f} "
                    f"val_2d={val_2d_const:.6f} "
                    f"dt={dt:.1f}s"
                )
            if np.isfinite(val_m1) and val_m1 < best_val:
                best_val = float(val_m1)
                best_epoch = int(epoch + 1)
                src = ema_model if ema_model is not None else model
                best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
                save_ckpt(state_dict=best_state, best_val_m1=best_val, best_epoch=best_epoch)
        else:
            dt = time.time() - t1
            print(f"epoch {epoch+1:03d}/{cfg.epochs} train={train_loss:.6f} dt={dt:.1f}s")

    if best_state is None:
        src = ema_model if ema_model is not None else model
        best_state = {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}
        best_epoch = int(cfg.epochs)
        best_val = float("nan")
    save_ckpt(state_dict=best_state, best_val_m1=best_val, best_epoch=best_epoch)
    print(f"saved Model1 1D residual to {out_path}")


if __name__ == "__main__":
    main()
