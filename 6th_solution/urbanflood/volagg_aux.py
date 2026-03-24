from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from urbanflood.baseline import (
    RegimeARKXConfig,
    build_coupled_1d_exo,
    fit_regime_arkx_exo_per_node,
    rollout_regime_arkx_exo,
)


@dataclass(frozen=True)
class VolAggAuxCfg:
    model_id: int = 2
    warmup: int = 10
    k: int = 10
    ridge: float = 1e-3
    bins: tuple[float, ...] = (0.0, 0.03, 0.05)
    equalize_events: bool = True


def build_volagg_exo(
    *,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
) -> np.ndarray:
    return build_coupled_1d_exo(
        y2_2d=np.asarray(y2_2d, dtype=np.float32),
        rain_2d=np.asarray(rain_2d, dtype=np.float32),
        area_2d=np.asarray(area_2d, dtype=np.float32),
        conn_src_1d=np.asarray(conn_src_1d, dtype=np.int64),
        conn_dst_2d=np.asarray(conn_dst_2d, dtype=np.int64),
        n_1d=int(n_1d),
    )


def fit_volagg_aux(
    *,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    exo_sequences: list[np.ndarray],
    cfg: VolAggAuxCfg,
) -> tuple[np.ndarray, np.ndarray]:
    return fit_regime_arkx_exo_per_node(
        sequences,
        exo_sequences=exo_sequences,
        cfg=RegimeARKXConfig(
            k=int(cfg.k),
            ridge=float(cfg.ridge),
            bins=tuple(float(x) for x in cfg.bins),
            equalize_events=bool(cfg.equalize_events),
        ),
    )


def predict_volagg_from_ckpt(
    *,
    ckpt: dict,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
    vagg_init: np.ndarray,
    warmup: int,
) -> np.ndarray:
    if str(ckpt.get("kind", "")) != "aux_volagg_m2_1d":
        raise ValueError("volagg ckpt must be kind aux_volagg_m2_1d")
    exo = build_volagg_exo(
        y2_2d=y2_2d,
        rain_2d=rain_2d,
        area_2d=area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    max_delta_raw = ckpt.get("max_delta", None)
    if max_delta_raw is None:
        max_delta = None
    elif isinstance(max_delta_raw, np.ndarray):
        max_delta = max_delta_raw.astype(np.float32, copy=False)
    elif hasattr(max_delta_raw, "numpy"):
        max_delta = max_delta_raw.numpy().astype(np.float32, copy=False)
    else:
        max_delta = float(max_delta_raw)
    return rollout_regime_arkx_exo(
        w=ckpt["w"].numpy() if hasattr(ckpt["w"], "numpy") else np.asarray(ckpt["w"], dtype=np.float32),
        bins=tuple(float(x) for x in ckpt["bins"]),
        y_init=np.asarray(vagg_init, dtype=np.float32),
        rain=np.asarray(rain_2d, dtype=np.float32),
        exo=exo,
        warmup=int(warmup),
        max_delta=max_delta,
    )


def build_payload(
    *,
    cfg: VolAggAuxCfg,
    split: dict,
    w: np.ndarray,
    counts: np.ndarray,
    exo_dim: int,
    max_delta: float | np.ndarray | None = None,
    metrics: dict | None = None,
) -> dict:
    payload = {
        "kind": "aux_volagg_m2_1d",
        "cfg": asdict(cfg),
        "split": split,
        "exo_kind": "coupled_2d_state_v1",
        "exo_dim": int(exo_dim),
        "bins": [float(x) for x in cfg.bins],
        "w": w,
        "regime_step_counts": counts,
    }
    if max_delta is not None:
        payload["max_delta"] = max_delta
    if metrics:
        payload["metrics"] = metrics
    return payload
