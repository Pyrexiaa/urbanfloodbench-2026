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
class InletAuxCfg:
    model_id: int = 2
    warmup: int = 10
    k: int = 10
    ridge: float = 1e-3
    bins: tuple[float, ...] = (0.0, 0.03, 0.05)
    equalize_events: bool = True
    use_surface_slots: bool = True
    use_local2d_slots: bool = False


def build_inlet_local_surface_exo(
    *,
    coupled_exo: np.ndarray,  # [T, N1, D]
    surface_slot_feats: np.ndarray | None = None,  # [T, N1, Ds]
    local2d_node_feats: np.ndarray | None = None,  # [T, N1, Dl]
    use_surface_slots: bool = True,
    use_local2d_slots: bool = False,
) -> np.ndarray:
    coupled = np.asarray(coupled_exo, dtype=np.float32)
    if coupled.ndim != 3:
        raise ValueError("coupled_exo must be [T, N, D]")
    parts: list[np.ndarray] = [coupled]

    if bool(use_surface_slots):
        if surface_slot_feats is None:
            raise ValueError("surface_slot_feats are required when use_surface_slots=True")
        surf = np.asarray(surface_slot_feats, dtype=np.float32)
        if surf.ndim != 3 or surf.shape[:2] != coupled.shape[:2]:
            raise ValueError("surface_slot_feats must be [T, N, D] aligned to coupled_exo")
        parts.append(surf)

    if bool(use_local2d_slots):
        if local2d_node_feats is None:
            raise ValueError("local2d_node_feats are required when use_local2d_slots=True")
        local = np.asarray(local2d_node_feats, dtype=np.float32)
        if local.ndim != 3 or local.shape[:2] != coupled.shape[:2]:
            raise ValueError("local2d_node_feats must be [T, N, D] aligned to coupled_exo")
        parts.append(local)

    return np.concatenate(parts, axis=2).astype(np.float32, copy=False)


def fit_inlet_aux(
    *,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    exo_sequences: list[np.ndarray],
    cfg: InletAuxCfg,
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


def predict_inlet_from_ckpt(
    *,
    ckpt: dict,
    y2_2d: np.ndarray,
    rain_2d: np.ndarray,
    area_2d: np.ndarray,
    conn_src_1d: np.ndarray,
    conn_dst_2d: np.ndarray,
    n_1d: int,
    q1_init: np.ndarray,
    warmup: int,
    surface_slot_feats: np.ndarray | None = None,
    local2d_node_feats: np.ndarray | None = None,
) -> np.ndarray:
    kind = str((ckpt.get("model_2", {}) or {}).get("kind", ckpt.get("kind", "")))
    if kind != "aux_inlet_m2_1d_surface_slots":
        raise ValueError("inlet aux ckpt must be kind aux_inlet_m2_1d_surface_slots")
    model_1d = ((ckpt.get("model_2", {}) or {}).get("parts", {}) or {}).get("1d", {})
    q_model = model_1d.get("q_model", None)
    if q_model is None:
        raise ValueError("aux inlet ckpt missing model_2.parts.1d.q_model")

    coupled_exo = build_coupled_1d_exo(
        y2_2d=y2_2d,
        rain_2d=rain_2d,
        area_2d=area_2d,
        conn_src_1d=conn_src_1d,
        conn_dst_2d=conn_dst_2d,
        n_1d=n_1d,
    )
    exo = build_inlet_local_surface_exo(
        coupled_exo=coupled_exo,
        surface_slot_feats=surface_slot_feats,
        local2d_node_feats=local2d_node_feats,
        use_surface_slots=bool(ckpt.get("use_surface_slots", True)),
        use_local2d_slots=bool(ckpt.get("use_local2d_slots", False)),
    )
    max_delta_raw = q_model.get("max_delta", None)
    if hasattr(max_delta_raw, "numpy"):
        max_delta = max_delta_raw.numpy().astype(np.float32, copy=False)
    elif max_delta_raw is None:
        max_delta = None
    else:
        max_delta = np.asarray(max_delta_raw, dtype=np.float32)
        if max_delta.ndim == 0:
            max_delta = float(max_delta)

    return rollout_regime_arkx_exo(
        w=q_model["w"].numpy() if hasattr(q_model["w"], "numpy") else np.asarray(q_model["w"], dtype=np.float32),
        bins=tuple(float(x) for x in q_model["bins"]),
        y_init=np.asarray(q1_init, dtype=np.float32),
        rain=np.asarray(rain_2d, dtype=np.float32),
        exo=exo,
        warmup=int(warmup),
        max_delta=max_delta,
    )


def build_payload(
    *,
    cfg: InletAuxCfg,
    split: dict,
    w: np.ndarray,
    counts: np.ndarray,
    exo_dim: int,
    node_count: int,
    source_baseline_ckpts: list[str],
    source_resid2d_ckpt: str,
    source_resid2d_coupling_ckpt: str,
    source_resid2d_coupling_blend: float,
    source_surfaceflow_ckpt: str,
    max_delta: float | np.ndarray | None = None,
    metrics: dict | None = None,
) -> dict:
    q_model = {
        "kind": "regime_arkx_exo",
        "k": int(cfg.k),
        "bins": [float(x) for x in cfg.bins],
        "equalize_events": bool(cfg.equalize_events),
        "exo_dim": int(exo_dim),
        "regime_step_counts": counts,
        "w": w,
    }
    if max_delta is not None:
        q_model["max_delta"] = max_delta

    payload = {
        "kind": "aux_inlet_m2_1d_surface_slots",
        "cfg": asdict(cfg),
        "split": split,
        "exo_kind": "coupled_2d_state_v1_plus_surface_slots_v1",
        "exo_dim": int(exo_dim),
        "node_count": int(node_count),
        "use_surface_slots": bool(cfg.use_surface_slots),
        "use_local2d_slots": bool(cfg.use_local2d_slots),
        "source_baseline_ckpts": [str(x) for x in source_baseline_ckpts],
        "source_resid2d_ckpt": str(source_resid2d_ckpt),
        "source_resid2d_coupling_ckpt": str(source_resid2d_coupling_ckpt),
        "source_resid2d_coupling_blend": float(source_resid2d_coupling_blend),
        "source_surfaceflow_ckpt": str(source_surfaceflow_ckpt),
        "model_2": {
            "kind": "aux_inlet_m2_1d_surface_slots",
            "n_1d": int(node_count),
            "parts": {"1d": {"kind": "regime_arkx_exo_one_pass", "q_model": q_model}},
        },
    }
    if metrics:
        payload["metrics"] = metrics
    return payload
