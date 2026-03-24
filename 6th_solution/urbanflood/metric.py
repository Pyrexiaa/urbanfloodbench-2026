# #####
# metric.py
# #####

# urbanflood/metric.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STD_DEV_DICT: dict[tuple[int, int], float] = {
    (1, 1): 16.877747,  # Model 1, 1D nodes
    (1, 2): 14.378797,  # Model 1, 2D nodes
    (2, 1): 3.191784,  # Model 2, 1D nodes
    (2, 2): 2.727131,  # Model 2, 2D nodes
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardized_rmse(y_true: np.ndarray, y_pred: np.ndarray, std_dev: float) -> float:
    if std_dev == 0.0 or np.isnan(std_dev):
        return float("nan")
    return rmse(y_true, y_pred) / float(std_dev)


@dataclass(frozen=True)
class EventScore:
    model_id: int
    event_id: int
    score: float
    score_1d: float
    score_2d: float


def score_event_from_full_grids(
    *,
    model_id: int,
    event_id: int,
    y_true_1d: np.ndarray,  # [T, N1]
    y_pred_1d: np.ndarray,  # [T, N1]
    y_true_2d: np.ndarray,  # [T, N2]
    y_pred_2d: np.ndarray,  # [T, N2]
    warmup: int = 10,
) -> EventScore:
    """
    Mirror the leaderboard structure closely:
    - node-wise RMSE over timesteps t>=warmup
    - standardize by (model_id, node_type) std dev
    - average nodes within type
    - average types equally
    """
    if y_true_1d.shape != y_pred_1d.shape:
        raise ValueError(f"1D shape mismatch: {y_true_1d.shape} vs {y_pred_1d.shape}")
    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(f"2D shape mismatch: {y_true_2d.shape} vs {y_pred_2d.shape}")

    t0 = int(warmup)
    if t0 < 0:
        raise ValueError("warmup must be >= 0")

    y_true_1d = np.asarray(y_true_1d, dtype=np.float64)
    y_pred_1d = np.asarray(y_pred_1d, dtype=np.float64)
    y_true_2d = np.asarray(y_true_2d, dtype=np.float64)
    y_pred_2d = np.asarray(y_pred_2d, dtype=np.float64)

    std_1d = float(STD_DEV_DICT[(model_id, 1)])
    std_2d = float(STD_DEV_DICT[(model_id, 2)])

    if y_true_1d.shape[0] <= t0 or y_true_2d.shape[0] <= t0:
        raise ValueError("event shorter than warmup window")

    # Be strict: any non-finite values in the evaluated window should be a hard failure.
    # The real competition scorer rejects NaNs/Infs in the submission.
    if not np.isfinite(y_true_1d[t0:]).all() or not np.isfinite(y_true_2d[t0:]).all():
        raise ValueError("ground truth contains non-finite values in evaluated window")
    if not np.isfinite(y_pred_1d[t0:]).all() or not np.isfinite(y_pred_2d[t0:]).all():
        return EventScore(
            model_id=int(model_id),
            event_id=int(event_id),
            score=float("inf"),
            score_1d=float("inf"),
            score_2d=float("inf"),
        )

    err_1d = (y_pred_1d[t0:] - y_true_1d[t0:]) / std_1d  # [T', N1]
    err_2d = (y_pred_2d[t0:] - y_true_2d[t0:]) / std_2d  # [T', N2]

    rmse_1d_per_node = np.sqrt(np.mean(err_1d**2, axis=0))  # [N1]
    rmse_2d_per_node = np.sqrt(np.mean(err_2d**2, axis=0))  # [N2]

    score_1d = float(np.mean(rmse_1d_per_node)) if rmse_1d_per_node.size else float("nan")
    score_2d = float(np.mean(rmse_2d_per_node)) if rmse_2d_per_node.size else float("nan")

    # Do not drop a node type. If one is non-finite, the event score is non-finite.
    if not np.isfinite(score_1d) or not np.isfinite(score_2d):
        score = float("inf")
        score_1d = float("inf") if not np.isfinite(score_1d) else score_1d
        score_2d = float("inf") if not np.isfinite(score_2d) else score_2d
    else:
        score = 0.5 * (score_1d + score_2d)

    return EventScore(
        model_id=int(model_id),
        event_id=int(event_id),
        score=score,
        score_1d=score_1d,
        score_2d=score_2d,
    )


def score_model_from_events(event_scores: Iterable[EventScore], *, model_id: int) -> float:
    vals_all = [es.score for es in event_scores if es.model_id == model_id]
    if not vals_all:
        return float("nan")
    if not np.isfinite(vals_all).all():
        return float("inf")
    return float(np.mean(vals_all))


def score_overall(event_scores: Iterable[EventScore]) -> float:
    s1 = score_model_from_events(event_scores, model_id=1)
    s2 = score_model_from_events(event_scores, model_id=2)
    if not np.isfinite(s1) or not np.isfinite(s2):
        return float("inf")
    return 0.5 * (float(s1) + float(s2))
