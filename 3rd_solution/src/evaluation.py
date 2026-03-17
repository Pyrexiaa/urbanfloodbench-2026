"""Standardized RMSE evaluation metric for UrbanFloodBench.

Metric hierarchy:
  node-level RMSE → standardize by std_dev →
  avg per node_type (1D/2D equal weight) →
  avg per event → avg per model → final score
"""
import numpy as np
from typing import Dict, Tuple


def compute_node_rmse(
    pred: np.ndarray,   # [num_timesteps, num_nodes]
    truth: np.ndarray,  # [num_timesteps, num_nodes]
) -> np.ndarray:
    """Per-node RMSE across timesteps. Returns [num_nodes]."""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))


def compute_std_dev_from_train(
    water_levels: np.ndarray,  # [total_timesteps_across_events, num_nodes]
) -> np.ndarray:
    """Compute per-node std_dev from training data. Returns [num_nodes]."""
    return np.std(water_levels, axis=0)


def compute_standardized_rmse(
    pred: np.ndarray,       # [num_timesteps, num_nodes]
    truth: np.ndarray,      # [num_timesteps, num_nodes]
    std_dev: float,         # scalar std_dev per (model, node_type)
) -> float:
    """Standardized RMSE for one node_type in one event.

    Returns avg of (per-node RMSE / std_dev) across nodes.
    """
    node_rmse = compute_node_rmse(pred, truth)  # [num_nodes]
    # std_devが0のノード (dry nodes) はRMSE比率が爆発するので除外 or clamp
    safe_std = max(std_dev, 1e-8)
    standardized = node_rmse / safe_std
    return float(np.mean(standardized))


def compute_event_score(
    pred_1d: np.ndarray,    # [T, num_1d_nodes]
    truth_1d: np.ndarray,
    pred_2d: np.ndarray,    # [T, num_2d_nodes]
    truth_2d: np.ndarray,
    std_1d: float,
    std_2d: float,
) -> float:
    """Event-level score = avg(standardized_rmse_1d, standardized_rmse_2d)."""
    s_rmse_1d = compute_standardized_rmse(pred_1d, truth_1d, std_1d)
    s_rmse_2d = compute_standardized_rmse(pred_2d, truth_2d, std_2d)
    return (s_rmse_1d + s_rmse_2d) / 2.0


def compute_model_score(
    event_scores: list[float],
) -> float:
    """Model-level score = avg across all event scores."""
    return float(np.mean(event_scores))


def compute_final_score(
    model_scores: list[float],
) -> float:
    """Final score = avg across model scores."""
    return float(np.mean(model_scores))


def compute_std_from_all_events(
    data_dir: str,
    model_id: int,
    node_type: str,  # '1d' or '2d'
) -> float:
    """Compute std_dev of water_level across all training events for a (model, node_type).

    水位の全ノード×全時刻における標準偏差を計算する。
    """
    import pandas as pd
    import os

    model_dir = os.path.join(data_dir, f"Model_{model_id}", "train")
    events = sorted([d for d in os.listdir(model_dir) if d.startswith("event_")])

    all_wl = []
    for event in events:
        fname = f"{node_type}_nodes_dynamic_all.csv"
        fpath = os.path.join(model_dir, event, fname)
        df = pd.read_csv(fpath)
        all_wl.append(df["water_level"].values)

    concat = np.concatenate(all_wl)
    return float(np.std(concat))
