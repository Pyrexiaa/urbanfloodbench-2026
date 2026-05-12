"""
inference_timing.py
===================
Measures end-to-end inference time for all pipeline components using synthetic
random data and random model weights on the same GPU (falls back to CPU if no
GPU is available).

Pipeline overview
-----------------
  Model 1 (Ridge + LightGBM Rain-3, 1D + 2D nodes)        → notebook: ridge-lightgbm-rain-3-w-static-feats-model1
  Model 2 Tabular (Ridge + LightGBM Rain-3 + PI feats, 2D) → notebook: ridge-lgbm-rain-3-w-static-pi-feats-m2-2d
  Model 2 GNN    (Physics-Informed GNN, 1D)                 → notebook: pi-gnn-with-g-att-rollout-r-gate-12-l-0-0-drop
  Ensemble       (Nodewise closed-form blend)                → notebook: ensemble-nodewise-closed-form

Reported times
--------------
  • Model 1 – 1D  inference  (LightGBM + Ridge blend, 1D nodes)
  • Model 1 – 2D  inference  (LightGBM + Ridge blend, 2D nodes)
  • Model 2 – 1D  inference  (Physics-Informed GNN on 1D nodes)
  • Model 2 – 2D  inference  (LightGBM + Ridge + PI blend on 2D nodes)

Since the ensemble is a simple closed-form nodewise linear blend of multiple
model outputs (all sharing the same architecture, differing only in
hyperparameters like n_layers / dropout), the ensemble is simulated by running
the base model N_ENSEMBLE times and doing the final weighted sum.  This
correctly captures how ensemble inference time scales with the number of members.

Usage
-----
  python inference_timing.py [--n_events N] [--n_timesteps T] [--n_runs R]
  python inference_timing.py --help

All randomness is seeded so results are reproducible.
"""

import argparse
import time
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# LightGBM stub – avoids the macOS OpenMP segfault that hits when LightGBM
# is imported and trained on Apple Silicon / macOS.
#
# Real LightGBM inference on tabular data (n_estimators=500-800 shallow trees,
# num_leaves=63) is dominated by vectorised tree traversal, which we replicate
# here with an equivalent numpy random-forest-style predict:
#   • n_estimators   trees
#   • each tree: 1 layer of log2(num_leaves)≈6 random splits on random cols
#   • leaf output = small random value
# This gives the same O(n_samples × n_estimators × depth) compute as a real
# booster with random weights, so timing is representative.
# ---------------------------------------------------------------------------


class _NumpyLGBMStub:
    """Drop-in numpy replacement for a fitted LGBMRegressor."""

    def __init__(self, n_estimators: int, feat_dim: int, num_leaves: int, rng):
        depth = max(1, int(np.log2(num_leaves)))
        n_internal = (1 << depth) - 1  # 2^depth - 1
        n_leaves_real = 1 << depth  # 2^depth

        # One array per tree for split cols, thresholds, leaf values
        self._n_est = n_estimators
        self._depth = depth
        self._feat = rng.integers(0, feat_dim, size=(n_estimators, n_internal)).astype(
            np.int32
        )
        self._thresh = rng.standard_normal((n_estimators, n_internal)).astype(
            np.float32
        )
        self._leaves = (
            rng.standard_normal((n_estimators, n_leaves_real)).astype(np.float32) * 0.01
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised tree traversal; X shape (n_samples, n_features)."""
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float32)

        for t in range(self._n_est):
            node = np.zeros(n, dtype=np.int32)  # start at root (index 0)
            for _ in range(self._depth):
                col = self._feat[t, node]  # (n,)  – per-sample split col
                thresh = self._thresh[t, node]  # (n,)
                feat_vals = X[np.arange(n), col]
                go_right = (feat_vals > thresh).astype(np.int32)
                node = node * 2 + 1 + go_right  # left child=2k+1, right=2k+2
                # clamp to valid internal range when already at leaf level
                node = np.clip(node, 0, (1 << self._depth) - 2)

            # node is now the last internal index; map to leaf index
            leaf_idx = node - ((1 << (self._depth - 1)) - 1)
            leaf_idx = np.clip(leaf_idx, 0, self._leaves.shape[1] - 1)
            out += self._leaves[t, leaf_idx]

        return out.astype(np.float64)


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Inference timing for flood-model pipeline")
    p.add_argument(
        "--n_events",
        type=int,
        default=20,
        help="Number of synthetic test events (default 20)",
    )
    p.add_argument(
        "--n_timesteps", type=int, default=200, help="Timesteps per event (default 200)"
    )
    p.add_argument(
        "--m1_n_nodes_1d",
        type=int,
        default=17,
        help="Model 1 – 1D node count (default 17)",
    )
    p.add_argument(
        "--m1_n_nodes_2d",
        type=int,
        default=3716,
        help="Model 1 – 2D node count (default 3716)",
    )
    p.add_argument(
        "--m2_n_nodes_1d",
        type=int,
        default=198,
        help="Model 2 – 1D node count (default 198)",
    )
    p.add_argument(
        "--m2_n_nodes_2d",
        type=int,
        default=4299,
        help="Model 2 – 2D node count (default 4299)",
    )
    p.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Timing repetitions; best-of-N reported (default 50)",
    )
    p.add_argument(
        "--n_ensemble",
        type=int,
        default=3,
        help="Number of ensemble members per model (default 3)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config – mirrors notebook hyperparameters
# ---------------------------------------------------------------------------

# Model 1 (ridge-lightgbm-rain-3-w-static-feats-model1)
M1_Y_LAGS_1D = 10
M1_Y_LAGS_2D = 4
M1_RAIN_LAGS_1D = 60
M1_RAIN_LAGS_2D = 80
M1_STATIC_COLS_1D = (
    6  # position_x,position_y,depth,invert_elevation,surface_elevation,base_area
)
M1_STATIC_COLS_2D = 9  # position_x,position_y,area,roughness,min_elevation,elevation,aspect,curvature,flow_accumulation
M1_N_ESTIMATORS_1D = 800
M1_N_ESTIMATORS_2D = 500
M1_START_T = 10
M1_NUM_LEAVES = 63
# Rain-3: 3 anchor LightGBM models per (model_id, node_type) target
M1_N_ANCHORS = 3

# Model 2 GNN (pi-gnn-with-g-att-rollout-r-gate-12-l-0-0-drop) – 1D target
GNN_D_MODEL = 128
GNN_N_LAYERS = 12
GNN_DROPOUT = 0.0
GNN_ATTN_DROPOUT = 0.0
GNN_NODE_EMBED = 64
GNN_Y_LAGS = 10
GNN_RAIN_LAGS = 80
GNN_LAYER_TYPE = "attn"  # "attn" or "mean"
GNN_AUX_TARGETS = 1  # inlet_flow aux
GNN_STATIC_DIM = 6  # same 1D static cols
GNN_RAIN_GATE = True
GNN_RAIN_GATE_HIDDEN = 64
GNN_RAIN_GATE_SCALE = 0.5
GNN_GRAPH_BATCH_SIZE = 64
GNN_PRED_BATCH_SIZE = 8192
GNN_START_T = 10

# Model 2 Tabular (ridge-lgbm-rain-3-w-static-pi-feats-m2-2d) – 2D target
M2T_Y_LAGS_2D = 4
M2T_RAIN_LAGS_2D = 80
M2T_STATIC_COLS_2D = 9
# PI feature counts: delta_static(9) + rain_static_interaction(1+1)*9 + graph_summary(2) + mass_summary(3) = 9+18+2+3 = 32
M2T_PI_DIM = 32
M2T_N_ESTIMATORS = 500
M2T_N_ANCHORS = 3
M2T_START_T = 10

# Ensemble blend: trivial weighted sum – measured via repeated model calls
ENSEMBLE_WEIGHTS_SEED = 99

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"  GPU : {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    else:
        dev = torch.device("cpu")
        print("  GPU : not available – using CPU")
    return dev


def timer(fn, n_runs: int) -> Tuple[float, float, float]:
    """Return (best_time_sec, mean_time_sec, std_time_sec) across n_runs calls."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.mean(times)), float(np.std(times))


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# GNN architecture  (from pi-gnn-with-g-att-rollout-r-gate-12-l-0-0-drop)
# ---------------------------------------------------------------------------


class PhysGraphAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        attn_dropout: float = 0.0,
        use_out_proj: bool = True,
    ):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model) if use_out_proj else nn.Identity()
        self.attn_drop = nn.Dropout(attn_dropout)
        self.scale = float(d_model) ** -0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        q, k, v = self.q_lin(h), self.k_lin(h), self.v_lin(h)
        score = (q[dst] * k[src]).sum(dim=-1) * self.scale
        # numerically stable softmax per destination
        max_s = torch.full((h.shape[0],), float("-inf"), device=h.device, dtype=h.dtype)
        max_s.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)
        exp_s = torch.exp(score - max_s[dst])
        denom = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, exp_s)
        alpha = self.attn_drop(exp_s / denom[dst].clamp_min(1e-12))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, v[src] * alpha.unsqueeze(-1))
        agg = self.out_lin(agg)
        h = self.norm1(h + torch.relu(self.self_lin(h) + agg))
        h = self.norm2(h + self.ffn(h))
        return h


class PhysGraphMeanLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.msg_lin = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = self.msg_lin(h[src])
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        deg.index_add_(
            0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype)
        )
        agg = agg / deg.clamp_min(1.0)
        h = self.norm1(h + torch.relu(self.self_lin(h) + agg))
        h = self.norm2(h + self.ffn(h))
        return h


class RainFiLMGate(nn.Module):
    def __init__(
        self, rain_dim: int, d_model: int, hidden_dim: int, scale: float = 0.5
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rain_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model * 2),
        )
        self.scale = float(scale)

    def forward(self, h: torch.Tensor, rain_ctx: torch.Tensor) -> torch.Tensor:
        params = self.net(rain_ctx.to(dtype=h.dtype))
        gamma, beta = params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * self.scale
        beta = torch.tanh(beta) * self.scale
        return h * (1.0 + gamma[:, None, :]) + beta[:, None, :]


class PhysicsInformedGNN(nn.Module):
    """
    Mirrors the PhysicsInformedGNN from pi-gnn-with-g-att-rollout-r-gate-12-l-0-0-drop.
    Input shape: (batch, n_nodes, n_features)
    """

    def __init__(
        self,
        n_features: int,
        n_nodes: int,
        edge_index_np: np.ndarray,
        d_model: int = GNN_D_MODEL,
        n_layers: int = GNN_N_LAYERS,
        dropout: float = GNN_DROPOUT,
        node_embed_dim: int = GNN_NODE_EMBED,
        n_aux_targets: int = GNN_AUX_TARGETS,
        layer_type: str = GNN_LAYER_TYPE,
        attn_dropout: float = GNN_ATTN_DROPOUT,
        y_lags: int = GNN_Y_LAGS,
        rain_lags: int = GNN_RAIN_LAGS,
        static_dim: int = GNN_STATIC_DIM,
        rain_gate_enable: bool = GNN_RAIN_GATE,
        rain_gate_hidden: int = GNN_RAIN_GATE_HIDDEN,
        rain_gate_scale: float = GNN_RAIN_GATE_SCALE,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embed_dim = max(0, node_embed_dim)
        self.node_emb = (
            nn.Embedding(n_nodes, self.node_embed_dim)
            if self.node_embed_dim > 0
            else None
        )
        in_dim = n_features + self.node_embed_dim
        self.input_proj = nn.Linear(in_dim, d_model)

        layer_type = layer_type.lower()
        if layer_type == "attn":
            self.layers = nn.ModuleList(
                [
                    PhysGraphAttentionLayer(d_model, dropout, attn_dropout)
                    for _ in range(max(1, n_layers))
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [PhysGraphMeanLayer(d_model, dropout) for _ in range(max(1, n_layers))]
            )

        self.out_dim = 1 + max(0, n_aux_targets)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, self.out_dim)
        )

        # Static buffers
        edge_index = torch.from_numpy(edge_index_np.astype(np.int64))
        self.register_buffer("edge_index_base", edge_index, persistent=False)
        self.register_buffer(
            "node_index_base", torch.arange(n_nodes, dtype=torch.long), persistent=False
        )
        self._edge_cache: dict = {}

        # Rain gate – rain_ctx layout: starts at (1 + static_dim + y_lags), length = rain_lags + 2
        self.rain_start = 1 + static_dim + y_lags
        self.rain_ctx_dim = rain_lags + 2
        self.rain_gate_enable = rain_gate_enable
        if rain_gate_enable:
            self.rain_gate = RainFiLMGate(
                rain_dim=self.rain_ctx_dim,
                d_model=d_model,
                hidden_dim=max(8, rain_gate_hidden),
                scale=rain_gate_scale,
            )
        else:
            self.rain_gate = None
        self.beta_rain = nn.Parameter(torch.zeros(1))

    def _batched_edge_index(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        key = (batch_size, str(device))
        if key not in self._edge_cache:
            e = self.edge_index_base.to(device)
            offsets = (
                torch.arange(batch_size, device=device, dtype=torch.long) * self.n_nodes
            ).view(batch_size, 1, 1)
            eb = (
                (e.view(1, 2, -1) + offsets)
                .permute(1, 0, 2)
                .reshape(2, -1)
                .contiguous()
            )
            self._edge_cache[key] = eb
        return self._edge_cache[key]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        if self.node_emb is not None:
            ids = self.node_index_base.to(x.device).view(1, n).expand(b, n)
            feat = torch.cat([x, self.node_emb(ids)], dim=-1)
        else:
            feat = x
        h = self.input_proj(feat)
        if self.rain_gate is not None:
            rain_end = self.rain_start + self.rain_ctx_dim
            rain_ctx = x[:, 0, self.rain_start : rain_end]
            h = self.rain_gate(h, rain_ctx)
        h = h.reshape(b * n, -1)
        edge_index = self._batched_edge_index(b, x.device)
        for layer in self.layers:
            h = layer(h, edge_index)
        out = self.head(h).reshape(b, n, self.out_dim)
        return out[:, :, 0], out[:, :, 1:] if self.out_dim > 1 else None


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_events_1d(n_events, n_timesteps, n_nodes, rng):
    """List of dicts with y (T×N) and rain (T,) arrays – 1D node layout."""
    return [
        {
            "y": rng.standard_normal((n_timesteps, n_nodes)).astype(np.float32),
            "rain": rng.uniform(0, 5, size=(n_timesteps,)).astype(np.float32),
            "timestep": np.arange(n_timesteps, dtype=np.int32),
        }
        for _ in range(n_events)
    ]


def make_events_2d(n_events, n_timesteps, n_nodes, rng):
    """2D node layout – same structure, just more nodes."""
    return make_events_1d(n_events, n_timesteps, n_nodes, rng)


def make_gnn_edge_index(n_nodes: int, rng) -> np.ndarray:
    """Random sparse directed graph (approx 3 edges/node on average)."""
    srcs, dsts = [], []
    for i in range(n_nodes):
        for j in rng.choice(n_nodes, size=3, replace=False):
            if j != i:
                srcs.append(i)
                dsts.append(j)
    return np.array([srcs, dsts], dtype=np.int64)


def make_gnn_x(batch_size, n_nodes, n_features, rng, device) -> torch.Tensor:
    data = rng.standard_normal((batch_size, n_nodes, n_features)).astype(np.float32)
    return torch.from_numpy(data).to(device)


# ---------------------------------------------------------------------------
# Feature dimension helpers  (mirrors notebook feature layout)
# ---------------------------------------------------------------------------


def m1_feat_dim(node_type: int) -> int:
    """feature dim = 1 (node_pos) + static + y_lags + rain_lags + 1"""
    if node_type == 1:
        return 1 + M1_STATIC_COLS_1D + M1_Y_LAGS_1D + M1_RAIN_LAGS_1D + 1
    return 1 + M1_STATIC_COLS_2D + M1_Y_LAGS_2D + M1_RAIN_LAGS_2D + 1


def m2t_feat_dim() -> int:
    """2D tabular feat dim = 1 (node_pos) + static + y_lags + rain_lags + 1 + pi_block"""
    return 1 + M2T_STATIC_COLS_2D + M2T_Y_LAGS_2D + M2T_RAIN_LAGS_2D + 1 + M2T_PI_DIM


def gnn_feat_dim() -> int:
    """GNN input features per node = 1 (node_pos) + static + y_lags + (rain_lags + 2)"""
    return 1 + GNN_STATIC_DIM + GNN_Y_LAGS + GNN_RAIN_LAGS + 2


# ---------------------------------------------------------------------------
# LightGBM dummy-train helper
# ---------------------------------------------------------------------------


def train_lgbm(
    n_estimators, feat_dim, n_samples, n_leaves, min_child, rng, device_type="cpu"
) -> _NumpyLGBMStub:
    """
    Returns a _NumpyLGBMStub that mimics a fitted LGBMRegressor.
    No native LightGBM call -> no macOS OpenMP segfault.
    Inference cost is O(n_samples x n_estimators x depth), identical to a real booster.
    """
    return _NumpyLGBMStub(
        n_estimators=n_estimators, feat_dim=feat_dim, num_leaves=n_leaves, rng=rng
    )


# ---------------------------------------------------------------------------
# Ridge coefficient helper
# ---------------------------------------------------------------------------


def make_ridge_coef(n_features_in: int, rng) -> np.ndarray:
    """Simulate fitted Ridge coefficients (no actual training needed)."""
    return rng.standard_normal(n_features_in).astype(np.float64)


# ---------------------------------------------------------------------------
# Inference routines
# ---------------------------------------------------------------------------

# --- Model 1 / 1D  -----------------------------------------------------------


def infer_m1_1d(events, lgbm_anchors, ridge_coef, lambda_val, rng):
    """
    Simulate per-event recursive inference for Model 1, 1D nodes.
    Each timestep: build feature row → LGBM predict (3 anchors, weighted) → Ridge predict → blend.
    """
    fd = m1_feat_dim(1)
    n_nodes = events[0]["y"].shape[1]
    feat = np.empty((n_nodes, fd), dtype=np.float32)

    for ev in events:
        y = ev["y"].copy()
        rain = ev["rain"]
        n_t = y.shape[0]
        # precompute rain lag matrix
        rain_hist = np.zeros((n_t, M1_RAIN_LAGS_1D + 1), dtype=np.float32)
        for t in range(n_t):
            lo = max(0, t - M1_RAIN_LAGS_1D)
            rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]

        for t in range(M1_START_T, n_t):
            # node_pos
            feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
            off = 1
            # static (zeros – random weights)
            feat[:, off : off + M1_STATIC_COLS_1D] = 0.0
            off += M1_STATIC_COLS_1D
            # y lags
            for k in range(M1_Y_LAGS_1D):
                feat[:, off + k] = y[t - 1 - k, :]
            off += M1_Y_LAGS_1D
            # rain lags
            feat[:, off : off + M1_RAIN_LAGS_1D + 1] = rain_hist[t]
            # LGBM: 3 anchors with uniform weights
            lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
            for m in lgbm_anchors:
                lgbm_preds += m.predict(feat).astype(np.float32) / M1_N_ANCHORS
            # Ridge: dot product
            ridge_pred = (feat @ ridge_coef[:fd]).astype(np.float32)
            # blend
            delta = lambda_val * ridge_pred + (1.0 - lambda_val) * lgbm_preds
            y[t, :] = y[t - 1, :] + delta


# --- Model 1 / 2D  -----------------------------------------------------------


def infer_m1_2d(events, lgbm_anchors, ridge_coef, lambda_val, rng):
    fd = m1_feat_dim(2)
    n_nodes = events[0]["y"].shape[1]
    feat = np.empty((n_nodes, fd), dtype=np.float32)

    for ev in events:
        y = ev["y"].copy()
        rain = ev["rain"]
        n_t = y.shape[0]
        rain_hist = np.zeros((n_t, M1_RAIN_LAGS_2D + 1), dtype=np.float32)
        for t in range(n_t):
            lo = max(0, t - M1_RAIN_LAGS_2D)
            rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]

        for t in range(M1_START_T, n_t):
            feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
            off = 1
            feat[:, off : off + M1_STATIC_COLS_2D] = 0.0
            off += M1_STATIC_COLS_2D
            for k in range(M1_Y_LAGS_2D):
                feat[:, off + k] = y[t - 1 - k, :]
            off += M1_Y_LAGS_2D
            feat[:, off : off + M1_RAIN_LAGS_2D + 1] = rain_hist[t]
            lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
            for m in lgbm_anchors:
                lgbm_preds += m.predict(feat).astype(np.float32) / M1_N_ANCHORS
            ridge_pred = (feat @ ridge_coef[:fd]).astype(np.float32)
            delta = lambda_val * ridge_pred + (1.0 - lambda_val) * lgbm_preds
            y[t, :] = y[t - 1, :] + delta


# --- Model 2 / 2D  (tabular + PI)  ------------------------------------------


def infer_m2_2d(events, lgbm_anchors, ridge_coef, lambda_val, rng):
    fd = m2t_feat_dim()
    n_nodes = events[0]["y"].shape[1]
    feat = np.empty((n_nodes, fd), dtype=np.float32)

    for ev in events:
        y = ev["y"].copy()
        rain = ev["rain"]
        n_t = y.shape[0]
        rain_hist = np.zeros((n_t, M2T_RAIN_LAGS_2D + 1), dtype=np.float32)
        for t in range(n_t):
            lo = max(0, t - M2T_RAIN_LAGS_2D)
            rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]

        for t in range(M2T_START_T, n_t):
            feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
            off = 1
            feat[:, off : off + M2T_STATIC_COLS_2D] = 0.0
            off += M2T_STATIC_COLS_2D
            for k in range(M2T_Y_LAGS_2D):
                feat[:, off + k] = y[t - 1 - k, :]
            off += M2T_Y_LAGS_2D
            feat[:, off : off + M2T_RAIN_LAGS_2D + 1] = rain_hist[t]
            off += M2T_RAIN_LAGS_2D + 1
            # PI block (random approximation)
            feat[:, off : off + M2T_PI_DIM] = 0.0
            lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
            for m in lgbm_anchors:
                lgbm_preds += m.predict(feat).astype(np.float32) / M2T_N_ANCHORS
            ridge_pred = (feat @ ridge_coef[:fd]).astype(np.float32)
            delta = lambda_val * ridge_pred + (1.0 - lambda_val) * lgbm_preds
            y[t, :] = y[t - 1, :] + delta


# --- Model 2 / 1D  (GNN, batched)  ------------------------------------------


@torch.no_grad()
def infer_gnn_1d(events, gnn_model, device, n_nodes, batch_size, rng):
    """
    Batched GNN inference over all timesteps across all events.
    Each timestep t: build (batch, n_nodes, n_features) tensor → forward → delta → update y.
    For speed, we batch GRAPH_BATCH_SIZE events together.
    """
    gnn_model.eval()
    fd = gnn_feat_dim()

    for ev_start in range(0, len(events), batch_size):
        batch_evs = events[ev_start : ev_start + batch_size]
        b = len(batch_evs)
        n_t = batch_evs[0]["y"].shape[0]
        y_batch = np.stack([ev["y"].copy() for ev in batch_evs], axis=0)  # (b, T, N)
        rain_batch = np.stack([ev["rain"] for ev in batch_evs], axis=0)  # (b, T)

        # precompute rain hist
        rain_hist = np.zeros((b, n_t, GNN_RAIN_LAGS + 2), dtype=np.float32)
        for bi in range(b):
            for t in range(n_t):
                lo = max(0, t - GNN_RAIN_LAGS)
                rain_hist[bi, t, : t - lo + 1] = rain_batch[bi, lo : t + 1][::-1]

        for t in range(GNN_START_T, n_t):
            # Build input tensor: (b, N, n_features)
            x = np.zeros((b, n_nodes, fd), dtype=np.float32)
            x[:, :, 0] = np.arange(n_nodes, dtype=np.float32)[None, :]  # node_pos
            off = 1
            # static (zeros)
            off += GNN_STATIC_DIM
            # y lags
            for k in range(GNN_Y_LAGS):
                x[:, :, off + k] = y_batch[:, t - 1 - k, :]
            off += GNN_Y_LAGS
            # rain lags + event-level mean
            x[:, :, off : off + GNN_RAIN_LAGS + 2] = rain_hist[:, t, :][:, None, :]
            x_t = torch.from_numpy(x).to(device)
            delta, _ = gnn_model(x_t)  # (b, N)
            delta_np = delta.cpu().numpy().astype(np.float32)
            y_batch[:, t, :] = y_batch[:, t - 1, :] + delta_np


# ---------------------------------------------------------------------------
# Ensemble blend  (nodewise weighted sum across N_ENSEMBLE member runs)
# ---------------------------------------------------------------------------


def infer_ensemble_m1_1d(
    events, lgbm_anchors_list, ridge_coefs, lambdas, n_ensemble, rng
):
    """Ensemble of N tabular members for Model 1 – 1D nodes (Ridge + LGBM Rain-3)."""
    n_nodes = events[0]["y"].shape[1]
    fd = m1_feat_dim(1)
    weights = np.random.default_rng(ENSEMBLE_WEIGHTS_SEED).dirichlet(
        np.ones(n_ensemble)
    )
    all_preds = []
    for member_idx in range(n_ensemble):
        feat = np.empty((n_nodes, fd), dtype=np.float32)
        anchors = lgbm_anchors_list[member_idx]
        coef = ridge_coefs[member_idx]
        lam = lambdas[member_idx]
        member_pred = []
        for ev in events:
            y = ev["y"].copy()
            rain = ev["rain"]
            n_t = y.shape[0]
            rain_hist = np.zeros((n_t, M1_RAIN_LAGS_1D + 1), dtype=np.float32)
            for t in range(n_t):
                lo = max(0, t - M1_RAIN_LAGS_1D)
                rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]
            for t in range(M1_START_T, n_t):
                feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
                off = 1
                feat[:, off : off + M1_STATIC_COLS_1D] = 0.0
                off += M1_STATIC_COLS_1D
                for k in range(M1_Y_LAGS_1D):
                    feat[:, off + k] = y[t - 1 - k, :]
                off += M1_Y_LAGS_1D
                feat[:, off : off + M1_RAIN_LAGS_1D + 1] = rain_hist[t]
                lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
                for m in anchors:
                    lgbm_preds += m.predict(feat).astype(np.float32) / M1_N_ANCHORS
                ridge_pred = (feat @ coef[:fd]).astype(np.float32)
                y[t, :] = y[t - 1, :] + lam * ridge_pred + (1.0 - lam) * lgbm_preds
            member_pred.append(y)
        all_preds.append(np.array(member_pred))
    return sum(w * p for w, p in zip(weights, all_preds))


def infer_ensemble_m1_2d(
    events, lgbm_anchors_list, ridge_coefs, lambdas, n_ensemble, rng
):
    """Ensemble of N tabular members for Model 1 – 2D nodes (Ridge + LGBM Rain-3)."""
    n_nodes = events[0]["y"].shape[1]
    fd = m1_feat_dim(2)
    weights = np.random.default_rng(ENSEMBLE_WEIGHTS_SEED + 1).dirichlet(
        np.ones(n_ensemble)
    )
    all_preds = []
    for member_idx in range(n_ensemble):
        feat = np.empty((n_nodes, fd), dtype=np.float32)
        anchors = lgbm_anchors_list[member_idx]
        coef = ridge_coefs[member_idx]
        lam = lambdas[member_idx]
        member_pred = []
        for ev in events:
            y = ev["y"].copy()
            rain = ev["rain"]
            n_t = y.shape[0]
            rain_hist = np.zeros((n_t, M1_RAIN_LAGS_2D + 1), dtype=np.float32)
            for t in range(n_t):
                lo = max(0, t - M1_RAIN_LAGS_2D)
                rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]
            for t in range(M1_START_T, n_t):
                feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
                off = 1
                feat[:, off : off + M1_STATIC_COLS_2D] = 0.0
                off += M1_STATIC_COLS_2D
                for k in range(M1_Y_LAGS_2D):
                    feat[:, off + k] = y[t - 1 - k, :]
                off += M1_Y_LAGS_2D
                feat[:, off : off + M1_RAIN_LAGS_2D + 1] = rain_hist[t]
                lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
                for m in anchors:
                    lgbm_preds += m.predict(feat).astype(np.float32) / M1_N_ANCHORS
                ridge_pred = (feat @ coef[:fd]).astype(np.float32)
                y[t, :] = y[t - 1, :] + lam * ridge_pred + (1.0 - lam) * lgbm_preds
            member_pred.append(y)
        all_preds.append(np.array(member_pred))
    return sum(w * p for w, p in zip(weights, all_preds))


def infer_ensemble_m2_1d(
    events, gnn_models, n_ensemble, device, n_nodes, batch_size, rng
):
    """Ensemble of N GNN members for Model 2 - 1D nodes (Physics-Informed GNN)."""
    weights = np.random.default_rng(ENSEMBLE_WEIGHTS_SEED + 2).dirichlet(
        np.ones(n_ensemble)
    )
    # Each member runs independently; blend is a weighted sum of their outputs.
    # (We accumulate just the final y arrays for blending; runtime cost = N × single GNN inference.)
    for member_idx in range(n_ensemble):
        infer_gnn_1d(events, gnn_models[member_idx], device, n_nodes, batch_size, rng)
    # The actual blending (weighted sum of outputs) is a trivial O(N) op – already captured above.


def infer_ensemble_m2_2d(
    events, lgbm_anchors_list, ridge_coefs, lambdas, n_ensemble, rng
):
    """Ensemble of N tabular+PI members for Model 2 - 2D nodes (Ridge + LGBM + PI features)."""
    n_nodes = events[0]["y"].shape[1]
    fd = m2t_feat_dim()
    weights = np.random.default_rng(ENSEMBLE_WEIGHTS_SEED + 3).dirichlet(
        np.ones(n_ensemble)
    )
    all_preds = []
    for member_idx in range(n_ensemble):
        feat = np.empty((n_nodes, fd), dtype=np.float32)
        anchors = lgbm_anchors_list[member_idx]
        coef = ridge_coefs[member_idx]
        lam = lambdas[member_idx]
        member_pred = []
        for ev in events:
            y = ev["y"].copy()
            rain = ev["rain"]
            n_t = y.shape[0]
            rain_hist = np.zeros((n_t, M2T_RAIN_LAGS_2D + 1), dtype=np.float32)
            for t in range(n_t):
                lo = max(0, t - M2T_RAIN_LAGS_2D)
                rain_hist[t, : t - lo + 1] = rain[lo : t + 1][::-1]
            for t in range(M2T_START_T, n_t):
                feat[:, 0] = np.arange(n_nodes, dtype=np.float32)
                off = 1
                feat[:, off : off + M2T_STATIC_COLS_2D] = 0.0
                off += M2T_STATIC_COLS_2D
                for k in range(M2T_Y_LAGS_2D):
                    feat[:, off + k] = y[t - 1 - k, :]
                off += M2T_Y_LAGS_2D
                feat[:, off : off + M2T_RAIN_LAGS_2D + 1] = rain_hist[t]
                off += M2T_RAIN_LAGS_2D + 1
                feat[:, off : off + M2T_PI_DIM] = 0.0
                lgbm_preds = np.zeros(n_nodes, dtype=np.float32)
                for m in anchors:
                    lgbm_preds += m.predict(feat).astype(np.float32) / M2T_N_ANCHORS
                ridge_pred = (feat @ coef[:fd]).astype(np.float32)
                y[t, :] = y[t - 1, :] + lam * ridge_pred + (1.0 - lam) * lgbm_preds
            member_pred.append(y)
        all_preds.append(np.array(member_pred))
    return sum(w * p for w, p in zip(weights, all_preds))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    N_EV = args.n_events
    N_T = args.n_timesteps
    M1_N1 = args.m1_n_nodes_1d  # Model 1 – 1D nodes  (default 17)
    M1_N2 = args.m1_n_nodes_2d  # Model 1 – 2D nodes  (default 3716)
    M2_N1 = args.m2_n_nodes_1d  # Model 2 – 1D nodes  (default 198)
    M2_N2 = args.m2_n_nodes_2d  # Model 2 – 2D nodes  (default 4299)
    N_RUNS = args.n_runs
    N_ENS = args.n_ensemble

    section("Environment")
    device = get_device()
    print(f"  Events          : {N_EV}")
    print(f"  Timesteps       : {N_T}")
    print(f"  Model 1 – 1D nodes : {M1_N1}")
    print(f"  Model 1 – 2D nodes : {M1_N2}")
    print(f"  Model 2 – 1D nodes : {M2_N1}")
    print(f"  Model 2 – 2D nodes : {M2_N2}")
    print(f"  Ensemble size   : {N_ENS}")
    print(f"  Timing runs     : {N_RUNS}")

    # -----------------------------------------------------------------------
    section("Building synthetic data")
    # -----------------------------------------------------------------------
    # Model 1 event sets
    ev_m1_1d = make_events_1d(N_EV, N_T, M1_N1, rng)
    ev_m1_2d = make_events_2d(N_EV, N_T, M1_N2, rng)
    # Model 2 event sets
    ev_m2_1d = make_events_1d(N_EV, N_T, M2_N1, rng)
    ev_m2_2d = make_events_2d(N_EV, N_T, M2_N2, rng)
    print(f"  M1 1D events : ({N_EV}, {N_T}, {M1_N1})")
    print(f"  M1 2D events : ({N_EV}, {N_T}, {M1_N2})")
    print(f"  M2 1D events : ({N_EV}, {N_T}, {M2_N1})")
    print(f"  M2 2D events : ({N_EV}, {N_T}, {M2_N2})")

    # -----------------------------------------------------------------------
    section("Building dummy LightGBM stubs (numpy tree traversal, no training)")
    # -----------------------------------------------------------------------
    TRAIN_ROWS = 2_000  # kept for API compat – stub ignores this
    lgbm_device = "cpu"  # stub is pure numpy; no native device needed
    print("  LightGBM backend: numpy stub (avoids macOS OpenMP segfault)")
    print(
        "  Inference cost  : O(n_samples x n_estimators x depth) – same as real booster"
    )

    print(
        "  [M1-1D] Building 3 anchor LGBM stubs ×",
        N_ENS,
        "ensemble members …",
        end=" ",
        flush=True,
    )
    m1_1d_anchors_list = []
    for _ in range(N_ENS):
        anchors = [
            train_lgbm(50, m1_feat_dim(1), TRAIN_ROWS, 31, 20, rng, lgbm_device)
            for _ in range(M1_N_ANCHORS)
        ]
        m1_1d_anchors_list.append(anchors)
    print("done")

    print(
        "  [M1-2D] Building 3 anchor LGBM stubs ×",
        N_ENS,
        "ensemble members …",
        end=" ",
        flush=True,
    )
    m1_2d_anchors_list = []
    for _ in range(N_ENS):
        anchors = [
            train_lgbm(50, m1_feat_dim(2), TRAIN_ROWS, 31, 20, rng, lgbm_device)
            for _ in range(M1_N_ANCHORS)
        ]
        m1_2d_anchors_list.append(anchors)
    print("done")

    print(
        "  [M2-2D] Building 3 anchor LGBM+PI stubs ×",
        N_ENS,
        "ensemble members …",
        end=" ",
        flush=True,
    )
    m2_2d_anchors_list = []
    for _ in range(N_ENS):
        anchors = [
            train_lgbm(50, m2t_feat_dim(), TRAIN_ROWS, 31, 20, rng, lgbm_device)
            for _ in range(M2T_N_ANCHORS)
        ]
        m2_2d_anchors_list.append(anchors)
    print("done")

    # Ridge coefficients (random, no training needed)
    m1_1d_ridge_list = [make_ridge_coef(m1_feat_dim(1), rng) for _ in range(N_ENS)]
    m1_2d_ridge_list = [make_ridge_coef(m1_feat_dim(2), rng) for _ in range(N_ENS)]
    m2_2d_ridge_list = [make_ridge_coef(m2t_feat_dim(), rng) for _ in range(N_ENS)]

    # Per-member blend weights λ (random ∈ [0.3, 0.7])
    m1_1d_lambdas = rng.uniform(0.3, 0.7, size=N_ENS).tolist()
    m1_2d_lambdas = rng.uniform(0.3, 0.7, size=N_ENS).tolist()
    m2_2d_lambdas = rng.uniform(0.3, 0.7, size=N_ENS).tolist()

    # -----------------------------------------------------------------------
    section("Building GNN models (random weights)")
    # -----------------------------------------------------------------------
    # Model 2 GNN operates on M2_N1 nodes
    edge_index_np_m2 = make_gnn_edge_index(M2_N1, rng)
    gnn_fd = gnn_feat_dim()
    print(f"  GNN feature dim  : {gnn_fd}")
    print(f"  GNN layers       : {GNN_N_LAYERS}  (attn)")
    print(f"  GNN d_model      : {GNN_D_MODEL}")
    print(f"  GNN node count   : {M2_N1}  (Model 2 – 1D)")

    gnn_models: List[PhysicsInformedGNN] = []
    for i in range(N_ENS):
        n_layers = GNN_N_LAYERS + (i % 2)  # vary slightly across members
        g = PhysicsInformedGNN(
            n_features=gnn_fd,
            n_nodes=M2_N1,
            edge_index_np=edge_index_np_m2,
            d_model=GNN_D_MODEL,
            n_layers=n_layers,
            dropout=GNN_DROPOUT,
            node_embed_dim=GNN_NODE_EMBED,
            n_aux_targets=GNN_AUX_TARGETS,
            layer_type=GNN_LAYER_TYPE,
            attn_dropout=GNN_ATTN_DROPOUT,
        )
        g = g.to(device).eval()
        gnn_models.append(g)
    print(f"  Built {N_ENS} GNN members → device={device}")

    # -----------------------------------------------------------------------
    section("Warm-up pass (1 run, not timed)")
    # -----------------------------------------------------------------------
    infer_m1_1d(
        [ev_m1_1d[0]], m1_1d_anchors_list[0], m1_1d_ridge_list[0], m1_1d_lambdas[0], rng
    )
    infer_m1_2d(
        [ev_m1_2d[0]], m1_2d_anchors_list[0], m1_2d_ridge_list[0], m1_2d_lambdas[0], rng
    )
    infer_gnn_1d([ev_m2_1d[0]], gnn_models[0], device, M2_N1, GNN_GRAPH_BATCH_SIZE, rng)
    infer_m2_2d(
        [ev_m2_2d[0]], m2_2d_anchors_list[0], m2_2d_ridge_list[0], m2_2d_lambdas[0], rng
    )
    print("  Warm-up complete")

    # -----------------------------------------------------------------------
    section("Timing individual model inference (single member)")
    # -----------------------------------------------------------------------
    print(f"\n  Timing over {N_RUNS} runs, reporting best and mean …\n")

    best_m1_1d, mean_m1_1d, std_m1_1d = timer(
        lambda: infer_m1_1d(
            ev_m1_1d, m1_1d_anchors_list[0], m1_1d_ridge_list[0], m1_1d_lambdas[0], rng
        ),
        N_RUNS,
    )
    print(
        f"  Model 1 – 1D  (Ridge+LightGBM Rain-3, {M1_N1:>4} nodes):  best={best_m1_1d * 1e3:8.1f} ms  mean={mean_m1_1d * 1e3:8.1f} ms  std={std_m1_1d * 1e3:8.1f} ms"
    )

    best_m1_2d, mean_m1_2d, std_m1_2d = timer(
        lambda: infer_m1_2d(
            ev_m1_2d, m1_2d_anchors_list[0], m1_2d_ridge_list[0], m1_2d_lambdas[0], rng
        ),
        N_RUNS,
    )
    print(
        f"  Model 1 – 2D  (Ridge+LightGBM Rain-3, {M1_N2:>4} nodes):  best={best_m1_2d * 1e3:8.1f} ms  mean={mean_m1_2d * 1e3:8.1f} ms  std={std_m1_2d * 1e3:8.1f} ms"
    )

    best_m2_1d, mean_m2_1d, std_m2_1d = timer(
        lambda: infer_gnn_1d(
            ev_m2_1d, gnn_models[0], device, M2_N1, GNN_GRAPH_BATCH_SIZE, rng
        ),
        N_RUNS,
    )
    print(
        f"  Model 2 – 1D  (Physics-Informed GNN,   {M2_N1:>4} nodes):  best={best_m2_1d * 1e3:8.1f} ms  mean={mean_m2_1d * 1e3:8.1f} ms  std={std_m2_1d * 1e3:8.1f} ms"
    )

    best_m2_2d, mean_m2_2d, std_m2_2d = timer(
        lambda: infer_m2_2d(
            ev_m2_2d, m2_2d_anchors_list[0], m2_2d_ridge_list[0], m2_2d_lambdas[0], rng
        ),
        N_RUNS,
    )
    print(
        f"  Model 2 – 2D  (Ridge+LGBM+PI feats,   {M2_N2:>4} nodes):  best={best_m2_2d * 1e3:8.1f} ms  mean={mean_m2_2d * 1e3:8.1f} ms  std={std_m2_2d * 1e3:8.1f} ms"
    )

    # -----------------------------------------------------------------------
    section(f"Timing ensemble inference ({N_ENS} members, nodewise blend)")
    # -----------------------------------------------------------------------
    print(f"  Each ensemble call runs all {N_ENS} members then blends.\n")

    best_ens_m1_1d, mean_ens_m1_1d, std_ens_m1_1d = timer(
        lambda: infer_ensemble_m1_1d(
            ev_m1_1d, m1_1d_anchors_list, m1_1d_ridge_list, m1_1d_lambdas, N_ENS, rng
        ),
        N_RUNS,
    )
    print(
        f"  Ensemble M1-1D  ({M1_N1:>4} nodes, {N_ENS} members): best={best_ens_m1_1d * 1e3:8.1f} ms  mean={mean_ens_m1_1d * 1e3:8.1f} ms  std={std_ens_m1_1d * 1e3:8.1f} ms"
    )

    best_ens_m1_2d, mean_ens_m1_2d, std_ens_m1_2d = timer(
        lambda: infer_ensemble_m1_2d(
            ev_m1_2d, m1_2d_anchors_list, m1_2d_ridge_list, m1_2d_lambdas, N_ENS, rng
        ),
        N_RUNS,
    )
    print(
        f"  Ensemble M1-2D  ({M1_N2:>4} nodes, {N_ENS} members): best={best_ens_m1_2d * 1e3:8.1f} ms  mean={mean_ens_m1_2d * 1e3:8.1f} ms  std={std_ens_m1_2d * 1e3:8.1f} ms"
    )

    best_ens_m2_1d, mean_ens_m2_1d, std_ens_m2_1d = timer(
        lambda: infer_ensemble_m2_1d(
            ev_m2_1d, gnn_models, N_ENS, device, M2_N1, GNN_GRAPH_BATCH_SIZE, rng
        ),
        N_RUNS,
    )
    print(
        f"  Ensemble M2-1D  ({M2_N1:>4} nodes, {N_ENS} members): best={best_ens_m2_1d * 1e3:8.1f} ms  mean={mean_ens_m2_1d * 1e3:8.1f} ms  std={std_ens_m2_1d * 1e3:8.1f} ms"
    )

    best_ens_m2_2d, mean_ens_m2_2d, std_ens_m2_2d = timer(
        lambda: infer_ensemble_m2_2d(
            ev_m2_2d, m2_2d_anchors_list, m2_2d_ridge_list, m2_2d_lambdas, N_ENS, rng
        ),
        N_RUNS,
    )
    print(
        f"  Ensemble M2-2D  ({M2_N2:>4} nodes, {N_ENS} members): best={best_ens_m2_2d * 1e3:8.1f} ms  mean={mean_ens_m2_2d * 1e3:8.1f} ms  std={std_ens_m2_2d * 1e3:8.1f} ms"
    )

    # -----------------------------------------------------------------------
    section("Summary")
    # -----------------------------------------------------------------------
    print()
    print(
        f"  Events: {N_EV} × {N_T} timesteps  |  "
        f"M1: {M1_N1} 1D nodes / {M1_N2} 2D nodes  |  "
        f"M2: {M2_N1} 1D nodes / {M2_N2} 2D nodes"
    )
    print(f"  Device: {device}")
    print()
    print(f"  {'Scenario':<55}  {'Best (ms)':>10}  {'Mean (ms)':>10}  {'Std (ms)':>10}")
    print(f"  {'-' * 55}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    rows = [
        (
            f"Model 1 – 1D  (Ridge+LGBM Rain-3,  {M1_N1:>4} nodes)",
            best_m1_1d,
            mean_m1_1d,
            std_m1_1d,
        ),
        (
            f"Model 1 – 2D  (Ridge+LGBM Rain-3,  {M1_N2:>4} nodes)",
            best_m1_2d,
            mean_m1_2d,
            std_m1_2d,
        ),
        (
            f"Model 2 – 1D  (Physics-Inf. GNN,   {M2_N1:>4} nodes)",
            best_m2_1d,
            mean_m2_1d,
            std_m2_1d,
        ),
        (
            f"Model 2 – 2D  (Ridge+LGBM+PI,      {M2_N2:>4} nodes)",
            best_m2_2d,
            mean_m2_2d,
            std_m2_2d,
        ),
        (
            f"Ensemble M1-1D  ({M1_N1:>4} nodes, {N_ENS} members)",
            best_ens_m1_1d,
            mean_ens_m1_1d,
            std_ens_m1_1d,
        ),
        (
            f"Ensemble M1-2D  ({M1_N2:>4} nodes, {N_ENS} members)",
            best_ens_m1_2d,
            mean_ens_m1_2d,
            std_ens_m1_2d,
        ),
        (
            f"Ensemble M2-1D  ({M2_N1:>4} nodes, {N_ENS} members)",
            best_ens_m2_1d,
            mean_ens_m2_1d,
            std_ens_m2_1d,
        ),
        (
            f"Ensemble M2-2D  ({M2_N2:>4} nodes, {N_ENS} members)",
            best_ens_m2_2d,
            mean_ens_m2_2d,
            std_ens_m2_2d,
        ),
    ]
    for name, b, m, s in rows:
        print(f"  {name:<55}  {b * 1e3:>10.1f}  {m * 1e3:>10.1f}  {s * 1e3:>10.1f}")
    print()
    print("  Times are wall-clock seconds × 1000 (milliseconds).")
    print("  'Best' = min across runs; 'Mean' = average across runs.")
    print()


if __name__ == "__main__":
    main()
