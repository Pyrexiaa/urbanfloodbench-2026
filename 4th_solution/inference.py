"""
Inference Time Benchmark for HeteroFloodGNN - Model 1 & Model 2
================================================================
Runs a forward pass on both models with random synthetic data (no dataset needed).
Reports inference time for:
  - Model 1  →  1D nodes output
  - Model 1  →  2D nodes output
  - Model 2  →  1D nodes output
  - Model 2  →  2D nodes output

Usage:
  python inference_benchmark.py
  python inference_benchmark.py --n-warmup 20 --n-runs 100 --device cpu
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, TransformerConv, SAGEConv
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Minimal re-implementations of helpers needed by HeteroFloodGNN
# ─────────────────────────────────────────────────────────────────────────────


class FusedLinearLayerNormGELU(nn.Module):
    """Linear → LayerNorm → GELU → Dropout block (mirrors original)."""

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.0, use_fused: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.linear(x))))


class HeteroFloodGNN(nn.Module):
    """
    Heterogeneous GNN for 1D-2D flood modelling.
    Stripped to the architecture only – no training or data-loading code.
    """

    def __init__(
        self,
        in_channels_1d: int,
        in_channels_2d: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        conv_type: str = "transformer",
        heads: int = 16,
        use_fused_ops: bool = True,
        edge_dim_1d: int = 4,
        edge_dim_2d: int = 0,
        temporal_bundle_k: int = 4,
        use_node_embeddings: bool = True,
        num_1d_nodes: Optional[int] = None,
        num_2d_nodes: Optional[int] = None,
        node_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        if conv_type in ("gatv2", "transformer") and hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )

        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim_1d = edge_dim_1d
        self.edge_dim_2d = edge_dim_2d
        self.temporal_bundle_k = max(int(temporal_bundle_k), 1)
        self.num_1d_nodes = int(num_1d_nodes) if num_1d_nodes is not None else None
        self.num_2d_nodes = int(num_2d_nodes) if num_2d_nodes is not None else None

        # ── Encoders ──────────────────────────────────────────────────────────
        self.encoder_1d = nn.Sequential(
            FusedLinearLayerNormGELU(
                in_channels_1d, hidden_channels, dropout, use_fused_ops
            ),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.encoder_2d = nn.Sequential(
            FusedLinearLayerNormGELU(
                in_channels_2d, hidden_channels, dropout, use_fused_ops
            ),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ── Optional node-ID embeddings ───────────────────────────────────────
        self.use_node_embeddings = (
            use_node_embeddings
            and self.num_1d_nodes is not None
            and self.num_1d_nodes > 0
            and self.num_2d_nodes is not None
            and self.num_2d_nodes > 0
        )
        if self.use_node_embeddings:
            self.node_embed_dim = max(
                1,
                int(
                    node_embed_dim
                    if node_embed_dim is not None
                    else hidden_channels // 4
                ),
            )
            self.node_embed_1d = nn.Embedding(self.num_1d_nodes, self.node_embed_dim)
            self.node_embed_2d = nn.Embedding(self.num_2d_nodes, self.node_embed_dim)
            self.node_fuse_1d = nn.Linear(
                hidden_channels + self.node_embed_dim, hidden_channels
            )
            self.node_fuse_2d = nn.Linear(
                hidden_channels + self.node_embed_dim, hidden_channels
            )
        else:
            self.node_embed_dim = 0

        # ── Message-passing layers ────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms_1d = nn.ModuleList()
        self.norms_2d = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                self._make_conv_dict(
                    hidden_channels, dropout, conv_type, heads, edge_dim_1d, edge_dim_2d
                ),
                aggr="mean",
            )
            self.convs.append(conv)
            self.norms_1d.append(nn.LayerNorm(hidden_channels))
            self.norms_2d.append(nn.LayerNorm(hidden_channels))

        self.final_norm_1d = nn.LayerNorm(hidden_channels)
        self.final_norm_2d = nn.LayerNorm(hidden_channels)

        # ── Global attention pool for 1D ──────────────────────────────────────
        self.global_pool_1d = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
        )

        # ── Decoders ─────────────────────────────────────────────────────────
        self.decoder_1d = nn.Sequential(
            FusedLinearLayerNormGELU(
                hidden_channels, hidden_channels, dropout, use_fused_ops
            ),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, self.temporal_bundle_k),
        )
        self.decoder_2d = nn.Sequential(
            FusedLinearLayerNormGELU(
                hidden_channels, hidden_channels, dropout, use_fused_ops
            ),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, self.temporal_bundle_k),
        )

    def _make_conv_dict(
        self, hidden_channels, dropout, conv_type, heads, edge_dim_1d=0, edge_dim_2d=0
    ):
        ed_1d = edge_dim_1d if edge_dim_1d > 0 else None
        ed_2d = edge_dim_2d if edge_dim_2d > 0 else None

        if conv_type == "gatv2":
            return {
                ("1d", "pipe", "1d"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ("2d", "surface", "2d"): GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_2d,
                ),
                ("1d", "couples", "2d"): GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ("2d", "couples", "1d"): GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
            }

        elif conv_type == "sage":
            return {
                ("1d", "pipe", "1d"): SAGEConv(hidden_channels, hidden_channels),
                ("2d", "surface", "2d"): SAGEConv(hidden_channels, hidden_channels),
                ("1d", "couples", "2d"): SAGEConv(
                    (hidden_channels, hidden_channels), hidden_channels
                ),
                ("2d", "couples", "1d"): SAGEConv(
                    (hidden_channels, hidden_channels), hidden_channels
                ),
            }

        elif conv_type == "transformer":
            return {
                ("1d", "pipe", "1d"): TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ("2d", "surface", "2d"): TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_2d,
                ),
                ("1d", "couples", "2d"): TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ("2d", "couples", "1d"): TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=ed_1d,
                ),
            }
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = {
            "1d": self.encoder_1d(x_dict["1d"]),
            "2d": self.encoder_2d(x_dict["2d"]),
        }

        if self.use_node_embeddings:
            node_ids_1d = (
                torch.arange(h["1d"].shape[0], device=h["1d"].device)
                % self.num_1d_nodes
            )
            node_ids_2d = (
                torch.arange(h["2d"].shape[0], device=h["2d"].device)
                % self.num_2d_nodes
            )
            h["1d"] = self.node_fuse_1d(
                torch.cat([h["1d"], self.node_embed_1d(node_ids_1d)], dim=-1)
            )
            h["2d"] = self.node_fuse_2d(
                torch.cat([h["2d"], self.node_embed_2d(node_ids_2d)], dim=-1)
            )

        for i, conv in enumerate(self.convs):
            h_normed = {
                "1d": self.norms_1d[i](h["1d"]),
                "2d": self.norms_2d[i](h["2d"]),
            }
            if edge_attr_dict and (self.edge_dim_1d > 0 or self.edge_dim_2d > 0):
                h_new = conv(h_normed, edge_index_dict, edge_attr_dict)
            else:
                h_new = conv(h_normed, edge_index_dict)

            h["1d"] = h["1d"] + F.dropout(
                h_new["1d"], p=self.dropout, training=self.training
            )
            h["2d"] = h["2d"] + F.dropout(
                h_new["2d"], p=self.dropout, training=self.training
            )

        h["1d"] = self.final_norm_1d(h["1d"])
        h["2d"] = self.final_norm_2d(h["2d"])

        n_total_1d = h["1d"].shape[0]
        if self.num_1d_nodes is not None and n_total_1d > self.num_1d_nodes:
            batch_size = n_total_1d // self.num_1d_nodes
            h_1d_view = h["1d"].view(batch_size, self.num_1d_nodes, -1)
            global_1d = h_1d_view.mean(dim=1, keepdim=True).expand_as(h_1d_view)
            h["1d"] = self.global_pool_1d(
                torch.cat([h_1d_view, global_1d], dim=-1)
            ).reshape(n_total_1d, -1)
        else:
            global_1d = h["1d"].mean(dim=0, keepdim=True).expand_as(h["1d"])
            h["1d"] = self.global_pool_1d(torch.cat([h["1d"], global_1d], dim=-1))

        delta_1d = self.decoder_1d(h["1d"])
        delta_2d = self.decoder_2d(h["2d"])

        if self.temporal_bundle_k == 1:
            delta_1d = delta_1d.squeeze(-1)
            delta_2d = delta_2d.squeeze(-1)

        return {"1d": delta_1d, "2d": delta_2d}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_random_graph(
    n_1d: int,
    n_2d: int,
    n_edges_1d: int,
    n_edges_2d: int,
    n_coupling: int,
    in_1d: int,
    in_2d: int,
    edge_dim_1d: int,
    device: torch.device,
):
    """
    Build a random HeteroData-like dict with synthetic node features and
    random edge indices.  Mirrors the structure expected by HeteroFloodGNN.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    def rand_edges(n_src, n_dst, n_e):
        src = torch.randint(0, n_src, (n_e,))
        dst = torch.randint(0, n_dst, (n_e,))
        return torch.stack([src, dst]).to(device)

    x_1d = torch.randn(n_1d, in_1d, generator=rng).to(device)
    x_2d = torch.randn(n_2d, in_2d, generator=rng).to(device)

    # Bidirectional edges (forward + reverse)
    ei_1d_fwd = rand_edges(n_1d, n_1d, n_edges_1d)
    ei_1d_rev = ei_1d_fwd.flip(0)
    ei_1d = torch.cat([ei_1d_fwd, ei_1d_rev], dim=1)

    ei_2d_fwd = rand_edges(n_2d, n_2d, n_edges_2d)
    ei_2d_rev = ei_2d_fwd.flip(0)
    ei_2d = torch.cat([ei_2d_fwd, ei_2d_rev], dim=1)

    ei_1to2 = rand_edges(n_1d, n_2d, n_coupling)
    ei_2to1 = rand_edges(
        n_2d, n_1d, n_coupling
    )  # independent draw, correct src/dst ranges

    edge_index_dict = {
        ("1d", "pipe", "1d"): ei_1d,
        ("2d", "surface", "2d"): ei_2d,
        ("1d", "couples", "2d"): ei_1to2,
        ("2d", "couples", "1d"): ei_2to1,
    }

    # TransformerConv with edge_dim > 0 requires edge_attr for EVERY edge type
    # that uses a conv with edge_dim set.  Supply random attrs for all of them.
    edge_attr_dict = None
    if edge_dim_1d > 0:
        edge_attr_dict = {
            ("1d", "pipe", "1d"): torch.randn(ei_1d.shape[1], edge_dim_1d).to(device),
            ("1d", "couples", "2d"): torch.randn(ei_1to2.shape[1], edge_dim_1d).to(
                device
            ),
            ("2d", "couples", "1d"): torch.randn(ei_2to1.shape[1], edge_dim_1d).to(
                device
            ),
        }

    return {"1d": x_1d, "2d": x_2d}, edge_index_dict, edge_attr_dict


# ─────────────────────────────────────────────────────────────────────────────
# Model specifications  (matching configs in file.py / train.py)
# ─────────────────────────────────────────────────────────────────────────────

#  Shared config params from BaseConfig
RAIN_LAG_STEPS = 5
RAIN_FUTURE_STEPS = 4
ATTENTION_HEADS = 16
CONV_TYPE = "transformer"
TEMPORAL_BUNDLE_K = 4
USE_EDGE_FEATURES = True
EDGE_DIM_1D = 4  # [hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v]

# Static feature dims: 6 for 1D, 9 for 2D
N_STATIC_1D = 6
N_STATIC_2D = 9

# Base dynamic dims (from FeatureBuilder)
N_DYNAMIC_1D_BASE = 19
N_DYNAMIC_2D_BASE = 6


def compute_in_channels(wl_prev_steps: int):
    """
    Compute total input channels matching FeatureBuilder exactly.

    FeatureBuilder.in_channels_1d  = x_1d_static.shape[1] + N_DYNAMIC_1D
      where N_DYNAMIC_1D = 19 + rain_lag_steps + wl_prev_steps + rain_future_steps

    FeatureBuilder.in_channels_2d  = x_2d_static.shape[1] + N_DYNAMIC_2D_BASE + rain_lag_steps
      where N_DYNAMIC_2D_BASE = 6 + wl_prev_steps + rain_future_steps
    """
    n_dynamic_1d = (
        N_DYNAMIC_1D_BASE  # 19 base
        + RAIN_LAG_STEPS  # global rain lags appended to 1D dyn
        + wl_prev_steps  # wl history (1D)
        + RAIN_FUTURE_STEPS
    )  # future rain leads (1D)

    n_dynamic_2d_base = (
        N_DYNAMIC_2D_BASE  # 6 base
        + wl_prev_steps  # wl history (2D)
        + RAIN_FUTURE_STEPS
    )  # future rain leads (2D)

    in_1d = N_STATIC_1D + n_dynamic_1d
    in_2d = (
        N_STATIC_2D + n_dynamic_2d_base + RAIN_LAG_STEPS
    )  # rain_lag added separately for 2D
    return in_1d, in_2d


MODEL_SPECS = {
    1: dict(
        n_1d=17,
        n_2d=3716,
        n_edges_1d=20,  # ~n_1d pipes
        n_edges_2d=12000,  # ~3×n_2d surface edges
        n_coupling=17,
        hidden_dim=64,
        num_layers=4,
        wl_prev_steps=2,
    ),
    2: dict(
        n_1d=198,
        n_2d=4299,
        n_edges_1d=250,
        n_edges_2d=14000,
        n_coupling=198,
        hidden_dim=96,
        num_layers=4,
        wl_prev_steps=10,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Build model from spec
# ─────────────────────────────────────────────────────────────────────────────


def build_model(spec: dict, device: torch.device) -> HeteroFloodGNN:
    in_1d, in_2d = compute_in_channels(spec["wl_prev_steps"])
    model = HeteroFloodGNN(
        in_channels_1d=in_1d,
        in_channels_2d=in_2d,
        hidden_channels=spec["hidden_dim"],
        num_layers=spec["num_layers"],
        dropout=0.0,  # eval mode: no dropout effect, but set 0 to be clean
        conv_type=CONV_TYPE,
        heads=ATTENTION_HEADS,
        use_fused_ops=True,
        edge_dim_1d=EDGE_DIM_1D if USE_EDGE_FEATURES else 0,
        edge_dim_2d=0,
        temporal_bundle_k=TEMPORAL_BUNDLE_K,
        use_node_embeddings=True,
        num_1d_nodes=spec["n_1d"],
        num_2d_nodes=spec["n_2d"],
    )
    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Timing loop
# ─────────────────────────────────────────────────────────────────────────────


def measure_inference(
    model, x_dict, edge_index_dict, edge_attr_dict, device, n_warmup=10, n_runs=50
):
    """
    Measure average single-step inference time.

    Warmup runs first to let CUDA JIT / cuDNN autotuner settle, then
    time n_runs forward passes using CUDA events for GPU-accurate timing.
    """
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # ── Warmup ────────────────────────────────────────────────────────
        for _ in range(n_warmup):
            _ = model(x_dict, edge_index_dict, edge_attr_dict)
        if use_cuda:
            torch.cuda.synchronize()

        # ── Timed runs ────────────────────────────────────────────────────
        times = []
        for _ in range(n_runs):
            if use_cuda:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                out = model(x_dict, edge_index_dict, edge_attr_dict)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(start_evt.elapsed_time(end_evt))  # ms
            else:
                t0 = time.perf_counter()
                out = model(x_dict, edge_index_dict, edge_attr_dict)
                times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    return out, times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Inference-time benchmark for HeteroFloodGNN models 1 & 2"
    )
    parser.add_argument(
        "--device", default="auto", help="'cuda', 'cpu', or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warm-up forward passes (default: 10)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of timed forward passes (default: 50)",
    )
    args = parser.parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 65)
    print("  HeteroFloodGNN  —  Inference Time Benchmark")
    print("=" * 65)
    print(
        f"  Device  : {device}"
        + (f"  ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )
    print(f"  Warmup  : {args.n_warmup} runs")
    print(f"  Timed   : {args.n_runs} runs")
    print("=" * 65)

    results = {}

    for model_id, spec in MODEL_SPECS.items():
        in_1d, in_2d = compute_in_channels(spec["wl_prev_steps"])

        print(
            f"\n  [Model {model_id}]  "
            f"n_1d={spec['n_1d']:>4d}  n_2d={spec['n_2d']:>5d}  "
            f"hidden={spec['hidden_dim']}  wl_prev={spec['wl_prev_steps']}"
        )
        print(f"            in_1d={in_1d}  in_2d={in_2d}")
        print()

        # Build model
        model = build_model(spec, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Build synthetic input
        x_dict, edge_index_dict, edge_attr_dict = make_random_graph(
            n_1d=spec["n_1d"],
            n_2d=spec["n_2d"],
            n_edges_1d=spec["n_edges_1d"],
            n_edges_2d=spec["n_edges_2d"],
            n_coupling=spec["n_coupling"],
            in_1d=in_1d,
            in_2d=in_2d,
            edge_dim_1d=EDGE_DIM_1D if USE_EDGE_FEATURES else 0,
            device=device,
        )

        out, times = measure_inference(
            model,
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            device=device,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )

        mean_ms = times.mean()
        std_ms = times.std()
        min_ms = times.min()
        max_ms = times.max()
        p95_ms = np.percentile(times, 95)

        # Output shapes: [n_nodes, temporal_bundle_k] or [n_nodes] if k==1
        shape_1d = tuple(out["1d"].shape)
        shape_2d = tuple(out["2d"].shape)

        results[model_id] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p95_ms": p95_ms,
            "shape_1d": shape_1d,
            "shape_2d": shape_2d,
        }

        print(f"\n  Output  →  1D shape: {shape_1d}   2D shape: {shape_2d}")
        print()
        print("  ┌─────────────────────────────────────────┐")
        print("  │  Full forward pass (1D + 2D combined)   │")
        print(f"  │  Mean  : {mean_ms:>8.3f} ms              │")
        print(f"  │  Std   : {std_ms:>8.3f} ms              │")
        print(f"  │  Min   : {min_ms:>8.3f} ms              │")
        print(f"  │  Max   : {max_ms:>8.3f} ms              │")
        print(f"  │  p95   : {p95_ms:>8.3f} ms              │")
        print("  └─────────────────────────────────────────┘")

        # Per-head timing estimate (proportional to node count)
        total_nodes = spec["n_1d"] + spec["n_2d"]
        frac_1d = spec["n_1d"] / total_nodes
        frac_2d = spec["n_2d"] / total_nodes
        est_1d_ms = mean_ms * frac_1d
        est_2d_ms = mean_ms * frac_2d

        print()
        print("  Estimated per-head inference time (proportional to node count):")
        print(
            f"    1D nodes ({spec['n_1d']:>4d} / {total_nodes} = {frac_1d * 100:.1f}%)  →  "
            f"~{est_1d_ms:.4f} ms"
        )
        print(
            f"    2D nodes ({spec['n_2d']:>5d} / {total_nodes} = {frac_2d * 100:.1f}%)  →  "
            f"~{est_2d_ms:.4f} ms"
        )

        results[model_id]["est_1d_ms"] = est_1d_ms
        results[model_id]["est_2d_ms"] = est_2d_ms

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  SUMMARY  —  Inference Times (mean ± std over {args.n_runs} runs)")
    print("=" * 65)
    print(f"  {'Variant':<30}  {'Mean (ms)':>10}  {'Std (ms)':>9}  {'p95 (ms)':>9}")
    print("  " + "-" * 61)

    rows = [
        ("Model 1  —  1D nodes", 1, "est_1d_ms"),
        ("Model 1  —  2D nodes", 1, "est_2d_ms"),
        ("Model 2  —  1D nodes", 2, "est_1d_ms"),
        ("Model 2  —  2D nodes", 2, "est_2d_ms"),
    ]
    for label, mid, key in rows:
        r = results[mid]
        mean = r[key]
        # std and p95 scaled by same fraction
        frac = mean / r["mean_ms"]
        std = r["std_ms"] * frac
        p95 = r["p95_ms"] * frac
        print(f"  {label:<30}  {mean:>10.4f}  {std:>9.4f}  {p95:>9.4f}")

    print()
    print("  Full forward pass (1D + 2D together):")
    print(
        f"  {'Model 1  —  full pass':<30}  "
        f"{results[1]['mean_ms']:>10.3f}  "
        f"{results[1]['std_ms']:>9.3f}  "
        f"{results[1]['p95_ms']:>9.3f}"
    )
    print(
        f"  {'Model 2  —  full pass':<30}  "
        f"{results[2]['mean_ms']:>10.3f}  "
        f"{results[2]['std_ms']:>9.3f}  "
        f"{results[2]['p95_ms']:>9.3f}"
    )
    print("=" * 65)
    print()
    print("  NOTE: 1D/2D per-head times are proportional estimates.")
    print("        The GNN processes both node types in a single fused")
    print("        forward pass; they cannot be isolated without")
    print("        architectural changes.")
    print("=" * 65)


if __name__ == "__main__":
    main()
