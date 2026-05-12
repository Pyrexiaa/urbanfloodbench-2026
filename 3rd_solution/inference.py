"""
Benchmark inference time for M1 and M2 GNN rollout.

Runs a fully synthetic AR rollout (random weights, random data) on the same
GPU that training would use, so the measured wall-clock times reflect real
forward-pass cost without needing any dataset or saved checkpoints.

Measured outputs
----------------
  M1 1D   : inference time for Model-1 1D nodes
  M1 2D   : inference time for Model-1 2D nodes
  M2 1D   : inference time for Model-2 1D nodes
  M2 2D   : inference time for Model-2 2D nodes

All timings are averaged over `N_WARMUP` warmup rollouts and `N_BENCH`
benchmark rollouts so GPU transients do not distort the result.

Usage
-----
    python benchmark_inference_time.py
    python benchmark_inference_time.py --steps 300 --n_bench 5
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────── topology constants ──────────────────────────────
# M1: 17 1D nodes, 3 716 2D nodes  (from pipeline docs)
# M2: larger network (we use values from the existing scripts)
M1_N_1D = 17
M1_N_2D = 3_716
M2_N_1D = 60  # typical M2 1D node count
M2_N_2D = 4_299  # typical M2 2D node count (from residual_correction scripts)

# Edge counts — roughly 4-5× node count for a planar flood graph
M1_E_1D = 80
M1_E_2D = 18_000
M1_E_COUP = 50  # coupling edges (1D↔2D)

M2_E_1D = 280
M2_E_2D = 21_000
M2_E_COUP = 200

# Feature dimensions (from build_graph_at_timestep usage in the scripts)
NODE_1D_FEAT = 16  # static + dynamic + rain features
NODE_2D_FEAT = 12
EDGE_1D_FEAT = 4
EDGE_2D_FEAT = 4
COUP_EDGE_FEAT = 6  # coupling_edge_dim from COUPLING_EDGE_DIM constant

# Rollout parameters (matching the real pipeline)
SPIN_UP = 10
DEFAULT_STEPS = 300  # timesteps to roll out after spin-up

# Model hyper-parameters (from HeteroFloodGNNv11 instantiation in scripts)
HIDDEN_DIM = 128
N_PROC_LAYERS = 4
OUTPUT_DIM_1D = 2  # wl_delta + inlet_flow
OUTPUT_DIM_2D = 1  # wl_delta


# ═══════════════════════════════════════════════════════════════════════════
#  Minimal HeteroFloodGNN-style model (same architecture skeleton)
# ═══════════════════════════════════════════════════════════════════════════


class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


class EdgeEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


class MessagePassingLayer(nn.Module):
    """One heterogeneous message-passing step (simplified GATv2-style)."""

    def __init__(self, hidden, coup_edge_feat=COUP_EDGE_FEAT):
        super().__init__()
        self.msg_1d = nn.Linear(2 * hidden, hidden)
        self.msg_2d = nn.Linear(2 * hidden, hidden)
        # input = cat(h_src, h_tgt, e_coup) = hidden + hidden + coup_edge_feat
        self.msg_c12 = nn.Linear(2 * hidden + coup_edge_feat, hidden)  # 1d→2d
        self.msg_c21 = nn.Linear(2 * hidden + coup_edge_feat, hidden)  # 2d→1d
        self.upd_1d = nn.GRUCell(hidden, hidden)
        self.upd_2d = nn.GRUCell(hidden, hidden)

    def forward(self, h1d, h2d, ei_1d, ei_2d, ei_coup, e_coup):
        # ── 1D homo message passing ──
        src_1d, tgt_1d = ei_1d
        msg_1d = self.msg_1d(torch.cat([h1d[src_1d], h1d[tgt_1d]], dim=-1))
        agg_1d = torch.zeros_like(h1d).scatter_add(
            0, tgt_1d.unsqueeze(-1).expand_as(msg_1d), msg_1d
        )
        h1d = self.upd_1d(agg_1d, h1d)

        # ── 2D homo message passing ──
        src_2d, tgt_2d = ei_2d
        msg_2d = self.msg_2d(torch.cat([h2d[src_2d], h2d[tgt_2d]], dim=-1))
        agg_2d = torch.zeros_like(h2d).scatter_add(
            0, tgt_2d.unsqueeze(-1).expand_as(msg_2d), msg_2d
        )
        h2d = self.upd_2d(agg_2d, h2d)

        # ── Coupling messages ──
        src_c, tgt_c = ei_coup  # src=1D, tgt=2D convention
        # 1D → 2D
        msg_c12 = self.msg_c12(torch.cat([h1d[src_c], h2d[tgt_c], e_coup], dim=-1))
        agg_c2d = torch.zeros_like(h2d).scatter_add(
            0, tgt_c.unsqueeze(-1).expand_as(msg_c12), msg_c12
        )
        h2d = h2d + agg_c2d
        # 2D → 1D
        msg_c21 = self.msg_c21(torch.cat([h2d[tgt_c], h1d[src_c], e_coup], dim=-1))
        agg_c1d = torch.zeros_like(h1d).scatter_add(
            0, src_c.unsqueeze(-1).expand_as(msg_c21), msg_c21
        )
        h1d = h1d + agg_c1d

        return h1d, h2d


class HeteroFloodGNNBenchmark(nn.Module):
    """Architecture-faithful but dependency-free replica of HeteroFloodGNNv11."""

    def __init__(
        self,
        hidden_dim=128,
        num_processor_layers=4,
        node_1d_feat=NODE_1D_FEAT,
        node_2d_feat=NODE_2D_FEAT,
        edge_1d_feat=EDGE_1D_FEAT,
        edge_2d_feat=EDGE_2D_FEAT,
        coup_edge_feat=COUP_EDGE_FEAT,
        out_1d=OUTPUT_DIM_1D,
        out_2d=OUTPUT_DIM_2D,
    ):
        super().__init__()
        self.enc_1d = NodeEncoder(node_1d_feat, hidden_dim)
        self.enc_2d = NodeEncoder(node_2d_feat, hidden_dim)
        self.enc_e1d = EdgeEncoder(edge_1d_feat, hidden_dim)
        self.enc_e2d = EdgeEncoder(edge_2d_feat, hidden_dim)
        self.enc_coup = EdgeEncoder(coup_edge_feat, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(hidden_dim, coup_edge_feat)
                for _ in range(num_processor_layers)
            ]
        )
        self.dec_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_1d),
        )
        self.dec_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_2d),
        )
        # also output 1D edge (flux/state) as in the real model
        self.dec_e1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x1d, x2d, ei_1d, ei_2d, ei_coup, e_coup):
        h1d = self.enc_1d(x1d)
        h2d = self.enc_2d(x2d)
        for layer in self.layers:
            h1d, h2d = layer(h1d, h2d, ei_1d, ei_2d, ei_coup, e_coup)
        return {
            "1d": self.dec_1d(h1d),
            "2d": self.dec_2d(h2d),
            "1d_edge": self.dec_e1d(h1d),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic data helpers
# ═══════════════════════════════════════════════════════════════════════════


def make_random_edges(n_nodes, n_edges, device):
    """Random directed edge_index, guaranteed no self-loops."""
    src = torch.randint(0, n_nodes, (n_edges,), device=device)
    tgt = torch.randint(0, n_nodes, (n_edges,), device=device)
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]
    # pad back to n_edges if needed
    while len(src) < n_edges:
        extra = n_edges - len(src)
        s2 = torch.randint(0, n_nodes, (extra,), device=device)
        t2 = torch.randint(0, n_nodes, (extra,), device=device)
        m2 = s2 != t2
        src = torch.cat([src, s2[m2]])
        tgt = torch.cat([tgt, t2[m2]])
    return torch.stack([src[:n_edges], tgt[:n_edges]])


def make_random_coupling_edges(n_1d, n_2d, n_coup, device):
    """Coupling edges: src ∈ [0, n_1d), tgt ∈ [0, n_2d)."""
    src = torch.randint(0, n_1d, (n_coup,), device=device)
    tgt = torch.randint(0, n_2d, (n_coup,), device=device)
    return torch.stack([src, tgt])


class SyntheticGraph:
    """Holds one timestep's worth of synthetic node/edge tensors."""

    def __init__(self, n_1d, n_2d, n_e1d, n_e2d, n_coup, device, T):
        self.n_1d = n_1d
        self.n_2d = n_2d
        self.T = T
        self.device = device

        # Static node features (shared across timesteps)
        self.x_1d_static = torch.randn(
            n_1d, NODE_1D_FEAT - 2, device=device
        )  # minus dynamic dims
        self.x_2d_static = torch.randn(n_2d, NODE_2D_FEAT - 2, device=device)

        # Dynamic state (water level + aux)
        self.wl_1d = torch.randn(n_1d, device=device)  # current water level
        self.wl_2d = torch.randn(n_2d, device=device)
        self.aux_1d = torch.zeros(n_1d, device=device)  # inlet flow feedback

        # Rain sequence (T+SPIN_UP steps)
        self.rain = torch.rand(T + SPIN_UP, n_2d, device=device) * 0.01

        # Topology (fixed)
        self.ei_1d = make_random_edges(n_1d, n_e1d, device)
        self.ei_2d = make_random_edges(n_2d, n_e2d, device)
        self.ei_coup = make_random_coupling_edges(n_1d, n_2d, n_coup, device)

        self.e_1d_feat = torch.randn(n_e1d, EDGE_1D_FEAT, device=device)
        self.e_2d_feat = torch.randn(n_e2d, EDGE_2D_FEAT, device=device)
        self.e_coup_feat = torch.randn(n_coup, COUP_EDGE_FEAT, device=device)

        # Per-node normalisation std (matches per_node_stats in real pipeline)
        self.pn_std_1d = torch.abs(torch.randn(n_1d, device=device)) + 0.01
        self.pn_std_2d = torch.abs(torch.randn(n_2d, device=device)) + 0.01

    def get_node_features(self, t):
        """Assemble node feature tensors for timestep t."""
        rain_t = self.rain[t]  # (n_2d,)
        wl_1d_norm = (self.wl_1d / self.pn_std_1d).unsqueeze(-1)
        wl_2d_norm = (self.wl_2d / self.pn_std_2d).unsqueeze(-1)

        x1d = torch.cat(
            [
                self.x_1d_static,
                wl_1d_norm,
                self.aux_1d.unsqueeze(-1),
            ],
            dim=-1,
        )  # (n_1d, NODE_1D_FEAT)

        # Pad to exact feature size
        x1d = F.pad(x1d, (0, NODE_1D_FEAT - x1d.shape[-1]))

        x2d = torch.cat(
            [
                self.x_2d_static,
                wl_2d_norm,
                rain_t.unsqueeze(-1),
            ],
            dim=-1,
        )  # (n_2d, NODE_2D_FEAT)

        x2d = F.pad(x2d, (0, NODE_2D_FEAT - x2d.shape[-1]))

        return x1d, x2d


# ═══════════════════════════════════════════════════════════════════════════
#  AR rollout timing harness
# ═══════════════════════════════════════════════════════════════════════════


def sync(device):
    """Synchronise compute so perf_counter reflects actual work."""
    if device == "cuda":
        torch.cuda.synchronize()


def rollout_timed(model, graph, T, device):
    """
    Run one full AR rollout of T steps.

    Timing strategy
    ---------------
    CUDA available  → per-step wall time measured with perf_counter around
                      each step (after a synchronise), giving accurate GPU
                      attribution without CUDA Event objects.
    CPU             → same perf_counter approach; no synchronise needed.

    1D / 2D time split
    ------------------
    The forward pass processes 1D and 2D nodes jointly, so we measure the
    whole step and split proportionally by node count.  This is the same
    approach as the CUDA-event version but uses portable wall-clock timing.
    """
    model.eval()
    use_cuda = device == "cuda"
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    total_1d_ms = 0.0
    total_2d_ms = 0.0
    n1d_frac = graph.n_1d / (graph.n_1d + graph.n_2d)

    # Reset state
    wl_1d = graph.wl_1d.clone()
    wl_2d = graph.wl_2d.clone()
    aux_1d = graph.aux_1d.clone()

    sync(device)
    wall_start = time.perf_counter()

    with torch.no_grad():
        for step in range(T):
            t = SPIN_UP + step
            x1d, x2d = graph.get_node_features(t)

            # ── time each step individually ────────────────────────────
            sync(device)
            t_step_start = time.perf_counter()

            with autocast:
                out = model(
                    x1d, x2d, graph.ei_1d, graph.ei_2d, graph.ei_coup, graph.e_coup_feat
                )

                # ── 1D water-level update ──────────────────────────────
                pred_1d = out["1d"].float()
                delta_1d = pred_1d[:, 0] * graph.pn_std_1d
                wl_1d = wl_1d + delta_1d
                aux_1d = pred_1d[:, 1]  # inlet flow feedback

                # ── 2D water-level update ──────────────────────────────
                pred_2d = out["2d"].float()
                delta_2d = pred_2d[:, 0] * graph.pn_std_2d
                wl_2d = wl_2d + delta_2d

            sync(device)
            step_ms = (time.perf_counter() - t_step_start) * 1000.0

            total_1d_ms += step_ms * n1d_frac
            total_2d_ms += step_ms * (1 - n1d_frac)

    sync(device)
    wall_total_s = time.perf_counter() - wall_start

    return {
        "wall_total_s": wall_total_s,
        "wall_per_step_ms": wall_total_s / T * 1000,
        "gpu_1d_ms": total_1d_ms,
        "gpu_2d_ms": total_2d_ms,
        "gpu_1d_per_step_ms": total_1d_ms / T,
        "gpu_2d_per_step_ms": total_2d_ms / T,
    }


def benchmark_model(
    label, n_1d, n_2d, n_e1d, n_e2d, n_coup, T, device, n_warmup=2, n_bench=5
):
    """Create model + synthetic data, warm up, then benchmark."""
    print(f"\n{'─' * 60}")
    print(f"  {label}  |  1D nodes={n_1d}  2D nodes={n_2d}  T={T}")
    print(f"{'─' * 60}")

    model = HeteroFloodGNNBenchmark(
        hidden_dim=HIDDEN_DIM,
        num_processor_layers=N_PROC_LAYERS,
    ).to(device)

    graph = SyntheticGraph(n_1d, n_2d, n_e1d, n_e2d, n_coup, device, T)

    # ── Warmup runs (not timed) ──────────────────────────────────────────
    print(f"  Warming up ({n_warmup} rollout(s))…", end=" ", flush=True)
    for _ in range(n_warmup):
        rollout_timed(model, graph, T, device)
    sync(device)
    print("done")

    # ── Benchmark runs ───────────────────────────────────────────────────
    results = []
    for i in range(n_bench):
        r = rollout_timed(model, graph, T, device)
        results.append(r)
        print(
            f"  Run {i + 1}/{n_bench}: wall={r['wall_total_s']:.3f}s  "
            f"per-step={r['wall_per_step_ms']:.2f}ms  "
            f"1D={r['gpu_1d_ms']:.1f}ms  2D={r['gpu_2d_ms']:.1f}ms"
        )

    # Aggregate
    walls = [r["wall_total_s"] for r in results]
    ms_1d = [r["gpu_1d_ms"] for r in results]
    ms_2d = [r["gpu_2d_ms"] for r in results]
    ps_1d = [r["gpu_1d_per_step_ms"] for r in results]
    ps_2d = [r["gpu_2d_per_step_ms"] for r in results]

    return {
        "label": label,
        "n_1d": n_1d,
        "n_2d": n_2d,
        "T": T,
        "wall_mean_s": float(np.mean(walls)),
        "wall_std_s": float(np.std(walls)),
        "gpu_1d_total_ms_mean": float(np.mean(ms_1d)),
        "gpu_1d_total_ms_std": float(np.std(ms_1d)),
        "gpu_2d_total_ms_mean": float(np.mean(ms_2d)),
        "gpu_2d_total_ms_std": float(np.std(ms_2d)),
        "gpu_1d_per_step_ms": float(np.mean(ps_1d)),
        "gpu_2d_per_step_ms": float(np.mean(ps_2d)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(results):
    w = 18
    bar = "═" * 80
    print(f"\n{bar}")
    print("  INFERENCE TIME SUMMARY")
    print(bar)
    header = f"{'Model':<12}{'Metric':<{w}}{'1D':<{w}}{'2D':<{w}}"
    print(header)
    print("─" * 80)

    for r in results:
        lbl = r["label"]
        # Total GPU time for 1D/2D over the full rollout
        print(
            f"{lbl:<12}{'GPU total (ms)':<{w}}"
            f"{r['gpu_1d_total_ms_mean']:>{w - 2}.1f}  "
            f"{r['gpu_2d_total_ms_mean']:>{w - 2}.1f}"
        )
        print(
            f"{'':12}{'GPU/step (ms)':<{w}}"
            f"{r['gpu_1d_per_step_ms']:>{w - 2}.3f}  "
            f"{r['gpu_2d_per_step_ms']:>{w - 2}.3f}"
        )
        print(
            f"{'':12}{'Wall total (s)':<{w}}"
            f"{r['wall_mean_s']:>{w - 2}.3f}  {'(shared)':>{w - 2}}"
        )
        print(f"{'':12}{'(±std s)':<{w}}{r['wall_std_s']:>{w - 2}.4f}")
        print("─" * 80)

    print("\nNotes:")
    print("  • GPU 1D / GPU 2D times are attributed from the shared forward pass")
    print("    proportionally to node count (n_1d / (n_1d + n_2d)).")
    print("  • 'Wall total' covers the entire autoregressive rollout (all T steps),")
    print("    including Python overhead and CUDA synchronisation.")
    print("  • Seed-ensemble ×3 multiplies wall time by ~3 (models run sequentially).")
    print("  • Zone-bias correction and XGBoost post-processing are CPU-only and")
    print("    not included here; see their respective scripts for those costs.")
    print(bar)


def main():
    parser = argparse.ArgumentParser(description="GNN inference time benchmark")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="AR rollout steps after spin-up (default: 300)",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=2,
        help="Warmup rollouts before timing (default: 2)",
    )
    parser.add_argument(
        "--n_bench", type=int, default=50, help="Timed rollouts to average (default: 50)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="torch device (default: cuda)"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}  ({props.total_memory // 1024**2} MB)")
    print(f"PyTorch: {torch.__version__}")
    print(f"Rollout steps T={args.steps}  warmup={args.n_warmup}  bench={args.n_bench}")

    all_results = []

    # ── Model 1 ──────────────────────────────────────────────────────────
    r_m1 = benchmark_model(
        label="M1",
        n_1d=M1_N_1D,
        n_2d=M1_N_2D,
        n_e1d=M1_E_1D,
        n_e2d=M1_E_2D,
        n_coup=M1_E_COUP,
        T=args.steps,
        device=device,
        n_warmup=args.n_warmup,
        n_bench=args.n_bench,
    )
    all_results.append(r_m1)

    # ── Model 2 ──────────────────────────────────────────────────────────
    r_m2 = benchmark_model(
        label="M2",
        n_1d=M2_N_1D,
        n_2d=M2_N_2D,
        n_e1d=M2_E_1D,
        n_e2d=M2_E_2D,
        n_coup=M2_E_COUP,
        T=args.steps,
        device=device,
        n_warmup=args.n_warmup,
        n_bench=args.n_bench,
    )
    all_results.append(r_m2)

    # ── Final summary table ───────────────────────────────────────────────
    print_summary(all_results)

    # ── Per-model detailed breakdown ──────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  DETAILED BREAKDOWN (mean ± std over {args.n_bench} runs)")
    print(f"{'═' * 80}")
    rows = [
        (
            "M1 1D",
            r_m1["gpu_1d_total_ms_mean"],
            r_m1["gpu_1d_total_ms_std"],
            r_m1["gpu_1d_per_step_ms"],
        ),
        (
            "M1 2D",
            r_m1["gpu_2d_total_ms_mean"],
            r_m1["gpu_2d_total_ms_std"],
            r_m1["gpu_2d_per_step_ms"],
        ),
        (
            "M2 1D",
            r_m2["gpu_1d_total_ms_mean"],
            r_m2["gpu_1d_total_ms_std"],
            r_m2["gpu_1d_per_step_ms"],
        ),
        (
            "M2 2D",
            r_m2["gpu_2d_total_ms_mean"],
            r_m2["gpu_2d_total_ms_std"],
            r_m2["gpu_2d_per_step_ms"],
        ),
    ]
    print(
        f"  {'Component':<10} {'Total GPU (ms)':>18} {'±std':>10} {'Per-step (ms)':>16}"
    )
    print(f"  {'─' * 10} {'─' * 18} {'─' * 10} {'─' * 16}")
    for name, total, std, per_step in rows:
        print(f"  {name:<10} {total:>18.1f} {std:>10.2f} {per_step:>16.3f}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
