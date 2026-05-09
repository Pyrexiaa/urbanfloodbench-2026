"""
Urban Flood Bench — Inference Time Benchmark
=============================================
Runs synthetic training forward passes for both Model 1 and Model 2
(SimpleFloodTGCN and EdgeAwareFloodModel) on the same GPU, then measures
per-step inference time broken down by node type (1D / 2D).

No real dataset required: graph topology and model weights are randomised.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Running on: {DEVICE}")

# ---------------------------------------------------------------------------
# Synthetic graph dimensions  (approximate Model_1 / Model_2 sizes)
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# You can adjust these to match your real graph sizes
MODEL1_NUM_1D = 17  # 1-D (sewer/channel) nodes
MODEL1_NUM_2D = 3716 # 2-D (surface) nodes
TOTAL_MODEL1_NODES = MODEL1_NUM_1D + MODEL1_NUM_2D

MODEL2_NUM_1D = 198  # 1-D (sewer/channel) nodes
MODEL2_NUM_2D = 4299  # 2-D (surface) nodes
TOTAL_MODEL2_NODES = MODEL2_NUM_1D + MODEL2_NUM_2D

# Edge topology: roughly 3× directed edges per node
AVG_EDGES_PER_NODE = 3
MODEL1_NUM_EDGES = TOTAL_MODEL1_NODES * AVG_EDGES_PER_NODE
MODEL2_NUM_EDGES = TOTAL_MODEL2_NODES * AVG_EDGES_PER_NODE

# Model hyper-parameters
HIDDEN_DIM = 512
EDGE_HIDDEN_DIM = 96
NUM_LAYERS = 2
DROPOUT = 0.2

# Node / edge feature dims (from Preprocessor with full MODEL_CONFIG)
# base: depth + rain + inlet_flow + water_volume = 4
# engineered: depth_slope + rolling_rain = 2
NODE_FEAT_DIM = 6  # 4 base + 2 engineered
EDGE_FEAT_DIM = 4  # flow + velocity + obs_mask + delta_flow (engineered)
STATIC_NODE_DIM = 8  # matches GraphBuilder.get_static_features() output width
STATIC_EDGE_DIM = 9  # matches GraphBuilder.get_edge_static_features()

SEQ_LENGTH = 10  # warm-up length
ROLLOUT_STEPS = 90  # inference horizon (Model 1); set 180 for Model 2 tests

WARMUP_RUNS = 10  # GPU warm-up iterations (discarded)
BENCH_RUNS = 50  # timed iterations


def amp_ctx():
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if DEVICE.type == "cuda"
        else nullcontext()
    )


# ===========================================================================
# Helper: build random graph data
# ===========================================================================


def make_edge_index(num_nodes, num_edges):
    """Random directed edge index with no self-loops."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    # Avoid src == dst
    mask = src == dst
    dst[mask] = (dst[mask] + 1) % num_nodes
    return torch.stack([src, dst], dim=0).to(DEVICE)


def build_gcn_csr(edge_index, num_nodes, device, dtype=torch.float16):
    """Normalised adjacency in CSR format (identical to original)."""
    edge_index = edge_index.to(device).long()
    idx = torch.arange(num_nodes, device=device)
    loops = torch.stack([idx, idx], dim=0)
    ei = torch.cat([edge_index, loops], dim=1)
    row, col = ei
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    w = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).to(dtype)
    A_coo = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=w,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()
    return A_coo.to_sparse_csr()


# ===========================================================================
# Model 1 — SimpleFloodTGCN  (GRU-based TGCN, node-only edges)
# ===========================================================================


class SimpleTGCNCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.gates = nn.Linear(in_dim + hidden_dim, hidden_dim * 2)
        self.cand = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, h):
        x = self.drop(x)
        combined = torch.cat([x, h], dim=1)
        gates = torch.sparse.mm(adj, self.gates(combined))
        r, u = torch.sigmoid(gates).chunk(2, dim=1)
        cand = torch.sparse.mm(adj, self.cand(torch.cat([x, r * h], dim=1)))
        c = torch.tanh(cand)
        return u * h + (1 - u) * c


class SimpleFloodTGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_features,
        hidden_dim,
        num_layers,
        dropout,
        static_features,
        adj,
        num_1d,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_1d = num_1d
        self.register_buffer("static_feat", static_features)
        self.register_buffer("adj", adj)

        self.fuse = nn.Sequential(
            nn.Linear(in_features + static_features.shape[1], hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(
                SimpleTGCNCell(
                    hidden_dim // 2 if i == 0 else hidden_dim, hidden_dim, dropout
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def _fuse(self, x):
        return self.fuse(torch.cat([x, self.static_feat], dim=1))

    def warmup(self, node_seq):
        T, N, _ = node_seq.shape
        h = [
            torch.zeros(N, self.hidden_dim, device=node_seq.device)
            for _ in range(self.num_layers)
        ]
        for t in range(T):
            x_t = self._fuse(node_seq[t])
            for i in range(self.num_layers):
                h_new = self.norms[i](self.cells[i](x_t, self.adj, h[i]))
                x_t = h_new if i == 0 else h_new + x_t
                h[i] = h_new
        return h

    def decode(self, state, current_depth):
        h = state[-1]
        d1 = self.head_1d(h[: self.num_1d])
        d2 = self.head_2d(h[self.num_1d :])
        delta = torch.cat([d1, d2], dim=0)
        return current_depth + self.res_scale * delta

    def step(self, state, node_dyn):
        x_t = self._fuse(node_dyn)
        new_state = []
        for i in range(self.num_layers):
            h_new = self.norms[i](self.cells[i](x_t, self.adj, state[i]))
            x_t = h_new if i == 0 else h_new + x_t
            new_state.append(h_new)
        return new_state


# ===========================================================================
# Model 2 — EdgeAwareFloodModel  (2-hop message passing + GRU)
# ===========================================================================


class EdgeAwareFloodModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_edges,
        node_in_dim,
        edge_in_dim,
        node_hidden_dim,
        edge_hidden_dim,
        dropout,
        node_static,
        edge_static,
        edge_index,
        num_1d,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_1d = num_1d
        self.register_buffer("node_static", node_static)
        self.register_buffer("edge_static", edge_static)
        self.register_buffer("edge_index", edge_index.long())

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim + node_static.shape[1], node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim + edge_static.shape[1], edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(node_hidden_dim * 2 + edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )
        self.msg_to_node = nn.Sequential(
            nn.Linear(edge_hidden_dim, node_hidden_dim),
            nn.GELU(),
        )
        self.message_mlp_2 = nn.Sequential(
            nn.Linear(node_hidden_dim * 2 + edge_hidden_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )
        self.msg_to_node_2 = nn.Sequential(
            nn.Linear(edge_hidden_dim, node_hidden_dim),
            nn.GELU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_hidden_dim * 2, node_hidden_dim),
            nn.GELU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(edge_hidden_dim * 2, edge_hidden_dim),
            nn.GELU(),
        )
        self.node_gru_1d = nn.GRUCell(node_hidden_dim, node_hidden_dim)
        self.node_gru_2d = nn.GRUCell(node_hidden_dim, node_hidden_dim)
        self.edge_gru = nn.GRUCell(edge_hidden_dim, edge_hidden_dim)
        self.node_norm = nn.LayerNorm(node_hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_hidden_dim)
        self.head_1d = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(node_hidden_dim // 2, 1),
        )
        self.head_2d = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(node_hidden_dim // 2, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.edge_dyn_dim = edge_in_dim  # dynamic part only

    def _zero_state(self):
        return {
            "node": torch.zeros(
                self.num_nodes,
                self.node_hidden_dim,
                device=next(self.parameters()).device,
            ),
            "edge": torch.zeros(
                self.num_edges,
                self.edge_hidden_dim,
                device=next(self.parameters()).device,
            ),
        }

    def _step_core(self, state, node_dyn, edge_dyn):
        node_lat = self.node_encoder(torch.cat([node_dyn, self.node_static], dim=1))
        edge_lat = self.edge_encoder(torch.cat([edge_dyn, self.edge_static], dim=1))
        node_base = node_lat + state["node"]
        edge_base = edge_lat + state["edge"]

        src, dst = self.edge_index
        in_deg = (
            torch.bincount(dst, minlength=self.num_nodes)
            .float()
            .clamp(min=1)
            .unsqueeze(1)
        )

        # Hop 1
        msg = self.message_mlp(
            torch.cat([node_base[src], node_base[dst], edge_base], dim=1)
        )
        agg = torch.zeros(
            self.num_nodes,
            self.edge_hidden_dim,
            device=node_base.device,
            dtype=msg.dtype,
        )
        agg.index_add_(0, dst, msg)
        agg = agg / in_deg
        node_after_hop1 = node_base + self.msg_to_node(agg)

        # Hop 2
        msg2 = self.message_mlp_2(
            torch.cat([node_after_hop1[src], node_after_hop1[dst], edge_base], dim=1)
        )
        agg2 = torch.zeros(
            self.num_nodes,
            self.edge_hidden_dim,
            device=node_base.device,
            dtype=msg2.dtype,
        )
        agg2.index_add_(0, dst, msg2)
        agg2 = agg2 / in_deg

        node_in = self.node_update(
            torch.cat([node_after_hop1, self.msg_to_node_2(agg2)], dim=1)
        )
        edge_in = self.edge_update(torch.cat([edge_base, msg], dim=1))

        node_state = state["node"]
        next_node = torch.empty_like(node_state)
        next_node[: self.num_1d] = self.node_gru_1d(
            node_in[: self.num_1d], node_state[: self.num_1d]
        )
        next_node[self.num_1d :] = self.node_gru_2d(
            node_in[self.num_1d :], node_state[self.num_1d :]
        )
        next_node = self.node_norm(next_node)
        next_edge = self.edge_norm(self.edge_gru(edge_in, state["edge"]))
        return {"node": next_node, "edge": next_edge}

    def warmup(self, node_seq, edge_seq):
        state = self._zero_state()
        for t in range(node_seq.shape[0]):
            state = self._step_core(state, node_seq[t], edge_seq[t])
        return state

    def decode(self, state, current_depth):
        d1 = self.head_1d(state["node"][: self.num_1d])
        d2 = self.head_2d(state["node"][self.num_1d :])
        delta = torch.cat([d1, d2], dim=0)
        return current_depth + self.res_scale * delta

    def step(self, state, node_dyn, edge_dyn):
        return self._step_core(state, node_dyn, edge_dyn)


# ===========================================================================
# Build random synthetic tensors
# ===========================================================================

print("\n--- Building synthetic graph and tensors ---")

model1_edge_index = make_edge_index(TOTAL_MODEL1_NODES, MODEL1_NUM_EDGES)
model1_adj_dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
model1_adj = build_gcn_csr(model1_edge_index, TOTAL_MODEL1_NODES, DEVICE, dtype=model1_adj_dtype)

model2_edge_index = make_edge_index(TOTAL_MODEL2_NODES, MODEL2_NUM_EDGES)
model2_adj_dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
model2_adj = build_gcn_csr(model2_edge_index, TOTAL_MODEL2_NODES, DEVICE, dtype=model2_adj_dtype)

# Static features: (N, STATIC_NODE_DIM); last col = 0 for 1-D, 1 for 2-D
model1_static_node = torch.randn(TOTAL_MODEL1_NODES, STATIC_NODE_DIM, device=DEVICE)
model1_static_node[MODEL1_NUM_1D:, -1] = 1.0  # 2-D flag
model1_static_node[:MODEL1_NUM_1D, -1] = 0.0  # 1-D flag

model2_static_node = torch.randn(TOTAL_MODEL2_NODES, STATIC_NODE_DIM, device=DEVICE)
model2_static_node[MODEL2_NUM_1D:, -1] = 1.0  # 2-D flag
model2_static_node[:MODEL2_NUM_1D, -1] = 0.0  # 1-D flag

model1_static_edge = torch.randn(MODEL1_NUM_EDGES, STATIC_EDGE_DIM, device=DEVICE)
model2_static_edge = torch.randn(MODEL2_NUM_EDGES, STATIC_EDGE_DIM, device=DEVICE)

# Node input sequences: (SEQ_LENGTH, TOTAL_NODES, NODE_FEAT_DIM)
model1_x_node_seq = torch.randn(SEQ_LENGTH, TOTAL_MODEL1_NODES, NODE_FEAT_DIM, device=DEVICE)
model2_x_node_seq = torch.randn(SEQ_LENGTH, TOTAL_MODEL2_NODES, NODE_FEAT_DIM, device=DEVICE)

# Edge dynamic sequences (for Model 2): (SEQ_LENGTH, NUM_EDGES, EDGE_FEAT_DIM)
model2_x_edge_seq = torch.randn(SEQ_LENGTH, MODEL2_NUM_EDGES, EDGE_FEAT_DIM, device=DEVICE)

# Future rain for rollout: (ROLLOUT_STEPS, TOTAL_NODES, 1)
model1_rain_future = torch.clamp(torch.randn(ROLLOUT_STEPS, TOTAL_MODEL1_NODES, 1, device=DEVICE), 0)
model2_rain_future = torch.clamp(torch.randn(ROLLOUT_STEPS, TOTAL_MODEL2_NODES, 1, device=DEVICE), 0)

print(f"  Model1 Nodes: {TOTAL_MODEL1_NODES}  (1-D={MODEL1_NUM_1D}, 2-D={MODEL1_NUM_2D})")
print(f"  Model1 Edges: {MODEL1_NUM_EDGES}")
print(f"  Model2 Nodes: {TOTAL_MODEL2_NODES}  (1-D={MODEL2_NUM_1D}, 2-D={MODEL2_NUM_2D})")
print(f"  Model2 Edges: {MODEL2_NUM_EDGES}")
print(f"  Node feature dim:  {NODE_FEAT_DIM}")
print(f"  Edge feature dim:  {EDGE_FEAT_DIM}")
print(f"  Warmup seq len: {SEQ_LENGTH}  |  Rollout steps: {ROLLOUT_STEPS}")


# ===========================================================================
# Instantiate models
# ===========================================================================

print("\n--- Instantiating models ---")

model1 = (
    SimpleFloodTGCN(
        num_nodes=TOTAL_MODEL1_NODES,
        in_features=NODE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        static_features=model1_static_node,
        adj=model1_adj,
        num_1d=MODEL1_NUM_1D,
    )
    .to(DEVICE)
    .eval()
)

model2 = (
    EdgeAwareFloodModel(
        num_nodes=TOTAL_MODEL2_NODES,
        num_edges=MODEL2_NUM_EDGES,
        node_in_dim=NODE_FEAT_DIM,
        edge_in_dim=EDGE_FEAT_DIM,
        node_hidden_dim=HIDDEN_DIM,
        edge_hidden_dim=EDGE_HIDDEN_DIM,
        dropout=DROPOUT,
        node_static=model2_static_node,
        edge_static=model2_static_edge,
        edge_index=model2_edge_index,
        num_1d=MODEL2_NUM_1D,
    )
    .to(DEVICE)
    .eval()
)

p1 = sum(p.numel() for p in model1.parameters())
p2 = sum(p.numel() for p in model2.parameters())
print(f"  Model 1 (SimpleFloodTGCN)    : {p1:,} parameters")
print(f"  Model 2 (EdgeAwareFloodModel): {p2:,} parameters")


# ===========================================================================
# Rollout helper shared by both models
# ===========================================================================


def rollout_model1(model, x_node_seq, rain_future, steps):
    """Warm-up then decode ROLLOUT_STEPS steps — Model 1 (no edge state)."""
    state = model.warmup(x_node_seq)
    current_depth = x_node_seq[-1, :, 0:1].clone()
    for s in range(steps):
        pred = model.decode(state, current_depth)
        current_depth = pred
        if s < steps - 1:
            # Build minimal rollout frame: depth + rain
            rain_t = rain_future[s, :, :]  # (N, 1)
            node_dyn = torch.cat(
                [
                    current_depth,
                    rain_t,
                    torch.zeros_like(current_depth),  # inlet_flow placeholder
                    torch.zeros_like(current_depth),  # water_volume placeholder
                    torch.zeros_like(current_depth),  # depth_slope placeholder
                    rain_t,
                ],
                dim=1,
            )  # rolling_rain placeholder
            state = model.step(state, node_dyn)
    return current_depth


def rollout_model2(model, x_node_seq, x_edge_seq, rain_future, steps):
    """Warm-up then decode ROLLOUT_STEPS steps — Model 2 (edge-aware)."""
    state = model.warmup(x_node_seq, x_edge_seq)
    current_depth = x_node_seq[-1, :, 0:1].clone()
    zero_edge_dyn = torch.zeros(model.num_edges, model.edge_dyn_dim, device=DEVICE)
    for s in range(steps):
        pred = model.decode(state, current_depth)
        current_depth = pred
        if s < steps - 1:
            rain_t = rain_future[s, :, :]
            node_dyn = torch.cat(
                [
                    current_depth,
                    rain_t,
                    torch.zeros_like(current_depth),
                    torch.zeros_like(current_depth),
                    torch.zeros_like(current_depth),
                    rain_t,
                ],
                dim=1,
            )
            state = model.step(state, node_dyn, zero_edge_dyn)
    return current_depth


# ===========================================================================
# Timing utility
# ===========================================================================


def cuda_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench(label, fn, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    # Warm-up (not timed)
    for _ in range(warmup):
        with torch.no_grad(), amp_ctx():
            fn()
        cuda_sync()

    # Timed runs
    times = []
    for _ in range(runs):
        cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad(), amp_ctx():
            fn()
        cuda_sync()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # → ms
    mean_ms = times.mean()
    std_ms = times.std()
    min_ms = times.min()
    max_ms = times.max()
    print(
        f"  {label:<45s}  mean={mean_ms:7.2f} ms  std={std_ms:6.2f} ms  "
        f"min={min_ms:7.2f} ms  max={max_ms:7.2f} ms"
    )
    return mean_ms, std_ms


# ===========================================================================
# Step-level timing: single decode + step  (no full rollout)
# ===========================================================================


def make_step_bench_m1(model):
    """Benchmark a single step of Model 1."""
    state_ref = [None]
    depth_ref = [None]

    def setup():
        with torch.no_grad(), amp_ctx():
            state_ref[0] = model.warmup(model1_x_node_seq)
            depth_ref[0] = model1_x_node_seq[-1, :, 0:1].clone()

    setup()
    rain_t = model1_rain_future[0]
    node_dyn = torch.cat(
        [
            depth_ref[0],
            rain_t,
            torch.zeros_like(depth_ref[0]),
            torch.zeros_like(depth_ref[0]),
            torch.zeros_like(depth_ref[0]),
            rain_t,
        ],
        dim=1,
    )

    def step_fn():
        _ = model.decode(state_ref[0], depth_ref[0])
        _ = model.step(state_ref[0], node_dyn)

    return step_fn


def make_step_bench_m2(model):
    """Benchmark a single step of Model 2."""
    state_ref = [None]
    depth_ref = [None]

    def setup():
        with torch.no_grad(), amp_ctx():
            state_ref[0] = model.warmup(model2_x_node_seq, model2_x_edge_seq)
            depth_ref[0] = model2_x_node_seq[-1, :, 0:1].clone()

    setup()
    rain_t = model2_rain_future[0]
    node_dyn = torch.cat(
        [
            depth_ref[0],
            rain_t,
            torch.zeros_like(depth_ref[0]),
            torch.zeros_like(depth_ref[0]),
            torch.zeros_like(depth_ref[0]),
            rain_t,
        ],
        dim=1,
    )
    zero_edge = torch.zeros(model.num_edges, model.edge_dyn_dim, device=DEVICE)

    def step_fn():
        _ = model.decode(state_ref[0], depth_ref[0])
        _ = model.step(state_ref[0], node_dyn, zero_edge)

    return step_fn


# ===========================================================================
# Run benchmarks
# ===========================================================================

print("\n" + "=" * 80)
print("BENCHMARK  —  Inference Time")
print("=" * 80)

print(
    f"\n[Config]  warmup_runs={WARMUP_RUNS}  bench_runs={BENCH_RUNS}  "
    f"rollout_steps={ROLLOUT_STEPS}  seq_len={SEQ_LENGTH}"
)

print("\n--- Full rollout (warm-up + all decode steps) ---")

m1_roll_mean, m1_roll_std = bench(
    f"Model 1 (TGCN)     full rollout  [{ROLLOUT_STEPS} steps]",
    lambda: rollout_model1(model1, model1_x_node_seq, model1_rain_future, ROLLOUT_STEPS),
)
m2_roll_mean, m2_roll_std = bench(
    f"Model 2 (EdgeAware) full rollout [{ROLLOUT_STEPS} steps]",
    lambda: rollout_model2(model2, model2_x_node_seq, model2_x_edge_seq, model2_rain_future, ROLLOUT_STEPS),
)

print("\n--- Single decode + step (per-timestep cost) ---")

step_fn_m1 = make_step_bench_m1(model1)
step_fn_m2 = make_step_bench_m2(model2)

m1_step_mean, _ = bench("Model 1 (TGCN)      single step", step_fn_m1)
m2_step_mean, _ = bench("Model 2 (EdgeAware)  single step", step_fn_m2)

# ---------------------------------------------------------------------------
# Per-node-type decomposition
# ---------------------------------------------------------------------------


def decode_1d_only(model, state, depth):
    """Return decoded delta for 1-D nodes only."""
    if hasattr(model, "head_1d"):
        if isinstance(state, list):  # Model 1: state is list of hidden
            h = state[-1]
        else:  # Model 2: state is dict
            h = state["node"]
        return model.head_1d(h[: model.num_1d])


def decode_2d_only(model, state, depth):
    """Return decoded delta for 2-D nodes only."""
    if isinstance(state, list):
        h = state[-1]
    else:
        h = state["node"]
    return model.head_2d(h[model.num_1d :])


# Pre-compute states once for node-type micro-benchmarks
with torch.no_grad(), amp_ctx():
    state_m1 = model1.warmup(model1_x_node_seq)
    depth_m1 = model1_x_node_seq[-1, :, 0:1].clone()
    state_m2 = model2.warmup(model2_x_node_seq, model2_x_edge_seq)
    depth_m2 = model2_x_node_seq[-1, :, 0:1].clone()

print("\n--- Decode head only (1-D vs 2-D node outputs) ---")

bench(
    "Model 1 — decode head 1-D nodes",
    lambda: decode_1d_only(model1, state_m1, depth_m1),
)
bench(
    "Model 1 — decode head 2-D nodes",
    lambda: decode_2d_only(model1, state_m1, depth_m1),
)
bench(
    "Model 2 — decode head 1-D nodes",
    lambda: decode_1d_only(model2, state_m2, depth_m2),
)
bench(
    "Model 2 — decode head 2-D nodes",
    lambda: decode_2d_only(model2, state_m2, depth_m2),
)

# ===========================================================================
# Summary table
# ===========================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
header = f"{'Metric':<50s} {'Model 1 (TGCN)':>20s} {'Model 2 (EdgeAware)':>22s}"
print(header)
print("-" * 95)

rows = [
    ("Full rollout  (ms)", f"{m1_roll_mean:>20.2f}", f"{m2_roll_mean:>22.2f}"),
    ("Per-step cost (ms)", f"{m1_step_mean:>20.2f}", f"{m2_step_mean:>22.2f}"),
    ("Rollout steps", f"{ROLLOUT_STEPS:>20d}", f"{ROLLOUT_STEPS:>22d}"),
    ("Model1 1-D nodes", f"{MODEL1_NUM_1D:>20d}", f"{MODEL1_NUM_1D:>22d}"),
    ("Model1 2-D nodes", f"{MODEL1_NUM_2D:>20d}", f"{MODEL1_NUM_2D:>22d}"),
    ("Model2 1-D nodes", f"{MODEL2_NUM_1D:>20d}", f"{MODEL2_NUM_1D:>22d}"),
    ("Model2 2-D nodes", f"{MODEL2_NUM_2D:>20d}", f"{MODEL2_NUM_2D:>22d}"),
    ("Parameters", f"{p1:>20,}", f"{p2:>22,}"),
]

for label, v1, v2 in rows:
    print(f"  {label:<48s}{v1}{v2}")

print("=" * 80)
print(f"\nDevice: {DEVICE}  |  Model 1 Nodes: {TOTAL_MODEL1_NODES}  |  Model 1 Edges: {MODEL1_NUM_EDGES}")
print(f"  Model 2 Nodes: {TOTAL_MODEL2_NODES}  |  Model 2 Edges: {MODEL2_NUM_EDGES}")
print(
    "All times are wall-clock with CUDA synchronisation (mean over "
    f"{BENCH_RUNS} runs after {WARMUP_RUNS} warm-up runs)."
)
