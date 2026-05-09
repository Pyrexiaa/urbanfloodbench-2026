"""
UFM Inference Benchmark — Model 1 & Model 2
=============================================
Measures inference time for:
  - Model 1  → 1D output (out_1d)
  - Model 1  → 2D output (out_2d)
  - Model 2  → 1D output (out_1d)
  - Model 2  → 2D output (out_2d)
"""

import os
# ---------------------------------------------------------------------------
# 1.  JAX / Keras setup
# ---------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"

import time
import numpy as np
import jax
import jax.numpy as jnp
import keras
import keras.layers as layers
import keras.ops as ops

print(f"keras  : {keras.__version__}")
print(f"jax    : {jax.__version__}")
print(f"devices: {jax.local_devices()}\n")

# ---------------------------------------------------------------------------
# 2.  Topology configs
# ---------------------------------------------------------------------------
MODEL_TOPOLOGIES = {
    "model_1": dict(
        num_1d_nodes=17, # model 1 1d nodes
        num_2d_nodes=3716, # model 1 2d nodes
        max_e_1d_from=16, # model 1 max edges from 1d node
        max_e_1d_to=16, # model 1 max edges to 1d node
        max_e_2d_from=7935, # model 1 max edges from 2d node
        max_e_2d_to=7935, # model 1 max edges to 2d node
        num_1d2d_conn=16, # model 1 number of 1d-2d connections
    ),
    "model_2": dict(
        num_1d_nodes=198,
        num_2d_nodes=4299,
        max_e_1d_from=197,
        max_e_1d_to=197,
        max_e_2d_from=9876,
        max_e_2d_to=9876,
        num_1d2d_conn=197,
    ),
}

# Shared hyper-parameters
BATCH_SIZE = 1  # single-sample inference
INPUT_WINDOW = 10  # history steps fed as context
FORECAST_STEPS = 1  # future steps to predict
D_MODEL = 10
N_WARMUP = 3  # JIT warm-up runs (not timed)
N_TIMED = 10  # timed runs

# ---------------------------------------------------------------------------
# 3.  Layer / model definitions  (identical to both notebooks)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="Hydrology")
class StaticEncoder(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.project_1d = layers.Dense(
            d_model, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.project_2d = layers.Dense(
            d_model, activation="swish", kernel_initializer="glorot_uniform"
        )

    def call(self, inputs, training=False):
        node_1d_s, node_2d_s, base_area_mask, connection_to_r_node_1d = inputs
        s_1d = ops.concatenate(
            [
                node_1d_s["depth"],
                node_1d_s["base_area"],
                node_1d_s["diameter_to_node"],
                node_1d_s["diameter_from_node"] * -1,
                node_1d_s["slope_to_node"],
                node_1d_s["slope_from_node"] * -1,
                node_1d_s["roughness_to_node"],
                node_1d_s["roughness_from_node"] * -1,
            ],
            -1,
        )
        s_2d = ops.concatenate(
            [
                node_2d_s["face_length_to_node"],
                node_2d_s["face_length_from_node"] * -1,
                node_2d_s["length_to_node"],
                node_2d_s["length_from_node"] * -1,
                node_2d_s["slope_to_node"],
                node_2d_s["slope_from_node"] * -1,
                node_2d_s["area"],
                node_2d_s["roughness"],
                node_2d_s["aspect"],
                node_2d_s["curvature"],
                node_2d_s["flow_accumulation"],
            ],
            -1,
        )
        return self.project_1d(s_1d), self.project_2d(s_2d)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(d_model=self.d_model)
        return cfg


@keras.saving.register_keras_serializable(package="Hydrology")
class CustomLSTMCell(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units * 4), initializer="glorot_uniform", name="W"
        )
        self.U = self.add_weight(
            shape=(self.units, self.units * 4), initializer="orthogonal", name="U"
        )
        self.b = self.add_weight(shape=(self.units * 4,), initializer="zeros", name="b")
        b_init = np.ones((self.units * 4,), dtype=np.float32)
        b_init[self.units : self.units * 2] = -0.01
        self.b.assign(b_init)
        self.built = True

    def call(self, x, h, c):
        gates = x @ self.W + h @ self.U + self.b
        u = self.units
        i = jax.nn.sigmoid(gates[:, 0 * u : 1 * u])
        f = jax.nn.sigmoid(gates[:, 1 * u : 2 * u])
        g = jnp.tanh(gates[:, 2 * u : 3 * u])
        o = jax.nn.sigmoid(gates[:, 3 * u : 4 * u])
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return h_new, c_new

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units)
        return cfg


@keras.saving.register_keras_serializable(package="Hydrology")
class CustomLSTM(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.cell = CustomLSTMCell(units)
        self.project = layers.Dense(units)
        self.ln = layers.LayerNormalization()

    def call(self, inputs, initial_state=None, training=False):
        x, neighbor_idx, coupling_idx = inputs
        b, t, _ = ops.shape(x)
        if initial_state is None:
            h = jnp.zeros((b, self.units), dtype=x.dtype)
            c = jnp.zeros((b, self.units), dtype=x.dtype)
        else:
            h, c = initial_state
        x_T = jnp.transpose(x, (1, 0, 2))

        def step(carry, x_t):
            h, c = carry
            h_new, c_new = self.cell(x_t, h, c)
            return (h_new, c_new), h_new

        (_, _), h_seq = jax.lax.scan(step, (h, c), x_T)
        return jnp.transpose(h_seq, (1, 0, 2))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units)
        return cfg


class NodeSpecificProjector(keras.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        num_nodes = input_shape[2]
        embed_dim = input_shape[3]

        def synced_init(shape, dtype=None):
            base_init = keras.initializers.VarianceScaling(
                scale=0.01, mode="fan_in", distribution="truncated_normal"
            )
            base_weights = base_init(shape=(embed_dim, self.units))
            return ops.broadcast_to(base_weights, (num_nodes, embed_dim, self.units))

        self.node_weights = self.add_weight(
            name="node_weights",
            shape=(num_nodes, embed_dim, self.units),
            initializer=synced_init,
            trainable=True,
        )
        self.node_bias = self.add_weight(
            name="node_bias",
            shape=(num_nodes, self.units),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        res = ops.einsum("blne,neo->blno", x, self.node_weights)
        return res + self.node_bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units)
        return cfg


@keras.saving.register_keras_serializable(package="Hydrology")
class FloodModel(keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.d_model = cfg["d_model"]
        self.static_encoder = StaticEncoder(self.d_model)
        self.lstm_layers_1d = [CustomLSTM(self.d_model) for _ in range(3)]
        self.lstm_layers_2d = [CustomLSTM(self.d_model) for _ in range(2)]
        self.project_1d = layers.Dense(self.d_model, activation="tanh")
        self.project_2d = layers.Dense(self.d_model, activation="tanh")
        self.ln_1d = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_2d = keras.layers.LayerNormalization(epsilon=1e-5)
        self.inp_expand_1d = NodeSpecificProjector(128)
        self.inp_expand_2d = NodeSpecificProjector(64)
        self.decoder_1d = NodeSpecificProjector(1)
        self.decoder_2d = NodeSpecificProjector(1)

    def call(self, inputs, training=False):
        node_1d_static = inputs["node_1d_static"]
        node_2d_static = inputs["node_2d_static"]
        rain_2d = inputs["rain_2d"]

        s_idx_from_1d = ops.cast(
            inputs["edge_connection"]["edge_from_node_1d"], "int32"
        )
        s_idx_to_1d = ops.cast(inputs["edge_connection"]["edge_to_node_1d"], "int32")
        s_idx_from_2d = ops.cast(
            inputs["edge_connection"]["edge_from_node_2d"], "int32"
        )
        s_idx_to_2d = ops.cast(inputs["edge_connection"]["edge_to_node_2d"], "int32")
        release_node_idx_1d = inputs["edge_connection"]["release_node_idx_1d"]
        c1d_idx = ops.cast(inputs["connection_1d2d"]["coupling_idx_1d"], "int32")
        c2d_idx = ops.cast(inputs["connection_1d2d"]["coupling_idx_2d"], "int32")

        base_area = node_1d_static["base_area"]
        base_area_mask = ops.cast(base_area != 0, "float32")
        connection_to_r_node_1d = ops.take(base_area_mask, s_idx_from_1d[0], axis=1)[
            ..., 0
        ]

        stat_1d, stat_2d = self.static_encoder(
            [node_1d_static, node_2d_static, base_area_mask, connection_to_r_node_1d],
            training=training,
        )

        b, n1d, d = ops.shape(stat_1d)
        b, t, n2d, _ = ops.shape(rain_2d[:, 10:])

        area = node_2d_static["area"]
        minimum_elevation = node_2d_static["min_elevation"]
        elevation = node_2d_static["elevation"]
        minimum_elevation = ops.where(
            ops.isfinite(minimum_elevation), minimum_elevation, elevation
        )

        flow_to_node_2d = inputs["node_2d"]["flow_to_node_2d"]
        vel_from_node_2d = inputs["node_2d"]["flow_from_node_2d"] * -1
        water_level = inputs["node_2d"]["water_level"] - minimum_elevation[:, None]
        zero_node_2d = ops.zeros((b, 10, 1, 1))
        water_level_pad = ops.concatenate([water_level, zero_node_2d], axis=2)

        wl_neighbors1_2d = ops.take(water_level_pad, s_idx_from_2d[0], axis=2)[..., 0]
        wl_neighbors2_2d = ops.take(water_level_pad, s_idx_to_2d[0], axis=2)[..., 0]
        delta_2d_full = water_level_pad[:, 1:] - water_level_pad[:, :-1]
        delta_2d_full = ops.concatenate(
            [ops.zeros_like(delta_2d_full[:, :1]), delta_2d_full], axis=1
        )
        delta_neighbors1_2d = ops.take(delta_2d_full, s_idx_from_2d[0], axis=2)[..., 0]
        delta_neighbors2_2d = ops.take(delta_2d_full, s_idx_to_2d[0], axis=2)[..., 0]
        delta_2d = delta_2d_full[:, :, :-1]
        water_level = water_level_pad[:, :, :-1]

        invert_elevation = node_1d_static["invert_elevation"]
        water_level_1d = inputs["node_1d"]["1d_water_level"]
        wl_from_invert = water_level_1d - invert_elevation[:, None]
        wl_from_surface = node_1d_static["surface_elevation"][:, None] - water_level_1d

        delta_1d = water_level_1d[:, 1:] - water_level_1d[:, :-1]
        delta_1d = ops.concatenate([ops.zeros_like(delta_1d[:, :1]), delta_1d], axis=1)
        zero_node_1d = ops.zeros((b, 10, 1, 1))
        delta_1d_pad = ops.concatenate([delta_1d, zero_node_1d], axis=2)
        delta_neighbors1_1d = ops.take(delta_1d_pad, s_idx_from_1d[0], axis=2)[..., 0]
        delta_neighbors2_1d = ops.take(delta_1d_pad, s_idx_to_1d[0], axis=2)[..., 0]

        wl_from_invert_pad = ops.concatenate([wl_from_invert, zero_node_1d], axis=2)
        wl_from_surface_pad = ops.concatenate([wl_from_surface, zero_node_1d], axis=2)
        wl_neighbors1_1d = ops.take(wl_from_invert_pad, s_idx_from_1d[0], axis=2)[
            ..., 0
        ]
        wl_neighbors2_1d = ops.take(wl_from_invert_pad, s_idx_to_1d[0], axis=2)[..., 0]

        flow_to_node = inputs["node_1d"]["flow_to_node_1d"]
        flow_from_node = inputs["node_1d"]["flow_from_node_1d"] * -1
        inlet_flow = inputs["node_1d"]["inlet_flow"]
        delta_1d = delta_1d_pad[:, :, :-1]
        wl_from_invert = wl_from_invert_pad[:, :, :-1]
        wl_from_surface = wl_from_surface_pad[:, :, :-1]

        x_1d = rain_2d[:, 10:, :n1d]
        x_2d = rain_2d[:, 10:, :n2d]

        x_1d = self.inp_expand_1d(x_1d)
        x_2d = self.inp_expand_2d(x_2d)

        x_1d = ops.reshape(ops.transpose(x_1d, (0, 2, 1, 3)), (b * n1d, t, -1))
        init_1d = ops.concatenate(
            [
                flow_to_node,
                flow_from_node,
                inlet_flow,
                wl_from_invert,
                wl_from_surface,
                wl_neighbors1_1d,
                wl_neighbors2_1d,
                delta_1d,
                delta_neighbors1_1d,
                delta_neighbors2_1d,
            ],
            -1,
        )[:, -1]
        init_1d = self.project_1d(init_1d)
        init_1d = ops.reshape(init_1d, (b * n1d, -1))
        c_1d = ops.reshape(stat_1d, (b * n1d, -1))
        initial_state_1d = (init_1d, c_1d)
        for layer in self.lstm_layers_1d:
            x_1d = layer(
                (x_1d, (s_idx_to_1d, s_idx_from_1d), c1d_idx),
                initial_state=initial_state_1d,
            )

        x_2d = ops.reshape(ops.transpose(x_2d, (0, 2, 1, 3)), (b * n2d, t, -1))
        init_2d = ops.concatenate(
            [
                flow_to_node_2d,
                flow_to_node_2d,
                water_level,
                wl_neighbors1_2d,
                wl_neighbors2_2d,
                delta_2d,
                delta_neighbors1_2d,
                delta_neighbors2_2d,
            ],
            -1,
        )[:, -1]
        init_2d = self.project_2d(ops.reshape(init_2d, (b * n2d, -1)))
        c_2d = ops.reshape(stat_2d, (b * n2d, -1))
        initial_state_2d = (init_2d, c_2d)
        for layer in self.lstm_layers_2d:
            x_2d = layer(
                (x_2d, (s_idx_to_2d, s_idx_from_2d), c2d_idx),
                initial_state=initial_state_2d,
            )

        x_1d = ops.transpose(ops.reshape(x_1d, (b, n1d, t, -1)), (0, 2, 1, 3))
        x_2d = ops.transpose(ops.reshape(x_2d, (b, n2d, t, -1)), (0, 2, 1, 3))

        base_area = ops.where(base_area == 0, 1, base_area)[:, None]
        area = ops.where(area <= 0, 1, area)[:, None]
        return {
            "out_1d": self.decoder_1d(self.ln_1d(x_1d)) / base_area,
            "out_2d": self.decoder_2d(self.ln_2d(x_2d)) / area,
        }

    def get_config(self):
        config = super().get_config()
        config.update({"cfg": self.cfg})
        return config


# ---------------------------------------------------------------------------
# 4.  Synthetic input generator
# ---------------------------------------------------------------------------


def make_random_inputs(topo, batch_size, input_window, forecast_steps):
    """Build a random input dict matching the FloodModel.call() signature."""
    B = batch_size
    IW = input_window
    FS = forecast_steps
    T = IW + FS  # total timesteps in rain_2d

    n1d = topo["num_1d_nodes"]
    n2d = topo["num_2d_nodes"]
    ef = topo["max_e_1d_from"]
    et = topo["max_e_1d_to"]
    e2f = topo["max_e_2d_from"]
    e2t = topo["max_e_2d_to"]

    def rnd(*shape):
        return np.random.randn(*shape).astype(np.float32)

    def idx_1d(rows, cols):
        return np.random.randint(0, n1d, size=(rows, cols)).astype(np.int32)

    def idx_2d(rows, cols):
        return np.random.randint(0, n2d, size=(rows, cols)).astype(np.int32)

    # --- static 1D node features ---
    node_1d_static = {
        "depth": rnd(B, n1d, 1),
        "base_area": np.abs(rnd(B, n1d, 1)) + 0.1,
        "invert_elevation": rnd(B, n1d, 1),
        "surface_elevation": rnd(B, n1d, 1),
        "diameter_to_node": rnd(B, n1d, ef),
        "diameter_from_node": rnd(B, n1d, et),
        "slope_to_node": rnd(B, n1d, ef),
        "slope_from_node": rnd(B, n1d, et),
        "roughness_to_node": rnd(B, n1d, ef),
        "roughness_from_node": rnd(B, n1d, et),
        "water_level_range": np.abs(rnd(B, n1d, 1)) + 0.1,
    }

    # --- static 2D node features ---
    node_2d_static = {
        "area": np.abs(rnd(B, n2d, 1)) + 0.1,
        "roughness": rnd(B, n2d, 1),
        "min_elevation": rnd(B, n2d, 1),
        "elevation": rnd(B, n2d, 1),
        "aspect": rnd(B, n2d, 1),
        "curvature": rnd(B, n2d, 1),
        "flow_accumulation": rnd(B, n2d, 1),
        "face_length_from_node": rnd(B, n2d, e2f),
        "face_length_to_node": rnd(B, n2d, e2t),
        "length_from_node": rnd(B, n2d, e2f),
        "length_to_node": rnd(B, n2d, e2t),
        "slope_from_node": rnd(B, n2d, e2f),
        "slope_to_node": rnd(B, n2d, e2t),
    }

    # --- dynamic 1D node features (input window only) ---
    node_1d = {
        "1d_water_level": rnd(B, IW, n1d, 1),
        "inlet_flow": rnd(B, IW, n1d, 1),
        "flow_from_node_1d": rnd(B, IW, n1d, ef),
        "velocity_from_node_1d": rnd(B, IW, n1d, ef),
        "flow_to_node_1d": rnd(B, IW, n1d, et),
        "velocity_to_node_1d": rnd(B, IW, n1d, et),
    }

    # --- dynamic 2D node features (input window only) ---
    node_2d = {
        "water_level": rnd(B, IW, n2d, 1),
        "water_volume": rnd(B, IW, n2d, 1),
        "flow_from_node_2d": rnd(B, IW, n2d, e2f),
        "velocity_from_node_2d": rnd(B, IW, n2d, e2f),
        "flow_to_node_2d": rnd(B, IW, n2d, e2t),
        "velocity_to_node_2d": rnd(B, IW, n2d, e2t),
    }

    # --- edge connectivity (random, clipped to valid node range) ---
    edge_connection = {
        "edge_from_node_1d": idx_1d(B, n1d * ef).reshape(B, n1d, ef),
        "edge_to_node_1d": idx_1d(B, n1d * et).reshape(B, n1d, et),
        "edge_from_node_2d": idx_2d(B, n2d * e2f).reshape(B, n2d, e2f),
        "edge_to_node_2d": idx_2d(B, n2d * e2t).reshape(B, n2d, e2t),
        "release_node_idx_1d": np.zeros((B, 1, 1), dtype=np.int32),
    }

    # --- 1D–2D coupling ---
    n_conn = topo["num_1d2d_conn"]
    coupling_1d = np.full((B, n1d, 1), -1, dtype=np.int32)
    coupling_2d = np.full((B, n2d, 1), -1, dtype=np.int32)
    idx_1 = np.random.choice(n1d, n_conn, replace=False)
    idx_2 = np.random.choice(n2d, n_conn, replace=False)
    coupling_1d[:, idx_1, 0] = idx_2
    coupling_2d[:, idx_2, 0] = idx_1

    connection_1d2d = {
        "coupling_idx_1d": coupling_1d,
        "coupling_idx_2d": coupling_2d,
    }

    # rain_2d shape: (B, T, max(n1d,n2d), 1)
    rain_2d = rnd(B, T, max(n1d, n2d), 1)

    return {
        "node_1d_static": node_1d_static,
        "node_2d_static": node_2d_static,
        "node_1d": node_1d,
        "node_2d": node_2d,
        "edge_connection": edge_connection,
        "connection_1d2d": connection_1d2d,
        "rain_2d": rain_2d,
    }


# ---------------------------------------------------------------------------
# 5.  Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(model_name, topo):
    print(f"\n{'=' * 60}")
    print(f"  Benchmarking {model_name.upper()}")
    print(f"  Nodes → 1D: {topo['num_1d_nodes']}  |  2D: {topo['num_2d_nodes']}")
    print(
        f"  Batch size: {BATCH_SIZE}  |  Input window: {INPUT_WINDOW}  |  Forecast: {FORECAST_STEPS}"
    )
    print(f"{'=' * 60}")

    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")  # stable for benchmarking

    model_cfg = {"d_model": D_MODEL}
    model = FloodModel(model_cfg, name=model_name)

    # Build the model with one forward pass (also JIT-compiles the JAX graph)
    print("  [1/3] Generating synthetic inputs …")
    inputs = make_random_inputs(topo, BATCH_SIZE, INPUT_WINDOW, FORECAST_STEPS)

    print("  [2/3] JIT warm-up …")
    for i in range(N_WARMUP):
        out = model(inputs, training=False)
        # Block until computation is done
        jax.block_until_ready(out["out_1d"])
        jax.block_until_ready(out["out_2d"])
        if i == 0:
            print(f"        out_1d shape: {np.array(out['out_1d']).shape}")
            print(f"        out_2d shape: {np.array(out['out_2d']).shape}")

    print(f"  [3/3] Timing {N_TIMED} inference runs …")
    times_total = []
    times_1d = []
    times_2d = []

    for _ in range(N_TIMED):
        # Full pass
        t0 = time.perf_counter()
        out = model(inputs, training=False)
        jax.block_until_ready(out["out_1d"])
        jax.block_until_ready(out["out_2d"])
        times_total.append(time.perf_counter() - t0)

        # Isolate 1D decode cost (approximate: just read out_1d)
        t0 = time.perf_counter()
        _ = np.array(out["out_1d"])
        times_1d.append(time.perf_counter() - t0)

        # Isolate 2D decode cost
        t0 = time.perf_counter()
        _ = np.array(out["out_2d"])
        times_2d.append(time.perf_counter() - t0)

    def stats(lst):
        arr = np.array(lst) * 1000  # convert to ms
        return arr.mean(), arr.std(), arr.min(), arr.max()

    m_total, s_total, mn_total, mx_total = stats(times_total)
    m_1d, s_1d, mn_1d, mx_1d = stats(times_1d)
    m_2d, s_2d, mn_2d, mx_2d = stats(times_2d)

    print(f"\n  ┌─ {model_name} inference timing (ms, n={N_TIMED}) ─────────────────┐")
    print(
        f"  │  Full forward pass  : {m_total:7.2f} ± {s_total:6.2f}  [min {mn_total:.2f}, max {mx_total:.2f}]  │"
    )
    print(
        f"  │  1D output read-out : {m_1d:7.2f} ± {s_1d:6.2f}  [min {mn_1d:.2f}, max {mx_1d:.2f}]  │"
    )
    print(
        f"  │  2D output read-out : {m_2d:7.2f} ± {s_2d:6.2f}  [min {mn_2d:.2f}, max {mx_2d:.2f}]  │"
    )
    print("  └──────────────────────────────────────────────────────────────────┘")

    return {
        "model": model_name,
        "full_ms_mean": m_total,
        "full_ms_std": s_total,
        "1d_readout_ms": m_1d,
        "2d_readout_ms": m_2d,
    }


# ---------------------------------------------------------------------------
# 6.  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    results = []

    for model_name, topo in MODEL_TOPOLOGIES.items():
        r = run_benchmark(model_name, topo)
        results.append(r)

    # ---- Summary table ----
    print("\n\n" + "=" * 70)
    print("  SUMMARY — Inference Time Comparison")
    print("=" * 70)
    print(
        f"  {'Model':<12} {'Full fwd (ms)':>16} {'1D readout (ms)':>18} {'2D readout (ms)':>18}"
    )
    print(f"  {'-' * 12} {'-' * 16} {'-' * 18} {'-' * 18}")
    for r in results:
        print(
            f"  {r['model']:<12} "
            f"{r['full_ms_mean']:>12.2f} ms  "
            f"{r['1d_readout_ms']:>14.3f} ms  "
            f"{r['2d_readout_ms']:>14.3f} ms"
        )
    print("=" * 70)
    print("\nNote: '1D readout' and '2D readout' are the host-side numpy copy")
    print("      costs for each output tensor, measured after the forward pass.")
    print("      They do NOT include the LSTM computation itself.\n")
