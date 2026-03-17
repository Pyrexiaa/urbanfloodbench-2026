"""Urban Flood Bench training and inference."""

import os
import gc
import re
import time
import random
from contextlib import nullcontext

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


MODEL_CONFIG = {
    "name": "feature_edge_aware_full_prefix",
    "model_kind": "edge",
    "use_inlet_flow": True,
    "use_water_volume": True,
    "use_dynamic_edges": True,
    "use_static_edges": True,
    "use_engineered_node_features": True,
    "use_engineered_edge_features": True,
    "use_feature_gates": False,
}


def parse_int_csv(value):
    return [int(x) for x in value.split(",") if x.strip()]


def resolve_ids(list_arg, single_arg, label):
    if list_arg:
        values = parse_int_csv(list_arg)
    elif single_arg is not None:
        values = [single_arg]
    else:
        raise ValueError(f"Missing --{label}. Provide --{label}s or --{label}.")
    if not values:
        raise ValueError(f"No values parsed for --{label}.")
    return values

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IS_COLAB = "google.colab" in str(get_ipython()) if "get_ipython" in dir() else False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if IS_KAGGLE:
    DATA_PATH, SAVE_PATH = "/kaggle/input/urban-flood-prediction", "/kaggle/working"
elif IS_COLAB:
    DATA_PATH = "/content/drive/MyDrive/Urban Flood Bench/datasets"
    SAVE_PATH = "/content/drive/MyDrive/Urban Flood Bench/results"
else:
    DATA_PATH, SAVE_PATH = "datasets", "results"
DEFAULT_SAMPLE_SUBMISSION_PATH = os.path.join(SAVE_PATH, "sample_submission.csv")
args = None
SAMPLE_SUBMISSION_PATH = None
OUTPUT_PATH = None
MODEL_IDS = []
ENSEMBLE_SEEDS = []

EPOCHS = 300
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 45
SEQ_LENGTH = 10
SPIN_UP_LEN = 10
HIDDEN_DIM = 512
EDGE_HIDDEN_DIM = 96
NUM_LAYERS = 2
DROPOUT = 0.2
SAMPLES_PER_EVENT = 10
VAL_SAMPLES_CAP = 30
INITIAL_TRAIN_STEPS = 10

# Per-model training horizon: M2 needs longer rollouts (test goes up to 409 steps)
TRAIN_STEPS_CONFIG = {1: 90, 2: 180}
PRED_HORIZON_CONFIG = {1: 90, 2: 180}
LR_WARMUP_EPOCHS = 5
VAL_EVERY_N_EPOCHS = 1
TF_SIGMOID_K = 15
TF_MIN_RATIO = 0.05
MIN_LR = 1e-6
SCHEDULER_PATIENCE = 7
RAIN_ROLL_WINDOW = 3
EMA_DECAY = 0.999
GRAD_ACCUM_STEPS = 4
TIME_WEIGHT_STRENGTH = 0.5

# Per-model, per-node-type noise (~5% of delta std from data analysis)
NOISE_CONFIG = {
    1: {"input_1d": 0.002, "input_2d": 0.001, "rollout_1d": 0.001, "rollout_2d": 0.0005},
    2: {"input_1d": 0.0014, "input_2d": 0.0004, "rollout_1d": 0.0008, "rollout_2d": 0.0002},
}

SCALE_CONFIG = {
    1: {"scale_1d": 2.0, "scale_2d": 1.0, "rainfall_scale": 0.05, "inlet_flow_scale": 5.0},
    2: {"scale_1d": 3.5, "scale_2d": 2.0, "rainfall_scale": 0.05, "inlet_flow_scale": 10.0},
}

STD_DEVS = {
    (1, 1): 16.877747,
    (1, 2): 14.378797,
    (2, 1): 3.191784,
    (2, 2): 2.727131,
}


def amp_context():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else nullcontext()


def signed_log1p(arr):
    return np.sign(arr) * np.log1p(np.abs(arr))


def model_ckpt_tag():
    tag = f"n{HIDDEN_DIM}_eh{EDGE_HIDDEN_DIM}_v2"
    if MODEL_CONFIG.get("use_feature_gates", False):
        tag += "_fg"
    return tag


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    parser.add_argument("--model_id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_ids", type=str)
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--sample_submission_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--allow_sequential_row_ids", action="store_true")
    return parser


def initialize_runtime(parsed_args):
    global args, SAMPLE_SUBMISSION_PATH, OUTPUT_PATH, MODEL_IDS, ENSEMBLE_SEEDS

    args = parsed_args
    MODEL_IDS = resolve_ids(args.model_ids, args.model_id, "model_id")
    ENSEMBLE_SEEDS = resolve_ids(args.seeds, args.seed, "seed")
    SAMPLE_SUBMISSION_PATH = args.sample_submission_path or DEFAULT_SAMPLE_SUBMISSION_PATH
    OUTPUT_PATH = args.output_path or os.path.join(SAVE_PATH, "submission.csv")
    os.makedirs(SAVE_PATH, exist_ok=True)


# ============================================================================
# 2. DATA LOADER
# ============================================================================


class FloodDataLoader:
    def __init__(self, base_path, model_id):
        self.base_path = base_path
        self.model_id = model_id
        self.static_data = {}
        self.stats = {}

    def load_static(self):
        files = {
            "1d_nodes_static": "1d_nodes_static.csv",
            "2d_nodes_static": "2d_nodes_static.csv",
            "1d_edges_static": "1d_edges_static.csv",
            "2d_edges_static": "2d_edges_static.csv",
            "1d_edge_index": "1d_edge_index.csv",
            "2d_edge_index": "2d_edge_index.csv",
            "1d2d_connections": "1d2d_connections.csv",
        }
        for key, fname in files.items():
            path = os.path.join(self.base_path, fname)
            if os.path.exists(path):
                self.static_data[key] = pd.read_csv(path)

        n1d = self.static_data["1d_nodes_static"]
        n2d = self.static_data["2d_nodes_static"]
        all_x = np.concatenate([n1d["position_x"], n2d["position_x"]])
        all_y = np.concatenate([n1d["position_y"], n2d["position_y"]])
        all_e = np.concatenate([n1d["invert_elevation"], n2d["min_elevation"]])
        self.stats = {
            "x_mean": all_x.mean(),
            "x_std": all_x.std() + 1e-8,
            "y_mean": all_y.mean(),
            "y_std": all_y.std() + 1e-8,
            "elev_mean": all_e.mean(),
            "elev_std": all_e.std() + 1e-8,
        }
        return self.static_data

    def scan_events(self):
        folders = [
            f
            for f in os.listdir(self.base_path)
            if f.startswith("event_") and os.path.isdir(os.path.join(self.base_path, f))
        ]

        def _num(name):
            m = re.search(r"\d+", name)
            return int(m.group()) if m else -1

        folders.sort(key=_num)
        return [{"event_id": _num(f), "path": os.path.join(self.base_path, f)} for f in folders if _num(f) >= 0]

    def load_event(self, meta):
        data = {}
        files = [
            ("1d_nodes", "1d_nodes_dynamic_all.csv"),
            ("2d_nodes", "2d_nodes_dynamic_all.csv"),
            ("1d_edges", "1d_edges_dynamic_all.csv"),
            ("2d_edges", "2d_edges_dynamic_all.csv"),
        ]
        for key, fname in files:
            path = os.path.join(meta["path"], fname)
            if os.path.exists(path):
                data[key] = pd.read_csv(path)
            else:
                return None
        return data

class GraphBuilder:
    def __init__(self, loader):
        self.loader = loader
        sd = loader.static_data
        self.num_1d = len(sd["1d_nodes_static"])
        self.num_2d = len(sd["2d_nodes_static"])
        self.total_nodes = self.num_1d + self.num_2d
        self.edge_index = None
        self.num_edges = 0
        self.num_1d_edges = len(sd["1d_edges_static"])
        self.num_2d_edges = len(sd["2d_edges_static"])
        self.num_conn_edges = len(sd["1d2d_connections"])

    def build(self):
        sd = self.loader.static_data
        ei1 = sd["1d_edge_index"].sort_values("edge_idx")
        s1 = ei1["from_node"].to_numpy()
        d1 = ei1["to_node"].to_numpy()

        ei2 = sd["2d_edge_index"].sort_values("edge_idx")
        s2 = ei2["from_node"].to_numpy() + self.num_1d
        d2 = ei2["to_node"].to_numpy() + self.num_1d

        ec = sd["1d2d_connections"].sort_values("connection_idx")
        sc = ec["node_1d"].to_numpy()
        dc = ec["node_2d"].to_numpy() + self.num_1d

        all_src = np.concatenate([s1, d1, s2, d2, sc, dc])
        all_dst = np.concatenate([d1, s1, d2, s2, dc, sc])
        self.edge_index = torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long)
        self.num_edges = self.edge_index.shape[1]

        print(f"Graph: {self.total_nodes} nodes, {self.num_edges} directed edges")
        return self.edge_index, None

    def get_static_features(self):
        sd = self.loader.static_data
        stats = self.loader.stats

        n1d = sd["1d_nodes_static"]
        f1 = np.column_stack(
            [
                (n1d["invert_elevation"] - stats["elev_mean"]) / stats["elev_std"],
                (n1d["surface_elevation"] - stats["elev_mean"]) / stats["elev_std"],
                n1d["depth"] / 20.0,
                np.log1p(n1d["base_area"]) / 5.0,
                (n1d["position_x"] - stats["x_mean"]) / stats["x_std"],
                (n1d["position_y"] - stats["y_mean"]) / stats["y_std"],
            ]
        ).astype(np.float32)

        n2d = sd["2d_nodes_static"]
        f2 = np.column_stack(
            [
                (n2d["min_elevation"] - stats["elev_mean"]) / stats["elev_std"],
                (n2d["elevation"] - stats["elev_mean"]) / stats["elev_std"],
                np.log1p(n2d["area"]) / 10.0,
                n2d["roughness"] / 0.1,
                np.log1p(n2d["flow_accumulation"]) / 10.0,
                (n2d["position_x"] - stats["x_mean"]) / stats["x_std"],
                (n2d["position_y"] - stats["y_mean"]) / stats["y_std"],
            ]
        ).astype(np.float32)

        max_dim = max(f1.shape[1], f2.shape[1])
        if f1.shape[1] < max_dim:
            f1 = np.hstack([f1, np.zeros((len(f1), max_dim - f1.shape[1]), dtype=np.float32)])
        if f2.shape[1] < max_dim:
            f2 = np.hstack([f2, np.zeros((len(f2), max_dim - f2.shape[1]), dtype=np.float32)])

        f1 = np.hstack([f1, np.zeros((len(f1), 1), dtype=np.float32)])
        f2 = np.hstack([f2, np.ones((len(f2), 1), dtype=np.float32)])
        out = np.vstack([f1, f2]).astype(np.float32)
        return torch.tensor(np.nan_to_num(out, nan=0.0), dtype=torch.float32)

    def get_edge_static_features(self):
        sd = self.loader.static_data
        stats = self.loader.stats
        e1 = sd["1d_edges_static"].sort_values("edge_idx")
        e2 = sd["2d_edges_static"].sort_values("edge_idx")

        def _pack_1d(df, reverse=False):
            rel_x = df["relative_position_x"].to_numpy(dtype=np.float32)
            rel_y = df["relative_position_y"].to_numpy(dtype=np.float32)
            if reverse:
                rel_x = -rel_x
                rel_y = -rel_y
            feats = np.column_stack(
                [
                    signed_log1p(rel_x),
                    signed_log1p(rel_y),
                    np.log1p(df["length"].to_numpy(dtype=np.float32)),
                    df["diameter"].to_numpy(dtype=np.float32) / 5.0,
                    df["roughness"].to_numpy(dtype=np.float32) / 0.1,
                    df["slope"].to_numpy(dtype=np.float32),
                    np.ones(len(df), dtype=np.float32),
                    np.zeros(len(df), dtype=np.float32),
                    np.zeros(len(df), dtype=np.float32),
                ]
            )
            return feats.astype(np.float32)

        def _pack_2d(df, reverse=False):
            rel_x = df["relative_position_x"].to_numpy(dtype=np.float32)
            rel_y = df["relative_position_y"].to_numpy(dtype=np.float32)
            if reverse:
                rel_x = -rel_x
                rel_y = -rel_y
            feats = np.column_stack(
                [
                    signed_log1p(rel_x),
                    signed_log1p(rel_y),
                    np.log1p(df["length"].to_numpy(dtype=np.float32)),
                    np.zeros(len(df), dtype=np.float32),
                    np.log1p(df["face_length"].to_numpy(dtype=np.float32)),
                    df["slope"].to_numpy(dtype=np.float32),
                    np.zeros(len(df), dtype=np.float32),
                    np.ones(len(df), dtype=np.float32),
                    np.zeros(len(df), dtype=np.float32),
                ]
            )
            return feats.astype(np.float32)

        conn_count = self.num_conn_edges
        conn_feats = np.column_stack(
            [
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.zeros(conn_count, dtype=np.float32),
                np.ones(conn_count, dtype=np.float32),
            ]
        ).astype(np.float32)

        feats = np.vstack(
            [
                _pack_1d(e1, reverse=False),
                _pack_1d(e1, reverse=True),
                _pack_2d(e2, reverse=False),
                _pack_2d(e2, reverse=True),
                conn_feats,
                conn_feats.copy(),
            ]
        )
        return torch.tensor(feats, dtype=torch.float32)

class Preprocessor:
    def __init__(self, loader, graph, cfg, model_config):
        self.loader = loader
        self.graph = graph
        self.num_1d = graph.num_1d
        self.num_2d = graph.num_2d
        self.scale_1d = cfg["scale_1d"]
        self.scale_2d = cfg["scale_2d"]
        self.rainfall_scale = cfg["rainfall_scale"]
        self.inlet_flow_scale = cfg["inlet_flow_scale"]
        self.model_config = model_config
        self.use_inlet_flow = model_config["use_inlet_flow"]
        self.use_water_volume = model_config["use_water_volume"]
        self.use_edges = model_config["use_dynamic_edges"]
        self.use_engineered_node_features = model_config["use_engineered_node_features"]
        self.use_engineered_edge_features = model_config["use_engineered_edge_features"]
        self._invert_elev = loader.static_data["1d_nodes_static"]["invert_elevation"].to_numpy(dtype=np.float32)
        self._min_elev = loader.static_data["2d_nodes_static"]["min_elevation"].to_numpy(dtype=np.float32)
        self.base_node_feature_dim = 2 + int(self.use_inlet_flow) + int(self.use_water_volume)
        self.engineered_node_feature_dim = 2 * int(self.use_engineered_node_features)
        self.node_feature_dim = self.base_node_feature_dim + self.engineered_node_feature_dim
        self.edge_feature_dim = (3 + int(self.use_engineered_edge_features)) if self.use_edges else 0
        self.rain_idx = 1

    @staticmethod
    def _rolling_sum(values, window):
        if window <= 1:
            return values.astype(np.float32)
        csum = np.cumsum(values, axis=0)
        out = csum.copy()
        out[window:] = csum[window:] - csum[:-window]
        return out.astype(np.float32)

    def _node_features(self, event_data):
        df1 = event_data["1d_nodes"].sort_values(["timestep", "node_idx"])
        df2 = event_data["2d_nodes"].sort_values(["timestep", "node_idx"])

        wl1 = df1.pivot(index="timestep", columns="node_idx", values="water_level").to_numpy(dtype=np.float32)
        wl2 = df2.pivot(index="timestep", columns="node_idx", values="water_level").to_numpy(dtype=np.float32)
        wl1, wl2 = np.nan_to_num(wl1), np.nan_to_num(wl2)
        raw_wl = np.concatenate([wl1, wl2], axis=1)

        d1 = np.clip((wl1 - self._invert_elev) / self.scale_1d, -1, 5)
        d2 = np.clip((wl2 - self._min_elev) / self.scale_2d, -1, 5)
        depth = np.concatenate([d1, d2], axis=1).astype(np.float32)

        rain = df2.pivot(index="timestep", columns="node_idx", values="rainfall").to_numpy(dtype=np.float32)
        rain = np.clip(np.nan_to_num(rain) / self.rainfall_scale, 0, 10)
        rain = np.concatenate([np.zeros((depth.shape[0], self.num_1d), dtype=np.float32), rain], axis=1)

        features = [depth, rain]

        if self.use_inlet_flow:
            flow = df1.pivot(index="timestep", columns="node_idx", values="inlet_flow").to_numpy(dtype=np.float32)
            flow = np.clip(np.nan_to_num(flow) / self.inlet_flow_scale, -2, 5)
            flow = np.concatenate([flow, np.zeros((depth.shape[0], self.num_2d), dtype=np.float32)], axis=1)
            features.append(flow)

        if self.use_water_volume:
            vol = df2.pivot(index="timestep", columns="node_idx", values="water_volume").to_numpy(dtype=np.float32)
            vol = signed_log1p(np.nan_to_num(vol))
            vol = np.concatenate([np.zeros((depth.shape[0], self.num_1d), dtype=np.float32), vol], axis=1)
            features.append(vol)

        if self.use_engineered_node_features:
            depth_slope = np.zeros_like(depth, dtype=np.float32)
            depth_slope[1:] = depth[1:] - depth[:-1]
            rain_roll_2d = self._rolling_sum(rain[:, self.num_1d :], RAIN_ROLL_WINDOW)
            rolling_rain = np.concatenate(
                [np.zeros((depth.shape[0], self.num_1d), dtype=np.float32), rain_roll_2d],
                axis=1,
            )
            features.extend([depth_slope, rolling_rain])

        x = np.stack(features, axis=2).astype(np.float32)
        y = depth[:, :, np.newaxis].astype(np.float32)
        ids_1d = df1["node_idx"].unique()
        ids_2d = df2["node_idx"].unique()
        return x, y, raw_wl, ids_1d, ids_2d

    def _edge_features(self, event_data):
        if not self.use_edges:
            return None

        e1 = event_data["1d_edges"].sort_values(["timestep", "edge_idx"])
        e2 = event_data["2d_edges"].sort_values(["timestep", "edge_idx"])

        flow1 = e1.pivot(index="timestep", columns="edge_idx", values="flow").to_numpy(dtype=np.float32)
        flow2 = e2.pivot(index="timestep", columns="edge_idx", values="flow").to_numpy(dtype=np.float32)
        if "velocity" in e1.columns:
            vel1 = e1.pivot(index="timestep", columns="edge_idx", values="velocity").to_numpy(dtype=np.float32)
        else:
            vel1 = np.zeros_like(flow1, dtype=np.float32)
        if "velocity" in e2.columns:
            vel2 = e2.pivot(index="timestep", columns="edge_idx", values="velocity").to_numpy(dtype=np.float32)
        else:
            vel2 = np.zeros_like(flow2, dtype=np.float32)

        # Compute observation masks BEFORE nan_to_num so they reflect true data availability
        obs1 = np.isfinite(flow1).astype(np.float32)
        obs2 = np.isfinite(flow2).astype(np.float32)
        raw_flow1 = np.nan_to_num(flow1)
        raw_flow2 = np.nan_to_num(flow2)
        flow1 = signed_log1p(raw_flow1)
        vel1 = signed_log1p(np.nan_to_num(vel1))
        flow2 = signed_log1p(raw_flow2)
        vel2 = signed_log1p(np.nan_to_num(vel2))

        conn_zeros = np.zeros((flow1.shape[0], self.graph.num_conn_edges), dtype=np.float32)
        channels = [
            np.concatenate([flow1, -flow1, flow2, -flow2, conn_zeros, conn_zeros], axis=1),
            np.concatenate([vel1, -vel1, vel2, -vel2, conn_zeros, conn_zeros], axis=1),
            np.concatenate([obs1, obs1, obs2, obs2, conn_zeros, conn_zeros], axis=1),
        ]
        if self.use_engineered_edge_features:
            delta1 = np.zeros_like(raw_flow1, dtype=np.float32)
            delta2 = np.zeros_like(raw_flow2, dtype=np.float32)
            delta1[1:] = raw_flow1[1:] - raw_flow1[:-1]
            delta2[1:] = raw_flow2[1:] - raw_flow2[:-1]
            delta1 = signed_log1p(delta1)
            delta2 = signed_log1p(delta2)
            channels.append(np.concatenate([delta1, -delta1, delta2, -delta2, conn_zeros, conn_zeros], axis=1))
        edge_dyn = np.stack(channels, axis=2).astype(np.float32)
        return edge_dyn

    def process_event(self, event_data):
        node_x, node_y, _, _, _ = self._node_features(event_data)
        edge_x = self._edge_features(event_data)
        return node_x, node_y, edge_x

    def process_for_inference(self, event_data):
        node_x, _, raw_wl, ids_1d, ids_2d = self._node_features(event_data)
        edge_x = self._edge_features(event_data)
        rain = node_x[:, :, self.rain_idx].astype(np.float32)
        return node_x, edge_x, raw_wl, rain, ids_1d, ids_2d

    def prepare_warmup_sequence(self, node_seq):
        return node_seq

    def init_rollout_context(self, warmup_node_seq):
        if not self.use_engineered_node_features:
            return None
        if RAIN_ROLL_WINDOW > 1:
            hist = warmup_node_seq[-(RAIN_ROLL_WINDOW - 1) :, :, self.rain_idx : self.rain_idx + 1].clone()
        else:
            hist = warmup_node_seq.new_zeros((0, warmup_node_seq.shape[1], 1))
        return {
            "prev_depth": warmup_node_seq[-1, :, 0:1].clone(),
            "rain_history": hist,
        }

    def build_rollout_node_frame(self, current_depth, rain_t, rollout_ctx=None):
        parts = [current_depth, rain_t]
        if self.use_inlet_flow:
            parts.append(torch.zeros_like(current_depth))
        if self.use_water_volume:
            parts.append(torch.zeros_like(current_depth))
        if self.use_engineered_node_features:
            assert rollout_ctx is not None
            depth_slope = current_depth - rollout_ctx["prev_depth"]
            if RAIN_ROLL_WINDOW > 1 and rollout_ctx["rain_history"].shape[0] > 0:
                rolling_rain = rollout_ctx["rain_history"].sum(dim=0) + rain_t
                next_hist = torch.cat([rollout_ctx["rain_history"], rain_t.unsqueeze(0)], dim=0)[-(RAIN_ROLL_WINDOW - 1) :]
            else:
                rolling_rain = rain_t
                next_hist = rain_t.unsqueeze(0)[0:0]
            parts.extend([depth_slope, rolling_rain])
            rollout_ctx["prev_depth"] = current_depth
            rollout_ctx["rain_history"] = next_hist
        return torch.cat(parts, dim=1), rollout_ctx

def build_gcn_csr(edge_index, num_nodes, device, dtype=torch.float16):
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
        indices=torch.stack([row, col], dim=0),
        values=w,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()
    return A_coo.to_sparse_csr()


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
        raw_base_feature_dim,
        engineered_feature_dim=0,
        use_feature_gates=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.raw_base_feature_dim = raw_base_feature_dim
        self.engineered_feature_dim = engineered_feature_dim
        self.use_feature_gates = use_feature_gates and engineered_feature_dim > 0
        self.register_buffer("static_feat", static_features)
        self.register_buffer("adj", adj)
        self.num_1d = int((static_features[:, -1] < 0.5).sum().item())
        self.num_2d = num_nodes - self.num_1d

        if self.use_feature_gates:
            self.base_fuse = nn.Sequential(
                nn.Linear(raw_base_feature_dim + static_features.shape[1], hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            self.eng_fuse = nn.Sequential(
                nn.Linear(engineered_feature_dim, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
            self.feature_gate_logits = nn.Parameter(torch.full((engineered_feature_dim,), 2.0))
        else:
            self.fuse = nn.Sequential(
                nn.Linear(in_features + static_features.shape[1], hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
            )
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(SimpleTGCNCell(hidden_dim // 2 if i == 0 else hidden_dim, hidden_dim, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # No dropout in decode heads — prevents compounding noise during rollout
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

    def _decode_delta(self, h_last):
        d1 = self.head_1d(h_last[: self.num_1d])
        d2 = self.head_2d(h_last[self.num_1d :])
        return torch.cat([d1, d2], dim=0)

    def _fuse_input(self, node_dyn):
        if not self.use_feature_gates:
            return self.fuse(torch.cat([node_dyn, self.static_feat], dim=1))
        base_dyn = node_dyn[:, : self.raw_base_feature_dim]
        eng_dyn = node_dyn[:, self.raw_base_feature_dim :]
        gates = torch.sigmoid(self.feature_gate_logits).to(node_dyn.dtype).view(1, -1)
        base_lat = self.base_fuse(torch.cat([base_dyn, self.static_feat], dim=1))
        eng_lat = self.eng_fuse(eng_dyn * gates)
        return self.fuse(torch.cat([base_lat, eng_lat], dim=1))

    def warmup(self, node_seq, edge_seq=None):
        T, N, _ = node_seq.shape
        h = [torch.zeros(N, self.hidden_dim, device=node_seq.device) for _ in range(self.num_layers)]
        for t in range(T):
            x_t = self._fuse_input(node_seq[t])
            for i in range(self.num_layers):
                h_new = self.norms[i](self.cells[i](x_t, self.adj, h[i]))
                x_t = h_new if i == 0 else h_new + x_t
                h[i] = h_new
        return h

    def decode(self, state, current_depth):
        return current_depth + self.res_scale * self._decode_delta(state[-1])

    def step(self, state, node_dyn, edge_dyn=None):
        frame = self._fuse_input(node_dyn)
        new_state = []
        x_t = frame
        for i in range(self.num_layers):
            h_new = self.norms[i](self.cells[i](x_t, self.adj, state[i]))
            x_t = h_new if i == 0 else h_new + x_t
            new_state.append(h_new)
        return new_state


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
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_1d = int((node_static[:, -1] < 0.5).sum().item())
        self.num_2d = num_nodes - self.num_1d
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
        # Hop 1: message passing
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
        # Hop 2: second message pass on updated node representations (doubles receptive field)
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
        # No dropout in decode heads — prevents compounding noise during rollout
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
        self.edge_dyn_dim = self.edge_encoder[0].in_features - self.edge_static.shape[1]

    def _zero_state(self, device):
        return {
            "node": torch.zeros(self.num_nodes, self.node_hidden_dim, device=device),
            "edge": torch.zeros(self.num_edges, self.edge_hidden_dim, device=device),
        }

    def _missing_edge_dyn(self, device, dtype):
        edge_dyn = torch.zeros(self.num_edges, self.edge_dyn_dim, device=device, dtype=dtype)
        if self.edge_dyn_dim >= 3:
            edge_dyn[:, 2] = 0.0
        return edge_dyn

    def _step_core(self, state, node_dyn, edge_dyn):
        node_lat = self.node_encoder(torch.cat([node_dyn, self.node_static], dim=1))
        edge_lat = self.edge_encoder(torch.cat([edge_dyn, self.edge_static], dim=1))
        node_base = node_lat + state["node"]
        edge_base = edge_lat + state["edge"]

        src, dst = self.edge_index
        in_deg = torch.bincount(dst, minlength=self.num_nodes).float().clamp(min=1).unsqueeze(1).to(node_base.device)

        # --- Hop 1 ---
        msg = self.message_mlp(torch.cat([node_base[src], node_base[dst], edge_base], dim=1))
        msg = msg.to(edge_base.dtype)
        agg = torch.zeros(self.num_nodes, self.edge_hidden_dim, device=node_base.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        agg = agg / in_deg
        node_after_hop1 = node_base + self.msg_to_node(agg)  # residual add

        # --- Hop 2 (uses updated node reps -> doubles receptive field) ---
        msg2 = self.message_mlp_2(torch.cat([node_after_hop1[src], node_after_hop1[dst], edge_base], dim=1))
        msg2 = msg2.to(edge_base.dtype)
        agg2 = torch.zeros(self.num_nodes, self.edge_hidden_dim, device=node_base.device, dtype=msg2.dtype)
        agg2.index_add_(0, dst, msg2)
        agg2 = agg2 / in_deg

        node_in = self.node_update(torch.cat([node_after_hop1, self.msg_to_node_2(agg2)], dim=1))
        # Edge update uses hop-1 messages (hop-2 messages already folded into node states)
        edge_in = self.edge_update(torch.cat([edge_base, msg], dim=1))

        node_state = state["node"]
        next_node = torch.empty_like(node_state)
        next_node[: self.num_1d] = self.node_gru_1d(node_in[: self.num_1d], node_state[: self.num_1d])
        next_node[self.num_1d :] = self.node_gru_2d(node_in[self.num_1d :], node_state[self.num_1d :])
        next_node = self.node_norm(next_node)
        next_edge = self.edge_norm(self.edge_gru(edge_in, state["edge"]))
        return {"node": next_node, "edge": next_edge}

    def warmup(self, node_seq, edge_seq):
        state = self._zero_state(node_seq.device)
        if edge_seq is None:
            edge_seq = torch.zeros(
                node_seq.shape[0],
                self.num_edges,
                self.edge_dyn_dim,
                device=node_seq.device,
                dtype=node_seq.dtype,
            )
        for t in range(node_seq.shape[0]):
            state = self._step_core(state, node_seq[t], edge_seq[t])
        return state

    def _decode_delta(self, node_state):
        d1 = self.head_1d(node_state[: self.num_1d])
        d2 = self.head_2d(node_state[self.num_1d :])
        return torch.cat([d1, d2], dim=0)

    def decode(self, state, current_depth):
        return current_depth + self.res_scale * self._decode_delta(state["node"])

    def step(self, state, node_dyn, edge_dyn=None):
        if edge_dyn is None:
            edge_dyn = self._missing_edge_dyn(node_dyn.device, node_dyn.dtype)
        return self._step_core(state, node_dyn, edge_dyn)

class FloodLoss(nn.Module):
    def __init__(self, num_1d, model_id, scale_1d, scale_2d):
        super().__init__()
        std_1d = STD_DEVS[(model_id, 1)]
        std_2d = STD_DEVS[(model_id, 2)]
        self.w1d = (scale_1d / std_1d) ** 2
        self.w2d = (scale_2d / std_2d) ** 2
        self.num_1d = num_1d

    def forward(self, pred, target):
        err = (pred - target) ** 2
        if pred.dim() == 3:
            mse_1d = err[:, : self.num_1d].mean()
            mse_2d = err[:, self.num_1d :].mean()
        else:
            mse_1d = err[: self.num_1d].mean()
            mse_2d = err[self.num_1d :].mean()
        return 0.5 * torch.sqrt(self.w1d * mse_1d + 1e-10) + 0.5 * torch.sqrt(self.w2d * mse_2d + 1e-10)


def kaggle_like_window_loss_from_sumerr2(sum_err2, steps, num_1d, w1d, w2d, eps=1e-10):
    mse_node = sum_err2 / float(steps)
    rmse_1d = torch.sqrt(w1d * mse_node[:num_1d] + eps).mean()
    rmse_2d = torch.sqrt(w2d * mse_node[num_1d:] + eps).mean()
    return 0.5 * rmse_1d + 0.5 * rmse_2d

class EMA:
    """Exponential Moving Average of model parameters for better generalization."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

EXTRACT_PATH = os.path.join(DATA_PATH, "Models") if not os.path.isdir(os.path.join(DATA_PATH, "Models")) else DATA_PATH
if os.path.isdir(os.path.join(DATA_PATH, "Models", "Model_1")):
    EXTRACT_PATH = DATA_PATH
elif os.path.isdir(os.path.join(DATA_PATH, "..", "Models", "Model_1")):
    EXTRACT_PATH = os.path.join(DATA_PATH, "..")

def prepare_model_runtime(mid, dataset_mode, cache_events=False):
    base = os.path.join(EXTRACT_PATH, "Models", f"Model_{mid}", dataset_mode)
    print(f"\n--- Model {mid} ({base}) ---")
    loader = FloodDataLoader(base, mid)
    loader.load_static()
    graph = GraphBuilder(loader)
    graph.build()

    cfg = SCALE_CONFIG[mid]
    pp = Preprocessor(loader, graph, cfg, MODEL_CONFIG)
    runtime = {
        "loader": loader,
        "graph": graph,
        "preprocessor": pp,
        "adj": build_gcn_csr(graph.edge_index, graph.total_nodes, DEVICE, dtype=torch.bfloat16),
        "static_feat": graph.get_static_features().to(DEVICE),
        "edge_static": graph.get_edge_static_features().to(DEVICE),
    }

    if not cache_events:
        return runtime

    events = loader.scan_events()
    print(f"  Events found: {len(events)}")

    cached = []
    for meta in tqdm(events, desc=f"Caching M{mid}"):
        try:
            ev = loader.load_event(meta)
            if ev is None:
                continue
            node_x, node_y, edge_x = pp.process_event(ev)
            T = node_x.shape[0]
            if T < SEQ_LENGTH + 1:
                continue
            node_x = np.clip(np.nan_to_num(node_x), -10, 10).astype(np.float32)
            node_y = np.nan_to_num(node_y).astype(np.float32)
            if edge_x is not None:
                edge_x = np.clip(np.nan_to_num(edge_x), -10, 10).astype(np.float32)

            max_horizon = min(PRED_HORIZON_CONFIG.get(mid, 90), T - SEQ_LENGTH)
            n_windows = T - SEQ_LENGTH - max_horizon + 1
            if n_windows <= 0:
                n_windows = 1

            cached.append(
                {
                    "node_feats": torch.from_numpy(node_x).float().contiguous().pin_memory() if DEVICE.type == "cuda" else torch.from_numpy(node_x).float().contiguous(),
                    "tgts": torch.from_numpy(node_y).float().contiguous().pin_memory() if DEVICE.type == "cuda" else torch.from_numpy(node_y).float().contiguous(),
                    "edge_feats": (
                        None
                        if edge_x is None
                        else (
                            torch.from_numpy(edge_x).to(torch.float16).contiguous().pin_memory()
                            if DEVICE.type == "cuda"
                            else torch.from_numpy(edge_x).to(torch.float16).contiguous()
                        )
                    ),
                    "T": T,
                    "n_windows": n_windows,
                    "event_id": meta["event_id"],
                    "sampler_state": {},
                }
            )
        except Exception:
            continue

    ncfg = NOISE_CONFIG[mid]
    runtime["cached"] = cached
    runtime["input_noise_scale"] = torch.cat([
        torch.full((graph.num_1d, 1), ncfg["input_1d"]),
        torch.full((graph.num_2d, 1), ncfg["input_2d"]),
    ]).to(DEVICE)
    runtime["rollout_noise_scale"] = torch.cat([
        torch.full((graph.num_1d, 1), ncfg["rollout_1d"]),
        torch.full((graph.num_2d, 1), ncfg["rollout_2d"]),
    ]).to(DEVICE)
    print(f"  Noise scales — input: 1d={ncfg['input_1d']}, 2d={ncfg['input_2d']} | rollout: 1d={ncfg['rollout_1d']}, 2d={ncfg['rollout_2d']}")

    total_windows = sum(d["n_windows"] for d in cached)
    total_timesteps = sum(d["T"] for d in cached)
    total_bytes = sum(
        d["node_feats"].nelement() * 4
        + d["tgts"].nelement() * 4
        + (0 if d["edge_feats"] is None else d["edge_feats"].nelement() * 2)
        for d in cached
    )
    print(f"  Cached: {len(cached)} events, {total_timesteps:,} timesteps, ~{total_windows:,} possible windows")
    print(f"  Approx storage: ~{total_bytes / 1024**2:.0f} MB")
    return runtime


def prepare_training_data(model_ids):
    model_data = {}
    for mid in model_ids:
        model_data[mid] = prepare_model_runtime(mid, "train", cache_events=True)
    gc.collect()
    print(f"\nData ready on {DEVICE}")
    return model_data

def inverse_sigmoid_tf_ratio(epoch, k=TF_SIGMOID_K):
    return 0.0


def slice_window(ev, start, seq_len, horizon):
    node_feats = ev["node_feats"]
    tgts = ev["tgts"]
    T = ev["T"]
    avail = T - start - seq_len
    h = min(horizon, avail)
    if h <= 0:
        return None, None, None, None
    x_node = node_feats[start : start + seq_len]
    y_tgt = tgts[start + seq_len : start + seq_len + h]
    rain_future = node_feats[start + seq_len : start + seq_len + h, :, 1]
    if ev["edge_feats"] is None:
        x_edge = None
    else:
        x_edge = ev["edge_feats"][start : start + seq_len]
    return x_node, x_edge, y_tgt, rain_future


def sample_windows(ev, n_samples, seq_len, horizon):
    T = ev["T"]
    max_start = T - seq_len - horizon
    if max_start < 0:
        return []
    total = max_start + 1
    if total <= n_samples:
        return list(range(total))

    state = ev.setdefault("sampler_state", {})
    horizon_state = state.get(max_start)
    starts = np.arange(total, dtype=np.int64)
    if horizon_state is None:
        thirds = np.array_split(starts, 3)
        buckets = [arr.copy() for arr in thirds]
        orders = [arr[np.random.permutation(len(arr))] if len(arr) > 0 else arr for arr in buckets]
        horizon_state = {
            "bucket_orders": orders,
            "bucket_pos": [0, 0, 0],
            "global_order": starts[np.random.permutation(total)],
            "global_pos": 0,
        }
        state[max_start] = horizon_state

    def take_from_order(order, pos, k):
        if k <= 0 or len(order) == 0:
            return np.empty((0,), dtype=np.int64), pos, order
        take = min(k, len(order))
        if pos + take <= len(order):
            out = order[pos : pos + take]
            new_pos = pos + take
            new_order = order
            if new_pos == len(order):
                new_order = order[np.random.permutation(len(order))]
                new_pos = 0
            return out, new_pos, new_order
        tail = order[pos:]
        need = take - len(tail)
        reshuffled = order[np.random.permutation(len(order))]
        head = reshuffled[:need]
        out = np.concatenate([tail, head])
        new_pos = need
        return out, new_pos, reshuffled

    quotas = [n_samples // 3, n_samples // 3, n_samples // 3]
    for idx in (1, 0, 2):
        if sum(quotas) < n_samples:
            quotas[idx] += 1

    chosen = []
    chosen_set = set()
    for bi, need in enumerate(quotas):
        order = horizon_state["bucket_orders"][bi]
        pos = horizon_state["bucket_pos"][bi]
        picks, new_pos, new_order = take_from_order(order, pos, need)
        horizon_state["bucket_orders"][bi] = new_order
        horizon_state["bucket_pos"][bi] = new_pos
        for s in picks.tolist():
            if s not in chosen_set:
                chosen.append(s)
                chosen_set.add(s)

    if len(chosen) < n_samples:
        order = horizon_state["global_order"]
        pos = horizon_state["global_pos"]
        extra_need = n_samples - len(chosen)
        seen_in_pass = 0
        max_scan = len(order) * 2
        while extra_need > 0 and seen_in_pass < max_scan:
            if pos >= len(order):
                order = order[np.random.permutation(len(order))]
                pos = 0
            s = int(order[pos])
            pos += 1
            seen_in_pass += 1
            if s in chosen_set:
                continue
            chosen.append(s)
            chosen_set.add(s)
            extra_need -= 1
        horizon_state["global_order"] = order
        horizon_state["global_pos"] = pos

    return chosen


def model_factory(data, model_config):
    graph = data["graph"]
    if model_config["model_kind"] == "baseline":
        return SimpleFloodTGCN(
            num_nodes=graph.total_nodes,
            in_features=data["preprocessor"].node_feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            static_features=data["static_feat"],
            adj=data["adj"],
            raw_base_feature_dim=data["preprocessor"].base_node_feature_dim,
            engineered_feature_dim=data["preprocessor"].node_feature_dim - data["preprocessor"].base_node_feature_dim,
            use_feature_gates=model_config.get("use_feature_gates", False),
        )

    return EdgeAwareFloodModel(
        num_nodes=graph.total_nodes,
        num_edges=graph.num_edges,
        node_in_dim=data["preprocessor"].node_feature_dim,
        edge_in_dim=data["preprocessor"].edge_feature_dim,
        node_hidden_dim=HIDDEN_DIM,
        edge_hidden_dim=EDGE_HIDDEN_DIM,
        dropout=DROPOUT,
        node_static=data["static_feat"],
        edge_static=data["edge_static"],
        edge_index=graph.edge_index.to(DEVICE),
    )


def predict_multistep(model, preprocessor, x_node, x_edge, rain_future, num_steps):
    x_node = preprocessor.prepare_warmup_sequence(x_node)
    rollout_ctx = preprocessor.init_rollout_context(x_node)
    state = model.warmup(x_node, None if x_edge is None else x_edge.to(dtype=torch.float32))
    current_depth = x_node[-1, :, 0:1].clone()
    preds = torch.empty((num_steps, current_depth.shape[0], 1), device=x_node.device, dtype=x_node.dtype)

    for step in range(num_steps):
        pred = model.decode(state, current_depth)
        preds[step] = pred
        current_depth = pred
        if step < num_steps - 1:
            rain_t = rain_future[step].unsqueeze(-1)
            node_dyn, rollout_ctx = preprocessor.build_rollout_node_frame(current_depth, rain_t, rollout_ctx)
            state = model.step(state, node_dyn, None)
    return preds


def checkpoint_path_for(mid, seed):
    candidates = [
        os.path.join(os.getcwd(), f"model_m{mid}_seed{seed}.pth"),
        os.path.join(os.getcwd(), f"m{mid}_seed{seed}.pth"),
        os.path.join("/content/drive/MyDrive/Urban Flood Bench", f"model_m{mid}_seed{seed}.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

def trainer_checkpoint_path_for(mid, seed):
    return os.path.join(os.getcwd(), f"trainer_m{mid}_seed{seed}.pth")

def _state_has_nan(state_dict):
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and torch.isnan(value).any().item():
            return True
    return False

def load_checkpoint_strict(model, ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    load_attempts = [state]
    if any(k.startswith("_orig_mod.") for k in state):
        load_attempts.append({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    else:
        load_attempts.append({f"_orig_mod.{k}": v for k, v in state.items()})

    errors = []
    for candidate in load_attempts:
        try:
            model.load_state_dict(candidate, strict=True)
            return model
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        f"Checkpoint {ckpt_path} does not match the expected architecture.\n"
        + "\n---\n".join(errors)
    )


def collect_sampler_state(events):
    sampler_state = {}
    for ev in events:
        sampler = ev.get("sampler_state")
        if sampler:
            sampler_state[ev["event_id"]] = sampler
    return sampler_state


def restore_sampler_state(events, saved_sampler_state):
    if not saved_sampler_state:
        return
    by_event_id = {ev["event_id"]: ev for ev in events}
    for event_id, sampler in saved_sampler_state.items():
        if event_id in by_event_id:
            by_event_id[event_id]["sampler_state"] = sampler


def save_training_checkpoint(path, model, optimizer, scheduler, ema, epoch, best_score, patience_ctr, history, train_events):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "epoch": epoch,
        "best_score": best_score,
        "patience_ctr": patience_ctr,
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_sampler_state": collect_sampler_state(train_events),
    }
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_training_checkpoint(path, model, optimizer, scheduler, ema, train_events):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Trainer checkpoint {path} has unsupported format: {type(checkpoint)!r}")
    for required_key in ("model_state", "optimizer_state", "scheduler_state", "ema_shadow"):
        if required_key not in checkpoint:
            raise RuntimeError(f"Trainer checkpoint {path} is missing required key {required_key!r}")

    # --- Align prefixes based on whether the current model is compiled ---
    expected_keys = set(model.state_dict().keys())
    is_compiled = any(k.startswith("_orig_mod.") for k in expected_keys)

    model_state = checkpoint["model_state"]
    aligned_model_state = {}
    for k, v in model_state.items():
        base_k = k.removeprefix("_orig_mod.")
        new_k = f"_orig_mod.{base_k}" if is_compiled else base_k
        aligned_model_state[new_k] = v

    ema_shadow = checkpoint["ema_shadow"]
    aligned_ema_shadow = {}
    for k, v in ema_shadow.items():
        base_k = k.removeprefix("_orig_mod.")
        new_k = f"_orig_mod.{base_k}" if is_compiled else base_k
        aligned_ema_shadow[new_k] = v
    # ---------------------------------------------------------------------

    if _state_has_nan(aligned_model_state) or _state_has_nan(aligned_ema_shadow):
        raise RuntimeError(f"Trainer checkpoint {path} contains NaN values.")

    model.load_state_dict(aligned_model_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    ema.shadow = aligned_ema_shadow

    restore_sampler_state(train_events, checkpoint.get("train_sampler_state"))

    # Safely restore RNG states
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        try:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        except Exception as exc:
            print(f"  ! Skipping CPU RNG restore from {path}: {exc}")

    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        try:
            # Ensure states are on CPU before loading them into the CUDA RNG
            cuda_states = [state.cpu() if isinstance(state, torch.Tensor) else state
                           for state in checkpoint["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as exc:
            print(f"  ! Skipping CUDA RNG restore from {path}: {exc}")

    return checkpoint


def predict_event_water_levels(model, runtime, event_data):
    preprocessor = runtime["preprocessor"]
    graph = runtime["graph"]
    cfg = SCALE_CONFIG[runtime["loader"].model_id]

    node_x, _, raw_wl, ids_1d, ids_2d = preprocessor._node_features(event_data)
    edge_x = preprocessor._edge_features(event_data)
    total_steps = node_x.shape[0]
    future_steps = total_steps - SPIN_UP_LEN
    if future_steps <= 0:
        return None, ids_1d, ids_2d

    node_x = np.clip(np.nan_to_num(node_x), -10, 10).astype(np.float32)
    if edge_x is not None:
        edge_x = np.clip(np.nan_to_num(edge_x), -10, 10).astype(np.float32)

    x_node = torch.from_numpy(node_x[:SPIN_UP_LEN]).to(DEVICE, dtype=torch.float32)
    x_edge = None if edge_x is None else torch.from_numpy(edge_x[:SPIN_UP_LEN]).to(DEVICE, dtype=torch.float32)
    rain_future = torch.from_numpy(node_x[SPIN_UP_LEN:, :, preprocessor.rain_idx]).to(DEVICE, dtype=torch.float32)

    with torch.inference_mode(), amp_context():
        pred_seq = predict_multistep(model, preprocessor, x_node, x_edge, rain_future, future_steps)

    pred_np = pred_seq.squeeze(-1).float().cpu().numpy()
    invert_elev = preprocessor._invert_elev
    min_elev = preprocessor._min_elev
    wl_1d = pred_np[:, : graph.num_1d] * cfg["scale_1d"] + invert_elev
    wl_2d = pred_np[:, graph.num_1d :] * cfg["scale_2d"] + min_elev
    wl_1d = np.clip(wl_1d, invert_elev - 5, invert_elev + 50)
    wl_2d = np.clip(wl_2d, min_elev - 2, min_elev + 30)
    pred_raw = np.concatenate([wl_1d, wl_2d], axis=1)

    if raw_wl.shape[0] >= SPIN_UP_LEN:
        last_wl = raw_wl[SPIN_UP_LEN - 1]
        pred_raw = np.where(np.isnan(pred_raw), last_wl[np.newaxis, :], pred_raw)

    return pred_raw.astype(np.float32), ids_1d, ids_2d


def run_inference():
    print(
        f"Device: {DEVICE} | Mode: {args.mode} | Config: {MODEL_CONFIG['name']} | "
        f"Models: {MODEL_IDS} | Seeds: {ENSEMBLE_SEEDS}"
    )
    print(f"Data: {DATA_PATH}")

    pred_columns = ["model_id", "event_id", "node_type", "node_id", "water_level"]
    predictions = []
    for mid in MODEL_IDS:
        runtime = prepare_model_runtime(mid, "test", cache_events=False)
        print(f"\n{'=' * 60}")
        print(
            f"Inference Model {mid} | config={MODEL_CONFIG['name']} | seeds={ENSEMBLE_SEEDS}"
        )
        print(f"{'=' * 60}")

        ensemble = []
        for seed in ENSEMBLE_SEEDS:
            ckpt_path = checkpoint_path_for(mid, seed)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            model = model_factory(runtime, MODEL_CONFIG).to(DEVICE)
            load_checkpoint_strict(model, ckpt_path)
            model.eval()
            ensemble.append(model)

        test_events = runtime["loader"].scan_events()
        rows = []
        for meta in tqdm(test_events, desc=f"Infer M{mid}"):
            ev = runtime["loader"].load_event(meta)
            if ev is None:
                continue
            preds = []
            ids_1d = None
            ids_2d = None
            for model in ensemble:
                pred_raw, ids_1d, ids_2d = predict_event_water_levels(model, runtime, ev)
                if pred_raw is not None:
                    preds.append(pred_raw)
            if not preds:
                continue

            pred = np.mean(preds, axis=0)
            for i, nid in enumerate(ids_1d):
                for t in range(pred.shape[0]):
                    rows.append(
                        {
                            "model_id": mid,
                            "event_id": meta["event_id"],
                            "node_type": 1,
                            "node_id": int(nid),
                            "water_level": float(pred[t, i]),
                        }
                    )
            for i, nid in enumerate(ids_2d):
                ni = runtime["graph"].num_1d + i
                for t in range(pred.shape[0]):
                    rows.append(
                        {
                            "model_id": mid,
                            "event_id": meta["event_id"],
                            "node_type": 2,
                            "node_id": int(nid),
                            "water_level": float(pred[t, ni]),
                        }
                    )

        pred_df = pd.DataFrame(rows, columns=pred_columns)
        pred_df["timestep_idx"] = pred_df.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()
        predictions.append(pred_df)
        print(f"  Rows: {len(pred_df):,}")
        del ensemble
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    all_preds = pd.concat(predictions, ignore_index=True)
    if all_preds.empty:
        raise RuntimeError("Inference produced no predictions.")

    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        sample["timestep_idx"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()
        merged = sample.drop(columns=["water_level"]).merge(
            all_preds,
            on=["model_id", "event_id", "node_type", "node_id", "timestep_idx"],
            how="left",
            validate="one_to_one",
        )
        if merged["water_level"].isna().any():
            missing = int(merged["water_level"].isna().sum())
            raise RuntimeError(f"Submission merge left {missing} rows without predictions.")
        submission = merged.drop(columns=["timestep_idx"])
    else:
        print(
            "⚠️ sample_submission.csv not found; using sequential row_id fallback with sorted prediction rows."
        )
        submission = all_preds.sort_values(
            ["model_id", "event_id", "node_type", "node_id", "timestep_idx"]
        ).reset_index(drop=True)
        submission.insert(0, "row_id", np.arange(len(submission), dtype=np.int64))
        submission = submission.drop(columns=["timestep_idx"])

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved submission to {OUTPUT_PATH}")


def train_single_model(model, train_events, val_events, criterion, mid, seed, preprocessor, input_noise_scale=None, rollout_noise_scale=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=DEVICE.type == "cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )
    ema = EMA(model, decay=EMA_DECAY)
    ckpt_path = checkpoint_path_for(mid, seed)
    trainer_ckpt_path = trainer_checkpoint_path_for(mid, seed)
    best_score = float("inf")
    patience_ctr = 0
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "score": [], "tf_ratio": []}
    def run_validation(events):
        ema.apply_shadow(model)
        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.inference_mode(), amp_context():
            for ev in events:
                if ev["T"] <= SEQ_LENGTH:
                    continue
                x_node = ev["node_feats"][:SEQ_LENGTH].to(DEVICE, dtype=torch.float32, non_blocking=True)
                x_edge = None
                if ev["edge_feats"] is not None:
                    x_edge = ev["edge_feats"][:SEQ_LENGTH].to(DEVICE, dtype=torch.float32, non_blocking=True)
                y_tgt = ev["tgts"][SEQ_LENGTH:].to(DEVICE, dtype=torch.float32, non_blocking=True)
                rain_future = ev["node_feats"][SEQ_LENGTH:, :, preprocessor.rain_idx].to(DEVICE, dtype=torch.float32, non_blocking=True)
                pred_seq = predict_multistep(model, preprocessor, x_node, x_edge, rain_future, y_tgt.shape[0])
                sum_err2 = (pred_seq - y_tgt).float().pow(2).sum(dim=0)
                loss = kaggle_like_window_loss_from_sumerr2(
                    sum_err2,
                    y_tgt.shape[0],
                    num_1d=criterion.num_1d,
                    w1d=criterion.w1d,
                    w2d=criterion.w2d,
                )
                if not torch.isnan(loss):
                    v_loss += loss.item()
                    v_n += 1
        ema.restore(model)
        return v_loss / max(v_n, 1)

    if os.path.exists(trainer_ckpt_path):
        print(f"  ↻ Warm resuming from {trainer_ckpt_path}")
        trainer_state = load_training_checkpoint(trainer_ckpt_path, model, optimizer, scheduler, ema, train_events)
        start_epoch = int(trainer_state["epoch"]) + 1
        history = trainer_state["history"]  # <--- YES, keep this so your charts don't break!

        # Grab the old stats just to print them out for your logs
        old_best = float(trainer_state["best_score"])
        old_pat = int(trainer_state["patience_ctr"])
        print(f"  Resume epoch: {start_epoch} | old best val: {old_best:.6f} | old patience: {old_pat}")

        # --- THE FIX: Force a reset of the best score and patience ---
        best_score = float("inf")
        patience_ctr = 0
        print("  ⚠️ Resetting best_score to infinity and patience to 0 for the new 210-step difficulty!")
        # -------------------------------------------------------------
    elif os.path.exists(ckpt_path):
        print(f"  ↻ Resuming from {ckpt_path}")
        load_checkpoint_strict(model, ckpt_path)
        ema = EMA(model, decay=EMA_DECAY)  # re-init EMA from loaded weights
        best_score = run_validation(val_events)
        print(f"  Resume baseline val: {best_score:.6f}")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        if LR_WARMUP_EPOCHS > 0:
            if epoch < LR_WARMUP_EPOCHS:
                frac = epoch / max(LR_WARMUP_EPOCHS - 1, 1)
                lr = LEARNING_RATE / 10.0 + (LEARNING_RATE - LEARNING_RATE / 10.0) * frac
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif epoch == LR_WARMUP_EPOCHS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE

        train_steps_max = TRAIN_STEPS_CONFIG.get(mid, 90)
        n_steps = min(INITIAL_TRAIN_STEPS + epoch * 2, train_steps_max)
        tf_ratio = inverse_sigmoid_tf_ratio(epoch)
        model.train()
        train_loss, train_n = 0.0, 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        order = np.random.permutation(len(train_events))

        pbar = tqdm(order, desc=f"[seed={seed}] Ep {epoch + 1}/{EPOCHS}", leave=False)
        for ei in pbar:
            ev = train_events[ei]
            active_ev = {
                "node_feats": ev["node_feats"].to(DEVICE, dtype=torch.float32, non_blocking=True),
                "tgts": ev["tgts"].to(DEVICE, dtype=torch.float32, non_blocking=True),
                "edge_feats": None if ev["edge_feats"] is None else ev["edge_feats"].to(DEVICE, dtype=torch.float32, non_blocking=True),
                "T": ev["T"],
            }
            starts = sample_windows(ev, SAMPLES_PER_EVENT, SEQ_LENGTH, n_steps)
            for s in starts:
                x_node, x_edge, y_tgt, rain_future = slice_window(active_ev, s, SEQ_LENGTH, n_steps)
                if x_node is None:
                    continue
                actual_steps = min(n_steps, y_tgt.shape[0])

                with amp_context():
                    # Input noise regularization (per-node-type scale)
                    if input_noise_scale is not None:
                        x_node_w = x_node + input_noise_scale.unsqueeze(0) * torch.randn_like(x_node)
                    else:
                        x_node_w = x_node
                    x_node_w = preprocessor.prepare_warmup_sequence(x_node_w)
                    rollout_ctx = preprocessor.init_rollout_context(x_node_w)
                    state = model.warmup(x_node_w, x_edge)
                    current_depth = x_node[-1, :, 0:1].clone()  # clean depth for init
                    sum_weighted_err2 = torch.zeros((model.num_nodes, 1), device=DEVICE, dtype=torch.float32)
                    total_weight = 0.0

                    for step in range(actual_steps):
                        pred = model.decode(state, current_depth)
                        err = (pred - y_tgt[step]).float()
                        # Time-weighted loss: later steps weighted more
                        tw = 1.0 + TIME_WEIGHT_STRENGTH * (step / max(actual_steps - 1, 1))
                        sum_weighted_err2 += tw * err * err
                        total_weight += tw
                        # Stochastic teacher forcing: per-step random decision
                        use_teacher = torch.rand(1).item() < tf_ratio
                        if use_teacher:
                            current_depth = y_tgt[step].detach()
                        else:
                            # Rollout noise: per-node-type scale
                            if rollout_noise_scale is not None:
                                current_depth = pred + rollout_noise_scale * torch.randn_like(pred)
                            else:
                                current_depth = pred
                        if step < actual_steps - 1:
                            rain_t = rain_future[step].unsqueeze(-1)
                            node_dyn, rollout_ctx = preprocessor.build_rollout_node_frame(current_depth, rain_t, rollout_ctx)
                            state = model.step(state, node_dyn, None)

                    loss = kaggle_like_window_loss_from_sumerr2(
                        sum_weighted_err2,
                        total_weight,
                        num_1d=criterion.num_1d,
                        w1d=criterion.w1d,
                        w2d=criterion.w2d,
                    )
                    if torch.isnan(loss):
                        continue
                    (loss / GRAD_ACCUM_STEPS).backward()
                    accum_count += 1

                    if accum_count % GRAD_ACCUM_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        ema.update(model)

                    train_loss += loss.item()
                    train_n += 1

            del active_ev

            pbar.set_postfix({"loss": f"{train_loss / max(train_n, 1):.6f}", "steps": n_steps, "tf": f"{tf_ratio:.2f}"})

        # Flush remaining accumulated gradients
        if accum_count % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        avg_t = train_loss / max(train_n, 1)
        avg_v = run_validation(val_events)
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        history["score"].append(avg_v)
        history["tf_ratio"].append(tf_ratio)

        dt = time.time() - t0
        print(
            f"  Ep {epoch + 1:3d} | train {avg_t:.6f} | val {avg_v:.6f} | "
            f"steps={n_steps} tf={tf_ratio:.3f}({int(n_steps * tf_ratio)}/{n_steps}) | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.0f}s"
        )

        scheduler.step(avg_v)
        if avg_v < best_score:
            best_score = avg_v
            patience_ctr = 0
            ema.apply_shadow(model)
            torch.save(model.state_dict(), ckpt_path)
            ema.restore(model)
            print(f"    -> saved best ({best_score:.6f})")
        else:
            patience_ctr += 1

        save_training_checkpoint(
            trainer_ckpt_path,
            model,
            optimizer,
            scheduler,
            ema,
            epoch,
            best_score,
            patience_ctr,
            history,
            train_events,
        )

        if start_epoch > 0:
            print(f"    -> saved trainer state ({trainer_ckpt_path})")

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    load_checkpoint_strict(model, ckpt_path)
    model.eval()
    return model, history

def run_training():
    print(
        f"Device: {DEVICE} | Mode: {args.mode} | Epochs: {EPOCHS} | Hidden: {HIDDEN_DIM} | "
        f"EdgeHidden: {EDGE_HIDDEN_DIM} | Layers: {NUM_LAYERS} | Config: {MODEL_CONFIG['name']}"
    )
    print(f"Data: {DATA_PATH}")
    print("Teacher Forcing: stochastic per-step | Cosine LR | Head dropout=0 | Mean edge agg")

    model_data = prepare_training_data(MODEL_IDS)
    trained_ensembles = {}
    trained_histories = {}

    for mid in MODEL_IDS:
        data = model_data[mid]
        cached = data["cached"]
        # Deterministic split by model_id for reproducibility across runs
        split_rng = np.random.RandomState(SEED + mid)
        split_indices = split_rng.permutation(len(cached))
        n_val = max(1, int(len(cached) * 0.15))
        val_events = [cached[i] for i in split_indices[:n_val]]
        train_events = [cached[i] for i in split_indices[n_val:]]
        criterion = FloodLoss(data["graph"].num_1d, mid, SCALE_CONFIG[mid]["scale_1d"], SCALE_CONFIG[mid]["scale_2d"])

        print(f"\n{'=' * 60}")
        print(
            f"ENSEMBLE Training Model {mid} | config={MODEL_CONFIG['name']} | "
            f"train={len(train_events)}, val={len(val_events)}"
        )
        print(f"{'=' * 60}")

        ensemble_models = []
        ensemble_histories = []
        total_ensemble = len(ENSEMBLE_SEEDS)
        for ens_idx, seed in enumerate(ENSEMBLE_SEEDS, start=1):
            print(f"\n--- Ensemble member {ens_idx}/{total_ensemble} (seed={seed}) ---")
            model = model_factory(data, MODEL_CONFIG).to(DEVICE)
            params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {params:,}")
            if DEVICE.type == "cuda":
                model = torch.compile(model, mode="default", dynamic=False)
                print("  torch.compile enabled")
            model, history = train_single_model(
                model, train_events, val_events, criterion, mid, seed, data["preprocessor"],
                input_noise_scale=data["input_noise_scale"],
                rollout_noise_scale=data["rollout_noise_scale"],
            )
            ensemble_models.append(model)
            ensemble_histories.append(history)
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        trained_ensembles[mid] = ensemble_models
        trained_histories[mid] = ensemble_histories
        print(f"\n✅ Model {mid} ensemble done")

    print("\n✅ All ensembles trained")
    return trained_ensembles, trained_histories


def main():
    parsed_args = build_arg_parser().parse_args()
    initialize_runtime(parsed_args)

    if args.mode == "infer":
        run_inference()
    else:
        run_training()


if __name__ == "__main__":
    main()
