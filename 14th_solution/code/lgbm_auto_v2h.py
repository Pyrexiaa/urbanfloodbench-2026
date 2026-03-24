"""
UrbanFloodBench: Autoregressive LightGBM v2h
- v2a baseline with warmup-aligned training
- Only M2_2D uses connected 1D state features from 1d2d links
- No 2D postprocess
"""

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

warnings.filterwarnings("ignore")


def log(msg=""):
    print(msg, flush=True)


# ============================================================
# 1. Constants & Configuration
# ============================================================
if Path("/kaggle/input").exists():
    DATA_DIR = Path("/kaggle/input/datasets/attiqueansari/urbanfloodbench/Models")
    SUBMISSION_PATH = Path("/kaggle/input/competitions/urban-flood-modelling/sample_submission.parquet")
    OUTPUT_PATH = Path("/kaggle/working/submission.csv")
else:
    DATA_DIR = Path("/Users/shionsuio/Downloads/Models")
    SUBMISSION_PATH = Path("/Users/shionsuio/Downloads/Urban Flood Modelling/sample_submission.parquet")
    OUTPUT_PATH = Path("/Users/shionsuio/urban-flood/submission_auto_v2h.csv")

WARMUP_STEPS = 10
MODELS = [1, 2]
NODE_TYPES = [1, 2]

LGB_PARAMS_M1 = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "device": "gpu",
}

LGB_PARAMS_M2 = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "min_child_samples": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "device": "gpu",
}

NUM_BOOST_ROUND_M1 = 5000
NUM_BOOST_ROUND_M2 = 10000
EARLY_STOPPING = 100
SUBSAMPLE_2D = 0.5
NOISE_STD = 0.02


# ============================================================
# 2. Data Loading Utilities
# ============================================================
def get_event_ids(model_id, split="train"):
    base = DATA_DIR / f"Model_{model_id}" / split
    return sorted(int(d.name.split("_")[1]) for d in base.iterdir() if d.name.startswith("event_"))


def load_static_1d(model_id):
    return pd.read_csv(DATA_DIR / f"Model_{model_id}" / "train" / "1d_nodes_static.csv").astype({"node_idx": np.int32})


def load_static_2d(model_id):
    return pd.read_csv(DATA_DIR / f"Model_{model_id}" / "train" / "2d_nodes_static.csv").astype({"node_idx": np.int32})


def load_edge_index(model_id, dim="2d"):
    return pd.read_csv(DATA_DIR / f"Model_{model_id}" / "train" / f"{dim}_edge_index.csv")


def load_event_dynamic(model_id, event_id, split, node_type_str):
    base = DATA_DIR / f"Model_{model_id}" / split / f"event_{event_id}"
    df = pd.read_csv(base / f"{node_type_str}_nodes_dynamic_all.csv")
    df["timestep"] = df["timestep"].astype(np.int32)
    df["node_idx"] = df["node_idx"].astype(np.int32)
    return df


def load_timesteps(model_id, event_id, split):
    return pd.read_csv(DATA_DIR / f"Model_{model_id}" / split / f"event_{event_id}" / "timesteps.csv")


def load_1d2d_connections(model_id, split):
    return pd.read_csv(DATA_DIR / f"Model_{model_id}" / split / "1d2d_connections.csv")


def use_conn_features(model_id, node_type):
    return model_id == 2 and node_type == 2


# ============================================================
# 3. Graph Features
# ============================================================
def build_graph_features(model_id, dim="2d"):
    ei = load_edge_index(model_id, dim)
    if dim == "2d":
        static = load_static_2d(model_id)
        static["min_elevation"] = static["min_elevation"].fillna(static["elevation"])
        elev_col = "min_elevation"
    else:
        static = load_static_1d(model_id)
        elev_col = "invert_elevation"

    elev = static.set_index("node_idx")[elev_col].to_dict()
    n_nodes = len(static)

    neighbors = defaultdict(list)
    for _, row in ei.iterrows():
        fn, tn = int(row["from_node"]), int(row["to_node"])
        neighbors[fn].append(tn)
        neighbors[tn].append(fn)

    degrees = np.zeros(n_nodes, dtype=np.float32)
    neigh_elev_mean = np.zeros(n_nodes, dtype=np.float32)
    neigh_elev_min = np.full(n_nodes, np.inf, dtype=np.float32)
    neigh_elev_max = np.full(n_nodes, -np.inf, dtype=np.float32)
    neigh_elev_std = np.zeros(n_nodes, dtype=np.float32)

    for nid in range(n_nodes):
        nbrs = neighbors.get(nid, [])
        deg = len(nbrs)
        degrees[nid] = deg
        if deg > 0:
            elevs = np.array([elev.get(n, 0.0) for n in nbrs], dtype=np.float32)
            neigh_elev_mean[nid] = elevs.mean()
            neigh_elev_min[nid] = elevs.min()
            neigh_elev_max[nid] = elevs.max()
            neigh_elev_std[nid] = elevs.std()
        else:
            neigh_elev_mean[nid] = elev.get(nid, 0.0)
            neigh_elev_min[nid] = elev.get(nid, 0.0)
            neigh_elev_max[nid] = elev.get(nid, 0.0)

    return pd.DataFrame({
        "node_idx": np.arange(n_nodes, dtype=np.int32),
        "degree": degrees,
        "neighbor_elev_mean": neigh_elev_mean,
        "neighbor_elev_min": neigh_elev_min,
        "neighbor_elev_max": neigh_elev_max,
        "neighbor_elev_std": neigh_elev_std,
    })


def build_adjacency(model_id, node_type, n_nodes, node_ids):
    dim = "1d" if node_type == 1 else "2d"
    ei = load_edge_index(model_id, dim)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    neighbors_list = [[] for _ in range(n_nodes)]
    for _, row in ei.iterrows():
        fn, tn = int(row["from_node"]), int(row["to_node"])
        fi_idx = node_to_idx.get(fn)
        ti_idx = node_to_idx.get(tn)
        if fi_idx is not None and ti_idx is not None:
            neighbors_list[fi_idx].append(ti_idx)
            neighbors_list[ti_idx].append(fi_idx)

    max_deg = max((len(n) for n in neighbors_list), default=1)
    neighbor_idx = np.zeros((n_nodes, max_deg), dtype=np.int32)
    neighbor_mask = np.zeros((n_nodes, max_deg), dtype=bool)
    for i, nbrs in enumerate(neighbors_list):
        for j, nb in enumerate(nbrs):
            neighbor_idx[i, j] = nb
            neighbor_mask[i, j] = True
    degree = np.maximum(neighbor_mask.sum(axis=1).astype(np.float32), 1.0)
    return neighbor_idx, neighbor_mask, degree


def compute_neighbor_wl(wl_vector, neighbor_idx, neighbor_mask, degree):
    neighbor_wl = wl_vector[neighbor_idx]
    neighbor_wl[~neighbor_mask] = 0.0
    neighbor_wl_mean = neighbor_wl.sum(axis=1) / degree
    neighbor_wl_diff = neighbor_wl_mean - wl_vector
    return neighbor_wl_mean, neighbor_wl_diff


def build_node_feature_table(model_id, node_type):
    dim = "1d" if node_type == 1 else "2d"

    if node_type == 2:
        static = load_static_2d(model_id)
        static["min_elevation"] = static["min_elevation"].fillna(static["elevation"])
        static["area"] = static["area"].clip(lower=0).fillna(0)
        feat_cols = ["position_x", "position_y",
                     "area", "roughness", "min_elevation", "elevation",
                     "aspect", "curvature", "flow_accumulation"]
        static["elev_above_min"] = static["elevation"] - static["min_elevation"]
        feat_cols.append("elev_above_min")
    else:
        static = load_static_1d(model_id)
        feat_cols = ["position_x", "position_y",
                     "depth", "invert_elevation", "surface_elevation", "base_area"]
        static["elev_range"] = static["surface_elevation"] - static["invert_elevation"]

        edge_path = DATA_DIR / f"Model_{model_id}" / "train" / "1d_edges_static.csv"
        edges = pd.read_csv(edge_path)
        ei = load_edge_index(model_id, "1d")

        pipe_cap = defaultdict(list)
        for _, row in ei.iterrows():
            eidx = int(row["edge_idx"])
            if eidx < len(edges):
                e = edges.iloc[eidx]
                diameter = e.get("diameter", 0)
                pipe_cap[int(row["from_node"])].append(diameter)
                pipe_cap[int(row["to_node"])].append(diameter)

        cap_vals = np.zeros(len(static), dtype=np.float32)
        for nid in range(len(static)):
            caps = pipe_cap.get(nid, [0])
            cap_vals[nid] = np.mean(caps)
        static["pipe_capacity"] = cap_vals
        feat_cols.extend(["elev_range", "pipe_capacity"])

    graph_feats = build_graph_features(model_id, dim)
    merged = static.merge(graph_feats, on="node_idx", how="left")

    all_feat_cols = feat_cols + ["degree", "neighbor_elev_mean", "neighbor_elev_min",
                                 "neighbor_elev_max", "neighbor_elev_std"]
    result = merged[["node_idx"] + all_feat_cols].copy()
    for c in all_feat_cols:
        result[c] = result[c].astype(np.float32)
    return result


# ============================================================
# 4. Rainfall Feature Precomputation
# ============================================================
def precompute_rainfall_features(rain_series):
    n = len(rain_series)
    rain = rain_series.astype(np.float32)

    feats = {}
    feats["rain_current"] = rain.copy()
    feats["rain_cumulative"] = np.cumsum(rain).astype(np.float32)

    cs = np.cumsum(rain)
    for w in [3, 6, 12, 24, 48]:
        rolled = np.zeros(n, dtype=np.float32)
        if w < n:
            rolled[w:] = cs[w:] - cs[:n - w]
            rolled[:w] = cs[:w]
        else:
            rolled[:] = cs
        feats[f"rain_roll_{w}"] = rolled

    rain_peak = np.zeros(n, dtype=np.float32)
    rain_time_since_peak = np.zeros(n, dtype=np.float32)
    cur_peak = 0.0
    peak_time = 0
    for i in range(n):
        if rain[i] >= cur_peak:
            cur_peak = rain[i]
            peak_time = i
        rain_peak[i] = cur_peak
        rain_time_since_peak[i] = i - peak_time
    feats["rain_peak"] = rain_peak
    feats["rain_time_since_peak"] = rain_time_since_peak

    deriv = np.zeros(n, dtype=np.float32)
    deriv[1:] = rain[1:] - rain[:-1]
    feats["rain_derivative"] = deriv

    rain_rev_cs = np.cumsum(rain[::-1])[::-1].astype(np.float32)
    for w in [6, 12, 24, 48]:
        future = np.zeros(n, dtype=np.float32)
        for i in range(n):
            end = min(i + w, n)
            future[i] = rain_rev_cs[i] - (rain_rev_cs[end] if end < n else 0.0)
        feats[f"rain_future_{w}"] = future

    feats["rain_is_zero"] = (rain == 0).astype(np.float32)
    return feats


def get_feature_names(model_id, node_type, rain_feat_names, static_cols):
    names = ["wl_current", "wl_above_elev", "wl_delta_prev", "wl_delta_prev2", "wl_change_from_warmup"]
    names += ["neighbor_wl_mean", "neighbor_wl_diff"]
    names += ["warmup_last", "warmup_mean", "warmup_std", "warmup_min", "warmup_max", "warmup_range"]
    names += rain_feat_names
    names += ["timestep", "timestep_ratio", "time_since_warmup"]
    names += static_cols
    if node_type == 1:
        names += ["inlet_flow"]
    elif use_conn_features(model_id, node_type):
        names += ["conn1d_wl_current", "conn1d_delta_prev", "conn1d_gap", "conn1d_exists"]
    return names


# ============================================================
# 5. Vectorized Feature Building (Autoregressive)
# ============================================================
def _prepare_event_data(model_id, event_id, node_type, node_feat_table, split):
    dim = "1d" if node_type == 1 else "2d"
    dyn = load_event_dynamic(model_id, event_id, split, dim)
    ts_df = load_timesteps(model_id, event_id, split)
    n_timesteps = len(ts_df)

    node_ids = sorted(dyn["node_idx"].unique())
    n_nodes = len(node_ids)

    dyn_sorted = dyn.sort_values(["timestep", "node_idx"])
    wl_matrix = dyn_sorted["water_level"].values.reshape(n_timesteps, n_nodes).astype(np.float32)

    if node_type == 2:
        rain_per_ts = dyn_sorted[dyn_sorted["node_idx"] == node_ids[0]].sort_values("timestep")["rainfall"].values.astype(np.float32)
    else:
        dyn_2d = load_event_dynamic(model_id, event_id, split, "2d")
        first_2d = sorted(dyn_2d["node_idx"].unique())[0]
        rain_per_ts = dyn_2d[dyn_2d["node_idx"] == first_2d].sort_values("timestep")["rainfall"].values.astype(np.float32)
        del dyn_2d

    inlet_matrix = None
    if node_type == 1 and "inlet_flow" in dyn.columns:
        inlet_matrix = dyn_sorted["inlet_flow"].values.reshape(n_timesteps, n_nodes).astype(np.float32)

    conn1d_wl_matrix = None
    conn1d_exists = None
    conn2d_to_1d = None
    if use_conn_features(model_id, node_type):
        conn_df = load_1d2d_connections(model_id, split)
        conn2d_to_1d = dict(zip(conn_df["node_2d"].astype(np.int32), conn_df["node_1d"].astype(np.int32)))

        dyn_1d = load_event_dynamic(model_id, event_id, split, "1d")
        node_ids_1d = sorted(dyn_1d["node_idx"].unique())
        dyn_1d_sorted = dyn_1d.sort_values(["timestep", "node_idx"])
        wl_matrix_1d = dyn_1d_sorted["water_level"].values.reshape(n_timesteps, len(node_ids_1d)).astype(np.float32)
        node1d_to_local = {nid: i for i, nid in enumerate(node_ids_1d)}

        conn1d_wl_matrix = np.zeros((n_timesteps, n_nodes), dtype=np.float32)
        conn1d_exists = np.zeros(n_nodes, dtype=np.float32)
        for i, nid2d in enumerate(node_ids):
            nid1d = conn2d_to_1d.get(nid2d)
            if nid1d is None:
                continue
            local_1d = node1d_to_local.get(nid1d)
            if local_1d is None:
                continue
            conn1d_wl_matrix[:, i] = wl_matrix_1d[:, local_1d]
            conn1d_exists[i] = 1.0

        del dyn_1d, dyn_1d_sorted, wl_matrix_1d

    del dyn, dyn_sorted
    gc.collect()

    rain_feats = precompute_rainfall_features(rain_per_ts[:n_timesteps])
    rain_feat_names = sorted(rain_feats.keys())
    rain_arrays = np.stack([rain_feats[k] for k in rain_feat_names], axis=1)

    nft = node_feat_table.set_index("node_idx")
    static_cols = list(nft.columns)
    static_matrix = np.zeros((n_nodes, len(static_cols)), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        if nid in nft.index:
            static_matrix[i] = nft.loc[nid, static_cols].values.astype(np.float32)

    if node_type == 2:
        elev_idx = static_cols.index("min_elevation")
    else:
        elev_idx = static_cols.index("invert_elevation")
    elevations = static_matrix[:, elev_idx]

    neighbor_idx, neighbor_mask, degree = build_adjacency(model_id, node_type, n_nodes, node_ids)

    return {
        "wl_matrix": wl_matrix,
        "rain_arrays": rain_arrays,
        "rain_feat_names": rain_feat_names,
        "inlet_matrix": inlet_matrix,
        "static_matrix": static_matrix,
        "static_cols": static_cols,
        "elevations": elevations,
        "node_ids": node_ids,
        "n_timesteps": n_timesteps,
        "n_nodes": n_nodes,
        "neighbor_idx": neighbor_idx,
        "neighbor_mask": neighbor_mask,
        "degree": degree,
        "conn1d_wl_matrix": conn1d_wl_matrix,
        "conn1d_exists": conn1d_exists,
        "conn2d_to_1d": conn2d_to_1d,
    }


def build_features_vectorized(model_id, event_id, node_type, node_feat_table, split="train", noise_std=0.0):
    d = _prepare_event_data(model_id, event_id, node_type, node_feat_table, split)
    wl_matrix_true = d["wl_matrix"]

    if noise_std > 0:
        rng = np.random.RandomState(event_id * 100 + node_type)
        noise = rng.normal(0, noise_std, wl_matrix_true.shape).astype(np.float32)
        noise[:WARMUP_STEPS] = 0
        wl_matrix = wl_matrix_true + noise
    else:
        wl_matrix = wl_matrix_true
    n_timesteps, n_nodes = d["n_timesteps"], d["n_nodes"]
    elevations = d["elevations"]
    rain_arrays = d["rain_arrays"]
    static_matrix = d["static_matrix"]
    inlet_matrix = d["inlet_matrix"]
    neighbor_idx = d["neighbor_idx"]
    neighbor_mask = d["neighbor_mask"]
    degree = d["degree"]
    conn1d_wl_matrix = d["conn1d_wl_matrix"]
    conn1d_exists = d["conn1d_exists"]

    warmup_end = min(WARMUP_STEPS, n_timesteps)
    warmup_wl = wl_matrix_true[:warmup_end]
    warmup_last = warmup_wl[-1]
    warmup_mean = warmup_wl.mean(axis=0)
    warmup_std = warmup_wl.std(axis=0)
    warmup_min_val = warmup_wl.min(axis=0)
    warmup_max_val = warmup_wl.max(axis=0)
    warmup_range = warmup_max_val - warmup_min_val

    n_rain = len(d["rain_feat_names"])
    n_static = len(d["static_cols"])
    extra_conn = 4 if use_conn_features(model_id, node_type) else 0
    n_feat = 5 + 2 + 6 + n_rain + 3 + n_static + (1 if node_type == 1 else extra_conn)

    # Align training targets with inference, which only starts after warmup.
    valid_ts = np.arange(max(warmup_end, 1), n_timesteps - 1, dtype=np.int32)
    n_valid = len(valid_ts)
    n_samples = n_valid * n_nodes

    X = np.zeros((n_samples, n_feat), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for ti, t in enumerate(valid_ts):
        sl = slice(ti * n_nodes, (ti + 1) * n_nodes)
        wl_curr = wl_matrix[t]
        wl_prev = wl_matrix[t - 1]
        delta_prev = wl_curr - wl_prev
        delta_prev2 = wl_prev - wl_matrix[t - 2] if t >= 2 else np.zeros(n_nodes, dtype=np.float32)

        n_wl_mean, n_wl_diff = compute_neighbor_wl(wl_curr, neighbor_idx, neighbor_mask, degree)

        fi = 0
        X[sl, fi] = wl_curr; fi += 1
        X[sl, fi] = wl_curr - elevations; fi += 1
        X[sl, fi] = delta_prev; fi += 1
        X[sl, fi] = delta_prev2; fi += 1
        X[sl, fi] = wl_curr - warmup_last; fi += 1
        X[sl, fi] = n_wl_mean; fi += 1
        X[sl, fi] = n_wl_diff; fi += 1
        X[sl, fi] = warmup_last; fi += 1
        X[sl, fi] = warmup_mean; fi += 1
        X[sl, fi] = warmup_std; fi += 1
        X[sl, fi] = warmup_min_val; fi += 1
        X[sl, fi] = warmup_max_val; fi += 1
        X[sl, fi] = warmup_range; fi += 1
        X[sl, fi:fi + n_rain] = rain_arrays[t][np.newaxis, :]
        fi += n_rain
        X[sl, fi] = t; fi += 1
        X[sl, fi] = t / max(n_timesteps - 1, 1); fi += 1
        X[sl, fi] = max(0, t - WARMUP_STEPS); fi += 1
        X[sl, fi:fi + n_static] = static_matrix
        fi += n_static
        if node_type == 1 and inlet_matrix is not None:
            X[sl, fi] = inlet_matrix[t]; fi += 1
        elif use_conn_features(model_id, node_type):
            conn_curr = conn1d_wl_matrix[t] if conn1d_wl_matrix is not None else np.zeros(n_nodes, dtype=np.float32)
            conn_prev = conn1d_wl_matrix[t - 1] if conn1d_wl_matrix is not None and t >= 1 else conn_curr
            X[sl, fi] = conn_curr; fi += 1
            X[sl, fi] = conn_curr - conn_prev; fi += 1
            X[sl, fi] = conn_curr - wl_curr; fi += 1
            X[sl, fi] = conn1d_exists if conn1d_exists is not None else 0.0; fi += 1

        y[sl] = wl_matrix_true[t + 1] - wl_matrix_true[t]

    return X, y, d["rain_feat_names"], d["static_cols"]


# ============================================================
# 6. Training
# ============================================================
def train_model(model_id, node_type, val_events=5):
    dim_str = "1D" if node_type == 1 else "2D"
    log(f"\n{'='*60}")
    log(f"Training Model {model_id} {dim_str} (Autoregressive v2h)")
    log(f"{'='*60}")

    lgb_params = LGB_PARAMS_M1.copy() if model_id == 1 else LGB_PARAMS_M2.copy()
    num_boost = NUM_BOOST_ROUND_M1 if model_id == 1 else NUM_BOOST_ROUND_M2
    log(f"  Hyperparams: lr={lgb_params['learning_rate']}, "
        f"leaves={lgb_params['num_leaves']}, boost={num_boost}, noise={NOISE_STD}")

    node_feat_table = build_node_feature_table(model_id, node_type)
    train_events = get_event_ids(model_id, "train")

    val_eids = train_events[-val_events:]
    train_eids = train_events[:-val_events]
    log(f"  Train events: {len(train_eids)}, Val events: {len(val_eids)}")
    log(f"  valid_ts starts at warmup_end={WARMUP_STEPS}")

    all_X, all_y = [], []
    feat_names = None
    total_samples = 0

    for i, eid in enumerate(train_eids):
        t0 = time.time()
        X, y, rain_fn, static_fn = build_features_vectorized(
            model_id, eid, node_type, node_feat_table, split="train", noise_std=NOISE_STD
        )
        if feat_names is None:
            feat_names = get_feature_names(model_id, node_type, rain_fn, static_fn)

        if node_type == 2 and SUBSAMPLE_2D < 1.0:
            n = len(X)
            rng = np.random.RandomState(eid)
            keep = rng.choice(n, size=int(n * SUBSAMPLE_2D), replace=False)
            X = X[keep]
            y = y[keep]

        all_X.append(X)
        all_y.append(y)
        total_samples += len(X)

        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0 or i == len(train_eids) - 1:
            log(f"  [{i+1}/{len(train_eids)}] event {eid}: {len(X):,} samples, {elapsed:.1f}s (total: {total_samples:,})")

        del X, y
        gc.collect()

    log(f"  Concatenating {total_samples:,} train samples...")
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    del all_X, all_y
    gc.collect()
    log(f"  Train array: {X_train.shape}, {X_train.nbytes / 1e9:.2f} GB")

    all_Xv, all_yv = [], []
    for eid in val_eids:
        X, y, _, _ = build_features_vectorized(model_id, eid, node_type, node_feat_table, split="train")
        if node_type == 2 and SUBSAMPLE_2D < 1.0:
            rng = np.random.RandomState(eid + 10000)
            keep = rng.choice(len(X), size=int(len(X) * SUBSAMPLE_2D), replace=False)
            X, y = X[keep], y[keep]
        all_Xv.append(X)
        all_yv.append(y)
    X_val = np.concatenate(all_Xv, axis=0)
    y_val = np.concatenate(all_yv, axis=0)
    del all_Xv, all_yv
    gc.collect()
    log(f"  Val array: {X_val.shape}")

    delta_clip_low = float(np.percentile(y_train, 0.01))
    delta_clip_high = float(np.percentile(y_train, 99.99))
    log(f"  Delta clip: [{delta_clip_low:.6f}, {delta_clip_high:.6f}]")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, free_raw_data=True)
    del X_train, y_train, X_val, y_val
    gc.collect()

    log(f"  Training LightGBM ({num_boost} rounds)...")
    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=True), lgb.log_evaluation(200)]
    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=num_boost,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )

    log(f"  Best iteration: {model.best_iteration}")
    log(f"  Best val RMSE: {model.best_score['val']['rmse']:.6f}")

    imp = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    log("  Top 10 features:")
    for _, row in imp_df.head(10).iterrows():
        log(f"    {row['feature']:30s} {row['importance']:.0f}")

    del dtrain, dval
    gc.collect()

    return model, node_feat_table, feat_names, (delta_clip_low, delta_clip_high)


# ============================================================
# 7. Autoregressive Inference
# ============================================================
def predict_event(model_id, event_id, node_type, lgb_model, node_feat_table,
                  feat_names, delta_clip, connected_1d_predictions=None):
    d = _prepare_event_data(model_id, event_id, node_type, node_feat_table, "test")
    wl_matrix = d["wl_matrix"]
    n_timesteps, n_nodes = d["n_timesteps"], d["n_nodes"]
    elevations = d["elevations"]
    rain_arrays = d["rain_arrays"]
    static_matrix = d["static_matrix"]
    inlet_matrix = d["inlet_matrix"]
    node_ids = d["node_ids"]
    neighbor_idx = d["neighbor_idx"]
    neighbor_mask = d["neighbor_mask"]
    degree = d["degree"]
    conn1d_wl_matrix = d["conn1d_wl_matrix"]
    conn1d_exists = d["conn1d_exists"]
    conn2d_to_1d = d["conn2d_to_1d"]

    if inlet_matrix is not None:
        inlet_matrix = np.nan_to_num(inlet_matrix, nan=0.0)

    warmup_wl = wl_matrix[:WARMUP_STEPS]
    warmup_last = warmup_wl[-1]
    warmup_mean = warmup_wl.mean(axis=0)
    warmup_std = warmup_wl.std(axis=0)
    warmup_min_val = warmup_wl.min(axis=0)
    warmup_max_val = warmup_wl.max(axis=0)
    warmup_range = warmup_max_val - warmup_min_val

    n_rain = len(d["rain_feat_names"])
    n_static = len(d["static_cols"])
    extra_conn = 4 if use_conn_features(model_id, node_type) else 0
    n_feat = 5 + 2 + 6 + n_rain + 3 + n_static + (1 if node_type == 1 else extra_conn)

    delta_clip_low, delta_clip_high = delta_clip
    pred_wl = wl_matrix.copy()

    if use_conn_features(model_id, node_type) and conn1d_wl_matrix is not None and connected_1d_predictions is not None:
        for i, nid2d in enumerate(node_ids):
            nid1d = conn2d_to_1d.get(nid2d) if conn2d_to_1d is not None else None
            if nid1d is None:
                continue
            pred_arr = connected_1d_predictions.get(nid1d)
            if pred_arr is None:
                continue
            conn1d_wl_matrix[WARMUP_STEPS:, i] = pred_arr[: max(0, n_timesteps - WARMUP_STEPS)]

    prev_delta = np.zeros(n_nodes, dtype=np.float32)
    prev_delta2 = np.zeros(n_nodes, dtype=np.float32)
    if WARMUP_STEPS >= 2:
        prev_delta = wl_matrix[WARMUP_STEPS - 1] - wl_matrix[WARMUP_STEPS - 2]
    if WARMUP_STEPS >= 3:
        prev_delta2 = wl_matrix[WARMUP_STEPS - 2] - wl_matrix[WARMUP_STEPS - 3]

    X_t = np.zeros((n_nodes, n_feat), dtype=np.float32)

    for t in range(WARMUP_STEPS, n_timesteps):
        wl_curr = pred_wl[t - 1]
        n_wl_mean, n_wl_diff = compute_neighbor_wl(wl_curr, neighbor_idx, neighbor_mask, degree)

        fi = 0
        X_t[:, fi] = wl_curr; fi += 1
        X_t[:, fi] = wl_curr - elevations; fi += 1
        X_t[:, fi] = prev_delta; fi += 1
        X_t[:, fi] = prev_delta2; fi += 1
        X_t[:, fi] = wl_curr - warmup_last; fi += 1
        X_t[:, fi] = n_wl_mean; fi += 1
        X_t[:, fi] = n_wl_diff; fi += 1
        X_t[:, fi] = warmup_last; fi += 1
        X_t[:, fi] = warmup_mean; fi += 1
        X_t[:, fi] = warmup_std; fi += 1
        X_t[:, fi] = warmup_min_val; fi += 1
        X_t[:, fi] = warmup_max_val; fi += 1
        X_t[:, fi] = warmup_range; fi += 1
        X_t[:, fi:fi + n_rain] = rain_arrays[t - 1][np.newaxis, :]
        fi += n_rain
        X_t[:, fi] = t - 1; fi += 1
        X_t[:, fi] = (t - 1) / max(n_timesteps - 1, 1); fi += 1
        X_t[:, fi] = max(0, t - 1 - WARMUP_STEPS); fi += 1
        X_t[:, fi:fi + n_static] = static_matrix
        fi += n_static
        if node_type == 1 and inlet_matrix is not None:
            X_t[:, fi] = inlet_matrix[t - 1]; fi += 1
        elif use_conn_features(model_id, node_type):
            conn_curr = conn1d_wl_matrix[t - 1] if conn1d_wl_matrix is not None else np.zeros(n_nodes, dtype=np.float32)
            conn_prev = conn1d_wl_matrix[t - 2] if conn1d_wl_matrix is not None and t >= 2 else conn_curr
            X_t[:, fi] = conn_curr; fi += 1
            X_t[:, fi] = conn_curr - conn_prev; fi += 1
            X_t[:, fi] = conn_curr - wl_curr; fi += 1
            X_t[:, fi] = conn1d_exists if conn1d_exists is not None else 0.0; fi += 1

        delta_pred = lgb_model.predict(X_t, num_iteration=lgb_model.best_iteration).astype(np.float32)
        delta_pred = np.clip(delta_pred, delta_clip_low, delta_clip_high)
        new_wl = np.maximum(wl_curr + delta_pred, elevations)
        pred_wl[t] = new_wl

        prev_delta2 = prev_delta.copy()
        prev_delta = delta_pred.copy()

    predictions = {}
    for i, nid in enumerate(node_ids):
        predictions[nid] = pred_wl[WARMUP_STEPS:, i].copy()

    return predictions


# ============================================================
# 8. Submission Generation
# ============================================================
def generate_submission(all_predictions):
    log(f"\n{'='*60}")
    log("Generating submission.csv")
    log(f"{'='*60}")

    sub = pd.read_parquet(SUBMISSION_PATH)
    log(f"  Sample submission shape: {sub.shape}")

    water_levels = np.empty(len(sub), dtype=np.float32)
    water_levels[:] = np.nan

    grouped = sub.groupby(["model_id", "event_id", "node_type"], sort=False)
    filled = 0

    for (mid, eid, nt), group_df in grouped:
        key = (mid, eid, nt)
        if key not in all_predictions:
            log(f"  WARNING: No predictions for M{mid} E{eid} NT{nt}")
            continue

        preds = all_predictions[key]
        indices = group_df.index.values
        node_id_arr = group_df["node_id"].values
        unique_nodes = np.unique(node_id_arr)

        for nid in unique_nodes:
            if nid not in preds:
                continue
            node_mask = node_id_arr == nid
            node_indices = indices[node_mask]
            pred_arr = preds[nid]
            n = min(len(node_indices), len(pred_arr))
            water_levels[node_indices[:n]] = pred_arr[:n]
            filled += n

    sub["water_level"] = water_levels
    n_nan = np.isnan(water_levels).sum()
    if n_nan > 0:
        log(f"  WARNING: {n_nan} NaN values remaining")
        nan_indices = np.where(np.isnan(water_levels))[0]
        nan_rows = sub.iloc[nan_indices]
        for (mid, eid, nt), grp in nan_rows.groupby(["model_id", "event_id", "node_type"]):
            dim = "1d" if nt == 1 else "2d"
            try:
                dyn = load_event_dynamic(mid, eid, "test", dim)
                warmup_data = dyn[dyn["timestep"] == WARMUP_STEPS - 1]
                wl_map = dict(zip(warmup_data["node_idx"].values, warmup_data["water_level"].values))
                for nid, ngrp in grp.groupby("node_id"):
                    fill_val = wl_map.get(nid, 0.0)
                    water_levels[ngrp.index.values] = fill_val
            except Exception as e:
                log(f"    Fallback for M{mid} E{eid} NT{nt}: {e}")
        np.nan_to_num(water_levels, copy=False, nan=0.0)
        sub["water_level"] = water_levels

    log(f"  Filled {filled:,} / {len(sub):,} rows")
    sub[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]].to_csv(OUTPUT_PATH, index=False)
    log(f"  Saved to {OUTPUT_PATH}")
    log(f"  WL stats: min={sub['water_level'].min():.2f}, "
        f"max={sub['water_level'].max():.2f}, mean={sub['water_level'].mean():.2f}")


# ============================================================
# 9. Main Pipeline
# ============================================================
def main():
    t_start = time.time()
    log("UrbanFloodBench Autoregressive LightGBM v2h")
    log("=" * 60)

    all_predictions = {}

    for model_id in MODELS:
        for node_type in NODE_TYPES:
            dim_str = "1D" if node_type == 1 else "2D"
            key = f"M{model_id}_{dim_str}"
            t_model = time.time()

            lgb_model, node_feat_table, feat_names, delta_clip = train_model(
                model_id, node_type, val_events=5
            )
            gc.collect()

            test_events = get_event_ids(model_id, "test")
            log(f"\n  Predicting {len(test_events)} test events for {key}...")

            for i, eid in enumerate(test_events):
                t0 = time.time()
                connected_1d_predictions = None
                if use_conn_features(model_id, node_type):
                    connected_1d_predictions = all_predictions.get((model_id, eid, 1))
                preds = predict_event(
                    model_id, eid, node_type, lgb_model, node_feat_table,
                    feat_names, delta_clip, connected_1d_predictions=connected_1d_predictions
                )
                all_predictions[(model_id, eid, node_type)] = preds
                elapsed = time.time() - t0

                if (i + 1) % 5 == 0 or i == 0 or i == len(test_events) - 1:
                    log(f"    [{i+1}/{len(test_events)}] event {eid}: {elapsed:.1f}s")

            del lgb_model, node_feat_table
            gc.collect()

            log(f"  {key} total time: {(time.time() - t_model) / 60:.1f} min")

    generate_submission(all_predictions)

    total_time = time.time() - t_start
    log(f"\nTotal time: {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
