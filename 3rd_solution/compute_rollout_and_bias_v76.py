"""Generate train rollout residuals + 10-zone bias for v76 model.

Combined script: rollout all train events -> compute bias -> save pickle.
"""
import os, sys, pickle, time
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import (
    load_model_config, load_event_data, build_graph_at_timestep, list_events,
)
from src.model import HeteroFloodGNNv11
from run_train_m2_v11c_multistep import (
    mask_2d_edges_only, MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM,
)

device = "cuda"
data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
cache_dir = os.path.join(BASE, "Models", "checkpoints")


def rollout_event(model, event, config, norm_stats, per_node_stats, spin_up=10):
    """Run AR rollout on one event, return (pred_1d, pred_2d, gt_1d, gt_2d, res_1d, res_2d)."""
    max_t = event.nodes_1d_dynamic.shape[0] - 1
    n_1d = event.nodes_1d_dynamic.shape[1]
    n_2d = event.nodes_2d_dynamic.shape[1]

    pn_std_1d = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_2d = torch.tensor(per_node_stats["2d_wl_std"], dtype=torch.float32, device=device)

    dyn_1d = event.nodes_1d_dynamic.copy()
    dyn_2d = event.nodes_2d_dynamic.copy()
    dyn_e1d = event.edges_1d_dynamic.copy()
    orig_1d = event.nodes_1d_dynamic
    orig_2d = event.nodes_2d_dynamic
    event.nodes_1d_dynamic = dyn_1d
    event.nodes_2d_dynamic = dyn_2d
    event.edges_1d_dynamic = dyn_e1d
    dyn_2d[:, :, 2] = 0.0

    cur_1d = torch.tensor(dyn_1d[spin_up, :, 0], dtype=torch.float32, device=device)
    cur_2d = torch.tensor(dyn_2d[spin_up, :, 1], dtype=torch.float32, device=device)

    T = max_t - spin_up
    res_1d = np.zeros((T, n_1d), dtype=np.float32)
    res_2d = np.zeros((T, n_2d), dtype=np.float32)

    with torch.no_grad():
        for step in range(T):
            t = spin_up + step
            if t + 1 > max_t:
                break

            if step > 0:
                dyn_1d[t, :, 0] = cur_1d.cpu().numpy()
                dyn_2d[t, :, 1] = cur_2d.cpu().numpy()
            mask_2d_edges_only(event, t)
            if t > 0:
                mask_2d_edges_only(event, t - 1)

            g = build_graph_at_timestep(
                config, event, t, prev_t=t-1, norm_stats=norm_stats,
                future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                per_node_stats=per_node_stats,
            ).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(g)

            pred_1d = out["1d"].float()
            delta_1d = pred_1d[:, 0] * pn_std_1d
            cur_1d = cur_1d + delta_1d

            pred_2d = out["2d"].float()
            delta_2d = pred_2d[:, 0] * pn_std_2d
            cur_2d = cur_2d + delta_2d

            # Aux feedback
            dyn_1d[t+1, :, 1] = pred_1d[:, 1].cpu().numpy()
            if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                ep_out = out["1d_edge"].float().cpu().numpy()
                dyn_e1d[t+1, :, 0] = ep_out[:, 0]
                dyn_e1d[t+1, :, 1] = ep_out[:, 1]

            # Residual = prediction - GT
            gt_1d = orig_1d[t+1, :, 0]
            gt_2d = orig_2d[t+1, :, 1]
            res_1d[step] = cur_1d.cpu().numpy() - gt_1d
            res_2d[step] = cur_2d.cpu().numpy() - gt_2d

    event.nodes_1d_dynamic = orig_1d
    event.nodes_2d_dynamic = orig_2d
    event.edges_1d_dynamic = event.edges_1d_dynamic  # already original

    return res_1d[:T], res_2d[:T]


def compute_zone_bias(res_list, n_zones=10):
    """Compute per-node zone bias from residual lists."""
    max_T = max(r.shape[0] for r in res_list)
    n_nodes = res_list[0].shape[1]
    zone_bounds = []
    zone_size = max_T / n_zones
    for z in range(n_zones):
        lo = int(z * zone_size)
        hi = int((z + 1) * zone_size) if z < n_zones - 1 else 9999
        zone_bounds.append((lo, hi))

    # Accumulate mean residual per node per zone
    zone_sum = np.zeros((n_nodes, n_zones), dtype=np.float64)
    zone_cnt = np.zeros((n_nodes, n_zones), dtype=np.float64)

    for res in res_list:
        T = res.shape[0]
        for zi, (lo, hi) in enumerate(zone_bounds):
            actual_hi = min(hi, T)
            if lo >= T:
                continue
            zone_sum[:, zi] += res[lo:actual_hi].sum(axis=0)
            zone_cnt[:, zi] += actual_hi - lo

    zone_mean = np.zeros_like(zone_sum)
    mask = zone_cnt > 0
    zone_mean[mask] = zone_sum[mask] / zone_cnt[mask]

    return zone_mean, zone_bounds


def main():
    print("Loading model and data...")
    ckpt_path = os.path.join(cache_dir, "best_model_2_v76_dw_r64.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cache_dir, "best_model_2_v76_aligned_r400.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=ckpt.get("coupling_edge_dim", COUPLING_EDGE_DIM),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded model: val={ckpt.get('best_val', 'N/A')}")

    config = load_model_config(data_dir, MODEL_ID)
    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    all_events = list_events(data_dir, MODEL_ID, "train")
    train_ids = all_events.tolist() if hasattr(all_events, 'tolist') else list(all_events)
    print(f"  Events: {len(train_ids)}")

    # Rollout all events
    res_1d_list = []
    res_2d_list = []
    t0 = time.time()
    for i, eid in enumerate(train_ids):
        event = load_event_data(data_dir, MODEL_ID, eid, config)
        r1, r2 = rollout_event(model, event, config, norm_stats, per_node_stats)
        res_1d_list.append(r1)
        res_2d_list.append(r2)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(train_ids)} events ({time.time()-t0:.1f}s)")

    print(f"  All events done in {time.time()-t0:.1f}s")

    # Save rollout
    rollout_path = os.path.join(cache_dir, "train_rollout_v76.pkl")
    with open(rollout_path, "wb") as f:
        pickle.dump({"res_1d": res_1d_list, "res_2d": res_2d_list}, f)
    print(f"  Saved rollout: {rollout_path}")

    # Compute 10-zone bias
    print("\nComputing 10-zone bias...")
    zone_mean_1d, zone_bounds = compute_zone_bias(res_1d_list, n_zones=10)
    zone_mean_2d, _ = compute_zone_bias(res_2d_list, n_zones=10)

    bias_stats = {
        "zone_bias_1d": {zi: zone_mean_1d[:, zi] for zi in range(10)},
        "zone_bias_2d": {zi: zone_mean_2d[:, zi] for zi in range(10)},
        "zone_bounds": zone_bounds,
    }

    bias_path = os.path.join(cache_dir, "bias_stats_v76_10zone.pkl")
    with open(bias_path, "wb") as f:
        pickle.dump(bias_stats, f)
    print(f"  Saved bias: {bias_path}")

    # Print summary
    for zi, (lo, hi) in enumerate(zone_bounds):
        b1d = zone_mean_1d[:, zi]
        b2d = zone_mean_2d[:, zi]
        print(f"  Zone {zi} [{lo:3d}-{hi:4d}]: 1d_bias_mean={b1d.mean():.5f} (|max|={np.abs(b1d).max():.5f}), "
              f"2d_bias_mean={b2d.mean():.5f} (|max|={np.abs(b2d).max():.5f})")


if __name__ == "__main__":
    main()
