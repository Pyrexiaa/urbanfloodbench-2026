"""M1 v35: M2手法のM1移植 — gradient chain + node-weighted loss + separate TBPTT.

M1 v4cb (val=0.0113) からのfine-tune。
M2 v30/v32cで検証済みの手法をM1に移植:
  - 1D gradient chain (k_1d=32)
  - 2D gradient chain (k_2d=8) with separate TBPTT
  - Node-weighted loss (σ_i / mean(σ))
  - Trajectory loss (w_traj=0.3, w_traj_2d=0.1)
  - Random start positions

M1: 17 1D nodes, 3716 2D nodes
"""
import sys
import os
import time
import pickle
import argparse
import gc
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import (
    load_model_config, load_event_data, build_graph_at_timestep, list_events,
)
from src.model import HeteroFloodGNNv4
from src.evaluation import compute_std_from_all_events, compute_standardized_rmse

MODEL_ID = 1
FUTURE_RAIN_STEPS = 10
COUPLING_EDGE_DIM = 2

# Feature column indices (per_node_stats + future_rain_steps=10 mode)
# 1D: [static(6), wl_t(1), inlet_t(1), wl_prev(1), inlet_prev(1), fr_max(1), fr_sum(1), rem_sum(1)] = 13
WL_1D_T_COL = 6
WL_1D_PREV_COL = 8
# 2D: [static(9), rain_t(1), wl_t(1), vol_t(1), rain_prev(1), wl_prev(1), vol_prev(1), fr_max(1), fr_sum(1), rem_sum(1)] = 18
WL_2D_T_COL = 10
WL_2D_PREV_COL = 13


def mask_test_unavailable_features(event, t: int):
    """テストで利用不可な特徴量をゼロ化 (M1: inlet + volume + edges)."""
    event.nodes_1d_dynamic[t, :, 1] = 0.0   # inlet_flow
    event.nodes_2d_dynamic[t, :, 2] = 0.0   # water_volume
    if t < event.edges_1d_dynamic.shape[0]:
        event.edges_1d_dynamic[t, :, :] = 0.0
    if t < event.edges_2d_dynamic.shape[0]:
        event.edges_2d_dynamic[t, :, :] = 0.0


def compute_node_weights(
    per_node_stats: dict, w_min: float = 0.1, w_max: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node loss weights: σ_i / mean(σ), clipped."""
    std_1d = per_node_stats["1d_wl_std"]
    std_2d = per_node_stats["2d_wl_std"]
    w_1d = np.clip(std_1d / std_1d.mean(), w_min, w_max)
    w_2d = np.clip(std_2d / std_2d.mean(), w_min, w_max)
    print(f"  Node weights 1D: mean={w_1d.mean():.3f}, min={w_1d.min():.3f}, max={w_1d.max():.3f}")
    print(f"  Node weights 2D: mean={w_2d.mean():.3f}, min={w_2d.min():.3f}, max={w_2d.max():.3f}")
    return w_1d, w_2d


def validate_full_rollout(model, val_events, config, norm_stats, std_1d, std_2d,
                          device, per_node_stats=None):
    """全ステップ autoregressive validation。per-node delta逆変換。"""
    model.eval()
    spin_up = 10
    scores, s1d, s2d = [], [], []
    pn_std_1d = per_node_stats["1d_wl_std"] if per_node_stats else None
    pn_std_2d = per_node_stats["2d_wl_std"] if per_node_stats else None

    with torch.no_grad():
        for event in val_events:
            max_t = event.nodes_1d_dynamic.shape[0]
            if max_t <= spin_up + 2:
                continue
            dyn_1d = event.nodes_1d_dynamic.copy()
            dyn_2d = event.nodes_2d_dynamic.copy()
            dyn_e1d = event.edges_1d_dynamic.copy()
            dyn_e2d = event.edges_2d_dynamic.copy()
            orig_1d, orig_2d = event.nodes_1d_dynamic, event.nodes_2d_dynamic
            orig_e1d, orig_e2d = event.edges_1d_dynamic, event.edges_2d_dynamic
            event.nodes_1d_dynamic = dyn_1d
            event.nodes_2d_dynamic = dyn_2d
            event.edges_1d_dynamic = dyn_e1d
            event.edges_2d_dynamic = dyn_e2d

            cur_1d = dyn_1d[spin_up, :, 0].copy()
            cur_2d = dyn_2d[spin_up, :, 1].copy()
            p1, p2, t1, t2 = [], [], [], []

            for t in range(spin_up, max_t - 1):
                if t > spin_up:
                    dyn_1d[t, :, 0] = cur_1d
                    dyn_2d[t, :, 1] = cur_2d
                mask_test_unavailable_features(event, t)
                if t > 0:
                    mask_test_unavailable_features(event, t - 1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    g = build_graph_at_timestep(
                        config, event, t, prev_t=t-1, norm_stats=norm_stats,
                        future_rain_steps=FUTURE_RAIN_STEPS,
                        coupling_features=True,
                        per_node_stats=per_node_stats,
                    ).to(device)
                    out = model(g)

                d1 = out["1d"].squeeze().float().cpu().numpy()
                d2 = out["2d"].squeeze().float().cpu().numpy()
                if pn_std_1d is not None:
                    d1 = d1 * pn_std_1d
                if pn_std_2d is not None:
                    d2 = d2 * pn_std_2d
                cur_1d = cur_1d + d1
                cur_2d = cur_2d + d2
                p1.append(cur_1d.copy())
                p2.append(cur_2d.copy())
                t1.append(orig_1d[t + 1, :, 0])
                t2.append(orig_2d[t + 1, :, 1])

            event.nodes_1d_dynamic = orig_1d
            event.nodes_2d_dynamic = orig_2d
            event.edges_1d_dynamic = orig_e1d
            event.edges_2d_dynamic = orig_e2d

            if p1:
                s_1d = compute_standardized_rmse(np.stack(p1), np.stack(t1), std_1d)
                s_2d = compute_standardized_rmse(np.stack(p2), np.stack(t2), std_2d)
                scores.append((s_1d + s_2d) / 2)
                s1d.append(s_1d)
                s2d.append(s_2d)

    return (float(np.mean(scores)) if scores else float("inf"),
            float(np.mean(s1d)) if s1d else float("inf"),
            float(np.mean(s2d)) if s2d else float("inf"))


def train_phase_v35(
    model, train_events, val_events, config, norm_stats, std_1d, std_2d,
    device, cache_dir, rollout, lr, epochs, tag, best_val,
    per_node_stats, node_weight_1d, node_weight_2d,
    w_1d_scale=3.0, w_traj=0.3, w_traj_2d=0.1,
    k_1d=32, k_2d=8, spin_up=10,
):
    """M1 v35 training with gradient chain + separate TBPTT + node-weighted loss."""
    pn_mean_1d = torch.tensor(per_node_stats["1d_wl_mean"], dtype=torch.float32, device=device)
    pn_std_1d_t = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_1d_np = per_node_stats["1d_wl_std"]
    pn_mean_2d = torch.tensor(per_node_stats["2d_wl_mean"], dtype=torch.float32, device=device)
    pn_std_2d_t = torch.tensor(per_node_stats["2d_wl_std"], dtype=torch.float32, device=device)
    pn_std_2d_np = per_node_stats["2d_wl_std"]

    nw_1d = torch.tensor(node_weight_1d, dtype=torch.float32, device=device)
    nw_2d = torch.tensor(node_weight_2d, dtype=torch.float32, device=device)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    eligible = [ev for ev in train_events if ev.nodes_1d_dynamic.shape[0] - 1 >= rollout + spin_up + 1]
    print(f"    Eligible: {len(eligible)}/{len(train_events)} (need {rollout + spin_up + 1} steps)")
    print(f"    k_1d={k_1d}, k_2d={k_2d}, rollout={rollout}")

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_traj, n_steps = 0.0, 0.0, 0
        start_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        t0 = time.time()

        for idx in np.random.permutation(len(eligible)):
            event = eligible[idx]
            max_t = event.nodes_1d_dynamic.shape[0] - 1

            # Random start position
            T = max_t
            candidates = [spin_up]
            for frac in [0.25, 0.5, 0.75]:
                pos = int(T * frac)
                if pos + rollout <= max_t and pos >= spin_up:
                    candidates.append(pos)
            start_t = candidates[np.random.randint(len(candidates))]
            quartile = candidates.index(start_t)
            start_dist[min(quartile, 3)] += 1

            optimizer.zero_grad()

            # Copy dynamic data (for restoration after training)
            dyn_1d = event.nodes_1d_dynamic.copy()
            dyn_2d = event.nodes_2d_dynamic.copy()
            dyn_e1d = event.edges_1d_dynamic.copy()
            dyn_e2d = event.edges_2d_dynamic.copy()
            orig_1d = event.nodes_1d_dynamic
            orig_2d = event.nodes_2d_dynamic
            orig_e1d = event.edges_1d_dynamic
            orig_e2d = event.edges_2d_dynamic
            event.nodes_1d_dynamic = dyn_1d
            event.nodes_2d_dynamic = dyn_2d
            event.edges_1d_dynamic = dyn_e1d
            event.edges_2d_dynamic = dyn_e2d
            # water_volume全体ゼロ化 (test条件模倣)
            dyn_2d[:, :, 2] = 0.0

            # Differentiable WL state
            cur_1d = torch.tensor(dyn_1d[start_t, :, 0], dtype=torch.float32, device=device)
            prev_1d = cur_1d.clone()
            cur_2d = torch.tensor(dyn_2d[start_t, :, 1], dtype=torch.float32, device=device)
            prev_2d = cur_2d.clone()

            acc_loss = 0.0
            step_total, step_traj = 0.0, 0.0
            steps_since_1d_detach = 0
            steps_since_2d_detach = 0

            for step in range(rollout):
                t = start_t + step
                if t + 1 > max_t:
                    break

                # Pushforward: update dynamic arrays with predictions
                if step > 0:
                    dyn_1d[t, :, 0] = cur_1d.detach().cpu().numpy()
                    dyn_2d[t, :, 1] = cur_2d.detach().cpu().numpy()

                # Mask test-unavailable features
                mask_test_unavailable_features(event, t)
                if t > 0:
                    mask_test_unavailable_features(event, t - 1)

                g = build_graph_at_timestep(
                    config, event, t, prev_t=t-1, norm_stats=norm_stats,
                    future_rain_steps=FUTURE_RAIN_STEPS,
                    coupling_features=True,
                    per_node_stats=per_node_stats,
                ).to(device)

                # --- 1D gradient chain ---
                if step >= 1 and steps_since_1d_detach > 0:
                    x = g["1d"].x
                    wl_t_n = (cur_1d - pn_mean_1d) / pn_std_1d_t
                    wl_p_n = (prev_1d - pn_mean_1d) / pn_std_1d_t
                    g["1d"].x = torch.cat([
                        x[:, :WL_1D_T_COL],
                        wl_t_n.unsqueeze(1),
                        x[:, WL_1D_T_COL+1:WL_1D_PREV_COL],
                        wl_p_n.unsqueeze(1),
                        x[:, WL_1D_PREV_COL+1:],
                    ], dim=1)

                # --- 2D gradient chain ---
                if step >= 1 and steps_since_2d_detach > 0:
                    x2 = g["2d"].x
                    wl_2d_t_n = (cur_2d - pn_mean_2d) / pn_std_2d_t
                    wl_2d_p_n = (prev_2d - pn_mean_2d) / pn_std_2d_t
                    g["2d"].x = torch.cat([
                        x2[:, :WL_2D_T_COL],
                        wl_2d_t_n.unsqueeze(1),
                        x2[:, WL_2D_T_COL+1:WL_2D_PREV_COL],
                        wl_2d_p_n.unsqueeze(1),
                        x2[:, WL_2D_PREV_COL+1:],
                    ], dim=1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(g)

                # Model output: delta WL (per-node normalized)
                pred_1d = out["1d"].squeeze().float()  # [N_1d]
                delta_raw_1d = pred_1d * pn_std_1d_t
                prev_1d = cur_1d
                cur_1d = cur_1d + delta_raw_1d

                pred_2d = out["2d"].squeeze().float()  # [N_2d]
                delta_raw_2d = pred_2d * pn_std_2d_t
                prev_2d = cur_2d
                cur_2d = cur_2d + delta_raw_2d
                steps_since_1d_detach += 1
                steps_since_2d_detach += 1

                # --- Node-weighted delta loss ---
                gt_d1 = torch.tensor(
                    (orig_1d[t+1, :, 0] - orig_1d[t, :, 0]) / pn_std_1d_np,
                    dtype=torch.float32, device=device)
                gt_d2 = torch.tensor(
                    (orig_2d[t+1, :, 1] - orig_2d[t, :, 1]) / pn_std_2d_np,
                    dtype=torch.float32, device=device)

                l_d1 = (nw_1d * (pred_1d - gt_d1).pow(2)).mean()
                l_d2 = (nw_2d * (pred_2d - gt_d2).pow(2)).mean()

                # --- Node-weighted trajectory loss ---
                gt_wl_1d = torch.tensor(orig_1d[t+1, :, 0], dtype=torch.float32, device=device)
                l_traj_1d = (nw_1d * ((cur_1d - gt_wl_1d) / pn_std_1d_t).pow(2)).mean()

                gt_wl_2d = torch.tensor(orig_2d[t+1, :, 1], dtype=torch.float32, device=device)
                l_traj_2d = (nw_2d * ((cur_2d - gt_wl_2d) / pn_std_2d_t).pow(2)).mean()

                s_loss = (w_1d_scale * l_d1 + l_d2
                          + w_traj * l_traj_1d + w_traj_2d * l_traj_2d) / rollout
                acc_loss = acc_loss + s_loss
                step_total += s_loss.item()
                step_traj += (l_traj_1d.item() + l_traj_2d.item()) / rollout

                # === SEPARATE TBPTT ===

                # 2D-only detach: 2Dのみdetach、1Dは保持
                if steps_since_2d_detach >= k_2d and step < rollout - 1:
                    cur_2d = cur_2d.detach().clone()
                    prev_2d = prev_2d.detach().clone()
                    steps_since_2d_detach = 0

                # 1D TBPTT: backward + 全detach
                if steps_since_1d_detach >= k_1d and step < rollout - 1:
                    acc_loss.backward()
                    acc_loss = 0.0
                    cur_1d = cur_1d.detach().clone()
                    cur_1d.requires_grad = True
                    prev_1d = prev_1d.detach().clone()
                    cur_2d = cur_2d.detach().clone()
                    prev_2d = prev_2d.detach().clone()
                    steps_since_1d_detach = 0
                    steps_since_2d_detach = 0

            # Final backward
            if isinstance(acc_loss, torch.Tensor) and acc_loss.requires_grad:
                acc_loss.backward()

            # Restore original data
            event.nodes_1d_dynamic = orig_1d
            event.nodes_2d_dynamic = orig_2d
            event.edges_1d_dynamic = orig_e1d
            event.edges_2d_dynamic = orig_e2d

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ep_loss += step_total
            ep_traj += step_traj
            n_steps += 1

        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()
        avg_l = ep_loss / max(n_steps, 1)
        avg_t = ep_traj / max(n_steps, 1)

        val_str = ""
        if epoch % 2 == 0 or epoch == epochs - 1:
            vs, v1d, v2d = validate_full_rollout(
                model, val_events, config, norm_stats, std_1d, std_2d, device,
                per_node_stats=per_node_stats)
            val_str = f"val={vs:.4f} (1d={v1d:.4f} 2d={v2d:.4f})"
            if vs < best_val:
                best_val = vs
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "val_1d": v1d, "val_2d": v2d,
                    "model_version": "v4cb", "rollout": rollout,
                    "per_node_norm": True,
                    "future_rain_steps": FUTURE_RAIN_STEPS,
                    "coupling_edge_dim": COUPLING_EDGE_DIM,
                    "gradient_chain": True,
                    "separate_tbptt": True,
                    "node_weighted_loss": True,
                    "k_1d": k_1d, "k_2d": k_2d,
                }, os.path.join(cache_dir, f"best_model_1_{tag}.pt"))
                val_str += " *"

        lr_now = scheduler.get_last_lr()[0]
        dist_str = f"Q=[{start_dist[0]}/{start_dist[1]}/{start_dist[2]}/{start_dist[3]}]"
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  r{rollout:3d} {epoch:3d} | loss={avg_l:.6f} traj={avg_t:.4f}"
              f" | {val_str} | best={best_val:.4f}"
              f" | lr={lr_now:.2e} | {dist_str} | vram={vram:.1f}GB | {time.time()-t0:.1f}s")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k_1d", type=int, default=32)
    parser.add_argument("--k_2d", type=int, default=8)
    parser.add_argument("--tag", type=str, default="v35")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = "cuda"
    data_dir = "Dataset_Rerelease/Models"
    cache_dir = "Models/checkpoints"

    print("=" * 70)
    print(f"  M1 v35: Gradient Chain + Node-Weighted Loss + Sep TBPTT")
    print(f"  Base: v4cb_r32 (val=0.0113)")
    print(f"  k_1d={args.k_1d}, k_2d={args.k_2d}, lr={args.lr}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"M1: {config.num_1d_nodes} 1D, {config.num_2d_nodes} 2D nodes")
    print(f"Global std: 1D={std_1d:.4f}, 2D={std_2d:.4f}")

    # norm_stats
    cache_v2 = os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl")
    with open(cache_v2, "rb") as f:
        norm_stats = pickle.load(f)

    # Per-node stats
    pn_path = os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl")
    with open(pn_path, "rb") as f:
        per_node_stats = pickle.load(f)
    print(f"  1D WL per-node std: min={per_node_stats['1d_wl_std'].min():.4f}, "
          f"max={per_node_stats['1d_wl_std'].max():.4f}, "
          f"mean={per_node_stats['1d_wl_std'].mean():.4f}")
    print(f"  2D WL per-node std: min={per_node_stats['2d_wl_std'].min():.4f}, "
          f"max={per_node_stats['2d_wl_std'].max():.4f}, "
          f"mean={per_node_stats['2d_wl_std'].mean():.4f}")

    # Node weights
    nw_1d, nw_2d = compute_node_weights(per_node_stats)

    # Val split: 固定seed=42
    all_events = list_events(data_dir, MODEL_ID, "train")
    n_val = max(1, int(len(all_events) * 0.15))
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(all_events)
    val_ids, train_ids = shuffled[:n_val].tolist(), shuffled[n_val:].tolist()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    print("Loading events...")
    t0 = time.time()
    train_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in train_ids]
    val_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in val_ids]
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Model
    model = HeteroFloodGNNv4(
        hidden_dim=128, num_processor_layers=4, noise_std=0.02,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)
    print(f"v35 params: {model.get_model_size():,}")

    # Transfer from v4cb_r32
    src_ckpt = os.path.join(cache_dir, "best_model_1_v4cb_r32.pt")
    if os.path.exists(src_ckpt):
        ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        src_val = ckpt.get("best_val", "?")
        print(f"  Loaded: {os.path.basename(src_ckpt)} (val={src_val})")
    else:
        print("  WARNING: v4cb_r32 checkpoint not found!")

    # Initial validation
    v0, v0_1d, v0_2d = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats)
    print(f"Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    best_val = v0

    # Phase 1: r=32
    print(f"\n--- Phase 1: r=32, k_1d={args.k_1d}, k_2d={args.k_2d}, lr={args.lr}, {args.epochs}ep ---")
    best_val = train_phase_v35(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=32, lr=args.lr, epochs=args.epochs,
        tag=f"{args.tag}_r32", best_val=best_val,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        w_1d_scale=3.0, w_traj=0.3, w_traj_2d=0.1,
        k_1d=args.k_1d, k_2d=args.k_2d,
    )

    # Reload best Phase 1 checkpoint
    p1_ckpt = os.path.join(cache_dir, f"best_model_1_{args.tag}_r32.pt")
    if os.path.exists(p1_ckpt):
        ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Reloaded best Phase 1 checkpoint")

    # Phase 2: r=64
    lr2 = args.lr / 5
    print(f"\n--- Phase 2: r=64, k_1d={args.k_1d}, k_2d={args.k_2d}, lr={lr2:.1e}, 10ep ---")
    best_val = train_phase_v35(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=64, lr=lr2, epochs=10,
        tag=f"{args.tag}_r64", best_val=best_val,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        w_1d_scale=3.0, w_traj=0.3, w_traj_2d=0.1,
        k_1d=args.k_1d, k_2d=args.k_2d,
    )

    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"(Baseline: {v0:.4f})")


if __name__ == "__main__":
    main()
