"""Model_2 v11c Multi-Step Backprop: 1D gradient chain through rollout.

核心: 1D (198 nodes) の gradient を rollout chain 全体に流す。
build_graph_at_timestep で構築した numpy グラフの 1D WL columns を
differentiable tensor で置換し、cur_1d → model → delta → cur_1d の
gradient chain を形成。

Loss:
  - loss_delta: 各ステップの正規化delta MSE (standard)
  - loss_traj: 累積WL誤差の正規化MSE (multi-step gradient signal)

2D (4299 nodes) は従来通り detach。
Memory: truncated BPTT every K steps。

Usage: python run_train_m2_v11c_multistep.py [--seed 42] [--base v11c_r32]
"""
import sys
import os
import time
import pickle
import argparse
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import (
    load_model_config, load_event_data, build_graph_at_timestep, list_events,
    compute_normalization_stats,
)
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events, compute_standardized_rmse

MODEL_ID = 2
FUTURE_RAIN_STEPS = 10
COUPLING_EDGE_DIM = 2
AUX_REL_WEIGHT = 0.3

# 1D node feature column indices (after build_graph_at_timestep with per_node_stats)
# [static(6), wl_t(1), inlet_t(1), wl_prev(1), inlet_prev(1), future_rain(3)] = 13 cols
WL_T_COL = 6
WL_PREV_COL = 8


def compute_flux_and_delta_stats(events):
    """Compute flux/delta statistics for aux loss weighting."""
    wl1d_d, wl2d_d = [], []
    inlet_abs, e_flow_abs, e_vel_abs = [], [], []
    for ev in events:
        wl1d_d.append(np.diff(ev.nodes_1d_dynamic[:, :, 0], axis=0).ravel())
        wl2d_d.append(np.diff(ev.nodes_2d_dynamic[:, :, 1], axis=0).ravel())
        inlet_abs.append(ev.nodes_1d_dynamic[:, :, 1].ravel())
        e_flow_abs.append(ev.edges_1d_dynamic[:, :, 0].ravel())
        e_vel_abs.append(ev.edges_1d_dynamic[:, :, 1].ravel())
    return {
        "wl_1d_delta_std": float(np.std(np.concatenate(wl1d_d))) + 1e-8,
        "wl_2d_delta_std": float(np.std(np.concatenate(wl2d_d))) + 1e-8,
        "inlet_abs_std": float(np.std(np.concatenate(inlet_abs))) + 1e-8,
        "edge_flow_abs_std": float(np.std(np.concatenate(e_flow_abs))) + 1e-8,
        "edge_vel_abs_std": float(np.std(np.concatenate(e_vel_abs))) + 1e-8,
    }


def mask_2d_edges_only(event, t):
    if t < event.edges_2d_dynamic.shape[0]:
        event.edges_2d_dynamic[t, :, :] = 0.0


def validate_full_rollout(model, val_events, config, norm_stats, std_1d, std_2d,
                          device, per_node_stats=None):
    """Full rollout validation (same as v11c)."""
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
            dyn_2d[:, :, 2] = 0.0

            cur_1d = dyn_1d[spin_up, :, 0].copy()
            cur_2d = dyn_2d[spin_up, :, 1].copy()
            p1, p2, t1, t2 = [], [], [], []

            for t in range(spin_up, max_t - 1):
                if t > spin_up:
                    dyn_1d[t, :, 0] = cur_1d
                    dyn_2d[t, :, 1] = cur_2d
                mask_2d_edges_only(event, t)
                if t > 0:
                    mask_2d_edges_only(event, t - 1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    g = build_graph_at_timestep(
                        config, event, t, prev_t=t-1, norm_stats=norm_stats,
                        future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                        per_node_stats=per_node_stats,
                    ).to(device)
                    out = model(g)

                out_1d = out["1d"].float().cpu().numpy()
                out_2d = out["2d"].float().cpu().numpy()
                delta_1d = out_1d[:, 0]
                delta_2d = out_2d[:, 0]
                if pn_std_1d is not None:
                    delta_1d = delta_1d * pn_std_1d
                if pn_std_2d is not None:
                    delta_2d = delta_2d * pn_std_2d
                cur_1d = cur_1d + delta_1d
                cur_2d = cur_2d + delta_2d

                dyn_1d[t + 1, :, 1] = out_1d[:, 1]
                if "1d_edge" in out and t + 1 < dyn_e1d.shape[0]:
                    ep = out["1d_edge"].float().cpu().numpy()
                    dyn_e1d[t + 1, :, 0] = ep[:, 0]
                    dyn_e1d[t + 1, :, 1] = ep[:, 1]

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


def train_phase_multistep(
    model, train_events, val_events, config, norm_stats, std_1d, std_2d,
    device, cache_dir, rollout, lr, epochs, samples, w_1d, tag, best_val,
    aux_w_inlet, aux_w_edge, per_node_stats,
    w_traj=0.1,
    tbptt_k=16,
):
    """Multi-step backprop with 1D gradient chain.

    build_graph_at_timestep 後、1D WL columns を differentiable tensor で置換。
    cur_1d → input feature → model → delta → cur_1d の gradient chain により
    optimizer が累積ドリフトを直接最適化可能。

    TBPTT every tbptt_k steps for memory management.
    """
    # Per-node stats as GPU tensors (for differentiable normalization)
    pn_mean_1d = torch.tensor(per_node_stats["1d_wl_mean"], dtype=torch.float32, device=device)
    pn_std_1d_t = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_1d_np = per_node_stats["1d_wl_std"]
    pn_std_2d_np = per_node_stats["2d_wl_std"]

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    mse_loss = torch.nn.MSELoss()

    eligible = [ev for ev in train_events if ev.nodes_1d_dynamic.shape[0] - 1 >= rollout + 1]
    print(f"    Eligible: {len(eligible)}/{len(train_events)} (need {rollout+2} steps)")
    print(f"    w_traj={w_traj}, tbptt_k={tbptt_k}")

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_traj, n_steps = 0.0, 0.0, 0
        t0 = time.time()

        for idx in np.random.permutation(len(eligible)):
            event = eligible[idx]
            max_t = event.nodes_1d_dynamic.shape[0] - 1

            for _ in range(samples):
                start_t = np.random.randint(1, max_t - rollout + 1)
                optimizer.zero_grad()

                # Copy dynamic data
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
                dyn_2d[:, :, 2] = 0.0  # v11c: zero water_volume

                # 1D state as tensor (gradient chain), 2D as numpy (detached)
                cur_1d = torch.tensor(
                    dyn_1d[start_t, :, 0], dtype=torch.float32, device=device)
                prev_1d = cur_1d.clone()  # GT at start_t, detached
                cur_2d = dyn_2d[start_t, :, 1].copy()

                acc_loss = 0.0  # will become tensor after first step
                step_total, step_traj = 0.0, 0.0

                for step in range(rollout):
                    t = start_t + step
                    if t + 1 >= orig_1d.shape[0]:
                        break

                    # Write numpy state for graph builder
                    if step > 0:
                        dyn_1d[t, :, 0] = cur_1d.detach().cpu().numpy()
                        dyn_2d[t, :, 1] = cur_2d
                    mask_2d_edges_only(event, t)
                    if t > 0:
                        mask_2d_edges_only(event, t - 1)

                    # Build graph (numpy)
                    g = build_graph_at_timestep(
                        config, event, t, prev_t=t-1, norm_stats=norm_stats,
                        future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                        per_node_stats=per_node_stats,
                    ).to(device)

                    # === Replace 1D WL columns with differentiable tensors ===
                    if step >= 1:
                        x = g["1d"].x  # [N_1d, F]
                        wl_t_n = (cur_1d - pn_mean_1d) / pn_std_1d_t
                        wl_p_n = (prev_1d - pn_mean_1d) / pn_std_1d_t
                        g["1d"].x = torch.cat([
                            x[:, :WL_T_COL],              # cols 0-5: static
                            wl_t_n.unsqueeze(1),           # col 6: WL_t (differentiable)
                            x[:, WL_T_COL+1:WL_PREV_COL], # col 7: inlet_t
                            wl_p_n.unsqueeze(1),           # col 8: WL_prev (differentiable)
                            x[:, WL_PREV_COL+1:],         # col 9+: rest
                        ], dim=1)

                    # Forward (bf16 autocast)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(g)

                    # 1D: keep in computational graph
                    pred_1d = out["1d"].float()
                    delta_n_1d = pred_1d[:, 0]
                    delta_r_1d = delta_n_1d * pn_std_1d_t
                    prev_1d = cur_1d
                    cur_1d = cur_1d + delta_r_1d  # GRADIENT CHAIN

                    # 2D: detach state, but loss still in graph for model params
                    pred_2d = out["2d"].float()
                    with torch.no_grad():
                        cur_2d = cur_2d + pred_2d[:, 0].cpu().numpy() * pn_std_2d_np
                        dyn_1d[t+1, :, 1] = pred_1d[:, 1].detach().cpu().numpy()
                        if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                            ep = out["1d_edge"].float().cpu().numpy()
                            dyn_e1d[t+1, :, 0] = ep[:, 0]
                            dyn_e1d[t+1, :, 1] = ep[:, 1]

                    # === Losses ===
                    gt_d1 = torch.tensor(
                        (orig_1d[t+1, :, 0] - orig_1d[t, :, 0]) / pn_std_1d_np,
                        dtype=torch.float32, device=device)
                    gt_d2 = torch.tensor(
                        (orig_2d[t+1, :, 1] - orig_2d[t, :, 1]) / pn_std_2d_np,
                        dtype=torch.float32, device=device)
                    l_d1 = mse_loss(delta_n_1d, gt_d1)
                    l_d2 = mse_loss(pred_2d[:, 0], gt_d2)

                    # Trajectory loss: cumulative WL error (1D only, gradient chain)
                    gt_wl = torch.tensor(orig_1d[t+1, :, 0], dtype=torch.float32, device=device)
                    l_traj = ((cur_1d - gt_wl) / pn_std_1d_t).pow(2).mean()

                    # Aux losses
                    gt_in = torch.tensor(orig_1d[t+1, :, 1], dtype=torch.float32, device=device)
                    l_inlet = mse_loss(pred_1d[:, 1], gt_in)
                    l_edge = torch.tensor(0.0, device=device)
                    if "1d_edge" in out and t+1 < orig_e1d.shape[0]:
                        gt_e = torch.tensor(orig_e1d[t+1], dtype=torch.float32, device=device)
                        l_edge = mse_loss(out["1d_edge"].float(), gt_e)

                    s_loss = (w_1d * l_d1 + l_d2 + w_traj * l_traj
                              + aux_w_inlet * l_inlet + aux_w_edge * l_edge) / rollout
                    acc_loss = acc_loss + s_loss
                    step_total += s_loss.item()
                    step_traj += l_traj.item() / rollout

                    # TBPTT: backward every K steps, accumulate gradients
                    if (step + 1) % tbptt_k == 0 and step < rollout - 1:
                        acc_loss.backward()
                        acc_loss = 0.0
                        cur_1d = cur_1d.detach().clone()
                        prev_1d = prev_1d.detach().clone()

                # Final backward
                if isinstance(acc_loss, torch.Tensor) and acc_loss.requires_grad:
                    acc_loss.backward()

                # Restore event data
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
                    "model_version": "v11c", "rollout": rollout,
                    "per_node_norm": True, "future_rain_steps": FUTURE_RAIN_STEPS,
                    "coupling_edge_dim": COUPLING_EDGE_DIM,
                    "water_volume_zeroed": True,
                    "multistep_backprop": True,
                }, os.path.join(cache_dir, f"best_model_2_{tag}.pt"))
                val_str += " *"

        lr_now = scheduler.get_last_lr()[0]
        print(f"  r{rollout:3d} {epoch:3d} | loss={avg_l:.6f} traj={avg_t:.4f}"
              f" | {val_str} | best={best_val:.4f}"
              f" | lr={lr_now:.2e} | {time.time()-t0:.1f}s")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=str, default="v11c_r32",
                        help="Base checkpoint: v11c_r32 or v11c_lr_r64")
    parser.add_argument("--w_traj", type=float, default=0.1,
                        help="Trajectory loss weight (default: 0.1)")
    parser.add_argument("--tag", type=str, default="ms",
                        help="Checkpoint tag prefix (default: ms)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = "cuda"
    data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
    cache_dir = os.path.join(BASE, "Models", "checkpoints")
    seed_tag = f"_s{args.seed}" if args.seed != 42 else ""

    w_traj_p1 = args.w_traj
    w_traj_p2 = args.w_traj  # same for both phases
    tag_prefix = args.tag

    print("=" * 70)
    print(f"  M2 v11c Multi-Step Backprop (seed={args.seed}, base={args.base})")
    print(f"  1D gradient chain (198 nodes), 2D detached (4299 nodes)")
    print(f"  TBPTT k=16, w_traj={w_traj_p1}, tag={tag_prefix}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"M2: {config.num_1d_nodes} 1D, {config.num_2d_nodes} 2D nodes")

    ns_cache = os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl")
    with open(ns_cache, "rb") as f:
        norm_stats = pickle.load(f)
    print("  Loaded norm_stats")

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

    train_lengths = [ev.nodes_1d_dynamic.shape[0] for ev in train_events]
    print(f"  Train lengths: min={min(train_lengths)}, max={max(train_lengths)}, "
          f"median={np.median(train_lengths):.0f}")

    pn_cache = os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl")
    with open(pn_cache, "rb") as f:
        per_node_stats = pickle.load(f)
    print("  Loaded per_node_stats")

    fstats = compute_flux_and_delta_stats(train_events)
    wl_avg_var = (fstats["wl_1d_delta_std"]**2 + fstats["wl_2d_delta_std"]**2) / 2
    aux_w_inlet = AUX_REL_WEIGHT * wl_avg_var / (fstats["inlet_abs_std"]**2)
    edge_avg_var = (fstats["edge_flow_abs_std"]**2 + fstats["edge_vel_abs_std"]**2) / 2
    aux_w_edge = AUX_REL_WEIGHT * wl_avg_var / edge_avg_var
    print(f"  aux weights: inlet={aux_w_inlet:.6f}, edge={aux_w_edge:.6f}")

    # Model
    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)

    # Load checkpoint
    if args.base == "v11c_lr_r64":
        ckpt_path = os.path.join(cache_dir, f"best_model_2_v11c_lr{seed_tag}_r64.pt")
    else:
        ckpt_path = os.path.join(cache_dir, f"best_model_2_v11c{seed_tag}_r32.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(cache_dir, "best_model_2_v11c_r32.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_val = ckpt.get("best_val", float("inf"))
    print(f"  Loaded: {os.path.basename(ckpt_path)} (val={baseline_val:.4f})")

    # Initial validation
    v0, v0_1d, v0_2d = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats)
    print(f"Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    best_val = v0

    # Phase 1: r=32, multi-step backprop
    print(f"\n--- Phase 1: r=32, lr=5e-5, 15ep, tbptt=16, w_traj={w_traj_p1} ---")
    best_val = train_phase_multistep(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=32, lr=5e-5, epochs=15, samples=2,
        w_1d=3.0, tag=f"v11c_{tag_prefix}{seed_tag}_r32", best_val=best_val,
        aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
        per_node_stats=per_node_stats,
        w_traj=w_traj_p1, tbptt_k=16)

    # Reload best r=32
    ckpt_32 = os.path.join(cache_dir, f"best_model_2_v11c_{tag_prefix}{seed_tag}_r32.pt")
    if os.path.exists(ckpt_32):
        model.load_state_dict(
            torch.load(ckpt_32, map_location=device, weights_only=False)["model_state_dict"])
        print(f"  Reloaded best r=32 checkpoint")

    # Phase 2: r=64, extended with multi-step
    print(f"\n--- Phase 2: r=64, lr=1e-5, 10ep, tbptt=16, w_traj={w_traj_p2} ---")
    best_val = train_phase_multistep(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=64, lr=1e-5, epochs=10, samples=1,
        w_1d=3.0, tag=f"v11c_{tag_prefix}{seed_tag}_r64", best_val=best_val,
        aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
        per_node_stats=per_node_stats,
        w_traj=w_traj_p2, tbptt_k=16)

    print(f"\nDone. Best val: {best_val:.4f}")
    print(f"(Baseline: {baseline_val:.4f})")


if __name__ == "__main__":
    main()
