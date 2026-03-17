"""v76: SRMSE-aligned loss — fix fundamental loss-metric misalignment.

Discovery: 2D node weights are 6.2x misaligned with SRMSE metric.
- SRMSE = RMSE / global_std, where RMSE weights all nodes equally in absolute space
- Training loss normalizes by per_node_std, over-weighting low-variance nodes
- Bottom 50% of 2D nodes: get 20.5% of training weight but contribute only 5.5% to SRMSE
- Top 10% of 2D nodes: get 28.8% of training weight but contribute 51.7% to SRMSE

Fix: node_weight = (per_node_std / global_std)^2, aligning loss gradient with SRMSE gradient.
"""
import sys, os, time, pickle, argparse, gc
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import load_model_config, load_event_data, list_events, build_graph_at_timestep
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events

from run_train_m2_v11c_multistep import (
    compute_flux_and_delta_stats, mask_2d_edges_only,
    MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM, AUX_REL_WEIGHT,
    WL_T_COL, WL_PREV_COL,
)
from run_train_m2_v55_noise import (
    validate_full_rollout,
    STEP_CHECKPOINTS, RAIN_THRESHOLD,
    WL_2D_T_COL, WL_2D_PREV_COL,
)
from break_recorder import BreakRecorder


def compute_srmse_aligned_weights(per_node_stats, global_std_1d, global_std_2d,
                                   w_min=0.01, w_max=10.0):
    """SRMSE-aligned node weights: (per_node_std / global_std)^2.

    This ensures the training loss gradient matches the SRMSE metric gradient.
    Nodes with high variance (which dominate SRMSE) get proportionally more weight.
    """
    std_1d = per_node_stats["1d_wl_std"]
    std_2d = per_node_stats["2d_wl_std"]

    # w_i = (std_i / global_std)^2 — aligns loss with SRMSE
    w_1d = np.clip((std_1d / global_std_1d) ** 2, w_min, w_max)
    w_2d = np.clip((std_2d / global_std_2d) ** 2, w_min, w_max)

    # Normalize so mean weight = 1.0 (keeps loss scale similar to previous versions)
    w_1d = w_1d / w_1d.mean()
    w_2d = w_2d / w_2d.mean()

    return w_1d, w_2d


def train_phase_mixed(
    model, train_events, val_events, config, norm_stats, std_1d, std_2d,
    device, cache_dir, max_rollout, lr, epochs, w_1d_scale, tag, best_val,
    aux_w_inlet, aux_w_edge, per_node_stats,
    node_weight_1d, node_weight_2d,
    w_traj=0.3, w_traj_2d=0.1,
    k_1d=32, k_2d=16, spin_up=10,
    recorder=None, phase_name="",
    recess_w=1.0,
    noise_std=0.0,
    step_zone_w=None,
    node_groups_1d=None,
    min_rollout=16,
):
    """Mixed-rollout training with SRMSE-aligned loss."""
    pn_mean_1d = torch.tensor(per_node_stats["1d_wl_mean"], dtype=torch.float32, device=device)
    pn_std_1d_t = torch.tensor(per_node_stats["1d_wl_std"], dtype=torch.float32, device=device)
    pn_std_1d_np = per_node_stats["1d_wl_std"]
    pn_std_2d_np = per_node_stats["2d_wl_std"]
    pn_mean_2d = torch.tensor(per_node_stats["2d_wl_mean"], dtype=torch.float32, device=device)
    pn_std_2d_t = torch.tensor(per_node_stats["2d_wl_std"], dtype=torch.float32, device=device)

    nw_1d = torch.tensor(node_weight_1d, dtype=torch.float32, device=device)
    nw_2d = torch.tensor(node_weight_2d, dtype=torch.float32, device=device)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    eligible = [ev for ev in train_events if ev.nodes_1d_dynamic.shape[0] - 1 >= min_rollout + spin_up + 1]
    event_max_rollout = {}
    for i, ev in enumerate(eligible):
        max_t = ev.nodes_1d_dynamic.shape[0] - 1
        event_max_rollout[i] = min(max_t - spin_up, max_rollout)

    rollout_lengths = list(event_max_rollout.values())
    print(f"    Eligible: {len(eligible)}/{len(train_events)} (min_rollout={min_rollout})")
    print(f"    Rollout range: {min(rollout_lengths)}-{max(rollout_lengths)}")
    print(f"    aux_w_edge={aux_w_edge:.6f}, recess_w={recess_w:.1f}")

    noise_1d_scale = torch.tensor(pn_std_1d_np, dtype=torch.float32, device=device)
    noise_2d_scale = torch.tensor(pn_std_2d_np, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        cur_noise_std = noise_std * epoch / max(epochs - 1, 1) if noise_std > 0 else 0.0

        model.train()
        ep_loss, ep_traj, ep_edge, ep_grad_norm, n_steps = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for idx in np.random.permutation(len(eligible)):
            event = eligible[idx]
            max_t = event.nodes_1d_dynamic.shape[0] - 1
            rollout = event_max_rollout[idx]

            T = max_t
            candidates = [spin_up]
            for frac in [0.25, 0.5, 0.75]:
                pos = int(T * frac)
                if pos + rollout <= max_t and pos >= spin_up:
                    candidates.append(pos)
            start_t = candidates[np.random.randint(len(candidates))]

            optimizer.zero_grad()

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
            dyn_2d[:, :, 2] = 0.0

            rain_avg_per_step = np.mean(orig_2d[:, :, 0], axis=1)

            cur_1d = torch.tensor(dyn_1d[start_t, :, 0], dtype=torch.float32, device=device)
            prev_1d = cur_1d.clone()
            cur_2d = torch.tensor(dyn_2d[start_t, :, 1], dtype=torch.float32, device=device)
            prev_2d = cur_2d.clone()

            acc_loss = 0.0
            step_total, step_traj, step_edge = 0.0, 0.0, 0.0
            steps_since_2d_detach = 0
            steps_since_1d_detach = 0

            for step in range(rollout):
                t = start_t + step
                if t + 1 > max_t:
                    break

                if step > 0:
                    dyn_1d[t, :, 0] = cur_1d.detach().cpu().numpy()
                    dyn_2d[t, :, 1] = cur_2d.detach().cpu().numpy()
                mask_2d_edges_only(event, t)
                if t > 0:
                    mask_2d_edges_only(event, t - 1)

                g = build_graph_at_timestep(
                    config, event, t, prev_t=t-1, norm_stats=norm_stats,
                    future_rain_steps=FUTURE_RAIN_STEPS, coupling_features=True,
                    per_node_stats=per_node_stats,
                ).to(device)

                if step >= 1 and steps_since_1d_detach > 0:
                    x = g["1d"].x
                    input_1d = cur_1d
                    if cur_noise_std > 0:
                        input_1d = cur_1d + (torch.randn_like(cur_1d) * noise_1d_scale * cur_noise_std).detach()
                    wl_t_n = (input_1d - pn_mean_1d) / pn_std_1d_t
                    wl_p_n = (prev_1d - pn_mean_1d) / pn_std_1d_t
                    g["1d"].x = torch.cat([
                        x[:, :WL_T_COL], wl_t_n.unsqueeze(1),
                        x[:, WL_T_COL+1:WL_PREV_COL], wl_p_n.unsqueeze(1),
                        x[:, WL_PREV_COL+1:],
                    ], dim=1)

                if step >= 1 and steps_since_2d_detach > 0:
                    x2 = g["2d"].x
                    input_2d = cur_2d
                    if cur_noise_std > 0:
                        input_2d = cur_2d + (torch.randn_like(cur_2d) * noise_2d_scale * cur_noise_std).detach()
                    wl_2d_t_n = (input_2d - pn_mean_2d) / pn_std_2d_t
                    wl_2d_p_n = (prev_2d - pn_mean_2d) / pn_std_2d_t
                    g["2d"].x = torch.cat([
                        x2[:, :WL_2D_T_COL], wl_2d_t_n.unsqueeze(1),
                        x2[:, WL_2D_T_COL+1:WL_2D_PREV_COL], wl_2d_p_n.unsqueeze(1),
                        x2[:, WL_2D_PREV_COL+1:],
                    ], dim=1)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(g)

                pred_1d = out["1d"].float()
                delta_n_1d = pred_1d[:, 0]
                delta_r_1d = delta_n_1d * pn_std_1d_t
                prev_1d = cur_1d
                cur_1d = cur_1d + delta_r_1d

                pred_2d = out["2d"].float()
                delta_n_2d = pred_2d[:, 0]
                delta_r_2d = delta_n_2d * pn_std_2d_t
                prev_2d = cur_2d
                cur_2d = cur_2d + delta_r_2d
                steps_since_2d_detach += 1
                steps_since_1d_detach += 1

                with torch.no_grad():
                    dyn_1d[t+1, :, 1] = pred_1d[:, 1].detach().cpu().numpy()
                    if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                        ep_out = out["1d_edge"].float().cpu().numpy()
                        dyn_e1d[t+1, :, 0] = ep_out[:, 0]
                        dyn_e1d[t+1, :, 1] = ep_out[:, 1]

                # Losses — same normalization, but node weights are SRMSE-aligned
                gt_d1 = torch.tensor(
                    (orig_1d[t+1, :, 0] - orig_1d[t, :, 0]) / pn_std_1d_np,
                    dtype=torch.float32, device=device)
                gt_d2 = torch.tensor(
                    (orig_2d[t+1, :, 1] - orig_2d[t, :, 1]) / pn_std_2d_np,
                    dtype=torch.float32, device=device)

                l_d1 = (nw_1d * (delta_n_1d - gt_d1).pow(2)).mean()
                l_d2 = (nw_2d * (delta_n_2d - gt_d2).pow(2)).mean()

                gt_wl_1d = torch.tensor(orig_1d[t+1, :, 0], dtype=torch.float32, device=device)
                l_traj_1d = (nw_1d * ((cur_1d - gt_wl_1d) / pn_std_1d_t).pow(2)).mean()
                gt_wl_2d = torch.tensor(orig_2d[t+1, :, 1], dtype=torch.float32, device=device)
                l_traj_2d = (nw_2d * ((cur_2d - gt_wl_2d) / pn_std_2d_t).pow(2)).mean()

                gt_in = torch.tensor(orig_1d[t+1, :, 1], dtype=torch.float32, device=device)
                l_inlet = torch.nn.functional.mse_loss(pred_1d[:, 1], gt_in)

                l_edge = torch.tensor(0.0, device=device)
                if "1d_edge" in out and t+1 < orig_e1d.shape[0]:
                    gt_e = torch.tensor(orig_e1d[t+1], dtype=torch.float32, device=device)
                    l_edge = torch.nn.functional.mse_loss(out["1d_edge"].float(), gt_e)

                sz_mul = 1.0
                if step_zone_w is not None:
                    for thresh in sorted(step_zone_w.keys()):
                        if t >= thresh:
                            sz_mul = step_zone_w[thresh]

                s_loss = (w_1d_scale * l_d1 + l_d2
                          + sz_mul * (w_traj * l_traj_1d + w_traj_2d * l_traj_2d)
                          + aux_w_inlet * l_inlet + aux_w_edge * l_edge)

                if recess_w > 1.0 and rain_avg_per_step[t] < RAIN_THRESHOLD:
                    s_loss = s_loss * recess_w

                acc_loss = acc_loss + s_loss / rollout
                step_total += s_loss.item() / rollout
                step_traj += (l_traj_1d.item() + l_traj_2d.item()) / rollout
                step_edge += l_edge.item() / rollout

                if steps_since_2d_detach >= k_2d and step < rollout - 1:
                    cur_2d = cur_2d.detach().clone()
                    prev_2d = prev_2d.detach().clone()
                    steps_since_2d_detach = 0

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

            if isinstance(acc_loss, torch.Tensor) and acc_loss.requires_grad:
                acc_loss.backward()

            event.nodes_1d_dynamic = orig_1d
            event.nodes_2d_dynamic = orig_2d
            event.edges_1d_dynamic = orig_e1d
            event.edges_2d_dynamic = orig_e2d
            del dyn_1d, dyn_2d, dyn_e1d, dyn_e2d

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ep_grad_norm += gn.item()
            optimizer.step()
            ep_loss += step_total
            ep_traj += step_traj
            ep_edge += step_edge
            n_steps += 1

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        avg_l = ep_loss / max(n_steps, 1)
        avg_t = ep_traj / max(n_steps, 1)
        avg_e = ep_edge / max(n_steps, 1)
        avg_gn = ep_grad_norm / max(n_steps, 1)

        val_str = ""
        step_rmse = {}
        vs, v1d, v2d = None, None, None
        if epoch % 2 == 0 or epoch == epochs - 1:
            vs, v1d, v2d, step_rmse = validate_full_rollout(
                model, val_events, config, norm_stats, std_1d, std_2d, device,
                per_node_stats=per_node_stats, step_checkpoints=STEP_CHECKPOINTS,
                node_groups_1d=node_groups_1d)
            val_str = f"val={vs:.4f} (1d={v1d:.4f} 2d={v2d:.4f})"
            if vs < best_val:
                best_val = vs
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "best_val": best_val,
                    "val_1d": v1d, "val_2d": v2d,
                    "model_version": "v11c", "rollout": max_rollout,
                    "per_node_norm": True, "future_rain_steps": FUTURE_RAIN_STEPS,
                    "coupling_edge_dim": COUPLING_EDGE_DIM,
                    "water_volume_zeroed": True,
                    "multistep_backprop": True,
                    "random_start": True,
                    "gradient_chain_2d": True,
                    "separate_tbptt": True,
                    "k_1d": k_1d, "k_2d": k_2d,
                    "loss_aligned": True,
                    "edge_w_boosted": True,
                    "srmse_aligned": True,  # v76 flag
                }, os.path.join(cache_dir, f"best_model_2_{tag}.pt"))
                val_str += " *"

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"  v76 {epoch:3d} | loss={avg_l:.6f} traj={avg_t:.4f} edge={avg_e:.4f}"
              f" | {val_str} | best={best_val:.4f} | lr={lr_now:.2e} gn={avg_gn:.2f} | {elapsed:.1f}s")

        if recorder and vs is not None:
            sr = {str(k): v for k, v in step_rmse.items()}
            sim = recorder.compute_similarity(sr)
            recorder.step(epoch, phase=phase_name, val_loss=vs,
                         val_rmse_1d=v1d, val_rmse_2d=v2d,
                         step_rmse=sr, similarity=sim,
                         train_loss=avg_l, lr=lr_now,
                         best_val=best_val)
            diag = recorder.diagnose()
            if diag:
                for line in recorder.format_alerts(diag):
                    print(f"  {line}")
            # v5: predictive auto-stop (combines rule-based + kNN prediction)
            stop, reason = recorder.should_stop_predictive()
            if stop:
                print(f"  >>> AUTO-STOP: {reason}")
                recorder.save()
                break
            # v5: show prediction at epoch 3+
            val_count = len([h for h in recorder.history
                            if "val_loss" in h and h.get("epoch", -99) >= 0])
            if val_count in (3, 5):
                pred = recorder.predict_outcome(epoch_cutoff=min(val_count, 5))
                if pred:
                    print(f"  [PRED] val={pred['predicted_val']:.4f} "
                          f"({pred['predicted_outcome']}) "
                          f"conf={pred['confidence']:.2f} "
                          f"nearest={pred['similar_runs'][0]['name']}")
            recorder.save()

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_ckpt", type=str, default="v75_dw_r64")
    parser.add_argument("--k_1d", type=int, default=32)
    parser.add_argument("--k_2d", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_rollout", type=int, default=400)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = "cuda"
    data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
    cache_dir = os.path.join(BASE, "Models", "checkpoints")

    step_zone_w = {100: 1.5, 200: 2.5, 300: 5.0}

    print("=" * 70)
    print(f"  v76: SRMSE-Aligned Loss (fundamental fix)")
    print(f"  2D misalignment was 6.2x -- now corrected")
    print(f"  Base: {args.base_ckpt}, lr={args.lr:.0e}, epochs={args.epochs}")
    print(f"  max_rollout={args.max_rollout}, step_zone_w={step_zone_w}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"  Global std: 1d={std_1d:.4f}, 2d={std_2d:.4f}")

    from src.model_lstm1d import build_adjacency
    _, _, degree_1d = build_adjacency(config.edge_index_1d, config.num_1d_nodes)
    node_groups_1d = {
        "deg1": np.where(degree_1d == 1)[0],
        "deg2": np.where(degree_1d == 2)[0],
        "hub": np.where(degree_1d >= 3)[0],
    }

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    # v76: SRMSE-aligned weights instead of variance-ratio weights
    nw_1d, nw_2d = compute_srmse_aligned_weights(
        per_node_stats, std_1d, std_2d,
        w_min=0.01, w_max=10.0)

    # deg1 boost (still useful — deg1 nodes are high-SRMSE-impact)
    nw_1d[node_groups_1d["deg1"]] *= 1.5

    print(f"  SRMSE-aligned weights:")
    print(f"    1d: min={nw_1d.min():.3f} max={nw_1d.max():.3f} mean={nw_1d.mean():.3f}")
    print(f"    2d: min={nw_2d.min():.3f} max={nw_2d.max():.3f} mean={nw_2d.mean():.3f}")

    all_events = list_events(data_dir, MODEL_ID, "train")
    train_ids = all_events.tolist() if hasattr(all_events, 'tolist') else list(all_events)
    val_ids = train_ids

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)} (in-sample)")
    print("Loading events...")
    t0 = time.time()
    train_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in train_ids]
    val_events = train_events
    print(f"  Loaded in {time.time()-t0:.1f}s")

    lengths = [e.nodes_1d_dynamic.shape[0] for e in train_events]
    for l in sorted(set(lengths)):
        n = sum(1 for x in lengths if x == l)
        max_r = min(l - 10 - 1, args.max_rollout)
        print(f"  T={l}: {n} events (max rollout: {max_r})")

    fstats = compute_flux_and_delta_stats(train_events)
    wl_avg_var = (fstats["wl_1d_delta_std"]**2 + fstats["wl_2d_delta_std"]**2) / 2
    aux_w_inlet = AUX_REL_WEIGHT * wl_avg_var / (fstats["inlet_abs_std"]**2)
    aux_w_edge = 0.001

    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)

    ckpt_path = os.path.join(cache_dir, f"best_model_2_{args.base_ckpt}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_val = ckpt.get("best_val", float("inf"))
    print(f"  Loaded {args.base_ckpt}: val={baseline_val:.4f}")

    recorder = BreakRecorder(
        log_dir=os.path.join(BASE, "break_logs"),
        run_name=f"v76_aligned_s{args.seed}")
    recorder.set_meta(
        version="v76_srmse_aligned", model="HeteroFloodGNNv11",
        hidden_dim=128, num_processor_layers=4, seed=args.seed,
        baseline_val=baseline_val, base_ckpt=args.base_ckpt,
        max_rollout=args.max_rollout, lr=args.lr, epochs=args.epochs,
        step_zone_w=step_zone_w,
        srmse_aligned=True,
        global_std_1d=std_1d, global_std_2d=std_2d)

    print("\nInitial validation...")
    v0, v0_1d, v0_2d, v0_step = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats, step_checkpoints=STEP_CHECKPOINTS,
        node_groups_1d=node_groups_1d)
    print(f"  Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    best_val = v0

    init_step_rmse = {str(k): v for k, v in v0_step.items()}
    sim_info = recorder.compute_similarity(init_step_rmse)
    recorder.step(-1, phase="init", val_loss=v0, val_rmse_1d=v0_1d, val_rmse_2d=v0_2d,
                  step_rmse=init_step_rmse, similarity=sim_info)
    recorder.save()

    # === Phase 1: SRMSE-aligned training ===
    tag = f"v76_aligned_r{args.max_rollout}_s{args.seed}"
    print(f"\n--- Phase 1: SRMSE-aligned, mixed rollout, lr={args.lr:.0e}, {args.epochs}ep ---")
    best_val = train_phase_mixed(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, max_rollout=args.max_rollout, lr=args.lr, epochs=args.epochs,
        w_1d_scale=3.0, tag=tag, best_val=best_val,
        aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        k_1d=args.k_1d, k_2d=args.k_2d,
        recess_w=2.0,
        noise_std=0.0,
        recorder=recorder, phase_name=f"p1_aligned_r{args.max_rollout}",
        step_zone_w=step_zone_w,
        node_groups_1d=node_groups_1d)
    recorder.save()

    # === Phase 2: Polish with lower lr ===
    p1_ckpt = os.path.join(cache_dir, f"best_model_2_{tag}.pt")
    if os.path.exists(p1_ckpt) and best_val < baseline_val:
        model.load_state_dict(
            torch.load(p1_ckpt, map_location=device, weights_only=False)["model_state_dict"])
        print(f"\n  Reloaded best Phase 1: val={best_val:.4f}")

        polish_lr = args.lr * 0.3
        tag_p2 = f"v76_aligned_r{args.max_rollout}_s{args.seed}_p2"
        print(f"\n--- Phase 2 Polish: lr={polish_lr:.0e}, 10ep ---")
        best_val = train_phase_mixed(
            model, train_events, val_events, config, norm_stats, std_1d, std_2d,
            device, cache_dir, max_rollout=args.max_rollout, lr=polish_lr, epochs=10,
            w_1d_scale=3.0, tag=tag_p2, best_val=best_val,
            aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
            per_node_stats=per_node_stats,
            node_weight_1d=nw_1d, node_weight_2d=nw_2d,
            k_1d=args.k_1d, k_2d=args.k_2d,
            recess_w=2.0,
            noise_std=0.0,
            recorder=recorder, phase_name="p2_polish",
            step_zone_w=step_zone_w,
            node_groups_1d=node_groups_1d)
        recorder.save()
    else:
        print(f"\n  Phase 1 did not improve (best={best_val:.4f} >= baseline={baseline_val:.4f})")

    # Copy best to standard name
    import shutil
    for tag_check in [f"v76_aligned_r{args.max_rollout}_s{args.seed}_p2", f"v76_aligned_r{args.max_rollout}_s{args.seed}"]:
        src = os.path.join(cache_dir, f"best_model_2_{tag_check}.pt")
        if os.path.exists(src):
            dst = os.path.join(cache_dir, f"best_model_2_v76s{args.seed}_dw_r64.pt")
            shutil.copy2(src, dst)
            print(f"\n  Best: {tag_check} -> v76_dw_r64")
            break

    print(f"\n{'='*70}")
    print(f"  v76 SRMSE-Aligned Complete!")
    print(f"  Baseline val: {baseline_val:.4f}")
    print(f"  Best val: {best_val:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
