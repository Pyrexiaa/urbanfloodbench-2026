"""v55: Noise Injection Training — 誤差修正能力の明示的強化.

EDA発見: delta誤差の自己相関が0.79 (1D) — 誤差が系統的に蓄積。
さらに、汚染入力の方がTFより正しいdeltaを出すケースあり (Event 15)。
→ モデルは自然に誤差修正を学習しているが、明示的に訓練されていない。

v55: 訓練中の各step, cur_1d/cur_2dにノイズ追加 (scheduled noise).
σ = noise_std * per_node_std * curriculum_ratio
これにより「汚染入力からでも正しいdeltaを予測する」能力を強化。
v54_rw2_r64 (val=0.0650) から fine-tune.
"""
import sys
import os
import gc
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
)
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events, compute_standardized_rmse
from break_recorder import BreakRecorder

from run_train_m2_v11c_multistep import (
    compute_flux_and_delta_stats, mask_2d_edges_only,
    MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM, AUX_REL_WEIGHT,
    WL_T_COL, WL_PREV_COL,
)

# 2D feature column indices (per_node_stats mode)
WL_2D_T_COL = 10
WL_2D_PREV_COL = 13
STEP_CHECKPOINTS = [50, 100, 200, 300, 399]
RAIN_THRESHOLD = 0.0005  # v54: recession判定閾値 (空間平均降雨量)


def compute_node_weights(per_node_stats, w_min=0.1, w_max=5.0):
    std_1d = per_node_stats["1d_wl_std"]
    std_2d = per_node_stats["2d_wl_std"]
    w_1d = np.clip(std_1d / std_1d.mean(), w_min, w_max)
    w_2d = np.clip(std_2d / std_2d.mean(), w_min, w_max)
    return w_1d, w_2d


def validate_full_rollout(model, val_events, config, norm_stats, std_1d, std_2d,
                          device, per_node_stats=None, step_checkpoints=None,
                          node_groups_1d=None):
    model.eval()
    spin_up = 10
    scores, s1d, s2d = [], [], []
    step_data = {k: {"1d": [], "2d": []} for k in (step_checkpoints or [])}
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
                p1a, t1a = np.stack(p1), np.stack(t1)
                p2a, t2a = np.stack(p2), np.stack(t2)
                s_1d = compute_standardized_rmse(p1a, t1a, std_1d)
                s_2d = compute_standardized_rmse(p2a, t2a, std_2d)
                scores.append((s_1d + s_2d) / 2)
                s1d.append(s_1d)
                s2d.append(s_2d)
                for k in (step_checkpoints or []):
                    if k <= len(p1):
                        e1 = p1a[k - 1] - t1a[k - 1]
                        e2 = p2a[k - 1] - t2a[k - 1]
                        step_data[k]["1d"].append(np.sqrt(np.mean(e1**2)) / std_1d)
                        step_data[k]["2d"].append(np.sqrt(np.mean(e2**2)) / std_2d)
                        # v2: per-group 1D RMSE
                        if node_groups_1d is not None:
                            for gname, gidx in node_groups_1d.items():
                                key = f"1d_{gname}"
                                if key not in step_data[k]:
                                    step_data[k][key] = []
                                step_data[k][key].append(
                                    np.sqrt(np.mean(e1[gidx]**2)) / std_1d)

    step_rmse = {}
    for k, v in step_data.items():
        if v["1d"]:
            entry = {"1d": float(np.mean(v["1d"])), "2d": float(np.mean(v["2d"]))}
            # v2: group RMSE
            for key in v:
                if key.startswith("1d_"):
                    entry[key] = float(np.mean(v[key]))
            step_rmse[k] = entry
    return (float(np.mean(scores)) if scores else float("inf"),
            float(np.mean(s1d)) if s1d else float("inf"),
            float(np.mean(s2d)) if s2d else float("inf"),
            step_rmse)


def train_phase(
    model, train_events, val_events, config, norm_stats, std_1d, std_2d,
    device, cache_dir, rollout, lr, epochs, w_1d_scale, tag, best_val,
    aux_w_inlet, aux_w_edge, per_node_stats,
    node_weight_1d, node_weight_2d,
    w_traj=0.3, w_traj_2d=0.1,
    k_1d=32, k_2d=16, spin_up=10,
    recorder=None, phase_name="",
    recess_w=1.0,
    noise_std=0.0,
    noise_curriculum=True,
    step_zone_w=None,  # v64: step-zone-aware traj weight {abs_step_threshold: multiplier}
    node_groups_1d=None,  # v2 break recorder: degree-based node groups
):
    """Sep TBPTT training (v30) with noise injection for error correction."""
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

    eligible = [ev for ev in train_events if ev.nodes_1d_dynamic.shape[0] - 1 >= rollout + spin_up + 1]
    print(f"    Eligible: {len(eligible)}/{len(train_events)} (need {rollout + spin_up + 1} steps)")
    print(f"    aux_w_edge={aux_w_edge:.6f} (v53 boosted), recess_w={recess_w:.1f}, noise_std={noise_std:.3f}")

    # v55: noise curriculum — linearly increase noise over epochs
    noise_1d_scale = torch.tensor(pn_std_1d_np, dtype=torch.float32, device=device)
    noise_2d_scale = torch.tensor(pn_std_2d_np, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        # v55: curriculum — noise ramps from 0 to noise_std
        if noise_curriculum and epochs > 1:
            cur_noise_std = noise_std * epoch / (epochs - 1)
        else:
            cur_noise_std = noise_std

        model.train()
        ep_loss, ep_traj, ep_edge, ep_grad_norm, n_steps = 0.0, 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for idx in np.random.permutation(len(eligible)):
            event = eligible[idx]
            max_t = event.nodes_1d_dynamic.shape[0] - 1

            # Random start position (v25)
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

            # v54: per-step rainfall for recession detection
            rain_avg_per_step = np.mean(orig_2d[:, :, 0], axis=1)  # [T]

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

                # 1D gradient chain
                if step >= 1 and steps_since_1d_detach > 0:
                    x = g["1d"].x
                    # v55: noise on input features (no grad through noise)
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

                # 2D gradient chain
                if step >= 1 and steps_since_2d_detach > 0:
                    x2 = g["2d"].x
                    # v55: noise on input features (no grad through noise)
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

                # Aux feedback (detached)
                with torch.no_grad():
                    dyn_1d[t+1, :, 1] = pred_1d[:, 1].detach().cpu().numpy()
                    if "1d_edge" in out and t+1 < dyn_e1d.shape[0]:
                        ep_out = out["1d_edge"].float().cpu().numpy()
                        dyn_e1d[t+1, :, 0] = ep_out[:, 0]
                        dyn_e1d[t+1, :, 1] = ep_out[:, 1]

                # Losses
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

                # v64: step-zone-aware trajectory weight
                # step_zone_w = {threshold: multiplier}, e.g. {100: 1.5, 200: 3.0}
                sz_mul = 1.0
                if step_zone_w is not None:
                    for thresh in sorted(step_zone_w.keys()):
                        if t >= thresh:
                            sz_mul = step_zone_w[thresh]

                s_loss = (w_1d_scale * l_d1 + l_d2
                          + sz_mul * (w_traj * l_traj_1d + w_traj_2d * l_traj_2d)
                          + aux_w_inlet * l_inlet + aux_w_edge * l_edge) / rollout

                # v54: recession loss weighting
                if recess_w > 1.0 and rain_avg_per_step[t] < RAIN_THRESHOLD:
                    s_loss = s_loss * recess_w

                acc_loss = acc_loss + s_loss
                step_total += s_loss.item()
                step_traj += (l_traj_1d.item() + l_traj_2d.item()) / rollout
                step_edge += l_edge.item() / rollout

                # Separate TBPTT (v30)
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
            # v66: RAM OOM対策 — numpy copy()のフラグメンテーション抑制
            del dyn_1d, dyn_2d, dyn_e1d, dyn_e2d

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ep_grad_norm += gn.item()
            optimizer.step()
            ep_loss += step_total
            ep_traj += step_traj
            ep_edge += step_edge
            n_steps += 1

        scheduler.step()
        gc.collect()  # v66: numpy fragment回収
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
                    "model_version": "v11c", "rollout": rollout,
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
                }, os.path.join(cache_dir, f"best_model_2_{tag}.pt"))
                val_str += " *"

        lr_now = scheduler.get_last_lr()[0]
        print(f"  r{rollout:3d} {epoch:3d} | loss={avg_l:.6f} traj={avg_t:.4f} edge={avg_e:.4f}"
              f" | {val_str} | best={best_val:.4f}"
              f" | lr={lr_now:.2e} gn={avg_gn:.2f} | {time.time()-t0:.1f}s")

        if recorder is not None:
            rec = {
                "phase": phase_name, "rollout": rollout,
                "train_loss": avg_l, "traj_loss": avg_t, "edge_loss": avg_e,
                "grad_norm": avg_gn, "lr": lr_now, "best_val": best_val,
            }
            if vs is not None:
                rec["val_loss"] = vs
                rec["val_rmse_1d"] = v1d
                rec["val_rmse_2d"] = v2d
                rec["step_rmse"] = {str(k): v for k, v in step_rmse.items()}
            recorder.step(epoch, **rec)
            # v3: actionable diagnostics on validation epochs
            if vs is not None:
                alerts = recorder.diagnose()
                for a in alerts:
                    print(f"  [!] {a}")
            recorder.save()

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_w", type=float, default=0.001,
                        help="Override aux_w_edge (default 0.001, was 0.000004)")
    parser.add_argument("--k_1d", type=int, default=32)
    parser.add_argument("--k_2d", type=int, default=16)
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="Override base checkpoint tag (default: v11c_v39_p3_r64)")
    parser.add_argument("--skip_p1", action="store_true",
                        help="Skip Phase 1 (r=32), start from Phase 2 directly")
    parser.add_argument("--lr_p1", type=float, default=2e-5)
    parser.add_argument("--lr_p2", type=float, default=4e-6)
    parser.add_argument("--ep_p1", type=int, default=15)
    parser.add_argument("--ep_p2", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v55_noise",
                        help="Checkpoint tag prefix")
    parser.add_argument("--recess_w", type=float, default=2.0,
                        help="Loss weight multiplier for recession steps (rain<thresh)")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Noise injection magnitude (fraction of per-node std)")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable noise curriculum (constant noise from start)")
    parser.add_argument("--rollout_p2", type=int, default=64,
                        help="Rollout length for Phase 2 (default 64)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = "cuda"
    data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
    cache_dir = os.path.join(BASE, "Models", "checkpoints")

    print("=" * 70)
    print(f"  v55: Noise Injection Training (seed={args.seed})")
    print(f"  noise_std={args.noise_std}, recess_w={args.recess_w}, aux_w_edge={args.edge_w}")
    print(f"  curriculum={'ON' if not args.no_curriculum else 'OFF'}")
    print(f"  FINE-TUNE from v54_rw2, k_1d={args.k_1d}, k_2d={args.k_2d}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"M2: {config.num_1d_nodes} 1D, {config.num_2d_nodes} 2D nodes")

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    nw_1d, nw_2d = compute_node_weights(per_node_stats)

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

    # Compute dynamic aux weights for comparison
    fstats = compute_flux_and_delta_stats(train_events)
    wl_avg_var = (fstats["wl_1d_delta_std"]**2 + fstats["wl_2d_delta_std"]**2) / 2
    aux_w_inlet = AUX_REL_WEIGHT * wl_avg_var / (fstats["inlet_abs_std"]**2)
    edge_avg_var = (fstats["edge_flow_abs_std"]**2 + fstats["edge_vel_abs_std"]**2) / 2
    aux_w_edge_orig = AUX_REL_WEIGHT * wl_avg_var / edge_avg_var
    aux_w_edge = args.edge_w  # v53 override
    print(f"  aux_w_inlet={aux_w_inlet:.6f}")
    print(f"  aux_w_edge: {aux_w_edge_orig:.6f} (original) → {aux_w_edge:.6f} (v53, {aux_w_edge/aux_w_edge_orig:.0f}x)")

    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)

    base_tag = args.base_ckpt or "v54_rw2_r64"
    ckpt_path = os.path.join(cache_dir, f"best_model_2_{base_tag}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_val = ckpt.get("best_val", float("inf"))
    print(f"  Loaded {base_tag}: val={baseline_val:.4f}")

    # Break recorder
    recorder = BreakRecorder(
        log_dir=os.path.join(BASE, "break_logs"),
        run_name=f"v55_noise{args.noise_std}_s{args.seed}",
    )
    recorder.set_meta(
        version="v55", model="HeteroFloodGNNv11",
        hidden_dim=128, num_processor_layers=4, seed=args.seed,
        baseline_val=baseline_val, edge_w=aux_w_edge,
        edge_w_original=aux_w_edge_orig, edge_w_boost=aux_w_edge / aux_w_edge_orig,
        k_1d=args.k_1d, k_2d=args.k_2d,
        recess_w=args.recess_w, noise_std=args.noise_std,
        noise_curriculum=not args.no_curriculum,
        n_train=len(train_ids), n_val=len(val_ids),
    )

    # Initial validation with step-level RMSE
    print("\nInitial validation...")
    v0, v0_1d, v0_2d, v0_step = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats, step_checkpoints=STEP_CHECKPOINTS)
    print(f"  Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    for k, v in sorted(v0_step.items()):
        print(f"    step {k:3d}: 1d={v['1d']:.4f} 2d={v['2d']:.4f}")
    best_val = v0
    recorder.step(-1, phase="init", val_loss=v0, val_rmse_1d=v0_1d, val_rmse_2d=v0_2d,
                  step_rmse={str(k): v for k, v in v0_step.items()})
    recorder.save()

    if not args.skip_p1:
        # Phase 1: r=32
        print(f"\n--- Phase 1: r=32, lr={args.lr_p1:.0e}, {args.ep_p1}ep,"
              f" k_1d={args.k_1d}, k_2d={args.k_2d} ---")
        best_val = train_phase(
            model, train_events, val_events, config, norm_stats, std_1d, std_2d,
            device, cache_dir, rollout=32, lr=args.lr_p1, epochs=args.ep_p1,
            w_1d_scale=3.0, tag=f"{args.tag}_r32", best_val=best_val,
            aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
            per_node_stats=per_node_stats,
            node_weight_1d=nw_1d, node_weight_2d=nw_2d,
            k_1d=args.k_1d, k_2d=args.k_2d,
            recorder=recorder, phase_name="p1_r32",
            recess_w=args.recess_w,
            noise_std=args.noise_std, noise_curriculum=not args.no_curriculum)

        # Reload best Phase 1
        p1_path = os.path.join(cache_dir, f"best_model_2_{args.tag}_r32.pt")
        if os.path.exists(p1_path):
            model.load_state_dict(
                torch.load(p1_path, map_location=device, weights_only=False)["model_state_dict"])
            print("  Reloaded best Phase 1")

    # Phase 2
    print(f"\n--- Phase 2: r={args.rollout_p2}, lr={args.lr_p2:.0e}, {args.ep_p2}ep ---")
    best_val = train_phase(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=args.rollout_p2, lr=args.lr_p2, epochs=args.ep_p2,
        w_1d_scale=3.0, tag=f"{args.tag}_r64", best_val=best_val,
        aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        k_1d=args.k_1d, k_2d=args.k_2d,
        recorder=recorder, phase_name="p2_r64",
        recess_w=args.recess_w,
        noise_std=args.noise_std, noise_curriculum=not args.no_curriculum)

    recorder.save()
    print(f"\nDone. Best val: {best_val:.4f} (v54_rw2 baseline: {baseline_val:.4f})")
    print(f"  Break log: {recorder.log_dir / (recorder.run_name + '.json')}")


if __name__ == "__main__":
    main()
