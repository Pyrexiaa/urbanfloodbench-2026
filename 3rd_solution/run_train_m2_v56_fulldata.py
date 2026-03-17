"""v56: Full-data training (69 events, no hold-out).

v55d checkpoint → 全69イベントで再訓練。
17%データ増加で1step精度を向上させ、LBで評価。

Usage: python run_train_m2_v56_fulldata.py
       python run_train_m2_v56_fulldata.py --noise_std 0.005
"""
import sys, os, time, pickle, argparse
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import load_model_config, load_event_data, list_events
from src.model import HeteroFloodGNNv11
from src.evaluation import compute_std_from_all_events

from run_train_m2_v11c_multistep import (
    compute_flux_and_delta_stats, mask_2d_edges_only,
    MODEL_ID, FUTURE_RAIN_STEPS, COUPLING_EDGE_DIM, AUX_REL_WEIGHT,
)
from run_train_m2_v55_noise import (
    train_phase, validate_full_rollout, compute_node_weights,
    STEP_CHECKPOINTS,
)
from break_recorder import BreakRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="v56_full")
    parser.add_argument("--base_ckpt", type=str, default="v55d_n002_r64",
                        help="Base checkpoint tag (default: v55d)")
    parser.add_argument("--edge_w", type=float, default=0.001)
    parser.add_argument("--k_1d", type=int, default=32)
    parser.add_argument("--k_2d", type=int, default=16)
    parser.add_argument("--recess_w", type=float, default=2.0)
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Noise injection (0=off, 0.005-0.01 recommended)")
    parser.add_argument("--lr_p1", type=float, default=2e-5)
    parser.add_argument("--lr_p2", type=float, default=4e-6)
    parser.add_argument("--ep_p1", type=int, default=15)
    parser.add_argument("--ep_p2", type=int, default=15)
    # 全データモード: valなし
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Validation ratio (0.0=full data, 0.10=7 val events)")
    parser.add_argument("--skip_p1", action="store_true")
    # v64: step-zone-aware trajectory loss (break recorderで構造変化を検証)
    parser.add_argument("--step_zone", action="store_true",
                        help="Enable step-zone traj weight: t<100=1x, 100-200=1.5x, 200+=3x")
    parser.add_argument("--sz_mid", type=float, default=1.5, help="Zone multiplier for t=100-199")
    parser.add_argument("--sz_late", type=float, default=3.0, help="Zone multiplier for t>=200")
    # v66: deg1ノード (末端, 48.5%) のloss重みブースト
    parser.add_argument("--deg1_w", type=float, default=1.0,
                        help="Loss weight multiplier for degree-1 nodes (default 1.0=no boost)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = "cuda"
    data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
    cache_dir = os.path.join(BASE, "Models", "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print(f"  v56: Full-Data Training (seed={args.seed})")
    print(f"  Base: {args.base_ckpt}")
    print(f"  val_ratio={args.val_ratio}, noise_std={args.noise_std}")
    print(f"  k_1d={args.k_1d}, k_2d={args.k_2d}")
    # v64: step-zone weight dict
    step_zone_w = None
    if args.step_zone:
        step_zone_w = {100: args.sz_mid, 200: args.sz_late}
        print(f"  step_zone_w={step_zone_w}")
    if args.deg1_w != 1.0:
        print(f"  deg1_w={args.deg1_w}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"M2: {config.num_1d_nodes} 1D, {config.num_2d_nodes} 2D nodes")

    # v2 break recorder: 1Dノードのdegree別グループ
    from src.model_lstm1d import build_adjacency
    _, _, degree_1d = build_adjacency(config.edge_index_1d, config.num_1d_nodes)
    node_groups_1d = {
        "deg1": np.where(degree_1d == 1)[0],
        "deg2": np.where(degree_1d == 2)[0],
        "hub": np.where(degree_1d >= 3)[0],
    }
    print(f"  1D node groups: deg1={len(node_groups_1d['deg1'])}, "
          f"deg2={len(node_groups_1d['deg2'])}, hub={len(node_groups_1d['hub'])}")

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    nw_1d, nw_2d = compute_node_weights(per_node_stats)

    # v66: deg1ノードのloss重みブースト (MSE寄与58% vs 人口48.5%)
    if args.deg1_w != 1.0:
        nw_1d[node_groups_1d["deg1"]] *= args.deg1_w
        print(f"  deg1 node weight boosted: ×{args.deg1_w} ({len(node_groups_1d['deg1'])} nodes)")

    # === データ分割 ===
    all_events = list_events(data_dir, MODEL_ID, "train")
    if args.val_ratio > 0:
        n_val = max(1, int(len(all_events) * args.val_ratio))
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(all_events)
        val_ids, train_ids = shuffled[:n_val].tolist(), shuffled[n_val:].tolist()
    else:
        # 全データ訓練: 69イベント全て訓練 + in-sample valとしても使用
        train_ids = all_events.tolist() if hasattr(all_events, 'tolist') else list(all_events)
        val_ids = train_ids  # in-sample monitor (真の評価はLB)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)} {'(in-sample)' if args.val_ratio == 0 else ''}")

    print("Loading events...")
    t0 = time.time()
    train_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in train_ids]
    if args.val_ratio > 0:
        val_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in val_ids]
    else:
        val_events = train_events  # in-sample monitor
    print(f"  Loaded in {time.time()-t0:.1f}s")

    fstats = compute_flux_and_delta_stats(train_events)
    wl_avg_var = (fstats["wl_1d_delta_std"]**2 + fstats["wl_2d_delta_std"]**2) / 2
    aux_w_inlet = AUX_REL_WEIGHT * wl_avg_var / (fstats["inlet_abs_std"]**2)
    aux_w_edge = args.edge_w

    model = HeteroFloodGNNv11(
        hidden_dim=128, num_processor_layers=4, noise_std=0.0,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)

    ckpt_path = os.path.join(cache_dir, f"best_model_2_{args.base_ckpt}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_val = ckpt.get("best_val", float("inf"))
    print(f"  Loaded {args.base_ckpt}: val={baseline_val:.4f}")

    # val_eventsが空の場合: 各epoch終了時にcheckpoint保存 (best_val tracking不可)
    # train_phaseを少し修正する必要がある → val_eventsが空ならvalidation skipして毎epoch保存

    # --- Break Recorder ---
    recorder = BreakRecorder(
        log_dir=os.path.join(BASE, "break_logs"),
        run_name=f"{args.tag}_s{args.seed}")
    recorder.set_meta(
        version="v56_fulldata", model="HeteroFloodGNNv11",
        hidden_dim=128, num_processor_layers=4, seed=args.seed,
        baseline_val=baseline_val, base_ckpt=args.base_ckpt,
        edge_w=args.edge_w, k_1d=args.k_1d, k_2d=args.k_2d,
        recess_w=args.recess_w, noise_std=args.noise_std,
        lr_p1=args.lr_p1, lr_p2=args.lr_p2,
        val_ratio=args.val_ratio, n_train=len(train_ids), n_val=len(val_ids),
        step_zone_w=step_zone_w, deg1_w=args.deg1_w)

    print("\nInitial validation...")
    v0, v0_1d, v0_2d, v0_step = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats, step_checkpoints=STEP_CHECKPOINTS,
        node_groups_1d=node_groups_1d)
    print(f"  Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    best_val = v0

    # epoch -1: init step_rmse記録 + similarity
    init_step_rmse = {str(k): v for k, v in v0_step.items()}
    sim_info = recorder.compute_similarity(init_step_rmse)
    recorder.step(-1, phase="init", val_loss=v0, val_rmse_1d=v0_1d, val_rmse_2d=v0_2d,
                  step_rmse=init_step_rmse, similarity=sim_info)
    if sim_info["closest_run"]:
        print(f"  Similarity: closest={sim_info['closest_run']} "
              f"cos={sim_info['cosine_sim']:.4f} pattern={sim_info['pattern']}")
    recorder.save()

    if not args.skip_p1:
        print(f"\n--- Phase 1: r=32, lr={args.lr_p1:.0e}, {args.ep_p1}ep ---")
        best_val = train_phase(
            model, train_events, val_events, config, norm_stats, std_1d, std_2d,
            device, cache_dir, rollout=32, lr=args.lr_p1, epochs=args.ep_p1,
            w_1d_scale=3.0, tag=f"{args.tag}_r32", best_val=best_val,
            aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
            per_node_stats=per_node_stats,
            node_weight_1d=nw_1d, node_weight_2d=nw_2d,
            k_1d=args.k_1d, k_2d=args.k_2d,
            recess_w=args.recess_w,
            noise_std=args.noise_std, noise_curriculum=True,
            recorder=recorder, phase_name="p1_r32",
            step_zone_w=step_zone_w,
            node_groups_1d=node_groups_1d)

        p1_path = os.path.join(cache_dir, f"best_model_2_{args.tag}_r32.pt")
        if os.path.exists(p1_path):
            model.load_state_dict(
                torch.load(p1_path, map_location=device, weights_only=False)["model_state_dict"])
            print("  Reloaded best Phase 1")

    print(f"\n--- Phase 2: r=64, lr={args.lr_p2:.0e}, {args.ep_p2}ep ---")
    best_val_p2 = train_phase(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=64, lr=args.lr_p2, epochs=args.ep_p2,
        w_1d_scale=3.0, tag=f"{args.tag}_r64", best_val=best_val,
        aux_w_inlet=aux_w_inlet, aux_w_edge=aux_w_edge,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        k_1d=args.k_1d, k_2d=args.k_2d,
        recess_w=args.recess_w,
        noise_std=args.noise_std, noise_curriculum=True,
        recorder=recorder, phase_name="p2_r64",
        step_zone_w=step_zone_w,
        node_groups_1d=node_groups_1d)

    recorder.save()
    print(f"  Break log: {recorder.log_dir / (recorder.run_name + '.json')}")

    print(f"\n{'='*70}")
    print(f"  v56 Full-Data Training Complete!")
    print(f"  Baseline val (v55d): {baseline_val:.4f}")
    print(f"  Best val: {best_val_p2:.4f} {'(in-sample)' if args.val_ratio == 0 else ''}")
    print(f"  Checkpoints: {cache_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
