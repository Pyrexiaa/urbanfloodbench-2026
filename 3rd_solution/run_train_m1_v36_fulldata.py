"""M1 v36: Full-data training (68 events → all for training).

v35 checkpoint → 全68 M1イベントで再訓練。
LBで評価 (M1 LB ≈ 0.004 → 0.002目標)。

Usage: python run_train_m1_v36_fulldata.py
"""
import sys, os, time, pickle, argparse
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_loader import load_model_config, load_event_data, list_events
from src.model import HeteroFloodGNNv4
from src.evaluation import compute_std_from_all_events

from run_train_m1_v35 import (
    train_phase_v35, validate_full_rollout, compute_node_weights,
    MODEL_ID, COUPLING_EDGE_DIM,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k_1d", type=int, default=32)
    parser.add_argument("--k_2d", type=int, default=8)
    parser.add_argument("--tag", type=str, default="v36_full")
    parser.add_argument("--base_ckpt", type=str, default="v35_r64",
                        help="Base M1 checkpoint (default: v35_r64)")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = "cuda"
    data_dir = os.path.join(BASE, "Dataset_Rerelease", "Models")
    cache_dir = os.path.join(BASE, "Models", "checkpoints")

    print("=" * 70)
    print(f"  M1 v36: Full-Data Training")
    print(f"  Base: {args.base_ckpt}")
    print(f"  k_1d={args.k_1d}, k_2d={args.k_2d}")
    print("=" * 70)

    config = load_model_config(data_dir, MODEL_ID)
    std_1d = compute_std_from_all_events(data_dir, MODEL_ID, "1d")
    std_2d = compute_std_from_all_events(data_dir, MODEL_ID, "2d")
    print(f"M1: {config.num_1d_nodes} 1D, {config.num_2d_nodes} 2D nodes")

    with open(os.path.join(cache_dir, f"norm_stats_model_{MODEL_ID}_v2.pkl"), "rb") as f:
        norm_stats = pickle.load(f)
    with open(os.path.join(cache_dir, f"per_node_stats_model_{MODEL_ID}.pkl"), "rb") as f:
        per_node_stats = pickle.load(f)

    nw_1d, nw_2d = compute_node_weights(per_node_stats)

    # 全データ: 全M1イベントを訓練 + in-sample val
    all_events = list_events(data_dir, MODEL_ID, "train")
    train_ids = list(all_events)
    val_ids = train_ids  # in-sample monitor
    print(f"Train: {len(train_ids)} (all events, in-sample val)")

    print("Loading events...")
    t0 = time.time()
    train_events = [load_event_data(data_dir, MODEL_ID, eid, config) for eid in train_ids]
    val_events = train_events  # same list (in-sample)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    model = HeteroFloodGNNv4(
        hidden_dim=128, num_processor_layers=4, noise_std=0.02,
        coupling_edge_dim=COUPLING_EDGE_DIM,
    ).to(device)

    # Load v35 checkpoint
    src_ckpt = os.path.join(cache_dir, f"best_model_1_{args.base_ckpt}.pt")
    ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    baseline_val = ckpt.get("best_val", "?")
    print(f"  Loaded: {args.base_ckpt} (val={baseline_val})")

    v0, v0_1d, v0_2d = validate_full_rollout(
        model, val_events, config, norm_stats, std_1d, std_2d, device,
        per_node_stats=per_node_stats)
    print(f"Initial val: {v0:.4f} (1d={v0_1d:.4f} 2d={v0_2d:.4f})")
    best_val = v0

    # Phase 1: r=32
    print(f"\n--- Phase 1: r=32, lr={args.lr}, {args.epochs}ep ---")
    best_val = train_phase_v35(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=32, lr=args.lr, epochs=args.epochs,
        tag=f"{args.tag}_r32", best_val=best_val,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        w_1d_scale=3.0, w_traj=0.3, w_traj_2d=0.1,
        k_1d=args.k_1d, k_2d=args.k_2d,
    )

    p1_ckpt = os.path.join(cache_dir, f"best_model_1_{args.tag}_r32.pt")
    if os.path.exists(p1_ckpt):
        ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Reloaded best Phase 1")

    # Phase 2: r=64
    lr2 = args.lr / 5
    print(f"\n--- Phase 2: r=64, lr={lr2:.1e}, 10ep ---")
    best_val = train_phase_v35(
        model, train_events, val_events, config, norm_stats, std_1d, std_2d,
        device, cache_dir, rollout=64, lr=lr2, epochs=10,
        tag=f"{args.tag}_r64", best_val=best_val,
        per_node_stats=per_node_stats,
        node_weight_1d=nw_1d, node_weight_2d=nw_2d,
        w_1d_scale=3.0, w_traj=0.3, w_traj_2d=0.1,
        k_1d=args.k_1d, k_2d=args.k_2d,
    )

    print(f"\n{'='*70}")
    print(f"  M1 v36 Full-Data Complete! Best val: {best_val:.4f} (in-sample)")
    print(f"  Baseline: {baseline_val}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
