#!/usr/bin/env python
"""Full training script for UrbanFloodNet with model and normalization checkpointing."""

import os
import sys
import json
import time
import pickle
import random
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import wandb

# Setup paths (works whether launched via train.py wrapper or directly)
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import get_recurrent_dataloader, get_model_config, make_x_dyn, NonBatchableGraph
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL

# Configuration
CONFIG = {
    'history_len': 10,
    'forecast_len': 256 if SELECTED_MODEL == 'Model_2' else 128,  # Max rollout horizon
    'batch_size': 24,
    'epochs': 62 if SELECTED_MODEL == 'Model_2' else 32,  # Model_2: 6@h1+4@h2..256 (62); Model_1: 4@h1..128 (32)
    'lr': 1e-3,
    'lr_final': 10**-4.5,          # ~3.16e-5; log-linear decay over all epochs (1.5 decades)
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'save_dir': 'checkpoints',
    'checkpoint_interval': 1,   # Save every N epochs
    'clip_norm': 1.0,              # Max gradient norm for clipping
    'early_stopping_patience': 5,  # Only active once max_h == forecast_len; None to disable
}

# Kaggle sigmas — used only for logging RMSE in meters and approx Kaggle score.
# Water level is normalized by these sigmas (meanstd), so sqrt(MSE_norm) == NRMSE directly.
KAGGLE_SIGMA = {
    (1, 1): 16.878,  # Model_1, 1D nodes
    (1, 2): 14.379,  # Model_1, 2D nodes
    (2, 1):  3.192,  # Model_2, 1D nodes
    (2, 2):  2.727,  # Model_2, 2D nodes
}

def save_normalization_stats(norm_stats, save_dir, model_id=None):
    """Save model-specific normalization statistics to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use model ID from SELECTED_MODEL if not provided
    if model_id is None:
        model_id = SELECTED_MODEL
    
    # Extract serializable components
    stats_to_save = {
        'static_1d_params': norm_stats.get('static_1d_params', {}),
        'static_2d_params': norm_stats.get('static_2d_params', {}),
        'dynamic_1d_params': norm_stats.get('dynamic_1d_params', {}),
        'dynamic_2d_params': norm_stats.get('dynamic_2d_params', {}),
        'node1d_cols': norm_stats.get('node1d_cols', []),
        'node2d_cols': norm_stats.get('node2d_cols', []),
        'edge1_cols': norm_stats.get('edge1_cols', []),
        'edge2_cols': norm_stats.get('edge2_cols', []),
        'feature_type_1d': norm_stats.get('feature_type_1d', {}),
        'feature_type_2d': norm_stats.get('feature_type_2d', {}),
    }
    
    # Convert torch tensors to lists for JSON serialization
    for key in ['oneD_mu', 'oneD_sigma', 'twoD_mu', 'twoD_sigma', 
                'edge1_mu', 'edge1_sigma', 'edge2_mu', 'edge2_sigma']:
        if key in norm_stats and isinstance(norm_stats[key], torch.Tensor):
            stats_to_save[key] = norm_stats[key].cpu().numpy().tolist()
    
    # Save as JSON
    stats_path = os.path.join(save_dir, f'{model_id}_normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    # Also save as pickle for full FeatureNormalizer objects (optional, for reference)
    normalizer_path = os.path.join(save_dir, f'{model_id}_normalizers.pkl')
    normalizers_data = {
        'normalizer_1d': norm_stats.get('normalizer_1d'),
        'normalizer_2d': norm_stats.get('normalizer_2d'),
    }
    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizers_data, f)
    
    print(f"[INFO] Saved normalization statistics to {stats_path}")
    print(f"[INFO] Saved normalizer objects to {normalizer_path}")

def save_checkpoint(model, epoch, loss, save_dir, config, model_id=None, global_step=None, scheduler=None, optimizer=None):
    """Save model checkpoint and related information."""
    os.makedirs(save_dir, exist_ok=True)

    # Use model ID from SELECTED_MODEL if not provided
    if model_id is None:
        model_id = SELECTED_MODEL

    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'config': config,
        'model_arch_config': {
            'h_dim': model.cell._h_dim,  # may be dict (per node type) or int
            'msg_dim': model.cell.msg_dim,
            'hidden_dim': model.cell._hidden_dim,
            'num_1d_extra_hops': model.cell.num_1d_extra_hops,
            'node_dyn_input_dims': CONFIG.get('node_dyn_input_dims'),
        },
        'loss': loss,
        'model_id': model_id,
        'global_step': global_step,
        'wandb_run_id': wandb.run.id if wandb.run is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
    }
    
    checkpoint_path = os.path.join(save_dir, f'{model_id}_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def evaluate_rollout(model, dataloader, criterion, device, norm_stats, rollout_steps, batched_static_graph=None, max_batches=None, use_mixed_precision=False, rain_1d_index=None):
    """Evaluate at a fixed multi-step rollout horizon.

    Returns (combined_norm, 1d_norm, 2d_norm, per_node_1d_mse) where
    per_node_1d_mse is a 1-D numpy array of shape [N_1d] with per-node
    mean MSE (normalized) averaged across all batches and timesteps.
    """
    model.eval()
    total, total_1d, total_2d = 0.0, 0.0, 0.0
    per_node_1d_accum = None  # [N_1d] accumulated MSE sum
    n = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch is None:
                continue
            avail = batch['y_future_1d'].shape[1]
            h = min(rollout_steps, avail)
            static_graph = batch['static_graph'].to(device)
            y_hist_1d     = batch['y_hist_1d'].to(device)
            y_hist_2d     = batch['y_hist_2d'].to(device)
            rain_hist_2d  = batch['rain_hist_2d'].to(device)
            y_future_1d   = batch['y_future_1d'].to(device)
            y_future_2d   = batch['y_future_2d'].to(device)
            rain_future_2d = batch['rain_future_2d'].to(device)
            _r1d = rain_1d_index if rain_1d_index is not None else getattr(static_graph, 'rain_1d_index', None)
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_1d=y_hist_1d, y_hist_2d=y_hist_2d,
                    rain_hist=rain_hist_2d, rain_future=rain_future_2d,
                    make_x_dyn=lambda y, r, d, _r=_r1d: make_x_dyn(
                        y['oneD'], y['twoD'], r, d,
                        rain_1d_index=_r,
                    ),
                    rollout_steps=h, device=device,
                    batched_data=batched_static_graph,
                )
                loss_1d = criterion(predictions['oneD'], y_future_1d[:, :h])
                loss_2d = criterion(predictions['twoD'], y_future_2d[:, :h])
                # Per-node 1D MSE: mean over batch (B) and time (h), keep node dim
                # predictions['oneD']: [B, h, N_1d, 1], y_future_1d: [B, h, N_1d, 1]
                node_mse_1d = ((predictions['oneD'] - y_future_1d[:, :h]) ** 2).mean(dim=(0, 1, 3))  # [N_1d]
            total_1d += loss_1d.item()
            total_2d += loss_2d.item()
            total += ((loss_1d + loss_2d) / 2).item()
            node_mse_cpu = node_mse_1d.float().cpu()
            if per_node_1d_accum is None:
                per_node_1d_accum = node_mse_cpu
            else:
                per_node_1d_accum += node_mse_cpu
            n += 1
    if n == 0:
        return float('nan'), float('nan'), float('nan'), None
    per_node_1d_mse = (per_node_1d_accum / n).numpy()
    return total / n, total_1d / n, total_2d / n, per_node_1d_mse


def evaluate_full_event_rollout(model, val_event_file_list, data, norm_stats, static_graph, device,
                                history_len=10, use_mixed_precision=False, rain_1d_index=None,
                                n_rain_channels=None):
    """Run full autoregressive rollout (t=0 to T) on each val event and compute NRMSE.

    Mirrors the inference loop exactly — warm-starts on history_len true timesteps,
    then predicts autoregressively for the rest of the event.

    Returns (combined_nrmse, nrmse_1d, nrmse_2d, per_event_results) averaged over all val events.
    per_event_results: list of dicts with keys 'event_name', 'combined_nrmse', 'nrmse_1d', 'nrmse_2d', 'T',
        't_worst' (timestep of max combined error), 't_worst_frac' (as fraction of T),
        'rain_total', 'rain_peak', 't_rain_peak_frac', 'rain_at_worst' (spatially-averaged rain channel 0)
    """
    from autoregressive_inference import prepare_event_tensors, autoregressive_rollout_both

    model.eval()
    total_1d, total_2d = 0.0, 0.0
    n_events = 0
    per_event_results = []

    with torch.no_grad():
        for _, event_path, _ in val_event_file_list:
            try:
                from autoregressive_inference import load_event_data
                node_1d, node_2d = load_event_data(event_path)
                y1_all, y2_all, rain2_all, _, _, _ = prepare_event_tensors(node_1d, node_2d, norm_stats, device)
                if n_rain_channels is not None:
                    rain2_all = rain2_all[:, :, :n_rain_channels]
                T = y1_all.size(0)
                if T <= history_len:
                    continue
                y1_hist = y1_all[:history_len]   # [H, N1, 1]
                y2_hist = y2_all[:history_len]   # [H, N2, 1]
                with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                    pred_1d, pred_2d = autoregressive_rollout_both(
                        model, static_graph, y1_hist, y2_hist, rain2_all, device, history_len=history_len
                    )
                # pred_1d: [T-H, N1, 1], y1_all[H:]: [T-H, N1, 1]
                T_pred = T - history_len
                # Per-timestep MSE: [T_pred]
                err_1d_t = ((pred_1d - y1_all[history_len:]) ** 2).mean(dim=(1, 2))  # [T_pred]
                err_2d_t = ((pred_2d - y2_all[history_len:]) ** 2).mean(dim=(1, 2))
                mse_1d = err_1d_t.mean().item()
                mse_2d = err_2d_t.mean().item()
                total_1d += mse_1d
                total_2d += mse_2d
                n_events += 1

                # Combined per-timestep instantaneous NRMSE
                combined_t = (err_1d_t ** 0.5 + err_2d_t ** 0.5) / 2  # [T_pred]
                # t_worst = timestep where error grows fastest (largest single-step jump)
                if T_pred > 1:
                    delta_t = combined_t[1:] - combined_t[:-1]  # [T_pred-1]
                    t_worst = delta_t.argmax().item() + 1        # +1 since diff is shifted
                else:
                    t_worst = 0
                t_worst_frac = t_worst / max(T_pred - 1, 1)

                # Rainfall stats over prediction window (rain2_all: [T, N_rain, C])
                rain_pred = rain2_all[history_len:, :, 0]  # [T_pred, N_rain] — first rain channel
                rain_mean_t = rain_pred.mean(dim=1)        # [T_pred] — spatial mean per timestep
                rain_total = rain_mean_t.sum().item()
                rain_peak = rain_mean_t.max().item()
                t_rain_peak = rain_mean_t.argmax().item()
                t_rain_peak_frac = t_rain_peak / max(T_pred - 1, 1)
                rain_at_worst = rain_mean_t[t_worst].item()

                event_name = Path(str(event_path)).name
                per_event_results.append({
                    'event_name': event_name,
                    'nrmse_1d': mse_1d ** 0.5,
                    'nrmse_2d': mse_2d ** 0.5,
                    'combined_nrmse': (mse_1d ** 0.5 + mse_2d ** 0.5) / 2,
                    'T': T_pred,
                    't_worst': t_worst,
                    't_worst_frac': round(t_worst_frac, 3),
                    'rain_total': round(rain_total, 4),
                    'rain_peak': round(rain_peak, 4),
                    't_rain_peak_frac': round(t_rain_peak_frac, 3),
                    'rain_at_worst': round(rain_at_worst, 4),
                })
            except Exception as e:
                print(f"[WARNING] Full-event val failed for {event_path}: {e}")

    if n_events == 0:
        return float('nan'), float('nan'), float('nan'), []
    nrmse_1d = (total_1d / n_events) ** 0.5
    nrmse_2d = (total_2d / n_events) ** 0.5
    return (nrmse_1d + nrmse_2d) / 2, nrmse_1d, nrmse_2d, per_event_results


def train(resume_from=None, use_mixed_precision=False, skip_validation=False, pretrain_from=None, train_split='train', extra_epochs=None, mirror_latest=True, custom_curriculum=None, keep_short_events=False):
    """Main training loop.
    
    Args:
        resume_from: Path to checkpoint directory to resume from
        use_mixed_precision: Whether to use mixed precision (float16) training
    """
    
    # Determine if resuming
    resume_path = None
    resume_specific_file = None  # Set when --resume points to a .pt file directly
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise ValueError(f"Resume checkpoint not found: {resume_path}")
        if resume_path.suffix == '.pt':
            # Specific file provided — use it directly
            resume_specific_file = resume_path
            resume_path = resume_path.parent
        print(f"\n[INFO] Resuming training from: {resume_specific_file or resume_path}")
    print("\n" + "="*70)
    print("UrbanFloodNet Training Script")
    print("="*70)

    # Peek at checkpoint to recover wandb run ID and global_step before init
    _wandb_resume_id = None
    _global_step_resume = 0
    if resume_path:
        try:
            if resume_specific_file:
                _peek_file = resume_specific_file
            else:
                _ckpt_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
                _peek_file = _ckpt_files[-1] if _ckpt_files else None
            if _peek_file:
                _peek = torch.load(_peek_file, map_location='cpu')
                _wandb_resume_id = _peek.get('wandb_run_id', None)
                _global_step_resume = _peek.get('global_step', 0) or 0
        except Exception:
            pass

    run_name = f"{SELECTED_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if _wandb_resume_id:
        wandb.init(project="UrbanFloodNet", id=_wandb_resume_id, resume="must", config=CONFIG)
        print(f"[INFO] Resuming wandb run: {_wandb_resume_id} (global_step offset: {_global_step_resume})")
    else:
        wandb.init(project="UrbanFloodNet", name=run_name, config=CONFIG)

    # Create a dated run subdirectory so each training run is isolated.
    # Also maintain a shared checkpoints/latest/ directory that always contains
    # the most recent best checkpoint + normalizers for EVERY model, so inference
    # can find Model_1 and Model_2 files in one place regardless of training order.
    run_dir = os.path.join(CONFIG['save_dir'], run_name)
    os.makedirs(run_dir, exist_ok=True)
    latest_dir = os.path.join(CONFIG['save_dir'], 'latest')
    os.makedirs(latest_dir, exist_ok=True)
    print(f"[INFO] Checkpoint dir: {run_dir}")
    print(f"[INFO] Latest dir:     {latest_dir}")

    device = torch.device(CONFIG['device'])
    print(f"[INFO] Device: {device}")

    # Initialize data and fetch preprocessing
    print(f"\n[INFO] Initializing data...")
    data = initialize_data()
    norm_stats = data['norm_stats']

    # Save normalization statistics into the run directory
    print(f"[INFO] Saving normalization statistics...")
    save_normalization_stats(norm_stats, run_dir)
    
    # Create dataloaders for train, val, test
    print(f"\n[INFO] Creating dataloaders...")
    print(f"  History: {CONFIG['history_len']}, Forecast: {CONFIG['forecast_len']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Split: train (data leakage prevention: normalization computed on train only)")
    
    train_dataloader = get_recurrent_dataloader(
        history_len=CONFIG['history_len'],
        forecast_len=CONFIG['forecast_len'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        split=train_split,
        keep_short_events=keep_short_events,
    )
    print(f"  Training split: {train_split}")
    if keep_short_events:
        print(f"  Keep short events: ENABLED (short events train at their max available horizon)")
    if skip_validation:
        val_dataloader = None
        print(f"  Validation: DISABLED (--no-val)")
    else:
        val_dataloader = get_recurrent_dataloader(
            history_len=CONFIG['history_len'],
            forecast_len=CONFIG['forecast_len'],
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            split='val',
        )
    
    # Get model config
    print(f"\n[INFO] Getting model configuration...")
    model_config = get_model_config()
    print(f"  Node types: {model_config['node_types']}")
    print(f"  Node static dims: {model_config['node_static_dims']}")
    print(f"  Node dynamic dims: {model_config['node_dyn_input_dims']}")
    CONFIG['node_dyn_input_dims'] = model_config['node_dyn_input_dims']
    _n_rain_channels = model_config['node_dyn_input_dims']['twoD'] - 1  # twoD = wl + rain_channels
    
    # Initialize model
    print(f"\n[INFO] Building model...")
    model_config.update({
        # Per-node-type GRU hidden size: Model_2 gives 1D nodes extra capacity
        # since its 190-node channel network is the hard bottleneck (order-of-mag harder).
        # Model_1 uses scalar 96 (17 nodes, shared dim is fine).
        'h_dim': {'oneD': 192, 'twoD': 96, 'global': 32} if SELECTED_MODEL == 'Model_2' else 96,
        'msg_dim': 64,
        # Edge-type-specific hidden dims (reduced from 96/192 to 64/128)
        'hidden_dim': {
            'oneDedge':    64,
            'oneDedgeRev': 64,
            'twoDedge':    128,
            'twoDedgeRev': 128,
            # Cross-type edges: Model_2 uses StaticDynamicEdgeMP with only ~170 edges
            # and 2 edge features — hidden_dim=32 is sufficient.
            # Model_1 uses GATv2CrossTypeMP which ignores hidden_dim entirely.
            'twoDoneD':    32,
            'oneDtwoD':    32,
        },
        # Model_2 has 190 1D nodes (vs 17 for Model_1) — extra hops propagate
        # information further along the larger channel network each timestep.
        'num_1d_extra_hops': 4 if SELECTED_MODEL == 'Model_2' else 0,
    })
    model = FloodAutoregressiveHeteroModel(**model_config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    wandb.watch(model, log='all', log_freq=50)  # logs gradients + weights every 50 steps
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()

    # Log-linear LR decay: lr_init → lr_final over CONFIG['epochs'] epochs
    # At epoch e (1-indexed), multiplier = (lr_final/lr_init)^((e-1)/(N-1))
    _lr_ratio = CONFIG['lr_final'] / CONFIG['lr']
    _total_epochs = CONFIG['epochs']
    def _lr_lambda(epoch_0indexed):
        if _total_epochs <= 1:
            return 1.0
        return _lr_ratio ** (epoch_0indexed / (_total_epochs - 1))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Mixed precision: GradScaler for stable fp16 backward pass
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None

    # Load checkpoint if resuming
    start_epoch = 1
    if resume_path:
        try:
            # Use specific file if provided, otherwise pick latest in directory
            if resume_specific_file:
                checkpoint_path = resume_specific_file
            else:
                checkpoint_files = sorted(resume_path.glob(f'{SELECTED_MODEL}*.pt'))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoints found for {SELECTED_MODEL} in {resume_path}")
                checkpoint_path = checkpoint_files[-1]
            print(f"[INFO] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Restore model and optimizer states
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                print(f"[INFO] Model weights restored")
            
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"[INFO] Optimizer state restored")

            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                # base_lrs is saved in scheduler state from the original run.
                # After load, base_lrs may reflect the old lr (e.g. 1e-3) — override
                # with the current CONFIG lr so scheduler.step() doesn't jump to old scale.
                scheduler.base_lrs = [CONFIG['lr']] * len(scheduler.base_lrs)
                print(f"[INFO] Scheduler state restored (base_lrs overridden to {CONFIG['lr']:.2e})")
            
            # Get start epoch from checkpoint metadata
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Resuming from epoch {start_epoch} (last saved epoch: {checkpoint['epoch']})")
            
            print(f"[INFO] Checkpoint successfully loaded")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            print(f"[WARNING] Starting fresh from epoch 1")
            start_epoch = 1

    # If extra_epochs is set (used by fine-tune pipeline), override CONFIG['epochs'] so that
    # exactly extra_epochs epochs run starting from start_epoch, regardless of what was
    # stored in CONFIG. This prevents the common mistake of --epochs 4 meaning "stop at epoch 4"
    # when start_epoch is already 25.
    if extra_epochs is not None:
        CONFIG['epochs'] = start_epoch - 1 + extra_epochs
        print(f"[INFO] extra_epochs={extra_epochs}: running epochs {start_epoch}..{CONFIG['epochs']}")

    # Transfer learning: load weights from another model (e.g. Model_1 -> Model_2)
    # Resets epoch + optimizer — fresh training schedule with warm weights.
    if pretrain_from:
        pretrain_path = Path(pretrain_from)
        # Prefer best, then any .pt
        for candidate in [
            pretrain_path / 'Model_1_best.pt',
            *sorted(pretrain_path.glob('Model_1*.pt')),
        ]:
            if candidate.exists():
                print(f"[INFO] Transfer learning: loading weights from {candidate}")
                ckpt = torch.load(candidate, map_location=device)
                if 'model_state' in ckpt:
                    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
                    if missing:
                        print(f"[INFO]   Missing keys (randomly initialized): {len(missing)}")
                    if unexpected:
                        print(f"[INFO]   Unexpected keys (ignored): {len(unexpected)}")
                    print(f"[INFO] Transfer weights loaded (epoch counter reset to 1, fresh optimizer)")
                break
        else:
            print(f"[WARNING] No Model_1 checkpoint found in {pretrain_from} — training from scratch")

    # Water level is normalized by kaggle_sigma (meanstd), so sqrt(MSE_norm) == NRMSE directly.
    # Loss = (loss_1d + loss_2d) / 2 — equal weight, directly Kaggle-aligned.
    model_id = int(SELECTED_MODEL.split('_')[-1])
    kaggle_sigma_1d = KAGGLE_SIGMA[(model_id, 1)]
    kaggle_sigma_2d = KAGGLE_SIGMA[(model_id, 2)]
    wl_1d = norm_stats['dynamic_1d_params']['water_level']
    wl_2d = norm_stats['dynamic_2d_params']['water_level']
    print(f"[INFO] Loss: (loss_1d + loss_2d) / 2  — water_level normalized by kaggle_sigma, so MSE_norm = NRMSE²")
    print(f"  1D: mean={wl_1d['mean']:.3f}m, sigma={wl_1d['sigma']:.3f}m (kaggle_σ={kaggle_sigma_1d})")
    print(f"  2D: mean={wl_2d['mean']:.3f}m, sigma={wl_2d['sigma']:.3f}m (kaggle_σ={kaggle_sigma_2d})")

    # Pre-build batched static graphs once — reused every forward pass to eliminate
    # per-batch CPU overhead from Batch.from_data_list.
    print(f"\n[INFO] Pre-building batched static graphs (B={CONFIG['batch_size']})...")
    # Ensure window index uses forecast_len before the first batch is drawn,
    # so all samples in the first batch have uniform shape.
    train_dataloader.dataset.set_min_future(CONFIG['forecast_len'])
    if val_dataloader is not None and hasattr(val_dataloader.dataset, 'set_min_future'):
        val_dataloader.dataset.set_min_future(CONFIG['forecast_len'])
    _static_graph_cpu = next(iter(train_dataloader))['static_graph']
    train_batched_graph = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device)
    val_batched_graph   = model._make_batched_graph(_static_graph_cpu, CONFIG['batch_size']).to(device) if not skip_validation else None
    _single_static_graph = _static_graph_cpu  # used by full-event rollout val (B=1 inference)
    _val_event_file_list = data.get('val_event_file_list', [])
    # Capture rain_1d_index on device for use in make_x_dyn closures (graph-level attr
    # may not survive Batch.from_data_list so we pin it to a closure variable instead).
    _rain_1d_index = _static_graph_cpu.rain_1d_index.to(device) if hasattr(_static_graph_cpu, 'rain_1d_index') else None
    print(f"[INFO] Batched graphs ready.")

    print(f"\n[INFO] Training configuration:")
    print(f"  Learning rate: {CONFIG['lr']} → {CONFIG['lr_final']:.2e} (log-linear over {CONFIG['epochs']} epochs)")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Checkpoint interval: {CONFIG['checkpoint_interval']}")

    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}\n")

    epoch_start_time = time.time()
    global_step = _global_step_resume  # Restored from checkpoint on resume; 0 for fresh runs

    best_kaggle_at_max_h = float('inf')   # best (NRMSE_1D + NRMSE_2D)/2 seen at full horizon
    best_kaggle_epoch = None

    no_improve_count = 0  # Early stopping counter (only active at full horizon)
    best_train_loss_at_max_h = float('inf')  # fallback for early stopping when val disabled
    val_loss_norm = None  # Set in epoch loop; initialized here to avoid UnboundLocalError if loop is empty

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_start_time = time.time()

        # Curriculum schedule
        # Model_1: 4 epochs per stage: 1→2→4→8→16→32→64→128 (32 total)
        # Model_2: 6@h1 then 4 epochs per stage: 2→4→6→8→12→16→24→32→48→64→96→128→196→256 (62 total)
        if custom_curriculum is not None:
            # custom_curriculum: list of (horizon, n_epochs) built relative to start_epoch
            # _cc_boundaries[i] = first epoch of the NEXT stage (exclusive upper bound)
            _cc_boundaries = []
            _cc_horizons = []
            _ep = start_epoch
            for (h, n) in custom_curriculum:
                _ep += n
                _cc_boundaries.append(_ep)
                _cc_horizons.append(h)
            _stage_idx = next((i for i, b in enumerate(_cc_boundaries) if epoch < b), len(_cc_boundaries) - 1)
            max_h = _cc_horizons[_stage_idx]
        elif SELECTED_MODEL == 'Model_2':
            _m2_boundaries = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]  # last epoch of each stage; 62 total
            _m2_horizons   = [1,  2,  4,  6,  8, 12, 16, 24, 32, 48, 64, 96, 128, 196, 256]
            _stage_idx = next((i for i, b in enumerate(_m2_boundaries) if epoch <= b), len(_m2_boundaries) - 1)
            max_h = _m2_horizons[_stage_idx]
        else:
            _m1_boundaries = [4, 8, 12, 16, 20, 24, 28, 32]  # last epoch of each stage; 32 total
            _m1_horizons   = [1, 2,  4,  8, 16, 32, 64, 128]
            _stage_idx = next((i for i, b in enumerate(_m1_boundaries) if epoch <= b), len(_m1_boundaries) - 1)
            max_h = _m1_horizons[_stage_idx]

        # Epoch boundary marker — visible as a vertical annotation in wandb
        wandb.log({'epoch': epoch, 'curriculum/max_h': max_h}, step=global_step)

        # Update window index to only include windows with >= max_h future steps.
        # This ensures short events contribute at low horizons but drop out at higher ones.
        _eff_min_future = max(1, max_h)
        if not hasattr(train_dataloader.dataset, '_last_min_future') or \
                train_dataloader.dataset._last_min_future != _eff_min_future:
            train_dataloader.dataset.set_min_future(_eff_min_future)
            train_dataloader.dataset._last_min_future = _eff_min_future
            n_windows = len(train_dataloader.dataset._window_index)
            print(f"[INFO] Window index updated: min_future={_eff_min_future}, "
                  f"windows={n_windows}")
            if keep_short_events and n_windows > 0 and len(train_dataloader.dataset._window_index[0]) == 3:
                from collections import Counter
                fut_counts = Counter(f for _, _, f in train_dataloader.dataset._window_index)
                for fut in sorted(fut_counts):
                    print(f"  h={fut}: {fut_counts[fut]} windows")
            # Also update val dataloader so val windows match the current horizon
            if val_dataloader is not None and hasattr(val_dataloader.dataset, 'set_min_future'):
                val_dataloader.dataset.set_min_future(_eff_min_future)
                val_dataloader.dataset._last_min_future = _eff_min_future

        print(f"[INFO] Curriculum: epoch={epoch}/{CONFIG['epochs']}, max_h={max_h}")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            # Extract batch data
            static_graph = batch['static_graph'].to(device, non_blocking=True)
            y_hist_1d = batch['y_hist_1d'].to(device, non_blocking=True)        # [B, H, N_1d, 1]
            y_hist_2d = batch['y_hist_2d'].to(device, non_blocking=True)        # [B, H, N_2d, 1]
            rain_hist_2d = batch['rain_hist_2d'].to(device, non_blocking=True)  # [B, H, N_2d, R]
            y_future_1d = batch['y_future_1d'].to(device, non_blocking=True)    # [B, T, N_1d, 1]
            y_future_2d = batch['y_future_2d'].to(device, non_blocking=True)    # [B, T, N_2d, 1]
            rain_future_2d = batch['rain_future_2d'].to(device, non_blocking=True)  # [B, T, N_2d, R]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # When keep_short_events is active, the batch carries its own
                # rollout_steps (may be < max_h for short events).
                rollout_steps = batch.get('rollout_steps', max_h)
                predictions = model.forward_unroll(
                    data=static_graph,
                    y_hist_1d=y_hist_1d,
                    y_hist_2d=y_hist_2d,
                    rain_hist=rain_hist_2d,
                    rain_future=rain_future_2d,
                    make_x_dyn=lambda y, r, d: make_x_dyn(
                        y['oneD'], y['twoD'], r, d,
                        rain_1d_index=_rain_1d_index,
                    ),
                    rollout_steps=rollout_steps,
                    device=device,
                    batched_data=train_batched_graph,
                    use_grad_checkpoint=use_mixed_precision,
                )
                loss_1d = criterion(predictions['oneD'], y_future_1d[:, :rollout_steps])
                loss_2d = criterion(predictions['twoD'], y_future_2d[:, :rollout_steps])
                loss = (loss_1d + loss_2d) / 2

            # Backward pass (scaler handles fp16 gradient scaling when enabled)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # must unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['clip_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['clip_norm'])
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            avg_loss = epoch_loss / num_batches

            # Build the wandb log dict for this step — always includes train loss
            log_dict = {
                'loss/train_batch': loss.item(),
                'loss/train_avg': avg_loss,
                'loss/train_1d': loss_1d.item(),
                'loss/train_2d': loss_2d.item(),
                f'loss/train_h{rollout_steps}': loss.item(),
                'curriculum/rollout_steps': rollout_steps,
                'curriculum/max_h': max_h,
                'epoch': epoch,
            }

            wandb.log(log_dict, step=global_step)

            # Console progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - batch_start_time
                print(f"Epoch {epoch}/{CONFIG['epochs']} | "
                      f"Batch {batch_idx+1:3d} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Avg: {avg_loss:.6f} | "
                      f"{elapsed:.1f}s")
        
        # End-of-epoch stats
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n[INFO] Epoch {epoch}/{CONFIG['epochs']}: Training Loss={avg_epoch_loss:.6f} | max_h={max_h} | Time: {epoch_time:.1f}s")
        
        # Full validation at current curriculum horizon — matches what we're training on.
        # This is used for early stopping and checkpointing.
        val_loss_norm = None
        val_nrmse_combined = None
        if not skip_validation:
            try:
                print(f"\n[INFO] Running full validation (h={max_h})...")
                val_combined, val_1d, val_2d, val_per_node_1d = evaluate_rollout(
                    model, val_dataloader, criterion, device, norm_stats,
                    rollout_steps=max_h, batched_static_graph=val_batched_graph,
                    use_mixed_precision=use_mixed_precision,
                    rain_1d_index=_rain_1d_index,
                )
                val_loss_norm = val_combined
                # sqrt(MSE_norm) == NRMSE directly; * kaggle_sigma gives RMSE in meters
                val_nrmse_1d = val_1d ** 0.5
                val_nrmse_2d = val_2d ** 0.5
                val_rmse_1d  = val_nrmse_1d * kaggle_sigma_1d
                val_rmse_2d  = val_nrmse_2d * kaggle_sigma_2d
                val_nrmse_combined = (val_nrmse_1d + val_nrmse_2d) / 2
                print(f"  h={max_h}  combined={val_combined:.6e}  "
                      f"1d={val_1d:.6e} (RMSE={val_rmse_1d:.4f}m, NRMSE={val_nrmse_1d:.4f})  "
                      f"2d={val_2d:.6e} (RMSE={val_rmse_2d:.4f}m, NRMSE={val_nrmse_2d:.4f})  "
                      f"approx_kaggle={val_nrmse_combined:.4f}")
                # Full-event rollout val — mirrors inference exactly, much cheaper than
                # windowed val and a far better proxy for the actual Kaggle score.
                full_event_nrmse_combined = None
                if _val_event_file_list:
                    try:
                        print(f"[INFO] Running full-event rollout val ({len(_val_event_file_list)} events)...")
                        fe_combined, fe_1d, fe_2d, fe_per_event = evaluate_full_event_rollout(
                            model, _val_event_file_list, data, norm_stats,
                            _single_static_graph, device,
                            history_len=CONFIG['history_len'],
                            use_mixed_precision=use_mixed_precision,
                            rain_1d_index=_rain_1d_index,
                            n_rain_channels=_n_rain_channels,
                        )
                        full_event_nrmse_combined = fe_combined
                        fe_rmse_1d = fe_1d * kaggle_sigma_1d
                        fe_rmse_2d = fe_2d * kaggle_sigma_2d
                        print(f"  full_event  combined={fe_combined:.4f}  "
                              f"1d={fe_1d:.4f} (RMSE={fe_rmse_1d:.4f}m)  "
                              f"2d={fe_2d:.4f} (RMSE={fe_rmse_2d:.4f}m)")
                        fe_log = {
                            'full_event_val/combined_nrmse': fe_combined,
                            'full_event_val/1d_nrmse': fe_1d,
                            'full_event_val/2d_nrmse': fe_2d,
                            'full_event_val/1d_rmse_m': fe_rmse_1d,
                            'full_event_val/2d_rmse_m': fe_rmse_2d,
                        }
                        # Per-event breakdown table — sortable in wandb, reveals hard events
                        if fe_per_event:
                            fe_sorted = sorted(fe_per_event, key=lambda x: x['combined_nrmse'], reverse=True)
                            fe_table = wandb.Table(columns=[
                                'rank', 'event_name', 'combined_nrmse', 'nrmse_1d', 'nrmse_2d', 'T_steps',
                                't_worst', 't_worst_frac', 'rain_total', 'rain_peak',
                                't_rain_peak_frac', 'rain_at_worst',
                            ])
                            for rank, ev in enumerate(fe_sorted, 1):
                                fe_table.add_data(
                                    rank, ev['event_name'],
                                    round(ev['combined_nrmse'], 5), round(ev['nrmse_1d'], 5), round(ev['nrmse_2d'], 5),
                                    ev['T'],
                                    ev.get('t_worst', -1), ev.get('t_worst_frac', -1.0),
                                    ev.get('rain_total', 0.0), ev.get('rain_peak', 0.0),
                                    ev.get('t_rain_peak_frac', -1.0), ev.get('rain_at_worst', 0.0),
                                )
                            fe_log['full_event_val/per_event_table'] = fe_table
                            # Std across events — how uneven the difficulty is
                            _fe_nrmses = [ev['combined_nrmse'] for ev in fe_per_event]
                            fe_log['full_event_val/combined_nrmse_std'] = float(np.std(_fe_nrmses))
                            fe_log['full_event_val/combined_nrmse_max'] = float(max(_fe_nrmses))
                        wandb.log(fe_log, step=global_step)
                    except Exception as e:
                        print(f"[WARNING] Full-event rollout val failed: {e}")

                if max_h == CONFIG['forecast_len']:
                    # Use full-event NRMSE for best-checkpoint tracking when available
                    _ckpt_metric = full_event_nrmse_combined if full_event_nrmse_combined is not None else val_nrmse_combined
                    if _ckpt_metric < best_kaggle_at_max_h:
                        best_kaggle_at_max_h = _ckpt_metric
                        best_kaggle_epoch = epoch
                        no_improve_count = 0
                        print(f"[INFO] New best validation metric at h={max_h}: {_ckpt_metric:.4f} (epoch {epoch})")
                    else:
                        no_improve_count += 1
                        patience = CONFIG['early_stopping_patience']
                        print(f"[INFO] No improvement at h={max_h}: {no_improve_count}/{patience} epochs without improvement")
                        if patience is not None and no_improve_count >= patience:
                            print(f"[INFO] Early stopping triggered after {no_improve_count} epochs without improvement at h={max_h}")
                            break
            except Exception as e:
                print(f"[WARNING] Validation failed (epoch {epoch}): {e} — skipping val this epoch")

        # Fallback early stopping on training loss when validation is disabled
        if skip_validation and max_h == CONFIG['forecast_len']:
            patience = CONFIG['early_stopping_patience']
            if avg_epoch_loss < best_train_loss_at_max_h:
                best_train_loss_at_max_h = avg_epoch_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"[INFO] No train loss improvement at h={max_h}: {no_improve_count}/{patience} epochs")
                if patience is not None and no_improve_count >= patience:
                    print(f"[INFO] Early stopping (train loss) triggered after {no_improve_count} epochs without improvement")
                    break

        model.train()

        wandb_payload = {'loss/train': avg_epoch_loss, 'curriculum/epoch_max_h': max_h, 'epoch': epoch, 'curriculum/max_h': max_h}
        if val_loss_norm is not None:
            wandb_payload.update({
                f'rollout_val/h{max_h}_combined': val_combined,
                f'rollout_val/h{max_h}_1d_mse_norm': val_1d,
                f'rollout_val/h{max_h}_2d_mse_norm': val_2d,
                f'rollout_val/h{max_h}_1d_rmse_m': val_rmse_1d,
                f'rollout_val/h{max_h}_2d_rmse_m': val_rmse_2d,
                f'rollout_val/h{max_h}_1d_nrmse': val_nrmse_1d,
                f'rollout_val/h{max_h}_2d_nrmse': val_nrmse_2d,
                f'rollout_val/h{max_h}_approx_kaggle': val_nrmse_combined,
            })
            # Per-node 1D MSE table — sortable in wandb UI to identify hard nodes
            if val_per_node_1d is not None:
                table = wandb.Table(columns=['node_id', 'mse_norm', 'nrmse', 'rmse_m'])
                for node_i, mse in enumerate(val_per_node_1d):
                    nrmse = float(mse) ** 0.5
                    table.add_data(node_i, float(mse), nrmse, nrmse * kaggle_sigma_1d)
                wandb_payload[f'rollout_val/h{max_h}_1d_per_node'] = table
        wandb.log(wandb_payload, step=global_step)

        # Step LR scheduler (log-linear decay, once per epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({'train/lr': current_lr}, step=global_step)

        # Checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0 or epoch == CONFIG['epochs']:
            save_checkpoint(model, epoch, avg_epoch_loss, run_dir, CONFIG, global_step=global_step, scheduler=scheduler, optimizer=optimizer)

        # Mirror epoch checkpoint + normalizers to latest/
        epoch_ckpt = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt')
        if os.path.exists(epoch_ckpt) and mirror_latest:
            shutil.copy(epoch_ckpt, os.path.join(latest_dir, f'{SELECTED_MODEL}_epoch_{epoch:03d}.pt'))
            for fname in [f'{SELECTED_MODEL}_normalizers.pkl',
                           f'{SELECTED_MODEL}_normalization_stats.json']:
                src = os.path.join(run_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(latest_dir, fname))
            _val_str = f"{val_loss_norm:.6e}" if val_loss_norm is not None else "N/A"
            print(f"[INFO] Checkpoint mirrored to latest/ (h={max_h} val_loss={_val_str})")

        print()

        epoch_start_time = time.time()

    # Final summary
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    if val_loss_norm is not None:
        print(f"Final epoch val loss (h={max_h}): {val_loss_norm:.6f}")
        print(f"  1D NRMSE={val_nrmse_1d:.4f}  2D NRMSE={val_nrmse_2d:.4f}")
        print(f"  Last epoch approx Kaggle score = {val_nrmse_combined:.4f}")
    else:
        print(f"Final epoch val loss: N/A (validation disabled)")
    if best_kaggle_epoch is not None:
        print(f"  *** {SELECTED_MODEL} best approx Kaggle score (h={CONFIG['forecast_len']}) = {best_kaggle_at_max_h:.4f}  (epoch {best_kaggle_epoch}) ***")
    else:
        print(f"  *** {SELECTED_MODEL} best approx Kaggle score: no full-horizon epochs completed ***")
    print(f"  (Competition score = mean of this over Model_1 and Model_2)")
    patience = CONFIG['early_stopping_patience']
    print(f"Early stopping: {'patience=' + str(patience) + ' (active at h=' + str(CONFIG['forecast_len']) + ')' if patience else 'disabled'}")
    print(f"Checkpoints saved to: {run_dir}")
    print(f"Latest dir:           {latest_dir}")
    final_model_path = os.path.join(run_dir, f'{SELECTED_MODEL}_epoch_{CONFIG["epochs"]:03d}.pt')
    print(f"Final model: {final_model_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UrbanFloodNet with optional resume from checkpoint")
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint directory to resume from (e.g., checkpoints/latest/ or checkpoints/Model_2_20260303_003721/)')
    parser.add_argument('--mixed-precision', action='store_true', 
                        help='Use mixed precision (float16) training to reduce GPU memory usage')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for resume training')
    parser.add_argument('--learning-rate', '--lr', type=float, default=None,
                        help='Override learning rate for resume training')
    parser.add_argument('--max-h', type=int, default=None,
                        help='Override max curriculum horizon (default: 64)')
    parser.add_argument('--no-val', action='store_true',
                        help='Skip validation each epoch (faster, disables early stopping and best-h64 tracking)')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Load Model_1 weights as starting point for Model_2 (transfer learning). '
                             'Pass path to checkpoint dir (e.g. checkpoints/latest). '
                             'Epoch counter and optimizer are reset — full training schedule runs.')
    parser.add_argument('--train-split', type=str, default='train', choices=['train', 'all'],
                        help='Which data split to use for training. "all" = train+val+test (use only for final submission fine-tuning).')
    parser.add_argument('--no-mirror-latest', action='store_true',
                        help='Skip mirroring checkpoints to checkpoints/latest/. Use for probe/experimental runs to avoid overwriting the best saved model.')
    parser.add_argument('--clip-norm', type=float, default=None,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--curriculum', type=str, default=None,
                        help='Custom curriculum as "h:epochs,h:epochs,..." relative to resume epoch. '
                             'E.g. "8:2,16:4,32:4,64:4,128:4". Overrides built-in schedule.')
    parser.add_argument('--keep-short-events', action='store_true',
                        help='Keep short events that have fewer future steps than the curriculum horizon. '
                             'Batches are grouped by available future length so all samples in a batch '
                             'have uniform shape. Short events roll out to their max available horizon.')
    args = parser.parse_args()
    
    # Apply command-line overrides to CONFIG
    # Note: --epochs when combined with --resume is treated as *additional* epochs to run
    # (resolved inside train() after start_epoch is known). Without --resume it's absolute.
    extra_epochs = None
    if args.epochs is not None:
        if args.resume is not None:
            extra_epochs = args.epochs
        else:
            CONFIG['epochs'] = args.epochs
    if args.batch_size is not None:
        CONFIG['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        CONFIG['lr'] = args.learning_rate
        CONFIG['lr_final'] = args.learning_rate  # flat LR when explicitly overridden (e.g. finetune)
    if args.clip_norm is not None:
        CONFIG['clip_norm'] = args.clip_norm
    if args.max_h is not None:
        CONFIG['forecast_len'] = args.max_h

    # Parse custom curriculum: "8:2,16:4,32:4,64:4,128:4" → [(8,2),(16,4),...]
    custom_curriculum = None
    if args.curriculum is not None:
        custom_curriculum = []
        for part in args.curriculum.split(','):
            h_str, n_str = part.strip().split(':')
            custom_curriculum.append((int(h_str), int(n_str)))
        # Update forecast_len to max horizon in curriculum
        CONFIG['forecast_len'] = max(h for h, _ in custom_curriculum)
        # Update total epochs to sum of all stages (relative to resume)
        total_stages_epochs = sum(n for _, n in custom_curriculum)
        extra_epochs = total_stages_epochs
        print(f"[INFO] Custom curriculum: {custom_curriculum} "
              f"(total {total_stages_epochs} epochs, max_h={CONFIG['forecast_len']})")

    # Enable mixed precision if requested
    if args.mixed_precision:
        print("[INFO] Mixed precision training enabled")
        torch.set_float32_matmul_precision('medium')  # Speeds up matmuls on L40S/A100
        # Actual fp16 autocast + GradScaler is applied inside train()
    
    train(resume_from=args.resume, use_mixed_precision=args.mixed_precision, skip_validation=args.no_val, pretrain_from=args.pretrain, train_split=args.train_split, extra_epochs=extra_epochs, mirror_latest=not args.no_mirror_latest, custom_curriculum=custom_curriculum, keep_short_events=args.keep_short_events)
