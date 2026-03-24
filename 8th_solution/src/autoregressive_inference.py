#!/usr/bin/env python
"""
Autoregressive inference script for UrbanFloodNet competition submission.

This script:
1. Loads trained model and normalization statistics
2. Processes test events with autoregressive rollout
3. Denormalizes predictions to original scale
4. Formats output to match sample_submission.csv

Usage:
    # Generate predictions from project root
    python src/autoregressive_inference.py --checkpoint-dir checkpoints --output submission.csv
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup paths (works whether launched from project root or another cwd)
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import unnormalize_col, get_model_config, create_static_hetero_graph, compute_rainfall_features, RAIN_N_CHANNELS
from model import FloodAutoregressiveHeteroModel
from data_lazy import initialize_data
from data_config import SELECTED_MODEL, DATA_FOLDER, BASE_PATH


def load_checkpoint(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config (graph topology)
    model_config = get_model_config()

    # Merge architecture hyperparams from checkpoint (h_dim, msg_dim, hidden_dim,
    # node_dyn_input_dims). Fallback for old checkpoints that predate serialization.
    arch_config = checkpoint.get('model_arch_config', {
        'h_dim': 96,
        'msg_dim': 64,
        'hidden_dim': {
            'oneDedge':    64,
            'oneDedgeRev': 64,
            'twoDedge':    128,
            'twoDedgeRev': 128,
            'twoDoneD':    64,
            'oneDtwoD':    64,
        },
    })
    # node_dyn_input_dims may differ between old checkpoints (oneD=3, twoD=2) and new
    # ones trained with augmented rainfall features (oneD=8, twoD=7). Always restore
    # from checkpoint so the model is built with the dims it was trained with.
    if 'node_dyn_input_dims' in arch_config:
        model_config['node_dyn_input_dims'] = arch_config['node_dyn_input_dims']
    else:
        # Old checkpoint — trained with raw rainfall only (1 channel)
        model_config['node_dyn_input_dims'] = {'oneD': 3, 'twoD': 2}
        if 'global' in model_config.get('node_types', []):
            model_config['node_dyn_input_dims']['global'] = 1
    model_config.update({k: v for k, v in arch_config.items() if k != 'node_dyn_input_dims'})

    # Determine how many rain channels this checkpoint expects.
    # twoD dyn_input = 1 (water_level) + n_rain_channels, so n_rain_channels = twoD_dim - 1.
    _twoD_dyn = model_config['node_dyn_input_dims'].get('twoD', 2)
    n_rain_channels = _twoD_dyn - 1  # 1 for old checkpoints, RAIN_N_CHANNELS for new ones

    # Initialize model
    model = FloodAutoregressiveHeteroModel(**model_config)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
    print(f"[INFO] Training loss: {checkpoint['loss']:.6f}")
    
    return model, checkpoint, n_rain_channels


def load_test_data():
    """
    Load live competition test events from data/<SELECTED_MODEL>/test.
    Returns test event directories plus preprocessed training cache for static graph and normalizers.
    """
    print(f"\n[INFO] Loading test data...")

    # Initialize training-derived cache (normalizers + static graph inputs)
    data = initialize_data()
    required_keys = ['norm_stats']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required data cache keys: {missing}")

    norm_stats = data['norm_stats']

    # Strict live test path requirement
    test_root = Path(BASE_PATH) / 'test'
    if not test_root.exists() or not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    test_event_dirs = sorted(
        [p for p in test_root.glob('event_*') if p.is_dir()],
        key=lambda p: int(p.name.split('_')[-1])
    )

    if len(test_event_dirs) == 0:
        raise RuntimeError(f"No test events found under {test_root}")
    
    print(f"[INFO] Found {len(test_event_dirs)} live test events in {test_root}")
    
    return test_event_dirs, norm_stats, data


def load_model_normalizers(model_id, checkpoint_dir='checkpoints'):
    """
    Load model-specific normalizers from checkpoint directory.
    
    Args:
        model_id: Model identifier (1 or 2)
        checkpoint_dir: Directory containing saved normalizers
        
    Returns:
        dict: Dictionary with normalizer_1d and normalizer_2d objects
    """
    normalizer_path = os.path.join(checkpoint_dir, f'Model_{model_id}_normalizers.pkl')
    
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(f"Model-specific normalizers not found: {normalizer_path}")
    
    print(f"[INFO] Loading model-specific normalizers from {normalizer_path}")
    with open(normalizer_path, 'rb') as f:
        normalizers = pickle.load(f)
    
    return normalizers


def get_event_metadata(event_path):
    """Extract event information from event path."""
    path = Path(event_path)
    event_id = int(path.name.split('_')[-1])
    return event_id, str(path)


def load_event_data(event_path):
    """Load event data using strict expected file structure."""
    path = Path(event_path)

    # Strict required dynamic files
    node_1d_csv = path / '1d_nodes_dynamic_all.csv'
    node_2d_csv = path / '2d_nodes_dynamic_all.csv'

    if not node_1d_csv.exists() or not node_2d_csv.exists():
        raise FileNotFoundError(
            f"Missing required event files in {path}: "
            f"{node_1d_csv.name} and/or {node_2d_csv.name}"
        )

    node_1d = pd.read_csv(node_1d_csv)
    node_2d = pd.read_csv(node_2d_csv)

    return node_1d, node_2d


def build_static_graph_from_cache(data):
    """Build static hetero graph from initialize_data() cache payload."""
    required_keys = [
        'static_1d_sorted',
        'static_2d_sorted',
        'edges1d',
        'edges2d',
        'edges1d2d',
        'edges1dfeats',
        'edges2dfeats',
        'static_1d_cols',
        'static_2d_cols',
        'edge1_cols',
        'edge2_cols',
        'NODE_ID_COL',
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required cache keys for static graph construction: {missing}")

    # Use explicit copies to avoid negative-stride numpy views when converting to torch tensors.
    return create_static_hetero_graph(
        static_1d_norm=data['static_1d_sorted'].copy().reset_index(drop=True),
        static_2d_norm=data['static_2d_sorted'].copy().reset_index(drop=True),
        edges1d=data['edges1d'].copy().reset_index(drop=True),
        edges2d=data['edges2d'].copy().reset_index(drop=True),
        edges1d2d=data['edges1d2d'].copy().reset_index(drop=True),
        edges1dfeats_norm=data['edges1dfeats'].copy().reset_index(drop=True),
        edges2dfeats_norm=data['edges2dfeats'].copy().reset_index(drop=True),
        static_1d_cols=data['static_1d_cols'],
        static_2d_cols=data['static_2d_cols'],
        edge1_cols=data['edge1_cols'],
        edge2_cols=data['edge2_cols'],
        node_id_col=data['NODE_ID_COL'],
        raw_spatial_1d=data.get('raw_spatial_1d'),
        raw_spatial_2d=data.get('raw_spatial_2d'),
    )


def prepare_event_tensors(node_1d, node_2d, norm_stats, device):
    """
    Prepare normalized tensors for one event.
    Returns:
      - y1_all: [T, N1, 1]
      - y2_all: [T, N2, 1]
      - rain2_all: [T, N2, 1]
      - timesteps, node_ids_1d, node_ids_2d
    """
    required_norm = ['normalizer_1d', 'normalizer_2d']
    missing_norm = [k for k in required_norm if k not in norm_stats]
    if missing_norm:
        raise KeyError(f"Missing required key(s) in norm_stats: {missing_norm}")

    exclude_1d = norm_stats['exclude_1d'] if 'exclude_1d' in norm_stats else []
    exclude_2d = norm_stats['exclude_2d'] if 'exclude_2d' in norm_stats else []

    node_1d = node_1d.drop(columns=[c for c in exclude_1d if c in node_1d.columns])
    node_2d = node_2d.drop(columns=[c for c in exclude_2d if c in node_2d.columns])

    node_1d = norm_stats['normalizer_1d'].transform_dynamic(node_1d, exclude_cols=None)
    node_2d = norm_stats['normalizer_2d'].transform_dynamic(node_2d, exclude_cols=None)

    timesteps_1d = sorted(node_1d['timestep'].unique())
    timesteps_2d = sorted(node_2d['timestep'].unique())
    if timesteps_1d != timesteps_2d:
        raise RuntimeError("1D and 2D timestep grids differ; strict structure violation")

    timesteps = timesteps_1d
    node_ids_1d = sorted(node_1d['node_idx'].unique())
    node_ids_2d = sorted(node_2d['node_idx'].unique())

    T = len(timesteps)
    N1 = len(node_ids_1d)
    N2 = len(node_ids_2d)

    y1_all = np.zeros((T, N1, 1), dtype=np.float32)
    y2_all = np.zeros((T, N2, 1), dtype=np.float32)
    rain2_raw = np.zeros((T, N2), dtype=np.float32)

    for t_idx, t in enumerate(timesteps):
        t1 = node_1d[node_1d['timestep'] == t].sort_values('node_idx')
        t2 = node_2d[node_2d['timestep'] == t].sort_values('node_idx')

        if len(t1) != N1 or len(t2) != N2:
            raise RuntimeError(f"Incomplete event rows at timestep {t} (strict structure violation)")

        y1_all[t_idx, :, 0] = t1['water_level'].values
        y2_all[t_idx, :, 0] = t2['water_level'].values
        if 'rainfall' not in t2.columns:
            raise KeyError("Missing required column in 2D event data: rainfall")
        rain2_raw[t_idx, :] = t2['rainfall'].values

    # Always compute augmented rainfall [T, N2, RAIN_N_CHANNELS].
    # Callers that loaded an old checkpoint (1-channel rain) should slice [:, :, :1].
    rain2_aug = compute_rainfall_features(
        torch.tensor(rain2_raw, dtype=torch.float32),
        rain_sum_maxes=norm_stats.get('rain_sum_maxes'),
    )

    return (
        torch.tensor(y1_all, dtype=torch.float32, device=device),
        torch.tensor(y2_all, dtype=torch.float32, device=device),
        rain2_aug.to(device),
        timesteps,
        node_ids_1d,
        node_ids_2d,
    )


def autoregressive_rollout_both(
    model, 
    static_graph,
    y1_hist,      # [H, N1, 1]
    y2_hist,      # [H, N2, 1]
    rain2_all,    # [T, N2, 1]
    device,
    history_len=10
):
    """
    Perform autoregressive rollout for entire event.
    
    Args:
        model: Trained UrbanFloodNet model
        static_graph: Static graph structure
        y1_hist: Initial 1D water levels [H, N1, 1]
        y2_hist: Initial 2D water levels [H, N2, 1]
        rain2_all: Full 2D rainfall [T, N2, 1]
        device: torch device
        history_len: History window size
        
    Returns:
        pred1: [T-H, N1, 1] and pred2: [T-H, N2, 1] predicted normalized water levels
    """
    model.eval()

    T_total = rain2_all.size(0)
    if T_total <= history_len:
        raise RuntimeError(f"Need >{history_len} timesteps for rollout, got {T_total}")

    # B=1 inference — use the vectorized API with a single-sample batch
    B = 1
    batched_graph = model._make_batched_graph(static_graph, B).to(device)
    h = model.init_hidden(static_graph, B, device=device)  # [B*N, h_dim] = [N, h_dim]

    N1 = y1_hist.size(1)
    N2 = y2_hist.size(1)

    # rain_1d_index: [N1] LongTensor mapping each 1D node to its connected 2D node.
    # Used to gather 2D rainfall as a dynamic input for 1D nodes.
    rain_1d_index = getattr(static_graph, 'rain_1d_index', None)
    if rain_1d_index is not None:
        rain_1d_index = rain_1d_index.to(device)

    # Detect if the model has a global context node (Model_2).
    _has_global = 'global' in model.cell.node_types

    def _build_x_dyn(y1, y2, rain2_t):
        """Build x_dyn dict for one timestep. y1: [N1,1], y2: [N2,1], rain2_t: [N2,1]"""
        if rain_1d_index is not None:
            rain_1d_t = rain2_t[rain_1d_index]   # [N1, 1]
            wl_2d_for_1d = y2[rain_1d_index]     # [N1, 1] water level of connected 2D node
            dyn_1d = torch.cat([y1, rain_1d_t, wl_2d_for_1d], dim=-1)  # [N1, 3]
        else:
            dyn_1d = y1
        x_dyn = {
            'oneD': dyn_1d.reshape(B * N1, -1),
            'twoD': torch.cat([y2, rain2_t], dim=-1).reshape(B * N2, -1),
        }
        if _has_global:
            # Global context node has no real observations — feed dummy zero [B, 1]
            x_dyn['global'] = torch.zeros(B, 1, device=device, dtype=y1.dtype)
        return x_dyn

    # Warm start with true history — inputs are [N, F], reshape to [B*N, F] = [N, F]
    for t in range(history_len):
        x_dyn_t = _build_x_dyn(y1_hist[t], y2_hist[t], rain2_all[t])
        h = model.cell(batched_graph, h, x_dyn_t)

    preds_1d = []
    preds_2d = []

    with torch.no_grad():
        for t in range(history_len, T_total):
            # Predict from current hidden state
            node_counts = {'oneD': N1, 'twoD': N2}
            y_next = model.predict_water_levels(h, B, node_counts)
            # y_next['oneD']: [B, N1, 1], y_next['twoD']: [B, N2, 1]

            y1_next = y_next['oneD'].squeeze(0)  # [N1, 1]
            y2_next = y_next['twoD'].squeeze(0)  # [N2, 1]

            preds_1d.append(y1_next)
            preds_2d.append(y2_next)

            x_dyn_next = _build_x_dyn(y1_next, y2_next, rain2_all[t])
            h = model.cell(batched_graph, h, x_dyn_next)

    return torch.stack(preds_1d, dim=0), torch.stack(preds_2d, dim=0)


def denormalize_predictions(predictions, norm_stats, node_type):
    """Denormalize water level predictions for a node type."""
    T, N, _ = predictions.shape

    cols = norm_stats['node1d_cols'] if node_type == 'oneD' else norm_stats['node2d_cols']
    wl_col = cols.index('water_level')

    denorm_preds = []
    for t in range(T):
        pred_t = unnormalize_col(predictions[t], norm_stats, col=wl_col, node_type=node_type)
        denorm_preds.append(pred_t.cpu().numpy())

    denorm_stack = np.stack(denorm_preds, axis=0)  # [T, N, 1]

    # Clamp to physical bounds: water level cannot be negative
    denorm_stack = np.maximum(denorm_stack, 0.0)

    return denorm_stack


def create_submission_rows(predictions, event_id, model_id, node_ids, node_type):
    """Create per-step submission rows for one event/node_type.

    predictions shape is [T_pred, N, 1] where step 0 corresponds to timestep 10.
    """
    rows = []
    T_pred, N, _ = predictions.shape

    for step_idx in range(T_pred):
        for n_idx, node_id in enumerate(node_ids):
            rows.append({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': int(node_type),
                'node_id': int(node_id),
                'step_idx': int(step_idx),
                'water_level': float(predictions[step_idx, n_idx, 0]),
            })

    return rows


def process_all_events(
    model,
    test_events,
    norm_stats,
    data,
    device,
    model_id,
    n_rain_channels,
    max_events=None
):
    """
    Process all test events with autoregressive rollout.
    
    Returns DataFrame with predictions.
    """
    print(f"\n[INFO] Processing {len(test_events)} test events...")
    
    all_rows = []
    # Build static graph once (same for all events)
    static_graph = build_static_graph_from_cache(data).to(device)

    for event_idx, event_path in enumerate(tqdm(test_events, desc="Processing events")):
        if max_events is not None and event_idx >= max_events:
            break

        try:
            # Get event metadata
            event_id, event_dir = get_event_metadata(event_path)

            # Load event data
            node_1d, node_2d = load_event_data(event_dir)

            # Prepare tensors — slice rain to the number of channels the model expects
            y1_all, y2_all, rain2_all, timesteps, node_ids_1d, node_ids_2d = prepare_event_tensors(
                node_1d, node_2d, norm_stats, device
            )
            rain2_all = rain2_all[:, :, :n_rain_channels]

            T_total = y2_all.size(0)
            
            if T_total < 11:  # Need at least 10 for history + 1 to predict
                print(f"[WARN] Skipping event {event_id}: only {T_total} timesteps")
                continue
            
            # First 10 timesteps are provided; rollout for remaining timesteps.
            pred1_norm, pred2_norm = autoregressive_rollout_both(
                model=model,
                static_graph=static_graph,
                y1_hist=y1_all[:10],
                y2_hist=y2_all[:10],
                rain2_all=rain2_all,
                device=device,
                history_len=10
            )
            
            # Denormalize
            pred1_denorm = denormalize_predictions(pred1_norm, norm_stats, node_type='oneD')
            pred2_denorm = denormalize_predictions(pred2_norm, norm_stats, node_type='twoD')

            event_rows = []
            event_rows.extend(create_submission_rows(pred1_denorm, event_id, model_id, node_ids_1d, node_type=1))
            event_rows.extend(create_submission_rows(pred2_denorm, event_id, model_id, node_ids_2d, node_type=2))

            all_rows.extend(event_rows)
            
        except Exception as e:
            print(f"[ERROR] Failed to process event {event_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows)
    
    print(f"\n[INFO] Generated {len(df)} prediction rows")
    print(f"[INFO] Events: {df['event_id'].nunique()}")
    print(f"[INFO] Rows per event: {len(df) // df['event_id'].nunique():.0f} avg")
    
    return df


def match_to_sample_submission(predictions_df, sample_path):
    """
    Match predictions to sample_submission.csv format and order.
    
    This ensures:
    1. All required rows are present (fills missing with NaN)
    2. Rows are in the exact order expected by Kaggle
    3. Extra rows are removed
    """
    print(f"\n[INFO] Matching to sample submission: {sample_path}")
    
    # Load sample submission - use chunking to avoid memory issues with 50M+ rows
    print(f"[INFO] Loading sample submission (this may take a moment for 50M+ rows)...")
    chunks = []
    for chunk in pd.read_csv(sample_path, chunksize=100000):
        chunks.append(chunk)
    sample = pd.concat(chunks, ignore_index=True)
    
    print(f"[INFO] Sample submission: {len(sample)} rows")
    print(f"[INFO] Predictions: {len(predictions_df)} rows")
    
    # Build per-key sequence index to disambiguate repeated rows across timesteps
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    sample = sample.copy()
    sample['step_idx'] = sample.groupby(key_cols).cumcount()

    required_pred_cols = key_cols + ['step_idx', 'water_level']
    missing_pred_cols = [c for c in required_pred_cols if c not in predictions_df.columns]
    if missing_pred_cols:
        raise KeyError(f"Predictions missing required columns: {missing_pred_cols}")

    result = sample.merge(
        predictions_df[required_pred_cols],
        on=key_cols + ['step_idx'],
        how='left',
        suffixes=('_orig', '_pred')
    )

    if 'water_level_pred' not in result.columns:
        print("[WARN] No predicted water levels found after merge - using NaN")
        result['water_level'] = np.nan
    else:
        result['water_level'] = result['water_level_pred']
    
    result = result[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']]

    # Report completeness (but don't fail)
    nan_count = result['water_level'].isna().sum()
    coverage_pct = 100 * (1 - nan_count / len(result))
    print(f"[INFO] Coverage: {coverage_pct:.1f}% ({len(result) - nan_count}/{len(result)} rows with predictions)")
    
    if nan_count > 0:
        print(f"[WARN] {nan_count} rows missing predictions (will submit with NaN)")
    
    # Verify row count
    if len(result) != len(sample):
        print(f"[WARN] Result has {len(result)} rows but expected {len(sample)}")
    
    print(f"[INFO] Final submission: {len(result)} rows")
    
    return result


def _find_best_by_val_loss(checkpoint_dir, model_id):
    """Scan all .pt files in checkpoint_dir for Model_model_id and return the path
    with the lowest val_loss stored in checkpoint metadata.  Falls back to None
    if no readable checkpoints are found."""
    import glob as _glob
    pattern = os.path.join(checkpoint_dir, f"Model_{model_id}_*.pt")
    candidates = sorted(_glob.glob(pattern))
    if not candidates:
        return None
    best_path = None
    best_loss = float('inf')
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            loss = ckpt.get('val_loss', None)
            if loss is None:
                loss = ckpt.get('loss', None)
            if loss is not None and float(loss) < best_loss:
                best_loss = float(loss)
                best_path = path
        except Exception:
            pass
    if best_path:
        print(f"[INFO] Best-by-val-loss in {checkpoint_dir}: {os.path.basename(best_path)} (val_loss={best_loss:.6e})")
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description='Autoregressive inference for UrbanFloodNet competition (automatically processes both Model_1 and Model_2)'
    )
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints (default: checkpoints/)')
    parser.add_argument('--model1-dir', type=str, default=None,
                        help='Checkpoint directory for Model_1 (overrides --checkpoint-dir for Model_1)')
    parser.add_argument('--model2-dir', type=str, default=None,
                        help='Checkpoint directory for Model_2 (overrides --checkpoint-dir for Model_2)')
    parser.add_argument('--model1-ckpt', type=str, default=None,
                        help='Exact .pt file to use for Model_1 (overrides --model1-dir and --select)')
    parser.add_argument('--model2-ckpt', type=str, default=None,
                        help='Exact .pt file to use for Model_2 (overrides --model2-dir and --select)')
    parser.add_argument('--select', type=str, default='val_loss',
                        choices=['val_loss', 'latest_epoch'],
                        help='How to select the checkpoint from a directory: '
                             'val_loss = scan all .pt files and pick lowest val_loss in metadata (default); '
                             'latest_epoch = use the highest-numbered epoch checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--sample', type=str, default='../FloodModel/sample_submission.csv',
                        help='Path to sample_submission.csv')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum number of events to process (for testing)')

    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print("UrbanFloodNet Autoregressive Inference (Both Models)")
    print(f"{'='*70}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint directory (default): {args.checkpoint_dir}")
    if args.model1_dir:
        print(f"[INFO] Model_1 checkpoint dir (override): {args.model1_dir}")
    if args.model2_dir:
        print(f"[INFO] Model_2 checkpoint dir (override): {args.model2_dir}")
    print(f"[INFO] Checkpoint select policy: {args.select}")
    print(f"[INFO] Output: {args.output}")
    
    # Collect predictions from both models
    all_predictions = []
    
    for model_id in [1, 2]:
        print(f"\n[INFO] ========== Processing Model {model_id} ==========")
        
        # Import modules to allow SELECTED_MODEL to be reset
        import importlib
        import data_config as dc
        
        # Update SELECTED_MODEL dynamically
        dc.SELECTED_MODEL = f"Model_{model_id}"
        dc.BASE_PATH = f"data/{dc.SELECTED_MODEL}"
        
        # Reload modules to pick up new paths (critical for data)
        import data_lazy
        importlib.reload(data_lazy)
        
        # CRITICAL: Re-import after reload to get fresh function refs
        from data_lazy import initialize_data as initialize_data_fresh
        
        # Clear any stale data references
        print(f"[INFO] Initializing fresh data for Model {model_id}...")
        data_fresh = initialize_data_fresh()
        if 'norm_stats' not in data_fresh:
            raise KeyError(f"Missing norm_stats in Model {model_id} data cache")
        
        norm_stats = data_fresh['norm_stats']
        
        # Find test events for this model
        from pathlib import Path as PathlibPath
        test_root = PathlibPath(dc.BASE_PATH) / 'test'
        if not test_root.exists():
            print(f"[ERROR] No test directory for Model {model_id}!")
            continue
        
        test_events = sorted(
            [str(p) for p in test_root.glob('event_*') if p.is_dir()],
            key=lambda p: int(PathlibPath(p).name.split('_')[-1])
        )
        
        if len(test_events) == 0:
            print(f"[ERROR] No test events found for Model {model_id}!")
            continue
        
        print(f"[INFO] Found {len(test_events)} test events for Model {model_id}")

        # If an exact checkpoint file was specified, use it directly.
        per_model_ckpt = args.model1_ckpt if model_id == 1 else args.model2_ckpt
        if per_model_ckpt is not None:
            model_checkpoint_path = per_model_ckpt
            print(f"[INFO] Using explicit checkpoint: {model_checkpoint_path}")
        else:
            # Resolve which directory to search for this model's checkpoint.
            # --model1-dir / --model2-dir override --checkpoint-dir per-model.
            per_model_dir = args.model1_dir if model_id == 1 else args.model2_dir
            checkpoint_dir = per_model_dir if per_model_dir is not None else args.checkpoint_dir

            # Find checkpoint according to --select policy.
            if args.select == 'val_loss':
                model_checkpoint_path = _find_best_by_val_loss(checkpoint_dir, model_id)
                if model_checkpoint_path is None:
                    print(f"[WARN] --select val_loss found nothing in {checkpoint_dir}; falling back to latest epoch")
            else:
                model_checkpoint_path = None

        if model_checkpoint_path is None:
            # Fallback: find the highest-numbered epoch checkpoint
            import glob as _glob
            epoch_files = sorted(_glob.glob(os.path.join(checkpoint_dir, f"Model_{model_id}_epoch_*.pt")))
            if epoch_files:
                model_checkpoint_path = epoch_files[-1]
            else:
                # Legacy fallback: try best.pt
                best_path = os.path.join(checkpoint_dir, f"Model_{model_id}_best.pt")
                if os.path.exists(best_path):
                    model_checkpoint_path = best_path

        if model_checkpoint_path is None:
            print(f"[ERROR] No checkpoint found for Model {model_id} in {checkpoint_dir}!")
            print(f"[ERROR] Please train Model_{model_id} first, or specify --model{model_id}-ckpt")
            continue

        print(f"[INFO] Using checkpoint: {model_checkpoint_path}")
        
        # Load model for this specific model_id (architecture depends on graph size)
        model, checkpoint, n_rain_channels = load_checkpoint(model_checkpoint_path, device)
        
        # Use freshly loaded data (Model_1 and Model_2 have different graph structures)
        data = data_fresh
        
        # Load model-specific normalizers (trained on this model's data)
        try:
            model_normalizers = load_model_normalizers(model_id, checkpoint_dir)
            norm_stats['normalizer_1d'] = model_normalizers['normalizer_1d']
            norm_stats['normalizer_2d'] = model_normalizers['normalizer_2d']
            print(f"[INFO] Loaded model-specific normalizers for Model {model_id}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print(f"[WARN] Using normalizers from cache (may not match training)")
        
        # Process events for this model
        predictions_df = process_all_events(
            model=model,
            test_events=test_events,
            norm_stats=norm_stats,
            data=data,
            device=device,
            model_id=model_id,
            n_rain_channels=n_rain_channels,
            max_events=args.max_events
        )
        
        all_predictions.append(predictions_df)
        print(f"[INFO] Generated {len(predictions_df)} rows for Model {model_id}")
    
    # Combine predictions from both models
    if len(all_predictions) > 0:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\n[INFO] Combined predictions: {len(combined_predictions)} total rows")
        
        # Match to sample submission format
        if os.path.exists(args.sample):
            submission_df = match_to_sample_submission(combined_predictions, args.sample)
        else:
            print(f"[WARN] Sample submission not found: {args.sample}")
            print(f"[WARN] Using raw predictions (may not match Kaggle format)")
            submission_df = combined_predictions
            # Add row_id if not present
            if 'row_id' not in submission_df.columns:
                submission_df.insert(0, 'row_id', range(len(submission_df)))
        
        # Save submission
        submission_df.to_csv(args.output, index=False)
        print(f"\n[INFO] Saved submission to: {args.output}")
        
        # Show preview
        print(f"\n[INFO] First 10 rows:")
        print(submission_df.head(10))
        
        print(f"\n[INFO] Last 10 rows:")
        print(submission_df.tail(10))
        
        # Summary statistics
        print(f"\n[INFO] Summary:")
        print(f"  Total rows: {len(submission_df)}")
        print(f"  Models: {sorted(submission_df['model_id'].unique())}")
        print(f"  Events: {submission_df['event_id'].nunique()}")
        print(f"  Water level range: [{submission_df['water_level'].min():.6f}, {submission_df['water_level'].max():.6f}]")
        print(f"  Water level mean: {submission_df['water_level'].mean():.6f}")
        print(f"  Water level std: {submission_df['water_level'].std():.6f}")
        
        # Check for NaN values
        nan_count = submission_df['water_level'].isna().sum()
        if nan_count > 0:
            print(f"\n[ERROR] Submission has {nan_count} NaN values!")
        else:
            print(f"\n[SUCCESS] All rows have valid predictions!")
        
        print(f"\n[INFO] To submit to Kaggle, run:")
        print(f"  python submit_to_kaggle.py {args.output} --message \"UrbanFloodNet submission\"")
    else:
        print("[ERROR] Failed to generate predictions for any model")
    
    print(f"\n{'='*70}")
    print("Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
