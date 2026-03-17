#%%writefile hetero_gnn_flood_with_rollout_ddp.py

'''
Heterogeneous GNN for Urban Flood Modeling
==========================================
SOTA pipeline for predicting water level changes in coupled 1D-2D systems.

Architecture:
- Separate encoders for 1D (pipe) and 2D (surface) nodes
- Bidirectional message passing: pipe, surface, and 1D-2D coupling
- Predicts Δwl (change in water level) for physics-informed learning
- Autoregressive inference for test rollout

Run on Colab with GPU:
  !pip install torch torch-geometric pandas numpy pyarrow tqdm -q
'''

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, TransformerConv, Linear
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import warnings
import random
warnings.filterwarnings('ignore')

# =============================================================================
# DDP SETUP
# =============================================================================

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_ddp() -> int:
    '''Initialize DDP if launched with torchrun. Returns local_rank.'''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}")
        )
        return local_rank
    return 0

def cleanup_ddp():
    if ddp_is_initialized():
        dist.destroy_process_group()

# =============================================================================
# LOGGING SETUP
# =============================================================================


def set_seed(seed: int = 42):
    '''Set random seeds for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    '''Configure logging with console and optional file output.'''
    logger = logging.getLogger("flood_gnn")
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
set_seed(42)  # Set random seeds for reproducibility
if is_main_process():
    logger.info("Random seed set to 42")

# =============================================================================
# CUDA OPTIMIZATIONS
# =============================================================================

def setup_cuda_optimizations():
    '''Apply CUDA-specific optimizations for maximum speed.'''
    if torch.cuda.is_available():
        # Enable cuDNN autotuner - finds fastest algorithms
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for Ampere+ GPUs (3x faster matmul with minimal precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable Flash Attention / Memory-Efficient Attention for TransformerConv
        # These use optimized CUDA kernels (FlashAttention, xFormers-style)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)  # Fallback for unsupported cases
        
        # Use expandable segments allocator to reduce memory fragmentation
        # Allows fitting larger batches before OOM on T4 16GB
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        torch.cuda.empty_cache()
        
        if is_main_process():
            logger.info("CUDA optimizations enabled")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Log SDP status
            if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                logger.info(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
            if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
                logger.info(f"  Mem-efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")

setup_cuda_optimizations()

# =============================================================================
# CONFIGURATION
# =============================================================================

class BaseConfig:
    '''Base configuration shared by all models.'''
    # Paths
    BASE_PATH = Path(os.environ.get("FLOOD_BASE_PATH", "/workspace/Models/"))
    OUTPUT_PATH = Path(os.environ.get("FLOOD_OUTPUT_PATH", "/workspace/gnn_output/"))
    
    # Data
    WARMUP_TIMESTEPS = 10
    VAL_RATIO = 0.2
    
    # Loss weighting (1D is often harder, upweight it)
    LOSS_WEIGHT_1D = 1.0
    
    # Loss function: 'mse' (standardized RMSE) or 'huber' (SmoothL1 for heavy tails)
    USE_HUBER_LOSS = False # Set to True for better handling of heavy-tailed delta distributions
    # Huber delta thresholds (in standardized units). If None, auto-computed from p95 of deltas
    HUBER_DELTA_1D = None  # Auto: ~0.20 / std_1d (from delta analysis p95)
    HUBER_DELTA_2D = None  # Auto: ~0.05 / std_2d (from delta analysis p95)
    
    # Bias loss: penalizes systematic over/under-prediction (mean error) to reduce AR drift
    USE_BIAS_LOSS = True
    BIAS_LOSS_WEIGHT = 0.1  # Relative weight of bias penalty vs main loss
    
    # Noise injection (MeshGraphNets/GNS-style): adds Gaussian noise to input
    # water levels during multi-step training to close the train/inference
    # distribution gap and improve autoregressive rollout robustness.
    NOISE_INJECTION = False
    NOISE_STD_1D = 0.003   # Noise std for 1D water levels (meters)
    NOISE_STD_2D = 0.003   # Noise std for 2D water levels (meters)
    
    # Previous water level features: feed wl(t-1), wl(t-2), ... as input
    # features so the model can infer velocity (from 1 lag) and acceleration
    # (from 2 lags).  All lags are normalized by the same wl_scale as
    # wl_current, keeping feature-space noise at ~0.001.
    # Set to 0 to disable, 1 for just wl(t-1), 2 for wl(t-1)+wl(t-2).
    WL_PREV_STEPS = None
    WL_PREV_DROPOUT = 0.0  # Per-sample probability of zeroing ALL wl_prev features
    
    # LR schedule: epoch → learning rate (piecewise-constant).
    # Evaluated like ROLLOUT_SCHEDULE: last entry whose epoch <= current epoch wins.
    # Set to None to use fixed LEARNING_RATE for all epochs.
    LR_SCHEDULE = None  # Default: no schedule (constant LR)
    
    # Partial BPTT: let gradients flow through this many consecutive rollout steps
    # before detaching. 1 = fully truncated (old behaviour), rollout_steps = full BPTT.
    # 3 is a good default: teaches the model about 3-step error compounding without
    # excessive memory cost. Override in per-model configs.
    BPTT_STEPS = 1  # Default: fully truncated (safe fallback)
    
    # Curriculum rollout schedule: gradually increase rollout length to teach
    # long-horizon stability.  Dict maps epoch → rollout_steps.
    # The schedule is evaluated in ascending epoch order; the last entry whose
    # epoch threshold is <= current epoch wins.
    # Set to None to disable curriculum (use fixed ROLLOUT_STEPS).
    ROLLOUT_SCHEDULE = None  # Default: no curriculum
    
    # Temporal bundling: number of Δwl steps decoded per forward pass.
    # K=1 → classic 1-step decoder; K>1 → GraphCast-style temporal bundling
    # where a single GNN forward predicts K consecutive deltas.
    TEMPORAL_BUNDLE_K = 4
    
    @classmethod
    def get_rollout_for_epoch(cls, epoch: int) -> int:
        '''Return the rollout length to use at the given epoch.
        
        If ROLLOUT_SCHEDULE is set, looks up the schedule; otherwise falls
        back to ROLLOUT_STEPS (fixed).
        '''
        schedule = cls.ROLLOUT_SCHEDULE
        if schedule is None:
            return cls.ROLLOUT_STEPS
        current = cls.ROLLOUT_STEPS
        for ep_threshold in sorted(schedule.keys()):
            if epoch >= ep_threshold:
                current = schedule[ep_threshold]
            else:
                break
        return current
    
    @classmethod
    def get_max_rollout(cls) -> int:
        '''Return the maximum rollout length across the entire schedule.'''
        if cls.ROLLOUT_SCHEDULE is None:
            return cls.ROLLOUT_STEPS
        return max(max(cls.ROLLOUT_SCHEDULE.values()), cls.ROLLOUT_STEPS)
    
    @classmethod
    def get_lr_for_epoch(cls, epoch: int) -> float:
        '''Return the learning rate to use at the given epoch.
        
        If LR_SCHEDULE is set, looks up the schedule; otherwise falls
        back to LEARNING_RATE (fixed).
        '''
        schedule = cls.LR_SCHEDULE
        if schedule is None:
            return cls.LEARNING_RATE
        current = cls.LEARNING_RATE
        for ep_threshold in sorted(schedule.keys()):
            if epoch >= ep_threshold:
                current = schedule[ep_threshold]
            else:
                break
        return current
    
    # GNN Convolution type: 'gatv2', 'sage', 'transformer'
    CONV_TYPE = 'transformer'
    ATTENTION_HEADS = 16  # For gatv2 and transformer
    # Per-node learnable ID embeddings (helps node-specific biases/corrections).
    # Kept disabled by default for backward compatibility with old checkpoints.
    USE_NODE_EMBEDDINGS = False
    # If None, defaults to hidden_dim // 4 inside HeteroFloodGNN.
    NODE_EMBED_DIM = None
    
    # Edge features: pass static edge attributes (length, slope, diameter, etc.)
    # to conv layers. Supported by 'gatv2' and 'transformer' conv types.
    # SAGEConv does not support edge features — they will be silently ignored.
    USE_EDGE_FEATURES = True
    
    # Temporal features
    RAIN_LAG_STEPS = 5      # Number of lagged rainfall timesteps to include as features (past)
    RAIN_FUTURE_STEPS = 4   # Number of future rainfall timesteps to include as features (leads). 0 disables.
    
    # Performance optimizations
    USE_CHANNELS_LAST = True  # Use channels_last memory format for potential speedup
    USE_FUSED_LAYERNORM = True  # Use fused LayerNorm+GELU where possible
    
    # Device-aware configuration
    if torch.cuda.is_available():
        local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
        DEVICE = torch.device(f'cuda:{local_rank_env}')
        NUM_WORKERS = 0  # Use 0 to share cache across samples
        USE_AMP = True
        USE_COMPILE = True
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        NUM_WORKERS = 0
        USE_AMP = True
        USE_COMPILE = False
    else:
        DEVICE = torch.device('cpu')
        NUM_WORKERS = 0
        USE_AMP = False
        USE_COMPILE = False


class Model1Config(BaseConfig):
    '''
    Config for Model 1: 17 1D nodes, 3716 2D nodes
    Smaller 1D network - easier to learn, converges quickly
    '''
    # Model architecture
    HIDDEN_DIM = 64
    NUM_LAYERS = 4
    DROPOUT = 0.1
    WL_PREV_STEPS = 2
    USE_NODE_EMBEDDINGS = True
    
    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 25
    PATIENCE = 10
    GRAD_ACCUM_STEPS = 1
    FIND_LR = False
    
    # Piecewise-constant LR schedule: epoch → learning rate
    LR_SCHEDULE = {0: 1e-4, 3: 1e-5}
    #LR_SCHEDULE = {0: 1e-5}
    
    # Multi-step rollout
    ROLLOUT_STEPS = 20
    BPTT_STEPS = 20            # Partial BPTT
    
    # Curriculum rollout: epoch → rollout_steps (model 1 is small, short schedule)
    ROLLOUT_SCHEDULE = {0: 20}
    
    # Validations
    AR_VAL_EVERY = 5


class Model2Config(BaseConfig):
    '''
    Config for Model 2: 198 1D nodes, 4299 2D nodes
    Larger 1D network - harder to learn, needs more capacity/epochs
    '''
    # Model architecture - slightly larger for more 1D nodes
    HIDDEN_DIM = 96
    NUM_LAYERS = 4
    DROPOUT = 0.1
    WL_PREV_STEPS = 10
    USE_NODE_EMBEDDINGS = True
    
    # Training - lower LR for stability, more epochs
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4      # Lower LR for more stable 1D learning
    WEIGHT_DECAY = 1e-5
    EPOCHS = 65               # roll10: 42 ep | roll20: 9 ep (1e-4×7) | roll40/80: 7 ep each
    PATIENCE = 15
    GRAD_ACCUM_STEPS = 1
    FIND_LR = False
    
    # Per rollout: large lr → 1e-4 use 5 → 5e-5 use 1 → 1e-5 use 1
    # LR_SCHEDULE = {
    #     # Rollout 10 (0-41)
    #     0: 5e-4,    # 0-34: 35 epochs
    #     35: 1e-4,   # 35-39: 5 epochs
    #     40: 5e-5,   # 40: 1 epoch
    #     41: 1e-5,   # 41: 1 epoch
    #     # Rollout 20 (42-50)
    #     42: 1e-4,   # 42-48: 7 epochs
    #     49: 5e-5,   # 49: 1 epoch
    #     50: 1e-5,   # 50: 1 epoch
    #     # Rollout 40 (51-57)
    #     51: 1e-4,   # 51-55: 5 epochs
    #     56: 5e-5,   # 56: 1 epoch
    #     57: 1e-5,   # 57: 1 epoch
    #     # Rollout 80 (58-64)
    #     58: 1e-4,   # 58-62: 5 epochs
    #     63: 5e-5,   # 63: 1 epoch
    #     64: 1e-5,   # 64: 1 epoch
    # 
    LR_SCHEDULE = {
        0: 1e-5,   # 41: 1 epoch}
        11: 5e-6,
        12: 1e-6}
    ROLLOUT_SCHEDULE = {0: 80}
    # Multi-step rollout - more steps to reduce 1D bias
    ROLLOUT_STEPS = 80         # Initial rollout (overridden by schedule below)
    BPTT_STEPS = 80           # Partial BPTT
    
    AR_VAL_EVERY = 2


# Model configs mapping
MODEL_CONFIGS = {
    1: Model1Config,
    2: Model2Config,
}

# Default config for backward compatibility
Config = Model1Config
Config.OUTPUT_PATH.mkdir(exist_ok=True)


def log_config(cfg, model_id: int, logger):
    '''Log all configuration parameters for a model.'''
    logger.info("-" * 60)
    logger.info(f"Full Configuration for Model {model_id}")
    logger.info("-" * 60)
    
    # Get all class attributes (not methods, not private)
    config_attrs = [attr for attr in dir(cfg) if not attr.startswith('_') and not callable(getattr(cfg, attr))]
    
    # Group by category
    categories = {
        'Paths': ['BASE_PATH', 'OUTPUT_PATH'],
        'Data': ['WARMUP_TIMESTEPS', 'VAL_RATIO'],
        'Model Architecture': ['HIDDEN_DIM', 'NUM_LAYERS', 'DROPOUT', 'CONV_TYPE', 'ATTENTION_HEADS',
                               'USE_EDGE_FEATURES', 'USE_NODE_EMBEDDINGS', 'NODE_EMBED_DIM'],
        'Training': ['BATCH_SIZE', 'LEARNING_RATE', 'LR_SCHEDULE', 'WEIGHT_DECAY', 'EPOCHS', 'PATIENCE', 'GRAD_ACCUM_STEPS', 'FIND_LR'],
        'Loss': ['LOSS_WEIGHT_1D', 'USE_HUBER_LOSS', 'HUBER_DELTA_1D', 'HUBER_DELTA_2D'],
        'Rollout': ['ROLLOUT_STEPS', 'ROLLOUT_SCHEDULE', 'AR_VAL_EVERY', 'BPTT_STEPS', 'TEMPORAL_BUNDLE_K'],
        'Temporal': ['RAIN_LAG_STEPS'],
        'Device': ['DEVICE', 'NUM_WORKERS', 'USE_AMP', 'USE_COMPILE'],
        'Performance': ['USE_CHANNELS_LAST', 'USE_FUSED_LAYERNORM'],
    }
    
    logged_attrs = set()
    for category, attrs in categories.items():
        logger.info(f"  [{category}]")
        for attr in attrs:
            if hasattr(cfg, attr):
                value = getattr(cfg, attr)
                logger.info(f"    {attr:20s}: {value}")
                logged_attrs.add(attr)
    
    # Log any remaining attributes not in categories
    remaining = [attr for attr in config_attrs if attr not in logged_attrs]
    if remaining:
        logger.info("  [Other]")
        for attr in remaining:
            value = getattr(cfg, attr)
            logger.info(f"    {attr:20s}: {value}")
    
    logger.info("-" * 60)


# =============================================================================
# FEATURE SCALE DEFAULTS
# =============================================================================

class FeatureScales:
    '''Encapsulates per-model dynamic feature normalization scales.
    
    Constructed once from the std_values dict (or defaults) and passed around
    instead of repeating the dict-lookup pattern in every function.
    '''
    
    # Default fallback values (backward compatibility)
    _DEFAULTS = {
        'wl_1d_scale': 100.0,
        'wl_2d_scale': 100.0,
        'pipe_fill_scale': 10.0,
        'head_above_scale': 10.0,
        'pond_depth_scale': 10.0,
        'rainfall_scale': 1.0,
        'cum_rainfall_scale': 10.0,
    }
    
    __slots__ = ('wl_1d_scale', 'wl_2d_scale', 'pipe_fill_scale',
                 'head_above_scale', 'pond_depth_scale', 'rainfall_scale',
                 'cum_rainfall_scale')
    
    def __init__(self, feature_scales: Optional[Dict[str, float]] = None):
        src = feature_scales if feature_scales is not None else {}
        for key, default in self._DEFAULTS.items():
            setattr(self, key, src.get(key, default))
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, float]]) -> 'FeatureScales':
        '''Create FeatureScales from a dict (or None for defaults).'''
        return cls(d)


# =============================================================================
# FEATURE BUILDER — Single source of truth for feature construction
# =============================================================================

class FeatureBuilder:
    '''Centralized dynamic feature construction for 1D and 2D flood model nodes.
    
    Eliminates feature-construction duplication across dataset, training,
    inference, and diagnostic code paths.
    
    TO ADD A NEW FEATURE — edit only this class:
      1. Add it in build_dynamic_np()  AND  build_dynamic_torch()
      2. Update N_DYNAMIC_1D  or  N_DYNAMIC_2D_BASE
      3. Update NAMES_1D_DYNAMIC  or  NAMES_2D_DYNAMIC_BASE
    All callers (datasets, training, inference, diagnostics) get it automatically.
    '''
    
    # ── Feature counts — update when adding/removing features ─────────
    N_DYNAMIC_1D = 19      # wl, pipe_fill, head_above, global_rain, global_rain_ind, global_cum_rain, norm_t,
                           # fill_fraction, mean_fill_frac(global), frac_surcharged(global), sys_wl_trend(global),
                           # global_cum_rain_3, downstream_mean_fill_fraction, global_storage_proxy,
                           # head_diff_1d_2d, n_timesteps_scaled, remaining_frac,
                           # wl_velocity (Δwl at t), wl_acceleration (Δ²wl at t)
                           # + coupled rain lag/lead channels added in __init__
    # Base 2D dynamic features (per node, per timestep) BEFORE adding:
    # - wl_prev history
    # - rain_lag_steps past rainfall lags
    # - rain_future_steps future rainfall leads
    N_DYNAMIC_2D_BASE = 6  # wl, rain, pond_depth, rain_ind, cum_rain, norm_t
    
    NAMES_1D_STATIC = ['position_x', 'position_y', 'depth',
                       'invert_elev', 'surface_elev', 'base_area']
    NAMES_1D_DYNAMIC = ['wl_1d', 'pipe_fill', 'head_above_surface',
                        'global_rainfall', 'global_rain_indicator',
                        'global_cum_rainfall', 'normalized_t',
                        'fill_fraction', 'mean_fill_fraction',
                        'frac_surcharged', 'system_wl_trend',
                        'global_cum_rainfall_3', 'downstream_mean_fill_fraction',
                        'global_storage_proxy',
                        'head_diff_1d_2d', 'n_timesteps_scaled', 'remaining_frac',
                        'wl_velocity', 'wl_acceleration']
    NAMES_2D_STATIC = ['position_x', 'position_y', 'area', 'roughness',
                       'min_elev', 'elevation', 'aspect', 'curvature', 'flow_acc']
    NAMES_2D_DYNAMIC_BASE = ['wl_2d', 'rainfall', 'pond_depth', 'rain_indicator',
                             'cum_rainfall', 'normalized_t']
    
    def __init__(self, static_data: 'StaticGraphData',
                 feature_scales: Optional[Dict[str, float]] = None,
                 rain_lag_steps: int = 10,
                 rain_future_steps: int = 0,
                 wl_prev_steps: int = 0,
                 use_edge_features: bool = False):
        self.static = static_data
        self.rain_lag_steps = rain_lag_steps
        self.rain_future_steps = max(int(rain_future_steps), 0)
        self.n_1d = static_data.num_1d
        self.n_2d = static_data.num_2d
        self.wl_prev_steps = wl_prev_steps
        self.use_edge_features = use_edge_features
        
        # Normalization scales
        fs = FeatureScales.from_dict(feature_scales)
        self.wl_1d_scale = fs.wl_1d_scale
        self.wl_2d_scale = fs.wl_2d_scale
        self.pipe_fill_scale = fs.pipe_fill_scale
        self.head_above_scale = fs.head_above_scale
        self.pond_depth_scale = fs.pond_depth_scale
        self.rainfall_scale = fs.rainfall_scale
        self.cum_rainfall_scale = fs.cum_rainfall_scale
        
        # Shadow class-level feature counts/names when wl_prev and/or future rain features are active
        base_dyn_1d = self.__class__.N_DYNAMIC_1D
        names_1d_dyn = list(self.__class__.NAMES_1D_DYNAMIC)
        base_dyn_2d = self.__class__.N_DYNAMIC_2D_BASE
        names_2d_dyn_base = list(self.__class__.NAMES_2D_DYNAMIC_BASE)
        
        # 1D receives coupled 2D rainfall lags/leads (mirrors 2D temporal rain context).
        if self.rain_lag_steps > 0:
            base_dyn_1d += self.rain_lag_steps
            names_1d_dyn += [f'global_rain_lag{i+1}' for i in range(self.rain_lag_steps)]
        
        if wl_prev_steps > 0:
            base_dyn_1d += wl_prev_steps
            base_dyn_2d += wl_prev_steps
            names_1d_dyn += [f'wl_prev{i+1}_1d' for i in range(wl_prev_steps)]
            names_2d_dyn_base += [f'wl_prev{i+1}_2d' for i in range(wl_prev_steps)]
        
        if self.rain_future_steps > 0:
            base_dyn_1d += self.rain_future_steps
            base_dyn_2d += self.rain_future_steps
            names_1d_dyn += [f'global_rain_lead{i+1}' for i in range(self.rain_future_steps)]
            names_2d_dyn_base += [f'rain_lead{i+1}' for i in range(self.rain_future_steps)]
        
        self.N_DYNAMIC_1D = base_dyn_1d
        self.N_DYNAMIC_2D_BASE = base_dyn_2d
        self.NAMES_1D_DYNAMIC = names_1d_dyn
        self.NAMES_2D_DYNAMIC_BASE = names_2d_dyn_base
        
        # Static elevations (numpy, for CPU/dataset paths)
        self.invert_elev = np.nan_to_num(
            static_data.nodes_1d['invert_elevation'].values.astype(np.float32), nan=0.0)
        self.surface_elev = np.nan_to_num(
            static_data.nodes_1d['surface_elevation'].values.astype(np.float32), nan=0.0)
        self.pipe_depth = np.maximum(self.surface_elev - self.invert_elev, 0.1).astype(np.float32)
        self.min_elev_2d = np.nan_to_num(
            static_data.nodes_2d['min_elevation'].values.astype(np.float32), nan=0.0)
        self.base_area_1d = np.nan_to_num(
            static_data.nodes_1d['base_area'].values.astype(np.float32), nan=0.0)
        
        # 1D→2D coupling map
        self.coupled_2d_for_1d = static_data.coupled_2d_for_1d
        
        # Downstream neighbor map (from_node -> to_node in 1D edge table).
        downstream_lists = [[] for _ in range(self.n_1d)]
        for fn, tn in zip(static_data.edges_1d['from_node'].values,
                          static_data.edges_1d['to_node'].values):
            fn_i = int(fn)
            tn_i = int(tn)
            if 0 <= fn_i < self.n_1d and 0 <= tn_i < self.n_1d:
                downstream_lists[fn_i].append(tn_i)
        max_down = max((len(v) for v in downstream_lists), default=0)
        self._downstream_idx = np.full((self.n_1d, max_down), -1, dtype=np.int64)
        self._downstream_mask = np.zeros((self.n_1d, max_down), dtype=np.float32)
        for i, neigh in enumerate(downstream_lists):
            if neigh:
                self._downstream_idx[i, :len(neigh)] = np.asarray(neigh, dtype=np.int64)
                self._downstream_mask[i, :len(neigh)] = 1.0

        # Directed 1D edge lookup (aligned with bidirectional edge_index_1d order):
        # [forward edges..., reverse edges...]
        src_1d = static_data.edges_1d['from_node'].values.astype(np.int64)
        dst_1d = static_data.edges_1d['to_node'].values.astype(np.int64)
        self._edge_src_1d = np.concatenate([src_1d, dst_1d]).astype(np.int64)
        self._edge_dst_1d = np.concatenate([dst_1d, src_1d]).astype(np.int64)
        if 'length' in static_data.edges_1d_static.columns:
            edge_len = static_data.edges_1d_static['length'].values.astype(np.float32)
            edge_len = np.nan_to_num(edge_len, nan=1.0, posinf=1.0, neginf=1.0)
            edge_len = np.maximum(edge_len, 1e-3)
        else:
            edge_len = np.ones(len(src_1d), dtype=np.float32)
        self._edge_len_1d = np.concatenate([edge_len, edge_len]).astype(np.float32)
        # 1D-2D coupling edge lookup (aligned with edge_index_1d_to_2d / edge_index_2d_to_1d order).
        self._edge_couple_1d = static_data.edges_1d2d['node_1d'].values.astype(np.int64)
        self._edge_couple_2d = static_data.edges_1d2d['node_2d'].values.astype(np.int64)
        
        # GPU cache (populated by .to())
        self._device: Optional[torch.device] = None
        self._gpu: Dict = {}
        self._batched_edge_cache: Dict[int, Dict] = {}
    
    @staticmethod
    def infer_wl_prev_steps(model_cfg: Dict, static_data: 'StaticGraphData') -> int:
        '''Infer wl_prev_steps from checkpoint config, falling back to model input dim.'''
        if 'wl_prev_steps' in model_cfg:
            return model_cfg['wl_prev_steps']
        base_in_1d = static_data.x_1d_static.shape[1] + FeatureBuilder.N_DYNAMIC_1D
        return max(0, model_cfg.get('in_1d', base_in_1d) - base_in_1d)

    # ── Dimension & name queries ──────────────────────────────────────
    
    @property
    def in_channels_1d(self) -> int:
        '''Total 1D input channels = static + dynamic.'''
        return self.static.x_1d_static.shape[1] + self.N_DYNAMIC_1D
    
    @property
    def in_channels_2d(self) -> int:
        '''Total 2D input channels = static + base dynamic + rain lags + future rain.'''
        return (self.static.x_2d_static.shape[1] +
                self.N_DYNAMIC_2D_BASE +
                self.rain_lag_steps)
    
    @property
    def edge_dim_1d(self) -> int:
        '''Number of 1D dynamic edge features, 0 if disabled.'''
        if not self.use_edge_features:
            return 0
        # [hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v]
        return 4
    
    @property
    def edge_dim_2d(self) -> int:
        '''2D-2D edge features are intentionally disabled.'''
        return 0
    
    @property
    def feature_names_1d(self) -> List[str]:
        return self.NAMES_1D_STATIC + self.NAMES_1D_DYNAMIC
    
    @property
    def feature_names_2d(self) -> List[str]:
        lag_names = [f'rain_lag{i+1}' for i in range(self.rain_lag_steps)]
        lead_names = [f'rain_lead{i+1}' for i in range(self.rain_future_steps)]
        return self.NAMES_2D_STATIC + self.NAMES_2D_DYNAMIC_BASE + lag_names + lead_names
    
    # ── GPU setup ─────────────────────────────────────────────────────
    
    def to(self, device: torch.device) -> 'FeatureBuilder':
        '''Move all static tensors to device. Call once before training/inference.'''
        self._device = device
        s = self.static
        self._gpu = {
            'x_1d_static': s.x_1d_static.to(device),
            'x_2d_static': s.x_2d_static.to(device),
            'invert_elev': torch.tensor(self.invert_elev, dtype=torch.float32, device=device),
            'surface_elev': torch.tensor(self.surface_elev, dtype=torch.float32, device=device),
            'pipe_depth': torch.tensor(self.pipe_depth, dtype=torch.float32, device=device),
            'base_area_1d': torch.tensor(self.base_area_1d, dtype=torch.float32, device=device),
            'min_elev_2d': torch.tensor(self.min_elev_2d, dtype=torch.float32, device=device),
            'coupled_map': torch.tensor(self.coupled_2d_for_1d, dtype=torch.long, device=device),
            'valid_coupling': torch.tensor(
                self.coupled_2d_for_1d >= 0, dtype=torch.bool, device=device),
            'downstream_idx': torch.tensor(self._downstream_idx, dtype=torch.long, device=device),
            'downstream_mask': torch.tensor(self._downstream_mask, dtype=torch.float32, device=device),
            'edge_src_1d': torch.tensor(self._edge_src_1d, dtype=torch.long, device=device),
            'edge_dst_1d': torch.tensor(self._edge_dst_1d, dtype=torch.long, device=device),
            'edge_len_1d': torch.tensor(self._edge_len_1d, dtype=torch.float32, device=device),
            'edge_couple_1d': torch.tensor(self._edge_couple_1d, dtype=torch.long, device=device),
            'edge_couple_2d': torch.tensor(self._edge_couple_2d, dtype=torch.long, device=device),
            'edge_index_dict': {
                ('1d', 'pipe', '1d'): s.edge_index_1d.to(device),
                ('2d', 'surface', '2d'): s.edge_index_2d.to(device),
                ('1d', 'couples', '2d'): s.edge_index_1d_to_2d.to(device),
                ('2d', 'couples', '1d'): s.edge_index_2d_to_1d.to(device),
            },
        }
        # Cache edge attributes on GPU when enabled
        if self.use_edge_features:
            ea = {}
            if s.edge_attr_1d is not None:
                ea[('1d', 'pipe', '1d')] = s.edge_attr_1d.to(device)
            self._gpu['edge_attr_dict'] = ea if ea else None
        else:
            self._gpu['edge_attr_dict'] = None
        self._batched_edge_cache = {}
        return self
    
    @property
    def device(self) -> Optional[torch.device]:
        return self._device
    
    @property
    def edge_index_dict(self) -> Dict:
        '''Cached edge_index_dict on device.'''
        return self._gpu['edge_index_dict']
    
    @property
    def edge_attr_dict(self) -> Optional[Dict]:
        '''Cached edge_attr_dict on device, or None if edge features disabled.'''
        return self._gpu.get('edge_attr_dict')

    def build_dynamic_edge_attr_1d_torch(
        self,
        wl_1d: torch.Tensor,
        wl_1d_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''Build dynamic 1D edge attrs from current/previous 1D water levels.

        Features per directed 1D edge:
          1) hydraulic_grad   = (wl_u - wl_v) / length
          2) dhydraulic_grad  = hydraulic_grad(t) - hydraulic_grad(t-1)
          3) dwl_u            = (wl_u(t) - wl_u(t-1)) / wl_1d_scale
          4) dwl_v            = (wl_v(t) - wl_v(t-1)) / wl_1d_scale
        '''
        g = self._gpu
        src = g['edge_src_1d']
        dst = g['edge_dst_1d']
        length = g['edge_len_1d']

        if wl_1d_prev is None:
            wl_1d_prev = wl_1d

        if wl_1d.dim() == 1:
            wl_u = wl_1d[src]
            wl_v = wl_1d[dst]
            wl_u_prev = wl_1d_prev[src]
            wl_v_prev = wl_1d_prev[dst]
            hydraulic_grad = (wl_u - wl_v) / length
            hydraulic_grad_prev = (wl_u_prev - wl_v_prev) / length
            dhydraulic_grad = hydraulic_grad - hydraulic_grad_prev
            dwl_u = (wl_u - wl_u_prev) / self.wl_1d_scale
            dwl_v = (wl_v - wl_v_prev) / self.wl_1d_scale
            return torch.stack([hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v], dim=1)

        if wl_1d.dim() == 2:
            wl_u = wl_1d[:, src]
            wl_v = wl_1d[:, dst]
            wl_u_prev = wl_1d_prev[:, src]
            wl_v_prev = wl_1d_prev[:, dst]
            hydraulic_grad = (wl_u - wl_v) / length.unsqueeze(0)
            hydraulic_grad_prev = (wl_u_prev - wl_v_prev) / length.unsqueeze(0)
            dhydraulic_grad = hydraulic_grad - hydraulic_grad_prev
            dwl_u = (wl_u - wl_u_prev) / self.wl_1d_scale
            dwl_v = (wl_v - wl_v_prev) / self.wl_1d_scale
            feat = torch.stack([hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v], dim=-1)
            return feat.reshape(-1, 4)

        raise ValueError(f"wl_1d must be [n_1d] or [batch, n_1d], got shape={tuple(wl_1d.shape)}")

    def build_edge_attr_dict_torch(
        self,
        wl_1d: torch.Tensor,
        wl_2d: torch.Tensor,
        wl_1d_prev: Optional[torch.Tensor] = None,
        wl_2d_prev: Optional[torch.Tensor] = None,
    ) -> Optional[Dict]:
        '''Build edge_attr_dict with dynamic 1D pipe + 1D-2D coupling edge features.'''
        if not self.use_edge_features:
            return None
        g = self._gpu
        c1 = g['edge_couple_1d']
        c2 = g['edge_couple_2d']
        if wl_1d_prev is None:
            wl_1d_prev = wl_1d
        if wl_2d_prev is None:
            wl_2d_prev = wl_2d

        if wl_1d.dim() == 1:
            wl1 = wl_1d[c1]
            wl2 = wl_2d[c2]
            wl1_prev = wl_1d_prev[c1]
            wl2_prev = wl_2d_prev[c2]
            head = (wl1 - wl2) / self.wl_1d_scale
            dhead = ((wl1 - wl2) - (wl1_prev - wl2_prev)) / self.wl_1d_scale
            dwl_1d = (wl1 - wl1_prev) / self.wl_1d_scale
            dwl_2d = (wl2 - wl2_prev) / self.wl_2d_scale
            couple_feat = torch.stack([head, dhead, dwl_1d, dwl_2d], dim=1)
        elif wl_1d.dim() == 2:
            wl1 = wl_1d[:, c1]
            wl2 = wl_2d[:, c2]
            wl1_prev = wl_1d_prev[:, c1]
            wl2_prev = wl_2d_prev[:, c2]
            head = (wl1 - wl2) / self.wl_1d_scale
            dhead = ((wl1 - wl2) - (wl1_prev - wl2_prev)) / self.wl_1d_scale
            dwl_1d = (wl1 - wl1_prev) / self.wl_1d_scale
            dwl_2d = (wl2 - wl2_prev) / self.wl_2d_scale
            couple_feat = torch.stack([head, dhead, dwl_1d, dwl_2d], dim=-1).reshape(-1, 4)
        else:
            raise ValueError(f"wl_1d must be [n_1d] or [batch, n_1d], got shape={tuple(wl_1d.shape)}")

        return {
            ('1d', 'pipe', '1d'): self.build_dynamic_edge_attr_1d_torch(
                wl_1d, wl_1d_prev=wl_1d_prev
            ),
            ('1d', 'couples', '2d'): couple_feat,
            ('2d', 'couples', '1d'): couple_feat,
        }
    
    # ── Batched edge indices (for multi-step training) ────────────────
    
    def get_batched_edge_indices(self, batch_size: int) -> Tuple[Dict, Optional[Dict]]:
        '''Get (or build & cache) batched edge indices and edge attrs.
        
        Returns:
            (edge_index_dict, edge_attr_dict) — edge_attr_dict is None when
            edge features are disabled.
        '''
        if batch_size in self._batched_edge_cache:
            return self._batched_edge_cache[batch_size]
        
        edge_dict = {}
        for key, edge_index in self._gpu['edge_index_dict'].items():
            offset_src = self.n_1d if key[0] == '1d' else self.n_2d
            offset_dst = self.n_1d if key[2] == '1d' else self.n_2d
            n_edges = edge_index.shape[1]
            batched = edge_index.repeat(1, batch_size)
            offsets = torch.arange(batch_size, device=self._device).repeat_interleave(n_edges)
            batched[0] += offsets * offset_src
            batched[1] += offsets * offset_dst
            edge_dict[key] = batched
        
        # Batch edge attributes (no offsets needed, just repeat per-edge features)
        ea_dict = None
        src_ea = self._gpu.get('edge_attr_dict')
        if src_ea:
            ea_dict = {}
            for key, ea in src_ea.items():
                ea_dict[key] = ea.repeat(batch_size, 1)
        
        result = (edge_dict, ea_dict)
        self._batched_edge_cache[batch_size] = result
        return result
    
    # ── Rain lag helpers ──────────────────────────────────────────────
    
    def compute_rain_lags_np(self, rain_2d_all: np.ndarray, t: int) -> np.ndarray:
        '''Compute [rain_lag_steps, n_2d] lagged rainfall at timestep t.'''
        lags = np.zeros((self.rain_lag_steps, self.n_2d), dtype=np.float32)
        for lag in range(self.rain_lag_steps):
            t_lag = t - lag - 1
            if t_lag >= 0:
                lags[lag] = rain_2d_all[t_lag]
        return lags
    
    def compute_rain_lags_torch(self, rain_2d_all: torch.Tensor, t: int) -> torch.Tensor:
        '''Compute [rain_lag_steps, n_2d] lagged rainfall at timestep t.'''
        lags = torch.zeros((self.rain_lag_steps, self.n_2d),
                           dtype=torch.float32, device=rain_2d_all.device)
        for lag in range(self.rain_lag_steps):
            t_lag = t - lag - 1
            if t_lag >= 0:
                lags[lag] = rain_2d_all[t_lag]
        return lags
    
    def compute_future_rain_np(self, rain_2d_all: np.ndarray, t: int) -> np.ndarray:
        '''Compute [rain_future_steps, n_2d] future rainfall at timestep t+1..t+F.'''
        if self.rain_future_steps <= 0:
            return np.zeros((0, self.n_2d), dtype=np.float32)
        n_timesteps = rain_2d_all.shape[0]
        leads = np.zeros((self.rain_future_steps, self.n_2d), dtype=np.float32)
        for lead in range(self.rain_future_steps):
            t_lead = t + lead + 1
            if t_lead < n_timesteps:
                leads[lead] = rain_2d_all[t_lead]
        return leads
    
    def compute_future_rain_torch(self, rain_2d_all: torch.Tensor, t: int) -> torch.Tensor:
        '''Compute [rain_future_steps, n_2d] future rainfall at timestep t+1..t+F on GPU.'''
        if self.rain_future_steps <= 0:
            return torch.zeros((0, self.n_2d), dtype=torch.float32, device=rain_2d_all.device)
        n_timesteps = rain_2d_all.shape[0]
        leads = torch.zeros((self.rain_future_steps, self.n_2d),
                            dtype=torch.float32, device=rain_2d_all.device)
        for lead in range(self.rain_future_steps):
            t_lead = t + lead + 1
            if t_lead < n_timesteps:
                leads[lead] = rain_2d_all[t_lead]
        return leads
    
    # ── Core: numpy dynamic features (for datasets) ──────────────────
    
    def build_dynamic_np(self, wl_1d: np.ndarray, wl_2d: np.ndarray,
                         rainfall: np.ndarray, rain_lags: np.ndarray,
                         cum_rainfall_2d: np.ndarray = None,
                         normalized_t: float = 0.0,
                         wl_prevs_1d: List[np.ndarray] = None,
                         wl_prevs_2d: List[np.ndarray] = None,
                         future_rain: np.ndarray = None,
                         n_total_timesteps: int = 0,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        '''Build dynamic features as numpy arrays for one timestep.
        
        Args:
            cum_rainfall_2d: [n_2d] cumulative rainfall up to this timestep
            normalized_t: float in [0,1] — event progress (t / n_total_timesteps)
            wl_prevs_1d: list of [n_1d] arrays — wl at t-1, t-2, ...
            wl_prevs_2d: list of [n_2d] arrays — wl at t-1, t-2, ...
            n_total_timesteps: total event length (for event-scale features)
        
        Returns:
            x_1d_dynamic [n_1d, N_DYNAMIC_1D]
            x_2d_dynamic [n_2d, N_DYNAMIC_2D_BASE + rain_lag_steps]
        '''
        if cum_rainfall_2d is None:
            cum_rainfall_2d = np.zeros(self.n_2d, dtype=np.float32)
        if future_rain is None and self.rain_future_steps > 0:
            future_rain = np.zeros((self.rain_future_steps, self.n_2d), dtype=np.float32)
        
        # ── 1D features ──
        pipe_fill = wl_1d - self.invert_elev
        head_above = wl_1d - self.surface_elev
        
        # Rain is treated as global forcing (shared across all 1D nodes).
        valid = self.coupled_2d_for_1d >= 0
        rain_global = float(np.mean(rainfall))
        coupled_rain = np.full(self.n_1d, rain_global, dtype=np.float32)
        coupled_rain_ind = np.full(self.n_1d, 1.0 if rain_global > 0.001 else 0.0, dtype=np.float32)
        
        cum_rain_global = float(np.mean(cum_rainfall_2d))
        coupled_cum_rain = np.full(self.n_1d, cum_rain_global, dtype=np.float32)
        
        # 3-step local rainfall accumulation at the coupled 2D node:
        # rain(t) + rain(t-1) + rain(t-2). This is less horizon-fragile than
        # high-order temporal derivatives and ranked highly in RF analysis.
        cum_rain_3_global = rain_global
        if self.rain_lag_steps > 0:
            cum_rain_3_global += float(np.mean(rain_lags[0]))
        if self.rain_lag_steps > 1:
            cum_rain_3_global += float(np.mean(rain_lags[1]))
        coupled_cum_rain_3 = np.full(self.n_1d, cum_rain_3_global, dtype=np.float32)
        
        norm_t_1d = np.full(self.n_1d, normalized_t, dtype=np.float32)
        
        # Global context features for 1D
        fill_frac = np.clip(pipe_fill / self.pipe_depth, -0.5, 3.0)
        downstream_mean_fill = fill_frac.copy()
        if self._downstream_idx.shape[1] > 0:
            for i in range(self.n_1d):
                neigh = self._downstream_idx[i]
                neigh = neigh[neigh >= 0]
                if neigh.size > 0:
                    downstream_mean_fill[i] = fill_frac[neigh].mean()
        mean_fill = np.full(self.n_1d, fill_frac.mean(), dtype=np.float32)
        frac_surch = np.full(self.n_1d, (head_above > 0).mean(), dtype=np.float32)
        global_storage_proxy = np.full(
            self.n_1d, float((pipe_fill * self.base_area_1d).mean()), dtype=np.float32
        )
        sys_trend = np.full(self.n_1d, 0.0, dtype=np.float32)
        if wl_prevs_1d and len(wl_prevs_1d) > 0:
            sys_trend[:] = ((wl_1d - wl_prevs_1d[0]) / self.wl_1d_scale).mean()
        
        # Head difference between 1D pipe node and its coupled 2D surface node.
        # Positive = pipe WL above surface = surcharge outflow to surface.
        # Negative = surface above pipe = inflow from surface into pipe.
        head_diff_1d_2d = np.zeros(self.n_1d, dtype=np.float32)
        if valid.any():
            coupled_2d_wl = wl_2d[self.coupled_2d_for_1d[valid]]
            head_diff_1d_2d[valid] = (wl_1d[valid] - coupled_2d_wl)
        
        # Event-scale features
        n_ts_scaled = np.full(self.n_1d, n_total_timesteps / 500.0, dtype=np.float32)
        remaining = 1.0 - normalized_t  # equivalent to (n_total - t) / n_total
        remaining_frac = np.full(self.n_1d, remaining, dtype=np.float32)
        
        # WL velocity and acceleration (first and second finite differences)
        wl_prev1 = wl_prevs_1d[0] if (wl_prevs_1d and len(wl_prevs_1d) > 0) \
            else np.zeros(self.n_1d, dtype=np.float32)
        wl_prev2 = wl_prevs_1d[1] if (wl_prevs_1d and len(wl_prevs_1d) > 1) \
            else np.zeros(self.n_1d, dtype=np.float32)
        wl_velocity = (wl_1d - wl_prev1) / self.wl_1d_scale
        wl_acceleration = (wl_1d - 2.0 * wl_prev1 + wl_prev2) / self.wl_1d_scale

        feats_1d = [
            wl_1d / self.wl_1d_scale,
            pipe_fill / self.pipe_fill_scale,
            head_above / self.head_above_scale,
            coupled_rain / self.rainfall_scale,
            coupled_rain_ind,
            coupled_cum_rain / self.cum_rainfall_scale,
            norm_t_1d,
            fill_frac,
            mean_fill,
            frac_surch,
            sys_trend,
            coupled_cum_rain_3 / self.cum_rainfall_scale,
            downstream_mean_fill,
            global_storage_proxy / (self.pipe_fill_scale * max(self.base_area_1d.mean(), 1.0)),
            head_diff_1d_2d / self.wl_1d_scale,
            n_ts_scaled,
            remaining_frac,
            wl_velocity,
            wl_acceleration,
        ]
        # 1D rain lag features from global rainfall history.
        for i in range(self.rain_lag_steps):
            lag_global = float(np.mean(rain_lags[i]))
            feats_1d.append(np.full(self.n_1d, lag_global, dtype=np.float32) / self.rainfall_scale)
        # 1D rain lead features from global future rainfall.
        for i in range(self.rain_future_steps):
            lead_global = float(np.mean(future_rain[i])) if (future_rain is not None and future_rain.size > 0) else 0.0
            feats_1d.append(np.full(self.n_1d, lead_global, dtype=np.float32) / self.rainfall_scale)
        for i in range(self.wl_prev_steps):
            wp = wl_prevs_1d[i] if (wl_prevs_1d and i < len(wl_prevs_1d)) \
                else np.zeros(self.n_1d, dtype=np.float32)
            feats_1d.append(wp / self.wl_1d_scale)
        x_1d = np.stack(feats_1d, axis=1).astype(np.float32)
        
        # ── 2D features ──
        pond_depth = wl_2d - self.min_elev_2d
        rain_ind = (rainfall > 0.001).astype(np.float32)
        norm_t_2d = np.full(self.n_2d, normalized_t, dtype=np.float32)
        
        base = [
            wl_2d / self.wl_2d_scale,
            rainfall / self.rainfall_scale,
            pond_depth / self.pond_depth_scale,
            rain_ind,
            cum_rainfall_2d / self.cum_rainfall_scale,
            norm_t_2d,
        ]
        for i in range(self.wl_prev_steps):
            wp = wl_prevs_2d[i] if (wl_prevs_2d and i < len(wl_prevs_2d)) \
                else np.zeros(self.n_2d, dtype=np.float32)
            base.append(wp / self.wl_2d_scale)
        lag_feats = [rain_lags[i] / self.rainfall_scale for i in range(self.rain_lag_steps)]
        lead_feats = []
        if self.rain_future_steps > 0 and future_rain is not None and future_rain.size > 0:
            for i in range(self.rain_future_steps):
                lead_feats.append(future_rain[i] / self.rainfall_scale)
        x_2d = np.stack(base + lag_feats + lead_feats, axis=1).astype(np.float32)
        
        return x_1d, x_2d
    
    # ── Core: torch dynamic features (for GPU training/inference) ─────
    
    def build_dynamic_torch(self, wl_1d: torch.Tensor, wl_2d: torch.Tensor,
                            rainfall: torch.Tensor, rain_lags: torch.Tensor,
                            cum_rainfall_2d: torch.Tensor = None,
                            normalized_t: float = 0.0,
                            wl_prevs_1d: List[torch.Tensor] = None,
                            wl_prevs_2d: List[torch.Tensor] = None,
                            future_rain: torch.Tensor = None,
                            n_total_timesteps = 0,
                            global_rain_1d: torch.Tensor = None,
                            global_rain_lags_1d: torch.Tensor = None,
                            global_future_rain_1d: torch.Tensor = None,
                            global_cum_rain_1d: torch.Tensor = None,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Build dynamic features on GPU.
        
        Handles both unbatched [n_nodes] and batched [batch, n_nodes].
        
        Args:
            rain_lags: [rain_lag_steps, n_2d] or [batch, rain_lag_steps, n_2d]
            cum_rainfall_2d: [n_2d] or [batch, n_2d] cumulative rainfall
            normalized_t: float or [batch] tensor — event progress in [0,1]
            wl_prevs_1d: list of [n_1d] or [batch, n_1d] — wl at t-1, t-2, ...
            wl_prevs_2d: list of [n_2d] or [batch, n_2d] — wl at t-1, t-2, ...
            n_total_timesteps: int, float, or [batch] tensor — total event length
        '''
        g = self._gpu
        batched = wl_1d.dim() == 2
        
        if cum_rainfall_2d is None:
            cum_rainfall_2d = torch.zeros_like(wl_2d)
        if future_rain is None and self.rain_future_steps > 0:
            if batched:
                future_rain = torch.zeros((wl_1d.shape[0], self.rain_future_steps, self.n_2d),
                                          dtype=torch.float32, device=self._device)
            else:
                future_rain = torch.zeros((self.rain_future_steps, self.n_2d),
                                          dtype=torch.float32, device=self._device)
        
        # ── 1D ──
        pipe_fill = wl_1d - g['invert_elev']
        head_above = wl_1d - g['surface_elev']
        
        if batched:
            # Rain is treated as global forcing shared by all 1D nodes.
            if global_rain_1d is None:
                rain_global = rainfall.mean(dim=-1, keepdim=True)
            else:
                rain_global = global_rain_1d.unsqueeze(-1) if global_rain_1d.dim() == 1 else global_rain_1d
            coupled_rain = rain_global.expand_as(wl_1d)
            if global_cum_rain_1d is None:
                cum_rain_global = cum_rainfall_2d.mean(dim=-1, keepdim=True)
            else:
                cum_rain_global = global_cum_rain_1d.unsqueeze(-1) if global_cum_rain_1d.dim() == 1 else global_cum_rain_1d
            coupled_cum_rain = cum_rain_global.expand_as(wl_1d)
            if isinstance(normalized_t, torch.Tensor):
                norm_t_1d = normalized_t.unsqueeze(-1).expand(-1, self.n_1d)
            else:
                norm_t_1d = torch.full_like(wl_1d, normalized_t)
        else:
            rain_global = rainfall.mean()
            coupled_rain = torch.full((self.n_1d,), rain_global,
                                      dtype=torch.float32, device=self._device)
            cum_rain_global = cum_rainfall_2d.mean()
            coupled_cum_rain = torch.full((self.n_1d,), cum_rain_global,
                                          dtype=torch.float32, device=self._device)
            norm_t_1d = torch.full((self.n_1d,), normalized_t,
                                   dtype=torch.float32, device=self._device)
        coupled_rain_ind = (coupled_rain > 0.001).float()
        
        # 3-step global rainfall accumulation.
        if batched:
            if global_rain_1d is None:
                cum_rain_3_global = rainfall.mean(dim=-1, keepdim=True)
            else:
                cum_rain_3_global = global_rain_1d.unsqueeze(-1) if global_rain_1d.dim() == 1 else global_rain_1d
            if self.rain_lag_steps > 0:
                if global_rain_lags_1d is None:
                    cum_rain_3_global = cum_rain_3_global + rain_lags[:, 0, :].mean(dim=-1, keepdim=True)
                else:
                    cum_rain_3_global = cum_rain_3_global + global_rain_lags_1d[:, 0:1]
            if self.rain_lag_steps > 1:
                if global_rain_lags_1d is None:
                    cum_rain_3_global = cum_rain_3_global + rain_lags[:, 1, :].mean(dim=-1, keepdim=True)
                else:
                    cum_rain_3_global = cum_rain_3_global + global_rain_lags_1d[:, 1:2]
            coupled_cum_rain_3 = cum_rain_3_global.expand_as(wl_1d)
        else:
            cum_rain_3_global = rainfall.mean()
            if self.rain_lag_steps > 0:
                cum_rain_3_global = cum_rain_3_global + rain_lags[0].mean()
            if self.rain_lag_steps > 1:
                cum_rain_3_global = cum_rain_3_global + rain_lags[1].mean()
            coupled_cum_rain_3 = torch.full_like(wl_1d, cum_rain_3_global)
        
        # Global context features for 1D
        fill_frac = (pipe_fill / g['pipe_depth']).clamp(-0.5, 3.0)
        # Downstream neighborhood mean fill fraction.
        down_idx = g['downstream_idx']
        down_mask = g['downstream_mask']
        if down_idx.numel() > 0 and down_idx.shape[1] > 0:
            down_idx_clamped = down_idx.clamp(min=0)
            if batched:
                down_fill_vals = fill_frac[:, down_idx_clamped]  # [B, n_1d, max_down]
                down_mask_b = down_mask.unsqueeze(0)  # [1, n_1d, max_down]
                denom = down_mask_b.sum(dim=-1).clamp(min=1.0)
                down_mean_fill = (down_fill_vals * down_mask_b).sum(dim=-1) / denom
                no_down = (down_mask_b.sum(dim=-1) < 0.5)
                down_mean_fill = torch.where(no_down, fill_frac, down_mean_fill)
            else:
                down_fill_vals = fill_frac[down_idx_clamped]  # [n_1d, max_down]
                denom = down_mask.sum(dim=-1).clamp(min=1.0)
                down_mean_fill = (down_fill_vals * down_mask).sum(dim=-1) / denom
                no_down = (down_mask.sum(dim=-1) < 0.5)
                down_mean_fill = torch.where(no_down, fill_frac, down_mean_fill)
        else:
            down_mean_fill = fill_frac
        if batched:
            mean_fill = fill_frac.mean(dim=-1, keepdim=True).expand_as(wl_1d)
            frac_surch = (head_above > 0).float().mean(dim=-1, keepdim=True).expand_as(wl_1d)
            storage_proxy = (pipe_fill * g['base_area_1d']).mean(dim=-1, keepdim=True).expand_as(wl_1d)
            if wl_prevs_1d and len(wl_prevs_1d) > 0:
                sys_trend = ((wl_1d - wl_prevs_1d[0]) / self.wl_1d_scale).mean(dim=-1, keepdim=True).expand_as(wl_1d)
            else:
                sys_trend = torch.zeros_like(wl_1d)
        else:
            mean_fill = torch.full_like(wl_1d, fill_frac.mean().item())
            frac_surch = torch.full_like(wl_1d, (head_above > 0).float().mean().item())
            storage_proxy = torch.full_like(wl_1d, (pipe_fill * g['base_area_1d']).mean().item())
            if wl_prevs_1d and len(wl_prevs_1d) > 0:
                sys_trend = torch.full_like(wl_1d, ((wl_1d - wl_prevs_1d[0]) / self.wl_1d_scale).mean().item())
            else:
                sys_trend = torch.zeros_like(wl_1d)
        
        # Head difference between 1D pipe and coupled 2D surface node
        if batched:
            head_diff_1d_2d = torch.zeros_like(wl_1d)
            head_diff_1d_2d[:, g['valid_coupling']] = (
                wl_1d[:, g['valid_coupling']] -
                wl_2d[:, g['coupled_map'][g['valid_coupling']]]
            )
        else:
            head_diff_1d_2d = torch.zeros_like(wl_1d)
            if g['valid_coupling'].any():
                head_diff_1d_2d[g['valid_coupling']] = (
                    wl_1d[g['valid_coupling']] -
                    wl_2d[g['coupled_map'][g['valid_coupling']]]
                )
        
        # Event-scale features: n_timesteps_scaled and remaining_frac
        if batched:
            if isinstance(n_total_timesteps, torch.Tensor):
                n_ts_scaled = (n_total_timesteps / 500.0).unsqueeze(-1).expand(-1, self.n_1d)
            else:
                n_ts_scaled = torch.full_like(wl_1d, float(n_total_timesteps) / 500.0)
            if isinstance(normalized_t, torch.Tensor):
                remaining_frac = (1.0 - normalized_t).unsqueeze(-1).expand(-1, self.n_1d)
            else:
                remaining_frac = torch.full_like(wl_1d, 1.0 - normalized_t)
        else:
            n_ts_val = float(n_total_timesteps) if not isinstance(n_total_timesteps, torch.Tensor) \
                else n_total_timesteps.item()
            n_ts_scaled = torch.full((self.n_1d,), n_ts_val / 500.0,
                                     dtype=torch.float32, device=self._device)
            remaining_frac = torch.full((self.n_1d,), 1.0 - float(normalized_t),
                                        dtype=torch.float32, device=self._device)
        
        # WL velocity and acceleration (first and second finite differences)
        wl_prev1_t = wl_prevs_1d[0] if (wl_prevs_1d and len(wl_prevs_1d) > 0) \
            else torch.zeros_like(wl_1d)
        wl_prev2_t = wl_prevs_1d[1] if (wl_prevs_1d and len(wl_prevs_1d) > 1) \
            else torch.zeros_like(wl_1d)
        wl_velocity_t = (wl_1d - wl_prev1_t) / self.wl_1d_scale
        wl_acceleration_t = (wl_1d - 2.0 * wl_prev1_t + wl_prev2_t) / self.wl_1d_scale

        feats_1d = [
            wl_1d / self.wl_1d_scale,
            pipe_fill / self.pipe_fill_scale,
            head_above / self.head_above_scale,
            coupled_rain / self.rainfall_scale,
            coupled_rain_ind,
            coupled_cum_rain / self.cum_rainfall_scale,
            norm_t_1d,
            fill_frac,
            mean_fill,
            frac_surch,
            sys_trend,
            coupled_cum_rain_3 / self.cum_rainfall_scale,
            down_mean_fill,
            storage_proxy / (self.pipe_fill_scale * max(float(self.base_area_1d.mean()), 1.0)),
            head_diff_1d_2d / self.wl_1d_scale,
            n_ts_scaled,
            remaining_frac,
            wl_velocity_t,
            wl_acceleration_t,
        ]
        # 1D rain lag features from global rainfall history.
        for i in range(self.rain_lag_steps):
            if batched:
                if global_rain_lags_1d is None:
                    lag_global = rain_lags[:, i, :].mean(dim=-1, keepdim=True)
                else:
                    lag_global = global_rain_lags_1d[:, i:i+1]
                coupled_rain_lag_t = lag_global.expand_as(wl_1d)
            else:
                coupled_rain_lag_t = torch.full_like(wl_1d, rain_lags[i].mean())
            feats_1d.append(coupled_rain_lag_t / self.rainfall_scale)
        # 1D rain lead features from global future rainfall.
        for i in range(self.rain_future_steps):
            if future_rain is not None and future_rain.numel() > 0:
                if batched:
                    if global_future_rain_1d is None:
                        lead_global = future_rain[:, i, :].mean(dim=-1, keepdim=True)
                    else:
                        lead_global = global_future_rain_1d[:, i:i+1]
                    coupled_rain_lead_t = lead_global.expand_as(wl_1d)
                else:
                    coupled_rain_lead_t = torch.full_like(wl_1d, future_rain[i].mean())
            else:
                coupled_rain_lead_t = torch.zeros_like(wl_1d)
            feats_1d.append(coupled_rain_lead_t / self.rainfall_scale)
        for i in range(self.wl_prev_steps):
            wp = wl_prevs_1d[i] if (wl_prevs_1d and i < len(wl_prevs_1d)) \
                else torch.zeros_like(wl_1d)
            feats_1d.append(wp / self.wl_1d_scale)
        x_1d_dyn = torch.stack(feats_1d, dim=-1)
        
        # ── 2D ──
        pond_depth = wl_2d - g['min_elev_2d']
        rain_ind = (rainfall > 0.001).float()
        
        if batched:
            if isinstance(normalized_t, torch.Tensor):
                norm_t_2d = normalized_t.unsqueeze(-1).expand(-1, self.n_2d)
            else:
                norm_t_2d = torch.full_like(wl_2d, normalized_t)
        else:
            norm_t_2d = torch.full((self.n_2d,), normalized_t,
                                   dtype=torch.float32, device=self._device)
        
        feats_2d_base = [
            wl_2d / self.wl_2d_scale,
            rainfall / self.rainfall_scale,
            pond_depth / self.pond_depth_scale,
            rain_ind,
            cum_rainfall_2d / self.cum_rainfall_scale,
            norm_t_2d,
        ]
        for i in range(self.wl_prev_steps):
            wp = wl_prevs_2d[i] if (wl_prevs_2d and i < len(wl_prevs_2d)) \
                else torch.zeros_like(wl_2d)
            feats_2d_base.append(wp / self.wl_2d_scale)
        x_2d_base = torch.stack(feats_2d_base, dim=-1)
        
        if batched:
            rain_lag_t = rain_lags.transpose(-2, -1) / self.rainfall_scale
        else:
            rain_lag_t = rain_lags.t() / self.rainfall_scale
        
        lead_t = None
        if self.rain_future_steps > 0 and future_rain is not None and future_rain.numel() > 0:
            if batched:
                lead_t = future_rain.transpose(-2, -1) / self.rainfall_scale
            else:
                lead_t = future_rain.t() / self.rainfall_scale
        
        if lead_t is not None:
            x_2d_dyn = torch.cat([x_2d_base, rain_lag_t, lead_t], dim=-1)
        else:
            x_2d_dyn = torch.cat([x_2d_base, rain_lag_t], dim=-1)
        return x_1d_dyn, x_2d_dyn
    
    # ── High-level: full x_dict with static + dynamic (GPU) ──────────
    
    def build_x_dict_torch(self, wl_1d: torch.Tensor, wl_2d: torch.Tensor,
                           rainfall: torch.Tensor, rain_lags: torch.Tensor,
                           cum_rainfall_2d: torch.Tensor = None,
                           normalized_t: float = 0.0,
                           wl_prevs_1d: List[torch.Tensor] = None,
                           wl_prevs_2d: List[torch.Tensor] = None,
                           future_rain: torch.Tensor = None,
                           n_total_timesteps = 0,
                           global_rain_1d: torch.Tensor = None,
                           global_rain_lags_1d: torch.Tensor = None,
                           global_future_rain_1d: torch.Tensor = None,
                           global_cum_rain_1d: torch.Tensor = None,
                           ) -> Dict[str, torch.Tensor]:
        '''Build complete x_dict for model forward pass.
        
        Unbatched → {'1d': [n_1d, F], '2d': [n_2d, F]}
        Batched   → {'1d': [B*n_1d, F], '2d': [B*n_2d, F]}  (flattened)
        '''
        g = self._gpu
        x_1d_dyn, x_2d_dyn = self.build_dynamic_torch(
            wl_1d, wl_2d, rainfall, rain_lags,
            cum_rainfall_2d=cum_rainfall_2d, normalized_t=normalized_t,
            wl_prevs_1d=wl_prevs_1d, wl_prevs_2d=wl_prevs_2d,
            future_rain=future_rain,
            n_total_timesteps=n_total_timesteps,
            global_rain_1d=global_rain_1d,
            global_rain_lags_1d=global_rain_lags_1d,
            global_future_rain_1d=global_future_rain_1d,
            global_cum_rain_1d=global_cum_rain_1d,
        )
        
        batched = wl_1d.dim() == 2
        if batched:
            bs = wl_1d.shape[0]
            x_1d_s = g['x_1d_static'].unsqueeze(0).expand(bs, -1, -1)
            x_2d_s = g['x_2d_static'].unsqueeze(0).expand(bs, -1, -1)
            x_1d = torch.cat([x_1d_s, x_1d_dyn], dim=-1).reshape(-1, self.in_channels_1d)
            x_2d = torch.cat([x_2d_s, x_2d_dyn], dim=-1).reshape(-1, self.in_channels_2d)
        else:
            x_1d = torch.cat([g['x_1d_static'], x_1d_dyn], dim=1)
            x_2d = torch.cat([g['x_2d_static'], x_2d_dyn], dim=1)
        
        return {'1d': x_1d, '2d': x_2d}
    
    # ── AR rollout convenience: one-shot x_dict from state ──────────
    
    def build_x_dict_from_state(self, wl_1d: torch.Tensor, wl_2d: torch.Tensor,
                                rainfall_all_gpu: torch.Tensor, 
                                cum_rain_gpu: torch.Tensor,
                                t: int, n_total_timesteps: int,
                                wl_prevs_1d: List[torch.Tensor] = None,
                                wl_prevs_2d: List[torch.Tensor] = None,
                                ) -> Dict[str, torch.Tensor]:
        '''Build x_dict from raw AR state — computes rain lags and normalized_t.
        
        Args:
            wl_1d: [n_1d] current water levels
            wl_2d: [n_2d] current water levels
            rainfall_all_gpu: [n_timesteps, n_2d] full rainfall on GPU
            cum_rain_gpu: [n_timesteps, n_2d] precomputed cumulative rainfall on GPU
            t: current timestep
            n_total_timesteps: total event length
            wl_prevs_1d: list of [n_1d] — wl at t-1, t-2, ...
            wl_prevs_2d: list of [n_2d] — wl at t-1, t-2, ...
        '''
        rain_lags = self.compute_rain_lags_torch(rainfall_all_gpu, t)
        cum_rain_t = cum_rain_gpu[t]
        normalized_t = t / max(n_total_timesteps - 1, 1)
        future_rain = None
        if self.rain_future_steps > 0:
            future_rain = self.compute_future_rain_torch(rainfall_all_gpu, t)
        return self.build_x_dict_torch(
            wl_1d, wl_2d, rainfall_all_gpu[t], rain_lags,
            cum_rainfall_2d=cum_rain_t, normalized_t=normalized_t,
            wl_prevs_1d=wl_prevs_1d, wl_prevs_2d=wl_prevs_2d,
            future_rain=future_rain,
            n_total_timesteps=n_total_timesteps,
        )
    
    # ── Dataset convenience: build complete HeteroData ────────────────
    
    def build_hetero_graph(self, event_data: Dict[str, np.ndarray], t: int,
                           model_id: int = 0) -> HeteroData:
        '''Build a complete HeteroData graph sample at timestep t.'''
        wl_1d = event_data['wl_1d'][t]
        wl_2d = event_data['wl_2d'][t]
        rainfall = event_data['rain_2d'][t]
        rain_lags = self.compute_rain_lags_np(event_data['rain_2d'], t)
        
        cum_rain = event_data['cum_rain_2d'][t] if 'cum_rain_2d' in event_data else \
                   event_data['rain_2d'][:t+1].sum(axis=0)
        n_total = event_data['wl_1d'].shape[0]
        normalized_t = t / max(n_total - 1, 1)
        
        wl_prevs_1d = None
        wl_prevs_2d = None
        if self.wl_prev_steps > 0:
            wl_prevs_1d = []
            wl_prevs_2d = []
            for lag in range(1, self.wl_prev_steps + 1):
                t_lag = max(t - lag, 0)
                wl_prevs_1d.append(event_data['wl_1d'][t_lag].astype(np.float32))
                wl_prevs_2d.append(event_data['wl_2d'][t_lag].astype(np.float32))
        
        future_rain = None
        if self.rain_future_steps > 0:
            future_rain = self.compute_future_rain_np(event_data['rain_2d'], t)
        
        x_1d_dyn, x_2d_dyn = self.build_dynamic_np(
            wl_1d, wl_2d, rainfall, rain_lags,
            cum_rainfall_2d=cum_rain, normalized_t=normalized_t,
            wl_prevs_1d=wl_prevs_1d, wl_prevs_2d=wl_prevs_2d,
            future_rain=future_rain,
            n_total_timesteps=n_total,
        )
        
        data = HeteroData()
        
        # Node features: static (pre-normalized) + dynamic
        data['1d'].x = torch.cat([
            self.static.x_1d_static, torch.from_numpy(x_1d_dyn)
        ], dim=1)
        data['2d'].x = torch.cat([
            self.static.x_2d_static, torch.from_numpy(x_2d_dyn)
        ], dim=1)
        
        # Targets: Δwl
        data['1d'].y = torch.from_numpy(
            (event_data['wl_1d'][t + 1] - wl_1d).astype(np.float32))
        data['2d'].y = torch.from_numpy(
            (event_data['wl_2d'][t + 1] - wl_2d).astype(np.float32))
        
        # Current water levels (for autoregressive)
        data['1d'].wl = torch.from_numpy(wl_1d)
        data['2d'].wl = torch.from_numpy(wl_2d)
        # Previous-step water levels (for single-step dynamic coupling edge attrs)
        t_prev = max(t - 1, 0)
        data['1d'].wl_prev = torch.from_numpy(event_data['wl_1d'][t_prev].astype(np.float32))
        data['2d'].wl_prev = torch.from_numpy(event_data['wl_2d'][t_prev].astype(np.float32))
        # Persist feature scales in-batch so extract_edge_attr_dict can match train-time scaling.
        data.wl_1d_scale = torch.tensor(self.wl_1d_scale, dtype=torch.float32)
        data.wl_2d_scale = torch.tensor(self.wl_2d_scale, dtype=torch.float32)
        
        # Edge indices (bidirectional)
        data['1d', 'pipe', '1d'].edge_index = self.static.edge_index_1d
        data['2d', 'surface', '2d'].edge_index = self.static.edge_index_2d
        data['1d', 'couples', '2d'].edge_index = self.static.edge_index_1d_to_2d
        data['2d', 'couples', '1d'].edge_index = self.static.edge_index_2d_to_1d
        data['1d', 'pipe', '1d'].edge_len = torch.from_numpy(self._edge_len_1d)
        
        # Edge attributes
        if self.static.edge_attr_1d is not None:
            data['1d', 'pipe', '1d'].edge_attr = self.static.edge_attr_1d
        
        data.model_id = model_id
        data.timestep = t
        return data


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

class StaticGraphData:
    '''Precomputed static graph structure for a model.'''
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        
        # Load static node features
        self.nodes_1d = pd.read_csv(model_path / '1d_nodes_static.csv')
        self.nodes_2d = pd.read_csv(model_path / '2d_nodes_static.csv')
        
        # Load edge indices
        self.edges_1d = pd.read_csv(model_path / '1d_edge_index.csv')
        self.edges_2d = pd.read_csv(model_path / '2d_edge_index.csv')
        self.edges_1d2d = pd.read_csv(model_path / '1d2d_connections.csv')
        
        # Load edge static features
        self.edges_1d_static = pd.read_csv(model_path / '1d_edges_static.csv')
        self.edges_2d_static = pd.read_csv(model_path / '2d_edges_static.csv')
        
        # Build tensors
        self._build_tensors()
    
    def _build_tensors(self):
        '''Pre-compute tensors for fast graph construction.'''
        
        # Node counts
        self.num_1d = len(self.nodes_1d)
        self.num_2d = len(self.nodes_2d)
        
        # 1D node static features: [position_x, position_y, depth, invert_elev, surface_elev, base_area]
        x_1d_np = self.nodes_1d[['position_x', 'position_y', 'depth', 
                          'invert_elevation', 'surface_elevation', 'base_area']].fillna(0).values
        self.x_1d_static = torch.tensor(
            np.nan_to_num(x_1d_np, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float32
        )
        
        # 2D node static features: [position_x, position_y, area, roughness, min_elev, elev, aspect, curvature, flow_acc]
        cols_2d = ['position_x', 'position_y', 'area', 'roughness', 
                   'min_elevation', 'elevation', 'aspect', 'curvature', 'flow_accumulation']
        x_2d_np = self.nodes_2d[cols_2d].fillna(0).values
        self.x_2d_static = torch.tensor(
            np.nan_to_num(x_2d_np, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float32
        )
        
        # Check for NaN and report
        if torch.isnan(self.x_1d_static).any():
            logger.warning("NaN detected in 1D static features!")
        if torch.isnan(self.x_2d_static).any():
            logger.warning("NaN detected in 2D static features!")
        
        # NORMALIZE static features using Min-Max scaling: (x - min) / (max - min) → [0, 1]
        # Min-max preserves relative relationships (especially important for elevations)
        self.x_1d_min = self.x_1d_static.min(dim=0, keepdim=True).values
        self.x_1d_max = self.x_1d_static.max(dim=0, keepdim=True).values
        self.x_1d_range = (self.x_1d_max - self.x_1d_min).clamp(min=1e-6)
        self.x_1d_static = (self.x_1d_static - self.x_1d_min) / self.x_1d_range
        
        self.x_2d_min = self.x_2d_static.min(dim=0, keepdim=True).values
        self.x_2d_max = self.x_2d_static.max(dim=0, keepdim=True).values
        self.x_2d_range = (self.x_2d_max - self.x_2d_min).clamp(min=1e-6)
        self.x_2d_static = (self.x_2d_static - self.x_2d_min) / self.x_2d_range
        
        logger.debug(f"Static features min-max normalized - 1D range: [{self.x_1d_static.min():.2f}, {self.x_1d_static.max():.2f}]")
        logger.debug(f"Static features min-max normalized - 2D range: [{self.x_2d_static.min():.2f}, {self.x_2d_static.max():.2f}]")
        
        # Edge indices (bidirectional)
        # 1D-1D pipes
        src_1d = self.edges_1d['from_node'].values
        dst_1d = self.edges_1d['to_node'].values
        self.edge_index_1d = torch.tensor(
            np.stack([np.concatenate([src_1d, dst_1d]), 
                      np.concatenate([dst_1d, src_1d])]),
            dtype=torch.long
        )
        
        # 2D-2D surface
        src_2d = self.edges_2d['from_node'].values
        dst_2d = self.edges_2d['to_node'].values
        self.edge_index_2d = torch.tensor(
            np.stack([np.concatenate([src_2d, dst_2d]), 
                      np.concatenate([dst_2d, src_2d])]),
            dtype=torch.long
        )
        
        # 1D-2D coupling (bidirectional)
        node_1d = self.edges_1d2d['node_1d'].values
        node_2d = self.edges_1d2d['node_2d'].values
        self.edge_index_1d_to_2d = torch.tensor(
            np.stack([node_1d, node_2d]), dtype=torch.long
        )
        self.edge_index_2d_to_1d = torch.tensor(
            np.stack([node_2d, node_1d]), dtype=torch.long
        )
        
        # 1D→2D coupling map: maps each 1D node to its coupled 2D node (or -1 if none)
        self.coupled_2d_for_1d = np.full(self.num_1d, -1, dtype=np.int64)
        for node_1d, node_2d in zip(self.edges_1d2d['node_1d'].values,
                                     self.edges_1d2d['node_2d'].values):
            self.coupled_2d_for_1d[node_1d] = node_2d
        
        # Edge static features
        self._build_edge_features()
    
    def _build_edge_features(self):
        '''Build edge attribute tensors with proper directionality.
        
        For bidirectional edges, we flip signed features (slope, relative positions)
        so the model can distinguish upstream from downstream.
        '''
        # 1D edge features: [length, diameter, roughness, slope]
        cols = ['length', 'diameter', 'roughness', 'slope']
        available = [c for c in cols if c in self.edges_1d_static.columns]
        if available:
            edge_feat = self.edges_1d_static[available].fillna(0).values.astype(np.float32)
            
            # Reverse direction: flip slope sign (critical for flow direction)
            edge_feat_rev = edge_feat.copy()
            if 'slope' in available:
                idx = available.index('slope')
                edge_feat_rev[:, idx] *= -1.0
            
            # Standardize edge features before concat
            mean = edge_feat.mean(axis=0, keepdims=True)
            std = edge_feat.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            edge_feat = (edge_feat - mean) / std
            edge_feat_rev = (edge_feat_rev - mean) / std  # Same scaling
            
            self.edge_attr_1d = torch.tensor(
                np.concatenate([edge_feat, edge_feat_rev], axis=0),
                dtype=torch.float32
            )
        else:
            self.edge_attr_1d = None
        
        # 2D edge attributes are disabled by design.
        self.edge_attr_2d = None


class FloodEventDataset(torch.utils.data.Dataset):
    '''
    Optimized dataset for flood events.
    Uses pre-indexed numpy arrays for O(1) timestep lookup instead of O(N) pandas filtering.
    '''
    
    def __init__(self, model_id: int, events: List[str], static_data: StaticGraphData, 
                 mode: str = 'train', feature_builder: 'FeatureBuilder' = None,
                 feature_scales: Dict[str, float] = None):
        self.model_id = model_id
        self.events = events
        self.static = static_data
        self.mode = mode
        self.model_path = static_data.model_path
        
        # FeatureBuilder: single source of truth for feature construction
        if feature_builder is not None:
            self.fb = feature_builder
        else:
            # Backward compat: create from feature_scales
            self.fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                                     rain_lag_steps=Config.RAIN_LAG_STEPS,
                                     rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
        
        # Get timestep counts per event
        self.event_info = []
        total = 0
        for event in events:
            event_path = self.model_path / event
            ts = pd.read_csv(event_path / 'timesteps.csv')
            n_timesteps = len(ts) - 1  # -1 because we predict next step (delta)
            
            # Skip warmup timesteps for training targets
            start_t = Config.WARMUP_TIMESTEPS if mode == 'train' else 0
            valid_timesteps = max(0, n_timesteps - start_t)
            
            self.event_info.append({
                'event': event,
                'start_idx': total,
                'n_timesteps': valid_timesteps,
                'warmup_offset': start_t,
                'total_timesteps': len(ts)
            })
            total += valid_timesteps
        
        self.total_samples = total
        self._cache = {}
        
        # Preload ALL events into cache (avoids per-batch loading delay)
        logger.info(f"Preloading {len(events)} events into memory...")
        for info in tqdm(self.event_info, desc="Preloading events"):
            self._load_event_fast(info['event'], info['total_timesteps'])
        logger.info(f"Cached {len(self._cache)} events")
    
    def __len__(self):
        return self.total_samples
    
    def _load_event_fast(self, event_name: str, n_timesteps: int) -> Dict[str, np.ndarray]:
        '''
        Load event data into pre-indexed numpy arrays for O(1) access.
        Shape: [n_timesteps, n_nodes] for each feature.
        '''
        if event_name not in self._cache:
            # This shouldn't happen if preloading is enabled
            import time
            load_start = time.time()
            event_path = self.model_path / event_name
            df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
            df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
            
            n_1d = self.static.num_1d
            n_2d = self.static.num_2d
            
            # Preallocate arrays [timestep, node_idx]
            # NOTE: Only using features available after warmup in test data!
            # inlet_flow and water_volume are NOT available after 10 timesteps
            wl_1d = np.zeros((n_timesteps, n_1d), dtype=np.float32)
            wl_2d = np.zeros((n_timesteps, n_2d), dtype=np.float32)
            rain_2d = np.zeros((n_timesteps, n_2d), dtype=np.float32)
            
            # Vectorized fill using pivot (much faster than iterrows)
            # 1D data - only water_level (other features derived from static data)
            pivot_1d_wl = df_1d.pivot(index='timestep', columns='node_idx', values='water_level')
            wl_1d[:len(pivot_1d_wl)] = pivot_1d_wl.values
            
            # 2D data - water_level + rainfall (only dynamic features available in test)
            pivot_2d_wl = df_2d.pivot(index='timestep', columns='node_idx', values='water_level')
            wl_2d[:len(pivot_2d_wl)] = pivot_2d_wl.values
            
            if 'rainfall' in df_2d.columns:
                pivot_2d_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall')
                rain_2d[:len(pivot_2d_rain)] = pivot_2d_rain.fillna(0).values
            
            # Handle NaN values (critical - NaN propagates and breaks training!)
            wl_1d = np.nan_to_num(wl_1d, nan=0.0, posinf=0.0, neginf=0.0)
            wl_2d = np.nan_to_num(wl_2d, nan=0.0, posinf=0.0, neginf=0.0)
            rain_2d = np.nan_to_num(rain_2d, nan=0.0, posinf=0.0, neginf=0.0)
            
            self._cache[event_name] = {
                'wl_1d': wl_1d,
                'wl_2d': wl_2d,
                'rain_2d': rain_2d,
                'cum_rain_2d': np.cumsum(rain_2d, axis=0),  # [n_timesteps, n_2d]
            }
        
        return self._cache[event_name]
    
    def __getitem__(self, idx: int) -> HeteroData:
        # Find event and timestep
        for info in self.event_info:
            if idx < info['start_idx'] + info['n_timesteps']:
                event_name = info['event']
                t = idx - info['start_idx'] + info['warmup_offset']
                n_timesteps = info['total_timesteps']
                break
        
        event_data = self._load_event_fast(event_name, n_timesteps)
        return self.fb.build_hetero_graph(event_data, t, model_id=self.model_id)


class SequenceFloodDataset(torch.utils.data.Dataset):
    '''
    Dataset for multi-step rollout training.
    Returns sequences of n consecutive timesteps for autoregressive training.
    '''
    
    def __init__(self, model_id: int, events: List[str], static_data: StaticGraphData, 
                 rollout_steps: int = 5, mode: str = 'train',
                 feature_builder: 'FeatureBuilder' = None,
                 feature_scales: Dict[str, float] = None):
        self.model_id = model_id
        self.events = events
        self.static = static_data
        self.mode = mode
        self.model_path = static_data.model_path
        self.rollout_steps = rollout_steps
        self._event_total_timesteps: Dict[str, int] = {}
        
        # FeatureBuilder: single source of truth
        if feature_builder is not None:
            self.fb = feature_builder
        else:
            self.fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                                     rain_lag_steps=Config.RAIN_LAG_STEPS,
                                     rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
        
        # Cache per-event total timesteps once; event indexing depends on rollout.
        for event in events:
            event_path = self.model_path / event
            ts = pd.read_csv(event_path / 'timesteps.csv')
            self._event_total_timesteps[event] = len(ts)
        
        self.event_info = []
        self.total_samples = 0
        self._rebuild_event_index()
        self._cache = {}
        
        # Preload ALL events into cache
        logger.info(f"Preloading {len(events)} events for sequence training...")
        for info in tqdm(self.event_info, desc="Preloading events"):
            self._load_event_fast(info['event'], info['total_timesteps'])
        logger.info(f"Cached {len(self._cache)} events")
    
    def _rebuild_event_index(self):
        '''Rebuild index mapping and sample count for current rollout_steps.'''
        self.event_info = []
        total = 0
        for event in self.events:
            total_ts = self._event_total_timesteps[event]
            n_timesteps = total_ts - 1  # -1 because we predict next step
            start_t = Config.WARMUP_TIMESTEPS if self.mode == 'train' else 0
            valid_timesteps = max(0, n_timesteps - start_t - self.rollout_steps + 1)
            self.event_info.append({
                'event': event,
                'start_idx': total,
                'n_timesteps': valid_timesteps,
                'warmup_offset': start_t,
                'total_timesteps': total_ts
            })
            total += valid_timesteps
        self.total_samples = total
    
    def set_rollout_steps(self, rollout_steps: int):
        '''Update rollout length and recompute sampling index without reloading cache.'''
        rollout_steps = int(rollout_steps)
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}")
        if rollout_steps == self.rollout_steps:
            return
        self.rollout_steps = rollout_steps
        self._rebuild_event_index()
    
    def __len__(self):
        return self.total_samples
    
    def _load_event_fast(self, event_name: str, n_timesteps: int) -> Dict[str, np.ndarray]:
        '''Load event data into pre-indexed numpy arrays.'''
        if event_name not in self._cache:
            event_path = self.model_path / event_name
            df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
            df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
            
            n_1d = self.static.num_1d
            n_2d = self.static.num_2d
            
            wl_1d = np.zeros((n_timesteps, n_1d), dtype=np.float32)
            wl_2d = np.zeros((n_timesteps, n_2d), dtype=np.float32)
            rain_2d = np.zeros((n_timesteps, n_2d), dtype=np.float32)
            
            pivot_1d_wl = df_1d.pivot(index='timestep', columns='node_idx', values='water_level')
            wl_1d[:len(pivot_1d_wl)] = pivot_1d_wl.values
            
            pivot_2d_wl = df_2d.pivot(index='timestep', columns='node_idx', values='water_level')
            wl_2d[:len(pivot_2d_wl)] = pivot_2d_wl.values
            
            if 'rainfall' in df_2d.columns:
                pivot_2d_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall')
                rain_2d[:len(pivot_2d_rain)] = pivot_2d_rain.fillna(0).values
            
            wl_1d = np.nan_to_num(wl_1d, nan=0.0, posinf=0.0, neginf=0.0)
            wl_2d = np.nan_to_num(wl_2d, nan=0.0, posinf=0.0, neginf=0.0)
            rain_2d = np.nan_to_num(rain_2d, nan=0.0, posinf=0.0, neginf=0.0)
            
            self._cache[event_name] = {
                'wl_1d': wl_1d,
                'wl_2d': wl_2d,
                'rain_2d': rain_2d,
                'cum_rain_2d': np.cumsum(rain_2d, axis=0),  # [n_timesteps, n_2d]
            }
        
        return self._cache[event_name]
    
    def __getitem__(self, idx: int) -> Dict:
        '''Returns a sequence of data for multi-step rollout training.'''
        # Find event and starting timestep
        for info in self.event_info:
            if idx < info['start_idx'] + info['n_timesteps']:
                event_name = info['event']
                t_start = idx - info['start_idx'] + info['warmup_offset']
                n_timesteps = info['total_timesteps']
                break
        
        event_data = self._load_event_fast(event_name, n_timesteps)
        return self._build_sequence(event_data, t_start)
    
    def _build_sequence(self, event_data: Dict[str, np.ndarray], t_start: int) -> Dict:
        '''Build a sequence of rollout_steps consecutive timesteps.
        
        Only returns essential data - dynamic features are recomputed on GPU during training.
        Static elevations are cached on GPU once in train_epoch_multistep.
        
        Pre-computes all rain lags for all rollout steps to avoid per-step computation.
        '''
        # Extract sequence data: [rollout_steps, n_nodes] for each
        wl_1d_seq = event_data['wl_1d'][t_start:t_start + self.rollout_steps + 1]  # +1 for final target
        wl_2d_seq = event_data['wl_2d'][t_start:t_start + self.rollout_steps + 1]
        rain_seq = event_data['rain_2d'][t_start:t_start + self.rollout_steps]
        
        # Initial water levels (at t_start)
        wl_1d_init = wl_1d_seq[0]
        wl_2d_init = wl_2d_seq[0]
        n_2d = len(wl_2d_init)
        rain_lag_steps = self.fb.rain_lag_steps
        rain_future_steps = getattr(self.fb, 'rain_future_steps', 0)
        
        # PRE-COMPUTE all rain lags for all rollout steps [rollout_steps, rain_lag_steps, n_2d]
        # Vectorized: single fancy-index into rain_2d instead of nested Python loops
        t_indices = np.arange(t_start, t_start + self.rollout_steps)        # [rollout_steps]
        lag_offsets = np.arange(1, rain_lag_steps + 1)                      # [rain_lag_steps]
        t_lag_matrix = t_indices[:, None] - lag_offsets[None, :]            # [rollout_steps, rain_lag_steps]
        valid_lag = t_lag_matrix >= 0
        t_lag_clipped = np.clip(t_lag_matrix, 0, None)
        rain_lag_all = event_data['rain_2d'][t_lag_clipped]                 # [rollout_steps, rain_lag_steps, n_2d]
        rain_lag_all[~valid_lag] = 0.0
        rain_lag_all = rain_lag_all.astype(np.float32)
        
        # PRE-COMPUTE all future rain for all rollout steps [rollout_steps, rain_future_steps, n_2d]
        if rain_future_steps > 0:
            lead_offsets = np.arange(1, rain_future_steps + 1)              # [rain_future_steps]
            t_lead_matrix = t_indices[:, None] + lead_offsets[None, :]      # [rollout_steps, rain_future_steps]
            n_total = event_data['wl_1d'].shape[0]
            valid_lead = t_lead_matrix < n_total
            t_lead_clipped = np.clip(t_lead_matrix, 0, n_total - 1)
            future_rain_all = event_data['rain_2d'][t_lead_clipped]         # [rollout_steps, rain_future_steps, n_2d]
            future_rain_all[~valid_lead] = 0.0
            future_rain_all = future_rain_all.astype(np.float32)
        else:
            future_rain_all = np.zeros((self.rollout_steps, 0, n_2d), dtype=np.float32)
        
        # Cumulative rainfall for each step [rollout_steps, n_2d] (O(1) via precomputed cumsum)
        cum_rain_seq = event_data['cum_rain_2d'][t_start:t_start + self.rollout_steps].astype(np.float32)
        
        # Pre-compute global rain summaries once per sequence (shared by all 1D nodes).
        # Shapes: [rollout_steps], [rollout_steps, rain_lag_steps], [rollout_steps, rain_future_steps]
        global_rain_seq = rain_seq.mean(axis=1).astype(np.float32)
        global_cum_rain_seq = cum_rain_seq.mean(axis=1).astype(np.float32)
        global_rain_lag_all = rain_lag_all.mean(axis=2).astype(np.float32)
        if rain_future_steps > 0:
            global_future_rain_all = future_rain_all.mean(axis=2).astype(np.float32)
        else:
            global_future_rain_all = np.zeros((self.rollout_steps, 0), dtype=np.float32)
        
        # Normalized timestep for each step [rollout_steps]
        n_total = event_data['wl_1d'].shape[0]
        t_indices = np.arange(t_start, t_start + self.rollout_steps, dtype=np.float32)
        normalized_t_seq = t_indices / max(n_total - 1, 1)
        
        # Ground truth water levels for each step (for loss computation)
        wl_1d_targets = wl_1d_seq[1:].astype(np.float32)  # [rollout_steps, n_1d]
        wl_2d_targets = wl_2d_seq[1:].astype(np.float32)  # [rollout_steps, n_2d]
        
        # Previous water levels for wl_prev features: [wl_prev_steps, n_nodes]
        wl_prev_steps = self.fb.wl_prev_steps
        wl_prevs_1d_init = np.zeros((max(wl_prev_steps, 1), len(wl_1d_init)), dtype=np.float32)
        wl_prevs_2d_init = np.zeros((max(wl_prev_steps, 1), len(wl_2d_init)), dtype=np.float32)
        for lag in range(wl_prev_steps):
            t_lag = max(t_start - lag - 1, 0)
            wl_prevs_1d_init[lag] = event_data['wl_1d'][t_lag].astype(np.float32)
            wl_prevs_2d_init[lag] = event_data['wl_2d'][t_lag].astype(np.float32)
        
        return {
            'wl_1d_init': torch.from_numpy(wl_1d_init.astype(np.float32)),
            'wl_2d_init': torch.from_numpy(wl_2d_init.astype(np.float32)),
            'wl_prevs_1d_init': torch.from_numpy(wl_prevs_1d_init),
            'wl_prevs_2d_init': torch.from_numpy(wl_prevs_2d_init),
            'rainfall_seq': torch.from_numpy(rain_seq.astype(np.float32)),
            'rain_lag_all': torch.from_numpy(rain_lag_all),          # [rollout_steps, rain_lag_steps, n_2d]
            'future_rain_all': torch.from_numpy(future_rain_all),    # [rollout_steps, rain_future_steps, n_2d]
            'cum_rain_seq': torch.from_numpy(cum_rain_seq),  # [rollout_steps, n_2d]
            'global_rain_seq': torch.from_numpy(global_rain_seq),  # [rollout_steps]
            'global_rain_lag_all': torch.from_numpy(global_rain_lag_all),  # [rollout_steps, rain_lag_steps]
            'global_future_rain_all': torch.from_numpy(global_future_rain_all),  # [rollout_steps, rain_future_steps]
            'global_cum_rain_seq': torch.from_numpy(global_cum_rain_seq),  # [rollout_steps]
            'normalized_t_seq': torch.from_numpy(normalized_t_seq),  # [rollout_steps]
            'wl_1d_targets': torch.from_numpy(wl_1d_targets),
            'wl_2d_targets': torch.from_numpy(wl_2d_targets),
            'n_total_timesteps': torch.tensor(n_total, dtype=torch.float32),  # scalar
        }


def collate_sequence(batch: List[Dict]) -> Dict:
    '''Custom collate for sequence data - stack along batch dimension.'''
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class FusedLayerNormGELU(nn.Module):
    '''Fused LayerNorm + GELU for reduced memory bandwidth and kernel launch overhead.
    
    GELU preserves negative activations (unlike ReLU), which is critical for
    physics-informed models where negative values carry physical meaning
    (e.g. negative slope = downhill, negative head = unsurcharged pipe).
    '''
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, use_fused: bool = True):
        super().__init__()
        self.use_fused = use_fused
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.act(x)


class FusedLinearLayerNormGELU(nn.Module):
    '''Fused Linear + LayerNorm + GELU block for encoder/decoder efficiency.
    
    GELU preserves negative activations, important for physics-informed features
    where negative values carry physical meaning.
    '''
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, 
                 use_fused: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class HeteroFloodGNN(nn.Module):
    '''
    Heterogeneous Graph Neural Network for flood prediction.
    
    Predicts Δwl (water level change) for both 1D and 2D nodes.
    Uses separate message passing for pipe, surface, and coupling edges.
    
    Architecture improvements for gradient flow:
    - Pre-norm residual blocks (clean gradient highway through residual path)
    - Final normalization after message passing for stability
    
    Supports multiple convolution types:
    - 'gatv2': GATv2Conv with attention (default, best for heterogeneous graphs)
    - 'sage': GraphSAGE (faster, good baseline)
    - 'transformer': TransformerConv (most expressive, slower)
    '''
    
    def __init__(self, 
                 in_channels_1d: int = 11,  # Use fb.in_channels_1d (6 static + 5 dynamic)
                 in_channels_2d: int = 23,  # Use fb.in_channels_2d (9 static + 4 base + 10 rain lags)
                 hidden_channels: int = 128,
                 num_layers: int = 8,
                 dropout: float = 0.1,
                 conv_type: str = 'gatv2',
                 heads: int = 4,
                 use_fused_ops: bool = True,
                 edge_dim_1d: int = 0,
                 edge_dim_2d: int = 0,
                 temporal_bundle_k: int = 1,
                 use_node_embeddings: bool = True,
                 num_1d_nodes: Optional[int] = None,
                 num_2d_nodes: Optional[int] = None,
                 node_embed_dim: Optional[int] = None):
        super().__init__()
        
        # Guard: attention-based convs require hidden_channels divisible by heads
        if conv_type in ('gatv2', 'transformer') and hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads}) "
                f"for conv_type='{conv_type}'. Got remainder {hidden_channels % heads}."
            )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.use_fused_ops = use_fused_ops
        self.edge_dim_1d = edge_dim_1d
        self.edge_dim_2d = edge_dim_2d
        # Number of Δwl steps decoded per forward pass (temporal bundling factor)
        self.temporal_bundle_k = max(int(temporal_bundle_k), 1)
        self.num_1d_nodes = int(num_1d_nodes) if num_1d_nodes is not None else None
        self.num_2d_nodes = int(num_2d_nodes) if num_2d_nodes is not None else None
        requested_node_embeddings = bool(use_node_embeddings)
        
        # Node encoders using fused operations
        self.encoder_1d = nn.Sequential(
            FusedLinearLayerNormGELU(in_channels_1d, hidden_channels, dropout, use_fused_ops),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        self.encoder_2d = nn.Sequential(
            FusedLinearLayerNormGELU(in_channels_2d, hidden_channels, dropout, use_fused_ops),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Optional node-ID embeddings for node-specific correction capacity.
        self.use_node_embeddings = (
            requested_node_embeddings and
            self.num_1d_nodes is not None and self.num_1d_nodes > 0 and
            self.num_2d_nodes is not None and self.num_2d_nodes > 0
        )
        if self.use_node_embeddings:
            self.node_embed_dim = max(
                1, int(node_embed_dim if node_embed_dim is not None else hidden_channels // 4)
            )
            self.node_embed_1d = nn.Embedding(self.num_1d_nodes, self.node_embed_dim)
            self.node_embed_2d = nn.Embedding(self.num_2d_nodes, self.node_embed_dim)
            self.node_fuse_1d = nn.Linear(hidden_channels + self.node_embed_dim, hidden_channels)
            self.node_fuse_2d = nn.Linear(hidden_channels + self.node_embed_dim, hidden_channels)
        else:
            self.node_embed_dim = 0
        
        # Message passing layers with pre-norm
        self.convs = nn.ModuleList()
        self.norms_1d = nn.ModuleList()
        self.norms_2d = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv(
                self._make_conv_dict(hidden_channels, dropout, conv_type, heads,
                                     edge_dim_1d, edge_dim_2d),
                aggr='mean'
            )
            self.convs.append(conv)
            # Pre-norm: normalize BEFORE the conv (applied in forward)
            self.norms_1d.append(nn.LayerNorm(hidden_channels))
            self.norms_2d.append(nn.LayerNorm(hidden_channels))
        
        # Final norms after all message passing (for pre-norm stability)
        self.final_norm_1d = nn.LayerNorm(hidden_channels)
        self.final_norm_2d = nn.LayerNorm(hidden_channels)
        
        # Global attention pool for 1D: gives every 1D node a learned summary
        # of the entire pipe network's post-MP state. Helps nodes deep in the
        # tree (beyond MP reach) sense system-wide pressure and flow regime.
        self.global_pool_1d = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
        )
        
        # Decoders
        self.decoder_1d = nn.Sequential(
            FusedLinearLayerNormGELU(hidden_channels, hidden_channels, dropout, use_fused_ops),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            # Final layer outputs K deltas per node for temporal bundling
            nn.Linear(hidden_channels // 2, self.temporal_bundle_k),
        )
        
        self.decoder_2d = nn.Sequential(
            FusedLinearLayerNormGELU(hidden_channels, hidden_channels, dropout, use_fused_ops),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, self.temporal_bundle_k),
        )
    
    def _make_conv_dict(self, hidden_channels: int, dropout: float, conv_type: str, heads: int,
                        edge_dim_1d: int = 0, edge_dim_2d: int = 0) -> dict:
        '''Create convolution dictionary based on conv_type.
        
        edge_dim_1d/edge_dim_2d: number of edge features for pipe/surface edges.
        Passed to conv layers that support edge_dim (GATv2, Transformer).
        '''
        ed_1d = edge_dim_1d if edge_dim_1d > 0 else None
        ed_2d = edge_dim_2d if edge_dim_2d > 0 else None
        
        if conv_type == 'gatv2':
            return {
                ('1d', 'pipe', '1d'): GATv2Conv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ('2d', 'surface', '2d'): GATv2Conv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_2d,
                ),
                ('1d', 'couples', '2d'): GATv2Conv(
                    (hidden_channels, hidden_channels), hidden_channels // heads, 
                    heads=heads, concat=True, add_self_loops=False, dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ('2d', 'couples', '1d'): GATv2Conv(
                    (hidden_channels, hidden_channels), hidden_channels // heads, 
                    heads=heads, concat=True, add_self_loops=False, dropout=dropout,
                    edge_dim=ed_1d,
                ),
            }
        
        elif conv_type == 'sage':
            # SAGEConv doesn't support edge features — silently ignored
            return {
                ('1d', 'pipe', '1d'): SAGEConv(hidden_channels, hidden_channels),
                ('2d', 'surface', '2d'): SAGEConv(hidden_channels, hidden_channels),
                ('1d', 'couples', '2d'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                ('2d', 'couples', '1d'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            }
        
        elif conv_type == 'transformer':
            return {
                ('1d', 'pipe', '1d'): TransformerConv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ('2d', 'surface', '2d'): TransformerConv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_2d,
                ),
                ('1d', 'couples', '2d'): TransformerConv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_1d,
                ),
                ('2d', 'couples', '1d'): TransformerConv(
                    hidden_channels, hidden_channels // heads, 
                    heads=heads, concat=True, dropout=dropout,
                    edge_dim=ed_1d,
                ),
            }
        
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}. Choose from 'gatv2', 'sage', 'transformer'")
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Encode
        h = {
            '1d': self.encoder_1d(x_dict['1d']),
            '2d': self.encoder_2d(x_dict['2d']),
        }

        # Inject node-ID embeddings before message passing.
        if self.use_node_embeddings:
            node_ids_1d = torch.arange(h['1d'].shape[0], device=h['1d'].device) % self.num_1d_nodes
            node_ids_2d = torch.arange(h['2d'].shape[0], device=h['2d'].device) % self.num_2d_nodes
            h['1d'] = self.node_fuse_1d(torch.cat([h['1d'], self.node_embed_1d(node_ids_1d)], dim=-1))
            h['2d'] = self.node_fuse_2d(torch.cat([h['2d'], self.node_embed_2d(node_ids_2d)], dim=-1))
        
        # Message passing with PRE-NORM residual connections
        # Pre-norm: normalize BEFORE conv, then add to clean residual path
        for i, conv in enumerate(self.convs):
            # Normalize before applying conv (pre-norm pattern)
            h_normed = {
                '1d': self.norms_1d[i](h['1d']),
                '2d': self.norms_2d[i](h['2d']),
            }
            
            # Apply conv on normalized features (with optional edge attributes)
            if edge_attr_dict and (self.edge_dim_1d > 0 or self.edge_dim_2d > 0):
                h_new = conv(h_normed, edge_index_dict, edge_attr_dict)
            else:
                h_new = conv(h_normed, edge_index_dict)
            
            # Clean residual addition (no norm on this path = better gradient flow)
            h['1d'] = h['1d'] + F.dropout(h_new['1d'], p=self.dropout, training=self.training)
            h['2d'] = h['2d'] + F.dropout(h_new['2d'], p=self.dropout, training=self.training)
        
        # Final normalization after all message passing layers
        h['1d'] = self.final_norm_1d(h['1d'])
        h['2d'] = self.final_norm_2d(h['2d'])
        
        # Global attention pool for 1D: pool → broadcast → fuse
        n_total_1d = h['1d'].shape[0]
        if self.num_1d_nodes is not None and n_total_1d > self.num_1d_nodes:
            batch_size = n_total_1d // self.num_1d_nodes
            h_1d_view = h['1d'].view(batch_size, self.num_1d_nodes, -1)
            global_1d = h_1d_view.mean(dim=1, keepdim=True).expand_as(h_1d_view)
            h['1d'] = self.global_pool_1d(
                torch.cat([h_1d_view, global_1d], dim=-1)
            ).reshape(n_total_1d, -1)
        else:
            global_1d = h['1d'].mean(dim=0, keepdim=True).expand_as(h['1d'])
            h['1d'] = self.global_pool_1d(
                torch.cat([h['1d'], global_1d], dim=-1)
            )
        
        # Decode to Δwl - no bounding, let the model learn natural ranges.
        # With temporal bundling, each node outputs K consecutive deltas.
        delta_1d = self.decoder_1d(h['1d'])
        delta_2d = self.decoder_2d(h['2d'])
        if self.temporal_bundle_k == 1:
            # Backward-compatible: return shape [N] for single-step models
            delta_1d = delta_1d.squeeze(-1)
            delta_2d = delta_2d.squeeze(-1)
        
        return {'1d': delta_1d, '2d': delta_2d}


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compute_std_values(model_path: Path, events: List[str], static_data: 'StaticGraphData' = None) -> Dict[str, float]:
    '''
    Compute standard deviation for standardized RMSE AND dynamic feature normalization.
    
    Returns scales for all dynamic features so they have std ≈ 1.0 after normalization.
    This is critical for balancing dynamic vs static features.
    '''
    wl_1d_all, wl_2d_all = [], []
    pipe_fill_all, head_above_all = [], []
    pond_depth_all, rainfall_all = [], []
    cum_rainfall_all = []  # Cumulative rainfall at each timestep
    delta_1d_all, delta_2d_all = [], []  # For lagged delta feature scaling
    
    # Load static elevations if not provided
    if static_data is None:
        nodes_1d = pd.read_csv(model_path / '1d_nodes_static.csv')
        nodes_2d = pd.read_csv(model_path / '2d_nodes_static.csv')
        invert_elev = nodes_1d['invert_elevation'].fillna(0).values
        surface_elev = nodes_1d['surface_elevation'].fillna(0).values
        min_elev_2d = nodes_2d['min_elevation'].fillna(0).values
    else:
        # Use cached values from static_data
        invert_elev = static_data.nodes_1d['invert_elevation'].fillna(0).values
        surface_elev = static_data.nodes_1d['surface_elevation'].fillna(0).values
        min_elev_2d = static_data.nodes_2d['min_elevation'].fillna(0).values
    
    # Sample events for speed (10 is enough for good std estimates)
    sample_events = events[:min(10, len(events))]
    
    for event in tqdm(sample_events, desc="Computing feature scales"):
        event_path = model_path / event
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        
        # Get number of nodes and timesteps
        n_1d = len(invert_elev)
        n_2d = len(min_elev_2d)
        
        # Reshape water levels to [timesteps, nodes]
        wl_1d = df_1d['water_level'].values.reshape(-1, n_1d)
        wl_2d = df_2d['water_level'].values.reshape(-1, n_2d)
        rainfall = df_2d['rainfall'].values.reshape(-1, n_2d)
        
        # Compute derived features
        pipe_fill = wl_1d - invert_elev  # [timesteps, n_1d]
        head_above = wl_1d - surface_elev  # [timesteps, n_1d]
        pond_depth = wl_2d - min_elev_2d  # [timesteps, n_2d]
        cum_rainfall = np.cumsum(rainfall, axis=0)  # [timesteps, n_2d]
        
        # Compute deltas (water level changes between timesteps)
        delta_1d = np.diff(wl_1d, axis=0)  # [timesteps-1, n_1d]
        delta_2d = np.diff(wl_2d, axis=0)  # [timesteps-1, n_2d]
        
        # Accumulate
        wl_1d_all.append(wl_1d.flatten())
        wl_2d_all.append(wl_2d.flatten())
        pipe_fill_all.append(pipe_fill.flatten())
        head_above_all.append(head_above.flatten())
        pond_depth_all.append(pond_depth.flatten())
        rainfall_all.append(rainfall.flatten())
        cum_rainfall_all.append(cum_rainfall.flatten())
        delta_1d_all.append(delta_1d.flatten())
        delta_2d_all.append(delta_2d.flatten())
    
    # Compute std for each feature (clamp to avoid division by zero)
    def safe_std(arr_list, min_val=0.1):
        return max(float(np.std(np.concatenate(arr_list))), min_val)
    
    # Compute raw rainfall std (don't clamp - rainfall is often sparse with low values)
    raw_rainfall_std = float(np.std(np.concatenate(rainfall_all)))
    logger.info(f"  Raw rainfall std: {raw_rainfall_std:.6f}")
    
    # Compute Huber deltas from p95 of delta distributions (normalized by WL std)
    std_1d = safe_std(wl_1d_all)
    std_2d = safe_std(wl_2d_all)
    delta_1d_concatenated = np.concatenate(delta_1d_all)
    delta_2d_concatenated = np.concatenate(delta_2d_all)
    
    # p95 of absolute deltas, normalized by water level std (for Huber threshold)
    huber_delta_1d = float(np.percentile(np.abs(delta_1d_concatenated), 95)) / std_1d
    huber_delta_2d = float(np.percentile(np.abs(delta_2d_concatenated), 95)) / std_2d
    
    scales = {
        # Loss normalization (original purpose)
        '1d': std_1d,
        '2d': std_2d,
        # Dynamic feature scales (NEW - for proper normalization)
        'wl_1d_scale': std_1d,
        'wl_2d_scale': std_2d,
        'pipe_fill_scale': safe_std(pipe_fill_all),
        'head_above_scale': safe_std(head_above_all),
        'pond_depth_scale': safe_std(pond_depth_all),
        # Rainfall: use actual std with very low floor (rainfall is sparse but important!)
        'rainfall_scale': max(raw_rainfall_std, 0.001),
        # Cumulative rainfall: grows over event duration
        'cum_rainfall_scale': safe_std(cum_rainfall_all, min_val=0.1),
        # Delta scales (for Huber loss threshold computation, not used as feature scales)
        'delta_1d_scale': safe_std(delta_1d_all, min_val=0.01),
        'delta_2d_scale': safe_std(delta_2d_all, min_val=0.01),
        # Huber loss thresholds (p95 of |delta| normalized by WL std)
        'huber_delta_1d': huber_delta_1d,
        'huber_delta_2d': huber_delta_2d,
    }
    
    logger.info(f"Computed feature scales:")
    logger.info(f"  WL 1D: {scales['wl_1d_scale']:.4f}, WL 2D: {scales['wl_2d_scale']:.4f}")
    logger.info(f"  Pipe fill: {scales['pipe_fill_scale']:.4f}, Head above: {scales['head_above_scale']:.4f}")
    logger.info(f"  Pond depth: {scales['pond_depth_scale']:.4f}, Rainfall: {scales['rainfall_scale']:.4f}")
    logger.info(f"  Cum rainfall: {scales['cum_rainfall_scale']:.4f}")
    logger.info(f"  Delta 1D: {scales['delta_1d_scale']:.4f}, Delta 2D: {scales['delta_2d_scale']:.4f}")
    logger.info(f"  Huber delta 1D: {scales['huber_delta_1d']:.4f}, Huber delta 2D: {scales['huber_delta_2d']:.4f}")
    
    return scales


def standardized_rmse(pred: torch.Tensor, target: torch.Tensor, std: float) -> torch.Tensor:
    '''Compute RMSE standardized by water level std.'''
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse / std


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    '''Huber loss: L2 for |error| <= delta, L1 for |error| > delta.
    
    Args:
        error: Prediction errors (pred - target)
        delta: Transition point between L2 and L1 loss
    
    Returns:
        Mean Huber loss
    '''
    abs_error = torch.abs(error)
    quadratic = 0.5 * error**2
    linear = delta * abs_error - 0.5 * delta**2
    return torch.where(abs_error <= delta, quadratic, linear).mean()


class FloodLoss(nn.Module):
    '''Combined loss for 1D and 2D with standardization and optional weighting.
    
    Supports both MSE (standardized RMSE) and Huber loss (SmoothL1) for better
    handling of heavy-tailed distributions (e.g., delta water levels).
    '''
    
    def __init__(self, std_1d: float, std_2d: float, weight_1d: float = 1.0,
                 use_huber: bool = False, huber_delta_1d: float = None, huber_delta_2d: float = None,
                 use_bias_loss: bool = False, bias_loss_weight: float = 0.1):
        super().__init__()
        self.std_1d = std_1d
        self.std_2d = std_2d
        self.weight_1d = weight_1d  # Upweight harder 1D predictions
        self.use_huber = use_huber
        self.use_bias_loss = use_bias_loss
        self.bias_loss_weight = bias_loss_weight
        
        if use_huber:
            # Huber delta is in standardized units (errors normalized by std)
            # Default values based on p95 of delta distributions
            self.huber_delta_1d = huber_delta_1d if huber_delta_1d is not None else 0.075  # Default: ~p95/std
            self.huber_delta_2d = huber_delta_2d if huber_delta_2d is not None else 0.019  # Default: ~p95/std
        else:
            self.huber_delta_1d = None
            self.huber_delta_2d = None
    
    def forward(self, pred_dict, target_dict):
        if self.use_huber:
            # Standardize errors for Huber loss
            error_1d = (pred_dict['1d'] - target_dict['1d']) / self.std_1d
            error_2d = (pred_dict['2d'] - target_dict['2d']) / self.std_2d
            
            # Apply Huber loss with custom delta thresholds
            loss_1d = huber_loss(error_1d, self.huber_delta_1d)
            loss_2d = huber_loss(error_2d, self.huber_delta_2d)
            
            # For reporting, compute approximate RMSE from Huber loss
            # In L2 region: loss ≈ 0.5 * error^2, so RMSE ≈ sqrt(2 * loss)
            # In L1 region: loss ≈ delta * |error| - 0.5*delta^2, approximate as RMSE ≈ loss/delta
            # Use a conservative approximation: assume mostly in L2 region
            rmse_1d = torch.sqrt(2.0 * loss_1d) if loss_1d > 0 else torch.tensor(0.0, device=loss_1d.device)
            rmse_2d = torch.sqrt(2.0 * loss_2d) if loss_2d > 0 else torch.tensor(0.0, device=loss_2d.device)
            
            # Weighted average
            total = (self.weight_1d * loss_1d + loss_2d) / (self.weight_1d + 1)
        else:
            # Standard RMSE for each node type
            rmse_1d = standardized_rmse(pred_dict['1d'], target_dict['1d'], self.std_1d)
            rmse_2d = standardized_rmse(pred_dict['2d'], target_dict['2d'], self.std_2d)
            
            # Weighted average - give more gradient to 1D (the bottleneck)
            total = (self.weight_1d * rmse_1d + rmse_2d) / (self.weight_1d + 1)
        
        if self.use_bias_loss:
            # Penalize systematic bias: |mean(pred - target)| / std
            # This directly fights the AR drift caused by consistent over/under-prediction
            bias_1d = torch.abs((pred_dict['1d'] - target_dict['1d']).mean()) / self.std_1d
            bias_2d = torch.abs((pred_dict['2d'] - target_dict['2d']).mean()) / self.std_2d
            bias_penalty = (self.weight_1d * bias_1d + bias_2d) / (self.weight_1d + 1)
            total = total + self.bias_loss_weight * bias_penalty
        
        return total, rmse_1d, rmse_2d


def unwrap_model(model: nn.Module) -> nn.Module:
    '''Return the underlying model for DDP-wrapped modules.'''
    return model.module if isinstance(model, DDP) else model


def collapse_temporal_bundle_for_single_step(pred_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''For temporally-bundled models, collapse K-step outputs to 1-step for single-step losses.
    
    Uses only the first decoded delta along the temporal dimension when output
    tensors have shape [N, K]. Keeps behaviour identical for K=1.
    '''
    collapsed = {}
    for key in ['1d', '2d']:
        tensor = pred_dict[key]
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
            # [N, K] → take first-step delta [N]
            collapsed[key] = tensor[:, 0]
        else:
            collapsed[key] = tensor
    return collapsed


def collate_hetero(batch: List[HeteroData]) -> HeteroData:
    '''Custom collate function for HeteroData batches.'''
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def extract_edge_attr_dict(batch) -> Optional[Dict]:
    '''Extract edge_attr_dict from a PyG HeteroData batch.
    
    Returns dict keyed by edge type, or None if no edge attributes exist.
    '''
    ea = {}
    # Keep optional static attrs only for relations that still use them.
    for edge_type in [('2d', 'surface', '2d')]:
        store = batch[edge_type]
        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
            ea[edge_type] = store.edge_attr

    # Dynamic coupling edge attrs for single-step/eval paths:
    # [head, dhead, dwl_1d, dwl_2d] with same scaling used in train/AR paths.
    has_wl = (
        hasattr(batch['1d'], 'wl') and batch['1d'].wl is not None and
        hasattr(batch['2d'], 'wl') and batch['2d'].wl is not None
    )
    if has_wl:
        wl_1d = batch['1d'].wl
        wl_2d = batch['2d'].wl
        wl_1d_prev = batch['1d'].wl_prev if hasattr(batch['1d'], 'wl_prev') and batch['1d'].wl_prev is not None else wl_1d
        wl_2d_prev = batch['2d'].wl_prev if hasattr(batch['2d'], 'wl_prev') and batch['2d'].wl_prev is not None else wl_2d

        wl_1d_scale = getattr(batch, 'wl_1d_scale', None)
        wl_2d_scale = getattr(batch, 'wl_2d_scale', None)
        if wl_1d_scale is None:
            wl_1d_scale = 1.0
        elif isinstance(wl_1d_scale, torch.Tensor):
            wl_1d_scale = float(wl_1d_scale.flatten()[0].item())
        if wl_2d_scale is None:
            wl_2d_scale = 1.0
        elif isinstance(wl_2d_scale, torch.Tensor):
            wl_2d_scale = float(wl_2d_scale.flatten()[0].item())

        # Dynamic 1D pipe edge attrs for single-step/eval:
        # [hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v]
        rel_11 = ('1d', 'pipe', '1d')
        if rel_11 in batch.edge_types:
            ei_11 = batch[rel_11].edge_index
            if ei_11 is not None and ei_11.numel() > 0:
                if hasattr(batch[rel_11], 'edge_len') and batch[rel_11].edge_len is not None:
                    edge_len = batch[rel_11].edge_len.to(wl_1d.device).clamp_min(1e-3)
                else:
                    edge_len = torch.ones(ei_11.shape[1], dtype=wl_1d.dtype, device=wl_1d.device)
                wl_u = wl_1d[ei_11[0]]
                wl_v = wl_1d[ei_11[1]]
                wl_u_prev = wl_1d_prev[ei_11[0]]
                wl_v_prev = wl_1d_prev[ei_11[1]]
                hydraulic_grad = (wl_u - wl_v) / edge_len
                hydraulic_grad_prev = (wl_u_prev - wl_v_prev) / edge_len
                dhydraulic_grad = hydraulic_grad - hydraulic_grad_prev
                dwl_u = (wl_u - wl_u_prev) / wl_1d_scale
                dwl_v = (wl_v - wl_v_prev) / wl_1d_scale
                ea[rel_11] = torch.stack([hydraulic_grad, dhydraulic_grad, dwl_u, dwl_v], dim=1)

        rel_12 = ('1d', 'couples', '2d')
        rel_21 = ('2d', 'couples', '1d')

        if rel_12 in batch.edge_types:
            ei_12 = batch[rel_12].edge_index
            if ei_12 is not None and ei_12.numel() > 0:
                wl1_12 = wl_1d[ei_12[0]]
                wl2_12 = wl_2d[ei_12[1]]
                wl1p_12 = wl_1d_prev[ei_12[0]]
                wl2p_12 = wl_2d_prev[ei_12[1]]
                head_12 = (wl1_12 - wl2_12) / wl_1d_scale
                dhead_12 = ((wl1_12 - wl2_12) - (wl1p_12 - wl2p_12)) / wl_1d_scale
                dwl1_12 = (wl1_12 - wl1p_12) / wl_1d_scale
                dwl2_12 = (wl2_12 - wl2p_12) / wl_2d_scale
                ea[rel_12] = torch.stack([head_12, dhead_12, dwl1_12, dwl2_12], dim=1)

        if rel_21 in batch.edge_types:
            ei_21 = batch[rel_21].edge_index
            if ei_21 is not None and ei_21.numel() > 0:
                # Same physical definition head = wl_1d - wl_2d
                wl1_21 = wl_1d[ei_21[1]]
                wl2_21 = wl_2d[ei_21[0]]
                wl1p_21 = wl_1d_prev[ei_21[1]]
                wl2p_21 = wl_2d_prev[ei_21[0]]
                head_21 = (wl1_21 - wl2_21) / wl_1d_scale
                dhead_21 = ((wl1_21 - wl2_21) - (wl1p_21 - wl2p_21)) / wl_1d_scale
                dwl1_21 = (wl1_21 - wl1p_21) / wl_1d_scale
                dwl2_21 = (wl2_21 - wl2p_21) / wl_2d_scale
                ea[rel_21] = torch.stack([head_21, dhead_21, dwl1_21, dwl2_21], dim=1)

    return ea if ea else None


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

# Feature names — sourced from FeatureBuilder class constants
# (kept here as module-level aliases for backward compatibility with feature importance code)
FEATURE_NAMES_1D_STATIC = FeatureBuilder.NAMES_1D_STATIC
FEATURE_NAMES_1D_DYNAMIC = FeatureBuilder.NAMES_1D_DYNAMIC
FEATURE_NAMES_1D = FEATURE_NAMES_1D_STATIC + FEATURE_NAMES_1D_DYNAMIC

FEATURE_NAMES_2D_STATIC = FeatureBuilder.NAMES_2D_STATIC
FEATURE_NAMES_2D_DYNAMIC_BASE = FeatureBuilder.NAMES_2D_DYNAMIC_BASE

def get_2d_dynamic_feature_names(rain_lag_steps: int = 10) -> List[str]:
    '''Get 2D dynamic feature names including configurable rain lags.'''
    lag_names = [f'rain_lag{i+1}' for i in range(rain_lag_steps)]
    return list(FEATURE_NAMES_2D_DYNAMIC_BASE) + lag_names

# Default feature names (for backward compatibility)
FEATURE_NAMES_2D_DYNAMIC = get_2d_dynamic_feature_names(Config.RAIN_LAG_STEPS)
FEATURE_NAMES_2D = list(FEATURE_NAMES_2D_STATIC) + FEATURE_NAMES_2D_DYNAMIC


@torch.no_grad()
def compute_permutation_importance(
    model,
    val_loader,
    criterion,
    device,
    n_repeats: int = 1,
    max_batches: int = 5,
):
    '''
    Compute permutation importance for all features.
    
    Shuffles each feature and measures increase in validation loss.
    Higher increase = more important feature.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        n_repeats: Number of shuffle repeats per feature (default 1 for speed)
        max_batches: Number of validation batches to cache for analysis
    
    Returns:
        Dict with feature names and their importance scores
    '''
    model.eval()
    
    # Cache a small number of batches once to avoid repeatedly iterating val_loader.
    cached_batches = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        cached_batches.append({
            'x_1d': batch['1d'].x,
            'x_2d': batch['2d'].x,
            'target_dict': {'1d': batch['1d'].y, '2d': batch['2d'].y},
            'edge_index_dict': {
                ('1d', 'pipe', '1d'): batch['1d', 'pipe', '1d'].edge_index,
                ('2d', 'surface', '2d'): batch['2d', 'surface', '2d'].edge_index,
                ('1d', 'couples', '2d'): batch['1d', 'couples', '2d'].edge_index,
                ('2d', 'couples', '1d'): batch['2d', 'couples', '1d'].edge_index,
            },
            'edge_attr_dict': extract_edge_attr_dict(batch),
        })
    if len(cached_batches) == 0:
        raise ValueError("Validation loader produced zero batches for permutation importance")

    # Baseline is computed once using cached batches.
    baseline_losses = []
    for cached in cached_batches:
        pred_dict = model(
            {'1d': cached['x_1d'], '2d': cached['x_2d']},
            cached['edge_index_dict'],
            cached['edge_attr_dict'],
        )
        pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
        loss, _, _ = criterion(pred_single, cached['target_dict'])
        baseline_losses.append(loss.item())
    baseline_loss = np.mean(baseline_losses)
    
    importance = {}
    
    n_total_features = len(FEATURE_NAMES_1D) + len(FEATURE_NAMES_2D)
    pbar = tqdm(total=n_total_features, desc="Permutation importance", leave=False)

    # 1D features
    n_features_1d = len(FEATURE_NAMES_1D)
    for feat_idx in range(n_features_1d):
        feat_name = FEATURE_NAMES_1D[feat_idx]
        repeat_losses = []
        
        for _ in range(n_repeats):
            losses = []
            for cached in cached_batches:
                x_1d = cached['x_1d'].clone()
                
                # Shuffle this feature across all nodes
                perm = torch.randperm(x_1d.shape[0], device=device)
                x_1d[:, feat_idx] = x_1d[perm, feat_idx]
                
                pred_dict = model(
                    {'1d': x_1d, '2d': cached['x_2d']},
                    cached['edge_index_dict'],
                    cached['edge_attr_dict'],
                )
                pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
                loss, _, _ = criterion(pred_single, cached['target_dict'])
                losses.append(loss.item())
            repeat_losses.append(np.mean(losses))
        
        importance[f"1d_{feat_name}"] = np.mean(repeat_losses) - baseline_loss
        pbar.update(1)
    
    # 2D features
    n_features_2d = len(FEATURE_NAMES_2D)
    for feat_idx in range(n_features_2d):
        feat_name = FEATURE_NAMES_2D[feat_idx]
        repeat_losses = []
        
        for _ in range(n_repeats):
            losses = []
            for cached in cached_batches:
                x_2d = cached['x_2d'].clone()
                
                # Shuffle this feature across all nodes
                perm = torch.randperm(x_2d.shape[0], device=device)
                x_2d[:, feat_idx] = x_2d[perm, feat_idx]
                
                pred_dict = model(
                    {'1d': cached['x_1d'], '2d': x_2d},
                    cached['edge_index_dict'],
                    cached['edge_attr_dict'],
                )
                pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
                loss, _, _ = criterion(pred_single, cached['target_dict'])
                losses.append(loss.item())
            repeat_losses.append(np.mean(losses))
        
        importance[f"2d_{feat_name}"] = np.mean(repeat_losses) - baseline_loss
        pbar.update(1)
    pbar.close()
    
    return importance, baseline_loss


def compute_gradient_importance(model, sample_batch, device):
    '''
    Compute gradient-based feature importance (faster than permutation).
    
    Measures how much the output changes with respect to each input feature.
    Higher gradient magnitude = more important feature.
    
    Args:
        model: Trained model
        sample_batch: A single batch of data
        device: Device to use
    
    Returns:
        importance_1d, importance_2d: Arrays of importance per feature
    '''
    model.eval()
    
    # Enable gradients for this computation
    sample_batch = sample_batch.to(device)
    
    x_1d = sample_batch['1d'].x.clone().requires_grad_(True)
    x_2d = sample_batch['2d'].x.clone().requires_grad_(True)
    
    edge_index_dict = {
        ('1d', 'pipe', '1d'): sample_batch['1d', 'pipe', '1d'].edge_index,
        ('2d', 'surface', '2d'): sample_batch['2d', 'surface', '2d'].edge_index,
        ('1d', 'couples', '2d'): sample_batch['1d', 'couples', '2d'].edge_index,
        ('2d', 'couples', '1d'): sample_batch['2d', 'couples', '1d'].edge_index,
    }
    edge_attr_dict = extract_edge_attr_dict(sample_batch)
    
    # Forward pass
    pred_dict = model({'1d': x_1d, '2d': x_2d}, edge_index_dict, edge_attr_dict)
    
    # Compute gradients w.r.t. outputs (sum to get scalar)
    output = pred_dict['1d'].abs().sum() + pred_dict['2d'].abs().sum()
    output.backward()
    
    # Mean absolute gradient per feature = importance
    importance_1d = x_1d.grad.abs().mean(dim=0).cpu().numpy()
    importance_2d = x_2d.grad.abs().mean(dim=0).cpu().numpy()
    
    return importance_1d, importance_2d


def log_feature_importance(importance: Dict[str, float], baseline_loss: float, method: str = "Permutation"):
    '''Log feature importance in a readable format.'''
    logger.info("=" * 60)
    logger.info(f"{method} Feature Importance (baseline loss: {baseline_loss:.4f})")
    logger.info("=" * 60)
    
    # Sort by importance (higher = more important)
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("Top 10 Most Important Features:")
    for i, (name, imp) in enumerate(sorted_imp[:10]):
        logger.info(f"  {i+1:2d}. {name:25s}: {imp:+.4f} (loss increase)")
    
    logger.info("")
    logger.info("1D Features:")
    for name, imp in sorted_imp:
        if name.startswith("1d_"):
            bar = "█" * int(max(0, imp) * 100)
            logger.info(f"  {name[3:]:20s}: {imp:+.4f} {bar}")
    
    logger.info("")
    logger.info("2D Features:")
    for name, imp in sorted_imp:
        if name.startswith("2d_"):
            bar = "█" * int(max(0, imp) * 100)
            logger.info(f"  {name[3:]:20s}: {imp:+.4f} {bar}")
    
    logger.info("=" * 60)
    
    return sorted_imp


def analyze_feature_importance(model, val_loader, criterion, device, method: str = "gradient"):
    '''
    Analyze and log feature importance for the trained model.
    
    Args:
        model: Trained model
        val_loader: Validation data loader  
        criterion: Loss function
        device: Device
        method: "gradient" (fast) or "permutation" (accurate)
    
    Returns:
        Dict of feature importances
    '''
    logger.info(f"Computing {method} feature importance...")
    
    if method == "gradient":
        # Use first batch for gradient importance (fast)
        sample_batch = next(iter(val_loader))
        imp_1d, imp_2d = compute_gradient_importance(model, sample_batch, device)
        
        # Convert to dict format
        importance = {}
        for i, name in enumerate(FEATURE_NAMES_1D):
            importance[f"1d_{name}"] = float(imp_1d[i])
        for i, name in enumerate(FEATURE_NAMES_2D):
            importance[f"2d_{name}"] = float(imp_2d[i])
        
        baseline = 0.0  # Not applicable for gradient method
        
        # Log results
        logger.info("=" * 60)
        logger.info("Gradient Feature Importance (mean |grad| per feature)")
        logger.info("=" * 60)
        
        # Sort and log
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 Most Important Features:")
        for i, (name, imp) in enumerate(sorted_imp[:10]):
            logger.info(f"  {i+1:2d}. {name:25s}: {imp:.4f}")
        
    else:  # permutation
        importance, baseline = compute_permutation_importance(
            model, val_loader, criterion, device, n_repeats=1, max_batches=5
        )
        log_feature_importance(importance, baseline, "Permutation")
    
    return importance


def find_optimal_lr(model, train_loader, criterion, device, 
                    start_lr: float = 1e-7, end_lr: float = 1e-1, 
                    num_iters: int = 100) -> float:
    '''
    Learning rate range test (Smith, 2017).
    Finds optimal LR by training with exponentially increasing LR and tracking loss.
    
    Returns the LR where loss decreases fastest (steepest gradient).
    '''
    import copy
    import math
    
    logger.info(f"Running LR finder: {start_lr:.0e} → {end_lr:.0e} over {num_iters} iterations")
    
    # Save model state to restore later
    model_state = copy.deepcopy(model.state_dict())
    
    # Setup optimizer with start LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-5,
                                   fused=device.type == 'cuda')
    
    # Exponential LR schedule
    gamma = (end_lr / start_lr) ** (1 / num_iters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    model.train()
    lrs, losses = [], []
    best_loss = float('inf')
    smoothed_loss = 0
    
    data_iter = iter(train_loader)
    use_amp = Config.USE_AMP and device.type == 'cuda'
    
    for i in range(num_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        x_dict = {'1d': batch['1d'].x, '2d': batch['2d'].x}
        edge_index_dict = {
            ('1d', 'pipe', '1d'): batch['1d', 'pipe', '1d'].edge_index,
            ('2d', 'surface', '2d'): batch['2d', 'surface', '2d'].edge_index,
            ('1d', 'couples', '2d'): batch['1d', 'couples', '2d'].edge_index,
            ('2d', 'couples', '1d'): batch['2d', 'couples', '1d'].edge_index,
        }
        edge_attr_dict = extract_edge_attr_dict(batch)
        target_dict = {'1d': batch['1d'].y, '2d': batch['2d'].y}
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
                loss, _, _ = criterion(pred_dict, target_dict)
        else:
            pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            loss, _, _ = criterion(pred_dict, target_dict)
        
        # Smoothed loss for stability
        if i == 0:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss.item()
        
        # Stop if loss explodes
        if i > 10 and smoothed_loss > 4 * best_loss:
            logger.info(f"  Loss exploded at LR={optimizer.param_groups[0]['lr']:.2e}, stopping")
            break
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smoothed_loss)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Restore model state
    model.load_state_dict(model_state)
    
    # Find LR with steepest negative gradient (fastest loss decrease)
    # Use the LR about 1 order of magnitude before loss starts increasing
    if len(losses) < 10:
        optimal_lr = 1e-3  # fallback
    else:
        # Find minimum loss point
        min_loss_idx = losses.index(min(losses))
        # Go back a bit from minimum for safety margin
        optimal_idx = max(0, min_loss_idx - num_iters // 10)
        optimal_lr = lrs[optimal_idx]
    
    logger.info(f"LR Finder Results:")
    logger.info(f"  Min loss: {min(losses):.4f} at LR={lrs[losses.index(min(losses))]:.2e}")
    logger.info(f"  Suggested LR: {optimal_lr:.2e}")
    
    return optimal_lr


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch_num=0):
    model.train()
    total_loss = 0
    total_loss_1d = 0
    total_loss_2d = 0
    n_batches = 0
    accum_steps = Config.GRAD_ACCUM_STEPS
    
    # Timing accumulators
    time_data = 0
    time_transfer = 0
    time_forward = 0
    time_backward = 0
    time_optim = 0
    
    # Use non-blocking transfers for CUDA
    non_blocking = device.type == 'cuda'
    
    import time
    epoch_start = time.time()
    data_start = time.time()
    
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Time: Data loading
        time_data += time.time() - data_start
        
        # Time: GPU transfer
        transfer_start = time.time()
        batch = batch.to(device, non_blocking=non_blocking)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure transfer is complete for timing
        time_transfer += time.time() - transfer_start
        
        x_dict = {'1d': batch['1d'].x, '2d': batch['2d'].x}
        edge_index_dict = {
            ('1d', 'pipe', '1d'): batch['1d', 'pipe', '1d'].edge_index,
            ('2d', 'surface', '2d'): batch['2d', 'surface', '2d'].edge_index,
            ('1d', 'couples', '2d'): batch['1d', 'couples', '2d'].edge_index,
            ('2d', 'couples', '1d'): batch['2d', 'couples', '1d'].edge_index,
        }
        edge_attr_dict = extract_edge_attr_dict(batch)
        target_dict = {'1d': batch['1d'].y, '2d': batch['2d'].y}
        
        # Time: Forward pass
        forward_start = time.time()
        
        # Forward pass with AMP
        if Config.USE_AMP and device.type == 'cuda':
            with torch.amp.autocast('cuda'):  # float16 (default) - more compatible
                pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
                # For temporally-bundled models, use only first-step delta for single-step training
                pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
                loss, loss_1d, loss_2d = criterion(pred_single, target_dict)
                loss = loss / accum_steps
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_forward += time.time() - forward_start
            
            # Time: Backward pass
            backward_start = time.time()
            scaler.scale(loss).backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_backward += time.time() - backward_start
            
            # Time: Optimizer step
            optim_start = time.time()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            time_optim += time.time() - optim_start
        else:
            pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
            loss, loss_1d, loss_2d = criterion(pred_single, target_dict)
            loss = loss / accum_steps

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss in train_epoch at batch {i}; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            data_start = time.time()
            continue
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_forward += time.time() - forward_start
            
            backward_start = time.time()
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_backward += time.time() - backward_start
            
            optim_start = time.time()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            time_optim += time.time() - optim_start
        
        backward_start = time.time()
        loss.backward()
        device_synchronize(device)
        time_backward += time.time() - backward_start
        
        optim_start = time.time()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        time_optim += time.time() - optim_start
        
        total_loss += loss.item() * accum_steps
        total_loss_1d += loss_1d.item()
        total_loss_2d += loss_2d.item()
        n_batches += 1
        
        # Log timing breakdown every 50 batches (first epoch only for visibility)
        if epoch_num == 0 and (i + 1) % 50 == 0:
            avg_per_batch = (time.time() - epoch_start) / (i + 1)
            logger.debug(f"[Batch {i+1}/{len(loader)}] Avg: {avg_per_batch:.3f}s/batch")
            logger.debug(f"  Data Load: {time_data/(i+1)*1000:.1f}ms | "
                  f"Transfer: {time_transfer/(i+1)*1000:.1f}ms | "
                  f"Forward: {time_forward/(i+1)*1000:.1f}ms | "
                  f"Backward: {time_backward/(i+1)*1000:.1f}ms | "
                  f"Optim: {time_optim/(i+1)*1000:.1f}ms")
            logger.debug(f"  Loss: {total_loss/n_batches:.4f} (1D: {total_loss_1d/n_batches:.4f}, 2D: {total_loss_2d/n_batches:.4f})")
            if device.type == 'cuda':
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_cached = torch.cuda.memory_reserved() / 1e9
                logger.debug(f"  GPU Memory: {mem_used:.2f}GB used, {mem_cached:.2f}GB reserved")
        
        # Reset timer for next data loading measurement
        data_start = time.time()
    
    # Log epoch summary
    epoch_time = time.time() - epoch_start
    logger.debug(f"Epoch Timing - Total: {epoch_time:.1f}s | Data: {time_data:.1f}s ({time_data/epoch_time*100:.1f}%) | "
          f"Forward: {time_forward:.1f}s ({time_forward/epoch_time*100:.1f}%) | "
          f"Backward: {time_backward:.1f}s ({time_backward/epoch_time*100:.1f}%)")
    
    if n_batches == 0:
        return float('nan'), float('nan'), float('nan')
    return total_loss / n_batches, total_loss_1d / n_batches, total_loss_2d / n_batches


def train_epoch_multistep(model, loader, optimizer, criterion, scaler, device, 
                          fb: FeatureBuilder, rollout_steps: int = 5, epoch_num: int = 0,
                          cfg=None):
    '''
    Multi-step rollout training epoch.
    Unrolls predictions for rollout_steps and accumulates loss over all steps.
    This bridges the gap between single-step training and autoregressive inference.
    '''
    if cfg is None:
        cfg = Config  # Fallback to default for backward compatibility
    
    model.train()
    total_loss = 0
    total_loss_1d = 0
    total_loss_2d = 0
    n_batches = 0
    accum_steps = cfg.GRAD_ACCUM_STEPS
    
    # Partial BPTT: how many rollout steps to let gradients flow through
    # before detaching. Teaches the model about multi-step error compounding.
    bptt_steps = getattr(cfg, 'BPTT_STEPS', 1)  # Default 1 = old fully-truncated behaviour
    
    # Noise injection (MeshGraphNets-style): corrupt input water levels with
    # small Gaussian noise so the model learns to recover from its own
    # prediction errors during autoregressive rollout.
    inject_noise = getattr(cfg, 'NOISE_INJECTION', False)
    noise_std_1d = getattr(cfg, 'NOISE_STD_1D', 0.003) if inject_noise else 0.0
    noise_std_2d = getattr(cfg, 'NOISE_STD_2D', 0.003) if inject_noise else 0.0
    
    # wl_prev features: previous water levels as input (self-consistent during rollout)
    wl_prev_steps = fb.wl_prev_steps
    wl_prev_dropout = getattr(cfg, 'WL_PREV_DROPOUT', 0.1) if wl_prev_steps > 0 else 0.0
    
    # All GPU setup is handled by FeatureBuilder
    fb.to(device)
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    
    use_amp = cfg.USE_AMP and device.type == 'cuda'
    non_blocking = device.type == 'cuda'
    # Temporal bundling factor (K deltas per forward pass)
    bundle_k = max(int(getattr(unwrap_model(model), 'temporal_bundle_k', 1)), 1)
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training (multistep)", leave=False)):
        batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
        
        batch_size = batch['wl_1d_init'].shape[0]
        edge_index_dict_batch, _ = fb.get_batched_edge_indices(batch_size)
        
        wl_1d_current = batch['wl_1d_init'].clone()  # [batch, n_1d]
        wl_2d_current = batch['wl_2d_init'].clone()  # [batch, n_2d]
        
        rain_lag_all = batch['rain_lag_all']
        future_rain_all = batch.get('future_rain_all', None)
        global_rain_seq = batch.get('global_rain_seq', None)
        global_rain_lag_all = batch.get('global_rain_lag_all', None)
        global_future_rain_all = batch.get('global_future_rain_all', None)
        global_cum_rain_seq = batch.get('global_cum_rain_seq', None)
        
        step_losses = []
        step_losses_1d = []
        step_losses_2d = []
        
        cum_rain_seq = batch['cum_rain_seq']          # [batch, rollout_steps, n_2d]
        normalized_t_seq = batch['normalized_t_seq']  # [batch, rollout_steps]
        
        # Initialize wl_prevs: list of [batch, n_nodes] for each lag step
        # wl_prevs_*d_init shape: [batch, wl_prev_steps, n_nodes]
        if wl_prev_steps > 0:
            wl_prevs_1d = [batch['wl_prevs_1d_init'][:, i, :].clone()
                           for i in range(wl_prev_steps)]
            wl_prevs_2d = [batch['wl_prevs_2d_init'][:, i, :].clone()
                           for i in range(wl_prev_steps)]
            if inject_noise:
                wl_prevs_1d = [wp + torch.randn_like(wp) * noise_std_1d for wp in wl_prevs_1d]
                wl_prevs_2d = [wp + torch.randn_like(wp) * noise_std_2d for wp in wl_prevs_2d]
        else:
            wl_prevs_1d = None
            wl_prevs_2d = None
        # Previous 1D state for dynamic edge features (t-1 baseline).
        wl_1d_prev_current = wl_prevs_1d[0].clone() if wl_prevs_1d else wl_1d_current.clone()
        wl_2d_prev_current = wl_prevs_2d[0].clone() if wl_prevs_2d else wl_2d_current.clone()
        
        # We iterate effective rollout steps in chunks of size bundle_k.
        step = 0
        while step < rollout_steps:
            rain_current = batch['rainfall_seq'][:, step, :]
            rain_lag_step = rain_lag_all[:, step, :, :]  # [batch, rain_lag_steps, n_2d]
            cum_rain_step = cum_rain_seq[:, step, :]     # [batch, n_2d]
            norm_t_step = normalized_t_seq[:, step]       # [batch]
            global_rain_step = global_rain_seq[:, step] if global_rain_seq is not None else None
            global_rain_lag_step = global_rain_lag_all[:, step, :] if global_rain_lag_all is not None else None
            global_cum_rain_step = global_cum_rain_seq[:, step] if global_cum_rain_seq is not None else None
            if future_rain_all is not None and future_rain_all.shape[1] > 0:
                future_rain_step = future_rain_all[:, step, :, :]  # [batch, rain_future_steps, n_2d]
            else:
                future_rain_step = None
            if global_future_rain_all is not None and global_future_rain_all.shape[2] > 0:
                global_future_rain_step = global_future_rain_all[:, step, :]
            else:
                global_future_rain_step = None
            
            # Noise injection on water level inputs
            if inject_noise:
                wl_1d_input = wl_1d_current + torch.randn_like(wl_1d_current) * noise_std_1d
                wl_2d_input = wl_2d_current + torch.randn_like(wl_2d_current) * noise_std_2d
            else:
                wl_1d_input = wl_1d_current
                wl_2d_input = wl_2d_current
            
            # wl_prev feature dropout: per-sample, zero ALL wl_prev lags for robustness
            if wl_prev_steps > 0 and wl_prev_dropout > 0:
                keep = (torch.rand(batch_size, 1, device=device) > wl_prev_dropout).float()
                wps_1d = [wp * keep for wp in wl_prevs_1d]
                wps_2d = [wp * keep for wp in wl_prevs_2d]
            else:
                wps_1d = wl_prevs_1d
                wps_2d = wl_prevs_2d

            edge_attr_dict_batch = fb.build_edge_attr_dict_torch(
                wl_1d_input, wl_2d_input,
                wl_1d_prev=wl_1d_prev_current,
                wl_2d_prev=wl_2d_prev_current,
            )
            
            n_total_ts = batch.get('n_total_timesteps', None)  # [batch]
            x_dict = fb.build_x_dict_torch(
                wl_1d_input, wl_2d_input, rain_current, rain_lag_step,
                cum_rainfall_2d=cum_rain_step, normalized_t=norm_t_step,
                wl_prevs_1d=wps_1d, wl_prevs_2d=wps_2d,
                future_rain=future_rain_step,
                n_total_timesteps=n_total_ts if n_total_ts is not None else 0,
                global_rain_1d=global_rain_step,
                global_rain_lags_1d=global_rain_lag_step,
                global_future_rain_1d=global_future_rain_step,
                global_cum_rain_1d=global_cum_rain_step,
            )
            
            # Forward once to get K-step bundled deltas
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(x_dict, edge_index_dict_batch, edge_attr_dict_batch)
                delta_1d_all = pred['1d'].view(batch_size, n_1d, -1).float()
                delta_2d_all = pred['2d'].view(batch_size, n_2d, -1).float()
            else:
                pred = model(x_dict, edge_index_dict_batch, edge_attr_dict_batch)
                delta_1d_all = pred['1d'].view(batch_size, n_1d, -1)
                delta_2d_all = pred['2d'].view(batch_size, n_2d, -1)
            
            # Apply each bundled delta sequentially to advance water levels
            inner_k = delta_1d_all.shape[-1]
            for k_idx in range(inner_k):
                if step >= rollout_steps:
                    break
                
                delta_1d_pred = delta_1d_all[:, :, k_idx]
                delta_2d_pred = delta_2d_all[:, :, k_idx]
                
                target_wl_1d = batch['wl_1d_targets'][:, step, :]
                target_wl_2d = batch['wl_2d_targets'][:, step, :]
                
                wl_1d_pred = wl_1d_current + delta_1d_pred
                wl_2d_pred = wl_2d_current + delta_2d_pred
                
                pred_dict = {'1d': wl_1d_pred.flatten(), '2d': wl_2d_pred.flatten()}
                target_dict = {'1d': target_wl_1d.flatten(), '2d': target_wl_2d.flatten()}
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        step_loss, step_1d, step_2d = criterion(pred_dict, target_dict)
                else:
                    step_loss, step_1d, step_2d = criterion(pred_dict, target_dict)
                
                step_losses.append(step_loss)
                step_losses_1d.append(step_1d.item())
                step_losses_2d.append(step_2d.item())
                
                # === Partial BPTT + self-consistent wl_prev shift register ===
                # Shift: current becomes prev[0], prev[0] becomes prev[1], etc.
                if (step + 1) % bptt_steps == 0:
                    if wl_prev_steps > 0:
                        wl_prevs_1d = [wl_1d_current.detach().clone()] + \
                            [wp.detach().clone() for wp in wl_prevs_1d[:-1]]
                        wl_prevs_2d = [wl_2d_current.detach().clone()] + \
                            [wp.detach().clone() for wp in wl_prevs_2d[:-1]]
                    wl_1d_prev_current = wl_1d_current.detach().clone()
                    wl_2d_prev_current = wl_2d_current.detach().clone()
                    wl_1d_current = wl_1d_pred.detach().clone()
                    wl_2d_current = wl_2d_pred.detach().clone()
                else:
                    if wl_prev_steps > 0:
                        wl_prevs_1d = [wl_1d_current] + wl_prevs_1d[:-1]
                        wl_prevs_2d = [wl_2d_current] + wl_prevs_2d[:-1]
                    wl_1d_prev_current = wl_1d_current
                    wl_2d_prev_current = wl_2d_current
                    wl_1d_current = wl_1d_pred
                    wl_2d_current = wl_2d_pred
                
                step += 1
        
        # Average loss over rollout steps
        loss = sum(step_losses) / rollout_steps
        loss_1d = sum(step_losses_1d) / rollout_steps
        loss_2d = sum(step_losses_2d) / rollout_steps
        
        loss = loss / accum_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accum_steps
        total_loss_1d += loss_1d
        total_loss_2d += loss_2d
        n_batches += 1
    
    return total_loss / n_batches, total_loss_1d / n_batches, total_loss_2d / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_loss_1d = 0
    total_loss_2d = 0
    n_batches = 0
    non_blocking = device.type == 'cuda'
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device, non_blocking=non_blocking)
        
        x_dict = {'1d': batch['1d'].x, '2d': batch['2d'].x}
        edge_index_dict = {
            ('1d', 'pipe', '1d'): batch['1d', 'pipe', '1d'].edge_index,
            ('2d', 'surface', '2d'): batch['2d', 'surface', '2d'].edge_index,
            ('1d', 'couples', '2d'): batch['1d', 'couples', '2d'].edge_index,
            ('2d', 'couples', '1d'): batch['2d', 'couples', '1d'].edge_index,
        }
        edge_attr_dict = extract_edge_attr_dict(batch)
        target_dict = {'1d': batch['1d'].y, '2d': batch['2d'].y}
        
        # Use AMP for inference too
        if Config.USE_AMP and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
                pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
                loss, loss_1d, loss_2d = criterion(pred_single, target_dict)
        else:
            pred_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            pred_single = collapse_temporal_bundle_for_single_step(pred_dict)
            loss, loss_1d, loss_2d = criterion(pred_single, target_dict)
        
        total_loss += loss.item()
        total_loss_1d += loss_1d.item()
        total_loss_2d += loss_2d.item()
        n_batches += 1
    
    return total_loss / n_batches, total_loss_1d / n_batches, total_loss_2d / n_batches


@torch.no_grad()
def run_ar_rollout(
    model,
    fb: 'FeatureBuilder',
    wl_1d_gt: np.ndarray,
    wl_2d_gt: np.ndarray,
    rainfall_all: np.ndarray,
    device: torch.device,
    warmup_steps: int = None,
    use_amp: bool = True,
    step_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''Single canonical autoregressive rollout for one event.

    All AR inference paths (validation, diagnostics, submission, spectral, phase)
    delegate here.

    Args:
        model: Trained HeteroFloodGNN (already in eval mode).
        fb: FeatureBuilder for this model (already on device).
        wl_1d_gt: [n_timesteps, n_1d] ground-truth 1D water levels (numpy).
        wl_2d_gt: [n_timesteps, n_2d] ground-truth 2D water levels (numpy).
        rainfall_all: [n_timesteps, n_2d] rainfall (numpy).
        device: torch device.
        warmup_steps: How many GT timesteps to use as warmup.
                      Defaults to Config.WARMUP_TIMESTEPS.
        use_amp: Whether to use torch.amp.autocast.
        step_callback: Optional callable(t, wl_1d_pred_t1, wl_2d_pred_t1,
                       delta_1d, delta_2d) called after each AR step.
                       Used by diagnostic callers to track per-step stats
                       without duplicating rollout code.

    Returns:
        wl_1d_pred: [n_timesteps, n_1d] predicted 1D water levels.
        wl_2d_pred: [n_timesteps, n_2d] predicted 2D water levels.
    '''
    if warmup_steps is None:
        warmup_steps = Config.WARMUP_TIMESTEPS

    n_timesteps = wl_1d_gt.shape[0]
    n_1d = fb.n_1d
    n_2d = fb.n_2d

    # Init predictions with GT warmup
    wl_1d_pred = np.zeros((n_timesteps, n_1d), dtype=np.float32)
    wl_2d_pred = np.zeros((n_timesteps, n_2d), dtype=np.float32)
    wl_1d_pred[:warmup_steps] = wl_1d_gt[:warmup_steps]
    wl_2d_pred[:warmup_steps] = wl_2d_gt[:warmup_steps]

    # Pre-load rainfall to GPU once
    rainfall_gpu = torch.as_tensor(rainfall_all, dtype=torch.float32, device=device)
    cum_rain_gpu = torch.cumsum(rainfall_gpu, dim=0)

    # Keep AR state on GPU throughout rollout (avoid NumPy->GPU copy each step)
    wl_1d_state = torch.as_tensor(
        wl_1d_pred[warmup_steps - 1], dtype=torch.float32, device=device
    )
    wl_2d_state = torch.as_tensor(
        wl_2d_pred[warmup_steps - 1], dtype=torch.float32, device=device
    )

    # Bootstrap wl_prevs from warmup GT: list of GPU tensors [t-1, t-2, ...]
    wl_prev_steps = fb.wl_prev_steps
    if wl_prev_steps > 0:
        wl_prevs_1d_gpu = []
        wl_prevs_2d_gpu = []
        for lag in range(1, wl_prev_steps + 1):
            t_lag = max(warmup_steps - 1 - lag, 0)
            wl_prevs_1d_gpu.append(torch.from_numpy(
                wl_1d_gt[t_lag].astype(np.float32)).to(device))
            wl_prevs_2d_gpu.append(torch.from_numpy(
                wl_2d_gt[t_lag].astype(np.float32)).to(device))
    else:
        wl_prevs_1d_gpu = None
        wl_prevs_2d_gpu = None
    wl_1d_prev_state = wl_prevs_1d_gpu[0].clone() if wl_prevs_1d_gpu else wl_1d_state.clone()
    wl_2d_prev_state = wl_prevs_2d_gpu[0].clone() if wl_prevs_2d_gpu else wl_2d_state.clone()

    # Temporal bundling factor (K deltas per forward pass)
    bundle_k = max(int(getattr(unwrap_model(model), 'temporal_bundle_k', 1)), 1)
    
    use_amp = bool(use_amp and device.type == 'cuda')
    t = warmup_steps - 1
    while t < n_timesteps - 1:
        x_dict = fb.build_x_dict_from_state(
            wl_1d_state, wl_2d_state, rainfall_gpu, cum_rain_gpu, t, n_timesteps,
            wl_prevs_1d=wl_prevs_1d_gpu, wl_prevs_2d=wl_prevs_2d_gpu)
        edge_attr_dict = fb.build_edge_attr_dict_torch(
            wl_1d_state, wl_2d_state,
            wl_1d_prev=wl_1d_prev_state,
            wl_2d_prev=wl_2d_prev_state,
        )
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred = model(x_dict, fb.edge_index_dict, edge_attr_dict)
        else:
            pred = model(x_dict, fb.edge_index_dict, edge_attr_dict)
        
        delta_1d_all = pred['1d'].float()
        delta_2d_all = pred['2d'].float()
        if delta_1d_all.dim() == 1:
            # K=1 backward-compatible path: [n_nodes]
            delta_1d_all = delta_1d_all.unsqueeze(-1)
            delta_2d_all = delta_2d_all.unsqueeze(-1)
        
        inner_k = min(delta_1d_all.shape[-1], n_timesteps - 1 - t)
        
        for k_idx in range(inner_k):
            next_t = t + 1 + k_idx
            prev_1d = wl_1d_state
            prev_2d = wl_2d_state
            delta_1d = delta_1d_all[:, k_idx]
            delta_2d = delta_2d_all[:, k_idx]
            
            # Shift register: current → prev[0], prev[0] → prev[1], ...
            if wl_prev_steps > 0:
                wl_prevs_1d_gpu = [prev_1d.clone()] + wl_prevs_1d_gpu[:-1]
                wl_prevs_2d_gpu = [prev_2d.clone()] + wl_prevs_2d_gpu[:-1]
            
            wl_1d_state = prev_1d + delta_1d
            wl_2d_state = prev_2d + delta_2d
            wl_1d_prev_state = prev_1d
            wl_2d_prev_state = prev_2d
            
            wl_1d_pred[next_t] = wl_1d_state.cpu().numpy()
            wl_2d_pred[next_t] = wl_2d_state.cpu().numpy()
            
            if step_callback is not None:
                step_callback(next_t - 1, prev_1d, prev_2d,
                              wl_1d_pred[next_t], wl_2d_pred[next_t],
                              delta_1d, delta_2d)
        
        t += inner_k

    return wl_1d_pred, wl_2d_pred


@torch.no_grad()
def autoregressive_validation(model, static_data: StaticGraphData, val_events: List[str], 
                               model_path: Path, std_1d: float, std_2d: float,
                               device: torch.device, verbose: bool = True,
                               feature_scales: Dict[str, float] = None,
                               fb: 'FeatureBuilder' = None,
                               plot_random_events: int = 0,
                               plot_output_dir: Optional[Path] = None,
                               plot_tag: str = "",
                               plot_seed: Optional[int] = None) -> Tuple[float, float, float]:
    '''
    Run autoregressive rollout on validation events to get realistic error.
    Returns standardized RMSE computed exactly as per competition rules.
    
    Also logs detailed diagnostics:
    - Per-timestep RMSE (horizon curve)
    - Delta prediction bias (mean/std)
    - Physical violations (negative depths, extreme jumps)
    '''
    model.eval()
    
    # Store per-event scores (competition gives equal weight to each event)
    event_scores = []
    
    # === DIAGNOSTIC ACCUMULATORS ===
    # 1) Per-horizon RMSE tracking (averaged across events)
    max_horizon = 200  # Max timesteps to track
    horizon_errors_1d = [[] for _ in range(max_horizon)]
    horizon_errors_2d = [[] for _ in range(max_horizon)]
    
    # 2) Delta bias tracking
    all_delta_errors_1d = []  # pred_delta - true_delta
    all_delta_errors_2d = []
    
    # 3) Physical violation tracking
    total_violations = {
        'negative_pond_depth': 0,
        'extreme_delta_1d': 0,  # |Δwl| > 5
        'extreme_delta_2d': 0,
    }
    total_predictions = {'1d': 0, '2d': 0}
    wl_ranges = {'1d_min': float('inf'), '1d_max': float('-inf'),
                 '2d_min': float('inf'), '2d_max': float('-inf')}
    
    # 4) NEW: Spatial smoothness tracking (Laplacian test)
    spatial_error_diffs_2d = []  # |error_i - error_j| for neighbors
    
    # 5) NEW: Error correlation with physical variables
    all_errors_1d = []
    all_errors_2d = []
    all_wl_1d = []  # Water levels at prediction time
    all_wl_2d = []
    all_rainfall = []  # Rainfall at prediction time
    
    # 6) NEW: Error growth rate tracking (for exponential vs linear detection)
    horizon_rmse_values = []  # Store (horizon, rmse_1d, rmse_2d) tuples
    
    # Use all validation events for testing
    events_to_eval = val_events
    
    # Optional: save per-event GT vs Pred temporal plots for a random subset
    events_to_plot = set()
    plots_dir = None
    if plot_random_events > 0 and len(events_to_eval) > 0:
        rng = np.random.default_rng(plot_seed)
        n_plot = min(plot_random_events, len(events_to_eval))
        selected = rng.choice(events_to_eval, size=n_plot, replace=False)
        events_to_plot = set(selected.tolist())
        plots_dir = Path(plot_output_dir) if plot_output_dir is not None else (Config.OUTPUT_PATH / "ar_event_timeseries")
        plots_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"AR event plots: selected {n_plot} event(s): {sorted(events_to_plot)}")
    
    def _save_event_temporal_plot(
        event_name: str,
        wl_1d_gt_evt: np.ndarray,
        wl_1d_pred_evt: np.ndarray,
        wl_2d_gt_evt: np.ndarray,
        wl_2d_pred_evt: np.ndarray,
        out_path: Path,
        event_rmse_1d_evt: float,
        event_rmse_2d_evt: float,
    ) -> None:
        '''Save event-level GT vs Pred temporal plot (mean + P10/P90 bands).'''
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Skipping AR event plot for {event_name}: matplotlib unavailable ({exc})")
            return
        
        t = np.arange(wl_1d_gt_evt.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        
        # 1D summary
        gt_1d_mean = wl_1d_gt_evt.mean(axis=1)
        pred_1d_mean = wl_1d_pred_evt.mean(axis=1)
        gt_1d_p10 = np.percentile(wl_1d_gt_evt, 10, axis=1)
        gt_1d_p90 = np.percentile(wl_1d_gt_evt, 90, axis=1)
        pred_1d_p10 = np.percentile(wl_1d_pred_evt, 10, axis=1)
        pred_1d_p90 = np.percentile(wl_1d_pred_evt, 90, axis=1)
        
        axes[0].plot(t, gt_1d_mean, color='royalblue', lw=2.0, label='GT mean')
        axes[0].plot(t, pred_1d_mean, color='crimson', lw=2.0, label='Pred mean')
        axes[0].fill_between(t, gt_1d_p10, gt_1d_p90, color='royalblue', alpha=0.16, label='GT P10-P90')
        axes[0].fill_between(t, pred_1d_p10, pred_1d_p90, color='crimson', alpha=0.16, label='Pred P10-P90')
        axes[0].axvline(Config.WARMUP_TIMESTEPS, color='gray', ls='--', lw=1.0, alpha=0.8)
        axes[0].set_title(f"1D nodes | std-RMSE={event_rmse_1d_evt:.4f}")
        axes[0].set_ylabel("WL (m)")
        axes[0].grid(alpha=0.25, ls=':')
        axes[0].legend(loc='best', fontsize=9)
        
        # 2D summary
        gt_2d_mean = wl_2d_gt_evt.mean(axis=1)
        pred_2d_mean = wl_2d_pred_evt.mean(axis=1)
        gt_2d_p10 = np.percentile(wl_2d_gt_evt, 10, axis=1)
        gt_2d_p90 = np.percentile(wl_2d_gt_evt, 90, axis=1)
        pred_2d_p10 = np.percentile(wl_2d_pred_evt, 10, axis=1)
        pred_2d_p90 = np.percentile(wl_2d_pred_evt, 90, axis=1)
        
        axes[1].plot(t, gt_2d_mean, color='royalblue', lw=2.0, label='GT mean')
        axes[1].plot(t, pred_2d_mean, color='crimson', lw=2.0, label='Pred mean')
        axes[1].fill_between(t, gt_2d_p10, gt_2d_p90, color='royalblue', alpha=0.16, label='GT P10-P90')
        axes[1].fill_between(t, pred_2d_p10, pred_2d_p90, color='crimson', alpha=0.16, label='Pred P10-P90')
        axes[1].axvline(Config.WARMUP_TIMESTEPS, color='gray', ls='--', lw=1.0, alpha=0.8)
        axes[1].set_title(f"2D nodes | std-RMSE={event_rmse_2d_evt:.4f}")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("WL (m)")
        axes[1].grid(alpha=0.25, ls=':')
        axes[1].legend(loc='best', fontsize=9)
        
        title_suffix = f" | {plot_tag}" if plot_tag else ""
        fig.suptitle(f"AR Validation Event: {event_name}{title_suffix}", fontsize=12, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Build FeatureBuilder if not provided
    if fb is None:
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                            rain_lag_steps=Config.RAIN_LAG_STEPS,
                            rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
    fb.to(device)
    
    use_amp = Config.USE_AMP and device.type == 'cuda'
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    min_elev = fb.min_elev_2d  # Keep numpy version for diagnostics
    
    for event_name in events_to_eval:
        event_path = model_path / event_name
        
        # Load dynamic data using vectorized pivot (much faster than row-by-row)
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        
        n_timesteps = int(df_1d['timestep'].max() + 1)
        
        # Vectorized data extraction using pivot (O(N) instead of O(N*T) loops)
        wl_1d_gt = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        wl_2d_gt = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
        if 'rainfall' in df_2d.columns:
            pivot_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0)
            rainfall_all[:len(pivot_rain)] = pivot_rain.values
        
        # === Diagnostic callback — collects per-step stats via run_ar_rollout ===
        def _diag_callback(t, wl_1d_cur, wl_2d_cur, wl_1d_next, wl_2d_next, d1d, d2d):
            # Physical violations
            pond_depth_diag = wl_2d_cur - fb._gpu['min_elev_2d']
            total_violations['negative_pond_depth'] += int((pond_depth_diag < -0.01).sum().item())
            # Delta errors
            d1d_np = d1d.cpu().numpy()
            d2d_np = d2d.cpu().numpy()
            all_delta_errors_1d.extend((d1d_np - (wl_1d_gt[t + 1] - wl_1d_gt[t])).tolist())
            all_delta_errors_2d.extend((d2d_np - (wl_2d_gt[t + 1] - wl_2d_gt[t])).tolist())
            # Extreme deltas
            total_violations['extreme_delta_1d'] += np.sum(np.abs(d1d_np) > 5)
            total_violations['extreme_delta_2d'] += np.sum(np.abs(d2d_np) > 5)
            # Horizon errors
            horizon = t - Config.WARMUP_TIMESTEPS + 1
            err_1d = wl_1d_next - wl_1d_gt[t + 1]
            err_2d = wl_2d_next - wl_2d_gt[t + 1]
            if horizon < max_horizon:
                horizon_errors_1d[horizon].extend(err_1d.tolist())
                horizon_errors_2d[horizon].extend(err_2d.tolist())
            # Correlation tracking
            all_errors_1d.extend(err_1d.tolist())
            all_errors_2d.extend(err_2d.tolist())
            all_wl_1d.extend(wl_1d_gt[t].tolist())
            all_wl_2d.extend(wl_2d_gt[t].tolist())
            all_rainfall.extend(rainfall_all[t].tolist())

        wl_1d_pred, wl_2d_pred = run_ar_rollout(
            model, fb, wl_1d_gt, wl_2d_gt, rainfall_all, device,
            use_amp=use_amp, step_callback=_diag_callback,
        )
        
        # === DIAGNOSTIC: Water level ranges ===
        wl_ranges['1d_min'] = min(wl_ranges['1d_min'], wl_1d_pred[Config.WARMUP_TIMESTEPS:].min())
        wl_ranges['1d_max'] = max(wl_ranges['1d_max'], wl_1d_pred[Config.WARMUP_TIMESTEPS:].max())
        wl_ranges['2d_min'] = min(wl_ranges['2d_min'], wl_2d_pred[Config.WARMUP_TIMESTEPS:].min())
        wl_ranges['2d_max'] = max(wl_ranges['2d_max'], wl_2d_pred[Config.WARMUP_TIMESTEPS:].max())
        
        total_predictions['1d'] += (n_timesteps - Config.WARMUP_TIMESTEPS) * n_1d
        total_predictions['2d'] += (n_timesteps - Config.WARMUP_TIMESTEPS) * n_2d
        
        # === DIAGNOSTIC: Spatial error smoothness (Laplacian test) ===
        # Sample a few timesteps to avoid memory issues
        sample_ts = list(range(Config.WARMUP_TIMESTEPS + 10, min(n_timesteps, Config.WARMUP_TIMESTEPS + 50), 5))
        edge_src = static_data.edge_index_2d[0].numpy()
        edge_dst = static_data.edge_index_2d[1].numpy()
        for sample_t in sample_ts:
            if sample_t < n_timesteps:
                err_2d_at_t = wl_2d_pred[sample_t] - wl_2d_gt[sample_t]
                # Compute |error_i - error_j| for each edge
                err_diff = np.abs(err_2d_at_t[edge_src] - err_2d_at_t[edge_dst])
                spatial_error_diffs_2d.extend(err_diff.tolist())
        
        # Compute per-event errors (after warmup)
        errors_1d = (wl_1d_pred[Config.WARMUP_TIMESTEPS:] - wl_1d_gt[Config.WARMUP_TIMESTEPS:]).flatten()
        errors_2d = (wl_2d_pred[Config.WARMUP_TIMESTEPS:] - wl_2d_gt[Config.WARMUP_TIMESTEPS:]).flatten()
        
        # Per-event standardized RMSE (as per competition rules)
        event_rmse_1d = np.sqrt(np.mean(errors_1d ** 2)) / std_1d
        event_rmse_2d = np.sqrt(np.mean(errors_2d ** 2)) / std_2d
        
        # Event-level score: average of 1D and 2D (equal contribution)
        event_score = (event_rmse_1d + event_rmse_2d) / 2
        
        # Optional event-level temporal plots for a random subset
        if plots_dir is not None and event_name in events_to_plot:
            safe_event_name = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in event_name)
            plot_file = plots_dir / f"ar_val_{safe_event_name}_{plot_tag}.png" if plot_tag else plots_dir / f"ar_val_{safe_event_name}.png"
            _save_event_temporal_plot(
                event_name=event_name,
                wl_1d_gt_evt=wl_1d_gt,
                wl_1d_pred_evt=wl_1d_pred,
                wl_2d_gt_evt=wl_2d_gt,
                wl_2d_pred_evt=wl_2d_pred,
                out_path=plot_file,
                event_rmse_1d_evt=event_rmse_1d,
                event_rmse_2d_evt=event_rmse_2d,
            )
            if verbose:
                logger.info(f"Saved AR event plot: {plot_file}")
        
        event_scores.append({
            'event': event_name,
            'rmse_1d': event_rmse_1d,
            'rmse_2d': event_rmse_2d,
            'score': event_score
        })
    
    # Average across all events (equal weight per event, as per competition)
    rmse_1d = np.mean([e['rmse_1d'] for e in event_scores])
    rmse_2d = np.mean([e['rmse_2d'] for e in event_scores])
    rmse_total = np.mean([e['score'] for e in event_scores])
    
    # === LOG DIAGNOSTICS ===
    if verbose:
        logger.info("=" * 50)
        logger.info("AR VALIDATION DIAGNOSTICS")
        logger.info("=" * 50)
        
        # 1) Horizon curve (sample at key points)
        logger.info("Horizon RMSE (steps after warmup):")
        for h in [0, 5, 10, 20, 50, 100]:
            if h < len(horizon_errors_1d) and len(horizon_errors_1d[h]) > 0:
                h_rmse_1d = np.sqrt(np.mean(np.array(horizon_errors_1d[h]) ** 2)) / std_1d
                h_rmse_2d = np.sqrt(np.mean(np.array(horizon_errors_2d[h]) ** 2)) / std_2d
                logger.info(f"  t+{h:3d}: 1D={h_rmse_1d:.4f}, 2D={h_rmse_2d:.4f}")
        
        # 2) Delta bias
        delta_err_1d = np.array(all_delta_errors_1d)
        delta_err_2d = np.array(all_delta_errors_2d)
        logger.info("Delta (Δwl) prediction bias:")
        logger.info(f"  1D: mean={delta_err_1d.mean():.4f}, std={delta_err_1d.std():.4f}")
        logger.info(f"  2D: mean={delta_err_2d.mean():.4f}, std={delta_err_2d.std():.4f}")
        
        if abs(delta_err_1d.mean()) > 0.01:
            logger.warning(f"  ⚠ 1D has bias of {delta_err_1d.mean():.4f} - will cause AR drift!")
        if abs(delta_err_2d.mean()) > 0.01:
            logger.warning(f"  ⚠ 2D has bias of {delta_err_2d.mean():.4f} - will cause AR drift!")
        
        # 3) Physical violations
        logger.info("Physical violations:")
        logger.info(f"  Negative pond depth: {total_violations['negative_pond_depth']:,} "
                   f"({100*total_violations['negative_pond_depth']/max(1,total_predictions['2d']):.2f}%)")
        logger.info(f"  Extreme Δwl (>5): 1D={total_violations['extreme_delta_1d']:,}, "
                   f"2D={total_violations['extreme_delta_2d']:,}")
        
        # 4) Water level ranges
        logger.info("Predicted water level ranges:")
        logger.info(f"  1D: [{wl_ranges['1d_min']:.2f}, {wl_ranges['1d_max']:.2f}]")
        logger.info(f"  2D: [{wl_ranges['2d_min']:.2f}, {wl_ranges['2d_max']:.2f}]")
        
        # 5) NEW: Spatial error smoothness (Laplacian test)
        if len(spatial_error_diffs_2d) > 0:
            spatial_diffs = np.array(spatial_error_diffs_2d)
            logger.info("Spatial error smoothness (2D neighbor differences):")
            logger.info(f"  Mean |err_i - err_j|: {spatial_diffs.mean():.4f}")
            logger.info(f"  Std |err_i - err_j|:  {spatial_diffs.std():.4f}")
            logger.info(f"  % edges with diff > 0.5: {100*np.mean(spatial_diffs > 0.5):.2f}%")
            if spatial_diffs.mean() > 0.3:
                logger.warning("  ⚠ High spatial non-smoothness - consider Laplacian regularization")
        
        # 6) NEW: Error growth rate analysis
        logger.info("Error growth analysis:")
        # Fit exponential vs linear to horizon RMSE
        horizons_to_analyze = [0, 5, 10, 20, 50, 100]
        valid_horizons = []
        rmse_at_horizons = []
        for h in horizons_to_analyze:
            if h < len(horizon_errors_1d) and len(horizon_errors_1d[h]) > 0:
                combined_rmse = 0.5 * (np.sqrt(np.mean(np.array(horizon_errors_1d[h]) ** 2)) / std_1d +
                                       np.sqrt(np.mean(np.array(horizon_errors_2d[h]) ** 2)) / std_2d)
                valid_horizons.append(h)
                rmse_at_horizons.append(combined_rmse)
        
        if len(valid_horizons) >= 3:
            horizons_arr = np.array(valid_horizons, dtype=float)
            rmse_arr = np.array(rmse_at_horizons)
            # Linear fit: RMSE = a * horizon + b
            linear_coeffs = np.polyfit(horizons_arr, rmse_arr, 1)
            linear_slope = linear_coeffs[0]
            # Exponential fit: log(RMSE) = a * horizon + b => RMSE = exp(b) * exp(a*horizon)
            safe_rmse = np.maximum(rmse_arr, 1e-6)
            exp_coeffs = np.polyfit(horizons_arr, np.log(safe_rmse), 1)
            exp_rate = exp_coeffs[0]  # Growth rate per timestep
            
            logger.info(f"  Linear growth rate: {linear_slope:.6f} per timestep")
            logger.info(f"  Exponential rate: {exp_rate:.6f} (doubling every {0.693/max(exp_rate, 1e-6):.1f} steps)")
            
            if exp_rate > 0.02:
                logger.warning("  ⚠ Exponential error growth detected - model is unstable (Lipschitz > 1)")
                logger.warning("  → Consider: spectral normalization, smaller model, bias regularization")
            elif linear_slope > 0.005:
                logger.warning("  ⚠ Linear error accumulation detected - systematic bias issue")
                logger.warning("  → Consider: focal loss on peak events, more AR training steps")
        
        # 7) NEW: Error correlation with physical variables
        logger.info("Error correlation with physical variables:")
        if len(all_errors_2d) > 1000:
            errors_2d_arr = np.array(all_errors_2d)
            wl_2d_arr = np.array(all_wl_2d)
            rainfall_arr = np.array(all_rainfall)
            
            # Correlation with water level magnitude
            corr_wl = np.corrcoef(np.abs(errors_2d_arr), wl_2d_arr)[0, 1]
            # Correlation with rainfall
            corr_rain = np.corrcoef(np.abs(errors_2d_arr), rainfall_arr)[0, 1]
            
            logger.info(f"  Corr(|error|, water_level): {corr_wl:.4f}")
            logger.info(f"  Corr(|error|, rainfall): {corr_rain:.4f}")
            
            if abs(corr_wl) > 0.3:
                logger.warning(f"  ⚠ Errors correlated with water level - model struggles at {'high' if corr_wl > 0 else 'low'} WL")
            if abs(corr_rain) > 0.3:
                logger.warning(f"  ⚠ Errors correlated with rainfall - model misses rainfall response")
        
        logger.info("=" * 50)
    
    return rmse_total, rmse_1d, rmse_2d


# =============================================================================
# PHYSICS-CONSISTENT MASS BALANCE DIAGNOSTIC
# =============================================================================

class OutletRatingCurve:
    '''
    Calibrated rating curve for outlet: Q = a * (h - h0)^b
    
    Fitted from ground truth using mass balance:
    Q_outlet(t) = Σ inlet_flow(t) - dV_1d/dt
    '''
    
    def __init__(self, a: float = 1.0, b: float = 1.5, h0: float = 0.0):
        self.a = a
        self.b = b
        self.h0 = h0  # Threshold head (flow starts above this)
    
    def predict(self, h: np.ndarray) -> np.ndarray:
        '''Predict discharge from head.'''
        h_eff = np.maximum(h - self.h0, 0)
        return self.a * np.power(h_eff, self.b)
    
    @classmethod
    def fit_from_mass_balance(cls, wl_outlet: np.ndarray, inlet_flow_total: np.ndarray,
                               storage_1d: np.ndarray, invert_elev: float, dt: float = 300.0):
        '''
        Fit rating curve from ground truth mass balance.
        
        Args:
            wl_outlet: Water level at outlet node [n_timesteps]
            inlet_flow_total: Total inlet_flow into 1D system [n_timesteps] (m³/s)
            storage_1d: 1D storage volume [n_timesteps] (m³)
            invert_elev: Outlet invert elevation (m)
            dt: Timestep duration (s)
        
        Returns:
            Fitted OutletRatingCurve
        '''
        from scipy.optimize import curve_fit
        
        # Compute outlet discharge from mass balance: Q = inlet - dV/dt
        dV_dt = np.gradient(storage_1d, dt)  # m³/s
        Q_outlet = inlet_flow_total - dV_dt  # m³/s
        
        # Head above invert
        h = wl_outlet - invert_elev
        
        # Filter valid points (positive Q and h)
        valid = (Q_outlet > 0.01) & (h > 0.1)
        if valid.sum() < 10:
            logger.warning("Not enough valid points for rating curve fit, using defaults")
            return cls(a=1.0, b=1.5, h0=0.0)
        
        h_valid = h[valid]
        Q_valid = Q_outlet[valid]
        
        # Fit Q = a * h^b using log-linear regression
        try:
            # Log transform: log(Q) = log(a) + b*log(h)
            log_h = np.log(h_valid)
            log_Q = np.log(Q_valid)
            
            # Linear fit
            coeffs = np.polyfit(log_h, log_Q, 1)
            b_fit = coeffs[0]
            a_fit = np.exp(coeffs[1])
            
            # Clamp to reasonable values
            b_fit = np.clip(b_fit, 0.5, 2.5)
            a_fit = np.clip(a_fit, 0.01, 100.0)
            
            return cls(a=float(a_fit), b=float(b_fit), h0=0.0)
        
        except Exception as e:
            logger.warning(f"Rating curve fit failed: {e}, using defaults")
            return cls(a=1.0, b=1.5, h0=0.0)


def compute_1d_storage(wl_1d: np.ndarray, static_data: StaticGraphData) -> np.ndarray:
    '''
    Compute 1D network storage volume.
    
    Args:
        wl_1d: Water levels [n_timesteps, n_1d_nodes]
        static_data: Static graph data with base_area and invert_elevation
    
    Returns:
        Storage volume [n_timesteps] in m³
    '''
    invert_elev = static_data.nodes_1d['invert_elevation'].fillna(0).values
    base_area = static_data.nodes_1d['base_area'].fillna(0).values
    
    # Depth = wl - invert (clipped to >= 0)
    depth = np.maximum(wl_1d - invert_elev, 0)
    
    # Volume = depth * base_area, summed over nodes
    storage = (depth * base_area).sum(axis=1)
    
    return storage


def compute_2d_storage(wl_2d: np.ndarray, static_data: StaticGraphData) -> np.ndarray:
    '''
    Compute 2D surface storage volume.
    
    Args:
        wl_2d: Water levels [n_timesteps, n_2d_nodes]
        static_data: Static graph data with area and min_elevation
    
    Returns:
        Storage volume [n_timesteps] in m³
    '''
    min_elev = static_data.nodes_2d['min_elevation'].fillna(0).values
    area = static_data.nodes_2d['area'].fillna(0).values
    
    # Pond depth = wl - min_elevation (clipped to >= 0)
    depth = np.maximum(wl_2d - min_elev, 0)
    
    # Volume = depth * area, summed over nodes
    storage = (depth * area).sum(axis=1)
    
    return storage


@torch.no_grad()
def mass_balance_diagnostic(model, static_data: StaticGraphData, val_events: List[str],
                            model_path: Path, device: torch.device,
                            feature_scales: Dict[str, float] = None,
                            outlet_node_1d: int = None,
                            fb: 'FeatureBuilder' = None) -> Dict[str, float]:
    '''
    Physics-Consistent Mass Balance Diagnostic.
    
    This is the gold-standard diagnostic that:
    1. Calibrates outlet rating curve from ground truth
    2. Computes storage for 1D and 2D
    3. Computes mass residuals
    4. Returns comprehensive physics metrics
    
    Args:
        model: Trained GNN model
        static_data: Static graph structure
        val_events: List of validation event names
        model_path: Path to model data
        device: Torch device
        feature_scales: Feature normalization scales
        outlet_node_1d: Index of 1D outlet node (auto-detected if None)
    
    Returns:
        Dict with all diagnostic metrics
    '''
    model.eval()
    dt = 300.0  # 5 minutes in seconds
    
    # Auto-detect outlet node (node with base_area = 0 or lowest invert)
    if outlet_node_1d is None:
        base_areas = static_data.nodes_1d['base_area'].values
        if (base_areas == 0).any():
            outlet_node_1d = int(np.where(base_areas == 0)[0][0])
        else:
            outlet_node_1d = int(static_data.nodes_1d['invert_elevation'].idxmin())
    
    outlet_invert = static_data.nodes_1d.loc[outlet_node_1d, 'invert_elevation']
    logger.info(f"1D Outlet node: {outlet_node_1d} (invert elevation: {outlet_invert:.2f} m)")
    
    # Accumulators for metrics
    all_results = []
    all_rating_curves = []
    
    # Build FeatureBuilder if not provided — all GPU setup is centralized
    if fb is None:
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                            rain_lag_steps=Config.RAIN_LAG_STEPS,
                            rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
    fb.to(device)
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    area_2d = static_data.nodes_2d['area'].fillna(0).values
    
    use_amp = Config.USE_AMP and device.type == 'cuda'
    
    for event_name in tqdm(val_events, desc="Mass Balance Diagnostic"):
        event_path = model_path / event_name
        
        # Load ground truth data
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        
        n_timesteps = int(df_1d['timestep'].max() + 1)
        
        # Extract ground truth arrays
        wl_1d_gt = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        wl_2d_gt = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        
        # Inlet flow (total into 1D from 2D coupling)
        inlet_flow_all = df_1d.pivot(index='timestep', columns='node_idx', values='inlet_flow').fillna(0).values
        inlet_flow_total = inlet_flow_all.sum(axis=1)  # m³/s
        
        # Rainfall
        rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
        if 'rainfall' in df_2d.columns:
            pivot_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0)
            rainfall_all[:len(pivot_rain)] = pivot_rain.values
        
        # Total rainfall volume per timestep (mm -> m³)
        rain_volume = (rainfall_all / 1000.0 * area_2d).sum(axis=1) * dt  # m³ per timestep
        total_rain = rain_volume.sum()
        
        # Compute ground truth storage
        storage_1d_gt = compute_1d_storage(wl_1d_gt, static_data)
        storage_2d_gt = compute_2d_storage(wl_2d_gt, static_data)
        storage_total_gt = storage_1d_gt + storage_2d_gt
        
        # Fit rating curve from ground truth
        wl_outlet_gt = wl_1d_gt[:, outlet_node_1d]
        rating_curve = OutletRatingCurve.fit_from_mass_balance(
            wl_outlet_gt, inlet_flow_total, storage_1d_gt, outlet_invert, dt
        )
        all_rating_curves.append((rating_curve.a, rating_curve.b))
        
        # Compute ground truth outlet discharge
        h_outlet_gt = wl_outlet_gt - outlet_invert
        Q_outlet_gt = rating_curve.predict(h_outlet_gt)
        
        # === RUN MODEL PREDICTIONS (Autoregressive) ===
        wl_1d_pred, wl_2d_pred = run_ar_rollout(
            model, fb, wl_1d_gt, wl_2d_gt, rainfall_all, device, use_amp=use_amp)
        
        # === COMPUTE METRICS ===
        # Storage from predictions
        storage_1d_pred = compute_1d_storage(wl_1d_pred, static_data)
        storage_2d_pred = compute_2d_storage(wl_2d_pred, static_data)
        storage_total_pred = storage_1d_pred + storage_2d_pred
        
        # Outlet head and discharge from predictions
        wl_outlet_pred = wl_1d_pred[:, outlet_node_1d]
        h_outlet_pred = wl_outlet_pred - outlet_invert
        Q_outlet_pred = rating_curve.predict(h_outlet_pred)
        
        # === METRIC 1: Storage Errors ===
        dV_1d_true = storage_1d_gt[-1] - storage_1d_gt[0]
        dV_1d_pred = storage_1d_pred[-1] - storage_1d_pred[0]
        dV_2d_true = storage_2d_gt[-1] - storage_2d_gt[0]
        dV_2d_pred = storage_2d_pred[-1] - storage_2d_pred[0]
        dV_total_true = storage_total_gt[-1] - storage_total_gt[0]
        dV_total_pred = storage_total_pred[-1] - storage_total_pred[0]
        
        total_inflow_1d = inlet_flow_total.sum() * dt
        
        eps_1d = abs(dV_1d_pred - dV_1d_true) / max(total_inflow_1d, 1.0) * 100
        eps_2d = abs(dV_2d_pred - dV_2d_true) / max(total_rain, 1.0) * 100
        eps_total = abs(dV_total_pred - dV_total_true) / max(total_rain, 1.0) * 100
        
        # === METRIC 2: Outlet Discharge Error ===
        # Only compare after warmup
        Q_outlet_gt_post = Q_outlet_gt[Config.WARMUP_TIMESTEPS:]
        Q_outlet_pred_post = Q_outlet_pred[Config.WARMUP_TIMESTEPS:]
        
        Q_rmse = np.sqrt(np.mean((Q_outlet_pred_post - Q_outlet_gt_post) ** 2))
        Q_mae = np.mean(np.abs(Q_outlet_pred_post - Q_outlet_gt_post))
        Q_mean = np.mean(Q_outlet_gt_post)
        eps_Q = Q_rmse / max(Q_mean, 0.01) * 100
        
        # === METRIC 3: Mass Residual (per timestep) ===
        # Residual = Rain_in - dV_total - Q_outlet*dt - Q_2d_boundary*dt
        # For 2D boundary, use GT-derived value
        dV_total_per_step_true = np.diff(storage_total_gt)
        dV_total_per_step_pred = np.diff(storage_total_pred)
        
        # Implied 2D boundary outflow from GT
        Q_2d_boundary_gt = (rain_volume[:-1] - dV_total_per_step_true - Q_outlet_gt[:-1] * dt) / dt
        
        # Mass residual for predictions (using same boundary)
        residual = (rain_volume[:-1] - dV_total_per_step_pred - 
                    Q_outlet_pred[:-1] * dt - Q_2d_boundary_gt * dt)
        
        mean_residual = np.mean(np.abs(residual))
        mean_residual_pct = mean_residual / max(rain_volume.mean(), 0.01) * 100
        
        # === METRIC 4: Peak and Timing ===
        h_outlet_gt_post = h_outlet_gt[Config.WARMUP_TIMESTEPS:]
        h_outlet_pred_post = h_outlet_pred[Config.WARMUP_TIMESTEPS:]
        
        peak_gt = h_outlet_gt_post.max()
        peak_pred = h_outlet_pred_post.max()
        t_peak_gt = h_outlet_gt_post.argmax()
        t_peak_pred = h_outlet_pred_post.argmax()
        
        eps_peak = abs(peak_pred - peak_gt) / max(peak_gt, 0.01) * 100
        dt_peak = abs(t_peak_pred - t_peak_gt)
        
        # === METRIC 5: Recession Rate ===
        def fit_recession(h, t_peak, min_points=5):
            '''Fit exponential recession: h = h_peak * exp(-k*t)'''
            h_after = h[t_peak:]
            if len(h_after) < min_points:
                return 0.0
            t_after = np.arange(len(h_after))
            h_safe = np.maximum(h_after, 1e-6)
            try:
                coeffs = np.polyfit(t_after, np.log(h_safe / h_safe[0]), 1)
                return -coeffs[0]  # Recession rate (positive = draining)
            except:
                return 0.0
        
        k_gt = fit_recession(h_outlet_gt_post, t_peak_gt)
        k_pred = fit_recession(h_outlet_pred_post, t_peak_pred)
        ratio_recession = k_pred / max(k_gt, 1e-6)
        
        # === METRIC 6: Hydrograph Fidelity Score ===
        HFS = (0.25 * (1 - min(eps_peak / 100, 1)) +
               0.15 * max(0, 1 - dt_peak / 5) +
               0.20 * (1 - min(abs(ratio_recession - 1), 1)) +
               0.20 * (1 - min(eps_Q / 100, 1)) +
               0.20 * (1 - min(eps_total / 100, 1)))
        
        all_results.append({
            'event': event_name,
            'eps_1d_storage': eps_1d,
            'eps_2d_storage': eps_2d,
            'eps_total_storage': eps_total,
            'eps_Q_outlet': eps_Q,
            'mean_residual_pct': mean_residual_pct,
            'eps_peak': eps_peak,
            'dt_peak': dt_peak,
            'ratio_recession': ratio_recession,
            'HFS': HFS,
            'total_rain_m3': total_rain,
            'total_inflow_1d_m3': total_inflow_1d,
        })
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    
    # Log summary
    logger.info("=" * 70)
    logger.info("PHYSICS-CONSISTENT MASS BALANCE DIAGNOSTIC")
    logger.info("=" * 70)
    
    logger.info(f"Evaluated {len(val_events)} events")
    logger.info(f"1D Outlet: Node {outlet_node_1d} (invert = {outlet_invert:.2f} m)")
    
    # Average rating curve parameters
    avg_a = np.mean([rc[0] for rc in all_rating_curves])
    avg_b = np.mean([rc[1] for rc in all_rating_curves])
    logger.info(f"Rating Curve: Q = {avg_a:.4f} × h^{avg_b:.2f}")
    
    logger.info("")
    logger.info("-" * 50)
    logger.info("STORAGE ERRORS (Mass Conservation)")
    logger.info("-" * 50)
    logger.info(f"  1D Storage Error:     {results_df['eps_1d_storage'].mean():6.2f}% ± {results_df['eps_1d_storage'].std():5.2f}%")
    logger.info(f"  2D Storage Error:     {results_df['eps_2d_storage'].mean():6.2f}% ± {results_df['eps_2d_storage'].std():5.2f}%")
    logger.info(f"  Total Storage Error:  {results_df['eps_total_storage'].mean():6.2f}% ± {results_df['eps_total_storage'].std():5.2f}%")
    
    logger.info("")
    logger.info("-" * 50)
    logger.info("OUTLET DISCHARGE")
    logger.info("-" * 50)
    logger.info(f"  Q Outlet Error:       {results_df['eps_Q_outlet'].mean():6.2f}% ± {results_df['eps_Q_outlet'].std():5.2f}%")
    logger.info(f"  Mass Residual:        {results_df['mean_residual_pct'].mean():6.2f}% ± {results_df['mean_residual_pct'].std():5.2f}%")
    
    logger.info("")
    logger.info("-" * 50)
    logger.info("HYDROGRAPH METRICS")
    logger.info("-" * 50)
    logger.info(f"  Peak Error:           {results_df['eps_peak'].mean():6.2f}% ± {results_df['eps_peak'].std():5.2f}%")
    logger.info(f"  Time to Peak Error:   {results_df['dt_peak'].mean():6.2f} ± {results_df['dt_peak'].std():5.2f} timesteps")
    logger.info(f"  Recession Ratio:      {results_df['ratio_recession'].mean():6.2f} ± {results_df['ratio_recession'].std():5.2f}")
    
    logger.info("")
    logger.info("-" * 50)
    logger.info("OVERALL SCORE")
    logger.info("-" * 50)
    hfs_mean = results_df['HFS'].mean()
    logger.info(f"  Hydrograph Fidelity:  {hfs_mean:.3f}")
    
    if hfs_mean >= 0.8:
        logger.info("  ✓ EXCELLENT - Model has learned flood physics")
    elif hfs_mean >= 0.6:
        logger.info("  ○ ACCEPTABLE - Some physics learned, room for improvement")
    else:
        logger.warning("  ✗ POOR - Model has not learned correct physics")
    
    # Interpretation
    logger.info("")
    logger.info("-" * 50)
    logger.info("INTERPRETATION")
    logger.info("-" * 50)
    
    if results_df['eps_1d_storage'].mean() > 15:
        logger.warning("  ⚠ 1D mass not conserved → Check 1D decoder & coupling edges")
    if results_df['eps_2d_storage'].mean() > 20:
        logger.warning("  ⚠ 2D mass not conserved → Check rainfall feature encoding")
    if results_df['ratio_recession'].mean() < 0.8:
        logger.warning("  ⚠ Recession too slow → Mass accumulating, check outlet learning")
    if results_df['ratio_recession'].mean() > 1.2:
        logger.warning("  ⚠ Recession too fast → Mass disappearing, possible instability")
    if results_df['dt_peak'].mean() > 3:
        logger.warning("  ⚠ Timing error > 15 min → Rainfall-outlet lag not learned correctly")
    
    logger.info("=" * 70)
    
    # Return summary metrics
    return {
        'eps_1d_storage': results_df['eps_1d_storage'].mean(),
        'eps_2d_storage': results_df['eps_2d_storage'].mean(),
        'eps_total_storage': results_df['eps_total_storage'].mean(),
        'eps_Q_outlet': results_df['eps_Q_outlet'].mean(),
        'mean_residual_pct': results_df['mean_residual_pct'].mean(),
        'eps_peak': results_df['eps_peak'].mean(),
        'dt_peak': results_df['dt_peak'].mean(),
        'ratio_recession': results_df['ratio_recession'].mean(),
        'HFS': hfs_mean,
        'rating_curve_a': avg_a,
        'rating_curve_b': avg_b,
        'results_df': results_df,
    }


def train_model(model_id: int, resume: bool = False, resume_weights_only: bool = False):
    '''Train GNN for a single urban model.
    
    Args:
        model_id: Which model to train (1 or 2)
        resume: If True, load checkpoint before training
        resume_weights_only: If True, load only model weights from checkpoint
            and reset optimizer/scheduler/scaler/start_epoch for a fresh
            learning-rate/curriculum phase.
    '''
    # Get model-specific config
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    
    logger.info("=" * 60)
    logger.info(f"Training Model {model_id}")
    logger.info("=" * 60)
    
    # Log full configuration
    log_config(cfg, model_id, logger)
    
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    # Get events
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    logger.info(f"Found {len(events)} events")
    
    # Train/val split
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    train_events = events[:-n_val]
    logger.info(f"Train: {len(train_events)}, Val: {len(val_events)}")
    
    # Load static data
    static_data = StaticGraphData(model_path)
    logger.info(f"1D nodes: {static_data.num_1d}, 2D nodes: {static_data.num_2d}")
    
    # Compute std for standardized RMSE AND per-model dynamic feature scales
    std_values = compute_std_values(model_path, train_events, static_data)
    logger.info(f"Std values - 1D: {std_values['1d']:.4f}, 2D: {std_values['2d']:.4f}")
    
    # Create FeatureBuilder — single source of truth for features
    wl_prev_steps = getattr(cfg, 'WL_PREV_STEPS', 0)
    use_edge_feat = getattr(cfg, 'USE_EDGE_FEATURES', False)
    fb = FeatureBuilder(static_data, feature_scales=std_values,
                        rain_lag_steps=cfg.RAIN_LAG_STEPS,
                        rain_future_steps=getattr(cfg, 'RAIN_FUTURE_STEPS', 0),
                        wl_prev_steps=wl_prev_steps, use_edge_features=use_edge_feat)
    logger.info(f"FeatureBuilder: in_1d={fb.in_channels_1d}, in_2d={fb.in_channels_2d}")
    
    # Create datasets with the current curriculum rollout length.
    initial_rollout = cfg.get_rollout_for_epoch(0)
    train_dataset = SequenceFloodDataset(
        model_id, train_events, static_data, 
        rollout_steps=initial_rollout, mode='train',
        feature_builder=fb
    )
    val_dataset = FloodEventDataset(model_id, val_events, static_data, mode='train',
                                     feature_builder=fb)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    bptt_steps = getattr(cfg, 'BPTT_STEPS', 1)
    if cfg.ROLLOUT_SCHEDULE is not None:
        logger.info(f"Curriculum rollout schedule: {cfg.ROLLOUT_SCHEDULE}")
        logger.info(f"  Initial rollout = {initial_rollout} steps")
        logger.info(f"  BPTT through {bptt_steps} steps per segment")
    elif bptt_steps > 1:
        logger.info(f"Multi-step training with {cfg.ROLLOUT_STEPS}-step rollout, "
                     f"partial BPTT through {bptt_steps} steps")
    else:
        logger.info(f"Multi-step training with {cfg.ROLLOUT_STEPS}-step rollout (fully truncated BPTT)")
    
    # Data loaders with prefetching for CUDA
    # Training loader uses sequence collate
    train_loader_kwargs = {
        'batch_size': cfg.BATCH_SIZE,
        'num_workers': cfg.NUM_WORKERS,
        'collate_fn': collate_sequence,  # Custom collate for sequence data
        'pin_memory': cfg.DEVICE.type == 'cuda',
        'persistent_workers': cfg.NUM_WORKERS > 0,
    }
    # Validation loader uses hetero collate
    val_loader_kwargs = {
        'batch_size': cfg.BATCH_SIZE,
        'num_workers': cfg.NUM_WORKERS,
        'collate_fn': collate_hetero,
        'pin_memory': cfg.DEVICE.type == 'cuda',
        'persistent_workers': cfg.NUM_WORKERS > 0,
    }
    if cfg.NUM_WORKERS > 0:
        train_loader_kwargs['prefetch_factor'] = 2
        val_loader_kwargs['prefetch_factor'] = 2
    
    train_sampler = None
    val_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False, drop_last=False
        )

    train_loader = DataLoader(
        train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, **train_loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, sampler=val_sampler, **val_loader_kwargs
    )
    
    # Model - dynamic features with coupled rainfall for 1D:
    # 1D: wl, pipe_fill, head_above, coupled_rainfall, coupled_rain_indicator (5 dynamic)
    # 2D: wl, rain, pond_depth, rain_indicator + RAIN_LAG_STEPS lagged rainfall (4 + N dynamic)
    # Note: rainfall uses linear scaling (no log transform)
    in_1d = fb.in_channels_1d
    in_2d = fb.in_channels_2d
    
    model = HeteroFloodGNN(
        in_channels_1d=in_1d,
        in_channels_2d=in_2d,
        hidden_channels=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        conv_type=cfg.CONV_TYPE,
        heads=cfg.ATTENTION_HEADS,
        use_fused_ops=cfg.USE_FUSED_LAYERNORM,
        edge_dim_1d=fb.edge_dim_1d,
        edge_dim_2d=fb.edge_dim_2d,
        temporal_bundle_k=getattr(cfg, 'TEMPORAL_BUNDLE_K', 1),
        use_node_embeddings=getattr(cfg, 'USE_NODE_EMBEDDINGS', False),
        num_1d_nodes=static_data.num_1d,
        num_2d_nodes=static_data.num_2d,
        node_embed_dim=getattr(cfg, 'NODE_EMBED_DIM', None),
    ).to(cfg.DEVICE)
    
    # Apply channels_last memory format if enabled (can improve perf on some GPUs)
    if cfg.USE_CHANNELS_LAST and torch.cuda.is_available():
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Applied channels_last memory format")
        except Exception as e:
            logger.warning(f"channels_last not supported: {e}")
    
    logger.info(f"Using {cfg.CONV_TYPE.upper()} convolutions with {cfg.ATTENTION_HEADS} heads")
    if getattr(cfg, 'USE_NODE_EMBEDDINGS', False):
        logger.info(
            f"Node ID embeddings enabled (dim={getattr(cfg, 'NODE_EMBED_DIM', None) or cfg.HIDDEN_DIM // 4})"
        )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _build_runtime_model(base_model: nn.Module, reason: str) -> nn.Module:
        '''Build runtime-wrapped model (torch.compile + optional DDP) from raw module.

        Important: optimizer state is kept untouched; this function only wraps the
        same underlying parameters for execution.
        '''
        runtime_model = base_model
        if cfg.USE_COMPILE and hasattr(torch, 'compile'):
            # Clear old Dynamo/Inductor graph cache before (re)compile, especially
            # when rollout length changes and execution traces differ.
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            try:
                runtime_model = torch.compile(runtime_model, mode='reduce-overhead', dynamic=False)
                logger.info(
                    f"torch.compile enabled (reduce-overhead, dynamic=False) [{reason}]"
                )
            except Exception as e:
                logger.warning(f"torch.compile failed [{reason}]: {e}")
        if get_world_size() > 1:
            runtime_model = DDP(
                runtime_model,
                device_ids=[cfg.DEVICE.index],
                output_device=cfg.DEVICE.index,
                find_unused_parameters=False,
                static_graph=True  # Cache DDP communication plan (graph structure is fixed)
            )
        return runtime_model

    model = _build_runtime_model(model, reason='initial')
    
    # Loss criterion with 1D upweighting (1D is the bottleneck)
    # Use Huber loss if enabled (better for heavy-tailed delta distributions)
    huber_delta_1d = cfg.HUBER_DELTA_1D if cfg.HUBER_DELTA_1D is not None else std_values.get('huber_delta_1d', 0.075)
    huber_delta_2d = cfg.HUBER_DELTA_2D if cfg.HUBER_DELTA_2D is not None else std_values.get('huber_delta_2d', 0.019)
    
    if cfg.USE_HUBER_LOSS and is_main_process():
        logger.info(f"Using Huber loss (SmoothL1) with thresholds:")
        logger.info(f"  1D: {huber_delta_1d:.4f}, 2D: {huber_delta_2d:.4f}")
    
    if cfg.USE_BIAS_LOSS and is_main_process():
        logger.info(f"Using bias loss penalty (weight={cfg.BIAS_LOSS_WEIGHT})")
    
    if getattr(cfg, 'NOISE_INJECTION', False) and is_main_process():
        logger.info(f"Noise injection enabled (1D std={cfg.NOISE_STD_1D:.4f}m, "
                     f"2D std={cfg.NOISE_STD_2D:.4f}m)")
    
    if wl_prev_steps > 0 and is_main_process():
        logger.info(f"WL_prev features: {wl_prev_steps} lag(s), "
                     f"dropout={getattr(cfg, 'WL_PREV_DROPOUT', 0.0):.2f}, "
                     f"wl_scale: 1D={std_values.get('wl_1d_scale', 100.0):.4f}, "
                     f"2D={std_values.get('wl_2d_scale', 100.0):.4f})")
    
    if use_edge_feat and is_main_process():
        logger.info(f"Edge features enabled (1D dim={fb.edge_dim_1d}, 2D dim={fb.edge_dim_2d})")
    
    criterion = FloodLoss(
        std_values['1d'], 
        std_values['2d'], 
        weight_1d=cfg.LOSS_WEIGHT_1D,
        use_huber=cfg.USE_HUBER_LOSS,
        huber_delta_1d=huber_delta_1d,
        huber_delta_2d=huber_delta_2d,
        use_bias_loss=cfg.USE_BIAS_LOSS,
        bias_loss_weight=cfg.BIAS_LOSS_WEIGHT
    )
    
    # Find optimal learning rate (or use config value)
    if cfg.FIND_LR:
        logger.info("=" * 40)
        optimal_lr = find_optimal_lr(model, train_loader, criterion, cfg.DEVICE)
        logger.info("=" * 40)
    else:
        optimal_lr = cfg.LEARNING_RATE
        logger.info(f"Using configured LR: {optimal_lr:.2e}")
    
    # Training setup - use fused optimizer with optimal LR
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=optimal_lr,  # Use found optimal LR
        weight_decay=cfg.WEIGHT_DECAY,
        fused=cfg.DEVICE.type == 'cuda'  # Fused optimizer for CUDA
    )
    
    # GradScaler for float16 mixed precision (prevents underflow)
    scaler = torch.amp.GradScaler('cuda') if cfg.USE_AMP else None
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    best_ar_val_loss = float('inf')
    patience_counter = 0
    
    checkpoint_path = cfg.OUTPUT_PATH / f"checkpoint_model_{model_id}.pt"
    if resume and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        
        # Handle torch.compile: strip '_orig_mod.' prefix if present
        state_dict = checkpoint['model_state_dict']
        is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        if is_compiled:
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # Load model (handle DDP + torch.compile)
        target_model = unwrap_model(model)
        if hasattr(target_model, '_orig_mod'):
            target_model._orig_mod.load_state_dict(state_dict)
        else:
            target_model.load_state_dict(state_dict)
        
        if resume_weights_only:
            # Fresh optimization phase: keep pretrained weights, reset optimizer/LR schedule.
            start_epoch = 0
            best_val_loss = float('inf')
            best_ar_val_loss = float('inf')
            patience_counter = 0
            logger.info("Loaded checkpoint weights only; optimizer/scheduler reset "
                        "(fresh LR and rollout schedule phase)")
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_ar_val_loss = checkpoint.get('best_ar_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            
            logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}, "
                       f"best_ar_val_loss: {best_ar_val_loss:.4f}")
    elif resume:
        logger.warning(f"No checkpoint found at {checkpoint_path}, starting fresh")
    
    # Warmup: test single batch to check for issues
    logger.info("Warmup: Testing single batch")
    import time
    warmup_batch = next(iter(train_loader))
    # Move dict batch to device
    warmup_batch = {k: v.to(cfg.DEVICE) for k, v in warmup_batch.items()}
    
    # Log batch info for sequence data
    logger.debug(f"Batch size: {warmup_batch['wl_1d_init'].shape[0]}")
    logger.debug(f"1D nodes: {warmup_batch['wl_1d_init'].shape[1]}")
    logger.debug(f"2D nodes: {warmup_batch['wl_2d_init'].shape[1]}")
    logger.debug(f"Rollout steps: {warmup_batch['rainfall_seq'].shape[1]}")
    
    # Check for NaN in inputs
    nan_1d = torch.isnan(warmup_batch['wl_1d_init']).sum().item()
    nan_2d = torch.isnan(warmup_batch['wl_2d_init']).sum().item()
    logger.debug(f"NaN check - 1D init: {nan_1d}, 2D init: {nan_2d}")
    
    if nan_1d > 0 or nan_2d > 0:
        logger.warning("NaN values detected in initial water levels!")
    
    # Log input ranges
    logger.debug(f"Init WL 1D range: [{warmup_batch['wl_1d_init'].min():.4f}, {warmup_batch['wl_1d_init'].max():.4f}]")
    logger.debug(f"Init WL 2D range: [{warmup_batch['wl_2d_init'].min():.4f}, {warmup_batch['wl_2d_init'].max():.4f}]")
    logger.debug(f"Target WL 1D range: [{warmup_batch['wl_1d_targets'].min():.4f}, {warmup_batch['wl_1d_targets'].max():.4f}]")
    logger.debug(f"Target WL 2D range: [{warmup_batch['wl_2d_targets'].min():.4f}, {warmup_batch['wl_2d_targets'].max():.4f}]")
    
    if cfg.DEVICE.type == 'cuda':
        logger.debug(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
    logger.info("Warmup complete")
    
    # Training loop (continue from start_epoch if resuming)
    prev_rollout = None  # Track for logging curriculum transitions
    prev_lr = None        # Track for logging LR transitions
    for epoch in range(start_epoch, cfg.EPOCHS):
        # --- Curriculum rollout: resolve current rollout length ---
        current_rollout = cfg.get_rollout_for_epoch(epoch)
        if current_rollout != prev_rollout:
            if prev_rollout is not None:
                logger.info(f"  ↑ Curriculum: rollout {prev_rollout} → {current_rollout} steps")
            # Rebuild dataset index and dataloader for the current rollout so
            # batch tensors are not over-allocated to max rollout.
            train_dataset.set_rollout_steps(current_rollout)
            if get_world_size() > 1:
                train_sampler = DistributedSampler(
                    train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True, drop_last=False
                )
            else:
                train_sampler = None
            train_loader = DataLoader(
                train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, **train_loader_kwargs
            )
            logger.info(f"  Rebuilt train loader for rollout={current_rollout} "
                        f"(samples={len(train_dataset)})")
            # Recompile model when rollout changes to avoid stale CUDA graph pool
            # state in torch.compile reduce-overhead mode.
            if prev_rollout is not None and cfg.USE_COMPILE and hasattr(torch, 'compile'):
                raw_model = unwrap_model(model)
                if hasattr(raw_model, '_orig_mod'):
                    raw_model = raw_model._orig_mod
                model = _build_runtime_model(raw_model, reason=f'rollout={current_rollout}')
            prev_rollout = current_rollout
        
        # --- Piecewise-constant LR: set LR for this epoch ---
        current_lr = cfg.get_lr_for_epoch(epoch)
        if current_lr != prev_lr:
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            if prev_lr is not None:
                logger.info(f"  ↑ LR schedule: {prev_lr:.2e} → {current_lr:.2e}")
            prev_lr = current_lr
        
        logger.info(f"Epoch {epoch+1}/{cfg.EPOCHS}  (rollout={current_rollout}, lr={current_lr:.2e})")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Use multi-step rollout training via FeatureBuilder.
        train_loss, train_1d, train_2d = train_epoch_multistep(
            model, train_loader, optimizer, criterion, scaler, cfg.DEVICE,
            fb=fb, rollout_steps=current_rollout, epoch_num=epoch, cfg=cfg
        )
        if is_main_process():
            val_loss, val_1d, val_2d = evaluate(model, val_loader, criterion, cfg.DEVICE)
        else:
            val_loss, val_1d, val_2d = 0.0, 0.0, 0.0

        if is_main_process():
            logger.info(
                f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} (1D: {train_1d:.4f}, 2D: {train_2d:.4f}) | "
                f"Val: {val_loss:.4f} (1D: {val_1d:.4f}, 2D: {val_2d:.4f}) | LR: {current_lr:.6f}"
            )
        
        # Autoregressive validation every N epochs (realistic error with compounding)
        if is_main_process() and ((epoch + 1) % cfg.AR_VAL_EVERY == 0 or epoch == 0):
            ar_loss, ar_1d, ar_2d = autoregressive_validation(
                model, static_data, val_events, model_path, 
                std_values['1d'], std_values['2d'], cfg.DEVICE,
                feature_scales=std_values, fb=fb,
                plot_random_events=3,
                plot_output_dir=cfg.OUTPUT_PATH / "ar_event_timeseries",
                plot_tag=f"epoch_{epoch + 1:03d}",
                plot_seed=42 + epoch,
            )
            logger.info(f"AR Val (rollout): {ar_loss:.4f} (1D: {ar_1d:.4f}, 2D: {ar_2d:.4f})")
            
            # Save best model based on AR validation score
            if ar_loss < best_ar_val_loss:
                best_ar_val_loss = ar_loss
                torch.save({
                    'model_state_dict': unwrap_model(model).state_dict(),
                    'std_values': std_values,
                    'config': {
                        'in_1d': in_1d, 'in_2d': in_2d,
                        'hidden': cfg.HIDDEN_DIM, 'layers': cfg.NUM_LAYERS,
                        'conv_type': cfg.CONV_TYPE, 'heads': cfg.ATTENTION_HEADS,
                        'use_fused_ops': cfg.USE_FUSED_LAYERNORM,
                        'rain_lag_steps': cfg.RAIN_LAG_STEPS,
                        'rain_future_steps': getattr(cfg, 'RAIN_FUTURE_STEPS', 0),
                        'edge_dim_1d': fb.edge_dim_1d, 'edge_dim_2d': fb.edge_dim_2d,
                        'wl_prev_steps': fb.wl_prev_steps,
                        'temporal_bundle_k': getattr(cfg, 'TEMPORAL_BUNDLE_K', 1),
                        'use_node_embeddings': getattr(cfg, 'USE_NODE_EMBEDDINGS', False),
                        'node_embed_dim': getattr(cfg, 'NODE_EMBED_DIM', None),
                        'num_1d_nodes': static_data.num_1d,
                        'num_2d_nodes': static_data.num_2d,
                    },
                    'ar_val_loss': ar_loss,
                    'ar_val_1d': ar_1d,
                    'ar_val_2d': ar_2d,
                }, cfg.OUTPUT_PATH / f"best_model_{model_id}_ar.pt")
                logger.info(f"Saved best AR model (ar_loss: {ar_loss:.4f})")
        
        # Early stopping based on step-wise validation
        if is_main_process():
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': unwrap_model(model).state_dict(),
                    'std_values': std_values,
                    'config': {
                        'in_1d': in_1d, 'in_2d': in_2d,
                        'hidden': cfg.HIDDEN_DIM, 'layers': cfg.NUM_LAYERS,
                        'conv_type': cfg.CONV_TYPE, 'heads': cfg.ATTENTION_HEADS,
                        'use_fused_ops': cfg.USE_FUSED_LAYERNORM,
                        'rain_lag_steps': cfg.RAIN_LAG_STEPS,
                        'rain_future_steps': getattr(cfg, 'RAIN_FUTURE_STEPS', 0),
                        'edge_dim_1d': fb.edge_dim_1d, 'edge_dim_2d': fb.edge_dim_2d,
                        'wl_prev_steps': fb.wl_prev_steps,
                        'temporal_bundle_k': getattr(cfg, 'TEMPORAL_BUNDLE_K', 1),
                        'use_node_embeddings': getattr(cfg, 'USE_NODE_EMBEDDINGS', False),
                        'node_embed_dim': getattr(cfg, 'NODE_EMBED_DIM', None),
                        'num_1d_nodes': static_data.num_1d,
                        'num_2d_nodes': static_data.num_2d,
                    }
                }, cfg.OUTPUT_PATH / f"best_model_{model_id}.pt")
                logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if get_world_size() == 1 and patience_counter >= cfg.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save checkpoint for resume (every epoch)
        if is_main_process():
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_ar_val_loss': best_ar_val_loss,
                'patience_counter': patience_counter,
                'std_values': std_values,
                'config': {
                    'in_1d': in_1d, 'in_2d': in_2d,
                    'hidden': cfg.HIDDEN_DIM, 'layers': cfg.NUM_LAYERS,
                    'conv_type': cfg.CONV_TYPE, 'heads': cfg.ATTENTION_HEADS,
                    'use_fused_ops': cfg.USE_FUSED_LAYERNORM,
                    'rain_lag_steps': cfg.RAIN_LAG_STEPS,
                    'rain_future_steps': getattr(cfg, 'RAIN_FUTURE_STEPS', 0),
                    'edge_dim_1d': fb.edge_dim_1d, 'edge_dim_2d': fb.edge_dim_2d,
                    'wl_prev_steps': fb.wl_prev_steps,
                    'temporal_bundle_k': getattr(cfg, 'TEMPORAL_BUNDLE_K', 1),
                    'use_node_embeddings': getattr(cfg, 'USE_NODE_EMBEDDINGS', False),
                    'node_embed_dim': getattr(cfg, 'NODE_EMBED_DIM', None),
                    'num_1d_nodes': static_data.num_1d,
                    'num_2d_nodes': static_data.num_2d,
                }
            }
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_data, cfg.OUTPUT_PATH / f"checkpoint_model_{model_id}.pt")
    
    # Analyze feature importance after training
    if is_main_process():
        logger.info("Analyzing feature importance...")
        try:
            feature_importance = analyze_feature_importance(
                model, val_loader, criterion, cfg.DEVICE, method="gradient"
            )
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            feature_importance = None

        # Run Physics-Consistent Mass Balance Diagnostic
        logger.info("Running Mass Balance Diagnostic...")
        try:
            mass_balance_results = mass_balance_diagnostic(
                model, static_data, val_events, model_path, cfg.DEVICE,
                feature_scales=std_values, fb=fb
            )
            # Save results
            if 'results_df' in mass_balance_results:
                mass_balance_results['results_df'].to_csv(
                    cfg.OUTPUT_PATH / f"mass_balance_model_{model_id}.csv", index=False
                )
                logger.info(f"Saved mass balance results to mass_balance_model_{model_id}.csv")
        except Exception as e:
            logger.warning(f"Mass balance diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
            mass_balance_results = None
    
    return model, std_values, static_data


# =============================================================================
# INFERENCE (AUTOREGRESSIVE)
# =============================================================================

def load_model_from_checkpoint(
    model_file: Path,
    device: torch.device,
) -> Tuple['HeteroFloodGNN', Dict, Dict]:
    '''Load a HeteroFloodGNN from a saved checkpoint file.

    Handles torch.compile state-dict prefix stripping automatically.

    Args:
        model_file: Path to .pt checkpoint.
        device: Target device.

    Returns:
        model: HeteroFloodGNN in eval mode, moved to device.
        model_cfg: Raw config dict from checkpoint.
        std_values: Feature normalization scales from checkpoint (may be empty dict).

    Raises:
        FileNotFoundError: If model_file does not exist.
    '''
    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model_cfg = checkpoint['config']
    std_values = checkpoint.get('std_values') or {}

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = HeteroFloodGNN(
        in_channels_1d=model_cfg['in_1d'],
        in_channels_2d=model_cfg['in_2d'],
        hidden_channels=model_cfg['hidden'],
        num_layers=model_cfg['layers'],
        conv_type=model_cfg.get('conv_type', 'gatv2'),
        heads=model_cfg.get('heads', 4),
        use_fused_ops=model_cfg.get('use_fused_ops', True),
        edge_dim_1d=model_cfg.get('edge_dim_1d', 0),
        edge_dim_2d=model_cfg.get('edge_dim_2d', 0),
        temporal_bundle_k=model_cfg.get('temporal_bundle_k', 1),
        use_node_embeddings=model_cfg.get('use_node_embeddings', False),
        num_1d_nodes=model_cfg.get('num_1d_nodes'),
        num_2d_nodes=model_cfg.get('num_2d_nodes'),
        node_embed_dim=model_cfg.get('node_embed_dim'),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_cfg, std_values

@torch.no_grad()
def autoregressive_inference(model, static_data: StaticGraphData, event_path: Path, 
                             device: torch.device,
                             feature_scales: Dict[str, float] = None,
                             fb: 'FeatureBuilder' = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Autoregressive rollout for test event.
    Uses first 10 timesteps as warmup, then predicts remaining.
    '''
    model.eval()
    
    # Build FeatureBuilder if not provided
    if fb is None:
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                            rain_lag_steps=Config.RAIN_LAG_STEPS,
                            rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
    fb.to(device)
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    
    # Load dynamic data using vectorized pivot
    df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
    df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
    timesteps = pd.read_csv(event_path / 'timesteps.csv')
    
    n_timesteps = len(timesteps)
    
    wl_1d_all = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values
    wl_2d_all = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values
    rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
    if 'rainfall' in df_2d.columns:
        rainfall_all[:len(df_2d.pivot(index='timestep', columns='node_idx', values='rainfall'))] = \
            df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0).values
    
    use_amp = Config.USE_AMP and device.type == 'cuda'
    return run_ar_rollout(
        model, fb, wl_1d_all, wl_2d_all, rainfall_all, device, use_amp=use_amp)


def generate_submission(output_path: Path = Config.OUTPUT_PATH, use_ar_model: bool = True):
    '''
    Generate submission file for all test events.
    
    Args:
        output_path: Directory containing model checkpoints
        use_ar_model: If True, use AR-validated models (best_model_X_ar.pt), 
                      otherwise use step-validated models (best_model_X.pt)
    '''
    
    submissions = []
    
    model_suffix = "_ar" if use_ar_model else ""
    logger.info(f"Using {'AR-validated' if use_ar_model else 'step-validated'} models")
    
    for model_id in [1, 2]:
        logger.info(f"Processing Model {model_id}")
        
        # Load model
        model_file = output_path / f"best_model_{model_id}{model_suffix}.pt"
        if not model_file.exists() and use_ar_model:
            logger.warning(f"AR model not found, falling back to step-validated model")
            model_file = output_path / f"best_model_{model_id}.pt"
        model, cfg, feature_scales = load_model_from_checkpoint(model_file, Config.DEVICE)
        if 'wl_1d_scale' in feature_scales:
            logger.info(f"Using saved feature scales for Model {model_id}")
        else:
            logger.warning(f"No feature scales found in checkpoint, using defaults")
            feature_scales = None
        
        # Load static data and build FeatureBuilder
        test_path = Config.BASE_PATH / f"Model_{model_id}" / "test"
        static_data = StaticGraphData(test_path)
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                        rain_lag_steps=cfg.get('rain_lag_steps', Config.RAIN_LAG_STEPS),
                        rain_future_steps=cfg.get('rain_future_steps', 0),
                        wl_prev_steps=FeatureBuilder.infer_wl_prev_steps(cfg, static_data),
                        use_edge_features=cfg.get('edge_dim_1d', 0) > 0 or cfg.get('edge_dim_2d', 0) > 0)
        
        # Get test events
        events = sorted([d.name for d in test_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
        logger.info(f"Test events: {len(events)}")
        
        for event in tqdm(events, desc=f"Model {model_id}"):
            event_id = int(event.split('_')[1])
            event_path = test_path / event
            
            # Rollout via FeatureBuilder
            wl_1d_pred, wl_2d_pred = autoregressive_inference(
                model, static_data, event_path, Config.DEVICE,
                feature_scales=feature_scales, fb=fb
            )
            
            # Skip warmup timesteps for submission
            timesteps_df = pd.read_csv(event_path / 'timesteps.csv')
            n_pred_timesteps = len(timesteps_df) - Config.WARMUP_TIMESTEPS
            n_1d = static_data.num_1d
            n_2d = static_data.num_2d
            
            # VECTORIZED submission building (100x faster than nested loops)
            # Create arrays for 1D nodes
            timesteps_1d = np.repeat(np.arange(Config.WARMUP_TIMESTEPS, len(timesteps_df)), n_1d)
            node_ids_1d = np.tile(np.arange(n_1d), n_pred_timesteps)
            water_levels_1d = wl_1d_pred[Config.WARMUP_TIMESTEPS:].flatten()
            
            df_1d = pd.DataFrame({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': 1,
                'node_id': node_ids_1d,
                'timestep': timesteps_1d,
                'water_level': water_levels_1d
            })
            
            # Create arrays for 2D nodes
            timesteps_2d = np.repeat(np.arange(Config.WARMUP_TIMESTEPS, len(timesteps_df)), n_2d)
            node_ids_2d = np.tile(np.arange(n_2d), n_pred_timesteps)
            water_levels_2d = wl_2d_pred[Config.WARMUP_TIMESTEPS:].flatten()
            
            df_2d = pd.DataFrame({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': 2,
                'node_id': node_ids_2d,
                'timestep': timesteps_2d,
                'water_level': water_levels_2d
            })
            
            submissions.append(df_1d)
            submissions.append(df_2d)
    
    # Concatenate all DataFrames and sort to match expected submission format
    df = pd.concat(submissions, ignore_index=True)
    df = df.sort_values(
        ['model_id', 'event_id', 'node_type', 'node_id', 'timestep']
    ).reset_index(drop=True)
    
    # Add row_id AFTER sorting to match sample_submission
    df['row_id'] = range(len(df))
    
    # Reorder columns and drop timestep (not in final submission)
    df = df[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']]
    
    # Save
    df.to_csv(output_path / 'submission.csv', index=False)
    df.to_parquet(output_path / 'submission.parquet', index=False)
    logger.info(f"Saved submission with {len(df):,} rows")
    
    return df


# =============================================================================
# STANDALONE MASS BALANCE DIAGNOSTIC
# =============================================================================

def run_ar_validation_standalone(model_id: int, use_ar_model: bool = True):
    '''
    Run autoregressive validation on a trained model.
    
    Args:
        model_id: Model to evaluate (1 or 2)
        use_ar_model: If True, use AR-validated model, else use step-validated
    '''
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    
    logger.info("=" * 60)
    logger.info(f"Running AR Validation for Model {model_id}")
    logger.info("=" * 60)
    
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    # Load model
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists() and use_ar_model:
        logger.warning(f"AR model not found, falling back to step-validated model")
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    logger.info(f"Loaded model from {model_file}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load static data and build FeatureBuilder
    static_data = StaticGraphData(model_path)
    logger.info(f"1D nodes: {static_data.num_1d}, 2D nodes: {static_data.num_2d}")
    fb = FeatureBuilder(static_data, feature_scales=std_values,
                        rain_lag_steps=model_cfg.get('rain_lag_steps', Config.RAIN_LAG_STEPS),
                        rain_future_steps=model_cfg.get('rain_future_steps', 0),
                        wl_prev_steps=FeatureBuilder.infer_wl_prev_steps(model_cfg, static_data),
                        use_edge_features=model_cfg.get('edge_dim_1d', 0) > 0 or model_cfg.get('edge_dim_2d', 0) > 0)
    
    # Get validation events (last 20% of training events)
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    
    logger.info(f"Evaluating on {len(val_events)} validation events: {val_events}")
    
    # Run AR validation with verbose diagnostics
    ar_loss, ar_1d, ar_2d = autoregressive_validation(
        model, static_data, val_events, model_path,
        std_values['1d'], std_values['2d'], cfg.DEVICE,
        verbose=True, feature_scales=std_values, fb=fb
    )
    
    logger.info("=" * 60)
    logger.info("AR VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Score: {ar_loss:.4f}")
    logger.info(f"  1D RMSE:     {ar_1d:.4f}")
    logger.info(f"  2D RMSE:     {ar_2d:.4f}")
    logger.info("=" * 60)
    
    return {'total': ar_loss, '1d': ar_1d, '2d': ar_2d}


def run_mass_balance_diagnostic_standalone(model_id: int, use_ar_model: bool = True):
    '''
    Run mass balance diagnostic on a trained model.
    
    Args:
        model_id: Model to evaluate (1 or 2)
        use_ar_model: If True, use AR-validated model, else use step-validated
    '''
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    
    logger.info("=" * 60)
    logger.info(f"Running Mass Balance Diagnostic for Model {model_id}")
    logger.info("=" * 60)
    
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    # Load model
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists() and use_ar_model:
        logger.warning(f"AR model not found, falling back to step-validated model")
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    
    # Load static data and build FeatureBuilder
    static_data = StaticGraphData(model_path)
    fb = FeatureBuilder(static_data, feature_scales=std_values,
                        rain_lag_steps=model_cfg.get('rain_lag_steps', Config.RAIN_LAG_STEPS),
                        rain_future_steps=model_cfg.get('rain_future_steps', 0),
                        wl_prev_steps=FeatureBuilder.infer_wl_prev_steps(model_cfg, static_data),
                        use_edge_features=model_cfg.get('edge_dim_1d', 0) > 0 or model_cfg.get('edge_dim_2d', 0) > 0)
    
    # Get validation events (last 20% of training events)
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    
    logger.info(f"Evaluating on {len(val_events)} validation events")
    
    # Run diagnostic
    results = mass_balance_diagnostic(
        model, static_data, val_events, model_path, cfg.DEVICE,
        feature_scales=std_values, fb=fb
    )
    
    # Save results
    if 'results_df' in results:
        output_file = cfg.OUTPUT_PATH / f"mass_balance_model_{model_id}.csv"
        results['results_df'].to_csv(output_file, index=False)
        logger.info(f"Saved detailed results to {output_file}")
    
    return results


# =============================================================================
# TEMPORAL DYNAMICS ANALYSIS
# =============================================================================

@torch.no_grad()
def spectral_fidelity_analysis(model, static_data: StaticGraphData, val_events: List[str],
                                model_path: Path, device: torch.device,
                                feature_scales: Dict[str, float] = None,
                                node_sample_1d: int = 20, node_sample_2d: int = 100,
                                dt: float = 300.0,
                                fb: 'FeatureBuilder' = None) -> Dict:
    '''
    Spectral Fidelity Analysis: compare power spectral density of predicted vs GT
    water level time series.
    
    Detects whether the model acts as a low-pass filter (smooths out fast transients)
    or reproduces the full frequency content of the dynamics.
    
    Reference: WeatherBench2 (Rasp et al. 2024), GenCast (Price et al. 2024).
    
    Args:
        model: Trained GNN model
        static_data: Static graph structure
        val_events: List of validation event names
        model_path: Path to model data
        device: Torch device
        feature_scales: Feature normalization scales
        node_sample_1d: Number of 1D nodes to sample for analysis
        node_sample_2d: Number of 2D nodes to sample for analysis
        dt: Timestep in seconds (default 300s = 5min)
    
    Returns:
        Dict with spectral fidelity metrics
    '''
    from scipy import signal as scipy_signal
    
    model.eval()
    fs_hz = 1.0 / dt  # Sampling frequency (Hz) - renamed to avoid shadowing FeatureScales
    
    # Build FeatureBuilder if not provided
    if fb is None:
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                            rain_lag_steps=Config.RAIN_LAG_STEPS,
                            rain_future_steps=getattr(Config, 'RAIN_FUTURE_STEPS', 0))
    fb.to(device)
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    
    # Accumulators across events
    all_psd_ratios_1d = []
    all_psd_ratios_2d = []
    all_coherences_1d = []
    all_coherences_2d = []
    ref_freqs = None
    
    use_amp = Config.USE_AMP and device.type == 'cuda'
    rain_lag_steps = fb.rain_lag_steps
    
    for event_name in tqdm(val_events, desc="Spectral Analysis"):
        event_path = model_path / event_name
        
        # Load dynamic data
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        n_timesteps = int(df_1d['timestep'].max() + 1)
        
        wl_1d_gt = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        wl_2d_gt = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
        if 'rainfall' in df_2d.columns:
            pivot_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0)
            rainfall_all[:len(pivot_rain)] = pivot_rain.values
        
        # --- AR rollout ---
        wl_1d_pred, wl_2d_pred = run_ar_rollout(
            model, fb, wl_1d_gt, wl_2d_gt, rainfall_all, device, use_amp=use_amp)
        
        # --- Spectral analysis on this event ---
        # Use post-warmup data only
        start = Config.WARMUP_TIMESTEPS
        gt_1d = wl_1d_gt[start:]
        pr_1d = wl_1d_pred[start:]
        gt_2d = wl_2d_gt[start:]
        pr_2d = wl_2d_pred[start:]
        n_post = gt_1d.shape[0]
        # Fixed nperseg ensures consistent frequency bins across all events
        nperseg = 128
        if n_post < nperseg * 2:
            continue
        
        # Sample nodes (pick those with highest variance for informative spectra)
        var_1d = np.var(gt_1d, axis=0)
        var_2d = np.var(gt_2d, axis=0)
        idx_1d = np.argsort(var_1d)[-min(node_sample_1d, n_1d):]
        idx_2d = np.argsort(var_2d)[-min(node_sample_2d, n_2d):]
        
        for ni in idx_1d:
            if np.std(gt_1d[:, ni]) < 1e-6:
                continue
            freqs, psd_gt = scipy_signal.welch(gt_1d[:, ni], fs=fs_hz, nperseg=nperseg)
            _, psd_pr = scipy_signal.welch(pr_1d[:, ni], fs=fs_hz, nperseg=nperseg)
            ratio = psd_pr / np.maximum(psd_gt, 1e-15)
            all_psd_ratios_1d.append(ratio)
            _, coh = scipy_signal.coherence(gt_1d[:, ni], pr_1d[:, ni], fs=fs_hz, nperseg=nperseg)
            all_coherences_1d.append(coh)
            if ref_freqs is None:
                ref_freqs = freqs
        
        for ni in idx_2d:
            if np.std(gt_2d[:, ni]) < 1e-6:
                continue
            freqs, psd_gt = scipy_signal.welch(gt_2d[:, ni], fs=fs_hz, nperseg=nperseg)
            _, psd_pr = scipy_signal.welch(pr_2d[:, ni], fs=fs_hz, nperseg=nperseg)
            ratio = psd_pr / np.maximum(psd_gt, 1e-15)
            all_psd_ratios_2d.append(ratio)
            _, coh = scipy_signal.coherence(gt_2d[:, ni], pr_2d[:, ni], fs=fs_hz, nperseg=nperseg)
            all_coherences_2d.append(coh)
    
    # Aggregate results
    if ref_freqs is None or len(all_psd_ratios_1d) == 0:
        logger.warning("Not enough data for spectral analysis")
        return {}
    
    psd_ratios_1d = np.array(all_psd_ratios_1d)
    psd_ratios_2d = np.array(all_psd_ratios_2d)
    coherences_1d = np.array(all_coherences_1d)
    coherences_2d = np.array(all_coherences_2d)
    
    # Skip DC component (index 0) for ratio/coherence summaries
    freqs_nz = ref_freqs[1:]
    periods_min = 1.0 / (freqs_nz * 60 + 1e-15)
    
    median_ratio_1d = np.median(psd_ratios_1d[:, 1:], axis=0)
    median_ratio_2d = np.median(psd_ratios_2d[:, 1:], axis=0)
    mean_coh_1d = np.mean(coherences_1d[:, 1:], axis=0)
    mean_coh_2d = np.mean(coherences_2d[:, 1:], axis=0)
    
    n_freqs = len(median_ratio_1d)
    
    # High-frequency damping: ratio at highest quarter of frequencies
    hf_quarter = max(1, 3 * n_freqs // 4)
    hf_damping_1d = float(np.mean(median_ratio_1d[hf_quarter:]))
    hf_damping_2d = float(np.mean(median_ratio_2d[hf_quarter:]))
    
    # Spectral rolloff: period where coherence drops below 0.5
    rolloff_idx_1d = np.where(mean_coh_1d < 0.5)[0]
    rolloff_period_1d = float(periods_min[rolloff_idx_1d[0]]) if len(rolloff_idx_1d) > 0 else float(periods_min[-1])
    rolloff_idx_2d = np.where(mean_coh_2d < 0.5)[0]
    rolloff_period_2d = float(periods_min[rolloff_idx_2d[0]]) if len(rolloff_idx_2d) > 0 else float(periods_min[-1])
    
    # Overall spectral score: mean coherence across all frequencies
    spectral_score_1d = float(np.mean(mean_coh_1d))
    spectral_score_2d = float(np.mean(mean_coh_2d))
    
    results = {
        'freqs': freqs_nz,
        'periods_min': periods_min,
        'psd_ratio_1d': median_ratio_1d,
        'psd_ratio_2d': median_ratio_2d,
        'coherence_1d': mean_coh_1d,
        'coherence_2d': mean_coh_2d,
        'hf_damping_1d': hf_damping_1d,
        'hf_damping_2d': hf_damping_2d,
        'rolloff_period_1d_min': rolloff_period_1d,
        'rolloff_period_2d_min': rolloff_period_2d,
        'spectral_score_1d': spectral_score_1d,
        'spectral_score_2d': spectral_score_2d,
    }
    
    # --- Log ---
    logger.info("=" * 60)
    logger.info("SPECTRAL FIDELITY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Analysed {len(all_psd_ratios_1d)} 1D node-event pairs, "
                f"{len(all_psd_ratios_2d)} 2D node-event pairs")
    logger.info("")
    logger.info("Overall Spectral Scores (mean coherence, 1.0 = perfect):")
    logger.info(f"  1D: {spectral_score_1d:.4f}")
    logger.info(f"  2D: {spectral_score_2d:.4f}")
    logger.info("")
    logger.info("High-Frequency Damping (PSD ratio at top-quarter freqs, 1.0 = faithful):")
    logger.info(f"  1D: {hf_damping_1d:.4f}")
    logger.info(f"  2D: {hf_damping_2d:.4f}")
    if hf_damping_1d < 0.5:
        logger.warning("  ⚠ 1D high-frequency smoothing detected — fast transients are damped")
    if hf_damping_2d < 0.5:
        logger.warning("  ⚠ 2D high-frequency smoothing detected — fast transients are damped")
    logger.info("")
    logger.info("Spectral Rolloff (period below which coherence < 0.5):")
    logger.info(f"  1D: {rolloff_period_1d:.1f} min")
    logger.info(f"  2D: {rolloff_period_2d:.1f} min")
    if rolloff_period_1d > 30:
        logger.warning("  ⚠ 1D model only accurate at timescales > 30 min")
    if rolloff_period_2d > 30:
        logger.warning("  ⚠ 2D model only accurate at timescales > 30 min")
    logger.info("")
    
    # Per-frequency-band breakdown
    logger.info("Coherence by frequency band:")
    band_edges = [0, 1/3600, 1/1800, 1/900, 1/600, 1/300, float('inf')]
    band_names = ['>60min', '30-60min', '15-30min', '10-15min', '5-10min', '<5min']
    for b in range(len(band_names)):
        mask = (freqs_nz >= band_edges[b]) & (freqs_nz < band_edges[b + 1])
        if mask.any():
            c1d = float(np.mean(mean_coh_1d[mask]))
            c2d = float(np.mean(mean_coh_2d[mask]))
            r1d = float(np.mean(median_ratio_1d[mask]))
            r2d = float(np.mean(median_ratio_2d[mask]))
            logger.info(f"  {band_names[b]:>10s}:  1D coh={c1d:.3f} ratio={r1d:.3f} | "
                        f"2D coh={c2d:.3f} ratio={r2d:.3f}")
    
    logger.info("=" * 60)
    
    return results


@torch.no_grad()
def jacobian_eigenvalue_analysis(model, static_data: StaticGraphData, val_events: List[str],
                                  model_path: Path, device: torch.device,
                                  feature_scales: Dict[str, float] = None,
                                  n_probes: int = 64, n_sample_timesteps: int = 10,
                                  fb: 'FeatureBuilder' = None) -> Dict:
    '''
    Jacobian Eigenvalue Analysis of the Learned Dynamics Operator.
    
    Estimates the spectral radius of J = ∂(Δwl)/∂(wl) — the Jacobian of the
    model's single-step dynamics w.r.t. the current water level state.
    
    - spectral_radius > 1  → errors amplify exponentially (unstable AR rollout)
    - spectral_radius ≈ 1  → marginally stable (ideal for conservation)
    - spectral_radius < 0.5 → over-damped (model smooths everything)
    
    Uses randomised SVD (Halko et al. 2011) for efficiency on large systems.
    
    Reference: Brandstetter et al. (2022) "Message Passing Neural PDE Solvers",
               Sanchez-Gonzalez et al. (2020) "Learning to Simulate".
    
    Args:
        model: Trained GNN model
        static_data: Static graph structure
        val_events: List of validation event names
        model_path: Path to model data
        device: Torch device
        feature_scales: Feature normalization scales
        n_probes: Number of random probes for Jacobian estimation (higher = more accurate)
        n_sample_timesteps: Number of timesteps per event to sample
    
    Returns:
        Dict with Jacobian analysis metrics
    '''
    model.eval()
    
    # Build FeatureBuilder if not provided
    if fb is None:
        fb = FeatureBuilder(static_data, feature_scales=feature_scales,
                            rain_lag_steps=Config.RAIN_LAG_STEPS)
    fb.to(device)
    n_1d = fb.n_1d
    n_2d = fb.n_2d
    rain_lag_steps = fb.rain_lag_steps
    
    use_amp = Config.USE_AMP and device.type == 'cuda'
    eps = 1e-3  # Finite difference step
    
    # Accumulators
    all_spectral_radii_1d = []
    all_spectral_radii_2d = []
    all_singular_values_1d = []
    all_singular_values_2d = []
    
    # _build_features captures cum_rain_t and normalized_t from enclosing scope (set per-timestep)
    _jac_cum_rain = [None]  # mutable container for closure
    _jac_norm_t = [0.0]
    
    def _build_features(wl_1d_vals, wl_2d_vals, rainfall_t, rain_lag_t):
        '''Build full feature vectors — delegates to FeatureBuilder.'''
        return fb.build_x_dict_torch(
            wl_1d_vals, wl_2d_vals, rainfall_t, rain_lag_t,
            cum_rainfall_2d=_jac_cum_rain[0], normalized_t=_jac_norm_t[0])
    
    def _model_forward(x_dict):
        '''Run model forward pass.'''
        if use_amp:
            with torch.amp.autocast('cuda'):
                out = model(x_dict, fb.edge_index_dict, fb.edge_attr_dict)
        else:
            out = model(x_dict, fb.edge_index_dict, fb.edge_attr_dict)
        return out['1d'].float(), out['2d'].float()
    
    for event_name in tqdm(val_events, desc="Jacobian Analysis"):
        event_path = model_path / event_name
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        n_timesteps = int(df_1d['timestep'].max() + 1)
        
        wl_1d_gt = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        wl_2d_gt = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
        if 'rainfall' in df_2d.columns:
            pivot_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0)
            rainfall_all[:len(pivot_rain)] = pivot_rain.values
        rainfall_all_gpu = torch.tensor(rainfall_all, dtype=torch.float32, device=device)
        cum_rain_gpu = torch.cumsum(rainfall_all_gpu, dim=0)
        
        # Sample timesteps (spread across the event, skip warmup)
        valid_range = list(range(Config.WARMUP_TIMESTEPS, n_timesteps - 1))
        if len(valid_range) == 0:
            continue
        step = max(1, len(valid_range) // n_sample_timesteps)
        sample_ts = valid_range[::step][:n_sample_timesteps]
        
        for t in sample_ts:
            # Build rain lags for this timestep
            rain_lag_t = fb.compute_rain_lags_torch(rainfall_all_gpu, t)
            
            # Set cum_rain and normalized_t for _build_features closure
            _jac_cum_rain[0] = cum_rain_gpu[t]
            _jac_norm_t[0] = t / max(n_timesteps - 1, 1)
            
            wl_1d_t = torch.tensor(wl_1d_gt[t], dtype=torch.float32, device=device)
            wl_2d_t = torch.tensor(wl_2d_gt[t], dtype=torch.float32, device=device)
            rainfall_t = rainfall_all_gpu[t]
            
            # Base prediction
            x_base = _build_features(wl_1d_t, wl_2d_t, rainfall_t, rain_lag_t)
            base_1d, base_2d = _model_forward(x_base)
            
            # --- Estimate Jacobian for 1D nodes via randomised probing ---
            n_probes_1d = min(n_probes, n_1d)
            Omega_1d = torch.randn(n_1d, n_probes_1d, device=device) / np.sqrt(n_probes_1d)
            JOmega_1d = torch.zeros(n_1d, n_probes_1d, device=device)
            
            for k in range(n_probes_1d):
                wl_1d_pert = wl_1d_t + eps * Omega_1d[:, k]
                x_pert = _build_features(wl_1d_pert, wl_2d_t, rainfall_t, rain_lag_t)
                pert_1d, _ = _model_forward(x_pert)
                JOmega_1d[:, k] = (pert_1d - base_1d) / eps
            
            # Randomised SVD via QR
            try:
                Q1, _ = torch.linalg.qr(JOmega_1d)
                # Re-probe with Q columns for better estimate
                QtJ_1d = torch.zeros(n_1d, Q1.shape[1], device=device)
                for k in range(Q1.shape[1]):
                    wl_1d_pert = wl_1d_t + eps * Q1[:, k]
                    x_pert = _build_features(wl_1d_pert, wl_2d_t, rainfall_t, rain_lag_t)
                    pert_1d, _ = _model_forward(x_pert)
                    QtJ_1d[:, k] = (pert_1d - base_1d) / eps
                
                _, sv_1d, _ = torch.linalg.svd(QtJ_1d, full_matrices=False)
                all_spectral_radii_1d.append(sv_1d[0].item())
                all_singular_values_1d.append(sv_1d.cpu().numpy())
            except Exception:
                pass  # Skip on numerical issues
            
            # --- Estimate Jacobian for 2D nodes via randomised probing ---
            n_probes_2d = min(n_probes, n_2d)
            Omega_2d = torch.randn(n_2d, n_probes_2d, device=device) / np.sqrt(n_probes_2d)
            JOmega_2d = torch.zeros(n_2d, n_probes_2d, device=device)
            
            for k in range(n_probes_2d):
                wl_2d_pert = wl_2d_t + eps * Omega_2d[:, k]
                x_pert = _build_features(wl_1d_t, wl_2d_pert, rainfall_t, rain_lag_t)
                _, pert_2d = _model_forward(x_pert)
                JOmega_2d[:, k] = (pert_2d - base_2d) / eps
            
            try:
                Q2, _ = torch.linalg.qr(JOmega_2d)
                QtJ_2d = torch.zeros(n_2d, Q2.shape[1], device=device)
                for k in range(Q2.shape[1]):
                    wl_2d_pert = wl_2d_t + eps * Q2[:, k]
                    x_pert = _build_features(wl_1d_t, wl_2d_pert, rainfall_t, rain_lag_t)
                    _, pert_2d = _model_forward(x_pert)
                    QtJ_2d[:, k] = (pert_2d - base_2d) / eps
                
                _, sv_2d, _ = torch.linalg.svd(QtJ_2d, full_matrices=False)
                all_spectral_radii_2d.append(sv_2d[0].item())
                all_singular_values_2d.append(sv_2d.cpu().numpy())
            except Exception:
                pass
    
    if len(all_spectral_radii_1d) == 0:
        logger.warning("No valid Jacobian samples collected")
        return {}
    
    sr_1d = np.array(all_spectral_radii_1d)
    sr_2d = np.array(all_spectral_radii_2d)
    
    # Effective rank: number of significant singular values (> 1% of max)
    def _effective_rank(sv_list):
        ranks = []
        for sv in sv_list:
            ranks.append(np.sum(sv > 0.01 * sv[0]))
        return float(np.mean(ranks))
    
    eff_rank_1d = _effective_rank(all_singular_values_1d)
    eff_rank_2d = _effective_rank(all_singular_values_2d)
    
    # Singular value decay rate (log-linear slope)
    def _sv_decay_rate(sv_list):
        rates = []
        for sv in sv_list:
            sv_safe = np.maximum(sv, 1e-12)
            if len(sv_safe) > 2:
                idx = np.arange(len(sv_safe))
                coeffs = np.polyfit(idx, np.log(sv_safe), 1)
                rates.append(coeffs[0])
        return float(np.mean(rates)) if rates else 0.0
    
    decay_1d = _sv_decay_rate(all_singular_values_1d)
    decay_2d = _sv_decay_rate(all_singular_values_2d)
    
    def _diagnosis(sr_median):
        if sr_median < 0.5:
            return 'OVER-DAMPED (smooths dynamics aggressively)'
        elif sr_median < 0.8:
            return 'STABLE (slightly over-damped)'
        elif sr_median < 1.0:
            return 'STABLE (well-tuned dynamics)'
        elif sr_median < 1.05:
            return 'MARGINALLY STABLE (monitor for drift)'
        else:
            return 'UNSTABLE (exponential error growth expected)'
    
    results = {
        'spectral_radius_1d_median': float(np.median(sr_1d)),
        'spectral_radius_1d_mean': float(np.mean(sr_1d)),
        'spectral_radius_1d_max': float(np.max(sr_1d)),
        'spectral_radius_1d_std': float(np.std(sr_1d)),
        'spectral_radius_2d_median': float(np.median(sr_2d)),
        'spectral_radius_2d_mean': float(np.mean(sr_2d)),
        'spectral_radius_2d_max': float(np.max(sr_2d)),
        'spectral_radius_2d_std': float(np.std(sr_2d)),
        'effective_rank_1d': eff_rank_1d,
        'effective_rank_2d': eff_rank_2d,
        'sv_decay_rate_1d': decay_1d,
        'sv_decay_rate_2d': decay_2d,
        'stable_1d': float(np.median(sr_1d)) < 1.0,
        'stable_2d': float(np.median(sr_2d)) < 1.0,
        'diagnosis_1d': _diagnosis(float(np.median(sr_1d))),
        'diagnosis_2d': _diagnosis(float(np.median(sr_2d))),
        'n_samples': len(sr_1d),
    }
    
    # --- Log ---
    logger.info("=" * 60)
    logger.info("JACOBIAN EIGENVALUE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Analysed {len(sr_1d)} state-timestep samples across {len(val_events)} events")
    logger.info(f"Probes per sample: 1D={min(n_probes, n_1d)}, 2D={min(n_probes, n_2d)}")
    logger.info("")
    
    logger.info("Spectral Radius ρ(J) = max singular value of ∂(Δwl)/∂(wl):")
    logger.info(f"  1D:  median={np.median(sr_1d):.4f}  mean={np.mean(sr_1d):.4f}  "
                f"max={np.max(sr_1d):.4f}  std={np.std(sr_1d):.4f}")
    logger.info(f"  2D:  median={np.median(sr_2d):.4f}  mean={np.mean(sr_2d):.4f}  "
                f"max={np.max(sr_2d):.4f}  std={np.std(sr_2d):.4f}")
    logger.info("")
    
    logger.info("Stability Assessment:")
    logger.info(f"  1D: {results['diagnosis_1d']}")
    logger.info(f"  2D: {results['diagnosis_2d']}")
    logger.info("")
    
    logger.info("Dynamics Complexity:")
    logger.info(f"  Effective rank: 1D={eff_rank_1d:.1f}, 2D={eff_rank_2d:.1f}")
    logger.info(f"  SV decay rate:  1D={decay_1d:.4f}, 2D={decay_2d:.4f}")
    logger.info(f"  (Faster decay → dynamics governed by fewer modes)")
    logger.info("")
    
    if np.median(sr_1d) > 1.05:
        logger.warning("  ⚠ 1D dynamics UNSTABLE — errors will grow exponentially during AR rollout")
        logger.warning("  → Consider: spectral normalization, smaller LR, more rollout training steps")
    if np.median(sr_2d) > 1.05:
        logger.warning("  ⚠ 2D dynamics UNSTABLE — errors will grow exponentially during AR rollout")
        logger.warning("  → Consider: spectral normalization, smaller LR, more rollout training steps")
    if np.median(sr_1d) < 0.5:
        logger.warning("  ⚠ 1D dynamics OVER-DAMPED — model smooths out rapid changes")
        logger.warning("  → Consider: higher learning rate, more expressive model, less weight decay")
    if np.median(sr_2d) < 0.5:
        logger.warning("  ⚠ 2D dynamics OVER-DAMPED — model smooths out rapid changes")
        logger.warning("  → Consider: higher learning rate, more expressive model, less weight decay")
    if np.max(sr_1d) > 1.5 and np.median(sr_1d) < 1.0:
        logger.warning("  ⚠ 1D has occasional unstable states (max ρ >> median ρ)")
        logger.warning("  → May cause sporadic AR divergence in certain flow regimes")
    if np.max(sr_2d) > 1.5 and np.median(sr_2d) < 1.0:
        logger.warning("  ⚠ 2D has occasional unstable states (max ρ >> median ρ)")
        logger.warning("  → May cause sporadic AR divergence in certain flow regimes")
    
    # Interpretation guide
    logger.info("")
    logger.info("Interpretation Guide:")
    logger.info("  ρ < 0.5  → Over-damped: fast dynamics smoothed out")
    logger.info("  ρ ≈ 0.8-1.0 → Ideal: stable AR rollout with faithful dynamics")
    logger.info("  ρ ≈ 1.0  → Marginal: errors neither grow nor decay (conservation)")
    logger.info("  ρ > 1.0  → Unstable: errors grow by ρ^T after T steps")
    logger.info(f"  Example: ρ=1.05 after 100 steps → error amplified {1.05**100:.1f}x")
    logger.info("=" * 60)
    
    return results


def run_temporal_analysis_standalone(model_id: int, use_ar_model: bool = True):
    '''
    Run temporal dynamics analysis on a trained model.
    Includes spectral fidelity analysis and Jacobian eigenvalue analysis.
    
    Args:
        model_id: Model to evaluate (1 or 2)
        use_ar_model: If True, use AR-validated model, else use step-validated
    '''
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    
    logger.info("=" * 60)
    logger.info(f"TEMPORAL DYNAMICS ANALYSIS — Model {model_id}")
    logger.info("=" * 60)
    
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    # Load model
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists() and use_ar_model:
        logger.warning("AR model not found, falling back to step-validated model")
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    logger.info(f"Loaded model from {model_file}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load static data and build FeatureBuilder
    static_data = StaticGraphData(model_path)
    logger.info(f"1D nodes: {static_data.num_1d}, 2D nodes: {static_data.num_2d}")
    fb = FeatureBuilder(static_data, feature_scales=std_values,
                        rain_lag_steps=model_cfg.get('rain_lag_steps', Config.RAIN_LAG_STEPS),
                        rain_future_steps=model_cfg.get('rain_future_steps', 0),
                        wl_prev_steps=FeatureBuilder.infer_wl_prev_steps(model_cfg, static_data),
                        use_edge_features=model_cfg.get('edge_dim_1d', 0) > 0 or model_cfg.get('edge_dim_2d', 0) > 0)
    
    # Get validation events
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    logger.info(f"Evaluating on {len(val_events)} validation events: {val_events}")
    
    # 1) Spectral Fidelity Analysis
    logger.info("")
    logger.info("Running Spectral Fidelity Analysis...")
    spectral_results = spectral_fidelity_analysis(
        model, static_data, val_events, model_path, cfg.DEVICE,
        feature_scales=std_values, fb=fb
    )
    
    # 2) Jacobian Eigenvalue Analysis
    logger.info("")
    logger.info("Running Jacobian Eigenvalue Analysis...")
    jacobian_results = jacobian_eigenvalue_analysis(
        model, static_data, val_events, model_path, cfg.DEVICE,
        feature_scales=std_values, fb=fb
    )
    
    # Combined summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEMPORAL DYNAMICS SUMMARY")
    logger.info("=" * 60)
    
    if spectral_results and jacobian_results:
        # Cross-diagnose
        sr_1d = jacobian_results.get('spectral_radius_1d_median', 0)
        sr_2d = jacobian_results.get('spectral_radius_2d_median', 0)
        hf_1d = spectral_results.get('hf_damping_1d', 0)
        hf_2d = spectral_results.get('hf_damping_2d', 0)
        coh_1d = spectral_results.get('spectral_score_1d', 0)
        coh_2d = spectral_results.get('spectral_score_2d', 0)
        
        logger.info(f"  {'Metric':30s} {'1D':>10s} {'2D':>10s}")
        logger.info(f"  {'-'*30} {'-'*10} {'-'*10}")
        logger.info(f"  {'Spectral Radius ρ(J)':30s} {sr_1d:10.4f} {sr_2d:10.4f}")
        logger.info(f"  {'HF Damping (PSD ratio)':30s} {hf_1d:10.4f} {hf_2d:10.4f}")
        logger.info(f"  {'Mean Coherence':30s} {coh_1d:10.4f} {coh_2d:10.4f}")
        logger.info(f"  {'Stable':30s} {'✓' if sr_1d < 1.0 else '✗':>10s} {'✓' if sr_2d < 1.0 else '✗':>10s}")
        logger.info("")
        
        # Combined interpretation
        if sr_1d > 1.0 and hf_1d < 0.5:
            logger.warning("  1D: Unstable AND over-smoothed — likely oscillating then diverging")
        elif sr_1d > 1.0:
            logger.warning("  1D: Unstable — spectral normalisation or more rollout steps recommended")
        elif hf_1d < 0.5:
            logger.warning("  1D: Stable but over-smoothed — temporal encoding may help (timestep features)")
        else:
            logger.info("  1D: Dynamics look healthy ✓")
        
        if sr_2d > 1.0 and hf_2d < 0.5:
            logger.warning("  2D: Unstable AND over-smoothed — likely oscillating then diverging")
        elif sr_2d > 1.0:
            logger.warning("  2D: Unstable — spectral normalisation or more rollout steps recommended")
        elif hf_2d < 0.5:
            logger.warning("  2D: Stable but over-smoothed — temporal encoding may help (timestep features)")
        else:
            logger.info("  2D: Dynamics look healthy ✓")
    
    logger.info("=" * 60)
    
    return {
        'spectral': spectral_results,
        'jacobian': jacobian_results,
    }


# =============================================================================
# LOSS LANDSCAPE VISUALIZATION
# =============================================================================

@torch.no_grad()
def get_random_direction(model, filter_normalize: bool = True) -> Dict[str, torch.Tensor]:
    '''
    Generate a random direction in parameter space.
    
    Filter normalization (Li et al. 2018) ensures directions are scale-invariant
    by normalizing each filter/layer to have the same norm as the model weights.
    
    Args:
        model: The neural network model
        filter_normalize: If True, apply filter normalization
        
    Returns:
        Dictionary mapping parameter names to random direction tensors
    '''
    direction = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Generate random direction
            d = torch.randn_like(param)
            
            if filter_normalize and len(param.shape) >= 2:
                # Filter normalization: scale direction to have same norm as weights
                # For each filter (first dimension), normalize separately
                for i in range(param.shape[0]):
                    d_filter = d[i]
                    w_filter = param[i]
                    d[i] = d_filter * (w_filter.norm() / (d_filter.norm() + 1e-10))
            elif filter_normalize:
                # For biases and 1D params, normalize the whole tensor
                d = d * (param.norm() / (d.norm() + 1e-10))
            
            direction[name] = d
    
    return direction


def perturb_model(model, direction: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    '''
    Perturb model weights: w' = w + alpha * direction
    
    Returns original weights dict for restoration.
    '''
    original_weights = {}
    
    for name, param in model.named_parameters():
        if name in direction:
            original_weights[name] = param.data.clone()
            param.data.add_(direction[name].to(param.device), alpha=alpha)
    
    return original_weights


def restore_model(model, original_weights: Dict[str, torch.Tensor]):
    '''Restore original model weights.'''
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])


def compute_loss_at_point(model, static_data: StaticGraphData, events: List[str],
                          model_path: Path, std_1d: float, std_2d: float,
                          device: torch.device, feature_scales: Dict[str, float],
                          n_samples: int = 3) -> float:
    '''
    Compute average loss over a few samples for loss landscape.
    Uses AR validation on subset of events for representative loss.
    '''
    # Sample subset of events for speed
    sample_events = events[:min(n_samples, len(events))]
    
    ar_loss, _, _ = autoregressive_validation(
        model, static_data, sample_events, model_path,
        std_1d, std_2d, device,
        verbose=False, feature_scales=feature_scales
    )
    
    return ar_loss


def visualize_loss_landscape_1d(model_id: int, use_ar_model: bool = True,
                                  n_points: int = 51, alpha_range: float = 1.0,
                                  n_samples: int = 3):
    '''
    1D loss landscape: Plot loss along a random direction.
    
    Args:
        model_id: Model to analyze (1 or 2)
        use_ar_model: Use AR-validated model
        n_points: Number of points along the direction
        alpha_range: Range of alpha values [-alpha_range, alpha_range]
        n_samples: Number of events to sample for loss computation
    '''
    import matplotlib.pyplot as plt
    
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    logger.info("=" * 60)
    logger.info(f"1D Loss Landscape for Model {model_id}")
    logger.info("=" * 60)
    
    # Load model
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists():
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None, None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    logger.info(f"Loaded model from {model_file}")
    
    # Load static data
    static_data = StaticGraphData(model_path)
    
    # Get validation events
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    
    # Generate random direction
    logger.info("Generating filter-normalized random direction...")
    direction = get_random_direction(model, filter_normalize=True)
    
    # Compute loss along direction
    alphas = np.linspace(-alpha_range, alpha_range, n_points)
    losses = []
    
    logger.info(f"Computing loss at {n_points} points...")
    for alpha in tqdm(alphas, desc="Loss landscape"):
        # Perturb model
        original_weights = perturb_model(model, direction, alpha)
        
        # Compute loss
        loss = compute_loss_at_point(
            model, static_data, val_events, model_path,
            std_values['1d'], std_values['2d'], cfg.DEVICE, std_values, n_samples
        )
        losses.append(loss)
        
        # Restore original weights
        restore_model(model, original_weights)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, losses, 'b-', linewidth=2)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Trained model')
    ax.set_xlabel('Perturbation α', fontsize=12)
    ax.set_ylabel('AR Validation Loss (Standardized RMSE)', fontsize=12)
    ax.set_title(f'1D Loss Landscape - Model {model_id}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    save_dir = cfg.OUTPUT_PATH / 'loss_landscape'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f'loss_landscape_1d_m{model_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 1D loss landscape to {save_dir / f'loss_landscape_1d_m{model_id}.png'}")
    logger.info(f"Min loss: {min(losses):.4f} at α={alphas[np.argmin(losses)]:.3f}")
    logger.info(f"Loss at α=0: {losses[n_points//2]:.4f}")
    
    return alphas, losses


def visualize_loss_landscape_2d(model_id: int, use_ar_model: bool = True,
                                  n_points: int = 21, alpha_range: float = 1.0,
                                  n_samples: int = 2):
    '''
    2D loss landscape: Contour plot along two random directions.
    
    Args:
        model_id: Model to analyze (1 or 2)
        use_ar_model: Use AR-validated model
        n_points: Number of points per dimension (total: n_points^2)
        alpha_range: Range of alpha values [-alpha_range, alpha_range]
        n_samples: Number of events to sample for loss computation
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    logger.info("=" * 60)
    logger.info(f"2D Loss Landscape for Model {model_id}")
    logger.info(f"Grid: {n_points}x{n_points} = {n_points**2} evaluations")
    logger.info("=" * 60)
    
    # Load model
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists():
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None, None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    logger.info(f"Loaded model from {model_file}")
    
    # Load static data
    static_data = StaticGraphData(model_path)
    
    # Get validation events
    events = sorted([d.name for d in model_path.iterdir() if d.is_dir() and d.name.startswith('event_')])
    n_val = max(1, int(len(events) * cfg.VAL_RATIO))
    val_events = events[-n_val:]
    
    # Generate two random directions
    logger.info("Generating filter-normalized random directions...")
    direction1 = get_random_direction(model, filter_normalize=True)
    direction2 = get_random_direction(model, filter_normalize=True)
    
    # Store original weights
    original_weights = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Compute loss grid
    alphas = np.linspace(-alpha_range, alpha_range, n_points)
    losses = np.zeros((n_points, n_points))
    
    logger.info(f"Computing loss at {n_points**2} points...")
    total_iters = n_points * n_points
    pbar = tqdm(total=total_iters, desc="2D Loss landscape")
    
    for i, alpha1 in enumerate(alphas):
        for j, alpha2 in enumerate(alphas):
            # Restore original weights first
            for name, param in model.named_parameters():
                if name in original_weights:
                    param.data.copy_(original_weights[name])
            
            # Perturb model in both directions
            for name, param in model.named_parameters():
                if name in direction1:
                    param.data.add_(direction1[name].to(param.device), alpha=alpha1)
                    param.data.add_(direction2[name].to(param.device), alpha=alpha2)
            
            # Compute loss
            loss = compute_loss_at_point(
                model, static_data, val_events, model_path,
                std_values['1d'], std_values['2d'], cfg.DEVICE, std_values, n_samples
            )
            losses[i, j] = loss
            pbar.update(1)
    
    pbar.close()
    
    # Restore original weights
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])
    
    # Plot
    fig = plt.figure(figsize=(16, 6))
    
    # Contour plot
    ax1 = fig.add_subplot(1, 2, 1)
    X, Y = np.meshgrid(alphas, alphas)
    contour = ax1.contourf(X, Y, losses.T, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Loss')
    ax1.scatter([0], [0], c='red', s=100, marker='*', label='Trained model', zorder=5)
    ax1.set_xlabel('Direction 1 (α₁)', fontsize=12)
    ax1.set_ylabel('Direction 2 (α₂)', fontsize=12)
    ax1.set_title(f'2D Loss Landscape - Model {model_id}', fontsize=14)
    ax1.legend()
    
    # 3D surface plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, losses.T, cmap='viridis', alpha=0.8)
    ax2.scatter([0], [0], [losses[n_points//2, n_points//2]], c='red', s=100, marker='*')
    ax2.set_xlabel('Direction 1')
    ax2.set_ylabel('Direction 2')
    ax2.set_zlabel('Loss')
    ax2.set_title('3D Surface View')
    
    plt.tight_layout()
    
    # Save
    save_dir = cfg.OUTPUT_PATH / 'loss_landscape'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f'loss_landscape_2d_m{model_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save raw data
    np.savez(save_dir / f'loss_landscape_data_m{model_id}.npz',
             alphas=alphas, losses=losses)
    
    logger.info(f"Saved 2D loss landscape to {save_dir / f'loss_landscape_2d_m{model_id}.png'}")
    logger.info(f"Min loss: {losses.min():.4f}")
    logger.info(f"Max loss: {losses.max():.4f}")
    logger.info(f"Loss at center: {losses[n_points//2, n_points//2]:.4f}")
    
    # Sharpness analysis
    center_loss = losses[n_points//2, n_points//2]
    avg_neighbor_loss = (losses[n_points//2-1, n_points//2] + losses[n_points//2+1, n_points//2] +
                        losses[n_points//2, n_points//2-1] + losses[n_points//2, n_points//2+1]) / 4
    sharpness = avg_neighbor_loss - center_loss
    logger.info(f"Local sharpness: {sharpness:.4f} (higher = sharper minimum)")
    
    return alphas, losses


def run_loss_landscape_standalone(model_id: int = None, mode: str = '2d',
                                   use_ar_model: bool = True):
    '''
    Standalone function to run loss landscape visualization.
    
    Args:
        model_id: Model to analyze (1, 2, or None for both)
        mode: '1d' or '2d' visualization
        use_ar_model: Use AR-validated model
    '''
    models_to_process = [model_id] if model_id else [1, 2]
    
    for mid in models_to_process:
        if mode == '1d':
            visualize_loss_landscape_1d(mid, use_ar_model, n_points=51, alpha_range=1.0)
        else:
            visualize_loss_landscape_2d(mid, use_ar_model, n_points=21, alpha_range=1.0)


# =============================================================================
# PHASE DETECTION TEST — GT vs MODEL COMPARISON
# =============================================================================

def _compute_pipe_crown_elevations(static_data: StaticGraphData) -> np.ndarray:
    '''Compute pipe crown elevation at each 1D node from static data.
    
    pipe_crown = invert_elevation + max(pipe_diameter at connected edges)
    Diameter in the static CSV is in feet → convert to meters (×0.3048).
    '''
    n_1d = static_data.num_1d
    invert_elev = static_data.nodes_1d['invert_elevation'].values.astype(np.float32)
    
    # Max pipe diameter at each node (from edge_index + edges_static)
    max_diam = np.zeros(n_1d, dtype=np.float32)
    diameters = static_data.edges_1d_static['diameter'].values.astype(np.float32) * 0.3048  # ft→m
    from_nodes = static_data.edges_1d['from_node'].values
    to_nodes   = static_data.edges_1d['to_node'].values
    for i, (fn, tn) in enumerate(zip(from_nodes, to_nodes)):
        max_diam[fn] = max(max_diam[fn], diameters[i])
        max_diam[tn] = max(max_diam[tn], diameters[i])
    
    pipe_crown = invert_elev + max_diam
    return pipe_crown


def detect_phases_from_wl(
    wl_1d: np.ndarray,           # [T, n_1d]
    wl_2d: np.ndarray,           # [T, n_2d]
    rainfall: np.ndarray,        # [T, n_2d]
    invert_elev: np.ndarray,     # [n_1d]
    terrain_elev: np.ndarray,    # [n_1d]  (surface_elevation)
    pipe_crown: np.ndarray,      # [n_1d]
    min_elev_2d: np.ndarray,     # [n_2d]
    label: str = "GT",
) -> dict:
    '''Detect hydraulic phases from water level arrays (GT or predicted).
    
    Returns dict with per-timestep hydraulic indicators + detected phase list.
    Works with ONLY water levels + rainfall (no inlet_flow needed — so it can
    run on model predictions too).
    '''
    n_ts, n_1d = wl_1d.shape
    n_2d = wl_2d.shape[1]
    
    # ── Aggregate signals ──
    mean_rain   = rainfall.mean(axis=1)
    mean_wl_1d  = wl_1d.mean(axis=1)
    mean_wl_2d  = wl_2d.mean(axis=1)
    
    # 2D "volume proxy" — sum of pond depth (wl - min_elev) across all 2D cells
    pond_2d = np.maximum(0, wl_2d - min_elev_2d[None, :])
    total_pond  = pond_2d.sum(axis=1)
    
    dwl_2d = np.gradient(mean_wl_2d)
    dwl_1d = np.gradient(mean_wl_1d)
    
    # ── Hydraulic states (per-timestep, per-node) ──
    # 0=free, 1=surcharge (WL > pipe crown), 2=flooding (WL > terrain)
    state_1d = np.zeros_like(wl_1d, dtype=np.int8)
    state_1d[wl_1d >= pipe_crown[None, :]] = 1
    state_1d[wl_1d >= terrain_elev[None, :]] = 2
    
    n_surcharging = (state_1d >= 1).sum(axis=1)
    n_flooding    = (state_1d >= 2).sum(axis=1)
    pct_surcharge = n_surcharging / n_1d * 100
    pct_flooding  = n_flooding / n_1d * 100
    
    # Surcharge depth
    surcharge_depth = np.maximum(0, wl_1d - pipe_crown[None, :])
    max_surcharge_depth = surcharge_depth.max(axis=1)
    
    # Flood depth
    flood_depth = np.maximum(0, wl_1d - terrain_elev[None, :])
    max_flood_depth = flood_depth.max(axis=1)
    
    # Pipe fullness = (wl - invert) / (crown - invert)
    pipe_height = np.maximum(pipe_crown - invert_elev, 0.01)
    fullness = np.clip((wl_1d - invert_elev[None, :]) / pipe_height[None, :], 0, None)
    mean_fullness = fullness.mean(axis=1)
    
    # ── Rainfall boundaries ──
    rain_threshold = max(mean_rain.max() * 0.02, 1e-6)
    rain_active = mean_rain > rain_threshold
    
    rain_start = int(np.argmax(rain_active)) if rain_active.any() else None
    rain_end   = int(n_ts - 1 - np.argmax(rain_active[::-1])) if rain_active.any() else None
    rain_peak  = int(np.argmax(mean_rain))
    
    peak_wl_2d_t = int(np.argmax(mean_wl_2d))
    peak_wl_1d_t = int(np.argmax(mean_wl_1d))
    peak_pond_t  = int(np.argmax(total_pond))
    peak_surcharge_t = int(np.argmax(n_surcharging))
    
    # ── Phase detection ──
    phases = []
    
    # 1. Baseflow
    if rain_start is not None and rain_start > 0:
        phases.append(('BASEFLOW', 0, rain_start - 1))
    
    # 2. Initial Wetting
    if rain_start is not None:
        wetting_end = min(rain_peak, rain_start + max(3, (rain_peak - rain_start) // 2))
        if wetting_end > rain_start:
            phases.append(('INITIAL_WETTING', rain_start, wetting_end))
        
        # 3. Rising Limb
        rising_start = wetting_end + 1
        if rising_start < rain_peak:
            phases.append(('RISING_LIMB', rising_start, rain_peak))
        
        # 4. Peak Rainfall
        peak_window = max(2, (rain_end - rain_peak) // 4) if rain_end > rain_peak else 2
        peak_end_t = min(rain_peak + peak_window, rain_end)
        phases.append(('PEAK_RAINFALL', rain_peak, peak_end_t))
        
        # 5. Declining Rainfall
        declining_start = peak_end_t + 1
        if declining_start < rain_end:
            phases.append(('DECLINING_RAINFALL', declining_start, rain_end))
        
        # 6. Lag-to-Peak
        post_rain_peak_end = max(peak_wl_2d_t, peak_wl_1d_t, peak_pond_t)
        if post_rain_peak_end > rain_end:
            phases.append(('LAG_TO_PEAK', rain_end + 1, post_rain_peak_end))
    
    # 7/8. Recession phases
    rec_start = max(peak_wl_2d_t, peak_wl_1d_t, peak_pond_t,
                    rain_end if rain_end is not None else 0) + 1
    if rec_start < n_ts - 1:
        recession_rates = -dwl_2d[rec_start:]
        if len(recession_rates) > 2 and recession_rates.max() > 0:
            fast_thresh = recession_rates.max() * 0.30
            slow_idx = np.where(recession_rates < fast_thresh)[0]
            fast_end = rec_start + (slow_idx[1] if len(slow_idx) > 1 else len(recession_rates) // 2)
        else:
            fast_end = min(rec_start + 15, n_ts - 1)
        fast_end = min(fast_end, n_ts - 1)
        if fast_end > rec_start:
            phases.append(('FAST_RECESSION', rec_start, fast_end))
        if fast_end + 1 < n_ts - 1:
            phases.append(('SLOW_RECESSION', fast_end + 1, n_ts - 1))
    
    # ── Surcharge / flooding phases (overlay) ──
    any_surcharge = n_surcharging > 0
    first_surcharge = int(np.argmax(any_surcharge)) if any_surcharge.any() else None
    last_surcharge  = int(n_ts - 1 - np.argmax(any_surcharge[::-1])) if any_surcharge.any() else None
    
    any_flood = n_flooding > 0
    first_flood = int(np.argmax(any_flood)) if any_flood.any() else None
    last_flood  = int(n_ts - 1 - np.argmax(any_flood[::-1])) if any_flood.any() else None
    
    if first_surcharge is not None and last_surcharge is not None and (last_surcharge - first_surcharge) > 0:
        phases.append(('SURCHARGE_ACTIVE', first_surcharge, last_surcharge))
    
    if first_flood is not None and last_flood is not None and n_flooding.max() > 0:
        phases.append(('FLOODING_ACTIVE', first_flood, last_flood))
    
    return {
        'label': label,
        'phases': phases,
        # Per-timestep signals for comparison
        'mean_rain': mean_rain,
        'mean_wl_1d': mean_wl_1d,
        'mean_wl_2d': mean_wl_2d,
        'total_pond': total_pond,
        'n_surcharging': n_surcharging,
        'n_flooding': n_flooding,
        'pct_surcharge': pct_surcharge,
        'pct_flooding': pct_flooding,
        'max_surcharge_depth': max_surcharge_depth,
        'max_flood_depth': max_flood_depth,
        'mean_fullness': mean_fullness,
        'peak_wl_1d_t': peak_wl_1d_t,
        'peak_wl_2d_t': peak_wl_2d_t,
        'peak_surcharge_t': peak_surcharge_t,
        'rain_peak': rain_peak,
        # Node-level state for detailed analysis
        'state_1d': state_1d,
    }


def compare_phases(gt_result: dict, pred_result: dict, n_1d: int, event_name: str) -> dict:
    '''Compare GT vs Predicted phase detection results for one event.
    
    Returns a comparison dict with:
      - phases present in GT but missed by model
      - phases detected by model but not in GT (false positives)
      - timing errors for matching phases
      - hydraulic state mismatches (surcharge/flood detection accuracy)
    '''
    gt_phases = {name: (s, e) for name, s, e in gt_result['phases']}
    pred_phases = {name: (s, e) for name, s, e in pred_result['phases']}
    
    all_phase_names = sorted(set(list(gt_phases.keys()) + list(pred_phases.keys())))
    
    comparison = {
        'event': event_name,
        'gt_phases': gt_phases,
        'pred_phases': pred_phases,
        'missed_phases': [],       # In GT but not in pred
        'false_phases': [],        # In pred but not in GT
        'timing_errors': {},       # Phase name → (start_err, end_err, duration_err)
        'peak_timing': {},         # Key peak timing comparisons
        'hydraulic_accuracy': {},  # Per-timestep accuracy of surcharge/flood detection
    }
    
    # ── Phase matching ──
    for name in all_phase_names:
        in_gt = name in gt_phases
        in_pred = name in pred_phases
        
        if in_gt and not in_pred:
            comparison['missed_phases'].append(name)
        elif in_pred and not in_gt:
            comparison['false_phases'].append(name)
        elif in_gt and in_pred:
            gt_s, gt_e = gt_phases[name]
            pr_s, pr_e = pred_phases[name]
            comparison['timing_errors'][name] = {
                'start_err': (pr_s - gt_s) * 5,  # in minutes
                'end_err': (pr_e - gt_e) * 5,
                'duration_err': ((pr_e - pr_s) - (gt_e - gt_s)) * 5,
                'gt_range': (gt_s, gt_e),
                'pred_range': (pr_s, pr_e),
            }
    
    # ── Peak timing comparison ──
    comparison['peak_timing'] = {
        'rain_peak':       {'gt': gt_result['rain_peak'], 'pred': pred_result['rain_peak'],
                            'err_min': 0},  # Rain is input, should be same
        'wl_1d_peak':      {'gt': gt_result['peak_wl_1d_t'], 'pred': pred_result['peak_wl_1d_t'],
                            'err_min': (pred_result['peak_wl_1d_t'] - gt_result['peak_wl_1d_t']) * 5},
        'wl_2d_peak':      {'gt': gt_result['peak_wl_2d_t'], 'pred': pred_result['peak_wl_2d_t'],
                            'err_min': (pred_result['peak_wl_2d_t'] - gt_result['peak_wl_2d_t']) * 5},
        'surcharge_peak':  {'gt': gt_result['peak_surcharge_t'], 'pred': pred_result['peak_surcharge_t'],
                            'err_min': (pred_result['peak_surcharge_t'] - gt_result['peak_surcharge_t']) * 5},
    }
    
    # ── Per-timestep hydraulic state accuracy ──
    # How well does the model reproduce surcharge/flood states at each node?
    gt_state = gt_result['state_1d']     # [T, n_1d]  values: 0/1/2
    pred_state = pred_result['state_1d'] # [T, n_1d]
    
    # Only compare after warmup
    warmup = Config.WARMUP_TIMESTEPS
    gt_s = gt_state[warmup:]
    pr_s = pred_state[warmup:]
    
    # Overall accuracy
    total_entries = gt_s.size
    exact_match = (gt_s == pr_s).sum()
    comparison['hydraulic_accuracy']['exact_match_pct'] = 100 * exact_match / total_entries
    
    # Surcharge detection accuracy (state >= 1)
    gt_surch = gt_s >= 1
    pr_surch = pr_s >= 1
    tp_surch = (gt_surch & pr_surch).sum()
    fn_surch = (gt_surch & ~pr_surch).sum()  # GT surcharges missed by model
    fp_surch = (~gt_surch & pr_surch).sum()  # Model predicts surcharge where GT doesn't
    comparison['hydraulic_accuracy']['surcharge'] = {
        'tp': int(tp_surch), 'fn': int(fn_surch), 'fp': int(fp_surch),
        'recall': 100 * tp_surch / max(1, gt_surch.sum()),
        'precision': 100 * tp_surch / max(1, pr_surch.sum()),
    }
    
    # Flood detection accuracy (state >= 2)
    gt_flood = gt_s >= 2
    pr_flood = pr_s >= 2
    tp_flood = (gt_flood & pr_flood).sum()
    fn_flood = (gt_flood & ~pr_flood).sum()
    fp_flood = (~gt_flood & pr_flood).sum()
    comparison['hydraulic_accuracy']['flooding'] = {
        'tp': int(tp_flood), 'fn': int(fn_flood), 'fp': int(fp_flood),
        'recall': 100 * tp_flood / max(1, gt_flood.sum()),
        'precision': 100 * tp_flood / max(1, pr_flood.sum()),
    }
    
    # Per-timestep surcharge/flood count error
    comparison['hydraulic_accuracy']['surcharge_count_mae'] = float(
        np.abs(gt_result['n_surcharging'][warmup:] - pred_result['n_surcharging'][warmup:]).mean())
    comparison['hydraulic_accuracy']['flood_count_mae'] = float(
        np.abs(gt_result['n_flooding'][warmup:] - pred_result['n_flooding'][warmup:]).mean())
    comparison['hydraulic_accuracy']['max_surcharge_depth_mae'] = float(
        np.abs(gt_result['max_surcharge_depth'][warmup:] - pred_result['max_surcharge_depth'][warmup:]).mean())
    comparison['hydraulic_accuracy']['max_flood_depth_mae'] = float(
        np.abs(gt_result['max_flood_depth'][warmup:] - pred_result['max_flood_depth'][warmup:]).mean())
    
    return comparison


def run_phase_test(model_id: int, use_ar_model: bool = True):
    '''Run hydraulic phase detection test: GT vs trained model predictions.
    
    For ALL training events, detects phases in GT and model predictions,
    then reports which phases the model misses or gets wrong.
    '''
    cfg = MODEL_CONFIGS.get(model_id, Model1Config)
    
    logger.info("=" * 80)
    logger.info(f"  HYDRAULIC PHASE DETECTION TEST — Model {model_id}")
    logger.info("=" * 80)
    
    model_path = cfg.BASE_PATH / f"Model_{model_id}" / "train"
    
    # ── Load trained model ──
    model_suffix = "_ar" if use_ar_model else ""
    model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}{model_suffix}.pt"
    if not model_file.exists() and use_ar_model:
        logger.warning("AR model not found, falling back to step-validated model")
        model_file = cfg.OUTPUT_PATH / f"best_model_{model_id}.pt"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return None
    
    model, model_cfg, std_values = load_model_from_checkpoint(model_file, cfg.DEVICE)
    logger.info(f"Loaded model from {model_file}")
    
    # ── Load static data ──
    static_data = StaticGraphData(model_path)
    fb = FeatureBuilder(static_data, feature_scales=std_values,
                        rain_lag_steps=model_cfg.get('rain_lag_steps', Config.RAIN_LAG_STEPS),
                        wl_prev_steps=FeatureBuilder.infer_wl_prev_steps(model_cfg, static_data),
                        use_edge_features=model_cfg.get('edge_dim_1d', 0) > 0 or model_cfg.get('edge_dim_2d', 0) > 0)
    fb.to(cfg.DEVICE)
    
    n_1d = static_data.num_1d
    n_2d = static_data.num_2d
    
    # ── Compute static thresholds ──
    invert_elev  = static_data.nodes_1d['invert_elevation'].values.astype(np.float32)
    terrain_elev = static_data.nodes_1d['surface_elevation'].values.astype(np.float32)
    pipe_crown   = _compute_pipe_crown_elevations(static_data)
    min_elev_2d  = static_data.nodes_2d['min_elevation'].values.astype(np.float32)
    
    logger.info(f"1D nodes: {n_1d}, 2D nodes: {n_2d}")
    logger.info(f"Pipe crown range: [{pipe_crown.min():.2f}, {pipe_crown.max():.2f}]")
    logger.info(f"Terrain range:    [{terrain_elev.min():.2f}, {terrain_elev.max():.2f}]")
    
    # ── Get ALL training events ──
    events = sorted([d.name for d in model_path.iterdir()
                     if d.is_dir() and d.name.startswith('event_')])
    logger.info(f"Running phase test on {len(events)} events")
    
    # Pre-allocate GPU tensors
    wl_1d_gpu = torch.zeros(n_1d, dtype=torch.float32, device=cfg.DEVICE)
    wl_2d_gpu = torch.zeros(n_2d, dtype=torch.float32, device=cfg.DEVICE)
    use_amp = Config.USE_AMP and cfg.DEVICE.type == 'cuda'
    
    # ── Accumulators ──
    all_comparisons = []
    phase_miss_counts = {}   # phase_name → count of events where it was missed
    phase_false_counts = {}  # phase_name → count of false detections
    phase_present_counts = {}  # phase_name → count of events where it exists in GT
    
    # Aggregate hydraulic accuracy
    total_surch_tp = total_surch_fn = total_surch_fp = 0
    total_flood_tp = total_flood_fn = total_flood_fp = 0
    all_peak_timing_errs = {'wl_1d': [], 'wl_2d': [], 'surcharge': []}
    
    for event_name in events:
        event_path = model_path / event_name
        
        # ── Load GT dynamic data ──
        df_1d = pd.read_csv(event_path / '1d_nodes_dynamic_all.csv')
        df_2d = pd.read_csv(event_path / '2d_nodes_dynamic_all.csv')
        
        n_timesteps = int(df_1d['timestep'].max() + 1)
        
        wl_1d_gt = df_1d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        wl_2d_gt = df_2d.pivot(index='timestep', columns='node_idx', values='water_level').values.astype(np.float32)
        rainfall_all = np.zeros((n_timesteps, n_2d), dtype=np.float32)
        if 'rainfall' in df_2d.columns:
            pivot_rain = df_2d.pivot(index='timestep', columns='node_idx', values='rainfall').fillna(0)
            rainfall_all[:len(pivot_rain)] = pivot_rain.values
        
        # ── Run model prediction (autoregressive rollout) ──
        wl_1d_pred, wl_2d_pred = run_ar_rollout(
            model, fb, wl_1d_gt, wl_2d_gt, rainfall_all, cfg.DEVICE, use_amp=use_amp)
        
        # ── Detect phases: GT ──
        gt_result = detect_phases_from_wl(
            wl_1d_gt, wl_2d_gt, rainfall_all,
            invert_elev, terrain_elev, pipe_crown, min_elev_2d, label="GT")
        
        # ── Detect phases: Predicted ──
        pred_result = detect_phases_from_wl(
            wl_1d_pred, wl_2d_pred, rainfall_all,
            invert_elev, terrain_elev, pipe_crown, min_elev_2d, label="PRED")
        
        # ── Compare ──
        comp = compare_phases(gt_result, pred_result, n_1d, event_name)
        all_comparisons.append(comp)
        
        # Accumulate
        for p in comp['missed_phases']:
            phase_miss_counts[p] = phase_miss_counts.get(p, 0) + 1
        for p in comp['false_phases']:
            phase_false_counts[p] = phase_false_counts.get(p, 0) + 1
        for name, _, _ in gt_result['phases']:
            phase_present_counts[name] = phase_present_counts.get(name, 0) + 1
        
        sa = comp['hydraulic_accuracy']['surcharge']
        fa = comp['hydraulic_accuracy']['flooding']
        total_surch_tp += sa['tp']; total_surch_fn += sa['fn']; total_surch_fp += sa['fp']
        total_flood_tp += fa['tp']; total_flood_fn += fa['fn']; total_flood_fp += fa['fp']
        
        for key in ['wl_1d', 'wl_2d', 'surcharge']:
            pk = key + '_peak' if key != 'surcharge' else 'surcharge_peak'
            all_peak_timing_errs[key].append(comp['peak_timing'][pk]['err_min'])
    
    # ===========================================================================
    # PRINT RESULTS
    # ===========================================================================
    
    n_events = len(events)
    W = 90
    
    logger.info("\n" + "=" * W)
    logger.info("  PHASE DETECTION TEST RESULTS")
    logger.info("=" * W)
    
    # -- Section 1: Phase detection summary --
    logger.info("\n  PHASE DETECTION SUMMARY (GT vs Model)")
    logger.info("  " + "-" * 70)
    logger.info(f"  {'Phase':<25s} {'In GT':>6s} {'Missed':>7s} {'Miss%':>6s} {'False+':>7s}")
    logger.info("  " + "-" * 70)
    
    all_phase_names = sorted(set(
        list(phase_present_counts.keys()) + 
        list(phase_miss_counts.keys()) + 
        list(phase_false_counts.keys())))
    
    for name in all_phase_names:
        in_gt = phase_present_counts.get(name, 0)
        missed = phase_miss_counts.get(name, 0)
        false_pos = phase_false_counts.get(name, 0)
        miss_pct = 100 * missed / max(1, in_gt)
        logger.info(f"  {name:<25s} {in_gt:6d} {missed:7d} {miss_pct:5.1f}% {false_pos:7d}")
    
    # -- Section 2: Timing accuracy for matched phases --
    logger.info(f"\n  PHASE TIMING ERRORS (for events where both GT & model detect the phase)")
    logger.info("  " + "-" * 70)
    
    # Aggregate timing errors per phase
    phase_timing_agg = {}
    for comp in all_comparisons:
        for pname, terr in comp['timing_errors'].items():
            if pname not in phase_timing_agg:
                phase_timing_agg[pname] = {'start': [], 'end': [], 'duration': []}
            phase_timing_agg[pname]['start'].append(terr['start_err'])
            phase_timing_agg[pname]['end'].append(terr['end_err'])
            phase_timing_agg[pname]['duration'].append(terr['duration_err'])
    
    logger.info(f"  {'Phase':<25s} {'N':>4s} {'Start Err':>10s} {'End Err':>10s} {'Dur Err':>10s}")
    for name in sorted(phase_timing_agg.keys()):
        d = phase_timing_agg[name]
        n = len(d['start'])
        s_mean = np.mean(d['start'])
        e_mean = np.mean(d['end'])
        dur_mean = np.mean(d['duration'])
        logger.info(f"  {name:<25s} {n:4d} {s_mean:+9.1f}m {e_mean:+9.1f}m {dur_mean:+9.1f}m")
    
    # -- Section 3: Peak timing accuracy --
    logger.info(f"\n  PEAK TIMING ERRORS (minutes)")
    logger.info("  " + "-" * 70)
    for key, errs in all_peak_timing_errs.items():
        arr = np.array(errs)
        logger.info(f"  {key:>15s} peak: mean={arr.mean():+.1f}m  std={arr.std():.1f}m  "
                    f"median={np.median(arr):+.1f}m  |max|={np.abs(arr).max():.0f}m")
    
    # -- Section 4: Hydraulic state accuracy --
    logger.info(f"\n  HYDRAULIC STATE ACCURACY (after warmup, across all events)")
    logger.info("  " + "-" * 70)
    
    surch_recall = 100 * total_surch_tp / max(1, total_surch_tp + total_surch_fn)
    surch_precision = 100 * total_surch_tp / max(1, total_surch_tp + total_surch_fp)
    flood_recall = 100 * total_flood_tp / max(1, total_flood_tp + total_flood_fn)
    flood_precision = 100 * total_flood_tp / max(1, total_flood_tp + total_flood_fp)
    
    logger.info(f"  Surcharge Detection:")
    logger.info(f"    Recall:    {surch_recall:.1f}% (of GT surcharge node-timesteps correctly predicted)")
    logger.info(f"    Precision: {surch_precision:.1f}% (of predicted surcharge that is real)")
    logger.info(f"    Missed:    {total_surch_fn:,} node-timesteps  |  False+: {total_surch_fp:,}")
    logger.info(f"  Flood Detection:")
    logger.info(f"    Recall:    {flood_recall:.1f}% (of GT flood node-timesteps correctly predicted)")
    logger.info(f"    Precision: {flood_precision:.1f}% (of predicted floods that are real)")
    logger.info(f"    Missed:    {total_flood_fn:,} node-timesteps  |  False+: {total_flood_fp:,}")
    
    # Aggregate MAE
    avg_surch_count_mae = np.mean([c['hydraulic_accuracy']['surcharge_count_mae'] for c in all_comparisons])
    avg_flood_count_mae = np.mean([c['hydraulic_accuracy']['flood_count_mae'] for c in all_comparisons])
    avg_surch_depth_mae = np.mean([c['hydraulic_accuracy']['max_surcharge_depth_mae'] for c in all_comparisons])
    avg_flood_depth_mae = np.mean([c['hydraulic_accuracy']['max_flood_depth_mae'] for c in all_comparisons])
    avg_exact_match = np.mean([c['hydraulic_accuracy']['exact_match_pct'] for c in all_comparisons])
    
    logger.info(f"\n  Per-timestep MAE (averaged across events):")
    logger.info(f"    Surcharging node count MAE: {avg_surch_count_mae:.1f} nodes/timestep")
    logger.info(f"    Flooding node count MAE:    {avg_flood_count_mae:.1f} nodes/timestep")
    logger.info(f"    Max surcharge depth MAE:    {avg_surch_depth_mae:.3f} m")
    logger.info(f"    Max flood depth MAE:        {avg_flood_depth_mae:.3f} m")
    logger.info(f"    Exact hydraulic state match: {avg_exact_match:.1f}%")
    
    # -- Section 5: Per-event detail (worst events) --
    logger.info(f"\n  WORST EVENTS BY MISSED PHASES")
    logger.info("  " + "-" * 70)
    
    # Sort events by number of missed phases
    events_by_misses = sorted(all_comparisons, 
                               key=lambda c: len(c['missed_phases']), reverse=True)
    for comp in events_by_misses[:10]:
        if len(comp['missed_phases']) == 0:
            break
        logger.info(f"  {comp['event']}: missed {comp['missed_phases']}")
    
    # Worst by hydraulic accuracy
    logger.info(f"\n  WORST EVENTS BY HYDRAULIC ACCURACY")
    logger.info("  " + "-" * 70)
    events_by_accuracy = sorted(all_comparisons,
                                 key=lambda c: c['hydraulic_accuracy']['exact_match_pct'])
    for comp in events_by_accuracy[:10]:
        ha = comp['hydraulic_accuracy']
        logger.info(f"  {comp['event']}: exact={ha['exact_match_pct']:.1f}%  "
                    f"surch_recall={ha['surcharge']['recall']:.1f}%  "
                    f"flood_recall={ha['flooding']['recall']:.1f}%")
    
    logger.info("\n" + "=" * W)
    logger.info("  PHASE TEST COMPLETE")
    logger.info("=" * W)
    
    return all_comparisons


# =============================================================================
# MAIN
# =============================================================================

def main():
    '''Main training and inference pipeline.'''
    import argparse
    local_rank = setup_ddp()
    if not is_main_process():
        logger.setLevel(logging.WARNING)
    set_seed(42 + get_rank())
    if is_main_process():
        Config.OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    if ddp_is_initialized():
        dist.barrier()

    parser = argparse.ArgumentParser(description="Heterogeneous GNN for Flood Modeling")
    parser.add_argument("--inference-only", action="store_true", 
                        help="Skip training and only generate submission")
    parser.add_argument("--use-step-model", action="store_true",
                        help="Use step-validated models instead of AR-validated")
    parser.add_argument("--model", type=int, choices=[1, 2], default=None,
                        help="Train only a specific model (1 or 2). Default: train both")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="Load only model weights from checkpoint and reset optimizer/scheduler "
                             "(fresh LR/curriculum phase)")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Only run mass balance diagnostic on trained models")
    parser.add_argument("--ar-val-only", action="store_true",
                        help="Only run AR validation on trained models")
    parser.add_argument("--loss-landscape", choices=['1d', '2d'], default=None,
                        help="Generate loss landscape visualization (1d or 2d)")
    parser.add_argument("--temporal-analysis", action="store_true",
                        help="Run temporal dynamics analysis (spectral fidelity + Jacobian eigenvalues)")
    parser.add_argument("--phase-test", action="store_true",
                        help="Run hydraulic phase detection test (GT vs model phases for all events)")
    args = parser.parse_args()
    
    if is_main_process():
        logger.info(f"Device: {Config.DEVICE}")
        logger.info(f"Mixed Precision: {Config.USE_AMP}")
    
    # Convenience: weights-only resume implies checkpoint loading.
    if args.resume_weights_only and not args.resume:
        args.resume = True
        if is_main_process():
            logger.info("--resume-weights-only set; enabling --resume automatically")
    
    # Determine which models to process
    models_to_process = [args.model] if args.model else [1, 2]
    
    # Run loss landscape visualization
    if args.loss_landscape:
        if not is_main_process():
            cleanup_ddp()
            return
        for model_id in models_to_process:
            run_loss_landscape_standalone(model_id, mode=args.loss_landscape, 
                                          use_ar_model=not args.use_step_model)
        logger.info("Loss landscape visualization complete!")
        cleanup_ddp()
        return
    
    # Run temporal dynamics analysis
    if args.temporal_analysis:
        if not is_main_process():
            cleanup_ddp()
            return
        for model_id in models_to_process:
            run_temporal_analysis_standalone(model_id, use_ar_model=not args.use_step_model)
        logger.info("Temporal dynamics analysis complete!")
        cleanup_ddp()
        return
    
    # Run phase detection test
    if args.phase_test:
        if not is_main_process():
            cleanup_ddp()
            return
        for model_id in models_to_process:
            run_phase_test(model_id, use_ar_model=not args.use_step_model)
        logger.info("Phase detection test complete!")
        cleanup_ddp()
        return
    
    # Run AR validation only
    if args.ar_val_only:
        if not is_main_process():
            cleanup_ddp()
            return
        for model_id in models_to_process:
            run_ar_validation_standalone(model_id, use_ar_model=not args.use_step_model)
        logger.info("AR Validation complete!")
        cleanup_ddp()
        return
    
    # Run diagnostic only
    if args.diagnostic_only:
        if not is_main_process():
            cleanup_ddp()
            return
        for model_id in models_to_process:
            run_mass_balance_diagnostic_standalone(model_id, use_ar_model=not args.use_step_model)
        logger.info("Diagnostic complete!")
        cleanup_ddp()
        return
    
    # Train models (unless inference-only)
    if not args.inference_only:
        for model_id in models_to_process:
            train_model(model_id, resume=args.resume, resume_weights_only=args.resume_weights_only)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generate submission
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Generating Submission")
        logger.info("=" * 60)
        generate_submission(use_ar_model=not args.use_step_model)
        logger.info("Pipeline complete!")

    cleanup_ddp()


if __name__ == "__main__":
    main()



