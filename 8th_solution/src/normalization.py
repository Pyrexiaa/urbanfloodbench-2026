"""
Normalization utilities for static and dynamic features with separate tracking.
Uses streaming statistics for memory efficiency.
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple


class RunningMinMaxSkew:
    """Compute min, max, and skewness incrementally without loading all data."""
    def __init__(self):
        self.n = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.M1 = 0.0  # mean
        self.M2 = 0.0  # variance moment
        self.M3 = 0.0  # skewness moment
    
    def update(self, x: np.ndarray):
        """Update statistics with new batch of data."""
        x = x.astype(float)
        n1 = self.n
        n2 = len(x)
        
        # Update min/max
        self.min_val = min(self.min_val, x.min())
        self.max_val = max(self.max_val, x.max())
        
        # Update moments for skewness (parallel algorithm)
        if n1 == 0:
            self.n = n2
            self.M1 = x.mean()
            self.M2 = ((x - self.M1) ** 2).sum()
            self.M3 = ((x - self.M1) ** 3).sum()
            return
        
        n = n1 + n2
        delta = x.mean() - self.M1
        delta_n = delta / n
        
        self.M3 = self.M3 + ((x - x.mean()) ** 3).sum() + \
                  delta ** 3 * n1 * n2 * (n1 - n2) / (n ** 2) + \
                  3 * delta * (n1 * self.M2) / n
        
        self.M2 = self.M2 + ((x - x.mean()) ** 2).sum() + \
                  delta ** 2 * n1 * n2 / n
        
        self.M1 = self.M1 + delta * n2 / n
        self.n = n
    
    def get_skewness(self):
        """Compute skewness from moments."""
        if self.n < 2:
            return 0.0
        variance = self.M2 / (self.n - 1)
        if variance < 1e-10:
            return 0.0
        std = np.sqrt(variance)
        skewness = (np.sqrt(self.n) * self.M3) / (self.M2 ** 1.5)
        return skewness


class FeatureNormalizer:
    """
    Handles normalization for static and dynamic features separately.
    - Streaming statistics for dynamic features (memory efficient)
    - Log transforms heavy-tailed features
    - Min-max scales to [0, 1]
    - Tracks params for unnormalization
    """
    
    def __init__(self, verbose: bool = False):
        self.static_params = {}  # {feature: {'min': ..., 'max': ..., 'log': bool}}
        self.dynamic_params = {}  # {feature: {'min': ..., 'max': ..., 'log': bool}}
        self.static_features = []
        self.dynamic_features = []
        self._running_stats = {}  # Temporary storage for streaming computation
        self.verbose = bool(verbose)
        
    def fit_static(self, df: pd.DataFrame, id_col: str, skew_threshold=2.0):
        """
        Fit normalization parameters on static features.
        
        Args:
            df: Static features dataframe
            id_col: ID column to exclude
            skew_threshold: Skewness threshold for log transform
        """
        self.static_features = [c for c in df.columns if c != id_col]
        
        for col in self.static_features:
            vals = df[col].astype(float).values
            skew = pd.Series(vals).skew()
            use_log = abs(skew) > skew_threshold

            if use_log:
                if self.verbose:
                    print(f"[INFO] Static {col}: skew={skew:.2f} → log transform")
                vals_transformed = np.log1p(np.abs(vals)) * np.sign(vals)
            else:
                vals_transformed = vals
            
            vmin = vals_transformed.min()
            vmax = vals_transformed.max()
            
            self.static_params[col] = {
                'min': float(vmin),
                'max': float(vmax),
                'log': bool(use_log),
            }
            
            # Print normalization range
            if self.verbose:
                if use_log:
                    print(f"[INFO] Static {col}: transformed range=[{vmin:.4f}, {vmax:.4f}]")
                else:
                    print(f"[INFO] Static {col}: range=[{vmin:.4f}, {vmax:.4f}]")
    
    def init_dynamic_streaming(self, feature_cols: List[str], exclude_cols: List[str] = None):
        """Initialize streaming statistics for dynamic features."""
        exclude_cols = exclude_cols or []
        self.dynamic_features = [c for c in feature_cols if c not in exclude_cols]
        self._running_stats = {col: RunningMinMaxSkew() for col in self.dynamic_features}
    
    def update_dynamic_streaming(self, df: pd.DataFrame, exclude_cols: List[str] = None):
        """Update running statistics with a new dataframe (streaming)."""
        exclude_cols = exclude_cols or []
        for col in self.dynamic_features:
            if col in df.columns and col not in exclude_cols:
                vals = df[col].astype(float).values
                self._running_stats[col].update(vals)
    
    def finalize_dynamic_streaming(self, skew_threshold=2.0, meanstd_overrides=None):
        """Finalize dynamic normalization parameters after all updates.

        Args:
            skew_threshold: Skewness threshold for log transform.
            meanstd_overrides: Optional dict {feature: sigma} — for these features, use
                mean/std normalization with the data mean and the provided sigma value.
                This ensures sqrt(MSE_normalized) == NRMSE for Kaggle-aligned metrics.
        """
        meanstd_overrides = meanstd_overrides or {}
        for col in self.dynamic_features:
            stats = self._running_stats[col]

            if col in meanstd_overrides:
                # Mean/std normalization: center on data mean, scale by provided sigma
                self.dynamic_params[col] = {
                    'type': 'meanstd',
                    'mean': float(stats.M1),
                    'sigma': float(meanstd_overrides[col]),
                    'log': False,
                }
                if self.verbose:
                    print(f"[INFO] Dynamic {col}: meanstd mean={stats.M1:.3f}, sigma={meanstd_overrides[col]}")
                continue

            skew = stats.get_skewness()
            use_log = abs(skew) > skew_threshold
            if col == "water_level":
                use_log = False

            # Get min/max from raw data
            vmin, vmax = stats.min_val, stats.max_val

            # If using log transform, recompute min/max on transformed data
            if use_log:
                if self.verbose:
                    print(f"[INFO] Dynamic {col}: skew={skew:.2f} → log transform")
                # Signed log transform: sign(x) * log1p(abs(x))
                if vmin < 0:
                    vmin_transformed = -np.log1p(abs(vmin))
                    vmax_transformed = np.log1p(abs(vmax)) if vmax > 0 else -np.log1p(abs(vmax))
                else:
                    vmin_transformed = np.log1p(vmin)
                    vmax_transformed = np.log1p(vmax)
                vmin, vmax = vmin_transformed, vmax_transformed

            self.dynamic_params[col] = {
                'min': float(vmin),
                'max': float(vmax),
                'log': bool(use_log),
            }

            # Print normalization range
            if self.verbose:
                if use_log:
                    print(f"[INFO] Dynamic {col}: transformed range=[{vmin:.4f}, {vmax:.4f}]")
                else:
                    print(f"[INFO] Dynamic {col}: range=[{vmin:.4f}, {vmax:.4f}]")

        # Clear streaming stats to free memory
        self._running_stats = {}
    
    def fit_dynamic(self, dataframes: List[pd.DataFrame], feature_cols: List[str], 
                    exclude_cols: List[str] = None, skew_threshold=2.0):
        """
        Fit normalization parameters on dynamic features using streaming.
        Memory efficient - doesn't concatenate all data.
        
        Args:
            dataframes: List of dynamic feature dataframes
            feature_cols: Columns to normalize
            exclude_cols: Columns to exclude (e.g., ['water_volume', 'inlet_flow'])
            skew_threshold: Skewness threshold for log transform
        """
        self.init_dynamic_streaming(feature_cols, exclude_cols)
        for df in dataframes:
            self.update_dynamic_streaming(df, exclude_cols)
        self.finalize_dynamic_streaming(skew_threshold)
    
    def transform_static(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Transform static features using fitted parameters."""
        df = df.copy()
        for col in self.static_features:
            if col not in df.columns:
                continue
            vals = df[col].astype(float).values
            params = self.static_params[col]

            mask = ~np.isnan(vals)
            if not mask.any():
                df[col] = vals.astype(np.float32)
                continue

            vals_out = vals.copy()
            
            if params['log']:
                vals_out[mask] = np.log1p(np.abs(vals_out[mask])) * np.sign(vals_out[mask])
            
            vmin, vmax = params['min'], params['max']
            if vmax > vmin:
                vals_out[mask] = (vals_out[mask] - vmin) / (vmax - vmin)
            else:
                vals_out[mask] = 0.0
            
            # Convert column to float to avoid dtype errors
            df[col] = vals_out.astype(np.float32)
        
        return df
    
    def transform_dynamic(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """Transform dynamic features using fitted parameters."""
        df = df.copy()
        exclude_cols = exclude_cols or []
        
        for col in self.dynamic_features:
            if col not in df.columns or col in exclude_cols:
                continue
            vals = df[col].astype(float).values
            params = self.dynamic_params[col]

            mask = ~np.isnan(vals)
            if not mask.any():
                df[col] = vals.astype(np.float32)
                continue

            vals_out = vals.copy()

            if params.get('type', 'minmax') == 'meanstd':
                vals_out[mask] = (vals_out[mask] - params['mean']) / params['sigma']
            else:
                if params['log']:
                    vals_out[mask] = np.log1p(np.abs(vals_out[mask])) * np.sign(vals_out[mask])
                vmin, vmax = params['min'], params['max']
                if vmax > vmin:
                    vals_out[mask] = (vals_out[mask] - vmin) / (vmax - vmin)
                else:
                    vals_out[mask] = 0.0

            # Convert column to float to avoid dtype errors
            df[col] = vals_out.astype(np.float32)

        return df

    def unnormalize(self, vals: torch.Tensor, feature_name: str, feature_type: str) -> torch.Tensor:
        """
        Unnormalize a feature.
        
        Args:
            vals: Normalized values [0, 1]
            feature_name: Name of the feature
            feature_type: 'static' or 'dynamic'
        """
        params_dict = self.static_params if feature_type == 'static' else self.dynamic_params
        if feature_name not in params_dict:
            return vals
        
        params = params_dict[feature_name]

        if params.get('type', 'minmax') == 'meanstd':
            return vals * params['sigma'] + params['mean']

        vmin, vmax = params['min'], params['max']
        # Reverse min-max
        vals_scaled = vals * (vmax - vmin) + vmin
        # Reverse log transform
        if params['log']:
            vals_scaled = torch.sign(vals_scaled) * (torch.exp(torch.abs(vals_scaled)) - 1)
        return vals_scaled
    
    def get_params_dict(self) -> Dict:
        """Get all parameters for saving/loading."""
        return {
            'static_params': self.static_params,
            'dynamic_params': self.dynamic_params,
            'static_features': self.static_features,
            'dynamic_features': self.dynamic_features,
        }
    
    def load_params_dict(self, params_dict: Dict):
        """Load parameters from dict."""
        self.static_params = params_dict['static_params']
        self.dynamic_params = params_dict['dynamic_params']
        self.static_features = params_dict['static_features']
        self.dynamic_features = params_dict['dynamic_features']
