import os
import numpy as np
import pandas as pd
import torch

from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Dict, List, Optional, Tuple

from .dataset_normalizer import DatasetNormalizer
from utils import file_utils


class CsvBoundaryCondition:
    def __init__(self, num_nodes: int, num_edges: int):
        self.boundary_nodes_mask = np.zeros(num_nodes, dtype=bool)
        self.boundary_edges_mask = np.zeros(num_edges, dtype=bool)
        self.inflow_edges_mask = np.zeros(num_edges, dtype=bool)
        self.outflow_edges_mask = np.zeros(num_edges, dtype=bool)


class CsvFloodDataset(Dataset):
    STATIC_NODE_FEATURES = [
        'position_x',
        'position_y',
        'depth',
        'invert_elevation',
        'surface_elevation',
        'base_area',
        'area',
        'roughness',
        'min_elevation',
        'centroid_elevation',
        'aspect',
        'curvature',
        'flow_accumulation',
        'node_type',
    ]
    DYNAMIC_NODE_FEATURES = ['rainfall', 'water_level', 'water_volume', 'inlet_flow']
    STATIC_EDGE_FEATURES = [
        'relative_position_x',
        'relative_position_y',
        'face_length',
        'length',
        'slope',
        'diameter',
        'shape',
        'roughness',
        'edge_type',
    ]
    DYNAMIC_EDGE_FEATURES = ['flow', 'velocity']
    NODE_TARGET_FEATURE = 'water_level'
    EDGE_TARGET_FEATURE = 'flow'

    def __init__(self,
                 mode: str,
                 root_dir: str,
                 dataset_summary_file: Optional[str] = None,
                 file_prefix: str = '',
                 static_dir: Optional[str] = None,
                 previous_timesteps: int = 1,
                 warm_start_timesteps: Optional[int] = None,
                 warn_on_warm_start: bool = True,
                 normalize: bool = True,
                 features_stats_file: str = 'features_stats.yaml',
                 with_global_mass_loss: bool = False,
                 with_local_mass_loss: bool = False,
                 bidirectional_1d2d: bool = True,
                 build_data_list: bool = True,
                 mask_future_inputs: bool = False,
                 random_window_sampling: bool = False,
                 random_window_seed: int = 0,
                 node_type_filter: Optional[int] = None,
                 node_derived_features_enabled: bool = False,
                 node_derived_feature_windows: Optional[List[int]] = None,
                 node_derived_rain_windows: Optional[List[int]] = None,
                 node_derived_rain_max_windows: Optional[List[int]] = None,
                 node_derived_rain_decay_alphas: Optional[List[float]] = None,
                 node_derived_interactions_enabled: bool = False,
                 node_derived_interaction_static_features: Optional[List[str]] = None,
                 external_qhat_enabled: bool = False,
                 external_qhat_file: Optional[str] = None,
                 external_qhat_feature_name: str = 'external_qhat',
                 external_qhat_fill_value: float = 0.0,
                 debug: bool = False,
                 logger=None,
                 **kwargs):
        assert mode in ['train', 'test'], f'Invalid mode: {mode}. Must be "train" or "test".'

        self.mode = mode
        self.root_dir = root_dir
        self.dataset_summary_file = dataset_summary_file
        self.file_prefix = file_prefix
        self.static_dir = static_dir
        self.previous_timesteps = previous_timesteps
        self.warm_start_timesteps = warm_start_timesteps
        self.warn_on_warm_start = bool(warn_on_warm_start)
        self.is_normalized = normalize
        self.features_stats_file = features_stats_file
        self.with_global_mass_loss = with_global_mass_loss
        self.with_local_mass_loss = with_local_mass_loss
        self.bidirectional_1d2d = bidirectional_1d2d
        self.build_data_list = build_data_list
        self._warm_start_warned = set()
        self._warm_start_summary_logged = False
        self.mask_future_inputs = bool(mask_future_inputs)
        self.random_window_sampling = bool(random_window_sampling)
        self.random_window_seed = int(random_window_seed or 0)
        self.random_window_epoch = 0
        self.node_type_filter = int(node_type_filter) if node_type_filter is not None else None
        self.node_derived_features_enabled = bool(node_derived_features_enabled)
        self.node_derived_feature_windows = tuple(node_derived_feature_windows or [1, 3, 5])
        self.node_derived_rain_windows = tuple(node_derived_rain_windows or [6, 10])
        self.node_derived_rain_max_windows = tuple(node_derived_rain_max_windows or [6, 12, 24])
        self.node_derived_rain_decay_alphas = tuple(node_derived_rain_decay_alphas or [0.9, 0.97, 0.99])
        self.node_derived_interactions_enabled = bool(node_derived_interactions_enabled)
        self.node_derived_interaction_static_features = tuple(
            node_derived_interaction_static_features or ['area', 'flow_accumulation', 'centroid_elevation']
        )
        self.num_node_derived_features = 0
        self._node_interaction_static_idx: List[int] = []
        self.external_qhat_enabled = bool(external_qhat_enabled)
        self.external_qhat_file = external_qhat_file
        self.external_qhat_feature_name = str(external_qhat_feature_name)
        self.external_qhat_fill_value = float(external_qhat_fill_value)

        self.log_func = print
        if logger is not None and hasattr(logger, 'log'):
            self.log_func = logger.log

        if not hasattr(self, 'num_label_timesteps'):
            self.num_label_timesteps = 1
        self.normalizer = None
        self.inlet_q_mean = 0.0
        self.inlet_q_std = 1.0
        self.netq_q_mean = 0.0
        self.netq_q_std = 1.0
        if self.is_normalized:
            self.normalizer = DatasetNormalizer(mode, root_dir, features_stats_file)

        self.events = self._load_event_list()
        self._load_static_graph()
        self._load_events_dynamic()
        self._apply_node_type_filter()
        self._attach_external_qhat()
        self._compute_netq_feature_stats_raw()

        if self.is_normalized:
            self._normalize_features()
            try:
                mu, sigma = self.normalizer.get_feature_mean_std('inlet_flow')
                self.inlet_q_mean = float(mu)
                self.inlet_q_std = float(sigma if sigma is not None and sigma > 0 else 1.0)
            except Exception:
                self.inlet_q_mean = 0.0
                self.inlet_q_std = 1.0
        self._apply_future_input_mask()
        self._cache_event_derived_features()

        self.boundary_condition = CsvBoundaryCondition(self.num_nodes, self.num_edges)
        self.data_list = self._build_data_list() if self.build_data_list else None

        super().__init__(root_dir, transform=None, pre_transform=None, pre_filter=None, log=debug)

    def _apply_node_type_filter(self) -> None:
        """
        Optionally filter graph/data to a single Kaggle node type:
        - 1: original 1D nodes
        - 2: original 2D nodes
        """
        if self.node_type_filter is None:
            return
        if self.node_type_filter not in (1, 2):
            raise ValueError(f'Invalid node_type_filter={self.node_type_filter}. Expected 1 or 2.')

        keep_nodes_mask = (self.kaggle_node_types == self.node_type_filter)
        keep_nodes_idx = np.where(keep_nodes_mask)[0]
        if keep_nodes_idx.size == 0:
            raise ValueError(f'node_type_filter={self.node_type_filter} selected zero nodes.')

        edge_index_np = self.edge_index.cpu().numpy()
        keep_edges_mask = keep_nodes_mask[edge_index_np[0]] & keep_nodes_mask[edge_index_np[1]]
        keep_edges_idx = np.where(keep_edges_mask)[0]

        old_to_new = np.full((self.kaggle_node_types.shape[0],), -1, dtype=np.int64)
        old_to_new[keep_nodes_idx] = np.arange(keep_nodes_idx.size, dtype=np.int64)
        filtered_edge_index = edge_index_np[:, keep_edges_idx]
        filtered_edge_index = old_to_new[filtered_edge_index]

        # Filter static graph tensors.
        self.static_nodes = self.static_nodes[keep_nodes_idx]
        self.static_nodes_raw = self.static_nodes_raw[keep_nodes_idx]
        self.node_area = self.node_area[keep_nodes_idx]
        self.node_base_area = self.node_base_area[keep_nodes_idx]
        self.static_edges = self.static_edges[keep_edges_idx]
        self.edge_index = torch.from_numpy(filtered_edge_index.copy())

        # Filter Kaggle ids/types metadata.
        self.kaggle_node_ids = self.kaggle_node_ids[keep_nodes_idx]
        self.kaggle_node_types = self.kaggle_node_types[keep_nodes_idx]

        # Filter dynamic event tensors.
        for event in self.events_dynamic:
            for key in ('dynamic_nodes', 'dynamic_nodes_input', 'dynamic_nodes_raw', 'external_qhat_input'):
                if key in event:
                    event[key] = event[key][:, keep_nodes_idx, :]
            for key in ('dynamic_edges', 'dynamic_edges_input', 'dynamic_edges_raw'):
                if key in event:
                    event[key] = event[key][:, keep_edges_idx, :]
            if 'node_rainfall_per_ts' in event:
                event['node_rainfall_per_ts'] = event['node_rainfall_per_ts'][:, keep_nodes_idx]

        # Update shape bookkeeping.
        self.num_nodes = int(keep_nodes_idx.size)
        self.num_edges = int(keep_edges_idx.size)
        self.num_nodes_1d = int(np.sum(self.kaggle_node_types == 1))
        self.num_nodes_2d = int(np.sum(self.kaggle_node_types == 2))

    def set_epoch(self, epoch: int) -> None:
        self.random_window_epoch = int(epoch or 0)

    def _get_random_window_offset(self, event_idx: int, idx: int, max_ts_offset: int) -> int:
        if max_ts_offset <= 0:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = (
            self.random_window_seed
            + self.random_window_epoch * 1000003
            + event_idx * 10007
            + idx * 97
            + worker_id * 1009
        ) % (2**32)
        rng = np.random.default_rng(seed)
        return int(rng.integers(0, max_ts_offset + 1))

    def _resolve_summary_path(self) -> Optional[str]:
        if self.dataset_summary_file is None:
            return None
        return file_utils.resolve_existing_path(
            self.dataset_summary_file,
            root_dir=self.root_dir,
            include_raw_fallback=True,
            label='CSV dataset summary file',
        )

    def _resolve_dir(self, dir_path: Optional[str]) -> Optional[str]:
        if dir_path is None:
            return None
        if os.path.isabs(dir_path):
            return dir_path
        return os.path.join(self.root_dir, dir_path)

    def _load_event_list(self) -> List[Dict[str, str]]:
        summary_path = self._resolve_summary_path()
        if summary_path is None:
            return [{'event_dir': '.', 'file_prefix': self.file_prefix}]

        summary_df = pd.read_csv(summary_path)
        if len(summary_df) == 0:
            raise ValueError(f'No events found in: {summary_path}')

        if 'event_dir' in summary_df.columns:
            event_dirs = summary_df['event_dir'].fillna('.').astype(str).tolist()
        elif 'event_path' in summary_df.columns:
            event_dirs = summary_df['event_path'].fillna('.').astype(str).tolist()
        else:
            raise ValueError(f'Expected an event_dir or event_path column in: {summary_path}')

        file_prefixes = None
        if 'file_prefix' in summary_df.columns:
            file_prefixes = summary_df['file_prefix'].fillna('').astype(str).tolist()

        event_ids = summary_df['event_id'].tolist() if 'event_id' in summary_df.columns else list(range(len(event_dirs)))
        static_dirs = None
        if 'static_dir' in summary_df.columns:
            static_dirs = summary_df['static_dir'].fillna(self.static_dir or '.').astype(str).tolist()

        events = []
        for i, event_dir in enumerate(event_dirs):
            prefix = self.file_prefix
            if file_prefixes is not None:
                prefix = file_prefixes[i]
            static_dir = self.static_dir or event_dir
            if static_dirs is not None:
                static_dir = static_dirs[i]
            events.append({
                'event_dir': event_dir,
                'file_prefix': prefix,
                'event_id': event_ids[i],
                'static_dir': static_dir,
            })
        return events

    def _read_csv(self, base_dir: str, filename: str) -> pd.DataFrame:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'CSV file not found: {path}')
        return pd.read_csv(path)

    def _check_warm_start_future_values(self,
                                        df: pd.DataFrame,
                                        columns: List[str],
                                        label: str,
                                        event_id) -> None:
        if (
            not self.warn_on_warm_start
            or self.warm_start_timesteps is None
            or self.mode != 'test'
        ):
            return
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing columns {missing_cols} in {label} for warm-start check.')

        timestep_col = None
        if 'timestep' in df.columns:
            timestep_col = 'timestep'
        elif 'timestep_idx' in df.columns:
            timestep_col = 'timestep_idx'
        else:
            raise ValueError(f'No timestep column found in {label} for warm-start check.')

        mask = df[timestep_col] >= self.warm_start_timesteps
        if not mask.any():
            return

        future = df.loc[mask, columns]
        non_nan = future.notna().sum().sum()
        if non_nan == 0:
            return

        warn_key = f'{event_id}:{label}'
        if warn_key in self._warm_start_warned:
            return
        self._warm_start_warned.add(warn_key)
        if not self._warm_start_summary_logged:
            self._warm_start_summary_logged = True
            self.log_func(
                'INFO: warm-start check detected non-NaN future values in raw dynamic CSVs; '
                f'future non-rain inputs are masked at timesteps >= {self.warm_start_timesteps}.'
            )

    def _load_static_graph(self) -> None:
        static_dir = self._resolve_dir(self.events[0].get('static_dir'))
        base_dir = static_dir if static_dir is not None else os.path.join(self.root_dir, self.events[0]['event_dir'])
        prefix = self.events[0]['file_prefix']

        df_1d_nodes = self._read_csv(base_dir, f'{prefix}1d_nodes_static.csv').sort_values('node_idx')
        df_2d_nodes = self._read_csv(base_dir, f'{prefix}2d_nodes_static.csv').sort_values('node_idx')

        df_1d_edges = self._read_csv(base_dir, f'{prefix}1d_edges_static.csv').sort_values('edge_idx')
        df_2d_edges = self._read_csv(base_dir, f'{prefix}2d_edges_static.csv').sort_values('edge_idx')

        df_1d_edge_index = self._read_csv(base_dir, f'{prefix}1d_edge_index.csv').sort_values('edge_idx')
        df_2d_edge_index = self._read_csv(base_dir, f'{prefix}2d_edge_index.csv').sort_values('edge_idx')
        df_1d2d_conn = self._read_csv(base_dir, f'{prefix}1d2d_connections.csv').sort_values('connection_idx')

        self.node_idx_map_1d = {idx: i for i, idx in enumerate(df_1d_nodes['node_idx'].tolist())}
        self.node_idx_map_2d = {idx: i for i, idx in enumerate(df_2d_nodes['node_idx'].tolist())}
        self.edge_idx_map_1d = {idx: i for i, idx in enumerate(df_1d_edges['edge_idx'].tolist())}
        self.edge_idx_map_2d = {idx: i for i, idx in enumerate(df_2d_edges['edge_idx'].tolist())}

        num_nodes_1d = len(df_1d_nodes)
        num_nodes_2d = len(df_2d_nodes)
        self.num_nodes_1d = num_nodes_1d
        self.num_nodes_2d = num_nodes_2d
        self.num_nodes = num_nodes_1d + num_nodes_2d

        static_nodes = np.zeros((self.num_nodes, len(self.STATIC_NODE_FEATURES)), dtype=np.float32)
        node_feat_idx = {name: i for i, name in enumerate(self.STATIC_NODE_FEATURES)}

        static_nodes[:num_nodes_1d, node_feat_idx['position_x']] = df_1d_nodes['position_x'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['position_y']] = df_1d_nodes['position_y'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['depth']] = df_1d_nodes['depth'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['invert_elevation']] = df_1d_nodes['invert_elevation'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['surface_elevation']] = df_1d_nodes['surface_elevation'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['base_area']] = df_1d_nodes['base_area'].to_numpy()
        static_nodes[:num_nodes_1d, node_feat_idx['node_type']] = 0.0

        node_2d_offset = num_nodes_1d
        static_nodes[node_2d_offset:, node_feat_idx['position_x']] = df_2d_nodes['position_x'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['position_y']] = df_2d_nodes['position_y'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['area']] = df_2d_nodes['area'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['roughness']] = df_2d_nodes['roughness'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['min_elevation']] = df_2d_nodes['min_elevation'].to_numpy()
        elevation_col = 'centroid_elevation'
        if elevation_col not in df_2d_nodes.columns:
            elevation_col = 'elevation'
        static_nodes[node_2d_offset:, node_feat_idx['centroid_elevation']] = df_2d_nodes[elevation_col].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['aspect']] = df_2d_nodes['aspect'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['curvature']] = df_2d_nodes['curvature'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['flow_accumulation']] = df_2d_nodes['flow_accumulation'].to_numpy()
        static_nodes[node_2d_offset:, node_feat_idx['node_type']] = 1.0

        self.kaggle_node_ids = np.concatenate([
            df_1d_nodes['node_idx'].to_numpy(dtype=np.int64),
            df_2d_nodes['node_idx'].to_numpy(dtype=np.int64),
        ])
        self.kaggle_node_types = np.concatenate([
            np.full(num_nodes_1d, 1, dtype=np.int64),
            np.full(num_nodes_2d, 2, dtype=np.int64),
        ])

        static_nodes = np.nan_to_num(static_nodes, nan=0.0, posinf=0.0, neginf=0.0)
        node_positions = static_nodes[:, [node_feat_idx['position_x'], node_feat_idx['position_y']]]
        # Keep raw static nodes for physics calculations (rainfall volume scaling).
        self.static_nodes_raw = static_nodes.copy()
        self.node_area = static_nodes[:, node_feat_idx['area']].copy()
        self.node_base_area = static_nodes[:, node_feat_idx['base_area']].copy()

        edge_feat_idx = {name: i for i, name in enumerate(self.STATIC_EDGE_FEATURES)}
        edge_index_1d = np.stack([
            df_1d_edge_index['from_node'].map(self.node_idx_map_1d).to_numpy(),
            df_1d_edge_index['to_node'].map(self.node_idx_map_1d).to_numpy(),
        ], axis=0)

        edge_index_2d = np.stack([
            df_2d_edge_index['from_node'].map(self.node_idx_map_2d).to_numpy() + node_2d_offset,
            df_2d_edge_index['to_node'].map(self.node_idx_map_2d).to_numpy() + node_2d_offset,
        ], axis=0)

        edge_index_1d2d = []
        for _, row in df_1d2d_conn.iterrows():
            n1 = self.node_idx_map_1d[row['node_1d']]
            n2 = self.node_idx_map_2d[row['node_2d']] + node_2d_offset
            edge_index_1d2d.append([n1, n2])
            if self.bidirectional_1d2d:
                edge_index_1d2d.append([n2, n1])
        edge_index_1d2d = np.array(edge_index_1d2d, dtype=np.int64).T if len(edge_index_1d2d) > 0 else np.empty((2, 0), dtype=np.int64)

        edge_index = np.concatenate([edge_index_1d, edge_index_2d, edge_index_1d2d], axis=1)
        self.edge_index = torch.from_numpy(edge_index.copy())

        static_edges_1d = np.zeros((len(df_1d_edges), len(self.STATIC_EDGE_FEATURES)), dtype=np.float32)
        static_edges_1d[:, edge_feat_idx['relative_position_x']] = df_1d_edges['relative_position_x'].to_numpy()
        static_edges_1d[:, edge_feat_idx['relative_position_y']] = df_1d_edges['relative_position_y'].to_numpy()
        static_edges_1d[:, edge_feat_idx['length']] = df_1d_edges['length'].to_numpy()
        static_edges_1d[:, edge_feat_idx['diameter']] = df_1d_edges['diameter'].to_numpy()
        static_edges_1d[:, edge_feat_idx['shape']] = df_1d_edges['shape'].to_numpy()
        static_edges_1d[:, edge_feat_idx['roughness']] = df_1d_edges['roughness'].to_numpy()
        static_edges_1d[:, edge_feat_idx['slope']] = df_1d_edges['slope'].to_numpy()
        static_edges_1d[:, edge_feat_idx['edge_type']] = 0.0

        static_edges_2d = np.zeros((len(df_2d_edges), len(self.STATIC_EDGE_FEATURES)), dtype=np.float32)
        static_edges_2d[:, edge_feat_idx['relative_position_x']] = df_2d_edges['relative_position_x'].to_numpy()
        static_edges_2d[:, edge_feat_idx['relative_position_y']] = df_2d_edges['relative_position_y'].to_numpy()
        static_edges_2d[:, edge_feat_idx['face_length']] = df_2d_edges['face_length'].to_numpy()
        static_edges_2d[:, edge_feat_idx['length']] = df_2d_edges['length'].to_numpy()
        static_edges_2d[:, edge_feat_idx['slope']] = df_2d_edges['slope'].to_numpy()
        static_edges_2d[:, edge_feat_idx['edge_type']] = 1.0

        edge_index_1d2d_count = edge_index_1d2d.shape[1]
        static_edges_1d2d = np.zeros((edge_index_1d2d_count, len(self.STATIC_EDGE_FEATURES)), dtype=np.float32)
        for i in range(edge_index_1d2d_count):
            from_idx = edge_index_1d2d[0, i]
            to_idx = edge_index_1d2d[1, i]
            dx, dy = node_positions[to_idx] - node_positions[from_idx]
            static_edges_1d2d[i, edge_feat_idx['relative_position_x']] = dx
            static_edges_1d2d[i, edge_feat_idx['relative_position_y']] = dy
            static_edges_1d2d[i, edge_feat_idx['length']] = np.sqrt(dx * dx + dy * dy)
            static_edges_1d2d[i, edge_feat_idx['edge_type']] = 2.0

        self.static_nodes = static_nodes
        static_edges = np.concatenate([static_edges_1d, static_edges_2d, static_edges_1d2d], axis=0)
        static_edges = np.nan_to_num(static_edges, nan=0.0, posinf=0.0, neginf=0.0)
        self.static_edges = static_edges

        self.num_static_node_features = self.static_nodes.shape[1]
        self._static_node_feat_idx = {name: i for i, name in enumerate(self.STATIC_NODE_FEATURES)}
        if self.node_derived_features_enabled:
            rain_mem_count = (
                len(self.node_derived_rain_windows)
                + len(self.node_derived_rain_max_windows)
                + len(self.node_derived_rain_decay_alphas)
            )
            interaction_count = 0
            if self.node_derived_interactions_enabled:
                self._node_interaction_static_idx = [
                    self._static_node_feat_idx[name]
                    for name in self.node_derived_interaction_static_features
                    if name in self._static_node_feat_idx
                ]
                interaction_count = rain_mem_count * len(self._node_interaction_static_idx)
            # Derived node features are appended between static and dynamic blocks.
            self.num_node_derived_features = (
                len(self.node_derived_feature_windows)  # dy windows
                + len(self.node_derived_feature_windows)  # std windows
                + rain_mem_count  # rain sums + decays
                + interaction_count  # rain-memory x static interactions
            )
            self.num_static_node_features += self.num_node_derived_features
        self.num_static_edge_features = self.static_edges.shape[1]
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        self.num_dynamic_edge_features = len(self.DYNAMIC_EDGE_FEATURES)
        self.num_edges = self.static_edges.shape[0]

    def _load_timesteps(self, base_dir: str, prefix: str) -> ndarray:
        df = self._read_csv(base_dir, f'{prefix}timesteps.csv')
        if 'timestep_idx' not in df.columns:
            raise ValueError(f'Expected timestep_idx in timesteps file at {base_dir}')
        timesteps = df['timestep_idx'].to_numpy()
        return timesteps

    def _build_dynamic_tensor(self,
                              df: pd.DataFrame,
                              timesteps: ndarray,
                              id_col: str,
                              id_map: Dict[int, int],
                              feature_cols: List[str]) -> ndarray:
        num_timesteps = len(timesteps)
        num_items = len(id_map)
        out = np.zeros((num_timesteps, num_items, len(feature_cols)), dtype=np.float32)
        if len(df) == 0:
            return out

        timestep_col = 'timestep' if 'timestep' in df.columns else 'timestep_idx'
        timestep_to_idx = {t: i for i, t in enumerate(timesteps)}

        timestep_idx = df[timestep_col].map(timestep_to_idx).to_numpy()
        item_idx = df[id_col].map(id_map).to_numpy()
        if np.any(pd.isna(timestep_idx)) or np.any(pd.isna(item_idx)):
            raise ValueError('Found unknown timestep or item index while building dynamic tensor.')

        timestep_idx = timestep_idx.astype(np.int64)
        item_idx = item_idx.astype(np.int64)
        for f_i, col in enumerate(feature_cols):
            out[timestep_idx, item_idx, f_i] = df[col].to_numpy()
        return out

    def _load_events_dynamic(self) -> None:
        self.events_dynamic = []
        self.event_timesteps = []
        self.event_start_idx = []
        self.hec_ras_run_ids = []
        self.timestep_interval = 1

        label_horizon = self.num_label_timesteps
        if getattr(self, 'dynamic_label_horizon', False):
            label_horizon = 1
        total_rollout = 0

        for event in self.events:
            base_dir = self._resolve_dir(event['event_dir'])
            prefix = event['file_prefix']
            event_id = event.get('event_id', len(self.hec_ras_run_ids))

            timesteps = self._load_timesteps(base_dir, prefix)
            if self.timestep_interval == 1:
                ts_file = self._read_csv(base_dir, f'{prefix}timesteps.csv')
                if 'timestamp' in ts_file.columns and len(ts_file) > 1:
                    ts_vals = pd.to_datetime(ts_file['timestamp'])
                    self.timestep_interval = int((ts_vals.iloc[1] - ts_vals.iloc[0]).total_seconds())

            df_1d_dyn_raw = self._read_csv(base_dir, f'{prefix}1d_nodes_dynamic_all.csv')
            df_2d_dyn_raw = self._read_csv(base_dir, f'{prefix}2d_nodes_dynamic_all.csv')
            df_1d_edge_dyn_raw = self._read_csv(base_dir, f'{prefix}1d_edges_dynamic_all.csv')
            df_2d_edge_dyn_raw = self._read_csv(base_dir, f'{prefix}2d_edges_dynamic_all.csv')

            self._check_warm_start_future_values(
                df_1d_dyn_raw,
                ['water_level', 'inlet_flow'],
                f'{prefix}1d_nodes_dynamic_all.csv',
                event_id,
            )
            self._check_warm_start_future_values(
                df_2d_dyn_raw,
                ['water_level', 'water_volume'],
                f'{prefix}2d_nodes_dynamic_all.csv',
                event_id,
            )
            self._check_warm_start_future_values(
                df_1d_edge_dyn_raw,
                ['flow', 'velocity'],
                f'{prefix}1d_edges_dynamic_all.csv',
                event_id,
            )
            self._check_warm_start_future_values(
                df_2d_edge_dyn_raw,
                ['flow', 'velocity'],
                f'{prefix}2d_edges_dynamic_all.csv',
                event_id,
            )

            df_1d_dyn = df_1d_dyn_raw.fillna(0)
            df_2d_dyn = df_2d_dyn_raw.fillna(0)
            df_1d_edge_dyn = df_1d_edge_dyn_raw.fillna(0)
            df_2d_edge_dyn = df_2d_edge_dyn_raw.fillna(0)

            dyn_1d_nodes = self._build_dynamic_tensor(
                df_1d_dyn,
                timesteps,
                id_col='node_idx',
                id_map=self.node_idx_map_1d,
                feature_cols=['water_level', 'inlet_flow'],
            )
            dyn_2d_nodes = self._build_dynamic_tensor(
                df_2d_dyn,
                timesteps,
                id_col='node_idx',
                id_map=self.node_idx_map_2d,
                feature_cols=['rainfall', 'water_level', 'water_volume'],
            )

            dynamic_nodes = np.zeros((len(timesteps), self.num_nodes, len(self.DYNAMIC_NODE_FEATURES)), dtype=np.float32)
            node_feat_idx = {name: i for i, name in enumerate(self.DYNAMIC_NODE_FEATURES)}
            dynamic_nodes[:, :len(self.node_idx_map_1d), node_feat_idx['water_level']] = dyn_1d_nodes[:, :, 0]
            dynamic_nodes[:, :len(self.node_idx_map_1d), node_feat_idx['inlet_flow']] = dyn_1d_nodes[:, :, 1]
            node_2d_offset = len(self.node_idx_map_1d)
            dynamic_nodes[:, node_2d_offset:, node_feat_idx['rainfall']] = dyn_2d_nodes[:, :, 0]
            dynamic_nodes[:, node_2d_offset:, node_feat_idx['water_level']] = dyn_2d_nodes[:, :, 1]
            dynamic_nodes[:, node_2d_offset:, node_feat_idx['water_volume']] = dyn_2d_nodes[:, :, 2]

            dyn_1d_edges = self._build_dynamic_tensor(
                df_1d_edge_dyn,
                timesteps,
                id_col='edge_idx',
                id_map=self.edge_idx_map_1d,
                feature_cols=['flow', 'velocity'],
            )
            dyn_2d_edges = self._build_dynamic_tensor(
                df_2d_edge_dyn,
                timesteps,
                id_col='edge_idx',
                id_map=self.edge_idx_map_2d,
                feature_cols=['flow', 'velocity'],
            )

            edge_1d_count = dyn_1d_edges.shape[1]
            edge_2d_count = dyn_2d_edges.shape[1]
            edge_1d2d_count = self.num_edges - edge_1d_count - edge_2d_count
            dyn_1d2d_edges = np.zeros((len(timesteps), edge_1d2d_count, len(self.DYNAMIC_EDGE_FEATURES)), dtype=np.float32)

            dynamic_edges = np.concatenate([dyn_1d_edges, dyn_2d_edges, dyn_1d2d_edges], axis=1)

            self.event_timesteps.append(timesteps)
            dynamic_nodes_input = dynamic_nodes.copy()
            dynamic_edges_input = dynamic_edges.copy()

            self.events_dynamic.append({
                'dynamic_nodes': dynamic_nodes,
                'dynamic_edges': dynamic_edges,
                'dynamic_nodes_input': dynamic_nodes_input,
                'dynamic_edges_input': dynamic_edges_input,
                'dynamic_nodes_raw': dynamic_nodes.copy(),
                'dynamic_edges_raw': dynamic_edges.copy(),
            })
            self.hec_ras_run_ids.append(event.get('event_id', len(self.hec_ras_run_ids)))

            num_timesteps = len(timesteps)
            event_total_rollout = num_timesteps - self.previous_timesteps - label_horizon
            if event_total_rollout <= 0:
                raise ValueError('Event has too few timesteps for the configured previous/label horizons.')

            self.event_start_idx.append(total_rollout)
            total_rollout += event_total_rollout

        self.total_rollout_timesteps = total_rollout

    def _normalize_features(self) -> None:
        if self.normalizer is None:
            return

        if self.mode == 'train':
            all_dyn_nodes = np.concatenate([e['dynamic_nodes'] for e in self.events_dynamic], axis=0)
            all_dyn_edges = np.concatenate([e['dynamic_edges'] for e in self.events_dynamic], axis=0)
            self._compute_feature_stats(self.STATIC_NODE_FEATURES, self.static_nodes)
            self._compute_feature_stats(self.DYNAMIC_NODE_FEATURES, all_dyn_nodes)
            self._compute_feature_stats(self.STATIC_EDGE_FEATURES, self.static_edges)
            self._compute_feature_stats(self.DYNAMIC_EDGE_FEATURES, all_dyn_edges)
            self.normalizer.save_feature_stats()

        self.static_nodes = self._apply_normalization(self.STATIC_NODE_FEATURES, self.static_nodes)
        self.static_edges = self._apply_normalization(self.STATIC_EDGE_FEATURES, self.static_edges)
        for event_idx, event in enumerate(self.events_dynamic):
            event['dynamic_nodes'] = self._apply_normalization(self.DYNAMIC_NODE_FEATURES, event['dynamic_nodes'])
            event['dynamic_edges'] = self._apply_normalization(self.DYNAMIC_EDGE_FEATURES, event['dynamic_edges'])
            event['dynamic_nodes_input'] = event['dynamic_nodes'].copy()
            event['dynamic_edges_input'] = event['dynamic_edges'].copy()

    def _compute_netq_series(self, node_dynamic_features: ndarray, edge_dynamic_features: ndarray) -> ndarray:
        inlet_idx = self.DYNAMIC_NODE_FEATURES.index('inlet_flow')
        edge_idx = self.DYNAMIC_EDGE_FEATURES.index('flow')
        inlet = node_dynamic_features[:, :, inlet_idx]
        edge_flow = edge_dynamic_features[:, :, edge_idx]
        netq = inlet.copy()
        edge_index_np = self.edge_index.cpu().numpy()
        src = edge_index_np[0]
        dst = edge_index_np[1]
        for t in range(edge_flow.shape[0]):
            ef = edge_flow[t]
            np.add.at(netq[t], dst, ef)
            np.add.at(netq[t], src, -ef)
        return netq.astype(np.float32, copy=False)

    def _compute_netq_feature_stats_raw(self) -> None:
        if not self.is_normalized:
            self.netq_q_mean = 0.0
            self.netq_q_std = 1.0
            return
        # Reuse training split stats in test mode when available to avoid train/val scaling mismatch.
        if self.normalizer is not None and self.mode == 'test':
            st = self.normalizer.feature_stats.get('netq_flow', None)
            if isinstance(st, dict) and ('mean' in st) and ('std' in st):
                mu = float(st.get('mean', 0.0))
                sd = float(st.get('std', 1.0))
                self.netq_q_mean = mu if np.isfinite(mu) else 0.0
                self.netq_q_std = sd if (np.isfinite(sd) and sd > 1e-8) else 1.0
                return
        vals = []
        for event in getattr(self, "events_dynamic", []):
            dn = event.get('dynamic_nodes_raw', event.get('dynamic_nodes'))
            de = event.get('dynamic_edges_raw', event.get('dynamic_edges'))
            if dn is None or de is None:
                continue
            nq = self._compute_netq_series(dn, de)
            if nq.size > 0:
                vals.append(nq.reshape(-1))
        if len(vals) == 0:
            self.netq_q_mean = 0.0
            self.netq_q_std = 1.0
            return
        all_v = np.concatenate(vals, axis=0)
        self.netq_q_mean = float(np.mean(all_v))
        std = float(np.std(all_v))
        self.netq_q_std = float(std if std > 1e-8 else 1.0)
        if self.normalizer is not None and self.mode == 'train':
            self.normalizer.feature_stats['netq_flow'] = {
                'mean': float(self.netq_q_mean),
                'std': float(self.netq_q_std),
                'min': float(np.min(all_v)),
                'max': float(np.max(all_v)),
            }

    def _normalize_netq(self, netq: ndarray) -> ndarray:
        if not self.is_normalized:
            return netq
        return (netq - self.netq_q_mean) / (self.netq_q_std + 1e-8)

    def _should_mask_future_inputs(self) -> bool:
        if self.warm_start_timesteps is None:
            return False
        return self.mode == 'test' or self.mask_future_inputs

    def _apply_future_input_mask(self) -> None:
        if not self._should_mask_future_inputs():
            return
        node_feat_idx = {name: i for i, name in enumerate(self.DYNAMIC_NODE_FEATURES)}
        for event_idx, event in enumerate(self.events_dynamic):
            mask_idx = np.arange(len(self.event_timesteps[event_idx])) >= self.warm_start_timesteps
            if not mask_idx.any():
                continue
            event['dynamic_nodes_input'][mask_idx, :, node_feat_idx['water_level']] = 0
            event['dynamic_nodes_input'][mask_idx, :, node_feat_idx['water_volume']] = 0
            event['dynamic_nodes_input'][mask_idx, :, node_feat_idx['inlet_flow']] = 0
            event['dynamic_edges_input'][mask_idx, :, :] = 0

    def _cache_event_derived_features(self) -> None:
        """
        Cache per-event derived tensors that are expensive to recompute in get().
        """
        for event in self.events_dynamic:
            dynamic_nodes_input = event.get('dynamic_nodes_input', event['dynamic_nodes'])
            event['node_rainfall_per_ts'] = self._get_rainfall_volume_per_ts(dynamic_nodes_input)

    def _effective_dynamic_node_features(self) -> List[str]:
        feats = list(self.DYNAMIC_NODE_FEATURES)
        if self.external_qhat_enabled:
            feats.append(self.external_qhat_feature_name)
        return feats

    def _load_external_qhat_table(self) -> Optional[pd.DataFrame]:
        if (not self.external_qhat_enabled) or (self.external_qhat_file is None):
            return None
        pth = os.path.abspath(self.external_qhat_file)
        if not os.path.exists(pth):
            self.log_func(f"WARN: external_qhat_file not found: {pth}; disabling external qhat.")
            self.external_qhat_enabled = False
            return None
        if pth.endswith('.csv') or pth.endswith('.csv.gz'):
            df = pd.read_csv(pth)
        elif pth.endswith('.parquet'):
            df = pd.read_parquet(pth)
        elif pth.endswith('.npz'):
            npz = np.load(pth)
            keys = set(npz.files)
            def pick(*names):
                for n in names:
                    if n in keys:
                        return npz[n]
                return None
            ev = pick('event_id', 'hec_ras_run_id', 'event')
            nd = pick('node_id', 'node_idx', 'node')
            ts = pick('timestep', 'timestep_idx', 'time', 't_rel')
            qh = pick('qhat', 'pred', 'prediction', 'y_pred')
            if ev is None or nd is None or ts is None or qh is None:
                raise ValueError('npz external_qhat_file must contain event/node/time/qhat arrays.')
            df = pd.DataFrame({'event_id': ev, 'node_id': nd, 'timestep': ts, 'qhat': qh})
        else:
            raise ValueError(f'Unsupported external_qhat_file format: {pth}')
        return df

    def _attach_external_qhat(self) -> None:
        # Keep base count unless we successfully attach an extra channel.
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        if not self.external_qhat_enabled:
            return
        df = self._load_external_qhat_table()
        if df is None or len(df) == 0:
            self.external_qhat_enabled = False
            return

        cols = {c.lower(): c for c in df.columns}
        def cfind(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        ev_col = cfind('event_id', 'hec_ras_run_id', 'event')
        node_col = cfind('node_id', 'node_idx', 'node')
        t_col = cfind('timestep', 'timestep_idx', 'time', 't_rel')
        q_col = cfind('qhat', 'pred', 'prediction', 'y_pred')
        if ev_col is None or node_col is None or t_col is None or q_col is None:
            raise ValueError('external_qhat_file requires event/node/time/qhat columns.')

        node_id_to_local = {int(nid): i for i, nid in enumerate(self.kaggle_node_ids.tolist())}
        warm = int(self.warm_start_timesteps or 10)

        # Collect mapped values for optional normalization stats.
        mapped_vals = []
        for event_idx, event in enumerate(self.events_dynamic):
            ev_id = int(self.hec_ras_run_ids[event_idx])
            timesteps = self.event_timesteps[event_idx]
            t_map = {int(t): i for i, t in enumerate(timesteps.tolist())}
            q_arr = np.full((len(timesteps), self.num_nodes, 1), self.external_qhat_fill_value, dtype=np.float32)

            sub = df[df[ev_col] == ev_id]
            if len(sub) > 0:
                for row in sub.itertuples(index=False):
                    node_id = int(getattr(row, node_col))
                    if node_id not in node_id_to_local:
                        continue
                    tl = getattr(row, t_col)
                    if t_col.lower() == 't_rel':
                        ts_idx = warm + int(tl)
                    else:
                        # Accept either true timestep values or direct indices.
                        tl_i = int(tl)
                        if tl_i in t_map:
                            ts_idx = t_map[tl_i]
                        else:
                            ts_idx = tl_i
                    if ts_idx < 0 or ts_idx >= len(timesteps):
                        continue
                    qv = float(getattr(row, q_col))
                    q_arr[ts_idx, node_id_to_local[node_id], 0] = qv

            event['external_qhat_input'] = q_arr
            mapped_vals.append(q_arr.reshape(-1))

        # Optional normalize external qhat to train stats.
        if self.is_normalized and self.normalizer is not None:
            feat_name = self.external_qhat_feature_name
            if self.mode == 'train':
                all_v = np.concatenate(mapped_vals, axis=0) if len(mapped_vals) > 0 else np.zeros((1,), dtype=np.float32)
                mu = float(np.mean(all_v))
                sd = float(np.std(all_v))
                if sd <= 1e-8:
                    sd = 1.0
                self.normalizer.feature_stats[feat_name] = {
                    'mean': mu,
                    'std': sd,
                    'min': float(np.min(all_v)),
                    'max': float(np.max(all_v)),
                }
            else:
                st = self.normalizer.feature_stats.get(feat_name)
                if st is None:
                    self.log_func(f"WARN: missing feature stats for {feat_name}; using identity normalization.")
                    st = {'mean': 0.0, 'std': 1.0}
                mu = float(st.get('mean', 0.0))
                sd = float(st.get('std', 1.0))
                if sd <= 1e-8:
                    sd = 1.0
            for event in self.events_dynamic:
                q = event.get('external_qhat_input')
                if q is None:
                    continue
                event['external_qhat_input'] = ((q - mu) / (sd + 1e-8)).astype(np.float32, copy=False)

        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES) + 1
        self.log_func(f"INFO: external qhat enabled from {self.external_qhat_file}; dynamic_node_features={self.num_dynamic_node_features}.")

    def _compute_feature_stats(self, feature_list: List[str], feature_vector: ndarray) -> None:
        for i, feature in enumerate(feature_list):
            data = feature_vector[..., i]
            mean = data.mean().item()
            std = data.std().item()
            min_val = data.min().item()
            max_val = data.max().item()
            self.normalizer.feature_stats[feature] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
            }

    def _apply_normalization(self, feature_list: List[str], feature_vector: ndarray) -> ndarray:
        normed = feature_vector.copy()
        for i, feature in enumerate(feature_list):
            mean, std = self.normalizer.get_feature_mean_std(feature)
            normed[..., i] = self.normalizer.normalize(normed[..., i], mean, std)
        return normed

    def _get_timestep_dynamic_features(self, dynamic_features: ndarray, dynamic_feature_list: List[str], timestep_idx: int, apply_warm_mask: bool = True) -> Tensor:
        _, num_elems, _ = dynamic_features.shape
        if timestep_idx < self.previous_timesteps:
            padding = self._get_empty_feature_tensor(
                dynamic_feature_list,
                (self.previous_timesteps - timestep_idx, num_elems),
                dtype=dynamic_features.dtype,
            )
            ts_dynamic_features = np.concatenate([padding, dynamic_features[:timestep_idx+1, :, :]], axis=0)
        else:
            ts_dynamic_features = dynamic_features[timestep_idx-self.previous_timesteps:timestep_idx+1, :, :]
        if self.warm_start_timesteps is None or (not apply_warm_mask):
            return ts_dynamic_features

        if timestep_idx < self.previous_timesteps:
            padding_len = self.previous_timesteps - timestep_idx
            data_indices = np.arange(timestep_idx + 1)
            window_indices = np.concatenate([np.full(padding_len, -1), data_indices])
        else:
            window_start = timestep_idx - self.previous_timesteps
            window_indices = np.arange(window_start, timestep_idx + 1)

        mask = window_indices >= self.warm_start_timesteps
        if mask.any():
            ts_dynamic_features = ts_dynamic_features.copy()
            if 'rainfall' in dynamic_feature_list:
                rainfall_idx = dynamic_feature_list.index('rainfall')
                rainfall_vals = ts_dynamic_features[:, :, rainfall_idx].copy()
                ts_dynamic_features[mask, :, :] = 0
                ts_dynamic_features[mask, :, rainfall_idx] = rainfall_vals[mask, :]
            else:
                ts_dynamic_features[mask, :, :] = 0

        return ts_dynamic_features

    def _get_timestep_features(self,
                               static_features: ndarray,
                               ts_dynamic_features: ndarray,
                               node_derived_features: Optional[ndarray] = None) -> Tensor:
        _, num_elems, _ = ts_dynamic_features.shape
        ts_dynamic_features = ts_dynamic_features.transpose(1, 0, 2)
        ts_dynamic_features = np.reshape(ts_dynamic_features, shape=(num_elems, -1), order='F')
        if node_derived_features is not None:
            ts_data = np.concatenate([static_features, node_derived_features, ts_dynamic_features], axis=1)
        else:
            ts_data = np.concatenate([static_features, ts_dynamic_features], axis=1)
        return torch.from_numpy(ts_data)

    def _compute_node_derived_features(self, ts_dynamic_features: ndarray, static_features: ndarray) -> ndarray:
        """
        Compute simple derived statistics from the available warm-start window only.
        Output shape: [num_nodes, num_derived_features].
        """
        wl_idx = self.DYNAMIC_NODE_FEATURES.index('water_level')
        rain_idx = self.DYNAMIC_NODE_FEATURES.index('rainfall')
        wl = ts_dynamic_features[:, :, wl_idx]   # [seq, nodes]
        rain = ts_dynamic_features[:, :, rain_idx]  # [seq, nodes]
        seq_len = wl.shape[0]
        eps = 1e-8
        feats: List[ndarray] = []

        # Delta windows: y_t - y_{t-w}
        for w in self.node_derived_feature_windows:
            w_eff = int(min(max(1, w), max(1, seq_len - 1)))
            feats.append((wl[-1] - wl[-1 - w_eff])[:, None])

        # Volatility windows: std over trailing windows
        for w in self.node_derived_feature_windows:
            w_eff = int(min(max(2, w), seq_len))
            x = wl[-w_eff:]
            mu = x.mean(axis=0, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=0)
            feats.append(np.sqrt(var + eps)[:, None])

        # Rain memory windows: trailing cumulative sums
        rain_mem_feats: List[ndarray] = []
        for w in self.node_derived_rain_windows:
            w_eff = int(min(max(1, w), seq_len))
            m = rain[-w_eff:].sum(axis=0)[:, None]
            rain_mem_feats.append(m)
            feats.append(m)

        # Rain intensity windows: trailing max
        for w in self.node_derived_rain_max_windows:
            w_eff = int(min(max(1, w), seq_len))
            m = rain[-w_eff:].max(axis=0)[:, None]
            rain_mem_feats.append(m)
            feats.append(m)

        # EMA rain memory channels
        if len(self.node_derived_rain_decay_alphas) > 0:
            for alpha in self.node_derived_rain_decay_alphas:
                a = float(alpha)
                ema = np.zeros((rain.shape[1],), dtype=np.float32)
                for t in range(seq_len):
                    ema = (a * ema) + ((1.0 - a) * rain[t].astype(np.float32))
                m = ema[:, None]
                rain_mem_feats.append(m)
                feats.append(m)

        # Cross terms: rain-memory x static scalars.
        if self.node_derived_interactions_enabled and len(self._node_interaction_static_idx) > 0 and len(rain_mem_feats) > 0:
            # static_features here is original normalized static block only.
            for m in rain_mem_feats:
                for s_idx in self._node_interaction_static_idx:
                    feats.append((m[:, 0] * static_features[:, s_idx])[:, None])

        return np.concatenate(feats, axis=1).astype(np.float32)

    def _get_empty_feature_tensor(self, features: List[str], other_dims: Tuple[int, ...], dtype: np.dtype = np.float32) -> ndarray:
        if not self.is_normalized:
            return np.zeros((*other_dims, len(features)), dtype=dtype)
        return self.normalizer.get_normalized_zero_tensor(features, other_dims, dtype)

    def _get_node_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int, external_qhat_features: Optional[ndarray] = None) -> Tensor:
        '''
        Builds the final feature vector for nodes at a given timestep by concatenating:
        - Static node features
        - Optionally, node-derived features computed from the warm-start window (anchored to the last legal state if warm-start masking is applied)
        - Dynamic node features over the warm-start window (with optional masking of future illegal states) 
        '''
        dyn_list = self._effective_dynamic_node_features()
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_NODE_FEATURES, timestep_idx)
        if external_qhat_features is not None:
            ts_qhat = self._get_timestep_dynamic_features(
                external_qhat_features,
                [self.external_qhat_feature_name],
                timestep_idx,
                apply_warm_mask=False,
            )
            ts_dynamic_features = np.concatenate([ts_dynamic_features, ts_qhat], axis=2)
        node_derived_features = None
        if self.node_derived_features_enabled:
            # After warm-start, keep derived state summaries anchored to the last legal state.
            if self.warm_start_timesteps is not None and timestep_idx >= self.warm_start_timesteps:
                derived_ts = self._get_timestep_dynamic_features(
                    dynamic_features,
                    self.DYNAMIC_NODE_FEATURES,
                    self.warm_start_timesteps - 1,
                )
            else:
                derived_ts = ts_dynamic_features
            node_derived_features = self._compute_node_derived_features(derived_ts, static_features)
        return self._get_timestep_features(static_features, ts_dynamic_features, node_derived_features=node_derived_features)
        # features =  self._get_timestep_features(static_features, ts_dynamic_features, node_derived_features=node_derived_features)
        # features = self.node_pca.transform(features)
        # return torch.tensor(features, dtype=torch.float32)


    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, timestep_idx)
        return self._get_timestep_features(static_features, ts_dynamic_features)
        # features = self._get_timestep_features(static_features, ts_dynamic_features)
        # features = self.edge_pca.transform(features)
        # return torch.tensor(features, dtype=torch.float32)

    def _get_timestep_labels(self,
                             node_dynamic_features: ndarray,
                             edge_dynamic_features: ndarray,
                             timestep_idx: int,
                             max_timesteps: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        if max_timesteps is None:
            max_timesteps = node_dynamic_features.shape[0]
        next_timestep = min(timestep_idx + 1, max_timesteps - 1)
        node_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        current_nodes = node_dynamic_features[timestep_idx, :, node_idx][:, None]
        next_nodes = node_dynamic_features[next_timestep, :, node_idx][:, None]
        label_nodes = torch.from_numpy(next_nodes - current_nodes)

        edge_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        current_edges = edge_dynamic_features[timestep_idx, :, edge_idx][:, None]
        next_edges = edge_dynamic_features[next_timestep, :, edge_idx][:, None]
        label_edges = torch.from_numpy(next_edges - current_edges)
        return label_nodes, label_edges

    def _get_timestep_inlet_labels(self,
                                   node_dynamic_features: ndarray,
                                   timestep_idx: int,
                                   max_timesteps: Optional[int] = None) -> Tensor:
        if max_timesteps is None:
            max_timesteps = node_dynamic_features.shape[0]
        next_timestep = min(timestep_idx + 1, max_timesteps - 1)
        inlet_idx = self.DYNAMIC_NODE_FEATURES.index('inlet_flow')
        next_inlet = node_dynamic_features[next_timestep, :, inlet_idx][:, None]
        return torch.from_numpy(next_inlet)

    def _get_timestep_netq_labels(self,
                                  node_dynamic_features: ndarray,
                                  edge_dynamic_features: ndarray,
                                  timestep_idx: int,
                                  max_timesteps: Optional[int] = None) -> Tensor:
        if max_timesteps is None:
            max_timesteps = node_dynamic_features.shape[0]
        next_timestep = min(timestep_idx + 1, max_timesteps - 1)
        netq = self._compute_netq_series(node_dynamic_features, edge_dynamic_features)
        next_netq = self._normalize_netq(netq[next_timestep])[:, None]
        return torch.from_numpy(next_netq.astype(np.float32, copy=False))

    def _get_global_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, timestep_idx: int) -> Dict[str, Tensor]:
        non_boundary_nodes_mask = ~self.boundary_condition.boundary_nodes_mask
        total_rainfall = node_rainfall_per_ts[timestep_idx, non_boundary_nodes_mask].sum(keepdims=True)

        total_rainfall = torch.from_numpy(total_rainfall)
        inflow_edges_mask = torch.from_numpy(self.boundary_condition.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(self.boundary_condition.outflow_edges_mask)
        non_boundary_nodes_mask = torch.from_numpy(non_boundary_nodes_mask)

        return {
            'total_rainfall': total_rainfall,
            'inflow_edges_mask': inflow_edges_mask,
            'outflow_edges_mask': outflow_edges_mask,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }

    def _get_local_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, timestep_idx: int) -> Dict[str, Tensor]:
        rainfall = node_rainfall_per_ts[timestep_idx]
        rainfall = torch.from_numpy(rainfall)
        non_boundary_nodes_mask = torch.from_numpy(~self.boundary_condition.boundary_nodes_mask)

        return {
            'rainfall': rainfall,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }

    def _get_rainfall_volume_per_ts(self, dynamic_nodes_input: ndarray) -> ndarray:
        rainfall_idx = self.DYNAMIC_NODE_FEATURES.index('rainfall')
        rainfall = dynamic_nodes_input[:, :, rainfall_idx]
        if self.is_normalized and self.normalizer is not None:
            rainfall = self.normalizer.denormalize('rainfall', rainfall)
        # Convert rainfall depth to volume using per-node area (1D areas are 0).
        return rainfall * self.node_area[None, :]

    def _build_data_list(self) -> List[Data]:
        data_list = []
        for event_idx, event in enumerate(self.events_dynamic):
            timesteps = self.event_timesteps[event_idx]
            dynamic_nodes = event['dynamic_nodes']
            dynamic_edges = event['dynamic_edges']
            dynamic_nodes_input = event.get('dynamic_nodes_input', dynamic_nodes)
            dynamic_edges_input = event.get('dynamic_edges_input', dynamic_edges)

            node_rainfall_per_ts = event.get('node_rainfall_per_ts')

            label_horizon = 1 if getattr(self, 'dynamic_label_horizon', False) else self.num_label_timesteps
            max_ts = len(timesteps) - label_horizon
            for ts in range(self.previous_timesteps, max_ts):
                node_features = self._get_node_timestep_data(self.static_nodes, dynamic_nodes_input, ts, event.get('external_qhat_input'))
                edge_features = self._get_edge_timestep_data(self.static_edges, dynamic_edges_input, ts)
                label_mask = None
                labels = self._get_timestep_labels(dynamic_nodes, dynamic_edges, ts, max_timesteps=len(timesteps))
                if isinstance(labels, tuple) and len(labels) == 3:
                    label_nodes, label_edges, label_mask = labels
                else:
                    label_nodes, label_edges = labels
                label_inlet = self._get_timestep_inlet_labels(dynamic_nodes, ts, max_timesteps=len(timesteps))
                label_netq = self._get_timestep_netq_labels(
                    event.get('dynamic_nodes_raw', dynamic_nodes),
                    event.get('dynamic_edges_raw', dynamic_edges),
                    ts,
                    max_timesteps=len(timesteps),
                )
                node_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
                edge_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
                node_base = torch.from_numpy(dynamic_nodes[ts, :, node_idx][:, None])
                inlet_idx = self.DYNAMIC_NODE_FEATURES.index('inlet_flow')
                inlet_base = torch.from_numpy(dynamic_nodes[ts, :, inlet_idx][:, None])
                edge_base = torch.from_numpy(dynamic_edges[ts, :, edge_idx][:, None])
                netq_series = self._compute_netq_series(
                    event.get('dynamic_nodes_raw', dynamic_nodes),
                    event.get('dynamic_edges_raw', dynamic_edges),
                )
                netq_base = torch.from_numpy(self._normalize_netq(netq_series[ts])[:, None].astype(np.float32, copy=False))

                global_mass_info = None
                if self.with_global_mass_loss:
                    global_mass_info = self._get_global_mass_info_for_timestep(node_rainfall_per_ts, ts)

                local_mass_info = None
                if self.with_local_mass_loss:
                    local_mass_info = self._get_local_mass_info_for_timestep(node_rainfall_per_ts, ts)

                data = Data(
                    x=node_features,
                    edge_index=self.edge_index,
                    edge_attr=edge_features,
                    y=label_nodes,
                    y_inlet=label_inlet,
                    y_netq=label_netq,
                    y_edge=label_edges,
                    y_base=node_base,
                    y_inlet_base=inlet_base,
                    y_netq_base=netq_base,
                    y_edge_base=edge_base,
                    inlet_q_mean=torch.tensor(self.inlet_q_mean, dtype=torch.float32),
                    inlet_q_std=torch.tensor(self.inlet_q_std, dtype=torch.float32),
                    netq_q_mean=torch.tensor(self.netq_q_mean, dtype=torch.float32),
                    netq_q_std=torch.tensor(self.netq_q_std, dtype=torch.float32),
                    node_area_raw=torch.from_numpy(self.node_area.copy()),
                    node_base_area_raw=torch.from_numpy(self.node_base_area.copy()),
                    dt_seconds=torch.tensor(float(self.timestep_interval), dtype=torch.float32),
                    timestep=timesteps[ts],
                    global_mass_info=global_mass_info,
                    local_mass_info=local_mass_info,
                    node_ids=torch.arange(self.kaggle_node_types.shape[0], dtype=torch.long),
                )
                if label_mask is not None:
                    data.label_mask = label_mask
                data_list.append(data)
        return data_list

    def len(self) -> int:
        if self.data_list is not None:
            return len(self.data_list)
        return int(self.total_rollout_timesteps)

    def get(self, idx: int) -> Data:
        if self.data_list is not None:
            return self.data_list[idx]

        from bisect import bisect_right

        event_idx = bisect_right(self.event_start_idx, idx) - 1
        if event_idx < 0 or event_idx >= len(self.event_start_idx):
            raise IndexError(f'Index {idx} out of range for dataset length {self.len()}')
        timesteps = self.event_timesteps[event_idx]
        event = self.events_dynamic[event_idx]
        ts_offset = idx - self.event_start_idx[event_idx]
        if self.random_window_sampling and self.mode == 'train':
            label_horizon = 1 if getattr(self, 'dynamic_label_horizon', False) else self.num_label_timesteps
            max_ts_offset = len(timesteps) - self.previous_timesteps - label_horizon - 1
            if max_ts_offset < 0:
                max_ts_offset = 0
            ts_offset = self._get_random_window_offset(event_idx, idx, max_ts_offset)
        ts = self.previous_timesteps + ts_offset
        dynamic_nodes = event['dynamic_nodes']
        dynamic_edges = event['dynamic_edges']
        dynamic_nodes_input = event.get('dynamic_nodes_input', dynamic_nodes)
        dynamic_edges_input = event.get('dynamic_edges_input', dynamic_edges)

        node_features = self._get_node_timestep_data(self.static_nodes, dynamic_nodes_input, ts, event.get('external_qhat_input'))
        edge_features = self._get_edge_timestep_data(self.static_edges, dynamic_edges_input, ts)
        label_mask = None
        labels = self._get_timestep_labels(dynamic_nodes, dynamic_edges, ts, max_timesteps=len(timesteps))
        if isinstance(labels, tuple) and len(labels) == 3:
            label_nodes, label_edges, label_mask = labels
        else:
            label_nodes, label_edges = labels
        label_inlet = self._get_timestep_inlet_labels(dynamic_nodes, ts, max_timesteps=len(timesteps))
        label_netq = self._get_timestep_netq_labels(
            event.get('dynamic_nodes_raw', dynamic_nodes),
            event.get('dynamic_edges_raw', dynamic_edges),
            ts,
            max_timesteps=len(timesteps),
        )
        node_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        inlet_idx = self.DYNAMIC_NODE_FEATURES.index('inlet_flow')
        edge_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        node_base = torch.from_numpy(dynamic_nodes[ts, :, node_idx][:, None])
        inlet_base = torch.from_numpy(dynamic_nodes[ts, :, inlet_idx][:, None])
        edge_base = torch.from_numpy(dynamic_edges[ts, :, edge_idx][:, None])
        netq_series = self._compute_netq_series(
            event.get('dynamic_nodes_raw', dynamic_nodes),
            event.get('dynamic_edges_raw', dynamic_edges),
        )
        netq_base = torch.from_numpy(self._normalize_netq(netq_series[ts])[:, None].astype(np.float32, copy=False))

        global_mass_info = None
        local_mass_info = None
        if self.with_global_mass_loss or self.with_local_mass_loss:
            node_rainfall_per_ts = event.get('node_rainfall_per_ts')
            if self.with_global_mass_loss:
                global_mass_info = self._get_global_mass_info_for_timestep(node_rainfall_per_ts, ts)
            if self.with_local_mass_loss:
                local_mass_info = self._get_local_mass_info_for_timestep(node_rainfall_per_ts, ts)

        data = Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=edge_features,
            y=label_nodes,
            y_inlet=label_inlet,
            y_netq=label_netq,
            y_edge=label_edges,
            y_base=node_base,
            y_inlet_base=inlet_base,
            y_netq_base=netq_base,
            y_edge_base=edge_base,
            inlet_q_mean=torch.tensor(self.inlet_q_mean, dtype=torch.float32),
            inlet_q_std=torch.tensor(self.inlet_q_std, dtype=torch.float32),
            netq_q_mean=torch.tensor(self.netq_q_mean, dtype=torch.float32),
            netq_q_std=torch.tensor(self.netq_q_std, dtype=torch.float32),
            node_area_raw=torch.from_numpy(self.node_area.copy()),
            node_base_area_raw=torch.from_numpy(self.node_base_area.copy()),
            dt_seconds=torch.tensor(float(self.timestep_interval), dtype=torch.float32),
            timestep=timesteps[ts],
            global_mass_info=global_mass_info,
            local_mass_info=local_mass_info,
            node_types=torch.from_numpy(self.kaggle_node_types.copy()),
            node_ids=torch.arange(self.kaggle_node_types.shape[0], dtype=torch.long),
        )
        if label_mask is not None:
            data.label_mask = label_mask
        return data


class CsvAutoregressiveFloodDataset(CsvFloodDataset):
    def __init__(self,
                 num_label_timesteps: int = 1,
                 dynamic_label_horizon: bool = False,
                 *args,
                 **kwargs):
        self.num_label_timesteps = num_label_timesteps
        self.dynamic_label_horizon = bool(dynamic_label_horizon)
        super().__init__(*args, **kwargs)

    def _get_node_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int, external_qhat_features: Optional[ndarray] = None) -> Tensor:
        ts_data = []
        end_ts = timestep_idx + self.num_label_timesteps
        max_ts = dynamic_features.shape[0]
        for ts_idx in range(timestep_idx, end_ts):
            if ts_idx >= max_ts:
                ts_dynamic_features = self._get_empty_feature_tensor(
                    self._effective_dynamic_node_features(),
                    (self.previous_timesteps + 1, dynamic_features.shape[1]),
                    dtype=dynamic_features.dtype,
                )
            else:
                ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_NODE_FEATURES, ts_idx)
                if external_qhat_features is not None:
                    ts_qhat = self._get_timestep_dynamic_features(
                        external_qhat_features,
                        [self.external_qhat_feature_name],
                        ts_idx,
                        apply_warm_mask=False,
                    )
                    ts_dynamic_features = np.concatenate([ts_dynamic_features, ts_qhat], axis=2)
            node_derived_features = None
            if self.node_derived_features_enabled:
                if self.warm_start_timesteps is not None and ts_idx >= self.warm_start_timesteps:
                    derived_ts = self._get_timestep_dynamic_features(
                        dynamic_features,
                        self.DYNAMIC_NODE_FEATURES,
                        self.warm_start_timesteps - 1,
                    )
                else:
                    derived_ts = ts_dynamic_features
                node_derived_features = self._compute_node_derived_features(derived_ts, static_features)
            ts_features = self._get_timestep_features(
                static_features,
                ts_dynamic_features,
                node_derived_features=node_derived_features,
            )
            ts_data.append(ts_features)
        ts_data = torch.stack(ts_data, dim=-1)
        return ts_data

    def _get_edge_timestep_data(self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int) -> Tensor:
        ts_data = []
        end_ts = timestep_idx + self.num_label_timesteps
        max_ts = dynamic_features.shape[0]
        for ts_idx in range(timestep_idx, end_ts):
            if ts_idx >= max_ts:
                ts_dynamic_features = self._get_empty_feature_tensor(
                    self.DYNAMIC_EDGE_FEATURES,
                    (self.previous_timesteps + 1, dynamic_features.shape[1]),
                    dtype=dynamic_features.dtype,
                )
            else:
                ts_dynamic_features = self._get_timestep_dynamic_features(dynamic_features, self.DYNAMIC_EDGE_FEATURES, ts_idx)
            ts_features = self._get_timestep_features(static_features, ts_dynamic_features)
            ts_data.append(ts_features)
        ts_data = torch.stack(ts_data, dim=-1)
        return ts_data

    def _get_timestep_labels(self,
                             node_dynamic_features: ndarray,
                             edge_dynamic_features: ndarray,
                             timestep_idx: int,
                             max_timesteps: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        start_label_idx = timestep_idx + 1
        max_ts = max_timesteps if max_timesteps is not None else node_dynamic_features.shape[0]
        available = max(0, max_ts - start_label_idx)
        horizon = min(self.num_label_timesteps, available)
        end_label_idx = start_label_idx + horizon

        node_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        if horizon > 0:
            current_nodes = node_dynamic_features[start_label_idx-1:end_label_idx-1, :, node_idx]
            next_nodes = node_dynamic_features[start_label_idx:end_label_idx, :, node_idx]
            label_nodes = next_nodes - current_nodes
            label_nodes = label_nodes.T[:, None, :]
        else:
            label_nodes = np.zeros((self.num_nodes, 1, 0), dtype=np.float32)
        if horizon < self.num_label_timesteps:
            pad = np.zeros((label_nodes.shape[0], label_nodes.shape[1], self.num_label_timesteps - horizon), dtype=np.float32)
            label_nodes = np.concatenate([label_nodes, pad], axis=2)
        label_nodes = torch.from_numpy(label_nodes)

        edge_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        if horizon > 0:
            current_edges = edge_dynamic_features[start_label_idx-1:end_label_idx-1, :, edge_idx]
            next_edges = edge_dynamic_features[start_label_idx:end_label_idx, :, edge_idx]
            label_edges = next_edges - current_edges
            label_edges = label_edges.T[:, None, :]
        else:
            label_edges = np.zeros((self.num_edges, 1, 0), dtype=np.float32)
        if horizon < self.num_label_timesteps:
            pad = np.zeros((label_edges.shape[0], label_edges.shape[1], self.num_label_timesteps - horizon), dtype=np.float32)
            label_edges = np.concatenate([label_edges, pad], axis=2)
        label_edges = torch.from_numpy(label_edges)

        if self.dynamic_label_horizon:
            label_mask = np.zeros((self.num_label_timesteps,), dtype=np.float32)
            if horizon > 0:
                label_mask[:horizon] = 1.0
        else:
            label_mask = np.ones((self.num_label_timesteps,), dtype=np.float32)
        label_mask = torch.from_numpy(label_mask)

        return label_nodes, label_edges, label_mask

    def _get_timestep_inlet_labels(self,
                                   node_dynamic_features: ndarray,
                                   timestep_idx: int,
                                   max_timesteps: Optional[int] = None) -> Tensor:
        start_label_idx = timestep_idx + 1
        max_ts = max_timesteps if max_timesteps is not None else node_dynamic_features.shape[0]
        available = max(0, max_ts - start_label_idx)
        horizon = min(self.num_label_timesteps, available)
        end_label_idx = start_label_idx + horizon

        inlet_idx = self.DYNAMIC_NODE_FEATURES.index('inlet_flow')
        if horizon > 0:
            next_inlet = node_dynamic_features[start_label_idx:end_label_idx, :, inlet_idx]
            label_inlet = next_inlet.T[:, None, :]
        else:
            label_inlet = np.zeros((self.num_nodes, 1, 0), dtype=np.float32)
        if horizon < self.num_label_timesteps:
            pad = np.zeros((label_inlet.shape[0], label_inlet.shape[1], self.num_label_timesteps - horizon), dtype=np.float32)
            label_inlet = np.concatenate([label_inlet, pad], axis=2)
        return torch.from_numpy(label_inlet)

    def _get_timestep_netq_labels(self,
                                  node_dynamic_features: ndarray,
                                  edge_dynamic_features: ndarray,
                                  timestep_idx: int,
                                  max_timesteps: Optional[int] = None) -> Tensor:
        start_label_idx = timestep_idx + 1
        max_ts = max_timesteps if max_timesteps is not None else node_dynamic_features.shape[0]
        available = max(0, max_ts - start_label_idx)
        horizon = min(self.num_label_timesteps, available)
        end_label_idx = start_label_idx + horizon

        netq = self._compute_netq_series(node_dynamic_features, edge_dynamic_features)
        if horizon > 0:
            next_netq = self._normalize_netq(netq[start_label_idx:end_label_idx])
            label_netq = next_netq.T[:, None, :]
        else:
            label_netq = np.zeros((self.num_nodes, 1, 0), dtype=np.float32)
        if horizon < self.num_label_timesteps:
            pad = np.zeros((label_netq.shape[0], label_netq.shape[1], self.num_label_timesteps - horizon), dtype=np.float32)
            label_netq = np.concatenate([label_netq, pad], axis=2)
        return torch.from_numpy(label_netq.astype(np.float32, copy=False))

    def _get_global_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, timestep_idx: int) -> Dict[str, Tensor]:
        end_idx = timestep_idx + self.num_label_timesteps
        non_boundary_nodes_mask = ~self.boundary_condition.boundary_nodes_mask
        total_rainfall = node_rainfall_per_ts[timestep_idx:end_idx, non_boundary_nodes_mask].sum(axis=1)[None, :]
        if total_rainfall.shape[1] < self.num_label_timesteps:
            pad = np.zeros((1, self.num_label_timesteps - total_rainfall.shape[1]), dtype=total_rainfall.dtype)
            total_rainfall = np.concatenate([total_rainfall, pad], axis=1)

        total_rainfall = torch.from_numpy(total_rainfall)
        inflow_edges_mask = torch.from_numpy(self.boundary_condition.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(self.boundary_condition.outflow_edges_mask)
        non_boundary_nodes_mask = torch.from_numpy(non_boundary_nodes_mask)

        return {
            'total_rainfall': total_rainfall,
            'inflow_edges_mask': inflow_edges_mask,
            'outflow_edges_mask': outflow_edges_mask,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }

    def _get_local_mass_info_for_timestep(self, node_rainfall_per_ts: ndarray, timestep_idx: int) -> Dict[str, Tensor]:
        end_ts = timestep_idx + self.num_label_timesteps
        rainfall = node_rainfall_per_ts[timestep_idx:end_ts].T
        if rainfall.shape[1] < self.num_label_timesteps:
            pad = np.zeros((rainfall.shape[0], self.num_label_timesteps - rainfall.shape[1]), dtype=rainfall.dtype)
            rainfall = np.concatenate([rainfall, pad], axis=1)

        rainfall = torch.from_numpy(rainfall)
        non_boundary_nodes_mask = torch.from_numpy(~self.boundary_condition.boundary_nodes_mask)

        return {
            'rainfall': rainfall,
            'non_boundary_nodes_mask': non_boundary_nodes_mask,
        }
