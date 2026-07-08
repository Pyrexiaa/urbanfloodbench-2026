import os
import torch
import numpy as np
import pandas as pd

from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Dataset, Data
from typing import Callable, Tuple, List, Literal, Dict, Optional, Union
from utils.logger import Logger
from ..utils.file_utils import read_yaml_file, save_to_yaml_file

from .dem_data_retrieval import (
    get_filled_dem,
    get_aspect,
    get_curvature,
    get_flow_accumulation,
)
from .hecras_1d2d_data_retrieval import (
    get_1d_flow,
    get_1d_inlet_flow,
    get_1d_velocity,
    get_1d_water_level,
    get_event_timesteps,
    get_cell_area,
    get_roughness,
    get_rainfall,
    get_water_level,
    get_water_volume,
    get_velocity,
    get_face_flow,
)
from .shp_1d2d_data_retrieval import (
    get_1d_base_area,
    get_1d_cell_depth,
    get_1d_edge_diameter,
    get_1d_edge_manning,
    get_1d_edge_relative_position,
    get_1d_edge_shape,
    get_1d_edge_slope,
    get_1d_invert_elevation,
    get_1d_surface_elevation,
    get_edge_index,
    get_cell_elevation,
    get_edge_length,
    get_edge_slope,
    get_cell_position_x,
    get_cell_position_y,
    get_cell_position,
    get_edge_direction_x,
    get_edge_direction_y,
    get_face_length,
    get_1d_edge_index,
    get_1d_edge_length,
    get_1d2d_edge_index,
    get_relative_position,
)
from .boundary_1d2d_condition import Boundary1d2dCondition
from .dataset_normalizer import DatasetNormalizer
from .unit_converter import UnitConverter


class FloodEvent1D2DDataset(Dataset):
    # --- Existing 2D Constants ---
    STATIC_NODE_FEATURES = [
        "position_x",
        "position_y",
        "area",
        "roughness",
        "elevation",
        "aspect",
        "curvature",
        "flow_accumulation",
    ]
    DYNAMIC_NODE_FEATURES = ["rainfall", "water_level"]  # Not included water volume
    STATIC_EDGE_FEATURES = [
        "relative_position_x",
        "relative_position_y",
        "face_length",
        "length",
        "slope",
    ]
    DYNAMIC_EDGE_FEATURES = ["flow"]  # Not included velocity
    NODE_TARGET_FEATURE = "water_level"
    EDGE_TARGET_FEATURE = "flow"

    # --- New 1D Constants ---
    STATIC_1D_NODE_FEATURES = [
        "position_x",
        "position_y",
        "depth",
        "invert_elevation",
        "surface_elevation",
        "base_area",
    ]
    DYNAMIC_1D_NODE_FEATURES = ["water_level"]  # Not included inlet flow
    STATIC_1D_EDGE_FEATURES = [
        "relative_position_x",
        "relative_position_y",
        "length",
        "diameter",
        "shape",
        "roughness",
        "slope",
    ]
    DYNAMIC_1D_EDGE_FEATURES = ["flow"]  # Not included velocity
    NODE_1D_TARGET_FEATURE = "water_level"
    EDGE_1D_TARGET_FEATURE = "flow"

    # Files
    BOUNDARY_CONDITION_NPZ_FILE = "boundary_condition_masks.npz"
    CONSTANT_VALUES_NPZ_FILE = "constant_values.npz"

    def __init__(
        self,
        mode: Literal["train", "test"],
        root_dir: str,
        dataset_summary_file: str,
        # --- 2D File Arguments ---
        nodes_2d_shp_file: str,
        edges_2d_shp_file: str,
        # --- New 1D File Arguments ---
        nodes_1d_shp_file: str,
        edges_1d_shp_file: str,
        edges_1d2d_shp_file: str,
        dem_file: str,
        event_stats_file: str = "event_stats.yaml",
        features_stats_file: str = "features_stats.yaml",
        previous_timesteps: int = 2,
        normalize: bool = True,
        timestep_interval: int = 30,  # in seconds
        spin_up_time: Union[int, Dict[str, int]] = None,
        time_from_peak: Optional[int] = None,
        inflow_boundary_nodes: List[int] = [],
        outflow_boundary_nodes: List[int] = [],
        with_global_mass_loss: bool = True,
        with_local_mass_loss: bool = True,
        debug: bool = False,
        logger: Optional[Logger] = None,
        force_reload: bool = False,
        save: bool = False,
        perimeter_name: str = "US Beaver",
        network_name: str = "Base",
        model_name: str = "Model1",
        node_1d_mapping: Dict[int, int] = None,
        source_units: str = "SI",
        target_units: Optional[str] = None,
    ):
        assert mode in ["train", "test"], (
            f'Invalid mode: {mode}. Must be "train" or "test".'
        )

        self.log_func = print
        if logger is not None and hasattr(logger, "log"):
            self.log_func = logger.log

        # File paths
        self.hec_ras_files, self.hec_ras_run_ids = self._get_hecras_files_from_summary(
            root_dir, dataset_summary_file
        )
        self.perimeter_name = perimeter_name
        self.network_name = network_name
        self.model_name = model_name
        self.nodes_2d_shp_file = nodes_2d_shp_file
        self.edges_2d_shp_file = edges_2d_shp_file
        self.nodes_1d_shp_file = nodes_1d_shp_file
        self.edges_1d_shp_file = edges_1d_shp_file
        self.edges_1d2d_shp_file = edges_1d2d_shp_file
        self.dem_file = dem_file
        self.event_stats_file = event_stats_file
        self.features_stats_file = features_stats_file
        self.node_1d_mapping = node_1d_mapping or {}

        # Raw Paths Indices
        # ['data/datasets/raw/Geometry/Nodes_1D.shp', 'data/datasets/raw/Geometry/Links_1D.shp', 'data/datasets/raw/Geometry/Node1D_to_Node2D_Links.shp', 'data/datasets/raw/Geometry/Nodes_2D.shp', 'data/datasets/raw/Geometry/Links_2D.shp', 'data/datasets/raw/Geometry/DEM.tif', 'data/datasets/raw/HEC-RAS_Results/BeaverLAKE_Pipe_Mod.p22.hdf...'

        # Dataset configurations
        self.mode = mode
        self.previous_timesteps = previous_timesteps
        self.is_normalized = normalize
        self.timestep_interval = timestep_interval
        self.spin_up_time = spin_up_time
        self.time_from_peak = time_from_peak
        self.inflow_boundary_nodes = inflow_boundary_nodes
        self.outflow_boundary_nodes = outflow_boundary_nodes
        self.with_global_mass_loss = with_global_mass_loss
        self.with_local_mass_loss = with_local_mass_loss

        # Dataset variables
        self.num_static_node_features = len(self.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(self.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(self.DYNAMIC_EDGE_FEATURES)
        # 1D Feature Counts
        self.num_static_1d_node_features = len(self.STATIC_1D_NODE_FEATURES)
        self.num_dynamic_1d_node_features = len(self.DYNAMIC_1D_NODE_FEATURES)
        self.num_static_1d_edge_features = len(self.STATIC_1D_EDGE_FEATURES)
        self.num_dynamic_1d_edge_features = len(self.DYNAMIC_1D_EDGE_FEATURES)
        event_stats = self._load_event_stats(root_dir, event_stats_file)
        self.event_start_idx, self.total_rollout_timesteps, processed_event_info = (
            event_stats
        )
        self._event_peak_idx = None
        self._event_num_timesteps = None
        self._event_base_timestep_interval = None

        # Optional unit conversion (off unless target_units is set and differs
        # from the model's native source_units). Applied before normalization so
        # that feature statistics are computed in the common target system.
        self.source_units = source_units
        self.target_units = target_units
        self.unit_converter = None
        if target_units is not None:
            converter = UnitConverter(
                source_system=source_units,
                target_system=target_units,
                log_func=self.log_func,
            )
            if not converter.is_noop:
                self.unit_converter = converter
                self.log_func(
                    f"Unit conversion enabled: {converter.source_system} -> "
                    f"{converter.target_system} for model '{model_name}'."
                )

        # Helper classes
        self.normalizer = DatasetNormalizer(mode, root_dir, features_stats_file)
        self.boundary_condition = Boundary1d2dCondition(
            root_dir=root_dir,
            hec_ras_file=self.hec_ras_files[0],
            inflow_boundary_nodes=self.inflow_boundary_nodes,
            outflow_boundary_nodes=self.outflow_boundary_nodes,
            saved_npz_file=self.BOUNDARY_CONDITION_NPZ_FILE,
            perimeter_name=self.perimeter_name,
        )

        force_reload = (
            self._is_previous_config_different(processed_event_info) or force_reload
        )

        # Save condition
        self.save = save
        super().__init__(
            root_dir,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            log=debug,
            force_reload=force_reload,
        )

    @property
    def raw_file_names(self):
        return [
            self.nodes_1d_shp_file,
            self.edges_1d_shp_file,
            self.edges_1d2d_shp_file,
            self.nodes_2d_shp_file,
            self.edges_2d_shp_file,
            self.dem_file,
            *self.hec_ras_files,
        ]

    @property
    def processed_file_names(self):
        dynamic_files = [
            f"dynamic_values_event_{run_id}.npz" for run_id in self.hec_ras_run_ids
        ]
        return [
            self.event_stats_file,
            self.features_stats_file,
            self.BOUNDARY_CONDITION_NPZ_FILE,
            self.CONSTANT_VALUES_NPZ_FILE,
            *dynamic_files,
        ]

    def download(self):
        # Data must be downloaded manually and placed in the raw dir
        pass

    def process(self):
        self.log_func("Processing Flood Event 1D-2D Dataset...")

        all_event_timesteps = self._set_event_properties()

        # --- 2D Processing ---
        edge_index = self._get_edge_index(self.raw_paths[4])
        static_nodes = self._get_static_node_features()
        dynamic_nodes = self._get_dynamic_node_features()
        static_edges = self._get_static_edge_features()
        dynamic_edges = self._get_dynamic_edge_features()

        self.boundary_condition.create(edge_index, dynamic_edges)
        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = (
            self.boundary_condition.remove(
                static_nodes,
                dynamic_nodes,
                static_edges,
                dynamic_edges,
                edge_index,
            )
        )
        node_edge_remapping_path = os.path.join(
            self.processed_dir, "node_edge_remapping", f"{self.mode}.json"
        )
        self.boundary_condition.save_remapping(node_edge_remapping_path)
        static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index = (
            self.boundary_condition.apply(
                static_nodes,
                dynamic_nodes,
                static_edges,
                dynamic_edges,
                edge_index,
            )
        )

        # --- 1D Processing ---
        edge_index_1d = self._get_1d_edge_index(self.raw_paths[1], self.raw_paths[0])
        edge_index_1d_2d = self._get_1d_2d_edge_index(self.raw_paths[2])
        static_nodes_1d = self._get_static_1d_node_features()
        static_edges_1d = self._get_static_1d_edge_features()
        dynamic_nodes_1d = self._get_dynamic_1d_node_features()
        dynamic_edges_1d = self._get_dynamic_1d_edge_features()

        # Optional unit conversion (before normalization). No-op unless a
        # differing target unit system was requested at construction time.
        if self.unit_converter is not None:
            static_nodes = self.unit_converter.convert_feature_vector(
                self.STATIC_NODE_FEATURES, static_nodes
            )
            dynamic_nodes = self.unit_converter.convert_feature_vector(
                self.DYNAMIC_NODE_FEATURES, dynamic_nodes
            )
            static_edges = self.unit_converter.convert_feature_vector(
                self.STATIC_EDGE_FEATURES, static_edges
            )
            dynamic_edges = self.unit_converter.convert_feature_vector(
                self.DYNAMIC_EDGE_FEATURES, dynamic_edges
            )
            static_nodes_1d = self.unit_converter.convert_feature_vector(
                self.STATIC_1D_NODE_FEATURES, static_nodes_1d
            )
            dynamic_nodes_1d = self.unit_converter.convert_feature_vector(
                self.DYNAMIC_1D_NODE_FEATURES, dynamic_nodes_1d
            )
            static_edges_1d = self.unit_converter.convert_feature_vector(
                self.STATIC_1D_EDGE_FEATURES, static_edges_1d
            )
            dynamic_edges_1d = self.unit_converter.convert_feature_vector(
                self.DYNAMIC_1D_EDGE_FEATURES, dynamic_edges_1d
            )
            self.log_func(
                f"Applied unit conversion "
                f"({self.unit_converter.source_system} -> "
                f"{self.unit_converter.target_system})."
            )

        # Physics-informed Loss Features (2D). Computed after unit conversion so
        # the rainfall used by the mass-conservation loss matches the (possibly
        # converted) dynamic node features.
        node_rainfall_per_ts = self._get_physics_info(dynamic_nodes)

        if self.is_normalized:
            # 2D Normalization
            static_nodes = self.normalizer.normalize_feature_vector(
                self.STATIC_NODE_FEATURES, static_nodes
            )
            dynamic_nodes = self.normalizer.normalize_feature_vector(
                self.DYNAMIC_NODE_FEATURES, dynamic_nodes
            )
            static_edges = self.normalizer.normalize_feature_vector(
                self.STATIC_EDGE_FEATURES, static_edges
            )
            dynamic_edges = self.normalizer.normalize_feature_vector(
                self.DYNAMIC_EDGE_FEATURES, dynamic_edges
            )

            # 1D Normalization
            static_nodes_1d = self.normalizer.normalize_feature_vector(
                self.STATIC_1D_NODE_FEATURES, static_nodes_1d
            )
            dynamic_nodes_1d = self.normalizer.normalize_feature_vector(
                self.DYNAMIC_1D_NODE_FEATURES, dynamic_nodes_1d
            )
            static_edges_1d = self.normalizer.normalize_feature_vector(
                self.STATIC_1D_EDGE_FEATURES, static_edges_1d
            )
            dynamic_edges_1d = self.normalizer.normalize_feature_vector(
                self.DYNAMIC_1D_EDGE_FEATURES, dynamic_edges_1d
            )

        # Save Constant Values
        np.savez(
            self.processed_paths[3],
            edge_index=edge_index,
            static_nodes=static_nodes,
            static_edges=static_edges,
            # 1D Constants
            edge_index_1d=edge_index_1d,
            edge_index_1d_2d=edge_index_1d_2d,
            static_nodes_1d=static_nodes_1d,
            static_edges_1d=static_edges_1d,
        )
        self.log_func(f"Saved constant values to {self.processed_paths[3]}")

        start_idx = 0
        for i, run_id in enumerate(self.hec_ras_run_ids):
            end_idx = start_idx + self._event_num_timesteps[i]
            event_timesteps = all_event_timesteps[start_idx:end_idx].copy()

            # 2D Slices
            event_dynamic_nodes = dynamic_nodes[start_idx:end_idx].copy()
            event_dynamic_edges = dynamic_edges[start_idx:end_idx].copy()
            event_rainfall_per_ts = node_rainfall_per_ts[start_idx:end_idx].copy()

            # 1D Slices
            event_dynamic_nodes_1d = dynamic_nodes_1d[start_idx:end_idx].copy()
            event_dynamic_edges_1d = dynamic_edges_1d[start_idx:end_idx].copy()

            save_path = self.processed_paths[i + 4]
            np.savez(
                save_path,
                event_timesteps=event_timesteps,
                dynamic_nodes=event_dynamic_nodes,
                dynamic_edges=event_dynamic_edges,
                node_rainfall_per_ts=event_rainfall_per_ts,
                # 1D Dynamic
                dynamic_nodes_1d=event_dynamic_nodes_1d,
                dynamic_edges_1d=event_dynamic_edges_1d,
            )
            self.log_func(f"Saved dynamic values for event {run_id} to {save_path}")

            start_idx = end_idx

        self._save_event_stats()
        self.log_func(f"Saved event stats to {self.processed_paths[0]}")
        if self.mode == "train":
            self.normalizer.save_feature_stats()
            self.log_func(f"Saved feature stats to {self.processed_paths[1]}")
        self.boundary_condition.save_data()
        self.log_func(f"Saved boundary condition info to {self.processed_paths[2]}")

        if self.save:
            saved_output_dir = os.path.join(
                self.processed_dir, "features_csv", self.mode
            )
            self.save_features_to_csv(saved_output_dir)

    def len(self):
        return self.total_rollout_timesteps

    def get(self, idx):
        # Load constant data
        constant_values = np.load(self.processed_paths[3])
        edge_index: ndarray = constant_values["edge_index"]
        static_nodes: ndarray = constant_values["static_nodes"].astype(np.float32)
        static_edges: ndarray = constant_values["static_edges"].astype(np.float32)

        # Load 1D constant data
        edge_index_1d: ndarray = constant_values["edge_index_1d"]
        edge_index_1d_2d: ndarray = constant_values["edge_index_1d_2d"]
        static_nodes_1d: ndarray = constant_values["static_nodes_1d"].astype(np.float32)
        static_edges_1d: ndarray = constant_values["static_edges_1d"].astype(np.float32)

        # Find the event this index belongs to using the start indices
        if idx < 0 or idx >= self.total_rollout_timesteps:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {self.total_rollout_timesteps} timesteps."
            )
        start_idx = 0
        for si in self.event_start_idx:
            if idx < si:
                break
            start_idx = si
        event_idx = self.event_start_idx.index(start_idx)

        # Load dynamic data
        dynamic_values_path = self.processed_paths[event_idx + 4]
        dynamic_values = np.load(dynamic_values_path)
        event_timesteps: ndarray = dynamic_values["event_timesteps"]
        dynamic_nodes: ndarray = dynamic_values["dynamic_nodes"].astype(np.float32)
        dynamic_edges: ndarray = dynamic_values["dynamic_edges"].astype(np.float32)

        # Load 1D dynamic data
        dynamic_nodes_1d: ndarray = dynamic_values["dynamic_nodes_1d"].astype(
            np.float32
        )
        dynamic_edges_1d: ndarray = dynamic_values["dynamic_edges_1d"].astype(
            np.float32
        )

        # Create Data object for timestep
        within_event_idx = (
            idx - start_idx + self.previous_timesteps
        )  # First timestep starts at self.previous_timesteps
        timestep = event_timesteps[within_event_idx]

        node_features = self._get_2d_node_timestep_data(
            static_nodes, dynamic_nodes, within_event_idx
        )
        edge_features = self._get_2d_edge_timestep_data(
            static_edges, dynamic_edges, within_event_idx
        )

        # 1D Data Construction
        node_features_1d = self._get_1d_node_timestep_data(
            static_nodes_1d, dynamic_nodes_1d, within_event_idx
        )
        edge_features_1d = self._get_1d_edge_timestep_data(
            static_edges_1d, dynamic_edges_1d, within_event_idx
        )

        label_nodes, label_edges, label_1d_nodes, label_1d_edges = (
            self._get_timestep_labels(
                dynamic_nodes,
                dynamic_edges,
                dynamic_nodes_1d,
                dynamic_edges_1d,
                within_event_idx,
            )
        )

        # Get physics-informed loss information
        global_mass_info = None
        local_mass_info = None
        if self.with_global_mass_loss or self.with_local_mass_loss:
            node_rainfall_per_ts: ndarray = dynamic_values["node_rainfall_per_ts"]
            if self.with_global_mass_loss:
                global_mass_info = self._get_global_mass_info_for_timestep(
                    node_rainfall_per_ts, within_event_idx
                )

            if self.with_local_mass_loss:
                local_mass_info = self._get_local_mass_info_for_timestep(
                    node_rainfall_per_ts, within_event_idx
                )

        edge_index = torch.from_numpy(edge_index).long()
        edge_index_1d = torch.from_numpy(edge_index_1d).long()
        edge_index_1d_2d = torch.from_numpy(edge_index_1d_2d).long()

        data = Data(
            # 2D
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=label_nodes,
            y_edge=label_edges,
            # 1D
            x_1d=node_features_1d,
            edge_index_1d=edge_index_1d,
            edge_attr_1d=edge_features_1d,
            y_1d=label_1d_nodes,
            y_1d_edge=label_1d_edges,
            # 1D-2D
            edge_index_1d_2d=edge_index_1d_2d,
            # Meta
            timestep=timestep,
            global_mass_info=global_mass_info,
            local_mass_info=local_mass_info,
        )

        return data

    def _load_event_stats(
        self, root_dir: str, event_stats_file: str
    ) -> Tuple[List[int], int, Dict]:
        event_stats_path = os.path.join(root_dir, "processed", event_stats_file)
        if not os.path.exists(event_stats_path):
            return [], 0, {}

        event_stats = read_yaml_file(event_stats_path)
        event_start_idx = event_stats["event_start_idx"]
        total_rollout_timesteps = event_stats["total_rollout_timesteps"]
        processed_event_info = {
            "timestep_interval": event_stats["timestep_interval"],
            "previous_timesteps": event_stats["previous_timesteps"],
            "normalize": event_stats["normalize"],
            "spin_up_time": event_stats["spin_up_time"],
            "time_from_peak": event_stats["time_from_peak"],
            "inflow_boundary_nodes": event_stats["inflow_boundary_nodes"],
            "outflow_boundary_nodes": event_stats["outflow_boundary_nodes"],
            # Backward-compatible: older stats files predate unit conversion.
            "converted_units": event_stats.get("converted_units", None),
        }
        return event_start_idx, total_rollout_timesteps, processed_event_info

    def _save_event_stats(self):
        event_stats = {
            "event_start_idx": self.event_start_idx,
            "total_rollout_timesteps": self.total_rollout_timesteps,
            "timestep_interval": self.timestep_interval,
            "previous_timesteps": self.previous_timesteps,
            "normalize": self.is_normalized,
            "spin_up_time": self.spin_up_time,
            "time_from_peak": self.time_from_peak,
            "inflow_boundary_nodes": self.inflow_boundary_nodes,
            "outflow_boundary_nodes": self.outflow_boundary_nodes,
            # Records the target unit system the features were converted to
            # (None when no conversion was applied).
            "converted_units": (
                self.unit_converter.target_system
                if self.unit_converter is not None
                else None
            ),
        }
        save_to_yaml_file(self.processed_paths[0], event_stats)

    def _get_hecras_files_from_summary(
        self, root_dir: str, dataset_summary_file: str
    ) -> Tuple[List[str], List[str]]:
        """Assumes all HEC-RAS files in the dataset summary are from the same catchment"""
        dataset_summary_path = os.path.join(root_dir, "raw", dataset_summary_file)
        summary_df = pd.read_csv(dataset_summary_path)
        assert len(summary_df) > 0, (
            f"No data found in summary file: {dataset_summary_path}"
        )

        hec_ras_run_ids = []
        hec_ras_files = []
        for _, row in summary_df.iterrows():
            run_id = row["Run_ID"]
            hec_ras_path = row["HECRAS_Filepath"]

            assert run_id not in hec_ras_run_ids, f"Duplicate Run_ID found: {run_id}"
            full_hec_ras_path = os.path.join(root_dir, "raw", hec_ras_path)
            assert os.path.exists(full_hec_ras_path), (
                f"HECRAS file not found: {hec_ras_path}"
            )

            hec_ras_run_ids.append(run_id)
            hec_ras_files.append(hec_ras_path)

        return hec_ras_files, hec_ras_run_ids

    def _is_previous_config_different(self, processed_event_info: Dict) -> bool:
        if processed_event_info is None or len(processed_event_info) == 0:
            self.log_func("No previous event stats found. Processing dataset.")
            return True
        if processed_event_info["timestep_interval"] != self.timestep_interval:
            self.log_func(
                f"Previous timestep interval {processed_event_info['timestep_interval']} differs from current {self.timestep_interval}. Reprocessing dataset."
            )
            return True
        if processed_event_info["previous_timesteps"] != self.previous_timesteps:
            self.log_func(
                f"Previous previous_timesteps {processed_event_info['previous_timesteps']} differs from current {self.previous_timesteps}. Reprocessing dataset."
            )
            return True
        if processed_event_info["normalize"] != self.is_normalized:
            self.log_func(
                f"Previous normalize {processed_event_info['normalize']} differs from current {self.is_normalized}. Reprocessing dataset."
            )
            return True
        if processed_event_info["spin_up_time"] != self.spin_up_time:
            self.log_func(
                f"Previous spin_up_time {processed_event_info['spin_up_time']} differs from current {self.spin_up_time}. Reprocessing dataset."
            )
            return True
        if processed_event_info["time_from_peak"] != self.time_from_peak:
            self.log_func(
                f"Previous time_from_peak {processed_event_info['time_from_peak']} differs from current {self.time_from_peak}. Reprocessing dataset."
            )
            return True
        current_converted_units = (
            self.unit_converter.target_system
            if self.unit_converter is not None
            else None
        )
        if processed_event_info.get("converted_units", None) != current_converted_units:
            self.log_func(
                f"Previous converted_units {processed_event_info.get('converted_units', None)} differs from current {current_converted_units}. Reprocessing dataset."
            )
            return True
        # if set(processed_event_info['inflow_boundary_nodes']) != set(self.inflow_boundary_nodes):
        #     self.log_func(f'Previous inflow_boundary_nodes {processed_event_info["inflow_boundary_nodes"]} differs from current {self.inflow_boundary_nodes}. Reprocessing dataset.')
        #     return True
        # if set(processed_event_info['outflow_boundary_nodes']) != set(self.outflow_boundary_nodes):
        #     self.log_func(f'Previous outflow_boundary_nodes {processed_event_info["outflow_boundary_nodes"]} differs from current {self.outflow_boundary_nodes}. Reprocessing dataset.')
        #     return True
        return False

    # =========== process() methods ===========

    def _set_event_properties(self) -> ndarray:
        self._event_peak_idx = []
        self._event_num_timesteps = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        current_total_ts = 0
        all_event_timesteps = []
        for event_idx, hec_ras_path in enumerate(self.raw_paths[6:]):
            timesteps = get_event_timesteps(hec_ras_path)
            event_ts_interval = int((timesteps[1] - timesteps[0]).total_seconds())
            assert self.timestep_interval % event_ts_interval == 0, (
                f"Event {self.hec_ras_run_ids[event_idx]} has a timestep interval of {event_ts_interval} seconds, which is not compatible with the dataset timestep interval of {self.timestep_interval} seconds."
            )
            self._event_base_timestep_interval.append(event_ts_interval)

            water_volume = get_water_volume(
                hec_ras_path, perimeter_name=self.perimeter_name
            )
            total_water_volume = water_volume.sum(axis=1)
            peak_idx = np.argmax(total_water_volume).item()
            num_timesteps_after_peak = (
                self.time_from_peak // event_ts_interval
                if self.time_from_peak is not None
                else 0
            )
            assert peak_idx + num_timesteps_after_peak < len(timesteps), (
                "Timesteps after peak exceeds the available timesteps."
            )
            self._event_peak_idx.append(peak_idx)

            timesteps = self._get_trimmed_dynamic_data(
                timesteps, event_idx, aggr="first"
            )
            all_event_timesteps.append(timesteps)

            num_timesteps = len(timesteps)
            self._event_num_timesteps.append(num_timesteps)

            event_total_rollout_ts = (
                num_timesteps - self.previous_timesteps - 1
            )  # First timestep starts at self.previous_timesteps; Last timestep is used for labels
            assert event_total_rollout_ts > 0, (
                f"Event {event_idx} has too few timesteps."
            )
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        assert len(self._event_peak_idx) == len(self.hec_ras_run_ids), (
            "Mismatch in number of events and peak indices."
        )
        assert len(self._event_num_timesteps) == len(self.hec_ras_run_ids), (
            "Mismatch in number of events and number of timesteps."
        )
        assert len(self.event_start_idx) == len(self.hec_ras_run_ids), (
            "Mismatch in number of events and start indices."
        )

        all_event_timesteps = np.concatenate(all_event_timesteps, axis=0)
        return all_event_timesteps

    def _get_edge_index(self, filepath) -> ndarray:
        edge_index = get_edge_index(filepath)
        return edge_index

    def _get_1d_edge_index(self, filepath, nodes_filepath) -> ndarray:
        edge_index = get_1d_edge_index(filepath, nodes_filepath)
        return edge_index

    def _get_1d_2d_edge_index(self, filepath) -> ndarray:
        edge_index = get_1d2d_edge_index(filepath)
        return edge_index

    # =========== 1D Feature Extraction Methods ===========

    def _get_static_1d_node_features(self) -> ndarray:
        STATIC_1D_NODE_RETRIEVAL_MAP = {
            "position_x": lambda: get_cell_position_x(self.raw_paths[0]),
            "position_y": lambda: get_cell_position_y(self.raw_paths[0]),
            "depth": lambda: get_1d_cell_depth(self.raw_paths[0]),
            "invert_elevation": lambda: get_1d_invert_elevation(self.raw_paths[0]),
            "surface_elevation": lambda: get_1d_surface_elevation(self.raw_paths[0]),
            "base_area": lambda: get_1d_base_area(self.raw_paths[0]),
        }
        static_features = self._get_features(
            feature_list=self.STATIC_1D_NODE_FEATURES,
            feature_retrieval_map=STATIC_1D_NODE_RETRIEVAL_MAP,
        )
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_static_1d_edge_features(self) -> ndarray:
        STATIC_1D_EDGE_RETRIEVAL_MAP = {
            "length": lambda: get_1d_edge_length(self.raw_paths[1]),
            "shape": lambda: get_1d_edge_shape(self.raw_paths[1]),
            "diameter": lambda: get_1d_edge_diameter(self.raw_paths[1]),
            "roughness": lambda: get_1d_edge_manning(self.raw_paths[1]),
            "slope": lambda: get_1d_edge_slope(self.raw_paths[1]),
            "relative_position_x": lambda: get_1d_edge_relative_position(
                "x", self.raw_paths[0], self.raw_paths[1]
            ),
            "relative_position_y": lambda: get_1d_edge_relative_position(
                "y", self.raw_paths[0], self.raw_paths[1]
            ),
        }
        static_features = self._get_features(
            feature_list=self.STATIC_1D_EDGE_FEATURES,
            feature_retrieval_map=STATIC_1D_EDGE_RETRIEVAL_MAP,
        )
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_dynamic_1d_node_features(self) -> ndarray:
        def remap_1d_nodes(retrieval_func):
            """Wrap a retrieval function to reorder its node axis using the 1D node mapping."""

            def wrapper(hec_ras_path, network_name):
                data = retrieval_func(hec_ras_path, network_name=network_name)
                # data shape: (timesteps, num_nodes)
                # mapping: {fid: hec_ras_col_idx}
                # Build an index array: output[t, fid] = data[t, hec_ras_col_idx]
                num_fids = len(self.node_1d_mapping)
                reordered = np.empty((data.shape[0], num_fids), dtype=data.dtype)
                for fid, hec_col in self.node_1d_mapping.items():
                    reordered[:, fid] = data[:, hec_col]
                return reordered

            return wrapper

        DYNAMIC_1D_NODE_RETRIEVAL_MAP = {
            "water_level": lambda: self._get_dynamic_from_all_unique_1d_events(
                remap_1d_nodes(get_1d_water_level), aggr="mean"
            ),
            "inlet_flow": lambda: self._get_dynamic_from_all_unique_1d_events(
                remap_1d_nodes(get_1d_inlet_flow), aggr="mean"
            ),
        }
        dynamic_features = self._get_features(
            feature_list=self.DYNAMIC_1D_NODE_FEATURES,
            feature_retrieval_map=DYNAMIC_1D_NODE_RETRIEVAL_MAP,
        )
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_1d_edge_features(self) -> ndarray:
        DYNAMIC_1D_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_unique_1d_events(
                get_1d_velocity, aggr="mean"
            ),
            "flow": lambda: self._get_dynamic_from_all_unique_1d_events(
                get_1d_flow, aggr="mean"
            ),
        }
        dynamic_features = self._get_features(
            feature_list=self.DYNAMIC_1D_EDGE_FEATURES,
            feature_retrieval_map=DYNAMIC_1D_EDGE_RETRIEVAL_MAP,
        )
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    # =========== 2D Feature Extraction Methods ===========

    def _get_static_node_features(self) -> ndarray:
        def _get_dem_based_feature(
            node_shp_path: str,
            dem_path: str,
            feature_func: Callable,
            *output_filenames: Tuple[str],
        ) -> ndarray:
            pos = get_cell_position(node_shp_path)
            dem_folder = os.path.dirname(dem_path)
            filled_dem_path = os.path.join(dem_folder, "filled_dem.tif")
            filled_dem = get_filled_dem(dem_path, filled_dem_path)

            output_paths = [os.path.join(dem_folder, fn) for fn in output_filenames]
            return feature_func(filled_dem, *output_paths, pos)

        def _get_aspect(nodes_shp_path: str, dem_path: str):
            return _get_dem_based_feature(
                nodes_shp_path, dem_path, get_aspect, "aspect_dem.tif"
            )

        def _get_curvature(nodes_shp_path: str, dem_path: str):
            return _get_dem_based_feature(
                nodes_shp_path, dem_path, get_curvature, "curvature_dem.tif"
            )

        def _get_flow_accumulation(nodes_shp_path: str, dem_path: str):
            return _get_dem_based_feature(
                nodes_shp_path,
                dem_path,
                get_flow_accumulation,
                "flow_dir_dem.tif",
                "flow_acc_dem.tif",
            )

        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(
                self.raw_paths[6], perimeter_name=self.perimeter_name
            ),
            "roughness": lambda: get_roughness(
                self.raw_paths[6], perimeter_name=self.perimeter_name
            ),
            "elevation": lambda: get_cell_elevation(self.raw_paths[3]),
            # "min_elevation": lambda: get_min_elevation(self.raw_paths[3]),
            "position_x": lambda: get_cell_position_x(self.raw_paths[3]),
            "position_y": lambda: get_cell_position_y(self.raw_paths[3]),
            "aspect": lambda: _get_aspect(self.raw_paths[3], self.raw_paths[5]),
            "curvature": lambda: _get_curvature(self.raw_paths[3], self.raw_paths[5]),
            "flow_accumulation": lambda: _get_flow_accumulation(
                self.raw_paths[3], self.raw_paths[5]
            ),
        }

        static_features = self._get_features(
            feature_list=self.STATIC_NODE_FEATURES,
            feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP,
        )
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_static_edge_features(self) -> ndarray:
        STATIC_EDGE_RETRIEVAL_MAP = {
            "direction_x": lambda: get_edge_direction_x(self.raw_paths[4]),
            "direction_y": lambda: get_edge_direction_y(self.raw_paths[4]),
            "face_length": lambda: get_face_length(self.raw_paths[4]),
            "length": lambda: get_edge_length(self.raw_paths[4]),
            "slope": lambda: get_edge_slope(self.raw_paths[4]),
            "relative_position_x": lambda: get_relative_position(
                "x", self.raw_paths[3], self.raw_paths[4]
            ),
            "relative_position_y": lambda: get_relative_position(
                "y", self.raw_paths[3], self.raw_paths[4]
            ),
        }

        static_features = self._get_features(
            feature_list=self.STATIC_EDGE_FEATURES,
            feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP,
        )
        static_features = np.array(static_features).transpose()
        return static_features

    def _get_dynamic_node_features(self) -> ndarray:
        def get_interval_rainfall(hec_ras_path: str):
            """Get rainfall"""
            cumulative_rainfall = get_rainfall(
                hec_ras_path, perimeter_name=self.perimeter_name
            ).astype(np.float64)
            interval_rainfall = np.empty_like(cumulative_rainfall)
            interval_rainfall[:] = np.nan
            interval_rainfall[1:] = cumulative_rainfall[1:] - cumulative_rainfall[:-1]
            interval_rainfall[0] = cumulative_rainfall[0]
            return interval_rainfall

        edge_index = get_edge_index(self.raw_paths[4])
        num_nodes = edge_index.max() + 1
        inflow_to_boundary_mask = np.isin(edge_index[1], self.inflow_boundary_nodes)
        inflow_edges_mask = np.any(
            np.isin(edge_index, self.inflow_boundary_nodes), axis=0
        )

        def get_inflow_hydrograph(hec_ras_path: str):
            """Get inflow at boundary nodes"""
            face_flow = get_face_flow(hec_ras_path, perimeter_name=self.perimeter_name)
            if np.any(inflow_to_boundary_mask):
                # Flip the dynamic edge features accordingly
                face_flow[:, inflow_to_boundary_mask] *= -1
            inflow = face_flow[:, inflow_edges_mask].sum(axis=1)[:, None]
            inflow = np.repeat(inflow, num_nodes, axis=-1)
            return inflow

        def get_water_depth(hec_ras_path: str):
            """Get water depth from water level and elevation"""
            water_level = get_water_level(
                hec_ras_path, perimeter_name=self.perimeter_name
            )
            elevation = get_cell_elevation(self.raw_paths[3])[None, :]
            water_depth = np.clip(water_level - elevation, a_min=0, a_max=None)
            return water_depth

        def get_clipped_water_volume(hec_ras_path: str):
            """Remove exterme values in water volume"""
            CLIP_VOLUME = 100000  # in cubic meters
            water_volume = get_water_volume(
                hec_ras_path, perimeter_name=self.perimeter_name
            )
            water_volume = np.clip(water_volume, a_min=0, a_max=CLIP_VOLUME)
            return water_volume

        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "inflow": lambda: self._get_dynamic_from_all_events(
                get_inflow_hydrograph, aggr="mean"
            ),
            "rainfall": lambda: self._get_dynamic_from_all_events(
                get_interval_rainfall, aggr="sum"
            ),
            "water_depth": lambda: self._get_dynamic_from_all_events(
                get_water_depth, aggr="mean"
            ),
            "water_volume": lambda: self._get_dynamic_from_all_events(
                get_clipped_water_volume, aggr="mean"
            ),
            "water_level": lambda: self._get_dynamic_from_all_unique_events(
                get_water_level, aggr="mean"
            ),
        }

        dynamic_features = self._get_features(
            feature_list=self.DYNAMIC_NODE_FEATURES,
            feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP,
        )
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self) -> ndarray:
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "velocity": lambda: self._get_dynamic_from_all_unique_events(
                get_velocity, aggr="mean"
            ),
            "flow": lambda: self._get_dynamic_from_all_unique_events(
                get_face_flow, aggr="mean"
            ),
        }

        dynamic_features = self._get_features(
            feature_list=self.DYNAMIC_EDGE_FEATURES,
            feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP,
        )
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_from_all_events(
        self, retrieval_func: Callable, aggr: str = "first"
    ) -> ndarray:
        all_event_data = []
        for i, hec_ras_path in enumerate(self.raw_paths[6:]):
            event_data = retrieval_func(hec_ras_path)
            event_data = self._get_trimmed_dynamic_data(event_data, i, aggr)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_dynamic_from_all_unique_events(
        self, retrieval_func: Callable, aggr: str = "first"
    ) -> ndarray:
        all_event_data = []
        for i, hec_ras_path in enumerate(self.raw_paths[6:]):
            event_data = retrieval_func(
                hec_ras_path, perimeter_name=self.perimeter_name
            )
            event_data = self._get_trimmed_dynamic_data(event_data, i, aggr)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_dynamic_from_all_unique_1d_events(
        self, retrieval_func: Callable, aggr: str = "first"
    ) -> ndarray:
        all_event_data = []
        for i, hec_ras_path in enumerate(self.raw_paths[6:]):
            event_data = retrieval_func(hec_ras_path, network_name=self.network_name)
            event_data = self._get_trimmed_dynamic_data(event_data, i, aggr)
            all_event_data.append(event_data)
        all_event_data = np.concatenate(all_event_data, axis=0)
        return all_event_data

    def _get_trimmed_dynamic_data(
        self, dynamic_data: ndarray, event_idx: int, aggr: str = "first"
    ) -> ndarray:
        start = 0
        if self.spin_up_time is not None:
            if isinstance(self.spin_up_time, int):
                start = (
                    self.spin_up_time // self._event_base_timestep_interval[event_idx]
                )
            elif isinstance(self.spin_up_time, dict):
                run_id = self.hec_ras_run_ids[event_idx]
                if run_id not in self.spin_up_time:
                    if "default" in self.spin_up_time:
                        run_id = "default"
                    else:
                        self.log_func(
                            f"WARNING: No spin-up timesteps defined for Run ID {run_id} and no default value in dict. Setting start to 0."
                        )
                start = (
                    self.spin_up_time.get(run_id, 0)
                    // self._event_base_timestep_interval[event_idx]
                )
            else:
                raise ValueError(
                    f"Invalid type for spin_up_time: {type(self.spin_up_time)}"
                )

        end = None
        if self.time_from_peak is not None:
            event_peak = self._event_peak_idx[event_idx]
            timesteps_from_peak = (
                self.time_from_peak // self._event_base_timestep_interval[event_idx]
            )
            end = event_peak + timesteps_from_peak

        trimmed = dynamic_data[start:end]

        step = self.timestep_interval // self._event_base_timestep_interval[event_idx]
        downsampled = self._downsample_dynamic_data(trimmed, step, aggr)

        return downsampled

    def _downsample_dynamic_data(
        self, dynamic_data: ndarray, step: int, aggr: str = "first"
    ) -> ndarray:
        if step == 1:
            return dynamic_data

        # Trim array to be divisible by step
        trimmed_length = (dynamic_data.shape[0] // step) * step
        trimmed_array = dynamic_data[:trimmed_length]

        if aggr == "first":
            return trimmed_array[::step]

        elif aggr in ["mean", "sum"]:
            # Reshape to group consecutive elements
            if dynamic_data.ndim == 1:
                reshaped = trimmed_array.reshape(-1, step)  # (timesteps, step)
            else:
                reshaped = trimmed_array.reshape(
                    -1, step, dynamic_data.shape[1]
                )  # (timesteps, step, feature)

            if aggr == "mean":
                return np.mean(reshaped, axis=1)
            elif aggr == "sum":
                return np.sum(reshaped, axis=1)

        raise ValueError(f"Aggregation method '{aggr}' is not supported")

    def _get_features(
        self, feature_list: List[str], feature_retrieval_map: Dict[str, Callable]
    ) -> List:
        features = []
        for feature in (
            feature_list
        ):  # Order in feature list determines the order of features in the output
            if feature not in feature_retrieval_map:
                continue

            feature_data: ndarray = feature_retrieval_map[feature]()
            features.append(feature_data)

        return features

    def _get_physics_info(self, dynamic_nodes: ndarray) -> ndarray:
        # Denormalized Rainfall
        rainfall_idx = self.DYNAMIC_NODE_FEATURES.index("rainfall")
        node_rainfall_per_ts = dynamic_nodes[:, :, rainfall_idx]

        return node_rainfall_per_ts

    # =========== get() methods ===========

    def _get_1d_node_timestep_data(
        self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int
    ) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(
            dynamic_features, self.DYNAMIC_1D_NODE_FEATURES, timestep_idx
        )
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_1d_edge_timestep_data(
        self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int
    ) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(
            dynamic_features, self.DYNAMIC_1D_EDGE_FEATURES, timestep_idx
        )
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_2d_node_timestep_data(
        self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int
    ) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(
            dynamic_features, self.DYNAMIC_NODE_FEATURES, timestep_idx
        )
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_2d_edge_timestep_data(
        self, static_features: ndarray, dynamic_features: ndarray, timestep_idx: int
    ) -> Tensor:
        ts_dynamic_features = self._get_timestep_dynamic_features(
            dynamic_features, self.DYNAMIC_EDGE_FEATURES, timestep_idx
        )
        return self._get_timestep_features(static_features, ts_dynamic_features)

    def _get_timestep_dynamic_features(
        self,
        dynamic_features: ndarray,
        dynamic_feature_list: List[str],
        timestep_idx: int,
    ) -> Tensor:
        _, num_elems, _ = dynamic_features.shape
        if timestep_idx < self.previous_timesteps:
            padding = self._get_empty_feature_tensor(
                dynamic_feature_list,
                (self.previous_timesteps - timestep_idx, num_elems),
                dtype=dynamic_features.dtype,
            )
            ts_dynamic_features = np.concat(
                [padding, dynamic_features[: timestep_idx + 1, :, :]], axis=0
            )
        else:
            ts_dynamic_features = dynamic_features[
                timestep_idx - self.previous_timesteps : timestep_idx + 1, :, :
            ]
        return ts_dynamic_features

    def _get_timestep_features(
        self, static_features: ndarray, ts_dynamic_features: ndarray
    ) -> Tensor:
        """Returns the data for a specific timestep in the format [static_features, dynamic_features (previous, current)]"""
        _, num_elems, _ = ts_dynamic_features.shape

        # (num_elems,  num_dynamic_features * num_timesteps)
        ts_dynamic_features = ts_dynamic_features.transpose(1, 0, 2)
        ts_dynamic_features = np.reshape(
            ts_dynamic_features, shape=(num_elems, -1), order="F"
        )

        ts_data = np.concat([static_features, ts_dynamic_features], axis=1)
        return torch.from_numpy(ts_data.astype(np.float32))

    def _get_empty_feature_tensor(
        self,
        features: List[str],
        other_dims: Tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> ndarray:
        if not self.is_normalized:
            return np.zeros((*other_dims, len(features)), dtype=dtype)
        return self.normalizer.get_normalized_zero_tensor(features, other_dims, dtype)

    def _get_timestep_labels(
        self,
        node_dynamic_features: ndarray,
        edge_dynamic_features: ndarray,
        node_1d_dynamic_features: ndarray,
        edge_1d_dynamic_features: ndarray,
        timestep_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        label_nodes_idx = self.DYNAMIC_NODE_FEATURES.index(self.NODE_TARGET_FEATURE)
        # (num_nodes, 1)
        assert timestep_idx + 1 < node_dynamic_features.shape[0], (
            "Attempting to access node label beyond available timesteps"
        )
        current_nodes = node_dynamic_features[timestep_idx, :, label_nodes_idx][:, None]
        next_nodes = node_dynamic_features[timestep_idx + 1, :, label_nodes_idx][
            :, None
        ]
        label_nodes = next_nodes - current_nodes
        label_nodes = torch.from_numpy(label_nodes)

        label_edges_idx = self.DYNAMIC_EDGE_FEATURES.index(self.EDGE_TARGET_FEATURE)
        # (num_edges, 1)
        assert timestep_idx + 1 < edge_dynamic_features.shape[0], (
            "Attempting to access edge label beyond available timesteps"
        )
        current_edges = edge_dynamic_features[timestep_idx, :, label_edges_idx][:, None]
        next_edges = edge_dynamic_features[timestep_idx + 1, :, label_edges_idx][
            :, None
        ]
        label_edges = next_edges - current_edges
        label_edges = torch.from_numpy(label_edges)

        label_1d_nodes_idx = self.DYNAMIC_1D_NODE_FEATURES.index(
            self.NODE_1D_TARGET_FEATURE
        )
        # (num_nodes, 1)
        assert timestep_idx + 1 < node_1d_dynamic_features.shape[0], (
            "Attempting to access node label beyond available timesteps"
        )
        current_1d_nodes = node_1d_dynamic_features[
            timestep_idx, :, label_1d_nodes_idx
        ][:, None]
        next_1d_nodes = node_1d_dynamic_features[
            timestep_idx + 1, :, label_1d_nodes_idx
        ][:, None]
        label_1d_nodes = next_1d_nodes - current_1d_nodes
        label_1d_nodes = torch.from_numpy(label_1d_nodes)

        label_1d_edges_idx = self.DYNAMIC_1D_EDGE_FEATURES.index(
            self.EDGE_1D_TARGET_FEATURE
        )
        # (num_edges, 1)
        assert timestep_idx + 1 < edge_1d_dynamic_features.shape[0], (
            "Attempting to access edge label beyond available timesteps"
        )
        current_1d_edges = edge_1d_dynamic_features[
            timestep_idx, :, label_1d_edges_idx
        ][:, None]
        next_1d_edges = edge_1d_dynamic_features[
            timestep_idx + 1, :, label_1d_edges_idx
        ][:, None]
        label_1d_edges = next_1d_edges - current_1d_edges
        label_1d_edges = torch.from_numpy(label_1d_edges)

        return label_nodes, label_edges, label_1d_nodes, label_1d_edges

    def _get_global_mass_info_for_timestep(
        self, node_rainfall_per_ts: ndarray, timestep_idx: int
    ) -> Dict[str, Tensor]:
        non_boundary_nodes_mask = ~self.boundary_condition.boundary_nodes_mask
        total_rainfall = node_rainfall_per_ts[
            timestep_idx, non_boundary_nodes_mask
        ].sum(keepdims=True)

        total_rainfall = torch.from_numpy(total_rainfall)
        inflow_edges_mask = torch.from_numpy(self.boundary_condition.inflow_edges_mask)
        outflow_edges_mask = torch.from_numpy(
            self.boundary_condition.outflow_edges_mask
        )
        non_boundary_nodes_mask = torch.from_numpy(non_boundary_nodes_mask)

        return {
            "total_rainfall": total_rainfall,
            "inflow_edges_mask": inflow_edges_mask,
            "outflow_edges_mask": outflow_edges_mask,
            "non_boundary_nodes_mask": non_boundary_nodes_mask,
        }

    def _get_local_mass_info_for_timestep(
        self, node_rainfall_per_ts: ndarray, timestep_idx: int
    ) -> Dict[str, Tensor]:
        rainfall = node_rainfall_per_ts[timestep_idx]

        rainfall = torch.from_numpy(rainfall)
        non_boundary_nodes_mask = torch.from_numpy(
            ~self.boundary_condition.boundary_nodes_mask
        )

        return {
            "rainfall": rainfall,
            "non_boundary_nodes_mask": non_boundary_nodes_mask,
        }

    def save_features_to_csv(self, output_dir: str = None):
        """
        Save all processed features to CSV files for inspection and analysis.

        Args:
            output_dir: Directory to save CSV files. If None, uses processed_dir/features_csv/
        """
        if output_dir is None:
            output_dir = os.path.join(self.processed_dir, "features_csv")

        os.makedirs(output_dir, exist_ok=True)

        self.log_func(f"Saving features to CSV in {output_dir}...")

        # Load constant values
        constant_values = np.load(self.processed_paths[3])

        # --- Save 2D Static Features ---
        static_nodes_2d = constant_values["static_nodes"]
        static_nodes_2d_df = pd.DataFrame(
            static_nodes_2d, columns=self.STATIC_NODE_FEATURES
        )
        static_nodes_2d_df.to_csv(
            os.path.join(output_dir, "2d_nodes_static.csv"), index_label="node_idx"
        )
        self.log_func(f"  Saved 2D static nodes: {static_nodes_2d_df.shape}")

        static_edges_2d = constant_values["static_edges"]
        static_edges_2d_df = pd.DataFrame(
            static_edges_2d, columns=self.STATIC_EDGE_FEATURES
        )
        static_edges_2d_df.to_csv(
            os.path.join(output_dir, "2d_edges_static.csv"), index_label="edge_idx"
        )
        self.log_func(f"  Saved 2D static edges: {static_edges_2d_df.shape}")

        # --- Save 2D Edge Index ---
        edge_index_2d = constant_values["edge_index"]
        edge_index_2d_df = pd.DataFrame(
            edge_index_2d.T, columns=["from_node", "to_node"]
        )
        edge_index_2d_df.to_csv(
            os.path.join(output_dir, "2d_edge_index.csv"), index_label="edge_idx"
        )
        self.log_func(f"  Saved 2D edge index: {edge_index_2d_df.shape}")

        # --- Save 1D Static Features ---
        static_nodes_1d = constant_values["static_nodes_1d"]
        static_nodes_1d_df = pd.DataFrame(
            static_nodes_1d, columns=self.STATIC_1D_NODE_FEATURES
        )
        static_nodes_1d_df.to_csv(
            os.path.join(output_dir, "1d_nodes_static.csv"), index_label="node_idx"
        )
        self.log_func(f"  Saved 1D static nodes: {static_nodes_1d_df.shape}")

        static_edges_1d = constant_values["static_edges_1d"]
        static_edges_1d_df = pd.DataFrame(
            static_edges_1d, columns=self.STATIC_1D_EDGE_FEATURES
        )
        static_edges_1d_df.to_csv(
            os.path.join(output_dir, "1d_edges_static.csv"), index_label="edge_idx"
        )
        self.log_func(f"  Saved 1D static edges: {static_edges_1d_df.shape}")

        # --- Save 1D Edge Indices ---
        edge_index_1d = constant_values["edge_index_1d"]
        edge_index_1d_df = pd.DataFrame(
            edge_index_1d.T, columns=["from_node", "to_node"]
        )
        edge_index_1d_df.to_csv(
            os.path.join(output_dir, "1d_edge_index.csv"), index_label="edge_idx"
        )
        self.log_func(f"  Saved 1D edge index: {edge_index_1d_df.shape}")

        # --- Save 1D-2D Connections ---
        edge_index_1d_2d = constant_values["edge_index_1d_2d"]
        edge_index_1d_2d_df = pd.DataFrame(
            edge_index_1d_2d.T, columns=["node_1d", "node_2d"]
        )
        edge_index_1d_2d_df.to_csv(
            os.path.join(output_dir, "1d2d_connections.csv"),
            index_label="connection_idx",
        )
        self.log_func(f"  Saved 1D-2D connections: {edge_index_1d_2d_df.shape}")

        # --- Save Dynamic Features for Each Event ---
        for event_idx, run_id in enumerate(self.hec_ras_run_ids):
            event_dir = os.path.join(output_dir, f"event_{run_id}")
            os.makedirs(event_dir, exist_ok=True)

            # Load dynamic values for this event
            dynamic_values_path = self.processed_paths[event_idx + 4]
            dynamic_values = np.load(dynamic_values_path, allow_pickle=True)

            # Save 2D dynamic features
            dynamic_nodes_2d = dynamic_values["dynamic_nodes"]
            dynamic_edges_2d = dynamic_values["dynamic_edges"]

            # Save 1D dynamic features
            dynamic_nodes_1d = dynamic_values["dynamic_nodes_1d"]
            dynamic_edges_1d = dynamic_values["dynamic_edges_1d"]

            # Save timesteps
            event_timesteps = dynamic_values["event_timesteps"]
            timesteps_df = pd.DataFrame(
                {
                    "timestep_idx": np.arange(len(event_timesteps)),
                    "timestamp": [str(ts) for ts in event_timesteps],
                }
            )
            timesteps_df.to_csv(os.path.join(event_dir, "timesteps.csv"), index=False)

            num_timesteps = dynamic_nodes_2d.shape[0]
            num_2d_nodes = dynamic_nodes_2d.shape[1]
            num_2d_edges = dynamic_edges_2d.shape[1]
            num_1d_nodes = dynamic_nodes_1d.shape[1]
            num_1d_edges = dynamic_edges_1d.shape[1]

            # --- 2D Nodes: All timesteps in one file ---
            dynamic_nodes_2d_flat = dynamic_nodes_2d.reshape(
                -1, len(self.DYNAMIC_NODE_FEATURES)
            )
            dynamic_nodes_2d_df = pd.DataFrame(
                dynamic_nodes_2d_flat, columns=self.DYNAMIC_NODE_FEATURES
            )
            dynamic_nodes_2d_df["timestep"] = np.repeat(
                np.arange(num_timesteps), num_2d_nodes
            )
            dynamic_nodes_2d_df["node_idx"] = np.tile(
                np.arange(num_2d_nodes), num_timesteps
            )
            # Reorder columns: timestep, node_idx, then features
            cols = ["timestep", "node_idx"] + self.DYNAMIC_NODE_FEATURES
            dynamic_nodes_2d_df = dynamic_nodes_2d_df[cols]
            dynamic_nodes_2d_df.to_csv(
                os.path.join(event_dir, "2d_nodes_dynamic_all.csv"), index=False
            )

            # --- 2D Edges: All timesteps in one file ---
            dynamic_edges_2d_flat = dynamic_edges_2d.reshape(
                -1, len(self.DYNAMIC_EDGE_FEATURES)
            )
            dynamic_edges_2d_df = pd.DataFrame(
                dynamic_edges_2d_flat, columns=self.DYNAMIC_EDGE_FEATURES
            )
            dynamic_edges_2d_df["timestep"] = np.repeat(
                np.arange(num_timesteps), num_2d_edges
            )
            dynamic_edges_2d_df["edge_idx"] = np.tile(
                np.arange(num_2d_edges), num_timesteps
            )
            cols = ["timestep", "edge_idx"] + self.DYNAMIC_EDGE_FEATURES
            dynamic_edges_2d_df = dynamic_edges_2d_df[cols]
            dynamic_edges_2d_df.to_csv(
                os.path.join(event_dir, "2d_edges_dynamic_all.csv"), index=False
            )

            # --- 1D Nodes: All timesteps in one file ---
            dynamic_nodes_1d_flat = dynamic_nodes_1d.reshape(
                -1, len(self.DYNAMIC_1D_NODE_FEATURES)
            )
            dynamic_nodes_1d_df = pd.DataFrame(
                dynamic_nodes_1d_flat, columns=self.DYNAMIC_1D_NODE_FEATURES
            )
            dynamic_nodes_1d_df["timestep"] = np.repeat(
                np.arange(num_timesteps), num_1d_nodes
            )
            dynamic_nodes_1d_df["node_idx"] = np.tile(
                np.arange(num_1d_nodes), num_timesteps
            )
            cols = ["timestep", "node_idx"] + self.DYNAMIC_1D_NODE_FEATURES
            dynamic_nodes_1d_df = dynamic_nodes_1d_df[cols]
            dynamic_nodes_1d_df.to_csv(
                os.path.join(event_dir, "1d_nodes_dynamic_all.csv"), index=False
            )

            # --- 1D Edges: All timesteps in one file ---
            dynamic_edges_1d_flat = dynamic_edges_1d.reshape(
                -1, len(self.DYNAMIC_1D_EDGE_FEATURES)
            )
            dynamic_edges_1d_df = pd.DataFrame(
                dynamic_edges_1d_flat, columns=self.DYNAMIC_1D_EDGE_FEATURES
            )
            dynamic_edges_1d_df["timestep"] = np.repeat(
                np.arange(num_timesteps), num_1d_edges
            )
            dynamic_edges_1d_df["edge_idx"] = np.tile(
                np.arange(num_1d_edges), num_timesteps
            )
            cols = ["timestep", "edge_idx"] + self.DYNAMIC_1D_EDGE_FEATURES
            dynamic_edges_1d_df = dynamic_edges_1d_df[cols]
            dynamic_edges_1d_df.to_csv(
                os.path.join(event_dir, "1d_edges_dynamic_all.csv"), index=False
            )

            self.log_func(
                f"  Saved event {run_id}: {num_timesteps} timesteps (all in single files)"
            )
        # --- Save Summary Statistics ---
        summary = {
            "dataset_mode": self.mode,
            "num_2d_nodes": static_nodes_2d.shape[0],
            "num_2d_edges": static_edges_2d.shape[0],
            "num_1d_nodes": static_nodes_1d.shape[0],
            "num_1d_edges": static_edges_1d.shape[0],
            "num_1d2d_connections": edge_index_1d_2d.shape[1],
            "num_events": len(self.hec_ras_run_ids),
            "total_rollout_timesteps": self.total_rollout_timesteps,
            "timestep_interval": self.timestep_interval,
            "previous_timesteps": self.previous_timesteps,
            "normalized": self.is_normalized,
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)

        self.log_func(f"✓ All features saved to {output_dir}")
