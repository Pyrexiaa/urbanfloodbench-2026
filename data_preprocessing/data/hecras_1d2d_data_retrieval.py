import numpy as np
import h5py

from datetime import datetime
from ..utils.file_utils import read_hdf_file_as_numpy

def explore_hdf5_structure(filepath: str, base_path: str = '', max_depth: int = 10, current_depth: int = 0):
    """
    Recursively explore and print HDF5 file structure.
    
    Args:
        filepath: Path to HDF5 file
        base_path: Starting path in HDF5 hierarchy
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth (internal use)
    """
    
    with h5py.File(filepath, 'r') as f:
        if base_path:
            # Navigate to base path
            parts = base_path.split('.')
            obj = f
            for part in parts:
                if part in obj:
                    obj = obj[part]
                else:
                    print(f"Path not found: {base_path}")
                    return
        else:
            obj = f
        
        def print_structure(name, obj, depth=0):
            indent = "  " * depth
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name}/ (Group)")
                if depth < max_depth:
                    for key in obj.keys():
                        print_structure(key, obj[key], depth + 1)
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {name} (Dataset) - Shape: {obj.shape}, Dtype: {obj.dtype}")
        
        print(f"\nStructure at: {base_path if base_path else 'ROOT'}")
        print("=" * 80)
        
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                print_structure(key, obj[key], current_depth)
        else:
            print(f"Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")

def get_event_timesteps(filepath: str) -> np.ndarray:
    property_path = 'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Time Date Stamp'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)

    def format(x: np.bytes_) -> datetime:
        TIMESTAMP_FORMAT = '%d%b%Y %H:%M:%S'
        time_str = x.decode('UTF-8')
        time_stamp = datetime.strptime(time_str, TIMESTAMP_FORMAT)
        return time_stamp

    vec_format = np.vectorize(format)
    time_series = vec_format(data)
    return time_series

def get_cell_area(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Surface Area'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_min_cell_elevation(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Minimum Elevation'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_roughness(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f"Geometry.2D Flow Areas.{perimeter_name}.Cells Center Manning's n"
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_cumulative_rainfall(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Cumulative Precipitation Depth'
        data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
        data = data.astype(dtype)
        area = get_cell_area(filepath, perimeter_name, dtype=dtype)
        data = (data / 1000) * area  # Convert mm to m³
    except KeyError:
        # Rainfall data not available in the file
        water_level = get_water_level(filepath, perimeter_name)
        data = np.zeros_like(water_level, dtype=dtype)

    return data

def get_rainfall(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Cumulative Precipitation Depth'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_water_level(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Water Surface'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_water_volume(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Volume'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_velocity(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Face Velocity'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_face_flow(filepath: str, perimeter_name: str = 'US Beaver', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Face Flow'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_wl_vol_interp_points_for_cell(cell_idx: int, filepath: str, perimeter_name: str = 'US Beaver') -> np.ndarray:
    info_property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Volume Elevation Info'
    info_data = read_hdf_file_as_numpy(filepath=filepath, property_path=info_property_path)

    values_property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Volume Elevation Values'
    values_data = read_hdf_file_as_numpy(filepath=filepath, property_path=values_property_path)

    assert cell_idx < info_data.shape[0], "Cell is not found in the data."
    start_idx, count = info_data[cell_idx]
    assert count > 1, "Cell has no elevation-volume points."
    link_data = values_data[start_idx:start_idx+count]
    water_level = link_data[:, 0]
    volume = link_data[:, 1]
    return water_level, volume

def get_1d_water_level(filepath: str, network_name: str = 'Base', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Pipe Networks.{network_name}.Nodes.Water Surface'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_1d_inlet_flow(filepath: str, network_name: str = 'Base', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Pipe Networks.{network_name}.Nodes.Drop Inlet Flow'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_1d_velocity(filepath: str, network_name: str = 'Base', dtype: np.dtype = np.float32) -> np.ndarray:
    base_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Pipe Networks.{network_name}.Pipes'
    
    path_us = f'{base_path}.Vel US'
    path_ds = f'{base_path}.Vel DS'
    
    vel_us = read_hdf_file_as_numpy(filepath=filepath, property_path=path_us).astype(dtype)
    vel_ds = read_hdf_file_as_numpy(filepath=filepath, property_path=path_ds).astype(dtype)
    
    # Calculate average velocity
    return (vel_us + vel_ds) / 2

def get_1d_flow(filepath: str, network_name: str = 'Base', dtype: np.dtype = np.float32) -> np.ndarray:
    base_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Pipe Networks.{network_name}.Pipes'
    
    path_us = f'{base_path}.Pipe Flow US'
    path_ds = f'{base_path}.Pipe Flow DS'
    
    flow_us = read_hdf_file_as_numpy(filepath=filepath, property_path=path_us).astype(dtype)
    flow_ds = read_hdf_file_as_numpy(filepath=filepath, property_path=path_ds).astype(dtype)
    
    # Calculate average flow
    return (flow_us + flow_ds) / 2