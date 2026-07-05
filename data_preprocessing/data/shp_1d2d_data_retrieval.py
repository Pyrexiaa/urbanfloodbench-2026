from typing import Literal
import numpy as np
import geopandas as gpd
from numpy import ndarray
from ..utils.file_utils import read_shp_file_as_numpy


def get_cell_position_x(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Extract X coordinates from Point geometry"""
    gdf = gpd.read_file(filepath)
    x_coords = gdf.geometry.x.to_numpy()
    return x_coords.astype(dtype)


def get_cell_position_y(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Extract Y coordinates from Point geometry"""
    gdf = gpd.read_file(filepath)
    y_coords = gdf.geometry.y.to_numpy()
    return y_coords.astype(dtype)


def get_cell_position(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Extract X,Y coordinates from Point geometry"""
    gdf = gpd.read_file(filepath)
    x_coords = gdf.geometry.x.to_numpy()
    y_coords = gdf.geometry.y.to_numpy()
    data = np.column_stack([x_coords, y_coords])
    return data.astype(dtype)

def get_relative_position(coord: Literal['x', 'y'], nodes_shp_path: str, edges_shp_path: str) -> ndarray:
    pos_retrieval_func = get_cell_position_x if coord == 'x' else get_cell_position_y
    position = pos_retrieval_func(nodes_shp_path)
    edge_index = get_edge_index(edges_shp_path)
    row, col = edge_index
    relative_pos = position[row] - position[col]
    return relative_pos

def get_edge_index(filepath: str) -> np.ndarray:
    """Get edge connectivity from shapefile"""
    columns = ["from_node", "to_node"]
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return data.astype(np.int64).transpose()

def get_cell_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get cell elevation - using Centre_ele (center elevation)"""
    columns = "Centre_ele"  # Changed from 'Elevation1'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_min_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get min elevation - using min_ele (minimum elevation)"""
    columns = "min_ele"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get edge length"""
    columns = "length"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)


def get_edge_slope(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get edge slope"""
    columns = "slope"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_direction_x(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Calculate edge x-direction (normal unit vector x-component) from edge geometry.
    
    Args:
        filepath: Path to EDGES shapefile (not HDF5)
    """
    gdf = gpd.read_file(filepath)
    
    directions_x = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            # Get vector along the edge
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            
            # Calculate normal vector (perpendicular, rotated 90 degrees)
            # Normal vector: (-dy, dx) then normalize
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                normal_x = -dy / length
            else:
                normal_x = 0.0
            
            directions_x.append(normal_x)
    
    return np.array(directions_x, dtype=dtype)


def get_edge_direction_y(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Calculate edge y-direction (normal unit vector y-component) from edge geometry.
    
    Args:
        filepath: Path to EDGES shapefile (not HDF5)
    """
    gdf = gpd.read_file(filepath)
    
    directions_y = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            # Get vector along the edge
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            
            # Calculate normal vector (perpendicular, rotated 90 degrees)
            # Normal vector: (-dy, dx) then normalize
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                normal_y = dx / length
            else:
                normal_y = 0.0
            
            directions_y.append(normal_y)
    
    return np.array(directions_y, dtype=dtype)


def get_face_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Calculate face/edge length from edge geometry.
    
    Args:
        filepath: Path to EDGES shapefile (not HDF5)
    """
    gdf = gpd.read_file(filepath)
    
    # Use the built-in length property
    lengths = gdf.geometry.length.to_numpy()
    
    return lengths.astype(dtype)

def get_1d_cell_depth(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get cell depth - using Depth (depth)"""
    columns = "Depth"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_invert_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get invert elevation - using InvertElev (invert elevation)"""
    columns = "InvertElev"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_surface_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get surface elevation - using TrrainElev (terain elevation)"""
    columns = "TrrainElev"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_base_area(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get base area - using BaseArea (base area)"""
    columns = "BaseArea"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_edge_index(filepath: str, nodes_filepath: str) -> np.ndarray:
    """
    Get 1D edge connectivity and map node names to integer indices.
    
    Args:
        filepath: Path to 1D links/edges shapefile
        nodes_filepath: Path to 1D nodes shapefile to create name->index mapping
    
    Returns:
        Edge index array of shape (2, num_edges) with integer node indices
    """
    # Read links
    links_gdf = gpd.read_file(filepath)
    us_nodes = links_gdf['USNode'].values
    ds_nodes = links_gdf['DSNode'].values
    
    # Read nodes to create mapping
    nodes_gdf = gpd.read_file(nodes_filepath)
    
    # Check if nodes have a 'Name' column or similar
    # Common column names: 'Name', 'ID', 'NodeName', 'Node_ID'
    node_name_col = None
    for col in ['Name', 'ID', 'NodeName', 'Node_ID', 'node_name']:
        if col in nodes_gdf.columns:
            node_name_col = col
            break
    
    if node_name_col is None:
        raise ValueError(
            f"Could not find node name column in {nodes_filepath}.\n"
            f"Available columns: {nodes_gdf.columns.tolist()}"
        )
    
    # Create mapping from node name to index
    node_names = nodes_gdf[node_name_col].values
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    
    # Map edge node names to indices
    edge_index = []
    missing_nodes = set()
    
    for us, ds in zip(us_nodes, ds_nodes):
        if us in name_to_idx and ds in name_to_idx:
            edge_index.append([name_to_idx[us], name_to_idx[ds]])
        else:
            if us not in name_to_idx:
                missing_nodes.add(us)
            if ds not in name_to_idx:
                missing_nodes.add(ds)
    
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} node names in links not found in nodes file:")
        print(f"  Missing nodes: {list(missing_nodes)[:10]}...")  # Show first 10
    
    if len(edge_index) == 0:
        raise ValueError("No valid edges found after mapping node names to indices!")
    
    return np.array(edge_index, dtype=np.int64).transpose()


def get_1d_edge_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get 1D link/conduit length"""
    columns = 'Length'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_edge_shape(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get 1D link/conduit shape"""
    SHAPE_TO_INT = {
        "circular": 0,
        "rectangular": 1,
        "box": 1, # Box is also rectangular
        "elliptical": 2,
        "arch": 3,
    }
    columns = 'Shape'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    data = data.squeeze()  # (N, 1) → (N,)

    encoded = np.array(
        [SHAPE_TO_INT[s.lower()] for s in data],
        dtype=dtype
    )
    return encoded

def get_1d_edge_diameter(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Get 1D link diameter/span.
    For circular pipes, this is the diameter.
    For other shapes, this might be the span/width.
    """
    columns = 'Span'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_edge_manning(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get 1D link Manning's n roughness coefficient"""
    columns = "Manning'sn"
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_edge_slope(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Get 1D link/conduit slope"""
    columns = 'Slope'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_1d_edge_relative_position(coord: Literal['x', 'y'], nodes_shp_path: str, edges_shp_path: str) -> ndarray:
    """
    Get relative position between connected nodes for each edge.
    
    Returns:
        Array of shape (num_edges,) with position[source] - position[target]
    """
    # Read nodes to get positions AND create consistent mapping
    nodes_gdf = gpd.read_file(nodes_shp_path)
    
    # Find node name column
    node_name_col = None
    for col in ['Name', 'ID', 'NodeName', 'Node_ID', 'node_name']:
        if col in nodes_gdf.columns:
            node_name_col = col
            break
    
    if node_name_col is None:
        raise ValueError(f"Could not find node name column in {nodes_shp_path}")
    
    # Get positions directly from the nodes_gdf
    if coord == 'x':
        # Assuming nodes have Point geometry
        positions = nodes_gdf.geometry.x.values
    else:
        positions = nodes_gdf.geometry.y.values
    
    # Create name-index mapping
    node_names = nodes_gdf[node_name_col].values
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    
    # Get edge connectivity
    links_gdf = gpd.read_file(edges_shp_path)
    us_nodes = links_gdf['USNode'].values
    ds_nodes = links_gdf['DSNode'].values
    
    # Calculate relative positions for each edge
    relative_positions = []
    
    for us, ds in zip(us_nodes, ds_nodes):
        if us in name_to_idx and ds in name_to_idx:
            us_idx = name_to_idx[us]
            ds_idx = name_to_idx[ds]
            rel_pos = positions[us_idx] - positions[ds_idx]
            relative_positions.append(rel_pos)
        else:
            # Handle missing nodes - maybe append NaN or skip
            print(f"Warning: Edge {us}->{ds} references missing node")
            relative_positions.append(np.nan)
    
    return np.array(relative_positions)

def get_1d2d_edge_index(filepath: str) -> np.ndarray:
    """
    Get 1D-2D connection edge index from shapefile.
    
    Returns edge index where:
    - First row contains 1D node indices
    - Second row contains 2D node indices
    
    Shape: (2, num_connections)
    """
    columns = ["node_1d", "node_2d"]
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return data.astype(np.int64).transpose()