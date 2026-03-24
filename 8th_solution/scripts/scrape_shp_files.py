import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import MultiLineString


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_endpoints(geom):
    """Return (x_start, y_start, x_end, y_end) for a line geometry."""
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan, np.nan
    coords = list(geom.geoms[0].coords if isinstance(geom, MultiLineString) else geom.coords)
    return coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]


# ---------------------------------------------------------------------------
# Node shapefiles
# ---------------------------------------------------------------------------

def process_node_shapefile(path: str):
    """
    Read one or more node shapefiles and return all their fields as DataFrames.

    Geometry columns added automatically:
      x, y  — centroid coordinates (from Point geometry, or centroid of polygon/line)

    Parameters
    ----------
    path : str path to a .shp file
            e.g. "/data/Nodes_2D.shp"

    Returns
    -------
    a DataFrame
    """
    
    print(f"[nodes] Reading {path}")
    gdf = gpd.read_file(path)
    print(f"        {len(gdf)} rows | CRS: {gdf.crs} | geom: {gdf.geom_type.unique()}")

    # Use centroid so x/y always exist regardless of geometry type
    centroids = gdf.geometry.centroid
    df = gdf.drop(columns="geometry").copy()
    df["x"] = centroids.x.values
    df["y"] = centroids.y.values

    print(f"        columns: {list(df.columns)}\n")

    return df


# ---------------------------------------------------------------------------
# Link / edge shapefiles
# ---------------------------------------------------------------------------

def process_link_shapefile(path: str):
    """
    Read one or more link/edge shapefiles and return all their fields as DataFrames.

    Geometry columns added automatically:
      length             — total length of the line in CRS units
      x_start, y_start  — coordinates of the first vertex
      x_end,   y_end    — coordinates of the last vertex

    Parameters
    ----------
    path : str path to a .shp file
            e.g. "/data/Node1D_to_Node2D_Links.shp"

    Returns
    -------
    a DataFrame
    """


    print(f"[links] Reading {path}")
    gdf = gpd.read_file(path)
    print(f"        {len(gdf)} rows | CRS: {gdf.crs} | geom: {gdf.geom_type.unique()}")

    df = gdf.drop(columns="geometry").copy()

    # Geometric derived columns
    df["length"] = gdf.geometry.length.values

    endpoints = gdf.geometry.apply(_line_endpoints)
    df["x_start"] = [ep[0] for ep in endpoints]
    df["y_start"] = [ep[1] for ep in endpoints]
    df["x_end"]   = [ep[2] for ep in endpoints]
    df["y_end"]   = [ep[3] for ep in endpoints]

    print(f"        columns: {list(df.columns)}\n")


    return df



# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

# All NodeType categories present across both models — fix the one-hot schema
# so both models always produce the same columns (absent categories filled with 0).
ALL_NODE_TYPES = ['Boundary', 'External', 'Junction', 'Start']
NODE_TYPE_COLS = [f'NodeType_{t}' for t in ALL_NODE_TYPES]

ROOT = str(Path(__file__).resolve().parent.parent)

for model in ['Model_1', 'Model_2']:
    model_path = f"{ROOT}/data/{model}"
    print(f"\n{'='*60}")
    print(f"Processing {model}")
    print('='*60)

    ######### NODES 1D #########
    nodes_1d = process_node_shapefile(f"{model_path}/shapefiles/Nodes_1D.shp")
    nodes_1d.rename(columns={'FID': 'node_idx'}, inplace=True)

    # One-hot NodeType with fixed schema across both models
    node_type_dummies = pd.get_dummies(nodes_1d['NodeType'], prefix='NodeType')
    for col in NODE_TYPE_COLS:
        if col not in node_type_dummies.columns:
            node_type_dummies[col] = 0
    nodes_1d = pd.concat([nodes_1d, node_type_dummies[NODE_TYPE_COLS]], axis=1)

    # Binary: has drop inlet
    nodes_1d['has_drop_inlet'] = nodes_1d['NodeStatus'].str.contains(
        'with drop inlet', case=False, na=False).astype(int)

    # Upstream / downstream connection counts
    nodes_1d[['ConnectUS', 'ConnectDS']] = (
        nodes_1d['ConnecUSDS'].str.split(':', expand=True).astype(int)
    )

    node_cols = ['node_idx'] + NODE_TYPE_COLS + ['has_drop_inlet', 'ConnectUS', 'ConnectDS']
    existing = pd.read_csv(f"{model_path}/train/1d_nodes_static.csv")
    merged = existing.merge(nodes_1d[node_cols], on='node_idx', how='left')
    out = f"{model_path}/train/1d_nodes_static_expanded.csv"
    merged.to_csv(out, index=False)
    print(f"Saved {out}  ({len(merged)} rows, {len(merged.columns)} cols)")

    ######### LINKS 1D #########
    links_1d = process_link_shapefile(f"{model_path}/shapefiles/Links_1D.shp")
    links_1d.rename(columns={'FID': 'edge_idx'}, inplace=True)

    edge_cols = ['edge_idx', 'USEnLoss', 'DSExLoss', 'USBFLoss', 'DSBFLoss']
    existing = pd.read_csv(f"{model_path}/train/1d_edges_static.csv")
    merged = existing.merge(links_1d[edge_cols], on='edge_idx', how='left')
    out = f"{model_path}/train/1d_edges_static_expanded.csv"
    merged.to_csv(out, index=False)
    print(f"Saved {out}  ({len(merged)} rows, {len(merged.columns)} cols)")

