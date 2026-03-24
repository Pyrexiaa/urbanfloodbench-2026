#!/usr/bin/env python3
"""
Extract NLCD raster features at 2D node locations for Model_1 and Model_2.

Features extracted per node:
  - nlcd_land_cover     : NLCD land cover class (integer, 2024)
  - nlcd_fct_imp        : Fractional impervious surface % (0-99, 2024)
  - nlcd_tcc_2023       : Tree canopy cover % (2023)

Source: NLCD rasters downloaded from https://www.mrlc.gov/viewer/

Writes output CSVs alongside 2d_nodes_static.csv:
  data/Model_1/train/2d_nodes_raster_features.csv
  data/Model_2/train/2d_nodes_raster_features.csv

Usage:
  python3 src/extract_raster_features.py
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CONFIGS = {
    'Model_1': {
        'nodes_csv':   os.path.join(ROOT, 'data/Model_1/train/2d_nodes_static.csv'),
        'raster_dir':  os.path.join(ROOT, 'data/Model_1/Model1 Rasters'),
        'node_prj':    os.path.join(ROOT, 'data/Model_1/shapefiles/Nodes_2D.prj'),
        'output_csv':  os.path.join(ROOT, 'data/Model_1/train/2d_nodes_raster_features.csv'),
    },
    'Model_2': {
        'nodes_csv':   os.path.join(ROOT, 'data/Model_2/train/2d_nodes_static.csv'),
        'raster_dir':  os.path.join(ROOT, 'data/Model_2/Model2 Rasters'),
        'node_prj':    os.path.join(ROOT, 'data/Model_2/shapefiles/Nodes_2D.prj'),
        'output_csv':  os.path.join(ROOT, 'data/Model_2/train/2d_nodes_raster_features.csv'),
    },
}

# NLCD land cover class mapping (for reference)
NLCD_CLASSES = {
    11: 'open_water', 12: 'perennial_ice_snow',
    21: 'developed_open', 22: 'developed_low', 23: 'developed_med', 24: 'developed_high',
    31: 'barren', 41: 'deciduous_forest', 42: 'evergreen_forest', 43: 'mixed_forest',
    52: 'shrub', 71: 'grassland', 81: 'pasture', 82: 'cultivated_crops',
    90: 'woody_wetlands', 95: 'emergent_wetlands',
}


def read_prj_as_crs(prj_path):
    with open(prj_path) as f:
        return CRS.from_wkt(f.read())


def sample_raster_at_points(raster_path, xs, ys, nodata_fill=np.nan):
    """Sample raster values at (xs, ys) coordinates (already in raster CRS)."""
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        coords = list(zip(xs, ys))
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
        if nodata is not None:
            vals[vals == nodata] = nodata_fill
    return vals


def extract_for_model(model_name, cfg):
    print(f"\n{'='*60}")
    print(f"Extracting raster features for {model_name}")
    print(f"{'='*60}")

    nodes = pd.read_csv(cfg['nodes_csv'])
    print(f"  Nodes: {len(nodes)}")

    # Build transformer: node CRS → raster CRS
    node_crs = read_prj_as_crs(cfg['node_prj'])

    # Get raster CRS from any raster
    rasters_all = glob.glob(os.path.join(cfg['raster_dir'], '*.tiff'))
    assert rasters_all, f"No rasters found in {cfg['raster_dir']}"
    with rasterio.open(rasters_all[0]) as src:
        raster_crs = src.crs

    print(f"  Node CRS:   {node_crs.to_epsg() or node_crs.name}")
    print(f"  Raster CRS: {raster_crs.to_epsg() or 'AEA'}")

    transformer = Transformer.from_crs(node_crs, raster_crs, always_xy=True)
    xs_r, ys_r = transformer.transform(nodes['position_x'].values, nodes['position_y'].values)

    raster_dir = cfg['raster_dir']

    results = pd.DataFrame({'node_idx': nodes['node_idx'].values})

    # --- Land Cover (categorical integer) ---
    lndcov_path = glob.glob(os.path.join(raster_dir, '*LndCov*.tiff'))
    assert lndcov_path, f"LndCov raster not found in {raster_dir}"
    lndcov_path = lndcov_path[0]
    print(f"  Sampling LndCov: {os.path.basename(lndcov_path)}")
    lndcov = sample_raster_at_points(lndcov_path, xs_r, ys_r, nodata_fill=0)
    results['nlcd_land_cover'] = lndcov.astype(np.int16)

    # --- Fractional Impervious Surface (0-99) ---
    fctImp_path = glob.glob(os.path.join(raster_dir, '*FctImp*.tiff'))
    assert fctImp_path, f"FctImp raster not found in {raster_dir}"
    fctImp_path = fctImp_path[0]
    print(f"  Sampling FctImp: {os.path.basename(fctImp_path)}")
    fctImp = sample_raster_at_points(fctImp_path, xs_r, ys_r, nodata_fill=0)
    results['nlcd_fct_imp'] = fctImp.astype(np.float32)

    # --- Tree Canopy Cover (2023 only) ---
    tcc_paths = glob.glob(os.path.join(raster_dir, '*[Tt][Cc][Cc]*.tiff'))
    if not tcc_paths:
        print(f"  WARNING: TCC 2023 not found, filling with NaN")
        results['nlcd_tcc_2023'] = np.nan
    else:
        tcc_path = tcc_paths[0]
        print(f"  Sampling TCC 2023: {os.path.basename(tcc_path)}")
        tcc = sample_raster_at_points(tcc_path, xs_r, ys_r, nodata_fill=np.nan)
        results['nlcd_tcc_2023'] = tcc.astype(np.float32)

    # Summary stats
    print(f"\n  Summary:")
    print(f"    land_cover unique classes: {sorted(results['nlcd_land_cover'].unique().tolist())}")
    print(f"    fct_imp:  mean={results['nlcd_fct_imp'].mean():.1f}  max={results['nlcd_fct_imp'].max():.1f}")
    print(f"    tcc_2023: mean={results['nlcd_tcc_2023'].mean():.1f}  nan={results['nlcd_tcc_2023'].isna().sum()}")

    results.to_csv(cfg['output_csv'], index=False)
    print(f"\n  Saved: {cfg['output_csv']}")
    return results


if __name__ == '__main__':
    for model_name, cfg in MODEL_CONFIGS.items():
        extract_for_model(model_name, cfg)
    print("\nDone.")
