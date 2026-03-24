#!/usr/bin/env python3
"""
Assemble the best submission: hz.parquet (Public LB = 0.0178)

Pipeline:
  (M1, NT1) + (M1, NT2):  boosting_m1_nt1_nt2.ipynb  (seed-blended foundation)
  (M2, NT2):               avg of boosting_m2_nt2_seed_{56..59}.ipynb  (4 seeds)
  (M2, NT1):               50% boosting avg (6 seeds) + 25% DL main + 25% DL optional avg(N2, N3, N9)

Required inputs:
  1) submission_v7_seedblend.parquet             ← boosting_m1_nt1_nt2.ipynb
  2) boosting_m2_nt1_seeds/seed_{56..61}.parquet ← boosting_m2_nt1_seed_*.ipynb  (6 files)
  3) boosting_m2_nt2_seeds/seed_{56..59}.parquet ← boosting_m2_nt2_seed_*.ipynb  (4 files)
  4) submission_m2_nt1_v2_dl_ensemble_top3.parquet  ← dl_m2_nt1_main_ensemble.ipynb
  5) submission_m2_nt1_N2_transformer_scse.parquet  ← dl_m2_nt1_n2_transformer_scse.ipynb
  6) submission_m2_nt1_N3_transformer_gcn.parquet   ← dl_m2_nt1_n3_transformer_gcn.ipynb
  7) submission_m2_nt1_N9_scse_gcn.parquet          ← dl_m2_nt1_n9_scse_gcn.ipynb

Usage:
    python make_submission_hz.py [--output hz.parquet]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def average_water_level(dfs):
    """Average water_level across DataFrames (positional alignment)."""
    wl = np.column_stack([df["water_level"].values for df in dfs])
    return wl.mean(axis=1)


def main():
    p = argparse.ArgumentParser(description="Assemble hz.parquet submission")
    p.add_argument("--output", default="hz.parquet")

    # --- foundation (M1 NT1 + M1 NT2 only) ---
    p.add_argument("--foundation", default="submission_v7_seedblend.parquet",
                   help="Base from boosting_m1_nt1_nt2.ipynb (used for M1 NT1 + M1 NT2 rows only)")

    # --- M2 NT2: boosting seeds ---
    p.add_argument("--boosting-m2nt2", nargs="+", default=[
        "boosting_m2_nt2_seeds/seed_56.parquet",
        "boosting_m2_nt2_seeds/seed_57.parquet",
        "boosting_m2_nt2_seeds/seed_58.parquet",
        "boosting_m2_nt2_seeds/seed_59.parquet",
    ], help="M2 NT2 boosting predictions (4 seeds)")

    # --- M2 NT1: boosting seeds ---
    p.add_argument("--boosting-m2nt1", nargs="+", default=[
        "boosting_m2_nt1_seeds/seed_56.parquet",
        "boosting_m2_nt1_seeds/seed_57.parquet",
        "boosting_m2_nt1_seeds/seed_58.parquet",
        "boosting_m2_nt1_seeds/seed_59.parquet",
        "boosting_m2_nt1_seeds/seed_60.parquet",
        "boosting_m2_nt1_seeds/seed_61.parquet",
    ], help="M2 NT1 boosting predictions (6 seeds)")

    # --- M2 NT1: DL ---
    p.add_argument("--dl-main", default="submission_m2_nt1_v2_dl_ensemble_top3.parquet",
                   help="DL main ensemble (N2+N5+N3)")
    p.add_argument("--dl-N2", default="submission_m2_nt1_N2_transformer_scse.parquet")
    p.add_argument("--dl-N3", default="submission_m2_nt1_N3_transformer_gcn.parquet")
    p.add_argument("--dl-N9", default="submission_m2_nt1_N9_scse_gcn.parquet")

    # --- M2 NT1 blend weights ---
    p.add_argument("--w-boosting", type=float, default=0.50, help="M2 NT1: boosting weight")
    p.add_argument("--w-dl-main", type=float, default=0.25, help="M2 NT1: DL main weight")
    p.add_argument("--w-dl-opt",  type=float, default=0.25, help="M2 NT1: DL optional avg weight")

    args = p.parse_args()

    w_total = args.w_boosting + args.w_dl_main + args.w_dl_opt
    assert abs(w_total - 1.0) < 1e-6, f"M2 NT1 weights must sum to 1.0, got {w_total}"

    # =========================================================================
    #  Load foundation (M1 NT1 + M1 NT2 + placeholder M2 rows)
    # =========================================================================
    print(f"Loading foundation: {args.foundation}")
    sub = load(args.foundation)
    print(f"  Total rows: {len(sub):,}")

    mask_m2_nt1 = (sub["model_id"] == 2) & (sub["node_type"] == 1)
    mask_m2_nt2 = (sub["model_id"] == 2) & (sub["node_type"] == 2)

    print(f"  M1 rows (keep):        {(sub['model_id'] == 1).sum():,}")
    print(f"  M2 NT1 rows (replace): {mask_m2_nt1.sum():,}")
    print(f"  M2 NT2 rows (replace): {mask_m2_nt2.sum():,}")

    # =========================================================================
    #  M2 NT2: average 4 boosting seeds
    # =========================================================================
    print(f"\n--- M2 NT2: averaging {len(args.boosting_m2nt2)} boosting seeds ---")
    m2nt2_dfs = []
    for path in args.boosting_m2nt2:
        df = load(path)
        print(f"  {path}: {len(df):,} rows")
        m2nt2_dfs.append(df)

    m2nt2_wl = average_water_level(m2nt2_dfs)
    sub.loc[mask_m2_nt2, "water_level"] = m2nt2_wl
    print(f"  M2 NT2 avg: [{m2nt2_wl.min():.4f}, {m2nt2_wl.max():.4f}]")

    # =========================================================================
    #  M2 NT1: blend boosting + DL main + DL optional
    # =========================================================================
    print(f"\n--- M2 NT1: boosting (6 seeds) ---")
    m2nt1_boost_dfs = []
    for path in args.boosting_m2nt1:
        df = load(path)
        print(f"  {path}: {len(df):,} rows")
        m2nt1_boost_dfs.append(df)
    boosting_wl = average_water_level(m2nt1_boost_dfs)

    print(f"\n--- M2 NT1: DL main ensemble ---")
    dl_main = load(args.dl_main)
    print(f"  {args.dl_main}: {len(dl_main):,} rows")

    print(f"\n--- M2 NT1: DL optional (N2, N3, N9) ---")
    dl_opt_dfs = []
    for label, path in [("N2", args.dl_N2), ("N3", args.dl_N3), ("N9", args.dl_N9)]:
        df = load(path)
        print(f"  {label}: {path} ({len(df):,} rows)")
        dl_opt_dfs.append(df)
    dl_opt_wl = average_water_level(dl_opt_dfs)

    blended_m2nt1 = (
        args.w_boosting * boosting_wl
        + args.w_dl_main * dl_main["water_level"].values
        + args.w_dl_opt  * dl_opt_wl
    )
    sub.loc[mask_m2_nt1, "water_level"] = blended_m2nt1
    print(f"\n  M2 NT1 blend: [{blended_m2nt1.min():.4f}, {blended_m2nt1.max():.4f}]")
    print(f"  Weights: boosting={args.w_boosting}, dl_main={args.w_dl_main}, dl_opt={args.w_dl_opt}")

    # =========================================================================
    #  Validate & save
    # =========================================================================
    nan_count = sub["water_level"].isna().sum()
    assert nan_count == 0, f"NaN in submission: {nan_count}"
    assert len(sub) > 0, "Empty submission!"

    out = Path(args.output)
    if out.suffix == ".parquet":
        sub.to_parquet(out, index=False)
    else:
        sub.to_csv(out, index=False)
    print(f"\nSaved: {out} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
