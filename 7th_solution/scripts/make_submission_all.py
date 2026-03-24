#!/usr/bin/env python3
"""
Assemble submission: all.parquet (Public LB = 0.0179)

Takes the best submission (hz.parquet) and further blends M2 NT1 with
additional DL optional predictions.

Components:
  (M1, NT1) + (M1, NT2) + (M2, NT2)  — unchanged from hz.parquet
  (M2, NT1)                           — 90% hz + 5% DL optional N5 + 5% DL optional GRU ensemble

Required inputs:
  1) hz.parquet                                    ← scripts/make_submission_hz.py
  2) submission_m2_nt1_N5_big_tf_w200.parquet      ← notebooks/dl_m2_nt1_n5_big_tf_w200.ipynb
  3) submission_m2_nt1_dl_ensemble_top3.parquet     ← notebooks/dl_m2_nt1_e678_bigru_transformer.ipynb
       (ensemble of E6_transformer + E7_bigru_attention + E8_bigru_gcn)

Usage:
    python make_submission_all.py [--output all.parquet]
"""

import argparse
import pandas as pd
from pathlib import Path


def load(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Assemble all.parquet submission")
    parser.add_argument("--output", default="all.parquet", help="Output path")

    parser.add_argument(
        "--base", default="hz.parquet",
        help="Best base submission (hz.parquet)"
    )
    parser.add_argument(
        "--dl-N5", default="submission_m2_nt1_N5_big_tf_w200.parquet",
        help="DL optional N5 model for M2 NT1"
    )
    parser.add_argument(
        "--dl-gru-ens", default="submission_m2_nt1_dl_ensemble_top3.parquet",
        help="DL optional GRU ensemble (E6+E7+E8) for M2 NT1"
    )
    # Weights for the M2 NT1 blend
    parser.add_argument("--w-base", type=float, default=0.90, help="Weight: base submission (hz)")
    parser.add_argument("--w-N5", type=float, default=0.05, help="Weight: DL N5")
    parser.add_argument("--w-gru", type=float, default=0.05, help="Weight: DL GRU ensemble")

    args = parser.parse_args()

    # --- Validate weights ---
    w_total = args.w_base + args.w_N5 + args.w_gru
    assert abs(w_total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {w_total}"

    # --- Load base ---
    print(f"Loading base: {args.base}")
    sub = load(args.base)
    print(f"  Total rows: {len(sub):,}")

    # --- Identify M2 NT1 rows ---
    mask = (sub["model_id"] == 2) & (sub["node_type"] == 1)
    n_m2nt1 = mask.sum()
    print(f"  M2 NT1 rows: {n_m2nt1:,}")

    base_wl = sub.loc[mask, "water_level"].values.copy()

    # --- Load DL N5 ---
    print(f"\nLoading DL N5: {args.dl_N5}")
    dl_n5 = load(args.dl_N5)
    print(f"  {len(dl_n5):,} rows")

    # --- Load DL GRU ensemble ---
    print(f"Loading DL GRU ensemble: {args.dl_gru_ens}")
    dl_gru = load(args.dl_gru_ens)
    print(f"  {len(dl_gru):,} rows")

    # --- Blend M2 NT1 ---
    blended_wl = (
        args.w_base * base_wl
        + args.w_N5 * dl_n5["water_level"].values
        + args.w_gru * dl_gru["water_level"].values
    )
    print(f"\nBlended M2 NT1: range=[{blended_wl.min():.4f}, {blended_wl.max():.4f}]")
    print(f"  Weights: base={args.w_base}, N5={args.w_N5}, gru={args.w_gru}")

    # --- Replace M2 NT1 ---
    sub.loc[mask, "water_level"] = blended_wl

    # --- Sanity checks ---
    assert sub["water_level"].isna().sum() == 0, "NaN values in submission!"
    assert len(sub) > 0, "Empty submission!"

    # --- Save ---
    out = Path(args.output)
    if out.suffix == ".parquet":
        sub.to_parquet(out, index=False)
    else:
        sub.to_csv(out, index=False)
    print(f"\nSaved: {out} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
