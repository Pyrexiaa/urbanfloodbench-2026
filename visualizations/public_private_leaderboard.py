import argparse
import os
import sys
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Standard deviations used for standardised RMSE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_STD_DEV = {
    (1, 1): 16.877747,
    (1, 2): 14.378797,
    (2, 1): 3.191784,
    (2, 2): 2.727131,
}


# ══════════════════════════════════════════════════════════════════════════════
# Core metric helpers
# ══════════════════════════════════════════════════════════════════════════════


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardised_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: float) -> float:
    if np.isnan(std) or std == 0:
        return np.nan
    return rmse(y_true, y_pred) / std


def _hierarchical_score(
    sub_df: pd.DataFrame,
    std_dev_dict: dict,
    node_type_filter=None,
    model_filter=None,
    standardise: bool = True,
) -> float:
    """
    Hierarchical average: node → node_type → event → model.

    Parameters
    ----------
    standardise : if True  → standardised RMSE (÷ std_dev)
                  if False → raw RMSE in original units
    """
    if sub_df.empty:
        return np.nan

    df = sub_df.copy()
    if model_filter is not None:
        df = df[df["model_id"] == model_filter]
    if df.empty:
        return np.nan

    node_types = [node_type_filter] if node_type_filter is not None else [1, 2]

    model_scores = []
    for model_id in sorted(df["model_id"].unique()):
        df_m = df[df["model_id"] == model_id]
        event_scores = []

        for event_id in sorted(df_m["event_id"].unique()):
            df_e = df_m[df_m["event_id"] == event_id]
            nt_scores = []

            for nt in node_types:
                df_t = df_e[df_e["node_type"] == nt]
                if df_t.empty:
                    continue

                std = std_dev_dict.get((model_id, nt), np.nan)
                node_scores = []

                for node_id in df_t["node_id"].unique():
                    nd = df_t[df_t["node_id"] == node_id]
                    if len(nd) <= 1:
                        continue
                    y_true = nd["target_water_level"].values
                    y_pred = nd["water_level"].values
                    score = (
                        standardised_rmse(y_true, y_pred, std)
                        if standardise
                        else rmse(y_true, y_pred)
                    )
                    if not np.isnan(score):
                        node_scores.append(score)

                if node_scores:
                    nt_scores.append(np.mean(node_scores))

            if nt_scores:
                event_scores.append(np.mean(nt_scores))

        if event_scores:
            model_scores.append(np.mean(event_scores))

    return float(np.mean(model_scores)) if model_scores else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Load & merge a single submission with the ground truth
# ══════════════════════════════════════════════════════════════════════════════


def load_submission(pred_path: str, gt_df: pd.DataFrame) -> pd.DataFrame:
    if pred_path.endswith(".parquet"):
        pred = pd.read_parquet(pred_path)
    else:
        pred = pd.read_csv(pred_path)

    # Build join keys present in both frames
    join_keys = ["row_id"]
    for col in ["model_id", "event_id", "node_type", "node_id"]:
        if col in pred.columns and col in gt_df.columns:
            join_keys.append(col)

    gt_cols = join_keys + [
        c for c in ["target_water_level", "Usage"] if c in gt_df.columns
    ]
    pred_cols = join_keys + ["water_level"]

    df = pd.merge(pred[pred_cols], gt_df[gt_cols], on=join_keys, how="inner")

    # Defaults for optional columns
    for col, default in [("event_id", 1), ("model_id", 1), ("Usage", "Public")]:
        if col not in df.columns:
            df[col] = default

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate one submission → dict of all metrics
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(df: pd.DataFrame, std_dev_dict: dict) -> dict:
    pub = df[df["Usage"].str.strip().str.lower() == "public"]
    priv = df[df["Usage"].str.strip().str.lower() == "private"]

    def s(sub, nt=None, m=None):
        return _hierarchical_score(
            sub, std_dev_dict, node_type_filter=nt, model_filter=m, standardise=True
        )

    def r(sub, nt=None, m=None):
        return _hierarchical_score(
            sub, std_dev_dict, node_type_filter=nt, model_filter=m, standardise=False
        )

    return {
        # Standardised RMSE — overall
        "pub_std_overall": s(pub),
        "priv_std_overall": s(priv),
        # Standardised RMSE — per model (private only)
        "priv_std_m1": s(priv, m=1),
        "priv_std_m2": s(priv, m=2),
        # Raw RMSE (ft) — private only, split by model × node_type
        "priv_raw_m1_1d": r(priv, nt=1, m=1),
        "priv_raw_m1_2d": r(priv, nt=2, m=1),
        "priv_raw_m2_1d": r(priv, nt=1, m=2),
        "priv_raw_m2_2d": r(priv, nt=2, m=2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Build the results DataFrame
# ══════════════════════════════════════════════════════════════════════════════


def build_table(names: list[str], metrics: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(metrics, index=names)
    df.index.name = "team"

    # Sort by private standardised RMSE (ascending = better)
    df.sort_values("priv_std_overall", inplace=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # Per-column sub-rankings (lower raw RMSE = better rank)
    for col in ["priv_raw_m1_1d", "priv_raw_m1_2d", "priv_raw_m2_1d", "priv_raw_m2_2d"]:
        df[f"rank_{col}"] = df[col].rank(method="min").astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

ORDINAL = {1: "1st", 2: "2nd", 3: "3rd"}


def _ord(n: int) -> str:
    return ORDINAL.get(n, f"{n}th")


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Flat CSV with all columns."""
    df.to_csv(path)
    print(f"  ✓ CSV saved  → {path}")


def save_latex(df: pd.DataFrame, path: str, unit: str = "ft") -> None:
    """
    LaTeX table layout:

      Col group A (no model split):
        Pub Std RMSE | Priv Std RMSE | Std RMSE M1 | Std RMSE M2

      Col group B (split by model × node_type):
        Model 1: 1D  Rk  2D  Rk
        Model 2: 1D  Rk  2D  Rk

    Pub/Priv/Overall Std RMSE share the same multi-row header block.
    Raw RMSE sub-columns carry the 3-level Model/1D-2D/Rank header.
    """
    lines = []
    a = lines.append

    # Column spec:
    #   rank + team + pub_std + priv_std + std_m1 + std_m2 + 4×(value rank)
    #   = 2 + 4 + 8 = 14 columns total
    a(r"\begin{table}[ht]")
    a(r"\centering")
    a(r"\small")
    a(r"\setlength{\tabcolsep}{4pt}")
    a(r"\resizebox{\linewidth}{!}{%")
    a(r"\begin{tabular}{cl cccc cccccccc}")
    a(r"\toprule")

    # ── Header row 1 ──────────────────────────────────────────────────────────
    # Columns 3-6: Std RMSE block (spans 4 cols with \multirow in rows 1-3)
    # Columns 7-14: Raw RMSE block
    a(
        r"\multicolumn{2}{c}{} & "
        r"\multicolumn{4}{c}{Standardised RMSE} & "
        r"\multicolumn{8}{c}{RMSE (" + unit + r")} \\"
    )
    a(r"\cmidrule(lr){3-6}\cmidrule(lr){7-14}")

    # ── Header row 2 ──────────────────────────────────────────────────────────
    a(
        r"\multicolumn{2}{c}{} & "
        r"Public & Private & Model 1 & Model 2 & "
        r"\multicolumn{4}{c}{Model 1} & "
        r"\multicolumn{4}{c}{Model 2} \\"
    )
    a(r"\cmidrule(lr){7-10}\cmidrule(lr){11-14}")

    # ── Header row 3 ──────────────────────────────────────────────────────────
    a(
        r"\multicolumn{1}{c}{\#} & Team & "
        r"& & & & "
        r"1D & Rk & 2D & Rk & "
        r"1D & Rk & 2D & Rk \\"
    )
    a(r"\midrule")

    for team, row in df.iterrows():
        rk = int(row["rank"])
        pub = f"{row['pub_std_overall']:.4f}"
        priv = f"\\textbf{{{row['priv_std_overall']:.4f}}}"
        std_m1 = f"{row['priv_std_m1']:.4f}"
        std_m2 = f"{row['priv_std_m2']:.4f}"
        m1_1d = f"{row['priv_raw_m1_1d']:.4f}"
        m1_1dr = _ord(int(row["rank_priv_raw_m1_1d"]))
        m1_2d = f"{row['priv_raw_m1_2d']:.4f}"
        m1_2dr = _ord(int(row["rank_priv_raw_m1_2d"]))
        m2_1d = f"{row['priv_raw_m2_1d']:.4f}"
        m2_1dr = _ord(int(row["rank_priv_raw_m2_1d"]))
        m2_2d = f"{row['priv_raw_m2_2d']:.4f}"
        m2_2dr = _ord(int(row["rank_priv_raw_m2_2d"]))

        row_str = (
            f"{rk} & {team} & "
            f"{pub} & {priv} & {std_m1} & {std_m2} & "
            f"{m1_1d} & {m1_1dr} & {m1_2d} & {m1_2dr} & "
            f"{m2_1d} & {m2_1dr} & {m2_2d} & {m2_2dr} \\\\"
        )
        a(row_str)

    a(r"\bottomrule")
    a(r"\end{tabular}%")
    a(r"}")
    a(
        r"\caption{Competition results. Teams ranked by private standardised RMSE "
        r"(lower is better). Model 1 and Model 2 standardised RMSE are overall "
        r"per-model scores. Rk = sub-metric rank within raw RMSE columns.}"
    )
    a(r"\label{tab:results}")
    a(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ LaTeX saved → {path}")


def save_text(df: pd.DataFrame, path: str, unit: str = "ft") -> None:
    """Fixed-width plain-text table for quick inspection."""
    sep = "─"
    W_TM = 20  # team name width
    W_SC = 12  # score width
    W_RK = 6  # rank width

    header1 = (
        f"{'':>4}  {'Team':<{W_TM}}  "
        f"{'Pub Std RMSE':>{W_SC}}  {'Priv Std RMSE':>{W_SC}}  "
        f"{'RMSE (' + unit + ')':^{4 * (W_SC + W_RK + 2)}}"
    )
    header2 = (
        f"{'':>4}  {'':^{W_TM}}  {'':>{W_SC}}  {'':>{W_SC}}  "
        f"{'Model 1':^{2 * (W_SC + W_RK + 2)}}  "
        f"{'Model 2':^{2 * (W_SC + W_RK + 2)}}"
    )
    header3 = (
        f"{'#':>4}  {'Team':<{W_TM}}  "
        f"{'Pub std':>{W_SC}}  {'Priv std':>{W_SC}}  "
        f"{'1D':>{W_SC}} {'Rk':>{W_RK}}  "
        f"{'2D':>{W_SC}} {'Rk':>{W_RK}}  "
        f"{'1D':>{W_SC}} {'Rk':>{W_RK}}  "
        f"{'2D':>{W_SC}} {'Rk':>{W_RK}}"
    )
    divider = sep * len(header3)

    rows = []
    for team, row in df.iterrows():
        rows.append(
            f"{int(row['rank']):>4}  {team:<{W_TM}}  "
            f"{row['pub_std_overall']:>{W_SC}.4f}  "
            f"{row['priv_std_overall']:>{W_SC}.4f}  "
            f"{row['priv_raw_m1_1d']:>{W_SC}.4f} {_ord(int(row['rank_priv_raw_m1_1d'])):>{W_RK}}  "
            f"{row['priv_raw_m1_2d']:>{W_SC}.4f} {_ord(int(row['rank_priv_raw_m1_2d'])):>{W_RK}}  "
            f"{row['priv_raw_m2_1d']:>{W_SC}.4f} {_ord(int(row['rank_priv_raw_m2_1d'])):>{W_RK}}  "
            f"{row['priv_raw_m2_2d']:>{W_SC}.4f} {_ord(int(row['rank_priv_raw_m2_2d'])):>{W_RK}}"
        )

    content = "\n".join(
        [divider, header1, header2, header3, divider] + rows + [divider]
    )
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  ✓ Text saved  → {path}")
    print()
    print(content)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate competition results table from submission files."
    )
    p.add_argument(
        "--submissions",
        nargs="+",
        required=True,
        help="Paths to submission files (.csv or .parquet), one per team.",
    )
    p.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Team names, in the same order as --submissions.",
    )
    p.add_argument(
        "--gt",
        required=True,
        help=(
            "Path to ground truth CSV/Parquet. "
            "Expected columns: row_id, model_id, event_id, node_type, node_id, "
            "water_level, Usage  (Usage = 'Public' | 'Private')."
        ),
    )
    p.add_argument(
        "--output",
        default="results_table",
        help="Output file stem (no extension). Default: results_table",
    )
    p.add_argument(
        "--unit",
        default="ft",
        help="Physical unit label for raw RMSE columns. Default: ft",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.submissions) != len(args.names):
        sys.exit(
            "ERROR: --submissions and --names must have the same number of entries."
        )

    # ── Load ground truth ────────────────────────────────────────────────────
    print(f"\nLoading ground truth: {args.gt}")
    gt_df = (
        pd.read_parquet(args.gt)
        if args.gt.endswith(".parquet")
        else pd.read_csv(args.gt)
    )

    # Normalise column name: accept either 'water_level' or 'target_water_level'
    if "water_level" in gt_df.columns and "target_water_level" not in gt_df.columns:
        gt_df.rename(columns={"water_level": "target_water_level"}, inplace=True)

    print(
        f"  Rows: {len(gt_df):,} | "
        f"Usage split: {gt_df['Usage'].value_counts().to_dict() if 'Usage' in gt_df.columns else 'N/A'}"
    )

    # ── Evaluate each submission ─────────────────────────────────────────────
    all_metrics = []
    for name, path in zip(args.names, args.submissions):
        print(f"\nEvaluating '{name}'  ({os.path.basename(path)})")
        df = load_submission(path, gt_df)
        mets = evaluate(df, DEFAULT_STD_DEV)
        all_metrics.append(mets)

        print(f"  Pub std RMSE : {mets['pub_std_overall']:.4f}")
        print(f"  Priv std RMSE: {mets['priv_std_overall']:.4f}")
        print(
            f"  M1-1D / M1-2D: {mets['priv_raw_m1_1d']:.4f} / {mets['priv_raw_m1_2d']:.4f} {args.unit}"
        )
        print(
            f"  M2-1D / M2-2D: {mets['priv_raw_m2_1d']:.4f} / {mets['priv_raw_m2_2d']:.4f} {args.unit}"
        )

    # ── Build & save table ───────────────────────────────────────────────────
    print("\nBuilding results table …")
    table = build_table(args.names, all_metrics)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_csv(table, f"{args.output}.csv")
    save_latex(table, f"{args.output}_latex.tex", unit=args.unit)
    save_text(table, f"{args.output}_text.txt", unit=args.unit)
    print("\nDone.")


if __name__ == "__main__":
    main()
