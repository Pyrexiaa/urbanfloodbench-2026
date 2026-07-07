"""
Produces:
  A. Score-distribution histograms (public, private, and side-by-side),
     binned into N intervals with per-interval team counts.
  B. Public-private "overfitting" analysis based on the percentage
     difference between each team's public and private score.

Scores are the competition SRMSE metric: LOWER IS BETTER.
The raw scores are extremely right-skewed (best ~0.011, worst ~16,600),
because many teams submitted non-competitive runs. Equal-width linear bins
would put almost every team in the first bar, so the DEFAULT distribution
plots use LOG-SPACED bins. A zoomed, linearly-binned view of the
"competitive" range is also produced.

Usage:
    python score_distribution_analysis.py
Optional CLI overrides:
    --public  PATH   public score csv   (default: ../<PUBLIC_CSV>)
    --private PATH   private score csv  (default: ../<PRIVATE_CSV>)
    --bins    N      number of intervals (default: 20)
    --competitive T  score cutoff for the zoomed view (default: 1.0)
    --outdir  DIR    where to write figures/tables (default: ./figures)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless / file output

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "mathtext.fontset": "stix",
    }
)

# ----------------------------------------------------------------------
# Config / paths
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PUBLIC = os.path.join(HERE, "..", "UrbanFloodBench_PublicScore.csv")
DEFAULT_PRIVATE = os.path.join(HERE, "..", "UrbanFloodBench_PrivateScore.csv")

# ---- blue theme (all figures) ----------------------------------------
PUB_COLOR = "#1F4E79"  # dark blue  (public)
PRV_COLOR = "#8FB8DE"  # light blue (private)
BAR_COLOR = "#2E75B6"  # medium blue for single-series bars/hists
BLUE_CMAP = "Blues"  # sequential blue colormap for the scatter
GRID_KW = dict(alpha=0.3, linewidth=0.6)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--public", default=DEFAULT_PUBLIC)
    p.add_argument("--private", default=DEFAULT_PRIVATE)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument(
        "--competitive",
        type=float,
        default=1.0,
        help="score <= this value is treated as the 'competitive' range",
    )
    p.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="drop teams whose PUBLIC score exceeds this value "
        "(e.g. 13 removes non-competitive/broken runs). "
        "Prints how many were removed.",
    )
    p.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    return p.parse_args()


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_scores(public_path, private_path):
    """Load both leaderboards and merge them on TeamId."""
    # utf-8-sig strips the BOM that Kaggle exports include
    pub = pd.read_csv(public_path, encoding="utf-8-sig")
    prv = pd.read_csv(private_path, encoding="utf-8-sig")

    for df in (pub, prv):
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

    pub = pub.dropna(subset=["Score"])
    prv = prv.dropna(subset=["Score"])

    merged = pub.merge(
        prv[["TeamId", "Score", "Rank"]],
        on="TeamId",
        suffixes=("_public", "_private"),
    )
    # percentage difference: positive => private WORSE than public (shakedown / overfit)
    merged["pct_diff"] = (
        (merged["Score_private"] - merged["Score_public"])
        / merged["Score_public"]
        * 100.0
    )
    merged["abs_pct_diff"] = merged["pct_diff"].abs()
    return pub, prv, merged


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_log_bins(values_list, n_bins):
    """Common log-spaced bin edges spanning all supplied value arrays."""
    allv = np.concatenate([np.asarray(v, float) for v in values_list])
    allv = allv[allv > 0]
    lo, hi = allv.min(), allv.max()
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def make_linear_bins(values_list, n_bins):
    allv = np.concatenate([np.asarray(v, float) for v in values_list])
    return np.linspace(allv.min(), allv.max(), n_bins + 1)


def counts_table(scores, edges):
    """Return a dataframe of interval -> team count."""
    counts, _ = np.histogram(scores, bins=edges)
    rows = []
    for i in range(len(edges) - 1):
        rows.append(
            {
                "interval": f"[{edges[i]:.4g}, {edges[i + 1]:.4g})",
                "bin_low": edges[i],
                "bin_high": edges[i + 1],
                "n_teams": int(counts[i]),
            }
        )
    return pd.DataFrame(rows)


def annotate_bars(ax, patches, counts):
    for patch, c in zip(patches, counts):
        if c > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=11,
            )


# ----------------------------------------------------------------------
# A. Score-distribution figures
# ----------------------------------------------------------------------
def plot_single_histogram(scores, edges, title, color, out_png, logx=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts, _, patches = ax.hist(
        scores, bins=edges, color=color, edgecolor="white", linewidth=0.6
    )
    annotate_bars(ax, patches, counts)
    if logx:
        ax.set_xscale("log")
    else:
        # leave a little breathing room before/after the limits so the
        # first/last bars aren't flush against the axis
        span = edges[-1] - edges[0]
        pad = 0.03 * span
        ax.set_xlim(edges[0] - pad, edges[-1] + pad)
    ax.set_xlabel("SRMSE score")
    ax.set_ylabel("Number of teams")
    ax.set_title(title)
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return counts


def plot_side_by_side(pub_scores, prv_scores, edges, out_png, logx=True):
    """Grouped bars: public vs private team-counts per shared interval."""
    pub_counts, _ = np.histogram(pub_scores, bins=edges)
    prv_counts, _ = np.histogram(prv_scores, bins=edges)
    centers = (
        np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
    )
    # bar width in log space
    fig, ax = plt.subplots(figsize=(11, 5.5))
    idx = np.arange(len(centers))
    w = 0.4
    ax.bar(idx - w / 2, pub_counts, width=w, color=PUB_COLOR, label="Public")
    ax.bar(idx + w / 2, prv_counts, width=w, color=PRV_COLOR, label="Private")
    for i, (pc, qc) in enumerate(zip(pub_counts, prv_counts)):
        if pc:
            ax.text(i - w / 2, pc, str(pc), ha="center", va="bottom", fontsize=9)
        if qc:
            ax.text(i + w / 2, qc, str(qc), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{e:.3g}" for e in edges[:-1]], rotation=60, ha="right", fontsize=11
    )
    ax.set_xlabel("SRMSE Score")
    ax.set_ylabel("Number of teams")
    ax.set_title(f"Public vs Private Score Distribution ({len(pub_scores)} teams)")
    ax.legend()
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return pub_counts, prv_counts


def plot_competitive_zoom(pub, prv, cutoff, n_bins, out_png):
    """Linear-binned histogram of the competitive range (score <= cutoff)."""
    p = pub
    q = prv
    edges = np.linspace(0, cutoff, n_bins + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(
        p,
        bins=edges,
        alpha=0.6,
        color=PUB_COLOR,
        edgecolor="white",
        label=f"Public (n={len(p)})",
    )
    ax.hist(
        q,
        bins=edges,
        alpha=0.6,
        color=PRV_COLOR,
        edgecolor="white",
        label=f"Private (n={len(q)})",
    )
    ax.set_xlabel(f"SRMSE score (<= {cutoff:g})")
    ax.set_ylabel("Number of teams")
    ax.set_title("Competitive-range score distribution")
    ax.legend()
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# B. Public-private overfitting analysis
# ----------------------------------------------------------------------
def plot_scatter(merged, out_png, logscale=True):
    """Public vs private scatter with a y=x reference line.

    logscale=True  -> log-log axes (needed when scores span many orders of
                      magnitude, i.e. unfiltered data).
    logscale=False -> raw linear axes (clean when scores are capped, e.g. <=1).
    """
    fig, ax = plt.subplots(figsize=(7.5, 7))
    sc = ax.scatter(
        merged["Score_public"],
        merged["Score_private"],
        c=np.clip(merged["abs_pct_diff"], 0, 50),
        cmap=BLUE_CMAP,
        s=32,
        edgecolor=PUB_COLOR,
        linewidth=0.4,
        alpha=0.9,
    )
    if logscale:
        lims = [
            min(merged["Score_public"].min(), merged["Score_private"].min()),
            max(merged["Score_public"].max(), merged["Score_private"].max()),
        ]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Public SRMSE (log)")
        ax.set_ylabel("Private SRMSE (log)")
    else:
        hi = max(merged["Score_public"].max(), merged["Score_private"].max())
        lims = [0, hi]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Public SRMSE")
        ax.set_ylabel("Private SRMSE")
    ax.plot(lims, lims, "k--", linewidth=1, label="Identical Score")
    ax.set_title("Public vs Private Score Difference Distribution (185 teams)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Percentage Difference")

    # legend: just the reference line (upper-left, where the diagonal leaves room)
    ax.legend(loc="upper left", framealpha=0.9)

    # summary stats in a separate, cleanly left-aligned text box in the
    # empty lower-right corner (avoids the misaligned blank-handle look)
    mean_d = merged["pct_diff"].mean()
    med_d = merged["pct_diff"].median()
    stats = f"mean = {mean_d:+.2f}%\nmedian = {med_d:+.2f}%"
    ax.text(
        0.97,
        0.03,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        multialignment="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.grid(True, which="both", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pct_diff_hist(merged, out_png, xclip=100, n_bins=40):
    """Distribution of signed % difference (private vs public)."""
    d = merged["pct_diff"].clip(-xclip, xclip)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(d, bins=n_bins, color=BAR_COLOR, edgecolor="white")
    ax.axvline(
        0,
        color="k",
        linestyle="--",
        linewidth=1,
        label="0% (identical public & private)",
    )
    med = merged["pct_diff"].median()
    ax.axvline(
        np.clip(med, -xclip, xclip),
        color=PUB_COLOR,
        linewidth=2,
        label=f"median = {med:+.1f}%",
    )
    ax.set_xlabel(
        f"% difference of private vs public score  (clipped to +/-{xclip}%)\n"
        "positive = private worse than public"
    )
    ax.set_ylabel("Number of teams")
    ax.set_title("Public-private score difference distribution")
    ax.legend()
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_overfit_bands(merged, out_png, bands=(5, 10, 25, 50, 100)):
    """How many teams fall within |% diff| bands => how much overfitting."""
    absd = merged["abs_pct_diff"]
    labels, counts = [], []
    prev = 0
    for i, b in enumerate(bands):
        labels.append(f"{prev}-{b}%")
        # first band is inclusive of its lower edge so teams with an exact
        # 0% shift (identical public & private score) are counted; later
        # bands use a strict lower bound to avoid double-counting edges.
        lo = (absd >= prev) if i == 0 else (absd > prev)
        counts.append(int((lo & (absd <= b)).sum()))
        prev = b
    labels.append(f">{bands[-1]}%")
    counts.append(int((absd > bands[-1]).sum()))
    assert sum(counts) == len(merged), (sum(counts), len(merged))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, counts, color=BAR_COLOR, edgecolor="white")
    total = len(merged)
    # extra headroom so the two-line label above the tallest bar isn't clipped
    ax.set_ylim(0, max(counts) * 1.18)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{c}\n({100 * c / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_xlabel("Percentage Difference between Public & Private Score")
    ax.set_ylabel("Number of teams")
    ax.set_title("Degree of Public-Private Score Shift (185 teams)")
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return dict(zip(labels, counts))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(f"{args.outdir}_{args.max_score}", exist_ok=True)
    modified_output_dir = f"{args.outdir}_{args.max_score}"

    pub, prv, merged = load_scores(args.public, args.private)
    n_total = len(merged)

    print(
        f"Loaded {len(pub)} public / {len(prv)} private rows; "
        f"{n_total} teams matched on TeamId."
    )

    # ---- optional filter of non-competitive / broken runs ----------------
    # Filter on TeamId (public score threshold) so both boards stay aligned.
    if args.max_score is not None:
        # keep a team only if BOTH its public AND private score are <= max_score
        keep_ids = merged.loc[
            (merged["Score_public"] <= args.max_score)
            & (merged["Score_private"] <= args.max_score),
            "TeamId",
        ]
        n_removed = n_total - len(keep_ids)
        pub = pub[pub["TeamId"].isin(keep_ids)]
        prv = prv[prv["TeamId"].isin(keep_ids)]
        merged = merged[merged["TeamId"].isin(keep_ids)]
        print(
            f"\nFilter applied: dropped {n_removed} teams whose public OR "
            f"private SRMSE > {args.max_score:g} ({100 * n_removed / n_total:.1f}% "
            f"of {n_total}); {len(merged)} competitive teams retained."
        )

    pub_scores = pub["Score"].values
    prv_scores = prv["Score"].values

    # After filtering the range is small enough for LINEAR bins (directly
    # comparable, no log axis needed); unfiltered data stays on log bins.
    filtered = args.max_score is not None
    logx = not filtered
    if filtered:
        # equal-width intervals spanning exactly 0 .. max_score, so the axis
        # ends at the cap and every bin is the same width (systematic).
        edges = np.linspace(0.0, args.max_score, args.bins + 1)
    else:
        edges = make_log_bins([pub_scores, prv_scores], args.bins)

    # ---- A. distributions ----
    plot_single_histogram(
        pub_scores,
        edges,
        f"Public score distribution ({len(pub)} teams)",
        PUB_COLOR,
        os.path.join(modified_output_dir, "01_public_histogram.png"),
        logx=logx,
    )

    plot_single_histogram(
        prv_scores,
        edges,
        f"Private score distribution ({len(prv)} teams)",
        PRV_COLOR,
        os.path.join(modified_output_dir, "02_private_histogram.png"),
        logx=logx,
    )

    plot_side_by_side(
        pub_scores,
        prv_scores,
        edges,
        os.path.join(modified_output_dir, "03_public_vs_private_sidebyside.png"),
        logx=logx,
    )
    log_edges = edges  # for interval count table below

    plot_competitive_zoom(
        pub_scores,
        prv_scores,
        args.competitive,
        args.bins,
        os.path.join(modified_output_dir, "04_competitive_zoom.png"),
    )

    # interval count tables (the "20 intervals -> how many teams" request)
    tbl = counts_table(pub_scores, log_edges).rename(
        columns={"n_teams": "n_teams_public"}
    )
    tbl["n_teams_private"] = counts_table(prv_scores, log_edges)["n_teams"]
    tbl.to_csv(os.path.join(modified_output_dir, "interval_counts.csv"), index=False)

    # ---- B. overfitting ----
    plot_scatter(
        merged,
        os.path.join(modified_output_dir, "05_public_private_scatter.png"),
        logscale=logx,
    )
    plot_pct_diff_hist(
        merged, os.path.join(modified_output_dir, "06_pct_diff_histogram.png")
    )
    band_counts = plot_overfit_bands(
        merged, os.path.join(modified_output_dir, "07_overfitting_bands.png")
    )

    # per-team overfitting table, sorted by public rank
    cols = [
        "Rank_public",
        "Rank_private",
        "TeamName",
        "SubmissionCount",
        "Score_public",
        "Score_private",
        "pct_diff",
        "abs_pct_diff",
    ]
    cols = [c for c in cols if c in merged.columns]
    merged.sort_values("Score_public")[cols].to_csv(
        os.path.join(modified_output_dir, "per_team_overfitting.csv"), index=False
    )

    # ---- console summary ----
    print("\n=== Score distribution (public, log bins) ===")
    print(tbl.to_string(index=False))

    print("\n=== Public-private difference summary ===")
    print(f"median signed %diff : {merged['pct_diff'].median():+.2f}%")
    print(f"mean   signed %diff : {merged['pct_diff'].mean():+.2f}%")
    print(
        f"teams private WORSE than public : "
        f"{(merged['pct_diff'] > 0).sum()} / {len(merged)}"
    )
    print(
        f"teams private BETTER than public: "
        f"{(merged['pct_diff'] < 0).sum()} / {len(merged)}"
    )
    print("\n=== |%diff| bands (stability) ===")
    for k, v in band_counts.items():
        print(f"  {k:>8}: {v:3d} teams ({100 * v / len(merged):.1f}%)")

    print(f"\nFigures + tables written to: {modified_output_dir}")


if __name__ == "__main__":
    main()
