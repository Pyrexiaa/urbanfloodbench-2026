from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from itertools import cycle
import matplotlib.patches as mpatches

DEFAULT_STD_DEV = {
    (1, 1): 16.877747,
    (1, 2): 14.378797,
    (2, 1): 3.191784,
    (2, 2): 2.727131,
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardised_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: float) -> float:
    if np.isnan(std) or std == 0:
        return np.nan
    return rmse(y_true, y_pred) / std


def plot_multi_csv_comparison_model1(
    csv_paths,
    csv_labels=None,
    csv_colors=None,
    csv_linestyles=None,
    csv_markers=None,
    output_dir=None,
):
    """
    Compare predictions from any number of CSV/Parquet files against the
    reference HEC-RAS simulation, one figure per node (Model 1).

    Style lists (labels/colors/linestyles/markers) default to sensible values
    and are cycle-extended to match the number of inputs.
    """
    n = len(csv_paths)
    if n == 0:
        raise ValueError("At least one CSV path is required.")

    if csv_labels is None:
        csv_labels = [f"Result {i + 1}" for i in range(n)]

    _default_colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "teal",
        "magenta",
        "cyan",
        "olive",
        "navy",
    ]
    if csv_colors is None:
        csv_colors = [c for c, _ in zip(cycle(_default_colors), range(n))]

    _default_ls = ["-", "--", "-.", ":"]
    if csv_linestyles is None:
        csv_linestyles = [ls for ls, _ in zip(cycle(_default_ls), range(n))]

    _default_markers = ["x", "^", "s", "D", "v", "o", "P", "*", "h", ">"]
    if csv_markers is None:
        csv_markers = [m for m, _ in zip(cycle(_default_markers), range(n))]

    # Cycle-extend any style list shorter than n so no series is dropped.
    def _fit(seq):
        return [x for x, _ in zip(cycle(seq), range(n))]

    csv_labels = _fit(csv_labels)
    csv_colors = _fit(csv_colors)
    csv_linestyles = _fit(csv_linestyles)
    csv_markers = _fit(csv_markers)

    model_id = 1
    event_id = 8

    # Define the 3 nodes to plot
    nodes_to_plot = [
        {"node_type": 1, "node_id": 9, "title": "1D Node ID 9"},
        {"node_type": 1, "node_id": 12, "title": "1D Node ID 12"},
        {"node_type": 1, "node_id": 0, "title": "1D Node ID 0"},
        {"node_type": 2, "node_id": 2854, "title": "2D Node ID 2854"},
        {"node_type": 2, "node_id": 2960, "title": "2D Node ID 2960"},
        {"node_type": 2, "node_id": 3487, "title": "2D Node ID 3487"},
    ]

    # Output directory (based on first CSV path)
    csv_path_obj = Path(csv_paths[0])
    if output_dir is None:
        output_dir = csv_path_obj.parent / "multi_csv_comparison_test"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = csv_path_obj.parent / f"{output_dir}"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Multi-CSV Comparison for Model {model_id}, Event {event_id}")
    print(f"{'=' * 60}")

    # Load all CSV files
    dfs = []
    for i, csv_path in enumerate(csv_paths):
        if csv_path.endswith(".parquet"):
            df = pd.read_parquet(csv_path)
        else:
            df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"  Loaded {csv_labels[i]}: {len(df)} samples from {csv_path}")

    # ========================
    # Create a figure for each node
    # ========================
    for node_config in nodes_to_plot:
        node_type = node_config["node_type"]
        node_id = node_config["node_id"]
        node_title = node_config["title"]

        print(f"\n{'-' * 60}")
        print(f"Plotting {node_title} (Type {node_type}, ID {node_id})")
        print(f"{'-' * 60}")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        ground_truth_plotted = False

        # Plot data from each CSV
        for i, (df, label, color, linestyle, marker) in enumerate(
            zip(dfs, csv_labels, csv_colors, csv_linestyles, csv_markers)
        ):
            # Get data for this specific node
            node_data = df[
                (df["node_id"] == node_id)
                & (df["model_id"] == model_id)
                & (df["event_id"] == event_id)
                & (df["node_type"] == node_type)
            ]

            # Handle missing timestep column
            if "timestep" in df.columns:
                node_data = node_data.sort_values("timestep")
                timesteps = node_data["timestep"].values
            else:
                # Preserve original row order and generate synthetic timesteps
                node_data = node_data.copy()
                node_data = node_data.reset_index(drop=True)
                timesteps = np.arange(len(node_data))

            if len(node_data) > 0:
                # Plot ground truth only once (from first CSV)
                if not ground_truth_plotted:
                    ax.plot(
                        timesteps,
                        node_data["target_water_level"],
                        label="Reference HEC-RAS Simulation",
                        color="blue",
                        marker="o",
                        linewidth=2.5,
                        markersize=5,
                        alpha=0.9,
                        zorder=10,  # Draw on top
                    )
                    ground_truth_plotted = True

                # Plot predictions from this CSV
                ax.plot(
                    timesteps,
                    node_data["water_level"],
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    alpha=0.8,
                    linestyle=linestyle,
                )

                # Calculate metrics for this CSV
                rmse = np.sqrt(
                    np.mean(
                        (node_data["target_water_level"] - node_data["water_level"])
                        ** 2
                    )
                )
                mae = np.mean(
                    np.abs(node_data["target_water_level"] - node_data["water_level"])
                )
                r2 = r2_score(node_data["target_water_level"], node_data["water_level"])

                print(
                    f"  ✓ {label}: {len(node_data)} timesteps | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
                )
            else:
                print(f"  ✗ {label}: No data found")

        # Set title and labels
        ax.set_title(
            f"Model {model_id} Event {event_id} {node_title}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        xlabel = "Timestep" if "timestep" in dfs[0].columns else "Timesteps"
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Water Level (ft)", fontsize=13)

        # Legend with better positioning
        ax.legend(
            loc="best",
            fontsize=13,
            framealpha=0.9,
            edgecolor="black",
        )

        ax.grid(True, alpha=0.3, linestyle="--")

        output_filename = f"comparison_model{model_id}_event{event_id}_type{node_type}_node{node_id}.png"
        output_path = output_dir / output_filename

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    print(f"\n{'=' * 60}")
    print("All comparison plots completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def plot_multi_csv_comparison_model2(
    csv_paths,
    csv_labels=None,
    csv_colors=None,
    csv_linestyles=None,
    csv_markers=None,
    output_dir=None,
):
    """
    Compare predictions from any number of CSV/Parquet files against the
    reference HEC-RAS simulation, one figure per node (Model 2).

    Style lists (labels/colors/linestyles/markers) default to sensible values
    and are cycle-extended to match the number of inputs.
    """
    n = len(csv_paths)
    if n == 0:
        raise ValueError("At least one CSV path is required.")

    if csv_labels is None:
        csv_labels = [f"Result {i + 1}" for i in range(n)]

    _default_colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "teal",
        "magenta",
        "cyan",
        "olive",
        "navy",
    ]
    if csv_colors is None:
        csv_colors = [c for c, _ in zip(cycle(_default_colors), range(n))]

    _default_ls = ["-", "--", "-.", ":"]
    if csv_linestyles is None:
        csv_linestyles = [ls for ls, _ in zip(cycle(_default_ls), range(n))]

    _default_markers = ["x", "^", "s", "D", "v", "o", "P", "*", "h", ">"]
    if csv_markers is None:
        csv_markers = [m for m, _ in zip(cycle(_default_markers), range(n))]

    # Cycle-extend any style list shorter than n so no series is dropped.
    def _fit(seq):
        return [x for x, _ in zip(cycle(seq), range(n))]

    csv_labels = _fit(csv_labels)
    csv_colors = _fit(csv_colors)
    csv_linestyles = _fit(csv_linestyles)
    csv_markers = _fit(csv_markers)

    model_id = 2
    event_id = 8

    # Define the 3 nodes to plot
    nodes_to_plot = [
        {"node_type": 1, "node_id": 7, "title": "1D Node ID 7"},
        {"node_type": 1, "node_id": 1, "title": "1D Node ID 1"},
        {"node_type": 1, "node_id": 0, "title": "1D Node ID 0"},
        {"node_type": 2, "node_id": 2854, "title": "2D Node ID 2854"},
        {"node_type": 2, "node_id": 2960, "title": "2D Node ID 2960"},
        {"node_type": 2, "node_id": 2329, "title": "2D Node ID 2329"},
    ]

    # Output directory (based on first CSV path)
    csv_path_obj = Path(csv_paths[0])
    if output_dir is None:
        output_dir = csv_path_obj.parent / "multi_csv_comparison_test"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = csv_path_obj.parent / f"{output_dir}"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Multi-CSV Comparison for Model {model_id}, Event {event_id}")
    print(f"{'=' * 60}")

    # Load all CSV files
    dfs = []
    for i, csv_path in enumerate(csv_paths):
        if csv_path.endswith(".parquet"):
            df = pd.read_parquet(csv_path)
        else:
            df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"  Loaded {csv_labels[i]}: {len(df)} samples from {csv_path}")

    # ========================
    # Create a figure for each node
    # ========================
    for node_config in nodes_to_plot:
        node_type = node_config["node_type"]
        node_id = node_config["node_id"]
        node_title = node_config["title"]

        print(f"\n{'-' * 60}")
        print(f"Plotting {node_title} (Type {node_type}, ID {node_id})")
        print(f"{'-' * 60}")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        ground_truth_plotted = False

        # Plot data from each CSV
        for i, (df, label, color, linestyle, marker) in enumerate(
            zip(dfs, csv_labels, csv_colors, csv_linestyles, csv_markers)
        ):
            # Get data for this specific node
            node_data = df[
                (df["node_id"] == node_id)
                & (df["model_id"] == model_id)
                & (df["event_id"] == event_id)
                & (df["node_type"] == node_type)
            ]

            # Handle missing timestep column
            if "timestep" in df.columns:
                node_data = node_data.sort_values("timestep")
                timesteps = node_data["timestep"].values
            else:
                # Preserve original row order and generate synthetic timesteps
                node_data = node_data.copy()
                node_data = node_data.reset_index(drop=True)
                timesteps = np.arange(len(node_data))

            if len(node_data) > 0:
                # Plot ground truth only once (from first CSV)
                if not ground_truth_plotted:
                    ax.plot(
                        timesteps,
                        node_data["target_water_level"],
                        label="Reference HEC-RAS Simulation",
                        color="blue",
                        marker="o",
                        linewidth=2.5,
                        markersize=5,
                        alpha=0.9,
                        zorder=10,  # Draw on top
                    )
                    ground_truth_plotted = True

                # Plot predictions from this CSV
                ax.plot(
                    timesteps,
                    node_data["water_level"],
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    alpha=0.8,
                    linestyle=linestyle,
                )

                # Calculate metrics for this CSV
                rmse = np.sqrt(
                    np.mean(
                        (node_data["target_water_level"] - node_data["water_level"])
                        ** 2
                    )
                )
                mae = np.mean(
                    np.abs(node_data["target_water_level"] - node_data["water_level"])
                )
                r2 = r2_score(node_data["target_water_level"], node_data["water_level"])

                print(
                    f"  ✓ {label}: {len(node_data)} timesteps | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
                )
            else:
                print(f"  ✗ {label}: No data found")

        # Set title and labels
        ax.set_title(
            f"Model {model_id} Event {event_id} {node_title}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        xlabel = "Timestep" if "timestep" in dfs[0].columns else "Timesteps"
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Water Level (ft)", fontsize=13)

        # Legend with better positioning
        ax.legend(
            loc="best",
            fontsize=13,
            framealpha=0.9,
            edgecolor="black",
        )

        ax.grid(True, alpha=0.3, linestyle="--")

        output_filename = f"comparison_model{model_id}_event{event_id}_type{node_type}_node{node_id}.png"
        output_path = output_dir / output_filename

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    print(f"\n{'=' * 60}")
    print("All comparison plots completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def plot_avg_rmse_across_nodes_v2(
    model_id,
    csv_paths,
    csv_labels=None,
    csv_colors=None,
    csv_linestyles=None,
    csv_markers=None,
    output_dir=None,
    exceedance_csv=None,  # NEW: path to the exceedance probability CSV
):
    n = len(csv_paths)
    if n == 0:
        raise ValueError("At least one CSV path is required.")

    if csv_labels is None:
        csv_labels = [f"Result {i + 1}" for i in range(n)]

    _default_colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "teal",
        "magenta",
        "cyan",
        "olive",
        "navy",
    ]
    if csv_colors is None:
        csv_colors = [c for c, _ in zip(cycle(_default_colors), range(n))]

    _default_ls = ["-", "--", "-.", ":"]
    if csv_linestyles is None:
        csv_linestyles = [ls for ls, _ in zip(cycle(_default_ls), range(n))]

    _default_markers = ["x", "^", "s", "D", "v", "o", "P", "*", "h", ">"]
    if csv_markers is None:
        csv_markers = [m for m, _ in zip(cycle(_default_markers), range(n))]

    # ── exceedance probability lookup ─────────────────────────────────────────
    # Maps event_id -> exceedance probability (1, 5, or 20)
    # Only events present in the exceedance CSV will be plotted when provided.
    exceedance_map = {}  # {event_id: exceedance_prob}
    if exceedance_csv is not None:
        exc_df = pd.read_csv(exceedance_csv)
        # Normalise column names: strip whitespace
        exc_df.columns = exc_df.columns.str.strip()
        # Expected columns after strip: 'Event', 'Exceedence Probability %'
        for _, row in exc_df.iterrows():
            eid = int(row["Event"])
            prob = int(row["Exceedence Probability %"])
            exceedance_map[eid] = prob
        print(
            f"  Loaded exceedance CSV: {len(exceedance_map)} events  ({exceedance_csv})"
        )

    # Colour assigned to each exceedance level for x-axis tick labels & shading
    _exc_level_colors = {
        1: "#d62728",  # red   → rarest / most severe
        5: "#ff7f0e",  # orange
        20: "#1f77b4",  # blue  → most frequent / least severe
    }

    # ── output directory ──────────────────────────────────────────────────────
    csv_path_obj = Path(csv_paths[0])
    if output_dir is None:
        out = csv_path_obj.parent / "avg_rmse_by_nodetype"
    else:
        out = csv_path_obj.parent / output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    dfs = []
    for label, path in zip(csv_labels, csv_paths):
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {label}: {len(df):,} rows  ({path})")

    # ── helper: compute avg std-RMSE for a single df, returning {event_id: rmse} ──
    def compute_avg_rmse_per_event(df, node_type_filter):
        candidate_events = sorted(
            df[(df["model_id"] == model_id) & (df["node_type"].isin(node_type_filter))][
                "event_id"
            ].unique()
        )

        # If exceedance CSV provided, restrict to only those events
        if exceedance_map:
            candidate_events = [e for e in candidate_events if e in exceedance_map]

        results = {}
        skipped = []

        for eid in candidate_events:
            ev_df = df[
                (df["model_id"] == model_id)
                & (df["event_id"] == eid)
                & (df["node_type"].isin(node_type_filter))
            ]
            if ev_df.empty:
                skipped.append((eid, "no rows after filter"))
                continue

            if len(node_type_filter) == 1:
                node_rmses = []
                for (ntype, nid), grp in ev_df.groupby(["node_type", "node_id"]):
                    if len(grp) < 2:
                        continue
                    std = DEFAULT_STD_DEV.get((model_id, ntype), np.nan)
                    sr = standardised_rmse(
                        grp["target_water_level"].to_numpy(),
                        grp["water_level"].to_numpy(),
                        std,
                    )
                    if not np.isnan(sr):
                        node_rmses.append(sr)

                if not node_rmses:
                    skipped.append((eid, "all nodes had <2 rows or nan RMSE"))
                    continue
                results[eid] = np.mean(node_rmses)

            else:
                type_means = []
                for ntype in node_type_filter:
                    type_df = ev_df[ev_df["node_type"] == ntype]
                    node_rmses = []
                    for nid, grp in type_df.groupby("node_id"):
                        if len(grp) < 2:
                            continue
                        std = DEFAULT_STD_DEV.get((model_id, ntype), np.nan)
                        sr = standardised_rmse(
                            grp["target_water_level"].to_numpy(),
                            grp["water_level"].to_numpy(),
                            std,
                        )
                        if not np.isnan(sr):
                            node_rmses.append(sr)
                    if node_rmses:
                        type_means.append(np.mean(node_rmses))

                if not type_means:
                    skipped.append((eid, "no valid type means"))
                    continue
                results[eid] = np.mean(type_means)

        print(f"      Plotted  events ({len(results)}): {sorted(results.keys())}")
        print(f"      Skipped events ({len(skipped)}):")
        for eid, reason in skipped:
            print(f"        event {eid}: {reason}")

        return results

    # ── helper: sort events by (exceedance_prob asc, event_id asc) ───────────
    def sort_events(event_ids):
        if exceedance_map:
            return sorted(event_ids, key=lambda e: (exceedance_map.get(e, 999), e))
        return sorted(event_ids)

    # ── 3 figures ─────────────────────────────────────────────────────────────
    figures = [
        {"node_types": [1], "label": "1D Nodes", "fname_tag": "1D"},
        {"node_types": [2], "label": "2D Nodes", "fname_tag": "2D"},
        {"node_types": [1, 2], "label": "All Nodes", "fname_tag": "all"},
    ]

    for fig_cfg in figures:
        node_types = fig_cfg["node_types"]
        fig_label = fig_cfg["label"]
        fname_tag = fig_cfg["fname_tag"]

        print(f"\n{'=' * 60}")
        print(f"Avg std-RMSE per event  |  {fig_label}  |  Model {model_id}")
        print(f"{'=' * 60}")

        all_series = []
        for df, label, color, ls, marker in zip(
            dfs, csv_labels, csv_colors, csv_linestyles, csv_markers
        ):
            print(f"\n  >> {label}")
            rmse_map = compute_avg_rmse_per_event(df, node_types)
            all_series.append((label, color, ls, marker, rmse_map))

        # Union of valid events, sorted by exceedance prob then event id
        all_valid_events = sort_events(
            set().union(*[set(s[4].keys()) for s in all_series])
        )

        if not all_valid_events:
            print("    Skipping - no data for any CSV")
            continue

        event_to_xpos = {eid: i for i, eid in enumerate(all_valid_events)}

        fig, ax = plt.subplots(figsize=(12, 5))
        any_data = False

        for label, color, ls, marker, rmse_map in all_series:
            if not rmse_map:
                print(f"    {label}: no data")
                continue

            xs = [event_to_xpos[eid] for eid in all_valid_events if eid in rmse_map]
            ys = [rmse_map[eid] for eid in all_valid_events if eid in rmse_map]

            ax.plot(
                xs,
                ys,
                label=label,
                color=color,
                linestyle=ls,
                marker=marker,
                linewidth=2,
                markersize=6,
                alpha=0.85,
            )
            print(f"    {label}: {len(xs)} events  |  mean = {np.mean(ys):.4f}")
            any_data = True

        if not any_data:
            print("    Skipping - no data")
            plt.close()
            continue

        # ── x-axis tick labels coloured by exceedance probability ─────────────
        tick_labels = [str(e) for e in all_valid_events]
        ax.set_xticks(range(len(all_valid_events)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        if exceedance_map:
            for tick, eid in zip(ax.get_xticklabels(), all_valid_events):
                prob = exceedance_map.get(eid)
                if prob in _exc_level_colors:
                    tick.set_color(_exc_level_colors[prob])

            # Background shading per exceedance group
            prev_prob = exceedance_map.get(all_valid_events[0])
            group_start = 0
            for i, eid in enumerate(all_valid_events):
                prob = exceedance_map.get(eid, prev_prob)
                if prob != prev_prob or i == len(all_valid_events) - 1:
                    end = i if prob != prev_prob else i + 1
                    ax.axvspan(
                        group_start - 0.5,
                        end - 0.5,
                        alpha=0.06,
                        color=_exc_level_colors.get(prev_prob, "grey"),
                        zorder=0,
                    )
                    group_start = i
                    prev_prob = prob

            # Legend entries for exceedance levels
            exc_patches = [
                mpatches.Patch(
                    color=_exc_level_colors[p],
                    label=f"Exceedance {p}%",
                )
                for p in sorted(_exc_level_colors)
                if p in set(exceedance_map.values())
            ]
            # Combine with existing line legend
            line_handles, line_labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=line_handles + exc_patches,
                labels=line_labels + [p.get_label() for p in exc_patches],
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
                edgecolor="black",
            )
        else:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="black")

        ax.set_title(
            f"Model {model_id} Avg {fig_label} Std RMSE Per Event",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel(
            "Event ID",
            fontsize=10,
        )
        ax.set_ylabel("Avg Std RMSE", fontsize=11)

        ax.margins(x=0.05)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.relim()
        ax.autoscale(axis="y", tight=False)
        ax.margins(y=0.15)

        plt.tight_layout()

        fname = f"avg_rmse_model{model_id}_{fname_tag}.png"
        fpath = out / fname
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        print(f"    Saved → {fpath}")
        plt.close()

    print(f"\n{'=' * 60}")
    print(f"Done.  Output directory: {out}")
    print(f"{'=' * 60}")


def concatenate_ground_truth(csv_file, gt_file, output_path):
    """
    Concatenate ground truth data to predictions file.
    Handles both CSV and Parquet formats for input files.
    Always saves output as CSV.

    Args:
        csv_file: Path to original predictions file (CSV or Parquet)
        gt_file: Path to ground truth file (CSV or Parquet)
        output_path: Path to save the combined CSV file
    """
    if csv_file.endswith(".parquet"):
        ori_file = pd.read_parquet(csv_file)
    else:
        ori_file = pd.read_csv(csv_file)

    if gt_file.endswith(".parquet"):
        gt_file_data = pd.read_parquet(gt_file)
    else:
        gt_file_data = pd.read_csv(gt_file)

    # Drop existing columns if they already exist
    if "target_water_level" in ori_file.columns:
        ori_file.drop(columns=["target_water_level"], inplace=True)
        print("⚠ Removed existing column: target_water_level")

    ori_file["target_water_level"] = gt_file_data["water_level"]

    ori_file.to_csv(output_path, index=False)
    print(f"✓ Saved combined data to {output_path}")


def main():
    # ----------------------------------------------------------------
    tiers = [
        {
            "tag": "top1_5",
            "paths": [
                "kaggle_submissions/top1_gt.csv",
                "kaggle_submissions/top2_gt.csv",
                "kaggle_submissions/top3_gt.csv",
                "kaggle_submissions/top4_gt.csv",
                "kaggle_submissions/top5_gt.csv",
            ],
            "labels": [f"Team {i}" for i in range(1, 6)],
        },
        {
            "tag": "top6_10",
            "paths": [
                "kaggle_submissions/top6_gt.csv",
                "kaggle_submissions/top7_gt.csv",
                "kaggle_submissions/top8_gt.csv",
                "kaggle_submissions/top9_gt.csv",
                "kaggle_submissions/top10_gt.csv",
            ],
            "labels": [f"Team {i}" for i in range(6, 11)],
        },
    ]
    # top 1-10 = tier 1 paths + tier 2 paths
    tiers.append(
        {
            "tag": "top1_10",
            "paths": tiers[0]["paths"] + tiers[1]["paths"],
            "labels": [f"Team {i}" for i in range(1, 11)],
        }
    )

    for tier in tiers:
        paths, labels, tag = tier["paths"], tier["labels"], tier["tag"]
        print(f"\n{'#' * 60}")
        print(f"#  {tag}  ({len(paths)} teams)")
        print(f"{'#' * 60}")

        # --- per-node water-level vs ground truth (Model 1 & Model 2) ---
        plot_multi_csv_comparison_model1(
            csv_paths=paths,
            csv_labels=labels,
            output_dir=f"multi_csv_comparison_baselines_{tag}_model1",
        )
        plot_multi_csv_comparison_model2(
            csv_paths=paths,
            csv_labels=labels,
            output_dir=f"multi_csv_comparison_baselines_{tag}_model2",
        )

        # --- avg std-RMSE per event, ordered/coloured by exceedance prob ---
        plot_avg_rmse_across_nodes_v2(
            model_id=1,
            csv_paths=paths,
            csv_labels=labels,
            output_dir=f"multi_csv_events_comparison_baselines_{tag}_model1_v2",
            exceedance_csv="test_dataset_model1.csv",
        )
        plot_avg_rmse_across_nodes_v2(
            model_id=2,
            csv_paths=paths,
            csv_labels=labels,
            output_dir=f"multi_csv_events_comparison_baselines_{tag}_model2_v2",
            exceedance_csv="test_dataset_model2.csv",
        )

    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
