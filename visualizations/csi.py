"""
For the 2D domain only, this script converts water level to water depth using
each cell's minimum ground elevation, thresholds the depth into a binary
flooded / non-flooded mask, and compares the surrogate-model prediction
against the HEC-RAS reference simulation (`target_water_level`).

    depth_pred = max(water_level        - min_elevation, 0)   # prediction
    depth_ref  = max(target_water_level - min_elevation, 0)   # HEC-RAS ref
    flooded    = depth >= threshold

Confusion counts are computed over 2D cell x timestep comparisons:
    TP: predicted flooded      AND reference flooded
    FP: predicted flooded      AND reference non-flooded
    TN: predicted non-flooded  AND reference non-flooded
    FN: predicted non-flooded  AND reference flooded

    CSI       = TP / (TP + FP + FN)                 # TN is not used in CSI
    Accuracy  = (TP + TN) / (TP + FP + TN + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)

Two thresholds:
    CSI5  : 5 cm  -> 0.05 m (SI)  |  ~0.167 ft (US customary)
    CSI30 : 30 cm -> 0.30 m (SI)  |  ~1.0   ft (US customary)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---- flood-depth thresholds -------------------------------------------------
THRESHOLDS_FT = {"CSI5": 0.167, "CSI30": 1.0}  # US customary (feet)
THRESHOLDS_M = {"CSI5": 0.05, "CSI30": 0.30}  # SI (metres)

UNDEFINED = 1.0  # value assigned when a metric denominator is 0

# metrics that share the event -> model -> team averaging philosophy
METRIC_NAMES = ["CSI", "Accuracy", "Precision", "Recall", "F1"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_min_elev(static_csv):
    """Return min_elevation indexed by node_idx, with NaN cells dropped."""
    s = pd.read_csv(static_csv).set_index("node_idx")["min_elevation"]
    n_nan = int(s.isna().sum())
    s = s.dropna()
    print(
        f"    static {Path(static_csv).name}: {len(s)} usable 2D cells "
        f"({n_nan} dropped for NaN min_elevation)"
    )
    return s


def ensure_timestep(df):
    """Guarantee a 'timestep' column.

    If absent, reconstruct timestep as the within-(model,event,node) row order.
    This assumes rows are already sorted in the correct temporal order.
    """
    if "timestep" in df.columns:
        return df

    df = df.copy()
    df["timestep"] = df.groupby(
        ["model_id", "event_id", "node_type", "node_id"]
    ).cumcount()
    return df


def _safe_ratio(num, den):
    return UNDEFINED if den == 0 else num / den


def confusion_metrics(tp, fp, tn, fn):
    """Scalar metrics from pooled confusion counts."""
    csi = _safe_ratio(tp, tp + fp + fn)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    total = tp + fp + tn + fn
    accuracy = _safe_ratio(tp + tn, total)
    denom_f1 = precision + recall
    f1 = UNDEFINED if denom_f1 == 0 else 2 * precision * recall / denom_f1

    return dict(
        TP=int(tp),
        FP=int(fp),
        TN=int(tn),
        FN=int(fn),
        CSI=csi,
        Accuracy=accuracy,
        Precision=precision,
        Recall=recall,
        F1=f1,
    )


def macro_csi(df):
    """Mean CSI over (event, timestep) spatial slices.

    For each event-timestep slice, TP/FP/FN are pooled over all 2D cells and
    one CSI is computed. The final value is the mean over slices.
    """
    g = df.groupby(["event_id", "timestep"])[["tp", "fp", "fn"]].sum()
    denom = g["tp"] + g["fp"] + g["fn"]
    csi_values = np.where(
        denom.values == 0,
        UNDEFINED,
        g["tp"].values / denom.replace(0, np.nan).values,
    )
    return float(np.nanmean(csi_values)) if len(csi_values) else np.nan


def event_level_metrics(df):
    """Per-event metrics for one urban model's 2D data.

    For each event:
        1. Pool TP/FP/TN/FN across all 2D cells and all timesteps.
        2. Compute ALL metrics (CSI, Accuracy, Precision, Recall, F1) for the
           event from the pooled counts.
    Then average the per-event metric values to obtain the model-level scores.

    This avoids computing metrics for individual cells, which is not meaningful
    for flood-extent evaluation, and follows the event/model weighting
    philosophy of the benchmark SRMSE metric.

    Returns
    -------
    model_avg : dict
        Event-averaged value for each metric in METRIC_NAMES (per-model score).
    per_event : pd.DataFrame
        Per-event metrics and counts, one row per event_id.
    """
    event_counts = df.groupby("event_id")[["tp", "fp", "tn", "fn"]].sum()

    per_event_rows = []
    for event_id, r in event_counts.iterrows():
        m = confusion_metrics(r["tp"], r["fp"], r["tn"], r["fn"])
        m["event_id"] = event_id
        per_event_rows.append(m)

    per_event = pd.DataFrame(per_event_rows)

    if per_event.empty:
        model_avg = {name: np.nan for name in METRIC_NAMES}
    else:
        # each event weighted equally
        model_avg = {name: float(per_event[name].mean()) for name in METRIC_NAMES}

    return model_avg, per_event


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def compute_metrics_2d(
    submission_paths,
    submission_labels,
    static_by_model,  # {model_id: path to that model's 2d_nodes_static.csv}
    units="ft",  # "ft" (US customary) or "m" (SI)
    out_model_csv=None,  # per-model + per-team + overall summary
    out_event_csv=None,  # per-event values (all metrics)
    out_team_csv=None,  # final per-team scores only
):
    if units not in {"ft", "m"}:
        raise ValueError("units must be either 'ft' or 'm'")

    thresholds = THRESHOLDS_FT if units == "ft" else THRESHOLDS_M
    print(f"Thresholds ({units}): {thresholds}")

    if len(submission_paths) != len(submission_labels):
        raise ValueError(
            "submission_paths and submission_labels must have the same length"
        )

    min_elev = {int(m): load_min_elev(p) for m, p in static_by_model.items()}

    model_rows = []  # per (team, model, threshold): event-averaged + micro/macro
    event_rows = []  # per (team, model, event, threshold): all metrics

    for path, label in zip(submission_paths, submission_labels):
        path = Path(path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

        required_cols = {
            "model_id",
            "event_id",
            "node_type",
            "node_id",
            "water_level",
            "target_water_level",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        df = df[df["node_type"] == 2].copy()  # 2D domain only
        df = ensure_timestep(df)
        print(f"\n{label}: {len(df):,} 2D rows")

        for model_id, sub in df.groupby("model_id"):
            model_id = int(model_id)
            if model_id not in min_elev:
                print(f"  ! model {model_id}: no static file provided, skipped")
                continue

            elev = sub["node_id"].map(min_elev[model_id])
            keep = elev.notna()
            sub = sub.loc[keep].copy()
            elev = elev.loc[keep].to_numpy()

            if sub.empty:
                print(
                    f"  ! model {model_id}: no usable 2D rows after elevation filtering, skipped"
                )
                continue

            # Convert water level / WSE to non-negative water depth using
            # minimum cell elevation, as specified for the CSI calculation.
            depth_pred = np.maximum(sub["water_level"].to_numpy() - elev, 0.0)
            depth_ref = np.maximum(sub["target_water_level"].to_numpy() - elev, 0.0)

            for tname, thr in thresholds.items():
                pred = depth_pred >= thr
                ref = depth_ref >= thr

                sub_metric = sub[["event_id", "timestep", "node_id"]].copy()
                sub_metric["tp"] = pred & ref
                sub_metric["fp"] = pred & ~ref
                sub_metric["tn"] = ~pred & ~ref
                sub_metric["fn"] = ~pred & ref

                # Micro: all metrics pooled over the full model dataset.
                micro = confusion_metrics(
                    sub_metric["tp"].sum(),
                    sub_metric["fp"].sum(),
                    sub_metric["tn"].sum(),
                    sub_metric["fn"].sum(),
                )

                # Macro CSI: mean over event-timestep spatial slices (diagnostic).
                macro = macro_csi(sub_metric)

                # Event-level: pool per event, compute all metrics per event,
                # then average events equally. This is the recommended metric.
                model_avg, per_event = event_level_metrics(sub_metric)

                for _, er in per_event.iterrows():
                    event_rows.append(
                        dict(
                            submission=label,
                            model_id=model_id,
                            event_id=er["event_id"],
                            metric_set=tname,
                            threshold=thr,
                            CSI=er["CSI"],
                            Accuracy=er["Accuracy"],
                            Precision=er["Precision"],
                            Recall=er["Recall"],
                            F1=er["F1"],
                            TP=int(er["TP"]),
                            FP=int(er["FP"]),
                            TN=int(er["TN"]),
                            FN=int(er["FN"]),
                        )
                    )

                model_rows.append(
                    dict(
                        submission=label,
                        model_id=model_id,
                        metric_set=tname,
                        threshold=thr,
                        N_events=int(len(per_event)),
                        # event-averaged (recommended manuscript numbers)
                        CSI=model_avg["CSI"],
                        Accuracy=model_avg["Accuracy"],
                        Precision=model_avg["Precision"],
                        Recall=model_avg["Recall"],
                        F1=model_avg["F1"],
                        # micro (pooled counts) diagnostics
                        CSI_micro=micro["CSI"],
                        Accuracy_micro=micro["Accuracy"],
                        Precision_micro=micro["Precision"],
                        Recall_micro=micro["Recall"],
                        F1_micro=micro["F1"],
                        # macro CSI diagnostic
                        CSI_macro=macro,
                        # pooled counts
                        TP=micro["TP"],
                        FP=micro["FP"],
                        TN=micro["TN"],
                        FN=micro["FN"],
                    )
                )

                print(
                    f"  model {model_id} {tname}: "
                    f"CSI={model_avg['CSI']:.4f}  Acc={model_avg['Accuracy']:.4f}  "
                    f"Prec={model_avg['Precision']:.4f}  Rec={model_avg['Recall']:.4f}  "
                    f"F1={model_avg['F1']:.4f}  (n_events={len(per_event)})"
                )

    model_res = pd.DataFrame(model_rows)
    event_res = pd.DataFrame(event_rows)

    if model_res.empty:
        raise RuntimeError("No results were computed. Check inputs and static files.")

    # ---- per-team overall (average of per-model event-averaged scores) ------
    # Every model contributes equally to the team score, matching the SRMSE
    # final step. Micro columns are recomputed from pooled counts as a check.
    team_rows = []
    for (label, tname), grp in model_res.groupby(["submission", "metric_set"]):
        tp, fp, tn, fn = grp[["TP", "FP", "TN", "FN"]].sum()
        micro = confusion_metrics(tp, fp, tn, fn)

        team_rows.append(
            dict(
                submission=label,
                model_id="TEAM (per-model avg)",
                metric_set=tname,
                threshold=grp["threshold"].iloc[0],
                N_events=int(grp["N_events"].sum()),
                # per-team event-averaged score = mean of per-model scores
                CSI=grp["CSI"].mean(),
                Accuracy=grp["Accuracy"].mean(),
                Precision=grp["Precision"].mean(),
                Recall=grp["Recall"].mean(),
                F1=grp["F1"].mean(),
                # micro from pooled counts
                CSI_micro=micro["CSI"],
                Accuracy_micro=micro["Accuracy"],
                Precision_micro=micro["Precision"],
                Recall_micro=micro["Recall"],
                F1_micro=micro["F1"],
                CSI_macro=grp["CSI_macro"].mean(),
                TP=micro["TP"],
                FP=micro["FP"],
                TN=micro["TN"],
                FN=micro["FN"],
            )
        )

    team_res = pd.DataFrame(team_rows)

    # combined summary: per-model rows followed by per-team rows
    summary = pd.concat([model_res, team_res], ignore_index=True)

    if out_model_csv:
        summary.to_csv(out_model_csv, index=False)
        print(f"\nSaved per-model + per-team summary -> {out_model_csv}")

    if out_event_csv:
        event_res.to_csv(out_event_csv, index=False)
        print(f"Saved per-event values -> {out_event_csv}")

    if out_team_csv:
        team_cols = [
            "submission",
            "metric_set",
            "threshold",
            "N_events",
            "CSI",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]
        team_res[team_cols].to_csv(out_team_csv, index=False)
        print(f"Saved final per-team scores -> {out_team_csv}")

    return summary, event_res, team_res


def main():
    here = Path(__file__).resolve().parent

    # One static file per model.
    static_by_model = {
        1: here / "data/model1_2d_nodes_static.csv",
        2: here / "data/model2_2d_nodes_static.csv",
    }

    submission_paths = [
        here / "kaggle_submissions" / f"top{i}_gt.csv" for i in range(1, 11)
    ]
    submission_labels = [f"Team {i}" for i in range(1, 11)]

    summary, event_res, team_res = compute_metrics_2d(
        submission_paths=submission_paths,
        submission_labels=submission_labels,
        static_by_model=static_by_model,
        units="ft",  # competition models 1 and 2 use US customary units
        out_model_csv=here / "additional_metrics/flood_extent_summary_v3.csv",
        out_event_csv=here / "additional_metrics/flood_extent_event_values_v3.csv",
        out_team_csv=here / "additional_metrics/flood_extent_team_scores_v3.csv",
    )

    # ---- tidy views ---------------------------------------------------------
    # Per-model + per-team, event-averaged metrics, one block per threshold.
    print(
        "\n================ EVENT-AVERAGED METRICS (per model & per team) ================"
    )
    print("Values are averaged over events (equal weight), then models (equal weight).")
    view = summary.pivot_table(
        index=["submission", "model_id"],
        columns="metric_set",
        values=METRIC_NAMES,
    )
    print(view.round(4))

    # Final leaderboard: per-team scores only.
    print("\n================ FINAL PER-TEAM LEADERBOARD ================")
    lead = team_res.pivot_table(
        index="submission",
        columns="metric_set",
        values=METRIC_NAMES,
    )
    print(lead.round(4))


if __name__ == "__main__":
    main()
