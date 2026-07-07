import os
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUTPUT_DIR = "taxonomy_tree_outputs"

# ── Palette ────────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
TC = "#0D0D0D"  # text colour on white
BADGE_TC = "#FFFFFF"

FAM = {
    "root": ("#E0F2F1", "#00695C"),
    "classical": ("#E3F2FD", "#1565C0"),
    "dl": ("#E3F2FD", "#1565C0"),
    "ensemble": ("#E3F2FD", "#1565C0"),
    "train": ("#E0F2F1", "#00695C"),
    "loss": ("#FFF3E0", "#EF6C00"),
    "optim": ("#FFF3E0", "#EF6C00"),
    "feat": ("#FFF3E0", "#EF6C00"),
    "post": ("#FFF3E0", "#EF6C00"),
    "reg": ("#FFF3E0", "#EF6C00"),
}

# ── Layout knobs ───────────────────────────────────────────────────────────────
BOX_W = 2.20  # horizontal span of each box (= column width unit)
BOX_H = 0.42  # box height
H_STEP = 2.40  # column-to-column spacing (centre-to-centre, x axis)
V_PAD = 0.24  # vertical gap between sibling boxes (in data units)
LW = 0.60  # connector line width
FS = 12  # base font size
WC = 22  # wrap chars per line


# ══════════════════════════════════════════════════════════════════════════════
# TAXONOMY  (same structure as before — dict = internal, str = leaf colour tag)
# ══════════════════════════════════════════════════════════════════════════════

ENSEMBLE_TAXONOMY = {
    "Homogeneous Ensemble": {
        "_fam": "ensemble",
        "Multi-Seed Averaging": "ensemble",
        "Nodewise Weight Blending": "ensemble",
        "Dynamic Event Weight Blending": "ensemble",
    },
    "Heterogeneous Ensemble": {
        "_fam": "ensemble",
        "GNN + XGBoost Residual Corrector": "ensemble",
        "ARX Baseline + GRU Corrector": "ensemble",
        "Ridge + LGB + GNN 3-Model Blend": "ensemble",
        "Gradient Boosting + DL Blend": "ensemble",
        "LGB + BiGRU Weighted": "ensemble",
        "LSTM + GNN Ensemble": "ensemble",
    },
    "Statistical Aggregation": {
        "_fam": "ensemble",
        "Central Tendency Blending": "ensemble",
        "Node-Average Baseline": "ensemble",
    },
}

ARCH_TAXONOMY = {
    "Classical ML": {
        "_fam": "classical",
        "Linear Models": {
            "_fam": "classical",
            "Ridge Regression": "classical",
            "ARX Baseline": "classical",
        },
        "Gradient Boosting": {
            "_fam": "classical",
            "LightGBM": "classical",
            "XGBoost": "classical",
            "CatBoost": "classical",
        },
        "Analogy-Based": {
            "_fam": "classical",
            "KNN Event Matching": "classical",
        },
        "Physics-Based": {
            "_fam": "classical",
            "Nash Cascade\nState-Space Model": "classical",
        },
    },
    "Deep Learning": {
        "_fam": "dl",
        "Graph Neural\nNetworks": {
            "_fam": "dl",
            "Homogeneous GNN": "dl",
            "Heterogeneous GNN": "dl",
        },
        "Sequential\nModels": {
            "_fam": "dl",
            "LSTM-based": "dl",
            "GRU-based": "dl",
            "Transformer-based": "dl",
            "Temporal CNN": "dl",
        },
    },
}

LOSS_FUNCTION_TAXONOMY = {
    "Weighted Loss": {
        "_fam": "loss",
        "SRMSE-Aligned Node Weighting": "loss",
        "Inverse-Variance Sample Weights": "loss",
        "Sigma-Weighted Normalised MSE": "loss",
        "Event-Balanced Loss": "loss",
    },
    "Temporal Loss": {
        "_fam": "loss",
        "Time-Weighted Loss (Late Steps)": "loss",
        "Trajectory Loss (Cumulative MSE)": "loss",
    },
    "Physics Constraints Loss": {
        "_fam": "loss",
        "Mass Conservation Penalty": "loss",
        "Non-Negativity Soft Loss Penalty": "loss",
        "Bias Penalty          (AR Drift)": "loss",
        "Multi-Component Physics Loss": "loss",
    },
}

OPTIMISATION_TAXONOMY = {
    "_fam": "optim",
    # How training is sequenced/paced over time
    "Training Curriculum": {
        "_fam": "optim",
        "Horizon-Expanding Curriculum": "optim",
        "Stochastic Teacher Forcing": "optim",
        "DAgger-style AR Robustness": "optim",
    },
    # Optimiser and gradient-level mechanics
    "Gradient Control": {
        "_fam": "optim",
        "Truncated BPTT": "optim",
        "Iterative Refine + Reset Adam": "optim",
    },
    # Techniques that stabilise training without touching the loss
    "Training Stabilisation": {
        "_fam": "optim",
        "EMA Weight Averaging": "optim",
        "Warm Restart from Checkpoint": "optim",
    },
}

POST_PROCESSING_TAXONOMY = {
    "_fam": "post",
    "Bias Correction": {
        "_fam": "post",
        "Zone-Aware Bias Subtraction": "post",
        "Node-Mean Residue Shift": "post",
    },
    "Smoothing": {
        "_fam": "post",
        "Savitzky-Golay Filter": "post",
        "EMA Smoothing       (AR Rollout)": "post",
        "Horizon-Capped      Forward Fill": "post",
    },
    "Physical Bounding": {
        "_fam": "post",
        "Min-Depth Hard Clip (Non-Negativity)": "post",
        "Physical Floor Clipping": "post",
        "Recession Gate      (Anti-Spike)": "post",
    },
}

FEATURE_ENGINEERING_TAXONOMY = {
    "_fam": "feat",
    # Captures memory + sequence patterns
    "Temporal Dynamics": {
        "_fam": "feat",
        "Temporal Lag / AR History": "feat",
        "Warmup State Summary": "feat",
        "Temporal Water-Level Derivatives": "feat",
        "Temporal Position Encoding": "feat",
        "Rainfall EMA / Multi-scale Smoothing": "feat",
    },
    # Drives the system (exogenous signals)
    "Hydrometeorological Inputs": {
        "_fam": "feat",
        "Cumulative & Rolling Rainfall": "feat",
        "Future Rainfall Lookahead": "feat",
    },
    # Encodes physics & constraints
    "Hydraulic & Physical State": {
        "_fam": "feat",
        "Pipe Fill Fraction & Surcharge Indicator": "feat",
        "Pipe Capacity Proxy (Manning's)": "feat",
        "Depth / Head Relative to Invert & Surface": "feat",
        "Mass Balance / Storage Deficit": "feat",
        "Physics-Informed Drainage / Recession": "feat",
        "Spatial Momentum / Kinematic Features": "feat",
        "Auxiliary Flow Prediction Stack": "feat",
    },
    # Enables propagation & spatial reasoning
    "Spatial & Structural Context": {
        "_fam": "feat",
        "Graph Neighbor Aggregation": "feat",
        "Multi-hop Graph Context": "feat",
        "Terrain-Derived (TWI / HAND / SPI)": "feat",
        "1D-2D Coupling / Cross-Domain Features": "feat",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Node class
# ══════════════════════════════════════════════════════════════════════════════
class Node:
    __slots__ = ("label", "children", "x", "y", "fam", "depth")

    def __init__(self, label, fam="root"):
        self.label = label
        self.children = []
        self.x = self.y = 0.0
        self.fam = fam
        self.depth = 0

    def is_leaf(self):
        return not self.children


# ══════════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════════
def _build(d, parent, inh_fam):
    fam = d.get("_fam", inh_fam)
    for key, val in d.items():
        if key == "_fam":
            continue
        if isinstance(val, dict):
            child = Node(key, fam=val.get("_fam", fam))
            _build(val, child, fam)
        else:
            child = Node(key, fam=val)
        parent.children.append(child)


def build_tree(taxonomy, root_label, root_fam="root"):
    root = Node(root_label, fam=root_fam)
    root.depth = 0
    fam = next(
        (v.get("_fam", root_fam) for v in taxonomy.values() if isinstance(v, dict)),
        root_fam,
    )
    root.fam = fam
    _build(taxonomy, root, fam)
    _set_depths(root, d=0)  # ← re-run depth assignment from root after build
    return root


# ══════════════════════════════════════════════════════════════════════════════
# Horizontal layout
#   x-axis = depth (column)
#   y-axis = position within column (row)
# ══════════════════════════════════════════════════════════════════════════════
def _set_depths(node, d=0):
    node.depth = d
    for c in node.children:
        _set_depths(c, d + 1)


def _subtree_height(node):
    """Total vertical span needed for this subtree."""
    if node.is_leaf():
        return BOX_H
    total = sum(_subtree_height(c) for c in node.children)
    total += V_PAD * (len(node.children) - 1)
    return total


def _assign_y(node, top_cursor):
    h = _subtree_height(node)
    node.y = top_cursor + h / 2  # default: centre over full span

    if not node.is_leaf():
        cursor = top_cursor
        for c in node.children:
            ch = _subtree_height(c)
            _assign_y(c, cursor)
            cursor += ch + V_PAD

        # Snap to middle child if odd number of children
        n = len(node.children)
        if n % 2 == 1:
            node.y = node.children[n // 2].y


def layout(root):
    _set_depths(root)
    _assign_y(root, 0.0)

    # x = depth × H_STEP
    def set_x(n):
        n.x = n.depth * H_STEP
        for c in n.children:
            set_x(c)

    set_x(root)

    def collect(n):
        yield n
        for c in n.children:
            yield from collect(c)

    all_nodes = list(collect(root))

    total_w = max(n.x for n in all_nodes)
    total_h = max(n.y for n in all_nodes) + BOX_H / 2
    return total_w, total_h, all_nodes


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════
def _fc_ec(fam):
    return FAM.get(fam, ("#F5F5F5", "#555"))


def _draw_box(ax, cx, cy, label, fam, is_root=False):
    fc, ec = _fc_ec(fam)
    bw = BOX_W * (1.10 if is_root else 1.0)
    bh = BOX_H * (1.50 if is_root else 1.3)
    lw = 0.9 if is_root else 0.55
    fw = "bold" if is_root else "normal"

    ax.add_patch(
        FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0,rounding_size=0.07",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
            clip_on=False,
        )
    )
    wrapped = "\n".join(textwrap.wrap(label.replace("\n", " "), WC))
    fs = FS + (0.8 if is_root else 0)
    ax.text(
        cx,
        cy,
        wrapped,
        ha="center",
        va="center",
        fontsize=fs,
        color=TC,
        fontweight=fw,
        zorder=4,
        linespacing=1.18,
        clip_on=False,
    )


def _draw_edges(ax, node):
    if node.is_leaf():
        return
    _, ec = _fc_ec(node.fam)

    # Match the actual rendered box width (root is 1.10× wider)
    actual_bw = BOX_W * (1.10 if not node.children or node.depth == 0 else 1.0)
    px_right = node.x + actual_bw / 2

    for c in node.children:
        cx_left = c.x - BOX_W / 2
        mid_x = (px_right + cx_left) / 2

        ax.plot(
            [px_right, mid_x],
            [node.y, node.y],
            color=ec,
            lw=LW,
            zorder=1,
            clip_on=False,
            solid_capstyle="round",
        )
        ax.plot([mid_x, mid_x], [node.y, c.y], color=ec, lw=LW, zorder=1, clip_on=False)
        ax.plot(
            [mid_x, cx_left],
            [c.y, c.y],
            color=ec,
            lw=LW,
            zorder=1,
            clip_on=False,
            solid_capstyle="round",
        )
        _draw_edges(ax, c)


# ══════════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════════
def _legend(ax, entries, loc="lower right"):
    patches = [
        mpatches.Patch(facecolor=FAM[k][0], edgecolor=FAM[k][1], label=lbl)
        for lbl, k in entries
    ]
    ax.legend(
        handles=patches,
        loc=loc,
        fontsize=7.0,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.6,
        labelspacing=0.4,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main draw
# ══════════════════════════════════════════════════════════════════════════════
ARCH_LEGEND = [
    ("Classical ML", "classical"),
    ("Deep Learning", "dl"),
    ("Ensemble Models", "ensemble"),
]

TRAIN_LEGEND = [
    ("Loss Function Design", "loss"),
    ("Optimisation Schedule", "optim"),
    ("Feature Engineering", "feat"),
    ("Regularisation & Norm.", "reg"),
    ("Post-Processing", "post"),
]


def draw_figure(taxonomy, root_label, root_fam, title, legend_entries, out_stem):
    root = build_tree(taxonomy, root_label, root_fam=root_fam)
    total_w, total_h, all_nodes = layout(root)

    # Scale: want figure ~10 inches wide (fits LaTeX \textwidth at 300 dpi)
    # total_w in data units ≈ max_depth × H_STEP
    scale_x = 10.0 / (total_w + BOX_W * 1.2)
    scale_y = scale_x  # keep aspect

    FIG_W = (total_w + BOX_W * 1.2) * scale_x
    FIG_H = max(5.0, (total_h + BOX_H * 2.5) * scale_y)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    pad_x = BOX_W * 0.65
    pad_y = BOX_H * 1.4
    ax.set_xlim(-pad_x, total_w + pad_x)
    ax.set_ylim(-pad_y, total_h + pad_y)
    ax.axis("off")

    _draw_edges(ax, root)
    for n in all_nodes:
        _draw_box(ax, n.x, n.y, n.label, n.fam, is_root=(n is root))

    plt.tight_layout(pad=0.12)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        p = os.path.join(OUTPUT_DIR, f"{out_stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=BG, **kw)
        print(f"[OK] {ext.upper()} → {p}")
    plt.close()


if __name__ == "__main__":
    draw_figure(
        ARCH_TAXONOMY,
        root_label="Model Architectures",
        root_fam="root",
        title="UrbanFloodBench — Model Architecture Taxonomy",
        legend_entries=ARCH_LEGEND,
        out_stem="taxonomy_model_architectures",
    )
    draw_figure(
        ENSEMBLE_TAXONOMY,
        root_label="Ensemble Architectures",
        root_fam="root",
        title="UrbanFloodBench — Ensemble Architecture Taxonomy",
        legend_entries=ARCH_LEGEND,
        out_stem="taxonomy_ensemble",
    )
    draw_figure(
        LOSS_FUNCTION_TAXONOMY,
        root_label="Loss Functions",
        root_fam="train",
        title="UrbanFloodBench — Loss Functions Taxonomy",
        legend_entries=TRAIN_LEGEND,
        out_stem="taxonomy_loss_functions",
    )
    draw_figure(
        POST_PROCESSING_TAXONOMY,
        root_label="Post-Processing",
        root_fam="train",
        title="UrbanFloodBench — Post Processing Taxonomy",
        legend_entries=TRAIN_LEGEND,
        out_stem="taxonomy_post_processing",
    )
    draw_figure(
        OPTIMISATION_TAXONOMY,
        root_label="Optimizations",
        root_fam="train",
        title="UrbanFloodBench — Training Optimizations Taxonomy",
        legend_entries=TRAIN_LEGEND,
        out_stem="taxonomy_training_optimizations",
    )
