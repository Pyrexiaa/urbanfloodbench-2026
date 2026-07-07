import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── THEME ────────────────────────────────────────────────────────────────────
BLACK = "#0D0D0D"
DARK = "#1C1C1C"
MID_GREY = "#4A4A4A"
LIGHT_GREY = "#AAAAAA"
DIVIDER = "#DDDDDD"
BG = "#FFFFFF"

BLUE_DARK = "#0A3D6B"
BLUE_MID = "#1565C0"
BLUE_STD = "#1E88E5"
BLUE_LITE = "#90CAF9"
BLUE_PALE = "#E3F2FD"

# Blue ramp for groupings (darkest → lightest)
BLUE_RAMP = [
    "#0A3D6B",
    "#1565C0",
    "#1E88E5",
    "#42A5F5",
    "#90CAF9",
    "#BBDEFB",
    "#E3F2FD",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": BLACK,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": DIVIDER,
        "axes.labelcolor": MID_GREY,
        "xtick.color": LIGHT_GREY,
        "ytick.color": BLACK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

kaggle = {
    "Entrants": 1056,
    "Participants": 292,
    "Teams": 264,
    "Submissions": 3799,
}

region = {
    "Japan": 5,
    "Pakistan": 5,
    "Indonesia": 1,
    "India": 6,
    "United States": 2,
    "Germany": 1,
    "Vietnam": 1,
    "South Korea": 1,
    "Sri Lanka": 1,
    "Philippines": 1,
    "Ukraine": 1,
    "Bangladesh": 1,
    "Canada": 2,
    "Malaysia": 2,
    "Singapore": 1,
    "Egypt": 1,
    "China": 2,
    "Georgia": 1,
    "Russia": 1,
    "United Kingdom": 1,
}

SURVEY_N = 39

survey = {
    "Primary Role": {
        "Student": 24,
        "Researcher": 7,
        "Industry": 3,
        "Other": 5,
    },
    "Prior Kaggle\nCompetitions": {
        "0": 8,
        "1-9": 20,
        "10-19": 4,
        "20+": 7,
    },
    "AI/ML in\nHydraulics": {
        "None": 26,
        "Beginner": 8,
        "Intermediate": 4,
        "Advanced": 1,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Kaggle Competition Stats
# ════════════════════════════════════════════════════════════════════════════
def plot_kaggle_stats(save_path="kaggle_stats"):
    # (label, value, sub-description shown in smaller font inside the box)
    funnel_items = [
        ("Entrants", kaggle["Entrants"], "Registered users"),
        ("Participants", kaggle["Participants"], "Submitted ≥1 entry"),
        ("Teams", kaggle["Teams"], "Unique teams formed"),
        ("Submissions", kaggle["Submissions"], "Total entries submitted"),
    ]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 4.4)

    funnel_blues = [BLUE_DARK, BLUE_MID, BLUE_STD, BLUE_LITE]
    max_val = max(v for _, v, _ in funnel_items)
    min_w = 0.54  # minimum width — always fits the longest label
    scale_range = 0.38  # extra width distributed proportionally above the floor

    for i, (label, val, desc) in enumerate(funnel_items):
        w = min_w + scale_range * (val / max_val)
        x0 = 0.5 - w / 2
        y_p = 3.65 - i * 1.05
        col = funnel_blues[i]
        txt_col = "white" if col in [BLUE_DARK, BLUE_MID, BLUE_STD] else BLACK

        rect = FancyBboxPatch(
            (x0, y_p), w, 0.72, boxstyle="round,pad=0.01", linewidth=0, facecolor=col
        )
        ax.add_patch(rect)

        # Bold label + count (upper half of box)
        ax.text(
            0.50,
            y_p + 0.50,
            f"{label}   {val:,}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=txt_col,
        )

        # Smaller sub-description (lower half of box)
        ax.text(
            0.50,
            y_p + 0.20,
            desc,
            ha="center",
            va="center",
            fontsize=8,
            color=txt_col,
            alpha=0.82,
        )

        # Drop-off annotation between boxes
        if i < len(funnel_items) - 1:
            next_val = funnel_items[i + 1][1]
            pct = next_val / val * 100
            ax.annotate(
                f"↓  {next_val:,} ({pct:.2f}%)",
                xy=(0.50, y_p - 0.08),
                ha="center",
                va="top",
                fontsize=8,
                color=BLACK,
            )

    plt.tight_layout(pad=2.0)
    fig.savefig(f"{save_path}.png", dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight", facecolor=BG)
    print(f"Saved {save_path}.png/.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Survey Overview (grouped bar, one cluster per question)
# ════════════════════════════════════════════════════════════════════════════
def plot_survey_overview(save_path="survey_overview"):
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Layout constants ─────────────────────────────────────────────────────
    CLUSTER_GAP = 0.50  # vertical gap between question clusters
    BAR_HEIGHT = 0.26  # height of each individual bar
    BAR_PAD = 0.10  # padding between bars in a cluster

    y_cursor = 0.0

    # assign a blue shade per option — per-question local ramp (restart each cluster)
    for question, data in survey.items():
        options = list(data.keys())
        counts = list(data.values())
        pcts = [c / SURVEY_N * 100 for c in counts]

        # sort descending within cluster
        paired = sorted(zip(pcts, options, counts), reverse=True)
        pcts_s = [p for p, _, _ in paired]
        opts_s = [o for _, o, _ in paired]
        counts_s = [c for _, _, c in paired]

        n_opts = len(opts_s)
        cluster_height = n_opts * (BAR_HEIGHT + BAR_PAD) - BAR_PAD
        cluster_mid = y_cursor + cluster_height / 2

        # use a local blue ramp per cluster so colours stay meaningful
        local_ramp = [BLUE_RAMP[k % len(BLUE_RAMP)] for k in range(n_opts)]

        for j, (opt, pct, cnt) in enumerate(zip(opts_s, pcts_s, counts_s)):
            y_bar = y_cursor + j * (BAR_HEIGHT + BAR_PAD)
            col = local_ramp[j]

            ax.barh(
                y_bar,
                pct,
                height=BAR_HEIGHT,
                color=col,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )

            # value label to the right of bar — actual count first, then percentage
            ax.text(
                pct + 0.6,
                y_bar,
                f"{cnt} ({pct:.2f}%)",
                va="center",
                fontsize=16,
                color=MID_GREY,
            )

            # option label to the left — pushed further out to avoid overlap
            ax.text(-1.5, y_bar, opt, va="center", ha="right", fontsize=16, color=BLACK)

        # question label drawn directly as text, far left — avoids overlap with option labels
        ax.text(
            -25,
            cluster_mid,
            question,
            va="center",
            ha="left",
            fontsize=16,
            fontweight="bold",
            color=BLACK,
        )

        y_cursor += cluster_height + CLUSTER_GAP

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlim(0, 70)  # axis starts at 0%; left space handled by text x-coords
    ax.set_ylim(-0.4, y_cursor - CLUSTER_GAP + 0.4)
    ax.set_xlabel(
        "Number of survey respondents (percentage)",
        fontsize=16,
        color=MID_GREY,
        labelpad=8,
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", labelsize=16, color=DIVIDER)

    ax.set_yticks([])  # no y-ticks — labels drawn manually above
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(DIVIDER)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.02)  # minimal; bbox_inches handles the rest
    fig.savefig(f"{save_path}.png", dpi=180, bbox_inches="tight", facecolor=BG)
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight", facecolor=BG)
    print(f"Saved {save_path}.png/.pdf")
    plt.close(fig)


def plot_geographical_region(region, save_path):
    region = dict(sorted(region.items(), key=lambda x: x[1], reverse=True))
    labels = list(region.keys())
    values = list(region.values())
    total = sum(values)

    COLORS = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
        "#AEC7E8",
        "#FFBB78",
        "#98DF8A",
        "#FF9896",
        "#C5B0D5",
    ]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    ax.set_facecolor("white")

    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=140,
        colors=COLORS[: len(labels)],
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )

    slice_info = []
    for i, (wedge, label, value) in enumerate(zip(wedges, labels, values)):
        pct = value / total * 100
        angle = (wedge.theta2 + wedge.theta1) / 2
        slice_info.append(dict(i=i, label=label, value=value, pct=pct, angle=angle))

    BASE_R = 1.25
    STEP = 0.13
    N_LEVELS = 4

    for idx, s in enumerate(slice_info):
        s["radius"] = BASE_R + (idx % N_LEVELS) * STEP

    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)

    for s in slice_info:
        angle_rad = np.deg2rad(s["angle"])
        r = s["radius"]

        x0 = np.cos(angle_rad) * 1.02
        y0 = np.sin(angle_rad) * 1.02
        x1 = np.cos(angle_rad) * r
        y1 = np.sin(angle_rad) * r

        ha = "left" if x1 >= 0 else "right"

        ax.annotate(
            f"{s['label']}\n{s['value']} ({s['pct']:.2f}%)",
            xy=(x0, y0),
            xytext=(x1, y1),
            ha=ha,
            va="center",
            fontsize=7.5,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8),
            bbox=bbox_props,
        )

    max_r = BASE_R + (N_LEVELS - 1) * STEP + 0.45
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        save_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.05
    )
    plt.close()
    print("Chart saved.")


# ── Run both ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_kaggle_stats(save_path="geographical_outputs/kaggle_stats")
    plot_survey_overview(save_path="geographical_outputs/survey_overview")
    plot_geographical_region(
        region=region, save_path="geographical_outputs/geographical_piechart.png"
    )
