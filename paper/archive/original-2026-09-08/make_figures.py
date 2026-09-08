"""Generate the paper's figures as vector PDFs from the recorded run results.

    python paper/make_figures.py

Numbers are hard-coded from the 2026-08-02 (model 1 + baselines) and
2026-08-10/11 (model 2) Kaggle runs, which is deliberate: the figures must be
reproducible from the paper's own repository even when `runs/` is not present.
Every value here also appears in `docs/RESULTS.md`; if one changes, change both.

Figures that need the saved predictions (reliability diagram, PR curves) or GIS
layers (the study-area map) are not generated here — the LaTeX source carries
sized placeholders for them.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito subset, validated colourblind-safe against a light surface
# (worst adjacent CVD dE 11.0 deutan, normal-vision floor 24.2).
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL = 3.4          # IEEE single-column width, inches


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# --------------------------------------------------------------- Fig: ladder

def fig_ladder():
    """PR-AUC up the MMF-Net ablation ladder, against the two strong baselines.

    Bars are coloured by *loss function*, not by rank — that is the attribute
    the figure is about, and it makes the focal-loss regression visible as a
    block rather than as two isolated bars.
    """
    rows = [
        ("N0  linear embeddings",        0.6083, "bce"),
        ("N1  + PLR embeddings",         0.7605, "bce"),
        ("N2  + feature attention",      0.7738, "bce"),
        ("N3  + FiLM terrain",           0.8269, "bce"),
        ("N4  + focal loss",             0.7592, "focal"),
        ("N5  + ensemble (focal)",       0.7846, "focal"),
        ("N5_bce  + ensemble (BCE)",     0.8355, "bce"),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    losses = [r[2] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(COL, 2.7))
    colors = [VERM if l == "focal" else BLUE for l in losses]
    y = range(len(vals))
    ax.barh(list(y), vals, height=0.62, color=colors, zorder=3)

    # Reference lines: the two bars the network has to clear. Labels are set
    # vertically along each line so they cannot collide with one another.
    for x, lab, style in ((0.8164, "discharge percentile", (0, (1, 1.6))),
                          (0.8496, "gradient-boosted trees", (0, (4, 1.8)))):
        ax.axvline(x, color=INK, lw=0.8, ls=style, zorder=4)
        # Top-right is the only region no bar reaches (N0-N2 all end < 0.78).
        ax.text(x - 0.006, len(vals) - 0.42, f"{lab}  {x:.4f}", fontsize=5.8,
                color=INK, rotation=90, va="top", ha="right")

    for i, v in enumerate(vals):
        ax.text(v - 0.012, i, f"{v:.4f}", va="center", ha="right",
                fontsize=6.5, color="white", zorder=5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("PR-AUC (test block, temporal protocol)")
    ax.set_xlim(0.55, 0.92)
    ax.set_ylim(-0.62, len(vals) - 0.3)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=VERM)]
    # Legend above the axes: every in-plot corner is occupied by a bar or a
    # reference label.
    ax.legend(handles, ["binary cross-entropy", "focal $\\times$ confidence"],
              loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.12), ncol=2,
              mode="expand", frameon=False, handlelength=1.1,
              borderaxespad=0)

    fig.savefig(os.path.join(OUT, "fig_ladder.pdf"))
    plt.close(fig)


# -------------------------------------------------------------- Fig: leakage

def fig_leakage():
    """RQ2. The same model under two splitting protocols.

    A slope chart, because the finding *is* the divergence: one metric rises
    while the other collapses, and a grouped bar chart hides that.
    """
    fig, ax = plt.subplots(figsize=(COL, 2.2))

    series = [
        ("PR-AUC",                       0.704, 0.905, BLUE, "o"),
        ("pre-onset event detection",    0.380, 0.089, VERM, "s"),
    ]
    for name, a, b, c, marker in series:
        ax.plot([0, 1], [a, b], color=c, lw=1.6, marker=marker, ms=5,
                mec="white", mew=0.8, zorder=3, label=name)
        ax.text(-0.045, a, f"{a:.3f}", ha="right", va="center",
                fontsize=7, color=INK)
        ax.text(1.045, b, f"{b:.3f}", ha="left", va="center",
                fontsize=7, color=INK)

    ax.annotate("", xy=(0.5, 0.86), xytext=(0.5, 0.135),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
    ax.text(0.52, 0.5, "the split changes\nwhich model is selected",
            fontsize=6.5, color=MUTED, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["block temporal split", "random split"])
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)

    fig.savefig(os.path.join(OUT, "fig_leakage.pdf"))
    plt.close(fig)


# ------------------------------------------------------ Fig: operating points

def fig_operating_points():
    """Why `ev.det` must not be read down a results column.

    Each model sits at its own validation-chosen threshold, so the models in a
    table are not at a common operating point. Plotting detection against
    false-alarm ratio makes that explicit.
    """
    # (name, FAR, ev.det, label dx, dy in points, ha) — offsets are hand-set:
    # five of the nine rungs fall inside a 0.04 x 0.03 box, so automatic
    # placement collides no matter the algorithm.
    models = [
        ("N0",     0.412, 0.290,   5,   2, "left"),
        ("N1",     0.325, 0.420,   6,  -1, "left"),
        ("N2",     0.313, 0.423, -10,   5, "right"),
        ("N3",     0.110, 0.197, -14,  -9, "right"),
        ("N4",     0.138, 0.220,   5,   6, "left"),
        ("N5",     0.145, 0.208,   6,  -8, "left"),
        ("N5_bce", 0.120, 0.220,  -6,   7, "right"),
        ("N6_sc",  0.344, 0.397,   6,  -3, "left"),
        ("N6_gat", 0.107, 0.217, -26,   4, "right"),
    ]
    baselines = [
        ("persistence", 0.231, 0.223,  5, -9, "left"),
        ("GBT",         0.080, 0.161,  6, -3, "left"),
        ("climatology", 0.882, 0.515, -6, -9, "right"),
    ]

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.scatter([m[1] for m in models], [m[2] for m in models], s=26,
               color=BLUE, marker="o", ec="white", lw=0.7, zorder=3,
               label="MMF-Net rungs")
    ax.scatter([b[1] for b in baselines], [b[2] for b in baselines], s=30,
               color=ORANGE, marker="^", ec="white", lw=0.7, zorder=3,
               label="baselines")

    for name, far, ev, dx, dy, ha in models + baselines:
        ax.annotate(name, (far, ev), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6, color=INK, ha=ha)

    ax.set_xlabel("false-alarm ratio (own threshold)")
    ax.set_ylabel("pre-onset event detection")
    # Left bound is negative only to give the "N6_gat" label room; ticks still
    # start at 0.0 so the scale reads normally.
    ax.set_xlim(-0.07, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.10, 0.60)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.0)

    fig.savefig(os.path.join(OUT, "fig_operating_points.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- Fig: seed noise

def fig_seed_noise():
    """The top three rungs against the seed-to-seed spread of a single rung.

    N5_bce's five seeds span 0.0481 in best-validation episode PR-AUC. The
    differences being ranked in the results table are smaller than that.
    """
    fig, ax = plt.subplots(figsize=(COL, 1.75))

    gaps = [("N5_bce - N6_gated", 0.0045), ("N5_bce - N3", 0.0086),
            ("N5_bce - N5 (loss)", 0.0509)]
    spread = 0.0481

    ax.axvspan(0, spread, color=GRID, zorder=0)
    ax.text(spread - 0.002, 2.72,
            f"seed-noise band, one rung, 5 seeds ({spread:.4f})",
            fontsize=6.2, color=MUTED, ha="right", va="top")

    y = range(len(gaps))
    # Colour encodes whether the difference clears the seed-noise band — the
    # figure's actual claim — not the size ranking of the bars.
    ax.barh(list(y), [g[1] for g in gaps], height=0.5,
            color=[MUTED, MUTED, GREEN], zorder=3)
    for i, (_, v) in enumerate(gaps):
        ax.text(v + 0.0015, i, f"{v:.4f}", va="center", fontsize=6.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["clears seed noise", "inside seed noise"],
              loc="lower right", frameon=False, handlelength=1.0, fontsize=6.2)

    ax.set_yticks(list(y))
    ax.set_yticklabels([g[0] for g in gaps])
    ax.set_xlabel("difference in test PR-AUC")
    ax.set_xlim(0, 0.062)
    ax.xaxis.grid(True, color="#eeeeee", lw=0.5, zorder=1)
    ax.set_axisbelow(False)
    _despine(ax)

    fig.savefig(os.path.join(OUT, "fig_seed_noise.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder()
    fig_leakage()
    fig_operating_points()
    fig_seed_noise()
    print(f"wrote 4 figures to {OUT}")
