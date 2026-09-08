"""Publication figures from recorded results and supplied spatial/event data.

Run from any directory: python paper/make_figures.py
Requires matplotlib and numpy. No training, prediction reconstruction or network
access occurs. See data/README.md for provenance and known evidence limits.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#222222", "#646464", "#dfdfdf"
WIDTH = 3.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.025, "figure.dpi": 160,
})


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def results():
    rows = read_csv(HERE / "data/model2_recorded_results.csv")
    for row in rows:
        for key in ("ap", "ece", "brier", "far", "event_detection",
                    "params_million_per_member"):
            row[key] = float(row[key]) if row[key] else None
    return {row["preset"]: row for row in rows}


def finish(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight",
                metadata={"Title": name, "Creator": "paper/make_figures.py"})
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.5)


def fig_study_area():
    nodes = read_csv(ROOT / "data/processed/nodes.csv")
    edges = read_csv(ROOT / "data/processed/edges.csv")
    outline = json.loads((HERE / "data/sri_lanka_outline.geojson").read_text())
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6), layout="constrained")
    geom = outline["features"][0]["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        ax.add_patch(Polygon(poly[0], facecolor="#f2f2ef", edgecolor="#92928b", lw=0.6))
    pos = {n["node_id"]: (float(n["lon"]), float(n["lat"])) for n in nodes}
    for e in edges:
        if e["edge_type"] == "flow":
            ax.annotate("", xy=pos[e["dst"]], xytext=pos[e["src"]],
                        arrowprops={"arrowstyle": "-|>", "color": "#747474",
                                    "lw": 0.65, "mutation_scale": 5}, zorder=2)
    for zone, col, marker in [("wet", BLUE, "o"), ("intermediate", GREEN, "s"),
                               ("dry", ORANGE, "^")]:
        group = [n for n in nodes if n["zone"] == zone]
        ax.scatter([float(n["lon"]) for n in group], [float(n["lat"]) for n in group],
                   s=19, color=col, marker=marker, edgecolor="white", lw=0.4,
                   label=f"{zone.capitalize()} ({len(group)})", zorder=3)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    ax.annotate("N", xy=(81.8, 9.35), xytext=(81.8, 8.97), ha="center", fontsize=8,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.8})
    ax.set(xlim=(79.45, 82.05), ylim=(5.75, 9.95), xlabel="Longitude (degrees E)",
           ylabel="Latitude (degrees N)")
    ax.set_xticks([79.5, 80.5, 81.5])
    ax.set_aspect(1 / np.cos(np.deg2rad(8)))
    ax.grid(color=GRID, lw=0.4, zorder=0)
    finish(fig, "fig_study_area")


def fig_episode_profile():
    events = read_csv(ROOT / "data/processed/events.csv")
    kept = [e for e in events if "2003-01-01" <= e["start"] <= "2024-12-31"]
    counts = Counter(int(e["start"][:4]) for e in kept)
    years = list(range(2003, 2025))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1), layout="constrained")
    ax.bar(years, [counts[y] for y in years], width=0.78, color=BLUE, zorder=3)
    for boundary in [2017.5, 2020.5]:
        ax.axvline(boundary, color=MUTED, lw=0.8, ls="--")
    top = max(counts.values()) * 1.23
    for x, label in [(2010, "Training"), (2019, "Val."), (2022.5, "Test")]:
        ax.text(x, top * 0.96, label, ha="center", va="top", fontsize=7)
    ax.set(xlim=(2002.3, 2024.7), ylim=(0, top), ylabel="Proxy episodes", xlabel="Onset year")
    ax.set_xticks([2003, 2007, 2011, 2015, 2019, 2024])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.5)
    finish(fig, "fig_episode_profile")
    # A reproducible count audit, separate from unavailable model eligibility counts.
    audit = {"catalogue_entries": len(events), "entries_starting_2003_2024": len(kept),
             "excluded_after_2024": len(events) - len(kept),
             "annual_counts": {str(y): counts[y] for y in years}}
    (HERE / "data/catalogue_counts.json").write_text(json.dumps(audit, indent=2) + "\n")


def fig_ladder(data):
    keys = ["N0", "N1", "N2", "N3", "N4", "N5", "N5_bce"]
    labels = ["N0  Linear", "N1  + PLR", "N2  + feature attention",
              "N3  + FiLM", "N4  Focal/confidence", "N5  Focal ensemble",
              "N5_bce  BCE ensemble"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    fig.subplots_adjust(left=0.43, right=0.96, bottom=0.18, top=0.86)
    for i, key in enumerate(keys):
        r = data[key]
        color, marker = (BLUE, "o") if r["objective"] == "bce" else (ORANGE, "s")
        ax.plot(r["ap"], i, marker=marker, color=color, ms=5)
        ax.annotate(f"{r['ap']:.4f}", (r["ap"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(range(len(keys)), labels)
    ax.set(xlim=(0.58, 0.93), ylim=(6.6, -0.6), xlabel="Test average precision")
    ax.set_xticks([0.6, 0.7, 0.8, 0.9])
    clean(ax)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="BCE"),
                        Line2D([], [], marker="s", color=ORANGE, ls="", label="Focal/confidence")],
               loc="upper center", bbox_to_anchor=(0.55, 1), ncol=2, frameon=False)
    finish(fig, "fig_ladder")


def fig_capacity(data):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3), layout="constrained")
    specs = [("N3", "N3 (1 seed)", BLUE, "o", 8, -16),
             ("N5_bce", "N5_bce (5 seeds)", BLUE, "D", 9, 10),
             ("N5", "N5 (5 seeds)", ORANGE, "s", 9, -2),
             ("N6_gated", "N6-gat (5 seeds)", ORANGE, "^", -8, -16)]
    for key, label, c, marker, dx, dy in specs:
        r = data[key]
        x, y = r["params_million_per_member"], r["ap"]
        ax.scatter(x, y, c=c, marker=marker, s=30, zorder=3)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.set(xlim=(0.5, 20), ylim=(0.765, 0.86), xlabel="Reported parameters / member (millions; log scale)",
           ylabel="Test average precision")
    ax.set_xticks([0.7, 1, 3, 12], ["0.7", "1", "3", "12"])
    ax.minorticks_off()
    clean(ax)
    finish(fig, "fig_capacity")


def fig_operating_points(data):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.35), layout="constrained")
    specs = [("N0", "N0", 6, -10), ("N2", "N2", -5, 7),
             ("N3", "N3", 8, -12), ("N5_bce", "N5_bce", 8, 8),
             ("N6_gated", "N6-gat", -8, 7)]
    for key, label, dx, dy in specs:
        r = data[key]
        color, marker = (BLUE, "o") if r["objective"] == "bce" else (ORANGE, "s")
        ax.scatter(r["far"], r["event_detection"], color=color, marker=marker, s=28, zorder=3)
        ax.annotate(label, (r["far"], r["event_detection"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha="left" if dx > 0 else "right")
    ax.set(xlim=(0, 0.5), ylim=(0, 0.52), xlabel="Node-day false-alarm ratio",
           ylabel="Seven-day pre-onset detection")
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    clean(ax)
    ax.grid(axis="y", color=GRID, lw=0.5)
    finish(fig, "fig_operating_points")


if __name__ == "__main__":
    recorded = results()
    fig_study_area()
    fig_episode_profile()
    fig_ladder(recorded)
    fig_capacity(recorded)
    fig_operating_points(recorded)
    print(f"Generated five vector figures in {OUT}")
