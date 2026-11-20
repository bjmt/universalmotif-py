"""Draw the scaling of TF-MoDISco and contribution_motifs() from
results/scaling.tsv: time against regions, time against the seqlets each
method used (with each method's power law carried on to 1,000,000), and
peak memory against regions.

    python plot_scaling.py OUTPUT.svg
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
COLOURS = {"TF-MoDISco": "#4D4D4D", "contribution_motifs(), every seqlet": "#255C99",
           "contribution_motifs(), default cap": "#D95F02"}

d = pl.read_csv(HERE / "results" / "scaling.tsv", separator="\t")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
for name, colour in COLOURS.items():
    r = d.filter(pl.col("method") == name).sort("regions")
    kw = dict(color=colour, marker="o", markersize=4, linewidth=1.5)
    axes[0].plot(r["regions"], r["seconds"] / 60, label=name, **kw)
    axes[2].plot(r["regions"], r["memory_gb"], **kw)
    if name != "contribution_motifs(), default cap":
        n, t = r["seqlets"].to_numpy().astype(float), r["seconds"].to_numpy().astype(float)
        axes[1].plot(n, t / 60, **kw)
        b, a = np.polyfit(np.log(n[-3:]), np.log(t[-3:]), 1)
        x = np.geomspace(n[-1], 1e6, 20)
        axes[1].plot(x, np.exp(a) * x ** b / 60, color=colour, linestyle="--", linewidth=1)
        axes[1].annotate(f"{np.exp(a) * 1e6 ** b / 60:.0f} min", (1e6, np.exp(a) * 1e6 ** b / 60),
                         textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=8, color=colour)
for ax, xlabel, ylabel, title in [
    (axes[0], "Regions", "Minutes", "Time"),
    (axes[1], "Seqlets clustered or searched", "Minutes", "Time by seqlets used"),
    (axes[2], "Regions", "GB", "Peak memory"),
]:
    ax.set_xscale("log")
    if ax is not axes[2]:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False, fontsize=9)
fig.savefig(sys.argv[1], facecolor="white")
