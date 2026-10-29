"""Examples of the logo, comparison, tree and score distribution plots."""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import universalmotif as um
from matplotlib.transforms import blended_transform_factory

from . import Example

_DNA = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}


@cache
def _motifs():
    return um.example_motifs()


def _by_name(*names):
    found = {m.name: m for m in _motifs()}
    return [found[n] for n in names]


@cache
def _targets():
    return um.example_sequences("targets")


@cache
def _cwms():
    """Contribution weight matrices made from a few example motifs, with the
    letters a column favours above zero and the others below it."""
    out = []
    for m in _by_name("AP-1", "W-box", "GATA"):
        m = um.normalize(m)
        ppm = um.convert_type(m, "PPM").matrix
        ic = um.convert_type(m, "ICM").matrix.sum(axis=0)
        # Letters below background get small negative contributions.
        cwm = (ppm - 0.25) * ic
        cwm[cwm < 0] *= 0.3
        out.append(um.create_motif(cwm, alphabet="DNA", type="CWM", name=f"{m.name} CWM"))
    return out


@cache
def _gapped():
    sites = ["AGGTCAAGGTCA", "AGGTCAAGTTCA", "GGGTCAAGGTCA", "AGGTGAAGGTCA", "AGTTCAAGGTCA",
             "AGGTCAGGGTCA", "AGGTCAAGGTGA", "TGGTCAAGGTCA", "AGGTCAAGGACA", "AGGGCAAGGTCA"]
    dr = um.create_motif(sites, name="DR", pseudocount=1)
    return um.add_gap(dr, gaploc=6, mingap=1, maxgap=5)


@cache
def _protein():
    sites = ["LDEIKR", "LNQLKR", "LDELRK", "LNQIKR", "LDEIKK", "LNELRR", "LDQIKR", "LNEVKR"]
    return um.create_motif(sites, alphabet="AA", name="leucine zipper", pseudocount=1)


# view_motifs ---------------------------------------------------------------


def _view_aligned():
    return um.view_motifs(_by_name("AP-1", "CRE", "G-box"))


def _view_types():
    wbox = _by_name("W-box")[0]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8))
    for ax, use_type in zip(axes, ["ICM", "PPM", "PWM"]):
        um.view_motifs(wbox, use_type=use_type, ax=ax)
        ax.set_title(f'use_type="{use_type}"')
    fig.tight_layout()
    return fig


def _view_cwm():
    return um.view_motifs(_cwms(), use_type="CWM", rc=False, align=False, names_pos="right",
                          show_positions=False, figsize=(6.5, 5))


# view_logo -----------------------------------------------------------------


def _attribution():
    seq = "CTTATGACTCATCAGCATGTTGACCGATCA"
    rng = np.random.default_rng(3)
    score = rng.normal(0, 0.08, len(seq))
    score[4:11] += [0.4, 0.9, 1.1, 0.8, 1.0, 0.9, 0.5]
    score[19:23] -= [0.3, 0.6, 0.5, 0.25]
    onehot = np.array([[c == a for c in seq] for a in "ACGT"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axhline(0, color="#BFBFBF", linewidth=0.5, zorder=0)
    um.view_logo(onehot * score, letters=list("ACGT"), colour_scheme=_DNA, ax=ax)
    return fig


def _protein_logo():
    chemistry = {"DE": "#D62839", "KRH": "#255C99", "NQST": "#9B5DE5", "GP": "#F7B32B",
                 "AVLIMFWYC": "#333333"}
    colours = {aa: colour for group, colour in chemistry.items() for aa in group}
    ppm = um.convert_type(_protein(), "PPM")
    fig, ax = plt.subplots(figsize=(4, 2.2))
    um.view_logo(ppm, colour_scheme=colours, sort_positions=True, ax=ax)
    return fig


def _multichar():
    heights = np.array([[1.2, 0.1, 0.3, 0.9, 0.05],
                        [0.2, 1.4, -0.4, 0.2, 0.8],
                        [0.0, 0.3, 0.6, -0.5, 0.3]])
    fig, ax = plt.subplots(figsize=(4, 2.4))
    um.view_logo(heights, letters=["CG", "TA", "A"], colour_scheme=_DNA, ax=ax)
    return fig


# draw_logo -----------------------------------------------------------------


def _bar_labels():
    motifs = sorted(_motifs()[:8], key=lambda m: m.icscore)
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = range(len(motifs))
    ax.barh(rows, [m.icscore for m in motifs], height=0.6, color="#255C99")
    ax.set_yticks(rows, [m.name for m in motifs])
    ax.set_xlabel("Information content (bits)")
    ax.spines[["top", "right"]].set_visible(False)
    beside = blended_transform_factory(ax.transAxes, ax.transData)
    # One width per position, so that the logos end just left of the labels.
    for row, m in enumerate(motifs):
        width = 0.025 * m.width
        um.draw_logo(ax, m, x=-0.2 - width, y=row - 0.4, width=width, height=0.8,
                     transform=beside)
    return fig


def _scatter():
    motifs = _motifs()[:6]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [m.nsites for m in motifs]
    ys = [m.icscore for m in motifs]
    ax.scatter(xs, ys, color="#7F7F7F")
    for m, x, y in zip(motifs, xs, ys):
        um.draw_logo(ax, m, x=x - 0.9 * m.width, y=y + 0.1, width=1.8 * m.width, height=0.5)
    ax.set_xlabel("Number of sites")
    ax.set_ylabel("Information content (bits)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _track():
    wbox = _by_name("W-box")[0]
    best = um.scan_sequences(wbox, _targets(), nthreads=1).sort("score", descending=True).row(0, named=True)
    seq = _targets()[best["sequence"]]
    a = best["start"] - 20
    window = seq[a - 1:a + 49]
    # Every site of the window, keeping the better strand at each start.
    hits = um.scan_sequences(wbox, {"window": window}, pvalue=0.999, nthreads=1)
    track = hits.group_by("start").agg(pl.col("score").max()).sort("start")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axvspan(best["start"] - 0.5, best["end"] + 0.5, color="#E8E8E8", zorder=0)
    # Each score is drawn at the middle of its site.
    middle = track["start"] + a - 1 + (wbox.width - 1) / 2
    ax.plot(middle, track["score"], color="#255C99", marker="o", markersize=3)
    ax.set_xlabel(f"Position in {best['sequence']}")
    ax.set_ylabel("W-box score")
    ax.spines[["top", "right"]].set_visible(False)
    top = float(track["score"].max())
    um.draw_logo(ax, wbox, x=best["start"] - 0.5, y=top + 2, width=wbox.width, height=12,
                 rc=best["strand"] == "-")
    return fig


# logo_svg ------------------------------------------------------------------


def _svg_default():
    return um.logo_svg(_by_name("AP-1")[0])


def _svg_gapped():
    return um.logo_svg(_gapped(), height=80, position_width=22)


def _svg_cwm():
    return um.logo_svg(_cwms()[0], height=80, position_width=22)


# plot_comparison -----------------------------------------------------------


def _comparison():
    return um.plot_comparison(_motifs())


def _comparison_pvalues():
    return um.plot_comparison(_motifs()[:8], value="pvalue", dendrogram="both", logos=False)


def _comparison_plain():
    return um.plot_comparison(_motifs(), cluster=False, logos=False, annotate=False, cmap="viridis")


# MotifTree -----------------------------------------------------------------


@cache
def _tree():
    return um.motif_tree(_motifs())


def _tree_logos():
    tree = _tree()
    return tree.plot(groups=tree.cutree(k=4), tip_logos=True, legend=False)


def _tree_circular():
    return _tree().plot(layout="circular", groups="family")


def _tree_lengths():
    return _tree().plot(layout="slanted", branch_length=True)


# plot_score_distribution ---------------------------------------------------


def _scores_cutoffs():
    return um.plot_score_distribution(_by_name("AP-1")[0], pvalues=[1e-2, 1e-3, 1e-4], bins=50)


def _scores_hits():
    wbox = _by_name("W-box")[0]
    hits = um.scan_sequences(wbox, _targets(), nthreads=1)
    return um.plot_score_distribution(wbox, pvalues=[1e-3, 1e-4], hits=hits, log_y=True)


def _scores_panels():
    return um.plot_score_distribution(_by_name("TATA-box", "G-box", "GATA", "ETS"), pvalues=1e-4,
                                      log_y=True, bkg={"A": 0.32, "C": 0.18, "G": 0.18, "T": 0.32})


EXAMPLES = {
    "view_motifs": [
        Example("The three bZIP example motifs aligned to CRE, which has the most information "
                "content, with G-box flipped to its reverse complement.", _view_aligned),
        Example("The W-box example motif drawn as information content, probabilities and log-odds "
                "scores, each into one panel of a larger figure through ax.", _view_types),
        Example("Three contribution weight matrices drawn without aligning them, with names on the "
                "right and the negative contributions below zero.", _view_cwm),
    ],
    "view_logo": [
        Example("Attribution scores along a sequence, drawn as the one-hot sequence times the "
                "score of each position.", _attribution),
        Example("The probabilities of a small protein motif, with letters coloured by side-chain "
                "chemistry and those of each position sorted by height.", _protein_logo),
        Example("A matrix with rows named by several characters and negative values, which are "
                "drawn upside down below zero.", _multichar),
    ],
    "draw_logo": [
        Example("Logos of eight example motifs beside the labels of a bar chart, placed in "
                "fractions of the axes across and in rows up.", _bar_labels),
        Example("Logos of six example motifs placed above their points in a scatter plot of "
                "sites against information content.", _scatter),
        Example("Scores of the W-box example motif along an example target sequence, with its logo "
                "drawn over the best site as its reverse complement, since the site is on the minus "
                "strand.", _track),
    ],
    "logo_svg": [
        Example("The SVG logo of the AP-1 example motif at the default size.", _svg_default),
        Example("A gapped motif of two AGGTCA half-sites, with its gap of 1 to 5 positions drawn as "
                "a dotted space.", _svg_gapped),
        Example("A contribution weight matrix, with the letters below zero turned upside down.",
                _svg_cwm),
    ],
    "plot_comparison": [
        Example("All twelve example motifs compared and clustered, with the dendrogram on the "
                "left and logos beside the row labels.", _comparison),
        Example("The P-values of comparing eight example motifs, drawn as -log10 with dendrograms "
                "on both sides.", _comparison_pvalues),
        Example("The example motifs in input order without clustering, logos or values in the "
                "cells.", _comparison_plain),
    ],
    "MotifTree": [
        Example("A tree of the example motifs cut into four groups, which colour the branches, "
                "with each motif's logo at its tip.", _tree_logos),
        Example("The same tree in the circular layout, with branches coloured by the families of "
                "the motifs.", _tree_circular),
        Example("The same tree in the slanted layout, with branch lengths drawn so that each "
                "merge sits at its height.", _tree_lengths),
    ],
    "plot_score_distribution": [
        Example("The score distribution of the AP-1 example motif with the cutoffs of three "
                "P-values.", _scores_cutoffs),
        Example("The W-box example motif on a log scale, with the scores of its hits in the "
                "example target sequences as ticks along the bottom.", _scores_hits),
        Example("Four example motifs as panels, scored against an AT-rich background on a log "
                "scale.", _scores_panels),
    ],
}
