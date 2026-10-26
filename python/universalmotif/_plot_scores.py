"""The null distribution of motif scores, for choosing scan cutoffs."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._analysis import _is_number, _make_unique, motif_pvalue
from ._api import _check_motif, convert_type
from ._core import Motif
from ._plot import _pyplot
from ._utils import motif_range

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_FILL = "#C9CDD3"
_EDGE = "#4A4F57"
_CUTOFF = "#C0504D"
_HITS = "#255C99"


def _motif_bkgs(bkg, n: int) -> list:
    """One background per motif from the forms motif_pvalue() takes."""
    if bkg is None or isinstance(bkg, Mapping):
        return [bkg] * n
    bkg = list(bkg)
    if bkg and not _is_number(bkg[0]):
        if len(bkg) == 1:
            return bkg * n
        if len(bkg) != n:
            raise ValueError("give one background for all motifs or one per motif")
        return bkg
    return [bkg] * n


def _score_max(motif: Motif, bkg) -> float:
    """The highest score the motif can give against ``bkg``. Against another
    background the motif is scored as its probabilities over that
    background, as motif_pvalue() does."""
    if bkg is None:
        return motif_range(motif)[1]
    ppm = convert_type(motif, "PPM") if motif.type not in ("PPM", "CWM") else motif
    return motif_range(ppm.replace(bkg=bkg))[1]


def _score_distribution(motif: Motif, bkg, bins: int, lo: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Bin edges over the motif's scores, the exact probability of a random
    site scoring within each bin, and the highest score.

    Scores lie on a grid of steps of 0.001, so the edges are put on that
    grid, each bin holding the same number of grid points, and the mass of a
    bin is the difference between the P-values of its two edges. The first
    edge has a P-value of 1 and the last one, past the highest score, of 0,
    so the masses add up to 1."""
    lo_g = round(lo * 1000)
    hi_g = max(round(_score_max(motif, bkg) * 1000), lo_g)
    npts = hi_g - lo_g + 1
    step = math.ceil(npts / bins)
    nbins = math.ceil(npts / step)
    grid = lo_g + step * np.arange(nbins + 1)
    grid[-1] = hi_g + 1
    tail = np.empty(nbins + 1)
    tail[0], tail[-1] = 1.0, 0.0
    if nbins > 1:
        with warnings.catch_warnings():
            # Against another background the highest score can fall a step
            # short of the estimate above; such edges correctly get 0.
            warnings.filterwarnings("ignore", message="scores above the highest possible score")
            tail[1:-1] = motif_pvalue(motif, score=(grid[1:-1] / 1000).tolist(), bkg_probs=bkg)
    mass = np.clip(-np.diff(tail), 0.0, None)
    return grid / 1000, mass, hi_g / 1000


def _hit_scores(hits, names: list[str]) -> list[np.ndarray | None]:
    """The hit scores for each motif, from a hit table, a mapping from motif
    names to scores, or a list of scores for a single motif."""
    if hits is None:
        return [None] * len(names)
    if isinstance(hits, pl.DataFrame):
        if "score" not in hits.columns:
            raise ValueError("hits has no score column; was it made by scan_sequences()?")
        if "motif" not in hits.columns:
            if len(names) > 1:
                raise ValueError("hits needs a motif column to be split between several motifs")
            return [hits["score"].drop_nulls().to_numpy().astype(float)]
        return [hits.filter(pl.col("motif") == name)["score"].drop_nulls().to_numpy().astype(float)
                for name in names]
    if isinstance(hits, Mapping):
        unknown = set(map(str, hits)) - set(names)
        if unknown:
            raise ValueError(f"hits names motifs that are not plotted: {', '.join(sorted(unknown)[:5])}")
        return [np.asarray(hits[name], dtype=float) if name in hits else None for name in names]
    if len(names) > 1:
        raise ValueError("give hits for several motifs as a table with a motif column or a "
                         "mapping from motif names to scores")
    return [np.asarray(list(hits), dtype=float)]


def _format_p(p: float) -> str:
    if p >= 1e-3:
        return f"{p:g}"
    exp = math.floor(math.log10(p))
    mant = round(p / 10**exp, 2)
    if mant >= 10:
        mant, exp = mant / 10, exp + 1
    return f"{mant:g}e{exp}" if mant != 1 else f"1e{exp}"


def _draw_panel(ax: Axes, motif: Motif, name: str, bkg, cutoffs: list[float], bins: int,
                log_y: bool, hits: np.ndarray | None) -> None:
    from matplotlib import patheffects
    from matplotlib.transforms import blended_transform_factory

    scores = np.asarray(motif_pvalue(motif, pvalue=[1.0, *cutoffs], bkg_probs=bkg), dtype=float)
    edges, mass, top = _score_distribution(motif, bkg, bins, float(scores[0]))
    if log_y:
        positive = mass[mass > 0]
        floor = float(positive.min()) / 3 if positive.size else 1e-300
        ax.set_yscale("log")
        ax.stairs(mass, edges, fill=True, baseline=floor, color=_FILL, linewidth=0)
        ax.stairs(mass, edges, baseline=None, color=_EDGE, linewidth=0.8)
        ax.set_ylim(floor, float(mass.max()) * 4)
    else:
        ax.stairs(mass, edges, fill=True, color=_FILL, linewidth=0)
        ax.stairs(mass, edges, baseline=None, color=_EDGE, linewidth=0.8)
        ax.set_ylim(0, float(mass.max()) * 1.32)
    ax.set_xlim(float(edges[0]), float(edges[-1]))

    blend = blended_transform_factory(ax.transData, ax.transAxes)
    if hits is not None and hits.size:
        from matplotlib.lines import Line2D

        ax.vlines(hits, 0, 0.06, transform=blend, color=_HITS, alpha=max(0.15, min(1.0, 20 / hits.size)),
                  linewidth=1.0)
        key = Line2D([], [], color=_HITS, linewidth=1.5, label=f"hits (n = {hits.size})")
        ax.legend(handles=[key], frameon=False, loc="upper left", fontsize=8, handlelength=1.0)

    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    unreachable = [_format_p(p) for p, s in zip(cutoffs, scores[1:]) if not np.isfinite(s)]
    for p, s in zip(cutoffs, scores[1:]):
        if not np.isfinite(s):
            continue
        ax.axvline(s, color=_CUTOFF, linewidth=1.1)
        pct = f" ({100 * s / top:.0f}%)" if top > 0 else ""
        ax.annotate(f"P = {_format_p(p)}, score {s:.2f}{pct}", xy=(s, 1), xycoords=blend,
                    xytext=(-4, -4), textcoords="offset points", rotation=90, ha="right", va="top",
                    fontsize=7.5, color="#333333", path_effects=halo, zorder=5,
                    # Close cutoffs put the next line behind a label.
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none",
                          "alpha": 0.85})
    if unreachable:
        warnings.warn(f"{name} cannot reach a P-value of {', '.join(unreachable)}, so no cutoff is "
                      f"drawn for it; the highest score is {top:.3f}", stacklevel=3)

    if top > 0:
        sec = ax.secondary_xaxis("top", functions=(lambda x: 100 * x / top, lambda q: q * top / 100))
        sec.set_xlabel("% of maximum score", fontsize=9)
        sec.tick_params(labelsize=8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability" + (" (log10)" if log_y else ""))
    ax.spines["right"].set_visible(False)


def plot_score_distribution(
    motifs: Motif | Iterable[Motif],
    pvalues: float | Sequence[float] | None = 1e-4,
    bkg: Sequence[float] | Mapping[str, float] | Sequence[object] | None = None,
    log_y: bool = False,
    hits: pl.DataFrame | Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    bins: int = 100,
    ncol: int = 2,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw the distribution of a motif's scores in random sequences, with
    the score cutoffs that P-values give, to help choose a cutoff for
    [`scan_sequences()`][universalmotif.scan_sequences].

    The distribution is the exact one that
    [`motif_pvalue()`][universalmotif.motif_pvalue] works from, gathered
    into ``bins`` bins of equal width between the lowest and highest
    possible scores, so that the height of each bin is the probability that
    a random site scores within it. Each cutoff in ``pvalues`` is drawn as a
    vertical line at the lowest score with at most that P-value, labelled
    with the P-value, the score and the score as a percentage of the
    maximum, which the top axis also shows. Several motifs are drawn as
    panels.

    Needs the ``plot`` extra.

    Parameters
    ----------
    motifs
        A motif or a list of motifs.
    pvalues
        One or several P-value cutoffs to mark, or ``None`` for none. A
        motif too short or too degenerate to reach a cutoff gets a warning
        instead of a line.
    bkg
        Background frequencies, in any of the forms
        [`motif_pvalue()`][universalmotif.motif_pvalue] takes as
        ``bkg_probs``. Each motif's own background by default.
    log_y
        Draw probabilities on a log scale, which shows the upper tail where
        cutoffs usually fall.
    hits
        Scores of real sites to mark as ticks along the bottom: a hit table
        from [`scan_sequences()`][universalmotif.scan_sequences], a mapping
        from motif names to scores, or for a single motif a list of scores.
    bins, ncol
        Number of bins, and of panels per row for several motifs.
    ax
        Draw a single motif on these axes instead of in a new figure.
        ``figsize`` is then ignored.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import universalmotif as um
    >>> wbox = um.example_motifs()[5]
    >>> hits = um.scan_sequences(wbox, um.example_sequences("targets"), nthreads=1)
    >>> fig = um.plot_score_distribution(wbox, pvalues=[1e-3, 1e-4], hits=hits, log_y=True)
    >>> type(fig).__name__
    'Figure'
    >>> plt.close(fig)

    Drawn into a panel of a larger figure:

    >>> fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    >>> ax = um.plot_score_distribution(wbox, ax=axes[1])
    >>> ax is axes[1]
    True
    >>> plt.close(fig)
    """
    ms = [_check_motif(motifs)] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("no motifs given")
    if ax is not None and len(ms) > 1:
        raise ValueError("ax can only be used with a single motif")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive whole number")
    if pvalues is None:
        cutoffs = []
    elif _is_number(pvalues):
        cutoffs = [float(pvalues)]
    else:
        cutoffs = [float(p) for p in pvalues]  # type: ignore[union-attr]
    if any(not 0 < p <= 1 for p in cutoffs):
        raise ValueError("pvalues must lie above 0 and at most 1")
    names = _make_unique([m.name for m in ms])
    bkgs = _motif_bkgs(bkg, len(ms))
    scores = _hit_scores(hits, names)
    if ax is not None:
        _draw_panel(ax, ms[0], names[0], bkgs[0], cutoffs, bins, log_y, scores[0])
        return ax
    plt = _pyplot()
    n = len(ms)
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (5.0 * ncol, 3.4 * nrow), squeeze=False)
    for k, (m, name, b, h) in enumerate(zip(ms, names, bkgs, scores)):
        _draw_panel(axes[k // ncol, k % ncol], m, name, b, cutoffs, bins, log_y, h)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    fig.tight_layout()
    return fig


__all__ = ["plot_score_distribution"]
