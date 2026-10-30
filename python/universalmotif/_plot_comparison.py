"""Heatmaps of motif comparisons with matplotlib."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique
from ._api import _check_motif
from ._compare import MotifTree, _tree
from ._core import Motif
from ._plot import _pyplot, draw_logo
from ._threads import _nthreads

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_VALUE_LABELS = {"score": "Similarity score", "pvalue": "-log10(P-value)", "qvalue": "-log10(q-value)"}
_MAX_LOGOS = 60
_MAX_ANNOTATED = 20


def _comparison_input(motifs, matrix, labels, value, min_overlap, rc, nthreads):
    """The motifs, the matrix of values as given or computed, the score
    matrix to cluster on when there is one, and the labels."""
    ms = None
    if motifs is not None:
        ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    columns = None
    if matrix is not None:
        if isinstance(matrix, pl.DataFrame):
            columns = list(matrix.columns)
            matrix = matrix.to_numpy()
        vals = np.asarray(matrix, dtype=float)
        if vals.ndim != 2 or vals.shape[0] != vals.shape[1]:
            raise ValueError("matrix must be square")
        if ms is not None and len(ms) != vals.shape[0]:
            raise ValueError("matrix must have one row and column per motif")
        scores = vals if value == "score" else None
    else:
        if ms is None:
            raise ValueError("give motifs, a matrix or both")
        if min_overlap < 1:
            raise ValueError("min_overlap must be at least 1")
        if len(ms) < 2:
            raise ValueError("at least two motifs are needed for a comparison")
        d = _core._compare_matrix(ms, min_overlap, rc, "empirical", None, value != "score", nthreads,
                                  False)
        vals = np.array(d[value], dtype=float)
        scores = np.array(d["score"], dtype=float)
    n = vals.shape[0]
    if n < 2:
        raise ValueError("at least two motifs are needed for a comparison")

    if isinstance(labels, str):
        if ms is not None:
            labs = ["" if getattr(m, labels) is None else str(getattr(m, labels)) for m in ms]
        else:
            labs = columns if columns is not None else [str(i + 1) for i in range(n)]
        if labels == "name" or ms is None:
            labs = _make_unique(labs)
    else:
        labs = [str(lab) for lab in labels]
        if len(labs) != n:
            raise ValueError("give one label per motif")
    return ms, vals, scores, labs


def _distances(vals: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
    """Distances to cluster on: (1 - score) / 2 for scores, and otherwise
    one minus the mean -log10 value of both directions over its largest."""
    if scores is not None:
        sym = (scores + scores.T) / 2
        dist = (1 - sym) / 2
        dist[~np.isfinite(dist)] = 1.0
    else:
        logs = _neg_log10(vals)
        sym = (logs + logs.T) / 2
        np.fill_diagonal(sym, np.nan)
        finite = sym[np.isfinite(sym)]
        top = float(finite.max()) if finite.size else 0.0
        dist = np.clip(1 - sym / top, 0.0, 1.0) if top > 0 else np.ones_like(sym)
        dist[~np.isfinite(dist)] = 1.0
    np.fill_diagonal(dist, 0.0)
    return dist


def _neg_log10(p: np.ndarray) -> np.ndarray:
    # Adding zero turns the -0.0 of a P-value of 1 into 0.0.
    with np.errstate(invalid="ignore"):
        return 0.0 - np.log10(np.clip(p, np.finfo(float).tiny, None))


def _dendrogram_segments(tree: MotifTree, pos: np.ndarray) -> tuple[list, float]:
    """Line segments of a dendrogram as (position, height) pairs, with the
    leaves at height 0, and the largest height."""
    n = len(tree.labels)
    node_pos = list(pos.astype(float))
    node_h = [0.0] * n
    segs = []
    for (a, b), h in zip(np.asarray(tree.merge, dtype=int), np.asarray(tree.height, dtype=float)):
        ia = -a - 1 if a < 0 else n + a - 1
        ib = -b - 1 if b < 0 else n + b - 1
        pa, pb = node_pos[ia], node_pos[ib]
        segs.append([(pa, node_h[ia]), (pa, h), (pb, h), (pb, node_h[ib])])
        node_pos.append((pa + pb) / 2)
        node_h.append(float(h))
    top = max([h for h in node_h if math.isfinite(h)] + [0.0])
    return segs, top or 1.0


def _text_inches(fig, texts: Sequence[str], size: float) -> float:
    """The width of the widest text, in inches."""
    if not texts:
        return 0.0
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for s in texts:
        t = fig.text(0, 0, s, fontsize=size)
        widest = max(widest, t.get_window_extent(renderer).width)
        t.remove()
    return widest / fig.dpi


def plot_comparison(
    motifs: Motif | Sequence[Motif] | None = None,
    matrix: Any = None,
    labels: str | Sequence[str] = "name",
    value: str = "score",
    min_overlap: int = 6,
    rc: bool = True,
    cluster: bool | MotifTree = True,
    method: str = "complete",
    dendrogram: str = "left",
    cmap: Any = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool | None = None,
    logos: bool | None = None,
    text_size: float = 10,
    figsize: tuple[float, float] | None = None,
    nthreads: int | None = None,
) -> Figure:
    """Draw motif similarities as a clustered heatmap.

    Motifs are compared all against all with
    [`compare_motifs()`][universalmotif.compare_motifs], or the comparison
    is given as ``matrix``, a square array or DataFrame such as
    ``compare_motifs()`` returns without ``compare_to``. Rows are queries and
    columns targets, both in the order of a hierarchical clustering drawn as
    a dendrogram beside the heatmap. Scores are drawn as they are, and P- and
    q-values as their -log10, with the diagonal left blank since it compares
    each motif with itself.

    Parameters
    ----------
    motifs
        Motifs to compare, or the motifs ``matrix`` was made from, which
        give the labels and logos.
    matrix
        A precomputed square matrix of the kind chosen by ``value``. A
        DataFrame's column names become the labels when no motifs are given.
        Scores are clustered as correlations, so the mean distances of
        ``compare_motifs(method="EUCL")`` need a tree from
        [`motif_tree()`][universalmotif.motif_tree] as ``cluster``, or no
        clustering.
    labels
        A motif attribute to label rows and columns with, such as
        ``"name"`` or ``"altname"``, or one label per motif.
    value
        What is drawn: ``"score"``, the mean column correlation, or
        ``"pvalue"`` or ``"qvalue"``.
    min_overlap, rc, nthreads
        Passed to ``compare_motifs()``, with ``nthreads`` by default the
        package setting from [`set_nthreads()`][universalmotif.set_nthreads].
        ``min_overlap`` follows the default of 6 of
        [`motif_tree()`][universalmotif.motif_tree], so the scores can differ
        slightly from those of ``compare_motifs()`` with its default of 5.
    cluster, method
        Order the motifs by clustering them with ``method``, as in
        [`motif_tree()`][universalmotif.motif_tree], or keep them in input
        order with ``False``. A tree from ``motif_tree()`` can be given
        instead. Motifs are clustered on their scores, with distances of
        ``(1 - score) / 2``, so that the order is that of ``motif_tree()``
        with the same ``min_overlap``. A matrix of P- or q-values given as
        ``matrix`` is clustered on the mean -log10 value of both directions,
        scaled to distances from 0 to 1.
    dendrogram
        Where to draw the tree: ``"left"``, ``"top"``, ``"both"`` or
        ``"none"``.
    cmap, vmin, vmax
        The colour map and its range, which by default spans the values
        drawn, from 0 for P- and q-values.
    annotate
        Write each value in its cell, by default for up to 20 motifs.
    logos
        Draw each motif's logo beside its row label, by default when motifs
        are given and there are up to 60 of them.
    text_size
        The largest font size of labels, which shrink to fit the cells of
        larger heatmaps and are left out once they would be unreadable.
    figsize
        The figure size, by default worked out from the number of motifs.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Its axes are labelled ``"heatmap"``, ``"colour bar"``,
        ``"logos"``, ``"labels"``, ``"dendrogram left"`` and ``"dendrogram
        top"``, as drawn, and can be found by ``ax.get_label()``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> fig = um.plot_comparison(motifs)
    >>> sorted(ax.get_label() for ax in fig.axes)
    ['colour bar', 'dendrogram left', 'heatmap', 'labels', 'logos']
    >>> plt.close(fig)

    A matrix computed beforehand can be drawn too, here as P-values:

    >>> pvalues = um.compare_motifs(motifs[:6], matrix_out="pvalue")
    >>> fig = um.plot_comparison(matrix=pvalues, labels=[m.name for m in motifs[:6]], value="pvalue",
    ...                       dendrogram="both")
    >>> plt.close(fig)
    """
    nthreads = _nthreads(nthreads)
    if value not in _VALUE_LABELS:
        raise ValueError("value must be 'score', 'pvalue' or 'qvalue'")
    if dendrogram not in ("left", "top", "both", "none"):
        raise ValueError("dendrogram must be 'left', 'top', 'both' or 'none'")
    ms, vals, scores, labs = _comparison_input(motifs, matrix, labels, value, min_overlap, rc, nthreads)
    n = len(labs)
    if logos and ms is None:
        raise ValueError("logos need the motifs")

    tree = None
    if isinstance(cluster, MotifTree):
        if len(cluster) != n:
            raise ValueError("the tree must have one tip per motif")
        tree = cluster
    elif cluster:
        dist = _distances(vals, scores)
        tree = _tree(_core._hclust(dist.tolist(), method, labs), labs, method)
    order = list(tree.order) if tree is not None else list(range(n))

    shown = vals if value == "score" else _neg_log10(vals)
    shown = shown[np.ix_(order, order)]
    if value != "score":
        np.fill_diagonal(shown, np.nan)
    finite = shown[np.isfinite(shown)]
    lo = vmin if vmin is not None else (0.0 if value != "score" else
                                        float(finite.min()) if finite.size else 0.0)
    hi = vmax if vmax is not None else (float(finite.max()) if finite.size else 1.0)
    if hi <= lo:
        hi = lo + 1.0

    # Cells shrink as motifs are added, down to a floor, and the labels with
    # them.
    cell = max(min(0.45, 7.0 / n), min(0.15, 12.0 / n))
    label_size = min(text_size, 0.75 * cell * 72)
    show_labels = label_size >= 4
    if annotate is None:
        annotate = n <= _MAX_ANNOTATED
    if logos is None:
        logos = ms is not None and n <= _MAX_LOGOS
    left_tree = tree is not None and dendrogram in ("left", "both")
    top_tree = tree is not None and dendrogram in ("top", "both")

    plt = _pyplot()
    import matplotlib as mpl
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    ordered = [labs[i] for i in order]
    label_w = _text_inches(fig, ordered, label_size) if show_labels else 0.0
    cbar_text = max(_text_inches(fig, [f"{hi:.2g}", f"{lo:.2g}", "0.00"], 0.8 * text_size), 0.3)
    widest_logo = max(ms[i].width for i in order) if logos else 0

    def layout(c: float):
        """Column and row sizes in inches for a cell size of c."""
        side = n * c
        tree_w = min(max(0.15 * side, 0.6), 1.5)
        cols = [("margin", 0.1)]
        if left_tree:
            cols += [("dendrogram left", tree_w), ("gap", 0.04)]
        cols.append(("heatmap", side))
        if logos:
            cols += [("gap", 0.08), ("logos", widest_logo * 0.3 * c)]
        if show_labels:
            cols += [("gap", 0.06), ("labels", label_w)]
        cols += [("gap", 0.25), ("colour bar", 0.12), ("margin", cbar_text + 0.35)]
        rows = [("margin", 0.1)]
        if top_tree:
            rows += [("dendrogram top", tree_w), ("gap", 0.04)]
        rows += [("heatmap", side), ("margin", label_w + 0.15 if show_labels else 0.1)]
        return cols, rows

    cols, rows = layout(cell)
    if figsize is not None:
        # The cells take whatever room the fixed parts leave.
        fixed_w = sum(w for k, w in cols if k not in ("heatmap", "logos"))
        fixed_h = sum(h for k, h in rows if k != "heatmap")
        per_cell_w = n + (widest_logo * 0.3 if logos else 0)
        cell = min((figsize[0] - fixed_w) / per_cell_w, (figsize[1] - fixed_h) / n)
        if cell <= 0:
            raise ValueError("figsize is too small for the labels and dendrograms")
        cols, rows = layout(cell)
        width, height = figsize
    else:
        width, height = sum(w for _, w in cols), sum(h for _, h in rows)
    fig.set_size_inches(width, height)
    # Spare room from a given figsize goes to the right and bottom margins.
    spare_w, spare_h = width - sum(w for _, w in cols), height - sum(h for _, h in rows)
    cols[-1] = ("margin", cols[-1][1] + spare_w)
    rows[-1] = ("margin", rows[-1][1] + spare_h)
    gs = fig.add_gridspec(len(rows), len(cols), width_ratios=[w for _, w in cols],
                          height_ratios=[h for _, h in rows], left=0, right=1, bottom=0, top=1,
                          wspace=0, hspace=0)
    col_of = {k: i for i, (k, _) in enumerate(cols)}
    row_of = {k: i for i, (k, _) in enumerate(rows)}
    hrow = row_of["heatmap"]

    ax = fig.add_subplot(gs[hrow, col_of["heatmap"]], label="heatmap")
    colours = mpl.colormaps.get_cmap(cmap).with_extremes(bad="#E6E6E6")
    edges = np.arange(n + 1) - 0.5
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(shown), cmap=colours, vmin=lo, vmax=hi,
                         edgecolors="white" if n <= 30 else "face", linewidth=0.5 if n <= 30 else 0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=2)
    if show_labels:
        ax.set_xticks(range(n), ordered, rotation=90, fontsize=label_size)
    else:
        ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        digits = "{:.2f}" if value == "score" else "{:.1f}"
        size = min(0.8 * text_size, 0.3 * cell * 72)
        for i in range(n):
            for j in range(n):
                v = shown[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = colours(mesh.norm(v))
                dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
                ax.text(j, i, digits.format(v), ha="center", va="center", fontsize=size,
                        color="white" if dark else "black")

    if left_tree or top_tree:
        pos = np.empty(n)
        pos[order] = np.arange(n)
        segs, top = _dendrogram_segments(tree, pos)
        if left_tree:
            tax = fig.add_subplot(gs[hrow, col_of["dendrogram left"]], label="dendrogram left")
            tax.add_collection(LineCollection([[(h, p) for p, h in s] for s in segs], colors="black",
                                              linewidths=0.8))
            tax.set_xlim(top * 1.02, 0)
            tax.set_ylim(n - 0.5, -0.5)
            tax.set_axis_off()
        if top_tree:
            tax = fig.add_subplot(gs[row_of["dendrogram top"], col_of["heatmap"]], label="dendrogram top")
            tax.add_collection(LineCollection(segs, colors="black", linewidths=0.8))
            tax.set_xlim(-0.5, n - 0.5)
            tax.set_ylim(0, top * 1.02)
            tax.set_axis_off()

    if logos:
        lax = fig.add_subplot(gs[hrow, col_of["logos"]], label="logos")
        # Logos are drawn upright, so this axis counts rows from the bottom.
        for k, i in enumerate(order):
            draw_logo(lax, ms[i], x=0, y=n - 1 - k - 0.4, height=0.8)
        lax.set_xlim(0, widest_logo)
        lax.set_ylim(-0.5, n - 0.5)
        lax.set_axis_off()

    if show_labels:
        nax = fig.add_subplot(gs[hrow, col_of["labels"]], label="labels")
        for k, lab in enumerate(ordered):
            nax.text(0, k, lab, ha="left", va="center", fontsize=label_size)
        nax.set_xlim(0, 1)
        nax.set_ylim(n - 0.5, -0.5)
        nax.set_axis_off()

    # The colour bar is at most 2.5 inches tall, from the top of the heatmap.
    bar = min(1.0, 2.5 / rows[hrow][1])
    cgs = gs[hrow, col_of["colour bar"]].subgridspec(2, 1, height_ratios=[bar, 1 - bar] if bar < 1
                                                     else [1, 0], hspace=0)
    cax = fig.add_subplot(cgs[0], label="colour bar")
    over = finite.size and float(finite.max()) > hi
    under = finite.size and float(finite.min()) < lo
    extend = {(True, True): "both", (True, False): "max", (False, True): "min"}.get((bool(over), bool(under)),
                                                                                    "neither")
    cbar = fig.colorbar(mesh, cax=cax, extend=extend)
    cbar.set_label(_VALUE_LABELS[value], fontsize=0.8 * text_size)
    cbar.ax.tick_params(labelsize=0.8 * text_size, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return fig


__all__ = ["plot_comparison"]
