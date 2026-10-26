"""Plots of motif enrichment and co-occurrence results."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique
from ._api import _check_motif
from ._core import Motif
from ._plot import _pyplot, draw_logo

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_SIGNIFICANT = "#255C99"
_NOT_SIGNIFICANT = "#B4B4B4"
_MISSING = "#D0D0D0"
_SMALLEST = 5e-324
_SIZE_TITLES = {
    "target_seq_hits": "Target sequences\nwith hits",
    "target_site_hits": "Target hits",
    "bkg_seq_hits": "Background sequences\nwith hits",
    "bkg_site_hits": "Background hits",
}


def _neg_log10(results: pl.DataFrame, col: str) -> np.ndarray:
    """``-log10`` of a P-value or q-value column, from its log10 column when
    there is one so that values too small for a double stay finite."""
    if f"log10_{col}" in results.columns:
        return -results[f"log10_{col}"].cast(pl.Float64).to_numpy()
    if col not in results.columns:
        raise ValueError(f"results has no {col} or log10_{col} column")
    v = results[col].cast(pl.Float64).to_numpy()
    return -np.log10(np.clip(v, _SMALLEST, None))


def _logo_lookup(motifs) -> dict[str, Motif]:
    if isinstance(motifs, Mapping):
        return {str(k): _check_motif(m) for k, m in motifs.items()}
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    return dict(zip(_make_unique([m.name for m in ms]), ms))


def _overlap(a, b) -> float:
    """The area shared by two boxes given as ``(x0, y0, x1, y1)``."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _place_labels(px: np.ndarray, py: np.ndarray, radius: np.ndarray, sizes: list[tuple[float, float]],
                  which: list[int], frame) -> list[tuple[float, float, bool]]:
    """Greedily place label boxes of the given sizes next to points.

    Everything is in points, as display positions. Each label in ``which``,
    taken in turn, tries spots on rings of growing distance around its point
    and takes the one that overlaps the labels placed before it least, then
    other points, then the edge of ``frame``, preferring closer spots.
    Returns the lower left corner of each box relative to its point, and
    whether it ended up far enough away to need a leader line.
    """
    placed: list[tuple[float, float, float, float]] = []
    dots = np.column_stack([px - radius, py - radius, px + radius, py + radius])
    out = []
    for k, i in enumerate(which):
        w, h = sizes[k]
        best, best_cost = None, math.inf
        for ring, gap in enumerate((3.0, 9.0, 17.0, 27.0, 40.0, 56.0, 76.0, 100.0, 130.0)):
            r = radius[i] + gap
            d = r * 0.7
            spots = ((r, -h / 2), (-r - w, -h / 2), (d, d), (-d - w, d), (d, -d - h),
                     (-d - w, -d - h), (-w / 2, r), (-w / 2, -r - h))
            for dx, dy in spots:
                box = (px[i] + dx, py[i] + dy, px[i] + dx + w, py[i] + dy + h)
                hit_labels = sum(_overlap(box, b) for b in placed)
                on_dots = ((np.minimum(box[2], dots[:, 2]) > np.maximum(box[0], dots[:, 0]))
                           & (np.minimum(box[3], dots[:, 3]) > np.maximum(box[1], dots[:, 1])))
                hit_dots = int(on_dots.sum()) - bool(on_dots[i])
                outside = (max(frame[0] - box[0], 0) + max(box[2] - frame[2], 0)
                           + max(frame[1] - box[1], 0) + max(box[3] - frame[3], 0))
                cost = 1e4 * hit_labels + 1e3 * outside + 5 * hit_dots + ring
                if cost < best_cost:
                    best, best_cost = (dx, dy, ring > 1), cost
            if best_cost < ring + 1:
                break
        dx, dy, _ = best
        placed.append((px[i] + dx, py[i] + dy, px[i] + dx + w, py[i] + dy + h))
        out.append(best)
    return out


def plot_enrichment(
    results: pl.DataFrame,
    y: str = "qvalue",
    cutoff: float = 0.1,
    labels: int | Sequence[str] = 10,
    size_by: str | None = "target_seq_hits",
    motifs: Motif | Iterable[Motif] | Mapping[str, Motif] | None = None,
    logo_height: float | None = None,
    colour: str = _SIGNIFICANT,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw motif enrichment from [`enrich_motifs()`][universalmotif.enrich_motifs] as a volcano plot.

    Each motif is a point at its log2 enrichment and ``-log10`` q-value, or
    P-value with ``y="pvalue"``, sized by the number of target sequences
    with hits. Motifs at or below ``cutoff`` are coloured, a dashed line
    marks the cutoff, and the most significant are labelled. Motifs with no
    target hits have an enrichment of zero, and are drawn as triangles at
    the left edge.

    Since [`enrich_motifs()`][universalmotif.enrich_motifs] keeps only
    motifs with q-values up to 0.1 by default, run it with ``qvalue=1`` to
    see all of them.

    Parameters
    ----------
    results
        A table from [`enrich_motifs()`][universalmotif.enrich_motifs].
    y
        ``"qvalue"`` or ``"pvalue"``. The log10 columns are used when present,
        so values too small for a double are still drawn.
    cutoff
        The q-value or P-value at or below which motifs are significant.
    labels
        The number of most significant motifs to label, or the name or names
        of motifs to label whether significant or not. Labels are nudged to
        keep them from covering each other.
    size_by
        A column of ``results`` to size points by, or ``None`` for points of
        one size.
    motifs
        The motifs given to [`enrich_motifs()`][universalmotif.enrich_motifs]
        or a mapping from names to motifs, to draw a logo beside each label.
        Motifs in a list are matched to the table by name, with duplicates
        named as [`enrich_motifs()`][universalmotif.enrich_motifs] names them.
    logo_height
        The height of the logos in points, by default twice the text size.
    colour
        The colour of significant points.
    ax
        Axes to draw into. They are returned, and otherwise a new figure is.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> targets = um.example_sequences("targets")
    >>> controls = um.example_sequences("controls")
    >>> res = um.enrich_motifs(motifs, targets, controls, qvalue=1, nthreads=1)
    >>> fig = um.plot_enrichment(res, labels=3, motifs=motifs)
    >>> sorted(t.get_text() for t in fig.axes[0].texts)
    ['G-box', 'W-box', 'q = 0.1']
    >>> plt.close(fig)
    """
    if "log2_enrichment" not in results.columns or "motif" not in results.columns:
        raise ValueError("results needs motif and log2_enrichment columns; was it made by enrich_motifs()?")
    if y not in ("qvalue", "pvalue"):
        raise ValueError("y must be 'qvalue' or 'pvalue'")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must be in (0, 1]")
    if size_by is not None and size_by not in results.columns:
        raise ValueError(f"results has no {size_by} column to size points by")
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels, numbers.Integral) and labels < 0:
        raise ValueError("labels must be a non-negative number or a list of motif names")
    if results.height == 0:
        raise ValueError("results has no rows to plot")
    names = results["motif"].cast(pl.String).fill_null("None").to_list()
    x = results["log2_enrichment"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    yv = _neg_log10(results, y)
    sv = (results[size_by].cast(pl.Float64).fill_null(0.0).to_numpy() if size_by is not None
          else np.zeros(len(names)))
    keep = ~np.isnan(x) & ~np.isnan(yv)
    if not keep.any():
        raise ValueError("results has no enrichment values to plot")
    sig = keep & (yv >= -math.log10(cutoff))

    if isinstance(labels, numbers.Integral):
        chosen = sorted(np.flatnonzero(sig), key=lambda i: (-yv[i], -x[i]))[:labels]
    else:
        wanted = list(labels)
        missing = [lab for lab in wanted if lab not in names]
        if missing:
            raise ValueError(f"results has no motifs named {', '.join(map(str, missing[:5]))}")
        chosen = sorted({names.index(lab) for lab in wanted if keep[names.index(lab)]},
                        key=lambda i: (-yv[i], -x[i]))
    logos: dict[str, Motif] = {}
    if motifs is not None and chosen:
        lookup = _logo_lookup(motifs)
        missing = [names[i] for i in chosen if names[i] not in lookup]
        if missing:
            raise ValueError(f"motifs has no motif named {', '.join(missing[:5])}")
        logos = {names[i]: lookup[names[i]] for i in chosen}

    lo, hi = float(sv[keep].min()), float(sv[keep].max())

    def area(v):
        v = np.asarray(v, dtype=float)
        return 12 + 108 * (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 36.0)

    sizes = area(sv)
    line = -math.log10(cutoff)
    edge = keep & np.isneginf(x)
    finite = keep & ~edge
    xlo = min(float(x[finite].min()), 0.0) if finite.any() else -1.0
    xhi = max(float(x[finite].max()), 0.0) if finite.any() else 1.0
    pad = 0.08 * max(xhi - xlo, 1.0)
    left = xlo - (2 if edge.any() else 1) * pad
    # Motifs with no target hits have no finite enrichment and sit at the
    # left edge instead.
    x = np.where(edge, left + 0.35 * pad, x)
    top = 1.12 * max(float(yv[keep].max()), line) or 1.0

    plt = _pyplot()
    own_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6.5, 4.8))
    else:
        fig = ax.figure
    ax.axvline(0, color="#E2E2E2", linewidth=0.8, zorder=0)
    ax.axhline(line, color="#7F7F7F", linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(f"{'q' if y == 'qvalue' else 'p'} = {cutoff:g}", xy=(1.0, line), xycoords=ax.get_yaxis_transform(),
                xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", fontsize=text_size - 1,
                color="#5E5E5E")
    for mask, col, z in ((keep & ~sig, _NOT_SIGNIFICANT, 2), (sig, colour, 3)):
        for marker, part in (("o", mask & ~edge), ("<", mask & edge)):
            if part.any():
                ax.scatter(x[part], yv[part], s=sizes[part], c=col, marker=marker, edgecolors="white",
                           linewidths=0.6, zorder=z)
    ax.set_xlim(left, xhi + pad)
    ax.set_ylim(0, top)
    ax.set_xlabel("log2(enrichment)")
    ax.set_ylabel("-log10(q-value)" if y == "qvalue" else "-log10(P-value)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if size_by is not None and hi > lo:
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator

        breaks = [b for b in MaxNLocator(nbins=3, integer=True).tick_values(lo, hi) if lo <= b <= hi]
        handles = [Line2D([0], [0], linestyle="", marker="o", color=_NOT_SIGNIFICANT,
                          markeredgecolor="white", markersize=math.sqrt(float(area(b)))) for b in breaks]
        # Beside the plot in a figure of its own, and inside axes passed in,
        # where it could otherwise run into neighbouring plots.
        where = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)} if own_figure else {"loc": "best"}
        ax.legend(handles, [f"{b:g}" for b in breaks], title=_SIZE_TITLES.get(size_by, size_by),
                  frameon=False, fontsize=text_size, title_fontsize=text_size, alignment="left", **where)
    if own_figure:
        fig.tight_layout()
    elif fig.get_layout_engine() is not None:
        # The layout moves the axes, and with them the points, so it is
        # settled before the labels are placed.
        fig.draw_without_rendering()
    if chosen:
        _label_points(ax, x, yv, sizes, keep, chosen, names, logos, text_size, logo_height)
    return fig if own_figure else ax


def _label_points(ax: Axes, x, y, sizes, keep, chosen, names, logos, text_size, logo_height) -> None:
    """Label the chosen points, with logos when there are any, nudging the
    labels apart."""
    import matplotlib.transforms as mtransforms

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    disp = ax.transData.transform(np.column_stack([x, y])) * to_pt
    # Points that are not drawn are left out of the way.
    disp[~keep] = np.nan
    radius = np.sqrt(sizes) / 2
    ext = ax.get_window_extent(renderer)
    frame = (ext.x0 * to_pt, ext.y0 * to_pt, ext.x1 * to_pt, ext.y1 * to_pt)
    logo_h = 2.0 * text_size if logo_height is None else float(logo_height)
    gap = 0.3 * text_size

    parts, boxes = [], []
    for i in chosen:
        probe = ax.text(0, 0, names[i], fontsize=text_size)
        text_ext = probe.get_window_extent(renderer)
        probe.remove()
        w, h = text_ext.width * to_pt, text_ext.height * to_pt
        lw = 0.4 * logo_h * logos[names[i]].width if names[i] in logos else 0.0
        parts.append((w, h, lw))
        boxes.append((w + gap + lw if lw else w, max(h, logo_h) if lw else h))
    spots = _place_labels(disp[:, 0], disp[:, 1], radius, boxes, chosen, frame)
    for i, (w, h, lw), (bw, bh), (dx, dy, far) in zip(chosen, parts, boxes, spots):
        # The text takes the side of the box nearer the point, and the logo
        # the side further away.
        text_right = lw > 0 and dx + bw / 2 < 0
        tx, ty = dx + (lw + gap if text_right else 0.0), dy + (bh - h) / 2
        arrow = None
        if far:
            relpos = (min(max(-tx / w, 0.0), 1.0), min(max(-ty / h, 0.0), 1.0))
            arrow = {"arrowstyle": "-", "color": "#7F7F7F", "linewidth": 0.6, "relpos": relpos,
                     "shrinkA": 1, "shrinkB": math.sqrt(sizes[i]) / 2 + 1}
        ax.annotate(names[i], xy=(x[i], y[i]), xytext=(tx, ty + h / 2), textcoords="offset points",
                    ha="left", va="center", fontsize=text_size, zorder=5, arrowprops=arrow)
        if lw:
            # A box measured in points from the point, so the logo stays
            # beside its label when the figure is resized or saved at
            # another resolution.
            at_point = (mtransforms.Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans
                        + mtransforms.ScaledTranslation(x[i], y[i], ax.transData))
            letters = draw_logo(ax, logos[names[i]], x=dx if text_right else dx + w + gap,
                                y=dy + (bh - logo_h) / 2, width=lw, height=logo_h, transform=at_point)
            letters.set_zorder(5)
            letters.set_clip_on(False)


def _coocc_matrix(results: pl.DataFrame, value: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The motifs, in order of first appearance, with the shown values and
    q-values of each pair in symmetric matrices, NaN where untested."""
    a = results["motif_a"].cast(pl.String).fill_null("None").to_list()
    b = results["motif_b"].cast(pl.String).fill_null("None").to_list()
    motifs = list(dict.fromkeys(a + b))
    index = {m: i for i, m in enumerate(motifs)}
    n = len(motifs)
    if value == "odds_ratio":
        with np.errstate(divide="ignore"):
            shown = np.log2(results["odds_ratio"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    else:
        shown = _neg_log10(results, value)
    q = results["qvalue"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    tested = ~np.isnan(results["pvalue"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    vals, qs = np.full((n, n), np.nan), np.full((n, n), np.nan)
    for k, (ma, mb) in enumerate(zip(a, b)):
        if tested[k]:
            i, j = index[ma], index[mb]
            vals[i, j] = vals[j, i] = shown[k]
            qs[i, j] = qs[j, i] = q[k]
    return motifs, vals, qs


def plot_coocc(
    results: pl.DataFrame,
    value: str = "odds_ratio",
    cutoff: float = 0.05,
    order: str | Sequence[str] = "cluster",
    cmap: str | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw motif co-occurrence from [`motif_coocc()`][universalmotif.motif_coocc] as a heatmap.

    Every pair of motifs is a cell, filled the same on both sides of the
    diagonal with the log2 odds ratio, on a scale centred on zero, or with
    the ``-log10`` q-value or P-value. Pairs with q-values at or below
    ``cutoff`` are marked with a dot, while untested pairs and the diagonal,
    unless the table has self pairs, are left grey. Odds ratios of zero or
    infinity are drawn at the ends of the scale.

    Parameters
    ----------
    results
        A table from [`motif_coocc()`][universalmotif.motif_coocc].
    value
        ``"odds_ratio"``, ``"qvalue"`` or ``"pvalue"``.
    cutoff
        The q-value at or below which pairs are marked.
    order
        ``"cluster"`` to order motifs by average-linkage clustering of their
        rows of shown values, ``"input"`` to keep them in the order they
        first appear in ``results``, or a list of motif names, which can
        leave some motifs out.
    cmap
        A matplotlib colour map, by default a diverging one for odds ratios
        and a sequential one otherwise.
    ax
        Axes to draw into. They are returned, and otherwise a new figure is.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import universalmotif as um
    >>> motifs = [um.example_motifs()[i] for i in (0, 3, 5)]
    >>> res = um.motif_coocc(motifs, um.example_sequences("targets"), nthreads=1)
    >>> fig = um.plot_coocc(res)
    >>> [t.get_text() for t in fig.axes[0].get_xticklabels()]
    ['W-box', 'TATA-box', 'G-box']
    >>> plt.close(fig)
    """
    for col in ("motif_a", "motif_b", "odds_ratio", "pvalue", "qvalue"):
        if col not in results.columns:
            raise ValueError(f"results has no {col} column; was it made by motif_coocc()?")
    if value not in ("odds_ratio", "qvalue", "pvalue"):
        raise ValueError("value must be 'odds_ratio', 'qvalue' or 'pvalue'")
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must be in (0, 1]")
    if isinstance(order, str) and order not in ("cluster", "input"):
        raise ValueError("order must be 'cluster', 'input' or a list of motif names")
    if results.height == 0:
        raise ValueError("results has no rows to plot")
    motifs, vals, qs = _coocc_matrix(results, value)
    if isinstance(order, str):
        idx = list(range(len(motifs)))
        if order == "cluster" and len(motifs) > 2:
            # Untested pairs count as zero, which is no association for log
            # odds ratios and a P-value of 1 otherwise, infinite log odds as
            # the largest finite one, and a missing diagonal as the strongest
            # association, so that two motifs found together end up close.
            finite = np.isfinite(vals)
            big = np.abs(vals[finite]).max() if finite.any() else 1.0
            rows = np.clip(np.nan_to_num(vals, nan=0.0), -big, big)
            diagonal = np.isnan(np.diag(vals))
            rows[diagonal, diagonal] = big
            dist = np.sqrt(((rows[:, None, :] - rows[None, :, :]) ** 2).sum(axis=2))
            idx = list(_core._hclust(dist.tolist(), "average", motifs)["order"])
    else:
        wanted = list(dict.fromkeys(order))
        missing = [m for m in wanted if m not in motifs]
        if missing:
            raise ValueError(f"results has no motifs named {', '.join(map(str, missing[:5]))}")
        if not wanted:
            raise ValueError("order names no motifs")
        idx = [motifs.index(m) for m in wanted]
    motifs = [motifs[i] for i in idx]
    vals, qs = vals[np.ix_(idx, idx)], qs[np.ix_(idx, idx)]
    n = len(motifs)

    import matplotlib as mpl

    finite = np.isfinite(vals)
    if value == "odds_ratio":
        lim = float(np.abs(vals[finite]).max()) if finite.any() else 1.0
        lim = lim or 1.0
        vmin, vmax = -lim, lim
        colours = mpl.colormaps[cmap or "RdBu_r"]
        label = "log2(odds ratio)"
    else:
        vmax = float(vals[finite].max()) if finite.any() else 1.0
        vmin, vmax = 0.0, vmax or 1.0
        colours = mpl.colormaps[cmap or "Blues"]
        label = "-log10(q-value)" if value == "qvalue" else "-log10(P-value)"
    colours = colours.with_extremes(bad=_MISSING)
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(bool(np.isneginf(vals).any()), bool(np.isposinf(vals).any()))]
    shown = np.ma.masked_invalid(np.clip(vals, vmin, vmax))

    plt = _pyplot()
    own_figure = ax is None
    if ax is None:
        side = 0.32 * n + 1.6
        fig, ax = plt.subplots(figsize=figsize or (max(4.5, side + 1.4), max(3.8, side)))
    else:
        fig = ax.figure
    image = ax.imshow(shown, cmap=colours, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(n), motifs, rotation=90, fontsize=text_size)
    ax.set_yticks(range(n), motifs, fontsize=text_size)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    marked = np.argwhere(qs <= cutoff)
    if marked.size:
        # Dots in white on dark cells and in black on light ones.
        rgba = image.to_rgba(shown[marked[:, 0], marked[:, 1]])
        light = rgba[:, :3] @ np.array([0.299, 0.587, 0.114]) > 0.55
        ax.scatter(marked[:, 1], marked[:, 0], s=min(16.0, 900.0 / n ** 1.2),
                   c=np.where(light[:, None], [[0.1, 0.1, 0.1, 1.0]], [[1.0, 1.0, 1.0, 1.0]]),
                   marker="o", linewidths=0, zorder=3)
    from matplotlib.lines import Line2D

    ax.legend([Line2D([0], [0], linestyle="", marker="o", color="#1A1A1A", markersize=4)],
              [f"q ≤ {cutoff:g}"], frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              fontsize=text_size, handletextpad=0.2, borderaxespad=0.2)
    bar = fig.colorbar(image, ax=ax, extend=extend, fraction=0.046, pad=0.03, shrink=0.8)
    bar.set_label(label, fontsize=text_size)
    bar.ax.tick_params(labelsize=text_size)
    bar.outline.set_visible(False)
    if own_figure:
        fig.tight_layout()
    return fig if own_figure else ax


__all__ = ["plot_coocc", "plot_enrichment"]
