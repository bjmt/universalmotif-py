"""Sequence logos, motif trees and motif peak plots with matplotlib."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique, _sequence_pairs
from ._api import _check_motif
from ._core import Motif
from ._logo import Glyph, matrix_glyphs, motif_glyphs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ._compare import MotifTree

DNA_COLOURS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}
RNA_COLOURS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "U": "#D62839"}
AA_COLOURS = {
    **dict.fromkeys("GSTYC", "#058644"),
    **dict.fromkeys("QN", "#720091"),
    **dict.fromkeys("KRH", "#0046C5"),
    **dict.fromkeys("DE", "#C5003E"),
    **dict.fromkeys("AVLIPWFM", "#2E2E2E"),
}
_MISSING_COLOUR = "#7F7F7F"
_YLABELS = {"PPM": "probability", "ICM": "bits", "PWM": "logodds", "PCM": "counts", "CWM": "contribution"}


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib; install it with `pip install universalmotif[plot]`"
        ) from e
    return plt


def _draw_glyphs(ax: Axes, glyphs: list[Glyph], colours: Mapping[str, str] | None, fill: str = "black",
                 x0: float = 0.0, y0: float = 0.0, xscale: float = 1.0, yscale: float = 1.0) -> None:
    from matplotlib.collections import PolyCollection

    if not glyphs:
        return
    verts = [np.column_stack([x0 + g.x * xscale, y0 + g.y * yscale]) for g in glyphs]
    faces = [fill if colours is None else colours.get(g.group, _MISSING_COLOUR) for g in glyphs]
    ax.add_collection(PolyCollection(verts, facecolors=faces, edgecolors="none", linewidths=0))


def _x_limits(n: int) -> tuple[float, float]:
    lo, hi = 1 - 0.501, n + 0.501
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def _logo_input(x, letters) -> tuple[np.ndarray, list[str]]:
    if isinstance(x, Motif):
        return np.asarray(x.matrix, dtype=float), list(x.letters)
    if isinstance(x, Mapping):
        return np.array([list(map(float, v)) for v in x.values()], dtype=float), [str(k) for k in x]
    m = np.atleast_2d(np.asarray(x, dtype=float))
    if letters is None:
        raise ValueError("give the row letters of the matrix with `letters`")
    letters = list(letters)
    if len(letters) != m.shape[0]:
        raise ValueError("`letters` must have one entry per matrix row")
    return m, letters


def view_logo(
    x,
    letters: Sequence[str] | None = None,
    colour_scheme: Mapping[str, str] | None = None,
    fill: str = "black",
    min_height: float = 0.01,
    x_spacer: float = 0.04,
    y_spacer: float = 0.01,
    sort_positions: bool = False,
    sort_positions_decreasing: bool = True,
    fit_to_height: float | None = None,
    flip_neg: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Draw a logo from any numeric matrix.

    ``x`` is a (rows, positions) array with ``letters`` naming the rows, a
    mapping from row names to values, or a motif (drawn as its matrix is).
    Row names may be several characters long, columns can have any height,
    and negative values are stacked below zero (upside down with
    ``flip_neg``). Letters are coloured by ``colour_scheme`` (a mapping from
    single characters to colours) or all filled with ``fill``.

    Letters shorter than ``min_height`` are left out, ``x_spacer`` and
    ``y_spacer`` set the gaps between letters, ``sort_positions`` stacks
    each position's letters by height, and ``fit_to_height`` scales every
    position to that height. The axes are left bare.
    """
    m, rows = _logo_input(x, letters)
    if colour_scheme is not None:
        chars = {c for r in rows for c in r}
        missing = chars - set(colour_scheme)
        if missing:
            raise ValueError(f"colour_scheme has no colours for: {', '.join(sorted(missing))}")
    glyphs = matrix_glyphs(m, rows, min_height, x_spacer, y_spacer, sort_positions,
                           sort_positions_decreasing, fit_to_height, 1.0, flip_neg)
    if ax is None:
        _, ax = _pyplot().subplots(figsize=(max(2.0, 0.5 * m.shape[1]), 2.0))
    _draw_glyphs(ax, glyphs, colour_scheme, fill)
    ax.set_xlim(*_x_limits(m.shape[1]))
    ax.autoscale(axis="y")
    ax.set_axis_off()
    return ax


def _default_colours(alphabet: str, letters: Sequence[str] = ()) -> dict[str, str]:
    """The R package's colours for DNA, RNA and amino acids, and a
    categorical palette for other alphabets."""
    known = {"DNA": DNA_COLOURS, "RNA": RNA_COLOURS, "AA": AA_COLOURS}.get(alphabet)
    if known is not None:
        return known
    import matplotlib

    palette = matplotlib.colormaps["tab20"].colors
    return {letter: matplotlib.colors.to_hex(palette[i % len(palette)]) for i, letter in enumerate(letters)}


def _rc_matrix(m: np.ndarray) -> np.ndarray:
    return m[::-1, ::-1]


class _Default:
    pass


_DEFAULT = _Default()


def _prepare_motifs(motifs, use_type, rc, min_overlap, relative_entropy, sort_by, rc_text, nthreads):
    """Display matrices of motifs aligned to a common frame, in display
    order: returns (names, matrices, offsets, widths, alphabet)."""
    if use_type not in _YLABELS:
        raise ValueError("use_type must be one of PCM, PPM, PWM, ICM and CWM")
    if sort_by not in ("none", "ic", "similarity"):
        raise ValueError("sort_by must be 'none', 'ic' or 'similarity'")
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("no motifs given")
    alphabet = ms[0].alphabet
    if any(m.alphabet != alphabet for m in ms):
        raise ValueError("all motifs must share the same alphabet")
    if use_type == "CWM" and any(m.type != "CWM" for m in ms):
        raise ValueError('use_type="CWM" needs every motif to be a CWM')

    ic_order = list(range(len(ms)))
    if len(ms) > 1:
        ic_order = sorted(ic_order, key=lambda i: -ms[i].icscore)
    ms = [ms[i] for i in ic_order]
    if use_type == "CWM":
        mats = [np.asarray(m.matrix, dtype=float) for m in ms]
    else:
        mats = [np.array(_core._display_matrix(m, use_type, relative_entropy), dtype=float).T for m in ms]
    names = _make_unique([m.name for m in ms])
    widths = [m.shape[1] for m in mats]
    if len(ms) == 1:
        return names, mats, [0], widths, alphabet

    placements, width = _core._anchor_layout(ms, 0, min_overlap, rc, nthreads)
    padded, offsets = [], []
    for i, (m, (left, flip)) in enumerate(zip(mats, placements)):
        if flip:
            m = _rc_matrix(m)
            names[i] = names[i] + rc_text
        out = np.zeros((m.shape[0], width))
        out[:, left : left + m.shape[1]] = m
        padded.append(out)
        offsets.append(left)

    if sort_by == "none":
        perm = sorted(range(len(ms)), key=lambda i: ic_order[i])
    elif sort_by == "ic":
        perm = list(range(len(ms)))
    else:
        n = len(padded)
        flat = [p.ravel(order="F") for p in padded]
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.corrcoef(flat[i], flat[j])[0, 1]
                if not np.isfinite(r):
                    r = 0.0
                dist[i][j] = dist[j][i] = 1 - r
        perm = list(_core._hclust(dist, "complete", [str(i) for i in range(n)])["order"])
    return ([names[i] for i in perm], [padded[i] for i in perm], [offsets[i] for i in perm],
            [widths[i] for i in perm], alphabet)


def _y_axis(mats: list[np.ndarray], use_type: str):
    """Tick positions and limits for a logo panel (as the R package)."""
    nrow = mats[0].shape[0]
    if use_type == "PPM":
        return [0, 0.5, 1], (0.0, 1.0)
    if use_type == "ICM":
        top = math.log2(nrow)
        return sorted(set(np.round([0, top / 2, top]).tolist())), (0.0, top)
    if use_type == "PCM":
        top = max(float(m.sum(axis=0).max()) for m in mats)
        return sorted({0.0, float(np.round(top / 2)), top}), (0.0, top)
    lo = min(float(np.where(m > 0, 0, m).sum(axis=0).min()) for m in mats)
    hi = max(float(np.where(m < 0, 0, m).sum(axis=0).max()) for m in mats)
    breaks = np.array([lo, lo / 2, 0, hi / 2, hi])
    if use_type == "PWM":
        breaks = np.round(breaks)
    breaks[0] -= max(0.1 * abs(breaks[0]), 0.1)
    return list(dict.fromkeys(breaks.tolist())), (float(breaks[0]), hi)


def _panel_glyphs(m, rows, use_type, min_height, x_spacer, y_spacer, sort_positions,
                  sort_positions_decreasing, fit_to_height, flip_neg):
    glyphs = motif_glyphs(m, rows, use_type, min_height, x_spacer, y_spacer, sort_positions,
                          sort_positions_decreasing, fit_to_height, flip_neg)
    for g in glyphs:
        g.y = g.y * 0.999
    return glyphs


def _style_axes(ax, ylabel, ticks, ylim, n, show_positions, labels=None, text_size=12):
    ax.set_xlim(*_x_limits(n))
    # Ticks outside the limits are dropped, as ggplot2 drops them (matplotlib
    # would otherwise widen the axis to show them).
    ax.set_yticks([t for t in ticks if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9])
    ax.set_ylim(*ylim)
    ax.set_xticks(range(1, n + 1))
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelbottom=show_positions)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="x", length=0, labelsize=text_size * 0.8)
    ax.tick_params(axis="y", width=0.5, labelsize=text_size * 0.8)
    ax.set_ylabel(ylabel, fontsize=text_size)


def view_motifs(
    motifs: Motif | Iterable[Motif],
    use_type: str = "ICM",
    rc: bool = True,
    min_overlap: int = 6,
    relative_entropy: bool = False,
    sort_by: str = "none",
    show_names: bool = True,
    names_pos: str = "top",
    show_positions: bool = True,
    show_positions_once: bool = True,
    colour_scheme: Mapping[str, str] | None = None,
    min_height: float = 0.01,
    x_spacer: float = 0.04,
    y_spacer: float = 0.01,
    sort_positions: bool | None = None,
    sort_positions_decreasing: bool = True,
    fit_to_height: Any = _DEFAULT,
    rc_text: str = " [RC]",
    flip_neg: bool = False,
    text_size: float = 12,
    figsize: tuple[float, float] | None = None,
    return_raw: bool = False,
    nthreads: int = 1,
):
    """Draw motifs as sequence logos.

    This is the port of universalmotif's ``view_motifs_lite()``, extended to
    any alphabet. Several motifs are aligned to the one with the highest
    information content (as :func:`compare_motifs` aligns them, trying
    reverse complements of DNA and RNA motifs with ``rc``), reverse
    complemented where that aligns better (their names get ``rc_text``),
    padded into a common frame and drawn one above the other.

    Parameters
    ----------
    use_type
        Matrix type to draw: ``"ICM"`` (bits), ``"PPM"``, ``"PWM"``,
        ``"PCM"``, or ``"CWM"`` (only for motifs that are CWMs). Motifs
        without ``nsites`` count as 100 sites.
    sort_by
        Display order: ``"none"`` (input order), ``"ic"`` (decreasing
        information content) or ``"similarity"`` (clustered by correlation).
    show_names, names_pos
        Name each logo, above it (``"top"``) or beside it (``"right"``).
    show_positions, show_positions_once
        Number the positions, once under the last logo or under every logo
        (counting each motif's own positions).
    colour_scheme
        Letter colours. DNA, RNA and amino acids default to the R package's
        colours, other alphabets to a categorical palette.
    min_height, x_spacer, y_spacer, sort_positions, sort_positions_decreasing, fit_to_height, flip_neg
        Letter layout, as in :func:`view_logo`. Letters are sorted by height
        except for PCMs and PPMs, and PPM positions are scaled to a height of
        1, unless told otherwise.
    return_raw
        Return the aligned matrices as a dict from names to (letters,
        positions) arrays instead of drawing them.

    Returns
    -------
    A matplotlib figure, or with ``return_raw`` a dict of matrices.
    """
    if names_pos not in ("top", "right"):
        raise ValueError("names_pos must be 'top' or 'right'")
    if sort_positions is None:
        sort_positions = use_type not in ("PCM", "PPM")
    if fit_to_height is _DEFAULT:
        fit_to_height = 1.0 if use_type == "PPM" else None
    elif fit_to_height is not None and use_type in ("ICM", "PWM"):
        import warnings

        warnings.warn('fit_to_height is ignored for use_type "ICM" and "PWM"', stacklevel=2)
        fit_to_height = None
    motifs = [motifs] if isinstance(motifs, Motif) else list(motifs)
    names, mats, offsets, widths, alphabet = _prepare_motifs(
        motifs, use_type, rc, min_overlap, relative_entropy, sort_by, rc_text, nthreads
    )
    if return_raw:
        return dict(zip(names, mats))
    if use_type == "PWM" and any(not np.isfinite(m).all() for m in mats):
        raise ValueError("found non-finite values in a motif")
    rows = list(motifs[0].letters)
    colours = colour_scheme if colour_scheme is not None else _default_colours(alphabet, rows)
    if colour_scheme is not None:
        missing = set(colour_scheme) - set(rows)
        if missing:
            raise ValueError(f"colour_scheme has letters not in the alphabet: {', '.join(sorted(missing))}")
    width = mats[0].shape[1]
    ticks, ylim = _y_axis(mats, use_type)
    plt = _pyplot()
    n = len(mats)
    if figsize is None:
        figsize = (max(3.0, 0.45 * width + 1.5 + (1.5 if names_pos == "right" and n > 1 else 0)),
                   1.0 + 1.6 * n)
    shared = n > 1 and show_positions_once
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False, sharex=shared)
    for k, (name, m, off, w) in enumerate(zip(names, mats, offsets, widths)):
        ax = axes[k, 0]
        glyphs = _panel_glyphs(m, rows, use_type, min_height, x_spacer, y_spacer, sort_positions,
                               sort_positions_decreasing, fit_to_height, flip_neg)
        _draw_glyphs(ax, glyphs, colours)
        if use_type in ("PWM", "CWM"):
            ax.axhline(0, linewidth=0.5, color="#BFBFBF")
        labels = None
        if n > 1 and not show_positions_once:
            labels = [str(i - off + 1) if off <= i < off + w else "" for i in range(width)]
        show = show_positions and (not shared or k == n - 1)
        _style_axes(ax, _YLABELS[use_type], ticks, ylim, width, show, labels, text_size)
        if n > 1 and show_names:
            if names_pos == "top":
                ax.set_title(name, fontsize=text_size)
            else:
                ax.text(1.02, 0.5, name, transform=ax.transAxes, va="center", ha="left", fontsize=text_size)
    fig.tight_layout()
    return fig


def plot_motif_peaks(
    peaks: pl.DataFrame,
    motifs: Sequence[str] | None = None,
    ncol: int = 2,
    bins: int = 50,
    fill_window: str = "#ffe4a3",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw the positions of motif hits found by :func:`motif_peaks`.

    One panel per motif (or per motif named in ``motifs``) shows a
    histogram of the hit centres, with the most enriched window shaded and
    the P-value and enrichment in the title.
    """
    if "centers" not in peaks.columns:
        raise ValueError("peaks has no centers column; was it made by motif_peaks()?")
    if peaks.height == 0:
        raise ValueError("peaks has no rows to plot")
    if motifs is not None:
        peaks = peaks.filter(pl.col("motif").is_in(list(motifs)))
        if peaks.height == 0:
            raise ValueError("none of the requested motifs are in peaks")
    plt = _pyplot()
    n = peaks.height
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    if figsize is None:
        figsize = (4.0 * ncol, 2.8 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    seq_length = int(peaks["seq_length"][0]) if "seq_length" in peaks.columns else None
    for k, row in enumerate(peaks.iter_rows(named=True)):
        ax = axes[k // ncol, k % ncol]
        if row["best_window"] is not None and row["best_center"] is not None:
            half = row["best_window"] / 2
            ax.axvspan(row["best_center"] - half, row["best_center"] + half, color=fill_window,
                       alpha=0.5, linewidth=0)
        centres = np.asarray(row["centers"], dtype=float)
        if centres.size:
            ax.hist(centres, bins=bins, color="black")
        if seq_length:
            ax.set_xlim(0, seq_length + 1)
        enr = row.get("enrichment")
        enr_text = "NA" if enr is None else f"{enr:.1f}"
        ax.set_title(f"{row['motif']}  (p = {row['pvalue']:.2g} | {enr_text}x)", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Hit centre position")
        ax.set_ylabel("Count")
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    fig.tight_layout()
    return fig


# --- Trees ------------------------------------------------------------------


def _tree_coords(tree: MotifTree, branch_length: bool):
    """x (distance from the root) and y of every node: leaves are 0..n-1,
    merge i is node n + i."""
    n = len(tree.labels)
    merge = np.asarray(tree.merge, dtype=int)
    m = merge.shape[0]

    def node(c: int) -> int:
        return -c - 1 if c < 0 else n + c - 1

    children = [(node(a), node(b)) for a, b in merge]
    y = np.zeros(n + m)
    for pos, leaf in enumerate(tree.order):
        y[leaf] = n - 1 - pos
    for i, (a, b) in enumerate(children):
        y[n + i] = (y[a] + y[b]) / 2
    x = np.zeros(n + m)
    if branch_length:
        height = np.concatenate([np.zeros(n), np.asarray(tree.height, dtype=float)])
        x = (height[-1] - height) / 2
    else:
        depth = np.zeros(n + m, dtype=int)
        for i in range(m - 1, -1, -1):
            for c in children[i]:
                depth[c] = depth[n + i] + 1
        x[:n] = depth[:n].max()
        for i, (a, b) in enumerate(children):
            x[n + i] = min(x[a], x[b]) - 1
        x -= x[-1]
    return x, y, children


def _tree_values(tree: MotifTree, value, what: str) -> list | None:
    if value is None:
        return None
    if isinstance(value, str):
        if tree.motifs is None:
            raise ValueError(f"{what} by motif attribute needs a tree made by motif_tree()")
        vals = [getattr(m, value) for m in tree.motifs]
        if value in ("pval", "qval", "eval"):
            vals = [None if v is None else -math.log10(v) for v in vals]
        return vals
    vals = list(value)
    if len(vals) != len(tree.labels):
        raise ValueError(f"{what} must have one value per motif")
    return vals


def plot_tree(
    tree: MotifTree,
    layout: str = "rectangular",
    labels: bool = True,
    groups=None,
    tip_size=None,
    tip_logos: bool = False,
    branch_length: bool = False,
    legend: bool = True,
    colours: Mapping[Any, str] | None = None,
    open_angle: float = 90,
    tip_logo_width: float | None = None,
    tip_logo_height: float = 0.9,
    text_size: float = 10,
    ax: Axes | None = None,
) -> Axes:
    if layout not in ("rectangular", "slanted", "circular", "fan"):
        raise ValueError("layout must be 'rectangular', 'slanted', 'circular' or 'fan'")
    n = len(tree.labels)
    x, y, children = _tree_coords(tree, branch_length)
    group_vals = _tree_values(tree, groups, "groups")
    size_vals = _tree_values(tree, tip_size, "tip_size")

    # A branch takes its group's colour when every leaf below it shares it.
    node_group: list[Any] = [None] * len(x)
    palette: dict[Any, str] = {}
    if group_vals is not None:
        import matplotlib as mpl

        uniq = [g for g in dict.fromkeys(group_vals) if g is not None]
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        palette = {g: (colours or {}).get(g, cycle[i % len(cycle)]) for i, g in enumerate(uniq)}
        below: list[set] = [{g} for g in group_vals] + [set() for _ in children]
        for i, (a, b) in enumerate(children):
            below[n + i] = below[a] | below[b]
        node_group = [next(iter(s)) if len(s) == 1 else None for s in below]

    def colour(node: int) -> str:
        g = node_group[node]
        return palette.get(g, "black") if g is not None else "black"

    if ax is None:
        _, ax = _pyplot().subplots(figsize=(6, max(3.0, 0.3 * n)) if layout in ("rectangular", "slanted")
                                   else (6, 6))
    segs, cols = [], []
    polar = layout in ("circular", "fan")
    span = 2 * math.pi - (math.radians(open_angle) if layout == "fan" else 0)
    start = math.pi / 2 + (math.radians(open_angle) / 2 if layout == "fan" else 0)

    def angle(yv: float) -> float:
        return start + span * (yv + 0.5) / n if layout == "circular" else \
            start + span * yv / max(n - 1, 1)

    def to_xy(r: float, t: float):
        return (r * math.cos(t), r * math.sin(t))

    for i, (a, b) in enumerate(children):
        p = n + i
        for c in (a, b):
            if layout == "rectangular":
                segs.append([(x[p], y[c]), (x[c], y[c])])
                segs.append([(x[p], y[p]), (x[p], y[c])])
                cols += [colour(c)] * 2
            elif layout == "slanted":
                segs.append([(x[p], y[p]), (x[c], y[c])])
                cols.append(colour(c))
            else:
                tc, tp = angle(y[c]), angle(y[p])
                segs.append([to_xy(x[p], tc), to_xy(x[c], tc)])
                arc = np.linspace(tp, tc, 30)
                segs.append([to_xy(x[p], t) for t in arc])
                cols += [colour(c)] * 2
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.0, capstyle="round"))

    tip_x = np.array([x[i] for i in range(n)])
    tip_y = np.array([y[i] for i in range(n)])
    xmax = float(tip_x.max()) if n else 0.0
    xrange = xmax - float(x.min()) or 1.0

    def area(v):
        return 15 + 85 * ((v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5))

    if size_vals is not None:
        vals = np.array([np.nan if v is None else float(v) for v in size_vals])
        finite = vals[np.isfinite(vals)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        scaled = area(vals)
        pts = [to_xy(tip_x[i], angle(tip_y[i])) if polar else (tip_x[i], tip_y[i]) for i in range(n)]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=np.nan_to_num(scaled, nan=0.0),
                   c=[colour(i) for i in range(n)], zorder=3)

    label_x = xmax + 0.02 * xrange
    if tip_logos:
        if polar:
            raise ValueError("tip logos need the rectangular or slanted layout")
        if tree.motifs is None:
            raise ValueError("tip logos need a tree made by motif_tree()")
        logos = view_motifs(tree.motifs, return_raw=True, sort_by="none")
        mats = list(logos.values())
        frame = mats[0].shape[1]
        alph = tree.motifs[0].alphabet
        rows = list(tree.motifs[0].letters)
        pos_width = tip_logo_width if tip_logo_width is not None else 0.6 * xrange / frame
        x0 = xmax + 0.08 * xrange
        for i, m in enumerate(mats):
            glyphs = motif_glyphs(m, rows, "ICM", 0.01, 0.04, 0.01, True, True, None, False)
            _draw_glyphs(ax, glyphs, _default_colours(alph, rows), x0=x0 - 0.5 * pos_width, y0=tip_y[i] - tip_logo_height / 2,
                         xscale=pos_width, yscale=tip_logo_height / 2)
        label_x = x0 + frame * pos_width + 0.02 * xrange

    texts = []
    if labels:
        for i, lab in enumerate(tree.labels):
            if polar:
                t = angle(tip_y[i])
                r = xmax + 0.03 * xrange
                deg = math.degrees(t) % 360
                left = 90 < deg < 270
                texts.append(ax.text(*to_xy(r, t), lab, rotation=deg - 180 if left else deg,
                                     rotation_mode="anchor", ha="right" if left else "left",
                                     va="center", fontsize=text_size))
            else:
                texts.append(ax.text(label_x, tip_y[i], lab, ha="left", va="center", fontsize=text_size))

    if legend and (palette or size_vals is not None):
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator

        if palette:
            handles = [Line2D([0], [0], color=c, lw=2) for c in palette.values()]
            groups_legend = ax.legend(handles, [str(g) for g in palette], frameon=False,
                                      loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.add_artist(groups_legend)
        if size_vals is not None and hi > lo:
            # A few round values across the range of tip sizes, drawn at the
            # sizes their tips would have.
            breaks = [b for b in MaxNLocator(nbins=4).tick_values(lo, hi) if lo <= b <= hi]
            handles = [Line2D([0], [0], linestyle="", marker="o", color="black",
                              markersize=math.sqrt(float(area(np.array(b))))) for b in breaks]
            title = None
            if isinstance(tip_size, str):
                title = f"-log10({tip_size})" if tip_size in ("pval", "qval", "eval") else tip_size
            ax.legend(handles, [f"{b:g}" for b in breaks], title=title, frameon=False,
                      loc="lower left", bbox_to_anchor=(1.0, 0.0), title_fontsize=text_size,
                      fontsize=text_size)

    if polar:
        r = xmax * 1.05 or 1.0
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
    else:
        ax.set_xlim(float(x.min()) - 0.02 * xrange, label_x)
        ax.set_ylim(-0.6, n - 0.4)
        if texts:
            # Widen the axes to take in the labels, so a legend beside them
            # does not cover them.
            renderer = ax.figure.canvas.get_renderer()
            right = max(t.get_window_extent(renderer).x1 for t in texts)
            ax.set_xlim(right=ax.transData.inverted().transform((right, 0))[0])
    ax.set_axis_off()
    return ax


def plot_motif_proximity(
    proximity: pl.DataFrame,
    motifs: Sequence[str] | None = None,
    ncol: int = 2,
    bins: int = 50,
    fill_zone: str = "#ffe4a3",
    log_x: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw each motif's distances to the nearest anchor, from
    :func:`motif_proximity`, with the most enriched zone shaded.

    Distances are signed (negative upstream of stranded anchors); with
    ``log_x`` the absolute distances plus one are drawn on a log scale.
    """
    if "dist_to_nearest" not in proximity.columns:
        raise ValueError("proximity has no dist_to_nearest column; was it made by motif_proximity()?")
    if proximity.height == 0:
        raise ValueError("proximity has no rows to plot")
    if motifs is not None:
        proximity = proximity.filter(pl.col("motif").is_in(list(motifs)))
        if proximity.height == 0:
            raise ValueError("none of the requested motifs are in proximity")
    plt = _pyplot()
    n = proximity.height
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (4.0 * ncol, 2.8 * nrow), squeeze=False)
    for k, row in enumerate(proximity.iter_rows(named=True)):
        ax = axes[k // ncol, k % ncol]
        d = np.array([x for x in (row["dist_to_nearest"] or []) if x is not None], dtype=float)
        d = d[np.isfinite(d)]
        if log_x:
            d = np.abs(d) + 1
        if row["best_dist"] is not None:
            lo, hi = (1.0, row["best_dist"] + 1.0) if log_x else (-row["best_dist"], row["best_dist"])
            ax.axvspan(lo, hi, color=fill_zone, alpha=0.5, linewidth=0)
        if d.size:
            edges = np.geomspace(1, d.max() * 1.01, bins + 1) if log_x and d.max() > 1 else bins
            ax.hist(d, bins=edges, color="black")
        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"{row['motif']}  (p = {row['pvalue']:.2g})", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Distance to nearest anchor (bp, log10)" if log_x
                      else "Signed distance to nearest anchor (bp)")
        ax.set_ylabel("Count")
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    fig.tight_layout()
    return fig


def _density(x: np.ndarray, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel density over the range of x, with R's bw.nrd0
    bandwidth (as ggplot2's geom_density)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.ptp(x) == 0:
        return np.array([x.mean() if x.size else 0.0]), np.array([0.0])
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 0.9 * (lo if lo > 0 else abs(x[0]) or 1.0) * x.size ** -0.2
    grid = np.linspace(x.min(), x.max(), n)
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(axis=1)
    return grid, dens / (x.size * bw * math.sqrt(2 * math.pi))


def plot_match_bkg(
    sequences: Sequence[str] | Mapping[str, str] | None = None,
    bkg: Sequence[str] | Mapping[str, str] | None = None,
    by: Sequence[str] | None = None,
    indices: pl.DataFrame | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare the composition of targets and their matched background
    from :func:`match_bkg`, as overlaid densities.

    Give the targets and the background (GC fraction and length are drawn),
    or the ``indices`` table from ``match_bkg(..., return_indices=True)``
    to also see any covariates. ``by`` picks the axes to draw.
    """
    labels = {"gc": "GC fraction", "length": "Sequence length"}
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    if indices is not None:
        axes_names = [c[len("target_"):] for c in indices.columns
                      if c.startswith("target_") and c != "target_i"
                      and f"universe_{c[len('target_'):]}" in indices.columns]
        if by is not None:
            axes_names = [a for a in axes_names if a in by]
        if not axes_names:
            raise ValueError("indices has no target_*/universe_* column pairs to draw")
        for a in axes_names:
            panels.append((labels.get(a, a), indices[f"target_{a}"].to_numpy(),
                           indices[f"universe_{a}"].to_numpy()))
    else:
        if sequences is None or bkg is None:
            raise ValueError("give the targets and the background, or indices")
        _, t = _sequence_pairs(sequences)
        _, b = _sequence_pairs(bkg)
        chosen = list(by) if by is not None else ["gc", "length"]
        if set(chosen) - {"gc", "length"}:
            raise ValueError("by must name 'gc' and/or 'length'")
        for a in chosen:
            if a == "gc":
                panels.append((labels[a], np.array(_core._gc_fraction(t)), np.array(_core._gc_fraction(b))))
            else:
                panels.append((labels[a], np.array([len(s) for s in t], float),
                               np.array([len(s) for s in b], float)))
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=figsize or (4.0 * len(panels), 3.2), squeeze=False)
    for ax, (label, tv, uv) in zip(axes[0], panels):
        for vals, name, colour in ((tv, "target", "#1f77b4"), (uv, "matched", "#ff7f0e")):
            if len(vals) and np.ptp(vals) == 0:
                # A constant has no density to draw; mark its value.
                ax.axvline(float(vals[0]), color=colour, linewidth=1.2, label=name)
                continue
            gx, gy = _density(vals)
            ax.fill_between(gx, gy, color=colour, alpha=0.25, linewidth=0)
            ax.plot(gx, gy, color=colour, linewidth=1.2, label=name)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Density")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=2)
    fig.tight_layout()
    return fig


__all__ = [
    "AA_COLOURS",
    "DNA_COLOURS",
    "RNA_COLOURS",
    "plot_match_bkg",
    "plot_motif_peaks",
    "plot_motif_proximity",
    "view_logo",
    "view_motifs",
]
