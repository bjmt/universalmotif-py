"""The seqlets of a TF-MoDISco pattern drawn as rows of contributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._plot import _MISSING_COLOUR, DNA_COLOURS, _nice_step, _pyplot, view_logo
from ._plot_modisco import _default_label

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LETTERS = "ACGT"


def _seqlet_scores(rows: pl.DataFrame) -> np.ndarray:
    """The scores of seqlets of one pattern as an array of shape (seqlets,
    4, width)."""
    col = rows["scores"]
    if isinstance(col.dtype, pl.Array):
        x = col.to_numpy()
    else:
        try:
            x = np.asarray(col.to_list(), dtype=np.float32)
        except ValueError:
            raise ValueError("the seqlets of a pattern must all have the same width") from None
    if x.ndim != 3 or x.shape[1] != len(_LETTERS):
        raise ValueError(f"scores must hold arrays of shape (4, width), not {x.shape[1:]}")
    return np.array(x, dtype=np.float32)


def _letter_codes(match: pl.Series, width: int) -> np.ndarray:
    """Letters as codes 0 to 3 for ACGT and 4 for anything else, of shape
    (seqlets, width)."""
    text = match.cast(pl.String).fill_null("").str.to_uppercase()
    if (text.str.len_bytes() != width).any():
        raise ValueError(f"match must hold {width} letters for each seqlet, as wide as its scores")
    lookup = np.full(256, 4, dtype=np.uint8)
    for i, letter in enumerate(_LETTERS):
        lookup[ord(letter)] = i
    raw = np.frombuffer("".join(text.to_list()).encode("ascii", "replace"), dtype=np.uint8)
    return lookup[raw].reshape(-1, width)


def plot_seqlets(
    seqlets: pl.DataFrame,
    pattern: str,
    sort_by: str | None = "score",
    max_seqlets: int | None = 500,
    letters: bool = True,
    logo: bool = True,
    vmax: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw the seqlets of a TF-MoDISco pattern, one row each, coloured by
    how much each letter contributes.

    Each row is a seqlet read along the pattern, so seqlets on the ``-``
    strand are turned around, and each position is coloured by the
    contribution of the letter there on a scale centred on zero, red for
    positive and blue for negative. A second panel beside it shows the
    letters themselves in the package's DNA colours, which makes shifted or
    reversed seqlets easy to spot. Above the panels are the pattern's
    contribution weight matrix, the average contribution of the letters of
    all its seqlets, and the information content of those letters as a
    sequence logo.

    Needs the ``plot`` extra.

    Parameters
    ----------
    seqlets
        Seqlets read with [`read_seqlets()`][universalmotif.read_seqlets]
        with ``scores``, either hypothetical or projected, and their
        ``match`` column. Only the rows of ``pattern`` are drawn.
    pattern
        The name of the pattern, as in the ``motif`` column, or its short
        label such as ``pos_3`` for ``pos_patterns.<run>_pattern_3``, as
        [`plot_modisco()`][universalmotif.plot_modisco] labels patterns.
    sort_by
        A numeric column to order the rows by, from the largest absolute
        value at the top, so that the strongest seqlets of both positive
        and negative patterns come first. The default, ``"score"``, is the
        total contribution of each seqlet. ``None`` keeps the order of the
        table.
    max_seqlets
        Draw at most this many rows, picked evenly along the sorted
        seqlets so that the whole range is shown, or all of them with
        ``None``. The logos are always made from every seqlet, and the
        title gives how many were drawn.
    letters, logo
        Draw the panel of letters, and the logos above the panels.
    vmax
        The contribution at either end of the colour scale. By default it
        is the 99.5th percentile of the absolute contributions drawn, so
        that a few extreme positions do not wash out the rest.
    title
        The title above the plot, by default the short label of the
        pattern, such as ``pos_3``. The name of the factor often reads
        better.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> match = ["CTGATAAGGT", "ATGATAAGCA", "GAGATAAGTC", "TTGATAACCA"]
    >>> onehot = np.array([[[c == b for c in m] for b in "ACGT"] for m in match])
    >>> scores = (rng.normal(0, 0.05, (4, 4, 10)) + 0.5 * onehot).astype(np.float32)
    >>> table = pl.DataFrame({"motif": ["GATA"] * 4, "strand": ["+"] * 4, "match": match,
    ...                       "score": (scores * onehot).sum(axis=(1, 2)), "scores": scores})
    >>> fig = um.plot_seqlets(table, "GATA")
    >>> [ax.get_ylabel() for ax in fig.axes]
    ['Seqlet, sorted by score', '', 'Contribution', 'Bits']
    >>> plt.close(fig)
    """
    for col in ("motif", "match"):
        if col not in seqlets.columns:
            raise ValueError(f"seqlets has no {col!r} column; read them with read_seqlets()")
    if "scores" not in seqlets.columns:
        raise ValueError('seqlets has no scores column; read them with read_seqlets(..., '
                         'scores="hypothetical") or scores="projected"')
    if max_seqlets is not None and max_seqlets < 1:
        raise ValueError("max_seqlets must be 1 or more, or None for every seqlet")
    motif = seqlets["motif"].cast(pl.String)
    rows = seqlets.filter(motif == pattern)
    if rows.height == 0:
        full = [m for m in motif.unique(maintain_order=True) if _default_label(m) == pattern]
        if len(full) == 1:
            rows = seqlets.filter(motif == full[0])
    if rows.height == 0:
        names = seqlets["motif"].cast(pl.String).unique(maintain_order=True).to_list()
        shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        raise ValueError(f"no seqlets of a pattern named {pattern!r}; the table has {shown or 'none'}")
    x = _seqlet_scores(rows)
    n, _, width = x.shape
    codes = _letter_codes(rows["match"], width)
    # Scores run along the genome, but the letters along the pattern.
    if "strand" in rows.columns:
        minus = (rows["strand"].cast(pl.String) == "-").fill_null(False).to_numpy()
        x[minus] = x[minus, ::-1, ::-1]
    onehot = (codes[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32)
    observed = x * onehot
    contrib = observed.sum(axis=1)

    if sort_by is None:
        order = np.arange(n)
    else:
        if sort_by in rows.columns:
            if not rows[sort_by].dtype.is_numeric():
                raise ValueError("sort_by must name a numeric column")
            key = rows[sort_by].cast(pl.Float64).fill_null(0.0).to_numpy()
        elif sort_by == "score":
            key = contrib.sum(axis=1, dtype=np.float64)
        else:
            raise ValueError(f"seqlets has no {sort_by!r} column to sort by")
        order = np.argsort(-np.abs(key), kind="stable")
    if max_seqlets is not None and n > max_seqlets:
        order = order[np.round(np.linspace(0, n - 1, max_seqlets)).astype(np.int64)]
    shown = contrib[order]
    if vmax is None:
        vmax = float(np.percentile(np.abs(shown), 99.5)) if shown.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    plt = _pyplot()
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    ncol = 2 if letters else 1
    if figsize is None:
        panel = min(max(2.4, 0.1 * width + 0.8), 6.0)
        figsize = (ncol * (panel + 0.9) + 0.8, 6.2 if logo else 5.2)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2 if logo else 1, ncol, height_ratios=[1, 5] if logo else None,
                            wspace=0.45, hspace=0.08, left=1.0 / figsize[0],
                            right=1 - (0.75 if letters else 1.0) / figsize[0],
                            top=1 - 0.55 / figsize[1], bottom=0.5 / figsize[1])
    extent = (0.5, width + 0.5, n, 0)
    heat = fig.add_subplot(grid[-1, 0])
    im = heat.imshow(shown, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     interpolation="antialiased", extent=extent)
    heat.set_ylabel("Seqlet" if sort_by is None else f"Seqlet, sorted by {sort_by}")
    heat.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bar = fig.colorbar(im, cax=heat.inset_axes((1.04, 0.0, 0.04, 0.4)))
    bar.set_label("Contribution", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    panels = [heat]
    if letters:
        colours = np.array([to_rgb(DNA_COLOURS[c]) for c in _LETTERS] + [to_rgb(_MISSING_COLOUR)])
        lets = fig.add_subplot(grid[-1, 1], sharey=heat)
        # Blending neighbouring rows would mix the letter colours into new ones.
        lets.imshow(colours[codes[order]], aspect="auto", interpolation="nearest", extent=extent)
        lets.tick_params(axis="y", left=False, labelleft=False)
        lets.spines["left"].set_visible(False)
        if not logo:
            keys = [Patch(facecolor=DNA_COLOURS[c], label=c) for c in _LETTERS]
            lets.legend(handles=keys, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False,
                        fontsize=8, handlelength=0.8, handleheight=0.8, columnspacing=0.8,
                        borderaxespad=0.2)
        panels.append(lets)
    step = _nice_step(width, 6)
    ticks = ([1] if step >= 5 else []) + list(range(step, width + 1, step))
    for ax in panels:
        ax.set_xticks(ticks)
        ax.set_xlabel("Position")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    logos = []
    if logo:
        logos.append(fig.add_subplot(grid[0, 0], sharex=heat))
        _logo_panel(logos[0], observed.mean(axis=0), "Contribution")
        if letters:
            counts = onehot.sum(axis=0)
            ppm = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ic = 2.0 + np.where(ppm > 0, ppm * np.log2(ppm), 0.0).sum(axis=0)
            logos.append(fig.add_subplot(grid[0, 1], sharex=lets))
            _logo_panel(logos[1], ppm * ic, "Bits", (0.0, 2.0))
        for ax in logos:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for ax in panels:
        ax.set_xlim(0.5, width + 0.5)
    heat.set_ylim(n, 0)
    first = logos[0] if logos else heat
    count = f"{n:,} seqlets" if n != 1 else "1 seqlet"
    if len(order) < n:
        count += f", {len(order):,} drawn"
    heading = first.set_title(_default_label(rows["motif"][0]) if title is None else title, loc="left", fontsize=10, pad=16)
    # Pattern names can be long, so the title is made smaller to fit.
    room = fig.bbox.x1 - heading.get_window_extent(fig.canvas.get_renderer()).x0 - 4
    wide = heading.get_window_extent(fig.canvas.get_renderer()).width
    if wide > room > 0:
        heading.set_fontsize(max(6.0, 10 * room / wide))
    first.annotate(count, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                   ha="left", va="bottom", fontsize=8, color="0.35")
    return fig


def _logo_panel(ax, matrix: np.ndarray, label: str, limits: tuple[float, float] | None = None) -> None:
    from matplotlib.ticker import MaxNLocator

    hi = float(np.clip(matrix, 0, None).sum(axis=0).max())
    lo = float(np.clip(matrix, None, 0).sum(axis=0).min())
    scale = max(hi, -lo) or 1.0
    view_logo(matrix, letters=list(_LETTERS), colour_scheme=DNA_COLOURS, sort_positions=True,
              min_height=1e-6 * scale, y_spacer=0.01 * scale, ax=ax)
    ax.set_axis_on()
    if limits is not None:
        lo, hi = limits
    pad = 0.05 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(lo, hi)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    if limits is not None:
        # Fixed limits, such as the 2 bits of DNA, are marked at both ends.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2))


__all__ = ["plot_seqlets"]
