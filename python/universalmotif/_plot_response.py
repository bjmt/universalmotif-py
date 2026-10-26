"""Plots of a sequence model's response to sites written into sequences."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._deep import _to_numpy
from ._plot import _pyplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LINE = "#255C99"
_BEFORE = "#A3A3A3"
_ZERO = "#8C8C8C"
_GUIDE = "#D4D4D4"
_MUTED = "#5E5E5E"
# A fixed order of hues that stay apart for the common forms of colour
# blindness, for several outputs.
_PALETTE = ("#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948")
_HELIX = 10.5
# The 97.5% quantile of Student's t for 1 to 30 degrees of freedom.
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042)


def _t975(df: int) -> float:
    if df <= len(_T975):
        return _T975[df - 1]
    # A series in 1 / df around the normal quantile, close to four decimals
    # past 30 degrees of freedom.
    z = 1.959964
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def _predictions(x: Any, what: str) -> np.ndarray:
    """Predictions as an array of shape (sequences, outputs)."""
    try:
        a = _to_numpy(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} must be numbers") from e
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{what} must have shape (sequences,) or (sequences, outputs), not {a.shape}; "
                         "summarize outputs such as profiles to one value per sequence first")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{what} is empty")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} has missing or infinite values")
    return a


def _output_names(names: Sequence[str] | None, k: int) -> list[str] | None:
    if names is None:
        return None
    names = [names] if isinstance(names, str) else [str(n) for n in names]
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} outputs")
    return names


def _output_index(output: Any, k: int, names: list[str] | None) -> int:
    if isinstance(output, str):
        if names is None or output not in names:
            raise ValueError(f"there is no output named {output!r}" + ("; give the outputs names" if names is None
                                                                      else ""))
        return names.index(output)
    if isinstance(output, bool) or not isinstance(output, numbers.Integral):
        raise TypeError("output must be the index or name of an output")
    if not -k <= output < k:
        raise ValueError(f"output {output} is out of range for {k} outputs")
    return int(output) % k


def _colours(colour: str | Sequence[str] | None, n: int) -> list[str]:
    if colour is None:
        if n == 1:
            return [_LINE]
        if n > len(_PALETTE):
            raise ValueError(f"{n} outputs are too many to tell apart by colour; pick some with output, "
                             "or give a colour for each")
        return list(_PALETTE[:n])
    if isinstance(colour, str):
        return [colour] * n
    colour = list(colour)
    if len(colour) != n:
        raise ValueError(f"{len(colour)} colours for {n} outputs")
    return colour


def _spread(values: np.ndarray, band: str | None) -> np.ndarray:
    """Half the width of the band around the mean of each column."""
    n = values.shape[0]
    if band is None or n < 2:
        return np.full(values.shape[1], np.nan)
    sd = values.std(axis=0, ddof=1)
    if band == "sd":
        return sd
    se = sd / math.sqrt(n)
    return se if band == "se" else _t975(n - 1) * se


def _tidy(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_spacing(
    predictions: Mapping[int, Any] | Any,
    baseline: Any = None,
    spacings: Sequence[int] | None = None,
    output: int | str | Sequence[int | str] | None = None,
    names: Sequence[str] | None = None,
    band: str | None = "ci",
    helix: bool | float = False,
    colour: str | Sequence[str] | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw a model's response to the spacing between two sites.

    This is the plot for predictions on the sequences made by
    [`site_spacings()`][universalmotif.site_spacings], which writes two
    sites into the same background sequences at a range of spacings. The
    mean prediction at each spacing is drawn as a line with points, inside
    a band that shows how certain or how variable it is. With ``baseline``,
    the model's predictions for the background sequences before any sites
    were written, each sequence's prediction is taken relative to its own
    baseline, so that the line is the mean change the pair of sites brings
    about, with a line at zero for no change. This is the usual way to show
    such experiments, since it removes the part of each prediction that
    comes from the background.

    A model with several outputs, such as one per cell type or assay, gets
    one line per output in a fixed order of colours, with a legend.

    Needs the ``plot`` extra.

    Parameters
    ----------
    predictions
        The model's predictions, as a dict from each spacing to the
        predictions for its sequences, the form
        [`site_spacings()`][universalmotif.site_spacings] returns with the
        sequences replaced by predictions. Each holds one value per
        sequence, or an array of shape (sequences, outputs). An array of
        shape (spacings, sequences) or (spacings, sequences, outputs) can be
        given instead, along with ``spacings``. NumPy arrays, lists and
        tensors all work.
    baseline
        Predictions for the background sequences without sites, in the same
        order and of the same shape as those for each spacing, or a single
        number to subtract from every prediction.
    spacings
        The spacing of each row of ``predictions`` when it is an array.
    output
        The index or name of the output to draw, or a list of them, for
        models with several outputs. All outputs are drawn by default.
    names
        Names of the outputs, for the legend and for picking them with
        ``output``.
    band
        What the band around the mean shows: ``"ci"`` for its 95%
        confidence interval, ``"se"`` for one standard error either side,
        ``"sd"`` for one standard deviation of the predictions either side,
        which shows how much the response varies between backgrounds, or
        ``None`` for no band.
    helix
        Mark the period of the DNA double helix, 10.5 bp, with faint
        vertical lines, to judge whether the response follows it. The lines
        pass through the spacing given, as one at which the two sites lie
        on the same face of the helix, or with ``True`` through the spacing
        of the strongest mean response.
    colour
        A colour for all lines, or one per output drawn.
    text_size
        The size of the legend text.
    figsize
        The size of a new figure, in inches.
    ax
        Axes to draw into. They are returned, and otherwise a new figure is.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    A stand-in for a model scores the GC content of a sequence and rewards
    a GATA site and an E-box, more so when their centres are a whole number
    of helical turns apart:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> def model(seqs):
    ...     out = []
    ...     for s in seqs:
    ...         a, b = s.find("GATAAG"), s.find("CACGTG")
    ...         gc = (s.count("G") + s.count("C")) / len(s)
    ...         pair = 1 + np.cos(2 * np.pi * (b - a) / 10.5) if 0 <= a < b else 0
    ...         out.append(gc + 0.5 * (a >= 0) + 0.5 * (b >= 0) + pair)
    ...     return np.array(out)
    >>> bkg = um.create_sequences(seqnum=50, seqlen=80, seed=1)
    >>> pairs = um.site_spacings(bkg, "GATAAG", "CACGTG", spacings=range(31))
    >>> preds = {d: model(seqs) for d, seqs in pairs.items()}
    >>> fig = um.plot_spacing(preds, baseline=model(bkg), helix=True)

    The mean change peaks at spacings a helical turn apart:

    >>> line = fig.axes[0].lines[-1]
    >>> [int(d) for d, y in zip(*line.get_data()) if y > 2.85]
    [4, 5, 15, 25, 26]
    >>> plt.close(fig)

    The same predictions as an array, drawn into a panel of a larger
    figure:

    >>> fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    >>> stacked = np.stack([preds[d] for d in range(31)])
    >>> ax = um.plot_spacing(stacked, spacings=range(31), band="sd", ax=axes[1])
    >>> ax is axes[1]
    True
    >>> plt.close(fig)
    """
    if band not in ("ci", "se", "sd", None):
        raise ValueError("band must be 'ci', 'se', 'sd' or None")
    if isinstance(predictions, Mapping):
        if spacings is not None:
            raise ValueError("spacings are taken from the keys of predictions, so give them only with an array")
        try:
            xs = [int(d) for d in predictions]
        except (TypeError, ValueError) as e:
            raise ValueError("the keys of predictions must be spacings, as whole numbers") from e
        arrays = [_predictions(v, f"the predictions at spacing {d}") for d, v in zip(xs, predictions.values())]
    else:
        if spacings is None:
            raise ValueError("give the spacings of the rows of predictions, or predictions as a dict "
                             "from spacing to predictions")
        stack = _to_numpy(predictions)
        xs = [int(d) for d in spacings]
        if stack.ndim not in (2, 3):
            raise ValueError("an array of predictions must have shape (spacings, sequences) or "
                             f"(spacings, sequences, outputs), not {stack.shape}")
        if stack.shape[0] != len(xs):
            raise ValueError(f"{len(xs)} spacings for {stack.shape[0]} rows of predictions")
        arrays = [_predictions(v, f"the predictions at spacing {d}") for d, v in zip(xs, stack)]
    if not xs:
        raise ValueError("predictions has no spacings")
    if len(set(xs)) != len(xs):
        raise ValueError("the spacings must all differ")
    k = arrays[0].shape[1]
    if any(a.shape[1] != k for a in arrays):
        raise ValueError("the predictions at every spacing must have the same number of outputs")
    if baseline is not None:
        if np.ndim(_to_numpy(baseline)) == 0:
            base = float(_to_numpy(baseline))
            if not math.isfinite(base):
                raise ValueError("baseline must be finite")
        else:
            base = _predictions(baseline, "baseline")
            if base.shape[1] != k:
                raise ValueError(f"baseline has {base.shape[1]} outputs and predictions {k}")
            for d, a in zip(xs, arrays):
                if a.shape[0] != base.shape[0]:
                    raise ValueError(f"{a.shape[0]} predictions at spacing {d} for {base.shape[0]} baseline "
                                     "sequences; they must be the same sequences in the same order")
        arrays = [a - base for a in arrays]
    labels = _output_names(names, k)
    if output is None:
        which = list(range(k))
    elif isinstance(output, str) or np.ndim(output) == 0:
        which = [_output_index(output, k, labels)]
    else:
        which = [_output_index(o, k, labels) for o in output]
        if not which:
            raise ValueError("output names no outputs")
    colours = _colours(colour, len(which))

    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    arrays = [arrays[i] for i in order]
    means = np.stack([a.mean(axis=0) for a in arrays])
    half = np.stack([_spread(a, band) for a in arrays])

    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    own_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6.0, 3.6))
    else:
        fig = ax.figure
    pad = max(0.6, 0.02 * (x[-1] - x[0]))
    ax.set_xlim(x[0] - pad, x[-1] + pad)
    if baseline is not None:
        ax.axhline(0, color=_ZERO, linewidth=0.8, zorder=1)
    if helix is not False and helix is not None:
        if helix is True:
            anchor = x[int(np.argmax(np.abs(means[:, which[0]])))]
        else:
            anchor = float(helix)
        first = anchor - math.floor((anchor - x[0] + pad) / _HELIX) * _HELIX
        for g in np.arange(first, x[-1] + pad, _HELIX):
            ax.axvline(g, color=_GUIDE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    marker = 4.5 if len(x) <= 30 else 3.2 if len(x) <= 70 else 0
    several = len(which) > 1
    for j, col in zip(which, colours):
        name = labels[j] if labels is not None else f"output {j}"
        if np.isfinite(half[:, j]).any():
            ax.fill_between(x, means[:, j] - half[:, j], means[:, j] + half[:, j], color=col,
                            alpha=0.16 if several else 0.22, linewidth=0, zorder=2)
        ax.plot(x, means[:, j], color=col, linewidth=1.6, marker="o" if marker else None, markersize=marker,
                markeredgecolor="white", markeredgewidth=0.6, label=name, zorder=3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.set_xlabel("Spacing between sites (bp)")
    ax.set_ylabel("Change in prediction" if baseline is not None else "Prediction")
    _tidy(ax)
    if several or labels is not None:
        where = {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0)} if own_figure else {"loc": "best"}
        ax.legend(frameon=False, fontsize=text_size, handlelength=1.6, **where)
    if own_figure:
        fig.tight_layout()
    return fig if own_figure else ax


def _density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density with Scott's bandwidth, from a histogram of
    fine bins so that large samples cost no more than small ones."""
    n = values.size
    bw = values.std(ddof=1) * n ** (-1 / 5)
    lo, hi = grid[0], grid[-1]
    counts, edges = np.histogram(values, bins=512, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    z = (grid[:, None] - centres[None, :]) / bw
    return np.exp(-0.5 * z * z) @ counts


def _violin(ax: Axes, values: np.ndarray, at: float, width: float, colour: str) -> None:
    lo, hi = float(values.min()), float(values.max())
    if hi > lo and values.std(ddof=1) > 0:
        grid = np.linspace(lo, hi, 200)
        dens = _density(values, grid)
        dens = width * dens / dens.max()
        ax.fill_betweenx(grid, at - dens, at + dens, facecolor=colour, alpha=0.35, edgecolor="none", zorder=2)
        ax.plot(np.concatenate([at - dens, (at + dens)[::-1], [at - dens[0]]]),
                np.concatenate([grid, grid[::-1], [grid[0]]]), color=colour, linewidth=0.9, zorder=2)
    else:
        ax.plot([at - width, at + width], [lo, lo], color=colour, linewidth=1.2, zorder=2)
    q1, q3 = np.percentile(values, [25, 75])
    reach = 1.5 * (q3 - q1)
    w_lo = float(values[values >= q1 - reach].min())
    w_hi = float(values[values <= q3 + reach].max())
    ax.plot([at, at], [w_lo, w_hi], color="#333333", linewidth=0.9, solid_capstyle="butt", zorder=4)
    ax.plot([at, at], [q1, q3], color="#333333", linewidth=4.5, solid_capstyle="butt", zorder=4)


def _box(ax: Axes, values: np.ndarray, at: float, width: float, colour: str) -> None:
    from matplotlib.colors import to_rgba

    ax.boxplot(values, positions=[at], widths=width, showfliers=False, patch_artist=True,
               manage_ticks=False, zorder=2,
               boxprops={"facecolor": to_rgba(colour, 0.35), "edgecolor": colour, "linewidth": 0.9},
               medianprops={"color": colour, "linewidth": 1.6},
               whiskerprops={"color": "#333333", "linewidth": 0.9},
               capprops={"color": "#333333", "linewidth": 0.9})


def plot_marginalization(
    before: Any,
    after: Any | Mapping[str, Any],
    output: int | str | None = None,
    names: Sequence[str] | None = None,
    kind: str = "violin",
    points: bool | None = None,
    lines: bool | None = None,
    colour: str = _LINE,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw how writing sites into sequences changes a model's predictions.

    This is the plot for predictions on the sequences made by
    [`insert_sites()`][universalmotif.insert_sites], which writes the same
    sites into every one of a set of background sequences, compared with
    the predictions for the same sequences before. The distribution of the
    predictions before is drawn in grey next to that after, in colour, each
    as a violin or a box marking the quartiles, with a white dot at the
    mean, and a faint line across the plot at the mean before. Above each
    distribution after insertion is the mean change in prediction, taken
    sequence by sequence, and the share of sequences whose prediction went
    up.

    Several insertions can be compared against the same background, for
    example different motifs or variants of one motif, by giving ``after``
    as a dict from labels to predictions, which draws one distribution for
    each next to the shared one before. For a single insertion and a few
    sequences, thin lines join each sequence's predictions before and
    after.

    Needs the ``plot`` extra.

    Parameters
    ----------
    before
        The model's predictions for the background sequences, one value per
        sequence or an array of shape (sequences, outputs). NumPy arrays,
        lists and tensors all work.
    after
        Predictions for the same sequences, in the same order, after
        inserting the sites, or a dict from labels to such predictions.
    output
        The index or name of the output to draw, needed when the model has
        several.
    names
        Names of the outputs, for picking one with ``output``. The name of
        the output drawn labels the y axis.
    kind
        ``"violin"`` for violins with the quartiles drawn inside them, or
        ``"box"`` for boxes, which are easier to read for skewed or very
        narrow distributions. Whiskers reach the furthest values within 1.5
        times the interquartile range.
    points
        Draw every sequence as a point. By default only for up to 300
        sequences.
    lines
        Join each sequence's predictions before and after with a line. By
        default only for a single insertion and up to 50 sequences; lines
        need a single insertion.
    colour
        The colour of the distributions after insertion.
    text_size
        The size of the text above each distribution.
    figsize
        The size of a new figure, in inches.
    ax
        Axes to draw into. They are returned, and otherwise a new figure is.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    A stand-in for a model scores the GC content of a sequence and rewards
    a GATA site and an E-box, more so together:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> def model(seqs):
    ...     out = []
    ...     for s in seqs:
    ...         a, b = s.find("GATAAG"), s.find("CACGTG")
    ...         gc = (s.count("G") + s.count("C")) / len(s)
    ...         out.append(gc + 0.5 * (a >= 0) + 0.5 * (b >= 0) + (a >= 0 and b >= 0))
    ...     return np.array(out)
    >>> bkg = um.create_sequences(seqnum=200, seqlen=80, seed=1)
    >>> after = {
    ...     "GATA": model(um.insert_sites(bkg, "GATAAG")),
    ...     "E-box": model(um.insert_sites(bkg, "CACGTG")),
    ...     "both": model(um.insert_sites(bkg, ["GATAAG", "CACGTG"], positions=[30, 44])),
    ... }
    >>> fig = um.plot_marginalization(model(bkg), after)
    >>> [t.get_text() for t in fig.axes[0].get_xticklabels()]
    ['Before', 'GATA', 'E-box', 'both']
    >>> plt.close(fig)
    """
    if kind not in ("violin", "box"):
        raise ValueError("kind must be 'violin' or 'box'")
    base = _predictions(before, "before")
    if isinstance(after, Mapping):
        if not after:
            raise ValueError("after has no predictions")
        groups = {str(lab): _predictions(v, f"the predictions after {lab!r}") for lab, v in after.items()}
    else:
        groups = {"After": _predictions(after, "after")}
    for lab, a in groups.items():
        if a.shape != base.shape:
            what = "after" if lab == "After" and not isinstance(after, Mapping) else f"the predictions after {lab!r}"
            raise ValueError(f"{what} has shape {a.shape} and before {base.shape}; they must be the same "
                             "sequences in the same order")
    n, k = base.shape
    labels = _output_names(names, k)
    if output is None:
        if k > 1:
            raise ValueError(f"the predictions have {k} outputs; pick one with output")
        j = 0
    else:
        j = _output_index(output, k, labels)
    if lines and len(groups) > 1:
        raise ValueError("lines join predictions before and after a single insertion")
    if points is None:
        points = n <= 300
    if lines is None:
        lines = len(groups) == 1 and n <= 50
    b = base[:, j]
    shown = [b] + [a[:, j] for a in groups.values()]

    plt = _pyplot()
    own_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(3.4, 1.5 + 1.15 * len(shown)), 4.0))
    else:
        fig = ax.figure
    width = 0.36 if kind == "violin" else 0.5
    draw = _violin if kind == "violin" else _box
    ax.axhline(float(b.mean()), color=_GUIDE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-1, 1, n) * (0.08 if lines else 0.14)
    for i, (vals, col) in enumerate(zip(shown, [_BEFORE] + [colour] * len(groups))):
        draw(ax, vals, i, width, col)
        if points:
            ax.scatter(i + jitter, vals, s=7 if n <= 100 else 4, color=col, alpha=0.75 if n <= 100 else 0.5,
                       linewidths=0, zorder=3)
    if lines:
        from matplotlib.collections import LineCollection

        segs = np.stack([np.column_stack([jitter, b]), np.column_stack([1 + jitter, shown[1]])], axis=1)
        ax.add_collection(LineCollection(segs, colors="#9A9A9A", linewidths=0.5, alpha=0.45, zorder=2.5))
    for i, vals in enumerate(shown):
        ax.scatter([i], [vals.mean()], s=26, color="white", edgecolors="#333333", linewidths=0.8, zorder=5)

    from matplotlib.transforms import blended_transform_factory

    on_top = blended_transform_factory(ax.transData, ax.transAxes)
    changes = [vals - b for vals in shown[1:]]
    for i, (diff, text) in enumerate(zip(changes, _signed([float(c.mean()) for c in changes])), start=1):
        ax.annotate(f"{100 * float((diff > 0).mean()):.0f}% up", xy=(i, 1.0), xycoords=on_top,
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=text_size - 1, color=_MUTED, annotation_clip=False)
        ax.annotate(f"Δ {text}", xy=(i, 1.0), xycoords=on_top,
                    xytext=(0, 2 + 1.3 * text_size), textcoords="offset points", ha="center", va="bottom",
                    fontsize=text_size, color="#1A1A1A", annotation_clip=False)
    ax.set_xticks(range(len(shown)), ["Before", *groups])
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(f"{n:,} sequences", color=_MUTED, fontsize=text_size)
    ax.set_ylabel(labels[j] if labels is not None else "Prediction")
    _tidy(ax)
    if own_figure:
        fig.tight_layout()
    return fig if own_figure else ax


def _signed(values: Sequence[float]) -> list[str]:
    """Changes with their signs, all to the same number of decimals, enough
    to give the largest three significant figures."""
    top = max(abs(v) for v in values)
    places = min(max(0, 2 - math.floor(math.log10(top))), 8) if top > 0 else 0
    out = []
    for v in values:
        s = f"{v:+.{places}f}"
        out.append("0" if float(s) == 0 else s.replace("-", "\u2212"))
    return out


__all__ = ["plot_marginalization", "plot_spacing"]
