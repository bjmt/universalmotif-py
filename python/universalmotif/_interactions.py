"""Pairwise interactions between the positions of a sequence: inputs for in
silico double mutagenesis, the epistasis between pairs of substitutions, and
maps of any square matrix of interactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._deep import _mutagenesis_region, _substitutions, _to_numpy, one_hot
from ._logo import matrix_glyphs
from ._plot import (
    _ALPHABET_NAMES,
    _HIT_COLOUR,
    _MISSING_COLOUR,
    _attribution_matrix,
    _default_colours,
    _hit_spans,
    _nice_step,
    _pyplot,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Letters swapped by each rule, as pairs; U stands in for T in RNA.
_RULES = {
    "complement": (("A", "T"), ("C", "G")),
    "transition": (("A", "G"), ("C", "T")),
}
_SUMMARIES = ("mean", "strongest", "min", "max")


def _pair_substitutions(sequence: str, alphabet: str, region, alt) -> tuple[int, int, np.ndarray,
                                                                               np.ndarray, bool]:
    """The region and the substitutions made in it, as positions and letter
    indices in order, and whether there is exactly one per position."""
    a, b = _mutagenesis_region(sequence, region)
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("the alphabet has repeated letters")
    ref = one_hot(sequence, alphabet=alphabet, dtype=np.int8)[0]
    if isinstance(alt, str) and alt == "all":
        pos, let = _substitutions(ref, a, b)
        return a, b, pos, let, False
    index = {c.upper(): i for i, c in enumerate(alphabet)}
    if alt is None or (isinstance(alt, str) and alt in _RULES):
        rule = "complement" if alt is None else alt
        upper = alphabet.upper()
        if sorted(upper) not in (sorted("ACGT"), sorted("ACGU")):
            raise ValueError(f'the {rule} rule is for DNA and RNA; give alt as one letter per position, '
                             'or "all"')
        u = "U" if "U" in upper else "T"
        swap = {}
        for x, y in _RULES[rule]:
            x, y = x.replace("T", u), y.replace("T", u)
            swap[x], swap[y] = y, x
        # Letters outside the alphabet, such as N, take its first letter.
        letters = [swap.get(c.upper(), upper[0]) for c in sequence[a:b]]
    else:
        letters = list(alt)
        if len(letters) != b - a:
            raise ValueError(f"alt must have one letter per position of the region, {b - a} here, "
                             f"not {len(letters)}")
    let = np.empty(b - a, dtype=np.intp)
    for k, c in enumerate(letters):
        c = str(c)
        i = index.get(c.upper()) if len(c) == 1 else None
        if i is None:
            raise ValueError(f"alt has {c!r} at position {a + k}, which is not a letter of {alphabet}")
        if ref[i, a + k]:
            raise ValueError(f"alt has the sequence's own letter at position {a + k}")
        let[k] = i
    return a, b, np.arange(a, b), let, True


def _pair_rows(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two substitutions of each double mutant, every pair at different
    positions in order."""
    i, j = np.triu_indices(len(pos), 1)
    keep = pos[i] != pos[j]
    return i[keep], j[keep]


def _pair_batch(ref: np.ndarray, pos, let, first, second, r0: int, r1: int) -> np.ndarray:
    """Rows ``r0`` to ``r1`` of the batch: the sequence, then each single
    substitution, then each pair."""
    rows = np.arange(r0, r1)
    out = np.repeat(ref, r1 - r0, axis=0)
    s = len(pos)
    single = np.nonzero((rows >= 1) & (rows <= s))[0]
    double = np.nonzero(rows > s)[0]
    k = rows[single] - 1
    d = rows[double] - 1 - s
    for idx, sub in ((single, k), (double, first[d]), (double, second[d])):
        out[idx, :, pos[sub]] = 0
        out[idx, let[sub], pos[sub]] = 1
    return out


class _PairBatches:
    """The double-mutagenesis batch, made a part at a time as it is needed."""

    def __init__(self, ref, pos, let, first, second, batch_size: int):
        self._args = (ref, pos, let, first, second)
        self._total = 1 + len(pos) + len(first)
        self._batch_size = batch_size

    def __len__(self) -> int:
        return (self._total + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        for r0 in range(0, self._total, self._batch_size):
            yield _pair_batch(*self._args, r0, min(r0 + self._batch_size, self._total))

    def __repr__(self) -> str:
        return f"<{len(self)} batches of up to {self._batch_size} sequences, {self._total} in all>"


def pair_mutagenesis(
    sequence: str,
    alphabet: str = "ACGT",
    region: tuple[int, int] | None = None,
    alt: str | Sequence[str] | None = None,
    batch_size: int | None = None,
    dtype: Any = np.float32,
):
    """Single and double substitutions of a sequence, one-hot encoded for a
    model, for measuring which positions it treats as interacting.

    The batch starts with the sequence itself, followed by the sequence
    with each single substitution, in order of position, and then by the
    sequence with each pair of substitutions at different positions, the
    first substitution of each pair before the second and the pairs in the
    order of their first and then their second. A model's predictions for
    the batch, in the same order, are turned into a matrix of interactions
    between positions by
    [`interaction_effects()`][universalmotif.interaction_effects].
    ``region`` is a (start, end) pair counted from 0 with the end left out,
    and is the whole sequence by default.

    Every pair of positions needs its own sequence, so by default each
    position gets a single substitution, its complement, which swaps A with
    T and C with G. This always changes the letter, and breaks the sites of
    most motifs while keeping the GC content of the sequence, so that a
    model's response to composition is not mistaken for an interaction.
    ``alt="transition"`` swaps A with G and C with T instead, and ``alt``
    can also give one letter for each position of the region, as a string
    or a list, such as the letter with the strongest effect at each
    position in [`mutagenesis_effects()`][universalmotif.mutagenesis_effects],
    which suits other alphabets too. With these rules, letters outside the
    alphabet, such as N, are replaced by its first letter. ``alt="all"``
    makes every substitution, as [`mutagenesis()`][universalmotif.mutagenesis]
    does, and every pair of them.

    For a region of L positions, the batch has 1 + L + L(L - 1)/2 sequences
    with one substitution per position, 20,101 for 200 positions, and with
    ``alt="all"`` on DNA, 1 + 3L + 9L(L - 1)/2, or 179,701 for 200
    positions. ``batch_size`` returns the batch as an iterable of arrays of
    up to that many sequences, made one at a time as they are needed, so
    that the whole batch is never held in memory; the predictions for them
    are joined, for example with ``np.concatenate``, before being passed on.
    Without it, the batch is returned as one array of shape (sequences,
    letters, length).

    Examples
    --------
    >>> import universalmotif as um
    >>> batch = um.pair_mutagenesis("ACGT")
    >>> batch.shape
    (11, 4, 4)
    >>> um.from_one_hot(batch[:7])
    ['ACGT', 'TCGT', 'AGGT', 'ACCT', 'ACGA', 'TGGT', 'TCCT']
    >>> len(um.pair_mutagenesis("ACGT", alt="all"))
    67
    >>> um.pair_mutagenesis("ACGTACGT", region=(2, 6), batch_size=4)
    <3 batches of up to 4 sequences, 11 in all>
    """
    _, _, pos, let, _ = _pair_substitutions(sequence, alphabet, region, alt)
    first, second = _pair_rows(pos)
    ref = one_hot(sequence, alphabet=alphabet, dtype=dtype)
    if batch_size is not None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be 1 or more")
        return _PairBatches(ref, pos, let, first, second, int(batch_size))
    return _pair_batch(ref, pos, let, first, second, 0, 1 + len(pos) + len(first))


def _outputs_first(x: np.ndarray, k: int, extra: int) -> np.ndarray:
    """Move the ``k`` axes of positions and letters after the axes of a
    prediction's outputs."""
    if not extra:
        return x
    return np.moveaxis(x, list(range(k)), list(range(-k, 0)))


def interaction_effects(
    predictions: Any,
    sequence: str,
    alphabet: str = "ACGT",
    region: tuple[int, int] | None = None,
    alt: str | Sequence[str] | None = None,
    summary: str | None = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """The interactions between pairs of positions, as the epistasis between
    substitutions, from a model's predictions for the batch made by
    [`pair_mutagenesis()`][universalmotif.pair_mutagenesis].

    ``predictions`` holds one prediction per entry of the batch, in order,
    with the sequence's own first; ``sequence``, ``alphabet``, ``region`` and
    ``alt`` must be those the batch was made with. The epistasis between
    substitutions at positions i and j is how far the effect of making both
    differs from the sum of their effects alone,
    f(ij) - f(i) - f(j) + f(sequence), which is 0 for a model that adds up
    the effects of its positions. Breaking two sites the model needs
    together, or a site twice over, costs less than the two substitutions
    apart, which gives positive values when the substitutions lower the
    prediction.

    Returns a pair of arrays. The first is the symmetric matrix of
    epistasis, of shape (positions, positions) over ``region``, with a
    diagonal of NaN since a position does not interact with itself in this
    sense, ready for [`plot_interactions()`][universalmotif.plot_interactions].
    The second holds the effect of each single substitution, the prediction
    minus the sequence's, with one value per position, or with
    ``alt="all"``, an array of shape (letters, positions) like that of
    [`mutagenesis_effects()`][universalmotif.mutagenesis_effects]. With
    ``alt="all"``, each pair of positions has several values, which
    ``summary`` reduces to one: their ``"mean"``, the ``"strongest"``, of
    the largest size, or the ``"min"`` or ``"max"``. ``summary=None`` keeps
    them all, as an array of shape (letters, positions, letters, positions)
    with NaN for the letters of the sequence. With several outputs per
    prediction, given as an array of shape (sequences, outputs), each array
    gets the outputs as its first axis.

    Examples
    --------
    >>> import numpy as np
    >>> import universalmotif as um
    >>> seq = "ACGTAC"
    >>> batch = um.pair_mutagenesis(seq)
    >>> x = um.from_one_hot(batch)
    >>> # A model that scores CG and TA steps, each pair of letters needed together
    >>> preds = np.array([s.count("CG") + s.count("TA") for s in x])
    >>> epistasis, single = um.interaction_effects(preds, seq)
    >>> single.astype(int).tolist()
    [0, -1, -1, -1, -1, 0]
    >>> np.nan_to_num(epistasis).astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    """
    if summary is not None and summary not in _SUMMARIES:
        raise ValueError(f"summary must be one of {', '.join(map(repr, _SUMMARIES))}, or None")
    a, b, pos, let, per_position = _pair_substitutions(sequence, alphabet, region, alt)
    first, second = _pair_rows(pos)
    s = len(pos)
    expected = 1 + s + len(first)
    pred = _to_numpy(predictions)
    if pred.ndim == 0 or pred.shape[0] != expected:
        raise ValueError(f"expected {expected} predictions, one for the sequence, one per substitution "
                         f"and one per pair, not {pred.shape[0] if pred.ndim else 0}")
    rest = pred.shape[1:]
    single = pred[1:1 + s] - pred[0]
    eps = pred[1 + s:] - pred[0] - single[first] - single[second]
    n = b - a
    pi, pj = pos[first] - a, pos[second] - a
    if per_position:
        out = np.full((n, n) + rest, np.nan)
        out[pi, pj] = eps
        out[pj, pi] = eps
        return _outputs_first(out, 2, len(rest)), _outputs_first(single, 1, len(rest))
    effects = np.zeros((len(alphabet), n) + rest)
    effects[let, pos - a] = single
    effects = _outputs_first(effects, 2, len(rest))
    li, lj = let[first], let[second]
    if summary is None:
        out = np.full((len(alphabet), n, len(alphabet), n) + rest, np.nan)
        out[li, pi, lj, pj] = eps
        out[lj, pj, li, pi] = eps
        return _outputs_first(out, 4, len(rest)), effects
    flat = eps.reshape(len(eps), -1)
    out = np.full((n, n, flat.shape[1]), np.nan)
    if summary == "mean":
        sums = np.zeros((n * n, flat.shape[1]))
        np.add.at(sums, pi * n + pj, flat)
        counts = np.bincount(pi * n + pj, minlength=n * n)
        has = counts > 0
        sums[has] /= counts[has, None]
        out.reshape(n * n, -1)[has] = sums[has]
    else:
        key = pi * n + pj
        for c in range(flat.shape[1]):
            v = flat[:, c]
            rank = {"strongest": -np.abs(v), "min": v, "max": -v}[summary]
            order = np.lexsort((rank, key))
            # The first of each pair after sorting is the one kept.
            keep = order[np.r_[True, key[order][1:] != key[order][:-1]]]
            out[pi[keep], pj[keep], c] = v[keep]
    upper = np.triu_indices(n, 1)
    out[upper[1], upper[0]] = out[upper]
    out = out.reshape((n, n) + rest)
    return _outputs_first(out, 2, len(rest)), effects


def _robust_limit(values: np.ndarray) -> float:
    """The 99th percentile of the sizes of the values other than 0, so that a
    few strong values do not wash out the rest of the scale."""
    sizes = np.abs(values[np.isfinite(values)])
    sizes = sizes[sizes > 0]
    return float(np.percentile(sizes, 99)) if sizes.size else 1.0


def _position_ticks(a: int, b: int, length_in: float) -> tuple[list[float], list[str]]:
    """Ticks for positions ``a`` to ``b`` drawn from 1 to b - a along an axis
    ``length_in`` inches long, with room for the labels side by side."""
    max_ticks = max(2, min(12, int(length_in / (0.08 * len(str(b)) + 0.25))))
    step = _nice_step(b - a, max_ticks)
    ticks = list(range(-(-a // step) * step, b, step))
    return [t - a + 1 for t in ticks], [str(t) for t in ticks]


def _label_rows(fig: Figure, spans, inch_per_pos: float, size: float) -> list[int]:
    """The row of each hit's name above its band, moving a name up a row when
    it would run into one already placed."""
    renderer = fig.canvas.get_renderer()
    rows: list[list[tuple[float, float]]] = []
    out = [0] * len(spans)
    for k in sorted(range(len(spans)), key=lambda k: spans[k][0] + spans[k][1]):
        x0, x1, name = spans[k]
        if name is None:
            out[k] = -1
            continue
        text = fig.text(0, 0, str(name), fontsize=size)
        width = text.get_window_extent(renderer).width / fig.dpi
        text.remove()
        centre = (x0 + x1) / 2 * inch_per_pos
        lo, hi = centre - width / 2 - 0.03, centre + width / 2 + 0.03
        row = 0
        while row < len(rows) and not all(hi <= u or lo >= v for u, v in rows[row]):
            row += 1
        if row == len(rows):
            rows.append([])
        rows[row].append((lo, hi))
        out[k] = row
    return out


def _score_length(scores: Any, alphabet: str) -> int:
    """The number of positions of scores in any of the forms view_attributions() takes."""
    x = _to_numpy(scores)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 2 and x.shape[0] != len(alphabet) and x.shape[1] == len(alphabet):
        return x.shape[0]
    return x.shape[-1] if x.ndim else 0


def _strip_glyphs(m: np.ndarray, alphabet: str):
    """The letters of a logo along a strip, as in view_attributions()."""
    hi = max(float(m.clip(min=0).sum(axis=0).max(initial=0.0)), 0.0)
    lo = min(float(m.clip(max=0).sum(axis=0).min(initial=0.0)), 0.0)
    scale = max(hi, -lo) or 1.0
    glyphs = matrix_glyphs(m, list(alphabet), 1e-9 * scale, 0.04, 0.01 * scale, False, True, None,
                           1.0, True)
    pad = 0.08 * ((hi - lo) or 1.0)
    return glyphs, (lo - (pad if lo < 0 else 0), hi + pad)


def _draw_strip(ax: Axes, glyphs, limits, colours, n: int, vertical: bool, zero: bool) -> None:
    """Draw a logo along a strip beside the map, with the positions along x,
    or along y from the top with the letters turned to face the map."""
    from matplotlib.collections import PolyCollection

    if glyphs:
        verts = [np.column_stack([g.y, g.x] if vertical else [g.x, g.y]) for g in glyphs]
        faces = [colours.get(g.group, _MISSING_COLOUR) for g in glyphs]
        ax.add_collection(PolyCollection(verts, facecolors=faces, edgecolors="none", linewidths=0))
    if vertical:
        ax.set_ylim(n + 0.5, 0.5)
        ax.set_xlim(*limits)
        if zero:
            ax.axvline(0, color="0.6", linewidth=0.6, zorder=0)
    else:
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(*limits)
        if zero:
            ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)


def plot_interactions(
    matrix: Any,
    sequence: str | None = None,
    scores: Any = None,
    alphabet: str = "ACGT",
    start: int = 0,
    region: tuple[int, int] | None = None,
    hits: pl.DataFrame | None = None,
    triangle: bool = False,
    max_distance: int | None = None,
    vmax: float | None = None,
    cmap: Any = "RdBu_r",
    label: str = "Interaction",
    colour: str = _HIT_COLOUR,
    colour_scheme: Mapping[str, str] | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw a matrix of interactions between the positions of a sequence as
    a heatmap, with the sequence or a logo of its scores along the edges.

    ``matrix`` is any square array with a row and a column per position,
    such as the epistasis from
    [`interaction_effects()`][universalmotif.interaction_effects] or a
    matrix of feature interactions computed with a model's own library,
    for example the change in the attribution of each position as each
    other position is changed. Values are drawn on a diverging colour scale
    centred on 0, which runs to ``vmax`` either way, by default the 99th
    percentile of their sizes, so that a few strong values do not wash out
    the rest, and missing values, such as the diagonal of epistasis, are
    grey. The colour bar is labelled with ``label``.

    Positions are labelled from ``start`` on both axes, and the matrix
    covers the whole of ``sequence``, or with a matrix shorter than the
    sequence, the part of it given by ``region``, a (start, end) pair in
    the same positions with the end left out, as made by
    ``interaction_effects()`` with a region. With a matrix as long as the
    sequence, ``region`` draws only part of it. ``sequence`` is drawn along
    the top and left edges, and ``scores``, one per position or one per
    letter and position in any of the forms
    [`view_attributions()`][universalmotif.view_attributions] takes and as
    long as the sequence or the matrix, are drawn there as a logo of the
    letters of the sequence instead, such as the contributions of a model
    or minus the single substitution effects. ``hits``, such as from
    [`scan_sequences()`][universalmotif.scan_sequences] run on ``sequence``
    and so with positions counted along it from 0, are shaded along the
    edges in ``colour``, labelled with their ``motif`` if the table has one,
    and each pair of hits is outlined on the map, where the interactions
    between and within their sites lie.

    ``triangle=True`` draws only the upper triangle, turned so that the
    diagonal lies flat, as is usual for Hi-C maps, with the sequence or
    logo below it and the distance between two positions as the height.
    ``max_distance`` then cuts the triangle off at that distance, which
    suits long regions where only nearby positions interact. Letters are
    coloured by ``colour_scheme``, by default the package's colours for DNA,
    RNA and amino acids, and ``text_size`` sets the size of labels.

    Needs the ``plot`` extra.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        A new figure, or with ``ax``, the axes the map is drawn in, with the
        strips along its edges and the colour bar as inset axes beside it.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> seq = "ACGTAC"
    >>> x = um.from_one_hot(um.pair_mutagenesis(seq))
    >>> preds = np.array([s.count("CG") + s.count("TA") for s in x])
    >>> epistasis, single = um.interaction_effects(preds, seq)
    >>> fig = um.plot_interactions(epistasis, seq, scores=-single)
    >>> fig.axes[0].get_xlabel()
    'Position'
    >>> fig = um.plot_interactions(epistasis, seq, start=100, triangle=True)
    >>> plt.close("all")
    """
    import matplotlib as mpl

    m = _to_numpy(matrix)
    while m.ndim > 2 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square, of shape (positions, positions); with several "
                         "outputs, pick one, such as matrix[0]")
    if np.isinf(m).any():
        raise ValueError("the matrix has infinite values")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("the alphabet has repeated letters")
    size = m.shape[0]
    length = len(sequence) if sequence is not None else size
    if size == length:
        a, b = (start, start + length) if region is None else (int(region[0]), int(region[1]))
        if not start <= a < b <= start + length:
            raise ValueError(f"region must lie within {start} to {start + length}")
        m = m[a - start:b - start, a - start:b - start]
    elif size < length:
        if region is None:
            raise ValueError(f"the matrix has {size} positions for a sequence of {length}; give the "
                             "region it covers")
        a, b = int(region[0]), int(region[1])
        if not start <= a < b <= start + length or b - a != size:
            raise ValueError(f"region must lie within {start} to {start + length} and cover the "
                             f"{size} positions of the matrix")
    else:
        raise ValueError(f"the matrix has {size} positions for a sequence of {length}")
    n = b - a
    logo = None
    if scores is not None:
        if size < length and _score_length(scores, alphabet) == n:
            window = None if sequence is None else sequence[a - start:b - start]
            logo = _attribution_matrix(scores, window, alphabet, "observed")
        else:
            s = _attribution_matrix(scores, sequence, alphabet, "observed")
            if s.shape[1] != length:
                raise ValueError(f"scores must have one value per position, {length} here")
            logo = s[:, a - start:b - start]
    elif sequence is not None:
        logo = one_hot(sequence[a - start:b - start], alphabet=alphabet, dtype=float)[0]
    spans = [] if hits is None else _hit_spans(hits, start, a, b)
    if max_distance is not None and int(max_distance) < 1:
        raise ValueError("max_distance must be 1 or more")
    depth = n if max_distance is None else min(int(max_distance), n)
    if colour_scheme is None:
        colour_scheme = _default_colours(_ALPHABET_NAMES.get(alphabet, ""), list(alphabet))

    top = _robust_limit(m) if vmax is None else float(vmax)
    if top <= 0:
        raise ValueError("vmax must be above 0")
    finite = m[np.isfinite(m)]
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max",
              (True, True): "both"}[(bool((finite < -top).any()), bool((finite > top).any()))]
    colours = mpl.colormaps.get_cmap(cmap).with_extremes(bad="#E6E6E6")
    small = text_size - 1

    plt = _pyplot()
    own_figure = ax is None
    thick = 0.0 if logo is None and not spans else 0.7 if scores is not None else (
        0.15 if logo is None else max(0.12, min(0.3, 1.3 * 6.0 / n)))
    gap = 0.04
    if own_figure:
        fig = plt.figure(figsize=figsize or (6, 6))
        width = (min(max(2.2 + 0.025 * n, 3.2), 6.0) if not triangle
                 else min(max(3.0 + 0.04 * n, 4.5), 10.0))
    else:
        fig = ax.figure
        ax.apply_aspect()
        width = ax.get_position().width * fig.get_figwidth()
    rows = _label_rows(fig, spans, width / n, small) if spans else []
    nrows = max(rows, default=-1) + 1
    label_in = nrows * 1.25 * small / 72 + (0.05 if nrows else 0.0)
    if own_figure:
        if triangle:
            height = width * depth / (2 * n)
            left, right, top_m = 0.6, 1.0, 0.1
            bottom = 0.55 + (thick + gap + label_in if thick else label_in)
        else:
            height = width
            left, right, bottom = 0.75 + (thick + gap if thick else 0), 1.0, 0.55
            top_m = 0.1 + (thick + gap if thick else 0) + label_in
        fw, fh = left + width + right, bottom + height + top_m
        if figsize is None:
            fig.set_size_inches(fw, fh)
        else:
            fw, fh = figsize
        ax = fig.add_axes((left / fw, bottom / fh, max(fw - left - right, 0.5) / fw,
                           max(fh - bottom - top_m, 0.5) / fh))

    if triangle:
        edges = np.arange(n + 1) + 0.5
        i_edge, j_edge = np.meshgrid(edges, edges, indexing="ij")
        image = ax.pcolormesh((i_edge + j_edge) / 2, j_edge - i_edge, np.ma.masked_invalid(m),
                              cmap=colours, vmin=-top, vmax=top, shading="flat", rasterized=True)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0, depth)
        ax.set_aspect(0.5, adjustable="box")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel("Distance", fontsize=text_size)
    else:
        image = ax.imshow(m, cmap=colours, vmin=-top, vmax=top, interpolation="nearest",
                          extent=(0.5, n + 0.5, n + 0.5, 0.5))
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=small)
    ax.apply_aspect()
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    xticks, xlabels = _position_ticks(a, b, w_in)

    strips = []
    if thick:
        glyphs, limits = _strip_glyphs(logo, alphabet) if logo is not None else ([], (0, 1))
        if logo is not None and scores is None:
            limits = (0.0, 1.0)
        zero = scores is not None
        if triangle:
            below = ax.inset_axes((0, -(gap + label_in + thick) / h_in, 1, thick / h_in))
            _draw_strip(below, glyphs, limits, colour_scheme, n, False, zero)
            strips.append((below, False))
            along = below
        else:
            above = ax.inset_axes((0, 1 + gap / h_in, 1, thick / h_in))
            beside = ax.inset_axes((-(gap + thick) / w_in, 0, thick / w_in, 1))
            _draw_strip(above, glyphs, limits, colour_scheme, n, False, zero)
            _draw_strip(beside, glyphs, limits, colour_scheme, n, True, zero)
            strips += [(above, False), (beside, True)]
            along = above
    else:
        along = ax

    if triangle:
        tick_ax = along
        tick_ax.set_xticks(xticks, xlabels, fontsize=small)
        tick_ax.set_xlabel("Position", fontsize=text_size)
        if tick_ax is not ax:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            tick_ax.tick_params(axis="x", length=3, labelsize=small)
        yt = [t for t in ax.get_yticks() if 0 <= t <= depth]
        ax.set_yticks(yt, [str(int(t)) for t in yt])
    else:
        ax.set_xticks(xticks, xlabels, fontsize=small)
        ax.set_xlabel("Position", fontsize=text_size)
        side = strips[1][0] if strips else ax
        side.set_yticks(xticks, xlabels, fontsize=small)
        side.tick_params(axis="y", length=3, labelsize=small)
        side.set_ylabel("Position", fontsize=text_size)
        if side is not ax:
            ax.set_yticks([])

    if spans:
        from matplotlib.patches import Polygon, Rectangle

        for x0, x1, _ in spans:
            for strip, vertical in strips:
                (strip.axhspan if vertical else strip.axvspan)(x0, x1, color=colour, alpha=0.15,
                                                               linewidth=0, zorder=0)
        style = {"fill": False, "edgecolor": colour, "linewidth": 0.9, "zorder": 3}
        for k, (r0, r1, _) in enumerate(spans):
            for c0, c1, _ in spans[k if triangle else 0:]:
                if triangle:
                    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
                    ax.add_patch(Polygon([((i + j) / 2, j - i) for i, j in corners], **style))
                else:
                    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, **style))
        for (x0, x1, name), row in zip(spans, rows):
            if row < 0:
                continue
            along.annotate(str(name), ((x0 + x1) / 2, 1.0), xycoords=("data", "axes fraction"),
                           xytext=(0, 1.25 * small * row + 1), textcoords="offset points",
                           ha="center", va="bottom", fontsize=small, color="0.3", annotation_clip=False)

    if triangle:
        bar_h = min(1.0, 1.6 / h_in) if h_in > 0 else 1.0
        cax = ax.inset_axes((1 + 0.12 / w_in, (1 - bar_h) / 2, 0.12 / w_in, bar_h))
    else:
        cax = ax.inset_axes((1 + 0.12 / w_in, 0, 0.12 / w_in, 1))
    bar = fig.colorbar(image, cax=cax, extend=extend)
    bar.set_label(label, fontsize=text_size)
    bar.ax.tick_params(labelsize=small)
    bar.outline.set_visible(False)
    return fig if own_figure else ax


__all__ = ["interaction_effects", "pair_mutagenesis", "plot_interactions"]
