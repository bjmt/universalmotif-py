"""The spacing between pairs of motif hits, against that expected by chance."""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._core import Motif
from ._plot import _pyplot
from ._plot_hits import _lengths

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_BARS = "#255C99"
_EXPECTED = "#333333"
_GUIDE = "#D4D4D4"
_MUTED = "#5E5E5E"
_HELIX = 10.5
_HETERO = ("tandem, first upstream", "tandem, second upstream", "convergent", "divergent")
_HOMO = ("tandem", "convergent", "divergent")
# Pairs are counted this many at a time, which bounds the memory used however
# densely the hits lie.
_CHUNK = 1 << 20
# The widest span of positions the positions null takes, as one array.
_MAX_SPAN = 50_000_000


def _motif_name(hits: pl.DataFrame, motif: Any, what: str) -> str:
    if isinstance(motif, Motif):
        motif = motif.name
    if isinstance(motif, str):
        if not (hits["motif"] == motif).any():
            raise ValueError(f"hits has no hits of a motif named {motif!r}")
        return motif
    if isinstance(motif, bool) or not isinstance(motif, numbers.Integral):
        raise TypeError(f"{what} must be the name or index of a motif")
    if "motif_i" not in hits.columns:
        raise ValueError(f"hits has no motif_i column to find motif {motif} in; give the motif's name")
    names = hits.filter(pl.col("motif_i") == int(motif))["motif"].unique()
    if names.len() == 0:
        raise ValueError(f"hits has no hits of motif {motif}")
    return names[0]


class _Hits:
    """The hits of one motif, sorted by sequence and then start."""

    def __init__(self, table: pl.DataFrame, need_strands: bool):
        table = table.sort("seq", "start")
        self.seq = table["seq"].to_numpy()
        self.start = table["start"].to_numpy()
        self.end = table["end"].to_numpy()
        self.names = table["name"].to_numpy() if "name" in table.columns else None
        if need_strands:
            strand = table["strand"]
            self.strand = np.where(strand.to_numpy() == "+", 1, -1).astype(np.int8)
            self.plus = float((strand == "+").mean()) if table.height else 0.5
        else:
            self.strand = np.ones(table.height, dtype=np.int8)
            self.plus = 1.0
        widths = self.end - self.start
        self.width = int(np.median(widths)) if widths.size else 0
        self.widest = int(widths.max()) if widths.size else 0

    def __len__(self) -> int:
        return self.seq.size


def _classify(sa, sb, ca, cb, homotypic: bool) -> np.ndarray:
    """The arrangement of each pair, as an index into _HOMO or _HETERO."""
    if homotypic:
        # The pair is read from the hit nearer the start, or from the one on
        # the forward strand when they share a centre.
        swap = (cb < ca) | ((cb == ca) & (sa < sb))
        s1, s2 = np.where(swap, sb, sa), np.where(swap, sa, sb)
        return np.where(s1 == s2, 0, np.where(s1 > 0, 1, 2)).astype(np.int64)
    down = sa * (cb - ca) >= 0
    return np.where(sa == sb, np.where(down, 0, 1), np.where(down, 2, 3)).astype(np.int64)


def _measure(sa, ea, sb, eb, how: str) -> tuple[np.ndarray, np.ndarray]:
    """The distance between each pair of hits, and whether they overlap."""
    overlap = (sb < ea) & (sa < eb)
    if how == "centre":
        dist = np.abs((sb + eb) // 2 - (sa + ea) // 2)
    else:
        dist = np.maximum(sb - ea, sa - eb)
    return dist, overlap


def _observed(a: _Hits, b: _Hits | None, how: str, overlaps: bool, split: bool, lo: int, hi: int,
              ncodes: int) -> np.ndarray:
    homotypic = b is None
    b = a if b is None else b
    nbins = hi - lo + 1
    counts = np.zeros(ncodes * nbins, dtype=np.int64)
    if len(a) == 0 or len(b) == 0:
        return counts.reshape(ncodes, nbins)
    reach = hi + a.widest + b.widest
    first = int(min(a.start.min(), b.start.min()))
    span = int(max(a.end.max(), b.end.max())) - first + 2 * reach + 1
    key_a = a.seq.astype(np.int64) * span + (a.start - first)
    key_b = key_a if homotypic else b.seq.astype(np.int64) * span + (b.start - first)
    if homotypic:
        left = np.arange(1, len(a), dtype=np.int64)
        left = np.append(left, len(a))
    else:
        left = np.searchsorted(key_b, key_a - reach, side="left")
    right = np.searchsorted(key_b, key_a + reach, side="right")
    n = np.maximum(right - left, 0)
    total = np.cumsum(n)
    cuts = np.searchsorted(total, np.arange(_CHUNK, int(total[-1]) if total.size else 0, _CHUNK), side="left")
    bounds = np.concatenate([[0], cuts + 1, [len(a)]])
    for i0, i1 in itertools.pairwise(bounds):
        if i1 <= i0:
            continue
        k = n[i0:i1]
        m = int(k.sum())
        if m == 0:
            continue
        ia = np.repeat(np.arange(i0, i1), k)
        ib = np.repeat(left[i0:i1] - (np.cumsum(k) - k), k) + np.arange(m)
        sa, ea, sb, eb = a.start[ia], a.end[ia], b.start[ib], b.end[ib]
        dist, overlap = _measure(sa, ea, sb, eb, how)
        keep = (dist >= lo) & (dist <= hi)
        if not overlaps:
            keep &= ~overlap
        if split:
            code = _classify(a.strand[ia[keep]], b.strand[ib[keep]], (sa + ea)[keep] // 2, (sb + eb)[keep] // 2,
                             homotypic)
        else:
            code = np.zeros(int(keep.sum()), dtype=np.int64)
        counts += np.bincount(code * nbins + (dist[keep] - lo), minlength=counts.size)
    return counts.reshape(ncodes, nbins)


def _pair_weight(a: _Hits, b: _Hits | None) -> float:
    """The number of ordered pairs of distinct hits sharing a sequence."""
    na = np.bincount(a.seq).astype(np.float64)
    if b is None:
        return float((na * (na - 1)).sum())
    nb = np.bincount(b.seq).astype(np.float64)
    k = min(na.size, nb.size)
    return float((na[:k] * nb[:k]).sum())


def _null_positions(a: _Hits, b: _Hits | None, reach: int) -> np.ndarray:
    """Expected ordered pairs at each start offset from -reach to reach,
    with each hit placed at random among the starts of its motif's hits."""
    other = a if b is None else b
    first = int(min(a.start.min(), other.start.min()))
    span = int(max(a.start.max(), other.start.max())) - first + 1
    if span > _MAX_SPAN:
        raise ValueError(f"the hits span {span:,} positions, too many for null='positions', which suits hits "
                         "counted from the start of each sequence; use the hits from before genomic_hits(), "
                         "or null='uniform' with the sequences")
    size = 1 << (2 * span + 2 * reach).bit_length()
    ha = np.bincount(a.start - first, minlength=span).astype(np.float64)
    hb = ha if b is None else np.bincount(b.start - first, minlength=span).astype(np.float64)
    corr = np.fft.irfft(np.conj(np.fft.rfft(ha, size)) * np.fft.rfft(hb, size), size)
    lags = np.arange(-reach, reach + 1)
    corr = np.clip(corr[lags % size], 0, None)
    return _pair_weight(a, b) * corr / (len(a) * len(other))


def _null_uniform(a: _Hits, b: _Hits | None, lengths: np.ndarray, reach: int) -> np.ndarray:
    """Expected ordered pairs at each start offset from -reach to reach,
    with each hit placed anywhere in its sequence with equal chance."""
    other = a if b is None else b
    na = np.bincount(a.seq, minlength=lengths.size).astype(np.float64)
    nb = na if b is None else np.bincount(b.seq, minlength=lengths.size).astype(np.float64)
    weight = na * (na - 1) if b is None else na * nb
    used = weight > 0
    lens, inverse = np.unique(lengths[used], return_inverse=True)
    weights = np.bincount(inverse, weights=weight[used], minlength=lens.size)
    lags = np.arange(-reach, reach + 1)
    out = np.zeros(lags.size)
    for i in range(0, lens.size, 4096):
        ln = lens[i:i + 4096, None]
        sa, sb = ln - a.width + 1, ln - other.width + 1
        ways = np.clip(np.minimum(sa, sb - lags) - np.maximum(0, -lags), 0, None)
        ok = (sa > 0) & (sb > 0)
        prob = np.where(ok, ways / np.where(ok, sa * sb, 1), 0.0)
        out += weights[i:i + 4096] @ prob
    return out


def _expected(a: _Hits, b: _Hits | None, null: str, lengths: np.ndarray | None, how: str, overlaps: bool,
              split: bool, lo: int, hi: int, ncodes: int) -> np.ndarray:
    homotypic = b is None
    other = a if b is None else b
    nbins = hi - lo + 1
    out = np.zeros((ncodes, nbins))
    if len(a) == 0 or len(other) == 0:
        return out
    reach = hi + a.widest + other.widest
    if null == "positions":
        per_lag = _null_positions(a, b, reach)
    else:
        per_lag = _null_uniform(a, b, lengths, reach)
    if homotypic:
        per_lag = per_lag / 2
    lags = np.arange(-reach, reach + 1)
    sa, ea = np.zeros_like(lags), np.full_like(lags, a.width)
    sb, eb = lags, lags + other.width
    dist, overlap = _measure(sa, ea, sb, eb, how)
    keep = (dist >= lo) & (dist <= hi)
    if not overlaps:
        keep &= ~overlap
    combos = ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if split else [(1, 1)])
    p, q = a.plus, other.plus
    for s1, s2 in combos:
        chance = (p if s1 > 0 else 1 - p) * (q if s2 > 0 else 1 - q) if split else 1.0
        if chance == 0:
            continue
        if split:
            code = _classify(np.full(lags.size, s1), np.full(lags.size, s2), (sa + ea) // 2, (sb + eb) // 2,
                             homotypic)
        else:
            code = np.zeros(lags.size, dtype=np.int64)
        np.add.at(out, (code[keep], dist[keep] - lo), chance * per_lag[keep])
    return out


def pair_spacings(
    hits: pl.DataFrame,
    first: str | int | Motif,
    second: str | int | Motif | None = None,
    max_distance: int = 100,
    distance: str = "centre",
    overlaps: bool = False,
    arrangements: bool = True,
    null: str = "positions",
    sequences=None,
) -> pl.DataFrame:
    """Count the pairs of hits of two motifs at each distance apart, and
    the number expected by chance.

    Every hit of ``first`` is paired with every hit of ``second`` in the
    same sequence, or with every other hit of ``first`` when ``second`` is
    not given, and the pairs are counted by the distance between them up to
    ``max_distance``. By default the distance runs between the centres of
    the two hits, taken as the middle letter, or the second of the two
    middle letters for an even width, which is the usual measure in
    studies of the syntax of motifs and the one to use for phasing along
    the helix. With ``distance="gap"`` it is instead the number of letters
    between the end of one hit and the start of the other, as
    [`site_spacings()`][universalmotif.site_spacings] spaces its sites,
    which is negative for hits that overlap.

    Pairs are also split by their arrangement, read along the strand of
    ``first``. Two hits on the same strand are in tandem, with ``first``
    either upstream or downstream of ``second``, and two hits on opposite strands
    either point towards each other, being convergent, or away from each
    other, being divergent. Since the strand of the sequence itself is
    arbitrary these four cover every pair, and the three of tandem,
    convergent and divergent cover those of a motif with itself.

    The number expected counts the pairs that would be seen if the hits of
    the two motifs lay independently of each other. With the default
    ``null="positions"``, each hit is placed at random at one of the
    positions where its motif was seen anywhere in the hits, and each
    sequence keeps its number of hits, so that a preference of both motifs
    for, say, the middle of the sequences is part of what is expected. This
    needs nothing but the hits, and suits sequences of the same or similar
    lengths, such as peaks or the windows of seqlets. It takes in part of
    any preference for a spacing when the sites themselves sit at fixed
    places, as when they were written into the same place in every
    sequence, and so errs on the side of expecting too many pairs. With
    ``null="uniform"``, each hit is placed with equal chance anywhere in its
    sequence instead, which needs the ``sequences`` and suits sequences of
    very different lengths, such as chromosomes. Either way, strands are
    taken as independent of positions and of each other, with each motif's
    hits falling on each strand in the shares seen. The counts expected are
    worked out exactly rather than by shuffling, so no seed is needed.

    Parameters
    ----------
    hits
        Hits with ``motif``, ``start`` and ``end`` columns, positions counted
        from 0 with the end left out, a ``sequence`` or ``sequence_i``
        column, and a ``strand`` column for the arrangements, as from
        [`scan_sequences()`][universalmotif.scan_sequences] or
        [`read_seqlets()`][universalmotif.read_seqlets].
    first, second
        The two motifs, by name, by their index in ``motif_i`` or as the
        motifs scanned. Without ``second``, or with the same motif twice,
        the hits of ``first`` are paired among themselves.
    max_distance
        The greatest distance counted, in letters.
    distance
        ``"centre"`` for the distance between centres or ``"gap"`` for the
        letters between the hits.
    overlaps
        Count pairs of hits that overlap. They are left out by default,
        since two motifs seldom bind the same letters and overlapping hits
        are often the same site matched twice. Seqlets carry flanks around
        their sites, so pairs closer than their width are left out too.
    arrangements
        Split the pairs by arrangement. Without it every pair is counted
        together, and hits need no strands.
    null
        ``"positions"`` or ``"uniform"``, how hits are placed by chance.
    sequences
        The sequences scanned, for ``null="uniform"``, as a list, a mapping
        from names to sequences or to lengths, or anything with a
        ``lengths`` mapping such as
        [`PackedSequences`][universalmotif.PackedSequences].

    Returns
    -------
    polars.DataFrame
        One row per arrangement and distance, with ``arrangement``,
        ``distance``, ``observed`` and ``expected`` columns, and
        ``enrichment`` for the ratio of the two, which is null where nothing
        is expected. Without ``arrangements``, the arrangement is
        ``"all"``.

    Examples
    --------
    Pairs of GATA sites written in tandem at random places in random
    sequences, their centres 21 letters apart, stand out from those
    expected:

    >>> import numpy as np
    >>> import universalmotif as um
    >>> seqs = um.create_sequences(seqnum=100, seqlen=120, seed=1)
    >>> starts = np.random.default_rng(1).integers(0, 90, len(seqs))
    >>> seqs = [um.insert_sites(s, ["AGATAAGA", "AGATAAGA"], positions=[int(p), int(p) + 21])
    ...         for s, p in zip(seqs, starts)]
    >>> hits = um.scan_sequences(um.example_motifs()[6], seqs, pvalue=1e-3)
    >>> res = um.pair_spacings(hits, "GATA", max_distance=30)
    >>> top = res.sort("observed", descending=True).row(0, named=True)
    >>> top["arrangement"], top["distance"], top["observed"], round(top["expected"], 1)
    ('tandem', 21, 57, 1.6)
    """
    if distance not in ("centre", "gap"):
        raise ValueError("distance must be 'centre' or 'gap'")
    if null not in ("positions", "uniform"):
        raise ValueError("null must be 'positions' or 'uniform'")
    if isinstance(max_distance, bool) or not isinstance(max_distance, numbers.Integral) or max_distance < 1:
        raise ValueError("max_distance must be a whole number, 1 or more")
    max_distance = int(max_distance)
    for col in ("motif", "start", "end"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if "sequence_i" not in hits.columns and "sequence" not in hits.columns:
        raise ValueError("hits must have a 'sequence' or 'sequence_i' column")
    if null == "uniform":
        if sequences is None:
            raise ValueError("null='uniform' places hits along the sequences; give their lengths with sequences")
        if "sequence" not in hits.columns:
            raise ValueError("null='uniform' needs a 'sequence' column to match hits to their sequences")
    elif sequences is not None:
        raise ValueError("sequences is only used with null='uniform'")
    hits = hits.with_columns(pl.col("motif").cast(pl.String))
    a_name = _motif_name(hits, first, "first")
    b_name = a_name if second is None else _motif_name(hits, second, "second")
    homotypic = a_name == b_name

    table = hits.filter(pl.col("motif").is_in([a_name, b_name]))
    if arrangements:
        if "strand" not in hits.columns:
            raise ValueError("hits has no 'strand' column to tell arrangements apart; give arrangements=False")
        bad = table.filter(~pl.col("strand").cast(pl.String).is_in(["+", "-"]))
        if bad.height:
            raise ValueError("arrangements need every hit on the '+' or '-' strand; give arrangements=False "
                             "to count pairs without them")
    cols = [pl.col("motif"), pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64)]
    if arrangements:
        cols.append(pl.col("strand").cast(pl.String))
    else:
        cols.append(pl.lit("+").alias("strand"))
    if "sequence_i" in table.columns:
        cols.append(pl.col("sequence_i").cast(pl.Int64).alias("seq"))
    else:
        cols.append(pl.col("sequence").cast(pl.String).rank("dense").cast(pl.Int64).alias("seq"))
    if "sequence" in table.columns:
        cols.append(pl.col("sequence").cast(pl.String).alias("name"))
    table = table.select(cols)
    if table["start"].null_count() or table["end"].null_count() or table["seq"].null_count():
        raise ValueError("start, end and the sequence columns cannot contain nulls")
    if (table["start"] > table["end"]).any():
        raise ValueError("all hits must have start <= end")

    lengths = None
    if null == "uniform":
        known = _lengths(sequences)
        # Sequences are renumbered from their names, so that lengths can be
        # looked up whichever column numbered them.
        names = table["name"].unique(maintain_order=True)
        missing = [s for s in names.to_list() if s not in known]
        if missing:
            raise ValueError(f"sequences has no lengths for: {', '.join(map(str, missing[:5]))}"
                             + (", ..." if len(missing) > 5 else ""))
        table = table.with_columns(pl.col("name").replace_strict(names, np.arange(names.len()),
                                                                 return_dtype=pl.Int64).alias("seq"))
        lengths = np.array([known[s] for s in names.to_list()], dtype=np.int64)
        if (table["end"] > pl.Series(lengths)[table["seq"]]).any():
            raise ValueError("some hits run past the end of their sequences; check that sequences are the "
                             "sequences scanned")
    a = _Hits(table.filter(pl.col("motif") == a_name), arrangements)
    b = None if homotypic else _Hits(table.filter(pl.col("motif") == b_name), arrangements)

    labels = (_HOMO if homotypic else _HETERO) if arrangements else ("all",)
    lo = 0
    if distance == "gap":
        lo = -max(a.widest, (a if b is None else b).widest) if overlaps else 0
    obs = _observed(a, b, distance, overlaps, arrangements, lo, max_distance, len(labels))
    exp = _expected(a, b, null, lengths, distance, overlaps, arrangements, lo, max_distance, len(labels))
    nbins = max_distance - lo + 1
    out = pl.DataFrame({
        "arrangement": np.repeat(np.array(labels), nbins),
        "distance": np.tile(np.arange(lo, max_distance + 1, dtype=np.int64), len(labels)),
        "observed": obs.ravel(),
        "expected": exp.ravel(),
    })
    return out.with_columns(
        pl.when(pl.col("expected") > 0).then(pl.col("observed") / pl.col("expected")).alias("enrichment"))


def _short(name: str, most: int = 24) -> str:
    return name if len(name) <= most else "…" + name[-(most - 1):]


def _arrows(code: int, a: str, b: str, homotypic: bool) -> str:
    r, lft = "→", "←"
    if homotypic:
        return (f"{a}{r} {a}{r}", f"{a}{r} {lft}{a}", f"{lft}{a} {a}{r}")[code]
    return (f"{a}{r} {b}{r}", f"{b}{r} {a}{r}", f"{a}{r} {lft}{b}", f"{lft}{b} {a}{r}")[code]


def _tidy(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_pair_spacing(
    hits: pl.DataFrame,
    first: str | int | Motif,
    second: str | int | Motif | None = None,
    max_distance: int = 100,
    distance: str = "centre",
    overlaps: bool = False,
    arrangements: bool = True,
    null: str = "positions",
    sequences=None,
    helix: bool | float = False,
    names: str | Sequence[str] | None = None,
    colour: str = _BARS,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw how far apart the hits of two motifs lie, against how far apart
    they would lie by chance.

    Pairs of hits of ``first`` and ``second`` in the same sequence are
    counted by the distance between them, as by
    [`pair_spacings()`][universalmotif.pair_spacings], and drawn as bars,
    one per letter of distance, with a line for the number expected if the
    two motifs lay independently of each other. A preference for a spacing
    shows as bars rising above the line at that distance, and one for
    sites on the same face of the DNA double helix as peaks every 10 or 11
    letters. Without ``second``, the hits of ``first`` are paired among
    themselves.

    Pairs are split by arrangement into panels stacked one above the other
    on the same axes, each labelled with arrows for the two hits read along
    the strand of ``first``: in tandem with either motif upstream, pointing
    towards each other (convergent), or pointing away from each other
    (divergent). This is the observed counterpart of
    [`plot_spacing()`][universalmotif.plot_spacing], which draws a model's
    response to sites written in at a range of spacings.

    Needs the ``plot`` extra.

    Parameters
    ----------
    hits
        Hits with ``motif``, ``start``, ``end`` and ``strand`` columns and a
        ``sequence`` or ``sequence_i`` column, as from
        [`scan_sequences()`][universalmotif.scan_sequences] or
        [`read_seqlets()`][universalmotif.read_seqlets].
    first, second
        The two motifs, by name, by their index in ``motif_i`` or as the
        motifs scanned.
    max_distance, distance, overlaps, arrangements, null, sequences
        How pairs are counted and how many are expected, as for
        [`pair_spacings()`][universalmotif.pair_spacings]. The distance is
        between centres by default. Without ``arrangements`` all pairs are
        drawn in one panel.
    helix
        Mark the period of the DNA double helix, 10.5 bp, with faint
        vertical lines. The lines pass through the distance given, or with
        ``True`` through the distance with the largest excess of pairs over
        those expected.
    names
        Names for the two motifs in the panel labels, or one for a motif
        paired with itself. Long names of motifs are otherwise cut to their
        last 24 letters.
    colour
        The colour of the bars.
    text_size
        The size of the panel labels and the legend.
    figsize
        The size of a new figure, in inches.
    ax
        Axes to draw a single panel into, which needs
        ``arrangements=False``. They are returned, and otherwise a new
        figure is.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    A W-box and a GATA site written into half of a set of random sequences
    at random places, the GATA site on the reverse strand and their centres
    21 letters apart:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> seqs = um.create_sequences(seqnum=300, seqlen=150, seed=2)
    >>> starts = np.random.default_rng(1).integers(0, 120, 150)
    >>> paired = [um.insert_sites(s, ["TTGACC", "TTATCTGG"], positions=[int(p), int(p) + 20])
    ...           for s, p in zip(seqs[:150], starts)]
    >>> motifs = [um.example_motifs()[i] for i in (6, 5)]
    >>> hits = um.scan_sequences(motifs, paired + seqs[150:], pvalue=1e-3)
    >>> fig = um.plot_pair_spacing(hits, "GATA", "W-box", max_distance=50, helix=True)
    >>> [ax.texts[0].get_text() for ax in fig.axes]
    ['GATA→ W-box→', 'W-box→ GATA→', 'GATA→ ←W-box', '←W-box GATA→']

    Read along the GATA site, the W-box lies downstream on the other
    strand, so the pairs show up in the convergent panel:

    >>> bars = fig.axes[2].patches
    >>> int(max(bars, key=lambda b: b.get_height()).get_center()[0])
    21
    >>> plt.close(fig)

    All pairs together, in a panel of a larger figure:

    >>> fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    >>> ax = um.plot_pair_spacing(hits, 0, 1, arrangements=False, ax=axes[1])
    >>> ax is axes[1]
    True
    >>> plt.close(fig)
    """
    if ax is not None and arrangements:
        raise ValueError("ax holds a single panel; give arrangements=False to draw every pair in it")
    res = pair_spacings(hits, first, second, max_distance=max_distance, distance=distance, overlaps=overlaps,
                        arrangements=arrangements, null=null, sequences=sequences)
    names_in = hits.with_columns(pl.col("motif").cast(pl.String))
    a_name = _motif_name(names_in, first, "first")
    b_name = a_name if second is None else _motif_name(names_in, second, "second")
    homotypic = a_name == b_name
    if names is None:
        shown = [_short(a_name), _short(b_name)]
    else:
        shown = [names] if isinstance(names, str) else [str(n) for n in names]
        if len(shown) != (1 if homotypic else 2):
            raise ValueError("give one name for the motif" if homotypic else "give names for both motifs")
        shown = shown * 2 if homotypic else shown
    labels = list(dict.fromkeys(res["arrangement"].to_list()))
    x = np.unique(res["distance"].to_numpy())
    obs = res["observed"].to_numpy().reshape(len(labels), x.size)
    exp = res["expected"].to_numpy().reshape(len(labels), x.size)

    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    own_figure = ax is None
    if own_figure:
        n = len(labels)
        fig, axes = plt.subplots(n, 1, sharex=True, sharey=True, squeeze=False,
                                 figsize=figsize or (6.4, 0.9 + 1.25 * n))
        axes = list(axes[:, 0])
    else:
        fig, axes = ax.figure, [ax]
    if helix is not False and helix is not None:
        excess = (obs - exp).sum(axis=0)
        anchor = float(x[int(np.argmax(excess))]) if helix is True else float(helix)
        start = anchor - math.floor((anchor - x[0]) / _HELIX) * _HELIX
        guides = np.arange(start, x[-1] + 0.5, _HELIX)
    else:
        guides = np.array([])
    top = max(float(obs.max()), float(exp.max()), 1.0)
    for i, (panel, label) in enumerate(zip(axes, labels)):
        for g in guides:
            panel.axvline(g, color=_GUIDE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
        panel.bar(x, obs[i], width=0.8 if x.size <= 150 else 1.0, color=colour, linewidth=0, zorder=2)
        # Distances that cannot be counted, such as those of overlapping
        # hits, are left without a line.
        panel.plot(x, np.where(exp[i] > 0, exp[i], np.nan), color=_EXPECTED, linewidth=1.3, zorder=3)
        if arrangements:
            panel.text(0.01, 0.97, _arrows(i, *shown, homotypic), transform=panel.transAxes, ha="left",
                       va="top", fontsize=text_size, color="#1A1A1A")
        total = int(obs[i].sum())
        panel.text(0.99, 0.97, f"{total:,} pair{'s' if total != 1 else ''}", transform=panel.transAxes,
                   ha="right", va="top", fontsize=text_size - 1, color=_MUTED)
        panel.set_ylim(0, top * 1.22)
        panel.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        panel.tick_params(labelsize=text_size - 1)
        _tidy(panel)
    axes[-1].set_xlim(x[0] - 0.8, x[-1] + 0.8)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    axes[-1].set_xlabel("Distance between centres (bp)" if distance == "centre" else "Gap between hits (bp)")
    handles = [Patch(facecolor=colour, label="Observed"), Line2D([], [], color=_EXPECTED, linewidth=1.3,
                                                                label="Expected")]
    if own_figure:
        fig.supylabel("Pairs", fontsize=plt.rcParams["axes.labelsize"])
        axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, frameon=False,
                       fontsize=text_size, borderaxespad=0.2)
        fig.tight_layout(h_pad=0.6)
        return fig
    axes[0].set_ylabel("Pairs")
    axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=text_size)
    return axes[0]


__all__ = ["pair_spacings", "plot_pair_spacing"]
