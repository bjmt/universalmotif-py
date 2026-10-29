"""Contribution scores around motif hits, and plots of them, for judging
which sites a sequence model relies on and whether it looks beyond them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._analysis import _sequence_pairs
from ._deep import ContributionScores, _letters_first, _seqlet_regions, _to_numpy
from ._plot import _pyplot
from ._plot_seqlets import _logo_panel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LETTERS = "ACGT"
_LINE = "#255C99"
_CHUNK = 8192


@dataclass(frozen=True, eq=False)
class HitContributions:
    """Contribution scores around motif hits, from
    [`hit_contributions()`][universalmotif.hit_contributions].

    ``hits`` is the hit table with a ``contribution`` column added, the
    total observed contribution of the letters of each hit. ``observed`` is
    an array of shape (hits, columns) holding the contribution of the letter
    present at each position of a window around each hit, and
    ``hypothetical``, when kept, one of shape (hits, letters, columns) with
    the scores of every letter. Windows are read along each hit's strand, so
    that every row reads like its motif, and start ``flank`` positions
    before the hit. Hits of a motif narrower than the widest in the table
    have their windows padded with NaN on the right, as do positions
    beyond the ends of a sequence.

    Indexing with a motif name, or a list of names, gives the hits of those
    motifs, and indexing with a boolean mask or row numbers gives those
    rows, each with the columns trimmed to the hits kept.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> hits = pl.DataFrame({"motif": ["a", "b"], "start": [2, 5], "end": [5, 7],
    ...                      "contribution": [1.5, 0.5]})
    >>> hc = um.HitContributions(hits, np.zeros((2, 5)), None, 1)
    >>> hc
    HitContributions(2 hits of 2 motifs, 3 wide with flanks of 1)
    >>> hc["b"].observed.shape
    (1, 4)
    """

    hits: pl.DataFrame
    observed: np.ndarray
    hypothetical: np.ndarray | None
    flank: int

    @property
    def width(self) -> int:
        """The width of the widest hit, the columns between the flanks."""
        return self.observed.shape[-1] - 2 * self.flank

    @property
    def positions(self) -> np.ndarray:
        """The position of each column relative to the first letter of the
        hits, from ``-flank`` up."""
        return np.arange(-self.flank, self.width + self.flank)

    def __len__(self) -> int:
        return self.hits.height

    def __repr__(self) -> str:
        n = len(self)
        motifs = self.hits["motif"].n_unique() if "motif" in self.hits.columns and n else 0
        what = f"{n:,} hit{'s' if n != 1 else ''}"
        if motifs:
            what += f" of {motifs:,} motif{'s' if motifs != 1 else ''}"
        return f"HitContributions({what}, {self.width} wide with flanks of {self.flank})"

    def __getitem__(self, key: Any) -> HitContributions:
        if isinstance(key, str) or (isinstance(key, Sequence) and key and all(isinstance(k, str) for k in key)):
            names = [key] if isinstance(key, str) else list(key)
            if "motif" not in self.hits.columns:
                raise KeyError("the hits have no motif column to pick motifs by")
            motif = self.hits["motif"].cast(pl.String)
            missing = [m for m in names if not (motif == m).any()]
            if missing:
                raise KeyError(f"no hits of {', '.join(map(repr, missing))}")
            rows = np.flatnonzero(motif.is_in(names).to_numpy())
        else:
            if isinstance(key, pl.Series):
                key = key.to_numpy()
            rows = np.asarray(key)
            if rows.dtype == bool:
                if rows.shape != (len(self),):
                    raise IndexError(f"a mask must have one value for each of the {len(self)} hits")
                rows = np.flatnonzero(rows)
            elif rows.dtype.kind not in "iu":
                raise TypeError("index hits with motif names, a boolean mask or row numbers")
        hits = self.hits[rows]
        widest = int((hits["end"] - hits["start"]).max() or 0) if hits.height else 0
        keep = widest + 2 * self.flank
        return HitContributions(hits, self.observed[rows, :keep],
                                None if self.hypothetical is None else self.hypothetical[rows, :, :keep],
                                self.flank)


@dataclass
class _Scores:
    """Contribution scores of a set of sequences as one array, padded to the
    longest sequence: ``values`` of shape (sequences, length) for observed
    scores or (sequences, letters, length), and the letters of the
    sequences as codes 0 to 3 for ACGT and 4 for anything else, when known."""

    values: np.ndarray
    codes: np.ndarray | None
    lengths: np.ndarray
    names: list[str] | None
    ranges: pl.DataFrame | None
    by_name: bool = False


def _codes(seqs: list[str], length: int) -> np.ndarray:
    lookup = np.full(256, 4, dtype=np.uint8)
    for i, letter in enumerate(_LETTERS):
        lookup[ord(letter)] = lookup[ord(letter.lower())] = i
    lookup[ord("U")] = lookup[ord("u")] = 3
    out = np.full((len(seqs), length), 4, dtype=np.uint8)
    if len({len(s) for s in seqs}) == 1 and len(seqs[0]) == length:
        raw = np.frombuffer("".join(seqs).encode("latin-1", "replace"), dtype=np.uint8)
        return lookup[raw].reshape(len(seqs), length)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = lookup[np.frombuffer(s.encode("latin-1", "replace"), dtype=np.uint8)]
    return out


def _sequence_codes(sequences: Any, n: int, length: int) -> tuple[np.ndarray, list[str] | None]:
    """The letters of ``sequences``, given as text or one-hot arrays, as
    codes, and their names when given as a mapping."""
    if isinstance(sequences, (str, Mapping)) or (isinstance(sequences, Sequence) and sequences
                                                 and isinstance(sequences[0], str)):
        names, seqs = _sequence_pairs(sequences)
        if not isinstance(sequences, Mapping):
            names = None
        if len(seqs) != n:
            raise ValueError(f"{len(seqs)} sequences for the scores of {n}")
        if max(map(len, seqs), default=0) > length:
            raise ValueError("the sequences are longer than their scores")
        return _codes(seqs, length), names
    x = _letters_first(_to_numpy(sequences, dtype=None), len(_LETTERS), "the one-hot sequences")
    if x.shape[0] != n or x.shape[-1] > length:
        raise ValueError(f"one-hot sequences of shape {x.shape} for the scores of {n} sequences")
    codes = np.where(x.max(axis=1) > 0, x.argmax(axis=1), 4).astype(np.uint8)
    if codes.shape[1] < length:
        codes = np.pad(codes, ((0, 0), (0, length - codes.shape[1])), constant_values=4)
    return codes, None


def _gather_scores(contributions: Any, sequences: Any, ranges: Any) -> _Scores:
    names = None
    if isinstance(contributions, ContributionScores):
        if sequences is not None:
            raise ValueError("ContributionScores hold their sequences; leave out sequences")
        values = _letters_first(np.asarray(contributions.scores), len(_LETTERS), "the scores")
        n, length = values.shape[0], values.shape[-1]
        codes = _codes(list(contributions.sequences), length)
        names = list(contributions.names)
        lengths = np.array([len(s) for s in contributions.sequences], dtype=np.int64)
        if ranges is None:
            ranges = contributions.ranges
        return _Scores(values, codes, lengths, names, None if ranges is None else _seqlet_regions(ranges))
    if isinstance(contributions, Mapping):
        names = [str(k) for k in contributions]
        items = [_to_numpy(v, dtype=None) for v in contributions.values()]
    elif isinstance(contributions, (list, tuple)):
        items = [_to_numpy(v, dtype=None) for v in contributions]
    else:
        items = None
    if items is None:
        values = _to_numpy(contributions, dtype=None)
        if values.ndim == 3:
            values = _letters_first(values, len(_LETTERS), "the scores")
        elif values.ndim != 2:
            raise ValueError("contributions must have shape (sequences, 4, length) or (sequences, length)")
        lengths = np.full(values.shape[0], values.shape[-1], dtype=np.int64)
    else:
        if not items:
            raise ValueError("no contribution scores given")
        dims = {x.ndim for x in items}
        if dims == {1}:
            lengths = np.array([x.shape[0] for x in items], dtype=np.int64)
            values = np.zeros((len(items), int(lengths.max())), dtype=np.float32)
            for i, x in enumerate(items):
                values[i, :len(x)] = x
        elif dims == {2}:
            items = [_letters_first(x, len(_LETTERS), "the scores of each sequence")[0] for x in items]
            lengths = np.array([x.shape[1] for x in items], dtype=np.int64)
            values = np.zeros((len(items), len(_LETTERS), int(lengths.max())), dtype=np.float32)
            for i, x in enumerate(items):
                values[i, :, :x.shape[1]] = x
        else:
            raise ValueError("the scores of each sequence must have shape (4, length) or (length,)")
    if values.dtype.kind != "f":
        values = values.astype(np.float32)
    n, length = values.shape[0], values.shape[-1]
    codes = None
    if sequences is not None:
        codes, seq_names = _sequence_codes(sequences, n, length)
        names = names or seq_names
    elif values.ndim == 3:
        # Scores of every letter only make sense with the letters present.
        sample = values[: max(1, (1 << 22) // max(1, 4 * length))]
        if (np.count_nonzero(sample, axis=1) > 1).any():
            raise ValueError("the contributions have scores for several letters at a position, as "
                             "hypothetical scores do; give the sequences to pick the letters present")
    return _Scores(values, codes, lengths, names, None if ranges is None else _seqlet_regions(ranges),
                   by_name=names is not None)


def _hit_sequences(hits: pl.DataFrame, scores: _Scores) -> np.ndarray:
    """The sequence of each hit, as a row of the scores: by name when the
    sequences were given by name or every hit's name is known, and
    otherwise by ``sequence_i``."""
    n = scores.values.shape[0]
    if scores.names is not None and "sequence" in hits.columns:
        lookup = pl.DataFrame({"sequence": scores.names, "_row": np.arange(n, dtype=np.int64)})
        if lookup["sequence"].is_duplicated().any():
            raise ValueError("the contributions have repeated sequence names")
        rows = hits.select(pl.col("sequence").cast(pl.String)).join(lookup, on="sequence", how="left",
                                                                    maintain_order="left")["_row"]
        if rows.null_count() == 0:
            return rows.to_numpy()
        if scores.by_name or "sequence_i" not in hits.columns:
            missing = hits["sequence"].filter(rows.is_null()).unique(maintain_order=True).to_list()
            raise ValueError(f"no contributions for the sequences {', '.join(map(str, missing[:5]))}")
    elif scores.by_name:
        raise ValueError("hits needs a sequence column to match the names of the contributions")
    if "sequence_i" not in hits.columns:
        raise ValueError("hits needs a sequence_i column, or sequence names matching the contributions")
    idx = hits["sequence_i"]
    if idx.null_count():
        raise ValueError("sequence_i cannot contain nulls")
    idx = idx.cast(pl.Int64).to_numpy()
    if len(idx) and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"hits refer to sequence {int(idx.max())}, counted from 0, but there are "
                         f"contributions for only {n}")
    return idx


def hit_contributions(
    hits: pl.DataFrame,
    contributions: Any,
    sequences: Any = None,
    flank: int = 0,
    hypothetical: bool = False,
    ranges: Any = None,
) -> HitContributions:
    """The contribution scores a sequence model gives each motif hit and the
    positions around it.

    For each hit, such as those of
    [`scan_sequences()`][universalmotif.scan_sequences] or the seqlets of
    [`read_seqlets()`][universalmotif.read_seqlets], this reads the scores
    in a window of the hit and ``flank`` positions on either side, along
    the hit's strand so that hits on the ``-`` strand are reverse
    complemented and every row reads like the motif. The observed scores,
    the contribution of the letter present at each position, are always
    kept, and with ``hypothetical`` the scores of every letter too. The
    total observed contribution of the letters of each hit, without the
    flanks, is added to the hits as a ``contribution`` column, which makes
    it easy to compare with the hits' scores. Positions beyond the ends of
    a sequence are NaN.

    Windows are aligned on the first letter of each hit. They are as wide
    as the widest hit plus the flanks, and those of motifs narrower than
    the widest are padded with NaN on the right, so that indexing the
    result with a motif name gives windows the width of that motif.

    Hits pick their sequence by its ``sequence`` name when the contributions
    or sequences are a mapping from names, or are a ContributionScores
    whose names every hit shares, and otherwise with ``sequence_i``,
    counted from 0. Hits with a ``chrom`` column, such as seqlets read with
    regions, are taken to be in genome positions, and are placed with the
    genomic ranges of the contributions, which a ContributionScores read
    from chromBPNet files knows and ``ranges`` gives otherwise.

    Parameters
    ----------
    hits
        A table of hits with ``start``, ``end`` and ``strand`` columns,
        positions counted from 0 with the end left out, and ``motif``,
        ``sequence_i`` or ``sequence`` columns.
    contributions
        The scores, as a [`ContributionScores`][universalmotif.ContributionScores],
        an array of shape (sequences, 4, length) with the letters in the
        order ACGT, an array of shape (sequences, length) of observed
        scores such as those read from a bigWig file with
        [`signal_values()`][universalmotif.signal_values], or a list or
        mapping of such arrays, one per sequence, which may differ in
        length.
    sequences
        The sequences the scores are for, as text or one-hot encoded, to
        pick the letters present from scores of every letter. Without them,
        scores of every letter are taken to be projected, zero for all but
        the letter present, and the observed score is their sum.
    flank
        The number of positions on either side of each hit to read too.
    hypothetical
        Also keep the scores of every letter, which are the projected
        scores for projected contributions. They take four times the
        memory of the observed scores, 180 MB for 100,000 hits with flanks
        of 50.
    ranges
        The genomic range of each sequence of the contributions, as a BED
        file or range table, for hits in genome positions.

    Returns
    -------
    HitContributions
        The hits with their total contributions, and the scores around them.

    Examples
    --------
    >>> import numpy as np
    >>> import universalmotif as um
    >>> seqs = ["TTGATAAGTT", "AACTTATCAA"]
    >>> weight = np.array([0, 0, 1, 1, 1, 1, 1, 0.5, 0, 0], dtype=np.float32)
    >>> scores = um.one_hot(seqs) * np.stack([weight, weight[::-1]])[:, None, :]
    >>> hits = um.scan_sequences(um.create_motif("GATAA"), seqs, pvalue=0.001)
    >>> hc = um.hit_contributions(hits, scores, flank=2)
    >>> hc.hits.select("sequence", "start", "strand", "contribution")
    shape: (2, 4)
    ┌──────────┬───────┬────────┬──────────────┐
    │ sequence ┆ start ┆ strand ┆ contribution │
    │ ---      ┆ ---   ┆ ---    ┆ ---          │
    │ str      ┆ i64   ┆ str    ┆ f64          │
    ╞══════════╪═══════╪════════╪══════════════╡
    │ seq1     ┆ 2     ┆ +      ┆ 5.0          │
    │ seq2     ┆ 3     ┆ -      ┆ 5.0          │
    └──────────┴───────┴────────┴──────────────┘
    >>> hc.positions.tolist()
    [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    >>> hc.observed.tolist()
    [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]]
    """
    for col in ("start", "end"):
        if col not in hits.columns:
            raise ValueError(f"hits needs a {col} column")
    flank = int(flank)
    if flank < 0:
        raise ValueError("flank must be 0 or more")
    scores = _gather_scores(contributions, sequences, ranges)
    if hypothetical and scores.values.ndim != 3:
        raise ValueError("hypothetical needs scores for every letter, of shape (sequences, 4, length)")
    if hits["start"].null_count() or hits["end"].null_count():
        raise ValueError("start and end cannot contain nulls")
    start = hits["start"].cast(pl.Int64).to_numpy()
    end = hits["end"].cast(pl.Int64).to_numpy()
    if (end < start).any():
        raise ValueError("all hits must have start <= end")
    width = end - start
    seq = _hit_sequences(hits, scores) if hits.height else np.zeros(0, dtype=np.int64)
    if "chrom" in hits.columns:
        if scores.ranges is None:
            raise ValueError("the hits are in genome positions, with a chrom column, so give the "
                             "genomic ranges of the contributions with ranges")
        table = scores.ranges
        if table.height != scores.values.shape[0]:
            raise ValueError(f"{table.height} ranges for the contributions of {scores.values.shape[0]} sequences")
        if (table["strand"] == "-").any():
            raise ValueError("ranges on the - strand are not supported; give the ranges as +")
        chrom = table["chrom"].cast(pl.String).gather(seq)
        if (chrom != hits["chrom"].cast(pl.String)).any():
            raise ValueError("some hits are on another chromosome than the range of their sequence; "
                             "check that the ranges are in the order of the contributions")
        offset = table["start"].cast(pl.Int64).to_numpy()[seq]
        start, end = start - offset, end - offset
    minus = ((hits["strand"].cast(pl.String) == "-").fill_null(False).to_numpy() if "strand" in hits.columns
             else np.zeros(hits.height, dtype=bool))
    if minus.any() and scores.values.ndim == 3 and hypothetical and scores.values.shape[1] != 4:
        raise ValueError("reverse complementing needs scores of four letters, ACGT")

    # Hits of one motif share their window width, that of the widest of them.
    if "motif" in hits.columns and hits.height:
        span = hits.select(pl.max_horizontal(pl.col("end") - pl.col("start"), pl.lit(0))
                           .max().over(pl.col("motif").cast(pl.String))).to_series().to_numpy()
    else:
        span = np.full(hits.height, int(width.max()) if hits.height else 0)
    span = span.astype(np.int64) + 2 * flank
    columns = int(span.max()) if hits.height else 2 * flank
    observed = np.full((hits.height, columns), np.nan, dtype=np.float32)
    letters = (np.full((hits.height, scores.values.shape[1], columns), np.nan, dtype=np.float32)
               if hypothetical else None)
    total = np.zeros(hits.height, dtype=np.float64)
    k = np.arange(columns)
    values, codes = scores.values, scores.codes
    for a in range(0, hits.height, _CHUNK):
        b = min(a + _CHUNK, hits.height)
        s, m = seq[a:b, None], minus[a:b, None]
        pos = np.where(m, end[a:b, None] - 1 + flank - k, start[a:b, None] - flank + k)
        ok = (pos >= 0) & (pos < scores.lengths[s]) & (k < span[a:b, None])
        pos = np.where(ok, pos, 0)
        if values.ndim == 2:
            obs = values[s, pos]
        elif codes is not None:
            c = codes[s, pos]
            obs = np.where(c < 4, values[s, np.minimum(c, 3), pos], 0)
        else:
            obs = None
        if hypothetical or obs is None:
            every = values[s, :, pos]
            if obs is None:
                obs = every.sum(axis=-1)
            if hypothetical:
                every = every.transpose(0, 2, 1)
                every[m[:, 0]] = every[m[:, 0], ::-1]
                letters[a:b] = np.where(ok[:, None, :], every, np.nan)
        obs = np.where(ok, obs, np.nan)
        observed[a:b] = obs
        core = (k >= flank) & (k < flank + width[a:b, None])
        total[a:b] = np.where(core, obs, 0.0).sum(axis=1, dtype=np.float64)
    out = hits.with_columns(pl.Series("contribution", total, dtype=pl.Float64))
    return HitContributions(out, observed, letters, flank)


def _motif_order(hits: pl.DataFrame) -> tuple[list[str], list[int]]:
    """Motif names in the order of the scan, and their places, which set
    their colours as in plot_hits()."""
    if "motif" not in hits.columns:
        return [""], [0]
    if "motif_i" in hits.columns:
        pairs = hits.select(pl.col("motif").cast(pl.String), pl.col("motif_i").cast(pl.Int64))
        pairs = pairs.unique("motif", keep="first", maintain_order=True).sort("motif_i", maintain_order=True)
        return pairs["motif"].to_list(), pairs["motif_i"].to_list()
    names = hits["motif"].cast(pl.String).unique(maintain_order=True).to_list()
    return names, list(range(len(names)))


def _palette(names: list[str], places: list[int], colours: Any) -> dict[str, str]:
    from ._plot_hits import _motif_colours

    if colours is None:
        return _motif_colours(names, places) if len(names) > 1 else {names[0]: _LINE}
    if isinstance(colours, str):
        return dict.fromkeys(names, colours)
    missing = [m for m in names if m not in colours]
    if missing:
        raise ValueError(f"colours has no colours for: {', '.join(missing[:5])}")
    return {m: colours[m] for m in names}


def _pick_motifs(hits: pl.DataFrame, motifs: str | Sequence[str] | None) -> list[str]:
    names, _ = _motif_order(hits)
    if motifs is None:
        return names
    wanted = [motifs] if isinstance(motifs, str) else list(motifs)
    missing = [m for m in wanted if m not in names]
    if missing:
        shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        raise ValueError(f"no hits of {', '.join(map(repr, missing[:5]))}; the hits have {shown}")
    return list(dict.fromkeys(wanted))


def _band(x: np.ndarray, band: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    """The mean of each column, ignoring NaN, and the half-width of the band
    around it."""
    n = np.sum(~np.isnan(x), axis=0)
    filled = n > 0
    mean = np.full(x.shape[1], np.nan)
    mean[filled] = np.nansum(x[:, filled], axis=0) / n[filled]
    if band is None:
        return mean, None
    dev = np.full(x.shape[1], np.nan)
    two = n > 1
    dev[two] = np.sqrt(np.nansum((x[:, two] - mean[two]) ** 2, axis=0) / (n[two] - 1))
    if band == "sd":
        return mean, dev
    se = dev / np.sqrt(np.maximum(n, 1))
    return mean, se if band == "se" else 1.96 * se


def _relative_ticks(lo: float, hi: float) -> list[int]:
    from ._plot import _nice_step

    step = _nice_step(max(1, int(hi - lo)), 7)
    first = int(np.ceil(lo / step)) * step
    return list(range(first, int(np.floor(hi)) + 1, step))


def plot_hit_profile(
    contributions: HitContributions,
    motifs: str | Sequence[str] | None = None,
    band: str | None = "ci",
    origin: str = "start",
    heatmap: bool = True,
    logo: bool | None = None,
    max_hits: int | None = 500,
    vmax: float | None = None,
    colours: str | Mapping[str, str] | None = None,
    title: str | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw the mean contribution at each position in and around the hits of
    a motif, to see whether a model uses the letters beyond it.

    The line is the mean observed contribution at each position of the
    windows from [`hit_contributions()`][universalmotif.hit_contributions],
    with a band around it, the hit itself shaded and its flanks on either
    side, so that contributions that stay above zero past the edges of the
    hit show that the model reads more than the motif, such as a spacer or
    a partner site. Below it, a heatmap shows the hits themselves, one row
    each sorted by their total contribution from the largest down, and
    above it, when the hypothetical scores were kept, a logo of the mean
    hypothetical contribution of every letter shows which letters the model
    would favour at each position. Several motifs are drawn side by side,
    each with its own panels.

    Needs the ``plot`` extra.

    Parameters
    ----------
    contributions
        Contributions around hits from
        [`hit_contributions()`][universalmotif.hit_contributions], read
        with flanks.
    motifs
        The motif or motifs to draw, by default all of them in the order of
        the scan.
    band
        The band around the mean: ``"ci"`` for its 95% confidence interval,
        ``"se"`` for one standard error, ``"sd"`` for one standard deviation
        of the hits, or ``None`` for none.
    origin
        Number positions from the first letter of the hits, ``"start"``, or
        from their centre, ``"centre"``.
    heatmap
        Draw the heatmap of the hits below the mean.
    logo
        Draw the logo of the mean hypothetical contributions above the
        mean, by default when the hypothetical scores were kept.
    max_hits
        Draw at most this many rows in the heatmap, picked evenly along the
        sorted hits so that the whole range is shown, or all of them with
        ``None``. The mean and logo are always made from every hit.
    vmax
        The contribution at either end of the heatmap's colour scale, by
        default the 99.5th percentile of the absolute contributions drawn.
    colours
        One colour for the lines, or a mapping from motif names to colours.
        By default a single motif is drawn in blue, and several in the
        colours [`plot_hits()`][universalmotif.plot_hits] gives them.
    title
        The title above the plot, by default the name of the motif, or
        nothing when there are several, each panel then being named after
        its motif.
    text_size
        The size of the text, in points.
    figsize
        The size of the figure in inches, by default sized to the motifs.
    ax
        Axes to draw the mean into, one line per motif, without the heatmap
        or logo. They are returned, and otherwise a new figure is.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure, or the axes when ``ax`` is given.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> seqs = ["".join(rng.choice(list("ACGT"), 40)) for _ in range(30)]
    >>> seqs = [s[:15] + "GATAAG" + s[21:] for s in seqs]
    >>> scores = rng.normal(0, 0.05, (30, 4, 40)).astype(np.float32)
    >>> scores[:, :, 13:23] += 0.3
    >>> hits = um.scan_sequences(um.create_motif("GATAAG"), seqs, pvalue=0.0005, rc=False)
    >>> hc = um.hit_contributions(hits, scores, seqs, flank=8, hypothetical=True)
    >>> fig = um.plot_hit_profile(hc)
    >>> [a.get_ylabel() for a in fig.axes[:3]]
    ['Hypothetical', 'Mean contribution', 'Hit, by contribution']
    >>> plt.close(fig)
    """
    if not isinstance(contributions, HitContributions):
        raise TypeError("contributions must come from hit_contributions()")
    if band not in ("ci", "se", "sd", None):
        raise ValueError('band must be "ci", "se", "sd" or None')
    if origin not in ("start", "centre", "center"):
        raise ValueError('origin must be "start" or "centre"')
    if max_hits is not None and max_hits < 1:
        raise ValueError("max_hits must be 1 or more, or None for every hit")
    if len(contributions) == 0:
        raise ValueError("there are no hits to draw")
    if logo and contributions.hypothetical is None:
        raise ValueError("the logo needs the hypothetical scores; read them with "
                         "hit_contributions(..., hypothetical=True)")
    hits = contributions.hits
    names = _pick_motifs(hits, motifs)
    order_names, places = _motif_order(hits)
    palette = _palette(names, [places[order_names.index(m)] for m in names], colours)
    flank = contributions.flank
    if logo is None:
        logo = contributions.hypothetical is not None and ax is None
    parts = [contributions[m] if "motif" in hits.columns else contributions for m in names]
    widths = [p.width for p in parts]

    def shift(w: int) -> float:
        return (w - 1) / 2 if origin != "start" else 0.0

    plt = _pyplot()
    import matplotlib
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    def axis_positions(target, w: int) -> None:
        # Columns are drawn at 1, 2, ... and labelled relative to the origin.
        lo, hi = 1 - flank - shift(w) - 1, w + flank - shift(w) - 1
        rel = _relative_ticks(lo, hi)
        target.xaxis.set_major_locator(FixedLocator([r + flank + shift(w) + 1 for r in rel]))
        target.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - flank - shift(w) - 1:g}"))

    label = "Position from hit centre" if origin != "start" else "Position from hit start"
    if ax is not None:
        fig = ax.figure
        same = len(set(widths)) == 1
        lo = min(-flank - shift(w) for w in widths)
        hi = max(w + flank - 1 - shift(w) for w in widths)
        for name, part, w in zip(names, parts, widths):
            mean, half = _band(part.observed.astype(np.float64), band)
            x = np.arange(part.observed.shape[1]) - flank - shift(w)
            colour = palette[name]
            if half is not None:
                ax.fill_between(x, mean - half, mean + half, color=colour, alpha=0.2, linewidth=0)
            ax.plot(x, mean, color=colour, linewidth=1.4, label=name or None)
            if not same:
                for edge in (-0.5 - shift(w), w - 0.5 - shift(w)):
                    ax.axvline(edge, color=colour, linewidth=0.8, linestyle=(0, (3, 2)))
        if same:
            ax.axvspan(-0.5 - shift(widths[0]), widths[0] - 0.5 - shift(widths[0]), color="0.92",
                       zorder=0, linewidth=0)
        ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_xlabel(label, fontsize=text_size)
        ax.set_ylabel("Mean contribution", fontsize=text_size)
        ax.tick_params(labelsize=text_size)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(names) > 1:
            ax.legend(frameon=False, fontsize=text_size)
        if title is not None:
            ax.set_title(title, loc="left", fontsize=text_size + 1)
        return ax

    shown = []
    for part in parts:
        key = part.hits["contribution"].fill_nan(None).fill_null(-np.inf).to_numpy()
        order = np.argsort(-key, kind="stable")
        if max_hits is not None and len(order) > max_hits:
            order = order[np.round(np.linspace(0, len(order) - 1, max_hits)).astype(np.int64)]
        shown.append(part.observed[order])
    if heatmap and vmax is None:
        drawn = np.abs(np.concatenate([s.ravel() for s in shown]))
        drawn = drawn[~np.isnan(drawn)]
        vmax = float(np.percentile(drawn, 99.5)) if drawn.size else 1.0
    vmax = vmax if vmax and vmax > 0 else 1.0

    rows = [r for r, on in (("logo", logo), ("mean", True), ("heat", heatmap)) if on]
    heights = {"logo": 1.0, "mean": 1.6, "heat": 3.2}
    cols = [w + 2 * flank for w in widths]
    if figsize is None:
        panel = [min(max(2.4, 0.07 * c + 1.0), 6.5) for c in cols]
        figsize = (sum(panel) + 0.75 * len(panel) + 0.6 + (0.6 if heatmap else 0),
                   sum(heights[r] for r in rows) + 1.1)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(len(rows), len(names), height_ratios=[heights[r] for r in rows],
                            width_ratios=cols, wspace=0.2, hspace=0.1)
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#E6E6E6")

    means, logos = [], []
    for j, (name, part, w) in enumerate(zip(names, parts, widths)):
        c = cols[j]
        panel_axes = {}
        first = None
        for i, r in enumerate(rows):
            panel_axes[r] = fig.add_subplot(grid[i, j], sharex=first)
            first = first or panel_axes[r]
        core = (flank + 0.5, flank + w + 0.5)
        mean_ax = panel_axes["mean"]
        mean, half = _band(part.observed.astype(np.float64), band)
        x = np.arange(1, c + 1)
        colour = palette[name]
        mean_ax.axvspan(*core, color="0.92", zorder=0, linewidth=0)
        mean_ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
        if half is not None:
            mean_ax.fill_between(x, mean - half, mean + half, color=colour, alpha=0.22, linewidth=0, zorder=2)
        mean_ax.plot(x, mean, color=colour, linewidth=1.4, zorder=3)
        mean_ax.set_ylabel("Mean contribution" if j == 0 else "", fontsize=text_size)
        mean_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        means.append(mean_ax)
        if logo:
            logo_ax = panel_axes["logo"]
            hyp = part.hypothetical
            n = np.sum(~np.isnan(hyp[:, 0, :]), axis=0)
            avg = np.where(n > 0, np.nansum(hyp, axis=0) / np.maximum(n, 1), 0.0)
            _logo_panel(logo_ax, avg, "Hypothetical" if j == 0 else "")
            logo_ax.axvspan(*core, color="0.92", zorder=-1, linewidth=0)
            logos.append(logo_ax)
        if heatmap:
            heat = panel_axes["heat"]
            rows_drawn = shown[j]
            im = heat.imshow(rows_drawn, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                             interpolation="antialiased", interpolation_stage="rgba",
                             extent=(0.5, c + 0.5, len(rows_drawn), 0))
            for edge in core:
                heat.axvline(edge, color="0.25", linewidth=0.6, linestyle=(0, (3, 2)))
            heat.set_ylabel("Hit, by contribution" if j == 0 else "", fontsize=text_size)
            heat.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            heat.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            heat.set_ylim(len(rows_drawn), 0)
        bottom = panel_axes[rows[-1]]
        for a in panel_axes.values():
            a.tick_params(labelsize=text_size - 1)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            if a is not bottom:
                a.tick_params(axis="x", labelbottom=False)
        axis_positions(bottom, w)
        bottom.set_xlim(0.5, c + 0.5)
        bottom.set_xlabel(label, fontsize=text_size)
        top = panel_axes[rows[0]]
        count = f"{len(part):,} hits" if len(part) != 1 else "1 hit"
        if heatmap and len(shown[j]) < len(part):
            count += f", {len(shown[j]):,} drawn"
        heading = name if len(names) > 1 or title is None else title
        top.set_title(heading, loc="left", fontsize=text_size + 1, pad=14)
        top.annotate(count, (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                     ha="left", va="bottom", fontsize=text_size - 1, color="0.35")
    # Motifs share their scales, so that their contributions compare.
    for group in (means, logos):
        if len(group) > 1:
            lo = min(a.get_ylim()[0] for a in group)
            hi = max(a.get_ylim()[1] for a in group)
            for a in group:
                a.set_ylim(lo, hi)
                if group is logos:
                    a.spines["left"].set_bounds(lo, hi)
    if len(names) > 1 and title is not None:
        fig.suptitle(title, x=0.02, ha="left", fontsize=text_size + 2)
    fig.subplots_adjust(left=0.75 / figsize[0], right=1 - (0.85 if heatmap else 0.25) / figsize[0],
                        top=1 - (0.6 if len(names) > 1 and title else 0.45) / figsize[1],
                        bottom=0.5 / figsize[1])
    if heatmap:
        pos = heat.get_position()
        cax = fig.add_axes((pos.x1 + 0.15 / figsize[0], pos.y0, 0.12 / figsize[0], min(pos.height, 1.4 / figsize[1])))
        bar = fig.colorbar(im, cax=cax)
        bar.set_label("Contribution", fontsize=text_size - 1)
        bar.ax.tick_params(labelsize=text_size - 2)
        bar.outline.set_visible(False)
    return fig


def _axis_values(hits: pl.DataFrame, col: str) -> tuple[np.ndarray, str]:
    """A numeric column of the hits, with P-values and the like as
    ``-log10``, from their log10 column when there is one, and its label."""
    labels = {"score": "Motif score", "score_pct": "Motif score (%)", "contribution": "Total contribution",
              "pvalue": "-log10(P-value)", "qvalue": "-log10(q-value)"}
    if col.lower().startswith(("pval", "qval", "eval")):
        if f"log10_{col}" in hits.columns:
            return -hits[f"log10_{col}"].cast(pl.Float64).to_numpy(), labels.get(col, f"-log10({col})")
        if col not in hits.columns:
            raise ValueError(f"the hits have no {col!r} column")
        v = hits[col].cast(pl.Float64).to_numpy()
        return -np.log10(np.clip(v, 5e-324, None)), labels.get(col, f"-log10({col})")
    if col not in hits.columns:
        raise ValueError(f"the hits have no {col!r} column")
    if not hits[col].dtype.is_numeric():
        raise ValueError(f"{col!r} is not a numeric column")
    return hits[col].cast(pl.Float64).fill_null(np.nan).to_numpy(), labels.get(col, col)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    r = pl.DataFrame({"x": x, "y": y}).select(pl.corr("x", "y", method="spearman")).item()
    return float("nan") if r is None else float(r)


def _trend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The median of ``x`` and ``y`` in 20 bins of equal width along ``x``,
    leaving out bins of fewer than 10 hits."""
    if len(x) < 10:
        return np.zeros(0), np.zeros(0)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        return np.array([np.median(x)]), np.array([np.median(y)])
    which = np.clip(((x - lo) / (hi - lo) * 20).astype(np.int64), 0, 19)
    order = np.argsort(which, kind="stable")
    bounds = np.searchsorted(which[order], np.arange(21))
    groups = [order[a:b] for a, b in pairwise(bounds) if b - a >= 10]
    return np.array([np.median(x[g]) for g in groups]), np.array([np.median(y[g]) for g in groups])


def plot_hit_contributions(
    contributions: HitContributions | pl.DataFrame,
    x: str = "score",
    y: str = "contribution",
    motifs: str | Sequence[str] | None = None,
    hexbin: bool = False,
    trend: bool | None = None,
    marginals: bool = True,
    correlation: bool = True,
    point_size: float | None = None,
    alpha: float | None = None,
    colours: str | Mapping[str, str] | None = None,
    title: str | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw the total contribution of each motif hit against its motif score,
    to see which sites a sequence model relies on.

    Each hit is a point at its score, or ``-log10`` P-value with
    ``x="pvalue"``, and the total contribution of its letters from
    [`hit_contributions()`][universalmotif.hit_contributions], coloured by
    motif. Hits that score well but contribute little are sites the model
    ignores, and weak sites that contribute a lot are ones it relies on
    anyway, often because of what lies around them. The legend gives the
    Spearman correlation of the two for each motif, and histograms along
    the edges show how each motif's hits spread along either axis.

    Points are drawn in a random order so that no motif hides another, and
    as an image within the plot, which keeps files small with hundreds of
    thousands of hits. With that many, ``hexbin`` draws the density of all
    hits in grey instead, with a line per motif through the median
    contribution of its hits at each score.

    Needs the ``plot`` extra.

    Parameters
    ----------
    contributions
        Contributions around hits from
        [`hit_contributions()`][universalmotif.hit_contributions], or a hit
        table with a ``contribution`` column or the column given as ``y``.
    x, y
        The columns of the hits to draw along each axis. P-values and
        q-values are drawn as ``-log10``, from their log10 column when the
        table has one.
    motifs
        The motif or motifs to draw, by default all of them.
    hexbin
        Draw the density of the hits in hexagons instead of the points.
    trend
        Draw a line through the median ``y`` of each motif's hits in 20
        bins along ``x``, by default only with ``hexbin``.
    marginals
        Draw the histograms of each motif along the edges, in a figure of
        its own.
    correlation
        Give the Spearman correlation of each motif in the legend.
    point_size, alpha
        The size of the points in points squared and their opacity, by
        default smaller and fainter the more hits there are.
    colours
        One colour for all motifs, or a mapping from motif names to
        colours, by default the colours
        [`plot_hits()`][universalmotif.plot_hits] gives them.
    title
        The title above the plot.
    text_size
        The size of the text, in points.
    figsize
        The size of the figure in inches.
    ax
        Axes to draw into, without the histograms. They are returned, and
        otherwise a new figure is.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure, or the axes when ``ax`` is given.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> score = rng.uniform(5, 12, 200)
    >>> hits = pl.DataFrame({"motif": np.repeat(["GATA", "CTCF"], 100), "score": score,
    ...                      "contribution": 0.2 * score + rng.normal(0, 0.5, 200)})
    >>> fig = um.plot_hit_contributions(hits)
    >>> fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()
    ('Motif score', 'Total contribution')
    >>> [t.get_text() for t in fig.legends[0].get_texts()]  # doctest: +ELLIPSIS
    ['GATA, ρ = 0.67', 'CTCF, ρ = 0.63']
    >>> plt.close(fig)
    """
    hits = contributions.hits if isinstance(contributions, HitContributions) else contributions
    if not isinstance(hits, pl.DataFrame):
        raise TypeError("contributions must come from hit_contributions(), or be a table of hits")
    if hits.height == 0:
        raise ValueError("there are no hits to draw")
    names = _pick_motifs(hits, motifs)
    order_names, places = _motif_order(hits)
    palette = _palette(names, [places[order_names.index(m)] for m in names], colours)
    xv, xlabel = _axis_values(hits, x)
    yv, ylabel = _axis_values(hits, y)
    group = (hits["motif"].cast(pl.String).to_numpy() if "motif" in hits.columns
             else np.full(hits.height, "", dtype=object))
    keep = np.isfinite(xv) & np.isfinite(yv) & np.isin(group, names)
    if not keep.any():
        raise ValueError(f"no hits have finite values of both {x} and {y}")
    if trend is None:
        trend = hexbin
    xv, yv, group = xv[keep], yv[keep], group[keep]
    n = len(xv)
    members = {m: np.flatnonzero(group == m) for m in names}

    plt = _pyplot()
    import matplotlib.patheffects as effects
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    own_figure = ax is None
    marginals = marginals and own_figure
    if own_figure:
        if figsize is None:
            figsize = (6.4, 5.2) if marginals else (6.2, 4.4)
        fig = plt.figure(figsize=figsize)
        if marginals:
            grid = fig.add_gridspec(2, 2, width_ratios=[4.5, 1], height_ratios=[1, 4.5], wspace=0.04,
                                    hspace=0.04)
            ax = fig.add_subplot(grid[1, 0])
            top = fig.add_subplot(grid[0, 0], sharex=ax)
            side = fig.add_subplot(grid[1, 1], sharey=ax)
            corner = grid[0, 1]
        else:
            ax = fig.add_subplot()
    else:
        fig = ax.figure
    ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
    if hexbin:
        image = ax.hexbin(xv, yv, gridsize=60, bins="log", mincnt=1, cmap="Greys", linewidths=0.2,
                          edgecolors="face", zorder=2)
        image.set_rasterized(True)
    else:
        if point_size is None:
            point_size = float(np.clip(16 * np.sqrt(500 / n), 1.5, 16))
        if alpha is None:
            alpha = float(np.clip(np.sqrt(2000 / n), 0.15, 0.85))
        shuffle = np.random.default_rng(0).permutation(n)
        rgba = np.array([to_rgba(palette[m]) for m in names])
        lookup = {m: i for i, m in enumerate(names)}
        colour_i = np.array([lookup[g] for g in group[shuffle]], dtype=np.int64)
        ax.scatter(xv[shuffle], yv[shuffle], s=point_size, c=rgba[colour_i], alpha=alpha,
                   linewidths=0, rasterized=True, zorder=2)
    if trend:
        outline = [effects.Stroke(linewidth=3.2, foreground="white"), effects.Normal()]
        for m in names:
            tx, ty = _trend(xv[members[m]], yv[members[m]])
            if len(tx):
                ax.plot(tx, ty, color=palette[m], linewidth=1.8, path_effects=outline, zorder=4)
    ax.set_xlabel(xlabel, fontsize=text_size)
    ax.set_ylabel(ylabel, fontsize=text_size)
    ax.tick_params(labelsize=text_size)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    labels = []
    for m in names:
        text = m or "hits"
        if correlation:
            r = _spearman(xv[members[m]], yv[members[m]])
            text += ", ρ = NA" if not np.isfinite(r) else f", ρ = {r:.2f}".replace("-0.00", "0.00")
        labels.append(text)
    handles = [Line2D([0], [0], linestyle="-" if trend and hexbin else "", marker="" if trend and hexbin else "o",
                      color=palette[m], markersize=6, linewidth=2) for m in names]
    show_legend = len(names) > 1 or correlation
    if marginals:
        hist = {"histtype": "stepfilled", "density": True, "alpha": 0.25, "linewidth": 0}
        edge = {"histtype": "step", "density": True, "linewidth": 1.0}
        xbins = np.histogram_bin_edges(xv, bins=min(60, max(10, int(np.sqrt(n)))))
        ybins = np.histogram_bin_edges(yv, bins=min(60, max(10, int(np.sqrt(n)))))
        for m in names:
            v = members[m]
            if len(v) < 2:
                continue
            top.hist(xv[v], bins=xbins, color=palette[m], **hist)
            top.hist(xv[v], bins=xbins, color=palette[m], **edge)
            side.hist(yv[v], bins=ybins, color=palette[m], orientation="horizontal", **hist)
            side.hist(yv[v], bins=ybins, color=palette[m], orientation="horizontal", **edge)
        side.axhline(0, color="0.6", linewidth=0.6, zorder=0)
        for a, hidden in ((top, ("top", "right", "left")), (side, ("top", "right", "bottom"))):
            for spine in hidden:
                a.spines[spine].set_visible(False)
        top.tick_params(axis="x", labelbottom=False)
        top.tick_params(axis="y", left=False, labelleft=False)
        side.tick_params(axis="y", labelleft=False)
        side.tick_params(axis="x", bottom=False, labelbottom=False)
        if show_legend:
            box = corner.get_position(fig)
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(box.x0, box.y1), frameon=False,
                       fontsize=text_size, handletextpad=0.3, borderaxespad=0.0, borderpad=0.0)
        if title is not None:
            top.set_title(title, loc="left", fontsize=text_size + 1)
    else:
        if show_legend:
            where = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)} if own_figure else {"loc": "best"}
            target = fig if own_figure else ax
            if own_figure:
                where["bbox_transform"] = ax.transAxes
            target.legend(handles, labels, frameon=False, fontsize=text_size, handletextpad=0.3, **where)
        if title is not None:
            ax.set_title(title, loc="left", fontsize=text_size + 1)
    if own_figure and not marginals:
        fig.tight_layout()
    return fig if own_figure else ax


__all__ = ["HitContributions", "hit_contributions", "plot_hit_contributions", "plot_hit_profile"]
