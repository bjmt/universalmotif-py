"""Genomes read a piece at a time from disk, sequences of genomic ranges, and
bigWig signal over them."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique

_COMPLEMENT = str.maketrans(
    "ACGTUNRYKMSWBDHVacgtunrykmswbdhv",
    "TGCAANYRMKSWVHDBtgcaanyrmkswvhdb",
)


class Genome(Mapping[str, str]):
    """A genome in a FASTA, bgzip or 2bit file, read a piece at a time.

    Plain FASTA files are read through their ``.fai`` index, and bgzipped
    ones through their ``.fai`` and ``.gzi`` indexes, all as written by
    ``samtools faidx``; indexes that are missing are built in memory when
    the genome is opened, which takes a pass over the file. 2bit files carry
    their own index. Only the ranges asked for are read, so a genome of any
    size takes almost no memory until its sequences are used.

    A genome is a read-only mapping from chromosome names to sequences, so
    it can go wherever a dict of sequences can, each chromosome being read
    whole when it is looked up. [`subseq()`][universalmotif.Genome.subseq]
    and [`extract()`][universalmotif.Genome.extract] read just parts of
    them. Letters come back as they are in the file, so soft-masking is kept.
    [`scan_sequences()`][universalmotif.scan_sequences] scans a genome one
    chromosome at a time, and [`match_bkg()`][universalmotif.match_bkg]
    reads only the windows it draws.

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = um.Genome("TAIR10.fa.gz")  # doctest: +SKIP
    >>> genome.subseq("1", 1000, 1020)  # doctest: +SKIP
    'GGTTATAAATTATTAGAAGT'
    """

    __slots__ = ("_inner", "path")

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self._inner = _core._Genome(self.path)

    def _index(self, name: object) -> int:
        i = self._inner.index(name) if isinstance(name, str) else None
        if i is None:
            raise KeyError(name)
        return i

    def __getitem__(self, name: str) -> str:
        i = self._index(name)
        return self._inner.fetch(i, 0, self._inner.lengths()[i])

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner.names())

    def __len__(self) -> int:
        return len(self._inner.names())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._inner.index(name) is not None

    @property
    def lengths(self) -> dict[str, int]:
        """The length of each chromosome."""
        return dict(zip(self._inner.names(), self._inner.lengths()))

    def subseq(self, name: str, start: int, end: int) -> str:
        """Positions ``start`` up to but not including ``end`` of the
        chromosome called ``name``, counted from 0 as in a slice, reading only
        those."""
        i = self._index(name)
        if start < 0:
            raise IndexError("start must be at least 0")
        return self._inner.fetch(i, start, end)

    def extract(self, ranges, nthreads: int = 1) -> dict[str, str]:
        """The sequences of genomic ranges; see
        [`extract_ranges()`][universalmotif.extract_ranges]."""
        return extract_ranges(self, ranges, nthreads=nthreads)

    def _fetch(self, chroms: list[str], starts: list[int], ends: list[int],
               nthreads: int = 1) -> list[str]:
        """Many ranges, with 0-based, half-open positions, read in parallel."""
        index = {c: self._index(c) for c in set(chroms)}
        return self._inner.fetch_many(
            [(index[c], s, e) for c, s, e in zip(chroms, starts, ends)], nthreads
        )

    def __repr__(self) -> str:
        total = sum(self._inner.lengths())
        return (f"Genome({os.path.basename(self.path)!r}, {len(self)} sequences, "
                f"{total:,} letters)")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def extract_ranges(
    genome: Mapping[str, str],
    ranges,
    nthreads: int = 1,
) -> dict[str, str]:
    """The sequences of genomic ranges, such as peaks, from a genome.

    ``genome`` is a [`Genome`][universalmotif.Genome] read from disk,
    [`PackedSequences`][universalmotif.PackedSequences] or any mapping from
    chromosome names to sequences. ``ranges`` is a DataFrame with ``chrom``
    (or ``seqnames``, ``chr`` and the like), ``start`` and ``end`` or
    ``width`` columns, or (chrom, start, end[, strand]) tuples, with
    positions counted from 0 and the end left out, as in BED files and
    Python slices. Ranges on the ``-`` strand are reverse complemented.
    Each sequence is named ``chrom:start-end`` in the same positions, with
    ``(-)`` added on the minus strand, and repeated ranges get ``.1``,
    ``.2`` and so on. A genome on disk reads the ranges with ``nthreads``
    threads.

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = {"chr1": "AACCGGTTACGT"}
    >>> um.extract_ranges(genome, [("chr1", 1, 5), ("chr1", 1, 5, "-")])
    {'chr1:1-5': 'ACCG', 'chr1:1-5(-)': 'CGGT'}
    """
    from ._genomic import _range_table
    from ._packed import PackedSequences

    df = _range_table(ranges, "ranges")
    chroms, starts, ends = df["chrom"].to_list(), df["start"].to_list(), df["end"].to_list()
    lengths = (genome.lengths if isinstance(genome, (Genome, PackedSequences)) else None)
    for c, s, e in zip(chroms, starts, ends):
        if c not in genome:
            raise ValueError(f"the chromosome {c!r} is not in the genome")
        n = lengths[c] if lengths is not None else len(genome[c])
        if s < 0 or e > n or s > e:
            raise ValueError(f"the range {c}:{s}-{e} does not fit in {c} (length {n})")
    if isinstance(genome, Genome):
        seqs = genome._fetch(chroms, starts, ends, nthreads)
    elif isinstance(genome, PackedSequences):
        seqs = [genome.subseq(c, s, e) for c, s, e in zip(chroms, starts, ends)]
    else:
        seqs = [genome[c][s:e] for c, s, e in zip(chroms, starts, ends)]
    strands = df["strand"].to_list()
    seqs = [_reverse_complement(x) if st == "-" else x for x, st in zip(seqs, strands)]
    names = [f"{c}:{s}-{e}" + ("(-)" if st == "-" else "")
             for c, s, e, st in zip(chroms, starts, ends, strands)]
    return dict(zip(_make_unique(names), seqs))


_STATS = ("mean", "max", "min")


def signal_stats(
    bigwig: str | os.PathLike[str] | Mapping[str, str | os.PathLike[str]],
    ranges,
    stat: str = "mean",
    uncovered: float | None = 0.0,
) -> pl.DataFrame:
    """Summarize bigWig signal, such as ATAC or ChIP coverage, over genomic
    ranges.

    ``bigwig`` is the path of one bigWig file, giving a column called
    ``signal``, or a mapping from column names to paths. ``ranges`` is a
    range table or (chrom, start, end[, strand]) tuples, with 0-based,
    half-open positions, as for
    [`extract_ranges()`][universalmotif.extract_ranges]. ``stat`` is
    ``"mean"``, ``"max"`` or ``"min"``. Bases the file has no value for
    count as ``uncovered``, zero by default as is usual for coverage; with
    ``uncovered=None`` they are left out, and a range with no values at all
    gets NaN.

    The values are useful as covariates for
    [`match_bkg()`][universalmotif.match_bkg], which can also compute them
    itself for the windows it draws from a genome.

    Needs the ``bigwig`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> peaks = [("chr1", 1000, 1200), ("chr2", 500, 700)]
    >>> um.signal_stats({"atac": "atac.bw"}, peaks)  # doctest: +SKIP
    """
    try:
        import pybigtools
    except ImportError as e:
        raise ImportError(
            "reading bigWig files needs pybigtools; install it with `pip install "
            "universalmotif[bigwig]`"
        ) from e
    from ._genomic import _range_table

    if stat not in _STATS:
        raise ValueError('stat must be "mean", "max" or "min"')
    files = dict(bigwig) if isinstance(bigwig, Mapping) else {"signal": bigwig}
    if not files:
        raise ValueError("no bigWig files given")
    df = _range_table(ranges, "ranges")
    chroms, starts, ends = df["chrom"].to_list(), df["start"].to_list(), df["end"].to_list()
    out = {}
    for name, path in files.items():
        reader = pybigtools.open(os.fspath(path))
        sizes = reader.chroms()
        values = np.empty(len(chroms))
        for k, (c, s, e) in enumerate(zip(chroms, starts, ends)):
            if s >= e:
                values[k] = np.nan
                continue
            if c not in sizes:
                # A chromosome with no signal at all, such as one left out of
                # the file for having no reads.
                values[k] = np.nan if uncovered is None else uncovered
                continue
            if s < 0 or e > sizes[c]:
                raise ValueError(f"the range {c}:{s}-{e} does not fit in {c} in {path}")
            values[k] = reader.values(c, s, e, bins=1, summary=stat, exact=True,
                                      fillna=np.nan, uncovered=uncovered)[0]
        out[str(name)] = values
    return pl.DataFrame(out)


def signal_values(
    bigwig: str | os.PathLike[str],
    chrom: str,
    start: int,
    end: int,
    uncovered: float | None = 0.0,
) -> np.ndarray:
    """The value of a bigWig file at each position of one genomic range.

    Positions run from ``start`` up to but not including ``end``, counted
    from 0. Bases the file has no value for get ``uncovered``, zero by
    default, or NaN with ``uncovered=None``. Where
    [`signal_stats()`][universalmotif.signal_stats] summarizes signal over
    many ranges, this gives every position of one, such as the per-base
    contribution scores some models write as bigWig tracks, which can be
    drawn along the genome's sequence with
    [`view_attributions()`][universalmotif.view_attributions].

    Needs the ``bigwig`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = um.Genome("genome.fa.gz")  # doctest: +SKIP
    >>> scores = um.signal_values("scores.bw", "1", 1000, 1100)  # doctest: +SKIP
    >>> um.view_attributions(scores, genome.subseq("1", 1000, 1100), start=1000)  # doctest: +SKIP
    """
    try:
        import pybigtools
    except ImportError as e:
        raise ImportError(
            "reading bigWig files needs pybigtools; install it with `pip install "
            "universalmotif[bigwig]`"
        ) from e
    start, end = int(start), int(end)
    reader = pybigtools.open(os.fspath(bigwig))
    sizes = reader.chroms()
    if chrom not in sizes:
        raise ValueError(f"the chromosome {chrom!r} is not in {bigwig}")
    if start < 0 or end > sizes[chrom] or start >= end:
        raise ValueError(f"the range {chrom}:{start}-{end} does not fit in {chrom} "
                         f"(length {sizes[chrom]})")
    values = reader.values(chrom, start, end, fillna=uncovered)
    return np.asarray(values, dtype=float)


__all__ = ["Genome", "extract_ranges", "signal_stats", "signal_values"]
