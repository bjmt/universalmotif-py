"""Hits in genomic coordinates, and ranges handed to PyRanges and polars-bio."""

from __future__ import annotations

import re

import polars as pl

from ._genomic import _CHROM_COLUMNS, _range_table, _ranges_frame

_REGION = re.compile(r"^(.+):(\d+)-(\d+)(\(-\))?(?:\.\d+)?$")


def genomic_hits(hits: pl.DataFrame, ranges=None) -> pl.DataFrame:
    """Hits from scanning genomic ranges, moved to genomic coordinates.

    [`scan_sequences()`][universalmotif.scan_sequences] counts positions
    from the start of each sequence scanned. When those sequences are
    ranges taken from a genome, such as peaks read with
    [`extract_ranges()`][universalmotif.extract_ranges], this puts the hits
    back on the genome: each gets a ``chrom`` column, and ``start``, ``end``
    and ``strand`` become positions on the chromosome's forward strand,
    counted from 1 and inclusive. Hits in ranges that were reverse
    complemented have their strand flipped. The other columns are kept, so
    ``sequence`` still names the range each hit came from.

    By default the ranges are read from the sequence names, which must be
    ``chrom:start-end`` with ``(-)`` after ranges on the minus strand, as
    [`extract_ranges()`][universalmotif.extract_ranges] and
    [`match_bkg()`][universalmotif.match_bkg] name them. Otherwise
    ``ranges`` gives them in the order of the sequences scanned, as a range
    table or (chrom, start, end[, strand]) tuples, and hits are matched to
    them by ``sequence_i``.

    Hits in overlapping ranges are reported once for each, so the same site
    can appear more than once, and
    ``.unique(["motif", "chrom", "start", "end", "strand"], maintain_order=True)``
    keeps one of each.

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = {"chr1": "CCCCCTGACTCACCCCCCCCCC"}
    >>> peaks = um.extract_ranges(genome, [("chr1", 3, 15), ("chr1", 3, 15, "-")])
    >>> hits = um.scan_sequences(um.example_motifs()[1], peaks, pvalue=1e-3, rc=False)
    >>> hits.select("sequence", "start", "end", "strand", "match").rows()
    [('chr1:3-15', 3, 11, '+', 'CTGACTCAC'), ('chr1:3-15(-)', 3, 11, '+', 'GTGAGTCAG')]
    >>> um.genomic_hits(hits).select("chrom", "start", "end", "strand").rows()
    [('chr1', 5, 13, '+'), ('chr1', 5, 13, '-')]
    """
    for col in ("sequence", "start", "end", "strand"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if ranges is None:
        names = hits["sequence"].cast(pl.String).to_list()
        parsed = {}
        for name in dict.fromkeys(names):
            m = _REGION.match(name)
            if m is None:
                raise ValueError(
                    f"the sequence name {name!r} is not chrom:start-end; give the ranges "
                    "with ranges"
                )
            parsed[name] = (m[1], int(m[2]), int(m[3]), "-" if m[4] else "+")
        where = [parsed[n] for n in names]
    else:
        if "sequence_i" not in hits.columns:
            raise ValueError("hits must have a 'sequence_i' column to match them to ranges")
        table = _range_table(ranges, "ranges")
        rows = list(zip(table["chrom"], table["start"], table["end"], table["strand"]))
        idx = hits["sequence_i"].to_list()
        if any(i is None or not 0 <= i < len(rows) for i in idx):
            raise ValueError(f"hits refer to sequences beyond the {len(rows)} ranges given")
        where = [rows[i] for i in idx]
    chrom, start, end, strand = [], [], [], []
    for (c, rs, re_, rstrand), s, e, st in zip(where, hits["start"].to_list(),
                                              hits["end"].to_list(), hits["strand"].to_list()):
        chrom.append(c)
        if rstrand == "-":
            start.append(re_ - e + 1)
            end.append(re_ - s + 1)
            strand.append({"+": "-", "-": "+"}.get(st, st))
        else:
            start.append(rs + s - 1)
            end.append(rs + e - 1)
            strand.append(st)
    out = hits.with_columns(
        pl.Series("start", start, dtype=pl.Int64),
        pl.Series("end", end, dtype=pl.Int64),
        pl.Series("strand", strand, dtype=pl.String),
    )
    if "chrom" in out.columns:
        out = out.drop("chrom")
    out = out.with_columns(pl.Series("chrom", chrom, dtype=pl.String))
    cols = [c for c in out.columns if c != "chrom"]
    at = cols.index("start")
    return out.select(cols[:at] + ["chrom"] + cols[at:])


def _chrom_column(df: pl.DataFrame) -> str:
    chrom = next((c for c in _CHROM_COLUMNS if c in df.columns), None)
    if chrom is None or "start" not in df.columns or "end" not in df.columns:
        raise ValueError(
            f"the table needs a chromosome column ({', '.join(_CHROM_COLUMNS)}), start and end"
        )
    return chrom


def to_pyranges(df: pl.DataFrame):
    """A table of ranges, such as hits or peaks, as a PyRanges object.

    The chromosome column, which can be any of ``chrom``, ``seqnames``,
    ``sequence``, ``chr`` or ``chromosome``, becomes ``Chromosome``, and
    ``start``, ``end`` and ``strand`` become ``Start``, ``End`` and
    ``Strand``, with positions moved from 1-based and inclusive to PyRanges'
    0-based, end-exclusive ones. Other columns are kept. PyRanges objects
    can be given back to any function that takes ranges.

    Needs the ``pyranges`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> gr = um.to_pyranges(um.genomic_hits(hits))  # doctest: +SKIP
    """
    try:
        import pyranges1 as pr
    except ImportError as e:
        raise ImportError(
            "converting to PyRanges needs pyranges1; install it with `pip install "
            "universalmotif[pyranges]`"
        ) from e
    df = _ranges_frame(df)
    if not isinstance(df, pl.DataFrame):
        df = _range_table(df, "the ranges")
    chrom = _chrom_column(df)
    renames = {chrom: "Chromosome", "start": "Start", "end": "End"}
    if "strand" in df.columns:
        renames["strand"] = "Strand"
    clash = [c for c in ("Chromosome", "Start", "End", "Strand") if c in df.columns]
    if clash:
        raise ValueError(f"the table already has columns named {clash}")
    df = df.rename(renames).with_columns(pl.col("Start") - 1)
    first = [c for c in ("Chromosome", "Start", "End", "Strand") if c in df.columns]
    df = df.select(first + [c for c in df.columns if c not in first])
    return pr.PyRanges(df.to_pandas())


def to_polars_bio(df: pl.DataFrame) -> pl.DataFrame:
    """A table of ranges, such as hits or peaks, ready for polars-bio.

    polars-bio works on polars DataFrames with ``chrom``, ``start`` and
    ``end`` columns, and keeps their coordinate system as metadata. This
    renames the chromosome column to ``chrom`` if it has another name, such
    as the ``sequence`` of hits, moves it first, and marks the positions as
    1-based and inclusive, as they are throughout this package, so that
    polars-bio's range operations treat them correctly. polars-bio frames
    can be given back to any function that takes ranges.

    Needs the ``polars-bio`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> import polars_bio as pb  # doctest: +SKIP
    >>> hits = um.to_polars_bio(um.genomic_hits(hits))  # doctest: +SKIP
    >>> pb.overlap(hits, um.to_polars_bio(peaks))  # doctest: +SKIP
    """
    try:
        import polars_bio  # noqa: F401  (registers the metadata namespace)
    except ImportError as e:
        raise ImportError(
            "preparing ranges for polars-bio needs polars-bio; install it with `pip install "
            "universalmotif[polars-bio]`"
        ) from e
    df = _ranges_frame(df)
    if not isinstance(df, pl.DataFrame):
        df = _range_table(df, "the ranges")
    chrom = _chrom_column(df)
    if chrom != "chrom":
        if "chrom" in df.columns:
            raise ValueError("the table has both chrom and another chromosome column")
        df = df.rename({chrom: "chrom"})
    first = ["chrom", "start", "end"]
    df = df.select(first + [c for c in df.columns if c not in first])
    df.config_meta.set(coordinate_system_zero_based=False)
    return df


__all__ = ["genomic_hits", "to_polars_bio", "to_pyranges"]
