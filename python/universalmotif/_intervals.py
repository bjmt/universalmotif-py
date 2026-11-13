"""Reading BED files, hits in genomic coordinates, and ranges handed to
PyRanges and polars-bio."""

from __future__ import annotations

import gzip
import io
import os
import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from ._genomic import _CHROM_COLUMNS, _range_table, _ranges_frame

_REGION = re.compile(r"^(.+):(\d+)-(\d+)(\(-\))?(?:\.\d+)?$")


_BED_FORMATS = {
    "bed": ["chrom", "start", "end", "name", "score", "strand", "thick_start", "thick_end",
            "item_rgb", "block_count", "block_sizes", "block_starts"],
    "narrowpeak": ["chrom", "start", "end", "name", "score", "strand", "signal",
                   "neg_log10_pvalue", "neg_log10_qvalue", "peak"],
    "broadpeak": ["chrom", "start", "end", "name", "score", "strand", "signal",
                  "neg_log10_pvalue", "neg_log10_qvalue"],
    "bedgraph": ["chrom", "start", "end", "value"],
}
_BED_EXTENSIONS = {".bed": "bed", ".narrowpeak": "narrowpeak", ".broadpeak": "broadpeak",
                   ".bedgraph": "bedgraph", ".bg": "bedgraph"}
# Columns where narrowPeak and broadPeak files write -1 for a value not given.
_PEAK_MISSING = ("neg_log10_pvalue", "neg_log10_qvalue", "peak")


def _bed_format(file, format: str | None) -> str:
    if format is not None:
        key = format.lower()
        if key not in _BED_FORMATS:
            raise ValueError("format must be 'bed', 'narrowPeak', 'broadPeak' or 'bedGraph'")
        return key
    name = Path(file).name.lower().removesuffix(".gz")
    return _BED_EXTENSIONS.get(Path(name).suffix, "bed")


def read_bed(file: str | os.PathLike, format: str | None = None) -> pl.DataFrame:
    r"""Read a BED file, or a narrowPeak, broadPeak or bedGraph file, into a
    DataFrame of ranges that the functions taking ranges accept, such as
    [`extract_ranges()`][universalmotif.extract_ranges],
    [`mask_ranges()`][universalmotif.mask_ranges] and
    [`match_bkg()`][universalmotif.match_bkg].

    Positions are kept as they are in the file, counted from 0 with the end
    left out. Columns are named after the format: ``chrom``, ``start`` and
    ``end``, then for BED files as many of ``name``, ``score``, ``strand``,
    ``thick_start``, ``thick_end``, ``item_rgb``, ``block_count``,
    ``block_sizes`` and ``block_starts`` as there are, with any further
    columns named ``column_13`` and so on. narrowPeak and broadPeak files
    have ``name``, ``score``, ``strand``, ``signal``, ``neg_log10_pvalue``
    and ``neg_log10_qvalue``, and narrowPeak files the summit as ``peak``,
    counted from ``start``; the -1 these formats write for a value not given
    is read as null. bedGraph files have ``value``.

    Parameters
    ----------
    file
        The file, which may be gzipped. Blank lines, comments and ``track``
        and ``browser`` lines are skipped.
    format
        ``"bed"``, ``"narrowPeak"``, ``"broadPeak"`` or ``"bedGraph"``; by
        default from the file's extension (``.narrowPeak``, ``.broadPeak``,
        ``.bedGraph`` or ``.bg``, with or without ``.gz``), and otherwise
        BED.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import universalmotif as um
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "peaks.narrowPeak"
    >>> _ = path.write_text("chr1\t100\t300\tpeak1\t500\t.\t8.2\t12.5\t9.8\t95\n"
    ...                     "chr2\t40\t240\tpeak2\t310\t.\t4.1\t-1\t-1\t110\n")
    >>> peaks = um.read_bed(path)
    >>> peaks.select("chrom", "start", "end", "neg_log10_qvalue", "peak")
    shape: (2, 5)
    ┌───────┬───────┬─────┬──────────────────┬──────┐
    │ chrom ┆ start ┆ end ┆ neg_log10_qvalue ┆ peak │
    │ ---   ┆ ---   ┆ --- ┆ ---              ┆ ---  │
    │ str   ┆ i64   ┆ i64 ┆ f64              ┆ i64  │
    ╞═══════╪═══════╪═════╪══════════════════╪══════╡
    │ chr1  ┆ 100   ┆ 300 ┆ 9.8              ┆ 95   │
    │ chr2  ┆ 40    ┆ 240 ┆ null             ┆ 110  │
    └───────┴───────┴─────┴──────────────────┴──────┘
    """
    raw = Path(file).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return _bed_frame(raw.decode(), _bed_format(file, format), file)


def _bed_frame(text: str, fmt: str, file) -> pl.DataFrame:
    """The ranges in the text of a BED-like file of format ``fmt``."""
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.startswith(("#", "track", "browser"))]
    if not lines:
        raise ValueError(f"no ranges in {file}")
    width = max(ln.count("\t") for ln in lines) + 1
    names = _BED_FORMATS[fmt]
    if fmt == "bed":
        if width < 3:
            raise ValueError(f"{file} has fewer than 3 tab-separated columns")
        names = names[:width] + [f"column_{i + 1}" for i in range(len(names), width)]
    elif width != len(names):
        raise ValueError(f"a {fmt} file has {len(names)} columns, but {file} has {width}")
    df = pl.read_csv(io.StringIO("\n".join(lines)), separator="\t", has_header=False,
                     new_columns=names, infer_schema=False, quote_char=None)
    for n in ("start", "end"):
        try:
            df = df.with_columns(pl.col(n).cast(pl.Int64))
        except pl.exceptions.InvalidOperationError:
            raise ValueError(f"the {n} positions in {file} must be whole numbers") from None
    # Other columns are whole numbers, decimals or text, as they fit.
    for n in df.columns[3:]:
        if n in ("name", "strand", "item_rgb", "block_sizes", "block_starts"):
            continue
        for dtype in (pl.Int64, pl.Float64):
            try:
                df = df.with_columns(pl.col(n).replace(".", None).cast(dtype))
                break
            except pl.exceptions.InvalidOperationError:
                pass
    for n in ("signal", "neg_log10_pvalue", "neg_log10_qvalue", "value"):
        if n in df.columns and df[n].dtype == pl.Int64:
            df = df.with_columns(pl.col(n).cast(pl.Float64))
    if fmt in ("narrowpeak", "broadpeak"):
        df = df.with_columns(pl.when(pl.col(c) == -1).then(None).otherwise(pl.col(c)).alias(c)
                             for c in _PEAK_MISSING if c in df.columns)
    return df


def _chrom_lengths(genome) -> dict[str, int]:
    """Chromosome lengths from a Genome, packed sequences, a dict of
    sequences or a dict of lengths."""
    lengths = getattr(genome, "lengths", None)
    if isinstance(lengths, dict):
        return {str(k): int(v) for k, v in lengths.items()}
    if hasattr(genome, "items"):
        return {str(k): v if isinstance(v, int) else len(v) for k, v in genome.items()}
    raise ValueError("genome must be a Genome, sequences by name or lengths by name")


def resize_ranges(
    ranges,
    width: int,
    anchor: str = "center",
    genome=None,
    out_of_bounds: str = "drop",
) -> pl.DataFrame:
    """Give every range the same width, such as a window around each peak.

    Each range is resized around its ``anchor``: its centre, its start or
    end on its strand, or for narrowPeak files read with
    [`read_bed()`][universalmotif.read_bed] its summit. Windows that run off
    a chromosome are dropped or trimmed when ``genome`` is given. The other
    columns are kept, and a narrowPeak ``peak`` column is moved so that it
    still marks the summit, or set to null when the window leaves it out.

    Parameters
    ----------
    ranges
        Ranges as a DataFrame with a chromosome column, ``start`` and
        ``end``, such as [`read_bed()`][universalmotif.read_bed] returns, or
        any of the forms the functions taking ranges accept.
    width
        The width of every range.
    anchor
        ``"center"``, ``"start"``, ``"end"`` or ``"summit"``. A centre falls
        on the boundary between two bases, rounding up when the range and
        the width differ by an odd number. Start and end follow the strand,
        so that the start of a range on the ``-`` strand is its last base.
        Summits come from a ``peak`` column counted from the start, and
        ranges without one are centred.
    genome
        A [`Genome`][universalmotif.Genome], sequences by name or lengths by
        name, to keep windows within the chromosomes.
    out_of_bounds
        What to do with windows that run off a chromosome: ``"drop"`` them,
        ``"trim"`` them to it, or ``"keep"`` them as they are.

    Examples
    --------
    >>> import polars as pl
    >>> import universalmotif as um
    >>> peaks = pl.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [100, 20, 300],
    ...                       "end": [300, 41, 341], "name": ["a", "b", "c"]})
    >>> um.resize_ranges(peaks, 100, genome={"chr1": 1000, "chr2": 350})
    shape: (1, 4)
    ┌───────┬───────┬─────┬──────┐
    │ chrom ┆ start ┆ end ┆ name │
    │ ---   ┆ ---   ┆ --- ┆ ---  │
    │ str   ┆ i64   ┆ i64 ┆ str  │
    ╞═══════╪═══════╪═════╪══════╡
    │ chr1  ┆ 150   ┆ 250 ┆ a    │
    └───────┴───────┴─────┴──────┘
    >>> um.resize_ranges(peaks, 100, genome={"chr1": 1000, "chr2": 350}, out_of_bounds="trim")["start", "end"].rows()
    [(150, 250), (0, 81), (271, 350)]
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a whole number above 0")
    if anchor not in ("center", "start", "end", "summit"):
        raise ValueError("anchor must be 'center', 'start', 'end' or 'summit'")
    if out_of_bounds not in ("drop", "trim", "keep"):
        raise ValueError("out_of_bounds must be 'drop', 'trim' or 'keep'")
    df = _ranges_frame(ranges)
    if not isinstance(df, pl.DataFrame):
        df = _range_table(df, "ranges")
    chrom = next((c for c in _CHROM_COLUMNS if c in df.columns), None)
    if chrom is None or "start" not in df.columns or "end" not in df.columns:
        raise ValueError(f"ranges need a chromosome column ({', '.join(_CHROM_COLUMNS)}), start and end")
    if anchor == "summit" and "peak" not in df.columns:
        raise ValueError("anchor='summit' needs a peak column, as narrowPeak files have")
    start, end = pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64)
    centred = start + (end - start - width + 1) // 2
    minus = pl.col("strand").cast(pl.String) == "-" if "strand" in df.columns else pl.lit(False)
    if anchor == "center":
        new = centred
    elif anchor == "start":
        new = pl.when(minus).then(end - width).otherwise(start)
    elif anchor == "end":
        new = pl.when(minus).then(start).otherwise(end - width)
    else:
        summit = start + pl.col("peak").cast(pl.Int64)
        new = pl.when(pl.col("peak").is_null() | (pl.col("peak") < 0)).then(centred) \
            .otherwise(summit - width // 2)
    out = df.with_columns(new.alias("_new_start"))
    if "peak" in out.columns:
        shifted = start + pl.col("peak").cast(pl.Int64) - pl.col("_new_start")
        out = out.with_columns(pl.when((shifted >= 0) & (shifted < width)).then(shifted).otherwise(None)
                               .alias("peak"))
    out = out.with_columns(pl.col("_new_start").alias("start"),
                           (pl.col("_new_start") + width).alias("end")).drop("_new_start")
    if genome is not None and out_of_bounds != "keep":
        lengths = _chrom_lengths(genome)
        unknown = set(out[chrom].cast(pl.String)) - set(lengths)
        if unknown:
            raise ValueError(f"ranges on chromosomes not in genome: {', '.join(sorted(unknown)[:5])}")
        size = pl.col(chrom).cast(pl.String).replace_strict(lengths, return_dtype=pl.Int64)
        if out_of_bounds == "drop":
            out = out.filter(pl.col("start") >= 0, pl.col("end") <= size)
        else:
            trimmed = pl.max_horizontal(pl.col("start"), 0)
            if "peak" in out.columns:
                out = out.with_columns(pl.col("peak") - (trimmed - pl.col("start")))
            out = out.with_columns(trimmed.alias("start"), pl.min_horizontal(pl.col("end"), size).alias("end"))
    return out


def write_bed(
    ranges,
    file: str | os.PathLike,
    name: str | None = None,
    score: str | None = None,
    overwrite: bool = False,
) -> None:
    r"""Write ranges, or hits moved to the genome with
    [`genomic_hits()`][universalmotif.genomic_hits], to a BED file, for a
    genome browser or other tools. A path ending in ``.gz`` is gzipped.

    Ranges without names, scores or strands are written as BED3, and others
    as BED6, whose name, score and strand are ``.`` where missing and whose
    unstranded ranges have the strand ``.``.

    Parameters
    ----------
    ranges
        Ranges as a DataFrame with a chromosome column, ``start`` and
        ``end``, counted from 0 with the end left out, or any of the forms
        the functions taking ranges accept.
    file
        The file to write.
    name, score
        Columns to write as the name and score. By default ``name``, or
        ``motif`` for hits, and ``score``. Genome browsers expect whole
        numbers from 0 to 1000 as scores, which motif scores are not, so for
        hits give a column made for them, or ``score=""`` to leave scores
        out.
    overwrite
        Replace ``file`` if it exists.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> import polars as pl
    >>> import universalmotif as um
    >>> hits = pl.DataFrame({"chrom": ["chr1"], "start": [120], "end": [127],
    ...                      "strand": ["-"], "motif": ["AP-1"], "score": [9.4]})
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "hits.bed"
    >>> um.write_bed(hits, path, score="")
    >>> path.read_text()
    'chr1\t120\t127\tAP-1\t.\t-\n'
    """
    df = _ranges_frame(ranges)
    if not isinstance(df, pl.DataFrame):
        df = _range_table(df, "ranges")
    table = _range_table(df, "ranges")
    if name is None:
        name = next((c for c in ("name", "motif") if c in df.columns), "")
    if score is None:
        score = "score" if "score" in df.columns else ""
    for what, col in (("name", name), ("score", score)):
        if col and col not in df.columns:
            raise ValueError(f"ranges have no {col} column for the {what}")
    strand = table["strand"].replace({"*": "."})
    cols = [table["chrom"], table["start"].cast(pl.String), table["end"].cast(pl.String)]
    if name or score or (strand != ".").any():
        dot = pl.Series(["."] * table.height)
        cols += [df[name].cast(pl.String).fill_null(".") if name else dot,
                 df[score].cast(pl.String).fill_null(".") if score else dot,
                 strand]
    lines = ["\t".join(row) for row in zip(*(c.to_list() for c in cols))]
    path = Path(file)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; set overwrite=True")
    with (gzip.open(path, "wt", compresslevel=6) if path.suffix == ".gz" else open(path, "w")) as f:
        f.write("".join(ln + "\n" for ln in lines))


def _merged_ranges(ranges) -> pl.DataFrame:
    """Ranges with overlapping and touching ones merged, ignoring strand,
    sorted by chromosome and start."""
    df = _range_table(ranges, "ranges").filter(pl.col("end") > pl.col("start")).sort("chrom", "start")
    out: list[list] = []
    for chrom, start, end in df.select("chrom", "start", "end").iter_rows():
        if out and out[-1][0] == chrom and start <= out[-1][2]:
            out[-1][2] = max(out[-1][2], end)
        else:
            out.append([chrom, start, end])
    return pl.DataFrame(out, schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64}, orient="row")


def _scan_ranges(motifs, genome, ranges, nthreads, kwargs) -> pl.DataFrame:
    """scan_sequences(ranges=): scan the merged ranges of a genome and place
    the hits on it."""
    from ._analysis import scan_sequences
    from ._genome import Genome, extract_ranges
    from ._packed import PackedSequences

    if isinstance(genome, (str, os.PathLike)):
        genome = Genome(genome)
    elif not isinstance(genome, (Genome, PackedSequences, Mapping)):
        raise TypeError("scanning ranges needs a Genome, packed sequences, sequences by name or a FASTA path")
    seqs = extract_ranges(genome, _merged_ranges(ranges), nthreads=nthreads)
    return genomic_hits(scan_sequences(motifs, seqs, **kwargs))


def genomic_hits(hits: pl.DataFrame, ranges=None) -> pl.DataFrame:
    """Hits from scanning genomic ranges, moved to genomic coordinates.

    [`scan_sequences()`][universalmotif.scan_sequences] counts positions
    from the start of each sequence scanned. When those sequences are
    ranges taken from a genome, such as peaks read with
    [`extract_ranges()`][universalmotif.extract_ranges], this puts the hits
    back on the genome: each gets a ``chrom`` column, and ``start``, ``end``
    and ``strand`` become positions on the chromosome's forward strand,
    counted from 0 with the end left out. Hits in ranges that were reverse
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
    >>> peaks = um.extract_ranges(genome, [("chr1", 2, 15), ("chr1", 2, 15, "-")])
    >>> hits = um.scan_sequences(um.example_motifs()[1], peaks, pvalue=1e-3, rc=False)
    >>> hits.select("sequence", "start", "end", "strand", "match").rows()
    [('chr1:2-15', 2, 11, '+', 'CTGACTCAC'), ('chr1:2-15(-)', 2, 11, '+', 'GTGAGTCAG')]
    >>> um.genomic_hits(hits).select("chrom", "start", "end", "strand").rows()
    [('chr1', 4, 13, '+'), ('chr1', 4, 13, '-')]
    """
    for col in ("sequence", "start", "end", "strand"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if ranges is None:
        names = hits["sequence"].cast(pl.String)
        unique = names.unique(maintain_order=True).to_list()
        parsed = []
        for name in unique:
            m = _REGION.match(name)
            if m is None:
                raise ValueError(
                    f"the sequence name {name!r} is not chrom:start-end; give the ranges "
                    "with ranges"
                )
            parsed.append((m[1], int(m[2]), int(m[3]), "-" if m[4] else "+"))
        # Each hit's range, found through the index of its name.
        idx = names.replace_strict(
            unique, list(range(len(unique))), return_dtype=pl.UInt32
        ).cast(pl.UInt32)
        rows = parsed
    else:
        if "sequence_i" not in hits.columns:
            raise ValueError("hits must have a 'sequence_i' column to match them to ranges")
        table = _range_table(ranges, "ranges")
        rows = list(zip(table["chrom"], table["start"], table["end"], table["strand"]))
        idx = hits["sequence_i"].cast(pl.Int64)
        if idx.null_count() or (idx < 0).any() or (idx >= len(rows)).any():
            raise ValueError(f"hits refer to sequences beyond the {len(rows)} ranges given")
    start, end = hits["start"].cast(pl.Int64), hits["end"].cast(pl.Int64)
    if start.null_count() or end.null_count():
        raise ValueError("start and end cannot contain nulls")
    columns = list(zip(*rows)) if rows else [[], [], [], []]
    where = pl.DataFrame(
        {
            "c": pl.Series(columns[0], dtype=pl.String),
            "rs": pl.Series(columns[1], dtype=pl.Int64),
            "re": pl.Series(columns[2], dtype=pl.Int64),
            "rstrand": pl.Series(columns[3], dtype=pl.String),
        }
    ).select(pl.all().gather(idx))
    minus = where["rstrand"] == "-"
    st = hits["strand"].cast(pl.String)
    flipped = (
        pl.select(
            pl.when(st == "+").then(pl.lit("-")).when(st == "-").then(pl.lit("+")).otherwise(st)
        ).to_series()
    )
    out = hits.with_columns(
        pl.Series("start", pl.select(
            pl.when(minus).then(where["re"] - end).otherwise(where["rs"] + start)).to_series()),
        pl.Series("end", pl.select(
            pl.when(minus).then(where["re"] - start).otherwise(where["rs"] + end)).to_series()),
        pl.Series("strand", pl.select(pl.when(minus).then(flipped).otherwise(st)).to_series()),
    )
    chrom = where["c"]
    if "chrom" in out.columns:
        out = out.drop("chrom")
    out = out.with_columns(chrom.alias("chrom"))
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
    ``Strand``. Positions are 0-based and half-open in both, so they are
    kept as they are. Other columns are kept too. PyRanges objects can be
    given back to any function that takes ranges.

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
    df = df.rename(renames)
    first = [c for c in ("Chromosome", "Start", "End", "Strand") if c in df.columns]
    df = df.select(first + [c for c in df.columns if c not in first])
    return pr.PyRanges(df.to_pandas())


def to_polars_bio(df: pl.DataFrame) -> pl.DataFrame:
    """A table of ranges, such as hits or peaks, ready for polars-bio.

    polars-bio works on polars DataFrames with ``chrom``, ``start`` and
    ``end`` columns, and keeps their coordinate system as metadata. This
    renames the chromosome column to ``chrom`` if it has another name, such
    as the ``sequence`` of hits, moves it first, and marks the positions as
    0-based, as they are throughout this package, so that polars-bio's
    range operations treat them correctly. polars-bio frames can be given
    back to any function that takes ranges.

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
    df.config_meta.set(coordinate_system_zero_based=True)
    return df


__all__ = ["genomic_hits", "read_bed", "resize_ranges", "to_polars_bio", "to_pyranges", "write_bed"]
