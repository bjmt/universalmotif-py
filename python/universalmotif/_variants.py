"""Sequence variants: reading them from VCF files, their effects on the motif
sites they fall in, along one sequence or across a genome, and their
sequences one-hot encoded for a model."""

from __future__ import annotations

import gzip
import io
import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import polars as pl

from ._analysis import _make_unique
from ._api import _check_motif
from ._core import Motif

_COMPLEMENTS = {"DNA": ("ACGT", "TGCA"), "RNA": ("ACGU", "UGCA")}
# Allele spellings that mean no letters, as some tables write insertions and
# deletions.
_EMPTY_ALLELES = ("-", "")
# Letters of alternative alleles that are not sequence: symbolic alleles such
# as <DEL>, breakends, the * of overlapping deletions and unsplit lists.
_SYMBOLIC = "<>[]*,."
# Variant tables name their chromosome column in any of these ways.
_CHROM_NAMES = ("chrom", "#chrom", "chr", "chromosome", "seqnames", "sequence")


# Variant tables -------------------------------------------------------------


def read_vcf(file: str | os.PathLike) -> pl.DataFrame:
    r"""Read the variants of a VCF file into a DataFrame that
    [`variant_motif_effects()`][universalmotif.variant_motif_effects] and
    [`variant_batches()`][universalmotif.variant_batches] take.

    The columns are ``chrom``, ``position``, ``id``, ``ref``, ``alt``,
    ``qual``, ``filter`` and ``info``. Positions are counted from 0, so that
    they are one less than in the file, as in BED files and the rest of the
    package. A line listing several alternative alleles becomes one row per
    allele, and the ``.`` the format writes for a value not given is read as
    null, except for an alternative allele, which stays ``.``. Genotypes and
    any other columns after ``info`` are not read.

    Parameters
    ----------
    file
        The file, which may be gzipped or bgzipped.

    References
    ----------
    - Danecek P, Auton A, Abecasis G, Albers CA, Banks E, DePristo MA,
      Handsaker RE, Lunter G, Marth GT, Sherry ST, McVean G, Durbin R, 1000
      Genomes Project Analysis Group (2011). "The variant call format and
      VCFtools." *Bioinformatics*, **27**, 2156-2158.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import universalmotif as um
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "calls.vcf"
    >>> _ = path.write_text(
    ...     "##fileformat=VCFv4.2\n"
    ...     "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    ...     "chr1\t101\trs1\tA\tG,T\t50\tPASS\tAF=0.1,0.2\n"
    ...     "chr1\t205\t.\tCTT\tC\t.\tPASS\t.\n")
    >>> um.read_vcf(path).select("chrom", "position", "id", "ref", "alt")
    shape: (3, 5)
    ┌───────┬──────────┬──────┬─────┬─────┐
    │ chrom ┆ position ┆ id   ┆ ref ┆ alt │
    │ ---   ┆ ---      ┆ ---  ┆ --- ┆ --- │
    │ str   ┆ i64      ┆ str  ┆ str ┆ str │
    ╞═══════╪══════════╪══════╪═════╪═════╡
    │ chr1  ┆ 100      ┆ rs1  ┆ A   ┆ G   │
    │ chr1  ┆ 100      ┆ rs1  ┆ A   ┆ T   │
    │ chr1  ┆ 204      ┆ null ┆ CTT ┆ C   │
    └───────┴──────────┴──────┴─────┴─────┘
    """
    # Only the lines before the header are read here, and polars reads the
    # rest, decompressing it itself, so that a large file takes about as
    # much memory as its table.
    with open(file, "rb") as fh:
        gzipped = fh.read(2) == b"\x1f\x8b"
    with (gzip.open(file, "rb") if gzipped else open(file, "rb")) as fh:
        skip, _ = _vcf_header(fh, file)
    return _vcf_table(os.fspath(file), skip, file)


def _vcf_header(lines, file) -> tuple[int, bytes]:
    """The number of lines before the header of a VCF file, and the header."""
    skip = 0
    for line in lines:
        if line.startswith(b"##") or not line.strip():
            skip += 1
            continue
        header = line.rstrip(b"\r\n")
        if not header.startswith(b"#CHROM"):
            break
        width = header.count(b"\t") + 1
        if width < 8:
            raise ValueError(f"{file} has {width} columns, but a VCF file has at least 8")
        return skip, header
    raise ValueError(f"{file} has no #CHROM header line, so it is not a VCF file")


def _vcf_frame(raw: bytes, file) -> pl.DataFrame:
    """The variants in the contents of a VCF file."""
    skip, _ = _vcf_header(io.BytesIO(raw), file)
    return _vcf_table(io.BytesIO(raw), skip, file)


def _vcf_table(source, skip: int, file) -> pl.DataFrame:
    names = ["chrom", "position", "id", "ref", "alt", "qual", "filter", "info"]
    df = pl.read_csv(source, separator="\t", has_header=True, skip_rows=skip, columns=list(range(8)),
                     infer_schema=False, quote_char=None)
    df = df.rename(dict(zip(df.columns, names)))
    try:
        df = df.with_columns(pl.col("position").cast(pl.Int64) - 1)
    except pl.exceptions.InvalidOperationError:
        raise ValueError(f"the positions in {file} must be whole numbers") from None
    try:
        df = df.with_columns(pl.col("qual").replace(".", None).cast(pl.Float64))
    except pl.exceptions.InvalidOperationError:
        raise ValueError(f"the qualities in {file} must be numbers") from None
    df = df.with_columns(pl.col("id", "filter", "info").replace(".", None), pl.col("alt").str.split(","))
    try:
        return df.explode("alt", empty_as_null=True)
    except TypeError:
        return df.explode("alt")


def _column(df: pl.DataFrame, name: str) -> str | None:
    """The column called ``name``, matched without regard to case."""
    if name in df.columns:
        return name
    return next((c for c in df.columns if c.lower() == name), None)


def _as_frame(variants: Any) -> pl.DataFrame | None:
    """A table of variants as a polars DataFrame, or None for an array."""
    if isinstance(variants, pl.LazyFrame):
        return variants.collect()
    if isinstance(variants, pl.DataFrame):
        return variants
    if type(variants).__module__.startswith("pandas") and hasattr(variants, "columns"):
        return pl.from_pandas(variants)
    if isinstance(variants, Mapping):
        return pl.DataFrame(dict(variants))
    rows = isinstance(variants, (list, tuple)) and variants and isinstance(variants[0], (list, tuple))
    if rows and any(isinstance(v, str) for v in variants[0]):
        if len(variants[0]) not in (3, 4):
            raise ValueError("variants given as rows must be (position, ref, alt) or "
                             "(position, ref, alt, effect)")
        names = ["position", "ref", "alt", "effect"][:len(variants[0])]
        return pl.DataFrame([list(v) for v in variants], schema=names, orient="row")
    return None


def _from_array(variants: Any, sequence: str | None, alphabet: str, start: int) -> pl.DataFrame:
    """Every substitution in an array of effects of shape (letters, length),
    such as from mutagenesis_effects()."""
    from ._deep import _to_numpy

    m = _to_numpy(variants)
    n = len(alphabet)
    if m.ndim == 2 and m.shape[0] != n and m.shape[1] == n:
        m = m.T
    if m.ndim == 3:
        raise ValueError("the effects have several outputs; pick one, such as effects[0]")
    if m.ndim != 2 or m.shape[0] != n:
        raise ValueError(f"an array of effects must have shape ({n}, length), one row per letter of "
                         f"{alphabet}")
    if sequence is None:
        raise ValueError("an array of effects needs the sequence, to know the reference letters")
    if len(sequence) != m.shape[1]:
        raise ValueError(f"the sequence has {len(sequence)} letters for {m.shape[1]} positions of effects; "
                         "for effects of part of a sequence, give that part and its start")
    if not np.isfinite(m).all():
        raise ValueError("the effects have missing or infinite values")
    letters = np.array(list(alphabet))
    seq = np.array(list(sequence.upper() if alphabet.isupper() else sequence))
    ref = seq[None, :] == letters[:, None]
    let, pos = np.nonzero(~ref)
    order = np.lexsort((let, pos))
    let, pos = let[order], pos[order]
    return pl.DataFrame({
        "position": pl.Series(pos + start, dtype=pl.Int64),
        "ref": np.array(list(sequence))[pos].tolist(),
        "alt": letters[let].tolist(),
        "effect": m[let, pos].astype(float),
    })


def _variants(variants: Any, effects: Any, sequence: str | None, alphabet: str, start: int,
              need_effects: bool) -> pl.DataFrame:
    """Variants as a table with position, ref and alt columns, and an effect
    column when ``need_effects``, with any other columns kept after them."""
    df = _as_frame(variants)
    if df is None:
        if effects is not None:
            raise ValueError("effects cannot be given with an array of effects")
        return _from_array(variants, sequence, alphabet, start)
    rename = {}
    for name in ("position", "ref", "alt"):
        col = _column(df, name)
        if col is None:
            raise ValueError(f"variants must have a {name!r} column")
        rename[col] = name
    if effects is None or isinstance(effects, str):
        col = _column(df, effects or "effect")
        if col is not None:
            rename[col] = "effect"
        elif effects is not None:
            raise ValueError(f"variants has no {effects!r} column")
        elif need_effects:
            raise ValueError("variants needs an 'effect' column, or give the effects with effects")
    df = df.rename({k: v for k, v in rename.items() if k != v})
    if effects is not None and not isinstance(effects, str):
        from ._deep import _to_numpy

        e = _to_numpy(effects).ravel()
        if len(e) != df.height:
            raise ValueError(f"{len(e)} effects for {df.height} variants")
        df = df.with_columns(pl.Series("effect", e.astype(float)))
    if df.height == 0:
        raise ValueError("there are no variants")
    if not df["position"].dtype.is_integer():
        raise ValueError("positions must be whole numbers, counted from 0")
    df = df.with_columns(pl.col("position").cast(pl.Int64), pl.col("ref").cast(pl.String),
                         pl.col("alt").cast(pl.String))
    if df["position"].null_count() or df["ref"].null_count():
        raise ValueError("positions and reference letters cannot be missing")
    if "effect" in df.columns:
        if not df["effect"].dtype.is_numeric():
            raise ValueError("effects must be numbers")
        df = df.with_columns(pl.col("effect").cast(pl.Float64))
        if df["effect"].null_count() or not np.isfinite(df["effect"].to_numpy()).all():
            raise ValueError("the effects have missing or infinite values")
    lead = ["position", "ref", "alt"] + (["effect"] if "effect" in df.columns else [])
    df = df.select(lead + [c for c in df.columns if c not in lead])
    if sequence is not None:
        _check_refs(df, sequence, start)
    return df


def _ref_letters(refs: list[str]) -> list[str]:
    return ["" if r in _EMPTY_ALLELES else r for r in refs]


def _check_refs(df: pl.DataFrame, sequence: str, start: int) -> None:
    i = df["position"].to_numpy() - start
    refs = _ref_letters(df["ref"].to_list())
    width = np.array([len(r) for r in refs])
    out = np.flatnonzero((i < 0) | (i + np.maximum(width, 1) > len(sequence)))
    if out.size:
        p = int(df["position"][int(out[0])])
        raise ValueError(f"variant at position {p} lies outside the sequence, which runs from {start} to "
                         f"{start + len(sequence)}")
    bad = [k for k in range(df.height) if sequence[i[k]:i[k] + width[k]].upper() != refs[k].upper()]
    if bad:
        shown = "; ".join(f"position {int(i[k]) + start} has {sequence[i[k]:i[k] + width[k]]}, not {refs[k]}"
                          for k in bad[:3])
        more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
        raise ValueError(f"the reference letters of {len(bad)} variants do not match the sequence: "
                         f"{shown}{more}")


def _motif_list(motifs: Motif | Iterable[Motif]) -> tuple[list[Motif], list[str]]:
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("motifs is empty")
    return ms, _make_unique([m.name for m in ms])


def _every_substitution(sequence: str, alphabet: str, start: int) -> pl.DataFrame:
    seq = sequence.upper() if alphabet.isupper() else sequence
    rows = [(start + i, sequence[i], b) for i, a in enumerate(seq) for b in alphabet if b != a]
    if not rows:
        raise ValueError("the sequence is empty")
    return pl.DataFrame(rows, schema=["position", "ref", "alt"], orient="row")


def _chrom_column(df: pl.DataFrame) -> str:
    for name in _CHROM_NAMES:
        col = _column(df, name)
        if col is not None:
            return col
    raise ValueError("variants on a genome need a chromosome column, named chrom, chr, chromosome, "
                     "seqnames or sequence")


def _genome(genome: Any) -> Mapping[str, str]:
    """A genome given as a mapping, or opened from the path of its file."""
    if isinstance(genome, (str, os.PathLike)):
        from ._genome import Genome

        return Genome(genome)
    if not isinstance(genome, Mapping):
        raise TypeError("give the genome as a Genome, packed sequences, sequences by name or the path "
                        "of a FASTA or 2bit file")
    return genome


def _lengths(genome: Mapping[str, str], chroms: list[str]) -> dict[str, int]:
    """The lengths of the chromosomes named, which must all be in the
    genome."""
    lengths = getattr(genome, "lengths", None)
    out = {}
    for c in chroms:
        if c not in genome:
            alt = c.removeprefix("chr") if c.startswith("chr") else "chr" + c
            hint = f"; the genome has {alt!r}, so the two name chromosomes differently" if alt in genome else ""
            raise ValueError(f"the chromosome {c!r} is not in the genome{hint}")
        out[c] = int(lengths[c]) if isinstance(lengths, Mapping) else len(genome[c])
    return out


def _fetch(genome: Mapping[str, str], chroms: list[str], starts: np.ndarray, ends: np.ndarray,
           nthreads: int | None) -> list[str]:
    from ._genome import _fetch_ranges

    table = pl.DataFrame({"chrom": chroms, "start": starts.astype(np.int64), "end": ends.astype(np.int64),
                          "strand": ["+"] * len(chroms)},
                         schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String})
    return _fetch_ranges(genome, table, nthreads)


def _usable(ref: list[str], alt: list[str | None]) -> np.ndarray:
    """Which variants change the sequence by plain letters."""
    def ok(r, a):
        if a is None or a == ".":
            return False
        a = "" if a in _EMPTY_ALLELES else a
        if any(c in _SYMBOLIC for c in a) or not a.isascii() or not r.isascii():
            return False
        return a.upper() != r.upper()
    return np.fromiter((ok(r, a) for r, a in zip(ref, alt)), dtype=bool, count=len(ref))


def _trim(p: int, r: str, a: str) -> tuple[int, str, str]:
    """A variant with the letters its alleles share at either end taken off,
    the end first, so that a variant in a repeat is placed leftmost."""
    k = 0
    n = min(len(r), len(a))
    while k < n and r[-1 - k].upper() == a[-1 - k].upper():
        k += 1
    r, a = r[:len(r) - k], a[:len(a) - k]
    k = 0
    n = min(len(r), len(a))
    while k < n and r[k].upper() == a[k].upper():
        k += 1
    return p + k, r[k:], a[k:]


def _check_genome_refs(chroms: list[str], pos: np.ndarray, refs: list[str], contexts: list[str],
                       offsets: np.ndarray) -> None:
    bad = [k for k in range(len(refs))
           if contexts[k][offsets[k]:offsets[k] + len(refs[k])].upper() != refs[k].upper()]
    if bad:
        shown = "; ".join(f"{chroms[k]}:{int(pos[k])} has {contexts[k][offsets[k]:offsets[k] + len(refs[k])]}, "
                          f"not {refs[k]}" for k in bad[:3])
        more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
        raise ValueError(f"the reference letters of {len(bad)} variants do not match the genome: {shown}{more}; "
                         "check that the variants are of this genome's assembly, with positions counted "
                         "from 0")


# Motif effects ----------------------------------------------------------------


def variant_motif_effects(
    variants: Any,
    sequence: Any,
    motifs: Motif | Iterable[Motif],
    pvalue: float = 1e-4,
    start: int = 0,
    hits: pl.DataFrame | None = None,
    rc: bool | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
) -> pl.DataFrame:
    """How variants change the motif sites they fall in, along a sequence or
    across a genome.

    Every window of each motif's width that a variant changes is scored in
    the reference sequence and in the sequence with the variant, as
    [`scan_sequences()`][universalmotif.scan_sequences] scores it, on both
    strands of DNA and RNA with ``rc``, which is on by default for DNA and
    off for RNA, and on the one strand of a motif whose ``strand`` is
    ``"+"`` or ``"-"``. A window is kept when it reaches the ``pvalue``
    cutoff with either allele, so that the table holds sites a variant
    weakens or destroys and sites it strengthens or creates. With ``hits``,
    such as from ``scan_sequences()`` run on the same sequence or genome,
    only those sites are scored instead, whatever their scores. Each
    variant is applied on its own, without any others near it. Gapped motifs
    are scored without their gaps.

    Variants can be substitutions of any length, insertions and deletions.
    A window that starts before a variant is the same window in both
    sequences, and so is one that starts at the same letter of both
    alleles, but an insertion or deletion leaves windows starting in the
    letters only the longer allele has with no counterpart. These are sites
    made or lost with those letters, and their score in the other sequence
    is missing. Variants whose alternative allele is missing or symbolic,
    such as ``.`` or ``<DEL>`` in a VCF file, or is the same as the
    reference, are left out with a warning.

    Parameters
    ----------
    variants
        A table of variants with ``position``, ``ref`` and ``alt`` columns,
        as a polars or pandas DataFrame or a list of (position, ref, alt)
        rows, such as from [`read_vcf()`][universalmotif.read_vcf].
        Positions are counted from 0 and are those of the first letter of
        ``ref``, which can be empty, or ``-``, for an insertion before that
        position. Along one sequence it can also be an array of effects from
        [`mutagenesis_effects()`][universalmotif.mutagenesis_effects], or
        ``None`` for every single-letter substitution of the sequence. On a
        genome the table needs a chromosome column too, named ``chrom``,
        ``chr``, ``chromosome``, ``seqnames`` or ``sequence``.
    sequence
        The sequence the variants are in, or a genome, as a
        [`Genome`][universalmotif.Genome], packed sequences, sequences by
        name or the path of a FASTA or 2bit file as a ``pathlib.Path``,
        since a string is taken as the sequence. Only the letters around
        each variant are read from a genome.
    motifs
        A motif or a list of motifs, all of one alphabet.
    pvalue
        The P-value a site must reach with either allele to be kept.
    start
        Along one sequence, the position of its first letter, such as its
        position in the genome, from which the positions of the variants and
        sites are counted. The positions of ``hits`` are counted along the
        sequence from 0, as ``scan_sequences()`` gives them.
    hits
        Sites to score in place of those found at the ``pvalue`` cutoff,
        with ``motif``, ``start`` and ``strand`` columns, and on a genome the
        chromosome in a ``sequence`` column, as ``scan_sequences()`` gives
        them, or ``chrom``.
    rc
        Also score the reverse strand of DNA and RNA, by default for DNA
        only.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the scan has run for a second; by
        default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    polars.DataFrame
        One row per variant and motif site, with the columns ``variant``,
        the row of the variant in ``variants``, then on a genome ``chrom``,
        then ``position``, ``ref`` and ``alt`` as given, ``id`` when the
        variants have one, ``motif``, ``start``, ``end``, ``strand``,
        ``ref_score``, ``alt_score``, ``delta``, the change in score,
        ``ref_pvalue`` and ``alt_pvalue``. A site's ``start`` and ``end``
        are counted like the positions, with the end left out, and for a
        site found only with an insertion or deletion they are its place in
        the sequence that has it, which past the variant is shifted by the
        change in length. Its missing scores and P-values, and its
        ``delta``, are null.

    Examples
    --------
    >>> import polars as pl
    >>> import universalmotif as um
    >>> gata = um.example_motifs()[6]
    >>> seq = "TTACGCTTAGATAAGGCTTACGCATCGT"
    >>> variants = [(10, "A", "C"), (11, "T", "C"), (20, "C", "T")]
    >>> effects = um.variant_motif_effects(variants, seq, gata, pvalue=1e-3)
    >>> effects.select("position", "alt", "strand", "delta").rows()
    [(10, 'C', '+', -4.296), (11, 'C', '+', -7.561)]
    >>> every = um.variant_motif_effects(None, seq, gata, pvalue=1e-3)
    >>> every.height, every.filter(pl.col("delta") > 0).height
    (24, 3)

    A deletion of the site's GA, which weakens it, and an insertion that
    makes a site, then the same variants on a genome:

    >>> indels = [(8, "AGA", "A"), (20, "C", "CAGATAAG")]
    >>> table = um.variant_motif_effects(indels, seq, gata, pvalue=1e-3)
    >>> table.select("variant", "start", "ref_score", "alt_score").rows()
    [(0, 7, 7.98, -7.502), (1, 20, 1.673, 9.406)]
    >>> genome = {"chr2": seq}
    >>> on_genome = pl.DataFrame(indels, schema=["position", "ref", "alt"], orient="row")
    >>> on_genome = on_genome.with_columns(chrom=pl.lit("chr2"))
    >>> um.variant_motif_effects(on_genome, genome, gata, pvalue=1e-3).select("chrom", "start").rows()
    [('chr2', 7), ('chr2', 20)]
    """
    if not 0 < pvalue <= 1:
        raise ValueError("pvalue must be above 0 and at most 1")
    ms, names = _motif_list(motifs)
    alphabets = {m.alphabet for m in ms}
    if len(alphabets) > 1:
        raise ValueError(f"the motifs must share one alphabet, not {', '.join(sorted(alphabets))}")
    alphabet = "".join(ms[0].letters)
    on_genome = not isinstance(sequence, str)
    if on_genome:
        if isinstance(sequence, (list, tuple)):
            raise TypeError("give the sequence as a string, or a genome as a Genome, packed sequences or "
                            "sequences by name")
        genome = _genome(sequence)
        if variants is None or _as_frame(variants) is None:
            raise ValueError("on a genome, variants must be a table with chromosome, position, ref and alt "
                             "columns")
        df = _variants(variants, None, None, alphabet, 0, need_effects=False)
        chrom = pl.col(_chrom_column(df)).cast(pl.String)
        if df.select(chrom.null_count()).item():
            raise ValueError("chromosomes cannot be missing")
        offset = 0
    else:
        if variants is None:
            variants = _every_substitution(sequence, alphabet, start)
        df = _variants(variants, None, sequence, alphabet, start, need_effects=False)
        genome = {"": sequence}
        chrom = pl.lit("")
        offset = start
    # Only the columns used are kept, so that the variants' other columns
    # cannot clash with those added here.
    id_col = _column(df, "id")
    df = df.select(chrom.alias("_chrom"), "position", "ref", "alt",
                   *([pl.col(id_col).alias("_id")] if id_col else [])).with_row_index("_row")
    keep = _usable(df["ref"].to_list(), df["alt"].to_list())
    if not keep.all():
        warnings.warn(f"{int((~keep).sum())} of {df.height} variants were left out, with an alternative allele "
                      "that is missing, symbolic or the same as the reference", stacklevel=2)
    use = df.filter(pl.Series(keep))

    scored = [m.replace(name=n) for m, n in zip(ms, names)]
    scored = [_ungapped(m) for m in scored]
    widths = np.array([m.width for m in scored])
    schema = {"variant": pl.UInt32, **({"chrom": pl.String} if on_genome else {}),
              "position": pl.Int64, "ref": pl.String, "alt": pl.String,
              **({"id": pl.String} if id_col else {}),
              "motif": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String,
              "ref_score": pl.Float64, "alt_score": pl.Float64, "delta": pl.Float64,
              "ref_pvalue": pl.Float64, "alt_pvalue": pl.Float64}
    if use.height == 0:
        return pl.DataFrame(schema=schema)

    # Each variant's alleles, with the letters they share taken off, and
    # the letters around it, enough for the widest motif on either side.
    chroms = use["_chrom"].to_list()
    pos = use["position"].to_numpy() - offset
    refs = _ref_letters(use["ref"].to_list())
    alts = ["" if a in _EMPTY_ALLELES else a for a in use["alt"].to_list()]
    upper = ms[0].alphabet in ("DNA", "RNA", "AA")
    lengths = _lengths(genome, sorted(set(chroms)))
    chrom_len = np.array([lengths[c] for c in chroms])
    if on_genome:
        out = np.flatnonzero((pos < 0) | (pos + np.array([len(r) for r in refs]) > chrom_len))
        if out.size:
            k = int(out[0])
            raise ValueError(f"the variant at {chroms[k]}:{int(pos[k])} lies outside its chromosome, of "
                             f"length {int(chrom_len[k])}")
    trimmed = [_trim(int(p), r, a) for p, r, a in zip(pos, refs, alts)]
    tp = np.array([t[0] for t in trimmed], dtype=np.int64)
    tr = [t[1] for t in trimmed]
    ta = [t[2] for t in trimmed]
    lr = np.array([len(r) for r in tr], dtype=np.int64)
    la = np.array([len(a) for a in ta], dtype=np.int64)
    flank = int(widths.max()) - 1
    ctx_start = np.maximum(pos - flank, 0)
    ctx_end = np.minimum(pos + np.array([len(r) for r in refs]) + flank, chrom_len)
    contexts = _fetch(genome, chroms, ctx_start, ctx_end, nthreads)
    if on_genome:
        _check_genome_refs(chroms, pos, refs, contexts, pos - ctx_start)
    if upper:
        contexts = [c.upper() for c in contexts]
        ta = [a.upper() for a in ta]
    left = tp - ctx_start
    right = np.array([len(c) for c in contexts]) - left - lr
    alt_contexts = [c[:k] + a + c[k + n:] for c, k, a, n in zip(contexts, left.tolist(), ta, lr.tolist())]

    # The contexts are joined into one sequence per allele, set apart by a
    # letter no motif has, so that no window spans two of them.
    letters = set("".join("".join(m.letters) for m in ms))
    letters |= {c.lower() for c in letters} | {c.upper() for c in letters}
    sep = next(c for c in "N#-.X@" if c not in letters)
    ref_seq, ref_off = _joined(contexts, sep)
    alt_seq, alt_off = _joined(alt_contexts, sep)
    sides = {"ref": (ref_seq, ref_off, lr), "alt": (alt_seq, alt_off, la)}

    if hits is None:
        from ._analysis import scan_sequences

        # Warnings from the scan, such as of motifs that cannot reach the
        # cutoff, are passed on as if from this function.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = scan_sequences(scored, {"ref": ref_seq, "alt": alt_seq}, pvalue=pvalue, rc=rc,
                                   nthreads=nthreads, use_gaps=False, progress=progress)
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=2)
        parts = []
        for side, (_, off, n) in sides.items():
            h = found.filter(pl.col("sequence") == side)
            s = h["start"].to_numpy().astype(np.int64)
            m = h["motif_i"].to_numpy().astype(np.int64)
            v = np.searchsorted(off, s, side="right") - 1
            rel = s - off[v] - left[v]
            ok = (rel > -widths[m]) & (rel < n[v])
            parts.append(pl.DataFrame({"_v": v[ok], "_m": m[ok], "_rel": rel[ok],
                                       "strand": h["strand"].cast(pl.String).to_numpy()[ok]},
                                      schema={"_v": pl.Int64, "_m": pl.Int64, "_rel": pl.Int64,
                                              "strand": pl.String}))
        sites = pl.concat(parts).unique()
    else:
        sites = _sites_from_hits(hits, names, widths, chroms, tp, lr, offset, on_genome)

    frames = []
    for m_i, group in sites.group_by("_m"):
        m_i = int(m_i[0])
        motif, w = scored[m_i], int(widths[m_i])
        v = group["_v"].to_numpy()
        rel = group["_rel"].to_numpy()
        minus = (group["strand"] == "-").to_numpy()
        scores = {}
        for side, (seq, off, n) in sides.items():
            valid = (rel < n[v]) & (rel >= -left[v]) & (rel + w <= n[v] + right[v])
            scores[side] = _window_scores(motif, seq, off[v] + left[v] + rel, w, valid, minus)
        part = group.with_columns(pl.Series("ref_score", scores["ref"], nan_to_null=True),
                                  pl.Series("alt_score", scores["alt"], nan_to_null=True))
        if hits is None:
            from ._analysis import motif_pvalue

            threshold = float(motif_pvalue(motif, pvalue=pvalue))
            part = part.filter((pl.col("ref_score") >= threshold).fill_null(False)
                               | (pl.col("alt_score") >= threshold).fill_null(False))
        if part.height:
            frames.append(_with_pvalues(part, motif))
    if not frames:
        return pl.DataFrame(schema=schema)
    info = use.select(pl.col("_row").alias("variant"), *(["_chrom"] if on_genome else []), "position", "ref", "alt",
                      *([pl.col("_id").cast(pl.String).alias("id")] if id_col else []))
    if on_genome:
        info = info.rename({"_chrom": "chrom"})
    info = info.with_columns(pl.Series("_tp", tp + offset), pl.int_range(pl.len(), dtype=pl.Int64).alias("_v"))
    motif_info = pl.DataFrame({"_m": np.arange(len(names), dtype=np.int64), "motif": names, "_w": widths})
    out = (pl.concat(frames)
           .join(info, on="_v", how="left")
           .join(motif_info, on="_m", how="left")
           .with_columns((pl.col("_tp") + pl.col("_rel")).alias("start"))
           .with_columns((pl.col("start") + pl.col("_w")).alias("end"),
                         (pl.col("alt_score") - pl.col("ref_score")).round(3).alias("delta"))
           .sort("variant", "_m", "start", "strand"))
    return out.select(list(schema)).cast(schema)


def _ungapped(m: Motif) -> Motif:
    if not m.isgapped:
        return m
    from ._utils import ungap

    return ungap(m)


def _joined(seqs: list[str], sep: str) -> tuple[str, np.ndarray]:
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens[:-1] + 1)]).astype(np.int64)
    return sep.join(seqs), off


def _window_scores(motif: Motif, seq: str, at: np.ndarray, w: int, valid: np.ndarray,
                   minus: np.ndarray) -> np.ndarray:
    """The scores of the windows of width ``w`` starting at ``at``, read on
    the minus strand where ``minus``, as NaN where not ``valid`` or holding
    a letter the motif lacks."""
    from ._utils import score_match

    out = np.full(len(at), np.nan)
    if not valid.any():
        return out
    codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = np.where(valid, at, 0)[:, None] + np.arange(w)[None, :]
    words = codes[np.clip(idx, 0, len(codes) - 1)]
    if minus.any():
        a, b = _COMPLEMENTS[motif.alphabet]
        table = np.arange(256, dtype=np.uint8)
        table[np.frombuffer(a.encode(), dtype=np.uint8)] = np.frombuffer(b.encode(), dtype=np.uint8)
        words[minus] = table[words[minus][:, ::-1]]
    ok = np.zeros(256, dtype=bool)
    ok[np.frombuffer("".join(motif.letters).encode(), dtype=np.uint8)] = True
    valid = valid & ok[words].all(axis=1)
    if valid.any():
        w_ok = np.ascontiguousarray(words[valid]).view(f"S{w}").ravel().astype(str).tolist()
        out[valid] = np.asarray(score_match(motif, w_ok), dtype=float)
    return out


def _with_pvalues(part: pl.DataFrame, motif: Motif) -> pl.DataFrame:
    from ._analysis import motif_pvalue

    out = {}
    for side in ("ref", "alt"):
        s = part[f"{side}_score"].to_numpy()
        p = np.full(len(s), np.nan)
        ok = ~np.isnan(s)
        if ok.any():
            p[ok] = np.asarray(motif_pvalue(motif, score=s[ok].tolist()), dtype=float)
        out[f"{side}_pvalue"] = p
    return part.with_columns(pl.Series("ref_pvalue", out["ref_pvalue"], nan_to_null=True),
                             pl.Series("alt_pvalue", out["alt_pvalue"], nan_to_null=True))


def _sites_from_hits(hits: pl.DataFrame, names: list[str], widths: np.ndarray, chroms: list[str],
                     tp: np.ndarray, lr: np.ndarray, offset: int, on_genome: bool) -> pl.DataFrame:
    """The sites of ``hits`` that overlap the variants, as the variant, the
    motif and the site's start from the variant's first changed letter."""
    for col in ("motif", "start"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if on_genome:
        col = next((c for c in ("sequence", "chrom", "chr", "seqnames", "chromosome") if c in hits.columns),
                   None)
        if col is None:
            raise ValueError("hits on a genome need the chromosome in a 'sequence' or 'chrom' column")
        hit_chrom = hits[col].cast(pl.String).to_list()
    else:
        hit_chrom = [""] * hits.height
    index = {n: i for i, n in enumerate(names)}
    m = np.array([index.get(n, -1) for n in hits["motif"].cast(pl.String).to_list()], dtype=np.int64)
    strand = (hits["strand"].cast(pl.String) if "strand" in hits.columns
              else pl.Series(["+"] * hits.height)).to_numpy()
    s = hits["start"].cast(pl.Int64).to_numpy()
    codes = {c: i for i, c in enumerate(sorted(set(chroms)))}
    hc = np.array([codes.get(c, -1) for c in hit_chrom], dtype=np.int64)
    known = (m >= 0) & (hc >= 0)
    m, s, hc, strand = m[known], s[known], hc[known], strand[known]
    # Variants sorted by chromosome and position, so that those a hit can
    # overlap are a run found by binary search.
    big = np.int64(1) << 40
    vkey = np.array([codes[c] for c in chroms], dtype=np.int64) * big + tp
    order = np.argsort(vkey, kind="stable")
    sk = vkey[order]
    w = widths[m]
    lo = np.searchsorted(sk, hc * big + s - int(lr.max()) + 1, side="left")
    hi = np.searchsorted(sk, hc * big + s + w - 1, side="right")
    n = np.maximum(hi - lo, 0)
    h = np.repeat(np.arange(len(s)), n)
    v = order[np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
    rel = s[h] - tp[v]
    ok = (rel > -w[h]) & (rel < lr[v])
    return pl.DataFrame({"_v": v[ok], "_m": m[h][ok], "_rel": rel[ok], "strand": strand[h][ok]},
                        schema={"_v": pl.Int64, "_m": pl.Int64, "_rel": pl.Int64,
                                "strand": pl.String}).unique()


# Sequences for models -----------------------------------------------------------


def variant_batches(
    genome: Any,
    variants: Any,
    width: int,
    batch_size: int = 256,
    alphabet: str = "ACGT",
    dtype: Any = np.float32,
    nthreads: int | None = None,
):
    """The reference and alternative sequences around each variant, one-hot
    encoded in batches for a model.

    Each variant gets a window of ``width`` letters of ``genome``, a
    [`Genome`][universalmotif.Genome], packed sequences, sequences by name
    or the path of a FASTA or 2bit file, with the variant's first reference
    letter at ``width // 2``, the middle or the letter after it. The
    alternative sequence has the same letters before that, then the
    alternative allele and the letters that follow the reference allele, so
    that an insertion pushes letters off the right end and a deletion pulls
    more in. Windows running off the end of a chromosome are filled out
    with columns of zeros, as for N.

    ``variants`` is a table such as from
    [`read_vcf()`][universalmotif.read_vcf], with a chromosome column named
    ``chrom``, ``chr``, ``chromosome``, ``seqnames`` or ``sequence``, and
    ``position``, ``ref`` and ``alt`` columns, with positions counted from 0.
    Every variant is checked before any batch is made: its reference letters
    must match the genome, and its alternative allele must be letters, so
    that symbolic alleles such as ``<DEL>`` must be filtered out first.

    Returns an iterable of pairs of arrays ``(ref, alt)``, each of shape
    (variants, letters, width), up to ``batch_size`` variants each and in
    the order of the table, with a length so that progress can be shown. A
    model's predictions for the two give each variant's effect, such as
    ``model(alt) - model(ref)``, for
    [`plot_variant_effects()`][universalmotif.plot_variant_effects]. For a
    model's average over both strands, ``x[:, ::-1, ::-1]`` is the reverse
    complement of a batch ``x`` with the letters in ACGT order.

    Examples
    --------
    >>> import polars as pl
    >>> import universalmotif as um
    >>> genome = {"chr1": "AACCGGTTACGTAACCGGTT"}
    >>> variants = pl.DataFrame({"chrom": "chr1", "position": [5, 9],
    ...                          "ref": ["G", "CG"], "alt": ["A", "C"]})
    >>> ref, alt = next(iter(um.variant_batches(genome, variants, width=8)))
    >>> um.from_one_hot(ref), um.from_one_hot(alt)
    (['ACCGGTTA', 'GTTACGTA'], ['ACCGATTA', 'GTTACTAA'])
    """
    width = int(width)
    if width < 1:
        raise ValueError("width must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    genome = _genome(genome)
    df = _variants(variants, None, None, alphabet, 0, need_effects=False)
    df = df.with_columns(pl.col(_chrom_column(df)).cast(pl.String).alias("_chrom"))
    refs = _ref_letters(df["ref"].to_list())
    alts = df["alt"].to_list()
    ok = _usable(refs, alts) | np.array([a is not None and r.upper() == a.upper() for r, a in zip(refs, alts)])
    if not ok.all():
        k = int(np.flatnonzero(~ok)[0])
        raise ValueError(f"{int((~ok).sum())} variants have an alternative allele that is not letters, such as "
                         f"{alts[k]!r} at row {k}; filter them out first")
    chroms = df["_chrom"].to_list()
    lengths = _lengths(genome, sorted(set(chroms)))
    pos = df["position"].to_numpy()
    lr = np.array([len(r) for r in refs], dtype=np.int64)
    n = np.array([lengths[c] for c in chroms])
    out = np.flatnonzero((pos < 0) | (pos + lr > n))
    if out.size:
        k = int(out[0])
        raise ValueError(f"the variant at {chroms[k]}:{int(pos[k])} lies outside its chromosome, of length "
                         f"{int(n[k])}")
    # Only the reference letters are checked here, which reads them alone.
    seen = _fetch(genome, chroms, pos, pos + lr, nthreads)
    _check_genome_refs(chroms, pos, refs, seen, np.zeros(len(refs), dtype=np.int64))
    alts = ["" if a in _EMPTY_ALLELES else a for a in alts]
    return _VariantBatches(genome, chroms, pos, lr, alts, lengths, width, int(batch_size), alphabet, dtype,
                           nthreads)


class _VariantBatches:
    """Pairs of reference and alternative batches, read from a genome as
    they are needed."""

    def __init__(self, genome, chroms, pos, lr, alts, lengths, width, batch_size, alphabet, dtype, nthreads):
        self._genome, self._chroms, self._pos, self._lr, self._alts = genome, chroms, pos, lr, alts
        self._lengths, self._width, self._batch_size = lengths, width, batch_size
        self._alphabet, self._dtype, self._nthreads = alphabet, dtype, nthreads

    def __len__(self) -> int:
        return (len(self._pos) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        from ._deep import one_hot

        w, half = self._width, self._width // 2
        for i in range(0, len(self._pos), self._batch_size):
            rows = slice(i, i + self._batch_size)
            chroms = self._chroms[rows]
            pos, lr = self._pos[rows], self._lr[rows]
            n = np.array([self._lengths[c] for c in chroms])
            # Enough letters for the reference window and for the letters
            # a deletion pulls in, padded where they run off the chromosome.
            a, b = pos - half, pos + lr + (w - half)
            lo, hi = np.maximum(a, 0), np.minimum(b, n)
            got = _fetch(self._genome, chroms, lo, hi, self._nthreads)
            ctx = ["N" * int(x) + s + "N" * int(y) for s, x, y in zip(got, lo - a, b - hi)]
            ref = [c[:w] for c in ctx]
            alt = [(c[:half] + al + c[half + k:])[:w] for c, al, k in zip(ctx, self._alts[rows], lr.tolist())]
            yield (one_hot(ref, alphabet=self._alphabet, dtype=self._dtype),
                   one_hot(alt, alphabet=self._alphabet, dtype=self._dtype))
