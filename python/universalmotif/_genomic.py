"""Motif proximity to genomic anchors, and composition-matched backgrounds."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique, _sequence_pairs, scan_sequences
from ._genome import Genome, signal_stats
from ._packed import PackedSequences
from ._core import Motif
from ._sequences import _seed

_CHROM_COLUMNS = ("chrom", "seqnames", "sequence", "chr", "chromosome")


_PYRANGES_COLUMNS = {"Chromosome": "chrom", "Start": "start", "End": "end", "Strand": "strand"}


def _one_based_polars_bio(x) -> bool:
    """Whether a polars frame is marked as 1-based by polars-bio, which keeps
    the coordinate system of each frame as metadata. Frames without that
    metadata are taken to be 0-based."""
    pb = sys.modules.get("polars_bio")
    if pb is None:
        return False
    try:
        return pb.get_metadata(x).get("coordinate_system_zero_based") is False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _ranges_frame(x):
    """Ranges from other libraries as a polars DataFrame with 0-based,
    half-open positions, or ``x`` unchanged if it is not a data frame.

    PyRanges objects, pandas DataFrames and polars frames, including
    LazyFrames, are taken to be 0-based and half-open already, as PyRanges
    and BED are. polars-bio frames marked as 1-based, as polars-bio makes
    them by default, are moved to 0-based."""
    if type(x).__name__ == "PyRanges" and not hasattr(x, "columns") and hasattr(x, "df"):
        x = x.df  # PyRanges before version 1
    one_based = False
    if isinstance(x, pl.LazyFrame):
        one_based = _one_based_polars_bio(x)
        x = x.collect()
    elif isinstance(x, pl.DataFrame):
        one_based = _one_based_polars_bio(x)
    else:
        pd = sys.modules.get("pandas")
        if pd is None or not isinstance(x, pd.DataFrame):
            return x
        x = pl.from_pandas(pd.DataFrame(x).reset_index(drop=True))
        x = x.rename({k: v for k, v in _PYRANGES_COLUMNS.items() if k in x.columns and v not in x.columns})
    if one_based and "start" in x.columns:
        x = x.with_columns(pl.col("start") - 1)
    return x


def _range_table(x, what: str) -> pl.DataFrame:
    """Ranges as a DataFrame with chrom, start, end and strand columns, from
    either a DataFrame or (chrom, start, end[, strand]) tuples. A DataFrame
    needs a chromosome column named chrom, seqnames, sequence, chr or
    chromosome, a start column, and an end or width column, with 0-based,
    half-open positions. PyRanges and pandas DataFrames, and polars-bio
    frames, are converted first."""
    x = _ranges_frame(x)
    if isinstance(x, pl.DataFrame):
        chrom = next((c for c in _CHROM_COLUMNS if c in x.columns), None)
        if chrom is None or "start" not in x.columns:
            raise ValueError(f"{what} needs a chromosome column ({', '.join(_CHROM_COLUMNS)}) and start")
        if "end" in x.columns:
            end = x["end"]
        elif "stop" in x.columns:
            end = x["stop"]
        elif "width" in x.columns:
            end = x["start"] + x["width"]
        else:
            raise ValueError(f"{what} needs an end, stop or width column")
        strand = x["strand"].cast(pl.String) if "strand" in x.columns else pl.Series(["*"] * x.height)
        df = pl.DataFrame({
            "chrom": x[chrom].cast(pl.String),
            "start": x["start"].cast(pl.Int64),
            "end": end.cast(pl.Int64),
            "strand": strand,
        })
    else:
        rows = [tuple(r) for r in x]
        if any(len(r) not in (3, 4) for r in rows):
            raise ValueError(f"{what} must be (chrom, start, end) or (chrom, start, end, strand) tuples")
        df = pl.DataFrame(
            {
                "chrom": [str(r[0]) for r in rows],
                "start": [int(r[1]) for r in rows],
                "end": [int(r[2]) for r in rows],
                "strand": [r[3] if len(r) == 4 else "*" for r in rows],
            },
            schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String},
        )
    df = df.with_columns(pl.col("strand").fill_null("*").replace({"+-": "*", "": "*", ".": "*"}))
    if df["start"].null_count() or df["end"].null_count():
        raise ValueError(f"{what} has missing coordinates")
    return df


def _cols(df: pl.DataFrame, index: dict[str, int]):
    return ([index[c] for c in df["chrom"]], df["start"].to_list(), df["end"].to_list(),
            df["strand"].to_list())


_PROXIMITY_SCHEMA = {
    "motif": pl.String,
    "motif_i": pl.Int64,
    "method": pl.String,
    "count": pl.String,
    "orientation": pl.String,
    "nhits": pl.Int64,
    "best_dist": pl.Int64,
    "hits_near": pl.Int64,
    "hits_far": pl.Int64,
    "frac_near": pl.Float64,
    "expected_near": pl.Float64,
    "enrichment": pl.Float64,
    "log2_enrichment": pl.Float64,
    "pvalue": pl.Float64,
    "qvalue": pl.Float64,
    "ks_d": pl.Float64,
    "dist_to_nearest": pl.List(pl.Float64),
}


def motif_proximity(
    hits: pl.DataFrame | None = None,
    anchors=None,
    motifs: Motif | Iterable[Motif] | None = None,
    sequences: Mapping[str, str] | None = None,
    universe=None,
    method: str = "binomial",
    count: str = "hits",
    orientation: str = "undirected",
    scan_pvalue: float = 1e-4,
    rc: bool = True,
    dist_thresholds: Sequence[int] | None = None,
    n_perm: int = 1000,
    perm_per_chrom: bool = True,
    ignore_strand: bool = True,
    qvalue: float = 0.1,
    seed: int | None = None,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Test whether motif hits sit closer to genomic anchors than chance.

    The genome-wide counterpart of [`motif_peaks()`][universalmotif.motif_peaks]. Each hit counts at
    its middle position. For each distance ``d`` in ``dist_thresholds``, the
    near zone is every scanned position within ``d`` of an anchor; the
    default distances are 10, 25, 50, 100 and so on, up to half the smallest
    universe region. Three nulls are available:

    ``"binomial"``
        Hits uniform over the universe, so the number in the near zone is
        binomial; the best distance is Bonferroni-corrected for the
        distances tried. Fast, but blind to composition bias.
    ``"permutation"``
        Anchors are placed at random in the universe ``n_perm`` times,
        keeping each on its own chromosome with ``perm_per_chrom``. This
        absorbs composition and mappability bias, and is reproducible with
        ``seed``.
    ``"ks"``
        A one-sided Kolmogorov-Smirnov test of each hit's distance to the
        nearest anchor, for undirected zones and ``count="hits"`` only.

    Parameters
    ----------
    hits
        Hits from [`scan_sequences()`][universalmotif.scan_sequences], with
        ``motif``, ``score``, ``start``, ``end`` and ``strand`` columns and
        the chromosome names in ``sequence``. Give this, or give ``motifs``
        and ``sequences`` as a mapping from chromosome names to sequences to
        scan them with ``scan_pvalue`` and ``rc``; only chromosomes with
        anchors are scanned, and a [`Genome`][universalmotif.Genome] on disk
        is read one of them at a time.
    anchors
        The anchor coordinates, as a DataFrame with ``chrom``, ``start``,
        ``end`` or ``width``, and optionally ``strand`` columns, or as
        (chrom, start, end[, strand]) tuples. Use 1 bp anchors for point
        hypotheses: wide anchors widen every zone.
    universe
        The regions scanned, as a range table or as a mapping from
        chromosome names to lengths for whole chromosomes. Required with
        ``hits``; by
        default the whole of ``sequences``. Zones are clipped at chromosome
        ends when lengths are known. Only chromosomes carrying anchors are
        used.
    count
        ``"hits"`` counts hits in the zone, where overlapping hits of a
        motif count once with the best score kept, across strands unless
        ``ignore_strand`` is false. ``"anchors"`` counts anchors with a hit
        nearby; its binomial null is a Poisson approximation, so the
        permutation test is preferred.
    orientation
        ``"undirected"``, or ``"upstream"``/``"downstream"`` of stranded
        anchors.

    Returns
    -------
    polars.DataFrame
        A DataFrame of motifs with q-values up to ``qvalue``, in order of
        P-value, with the columns ``motif``, ``motif_i``, ``method``,
        ``count``, ``orientation``, ``nhits``, ``best_dist``, ``hits_near``,
        ``hits_far``, ``frac_near``, ``expected_near``, ``enrichment``,
        ``log2_enrichment``, ``pvalue``, ``qvalue``, ``ks_d`` and
        ``dist_to_nearest``. ``motif_i`` is the motif's 0-based index among
        ``motifs``, or in ``hits`` the ``motif_i`` from
        [`scan_sequences()`][universalmotif.scan_sequences] if it is there and
        otherwise the order in which motifs first appear, and
        ``dist_to_nearest`` holds the number of positions from each
        hit's middle to its nearest anchor, which is 0 inside it and
        negative upstream in the anchor's orientation.

    Examples
    --------
    Treating each example target as a chromosome with an anchor at its
    middle, the W-box hits sit closer to the anchors than chance:

    >>> import universalmotif as um
    >>> motifs = [um.example_motifs()[i] for i in (3, 5)]
    >>> targets = um.example_sequences("targets")
    >>> anchors = [(name, 499, 500) for name in targets]
    >>> res = um.motif_proximity(anchors=anchors, motifs=motifs, sequences=targets, nthreads=1)
    >>> res.select("motif", "nhits", "best_dist", "hits_near").row(0)
    ('W-box', 30, 250, 27)
    >>> f"{res['pvalue'][0]:.1e}"
    '2.2e-05'
    """
    if method not in ("binomial", "permutation", "ks"):
        raise ValueError("method must be 'binomial', 'permutation' or 'ks'")
    if count not in ("hits", "anchors"):
        raise ValueError("count must be 'hits' or 'anchors'")
    if orientation not in ("undirected", "upstream", "downstream"):
        raise ValueError("orientation must be 'undirected', 'upstream' or 'downstream'")
    if not 0 < qvalue <= 1:
        raise ValueError("qvalue must be in (0, 1]")
    if anchors is None:
        raise ValueError("anchors are required")
    anc = _range_table(anchors, "anchors")
    have_scan = motifs is not None or sequences is not None
    if (hits is not None) == have_scan:
        raise ValueError("give either hits, or motifs and sequences")
    lengths: dict[str, int] = {}
    if hits is None:
        if motifs is None or sequences is None:
            raise ValueError("scanning needs both motifs and sequences")
        if not isinstance(sequences, Mapping):
            raise ValueError("sequences must map chromosome names to sequences")
        anchored = set(anc["chrom"])
        if isinstance(sequences, PackedSequences):
            # Scanned without decoding; hits away from the anchors' chromosomes
            # are dropped afterwards.
            if not anchored & set(sequences):
                raise ValueError("none of the anchor chromosomes are in sequences")
            hits = scan_sequences(motifs, sequences, pvalue=scan_pvalue, rc=rc, nthreads=nthreads)
            hits = hits.filter(pl.col("sequence").is_in(list(anchored)))
            lengths = sequences.lengths
        elif isinstance(sequences, Genome):
            # Read and scanned one anchored chromosome at a time.
            chroms = [c for c in sequences if c in anchored]
            if not chroms:
                raise ValueError("none of the anchor chromosomes are in sequences")
            hits = pl.concat([
                scan_sequences(motifs, {c: sequences[c]}, pvalue=scan_pvalue, rc=rc, nthreads=nthreads)
                for c in chroms
            ])
            lengths = sequences.lengths
        else:
            chroms = {k: v for k, v in sequences.items() if k in anchored}
            if not chroms:
                raise ValueError("none of the anchor chromosomes are in sequences")
            hits = scan_sequences(motifs, chroms, pvalue=scan_pvalue, rc=rc, nthreads=nthreads)
            lengths = {k: len(v) for k, v in sequences.items()}
    hits = _ranges_frame(hits)
    for col in ("motif", "score", "start"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    h = _range_table(hits, "hits")
    if universe is None:
        if not lengths:
            raise ValueError("give the universe (chromosome lengths or scanned regions) with hits")
        universe = lengths
    if isinstance(universe, Mapping):
        lengths = {str(k): int(v) for k, v in universe.items()}
        uni = pl.DataFrame(
            {"chrom": list(lengths), "start": [0] * len(lengths), "end": list(lengths.values()),
             "strand": ["*"] * len(lengths)},
            schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String},
        )
    else:
        uni = _range_table(universe, "universe")
        lengths = {}
    names = list(dict.fromkeys([*uni["chrom"], *anc["chrom"], *h["chrom"]]))
    index = {c: i for i, c in enumerate(names)}
    chrom_lengths = [lengths.get(c) for c in names]
    motif_names = [str(m) for m in hits["motif"].to_list()]
    # Motifs keep the numbers scan_sequences() gave them, or else are
    # numbered in order of first appearance.
    if "motif_i" in hits.columns:
        given = dict(zip(motif_names, hits["motif_i"].to_list()))
        ids = sorted(given, key=lambda m: given[m])
    else:
        ids = list(dict.fromkeys(motif_names))
    motif_index = {m: i for i, m in enumerate(ids)}
    cols = _core._proximity(
        _cols(h, index),
        [motif_index[m] for m in motif_names],
        hits["score"].cast(pl.Float64).to_list(),
        _cols(anc, index),
        _cols(uni, index),
        chrom_lengths,
        method,
        count,
        orientation,
        None if dist_thresholds is None else [int(d) for d in dist_thresholds],
        n_perm,
        perm_per_chrom,
        ignore_strand,
        _seed(seed),
        nthreads,
    )
    n = len(cols["motif"])
    cols["motif"] = [ids[i] for i in cols["motif"]]
    cols["motif_i"] = (
        [given[m] for m in cols["motif"]] if "motif_i" in hits.columns
        else [motif_index[m] for m in cols["motif"]]
    )
    cols["method"], cols["count"], cols["orientation"] = [method] * n, [count] * n, [orientation] * n
    df = pl.DataFrame({k: cols[k] for k in _PROXIMITY_SCHEMA}, schema=_PROXIMITY_SCHEMA)
    return df.filter(pl.col("qvalue") <= qvalue).sort("pvalue", "motif_i", maintain_order=True)


_NUCLEOTIDES = set("ACGTUNRYKMSWBDHV-acgtunrykmswbdhv")


def _covariate_table(x, n: int, what: str) -> dict[str, list[float]]:
    if isinstance(x, pl.DataFrame):
        table = {c: x[c].cast(pl.Float64).to_list() for c in x.columns}
    elif isinstance(x, Mapping):
        table = {str(k): [float(v) for v in vals] for k, vals in x.items()}
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            table = {"covariate1": arr.tolist()}
        elif arr.ndim == 2:
            table = {f"covariate{j + 1}": arr[:, j].tolist() for j in range(arr.shape[1])}
        else:
            raise ValueError(f"{what} must be a DataFrame, a mapping or a 1-D or 2-D array")
    if not table:
        raise ValueError(f"{what} must have at least one column")
    if any(len(v) != n for v in table.values()):
        raise ValueError(f"{what} must have one row per sequence ({n})")
    return table


def _per_covariate(x, names: list[str], what: str) -> list:
    if isinstance(x, Mapping):
        if set(x) != set(names):
            raise ValueError(f"{what} must name every covariate exactly")
        return [x[nm] for nm in names]
    return [x] * len(names)


def match_bkg(
    sequences: Sequence[str] | Mapping[str, str],
    universe: Sequence[str] | Mapping[str, str] | None = None,
    genome: Mapping[str, str] | None = None,
    exclude=None,
    n_candidates: int | None = None,
    chromosomes: Sequence[str] | None = None,
    n_per_target: int = 1,
    n_bins_gc: int = 20,
    n_bins_length: int = 10,
    covariates=None,
    universe_covariates=None,
    bigwig=None,
    target_ranges=None,
    n_bins_covariates: int | Mapping[str, int] = 10,
    bin_type_covariates: str | Mapping[str, str] = "quantile",
    unique: bool = True,
    seed: int | None = None,
    return_indices: bool = False,
):
    """Draw background sequences matched to targets on GC content and length.

    HOMER-style matching: the universe is binned by GC fraction into
    ``n_bins_gc`` equal-width bins, by length into ``n_bins_length``
    quantile bins and by any ``covariates``, and each target draws
    ``n_per_target`` sequences from its own bin. When the bin runs short,
    neighbouring bins are searched ring by ring, up to two rings or one per
    axis; after that
    the remaining draws repeat sequences from the bins searched, with a
    warning. With ``unique``, each universe sequence is otherwise drawn at
    most once.

    Parameters
    ----------
    universe
        Candidate background sequences, as a list or a mapping from names.
    genome
        Instead of ``universe``, a mapping from chromosome names to
        sequences, such as a [`Genome`][universalmotif.Genome] on disk or a
        genome read with [`read_fasta()`][universalmotif.read_fasta], to draw
        ``n_candidates`` random windows from; from a genome on disk, only the
        windows are read. The windows take their widths
        from the targets, lie on ``chromosomes``, which defaults to all of
        them, and avoid ``exclude``, a range table such as the target peaks.
        ``n_candidates`` defaults to ``max(10000, 10 * len(sequences))``.
    covariates, universe_covariates
        Extra values to match on, one row per target and per universe
        sequence: DataFrames or mappings with the same columns, or arrays.
        ``n_bins_covariates`` and ``bin_type_covariates``, which is
        ``"quantile"`` or ``"equalwidth"``, apply to all of them or are given
        per name.
    bigwig, target_ranges
        With ``genome``, match on bigWig signal as well, such as chromatin
        accessibility: ``bigwig`` is a path or a mapping from covariate names
        to paths, and ``target_ranges`` holds the coordinates of the targets
        in the same order, as a range table or (chrom, start, end) tuples.
        The mean signal over each target and each window, computed with
        [`signal_stats()`][universalmotif.signal_stats], becomes a covariate
        binned as above. Needs the ``bigwig`` extra.
    return_indices
        Return a DataFrame of the matches instead, with 0-based ``target_i``
        and ``universe_i``, the GC fraction and length of each, any
        covariates, and ``universe_name`` when the universe has names.

    Returns
    -------
    list[str] | dict[str, str]
        The matched sequences, ``n_per_target`` per target in target order,
        as a list or as a dict when the universe has names. In a dict,
        repeats get ``.1``, ``.2`` and so on, and genome windows are named
        ``chrom:start-end``.

    Examples
    --------
    >>> import universalmotif as um
    >>> targets = um.example_sequences("targets")
    >>> controls = um.example_sequences("controls")
    >>> bkg = um.match_bkg(targets, controls, seed=1)
    >>> len(bkg), list(bkg)[:2]
    (100, ['control_047', 'control_090'])

    With ``return_indices``, the matches can be checked:

    >>> idx = um.match_bkg(targets, controls, seed=1, return_indices=True)
    >>> idx.select("target_i", "target_gc", "universe_i", "universe_gc").row(0)
    (0, 0.339, 46, 0.344)
    """
    _, seqs = _sequence_pairs(sequences)
    if not seqs:
        raise ValueError("no target sequences given")
    if (universe is None) == (genome is None):
        raise ValueError("give exactly one of universe and genome")
    if covariates is not None and genome is not None:
        raise ValueError("covariates need an explicit universe with universe_covariates, not a genome; "
                         "use bigwig and target_ranges for signal over the genome")
    if (bigwig is None) != (target_ranges is None):
        raise ValueError("bigwig and target_ranges go together")
    if bigwig is not None and genome is None:
        raise ValueError("bigwig needs a genome to draw windows from; with a universe, give "
                         "covariates from signal_stats() instead")
    if any(set(s) - _NUCLEOTIDES for s in seqs):
        raise ValueError("match_bkg() only supports DNA and RNA sequences")
    names: list[str] | None
    if genome is not None:
        chroms = list(genome) if chromosomes is None else [str(c) for c in chromosomes]
        missing = [c for c in chroms if c not in genome]
        if missing:
            raise ValueError(f"chromosomes not in the genome: {', '.join(missing[:5])}")
        index = {c: i for i, c in enumerate(chroms)}
        excl = _range_table(exclude, "exclude").filter(pl.col("chrom").is_in(chroms)) \
            if exclude is not None else None
        exclude_cols = _cols(excl, index) if excl is not None else ([], [], [], [])
        if n_candidates is None:
            n_candidates = max(10000, 10 * len(seqs))
        if isinstance(genome, (Genome, PackedSequences)):
            # Read or decode just the windows, not whole chromosomes.
            chrom_lengths = genome.lengths
            lens = [chrom_lengths[c] for c in chroms]
        else:
            lens = [len(genome[c]) for c in chroms]
        windows = _core._genome_windows(lens, [len(s) for s in seqs], n_candidates, exclude_cols,
                                        _seed(seed))
        win_chroms = [chroms[c] for c, _, _ in windows]
        win_starts = [s for _, s, _ in windows]
        win_ends = [e for _, _, e in windows]
        if isinstance(genome, Genome):
            uni = genome._fetch(win_chroms, win_starts, win_ends)
        elif isinstance(genome, PackedSequences):
            uni = [genome.subseq(c, s, e) for c, s, e in zip(win_chroms, win_starts, win_ends)]
        else:
            uni = [genome[c][s:e] for c, s, e in zip(win_chroms, win_starts, win_ends)]
        names = [f"{c}:{s}-{e}" for c, s, e in zip(win_chroms, win_starts, win_ends)]
        seed = None if seed is None else seed + 1
        if bigwig is not None:
            tr = _range_table(target_ranges, "target_ranges")
            if tr.height != len(seqs):
                raise ValueError(f"target_ranges has {tr.height} ranges for {len(seqs)} sequences")
            covariates = signal_stats(bigwig, tr)
            universe_covariates = signal_stats(bigwig, list(zip(win_chroms, win_starts, win_ends)))
    else:
        names_u, uni = _sequence_pairs(universe)
        names = names_u if isinstance(universe, Mapping) else None
    if not uni:
        raise ValueError("the universe is empty")

    cov_specs = []
    cov_t: dict[str, list[float]] = {}
    cov_u: dict[str, list[float]] = {}
    if covariates is not None or universe_covariates is not None:
        if covariates is None or universe_covariates is None:
            raise ValueError("covariates and universe_covariates go together")
        cov_t = _covariate_table(covariates, len(seqs), "covariates")
        cov_u = _covariate_table(universe_covariates, len(uni), "universe_covariates")
        if list(cov_t) != list(cov_u):
            raise ValueError("covariates and universe_covariates must have the same columns in the same order")
        reserved = {"i", "gc", "length", "name"} & set(cov_t)
        if reserved:
            raise ValueError(f"covariate names {sorted(reserved)} are reserved; rename them")
        cn = list(cov_t)
        bins = _per_covariate(n_bins_covariates, cn, "n_bins_covariates")
        types = _per_covariate(bin_type_covariates, cn, "bin_type_covariates")
        cov_specs = [(nm, cov_t[nm], cov_u[nm], int(b), t) for nm, b, t in zip(cn, bins, types)]

    if unique and len(uni) < n_per_target * len(seqs):
        warnings.warn(
            f"the universe ({len(uni)}) is smaller than n_per_target * len(sequences) "
            f"({n_per_target * len(seqs)}); some sequences will be drawn more than once",
            stacklevel=2,
        )
    picked, fallbacks = _core._match_bkg(seqs, uni, cov_specs, n_per_target, n_bins_gc,
                                         n_bins_length, unique, _seed(seed))
    if fallbacks:
        warnings.warn(
            f"{fallbacks} target(s) ran out of matching sequences nearby and repeated some; "
            "consider fewer bins or a bigger universe",
            stacklevel=2,
        )
    if return_indices:
        target_i = [t for t in range(len(seqs)) for _ in range(n_per_target)]
        gc_t, gc_u = _core._gc_fraction(seqs), _core._gc_fraction(uni)
        data: dict[str, list] = {
            "target_i": target_i,
            "target_gc": [gc_t[t] for t in target_i],
            "target_length": [len(seqs[t]) for t in target_i],
            "universe_i": picked,
            "universe_gc": [gc_u[u] for u in picked],
            "universe_length": [len(uni[u]) for u in picked],
        }
        for nm in cov_t:
            data[f"target_{nm}"] = [cov_t[nm][t] for t in target_i]
            data[f"universe_{nm}"] = [cov_u[nm][u] for u in picked]
        if names is not None:
            data["universe_name"] = [names[u] for u in picked]
        return pl.DataFrame(data)
    if names is None:
        return [uni[u] for u in picked]
    return dict(zip(_make_unique([names[u] for u in picked]), (uni[u] for u in picked)))


__all__ = ["match_bkg", "motif_proximity"]
