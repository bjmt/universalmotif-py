"""Motif enrichment, co-occurrence and positional enrichment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import polars as pl

from . import _core
from ._analysis import _make_unique, _sequence_pairs, scan_sequences
from ._api import _check_motif
from ._core import Motif
from ._sequences import _seed


def _motif_list(motifs: Motif | Iterable[Motif]) -> list[Motif]:
    return [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]


_ENRICH_SCHEMA = {
    "motif": pl.String,
    "motif_i": pl.Int64,
    "consensus": pl.String,
    "target_seq_n": pl.Int64,
    "target_seq_hits": pl.Int64,
    "target_site_hits": pl.Int64,
    "bkg_seq_n": pl.Int64,
    "bkg_seq_hits": pl.Int64,
    "bkg_site_hits": pl.Int64,
    "enrichment": pl.Float64,
    "log2_enrichment": pl.Float64,
    "pvalue": pl.Float64,
    "qvalue": pl.Float64,
}


def enrich_motifs(
    motifs: Motif | Iterable[Motif],
    sequences: Sequence[str] | Mapping[str, str],
    bkg_sequences: Sequence[str] | Mapping[str, str] | None = None,
    pvalue: float = 1e-4,
    qvalue: float = 0.1,
    test: str = "seqs",
    rc: bool = True,
    shuffle_k: int = 2,
    seed: int | None = None,
    pseudocount: int = 1,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Test motifs for enrichment in target sequences.

    Target and background sequences of any alphabet are scanned with
    [`scan_sequences()`][universalmotif.scan_sequences], keeping hits with
    P-values below ``pvalue``, and each motif gets a one-sided Fisher's exact
    test with ``pseudocount`` added to every cell. With ``test="seqs"`` it
    tests the number of sequences with a hit, and with ``test="sites"`` the
    number of hits out of all the positions the motif could occupy.
    Q-values are Benjamini-Hochberg adjusted across all motifs.

    Without ``bkg_sequences``, the background is the targets shuffled with
    [`shuffle_sequences()`][universalmotif.shuffle_sequences] using the
    Euler method with ``k=shuffle_k`` and ``seed``.

    Returns a DataFrame of the motifs with q-values up to ``qvalue``, in
    order of P-value, with the columns ``motif``, ``motif_i``,
    ``consensus``, ``target_seq_n``, ``target_seq_hits``,
    ``target_site_hits``, ``bkg_seq_n``, ``bkg_seq_hits``, ``bkg_site_hits``,
    ``enrichment``, ``log2_enrichment``, ``pvalue`` and ``qvalue``.
    ``motif_i`` is 0-based, and ``enrichment`` is the ratio of hit rates,
    with a floor of half a hit on the background rate.
    """
    if test not in ("seqs", "sites"):
        raise ValueError("test must be 'seqs' or 'sites'")
    if not 0 < qvalue <= 1:
        raise ValueError("qvalue must be in (0, 1]")
    if shuffle_k < 1:
        raise ValueError("shuffle_k must be at least 1")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    ms = _motif_list(motifs)
    _, seqs = _sequence_pairs(sequences)
    bkg = None if bkg_sequences is None else _sequence_pairs(bkg_sequences)[1]
    cols = _core._enrich(ms, seqs, bkg, pvalue, test, rc, shuffle_k, _seed(seed), pseudocount, nthreads)
    names = _make_unique([m.name for m in ms])
    cols["motif"] = [names[i] for i in cols["motif_i"]]
    cols["consensus"] = [ms[i].consensus or "" for i in cols["motif_i"]]
    df = pl.DataFrame({k: cols[k] for k in _ENRICH_SCHEMA}, schema=_ENRICH_SCHEMA)
    return df.filter(pl.col("qvalue") <= qvalue).sort("pvalue", "motif_i", maintain_order=True)


def _coocc_hits(hits: pl.DataFrame, names: list[str], n_sequences: int, need_start: bool):
    if "sequence_i" not in hits.columns:
        raise ValueError("hits must have a sequence_i column")
    if "motif_i" in hits.columns and hits["motif_i"].dtype.is_integer():
        motif = hits["motif_i"].to_list()
    else:
        col = "motif_i" if "motif_i" in hits.columns else "motif"
        if col not in hits.columns:
            raise ValueError("hits must have a motif_i or motif column")
        index = {n: i for i, n in enumerate(names)}
        motif = [index.get(str(m), -1) for m in hits[col].to_list()]
    if any(m is None or m < 0 or m >= len(names) for m in motif):
        raise ValueError("some hits could not be matched to the motifs; check names and indices")
    sequence = hits["sequence_i"].to_list()
    if any(s is None or s < 0 or s >= n_sequences for s in sequence):
        raise ValueError("hits refer to sequence indices outside 0..n_sequences-1")
    start: list[int] = []
    if need_start:
        if "start" not in hits.columns:
            raise ValueError("hits must have a start column when max_distance is given")
        start = hits["start"].to_list()
        if any(s is None or s < 1 for s in start):
            raise ValueError("hit starts must be positive integers")
    return motif, sequence, start


_COOCC_SCHEMA = {
    "motif_a": pl.String,
    "motif_b": pl.String,
    "a_only": pl.Int64,
    "b_only": pl.Int64,
    "both": pl.Int64,
    "neither": pl.Int64,
    "odds_ratio": pl.Float64,
    "pvalue": pl.Float64,
    "qvalue": pl.Float64,
}


def motif_coocc(
    motifs: Motif | Iterable[Motif],
    sequences: Sequence[str] | Mapping[str, str] | None = None,
    hits: pl.DataFrame | None = None,
    n_sequences: int | None = None,
    pvalue: float = 1e-4,
    rc: bool = True,
    no_overlaps: bool = False,
    no_overlaps_by: str = "pvalue",
    max_distance: int | None = None,
    min_coocc: int = 1,
    pseudocount: int = 0,
    self_pairs: bool = False,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Find motif pairs that share sequences more often than expected.

    Every pair of motifs gets a one-sided Fisher's exact test on the 2x2
    table of sequences with hits from one motif, the other, both or neither,
    with ``pseudocount`` added to every cell. Pairs sharing fewer than
    ``min_coocc`` sequences are not tested. Q-values are Benjamini-Hochberg
    adjusted across the tested pairs.

    Give either ``sequences`` or a ``hits`` table plus ``n_sequences``.
    Sequences are scanned with
    [`scan_sequences()`][universalmotif.scan_sequences] using ``pvalue`` and
    ``rc``, and ``no_overlaps`` removes overlapping hits of the same motif. A
    hits table, which works for any alphabet, needs a 0-based
    ``sequence_i`` column and either a 0-based ``motif_i`` column or a
    ``motif`` column of names.

    With ``max_distance``, two descriptive columns that do not affect the
    test are added: ``both_clustered``, the number of shared sequences whose
    nearest pair of hits by start position is within
    ``max_distance``, and ``median_distance``, the median of those nearest
    distances. ``self_pairs`` also pairs each motif with itself.
    ``nthreads`` applies to the scan and to testing the pairs.

    Returns a DataFrame with the columns ``motif_a``, ``motif_b``,
    ``a_only``, ``b_only``, ``both``, ``neither``, ``odds_ratio``,
    ``pvalue`` and ``qvalue``, sorted by q-value then P-value, with untested
    pairs and their null P-values last.
    """
    if (sequences is None) == (hits is None):
        raise ValueError("give exactly one of sequences and hits")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be a non-negative integer")
    if min_coocc < 0 or pseudocount < 0:
        raise ValueError("min_coocc and pseudocount must be non-negative")
    ms = _motif_list(motifs)
    if len(ms) < 2 and not self_pairs:
        raise ValueError("need at least 2 motifs (or self_pairs=True)")
    names = [m.name or f"motif_{i + 1}" for i, m in enumerate(ms)]
    names = _make_unique(names)
    if hits is None:
        _, seqs = _sequence_pairs(sequences)
        n_sequences = len(seqs)
        h = scan_sequences(ms, seqs, pvalue=pvalue, rc=rc, nthreads=nthreads,
                           no_overlaps=no_overlaps, no_overlaps_by=no_overlaps_by)
        motif, sequence, start = h["motif_i"].to_list(), h["sequence_i"].to_list(), h["start"].to_list()
    else:
        if n_sequences is None or n_sequences < 1:
            raise ValueError("n_sequences must be a positive integer when hits are given")
        motif, sequence, start = _coocc_hits(hits, names, n_sequences, max_distance is not None)
    pairs = _core._coocc(
        motif, sequence, start, len(ms), n_sequences, max_distance, min_coocc, pseudocount, self_pairs,
        nthreads, names,
    )
    schema = dict(_COOCC_SCHEMA)
    if max_distance is not None:
        schema.update(both_clustered=pl.Int64, median_distance=pl.Float64)
    return pl.DataFrame(pairs).cast(schema)


_PEAKS_SCHEMA = {
    "motif": pl.String,
    "motif_i": pl.Int64,
    "mode": pl.String,
    "seq_length": pl.Int64,
    "nhits": pl.Int64,
    "best_window": pl.Int64,
    "best_center": pl.Float64,
    "hits_in": pl.Int64,
    "hits_out": pl.Int64,
    "expected_in": pl.Float64,
    "enrichment": pl.Float64,
    "log2_enrichment": pl.Float64,
    "pvalue": pl.Float64,
    "qvalue": pl.Float64,
    "centers": pl.List(pl.Int64),
}


def motif_peaks(
    hits: pl.DataFrame,
    seq_length: int,
    mode: str = "central",
    seq_strand: Mapping[object, str] | None = None,
    qvalue: float = 0.1,
    min_window: int = 10,
    max_window: int | None = None,
    window_step: int = 10,
    position_step: int = 5,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Test whether motif hits cluster along sequences of equal length.

    A CentriMo-style test (Bailey and Machanick 2012). Only each motif's
    highest-scoring hit per sequence is used, at its centre ``(start + end) /
    2``. If centres were uniform along the sequences, the number falling in a
    window of width ``w`` would be binomial with probability ``w /
    seq_length``. Windows from ``min_window`` to ``max_window`` in steps of
    ``window_step`` are tested, centred on the middle of the sequences with
    ``mode="central"`` or every ``position_step`` along them with
    ``mode="local"``, and ``max_window`` defaults to half the sequence length.
    The best window's P-value is Bonferroni-corrected for the number of windows
    tested, and q-values are Benjamini-Hochberg adjusted across motifs.

    Parameters
    ----------
    hits
        A hit table such as from
        [`scan_sequences()`][universalmotif.scan_sequences], with ``motif``,
        ``score``, ``start``, ``end`` or ``stop``, and ``sequence`` or
        ``sequence_i`` columns.
    seq_strand
        A mapping from the values in the sequence column to ``"+"`` or
        ``"-"``. Centres on ``-`` sequences are reflected to
        ``seq_length + 1 - centre`` so they read in the feature's
        orientation, which matters only for local mode.

    Returns
    -------
    polars.DataFrame
        A DataFrame of the motifs with q-values up to ``qvalue``, in order of
        P-value, with the columns ``motif``, ``motif_i``, ``mode``,
        ``seq_length``, ``nhits``, ``best_window``, ``best_center``,
        ``hits_in``, ``hits_out``, ``expected_in``, ``enrichment``,
        ``log2_enrichment``, ``pvalue``, ``qvalue`` and ``centers``.
        ``motif_i`` is the 0-based rank of the name among those in ``hits``,
        and ``centers`` holds the rounded hit centres used.
    """
    if mode not in ("central", "local"):
        raise ValueError("mode must be 'central' or 'local'")
    if not 0 < qvalue <= 1:
        raise ValueError("qvalue must be in (0, 1]")
    for col in ("motif", "score", "start"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    end_col = "end" if "end" in hits.columns else "stop" if "stop" in hits.columns else None
    if end_col is None:
        raise ValueError("hits must have an 'end' or 'stop' column")
    seq_col = "sequence" if "sequence" in hits.columns else "sequence_i" if "sequence_i" in hits.columns else None
    if seq_col is None:
        raise ValueError("hits must have a 'sequence' or 'sequence_i' column")
    motif_names = [str(m) for m in hits["motif"].to_list()]
    seq_values = hits[seq_col].to_list()
    centers = ((hits["start"].cast(pl.Float64) + hits[end_col].cast(pl.Float64)) / 2).to_list()
    if seq_strand is not None:
        missing = {s for s in seq_values if s not in seq_strand}
        if missing:
            raise ValueError(f"seq_strand is missing sequences: {sorted(map(str, missing))[:5]}")
        if any(seq_strand[s] not in ("+", "-") for s in set(seq_values)):
            raise ValueError("seq_strand values must be '+' or '-'")
        centers = [seq_length + 1 - c if seq_strand[s] == "-" else c for s, c in zip(seq_values, centers)]
    ids = sorted(set(motif_names))
    motif_index = {m: i for i, m in enumerate(ids)}
    seq_index: dict[object, int] = {}
    sequence = [seq_index.setdefault(s, len(seq_index)) for s in seq_values]
    cols = _core._peaks(
        [motif_index[m] for m in motif_names],
        sequence,
        centers,
        hits["score"].cast(pl.Float64).to_list(),
        seq_length,
        mode,
        min_window,
        max_window,
        window_step,
        position_step,
        nthreads,
    )
    n = len(cols["motif"])
    cols["motif_i"] = cols["motif"]
    cols["motif"] = [ids[i] for i in cols["motif_i"]]
    cols["mode"] = [mode] * n
    cols["seq_length"] = [seq_length] * n
    df = pl.DataFrame({k: cols[k] for k in _PEAKS_SCHEMA}, schema=_PEAKS_SCHEMA)
    return df.filter(pl.col("qvalue") <= qvalue).sort("pvalue", "motif_i", maintain_order=True)


__all__ = ["enrich_motifs", "motif_coocc", "motif_peaks"]
