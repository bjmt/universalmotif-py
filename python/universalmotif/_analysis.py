"""P-values, sequence scanning and hit deduplication."""

from __future__ import annotations

import numbers
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from . import _core
from ._api import _check_motif
from ._core import Motif
from ._packed import PackedSequences


def _is_number(x: object) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def motif_pvalue(
    motifs: Motif | Iterable[Motif],
    score: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    pvalue: float | Sequence[float] | Sequence[Sequence[float]] | None = None,
    bkg_probs: Sequence[float] | Mapping[str, float] | Sequence[object] | None = None,
    nthreads: int = 1,
):
    """P-values for motif scores, or score thresholds for P-values.

    Give either ``score`` or ``pvalue``. A motif's P-value for a score is the
    probability that a random site, drawn from the background, scores at
    least that much; it is computed exactly from the motif's score
    distribution. A score above the motif's highest possible score has a
    P-value of 0, with a warning. Going the other way gives the lowest score
    reached with at most that probability, or infinity if no score is that
    unlikely.

    Scores are computed on an integer grid, with PWM entries multiplied by
    1000 and truncated, so they are exact to 0.001.

    Parameters
    ----------
    motifs
        A motif or a list of motifs. They are converted to PWMs; a motif
        whose PWM would have infinite values gets its pseudocount reapplied
        first.
    score, pvalue
        For a single motif, a number or a sequence of numbers. For a list of
        motifs, a number used for every motif, one number per motif, or a
        list of sequences with either one per motif or a single one for all.
    bkg_probs
        Background frequencies, either one set for all motifs or a list with
        one per motif, where a set is a sequence in alphabet order or a
        mapping from letters. Each motif's own background by default.
    nthreads
        Threads to use; 0 means all available.

    Returns
    -------
    float | numpy.ndarray | list[numpy.ndarray]
        A float for one motif and one value, a numpy array when values pair up
        with motifs, or a list of arrays when each motif gets several values.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> wbox = motifs[5]
    >>> round(um.motif_pvalue(wbox, score=10), 6)
    0.000168
    >>> um.motif_pvalue(wbox, score=[5, 8, 10]).round(6).tolist()
    [0.002483, 0.000492, 0.000168]

    Score thresholds for a P-value of 1e-4, one per motif:

    >>> um.motif_pvalue(motifs[:3], pvalue=1e-4).round(3).tolist()
    [10.252, 10.419, 10.806]
    """
    if (score is None) == (pvalue is None):
        raise ValueError("give exactly one of score and pvalue")
    values = score if score is not None else pvalue
    single = isinstance(motifs, Motif)
    ms = [motifs] if single else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("no motifs given")
    n = len(ms)

    def run(queries):
        out = np.array(
            _core._motif_pvalue(ms, queries, bkg_probs, nthreads, score is None), dtype=float
        )
        if score is not None:
            # A P-value of 0 only comes from a score that no site can reach.
            above = sorted({ms[queries[k][0]].name for k in np.flatnonzero(out == 0)})
            if above:
                warnings.warn(
                    f"scores above the highest possible score get a P-value of 0 "
                    f"(motifs: {', '.join(above[:5])}{', ...' if len(above) > 5 else ''})",
                    stacklevel=3,
                )
        return out

    if _is_number(values):
        if single:
            return float(run([(0, float(values))])[0])
        return run([(i, float(values)) for i in range(n)])

    values = list(values)
    if values and not _is_number(values[0]):
        groups = [list(map(float, v)) for v in values]
        if len(groups) == 1:
            groups = groups * n
        elif len(groups) != n:
            raise ValueError("give one sequence of values for all motifs or one per motif")
        flat = run([(i, v) for i, g in enumerate(groups) for v in g])
        out, k = [], 0
        for g in groups:
            out.append(flat[k : k + len(g)])
            k += len(g)
        return out

    values = [float(v) for v in values]
    if single:
        return run([(0, v) for v in values])
    if len(values) == 1:
        values = values * n
    if len(values) != n:
        raise ValueError("give one value for all motifs, one per motif, or a list per motif")
    return run(list(enumerate(values)))


def _make_unique(names: list[str]) -> list[str]:
    """Make names unique: the first of each is kept as it is, and later
    duplicates get ".1", ".2" and so on."""
    seen: set[str] = set()
    out = []
    for name in names:
        new, k = name, 0
        while new in seen:
            k += 1
            new = f"{name}.{k}"
        seen.add(new)
        out.append(new)
    return out


def _sequence_pairs(sequences) -> tuple[list[str], list[str]]:
    if isinstance(sequences, str):
        return ["seq1"], [sequences]
    if isinstance(sequences, Mapping):
        return [str(k) for k in sequences], [str(v) for v in sequences.values()]
    seqs = [str(s) for s in sequences]
    return [f"seq{i + 1}" for i in range(len(seqs))], seqs


_HIT_SCHEMA = {
    "motif": pl.String,
    "motif_i": pl.UInt32,
    "sequence": pl.String,
    "sequence_i": pl.UInt32,
    "start": pl.Int64,
    "end": pl.Int64,
    "strand": pl.String,
    "score": pl.Float64,
    "score_pct": pl.Float64,
    "match": pl.String,
    "pvalue": pl.Float64,
}


def _warn_unreachable(ms: list[Motif], pvalue: float, stacklevel: int = 3) -> None:
    """Warn about motifs that cannot give a P-value as small as ``pvalue``,
    and so can have no hits, with the smallest P-value each can give."""
    best = _core._min_pvalues(ms)
    bad = [i for i, p in enumerate(best) if p > pvalue]
    if not bad:
        return
    listed = ", ".join(f"{ms[i].name} ({best[i]:.3g})" for i in bad[:5])
    more = f" and {len(bad) - 5} more" if len(bad) > 5 else ""
    warnings.warn(
        f"{len(bad)} motif{'s' if len(bad) > 1 else ''} cannot reach a P-value of {pvalue:g} and "
        f"will have no hits; the smallest P-value each can give is: {listed}{more}",
        stacklevel=stacklevel,
    )


def _sample_letters(seqs: list[str], n_seqs: int = 100, n_letters: int = 10_000) -> set[str]:
    """The letters, in upper case, of the start of the first few sequences."""
    return set("".join(s[:n_letters] for s in seqs[:n_seqs]).upper())


def _warn_alphabet_mismatch(ms: list[Motif], letters: set[str], stacklevel: int = 3) -> None:
    """Warn when DNA motifs meet sequences that look like RNA, or the other
    way round, judging the sequences by the letters given."""
    for alph, own, other in (("DNA", "T", "U"), ("RNA", "U", "T")):
        if any(m.alphabet == alph for m in ms) and other in letters and own not in letters:
            warnings.warn(
                f"{alph} motifs are scanned against sequences that contain {other} but not "
                f"{own}, so they can only match where {own} is not needed; convert the motifs "
                "with switch_alph() or the sequences first",
                stacklevel=stacklevel,
            )


def scan_sequences(
    motifs: Motif | Iterable[Motif],
    sequences: str | Sequence[str] | Mapping[str, str] | os.PathLike[str],
    pvalue: float = 1e-4,
    rc: bool = True,
    nthreads: int = 1,
    no_overlaps: bool = False,
    no_overlaps_by: str = "pvalue",
    no_overlaps_by_strand: bool = False,
    no_overlaps_by_motif: bool = False,
    use_gaps: bool = True,
    correct_gaps: bool = False,
) -> pl.DataFrame:
    """Scan sequences for motif hits.

    Motifs and sequences of any alphabet can be scanned. Each motif's score
    threshold comes from the ``pvalue`` cutoff, and every window scoring at
    least that much is reported with its P-value. Only DNA and RNA have a
    reverse strand, which is scanned with ``rc``.

    A motif too short or too degenerate to reach the ``pvalue`` cutoff can
    have no hits, and is named in a warning with the smallest P-value it can
    give. There are warnings too when DNA motifs meet sequences that look
    like RNA or the other way round, and when a string that names a file is
    given as the sequence.

    Parameters
    ----------
    sequences
        A sequence, a list of sequences, or a mapping from names to
        sequences such as the result of
        [`read_fasta()`][universalmotif.read_fasta]. DNA, RNA and protein
        letters are matched case-insensitively and custom alphabets exactly;
        anything else, such as ``N`` or ``X``, never forms part of a hit.
        [`PackedSequences`][universalmotif.PackedSequences] are scanned
        without decoding them, and a gzipped or plain FASTA file given as a
        path object such as a ``pathlib.Path`` is read and scanned a batch at
        a time, so a genome can be scanned in a few hundred MB. A
        [`Genome`][universalmotif.Genome] is scanned one chromosome at a
        time.
    rc
        Also scan the reverse strand.
    no_overlaps
        Remove overlapping hits with
        [`dedup_hits()`][universalmotif.dedup_hits], keeping the best by
        ``no_overlaps_by``, which is ``"pvalue"`` or ``"score"``.
        ``no_overlaps_by_strand`` and ``no_overlaps_by_motif`` also compare
        hits across strands and across motifs.
    nthreads
        Threads to use; 0 means all available.
    use_gaps
        Scan gapped motifs, such as those made with
        [`add_gap()`][universalmotif.add_gap], with every combination of
        their gap sizes; otherwise their gaps are ignored. A gapped hit's
        ``match`` shows the gap as dots, and its P-value is that of its score
        under the motif without the gap. Gapped motifs are scanned on both
        strands like any other.
    correct_gaps
        Bonferroni-correct gapped motifs' P-values for the number of
        combinations of gap sizes tried: hits must reach ``pvalue`` divided
        by that number, and their P-values are multiplied by it and capped
        at 1.

    Returns
    -------
    polars.DataFrame
        A DataFrame with one row per hit and the columns ``motif``,
        ``motif_i``, ``sequence``, ``sequence_i``, ``start``, ``end``,
        ``strand``, ``score``, ``score_pct``, ``match`` and ``pvalue``.
        ``score_pct`` is the score as a percentage of the motif's maximum, and
        ``match`` is the matched sequence, reverse complemented on the ``-``
        strand. ``motif_i`` and ``sequence_i`` are 0-based indices into the
        inputs; ``start`` and ``end`` are positions on the forward strand,
        counted from 0 with the end left out, so a hit's letters are
        ``sequence[start:end]``. Rows are ordered by motif, sequence and
        start.

    Examples
    --------
    >>> import universalmotif as um
    >>> wbox = um.example_motifs()[5]
    >>> hits = um.scan_sequences(wbox, um.example_sequences())
    >>> hits.shape
    (30, 11)
    >>> hits.select("sequence", "start", "end", "strand", "match").row(0)
    ('target_004', 448, 457, '-', 'GTTTGACCG')
    >>> round(hits["pvalue"][0], 7)
    3.43e-05
    """
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    overlaps = (no_overlaps_by, no_overlaps_by_strand, no_overlaps_by_motif) if no_overlaps else None
    motif_names = _make_unique([m.name for m in ms])
    if isinstance(sequences, str) and len(sequences) < 4096 and os.path.isfile(sequences):
        warnings.warn(
            f"the string {sequences!r} names a file but is scanned as a sequence; pass "
            "pathlib.Path(...) to scan the file",
            stacklevel=2,
        )
    _warn_unreachable(ms, pvalue)
    from ._genome import Genome

    if isinstance(sequences, Genome):
        # One chromosome at a time, so only one is held in memory.
        parts = []
        for i, name in enumerate(sequences):
            seq = sequences[name]
            if i == 0:
                _warn_alphabet_mismatch(ms, _sample_letters([seq]))
            part = pl.DataFrame(
                _core._scan_sequences(ms, [seq], pvalue, rc, overlaps, nthreads, use_gaps,
                                      correct_gaps, motif_names, [name])
            ).cast(_HIT_SCHEMA)
            parts.append(part.with_columns(pl.lit(i, dtype=pl.UInt32).alias("sequence_i")))
            del seq
        if not parts:
            return pl.DataFrame(schema=_HIT_SCHEMA)
        hits = pl.concat(parts).sort("motif_i", "sequence_i", "start", maintain_order=True)
        return hits
    if isinstance(sequences, PackedSequences):
        hits = _core._scan_packed(ms, sequences._inner, pvalue, rc, overlaps, nthreads, use_gaps, correct_gaps, motif_names)
    elif isinstance(sequences, os.PathLike):
        path = os.fspath(sequences)
        hits = _core._scan_fasta(ms, path, pvalue, rc, overlaps, nthreads, use_gaps, correct_gaps, motif_names)
    else:
        names, seqs = _sequence_pairs(sequences)
        _warn_alphabet_mismatch(ms, _sample_letters(seqs))
        hits = _core._scan_sequences(ms, seqs, pvalue, rc, overlaps, nthreads, use_gaps, correct_gaps, motif_names, names)
    return pl.DataFrame(hits).cast(_HIT_SCHEMA)


def dedup_hits(
    hits: pl.DataFrame,
    by: str = "pvalue",
    reverse: bool = False,
    ignore_strand: bool = False,
    ignore_motif: bool = False,
    seq: str = "sequence",
    motif: str = "motif",
    strand: str = "strand",
) -> pl.DataFrame:
    """Remove overlapping hits.

    Hits are grouped by sequence, motif and strand. Within a group, overlapping
    hits are taken in order of the ``by`` column and each is kept unless it
    overlaps one already kept. Lower values come first for columns whose name
    starts with ``pval`` and higher values otherwise; ``reverse`` flips this,
    and ties go to the earlier row. ``ignore_strand`` and ``ignore_motif``
    compare hits across strands and motifs. ``start`` and ``end`` columns are
    required, with the end left out as in a slice.

    Examples
    --------
    A palindromic G-box site matches on both strands at the same place:

    >>> import universalmotif as um
    >>> gbox = um.example_motifs()[3]
    >>> hits = um.scan_sequences(gbox, "TTACACGTGTTAAACCACGTGGTT", pvalue=1e-3)
    >>> hits.select("start", "end", "strand").rows()
    [(2, 10, '+'), (2, 10, '-'), (14, 22, '+'), (14, 22, '-')]
    >>> um.dedup_hits(hits, ignore_strand=True).select("start", "end", "strand").rows()
    [(2, 10, '+'), (14, 22, '+')]
    """
    for col in (seq, motif, strand, "start", "end", by):
        if col not in hits.columns:
            raise ValueError(f"hits is missing the column {col!r}")
    if hits.height == 0:
        return hits
    start = hits["start"].cast(pl.Int64).to_list()
    end = hits["end"].cast(pl.Int64).to_list()
    if any(s is None or e is None for s, e in zip(start, end)):
        raise ValueError("start and end cannot contain nulls")
    if any(s > e for s, e in zip(start, end)):
        raise ValueError("all rows must satisfy start <= end")
    priority = hits[by].cast(pl.Float64).to_list()
    if any(p is None or p != p for p in priority):
        raise ValueError(f"the column {by!r} cannot contain missing values")
    lower_is_better = by.lower().startswith("pval")
    if reverse:
        lower_is_better = not lower_is_better
    if lower_is_better:
        priority = [-p for p in priority]

    def codes(values):
        index: dict[object, int] = {}
        return [index.setdefault(v, len(index)) for v in values]

    strand_codes = [{"+": 0, "-": 1, "*": 2}.get(s, -1) for s in hits[strand].to_list()]
    keep = _core._dedup(
        codes(hits[seq].to_list()),
        codes(hits[motif].to_list()),
        strand_codes,
        start,
        end,
        priority,
        ignore_strand,
        ignore_motif,
    )
    return hits.filter(pl.Series(keep))


__all__ = ["dedup_hits", "motif_pvalue", "scan_sequences"]
