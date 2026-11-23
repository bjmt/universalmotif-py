"""Choosing and refining motifs after discovery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from . import _core, _progress
from ._analysis import _sequence_pairs
from ._api import _check_motif
from ._core import Motif
from ._threads import _nthreads


def _motifs_of(motifs: Any) -> list[Motif]:
    """Motifs given as a motif, a list of them, or a table with a ``motif``
    column such as motif_finder() returns."""
    if isinstance(motifs, Motif):
        return [motifs]
    if isinstance(motifs, pl.DataFrame):
        if "motif" not in motifs.columns:
            raise ValueError("a table of motifs needs a 'motif' column, as motif_finder() gives")
        return [_check_motif(m) for m in motifs["motif"].to_list()]
    ms = [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("motifs is empty")
    return ms


def _sites_of(sites: Any, names: list[str]) -> list[tuple[int, int, int]] | None:
    """Sites as (sequence index, start, end), from a table with ``start`` and
    ``end`` columns and a ``sequence`` column of names or a ``sequence_i``
    column of indices, such as the hits of scan_sequences() or the seqlets of
    read_seqlets()."""
    if sites is None:
        return None
    if not isinstance(sites, pl.DataFrame):
        if type(sites).__module__.startswith("pandas"):
            sites = pl.from_pandas(sites)
        else:
            raise TypeError("sites must be a polars or pandas DataFrame")
    for col in ("start", "end"):
        if col not in sites.columns:
            raise ValueError(f"sites must have a {col!r} column")
    index = {n: i for i, n in enumerate(names)}
    if "sequence" in sites.columns and set(sites["sequence"].cast(pl.String).unique().to_list()) <= set(index):
        seq = [index[s] for s in sites["sequence"].cast(pl.String).to_list()]
    elif "sequence_i" in sites.columns:
        seq = sites["sequence_i"].cast(pl.Int64).to_list()
        if any(s < 0 or s >= len(names) for s in seq):
            raise ValueError(f"sites refer to sequences beyond the {len(names)} given")
    elif "sequence" in sites.columns:
        missing = sorted(set(sites["sequence"].cast(pl.String).unique().to_list()) - set(index))
        raise ValueError(f"sites refer to sequences not given, such as {missing[0]!r}")
    else:
        raise ValueError("sites need a 'sequence' column of names or a 'sequence_i' column of indices")
    start = sites["start"].cast(pl.Int64).to_list()
    end = sites["end"].cast(pl.Int64).to_list()
    if any(s < 0 for s in start) or any(e <= s for s, e in zip(start, end)):
        raise ValueError("sites must start at 0 or later and end after they start")
    return list(zip(seq, start, end))


def _rc(rc: bool | None, seqs: list[str]) -> bool:
    return not any("U" in s or "u" in s for s in seqs) if rc is None else rc


def select_motifs(
    motifs: Motif | Iterable[Motif] | pl.DataFrame,
    sequences: str | Sequence[str] | Mapping[str, str],
    sites: pl.DataFrame | None = None,
    pvalue: float = 1e-4,
    min_gain: float | Sequence[float] = 0.01,
    min_overlap: int = 1,
    max_similarity: float | None = None,
    compare_as: Iterable[Motif] | None = None,
    rc: bool | None = None,
    nthreads: int | None = None,
    reachable: bool = False,
    progress: bool | None = None,
) -> pl.DataFrame:
    """Rank motifs by how many new sequences or sites each reaches, and keep
    those that add enough.

    Discovery tends to give more motifs than the sequences need. Some are
    rare, and some are spellings of one site that each catch some of its
    instances. Here motifs are chosen greedily, each step taking the motif
    whose hits reach the most targets not yet reached. A motif is kept if
    it adds at least ``min_gain`` of all targets when its turn comes, and
    only kept motifs count as reaching their targets. The gains only shrink
    along the order, so the kept motifs come first.

    ``min_gain`` can also be given motif by motif. Motifs found in a second
    search among the targets the first left unreached are rare by design,
    for instance, and may deserve a lower cutoff than the rest. A motif held
    to a lower cutoff can then be kept after others were left out.

    Targets are the ``sites`` when given, such as the seqlets of a sequence
    model or the summits of peaks. A hit reaches a site when it shares at
    least ``min_overlap`` positions with it. Without ``sites``, the targets
    are the sequences, and any hit reaches its sequence.

    With ``max_similarity``, a motif is left out instead when its
    similarity to one already kept reaches that value at its turn. The
    targets it reaches are then not counted. Similarities are those of
    [`compare_motifs()`][universalmotif.compare_motifs] with its default
    method, between the motifs themselves. With ``compare_as``, they are
    between other motifs standing for them one for one, such as the CWMs
    of their hits. CWMs show when two spellings of a site are the same to
    a model.

    Parameters
    ----------
    motifs
        Motifs, as a list or as the table of
        [`motif_finder()`][universalmotif.motif_finder].
    sequences
        The sequences to reach, as for ``motif_finder()``.
    sites
        Targets within the sequences, as a table with ``start`` and ``end``
        columns, counted from 0 with the end left out. The table also needs
        a ``sequence`` column or a ``sequence_i`` column. A ``sequence``
        column names the sequences given as a mapping, or uses ``seq1``,
        ``seq2`` and so on for a list. A ``sequence_i`` column holds their
        indices from 0. The hits of
        [`scan_sequences()`][universalmotif.scan_sequences] and the seqlets
        of [`read_seqlets()`][universalmotif.read_seqlets] have these
        columns.
    pvalue
        The P-value cutoff for hits.
    min_gain
        The share of all targets a motif must add to be kept, as one value
        or as one value per motif.
    min_overlap
        With sites, the positions a hit must share with a site to reach it.
    max_similarity, compare_as
        Leave out motifs this similar to one already kept, as compared
        through ``compare_as`` if given.
    rc
        Also scan the reverse strand; by default on unless the sequences
        hold a U.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    reachable
        Let motifs too short or too degenerate to reach ``pvalue`` hit at
        the best score they can give, rather than not at all.
    progress
        Show a progress bar on stderr once the call has run for a second;
        by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    polars.DataFrame
        One row per motif, in the greedy order with motifs left out as
        similar last. The columns are these.

        - ``motif``, its name.
        - ``motif_i``, its place in ``motifs``.
        - ``rank``, from 1.
        - ``targets``, those it reaches on its own.
        - ``gain`` and ``cumulative``, the shares of all targets it adds,
          or would add if kept, and that the kept motifs up to it reach.
        - ``kept``.
        - ``similar_to``, the kept motif a left-out one resembles.
        - ``object``, the motif itself.

    Examples
    --------
    AP-1 sites are planted in half of the sequences and GATA sites in a
    third. A longer AP-1 motif reaches only sequences that AP-1 already
    has:

    >>> import universalmotif as um
    >>> seqs = um.create_sequences(seqlen=200, seqnum=300, seed=1)
    >>> ap1 = um.create_motif("TGACTCA", name="AP-1")
    >>> gata = um.create_motif("AGATAAG", name="GATA")
    >>> seqs = um.implant_motifs([ap1], seqs[:150], seed=1) + seqs[150:]
    >>> seqs = seqs[:100] + um.implant_motifs([gata], seqs[100:200], seed=2) + seqs[200:]
    >>> longer = um.create_motif("TGACTCAT", name="AP-1, longer")
    >>> sel = um.select_motifs([gata, ap1, longer], seqs)
    >>> sel.select("motif", "rank", "targets", "kept").rows()
    [('AP-1', 1, 152, True), ('GATA', 2, 106, True), ('AP-1, longer', 3, 47, False)]
    >>> kept = sel.filter("kept")["object"].to_list()
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: select_motifs(**args))
    ms = _motifs_of(motifs)
    names, seqs = _sequence_pairs(sequences)
    min_gains = None
    if np.ndim(min_gain) == 0:
        min_gain = float(min_gain)
        if not 0 <= min_gain <= 1:
            raise ValueError("min_gain must be between 0 and 1")
    else:
        min_gains = [float(g) for g in min_gain]
        if len(min_gains) != len(ms):
            raise ValueError(f"min_gain has {len(min_gains)} values for {len(ms)} motifs")
        if any(not 0 <= g <= 1 for g in min_gains):
            raise ValueError("min_gain must be between 0 and 1")
        min_gain = 0.0
    similarity = _similarity(ms, max_similarity, compare_as, nthreads)
    out = _core._select_motifs(ms, seqs, _sites_of(sites, names), similarity, pvalue, min_gain, min_overlap,
                               max_similarity, _rc(rc, seqs), _nthreads(nthreads), reachable, min_gains)
    return _selection(ms, out)


def _select_reached(ms: list[Motif], reached: list[list[int]], n_targets: int, min_gains: list[float],
                    max_similarity: float | None, compare_as: list[Motif] | None, nthreads: int) -> pl.DataFrame:
    """select_motifs() for motifs whose hits are already known, from the
    targets each reaches, as indices below ``n_targets``."""
    similarity = _similarity(ms, max_similarity, compare_as, nthreads)
    return _selection(ms, _core._select_reached(reached, n_targets, similarity, 0.0, max_similarity, min_gains))


def _similarity(ms: list[Motif], max_similarity: float | None, compare_as: Any,
                nthreads: int | None) -> list[list[float]] | None:
    """The similarities select_motifs() compares with ``max_similarity``."""
    if max_similarity is None:
        if compare_as is not None:
            raise ValueError("compare_as needs max_similarity")
        return None
    from ._compare import compare_motifs

    cmp = ms if compare_as is None else [_check_motif(m) for m in compare_as]
    if len(cmp) != len(ms):
        raise ValueError(f"compare_as has {len(cmp)} motifs for {len(ms)} motifs")
    # A single motif has nothing to be like.
    return (np.asarray(compare_motifs(cmp, nthreads=nthreads), dtype=float).tolist() if len(cmp) > 1
            else [[1.0]] * len(cmp))


def _selection(ms: list[Motif], out: dict) -> pl.DataFrame:
    """The table of select_motifs() from the choice made."""
    labels = [m.name for m in ms]
    similar = [labels[k] if k is not None else None for k in out["similar_to"]]
    df = pl.DataFrame({
        "motif": labels,
        "motif_i": pl.Series(range(len(ms)), dtype=pl.UInt32),
        "rank": pl.Series(out["rank"], dtype=pl.UInt32),
        "targets": pl.Series(out["targets"], dtype=pl.UInt64),
        "gain": out["gain"],
        "cumulative": out["cumulative"],
        "kept": out["kept"],
        "similar_to": pl.Series(similar, dtype=pl.String),
    })
    df = df.with_columns(pl.Series("object", ms, dtype=pl.Object))
    return df.sort("rank", nulls_last=True, maintain_order=True)


def refine_motifs(
    motifs: Motif | Iterable[Motif] | pl.DataFrame,
    sequences: str | Sequence[str] | Mapping[str, str],
    sites: pl.DataFrame | None = None,
    pvalue: float = 1e-4,
    flank: int = 5,
    iterations: int = 3,
    competitive: bool = True,
    min_ic: float = 0.25,
    max_width: int = 30,
    min_sites: int = 20,
    min_overlap: int = 1,
    rc: bool | None = None,
    nthreads: int | None = None,
    reachable: bool = False,
    progress: bool | None = None,
) -> list[Motif]:
    """Rebuild motifs from their hits, so that their edges and letters follow
    the sites they match.

    In each of ``iterations`` rounds the motifs are scanned. Each is
    rebuilt from its hits, widened by ``flank`` positions on either side
    and read along their strands. Edge columns below ``min_ic`` bits are
    then trimmed. A motif can thus grow to take in letters its sites share
    beyond its edges, or lose edges they don't share. With ``sites``, such
    as the seqlets of a sequence model, only hits reaching them are used. A
    motif wider than ``max_width`` keeps its most informative stretch of
    that width.

    Gapped motifs, such as those of
    [`gapped_motif_finder()`][universalmotif.gapped_motif_finder], are
    rebuilt part by part from the letters their hits put in each part,
    whatever the spacer. Their flanks are added and trimmed at their outer
    ends only, and their gaps keep their ranges.

    Motifs with fewer than ``min_sites`` hits are left as they are. Each
    rebuilt motif keeps the name, type and other details of the one given,
    with ``nsites`` the number of hits it was built from.

    With ``competitive``, as by default, overlapping hits of different
    motifs are reduced to the best one, so that each site builds only one
    motif. Without it, a weak motif whose hits fall on a stronger motif's
    sites can drift into a copy of it.

    Parameters
    ----------
    motifs
        Motifs, as a list or as the table of
        [`motif_finder()`][universalmotif.motif_finder].
    sequences
        The sequences to rebuild the motifs from, as for ``motif_finder()``.
    sites
        Stretches of the sequences that hits must reach to be used, as for
        [`select_motifs()`][universalmotif.select_motifs].
    pvalue
        The P-value cutoff for hits.
    flank
        Positions added on either side of each hit before rebuilding.
    iterations
        Rounds of scanning and rebuilding.
    competitive
        Give each site only to the motif that matches it best.
    min_ic
        The information content, in bits, below which edge columns are
        trimmed.
    max_width
        The widest a rebuilt motif may be.
    min_sites
        The fewest hits a motif is rebuilt from.
    min_overlap
        With sites, the positions a hit must share with a site to be used.
    rc
        Also scan the reverse strand; by default on unless the sequences
        hold a U.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    reachable
        Let motifs too short or too degenerate to reach ``pvalue`` hit at
        the best score they can give, rather than not at all.
    progress
        Show a progress bar on stderr once the call has run for a second;
        by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    list[Motif]
        The motifs in the order given.

    Examples
    --------
    A motif found as the core of a longer site grows to cover the whole of
    it:

    >>> import universalmotif as um
    >>> seqs = um.create_sequences(seqlen=200, seqnum=200, seed=3)
    >>> site = um.create_motif("GCTTGACTCATGAC")
    >>> seqs = um.implant_motifs([site], seqs[:100], seed=3) + seqs[100:]
    >>> core = um.create_motif("TGACTCA", nsites=20, pseudocount=1)
    >>> refined = um.refine_motifs(core, seqs)[0]
    >>> refined.consensus, int(refined.nsites)
    ('GCTTGACTCATGAC', 103)
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: refine_motifs(**args))
    ms = _motifs_of(motifs)
    names, seqs = _sequence_pairs(sequences)
    for name, value in (("flank", flank), ("iterations", iterations), ("min_sites", min_sites)):
        if value < 0:
            raise ValueError(f"{name} must be zero or positive")
    refined, _ = _core._refine_motifs(ms, seqs, _sites_of(sites, names), pvalue, flank, iterations, competitive,
                                      min_ic, max_width, min_sites, min_overlap, _rc(rc, seqs),
                                      _nthreads(nthreads), reachable)
    return refined


__all__ = ["refine_motifs", "select_motifs"]
