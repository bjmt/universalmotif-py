"""De novo discovery of gapped motifs, two parts with a spacer of varying
length between them, and of pairs of motifs that favour a spacing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from . import _core, _progress
from ._analysis import _sequence_pairs
from ._enrich import _motif_list
from ._sequences import _seed
from ._threads import _nthreads

_GAPPED_SCHEMA = {
    "name": pl.String,
    "consensus": pl.String,
    "left": pl.Int64,
    "right": pl.Int64,
    "min_gap": pl.Int64,
    "max_gap": pl.Int64,
    "best_spacer": pl.Int64,
    "spacer_pairs": pl.Int64,
    "spacer_fold": pl.Float64,
    "strong_spacers": pl.List(pl.Int64),
    "k": pl.Int64,
    "seed_pair": pl.String,
    "seqs_pos": pl.Int64,
    "seqs_neg": pl.Int64,
    "n_pos": pl.Int64,
    "n_neg": pl.Int64,
    "candidates": pl.Int64,
    "seed_pairs": pl.Int64,
    "pvalue": pl.Float64,
    "log10_pvalue": pl.Float64,
    "spacing_pvalue": pl.Float64,
    "log10_spacing_pvalue": pl.Float64,
}


def gapped_motif_finder(
    sequences: Sequence[str] | Mapping[str, str],
    bkg_sequences: Sequence[str] | Mapping[str, str] | None = None,
    k: int | Sequence[int] = (4, 5, 6),
    min_gap: int = 0,
    max_gap: int = 31,
    nmotifs: int = 5,
    hit_pvalue: float = 1e-4,
    stop_pvalue: float = 1e-3,
    flank: int = 3,
    min_ic: float = 0.5,
    n_seed_pairs: int = 6,
    min_fraction: float = 0.02,
    holdout: float = 0.3,
    mismatch: bool | str = "auto",
    mask: bool = True,
    shuffle_k: int = 2,
    rc: bool = True,
    seed: int | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
    bkg_shuffles: int = 3,
) -> pl.DataFrame:
    """Find motifs made of two parts with a spacer of varying length.

    The finder is experimental, so its options, its results and the motifs
    it finds may change in later releases.

    First, a share of the sequences and of the background, ``holdout``, is
    set aside. The background is by default dinucleotide shuffles of the
    sequences. In the rest, the finder takes pairs of ``k``-mers at every
    spacing from ``min_gap`` to ``max_gap`` that are found in at least
    ``min_fraction`` of the sequences. For each k-mer size, these are
    counted in the sequences and in the background, and tested for
    enrichment with Fisher's exact test.

    For each size, each of the ``n_seed_pairs`` most enriched distinct
    pairs seeds a motif, over the window of spacings that gives it. Its
    sites, with ``flank`` letters on either side of each part, make a
    gapped motif. This is refined by rescanning, and columns below
    ``min_ic`` bits are trimmed at either end. The spacer sizes kept are
    those whose hits outnumber both the background and what the two parts
    would give if they were placed independently of each other. Each part
    then grows into informative columns. A motif that ends up with a
    single part is tried again as a direct and as an inverted repeat of
    itself. Motifs left without a gap are dropped.

    Each motif is then tested in the sequences held out, where it must pass
    two tests.

    - The P-value of its spacing, multiplied by the number of motifs
      tested, must be at most ``stop_pvalue``. This P-value is against
      independent placement of its parts and against the background at
      each spacing.
    - Sequences with hits must be enriched over the background at 0.05.

    Of the motifs that pass, the one with the strongest spacing is kept.
    Its hits are masked and the search is repeated. The tests use sequences
    the motifs were not fitted to, so the P-values stay honest however
    widely the search looks. More seeds and a lower ``min_fraction`` can
    therefore be used to look for rarer motifs.

    For genomic sequences, a background drawn from the genome and matched
    for composition, as from [`match_bkg()`][universalmotif.match_bkg],
    gives far better calibrated P-values than the default. Dinucleotide
    shuffles lose the loose repeats that real genomes are full of.

    Parameters
    ----------
    sequences
        The positive sequences, as a list or a mapping from names to
        sequences.
    bkg_sequences
        The background sequences, given in the same way. Without them, the
        background is the positives shuffled with
        [`shuffle_sequences()`][universalmotif.shuffle_sequences], keeping
        their ``shuffle_k``-mer counts.
    k
        The k-mer size, or a list of sizes, to seed motifs from, each from
        2 to 6.
    min_gap, max_gap
        The shortest and longest spacers searched, with ``max_gap`` at most
        31, so that the motifs found can be compared with their gaps.
    nmotifs
        The most motifs to find.
    hit_pvalue
        The P-value a site must reach to count as a hit, in (0, 1).
    stop_pvalue
        Keep a motif if the P-value of its spacing, corrected as described
        above, is at most this, in (0, 1]. With ``holdout`` at 0, the
        enrichment of its hits must reach this too.
    flank
        The letters added on either side of each part of a seed pair.
    min_ic
        Columns with fewer bits than this are trimmed from the ends of each
        part, and parts grow into columns with at least this many.
    n_seed_pairs
        The most enriched pairs of k-mers built into motifs for each k-mer
        size and each motif found.
    min_fraction
        Test only pairs found in at least this fraction of the sequences,
        from 0 to 1, and in at least 5. Lower values find rarer motifs at
        the cost of more tests to correct for.
    holdout
        The fraction of the sequences and of the background set aside to
        test the motifs found in the rest, at least 0 and below 0.9. With 0,
        both P-values come from all the sequences instead, corrected for the
        pairs and seeds tried. They are then somewhat optimistic.
    mismatch
        Whether to count pairs of 6-mers in each sequence that holds them
        with at most one mismatch, which finds the spacing of rarer motifs
        better but takes several times longer. With ``"auto"``, the
        search runs with exact matches and is run again with mismatches,
        keeping motifs at half of ``stop_pvalue``, only if it finds
        nothing. Shorter k-mers always match exactly.
    mask
        Mask low-complexity regions, such as homopolymers and short tandem
        repeats, with DUST in the sequences and the background before the
        search. Their repeated units otherwise look like parts with a
        favoured spacing.
    shuffle_k
        The size of the k-mers whose counts the shuffled background keeps.
        Only used without ``bkg_sequences``.
    bkg_shuffles
        How many shuffles of the sequences make up the background, without
        ``bkg_sequences``. Several steady it, so that rarer motifs are found
        whatever the seed.
    rc
        Search both strands.
    seed
        Seed for setting sequences aside and for shuffling the background;
        a fresh one by default.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the call has run for a second,
        and let Ctrl-C stop it; by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    polars.DataFrame
        A DataFrame in order of discovery, with one row per motif. The
        columns are these.

        - ``name``. The motifs are named ``gapped_1``, ``gapped_2`` and so
          on.
        - ``consensus``, which shows the gap as ``[min-max]``.
        - ``left`` and ``right``, the widths of the two parts.
        - ``min_gap`` and ``max_gap``.
        - ``best_spacer``, the spacer with the most hits beyond those
          expected, ``spacer_pairs``, the hits there, and ``spacer_fold``,
          those hits over the number expected. ``strong_spacers`` lists
          every spacer searched with clearly more hits than expected, by
          two standard errors, whether or not they are next to each other,
          so that a second favoured spacer outside the gap shows too. All
          four come from the sequences the motif was found in, not those
          held out, and the hits expected are those of its spacing test.
        - ``k`` and ``seed_pair``, the k-mer size and the pair each motif
          grew from.
        - ``seqs_pos`` and ``seqs_neg``, which count sequences with a hit.
        - ``n_pos`` and ``n_neg``.
        - ``candidates`` and ``seed_pairs``, which count the pairs tested
          and built into motifs at that size.
        - ``pvalue``, ``log10_pvalue``, ``spacing_pvalue`` and
          ``log10_spacing_pvalue``, the P-values described above.
        - ``motif``, an object column of the motifs as gapped PPMs.

    References
    ----------
    - Morgulis A, Gertz EM, Schäffer AA, Agarwala R (2006). A fast and
      symmetric DUST implementation to mask low-complexity DNA sequences.
      *Journal of Computational Biology* 13(5):1028–1040.
      <https://doi.org/10.1089/cmb.2006.13.1028>

    Examples
    --------
    Direct repeats of ``AGGTCA`` with spacers of 1 to 5 letters are planted
    in half of 400 random sequences. They are found with their spacing,
    here as their reverse complement:

    >>> import universalmotif as um
    >>> dr = um.add_gap(um.create_motif("AGGTCAAGGTCA"), gaploc=6, mingap=1, maxgap=5)
    >>> seqs = um.create_sequences(seqnum=400, seqlen=150, seed=1)
    >>> seqs = um.implant_motifs(dr, seqs[:200], seed=1) + seqs[200:]
    >>> found = um.gapped_motif_finder(seqs, k=5, seed=1)
    >>> found.select("consensus", "seqs_pos", "seqs_neg").row(0)
    ('TGACCT[1-5]TGACCT', 200, 5)
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: gapped_motif_finder(**args))
    if bkg_shuffles < 1:
        raise ValueError("bkg_shuffles must be at least 1")
    _, pos = _sequence_pairs(sequences)
    neg = None if bkg_sequences is None else list(_sequence_pairs(bkg_sequences)[1])
    ks = [int(k)] if isinstance(k, (int, np.integer)) else [int(x) for x in k]
    if mismatch not in (False, True, "auto"):
        raise ValueError("mismatch must be False, True or 'auto'")
    mismatch = {False: "off", True: "on", "auto": "auto"}[mismatch]
    found, info = _core._gapped_motif_finder(
        list(pos), neg, ks, min_gap, max_gap, nmotifs, hit_pvalue, stop_pvalue, flank, min_ic, n_seed_pairs,
        min_fraction, holdout, mismatch, mask, rc, shuffle_k, bkg_shuffles, _seed(seed), _nthreads(nthreads),
    )
    motifs = list(found)
    cols = {
        "name": [m.name for m in motifs],
        "consensus": [_consensus(m) for m in motifs],
        "left": [m.gaploc[0] for m in motifs],
        "right": [m.width - m.gaploc[0] for m in motifs],
        "min_gap": [m.mingap[0] for m in motifs],
        "max_gap": [m.maxgap[0] for m in motifs],
        "best_spacer": info["best_spacer"],
        "spacer_pairs": info["spacer_pairs"],
        "spacer_fold": info["spacer_fold"],
        "strong_spacers": info["strong_spacers"],
        "k": info["k"],
        "seed_pair": info["seed_pair"],
        "seqs_pos": info["seqs_pos"],
        "seqs_neg": info["seqs_neg"],
        "n_pos": info["n_pos"],
        "n_neg": info["n_neg"],
        "candidates": info["candidates"],
        "seed_pairs": info["seed_pairs"],
        "pvalue": info["pvalue"],
        "log10_pvalue": info["log10_pvalue"],
        "spacing_pvalue": info["spacing_pvalue"],
        "log10_spacing_pvalue": info["log10_spacing_pvalue"],
    }
    df = pl.DataFrame(cols, schema=_GAPPED_SCHEMA)
    return df.with_columns(pl.Series("motif", motifs, dtype=pl.Object))


def _consensus(m) -> str:
    from ._io import _gap_notation

    return _gap_notation(m) if m.isgapped else m.consensus


_PAIR_SCHEMA = {
    "name": pl.String,
    "consensus": pl.String,
    "left": pl.String,
    "right": pl.String,
    "left_strand": pl.String,
    "right_strand": pl.String,
    "arrangement": pl.String,
    "min_gap": pl.Int64,
    "max_gap": pl.Int64,
    "pairs": pl.Int64,
    "expected": pl.Float64,
    "fold": pl.Float64,
    "pvalue": pl.Float64,
    "log10_pvalue": pl.Float64,
    "repeat": pl.Boolean,
}


def motif_pair_finder(
    sequences: Sequence[str] | Mapping[str, str],
    bkg_sequences: Sequence[str] | Mapping[str, str] | None = None,
    motifs=None,
    nmotifs: int = 8,
    max_gap: int = 31,
    hit_pvalue: float = 1e-4,
    train_pvalue: float = 1e-3,
    stop_pvalue: float = 1e-3,
    min_pairs: int = 5,
    min_fold: float = 2.0,
    max_window: int = 3,
    holdout: float = 0.3,
    merge_overlap: float = 0.3,
    mask: bool = True,
    shuffle_k: int = 2,
    bkg_shuffles: int = 3,
    seed: int | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
) -> pl.DataFrame:
    """Find pairs of motifs whose sites favour a spacing.

    The finder is experimental, so its options, its results and the pairs
    it finds may change in later releases.

    Where [`gapped_motif_finder()`][universalmotif.gapped_motif_finder]
    builds both parts of a gapped motif from pairs of k-mers, this finder
    takes its parts as given and searches only their spacing. The parts are
    the ``motifs`` given, such as a factor's known motif, followed by up to
    ``nmotifs`` motifs from [`motif_finder()`][universalmotif.motif_finder]
    run on the same sequences and background, so that a partner no database
    holds can still be found. A part whose sites in the sequences mostly
    overlap those of an earlier part is dropped, as ``motif_finder()`` often
    returns shifted copies of one site. Because the parts are scanned rather
    than counted as exact k-mers, pairs found in only a few percent of the
    sequences, or with variable parts, are found far more often than by
    ``gapped_motif_finder()``.

    Each part is scanned over the sequences and the background on both
    strands, and every pair of sites up to ``max_gap`` letters apart is
    counted by arrangement and spacer. A share of the sequences and of the
    background, ``holdout``, is set aside. In the rest, each arrangement's
    window of 1 to ``max_window`` spacers with the smallest binomial P-value
    against the background's spacers is chosen, and those reaching
    ``train_pvalue`` are tested once in the sequences held out. There the
    P-value, multiplied by the number of windows tested, must reach
    ``stop_pvalue``, and the pairs must be at least ``min_fold`` times those
    expected. Pairs that pass and share most of their sites are merged,
    keeping the most significant.

    Tandem repeats, such as the telomeric CCCTAAA, put the sites of a motif
    for one unit at every unit, so they can pass as a pair of that motif
    with itself. Such pairs are marked in the ``repeat`` column. Masking
    low-complexity regions removes most of them beforehand.

    Parameters
    ----------
    sequences
        The positive sequences, as a list or a mapping from names to
        sequences.
    bkg_sequences
        The background sequences, given in the same way. Without them, the
        background is the positives shuffled with
        [`shuffle_sequences()`][universalmotif.shuffle_sequences], keeping
        their ``shuffle_k``-mer counts. For genomic sequences, a background
        drawn from the genome and matched for composition, as from
        [`match_bkg()`][universalmotif.match_bkg], is far better.
    motifs
        A DNA motif or a list of them to use as parts, before those found.
    nmotifs
        The most motifs ``motif_finder()`` adds to the parts, from 0.
    max_gap
        The longest spacer searched, at most 31.
    hit_pvalue
        The P-value a site of a part must reach. Parts too short to reach
        it are scanned at sixteen times the P-value of a perfect site of
        their width, up to 1e-3.
    train_pvalue
        The P-value a window must reach in the sequences it is chosen in to
        be tested in those held out.
    stop_pvalue
        The P-value a window must reach in the sequences held out, after
        multiplying by the number of windows tested.
    min_pairs
        The fewest pairs of sites a window needs where it is chosen.
    min_fold
        The fewest times more pairs than expected a window needs in the
        sequences held out.
    max_window
        The most spacers a window may span.
    holdout
        The share of the sequences and of the background held out, above 0
        and below 0.9.
    merge_overlap
        Pairs are merged when this share or more of the pairs of sites of
        one, the one with fewer, overlap those of the other. Above 1,
        nothing is merged.
    mask
        Mask low-complexity regions with DUST in the sequences and the
        background before the search.
    shuffle_k
        The size of the k-mers whose counts the shuffled background keeps.
        Only used without ``bkg_sequences``.
    bkg_shuffles
        How many shuffles of the sequences make up the background, without
        ``bkg_sequences``.
    seed
        Seed for the background, ``motif_finder()`` and the sequences held
        out; a fresh one by default.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the call has run for a second,
        and let Ctrl-C stop it; by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    polars.DataFrame
        One row per pair, most significant first, with these columns.

        - ``name``. The pairs are named ``pair_1``, ``pair_2`` and so on.
        - ``consensus``, which shows the spacer as ``[min-max]``.
        - ``left`` and ``right``, the names of the two parts, and
          ``left_strand`` and ``right_strand``, their strands. Pairs are
          read so that the first part is on the plus strand where the
          arrangement allows. Strands are those of each part as given or
          as found, and ``motif_finder()`` may give a motif on either
          strand, so the arrangement of two parts found de novo depends on
          how each happens to be written. The sites of a part that reads
          the same on both strands, as many dimers' sites do, are all taken
          to be on the plus strand.
        - ``arrangement``, ``"tandem"`` for parts on the same strand, and
          ``"convergent"`` or ``"divergent"`` for parts on opposite strands
          pointing towards or away from each other.
        - ``min_gap`` and ``max_gap``, the spacers of the window.
        - ``pairs``, the pairs of sites in the window in the sequences held
          out, ``expected``, those expected from the background, and
          ``fold``, the ratio of the two.
        - ``pvalue`` and ``log10_pvalue``, the corrected P-value from the
          sequences held out.
        - ``repeat``, described above.
        - ``motif``, the pair as one motif, gapped over the window (a
          spacer of 0 is written as a gap of 0 to 1), or ungapped when its
          parts sit side by side. ``left_motif`` and ``right_motif`` are
          the parts, as PPMs.

    Examples
    --------
    TGTCGG followed by TTGACT seven letters later is planted in a
    twentieth of 1,500 random sequences, with each site also alone in more
    of them. Given TGTCGG, the finder finds its partner, here with the
    letter before it, which ``motif_finder()`` took into its motif:

    >>> import universalmotif as um
    >>> seqs = um.create_sequences(seqnum=1500, seqlen=200, seed=1)
    >>> pair = um.add_gap(um.create_motif("TGTCGGTTGACT"), gaploc=6, mingap=7, maxgap=7)
    >>> seqs[::20] = um.implant_motifs(pair, seqs[::20], seed=1)
    >>> seqs[1::3] = um.implant_motifs(um.create_motif("TGTCGG"), seqs[1::3], seed=2)
    >>> seqs[2::6] = um.implant_motifs(um.create_motif("TTGACT"), seqs[2::6], seed=3)
    >>> found = um.motif_pair_finder(seqs, motifs=um.create_motif("TGTCGG", name="ARF"), seed=1)
    >>> found.select("consensus", "left").row(0)
    ('TGTCGG[6-6]GTTGACT', 'ARF')
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: motif_pair_finder(**args))
    _, pos = _sequence_pairs(sequences)
    neg = None if bkg_sequences is None else list(_sequence_pairs(bkg_sequences)[1])
    given = [] if motifs is None else _motif_list(motifs)
    parts, found, info = _core._motif_pair_finder(
        list(pos), neg, given, nmotifs, max_gap, hit_pvalue, train_pvalue, stop_pvalue, min_pairs, min_fold,
        max_window, holdout, merge_overlap, mask, shuffle_k, bkg_shuffles, _seed(seed), _nthreads(nthreads),
    )
    names = [m.name for m in parts]
    cols = {
        "name": [m.name for m in found],
        "consensus": [_consensus(m) for m in found],
        "left": [names[i] for i in info["left"]],
        "right": [names[i] for i in info["right"]],
        **{k: info[k] for k in ("left_strand", "right_strand", "arrangement", "min_gap", "max_gap", "pairs",
                                "expected", "fold", "pvalue", "log10_pvalue", "repeat")},
    }
    df = pl.DataFrame(cols, schema=_PAIR_SCHEMA)
    return df.with_columns(
        pl.Series("motif", list(found), dtype=pl.Object),
        pl.Series("left_motif", [parts[i] for i in info["left"]], dtype=pl.Object),
        pl.Series("right_motif", [parts[i] for i in info["right"]], dtype=pl.Object),
    )


__all__ = ["gapped_motif_finder", "motif_pair_finder"]
