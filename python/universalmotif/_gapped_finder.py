"""De novo discovery of gapped motifs: two parts with a spacer of varying
length between them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from . import _core, _progress
from ._analysis import _sequence_pairs
from ._sequences import _seed
from ._threads import _nthreads

_GAPPED_SCHEMA = {
    "name": pl.String,
    "consensus": pl.String,
    "left": pl.Int64,
    "right": pl.Int64,
    "min_gap": pl.Int64,
    "max_gap": pl.Int64,
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
    max_gap: int = 20,
    nmotifs: int = 3,
    hit_pvalue: float = 1e-4,
    stop_pvalue: float = 1e-3,
    flank: int = 3,
    min_ic: float = 0.5,
    n_seed_pairs: int = 6,
    min_fraction: float = 0.02,
    holdout: float = 0.3,
    mismatch: bool = False,
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
        Count pairs of 6-mers in each sequence that holds them with at most
        one mismatch. This finds the spacing of rarer motifs better but
        takes several times longer. Shorter k-mers always match exactly.
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
    c = m.consensus
    loc = m.gaploc[0]
    return f"{c[:loc]}[{m.mingap[0]}-{m.maxgap[0]}]{c[loc:]}"


__all__ = ["gapped_motif_finder"]
