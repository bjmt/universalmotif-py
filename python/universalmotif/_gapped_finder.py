"""De novo discovery of gapped motifs: two parts with a spacer of varying
length between them."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from . import _core
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
    "seed": pl.String,
    "seqs_pos": pl.Int64,
    "seqs_neg": pl.Int64,
    "n_pos": pl.Int64,
    "n_neg": pl.Int64,
    "candidates": pl.Int64,
    "seeds": pl.Int64,
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
    n_seeds: int = 6,
    min_fraction: float = 0.02,
    holdout: float = 0.3,
    mismatch: bool = False,
    mask: bool = True,
    shuffle_k: int = 2,
    rc: bool = True,
    seed: int | None = None,
    nthreads: int | None = None,
) -> pl.DataFrame:
    """Find motifs made of two parts with a spacer of varying length.

    A share of the sequences and of the background, ``holdout``, is first
    set aside. In the rest, pairs of ``k``-mers at every spacing from
    ``min_gap`` to ``max_gap`` found in at least ``min_fraction`` of the
    sequences are counted in the sequences and in the background, by default
    dinucleotide shuffles of the sequences, and tested for enrichment with
    Fisher's exact test, for each k-mer size in ``k``, from 2 to 6. For each
    size, each of the ``n_seeds`` most enriched distinct pairs, over the
    window of spacings that gives it, seeds a motif: its sites, with
    ``flank`` letters on either side of each part, make a gapped motif,
    refined by rescanning and trimmed of columns below ``min_ic`` bits at
    either end. The spacer sizes kept are those whose hits outnumber both the
    background and what the two parts would give if they were placed
    independently of each other. Each part then grows into informative
    columns, a motif that ends up with a single part is tried again as a
    direct and as an inverted repeat of itself, and motifs left without a gap
    are dropped.

    Each motif is then tested in the sequences held out: the P-value of its
    spacing, against independent placement of its parts and against the
    background at each spacing, multiplied by the number of motifs tested,
    must be at most ``stop_pvalue``, and sequences with hits must be enriched
    over the background at 0.05. Of those that pass, the one with the
    strongest spacing is kept, its hits are masked and the search repeated.
    As the tests use sequences the motifs were not fitted to, the P-values
    stay honest however widely the search looks, so more seeds and a lower
    ``min_fraction`` can be used to look for rarer motifs. With ``holdout``
    at 0, both P-values come from all the sequences instead, corrected for
    the pairs and seeds tried, and are somewhat optimistic.

    With ``mismatch``, pairs of 6-mers are counted in each sequence that
    holds them with at most one mismatch, which finds the spacing of rarer
    motifs better but takes several times longer. With ``mask``,
    low-complexity regions such as homopolymers and short tandem repeats are
    masked with DUST in the sequences and the background before the search,
    as their repeated units otherwise look like parts with a favoured
    spacing. For genomic sequences, a background drawn from the genome and
    matched for composition, as from
    [`match_bkg()`][universalmotif.match_bkg], gives far better calibrated
    P-values than the default dinucleotide shuffles, which lose the loose
    repeats that real genomes are full of.

    Returns a DataFrame in order of discovery, with the motifs in an object
    column ``motif`` alongside ``name``, ``consensus``, ``left``,
    ``right``, ``min_gap``, ``max_gap``, ``k``, ``seed``, ``seqs_pos``,
    ``seqs_neg``, ``n_pos``, ``n_neg``, ``candidates``, ``seeds``,
    ``pvalue``, ``log10_pvalue``, ``spacing_pvalue`` and
    ``log10_spacing_pvalue``. ``left`` and ``right`` are the widths of the
    two parts, ``k`` and ``seed`` give the k-mer size and the pair each
    motif grew from, and ``candidates`` counts the pairs tested at that
    size. The
    motifs are gapped PPMs named ``gapped1``, ``gapped2`` and so on, and the
    consensus shows the gap as ``[min-max]``. ``seqs_pos`` and ``seqs_neg``
    count sequences with a hit; the P-values are those described above.
    ``seed`` defaults to a random one, and ``nthreads`` to the package
    setting from [`set_nthreads()`][universalmotif.set_nthreads].

    References
    ----------
    - Morgulis A, Gertz EM, Schäffer AA, Agarwala R (2006). A fast and
      symmetric DUST implementation to mask low-complexity DNA sequences.
      *Journal of Computational Biology* 13(5):1028–1040.
      <https://doi.org/10.1089/cmb.2006.13.1028>

    Examples
    --------
    Direct repeats of ``AGGTCA`` with spacers of 1 to 5 letters, planted in
    half of 400 random sequences, are found with their spacing:

    >>> import universalmotif as um
    >>> dr = um.add_gap(um.create_motif("AGGTCAAGGTCA"), gaploc=6, mingap=1, maxgap=5)
    >>> seqs = um.create_sequences(seqnum=400, seqlen=150, seed=1)
    >>> seqs = um.implant_motifs(dr, seqs[:200], seed=1) + seqs[200:]
    >>> found = um.gapped_motif_finder(seqs, k=5, seed=1)
    >>> found.select("consensus", "seqs_pos", "seqs_neg").row(0)
    ('AGGTCA[1-5]AGGTCA', 200, 1)
    """
    _, pos = _sequence_pairs(sequences)
    neg = None if bkg_sequences is None else list(_sequence_pairs(bkg_sequences)[1])
    ks = [k] if isinstance(k, int) else [int(x) for x in k]
    found, info = _core._gapped_motif_finder(
        list(pos), neg, ks, min_gap, max_gap, nmotifs, hit_pvalue, stop_pvalue, flank, min_ic, n_seeds,
        min_fraction, holdout, mismatch, mask, rc, shuffle_k, _seed(seed), _nthreads(nthreads),
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
        "seed": info["seed"],
        "seqs_pos": info["seqs_pos"],
        "seqs_neg": info["seqs_neg"],
        "n_pos": info["n_pos"],
        "n_neg": info["n_neg"],
        "candidates": info["candidates"],
        "seeds": info["seeds"],
        "pvalue": [math.exp(x) for x in info["ln_pvalue"]],
        "log10_pvalue": [x / math.log(10) for x in info["ln_pvalue"]],
        "spacing_pvalue": [math.exp(x) for x in info["ln_spacing_pvalue"]],
        "log10_spacing_pvalue": [x / math.log(10) for x in info["ln_spacing_pvalue"]],
    }
    df = pl.DataFrame(cols, schema=_GAPPED_SCHEMA)
    return df.with_columns(pl.Series("motif", motifs, dtype=pl.Object))


def _consensus(m) -> str:
    c = m.consensus
    loc = m.gaploc[0]
    return f"{c[:loc]}[{m.mingap[0]}-{m.maxgap[0]}]{c[loc:]}"


__all__ = ["gapped_motif_finder"]
