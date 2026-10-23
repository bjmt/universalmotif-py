"""k-mers, backgrounds, complexity, shuffling, random sequences, implanting
and motif discovery."""

from __future__ import annotations

import os
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import IO

import numpy as np
import polars as pl

from . import _core
from ._analysis import _sequence_pairs
from ._api import _check_motif
from ._core import Motif

_ALPHABETS = {"DNA": "ACGT", "RNA": "ACGU", "AA": "ACDEFGHIKLMNPQRSTVWY"}


def _seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().randrange(2**63)
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _letters(alphabet: str | Iterable[str]) -> str:
    """Sorted unique letters of a string or of a sequence of letters."""
    return "".join(sorted(set("".join(alphabet))))


def _restore(sequences, names: list[str], seqs: list[str]):
    """Return new sequences in the same form as the input."""
    if isinstance(sequences, str):
        return seqs[0]
    if isinstance(sequences, Mapping):
        return dict(zip(names, seqs))
    return seqs


def get_kmers(letters: str | Sequence[str], k: int = 1) -> list[str]:
    """All k-mers of some letters, in order, with the first position varying
    slowest. The letters are used in the order given.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.get_kmers("AC", k=2)
    ['AA', 'AC', 'CA', 'CC']
    >>> len(um.get_kmers("ACGT", k=3))
    64
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    letters = "".join(letters)
    return _core._kmers(letters, k)


def count_kmers(string: str, k: int = 1, alph: str | Sequence[str] | None = None) -> pl.DataFrame:
    """Count the k-mers in a string.

    Every k-mer of ``alph``, which defaults to the letters in the string,
    gets a row in sorted order; k-mers with letters outside ``alph`` are not
    counted. Returns a DataFrame with columns ``kmer`` and ``count``, as in
    [`get_bkg()`][universalmotif.get_bkg].

    Examples
    --------
    >>> import universalmotif as um
    >>> um.count_kmers("AACGTTAC").rows()
    [('A', 3), ('C', 2), ('G', 1), ('T', 2)]
    >>> counts = um.count_kmers("AACGTTAC", k=2)
    >>> counts.shape
    (16, 2)
    >>> counts.filter(counts["count"] > 0).rows()
    [('AA', 1), ('AC', 2), ('CG', 1), ('GT', 1), ('TA', 1), ('TT', 1)]
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not string:
        raise ValueError("string cannot be empty")
    letters = _letters(string if alph is None else alph)
    if not letters:
        raise ValueError("alph cannot be empty")
    return pl.DataFrame(
        {"kmer": _core._kmers(letters, k), "count": _core._count_kmers(string, k, letters)},
        schema={"kmer": pl.String, "count": pl.Int64},
    )


def calc_windows(n: int, window: int = 1, overlap: int = 0, return_incomp: bool = True) -> pl.DataFrame:
    """Start and end positions of sliding windows over a sequence of length
    ``n``, counted from 0 with the end left out, so that each window is
    ``sequence[start:end]``.

    The last window may be shorter than the others unless ``return_incomp``
    is false. A window wider than ``n`` shrinks to ``n``.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.calc_windows(11, window=4, overlap=1).rows()
    [(0, 4), (3, 7), (6, 10), (9, 11)]
    >>> um.calc_windows(11, window=4, overlap=1, return_incomp=False).rows()
    [(0, 4), (3, 7), (6, 10)]
    """
    n, window, overlap = int(n), int(window), int(overlap)
    if n < 1:
        raise ValueError("n should be a positive integer")
    if window < 1:
        raise ValueError("window should be a positive integer")
    if overlap < 0:
        raise ValueError("overlap should be a non-negative integer")
    if window > n:
        window, overlap = n, 0
    if overlap >= window:
        raise ValueError("overlap cannot be greater or equal to window")
    wins = _core._windows(n, window, overlap, return_incomp)
    return pl.DataFrame(
        {"start": [w[0] for w in wins], "end": [w[1] for w in wins]},
        schema={"start": pl.Int64, "end": pl.Int64},
    )


def window_string(string: str, window: int = 1, overlap: int = 0, return_incomp: bool = True) -> list[str]:
    """Split a string into sliding windows, as laid out by
    [`calc_windows()`][universalmotif.calc_windows].

    Examples
    --------
    >>> import universalmotif as um
    >>> um.window_string("ACGTACGTACG", window=4)
    ['ACGT', 'ACGT', 'ACG']
    >>> um.window_string("ACGTACGTACG", window=4, overlap=1)
    ['ACGT', 'TACG', 'GTAC', 'CG']
    """
    if not string:
        raise ValueError("string cannot be empty")
    w = calc_windows(len(string), window, overlap, return_incomp)
    return [string[s:e] for s, e in w.iter_rows()]


_COMPLEXITY_METHODS = ("WoottonFederhen", "WoottonFederhenFast", "Trifonov", "TrifonovFast", "DUST")


def calc_complexity(
    string: str | Sequence[str],
    complexity_method: str = "WoottonFederhen",
    alph: str | Sequence[str] | None = None,
    trifonov_max_word_size: int = 7,
):
    """Sequence complexity.

    ``complexity_method`` is one of ``"WoottonFederhen"``, its approximation
    ``"WoottonFederhenFast"``, ``"Trifonov"``, ``"TrifonovFast"`` or
    ``"DUST"``. The methods come from Wootton and Federhen 1993, Trifonov
    1990 and Morgulis et al. 2006, and lower DUST values mean higher
    complexity. ``alph`` gives the possible letters. By default it is ``ACGT``
    when the strings hold only those letters and N, ``ACGU`` likewise for RNA,
    and otherwise every letter found in the strings, so that all strings are
    scored against the same alphabet. Returns a float for one string and a
    numpy array for several.

    Examples
    --------
    >>> import universalmotif as um
    >>> round(um.calc_complexity("ACGTTGCAAC"), 3)
    0.731
    >>> um.calc_complexity(["ACGTTGCAAC", "ATATATATAT"], alph="ACGT").round(3).tolist()
    [0.731, 0.399]
    """
    if complexity_method not in _COMPLEXITY_METHODS:
        raise ValueError(f"complexity_method must be one of {', '.join(_COMPLEXITY_METHODS)}")
    if trifonov_max_word_size < 1:
        raise ValueError("trifonov_max_word_size must be a positive integer")
    single = isinstance(string, str)
    strings = [string] if single else [str(s) for s in string]
    a = _default_alphabet(strings) if alph is None else "".join(alph)
    out = np.array(_core._complexity(strings, complexity_method, a, trifonov_max_word_size))
    return float(out[0]) if single else out


def sequence_complexity(
    sequences: str | Sequence[str] | Mapping[str, str],
    window_size: float = 20,
    window_overlap: float | None = None,
    method: str = "WoottonFederhen",
    trifonov_max_word_size: int = 7,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Complexity in sliding windows along sequences.

    ``window_size`` below 1 is a fraction of each sequence's length, and so
    is ``window_overlap`` between 0 and 1, which defaults to half the
    window, rounded. A window longer than a sequence shrinks to it. See
    [`calc_complexity()`][universalmotif.calc_complexity] for the methods. Returns a DataFrame with
    columns ``sequence``, ``start``, ``end`` and ``complexity``, where
    positions count from 0 and leave the end out.

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = {"seq": "ACGTTGCAACATATATATATATATAT"}
    >>> res = um.sequence_complexity(seqs, window_size=10, window_overlap=5)
    >>> res.select("start", "end").rows()
    [(0, 10), (5, 15), (10, 20), (15, 25), (20, 26)]
    >>> res["complexity"].round(3).to_list()
    [0.731, 0.644, 0.399, 0.399, 0.36]
    """
    if method not in _COMPLEXITY_METHODS:
        raise ValueError(f"method must be one of {', '.join(_COMPLEXITY_METHODS)}")
    names, seqs = _sequence_pairs(sequences)
    windows = _core._sequence_complexity(
        seqs, names, method, float(window_size), window_overlap, _default_alphabet(seqs),
        trifonov_max_word_size, nthreads,
    )
    return pl.DataFrame(windows).cast(
        {"sequence": pl.String, "start": pl.Int64, "end": pl.Int64, "complexity": pl.Float64}
    )


def _default_alphabet(seqs: list[str]) -> str:
    letters = set("".join(seqs))
    if letters <= set("ACGTN"):
        return "ACGT"
    if letters <= set("ACGUN"):
        return "ACGU"
    return _letters(letters)


def _write_meme_bkg(df: pl.DataFrame, file: str | os.PathLike[str] | IO[str]) -> None:
    lines = []
    order = -1
    for kmer, p in zip(df["kmer"], df["probability"]):
        if len(kmer) - 1 != order:
            order = len(kmer) - 1
            lines.append(f"#   order {order}")
        lines.append(f"{kmer:<8}{p:.3e}")
    text = "\n".join(lines) + "\n"
    if isinstance(file, (str, os.PathLike)):
        with open(file, "w") as f:
            f.write(text)
    else:
        file.write(text)


def get_bkg(
    sequences: str | Sequence[str] | Mapping[str, str],
    k: int | Sequence[int] = (1, 2, 3),
    pseudocount: int = 0,
    alphabet: str | Sequence[str] | None = None,
    rc: bool = False,
    merge: bool = True,
    window: bool = False,
    window_size: float = 0.1,
    window_overlap: float = 0,
    to_meme: str | os.PathLike[str] | IO[str] | None = None,
    nthreads: int = 1,
) -> pl.DataFrame | None:
    """k-mer counts and frequencies of sequences.

    Parameters
    ----------
    k
        k-mer sizes to count.
    pseudocount
        Added to every count.
    alphabet
        Letters to count. Sequences made only of A, C, G, T and N count as
        DNA and those made of A, C, G, U and N as RNA, with their four
        letters as the default; otherwise the default is the letters in the
        sequences.
    rc
        Also count the reverse complements.
    merge
        Combine all sequences into one set of counts; otherwise each
        sequence gets its own rows.
    window, window_size, window_overlap
        Count in sliding windows. Values of ``window_size`` below 1, and of
        ``window_overlap`` between 0 and 1, are fractions of each sequence's
        length. Merged windowed counts need sequences of equal length.
    to_meme
        A path or text file to write a MEME background file to instead of
        returning the counts. This needs ``k`` to run from 1 upwards, merged
        counts without windows, and no frequencies of exactly 0 or 1, which
        a pseudocount avoids.

    Returns
    -------
    polars.DataFrame
        A DataFrame with columns ``kmer``, ``count`` and ``probability``, plus
        ``sequence`` for per-sequence counts and ``start`` and ``end`` for
        windows, counted from 0 with the end left out. Without windows, rows are sorted by k-mer size and then
        alphabetically, and per-sequence counts list every sequence in turn
        under each k-mer. With windows, k-mers are sorted alphabetically
        whatever their size and each k-mer's windows follow one another,
        with per-sequence counts grouped by sequence first.

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = um.example_sequences()
    >>> um.get_bkg(seqs, k=1).rows()
    [('A', 32793, 0.32793), ('C', 17095, 0.17095), ('G', 17101, 0.17101), ('T', 33011, 0.33011)]
    >>> um.get_bkg(seqs, k=[1, 2]).shape
    (20, 3)
    """
    names, seqs = _sequence_pairs(sequences)
    ks = [int(k)] if isinstance(k, (int, np.integer)) else [int(x) for x in k]
    if any(x < 1 for x in ks):
        raise ValueError("k must be at least 1")
    if to_meme is not None:
        if window:
            raise ValueError("window=True cannot be used with to_meme")
        if not merge:
            raise ValueError("merge=False cannot be used with to_meme")
        if sorted(set(ks)) != list(range(1, max(ks) + 1)):
            raise ValueError("a MEME background file needs k to run from 1 upwards")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    letters = _default_alphabet(seqs) if alphabet is None else _letters(alphabet)
    win = (float(window_size), float(window_overlap)) if window else None
    df = pl.DataFrame(_core._get_bkg(seqs, names, ks, pseudocount, letters, rc, merge, win, nthreads))
    if to_meme is None:
        return df
    if pseudocount < 1 and df["probability"].is_in([0.0, 1.0]).any():
        raise ValueError(
            "MEME background files do not allow 0 or 1 values, please try again "
            "with a pseudocount higher than 0"
        )
    _write_meme_bkg(df, to_meme)
    return None


_SHUFFLE_METHODS = ("euler", "markov", "linear")


def shuffle_sequences(
    sequences: str | Sequence[str] | Mapping[str, str],
    k: int = 1,
    method: str = "euler",
    seed: int | None = None,
    window: bool = False,
    window_size: float = 0.1,
    window_overlap: float = 0.01,
    nthreads: int = 1,
):
    """Shuffle sequences while preserving their k-mer composition.

    The default ``method``, ``"euler"``, keeps k-mer counts exactly, along
    with the first and last k-1 letters. ``"linear"`` shuffles
    non-overlapping blocks of k letters, keeping only those blocks' counts,
    and ``"markov"`` draws a new sequence from the k-mer frequencies,
    keeping them only approximately. With ``window``, each
    sliding window is shuffled in turn; ``window_size`` below 1 and
    ``window_overlap`` between 0 and 1 are fractions of each sequence's
    length.

    Results depend on ``seed`` but not on ``nthreads``.

    Returns new sequences in the same form as the input, whether a string, a
    list or a dict with the same keys.

    Examples
    --------
    >>> import universalmotif as um
    >>> seq = "AACCGGTTACGTACGT"
    >>> shuffled = um.shuffle_sequences(seq, k=2, seed=1)
    >>> shuffled
    'ACCGTACGGTAACGTT'
    >>> um.count_kmers(shuffled, k=2).equals(um.count_kmers(seq, k=2))
    True
    """
    if method not in _SHUFFLE_METHODS:
        raise ValueError("method must be 'euler', 'markov' or 'linear'")
    if k < 1:
        raise ValueError("k must be at least 1")
    names, seqs = _sequence_pairs(sequences)
    win = (float(window_size), float(window_overlap)) if window else None
    out = _core._shuffle_sequences(seqs, k, method, _seed(seed), win, nthreads)
    return _restore(sequences, names, out)


def create_sequences(
    alphabet: str = "DNA",
    seqnum: int = 100,
    seqlen: int = 100,
    freqs: Sequence[float] | Mapping[str, float] | None = None,
    seed: int | None = None,
    nthreads: int = 1,
) -> list[str]:
    """Random sequences.

    ``alphabet`` is ``"DNA"``, ``"RNA"``, ``"AA"`` or a string of letters.
    ``freqs`` gives the frequency of every k-mer of the alphabet, either as a
    mapping from k-mers or in the sorted order that
    [`get_kmers()`][universalmotif.get_kmers] gives; their number sets k, and
    sequences are drawn from the Markov model of order k-1 they define. Letters
    are uniform by default.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.create_sequences(seqnum=3, seqlen=12, seed=1)
    ['TTTGGACAGCGC', 'TGGTAGCAAGTA', 'ATAGAAGGACGT']

    AT-rich sequences, from letter frequencies:

    >>> freqs = {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}
    >>> um.create_sequences(seqnum=2, seqlen=12, freqs=freqs, seed=1)
    ['TTTTGAAATAGA', 'TGGTATCAATTA']
    """
    letters = _ALPHABETS.get(alphabet, None) or _letters(alphabet)
    if len(letters) < 1:
        raise ValueError("the alphabet cannot be empty")
    if seqnum < 0 or seqlen < 1:
        raise ValueError("seqnum must be non-negative and seqlen positive")
    f = None
    if freqs is not None:
        if isinstance(freqs, Mapping):
            k = len(next(iter(freqs)))
            kl = _core._kmers(letters, k)
            missing = [x for x in kl if x not in freqs]
            if missing or len(freqs) != len(kl):
                raise ValueError(f"freqs must have one value for each of the {len(kl)} {k}-lets")
            f = [float(freqs[x]) for x in kl]
        else:
            f = [float(x) for x in freqs]
    return _core._create_sequences(letters, seqnum, seqlen, f, _seed(seed), nthreads)


def shuffle_motifs(motifs: Motif | Iterable[Motif], k: int = 2, seed: int | None = None):
    """Shuffle motif columns among the motifs, in blocks of ``k`` columns.

    The new motifs keep the original widths and types, with ``" [shuffled]"``
    added to their names. With ``k`` above 1, columns move in blocks of
    ``k`` and leftover columns stay at the ends.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:3]
    >>> shuffled = um.shuffle_motifs(motifs, seed=1)
    >>> [(m.name, m.width) for m in shuffled]
    [('TATA-box [shuffled]', 10), ('AP-1 [shuffled]', 9), ('CRE [shuffled]', 8)]
    >>> [m.consensus for m in shuffled]
    ['KASTAACNNW', 'AWRTATCTG', 'ANCAGTTG']
    """
    single = isinstance(motifs, Motif)
    ms = [motifs] if single else [_check_motif(m) for m in motifs]
    if k < 1:
        raise ValueError("k must be at least 1")
    out = _core._shuffle_motifs(ms, k, _seed(seed))
    return out[0] if single else out


def sample_sites(motif: Motif, n: int = 100, seed: int | None = None) -> list[str]:
    """Draw ``n`` sites from a motif, each letter chosen with the motif's
    probabilities at its position.

    Examples
    --------
    >>> import universalmotif as um
    >>> wbox = um.example_motifs()[5]
    >>> wbox.consensus
    'GBTTGACYS'
    >>> um.sample_sites(wbox, n=3, seed=1)
    ['TTTTGACCG', 'CGTTGACCC', 'ACTTGACTG']
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return _core._sample_sites(_check_motif(motif), n, _seed(seed))


_IMPLANT_SCHEMA = {
    "sequence_i": pl.Int64,
    "motif_i": pl.Int64,
    "start": pl.Int64,
    "width": pl.Int64,
    "strand": pl.String,
    "planted": pl.String,
}


def implant_motifs(
    motifs: Motif | Iterable[Motif],
    sequences: str | Sequence[str] | Mapping[str, str],
    n_per_seq: int = 1,
    rate: float | None = None,
    positions: Sequence[Sequence[int]] | None = None,
    strand: str = "both",
    centre_bias: int = 1,
    min_spacing: int = 0,
    max_retries: int = 100,
    seed: int | None = None,
    return_indices: bool = False,
):
    """Plant sites sampled from DNA or RNA motifs into sequences.

    Each sequence gets ``n_per_seq`` sites, a Poisson number with mean
    ``rate`` times its length if ``rate`` is given, or one site at each of
    its ``positions``, given as starts counted from 0 with one list per
    sequence.
    Every site
    comes from a motif chosen at random, on a random strand unless
    ``strand="+"``, and overwrites the sequence where it lands.

    Random positions are the mean of ``centre_bias`` uniform draws, so values
    above 1 favour the middle of sequences, and keep at least
    ``min_spacing`` letters from other sites; a site that cannot be placed
    in ``max_retries`` attempts is skipped with a warning.

    Returns the new sequences in the same form as the input, and with
    ``return_indices`` also a DataFrame of the sites with the columns
    ``sequence_i``, ``motif_i``, ``start``, ``width``, ``strand`` and
    ``planted``. The indices and ``start`` count from 0, and
    ``planted`` is the site as it reads on the forward strand.

    Examples
    --------
    >>> import universalmotif as um
    >>> wbox = um.example_motifs()[5]
    >>> seqs = um.create_sequences(seqnum=3, seqlen=20, seed=1)
    >>> new, sites = um.implant_motifs(wbox, seqs, seed=1, return_indices=True)
    >>> new
    ['TTTGGATCTTGACTATCGAA', 'TGGTAGGTCAAAGTGAGCTG', 'AGCTCGACCGGTCTGTCTAG']
    >>> sites.select("sequence_i", "start", "strand", "planted").rows()
    [(0, 6, '+', 'TCTTGACTA'), (1, 4, '-', 'AGGTCAAAG'), (2, 1, '+', 'GCTCGACCG')]
    """
    if strand not in ("both", "+"):
        raise ValueError("strand must be 'both' or '+'")
    if n_per_seq < 0:
        raise ValueError("n_per_seq must be a non-negative integer")
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    names, seqs = _sequence_pairs(sequences)
    pos = None if positions is None else [[int(p) for p in ps] for ps in positions]
    if pos is not None and any(p < 0 for ps in pos for p in ps):
        raise ValueError("positions must be at least 0")
    new, idx, warns = _core._implant_motifs(
        ms, seqs, n_per_seq, rate, pos, strand == "both", centre_bias, min_spacing, max_retries, _seed(seed)
    )
    for w in warns:
        warnings.warn(w, stacklevel=2)
    out = _restore(sequences, names, new)
    if return_indices:
        return out, pl.DataFrame(idx, schema=_IMPLANT_SCHEMA)
    return out


_FINDER_SCHEMA = {
    "name": pl.String,
    "consensus": pl.String,
    "rank": pl.Int64,
    "width": pl.Int64,
    "nsites": pl.Float64,
    "seqs_pos": pl.Int64,
    "seqs_neg": pl.Int64,
    "sites_pos": pl.Int64,
    "sites_neg": pl.Int64,
    "n_pos": pl.Int64,
    "n_neg": pl.Int64,
    "candidates": pl.Int64,
    "pvalue": pl.Float64,
    "log10_pvalue": pl.Float64,
    "qvalue": pl.Float64,
    "log10_qvalue": pl.Float64,
}


def motif_finder(
    sequences: Sequence[str] | Mapping[str, str],
    bkg_sequences: Sequence[str] | Mapping[str, str] | None = None,
    min_width: int = 6,
    max_width: int = 15,
    nmotifs: int = 10,
    hit_pvalue: float = 1e-4,
    stop_pvalue: float = 1e-3,
    qvalue: float = 0.05,
    dedup_overlap: float = 0.5,
    rc: bool = True,
    shuffle_k: int = 2,
    seed: int | None = None,
    pseudocount: int = 1,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Discover motifs enriched in DNA or RNA sequences.

    A minimal STREME-like search (Bailey 2021). At each width from
    ``min_width`` to ``max_width``, which is at most 30, k-mers that a
    one-sided Fisher's exact test finds in more positive than background
    sequences seed PPMs built from their Hamming-1 neighbours, which are
    refined over two rescans. A motif whose hits, the sites scoring with a
    P-value below ``hit_pvalue``, are enriched with a P-value of at most
    ``stop_pvalue`` is kept and its sites masked, up to ``nmotifs`` per width.
    Motifs from different widths sharing more than ``dedup_overlap`` of their
    sites are deduplicated, Benjamini-Hochberg q-values are computed,
    low-information flanks are trimmed, and motifs with q-values above
    ``qvalue`` are dropped.

    Each motif is the best of many candidate words, so its P-value, from
    Fisher's exact test on the sequences with hits, is multiplied by the
    number of words tested in the search that found it, given in
    ``candidates``, and the q-values count every word tested at every width.
    This keeps random sequences from giving motifs, though the refinement of
    each motif from its word is not accounted for, so the P-values are still
    somewhat optimistic. ``log10_pvalue`` and ``log10_qvalue`` stay finite
    when the P-values are too small for a double.

    Without ``bkg_sequences``, the background is the positives shuffled
    with [`shuffle_sequences()`][universalmotif.shuffle_sequences] using the
    Euler method with ``k=shuffle_k`` and ``seed``. Parallelism is across
    widths.

    Returns a DataFrame in order of P-value, with the motifs in an object
    column ``motif`` alongside ``name``, ``consensus``, ``rank``, ``width``,
    ``nsites``, ``seqs_pos``, ``seqs_neg``, ``sites_pos``, ``sites_neg``,
    ``n_pos``, ``n_neg``, ``candidates``, ``pvalue``, ``log10_pvalue``,
    ``qvalue`` and ``log10_qvalue``. The motifs are PPMs
    named ``motif_1``, ``motif_2`` and so on, with their q-values as
    E-values. ``seqs_pos`` and ``seqs_neg`` count sequences with a hit, and
    ``sites_pos`` and ``sites_neg`` count hits.

    Examples
    --------
    A site planted in each of 20 random sequences is found again, here as
    its reverse complement:

    >>> import universalmotif as um
    >>> site = um.create_motif("TTGACCAT", nsites=50)
    >>> seqs = um.create_sequences(seqnum=20, seqlen=100, seed=1)
    >>> seqs = um.implant_motifs(site, seqs, seed=1)
    >>> res = um.motif_finder(seqs, min_width=6, max_width=10, seed=1)
    >>> res.select("consensus", "seqs_pos", "seqs_neg").rows()
    [('TGGTCAA', 20, 0)]
    """
    if min_width < 3 or max_width > 30 or min_width > max_width:
        raise ValueError("widths must satisfy 3 <= min_width <= max_width <= 30")
    if shuffle_k < 1:
        raise ValueError("shuffle_k must be at least 1")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    _, seqs = _sequence_pairs(sequences)
    neg = None if bkg_sequences is None else _sequence_pairs(bkg_sequences)[1]
    motifs, cols = _core._motif_finder(
        seqs,
        neg,
        min_width,
        max_width,
        nmotifs,
        hit_pvalue,
        stop_pvalue,
        qvalue,
        dedup_overlap,
        rc,
        shuffle_k,
        _seed(seed),
        pseudocount,
        nthreads,
    )
    cols["name"] = [m.name for m in motifs]
    cols["consensus"] = [m.consensus or "" for m in motifs]
    cols["nsites"] = [m.nsites for m in motifs]
    df = pl.DataFrame({k: cols[k] for k in _FINDER_SCHEMA}, schema=_FINDER_SCHEMA)
    return df.with_columns(pl.Series("motif", motifs, dtype=pl.Object))


__all__ = [
    "calc_complexity",
    "calc_windows",
    "count_kmers",
    "create_sequences",
    "get_bkg",
    "get_kmers",
    "implant_motifs",
    "motif_finder",
    "sample_sites",
    "sequence_complexity",
    "shuffle_motifs",
    "shuffle_sequences",
    "window_string",
]
