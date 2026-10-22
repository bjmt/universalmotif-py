"""k-lets, backgrounds, complexity, shuffling, random sequences, implanting
and motif discovery."""

from __future__ import annotations

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


def get_klets(letters: str | Sequence[str], k: int = 1) -> list[str]:
    """All k-lets of some letters, in order (the first position varies
    slowest). The letters are used in the order given."""
    if k < 1:
        raise ValueError("k must be at least 1")
    letters = "".join(letters)
    return _core._klets(letters, k)


def count_klets(string: str, k: int = 1, alph: str | Sequence[str] | None = None) -> pl.DataFrame:
    """Count the k-lets in a string.

    Every k-let of ``alph`` (by default the letters in the string) gets a
    row, in sorted order; k-lets with letters outside ``alph`` are not
    counted. Returns a DataFrame with columns ``klets`` and ``counts``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not string:
        raise ValueError("string cannot be empty")
    letters = _letters(string if alph is None else alph)
    if not letters:
        raise ValueError("alph cannot be empty")
    return pl.DataFrame(
        {"klets": _core._klets(letters, k), "counts": _core._count_klets(string, k, letters)},
        schema={"klets": pl.String, "counts": pl.Int64},
    )


def calc_windows(n: int, window: int = 1, overlap: int = 0, return_incomp: bool = True) -> pl.DataFrame:
    """Start and stop coordinates (1-based, inclusive) of sliding windows
    over a sequence of length ``n``.

    The last window may be shorter than the others unless ``return_incomp``
    is false. A window wider than ``n`` shrinks to ``n``.
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
        {"start": [w[0] for w in wins], "stop": [w[1] for w in wins]},
        schema={"start": pl.Int64, "stop": pl.Int64},
    )


def window_string(string: str, window: int = 1, overlap: int = 0, return_incomp: bool = True) -> list[str]:
    """Split a string into sliding windows (see :func:`calc_windows`)."""
    if not string:
        raise ValueError("string cannot be empty")
    w = calc_windows(len(string), window, overlap, return_incomp)
    return [string[s - 1 : e] for s, e in w.iter_rows()]


_COMPLEXITY_METHODS = ("WoottonFederhen", "WoottonFederhenFast", "Trifonov", "TrifonovFast", "DUST")


def calc_complexity(
    string: str | Sequence[str],
    complexity_method: str = "WoottonFederhen",
    alph: str | Sequence[str] | None = None,
    trifonov_max_word_size: int = 7,
):
    """Sequence complexity.

    ``complexity_method`` is one of ``"WoottonFederhen"`` (Wootton and
    Federhen 1993), ``"WoottonFederhenFast"`` (its approximation),
    ``"Trifonov"`` (Trifonov 1990), ``"TrifonovFast"`` or ``"DUST"`` (Morgulis
    et al. 2006, where lower values mean higher complexity). ``alph`` gives
    the possible letters, by default those in the string. Returns a float for
    one string and a numpy array for several.
    """
    if complexity_method not in _COMPLEXITY_METHODS:
        raise ValueError(f"complexity_method must be one of {', '.join(_COMPLEXITY_METHODS)}")
    if trifonov_max_word_size < 1:
        raise ValueError("trifonov_max_word_size must be a positive integer")
    single = isinstance(string, str)
    strings = [string] if single else [str(s) for s in string]
    a = None if alph is None else "".join(alph)
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
    is ``window_overlap`` between 0 and 1 (by default half the window,
    rounded). A window longer than a sequence shrinks to it. See
    :func:`calc_complexity` for the methods. Returns a DataFrame with
    columns ``sequence``, ``start``, ``stop`` and ``complexity``.
    """
    if method not in _COMPLEXITY_METHODS:
        raise ValueError(f"method must be one of {', '.join(_COMPLEXITY_METHODS)}")
    names, seqs = _sequence_pairs(sequences)
    windows = _core._sequence_complexity(
        seqs, names, method, float(window_size), window_overlap, None, trifonov_max_word_size, nthreads
    )
    return pl.DataFrame(windows).cast(
        {"sequence": pl.String, "start": pl.Int64, "stop": pl.Int64, "complexity": pl.Float64}
    )


def _default_alphabet(seqs: list[str]) -> str:
    letters = set("".join(seqs))
    if letters <= set("ACGTN"):
        return "ACGT"
    if letters <= set("ACGUN"):
        return "ACGU"
    return _letters(letters)


def _write_meme_bkg(df: pl.DataFrame, file: str | IO[str]) -> None:
    lines = []
    order = -1
    for klet, p in zip(df["klet"], df["probability"]):
        if len(klet) - 1 != order:
            order = len(klet) - 1
            lines.append(f"#   order {order}")
        lines.append(f"{klet:<8}{p:.3e} ")
    text = "\n".join(lines) + "\n"
    if isinstance(file, str):
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
    to_meme: str | IO[str] | None = None,
    nthreads: int = 1,
) -> pl.DataFrame | None:
    """k-let counts and frequencies of sequences.

    Parameters
    ----------
    k
        k-let sizes to count.
    pseudocount
        Added to every count.
    alphabet
        Letters to count. By default A, C, G and T for DNA, A, C, G and U for
        RNA (sequences made of those letters and N), and otherwise the letters
        in the sequences.
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
        counts without windows, and no frequencies of exactly 0 or 1 (use a
        pseudocount).

    Returns
    -------
    A DataFrame with columns ``klet``, ``count`` and ``probability``, plus
    ``sequence`` for per-sequence counts and ``start`` and ``stop`` for
    windows. Rows are ordered as by the R package.
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
    """Shuffle sequences while preserving their k-let composition.

    ``method`` is ``"euler"`` (the default, which keeps k-let counts
    exactly, along with the first and last k-1 letters), ``"linear"``
    (shuffles non-overlapping blocks of k letters, keeping only those
    blocks' counts) or ``"markov"`` (draws a new sequence from the k-let
    frequencies, keeping them only approximately). With ``window``, each
    sliding window is shuffled in turn; ``window_size`` below 1 and
    ``window_overlap`` between 0 and 1 are fractions of each sequence's
    length.

    Results depend on ``seed`` but not on ``nthreads``. They differ from the
    R package's for the same seed.

    Returns new sequences in the same form as the input (a string, a list,
    or a dict with the same keys).
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
    ``freqs`` gives the frequency of every k-let of the alphabet (letters
    sorted, as from :func:`get_klets`), either in that order or as a mapping
    from k-lets; their number sets k, and sequences are drawn from the
    Markov model of order k-1 they define. Letters are uniform by default.
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
            kl = _core._klets(letters, k)
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
    ``k`` and leftover columns stay at the ends (R's ``method = "linear"``).
    """
    single = isinstance(motifs, Motif)
    ms = [motifs] if single else [_check_motif(m) for m in motifs]
    if k < 1:
        raise ValueError("k must be at least 1")
    out = _core._shuffle_motifs(ms, k, _seed(seed))
    return out[0] if single else out


def sample_sites(motif: Motif, n: int = 100, seed: int | None = None) -> list[str]:
    """Draw ``n`` sites from a motif, each letter chosen with the motif's
    probabilities at its position."""
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
    its ``positions`` (1-based starts, one list per sequence). Every site
    comes from a motif chosen at random, on a random strand unless
    ``strand="+"``, and overwrites the sequence where it lands.

    Random positions are the mean of ``centre_bias`` uniform draws, so values
    above 1 favour the middle of sequences, and keep at least
    ``min_spacing`` letters from other sites; a site that cannot be placed
    in ``max_retries`` attempts is skipped with a warning.

    Returns the new sequences in the same form as the input, and with
    ``return_indices`` also a DataFrame of the sites: ``sequence_i`` and
    ``motif_i`` (0-based), ``start`` (1-based), ``width``, ``strand`` and
    ``planted`` (the site as it reads on the forward strand).
    """
    if strand not in ("both", "+"):
        raise ValueError("strand must be 'both' or '+'")
    if n_per_seq < 0:
        raise ValueError("n_per_seq must be a non-negative integer")
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    names, seqs = _sequence_pairs(sequences)
    pos = None if positions is None else [[int(p) for p in ps] for ps in positions]
    if pos is not None and any(p < 1 for ps in pos for p in ps):
        raise ValueError("positions must be 1-based")
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
    "pvalue": pl.Float64,
    "qvalue": pl.Float64,
}


def motif_finder(
    sequences: Sequence[str] | Mapping[str, str],
    bkg_sequences: Sequence[str] | Mapping[str, str] | None = None,
    min_width: int = 6,
    max_width: int = 15,
    nmotifs: int = 10,
    hit_pvalue: float = 1e-4,
    stop_pvalue: float = 1e-3,
    qvalue: float = 1e-3,
    dedup_overlap: float = 0.5,
    rc: bool = True,
    shuffle_k: int = 2,
    seed: int | None = None,
    pseudocount: int = 1,
    nthreads: int = 1,
) -> pl.DataFrame:
    """Discover motifs enriched in DNA or RNA sequences.

    A minimal STREME-like search (Bailey 2021). At each width from
    ``min_width`` to ``max_width`` (at most 30), k-mers present in more
    positive than background sequences (by a one-sided Fisher's exact test)
    seed PPMs built from their Hamming-1 neighbours, which are refined over
    two rescans. A motif whose hits (sites scoring with a P-value below
    ``hit_pvalue``) are enriched with a P-value of at most ``stop_pvalue``
    is kept and its sites masked, up to ``nmotifs`` per width. Motifs from
    different widths sharing more than ``dedup_overlap`` of their sites are
    deduplicated, Benjamini-Hochberg q-values are computed, low-information
    flanks are trimmed, and motifs with q-values above ``qvalue`` are
    dropped.

    Without ``bkg_sequences``, the background is the positives shuffled
    with :func:`shuffle_sequences` (Euler method, ``k=shuffle_k``,
    ``seed``). Parallelism is across widths.

    Returns a DataFrame in order of P-value, with the motifs (PPMs named
    ``motif_1``, ``motif_2`` and so on, with their q-values as E-values) in
    an object column ``motif``, alongside ``name``, ``consensus``, ``rank``,
    ``width``, ``nsites``, ``seqs_pos`` and ``seqs_neg`` (sequences with a
    hit), ``sites_pos`` and ``sites_neg`` (hits), ``n_pos``, ``n_neg``,
    ``pvalue`` and ``qvalue``.
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
    "count_klets",
    "create_sequences",
    "get_bkg",
    "get_klets",
    "implant_motifs",
    "motif_finder",
    "sample_sites",
    "sequence_complexity",
    "shuffle_motifs",
    "shuffle_sequences",
    "window_string",
]
