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

from . import _core, _progress
from ._analysis import _sequence_pairs
from ._api import _bkg_table, _check_motif
from ._core import Motif
from ._onehot import _array_strings, _is_array, _restore_array, _strings_array
from ._threads import _nthreads

_ALPHABETS = {"DNA": "ACGT", "RNA": "ACGU", "AA": "ACDEFGHIKLMNPQRSTVWY"}


def _seed(seed: int | None) -> int:
    if seed is None:
        return random.SystemRandom().randrange(2**63)
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _letter_set(seqs: Iterable[str], chunk: int = 4096) -> set[str]:
    """The distinct letters of some strings. They are read a few thousand at a
    time, and long stretches of ASCII text have their byte values counted,
    which is quicker than building a set from them."""
    letters: set[str] = set()
    seqs = list(seqs)
    for i in range(0, len(seqs), chunk):
        text = "".join(seqs[i:i + chunk])
        if len(text) > 1 << 16 and text.isascii():
            counts = np.bincount(np.frombuffer(text.encode("ascii"), dtype=np.uint8), minlength=128)
            letters.update(chr(b) for b in np.flatnonzero(counts))
        else:
            letters.update(text)
    return letters


def _letters(alphabet: str | Iterable[str]) -> str:
    """Sorted unique letters of a string or of a sequence of letters."""
    if isinstance(alphabet, str):
        letters = _letter_set([alphabet]) if len(alphabet) > 1 << 16 else set(alphabet)
    else:
        letters = set("".join(alphabet))
    return "".join(sorted(letters))


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

    Every k-mer of ``alph`` gets a row, in sorted order. ``alph`` defaults
    to the letters in the string, and k-mers with letters outside it are
    not counted. Returns a DataFrame with columns ``kmer`` and ``count``, as
    in [`get_bkg()`][universalmotif.get_bkg].

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
    ``n``.

    Positions count from 0 with the end left out, so each window is
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
    start, end = _core._windows(n, window, overlap, return_incomp)
    return pl.DataFrame({"start": start, "end": end}, schema={"start": pl.Int64, "end": pl.Int64})


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

    ``complexity_method`` is one of these.

    | Method | Reference |
    | --- | --- |
    | ``"WoottonFederhen"`` | Wootton and Federhen 1993 |
    | ``"WoottonFederhenFast"`` | Wootton and Federhen 1993, approximated |
    | ``"Trifonov"``, ``"TrifonovFast"`` | Trifonov 1990 |
    | ``"DUST"`` | Morgulis et al. 2006 |

    Lower DUST values mean higher complexity. ``alph`` gives the possible
    letters, so that all strings are scored against the same alphabet. By
    default it is ``ACGT`` when the strings hold only those letters and N,
    and ``ACGU`` likewise for RNA. Otherwise it is every letter found in
    the strings. Returns a float for one string and a numpy array for
    several.

    References
    ----------
    - Wootton JC, Federhen S (1993). Statistics of local complexity in amino
      acid sequences and sequence databases. *Computers & Chemistry*
      17(2):149–163. <https://doi.org/10.1016/0097-8485(93)85006-X>
    - Trifonov EN (1990). Making sense of the human genome. In Sarma RH, Sarma
      MH (eds), *Structure & Methods, Volume 1: Human Genome Initiative & DNA
      Recombination*, 69–77. Adenine Press.
    - Morgulis A, Gertz EM, Schäffer AA, Agarwala R (2006). A fast and
      symmetric DUST implementation to mask low-complexity DNA sequences.
      *Journal of Computational Biology* 13(5):1028–1040.
      <https://doi.org/10.1089/cmb.2006.13.1028>

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
    nthreads: int | None = None,
) -> pl.DataFrame:
    """Complexity in sliding windows along sequences.

    A ``window_size`` below 1 is a fraction of each sequence's length, and
    so is a ``window_overlap`` between 0 and 1. ``window_overlap`` defaults
    to half the window, rounded. A window longer than a sequence shrinks to
    it. See [`calc_complexity()`][universalmotif.calc_complexity] for the
    methods.

    Returns a DataFrame with columns ``sequence``, ``start``, ``end`` and
    ``complexity``, where positions count from 0 and leave the end out.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

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
    nthreads = _nthreads(nthreads)
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
    letters = _letter_set(seqs)
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
    nthreads: int | None = None,
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
        DNA, and those made of A, C, G, U and N as RNA. For these the four
        letters are the default. Otherwise the default is the letters in the
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
        A path or text file to write a MEME background file to, instead of
        returning the counts. This needs the following.

        - ``k`` running from 1 upwards.
        - Merged counts without windows.
        - No frequencies of exactly 0 or 1, which a pseudocount avoids.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].

    Returns
    -------
    polars.DataFrame
        A DataFrame with columns ``kmer``, ``count`` and ``probability``.
        Per-sequence counts add ``sequence``, and windows add ``start`` and
        ``end``, counted from 0 with the end left out.

        Without windows, rows are sorted by k-mer size and then
        alphabetically. Per-sequence counts then list every sequence in turn
        under each k-mer. With windows, k-mers are sorted alphabetically
        whatever their size, and each k-mer's windows follow one another.
        Per-sequence windowed counts are grouped by sequence first.

        Merged counts can be given as they are wherever a background is
        taken, such as the background of a motif. The single letters are
        then used.

    References
    ----------
    - Bailey TL, Johnson J, Grant CE, Noble WS (2015). The MEME Suite. *Nucleic
      Acids Research* 43(W1):W39–W49. <https://doi.org/10.1093/nar/gkv416>

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = um.example_sequences()
    >>> um.get_bkg(seqs, k=1).rows()
    [('A', 32793, 0.32793), ('C', 17095, 0.17095), ('G', 17101, 0.17101), ('T', 33011, 0.33011)]
    >>> um.get_bkg(seqs, k=[1, 2]).shape
    (20, 3)
    """
    nthreads = _nthreads(nthreads)
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
    sequences,
    k: int = 2,
    method: str = "euler",
    seed: int | None = None,
    window: bool = False,
    window_size: float = 0.1,
    window_overlap: float = 0.01,
    nthreads: int | None = None,
    alphabet: str = "ACGT",
    progress: bool | None = None,
):
    """Shuffle sequences while preserving their k-mer composition.

    By default ``k`` is 2, which keeps dinucleotide counts. The
    dinucleotide make-up of DNA, above all CpG depletion, changes how often
    motifs match, and shuffled sequences mostly serve as backgrounds for
    motifs. Results depend on ``seed`` but not on ``nthreads``.

    One-hot sequences are shuffled as the sequences they encode, with
    positions holding no letter read as N. The result is the one-hot
    encoding of shuffling the sequences as text, in the same layout and
    dtype. For several shuffles of each sequence that keep any Ns in place,
    as references for attribution, see
    [`shuffled_references()`][universalmotif.shuffled_references].

    Parameters
    ----------
    sequences
        A sequence, a list of sequences, a mapping from names to sequences,
        or one-hot sequences given as for
        [`insert_sites()`][universalmotif.insert_sites]. Each sequence must
        be at least ``k`` letters long.
    k
        The size of the k-mers whose counts are kept, at least 1.
    method
        How to shuffle.

        - ``"euler"`` keeps k-mer counts exactly, along with the first and
          last k-1 letters.
        - ``"linear"`` shuffles non-overlapping blocks of k letters, keeping
          only those blocks' counts.
        - ``"markov"`` draws a new sequence from the k-mer frequencies,
          keeping them only approximately.
    seed
        Seed for the shuffles; a fresh one by default.
    window
        Shuffle each sliding window of each sequence in turn, rather than
        whole sequences. This keeps changes in composition along the
        sequences. A sequence shorter than the window is shuffled whole.
    window_size
        The width of the windows, as a fraction of each sequence's length
        when below 1 and as a number of letters otherwise. It must come to
        at least ``k`` letters.
    window_overlap
        How much consecutive windows overlap, as a fraction of each
        sequence's length when below 1 and as a number of letters otherwise.
        It must be smaller than the windows.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    alphabet
        The letters of one-hot sequences, in the order of their rows or
        columns. Not used for sequences given as text.
    progress
        Show a progress bar on stderr once the call has run for a second,
        and let Ctrl-C stop it; by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    str | list[str] | dict[str, str] | numpy.ndarray
        New sequences in the same form as the input, whether a string, a
        list, a dict with the same keys, or one-hot sequences.

    References
    ----------
    - Altschul SF, Erickson BW (1985). Significance of nucleotide sequence
      alignments: a method for random sequence permutation that preserves
      dinucleotide and codon usage. *Molecular Biology and Evolution*
      2(6):526–538. <https://doi.org/10.1093/oxfordjournals.molbev.a040370>
    - Jiang M, Anderson J, Gillespie J, Mayne M (2008). uShuffle: a useful tool
      for shuffling biological sequences while preserving the k-let counts.
      *BMC Bioinformatics* 9:192. <https://doi.org/10.1186/1471-2105-9-192>

    Examples
    --------
    >>> import universalmotif as um
    >>> seq = "AACCGGTTACGTACGT"
    >>> shuffled = um.shuffle_sequences(seq, seed=1)
    >>> shuffled
    'ACCGTACGGTAACGTT'
    >>> um.count_kmers(shuffled, k=2).equals(um.count_kmers(seq, k=2))
    True
    >>> um.from_one_hot(um.shuffle_sequences(um.one_hot([seq]), seed=1))
    ['ACCGTACGGTAACGTT']
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: shuffle_sequences(**args))
    nthreads = _nthreads(nthreads)
    if method not in _SHUFFLE_METHODS:
        raise ValueError("method must be 'euler', 'markov' or 'linear'")
    if k < 1:
        raise ValueError("k must be at least 1")
    win = (float(window_size), float(window_overlap)) if window else None
    if _is_array(sequences):
        seqs, form = _array_strings(sequences, alphabet)
        out = _core._shuffle_sequences(seqs, k, method, _seed(seed), win, nthreads)
        return _strings_array(out, form, alphabet)
    names, seqs = _sequence_pairs(sequences)
    out = _core._shuffle_sequences(seqs, k, method, _seed(seed), win, nthreads)
    return _restore(sequences, names, out)


def shuffle_region(
    sequences,
    start: int,
    end: int,
    n: int = 20,
    k: int = 2,
    method: str = "euler",
    seed: int | None = None,
    nthreads: int | None = None,
    alphabet: str = "ACGT",
):
    """Shuffle one stretch of each sequence ``n`` times, leaving the rest as
    it is.

    This is for measuring what a model loses when a site or region is taken
    apart.
    Letters from ``start`` up to ``end``, counted from 0 with the end left
    out, are shuffled as by
    [`shuffle_sequences()`][universalmotif.shuffle_sequences], keeping their
    k-mer counts. The region must lie within every sequence. Comparing a
    model's predictions for the sequences with the mean of those for their
    shuffles gives the effect of the region, as with
    [`plot_marginalization()`][universalmotif.plot_marginalization].

    Sequences are given as for ``shuffle_sequences()``, including as one-hot
    arrays. The output depends on the input.

    | Input | Output |
    | --- | --- |
    | a string | a list of ``n`` strings |
    | a list | a list of lists |
    | a dict | a dict of lists |
    | one-hot, shape (sequences, letters, length) | an array of shape (sequences, n, letters, length) |

    A one-hot result keeps the layout and dtype of the input, and takes
    ``n`` times its memory. Results depend on ``seed`` and ``n`` but not on
    ``nthreads``, which defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

    References
    ----------
    - Altschul SF, Erickson BW (1985). Significance of nucleotide sequence
      alignments: a method for random sequence permutation that preserves
      dinucleotide and codon usage. *Molecular Biology and Evolution*
      2(6):526–538. <https://doi.org/10.1093/oxfordjournals.molbev.a040370>
    - Koo PK, Majdandzic A, Ploenzke M, Anand P, Paul SB (2021). Global
      importance analysis: an interpretability method to quantify importance of
      genomic features in deep neural networks. *PLOS Computational Biology*
      17(5):e1008925. <https://doi.org/10.1371/journal.pcbi.1008925>

    Examples
    --------
    >>> import universalmotif as um
    >>> um.shuffle_region("AAAAATGACTCAGTCATGACCCCCC", 5, 20, n=3, seed=1)
    ['AAAAATGTCATGAGACACTCCCCCC', 'AAAAATGTGACAGACTCATCCCCCC', 'AAAAATGACATGAGTCACTCCCCCC']
    >>> x = um.one_hot(["AAAAATGACTCAGTCATGACCCCCC", "TTTTTTGACTCAGTCATGAGGGGGG"])
    >>> um.shuffle_region(x, 5, 20, n=3).shape
    (2, 3, 4, 25)
    """
    nthreads = _nthreads(nthreads)
    if method not in _SHUFFLE_METHODS:
        raise ValueError("method must be 'euler', 'markov' or 'linear'")
    if k < 1:
        raise ValueError("k must be at least 1")
    if n < 0:
        raise ValueError("n must be 0 or more")
    start, end = int(start), int(end)
    array = _is_array(sequences)
    if array:
        seqs, form = _array_strings(sequences, alphabet)
    else:
        names, seqs = _sequence_pairs(sequences)
    if not 0 <= start < end:
        raise ValueError("the region must start at 0 or later and end after it starts")
    short = [len(s) for s in seqs if len(s) < end]
    if short:
        raise ValueError(f"the region from {start} to {end} does not fit in a sequence of length {min(short)}")
    if end - start < k:
        raise ValueError(f"the region is shorter than k ({k})")
    parts = [s[start:end] for s in seqs for _ in range(n)]
    shuffled = _core._shuffle_sequences(parts, k, method, _seed(seed), None, nthreads) if parts else []
    out = [seqs[i // n][:start] + part + seqs[i // n][end:] for i, part in enumerate(shuffled)] if n else []
    if array:
        if not n:
            empty = np.zeros((len(seqs), 0, len(alphabet), len(seqs[0]) if seqs else 0), dtype=form.dtype)
            return _restore_array(empty, form)
        return _strings_array(out, form, alphabet, lead=(n,))
    groups = [out[i * n:(i + 1) * n] for i in range(len(seqs))]
    if isinstance(sequences, str):
        return groups[0]
    if isinstance(sequences, Mapping):
        return dict(zip(names, groups))
    return groups


def create_sequences(
    alphabet: str = "DNA",
    seqnum: int = 100,
    seqlen: int = 100,
    freqs: Sequence[float] | Mapping[str, float] | None = None,
    seed: int | None = None,
    nthreads: int | None = None,
    output: str = "strings",
    dtype=np.float32,
):
    """Random sequences.

    Letters are drawn independently and uniformly by default. With
    ``freqs``, sequences are drawn from the Markov model of order k-1
    defined by the frequencies of every k-mer of the alphabet. Frequencies
    of single letters thus give independent letters with those frequencies.
    Frequencies of dinucleotides keep features such as CpG depletion.

    Parameters
    ----------
    alphabet
        ``"DNA"``, ``"RNA"``, ``"AA"`` or a string of letters, which are
        used in sorted order.
    seqnum
        The number of sequences, 0 or more.
    seqlen
        The length of each sequence, at least 1.
    freqs
        The frequency of every k-mer of the alphabet, given in one of these
        forms.

        - A mapping from k-mers.
        - A table from [`get_bkg()`][universalmotif.get_bkg], whose largest
          k-mers are used.
        - Values in the sorted order that
          [`get_kmers()`][universalmotif.get_kmers] gives.

        Their number sets k. Uniform letters by default.
    seed
        Seed for drawing the sequences; a fresh one by default.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    output
        ``"strings"`` for a list of strings, or ``"onehot"`` for an array as
        from [`one_hot()`][universalmotif.one_hot], ready for a model.
    dtype
        The numpy dtype of the one-hot array. Not used for strings.

    Returns
    -------
    list[str] | numpy.ndarray
        A list of ``seqnum`` strings, or with ``output="onehot"`` an array
        of shape (seqnum, letters, seqlen).

    Examples
    --------
    >>> import universalmotif as um
    >>> um.create_sequences(seqnum=3, seqlen=12, seed=1)
    ['TTTGGACAGCGC', 'TGGTAGCAAGTA', 'ATAGAAGGACGT']
    >>> um.create_sequences(seqnum=3, seqlen=12, seed=1, output="onehot").shape
    (3, 4, 12)

    AT-rich sequences, from letter frequencies:

    >>> freqs = {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}
    >>> um.create_sequences(seqnum=2, seqlen=12, freqs=freqs, seed=1)
    ['TTTTGAAATAGA', 'TGGTATCAATTA']
    """
    nthreads = _nthreads(nthreads)
    letters = _ALPHABETS.get(alphabet, None) or _letters(alphabet)
    if len(letters) < 1:
        raise ValueError("the alphabet cannot be empty")
    if output not in ("strings", "onehot"):
        raise ValueError("output must be 'strings' or 'onehot'")
    if seqnum < 0 or seqlen < 1:
        raise ValueError("seqnum must be non-negative and seqlen positive")
    f = None
    if freqs is not None:
        table = _bkg_table(freqs)
        if table is not freqs and table:
            k_max = max(len(x) for x in table)
            freqs = {x: v for x, v in table.items() if len(x) == k_max}
        if isinstance(freqs, Mapping):
            k = len(next(iter(freqs)))
            kl = _core._kmers(letters, k)
            missing = [x for x in kl if x not in freqs]
            if missing or len(freqs) != len(kl):
                raise ValueError(f"freqs must have one value for each of the {len(kl)} {k}-mers")
            f = [float(freqs[x]) for x in kl]
        else:
            f = [float(x) for x in freqs]
    seqs = _core._create_sequences(letters, seqnum, seqlen, f, _seed(seed), nthreads)
    if output == "onehot":
        from ._deep import one_hot

        alphabet = "".join(letters)
        return one_hot(seqs, alphabet=alphabet, dtype=dtype) if seqs else np.zeros((0, len(alphabet), seqlen), dtype)
    return seqs


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
    """Draw ``n`` sites from a motif.

    Each letter is chosen with the motif's probabilities at its position.
    Sites of a gapped motif have a spacer in each gap, of a size drawn
    uniformly within its range. Spacers are filled with letters drawn from
    the motif's background.

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
    "sequence_i": pl.UInt32,
    "motif_i": pl.UInt32,
    "start": pl.Int64,
    "width": pl.Int64,
    "strand": pl.String,
    "planted": pl.String,
}


def implant_motifs(
    motifs: Motif | Iterable[Motif],
    sequences,
    n_per_seq: int = 1,
    rate: float | None = None,
    positions: Sequence[Sequence[int]] | None = None,
    strand: str = "both",
    centre_bias: int = 1,
    min_spacing: int = 0,
    max_retries: int = 100,
    seed: int | None = None,
    return_indices: bool = False,
    alphabet: str = "ACGT",
):
    """Plant sites sampled from DNA or RNA motifs into sequences.

    How many sites each sequence gets depends on the arguments.

    - By default, ``n_per_seq`` sites.
    - With ``rate``, a Poisson number with mean ``rate`` times its length.
    - With ``positions``, one site at each position, given as starts
      counted from 0 with one list per sequence.

    Every site comes from a motif chosen at random, on a random strand
    unless ``strand="+"``. It overwrites the sequence where it lands.

    A gapped motif's sites get a spacer in each gap, of a size drawn
    uniformly within its range. The spacer keeps the sequence's own
    letters, so only the motif's columns are written.

    Random positions are the mean of ``centre_bias`` uniform draws, so values
    above 1 favour the middle of sequences. They keep at least
    ``min_spacing`` letters from other sites. A site that cannot be placed
    in ``max_retries`` attempts is skipped with a warning.

    Returns the new sequences in the same form as the input. This may be
    one-hot as for [`insert_sites()`][universalmotif.insert_sites], with the
    letters in the order of ``alphabet``. With ``return_indices``, a
    DataFrame of the sites is returned too, with the columns
    ``sequence_i``, ``motif_i``, ``start``, ``width``, ``strand`` and
    ``planted``. The indices and ``start`` count from 0, and ``width``
    includes any spacers. ``planted`` is the site as it reads on the
    forward strand.

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
    array = _is_array(sequences)
    if array:
        seqs, form = _array_strings(sequences, alphabet)
    else:
        names, seqs = _sequence_pairs(sequences)
    pos = None if positions is None else [[int(p) for p in ps] for ps in positions]
    if pos is not None and any(p < 0 for ps in pos for p in ps):
        raise ValueError("positions must be at least 0")
    new, idx, warns = _core._implant_motifs(
        ms, seqs, n_per_seq, rate, pos, strand == "both", centre_bias, min_spacing, max_retries, _seed(seed)
    )
    for w in warns:
        warnings.warn(w, stacklevel=2)
    out = _strings_array(new, form, alphabet) if array else _restore(sequences, names, new)
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
    rc: bool | None = None,
    shuffle_k: int = 2,
    seed: int | None = None,
    pseudocount: int = 1,
    nthreads: int | None = None,
    progress: bool | None = None,
    stability: int = 0,
    stability_fraction: float = 0.5,
    bkg_shuffles: int = 3,
) -> pl.DataFrame:
    """Discover motifs enriched in DNA or RNA sequences.

    This is a minimal STREME-like search (Bailey 2021). At each width, it
    runs as follows.

    1. A one-sided Fisher's exact test finds k-mers in more positive than
       background sequences.
    2. Each such k-mer seeds a PPM built from its Hamming-1 neighbours,
       which is refined over two rescans.
    3. A motif whose hits are enriched with a P-value of at most
       ``stop_pvalue`` is kept and its sites masked. The search at that
       width then goes on for the next motif.

    Afterwards, motifs from different widths that share too many sites are
    deduplicated, and Benjamini-Hochberg q-values are computed.
    Low-information flanks are trimmed, and motifs with q-values above
    ``qvalue`` are dropped. Motifs found on one strand have the ``strand``
    ``"+"``, so they are scanned on that strand alone.

    Each motif is the best of many candidate words. Its P-value, from
    Fisher's exact test on the sequences with hits, is therefore multiplied
    by the number of words tested in the search that found it, given in
    ``candidates``. The q-values count every word tested at every width.
    This keeps random sequences from giving motifs. The refinement of each
    motif from its word is not accounted for, though, so the P-values are
    still somewhat optimistic.

    Shuffles suit sequences without much structure. Genomic sequences hold
    many loose repeats, such as GA-rich stretches, that shuffling breaks up.
    Such repeats then come back as motifs enriched over the shuffles, even
    in regions with nothing to find. For genomic sequences, a background
    drawn from the genome and matched for composition gives far fewer false
    motifs, as from [`match_bkg()`][universalmotif.match_bkg].

    Parameters
    ----------
    sequences
        The positive sequences, as a list or a mapping from names to
        sequences.
    bkg_sequences
        The background sequences, given in the same way. Without them, the
        background is the positives shuffled with
        [`shuffle_sequences()`][universalmotif.shuffle_sequences] using the
        Euler method, keeping their ``shuffle_k``-mer counts.
    min_width, max_width
        The narrowest and widest motifs to search for, with
        ``3 <= min_width <= max_width <= 30``. Every width in between is
        searched.
    nmotifs
        The most motifs to keep at each width.
    hit_pvalue
        The P-value a site must reach to count as a hit.
    stop_pvalue
        Keep a motif if its hits are enriched with a P-value of at most
        this; the search at a width stops when no seed gives such a motif.
    qvalue
        Drop motifs with q-values above this.
    dedup_overlap
        The fraction of shared site positions above which the less
        significant of two motifs from different widths is dropped.
    rc
        Search both strands. By default on for DNA and off for RNA, which is
        recognized by a ``U`` in any sequence.
    shuffle_k
        The size of the k-mers whose counts the shuffled background keeps,
        at least 1. Only used without ``bkg_sequences``.
    bkg_shuffles
        How many shuffles of the sequences make up the background, without
        ``bkg_sequences``. Several steady the background, so motifs in only
        a few percent of the sequences are found whatever the seed. With
        one shuffle, a motif in a tenth of 400 sequences was missed with
        three seeds in eight. With three, it was found with all of them.
    seed
        Seed for shuffling the background; a fresh one by default.
    pseudocount
        A non-negative pseudocount added to the motifs' counts when scoring
        sites, which the motifs found keep as their ``pseudocount``.
    nthreads
        Threads to use, with 0 for all available, each searching one width
        at a time; by default the package setting from
        [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the call has run for a second,
        and let Ctrl-C stop it; by default the package setting from
        [`set_progress()`][universalmotif.set_progress].
    stability, stability_fraction
        With ``stability`` above 0, rerun the search that many times with
        the same options. Each rerun uses a random ``stability_fraction`` of
        the sequences and of the background. Each motif is given the share
        of reruns that find a motif with a similarity of at least 0.9 to
        it, as [`compare_motifs()`][universalmotif.compare_motifs] scores it
        by default. Motifs found in most reruns are those the search would
        find again in other data. Keeping them makes the motifs found from
        one sample of sequences to the next more alike. Each rerun takes
        about as long as the search itself.

    Returns
    -------
    polars.DataFrame
        A DataFrame in order of P-value. The motifs are in an object column
        ``motif``, alongside ``name``, ``consensus``, ``rank``, ``width``,
        ``nsites``, ``seqs_pos``, ``seqs_neg``, ``sites_pos``,
        ``sites_neg``, ``n_pos``, ``n_neg``, ``candidates``, ``pvalue``,
        ``log10_pvalue``, ``qvalue`` and ``log10_qvalue``.

        - The motifs are PPMs named ``motif_1``, ``motif_2`` and so on,
          with their q-values as E-values.
        - ``seqs_pos`` and ``seqs_neg`` count sequences with a hit, and
          ``sites_pos`` and ``sites_neg`` count hits.
        - ``log10_pvalue`` and ``log10_qvalue`` stay finite when the
          P-values are too small for a double.
        - With ``stability``, a ``stability`` column gives the share of
          reruns that found each motif again.

    References
    ----------
    - Bailey TL (2021). STREME: accurate and versatile sequence motif
      discovery. *Bioinformatics* 37(18):2834–2840.
      <https://doi.org/10.1093/bioinformatics/btab203>
    - Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a
      practical and powerful approach to multiple testing. *Journal of the
      Royal Statistical Society: Series B* 57(1):289–300.
      <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>

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
    [('ATGGTCA', 20, 1)]
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: motif_finder(**args))
    nthreads = _nthreads(nthreads)
    if min_width < 3 or max_width > 30 or min_width > max_width:
        raise ValueError("widths must satisfy 3 <= min_width <= max_width <= 30")
    if shuffle_k < 1:
        raise ValueError("shuffle_k must be at least 1")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    if bkg_shuffles < 1:
        raise ValueError("bkg_shuffles must be at least 1")
    if stability < 0 or not 0 < stability_fraction < 1:
        raise ValueError("stability must be zero or positive, and stability_fraction above 0 and below 1")
    _, seqs = _sequence_pairs(sequences)
    neg = None if bkg_sequences is None else _sequence_pairs(bkg_sequences)[1]
    if rc is None:
        rc = not any("U" in s or "u" in s for s in seqs)
    seed = _seed(seed)
    args = (min_width, max_width, nmotifs, hit_pvalue, stop_pvalue, qvalue, dedup_overlap, rc, shuffle_k,
            bkg_shuffles, seed, pseudocount, nthreads)
    motifs, cols = _core._motif_finder(seqs, neg, *args)
    cols["name"] = [m.name for m in motifs]
    cols["consensus"] = [m.consensus or "" for m in motifs]
    cols["nsites"] = [m.nsites for m in motifs]
    df = pl.DataFrame({k: cols[k] for k in _FINDER_SCHEMA}, schema=_FINDER_SCHEMA)
    if stability:
        st = _core._motif_stability(motifs, seqs, neg, *args, stability, stability_fraction, 0.9) if motifs else []
        df = df.with_columns(pl.Series("stability", st, dtype=pl.Float64))
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
    "shuffle_region",
    "shuffle_sequences",
    "window_string",
]
