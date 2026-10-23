"""Column conversions, scoring helpers, column metrics, gap and alphabet
operations, motif filtering, sequence masking and small helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from ._api import _apply, _check_motif
from ._core import Motif

_COMPARE_METRICS = ("PCC", "EUCL", "SW", "KL", "WEUCL", "ALLR", "BHAT", "HELL", "WPCC", "SEUCL",
                    "MAN", "ALLR_LL")


def _floats(x: Iterable[float]) -> list[float]:
    return [float(v) for v in x]


def _bkg(bkg: Sequence[float] | None) -> list[float]:
    return [] if bkg is None else _floats(bkg)


# Column conversions ---------------------------------------------------------


def pcm_to_ppm(position: Sequence[float], pseudocount: float = 0.0) -> npt.NDArray[np.float64]:
    """Convert a column of counts to probabilities, spreading a positive
    ``pseudocount`` evenly across the letters.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.pcm_to_ppm([3, 1, 0, 4]).tolist()
    [0.375, 0.125, 0.0, 0.5]
    >>> um.pcm_to_ppm([3, 1, 0, 4], pseudocount=1).round(3).tolist()
    [0.361, 0.139, 0.028, 0.472]
    """
    return np.array(_core._pcm_to_ppm(_floats(position), pseudocount))


def ppm_to_pcm(position: Sequence[float], nsites: float = 100.0) -> npt.NDArray[np.float64]:
    """Convert a column of probabilities to whole counts summing to
    ``nsites``.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.ppm_to_pcm([0.25, 0.1, 0.05, 0.6], nsites=20).tolist()
    [5.0, 2.0, 1.0, 12.0]
    """
    return np.array(_core._ppm_to_pcm(_floats(position), nsites))


def ppm_to_pwm(
    position: Sequence[float],
    bkg: Sequence[float] | None = None,
    pseudocount: float = 1.0,
    nsites: float = 100.0,
    smooth: bool = True,
) -> npt.NDArray[np.float64]:
    """Convert a column of probabilities to log2 odds against ``bkg``, which
    is uniform by default. With ``smooth``, the pseudocount is first applied
    by converting to ``nsites`` counts and back.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.ppm_to_pwm([0.7, 0.1, 0.1, 0.1], smooth=False).round(3).tolist()
    [1.485, -1.322, -1.322, -1.322]
    >>> um.ppm_to_pwm([0.7, 0.1, 0.1, 0.1]).round(3).tolist()
    [1.476, -1.301, -1.301, -1.301]
    """
    return np.array(_core._ppm_to_pwm(_floats(position), _bkg(bkg), pseudocount if smooth else 0.0,
                                      nsites))


def pwm_to_ppm(position: Sequence[float], bkg: Sequence[float] | None = None) -> npt.NDArray[np.float64]:
    """Convert a column of log2 odds back to probabilities.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.pwm_to_ppm([1.485, -1.322, -1.322, -1.322]).round(3).tolist()
    [0.7, 0.1, 0.1, 0.1]
    """
    return np.array(_core._pwm_to_ppm(_floats(position), _bkg(bkg)))


def ppm_to_icm(
    position: Sequence[float], bkg: Sequence[float] | None = None, relative_entropy: bool = False
) -> npt.NDArray[np.float64]:
    """Convert a column of probabilities to information content, as relative
    entropy against ``bkg`` if asked. With relative entropy, letters less
    common than in the background get 0 rather than a negative height.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.ppm_to_icm([0.7, 0.1, 0.1, 0.1]).round(3).tolist()
    [0.45, 0.064, 0.064, 0.064]
    >>> um.ppm_to_icm([0.7, 0.1, 0.1, 0.1], relative_entropy=True).round(3).tolist()
    [1.04, 0.0, 0.0, 0.0]
    """
    return np.array(_core._ppm_to_icm(_floats(position), _bkg(bkg), relative_entropy))


def icm_to_ppm(position: Sequence[float]) -> npt.NDArray[np.float64]:
    """Convert a column of information content back to probabilities.

    Examples
    --------
    >>> import universalmotif as um
    >>> icm = um.ppm_to_icm([0.7, 0.1, 0.1, 0.1])
    >>> um.icm_to_ppm(icm).round(3).tolist()
    [0.7, 0.1, 0.1, 0.1]
    """
    return np.array(_core._icm_to_ppm(_floats(position)))


# Scoring helpers ------------------------------------------------------------


def _one_motif(motif: object) -> Motif:
    if isinstance(motif, Motif):
        return motif
    raise TypeError(f"expected a single Motif, got {type(motif).__name__}")


def motif_range(motif: Motif, allow_nonfinite: bool = False) -> tuple[float, float]:
    """The lowest and highest scores the motif can give.

    Scores use the motif's PWM with each entry truncated to three decimals,
    as scanning does. Unless ``allow_nonfinite``, a PWM with ``-inf`` entries
    gets a pseudocount first; otherwise the lowest score is ``-inf``.

    Examples
    --------
    >>> import universalmotif as um
    >>> sites = ["TGACTCA", "TGAGTCA", "TGACTCA", "TGAGTCA", "TGACTAA"]
    >>> motif = um.create_motif(sites, pseudocount=1)
    >>> um.motif_range(motif)
    (-18.088, 11.652)
    """
    return _core._motif_range(_one_motif(motif), allow_nonfinite)


def motif_score(
    motif: Motif,
    threshold: float | Sequence[float] = (0.0, 1.0),
    threshold_type: str = "total",
    allow_nonfinite: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Scores at fractions of the motif's score range.

    With ``threshold_type="total"``, 0 is the lowest possible score and 1 the
    highest; with ``"fromzero"``, 0 is a score of zero. A single threshold
    gives a single score.

    Examples
    --------
    >>> import universalmotif as um
    >>> sites = ["TGACTCA", "TGAGTCA", "TGACTCA", "TGAGTCA", "TGACTAA"]
    >>> motif = um.create_motif(sites, pseudocount=1)
    >>> round(um.motif_score(motif, 0.9), 3)
    8.678
    >>> um.motif_score(motif, [0, 0.5, 1]).round(3).tolist()
    [-18.088, -3.218, 11.652]
    """
    single = np.ndim(threshold) == 0
    t = [float(threshold)] if single else _floats(threshold)  # type: ignore[arg-type]
    out = np.array(_core._motif_score(_one_motif(motif), t, threshold_type, allow_nonfinite))
    return float(out[0]) if single else out


def score_match(
    motif: Motif, match: str | Sequence[str], allow_nonfinite: bool = False
) -> float | npt.NDArray[np.float64]:
    """The score of each word, which must be as long as the motif, with each
    PWM entry truncated to three decimals and then added up, so that a word
    scores exactly what [`scan_sequences()`][universalmotif.scan_sequences]
    reports for it. As in
    [`motif_pvalue()`][universalmotif.motif_pvalue], a motif whose PWM would
    have infinite values, such as a count matrix without a pseudocount, gets
    its pseudocount reapplied first, or 1 if it has none, unless
    ``allow_nonfinite``.

    Examples
    --------
    >>> import universalmotif as um
    >>> sites = ["TGACTCA", "TGAGTCA", "TGACTCA", "TGAGTCA", "TGACTAA"]
    >>> motif = um.create_motif(sites, pseudocount=1)
    >>> um.score_match(motif, "TGACTCA")
    11.652
    >>> um.score_match(motif, ["TGAGTAA", "AAAAAAA"]).round(3).tolist()
    [9.356, -6.985]
    """
    words = [match] if isinstance(match, str) else list(match)
    out = np.array(_core._score_match(_one_motif(motif), words, allow_nonfinite))
    return float(out[0]) if isinstance(match, str) else out


def get_scores(motif: Motif, allow_nonfinite: bool = False) -> npt.NDArray[np.float64]:
    """Every possible score of the motif, highest first. There are as many as
    the alphabet size to the power of the width, so this is for small
    motifs.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.create_motif(["TCA", "TCA", "TGA"], pseudocount=1)
    >>> scores = um.get_scores(motif)
    >>> len(scores), scores[:3].round(3).tolist()
    (64, [4.569, 3.721, 1.4])
    """
    return np.array(_core._get_scores(_one_motif(motif), allow_nonfinite))


def get_matches(motif: Motif, score: float, allow_nonfinite: bool = False) -> list[str]:
    """Every word scoring at least ``score``. A score of ``inf``, which
    [`motif_pvalue()`][universalmotif.motif_pvalue] gives for an
    unreachable P-value, has no matches.

    Examples
    --------
    >>> import universalmotif as um
    >>> sites = ["TGACTCA", "TGAGTCA", "TGACTCA", "TGAGTCA", "TGACTAA"]
    >>> motif = um.create_motif(sites, pseudocount=1)
    >>> um.get_matches(motif, um.motif_score(motif, 0.9))
    ['TGACTCA', 'TGACTAA', 'TGAGTCA', 'TGAGTAA']
    """
    return _core._get_matches(_one_motif(motif), float(score), allow_nonfinite)


def prob_match(
    motif: Motif, match: str | Sequence[str], allow_zero: bool = True
) -> float | npt.NDArray[np.float64]:
    """The probability of each word under the motif's background. Each word
    must have as many letters as the motif has positions. Unless
    ``allow_zero``, a background with zeros gets a small pseudocount first.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.create_motif("TCA", bkg=[0.3, 0.2, 0.2, 0.3])
    >>> um.prob_match(motif, ["TCA", "GGG"]).round(4).tolist()
    [0.018, 0.008]
    """
    words = [match] if isinstance(match, str) else list(match)
    out = np.array(_core._prob_match(_one_motif(motif), words, allow_zero))
    return float(out[0]) if isinstance(match, str) else out


def prob_match_bkg(bkg: Mapping[str, float], match: str | Sequence[str]) -> float | npt.NDArray[np.float64]:
    """The probability of each word under the background frequencies
    ``bkg``, a mapping from letters to frequencies.

    Examples
    --------
    >>> import universalmotif as um
    >>> bkg = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    >>> round(um.prob_match_bkg(bkg, "TATA"), 4)
    0.0081
    >>> um.prob_match_bkg(bkg, ["AAA", "GCG"]).round(4).tolist()
    [0.027, 0.008]
    """
    pairs = [(str(k), float(v)) for k, v in bkg.items()]
    if any(len(k) != 1 for k, _ in pairs):
        raise ValueError("bkg keys must be single letters")
    words = [match] if isinstance(match, str) else list(match)
    out = np.array(_core._prob_match_bkg(pairs, words))
    return float(out[0]) if isinstance(match, str) else out


# Motif utilities ------------------------------------------------------------


def compare_columns(
    x: Sequence[float],
    y: Sequence[float],
    method: str,
    bkg1: Sequence[float] | None = None,
    bkg2: Sequence[float] | None = None,
    nsites1: float = 100.0,
    nsites2: float = 100.0,
) -> float:
    """Compare two columns with one of the comparison metrics: ``"PCC"``,
    ``"EUCL"``, ``"SW"``, ``"KL"``, ``"WEUCL"``, ``"ALLR"``, ``"BHAT"``,
    ``"HELL"``, ``"WPCC"``, ``"SEUCL"``, ``"MAN"`` or ``"ALLR_LL"``. The
    backgrounds are uniform by default and are used by the weighted metrics
    and ALLR, which also uses the site counts.

    Examples
    --------
    >>> import universalmotif as um
    >>> x = [0.7, 0.1, 0.1, 0.1]
    >>> y = [0.6, 0.2, 0.1, 0.1]
    >>> round(um.compare_columns(x, y, "PCC"), 3)
    0.98
    >>> round(um.compare_columns(x, y, "EUCL"), 3)
    0.141
    """
    if method not in _COMPARE_METRICS:
        raise ValueError(f"method must be one of {', '.join(_COMPARE_METRICS)}")
    xs, ys = _floats(x), _floats(y)
    b1 = [1 / len(xs)] * len(xs) if bkg1 is None else _floats(bkg1)
    b2 = [1 / len(ys)] * len(ys) if bkg2 is None else _floats(bkg2)
    return _core._compare_columns(xs, ys, method, b1, b2, nsites1, nsites2)


def average_ic(motifs: Motif | Iterable[Motif], average: str = "a.mean") -> float | npt.NDArray[np.float64]:
    """The average information content of the motifs' positions, taken as
    the arithmetic mean with ``"a.mean"``, the geometric mean with
    ``"g.mean"``, the median with ``"median"`` or the Fisher z-transform
    with ``"fzt"``.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> round(um.average_ic(motifs[0]), 3)
    0.883
    >>> um.average_ic(motifs[:3], average="median").round(3).tolist()
    [0.865, 1.215, 1.395]
    """
    if isinstance(motifs, Motif):
        return float(_core._average_ic([motifs], average)[0])
    return np.array(_core._average_ic([_check_motif(m) for m in motifs], average))


def _ints(x: int | Sequence[int], what: str) -> list[int]:
    v = [int(x)] if np.ndim(x) == 0 else [int(i) for i in x]  # type: ignore[union-attr]
    if any(i < 0 for i in v):
        raise ValueError(f"{what} must not be negative")
    return v


def add_gap(motifs, gaploc: int | Sequence[int] | None = None, mingap: int | Sequence[int] = 1,
            maxgap: int | Sequence[int] = 5):
    """Add gaps to motifs. Gap ``i`` comes after the first ``gaploc[i]``
    positions, in the middle of the motif by default, and spans
    ``mingap[i]`` to ``maxgap[i]`` positions; shorter lists are recycled.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.create_motif("TGACGTCA")
    >>> gapped = um.add_gap(motif, gaploc=4, mingap=1, maxgap=5)
    >>> gapped.isgapped, gapped.gaploc, gapped.mingap, gapped.maxgap
    (True, [4], [1], [5])
    """
    loc = None if gaploc is None else _ints(gaploc, "gaploc")
    lo, hi = _ints(mingap, "mingap"), _ints(maxgap, "maxgap")
    return _apply(motifs, lambda m: _core._add_gap(m, loc, lo, hi))


def ungap(motifs, delete: bool = False):
    """Turn motifs' gaps off. Unless ``delete``, the gap positions and sizes
    are kept, and passing them back to [`add_gap()`][universalmotif.add_gap],
    as in ``add_gap(m, gaploc=m.gaploc, mingap=m.mingap, maxgap=m.maxgap)``,
    turns the gaps on again.

    Examples
    --------
    >>> import universalmotif as um
    >>> gapped = um.add_gap(um.create_motif("TGACGTCA"), gaploc=4)
    >>> off = um.ungap(gapped)
    >>> off.isgapped, off.gaploc
    (False, [4])
    >>> um.ungap(gapped, delete=True).gaploc
    []
    """
    return _apply(motifs, lambda m: _core._ungap(m, delete))


def switch_alph(motifs):
    """Switch motifs between the DNA and RNA alphabets.

    Examples
    --------
    >>> import universalmotif as um
    >>> rna = um.switch_alph(um.create_motif("TGACGTCA"))
    >>> rna.alphabet, rna.consensus
    ('RNA', 'UGACGUCA')
    """
    return _apply(motifs, _core._switch_alph)


def _as_set(x: str | Iterable[str]) -> set[str]:
    return {x} if isinstance(x, str) else {str(v) for v in x}


def filter_motifs(
    motifs: Iterable[Motif],
    name: str | Iterable[str] | None = None,
    altname: str | Iterable[str] | None = None,
    family: str | Iterable[str] | None = None,
    organism: str | Iterable[str] | None = None,
    width: int | None = None,
    alphabet: str | Iterable[str] | None = None,
    type: str | Iterable[str] | None = None,
    icscore: float | None = None,
    nsites: float | None = None,
    strand: str | Iterable[str] | None = None,
    pval: float | None = None,
    qval: float | None = None,
    eval: float | None = None,
    extrainfo: Mapping[str, str] | None = None,
) -> list[Motif]:
    """Keep the motifs that pass every filter given.

    Text filters keep motifs whose value is one of those given. ``width``,
    ``icscore`` and ``nsites`` are minimums, and ``pval``, ``qval`` and
    ``eval`` maximums; motifs without the value are dropped. ``extrainfo``
    keeps motifs with every one of its entries.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> [m.name for m in um.filter_motifs(motifs, family="bZIP")]
    ['AP-1', 'CRE', 'G-box']
    >>> [m.name for m in um.filter_motifs(motifs, family=["bZIP", "ETS"], icscore=10)]
    ['CRE', 'ETS']
    """
    ms = [_check_motif(m) for m in motifs]
    text = {"name": name, "altname": altname, "family": family, "organism": organism,
            "alphabet": alphabet, "type": type, "strand": strand}
    for attr, want in text.items():
        if want is not None:
            keep = _as_set(want)
            ms = [m for m in ms if getattr(m, attr) is not None and getattr(m, attr) in keep]
    for attr, low in (("width", width), ("icscore", icscore), ("nsites", nsites)):
        if low is not None:
            ms = [m for m in ms if getattr(m, attr) is not None and getattr(m, attr) >= low]
    for attr, high in (("pval", pval), ("qval", qval), ("eval", eval)):
        if high is not None:
            ms = [m for m in ms if getattr(m, attr) is not None and getattr(m, attr) <= high]
    if extrainfo is not None:
        want = {str(k): str(v) for k, v in extrainfo.items()}
        ms = [m for m in ms if all(m.extrainfo.get(k) == v for k, v in want.items())]
    return ms


# Sequence utilities ---------------------------------------------------------


def _reshape(sequences, values: list[str]):
    """Return ``values`` in the shape ``sequences`` came in."""
    if isinstance(sequences, str):
        return values[0]
    if isinstance(sequences, Mapping):
        return dict(zip(sequences, values))
    return values


def mask_seqs(sequences, pattern: str, rc: bool = False, letter: str = "-", alphabet: str = "DNA"):
    """Replace every occurrence of ``pattern`` in the sequences with
    ``letter``. Overlapping occurrences are all masked and letters match in
    either case. With ``rc``, occurrences of the reverse complement of DNA
    and RNA patterns are masked as well. The sequences come back in the form
    they were given, whether a string, a list or a dict.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.mask_seqs("AAATTTGCAAATTT", "AAA")
    '---TTTGC---TTT'
    >>> um.mask_seqs({"s1": "GATTACAGG", "s2": "TTGTAATCT"}, "GATTACA", rc=True)
    {'s1': '-------GG', 's2': 'T-------T'}
    """
    if len(letter) != 1:
        raise ValueError("letter must be a single character")
    from ._analysis import _sequence_pairs

    _, seqs = _sequence_pairs(sequences)
    return _reshape(sequences, _core._mask_seqs(seqs, pattern, rc, letter, alphabet))


def mask_ranges(sequences: Mapping[str, str], ranges, letter: str = "-") -> dict[str, str]:
    """Replace the positions in ``ranges`` with ``letter``.

    ``sequences`` maps names to sequences and ``ranges`` names positions on
    them, counted from 0 with the end left out as in a slice, either as
    ``(chrom, start, end)`` tuples or
    as a DataFrame with a ``start`` column, an ``end``, ``stop`` or
    ``width`` column, and the sequence names in a column called ``chrom``,
    ``seqnames``, ``sequence``, ``chr`` or ``chromosome``, so that the hits
    from [`scan_sequences()`][universalmotif.scan_sequences] can be masked
    directly. Ranges may overlap; ranges on unknown sequences or past their
    ends are errors.

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = {"chr1": "ACGTACGTACGT", "chr2": "TTTTGGGG"}
    >>> um.mask_ranges(seqs, [("chr1", 2, 6), ("chr2", 0, 2)])
    {'chr1': 'AC----GTACGT', 'chr2': '--TTGGGG'}
    """
    if len(letter) != 1:
        raise ValueError("letter must be a single character")
    from ._genomic import _range_table

    r = _range_table(ranges, "ranges")
    unknown = set(r["chrom"]) - set(sequences)
    if unknown:
        raise ValueError(f"ranges on sequences not given: {', '.join(sorted(unknown)[:5])}")
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for c, s, e in zip(r["chrom"], r["start"], r["end"]):
        by_chrom.setdefault(c, []).append((int(s), int(e)))
    return {name: _core._mask_ranges(seq, by_chrom[name], letter) if name in by_chrom else seq
            for name, seq in sequences.items()}


def shuffle_string(string: str, k: int = 1, method: str = "euler", seed: int | None = None) -> str:
    """Shuffle one string while keeping its k-mer counts, as
    [`shuffle_sequences()`][universalmotif.shuffle_sequences] does.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.shuffle_string("ACGTACGTTTGGCCAA", k=2, seed=1)
    'ACCGTTGGCAACGTTA'
    """
    from ._sequences import shuffle_sequences

    if not isinstance(string, str):
        raise TypeError("string must be a single string")
    return shuffle_sequences([string], k=k, method=method, seed=seed)[0]


def slide_fun(string: str, fun: Callable[[str], Any], window: int = 1, overlap: int = 0,
              return_incomp: bool = True) -> list[Any]:
    """Apply ``fun`` to each sliding window of ``string``, as made by
    [`window_string()`][universalmotif.window_string]. A window longer than the
    string is shortened to it when ``return_incomp`` is set and is an error
    otherwise.

    Examples
    --------
    >>> import universalmotif as um
    >>> def gc(window):
    ...     return (window.count("G") + window.count("C")) / len(window)
    >>> um.slide_fun("ACGTTTAGCAGG", gc, window=4)
    [0.5, 0.25, 0.75]
    """
    from ._sequences import window_string

    if not string:
        raise ValueError("string must not be empty")
    if window < 1:
        raise ValueError("window should be a positive integer")
    if overlap < 0:
        raise ValueError("overlap should be a non-negative integer")
    if window > len(string):
        if not return_incomp:
            raise ValueError("string is smaller than window; set return_incomp=True to shorten the window")
        window, overlap = len(string), 0
    if overlap >= window:
        raise ValueError("overlap cannot be greater or equal to window")
    return [fun(w) for w in window_string(string, window, overlap, return_incomp)]


# Small helpers --------------------------------------------------------------


def meme_alph(
    core: str,
    file: str | os.PathLike[str] | None = None,
    complements: str | None = None,
    ambiguity: Mapping[str, str] | None = None,
    like: str | None = None,
    alph_name: str | None = None,
    letter_names: Mapping[str, str] | None = None,
    colours: Mapping[str, str] | None = None,
) -> str | None:
    """Write a MEME custom alphabet definition.

    ``core`` holds the alphabet's letters and ``complements`` their
    complements, if it has any. ``ambiguity`` maps ambiguous letters to the
    core letters they stand for, ``like`` is ``"DNA"``, ``"RNA"`` or
    ``"PROTEIN"`` for alphabets that extend one, and ``letter_names`` and
    ``colours`` describe each letter, with colours as hexadecimal RGB. The
    definition is returned as text, or written to ``file`` if one is given.

    Examples
    --------
    >>> import universalmotif as um
    >>> text = um.meme_alph("ACGT", complements="TGCA", like="DNA", alph_name="MyDNA")
    >>> print(text, end="")
    ALPHABET "MyDNA" DNA-LIKE
    <BLANKLINE>
    # Core symbols
    A ~ T
    C ~ G
    """
    def pairs(m: Mapping[str, str] | None) -> list[tuple[str, str]]:
        return [] if m is None else [(str(k), str(v)) for k, v in m.items()]

    text = _core._meme_alph(core, complements, pairs(ambiguity), like, alph_name, pairs(letter_names),
                            pairs(colours))
    if file is None:
        return text
    with open(file, "w") as f:
        f.write(text)
    return None


def log_string_pval(pval: str) -> float:
    """The natural logarithm of a P-value written as text, such as
    ``"1e-400"``, which may be far smaller than the smallest float.

    Examples
    --------
    >>> import universalmotif as um
    >>> round(um.log_string_pval("1e-400"), 3)
    -921.034
    >>> round(um.log_string_pval("0.05"), 3)
    -2.996
    """
    return _core._log_string_pval(str(pval))
