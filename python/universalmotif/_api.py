"""User-facing functions. The work is done by the compiled Rust
bindings in ``universalmotif._core``; this module handles argument checking
and lets most functions take either a single motif or a list of them."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, TypeVar, overload

import numpy as np
import polars as pl

from . import _core
from ._core import Motif

MotifType = Literal["PCM", "PPM", "PWM", "ICM", "CWM"]
TrimFrom = Literal["both", "left", "right"]

_T = TypeVar("_T")


def _check_motif(m: object) -> Motif:
    if not isinstance(m, Motif):
        raise TypeError(f"expected a Motif, got {type(m).__name__}")
    return m


@overload
def _apply(motifs: Motif, f: Callable[[Motif], _T]) -> _T: ...
@overload
def _apply(motifs: Iterable[Motif], f: Callable[[Motif], _T]) -> list[_T]: ...
def _apply(motifs, f):
    """Apply ``f`` to one motif, or to each motif of an iterable."""
    if isinstance(motifs, Motif):
        return f(motifs)
    return [f(_check_motif(m)) for m in motifs]


def create_motif(
    input: object = None,
    alphabet: str | None = None,
    type: MotifType | None = None,
    *,
    letters: Sequence[str] | None = None,
    name: str = "motif",
    pseudocount: float = 0.0,
    bkg: Sequence[float] | Mapping[str, float] | None = None,
    nsites: int | None = None,
    altname: str | None = None,
    family: str | None = None,
    organism: str | None = None,
    bkgsites: float | None = None,
    strand: str = "+-",
    pval: float | None = None,
    qval: float | None = None,
    eval: float | None = None,
    extrainfo: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> Motif:
    """Create a motif.

    Parameters
    ----------
    input
        What to build the motif from:

        - ``None`` or an integer: a random motif of that width, 10 by
          default. Each column is drawn from a Dirichlet distribution
          parameterized by the background.
        - A string: a consensus sequence. IUPAC ambiguity codes are allowed
          for DNA and amino acids.
        - A sequence of equal-length strings: aligned sites, which are
          counted. ``nsites`` then defaults to the number of strings.
        - A 2-D array of shape (letters, positions). Give the row letters
          with ``letters``, which are sorted with the rows reordered to
          match, or give ``alphabet`` and supply the rows in alphabet order.
          The type is detected from the values.
    alphabet
        ``"DNA"``, ``"RNA"``, ``"AA"``, or a string of letters for a custom
        alphabet. Guessed from the input when omitted.
    type
        Type of the returned motif, a PPM by default. A matrix with negative
        values needs its type given, as ``"PWM"`` or ``"CWM"``, since the two
        are read differently.
    letters
        Row letters of a matrix ``input``.
    name, altname, family, organism, nsites, bkgsites, strand, pval, qval, eval, extrainfo
        Motif metadata.
    pseudocount
        Pseudocount used when converting between types.
    bkg
        Background frequencies, either one value per letter in alphabet
        order or a mapping from letters to values. Uniform by default.
    seed
        Random seed, for random motifs only.

    Examples
    --------
    A motif from a consensus sequence:

    >>> import universalmotif as um
    >>> motif = um.create_motif("TATAWAW", name="TATA", nsites=10)
    >>> print(motif)
           Motif name:   TATA
             Alphabet:   DNA
                 Type:   PPM
              Strands:   +-
             Total IC:   12
          Pseudocount:   0
            Consensus:   TATAWAW
         Target sites:   10
    <BLANKLINE>
      T A T A   W A   W
    A 0 1 0 1 0.5 1 0.5
    C 0 0 0 0 0.0 0 0.0
    G 0 0 0 0 0.0 0 0.0
    T 1 0 1 0 0.5 0 0.5

    A count matrix from aligned sites:

    >>> import universalmotif as um
    >>> motif = um.create_motif(["TATAAA", "TATAAT", "TATATA"], type="PCM")
    >>> motif.nsites, motif.consensus
    (3.0, 'TATAWW')
    >>> motif.matrix.tolist()
    [[0.0, 3.0, 0.0, 3.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 3.0, 0.0, 1.0, 1.0]]
    """
    opts = dict(
        type=type or "PPM",
        name=name,
        pseudocount=pseudocount,
        bkg=bkg,
        nsites=nsites,
        altname=altname,
        family=family,
        organism=organism,
        bkgsites=bkgsites,
        strand=strand,
        pval=pval,
        qval=qval,
        eval=eval,
        extrainfo=extrainfo,
    )
    is_random = input is None or (
        isinstance(input, numbers.Integral) and not isinstance(input, bool)
    )
    if seed is not None and not is_random:
        raise TypeError("seed is only used for random motifs")
    if letters is not None and (is_random or isinstance(input, str)):
        raise TypeError("letters is only used for matrix input")

    if is_random:
        width = 10 if input is None else int(input)
        if width <= 0:
            raise ValueError("width must be greater than 0")
        return _core._create_random(width, alphabet or "DNA", seed, **opts)
    if isinstance(input, str):
        return _core._create_from_strings([input], alphabet, **opts)
    if isinstance(input, Sequence) and len(input) > 0 and all(isinstance(s, str) for s in input):
        if letters is not None:
            raise TypeError("letters is only used for matrix input")
        return _core._create_from_strings(list(input), alphabet, **opts)

    matrix = np.asarray(input, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix input must be 2-D, got {matrix.ndim} dimensions")
    if type is None and (matrix < 0).any():
        raise ValueError(
            "the matrix has negative values, so it holds log-odds scores or contribution "
            'weights; say which with type="PWM" or type="CWM"'
        )
    return _core._create_from_matrix(
        matrix, None if letters is None else list(letters), alphabet, **opts
    )


@overload
def convert_type(
    motifs: Motif, type: MotifType, pseudocount: float | None = ..., relative_entropy: bool = ...
) -> Motif: ...
@overload
def convert_type(
    motifs: Iterable[Motif],
    type: MotifType,
    pseudocount: float | None = ...,
    relative_entropy: bool = ...,
) -> list[Motif]: ...
def convert_type(motifs, type, pseudocount=None, relative_entropy=False):
    """Convert motifs to another type.

    Parameters
    ----------
    motifs
        A motif or an iterable of motifs.
    type
        ``"PCM"``, ``"PPM"``, ``"PWM"`` or ``"ICM"``. Probability-based
        motifs cannot be turned into ``"CWM"`` contribution weight matrices,
        so asking for that is an error unless the motif already is one.
    pseudocount
        Replaces each motif's own pseudocount.
    relative_entropy
        For ICMs, compute heights as relative entropy against the background
        instead of Shannon information content.

    Notes
    -----
    A motif without ``nsites`` is treated as having 100 sites when counts are
    needed. The consensus and total IC are kept from the input motif.

    Examples
    --------
    >>> import universalmotif as um
    >>> ap1 = um.example_motifs()[1]
    >>> ap1.type, ap1.consensus
    ('PCM', 'NTGASTCAN')
    >>> ppm = um.convert_type(ap1, "PPM")
    >>> ppm.matrix[:, 1].round(3).tolist()
    [0.065, 0.025, 0.005, 0.905]
    >>> [m.type for m in um.convert_type(um.example_motifs()[:3], "PWM")]
    ['PWM', 'PWM', 'PWM']
    """
    return _apply(
        motifs, lambda m: _core._convert_type(m, type, pseudocount, relative_entropy)
    )


def motif_rc(motifs, ignore_alphabet: bool = False):
    """Reverse complement DNA or RNA motifs.

    With ``ignore_alphabet=True`` other alphabets are accepted, and their
    rows and columns are simply reversed.

    Examples
    --------
    >>> import universalmotif as um
    >>> wbox = um.example_motifs()[5]
    >>> wbox.consensus, um.motif_rc(wbox).consensus
    ('GBTTGACYS', 'SRGTCAAVC')
    """
    return _apply(motifs, lambda m: _core._motif_rc(m, ignore_alphabet))


def _drop_trimmed(motifs, f: Callable[[Motif], Motif | None]):
    if isinstance(motifs, Motif):
        out = f(motifs)
        if out is None:
            raise ValueError(f"motif [{motifs.name}] was completely trimmed")
        return out
    motifs = [_check_motif(m) for m in motifs]
    results = [f(m) for m in motifs]
    gone = [m.name for m, r in zip(motifs, results) if r is None]
    if motifs and len(gone) == len(motifs):
        raise ValueError("all motifs were completely trimmed")
    if gone:
        warnings.warn(
            "the following motifs were completely trimmed: " + ", ".join(gone), stacklevel=3
        )
    return [r for r in results if r is not None]


def trim_motifs(motifs, min_ic: float = 0.25, trim_from: TrimFrom = "both"):
    """Trim low information content columns from the edges of motifs.

    Columns with an information content below ``min_ic`` are removed from the
    chosen ends until a column at or above it is reached. Motifs that are
    trimmed away completely are dropped from a list with a warning; a single
    motif that is trimmed away raises ``ValueError``.

    Examples
    --------
    >>> import universalmotif as um
    >>> ap1 = um.example_motifs()[1]
    >>> ap1.consensus, um.trim_motifs(ap1).consensus
    ('NTGASTCAN', 'TGASTCA')
    >>> um.trim_motifs(ap1, trim_from="left").consensus
    'TGASTCAN'
    """
    return _drop_trimmed(motifs, lambda m: _core._trim_motif(m, min_ic, trim_from))


def trim_cwm(
    motifs,
    trim_threshold: float = 0.3,
    abs_threshold: float | None = None,
    trim_from: TrimFrom = "both",
):
    """Trim the edges of contribution weight matrices.

    A column is kept if the sum of its absolute values reaches the cutoff,
    which is ``abs_threshold`` if given and otherwise ``trim_threshold``
    times the largest column sum. Everything between the first and last kept
    columns is retained.

    Examples
    --------
    >>> import universalmotif as um
    >>> cwm = um.create_motif(
    ...     [[0.01, 0.0, 0.6, 0.0, 0.02],
    ...      [0.0, 0.5, 0.0, 0.0, 0.0],
    ...      [0.02, 0.0, 0.0, 0.7, 0.0],
    ...      [0.0, 0.0, -0.1, 0.0, 0.01]],
    ...     alphabet="DNA",
    ...     type="CWM",
    ... )
    >>> trimmed = um.trim_cwm(cwm)
    >>> trimmed.width
    3
    >>> trimmed.matrix.tolist()
    [[0.0, 0.6, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, -0.1, 0.0]]
    """
    return _drop_trimmed(
        motifs, lambda m: _core._trim_cwm(m, trim_threshold, abs_threshold, trim_from)
    )


def round_motif(motifs, pct_tolerance: float | str = 0.05):
    """Set small probabilities to zero.

    Probabilities below ``pct_tolerance``, given as a fraction or as a
    string such as ``"5%"``, are set to zero and each column is
    renormalized. The motif is
    returned in its original type.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.create_motif(
    ...     [[0.9, 0.02], [0.04, 0.48], [0.03, 0.48], [0.03, 0.02]], alphabet="DNA"
    ... )
    >>> um.round_motif(motif).matrix.tolist()
    [[1.0, 0.0], [0.0, 0.5], [0.0, 0.5], [0.0, 0.0]]
    >>> um.round_motif(motif, "3%").matrix.tolist()
    [[0.9, 0.0], [0.04, 0.5], [0.03, 0.5], [0.03, 0.0]]
    """
    if isinstance(pct_tolerance, str):
        pct_tolerance = float(pct_tolerance.strip().rstrip("%")) / 100
    return _apply(motifs, lambda m: _core._round_motif(m, pct_tolerance))


def normalize(motifs):
    """Re-apply each motif's pseudocount by converting to counts and back.

    A pseudocount of zero is replaced by 1. PCMs keep their counts and only
    take the new pseudocount, which applies when they are converted.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.create_motif(["TGACGTCA"] * 10)
    >>> motif.pseudocount, motif.matrix[:, 0].tolist()
    (0.0, [0.0, 0.0, 0.0, 1.0])
    >>> normalized = um.normalize(motif)
    >>> normalized.pseudocount, normalized.matrix[:, 0].round(3).tolist()
    (1.0, [0.023, 0.023, 0.023, 0.932])
    """
    return _apply(motifs, _core._normalize)


def position_icscore(
    position: Sequence[float],
    bkg: Sequence[float] | None = None,
    type: MotifType = "PPM",
    pseudocount: float = 1.0,
    nsites: float = 100.0,
    relative_entropy: bool = False,
) -> float:
    """Information content of a single motif column.

    PPM columns are rounded to ``nsites`` counts and given the pseudocount
    before their information content is taken, as they are for a motif's
    total IC, so with the defaults a column of pure A gives about 1.93 bits
    rather than 2; set ``pseudocount=0`` for the information content of the
    column as it is. PCM columns get the pseudocount directly.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.position_icscore([0.25, 0.25, 0.25, 0.25])
    0.0
    >>> um.position_icscore([1.0, 0.0, 0.0, 0.0], pseudocount=0)
    2.0
    >>> round(um.position_icscore([0.5, 0.0, 0.0, 0.5], pseudocount=0), 3)
    1.0
    """
    return _core._position_icscore(
        [float(x) for x in position],
        [] if bkg is None else [float(x) for x in bkg],
        type,
        pseudocount,
        nsites,
        relative_entropy,
    )


def get_consensus(
    position: Sequence[float],
    alphabet: Literal["DNA", "RNA"] = "DNA",
    type: MotifType = "PPM",
    pseudocount: float = 1.0,
) -> str:
    """IUPAC consensus letter of a DNA or RNA column, with its values in A,
    C, G and T or U order.

    Examples
    --------
    >>> import universalmotif as um
    >>> columns = [[0.9, 0.05, 0.05, 0.0], [0.5, 0.0, 0.5, 0.0], [0.25, 0.25, 0.25, 0.25]]
    >>> [um.get_consensus(col) for col in columns]
    ['A', 'R', 'N']
    >>> um.get_consensus([1, 0, 1, 18], alphabet="RNA", type="PCM")
    'U'
    """
    if alphabet not in ("DNA", "RNA"):
        raise ValueError("alphabet must be one of DNA, RNA")
    return _core._get_consensus(
        [float(x) for x in position], alphabet == "RNA", type, pseudocount
    )


def get_consensus_aa(
    position: Sequence[float], type: MotifType = "PPM", pseudocount: float = 0.0
) -> str:
    """Consensus letter of an amino acid column, including B, Z, J and X.

    Examples
    --------
    The column is in alphabetical order of the one-letter codes, so here
    D and N share the probability:

    >>> import universalmotif as um
    >>> col = [0.0] * 20
    >>> col[2] = col[11] = 0.5
    >>> um.get_consensus_aa(col)
    'B'
    """
    return _core._get_consensus_aa([float(x) for x in position], type, pseudocount)


def consensus_to_ppm(letter: str) -> np.ndarray:
    """The DNA probability column for an IUPAC letter. A single base gets
    0.997, leaving 0.001 for each of the others, unlike a motif made by
    [`create_motif()`][universalmotif.create_motif] from a consensus string,
    whose columns hold exact zeros.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.consensus_to_ppm("W").tolist()
    [0.499, 0.001, 0.001, 0.499]
    >>> um.consensus_to_ppm("N").tolist()
    [0.25, 0.25, 0.25, 0.25]
    """
    return np.array(_core._consensus_to_ppm(letter))


def consensus_to_ppm_aa(letter: str) -> np.ndarray:
    """The amino acid probability column for a consensus letter.

    Examples
    --------
    >>> import universalmotif as um
    >>> col = um.consensus_to_ppm_aa("B")
    >>> letters = "ACDEFGHIKLMNPQRSTVWY"
    >>> {a: round(float(p), 3) for a, p in zip(letters, col) if p > 0.1}
    {'D': 0.491, 'N': 0.491}
    """
    return np.array(_core._consensus_to_ppm_aa(letter))


_SUMMARY_SCHEMA = {
    "name": pl.String,
    "altname": pl.String,
    "family": pl.String,
    "organism": pl.String,
    "consensus": pl.String,
    "alphabet": pl.String,
    "strand": pl.String,
    "icscore": pl.Float64,
    "nsites": pl.Float64,
    "bkgsites": pl.Float64,
    "pval": pl.Float64,
    "qval": pl.Float64,
    "eval": pl.Float64,
}


def summarize_motifs(motifs, na_rm: bool = True) -> pl.DataFrame:
    """Summarize motifs as a polars DataFrame, one row per motif.

    With ``na_rm=True``, columns that are empty for every motif are dropped.

    Examples
    --------
    >>> import universalmotif as um
    >>> summary = um.summarize_motifs(um.example_motifs())
    >>> summary.shape
    (12, 8)
    >>> summary.columns
    ['name', 'family', 'organism', 'consensus', 'alphabet', 'strand', 'icscore', 'nsites']
    >>> summary["consensus"].to_list()[:4]
    ['KTATAWAWRN', 'NTGASTCAN', 'TGACGTCA', 'NCACGTGN']
    >>> [round(x, 2) for x in summary["icscore"].to_list()[:4]]
    [8.83, 9.65, 10.73, 7.9]
    """
    if isinstance(motifs, Motif):
        motifs = [motifs]
    motifs = [_check_motif(m) for m in motifs]
    df = pl.DataFrame(_core._summarize(motifs), schema=_SUMMARY_SCHEMA)
    if na_rm and df.height > 0:
        df = df.select(c for c in df.columns if df[c].null_count() < df.height)
    return df
