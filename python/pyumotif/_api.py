"""User-facing functions. The work is done by the compiled umotif
bindings in ``pyumotif._umotif``; this module handles argument checking
and lets most functions take either a single motif or a list of them."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, TypeVar, overload

import numpy as np
import polars as pl

from . import _umotif
from ._umotif import Motif

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
    type: MotifType = "PPM",
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

        - ``None`` or an integer: a random motif of that width (10 by
          default). Each column is drawn from a Dirichlet distribution
          parameterised by the background.
        - A string: a consensus sequence. IUPAC ambiguity codes are allowed
          for DNA and amino acids.
        - A sequence of equal-length strings: aligned sites, which are
          counted. ``nsites`` then defaults to the number of strings.
        - A 2-D array of shape (letters, positions). Give the row letters
          with ``letters`` (they are sorted, with the rows reordered to
          match), or give ``alphabet`` and supply the rows in alphabet order.
          The type is detected from the values.
    alphabet
        ``"DNA"``, ``"RNA"``, ``"AA"``, or a string of letters for a custom
        alphabet. Guessed from the input when omitted.
    type
        Type of the returned motif.
    letters
        Row letters of a matrix ``input``.
    name, altname, family, organism, nsites, bkgsites, strand, pval, qval, eval, extrainfo
        Motif metadata.
    pseudocount
        Pseudocount used when converting between types.
    bkg
        Background frequencies, either one value per letter (in alphabet
        order) or a mapping from letters to values. Uniform by default.
    seed
        Random seed, for random motifs only.
    """
    opts = dict(
        type=type,
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
        return _umotif._create_random(width, alphabet or "DNA", seed, **opts)
    if isinstance(input, str):
        return _umotif._create_from_strings([input], alphabet, **opts)
    if isinstance(input, Sequence) and len(input) > 0 and all(isinstance(s, str) for s in input):
        if letters is not None:
            raise TypeError("letters is only used for matrix input")
        return _umotif._create_from_strings(list(input), alphabet, **opts)

    matrix = np.asarray(input, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix input must be 2-D, got {matrix.ndim} dimensions")
    return _umotif._create_from_matrix(
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
        motifs cannot be turned into contribution weight matrices
        (``"CWM"``), so asking for that is an error unless the motif already
        is one.
    pseudocount
        Overrides (and replaces) each motif's pseudocount.
    relative_entropy
        For ICMs, compute heights as relative entropy against the background
        instead of Shannon information content.

    Notes
    -----
    A motif without ``nsites`` is treated as having 100 sites when counts are
    needed. As in universalmotif, the consensus and total IC are kept from
    the input motif.
    """
    return _apply(
        motifs, lambda m: _umotif._convert_type(m, type, pseudocount, relative_entropy)
    )


def motif_rc(motifs, ignore_alphabet: bool = False):
    """Reverse complement DNA or RNA motifs.

    With ``ignore_alphabet=True`` other alphabets are accepted, and their
    rows and columns are simply reversed.
    """
    return _apply(motifs, lambda m: _umotif._motif_rc(m, ignore_alphabet))


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
    """
    return _drop_trimmed(motifs, lambda m: _umotif._trim_motif(m, min_ic, trim_from))


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
    """
    return _drop_trimmed(
        motifs, lambda m: _umotif._trim_cwm(m, trim_threshold, abs_threshold, trim_from)
    )


def round_motif(motifs, pct_tolerance: float | str = 0.05):
    """Set small probabilities to zero.

    Probabilities below ``pct_tolerance`` (a fraction, or a string such as
    ``"5%"``) are set to zero and each column is renormalised. The motif is
    returned in its original type.
    """
    if isinstance(pct_tolerance, str):
        pct_tolerance = float(pct_tolerance.strip().rstrip("%")) / 100
    return _apply(motifs, lambda m: _umotif._round_motif(m, pct_tolerance))


def normalize(motifs):
    """Re-apply each motif's pseudocount by converting to counts and back.

    A pseudocount of zero is replaced by 1. PCMs are returned unchanged.
    """
    return _apply(motifs, _umotif._normalize)


def position_icscore(
    position: Sequence[float],
    bkg: Sequence[float] | None = None,
    type: MotifType = "PPM",
    pseudocount: float = 1.0,
    nsites: float = 100.0,
    relative_entropy: bool = False,
) -> float:
    """Information content of a single motif column."""
    return _umotif._position_icscore(
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
    """IUPAC consensus letter of a DNA or RNA column (A, C, G, T/U order)."""
    if alphabet not in ("DNA", "RNA"):
        raise ValueError("alphabet must be one of DNA, RNA")
    return _umotif._get_consensus(
        [float(x) for x in position], alphabet == "RNA", type, pseudocount
    )


def get_consensus_aa(
    position: Sequence[float], type: MotifType = "PPM", pseudocount: float = 0.0
) -> str:
    """Consensus letter of an amino acid column, including B, Z, J and X."""
    return _umotif._get_consensus_aa([float(x) for x in position], type, pseudocount)


def consensus_to_ppm(letter: str) -> np.ndarray:
    """The DNA probability column for an IUPAC letter."""
    return np.array(_umotif._consensus_to_ppm(letter))


def consensus_to_ppm_aa(letter: str) -> np.ndarray:
    """The amino acid probability column for a consensus letter."""
    return np.array(_umotif._consensus_to_ppm_aa(letter))


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


def summarise_motifs(motifs, na_rm: bool = True) -> pl.DataFrame:
    """Summarise motifs as a polars DataFrame, one row per motif.

    With ``na_rm=True``, columns that are empty for every motif are dropped.
    """
    if isinstance(motifs, Motif):
        motifs = [motifs]
    motifs = [_check_motif(m) for m in motifs]
    df = pl.DataFrame(_umotif._summarise(motifs), schema=_SUMMARY_SCHEMA)
    if na_rm and df.height > 0:
        df = df.select(c for c in df.columns if df[c].null_count() < df.height)
    return df
