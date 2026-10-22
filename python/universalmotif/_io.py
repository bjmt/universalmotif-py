"""Reading and writing motif files. Parsing and formatting happen in the Rust crate;
this module deals with files, line skipping and appending."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from . import _core
from ._api import MotifType, _check_motif
from ._core import Motif

Positions = Literal["columns", "rows"]
PathLike = str | os.PathLike[str]


def _read_text(file: PathLike, skip: int) -> str:
    text = Path(file).read_text()
    if skip:
        if skip < 0:
            raise ValueError("skip must be zero or positive")
        text = "".join(text.splitlines(keepends=True)[skip:])
    return text


def _motif_list(motifs: Motif | Iterable[Motif]) -> list[Motif]:
    if isinstance(motifs, Motif):
        return [motifs]
    out = [_check_motif(m) for m in motifs]
    if not out:
        raise ValueError("no motifs to write")
    return out


def _warn_gaps(motifs: list[Motif], fmt: str) -> None:
    if any(m.isgapped for m in motifs):
        warnings.warn(f"{fmt} format cannot preserve gap definitions", stacklevel=3)


def _write_text(file: PathLike, text: str, overwrite: bool, append: bool) -> None:
    path = Path(file)
    if path.exists() and not overwrite and not append:
        raise FileExistsError(f"{path} exists; set overwrite=True or append=True")
    with open(path, "a" if append else "w") as f:
        f.write(text)


def read_jaspar(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read JASPAR count matrices.

    Rows may be labelled (``A [ 1 2 3 ]``) or bare numbers, which are read as
    DNA (four rows) or amino acids (twenty rows). Motifs are returned as PCMs
    with ``nsites`` set to the largest column sum. ``skip`` drops lines from
    the start of the file.
    """
    return _core._read_jaspar(_read_text(file, skip))


def read_homer(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read HOMER motifs.

    The log-odds detection threshold is kept in ``extrainfo`` as
    ``"logodds.threshold"``, and site counts and P-values are read from
    HOMER's statistics column when present.
    """
    return _core._read_homer(_read_text(file, skip))


def read_transfac(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read TRANSFAC matrices, returned as PCMs.

    The accession (``AC``) is used as the name if there is one, with the
    identifier (``ID``) or factor name (``NA``) as the alternate name; ``HC``
    and ``OS`` give the family and organism.
    """
    return _core._read_transfac(_read_text(file, skip))


def read_cisbp(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read CIS-BP motifs, using the ``TF``, ``TF Name``, ``Family`` and
    ``Species`` fields as the name, alternate name, family and organism."""
    return _core._read_cisbp(_read_text(file, skip))


def read_uniprobe(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read UniPROBE motifs from full output (keeping the enrichment score in
    ``extrainfo``) or from minimal files."""
    return _core._read_uniprobe(_read_text(file, skip))


def read_meme(
    file: PathLike,
    skip: int = 0,
    readsites: bool = False,
    readsites_meta: bool = False,
    cwm: bool = False,
) -> list[Motif] | dict[str, object]:
    """Read MEME motifs, from MEME, DREME or STREME output or minimal MEME
    files.

    Parameters
    ----------
    readsites
        Also read the sites listed in the BLOCKS sections of MEME output.
    readsites_meta
        Also read the per-site P-value tables and the combined block
        diagrams of MEME output.
    cwm
        Read the matrices as contribution weight matrices (as written by
        TF-MoDISco) instead of probabilities.

    Returns
    -------
    A list of motifs, or with ``readsites`` or ``readsites_meta`` a dict with
    the key ``"motifs"`` plus:

    - ``"sites"``: a dict from motif name to a DataFrame of ``sequence`` and
      ``site``.
    - ``"sites_meta"``: one DataFrame of ``motif``, ``sequence``, ``strand``,
      ``position``, ``pvalue`` and ``site``.
    - ``"sites_meta_combined"``: a DataFrame of ``sequence``, ``pvalue`` and
      ``diagram``.
    """
    text = _read_text(file, skip)
    motifs = _core._read_meme(text, cwm)
    if not (readsites or readsites_meta):
        return motifs
    out: dict[str, object] = {"motifs": motifs}
    if readsites:
        out["sites"] = {
            name: pl.DataFrame(
                {"sequence": [s[0] for s in rows], "site": [s[1] for s in rows]},
                schema={"sequence": pl.String, "site": pl.String},
            )
            for name, rows in _core._read_meme_sites(text)
        }
    if readsites_meta:
        rows = [
            (name, *r) for name, table in _core._read_meme_site_pvalues(text) for r in table
        ]
        out["sites_meta"] = pl.DataFrame(
            rows,
            schema={
                "motif": pl.String,
                "sequence": pl.String,
                "strand": pl.String,
                "position": pl.Int64,
                "pvalue": pl.Float64,
                "site": pl.String,
            },
            orient="row",
        )
        out["sites_meta_combined"] = pl.DataFrame(
            _core._read_meme_combined(text),
            schema={"sequence": pl.String, "pvalue": pl.Float64, "diagram": pl.String},
            orient="row",
        )
    return out


def read_matrix(
    file: PathLike,
    skip: int = 0,
    type: MotifType | None = None,
    positions: Positions = "columns",
    alphabet: str = "DNA",
    sep: str | None = "",
    headers: bool | str = True,
    rownames: bool = False,
    comment: str | None = None,
) -> list[Motif]:
    """Read motifs stored as plain matrices of numbers.

    Parameters
    ----------
    type
        The matrix type; detected from the values by default.
    positions
        Whether positions are ``"columns"`` or ``"rows"``.
    alphabet
        Alphabet of the motifs; rows (or columns) are read in its order.
    sep
        Line separating motifs; ``""`` means blank lines and ``None`` means
        motifs are only separated by headers.
    headers
        ``True`` if the first line of each motif is its name, ``False`` for
        no names, or a string marking header lines (each header starts a new
        motif, and the name is the line with the string removed).
    rownames
        Whether the values come with letter labels, which are skipped.
    comment
        Lines starting with this string are ignored.
    """
    return _core._read_matrix(
        _read_text(file, skip), positions, alphabet, type, sep, headers, rownames, comment
    )


def write_meme(
    motifs: Motif | Iterable[Motif],
    file: PathLike,
    version: int = 5,
    bkg: Sequence[float] | None = None,
    strand: str | None = None,
    overwrite: bool = False,
    append: bool = False,
    cwm: bool = False,
) -> None:
    """Write motifs in MEME format.

    Motifs are converted to PPMs, unless ``cwm=True``, in which case every
    motif must be a CWM and is written as is. ``bkg`` gives the background in
    the order letters are written (A, C, G, T for DNA); by default the
    motifs' shared background is used, or a uniform one if they differ.
    ``strand`` overrides the ``strands:`` line. When appending to an existing
    file only the motifs are written. Equal signs in names become periods and
    spaces become dashes.
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "MEME")
    header = not (append and Path(file).exists())
    text = _core._write_meme(
        ms, version, None if bkg is None else [float(b) for b in bkg], strand, cwm, header
    )
    _write_text(file, text, overwrite, append)


def write_jaspar(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA, RNA or amino acid motifs as JASPAR count matrices."""
    ms = _motif_list(motifs)
    _warn_gaps(ms, "JASPAR")
    _write_text(file, _core._write_jaspar(ms), overwrite, append)


def write_homer(
    motifs: Motif | Iterable[Motif],
    file: PathLike,
    threshold: float | Sequence[float] = 0.8,
    threshold_type: Literal["logodds", "logodds_abs", "pvalue"] = "logodds",
    overwrite: bool = False,
    append: bool = False,
) -> None:
    """Write DNA or RNA motifs in HOMER format.

    The detection threshold in each header is on HOMER's own score scale:
    HOMER scores a site as the sum of ``ln(p / 0.25)`` over the probabilities
    written in the file, floored at 0.001. With ``threshold_type="logodds"``
    it is ``threshold`` times the motif's maximum score, with
    ``"logodds_abs"`` it is ``threshold`` itself, and with ``"pvalue"`` it is
    a score random sites (from a uniform background) reach with at most that
    probability. Give one value or one per motif.
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "HOMER")
    t = [float(threshold)] if isinstance(threshold, (int, float)) else [float(x) for x in threshold]
    _write_text(file, _core._write_homer(ms, t, threshold_type), overwrite, append)


def write_transfac(
    motifs: Motif | Iterable[Motif],
    file: PathLike,
    overwrite: bool = False,
    append: bool = False,
    name_tag: str = "ID",
    altname_tag: str = "NA",
) -> None:
    """Write DNA or RNA motifs as TRANSFAC count matrices. ``name_tag`` and
    ``altname_tag`` choose the fields used for the name and alternate name."""
    ms = _motif_list(motifs)
    _warn_gaps(ms, "TRANSFAC")
    _write_text(file, _core._write_transfac(ms, name_tag, altname_tag), overwrite, append)


def write_cisbp(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA or RNA motifs in CIS-BP format, as probability matrices.

    The name, alternate name, family and organism go in the ``TF``,
    ``TF Name``, ``Family`` and ``Species`` fields. The R package has no
    CIS-BP writer; files written here can be read by its ``read_cisbp()``.
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "CIS-BP")
    _write_text(file, _core._write_cisbp(ms), overwrite, append)


def write_uniprobe(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA motifs in the minimal UniPROBE format: a name line followed
    by ``A:``, ``C:``, ``G:`` and ``T:`` rows of probabilities.

    The R package has no UniPROBE writer; files written here can be read by
    its ``read_uniprobe()``.
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "UniPROBE")
    _write_text(file, _core._write_uniprobe(ms), overwrite, append)


def read_fasta(file: PathLike, skip: int = 0) -> dict[str, str]:
    """Read a FASTA file into a dict from names (the whole header line after
    ``>``) to sequences. The file is read line by line, so a genome needs
    little more memory than its sequences."""
    if skip < 0:
        raise ValueError("skip must be zero or positive")
    return dict(_core._read_fasta_path(os.fspath(file), skip))


def write_fasta(
    sequences: Mapping[str, str] | Sequence[str],
    file: PathLike,
    width: int = 80,
    overwrite: bool = False,
    append: bool = False,
) -> None:
    """Write sequences to a FASTA file, wrapping lines at ``width``
    characters (0 for no wrapping). Unnamed sequences are called ``seq1``,
    ``seq2`` and so on."""
    if isinstance(sequences, Mapping):
        records = [(str(k), str(v)) for k, v in sequences.items()]
    else:
        records = [(f"seq{i + 1}", str(s)) for i, s in enumerate(sequences)]
    _write_text(file, _core._write_fasta(records, width), overwrite, append)


def write_matrix(
    motifs: Motif | Iterable[Motif],
    file: PathLike,
    positions: Positions = "columns",
    rownames: bool = False,
    type: MotifType | None = None,
    sep: str | None = "",
    headers: bool | str = True,
    overwrite: bool = False,
    append: bool = False,
    digits: int = 6,
) -> None:
    """Write motifs as plain matrices of numbers.

    ``headers`` writes the name on its own line (``True``), prefixed with a
    string, or not at all (``False``). ``sep`` is written after each motif
    (``None`` for nothing), ``type`` converts the motifs first, and
    ``digits`` sets the number of decimal places.
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "Raw matrix")
    text = _core._write_matrix(ms, positions, rownames, type, sep, headers, digits)
    _write_text(file, text, overwrite, append)


__all__ = [
    "read_cisbp",
    "read_fasta",
    "read_homer",
    "read_jaspar",
    "read_matrix",
    "read_meme",
    "read_transfac",
    "read_uniprobe",
    "write_cisbp",
    "write_fasta",
    "write_homer",
    "write_jaspar",
    "write_matrix",
    "write_meme",
    "write_transfac",
    "write_uniprobe",
]
