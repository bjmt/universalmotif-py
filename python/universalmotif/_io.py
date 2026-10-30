"""Reading and writing motif files. Parsing and formatting happen in the Rust crate;
this module deals with files, line skipping and appending."""

from __future__ import annotations

import gzip
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from . import _core
from ._api import MotifType, _check_motif
from ._core import Motif
from ._packed import PackedSequences

Positions = Literal["columns", "rows"]
PathLike = str | os.PathLike[str]


def _read_text(file: PathLike, skip: int) -> str:
    raw = Path(file).read_bytes()
    # Gzipped files are recognized from their first bytes.
    text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode()
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
    # A .gz path is written gzipped; appending adds a gzip member, which
    # readers treat as a continuation of the file.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "at" if append else "wt") as f:
        # Written a slice at a time, so the text is never encoded whole.
        step = 1 << 20
        for i in range(0, len(text), step):
            f.write(text[i:i + step])


def read_jaspar(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read JASPAR count matrices.

    Rows may be labelled, as in ``A [ 1 2 3 ]``, or bare numbers, which are
    read as DNA when there are four rows and as amino acids when there are
    twenty. Motifs are returned as PCMs
    with ``nsites`` set to the largest column sum. ``skip`` drops lines from
    the start of the file.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_jaspar(um.example_file("jaspar"))
    >>> [m.name for m in motifs]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    >>> motifs[0].type, motifs[0].nsites, motifs[0].consensus
    ('PCM', 23.0, 'KTATAWAWRN')
    """
    return _core._read_jaspar(_read_text(file, skip))


def read_homer(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read HOMER motifs.

    The log-odds detection threshold is kept in ``extrainfo`` as
    ``"logodds.threshold"``, and site counts and P-values are read from
    HOMER's statistics column when present.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_homer(um.example_file("homer"))
    >>> [(m.name, m.type) for m in motifs[:2]]
    [('TATA-box', 'PPM'), ('AP-1', 'PPM')]
    >>> round(float(motifs[0].extrainfo["logodds.threshold"]), 2)
    7.46
    """
    return _core._read_homer(_read_text(file, skip))


def read_transfac(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read TRANSFAC matrices, returned as PCMs.

    The accession in ``AC`` is used as the name if there is one, with the
    identifier in ``ID`` or the factor name in ``NA`` as the alternate name;
    ``HC`` and ``OS`` give the family and organism.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_transfac(um.example_file("transfac"))
    >>> [(m.name, m.family) for m in motifs]
    [('TATA-box', 'TBP'), ('AP-1', 'bZIP'), ('CRE', 'bZIP'), ('G-box', 'bZIP')]
    >>> motifs[0].type, motifs[0].organism
    ('PCM', 'synthetic')
    """
    return _core._read_transfac(_read_text(file, skip))


def read_cisbp(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read CIS-BP motifs, using the ``TF``, ``TF Name``, ``Family`` and
    ``Species`` fields as the name, alternate name, family and organism.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_cisbp(um.example_file("cisbp"))
    >>> [m.name for m in motifs]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    >>> motifs[0].type, motifs[0].family, motifs[0].organism
    ('PPM', 'TBP', 'synthetic')
    """
    return _core._read_cisbp(_read_text(file, skip))


def read_uniprobe(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read UniPROBE motifs from minimal files or from full output, whose
    enrichment scores are kept in ``extrainfo``.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_uniprobe(um.example_file("uniprobe"))
    >>> [(m.name, m.consensus) for m in motifs[:2]]
    [('TATA-box', 'KTATAWAWRN'), ('AP-1', 'NTGASTCAN')]
    """
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

    The ``strands:`` line gives every motif its ``strand``, which scanning
    respects. Without it, DNA and RNA motifs are for both strands and those
    of other alphabets for one.

    Parameters
    ----------
    readsites
        Also read the sites listed in the BLOCKS sections of MEME output.
    readsites_meta
        Also read the per-site P-value tables and the combined block
        diagrams of MEME output.
    cwm
        Read the matrices as contribution weight matrices, as TF-MoDISco
        writes them, instead of probabilities.

    Returns
    -------
    list[Motif] | dict
        A list of motifs, or with ``readsites`` or ``readsites_meta`` a dict with
        the key ``"motifs"`` plus:

        - ``"sites"``: a dict from motif name to a DataFrame of ``sequence``
            and ``site``.
        - ``"sites_meta"``: one DataFrame of ``motif``, ``sequence``,
            ``strand``, ``position``, ``pvalue`` and ``site``.
        - ``"sites_meta_combined"``: a DataFrame of ``sequence``, ``pvalue``
            and ``diagram``.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.read_meme(um.example_file("meme"))
    >>> [m.name for m in motifs]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    >>> motifs[1].type, motifs[1].nsites, motifs[1].consensus
    ('PPM', 49.0, 'NTGASTCAN')
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
        Alphabet of the motifs; the letters' rows or columns are read in its
        order.
    sep
        Line separating motifs; ``""`` means blank lines and ``None`` means
        motifs are only separated by headers.
    headers
        ``True`` if the first line of each motif is its name, ``False`` for
        no names, or a string marking header lines. Each header line starts
        a new motif, and the name is the line with the string removed.
    rownames
        Whether the values come with letter labels, which are skipped.
    comment
        Lines starting with this string are ignored.

    Examples
    --------
    The example file has each motif's name on its own line, followed by
    one row of counts per letter:

    >>> import universalmotif as um
    >>> motifs = um.read_matrix(um.example_file("matrix"))
    >>> [m.name for m in motifs]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    >>> motifs[0].type, motifs[0].consensus
    ('PCM', 'KTATAWAWRN')
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
    the order letters are written, which for DNA is A, C, G, T; by default the
    motifs' shared background is used, or a uniform one if they differ.
    ``strand`` overrides the ``strands:`` line. When appending to an existing
    file only the motifs are written. Equal signs in names become periods and
    spaces become dashes.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.meme"
    ...     um.write_meme(motifs, path)
    ...     lines = path.read_text().splitlines()
    ...     back = um.read_meme(path)
    >>> lines[:3]
    ['MEME version 5', '', 'ALPHABET= ACGT']
    >>> [(m.name, m.type) for m in back]
    [('TATA-box', 'PPM'), ('AP-1', 'PPM')]
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
    """Write DNA, RNA or amino acid motifs as JASPAR count matrices.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.jaspar"
    ...     um.write_jaspar(motifs, path)
    ...     for line in path.read_text().splitlines()[:5]:
    ...         print(line)
    >TATA-box
    A [  2  2 23  1 18 11 20  8 10  7 ]
    C [  1  0  0  1  3  1  3  1  0  3 ]
    G [ 11  0  0  1  1  1  0  1 13  5 ]
    T [  9 21  0 20  1 10  0 13  0  8 ]
    """
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
    a score that random sites from a uniform background reach with at most
    that probability. Give one value or one per motif.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.motif"
    ...     um.write_homer(motifs, path, threshold=0.9)
    ...     back = um.read_homer(path)
    >>> [(m.name, round(float(m.extrainfo["logodds.threshold"]), 2)) for m in back]
    [('TATA-box', 8.39), ('AP-1', 8.34)]
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
    ``altname_tag`` choose the fields used for the name and alternate name.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.transfac"
    ...     um.write_transfac(motifs, path)
    ...     for line in path.read_text().splitlines()[:3]:
    ...         print(line)
    ...     back = um.read_transfac(path)
    ID TATA-box
    HC TBP
    OS synthetic
    >>> [(m.name, m.type, m.nsites) for m in back]
    [('TATA-box', 'PCM', 23.0), ('AP-1', 'PCM', 49.0)]
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "TRANSFAC")
    _write_text(file, _core._write_transfac(ms, name_tag, altname_tag), overwrite, append)


def write_cisbp(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA or RNA motifs in CIS-BP format, as probability matrices.

    The name, alternate name, family and organism go in the ``TF``,
    ``TF Name``, ``Family`` and ``Species`` fields. Files written here can be
    read back with [`read_cisbp()`][universalmotif.read_cisbp].

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.txt"
    ...     um.write_cisbp(motifs, path)
    ...     back = um.read_cisbp(path)
    >>> [(m.name, m.type, m.family) for m in back]
    [('TATA-box', 'PPM', 'TBP'), ('AP-1', 'PPM', 'bZIP')]
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "CIS-BP")
    _write_text(file, _core._write_cisbp(ms), overwrite, append)


def write_uniprobe(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA motifs in the minimal UniPROBE format: a name line followed
    by ``A:``, ``C:``, ``G:`` and ``T:`` rows of probabilities.

    Files written here can be read back with
    [`read_uniprobe()`][universalmotif.read_uniprobe].

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.txt"
    ...     um.write_uniprobe(motifs, path)
    ...     back = um.read_uniprobe(path)
    >>> [(m.name, m.type, m.consensus) for m in back]
    [('TATA-box', 'PPM', 'KTATAWAWRN'), ('AP-1', 'PPM', 'NTGASTCAN')]
    """
    ms = _motif_list(motifs)
    _warn_gaps(ms, "UniPROBE")
    _write_text(file, _core._write_uniprobe(ms), overwrite, append)


def read_fasta(
    file: PathLike,
    skip: int = 0,
    packed: bool = False,
    alphabet: str = "DNA",
    trim_names: bool = True,
) -> dict[str, str] | PackedSequences:
    """Read a FASTA file into a dict from names to sequences, where each name
    is the first word of the header line after ``>``, as most tools and
    [`Genome`][universalmotif.Genome] use, and two sequences with the same
    name are an error. With ``trim_names`` off, names are whole header lines,
    and when they repeat, the last sequence with each name is kept, with a
    warning. The file is read line by line, so a genome needs little more
    memory than its sequences, and the first ``skip`` lines are ignored.
    Gzipped files, bgzip included, are recognized from their first bytes and
    decompressed as they are read.

    With ``packed``, the sequences are packed for scanning as they are read
    and returned as [`PackedSequences`][universalmotif.PackedSequences] for
    ``alphabet``, which is ``"DNA"``, ``"RNA"``, ``"AA"`` or the letters of a
    custom alphabet. Packed sequences hold a genome in a quarter of the
    memory, and names must then be unique. To scan a
    file just once, pass its path to [`scan_sequences()`][universalmotif.scan_sequences] instead.

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = um.read_fasta(um.example_file("fasta"))
    >>> len(seqs), len(seqs["target_001"])
    (100, 1000)
    >>> packed = um.read_fasta(um.example_file("fasta"), packed=True)
    >>> len(packed), packed.alphabet
    (100, 'DNA')
    """
    if skip < 0:
        raise ValueError("skip must be zero or positive")
    if packed:
        return PackedSequences(
            _core._read_fasta_packed(os.fspath(file), alphabet, skip, trim_names)
        )
    records = _core._read_fasta_path(os.fspath(file), skip, trim_names)
    out = dict(records)
    if len(out) < len(records):
        warnings.warn(
            f"{len(records) - len(out)} sequence names appear more than once; only the last "
            "sequence with each name is kept",
            stacklevel=2,
        )
    return out


def write_fasta(
    sequences: Mapping[str, str] | Sequence[str],
    file: PathLike,
    width: int = 80,
    overwrite: bool = False,
    append: bool = False,
) -> None:
    """Write sequences to a FASTA file, wrapping lines at ``width``
    characters, or not at all when it is 0. Unnamed sequences are called
    ``seq1``, ``seq2`` and so on. A path ending in ``.gz`` is written
    gzipped, as it is for every writer in the package.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "seqs.fa"
    ...     um.write_fasta({"first": "ACGTACGTAC", "second": "TTGA"}, path, width=4)
    ...     print(path.read_text(), end="")
    >first
    ACGT
    ACGT
    AC
    >second
    TTGA
    """
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

    ``headers`` writes the name on its own line when ``True``, prefixed with a
    string when given one, or not at all when ``False``. ``sep`` is written
    after each motif unless it is ``None``, ``type`` converts the motifs first,
    and ``digits`` sets the number of decimal places.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:2]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.txt"
    ...     um.write_matrix(motifs, path, headers=">", digits=1)
    ...     lines = path.read_text().splitlines()
    ...     back = um.read_matrix(path, headers=">")
    >>> lines[0], lines[1].split()
    ('>TATA-box', ['2.0', '2.0', '23.0', '1.0', '18.0', '11.0', '20.0', '8.0', '10.0', '7.0'])
    >>> [(m.name, m.type) for m in back]
    [('TATA-box', 'PCM'), ('AP-1', 'PCM')]
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
