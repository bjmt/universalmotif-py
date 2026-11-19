"""Reading and writing motif files.

Parsing and formatting happen in the Rust crate. This module deals with
files, line skipping and appending."""

from __future__ import annotations

import gzip
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from . import _core
from ._api import MotifType, _bkg_letters, _check_motif
from ._core import Motif
from ._packed import PackedSequences

Positions = Literal["columns", "rows"]
PathLike = str | os.PathLike[str]


def _gunzip(raw: bytes) -> bytes:
    """Gzipped bytes decompressed.

    A bgzipped file is many gzip members one after another. For each one,
    gzip.decompress() copies the rest of the file, which takes minutes for a
    large file. Reading it as a stream takes seconds."""
    import io

    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        return fh.read()


def _read_text(file: PathLike, skip: int) -> str:
    raw = Path(file).read_bytes()
    # Gzipped files are recognized from their first bytes.
    text = (_gunzip(raw) if raw[:2] == b"\x1f\x8b" else raw).decode()
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


def _gap_notation(m: Motif) -> str | None:
    """A gapped motif's gaps written into its consensus, as in
    AGGTCA[1-5]AGGTCA."""
    if not m.isgapped or m.consensus is None:
        return None
    c, out, at = m.consensus, "", 0
    for loc, lo, hi in zip(m.gaploc, m.mingap, m.maxgap):
        out += f"{c[at:loc]}[{lo}-{hi}]"
        at = loc
    return out + c[at:]


def _warn_gaps(motifs: list[Motif], fmt: str, notation: bool = False) -> None:
    """Warn of gaps a format loses. Formats that keep them in the
    alternative name (``notation``) lose them only for motifs without a
    consensus, and replace the alternative names of gapped motifs."""
    gapped = [m for m in motifs if m.isgapped]
    if not gapped:
        return
    if not notation or any(m.consensus is None for m in gapped):
        warnings.warn(
            f"{fmt} format cannot keep the gaps of {'these motifs' if notation else 'motifs'}; "
            "write_universalmotif() keeps them" + ("" if notation else ", and MEME, JASPAR and "
                                                   "TRANSFAC files keep them in the alternative name"),
            stacklevel=3,
        )
    replaced = [m.name for m in gapped if m.altname and _gap_notation(m) not in (None, m.altname)]
    if notation and replaced:
        warnings.warn(
            f"the alternative names of gapped motifs ({', '.join(replaced[:3])}"
            f"{', ...' if len(replaced) > 3 else ''}) are replaced by their gaps, written as in "
            "AGGTCA[1-5]AGGTCA",
            stacklevel=3,
        )


def _write_text(file: PathLike, text: str, overwrite: bool, append: bool) -> None:
    path = Path(file)
    if path.exists() and not overwrite and not append:
        raise FileExistsError(f"{path} exists; set overwrite=True or append=True")
    # A .gz path is written gzipped at gzip's own default level of 6. This
    # is several times faster than Python's default of 9, for files barely
    # larger. Appending adds a gzip member, which readers treat as a
    # continuation of the file.
    mode = "at" if append else "wt"
    with (gzip.open(path, mode, compresslevel=6) if path.suffix == ".gz" else open(path, mode)) as f:
        # Written a slice at a time, so the text is never encoded whole.
        step = 1 << 20
        for i in range(0, len(text), step):
            f.write(text[i:i + step])


def read_jaspar(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read JASPAR count matrices.

    Rows may be labelled, as in ``A [ 1 2 3 ]``, or bare numbers. Bare rows
    are read as DNA when there are four and as amino acids when there are
    twenty. Motifs are returned as PCMs with ``nsites`` set to the largest
    column sum. ``skip`` drops lines from the start of the file.

    References
    ----------
    - Rauluseviciute I, Riudavets-Puig R, Blanc-Mathieu R, Castro-Mondragon JA,
      Ferenc K, Kumar V, Lemma RB, Lucas J, et al. (2024). JASPAR 2024: 20th
      anniversary of the open-access database of transcription factor binding
      profiles. *Nucleic Acids Research* 52(D1):D174–D182.
      <https://doi.org/10.1093/nar/gkad1059>

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
    ``"logodds.threshold"``. Site counts and P-values are read from HOMER's
    statistics column when present.

    References
    ----------
    - Heinz S, Benner C, Spann N, Bertolino E, Lin YC, Laslo P, Cheng JX, Murre
      C, et al. (2010). Simple combinations of lineage-determining
      transcription factors prime cis-regulatory elements required for
      macrophage and B cell identities. *Molecular Cell* 38(4):576–589.
      <https://doi.org/10.1016/j.molcel.2010.05.004>

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

    The accession in ``AC`` is used as the name if there is one. The
    identifier in ``ID`` or the factor name in ``NA`` becomes the alternate
    name. ``HC`` and ``OS`` give the family and organism.

    References
    ----------
    - Matys V, Kel-Margoulis OV, Fricke E, et al. (2006). TRANSFAC and its
      module TRANSCompel: transcriptional gene regulation in eukaryotes.
      *Nucleic Acids Research* 34(suppl_1):D108–D110.
      <https://doi.org/10.1093/nar/gkj143>

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
    """Read CIS-BP motifs.

    The ``TF``, ``TF Name``, ``Family`` and ``Species`` fields give the
    name, alternate name, family and organism.

    References
    ----------
    - Weirauch MT, Yang A, Albu M, Cote AG, Montenegro-Montero A, Drewe P,
      Najafabadi HS, Lambert SA, et al. (2014). Determination and inference of
      eukaryotic transcription factor sequence specificity. *Cell*
      158(6):1431–1443. <https://doi.org/10.1016/j.cell.2014.08.009>

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
    """Read UniPROBE motifs from minimal files or from full output.

    The enrichment scores of full output are kept in ``extrainfo``.

    References
    ----------
    - Hume MA, Barrera LA, Gisselbrecht SS, Bulyk ML (2015). UniPROBE, update
      2015: new tools and content for the online database of protein-binding
      microarray data on protein–DNA interactions. *Nucleic Acids Research*
      43(D1):D117–D122. <https://doi.org/10.1093/nar/gku1045>

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
    of other alphabets for one. A motif whose alternative name gives gaps,
    as in ``AGGTCA[1-5]AGGTCA``, gets those gaps, as do those read by
    [`read_jaspar()`][universalmotif.read_jaspar] and
    [`read_transfac()`][universalmotif.read_transfac].

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

    References
    ----------
    - Bailey TL, Johnson J, Grant CE, Noble WS (2015). The MEME Suite. *Nucleic
      Acids Research* 43(W1):W39–W49. <https://doi.org/10.1093/nar/gkv416>

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
        The matrix type, detected from the values by default.
    positions
        Whether positions are ``"columns"`` or ``"rows"``.
    alphabet
        Alphabet of the motifs. The letters' rows or columns are read in its
        order.
    sep
        Line separating motifs. ``""`` means blank lines, and ``None`` means
        motifs are only separated by headers.
    headers
        ``True`` if the first line of each motif is its name, ``False`` for
        no names, or a string marking header lines. Each header line starts
        a new motif, and the name is the line with the string removed.
    rownames
        Whether the values come with letter labels, which are skipped.
    comment
        Lines starting with this string are ignored. A first line such as
        ``# type: CWM`` gives the type when ``type`` is not given, and
        makes lines starting with ``#`` comments.

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
    text = _read_text(file, skip)
    # A first line such as "# type: CWM", as the command line writes for CWMs, gives the type.
    first = text.lstrip().split("\n", 1)[0]
    marked = re.match(r"#\s*type:\s*(\w+)\s*$", first)
    if marked:
        if type is None:
            type = marked.group(1)
        if comment is None:
            comment = "#"
    return _core._read_matrix(text, positions, alphabet, type, sep, headers, rownames, comment)


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

    Motifs are converted to PPMs. With ``cwm=True``, every motif must
    instead be a CWM and is written as is.

    ``bkg`` gives the background in the order letters are written, which
    for DNA is A, C, G, T. By default the motifs' shared background is used,
    or a uniform one if they differ. ``strand`` overrides the ``strands:``
    line. When appending to an existing file, only the motifs are written.
    Equal signs in names become periods and spaces become dashes. A gapped
    motif gets its gaps as its alternative name, written into its consensus
    as in ``AGGTCA[1-5]AGGTCA``, which the readers turn back into gaps.

    References
    ----------
    - Bailey TL, Johnson J, Grant CE, Noble WS (2015). The MEME Suite. *Nucleic
      Acids Research* 43(W1):W39–W49. <https://doi.org/10.1093/nar/gkv416>

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
    _warn_gaps(ms, "MEME", notation=True)
    header = not (append and Path(file).exists())
    text = _core._write_meme(
        ms, version, None if bkg is None else _bkg_letters(bkg), strand, cwm, header
    )
    _write_text(file, text, overwrite, append)


def write_jaspar(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA, RNA or amino acid motifs as JASPAR count matrices.

    A gapped motif gets its gaps as its alternative name, as
    [`write_meme()`][universalmotif.write_meme] describes.

    References
    ----------
    - Rauluseviciute I, Riudavets-Puig R, Blanc-Mathieu R, Castro-Mondragon JA,
      Ferenc K, Kumar V, Lemma RB, Lucas J, et al. (2024). JASPAR 2024: 20th
      anniversary of the open-access database of transcription factor binding
      profiles. *Nucleic Acids Research* 52(D1):D174–D182.
      <https://doi.org/10.1093/nar/gkad1059>

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
    _warn_gaps(ms, "JASPAR", notation=True)
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

    The detection threshold in each header is on HOMER's own score scale.
    HOMER scores a site as the sum of ``ln(p / 0.25)`` over the
    probabilities written in the file, floored at 0.001. The threshold
    depends on ``threshold_type``.

    - ``"logodds"``: ``threshold`` times the motif's maximum score.
    - ``"logodds_abs"``: ``threshold`` itself.
    - ``"pvalue"``: a score that random sites from a uniform background
      reach with at most that probability.

    Give one value or one per motif.

    References
    ----------
    - Heinz S, Benner C, Spann N, Bertolino E, Lin YC, Laslo P, Cheng JX, Murre
      C, et al. (2010). Simple combinations of lineage-determining
      transcription factors prime cis-regulatory elements required for
      macrophage and B cell identities. *Molecular Cell* 38(4):576–589.
      <https://doi.org/10.1016/j.molcel.2010.05.004>

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
    """Write DNA or RNA motifs as TRANSFAC count matrices.

    ``name_tag`` and ``altname_tag`` choose the fields used for the name and
    alternate name. A gapped motif gets its gaps as its alternative name, as
    [`write_meme()`][universalmotif.write_meme] describes.

    References
    ----------
    - Matys V, Kel-Margoulis OV, Fricke E, et al. (2006). TRANSFAC and its
      module TRANSCompel: transcriptional gene regulation in eukaryotes.
      *Nucleic Acids Research* 34(suppl_1):D108–D110.
      <https://doi.org/10.1093/nar/gkj143>

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
    _warn_gaps(ms, "TRANSFAC", notation=True)
    _write_text(file, _core._write_transfac(ms, name_tag, altname_tag), overwrite, append)


def write_cisbp(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write DNA or RNA motifs in CIS-BP format, as probability matrices.

    The name, alternate name, family and organism go in the ``TF``,
    ``TF Name``, ``Family`` and ``Species`` fields. Files written here can be
    read back with [`read_cisbp()`][universalmotif.read_cisbp].

    References
    ----------
    - Weirauch MT, Yang A, Albu M, Cote AG, Montenegro-Montero A, Drewe P,
      Najafabadi HS, Lambert SA, et al. (2014). Determination and inference of
      eukaryotic transcription factor sequence specificity. *Cell*
      158(6):1431–1443. <https://doi.org/10.1016/j.cell.2014.08.009>

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
    """Write DNA motifs in the minimal UniPROBE format.

    Each motif is a name line followed by ``A:``, ``C:``, ``G:`` and ``T:``
    rows of probabilities. Files written here can be read back with
    [`read_uniprobe()`][universalmotif.read_uniprobe].

    References
    ----------
    - Hume MA, Barrera LA, Gisselbrecht SS, Bulyk ML (2015). UniPROBE, update
      2015: new tools and content for the online database of protein-binding
      microarray data on protein–DNA interactions. *Nucleic Acids Research*
      43(D1):D117–D122. <https://doi.org/10.1093/nar/gku1045>

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


def read_universalmotif(file: PathLike, skip: int = 0) -> list[Motif]:
    """Read motifs saved by the R package universalmotif.

    The R package's ``write_motifs()`` saves motifs in a text format of its
    own, which keeps every field of a motif: its names, type, background,
    pseudocount, ``nsites``, P-values, extra information and gaps. Files
    from version 1.1.67 of the R package onwards can be read, and gaps from
    version 1.31.51. Higher-order matrices (``multifreq``), background
    frequencies of words longer than one letter, and the extra P-values of
    gapped motifs are left out with a warning, as this package does not
    keep them. The total IC and the consensus are computed again from the
    matrix. ``skip`` drops lines from the start of the file.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> dr = um.add_gap(um.create_motif("AGGTCAAGGTCA", name="DR"), gaploc=6, mingap=1, maxgap=5)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.yml"
    ...     um.write_universalmotif(dr, path)
    ...     back = um.read_universalmotif(path)[0]
    >>> back.name, back.gaploc, back.mingap, back.maxgap
    ('DR', [6], [1], [5])
    """
    motifs, notes = _core._read_universalmotif(_read_text(file, skip))
    for note in notes:
        warnings.warn(note, stacklevel=2)
    return motifs


def write_universalmotif(
    motifs: Motif | Iterable[Motif], file: PathLike, overwrite: bool = False, append: bool = False
) -> None:
    """Write motifs in the text format of the R package universalmotif.

    The R package reads these files with ``read_motifs()``. Every field of
    each motif is kept, gaps included, with the numbers written in full so
    that [`read_universalmotif()`][universalmotif.read_universalmotif]
    gives back the same motifs. Versions of the R package before 1.31.51
    fail on a file holding a gapped motif, so for those, write the motifs
    with their gaps turned off by [`ungap()`][universalmotif.ungap].

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import universalmotif as um
    >>> motif = um.create_motif("TATAAA", name="TATA", nsites=20)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "motifs.yml"
    ...     um.write_universalmotif(motif, path)
    ...     for line in path.read_text().splitlines()[:6]:
    ...         print(line)
    # universalmotif version 1.31.51
    <BLANKLINE>
    ---
    name: TATA
    alphabet: DNA
    type: PPM
    """
    ms = _motif_list(motifs)
    header = not (append and Path(file).exists())
    _write_text(file, _core._write_universalmotif(ms, header), overwrite, append)


def read_fasta(
    file: PathLike,
    skip: int = 0,
    packed: bool = False,
    alphabet: str = "DNA",
    trim_names: bool = True,
) -> dict[str, str] | PackedSequences:
    """Read a FASTA file into a dict from names to sequences.

    Each name is the first word of the header line after ``>``, as most
    tools and [`Genome`][universalmotif.Genome] use. Two sequences with the
    same name are an error. With ``trim_names`` off, names are whole header
    lines. When these repeat, the last sequence with each name is kept, with
    a warning.

    The file is read line by line, so a genome needs little more memory than
    its sequences. The first ``skip`` lines are ignored. Gzipped files,
    bgzip included, are recognized from their first bytes and decompressed
    as they are read.

    With ``packed``, the sequences are packed for scanning as they are read.
    They are returned as [`PackedSequences`][universalmotif.PackedSequences]
    for ``alphabet``, which is ``"DNA"``, ``"RNA"``, ``"AA"`` or the letters
    of a custom alphabet. Packed sequences hold a genome in a quarter of the
    memory, and names must then be unique. To scan a file just once, pass
    its path to [`scan_sequences()`][universalmotif.scan_sequences] instead.

    References
    ----------
    - Pearson WR, Lipman DJ (1988). Improved tools for biological sequence
      comparison. *Proceedings of the National Academy of Sciences*
      85(8):2444–2448. <https://doi.org/10.1073/pnas.85.8.2444>

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
    """Write sequences to a FASTA file.

    Lines are wrapped at ``width`` characters, or not at all when it is 0.
    Unnamed sequences are called ``seq1``, ``seq2`` and so on. A path ending
    in ``.gz`` is written gzipped, as it is for every writer in the package.

    References
    ----------
    - Pearson WR, Lipman DJ (1988). Improved tools for biological sequence
      comparison. *Proceedings of the National Academy of Sciences*
      85(8):2444–2448. <https://doi.org/10.1073/pnas.85.8.2444>

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

    ``headers`` controls how each name is written.

    - ``True`` writes the name on its own line.
    - A string writes it on its own line, prefixed with that string.
    - ``False`` leaves names out.

    ``sep`` is written after each motif unless it is ``None``. ``type``
    converts the motifs first, and ``digits`` sets the number of decimal
    places.

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
