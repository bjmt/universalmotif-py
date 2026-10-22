"""Conversion to and from Biopython's ``Bio.motifs``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, overload

import numpy as np

from ._api import _check_motif, convert_type, create_motif
from ._core import Motif

_DNA = "ACGT"
_RNA = "ACGU"
_AA = "ACDEFGHIKLMNPQRSTVWY"


def _bio_motifs():
    try:
        from Bio import motifs
    except ImportError as e:
        raise ImportError(
            "converting to and from Biopython needs biopython; install it with "
            "`pip install universalmotif[biopython]`"
        ) from e
    return motifs


def _text(value: Any) -> str | None:
    """A Biopython attribute as text: lists are joined, empty values dropped."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if v not in (None, ""))
    value = str(value).strip()
    return value or None


@overload
def from_biopython(motifs: Any) -> Motif | list[Motif]: ...
def from_biopython(motifs):
    """Convert motifs from Biopython's ``Bio.motifs``.

    Takes a ``Bio.motifs.Motif``, or a list of them such as a
    ``Bio.motifs.Record`` read with ``Bio.motifs.parse()``, and returns a
    motif or a list of motifs. Counts become a PCM, and frequencies that do
    not look like counts a PPM. The alphabet is DNA, RNA or amino acids when
    the motif's letters are exactly those, and a custom alphabet otherwise.

    JASPAR motifs, including those from pyJASPAR, keep their matrix ID as the
    name, the transcription factor as the alternative name, the family as
    the family and the species as the organism, with the class, taxonomic
    group, data type and accession kept in ``extrainfo``. MEME motifs keep
    their alternative ID and E-value. Background frequencies come across,
    and Biopython's per-letter pseudocounts become a single pseudocount, the
    sum of the per-letter ones.

    Needs the ``biopython`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> from Bio import motifs
    >>> bio = motifs.create(["TGACTCA", "TGAGTCA", "TGACTAA"])
    >>> bio.name = "AP-1"
    >>> motif = um.from_biopython(bio)
    >>> motif.name, motif.type, motif.consensus
    ('AP-1', 'PCM', 'TGASTMA')
    """
    _bio_motifs()
    single = not isinstance(motifs, (list, tuple)) and not _is_record(motifs)
    out = [_from_one(m) for m in ([motifs] if single else list(motifs))]
    return out[0] if single else out


def _is_record(x: Any) -> bool:
    return isinstance(x, list) or type(x).__name__ == "Record"


def _from_one(m: Any) -> Motif:
    letters = "".join(sorted(str(m.alphabet)))
    counts = m.counts
    matrix = np.array([[float(v) for v in counts[a]] for a in letters], dtype=float)
    if letters == _DNA:
        alphabet = "DNA"
    elif letters == _RNA:
        alphabet = "RNA"
    elif letters == _AA:
        alphabet = "AA"
    else:
        alphabet = letters
    is_counts = np.all((matrix == np.round(matrix)) & (matrix >= 0)) and matrix.sum(axis=0).min() >= 1
    kind = "PCM" if is_counts else "PPM"

    matrix_id = getattr(m, "matrix_id", None)
    name = _text(matrix_id) or _text(getattr(m, "name", None)) or _text(getattr(m, "id", None))
    altname = _text(getattr(m, "name", None)) if matrix_id else _text(getattr(m, "alt_id", None))
    if altname == name:
        altname = None
    extrainfo = []
    for key, attr in (("class", "tf_class"), ("tax_group", "tax_group"),
                      ("data_type", "data_type"), ("accession", "acc"), ("collection", "collection")):
        value = _text(getattr(m, attr, None))
        if value:
            extrainfo.append((key, value))

    bkg = getattr(m, "background", None)
    # Biopython adds its pseudocount to each letter, either the same number
    # for all of them or one per letter; the total spread over the column is
    # the pseudocount here.
    pseudo = getattr(m, "pseudocounts", None)
    if isinstance(pseudo, dict):
        pseudo = sum(float(pseudo.get(a, 0.0)) for a in letters)
    elif pseudo is not None:
        pseudo = float(pseudo) * len(letters)
    evalue = getattr(m, "evalue", None)
    return create_motif(
        matrix,
        alphabet=alphabet,
        type=kind,
        name=name or "motif",
        altname=altname,
        family=_text(getattr(m, "tf_family", None)),
        organism=_text(getattr(m, "species", None)),
        bkg=None if not bkg else [float(bkg[a]) for a in letters],
        pseudocount=float(pseudo or 0.0),
        nsites=round(matrix.sum(axis=0).max()) if is_counts else None,
        eval=float(evalue) if evalue is not None else None,
        extrainfo=dict(extrainfo) or None,
    )


@overload
def to_biopython(motifs: Motif, jaspar: bool = ...) -> Any: ...
@overload
def to_biopython(motifs: Iterable[Motif], jaspar: bool = ...) -> list[Any]: ...
def to_biopython(motifs, jaspar=False):
    """Convert motifs to Biopython's ``Bio.motifs``.

    Each motif becomes a ``Bio.motifs.Motif`` holding counts, from the
    motif's PCM with ``nsites``, or 100 sites when it has none. The name,
    background and pseudocount come across, the pseudocount spread evenly
    over the letters as Biopython keeps one per letter. With ``jaspar``, the
    motifs become ``Bio.motifs.jaspar.Motif`` instead, with the name as the
    matrix ID and the alternative name, family and organism filled in, as
    Biopython's JASPAR writer expects. Returns a motif for a motif and a list
    for a list.

    Needs the ``biopython`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> bio = um.to_biopython(um.example_motifs()[1])
    >>> bio.name, bio.length, bio.consensus
    ('AP-1', 9, Seq('TTGACTCAG'))
    """
    bio = _bio_motifs()
    single = isinstance(motifs, Motif)
    ms = [motifs] if single else [_check_motif(m) for m in motifs]
    out = []
    for m in ms:
        pcm = convert_type(m, "PCM")
        letters = "".join(m.letters)
        counts = {a: [float(v) for v in row] for a, row in zip(letters, pcm.matrix)}
        if jaspar:
            from Bio.motifs import jaspar as bio_jaspar

            b = bio_jaspar.Motif(
                m.name,
                m.altname or m.name,
                alphabet=letters,
                counts=counts,
                tf_family=m.family,
                species=m.organism,
            )
        else:
            b = bio.Motif(alphabet=letters, counts=counts)
            b.name = m.name
        b.background = {a: float(v) for a, v in m.bkg.items()}
        b.pseudocounts = {a: m.pseudocount / len(letters) for a in letters}
        out.append(b)
    return out[0] if single else out
