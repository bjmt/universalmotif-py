"""Example data for trying out the package."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ._core import Motif

_FILES = {
    "meme": "example.meme",
    "jaspar": "example.jaspar",
    "homer": "example.motif",
    "transfac": "example.transfac",
    "cisbp": "example.cisbp",
    "uniprobe": "example.uniprobe",
    "matrix": "example.matrix",
    "fasta": "example_targets.fa.gz",
}


def _path(name: str) -> Path:
    return Path(str(files("universalmotif") / "data" / name))


def example_motifs() -> list[Motif]:
    """Twelve example DNA motifs.

    The motifs are synthetic. Each is built around a well-known consensus
    site, such as the TATA box, the AP-1 site or the plant W-box and G-box.
    Probabilities are drawn at random around that site, and counts come from
    a random number of sites sampled from those probabilities. The motifs are
    count matrices with their family set and a pseudocount of 1.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> [m.name for m in motifs[:4]]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    """
    from ._io import read_transfac

    return read_transfac(_path("example_motifs.transfac"))


def example_sequences(kind: str = "targets") -> dict[str, str]:
    """A hundred example DNA sequences of 1000 bases each.

    The sequences are random and AT-rich, much like plant promoters. Among
    the ``"targets"``, 60 have a W-box site planted and 40 have a G-box site.
    Both are drawn from [`example_motifs()`][universalmotif.example_motifs]
    and placed towards the middle of the sequences. The ``"controls"`` have
    nothing planted, so the two sets make a small test case for enrichment.

    Examples
    --------
    >>> import universalmotif as um
    >>> targets = um.example_sequences()
    >>> len(targets), len(targets["target_001"])
    (100, 1000)
    """
    if kind not in ("targets", "controls"):
        raise ValueError('kind must be "targets" or "controls"')
    from ._io import read_fasta

    return read_fasta(_path(f"example_{kind}.fa.gz"))


def example_file(format: str) -> Path:
    """The path to an example file in one of the supported formats.

    ``format`` is one of the motif formats ``"meme"``, ``"jaspar"``,
    ``"homer"``, ``"transfac"``, ``"cisbp"``, ``"uniprobe"`` and
    ``"matrix"``. Each of these files holds the first four
    [`example_motifs()`][universalmotif.example_motifs]. Use ``"fasta"`` for
    the gzipped example target sequences.

    Examples
    --------
    >>> import universalmotif as um
    >>> path = um.example_file("jaspar")
    >>> [m.name for m in um.read_jaspar(path)]
    ['TATA-box', 'AP-1', 'CRE', 'G-box']
    """
    if format not in _FILES:
        raise ValueError(f"format must be one of {', '.join(_FILES)}")
    return _path(_FILES[format])
