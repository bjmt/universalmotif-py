"""universalmotif: sequence motif manipulation and analysis.

A Python port of the universalmotif R/Bioconductor package, built on its
Rust counterpart.
"""

from importlib.metadata import PackageNotFoundError, version

from ._api import (
    consensus_to_ppm,
    consensus_to_ppm_aa,
    convert_type,
    create_motif,
    get_consensus,
    get_consensus_aa,
    motif_rc,
    normalize,
    position_icscore,
    round_motif,
    summarise_motifs,
    trim_cwm,
    trim_motifs,
)
from ._io import (
    read_cisbp,
    read_homer,
    read_jaspar,
    read_matrix,
    read_meme,
    read_transfac,
    read_uniprobe,
    write_cisbp,
    write_homer,
    write_jaspar,
    write_matrix,
    write_meme,
    write_transfac,
    write_uniprobe,
)
from ._core import Motif

try:
    __version__ = version("universalmotif")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "Motif",
    "consensus_to_ppm",
    "consensus_to_ppm_aa",
    "convert_type",
    "create_motif",
    "get_consensus",
    "get_consensus_aa",
    "motif_rc",
    "normalize",
    "position_icscore",
    "read_cisbp",
    "read_homer",
    "read_jaspar",
    "read_matrix",
    "read_meme",
    "read_transfac",
    "read_uniprobe",
    "round_motif",
    "summarise_motifs",
    "trim_cwm",
    "trim_motifs",
    "write_cisbp",
    "write_homer",
    "write_jaspar",
    "write_matrix",
    "write_meme",
    "write_transfac",
    "write_uniprobe",
]
