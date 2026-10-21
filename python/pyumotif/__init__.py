"""pyumotif: sequence motif manipulation and analysis.

A Python port of the universalmotif R/Bioconductor package, built on the
umotif Rust library.
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
from ._umotif import Motif

try:
    __version__ = version("pyumotif")
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
    "round_motif",
    "summarise_motifs",
    "trim_cwm",
    "trim_motifs",
]
