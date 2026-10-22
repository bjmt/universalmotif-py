"""universalmotif: sequence motif manipulation and analysis.

A Python port of the universalmotif R/Bioconductor package, built on its
Rust counterpart.
"""

from importlib.metadata import PackageNotFoundError, version

from ._analysis import dedup_hits, motif_pvalue, scan_sequences
from ._compare import MotifTree, compare_motifs, merge_motifs, merge_similar, motif_tree
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
    read_fasta,
    read_homer,
    read_jaspar,
    read_matrix,
    read_meme,
    read_transfac,
    read_uniprobe,
    write_cisbp,
    write_fasta,
    write_homer,
    write_jaspar,
    write_matrix,
    write_meme,
    write_transfac,
    write_uniprobe,
)
from ._core import Motif
from ._enrich import enrich_motifs, motif_coocc, motif_peaks
from ._sequences import (
    calc_complexity,
    calc_windows,
    count_klets,
    create_sequences,
    get_bkg,
    get_klets,
    implant_motifs,
    motif_finder,
    sample_sites,
    sequence_complexity,
    shuffle_motifs,
    shuffle_sequences,
    window_string,
)

try:
    __version__ = version("universalmotif")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "Motif",
    "MotifTree",
    "calc_complexity",
    "calc_windows",
    "compare_motifs",
    "consensus_to_ppm",
    "consensus_to_ppm_aa",
    "convert_type",
    "count_klets",
    "create_motif",
    "create_sequences",
    "dedup_hits",
    "enrich_motifs",
    "get_bkg",
    "get_consensus",
    "get_consensus_aa",
    "get_klets",
    "implant_motifs",
    "merge_motifs",
    "merge_similar",
    "motif_coocc",
    "motif_finder",
    "motif_peaks",
    "motif_pvalue",
    "motif_rc",
    "motif_tree",
    "normalize",
    "position_icscore",
    "read_cisbp",
    "read_fasta",
    "read_homer",
    "read_jaspar",
    "read_matrix",
    "read_meme",
    "read_transfac",
    "read_uniprobe",
    "round_motif",
    "sample_sites",
    "scan_sequences",
    "sequence_complexity",
    "shuffle_motifs",
    "shuffle_sequences",
    "summarise_motifs",
    "trim_cwm",
    "trim_motifs",
    "window_string",
    "write_cisbp",
    "write_fasta",
    "write_homer",
    "write_jaspar",
    "write_matrix",
    "write_meme",
    "write_transfac",
    "write_uniprobe",
]
