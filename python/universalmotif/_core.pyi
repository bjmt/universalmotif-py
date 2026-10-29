from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

class ArrowTable:
    """An Arrow table, readable by polars, pandas and pyarrow through the
    Arrow PyCapsule interface."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

class Motif:
    """A sequence motif.

    Motifs are immutable. The functions in the package, such as
    [`convert_type()`][universalmotif.convert_type] or
    [`trim_motifs()`][universalmotif.trim_motifs], return new motifs, and
    `replace()` returns a copy with some of the metadata changed. New motifs
    are made with [`create_motif()`][universalmotif.create_motif] or read from
    a file.

    Examples
    --------
    >>> import universalmotif as um
    >>> motif = um.example_motifs()[3]
    >>> print(motif)
           Motif name:   G-box
               Family:   bZIP
             Organism:   synthetic
             Alphabet:   DNA
                 Type:   PCM
              Strands:   +-
             Total IC:   7.9
          Pseudocount:   1
            Consensus:   NCACGTGN
         Target sites:   71
    <BLANKLINE>
       N  C  A  C  G  T  G  N
    A 17  6 62  0  2  0 12 27
    C  9 62  3 69  1 15  1 10
    G 23  1  2  0 64  3 58 21
    T 22  2  4  2  4 53  0 13

    The matrix has one row per letter and one column per position:

    >>> motif.letters, motif.matrix.shape
    (['A', 'C', 'G', 'T'], (4, 8))
    >>> motif.matrix[:, 2].tolist()
    [62.0, 3.0, 2.0, 4.0]

    Metadata is changed by making a copy:

    >>> renamed = motif.replace(name="G-box copy", organism=None)
    >>> renamed.name, renamed.organism, motif.name
    ('G-box copy', None, 'G-box')
    """

    @property
    def name(self) -> str: ...
    @property
    def altname(self) -> str | None: ...
    @property
    def family(self) -> str | None: ...
    @property
    def organism(self) -> str | None: ...
    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Read-only array of shape (letters, positions)."""
    @property
    def alphabet(self) -> str: ...
    @property
    def letters(self) -> list[str]: ...
    @property
    def type(self) -> str: ...
    @property
    def icscore(self) -> float: ...
    @property
    def nsites(self) -> float | None: ...
    @property
    def pseudocount(self) -> float: ...
    @property
    def bkg(self) -> dict[str, float]: ...
    @property
    def bkgsites(self) -> float | None: ...
    @property
    def consensus(self) -> str | None:
        """One IUPAC letter per position, or None for custom alphabets. Gaps
        are not shown, so it has one letter per column even for a gapped
        motif, whose printout marks the gaps with dots. For a CWM each letter
        describes the shape of its column, however small its contributions."""
    @property
    def strand(self) -> str:
        """``"+"``, ``"-"`` or ``"+-"``. Only DNA and RNA motifs have a reverse
        strand, so for other alphabets it has no effect."""
    @property
    def pval(self) -> float | None: ...
    @property
    def qval(self) -> float | None: ...
    @property
    def eval(self) -> float | None: ...
    @property
    def extrainfo(self) -> dict[str, str]: ...
    @property
    def isgapped(self) -> bool: ...
    @property
    def gaploc(self) -> list[int]: ...
    @property
    def mingap(self) -> list[int]: ...
    @property
    def maxgap(self) -> list[int]: ...
    @property
    def width(self) -> int: ...
    def __len__(self) -> int: ...
    def _repr_html_(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def replace(
        self,
        *,
        name: str = ...,
        altname: str | None = ...,
        family: str | None = ...,
        organism: str | None = ...,
        nsites: float | None = ...,
        pseudocount: float = ...,
        bkg: Sequence[float] | Mapping[str, float] = ...,
        bkgsites: float | None = ...,
        strand: str = ...,
        pval: float | None = ...,
        qval: float | None = ...,
        eval: float | None = ...,
        extrainfo: Mapping[str, str] | None = ...,
    ) -> Motif:
        """Return a copy with some metadata replaced.

        The matrix, type and alphabet cannot be changed this way. Changing
        the pseudocount, background or ``nsites`` recomputes the total IC
        and the consensus.
        """
    def _state(self) -> dict[str, Any]: ...

def _motif_from_state(state: Mapping[str, Any]) -> Motif: ...
def _create_from_matrix(
    matrix: npt.NDArray[np.float64],
    letters: list[str] | None = ...,
    alphabet: str | None = ...,
    matrix_type: str | None = ...,
    **kwargs: Any,
) -> Motif: ...
def _create_from_strings(strings: list[str], alphabet: str | None = ..., **kwargs: Any) -> Motif: ...
def _create_random(width: int, alphabet: str = ..., seed: int | None = ..., **kwargs: Any) -> Motif: ...
def _convert_type(
    motif: Motif, type: str, pseudocount: float | None = ..., relative_entropy: bool = ...
) -> Motif: ...
def _motif_rc(motif: Motif, ignore_alphabet: bool) -> Motif: ...
def _trim_motif(motif: Motif, min_ic: float, trim_from: str) -> Motif | None: ...
def _trim_cwm(
    motif: Motif, trim_threshold: float, abs_threshold: float | None, trim_from: str
) -> Motif | None: ...
def _round_motif(motif: Motif, pct_tolerance: float) -> Motif: ...
def _normalize(motif: Motif) -> Motif: ...
def _position_icscore(
    position: list[float],
    bkg: list[float],
    type: str,
    pseudocount: float,
    nsites: float,
    relative_entropy: bool,
) -> float: ...
def _get_consensus(position: list[float], rna: bool, type: str, pseudocount: float) -> str: ...
def _get_consensus_aa(position: list[float], type: str, pseudocount: float) -> str: ...
def _consensus_to_ppm(letter: str) -> list[float]: ...
def _consensus_to_ppm_aa(letter: str) -> list[float]: ...
def _summarize(motifs: list[Motif]) -> dict[str, list[Any]]: ...

def _read_jaspar(text: str) -> list[Motif]: ...
def _read_homer(text: str) -> list[Motif]: ...
def _read_transfac(text: str) -> list[Motif]: ...
def _read_cisbp(text: str) -> list[Motif]: ...
def _read_uniprobe(text: str) -> list[Motif]: ...
def _read_meme(text: str, cwm: bool) -> list[Motif]: ...
def _read_meme_sites(text: str) -> list[tuple[str, list[tuple[str, str]]]]: ...
def _read_meme_site_pvalues(
    text: str,
) -> list[tuple[str, list[tuple[str, str | None, float, float, str]]]]: ...
def _read_meme_combined(text: str) -> list[tuple[str, float, str]]: ...
def _read_matrix(
    text: str,
    positions: str,
    alphabet: str,
    type: str | None,
    sep: str | None,
    headers: bool | str,
    rownames: bool,
    comment: str | None,
) -> list[Motif]: ...
def _write_meme(
    motifs: list[Motif],
    version: int,
    bkg: list[float] | None,
    strand: str | None,
    cwm: bool,
    header: bool,
) -> str: ...
def _write_jaspar(motifs: list[Motif]) -> str: ...
def _write_homer(motifs: list[Motif], threshold: list[float], threshold_type: str) -> str: ...
def _write_transfac(motifs: list[Motif], name_tag: str, altname_tag: str) -> str: ...
def _write_matrix(
    motifs: list[Motif],
    positions: str,
    rownames: bool,
    type: str | None,
    sep: str | None,
    headers: bool | str,
    digits: int,
) -> str: ...
def _write_cisbp(motifs: list[Motif]) -> str: ...
def _write_uniprobe(motifs: list[Motif]) -> str: ...
def _read_fasta(text: str) -> list[tuple[str, str]]: ...
def _read_fasta_path(path: str, skip: int, trim_names: bool = ...) -> list[tuple[str, str]]: ...
def _read_fasta_packed(
    path: str, alphabet: str, skip: int, trim_names: bool = ...
) -> _PackedSequences: ...
def _pack_sequences(
    names: list[str], sequences: list[str], alphabet: str, nthreads: int
) -> _PackedSequences: ...

class _PackedSequences:
    def __len__(self) -> int: ...
    def names(self) -> list[str]: ...
    def index(self, name: str) -> int | None: ...
    def decode(self, i: int) -> str: ...
    def decode_range(self, i: int, start: int, end: int) -> str: ...
    def lengths(self) -> list[int]: ...
    @property
    def alphabet(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
class _Genome:
    def __init__(self, path: str) -> None: ...
    def names(self) -> list[str]: ...
    def lengths(self) -> list[int]: ...
    def index(self, name: str) -> int | None: ...
    def fetch(self, i: int, start: int, end: int) -> str: ...
    def fetch_many(self, ranges: list[tuple[int, int, int]], nthreads: int = ...) -> list[str]: ...
def _write_fasta(records: list[tuple[str, str]], width: int) -> str: ...
def _min_pvalues(motifs: list[Motif], nthreads: int = ...) -> list[float]: ...
def _motif_pvalue(
    motifs: list[Motif],
    queries: list[tuple[int, float]],
    bkg: object,
    nthreads: int,
    to_score: bool,
) -> list[float]: ...
def _scan_sequences(
    motifs: list[Motif],
    sequences: list[str],
    pvalue: float,
    rc: bool,
    no_overlaps: tuple[str, bool, bool] | None,
    nthreads: int,
    use_gaps: bool,
    correct_gaps: bool,
    first_hit: bool,
    motif_names: list[str],
    sequence_names: list[str],
) -> ArrowTable: ...
def _motif_matches(
    motifs: list[Motif],
    sequences: list[str] | None,
    packed: _PackedSequences | None,
    pvalue: float,
    rc: bool,
    no_overlaps: tuple[str, bool, bool] | None,
    nthreads: int,
    use_gaps: bool,
    value: str,
) -> tuple[int, int, npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float64]]: ...
def _scan_packed(
    motifs: list[Motif],
    packed: _PackedSequences,
    pvalue: float,
    rc: bool,
    no_overlaps: tuple[str, bool, bool] | None,
    nthreads: int,
    use_gaps: bool,
    correct_gaps: bool,
    first_hit: bool,
    motif_names: list[str],
) -> ArrowTable: ...
def _scan_fasta(
    motifs: list[Motif],
    path: str,
    pvalue: float,
    rc: bool,
    no_overlaps: tuple[str, bool, bool] | None,
    nthreads: int,
    use_gaps: bool,
    correct_gaps: bool,
    first_hit: bool,
    motif_names: list[str],
) -> ArrowTable: ...
def _dedup(
    sequence: list[int],
    motif: list[int],
    strand: list[int],
    start: list[int],
    end: list[int],
    priority: list[float],
    ignore_strand: bool,
    ignore_motif: bool,
) -> list[bool]: ...
def _compare_matrix(
    motifs: list[Motif],
    min_overlap: int,
    rc: bool,
    null: str,
    bkg: list[float] | None,
    need_pvalues: bool,
    nthreads: int,
    complete: bool = ...,
    method: str = ...,
    overhang: str = ...,
) -> dict[str, Any]: ...
def _compare_table(
    motifs: list[Motif],
    queries: list[int],
    min_overlap: int,
    rc: bool,
    null: str,
    bkg: list[float] | None,
    nthreads: int,
    names: list[str],
    complete: bool = ...,
    method: str = ...,
    overhang: str = ...,
) -> ArrowTable: ...
def _merge_motifs(
    motifs: list[Motif],
    min_overlap: int,
    rc: bool,
    name: str | None,
    weighted: bool,
    nthreads: int,
) -> tuple[Motif, list[int]]: ...
def _merge_similar(
    motifs: list[Motif],
    qvalue: float,
    null: str,
    min_overlap: int,
    rc: bool,
    bkg: list[float] | None,
    weighted: bool,
    nthreads: int,
) -> tuple[list[int], list[Motif]]: ...
def _hclust(dist: list[list[float]], method: str, labels: list[str]) -> dict[str, Any]: ...
def _motif_tree(
    motifs: list[Motif],
    min_overlap: int,
    rc: bool,
    method: str,
    labels: list[str],
    nthreads: int,
) -> dict[str, Any]: ...
def _kmers(letters: str, k: int) -> list[str]: ...
def _count_kmers(seq: str, k: int, alphabet: str) -> list[int]: ...
def _windows(len: int, window: int, overlap: int, incomplete: bool) -> list[tuple[int, int]]: ...
def _complexity(seqs: list[str], method: str, alph: str | None, max_word: int) -> list[float]: ...
def _sequence_complexity(
    seqs: list[str],
    names: list[str],
    method: str,
    window: float,
    overlap: float | None,
    alphabet: str | None,
    max_word: int,
    nthreads: int,
) -> ArrowTable: ...
def _get_bkg(
    seqs: list[str],
    names: list[str],
    k: list[int],
    pseudocount: int,
    alphabet: str | None,
    rc: bool,
    merge: bool,
    window: tuple[float, float] | None,
    nthreads: int,
) -> ArrowTable: ...
def _shuffle_sequences(
    seqs: list[str],
    k: int,
    method: str,
    seed: int,
    window: tuple[float, float] | None,
    nthreads: int,
) -> list[str]: ...
def _create_sequences(
    alphabet: str, n: int, len: int, freqs: list[float] | None, seed: int, nthreads: int
) -> list[str]: ...
def _shuffle_motifs(motifs: list[Motif], k: int, seed: int) -> list[Motif]: ...
def _sample_sites(motif: Motif, n: int, seed: int) -> list[str]: ...
def _implant_motifs(
    motifs: list[Motif],
    seqs: list[str],
    n_per_seq: int,
    rate: float | None,
    positions: list[list[int]] | None,
    both_strands: bool,
    centre_bias: int,
    min_spacing: int,
    max_retries: int,
    seed: int,
) -> tuple[list[str], dict[str, Any], list[str]]: ...
def _motif_finder(
    seqs: list[str],
    negatives: list[str] | None,
    min_width: int,
    max_width: int,
    nmotifs: int,
    hit_pvalue: float,
    stop_pvalue: float,
    qvalue: float,
    dedup_overlap: float,
    rc: bool,
    shuffle_k: int,
    seed: int,
    pseudocount: int,
    nthreads: int,
) -> tuple[list[Motif], dict[str, Any]]: ...
def _enrich(
    motifs: list[Motif],
    sequences: list[str],
    background: list[str] | None,
    pvalue: float,
    test: str,
    rc: bool,
    shuffle_k: int,
    seed: int,
    pseudocount: int,
    nthreads: int,
) -> dict[str, Any]: ...
def _coocc(
    motif: list[int],
    sequence: list[int],
    start: list[int],
    n_motifs: int,
    n_sequences: int,
    max_distance: int | None,
    min_coocc: int,
    pseudocount: int,
    self_pairs: bool,
    nthreads: int,
    names: list[str],
) -> ArrowTable: ...
def _peaks(
    motif: list[int],
    sequence: list[int],
    centre: list[float],
    score: list[float],
    seq_length: int,
    mode: str,
    min_window: int,
    max_window: int | None,
    window_step: int,
    position_step: int,
    nthreads: int,
) -> dict[str, Any]: ...
def _cutree(merge: list[tuple[int, int]], height: list[float], k: int | None, h: float | None) -> list[int]: ...
def _display_matrix(motif: Motif, use_type: str, relative_entropy: bool) -> list[list[float]]: ...
def _anchor_layout(
    motifs: list[Motif], anchor: int, min_overlap: int, rc: bool, nthreads: int
) -> tuple[list[tuple[int, bool]], int]: ...
def _proximity(
    hits: tuple[list[int], list[int], list[int], list[str]],
    motif: list[int],
    score: list[float],
    anchors: tuple[list[int], list[int], list[int], list[str]],
    universe: tuple[list[int], list[int], list[int], list[str]],
    chrom_lengths: list[int | None],
    method: str,
    count: str,
    orientation: str,
    dist_thresholds: list[int] | None,
    n_perm: int,
    perm_per_chrom: bool,
    ignore_strand: bool,
    seed: int,
    nthreads: int,
) -> dict[str, Any]: ...
def _gc_fraction(seqs: list[str]) -> list[float]: ...
def _match_bkg(
    targets: list[str],
    universe: list[str],
    covariates: list[tuple[str, list[float], list[float], int, str]],
    n_per_target: int,
    n_bins_gc: int,
    n_bins_length: int,
    unique: bool,
    seed: int,
) -> tuple[list[int], int]: ...
def _genome_windows(
    chrom_lengths: list[int],
    target_widths: list[int],
    n_candidates: int,
    exclude: tuple[list[int], list[int], list[int], list[str]],
    seed: int,
) -> list[tuple[int, int, int]]: ...

def _pcm_to_ppm(pos: list[float], pseudocount: float) -> list[float]: ...
def _ppm_to_pcm(pos: list[float], nsites: float) -> list[float]: ...
def _ppm_to_pwm(pos: list[float], bkg: list[float], pseudocount: float, nsites: float) -> list[float]: ...
def _pwm_to_ppm(pos: list[float], bkg: list[float]) -> list[float]: ...
def _ppm_to_icm(pos: list[float], bkg: list[float], relative_entropy: bool) -> list[float]: ...
def _icm_to_ppm(pos: list[float]) -> list[float]: ...
def _motif_range(motif: Motif, allow_nonfinite: bool) -> tuple[float, float]: ...
def _motif_score(
    motif: Motif, thresholds: list[float], threshold_type: str, allow_nonfinite: bool
) -> list[float]: ...
def _score_match(motif: Motif, words: list[str], allow_nonfinite: bool) -> list[float]: ...
def _get_scores(motif: Motif, allow_nonfinite: bool) -> list[float]: ...
def _get_matches(motif: Motif, score: float, allow_nonfinite: bool) -> list[str]: ...
def _prob_match(motif: Motif, words: list[str], allow_zero: bool) -> list[float]: ...
def _prob_match_bkg(bkg: list[tuple[str, float]], words: list[str]) -> list[float]: ...
def _compare_columns(
    x: list[float], y: list[float], method: str, bkg1: list[float], bkg2: list[float],
    nsites1: float, nsites2: float,
) -> float: ...
def _average_ic(motifs: list[Motif], average: str) -> list[float]: ...
def _add_gap(motif: Motif, gaploc: list[int] | None, mingap: list[int], maxgap: list[int]) -> Motif: ...
def _ungap(motif: Motif, delete: bool) -> Motif: ...
def _switch_alph(motif: Motif) -> Motif: ...
def _mask_seqs(seqs: list[str], pattern: str, rc: bool, letter: str, alphabet: str) -> list[str]: ...
def _mask_ranges(seq: str, ranges: list[tuple[int, int]], letter: str) -> str: ...
def _meme_alph(
    core: str, complements: str | None, ambiguity: list[tuple[str, str]], like: str | None,
    name: str | None, letter_names: list[tuple[str, str]], colours: list[tuple[str, str]],
) -> str: ...
def _log_string_pval(pval: str) -> float: ...
