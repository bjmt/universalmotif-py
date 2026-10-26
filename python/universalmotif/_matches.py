"""Sparse sequence by motif matrices of motif matches, for single-cell ATAC
analyses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique, _sequence_pairs, _warn_unreachable
from ._api import _check_motif
from ._core import Motif
from ._packed import PackedSequences

_DTYPES = {"matches": np.uint8, "counts": np.uint32, "scores": np.float64}


def _scipy_sparse():
    try:
        import scipy.sparse as sp
    except ImportError as e:
        raise ImportError(
            "motif match matrices need scipy; install it with `pip install universalmotif[scipy]`"
        ) from e
    return sp


@dataclass(frozen=True)
class MotifMatches:
    """A sparse matrix of motif matches in sequences, from
    [`match_motifs()`][universalmotif.match_motifs].

    ``matrix`` is a SciPy CSR array with one row per sequence and one column
    per motif, named by ``sequences`` and ``motifs``. Its values are those
    asked for with ``value``: 1 for a match, the number of hits, or the best
    score, and pairs without a hit are left out.

    Examples
    --------
    >>> import universalmotif as um
    >>> matches = um.match_motifs(um.example_motifs()[3:6], um.example_sequences())
    >>> matches
    MotifMatches(100 sequences x 3 motifs, 32 matches)
    >>> matches.matrix.sum(axis=0).tolist()
    [12, 3, 17]
    """

    matrix: Any
    sequences: list[str]
    motifs: list[str]
    value: str

    def __repr__(self) -> str:
        n, m = self.matrix.shape
        return f"MotifMatches({n} sequences x {m} motifs, {self.matrix.nnz} {self.value})"

    def to_polars(self) -> pl.DataFrame:
        """The matches as a long DataFrame with columns ``sequence``,
        ``motif`` and ``value``, one row per sequence and motif with a hit.

        Examples
        --------
        >>> import universalmotif as um
        >>> matches = um.match_motifs(um.example_motifs()[3:6], um.example_sequences())
        >>> matches.to_polars().head(2).rows()
        [('target_004', 'W-box', 1.0), ('target_005', 'W-box', 1.0)]
        """
        coo = self.matrix.tocoo()
        order = np.lexsort((coo.col, coo.row))
        seqs = pl.Series(list(self.sequences), dtype=pl.String)
        motifs = pl.Series(list(self.motifs), dtype=pl.String)
        return pl.DataFrame(
            {
                "sequence": seqs.gather(coo.row[order]),
                "motif": motifs.gather(coo.col[order]),
                "value": coo.data[order].astype(float),
            },
            schema={"sequence": pl.String, "motif": pl.String, "value": pl.Float64},
        )

    def to_anndata(self):
        """The matches as an AnnData object, with the sequences as
        observations and the motifs as variables. Needs the ``anndata``
        extra.

        Examples
        --------
        >>> import universalmotif as um
        >>> matches = um.match_motifs(um.example_motifs()[3:6], um.example_sequences())
        >>> adata = matches.to_anndata()  # doctest: +SKIP
        >>> adata.shape  # doctest: +SKIP
        (100, 3)
        """
        try:
            import anndata as ad
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "to_anndata needs anndata; install it with `pip install universalmotif[anndata]`"
            ) from e
        return ad.AnnData(
            X=self.matrix,
            obs=pd.DataFrame(index=pd.Index(self.sequences, dtype=str)),
            var=pd.DataFrame(index=pd.Index(self.motifs, dtype=str)),
        )


def match_motifs(
    motifs: Motif | Iterable[Motif],
    sequences: str | Sequence[str] | Mapping[str, str] | PackedSequences,
    pvalue: float = 5e-5,
    value: str = "matches",
    rc: bool = True,
    no_overlaps: bool = False,
    use_gaps: bool = True,
    nthreads: int = 1,
) -> MotifMatches:
    """Which motifs match which sequences, as a sparse matrix.

    This is the motif annotation used by chromVAR and similar single-cell
    ATAC methods: the sequences, usually peaks, are scanned as by
    [`scan_sequences()`][universalmotif.scan_sequences] with the ``pvalue``
    cutoff, and only a summary of the hits is kept for each sequence and
    motif, so that hundreds of thousands of peaks and hundreds of motifs
    take no more memory than the matrix itself. ``value`` is ``"matches"``
    for 1 where a motif has a hit, ``"counts"`` for the number of hits, or
    ``"scores"`` for the best score of any hit. Palindromic sites are hit on
    both strands, so ``no_overlaps`` drops overlapping hits of each motif,
    on either strand, before they are counted, keeping the one with the
    lowest P-value.

    Needs the ``scipy`` extra.

    Returns
    -------
    MotifMatches
        The matrix with one row per sequence and one column per motif, and
        their names.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()
    >>> counts = um.match_motifs(motifs, um.example_sequences(), value="counts")
    >>> counts.matrix.shape, counts.motifs[3]
    ((100, 12), 'G-box')
    >>> int(counts.matrix[:, 3].sum()), int(counts.matrix[:, 3].max())
    (18, 2)

    The G-box is close to a palindrome, so some of its sites are hit on both
    strands, which ``no_overlaps`` counts once:

    >>> once = um.match_motifs(motifs, um.example_sequences(), value="counts", no_overlaps=True)
    >>> int(once.matrix[:, 3].sum()), int(once.matrix[:, 3].max())
    (12, 1)
    """
    sp = _scipy_sparse()
    if value not in _DTYPES:
        raise ValueError('value must be "matches", "counts" or "scores"')
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("no motifs given")
    _warn_unreachable(ms, pvalue)
    overlaps = ("pvalue", True, False) if no_overlaps else None
    if isinstance(sequences, PackedSequences):
        names = list(sequences)
        args = (None, sequences._inner)
    else:
        names, seqs = _sequence_pairs(sequences)
        args = (seqs, None)
    n, m, rows, cols, vals = _core._motif_matches(
        ms, *args, pvalue, rc, overlaps, nthreads, use_gaps, value
    )
    matrix = sp.csr_array((vals.astype(_DTYPES[value]), (rows, cols)), shape=(n, m))
    return MotifMatches(matrix, names, _make_unique([x.name for x in ms]), value)


def add_motif_matches(
    adata: Any,
    motifs: Motif | Iterable[Motif],
    sequences: Sequence[str] | Mapping[str, str] | None = None,
    key: str = "motif_match",
    names_key: str = "motif_name",
    dense: bool = True,
    **kwargs: Any,
) -> None:
    """Add motif matches to an AnnData object of peaks, as pychromVAR expects.

    ``adata`` has the peaks as its variables, as in a cells by peaks matrix.
    Their sequences come from ``sequences``, either a list in the order of
    the peaks or a mapping from peak names, and by default from
    ``adata.uns["peak_seq"]``, where pychromVAR's ``add_peak_seq()`` keeps
    them. The peaks by motifs matrix from
    [`match_motifs()`][universalmotif.match_motifs], given ``kwargs``, is
    stored in ``adata.varm[key]``, as a dense array unless ``dense`` is
    false, and the motif names in ``adata.uns[names_key]``. With the
    defaults, the result can go straight to pychromVAR's
    ``compute_deviations()``.

    Needs the ``scipy`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.add_motif_matches(adata, um.example_motifs())  # doctest: +SKIP
    >>> adata.varm["motif_match"].shape  # doctest: +SKIP
    (100, 12)
    """
    if sequences is None:
        if "peak_seq" not in adata.uns:
            raise ValueError('give the peak sequences, or store them in adata.uns["peak_seq"]')
        sequences = list(adata.uns["peak_seq"])
    if isinstance(sequences, Mapping):
        missing = [p for p in adata.var_names if p not in sequences]
        if missing:
            raise ValueError(f"no sequence for the peaks {', '.join(map(str, missing[:5]))}")
        seqs = [sequences[p] for p in adata.var_names]
    else:
        seqs = list(sequences)
    if len(seqs) != adata.n_vars:
        raise ValueError(f"{len(seqs)} sequences for {adata.n_vars} peaks")
    matches = match_motifs(motifs, seqs, **kwargs)
    adata.varm[key] = matches.matrix.toarray() if dense else matches.matrix
    adata.uns[names_key] = list(matches.motifs)


__all__ = ["MotifMatches", "add_motif_matches", "match_motifs"]
