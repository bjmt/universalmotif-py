"""Motif comparison, merging and trees."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique
from ._api import _check_motif, summarise_motifs
from ._core import Motif


def _motif_list(motifs) -> list[Motif]:
    if isinstance(motifs, Motif):
        return [motifs]
    return [_check_motif(m) for m in motifs]


def _bkg4(bkg) -> list[float] | None:
    if bkg is None:
        return None
    if isinstance(bkg, Mapping):
        last = "U" if "U" in bkg else "T"
        return [float(bkg[k]) for k in ("A", "C", "G", last)]
    return [float(b) for b in bkg]


def compare_motifs(
    motifs: Iterable[Motif],
    compare_to: int | str | Sequence[int | str] | None = None,
    qvalue: float = 0.1,
    min_overlap: int = 5,
    rc: bool = True,
    null: str = "empirical",
    bkg: Sequence[float] | Mapping[str, float] | None = None,
    matrix_out: str = "score",
    nthreads: int = 1,
):
    """Compare DNA or RNA motifs.

    This is the port of universalmotif's ``compare_motifs_lite()``. Each
    target is slid along the query (and its reverse complement, with
    ``rc``) and the alignment with the highest sum of column correlations
    is kept, among those overlapping at least ``min_overlap`` columns.
    Its P-value comes from the exact distribution of that sum under a null
    model of columns (``"empirical"``: the columns of all the motifs given;
    ``"parametric"``: a Dirichlet grid around ``bkg``), Bonferroni-corrected
    for the alignments tried. Q-values are Benjamini-Hochberg adjusted
    within each query.

    Returns
    -------
    Without ``compare_to``, an all-against-all numpy array (rows are
    queries, columns targets) of the mean column correlation, P-value or
    q-value (``matrix_out``). With ``compare_to`` (motif indices, 0-based,
    or names), a DataFrame with one row per significant (``qvalue``) match:
    ``subject``, ``subject_i``, ``target``, ``target_i``, ``offset``,
    ``strand``, ``overlap``, ``score`` (mean column correlation),
    ``subject_consensus``, ``target_consensus``, ``pvalue`` and ``qvalue``,
    ordered by q-value.
    """
    ms = _motif_list(motifs)
    names = _make_unique([m.name for m in ms])
    if matrix_out not in ("score", "pvalue", "qvalue"):
        raise ValueError("matrix_out must be 'score', 'pvalue' or 'qvalue'")
    if not (0 < qvalue <= 1):
        raise ValueError("qvalue must be in (0, 1]")
    if min_overlap < 1:
        raise ValueError("min_overlap must be at least 1")
    b = _bkg4(bkg)

    if compare_to is None:
        if qvalue != 0.1:
            warnings.warn("qvalue is ignored without compare_to", stacklevel=2)
        d = _core._compare_matrix(ms, min_overlap, rc, null, b, matrix_out != "score", nthreads)
        return np.array(d[matrix_out], dtype=float)

    targets = [compare_to] if isinstance(compare_to, (int, str)) else list(compare_to)
    queries = []
    for t in targets:
        if isinstance(t, str):
            if t not in names:
                raise ValueError(f"no motif named {t!r}")
            queries.append(names.index(t))
        else:
            if not 0 <= int(t) < len(ms):
                raise ValueError(f"compare_to index {t} is out of range")
            queries.append(int(t))
    df = pl.DataFrame(_core._compare_table(ms, queries, min_overlap, rc, null, b, nthreads, names))
    df = df.filter((pl.col("qvalue") <= qvalue) & (pl.col("overlap") > 0))
    return df.sort(["qvalue", "pvalue", "target_i"], maintain_order=True)


def merge_motifs(
    motifs: Iterable[Motif],
    min_overlap: int = 5,
    rc: bool = True,
    new_name: str | None = None,
    weighted: bool = False,
    nthreads: int = 1,
) -> Motif:
    """Merge DNA or RNA motifs into one PPM.

    This is the port of universalmotif's ``merge_motifs_lite()``. The motif
    with the highest total IC is the anchor; the others are aligned to it
    (reverse complemented if that fits better, with ``rc``) and the aligned
    probabilities averaged, weighted by ``nsites`` if ``weighted``. Motifs
    that cannot overlap the anchor by ``min_overlap`` columns are left out.
    The default name joins the merged motifs' names.
    """
    return _core._merge_motifs(_motif_list(motifs), min_overlap, rc, new_name, weighted, nthreads)


def merge_similar(
    motifs: Iterable[Motif],
    qvalue: float = 0.01,
    null: str = "empirical",
    min_overlap: int = 5,
    rc: bool = True,
    bkg: Sequence[float] | Mapping[str, float] | None = None,
    weighted: bool = False,
    return_clusters: bool = False,
    nthreads: int = 1,
):
    """Cluster similar DNA or RNA motifs and merge each cluster.

    This is the port of universalmotif's ``merge_similar_lite()``. Motifs
    are linked when :func:`compare_motifs` gives them a q-value of at most
    ``qvalue`` in either direction, clusters are the connected groups, and
    each cluster of several motifs is merged with :func:`merge_motifs`.

    Returns one motif per cluster, or with ``return_clusters`` a summary
    DataFrame of the input motifs with ``motif_i`` and ``cluster`` columns
    (both 0-based).
    """
    ms = _motif_list(motifs)
    clusters, merged = _core._merge_similar(
        ms, qvalue, null, min_overlap, rc, _bkg4(bkg), weighted, nthreads
    )
    if return_clusters:
        if not ms:
            return pl.DataFrame(schema={"name": pl.String, "motif_i": pl.Int64, "cluster": pl.Int64})
        return summarise_motifs(ms).with_columns(
            pl.Series("motif_i", list(range(len(ms))), dtype=pl.Int64),
            pl.Series("cluster", clusters, dtype=pl.Int64),
        )
    return merged


@dataclass(frozen=True)
class MotifTree:
    """A clustering tree of motifs.

    Attributes
    ----------
    labels
        Tip labels, in input order.
    merge
        Merge steps as in R's ``hclust``: row ``i`` joins two items, where
        ``-k`` is input ``k`` (1-based) and ``k`` the cluster from row ``k``.
    height
        Height of each merge.
    order
        Input indices (0-based) in dendrogram order.
    linkage
        The same tree as a SciPy linkage matrix, for
        ``scipy.cluster.hierarchy`` (``dendrogram``, ``fcluster`` and so on).
    newick
        The tree in Newick format, as R's ape package writes it.
    method
        Agglomeration method.
    motifs
        The motifs at the tips, for trees made from motifs.
    """

    labels: list[str]
    merge: np.ndarray = field(repr=False)
    height: np.ndarray = field(repr=False)
    order: list[int] = field(repr=False)
    linkage: np.ndarray = field(repr=False)
    newick: str = field(repr=False)
    method: str = "complete"
    motifs: list[Motif] | None = field(default=None, repr=False)

    def write_newick(self, file) -> None:
        """Write the Newick string to a file."""
        with open(file, "w") as f:
            f.write(self.newick + "\n")

    def __len__(self) -> int:
        return len(self.labels)

    def cutree(self, k: int | None = None, h: float | None = None) -> np.ndarray:
        """Cut the tree into ``k`` groups, or at height ``h``, as R's
        ``cutree()``.

        Returns one group per motif, numbered from 0 in the order the groups
        first appear among the motifs. Cutting by height needs increasing
        merge heights, which the centroid and median methods do not
        guarantee.
        """
        if (k is None) == (h is None):
            raise ValueError("give exactly one of k and h")
        merge = [tuple(int(v) for v in row) for row in self.merge]
        ids = _core._cutree(merge, [float(x) for x in self.height], k, None if h is None else float(h))
        return np.array(ids, dtype=np.int64)

    def plot(
        self,
        layout: str = "rectangular",
        labels: bool = True,
        groups=None,
        tip_size=None,
        tip_logos: bool = False,
        branch_length: bool = False,
        legend: bool = True,
        colours=None,
        open_angle: float = 90,
        tip_logo_width: float | None = None,
        tip_logo_height: float = 0.9,
        text_size: float = 10,
        ax=None,
    ):
        """Draw the tree with matplotlib (as universalmotif's
        ``motif_tree_lite()`` draws it with ggtree).

        Parameters
        ----------
        layout
            ``"rectangular"``, ``"slanted"``, ``"circular"`` or ``"fan"`` (a
            circle with an ``open_angle`` gap, in degrees).
        labels
            Label the tips with ``self.labels``.
        groups
            Colour branches by group: a motif attribute such as
            ``"family"``, or one value per motif. A branch takes a group's
            colour when every motif below it is in that group; ``colours``
            maps groups to colours.
        tip_size
            Size the tips by a motif attribute (``"icscore"``, or
            ``"pval"``, ``"qval"`` and ``"eval"`` as their -log10) or by one
            number per motif.
        tip_logos
            Draw each motif's logo at its tip, aligned as in
            :func:`view_motifs` (rectangular and slanted layouts only).
            ``tip_logo_width`` is the width of one position in tree units
            and ``tip_logo_height`` the logo height in tip spacings.
        branch_length
            Draw branch lengths (half the merge heights, as the Newick
            string); by default tips are aligned, as a cladogram.
        ax
            Axes to draw on; a new figure by default.

        Returns
        -------
        The matplotlib axes.
        """
        from ._plot import plot_tree

        return plot_tree(self, layout, labels, groups, tip_size, tip_logos, branch_length, legend,
                         colours, open_angle, tip_logo_width, tip_logo_height, text_size, ax)


def _tree(d: dict, labels: list[str], method: str, motifs: list[Motif] | None = None) -> MotifTree:
    return MotifTree(
        labels=labels,
        merge=np.array(d["merge"], dtype=np.int64).reshape(-1, 2),
        height=np.array(d["height"], dtype=float),
        order=list(d["order"]),
        linkage=np.array(d["linkage"], dtype=float).reshape(-1, 4),
        newick=d["newick"],
        method=method,
        motifs=motifs,
    )


def motif_tree(
    motifs,
    min_overlap: int = 6,
    rc: bool = True,
    method: str = "complete",
    labels: str | Sequence[str] = "name",
    nthreads: int = 1,
) -> MotifTree:
    """Cluster motifs into a tree.

    This builds the tree that universalmotif's ``motif_tree_lite()`` draws:
    DNA or RNA motifs are compared with :func:`compare_motifs`, mean column
    correlations become distances as ``(1 - score) / 2``, and the distances
    are clustered with R's ``hclust`` algorithm (``method`` is any of its
    methods: ``"complete"``, ``"average"``, ``"single"``, ``"ward.D"``,
    ``"ward.D2"``, ``"mcquitty"``, ``"median"`` or ``"centroid"``).

    ``motifs`` can also be a square distance matrix (only the lower triangle
    is used). ``labels`` is a motif attribute to label tips with (such as
    ``"name"``, ``"altname"`` or ``"family"``), or a list of labels.
    Duplicate motif names get ``.1``, ``.2`` and so on, as in R.
    """
    if isinstance(motifs, np.ndarray) or (
        isinstance(motifs, Sequence) and motifs and not isinstance(motifs[0], Motif)
    ):
        dist = np.asarray(motifs, dtype=float)
        if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
            raise ValueError("a distance matrix must be square")
        n = dist.shape[0]
        labs = [str(i + 1) for i in range(n)] if isinstance(labels, str) else list(labels)
        if len(labs) != n:
            raise ValueError("give one label per row of the distance matrix")
        return _tree(_core._hclust(dist.tolist(), method, labs), labs, method)

    ms = _motif_list(motifs)
    if isinstance(labels, str):
        labs = [getattr(m, labels) for m in ms]
        labs = [("" if v is None else str(v)) for v in labs]
        if labels == "name":
            labs = _make_unique(labs)
    else:
        labs = list(labels)
    if len(labs) != len(ms):
        raise ValueError("give one label per motif")
    d = _core._motif_tree(ms, min_overlap, rc, method, labs, nthreads)
    return _tree(d, labs, method, ms)


__all__ = ["MotifTree", "compare_motifs", "merge_motifs", "merge_similar", "motif_tree"]
