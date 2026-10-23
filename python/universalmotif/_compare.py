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
    complete: bool = False,
):
    """Compare motifs.

    Motifs of any alphabet can be compared. Each target is slid along the
    query, and with ``rc`` along the reverse complement of a DNA or RNA
    query too, and the alignment with the highest sum of column correlations
    is kept, among those overlapping at least ``min_overlap`` columns, or all of
    the shorter motif's columns when it is narrower than that. Column
    correlations are Pearson correlations of the probabilities. Each P-value
    comes from the exact distribution of that sum under a null model of
    columns, Bonferroni-corrected for the alignments tried. The
    ``"empirical"`` null uses the columns of all the motifs given, while the
    ``"parametric"`` null, for DNA and RNA only, uses a Dirichlet grid around
    ``bkg``. Q-values are Benjamini-Hochberg adjusted within each query.

    By default only the overlap counts, so a short target that matches part
    of a long query can score as well as one that covers all of it. With
    ``complete``, whole queries are scored instead, as TOMTOM does by
    default: each query column's correlations are measured against the
    median of its correlations with the null columns, so that a typical
    column scores zero and query columns hanging over the ends of the target
    count as typical, and the P-value of the best alignment is that of the
    best of all the alignments tried. This favours alignments covering more
    of the query, and comparisons then depend on which motif is the query,
    so P-value matrices need not be symmetric. Either way, ``score`` is the
    mean column correlation over the overlap.

    Returns
    -------
    numpy.ndarray | polars.DataFrame
        Without ``compare_to``, an all-against-all numpy array with queries
        as rows and targets as columns, holding the mean column correlation,
        P-value or q-value as chosen by ``matrix_out``. With ``compare_to``,
        given as 0-based motif indices or as names, a DataFrame ordered by
        q-value with one row per match passing ``qvalue`` and the columns
        ``subject``, ``subject_i``, ``target``, ``target_i``, ``offset``,
        ``strand``, ``overlap``, ``score``, ``subject_consensus``,
        ``target_consensus``, ``pvalue`` and ``qvalue``, where ``score`` is
        the mean column correlation.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:4]
    >>> um.compare_motifs(motifs).round(2).tolist()
    [[1.0, 0.33, 0.11, 0.19], [0.33, 1.0, 0.46, 0.25], [0.11, 0.46, 1.0, 0.58], [0.19, 0.25, 0.58, 1.0]]

    With ``compare_to``, each match passing ``qvalue`` gets a row:

    >>> wbox = um.example_motifs()[5]
    >>> site = um.create_motif(["CTTTGACCAA", "GTTTGACCAA", "CTTTGACTAA"], name="site")
    >>> hits = um.compare_motifs([wbox, site], compare_to="W-box")
    >>> hits["target"].to_list(), hits["score"].round(2).to_list()
    (['W-box', 'site'], [1.0, 0.65])
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
        d = _core._compare_matrix(ms, min_overlap, rc, null, b, matrix_out != "score", nthreads,
                                  complete)
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
    df = pl.DataFrame(
        _core._compare_table(ms, queries, min_overlap, rc, null, b, nthreads, names, complete)
    )
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
    """Merge motifs into one PPM.

    Motifs of any alphabet can be merged. The motif with the highest total
    IC is the anchor. The others are aligned to it, with DNA and RNA motifs
    reverse complemented under ``rc`` where that fits better, and the aligned
    probabilities are averaged, weighted by ``nsites`` if ``weighted``. Motifs
    that cannot overlap the anchor by ``min_overlap`` columns are left out,
    with a warning.
    The default name joins the merged motifs' names.

    Examples
    --------
    >>> import universalmotif as um
    >>> v1 = um.create_motif(["CTTTGACCAA", "GTTTGACCAA", "CTTTGACTAA"], name="v1")
    >>> v2 = um.create_motif(["TTTGACCAAT", "TTTGACTAAT", "TTTGACCAAA"], name="v2")
    >>> merged = um.merge_motifs([v1, v2], new_name="W-box merged")
    >>> merged.name, merged.consensus, merged.type
    ('W-box merged', 'STTTGACYAAW', 'PPM')
    """
    ms = _motif_list(motifs)
    merged, used = _core._merge_motifs(ms, min_overlap, rc, new_name, weighted, nthreads)
    if len(used) < len(ms):
        left = [ms[i].name for i in sorted(set(range(len(ms))) - set(used))]
        warnings.warn(
            f"{len(left)} motif{'s' if len(left) > 1 else ''} could not overlap the anchor by "
            f"{min_overlap} columns and {'were' if len(left) > 1 else 'was'} left out: "
            f"{', '.join(left[:5])}{', ...' if len(left) > 5 else ''}",
            stacklevel=2,
        )
    return merged


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
    """Cluster similar motifs and merge each cluster.

    Motifs are linked when [`compare_motifs()`][universalmotif.compare_motifs]
    gives them a q-value of at most ``qvalue`` in either direction, clusters
    are the connected groups, and each cluster of several motifs is merged with
    [`merge_motifs()`][universalmotif.merge_motifs]. Since a single link is
    enough, motifs that are only alike in part, such as those of one family of
    transcription factors, can chain into one large cluster; cutting a
    [`motif_tree()`][universalmotif.motif_tree] at a fixed height avoids that.

    Returns one motif per cluster, or with ``return_clusters`` a summary
    DataFrame of the input motifs with 0-based ``motif_i`` and ``cluster``
    columns.

    Examples
    --------
    >>> import universalmotif as um
    >>> v1 = um.create_motif(["CTTTGACCAA", "GTTTGACCAA", "CTTTGACTAA"], name="v1")
    >>> v2 = um.create_motif(["TTTGACCAAT", "TTTGACTAAT", "TTTGACCAAA"], name="v2")
    >>> gbox = um.example_motifs()[3]
    >>> [m.name for m in um.merge_similar([v1, v2, gbox])]
    ['merged_v1+v2', 'G-box']
    >>> um.merge_similar([v1, v2, gbox], return_clusters=True)["cluster"].to_list()
    [0, 0, 1]
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
        Merge steps, one per row. Row ``i`` joins two items, where ``-k`` is
        the ``k``-th input and ``k`` is the cluster formed in row ``k``, both
        counting from 1.
    height
        Height of each merge.
    order
        0-based input indices in dendrogram order.
    linkage
        The same tree as a SciPy linkage matrix, for use with
        ``scipy.cluster.hierarchy`` functions such as ``dendrogram`` and
        ``fcluster``.
    newick
        The tree in Newick format. Branch lengths are half the differences in
        merge height, written with up to 10 significant digits, and whitespace
        in tip labels becomes ``_`` while Newick punctuation becomes ``-``.
    method
        Agglomeration method.
    motifs
        The motifs at the tips, for trees made from motifs.

    Examples
    --------
    >>> import universalmotif as um
    >>> dist = [[0.0, 0.1, 0.8], [0.1, 0.0, 0.7], [0.8, 0.7, 0.0]]
    >>> tree = um.motif_tree(dist, labels=["a", "b", "c"])
    >>> tree.merge.tolist(), tree.height.tolist()
    ([[-1, -2], [-3, 1]], [0.1, 0.8])
    >>> tree.newick
    '(c:0.4,(a:0.05,b:0.05):0.35);'
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
        """Write the Newick string to a file.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import universalmotif as um
        >>> tree = um.motif_tree(um.example_motifs()[:3])
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "motifs.nwk"
        ...     tree.write_newick(path)
        ...     path.read_text().strip() == tree.newick
        True
        """
        with open(file, "w") as f:
            f.write(self.newick + "\n")

    def __len__(self) -> int:
        return len(self.labels)

    def cutree(self, k: int | None = None, h: float | None = None) -> np.ndarray:
        """Cut the tree into ``k`` groups, or at height ``h``.

        Returns one group per motif, numbered from 0 in the order the groups
        first appear among the motifs. Cutting by height needs increasing
        merge heights, which the centroid and median methods do not
        guarantee.

        Examples
        --------
        >>> import universalmotif as um
        >>> dist = [[0.0, 0.1, 0.8], [0.1, 0.0, 0.7], [0.8, 0.7, 0.0]]
        >>> tree = um.motif_tree(dist, labels=["a", "b", "c"])
        >>> tree.cutree(k=2).tolist()
        [0, 0, 1]
        >>> tree.cutree(h=0.05).tolist()
        [0, 1, 2]
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
        """Draw the tree with matplotlib.

        Parameters
        ----------
        layout
            ``"rectangular"``, ``"slanted"``, ``"circular"`` or ``"fan"``,
            which is a circle with a gap of ``open_angle`` degrees.
        labels
            Label the tips with ``self.labels``.
        groups
            Colour branches by group: a motif attribute such as
            ``"family"``, one value per motif in the order of the tree's
            labels, or a mapping from labels to values. A branch takes a group's
            colour when every motif below it is in that group; ``colours``
            maps groups to colours.
        tip_size
            Size the tips by one number per motif, given as for ``groups``,
            or by a motif attribute,
            either ``"icscore"`` or one of ``"pval"``, ``"qval"`` and
            ``"eval"``, which are taken as their -log10.
        tip_logos
            Draw each motif's logo at its tip, aligned as in
            [`view_motifs()`][universalmotif.view_motifs], in the rectangular
            and slanted layouts only.
            ``tip_logo_width`` is the width of one position in tree units
            and ``tip_logo_height`` the logo height in tip spacings.
        branch_length
            Draw branch lengths, which are half the merge heights as in the
            Newick string; by default tips are aligned, as a cladogram.
        ax
            Axes to draw on; a new figure by default.

        Returns
        -------
        The matplotlib axes.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import universalmotif as um
        >>> tree = um.motif_tree(um.example_motifs())
        >>> ax = tree.plot(groups=tree.cutree(k=3), legend=False)
        >>> type(ax).__name__
        'Axes'
        >>> plt.close("all")
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

    Motifs are compared with [`compare_motifs()`][universalmotif.compare_motifs], mean column
    correlations become distances as ``(1 - score) / 2``, and the distances
    are clustered hierarchically. The agglomeration ``method`` is one of
    ``"complete"``, ``"average"``, ``"single"``, ``"ward.D"``, ``"ward.D2"``,
    ``"mcquitty"``, ``"median"`` or ``"centroid"``; ``"ward.D"`` applies
    Ward's criterion to the distances as they are and ``"ward.D2"`` to the
    squared distances.

    ``motifs`` can also be a square distance matrix, of which only the lower
    triangle is used. ``labels`` is either a list of labels or a motif
    attribute to label tips with, such as ``"name"``, ``"altname"`` or
    ``"family"``. Duplicate motif names get ``.1``, ``.2`` and so on, while
    the first of each is left as it is.

    Examples
    --------
    >>> import universalmotif as um
    >>> tree = um.motif_tree(um.example_motifs()[:6])
    >>> tree
    MotifTree(labels=['TATA-box', 'AP-1', 'CRE', 'G-box', 'E-box', 'W-box'], method='complete')
    >>> [tree.labels[i] for i in tree.order]
    ['TATA-box', 'AP-1', 'W-box', 'E-box', 'CRE', 'G-box']
    >>> tree.height.round(3).tolist()
    [0.096, 0.211, 0.354, 0.384, 0.447]
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
