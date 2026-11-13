"""Motif comparison, merging and trees."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from . import _core
from ._analysis import _make_unique
from ._api import _bkg_table, _check_motif, summarize_motifs
from ._core import Motif
from ._threads import _nthreads


def _motif_list(motifs) -> list[Motif]:
    if isinstance(motifs, Motif):
        return [motifs]
    return [_check_motif(m) for m in motifs]


def _bkg4(bkg) -> list[float] | None:
    bkg = _bkg_table(bkg)
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
    nthreads: int | None = None,
    complete: bool = False,
    method: str | None = None,
    overhang: str = "ignore",
    sign: str = "match",
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
    P-values are worked out for each query's own columns and alignments, so
    a P-value matrix need not be symmetric even where the scores are.

    With ``method="EUCL"``, columns are compared by the Euclidean distance
    between their probabilities instead, and ``score`` is the mean distance,
    from 0 for identical columns up to the square root of 2, so that lower
    means more alike. Since summed distances favour short overlaps, these
    alignments are chosen by their mean distance, unless ``complete`` or
    background overhangs make every alignment cover the same number of
    columns. Everything else, P-values included, works as it does for
    correlations, with distances counting as negative similarities.

    With ``overhang="background"``, the columns of either motif that hang
    over the ends of the other are compared with ``bkg``, which is uniform
    by default and always for alphabets other than DNA and RNA. Every column
    of both motifs is then scored against what faces it, a column of the
    other motif or the background, so that each alignment counts as many
    column scores as the two motifs have columns. The alignment with the
    highest total is kept, ``score`` is the mean of these column scores,
    and P-values come from the exact distribution of the total, in which
    the target's overhanging columns are drawn from the null as well. A
    partial overlap thus pays for however much its overhanging columns
    differ from the background. With correlations and a uniform background,
    overhanging columns score 0, which leaves alignments and P-values much
    as they are and only lowers the scores of partial overlaps. Background
    overhangs cannot be combined with ``complete``.

    By default only the overlap counts, so a short target that matches part
    of a long query can score as well as one that covers all of it. With
    ``complete``, whole queries are scored instead, as TOMTOM does by
    default: each query column's correlations are measured against the
    median of its correlations with the null columns, so that a typical
    column scores zero and query columns hanging over the ends of the target
    count as typical, and the P-value of the best alignment is that of the
    best of all the alignments tried. This favours alignments covering more
    of the query, and scores then depend on which motif is the query too.
    Either way, ``score`` is the mean column correlation or distance over
    the overlap, as it is whenever overhangs are ignored.

    The gaps of gapped motifs are not used: their columns are compared as if
    they followed one another, so a gapped motif scores as its ungapped
    columns would.

    To search a database with a query motif, such as to name a motif from
    [`motif_finder()`][universalmotif.motif_finder] or a TF-MoDISco pattern,
    ``complete=True`` is the better choice, as it picks the same best match
    as TOMTOM for 94% of queries against 78% with the default, as the Notes
    show. All-against-all matrices,
    [`motif_tree()`][universalmotif.motif_tree] and
    [`merge_similar()`][universalmotif.merge_similar] keep the default,
    whose scores do not depend on which motif is the query.

    CWMs, such as the patterns of
    [`read_modisco()`][universalmotif.read_modisco], are compared by default
    with ``method="COSINE"`` when every motif given is one, which keeps the
    sign and magnitude of their contributions; comparing them by column
    correlation turns each column into probabilities of its absolute values,
    so that a negative pattern looks like a positive one and columns
    carrying almost no contribution count as much as the core. The cosine of
    two aligned CWMs is the sum of the products of facing values divided by
    the product of the two whole matrices' Euclidean norms, as if the
    columns hanging over either end faced zeros, and ranges from -1 to 1.
    ``method="CJACCARD"`` uses the continuous Jaccard similarity of
    TF-MoDISco instead, on matrices scaled to unit L1 norm: the sum of
    ``min(|a|, |b|)``, signed by whether ``a`` and ``b`` agree, over the sum
    of ``max(|a|, |b|)``, which is harsher on differences in magnitude. The
    best alignment is the most similar over both strands and every offset
    with ``min_overlap`` columns in common. A pattern scores about 0 against
    its opposite, a negative pattern with the same letters, unless
    ``sign="ignore"`` compares absolute values. Cosine P-values come from a
    null of targets with columns drawn from all the CWMs given, in either
    orientation, Bonferroni-corrected for the alignments tried, as for
    correlations; continuous Jaccard has scores only, so with ``compare_to``
    every match is kept, best first. ``null``, ``bkg``, ``complete`` and
    ``overhang`` do not apply to CWM similarity. To match CWMs against a
    database of PPMs, compare their PPMs, such as those from
    ``read_modisco(type="PPM")``, by correlation.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

    Returns
    -------
    numpy.ndarray | polars.DataFrame
        Without ``compare_to``, an all-against-all numpy array with queries
        as rows and targets as columns, holding the score, P-value or
        q-value as chosen by ``matrix_out``. Scores from ``method="EUCL"``
        are mean distances, which
        [`motif_tree()`][universalmotif.motif_tree] can cluster as they
        are. With ``compare_to``,
        given as 0-based motif indices or as names, a DataFrame ordered by
        q-value with one row per match passing ``qvalue`` and the columns
        ``subject``, ``subject_i``, ``target``, ``target_i``, ``offset``,
        ``strand``, ``overlap``, ``score``, ``subject_consensus``,
        ``target_consensus``, ``pvalue`` and ``qvalue``.

    Notes
    -----
    Scoring alignments by summed Pearson correlations, with P-values from
    a null of database columns, is close to what TOMTOM does with its
    Pearson distance, so the two largely agree. The script
    ``scripts/compare_tomtom.py`` in the repository runs both on the same
    motifs. With 72 queries, 60 from CIS-BP and the 12 example motifs,
    against the 805 plant motifs of JASPAR 2024:

    | | ``complete=True``, TOMTOM default | default, TOMTOM ``-incomplete-scores`` |
    | --- | --- | --- |
    | log10 P-values, correlation over all 57,960 pairs | 0.999 | 0.988 |
    | P-value rank correlation within queries, median | 0.999 | 0.927 |
    | Each query's five best matches shared | 95% | 88% |
    | Same best match | 94% | 78% |
    | Strong matches on the same strand | 99% | 99% |

    Where the best match differs, the two choices are near ties, a tenth to
    a fifth of an order of magnitude apart.

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

    Euclidean distances, here with overhangs compared with the background,
    are lower for more alike motifs:

    >>> um.compare_motifs(motifs, method="EUCL").round(2).tolist()[0]
    [0.0, 0.72, 0.74, 0.65]
    >>> hits = um.compare_motifs([wbox, site], compare_to="W-box", method="EUCL",
    ...                          overhang="background")
    >>> hits["score"].round(2).to_list()
    [0.0, 0.38]
    """
    nthreads = _nthreads(nthreads)
    ms = _motif_list(motifs)
    names = _make_unique([m.name for m in ms])
    if matrix_out not in ("score", "pvalue", "qvalue"):
        raise ValueError("matrix_out must be 'score', 'pvalue' or 'qvalue'")
    if not (0 < qvalue <= 1):
        raise ValueError("qvalue must be in (0, 1]")
    if min_overlap < 1:
        raise ValueError("min_overlap must be at least 1")
    method = _comparison_method(ms, method)
    cwm = method in _CWM_METHODS
    _check_sign(sign, cwm)
    if cwm:
        if complete or overhang != "ignore" or null != "empirical" or bkg is not None:
            raise ValueError("null, bkg, complete and overhang do not apply to COSINE and CJACCARD")
        if matrix_out != "score" and method == "CJACCARD":
            raise ValueError("CJACCARD has scores only; use COSINE for P-values")
    b = None if cwm else _bkg4(bkg)

    def cwm_rows(queries):
        return pl.DataFrame(_core._compare_cwms(ms, queries, method, sign == "ignore", min_overlap, rc,
                                                nthreads, names))

    if compare_to is None:
        if qvalue != 0.1:
            warnings.warn("qvalue is ignored without compare_to", stacklevel=2)
        if cwm:
            if matrix_out == "score":
                return np.array(_core._cwm_matrix(ms, method, sign == "ignore", min_overlap, rc, nthreads),
                                dtype=float)
            rows = cwm_rows(list(range(len(ms))))
            return rows[matrix_out].to_numpy().reshape(len(ms), len(ms))
        d = _core._compare_matrix(ms, min_overlap, rc, null, b, matrix_out != "score", nthreads,
                                  complete, method, overhang)
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
    if method == "CJACCARD":
        df = cwm_rows(queries).filter(pl.col("overlap") > 0)
        return df.sort(["subject_i", "score", "target_i"], descending=[False, True, False],
                       maintain_order=True)
    if cwm:
        df = cwm_rows(queries)
    else:
        df = pl.DataFrame(
            _core._compare_table(ms, queries, min_overlap, rc, null, b, nthreads, names, complete,
                                 method, overhang)
        )
    df = df.filter((pl.col("qvalue") <= qvalue) & (pl.col("overlap") > 0))
    return df.sort(["qvalue", "pvalue", "target_i"], maintain_order=True)


_CWM_METHODS = ("COSINE", "CJACCARD")


def _comparison_method(ms: list[Motif], method: str | None) -> str:
    """The comparison method: the one given, or by default the cosine when
    every motif is a CWM and column correlations otherwise."""
    if method is None:
        return "COSINE" if ms and all(m.type == "CWM" for m in ms) else "PCC"
    upper = str(method).upper()
    if upper in _CWM_METHODS:
        if not all(m.type == "CWM" for m in ms):
            raise ValueError(f"{upper} compares CWMs; compare other motifs by PCC or EUCL")
        return upper
    return method


def _check_sign(sign: str, cwm: bool) -> None:
    if sign not in ("match", "ignore"):
        raise ValueError("sign must be 'match' or 'ignore'")
    if sign == "ignore" and not cwm:
        raise ValueError("sign='ignore' applies to COSINE and CJACCARD only")


def merge_motifs(
    motifs: Iterable[Motif],
    min_overlap: int = 5,
    rc: bool = True,
    new_name: str | None = None,
    weighted: bool = False,
    nthreads: int | None = None,
) -> Motif:
    """Merge motifs into one.

    Motifs of any alphabet can be merged. The motif with the highest total
    IC is the anchor. The others are aligned to it, with DNA and RNA motifs
    reverse complemented under ``rc`` where that fits better, and the aligned
    probabilities are averaged, weighted by ``nsites`` if ``weighted``. Motifs
    that cannot overlap the anchor by ``min_overlap`` columns are left out,
    with a warning.

    When every motif is a CWM, the result is a CWM too: motifs are aligned to
    the one with the largest total contribution by cosine similarity,
    keeping their sign, and their aligned contributions are averaged, with
    positions covered by no motif left at 0.

    Gaps are not used: gapped motifs are aligned with their columns running
    together, and the merged motif has no gaps.

    The default name joins the merged motifs' names.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

    Examples
    --------
    >>> import universalmotif as um
    >>> v1 = um.create_motif(["CTTTGACCAA", "GTTTGACCAA", "CTTTGACTAA"], name="v1")
    >>> v2 = um.create_motif(["TTTGACCAAT", "TTTGACTAAT", "TTTGACCAAA"], name="v2")
    >>> merged = um.merge_motifs([v1, v2], new_name="W-box merged")
    >>> merged.name, merged.consensus, merged.type
    ('W-box merged', 'STTTGACYAAW', 'PPM')
    """
    nthreads = _nthreads(nthreads)
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
    linkage: str = "single",
    nthreads: int | None = None,
    method: str | None = None,
    sign: str = "match",
    min_score: float | None = None,
):
    """Cluster similar motifs and merge each cluster.

    Motifs are linked when [`compare_motifs()`][universalmotif.compare_motifs]
    gives them a q-value of at most ``qvalue`` in either direction, and each
    cluster of several motifs is merged with
    [`merge_motifs()`][universalmotif.merge_motifs]. With
    ``linkage="single"``, clusters are the connected groups, so one link to
    any member is enough to join a cluster, and motifs that are only alike in
    part, such as those of one family of transcription factors, can chain
    into one large cluster. With ``linkage="complete"``, every pair of motifs
    in a cluster is linked: the clusters come from complete-linkage
    clustering of the q-values cut at ``qvalue``, which joins the closest
    clusters first and breaks ties by input order, so the result does not
    depend on chance. A cluster of more than two motifs that holds over a
    quarter of them is named in a warning, as a sign that unlike motifs were
    joined.

    When every motif is a CWM, they are linked instead when their similarity,
    by ``method``, ``"COSINE"`` by default or ``"CJACCARD"``, and ``sign``,
    as [`compare_motifs()`][universalmotif.compare_motifs] describes them,
    reaches ``min_score``, and ``qvalue`` and ``null`` are not used. By
    default ``min_score`` is 0.8 for the cosine and 0.45 for continuous
    Jaccard, about where duplicate patterns of one TF-MoDISco run begin:
    among the 60 patterns of an ENCODE K562 ATAC-seq model, pairs that are
    clearly the same motif scored 0.82 to 0.95 by cosine, while only 1% of
    all pairs scored above 0.79. The merged motifs stay CWMs, as
    [`merge_motifs()`][universalmotif.merge_motifs] makes them.

    Returns one motif per cluster, or with ``return_clusters`` a summary
    DataFrame of the input motifs with 0-based ``motif_i`` and ``cluster``
    columns.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

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
    if linkage not in ("single", "complete"):
        raise ValueError("linkage must be 'single' or 'complete'")
    nthreads = _nthreads(nthreads)
    ms = _motif_list(motifs)
    cwm = bool(ms) and all(m.type == "CWM" for m in ms)
    if method is not None and str(method).upper() not in _CWM_METHODS:
        raise ValueError("method must be 'COSINE' or 'CJACCARD', for CWMs")
    metric = _comparison_method(ms, method) if cwm or method is not None else "COSINE"
    _check_sign(sign, cwm)
    if min_score is not None and not cwm:
        raise ValueError("min_score applies to CWMs, which every motif must be")
    clusters, merged, notes = _core._merge_similar(
        ms, qvalue, null, min_overlap, rc, None if cwm else _bkg4(bkg), weighted, linkage, nthreads,
        metric, sign == "ignore", min_score,
    )
    for note in notes:
        warnings.warn(note, stacklevel=2)
    if return_clusters:
        if not ms:
            return pl.DataFrame(schema={"name": pl.String, "motif_i": pl.UInt32, "cluster": pl.Int64})
        return summarize_motifs(ms).with_columns(
            pl.Series("motif_i", list(range(len(ms))), dtype=pl.UInt32),
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
            Draw each motif's logo at its tip, aligned and drawn as by
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
    nthreads: int | None = None,
    metric: str | None = None,
    sign: str = "match",
) -> MotifTree:
    """Cluster motifs into a tree.

    Motifs are compared with [`compare_motifs()`][universalmotif.compare_motifs], mean column
    correlations become distances as ``(1 - score) / 2``, and the distances
    are clustered hierarchically. The agglomeration ``method`` is one of
    ``"complete"``, ``"average"``, ``"single"``, ``"ward.D"``, ``"ward.D2"``,
    ``"mcquitty"``, ``"median"`` or ``"centroid"``; ``"ward.D"`` applies
    Ward's criterion to the distances as they are and ``"ward.D2"`` to the
    squared distances.

    CWMs are compared by cosine similarity when every motif is one, and
    ``metric`` chooses another comparison, ``"PCC"``, ``"COSINE"`` or
    ``"CJACCARD"``, with ``sign`` as
    [`compare_motifs()`][universalmotif.compare_motifs] describes them; the
    similarities become distances in the same way.

    ``motifs`` can also be a square distance matrix, of which only the lower
    triangle is used, such as the scores of ``compare_motifs()`` with
    ``method="EUCL"``. ``labels`` is either a list of labels or a motif
    attribute to label tips with, such as ``"name"``, ``"altname"`` or
    ``"family"``. Duplicate motif names get ``.1``, ``.2`` and so on, while
    the first of each is left as it is.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

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
    nthreads = _nthreads(nthreads)
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
    metric = _comparison_method(ms, metric)
    if metric not in ("PCC", *_CWM_METHODS):
        raise ValueError("metric must be 'PCC', 'COSINE' or 'CJACCARD'")
    _check_sign(sign, metric in _CWM_METHODS)
    if metric in _CWM_METHODS:
        s = np.asarray(_core._cwm_matrix(ms, metric, sign == "ignore", min_overlap, rc, nthreads))
        dist = np.clip((1 - np.maximum(s, s.T)) / 2, 0.0, 1.0)
        return _tree(_core._hclust(dist.tolist(), method, labs), labs, method, ms)
    d = _core._motif_tree(ms, min_overlap, rc, method, labs, nthreads)
    return _tree(d, labs, method, ms)


__all__ = ["MotifTree", "compare_motifs", "merge_motifs", "merge_similar", "motif_tree"]
