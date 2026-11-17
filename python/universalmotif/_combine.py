"""The patterns of many TF-MoDISco runs or contribution_motifs() results,
combined into one set by clustering their CWMs and averaging each cluster."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from . import _core
from ._compare import compare_motifs
from ._core import Motif
from ._threads import _nthreads

_SIGNS = {"pos": "+", "neg": "-"}
# The cutoffs of the flags. A pattern is truncated when a column at its edge
# holds this share of its largest column.
_EDGE_SHARE = 0.5
# Mixed sign: the contribution of the other sign is above this share of the
# pattern's own.
_MIXED_SHARE = 0.5
# Low complexity: the letters of the core repeat with a period of 1 to 3 at
# this share of positions, over a core of at least this many columns, where
# core columns hold this share of the largest.
_REPEAT_SHARE = 0.8
_REPEAT_MIN = 5
_CORE_SHARE = 0.3
# Diffuse: the best window of this many columns holds under this share of
# the total contribution.
_DIFFUSE_WINDOW = 10
_DIFFUSE_SHARE = 0.5


@dataclass
class CombinedPatterns:
    """The result of [`combine_patterns()`][universalmotif.combine_patterns].

    - ``patterns`` has one row per pattern given, with its source and the
      cluster it joined.
    - ``clusters`` has one row per cluster, with its averaged CWM and PPM,
      the sources of its patterns and its flags.
    - ``cwms`` and ``ppms`` give the clusters' motifs as lists.

    [`plot_modisco()`][universalmotif.plot_modisco] draws the clusters as a
    report, and its HTML page lists the patterns of each.
    """

    patterns: pl.DataFrame
    clusters: pl.DataFrame

    @property
    def cwms(self) -> list[Motif]:
        """The clusters' averaged CWMs, in the order of ``clusters``."""
        return self.clusters["cwm"].to_list()

    @property
    def ppms(self) -> list[Motif | None]:
        """The clusters' averaged PPMs, or ``None`` for clusters whose
        patterns came without them."""
        return self.clusters["ppm"].to_list()

    def report(self, path: str | os.PathLike[str] | None = None, database: Sequence[Motif] | None = None,
               n_matches: int = 3, qvalue: float = 0.05, title: str = "Combined patterns",
               nthreads: int | None = None) -> str:
        """The report of [`plot_modisco()`][universalmotif.plot_modisco] as
        one self-contained HTML page, without drawing the figure.

        Each cluster's row shows its seqlets, the number of sources its
        patterns come from, its averaged CWM and PPM, and with a
        ``database`` its closest database motifs. Its CWMs column opens to
        show each of its patterns with their sources. The page is
        returned, and written to ``path`` when one is given.
        """
        from ._plot_modisco import _report

        return _report(self, path, database, n_matches, qvalue, title, nthreads)

    def __len__(self) -> int:
        return self.clusters.height

    def __repr__(self) -> str:
        n_src = self.patterns["source"].n_unique()
        return (f"CombinedPatterns({self.clusters.height} clusters of {self.patterns.height} patterns "
                f"from {n_src} source{'s' if n_src != 1 else ''})")


def _edge_heavy(matrix: np.ndarray) -> bool:
    """Whether a column at either edge holds at least _EDGE_SHARE of the
    largest column, as when a pattern runs off its window."""
    total = np.abs(matrix).sum(axis=0)
    return bool(total.size and total.max() > 0 and max(total[0], total[-1]) >= _EDGE_SHARE * total.max())


def _source_patterns(source: str, given: Any, trim_threshold: float | None) -> list[dict[str, Any]]:
    """The patterns of one source, each with its label, sign, seqlets, CWM,
    PPM and whether it runs off its window."""
    from ._contribution_motifs import ContributionMotifs
    from ._deep import read_modisco
    from ._plot_modisco import _default_label, _group, _pattern_motifs

    out = []
    if isinstance(given, ContributionMotifs):
        # Motifs found by search have no window to run off.
        for r in given.patterns.iter_rows(named=True):
            out.append({"label": r["pattern"], "sign": r["sign"], "seqlets": int(r["seqlets"]),
                        "cwm": r["cwm"], "ppm": r["motif"], "truncated": False})
        return out
    is_file = isinstance(given, (str, os.PathLike))
    motifs = _pattern_motifs(given, ["CWM", "PPM"] if is_file else [], trim_threshold)
    if "CWM" not in motifs:
        raise ValueError(f"source {source!r} has no CWMs; give its patterns as CWMs, or as a dict with a 'CWM' list")
    # Only patterns read from a file have their whole window to judge.
    whole = read_modisco(given, type="CWM", trim_threshold=None) if is_file else [None] * len(motifs["CWM"])
    ppms = motifs.get("PPM", [None] * len(motifs["CWM"]))
    for cwm, ppm, full in zip(motifs["CWM"], ppms, whole):
        group = _group(cwm.name)
        nsites = cwm.nsites if cwm.nsites is not None else (ppm.nsites if ppm is not None else None)
        out.append({"label": _default_label(cwm.name), "sign": _SIGNS.get(group, "+"),
                    "seqlets": int(nsites) if nsites is not None else 0, "cwm": cwm, "ppm": ppm,
                    "truncated": full is not None and _edge_heavy(np.asarray(full.matrix, dtype=float))})
    return out


def _average(cwms: list[Motif], ppms: list[Motif | None], weights: list[int], trim_threshold: float | None,
             nthreads: int) -> tuple[np.ndarray, np.ndarray | None, slice]:
    """The weighted averages of a cluster's CWMs and of its PPMs, in one
    frame, and the columns trimming keeps. Each pattern is aligned by its
    CWM to the first, on either strand, so the two averages line up."""
    from ._api import motif_rc

    w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, 1.0)
    where = {}
    if len(cwms) > 1:
        cmp = compare_motifs(cwms, compare_to=0, qvalue=1, min_overlap=1, nthreads=nthreads)
        where = {r["target_i"]: (r["offset"], r["strand"]) for r in cmp.iter_rows(named=True)}
    width0 = cwms[0].width
    placed = []
    for i, (c, q) in enumerate(zip(cwms, ppms)):
        offset, strand = (0, "+") if i == 0 else where[i]
        if strand == "-":
            # Offsets of reverse-complement matches count along the reversed first pattern.
            c, q = motif_rc(c), (None if q is None else motif_rc(q))
            offset = width0 - offset - c.width
        placed.append((offset, np.asarray(c.matrix, dtype=float),
                       None if q is None else np.asarray(q.matrix, dtype=float)))
    lo = min(o for o, *_ in placed)
    hi = max(o + m.shape[1] for o, m, _ in placed)
    cwm = np.zeros((4, hi - lo))
    ppm = np.zeros((4, hi - lo))
    cover = np.zeros(hi - lo)
    for (o, c, q), wi in zip(placed, w):
        cwm[:, o - lo:o - lo + c.shape[1]] += wi * c
        if q is not None:
            ppm[:, o - lo:o - lo + q.shape[1]] += wi * q
            cover[o - lo:o - lo + q.shape[1]] += wi
    cwm /= w.sum()
    keep = slice(None)
    if trim_threshold is not None:
        total = np.abs(cwm).sum(axis=0)
        idx = np.flatnonzero(total >= trim_threshold * total.max())
        if idx.size:
            keep = slice(idx[0], idx[-1] + 1)
    if any(q is None for q in ppms):
        return cwm, None, keep
    # Columns no pattern covers are uniform.
    return cwm, np.where(cover > 0, ppm / np.maximum(cover, 1e-300), 0.25), keep


def _flags(cwm: np.ndarray, keep: slice, sign: str) -> dict[str, bool]:
    """Whether a cluster's average has mixed signs, a repetitive core or no
    core, from its whole frame and the columns trimming keeps."""
    own = cwm if sign == "+" else -cwm
    pos, neg = own[own > 0].sum(), -own[own < 0].sum()
    total = np.abs(cwm).sum(axis=0)
    # The core: columns from the first to the last holding _CORE_SHARE of the largest.
    trimmed = own[:, keep]
    t_total = np.abs(trimmed).sum(axis=0)
    idx = np.flatnonzero(t_total >= _CORE_SHARE * t_total.max()) if t_total.size and t_total.max() > 0 else []
    repeat = False
    if len(idx) and idx[-1] - idx[0] + 1 >= _REPEAT_MIN:
        letters = trimmed[:, idx[0]:idx[-1] + 1].argmax(axis=0)
        for period in (1, 2, 3):
            same = letters[period:] == letters[:-period]
            repeat = repeat or bool(same.mean() >= _REPEAT_SHARE)
    window = 0.0
    if total.sum() > 0:
        n = min(_DIFFUSE_WINDOW, total.size)
        sums = np.convolve(total, np.ones(n), mode="valid")
        window = float(sums.max() / total.sum())
    return {"mixed_sign": bool(neg > _MIXED_SHARE * pos), "low_complexity": repeat,
            "diffuse": bool(total.size > _DIFFUSE_WINDOW and window < _DIFFUSE_SHARE)}


def _motif(matrix: np.ndarray, kind: str, name: str, nsites: int) -> Motif:
    from ._api import create_motif

    return create_motif(matrix, alphabet="DNA", type=kind, name=name, nsites=max(nsites, 1))


def _cluster(cwms: list[Motif], similarity: float, linkage: str, nthreads: int) -> list[int]:
    """The cluster of each CWM, numbered from 0 in order of first member."""
    if len(cwms) == 1:
        return [0]
    clusters, _, _ = _core._merge_similar(cwms, 0.01, "empirical", 5, True, None, True, linkage, nthreads,
                                          "COSINE", False, similarity, True)
    return list(clusters)


def combine_patterns(
    sources: Mapping[str, Any],
    similarity: float = 0.9,
    linkage: str = "complete",
    trim_threshold: float | None = 0.2,
    database: Sequence[Motif] | None = None,
    qvalue: float = 0.05,
    nthreads: int | None = None,
) -> CombinedPatterns:
    """Combine the patterns of many TF-MoDISco runs or
    [`contribution_motifs()`][universalmotif.contribution_motifs] results
    into one set, by clustering their CWMs and averaging each cluster.

    ``sources`` maps a name for each source, such as its model, to its
    patterns. These can be a TF-MoDISco results file, read with
    [`read_modisco()`][universalmotif.read_modisco] using
    ``trim_threshold``, or the result of ``contribution_motifs()``, whose
    patterns are taken by their representatives. They can also be a list of
    CWMs, or a dict of lists keyed by ``"CWM"`` and ``"PPM"`` with one motif
    per pattern in the same order. A pattern's group, and so its sign,
    comes from the start of its name, ``pos`` or ``neg``, and its seqlets
    from its ``nsites``.

    Patterns are compared by the cosine of their CWMs, as
    [`compare_motifs()`][universalmotif.compare_motifs] compares CWMs, over
    both strands and every offset. Positive and negative patterns are
    clustered apart. With complete linkage, every pair of patterns in a
    cluster has a similarity of at least ``similarity``. Two clusters can
    still have averages more alike than that, when some pair across them is
    less alike, and ``nearest_similarity`` shows these. With single
    linkage, a pattern joins a cluster through any one of its members, so
    clusters can chain together patterns that are alike only in part.

    Each cluster's CWM is the average of its patterns' CWMs, weighted by
    their seqlets, and so is its PPM when every pattern has one. Every
    pattern is aligned by its CWM to the one with the most seqlets, so the
    two averages line up. They are trimmed to the columns from the first to
    the last whose total absolute contribution reaches ``trim_threshold``
    of the largest. Clusters are named by sign and numbered by their
    seqlets, so ``pos_c1`` holds the most seqlets of the positive clusters.
    With a ``database``, each cluster is also named by its closest motif
    there with [`name_motifs()`][universalmotif.name_motifs].

    Four flags mark clusters that may be noise. They only mark clusters, so
    that they can be checked before any are dropped. Real motifs can raise
    them too: GC-rich motifs can look repetitive, and a database match is a
    good reason to keep a flagged cluster.

    - ``truncated``: most of its seqlets come from patterns that run off
      their window, with a column at an edge of the untrimmed pattern
      holding at least half the contribution of its largest column. Only
      patterns read from TF-MoDISco files are judged, as only they come
      with their whole window.
    - ``mixed_sign``: its average carries contribution of the other sign
      above half of that of its own.
    - ``low_complexity``: its core, the columns from the first to the last
      holding 30% of the largest, is at least 5 columns long and its
      letters repeat with a period of 1 to 3, as in ``AAAAAA``, ``CACACA``
      or ``CAGCAGCAG``, at 80% of positions.
    - ``diffuse``: no 10 columns of its average hold half of its total
      contribution, so it has no clear core.

    Parameters
    ----------
    sources
        A dict from source names to their patterns.
    similarity
        The least cosine between two CWMs for them to be clustered
        together.
    linkage
        ``"complete"`` or ``"single"``.
    trim_threshold
        Passed to ``read_modisco()`` when reading a file, and used to trim
        each cluster's averages, or ``None`` to keep every column.
    database
        Motifs to name the clusters by, such as those of JASPAR.
    qvalue
        The largest q-value of a database match.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].

    Returns
    -------
    CombinedPatterns
        The patterns and their clusters.

        ``patterns`` has one row per pattern, in order of source and
        pattern, with these columns.

        - ``source`` and ``pattern``, its label within its source.
        - ``sign``, ``+`` or ``-``, and ``seqlets``.
        - ``cluster``, the cluster it joined, and ``similarity``, the
          cosine of its CWM to the cluster's.
        - ``truncated``, whether it runs off its window.
        - ``cwm`` and ``ppm``, its motifs.

        ``clusters`` has one row per cluster, in order of sign and
        seqlets, with these columns.

        - ``cluster``, its name, and ``sign``.
        - ``patterns``, the number of its patterns, and ``seqlets``, the
          total of theirs.
        - ``sources``, the sources of its patterns, most seqlets first,
          and ``n_sources``.
        - ``min_similarity``, the least cosine between two of its patterns.
        - ``nearest`` and ``nearest_similarity``, the most similar other
          cluster of its sign and their cosine.
        - ``consensus``.
        - With a ``database``, ``match``, ``match_name`` and
          ``match_qvalue``, its closest database motif, which are empty
          when none reaches ``qvalue``.
        - ``truncated``, ``mixed_sign``, ``low_complexity`` and
          ``diffuse``, its flags.
        - ``cwm`` and ``ppm``, its averaged motifs.

    Examples
    --------
    Two runs that found the same AP-1 site, on opposite strands, and one
    site of their own each:

    >>> import numpy as np
    >>> import universalmotif as um
    >>> def cwm(core, name, n):
    ...     m = np.full((4, len(core)), 0.01)
    ...     m[["ACGT".index(c) for c in core], range(len(core))] = 1.0
    ...     return um.create_motif(m, alphabet="DNA", type="CWM", name=name, nsites=n)
    >>> runs = {
    ...     "run1": [cwm("TGACTCAT", "pos_patterns.pattern_0", 300),
    ...              cwm("CCAATCAG", "pos_patterns.pattern_1", 100)],
    ...     "run2": [cwm("ATGAGTCA", "pos_patterns.pattern_0", 200),
    ...              cwm("AAAAAAAA", "pos_patterns.pattern_1", 50)],
    ... }
    >>> combined = um.combine_patterns(runs)
    >>> combined
    CombinedPatterns(3 clusters of 4 patterns from 2 sources)
    >>> combined.clusters.select("cluster", "seqlets", "sources", "low_complexity").rows()
    [('pos_c1', 500, ['run1', 'run2'], False), ('pos_c2', 100, ['run1'], False), ('pos_c3', 50, ['run2'], True)]
    """
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a dict from source names to their patterns")
    if linkage not in ("single", "complete"):
        raise ValueError("linkage must be 'single' or 'complete'")
    if not -1 <= similarity <= 1:
        raise ValueError("similarity must be a cosine, from -1 to 1")
    nthreads = _nthreads(nthreads)

    pats = []
    for source, given in sources.items():
        for p in _source_patterns(str(source), given, trim_threshold):
            pats.append({"source": str(source), **p})
    if not pats:
        raise ValueError("no patterns given")
    for i, p in enumerate(pats):
        p["cwm"] = p["cwm"].replace(name=f"p{i}", nsites=max(p["seqlets"], 1))

    clusters_out: list[dict[str, Any]] = []
    for sign in ("+", "-"):
        idx = [i for i, p in enumerate(pats) if p["sign"] == sign]
        if not idx:
            continue
        labels = _cluster([pats[i]["cwm"] for i in idx], similarity, linkage, nthreads)
        for k in range(max(labels) + 1):
            # The pattern with the most seqlets leads, and the others are aligned to it.
            members = sorted((i for i, lab in zip(idx, labels) if lab == k), key=lambda i: (-pats[i]["seqlets"], i))
            clusters_out.append({"sign": sign, "members": members})

    for c in clusters_out:
        ms = c["members"]
        c["seqlets"] = sum(pats[i]["seqlets"] for i in ms)
        by_source: dict[str, int] = {}
        for i in ms:
            by_source[pats[i]["source"]] = by_source.get(pats[i]["source"], 0) + pats[i]["seqlets"]
        c["sources"] = sorted(by_source, key=lambda s: -by_source[s])
        cwm, ppm, keep = _average([pats[i]["cwm"] for i in ms], [pats[i]["ppm"] for i in ms],
                                  [pats[i]["seqlets"] for i in ms], trim_threshold, nthreads)
        c.update(_flags(cwm, keep, c["sign"]))
        cut = sum(pats[i]["seqlets"] for i in ms if pats[i]["truncated"])
        c["truncated"] = cut > c["seqlets"] / 2
        c["cwm"] = _motif(cwm[:, keep], "CWM", "", c["seqlets"])
        c["ppm"] = None if ppm is None else _motif(ppm[:, keep], "PPM", "", c["seqlets"])
        # Each pattern's similarity to the average, and the least between two patterns.
        sim = np.asarray(compare_motifs([c["cwm"]] + [pats[i]["cwm"] for i in ms], nthreads=nthreads),
                         dtype=float) if len(ms) > 1 else np.ones((2, 2))
        for j, i in enumerate(ms):
            pats[i]["similarity"] = float(sim[0, j + 1])
        inner = sim[1:, 1:]
        c["min_similarity"] = float(inner[~np.eye(len(ms), dtype=bool)].min()) if len(ms) > 1 else 1.0

    # Clusters are numbered within each sign by their seqlets.
    clusters_out.sort(key=lambda c: (c["sign"] != "+", -c["seqlets"]))
    counts = {"+": 0, "-": 0}
    for c in clusters_out:
        counts[c["sign"]] += 1
        c["name"] = f"{'pos' if c['sign'] == '+' else 'neg'}_c{counts[c['sign']]}"
        c["cwm"] = c["cwm"].replace(name=c["name"])
        if c["ppm"] is not None:
            c["ppm"] = c["ppm"].replace(name=c["name"])
        for i in c["members"]:
            pats[i]["cluster"] = c["name"]
        c["nearest"], c["nearest_similarity"] = None, None
    for sign in ("+", "-"):
        own = [c for c in clusters_out if c["sign"] == sign]
        if len(own) > 1:
            sim = np.asarray(compare_motifs([c["cwm"] for c in own], nthreads=nthreads), dtype=float)
            np.fill_diagonal(sim, -np.inf)
            for k, c in enumerate(own):
                j = int(sim[k].argmax())
                c["nearest"], c["nearest_similarity"] = own[j]["name"], float(sim[k, j])

    patterns = pl.DataFrame({
        "source": [p["source"] for p in pats],
        "pattern": [p["label"] for p in pats],
        "sign": [p["sign"] for p in pats],
        "seqlets": pl.Series([p["seqlets"] for p in pats], dtype=pl.Int64),
        "cluster": [p["cluster"] for p in pats],
        "similarity": pl.Series([p["similarity"] for p in pats], dtype=pl.Float64),
        "truncated": pl.Series([p["truncated"] for p in pats], dtype=pl.Boolean),
    }).with_columns(
        pl.Series("cwm", [p["cwm"].replace(name=f"{p['source']}.{p['label']}") for p in pats], dtype=pl.Object),
        pl.Series("ppm", [p["ppm"] for p in pats], dtype=pl.Object),
    )
    clusters = pl.DataFrame({
        "cluster": [c["name"] for c in clusters_out],
        "sign": [c["sign"] for c in clusters_out],
        "patterns": pl.Series([len(c["members"]) for c in clusters_out], dtype=pl.Int64),
        "seqlets": pl.Series([c["seqlets"] for c in clusters_out], dtype=pl.Int64),
        "sources": pl.Series([c["sources"] for c in clusters_out], dtype=pl.List(pl.String)),
        "n_sources": pl.Series([len(c["sources"]) for c in clusters_out], dtype=pl.Int64),
        "min_similarity": pl.Series([c["min_similarity"] for c in clusters_out], dtype=pl.Float64),
        "nearest": pl.Series([c["nearest"] for c in clusters_out], dtype=pl.String),
        "nearest_similarity": pl.Series([c["nearest_similarity"] for c in clusters_out], dtype=pl.Float64),
        "consensus": [(c["ppm"] or c["cwm"]).consensus or "" for c in clusters_out],
    })
    if database is not None:
        from ._compare import name_motifs

        match = [None] * len(clusters_out)
        named = [k for k, c in enumerate(clusters_out) if c["ppm"] is not None]
        if named:
            found = name_motifs([clusters_out[k]["ppm"] for k in named], database, top=1, qvalue=qvalue,
                                nthreads=nthreads)
            for r in found.iter_rows(named=True):
                match[named[r["motif_i"]]] = (r["match"], r["altname"], r["qvalue"])
        clusters = clusters.with_columns(
            pl.Series("match", [m and m[0] for m in match], dtype=pl.String),
            pl.Series("match_name", [m and m[1] for m in match], dtype=pl.String),
            pl.Series("match_qvalue", [m and m[2] for m in match], dtype=pl.Float64),
        )
    clusters = clusters.with_columns(
        *[pl.Series(f, [c[f] for c in clusters_out], dtype=pl.Boolean)
          for f in ("truncated", "mixed_sign", "low_complexity", "diffuse")],
        pl.Series("cwm", [c["cwm"] for c in clusters_out], dtype=pl.Object),
        pl.Series("ppm", [c["ppm"] for c in clusters_out], dtype=pl.Object),
    )
    return CombinedPatterns(patterns, clusters)


__all__ = ["CombinedPatterns", "combine_patterns"]
