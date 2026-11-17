"""Finding motifs in a sequence model's contribution scores: seqlets, motif
discovery in them, refinement and selection, in one call."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from . import _progress
from ._core import Motif
from ._deep import idealize_sequences
from ._gapped_finder import gapped_motif_finder
from ._hit_contrib import (
    _LETTERS,
    _gather_scores,
    hit_contributions,
    scan_contributions,
)
from ._select import refine_motifs, select_motifs
from ._seqlets import call_seqlets
from ._sequences import _seed, motif_finder
from ._threads import _nthreads
from ._utils import add_gap

# Positions kept around each seqlet when refining and selecting motifs, and
# how far refine_motifs() can widen a motif on each side over its iterations.
_WINDOW_PAD = 64
_REFINE_GROWTH = 15
_NOT_ACGT = re.compile("[^ACGTacgt]+")
# Seqlets shorter than this are widened around their middles for discovery,
# and for the gapped search, to fit two parts and their spacer.
_MIN_TARGET = 20
_MIN_GAPPED_TARGET = 30
# Hit thresholds for motifs with no hit within a seqlet: scan_contributions()'s
# defaults.
_DEFAULT_THRESHOLDS = (0.7, 0.5)
# The least similarity to the CWM, as a cosine, for a hit. A motif's median
# site in the seqlets must also reach it for the motif to be kept.
_MIN_SIMILARITY = 0.5
# The fewest letters between the parts of a composite motif.
_MIN_SPACER = 2
# The positions a hit must share with a seqlet to count as reaching it.
_MIN_OVERLAP = 5


@dataclass
class ContributionMotifs:
    """The result of
    [`contribution_motifs()`][universalmotif.contribution_motifs].

    - ``motifs`` has one row per motif kept, in the order they were chosen.
    - ``seqlets`` holds the seqlets they were found in and chosen to reach.
    - ``ppms`` and ``cwms`` give the motifs as lists.
    - ``thresholds`` gives their hit thresholds.
    - ``hits`` holds the hits of the CWMs in the scores.
    - ``patterns`` gathers motifs with similar CWMs, such as spellings of
      one factor's site, under the one given the most seqlets.
    """

    motifs: pl.DataFrame
    seqlets: pl.DataFrame
    hits: pl.DataFrame | None = None
    #: For each motif, the rows of ``seqlets`` given to it.
    reach: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def ppms(self) -> list[Motif]:
        """The motifs, as PPMs."""
        return self.motifs["motif"].to_list()

    @property
    def thresholds(self) -> pl.DataFrame:
        """The motifs' hit thresholds, as
        [`scan_contributions()`][universalmotif.scan_contributions] takes
        them with their ``cwms``."""
        return self.motifs.select(motif="name", min_similarity="min_similarity", min_importance="min_importance")

    @property
    def cwms(self) -> list[Motif]:
        """The CWMs of the motifs' hits in the seqlets."""
        return self.motifs["cwm"].to_list()

    @property
    def patterns(self) -> pl.DataFrame:
        """One row per pattern, in order of the seqlets it reaches.

        The columns are these.

        - ``pattern``, its name, and ``sign``.
        - ``representative``, the member given the most seqlets, and that
          motif's ``consensus``.
        - ``seqlets``, the seqlets given to its members.
        - ``share``, those seqlets as a share of the seqlets of its sign.
        - ``members``, the names of its members, the representative first.
        - ``motif`` and ``cwm``, the representative's PPM and CWM.
        """
        t = self.motifs.with_row_index("_row").sort("seqlets", "_row", descending=[True, False])
        rep = t.group_by("pattern", maintain_order=True).first()
        members = t.group_by("pattern", maintain_order=True).agg(members=pl.col("name"))
        pats = rep.select("pattern", "sign", representative="name", consensus="consensus", motif="motif",
                          cwm="cwm").join(members, on="pattern", maintain_order="left")
        reach = self.reach or {}
        union = {p: np.unique(np.concatenate([reach.get(m, np.empty(0, np.int64)) for m in ms]))
                 for p, ms in pats.select("pattern", "members").iter_rows()}
        n_sign = {s: int((self.seqlets["sign"] == s).sum()) if "sign" in self.seqlets.columns else self.seqlets.height
                  for s in pats["sign"].unique()}
        rows = []
        for p, sg in pats.select("pattern", "sign").iter_rows():
            u = union[p]
            rows.append((len(u), len(u) / max(n_sign[sg], 1)))
        stats = pl.DataFrame(rows, schema={"seqlets": pl.Int64, "share": pl.Float64}, orient="row")
        return (pats.with_columns(stats.get_columns())
                .sort("seqlets", descending=True, maintain_order=True)
                .select("pattern", "sign", "representative", "consensus", "seqlets", "share", "members", "motif",
                        "cwm"))

    def report(self, path: str | os.PathLike[str] | None = None, database: Iterable[Motif] | None = None,
               n_matches: int = 3, qvalue: float = 0.05, title: str = "Motifs from contribution scores",
               nthreads: int | None = None) -> str:
        """The report of [`plot_modisco()`][universalmotif.plot_modisco] as
        one self-contained HTML page, without drawing the figure.

        Each pattern's row shows the following.

        - Its representative's consensus, and the seqlets given to its
          members with their share of all seqlets.
        - The logos of its representative's CWM, and of its PPM as
          information content.
        - With a ``database`` such as JASPAR, the ``n_matches`` database
          motifs it resembles most by
          [`name_motifs()`][universalmotif.name_motifs], with a q-value of
          at most ``qvalue``.
        - Each of its members.

        ``plot_modisco(found, database=..., html=path)`` draws the same
        report as a figure and writes this page too. The page is returned,
        and written to ``path`` when one is given.
        """
        from ._plot_modisco import _report

        return _report(self, path, database, n_matches, qvalue, title, nthreads)

    def __len__(self) -> int:
        return self.motifs.height

    def __repr__(self) -> str:
        n = self.motifs["pattern"].n_unique() if "pattern" in self.motifs.columns else self.motifs.height
        return (f"ContributionMotifs({self.motifs.height} motifs in {n} pattern{'s' if n != 1 else ''} "
                f"from {self.seqlets.height:,} seqlets)")


def _strings(codes: np.ndarray, lengths: np.ndarray) -> list[str]:
    letters = np.frombuffer((_LETTERS + "N").encode(), dtype=np.uint8)
    return [letters[np.minimum(codes[i, :n], 4)].tobytes().decode() for i, n in enumerate(lengths)]


def _letters_of(scores, sequences) -> list[str]:
    """The sequences as text: as given, from the scores' own letters, or from
    which letter carries the score at each position of projected scores."""
    if scores.codes is not None:
        return _strings(scores.codes, scores.lengths)
    if scores.values.ndim == 3:
        v = np.abs(scores.values)
        codes = np.where(v.max(axis=1) > 0, v.argmax(axis=1), 4)
        return _strings(codes, scores.lengths)
    raise ValueError("give the sequences, which observed scores alone do not hold")


def _widen(start: np.ndarray, end: np.ndarray, lengths: np.ndarray,
           width: int = _MIN_TARGET) -> tuple[np.ndarray, np.ndarray]:
    """Windows of at least ``width`` letters around each seqlet, kept within
    their sequences."""
    grow = np.maximum(width - (end - start), 0)
    s, e = start - grow // 2, end + (grow - grow // 2)
    shift = np.where(s < 0, -s, 0) - np.where(e > lengths, e - lengths, 0)
    return np.maximum(s + shift, 0), np.minimum(e + shift, lengths)


def _background(seqs: list[str], seqlets: pl.DataFrame, widths: np.ndarray, rng, ratio: int = 2) -> list[str]:
    """Windows of the seqlets' widths drawn from the sequences where there is
    no seqlet and no letter outside ACGT, ``ratio`` per seqlet."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(lengths)])
    # The stretches each window must avoid, the seqlets and the runs of other
    # letters, in one coordinate along all the sequences, sorted, with the
    # running furthest end.
    si = seqlets["sequence_i"].to_numpy().astype(np.int64)
    starts = [offset[si] + np.maximum(seqlets["start"].to_numpy(), 0)]
    ends = [offset[si] + seqlets["end"].to_numpy()]
    # A lookup table and the positions of other letters alone keep this to a
    # byte per letter of the sequences.
    bad = np.ones(256, dtype=bool)
    bad[np.frombuffer(b"ACGTacgt", dtype=np.uint8)] = False
    other = np.flatnonzero(bad[np.frombuffer("".join(seqs).encode("latin-1", "replace"), dtype=np.uint8)])
    if len(other):
        # Runs crossing from one sequence into the next are split at the boundary,
        # which only adds stretches.
        breaks = np.flatnonzero(np.diff(other) != 1)
        starts.append(other[np.concatenate([[0], breaks + 1])])
        ends.append(other[np.concatenate([breaks, [len(other) - 1]])] + 1)
    st, en = np.concatenate(starts), np.concatenate(ends)
    order = np.argsort(st, kind="stable")
    st, en = st[order], np.maximum.accumulate(en[order]) if len(order) else en
    want = np.repeat(widths, ratio)
    out, todo = [None] * len(want), np.arange(len(want))
    for _ in range(200):
        if not len(todo):
            break
        seq = rng.integers(0, len(seqs), len(todo))
        room = lengths[seq] - want[todo]
        ok = room >= 0
        start = (rng.random(len(todo)) * np.maximum(room + 1, 1)).astype(np.int64)
        gs = offset[seq] + start
        # The first stretch ending after the window starts must begin at or after its end.
        j = np.searchsorted(en, gs, side="right")
        free = ok & ((j == len(en)) | (st[np.minimum(j, max(len(st) - 1, 0))] >= gs + want[todo]) if len(st)
                     else ok)
        for k in np.flatnonzero(free):
            out[todo[k]] = seqs[seq[k]][start[k]:start[k] + want[todo[k]]]
        todo = todo[~free]
    return [w for w in out if w is not None]


def _site_hits(motifs: list[Motif], seqs: list[str], sites: pl.DataFrame, nthreads: int,
               within: bool = False) -> pl.DataFrame:
    """The motifs' hits that reach the sites, or with ``within`` that lie
    within them, with the index of the site in ``site``. Only windows around
    the sites wide enough to hold such hits are scanned."""
    from ._analysis import _scan_strings

    pad = max(m.width + (sum(m.maxgap) if m.isgapped else 0) for m in motifs)
    si = sites["sequence_i"].to_numpy()
    lengths = np.array([len(seqs[i]) for i in si], dtype=np.int64)
    lo = np.maximum(sites["start"].to_numpy() - pad, 0)
    hi = np.minimum(sites["end"].to_numpy() + pad, lengths)
    windows = [seqs[i][a:b] for i, a, b in zip(si, lo, hi)]
    # Motifs too short to reach the cutoff, such as one of six letters, hit at their best score.
    hits = _scan_strings(motifs, windows, 1e-4, nthreads, reachable=True)
    w = hits["sequence_i"].to_numpy()
    hits = hits.drop("sequence", "sequence_i", strict=False).with_columns(
        pl.Series("site", w, dtype=pl.UInt32), pl.Series("sequence_i", si[w], dtype=pl.Int64),
        pl.col("start") + pl.Series(lo[w]), pl.col("end") + pl.Series(lo[w]),
        s=pl.Series(sites["start"].to_numpy()[w]), e=pl.Series(sites["end"].to_numpy()[w]))
    return (hits.filter((pl.col("start") >= pl.col("s")) & (pl.col("end") <= pl.col("e")) if within else
                        pl.min_horizontal("end", "e") - pl.max_horizontal("start", "s") >= _MIN_OVERLAP)
            .drop("s", "e"))


def _composite_parts(rows) -> dict[str, tuple[int, int, float, float, float]]:
    """For each composite of two parts, as (name, PPM, CWM, min_importance)
    rows: the widths of its parts, their weights in its CWM, and the share of
    them a hit must carry."""
    out = {}
    for name, m, cwm, imp in rows:
        if m is None or cwm is None or not m.isgapped or len(m.gaploc) != 1:
            continue
        x = np.abs(np.asarray(cwm.matrix, dtype=float))
        w1 = int(m.gaploc[0])
        w2 = m.width - w1
        if w1 <= 0 or w2 <= 0 or w1 + w2 > x.shape[1]:
            continue
        out[name] = (w1, w2, float(x[:, :w1].sum()), float(x[:, x.shape[1] - w2:].sum()), float(imp))
    return out


def _parts_kept(names: np.ndarray, seq_i: np.ndarray, start: np.ndarray, end: np.ndarray, minus: np.ndarray,
                observed: np.ndarray, parts: dict) -> np.ndarray:
    """Which hits of composites carry at least their share of the weight of
    each part, as a hit carries it of the whole CWM. A strong first part
    alone can otherwise give a composite's CWM a close cosine, so that sites
    of that part alone pass as hits. Hits of other motifs are all kept."""
    keep = np.ones(len(names), dtype=bool)
    for name, (w1, w2, c1, c2, imp) in parts.items():
        k = np.flatnonzero(names == name)
        if not len(k):
            continue
        first = np.where(minus[k], end[k] - w1, start[k])
        second = np.where(minus[k], start[k], end[k] - w2)

        rows = seq_i[k][:, None]
        weight1 = np.abs(np.nan_to_num(observed[rows, first[:, None] + np.arange(w1)])).sum(axis=1)
        weight2 = np.abs(np.nan_to_num(observed[rows, second[:, None] + np.arange(w2)])).sum(axis=1)
        keep[k] = (weight1 >= imp * c1) & (weight2 >= imp * c2)
    return keep


def _score_reach(kept: list[dict], seqs: list[str], sites: pl.DataFrame, observed: np.ndarray,
                 nthreads: int) -> dict[str, set[int]]:
    """The sites each kept motif's CWM reaches best, matched against the
    scores around them at the motif's thresholds. A hit reaches a site when
    it shares at least _MIN_OVERLAP positions with it, or all of its own
    when narrower. Each site goes to the motif whose hit there is most like
    its CWM, as TF-MoDISco gives each seqlet to one pattern."""
    cwms = [r["cwm"].replace(name=r["name"]) for r in kept]
    thresholds = pl.DataFrame({"motif": [r["name"] for r in kept],
                               "min_similarity": [r["min_similarity"] for r in kept],
                               "min_importance": [r["min_importance"] for r in kept]})
    pad = max(c.width for c in cwms)
    si = sites["sequence_i"].to_numpy()
    lengths = np.array([len(seqs[i]) for i in si], dtype=np.int64)
    lo = np.maximum(sites["start"].to_numpy() - pad, 0)
    hi = np.minimum(sites["end"].to_numpy() + pad, lengths)
    s_start, s_end = sites["start"].to_numpy(), sites["end"].to_numpy()
    parts = _composite_parts([(r["name"], r["motif"], r["cwm"], r["min_importance"]) for r in kept])
    out: dict[str, set[int]] = {}
    # A chunk at a time, as motifs whose thresholds are loose can hit almost everywhere.
    for c0 in range(0, len(si), _SCAN_CHUNK):
        k = np.arange(c0, min(c0 + _SCAN_CHUNK, len(si)))
        hits = scan_contributions(cwms, [observed[si[j], lo[j]:hi[j]] for j in k],
                                  sequences=[seqs[si[j]][lo[j]:hi[j]] for j in k], thresholds=thresholds,
                                  nthreads=nthreads)
        w = k[hits["sequence_i"].to_numpy()]
        start, end = hits["start"].to_numpy() + lo[w], hits["end"].to_numpy() + lo[w]
        overlap = np.minimum(end, s_end[w]) - np.maximum(start, s_start[w])
        ok = overlap >= np.minimum(end - start, _MIN_OVERLAP)
        if parts:
            ok &= _parts_kept(hits["motif"].cast(pl.String).to_numpy(), si[w], start, end,
                              hits["strand"].to_numpy() == "-", observed, parts)
        best = (pl.DataFrame({"motif": hits["motif"].cast(pl.String), "site": w,
                              "similarity": hits["similarity"]}).filter(pl.Series(ok))
                .sort(["site", "similarity"], descending=[False, True]).unique("site", keep="first"))
        for name, j in best.select("motif", "site").iter_rows():
            out.setdefault(name, set()).add(int(j))
    return out


def _cwms(hits: pl.DataFrame, observed: Any) -> dict:
    """The CWM of each motif's hits, from those at its commonest spacing for
    a gapped motif."""
    hits = (hits.unique(["motif", "sequence_i", "start", "strand"], maintain_order=True)
            .with_columns(w=pl.col("end") - pl.col("start"))
            .filter(pl.col("w") == pl.col("w").mode().min().over("motif"))
            .drop("w", "site"))
    if not hits.height:
        return {}
    return {m.name: m for m in hit_contributions(hits, observed).to_cwms()}


def _thresholds(hits: pl.DataFrame, observed: Any, cwms: dict, quantile: float) -> dict:
    """Hit thresholds for scan_contributions() from each motif's hits in the
    seqlets, scored as it scores a window. They are the ``quantile`` of the
    hits' similarity, the cosine of their scores with the CWM, and of their
    importance. Importance is the sum of their absolute scores as a share
    of the CWM's."""
    hits = (hits.unique(["motif", "sequence_i", "start", "strand"], maintain_order=True)
            .with_columns(w=pl.col("end") - pl.col("start"))
            .filter(pl.col("w") == pl.col("w").mode().min().over("motif"))
            .drop("w", "site"))
    if not hits.height:
        return {}
    hc = hit_contributions(hits, observed)
    motif = hc.hits["motif"].cast(pl.String).to_numpy()
    lookup = np.full(256, -1)
    for i, c in enumerate(_LETTERS):
        lookup[ord(c)] = lookup[ord(c.lower())] = i
    out = {}
    for name, cwm in cwms.items():
        rows = np.flatnonzero(motif == name)
        w = cwm.width
        matches = hc.hits["match"][rows].to_list()
        if not len(rows) or any(len(m) != w for m in matches):
            continue
        codes = lookup[np.frombuffer("".join(matches).encode(), dtype=np.uint8)]
        codes = codes.reshape(len(rows), w)
        obs = np.nan_to_num(hc.observed[rows, :w])
        x = np.zeros((len(rows), len(_LETTERS), w))
        r, j = np.nonzero(codes >= 0)
        x[r, codes[r, j], j] = obs[r, j]
        c = np.asarray(cwm.matrix, dtype=float)
        norm = np.sqrt((x ** 2).sum(axis=(1, 2)))
        sim = np.where(norm > 0, np.einsum("nal,al->n", x, c) / np.maximum(norm * np.sqrt((c ** 2).sum()), 1e-300),
                       0.0)
        imp = np.abs(x).sum(axis=(1, 2)) / max(float(np.abs(c).sum()), 1e-300)
        out[name] = (float(np.quantile(sim, quantile)), float(np.quantile(imp, quantile)), float(np.median(sim)))
    return out


def _reach(hits: pl.DataFrame) -> dict[str, set[int]]:
    return {name: set(s) for name, s in hits.group_by("motif").agg(pl.col("site").unique()).iter_rows()}


def contribution_motifs(
    contributions: Any,
    sequences: Any = None,
    hypothetical: Any = None,
    idealized: Any = None,
    seqlets: pl.DataFrame | None = None,
    null_scores: Any = None,
    null_sequences: Any = None,
    sign: str = "positive",
    min_width: int = 6,
    max_width: int = 15,
    nmotifs: int = 60,
    refine: bool = True,
    min_gain: float = 0.0025,
    max_similarity: float | None = 0.9,
    stability: int = 0,
    min_stability: float = 0.75,
    idealize_threshold: float = 0.01,
    rounds: int = 2,
    gapped: bool = True,
    max_gap: int = 12,
    hits: bool = True,
    hit_quantile: float = 0.05,
    pattern_similarity: float | None = 0.8,
    seed: int | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
) -> ContributionMotifs:
    """Find the motifs a sequence model relies on, from its contribution
    scores.

    The workflow is experimental, so its options, its results and the
    motifs it finds may change in later releases.

    Each step uses one of the functions named below, with options suited
    to attributions.

    1. [`call_seqlets()`][universalmotif.call_seqlets] calls seqlets in the
       scores. Seqlets can instead be given as ``seqlets``, such as those
       of [`read_seqlets()`][universalmotif.read_seqlets].
    2. [`motif_finder()`][universalmotif.motif_finder] searches the seqlets'
       letters, widened to at least 20. The background is twice as many
       windows of the same widths, taken from the sequences away from any
       seqlet of the sign searched. With ``hypothetical`` scores, it also searches the seqlets
       rewritten in the letters the model prefers by
       [`idealize_sequences()`][universalmotif.idealize_sequences], or
       rewritten beforehand and given as ``idealized``. This
       finds degenerate sites, such as spaced palindromes, whose letters
       vary too much for a plain search.
    3. With ``stability``, each search is rerun that many times on half of
       the seqlets. Motifs found again in fewer than ``min_stability`` of
       the reruns are left out.
    4. With ``refine``, [`refine_motifs()`][universalmotif.refine_motifs]
       rebuilds every motif from its hits in the seqlets of the real
       sequences.
    5. [`select_motifs()`][universalmotif.select_motifs] keeps each motif
       that reaches at least ``min_gain`` of the seqlets not reached by
       those before it. With ``max_similarity``, it also leaves out motifs
       whose CWMs repeat one already kept. A motif's CWM is the mean
       contribution at its hits in the seqlets.
    6. With ``rounds`` of 2 or more, the searches are run again on the
       seqlets that no kept motif reaches. Everything found so far is then
       refined and selected again. Rare motifs make up a larger share of
       the leftover seqlets than of all of them, so this finds them more
       reliably.
    7. With ``gapped``,
       [`gapped_motif_finder()`][universalmotif.gapped_motif_finder]
       searches the seqlets, widened to 30 letters, for composite motifs
       made of two parts at least 2 letters apart. Each composite's gap
       range is cut to the spacings around its commonest that hold at least
       half as many hits. It is kept when it adds ``min_gain`` of the
       seqlets to those reached. One of its parts is often a motif kept
       already, so its CWM is only compared with those of other composites.
    8. With ``hits``,
       [`scan_contributions()`][universalmotif.scan_contributions] matches
       the motifs' CWMs against all the scores. Each motif's thresholds
       come from its hits within the seqlets, as
       [`modisco_thresholds()`][universalmotif.modisco_thresholds] sets
       them from a pattern's seqlets. They are the ``hit_quantile`` of the
       hits' similarity and of their importance, with a similarity of at
       least 0.5. Motifs with no hit within a seqlet take the defaults of
       ``scan_contributions()`` instead. Composites are typical of these,
       as their parts are often seqlets of their own. A composite's hit
       must also carry at least ``min_importance`` of the weight of each of
       its parts in the CWM, as a site of its stronger part alone could
       otherwise pass. Motifs whose median site in the seqlets has a
       similarity under 0.5 are left out, since their sites share no shape
       of contribution.

    Each seqlet is then given to one motif, as TF-MoDISco gives it to one
    pattern. It goes to the motif whose CWM matches its scores most closely,
    by cosine, among those with a hit there at their thresholds. These
    counts include the sites of short, degenerate motifs such as the
    initiator, whose letters alone rarely reach the P-value cutoff of the
    sequence matches used to choose motifs.

    Finally, motifs are gathered into patterns, as TF-MoDISco gathers
    seqlets. In order of the seqlets their sequence matches reach, each motif
    joins the first pattern whose first motif's CWM is at least
    ``pattern_similarity`` like its own. A factor's site found in several
    spellings thus becomes one pattern, with those spellings as its members.
    Composites start patterns of their own. Each pattern's representative is
    its member given the most seqlets. Every motif is kept and scanned for,
    whatever its pattern, as the patterns only arrange them.

    Each sign of ``sign`` is searched on its own. The defaults aim at
    finding the patterns a model has learned, much as TF-MoDISco does.
    They keep rarer motifs at the cost of reaching fewer seqlets with each.
    For a short list of motifs that reach as many seqlets as possible, use
    ``min_gain=0.01`` and ``max_similarity=None``. Rarer motifs, such as
    those found only in idealized seqlets, are often missing from the
    half-sized reruns. Setting ``stability`` therefore trades them for
    motifs that other samples of regions would give too.

    On a chromBPNet model of chromatin accessibility in K562 cells, over
    21,000 windows of 500 bases, the defaults with hypothetical scores
    recovered 28 of the model's 44 published TF-MoDISco patterns at a CWM
    cosine of 0.8, against 25 for TF-MoDISco on the same windows. The call
    took 10 seconds, against 41 minutes for TF-MoDISco, and the patterns
    found included the NFI palindrome and the TAL1–GATA composite. Over all
    187,000 of the model's windows it took 72 seconds on two threads, with
    memory peaking at 2.6 GB. The
    [motifs from contribution scores](../guides/contribution-motifs.ipynb)
    guide gives the full comparison.

    Scores from a model's profile head are spikier than those of its counts
    head, with a few strong positions and weak ones between. In the default
    seqlets, every stretch of 4 or more positions must stand out, so they
    miss some of these sites. Seqlets called with
    ``call_seqlets(scores, sequences, min_len=2, threshold=0.05)`` and given
    as ``seqlets`` often suit them better. TF-MoDISco's profile-head
    patterns are wide, with much of their weight in the flanks, so our
    motifs match many of them only in their cores.

    The motifs found in different samples of regions, or with different
    seeds, differ mostly in the rarer ones, while the common patterns are
    found every time. On a BPNet model of GATA1, three seeds all gave the
    GATA motif and the TAL1–GATA composite, but between none and three
    rarer spellings of the GATA site. Give ``seed`` for results that can be
    repeated, and use ``stability``, or
    [`combine_patterns()`][universalmotif.combine_patterns] over runs with
    several seeds, to keep only the motifs found again. A lower
    ``max_similarity`` keeps fewer near-identical motifs, which makes the
    motifs match more often across samples but loses some rarer patterns.

    Parameters
    ----------
    contributions
        The scores, in any form
        [`hit_contributions()`][universalmotif.hit_contributions] takes,
        such as a [`ContributionScores`][universalmotif.ContributionScores]
        or an array of shape (sequences, 4, length) or (sequences,
        length).
    sequences
        The sequences the scores are for, as text or one-hot encoded. They
        are not needed with ContributionScores, or with projected scores,
        which are zero for all but the letter present.
    hypothetical
        Hypothetical scores, giving the score of every letter at every
        position, for the search on idealized seqlets. They can be an array
        of shape (sequences, 4, length) or a ContributionScores of
        hypothetical scores.
    idealized
        The sequences already rewritten by ``idealize_sequences()``, in
        place of ``hypothetical``, in any of the forms ``sequences`` takes.
        Idealizing sequences as a model gives their hypothetical scores
        means those scores need not be kept. With ``sign="both"``, give a
        dict of the ``"positive"`` and ``"negative"`` rewritings.
    seqlets
        Seqlets to use instead of calling them, as a table with ``start``
        and ``end`` columns and a ``sequence_i`` column, or a ``sequence``
        column naming the sequences. To search both signs, add a ``sign``
        column of ``+`` or ``-``.
    null_scores, null_sequences
        Null scores for calling seqlets, such as the model's contributions
        for shuffled sequences, and their sequences when the scores are
        hypothetical; see ``call_seqlets()``.
    sign
        ``"positive"``, ``"negative"`` or ``"both"``.
    min_width, max_width, nmotifs
        As for ``motif_finder()``. Refinement can widen motifs beyond
        ``max_width``, up to 30 letters.
    refine
        Rebuild the motifs from their hits.
    min_gain, max_similarity
        As for ``select_motifs()``.
    stability, min_stability
        Reruns of each search for checking stability, and the share of them
        a motif must be found in to be kept.
    idealize_threshold
        How much more the model must prefer a letter for an idealized
        seqlet to take it.
    rounds
        How many times to search: the first on all the seqlets, and each
        later one on those the motifs kept so far do not reach.
    gapped, max_gap
        Search for composite motifs, and the longest spacer to try.
    hits, hit_quantile
        Match the CWMs against the scores, and the quantile of their hits
        within the seqlets that sets each motif's thresholds.
    pattern_similarity
        The least cosine between two motifs' CWMs for the second to join
        the first's pattern, or None to give every motif a pattern of its
        own.
    seed
        Seed for the background, the reruns and the shuffles of the null.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the call has run for a second;
        by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    ContributionMotifs
        Its ``motifs`` table has one row per motif kept, in the order chosen
        and with composites last. The columns are these.

        - ``name``. Names of motifs from later rounds start with ``r``, and
          those of composites with ``g``.
        - ``pattern``, the pattern it belongs to, which ``patterns`` lists.
        - ``sign``.
        - ``source``, one of ``seqlets``, ``idealized`` or ``gapped``.
        - ``consensus`` and ``width``.
        - ``seqlets``, the seqlets given to it.
        - ``gain`` and ``cumulative``, the shares of the seqlets of its sign
          that its sequence matches added when it was chosen, and that those
          of the motifs up to it reach.
        - ``stability``, with reruns.
        - ``min_similarity`` and ``min_importance``, its hit thresholds.
        - ``motif``, the motif as a PPM.
        - ``cwm``, the motif as a CWM, at its commonest spacing for a
          composite.

        Its ``seqlets`` table holds the seqlets searched, and ``hits`` the
        hits of the CWMs as ``scan_contributions()`` gives them.

    References
    ----------
    - Shrikumar A, Tian K, Avsec Ž, Shcherbina A, Banerjee A, Sharmin M, Nair
      S, Kundaje A (2018). Technical note on transcription factor motif
      discovery from importance scores (TF-MoDISco) version 0.5.6.5.
      arXiv:1811.00416. <https://doi.org/10.48550/arXiv.1811.00416>
    - Avsec Ž, Weilert M, Shrikumar A, Krueger S, Alexandari A, Dalal K, Fropf
      R, McAnany C, et al. (2021). Base-resolution models of
      transcription-factor binding reveal soft motif syntax. *Nature Genetics*
      53(3):354–366. <https://doi.org/10.1038/s41588-021-00782-6>
    - Bailey TL (2021). STREME: accurate and versatile sequence motif
      discovery. *Bioinformatics* 37(18):2834–2840.
      <https://doi.org/10.1093/bioinformatics/btab203>
    - Pampari A, Shcherbina A, Kvon EZ, Kosicki M, Nair S, Kundu S, Kathiria
      AS, Risca VI, et al. (2024). ChromBPNet: bias factorized, base-resolution
      deep learning models of chromatin accessibility reveal cis-regulatory
      sequence syntax, transcription factor footprints and regulatory variants.
      *bioRxiv* 2024.12.25.630221. <https://doi.org/10.1101/2024.12.25.630221>
    - Wadman IA, Osada H, Grütz GG, Agulnick AD, Westphal H, Forster A,
      Rabbitts TH (1997). The LIM-only protein Lmo2 is a bridging molecule
      assembling an erythroid, DNA-binding complex which includes the TAL1,
      E47, GATA-1 and Ldb1/NLI proteins. *The EMBO Journal* 16(11):3145–3157.
      <https://doi.org/10.1093/emboj/16.11.3145>

    Examples
    --------
    A model that relies on AP-1 sites, planted in half of 300 sequences:

    >>> import numpy as np
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(300)]
    >>> seqs = [s[:50] + "TGACTCA" + s[57:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    >>> scores = rng.normal(0, 0.005, (300, 120))
    >>> scores[::2, 50:57] += 0.1
    >>> found = um.contribution_motifs(scores, seqs, seed=1)
    >>> found.motifs.select("name", "consensus", "seqlets").rows()
    [('pos_1', 'TGACTCA', 134)]
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: contribution_motifs(**args))
    if sign not in ("positive", "negative", "both"):
        raise ValueError("sign must be 'positive', 'negative' or 'both'")
    if hypothetical is not None and idealized is not None:
        raise ValueError("give hypothetical scores or idealized sequences, not both")
    nthreads = _nthreads(nthreads)
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    scores = _gather_scores(contributions, sequences, None)
    seqs = _letters_of(scores, sequences)
    observed = scores.values
    if observed.ndim == 3:
        if scores.codes is not None:
            onehot = scores.codes[:, None, :] == np.arange(len(_LETTERS))[None, :, None]
            observed = (observed * onehot).sum(axis=1)
        else:
            observed = observed.sum(axis=1)
    observed = np.asarray(observed, dtype=np.float32)
    lengths = np.array([len(s) for s in seqs])

    if seqlets is None:
        # Scores of different lengths come padded; each is called at its own length.
        given = ([observed[i, :n] for i, n in enumerate(lengths)] if (lengths != observed.shape[1]).any()
                 else observed)
        seqlets = call_seqlets(given, seqs, null_scores=null_scores, null_sequences=null_sequences, sign=sign,
                               seed=seed, nthreads=nthreads)
        if scores.names is not None:
            seqlets = seqlets.with_columns(pl.Series("sequence", np.asarray(scores.names, dtype=object)[
                seqlets["sequence_i"].to_numpy()], dtype=pl.String))
    else:
        seqlets = _given_seqlets(seqlets, scores.names, lengths)
        if "sign" not in seqlets.columns and sign == "both":
            raise ValueError("seqlets without a 'sign' column can be searched for one sign only; give "
                             "sign='positive' or 'negative', or add a sign column of '+' and '-'")
    signs = {"positive": ["+"], "negative": ["-"], "both": ["+", "-"]}[sign]
    ideal = {} if idealized is None else _given_idealized(idealized, seqs, sign)
    rows = []
    reach_by_motif: dict[str, np.ndarray] = {}
    for sg in signs:
        sub = seqlets.filter(pl.col("sign") == sg) if "sign" in seqlets.columns else seqlets
        sub_ids = (seqlets.with_row_index("_id").filter(pl.col("sign") == sg)["_id"].to_numpy()
                   if "sign" in seqlets.columns else np.arange(seqlets.height))
        label = "pos" if sg == "+" else "neg"
        if sub.height < 20:
            # A sign searched alongside the other may well have no seqlets at all.
            if sub.height or sign != "both":
                kind = "positive" if sg == "+" else "negative"
                warnings.warn(f"only {sub.height} {kind} seqlets, too few to search", stacklevel=2)
            continue
        sources = {"seqlets": seqs}
        key = "positive" if sg == "+" else "negative"
        if hypothetical is not None and key not in ideal:
            ideal[key] = idealize_sequences(seqs, hypothetical, threshold=idealize_threshold, sign=key)
        if key in ideal:
            sources["idealized"] = ideal[key]
        sites = sub.select("sequence_i", "start", "end")
        found, kept, reached = [], [], set()
        for round_ in range(rounds):
            part = sub if not round_ else sub.with_row_index("_site").filter(
                ~pl.col("_site").is_in(list(reached))).drop("_site")
            if part.height < 20:
                break
            found += _search(part, sub, sources, lengths, label, "r" * round_, min_width, max_width, nmotifs,
                             stability, min_stability, rng, nthreads)
            kept, cand_hits = _choose(found, seqs, sites, observed, sg, refine, min_gain, max_similarity, nthreads)
            reach = _reach(cand_hits) if cand_hits is not None else {}
            reached = set().union(*(reach[r["name"]] for r in kept)) if kept else set()
        if gapped:
            si = sub["sequence_i"].to_numpy().astype(np.int64)
            kept += _gapped_pass(seqs, sub, sites, si, lengths, observed, reached, label, sg, max_gap, min_gain,
                                 max_similarity, refine, rng, nthreads)
        if kept:
            # Thresholds come from the hits within the seqlets. Motifs wider than their seqlets take the
            # defaults instead, since hits only overlapping a seqlet are mostly positions the model gave
            # little weight. Composites are typical of these, as their parts may be seqlets of their own.
            th = _thresholds(_site_hits([r["motif"] for r in kept], seqs, sites, nthreads, within=True), observed,
                             {r["name"]: r["cwm"] for r in kept}, hit_quantile)
            # A motif whose sites in the seqlets mostly differ in shape from its CWM is not one pattern.
            kept = [r for r in kept if r["name"] not in th or th[r["name"]][2] >= _MIN_SIMILARITY]
            for r in kept:
                sim, imp = th[r["name"]][:2] if r["name"] in th else _DEFAULT_THRESHOLDS
                r["min_similarity"], r["min_importance"] = max(sim, _MIN_SIMILARITY), imp
        if kept:
            # Seqlets go to motifs by their CWMs' hits, as in the final scan. Few sites of a short,
            # degenerate motif such as the initiator reach the P-value cutoff of the sequence matches,
            # though the model weighs them alike.
            reach = _score_reach(kept, seqs, sites, observed, nthreads)
            for r in kept:
                r["seqlets"] = len(reach.get(r["name"], ()))
            # Patterns still form in order of the seqlets the sequence matches reach. A motif given many
            # seqlets by its CWM can have a CWM much like others, such as one weighted on a CpG, and
            # would gather unrelated motifs as the first of a pattern.
            matched = _reach(_site_hits([r["motif"] for r in kept], seqs, sites, nthreads))
            _group(kept, label, pattern_similarity, nthreads, {q: len(v) for q, v in matched.items()})
            for r in kept:
                reach_by_motif[r["name"]] = sub_ids[np.sort(np.fromiter(reach.get(r["name"], ()), dtype=np.int64))]
            # Number the patterns by the seqlets their members reach between them.
            union: dict[str, set] = {}
            for r in kept:
                union.setdefault(r["pattern"], set()).update(reach.get(r["name"], ()))
            order = sorted(union, key=lambda q: (-len(union[q]), int(q.rsplit("_p", 1)[1])))
            new_name = {q: f"{label}_p{k + 1}" for k, q in enumerate(order)}
            for r in kept:
                r["pattern"] = new_name[r["pattern"]]
        rows += kept
    schema = {"name": pl.String, "pattern": pl.String, "sign": pl.String, "source": pl.String, "consensus": pl.String, "width": pl.Int64,
              "seqlets": pl.Int64, "gain": pl.Float64, "cumulative": pl.Float64, "stability": pl.Float64,
              "min_similarity": pl.Float64, "min_importance": pl.Float64}
    table = pl.DataFrame([{k: r[k] for k in schema} for r in rows], schema=schema, orient="row")
    if not stability:
        table = table.drop("stability")
    table = table.with_columns(pl.Series("motif", [r["motif"] for r in rows], dtype=pl.Object),
                               pl.Series("cwm", [r["cwm"] for r in rows], dtype=pl.Object))
    found_hits = None
    if hits:
        found_hits = _scan(table, observed, seqs, scores.names, nthreads)
    return ContributionMotifs(table, seqlets, found_hits, reach_by_motif)


def _search(part, sub, sources, lengths, label, prefix, min_width, max_width, nmotifs, stability, min_stability,
            rng, nthreads) -> list[tuple]:
    """motif_finder() on the seqlets of ``part`` in each source of letters,
    against windows away from every seqlet of ``sub``."""
    si = part["sequence_i"].to_numpy().astype(np.int64)
    s, e = _widen(part["start"].to_numpy(), part["end"].to_numpy(), lengths[si])
    found = []
    for source, letters in sources.items():
        targets = [letters[i][a:b] for i, a, b in zip(si, s, e)]
        back = _background(letters, sub, e - s, np.random.default_rng(rng.integers(2**63)))
        # Sequences with no room beside their seqlets, such as short oligos, fall back to shuffles.
        res = motif_finder(targets, back or None, min_width=min_width, max_width=max_width, nmotifs=nmotifs, rc=True,
                           seed=int(rng.integers(2**63)), nthreads=nthreads, stability=stability, progress=False)
        if stability:
            res = res.filter(pl.col("stability") >= min_stability)
        tag = prefix + ("" if source == "seqlets" else "i")
        for k, (m, st) in enumerate(zip(res["motif"].to_list(),
                                        res["stability"].to_list() if stability else [None] * res.height)):
            found.append((m.replace(name=f"{label}_{tag}{k + 1}"), source, st))
    return found


def _windows(seqs: list[str], sites: pl.DataFrame, motifs: list[Motif]) -> tuple[list[str], pl.DataFrame]:
    """The stretches of the sequences around the sites, merged where they
    meet, and the sites placed in them, row for row. Every hit that can
    reach a site lies within its stretch. Refining and selecting motifs on
    these thus gives what the whole sequences would, for a fraction of the
    scanning."""
    widest = max((m.width + (sum(m.maxgap) if m.isgapped else 0) for m in motifs), default=0)
    # Refinement can widen a motif by a few positions on each side.
    pad = max(_WINDOW_PAD, widest + 2 * _REFINE_GROWTH)
    si = sites["sequence_i"].to_numpy().astype(np.int64)
    start, end = sites["start"].to_numpy(), sites["end"].to_numpy()
    lengths = np.array([len(seqs[i]) for i in si], dtype=np.int64)
    lo, hi = np.maximum(start - pad, 0), np.minimum(end + pad, lengths)
    order = np.lexsort((lo, si))
    window = np.empty(len(si), dtype=np.int64)
    offset = np.empty(len(si), dtype=np.int64)
    out: list[str] = []
    cur_seq, cur_lo, cur_hi = -1, 0, 0
    members: list[int] = []

    def close():
        if members:
            out.append(seqs[cur_seq][cur_lo:cur_hi])
            window[members] = len(out) - 1
            offset[members] = cur_lo

    for k in order:
        if si[k] == cur_seq and lo[k] <= cur_hi:
            cur_hi = max(cur_hi, hi[k])
            members.append(k)
        else:
            close()
            cur_seq, cur_lo, cur_hi, members = si[k], lo[k], hi[k], [k]
    close()
    return out, pl.DataFrame({"sequence_i": window, "start": start - offset, "end": end - offset})


def _choose(found, seqs, sites, observed, sg, refine, min_gain, max_similarity, nthreads):
    """The motifs found, refined, that select_motifs() keeps, as rows of the
    table, and the hits of all of them in the seqlets.

    Motifs too short to reach the P-value cutoff of the sequence matches,
    such as one of six letters, are refined and chosen apart. Refined with
    the others, their exact matches would compete for the same sites, and
    chosen with them, short words common in GC-rich regions would win the
    greedy order over longer motifs. They instead hit at their best score,
    and come after the other motifs, each kept when it reaches at least
    ``min_gain`` of the seqlets not reached yet and, with ``max_similarity``,
    when its CWM differs from those kept."""
    ms = [f[0] for f in found]
    if not ms:
        return [], None
    short = {m.name for m in ms if _unreachable(m)}
    if refine:
        refined = []
        for group, reachable in (([m for m in ms if m.name not in short], False),
                                 ([m for m in ms if m.name in short], True)):
            if group:
                wseqs, wsites = _windows(seqs, sites, group)
                refined += refine_motifs(group, wseqs, sites=wsites, min_overlap=_MIN_OVERLAP, nthreads=nthreads,
                                         reachable=reachable, progress=False)
        place = {m.name: i for i, m in enumerate(ms)}
        ms = sorted(refined, key=lambda m: place[m.name])
    hits = _site_hits(ms, seqs, sites, nthreads)
    cwms = _cwms(hits, observed)
    have = [m for m in ms if m.name in cwms and m.name not in short]
    meta = {f[0].name: f for f in found}
    kept = []
    if have:
        wseqs, wsites = _windows(seqs, sites, have)
        sel = select_motifs(have, wseqs, sites=wsites, min_gain=min_gain, min_overlap=_MIN_OVERLAP,
                            max_similarity=max_similarity,
                            compare_as=[cwms[m.name] for m in have] if max_similarity is not None else None,
                            nthreads=nthreads, progress=False)
        for r in sel.filter("kept").iter_rows(named=True):
            m = r["object"]
            kept.append({"name": m.name, "sign": sg, "source": meta[m.name][1], "consensus": m.consensus or "",
                         "width": m.width, "seqlets": int(r["targets"]), "gain": r["gain"],
                         "cumulative": r["cumulative"], "stability": meta[m.name][2], "motif": m,
                         "cwm": cwms[m.name]})
    shorts = [m for m in ms if m.name in short and m.name in cwms]
    if shorts:
        kept += _choose_short(shorts, hits, cwms, kept, meta, sites.height, sg, min_gain, max_similarity, nthreads)
    return kept, hits


def _unreachable(m: Motif, pvalue: float = 1e-4) -> bool:
    """Whether even the best site of an ungapped motif has a P-value above
    the cutoff."""
    from ._analysis import motif_pvalue
    from ._utils import motif_score

    if m.isgapped:
        return False
    best = float(np.ravel(motif_pvalue(m, score=float(motif_score(m, 1.0))))[0])
    return best > pvalue


def _choose_short(shorts, hits, cwms, kept, meta, n, sg, min_gain, max_similarity, nthreads) -> list[dict]:
    """The short motifs that add enough of the seqlets the motifs kept leave
    unreached, best first, as rows of the table."""
    from ._compare import compare_motifs

    reach = _reach(hits)
    reached = set().union(*(reach.get(r["name"], set()) for r in kept)) if kept else set()
    kept_cwms = [r["cwm"] for r in kept]
    out = []
    left = list(shorts)
    while left:
        m = max(left, key=lambda q: len(reach.get(q.name, set()) - reached))
        left.remove(m)
        new = reach.get(m.name, set()) - reached
        if len(new) < min_gain * n:
            break
        if max_similarity is not None and kept_cwms:
            sim = np.asarray(compare_motifs([*kept_cwms, cwms[m.name]], nthreads=nthreads), dtype=float)
            if sim[-1, :-1].max() >= max_similarity:
                continue
        reached |= new
        out.append({"name": m.name, "sign": sg, "source": meta[m.name][1], "consensus": m.consensus or "",
                    "width": m.width, "seqlets": len(reach[m.name]), "gain": len(new) / n,
                    "cumulative": len(reached) / n, "stability": meta[m.name][2], "motif": m,
                    "cwm": cwms[m.name]})
        kept_cwms.append(cwms[m.name])
    return out


# Sequences matched at a time in the final scan, which holds their scores
# for every letter.
_SCAN_CHUNK = 20000


def _scan(table: pl.DataFrame, observed: np.ndarray, seqs: list[str], names: list[str] | None,
          nthreads: int) -> pl.DataFrame:
    """The hits of the motifs' CWMs in all the scores, at their thresholds."""
    ok = table.filter(pl.col("min_similarity").is_not_null())
    cwms = ok["cwm"].to_list()
    thresholds = ok.select(motif="name", min_similarity="min_similarity", min_importance="min_importance")
    if not cwms:
        return pl.DataFrame(schema={"motif": pl.String, "motif_i": pl.UInt32, "sequence": pl.String,
                                    "sequence_i": pl.Int64, "start": pl.Int64, "end": pl.Int64,
                                    "strand": pl.String, "score": pl.Float64, "similarity": pl.Float64,
                                    "importance": pl.Float64, "match": pl.String})
    parts = []
    for lo in range(0, len(seqs), _SCAN_CHUNK):
        hi = min(lo + _SCAN_CHUNK, len(seqs))
        h = scan_contributions(cwms, observed[lo:hi], sequences=seqs[lo:hi], thresholds=thresholds,
                               nthreads=nthreads)
        parts.append(h.with_columns(pl.col("sequence_i").cast(pl.Int64) + lo))
    hits = pl.concat(parts)
    composites = _composite_parts(ok.select("name", "motif", "cwm", "min_importance").iter_rows())
    if composites and hits.height:
        hits = hits.filter(pl.Series(_parts_kept(
            hits["motif"].cast(pl.String).to_numpy(), hits["sequence_i"].to_numpy(), hits["start"].to_numpy(),
            hits["end"].to_numpy(), hits["strand"].to_numpy() == "-", observed, composites)))
    i = hits["sequence_i"].to_numpy()
    label = np.asarray(names, dtype=object)[i] if names else np.char.add("seq", (i + 1).astype(str))
    return hits.with_columns(pl.Series("sequence", label, dtype=pl.String))


def _group(kept: list[dict], label: str, similarity: float | None, nthreads: int,
           matched: dict[str, int]) -> None:
    """Give each kept motif a pattern. In order of the seqlets their
    sequence matches reach, in ``matched``, each motif joins the first
    pattern whose first motif's CWM is at least ``similarity`` like its own
    by cosine, or starts a pattern of its own. Composites always start their
    own, as one of their parts is often another pattern."""
    from ._compare import compare_motifs

    order = sorted(range(len(kept)), key=lambda k: (-matched.get(kept[k]["name"], 0), k))
    sim = None
    if similarity is not None and len(kept) > 1:
        sim = np.asarray(compare_motifs([r["cwm"] for r in kept], nthreads=nthreads), dtype=float)
    reps: list[int] = []
    for k in order:
        r = kept[k]
        home = None
        if sim is not None and r["source"] != "gapped":
            home = next((j for j in reps if kept[j]["source"] != "gapped" and sim[k, j] >= similarity), None)
        if home is None:
            reps.append(k)
            r["pattern"] = f"{label}_p{len(reps)}"
        else:
            r["pattern"] = kept[home]["pattern"]


def _narrow(m: Motif, hits: pl.DataFrame) -> Motif | None:
    """A gapped motif with its gap range cut to the spacings around its
    commonest one that hold at least half as many hits. None is returned
    unless its parts are then at least _MIN_SPACER apart, as composite
    sites are."""
    if len(m.gaploc) != 1:
        return None
    sub = hits.filter(pl.col("motif") == m.name).unique(["sequence_i", "start", "strand"])
    counts = np.bincount((sub["end"] - sub["start"]).to_numpy() - m.width, minlength=m.maxgap[0] + 1)
    if not counts.sum():
        return None
    lo = hi = int(counts.argmax())
    while lo > 0 and counts[lo - 1] * 2 >= counts.max():
        lo -= 1
    while hi + 1 < len(counts) and counts[hi + 1] * 2 >= counts.max():
        hi += 1
    if lo < _MIN_SPACER:
        return None
    return add_gap(m, gaploc=m.gaploc, mingap=lo, maxgap=hi).replace(name=m.name)


def _gapped_pass(seqs, sub, sites, si, lengths, observed, reached, label, sg, max_gap, min_gain, max_similarity,
                 refine, rng, nthreads) -> list[dict]:
    """Composite motifs, of two parts at least _MIN_SPACER apart, from
    gapped_motif_finder() on the seqlets widened to fit them. Each must add
    at least ``min_gain`` of the seqlets to those reached already. With
    ``max_similarity``, its CWM must also differ from those of the
    composites kept before it. It is not compared with ungapped motifs,
    which often match one of its parts."""
    from ._compare import compare_motifs

    s, e = _widen(sub["start"].to_numpy(), sub["end"].to_numpy(), lengths[si], _MIN_GAPPED_TARGET)
    targets = [seqs[i][a:b] for i, a, b in zip(si, s, e)]
    bkg = _background(seqs, sub, e - s, np.random.default_rng(rng.integers(2**63)))
    res = gapped_motif_finder(targets, bkg or None, max_gap=max_gap, nmotifs=10, seed=int(rng.integers(2**63)),
                              nthreads=nthreads, progress=False)
    ms = [m.replace(name=f"{label}_g{k + 1}") for k, m in enumerate(res["motif"].to_list())]
    if refine and ms:
        wseqs, wsites = _windows(seqs, sites, ms)
        ms = refine_motifs(ms, wseqs, sites=wsites, min_overlap=_MIN_OVERLAP, nthreads=nthreads, progress=False)
    if not ms:
        return []
    hits = _site_hits(ms, seqs, sites, nthreads)
    ms = [n for n in (_narrow(m, hits) for m in ms) if n is not None]
    if not ms:
        return []
    hits = _site_hits(ms, seqs, sites, nthreads)
    cwms, reach = _cwms(hits, observed), _reach(hits)
    n = sites.height
    order = sorted((m for m in ms if m.name in cwms), key=lambda m: -len(reach[m.name] - reached))
    out, kept_cwms = [], []
    for m in order:
        new = reach[m.name] - reached
        if len(new) < min_gain * n:
            continue
        if max_similarity is not None and kept_cwms:
            sim = np.asarray(compare_motifs([*kept_cwms, cwms[m.name]], nthreads=nthreads), dtype=float)
            if sim[-1, :-1].max() >= max_similarity:
                continue
        reached |= new
        out.append({"name": m.name, "sign": sg, "source": "gapped", "consensus": m.consensus or "",
                    "width": m.width, "seqlets": len(reach[m.name]), "gain": len(new) / n,
                    "cumulative": len(reached) / n, "stability": None, "motif": m, "cwm": cwms[m.name]})
        kept_cwms.append(cwms[m.name])
    return out


def _given_idealized(idealized: Any, seqs: list[str], sign: str) -> dict[str, list[str]]:
    """Idealized sequences for each sign searched, checked against the
    sequences."""
    if isinstance(idealized, Mapping) and idealized and set(idealized) <= {"positive", "negative"}:
        given = dict(idealized)
    elif sign == "both":
        raise ValueError("with sign='both', give idealized as a dict of 'positive' and 'negative' sequences")
    else:
        given = {sign: idealized}
    out = {}
    for key, value in given.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            from ._deep import from_one_hot

            value = from_one_hot(value)
        values = [value] if isinstance(value, str) else list(value.values() if isinstance(value, Mapping) else value)
        if len(values) != len(seqs) or any(len(a) != len(b) for a, b in zip(values, seqs)):
            raise ValueError(f"the {key} idealized sequences must match the sequences in number and length")
        out[key] = [str(v) for v in values]
    return out


def _given_seqlets(seqlets: pl.DataFrame, names: list[str] | None, lengths: np.ndarray) -> pl.DataFrame:
    n = len(lengths)
    for col in ("start", "end"):
        if col not in seqlets.columns:
            raise ValueError(f"seqlets need a {col!r} column")
    if "sequence_i" not in seqlets.columns:
        if "sequence" not in seqlets.columns or names is None:
            raise ValueError("seqlets need a 'sequence_i' column, or a 'sequence' column with named sequences")
        index = {nm: i for i, nm in enumerate(names)}
        missing = set(seqlets["sequence"].cast(pl.String)) - set(index)
        if missing:
            raise ValueError(f"seqlets name sequences not given, such as {min(missing)!r}")
        seqlets = seqlets.with_columns(sequence_i=pl.col("sequence").cast(pl.String).replace_strict(index))
    if ((seqlets["sequence_i"] < 0) | (seqlets["sequence_i"] >= n)).any():
        raise ValueError(f"seqlets refer to sequences beyond the {n} given")
    seqlets = seqlets.with_columns(pl.col("sequence_i").cast(pl.Int64), pl.col("start").cast(pl.Int64),
                                   pl.col("end").cast(pl.Int64))
    # Positions count from the start of each sequence, not along a chromosome.
    size = lengths[seqlets["sequence_i"].to_numpy()]
    start, end = seqlets["start"].to_numpy(), seqlets["end"].to_numpy()
    bad = (start < 0) | (end <= start) | (end > size)
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"a seqlet runs from {start[k]} to {end[k]} in a sequence of {size[k]} letters; "
                         "seqlet positions count from 0 at the start of each sequence")
    return seqlets


__all__ = ["ContributionMotifs", "contribution_motifs"]
