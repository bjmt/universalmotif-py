"""Finding motifs in a sequence model's contribution scores: seqlets, motif
discovery in them, refinement and selection, in one call."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
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
# The fewest letters between the parts of a composite motif.
_MIN_SPACER = 2
# The positions a hit must share with a seqlet to count as reaching it.
_MIN_OVERLAP = 5


@dataclass
class ContributionMotifs:
    """The result of
    [`contribution_motifs()`][universalmotif.contribution_motifs].

    ``motifs`` has one row per motif kept, in the order they were chosen,
    and ``seqlets`` the seqlets they were found in and chosen to reach.
    ``ppms`` and ``cwms`` give the motifs as lists, ``thresholds`` their hit
    thresholds, and ``hits`` holds the hits of the CWMs in the scores.
    """

    motifs: pl.DataFrame
    seqlets: pl.DataFrame
    hits: pl.DataFrame | None = None

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

    def __len__(self) -> int:
        return self.motifs.height

    def __repr__(self) -> str:
        return f"ContributionMotifs({self.motifs.height} motifs from {self.seqlets.height:,} seqlets)"


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
    lengths = np.array([len(s) for s in seqs])
    # The stretches each window must avoid, merged and sorted, per sequence.
    by_seq = seqlets.group_by("sequence_i").agg("start", "end")
    spans = {int(i): (list(s), list(e)) for i, s, e in by_seq.iter_rows()}
    blocked: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, s in enumerate(seqs):
        starts, ends = spans.get(i, ([], []))
        for m in _NOT_ACGT.finditer(s):
            starts.append(m.start())
            ends.append(m.end())
        if starts:
            order = np.argsort(starts)
            st, en = np.asarray(starts)[order], np.maximum.accumulate(np.asarray(ends)[order])
            blocked[i] = (np.maximum(st, 0), en)
    want = np.repeat(widths, ratio)
    out, todo = [None] * len(want), np.arange(len(want))
    for _ in range(200):
        if not len(todo):
            break
        seq = rng.integers(0, len(seqs), len(todo))
        room = lengths[seq] - want[todo]
        ok = room >= 0
        start = (rng.random(len(todo)) * np.maximum(room + 1, 1)).astype(np.int64)
        free = np.zeros(len(todo), dtype=bool)
        for k in np.flatnonzero(ok):
            b = blocked.get(int(seq[k]))
            if b is None:
                free[k] = True
                continue
            # The first stretch ending after the window starts must begin at or after its end.
            j = np.searchsorted(b[1], start[k], side="right")
            free[k] = j == len(b[1]) or b[0][j] >= start[k] + want[todo[k]]
        for k in np.flatnonzero(free):
            out[todo[k]] = seqs[seq[k]][start[k]:start[k] + want[todo[k]]]
        todo = todo[~free]
    return [w for w in out if w is not None]


def _site_hits(motifs: list[Motif], seqs: list[str], sites: pl.DataFrame, nthreads: int,
               within: bool = False) -> pl.DataFrame:
    """The motifs' hits that reach the sites, or with ``within`` that lie
    within them, with the index of the site in ``site``. Only windows around
    the sites wide enough to hold such hits are scanned."""
    from ._analysis import scan_sequences

    pad = max(m.width + (sum(m.maxgap) if m.isgapped else 0) for m in motifs)
    si = sites["sequence_i"].to_numpy()
    lengths = np.array([len(seqs[i]) for i in si], dtype=np.int64)
    lo = np.maximum(sites["start"].to_numpy() - pad, 0)
    hi = np.minimum(sites["end"].to_numpy() + pad, lengths)
    windows = [seqs[i][a:b] for i, a, b in zip(si, lo, hi)]
    with warnings.catch_warnings():
        # Motifs too short to reach the cutoff simply have no hits.
        warnings.simplefilter("ignore")
        hits = scan_sequences(motifs, windows, nthreads=nthreads)
    w = hits["sequence_i"].to_numpy()
    hits = hits.drop("sequence", "sequence_i", strict=False).with_columns(
        pl.Series("site", w, dtype=pl.UInt32), pl.Series("sequence_i", si[w], dtype=pl.Int64),
        pl.col("start") + pl.Series(lo[w]), pl.col("end") + pl.Series(lo[w]),
        s=pl.Series(sites["start"].to_numpy()[w]), e=pl.Series(sites["end"].to_numpy()[w]))
    return (hits.filter((pl.col("start") >= pl.col("s")) & (pl.col("end") <= pl.col("e")) if within else
                        pl.min_horizontal("end", "e") - pl.max_horizontal("start", "s") >= _MIN_OVERLAP)
            .drop("s", "e"))


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
    seqlets, scored as it scores a window: the ``quantile`` of their
    similarity, the cosine of their scores with the CWM, and of their
    importance, the sum of their absolute scores as a share of the CWM's."""
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
        out[name] = (float(np.quantile(sim, quantile)), float(np.quantile(imp, quantile)))
    return out


def _reach(hits: pl.DataFrame) -> dict[str, set[int]]:
    return {name: set(s) for name, s in hits.group_by("motif").agg(pl.col("site").unique()).iter_rows()}


def contribution_motifs(
    contributions: Any,
    sequences: Any = None,
    hypothetical: Any = None,
    seqlets: pl.DataFrame | None = None,
    decoys: Any = None,
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
    seed: int | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
) -> ContributionMotifs:
    """Find the motifs a sequence model relies on, from its contribution
    scores.

    The steps are those of the functions named, with options suited to
    attributions:

    1. Seqlets are called in the scores with
       [`call_seqlets()`][universalmotif.call_seqlets], unless ``seqlets``
       are given, such as those of
       [`read_seqlets()`][universalmotif.read_seqlets].
    2. [`motif_finder()`][universalmotif.motif_finder] searches the seqlets'
       letters, widened to at least 20, against twice as many windows of
       the same widths from the sequences away from any seqlet. With
       ``hypothetical`` scores, it also searches the seqlets rewritten in
       the letters the model prefers by
       [`idealize_sequences()`][universalmotif.idealize_sequences], which
       finds degenerate sites, such as spaced palindromes, whose letters
       vary too much for a plain search.
    3. With ``stability``, each search is rerun that many times on half of
       the seqlets, and motifs found again in fewer than ``min_stability``
       of the reruns are left out.
    4. With ``refine``, [`refine_motifs()`][universalmotif.refine_motifs]
       rebuilds every motif from its hits in the seqlets of the real
       sequences.
    5. [`select_motifs()`][universalmotif.select_motifs] keeps the motifs
       that each reach at least ``min_gain`` of the seqlets not reached by
       those before them, leaving out, with ``max_similarity``, those whose
       CWMs repeat one already kept, the CWMs being the mean contributions
       at their hits in the seqlets.
    6. With ``rounds`` of 2 or more, the searches are run again on the
       seqlets that no motif kept reaches, and everything found so far is
       refined and selected again. Rare motifs, a smaller share of all the
       seqlets than of those left over, are found more reliably this way.
    7. With ``gapped``,
       [`gapped_motif_finder()`][universalmotif.gapped_motif_finder]
       searches the seqlets widened to 30 letters for composite motifs,
       made of two parts at least 2 letters apart. Each one's gap range is
       cut to the spacings around its commonest that hold at least half as
       many hits, and it is kept when it adds ``min_gain`` of the seqlets
       to those reached. Since one of its parts is often a motif kept
       already, its CWM is only compared with those of other composites.
    8. With ``hits``, the motifs' CWMs are matched against all the scores
       by [`scan_contributions()`][universalmotif.scan_contributions], with
       thresholds for each motif set from its hits within the seqlets as
       [`modisco_thresholds()`][universalmotif.modisco_thresholds] sets
       them from a pattern's seqlets: the ``hit_quantile`` of their
       similarity and of their importance. Motifs with no hit within a
       seqlet, such as composites, whose parts are often seqlets of their
       own, take the defaults of ``scan_contributions()``.

    Each sign of ``sign`` is searched on its own. The defaults aim at
    finding the patterns a model has learned, much as TF-MoDISco does, and
    keep rarer motifs at the cost of reaching fewer seqlets with each; for
    a short list of motifs that reach as many seqlets as possible, use
    ``min_gain=0.01`` and ``max_similarity=None``. Rarer motifs, such as
    those found only in idealized seqlets, are often missing from the
    half-sized reruns, so ``stability`` trades them for motifs that other
    samples of regions would give too.

    On the contribution scores of a chromBPNet model of chromatin
    accessibility in K562 cells, over 21,000 regions, the defaults with
    hypothetical scores recovered more of the model's published TF-MoDISco
    patterns closely (a CWM cosine of 0.8 or more) than TF-MoDISco did on
    the same regions, 28 against 25, in 15 seconds against its 40 minutes.
    These included the NFI palindrome, held by about 1% of the seqlets, and
    the TAL1–GATA composite, CTG then WGATAA 7 or 8 letters on. Without
    hypothetical scores, NFI was found in only some of the runs. The hits
    were about as many as those of the published CWMs at the thresholds
    of ``modisco_thresholds()``, and fewer of them fell outside the
    published seqlets. Over all 187,000 of the model's regions, the call
    took two minutes on two threads, with a peak of 2.6 GB, half of it the
    scores and sequences themselves.

    The motifs found in different samples of regions differ mostly in the
    rarer ones: in three random 70% samples of the 30,000 regions, the
    same 25 published patterns were recovered every time and up to four
    others only some of the time, while TF-MoDISco recovered the same 25
    each time and no others. Keeping fewer near-identical motifs, with a
    lower ``max_similarity``, made the motifs themselves match more
    often across samples but lost some of those patterns.

    Parameters
    ----------
    contributions
        The scores, in any form
        [`hit_contributions()`][universalmotif.hit_contributions] takes,
        such as a [`ContributionScores`][universalmotif.ContributionScores]
        or an array of shape (sequences, 4, length) or (sequences,
        length).
    sequences
        The sequences the scores are for, as text or one-hot encoded; not
        needed with ContributionScores, or with projected scores, zero for
        all but the letter present.
    hypothetical
        Hypothetical scores, the score of every letter at every position,
        for the search on idealized seqlets: an array of shape (sequences,
        4, length) or a ContributionScores of hypothetical scores.
    seqlets
        Seqlets to use instead of calling them, as a table with
        ``sequence_i`` (or ``sequence``, naming the sequences), ``start`` and
        ``end`` columns, and a ``sign`` column of ``+`` or ``-`` to search
        both signs.
    decoys
        Decoy scores for calling seqlets, such as the model's contributions
        for shuffled sequences; see ``call_seqlets()``.
    sign
        ``"positive"``, ``"negative"`` or ``"both"``.
    min_width, max_width, nmotifs
        As for ``motif_finder()``.
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
    seed
        Seed for the background, the reruns and the decoy shuffles.
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
        and with composites last, with the columns ``name``, ``sign``,
        ``source`` (``seqlets``, ``idealized`` or ``gapped``), ``consensus``,
        ``width``, ``seqlets`` (those its hits reach), ``gain`` and
        ``cumulative`` (the shares of the seqlets of its sign it adds and
        that the motifs up to it reach), ``stability`` with reruns,
        ``min_similarity`` and ``min_importance``, its hit thresholds, and
        the motif as a PPM in ``motif`` and as a CWM in ``cwm``, that of a
        composite at its commonest spacing. Names of motifs from later
        rounds start with ``r``, and those of composites with ``g``. Its
        ``seqlets`` table holds the seqlets searched, and ``hits`` the hits
        of the CWMs, as ``scan_contributions()`` gives them.

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
    [('pos_1', 'TGACTCA', 150)]
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: contribution_motifs(**args))
    if sign not in ("positive", "negative", "both"):
        raise ValueError("sign must be 'positive', 'negative' or 'both'")
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
        seqlets = call_seqlets(observed, seqs, decoys=decoys, sign=sign, seed=seed, nthreads=nthreads)
    else:
        seqlets = _given_seqlets(seqlets, scores.names, len(seqs))
    signs = {"positive": ["+"], "negative": ["-"], "both": ["+", "-"]}[sign]
    ideal = {}
    rows = []
    for sg in signs:
        sub = seqlets.filter(pl.col("sign") == sg) if "sign" in seqlets.columns else seqlets
        if sub.height < 20:
            continue
        label = "pos" if sg == "+" else "neg"
        sources = {"seqlets": seqs}
        if hypothetical is not None:
            key = "positive" if sg == "+" else "negative"
            if key not in ideal:
                ideal[key] = idealize_sequences(seqs, hypothetical, threshold=idealize_threshold, sign=key)
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
            # Thresholds come from the hits within the seqlets. Motifs wider than their seqlets, such as
            # composites, whose parts may be seqlets of their own, take the defaults instead, as hits only
            # overlapping a seqlet are mostly positions the model gave little weight.
            th = _thresholds(_site_hits([r["motif"] for r in kept], seqs, sites, nthreads, within=True), observed,
                             {r["name"]: r["cwm"] for r in kept}, hit_quantile)
            for r in kept:
                r["min_similarity"], r["min_importance"] = th.get(r["name"], _DEFAULT_THRESHOLDS)
        rows += kept
    schema = {"name": pl.String, "sign": pl.String, "source": pl.String, "consensus": pl.String, "width": pl.Int64,
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
    return ContributionMotifs(table, seqlets, found_hits)


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
        res = motif_finder(targets, back, min_width=min_width, max_width=max_width, nmotifs=nmotifs, rc=True,
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
    reach a site lies within its stretch, so refining and selecting motifs
    on these gives what the whole sequences would, for a fraction of the
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
    table, and the hits of all of them in the seqlets."""
    ms = [f[0] for f in found]
    if not ms:
        return [], None
    wseqs, wsites = _windows(seqs, sites, ms)
    if refine:
        ms = refine_motifs(ms, wseqs, sites=wsites, min_overlap=_MIN_OVERLAP, nthreads=nthreads, progress=False)
    hits = _site_hits(ms, seqs, sites, nthreads)
    cwms = _cwms(hits, observed)
    have = [m for m in ms if m.name in cwms]
    if not have:
        return [], hits
    meta = {f[0].name: f for f in found}
    wseqs, wsites = _windows(seqs, sites, have)
    sel = select_motifs(have, wseqs, sites=wsites, min_gain=min_gain, min_overlap=_MIN_OVERLAP,
                        max_similarity=max_similarity,
                        compare_as=[cwms[m.name] for m in have] if max_similarity is not None else None,
                        nthreads=nthreads, progress=False)
    kept = []
    for r in sel.filter("kept").iter_rows(named=True):
        m = r["object"]
        kept.append({"name": m.name, "sign": sg, "source": meta[m.name][1], "consensus": m.consensus or "",
                     "width": m.width, "seqlets": int(r["targets"]), "gain": r["gain"],
                     "cumulative": r["cumulative"], "stability": meta[m.name][2], "motif": m, "cwm": cwms[m.name]})
    return kept, hits


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
        return pl.DataFrame(schema={"motif": pl.String, "sequence": pl.String, "sequence_i": pl.Int64,
                                    "start": pl.Int64, "end": pl.Int64, "strand": pl.String})
    parts = []
    for lo in range(0, len(seqs), _SCAN_CHUNK):
        hi = min(lo + _SCAN_CHUNK, len(seqs))
        h = scan_contributions(cwms, observed[lo:hi], sequences=seqs[lo:hi], thresholds=thresholds,
                               nthreads=nthreads)
        parts.append(h.with_columns(pl.col("sequence_i").cast(pl.Int64) + lo))
    hits = pl.concat(parts)
    i = hits["sequence_i"].to_numpy()
    label = np.asarray(names, dtype=object)[i] if names else np.char.add("seq", (i + 1).astype(str))
    return hits.with_columns(pl.Series("sequence", label, dtype=pl.String))


def _narrow(m: Motif, hits: pl.DataFrame) -> Motif | None:
    """A gapped motif with its gap range cut to the spacings around its
    commonest one that hold at least half as many hits, or None unless its
    parts are then at least _MIN_SPACER apart, as composite sites are."""
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
    at least ``min_gain`` of the seqlets to those reached already, and with
    ``max_similarity`` its CWM must differ from those of the composites kept
    before it, though not from those of ungapped motifs, which often match
    one of its parts."""
    from ._compare import compare_motifs

    s, e = _widen(sub["start"].to_numpy(), sub["end"].to_numpy(), lengths[si], _MIN_GAPPED_TARGET)
    targets = [seqs[i][a:b] for i, a, b in zip(si, s, e)]
    bkg = _background(seqs, sub, e - s, np.random.default_rng(rng.integers(2**63)))
    res = gapped_motif_finder(targets, bkg, max_gap=max_gap, nmotifs=10, seed=int(rng.integers(2**63)),
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


def _given_seqlets(seqlets: pl.DataFrame, names: list[str] | None, n: int) -> pl.DataFrame:
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
    return seqlets.with_columns(pl.col("sequence_i").cast(pl.Int64), pl.col("start").cast(pl.Int64),
                                pl.col("end").cast(pl.Int64))


__all__ = ["ContributionMotifs", "contribution_motifs"]
