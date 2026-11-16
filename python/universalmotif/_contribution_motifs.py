"""Finding motifs in a sequence model's contribution scores: seqlets, motif
discovery in them, refinement and selection, in one call."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from . import _progress
from ._core import Motif
from ._deep import idealize_sequences
from ._hit_contrib import _LETTERS, _gather_scores, hit_contributions
from ._select import refine_motifs, select_motifs
from ._seqlets import call_seqlets
from ._sequences import _seed, motif_finder
from ._threads import _nthreads

# Seqlets shorter than this are widened around their middles for discovery.
_MIN_TARGET = 20
# The positions a hit must share with a seqlet to count as reaching it.
_MIN_OVERLAP = 5


@dataclass
class ContributionMotifs:
    """The result of
    [`contribution_motifs()`][universalmotif.contribution_motifs].

    ``motifs`` has one row per motif kept, in the order they were chosen,
    and ``seqlets`` the seqlets they were found in and chosen to reach.
    ``ppms`` and ``cwms`` give the motifs as lists.
    """

    motifs: pl.DataFrame
    seqlets: pl.DataFrame

    @property
    def ppms(self) -> list[Motif]:
        """The motifs, as PPMs."""
        return self.motifs["motif"].to_list()

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


def _widen(start: np.ndarray, end: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windows of at least _MIN_TARGET letters around each seqlet, kept
    within their sequences."""
    grow = np.maximum(_MIN_TARGET - (end - start), 0)
    s, e = start - grow // 2, end + (grow - grow // 2)
    shift = np.where(s < 0, -s, 0) - np.where(e > lengths, e - lengths, 0)
    return np.maximum(s + shift, 0), np.minimum(e + shift, lengths)


def _background(seqs: list[str], seqlets: pl.DataFrame, widths: np.ndarray, rng, ratio: int = 2) -> list[str]:
    """Windows of the seqlets' widths drawn from the sequences where there is
    no seqlet and no letter outside ACGT, ``ratio`` per seqlet."""
    lengths = np.array([len(s) for s in seqs])
    blocked = []
    by_seq = seqlets.group_by("sequence_i").agg("start", "end")
    spans = {int(i): (np.array(s), np.array(e)) for i, s, e in by_seq.iter_rows()}
    for i, s in enumerate(seqs):
        bad = np.frombuffer(s.upper().encode("latin-1", "replace"), dtype=np.uint8)
        b = ~np.isin(bad, np.frombuffer(b"ACGT", dtype=np.uint8))
        if i in spans:
            for a, z in zip(*spans[i]):
                b[max(a, 0):z] = True
        blocked.append(np.concatenate([[0], np.cumsum(b)]))
    want = np.repeat(widths, ratio)
    out, todo = [None] * len(want), np.arange(len(want))
    for _ in range(200):
        if not len(todo):
            break
        seq = rng.integers(0, len(seqs), len(todo))
        room = lengths[seq] - want[todo]
        ok = room >= 0
        start = (rng.random(len(todo)) * np.maximum(room + 1, 1)).astype(np.int64)
        free = np.array([ok[k] and blocked[seq[k]][start[k] + want[todo[k]]] == blocked[seq[k]][start[k]]
                         for k in range(len(todo))], dtype=bool)
        for k in np.flatnonzero(free):
            out[todo[k]] = seqs[seq[k]][start[k]:start[k] + want[todo[k]]]
        todo = todo[~free]
    return [w for w in out if w is not None]


def _cwms(motifs: list[Motif], seqs: list[str], sites: pl.DataFrame, observed: Any, nthreads: int) -> dict:
    """The CWM of each motif's hits that reach the sites."""
    from ._analysis import scan_sequences

    if not motifs:
        return {}
    with warnings.catch_warnings():
        # Motifs too short to reach the cutoff simply have no CWM.
        warnings.simplefilter("ignore")
        hits = scan_sequences(motifs, seqs, nthreads=nthreads)
    hits = (hits.join(sites.select("sequence_i", s="start", e="end"), on="sequence_i")
            .filter(pl.min_horizontal("end", "e") - pl.max_horizontal("start", "s") >= _MIN_OVERLAP)
            .unique(["motif", "sequence_i", "start", "strand"], maintain_order=True)
            .drop("s", "e"))
    if not hits.height:
        return {}
    return {m.name: m for m in hit_contributions(hits, observed).to_cwms()}


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

    Each sign of ``sign`` is searched on its own. The defaults aim at
    finding the patterns a model has learned, much as TF-MoDISco does, and
    keep rarer motifs at the cost of reaching fewer seqlets with each; for
    a short list of motifs that reach as many seqlets as possible, use
    ``min_gain=0.01`` and ``max_similarity=None``. Rarer motifs, such as
    those found only in idealized seqlets, are often missing from the
    half-sized reruns, so ``stability`` trades them for motifs that other
    samples of regions would give too.

    On the contribution scores of a chromBPNet model of chromatin
    accessibility, over 21,000 regions, the defaults with hypothetical
    scores recover about as many of the model's published TF-MoDISco
    patterns closely as TF-MoDISco does on the same regions, in 12 seconds
    against its 40 minutes. Rare degenerate sites remain the weak point: the
    model's NFI palindrome, held by about 1% of the seqlets, was found in
    some runs and missed in others, depending on the seed.

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
        Its ``motifs`` table has one row per motif kept, in the order chosen,
        with the columns ``name``, ``sign``, ``source`` (``seqlets`` or
        ``idealized``), ``consensus``, ``width``, ``seqlets`` (those its hits
        reach), ``gain`` and ``cumulative`` (the shares of the seqlets of its
        sign it adds and that the motifs up to it reach), ``stability`` with
        reruns, and the motif as a PPM in ``motif`` and as a CWM in
        ``cwm``. Its ``seqlets`` table holds the seqlets searched.

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
        si = sub["sequence_i"].to_numpy().astype(np.int64)
        s, e = _widen(sub["start"].to_numpy(), sub["end"].to_numpy(), lengths[si])
        bkg_rng = np.random.default_rng(rng.integers(2**63))
        bkg = _background(seqs, sub, e - s, bkg_rng)
        sources = {"seqlets": seqs}
        if hypothetical is not None:
            key = "positive" if sg == "+" else "negative"
            if key not in ideal:
                ideal[key] = idealize_sequences(seqs, hypothetical, threshold=idealize_threshold, sign=key)
            sources["idealized"] = ideal[key]
        found = []
        for source, letters in sources.items():
            targets = [letters[i][a:b] for i, a, b in zip(si, s, e)]
            back = bkg if source == "seqlets" else _background(letters, sub, e - s,
                                                               np.random.default_rng(rng.integers(2**63)))
            res = motif_finder(targets, back, min_width=min_width, max_width=max_width, nmotifs=nmotifs, rc=True,
                               seed=int(rng.integers(2**63)), nthreads=nthreads, stability=stability,
                               progress=False)
            if stability:
                res = res.filter(pl.col("stability") >= min_stability)
            tag = "" if source == "seqlets" else "i"
            for k, (m, st) in enumerate(zip(res["motif"].to_list(),
                                            res["stability"].to_list() if stability else [None] * res.height)):
                found.append((m.replace(name=f"{label}_{tag}{k + 1}"), source, st))
        if not found:
            continue
        sites = sub.select("sequence_i", "start", "end")
        ms = [f[0] for f in found]
        if refine:
            ms = refine_motifs(ms, seqs, sites=sites, min_overlap=_MIN_OVERLAP, nthreads=nthreads, progress=False)
        cwms = _cwms(ms, seqs, sites, observed, nthreads)
        have = [m for m in ms if m.name in cwms]
        if not have:
            continue
        sel = select_motifs(have, seqs, sites=sites, min_gain=min_gain, min_overlap=_MIN_OVERLAP,
                            max_similarity=max_similarity,
                            compare_as=[cwms[m.name] for m in have] if max_similarity is not None else None,
                            nthreads=nthreads, progress=False)
        meta = {f[0].name: f for f in found}
        for r in sel.filter("kept").iter_rows(named=True):
            m = r["object"]
            rows.append({"name": m.name, "sign": sg, "source": meta[m.name][1], "consensus": m.consensus or "",
                         "width": m.width, "seqlets": int(r["targets"]), "gain": r["gain"],
                         "cumulative": r["cumulative"], "stability": meta[m.name][2], "motif": m,
                         "cwm": cwms[m.name]})
    schema = {"name": pl.String, "sign": pl.String, "source": pl.String, "consensus": pl.String, "width": pl.Int64,
              "seqlets": pl.Int64, "gain": pl.Float64, "cumulative": pl.Float64, "stability": pl.Float64}
    table = pl.DataFrame([{k: r[k] for k in schema} for r in rows], schema=schema, orient="row")
    if not stability:
        table = table.drop("stability")
    table = table.with_columns(pl.Series("motif", [r["motif"] for r in rows], dtype=pl.Object),
                               pl.Series("cwm", [r["cwm"] for r in rows], dtype=pl.Object))
    return ContributionMotifs(table, seqlets)


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
