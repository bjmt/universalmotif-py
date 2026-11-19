"""Calling seqlets in a sequence model's contribution scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from . import _core, _progress
from ._hit_contrib import _LETTERS, _gather_scores
from ._sequences import _seed
from ._threads import _nthreads


def _observed(contributions: Any, sequences: Any) -> tuple[np.ndarray, list[int], list[str], np.ndarray | None]:
    """The observed score of each position of each sequence as one
    contiguous array, the sequences' lengths and names, and their letters
    as codes when known."""
    scores = _gather_scores(contributions, sequences, None)
    values = scores.values
    if values.ndim == 3:
        if scores.codes is not None:
            onehot = scores.codes[:, None, :] == np.arange(len(_LETTERS))[None, :, None]
            values = (values * onehot).sum(axis=1)
        else:
            values = values.sum(axis=1)
    names = scores.names or [f"seq{i + 1}" for i in range(values.shape[0])]
    lengths = [int(n) for n in scores.lengths]
    # Single precision, which the extension reads in place, halves the memory of a copy.
    return np.ascontiguousarray(values, dtype=np.float32), lengths, names, scores.codes


def call_seqlets(
    contributions: Any,
    sequences: Any = None,
    null_scores: Any = None,
    null_sequences: Any = None,
    threshold: float = 0.01,
    min_len: int = 4,
    max_len: int = 25,
    flank: int = 3,
    fdr: float | None = 0.2,
    sign: str = "both",
    n_bins: int = 1000,
    passes: int = 1,
    seed: int | None = None,
    nthreads: int | None = None,
    progress: bool | None = None,
) -> pl.DataFrame:
    """Call seqlets, the short stretches of high contribution, in a sequence
    model's contribution scores.

    A stretch of positions is a seqlet when its total contribution is
    unlikely under a null. The total of every stretch inside it of at
    least ``min_len`` positions must be unlikely too, so that a seqlet
    holds no run of unremarkable positions. Seqlets from ``min_len`` to
    ``max_len`` positions long are called from the longest down, in order
    of P-value, without overlapping one another. They are then widened by
    ``flank`` positions on either side. Positive and negative seqlets,
    which raise and lower the prediction, are each called from their own
    null.

    The null comes from null scores, the scores of sequences with nothing
    to find. The null for each length comes from their distribution, so a
    set share of stretches passes even in noise. With ``fdr``, the same
    calls are also made on the null scores. Seqlets are then kept down to
    the total at which the seqlets of the null scores make up at most
    ``fdr`` of those passing, counting the null scores as if they had as
    many positions as the scores.

    The best null scores are the model's own contributions for sequences
    with nothing to find, such as shuffled versions of the sequences from
    [`shuffled_references()`][universalmotif.shuffled_references]. These
    show how the model treats sequence without sites. Without them, each
    sequence's scores are shuffled to stand in for them. Shuffled scores
    keep the scores' values but not the model's view of empty sequence, so
    they let through more seqlets of weak contribution.

    Batches of sequences called with the same ``null_scores`` all get the
    same P-values, so together they give the seqlets of one call. Calling
    each batch with ``fdr=None`` keeps every candidate, and
    [`seqlet_fdr()`][universalmotif.seqlet_fdr] then applies one false
    discovery rate to them all. This way scores can be called as a model
    produces them, without keeping them all.

    The seqlets can go straight to
    [`motif_finder()`][universalmotif.motif_finder] as the sequences of
    their windows. They can also go to
    [`select_motifs()`][universalmotif.select_motifs] and
    [`refine_motifs()`][universalmotif.refine_motifs] as the sites that
    motifs should reach.

    Parameters
    ----------
    contributions
        The scores, in any of the forms
        [`hit_contributions()`][universalmotif.hit_contributions] takes: a
        [`ContributionScores`][universalmotif.ContributionScores], an array
        of shape (sequences, 4, length) or (sequences, length), or a list
        or mapping of per-sequence arrays. Seqlets are called in the
        observed scores, those of the letters present.
    sequences
        The sequences the scores are for, as text or one-hot encoded. They
        pick the letters present from scores of every letter, and give each
        seqlet's letters.
    null_scores, null_sequences
        Null scores, and their sequences when needed, in the same forms as
        ``contributions`` and ``sequences``.
    threshold
        The P-value every stretch inside a seqlet must reach.
    min_len, max_len
        The shortest and longest seqlets, before flanks.
    flank
        Positions added on either side of each seqlet, within its sequence.
    fdr
        The largest share of the null scores' seqlets among the seqlets
        kept, or ``None`` to keep every seqlet.
    sign
        ``"both"``, ``"positive"`` or ``"negative"``.
    n_bins
        The bins the scores are counted in to build the null.
    passes
        How many times to call seqlets. Each pass after the first calls
        them again in the scores with the seqlets so far set to 0. Without
        ``null_scores``, the null comes from the scores themselves, and the
        large scores of strong sites stretch its top, so weaker sites (such
        as a cofactor's next to the factor's own) are often missed by the
        first pass. With ``null_scores`` the null stays the same, so later
        passes add little. Seqlets of later passes come after those of the
        first.
    seed
        Seed for the shuffles behind ``fdr``, so that the seqlets kept near
        the cutoff are the same from run to run.
    nthreads
        Threads to use, with 0 for all available; by default the package
        setting from [`set_nthreads()`][universalmotif.set_nthreads].
    progress
        Show a progress bar on stderr once the call has run for a second;
        by default the package setting from
        [`set_progress()`][universalmotif.set_progress].

    Returns
    -------
    polars.DataFrame
        One row per seqlet, in order of sequence and position. The columns
        are these.

        - ``sequence``, its name. Unless the scores or sequences are named,
          the names are ``seq1``, ``seq2`` and so on.
        - ``sequence_i``.
        - ``start`` and ``end``, flanks included, counted from 0 with the
          end left out.
        - ``core_start`` and ``core_end``, without the flanks.
        - ``sign``, ``+`` or ``-``.
        - ``score``, the total contribution without the flanks.
        - ``pvalue``.
        - ``match``, the seqlet's letters, when they are known.

    References
    ----------
    - Shrikumar A, Tian K, Avsec Ž, Shcherbina A, Banerjee A, Sharmin M, Nair
      S, Kundaje A (2018). Technical note on transcription factor motif
      discovery from importance scores (TF-MoDISco) version 0.5.6.5.
      arXiv:1811.00416. <https://doi.org/10.48550/arXiv.1811.00416>
    - Schreiber J (2026). Tangermeme: a toolkit for understanding
      cis-regulatory logic using deep learning models. *Nature Methods*
      23(11):2136–2140. <https://doi.org/10.1038/s41592-026-03254-z>

    Examples
    --------
    Noisy scores have a block of high contribution at positions 40 to 47 of
    every other sequence. Each block gives a seqlet that takes in a couple
    of the positions around it, and a few more seqlets turn up in the
    noise:

    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> scores = rng.normal(0, 0.01, (200, 100))
    >>> scores[::2, 40:48] += 0.2
    >>> seqlets = um.call_seqlets(scores, flank=0, seed=1)
    >>> seqlets.height, seqlets.filter(pl.col("sequence_i") % 2 == 0).height
    (102, 101)
    >>> seqlets.select(pl.col("core_start", "core_end").median()).row(0)
    (38.0, 50.0)
    """
    if _progress.wanted(progress):
        args = {**locals(), "progress": False}
        return _progress.run(lambda: call_seqlets(**args))
    if min_len < 1 or max_len < min_len:
        raise ValueError("min_len must be at least 1 and at most max_len")
    if max_len > 200:
        raise ValueError("max_len must be at most 200")
    if flank < 0:
        raise ValueError("flank must be zero or positive")
    if passes < 1:
        raise ValueError("passes must be at least 1")
    values, lengths, names, codes = _observed(contributions, sequences)
    null_values, null_lengths = (None, None) if null_scores is None else _observed(null_scores, null_sequences)[:2]
    seed = _seed(seed)
    out = None
    for _ in range(passes):
        found = _core._call_seqlets(values, lengths, null_values, null_lengths, threshold, min_len, max_len, flank,
                                    fdr, sign, n_bins, seed, _nthreads(nthreads))
        out = found if out is None else {k: list(out[k]) + list(found[k]) for k in out}
        if passes > 1:
            values = np.array(values, copy=True)
            for i, a, b in zip(found["sequence_i"], found["start"], found["end"]):
                values[i, a:b] = 0
    seq_i = np.asarray(out["sequence_i"], dtype=np.int64)
    df = pl.DataFrame({
        "sequence": pl.Series([names[i] for i in seq_i], dtype=pl.String),
        "sequence_i": pl.Series(seq_i, dtype=pl.UInt32),
        "start": pl.Series(out["start"], dtype=pl.Int64),
        "end": pl.Series(out["end"], dtype=pl.Int64),
        "core_start": pl.Series(out["core_start"], dtype=pl.Int64),
        "core_end": pl.Series(out["core_end"], dtype=pl.Int64),
        "sign": pl.Series(["+" if p else "-" for p in out["positive"]], dtype=pl.String),
        "score": pl.Series(out["score"], dtype=pl.Float64),
        "pvalue": pl.Series(out["pvalue"], dtype=pl.Float64),
    })
    if codes is not None:
        letters = np.frombuffer((_LETTERS + "N").encode(), dtype=np.uint8)
        match = [letters[codes[i, s:e]].tobytes().decode() for i, s, e in
                 zip(seq_i, df["start"].to_numpy(), df["end"].to_numpy())]
        df = df.with_columns(pl.Series("match", match, dtype=pl.String))
    return df


def seqlet_fdr(
    seqlets: pl.DataFrame,
    null_seqlets: pl.DataFrame,
    fdr: float = 0.2,
    positions: int | None = None,
    null_positions: int | None = None,
) -> pl.DataFrame:
    """Keep seqlets down to a false discovery rate set by the seqlets of
    null scores.

    Seqlets called in batches by
    [`call_seqlets()`][universalmotif.call_seqlets] with ``fdr=None`` and
    the same ``null_scores``, and the seqlets of those null scores called
    the same way, get the cut ``call_seqlets()`` gives them in one call.
    For each sign, seqlets are kept down to the lowest total at which null
    seqlets of at least that total make up at most ``fdr`` of the seqlets.
    The null scores are counted as if they had as many positions as the
    scores. ``positions`` and ``null_positions`` give those numbers of
    positions, and by default they are taken to be equal.

    Returns the seqlets kept, in their order.

    Examples
    --------
    Scores with a block of high contribution in every other sequence,
    called in two batches. The null scores are the first batch's scores
    shuffled, which have the same values and so the same null:

    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> scores = rng.normal(0, 0.01, (200, 100))
    >>> scores[::2, 40:48] += 0.2
    >>> null = rng.permuted(scores[:100], axis=1)
    >>> found = []
    >>> for lo in (0, 100):
    ...     batch = um.call_seqlets(scores[lo:lo + 100], null_scores=null, fdr=None, flank=0)
    ...     found.append(batch.with_columns(pl.col("sequence_i") + lo))
    >>> null_seqlets = um.call_seqlets(null, null_scores=null, fdr=None, flank=0)
    >>> kept = um.seqlet_fdr(pl.concat(found), null_seqlets, positions=scores.size, null_positions=null.size)

    These are the seqlets of a single call with the same null scores:

    >>> whole = um.call_seqlets(scores, null_scores=null, flank=0)
    >>> kept.height, kept.select("sequence_i", "start", "end").equals(whole.select("sequence_i", "start", "end"))
    (101, True)
    """
    if not 0 < fdr <= 1:
        raise ValueError("fdr must be above 0 and at most 1")
    for name, df in (("seqlets", seqlets), ("null_seqlets", null_seqlets)):
        if not isinstance(df, pl.DataFrame) or "score" not in df.columns:
            raise ValueError(f"{name} must be a DataFrame from call_seqlets(), with a 'score' column")
    if (positions is None) != (null_positions is None):
        raise ValueError("give both positions and null_positions, or neither")
    scale = 1.0 if positions is None else max(positions, 1) / max(null_positions, 1)
    sign = "sign" in seqlets.columns
    keep = np.zeros(seqlets.height, dtype=bool)
    for sg in (seqlets["sign"].unique().to_list() if sign else [None]):
        mine = (seqlets["sign"] == sg).to_numpy() if sign else np.ones(seqlets.height, dtype=bool)
        theirs = (null_seqlets.filter(pl.col("sign") == sg) if sg is not None and "sign" in null_seqlets.columns
                  else null_seqlets)
        real = np.abs(seqlets["score"].to_numpy()[mine])
        fake = np.sort(np.abs(theirs["score"].to_numpy()))
        # The lowest total at which the scaled null seqlets at or above it are at most fdr of the seqlets.
        v = np.sort(real)[::-1]
        n_null = len(fake) - np.searchsorted(fake, v, side="left")
        ok = np.flatnonzero(n_null * scale / np.arange(1, len(v) + 1) <= fdr)
        cut = v[ok[-1]] if ok.size else np.inf
        keep[np.flatnonzero(mine)[real >= cut]] = True
    return seqlets.filter(pl.Series(keep))


__all__ = ["call_seqlets", "seqlet_fdr"]
