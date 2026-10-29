"""Effects of many sequence variants drawn along a region, and their effects
on the motif sites they fall in."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._analysis import _make_unique
from ._api import _check_motif
from ._core import Motif
from ._plot import _ALPHABET_NAMES, _default_colours, _nice_step, _pyplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_UP = "#C0504D"
_DOWN = "#4F81BD"
_FLAT = "#9E9E9E"
_RING = "#1A1A1A"
_MUTED = "#5E5E5E"
_SUMMARIES = ("mean", "min", "max", "strongest")
_COMPLEMENTS = {"DNA": ("ACGT", "TGCA"), "RNA": ("ACGU", "UGCA")}
# Heights in inches of the panels of a figure made here.
_EFFECT_INCHES = 2.4
_LETTER_INCHES = 0.24
_LANE_INCHES = 0.17


def _column(df: pl.DataFrame, name: str) -> str | None:
    """The column called ``name``, matched without regard to case."""
    if name in df.columns:
        return name
    return next((c for c in df.columns if c.lower() == name), None)


def _as_frame(variants: Any) -> pl.DataFrame | None:
    """A table of variants as a polars DataFrame, or None for an array."""
    if isinstance(variants, pl.LazyFrame):
        return variants.collect()
    if isinstance(variants, pl.DataFrame):
        return variants
    if type(variants).__module__.startswith("pandas") and hasattr(variants, "columns"):
        return pl.from_pandas(variants)
    if isinstance(variants, Mapping):
        return pl.DataFrame(dict(variants))
    rows = isinstance(variants, (list, tuple)) and variants and isinstance(variants[0], (list, tuple))
    if rows and any(isinstance(v, str) for v in variants[0]):
        if len(variants[0]) not in (3, 4):
            raise ValueError("variants given as rows must be (position, ref, alt) or "
                             "(position, ref, alt, effect)")
        names = ["position", "ref", "alt", "effect"][:len(variants[0])]
        return pl.DataFrame([list(v) for v in variants], schema=names, orient="row")
    return None


def _from_array(variants: Any, sequence: str | None, alphabet: str, start: int) -> pl.DataFrame:
    """Every substitution in an array of effects of shape (letters, length),
    such as from mutagenesis_effects()."""
    from ._deep import _to_numpy

    m = _to_numpy(variants)
    n = len(alphabet)
    if m.ndim == 2 and m.shape[0] != n and m.shape[1] == n:
        m = m.T
    if m.ndim == 3:
        raise ValueError("the effects have several outputs; pick one, such as effects[0]")
    if m.ndim != 2 or m.shape[0] != n:
        raise ValueError(f"an array of effects must have shape ({n}, length), one row per letter of "
                         f"{alphabet}")
    if sequence is None:
        raise ValueError("an array of effects needs the sequence, to know the reference letters")
    if len(sequence) != m.shape[1]:
        raise ValueError(f"the sequence has {len(sequence)} letters for {m.shape[1]} positions of effects; "
                         "for effects of part of a sequence, give that part and its start")
    if not np.isfinite(m).all():
        raise ValueError("the effects have missing or infinite values")
    letters = np.array(list(alphabet))
    seq = np.array(list(sequence.upper() if alphabet.isupper() else sequence))
    ref = seq[None, :] == letters[:, None]
    let, pos = np.nonzero(~ref)
    order = np.lexsort((let, pos))
    let, pos = let[order], pos[order]
    return pl.DataFrame({
        "position": pl.Series(pos + start, dtype=pl.Int64),
        "ref": np.array(list(sequence))[pos].tolist(),
        "alt": letters[let].tolist(),
        "effect": m[let, pos].astype(float),
    })


def _variants(variants: Any, effects: Any, sequence: str | None, alphabet: str, start: int,
              need_effects: bool) -> pl.DataFrame:
    """Variants as a table with position, ref and alt columns, and an effect
    column when ``need_effects``, with any other columns kept after them."""
    df = _as_frame(variants)
    if df is None:
        if effects is not None:
            raise ValueError("effects cannot be given with an array of effects")
        return _from_array(variants, sequence, alphabet, start)
    rename = {}
    for name in ("position", "ref", "alt"):
        col = _column(df, name)
        if col is None:
            raise ValueError(f"variants must have a {name!r} column")
        rename[col] = name
    if effects is None or isinstance(effects, str):
        col = _column(df, effects or "effect")
        if col is not None:
            rename[col] = "effect"
        elif effects is not None:
            raise ValueError(f"variants has no {effects!r} column")
        elif need_effects:
            raise ValueError("variants needs an 'effect' column, or give the effects with effects")
    df = df.rename({k: v for k, v in rename.items() if k != v})
    if effects is not None and not isinstance(effects, str):
        from ._deep import _to_numpy

        e = _to_numpy(effects).ravel()
        if len(e) != df.height:
            raise ValueError(f"{len(e)} effects for {df.height} variants")
        df = df.with_columns(pl.Series("effect", e.astype(float)))
    if df.height == 0:
        raise ValueError("there are no variants")
    if not df["position"].dtype.is_integer():
        raise ValueError("positions must be whole numbers, counted from 0")
    df = df.with_columns(pl.col("position").cast(pl.Int64), pl.col("ref").cast(pl.String),
                         pl.col("alt").cast(pl.String))
    if df["position"].null_count() or df["ref"].null_count():
        raise ValueError("positions and reference letters cannot be missing")
    if "effect" in df.columns:
        if not df["effect"].dtype.is_numeric():
            raise ValueError("effects must be numbers")
        df = df.with_columns(pl.col("effect").cast(pl.Float64))
        if df["effect"].null_count() or not np.isfinite(df["effect"].to_numpy()).all():
            raise ValueError("the effects have missing or infinite values")
    lead = ["position", "ref", "alt"] + (["effect"] if "effect" in df.columns else [])
    df = df.select(lead + [c for c in df.columns if c not in lead])
    if sequence is not None:
        _check_refs(df, sequence, start)
    return df


def _check_refs(df: pl.DataFrame, sequence: str, start: int) -> None:
    i = df["position"].to_numpy() - start
    refs = df["ref"].to_list()
    width = np.array([len(r) for r in refs])
    out = np.flatnonzero((i < 0) | (i + np.maximum(width, 1) > len(sequence)))
    if out.size:
        p = int(df["position"][int(out[0])])
        raise ValueError(f"variant at position {p} lies outside the sequence, which runs from {start} to "
                         f"{start + len(sequence)}")
    bad = [k for k in range(df.height) if sequence[i[k]:i[k] + width[k]].upper() != refs[k].upper()]
    if bad:
        shown = "; ".join(f"position {int(i[k]) + start} has {sequence[i[k]:i[k] + width[k]]}, not {refs[k]}"
                          for k in bad[:3])
        more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
        raise ValueError(f"the reference letters of {len(bad)} variants do not match the sequence: "
                         f"{shown}{more}")


def _summarize(df: pl.DataFrame, summary: str) -> pl.DataFrame:
    """One variant per position: the mean effect, or the variant with the
    lowest, highest or largest absolute effect."""
    if summary not in _SUMMARIES:
        raise ValueError(f"summary must be one of {', '.join(map(repr, _SUMMARIES))}, or None")
    if summary == "mean":
        return (df.group_by("position", "ref", maintain_order=True)
                .agg(pl.col("effect").mean())
                .with_columns(pl.lit(None, dtype=pl.String).alias("alt"))
                .select("position", "ref", "alt", "effect").sort("position", maintain_order=True))
    key = {"min": pl.col("effect"), "max": -pl.col("effect"), "strongest": -pl.col("effect").abs()}[summary]
    return (df.with_columns(key.alias("_key")).sort("position", "_key", maintain_order=True)
            .unique("position", keep="first", maintain_order=True).drop("_key"))


def _motif_list(motifs: Motif | Iterable[Motif]) -> tuple[list[Motif], list[str]]:
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("motifs is empty")
    return ms, _make_unique([m.name for m in ms])


def _windows(motif: Motif, codes: np.ndarray, idx: np.ndarray, alt: np.ndarray):
    """The windows of the motif's width around each variant, as the variant
    of each, its start, and its letters with the reference and the
    alternative letter, keeping those made only of the motif's letters."""
    w, n = motif.width, len(codes)
    s = idx[:, None] - np.arange(w)[None, :]
    rows, offs = np.nonzero((s >= 0) & (s <= n - w))
    s = s[rows, offs]
    ref = codes[s[:, None] + np.arange(w)]
    alts = ref.copy()
    alts[np.arange(len(s)), offs] = alt[rows]
    ok = np.zeros(256, dtype=bool)
    ok[np.frombuffer("".join(motif.letters).encode(), dtype=np.uint8)] = True
    keep = ok[ref].all(axis=1) & ok[alts].all(axis=1)
    return rows[keep], s[keep], ref[keep], alts[keep]


def _words(x: np.ndarray) -> list[str]:
    return np.ascontiguousarray(x).view(f"S{x.shape[1]}").ravel().astype(str).tolist() if len(x) else []


def variant_motif_effects(
    variants: Any,
    sequence: str,
    motifs: Motif | Iterable[Motif],
    pvalue: float = 1e-4,
    start: int = 0,
    hits: pl.DataFrame | None = None,
    rc: bool = True,
) -> pl.DataFrame:
    """How single-letter variants change the motif sites they fall in.

    Every window of each motif's width that holds a variant is scored before
    and after the change, as [`scan_sequences()`][universalmotif.scan_sequences]
    scores it, on both strands of DNA and RNA with ``rc``. A window is kept
    when it reaches the ``pvalue`` cutoff in either sequence, so that the
    table holds sites a variant weakens or destroys and sites it strengthens
    or creates. With ``hits``, such as from ``scan_sequences()`` run on
    ``sequence``, only those sites are scored instead, whatever their
    scores.

    ``variants`` is a table with ``position``, ``ref`` and ``alt`` columns,
    as a polars or pandas DataFrame or a list of (position, ref, alt) rows,
    an array of effects from
    [`mutagenesis_effects()`][universalmotif.mutagenesis_effects], or
    ``None`` for every single-letter substitution of ``sequence``.
    Positions are counted from ``start``, such as the genomic position of
    the first letter, while the positions of ``hits`` are counted along the
    sequence from 0, as ``scan_sequences()`` gives them. Variants that
    change more or fewer than one letter are left out.

    Returns
    -------
    polars.DataFrame
        One row per variant and motif site, with the columns ``variant``,
        the row of the variant in ``variants``, ``position``, ``ref``,
        ``alt``, ``motif``, ``start``, ``end``, ``strand``, ``ref_score``,
        ``alt_score``, ``delta``, the change in score, ``ref_pvalue`` and
        ``alt_pvalue``. The site's ``start`` and ``end`` are counted from
        ``start`` like the positions, with the end left out.

    Examples
    --------
    >>> import polars as pl
    >>> import universalmotif as um
    >>> gata = um.example_motifs()[6]
    >>> seq = "TTACGCTTAGATAAGGCTTACGCATCGT"
    >>> variants = [(10, "A", "C"), (11, "T", "C"), (20, "C", "T")]
    >>> effects = um.variant_motif_effects(variants, seq, gata, pvalue=1e-3)
    >>> effects.select("position", "alt", "strand", "delta").rows()
    [(10, 'C', '+', -4.296), (11, 'C', '+', -7.561)]
    >>> every = um.variant_motif_effects(None, seq, gata, pvalue=1e-3)
    >>> every.height, every.filter(pl.col("delta") > 0).height
    (24, 3)
    """
    if not isinstance(sequence, str):
        raise TypeError("give the sequence as a string")
    if not 0 < pvalue <= 1:
        raise ValueError("pvalue must be above 0 and at most 1")
    ms, names = _motif_list(motifs)
    alphabet = "".join(ms[0].letters)
    if variants is None:
        variants = _every_substitution(sequence, alphabet, start)
    df = _variants(variants, None, sequence, alphabet, start, need_effects=False)
    df = df.with_row_index("variant")
    snv = df.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
    allowed = None
    if hits is not None:
        for col in ("motif", "start", "end"):
            if col not in hits.columns:
                raise ValueError(f"hits must have a {col!r} column")
        allowed = hits.select(pl.col("motif").cast(pl.String), pl.col("start").cast(pl.Int64),
                              (pl.col("strand").cast(pl.String) if "strand" in hits.columns
                               else pl.lit("+")).alias("strand")).unique()
    from ._analysis import motif_pvalue
    from ._utils import score_match

    frames = []
    variant_ids = snv["variant"].to_numpy()
    idx = snv["position"].to_numpy() - start
    for name, motif in zip(names, ms):
        upper = motif.alphabet in ("DNA", "RNA", "AA")
        seq = sequence.upper() if upper else sequence
        alt = snv["alt"].str.to_uppercase() if upper else snv["alt"]
        try:
            codes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            alt_codes = np.frombuffer("".join(alt.to_list()).encode("ascii"), dtype=np.uint8)
        except UnicodeEncodeError as e:
            raise ValueError("the sequence and variants must be plain ASCII letters") from e
        rows, s, ref_w, alt_w = _windows(motif, codes, idx, alt_codes)
        strands = [("+", ref_w, alt_w)]
        if rc and motif.alphabet in _COMPLEMENTS:
            table = np.arange(256, dtype=np.uint8)
            a, b = _COMPLEMENTS[motif.alphabet]
            table[np.frombuffer(a.encode(), dtype=np.uint8)] = np.frombuffer(b.encode(), dtype=np.uint8)
            strands.append(("-", table[ref_w[:, ::-1]], table[alt_w[:, ::-1]]))
        threshold = float(motif_pvalue(motif, pvalue=pvalue))
        for strand, r, a in strands:
            ref_score = np.asarray(score_match(motif, _words(r)), dtype=float)
            alt_score = np.asarray(score_match(motif, _words(a)), dtype=float)
            part = pl.DataFrame({
                "variant": pl.Series(variant_ids[rows], dtype=pl.UInt32),
                "motif": pl.Series([name] * len(rows), dtype=pl.String),
                "start": pl.Series(s, dtype=pl.Int64),
                "strand": pl.Series([strand] * len(rows), dtype=pl.String),
                "ref_score": ref_score,
                "alt_score": alt_score,
            })
            if allowed is not None:
                part = part.join(allowed, on=["motif", "start", "strand"], how="semi")
            else:
                part = part.filter((pl.col("ref_score") >= threshold) | (pl.col("alt_score") >= threshold))
            if part.height:
                pv = np.asarray(motif_pvalue(motif, score=np.concatenate(
                    [part["ref_score"].to_numpy(), part["alt_score"].to_numpy()]).tolist()), dtype=float)
                part = part.with_columns(pl.Series("ref_pvalue", pv[:part.height]),
                                         pl.Series("alt_pvalue", pv[part.height:]),
                                         pl.lit(names.index(name)).alias("_m"))
                frames.append(part.with_columns((pl.col("start") + motif.width).alias("end")))
    schema = {"variant": pl.UInt32, "position": pl.Int64, "ref": pl.String, "alt": pl.String,
              "motif": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String,
              "ref_score": pl.Float64, "alt_score": pl.Float64, "delta": pl.Float64,
              "ref_pvalue": pl.Float64, "alt_pvalue": pl.Float64}
    if not frames:
        return pl.DataFrame(schema=schema)
    out = (pl.concat(frames)
           .join(snv.select("variant", "position", "ref", "alt"), on="variant", how="left")
           .with_columns((pl.col("start") + start), (pl.col("end") + start),
                         (pl.col("alt_score") - pl.col("ref_score")).round(3).alias("delta"))
           .sort("variant", "_m", "start", "strand"))
    return out.select(list(schema)).cast(schema)


def _every_substitution(sequence: str, alphabet: str, start: int) -> pl.DataFrame:
    seq = sequence.upper() if alphabet.isupper() else sequence
    rows = [(start + i, sequence[i], b) for i, a in enumerate(seq) for b in alphabet if b != a]
    if not rows:
        raise ValueError("the sequence is empty")
    return pl.DataFrame(rows, schema=["position", "ref", "alt"], orient="row")


def _sizes(df: pl.DataFrame, col: str) -> tuple[np.ndarray, str]:
    if col not in df.columns:
        raise ValueError(f"variants has no {col!r} column to size by")
    if not df[col].dtype.is_numeric():
        raise ValueError("size_by must name a numeric column")
    v = df[col].cast(pl.Float64).to_numpy().copy()
    title = col
    if col.lower().startswith(("pval", "qval", "padj", "fdr")):
        v = -np.log10(np.clip(v, 1e-300, None))
        title = f"-log10({col})"
    return v, title


def _hits_frame(hits: pl.DataFrame) -> pl.DataFrame:
    for col in ("motif", "start", "end"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if "sequence" in hits.columns and hits["sequence"].n_unique() > 1:
        raise ValueError("hits are from several sequences; give only those of this one")
    return hits.with_columns(pl.col("motif").cast(pl.String), pl.col("start").cast(pl.Int64),
                             pl.col("end").cast(pl.Int64))


def _stack(ax: Axes, legends: list, make) -> Any:
    """Add a legend below those already beside the axes."""
    if legends:
        renderer = ax.figure.canvas.get_renderer()
        below = legends[-1].get_window_extent(renderer)
        y = ax.transAxes.inverted().transform((0, below.y0))[1] - 0.04
    else:
        y = 1.0
    leg = make(y)
    ax.add_artist(leg)
    # Added artists are clipped to the axes, which would leave the legend
    # out of tight layouts and bbox_inches="tight".
    leg.set_clip_on(False)
    legends.append(leg)
    return leg


def _fit_figure(fig: Figure, axes: list, legends: list, own_figure: bool) -> None:
    """Make room for legends running off the figure: a figure made here is
    widened and made taller with its panels kept the same size, and axes
    passed in are narrowed."""
    renderer = fig.canvas.get_renderer()
    boxes = [leg.get_window_extent(renderer) for leg in legends]
    right = max(b.x1 for b in boxes) - fig.bbox.x1
    below = max(0.0, -min(b.y0 for b in boxes))
    if right <= 0 and below <= 0:
        return
    if not own_figure:
        shrink = (max(right, 0) + 0.1 * fig.dpi) / fig.bbox.width
        for a in axes:
            pos = a.get_position()
            a.set_position([pos.x0, pos.y0, max(pos.width - shrink, 0.1), pos.height])
        return
    w, h = fig.get_size_inches()
    new_w = w + max(right, 0) / fig.dpi + (0.1 if right > 0 else 0)
    extra = below / fig.dpi + (0.1 if below > 0 else 0)
    new_h = h + extra
    for a in axes:
        pos = a.get_position()
        a.set_position([pos.x0 * w / new_w, (pos.y0 * h + extra) / new_h, pos.width * w / new_w,
                        pos.height * h / new_h])
    fig.set_size_inches(new_w, new_h)


def plot_variant_effects(
    variants: Any,
    effects: Any = None,
    sequence: str | None = None,
    motifs: Motif | Iterable[Motif] | None = None,
    hits: pl.DataFrame | None = None,
    start: int = 0,
    region: tuple[int, int] | None = None,
    alphabet: str = "ACGT",
    summary: str | None = None,
    colour_by: str = "sign",
    colour_scheme: Mapping[str, str] | None = None,
    size_by: str | None = None,
    labels: int = 5,
    label_by: str | None = None,
    highlight: bool = True,
    show_sequence: bool | None = None,
    pvalue: float = 1e-4,
    ylabel: str = "Effect",
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw the effects of many variants along a sequence.

    Each variant is a lollipop at its position, rising or falling from zero
    to its effect, such as a model's predicted change in signal for the
    variants of a VCF file or a QTL study, or for every substitution from
    [`mutagenesis_effects()`][universalmotif.mutagenesis_effects]. Below
    them, motif hits along the sequence are drawn as arrows pointing along
    their strand, as [`plot_hits()`][universalmotif.plot_hits] draws them,
    and the letters of the sequence when there is room for them.

    With ``motifs``, the hits are found by scanning ``sequence`` at the
    ``pvalue`` cutoff, unless ``hits`` are given, and each single-letter
    variant's change to the motif sites it falls in is found with
    [`variant_motif_effects()`][universalmotif.variant_motif_effects], which
    gives the full table. The head of a variant that strengthens or
    creates a site then points up, and that of one that weakens or destroys
    a site points down, after the site it changes most. With ``highlight``,
    variants inside a drawn hit are ringed. The variants with the largest
    effects are labelled with their position and change, and with the site
    they change most.

    Needs the ``plot`` extra.

    Parameters
    ----------
    variants
        The variants, as a polars or pandas DataFrame with ``position``,
        ``ref`` and ``alt`` columns and an ``effect`` column, or as a list
        of (position, ref, alt, effect) rows. Column names are matched
        without regard to case, so ``REF`` and ``ALT`` work too, and other
        columns can be used for ``size_by`` and ``label_by``. Positions
        are counted from 0 at the first letter of the sequence, or from
        ``start``. An array of shape (letters, length) from
        ``mutagenesis_effects()`` draws every substitution, with the
        letters in the order of ``alphabet``.
    effects
        The name of the column of effects, ``"effect"`` by default, or the
        effects themselves as an array with one per variant.
    sequence
        The reference sequence, needed for arrays of effects, for
        ``motifs`` and to draw the letters. The reference letters of the
        variants must match it.
    motifs
        Motifs to find in the sequence and to score the variants with.
        They set the order of the motifs in the legend and add their
        logos to it.
    hits
        Motif hits to draw, such as from
        [`scan_sequences()`][universalmotif.scan_sequences] run on
        ``sequence``, with positions counted along the sequence from 0.
    start
        The position of the first letter of the sequence, such as its
        genomic position, which the axis and the positions of variants
        count from.
    region
        Draw only the positions from the first of this (start, end) pair up
        to the second, counted as the axis is.
    alphabet
        The letters of the sequence, in the order of the rows of an array of
        effects.
    summary
        Draw one variant per position: ``"mean"`` for the mean effect of
        those at the position, or ``"min"``, ``"max"`` or ``"strongest"``
        for the one with the lowest, highest or largest absolute effect.
    colour_by
        ``"sign"`` to colour effects above zero red and those below blue,
        or ``"alt"`` to colour them by their alternative letter.
    colour_scheme
        Colours of the letters for ``colour_by="alt"`` and for the letters
        of the sequence, by default the package's colours for DNA, RNA and
        amino acids.
    size_by
        A numeric column to size the heads by, from its smallest value to
        its largest, such as an allele frequency or the significance of a
        QTL. P-values and q-values are taken as -log10.
    labels
        How many of the variants with the largest absolute effects to
        label. Labels are set in a row below the lowest effect, or above
        the highest, and joined to their variants by lines, so that those
        of variants bunched together stay apart; when they do not fit there,
        they are placed beside their variants instead.
    label_by
        A column to label variants by, such as variant IDs, instead of
        their position and change.
    highlight
        Ring the variants inside a drawn motif hit.
    show_sequence
        Draw the letters of the sequence, or by default only when they fit.
    pvalue
        The P-value cutoff for motif sites.
    ylabel
        The label of the y-axis.
    text_size
        The size of the labels and legends.
    figsize
        The size of a new figure, in inches.
    ax
        Axes to draw into, which are split to make room for the hits and
        letters below the effects. The axes of the effects are returned,
        and otherwise a new figure is.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    A stand-in for a model scores a sequence by its best GATA site, so that
    only the variants in that site have an effect:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> gata = um.example_motifs()[6]
    >>> seq = "TTACGCTTAGATAAGGCTTACGCATCGTAAGCT"
    >>> def model(batch):
    ...     words = [s[i:i + 8] for s in um.from_one_hot(batch) for i in range(len(s) - 7)]
    ...     return um.score_match(gata, words).reshape(len(batch), -1).max(axis=1)
    >>> ism = um.mutagenesis_effects(model(um.mutagenesis(seq)), seq)
    >>> fig = um.plot_variant_effects(ism, sequence=seq, motifs=gata, pvalue=1e-3)
    >>> [ax.get_ylabel() for ax in fig.axes]
    ['Effect', '', 'Hits']
    >>> variants = pl.DataFrame({"position": [9, 11, 20], "ref": ["G", "T", "C"],
    ...                          "alt": ["A", "C", "T"], "effect": [-0.4, -1.2, 0.2],
    ...                          "id": ["rs1", "rs2", "rs3"]})
    >>> fig = um.plot_variant_effects(variants, sequence=seq, motifs=gata, label_by="id",
    ...                               pvalue=1e-3)
    >>> plt.close("all")
    """
    if colour_by not in ("sign", "alt"):
        raise ValueError('colour_by must be "sign" or "alt"')
    if labels is None:
        labels = 0
    if isinstance(labels, bool) or not isinstance(labels, (int, np.integer)) or labels < 0:
        raise ValueError("labels must be a number of variants to label, 0 for none")
    if sequence is not None and not isinstance(sequence, str):
        raise TypeError("give the sequence as a string")
    if motifs is not None and sequence is None:
        raise ValueError("motifs need the sequence, to find their sites and score the variants")
    df = _variants(variants, effects, sequence, alphabet, start, need_effects=True)
    if summary is not None:
        df = _summarize(df, summary)
    if colour_by == "alt" and df["alt"].null_count():
        raise ValueError('colour_by="alt" needs an alternative letter for each variant, which summary="mean" '
                         "leaves out")
    if label_by is not None and label_by not in df.columns:
        raise ValueError(f"variants has no {label_by!r} column to label by")

    ms: list[Motif] = []
    motif_names: list[str] = []
    if motifs is not None:
        ms, motif_names = _motif_list(motifs)
    if hits is None and ms:
        from ._analysis import scan_sequences

        hits = scan_sequences(ms, sequence, pvalue=pvalue)
    if hits is not None:
        hits = _hits_frame(hits)

    # The positions drawn.
    if sequence is not None:
        lo_all, hi_all = start, start + len(sequence)
    else:
        lo_all = int(df["position"].min())
        hi_all = int((df["position"] + df["ref"].str.len_chars().clip(1)).max())
        if hits is not None and hits.height:
            lo_all = min(lo_all, int(hits["start"].min()) + start)
            hi_all = max(hi_all, int(hits["end"].max()) + start)
        # Without a sequence to end it, the axis runs a little past the
        # variants at either end.
        edge = max(1, round(0.02 * (hi_all - lo_all)))
        lo_all, hi_all = lo_all - edge, hi_all + edge
    if region is None:
        a, b = lo_all, hi_all
    else:
        a, b = int(region[0]), int(region[1])
        if not lo_all <= a < b <= hi_all:
            raise ValueError(f"region must lie within {lo_all} to {hi_all}")
    df = df.filter((pl.col("position") >= a) & (pl.col("position") < b))
    if df.height == 0:
        raise ValueError("there are no variants in the region drawn")
    if hits is not None:
        hits = hits.filter((pl.col("end") + start > a) & (pl.col("start") + start < b))

    # Motif changes, and whether each variant lies in a drawn hit.
    n = df.height
    direction = np.zeros(n)
    in_hit = np.zeros(n, dtype=bool)
    best: list[tuple[str, float] | None] = [None] * n
    # Variants are numbered by row of the table scored, which leaves out
    # those without an alternative letter.
    rows = np.flatnonzero(df["alt"].is_not_null().to_numpy())
    if ms and rows.size:
        changes = variant_motif_effects(df[rows].drop("effect"), sequence, ms, pvalue=pvalue, start=start)
        if changes.height:
            top = (changes.with_columns(pl.col("delta").abs().alias("_size"))
                   .sort("variant", "_size", descending=[False, True], maintain_order=True)
                   .unique("variant", keep="first", maintain_order=True))
            for v, motif, delta in top.select("variant", "motif", "delta").iter_rows():
                i = rows[v]
                direction[i] = np.sign(delta)
                best[i] = (motif, delta)
    if hits is not None and hits.height:
        pos = df["position"].to_numpy() - start
        end = pos + np.maximum(df["ref"].str.len_chars().to_numpy(), 1)
        # The hits are merged into disjoint spans, so that each variant need
        # only be checked against the last span starting before its end.
        spans = hits.select("start", "end").sort("start")
        hs, he = spans["start"].to_numpy(), np.maximum.accumulate(spans["end"].to_numpy())
        new = np.r_[True, hs[1:] >= he[:-1]]
        hs, he = hs[new], np.r_[he[np.flatnonzero(new)[1:] - 1], he[-1]]
        k = np.searchsorted(hs, end, side="left") - 1
        in_hit = (k >= 0) & (he[np.maximum(k, 0)] > pos)

    # Variants at the same position are spread a little apart.
    x = df["position"].to_numpy().astype(float) + 0.5
    group = df.select(pl.col("position").cum_count().over("position").alias("k"),
                      pl.col("position").count().over("position").alias("n"))
    k, m = group["k"].to_numpy() - 1, group["n"].to_numpy()
    x = x + (k - (m - 1) / 2) * np.minimum(0.25, 0.8 / m)
    y = df["effect"].to_numpy()

    letter_colours = colour_scheme or _default_colours(_ALPHABET_NAMES.get(alphabet, ""), list(alphabet))
    if colour_by == "sign":
        colours = np.where(y > 0, _UP, np.where(y < 0, _DOWN, _FLAT))
    else:
        alts = df["alt"].to_list()
        colours = np.array([letter_colours.get(c, letter_colours.get(c.upper(), _FLAT)) for c in alts])
        colours[y == 0] = _FLAT

    # Panels and their heights.
    span = b - a
    plt = _pyplot()
    own_figure = ax is None
    fig_w = figsize[0] if figsize else 8.0
    if own_figure:
        width_pts = (fig_w - 1.6) * 72
    else:
        width_pts = ax.get_position().width * ax.figure.get_figwidth() * 72
    per_pos = width_pts / span
    if show_sequence is None:
        show_sequence = sequence is not None and per_pos >= 0.65 * text_size
    elif show_sequence and sequence is None:
        raise ValueError("show_sequence needs the sequence")
    nlanes = 0
    if hits is not None:
        from ._plot_hits import _lanes

        if hits.height:
            hs = np.maximum(hits["start"].to_numpy() + start, a).astype(float)
            he = np.minimum(hits["end"].to_numpy() + start, b).astype(float)
            nlanes = _lanes(hs, he, 2.0 * span / width_pts)[1]
        nlanes = max(nlanes, 1)
    heights = [_EFFECT_INCHES]
    if show_sequence:
        heights.append(_LETTER_INCHES)
    if hits is not None:
        heights.append((nlanes + 0.6) * _LANE_INCHES + 0.1)
    if own_figure:
        fig_h = figsize[1] if figsize else sum(heights) + 0.75
        fig, axes = plt.subplots(len(heights), 1, figsize=(fig_w, fig_h), sharex=True, squeeze=False,
                                 gridspec_kw={"height_ratios": heights})
        axes = list(axes[:, 0])
        ax = axes[0]
    else:
        fig = ax.figure
        pos = ax.get_position()
        gap = 0.02 * pos.height
        unit = (pos.height - gap * (len(heights) - 1)) / sum(heights)
        top = pos.y1
        axes = []
        for h in heights:
            box = [pos.x0, top - h * unit, pos.width, h * unit]
            if not axes:
                ax.set_position(box)
                axes.append(ax)
            else:
                axes.append(fig.add_axes(box, sharex=ax))
            top -= h * unit + gap
    eff_ax = axes[0]
    seq_ax = axes[1] if show_sequence else None
    hit_ax = axes[-1] if hits is not None else None

    # Hits, drawn by plot_hits() on a single range of the sequence.
    palette: dict[str, str] = {}
    logos: dict[str, Motif] = {}
    if hit_ax is not None:
        from ._plot_hits import _motif_colours, plot_hits

        if ms:
            names = motif_names
            logos = dict(zip(motif_names, ms))
            places: Iterable[int] = range(len(names))
        elif "motif_i" in hits.columns:
            pairs = (hits.select("motif", pl.col("motif_i").cast(pl.Int64))
                     .unique("motif", keep="first", maintain_order=True).sort("motif_i", maintain_order=True))
            names, places = pairs["motif"].to_list(), pairs["motif_i"].to_list()
        else:
            names = hits["motif"].unique(maintain_order=True).to_list()
            places = range(len(names))
        palette = _motif_colours(names, places)
        shown = hits.with_columns(pl.lit("_").alias("chrom"), pl.col("start") + start, pl.col("end") + start)
        # Only the motifs with a hit in view are drawn and listed.
        in_view = set(shown.filter((pl.col("end") > a) & (pl.col("start") < b))["motif"].to_list())
        motif_names = [m for m in names if m in in_view]
        plot_hits(shown, ranges=[("_", a, b)], colours=palette, legend=False, text_size=text_size,
                  motifs=motif_names or None, ax=hit_ax)
        hit_ax.set_yticks([])
        hit_ax.set_ylabel("Hits", rotation=0, ha="right", va="center", fontsize=text_size)
        hit_ax.spines["bottom"].set_visible(True)

    # Lollipops.
    from matplotlib.collections import LineCollection

    size = None
    if size_by is not None:
        from ._plot_hits import _scaled

        v, size_title = _sizes(df, size_by)
        frac, s_lo, s_hi = _scaled(v)
        area = 10.0 + 70.0 * frac
        size = (size_title, s_lo, s_hi)
    else:
        area = np.full(n, float(np.clip(0.8 * per_pos**2, 9.0, 36.0)))
        # Variants without an effect are kept small, so that they read as a
        # baseline rather than as data points of their own.
        area = np.where(y == 0, 0.5 * area, area)
    order = np.argsort(np.abs(y), kind="stable")
    eff_ax.add_collection(LineCollection(
        [[(xi, 0.0), (xi, yi)] for xi, yi in zip(x[order], y[order])], colors=list(colours[order]),
        linewidths=0.9 if per_pos >= 2 else 0.6, zorder=2))
    markers = {1.0: "^", -1.0: "v", 0.0: "o"}
    for d, marker in markers.items():
        sel = order[direction[order] == d]
        if not sel.size:
            continue
        ring = in_hit[sel] & highlight
        eff_ax.scatter(x[sel], y[sel], s=area[sel] * (1.3 if d else 1.0), c=list(colours[sel]),
                       marker=marker, edgecolors=np.where(ring, _RING, "white").tolist(),
                       linewidths=np.where(ring, 1.1, 0.5), zorder=3)
    eff_ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
    y_lo, y_hi = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    pad = 0.12 * ((y_hi - y_lo) or 1.0)
    eff_ax.set_ylim(y_lo - (pad if y_lo < 0 else 0.02 * pad), y_hi + (pad if y_hi > 0 else 0.02 * pad))
    eff_ax.set_ylabel(ylabel, fontsize=text_size)
    eff_ax.tick_params(labelsize=text_size)
    for side in ("top", "right", "bottom"):
        eff_ax.spines[side].set_visible(False)

    if seq_ax is not None:
        fs = min(text_size, 0.85 * per_pos)
        for p in range(a, b):
            c = sequence[p - start]
            seq_ax.text(p + 0.5, 0.5, c, ha="center", va="center", fontsize=fs, family="monospace",
                        fontweight="bold", color=letter_colours.get(c.upper(), letter_colours.get(c, _FLAT)),
                        clip_on=True)
        seq_ax.set_ylim(0, 1)
        seq_ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            seq_ax.spines[side].set_visible(False)

    bottom = axes[-1]
    for other in axes[:-1]:
        other.tick_params(axis="x", which="both", labelbottom=False, length=0)
    bottom.set_xlim(a, b)
    if own_figure:
        fig.tight_layout()
        if len(axes) > 1:
            # tight_layout() spaces panels for the tick labels they no longer have.
            fig.subplots_adjust(hspace=0.08)
    width_in = bottom.get_position().width * fig.get_figwidth()
    max_ticks = max(2, min(12, int(width_in / (0.08 * len(f"{b:,}") + 0.25))))
    step = _nice_step(span, max_ticks)
    ticks = list(range(-(-a // step) * step, b, step))
    bottom.set_xticks([t + 0.5 for t in ticks], [f"{t:,}" for t in ticks], fontsize=text_size)
    bottom.set_xlim(a, b)
    bottom.set_xlabel("Position", fontsize=text_size)
    bottom.tick_params(axis="x", length=3, labelbottom=True)
    if bottom is not eff_ax:
        bottom.spines["bottom"].set_visible(True)
    else:
        eff_ax.spines["bottom"].set_visible(False)

    legends = _legends(eff_ax, colour_by, colours, df, letter_colours, direction, in_hit & highlight,
                       size, motif_names, palette, logos, text_size)
    if legends:
        _fit_figure(fig, axes, legends, own_figure)

    if labels:
        from ._plot_results import _label_points

        # One label per position, for the variant there with the largest
        # effect, skipping variants without one.
        ranked = np.argsort(-np.abs(y), kind="stable")
        positions = df["position"].to_numpy()
        _, first = np.unique(positions[ranked], return_index=True)
        chosen = [int(i) for i in ranked[np.sort(first)] if y[i] != 0][:labels]
        text = [""] * n
        ids = df[label_by].cast(pl.String) if label_by is not None else None
        for i in chosen:
            if ids is not None:
                t = str(ids[i])
            else:
                p, r, alt = df["position"][i], df["ref"][i], df["alt"][i]
                t = f"{p:,} {r}>{alt}" if alt is not None else f"{p:,} {r}"
            if best[i] is not None:
                t += f"\n{best[i][0]} {best[i][1]:+.1f}"
            text[i] = t
        if not _label_rows(eff_ax, x, y, area, chosen, text, 0.9 * text_size):
            # Labels keep clear of the larger effects only, so that in a
            # dense cluster they stay close to their variants.
            big = np.abs(y) >= 0.5 * np.abs(y).max()
            big[chosen] = True
            _label_points(eff_ax, x, y, area, big, chosen, text, {}, 0.9 * text_size, None)
    return fig if own_figure else eff_ax


def _spread(want: np.ndarray, widths: np.ndarray, lo: float, hi: float, gap: float) -> np.ndarray | None:
    """Centres for boxes of the given widths on a line from ``lo`` to
    ``hi``, as near the wanted centres as they can be without overlapping,
    found by merging boxes that collide into blocks centred on their wanted
    places. Returns ``None`` when they do not fit."""
    if widths.sum() + gap * (len(widths) - 1) > hi - lo:
        return None
    order = np.argsort(want, kind="stable")
    # Blocks of consecutive boxes, as (first, last, left edge).
    blocks: list[list[float]] = []
    offsets = np.concatenate([[0.0], np.cumsum(widths[order] + gap)[:-1]])
    for k in range(len(order)):
        blocks.append([k, k, want[order[k]] - widths[order[k]] / 2])
        while True:
            first, last, left = blocks[-1]
            f, la = int(first), int(last)
            size = offsets[la] - offsets[f] + widths[order[la]]
            # The left edge that puts the block's boxes nearest their wanted
            # centres on average, kept on the line.
            left = float(np.mean(want[order[f:la + 1]] - (offsets[f:la + 1] - offsets[f])
                                 - widths[order[f:la + 1]] / 2))
            left = min(max(left, lo), hi - size)
            blocks[-1][2] = left
            if len(blocks) < 2:
                break
            pf, pl_, pleft = blocks[-2]
            pend = pleft + offsets[int(pl_)] - offsets[int(pf)] + widths[order[int(pl_)]]
            if pend + gap <= left:
                break
            blocks[-2:] = [[pf, last, pleft]]
    centres = np.empty(len(order))
    for first, last, left in blocks:
        f, la = int(first), int(last)
        idx = np.arange(f, la + 1)
        centres[order[idx]] = left + offsets[idx] - offsets[f] + widths[order[idx]] / 2
    return centres


def _label_rows(ax: Axes, x, y, area, chosen: list[int], text: list[str], size: float) -> bool:
    """Label the chosen variants in rows below the lowest effect and above
    the highest, spread out along the axis and joined to their variants by
    leader lines, making room for the rows on the y-axis. Returns ``False``,
    drawing nothing, when they do not fit."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    box = ax.get_window_extent(renderer)
    width, height = box.width * to_pt, box.height * to_pt
    measured = {}
    for i in chosen:
        probe = ax.text(0, 0, text[i], fontsize=size, ha="center")
        ext = probe.get_window_extent(renderer)
        probe.remove()
        measured[i] = (ext.width * to_pt, ext.height * to_pt)
    radius = np.sqrt(np.asarray(area, dtype=float) * 1.3) / 2
    lead, gap, row_gap = 8.0, 0.6 * size, 0.3 * size
    x_pts = (ax.transData.transform(np.column_stack([x, np.zeros(len(x))]))[:, 0] - box.x0) * to_pt

    sides = {}
    for sign in (-1, 1):
        ids = [i for i in chosen if np.sign(y[i]) == sign]
        if not ids:
            continue
        rows = None
        for nrows in (1, 2):
            # With two rows, labels alternate between them along the axis.
            ordered = sorted(ids, key=lambda i: x[i])
            groups = [ordered[r::nrows] for r in range(nrows)]
            spots = [_spread(x_pts[g], np.array([measured[i][0] for i in g]), 0.0, width, gap)
                     for g in groups]
            if all(s is not None for s in spots):
                rows = list(zip(groups, spots))
                break
        if rows is None:
            return False
        tall = max(measured[i][1] for i in ids)
        need = float(radius[ids].max()) + lead + len(rows) * tall + (len(rows) - 1) * row_gap
        sides[sign] = (rows, tall, need)

    lo, hi = ax.get_ylim()
    need = sum(v[2] for v in sides.values())
    if height - need < 0.4 * height:
        return False
    # The rows take ``need`` points beyond the old limits, which leaves the
    # rest of the axes to the data.
    scale = (height - need) / (hi - lo)
    new_lo = lo - sides[-1][2] / scale if -1 in sides else lo
    new_hi = hi + sides[1][2] / scale if 1 in sides else hi
    from matplotlib.ticker import MaxNLocator

    # As many ticks as fit the shorter span of the effects.
    nbins = max(2, int((height - need) // (3.5 * size)))
    ticks = [t for t in MaxNLocator(nbins=nbins).tick_values(lo, hi) if lo - 1e-12 <= t <= hi + 1e-12]
    ax.set_ylim(new_lo, new_hi)
    # The axis spans the effects only, not the rows of labels.
    ax.set_yticks(ticks)
    ax.spines["left"].set_bounds(lo, hi)
    for sign, (rows, tall, _) in sides.items():
        edge = min(y[i] for g, _ in rows for i in g) if sign < 0 else max(y[i] for g, _ in rows for i in g)
        edge = (edge - new_lo) * height / (new_hi - new_lo)
        ids = [i for g, _ in rows for i in g]
        start = edge + sign * (float(radius[ids].max()) + lead)
        for r, (group, centres) in enumerate(rows):
            ty = start + sign * r * (tall + row_gap)
            for i, cx in zip(group, centres):
                arrow = {"arrowstyle": "-", "color": "#7F7F7F", "linewidth": 0.6,
                         "relpos": (0.5, 1.0 if sign < 0 else 0.0), "shrinkA": 1,
                         "shrinkB": float(radius[i]) + 1}
                ax.annotate(text[i], xy=(x[i], y[i]), xytext=(cx / width, ty / height),
                            textcoords="axes fraction", ha="center", va="top" if sign < 0 else "bottom",
                            fontsize=size, zorder=5, arrowprops=arrow)
    return True


def _legends(ax, colour_by, colours, df, letter_colours, direction, ringed, size, motif_names, palette,
             logos, text_size) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    def dot(colour, marker="o", edge="white", size=6.0, width=0.5):
        return Line2D([], [], linestyle="", marker=marker, markersize=size, markerfacecolor=colour,
                      markeredgecolor=edge, markeredgewidth=width)

    def legend(handles, names, title):
        return lambda y: ax.legend(handles, names, title=title, frameon=False, loc="upper left",
                                   bbox_to_anchor=(1.01, y), fontsize=text_size, title_fontsize=text_size,
                                   borderaxespad=0.0, handlelength=1.0, alignment="left")

    legends: list = []
    if colour_by == "sign":
        entries = [(c, name) for c, name in ((_UP, "Positive"), (_DOWN, "Negative")) if (colours == c).any()]
        title = "Effect"
    else:
        present = set(df["alt"].to_list())
        entries = [(letter_colours[c], c) for c in letter_colours if c in present]
        title = "Alternative"
    if entries:
        _stack(ax, legends, legend([dot(c) for c, _ in entries], [t for _, t in entries], title))
    sites = []
    if (direction > 0).any():
        sites.append((dot(_MUTED, "^", size=7), "Strengthened"))
    if (direction < 0).any():
        sites.append((dot(_MUTED, "v", size=7), "Weakened"))
    if ringed.any():
        sites.append((dot("white", edge=_RING, width=1.1), "In a hit"))
    if sites:
        _stack(ax, legends, legend([h for h, _ in sites], [t for _, t in sites], "Motif sites"))
    if size:
        title, lo, hi = size
        if hi > lo:
            breaks = [v for v in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= v <= hi]
            handles = [dot(_MUTED, size=float(np.sqrt(10.0 + 70.0 * (v - lo) / (hi - lo)))) for v in breaks]
            _stack(ax, legends, legend(handles, [f"{v:g}" for v in breaks], title))
    if motif_names:
        from ._plot_hits import _draw_legends

        made = _draw_legends(ax, motif_names, palette, logos, [], text_size, 0.8 * text_size)
        if made:
            leg = made[0]
            if legends:
                renderer = ax.figure.canvas.get_renderer()
                below = legends[-1].get_window_extent(renderer)
                leg.set_bbox_to_anchor((1.01, ax.transAxes.inverted().transform((0, below.y0))[1] - 0.04),
                                       transform=ax.transAxes)
            legends.append(leg)
    return legends
