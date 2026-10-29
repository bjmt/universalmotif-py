"""A TF-MoDISco report as a single figure."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from . import _core
from ._api import _check_motif, motif_rc
from ._core import Motif
from ._deep import read_modisco
from ._plot import _pyplot, draw_logo

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LOGO_TYPES = ("CWM", "hCWM", "PPM")
_COLUMNS = ("seqlets", *_LOGO_TYPES)
_GROUPS = {"pos": "Positive patterns", "neg": "Negative patterns", None: None}
_MIN_OVERLAP = 5
_GREY = "#6E6E6E"
_RULE = "#E3E3E3"


@dataclass
class _Pattern:
    name: str
    label: str
    group: str | None
    nsites: int | None
    motifs: dict[str, Motif]
    matches: list[tuple[str, Motif | None, float | None, float | None, str]] = field(default_factory=list)
    placements: list[int] = field(default_factory=list)
    span: int = 0


def _default_label(name: str) -> str:
    """``pos_patterns.anything_pattern_3`` as ``pos_3``, and other names as
    they are."""
    group, _, rest = name.partition(".")
    if group not in ("pos_patterns", "neg_patterns") or not rest:
        return name
    nums = [part.rsplit("_", 1)[-1] for part in rest.split(".")]
    if not all(n.isdigit() for n in nums):
        return name
    return f"{group[:3]}_{'.'.join(nums)}"


def _group(name: str) -> str | None:
    for g in ("pos", "neg"):
        if name.startswith(g):
            return g
    return None


def _pattern_motifs(modisco, types: list[str], trim_threshold) -> dict[str, list[Motif]]:
    if isinstance(modisco, (str, os.PathLike)):
        return {t: read_modisco(modisco, type=t, trim_threshold=trim_threshold) for t in types}
    if isinstance(modisco, Motif):
        modisco = [modisco]
    if isinstance(modisco, Mapping):
        bad = [k for k in modisco if k not in _LOGO_TYPES]
        if bad:
            raise ValueError(f"motif lists must be keyed by 'CWM', 'hCWM' or 'PPM', not {bad[0]!r}")
        given = {k: [_check_motif(m) for m in v] for k, v in modisco.items()}
    else:
        ms = [_check_motif(m) for m in modisco]
        if not ms:
            raise ValueError("no patterns given")
        if all(m.type == "CWM" for m in ms):
            given = {"CWM": ms}
        elif not any(m.type == "CWM" for m in ms):
            given = {"PPM": ms}
        else:
            raise ValueError("give CWMs and PPMs as a dict of lists, such as {'CWM': cwms, 'PPM': ppms}")
    sizes = {len(v) for v in given.values()}
    if len(sizes) != 1:
        raise ValueError("every list of motifs must have one motif per pattern")
    if not sizes.pop():
        raise ValueError("no patterns given")
    for k in ("CWM", "hCWM"):
        if k in given and any(m.type != "CWM" for m in given[k]):
            raise ValueError(f"the {k} motifs must be CWMs")
    return given


def _select(pats: list[_Pattern], patterns) -> list[_Pattern]:
    if patterns is None:
        return pats
    if isinstance(patterns, int) or (isinstance(patterns, tuple) and len(patterns) == 2
                                     and all(isinstance(n, int) for n in patterns)):
        n_pos, n_neg = (patterns, patterns) if isinstance(patterns, int) else patterns
        if n_pos < 0 or n_neg < 0:
            raise ValueError("the number of patterns cannot be negative")
        limit = {"pos": n_pos, "neg": n_neg, None: n_pos}
        seen = dict.fromkeys(limit, 0)
        out = []
        for p in pats:
            if seen[p.group] < limit[p.group]:
                out.append(p)
                seen[p.group] += 1
        return out
    if isinstance(patterns, str):
        patterns = [patterns]
    lookup = {}
    for p in pats:
        lookup.setdefault(p.name, p)
        lookup.setdefault(p.label, p)
    out = []
    for key in patterns:
        if key not in lookup:
            raise ValueError(f"no pattern named {key!r}")
        out.append(lookup[key])
    order = list(_GROUPS)
    return sorted(out, key=lambda p: order.index(p.group))


def _find_matches(pats: list[_Pattern], every: list[_Pattern], matches: pl.DataFrame, database,
                  n_matches: int) -> None:
    if not isinstance(matches, pl.DataFrame):
        raise TypeError("matches must be a DataFrame from compare_motifs() with compare_to")
    missing = [c for c in ("subject", "target", "qvalue") if c not in matches.columns]
    if missing:
        raise ValueError(f"matches has no {missing[0]!r} column; give the DataFrame compare_motifs() "
                         "returns with compare_to")
    db: dict[str, Motif] | None = None
    if database is not None:
        if isinstance(database, Motif):
            database = [database]
        if isinstance(database, Mapping):
            db = {str(k): _check_motif(v) for k, v in database.items()}
        else:
            db = {}
            for m in database:
                m = _check_motif(m)
                db.setdefault(m.name, m)
    # Patterns compared with each other are left out by what they are, the
    # motifs compared as subjects, and not by their labels, which can be the
    # name of the very database motif a pattern matches.
    names = {p.name for p in every}
    labels = {p.label for p in every}
    subjects = set(matches["subject"].cast(pl.String).to_list())
    if "subject_i" in matches.columns and "target_i" in matches.columns:
        ids = matches.filter(pl.col("subject").is_in(list(names | labels)))["subject_i"].unique().to_list()
        own = pl.col("target_i").is_in(ids) | pl.col("target").is_in(list(names))
    else:
        own = pl.col("target").is_in(list(names | (labels & subjects)))
    keys = [c for c in ("qvalue", "pvalue") if c in matches.columns]
    table = matches.filter(~own)
    if db is not None:
        table = table.filter(pl.col("target").is_in(list(db)))
    if "score" in table.columns:
        table = table.sort([*keys, "score"], descending=[False] * len(keys) + [True], maintain_order=True)
    else:
        table = table.sort(keys, maintain_order=True)
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in table.iter_rows(named=True):
        rows.setdefault(row["subject"], []).append(row)
    for p in pats:
        found = rows.get(p.label) or rows.get(p.name) or []
        for row in found[:n_matches]:
            m = None if db is None else db[row["target"]]
            strand = row.get("strand") or "+"
            if m is not None and strand == "-":
                m = motif_rc(m)
            p.matches.append((row["target"], m, row.get("score"), row.get("qvalue"), strand))


def _place(p: _Pattern, shown: list[str], align: bool) -> None:
    """Where each logo of a row starts in the row's frame of positions: the
    pattern's own logos at ``placements[0]`` and the matches after it."""
    widths = [p.motifs[t].width for t in shown]
    logos = [m for _, m, *_ in p.matches if m is not None]
    ref = p.motifs.get("PPM") or p.motifs.get("CWM") or p.motifs.get("hCWM")
    lefts = [0] * (1 + len(logos))
    if align and logos and ref is not None:
        try:
            placed, _ = _core._anchor_layout([ref, *logos], 0, _MIN_OVERLAP, False, 1)
            lefts = [left for left, _ in placed]
        except ValueError:
            pass
    ends = [lefts[0] + w for w in widths] + [left + m.width for left, m in zip(lefts[1:], logos)]
    p.placements = lefts
    p.span = max(ends, default=0)


def _text_width(fig, texts, size: float, weight: str = "normal") -> float:
    """The width of the widest text, in inches. Small text is narrower at
    low resolutions, so it is measured at 200 dpi or more to leave room
    for saving at a higher one."""
    dpi = fig.dpi
    fig.dpi = max(dpi, 200)
    try:
        renderer = fig.canvas.get_renderer()
        widest = 0.0
        for s in texts:
            if not s:
                continue
            t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
            widest = max(widest, t.get_window_extent(renderer).width)
            t.remove()
        return widest / fig.dpi
    finally:
        fig.dpi = dpi


def _match_lines(target: str, m: Motif | None, score, qvalue) -> list[str]:
    stats = []
    if qvalue is not None:
        stats.append(f"q = {qvalue:.2g}")
    if score is not None:
        stats.append(f"score {score:.2f}")
    if m is not None and m.altname:
        return [m.altname, target, ", ".join(stats)]
    return [target, ", ".join(stats)]


def plot_modisco(
    modisco: str | os.PathLike[str] | Sequence[Motif] | Mapping[str, Sequence[Motif]],
    matches: pl.DataFrame | None = None,
    database: Sequence[Motif] | Mapping[str, Motif] | None = None,
    names: Mapping[str, str] | Callable[[str], str] | None = None,
    patterns: int | tuple[int, int] | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    n_matches: int = 1,
    align: bool = False,
    use_type: str = "ICM",
    trim_threshold: float | None = 0.3,
    text_size: float = 10,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Draw a TF-MoDISco report, one row per pattern.

    Each row gives a pattern's name, its number of seqlets and its logos,
    by default its contribution weight matrix (CWM) and the letter
    frequencies of its seqlets drawn as information content, followed by its
    best matches in a motif database with their logos, names and q-values.
    Positive and negative patterns are drawn under their own headings.

    ``modisco`` is the path of a TF-MoDISco results file, which is read with
    [`read_modisco()`][universalmotif.read_modisco] using
    ``trim_threshold``, so that every logo of a pattern spans the same
    positions. It can also be the patterns read beforehand, as a list of
    CWMs or PPMs or as a dict of lists keyed by ``"CWM"``, ``"hCWM"`` and
    ``"PPM"``, each with one motif per pattern in the same order.

    Patterns are labelled by their group and number, so that
    ``pos_patterns.pattern_3`` becomes ``pos_3``, unless ``names`` gives
    labels of its own, as a dict from the names patterns are read with to
    labels or as a function of the name. The group of a pattern comes from
    the start of its name, ``pos`` or ``neg``.

    Matches come from ``matches``, the DataFrame
    [`compare_motifs()`][universalmotif.compare_motifs] returns with
    ``compare_to`` set to the patterns. Its ``subject`` column is matched to
    either the label or the full name of each pattern, so the comparison
    can be run on renamed motifs, and patterns compared with each other are
    left out. With ``database``, the motifs compared against, only matches
    to its motifs are kept and each is drawn as a logo, labelled with its
    ``altname`` when it has one. A match that was found on the other strand
    is drawn as its reverse complement, so that it reads the same way as the
    pattern. With ``align``, the logos of each row are also placed in a
    shared frame of positions, found by aligning each match to the pattern
    as drawn, so that the part of a match lining up with the pattern sits at
    the same place in its cell as the pattern does in the cells to its left.
    Long matches then widen every logo column, which is why it is off by
    default.

    Needs the ``plot`` extra, and the ``h5py`` extra to read a file.

    Parameters
    ----------
    modisco
        A TF-MoDISco results file, or the patterns as motifs.
    matches
        Matches of the patterns to known motifs, from ``compare_motifs()``.
    database
        The known motifs, as a list or a dict keyed by name, to draw the
        matches with.
    names
        Labels for the patterns, as a dict or a function of the full name.
    patterns
        Which patterns to draw: all of them by default, the first ``n`` of
        each group for an integer ``n``, the first ``n_pos`` positive and
        ``n_neg`` negative patterns for a pair, or the patterns named in a
        list of labels or names.
    columns
        The columns to draw after the label, from ``"seqlets"``, ``"CWM"``,
        ``"hCWM"`` and ``"PPM"``, by default ``seqlets``, ``CWM`` and
        ``PPM`` for a file and the seqlets and each type given otherwise.
    n_matches
        The number of matches to draw for each pattern.
    align
        Place the logos of each row in a shared frame of positions.
    use_type
        How the PPMs and matches are drawn, ``"ICM"`` for information
        content or ``"PPM"`` for probabilities.
    trim_threshold
        Passed to ``read_modisco()`` when reading a file.
    text_size
        The font size of the labels.
    figsize
        The figure size, by default worked out from the number of patterns
        and the widest frame of positions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, whose single axes is labelled ``"modisco"``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import universalmotif as um
    >>> def pattern(name, core, nsites):
    ...     cwm = np.zeros((4, len(core)))
    ...     cwm[["ACGT".index(c) for c in core], range(len(core))] = 1.0
    ...     ppm = cwm * 0.9 + 0.025
    ...     return (um.create_motif(cwm, alphabet="DNA", type="CWM", name=name, nsites=nsites),
    ...             um.create_motif(ppm, alphabet="DNA", type="PPM", name=name, nsites=nsites))
    >>> found = [pattern("pos_patterns.pattern_0", "TTGACGTCAT", 120),
    ...          pattern("pos_patterns.pattern_1", "CTATAAAAGG", 40),
    ...          pattern("neg_patterns.pattern_0", "CCACGTGG", 30)]
    >>> cwms = [cwm for cwm, _ in found]
    >>> ppms = [ppm for _, ppm in found]
    >>> known = um.example_motifs()
    >>> cmp = um.compare_motifs(ppms + known, compare_to=[0, 1, 2], qvalue=0.5)
    >>> fig = um.plot_modisco({"CWM": cwms, "PPM": ppms}, matches=cmp, database=known)
    >>> [t.get_text() for t in fig.axes[0].texts][:5]
    ['Pattern', 'Seqlets', 'CWM', 'PPM', 'Best match']
    >>> plt.close(fig)

    A results file is read directly, here keeping the three most common
    positive patterns and the most common negative one:

    >>> fig = um.plot_modisco("modisco_results.h5", matches=cmp, database=known,
    ...                       patterns=(3, 1))  # doctest: +SKIP
    """
    if columns is not None:
        columns = [columns] if isinstance(columns, str) else list(columns)
        bad = [c for c in columns if c not in _COLUMNS]
        if bad:
            raise ValueError(f"columns must be among 'seqlets', 'CWM', 'hCWM' and 'PPM', not {bad[0]!r}")
    if n_matches < 1:
        raise ValueError("n_matches must be at least 1")
    if use_type not in ("ICM", "PPM"):
        raise ValueError("use_type must be 'ICM' or 'PPM'")
    if database is not None and matches is None:
        raise ValueError("database needs matches to know which motifs to draw")

    is_file = isinstance(modisco, (str, os.PathLike))
    if is_file:
        if columns is None:
            columns = ["seqlets", "CWM", "PPM"]
        types = [t for t in _LOGO_TYPES if t in columns]
        if "PPM" not in types and matches is not None:
            # The PPMs are what the matches are aligned to.
            types.append("PPM")
        given = _pattern_motifs(modisco, types or ["CWM"], trim_threshold)
    else:
        given = _pattern_motifs(modisco, [], trim_threshold)
        if columns is None:
            columns = ["seqlets", *[t for t in _LOGO_TYPES if t in given]]
        missing = [c for c in columns if c in _LOGO_TYPES and c not in given]
        if missing:
            raise ValueError(f"no {missing[0]} motifs were given for that column")
    shown = [c for c in columns if c in _LOGO_TYPES]

    first = next(iter(given.values()))
    every = []
    for i, m in enumerate(first):
        motifs = {t: given[t][i] for t in given}
        nsites = next((int(x.nsites) for x in motifs.values() if x.nsites is not None), None)
        if names is None:
            label = _default_label(m.name)
        elif callable(names):
            label = str(names(m.name))
        else:
            label = str(names.get(m.name, _default_label(m.name)))
        every.append(_Pattern(m.name, label, _group(m.name), nsites, motifs))
    pats = _select(every, patterns)
    if not pats:
        raise ValueError("no patterns to draw")
    if matches is not None:
        _find_matches(pats, every, matches, database, n_matches)
    for p in pats:
        _place(p, shown, align)
    n_cols = n_matches if matches is not None else 0

    plt = _pyplot()
    fig = plt.figure()
    small = 0.8 * text_size
    labels = [p.label for p in pats]
    seqlets = ["" if p.nsites is None else f"{p.nsites:,}" for p in pats]
    head = {"label": "Pattern", "seqlets": "Seqlets", "CWM": "CWM", "hCWM": "hCWM", "PPM": "PPM"}
    match_head = ["Best match"] if n_cols == 1 else [f"Match {k + 1}" for k in range(n_cols)]
    lines = [_match_lines(*m[:4]) for p in pats for m in p.matches]

    gap, pad = 0.15, 0.1
    widths = {
        "label": max(_text_width(fig, labels, text_size, "bold"), _text_width(fig, [head["label"]], small,
                                                                             "bold")),
        "seqlets": max(_text_width(fig, seqlets, text_size), _text_width(fig, [head["seqlets"]], small,
                                                                         "bold")),
    }
    match_text = max([_text_width(fig, [ls[0]], small) for ls in lines]
                     + [_text_width(fig, ls[1:], 0.75 * text_size) for ls in lines]
                     + [_text_width(fig, ["no match"], small)], default=0.0)
    # With align every logo column is as wide as the widest row's frame,
    # and otherwise as wide as its own widest logo.
    frame = max(p.span for p in pats) or 1
    units = [frame if align else max(p.motifs[t].width for p in pats) for t in shown]
    match_units = []
    if database is not None:
        for k in range(n_cols):
            ws = [p.matches[k][1].width for p in pats if k < len(p.matches) and p.matches[k][1] is not None]
            match_units.append(frame if align else max(ws, default=4))
    total_units = sum(units) + sum(match_units)
    has_logos = total_units > 0
    texts_w = (2 * pad + widths["label"] + gap
               + ("seqlets" in columns) * (widths["seqlets"] + gap)
               + n_cols * (match_text + gap))
    n_logo_cols = len(shown) + (n_cols if database is not None else 0)
    n_heads = sum(1 for g in dict.fromkeys(p.group for p in pats) if _GROUPS[g])
    head_h, group_h = 0.3, 0.32
    if figsize is None:
        unit = 0.1 if not has_logos else min(0.1, max(0.05, 10.0 / total_units))
        row_h = 0.55
        width = texts_w + total_units * unit + n_logo_cols * gap
        height = 2 * pad + head_h + n_heads * group_h + len(pats) * row_h
    else:
        width, height = figsize
        row_h = (height - 2 * pad - head_h - n_heads * group_h) / len(pats)
        unit = (width - texts_w - n_logo_cols * gap) / total_units if has_logos else 0.1
        if row_h <= 0 or unit <= 0:
            raise ValueError("figsize is too small for the labels")
    fig.set_size_inches(width, height)
    ax = fig.add_axes((0, 0, 1, 1), label="modisco")
    ax.set_axis_off()

    # Left edges of the columns, in inches.
    x = pad
    left = {"label": x}
    x += widths["label"] + gap
    if "seqlets" in columns:
        left["seqlets"] = x
        x += widths["seqlets"] + gap
    for t, u in zip(shown, units):
        left[t] = x
        x += u * unit + gap
    match_left = []
    for k in range(n_cols):
        logo_x = x if database is not None else None
        if database is not None:
            x += match_units[k] * unit + gap
        match_left.append((logo_x, x))
        x += match_text + gap
    right = x - gap

    def heading(s, x0, y, ha="left"):
        ax.text(x0, y, s, ha=ha, va="center", fontsize=small, fontweight="bold", color="#333333")

    y = height - pad - head_h / 2
    heading(head["label"], left["label"], y)
    if "seqlets" in columns:
        heading(head["seqlets"], left["seqlets"] + widths["seqlets"], y, ha="right")
    for t, u in zip(shown, units):
        heading(head[t], left[t] + u * unit / 2, y, ha="center")
    for k, (logo_x, text_x) in enumerate(match_left):
        if logo_x is None:
            heading(match_head[k], text_x, y)
        else:
            heading(match_head[k], logo_x + match_units[k] * unit / 2, y, ha="center")
    y = height - pad - head_h
    ax.plot([pad, right], [y, y], color="#9A9A9A", linewidth=0.8, solid_capstyle="butt")

    def centred(u, w, span):
        # Logos sit in the middle of their column, and aligned rows keep
        # their shared frame together.
        return (u - (span if align else w)) / 2 * unit

    group = object()
    for i, p in enumerate(pats):
        if p.group != group:
            group = p.group
            if _GROUPS[group]:
                ax.text(pad, y - group_h * 0.55, _GROUPS[group], ha="left", va="center",
                        fontsize=text_size, fontstyle="italic", color="#333333")
                y -= group_h
                ax.plot([pad, right], [y, y], color=_RULE, linewidth=0.6, solid_capstyle="butt")
        mid = y - row_h / 2
        ax.text(left["label"], mid, p.label, ha="left", va="center", fontsize=text_size,
                fontweight="bold")
        if "seqlets" in columns:
            ax.text(left["seqlets"] + widths["seqlets"], mid, seqlets[i], ha="right", va="center",
                    fontsize=text_size)
        logo_y, logo_h = y - 0.88 * row_h, 0.76 * row_h
        for t, u in zip(shown, units):
            draw_logo(ax, p.motifs[t], x=left[t] + centred(u, p.motifs[t].width, p.span) + p.placements[0] * unit,
                      y=logo_y,
                      width=p.motifs[t].width * unit, height=logo_h,
                      use_type="CWM" if t != "PPM" else use_type)
        k_logo = 1
        for k, (logo_x, text_x) in enumerate(match_left):
            if k >= len(p.matches):
                if k == 0:
                    ax.text(logo_x if logo_x is not None else text_x, mid, "no match", ha="left",
                            va="center", fontsize=small, color=_GREY)
                continue
            target, m, score, qvalue, _ = p.matches[k]
            if m is not None and logo_x is not None:
                draw_logo(ax, m, x=logo_x + centred(match_units[k], m.width, p.span) + p.placements[k_logo] * unit,
                          y=logo_y,
                          width=m.width * unit, height=logo_h, use_type=use_type)
                k_logo += 1
            ls = _match_lines(target, m, score, qvalue)
            sizes = [small] + [0.75 * text_size] * (len(ls) - 1)
            step = 1.25 * max(sizes) / 72
            top = mid + step * (len(ls) - 1) / 2
            for j, (s, size) in enumerate(zip(ls, sizes)):
                ax.text(text_x, top - j * step, s, ha="left", va="center", fontsize=size,
                        color="black" if j == 0 else _GREY)
        y -= row_h
        if i + 1 < len(pats) and pats[i + 1].group == p.group:
            ax.plot([pad, right], [y, y], color=_RULE, linewidth=0.5, solid_capstyle="butt")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    return fig


__all__ = ["plot_modisco"]
