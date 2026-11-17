"""A report of motif patterns, from TF-MoDISco or contribution_motifs(), as a
figure and as an HTML page."""

from __future__ import annotations

import html as _html
import os
import pathlib
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

    from ._contribution_motifs import ContributionMotifs

_LOGO_TYPES = ("CWM", "hCWM", "PPM")
_COLUMNS = ("seqlets", "sources", "hits", *_LOGO_TYPES)
# The columns of numbers, in the order they are drawn.
_COUNTS = ("seqlets", "sources", "hits")
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
    # Known only for the patterns of contribution_motifs().
    share: float | None = None
    sources: int | None = None
    hits: int | None = None
    flags: list[str] | None = None
    consensus: str | None = None
    members: list[dict[str, Any]] = field(default_factory=list)


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
        raise TypeError("matches must be a DataFrame from name_motifs() or compare_motifs()")
    if "subject" not in matches.columns and {"motif", "match"} <= set(matches.columns):
        # A table from name_motifs(), whose motifs were named apart from one another.
        matches = matches.rename({"motif": "subject", "match": "target"}).drop("motif_i", strict=False)
    missing = [c for c in ("subject", "target", "qvalue") if c not in matches.columns]
    if missing:
        raise ValueError(f"matches has no {missing[0]!r} column; give the DataFrame name_motifs() returns, "
                         "or that of compare_motifs() with compare_to")
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
    # Patterns compared with each other are left out by what they are (the
    # motifs compared as subjects), not by their labels. A label can be the
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
    """Where each logo of a row starts in the row's frame of positions. The
    pattern's own logos are at ``placements[0]``, and the matches after
    it."""
    widths = [p.motifs[t].width for t in shown]
    logos = [m for _, m, *_ in p.matches if m is not None]
    ref = p.motifs.get("PPM") or p.motifs.get("CWM") or p.motifs.get("hCWM")
    lefts = [0] * (1 + len(logos))
    if align and logos and ref is not None:
        try:
            placed, _, _ = _core._anchor_layout([ref, *logos], 0, _MIN_OVERLAP, False, 1)
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


def _match_lines(target: str, m: Motif | None, qvalue) -> list[str]:
    """A match's name, then its database ID and q-value."""
    q = "" if qvalue is None else f"q = {qvalue:.2g}"
    if m is not None and m.altname:
        return [m.altname, f"{target} · {q}" if q else target]
    return [target, q] if q else [target]


def _count(p: _Pattern, column: str) -> int | None:
    return {"seqlets": p.nsites, "sources": p.sources, "hits": p.hits}[column]


_FLAGS = ("truncated", "mixed_sign", "low_complexity", "diffuse")


def _combined_motifs(comp) -> tuple[dict[str, list[Motif]], dict[str, dict[str, Any]]]:
    """The clusters of combine_patterns() as CWMs and PPMs, with their
    patterns as members."""
    cl = comp.clusters
    if not cl.height:
        raise ValueError("no patterns to draw")
    given: dict[str, list[Motif]] = {"CWM": cl["cwm"].to_list()}
    if all(m is not None for m in cl["ppm"]):
        given["PPM"] = cl["ppm"].to_list()
    extra = {}
    for row in cl.iter_rows(named=True):
        members = comp.patterns.filter(pl.col("cluster") == row["cluster"]).sort("seqlets", descending=True)
        extra[row["cluster"]] = {
            "nsites": row["seqlets"], "sources": row["n_sources"], "consensus": row["consensus"] or None,
            "flags": [f.replace("_", " ") for f in _FLAGS if row[f]],
            "members": [{"name": f"{r['source']} {r['pattern']}", "seqlets": r["seqlets"], "cwm": r["cwm"]}
                        for r in members.iter_rows(named=True)],
        }
    return given, extra


def _found_motifs(found: ContributionMotifs) -> tuple[dict[str, list[Motif]], dict[str, dict[str, Any]]]:
    """The patterns of contribution_motifs() as CWMs and PPMs named by
    pattern, and what else is known of each."""
    pats = found.patterns
    if not pats.height:
        raise ValueError("no patterns to draw")
    counts = {}
    if found.hits is not None:
        counts = dict(found.hits.group_by("motif").len().iter_rows())
    given: dict[str, list[Motif]] = {"CWM": [], "PPM": []}
    extra = {}
    for row in pats.iter_rows(named=True):
        name = row["pattern"]
        given["CWM"].append(row["cwm"].replace(name=name))
        given["PPM"].append(row["motif"].replace(name=name))
        members = found.motifs.filter(pl.col("pattern") == name)
        extra[name] = {
            "nsites": row["seqlets"], "share": row["share"], "consensus": row["consensus"],
            "hits": sum(counts.get(m, 0) for m in row["members"]) if found.hits is not None else None,
            "members": list(members.select("name", "consensus", "seqlets", "source", "cwm").iter_rows(named=True)),
        }
    return given, extra


def _subpatterns(path, trim_threshold) -> dict[str, dict[str, Any]]:
    """The CWMs of each pattern's subpatterns in a TF-MoDISco file."""
    out: dict[str, dict[str, Any]] = {}
    for m in read_modisco(path, type="CWM", trim_threshold=trim_threshold, subpatterns=True):
        parent, _, sub = m.name.rpartition(".")
        if sub.startswith("subpattern_"):
            out.setdefault(parent, {"members": []})["members"].append(
                {"name": f"sub_{sub.rsplit('_', 1)[1]}", "seqlets": int(m.nsites or 0), "cwm": m})
    return out


def _prepare(modisco, matches, database, names, patterns, columns, n_matches, qvalue, use_type, trim_threshold,
             align, nthreads):
    """The patterns to report, with their matches, and the columns shown."""
    from ._combine import CombinedPatterns
    from ._contribution_motifs import ContributionMotifs

    if columns is not None:
        columns = [columns] if isinstance(columns, str) else list(columns)
        bad = [c for c in columns if c not in _COLUMNS]
        if bad:
            raise ValueError(f"columns must be among 'seqlets', 'sources', 'hits', 'CWM', 'hCWM' and 'PPM', "
                             f"not {bad[0]!r}")
    if n_matches < 1:
        raise ValueError("n_matches must be at least 1")
    if use_type not in ("ICM", "PPM"):
        raise ValueError("use_type must be 'ICM' or 'PPM'")

    extra: dict[str, dict[str, Any]] = {}
    if isinstance(modisco, ContributionMotifs):
        given, extra = _found_motifs(modisco)
        if columns is None:
            columns = ["seqlets", "CWM", "PPM"]
    elif isinstance(modisco, CombinedPatterns):
        given, extra = _combined_motifs(modisco)
        if columns is None:
            columns = ["seqlets", "sources", *given]
    elif isinstance(modisco, (str, os.PathLike)):
        if columns is None:
            columns = ["seqlets", "CWM", "PPM"]
        types = [t for t in _LOGO_TYPES if t in columns]
        if "PPM" not in types and (matches is not None or database is not None):
            # The PPMs are what the matches are aligned to.
            types.append("PPM")
        given = _pattern_motifs(modisco, types or ["CWM"], trim_threshold)
        extra = _subpatterns(modisco, trim_threshold)
    else:
        given = _pattern_motifs(modisco, [], trim_threshold)
        if columns is None:
            columns = ["seqlets", *[t for t in _LOGO_TYPES if t in given]]
    missing = [c for c in columns if c in _LOGO_TYPES and c not in given]
    if missing:
        raise ValueError(f"no {missing[0]} motifs were given for that column")
    if "hits" in columns and not any(e.get("hits") is not None for e in extra.values()):
        raise ValueError("the hits column needs the result of contribution_motifs(), with its hits")
    if "sources" in columns and not any(e.get("sources") is not None for e in extra.values()):
        raise ValueError("the sources column needs the result of combine_patterns()")
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
        p = _Pattern(m.name, label, _group(m.name), nsites, motifs)
        for k, v in extra.get(m.name, {}).items():
            setattr(p, k, v)
        every.append(p)
    pats = _select(every, patterns)
    if not pats:
        raise ValueError("no patterns to draw")
    if matches is None and database is not None:
        # Name the patterns by the database the TOMTOM way.
        from ._compare import name_motifs

        db = [database] if isinstance(database, Motif) else list(
            database.values() if isinstance(database, Mapping) else database)
        query = [p.motifs.get("PPM") or p.motifs.get("CWM") or p.motifs["hCWM"] for p in every]
        matches = name_motifs([q.replace(name=p.name) for q, p in zip(query, every)], db, top=n_matches,
                              qvalue=qvalue, nthreads=nthreads)
    if matches is not None:
        _find_matches(pats, every, matches, database, n_matches)
    for p in pats:
        _place(p, shown, align)
    n_cols = n_matches if matches is not None else 0
    return pats, columns, shown, n_cols


def plot_modisco(
    modisco: str | os.PathLike[str] | Sequence[Motif] | Mapping[str, Sequence[Motif]] | ContributionMotifs,
    matches: pl.DataFrame | None = None,
    database: Sequence[Motif] | Mapping[str, Motif] | None = None,
    names: Mapping[str, str] | Callable[[str], str] | None = None,
    patterns: int | tuple[int, int] | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    n_matches: int = 3,
    qvalue: float = 0.05,
    align: bool = False,
    use_type: str = "ICM",
    trim_threshold: float | None = 0.3,
    text_size: float = 10,
    figsize: tuple[float, float] | None = None,
    html: str | os.PathLike[str] | None = None,
    title: str | None = None,
    nthreads: int | None = None,
) -> Figure:
    """Draw a report of motif patterns, one row per pattern, from TF-MoDISco
    or from [`contribution_motifs()`][universalmotif.contribution_motifs].

    Each row holds, from left to right:

    - the pattern's name and its number of seqlets;
    - its logos, by default its contribution weight matrix (CWM) and the
      letter frequencies of its seqlets drawn as information content;
    - its best matches in a motif database, with their logos, names and
      q-values.

    Positive and negative patterns are drawn under their own headings.
    With ``html``, the same report is also written as an HTML page, which
    [`ContributionMotifs.report()`][universalmotif.ContributionMotifs.report]
    writes on its own. Each row of the page has a button that shows its
    logos and consensus as their reverse complements.

    ``modisco`` is the path of a TF-MoDISco results file. It is read with
    [`read_modisco()`][universalmotif.read_modisco] using
    ``trim_threshold``, so every logo of a pattern spans the same
    positions. It can also be the patterns read beforehand, as a list of
    CWMs or PPMs. A dict of lists keyed by ``"CWM"``, ``"hCWM"`` and
    ``"PPM"`` also works, each with one motif per pattern in the same
    order. Finally, it can be the result of ``contribution_motifs()``,
    whose patterns are drawn by their leading motifs.

    Patterns are labelled by their group and number, so
    ``pos_patterns.pattern_3`` becomes ``pos_3``. ``names`` can give labels
    of its own instead, as a dict from the names patterns are read with to
    labels, or as a function of the name. The group of a pattern comes from
    the start of its name, ``pos`` or ``neg``.

    Given a ``database`` alone, the patterns are named by it with
    [`name_motifs()`][universalmotif.name_motifs], keeping matches with a
    q-value of at most ``qvalue``. Matches can also be given as
    ``matches``, a table from ``name_motifs()`` or the DataFrame
    [`compare_motifs()`][universalmotif.compare_motifs] returns with
    ``compare_to`` set to the patterns. Each match is found by either the
    label or the full name of its pattern, so the comparison can be run on
    renamed motifs. Patterns compared with each other are left out.

    With a ``database``, each match is drawn as a logo labelled with its
    ``altname`` when it has one, and only matches to its motifs are kept.
    A match found on the other strand is drawn as its reverse complement,
    so it reads the same way as the pattern.

    With ``align``, the logos of each row are also placed in a shared frame
    of positions, found by aligning each match to the pattern as drawn. The
    part of a match lining up with the pattern then sits at the same place
    in its cell as the pattern does in the cells to its left. Long matches
    widen every logo column, which is why ``align`` is off by default. The
    HTML page leaves logos unaligned.

    Needs the ``plot`` extra, and the ``h5py`` extra to read a file.

    Parameters
    ----------
    modisco
        A TF-MoDISco results file, the patterns as motifs, or the result of
        ``contribution_motifs()``.
    matches
        Matches of the patterns to known motifs, from
        [`name_motifs()`][universalmotif.name_motifs] or from
        ``compare_motifs()`` with ``compare_to``.
    database
        The known motifs, as a list or a dict keyed by name, to draw the
        matches with, and to name the patterns by when no ``matches`` are
        given.
    names
        Labels for the patterns, as a dict or a function of the full name.
    patterns
        Which patterns to draw.

        - ``None`` (the default): all of them.
        - An integer ``n``: the first ``n`` of each group.
        - A pair ``(n_pos, n_neg)``: the first ``n_pos`` positive and
          ``n_neg`` negative patterns.
        - A list of labels or names: the patterns named.
    columns
        The columns to draw after the label, from ``"seqlets"``,
        ``"hits"``, ``"CWM"``, ``"hCWM"`` and ``"PPM"``. For a file or the
        result of ``contribution_motifs()`` they default to ``seqlets``,
        ``CWM`` and ``PPM``. Otherwise they default to the seqlets and each
        type given. ``hits`` counts the hits of a pattern's motifs in all
        the regions, and needs the result of ``contribution_motifs()``.
    n_matches
        The number of matches to draw for each pattern.
    qvalue
        The largest q-value of a match, when the patterns are named by
        ``database``.
    align
        Place the logos of each row in a shared frame of positions.
    use_type
        How the PPMs and matches are drawn, as ``"ICM"`` for information
        content or ``"PPM"`` for probabilities. It also heads their column.
    trim_threshold
        Passed to ``read_modisco()`` when reading a file.
    text_size
        The font size of the labels.
    figsize
        The figure size, by default worked out from the number of patterns
        and the widest frame of positions.
    html
        A path to also write the report to as an HTML page.
    title
        The title of the HTML page.
    nthreads
        Threads for naming the patterns, with 0 for all available; by
        default the package setting from
        [`set_nthreads()`][universalmotif.set_nthreads].

    Returns
    -------
    matplotlib.figure.Figure
        The figure, whose single axes is labelled ``"modisco"``.

    References
    ----------
    - Shrikumar A, Tian K, Avsec Ž, Shcherbina A, Banerjee A, Sharmin M, Nair
      S, Kundaje A (2018). Technical note on transcription factor motif
      discovery from importance scores (TF-MoDISco) version 0.5.6.5.
      arXiv:1811.00416. <https://doi.org/10.48550/arXiv.1811.00416>

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
    ['Pattern', 'Seqlets', 'CWM', 'ICM', 'Matches']
    >>> plt.close(fig)

    A results file is read directly and named by a database, here keeping
    the three most common positive patterns and the most common negative
    one, and also writing the report as an HTML page:

    >>> fig = um.plot_modisco("modisco_results.h5", database=known, patterns=(3, 1),
    ...                       html="report.html")  # doctest: +SKIP
    """
    pats, columns, shown, n_cols = _prepare(modisco, matches, database, names, patterns, columns, n_matches,
                                            qvalue, use_type, trim_threshold, align, nthreads)
    if html is not None:
        pathlib.Path(html).write_text(_html_page(modisco, pats, columns, shown, n_cols, use_type, title))
    plt = _pyplot()
    fig = plt.figure()
    small = 0.8 * text_size
    labels = [p.label for p in pats]
    counted = [c for c in _COUNTS if c in columns]
    numbers = {c: ["" if v is None else f"{v:,}" for v in (_count(p, c) for p in pats)] for c in counted}
    head = _headings(use_type, n_cols)
    lines = [_match_lines(m[0], m[1], m[3]) for p in pats for m in p.matches]
    has_db = database is not None

    gap, pad = 0.15, 0.1
    widths = {
        "label": max(_text_width(fig, labels, text_size, "bold"), _text_width(fig, [head["label"]], small,
                                                                             "bold")),
        **{c: max(_text_width(fig, numbers[c], text_size), _text_width(fig, [head[c]], small, "bold"))
           for c in counted},
    }
    match_text = max([_text_width(fig, [ls[0]], small) for ls in lines]
                     + [_text_width(fig, ls[1:], 0.75 * text_size) for ls in lines]
                     + [_text_width(fig, ["no match", head["match"]], small)], default=0.0)
    # With align, every logo column is as wide as the widest row's frame.
    # Otherwise it is as wide as its own widest logo.
    frame = max(p.span for p in pats) or 1
    units = [frame if align else max(p.motifs[t].width for p in pats) for t in shown]
    match_units = 0
    if has_db and n_cols:
        ws = [m.width for p in pats for _, m, *_ in p.matches if m is not None]
        match_units = frame if align else max(ws, default=4)
    total_units = sum(units) + match_units
    has_logos = total_units > 0
    texts_w = (2 * pad + widths["label"] + gap + sum(widths[c] + gap for c in counted)
               + (match_text + gap if n_cols else 0))
    n_logo_cols = len(shown) + (1 if has_db and n_cols else 0)
    n_heads = sum(1 for g in dict.fromkeys(p.group for p in pats) if _GROUPS[g])
    head_h, group_h = 0.3, 0.32
    # A row holds its matches one above another, each in a band of its own.
    band = 0.32
    if figsize is None:
        unit = 0.1 if not has_logos else min(0.1, max(0.05, 10.0 / total_units))
        row_h = max(0.55, n_cols * band + 0.08)
        width = texts_w + total_units * unit + n_logo_cols * gap
        height = 2 * pad + head_h + n_heads * group_h + len(pats) * row_h
    else:
        width, height = figsize
        row_h = (height - 2 * pad - head_h - n_heads * group_h) / len(pats)
        unit = (width - texts_w - n_logo_cols * gap) / total_units if has_logos else 0.1
        if row_h <= 0 or unit <= 0:
            raise ValueError("figsize is too small for the labels")
        band = (row_h - 0.08) / n_cols if n_cols else band
    fig.set_size_inches(width, height)
    ax = fig.add_axes((0, 0, 1, 1), label="modisco")
    ax.set_axis_off()

    # Left edges of the columns, in inches.
    x = pad
    left = {"label": x}
    x += widths["label"] + gap
    for c in counted:
        left[c] = x
        x += widths[c] + gap
    for t, u in zip(shown, units):
        left[t] = x
        x += u * unit + gap
    match_logo_x = match_text_x = None
    if n_cols:
        # Each match's name comes first, with its logo after it.
        match_text_x = x
        x += match_text + gap
        if has_db:
            match_logo_x = x
            x += match_units * unit + gap
    right = x - gap

    def heading(s, x0, y, ha="left"):
        ax.text(x0, y, s, ha=ha, va="center", fontsize=small, fontweight="bold", color="#333333")

    y = height - pad - head_h / 2
    heading(head["label"], left["label"], y)
    for c in counted:
        heading(head[c], left[c] + widths[c], y, ha="right")
    for t, u in zip(shown, units):
        heading(head[t], left[t] + u * unit / 2, y, ha="center")
    if n_cols:
        heading(head["match"], match_text_x, y)
    y = height - pad - head_h
    ax.plot([pad, right], [y, y], color="#9A9A9A", linewidth=0.8, solid_capstyle="butt")

    def centred(u, w, span):
        # Logos sit in the middle of their column, and aligned rows keep
        # their shared frame together.
        return (u - (span if align else w)) / 2 * unit

    logo_h = 0.76 * min(row_h, 0.55)
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
        for c in counted:
            ax.text(left[c] + widths[c], mid, numbers[c][i], ha="right", va="center", fontsize=text_size)
        for t, u in zip(shown, units):
            draw_logo(ax, p.motifs[t], x=left[t] + centred(u, p.motifs[t].width, p.span) + p.placements[0] * unit,
                      y=mid - logo_h / 2,
                      width=p.motifs[t].width * unit, height=logo_h,
                      use_type="CWM" if t != "PPM" else use_type)
        if n_cols and not p.matches:
            ax.text(match_text_x, mid, "no match", ha="left",
                    va="center", fontsize=small, color=_GREY)
        # The matches are stacked in bands centred on the row.
        top_band = mid + band * len(p.matches) / 2
        k_logo = 1
        for k, (target, m, _, q, _) in enumerate(p.matches):
            band_mid = top_band - band * (k + 0.5)
            if m is not None and match_logo_x is not None:
                # Unaligned logos sit against their names.
                shift = centred(match_units, m.width, p.span) + p.placements[k_logo] * unit if align else 0
                draw_logo(ax, m, x=match_logo_x + shift,
                          y=band_mid - 0.4 * band, width=m.width * unit, height=0.8 * band, use_type=use_type)
                k_logo += 1
            ls = _match_lines(target, m, q)
            sizes = [small] + [0.75 * text_size] * (len(ls) - 1)
            step = 1.2 * max(sizes) / 72
            top = band_mid + step * (len(ls) - 1) / 2
            for j, (s_, size) in enumerate(zip(ls, sizes)):
                ax.text(match_text_x, top - j * step, s_, ha="left", va="center", fontsize=size,
                        color="black" if j == 0 else _GREY)
        y -= row_h
        if i + 1 < len(pats) and pats[i + 1].group == p.group:
            ax.plot([pad, right], [y, y], color=_RULE, linewidth=0.5, solid_capstyle="butt")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    return fig


def _headings(use_type: str, n_cols: int) -> dict[str, str]:
    # The PPM column is headed by what it shows: information content (ICM) or probabilities (PPM).
    head = {"label": "Pattern", "seqlets": "Seqlets", "sources": "Sources", "hits": "Hits", "CWM": "CWM",
            "hCWM": "hCWM", "PPM": use_type}
    head["match"] = "Best match" if n_cols == 1 else "Matches"
    return head


_IUPAC_COMPLEMENT = str.maketrans("ACGTURYKMSWBDHVNacgturykmswbdhvn", "TGCAAYRMKSWVHDBNtgcaayrmkswvhdbn")

# How a motif of contribution_motifs() was found, when not in the seqlets as
# they are.
_SOURCES = {"idealized": "found in idealized seqlets", "gapped": "gapped composite"}

_SORT_SCRIPT = """<script>
// Clicking a heading sorts the rows by that column, and clicking it again
// turns the order around. Rows without a value go last either way.
document.querySelectorAll("th.sort").forEach((th) => {
  th.addEventListener("click", () => {
    const col = [...th.parentNode.children].indexOf(th);
    const body = th.closest("table").tBodies[0];
    const dir = th.dataset.dir ? (th.dataset.dir === "asc" ? "desc" : "asc") : th.dataset.first;
    th.parentNode.querySelectorAll("th").forEach((h) => delete h.dataset.dir);
    th.dataset.dir = dir;
    const key = (row) => {
      const v = row.children[col].dataset.v;
      return v === "" || v === undefined ? null : Number(v);
    };
    const rows = [...body.rows].sort((a, b) => {
      const x = key(a), y = key(b);
      if (x === null || y === null) return (x === null) - (y === null);
      return dir === "asc" ? x - y : y - x;
    });
    rows.forEach((r) => body.appendChild(r));
  });
});
// Each row's button shows its logos and consensus as reverse complements.
document.querySelectorAll("button.rc").forEach((b) => {
  b.addEventListener("click", () => {
    const on = b.closest("tr").classList.toggle("flip");
    b.setAttribute("aria-pressed", on);
  });
});
</script>"""

_HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #ffffff; --fg: #000000; --muted: #666666; --rule: #e3e3e3; --head: #ffffff; color-scheme: light; }
body { background: var(--bg); color: var(--fg); font: 13px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; }
main { max-width: 1600px; margin: 0 auto; padding: 20px 16px 48px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th { position: sticky; top: 0; background: var(--head); font-weight: 600; color: var(--muted); text-align: left;
     padding: 6px 10px 6px 0; border-bottom: 1px solid var(--rule); white-space: nowrap; z-index: 1; }
th.n { text-align: right; }
td { padding: 4px 10px 4px 0; border-bottom: 1px solid var(--rule); vertical-align: middle; }
td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td svg { display: block; max-width: none; }
th.sort { cursor: pointer; user-select: none; }
th.sort::after { content: " \\2195"; color: var(--rule); }
th.sort[data-dir="asc"]::after { content: " \\2191"; color: var(--fg); }
th.sort[data-dir="desc"]::after { content: " \\2193"; color: var(--fg); }
.mg { display: grid; grid-template-columns: max-content max-content; gap: 2px 8px; align-items: center; }
.mg > span { white-space: nowrap; }
.m { color: var(--muted); }
code { font-family: ui-monospace, Menlo, monospace; font-size: 0.92em; }
details summary { cursor: pointer; color: var(--muted); }
.mem { display: flex; align-items: center; gap: 8px; white-space: nowrap; margin: 2px 0; }
.rv { display: none; }
tr.flip .fw { display: none; }
tr.flip .rv { display: inline-block; }
button.rc { font: inherit; font-size: 11px; color: var(--muted); background: var(--bg); border: 1px solid var(--rule);
            border-radius: 4px; padding: 0 5px; margin-left: 6px; cursor: pointer; vertical-align: 1px; }
button.rc[aria-pressed="true"] { color: var(--fg); border-color: var(--fg); }
</style></head><body><main>
"""


def _html_page(modisco, pats: list[_Pattern], columns: list[str], shown: list[str], n_cols: int, use_type: str,
               title: str | None) -> str:
    """The report as one self-contained HTML page, with the same columns as
    the figure and one more that lists the CWMs making up each pattern."""
    from ._combine import CombinedPatterns
    from ._contribution_motifs import ContributionMotifs
    from ._display import logo_svg

    if title is None:
        title = ("Motifs from contribution scores" if isinstance(modisco, ContributionMotifs)
                 else "Combined patterns" if isinstance(modisco, CombinedPatterns)
                 else "TF-MoDISco patterns" if isinstance(modisco, (str, os.PathLike)) else "Patterns")
    esc = _html.escape
    head = _headings(use_type, n_cols)

    def both(m, **kw) -> str:
        # A logo as drawn and as its reverse complement, of which a row shows one.
        return (f'<span class="fw">{logo_svg(m, **kw)}</span>'
                f'<span class="rv">{logo_svg(motif_rc(m), **kw)}</span>')

    def both_text(text: str) -> str:
        return (f'<code class="fw">{esc(text)}</code>'
                f'<code class="rv">{esc(text.translate(_IUPAC_COMPLEMENT)[::-1])}</code>')

    counted = [c for c in _COUNTS if c in columns]
    with_parts = any(p.members for p in pats)
    with_flags = any(p.flags is not None for p in pats)

    # Columns that sort, and the way each sorts first: counts from the most,
    # q-values from the least, and the rest from the top.
    cols = [("Pattern", "asc"), ("Sign", "asc"), *[(head[c], "desc") for c in counted],
            *[(head[t], None) for t in shown], *([(head["match"], "asc")] if n_cols else []),
            *([("CWMs", "desc")] if with_parts else []), *([("Flags", "desc")] if with_flags else [])]
    out = [_HTML_HEAD.replace("__TITLE__", esc(title))]
    heads = []
    for name, first in cols:
        cls = " ".join(x for x in ("sort" if first else "", "n" if name in ("Seqlets", "Sources", "Hits") else "") if x)
        attrs = (f' class="{cls}"' if cls else "") + (f' data-first="{first}"' if first else "")
        heads.append(f"<th{attrs}>{esc(name)}</th>")
    out.append('<div class="wrap"><table><thead><tr>' + "".join(heads) + "</tr></thead><tbody>")
    for i, p in enumerate(pats):
        sign = {"pos": "positive", "neg": "negative"}.get(p.group, "")
        cells = [f'<td data-v="{i}"><b>{esc(p.label)}</b>'
                 + '<button class="rc" type="button" aria-pressed="false" title="Reverse complement">RC</button>'
                 + (f"<br>{both_text(p.consensus)}" if p.consensus else "") + "</td>",
                 f'<td data-v="{"pos neg".find(p.group or "x")}">{sign}</td>']
        for c in counted:
            v = _count(p, c)
            text = "" if v is None else f"{v:,}"
            if c == "seqlets" and p.share is not None:
                text += f'<br><span class="m">{100 * p.share:.1f}%</span>'
            cells.append(f'<td class="n" data-v="{"" if v is None else v}">{text}</td>')
        for t in shown:
            logo = both(p.motifs[t], use_type="CWM" if t != "PPM" else use_type, height=34, position_width=8)
            cells.append(f"<td>{logo}</td>")
        if n_cols:
            if p.matches:
                lines = []
                for target, m, _, q, _ in p.matches:
                    ls = _match_lines(target, m, q)
                    label = f"<span><b>{esc(ls[0])}</b>" + "".join(f' <span class="m">{esc(x)}</span>' for x in ls[1:])
                    logo = "" if m is None else both(m, use_type=use_type, height=18, position_width=6)
                    lines.append(label + "</span>" + (logo or "<span></span>"))
                best = min((q for *_, q, _ in p.matches if q is not None), default="")
                cells.append(f'<td data-v="{best}"><div class="mg">{"".join(lines)}</div></td>')
            else:
                cells.append('<td class="m" data-v="">no match</td>')
        if with_parts:
            if p.members:
                inner = "".join(
                    f'<div class="mem"><span><b>{esc(r["name"])}</b>'
                    + (f" {both_text(r['consensus'])}" if r.get("consensus") else "")
                    + f' <span class="m">{r["seqlets"]:,} seqlets'
                    + (f", {_SOURCES[r['source']]}" if r.get("source") in _SOURCES else "") + "</span></span>"
                    + f'{both(r["cwm"], height=24, position_width=6)}</div>' for r in p.members)
                n = len(p.members)
                cells.append(f'<td data-v="{n}"><details><summary>{n} CWM{"s" if n != 1 else ""}</summary>'
                             f"{inner}</details></td>")
            else:
                cells.append('<td class="m" data-v="0">none</td>')
        if with_flags:
            flags = p.flags or []
            cells.append(f'<td data-v="{len(flags)}">{esc(", ".join(flags))}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div></main>" + _SORT_SCRIPT + "</body></html>")
    return "\n".join(out)


def _report(modisco, path, database, n_matches, qvalue, title, nthreads) -> str:
    """The HTML page of plot_modisco(), without drawing the figure."""
    pats, columns, shown, n_cols = _prepare(modisco, None, database, None, None, None, n_matches, qvalue, "ICM",
                                            0.3, False, nthreads)
    page = _html_page(modisco, pats, columns, shown, n_cols, "ICM", title)
    if path is not None:
        pathlib.Path(path).write_text(page)
    return page


__all__ = ["plot_modisco"]
