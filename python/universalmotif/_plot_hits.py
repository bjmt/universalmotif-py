"""Motif hits drawn along sequences and genomic regions."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ._analysis import _make_unique
from ._core import Motif
from ._plot import _fit_legends, _pyplot, draw_logo

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Each lane is one unit tall, hits fill most of it, and tracks are kept apart
# by a gap. Sizes in inches set the height of a figure made here.
_HIT_HEIGHT = 0.8
_TRACK_GAP = 1.2
_LANE_INCHES = 0.17


def _motif_colours(names: list[str], places: Iterable[int]) -> dict[str, str]:
    """Colours for motifs by their places: tab10 for the first ten, and
    the light half of tab20 after them, so that neighbouring motifs never
    get a dark and a light shade of the same hue."""
    import matplotlib

    tab20 = matplotlib.colormaps["tab20"].colors
    palette = [matplotlib.colors.to_hex(c) for c in tab20[0::2] + tab20[1::2]]
    return {name: palette[int(i) % len(palette)] for name, i in zip(names, places)}


def _lengths(sequences) -> dict[str, int]:
    """Sequence lengths from a mapping of lengths or of sequences, a list of
    sequences named as scan_sequences() names them, or anything with a
    ``lengths`` mapping such as PackedSequences and Genome."""
    lengths = getattr(sequences, "lengths", None)
    if isinstance(lengths, Mapping):
        return {str(k): int(v) for k, v in lengths.items()}
    if isinstance(sequences, str):
        return {"seq1": len(sequences)}
    if isinstance(sequences, Mapping):
        return {str(k): v if isinstance(v, (int, np.integer)) else len(v) for k, v in sequences.items()}
    return {f"seq{i + 1}": v if isinstance(v, (int, np.integer)) else len(v)
            for i, v in enumerate(sequences)}


def _lanes(starts: np.ndarray, ends: np.ndarray, gap: float) -> tuple[np.ndarray, int]:
    """Put intervals in lanes so that none in a lane overlap or come within
    ``gap`` of each other, each going in the lowest lane free at its start.
    Returns the lane of each and the number of lanes."""
    lane = np.zeros(len(starts), dtype=np.int64)
    busy: list[tuple[float, int]] = []
    free: list[int] = []
    n = 0
    for i in np.lexsort((ends, starts)):
        while busy and busy[0][0] + gap <= starts[i]:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if free:
            k = heapq.heappop(free)
        else:
            k, n = n, n + 1
        lane[i] = k
        heapq.heappush(busy, (float(ends[i]), k))
    return lane, n


def _values(hits: pl.DataFrame, col: str, what: str) -> tuple[np.ndarray, str]:
    """A numeric column scaled from 0 at its smallest value to 1 at its
    largest, with P-values and the like taken as -log10. Returns the scaled
    values and the legend title."""
    if col not in hits.columns:
        raise ValueError(f"hits has no {col!r} column to {what} by")
    if not hits[col].dtype.is_numeric():
        raise ValueError(f"{what}_by must name a numeric column")
    v = hits[col].cast(pl.Float64).to_numpy().copy()
    title = col
    if col.lower().startswith(("pval", "qval", "eval")):
        v = -np.log10(np.clip(v, 1e-300, None))
        title = f"-log10({col})"
    return v, title


def _scaled(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = v[np.isfinite(v)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    frac = np.ones_like(v) if hi <= lo else (v - lo) / (hi - lo)
    return np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0), lo, hi


class _Key:
    def __init__(self, colour: str, motif: Motif | None):
        self.colour, self.motif = colour, motif


class _KeyHandler:
    """Legend keys drawn as a swatch of the motif's colour followed by its
    logo, when it has one."""

    def __init__(self, logo_width: float, widest: int):
        self.logo_width, self.widest = logo_width, widest

    def legend_artist(self, legend, key, fontsize, handlebox):
        from matplotlib.patches import Rectangle

        w, h = handlebox.width, handlebox.height
        x0, y0 = -handlebox.xdescent, -handlebox.ydescent
        sw = 1.2 * fontsize if key.motif is not None else w
        sh = min(h, 0.7 * fontsize)
        swatch = Rectangle((x0, y0 + (h - sh) / 2), sw, sh, facecolor=key.colour, edgecolor="none",
                           transform=handlebox.get_transform())
        handlebox.add_artist(swatch)
        if key.motif is not None:
            ax = legend.axes
            letters = draw_logo(ax, key.motif, x=x0 + sw + 0.4 * fontsize, y=y0,
                                width=self.logo_width * fontsize * key.motif.width / self.widest,
                                height=h, transform=handlebox.get_transform())
            # The letters were added to the axes, and clipped to them, by
            # draw_logo(), so they are moved to the legend unclipped.
            letters.remove()
            letters.set_clip_path(None)
            letters.set_clip_on(False)
            handlebox.add_artist(letters)
        return swatch


def plot_hits(
    hits: pl.DataFrame,
    sequences=None,
    ranges=None,
    names: Sequence[str] | None = None,
    max_tracks: int | None = 25,
    motifs: Iterable[Motif | str] | None = None,
    colours: Mapping[str, str] | None = None,
    shade_by: str | None = None,
    size_by: str | None = None,
    labels: bool = True,
    legend: bool = True,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw motif hits along the sequences they were found in.

    Each sequence is a track, drawn as a thin line, with each hit from
    [`scan_sequences()`][universalmotif.scan_sequences] drawn on it from its
    start to its end as an arrow pointing along its strand, or as a block
    when it has none. Hits that overlap are stacked in lanes below the line
    so that none hide another, and are coloured by motif.

    Hits moved to the genome with [`genomic_hits()`][universalmotif.genomic_hits]
    are drawn in genomic coordinates. With ``ranges``, each range is a track
    showing the hits that overlap it, with the x-axis in genomic positions
    for a single range and in positions from the start of each range for
    several. Without ``ranges``, each chromosome is a track.

    Parameters
    ----------
    hits
        Hits with ``motif``, ``start`` and ``end`` columns, positions counted
        from 0 with the end left out, and a ``sequence`` or ``chrom``
        column. A ``strand`` column sets the direction of the arrows.
    sequences
        The sequences scanned, as a list, a mapping from names to sequences,
        [`PackedSequences`][universalmotif.PackedSequences] or a
        [`Genome`][universalmotif.Genome], or a mapping from names to
        lengths. Every sequence is then drawn at its full length in the
        order given, including those without hits. Otherwise the sequences
        with hits are drawn up to their last hit. It is not used with
        ``ranges``.
    ranges
        Genomic ranges to draw, as a range table or (chrom, start, end)
        tuples, for hits with a ``chrom`` column. A ``name`` column in the
        table labels the tracks, which are otherwise labelled
        ``chrom:start-end``.
    names, max_tracks
        Draw only the tracks with these names, in this order, and at most
        ``max_tracks`` of them, or all of them with ``None``. A note above
        the plot says when tracks were left out.
    motifs
        Motifs to draw, by name or as the motifs scanned, which sets their
        order in the legend and adds their logos to it. Duplicated names are
        made unique as ``scan_sequences()`` makes them.
    colours
        A mapping from motif names to colours. Motifs default to the tab10
        palette in the order they were scanned or given in ``motifs``,
        followed by the lighter colours of tab20 when there are more than
        ten.
    shade_by, size_by
        A numeric column to set the opacity or the height of the hits by,
        from the smallest value to the largest, such as ``"score"`` or
        ``"pvalue"``. P-values are taken as -log10, so that stronger hits
        are always darker and taller.
    labels
        Name the motif on each hit long enough to hold it.
    legend
        Draw a legend of the motifs beside the plot.
    text_size
        Size of the track names and the legend; labels on hits are a little
        smaller.
    ax
        Draw on these axes, such as a panel of a larger figure, instead of
        in a new figure. ``figsize`` is then ignored.

    Returns
    -------
    matplotlib.figure.Figure | matplotlib.axes.Axes
        A matplotlib figure, or the axes drawn on when ``ax`` is given.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import universalmotif as um
    >>> motifs = [um.example_motifs()[i] for i in (3, 5)]
    >>> targets = um.example_sequences("targets")
    >>> hits = um.scan_sequences(motifs, targets, nthreads=1)
    >>> fig = um.plot_hits(hits, targets, names=["target_015", "target_019", "target_040"])
    >>> [t.get_text() for t in fig.axes[0].get_yticklabels()]
    ['target_015', 'target_019', 'target_040']
    >>> fig = um.plot_hits(hits, targets, motifs=motifs, shade_by="pvalue")
    >>> plt.close("all")
    """
    for col in ("motif", "start", "end"):
        if col not in hits.columns:
            raise ValueError(f"hits must have a {col!r} column")
    if max_tracks is not None and max_tracks < 1:
        raise ValueError("max_tracks must be at least 1, or None for every track")
    if names is not None and isinstance(names, str):
        names = [names]
    start = hits["start"].cast(pl.Int64)
    end = hits["end"].cast(pl.Int64)
    if start.null_count() or end.null_count():
        raise ValueError("start and end cannot contain nulls")
    if (start > end).any():
        raise ValueError("all hits must have start <= end")

    # The motifs drawn, their order and their logos.
    logos: dict[str, Motif] = {}
    if motifs is not None:
        motifs = [motifs] if isinstance(motifs, (Motif, str)) else list(motifs)
        if not all(isinstance(m, (Motif, str)) for m in motifs):
            raise ValueError("motifs must be motif names or Motif objects")
        given = [m.name if isinstance(m, Motif) else m for m in motifs]
        if all(isinstance(m, Motif) for m in motifs):
            given = _make_unique(given)
        logos = {n: m for n, m in zip(given, motifs) if isinstance(m, Motif)}
        motif_names = list(dict.fromkeys(given))
        palette = _motif_colours(motif_names, range(len(motif_names)))
        hits = hits.filter(pl.col("motif").is_in(motif_names))
    else:
        # Colours follow the motifs' places in the scan, so that they stay the
        # same whichever tracks are drawn.
        if "motif_i" in hits.columns:
            pairs = hits.select(pl.col("motif").cast(pl.String), pl.col("motif_i").cast(pl.Int64))
            pairs = pairs.unique("motif", keep="first", maintain_order=True).sort("motif_i", maintain_order=True)
            motif_names, places = pairs["motif"].to_list(), pairs["motif_i"].to_list()
        else:
            motif_names = hits["motif"].cast(pl.String).unique(maintain_order=True).to_list()
            places = range(len(motif_names))
        palette = _motif_colours(motif_names, places)
    if colours is not None:
        missing = [m for m in motif_names if m not in colours]
        if missing:
            raise ValueError(f"colours has no colours for: {', '.join(missing)}")
        palette = {m: colours[m] for m in motif_names}

    tracks, offsets, bounds, hit_track, hits, xlabel = _tracks(hits, sequences, ranges)
    if names is not None:
        index = {t: i for i, t in enumerate(tracks)}
        unknown = [n for n in names if n not in index]
        if unknown:
            raise ValueError(f"no sequences or ranges named: {', '.join(map(str, unknown))}")
        chosen = [index[n] for n in names]
    else:
        chosen = list(range(len(tracks)))
    total = len(chosen)
    if max_tracks is not None:
        chosen = chosen[:max_tracks]
    if not chosen:
        raise ValueError("hits has no rows to plot; give sequences to draw tracks without hits")
    remap = np.full(len(tracks), -1, dtype=np.int64)
    remap[chosen] = np.arange(len(chosen))
    keep = remap[hit_track] >= 0 if len(hit_track) else np.zeros(0, dtype=bool)
    hits = hits.filter(pl.Series(keep))
    track = remap[hit_track[keep]]
    tracks = [tracks[i] for i in chosen]
    offsets = offsets[chosen]
    bounds = bounds[chosen]
    ntracks = len(tracks)
    if motifs is None:
        drawn = set(hits["motif"].cast(pl.String).to_list())
        motif_names = [m for m in motif_names if m in drawn]

    # Hits running off the end of a range are cut at its ends.
    s = np.maximum(hits["start"].cast(pl.Float64).to_numpy() - offsets[track], bounds[track, 0])
    e = np.minimum(hits["end"].cast(pl.Float64).to_numpy() - offsets[track], bounds[track, 1])
    strand = (hits["strand"].cast(pl.String).fill_null("*").to_numpy() if "strand" in hits.columns
              else np.full(hits.height, "*"))
    motif_col = hits["motif"].cast(pl.String).to_list()
    alpha = np.ones(hits.height)
    height = np.full(hits.height, _HIT_HEIGHT)
    value_legends: list[tuple[str, str, float, float]] = []
    if shade_by is not None:
        v, title = _values(hits, shade_by, "shade")
        frac, lo, hi = _scaled(v)
        alpha = 0.25 + 0.75 * frac
        value_legends.append(("shade", title, lo, hi))
    if size_by is not None:
        v, title = _values(hits, size_by, "size")
        frac, lo, hi = _scaled(v)
        height = _HIT_HEIGHT * (0.35 + 0.65 * frac)
        if value_legends and value_legends[0][1] == title:
            value_legends[0] = ("both", title, lo, hi)
        else:
            value_legends.append(("size", title, lo, hi))

    plt = _pyplot()
    xmin, xmax = float(bounds[:, 0].min()), float(bounds[:, 1].max())
    if xmax <= xmin:
        xmax = xmin + 1
    xpad = 0.01 * (xmax - xmin)
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=figsize or (8.0, 3.0))
    else:
        fig = ax.figure
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_yticks(range(ntracks), [str(t) for t in tracks], fontsize=text_size)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", labelsize=text_size)
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(xlabel, fontsize=text_size)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    note = None
    if total > len(chosen):
        what = "ranges" if ranges is not None else "sequences"
        note = ax.text(0.0, 1.02, f"Showing the first {len(chosen)} of {total} {what}; "
                       "set max_tracks to draw more", transform=ax.transAxes, ha="left", va="bottom",
                       fontsize=0.85 * text_size, color="0.4")
    if own_figure:
        fig.tight_layout()

    # Arrow heads, the narrowest hit drawn and the gap between hits in a lane
    # are set in points, and so depend on the width of the axes.
    width_pts = ax.get_position().width * fig.get_figwidth() * 72
    per_pt = (xmax - xmin + 2 * xpad) / width_pts
    min_width = 3.0 * per_pt
    centre = (s + e) / 2
    narrow = e - s < min_width
    ds = np.where(narrow, centre - min_width / 2, s)
    de = np.where(narrow, centre + min_width / 2, e)
    lane = np.zeros(len(ds), dtype=np.int64)
    nlanes = np.ones(ntracks, dtype=np.int64)
    if len(ds):
        order = np.argsort(track, kind="stable")
        cuts = np.flatnonzero(np.diff(track[order])) + 1
        for idx in np.split(order, cuts):
            lane[idx], n = _lanes(ds[idx], de[idx], 2.0 * per_pt)
            nlanes[track[idx[0]]] = max(n, 1)
    top = np.concatenate([[0.0], np.cumsum(nlanes + _TRACK_GAP)[:-1]])
    units = float(top[-1] + nlanes[-1])
    if own_figure and figsize is None:
        ax_h = ax.get_position().height * fig.get_figheight()
        margins = fig.get_figheight() - ax_h
        # Very many lanes are squeezed rather than making the figure taller still.
        fig.set_size_inches(fig.get_figwidth(), min(max(1.6, margins + (units + 0.2) * _LANE_INCHES), 24.0))
    ax.set_ylim(units - 0.4, -0.6)
    ax.set_yticks(top, [str(t) for t in tracks], fontsize=text_size)
    if own_figure:
        fig.tight_layout()

    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_rgba

    segs = np.stack([np.column_stack([bounds[:, 0], top]), np.column_stack([bounds[:, 1], top])], axis=1)
    ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.8, zorder=1))

    y = top[track] + lane if len(track) else np.zeros(0)
    pts_per_unit = ax.get_position().height * fig.get_figheight() * 72 / (units + 0.2)
    head = np.minimum(0.5 * (de - ds), 0.45 * height * pts_per_unit * per_pt)
    fwd, rev = strand == "+", strand == "-"
    b, t = y - height / 2, y + height / 2
    body_s = np.where(rev, ds + head, ds)
    body_e = np.where(fwd, de - head, de)
    tip_left = np.where(rev, ds, body_s)
    tip_right = np.where(fwd, de, body_e)
    verts = np.stack([
        np.column_stack([body_s, b]), np.column_stack([body_e, b]), np.column_stack([tip_right, y]),
        np.column_stack([body_e, t]), np.column_stack([body_s, t]), np.column_stack([tip_left, y]),
    ], axis=1)
    rgba = np.array([to_rgba(palette[m]) for m in motif_col]).reshape(-1, 4)
    rgba[:, 3] = alpha
    ax.add_collection(PolyCollection(verts, facecolors=rgba, edgecolors="none", linewidths=0, zorder=2))

    if labels and len(ds):
        fs = min(0.85 * text_size, 0.7 * _HIT_HEIGHT * pts_per_unit)
        room = (body_e - body_s) / per_pt
        need = np.array([0.6 * fs * len(m) + 4 for m in motif_col])
        tall = height * pts_per_unit >= 1.05 * fs
        # Text goes black or white, whichever reads better on the colour as
        # it shows against the white page.
        shown = rgba[:, :3] * alpha[:, None] + (1 - alpha[:, None])
        light = shown @ np.array([0.299, 0.587, 0.114]) > 0.55
        for i in np.flatnonzero((room >= need) & tall & (fs >= 4)):
            ax.text((body_s[i] + body_e[i]) / 2, y[i], motif_col[i], ha="center", va="center_baseline",
                    fontsize=fs, color="black" if light[i] else "white", zorder=3, clip_on=True)

    legends = []
    if legend:
        legends = _draw_legends(ax, motif_names, palette, logos, value_legends, text_size,
                                _HIT_HEIGHT * pts_per_unit)
        if legends:
            _fit_legends(ax, legends, own_figure)
            if own_figure:
                _fit_height(ax, legends + ([note] if note is not None else []))
    return fig if own_figure else ax


def _tracks(hits: pl.DataFrame, sequences, ranges):
    """The tracks to draw: their labels, the position each is drawn from,
    their (start, end) bounds in drawn positions, the track of each hit,
    the hits, and the x-axis label. Hits in several ranges are drawn in
    each."""
    if ranges is not None:
        from ._genomic import _range_table, _ranges_frame

        if "chrom" not in hits.columns:
            raise ValueError("ranges need hits in genomic coordinates with a chrom column, "
                             "such as from genomic_hits()")
        frame = _ranges_frame(ranges)
        table = _range_table(frame, "ranges")
        if table.height == 0:
            raise ValueError("ranges has no rows to plot")
        if isinstance(frame, pl.DataFrame) and "name" in frame.columns:
            labels = frame["name"].cast(pl.String).to_list()
        else:
            labels = [f"{c}:{a}-{b}" for c, a, b in table.select("chrom", "start", "end").iter_rows()]
        if (table["end"] <= table["start"]).any():
            raise ValueError("ranges must have start < end")
        regions = table.select(
            pl.col("chrom").alias("_chrom"), pl.col("start").alias("_rs"), pl.col("end").alias("_re"),
        ).with_row_index("_track")
        joined = (
            hits.with_row_index("_hit")
            .with_columns(pl.col("chrom").cast(pl.String).alias("_chrom"))
            .join(regions, on="_chrom", how="inner")
            .filter((pl.col("start") < pl.col("_re")) & (pl.col("end") > pl.col("_rs")))
            .sort("_track", "_hit")
        )
        # Hits from overlapping ranges come once for each, but are drawn once
        # in each track.
        same = ["_track", "motif", "start", "end"] + (["strand"] if "strand" in hits.columns else [])
        joined = joined.unique(same, keep="first", maintain_order=True)
        starts = table["start"].to_numpy().astype(float)
        ends = table["end"].to_numpy().astype(float)
        if table.height == 1:
            offsets = np.zeros(1)
            xlabel = f"Position on {table['chrom'][0]}"
        else:
            offsets = starts
            xlabel = "Position from the start of each range"
        bounds = np.column_stack([starts - offsets, ends - offsets])
        hit_track = joined["_track"].cast(pl.Int64).to_numpy()
        out = joined.drop("_hit", "_chrom", "_rs", "_re", "_track")
        return labels, offsets, bounds, hit_track, out, xlabel

    key = "chrom" if "chrom" in hits.columns else "sequence"
    if key not in hits.columns:
        raise ValueError("hits must have a 'sequence' or 'chrom' column")
    seq = hits[key].cast(pl.String)
    lengths = _lengths(sequences) if sequences is not None else None
    if lengths is not None:
        tracks = list(lengths)
        if hits.height and not seq.is_in(tracks).any():
            raise ValueError("none of the hits are on the sequences given")
    elif key == "sequence" and "sequence_i" in hits.columns:
        tracks = hits.sort("sequence_i", maintain_order=True)[key].cast(pl.String).unique(
            maintain_order=True).to_list()
    else:
        tracks = seq.unique(maintain_order=True).to_list()
    index = {t: i for i, t in enumerate(tracks)}
    hit_track = seq.replace_strict(list(index), list(index.values()), default=-1,
                                   return_dtype=pl.Int64).to_numpy()
    on = hit_track >= 0
    hits = hits.filter(pl.Series(on))
    hit_track = hit_track[on]
    last = np.zeros(len(tracks))
    if len(hit_track):
        np.maximum.at(last, hit_track, hits["end"].cast(pl.Float64).to_numpy())
    if lengths is not None:
        size = np.array([lengths[t] for t in tracks], dtype=float)
        over = np.flatnonzero(last > size)
        if over.size:
            raise ValueError(f"hits run past the end of {tracks[over[0]]}")
    else:
        size = last
    bounds = np.column_stack([np.zeros(len(tracks)), size])
    return tracks, np.zeros(len(tracks)), bounds, hit_track, hits, "Position"


def _draw_legends(ax, motif_names, palette, logos, value_legends, text_size, hit_pts) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    legends = []
    if motif_names:
        keys = [_Key(palette[m], logos.get(m)) for m in motif_names]
        widest = max((m.width for m in logos.values()), default=1)
        # Logos are about a third as wide per position as they are tall.
        logo_width = 0.75 * widest
        kw: dict[str, Any] = {"handlelength": 1.2}
        if logos:
            kw = {"handlelength": 1.6 + logo_width, "handleheight": 2.2, "labelspacing": 0.3}
        handler = _KeyHandler(logo_width, widest)
        leg = ax.legend(keys, motif_names, handler_map={_Key: handler}, frameon=False,
                        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=text_size,
                        borderaxespad=0.0, **kw)
        ax.add_artist(leg)
        # Added artists are clipped to the axes, which would leave the legend
        # out of tight layouts and bbox_inches="tight".
        leg.set_clip_on(False)
        legends.append(leg)
    for kind, title, lo, hi in value_legends:
        if hi <= lo:
            continue
        breaks = [v for v in MaxNLocator(nbins=3).tick_values(lo, hi) if lo <= v <= hi]
        handles = []
        for v in breaks:
            frac = (v - lo) / (hi - lo)
            size = (0.35 + 0.65 * frac) * hit_pts if kind in ("size", "both") else 0.8 * hit_pts
            a = 0.25 + 0.75 * frac if kind in ("shade", "both") else 1.0
            handles.append(Line2D([], [], linestyle="", marker=">", markersize=min(size, 2 * text_size),
                                  markerfacecolor="0.3", markeredgecolor="none", alpha=a))
        renderer = ax.figure.canvas.get_renderer()
        if legends:
            below = legends[-1].get_window_extent(renderer)
            y = ax.transAxes.inverted().transform((0, below.y0))[1] - 0.04
        else:
            y = 1.0
        leg = ax.legend(handles, [f"{v:g}" for v in breaks], title=title, frameon=False,
                        loc="upper left", bbox_to_anchor=(1.01, y), fontsize=text_size,
                        title_fontsize=text_size, borderaxespad=0.0, handlelength=1.2,
                        alignment="left")
        ax.add_artist(leg)
        # Added artists are clipped to the axes, which would leave the legend
        # out of tight layouts and bbox_inches="tight".
        leg.set_clip_on(False)
        legends.append(leg)
    return legends


def _fit_height(ax: Axes, artists: list) -> None:
    """Make a figure made here taller when legends run below or text runs
    above it, keeping the axes the same size."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in artists]
    below = max(0.0, -min(b.y0 for b in boxes))
    above = max(0.0, max(b.y1 for b in boxes) - fig.bbox.y1)
    if below <= 0 and above <= 0:
        return
    w, h = fig.get_size_inches()
    pos = ax.get_position()
    extra_below = below / fig.dpi + (0.1 if below else 0)
    extra_above = above / fig.dpi + (0.05 if above else 0)
    new_h = h + extra_below + extra_above
    fig.set_size_inches(w, new_h)
    ax.set_position([pos.x0, (pos.y0 * h + extra_below) / new_h, pos.width, pos.height * h / new_h])
