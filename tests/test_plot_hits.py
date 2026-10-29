import itertools

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex, to_rgba
from universalmotif._plot_hits import _lanes, _motif_colours, plot_hits

TAB10 = [to_hex(c) for c in matplotlib.colormaps["tab10"].colors]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _hits(rows, chrom=False):
    """A hit table from (motif, motif_i, sequence, sequence_i, start, end,
    strand, pvalue) tuples."""
    cols = list(zip(*rows)) if rows else [[]] * 8
    df = pl.DataFrame({
        "motif": pl.Series(cols[0], dtype=pl.String),
        "motif_i": pl.Series(cols[1], dtype=pl.UInt32),
        "sequence": pl.Series(cols[2], dtype=pl.String),
        "sequence_i": pl.Series(cols[3], dtype=pl.UInt32),
        "start": pl.Series(cols[4], dtype=pl.Int64),
        "end": pl.Series(cols[5], dtype=pl.Int64),
        "strand": pl.Series(cols[6], dtype=pl.String),
        "score": pl.Series([10.0 - 100 * p for p in cols[7]], dtype=pl.Float64),
        "pvalue": pl.Series(cols[7], dtype=pl.Float64),
    })
    if chrom:
        df = df.with_columns(pl.col("sequence").alias("chrom"))
    return df


ROWS = [
    ("A", 0, "s1", 0, 10, 20, "+", 1e-5),
    ("B", 1, "s1", 0, 15, 25, "-", 1e-4),
    ("A", 0, "s1", 0, 40, 50, "*", 1e-3),
    ("B", 1, "s2", 1, 5, 15, "-", 1e-4),
]
LENGTHS = {"s1": 60, "s2": 40, "s3": 50}


def _parts(ax):
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(lines) == 1 and len(polys) == 1
    return lines[0], polys[0]


def _boxes(poly):
    """Each hit's (left, right, bottom, top, centre y, tip x) from its polygon."""
    out = []
    for path in poly.get_paths():
        v = path.vertices[:6]
        out.append((v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), v[2, 1],
                    v[2, 0] if v[2, 0] > v[1, 0] else (v[5, 0] if v[5, 0] < v[0, 0] else None)))
    return out


def _labels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def test_lanes_put_overlaps_apart():
    s = np.array([0.0, 5.0, 12.0, 30.0, 31.0])
    e = np.array([10.0, 15.0, 20.0, 40.0, 33.0])
    lane, n = _lanes(s, e, 0.0)
    assert n == 2
    assert lane.tolist() == [0, 1, 0, 0, 1]
    # A gap pushes hits that only just clear one another into a new lane.
    lane, n = _lanes(np.array([0.0, 10.5]), np.array([10.0, 20.0]), 1.0)
    assert (lane.tolist(), n) == ([0, 1], 2)
    assert _lanes(np.zeros(0), np.zeros(0), 1.0)[1] == 0


def test_motif_colours_follow_places():
    cols = _motif_colours(["a", "b", "c"], [0, 3, 10])
    tab20 = [to_hex(c) for c in matplotlib.colormaps["tab20"].colors]
    assert cols == {"a": TAB10[0], "b": TAB10[3], "c": tab20[1]}


def test_hits_drawn_where_they_are():
    hits = _hits(ROWS)
    fig = plot_hits(hits, LENGTHS, labels=False)
    assert type(fig).__name__ == "Figure"
    ax = fig.axes[0]
    assert _labels(ax) == ["s1", "s2", "s3"]
    line, poly = _parts(ax)
    # Each track's line runs the length of its sequence.
    segs = line.get_segments()
    assert [(s[0, 0], s[1, 0]) for s in segs] == [(0, 60), (0, 40), (0, 50)]
    np.testing.assert_allclose([s[0, 1] for s in segs], ax.get_yticks())
    boxes = _boxes(poly)
    assert len(boxes) == hits.height
    for (left, right, *_), (s, e) in zip(boxes, zip(hits["start"], hits["end"])):
        assert (left, right) == pytest.approx((s, e))
    # Arrows point along their strands, and unstranded hits are blocks.
    assert boxes[0][5] == pytest.approx(20)
    assert boxes[1][5] == pytest.approx(15)
    assert boxes[2][5] is None
    # Hits sit on their own tracks: s2's hit is centred on s2's line.
    assert boxes[3][4] == pytest.approx(segs[1][0, 1])
    faces = [to_hex(c) for c in poly.get_facecolors()]
    assert faces == [TAB10[0], TAB10[1], TAB10[0], TAB10[1]]


def test_overlapping_hits_go_in_lanes():
    rng = np.random.default_rng(1)
    starts = rng.integers(0, 400, 300)
    rows = [("A" if i % 2 else "B", i % 2, f"s{i % 3}", i % 3, int(s), int(s) + 12,
             "+" if i % 4 else "-", 1e-4) for i, s in enumerate(starts)]
    fig = plot_hits(_hits(rows))
    _, poly = _parts(fig.axes[0])
    boxes = _boxes(poly)
    assert len(boxes) == 300
    by_lane: dict[float, list] = {}
    for left, right, _, _, y, _ in boxes:
        by_lane.setdefault(round(y, 6), []).append((left, right))
    assert len(by_lane) > 3
    for spans in by_lane.values():
        spans.sort()
        for (_, r1), (l2, _) in itertools.pairwise(spans):
            assert l2 > r1
    # Polygons never overlap one another, across lanes either.
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]


def test_tracks_from_sequences_and_lengths():
    hits = _hits(ROWS)
    seqs = {"s1": "A" * 60, "s2": "C" * 40}
    line, _ = _parts(plot_hits(hits, seqs).axes[0])
    assert [s[1, 0] for s in line.get_segments()] == [60, 40]
    # A list of sequences is named as scan_sequences() names it.
    listed = hits.with_columns(pl.col("sequence").replace({"s1": "seq1", "s2": "seq2"}))
    ax = plot_hits(listed, ["A" * 70, "C" * 30]).axes[0]
    assert _labels(ax) == ["seq1", "seq2"]
    # Without lengths, tracks run to their last hit, in sequence_i order.
    swapped = hits.with_columns(pl.col("sequence_i").replace({0: 5}))
    ax = plot_hits(swapped).axes[0]
    assert _labels(ax) == ["s2", "s1"]
    line, _ = _parts(ax)
    assert [s[1, 0] for s in line.get_segments()] == [15, 50]


def test_real_scan():
    motifs = [um.example_motifs()[i] for i in (3, 5)]
    targets = um.example_sequences("targets")
    hits = um.scan_sequences(motifs, targets, nthreads=1)
    names = ["target_015", "target_019", "target_040"]
    fig = plot_hits(hits, targets, names=names)
    ax = fig.axes[0]
    assert _labels(ax) == names
    _, poly = _parts(ax)
    drawn = hits.filter(pl.col("sequence").is_in(names))
    assert len(poly.get_paths()) == drawn.height
    legend = ax.artists[0]
    assert [t.get_text() for t in legend.get_texts()] == drawn["motif"].unique(maintain_order=True).to_list()


def test_names_and_truncation_note():
    hits = _hits(ROWS)
    ax = plot_hits(hits, LENGTHS, names=["s3", "s1"]).axes[0]
    assert _labels(ax) == ["s3", "s1"]
    assert not [t for t in ax.texts if t.get_text().startswith("Showing")]
    _, poly = _parts(ax)
    assert len(poly.get_paths()) == 3

    ax = plot_hits(hits, LENGTHS, max_tracks=2).axes[0]
    assert _labels(ax) == ["s1", "s2"]
    notes = [t.get_text() for t in ax.texts if t.get_text().startswith("Showing")]
    assert notes == ["Showing the first 2 of 3 sequences; set max_tracks to draw more"]
    # The note fits on the figure.
    fig = ax.figure
    box = ax.texts[0].get_window_extent(fig.canvas.get_renderer())
    assert box.y1 <= fig.bbox.y1 + 1

    ax = plot_hits(hits, {f"x{i}": 10 for i in range(40)} | LENGTHS, max_tracks=None).axes[0]
    assert len(_labels(ax)) == 43
    assert not [t for t in ax.texts if t.get_text().startswith("Showing")]
    ax = plot_hits(hits, {f"x{i}": 10 for i in range(40)} | LENGTHS).axes[0]
    assert len(_labels(ax)) == 25


def test_motifs_filter_and_order():
    hits = _hits(ROWS)
    ax = plot_hits(hits, motifs=["B"]).axes[0]
    _, poly = _parts(ax)
    assert len(poly.get_paths()) == 2
    assert {to_hex(c) for c in poly.get_facecolors()} == {TAB10[0]}
    ax = plot_hits(hits, motifs=["B", "A", "C"]).axes[0]
    assert [t.get_text() for t in ax.artists[0].get_texts()] == ["B", "A", "C"]
    # Without motifs, colours follow motif_i, so a subset keeps them.
    ax = plot_hits(hits, names=["s2"]).axes[0]
    _, poly = _parts(ax)
    assert [to_hex(c) for c in poly.get_facecolors()] == [TAB10[1]]
    assert [t.get_text() for t in ax.artists[0].get_texts()] == ["B"]


def test_custom_colours():
    hits = _hits(ROWS)
    ax = plot_hits(hits, colours={"A": "black", "B": "#00ff00"}).axes[0]
    _, poly = _parts(ax)
    assert [to_hex(c) for c in poly.get_facecolors()] == ["#000000", "#00ff00", "#000000", "#00ff00"]
    with pytest.raises(ValueError, match="colours has no colours for: B"):
        plot_hits(hits, colours={"A": "black"})


def test_shade_and_size():
    hits = _hits(ROWS)
    ax = plot_hits(hits, shade_by="pvalue").axes[0]
    _, poly = _parts(ax)
    alpha = poly.get_facecolors()[:, 3]
    # -log10 P-values of 5, 4, 3 and 4.
    np.testing.assert_allclose(alpha, [1.0, 0.625, 0.25, 0.625])
    legends = ax.artists
    assert legends[1].get_title().get_text() == "-log10(pvalue)"

    ax = plot_hits(hits, size_by="score").axes[0]
    _, poly = _parts(ax)
    heights = [b[3] - b[2] for b in _boxes(poly)]
    assert heights[0] == pytest.approx(0.8)
    assert heights[2] == pytest.approx(0.8 * 0.35)
    assert heights[1] == pytest.approx(heights[3])
    assert ax.artists[1].get_title().get_text() == "score"

    # One legend serves both when they use the same column.
    ax = plot_hits(hits, shade_by="score", size_by="score").axes[0]
    assert len(ax.artists) == 2
    with pytest.raises(ValueError, match="hits has no 'qval' column to shade by"):
        plot_hits(hits, shade_by="qval")
    with pytest.raises(ValueError, match="size_by must name a numeric column"):
        plot_hits(hits, size_by="strand")


def test_labels_when_room():
    hits = _hits(ROWS)
    ax = plot_hits(hits, LENGTHS).axes[0]
    texts = [t for t in ax.texts if t.get_text() in ("A", "B")]
    assert len(texts) == 4
    _, poly = _parts(ax)
    for t, (left, right, *_rest) in zip(texts, _boxes(poly)):
        assert left < t.get_position()[0] < right
    # Black text on light colours and white on dark ones.
    light = plot_hits(hits, colours={"A": "#ffff00", "B": "#000080"}).axes[0]
    assert {t.get_text(): t.get_color() for t in light.texts} == {"A": "black", "B": "white"}
    assert not plot_hits(hits, labels=False).axes[0].texts
    # Hits too short for their names are left unlabelled.
    long = hits.with_columns(pl.lit("a-very-long-motif-name").alias("motif"))
    assert not plot_hits(long, {"s1": 5000, "s2": 5000}).axes[0].texts


def test_legend_logos():
    motifs = [um.example_motifs()[i] for i in (3, 5)]
    hits = _hits([("G-box", 0, "s1", 0, 10, 18, "+", 1e-5), ("W-box", 1, "s1", 0, 30, 39, "-", 1e-5)])
    ax = plot_hits(hits, motifs=motifs).axes[0]
    legend = ax.artists[0]
    logos = [c for c in legend.findobj(PolyCollection)]
    assert len(logos) == 2
    assert all(len(c.get_paths()) > 0 for c in logos)
    # The logos are in the legend rather than the axes.
    _, poly = _parts(ax)
    assert len(poly.get_paths()) == 2
    assert not plot_hits(hits, motifs=["G-box", "W-box"]).axes[0].artists[0].findobj(PolyCollection)
    with pytest.raises(ValueError, match="motifs must be motif names or Motif objects"):
        plot_hits(hits, motifs=[1])


def test_draws_on_given_axes():
    _, (a, b) = plt.subplots(2, 1)
    out = plot_hits(_hits(ROWS), LENGTHS, ax=b)
    assert out is b
    assert _labels(b) == ["s1", "s2", "s3"]
    assert not a.collections


def test_empty_hits():
    empty = _hits([])
    ax = plot_hits(empty, LENGTHS).axes[0]
    line, poly = _parts(ax)
    assert len(line.get_segments()) == 3
    assert len(poly.get_paths()) == 0
    assert not ax.artists
    with pytest.raises(ValueError, match="hits has no rows to plot"):
        plot_hits(empty)


def test_genomic_ranges():
    genome = {"chr1": "ACGT" * 100, "chr2": "TTGACC" * 50}
    ranges = [("chr1", 100, 200), ("chr2", 40, 90, "-")]
    hits = pl.DataFrame({
        "motif": ["A", "A", "B", "B"],
        "motif_i": pl.Series([0, 0, 1, 1], dtype=pl.UInt32),
        "sequence": ["chr1:100-200", "chr1:100-200", "chr2:40-90(-)", "chr2:40-90(-)"],
        "sequence_i": pl.Series([0, 0, 1, 1], dtype=pl.UInt32),
        "chrom": ["chr1", "chr1", "chr2", "chr2"],
        "start": [110, 195, 50, 300],
        "end": [120, 205, 58, 308],
        "strand": ["+", "-", "+", "-"],
        "pvalue": [1e-5] * 4,
    })
    assert genome  # the ranges lie within these chromosomes
    ax = plot_hits(hits, ranges=ranges, labels=False).axes[0]
    assert _labels(ax) == ["chr1:100-200", "chr2:40-90"]
    assert ax.get_xlabel() == "Position from the start of each range"
    line, poly = _parts(ax)
    assert [(s[0, 0], s[1, 0]) for s in line.get_segments()] == [(0, 100), (0, 50)]
    boxes = _boxes(poly)
    # The hit outside every range is left out, and the one running off the
    # end of chr1's range is cut at its end.
    assert [(b[0], b[1]) for b in boxes] == [(10, 20), (95, 100), (10, 18)]

    ax = plot_hits(hits, ranges=ranges[:1]).axes[0]
    assert ax.get_xlabel() == "Position on chr1"
    line, poly = _parts(ax)
    assert [(s[0, 0], s[1, 0]) for s in line.get_segments()] == [(100, 200)]
    assert [(b[0], b[1]) for b in _boxes(poly)] == [(110, 120), (195, 200)]
    lo, hi = ax.get_xlim()
    assert lo < 100 and hi > 200

    # A name column labels the tracks, and each range is a track even when
    # they overlap.
    table = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 150], "end": [200, 300],
                          "name": ["peak1", "peak2"]})
    ax = plot_hits(hits, ranges=table).axes[0]
    assert _labels(ax) == ["peak1", "peak2"]
    _, poly = _parts(ax)
    assert len(poly.get_paths()) == 3

    # Without ranges, each chromosome is a track, measured by sequences.
    ax = plot_hits(hits, sequences={"chr1": 400, "chr2": 400}).axes[0]
    assert _labels(ax) == ["chr1", "chr2"]
    line, _ = _parts(ax)
    assert [s[1, 0] for s in line.get_segments()] == [400, 400]


def test_genomic_hits_round_trip():
    genome = {"chr1": "CCCCCTGACTCACCCCCCCCCCCCCCCCCCCC"}
    ranges = [("chr1", 2, 15), ("chr1", 2, 25)]
    peaks = um.extract_ranges(genome, ranges)
    hits = um.genomic_hits(um.scan_sequences(um.example_motifs()[1], peaks, pvalue=1e-3, rc=False))
    ax = plot_hits(hits, ranges=ranges, labels=False).axes[0]
    _, poly = _parts(ax)
    starts = sorted(round(b[0]) for b in _boxes(poly))
    # Both ranges hold the site at chr1:4-13, two positions in.
    assert starts == [2, 2]


def test_errors():
    hits = _hits(ROWS)
    with pytest.raises(ValueError, match="hits must have a 'start' column"):
        plot_hits(hits.drop("start"))
    with pytest.raises(ValueError, match="hits must have a 'sequence' or 'chrom' column"):
        plot_hits(hits.drop("sequence"))
    with pytest.raises(ValueError, match="no sequences or ranges named: s9"):
        plot_hits(hits, names=["s1", "s9"])
    with pytest.raises(ValueError, match="max_tracks must be at least 1"):
        plot_hits(hits, max_tracks=0)
    with pytest.raises(ValueError, match="hits run past the end of s1"):
        plot_hits(hits, {"s1": 30, "s2": 40})
    with pytest.raises(ValueError, match="none of the hits are on the sequences given"):
        plot_hits(hits, ["A" * 100])
    with pytest.raises(ValueError, match="ranges need hits in genomic coordinates"):
        plot_hits(hits, ranges=[("s1", 0, 10)])
    with pytest.raises(ValueError, match="ranges must have start < end"):
        plot_hits(_hits(ROWS, chrom=True), ranges=[("s1", 10, 10)])
    with pytest.raises(ValueError, match="start and end cannot contain nulls"):
        plot_hits(hits.with_columns(pl.lit(None, dtype=pl.Int64).alias("end")))
    with pytest.raises(ValueError, match="all hits must have start <= end"):
        plot_hits(hits.with_columns(pl.col("start") + 20))


def test_unstranded_without_strand_column():
    ax = plot_hits(_hits(ROWS).drop("strand")).axes[0]
    _, poly = _parts(ax)
    assert all(b[5] is None for b in _boxes(poly))


def test_rgba_matches_palette():
    ax = plot_hits(_hits(ROWS)).axes[0]
    _, poly = _parts(ax)
    np.testing.assert_allclose(poly.get_facecolors()[0], to_rgba(TAB10[0]))


def test_legend_kept_by_tight_layouts():
    ms = um.example_motifs()
    hits = um.scan_sequences(ms[3:5], um.example_sequences("targets"))
    fig = um.plot_hits(hits, max_tracks=4, motifs=ms[3:5], shade_by="pvalue")
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    legends = [c for c in fig.axes[0].get_children() if type(c).__name__ == "Legend"]
    assert legends
    for leg in legends:
        # The tight box is in inches, the legend's in pixels.
        assert leg.get_window_extent(renderer).x1 / fig.dpi <= tight.x1 + 1e-6
    plt.close(fig)


def test_size_legend_with_few_lanes():
    # With few lanes the hits are tall, and the legend's markers once grew
    # past their rows and the legend's title ran into the last x tick label.
    seqs = um.create_sequences(seqnum=1, seqlen=3000, seed=1)
    hits = um.scan_sequences(um.example_motifs()[:4], seqs, pvalue=1e-3)
    for kw in ({"size_by": "score"}, {"size_by": "score", "shade_by": "score"}):
        fig = um.plot_hits(hits, seqs, **kw)
        ax = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        sizes = ax.artists[1]
        texts = sizes.get_texts()
        handles = sizes.legend_handles
        assert max(h.get_markersize() for h in handles) <= 1.5 * 9
        # Rows are at least as far apart as the largest marker is tall.
        ys = sorted(t.get_window_extent(renderer).y0 for t in texts)
        tallest = max(h.get_markersize() for h in handles) * fig.dpi / 72
        assert all(b - a >= tallest - 1 for a, b in itertools.pairwise(ys))
        box = sizes.get_window_extent(renderer)
        for label in ax.get_xticklabels():
            if label.get_text():
                e = label.get_window_extent(renderer)
                assert not (e.x0 < box.x1 and box.x0 < e.x1 and e.y0 < box.y1 and box.y0 < e.y1)
        tight = fig.get_tightbbox(renderer)
        assert box.x1 / fig.dpi <= tight.x1 + 1e-6
