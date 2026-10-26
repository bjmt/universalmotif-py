import math

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from universalmotif._plot_results import _place_labels, plot_coocc, plot_enrichment


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _enrich_table(n=40, seed=1):
    """A table shaped like enrich_motifs() output, with a group of strongly
    enriched motifs and a cloud of unenriched ones."""
    rng = np.random.default_rng(seed)
    k = n // 4
    x = np.concatenate([rng.normal(2.0, 0.3, k), rng.normal(0.0, 0.4, n - k)])
    log10_q = -np.concatenate([rng.uniform(2, 6, k), rng.uniform(0, 0.9, n - k)])
    log10_p = log10_q - 0.5
    return pl.DataFrame({
        "motif": [f"m{i}" for i in range(n)],
        "motif_i": pl.Series(range(n), dtype=pl.UInt32),
        "target_seq_hits": rng.integers(1, 80, n),
        "target_site_hits": rng.integers(1, 120, n),
        "log2_enrichment": x,
        "pvalue": 10 ** log10_p,
        "log10_pvalue": log10_p,
        "qvalue": 10 ** log10_q,
        "log10_qvalue": log10_q,
    })


def _points(ax):
    """Every drawn point as (x, y, face colour, size, marker path vertices)."""
    out = []
    for c in ax.collections:
        if isinstance(c, PathCollection):
            faces = c.get_facecolors()
            sizes = c.get_sizes()
            for k, (px, py) in enumerate(c.get_offsets()):
                out.append((float(px), float(py), to_hex(faces[k % len(faces)]), float(sizes[k % len(sizes)]),
                            len(c.get_paths()[0].vertices)))
    return out


def _labels(ax):
    """The motif labels, leaving out the label of the cutoff line."""
    return {t.get_text(): t for t in ax.texts if not t.get_text().startswith(("q =", "p ="))}


def _text_box(t, renderer):
    """The extent of a label's text alone, without its leader line."""
    t.update_positions(renderer)
    return Text.get_window_extent(t, renderer)


def _boxes_overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


# --- plot_enrichment ----------------------------------------------------------


def test_enrichment_points_are_the_table_values():
    res = _enrich_table()
    fig = plot_enrichment(res)
    assert type(fig).__name__ == "Figure"
    pts = sorted((x, y) for x, y, *_ in _points(fig.axes[0]))
    want = sorted(zip(res["log2_enrichment"].to_list(), (-res["log10_qvalue"]).to_list()))
    np.testing.assert_allclose(pts, want, rtol=1e-12)


def test_enrichment_pvalue_axis_and_underflow():
    res = _enrich_table().with_columns(
        pl.when(pl.col("motif") == "m0").then(0.0).otherwise(pl.col("qvalue")).alias("qvalue"),
        pl.when(pl.col("motif") == "m0").then(-400.0).otherwise(pl.col("log10_qvalue")).alias("log10_qvalue"),
    )
    ax = plot_enrichment(res).axes[0]
    ys = [y for _, y, *_ in _points(ax)]
    assert max(ys) == 400.0
    assert ax.get_ylim()[1] > 400
    ax = plot_enrichment(res, y="pvalue").axes[0]
    np.testing.assert_allclose(sorted(y for _, y, *_ in _points(ax)), sorted((-res["log10_pvalue"]).to_list()))
    assert ax.get_ylabel() == "-log10(P-value)"


def test_enrichment_without_log10_columns():
    res = _enrich_table().drop("log10_qvalue", "log10_pvalue")
    ax = plot_enrichment(res).axes[0]
    np.testing.assert_allclose(sorted(y for _, y, *_ in _points(ax)),
                               sorted((-np.log10(res["qvalue"].to_numpy())).tolist()), rtol=1e-12)


def test_enrichment_colours_and_cutoff_line():
    res = _enrich_table()
    cutoff = 0.01
    ax = plot_enrichment(res, cutoff=cutoff, colour="#FF0000").axes[0]
    y_of = dict(zip(res["log2_enrichment"].to_list(), (-res["log10_qvalue"]).to_list()))
    for x, y, face, *_ in _points(ax):
        assert y == y_of[x]
        assert face == ("#ff0000" if y >= 2 else "#b4b4b4")
    flat = [ln for ln in ax.lines if np.ptp(ln.get_ydata()) == 0 and ln.get_linestyle() == "--"]
    assert len(flat) == 1
    assert flat[0].get_ydata()[0] == pytest.approx(-math.log10(cutoff))
    assert any(t.get_text() == "q = 0.01" for t in ax.texts)


def test_enrichment_point_sizes():
    res = _enrich_table()
    ax = plot_enrichment(res, size_by="target_site_hits").axes[0]
    by_x = {x: s for x, _, _, s, _ in _points(ax)}
    sizes = np.array([by_x[x] for x in res["log2_enrichment"]])
    hits = res["target_site_hits"].to_numpy()
    # Areas grow in step with the number of hits.
    np.testing.assert_allclose(np.diff(sizes[np.argsort(hits)]) >= -1e-12, True)
    assert sizes[hits.argmin()] < sizes[hits.argmax()]
    leg = ax.get_legend()
    assert leg.get_title().get_text() == "Target hits"
    ax = plot_enrichment(res, size_by=None).axes[0]
    assert len({s for *_, s, _ in _points(ax)}) == 1
    assert ax.get_legend() is None


def test_enrichment_labels_top_motifs():
    res = _enrich_table()
    ax = plot_enrichment(res, labels=5).axes[0]
    top = res.sort("log10_qvalue").head(5)["motif"].to_list()
    labels = _labels(ax)
    assert set(labels) == set(top)
    # Each label points at its motif.
    for name, t in labels.items():
        row = res.row(res["motif"].to_list().index(name), named=True)
        assert t.xy == pytest.approx((row["log2_enrichment"], -row["log10_qvalue"]))
    assert _labels(plot_enrichment(res, labels=0).axes[0]) == {}
    # Only significant motifs are labelled by number.
    assert len(_labels(plot_enrichment(res, labels=100, cutoff=1e-3).axes[0])) == \
        int((res["log10_qvalue"] <= -3).sum())


def test_enrichment_labels_by_name():
    res = _enrich_table()
    labels = _labels(plot_enrichment(res, labels=["m30", "m31"]).axes[0])
    assert set(labels) == {"m30", "m31"}
    assert set(_labels(plot_enrichment(res, labels="m30").axes[0])) == {"m30"}
    assert len(_labels(plot_enrichment(res, labels=np.int64(2)).axes[0])) == 2
    with pytest.raises(ValueError, match="no motifs named nope"):
        plot_enrichment(res, labels=["m1", "nope"])


def test_enrichment_labels_do_not_overlap():
    res = _enrich_table(n=60, seed=3)
    fig = plot_enrichment(res, labels=12)
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    boxes = [_text_box(t, renderer) for t in _labels(ax).values()]
    assert len(boxes) == 12
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _boxes_overlap(boxes[i], boxes[j])
    frame = ax.get_window_extent(renderer)
    assert all(frame.x0 <= b.x0 and b.x1 <= frame.x1 and frame.y0 <= b.y0 and b.y1 <= frame.y1 for b in boxes)


def test_place_labels_nudges_apart():
    # Three points on top of each other with labels of the same size.
    px, py = np.array([100.0, 100.0, 100.0]), np.array([100.0, 100.0, 100.0])
    spots = _place_labels(px, py, np.full(3, 3.0), [(30.0, 10.0)] * 3, [0, 1, 2], (0, 0, 300, 300))
    boxes = [(dx, dy, dx + 30, dy + 10) for dx, dy, _ in spots]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = boxes[i], boxes[j]
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    # A lone label sits next to its point, with no leader line.
    (dx, dy, far), = _place_labels(px[:1], py[:1], np.full(1, 3.0), [(30.0, 10.0)], [0], (0, 0, 300, 300))
    assert not far
    assert math.hypot(dx, dy + 5) < 10


def test_enrichment_no_target_hits_at_left_edge():
    res = _enrich_table().with_columns(
        pl.when(pl.col("motif") == "m35").then(-math.inf).otherwise(pl.col("log2_enrichment"))
        .alias("log2_enrichment"))
    ax = plot_enrichment(res).axes[0]
    pts = _points(ax)
    left = ax.get_xlim()[0]
    finite = [x for x, *_ in pts if x > left + 1e-9 and x != min(p[0] for p in pts)]
    edge = [p for p in pts if p[0] == min(q[0] for q in pts)]
    assert len(edge) == 1
    assert left < edge[0][0] < min(finite)
    assert edge[0][1] == pytest.approx(-res.filter(pl.col("motif") == "m35")["log10_qvalue"][0])
    # Drawn with a different marker from the others.
    assert edge[0][4] != next(p[4] for p in pts if p is not edge[0])


def test_enrichment_logos():
    res = _enrich_table()
    motifs = [um.example_motifs()[i].replace(name=f"m{i}") for i in range(10)]
    fig = plot_enrichment(res, labels=4, motifs=motifs)
    ax = fig.axes[0]
    logos = [c for c in ax.collections if isinstance(c, PolyCollection) and not isinstance(c, PathCollection)]
    assert len(logos) == 4
    renderer = fig.canvas.get_renderer()
    texts = [_text_box(t, renderer) for t in _labels(ax).values()]
    for logo in logos:
        corners = np.concatenate([logo.get_transform().transform(p.vertices) for p in logo.get_paths()])
        box = Bbox([corners.min(axis=0), corners.max(axis=0)])
        assert box.width > 0
        # Every logo sits beside one label, level with it, and covers none.
        assert not any(_boxes_overlap(box, t) for t in texts)
        level = [max(t.x0 - box.x1, box.x0 - t.x1) for t in texts if t.y0 < box.y1 and box.y0 < t.y1]
        assert level and 0 < min(level) < 25
    with pytest.raises(ValueError, match="motifs has no motif named"):
        plot_enrichment(res, labels=4, motifs=motifs[:2])
    # A mapping from names works too.
    top = res.sort("log10_qvalue")["motif"][0]
    fig = plot_enrichment(res, labels=1, motifs={top: um.example_motifs()[3]})
    assert sum(isinstance(c, PolyCollection) and not isinstance(c, PathCollection)
               for c in fig.axes[0].collections) == 1


def test_enrichment_real_results_and_axes():
    motifs = um.example_motifs()
    res = um.enrich_motifs(motifs, um.example_sequences("targets"), um.example_sequences("controls"),
                           qvalue=1, nthreads=1)
    _, (a, b) = plt.subplots(1, 2)
    out = plot_enrichment(res, labels=2, motifs=motifs, ax=b)
    assert out is b
    assert set(_labels(b)) == {"G-box", "W-box"}
    assert len(_points(b)) == res.height
    assert not a.collections


def test_enrichment_nothing_significant():
    # A cutoff of 1 puts the line at zero, and all P-values of 1 leave
    # nothing above it.
    res = _enrich_table().with_columns(pl.lit(0.0).alias("log10_qvalue"), pl.lit(1.0).alias("qvalue"))
    ax = plot_enrichment(res, cutoff=1).axes[0]
    assert ax.get_ylim()[1] > 0
    assert all(face == "#255c99" for _, _, face, *_ in _points(ax))
    ax = plot_enrichment(res, cutoff=0.5).axes[0]
    assert _labels(ax) == {}
    assert all(face == "#b4b4b4" for _, _, face, *_ in _points(ax))


def test_enrichment_errors():
    res = _enrich_table()
    co = um.motif_coocc([um.example_motifs()[i] for i in (3, 5)], um.example_sequences("targets"), nthreads=1)
    with pytest.raises(ValueError, match="enrich_motifs"):
        plot_enrichment(co)
    with pytest.raises(ValueError, match="no rows"):
        plot_enrichment(res.clear())
    with pytest.raises(ValueError, match="y must be"):
        plot_enrichment(res, y="evalue")
    with pytest.raises(ValueError, match="cutoff"):
        plot_enrichment(res, cutoff=0)
    with pytest.raises(ValueError, match="no nhits column"):
        plot_enrichment(res, size_by="nhits")
    with pytest.raises(ValueError, match="no qvalue or log10_qvalue"):
        plot_enrichment(res.drop("qvalue", "log10_qvalue"))
    with pytest.raises(ValueError, match="labels must be"):
        plot_enrichment(res, labels=-1)


# --- plot_coocc ---------------------------------------------------------------


def _coocc_table():
    """Two blocks of motifs that turn up together, A1-A3 and B1-B2, with an
    untested pair."""
    pairs = {
        ("A1", "A2"): (8.0, 1e-6), ("A1", "A3"): (6.0, 1e-4), ("A2", "A3"): (4.0, 1e-3),
        ("B1", "B2"): (5.0, 1e-5),
        ("A1", "B1"): (0.25, 0.9), ("A2", "B1"): (0.5, 0.8), ("A3", "B1"): (0.6, 0.7),
        ("A1", "B2"): (0.3, 0.9), ("A2", "B2"): (0.4, 0.9),
    }
    rows = [(a, b, 1, 1, 1, 1, odds, q / 2, q) for (a, b), (odds, q) in pairs.items()]
    rows.append(("A3", "B2", 1, 1, 0, 1, None, None, None))
    # Interleaved, so that clustering has to bring the blocks together.
    order = [0, 4, 1, 7, 3, 2, 5, 8, 6, 9]
    rows = [rows[i] for i in order]
    return pl.DataFrame(rows, schema={"motif_a": pl.String, "motif_b": pl.String, "a_only": pl.Int64,
                                      "b_only": pl.Int64, "both": pl.Int64, "neither": pl.Int64,
                                      "odds_ratio": pl.Float64, "pvalue": pl.Float64, "qvalue": pl.Float64},
                        orient="row")


def _heatmap(fig_or_ax):
    ax = fig_or_ax if isinstance(fig_or_ax, Axes) else fig_or_ax.axes[0]
    image = ax.get_images()[0]
    names = [t.get_text() for t in ax.get_xticklabels()]
    return ax, image, names


def test_coocc_cells_are_log2_odds_and_symmetric():
    res = _coocc_table()
    ax, image, names = _heatmap(plot_coocc(res, order="input"))
    assert names == ["A1", "B1", "A2", "A3", "B2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == names
    m = image.get_array()
    np.testing.assert_array_equal(m.mask, m.mask.T)
    np.testing.assert_allclose(m.filled(np.nan), m.filled(np.nan).T)
    for row in res.iter_rows(named=True):
        i, j = names.index(row["motif_a"]), names.index(row["motif_b"])
        if row["odds_ratio"] is None:
            assert m.mask[i, j]
        else:
            assert m[i, j] == pytest.approx(math.log2(row["odds_ratio"]))
    assert m.mask.diagonal().all()
    # The scale is centred on zero.
    lo, hi = image.get_clim()
    assert lo == -hi == pytest.approx(-3.0)


def test_coocc_qvalue_cells():
    res = _coocc_table()
    fig = plot_coocc(res, value="qvalue", order="input")
    _, image, names = _heatmap(fig)
    m = image.get_array()
    for row in res.drop_nulls().iter_rows(named=True):
        i, j = names.index(row["motif_a"]), names.index(row["motif_b"])
        assert m[i, j] == m[j, i] == pytest.approx(-math.log10(row["qvalue"]))
    assert image.get_clim()[0] == 0
    assert fig.axes[1].get_ylabel() == "-log10(q-value)"


def test_coocc_significance_marks():
    res = _coocc_table()
    ax, _, names = _heatmap(plot_coocc(res, cutoff=1e-3))
    marks = {(int(x), int(y)) for c in ax.collections if isinstance(c, PathCollection) for x, y in c.get_offsets()}
    want = set()
    for row in res.drop_nulls().filter(pl.col("qvalue") <= 1e-3).iter_rows(named=True):
        i, j = names.index(row["motif_a"]), names.index(row["motif_b"])
        want |= {(i, j), (j, i)}
    assert marks == want
    assert len(want) == 8
    assert ax.get_legend().get_texts()[0].get_text() == "q ≤ 0.001"


def test_coocc_clustered_order():
    _, _, names = _heatmap(plot_coocc(_coocc_table()))
    assert sorted(names) == ["A1", "A2", "A3", "B1", "B2"]
    blocks = "".join(n[0] for n in names)
    assert blocks in ("AAABB", "BBAAA")


def test_coocc_given_order():
    _, image, names = _heatmap(plot_coocc(_coocc_table(), order=["B2", "A1", "A2"]))
    assert names == ["B2", "A1", "A2"]
    m = image.get_array()
    assert m[1, 2] == m[2, 1] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="no motifs named C1"):
        plot_coocc(_coocc_table(), order=["A1", "C1"])


def test_coocc_infinite_odds_and_self_pairs():
    res = _coocc_table().with_columns(
        pl.when(pl.col("motif_a").eq("A1") & pl.col("motif_b").eq("A2")).then(math.inf)
        .when(pl.col("motif_a").eq("A1") & pl.col("motif_b").eq("B1")).then(0.0)
        .otherwise(pl.col("odds_ratio")).alias("odds_ratio"))
    res = pl.concat([res, pl.DataFrame([("A1", "A1", 3, 3, 3, 3, 2.0, 0.5, 0.5)], schema=res.schema,
                                       orient="row")])
    fig = plot_coocc(res, order="input")
    ax, image, names = _heatmap(fig)
    m = image.get_array()
    lo, hi = image.get_clim()
    i, j, k = names.index("A1"), names.index("A2"), names.index("B1")
    assert hi == pytest.approx(math.log2(6.0))
    assert m[i, j] == hi and m[i, k] == lo
    assert m[i, i] == pytest.approx(1.0)
    assert fig.axes[1].get_label() == "<colorbar>"
    assert ax.figure.axes[1].get_ylabel() == "log2(odds ratio)"


def test_coocc_real_results_into_axes():
    motifs = um.example_motifs()[:6]
    res = um.motif_coocc(motifs, um.example_sequences("targets"), nthreads=1, max_distance=50)
    fig, ax = plt.subplots()
    out = plot_coocc(res, ax=ax)
    assert out is ax
    _, image, names = _heatmap(ax)
    assert sorted(names) == sorted(m.name for m in motifs)
    assert image.get_array().shape == (6, 6)
    assert len(fig.axes) == 2


def test_coocc_errors():
    res = _coocc_table()
    with pytest.raises(ValueError, match="motif_coocc"):
        plot_coocc(_enrich_table())
    with pytest.raises(ValueError, match="no rows"):
        plot_coocc(res.clear())
    with pytest.raises(ValueError, match="value must be"):
        plot_coocc(res, value="both")
    with pytest.raises(ValueError, match="order must be"):
        plot_coocc(res, order="name")
    with pytest.raises(ValueError, match="cutoff"):
        plot_coocc(res, cutoff=2)
