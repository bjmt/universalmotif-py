import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from universalmotif import _core
from universalmotif._plot_comparison import plot_comparison


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def motifs():
    return um.example_motifs()


def _axes(fig):
    return {ax.get_label(): ax for ax in fig.axes}


def _row_labels(fig):
    return [t.get_text() for t in _axes(fig)["labels"].texts]


def _cells(fig):
    mesh = _axes(fig)["heatmap"].collections[0]
    return np.ma.filled(mesh.get_array(), np.nan).reshape(mesh.get_coordinates().shape[0] - 1, -1)


def test_order_matches_motif_tree(motifs):
    for method in ("complete", "average"):
        tree = um.motif_tree(motifs, min_overlap=6, method=method)
        fig = plot_comparison(motifs, method=method)
        want = [tree.labels[i] for i in tree.order]
        assert _row_labels(fig) == want
        ticks = [t.get_text() for t in _axes(fig)["heatmap"].get_xticklabels()]
        assert ticks == want


def test_cells_hold_the_comparison(motifs):
    scores = um.compare_motifs(motifs, min_overlap=6)
    order = um.motif_tree(motifs, min_overlap=6).order
    fig = plot_comparison(motifs)
    want = scores[np.ix_(order, order)]
    np.testing.assert_allclose(_cells(fig), want, rtol=1e-12)
    texts = _axes(fig)["heatmap"].texts
    assert len(texts) == len(motifs) ** 2
    for t in texts:
        i, j = (round(v) for v in t.get_position()[::-1])
        assert t.get_text() == f"{want[i, j]:.2f}"


def test_pvalues_are_drawn_as_log10_without_the_diagonal(motifs):
    ms = motifs[:6]
    p = um.compare_motifs(ms, min_overlap=6, matrix_out="pvalue")
    order = um.motif_tree(ms, min_overlap=6).order
    fig = plot_comparison(ms, value="pvalue")
    cells = _cells(fig)
    want = -np.log10(p[np.ix_(order, order)])
    assert np.isnan(np.diag(cells)).all()
    off = ~np.eye(len(ms), dtype=bool)
    np.testing.assert_allclose(cells[off], want[off], rtol=1e-12)
    # The order still comes from the scores.
    assert _row_labels(fig) == [ms[i].name for i in order]
    texts = [t.get_text() for t in _axes(fig)["heatmap"].texts]
    assert len(texts) == len(ms) * (len(ms) - 1)
    assert "-0.0" not in texts
    assert _axes(fig)["colour bar"].get_ylabel() == "-log10(P-value)"


def test_colour_bar_range(motifs):
    scores = um.compare_motifs(motifs, min_overlap=6)
    fig = plot_comparison(motifs)
    cax = _axes(fig)["colour bar"]
    assert cax.get_ylim() == pytest.approx((scores.min(), scores.max()))
    assert cax.get_ylabel() == "Similarity score"

    ms = motifs[:6]
    logs = -np.log10(um.compare_motifs(ms, min_overlap=6, matrix_out="qvalue"))
    np.fill_diagonal(logs, np.nan)
    fig = plot_comparison(ms, value="qvalue")
    assert _axes(fig)["colour bar"].get_ylim() == pytest.approx((0.0, np.nanmax(logs)))

    fig = plot_comparison(motifs, vmin=0.2, vmax=0.6, cmap="viridis")
    mesh = _axes(fig)["heatmap"].collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0.2, 0.6)
    assert mesh.cmap.name.startswith("viridis")
    assert _axes(fig)["colour bar"].get_ylim() == pytest.approx((0.2, 0.6))


def test_logos_beside_labels(motifs):
    fig = plot_comparison(motifs)
    axes = _axes(fig)
    logos = axes["logos"].collections
    assert len(logos) == len(motifs)
    assert all(len(c.get_paths()) > 0 for c in logos)
    # Each logo sits on its row and the labels follow the logos.
    assert axes["heatmap"].get_position().x1 <= axes["logos"].get_position().x0
    assert axes["logos"].get_position().x1 <= axes["labels"].get_position().x0
    names = _row_labels(fig)
    n = len(motifs)
    for k, c in enumerate(logos):
        ys = np.concatenate([p.vertices[:, 1] for p in c.get_paths()])
        assert n - 1 - k - 0.4 - 1e-9 <= ys.min() and ys.max() <= n - 1 - k + 0.4 + 1e-9, names[k]

    assert "logos" not in _axes(plot_comparison(motifs, logos=False))
    assert "logos" not in _axes(plot_comparison(matrix=um.compare_motifs(motifs[:4])))


def test_many_motifs_drop_logos_and_values(motifs):
    ms = [m.replace(name=f"{m.name}.{k}") for k in range(6) for m in motifs][:61]
    scores = np.full((61, 61), 0.2)
    np.fill_diagonal(scores, 1.0)
    fig = plot_comparison(ms, matrix=scores)
    axes = _axes(fig)
    assert "logos" not in axes
    assert not axes["heatmap"].texts
    assert len(_row_labels(fig)) == 61
    forced = _axes(plot_comparison(ms, matrix=scores, logos=True, annotate=True))
    assert len(forced["logos"].collections) == 61
    assert len(forced["heatmap"].texts) == 61 * 61
    # Far too many motifs for readable labels leaves them out.
    big = np.eye(400) * 0.5 + 0.5
    assert "labels" not in _axes(plot_comparison(matrix=big, cluster=False, dendrogram="none"))


def test_precomputed_matrix(motifs):
    ms = motifs[:6]
    names = [m.name for m in ms]
    scores = um.compare_motifs(ms, min_overlap=6)
    computed = plot_comparison(ms)
    from_matrix = plot_comparison(ms, matrix=scores)
    np.testing.assert_allclose(_cells(from_matrix), _cells(computed))
    assert _row_labels(from_matrix) == _row_labels(computed)
    assert len(_axes(from_matrix)["logos"].collections) == len(ms)

    frame = pl.DataFrame(scores, schema=names)
    fig = plot_comparison(matrix=frame)
    assert _row_labels(fig) == _row_labels(computed)
    fig = plot_comparison(matrix=scores.tolist())
    assert sorted(_row_labels(fig)) == sorted(str(i + 1) for i in range(len(ms)))
    fig = plot_comparison(matrix=scores, labels=list("abcdef"))
    assert sorted(_row_labels(fig)) == list("abcdef")


def test_pvalue_matrix_without_motifs_clusters_on_log10(motifs):
    ms = motifs[:8]
    p = um.compare_motifs(ms, min_overlap=6, matrix_out="pvalue")
    logs = -np.log10(np.clip(p, np.finfo(float).tiny, None))
    sym = (logs + logs.T) / 2
    np.fill_diagonal(sym, 0)
    off = ~np.eye(len(ms), dtype=bool)
    dist = 1 - sym / sym[off].max()
    np.fill_diagonal(dist, 0)
    names = [m.name for m in ms]
    order = _core._hclust(dist.tolist(), "complete", names)["order"]
    fig = plot_comparison(matrix=p, labels=names, value="pvalue")
    assert _row_labels(fig) == [names[i] for i in order]


def test_cluster_choices(motifs):
    ms = motifs[:6]
    fig = plot_comparison(ms, cluster=False)
    assert _row_labels(fig) == [m.name for m in ms]
    assert "dendrogram left" not in _axes(fig)

    tree = um.motif_tree(ms, method="single")
    fig = plot_comparison(ms, cluster=tree)
    assert _row_labels(fig) == [tree.labels[i] for i in tree.order]


def test_dendrogram_geometry(motifs):
    ms = motifs[:6]
    tree = um.motif_tree(ms)
    fig = plot_comparison(ms, dendrogram="both")
    axes = _axes(fig)
    left = axes["dendrogram left"].collections[0].get_segments()
    top = axes["dendrogram top"].collections[0].get_segments()
    assert len(left) == len(top) == len(ms) - 1
    # Each merge is a bracket at its height, and the leaves sit on the rows.
    np.testing.assert_allclose(sorted(s[1, 0] for s in left), sorted(tree.height))
    np.testing.assert_allclose(sorted(s[1, 1] for s in top), sorted(tree.height))
    leaves = sorted({s[0, 1] for s in left if s[0, 0] == 0} | {s[3, 1] for s in left if s[3, 0] == 0})
    assert leaves == list(range(len(ms)))
    assert axes["dendrogram left"].get_ylim() == axes["heatmap"].get_ylim()
    assert axes["dendrogram top"].get_xlim() == axes["heatmap"].get_xlim()
    assert "dendrogram left" not in _axes(plot_comparison(ms, dendrogram="top"))
    assert {"dendrogram left", "dendrogram top"}.isdisjoint(_axes(plot_comparison(ms, dendrogram="none")))


def test_figsize_and_annotate(motifs):
    fig = plot_comparison(motifs[:6], figsize=(9, 5), annotate=False)
    assert tuple(fig.get_size_inches()) == (9, 5)
    assert not _axes(fig)["heatmap"].texts
    # Cells stay square.
    pos = _axes(fig)["heatmap"].get_position()
    assert pos.width * 9 == pytest.approx(pos.height * 5)


def test_errors(motifs):
    with pytest.raises(ValueError, match="give motifs, a matrix or both"):
        plot_comparison()
    with pytest.raises(ValueError, match="value must be"):
        plot_comparison(motifs, value="evalue")
    with pytest.raises(ValueError, match="dendrogram must be"):
        plot_comparison(motifs, dendrogram="right")
    with pytest.raises(ValueError, match="matrix must be square"):
        plot_comparison(matrix=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="one row and column per motif"):
        plot_comparison(motifs[:3], matrix=np.eye(4))
    with pytest.raises(ValueError, match="at least two motifs"):
        plot_comparison(motifs[:1])
    with pytest.raises(ValueError, match="one label per motif"):
        plot_comparison(motifs[:3], labels=["a", "b"])
    with pytest.raises(ValueError, match="logos need the motifs"):
        plot_comparison(matrix=np.eye(3), logos=True)
    with pytest.raises(ValueError, match="one tip per motif"):
        plot_comparison(motifs[:4], cluster=um.motif_tree(motifs[:3]))
    with pytest.raises(ValueError, match="unknown clustering method"):
        plot_comparison(motifs[:4], method="ward")
    with pytest.raises(ValueError, match="min_overlap"):
        plot_comparison(motifs[:4], min_overlap=0)
    with pytest.raises(ValueError, match="figsize is too small"):
        plot_comparison(motifs, figsize=(2, 2))
