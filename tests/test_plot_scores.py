import itertools
import math

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from matplotlib.collections import LineCollection
from matplotlib.patches import StepPatch
from universalmotif._plot_scores import _format_p, plot_score_distribution


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _small():
    return um.create_motif(["TCAG", "TCAG", "TGAG", "TCTG"], pseudocount=1)


def _wbox():
    return um.example_motifs()[5]


def _bars(ax):
    """The edges and probabilities of the filled distribution."""
    (patch,) = [p for p in ax.patches if isinstance(p, StepPatch) and p.get_fill()]
    data = patch.get_data()
    return np.asarray(data.edges), np.asarray(data.values)


def _cutoff_lines(ax):
    return sorted(float(line.get_xdata()[0]) for line in ax.lines)


def _rug(ax):
    (rug,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    return np.sort([seg[0][0] for seg in rug.get_segments()])


def _brute_force(motif, edges):
    """The probability of each bin from scoring every word, weighted by its
    background probability."""
    letters = motif.letters
    bkg = np.array([motif.bkg[c] for c in letters])
    words = ["".join(w) for w in itertools.product(letters, repeat=motif.width)]
    probs = np.array([math.prod(bkg[letters.index(c)] for c in w) for w in words])
    scores = np.asarray(um.score_match(motif, words))
    grid = np.round(edges * 1000).astype(int)
    idx = np.searchsorted(grid, np.round(scores * 1000).astype(int), side="right") - 1
    assert idx.min() >= 0 and idx.max() < len(grid) - 1
    return np.bincount(idx, weights=probs, minlength=len(grid) - 1)


def test_probabilities_sum_to_one_and_match_enumeration():
    m = _small()
    for bins in (7, 30, 100_000):
        fig = plot_score_distribution(m, pvalues=None, bins=bins)
        edges, mass = _bars(fig.axes[0])
        assert len(mass) <= bins
        assert mass.sum() == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(mass, _brute_force(m, edges), atol=1e-14)
        lo, hi = um.motif_range(m)
        assert edges[0] == pytest.approx(lo, abs=1e-9)
        assert edges[-1] == pytest.approx(hi + 0.001, abs=1e-9)
        plt.close(fig)


def test_one_bin_per_score_is_the_enumerated_distribution():
    m = _small()
    fig = plot_score_distribution(m, pvalues=None, bins=100_000)
    edges, mass = _bars(fig.axes[0])
    scores = um.get_scores(m)
    want = {}
    for s in np.round(scores * 1000).astype(int):
        want[s] = want.get(s, 0) + 1 / len(scores)
    grid = np.round(edges[:-1] * 1000).astype(int)
    got = {g: p for g, p in zip(grid, mass) if p > 0}
    assert got.keys() == want.keys()
    for g, p in want.items():
        assert got[g] == pytest.approx(p, abs=1e-14)


def test_uneven_background_matches_enumeration():
    m = _small().replace(bkg=[0.1, 0.4, 0.3, 0.2])
    fig = plot_score_distribution(m, pvalues=None, bins=25)
    edges, mass = _bars(fig.axes[0])
    np.testing.assert_allclose(mass, _brute_force(m, edges), atol=1e-14)


def test_bkg_argument_scores_against_that_background():
    m = _wbox()
    bkg = [0.3, 0.2, 0.2, 0.3]
    fig = plot_score_distribution(m, bkg=bkg, pvalues=1e-4, bins=60)
    edges, mass = _bars(fig.axes[0])
    assert mass.sum() == pytest.approx(1.0, abs=1e-12)
    tail = um.motif_pvalue(m, score=edges[1:-1].tolist(), bkg_probs=bkg)
    np.testing.assert_allclose(mass, -np.diff(np.r_[1.0, tail, 0.0]), atol=1e-15)
    assert mass[-1] > 0
    assert _cutoff_lines(fig.axes[0]) == [um.motif_pvalue(m, pvalue=1e-4, bkg_probs=bkg)]
    # A background mapping, and one per motif, give the same picture.
    by_letter = dict(zip("ACGT", bkg))
    fig2 = plot_score_distribution([m, m], bkg=[by_letter, bkg], pvalues=1e-4, bins=60)
    for ax in fig2.axes[:2]:
        np.testing.assert_array_equal(_bars(ax)[1], mass)


def test_cutoff_line_sits_at_the_threshold_and_bounds_the_tail():
    m = _wbox()
    p = 1e-4
    fig = plot_score_distribution(m, pvalues=p, bins=100_000)
    ax = fig.axes[0]
    threshold = um.motif_pvalue(m, pvalue=p)
    assert _cutoff_lines(ax) == [threshold]
    edges, mass = _bars(ax)
    grid = np.round(edges[:-1] * 1000).astype(int)
    at = np.searchsorted(grid, round(threshold * 1000))
    assert grid[at] == round(threshold * 1000)
    assert mass[at:].sum() <= p
    assert mass[at:].sum() == pytest.approx(um.motif_pvalue(m, score=threshold), rel=1e-9)
    # The threshold is the lowest such score: one more step down exceeds the cutoff.
    lower = at - 1
    while mass[lower] == 0:
        lower -= 1
    assert mass[lower:].sum() > p
    texts = [t.get_text() for t in ax.texts]
    hi = um.motif_range(m)[1]
    assert texts == [f"P = 1e-4, score {threshold:.2f} ({100 * threshold / hi:.0f}%)"]


def test_several_cutoffs():
    m = _wbox()
    ps = [1e-3, 1e-5, 1e-4]
    fig = plot_score_distribution(m, pvalues=ps)
    ax = fig.axes[0]
    want = sorted(um.motif_pvalue(m, pvalue=ps).tolist())
    assert _cutoff_lines(ax) == want
    assert sorted(t.get_text().split(",")[0] for t in ax.texts) == ["P = 0.001", "P = 1e-4", "P = 1e-5"]


def test_unreachable_cutoff_warns_and_is_skipped():
    m = um.example_motifs()[3]
    with pytest.warns(UserWarning, match="cannot reach a P-value of 1e-9"):
        fig = plot_score_distribution(m, pvalues=[1e-4, 1e-9])
    assert _cutoff_lines(fig.axes[0]) == [um.motif_pvalue(m, pvalue=1e-4)]


def test_hits_from_a_scan_table_list_and_mapping():
    motifs = [um.example_motifs()[i] for i in (3, 5)]
    hits = um.scan_sequences(motifs, um.example_sequences("targets"), pvalue=1e-3, nthreads=1)
    fig = plot_score_distribution(motifs, hits=hits)
    for ax, m in zip(fig.axes, motifs):
        want = np.sort(hits.filter(pl.col("motif") == m.name)["score"].to_numpy())
        assert want.size > 0
        np.testing.assert_allclose(_rug(ax), want)
        assert ax.get_legend().get_texts()[0].get_text() == f"hits (n = {want.size})"

    wbox = motifs[1]
    scores = [2.5, 9.0, 12.25]
    ax = plot_score_distribution(wbox, hits=scores, ax=plt.subplots()[1])
    np.testing.assert_allclose(_rug(ax), scores)
    fig = plot_score_distribution(motifs, hits={"W-box": scores})
    assert not [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    np.testing.assert_allclose(_rug(fig.axes[1]), scores)


def test_log_y():
    m = _wbox()
    fig = plot_score_distribution(m, log_y=True)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    _, mass = _bars(ax)
    lo, hi = ax.get_ylim()
    assert 0 < lo < mass[mass > 0].min() and hi > mass.max()
    # The tail is drawn: the top bin, holding the best words, is visible.
    assert mass[-1] >= um.motif_pvalue(m, score=um.motif_range(m)[1])
    assert ax.get_ylabel() == "Probability (log10)"


def test_amino_acid_motif():
    m = um.create_motif(10, alphabet="AA", seed=3, pseudocount=1)
    fig = plot_score_distribution(m, pvalues=[1e-4, 1e-8], log_y=True)
    _, mass = _bars(fig.axes[0])
    assert mass.sum() == pytest.approx(1.0, abs=1e-9)
    assert _cutoff_lines(fig.axes[0]) == sorted(um.motif_pvalue(m, pvalue=[1e-4, 1e-8]).tolist())


def test_panels_ax_and_percent_axis():
    motifs = um.example_motifs()[:3]
    fig = plot_score_distribution(motifs, ncol=2)
    main = [ax for ax in fig.axes if ax.get_title()]
    assert [ax.get_title() for ax in main] == [m.name for m in motifs]
    assert sum(not ax.get_visible() for ax in fig.axes) == 1

    fig, axes = plt.subplots(1, 2)
    ax = plot_score_distribution(motifs[0], ax=axes[0])
    assert ax is axes[0]
    (sec,) = ax.child_axes
    hi = um.motif_range(motifs[0])[1]
    assert sec.get_xlabel() == "% of maximum score"
    fig.canvas.draw()
    assert sec.get_xlim() == pytest.approx(tuple(100 * x / hi for x in ax.get_xlim()))


def test_errors():
    m = _wbox()
    with pytest.raises(ValueError, match="no motifs"):
        plot_score_distribution([])
    with pytest.raises(ValueError, match="single motif"):
        plot_score_distribution([m, m], ax=plt.subplots()[1])
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="bins"):
            plot_score_distribution(m, bins=bad)
    for bad in (0, 1.5, [1e-4, -1]):
        with pytest.raises(ValueError, match="pvalues"):
            plot_score_distribution(m, pvalues=bad)
    with pytest.raises(ValueError, match="one per motif"):
        plot_score_distribution([m, m, m], bkg=[[0.25] * 4, [0.25] * 4])
    with pytest.raises(ValueError, match="score column"):
        plot_score_distribution(m, hits=pl.DataFrame({"motif": ["W-box"]}))
    with pytest.raises(ValueError, match="motif column"):
        plot_score_distribution([m, m], hits=pl.DataFrame({"score": [1.0]}))
    with pytest.raises(ValueError, match="mapping"):
        plot_score_distribution([m, m], hits=[1.0, 2.0])
    with pytest.raises(ValueError, match="not plotted: G-box"):
        plot_score_distribution(m, hits={"G-box": [1.0]})
    with pytest.raises(TypeError):
        plot_score_distribution(["TGAC"])


def test_format_p():
    assert [_format_p(p) for p in (0.05, 1e-3, 1e-4, 2.5e-6, 9.999e-5)] == ["0.05", "0.001", "1e-4", "2.5e-6",
                                                                           "1e-4"]
