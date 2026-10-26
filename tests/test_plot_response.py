import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from universalmotif._plot_response import (
    _signed,
    _t975,
    plot_marginalization,
    plot_spacing,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Tensor:
    """Stands in for a tensor on a device, which numpy cannot read
    directly."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __array__(self, *args, **kwargs):
        raise TypeError("convert with .numpy()")


def _spacing_preds(n=20, outputs=None, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (n,) if outputs is None else (n, outputs))
    preds = {}
    for d in range(25):
        effect = 1 + np.cos(2 * np.pi * d / 10.5)
        preds[d] = base + effect + rng.normal(0, 0.2, base.shape)
    return preds, base


def _data_lines(ax):
    """The lines drawn for outputs, leaving out reference lines and guides."""
    return [line for line in ax.lines if line.get_label() and not line.get_label().startswith("_")]


def _bands(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def _band_limits(band, xs):
    v = band.get_paths()[0].vertices
    lo, hi = [], []
    for x in xs:
        ys = v[np.isclose(v[:, 0], x), 1]
        lo.append(ys.min())
        hi.append(ys.max())
    return np.array(lo), np.array(hi)


def _guides(ax):
    return sorted(float(line.get_xdata()[0]) for line in ax.lines
                  if line.get_linestyle() not in ("-", "None") and len(set(line.get_xdata())) == 1)


def test_spacing_draws_the_mean_change_from_the_baseline():
    preds, base = _spacing_preds()
    fig = plot_spacing(preds, baseline=base)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    (line,) = _data_lines(ax)
    x, y = line.get_data()
    assert list(x) == list(range(25))
    np.testing.assert_allclose(y, [np.mean(preds[d] - base) for d in range(25)])
    assert ax.get_ylabel() == "Change in prediction"
    assert ax.get_xlabel() == "Spacing between sites (bp)"
    zero = [ln for ln in ax.lines if list(ln.get_ydata()) == [0, 0]]
    assert len(zero) == 1
    assert ax.get_legend() is None


def test_spacing_without_baseline_draws_raw_means_and_no_zero_line():
    preds, _ = _spacing_preds()
    ax = plot_spacing(preds).axes[0]
    (line,) = _data_lines(ax)
    np.testing.assert_allclose(line.get_ydata(), [preds[d].mean() for d in range(25)])
    assert ax.get_ylabel() == "Prediction"
    assert not [ln for ln in ax.lines if list(ln.get_ydata()) == [0, 0]]


def test_spacing_takes_a_single_number_as_baseline():
    preds, _ = _spacing_preds()
    (line,) = _data_lines(plot_spacing(preds, baseline=2.0).axes[0])
    np.testing.assert_allclose(line.get_ydata(), [preds[d].mean() - 2 for d in range(25)])


@pytest.mark.parametrize("band", ["ci", "se", "sd"])
def test_spacing_band_widths(band):
    preds, base = _spacing_preds(n=12)
    ax = plot_spacing(preds, baseline=base, band=band).axes[0]
    (poly,) = _bands(ax)
    lo, hi = _band_limits(poly, range(25))
    diffs = [preds[d] - base for d in range(25)]
    sd = np.array([np.std(v, ddof=1) for v in diffs])
    half = {"sd": sd, "se": sd / np.sqrt(12), "ci": 2.201 * sd / np.sqrt(12)}[band]
    np.testing.assert_allclose((hi - lo) / 2, half, rtol=1e-6)
    np.testing.assert_allclose((hi + lo) / 2, [v.mean() for v in diffs])


def test_spacing_with_no_band_or_one_sequence_draws_no_band():
    preds, _ = _spacing_preds()
    assert not _bands(plot_spacing(preds, band=None).axes[0])
    one = {d: v[:1] for d, v in preds.items()}
    ax = plot_spacing(one).axes[0]
    assert not _bands(ax)
    assert len(_data_lines(ax)) == 1


def test_t_quantiles():
    assert _t975(1) == 12.706
    assert _t975(30) == 2.042
    for df, q in ((40, 2.0211), (100, 1.9840), (1000, 1.9623)):
        assert abs(_t975(df) - q) < 1e-3


def test_spacing_from_an_array_matches_the_dict():
    preds, base = _spacing_preds()
    stacked = np.stack([preds[d] for d in range(25)])
    a = _data_lines(plot_spacing(preds, baseline=base).axes[0])[0].get_ydata()
    b = _data_lines(plot_spacing(stacked, baseline=base, spacings=range(25)).axes[0])[0].get_ydata()
    np.testing.assert_allclose(a, b)


def test_spacing_takes_tensors_and_lists():
    preds, base = _spacing_preds()
    as_tensors = {d: _Tensor(v) for d, v in preds.items()}
    as_lists = {d: v.tolist() for d, v in preds.items()}
    want = _data_lines(plot_spacing(preds, baseline=base).axes[0])[0].get_ydata()
    for p, b in ((as_tensors, _Tensor(base)), (as_lists, base.tolist())):
        got = _data_lines(plot_spacing(p, baseline=b).axes[0])[0].get_ydata()
        np.testing.assert_allclose(got, want)
    stacked = _Tensor(np.stack([preds[d] for d in range(25)]))
    got = _data_lines(plot_spacing(stacked, baseline=base, spacings=range(25)).axes[0])[0].get_ydata()
    np.testing.assert_allclose(got, want)


def test_spacing_sorts_spacings():
    preds, _ = _spacing_preds()
    shuffled = {d: preds[d] for d in [10, 3, 24, 0, 7]}
    (line,) = _data_lines(plot_spacing(shuffled).axes[0])
    assert list(line.get_xdata()) == [0, 3, 7, 10, 24]
    np.testing.assert_allclose(line.get_ydata(), [preds[d].mean() for d in [0, 3, 7, 10, 24]])


def test_spacing_draws_one_line_per_output_with_a_legend():
    preds, base = _spacing_preds(outputs=3)
    fig = plot_spacing(preds, baseline=base, names=["a", "b", "c"])
    ax = fig.axes[0]
    lines = _data_lines(ax)
    assert [ln.get_label() for ln in lines] == ["a", "b", "c"]
    assert len({ln.get_color() for ln in lines}) == 3
    for j, ln in enumerate(lines):
        np.testing.assert_allclose(ln.get_ydata(), [np.mean(preds[d][:, j] - base[:, j]) for d in range(25)])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b", "c"]
    unnamed = _data_lines(plot_spacing(preds).axes[0])
    assert [ln.get_label() for ln in unnamed] == ["output 0", "output 1", "output 2"]


def test_spacing_picks_outputs():
    preds, base = _spacing_preds(outputs=3)
    names = ["a", "b", "c"]
    (line,) = _data_lines(plot_spacing(preds, baseline=base, output=1).axes[0])
    np.testing.assert_allclose(line.get_ydata(), [np.mean(preds[d][:, 1] - base[:, 1]) for d in range(25)])
    assert plot_spacing(preds, output=1).axes[0].get_legend() is None
    (line,) = _data_lines(plot_spacing(preds, names=names, output="c").axes[0])
    np.testing.assert_allclose(line.get_ydata(), [preds[d][:, 2].mean() for d in range(25)])
    lines = _data_lines(plot_spacing(preds, names=names, output=["c", 0]).axes[0])
    assert [ln.get_label() for ln in lines] == ["c", "a"]
    (line,) = _data_lines(plot_spacing(preds, output=-1).axes[0])
    np.testing.assert_allclose(line.get_ydata(), [preds[d][:, 2].mean() for d in range(25)])


def test_spacing_colours():
    preds, _ = _spacing_preds(outputs=2)
    lines = _data_lines(plot_spacing(preds, colour=["red", "black"]).axes[0])
    assert [ln.get_color() for ln in lines] == ["red", "black"]
    lines = _data_lines(plot_spacing(preds, colour="green").axes[0])
    assert {ln.get_color() for ln in lines} == {"green"}
    with pytest.raises(ValueError, match="2 colours for 1 outputs"):
        plot_spacing(preds, output=0, colour=["red", "black"])
    many, _ = _spacing_preds(outputs=9)
    with pytest.raises(ValueError, match="too many"):
        plot_spacing(many)
    assert len(_data_lines(plot_spacing(many, colour="grey").axes[0])) == 9


def test_spacing_helix_guides():
    preds, _ = _spacing_preds()
    ax = plot_spacing(preds, helix=5).axes[0]
    np.testing.assert_allclose(_guides(ax), [5, 15.5])
    assert not _guides(plot_spacing(preds).axes[0])
    # The strongest response of the test data is at spacing 0.
    np.testing.assert_allclose(_guides(plot_spacing(preds, helix=True).axes[0]), [0, 10.5, 21])
    assert set(ax.get_xticks()) >= {0, 5, 10, 15, 20}
    assert all(float(t).is_integer() for t in ax.get_xticks())


def test_spacing_into_axes():
    preds, base = _spacing_preds()
    fig, axes = plt.subplots(1, 2)
    out = plot_spacing(preds, baseline=base, ax=axes[1])
    assert out is axes[1]
    assert isinstance(out, Axes)
    assert _data_lines(axes[1]) and not _data_lines(axes[0])
    fig = plot_spacing(preds, figsize=(3, 2))
    assert tuple(fig.get_size_inches()) == (3, 2)


def test_spacing_errors():
    preds, base = _spacing_preds()
    stacked = np.stack([preds[d] for d in range(25)])
    with pytest.raises(ValueError, match="give the spacings"):
        plot_spacing(stacked)
    with pytest.raises(ValueError, match="3 spacings for 25 rows"):
        plot_spacing(stacked, spacings=[1, 2, 3])
    with pytest.raises(ValueError, match="only with an array"):
        plot_spacing(preds, spacings=range(25))
    with pytest.raises(ValueError, match="must all differ"):
        plot_spacing(stacked[:2], spacings=[1, 1])
    with pytest.raises(ValueError, match="band must be"):
        plot_spacing(preds, band="iqr")
    with pytest.raises(ValueError, match="same sequences in the same order"):
        plot_spacing(preds, baseline=base[:5])
    with pytest.raises(ValueError, match="baseline has 2 outputs"):
        plot_spacing(preds, baseline=np.zeros((20, 2)))
    with pytest.raises(ValueError, match="same number of outputs"):
        plot_spacing({0: np.zeros(4), 1: np.zeros((4, 2))})
    with pytest.raises(ValueError, match="missing or infinite"):
        plot_spacing({0: [1.0, np.nan]})
    with pytest.raises(ValueError, match="summarize"):
        plot_spacing({0: np.zeros((4, 2, 10))})
    with pytest.raises(ValueError, match="keys of predictions"):
        plot_spacing({"near": np.zeros(4)})
    with pytest.raises(ValueError, match="no spacings"):
        plot_spacing({})
    with pytest.raises(ValueError, match="no output named 'x'; give the outputs names"):
        plot_spacing(preds, output="x")
    with pytest.raises(ValueError, match="out of range"):
        plot_spacing(preds, output=1)
    with pytest.raises(TypeError, match="index or name"):
        plot_spacing(preds, output=1.5)
    with pytest.raises(ValueError, match="2 names for 1 outputs"):
        plot_spacing(preds, names=["a", "b"])


def _paired(n=200, seed=2):
    rng = np.random.default_rng(seed)
    before = rng.normal(0, 1, n)
    return before, {"A": before + 0.5 + rng.normal(0, 0.3, n), "B": before - 0.2 + rng.normal(0, 0.3, n)}


def _header(ax):
    return [t.get_text() for t in ax.texts]


def test_marginalization_groups_and_changes():
    before, after = _paired()
    fig = plot_marginalization(before, after)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Before", "A", "B"]
    da, db = after["A"] - before, after["B"] - before
    # The larger change, below 1, to three significant figures.
    assert 0.1 < da.mean() < 1 and abs(db.mean()) < da.mean()
    places = 3
    want = [f"{100 * (da > 0).mean():.0f}% up", f"Δ {da.mean():+.{places}f}",
            f"{100 * (db > 0).mean():.0f}% up", f"Δ {db.mean():+.{places}f}".replace("-", "−")]
    assert _header(ax) == want
    assert ax.get_xlabel() == "200 sequences"
    assert ax.get_ylabel() == "Prediction"
    # A dot at the mean of each group, and a line across at the mean before.
    dots = [c for c in ax.collections if isinstance(c, PathCollection) and c.get_offsets().shape[0] == 1]
    np.testing.assert_allclose([d.get_offsets()[0] for d in dots],
                               [[0, before.mean()], [1, after["A"].mean()], [2, after["B"].mean()]])
    guide = [ln for ln in ax.lines if len(set(ln.get_ydata())) == 1 and ln.get_linestyle() != "-"]
    assert len(guide) == 1 and guide[0].get_ydata()[0] == pytest.approx(before.mean())


def test_marginalization_single_insertion_is_labelled_after():
    before, after = _paired(n=30)
    ax = plot_marginalization(before, after["A"]).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Before", "After"]
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    segs = lines.get_segments()
    assert len(segs) == 30
    np.testing.assert_allclose([s[0][1] for s in segs], before)
    np.testing.assert_allclose([s[1][1] for s in segs], after["A"])


def test_marginalization_points_and_lines_by_size():
    def counts(ax):
        pts = [c for c in ax.collections if isinstance(c, PathCollection) and c.get_offsets().shape[0] > 1]
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        return len(pts), len(lines)

    before, after = _paired(n=40)
    assert counts(plot_marginalization(before, after).axes[0]) == (3, 0)
    assert counts(plot_marginalization(before, after["A"]).axes[0]) == (2, 1)
    assert counts(plot_marginalization(before, after["A"], points=False, lines=False).axes[0]) == (0, 0)
    before, after = _paired(n=400)
    assert counts(plot_marginalization(before, after["A"]).axes[0]) == (0, 0)
    assert counts(plot_marginalization(before, after["A"], points=True, lines=True).axes[0]) == (2, 1)
    with pytest.raises(ValueError, match="single insertion"):
        plot_marginalization(before, after, lines=True)


def test_marginalization_kinds():
    before, after = _paired(n=10_000)
    violin = plot_marginalization(before, after).axes[0]
    assert len([c for c in violin.collections if isinstance(c, PolyCollection)]) == 3
    box = plot_marginalization(before, after, kind="box").axes[0]
    assert len(box.patches) == 3
    with pytest.raises(ValueError, match="kind must be"):
        plot_marginalization(before, after, kind="strip")


def test_marginalization_with_constant_predictions():
    ax = plot_marginalization(np.zeros(20), {"none": np.zeros(20), "up": np.ones(20)}).axes[0]
    assert _header(ax) == ["0% up", "Δ 0", "100% up", "Δ +1.00"]


def test_marginalization_outputs():
    rng = np.random.default_rng(3)
    before = rng.normal(0, 1, (50, 2))
    after = before + [1.0, -1.0]
    with pytest.raises(ValueError, match="pick one with output"):
        plot_marginalization(before, after)
    ax = plot_marginalization(_Tensor(before), _Tensor(after), output="counts", names=["profile", "counts"]).axes[0]
    assert ax.get_ylabel() == "counts"
    assert _header(ax) == ["0% up", "Δ −1.00"]
    ax = plot_marginalization(before.tolist(), {"x": after}, output=0).axes[0]
    assert _header(ax) == ["100% up", "Δ +1.00"]


def test_marginalization_into_axes():
    before, after = _paired()
    fig, axes = plt.subplots(1, 2, layout="constrained")
    out = plot_marginalization(before, after, ax=axes[0])
    assert out is axes[0]
    fig.canvas.draw()


def test_marginalization_errors():
    before, after = _paired()
    with pytest.raises(ValueError, match="has shape \\(199, 1\\) and before \\(200, 1\\)"):
        plot_marginalization(before, after["A"][:-1])
    with pytest.raises(ValueError, match="after 'B' has shape"):
        plot_marginalization(before, {"A": after["A"], "B": after["B"][:5]})
    with pytest.raises(ValueError, match="no predictions"):
        plot_marginalization(before, {})
    with pytest.raises(ValueError, match="before is empty"):
        plot_marginalization([], [])
    with pytest.raises(ValueError, match="missing or infinite"):
        plot_marginalization(before, np.full(200, np.inf))
    with pytest.raises(ValueError, match="out of range"):
        plot_marginalization(before, after, output=3)


def test_signed():
    assert _signed([0.4041, 1.284, -0.0001]) == ["+0.40", "+1.28", "0"]
    assert _signed([0.0, 0.0]) == ["0", "0"]
    assert _signed([1234.5, -2.0]) == ["+1234", "−2"]
    assert _signed([0.00123]) == ["+0.00123"]
