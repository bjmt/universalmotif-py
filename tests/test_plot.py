import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import universalmotif as um  # noqa: E402
from universalmotif._logo import _flip_negative, matrix_glyphs, position_glyphs  # noqa: E402
from universalmotif._plot import _attribution_matrix, _panel_glyphs, _prepare_motifs, _y_axis  # noqa: E402

from conftest import args, load_reference, motif_from_record, nums  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ref(key):
    return load_reference("plot.json", key)


@pytest.fixture
def flush_negative_letters(monkeypatch):
    # The R package stacks negative letters with no gaps between them, where
    # the port leaves the same gaps as above zero, so the reference
    # comparisons stack them as R does.
    monkeypatch.setattr("universalmotif._logo._flip_negative",
                        lambda glyphs, s, d, f, y_spacer=0.0: _flip_negative(glyphs, s, d, f))


def _motifs():
    return [motif_from_record(r) for r in _ref("motifs")]


def _check_glyphs(glyphs, want, ctx):
    ids = [g.letter_id for g in glyphs for _ in g.x]
    assert ids == want["letter_id"], ctx
    groups = [g.group for g in glyphs for _ in g.x]
    assert groups == want["group"], ctx
    x = np.concatenate([g.x for g in glyphs])
    y = np.concatenate([g.y for g in glyphs])
    np.testing.assert_allclose(x, nums(want["x"]), rtol=1e-12, atol=1e-13, err_msg=ctx)
    np.testing.assert_allclose(y, nums(want["y"]), rtol=1e-12, atol=1e-13, err_msg=ctx)


def test_logo_polygons_reference(flush_negative_letters):
    mats = _ref("logo_matrices")
    for case in _ref("logo"):
        m = mats[case["matrix"]]
        a = args(case["args"])
        glyphs = matrix_glyphs(
            np.array([nums(c) for c in m["columns"]]).T,
            m["rows"],
            min_height=a.get("min.height", 0.01),
            x_spacer=a.get("x.spacer", 0.04),
            y_spacer=a.get("y.spacer", 0.01),
            sort_positions=a.get("sort.positions", False),
            sort_positions_decreasing=a.get("sort.positions.decreasing", True),
            fit_to_height=a.get("fit.to.height"),
            flip_neg=a.get("flip.neg", False),
        )
        _check_glyphs(glyphs, case["polygons"], (case["matrix"], a))


def _raw_kwargs(a):
    return dict(
        use_type=a.get("use.type", "ICM"),
        rc=a.get("tryRC", True),
        min_overlap=a.get("min.overlap", 6),
        relative_entropy=a.get("relative_entropy", False),
        sort_by=a.get("sort.by", "none"),
        rc_text=a.get("RC.text", " [RC]"),
    )


def test_aligned_matrices_reference():
    dna = _motifs()
    rna = [um.create_motif(m.matrix, alphabet="RNA", type=m.type, name=m.name, nsites=m.nsites,
                           pseudocount=m.pseudocount, bkg=list(m.bkg.values())) for m in dna[:3]]
    for case in _ref("raw"):
        a = args(case["args"])
        ms = [(dna if case["set"] == "dna" else rna)[int(i) - 1] for i in nums(case["idx"])]
        got = um.view_motifs(ms, return_raw=True, **_raw_kwargs(a))
        assert list(got) == case["names"], a
        for (name, m), want in zip(got.items(), case["matrices"]):
            want = np.array([nums(c) for c in want], dtype=float).T
            # The R package's relative entropy is NaN for zero probabilities,
            # which the port takes as 0.
            want[np.isnan(want)] = 0.0
            np.testing.assert_allclose(m, want, rtol=1e-12, atol=1e-14, err_msg=f"{a} {name}")


def test_single_motif_polygons_reference(flush_negative_letters):
    ms = _motifs()
    for case in _ref("single"):
        m = ms[case["idx"] - 1]
        ut = case["use_type"]
        if "error" in case:
            with pytest.raises(ValueError, match="non-finite"):
                um.view_motifs(m, use_type=ut)
            continue
        _, mats, _, _, _ = _prepare_motifs(m, ut, True, 6, False, "none", " [RC]", 1)
        glyphs = _panel_glyphs(mats[0], list("ACGT"), ut, 0.01, 0.04, 0.01, ut not in ("PCM", "PPM"),
                               True, 1.0 if ut == "PPM" else None, False)
        _check_glyphs(glyphs, case["polygons"], (case["idx"], ut))


def test_multi_motif_polygons_reference():
    ms = _motifs()[:4]
    want = _ref("multi")
    names, mats, _, _, _ = _prepare_motifs(ms, "ICM", True, 6, False, "ic", " [RC]", 1)
    glyphs, motif_ids = [], []
    for name, m in zip(names, mats):
        g = _panel_glyphs(m, list("ACGT"), "ICM", 0.01, 0.04, 0.01, True, True, None, False)
        glyphs += g
        motif_ids += [name for gg in g for _ in gg.x]
    _check_glyphs(glyphs, want, "multi")
    assert motif_ids == want["motif"]


def test_view_motifs_draws():
    ms = _motifs()
    fig = um.view_motifs(ms[:4])
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes] == list(um.view_motifs(ms[:4], return_raw=True))
    fig = um.view_motifs([ms[0], ms[1], ms[3]], use_type="PWM", names_pos="right",
                         show_positions_once=False)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "1" in labels and "" in labels
    fig = um.view_motifs(ms[0], use_type="PPM")
    assert len(fig.axes) == 1 and fig.axes[0].get_ylabel() == "probability"
    colours = {tuple(c) for c in fig.axes[0].collections[0].get_facecolors().round(3)}
    assert len(colours) == 4
    with pytest.raises(ValueError, match="same alphabet"):
        um.view_motifs([um.create_motif("MKVL", alphabet="AA"), ms[0]])


def test_view_logo_draws():
    ax = um.view_logo(np.array([[1.0, -0.5], [0.2, 2.0]]), letters=["AA", "C"],
                      colour_scheme={"A": "red", "C": "blue"})
    assert not ax.axison
    assert len(ax.collections[0].get_paths()) == 6  # "AA" is two glyphs
    ax = um.view_logo({"A": [1, 0], "B": [0, 1]})
    assert len(ax.collections[0].get_paths()) == 2
    with pytest.raises(ValueError):
        um.view_logo(np.ones((2, 2)))
    with pytest.raises(ValueError):
        um.view_logo(np.ones((2, 2)), letters=["A", "C"], colour_scheme={"A": "red"})


def test_tree_plots():
    ms = _motifs()
    ms = [m.replace(family="bZIP" if "creb" in m.name else "other") for m in ms]
    tree = um.motif_tree(ms, min_overlap=4)
    for layout in ("rectangular", "slanted", "circular", "fan"):
        ax = tree.plot(layout=layout, groups="family", tip_size="icscore")
        assert ax.collections, layout
    ax = tree.plot(tip_logos=True, branch_length=True)
    assert len(ax.texts) == len(ms)
    assert len(ax.collections) == 1 + len(ms)
    ax = tree.plot(groups=["a", "a", "b", "b", "c", "c"], legend=True)
    assert ax.get_legend() is not None
    ax = tree.plot(groups="family", tip_size="icscore")
    sizes = ax.get_legend()
    assert sizes.get_title().get_text() == "icscore"
    values = [float(t.get_text()) for t in sizes.get_texts()]
    ics = [m.icscore for m in ms]
    assert len(values) >= 2 and min(ics) <= min(values) and max(values) <= max(ics)
    groups = [a for a in ax.artists if hasattr(a, "get_texts")]
    assert [t.get_text() for t in groups[0].get_texts()] == ["bZIP", "other"]
    assert tree.plot(tip_size="icscore", legend=False).get_legend() is None
    with pytest.raises(ValueError):
        tree.plot(layout="circular", tip_logos=True)
    with pytest.raises(ValueError):
        tree.plot(layout="radial")
    from_dist = um.motif_tree(np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=float))
    from_dist.plot()
    with pytest.raises(ValueError):
        from_dist.plot(tip_logos=True)


def test_plot_motif_peaks():
    seqs = um.create_sequences(seqnum=100, seqlen=200, seed=1)
    m = um.create_motif("TTGACGTCAA", nsites=20, pseudocount=1, name="creb")
    seqs = um.implant_motifs(m, seqs, positions=[[95 + i % 5] for i in range(100)], seed=2)
    peaks = um.motif_peaks(um.scan_sequences(m, seqs, pvalue=1e-3), 200, qvalue=1)
    fig = um.plot_motif_peaks(peaks)
    assert fig.axes[0].get_title().startswith("creb  (p = ")
    with pytest.raises(ValueError):
        um.plot_motif_peaks(peaks.drop("centres"))
    with pytest.raises(ValueError):
        um.plot_motif_peaks(peaks, motifs=["nothing"])
    empty = pl.DataFrame(schema=peaks.schema)
    with pytest.raises(ValueError):
        um.plot_motif_peaks(empty)


def test_top_logo_tick_stays_on_the_axis():
    # log2(6) = 2.58 bits would round up to a tick above the axis.
    fig = um.view_motifs(um.create_motif(6, alphabet="QWERTY", seed=1))
    assert list(fig.axes[0].get_yticks()) == [0.0, 1.0, 2.0]
    plt.close(fig)



def test_small_contributions_are_drawn():
    # Contribution weights are often far below 0.03, the smallest negative
    # letter once drawn, and the axis was padded by a fixed 0.1.
    cwm = np.array([[-0.02, 0.0, 0.004], [0.0, -0.012, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.001]])
    glyphs = _panel_glyphs(cwm, list("ACGT"), "CWM", 0.01, 0.04, 0.01, False, True, None, False)
    assert {g.letter_id for g in glyphs} >= {"1.A.1", "2.C.1", "3.A.1"}
    assert min(g.y.min() for g in glyphs) < -0.015
    ticks, ylim = _y_axis([cwm], "CWM")
    assert ticks == [-0.02, -0.01, 0.0]
    assert ylim[0] == pytest.approx(-0.02 - 0.05 * 0.024) and ylim[1] == pytest.approx(0.004)
    # Attributions at any scale keep their letters.
    seq = "ACGTACGTAC"
    tiny = np.linspace(1, 10, len(seq)) * 1e-5
    ax = um.view_attributions(tiny, seq)
    assert len(ax.collections[0].get_paths()) == len(seq)
    # Letters keep their shape, so gaps don't swallow them at small scales.
    big = um.view_attributions(tiny * 1e5, seq)
    ys = [np.ptp(p.vertices[:, 1]) for p in ax.collections[0].get_paths()]
    big_ys = [np.ptp(p.vertices[:, 1]) for p in big.collections[0].get_paths()]
    np.testing.assert_allclose(np.array(ys) * 1e5, big_ys, rtol=1e-6)


def test_negative_letters_are_spaced_like_positive_ones():
    for flip in (False, True):
        glyphs = position_glyphs(list("ACGT"), np.array([0.5, 0.3, -0.4, -0.2]), 1, 0.01, 0.04, 0.05,
                                 flip_neg=flip)
        ext = {}
        for g in glyphs:
            lo, hi = ext.get(g.letter_id, (np.inf, -np.inf))
            ext[g.letter_id] = (min(lo, g.y.min()), max(hi, g.y.max()))
        a, c, g_, t = (ext[f"1.{x}.1"] for x in "ACGT")
        assert a[0] - c[1] == pytest.approx(0.025)
        assert g_[0] - t[1] == pytest.approx(0.025)
        # Each stack spans the sum of its values.
        assert a[1] == pytest.approx(0.8) and t[0] == pytest.approx(-0.6)


def test_view_attributions():
    seq = "ACGTTTGACCAAGTNA"
    n = len(seq)
    per_pos = np.linspace(-1, 1, n)
    per_letter = np.arange(4 * n, dtype=float).reshape(4, n) - 2 * n
    # One score per position goes to the observed letter; N gets nothing.
    m = _attribution_matrix(per_pos, seq, "ACGT", "observed")
    assert m.shape == (4, n)
    assert m[3, 3] == per_pos[3] and m[:, 3].sum() == per_pos[3]
    assert not m[:, 14].any()
    # Per-letter scores keep only the observed letter's, unless all are drawn.
    m = _attribution_matrix(per_letter, seq, "ACGT", "observed")
    assert m[1, 1] == per_letter[1, 1] and m[0, 1] == 0
    assert np.array_equal(_attribution_matrix(per_letter, seq, "ACGT", "all"), per_letter)
    # (length, letters) arrays and a leading batch axis are accepted.
    assert np.array_equal(_attribution_matrix(per_letter.T, None, "ACGT", "all"), per_letter)
    assert np.array_equal(_attribution_matrix(per_letter[None], None, "ACGT", "all"), per_letter)

    hits = um.scan_sequences(um.example_motifs()[5], seq, pvalue=1e-2)
    ax = um.view_attributions(per_pos, seq, start=1000, hits=hits)
    assert ax.get_xticklabels()[0].get_text() == "1000"
    assert [t.get_text() for t in ax.texts] == hits["motif"].to_list()
    um.view_attributions(per_pos, seq, flip_neg=True)
    ax = um.view_attributions(per_letter, seq, kind="all", region=(4, 12))
    assert ax.get_xlim() == (0.5, 8.5)
    ax = um.view_attributions(per_letter, seq, kind="heatmap")
    assert len(ax.figure.axes) == 2  # the heatmap and its colour bar
    plt.close("all")

    with pytest.raises(ValueError, match="need the sequence"):
        um.view_attributions(per_pos)
    with pytest.raises(ValueError, match="one score per letter"):
        um.view_attributions(per_pos, seq, kind="all")
    with pytest.raises(ValueError, match="letters for"):
        um.view_attributions(per_pos, seq[:-1])
    with pytest.raises(ValueError, match="region"):
        um.view_attributions(per_pos, seq, region=(0, n + 1))
    with pytest.raises(ValueError, match="kind"):
        um.view_attributions(per_pos, seq, kind="bars")
    with pytest.raises(ValueError, match="missing or infinite"):
        um.view_attributions(np.full(n, np.nan), seq)
    plt.close("all")


def test_view_motifs_into_existing_axes():
    motifs = um.example_motifs()[:2]
    fig, (left, right) = plt.subplots(1, 2)
    ax = um.view_motifs(motifs[0], ax=left)
    assert ax is left and len(fig.axes) == 2
    assert left.get_ylabel() == "bits" and left.get_title() == motifs[0].name
    assert len(left.collections) > 0 and not right.collections
    with pytest.raises(ValueError, match="single motif"):
        um.view_motifs(motifs, ax=right)
