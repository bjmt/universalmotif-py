import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import universalmotif as um  # noqa: E402
from universalmotif._logo import _flip_negative, matrix_glyphs, motif_glyphs, position_glyphs  # noqa: E402
from universalmotif._plot import (  # noqa: E402
    AA_COLOURS,
    DNA_COLOURS,
    RNA_COLOURS,
    _attribution_matrix,
    _density,
    _nice_step,
    _panel_glyphs,
    _prepare_motifs,
    _y_axis,
)

from conftest import args, load_reference, motif_from_record, nums  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _ref(key):
    return load_reference("plot.json", key)


@pytest.fixture
def flush_negative_letters(monkeypatch):
    # The R package stacks negative letters with no gaps between them and
    # the largest next to zero, where the port leaves the same gaps as above
    # zero and mirrors the letters above, so the reference comparisons
    # stack them as R does.
    monkeypatch.setattr("universalmotif._logo._flip_negative",
                        lambda glyphs, s, d, f, y_spacer=0.0: _flip_negative(glyphs, s, not d, f))


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
    um.view_attributions(per_pos, seq, flip_neg=False)
    ax = um.view_attributions(per_letter, seq, kind="all", region=(4, 12))
    assert ax.get_xlim() == (0.5, 8.5)
    ax = um.view_attributions(per_letter, seq, kind="heatmap")
    assert len(ax.child_axes) == 1  # the colour bar, outside the panel
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


def test_view_variant():
    seq = "ACGTTGACTCAGT"
    ref = np.zeros(len(seq))
    ref[5:11] = [0.3, 0.6, 0.5, 0.6, 0.6, 0.4]
    alt = ref * 0.2
    fig = um.view_variant(ref, alt, seq, position=107, alt="A", start=100)
    top, middle, bottom = fig.axes
    assert top.get_ylim() == middle.get_ylim()
    assert middle.get_ylabel() == "Alternative (A)"
    heights = [p.get_height() for p in bottom.containers[0]]
    assert np.allclose(heights, alt - ref)
    assert [t.get_text() for t in bottom.get_xticklabels()] == ["100", "105", "110"]
    assert not top.spines["bottom"].get_visible()
    fig = um.view_variant(ref, alt, seq, position=7, alt="A", region=(4, 12))
    assert len(fig.axes[2].containers[0]) == 8
    with pytest.raises(ValueError, match="kind"):
        um.view_variant(ref, alt, seq, position=7, alt="A", kind="heatmap")
    with pytest.raises(ValueError, match="single letter"):
        um.view_variant(ref, alt, seq, position=7, alt="N")
    with pytest.raises(ValueError, match="include position"):
        um.view_variant(ref, alt, seq, position=7, alt="A", region=(0, 5))


def test_attribution_letters_keep_their_heights():
    # Short letters are not dropped and the rest scaled up to fill in.
    scores = np.array([[0.02, 1.0], [0.009, 0.0], [0.009, 0.0], [0.009, 0.0]])
    ax = um.view_attributions(scores, "AC", kind="all")
    heights = [np.ptp(p.vertices[:, 1]) for p in ax.collections[0].get_paths()]
    assert max(h for h in heights if h < 0.5) < 0.021


def test_negative_letters_mirror_positive_ones():
    glyphs = position_glyphs(list("ACGT"), np.array([-0.5, -0.1, 0.3, 0.6]), 1, 0.01, 0.04, 0.0,
                             sort_positions=True)
    ext = {}
    for g in glyphs:
        lo, hi = ext.get(g.letter_id, (np.inf, -np.inf))
        ext[g.letter_id] = (min(lo, g.y.min()), max(hi, g.y.max()))
    # The largest letter on each side is furthest from zero.
    assert ext["1.T.1"][1] == pytest.approx(0.9) and ext["1.G.1"][0] == pytest.approx(0.0)
    assert ext["1.A.1"][0] == pytest.approx(-0.6) and ext["1.C.1"][1] == pytest.approx(0.0)


# --- Shared helpers for the tests below --------------------------------------


def _glyph_extents(ax, colours=None):
    """The (x min, x max, y min, y max) of each letter drawn on a logo, in
    drawing order, with the letter it is coloured as when ``colours`` maps
    letters to distinct colours."""
    coll = ax.collections[0]
    faces = [matplotlib.colors.to_hex(c) for c in coll.get_facecolors()]
    if len(faces) == 1:
        faces = faces * len(coll.get_paths())
    by_colour = {matplotlib.colors.to_hex(v): k for k, v in (colours or {}).items()}
    out = []
    for p, face in zip(coll.get_paths(), faces):
        v = p.vertices
        out.append((by_colour.get(face), v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))
    return out


def _column_extents(ax):
    """The lowest and highest point of the letters of each logo column,
    keyed by the column's centre."""
    ext = {}
    for _, x0, x1, y0, y1 in _glyph_extents(ax):
        col = round((x0 + x1) / 2)
        lo, hi = ext.get(col, (np.inf, -np.inf))
        ext[col] = (min(lo, y0), max(hi, y1))
    return ext


def _span(patch):
    """The x range of a shaded span from axvspan."""
    x = patch.get_x()
    return (x, x + patch.get_width())


_RGB = {"A": "#ff0000", "C": "#00ff00", "G": "#0000ff", "T": "#000000"}


# --- view_logo ----------------------------------------------------------------


def test_view_logo_column_heights():
    m = np.array([[1.0, 3.0, 0.5], [1.0, 1.0, 0.25]])
    ax = um.view_logo(m, letters=["A", "C"])
    # Each column is as tall as its values, from zero up.
    ext = _column_extents(ax)
    assert sorted(ext) == [1, 2, 3]
    for col, total in zip((1, 2, 3), m.sum(axis=0)):
        assert ext[col] == pytest.approx((0.0, total))
    assert ax.get_xlim() == pytest.approx((0.499 - 0.02 * 3.002, 3.501 + 0.02 * 3.002))
    # fit_to_height scales every column to the same height.
    ax = um.view_logo(m, letters=["A", "C"], fit_to_height=2.0)
    assert all(e == pytest.approx((0.0, 2.0)) for e in _column_extents(ax).values())


def test_view_logo_negative_values_and_flip():
    m = np.array([[1.0, -2.0], [0.5, 0.3]])
    ax = um.view_logo(m, letters=["A", "C"])
    ext = _column_extents(ax)
    assert ext[1] == pytest.approx((0.0, 1.5))
    assert ext[2] == pytest.approx((-2.0, 0.3))
    # The y-axis is scaled to the letters, below zero too.
    lo, hi = ax.get_ylim()
    assert lo <= -2.0 and hi >= 1.5
    # Letters below zero are drawn upside down by default: their outline is
    # the mirror image of the one drawn with flip_neg=False.
    neg = np.array([[-1.0]])
    up = um.view_logo(neg, letters=["A"], flip_neg=False).collections[0].get_paths()[0].vertices
    down = um.view_logo(neg, letters=["A"]).collections[0].get_paths()[0].vertices
    np.testing.assert_allclose(down[:, 0], up[:, 0])
    np.testing.assert_allclose(down[:, 1], -1.0 - up[:, 1], atol=1e-12)


def test_view_logo_sorting_and_min_height():
    m = np.array([[0.2], [0.5], [0.3], [0.005]])
    stacks = {}
    for key, kw in (("input", {}), ("decreasing", {"sort_positions": True}),
                    ("increasing", {"sort_positions": True, "sort_positions_decreasing": False})):
        ax = um.view_logo(m, letters=list("ACGT"), colour_scheme=_RGB, **kw)
        glyphs = _glyph_extents(ax, _RGB)
        # T is below min_height, so it is left out.
        assert {g[0] for g in glyphs} == {"A", "C", "G"}
        stacks[key] = [g[0] for g in sorted(glyphs, key=lambda g: -g[3])]
    assert stacks["input"] == ["A", "C", "G"]
    assert stacks["decreasing"] == ["C", "G", "A"]
    assert stacks["increasing"] == ["A", "G", "C"]


def test_view_logo_multi_character_letters():
    single = _glyph_extents(um.view_logo(np.array([[1.0]]), letters=["A"]))
    pair = _glyph_extents(um.view_logo(np.array([[1.0]]), letters=["AA"]))
    assert len(single) == 1 and len(pair) == 2
    (_, lx0, lx1, ly0, ly1), (_, rx0, rx1, ry0, ry1) = sorted(pair, key=lambda g: g[1])
    # Side by side in the column, as wide as each other and as tall as the
    # value, together spanning about the width of a single letter.
    assert 0.5 < lx0 < lx1 <= rx0 < rx1 < 1.5
    assert lx1 - lx0 == pytest.approx(rx1 - rx0)
    assert (ly0, ly1) == pytest.approx((0.0, 1.0)) and (ry0, ry1) == pytest.approx((0.0, 1.0))
    assert lx0 == pytest.approx(single[0][1]) and rx1 == pytest.approx(single[0][2], abs=0.01)
    with pytest.raises(ValueError, match="colour_scheme has no colours for: B, c"):
        um.view_logo(np.ones((2, 1)), letters=["A", "Bc"], colour_scheme={"A": "red"})


def test_view_logo_inputs():
    # A motif is drawn as its matrix is, here a PPM with columns of height 1.
    motif = um.create_motif("TTGACA")
    ax = um.view_logo(motif)
    assert len(ax.collections[0].get_paths()) == 6
    assert all(e == pytest.approx((0.0, 1.0)) for e in _column_extents(ax).values())
    assert {matplotlib.colors.to_hex(c) for c in ax.collections[0].get_facecolors()} == {"#000000"}
    # Nothing at all to draw leaves the axes empty.
    ax = um.view_logo(np.zeros((2, 3)), letters=["A", "C"])
    assert not ax.collections
    with pytest.raises(ValueError, match="one entry per matrix row"):
        um.view_logo(np.ones((2, 3)), letters=["A", "C", "G"])


def test_view_logo_letter_without_outline():
    with pytest.warns(UserWarning, match='"é" has no outline'):
        ax = um.view_logo(np.array([[1.0], [1.0]]), letters=["A", "é"])
    assert len(ax.collections[0].get_paths()) == 1


# --- view_motifs --------------------------------------------------------------


def test_view_motifs_argument_errors():
    m = um.example_motifs()[3]
    with pytest.raises(ValueError, match="use_type must be one of"):
        um.view_motifs(m, use_type="BITS")
    with pytest.raises(ValueError, match="sort_by must be"):
        um.view_motifs([m, m], sort_by="name")
    with pytest.raises(ValueError, match="no motifs given"):
        um.view_motifs([])
    with pytest.raises(ValueError, match="every motif to be a CWM"):
        um.view_motifs(m, use_type="CWM")
    with pytest.raises(ValueError, match="names_pos must be"):
        um.view_motifs(m, names_pos="left")
    with pytest.raises(ValueError, match="not in the alphabet: U"):
        um.view_motifs(m, colour_scheme={**DNA_COLOURS, "U": "red"})


def test_view_motifs_alignment_and_position_labels():
    ms = um.example_motifs()[:3]
    raw = um.view_motifs(ms, use_type="PPM", return_raw=True)
    assert list(raw) == ["TATA-box [RC]", "AP-1", "CRE"]
    assert {m.shape for m in raw.values()} == {(4, 11)}
    # The flipped motif is the reverse complement of the original, padded
    # with empty columns into the common frame.
    tata = raw["TATA-box [RC]"]
    used = np.flatnonzero(tata.sum(axis=0))
    ppm = np.asarray(um.convert_type(ms[0], "PPM").matrix)
    np.testing.assert_allclose(tata[:, used], ppm[::-1, ::-1], rtol=1e-12)
    # Numbered under every logo, each motif counts its own positions over
    # the columns it fills.
    fig = um.view_motifs(ms, show_positions_once=False)
    for ax, m in zip(fig.axes, raw.values()):
        used = set(np.flatnonzero(m.sum(axis=0)).tolist())
        labels = [t.get_text() for t in ax.get_xticklabels()]
        want = []
        for i in range(11):
            want.append(str(sum(1 for j in used if j <= i)) if i in used else "")
        assert labels == want
    # Without rc nothing is flipped, and rc_text marks flipped motifs.
    assert list(um.view_motifs(ms, rc=False, return_raw=True)) == ["TATA-box", "AP-1", "CRE"]
    assert "TATA-box (rc)" in um.view_motifs(ms, rc_text=" (rc)", return_raw=True)


def test_view_motifs_display_order():
    ms = um.example_motifs()[:3]
    by_ic = list(um.view_motifs(ms, sort_by="ic", return_raw=True))
    ics = {m.name: m.icscore for m in ms}
    assert [ics[n.removesuffix(" [RC]")] for n in by_ic] == sorted(ics.values(), reverse=True)
    fig = um.view_motifs(ms, sort_by="ic")
    assert [ax.get_title() for ax in fig.axes] == by_ic
    # Clustering by similarity puts a copy of a motif next to it.
    a, b, c = ms
    names = list(um.view_motifs([b, a, c, a.replace(name="copy")], sort_by="similarity",
                                return_raw=True))
    i, j = (k for k, n in enumerate(names) if n.startswith(("TATA", "copy")))
    assert abs(i - j) == 1 and len(names) == 4


def test_view_motifs_names_and_positions():
    ms = um.example_motifs()[:3]

    def labelled(ax):
        return any(t.label1.get_visible() for t in ax.xaxis.get_major_ticks())

    fig = um.view_motifs(ms)
    assert [labelled(ax) for ax in fig.axes] == [False, False, True]
    assert all(ax.get_shared_x_axes().joined(ax, fig.axes[0]) for ax in fig.axes)
    fig = um.view_motifs(ms, show_positions=False)
    assert not any(labelled(ax) for ax in fig.axes)
    fig = um.view_motifs(ms, show_positions_once=False)
    assert all(labelled(ax) for ax in fig.axes)
    # Names go beside the logos with names_pos="right", but a single logo is
    # always named above.
    fig = um.view_motifs(ms, names_pos="right")
    assert all(ax.get_title() == "" for ax in fig.axes)
    assert [[t.get_text() for t in ax.texts] for ax in fig.axes] == [[n] for n in
                                                                    um.view_motifs(ms, return_raw=True)]
    fig = um.view_motifs(ms[1], names_pos="right")
    assert fig.axes[0].get_title() == "AP-1" and not fig.axes[0].texts
    fig = um.view_motifs(ms, show_names=False)
    assert all(ax.get_title() == "" and not ax.texts for ax in fig.axes)
    fig = um.view_motifs(ms[1], text_size=20, figsize=(5, 2))
    assert fig.axes[0].title.get_fontsize() == 20 and tuple(fig.get_size_inches()) == (5, 2)


def test_view_motifs_count_axis():
    m = um.create_motif("TTGACGTCAA", nsites=20, pseudocount=1)
    ax = um.view_motifs(m, use_type="PCM").axes[0]
    assert ax.get_ylabel() == "counts"
    assert list(ax.get_yticks()) == [0.0, 10.0, 20.0] and ax.get_ylim() == (0.0, 20.0)
    # Every column holds one letter of all 20 sites, drawn a hair short of
    # the top so that it stays inside the axis.
    ext = _column_extents(ax)
    assert len(ext) == 10
    assert all(e == pytest.approx((0.0, 20 * 0.999)) for e in ext.values())


def test_view_motifs_contribution_weights():
    cwm = um.create_motif(np.array([[0.5, -0.2, 0.1, 0], [0, 0, 0.3, -0.1]]).T, alphabet="DNA",
                          type="CWM")
    raw = um.view_motifs(cwm, use_type="CWM", return_raw=True)
    np.testing.assert_array_equal(next(iter(raw.values())), cwm.matrix)
    ax = um.view_motifs(cwm, use_type="CWM").axes[0]
    assert ax.get_ylabel() == "contribution"
    assert any(np.allclose(line.get_ydata(), 0) for line in ax.lines), "a zero line is drawn"
    ext = _column_extents(ax)
    assert ext[1][0] < -0.15 and ext[1][1] > 0.5
    assert ext[2][0] < -0.05 and ext[2][1] == pytest.approx(0.3, abs=0.01)


@pytest.mark.xfail(strict=True, reason="fit_to_height is silently ignored for CWMs, where ICMs "
                   "and PWMs warn that it is")
def test_view_motifs_fit_to_height_ignored_for_cwm_warns():
    cwm = um.create_motif(np.array([[0.5, -0.2, 0.1, 0], [0, 0, 0.3, -0.1]]).T, alphabet="DNA",
                          type="CWM")
    with pytest.warns(UserWarning, match="fit_to_height is ignored"):
        um.view_motifs(cwm, use_type="CWM", fit_to_height=1.0)


def test_view_motifs_fit_to_height_warning():
    m = um.example_motifs()[3]
    plain = [p.vertices for p in um.view_motifs(m).axes[0].collections[0].get_paths()]
    for ut in ("ICM", "PWM"):
        with pytest.warns(UserWarning, match='ignored for use_type "ICM" and "PWM"'):
            fig = um.view_motifs(m, use_type=ut, fit_to_height=1.0)
        if ut == "ICM":
            got = [p.vertices for p in fig.axes[0].collections[0].get_paths()]
            assert len(got) == len(plain) and all(np.allclose(a, b) for a, b in zip(got, plain))
    # Counts are scaled to the height asked for.
    pcm = um.create_motif("TTGACGTCAA", nsites=20, pseudocount=1)
    ax = um.view_motifs(pcm, use_type="PCM", fit_to_height=5.0).axes[0]
    assert all(e == pytest.approx((0.0, 5 * 0.999)) for e in _column_extents(ax).values())


def test_view_motifs_default_colours():
    def faces(fig):
        return {matplotlib.colors.to_hex(c).upper() for c in fig.axes[0].collections[0].get_facecolors()}

    rna = um.create_motif("ACGU", alphabet="RNA", nsites=10)
    assert faces(um.view_motifs(rna, use_type="PPM")) == set(RNA_COLOURS.values())
    aa = um.create_motif("GQKDA", alphabet="AA", nsites=10)
    assert faces(um.view_motifs(aa, use_type="PPM")) == set(AA_COLOURS.values())
    custom = um.create_motif(["XYZ", "XYZ"], alphabet="XYZ")
    assert len(faces(um.view_motifs(custom, use_type="PPM"))) == 3
    # A colour scheme may leave letters out, which are then drawn grey.
    dna = um.create_motif("ACGT", nsites=10)
    assert faces(um.view_motifs(dna, use_type="PPM", colour_scheme={"A": "#FF0000"})) == {"#FF0000", "#7F7F7F"}


# --- Trees ----------------------------------------------------------------------


def _small_tree():
    dist = [[0.0, 0.1, 0.8], [0.1, 0.0, 0.7], [0.8, 0.7, 0.0]]
    return um.motif_tree(dist, labels=["a", "b", "c"])


def _text_positions(ax):
    return {t.get_text(): tuple(float(v) for v in t.get_position()) for t in ax.texts}


def test_tree_rectangular_geometry():
    tree = _small_tree()
    assert tree.order == [2, 0, 1]
    # Tips are stacked from the top in dendrogram order, and aligned at the
    # depth of the deepest tip in a cladogram.
    ax = tree.plot()
    assert _text_positions(ax) == {"c": (2.04, 2.0), "a": (2.04, 1.0), "b": (2.04, 0.0)}
    assert ax.get_ylim() == (-0.6, 2.6) and not ax.axison
    # With branch lengths, tips sit at half the root's height, as in the
    # Newick string, and each child hangs off its parent's vertical line.
    ax = tree.plot(branch_length=True)
    segs = [np.asarray(s) for s in ax.collections[0].get_segments()]
    assert len(segs) == 8
    tips = sorted(tuple(s[1]) for s in segs if s[1][0] == pytest.approx(0.4))
    assert tips == [(0.4, 0.0), (0.4, 1.0), (0.4, 2.0)]
    assert any(np.allclose(s, [[0.0, 0.5], [0.35, 0.5]]) for s in segs)
    assert {k: v[1] for k, v in _text_positions(ax).items()} == {"c": 2.0, "a": 1.0, "b": 0.0}
    assert not tree.plot(labels=False).texts


def test_tree_slanted_and_polar_layouts():
    tree = _small_tree()
    segs = tree.plot(layout="slanted").collections[0].get_segments()
    assert len(segs) == 4 and all(len(s) == 2 for s in segs)

    def tip_angles(ax):
        return {k: round(np.degrees(np.arctan2(y, x)) % 360) for k, (x, y) in _text_positions(ax).items()}

    # A circle leaves a share of the circle to each tip, and a fan leaves a
    # gap of open_angle degrees.
    ax = tree.plot(layout="circular")
    assert tip_angles(ax) == {"c": 30, "a": 270, "b": 150}
    assert ax.get_aspect() == 1.0
    lo, hi = ax.get_xlim()
    assert lo == -hi and ax.get_ylim() == (lo, hi)
    assert tip_angles(tree.plot(layout="fan", open_angle=90)) == {"c": 45, "a": 270, "b": 135}


def test_tree_groups_and_colours():
    tree = _small_tree()
    ax = tree.plot(groups={"a": "x", "b": "x", "c": "y"}, colours={"x": "#ff0000"})
    lines = ax.collections[0]
    colours = [matplotlib.colors.to_hex(c) for c in lines.get_colors()]
    red = [np.asarray(s) for s, c in zip(lines.get_segments(), colours) if c == "#ff0000"]
    # The tips of group x and the branch joining them to the root, but not
    # the root's branch to c, which is in group y.
    assert len(red) == 6
    assert all(s[:, 1].max() < 2.0 for s in red)
    assert {c for s, c in zip(lines.get_segments(), colours) if np.asarray(s)[:, 1].max() == 2.0} == {
        matplotlib.colors.to_hex(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][1])}
    handles = ax.get_legend().legend_handles
    assert matplotlib.colors.to_hex(handles[0].get_color()) == "#ff0000"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y"]
    with pytest.raises(ValueError, match="groups has no value for c"):
        tree.plot(groups={"a": "x", "b": "x"})
    with pytest.raises(ValueError, match="one value per motif"):
        tree.plot(groups=["x", "y"])
    with pytest.raises(ValueError, match="needs a tree made by motif_tree"):
        tree.plot(groups="family")


def test_tree_tip_sizes_from_pvalues():
    ms = [m.replace(pval=p) for m, p in zip(um.example_motifs()[:4], [1e-2, 1e-5, None, 1e-8])]
    ax = um.motif_tree(ms).plot(tip_size="pval")
    sizes = ax.get_legend()
    assert sizes.get_title().get_text() == "-log10(pval)"
    values = [float(t.get_text()) for t in sizes.get_texts()]
    assert 2 <= min(values) and max(values) <= 8
    # Tips are scaled by -log10(p) between the smallest and largest, and a
    # motif with no P-value has no tip.
    np.testing.assert_allclose(ax.collections[1].get_sizes(), [15, 57.5, 0, 100])


def test_tree_tip_logos_sit_left_of_labels():
    ms = um.example_motifs()[:4]
    ax = um.motif_tree(ms).plot(tip_logos=True, tip_logo_width=0.1)
    label_x = {t.get_position()[0] for t in ax.texts}
    assert len(label_x) == 1
    logo_x = max(p.vertices[:, 0].max() for c in ax.collections[1:] for p in c.get_paths())
    assert logo_x < label_x.pop()


def test_tree_legends_are_fitted():
    tree = _small_tree()
    # A figure made by the plot is widened to take in the legend.
    ax = tree.plot(groups=["g1", "g1", "g2"])
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    assert fig.get_size_inches()[0] > 6
    assert ax.get_legend().get_window_extent(renderer).x1 <= fig.bbox.x1
    # Axes passed in are narrowed instead, leaving the figure as it is.
    fig, ax = plt.subplots(figsize=(4, 3))
    before = ax.get_position().bounds
    tree.plot(groups=["g1", "g1", "g2"], ax=ax)
    after = ax.get_position().bounds
    assert tuple(fig.get_size_inches()) == (4, 3)
    assert after[0] == before[0] and after[2] < before[2]
    assert ax.get_legend().get_window_extent(fig.canvas.get_renderer()).x1 <= fig.bbox.x1
    # Axes with room for the legend are left alone.
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_axes((0.05, 0.1, 0.3, 0.8))
    tree.plot(groups=["g1", "g1", "g2"], ax=ax)
    assert ax.get_position().bounds == pytest.approx((0.05, 0.1, 0.3, 0.8))


# --- Peaks, proximity and matched backgrounds ------------------------------------


def _peaks():
    return pl.DataFrame({
        "motif": ["a", "b", "c"],
        "pvalue": [1e-3, 0.5, 0.01],
        "enrichment": [2.0, None, 1.5],
        "best_window": [20, None, 10],
        "best_centre": [100.0, None, 50.0],
        "centres": [[95, 100, 105, 100], [1, 2], []],
        "seq_length": [200, 200, 200],
    })


def test_plot_motif_peaks_panels():
    fig = um.plot_motif_peaks(_peaks())
    assert len(fig.axes) == 4
    assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    a, b, c, _ = fig.axes
    assert [ax.get_title() for ax in (a, b, c)] == [
        "a  (p = 0.001 | 2.0x)", "b  (p = 0.5 | NAx)", "c  (p = 0.01 | 1.5x)"]
    # The best window is shaded around its centre, behind a histogram of
    # every hit centre, over the length of the sequences.
    assert _span(a.patches[0]) == (90.0, 110.0)
    assert sum(p.get_height() for p in a.patches[1:]) == 4
    assert a.get_xlim() == (-1.0, 200.0)
    assert a.get_xlabel() == "Hit centre position" and a.get_ylabel() == "Count"
    assert sum(p.get_height() for p in b.patches) == 2, "no window, so no shading"
    assert len(c.patches) == 1 and _span(c.patches[0]) == (45.0, 55.0)
    fig = um.plot_motif_peaks(_peaks(), motifs=["c", "a"], ncol=3)
    assert [ax.get_title()[0] for ax in fig.axes] == ["a", "c"]
    # Without seq_length the axis fits the hits.
    ax = um.plot_motif_peaks(_peaks().drop("seq_length"), motifs=["b"]).axes[0]
    assert ax.get_xlim()[1] < 10


def _proximity():
    return pl.DataFrame({
        "motif": ["a", "b", "c"],
        "pvalue": [1e-4, 0.2, 0.05],
        "best_dist": [50, None, 10],
        "dist_to_nearest": [[-40.0, -5.0, 0.0, 5.0, 30.0, None, float("inf")], [100.0, -300.0], None],
    })


def test_plot_motif_proximity_panels():
    fig = um.plot_motif_proximity(_proximity())
    assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    a, b, c, _ = fig.axes
    assert a.get_title() == "a  (p = 0.0001)"
    assert a.get_xlabel() == "Signed distance to nearest anchor (bp)"
    # The zone is shaded on both sides of the anchor, and missing and
    # infinite distances are left out of the histogram.
    assert _span(a.patches[0]) == (-50.0, 50.0)
    assert sum(p.get_height() for p in a.patches[1:]) == 5
    assert sum(p.get_height() for p in b.patches) == 2, "no zone, so no shading"
    assert [_span(p) for p in c.patches] == [(-10.0, 10.0)], "no distances, so no histogram"
    # On a log scale, distances are absolute and shifted by one.
    a = um.plot_motif_proximity(_proximity(), motifs=["a"], log_x=True).axes[0]
    assert a.get_xscale() == "log"
    assert a.get_xlabel() == "Distance to nearest anchor (bp, log10)"
    assert _span(a.patches[0]) == (1.0, 51.0)
    bars = a.patches[1:]
    assert sum(p.get_height() for p in bars) == 5
    assert bars[0].get_x() == pytest.approx(1.0)
    assert bars[-1].get_x() + bars[-1].get_width() == pytest.approx(41 * 1.01)


def test_plot_motif_proximity_errors():
    with pytest.raises(ValueError, match="no dist_to_nearest column"):
        um.plot_motif_proximity(_proximity().drop("dist_to_nearest"))
    with pytest.raises(ValueError, match="no rows"):
        um.plot_motif_proximity(_proximity().head(0))
    with pytest.raises(ValueError, match="none of the requested motifs"):
        um.plot_motif_proximity(_proximity(), motifs=["z"])


def test_density_bandwidth_matches_r():
    # The bandwidths of bw.nrd0() in R, from the standard deviation for 1:10
    # and from the interquartile range for c(1:9, 100).
    for x, bw in ((np.arange(1.0, 11.0), 1.719286404692283),
                  (np.r_[np.arange(1.0, 10.0), 100.0], 1.906997944137898)):
        grid, dens = _density(x)
        assert len(grid) == 512 and grid[0] == x.min() and grid[-1] == x.max()
        want = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(axis=1)
        want /= len(x) * bw * np.sqrt(2 * np.pi)
        np.testing.assert_allclose(dens, want, rtol=1e-10)
    # A constant has no spread to estimate a density from.
    grid, dens = _density(np.array([3.0, 3.0]))
    assert grid.tolist() == [3.0] and dens.tolist() == [0.0]


def test_plot_match_bkg_from_sequences():
    targets = {"t1": "GGGCAT", "t2": "ATATAT", "t3": "GCGCGCAT"}
    bkg = {"u1": "AAAAAA", "u2": "GGCCAT", "u3": "ATGCAA"}
    fig = um.plot_match_bkg(targets, bkg)
    gc, length = fig.axes
    assert gc.get_title() == "GC fraction" and length.get_title() == "Sequence length"
    assert [line.get_label() for line in gc.lines] == ["target", "matched"]
    np.testing.assert_allclose(gc.lines[0].get_ydata(), _density(np.array([4 / 6, 0, 6 / 8]))[1])
    assert len(gc.collections) == 2, "both densities are filled"
    # The background lengths are all the same, which is marked with a line.
    assert list(length.lines[1].get_xdata()) == [6.0, 6.0]
    assert [t.get_text() for t in gc.get_legend().get_texts()] == ["target", "matched"]
    assert len(um.plot_match_bkg(targets, bkg, by=["length"]).axes) == 1
    with pytest.raises(ValueError, match="give the targets and the background"):
        um.plot_match_bkg(targets)
    with pytest.raises(ValueError, match="by must name"):
        um.plot_match_bkg(targets, bkg, by=["gc", "cpg"])


def test_plot_match_bkg_from_indices():
    indices = pl.DataFrame({
        "target_i": [0, 1, 2], "target_gc": [0.4, 0.5, 0.6], "target_length": [100, 120, 90],
        "target_signal": [1.0, 2.0, 4.0], "universe_i": [5, 7, 9], "universe_gc": [0.41, 0.52, 0.58],
        "universe_length": [100, 118, 95], "universe_signal": [1.1, 2.5, 3.0],
        "universe_name": ["x", "y", "z"],
    })
    fig = um.plot_match_bkg(indices=indices)
    assert [ax.get_title() for ax in fig.axes] == ["GC fraction", "Sequence length", "signal"]
    np.testing.assert_allclose(fig.axes[2].lines[1].get_ydata(), _density(np.array([1.1, 2.5, 3.0]))[1])
    fig = um.plot_match_bkg(indices=indices, by=["signal", "gc"])
    assert [ax.get_title() for ax in fig.axes] == ["GC fraction", "signal"]
    with pytest.raises(ValueError, match="no target_\\*/universe_\\* column pairs"):
        um.plot_match_bkg(indices=indices.select("target_i", "universe_i"))
    with pytest.raises(ValueError, match="no target_\\*/universe_\\* column pairs"):
        um.plot_match_bkg(indices=indices, by=["cpg"])


# --- Attributions and variants ------------------------------------------------------


def test_view_attributions_letter_heights():
    seq = "ACGTNA"
    scores = np.array([0.5, -0.25, 1.0, 0.0, 0.7, -0.4])
    ax = um.view_attributions(scores, seq)
    # Each observed letter is as tall as its score, on its side of zero, and
    # N and zero scores draw nothing.
    ext = _column_extents(ax)
    assert sorted(ext) == [1, 2, 3, 6]
    for col in (1, 2, 3, 6):
        s = scores[col - 1]
        assert ext[col] == pytest.approx((min(s, 0.0), max(s, 0.0)))
    assert ax.get_ylim() == pytest.approx((-0.4 - 0.08 * 1.4, 1.0 + 0.08 * 1.4))
    assert ax.get_ylabel() == "Attribution" and ax.get_xlabel() == "Position"
    assert ax.get_xlim() == (0.5, 6.5)
    assert {matplotlib.colors.to_hex(c).upper() for c in ax.collections[0].get_facecolors()} == {
        DNA_COLOURS[x] for x in "ACG"}


def test_view_attributions_all_letters_and_heatmap():
    seq = "ACGTACGT"
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 1, (4, len(seq)))
    # Columns with letters on both sides of zero.
    scores[:, 2:6] = [[0.6, -0.3, 0.2, -0.9], [-0.4, 0.5, 0.1, 0.3], [0.2, -0.7, -0.5, 0.4], [-0.1, 0.8, 0.7, -0.2]]
    ax = um.view_attributions(scores, seq, kind="all", region=(2, 6), colour_scheme=_RGB)
    glyphs = _glyph_extents(ax, _RGB)
    assert len(glyphs) == 16
    # Columns 3 to 6 of the scores, with the largest letters furthest from
    # zero on both sides.
    for col in range(1, 5):
        s = scores[:, col + 1]
        here = [g for g in glyphs if round((g[1] + g[2]) / 2) == col]
        above = sorted((g for g in here if g[3] > -1e-9), key=lambda g: -g[4])
        below = sorted((g for g in here if g[3] <= -1e-9), key=lambda g: g[3])
        assert [g[0] for g in above] == ["ACGT"[i] for i in np.argsort(-s) if s[i] > 0]
        assert [g[0] for g in below] == ["ACGT"[i] for i in np.argsort(s) if s[i] < 0]
        assert max(g[4] for g in here) == pytest.approx(max(s[s > 0].sum(), 0.0))
        assert min(g[3] for g in here) == pytest.approx(min(s[s < 0].sum(), 0.0))
    ax = um.view_attributions(scores, seq, kind="heatmap", ylabel="Effect")
    image = ax.images[0]
    np.testing.assert_array_equal(image.get_array(), scores)
    top = np.abs(scores).max()
    assert image.get_clim() == (-top, top)
    assert [t.get_text() for t in ax.get_yticklabels()] == list("ACGT")
    # The sequence's letters are marked with dots.
    dots = ax.collections[0].get_offsets()
    np.testing.assert_array_equal(dots, [[j + 1, "ACGT".index(c)] for j, c in enumerate(seq)])
    assert ax.child_axes[0].get_ylabel() == "Effect"


def test_view_attributions_other_alphabets_and_axes():
    seq = "ACGUA"
    ax = um.view_attributions(np.ones(5) * 0.1, seq, alphabet="ACGU")
    assert matplotlib.colors.to_hex(ax.collections[0].get_facecolors()[3]).upper() == RNA_COLOURS["U"]
    _, (left, right) = plt.subplots(1, 2)
    assert um.view_attributions(np.ones(5) * 0.1, seq, alphabet="ACGU", ax=right) is right
    assert not left.collections
    with pytest.raises(ValueError, match="repeated letters"):
        um.view_attributions(np.ones(5), seq, alphabet="ACGA")
    with pytest.raises(ValueError, match=r"shape \(length,\) or \(4, length\)"):
        um.view_attributions(np.ones((2, 4, 5)), seq, alphabet="ACGU", kind="all")


def test_view_attributions_hits_in_a_region():
    seq = "ACGTTTGACCAAGTCA"
    hits = pl.DataFrame({"start": [0, 5, 10], "end": [3, 7, 14], "motif": ["early", "mid", "late"]})
    ax = um.view_attributions(np.full(len(seq), 0.1), seq, start=100, region=(104, 112), hits=hits)
    # Hits are counted along the sequence from 0, so the first lies before
    # the region and the last is cut off at its end.
    assert [_span(p) for p in ax.patches] == [(1.5, 3.5), (6.5, 8.5)]
    assert [t.get_text() for t in ax.texts] == ["mid", "late"]
    ax = um.view_attributions(np.full(len(seq), 0.1), seq, hits=hits.drop("motif"))
    assert len(ax.patches) == 3 and not ax.texts
    with pytest.raises(ValueError, match="hits must have a 'end' column"):
        um.view_attributions(np.full(len(seq), 0.1), seq, hits=hits.drop("end"))


def test_view_attributions_position_ticks():
    assert [_nice_step(s) for s in (10, 12, 13, 30, 300, 2500)] == [1, 1, 2, 5, 50, 500]
    n = 300
    ax = um.view_attributions(np.full(n, 0.1), "A" * n, start=1000, region=(1010, 1290))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(p) for p in range(1050, 1290, 50)]
    np.testing.assert_array_equal(ax.get_xticks(), [p - 1010 + 1 for p in range(1050, 1290, 50)])


def test_view_variant_panels():
    seq = "ACGTTGACTCAGT"
    ref = np.zeros(len(seq))
    ref[5:11] = [0.3, 0.6, 0.5, 0.6, 0.6, 0.4]
    alt = ref * 0.2
    alt[2] = 0.5
    fig = um.view_variant(ref, alt, seq, position=7, alt="A", region=(2, 12))
    top, middle, bottom = fig.axes
    assert top.get_ylim() == middle.get_ylim()
    assert top.get_ylim()[1] >= 0.6 and top.get_ylabel() == "Reference"
    # Bars rise where the alternative scores higher and fall where it scores
    # lower, coloured to match.
    bars = bottom.containers[0]
    diff = (alt - ref)[2:12]
    np.testing.assert_allclose([p.get_height() for p in bars], diff)
    colours = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in bars]
    assert colours == ["#C0504D" if d >= 0 else "#4F81BD" for d in diff]
    assert bottom.get_xlim() == top.get_xlim() and bottom.get_ylabel() == "Difference"
    # The changed position is shaded across all three panels.
    band = fig.artists[0]
    assert band.get_x() == 7 - 2 + 0.5 and band.get_width() == 1
    assert band.get_y() == pytest.approx(bottom.get_position().y0)
    assert band.get_y() + band.get_height() == pytest.approx(top.get_position().y1)
    # The alternative logo draws the new letter.
    fill = {matplotlib.colors.to_hex(c).upper() for c in middle.collections[0].get_facecolors()}
    assert DNA_COLOURS["A"] in fill
    with pytest.raises(ValueError, match="position must lie within 100 to 113"):
        um.view_variant(ref, alt, seq, position=7, alt="A", start=100)


def test_view_variant_all_letters():
    seq = "ACGTACGT"
    rng = np.random.default_rng(4)
    ref, alt = rng.normal(0, 1, (4, 8)), rng.normal(0, 1, (4, 8))
    fig = um.view_variant(ref, alt, seq, position=3, alt="G", kind="all")
    np.testing.assert_allclose([p.get_height() for p in fig.axes[2].containers[0]],
                               alt.sum(axis=0) - ref.sum(axis=0))
    assert len(fig.axes[0].collections[0].get_paths()) == 32


# --- Letter layout ------------------------------------------------------------------


def test_motif_glyphs_edge_cases():
    assert motif_glyphs(np.zeros((4, 3)), list("ACGT"), "CWM") == []
    with pytest.raises(ValueError, match="use_type must be one of"):
        motif_glyphs(np.ones((4, 3)), list("ACGT"), "BITS")
