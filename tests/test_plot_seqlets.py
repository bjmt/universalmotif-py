import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rc(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _onehot(seq):
    return np.array([[c == b for c in seq] for b in "ACGT"], dtype=np.float32)


def _seqlets(n=6, width=10, seed=1):
    """A seqlet table like um.read_seqlets(..., scores="hypothetical") gives,
    of one pattern with GATA in the middle, half of them on the - strand."""
    rng = np.random.default_rng(seed)
    match, strand, scores = [], [], []
    for i in range(n):
        letters = list(rng.choice(list("ACGT"), width))
        letters[3:7] = "GATA"
        seq = "".join(letters)
        hyp = rng.normal(0, 0.05, (4, width)) + 0.2 * (i + 1) * _onehot(seq)
        minus = i % 2 == 1
        match.append(seq)
        strand.append("-" if minus else "+")
        # Scores are stored along the genome.
        scores.append(hyp[::-1, ::-1] if minus else hyp)
    scores = np.array(scores, dtype=np.float32)
    observed = np.array([s[::-1, ::-1] if st == "-" else s for s, st in zip(scores, strand)])
    observed = observed * np.array([_onehot(m) for m in match])
    return pl.DataFrame({
        "motif": ["pos_patterns.pattern_0"] * n,
        "motif_i": pl.Series([0] * n, dtype=pl.UInt32),
        "strand": strand,
        "score": observed.sum(axis=(1, 2)).astype(float),
        "match": match,
        "scores": scores,
    })


def _images(fig):
    return [im for ax in fig.axes for im in ax.get_images()]


def test_plot_seqlets_panels():
    table = _seqlets()
    fig = um.plot_seqlets(table, "pos_patterns.pattern_0")
    assert len(fig.axes) == 4
    heat, letters = _images(fig)
    assert heat.get_array().shape == (6, 10) and letters.get_array().shape == (6, 10, 3)
    # The colour scale is centred on zero.
    lo, hi = heat.get_clim()
    assert lo == -hi and hi > 0
    assert fig.axes[2].get_title(loc="left") == "pos_0"
    texts = [t.get_text() for t in fig.axes[2].texts]
    assert texts == ["6 seqlets"]
    # Every seqlet reads along the pattern, so GATA lines up in every row.
    colours = letters.get_array()[:, 3:7]
    assert np.allclose(colours, colours[:1])
    assert fig.axes[0].get_xlim() == (0.5, 10.5)

    fig = um.plot_seqlets(table, "pos_patterns.pattern_0", letters=False, logo=False, title="GATA")
    assert len(fig.axes) == 1 and fig.axes[0].get_title(loc="left") == "GATA"
    fig = um.plot_seqlets(table, "pos_patterns.pattern_0", logo=False)
    assert [t.get_text() for t in fig.axes[1].get_legend().get_texts()] == list("ACGT")


def test_plot_seqlets_contributions_and_order():
    table = _seqlets()
    fig = um.plot_seqlets(table, "pos_patterns.pattern_0", letters=False, logo=False)
    image = np.asarray(_images(fig)[0].get_array())
    # The strongest seqlet is drawn first, and the - strand seqlets are
    # turned to read along the pattern.
    order = np.argsort(-np.abs(table["score"].to_numpy()), kind="stable")
    for row, i in zip(image, order):
        s = table["scores"].to_numpy()[i]
        if table["strand"][int(i)] == "-":
            s = s[::-1, ::-1]
        np.testing.assert_allclose(row, (s * _onehot(table["match"][int(i)])).sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(image.sum(axis=1), table["score"].to_numpy()[order], rtol=1e-5)

    fig = um.plot_seqlets(table, "pos_patterns.pattern_0", sort_by=None, letters=False, logo=False)
    np.testing.assert_allclose(np.asarray(_images(fig)[0].get_array()).sum(axis=1),
                               table["score"].to_numpy(), rtol=1e-5)
    assert fig.axes[0].get_ylabel() == "Seqlet"
    # Without a score column the total contribution is used.
    fig = um.plot_seqlets(table.drop("score"), "pos_patterns.pattern_0", letters=False, logo=False)
    np.testing.assert_allclose(np.asarray(_images(fig)[0].get_array()).sum(axis=1),
                               table["score"].to_numpy()[order], rtol=1e-5)


def test_plot_seqlets_subsampling():
    table = _seqlets(n=50)
    fig = um.plot_seqlets(table, "pos_patterns.pattern_0", max_seqlets=10, vmax=0.5)
    heat = _images(fig)[0]
    assert heat.get_array().shape == (10, 10) and heat.get_clim() == (-0.5, 0.5)
    assert [t.get_text() for t in fig.axes[2].texts] == ["50 seqlets, 10 drawn"]
    # Rows are picked evenly from the strongest to the weakest.
    ranked = np.sort(np.abs(table["score"].to_numpy()))[::-1]
    drawn = np.abs(np.asarray(heat.get_array()).sum(axis=1))
    np.testing.assert_allclose(drawn[[0, -1]], ranked[[0, -1]], rtol=1e-5)
    assert heat.get_extent()[2] == 50


def test_plot_seqlets_checks():
    table = _seqlets()
    with pytest.raises(ValueError, match="no scores column"):
        um.plot_seqlets(table.drop("scores"), "pos_patterns.pattern_0")
    with pytest.raises(ValueError, match="no 'match' column"):
        um.plot_seqlets(table.drop("match"), "pos_patterns.pattern_0")
    with pytest.raises(ValueError, match="no seqlets of a pattern named 'pattern_0'.*pos_patterns.pattern_0"):
        um.plot_seqlets(table, "pattern_0")
    with pytest.raises(ValueError, match="max_seqlets"):
        um.plot_seqlets(table, "pos_patterns.pattern_0", max_seqlets=0)
    with pytest.raises(ValueError, match="no 'pvalue' column"):
        um.plot_seqlets(table, "pos_patterns.pattern_0", sort_by="pvalue")
    with pytest.raises(ValueError, match="numeric"):
        um.plot_seqlets(table, "pos_patterns.pattern_0", sort_by="strand")
    with pytest.raises(ValueError, match="10 letters"):
        um.plot_seqlets(table.with_columns(match=pl.lit("GATA")), "pos_patterns.pattern_0")


def test_seqlets_from_a_modisco_file(tmp_path):
    h5py = pytest.importorskip("h5py")

    rng = np.random.default_rng(4)
    inputs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(5)]
    path = tmp_path / "modisco.h5"
    with h5py.File(path, "w") as f:
        for group, n in (("pos_patterns", 12), ("neg_patterns", 4)):
            g = f.create_group(group).create_group("pattern_0")
            example, start = rng.integers(0, 5, n), rng.integers(0, 50, n)
            rc = rng.random(n) < 0.5
            seqs = [inputs[e][s:s + 10] for e, s in zip(example, start)]
            seqs = [_rc(x) if r else x for x, r in zip(seqs, rc)]
            onehot = np.array([_onehot(x).T for x in seqs])
            hyp = rng.normal(0, 1, (n, 10, 4)).astype(np.float32)
            s = g.create_group("seqlets")
            for key, value in (("example_idx", example), ("start", start), ("end", start + 10),
                               ("is_revcomp", rc), ("sequence", onehot), ("hypothetical_contribs", hyp),
                               ("contrib_scores", hyp * onehot), ("n_seqlets", np.array([n]))):
                s.create_dataset(key, data=value)
    seqlets = um.read_seqlets(path, scores="hypothetical")
    fig = um.plot_seqlets(seqlets, "neg_patterns.pattern_0")
    assert _images(fig)[0].get_array().shape == (4, 10)
    # Patterns can be named by their short labels too.
    fig = um.plot_seqlets(seqlets, "neg_0")
    assert _images(fig)[0].get_array().shape == (4, 10)
    assert fig.axes[2].get_title(loc="left") == "neg_0"
    fig = um.plot_seqlets(seqlets, "pos_patterns.pattern_0", letters=False)
    assert len(fig.axes) == 2

    # The table is a hit table for the other plots.
    um.plot_hits(seqlets.drop("scores"), inputs)
    row = seqlets.row(0, named=True)
    seq = inputs[row["sequence_i"]]
    window = (max(0, row["start"] - 5), min(60, row["end"] + 5))
    scores = np.zeros((4, 60), dtype=np.float32)
    scores[:, row["start"]:row["end"]] = np.asarray(row["scores"])
    ax = um.view_attributions(scores, seq, kind="all", region=window, hits=seqlets[:1])
    assert ax.texts[0].get_text() == "pos_patterns.pattern_0"


def test_bits_axis_runs_to_two():
    fig = um.plot_seqlets(_seqlets(), "pos_patterns.pattern_0")
    bits = next(ax for ax in fig.axes if ax.get_ylabel() == "Bits")
    assert list(bits.get_yticks()) == [0, 1, 2]
    assert bits.spines["left"].get_bounds() == (0.0, 2.0)
