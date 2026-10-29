"""Double mutagenesis, epistasis between positions and maps of interactions."""

import itertools

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from matplotlib.colors import to_hex


def _changed(variant: str, seq: str) -> list[int]:
    return [i for i, (x, y) in enumerate(zip(variant, seq)) if x != y]


def test_batch_layout_and_counts():
    seq = "ACGNT"
    batch = um.pair_mutagenesis(seq)
    assert batch.shape == (1 + 5 + 10, 4, 5) and batch.dtype == np.float32
    x = um.from_one_hot(batch)
    assert x[0] == "ACGNT"
    # Complements, with the N taking the first letter of the alphabet.
    assert x[1:6] == ["TCGNT", "AGGNT", "ACCNT", "ACGAT", "ACGNA"]
    pairs = [_changed(v, seq) for v in x[6:]]
    assert pairs == [list(p) for p in itertools.combinations(range(5), 2)]
    assert x[6] == "TGGNT" and x[-1] == "ACGAA"
    assert um.from_one_hot(um.pair_mutagenesis(seq, alt="transition")[1:6]) == [
        "GCGNT", "ATGNT", "ACANT", "ACGAT", "ACGNC"]
    assert um.from_one_hot(um.pair_mutagenesis("ACGU", alphabet="ACGU")[1:5], alphabet="ACGU") == [
        "UCGU", "AGGU", "ACCU", "ACGA"]
    # One letter per position, as a string or a list.
    given = um.from_one_hot(um.pair_mutagenesis(seq, alt="GGAAA")[1:6])
    assert given == ["GCGNT", "AGGNT", "ACANT", "ACGAT", "ACGNA"]
    assert np.array_equal(um.pair_mutagenesis(seq, alt=list("GGAAA")), um.pair_mutagenesis(seq, alt="GGAAA"))
    # Every substitution: three per known letter and four for the N, and every
    # pair of them at different positions.
    every = um.from_one_hot(um.pair_mutagenesis(seq, alt="all"))
    n_pairs = 16 * 15 // 2 - 4 * 3 - 6
    assert len(every) == 1 + 16 + n_pairs
    assert every[1:17] == um.from_one_hot(um.mutagenesis(seq)[1:])
    assert len(set(every)) == len(every)
    assert all(len(_changed(v, seq)) == 2 for v in every[17:])
    firsts = [_changed(v, seq) for v in every[17:]]
    assert firsts == sorted(firsts, key=lambda p: p[0])


def test_region_and_batches():
    seq = "ACGTACGTAC"
    batch = um.pair_mutagenesis(seq, region=(3, 7))
    assert batch.shape == (1 + 4 + 6, 4, 10)
    x = um.from_one_hot(batch)
    assert all(set(_changed(v, seq)) <= set(range(3, 7)) for v in x)
    assert x[1] == "ACGAACGTAC"
    parts = um.pair_mutagenesis(seq, region=(3, 7), batch_size=4, dtype=np.float64)
    assert len(parts) == 3 and "11 in all" in repr(parts)
    chunks = list(parts)
    assert [len(c) for c in chunks] == [4, 4, 3] and chunks[0].dtype == np.float64
    assert np.array_equal(np.concatenate(chunks), batch)
    # Each pass makes the batches anew.
    assert np.array_equal(np.concatenate(list(parts)), batch)
    every = um.pair_mutagenesis(seq, region=(3, 7), alt="all", batch_size=50)
    assert np.array_equal(np.concatenate(list(every)), um.pair_mutagenesis(seq, region=(3, 7), alt="all"))


def test_effects_map_back_to_variants():
    seq = "TTGACGCAT"
    batch = um.pair_mutagenesis(seq, region=(1, 8))
    x = um.from_one_hot(batch)
    rng = np.random.default_rng(3)
    value = {v: rng.normal() for v in x}
    preds = np.array([value[v] for v in x])
    epistasis, single = um.interaction_effects(preds, seq, region=(1, 8))
    assert epistasis.shape == (7, 7) and single.shape == (7,)
    assert np.isnan(np.diag(epistasis)).all()
    assert np.array_equal(epistasis, epistasis.T, equal_nan=True)
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}

    def sub(s, *pos):
        s = list(s)
        for p in pos:
            s[p] = comp[s[p]]
        return "".join(s)

    for i in range(1, 8):
        assert single[i - 1] == pytest.approx(value[sub(seq, i)] - value[seq])
        for j in range(i + 1, 8):
            want = value[sub(seq, i, j)] - value[sub(seq, i)] - value[sub(seq, j)] + value[seq]
            assert epistasis[i - 1, j - 1] == pytest.approx(want)


def test_additive_models_have_no_epistasis():
    seq = "ACGTTGCAAC"
    weights = np.random.default_rng(5).integers(-9, 10, (4, len(seq))).astype(float)
    for alt in (None, "transition", "all"):
        batch = um.pair_mutagenesis(seq, alt=alt, dtype=np.float64)
        preds = np.einsum("nlp,lp->n", batch, weights)
        epistasis, single = um.interaction_effects(preds, seq, alt=alt)
        off = ~np.eye(len(seq), dtype=bool)
        assert (epistasis[off] == 0).all() and np.isnan(np.diag(epistasis)).all()
        if alt == "all":
            assert np.array_equal(single, um.mutagenesis_effects(
                np.einsum("nlp,lp->n", um.mutagenesis(seq, dtype=np.float64), weights), seq))
            full, _ = um.interaction_effects(preds, seq, alt="all", summary=None)
            assert full.shape == (4, 10, 4, 10)
            assert (full[np.isfinite(full)] == 0).all()
            # Each pair of positions has nine pairs of substitutions.
            assert np.isfinite(full[:, 0, :, 1]).sum() == 9 and np.isnan(full[:, 0, :, 0]).all()


def _site_model(batch, sites):
    """The product of the number of letters each site has in place."""
    out = np.ones(len(batch))
    for start, letters in sites:
        ref = um.one_hot(letters, dtype=float)[0]
        out *= np.einsum("nlp,lp->n", batch[:, :, start:start + len(letters)], ref)
    return out


def test_multiplicative_interaction_is_recovered():
    seq = "CCGATAAGCCTTCACGTGTT"
    sites = [(2, "GATAAG"), (12, "CACGTG")]
    batch = um.pair_mutagenesis(seq, dtype=np.float64)
    epistasis, single = um.interaction_effects(_site_model(batch, sites), seq)
    # Losing a letter of one site costs as many letters as the other has, and
    # losing one of each costs one less than the two apart.
    assert single[3] == -6 and single[0] == 0 and single[14] == -6
    expected = np.zeros((20, 20))
    expected[2:8, 12:18] = expected[12:18, 2:8] = 1
    off = ~np.eye(20, dtype=bool)
    assert np.array_equal(epistasis[off], expected[off])
    # The same over part of the sequence.
    part = um.pair_mutagenesis(seq, region=(5, 15), dtype=np.float64)
    sub, _ = um.interaction_effects(_site_model(part, sites), seq, region=(5, 15))
    assert np.array_equal(sub[~np.eye(10, dtype=bool)], epistasis[5:15, 5:15][~np.eye(10, dtype=bool)])


def test_multiple_outputs_and_summaries():
    seq = "CGATAAGCACGTG"
    sites = [(1, "GATAAG"), (7, "CACGTG")]
    batch = um.pair_mutagenesis(seq, dtype=np.float64)
    preds = _site_model(batch, sites)
    one, s_one = um.interaction_effects(preds, seq)
    three, s_three = um.interaction_effects(np.stack([preds, -preds, 2 * preds], axis=1), seq)
    assert three.shape == (3, 13, 13) and s_three.shape == (3, 13)
    assert np.array_equal(three[1], -one, equal_nan=True) and np.array_equal(s_three[2], 2 * s_one)
    every = um.pair_mutagenesis(seq, alt="all", dtype=np.float64)
    preds = _site_model(every, sites) + np.random.default_rng(2).normal(0, 0.1, len(every))
    full, effects = um.interaction_effects(preds, seq, alt="all", summary=None)
    assert effects.shape == (4, 13)
    pairs = full.transpose(1, 3, 0, 2).reshape(13, 13, 16)
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        want = {"mean": np.nanmean(pairs, axis=2), "max": np.nanmax(pairs, axis=2),
                "min": np.nanmin(pairs, axis=2)}
    idx = np.nanargmax(np.where(np.isnan(pairs), -1, np.abs(pairs)), axis=2)
    want["strongest"] = np.take_along_axis(pairs, idx[..., None], axis=2)[..., 0]
    off = ~np.eye(13, dtype=bool)
    for summary, expected in want.items():
        got, _ = um.interaction_effects(preds, seq, alt="all", summary=summary)
        assert np.isnan(np.diag(got)).all()
        np.testing.assert_allclose(got[off], expected[off])
    two, effects2 = um.interaction_effects(np.stack([preds, preds * 3], axis=1), seq, alt="all",
                                           summary="strongest")
    assert two.shape == (2, 13, 13) and effects2.shape == (2, 4, 13)
    np.testing.assert_allclose(two[1][off], 3 * want["strongest"][off])
    assert um.interaction_effects(np.stack([preds, preds], axis=1), seq, alt="all",
                                  summary=None)[0].shape == (2, 4, 13, 4, 13)


def test_errors():
    seq = "ACGTA"
    with pytest.raises(ValueError, match="expected 16 predictions"):
        um.interaction_effects(np.zeros(15), seq)
    with pytest.raises(ValueError, match="expected 7 predictions"):
        um.interaction_effects(np.zeros(15), seq, region=(1, 4))
    with pytest.raises(ValueError, match="one letter per position"):
        um.pair_mutagenesis(seq, alt="GG")
    with pytest.raises(ValueError, match="not a letter"):
        um.pair_mutagenesis(seq, alt="GGXGG")
    with pytest.raises(ValueError, match="own letter at position 2"):
        um.pair_mutagenesis(seq, alt="TTGAT")
    with pytest.raises(ValueError, match="for DNA and RNA"):
        um.pair_mutagenesis("ACDEF", alphabet="ACDEFGHIKLMNPQRSTVWY")
    assert len(um.pair_mutagenesis("ACDEF", alphabet="ACDEFGHIKLMNPQRSTVWY", alt="GGGGG")) == 16
    with pytest.raises(ValueError, match="summary"):
        um.interaction_effects(np.zeros(16), seq, summary="median")
    with pytest.raises(ValueError, match="region"):
        um.pair_mutagenesis(seq, region=(2, 9))
    with pytest.raises(ValueError, match="batch_size"):
        um.pair_mutagenesis(seq, batch_size=0)
    with pytest.raises(TypeError, match="single sequence"):
        um.pair_mutagenesis(["ACGT"])
    with pytest.raises(ValueError, match="repeated"):
        um.pair_mutagenesis(seq, alphabet="ACGA")


@pytest.fixture
def toy():
    seq = "TTCGATAAGCCTTCACGTGTTAA"
    sites = [(3, "GATAAG"), (13, "CACGTG")]
    batch = um.pair_mutagenesis(seq, dtype=np.float64)
    epistasis, single = um.interaction_effects(_site_model(batch, sites), seq)
    hits = pl.DataFrame({"motif": ["GATA", "E-box", "late"], "start": [3, 13, 21], "end": [9, 19, 23]})
    yield seq, epistasis, single, hits
    plt.close("all")


def test_plot_square(toy):
    seq, epistasis, single, hits = toy
    fig = um.plot_interactions(epistasis, seq, start=1000, hits=hits)
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.5, 23.5) and ax.get_ylim() == (23.5, 0.5)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "1000" and all(int(t) % 5 == 0 for t in labels)
    image = ax.images[0]
    # The scale is centred on 0 and runs to the 99th percentile of the sizes.
    assert image.get_clim() == (-1.0, 1.0)
    top, left, bar = ax.child_axes
    assert left.get_ylim() == (23.5, 0.5) and left.get_ylabel() == "Position"
    assert [t.get_text() for t in left.get_yticklabels()] == labels
    assert bar.get_ylabel() == "Interaction"
    # Each pair of hits is outlined on the map, and the hits are named once.
    boxes = [p for p in ax.patches if not p.get_fill()]
    assert len(boxes) == 9
    assert min((b.get_x(), b.get_y()) for b in boxes) == (3.5, 3.5)
    assert [t.get_text() for t in top.texts] == ["GATA", "E-box", "late"]
    assert len(left.patches) == 3 and not left.texts
    fig = um.plot_interactions(epistasis, seq, scores=-single, vmax=0.5, label="Epistasis")
    assert fig.axes[0].images[0].get_clim() == (-0.5, 0.5)
    assert fig.axes[0].child_axes[2].get_ylabel() == "Epistasis"
    fig = um.plot_interactions(epistasis, seq, hits=hits, colour="#AA3377", figsize=(5, 5))
    assert tuple(fig.get_size_inches()) == (5, 5)
    assert to_hex(fig.axes[0].patches[0].get_edgecolor()) == "#aa3377"


def test_plot_regions_and_inputs(toy):
    seq, epistasis, single, hits = toy
    # A region of the whole map.
    fig = um.plot_interactions(epistasis, seq, start=10, region=(15, 30), hits=hits)
    ax = fig.axes[0]
    assert ax.images[0].get_array().shape == (15, 15) and ax.get_xlim() == (0.5, 15.5)
    # Hits are cut to the region, and those outside it are left out.
    assert [t.get_text() for t in ax.child_axes[0].texts] == ["GATA", "E-box"]
    # A map of only part of the sequence, with scores of the sequence or of the part.
    part = epistasis[2:20, 2:20]
    for scores in (-single, -single[2:20]):
        fig = um.plot_interactions(part, seq, scores=scores, region=(2, 20), hits=hits)
        assert fig.axes[0].images[0].get_array().shape == (18, 18)
    # Any square matrix, such as a Jacobian of attributions with a diagonal.
    jac = np.random.default_rng(0).normal(size=(30, 30))
    ax = um.plot_interactions(jac, ax=plt.subplots()[1])
    assert ax.get_xlabel() == "Position" and len(ax.child_axes) == 1
    assert um.plot_interactions(jac[None], triangle=True).axes[0].get_xlabel() == "Position"


def test_plot_triangle(toy):
    seq, epistasis, single, hits = toy
    fig = um.plot_interactions(epistasis, seq, scores=-single, start=50, hits=hits, triangle=True)
    ax = fig.axes[0]
    mesh = ax.collections[0]
    xy = mesh.get_coordinates()
    # Positions run along the bottom, and the distance between them upward.
    assert tuple(xy[0, 0]) == (0.5, 0.0) and tuple(xy[0, 23]) == (12.0, 23.0)
    assert tuple(xy[23, 23]) == (23.5, 0.0)
    assert ax.get_ylim() == (0.0, 23.0) and ax.get_aspect() == 0.5
    below, _ = ax.child_axes
    fig.canvas.draw()
    assert below.get_xlabel() == "Position" and below.get_xticklabels()[0].get_text() == "50"
    assert below.get_position().y1 < ax.get_position().y0
    # Each hit with itself and with each later one.
    diamonds = [p for p in ax.patches if not p.get_fill()]
    assert len(diamonds) == 6
    assert np.allclose(diamonds[0].get_xy()[:4], [(3.5, 0), (6.5, 6), (9.5, 0), (6.5, -6)])
    fig = um.plot_interactions(epistasis, triangle=True, max_distance=8)
    assert fig.axes[0].get_ylim() == (0.0, 8.0)
    assert fig.get_figheight() < um.plot_interactions(epistasis, triangle=True).get_figheight()


def test_plot_errors(toy):
    seq, epistasis, _, _ = toy
    with pytest.raises(ValueError, match="square"):
        um.plot_interactions(epistasis[:, :5])
    with pytest.raises(ValueError, match="pick one"):
        um.plot_interactions(np.stack([epistasis, epistasis]))
    with pytest.raises(ValueError, match="give the region"):
        um.plot_interactions(epistasis[:10, :10], seq)
    with pytest.raises(ValueError, match="cover the 10 positions"):
        um.plot_interactions(epistasis[:10, :10], seq, region=(0, 12))
    with pytest.raises(ValueError, match="for a sequence of 5"):
        um.plot_interactions(epistasis, seq[:5])
    with pytest.raises(ValueError, match="region must lie"):
        um.plot_interactions(epistasis, seq, region=(5, 40))
    with pytest.raises(ValueError, match="one value per position"):
        um.plot_interactions(epistasis, scores=np.ones((4, 10)))
    with pytest.raises(ValueError, match="vmax"):
        um.plot_interactions(epistasis, vmax=0)
    with pytest.raises(ValueError, match="max_distance"):
        um.plot_interactions(epistasis, triangle=True, max_distance=0)
    with pytest.raises(ValueError, match="infinite"):
        um.plot_interactions(np.full((3, 3), np.inf))
