"""One-hot arrays in and out of the functions that make synthetic sequences."""

import numpy as np
import polars as pl
import pytest
import universalmotif as um
from test_deep import _modisco_file, _modisco_inputs, _rc


def _seqs(n=5, length=31, seed=1):
    return um.create_sequences(seqnum=n, seqlen=length, seed=seed)


@pytest.mark.parametrize("length", [30, 31])
@pytest.mark.parametrize("site", ["TGACTCA", "TGACTC"])
def test_insert_sites_matches_the_strings(length, site):
    seqs = _seqs(length=length)
    x = um.one_hot(seqs)
    np.testing.assert_array_equal(um.insert_sites(x, site), um.one_hot(um.insert_sites(seqs, site)))
    both = um.insert_sites(x, ["GG", um.example_motifs()[3]], positions=[0, 20])
    np.testing.assert_array_equal(both, um.one_hot(um.insert_sites(seqs, ["GG", um.example_motifs()[3]],
                                                                   positions=[0, 20])))
    # Inputs are left alone.
    np.testing.assert_array_equal(x, um.one_hot(seqs))


def test_insert_sites_layouts_dtypes_and_ns():
    seqs = _seqs()
    want = um.one_hot(um.insert_sites(seqs, "TGANTCA"))
    last = um.insert_sites(um.one_hot(seqs, dtype=np.int8).transpose(0, 2, 1), "TGANTCA")
    assert last.shape == (5, 31, 4) and last.dtype == np.int8
    np.testing.assert_array_equal(last.transpose(0, 2, 1), want)
    single = um.insert_sites(um.one_hot(seqs[:1])[0], "TGANTCA")
    assert single.shape == (4, 31)
    np.testing.assert_array_equal(single, want[0])
    # The N leaves its column empty.
    assert want[:, :, 15].sum() == 0
    assert um.insert_sites(um.one_hot(seqs).astype(bool), "tga").dtype == bool
    with pytest.raises(ValueError, match="does not fit"):
        um.insert_sites(um.one_hot(seqs), "ACGT", positions=29)


def test_site_spacings_match_the_strings():
    seqs = _seqs(length=40)
    got = um.site_spacings(um.one_hot(seqs), "GATAAG", "CACGTG", spacings=[0, 3, 10])
    want = um.site_spacings(seqs, "GATAAG", "CACGTG", spacings=[0, 3, 10])
    assert list(got) == [0, 3, 10]
    for d in got:
        np.testing.assert_array_equal(got[d], um.one_hot(want[d]))


def test_shuffle_sequences_and_implant_motifs_match_the_strings():
    seqs = _seqs(length=60)
    seqs[2] = seqs[2][:20] + "NN" + seqs[2][22:]
    x = um.one_hot(seqs, dtype=np.int8)
    for kw in ({}, {"k": 3}, {"window": True, "window_size": 20, "window_overlap": 5}):
        got = um.shuffle_sequences(x, seed=4, **kw)
        assert got.dtype == np.int8
        np.testing.assert_array_equal(got, um.one_hot(um.shuffle_sequences(seqs, seed=4, **kw), dtype=np.int8))
    motif = um.example_motifs()[5]
    got, sites = um.implant_motifs(motif, x, seed=2, return_indices=True)
    want, want_sites = um.implant_motifs(motif, um.from_one_hot(x), seed=2, return_indices=True)
    np.testing.assert_array_equal(got, um.one_hot(want, dtype=np.int8))
    assert sites.equals(want_sites)
    with pytest.raises(ValueError, match="one-hot"):
        um.shuffle_sequences(x * 0 + 1)


def test_create_sequences_as_one_hot():
    x = um.create_sequences(seqnum=4, seqlen=25, seed=3, output="onehot", dtype=np.int8)
    assert x.shape == (4, 4, 25) and x.dtype == np.int8
    np.testing.assert_array_equal(x, um.one_hot(um.create_sequences(seqnum=4, seqlen=25, seed=3), dtype=np.int8))
    assert um.create_sequences("AA", seqnum=2, seqlen=10, output="onehot").shape == (2, 20, 10)
    assert um.create_sequences(seqnum=0, seqlen=10, output="onehot").shape == (0, 4, 10)
    with pytest.raises(ValueError, match="output"):
        um.create_sequences(output="arrays")


def test_shuffle_region():
    seqs = _seqs(n=3, length=40)
    out = um.shuffle_region(seqs, 10, 30, n=4, seed=1)
    assert len(out) == 3 and all(len(group) == 4 for group in out)
    for seq, group in zip(seqs, out):
        assert sum(s != seq for s in group) >= 3
        for s in group:
            assert s[:10] == seq[:10] and s[30:] == seq[30:]
            assert um.count_kmers(s[10:30], k=2).equals(um.count_kmers(seq[10:30], k=2))
    assert um.shuffle_region(seqs[0], 10, 30, n=4, seed=1) == out[0]
    named = um.shuffle_region(dict(zip("abc", seqs)), 10, 30, n=4, seed=1)
    assert list(named) == ["a", "b", "c"] and named["b"] == out[1]
    x = um.shuffle_region(um.one_hot(seqs), 10, 30, n=4, seed=1)
    assert x.shape == (3, 4, 4, 40)
    np.testing.assert_array_equal(x, um.one_hot([s for group in out for s in group]).reshape(3, 4, 4, 40))
    last = um.shuffle_region(um.one_hot(seqs).transpose(0, 2, 1), 10, 30, n=4, seed=1)
    np.testing.assert_array_equal(last, x.transpose(0, 1, 3, 2))
    assert um.shuffle_region(um.one_hot(seqs), 10, 30, n=0).shape == (3, 0, 4, 40)
    with pytest.raises(ValueError, match="does not fit"):
        um.shuffle_region(seqs, 10, 41)
    with pytest.raises(ValueError, match="end after"):
        um.shuffle_region(seqs, 10, 10)


def test_plot_spacing_takes_sequences_first():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = np.random.default_rng(1).normal(size=(6, 40))
    a = um.plot_spacing(preds, spacings=range(6))
    b = um.plot_spacing(preds.T, spacings=range(6), axis=1)
    for la, lb in zip(a.axes[0].lines, b.axes[0].lines):
        np.testing.assert_allclose(la.get_ydata(), lb.get_ydata())
    with pytest.raises(ValueError, match="axis"):
        um.plot_spacing(preds, spacings=range(6), axis=2)
    plt.close("all")


def test_read_seqlets_trimmed(tmp_path):
    pytest.importorskip("h5py")
    path = _modisco_file(tmp_path / "modisco.h5")
    full = um.read_seqlets(path, scores="projected")
    trimmed = um.read_seqlets(path, trim=0.3, scores="projected")
    widths = {m.name: m.width for m in um.read_modisco(path, trim_threshold=0.3)}
    assert (trimmed["end"] - trimmed["start"]).to_list() == [widths[m] for m in trimmed["motif"]]
    assert ((trimmed["start"] >= full["start"]) & (trimmed["end"] <= full["end"])).all()
    # Each trimmed seqlet still reads the input's letters where it now lies.
    inputs = _modisco_inputs()
    for row in trimmed.iter_rows(named=True):
        letters = inputs[row["sequence_i"]][row["start"]:row["end"]]
        assert row["match"] == (_rc(letters) if row["strand"] == "-" else letters)
    sums = [float(np.asarray(v).sum()) for v in trimmed["scores"]]
    np.testing.assert_allclose(sums, trimmed["score"], rtol=1e-5, atol=1e-6)
    assert um.read_seqlets(path, trim=None).equals(um.read_seqlets(path))
    with pytest.raises(ValueError, match="trim"):
        um.read_seqlets(path, trim=1.5)
    regions = pl.DataFrame({"chrom": ["chr1"] * 50, "start": np.arange(50) * 1000,
                            "end": np.arange(50) * 1000 + 100})
    genomic = um.read_seqlets(path, trim=0.3, regions=regions)
    np.testing.assert_array_equal(genomic["start"] - genomic["sequence_i"].cast(pl.Int64) * 1000, trimmed["start"])


def test_tensors_come_back_as_tensors():
    torch = pytest.importorskip("torch")
    seqs = _seqs(length=40)
    x = torch.from_numpy(um.one_hot(seqs))
    for out in (um.insert_sites(x, "TGACTCA"), um.shuffle_sequences(x, seed=1), um.implant_motifs(um.example_motifs()[5], x, seed=1),
                um.shuffle_region(x, 5, 30, n=2)):
        assert isinstance(out, torch.Tensor)
    assert all(isinstance(v, torch.Tensor) for v in um.site_spacings(x, "GG", "TT", spacings=[0, 1]).values())
