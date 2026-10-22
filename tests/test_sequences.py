import io
import warnings

import numpy as np
import polars as pl
import pytest

import universalmotif as um

from conftest import args, assert_motif_matches, load_reference, nums


def _ref(key):
    return load_reference("sequences.json", key)


def _as_list(x):
    return x if isinstance(x, list) else [x]


def test_klets_reference():
    for case in _ref("klets"):
        assert um.get_klets(case["letters"], case["k"]) == case["klets"]
    for case in _ref("count"):
        got = um.count_klets(case["string"], case["k"], case["alph"])
        assert got["klets"].to_list() == _as_list(case["klets"])
        assert got["counts"].to_list() == nums(case["counts"])


def test_windows_reference():
    for case in _ref("windows"):
        if not case["incomplete"]:
            continue  # the R package also drops complete last windows
        got = um.calc_windows(case["n"], case["window"], case["overlap"])
        assert got["start"].to_list() == nums(case["start"])
        assert got["stop"].to_list() == nums(case["stop"])
        complete = got.filter(pl.col("stop") - pl.col("start") + 1 == case["window"])
        assert um.calc_windows(case["n"], case["window"], case["overlap"], False).equals(complete)
    assert um.calc_windows(5, 10).rows() == [(1, 5)]
    assert um.window_string("ACGTACGTAC", 4, 1) == ["ACGT", "TACG", "GTAC"]
    assert um.window_string("ACGTACGTAC", 4, 1, return_incomp=False) == ["ACGT", "TACG", "GTAC"]
    assert um.window_string("ACGTACGTA", 4, 0, return_incomp=False) == ["ACGT", "ACGT"]
    with pytest.raises(ValueError):
        um.calc_windows(10, 3, 3)


def test_complexity_reference():
    for case in _ref("complexity"):
        got = um.calc_complexity(case["string"], case["method"], case["alph"], case["max_word"])
        assert got == pytest.approx(nums(case["value"])[0], rel=1e-12, abs=1e-15, nan_ok=True), case
    got = um.calc_complexity(["ACGT", "AAAA"], "WoottonFederhen")
    assert isinstance(got, np.ndarray) and got.shape == (2,)


def test_sequence_complexity_reference():
    seqs = dict(zip(_ref("complexity_names"), _ref("complexity_sequences")))
    for case in _ref("sequence_complexity"):
        a = args(case["args"])
        got = um.sequence_complexity(
            seqs,
            window_size=a.get("window.size", 20),
            window_overlap=a.get("window.overlap"),
            method=a.get("method", "WoottonFederhen"),
            trifonov_max_word_size=a.get("trifonov.max.word.size", 7),
        )
        t = case["table"]
        assert got["sequence"].to_list() == t["sequence"], a
        assert got["start"].to_list() == nums(t["start"]), a
        assert got["stop"].to_list() == nums(t["stop"]), a
        np.testing.assert_allclose(got["complexity"].to_numpy(), nums(t["complexity"]), rtol=1e-12)


def test_get_bkg_reference():
    seqs = dict(zip(_ref("bkg_names"), _ref("bkg_sequences")))
    for case in _ref("bkg"):
        a = args(case["args"])
        got = um.get_bkg(
            seqs,
            k=a.get("k", [1, 2, 3]),
            pseudocount=a.get("pseudocount", 0),
            alphabet=a.get("alphabet"),
            rc=a.get("RC", False),
            merge=a.get("merge.res", True),
            window=a.get("window", False),
            window_size=a.get("window.size", 0.1),
            window_overlap=a.get("window.overlap", 0),
        )
        t = case["table"]
        assert got.columns == list(t), a
        for col in t:
            if col == "probability":
                np.testing.assert_allclose(got[col].to_numpy(), nums(t[col]), rtol=1e-12)
            elif col in ("sequence", "klet"):
                assert got[col].to_list() == t[col], (a, col)
            else:
                assert got[col].to_list() == nums(t[col]), (a, col)


def test_meme_background_reference(tmp_path):
    seqs = _ref("bkg_sequences")
    f = io.StringIO()
    assert um.get_bkg(seqs, k=[1, 2, 3], pseudocount=1, to_meme=f) is None
    assert f.getvalue() == _ref("meme_bkg") + "\n"
    path = tmp_path / "bkg.txt"
    um.get_bkg(seqs, k=[1, 2, 3], pseudocount=1, to_meme=str(path))
    assert path.read_text() == f.getvalue()
    with pytest.raises(ValueError, match="pseudocount"):
        um.get_bkg(["AAAA"], k=1, alphabet="ACGT", to_meme=io.StringIO())
    with pytest.raises(ValueError, match="from 1"):
        um.get_bkg(seqs, k=[2, 3], pseudocount=1, to_meme=io.StringIO())


def test_motif_finder_reference():
    pos = _ref("finder_positives")
    neg = _ref("finder_negatives")
    for case in _ref("finder"):
        a = args(case["args"])
        kw = {
            k.replace(".", "_"): v
            for k, v in a.items()
            if k not in ("RC",)
        }
        got = um.motif_finder(pos, neg, rc=a.get("RC", True), **kw)
        assert got.height == len(case["motifs"]), a
        s = case["stats"]
        for col in ("rank", "width", "seqs_pos", "seqs_neg", "sites_pos", "sites_neg", "n_pos", "n_neg"):
            assert got[col].to_list() == nums(s.get(col)), (a, col)
        for col in ("pvalue", "qvalue"):
            np.testing.assert_allclose(got[col].to_numpy(), nums(s.get(col)), rtol=1e-10)
        for m, rec in zip(got["motif"], case["motifs"]):
            assert_motif_matches(m, rec, str(a))
        assert got["name"].to_list() == [r["name"] for r in case["motifs"]]


def test_motif_finder_shuffled_background():
    pos = _ref("finder_positives")
    a = um.motif_finder(pos, min_width=8, max_width=10, seed=4, qvalue=1)
    b = um.motif_finder(pos, min_width=8, max_width=10, seed=4, qvalue=1, nthreads=2)
    assert a.drop("motif").equals(b.drop("motif"))
    assert a.height >= 1
    assert "GACGTCA" in a["consensus"][0] or "TGACGTC" in a["consensus"][0]
    empty = um.motif_finder(pos[:5], min_width=6, max_width=6, seed=1)
    assert empty.height == 0 and "motif" in empty.columns


def test_shuffle_sequences():
    seqs = um.create_sequences(seqnum=10, seqlen=80, seed=1)
    out = um.shuffle_sequences(seqs, k=3, seed=5)
    assert out == um.shuffle_sequences(seqs, k=3, seed=5)
    assert out != seqs
    for a, b in zip(seqs, out):
        assert um.count_klets(a, 3, "ACGT").equals(um.count_klets(b, 3, "ACGT"))
    named = um.shuffle_sequences({"x": seqs[0]}, k=1, method="linear", seed=1)
    assert list(named) == ["x"] and sorted(named["x"]) == sorted(seqs[0])
    assert isinstance(um.shuffle_sequences(seqs[0], k=2, method="markov", seed=1), str)
    windowed = um.shuffle_sequences(seqs[0], window=True, window_size=20, window_overlap=0, seed=2)
    for i in range(0, 80, 20):
        assert sorted(windowed[i : i + 20]) == sorted(seqs[0][i : i + 20])
    with pytest.raises(ValueError):
        um.shuffle_sequences(seqs, method="random")


def test_create_sequences():
    seqs = um.create_sequences("RNA", 20, 50, seed=3)
    assert len(seqs) == 20 and all(len(s) == 50 and set(s) <= set("ACGU") for s in seqs)
    aa = um.create_sequences("AA", 2, 10, seed=3)
    assert all(set(s) <= set("ACDEFGHIKLMNPQRSTVWY") for s in aa)
    freqs = {k: 0.0 for k in um.get_klets("ACGT", 2)}
    freqs["AA"] = freqs["TT"] = 0.5
    for s in um.create_sequences(seqnum=5, seqlen=20, freqs=freqs, seed=1):
        assert len(set(s)) == 1
    biased = "".join(um.create_sequences(seqnum=20, seqlen=500, freqs=[0.7, 0.1, 0.1, 0.1], seed=2))
    assert abs(biased.count("A") / len(biased) - 0.7) < 0.02


def test_shuffle_motifs_and_sample_sites():
    a = um.create_motif(["ACGTAC", "ACGTTC"], name="a")
    a = um.convert_type(a, "PCM")
    b = um.convert_type(um.create_motif("TTTGCA", name="b"), "PWM")
    out = um.shuffle_motifs([a, b], k=1, seed=2)
    assert [m.name for m in out] == ["a [shuffled]", "b [shuffled]"]
    assert [m.type for m in out] == ["PCM", "PWM"]
    assert um.shuffle_motifs(a, seed=1).width == 6
    sites = um.sample_sites(a, 50, seed=1)
    assert len(sites) == 50 and all(s[:4] == "ACGT" and s[5] == "C" for s in sites)
    assert sites == um.sample_sites(a, 50, seed=1)


def test_implant_motifs():
    m = um.create_motif("GGGGGG")
    seqs = {"s1": "A" * 100, "s2": "A" * 100}
    out, idx = um.implant_motifs(m, seqs, n_per_seq=3, min_spacing=2, seed=1, return_indices=True)
    assert list(out) == ["s1", "s2"]
    assert idx.height == 6
    assert idx.columns == ["sequence_i", "motif_i", "start", "width", "strand", "planted"]
    for row in idx.iter_rows(named=True):
        s = out[["s1", "s2"][row["sequence_i"]]]
        assert s[row["start"] - 1 : row["start"] - 1 + row["width"]] == row["planted"]
    out = um.implant_motifs(m, ["A" * 20], positions=[[1, 14]], strand="+", seed=1)
    assert out[0].startswith("GGGGGG")
    with pytest.warns(UserWarning, match="out-of-bounds"):
        um.implant_motifs(m, ["A" * 20], positions=[[18]], seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        um.implant_motifs(m, ["A" * 20], rate=0.0, seed=1)
    with pytest.raises(ValueError):
        um.implant_motifs(um.create_motif("MKV", alphabet="AA"), ["MKVLL"])


def test_get_bkg_defaults():
    df = um.get_bkg(["AAAA", "CCCC"], k=1)
    assert df["klet"].to_list() == ["A", "C", "G", "T"]
    assert isinstance(df, pl.DataFrame)
    df = um.get_bkg(["MKV"], k=1)
    assert df["klet"].to_list() == ["K", "M", "V"]
