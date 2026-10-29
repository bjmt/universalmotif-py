import io
import itertools
import warnings

import numpy as np
import polars as pl
import pytest

import universalmotif as um

from conftest import args, load_reference, motif_from_record, nums


def _ref(key):
    return load_reference("sequences.json", key)


def _as_list(x):
    return x if isinstance(x, list) else [x]


def test_kmers_reference():
    for case in _ref("klets"):
        assert um.get_kmers(case["letters"], case["k"]) == case["klets"]
    for case in _ref("count"):
        got = um.count_kmers(case["string"], case["k"], case["alph"])
        assert got["kmer"].to_list() == _as_list(case["klets"])
        assert got["count"].to_list() == nums(case["counts"])


def test_windows_reference():
    for case in _ref("windows"):
        if not case["incomplete"]:
            continue  # the R package also drops complete last windows
        got = um.calc_windows(case["n"], case["window"], case["overlap"])
        # R counts positions from 1, with the end included.
        assert got["start"].to_list() == [x - 1 for x in nums(case["start"])]
        assert got["end"].to_list() == nums(case["stop"])
        complete = got.filter(pl.col("end") - pl.col("start") == case["window"])
        assert um.calc_windows(case["n"], case["window"], case["overlap"], False).equals(complete)
    assert um.calc_windows(5, 10).rows() == [(0, 5)]
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
    assert got[1] == 0
    at = um.calc_complexity(["ATATATATAT"])
    assert at[0] == pytest.approx(um.calc_complexity("ATATATATAT", alph="ACGT"))


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
        assert got["start"].to_list() == [x - 1 for x in nums(t["start"])], a
        assert got["end"].to_list() == nums(t["stop"]), a
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
        # The reference tables name the k-mer column klet and the window end
        # stop, and count positions from 1.
        names = {"klet": "kmer", "stop": "end"}
        t = {names.get(k, k): v for k, v in case["table"].items()}
        if "start" in t:
            t["start"] = [x - 1 for x in nums(t["start"])]
        assert got.columns == list(t), a
        for col, want in t.items():
            if col == "probability":
                np.testing.assert_allclose(got[col].to_numpy(), nums(want), rtol=1e-12)
            elif col in ("sequence", "kmer"):
                assert got[col].to_list() == want, (a, col)
            else:
                assert got[col].to_list() == nums(want), (a, col)


def test_meme_background_reference(tmp_path):
    seqs = _ref("bkg_sequences")
    f = io.StringIO()
    assert um.get_bkg(seqs, k=[1, 2, 3], pseudocount=1, to_meme=f) is None
    # The R package leaves a trailing space on each line.
    want = "".join(line.rstrip() + "\n" for line in _ref("meme_bkg").splitlines())
    assert f.getvalue() == want
    path = tmp_path / "bkg.txt"
    um.get_bkg(seqs, k=[1, 2, 3], pseudocount=1, to_meme=str(path))
    assert path.read_text() == f.getvalue()
    um.get_bkg(seqs, k=[1, 2, 3], pseudocount=1, to_meme=tmp_path / "bkg2.txt")
    assert (tmp_path / "bkg2.txt").read_text() == f.getvalue()
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
        # Each motif R reports is looked for among all of the port's.
        kw.setdefault("qvalue", 0.05)
        kept = um.motif_finder(pos, neg, rc=a.get("RC", True), **kw)
        assert kept.height <= len(case["motifs"]), a
        assert (kept["qvalue"] <= kw["qvalue"]).all(), a
        kw["qvalue"] = 1
        got = um.motif_finder(pos, neg, rc=a.get("RC", True), **kw)
        s = case["stats"]
        for i, rec in enumerate(case["motifs"]):
            want = motif_from_record(rec)
            k = next(j for j, m in enumerate(got["motif"]) if np.array_equal(m.matrix, want.matrix))
            row = got.row(k, named=True)
            for col in ("width", "seqs_pos", "seqs_neg", "sites_pos", "sites_neg", "n_pos", "n_neg"):
                assert row[col] == nums(s.get(col))[i], (a, col)
            assert row["pvalue"] == pytest.approx(nums(s["pvalue"])[i], rel=1e-9), a
            assert row["qvalue"] == pytest.approx(nums(s["qvalue"])[i], rel=1e-9), a
            assert row["log10_pvalue"] == pytest.approx(np.log10(row["pvalue"]), rel=1e-9), a
        assert got["rank"].to_list() == list(range(1, got.height + 1))
        assert got["pvalue"].is_sorted()


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
        assert um.count_kmers(a, 3, "ACGT").equals(um.count_kmers(b, 3, "ACGT"))
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
    freqs = {k: 0.0 for k in um.get_kmers("ACGT", 2)}
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
        assert s[row["start"]:row["start"] + row["width"]] == row["planted"]
    out = um.implant_motifs(m, ["A" * 20], positions=[[0, 13]], strand="+", seed=1)
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
    assert df["kmer"].to_list() == ["A", "C", "G", "T"]
    assert isinstance(df, pl.DataFrame)
    df = um.get_bkg(["MKV"], k=1)
    assert df["kmer"].to_list() == ["K", "M", "V"]


_M = um.create_motif("ACGT")


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: um.shuffle_sequences("ACGT", seed=-1), "seed must be a non-negative"),
        (lambda: um.get_kmers("ACGT", 0), "k must be at least 1"),
        (lambda: um.count_kmers("ACGT", 0), "k must be at least 1"),
        (lambda: um.count_kmers("", 1), "string cannot be empty"),
        (lambda: um.count_kmers("ACGT", 1, alph=""), "alph cannot be empty"),
        (lambda: um.calc_windows(0), "n should be a positive"),
        (lambda: um.calc_windows(5, 0), "window should be a positive"),
        (lambda: um.calc_windows(5, 2, -1), "overlap should be a non-negative"),
        (lambda: um.window_string("", 2), "string cannot be empty"),
        (lambda: um.calc_complexity("ACGT", "Shannon"), "complexity_method must be one of"),
        (lambda: um.calc_complexity("ACGT", "Trifonov", trifonov_max_word_size=0), "positive integer"),
        (lambda: um.sequence_complexity("ACGT", method="Shannon"), "method must be one of"),
        (lambda: um.get_bkg("ACGT", k=[1, 0]), "k must be at least 1"),
        (lambda: um.get_bkg("ACGT", k=1, window=True, to_meme=io.StringIO()), "window=True"),
        (lambda: um.get_bkg("ACGT", k=1, merge=False, to_meme=io.StringIO()), "merge=False"),
        (lambda: um.get_bkg("ACGT", k=1, pseudocount=-1), "pseudocount must be non-negative"),
        (lambda: um.shuffle_sequences("ACGT", k=0), "k must be at least 1"),
        (lambda: um.shuffle_sequences("AC", k=3, seed=1), "must not exceed"),
        (lambda: um.create_sequences(""), "alphabet cannot be empty"),
        (lambda: um.create_sequences(seqnum=-1), "seqnum must be non-negative"),
        (lambda: um.create_sequences(seqlen=0), "seqlen positive"),
        (lambda: um.create_sequences(freqs={"A": 0.5, "C": 0.5}), "one value for each of the 4 1-mers"),
        (lambda: um.create_sequences(freqs={"AA": 0.5, "AC": 0.5}), "16 2-mers"),
        (lambda: um.create_sequences(freqs=[0.5, 0.5]), "power of the alphabet size"),
        (lambda: um.create_sequences(freqs=[-1, 1, 1, 1]), "non-negative"),
        (lambda: um.shuffle_motifs(_M, k=0), "k must be at least 1"),
        (lambda: um.sample_sites(_M, n=-1), "n must be non-negative"),
        (lambda: um.implant_motifs(_M, ["A" * 10], strand="-"), "strand must be"),
        (lambda: um.implant_motifs(_M, ["A" * 10], n_per_seq=-1), "n_per_seq must be"),
        (lambda: um.implant_motifs(_M, ["A" * 10], positions=[[-1]]), "positions must be at least 0"),
        (lambda: um.motif_finder(["A" * 20], min_width=2), "3 <= min_width"),
        (lambda: um.motif_finder(["A" * 20], max_width=31), "max_width <= 30"),
        (lambda: um.motif_finder(["A" * 20], min_width=8, max_width=7), "min_width <= max_width"),
        (lambda: um.motif_finder(["A" * 20], shuffle_k=0), "shuffle_k must be at least 1"),
        (lambda: um.motif_finder(["A" * 20], pseudocount=-1), "pseudocount must be non-negative"),
    ],
)
def test_argument_checks(call, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace("+", r"\+")):
        call()


def test_count_kmers_letters():
    # Letters outside the alphabet break k-mers rather than being counted.
    got = um.count_kmers("ACNGT", 2, "ACGT")
    assert got.height == 16
    assert got.filter(pl.col("count") > 0).rows() == [("AC", 1), ("GT", 1)]
    assert um.count_kmers("ACGTN", 1, "ACGT")["count"].sum() == 4
    # Without an alphabet, every letter in the string counts, N included.
    assert um.count_kmers("ACGTN")["kmer"].to_list() == ["A", "C", "G", "N", "T"]
    # Letters are compared as they are, so lowercase stays separate.
    assert um.count_kmers("acgt").rows() == [("a", 1), ("c", 1), ("g", 1), ("t", 1)]
    # The counts of every k-mer add up to the number of windows.
    seq = um.create_sequences(seqnum=1, seqlen=57, seed=8)[0]
    for k in (1, 2, 3, 5):
        assert um.count_kmers(seq, k, "ACGT")["count"].sum() == len(seq) - k + 1
    assert um.get_kmers(["C", "A"], 2) == ["CC", "CA", "AC", "AA"]


def test_rna_default_alphabet():
    rna = ["ACGUUGCAAC", "UUUUACGNNA"]
    bkg = um.get_bkg(rna, k=1)
    assert bkg["kmer"].to_list() == ["A", "C", "G", "U"]
    assert bkg["count"].sum() == sum(len(s) - s.count("N") for s in rna)
    rc = um.get_bkg(["ACGU"], k=2, rc=True)
    assert rc.filter(pl.col("count") > 0)["kmer"].to_list() == ["AC", "CG", "GU"]
    # Complexity only depends on composition, so RNA scores as its DNA copy.
    dna = [s.replace("U", "T") for s in rna]
    np.testing.assert_allclose(um.calc_complexity(rna), um.calc_complexity(dna))
    for method in ("WoottonFederhen", "Trifonov", "DUST"):
        a = um.sequence_complexity(rna, window_size=5, method=method)
        b = um.sequence_complexity(dna, window_size=5, method=method)
        assert a.equals(b), method


@pytest.mark.parametrize("n, window, overlap", [(1, 1, 0), (10, 3, 1), (11, 4, 1), (100, 7, 6), (9, 20, 5)])
def test_windows_tile_the_sequence(n, window, overlap):
    w = um.calc_windows(n, window, overlap)
    starts, ends = w["start"].to_list(), w["end"].to_list()
    width = min(window, n)
    step = width - (overlap if window <= n else 0)
    assert starts[0] == 0 and ends[-1] == n
    assert all(b - a == step for a, b in itertools.pairwise(starts))
    assert all(0 < e - s <= width for s, e in zip(starts, ends))
    assert all(e - s == width for s, e in zip(starts[:-1], ends[:-1], strict=True))
    string = "".join("ACGT"[i % 4] for i in range(n))
    assert um.window_string(string, window, overlap) == [string[s:e] for s, e in zip(starts, ends)]


def _kmer_counts(s, k):
    out = {}
    for i in range(len(s) - k + 1):
        out[s[i:i + k]] = out.get(s[i:i + k], 0) + 1
    return out


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "seq",
    [
        "ACGTNNACGTTTGACANNACGTAAGGCTTACGAT",
        "ACGUUGCAACGUAGCUAGCUAGGCUAACGUUAGC",
        "MKVLAAGHHWWMKVLPPQMKAAGHVLWPQ",
        "acgtacgttgcaacgtagct",
    ],
)
def test_euler_shuffle_keeps_kmers_and_ends(seq, k):
    out = um.shuffle_sequences(seq, k=k, seed=11)
    assert len(out) == len(seq)
    assert _kmer_counts(out, k) == _kmer_counts(seq, k)
    assert out[:k - 1] == seq[:k - 1] and out[len(out) - k + 1:] == seq[len(seq) - k + 1:]
    assert out == um.shuffle_sequences(seq, k=k, seed=11, nthreads=2)


def test_linear_and_markov_shuffles():
    seq = "AACCGGTTACGTACGTAACCG"
    for k in (1, 2, 3):
        out = um.shuffle_sequences(seq, k=k, method="linear", seed=3)
        n = len(seq) // k * k
        blocks = sorted(seq[i:i + k] for i in range(0, n, k))
        assert sorted(out[i:i + k] for i in range(0, n, k)) == blocks
        # A partial block at the end stays where it is.
        assert out[n:] == seq[n:]
    for k in (1, 2, 3):
        out = um.shuffle_sequences(seq, k=k, method="markov", seed=3)
        assert len(out) == len(seq) and set(out) <= set(seq)
    assert um.shuffle_sequences([], seed=1) == []
    assert um.shuffle_sequences({}, seed=1) == {}


def test_create_sequences_shapes():
    assert um.create_sequences(seqnum=0) == []
    custom = um.create_sequences("ZYXYZ", 4, 9, seed=2)
    assert len(custom) == 4 and all(len(s) == 9 and set(s) <= set("XYZ") for s in custom)
    rna = um.create_sequences("RNA", 3, 40, seed=2)
    # The same seed draws the same letters whatever the alphabet.
    assert rna == [s.replace("T", "U") for s in um.create_sequences("DNA", 3, 40, seed=2)]
    # Dinucleotide frequencies in get_kmers() order set a first-order Markov
    # chain; with only AC and CA possible, letters alternate.
    freqs = [0.0] * 16
    freqs[1] = freqs[4] = 0.5
    for s in um.create_sequences(seqnum=4, seqlen=15, freqs=freqs, seed=1):
        assert set(s) == {"A", "C"} and "AA" not in s and "CC" not in s
    assert um.create_sequences(seqnum=3, seqlen=8, seed=5) == um.create_sequences(
        seqnum=3, seqlen=8, seed=5, nthreads=2)


def test_create_sequences_rejects_zero_frequencies():
    with pytest.raises(ValueError):
        um.create_sequences(seqnum=2, seqlen=10, freqs=[0, 0, 0, 0], seed=1)


def test_shuffle_motifs_moves_whole_columns():
    ppms = um.convert_type(um.example_motifs()[:3], "PPM")

    def columns(ms):
        return sorted(map(tuple, np.round(np.concatenate([m.matrix for m in ms], axis=1).T, 9)))

    for k in (1, 2, 3):
        out = um.shuffle_motifs(ppms, k=k, seed=4)
        assert [m.width for m in out] == [m.width for m in ppms]
        assert columns(out) == columns(ppms)
    # Counts keep each motif's own number of sites.
    pcms = um.example_motifs()[:3]
    out = um.shuffle_motifs(pcms, k=1, seed=4)
    assert [m.nsites for m in out] == [m.nsites for m in pcms]
    assert all((m.matrix.sum(axis=0) == m.nsites).all() for m in out)


def test_sample_sites_follow_probabilities():
    ppm = um.create_motif(np.array([[0.7, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.7]]), type="PPM",
                          alphabet="DNA")
    sites = um.sample_sites(ppm, 4000, seed=1)
    assert abs(sum(s[0] == "A" for s in sites) / 4000 - 0.7) < 0.03
    assert abs(sum(s[1] == "T" for s in sites) / 4000 - 0.7) < 0.03
    assert um.sample_sites(ppm, 0, seed=1) == []
    assert set(um.sample_sites(um.create_motif("MKV", alphabet="AA"), 5, seed=1)) == {"MKV"}
    counts = um.create_motif(["AC", "AG"], type="PCM")
    assert set(um.sample_sites(counts, 50, seed=1)) == {"AC", "AG"}


def test_implant_motifs_rna_and_strands():
    site = um.create_motif("ACGGU", alphabet="RNA")
    seqs = ["A" * 30] * 6
    out, idx = um.implant_motifs(site, seqs, seed=2, return_indices=True)
    assert set(idx["strand"]) == {"+", "-"}
    for i, start, width, strand, planted in idx.select(
        "sequence_i", "start", "width", "strand", "planted"
    ).iter_rows():
        assert out[i][start:start + width] == planted
        # Sites on the minus strand read as the reverse complement, with U.
        assert planted == ("ACGGU" if strand == "+" else "ACCGU")
    plus = um.implant_motifs(site, seqs, strand="+", seed=2, return_indices=True)[1]
    assert set(plus["strand"]) == {"+"}
    none = um.implant_motifs(site, seqs, n_per_seq=0, return_indices=True)
    assert none[0] == seqs and none[1].height == 0
