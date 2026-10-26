"""Convolution filters, one-hot sequences and TF-MoDISco results."""

import numpy as np
import polars as pl
import pytest

import universalmotif as um


def test_filters_score_like_scanning():
    motifs = um.example_motifs()[:4]
    filters = um.to_filters(motifs)
    assert filters.shape == (4, 4, 10) and filters.dtype == np.float32
    for f, m in zip(filters, motifs):
        np.testing.assert_allclose(f[:, :m.width], um.convert_type(m, "PWM").matrix, rtol=1e-6)
        assert (f[:, m.width:] == 0).all()
    # Sliding a filter along one-hot sequences gives the scores of the windows.
    seq = um.example_sequences()["target_001"][:200]
    x = um.one_hot([seq])[0]
    f = filters[1, :, :motifs[1].width].astype(float)
    conv = np.array([(f * x[:, i:i + f.shape[1]]).sum() for i in range(200 - f.shape[1] + 1)])
    words = [seq[i:i + f.shape[1]] for i in range(len(conv))]
    np.testing.assert_allclose(conv, um.score_match(motifs[1], words), atol=0.01)


def test_filter_options():
    motifs = um.example_motifs()[1:3]
    # CRE, 8 wide, sits in the middle of the TATA-box's 10 columns.
    centred = um.to_filters(um.example_motifs()[:3], align="centre")
    assert (centred[2, :, 0] == 0).all() and (centred[2, :, 9] == 0).all()
    np.testing.assert_allclose(centred[2, :, 1:9], um.convert_type(um.example_motifs()[2], "PWM").matrix,
                               rtol=1e-6)
    rc = um.to_filters(motifs, rc=True)
    assert rc.shape[0] == 4
    np.testing.assert_allclose(rc[1, :, :9], rc[0, ::-1, 8::-1])
    ppm = um.to_filters(motifs, use_type="PPM")
    np.testing.assert_allclose(ppm[0, :, :9].sum(axis=0), 1, rtol=1e-6)
    bare = um.create_motif(["TGA", "TGA"], type="PCM", pseudocount=0)
    with pytest.raises(ValueError, match="pseudocount"):
        um.to_filters(bare)
    with pytest.raises(ValueError, match="alphabet"):
        um.to_filters([motifs[0], um.create_motif("MKV", alphabet="AA")])
    with pytest.raises(ValueError, match="reverse complements"):
        um.to_filters(um.create_motif("MKV", alphabet="AA"), rc=True)


def test_from_filters():
    motifs = um.example_motifs()[:3]
    back = um.from_filters(um.to_filters(motifs, align="centre"), names=[m.name for m in motifs])
    for b, m in zip(back, motifs):
        assert b.width == m.width and b.type == "PWM"
        np.testing.assert_allclose(b.matrix, um.convert_type(m, "PWM").matrix, rtol=1e-6)
    # Mappings of (letters, width) matrices, such as tangermeme keeps motifs in.
    ppms = {m.name: um.convert_type(m, "PPM").matrix for m in motifs}
    from_dict = um.from_filters(ppms, type="PPM")
    assert [m.name for m in from_dict] == list(ppms)
    np.testing.assert_allclose(from_dict[0].matrix, ppms["TATA-box"])

    class Tensor:
        """Stands in for a torch tensor."""

        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    assert um.from_filters(Tensor(um.to_filters(motifs)))[0].name == "filter_1"
    with pytest.raises(ValueError, match="2 names for 3"):
        um.from_filters(um.to_filters(motifs), names=["a", "b"])


def test_one_hot():
    x = um.one_hot({"a": "ACGTN", "b": "acgtt"})
    assert x.shape == (2, 4, 5) and x.dtype == np.float32
    assert x[0, :, 4].sum() == 0 and x[1, 3, 3] == 1
    assert um.from_one_hot(x) == ["ACGTN", "ACGTT"]
    assert um.from_one_hot(x[0], unknown="-") == ["ACGT-"]
    rna = um.one_hot("ACGU", alphabet="ACGU", dtype=np.int8)
    assert rna.dtype == np.int8 and um.from_one_hot(rna, alphabet="ACGU") == ["ACGU"]
    soft = np.array([[[0.7, 0.1], [0.1, 0.1], [0.1, 0.6], [0.1, 0.2]]])
    assert um.from_one_hot(soft) == ["AG"]
    with pytest.raises(ValueError, match="same length"):
        um.one_hot(["ACG", "AC"])


def _modisco_inputs():
    """The 50 input sequences of 100 letters the fake modisco file's seqlets
    come from."""
    rng = np.random.default_rng(3)
    return ["".join(rng.choice(list("ACGT"), 100)) for _ in range(50)]


def _rc(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _add_seqlets(g, n, width, rng):
    """Seqlets of a fake modisco pattern, cut from _modisco_inputs() and
    stored in the pattern's orientation."""
    inputs = _modisco_inputs()
    example = rng.integers(0, len(inputs), n)
    start = rng.integers(0, 100 - width + 1, n)
    rc = rng.random(n) < 0.5
    letters = [inputs[i][s:s + width] for i, s in zip(example, start)]
    letters = [_rc(x) if r else x for x, r in zip(letters, rc)]
    onehot = np.array([[[c == b for b in "ACGT"] for c in x] for x in letters], dtype=np.float32)
    hyp = rng.normal(0, 1, (n, width, 4)).astype(np.float32)
    s = g["seqlets"]
    s.create_dataset("example_idx", data=example)
    s.create_dataset("start", data=start)
    s.create_dataset("end", data=start + width)
    s.create_dataset("is_revcomp", data=rc)
    s.create_dataset("sequence", data=onehot)
    s.create_dataset("hypothetical_contribs", data=hyp)
    s.create_dataset("contrib_scores", data=hyp * onehot)


def _modisco_file(path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)
    seqlet_rng = np.random.default_rng(2)

    def pattern(grp, name, n, core):
        g = grp.create_group(name)
        contrib = rng.normal(0, 0.01, (12, 4))
        seq = np.full((12, 4), 0.25)
        for i, letter in enumerate(core):
            j = "ACGT".index(letter)
            contrib[4 + i] = 0
            contrib[4 + i, j] = 1.0
            seq[4 + i] = 0.02
            seq[4 + i, j] = 0.94
        g.create_dataset("contrib_scores", data=contrib)
        g.create_dataset("hypothetical_contribs", data=contrib * 2)
        g.create_dataset("sequence", data=seq)
        g.create_group("seqlets").create_dataset("n_seqlets", data=np.array([n]))
        _add_seqlets(g, n, 12, seqlet_rng)
        return g

    with h5py.File(path, "w") as f:
        pos = f.create_group("pos_patterns")
        p0 = pattern(pos, "pattern_0", 120, "TGAC")
        pattern(pos, "pattern_10", 15, "CACG")
        pattern(pos, "pattern_2", 40, "GATA")
        pattern(p0, "subpattern_0", 70, "TGAC")
        pattern(f.create_group("neg_patterns"), "pattern_0", 30, "AAAT")
    return path


def test_read_modisco(tmp_path):
    path = _modisco_file(tmp_path / "modisco.h5")
    cwms = um.read_modisco(path)
    assert [m.name for m in cwms] == ["pos_patterns.pattern_0", "pos_patterns.pattern_2",
                                      "pos_patterns.pattern_10", "neg_patterns.pattern_0"]
    assert all(m.type == "CWM" and m.width == 4 for m in cwms)
    assert [m.nsites for m in cwms] == [120, 40, 15, 30]
    assert cwms[0].consensus == "TGAC"
    ppms = um.read_modisco(path, type="PPM")
    assert ppms[0].type == "PPM" and ppms[0].width == 4 and ppms[0].consensus == "TGAC"
    full = um.read_modisco(path, type="hCWM", trim_threshold=None)
    assert full[0].width == 12
    np.testing.assert_allclose(full[0].matrix[:, 4:8], 2 * cwms[0].matrix)
    subs = um.read_modisco(path, subpatterns=True)
    assert subs[1].name == "pos_patterns.pattern_0.subpattern_0" and subs[1].nsites == 70
    with pytest.raises(ValueError, match="type"):
        um.read_modisco(path, type="PWM")


def _chrombpnet_files(tmp_path, seqs, peaks):
    h5py = pytest.importorskip("h5py")
    onehot = um.one_hot(seqs, dtype=np.int8)
    rng = np.random.default_rng(1)
    hyp = rng.normal(size=onehot.shape).astype(np.float16)
    path = tmp_path / "model.counts_scores.h5"
    # chromBPNet writes (sequences, letters, length).
    with h5py.File(path, "w") as f:
        f["raw/seq"] = onehot
        f["shap/seq"] = hyp
        f["projected_shap/seq"] = hyp * onehot
    (tmp_path / "model.interpreted_regions.bed").write_text("".join(
        f"{chrom}\t{start}\t{start + 500}\tp\t0\t.\t1\t1\t1\t{summit}\n"
        for chrom, start, summit in peaks))
    return path, hyp.astype(np.float32), onehot


def test_read_contributions_chrombpnet(tmp_path):
    seqs = ["ACGTACGTAA", "TTTTGGGGCA", "GATTACAGAT"]
    peaks = [("chr1", 100, 50), ("chr2", 0, 20), ("chr1", 1000, 7)]
    path, hyp, onehot = _chrombpnet_files(tmp_path, seqs, peaks)
    c = um.read_contributions(path)
    assert len(c) == 3 and c.sequences == seqs
    assert c.scores.dtype == np.float32 and c.scores.shape == (3, 4, 10)
    np.testing.assert_allclose(c.scores, hyp)
    assert c.names == ["chr1:145-155", "chr2:15-25", "chr1:1002-1012"]
    assert c.ranges.columns == ["chrom", "start", "end"]
    proj = um.read_contributions(path, scores="projected")
    np.testing.assert_allclose(proj.scores, hyp * onehot)
    some = um.read_contributions(path, indices=[2, 0])
    assert some.sequences == [seqs[2], seqs[0]]
    assert some.names == [c.names[2], c.names[0]]
    np.testing.assert_allclose(some.scores, hyp[[2, 0]])
    scores, seq = c[1]
    assert seq == seqs[1] and scores.shape == (4, 10)
    ax = c.view(0)
    assert ax.get_xlabel() == "Position"
    (tmp_path / "model.interpreted_regions.bed").unlink()
    bare = um.read_contributions(path)
    assert bare.ranges is None and bare.names == ["seq1", "seq2", "seq3"]
    with pytest.raises(ValueError, match="2 regions"):
        um.read_contributions(path, regions=pl.DataFrame(
            {"chrom": ["chr1", "chr1"], "start": [0, 0], "summit": [5, 5]}))
    with pytest.raises(ValueError, match="hypothetical"):
        um.read_contributions(path, scores="raw")
    with pytest.raises(ValueError, match="repeat"):
        um.read_contributions(path, indices=[0, 0])


def test_read_contributions_numpy(tmp_path):
    seqs = ["ACGTA", "GGCAT"]
    onehot = um.one_hot(seqs, dtype=np.int8)
    hyp = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
    np.savez(tmp_path / "shap.npz", hyp)
    np.savez(tmp_path / "ohe.npz", onehot)
    c = um.read_contributions(tmp_path / "shap.npz", sequences=tmp_path / "ohe.npz")
    assert c.sequences == seqs and c.names == ["seq1", "seq2"] and c.ranges is None
    np.testing.assert_allclose(c.scores, hyp)
    # (sequences, length, letters) and strings work too.
    np.save(tmp_path / "shap.npy", hyp.transpose(0, 2, 1))
    p = um.read_contributions(tmp_path / "shap.npy", sequences=seqs, scores="projected")
    np.testing.assert_allclose(p.scores, hyp * onehot)
    one = um.read_contributions(tmp_path / "shap.npz", sequences=onehot, indices=[1])
    assert one.sequences == ["GGCAT"] and one.names == ["seq2"]
    with pytest.raises(ValueError, match="give the sequences"):
        um.read_contributions(tmp_path / "shap.npz")
    with pytest.raises(ValueError, match="sequences of length"):
        um.read_contributions(tmp_path / "shap.npz", sequences=["ACGTA"])


def test_mutagenesis_round_trip():
    seq = "ACGNT"
    batch = um.mutagenesis(seq)
    # Three substitutions per known letter and four for the N.
    assert batch.shape == (1 + 3 * 4 + 4, 4, 5)
    assert um.from_one_hot(batch[:1]) == [seq]
    variants = um.from_one_hot(batch[1:])
    assert len(set(variants)) == len(variants) and seq not in variants
    assert all(sum(x != y for x, y in zip(v, seq)) == 1 for v in variants)
    # A model scoring the number of Gs gives +1 for every new G and -1 for losing one.
    preds = batch[:, 2, :].sum(axis=1)
    eff = um.mutagenesis_effects(preds, seq)
    assert eff.shape == (4, 5)
    assert eff[2, 0] == 1 and eff[0, 2] == -1 and eff[2, 2] == 0 and eff[2, 3] == 1
    two = um.mutagenesis_effects(np.stack([preds, -preds], axis=1), seq)
    assert two.shape == (2, 4, 5) and np.array_equal(two[1], -eff)
    centred = um.mutagenesis_effects(preds, seq, centre=True)
    assert np.allclose(centred.mean(axis=0), 0)
    sub = um.mutagenesis(seq, region=(1, 3))
    assert sub.shape == (7, 4, 5)
    assert um.mutagenesis_effects(sub[:, 2, :].sum(axis=1), seq, region=(1, 3)).shape == (4, 2)
    with pytest.raises(ValueError, match="expected 7 predictions"):
        um.mutagenesis_effects(np.zeros(5), seq, region=(1, 3))
    with pytest.raises(ValueError, match="region"):
        um.mutagenesis(seq, region=(3, 9))
    with pytest.raises(TypeError, match="single sequence"):
        um.mutagenesis(["ACGT"])


def test_insert_sites_and_spacings():
    bkg = {"a": "AAAAAAAAAA", "b": "CCCCCCCCCC"}
    out = um.insert_sites(bkg, "TGA")
    assert out == {"a": "AAATGAAAAA", "b": "CCCTGACCCC"}
    motif = um.create_motif("TTGACG")
    assert um.insert_sites("AAAAAAAAAA", motif, positions=4) == "AAAATTGACG"
    assert um.insert_sites(["AAAAAA"], ["GG", "TT"], positions=[0, 4]) == ["GGAATT"]
    with pytest.raises(ValueError, match="does not fit"):
        um.insert_sites("AAAA", "TTT", positions=2)
    with pytest.raises(ValueError, match="positions of several"):
        um.insert_sites("AAAA", ["T", "G"])
    res = um.site_spacings(["AAAAAAAAAAAA", "CCCCCCCCCCCC"], "GG", "TT", spacings=range(0, 5, 2))
    assert list(res) == [0, 2, 4]
    assert res[4] == ["AAGGAAAATTAA", "CCGGCCCCTTCC"]
    # The first site stays put as the second moves.
    assert {s.index("GG") for v in res.values() for s in v} == {2}
    stacked = np.stack([um.one_hot(v) for v in res.values()])
    assert stacked.shape == (3, 2, 4, 12)
    with pytest.raises(ValueError, match="spacings"):
        um.site_spacings("AAAA", "G", "T", spacings=[-1])


def test_one_hot_batches(tmp_path):
    seq = "AACCGGTTACGTACGTAAAACCCCGGGGTTTT"
    ranges = pl.DataFrame({"chrom": ["chr1"] * 5, "start": [0, 4, 8, 12, 20],
                           "end": [8, 12, 16, 20, 28], "strand": ["+", "-", "+", "+", "+"]})
    batches = um.one_hot_batches({"chr1": seq}, ranges, batch_size=2)
    assert len(batches) == 3
    arrays = list(batches)
    assert [a.shape for a in arrays] == [(2, 4, 8), (2, 4, 8), (1, 4, 8)]
    expected = list(um.extract_ranges({"chr1": seq}, ranges).values())
    assert [s for a in arrays for s in um.from_one_hot(a)] == expected
    # Windows around the middles, from a genome on disk.
    path = tmp_path / "g.fa"
    path.write_text(">chr1\n" + seq + "\n")
    genome = um.Genome(path)
    centred = um.one_hot_batches(genome, [("chr1", 0, 9), ("chr1", 10, 30)], width=6)
    assert um.from_one_hot(next(iter(centred))) == [seq[1:7], seq[17:23]]
    with pytest.raises(ValueError, match="different widths"):
        um.one_hot_batches({"chr1": seq}, [("chr1", 0, 4), ("chr1", 0, 5)])
    with pytest.raises(ValueError, match="does not fit"):
        um.one_hot_batches({"chr1": seq}, [("chr1", 0, 4)], width=12)
    with pytest.raises(ValueError, match="not in the genome"):
        um.one_hot_batches({"chr1": seq}, [("chr2", 0, 4)])


def test_read_contributions_edge_cases(tmp_path):
    # Odd lengths give ranges as wide as the sequences.
    seqs = ["ACGTA", "TTGAC"]
    path, hyp, _ = _chrombpnet_files(tmp_path, seqs, [("chr1", 200, 25), ("chr2", 0, 20)])
    c = um.read_contributions(path)
    assert all(e - s == 5 for _, s, e in c.ranges.iter_rows())
    with pytest.warns(UserWarning, match="sequences is ignored"):
        um.read_contributions(path, sequences=seqs)
    # Separate objects compare by identity rather than failing on arrays.
    other = um.read_contributions(path)
    assert other != c and not {other} & {c}
    # Indices are applied to memory-mapped .npy files before converting.
    np.save(tmp_path / "shap.npy", hyp.astype(np.float64))
    np.save(tmp_path / "ohe.npy", um.one_hot(seqs, dtype=np.int8))
    one = um.read_contributions(tmp_path / "shap.npy", sequences=tmp_path / "ohe.npy", indices=[1])
    assert one.sequences == ["TTGAC"] and one.scores.dtype == np.float32
    np.testing.assert_allclose(one.scores[0], hyp[1], rtol=1e-6)
    with pytest.warns(UserWarning, match="regions is only used"):
        um.read_contributions(tmp_path / "shap.npy", sequences=tmp_path / "ohe.npy", regions="x.bed")


def test_filter_argument_checks():
    with pytest.raises(ValueError, match="no motifs"):
        um.to_filters([])
    with pytest.raises(ValueError, match="align"):
        um.to_filters(um.create_motif("ACGT"), align="right")
    with pytest.raises(ValueError, match="shape"):
        um.from_filters(np.zeros((2, 2, 4, 3)))
    with pytest.raises(ValueError, match="'a' is not two-dimensional"):
        um.from_filters({"a": np.zeros((2, 4, 3))})


def test_filters_round_trip_types_and_alphabets():
    motifs = um.example_motifs()[:3]
    for use_type in ("PPM", "PWM", "ICM"):
        filters = um.to_filters(motifs, use_type=use_type, align="centre")
        back = um.from_filters(filters, names=[m.name for m in motifs], type=use_type)
        for b, m in zip(back, motifs):
            assert b.type == use_type and b.width == m.width
            np.testing.assert_allclose(b.matrix, um.convert_type(m, use_type).matrix, rtol=1e-6,
                                       atol=1e-6)
            np.testing.assert_allclose(um.convert_type(b, "PPM").matrix,
                                       um.convert_type(m, "PPM").matrix, atol=1e-5)
    # A single (letters, width) matrix is one filter.
    one = um.from_filters(um.to_filters(motifs[0])[0])
    assert len(one) == 1 and one[0].name == "filter_1" and one[0].width == motifs[0].width
    aa = um.create_motif(["MKVLA", "MKVLW"], alphabet="AA", pseudocount=1)
    f = um.to_filters(aa, use_type="PPM")
    assert f.shape == (1, 20, 5)
    back = um.from_filters(f, type="PPM", alphabet="AA")[0]
    assert back.alphabet == "AA" and back.consensus == aa.consensus
    # Filters of zeros are left as they are rather than trimmed away.
    assert um.from_filters(np.zeros((1, 4, 3)))[0].width == 3
    padded = np.pad(um.to_filters(motifs[2]), ((0, 0), (0, 0), (1, 2)))
    assert um.from_filters(padded, trim=False)[0].width == motifs[2].width + 3
    assert um.from_filters(padded)[0].width == motifs[2].width


def test_one_hot_edge_cases():
    assert um.one_hot([]).shape == (0, 4, 0)
    assert um.one_hot({}, alphabet="ACGU").shape == (0, 4, 0)
    with pytest.raises(ValueError, match="repeated"):
        um.one_hot(["ACG"], alphabet="AAC")
    with pytest.raises(ValueError, match=r"\(sequences, 4, length\)"):
        um.from_one_hot(np.zeros(4))
    with pytest.raises(ValueError, match=r"\(sequences, 4, length\)"):
        um.from_one_hot(np.zeros((1, 3, 4)))
    # Each column holds at most a single 1, and only letters of the alphabet get one.
    seqs = um.create_sequences(seqnum=5, seqlen=30, seed=1)
    seqs = [s[:10] + "N" + s[11:] for s in seqs]
    x = um.one_hot(seqs)
    assert set(np.unique(x.sum(axis=1))) == {0.0, 1.0}
    assert (x.sum(axis=1) == 0).sum() == 5
    assert um.from_one_hot(x) == seqs
    aa = "ACDEFGHIKLMNPQRSTVWY"
    assert um.from_one_hot(um.one_hot(["MKVWY", "acdef"], alphabet=aa), alphabet=aa) == ["MKVWY", "ACDEF"]
    assert um.from_one_hot(um.one_hot(["xyzzy"], alphabet="xyz"), alphabet="xyz") == ["xyzzy"]
    # The channels follow the alphabet as given, not sorted.
    assert um.one_hot("TA", alphabet="TA")[0].astype(int).tolist() == [[1, 0], [0, 1]]


def test_one_hot_round_trips_random_sequences():
    rng = np.random.default_rng(3)
    for alphabet in ("ACGT", "ACGU", "ACDEFGHIKLMNPQRSTVWY"):
        seqs = ["".join(rng.choice(list(alphabet), 25)) for _ in range(4)]
        x = um.one_hot(seqs, alphabet=alphabet)
        assert x.shape == (4, len(alphabet), 25)
        assert um.from_one_hot(x, alphabet=alphabet) == seqs
        assert np.array_equal(um.one_hot(um.from_one_hot(x, alphabet=alphabet), alphabet=alphabet), x)


def test_modisco_without_negative_patterns(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _modisco_file(tmp_path / "modisco.h5")
    with h5py.File(path, "a") as f:
        del f["neg_patterns"]
    assert [m.name for m in um.read_modisco(path)] == [
        "pos_patterns.pattern_0", "pos_patterns.pattern_2", "pos_patterns.pattern_10"]


def test_missing_h5py(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="universalmotif\\[h5py\\]"):
        um.read_modisco(tmp_path / "x.h5")
    with pytest.raises(ImportError, match="universalmotif\\[h5py\\]"):
        um.read_contributions(tmp_path / "x.h5")


def test_read_contributions_shape_checks(tmp_path):
    h5py = pytest.importorskip("h5py")
    seqs = ["ACGTA", "GGCAT"]
    onehot = um.one_hot(seqs, dtype=np.int8)
    # A single sequence stored without its first axis.
    path = tmp_path / "one.counts_scores.h5"
    with h5py.File(path, "w") as f:
        f["raw/seq"] = onehot[0]
        f["shap/seq"] = np.ones((4, 5), dtype=np.float32)
        f["projected_shap/seq"] = onehot[0].astype(np.float32)
    one = um.read_contributions(path)
    assert one.sequences == ["ACGTA"] and one.scores.shape == (1, 4, 5)
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f["something"] = np.zeros(3)
    with pytest.raises(ValueError, match="not a chromBPNet scores file; it holds something"):
        um.read_contributions(other)
    np.save(tmp_path / "flat.npy", np.zeros(10))
    with pytest.raises(ValueError, match="scores must have shape"):
        um.read_contributions(tmp_path / "flat.npy", sequences=seqs)
    np.save(tmp_path / "wide.npy", np.zeros((2, 5, 6)))
    with pytest.raises(ValueError, match=r"not \(2, 5, 6\)"):
        um.read_contributions(tmp_path / "wide.npy", sequences=seqs)
    np.savez(tmp_path / "two.npz", a=np.zeros((2, 4, 5)), b=np.zeros((2, 4, 5)))
    with pytest.raises(ValueError, match=r"several arrays \(a, b\)"):
        um.read_contributions(tmp_path / "two.npz", sequences=seqs)
    # With several arrays, the unnamed first one is taken.
    np.savez(tmp_path / "named.npz", np.ones((2, 4, 5)), extra=np.zeros(1))
    assert um.read_contributions(tmp_path / "named.npz", sequences=seqs).scores.sum() == 40
    # Indices pick from sequences given as strings too.
    np.save(tmp_path / "s.npy", np.arange(40, dtype=np.float32).reshape(2, 4, 5))
    picked = um.read_contributions(tmp_path / "s.npy", sequences=seqs, indices=[1])
    assert picked.sequences == ["GGCAT"] and picked.names == ["seq2"]
    np.testing.assert_array_equal(picked.scores[0], np.arange(20, 40).reshape(4, 5))
    assert repr(picked) == "ContributionScores(1 sequences of 5 letters)"


def test_read_contributions_regions(tmp_path):
    seqs = ["ACGTACGTAA", "TTTTGGGGCA"]
    path, _, _ = _chrombpnet_files(tmp_path, seqs, [("chr1", 100, 50), ("chr2", 0, 20)])
    short = tmp_path / "short.bed"
    short.write_text("chr1\t100\t200\n")
    with pytest.raises(ValueError, match="ten columns of a narrowPeak"):
        um.read_contributions(path, regions=short)
    peaks = tmp_path / "peaks.narrowPeak"
    peaks.write_text("chr3\t10\t500\tp\t0\t.\t1\t1\t1\t30\nchr4\t0\t50\tp\t0\t.\t1\t1\t1\t5\n")
    c = um.read_contributions(path, regions=peaks)
    # Each window of 10 is centred on its summit.
    assert c.names == ["chr3:35-45", "chr4:0-10"]
    some = um.read_contributions(path, regions=peaks, indices=[1])
    assert some.names == ["chr4:0-10"]


def test_site_checks():
    with pytest.raises(TypeError, match="non-empty string or a motif"):
        um.insert_sites("AAAA", "")
    with pytest.raises(TypeError, match="non-empty string or a motif"):
        um.insert_sites("AAAA", 5)
    with pytest.raises(ValueError, match="1 positions for 2 sites"):
        um.insert_sites("AAAA", ["G", "T"], positions=[0])
    # Later sites overwrite earlier ones where they overlap.
    assert um.insert_sites("AAAAAA", ["GGG", "TT"], positions=[1, 2]) == "AGTTAA"
    # A motif's most probable letters are used, for RNA as well.
    rna = um.create_motif(["ACGUU", "ACGUA", "AGGUU"], alphabet="RNA")
    assert um.insert_sites("CCCCCCC", rna, positions=1) == "CACGUUC"


def test_one_hot_batch_checks():
    genome = {"chr1": "ACGTACGTAC"}
    with pytest.raises(ValueError, match="batch_size"):
        um.one_hot_batches(genome, [("chr1", 0, 2)], batch_size=0)
    with pytest.raises(ValueError, match="width must be 1 or more"):
        um.one_hot_batches(genome, [("chr1", 0, 2)], width=0)
    batches = um.one_hot_batches(genome, [("chr1", i, i + 2) for i in range(5)], batch_size=2)
    assert repr(batches) == "<3 batches of up to 2 sequences>"
    assert [s for x in batches for s in um.from_one_hot(x)] == ["AC", "CG", "GT", "TA", "AC"]
    # Iterating again reads the batches afresh.
    assert sum(x.shape[0] for x in batches) == 5


def test_filters_are_read_as_their_type():
    # Log-odds filters need not have negative values, and are not guessed
    # to be probabilities or counts.
    f = np.array([[0.5, 1.2, 0.1], [0.2, 0.1, 0.3], [0.1, 0.4, 0.2], [1.5, 0.2, 0.9]])
    m = um.from_filters(f)[0]
    assert m.type == "PWM"
    np.testing.assert_allclose(m.matrix, f)
    np.testing.assert_allclose(um.from_filters(np.ones((4, 2)) * 2, type="PCM")[0].matrix, 2)


def test_read_seqlets(tmp_path):

    path = _modisco_file(tmp_path / "modisco.h5")
    inputs = _modisco_inputs()
    s = um.read_seqlets(path)
    assert s.columns == ["motif", "motif_i", "sequence", "sequence_i", "start", "end", "strand",
                         "score", "match"]
    assert s.schema["motif_i"] == pl.UInt32 and s.schema["sequence_i"] == pl.UInt32
    assert s.schema["start"] == pl.Int64 and s.schema["score"] == pl.Float64
    # Patterns come as read_modisco() gives them, without their subpatterns.
    names = [m.name for m in um.read_modisco(path)]
    counts = s.group_by("motif", maintain_order=True).len()
    assert counts["motif"].to_list() == names and counts["len"].to_list() == [120, 40, 15, 30]
    assert s.group_by("motif", maintain_order=True).agg(pl.col("motif_i").first())["motif_i"].to_list() == [
        0, 1, 2, 3]
    assert s.equals(s.sort("motif_i", "sequence_i", "start", maintain_order=True))
    assert (s["sequence"] == ("seq" + (s["sequence_i"] + 1).cast(pl.String))).all()
    assert ((s["end"] - s["start"]) == 12).all()
    # Letters read along each seqlet's strand, as in scan results.
    for row in s.iter_rows(named=True):
        site = inputs[row["sequence_i"]][row["start"]:row["end"]]
        assert row["match"] == (site if row["strand"] == "+" else _rc(site))

    projected = um.read_seqlets(path, scores="projected")
    assert projected.drop("scores").equals(s)
    x = projected["scores"].to_numpy()
    assert x.shape == (205, 4, 12) and x.dtype == np.float32
    np.testing.assert_allclose(x.sum(axis=(1, 2)), s["score"].to_numpy(), rtol=1e-5, atol=1e-5)
    # Scores run along the input sequences whatever the strand.
    for row, scores in zip(s.iter_rows(named=True), x):
        site = inputs[row["sequence_i"]][row["start"]:row["end"]]
        assert um.from_one_hot(np.abs(scores) > 0)[0] == site

    hyp = um.read_seqlets(path, patterns="pos_patterns.pattern_2", scores="hypothetical")
    assert set(hyp["motif"]) == {"pos_patterns.pattern_2"} and hyp.height == 40
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as f:
        g = f["pos_patterns/pattern_2/seqlets"]
        order = np.lexsort((g["start"][()], g["example_idx"][()]))
        stored, rc = g["hypothetical_contribs"][()][order], g["is_revcomp"][()][order]
    for scores, st, r in zip(hyp["scores"].to_numpy(), stored, rc):
        np.testing.assert_array_equal(scores, (st[::-1, ::-1] if r else st).T)
    # A row's scores go straight to view_attributions-style arrays.
    assert np.asarray(hyp.row(0, named=True)["scores"]).shape == (4, 12)
    two = um.read_seqlets(path, patterns=["neg_patterns.pattern_0", "pos_patterns.pattern_0"])
    assert two["motif"].unique(maintain_order=True).to_list() == ["pos_patterns.pattern_0",
                                                                  "neg_patterns.pattern_0"]
    assert two.filter(pl.col("motif") == "neg_patterns.pattern_0")["motif_i"].unique().to_list() == [3]


def test_read_seqlets_regions(tmp_path):

    path = _modisco_file(tmp_path / "modisco.h5")
    s = um.read_seqlets(path, scores="projected")
    regions = pl.DataFrame({"chrom": ["chr1"] * 50, "start": [1000 * i for i in range(50)],
                            "end": [1000 * i + 200 for i in range(50)]})
    g = um.read_seqlets(path, regions=regions, window=100, scores="projected")
    assert g.columns[:6] == ["motif", "motif_i", "sequence", "sequence_i", "chrom", "start"]
    assert (g["start"] == 1000 * s["sequence_i"].cast(pl.Int64) + 50 + s["start"]).all()
    assert (g["end"] - g["start"] == 12).all() and (g["strand"] == s["strand"]).all()
    assert g["sequence"][0] == f"chr1:{1000 * g['sequence_i'][0]}-{1000 * g['sequence_i'][0] + 200}"
    np.testing.assert_array_equal(g["scores"].to_numpy(), s["scores"].to_numpy())
    # Without a window the seqlets are placed from the start of each region,
    # and a BED file gives the same table as a DataFrame.
    bed = tmp_path / "regions.bed"
    regions.write_csv(bed, separator="\t", include_header=False)
    plain = um.read_seqlets(path, regions=bed)
    assert (plain["start"] == 1000 * s["sequence_i"].cast(pl.Int64) + s["start"]).all()
    assert plain.equals(um.read_seqlets(path, regions=regions))
    assert um.read_seqlets(path, regions=bed, window=100).equals(g.drop("scores"))

    # Regions on the - strand were reverse complemented for the model.
    minus = regions.with_columns(strand=pl.lit("-"))
    r = um.read_seqlets(path, regions=minus, window=100, scores="projected")
    region_end = 1000 * s["sequence_i"].cast(pl.Int64) + 200
    assert (r["start"] == region_end - 50 - s["end"]).all()
    assert (r["strand"] != s["strand"]).all() and (r["match"] == s["match"]).all()
    np.testing.assert_array_equal(r["scores"].to_numpy(), s["scores"].to_numpy()[:, ::-1, ::-1])


def test_read_seqlets_checks(tmp_path, monkeypatch):

    path = _modisco_file(tmp_path / "modisco.h5")
    regions = [("chr1", 1000 * i, 1000 * i + 200) for i in range(50)]
    with pytest.raises(ValueError, match="no patterns named pattern_0; the file has pos_patterns"):
        um.read_seqlets(path, patterns="pattern_0")
    with pytest.raises(ValueError, match="scores must be"):
        um.read_seqlets(path, scores="contrib")
    with pytest.raises(ValueError, match="only 10 regions"):
        um.read_seqlets(path, regions=regions[:10])
    with pytest.raises(ValueError, match="give regions too"):
        um.read_seqlets(path, window=100)
    with pytest.raises(ValueError, match="window is 300 wide"):
        um.read_seqlets(path, regions=regions, window=300)
    with pytest.raises(ValueError, match="past the window of 50"):
        um.read_seqlets(path, regions=regions, window=50)
    with pytest.raises(ValueError, match="window must be"):
        um.read_seqlets(path, regions=regions, window=0)
    import sys

    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="universalmotif\\[h5py\\]"):
        um.read_seqlets(path)


def test_read_seqlets_widths_and_hits(tmp_path):

    h5py = pytest.importorskip("h5py")
    path = _modisco_file(tmp_path / "modisco.h5")
    with h5py.File(path, "a") as f:
        g = f["neg_patterns/pattern_0"]
        del g["seqlets"]
        g.create_group("seqlets").create_dataset("n_seqlets", data=np.array([8]))
        _add_seqlets(g, 8, 9, np.random.default_rng(5))
    s = um.read_seqlets(path, scores="hypothetical")
    assert s.schema["scores"] == pl.List(pl.List(pl.Float32))
    assert np.asarray(s.row(0, named=True)["scores"]).shape == (4, 12)
    assert np.asarray(s.row(s.height - 1, named=True)["scores"]).shape == (4, 9)
    # Seqlets are hits like any others.
    peaks = um.motif_peaks(s.drop("scores"), seq_length=100, qvalue=1, nthreads=1)
    assert set(peaks["motif"]) <= set(s["motif"])
    with h5py.File(path, "a") as f:
        del f["pos_patterns/pattern_10/seqlets/start"]
    with pytest.raises(ValueError, match="pos_patterns.pattern_10 has no seqlets"):
        um.read_seqlets(path, patterns=["pos_patterns.pattern_10"])
