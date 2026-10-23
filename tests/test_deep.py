"""Convolution filters, one-hot sequences and TF-MoDISco results."""

import numpy as np
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


def _modisco_file(path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)

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
