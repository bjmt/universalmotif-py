"""The Motif object and the Python-level behaviour of the API."""

import math
import pickle
import warnings

import numpy as np
import polars as pl
import pytest

import universalmotif as um


@pytest.fixture
def tata():
    return um.create_motif("TATAWAW", nsites=50, name="tata", pseudocount=1)


def test_properties(tata):
    assert tata.name == "tata"
    assert tata.alphabet == "DNA"
    assert tata.letters == ["A", "C", "G", "T"]
    assert tata.type == "PPM"
    assert tata.consensus == "TATAWAW"
    assert tata.nsites == 50
    assert tata.pseudocount == 1
    assert tata.bkg == {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    assert tata.strand == "+-"
    assert tata.altname is None and tata.pval is None
    assert len(tata) == tata.width == 7
    assert not tata.isgapped and tata.gaploc == []


def test_matrix_is_read_only_copy(tata):
    m = tata.matrix
    assert m.shape == (4, 7)
    assert m.dtype == np.float64
    with pytest.raises(ValueError):
        m[0, 0] = 5.0
    np.testing.assert_allclose(m.sum(axis=0), 1.0)
    # The pseudocount of 1 is applied when the consensus columns are cleaned up.
    np.testing.assert_allclose(m[:, 0], [0.25 / 51, 0.25 / 51, 0.25 / 51, 50.25 / 51])


def test_repr(tata):
    text = repr(tata)
    assert "Motif name:   tata" in text
    assert "Consensus:   TATAWAW" in text
    assert text.splitlines()[-1].startswith("T ")


def test_equality_and_pickle(tata):
    same = um.create_motif("TATAWAW", nsites=50, name="tata", pseudocount=1)
    assert tata == same
    assert tata != um.create_motif("TATAWAT", nsites=50, name="tata", pseudocount=1)
    assert tata != "TATAWAW"
    restored = pickle.loads(pickle.dumps(tata))
    assert restored == tata
    assert repr(restored) == repr(tata)


def test_replace(tata):
    m = tata.replace(name="new", pval=1e-5, bkg={"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3},
                     extrainfo={"source": "test"})
    assert m.name == "new" and m.pval == 1e-5
    assert m.bkg["A"] == 0.3
    assert m.extrainfo == {"source": "test"}
    assert tata.name == "tata"
    with pytest.raises(TypeError):
        tata.replace(consensus="AAAA")
    with pytest.raises(ValueError, match="add up to 1"):
        tata.replace(bkg=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="strand"):
        tata.replace(strand="x")


def test_create_from_matrix():
    counts = [[5, 2], [0, 2], [0, 3], [5, 3]]
    m = um.create_motif(counts, letters="ACGT", type="PCM")
    assert m.type == "PCM" and m.nsites == 10
    shuffled = um.create_motif([[5, 3], [0, 3], [0, 2], [5, 2]], letters="TGCA", type="PCM")
    np.testing.assert_array_equal(shuffled.matrix, m.matrix)
    with pytest.raises(ValueError, match="alphabet"):
        um.create_motif(np.array(counts))
    ppm = um.create_motif(np.array(counts), alphabet="DNA")
    assert ppm.type == "PPM"
    with pytest.raises(ValueError, match="NaN"):
        um.create_motif([[math.nan], [1], [0], [0]], alphabet="DNA")
    with pytest.raises(ValueError, match="2-D"):
        um.create_motif(np.zeros(4))


def test_create_from_strings():
    m = um.create_motif(["AAGT", "ACGT", "ACCT", "TCGT"], type="PCM")
    assert m.nsites == 4
    np.testing.assert_array_equal(m.matrix[:, 0], [3, 0, 0, 1])
    aa = um.create_motif("ACDEFX")
    assert aa.alphabet == "AA" and aa.matrix.shape == (20, 6)
    custom = um.create_motif(["xyz", "xzz"])
    assert custom.alphabet == "xyz" and custom.consensus is None
    with pytest.raises(ValueError, match="equal length"):
        um.create_motif(["AAA", "AA"])


def test_create_random():
    a = um.create_motif(8, seed=1)
    b = um.create_motif(8, seed=1)
    assert a == b and a.width == 8 and a.type == "PPM"
    assert 50 <= a.nsites <= 250
    assert um.create_motif(seed=2).width == 10
    pcm = um.create_motif(5, type="PCM", seed=3, bkg=[0.25] * 4)
    assert pcm.nsites == 100 and pcm.matrix.sum(axis=0).tolist() == [100] * 5
    rna = um.create_motif(4, alphabet="RNA", seed=4)
    assert rna.letters[-1] == "U"
    with pytest.raises(TypeError, match="seed"):
        um.create_motif("ACGT", seed=1)


def test_lists_of_motifs(tata):
    out = um.convert_type([tata, tata], "PCM")
    assert isinstance(out, list) and [m.type for m in out] == ["PCM", "PCM"]
    assert um.convert_type((m for m in [tata]), "PWM")[0].type == "PWM"
    with pytest.raises(TypeError, match="expected a Motif"):
        um.convert_type([tata, "x"], "PCM")
    with pytest.raises(ValueError, match="CWM"):
        um.convert_type(tata, "CWM")


def test_trim_motifs_drops_and_warns(tata):
    flat = um.create_motif("NNNN", name="flat")
    with pytest.raises(ValueError, match="completely trimmed"):
        um.trim_motifs(flat)
    with pytest.warns(UserWarning, match="flat"):
        out = um.trim_motifs([tata, flat])
    assert [m.name for m in out] == ["tata"]
    with pytest.raises(ValueError, match="all motifs"):
        um.trim_motifs([flat, flat])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert um.trim_motifs(tata, trim_from="left").consensus == "TATAWAW"


def test_motif_rc_and_round(tata):
    rc = um.motif_rc(tata)
    assert rc.consensus == "WTWTATA"
    back = um.motif_rc(rc)
    assert back.consensus == tata.consensus
    np.testing.assert_array_equal(back.matrix, tata.matrix)
    aa = um.create_motif("ACDE")
    with pytest.raises(ValueError, match="DNA/RNA"):
        um.motif_rc(aa)
    rounded = um.round_motif(um.create_motif("ACGTN", nsites=100, pseudocount=0), "20%")
    np.testing.assert_allclose(rounded.matrix.sum(axis=0), 1.0)


def test_column_helpers():
    assert um.get_consensus([0.7, 0.1, 0.1, 0.1]) == "A"
    assert um.get_consensus([0.1, 0.1, 0.1, 0.7], alphabet="RNA") == "U"
    assert um.get_consensus([5, 5, 0, 0], type="PCM") == "M"
    with pytest.raises(ValueError):
        um.get_consensus([0.5, 0.5])
    with pytest.raises(ValueError):
        um.get_consensus([0.25] * 4, alphabet="AA")
    assert um.get_consensus_aa(um.consensus_to_ppm_aa("W")) == "W"
    np.testing.assert_allclose(um.consensus_to_ppm("R"), [0.499, 0.001, 0.499, 0.001])
    with pytest.raises(ValueError, match="IUPAC"):
        um.consensus_to_ppm("Z")
    assert round(um.position_icscore([0.25, 0.75]), 3) == 0.185


def test_summarize_motifs(tata):
    pcm = um.create_motif(["AAGT", "ACGT"], type="PCM", name="sites", pval=0.01)
    df = um.summarize_motifs([tata, pcm])
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["name", "consensus", "alphabet", "strand", "icscore", "nsites", "pval"]
    assert df["name"].to_list() == ["tata", "sites"]
    assert df["pval"].to_list() == [None, 0.01]
    full = um.summarize_motifs(tata, na_rm=False)
    assert full.height == 1 and "altname" in full.columns
    assert full.schema["nsites"] == pl.Float64


def test_replace_recomputes_derived_values():
    m = um.create_motif(["TTGACC", "TTGACT", "ATGACC"], type="PCM", pseudocount=0)
    more = m.replace(pseudocount=10)
    assert more.icscore < m.icscore
    assert m.replace(name="x").icscore == m.icscore


def test_create_motif_argument_checks():
    for width in (0, -2):
        with pytest.raises(ValueError, match="width must be greater than 0"):
            um.create_motif(width)
    for call in (lambda: um.create_motif("ACGT", letters="ACGT"),
                 lambda: um.create_motif(["ACGT", "ACGA"], letters="ACGT"),
                 lambda: um.create_motif(5, letters="ACGT"),
                 lambda: um.create_motif(letters="ACGT")):
        with pytest.raises(TypeError, match="letters is only used for matrix input"):
            call()


def test_random_motifs_other_alphabets():
    for alphabet, n in (("AA", 20), ("XYZ", 3), ("RNA", 4)):
        a = um.create_motif(6, alphabet=alphabet, seed=7)
        assert a == um.create_motif(6, alphabet=alphabet, seed=7)
        assert a.matrix.shape == (n, 6)
        np.testing.assert_allclose(a.matrix.sum(axis=0), 1)
    assert um.create_motif(6, seed=7) != um.create_motif(6, seed=8)


@pytest.mark.parametrize("alphabet", ["DNA", "RNA"])
def test_reverse_complement_is_an_involution(alphabet):
    for m in um.example_motifs()[:6]:
        if alphabet == "RNA":
            m = um.switch_alph(m)
        rc = um.motif_rc(m)
        assert rc.width == m.width and rc.alphabet == alphabet
        # Each column becomes its complement, in reverse order.
        np.testing.assert_array_equal(rc.matrix, m.matrix[::-1, ::-1])
        back = um.motif_rc(rc)
        np.testing.assert_array_equal(back.matrix, m.matrix)
        assert back.consensus == m.consensus and back.icscore == pytest.approx(m.icscore)
