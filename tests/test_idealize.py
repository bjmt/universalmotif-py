import numpy as np
import polars as pl
import pytest
import universalmotif as um


def test_idealize_swaps_strongly_preferred_letters():
    hyp = np.zeros((2, 4, 6))
    hyp[0, 2, 1] = 0.5
    hyp[0, 0, 3] = 0.005
    hyp[1, 3, 0] = -0.5
    seqs = ["ACCTTT", "ANGGGG"]
    assert um.idealize_sequences(seqs, hyp) == ["AGCTTT", "ANGGGG"]
    # Letters the model wants least, for negative contributions.
    assert um.idealize_sequences(seqs, hyp, sign="negative")[1] == "TNGGGG"
    # The cutoff, the length-last layout and mappings.
    assert um.idealize_sequences(seqs, hyp, threshold=0.001)[0] == "AGCATT"
    assert um.idealize_sequences(seqs, np.moveaxis(hyp, 1, 2)) == ["AGCTTT", "ANGGGG"]
    assert um.idealize_sequences({"a": "ACCTTT", "b": "ANGGGG"}, hyp) == {"a": "AGCTTT", "b": "ANGGGG"}
    # Lower case is kept where nothing changes.
    assert um.idealize_sequences("acctTT", hyp[:1]) == "aGctTT"


def test_idealize_errors():
    hyp = np.zeros((2, 4, 6))
    with pytest.raises(ValueError, match="sign"):
        um.idealize_sequences(["ACGTAC"] * 2, hyp, sign="up")
    with pytest.raises(ValueError, match="2 sequences"):
        um.idealize_sequences(["ACGTAC"] * 3, hyp)
    with pytest.raises(ValueError, match="longer"):
        um.idealize_sequences(["ACGTACG", "A"], hyp)


def test_cwms_from_hits():
    rng = np.random.default_rng(5)
    seqs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(50)]
    seqs = [s[:20] + "TGACTCA" + s[27:] for s in seqs]
    scores = np.full((50, 60), 0.01)
    scores[:, 20:27] = [0.1, 0.2, 0.3, 0.1, 0.3, 0.2, 0.1]
    hits = um.scan_sequences(um.create_motif("TGACTCA", name="ap1"), seqs, rc=False)
    hc = um.hit_contributions(hits, scores)
    cwm = next(m for m in hc.to_cwms() if m.name == "ap1")
    assert cwm.type == "CWM" and cwm.width == 7 and cwm.nsites == hits.height
    assert cwm.consensus == "TGACTCA"
    assert cwm.matrix.max(axis=0).round(2).tolist() == [0.1, 0.2, 0.3, 0.1, 0.3, 0.2, 0.1]
    hyp = um.hit_contributions(hits, um.one_hot(seqs) * scores[:, None, :], hypothetical=True, flank=2)
    wide = hyp.to_cwms(hypothetical=True)[0]
    assert wide.width == 11
    with pytest.raises(ValueError, match="hypothetical"):
        hc.to_cwms(hypothetical=True)
    with pytest.raises(ValueError, match="match"):
        um.hit_contributions(hits.drop("match"), scores).to_cwms()


def test_finder_stability():
    site = um.create_motif("TTGACCAT", nsites=50)
    seqs = um.create_sequences(seqnum=200, seqlen=100, seed=1)
    seqs = um.implant_motifs(site, seqs[:100], seed=1) + seqs[100:]
    res = um.motif_finder(seqs, nmotifs=3, seed=1, stability=4)
    assert "stability" in res.columns and res["stability"][0] == 1.0
    assert "stability" not in um.motif_finder(seqs, nmotifs=3, seed=1).columns
    with pytest.raises(ValueError, match="stability"):
        um.motif_finder(seqs, stability=2, stability_fraction=1.0)
    assert isinstance(res, pl.DataFrame)
