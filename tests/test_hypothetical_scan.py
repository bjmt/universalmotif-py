import numpy as np
import pytest
import universalmotif as um

LETTERS = "ACGT"


def scores(seq, wanted, w=0.5):
    """Hypothetical scores for seq, the model wanting the letters of
    `wanted` (or nothing much at a '.'), in the layout (1, 4, length)."""
    hyp = np.full((1, 4, len(seq)), 0.01, dtype=np.float32)
    for p, c in enumerate(wanted):
        if c != ".":
            hyp[0, :, p] = -w / 3
            hyp[0, LETTERS.index(c), p] = w
    return hyp


def test_a_database_style_motif_matches_what_the_model_wants():
    seq, wanted = "AAAATGAATCAAAA", "....TGACTCA..."
    hits = um.scan_contributions(um.create_motif("TGACTCA", name="AP-1"), scores(seq, wanted), seq,
                                 hypothetical=True, min_importance=0.1)
    best = hits.sort("score", descending=True).row(0, named=True)
    # The site has the wrong middle letter, but the model wants it there.
    assert (best["start"], best["end"], best["match"]) == (4, 11, "TGAATCA")
    assert best["similarity"] > 0.9


def test_thresholds_and_their_defaults():
    seq, wanted = "CCCTTATCACCCGGGGGGGGGGGGGGGGGG", "...TTATCA....................."
    hyp = scores(seq, wanted)
    gata = um.create_motif("TGATAA", name="GATA")
    hits = um.scan_contributions(gata, hyp, seq, hypothetical=True, min_importance=0.1)
    assert hits.sort("score", descending=True)["strand"][0] == "-"
    # For a PPM, min_importance is a mean per position; the site's is 0.5.
    assert um.scan_contributions(gata, hyp, seq, hypothetical=True, min_importance=0.6).height == 0
    # By default it comes from the scores: the site stands out from the rest.
    assert um.scan_contributions(gata, hyp, seq, hypothetical=True).height >= 1


def test_cwms_and_ppms_can_be_mixed():
    seq, wanted = "AAGATAAGAATGACTCAAA", "..GATAAG..TGACTCA.."
    hyp = scores(seq, wanted, 0.4)
    cols = np.array([[0.4 if c == x else -0.4 / 3 for c in "GATAAG"] for x in LETTERS])
    cwm = um.create_motif(cols, type="CWM", alphabet="DNA", name="GATA-hCWM")
    hits = um.scan_contributions([cwm, um.create_motif("TGACTCA", name="AP-1")], hyp, seq,
                                 hypothetical=True, min_importance={"GATA-hCWM": 0.9, "AP-1": 0.1})
    assert set(hits["motif"]) == {"GATA-hCWM", "AP-1"}


def test_it_refuses_observed_scores_and_ppms_without_it():
    seq = "AAAATGACTCAAAA"
    hyp = scores(seq, "....TGACTCA...")
    observed = hyp * um.one_hot([seq])
    m = um.create_motif("TGACTCA")
    with pytest.raises(ValueError, match="at most one letter"):
        um.scan_contributions(m, observed, seq, hypothetical=True)
    with pytest.raises(ValueError, match="give the sequences"):
        um.scan_contributions(m, hyp, hypothetical=True)
    with pytest.raises(ValueError, match="matched with CWMs"):
        um.scan_contributions(m, hyp, seq)
