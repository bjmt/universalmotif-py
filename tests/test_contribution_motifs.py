import numpy as np
import polars as pl
import pytest
import universalmotif as um


@pytest.fixture(scope="module")
def model():
    """300 random sequences, AP-1 sites in the even ones and GATA sites in
    every third, scored by a stand-in model that relies on both."""
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(300)]
    seqs = [s[:30] + "TGACTCA" + s[37:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    seqs = [s[:80] + "AGATAAG" + s[87:] if i % 3 == 0 else s for i, s in enumerate(seqs)]
    scores = rng.normal(0, 0.005, (300, 120))
    scores[::2, 30:37] += 0.1
    scores[::3, 80:87] += 0.08
    return {"seqs": seqs, "scores": scores}


def test_finds_the_motifs_the_model_uses(model):
    found = um.contribution_motifs(model["scores"], model["seqs"], seed=1)
    assert isinstance(found, um.ContributionMotifs)
    assert found.motifs.columns == ["name", "sign", "source", "consensus", "width", "seqlets", "gain", "cumulative",
                                    "motif", "cwm"]
    cons = found.motifs["consensus"].to_list()
    assert any("TGACTCA" in c or "TGAGTCA" in c for c in cons)
    assert any("GATAAG" in c or "CTTATC" in c for c in cons)
    assert set(found.motifs["sign"]) == {"+"} and set(found.motifs["source"]) == {"seqlets"}
    assert all(m.type == "CWM" for m in found.cwms) and len(found.ppms) == len(found)
    assert found.seqlets.height >= 240
    gains = found.motifs["gain"].to_list()
    assert gains == sorted(gains, reverse=True)
    assert "ContributionMotifs(" in repr(found)


def test_inputs_and_options(model):
    seqs, scores = model["seqs"], model["scores"]
    # Projected scores carry their letters, so the sequences can be left out.
    projected = um.one_hot(seqs) * scores[:, None, :]
    same = um.contribution_motifs(projected, seed=1)
    assert same.motifs["consensus"].to_list() == um.contribution_motifs(scores, seqs, seed=1).motifs[
        "consensus"].to_list()
    # Given seqlets, by index or by name.
    seqlets = um.call_seqlets(scores, seqs, sign="positive", seed=1)
    by_index = um.contribution_motifs(scores, seqs, seqlets=seqlets.drop("sequence"), seed=1)
    assert len(by_index) >= 2
    named = {f"r{i}": s for i, s in enumerate(seqs)}
    by_name = um.contribution_motifs({f"r{i}": x for i, x in enumerate(scores)}, named,
                                     seqlets=seqlets.with_columns(sequence="r" + pl.col("sequence_i").cast(pl.String))
                                     .drop("sequence_i"), seed=1)
    assert len(by_name) == len(by_index)
    # Hypothetical scores add a search on idealized seqlets.
    hyp = projected * 1.0
    both = um.contribution_motifs(scores, seqs, hypothetical=hyp, seed=1)
    assert set(both.motifs["source"]) <= {"seqlets", "idealized"}
    stable = um.contribution_motifs(scores, seqs, stability=2, seed=1)
    assert "stability" in stable.motifs.columns and (stable.motifs["stability"] >= 0.75).all()
    neg = um.contribution_motifs(-scores, seqs, sign="negative", seed=1)
    assert set(neg.motifs["sign"]) == {"-"} and len(neg) >= 2


def test_errors(model):
    with pytest.raises(ValueError, match="sign"):
        um.contribution_motifs(model["scores"], model["seqs"], sign="up")
    with pytest.raises(ValueError, match="give the sequences"):
        um.contribution_motifs(model["scores"])
    with pytest.raises(ValueError, match="sequence_i"):
        um.contribution_motifs(model["scores"], model["seqs"], seqlets=pl.DataFrame({"start": [1], "end": [5]}))


def test_decoys_set_the_seqlet_cut(model):
    scores = model["scores"]
    loud = np.random.default_rng(2).normal(0, 0.005, (300, 120))
    loud[:, 50:57] += 0.1
    quiet = np.random.default_rng(3).normal(0, 0.005, (300, 120))
    assert um.call_seqlets(scores, decoys=loud, sign="positive").height < 20
    assert um.call_seqlets(scores, decoys=quiet, sign="positive").height >= 240
