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
    assert found.motifs.columns == ["name", "pattern", "sign", "source", "consensus", "width", "seqlets", "gain",
                                    "cumulative", "min_similarity", "min_importance", "motif", "cwm"]
    cons = found.motifs["consensus"].to_list()
    assert any("TGACTCA" in c or "TGAGTCA" in c for c in cons)
    assert any("GATAAG" in c or "CTTATC" in c for c in cons)
    assert set(found.motifs["sign"]) == {"+"} and set(found.motifs["source"]) == {"seqlets"}
    assert all(m.type == "CWM" for m in found.cwms) and len(found.ppms) == len(found)
    assert found.seqlets.height >= 240
    gains = found.motifs["gain"].to_list()
    assert gains == sorted(gains, reverse=True)
    assert "ContributionMotifs(" in repr(found)
    # AP-1 and GATA are patterns of their own, each led by the motif reaching the most seqlets.
    pats = found.patterns
    assert pats.columns == ["pattern", "sign", "representative", "consensus", "seqlets", "members", "gain", "motif",
                            "cwm"]
    assert pats.height == found.motifs["pattern"].n_unique() >= 2
    assert sorted(sum(pats["members"].to_list(), [])) == sorted(found.motifs["name"])
    assert pats["seqlets"].to_list() == sorted(pats["seqlets"], reverse=True)
    assert abs(pats["gain"].sum() - found.motifs["gain"].sum()) < 1e-9
    # Hits of the CWMs at thresholds set from the seqlets reach most planted sites.
    assert found.thresholds.columns == ["motif", "min_similarity", "min_importance"]
    assert (found.thresholds["min_similarity"] > 0.9).all()
    hits = found.hits
    assert {"motif", "sequence_i", "start", "end", "strand", "similarity", "importance"} <= set(hits.columns)
    ap1 = hits.filter(pl.col("motif") == found.motifs.filter(pl.col("consensus").str.contains("TGA.TCA"))["name"][0])
    assert ap1["sequence_i"].n_unique() >= 130 and (ap1["sequence_i"] % 2 == 0).mean() > 0.95
    assert um.contribution_motifs(model["scores"], model["seqs"], seed=1, hits=False).hits is None


def test_composites():
    """CAGCTG then AGATAAG 7 or 8 letters on, in every third of 400
    sequences, beside AP-1 sites in half."""
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(400)]
    scores = rng.normal(0, 0.005, (400, 120))
    for i, s in enumerate(seqs):
        if i % 2 == 0:
            s = s[:20] + "TGACTCA" + s[27:]
            scores[i, 20:27] += 0.1
        if i % 3 == 0:
            gap = 7 + (i // 3) % 2
            site = "CAGCTG" + s[56:56 + gap] + "AGATAAG"
            s = s[:50] + site + s[50 + len(site):]
            scores[i, 50:56] += 0.05
            scores[i, 56 + gap:63 + gap] += 0.08
        seqs[i] = s
    found = um.contribution_motifs(scores, seqs, seed=1)
    comp = found.motifs.filter(pl.col("source") == "gapped")
    assert comp.height >= 1 and found.motifs["source"].to_list()[-comp.height:] == ["gapped"] * comp.height
    for m in comp["motif"]:
        assert m.isgapped and m.mingap[0] >= 2
        assert "CAGCTG" in m.consensus and ("CTTATCT" in m.consensus or "AGATAAG" in m.consensus)
    assert comp["min_similarity"].is_not_null().all()
    plain = um.contribution_motifs(scores, seqs, seed=1, gapped=False, rounds=1)
    assert not (plain.motifs["source"] == "gapped").any()
    # Composites lead patterns of their own; without grouping every motif is a pattern.
    assert all(len(m) == 1 for m in found.patterns.filter(pl.col("representative").is_in(comp["name"]))["members"])
    ungrouped = um.contribution_motifs(scores, seqs, seed=1, pattern_similarity=None)
    assert ungrouped.patterns.height == len(ungrouped)


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


def test_report_and_names(model, tmp_path):
    found = um.contribution_motifs(model["scores"], model["seqs"], seed=1)
    db = [um.create_motif("TGACTCA", name="MA0099", altname="FOS::JUN"),
          um.create_motif("AGATAAG", name="MA0035", altname="GATA1"),
          um.create_motif("CACGTG", name="MA0004", altname="Arnt")]
    names = um.name_motifs(found.patterns["motif"].to_list(), db, top=2, qvalue=1)
    best = dict(names.filter(pl.col("rank") == 1).select("consensus", "altname").iter_rows())
    assert any("TGA" in c and v == "FOS::JUN" for c, v in best.items())
    assert any("GATA" in c or "TATC" in c for c, v in best.items() if v == "GATA1")
    assert names.group_by("motif_i").len()["len"].max() <= 2
    assert (names["qvalue"] >= names["pvalue"]).all()
    path = tmp_path / "report.html"
    page = found.report(path, database=db)
    assert path.read_text() == page and page.startswith("<!doctype html>")
    # One row per pattern, after the header's.
    assert page.count("</tr>") == found.patterns.height + 1
    assert "FOS::JUN" in page and "<svg" in page
    # Without a database the report still lists the patterns.
    assert found.report().count("</tr>") == found.patterns.height + 1
    with pytest.raises(ValueError, match="top"):
        um.name_motifs(found.ppms, db, top=0)
