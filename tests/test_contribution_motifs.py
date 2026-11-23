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
    assert pats.columns == ["pattern", "sign", "representative", "consensus", "seqlets", "share", "members",
                            "motif", "cwm"]
    assert pats.height == found.motifs["pattern"].n_unique() >= 2
    assert sorted(sum(pats["members"].to_list(), [])) == sorted(found.motifs["name"])
    assert pats["seqlets"].to_list() == sorted(pats["seqlets"], reverse=True)
    # Patterns are numbered by the seqlets they reach.
    assert pats["pattern"].to_list() == [f"pos_p{k + 1}" for k in range(pats.height)]
    assert ((pats["share"] > 0) & (pats["share"] <= 1)).all()
    reach = set().union(*(set(found.reach[m]) for m in found.motifs["name"]))
    assert reach <= set(range(found.seqlets.height))
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


def test_later_rounds_can_have_a_lower_min_gain():
    """AP-1 in half of 600 sequences, GATA in a third, and a rare E-box in
    one in 25, under 5% of the seqlets."""
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(600)]
    scores = rng.normal(0, 0.005, (600, 120))
    for i, s in enumerate(seqs):
        if i % 2 == 0:
            s = s[:30] + "TGACTCA" + s[37:]
            scores[i, 30:37] += 0.1
        if i % 3 == 0:
            s = s[:80] + "AGATAAG" + s[87:]
            scores[i, 80:87] += 0.08
        if i % 25 == 7:
            s = s[:55] + "GCCACGTGAC" + s[65:]
            scores[i, 55:65] += 0.1
        seqs[i] = s
    strict = um.contribution_motifs(scores, seqs, seed=1, gapped=False, hits=False, min_gain=0.06)
    assert not any("CACGTG" in c for c in strict.motifs["consensus"])
    # The second round searches the seqlets left unreached, where the E-box is common, and keeps it.
    later = um.contribution_motifs(scores, seqs, seed=1, gapped=False, hits=False, min_gain=0.06,
                                   later_min_gain=0.01)
    ebox = later.motifs.filter(pl.col("consensus").str.contains("ACGTG|CACGT"))
    assert ebox.height == 1 and ebox["name"][0].startswith("pos_r") and ebox["gain"][0] < 0.06
    assert later.motifs.head(2)["consensus"].to_list() == strict.motifs["consensus"].to_list()
    with pytest.raises(ValueError, match="later_min_gain"):
        um.contribution_motifs(scores, seqs, later_min_gain=2)


def test_contribution_test_drops_motifs_of_composition_alone():
    """AP-1 in half of 400 sequences, and in the other half a run of ten G
    and C in any order, which the stand-in model rewards for its
    composition alone."""
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(400)]
    scores = rng.normal(0, 0.005, (400, 120))
    for i, s in enumerate(seqs):
        if i % 2 == 0:
            s = s[:30] + "TGACTCA" + s[37:]
            scores[i, 30:37] += 0.1
        else:
            s = s[:70] + "".join(rng.choice(list("GC"), 10)) + s[80:]
            scores[i, 70:80] += 0.06
        seqs[i] = s
    def gc_only(found):
        return [c for c in found.motifs["consensus"] if set(c) <= set("GCS")]

    loose = um.contribution_motifs(scores, seqs, seed=1, gapped=False, hits=False, contribution_test=None)
    assert gc_only(loose)
    tested = um.contribution_motifs(scores, seqs, seed=1, gapped=False, hits=False)
    assert tested.motifs["consensus"][0] == "TGACTCA" and not gc_only(tested)
    # Gains count only the motifs left after the test.
    assert tested.motifs["cumulative"].max() == pytest.approx(tested.motifs["gain"].sum())
    assert len(tested) < len(loose)
    # Missing scores count as 0 rather than voiding every match after them.
    gaps = scores.copy()
    gaps[1::40, 0] = np.nan
    gaps[3::2, 110:] = np.nan
    with_gaps = um.contribution_motifs(gaps, seqs, seed=1, gapped=False, hits=False)
    assert with_gaps.motifs["consensus"][0] == "TGACTCA"
    with pytest.raises(ValueError, match="contribution_test"):
        um.contribution_motifs(scores, seqs, contribution_test=0)


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
    for bad, match in [({"min_gain": 1.5}, "min_gain must"), ({"seqlet_passes": 0}, "seqlet_passes"),
                       ({"max_seqlets": 100.0}, "max_seqlets")]:
        with pytest.raises(ValueError, match=match):
            um.contribution_motifs(model["scores"], model["seqs"], **bad)


def test_motifs_shuffling_cannot_change_fail():
    """A motif with one informative column has no shuffled copies that
    differ from it, so its order cannot be shown to matter and it fails,
    as a run of one letter does."""
    from universalmotif._contribution_motifs import _contribution_test

    rng = np.random.default_rng(2)
    seqs = ["".join(rng.choice(list("ACGT"), 100)) for _ in range(400)]
    scores = rng.normal(0, 0.005, (400, 100))
    m = um.create_motif(np.array([[0.4, 0.2, 0.2, 0.2], [0.97, 0.01, 0.01, 0.01], [0.4, 0.2, 0.2, 0.2],
                                  [0.4, 0.2, 0.2, 0.2], [0.4, 0.2, 0.2, 0.2], [0.4, 0.2, 0.2, 0.2]]).T,
                        type="PPM", name="one", alphabet="DNA")
    failed = _contribution_test([{"name": "one", "motif": m}], seqs, scores, "+", 0.01,
                                np.random.default_rng(1), 1)
    assert failed == {"one"}
    run = um.create_motif("AAAAAAA", name="run")
    seqs = [s[:40] + "AAAAAAA" + s[47:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    scores[::2, 40:47] += 0.1
    failed = _contribution_test([{"name": "run", "motif": run}], seqs, scores, "+", 0.01,
                                np.random.default_rng(1), 1)
    assert failed == {"run"}


def test_null_scores_set_the_seqlet_cut(model):
    scores = model["scores"]
    loud = np.random.default_rng(2).normal(0, 0.005, (300, 120))
    loud[:, 50:57] += 0.1
    quiet = np.random.default_rng(3).normal(0, 0.005, (300, 120))
    assert um.call_seqlets(scores, null_scores=loud, sign="positive").height < 20
    assert um.call_seqlets(scores, null_scores=quiet, sign="positive").height >= 240


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
    # One row per pattern, after the header's and the group heading's.
    assert page.count("<tr>") == found.patterns.height + 1
    assert page.count(">positive</td>") == found.patterns.height
    assert "FOS::JUN" in page and "<svg" in page and ">Matches</th>" in page
    # Each pattern's motifs can be expanded to their CWMs.
    assert page.count("<details>") == found.patterns.height and "CWM" in page
    # Each row has a button for its reverse complement, with every logo drawn both ways.
    assert page.count('<button class="rc"') == found.patterns.height
    assert page.count('<span class="fw">') == page.count('<span class="rv">') > 0
    # Without a database the report still lists the patterns, with no match columns.
    bare = found.report()
    assert bare.count("<tr>") == found.patterns.height + 1 and "Matches" not in bare
    # plot_modisco() draws the same report, and writes the same page.
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    fig = um.plot_modisco(found, database=db, html=tmp_path / "same.html")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts[:5] == ["Pattern", "Seqlets", "CWM", "ICM", "Matches"]
    assert ">Hits</th>" not in page
    hits = um.plot_modisco(found, columns=["seqlets", "hits", "CWM"])
    counts = found.hits.group_by("motif").len()
    first = found.patterns.row(0, named=True)
    n = counts.filter(pl.col("motif").is_in(first["members"]))["len"].sum()
    assert [t.get_text() for t in hits.axes[0].texts][:4] == ["Pattern", "Seqlets", "Hits", "CWM"]
    assert f"{n:,}" in [t.get_text() for t in hits.axes[0].texts]
    matplotlib.pyplot.close(hits)
    assert [t for t in texts if t.startswith("pos_p")] == found.patterns["pattern"].to_list()
    assert "FOS::JUN" in texts
    assert (tmp_path / "same.html").read_text() == page
    matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError, match="top"):
        um.name_motifs(found.ppms, db, top=0)


def test_idealized_sequences_stand_in_for_hypothetical_scores(model):
    seqs, scores = model["seqs"], model["scores"]
    rng = np.random.default_rng(5)
    # Hypothetical scores that prefer the planted letters, and other letters elsewhere.
    hyp = rng.normal(0, 0.004, (300, 4, 120))
    codes = np.array([["ACGT".index(c) for c in s] for s in seqs])
    np.put_along_axis(hyp, codes[:, None, :], scores[:, None, :], axis=1)
    from_scores = um.contribution_motifs(scores, seqs, hypothetical=hyp, seed=1, hits=False)
    ideal = um.idealize_sequences(seqs, hyp)
    given = um.contribution_motifs(scores, seqs, idealized=ideal, seed=1, hits=False)
    assert given.motifs.drop("motif", "cwm").equals(from_scores.motifs.drop("motif", "cwm"))
    both = um.contribution_motifs(scores, seqs, idealized={"positive": ideal, "negative": seqs}, sign="both",
                                  seed=1, hits=False)
    assert both.motifs.height >= 1
    with pytest.raises(ValueError, match="not both"):
        um.contribution_motifs(scores, seqs, hypothetical=hyp, idealized=ideal)
    with pytest.raises(ValueError, match="dict of 'positive' and 'negative'"):
        um.contribution_motifs(scores, seqs, idealized=ideal, sign="both")
    with pytest.raises(ValueError, match="number and length"):
        um.contribution_motifs(scores, seqs, idealized=ideal[:10])


def test_max_seqlets_searches_a_sample(model):
    full = um.contribution_motifs(model["scores"], model["seqs"], seed=1, hits=False)
    capped = um.contribution_motifs(model["scores"], model["seqs"], seed=1, hits=False, max_seqlets=100)
    assert full.seqlets.height > 100 and capped.seqlets.height == 100
    # The sample is of the seqlets called, and the same for the same seed.
    key = ["sequence_i", "start", "end"]
    assert capped.seqlets.join(full.seqlets, on=key, how="semi").height == 100
    again = um.contribution_motifs(model["scores"], model["seqs"], seed=1, hits=False, max_seqlets=100)
    assert again.seqlets.equals(capped.seqlets)
    cons = capped.motifs["consensus"].to_list()
    assert any("TGACTCA" in c or "TGAGTCA" in c for c in cons)
    # A cap above the seqlets there are changes nothing.
    loose = um.contribution_motifs(model["scores"], model["seqs"], seed=1, hits=False, max_seqlets=10**6)
    assert loose.motifs["name"].to_list() == full.motifs["name"].to_list()
    with pytest.raises(ValueError, match="max_seqlets"):
        um.contribution_motifs(model["scores"], model["seqs"], max_seqlets=5)
