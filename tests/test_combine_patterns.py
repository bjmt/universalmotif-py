import numpy as np
import polars as pl
import pytest
import universalmotif as um


def _cwm(core, name, n=100, sign=1.0):
    m = np.full((4, len(core)), 0.01)
    m[["ACGT".index(c) for c in core], range(len(core))] = 1.0
    return um.create_motif(sign * m, alphabet="DNA", type="CWM", name=name, nsites=n)


def _ppm(core, name, n=100):
    m = np.full((4, len(core)), 0.02)
    m[["ACGT".index(c) for c in core], range(len(core))] = 0.94
    return um.create_motif(m, alphabet="DNA", type="PPM", name=name, nsites=n)


def test_clusters_across_sources():
    runs = {
        "a": [_cwm("TGACTCAT", "pos_patterns.pattern_0", 300), _cwm("CCAATCAG", "pos_patterns.pattern_1", 100)],
        "b": [_cwm("TGACTCAT", "pos_patterns.pattern_0", 200), _cwm("GATAAGGC", "pos_patterns.pattern_1", 50)],
        "c": [_cwm("CCAATCAG", "pos_patterns.pattern_0", 80)],
    }
    comp = um.combine_patterns(runs)
    assert len(comp) == 3 and repr(comp) == "CombinedPatterns(3 clusters of 5 patterns from 3 sources)"
    assert comp.clusters.columns[-6:] == ["truncated", "mixed_sign", "low_complexity", "diffuse", "cwm", "ppm"]
    cl = comp.clusters
    assert cl["cluster"].to_list() == ["pos_c1", "pos_c2", "pos_c3"]
    assert cl["seqlets"].to_list() == [500, 180, 50]
    assert cl["sources"].to_list() == [["a", "b"], ["a", "c"], ["b"]]
    assert cl["n_sources"].to_list() == [2, 2, 1]
    assert cl["min_similarity"][0] == pytest.approx(1.0) and cl["min_similarity"][2] == 1.0
    assert cl["nearest"][0] in ("pos_c2", "pos_c3") and cl["nearest_similarity"][0] < 0.9
    pats = comp.patterns
    assert pats.columns == ["source", "pattern", "sign", "seqlets", "cluster", "similarity", "truncated", "cwm", "ppm"]
    assert pats.filter(pl.col("source") == "c")["cluster"].item() == "pos_c2"
    assert pats["pattern"][0] == "pos_0"
    assert comp.cwms[0].name == "pos_c1" and comp.cwms[0].nsites == 500
    # Patterns came without PPMs, so the clusters have none.
    assert comp.ppms == [None, None, None]


def test_averages_line_up_across_strands():
    fwd = _cwm("GGTGACTCATCC", "pos_patterns.pattern_0", 300)
    rev = um.motif_rc(_cwm("GGTGACTCATCC", "pos_patterns.pattern_0", 100))
    runs = {"a": {"CWM": [fwd], "PPM": [_ppm("GGTGACTCATCC", "pos_patterns.pattern_0", 300)]},
            "b": {"CWM": [rev], "PPM": [um.motif_rc(_ppm("GGTGACTCATCC", "pos_patterns.pattern_0", 100))]}}
    comp = um.combine_patterns(runs)
    assert len(comp) == 1
    cwm, ppm = comp.cwms[0], comp.ppms[0]
    # The pattern with the most seqlets sets the strand, and the PPM follows the CWM.
    assert ppm.consensus == "GGTGACTCATCC" == comp.clusters["consensus"][0]
    assert np.asarray(cwm.matrix).argmax(axis=0).tolist() == np.asarray(ppm.matrix).argmax(axis=0).tolist()
    assert comp.patterns["similarity"].to_list() == pytest.approx([1.0, 1.0])


def test_trimmed_averages():
    a = _cwm("TGACTCA", "pos_patterns.pattern_0", 300)
    # Columns at 0.05 of the largest go at the default trim.
    m = np.zeros((4, 13))
    m[:, 3:10] = np.asarray(a.matrix)
    m[0, [0, 12]] = 0.05
    wide = um.create_motif(m, alphabet="DNA", type="CWM", name="pos_patterns.pattern_0", nsites=100)
    comp = um.combine_patterns({"a": [a], "b": [wide]})
    assert comp.cwms[0].width == 7
    assert um.combine_patterns({"a": [a], "b": [wide]}, trim_threshold=None).cwms[0].width == 13


def test_linkage():
    # b is like both a and c, but a and c are less alike.
    base = np.full((4, 10), 0.01)
    base[["ACGT".index(x) for x in "TGACTCATGC"], range(10)] = 1.0
    mid = base.copy()
    mid[:, 8:] = 0.3
    far = mid.copy()
    far[:, 6:8] = 0.3
    ms = [um.create_motif(x, alphabet="DNA", type="CWM", name=f"pos_patterns.pattern_{i}", nsites=10)
          for i, x in enumerate((base, mid, far))]
    sim = np.asarray(um.compare_motifs(ms))
    cut = (sim[0, 2] + min(sim[0, 1], sim[1, 2])) / 2
    assert sim[0, 2] < cut <= min(sim[0, 1], sim[1, 2])
    assert len(um.combine_patterns({"x": ms}, similarity=cut, linkage="single")) == 1
    complete = um.combine_patterns({"x": ms}, similarity=cut)
    assert len(complete) == 2 and (complete.clusters["min_similarity"] >= cut).all()


def test_signs_cluster_apart():
    pos = _cwm("TGACTCA", "pos_patterns.pattern_0", 100)
    neg = _cwm("TGACTCA", "neg_patterns.pattern_0", 40, sign=-1.0)
    comp = um.combine_patterns({"a": [pos, neg], "b": [_cwm("TGACTCA", "pos_patterns.pattern_0", 50)]})
    assert comp.clusters["cluster"].to_list() == ["pos_c1", "neg_c1"]
    assert comp.clusters["sign"].to_list() == ["+", "-"]
    assert comp.clusters["nearest"].to_list() == [None, None]


def test_flags():
    rng = np.random.default_rng(3)

    def raw(m, name, n=100):
        return um.create_motif(m, alphabet="DNA", type="CWM", name=name, nsites=n)

    clean = _cwm("TTGACTCATG", "pos_patterns.pattern_0", 300)
    run = _cwm("AAAAAAAAAA", "pos_patterns.pattern_1", 200)
    rep = _cwm("CACACACACA", "pos_patterns.pattern_2", 150)
    mixed = np.asarray(_cwm("GGATTACCGT", "x").matrix).copy()
    mixed[:, 5:] *= -0.9
    smear = np.abs(rng.normal(0, 0.1, (4, 30)))
    combined = um.combine_patterns({"a": [clean, run, rep, raw(mixed, "pos_patterns.pattern_3", 120),
                                          raw(smear, "pos_patterns.pattern_4", 110)]}, trim_threshold=None)
    by_n = {r[0]: r[1:] for r in combined.clusters.select("seqlets", "truncated", "mixed_sign",
                                                          "low_complexity", "diffuse").rows()}
    assert by_n[300] == (False, False, False, False)
    assert by_n[200] == (False, False, True, False)
    assert by_n[150] == (False, False, True, False)
    assert by_n[120] == (False, True, False, False)
    assert by_n[110][3] is True
    # Motifs given directly have no window, so none is truncated.
    assert not combined.patterns["truncated"].any()


def test_truncated_patterns_of_a_file(tmp_path):
    h5py = pytest.importorskip("h5py")

    def pattern(grp, name, n, core, start):
        g = grp.create_group(name)
        contrib = np.full((20, 4), 0.001)
        seq = np.full((20, 4), 0.25)
        for i, letter in enumerate(core):
            contrib[start + i, "ACGT".index(letter)] = 1.0
            seq[start + i] = 0.02
            seq[start + i, "ACGT".index(letter)] = 0.94
        g.create_dataset("contrib_scores", data=contrib)
        g.create_dataset("hypothetical_contribs", data=contrib)
        g.create_dataset("sequence", data=seq)
        g.create_group("seqlets").create_dataset("n_seqlets", data=np.array([n]))

    path = tmp_path / "modisco.h5"
    with h5py.File(path, "w") as f:
        pos = f.create_group("pos_patterns")
        pattern(pos, "pattern_0", 300, "TGACTCAT", 6)
        # This core runs into the end of its window.
        pattern(pos, "pattern_1", 100, "CCGTTA", 14)
    combined = um.combine_patterns({"run": path})
    assert combined.patterns["truncated"].to_list() == [False, True]
    assert combined.clusters["truncated"].to_list() == [False, True]


def test_named_by_database():
    jun = _ppm("TGACTCAT", "MA0099.3")
    jun = jun.replace(altname="FOS::JUN")
    runs = {"a": {"CWM": [_cwm("TGACTCAT", "pos_patterns.pattern_0", 300), _cwm("GGCCAATG", "pos_patterns.pattern_1")],
                  "PPM": [_ppm("TGACTCAT", "pos_patterns.pattern_0", 300), _ppm("GGCCAATG", "pos_patterns.pattern_1")]}}
    combined = um.combine_patterns(runs, database=[jun, _ppm("CCCCGGGG", "MA0001.1")], qvalue=1)
    cl = combined.clusters
    assert cl["match"][0] == "MA0099.3" and cl["match_name"][0] == "FOS::JUN" and cl["match_qvalue"][0] <= 1
    assert "match" not in um.combine_patterns(runs).clusters.columns


def test_contribution_motifs_source():
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(300)]
    seqs = [s[:50] + "TGACTCA" + s[57:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    scores = rng.normal(0, 0.005, (300, 120))
    scores[::2, 50:57] += 0.1
    found = um.contribution_motifs(scores, seqs, seed=1)
    comp = um.combine_patterns({"model": found, "other": [_cwm("TGACTCA", "pos_patterns.pattern_0", 20)]})
    assert len(comp) == 1 and comp.clusters["sources"].to_list() == [["model", "other"]]
    assert comp.patterns["pattern"].to_list() == ["pos_p1", "pos_0"]
    # The other source has no PPM, so neither has the cluster.
    assert comp.ppms == [None]
    assert len(um.combine_patterns({"model": found}).ppms[0].consensus) == 7


def test_report(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    runs = {
        "run 1": {"CWM": [_cwm("TGACTCAT", "pos_patterns.pattern_0", 300)],
                  "PPM": [_ppm("TGACTCAT", "pos_patterns.pattern_0", 300)]},
        "run 2": {"CWM": [_cwm("TGACTCAT", "pos_patterns.pattern_0", 200)],
                  "PPM": [_ppm("TGACTCAT", "pos_patterns.pattern_0", 200)]},
    }
    comp = um.combine_patterns(runs)
    fig = um.plot_modisco(comp, html=tmp_path / "comp.html")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts[:5] == ["Pattern", "Seqlets", "Sources", "CWM", "ICM"]
    assert "pos_c1" in texts and "500" in texts and "2" in texts
    page = (tmp_path / "comp.html").read_text()
    assert "<title>Combined patterns</title>" in page and ">Sources</th>" in page
    # The cluster's patterns are listed with their sources.
    assert "run 1 pos_0" in page and "run 2 pos_0" in page and "2 CWMs" in page
    assert ">Flags</th>" in page
    assert comp.report() == page
    matplotlib.pyplot.close(fig)


def test_argument_checks():
    ok = [_cwm("TGACTCA", "pos_patterns.pattern_0")]
    with pytest.raises(ValueError, match="dict from source names"):
        um.combine_patterns({})
    with pytest.raises(ValueError, match="linkage"):
        um.combine_patterns({"a": ok}, linkage="average")
    with pytest.raises(ValueError, match="similarity must be a cosine"):
        um.combine_patterns({"a": ok}, similarity=2)
    with pytest.raises(ValueError, match="has no CWMs"):
        um.combine_patterns({"a": [_ppm("TGACTCA", "pos_patterns.pattern_0")]})
