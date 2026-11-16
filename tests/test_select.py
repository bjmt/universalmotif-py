import polars as pl
import pytest
import universalmotif as um


@pytest.fixture(scope="module")
def data():
    """300 random sequences with AP-1 sites in the first half and GATA sites
    in the second hundred."""
    seqs = um.create_sequences(seqlen=200, seqnum=300, seed=1)
    ap1 = um.create_motif("TGACTCA", name="AP-1")
    gata = um.create_motif("AGATAAG", name="GATA")
    seqs = um.implant_motifs([ap1], seqs[:150], seed=1) + seqs[150:]
    seqs = seqs[:100] + um.implant_motifs([gata], seqs[100:200], seed=2) + seqs[200:]
    return {"seqs": seqs, "ap1": ap1, "gata": gata, "longer": um.create_motif("TGACTCAT", name="AP-1, longer")}


def test_select_ranks_by_new_sequences(data):
    sel = um.select_motifs([data["gata"], data["ap1"], data["longer"]], data["seqs"])
    assert sel.columns == ["motif", "motif_i", "rank", "targets", "gain", "cumulative", "kept", "similar_to",
                           "object"]
    assert sel["motif"].to_list() == ["AP-1", "GATA", "AP-1, longer"]
    assert sel["rank"].to_list() == [1, 2, 3]
    assert sel["kept"].to_list() == [True, True, False]
    gains = sel["gain"].to_list()
    assert gains == sorted(gains, reverse=True)
    assert sel["cumulative"].to_list() == pytest.approx([sum(gains[:i + 1]) for i in range(3)])
    assert sel["targets"][0] >= 150
    assert sel.filter("kept")["object"].to_list() == [data["ap1"], data["gata"]]
    # min_gain of 0 keeps any motif adding a sequence.
    loose = um.select_motifs([data["gata"], data["ap1"], data["longer"]], data["seqs"], min_gain=0)
    assert loose["kept"][2] == (loose["gain"][2] > 0)


def test_select_with_sites(data):
    seqs = data["seqs"]
    hits = um.scan_sequences(data["gata"], seqs)
    sites = hits.select("sequence", "start", "end")
    sel = um.select_motifs([data["ap1"], data["gata"]], seqs, sites=sites)
    assert sel["motif"][0] == "GATA"
    assert sel["gain"][0] == pytest.approx(1.0)
    by_index = um.select_motifs([data["ap1"], data["gata"]], seqs, sites=hits.select("sequence_i", "start", "end"))
    assert by_index.drop("object").equals(sel.drop("object"))
    named = {f"s{i}": s for i, s in enumerate(seqs)}
    renamed = sites.with_columns(pl.col("sequence").str.replace("seq", "").cast(pl.Int64).sub(1)
                                 .cast(pl.String).str.replace("^", "s"))
    assert um.select_motifs([data["ap1"], data["gata"]], named, sites=renamed).drop("object").equals(
        sel.drop("object"))
    # A hit must share min_overlap positions with a site.
    shifted = sites.with_columns(pl.col("start") + 5, pl.col("end") + 5)
    far = um.select_motifs([data["gata"]], seqs, sites=shifted, min_overlap=3)
    assert far["gain"][0] == 0 and not far["kept"][0]


def test_select_leaves_out_similar_motifs(data):
    ms = [data["ap1"], data["longer"], data["gata"]]
    sel = um.select_motifs(ms, data["seqs"], max_similarity=0.8, min_gain=0)
    row = sel.filter(pl.col("motif") == "AP-1, longer").row(0, named=True)
    assert row["similar_to"] == "AP-1" and row["rank"] is None and not row["kept"]
    assert sel["motif"][-1] == "AP-1, longer"
    # Similarities can come from other motifs standing in for them: here
    # GATA and the longer AP-1 share a stand-in, and GATA comes first.
    stand_ins = [data["gata"], data["ap1"], data["ap1"]]
    other = um.select_motifs(ms, data["seqs"], max_similarity=0.99, compare_as=stand_ins, min_gain=0)
    assert other.filter(pl.col("motif") == "AP-1, longer")["similar_to"][0] == "GATA"
    assert other.filter(pl.col("motif") == "GATA")["kept"][0]
    # A single motif is kept whatever the cutoff.
    one = um.select_motifs([data["ap1"]], data["seqs"], max_similarity=0.5)
    assert one["kept"].to_list() == [True]


def test_select_inputs_and_errors(data):
    found = pl.DataFrame({"motif": [data["ap1"], data["gata"]]}, schema={"motif": pl.Object})
    assert um.select_motifs(found, data["seqs"])["motif"].to_list() == ["AP-1", "GATA"]
    with pytest.raises(ValueError, match="compare_as needs max_similarity"):
        um.select_motifs([data["ap1"]], data["seqs"], compare_as=[data["ap1"]])
    with pytest.raises(ValueError, match="1 motifs for 2"):
        um.select_motifs([data["ap1"], data["gata"]], data["seqs"], max_similarity=0.9, compare_as=[data["ap1"]])
    with pytest.raises(ValueError, match="min_gain"):
        um.select_motifs([data["ap1"]], data["seqs"], min_gain=2)
    with pytest.raises(ValueError, match="not given"):
        um.select_motifs([data["ap1"]], data["seqs"], sites=pl.DataFrame({"sequence": ["x"], "start": [0], "end": [5]}))
    with pytest.raises(ValueError, match="end after"):
        um.select_motifs([data["ap1"]], data["seqs"], sites=pl.DataFrame({"sequence_i": [0], "start": [5], "end": [5]}))
    with pytest.raises(ValueError, match="'end' column"):
        um.select_motifs([data["ap1"]], data["seqs"], sites=pl.DataFrame({"sequence_i": [0], "start": [5]}))
    with pytest.raises(ValueError, match="'motif' column"):
        um.select_motifs(pl.DataFrame({"x": [1]}), data["seqs"])


def test_refine_grows_a_core_to_the_whole_site():
    seqs = um.create_sequences(seqlen=200, seqnum=200, seed=3)
    site = um.create_motif("GCTTGACTCATGAC")
    seqs = um.implant_motifs([site], seqs[:100], seed=3) + seqs[100:]
    core = um.create_motif("TGACTCA", name="core", nsites=20, pseudocount=1)
    refined = um.refine_motifs(core, seqs)[0]
    assert refined.consensus == "GCTTGACTCATGAC"
    assert refined.name == "core" and refined.type == core.type
    assert refined.nsites == pytest.approx(103)
    capped = um.refine_motifs(core, seqs, max_width=9)[0]
    assert capped.width == 9
    # Hits outside the sites are not used, and too few leave a motif alone.
    sites = pl.DataFrame({"sequence_i": [0], "start": [0], "end": [200]})
    assert um.refine_motifs(core, seqs, sites=sites)[0] == core
    assert um.refine_motifs(core, seqs, iterations=0)[0] == core


def test_refine_keeps_a_weak_motif_apart():
    seqs = um.create_sequences(seqlen=250, seqnum=400, seed=9)
    ap1 = um.create_motif("TTGACTCATG")
    gata = um.create_motif("AGATAAGA")
    seqs = um.implant_motifs([ap1], seqs[:200], seed=4) + um.implant_motifs([gata], seqs[200:300], seed=5) + seqs[300:]
    start = [um.create_motif("TGACTCA", name="ap1", nsites=20, pseudocount=1),
             um.create_motif("AGATAAG", name="gata", nsites=20, pseudocount=1)]
    out = um.refine_motifs(start, seqs)
    assert [m.name for m in out] == ["ap1", "gata"]
    assert "GATAAG" in out[1].consensus or "CTTATC" in out[1].consensus
    assert "TGACTCA" in out[0].consensus or "TGAGTCA" in out[0].consensus
    with pytest.raises(ValueError, match="flank"):
        um.refine_motifs(start, seqs, flank=-1)
    with pytest.raises(ValueError, match="max_width"):
        um.refine_motifs(start, seqs, max_width=0)


def test_refine_gapped_motifs_part_by_part():
    seqs = um.create_sequences(seqlen=250, seqnum=300, seed=11)
    spacers = ["ACG", "TTCA", "GACTT", "CATGCA"]
    seqs = [s[:100] + f"GCTGACTCA{spacers[i % 4]}AGATAAGCG" + s[100 + 12 + 6 + len(spacers[i % 4]):]
            if i % 2 == 0 else s for i, s in enumerate(seqs)]
    start = um.add_gap(um.create_motif("TGACTCAAGATAAG", name="pair", nsites=20, pseudocount=1), gaploc=7,
                       mingap=3, maxgap=6)
    refined = um.refine_motifs(start, seqs)[0]
    assert refined.isgapped and refined.gaploc == [9]
    assert (refined.mingap, refined.maxgap) == ([3], [6])
    assert refined.consensus == "GCTGACTCAAGATAAGCG"
