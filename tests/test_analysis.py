"""P-values, scanning and deduplication against the R package."""

import math

import numpy as np
import polars as pl
import pytest

import universalmotif as um

from conftest import args, load_reference, motif_from_record, nums, reference_file


@pytest.mark.filterwarnings("ignore:scores above the highest")
def test_motif_pvalue_reference():
    for i, case in enumerate(load_reference("pvalue.json")):
        m = motif_from_record(case["input"])
        bkg = args(case["args"]).get("bkg")
        bkg = None if bkg is None else nums(bkg)
        ctx = f"case {i}: {m.name} {bkg}"
        got = um.motif_pvalue(m, score=nums(case["scores"]), bkg_probs=bkg)
        np.testing.assert_allclose(got, nums(case["pvalues"]), rtol=1e-12, err_msg=ctx)
        got = um.motif_pvalue(m, pvalue=nums(case["pvalue_in"]), bkg_probs=bkg)
        np.testing.assert_allclose(got, nums(case["score_out"]), rtol=1e-12, err_msg=ctx)


def _scan_motifs():
    return [motif_from_record(r) for r in load_reference("scan.json", "motifs")]


@pytest.mark.filterwarnings("ignore:.*cannot reach a P-value")
def test_scan_sequences_reference():
    motifs = _scan_motifs()
    for i, case in enumerate(load_reference("scan.json")):
        a = args(case["args"])
        ctx = f"scan case {i}: {case['file']} {a}"
        seqs = um.read_fasta(reference_file(case["file"]))
        if case["file"] == "extra.fa":
            seqs.pop("short")
        got = um.scan_sequences(
            motifs,
            seqs,
            pvalue=a.get("pvalue", 1e-4),
            rc=a.get("RC", True),
            no_overlaps=a.get("no.overlaps", False),
            no_overlaps_by=a.get("no.overlaps.by", "pvalue"),
            no_overlaps_by_strand=a.get("no.overlaps.by.strand", False),
            no_overlaps_by_motif=a.get("no.overlaps.by.motif", False),
        )
        w = case["hits"]
        assert got.height == len(w["start"]), ctx
        for col in ("motif", "sequence", "strand", "match"):
            assert got[col].to_list() == w[col], f"{ctx}: {col}"
        # R counts positions from 1, with the end included.
        assert got["start"].to_list() == [int(x) - 1 for x in w["start"]], ctx
        assert got["end"].to_list() == [int(x) for x in w["end"]], ctx
        # The R package's indices are 1-based; these are 0-based.
        assert got["motif_i"].to_list() == [int(x) - 1 for x in w["motif_i"]], ctx
        assert got["sequence_i"].to_list() == [int(x) - 1 for x in w["sequence_i"]], ctx
        for col in ("score", "score_pct", "pvalue"):
            np.testing.assert_allclose(got[col].to_numpy(), nums(w[col]), rtol=1e-12,
                                       err_msg=f"{ctx}: {col}")


def test_dedup_hits_matches_scan_option():
    motifs = _scan_motifs()
    seqs = um.read_fasta(reference_file("promoters.fa"))
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-3)
    for by in ("pvalue", "score"):
        a = um.dedup_hits(hits, by=by)
        b = um.scan_sequences(motifs, seqs, pvalue=1e-3, no_overlaps=True, no_overlaps_by=by)
        assert a.equals(b)
    both = um.dedup_hits(hits, ignore_strand=True, ignore_motif=True)
    assert both.height < hits.height
    starts = both.group_by("sequence").agg("start", "end")
    for row in starts.iter_rows(named=True):
        spans = sorted(zip(row["start"], row["end"]))
        assert all(e <= s for (_, e), (s, _) in zip(spans, spans[1:]))
    with pytest.raises(ValueError, match="missing"):
        um.dedup_hits(hits.drop("start"))


def test_motif_pvalue_shapes():
    motifs = _scan_motifs()[:3]
    assert isinstance(um.motif_pvalue(motifs[0], score=5.0), float)
    assert um.motif_pvalue(motifs[0], score=[4, 5, 6]).shape == (3,)
    paired = um.motif_pvalue(motifs, pvalue=[1e-3, 1e-4, 1e-5])
    assert paired.shape == (3,)
    assert paired[1] == um.motif_pvalue(motifs[1], pvalue=1e-4)
    per = um.motif_pvalue(motifs, pvalue=[[1e-3, 1e-4], [0.5], [0.1, 0.2, 0.3]])
    assert [len(x) for x in per] == [2, 1, 3]
    assert um.motif_pvalue(motifs, pvalue=1e-300).tolist() == [math.inf] * 3
    with pytest.raises(ValueError, match="exactly one"):
        um.motif_pvalue(motifs, score=1, pvalue=0.1)
    with pytest.raises(ValueError):
        um.motif_pvalue(motifs, pvalue=[0.1, 0.2])
    bkg = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    assert um.motif_pvalue(motifs[0], score=5, bkg_probs=bkg) != um.motif_pvalue(motifs[0], score=5)


def test_scan_inputs():
    m = um.create_motif("TTGACT")
    as_str = um.scan_sequences(m, "aattgactaa", pvalue=0.01)
    assert as_str["sequence"].to_list()[0] == "seq1"
    assert as_str.filter(as_str["strand"] == "+")["match"].to_list() == ["TTGACT"]
    as_dict = um.scan_sequences(m, {"short": "ACG", "s2": "AATTGACTAA"}, pvalue=0.01)
    assert set(as_dict["sequence"].to_list()) == {"s2"}
    assert um.scan_sequences(m, ["GGGG"], pvalue=0.01).height == 0
    # Protein motifs scan the forward strand only.
    aa = um.scan_sequences(um.create_motif("ACDEF", alphabet="AA"), "xxacdefgg", pvalue=0.01)
    assert aa["match"].to_list() == ["ACDEF"]
    assert aa["strand"].to_list() == ["+"]


def test_fasta_round_trip(tmp_path):
    seqs = um.read_fasta(reference_file("promoters.fa"))
    um.write_fasta(seqs, tmp_path / "x.fa", width=60)
    assert um.read_fasta(tmp_path / "x.fa") == seqs
    um.write_fasta(["ACGT", "GGCC"], tmp_path / "y.fa")
    assert um.read_fasta(tmp_path / "y.fa") == {"seq1": "ACGT", "seq2": "GGCC"}


def test_motif_pvalue_broadcasting():
    motifs = um.example_motifs()[:3]
    with pytest.raises(ValueError, match="no motifs"):
        um.motif_pvalue([], score=1)
    each = [um.motif_pvalue(m, score=[5, 8]) for m in motifs]
    shared = um.motif_pvalue(motifs, score=[[5, 8]])
    assert len(shared) == 3
    for got, want in zip(shared, each):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="one sequence of values for all motifs or one per motif"):
        um.motif_pvalue(motifs, score=[[5], [8]])
    np.testing.assert_array_equal(um.motif_pvalue(motifs, score=[5]),
                                  [um.motif_pvalue(m, score=5) for m in motifs])
    np.testing.assert_array_equal(um.motif_pvalue(motifs, score=5), um.motif_pvalue(motifs, score=[5]))


def _grid_scores(motif, words):
    """Scores as documented: PWM entries truncated to three decimals and
    summed exactly."""
    pwm = np.trunc(um.convert_type(motif, "PWM").matrix * 1000).astype(int)
    rows = {letter: i for i, letter in enumerate(motif.letters)}
    return np.array([sum(pwm[rows[c], i] for i, c in enumerate(w)) for w in words])


@pytest.mark.parametrize("bkg", [None, [0.4, 0.1, 0.1, 0.4]])
def test_motif_pvalue_matches_enumeration(bkg):
    import itertools

    m = um.create_motif(["TGACA", "TGAGA", "TCACA", "AGACT"], pseudocount=1, type="PCM", bkg=bkg)
    words = ["".join(w) for w in itertools.product("ACGT", repeat=5)]
    grid = _grid_scores(m, words)
    probs = um.prob_match(m, words)
    assert probs.sum() == pytest.approx(1)
    for score in (-5, 0, 2.5, 4, grid.max() / 1000):
        want = probs[grid >= round(score * 1000)].sum()
        assert um.motif_pvalue(m, score=score) == pytest.approx(want, rel=1e-9)
    if bkg is None:
        assert sorted(um.get_scores(m) * 1000, reverse=True) == pytest.approx(sorted(grid, reverse=True))
    assert um.motif_range(m) == pytest.approx((grid.min() / 1000, grid.max() / 1000))


def test_motif_pvalue_and_score_are_inverse():
    m = um.example_motifs()[5]
    for p in (0.05, 1e-2, 1e-3, 1e-4, 1e-5):
        s = um.motif_pvalue(m, pvalue=p)
        # The threshold is the lowest score on the 0.001 grid reaching p.
        assert um.motif_pvalue(m, score=s) <= p
        assert um.motif_pvalue(m, score=s - 0.001) > p
    # P-values fall as scores rise.
    scores = np.linspace(*um.motif_range(m), 25)
    assert np.all(np.diff(um.motif_pvalue(m, score=scores)) <= 0)
    assert um.motif_pvalue(m, score=um.motif_range(m)[0]) == pytest.approx(1)


def test_scan_hits_are_consistent():
    motifs = um.example_motifs()[3:6]
    seqs = dict(list(um.example_sequences().items())[:20])
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-3)
    assert hits.height > 0
    comp = str.maketrans("ACGT", "TGCA")
    for r in hits.iter_rows(named=True):
        site = seqs[r["sequence"]][r["start"]:r["end"]]
        assert r["match"] == (site if r["strand"] == "+" else site.translate(comp)[::-1])
        assert r["end"] - r["start"] == motifs[r["motif_i"]].width
        assert r["score"] == _grid_scores(motifs[r["motif_i"]], [r["match"]])[0] / 1000
        assert r["pvalue"] <= 1e-3
        assert r["pvalue"] == pytest.approx(um.motif_pvalue(motifs[r["motif_i"]], score=r["score"]))
    # RNA motifs find the same sites in RNA.
    rna = um.scan_sequences([um.switch_alph(m) for m in motifs],
                            {k: v.replace("T", "U") for k, v in seqs.items()}, pvalue=1e-3)
    assert rna.drop("match").equals(hits.drop("match"))
    assert rna["match"].to_list() == [s.replace("T", "U") for s in hits["match"]]
    # Windows holding an N are skipped.
    near_n = um.scan_sequences(um.create_motif("TTGACT"), "TTGACTNNNTTGACT", pvalue=0.01, rc=False)
    assert near_n.select("start", "end").rows() == [(0, 6), (9, 15)]


@pytest.mark.xfail(strict=True, reason="score_match sums truncated entries in floating point, "
                   "then truncates the total, and can come out 0.001 below the scan score")
def test_score_match_agrees_with_scanning():
    gbox = um.example_motifs()[3]
    hit = um.scan_sequences(gbox, "TAACGTGG", pvalue=0.01, rc=False)
    assert hit["score"].to_list() == [7.844]
    assert um.score_match(gbox, "TAACGTGG") == 7.844


def test_scan_empty_genome(tmp_path):
    path = tmp_path / "empty.fa"
    path.write_text("")
    hits = um.scan_sequences(um.example_motifs()[:2], um.Genome(path))
    assert hits.height == 0
    assert hits.columns == ["motif", "motif_i", "sequence", "sequence_i", "start", "end", "strand",
                            "score", "score_pct", "match", "pvalue"]
    assert hits.schema["start"] == pl.Int64


def test_dedup_hits_checks():
    base = {"sequence": ["a", "a"], "motif": ["m", "m"], "strand": ["+", "+"], "start": [0, 2],
            "end": [5, 6], "pvalue": [0.1, 0.01], "score": [1.0, 2.0]}
    hits = pl.DataFrame(base)
    assert um.dedup_hits(hits)["start"].to_list() == [2]
    assert um.dedup_hits(hits, by="score")["start"].to_list() == [2]
    assert um.dedup_hits(hits, by="score", reverse=True)["start"].to_list() == [0]
    assert um.dedup_hits(hits, reverse=True)["start"].to_list() == [0]
    # Touching hits do not overlap, as ends are left out.
    touching = pl.DataFrame({**base, "start": [0, 5], "end": [5, 10]})
    assert um.dedup_hits(touching).height == 2
    empty = hits.clear()
    assert um.dedup_hits(empty).equals(empty)
    with pytest.raises(ValueError, match="nulls"):
        um.dedup_hits(pl.DataFrame({**base, "start": [0, None]}))
    with pytest.raises(ValueError, match="start <= end"):
        um.dedup_hits(pl.DataFrame({**base, "start": [7, 2]}))
    for bad in (None, float("nan")):
        with pytest.raises(ValueError, match="'pvalue' cannot contain missing values"):
            um.dedup_hits(pl.DataFrame({**base, "pvalue": [bad, 0.1]}))
