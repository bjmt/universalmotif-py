"""P-values, scanning and deduplication against the R package."""

import math

import numpy as np
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
