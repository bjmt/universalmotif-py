"""Scanning, comparison, merging, enrichment and logos for amino acid and
custom-alphabet motifs, checked against the R package's full
``scan_sequences()`` and ``compare_motifs()``."""

import numpy as np
import pytest

import universalmotif as um

from conftest import load_reference, motif_from_record, nums


def _ref():
    keys = ("aa_motifs", "aa_sequences", "aa_scan", "aa_compare",
            "custom_motifs", "custom_sequences", "custom_scan", "custom_compare")
    return {k: load_reference("alphabets.json", k) for k in keys}


def _check_scans(motifs, seqs, cases, label):
    for case in cases:
        threshold = case["threshold"]
        ms = motifs[:3] if label == "aa" and threshold > 0.005 else motifs
        ctx = f"{label} scan at {threshold}"
        got = um.scan_sequences(ms, seqs, pvalue=threshold)
        w = case["table"]
        assert got.height == len(w["start"]), ctx
        assert got["motif_i"].to_list() == [int(x) - 1 for x in w["motif.i"]], ctx
        assert got["sequence_i"].to_list() == [int(x) - 1 for x in w["sequence.i"]], ctx
        assert got["start"].to_list() == [int(x) for x in w["start"]], ctx
        assert got["end"].to_list() == [int(x) for x in w["stop"]], ctx
        assert set(got["strand"].to_list()) <= {"+"}, ctx
        assert got["match"].to_list() == w["match"], ctx
        np.testing.assert_allclose(got["score"].to_numpy(), nums(w["score"]), rtol=1e-12, err_msg=ctx)
        np.testing.assert_allclose(got["pvalue"].to_numpy(), nums(w["pvalue"]), rtol=1e-9, err_msg=ctx)


def test_scans_match_r():
    r = _ref()
    aa = [motif_from_record(m) for m in r["aa_motifs"]]
    _check_scans(aa, r["aa_sequences"], r["aa_scan"], "aa")
    custom = [motif_from_record(m) for m in r["custom_motifs"]]
    _check_scans(custom, r["custom_sequences"], r["custom_scan"], "custom")


def test_comparisons_match_r():
    r = _ref()
    aa = [motif_from_record(m) for m in r["aa_motifs"]]
    custom = [motif_from_record(m) for m in r["custom_motifs"]]
    cases = [(aa, [0, 1, 2, 3, 4], r["aa_compare"][0]), (aa, [0, 2, 3, 4], r["aa_compare"][1]),
             (custom, [0, 1, 2], r["custom_compare"][0])]
    for motifs, subset, case in cases:
        ms = [motifs[i] for i in subset]
        mo = int(case["min_overlap"])
        scores = um.compare_motifs(ms, min_overlap=mo)
        # The R values are sums over the aligned columns.
        sums = np.array([nums(row) for row in case["sums"]])
        long = um.compare_motifs(ms, compare_to=list(range(len(ms))), min_overlap=mo, qvalue=1)
        for row in long.iter_rows(named=True):
            q, t = row["subject_i"], row["target_i"]
            assert row["strand"] == "+"
            width = len(row["subject_consensus"])
            assert row["score"] * width == pytest.approx(sums[q, t], rel=1e-12, abs=1e-12)
            assert scores[q, t] == pytest.approx(row["score"], rel=1e-12)


def test_protein_consensus_and_merge():
    r = _ref()
    aa = [motif_from_record(m) for m in r["aa_motifs"]]
    hits = um.compare_motifs(aa, compare_to=[0], qvalue=1)
    short = hits.filter(hits["target_i"] == 4).row(0, named=True)
    assert (short["subject_consensus"], short["target_consensus"]) == ("MKVLAW", "MXVLAW")
    merged = um.merge_motifs([aa[0], aa[4]], min_overlap=4)
    assert merged.alphabet == "AA"
    assert merged.width == aa[0].width
    np.testing.assert_allclose(merged.matrix.sum(axis=0), 1.0)
    tree = um.motif_tree(aa)
    assert sorted(tree.order) == list(range(len(aa)))


def test_protein_enrichment():
    r = _ref()
    aa = [motif_from_record(m) for m in r["aa_motifs"]]
    seqs = r["aa_sequences"]
    out = um.enrich_motifs(aa[:2], seqs, seed=1, pvalue=1e-3, qvalue=1)
    assert out.height >= 1
    assert out["motif"].to_list()[0] == aa[0].name


def test_view_motifs_any_alphabet():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    r = _ref()
    aa = [motif_from_record(m) for m in r["aa_motifs"]]
    fig = um.view_motifs(aa[:2])
    assert fig is not None
    custom = [motif_from_record(m) for m in r["custom_motifs"]]
    raw = um.view_motifs(iter(custom), return_raw=True)
    assert all(m.shape[0] == 5 for m in raw.values())
    um.motif_tree(custom).plot(tip_logos=True)
    matplotlib.pyplot.close("all")
