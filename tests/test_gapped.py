"""Gapped scanning against the R package (forward strand only, which is all
the R package manages; the crate checks both strands against plain scoring)."""

import numpy as np

import universalmotif as um

from conftest import load_reference, motif_from_record, nums


def test_gapped_scans_match_r(tmp_path):
    motifs = [motif_from_record(r) for r in load_reference("gapped.json", "motifs")]
    seqs = {f"s{i + 1}": s for i, s in enumerate(load_reference("gapped.json", "sequences"))}
    um.write_fasta(seqs, tmp_path / "gapped.fa")
    packed = um.pack_sequences(seqs)
    for case in load_reference("gapped.json"):
        w = case["table"]
        for inp in (seqs, tmp_path / "gapped.fa", packed):
            got = um.scan_sequences(motifs, inp, pvalue=case["threshold"], rc=False, nthreads=2)
            got = got.sort(["motif_i", "sequence_i", "start", "match"])
            assert got["motif_i"].to_list() == [int(x) - 1 for x in nums(w["motif.i"])]
            assert got["sequence_i"].to_list() == [int(x) - 1 for x in nums(w["sequence.i"])]
            assert got["start"].to_list() == [int(x) for x in nums(w["start"])]
            assert got["end"].to_list() == [int(x) for x in nums(w["stop"])]
            assert got["match"].to_list() == list(np.atleast_1d(w["match"]))
            np.testing.assert_allclose(got["score"].to_numpy(), nums(w["score"]), rtol=1e-12)
            np.testing.assert_allclose(got["pvalue"].to_numpy(), nums(w["pvalue"]), rtol=1e-9)

    # Both strands, and ignoring the gaps.
    both = um.scan_sequences(motifs, seqs, pvalue=1e-3)
    minus = both.filter(both["strand"] == "-")
    assert minus.height > 0 and minus["match"].str.contains(r"\.").any()
    plain = um.scan_sequences(motifs, seqs, pvalue=1e-3, use_gaps=False)
    ungapped = um.scan_sequences(um.ungap(motifs, delete=True), seqs, pvalue=1e-3)
    assert plain.equals(ungapped)


def test_gap_correction():
    motifs = [motif_from_record(r) for r in load_reference("gapped.json", "motifs")]
    seqs = load_reference("gapped.json", "sequences")
    raw = um.scan_sequences(motifs, seqs, pvalue=1e-2)
    adj = um.scan_sequences(motifs, seqs, pvalue=1e-2, correct_gaps=True)
    combos = {0: 4, 1: 12, 2: 1}
    n = raw["motif_i"].replace_strict(combos, return_dtype=float)
    keep = raw.with_columns((raw["pvalue"] * n).clip(upper_bound=1.0).alias("pvalue")).filter(
        raw["pvalue"] * n <= 1e-2)
    cols = ["motif_i", "sequence_i", "start", "strand", "match", "pvalue"]
    assert adj.select(cols).equals(keep.select(cols))
    assert adj.height < raw.height
