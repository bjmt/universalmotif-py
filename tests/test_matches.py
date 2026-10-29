"""Sparse motif match matrices, against the hits of scan_sequences."""

import numpy as np
import polars as pl
import pytest

import universalmotif as um

pytest.importorskip("scipy")


def _expected(hits, n_seq, n_motif, value):
    out = np.zeros((n_seq, n_motif))
    for s, m, score in hits.select("sequence_i", "motif_i", "score").iter_rows():
        if value == "matches":
            out[s, m] = 1
        elif value == "counts":
            out[s, m] += 1
        else:
            out[s, m] = max(out[s, m], score) if out[s, m] else score
    return out


@pytest.mark.parametrize("value", ["matches", "counts", "scores"])
def test_agrees_with_scan_sequences(value):
    motifs = um.example_motifs()
    seqs = um.example_sequences()
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-4)
    got = um.match_motifs(motifs, seqs, pvalue=1e-4, value=value, nthreads=2)
    assert got.matrix.shape == (100, 12) and got.value == value
    assert got.sequences == list(seqs) and got.motifs == [m.name for m in motifs]
    np.testing.assert_allclose(got.matrix.toarray(), _expected(hits, 100, 12, value))
    assert got.matrix.dtype == {"matches": np.uint8, "counts": np.uint32,
                                "scores": np.float64}[value]


def test_inputs_and_outputs():
    motifs = um.example_motifs()[3:6]
    seqs = um.example_sequences()
    plain = um.match_motifs(motifs, seqs)
    packed = um.match_motifs(motifs, um.pack_sequences(seqs))
    assert (packed.matrix != plain.matrix).nnz == 0 and packed.sequences == plain.sequences
    listed = um.match_motifs(motifs, list(seqs.values()))
    assert listed.sequences[:2] == ["seq1", "seq2"]
    long = plain.to_polars()
    assert long.columns == ["sequence", "motif", "value"] and long.height == plain.matrix.nnz
    assert long.filter(pl.col("motif") == "G-box").height == plain.matrix[:, 0].sum()
    dedup = um.match_motifs(motifs, seqs, value="counts", no_overlaps=True)
    assert dedup.matrix.sum() <= um.match_motifs(motifs, seqs, value="counts").matrix.sum()
    with pytest.raises(ValueError, match="value"):
        um.match_motifs(motifs, seqs, value="hits")
    with pytest.warns(UserWarning, match="cannot reach"):
        um.match_motifs(motifs, seqs, pvalue=1e-7)


def test_anndata():
    ad = pytest.importorskip("anndata")
    motifs = um.example_motifs()
    seqs = um.example_sequences()
    adata = um.match_motifs(motifs, seqs).to_anndata()
    assert adata.shape == (100, 12) and list(adata.var_names) == [m.name for m in motifs]

    # A cells by peaks object, as pychromVAR uses.
    cells = ad.AnnData(np.zeros((3, 100), dtype=np.float32))
    cells.var_names = list(seqs)
    cells.uns["peak_seq"] = list(seqs.values())
    um.add_motif_matches(cells, motifs)
    assert cells.varm["motif_match"].shape == (100, 12)
    assert cells.varm["motif_match"].dtype == np.uint8
    assert cells.uns["motif_name"] == [m.name for m in motifs]
    # Sequences by name are put in the order of the peaks.
    shuffled = dict(reversed(list(seqs.items())))
    um.add_motif_matches(cells, motifs, shuffled, key="by_name", dense=False)
    assert np.array_equal(cells.varm["by_name"].toarray(), cells.varm["motif_match"])
    with pytest.raises(ValueError, match="no sequence"):
        um.add_motif_matches(cells, motifs, {"x": "ACGT"})
    with pytest.raises(ValueError, match="3 sequences for 100 peaks"):
        um.add_motif_matches(cells, motifs, ["ACGT"] * 3)
    del cells.uns["peak_seq"]
    with pytest.raises(ValueError, match="peak_seq"):
        um.add_motif_matches(cells, motifs)


def test_checks_and_missing_extras(monkeypatch):
    import sys

    motifs = um.example_motifs()[3:6]
    seqs = um.example_sequences()
    with pytest.raises(ValueError, match="no motifs given"):
        um.match_motifs([], seqs)
    matches = um.match_motifs(motifs, seqs)
    assert repr(matches) == f"MotifMatches(100 sequences x 3 motifs, {matches.matrix.nnz} matches)"
    monkeypatch.setitem(sys.modules, "anndata", None)
    with pytest.raises(ImportError, match=r"universalmotif\[anndata\]"):
        matches.to_anndata()
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    with pytest.raises(ImportError, match=r"universalmotif\[scipy\]"):
        um.match_motifs(motifs, seqs)


def test_counts_and_scores_by_hand():
    # A palindrome, so each site is hit on both strands.
    site = um.create_motif("CACGTG", nsites=20, pseudocount=1)
    seqs = {"none": "CCCCCCCCCCCC", "one": "CCCACGTGCCCC", "two": "CACGTGACACGTG"}
    counts = um.match_motifs(site, seqs, value="counts", pvalue=5e-4)
    assert counts.matrix.toarray()[:, 0].tolist() == [0, 2, 4]
    once = um.match_motifs(site, seqs, value="counts", pvalue=5e-4, no_overlaps=True)
    # The two hits of a site overlap, so each site counts once.
    assert once.matrix.toarray()[:, 0].tolist() == [0, 1, 2]
    best = um.match_motifs(site, seqs, value="scores", pvalue=5e-4)
    top = um.motif_range(site)[1]
    assert best.matrix.toarray()[:, 0].tolist() == [0, top, top]
    assert best.to_polars().rows() == [("one", "motif", top), ("two", "motif", top)]
    # RNA sequences and motifs match alike.
    rna = um.match_motifs(um.switch_alph(site), {k: v.replace("T", "U") for k, v in seqs.items()},
                          value="counts", pvalue=5e-4)
    assert (rna.matrix != counts.matrix).nnz == 0


@pytest.mark.parametrize("no_overlaps", [False, True])
def test_presence_with_long_and_gapped_sequences(no_overlaps):
    # Presence stops scanning a sequence at a motif's first hit, so long
    # sequences are scanned in pieces that each stop on their own.
    motifs = [um.create_motif(6 + i % 9, bkg=[0.25] * 4, seed=100 + i) for i in range(10)]
    motifs.append(um.add_gap(motifs[3], 4, 0, 4))
    motifs.append(um.example_motifs()[3])
    seqs = um.create_sequences(seqnum=300, seqlen=400, seed=7)
    seqs += ["AC" * 70_000 + s for s in um.create_sequences(seqnum=2, seqlen=100_000, seed=8)]
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-3, no_overlaps=no_overlaps,
                             no_overlaps_by_strand=True, nthreads=2)
    want = _expected(hits, len(seqs), len(motifs), "matches")
    for nthreads in (1, 2):
        got = um.match_motifs(motifs, seqs, pvalue=1e-3, no_overlaps=no_overlaps, nthreads=nthreads)
        np.testing.assert_array_equal(got.matrix.toarray(), want)
    first = um.scan_sequences(motifs, seqs, pvalue=1e-3, first_hit=True, nthreads=2)
    assert first.height == int(want.sum())
