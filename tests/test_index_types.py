"""Motif and sequence indices are UInt32 columns in every table, so that
tables can be joined on them."""

import warnings

import polars as pl
import universalmotif as um


def _index_types(df):
    return {c: t for c, t in df.schema.items() if c.endswith("_i")}


def test_index_columns_are_uint32():
    ms = um.example_motifs()[:4]
    seqs = um.create_sequences(seqnum=20, seqlen=300, seed=1)
    named = {f"s{i}": s for i, s in enumerate(seqs)}
    hits = um.scan_sequences(ms, seqs, pvalue=1e-2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = {
            "scan": hits,
            "scan of named sequences": um.scan_sequences(ms, named, pvalue=1e-2),
            "compare": um.compare_motifs(ms, compare_to=0, qvalue=1),
            "merge clusters": um.merge_similar(ms, return_clusters=True),
            "enrich": um.enrich_motifs(ms, seqs, qvalue=1),
            "peaks": um.motif_peaks(hits, seq_length=300, qvalue=1),
            "implants": um.implant_motifs(ms[0], seqs, seed=1, return_indices=True)[1],
            "matched background": um.match_bkg(seqs[:5], seqs, return_indices=True),
        }
    for name, df in tables.items():
        types = _index_types(df)
        assert types, name
        assert all(t == pl.UInt32 for t in types.values()), (name, types)
    # Tables join on the indices without casting.
    joined = hits.join(tables["enrich"].select("motif_i", "pvalue"), on="motif_i", how="left")
    assert joined.height == hits.height
