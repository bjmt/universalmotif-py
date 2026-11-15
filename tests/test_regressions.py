"""Bugs found in review, each checked so that it stays fixed."""

import numpy as np
import polars as pl
import pytest
import universalmotif as um
from universalmotif._cli import main


def test_bytes_are_read_as_letters_and_other_objects_refused():
    m = um.create_motif("GATAA")
    hits = um.scan_sequences(m, [b"TTGATAAGTT"], pvalue=0.01)
    assert hits["start"].to_list()[0] == 2
    assert um.shuffle_sequences([b"AACCGGTTAC"], seed=1)[0].count("A") == 3
    with pytest.raises(TypeError, match="NoneType"):
        um.scan_sequences(m, ["ACGT", None], pvalue=0.01)
    with pytest.raises(TypeError, match="ndarray"):
        um.shuffle_sequences([np.zeros((4, 5))], seed=1)


def test_string_arrays_are_sequences_not_one_hot():
    out = um.shuffle_sequences(np.array(["AACCGGTTAC", "ACGTACGTAC"]), seed=1)
    assert len(out) == 2 and sorted(out[0]) == sorted("AACCGGTTAC")


def test_resizing_keeps_a_missing_summit_missing():
    df = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [141], "peak": [-1]})
    assert um.resize_ranges(df, 100)["peak"].to_list() == [None]
    assert um.resize_ranges(df, 100, anchor="centre").rows() == um.resize_ranges(df, 100, anchor="center").rows()


def test_numpy_integers_pick_queries():
    ms = um.example_motifs()[:3]
    a = um.compare_motifs(ms, compare_to=np.int64(0), qvalue=1)
    b = um.compare_motifs(ms, compare_to=0, qvalue=1)
    assert a.equals(b)


def test_one_hot_takes_letters_beyond_latin1():
    x = um.one_hot(["ACGT–"])
    assert x.shape == (1, 4, 5) and x[0, :, 4].sum() == 0


def test_bkg_table_describes_the_sequences_written(tmp_path):
    targets = tmp_path / "t.fa"
    universe = tmp_path / "u.fa"
    um.write_fasta(um.example_sequences("targets"), targets)
    um.write_fasta(um.example_sequences("controls"), universe)
    out, table = tmp_path / "out.fa", tmp_path / "t.tsv"
    assert main(["bkg", "-s", str(targets), "-u", str(universe), "--table", str(table), "-o", str(out)]) == 0
    written = list(um.read_fasta(out))
    picked = pl.read_csv(table, separator="\t")["universe_name"].to_list()
    assert [w.split(".")[0] for w in written] == picked


def test_compare_against_a_database_corrects_within_its_rows(tmp_path):
    ms = um.example_motifs()
    um.write_meme(ms[:2], tmp_path / "q.meme")
    um.write_meme(ms[2:], tmp_path / "db.meme")
    out = tmp_path / "res.tsv"
    assert main(["compare", str(tmp_path / "q.meme"), "--db", str(tmp_path / "db.meme"), "--all",
                 "-o", str(out)]) == 0
    res = pl.read_csv(out, separator="\t")
    for _, rows in res.group_by("query_i"):
        p = np.sort(rows["pvalue"].to_numpy())
        m = len(p)
        q = np.minimum(np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1], 1)
        assert np.allclose(np.sort(rows["qvalue"].to_numpy()), q)
        assert m == len(ms) - 2


def test_old_argument_names_still_work_with_a_warning():
    gbox = um.example_motifs()[3]
    seqs = um.example_sequences()
    new = um.scan_sequences(gbox, seqs, no_overlaps=True, no_overlaps_across_strands=False)
    with pytest.warns(FutureWarning, match="no_overlaps_across_strands"):
        old = um.scan_sequences(gbox, seqs, no_overlaps=True, no_overlaps_by_strand=False)
    assert old.equals(new)
    hits = um.scan_sequences(gbox, seqs)
    with pytest.warns(FutureWarning, match="across_strands"):
        assert um.dedup_hits(hits, ignore_strand=False).equals(um.dedup_hits(hits, across_strands=False))
    with pytest.warns(FutureWarning, match="bkg"):
        um.motif_pvalue(gbox, score=5, bkg_probs=[0.25] * 4)
    ms = um.example_motifs()[:5]
    with pytest.warns(FutureWarning, match="linkage"):
        old_tree = um.motif_tree(ms, method="average")
    assert old_tree.order == um.motif_tree(ms, linkage="average").order
