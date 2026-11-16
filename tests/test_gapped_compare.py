import pytest
import universalmotif as um


def dr(lo, hi, name="DR"):
    """A direct repeat of AGGTCA with a spacer of lo to hi letters."""
    return um.add_gap(um.create_motif("AGGTCAAGGTCA", nsites=20, name=name), gaploc=6, mingap=lo, maxgap=hi)


def test_comparisons_find_the_gap_size_that_fits():
    fixed = um.create_motif("AGGTCANNNNAGGTCA", name="DR4")
    res = um.compare_motifs([dr(1, 6), fixed], compare_to=0, qvalue=1)
    row = res.filter(target="DR4").row(0, named=True)
    assert row["subject_gaps"] == [4] and row["target_gaps"] == []
    assert row["overlap"] == 16 and row["score"] > 0.99
    # Run together, only one half lines up.
    old = um.compare_motifs([dr(1, 6), fixed], compare_to=0, qvalue=1, use_gaps=False)
    assert "subject_gaps" not in old.columns
    assert old.filter(target="DR4")["overlap"][0] == 6
    assert old.filter(target="DR4")["pvalue"][0] > 100 * row["pvalue"]
    # Without gapped motifs, the columns stay as they were.
    plain = um.compare_motifs(um.example_motifs()[:3], compare_to=0, qvalue=1)
    assert "subject_gaps" not in plain.columns


def test_matrices_and_trees_use_the_gaps():
    fixed = um.create_motif("AGGTCANNNNAGGTCA", name="DR4")
    ms = [dr(1, 6), fixed, um.create_motif("CACGTGAC", name="E-box")]
    with_gaps = um.compare_motifs(ms)
    without = um.compare_motifs(ms, use_gaps=False)
    assert with_gaps[0, 1] > 0.99 and with_gaps[0, 1] >= without[0, 1]


def test_merging_keeps_the_gaps():
    merged = um.merge_motifs([dr(1, 3, "a"), dr(3, 5, "b")])
    assert merged.isgapped
    assert (merged.gaploc, merged.mingap, merged.maxgap) == ([6], [1], [5])
    assert merged.width == 12
    assert not um.merge_motifs([dr(1, 3, "a"), dr(3, 5, "b")], use_gaps=False).isgapped


def test_mixing_gapped_and_ungapped_motifs_warns():
    fixed = um.create_motif("AGGTCANNNNAGGTCA", name="DR4")
    with pytest.warns(UserWarning, match="filled with blank columns"):
        merged = um.merge_motifs([dr(2, 6), fixed])
    assert not merged.isgapped and merged.width == 16
    with pytest.warns(UserWarning, match="cluster 0"):
        um.merge_similar([dr(2, 6), um.create_motif("AGGTCANNNNAGGTCA", name="DR4")])
