"""Comparison, merging and trees against the R package's lite functions."""

import numpy as np
import pytest

import universalmotif as um

from conftest import args, assert_motif_matches, load_reference, motif_from_record, nums


def _set(name):
    return [motif_from_record(r) for r in load_reference("compare.json", name)]


def _opts(a):
    out = {}
    if "RC" in a:
        out["rc"] = a["RC"]
    if "min.overlap" in a:
        out["min_overlap"] = int(a["min.overlap"])
    if "null" in a:
        out["null"] = a["null"]
    if "bkg" in a:
        out["bkg"] = nums(a["bkg"])
    return out


def _beyond_null(got, want, a):
    """The R package gives a P-value of 0 to scores beyond what the parametric
    null can reach, where the port gives the null's highest score's; such
    values must be small and positive and are then set to R's 0."""
    got = np.array(got, dtype=float)
    if a.get("null") == "parametric":
        fix = (want == 0) & (got != 0)
        assert np.all((got[fix] > 0) & (got[fix] < 1e-6))
        got[fix] = 0.0
    return got


def test_compare_motifs_reference():
    for i, case in enumerate(load_reference("compare.json", "compare")):
        a = args(case["args"])
        ctx = f"compare case {i}: {case['set']} {a}"
        motifs = _set(case["set"])
        if "matrix" in case:
            got = um.compare_motifs(motifs, matrix_out=a.get("matrix.out", "score"), **_opts(a))
            want = np.array([nums(r) for r in case["matrix"]])
            got = _beyond_null(got, want, a)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300, err_msg=ctx)
        else:
            to = [int(x) - 1 for x in nums(a["compare.to"])]
            got = um.compare_motifs(motifs, compare_to=to, qvalue=a.get("qvalue", 0.1), **_opts(a))
            w = case["long"]
            assert got["subject_i"].to_list() == [int(x) - 1 for x in nums(w["subject_i"])], ctx
            assert got["target_i"].to_list() == [int(x) - 1 for x in nums(w["target_i"])], ctx
            assert got["offset"].to_list() == [int(x) for x in nums(w["offset"])], ctx
            assert got["strand"].to_list() == list(np.atleast_1d(w["strand"])), ctx
            # The R package writes T in RNA consensus strings; the port
            # writes U.
            fix = (lambda s: s.replace("T", "U")) if case["set"] == "rna" else str
            # The R package builds these with its own 75% rule rather than
            # the motif consensus rule the port uses. A
            # single base under the 75% rule is also one under the motif rule,
            # so those letters must agree.
            for col in ("subject_consensus", "target_consensus"):
                want = [fix(s) for s in np.atleast_1d(w[col])]
                assert len(got[col]) == len(want), ctx
                for g, r in zip(got[col].to_list(), want):
                    assert len(g) == len(r), (ctx, g, r)
                    assert all(a == b for a, b in zip(g, r) if b in "ACGTU"), (ctx, g, r)
            for col in ("score", "pvalue", "qvalue"):
                want = np.array(nums(w[col]), dtype=float)
                np.testing.assert_allclose(_beyond_null(got[col].to_numpy(), want, a), want,
                                           rtol=1e-12, err_msg=f"{ctx}: {col}")


@pytest.mark.filterwarnings("ignore:.*could not overlap the anchor")
def test_merge_motifs_reference():
    for i, case in enumerate(load_reference("compare.json", "merge")):
        a = args(case["args"])
        motifs = [_set(case["set"])[int(k) - 1] for k in nums(case["idx"])]
        got = um.merge_motifs(
            motifs,
            min_overlap=int(a.get("min.overlap", 5)),
            rc=a.get("RC", True),
            new_name=a.get("new.name"),
            weighted=a.get("weighted", False),
        )
        assert_motif_matches(got, case["motif"], f"merge case {i}")


def test_merge_similar_reference():
    motifs = _set("dna")
    for i, case in enumerate(load_reference("compare.json", "similar")):
        a = args(case["args"])
        kw = dict(qvalue=a.get("qvalue", 0.01), weighted=a.get("weighted", False), **_opts(a))
        clusters = um.merge_similar(motifs, return_clusters=True, **kw)
        assert clusters["cluster"].to_list() == [int(x) - 1 for x in nums(case["clusters"])]
        assert clusters["motif_i"].to_list() == list(range(len(motifs)))
        merged = um.merge_similar(motifs, **kw)
        assert len(merged) == len(case["motifs"])
        for k, (g, w) in enumerate(zip(merged, case["motifs"])):
            assert_motif_matches(g, w, f"similar case {i}, motif {k}")


def test_trees_reference():
    motifs = _set("dna")
    for i, case in enumerate(load_reference("compare.json", "tree")):
        ctx = f"tree case {i}: {case['source']} {case['method']}"
        dist = np.array([nums(r) for r in case["dist"]])
        labels = list(case["labels"])
        tree = um.motif_tree(dist, method=case["method"], labels=labels)
        assert tree.merge.tolist() == [[int(v) for v in nums(r)] for r in case["merge"]], ctx
        assert tree.order == [int(x) - 1 for x in nums(case["order"])], ctx
        np.testing.assert_allclose(tree.height, nums(case["height"]), rtol=1e-12, err_msg=ctx)
        assert tree.newick == case["newick"], ctx
        if case["source"] == "motifs":
            a = args(case["args"])
            own = um.motif_tree(motifs, min_overlap=int(a["min.overlap"]), rc=a["RC"],
                                method=case["method"])
            assert own.labels == labels, ctx
            assert own.merge.tolist() == tree.merge.tolist(), ctx
            assert own.order == tree.order, ctx
            np.testing.assert_allclose(own.height, tree.height, rtol=1e-12, err_msg=ctx)


def test_tree_outputs(tmp_path):
    motifs = _set("dna")
    tree = um.motif_tree(motifs, labels="consensus")
    assert len(tree) == len(motifs)
    assert tree.labels[0] == motifs[0].consensus
    assert tree.linkage.shape == (len(motifs) - 1, 4)
    assert tree.linkage[-1, 3] == len(motifs)
    np.testing.assert_allclose(tree.linkage[:, 2], tree.height)
    tree.write_newick(tmp_path / "t.nwk")
    assert (tmp_path / "t.nwk").read_text().strip() == tree.newick
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    assert hierarchy.is_valid_linkage(tree.linkage)
    assert hierarchy.dendrogram(tree.linkage, no_plot=True)["leaves"] == tree.order


def test_compare_errors():
    motifs = _set("dna")
    with pytest.raises(ValueError, match="no motif named"):
        um.compare_motifs(motifs, compare_to="nope")
    with pytest.raises(ValueError, match="parametric null"):
        um.compare_motifs([um.create_motif("ACDEF"), um.create_motif("ACDEG")], compare_to=0, null="parametric")
    with pytest.raises(ValueError, match="same alphabet"):
        um.compare_motifs([um.create_motif("ACDEF"), um.create_motif("ACGT")])
    with pytest.warns(UserWarning, match="qvalue"):
        um.compare_motifs(motifs[:3], qvalue=0.5)


def test_complete_scores():
    make = lambda s: um.create_motif(s, nsites=20, pseudocount=1)  # noqa: E731
    motifs = [make("TTGACGTCAGCCATG"), make("TTGACGTCAGCCATG"), make("TGACGTCA")]
    motifs += [make(s) for s in ("CACGTGAA", "GATAAGCT", "CCAATCAG", "AAACCCTA", "TATAAAAG")]
    names = [f"m{i}" for i in range(len(motifs))]
    motifs = [m.replace(name=n) for m, n in zip(motifs, names)]

    def pvalues(complete):
        rows = um.compare_motifs(motifs, compare_to=0, qvalue=1, complete=complete)
        return dict(zip(rows["target"], rows["pvalue"]))

    partial, complete = pvalues(False), pvalues(True)
    # Scoring the whole query sets the full copy much further apart from a
    # short motif matching only part of it.
    assert complete["m2"] / complete["m1"] > 5 * partial["m2"] / partial["m1"]
    matrix = um.compare_motifs(motifs, matrix_out="pvalue", complete=True)
    assert matrix.shape == (8, 8) and not np.allclose(matrix, matrix.T)
