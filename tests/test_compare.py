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


def test_compare_argument_checks():
    motifs = um.example_motifs()[:3]
    with pytest.raises(ValueError, match="matrix_out"):
        um.compare_motifs(motifs, matrix_out="distance")
    for q in (0, 1.5):
        with pytest.raises(ValueError, match=r"qvalue must be in \(0, 1\]"):
            um.compare_motifs(motifs, qvalue=q)
    with pytest.raises(ValueError, match="min_overlap"):
        um.compare_motifs(motifs, min_overlap=0)
    for i in (3, -1):
        with pytest.raises(ValueError, match=f"compare_to index {i} is out of range"):
            um.compare_motifs(motifs, compare_to=i)
    with pytest.raises(ValueError, match="no motif named 'nope'"):
        um.compare_motifs(motifs, compare_to=[0, "nope"])
    with pytest.raises(ValueError, match="null must be"):
        um.compare_motifs(motifs, null="normal")


def test_compare_backgrounds():
    motifs = um.example_motifs()[:5]
    bkg = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    as_dict = um.compare_motifs(motifs, null="parametric", bkg=bkg, matrix_out="pvalue")
    as_list = um.compare_motifs(motifs, null="parametric", bkg=[0.3, 0.2, 0.2, 0.3], matrix_out="pvalue")
    np.testing.assert_array_equal(as_dict, as_list)
    assert not np.array_equal(as_dict, um.compare_motifs(motifs, null="parametric", matrix_out="pvalue"))
    # RNA backgrounds are keyed by U.
    rna = [um.switch_alph(m) for m in motifs]
    rna_bkg = {"A": 0.3, "C": 0.2, "G": 0.2, "U": 0.3}
    np.testing.assert_array_equal(
        um.compare_motifs(rna, null="parametric", bkg=rna_bkg, matrix_out="pvalue"), as_dict)


def test_compare_invariants():
    motifs = um.example_motifs()[:6]
    scores = um.compare_motifs(motifs)
    np.testing.assert_allclose(scores, scores.T)
    np.testing.assert_allclose(np.diag(scores), 1)
    assert (scores <= 1 + 1e-12).all() and (scores >= -1).all()
    # Reverse complementing every motif changes nothing with rc.
    np.testing.assert_allclose(um.compare_motifs([um.motif_rc(m) for m in motifs]), scores)
    np.testing.assert_allclose(um.compare_motifs(motifs, rc=False), um.compare_motifs(motifs, rc=False).T)
    # Q-values are no smaller than P-values, and rows match the matrix.
    p = um.compare_motifs(motifs, matrix_out="pvalue")
    q = um.compare_motifs(motifs, matrix_out="qvalue")
    assert (q >= p - 1e-15).all()
    by_index = um.compare_motifs(motifs, compare_to=5, qvalue=1)
    assert by_index.equals(um.compare_motifs(motifs, compare_to="W-box", qvalue=1))
    assert by_index["qvalue"].is_sorted()
    for row in by_index.iter_rows(named=True):
        assert row["score"] == pytest.approx(scores[5, row["target_i"]])
        assert row["pvalue"] == pytest.approx(p[5, row["target_i"]])
        assert row["subject"] == "W-box" and row["overlap"] >= 5


def test_merge_edge_cases():
    w = um.convert_type(um.example_motifs()[5], "PPM")
    same = um.merge_motifs([w, w])
    np.testing.assert_allclose(same.matrix, w.matrix)
    assert same.name == "merged_W-box+W-box" and same.type == "PPM"
    # A motif and its reverse complement line up as one, on either strand.
    both = um.merge_motifs([w, um.motif_rc(w)])
    assert both.width == w.width
    rc = um.motif_rc(w).matrix
    assert np.allclose(both.matrix, w.matrix) or np.allclose(both.matrix, rc)
    assert um.merge_similar([]) == []
    clusters = um.merge_similar([], return_clusters=True)
    assert clusters.height == 0 and clusters.columns == ["name", "motif_i", "cluster"]


def test_tree_from_distances_and_labels():
    motifs = um.example_motifs()[:4]
    tree = um.motif_tree(motifs)
    # A tree from the documented distances is the same tree.
    dist = (1 - um.compare_motifs(motifs, min_overlap=6)) / 2
    from_dist = um.motif_tree(dist)
    assert from_dist.labels == ["1", "2", "3", "4"] and from_dist.motifs is None
    assert from_dist.merge.tolist() == tree.merge.tolist() and from_dist.order == tree.order
    np.testing.assert_allclose(from_dist.height, tree.height)
    assert um.motif_tree(dist.tolist(), labels=list("wxyz")).labels == list("wxyz")
    for bad in (np.zeros((2, 3)), np.zeros(3)):
        with pytest.raises(ValueError, match="must be square"):
            um.motif_tree(bad)
    with pytest.raises(ValueError, match="one label per row"):
        um.motif_tree(dist, labels=["a"])
    named = um.motif_tree(motifs, labels=["a", "b", "c", "d"])
    assert named.labels == ["a", "b", "c", "d"] and named.order == tree.order
    with pytest.raises(ValueError, match="one label per motif"):
        um.motif_tree(motifs, labels=["a"])
    # Missing attributes become empty labels.
    assert um.motif_tree(motifs, labels="altname").labels == [""] * 4
    # A single motif merges into itself.
    one = um.merge_motifs(motifs[0])
    assert one.name == "TATA-box" and one.type == "PPM"
    np.testing.assert_allclose(one.matrix, um.convert_type(motifs[0], "PPM").matrix)


def test_merging_with_a_reverse_complement_keeps_the_first_motif():
    m = um.example_motifs()[1]
    merged = um.merge_motifs([m, um.motif_rc(m)])
    assert merged.consensus == m.consensus


def test_whole_number_matrices_with_uneven_columns():
    with pytest.raises(ValueError, match="read as counts"):
        um.create_motif(np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]).T, alphabet="DNA")
    # With nsites the counts are rescaled instead.
    m = um.create_motif(np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]).T, alphabet="DNA", nsites=10,
                        type="PCM")
    assert m.matrix.sum(axis=0).tolist() == [10, 10]


def _ppm(m):
    return np.asarray(um.convert_type(m, "PPM").matrix, dtype=float)


def _column_scores(a, b, method):
    """Column scores as comparisons use them, for columns along the last axis:
    Pearson correlations centred on 1/4, 0 for uniform columns, or negated
    Euclidean distances."""
    if method == "EUCL":
        return -np.sqrt(((a - b) ** 2).sum(axis=-1))
    a, b = a - 0.25, b - 0.25
    na, nb = np.sqrt((a * a).sum(axis=-1)), np.sqrt((b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=-1) / (na * nb)
    return np.where((na * na < 1e-12) | (nb * nb < 1e-12), 0.0, r)


def _target_terms(query, row, wt, method, bkg):
    """For each target column, the query column it faces or None, and the
    weight of its score: 2 for facing columns with background overhangs, 1
    otherwise, and 0 for overhanging columns without them."""
    t_start = max(-row["offset"], 0)
    q_start = row["offset"] + t_start
    facing = [q_start + j - t_start if t_start <= j < t_start + row["overlap"] else None
              for j in range(wt)]
    weights = [(2 if bkg is not None else 1) if f is not None else (1 if bkg is not None else 0)
               for f in facing]
    q_out = 0.0
    if bkg is not None:
        out = [i for i in range(query.shape[1]) if not q_start <= i < q_start + row["overlap"]]
        q_out = float(_column_scores(query[:, out].T, np.asarray(bkg), method).sum())
    return facing, weights, q_out


def _alignment_total(query, target_cols, facing, weights, method, bkg):
    """The target's part of an alignment's score for targets given as arrays
    of shape (..., width, 4)."""
    total = 0.0
    for j, (f, w) in enumerate(zip(facing, weights)):
        if w == 0:
            continue
        other = query[:, f] if f is not None else np.asarray(bkg)
        total = total + w * _column_scores(target_cols[..., j, :], other, method)
    return total


def test_euclidean_and_background_scores():
    motifs = um.example_motifs()[:6]
    for method, overhang in (("EUCL", "ignore"), ("PCC", "background"), ("EUCL", "background")):
        kw = {"method": method, "overhang": overhang}
        scores = um.compare_motifs(motifs, **kw)
        np.testing.assert_allclose(scores, scores.T, atol=1e-12)
        if method == "EUCL":
            # Distances of 0 are not -0.
            assert not np.signbit(np.diag(scores)).any() and (np.diag(scores) == 0).all()
            assert (scores >= 0).all() and (scores <= np.sqrt(2)).all()
        else:
            np.testing.assert_allclose(np.diag(scores), 1)
        p = um.compare_motifs(motifs, matrix_out="pvalue", **kw)
        assert ((p >= 0) & (p <= 1)).all()
        assert (np.argmin(p, axis=1) == np.arange(6)).all()
        bkg = [0.25] * 4 if overhang == "background" else None
        rows = um.compare_motifs(motifs, compare_to=list(range(6)), qvalue=1, **kw)
        for row in rows.iter_rows(named=True):
            q, t = motifs[row["subject_i"]], motifs[row["target_i"]]
            assert row["score"] == pytest.approx(scores[row["subject_i"], row["target_i"]])
            query = _ppm(q)[::-1, ::-1] if row["strand"] == "-" else _ppm(q)
            target = _ppm(t)
            facing, weights, q_out = _target_terms(query, row, t.width, method, bkg)
            total = _alignment_total(query, target.T, facing, weights, method, bkg) + q_out
            n = q.width + t.width if bkg is not None else row["overlap"]
            want = total / n
            assert row["score"] == pytest.approx(-want if method == "EUCL" else want, abs=1e-12)

    # A uniform background correlates 0 with anything, so with correlations
    # the alignments stay as they are and only the scores are averaged over
    # both motifs.
    plain = um.compare_motifs(motifs, compare_to=0, qvalue=1)
    padded = um.compare_motifs(motifs, compare_to=0, qvalue=1, overhang="background")
    plain, padded = plain.sort("target_i"), padded.sort("target_i")
    assert plain.select("offset", "strand", "overlap").equals(padded.select("offset", "strand", "overlap"))
    widths = np.array([motifs[0].width + motifs[i].width for i in plain["target_i"]])
    np.testing.assert_allclose(padded["score"], plain["score"] * 2 * plain["overlap"] / widths)
    # Distance matrices cluster as they are.
    tree = um.motif_tree(um.compare_motifs(motifs, method="EUCL"), labels=[m.name for m in motifs])
    assert len(tree.merge) == 5


def test_compare_method_checks():
    motifs = um.example_motifs()[:3]
    with pytest.raises(ValueError, match="method must be 'PCC' or 'EUCL'"):
        um.compare_motifs(motifs, method="SW")
    with pytest.raises(ValueError, match="overhang must be 'ignore' or 'background'"):
        um.compare_motifs(motifs, overhang="pad")
    with pytest.raises(ValueError, match="cannot be combined"):
        um.compare_motifs(motifs, complete=True, overhang="background")
    with pytest.raises(ValueError, match="bkg must"):
        um.compare_motifs(motifs, overhang="background", bkg=[0.5, 0.5, 0.5, 0.5])
    # Complete scores work with distances too, and the defaults are PCC
    # without overhangs.
    assert um.compare_motifs(motifs, method="EUCL", complete=True, matrix_out="pvalue").shape == (3, 3)
    np.testing.assert_array_equal(um.compare_motifs(motifs, method="PCC", overhang="ignore"),
                                  um.compare_motifs(motifs))


def test_euclidean_and_background_pvalues_against_simulation():
    """Each P-value, less its Bonferroni correction, is the probability that a
    target of random columns, drawn from the columns of all the motifs as the
    empirical null has them, scores at least as well in the same alignment.
    Column scores are rounded to 51 bins for correlations and 201 for
    distances for the exact null, which can move a total by up to half a bin
    per column, so the simulated probabilities are taken that much either
    side of the observed total."""
    def make(s):
        return um.create_motif(s, nsites=20, pseudocount=1)

    motifs = [make(["TGACTCA", "TGAGTCA"]), make("TGACGTCA"), make(["CACGTGA", "CACATGA"]),
              make("GATTACAG"), make(["AACCCTA", "AACCGTA"]), make("TTGACGTCA")]
    pool = np.concatenate([_ppm(m).T for m in motifs])
    rng = np.random.default_rng(1)
    checked = 0
    for method, overhang, bkg in (("EUCL", "ignore", None), ("EUCL", "background", [0.25] * 4),
                                  ("PCC", "background", [0.35, 0.15, 0.15, 0.35]),
                                  ("EUCL", "background", [0.35, 0.15, 0.15, 0.35])):
        lo, hi, steps = (-np.sqrt(2), 0.0, 200) if method == "EUCL" else (-1.0, 1.0, 50)
        half_bin = (hi - lo) / steps / 2
        rows = um.compare_motifs(motifs, compare_to=[0, 3], qvalue=1, min_overlap=4, method=method,
                                 overhang=overhang, bkg=bkg)
        for row in rows.iter_rows(named=True):
            q, t = motifs[row["subject_i"]], motifs[row["target_i"]]
            mo = min(4, q.width, t.width)
            n_tested = 2 * (q.width + t.width - 2 * mo + 1)
            tail = row["pvalue"] / n_tested
            if row["pvalue"] >= 1 or not 1e-3 < tail < 0.5:
                continue
            query = _ppm(q)[::-1, ::-1] if row["strand"] == "-" else _ppm(q)
            facing, weights, _ = _target_terms(query, row, t.width, method, bkg)
            observed = _alignment_total(query, _ppm(t).T, facing, weights, method, bkg)
            random = pool[rng.integers(len(pool), size=(200_000, t.width))]
            sims = _alignment_total(query, random, facing, weights, method, bkg)
            slack = (sum(weights) + 1) * half_bin + 1e-9
            low, high = (sims >= observed + slack).mean(), (sims >= observed - slack).mean()
            se = np.sqrt(max(high, 1e-4) * (1 - low) / len(sims))
            ctx = f"{method} {overhang} {bkg} {row['subject']} vs {row['target']}"
            assert low - 4 * se <= tail <= high + 4 * se, (ctx, low, tail, high)
            checked += 1
    assert checked >= 10
