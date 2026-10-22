"""Column conversions, scoring helpers, column metrics, gaps, alphabets,
filtering, masking and small helpers against the R package."""

import math

import numpy as np
import pytest

import universalmotif as um

from conftest import args, assert_motif_matches, load_reference, motif_from_record, nums


def _ref(key):
    return load_reference("utils.json", key)


def _motif(name):
    return motif_from_record(_ref("motifs")[name])


def test_column_conversions():
    fns = {"pcm_to_ppm": lambda a: um.pcm_to_ppm(nums(a["pos"]), a["pseudocount"]),
           "ppm_to_pcm": lambda a: um.ppm_to_pcm(nums(a["pos"]), nums(a["nsites"])[0]),
           "ppm_to_pwm": lambda a: um.ppm_to_pwm(nums(a["pos"]), nums(a["bkg"]) or None,
                                                 a["pseudocount"], a["nsites"]),
           "pwm_to_ppm": lambda a: um.pwm_to_ppm(nums(a["pos"]), nums(a["bkg"]) or None),
           "ppm_to_icm": lambda a: um.ppm_to_icm(nums(a["pos"]), nums(a["bkg"]) or None,
                                                 a["relative_entropy"]),
           "icm_to_ppm": lambda a: um.icm_to_ppm(nums(a["pos"]))}
    n = 0
    for case in load_reference("columns.json"):
        if case["fn"] in fns:
            got = fns[case["fn"]](case["args"])
            np.testing.assert_allclose(got, nums(case["out"]), rtol=1e-12, atol=1e-15,
                                       err_msg=f"{case['fn']} {case['args']}")
            n += 1
    assert n > 100
    # Without smoothing, the pseudocount is not applied.
    np.testing.assert_allclose(um.ppm_to_pwm([0.5, 0.5, 0, 0], smooth=False)[:2], [1.0, 1.0])


def test_scoring_helpers():
    for case in _ref("scoring"):
        m = _motif(case["motif"])
        a = case["args"]
        af = a.get("allow_nonfinite", False)
        ctx = f"{case['fn']} {case['motif']} {a}"
        if "error" in case:
            with pytest.raises(ValueError):
                um.get_matches(m, a["score"], af)
            continue
        want = case["value"]
        f = case["fn"]
        if f == "motif_range":
            got = um.motif_range(m, af)
        elif f == "motif_score":
            got = um.motif_score(m, nums(a["threshold"]), a["threshold_type"], af)
        elif f == "score_match":
            got = um.score_match(m, a["matches"], af)
        elif f == "get_scores":
            got = um.get_scores(m, af)
        elif f == "get_matches":
            assert um.get_matches(m, a["score"], af) == list(np.atleast_1d(want)), ctx
            continue
        else:
            got = um.prob_match(m, a["matches"], a["allow_zero"])
        np.testing.assert_allclose(got, nums(want), rtol=1e-12, err_msg=ctx)
    p = _ref("prob_match_bkg")
    bkg = dict(zip(p["letters"], nums(p["bkg"])))
    np.testing.assert_allclose(um.prob_match_bkg(bkg, p["matches"]), nums(p["value"]), rtol=1e-12)
    assert um.prob_match_bkg(bkg, "AT") == pytest.approx(0.09)
    m = _motif("example")
    assert isinstance(um.score_match(m, "TATATAT"), float)
    assert isinstance(um.motif_score(m, 0.5), float)
    lo, hi = um.motif_range(_motif("zero"), allow_nonfinite=True)
    assert lo == -math.inf and hi > 0
    assert um.get_matches(m, math.inf) == []


def test_column_metrics_and_averages():
    for case in _ref("compare_columns"):
        call = lambda: um.compare_columns(nums(case["x"]), nums(case["y"]), case["method"],  # noqa: E731
                                          nums(case["bkg1"]), nums(case["bkg2"]),
                                          case["nsites1"], case["nsites2"])
        if "error" in case:
            with pytest.raises(ValueError):
                call()
        else:
            assert call() == pytest.approx(case["value"], rel=1e-12, abs=1e-15), case
    with pytest.raises(ValueError, match="method"):
        um.compare_columns([0.5, 0.5], [0.5, 0.5], "IS")
    avg = _ref("average_ic")
    motifs = [motif_from_record(r) for r in avg["motifs"]]
    for case in avg["cases"]:
        np.testing.assert_allclose(um.average_ic(motifs, case["average"]), nums(case["value"]),
                                   rtol=1e-12, err_msg=case["average"])
    assert isinstance(um.average_ic(motifs[0]), float)


def test_gaps_and_alphabets():
    random = _motif("random")
    for case in _ref("add_gap"):
        a = args(case["args"])
        call = lambda: um.add_gap(random, a.get("gaploc"), a.get("mingap", 1), a.get("maxgap", 5))  # noqa: E731
        if "motif" in case:
            assert_motif_matches(call(), case["motif"], f"add_gap {a}")
        else:
            with pytest.raises(ValueError):
                call()
    gapped = um.add_gap(random, 2, 1, 4)
    for case in _ref("ungap"):
        assert_motif_matches(um.ungap(gapped, case["delete"]), case["motif"], "ungap")
    assert um.add_gap(um.ungap(gapped)).gaploc == [3]
    for case in _ref("switch_alph"):
        m = motif_from_record(case["input"])
        if "output" in case:
            assert_motif_matches(um.switch_alph(m), case["output"], "switch_alph")
        else:
            with pytest.raises(ValueError):
                um.switch_alph(m)
    with pytest.raises(ValueError):
        um.add_gap(random, -1)


def test_filter_motifs():
    f = _ref("filter")
    motifs = [motif_from_record(r) for r in f["motifs"]]
    for case in f["cases"]:
        kw = {k.replace(".", "_"): v for k, v in args(case["args"]).items()}
        got = [m.name for m in um.filter_motifs(motifs, **kw)]
        assert got == list(np.atleast_1d(case["names"])), kw


def test_masking():
    for case in _ref("mask_seqs"):
        call = lambda: um.mask_seqs(case["seqs"], case["pattern"], case["rc"], case["letter"],  # noqa: E731
                                    case["alphabet"])
        if "value" in case:
            assert call() == list(np.atleast_1d(case["value"])), case
        else:
            with pytest.raises(ValueError):
                call()
    # Letters match in either case; unmasked ones keep theirs.
    assert um.mask_seqs({"a": "ACGTacgt"}, "acg") == {"a": "---T---t"}
    assert um.mask_seqs("AAA", "AA", letter="N") == "NNN"

    mr = _ref("mask_ranges")
    seqs = dict(zip(mr["names"], mr["seqs"]))
    # The R package fails on the first and third cases (see DIVERGENCES.md).
    fixed = {0: ["------GTACGTACGT", "GGGG----AAAATTTT"], 2: ["NNNNNCGTACGTANNN", "GGGGCCCCAAAATTTT"]}
    for i, case in enumerate(mr["cases"]):
        ranges = list(zip(case["chrom"], case["start"], case["end"]))
        call = lambda: um.mask_ranges(seqs, ranges, case["letter"])  # noqa: E731
        want = fixed.get(i, case.get("value"))
        if want is not None:
            assert list(call().values()) == list(want), case
        else:
            with pytest.raises(ValueError):
                call()


def test_small_helpers(tmp_path):
    for case in _ref("meme_alph"):
        a = case["args"]
        call = lambda: um.meme_alph(a["core"], complements=a.get("complements"),  # noqa: E731
                                    ambiguity=a.get("ambiguity"), like=a.get("like"),
                                    alph_name=a.get("alph.name"), letter_names=a.get("letter.names"),
                                    colours=a.get("colours"))
        if "value" in case:
            assert call().rstrip("\n") == case["value"], a
        else:
            with pytest.raises(ValueError):
                call()
    um.meme_alph("QWER", file=tmp_path / "alph.txt")
    assert (tmp_path / "alph.txt").read_text().startswith("ALPHABET\n")
    for case in _ref("log_string_pval"):
        assert um.log_string_pval(case["pval"]) == pytest.approx(nums(case["value"])[0], rel=1e-12)
    assert um.log_string_pval("1e-400") == pytest.approx(-400 * math.log(10))
    for case in _ref("slide_fun"):
        got = um.slide_fun(case["string"], lambda w: sum(c in "GC" for c in w), case["window"],
                           case["overlap"], case["return_incomp"])
        want = nums(case["gc"])
        if not case["return_incomp"] and len(got) == len(want) + 1:
            # The R package drops a complete last window here (see
            # calc_windows in DIVERGENCES.md).
            got = got[:-1]
        assert got == want, case
    with pytest.raises(ValueError):
        um.slide_fun("ACGT", len, window=10, return_incomp=False)
    s = um.shuffle_string("AACCGGTTAACCGGTT", k=2, seed=1)
    assert sorted(s) == sorted("AACCGGTTAACCGGTT") and s == um.shuffle_string("AACCGGTTAACCGGTT", k=2, seed=1)
