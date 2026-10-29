"""Column conversions, scoring helpers, column metrics, gaps, alphabets,
filtering, masking and small helpers against the R package."""

import math

import numpy as np
import pytest

import universalmotif as um

from conftest import (
    args,
    assert_motif_matches,
    load_reference,
    motif_from_record,
    nums,
    r_rounding_differs,
)


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
        if case["fn"] == "ppm_to_pcm":
            pos, nsites = nums(case["args"]["pos"]), nums(case["args"]["nsites"])[0]
            if r_rounding_differs(pos, nsites):
                # R adds the whole rounding error to the largest count.
                got = um.ppm_to_pcm(pos, nsites)
                assert got.sum() == nsites
                assert np.all(np.abs(got - np.array(pos) * nsites) < 1)
                n += 1
                continue
        if case["fn"] in fns:
            got = fns[case["fn"]](case["args"])
            want = np.array(nums(case["out"]), dtype=float)
            if case["fn"] == "ppm_to_icm":
                # R's NaN for zero probabilities in relative entropy is 0 here.
                want[np.isnan(want) & (np.array(nums(case["args"]["pos"])) == 0)] = 0.0
            if case["fn"] == "icm_to_ppm" and not any(nums(case["args"]["pos"])):
                # R gives NaN for a column of zeros; the port a uniform one.
                want = np.full(len(want), 1 / len(want))
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15,
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
            # R adds the rounded entries as floats and truncates the total
            # again, so it can be 0.001 off the grid sum used in scanning.
            got = um.score_match(m, a["matches"], af)
            np.testing.assert_allclose(got, nums(want), rtol=0, atol=0.001 + 1e-9, err_msg=ctx)
            continue
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
        elif case["method"] == "WPCC":
            # The R package does not normalize the WPCC weights, so check against a weighted
            # correlation computed here.
            w = np.add(nums(case["bkg1"]), nums(case["bkg2"]))
            x, y = np.array(nums(case["x"])), np.array(nums(case["y"]))
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            bot = np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))
            want = 1.0 if bot == 0 else np.sum(w * (x - mx) * (y - my)) / bot
            assert call() == pytest.approx(want, rel=1e-12, abs=1e-15), case
        else:
            assert call() == pytest.approx(case["value"], rel=1e-12, abs=1e-15), case
    u = [0.25] * 4
    x, y = [0.5, 0.1, 0.1, 0.3], [0.7, 0.1, 0.1, 0.1]
    assert um.compare_columns(x, y, "WPCC", u, u) == pytest.approx(um.compare_columns(x, y, "PCC"))
    assert um.compare_columns([0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], "KL") == 0
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
    for case in mr["cases"]:
        # R counts positions from 1, with the end included.
        ranges = [(c, s - 1, e) for c, s, e in zip(case["chrom"], case["start"], case["end"])]
        call = lambda: um.mask_ranges(seqs, ranges, case["letter"])  # noqa: E731
        want = case.get("value")
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
            # The R package declares each complementary pair twice, once from
            # each letter; the port declares it once.
            seen, want = set(), []
            for line in case["value"].split("\n"):
                if " ~ " in line:
                    x, y = line[0], line.split(" ~ ")[1][0]
                    if (y, x) in seen:
                        continue
                    seen.add((x, y))
                want.append(line)
            assert call().rstrip("\n") == "\n".join(want), a
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
            # The R package drops a complete last window here.
            got = got[:-1]
        assert got == want, case
    with pytest.raises(ValueError):
        um.slide_fun("ACGT", len, window=10, return_incomp=False)
    s = um.shuffle_string("AACCGGTTAACCGGTT", k=2, seed=1)
    assert sorted(s) == sorted("AACCGGTTAACCGGTT") and s == um.shuffle_string("AACCGGTTAACCGGTT", k=2, seed=1)


@pytest.mark.parametrize("fn", [
    lambda m: um.motif_range(m),
    lambda m: um.motif_score(m, 0.5),
    lambda m: um.score_match(m, "ACGT"),
    lambda m: um.get_scores(m),
    lambda m: um.get_matches(m, 1.0),
    lambda m: um.prob_match(m, "ACGT"),
])
def test_scoring_helpers_take_one_motif(fn):
    m = um.create_motif("ACGT", pseudocount=1)
    with pytest.raises(TypeError, match="expected a single Motif, got list"):
        fn([m])
    with pytest.raises(TypeError, match="got str"):
        fn("ACGT")


def test_column_conversion_round_trips():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    bkg = [0.3, 0.2, 0.2, 0.3]
    np.testing.assert_allclose(um.pcm_to_ppm(um.ppm_to_pcm(p, 10)), p)
    np.testing.assert_allclose(um.pwm_to_ppm(um.ppm_to_pwm(p, smooth=False)), p)
    np.testing.assert_allclose(um.pwm_to_ppm(um.ppm_to_pwm(p, bkg, smooth=False), bkg), p)
    np.testing.assert_allclose(um.icm_to_ppm(um.ppm_to_icm(p)), p)
    # Smoothing spreads the pseudocount evenly over the letters, whatever
    # the background, as counts of nsites.
    smoothed = um.pwm_to_ppm(um.ppm_to_pwm(p, bkg, pseudocount=2, nsites=20), bkg)
    np.testing.assert_allclose(smoothed, (p * 20 + 2 / 4) / 22)
    # Each column's information content sums to log2(4) minus its entropy.
    ic = um.ppm_to_icm(p)
    assert ic.sum() == pytest.approx(2 + (p * np.log2(p)).sum())
    np.testing.assert_allclose(ic / ic.sum(), p)
    rel = um.ppm_to_icm(p, relative_entropy=True)
    assert (rel >= 0).all() and rel[:2].tolist() == [0, 0]


def test_motif_conversion_chains():
    pcm = um.example_motifs()[0]
    for chain in (["PPM", "PCM"], ["PWM", "PPM", "PCM"], ["ICM", "PPM", "PCM"], ["PWM", "ICM", "PCM"]):
        m = pcm
        for t in chain:
            m = um.convert_type(m, t)
            assert m.type == t
        np.testing.assert_allclose(m.matrix, pcm.matrix, atol=1e-9)
        assert m.nsites == pcm.nsites
    ppm = um.convert_type(pcm, "PPM")
    np.testing.assert_allclose(ppm.matrix.sum(axis=0), 1)
    for t in ("PWM", "ICM", "PCM"):
        np.testing.assert_allclose(um.convert_type(um.convert_type(ppm, t), "PPM").matrix, ppm.matrix,
                                   atol=1e-12)


def test_score_helpers_agree():
    m = um.example_motifs()[5]
    lo, hi = um.motif_range(m)
    assert um.motif_score(m, [0, 1]).tolist() == [lo, hi]
    assert um.motif_score(m, [0, 1], "fromzero").tolist() == [0, hi]
    with pytest.raises(ValueError, match="threshold type"):
        um.motif_score(m, 0.5, "logodds")
    small = um.create_motif(["TGACA", "TGAGA"], pseudocount=1, type="PCM")
    scores = um.get_scores(small)
    assert len(scores) == 4 ** 5 and np.all(np.diff(scores) <= 0)
    assert scores[0] == um.motif_range(small)[1] and scores[-1] == um.motif_range(small)[0]
    for s in (0, 3, 5):
        matches = um.get_matches(small, s)
        assert len(matches) == (scores >= s).sum() and len(set(matches)) == len(matches)
    # The probability of a word only depends on the background.
    bkg = {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}
    words = ["ACG", "TTT", "GGC"]
    np.testing.assert_allclose(um.prob_match(um.create_motif("ACG", bkg=list(bkg.values())), words),
                               um.prob_match_bkg(bkg, words))
    with pytest.raises(ValueError, match="single letters"):
        um.prob_match_bkg({"AA": 1.0}, "A")


def test_mask_and_slide_checks():
    with pytest.raises(ValueError, match="single character"):
        um.mask_seqs("ACGT", "AC", letter="--")
    with pytest.raises(ValueError, match="single character"):
        um.mask_ranges({"a": "ACGT"}, [("a", 0, 1)], letter="")
    # Sites on either strand are masked and the rest is left as it is.
    seq = "GATTACAxxTGTAATCcc"
    masked = um.mask_seqs(seq, "GATTACA", rc=True)
    assert masked == "-------xx-------cc" and len(masked) == len(seq)
    assert um.mask_seqs(seq, "GATTACA") == "-------xxTGTAATCcc"
    assert um.mask_ranges({"a": "ACGTACGT"}, [("a", 0, 2), ("a", 6, 8)], letter="N") == {"a": "NNGTACNN"}
    with pytest.raises(TypeError, match="single string"):
        um.shuffle_string(["ACGT"])
    seq = "AACCGGTTACGTACGTAAAC"
    assert um.shuffle_string(seq, k=2, seed=3) == um.shuffle_sequences([seq], k=2, seed=3)[0]
    assert um.shuffle_string(seq, method="linear", k=2, seed=3) == um.shuffle_sequences(
        seq, method="linear", k=2, seed=3)
    with pytest.raises(ValueError, match="must not be empty"):
        um.slide_fun("", len)
    with pytest.raises(ValueError, match="window should be a positive"):
        um.slide_fun("ACGT", len, window=0)
    with pytest.raises(ValueError, match="overlap should be a non-negative"):
        um.slide_fun("ACGT", len, window=2, overlap=-1)
    with pytest.raises(ValueError, match="greater or equal to window"):
        um.slide_fun("ACGT", len, window=2, overlap=2)
    assert um.slide_fun("ACGT", lambda w: w, window=10) == ["ACGT"]
    assert um.slide_fun("ACGTA", lambda w: w, window=2) == ["AC", "GT", "A"]
    assert um.slide_fun("ACGTA", lambda w: w, window=2, return_incomp=False) == ["AC", "GT"]
