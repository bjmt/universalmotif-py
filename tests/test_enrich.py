import numpy as np
import polars as pl
import pytest

import universalmotif as um
from universalmotif._compare import _tree

from conftest import args, load_reference, motif_from_record, nums


def _ref(key):
    return load_reference("enrich.json", key)


def _motifs():
    return [motif_from_record(r) for r in _ref("motifs")]


def _close(got, want, ctx):
    got = np.array([np.nan if v is None else v for v in got], dtype=float)
    np.testing.assert_allclose(got, np.array(nums(want), dtype=float), rtol=1e-12, err_msg=ctx)


def _as_list(x):
    return x if isinstance(x, list) else [x]


@pytest.mark.filterwarnings("ignore:.*cannot reach a P-value")
def test_enrich_reference():
    ms = _motifs()
    tgt, bkg = _ref("targets"), _ref("background")
    for case in _ref("enrich"):
        a = args(case["args"])
        got = um.enrich_motifs(
            ms,
            tgt,
            bkg,
            pvalue=a.get("pvalue", 1e-4),
            qvalue=a["qvalue"],
            test=a.get("test", "seqs"),
            rc=a.get("RC", True),
            pseudocount=a.get("pseudocount", 1),
        )
        t = case["table"]
        assert got["motif"].to_list() == _as_list(t["motif"]), a
        assert got["motif_i"].to_list() == [i - 1 for i in nums(t["motif.i"])], a
        assert got["consensus"].to_list() == _as_list(t["consensus"]), a
        for col in ("target_seq_n", "target_seq_hits", "target_site_hits", "bkg_seq_n",
                    "bkg_seq_hits", "bkg_site_hits"):
            assert got[col].to_list() == nums(t[col.replace("_", ".")]), (a, col)
        for col in ("enrichment", "log2_enrichment", "pvalue", "qvalue"):
            _close(got[col].to_list(), t[col.replace("_", ".")], f"{a} {col}")


@pytest.mark.filterwarnings("ignore:.*cannot reach a P-value")
def test_enrich_shuffled_background():
    ms = _motifs()
    tgt = _ref("targets")
    a = um.enrich_motifs(ms, tgt, seed=2, qvalue=1)
    b = um.enrich_motifs(ms, tgt, seed=2, qvalue=1, nthreads=2)
    assert a.equals(b)
    assert set(a["motif"][:2]) == {"creb", "sp1"}
    with pytest.raises(ValueError):
        um.enrich_motifs([um.create_motif("MKVL", alphabet="AA"), ms[0]], tgt)


def _bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)[::-1]
    q = np.minimum.accumulate(p[order] * len(p) / np.arange(len(p), 0, -1))
    out = np.empty_like(p)
    out[order] = np.minimum(q, 1)
    return out


def _check_coocc(got, t, ctx):
    # The port does not test a motif against itself, so
    # self pairs are checked on their own and left out of the comparison with
    # R, whose q-values are recomputed without them.
    own = got.filter(pl.col("motif_a") == pl.col("motif_b"))
    assert own["pvalue"].null_count() == own.height, ctx
    assert (own["b_only"] == 0).all(), ctx
    got = got.filter(pl.col("motif_a") != pl.col("motif_b"))
    t = {k: _as_list(v) for k, v in t.items()}
    keep = [k for k in range(len(t["motif_a"])) if t["motif_a"][k] != t["motif_b"][k]]
    t = {k: [v[i] for i in keep] if len(v) == len(t["motif_a"]) else v for k, v in t.items()}
    for col in ("motif_a", "motif_b"):
        assert got[col].to_list() == t[col], (ctx, col)
    for col in ("a_only", "b_only", "both", "neither"):
        assert got[col].to_list() == nums(t[col]), (ctx, col)
    for col in ("odds_ratio", "pvalue"):
        _close(got[col].to_list(), t[col], f"{ctx} {col}")
    p = np.array(nums(t["pvalue"]), dtype=float)
    q = np.full(len(p), np.nan)
    tested = ~np.isnan(p)
    q[tested] = _bh(p[tested])
    _close(got["qvalue"].to_list(), list(q), f"{ctx} qvalue")
    if "both.clustered" in t:
        assert got["both_clustered"].to_list() == nums(t["both.clustered"]), ctx
        _close(got["median_distance"].to_list(), t["median.distance"], f"{ctx} median")
    else:
        assert "both_clustered" not in got.columns


def test_coocc_reference():
    ms = _motifs()
    tgt = _ref("targets")
    for case in _ref("coocc"):
        a = args(case["args"])
        got = um.motif_coocc(
            ms,
            tgt,
            pvalue=a["pvalue"],
            rc=a.get("RC", True),
            no_overlaps=a.get("no.overlaps", False),
            max_distance=a.get("max.distance"),
            min_coocc=a.get("min.coocc", 1),
            pseudocount=a.get("pseudocount", 0),
            self_pairs=a.get("self.pairs", False),
        )
        _check_coocc(got, case["table"], a)

    h = _ref("coocc_hits")
    hits = pl.DataFrame({
        "motif_i": [int(x) - 1 for x in h["motif.i"]],
        "sequence_i": [int(x) - 1 for x in h["sequence.i"]],
        "start": [int(x) for x in h["start"]],
    })
    for case in _ref("coocc_hit_cases"):
        a = args(case["args"])
        kw = dict(
            max_distance=a.get("max.distance"),
            min_coocc=a.get("min.coocc", 1),
            pseudocount=a.get("pseudocount", 0),
            self_pairs=a.get("self.pairs", False),
        )
        got = um.motif_coocc(ms[:4], hits=hits, n_sequences=10, **kw)
        _check_coocc(got, case["table"], a)
        # Motif names work in place of indices.
        names = ["creb", "sp1", "ebox", "tata"]
        by_name = hits.with_columns(pl.Series("motif", [names[i] for i in hits["motif_i"]])).drop("motif_i")
        assert um.motif_coocc(ms[:4], hits=by_name, n_sequences=10, **kw).equals(got)
    with pytest.raises(ValueError):
        um.motif_coocc(ms, hits=hits)
    with pytest.raises(ValueError):
        um.motif_coocc(ms, tgt, hits=hits, n_sequences=10)


def test_peaks_reference():
    h = _ref("peak_hits")
    hits = pl.DataFrame({
        "motif": h["motif"],
        "sequence": h["sequence"],
        # R counts positions from 1.
        "start": [int(x) - 1 for x in h["start"]],
        "end": [int(x) for x in h["end"]],
        "score": nums(h["score"]),
    })
    for case in _ref("peaks"):
        a = args(case["args"])
        got = um.motif_peaks(
            hits,
            300,
            mode=a.get("mode", "central"),
            seq_strand=a.get("seq.strand"),
            qvalue=a["qvalue"],
            min_window=a.get("min.window", 10),
            max_window=a.get("max.window"),
            window_step=a.get("window.step", 10),
            position_step=a.get("position.step", 5),
        )
        t = case["table"]
        ctx = {k: v for k, v in a.items() if k != "seq.strand"}
        assert got["motif"].to_list() == _as_list(t["motif"]), ctx
        assert got["motif_i"].to_list() == [i - 1 for i in nums(t["motif.i"])], ctx
        for col in ("nhits", "hits_in", "hits_out"):
            assert got[col].to_list() == nums(t[col.replace("_", ".")]), (ctx, col)
        for col in ("best_window", "best_centre", "expected_in", "enrichment", "log2_enrichment",
                    "pvalue", "qvalue"):
            want = t[col.replace("_", ".").replace("centre", "center")]
            if col == "best_centre":
                want = [None if x is None else x - 1 for x in nums(want)]
            _close(got[col].to_list(), want, f"{ctx} {col}")
        assert got["centres"].to_list() == [[int(v) - 1 for v in nums(c)] for c in t["centers"]], ctx
    # Hits straight from scan_sequences work too.
    scanned = um.scan_sequences(_motifs()[:1], _ref("peak_sequences"), pvalue=1e-3)
    assert um.motif_peaks(scanned, 300, qvalue=1)["best_centre"].to_list() == [149.0]


def test_cutree():
    for case in load_reference("enrich.json", "cutree"):
        dist = np.array([nums(r) for r in case["dist"]])
        labels = [str(i) for i in range(len(dist))]
        d = um._core._hclust(dist.tolist(), case["method"], labels)
        tree = _tree(d, labels, case["method"])
        for k, want in enumerate(case["k"], start=1):
            assert (tree.cutree(k) + 1).tolist() == nums(want)
        for h, want in zip(nums(case["h"]), case["by_h"]):
            assert (tree.cutree(h=h) + 1).tolist() == nums(want)
    with pytest.raises(ValueError):
        tree.cutree()
    with pytest.raises(ValueError):
        tree.cutree(0)


def test_coocc_hits_can_start_at_zero():
    motifs = um.example_motifs()[:2]
    hits = pl.DataFrame({"motif_i": [0, 1], "sequence_i": [0, 0], "start": [0, 5]})
    res = um.motif_coocc(motifs, hits=hits, n_sequences=2, max_distance=10)
    assert res["both"].to_list() == [1]
    assert res["median_distance"].to_list() == [5.0]


# Small synthetic cases. With poly-A flanks and rc=False, the only hits
# are the exact sites planted.

_FLANK = "A" * 25


def _plant(*sites):
    seq = list(_FLANK)
    for pos, site in sites:
        seq[pos:pos + len(site)] = site
    return "".join(seq)


def _ap1():
    return um.create_motif("TGACTCA", name="ap1")


def test_enrich_counts_and_fisher():
    stats = pytest.importorskip("scipy.stats")
    gbox = um.create_motif("CACGTGC", name="gbox")
    tgt = [_plant((5, "TGACTCA"))] * 6 + [_plant((5, "TGACTCA"), (15, "TGACTCA"))] * 2 + [_FLANK] * 2
    bkg = [_plant((5, "TGACTCA"))] + [_FLANK] * 9
    for pc in (0, 1):
        res = um.enrich_motifs([_ap1(), gbox], tgt, bkg, pseudocount=pc, qvalue=1, rc=False)
        assert res["motif"].to_list() == ["ap1", "gbox"]
        assert res["consensus"].to_list() == ["TGACTCA", "CACGTGC"]
        ap1 = res.row(0, named=True)
        assert (ap1["target_seq_n"], ap1["target_seq_hits"], ap1["target_site_hits"]) == (10, 8, 10)
        assert (ap1["bkg_seq_n"], ap1["bkg_seq_hits"], ap1["bkg_site_hits"]) == (10, 1, 1)
        assert ap1["enrichment"] == pytest.approx(8.0)
        assert ap1["log2_enrichment"] == pytest.approx(3.0)
        want = stats.fisher_exact([[8 + pc, 2 + pc], [1 + pc, 9 + pc]], alternative="greater").pvalue
        assert ap1["pvalue"] == pytest.approx(want)
        assert ap1["log10_pvalue"] == pytest.approx(np.log10(want))
        assert res["qvalue"].to_list() == pytest.approx(_bh(res["pvalue"].to_list()))
        assert res["log10_qvalue"].to_list() == pytest.approx(np.log10(res["qvalue"].to_list()))
    # By sites, each sequence offers 25 - 7 + 1 = 19 positions.
    sites = um.enrich_motifs(_ap1(), tgt, bkg, test="sites", qvalue=1, rc=False).row(0, named=True)
    assert sites["enrichment"] == pytest.approx(10.0)
    want = stats.fisher_exact([[11, 190 - 10 + 1], [2, 190 - 1 + 1]], alternative="greater").pvalue
    assert sites["pvalue"] == pytest.approx(want)


def test_enrich_rc_and_names():
    # Sites on the reverse strand count only with rc.
    tgt = [_plant((5, "TGAGTCA"))] * 5
    bkg = [_FLANK] * 5
    fwd = um.enrich_motifs(_ap1(), tgt, bkg, rc=False, qvalue=1)
    both = um.enrich_motifs(_ap1(), tgt, bkg, rc=True, qvalue=1)
    assert fwd["target_seq_hits"].to_list() == [0]
    assert both["target_seq_hits"].to_list() == [5]
    # Repeated names are made unique, and the order of motifs is kept in
    # motif_i.
    res = um.enrich_motifs([_ap1(), _ap1()], tgt, bkg, qvalue=1)
    assert res["motif"].to_list() == ["ap1", "ap1.1"]
    assert res["motif_i"].to_list() == [0, 1]
    # A dict of sequences works like a list.
    named = um.enrich_motifs(_ap1(), dict(enumerate(tgt)), {"b": bkg[0]}, qvalue=1)
    assert named["target_seq_n"].to_list() == [5] and named["bkg_seq_n"].to_list() == [1]


def test_enrich_qvalue_filter():
    tgt = [_plant((5, "TGACTCA"))] * 8 + [_FLANK] * 2
    bkg = [_FLANK] * 10
    other = um.create_motif("CACGTGC", name="gbox")
    res = um.enrich_motifs([other, _ap1()], tgt, bkg, rc=False, qvalue=0.1)
    assert res["motif"].to_list() == ["ap1"] and res["motif_i"].to_list() == [1]
    assert um.enrich_motifs([other, _ap1()], tgt, bkg, rc=False, qvalue=1).height == 2
    # Every motif is returned by default, the most significant first.
    every = um.enrich_motifs([other, _ap1()], tgt, bkg, rc=False)
    assert every["motif"].to_list() == ["ap1", "gbox"] and every["qvalue"][1] > 0.1


def test_enrich_argument_errors():
    seqs = [_FLANK]
    cases = [
        ({"test": "positions"}, "test must be 'seqs' or 'sites'"),
        ({"qvalue": 0}, r"qvalue must be in \(0, 1\]"),
        ({"qvalue": 2}, r"qvalue must be in \(0, 1\]"),
        ({"shuffle_k": 0}, "shuffle_k must be at least 1"),
        ({"pseudocount": -1}, "pseudocount must be non-negative"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            um.enrich_motifs(_ap1(), seqs, seqs, **kw)
    with pytest.raises(ValueError, match="no motifs"):
        um.enrich_motifs([], seqs, seqs)
    with pytest.raises(ValueError, match="no background sequences"):
        um.enrich_motifs(_ap1(), seqs, [])


def _coocc_motifs():
    return [um.create_motif(s, name=n) for s, n in (("TGACTCA", "a"), ("CACGTGA", "b"),
                                                    ("TTTTTTT", "c"))]


def _coocc_hits():
    # a in sequences 0-5, b in 3-7 and c in 9 only.
    rows = [(0, s, 10) for s in range(6)] + [(1, s, 30) for s in range(3, 8)] + [(2, 9, 0)]
    return pl.DataFrame(rows, schema=["motif_i", "sequence_i", "start"], orient="row")


def test_coocc_tables_by_hand():
    stats = pytest.importorskip("scipy.stats")
    res = um.motif_coocc(_coocc_motifs(), hits=_coocc_hits(), n_sequences=10)
    assert res.columns == ["motif_a", "motif_b", "a_only", "b_only", "both", "neither",
                           "odds_ratio", "pvalue", "qvalue"]
    assert res.select("motif_a", "motif_b", "a_only", "b_only", "both", "neither").rows() == [
        ("a", "b", 3, 2, 3, 2), ("a", "c", 6, 1, 0, 3), ("b", "c", 5, 1, 0, 4),
    ]
    want = stats.fisher_exact([[3, 3], [2, 2]], alternative="greater")
    assert res["odds_ratio"][0] == pytest.approx(want.statistic)
    assert res["pvalue"][0] == pytest.approx(want.pvalue)
    # Pairs sharing no sequence are not tested, and come last.
    assert res["pvalue"][1:].null_count() == 2 and res["qvalue"][1:].null_count() == 2
    # With a pseudocount, every cell gains one.
    pc = um.motif_coocc(_coocc_motifs(), hits=_coocc_hits(), n_sequences=10, pseudocount=1)
    assert pc["pvalue"][0] == pytest.approx(
        stats.fisher_exact([[4, 4], [3, 3]], alternative="greater").pvalue)
    assert pc["a_only"].to_list() == res["a_only"].to_list()
    # min_coocc above the shared count leaves the pair untested.
    strict = um.motif_coocc(_coocc_motifs(), hits=_coocc_hits(), n_sequences=10, min_coocc=4)
    assert strict["pvalue"].null_count() == 3
    # Sequences without any hit still count in neither.
    more = um.motif_coocc(_coocc_motifs(), hits=_coocc_hits(), n_sequences=20)
    assert more["neither"].to_list() == [12, 13, 14]


def test_coocc_distances():
    hits = pl.DataFrame({"motif_i": [0, 1, 0, 1, 1, 0, 1], "sequence_i": [0, 0, 1, 1, 1, 2, 2],
                         "start": [10, 14, 50, 90, 58, 0, 40]})
    ms = _coocc_motifs()[:2]
    # The nearest pairs are 4, 8 and 40 apart.
    res = um.motif_coocc(ms, hits=hits, n_sequences=3, max_distance=10)
    assert res["both"].to_list() == [3]
    assert res["both_clustered"].to_list() == [2]
    assert res["median_distance"].to_list() == [6.0]
    res = um.motif_coocc(ms, hits=hits, n_sequences=3, max_distance=100)
    assert res["both_clustered"].to_list() == [3] and res["median_distance"].to_list() == [8.0]
    none = um.motif_coocc(ms, hits=hits, n_sequences=3, max_distance=2)
    assert none["both_clustered"].to_list() == [0] and none["median_distance"].to_list() == [None]


def test_coocc_self_pairs_and_names():
    ms = _coocc_motifs()
    one = um.motif_coocc(ms[:1], hits=_coocc_hits().filter(pl.col("motif_i") == 0), n_sequences=10,
                         self_pairs=True)
    # A motif is not tested against itself: its sequences with hits all go
    # in a_only.
    assert one.select("motif_a", "motif_b", "a_only", "b_only", "both", "neither").rows() == [
        ("a", "a", 6, 0, 0, 4),
    ]
    assert one["pvalue"].to_list() == [None]
    # Repeated names are made unique, and hits can use the new names.
    repeated = [um.create_motif("TGACTCA", name="a"), um.create_motif("CACGTGA", name="b"),
                um.create_motif("TTTTTTT", name="b")]
    hits = _coocc_hits().with_columns(
        pl.col("motif_i").replace_strict({0: "a", 1: "b", 2: "b.1"}).alias("motif")
    ).drop("motif_i")
    res = um.motif_coocc(repeated, hits=hits, n_sequences=10)
    assert res.select("motif_a", "motif_b").rows() == [("a", "b"), ("a", "b.1"), ("b", "b.1")]
    assert res["both"].to_list() == [3, 0, 0]
    # A string motif_i column is read as names.
    as_str = hits.rename({"motif": "motif_i"})
    assert um.motif_coocc(repeated, hits=as_str, n_sequences=10).equals(res)


def test_coocc_scans_sequences():
    ms = _coocc_motifs()[:2]
    seqs = [_plant((2, "TGACTCA"), (15, "CACGTGA"))] * 3 + [_plant((2, "TGACTCA"))] * 2 + [_FLANK]
    res = um.motif_coocc(ms, seqs, rc=False, max_distance=20)
    assert res.select("a_only", "b_only", "both", "neither").row(0) == (2, 0, 3, 1)
    assert res["median_distance"].to_list() == [13.0]


def test_coocc_argument_errors():
    ms = _coocc_motifs()
    hits = _coocc_hits()
    cases = [
        ({"hits": hits, "n_sequences": 10, "max_distance": -1}, "max_distance must be"),
        ({"hits": hits, "n_sequences": 10, "min_coocc": -1}, "must be non-negative"),
        ({"hits": hits, "n_sequences": 10, "pseudocount": -1}, "must be non-negative"),
        ({"hits": hits, "n_sequences": 0}, "n_sequences must be a positive integer"),
        ({"hits": hits.drop("sequence_i"), "n_sequences": 10}, "sequence_i column"),
        ({"hits": hits.drop("motif_i"), "n_sequences": 10}, "motif_i or motif column"),
        ({"hits": hits.with_columns(pl.lit(3).alias("motif_i")), "n_sequences": 10},
         "could not be matched"),
        ({"hits": hits.with_columns(pl.Series("motif", ["zz"] * hits.height)).drop("motif_i"),
          "n_sequences": 10}, "could not be matched"),
        ({"hits": hits, "n_sequences": 9}, r"outside 0..n_sequences-1"),
        ({"hits": hits.drop("start"), "n_sequences": 10, "max_distance": 5}, "start column"),
        ({"hits": hits.with_columns(pl.lit(-1).alias("start")), "n_sequences": 10,
          "max_distance": 5}, "non-negative integers"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            um.motif_coocc(ms, **kw)
    with pytest.raises(ValueError, match="need at least 2 motifs"):
        um.motif_coocc(ms[:1], hits=hits, n_sequences=10)
    with pytest.raises(ValueError, match="exactly one of sequences and hits"):
        um.motif_coocc(ms)


def _peak_hits():
    # Twenty sequences of 100, each with its best hit centred at 20 in the
    # sequence's own orientation and a weaker one at 54.
    rows = []
    for s in range(20):
        start = 16 if s % 2 == 0 else 75
        rows += [("a", f"s{s}", start, start + 9, 5.0), ("a", f"s{s}", 50, 59, 1.0)]
    return pl.DataFrame(rows, schema=["motif", "sequence", "start", "end", "score"], orient="row")


def test_peaks_strands_and_best_hits():
    stats = pytest.importorskip("scipy.stats")
    hits = _peak_hits()
    strand = {f"s{s}": "+" if s % 2 == 0 else "-" for s in range(20)}
    plain = um.motif_peaks(hits, 100, mode="local", qvalue=1, max_window=30).row(0, named=True)
    # Only the best hit of each sequence is used, at (start + end - 1) / 2.
    assert plain["centres"] == [20, 79] * 10
    assert plain["nhits"] == 20 and plain["hits_in"] == 10
    flipped = um.motif_peaks(hits, 100, mode="local", seq_strand=strand, qvalue=1,
                             max_window=30).row(0, named=True)
    # Centres on - sequences are read from the other end: 99 - 79 = 20.
    assert flipped["centres"] == [20] * 20
    assert flipped["hits_in"] == 20 and flipped["hits_out"] == 0
    assert abs(flipped["best_centre"] - 20) <= flipped["best_window"] / 2
    assert flipped["expected_in"] == pytest.approx(20 * flipped["best_window"] / 100)
    # Both are Bonferroni-corrected for the same windows.
    w = plain["best_window"] / 100
    k_plain = plain["pvalue"] / stats.binom.sf(9, 20, w)
    k_flipped = flipped["pvalue"] / stats.binom.sf(19, 20, flipped["best_window"] / 100)
    assert k_plain == pytest.approx(k_flipped) and k_plain == pytest.approx(round(k_plain))
    assert flipped["pvalue"] < plain["pvalue"]


def test_peaks_column_variants():
    hits = _peak_hits()
    base = um.motif_peaks(hits, 100, mode="local", qvalue=1, max_window=30)
    # stop for end, sequence_i for sequence.
    alt = hits.rename({"end": "stop"}).with_columns(
        pl.col("sequence").str.slice(1).cast(pl.Int64).alias("sequence_i")).drop("sequence")
    got = um.motif_peaks(alt, 100, mode="local", qvalue=1, max_window=30)
    assert got.equals(base)
    strand = {s: "-" for s in range(20)}
    flipped = um.motif_peaks(alt, 100, mode="local", qvalue=1, max_window=30, seq_strand=strand)
    assert flipped["centres"][0].to_list() == [79, 20] * 10


def test_peaks_motif_numbers_and_empty():
    hits = _peak_hits().with_columns(pl.lit("zeta").alias("motif"))
    both = pl.concat([hits, _peak_hits()])
    # Without motif_i, motifs are numbered by name.
    res = um.motif_peaks(both, 100, qvalue=1).sort("motif")
    assert dict(res.select("motif", "motif_i").iter_rows()) == {"a": 0, "zeta": 1}
    given = both.with_columns(pl.when(pl.col("motif") == "a").then(5).otherwise(2).alias("motif_i"))
    res = um.motif_peaks(given, 100, qvalue=1).sort("motif")
    assert dict(res.select("motif", "motif_i").iter_rows()) == {"a": 5, "zeta": 2}
    assert res["mode"].to_list() == ["central"] * 2 and res["seq_length"].to_list() == [100] * 2
    empty = um.motif_peaks(hits.clear(), 100)
    assert empty.height == 0 and empty.columns[-1] == "centres"


def test_peaks_argument_errors():
    hits = _peak_hits()
    cases = [
        ({"mode": "global"}, "mode must be 'central' or 'local'"),
        ({"qvalue": 0}, r"qvalue must be in \(0, 1\]"),
        ({"seq_strand": {"s0": "+"}}, "seq_strand is missing sequences"),
        ({"seq_strand": {f"s{s}": "." for s in range(20)}}, "must be '\\+' or '-'"),
        ({"min_window": 60, "max_window": 40}, "narrower than min_window"),
        ({"window_step": 0}, "must be at least 1"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            um.motif_peaks(hits, 100, **kw)
    for col, msg in (("motif", "'motif' column"), ("score", "'score' column"),
                     ("start", "'start' column"), ("end", "'end' or 'stop'"),
                     ("sequence", "'sequence' or 'sequence_i'")):
        with pytest.raises(ValueError, match=msg):
            um.motif_peaks(hits.drop(col), 100)


def test_motif_peaks_rejects_hits_outside_the_sequences():
    hits = pl.DataFrame({"motif": ["m", "m"], "sequence": ["s1", "s2"], "start": [10, 103],
                         "end": [16, 109], "score": [1.0, 1.0]})
    with pytest.raises(ValueError, match="outside sequences of length 100"):
        um.motif_peaks(hits, seq_length=100)
    with pytest.raises(ValueError, match="seq_length must be at least 1"):
        um.motif_peaks(hits, seq_length=0)
