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
        "start": [int(x) for x in h["start"]],
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
            _close(got[col].to_list(), t[col.replace("_", ".").replace("centre", "center")], f"{ctx} {col}")
        assert got["centres"].to_list() == [[int(v) for v in nums(c)] for c in t["centers"]], ctx
    # Hits straight from scan_sequences work too.
    scanned = um.scan_sequences(_motifs()[:1], _ref("peak_sequences"), pvalue=1e-3)
    assert um.motif_peaks(scanned, 300, qvalue=1)["best_centre"].to_list() == [150.0]


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
