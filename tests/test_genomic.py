import warnings

import numpy as np
import polars as pl
import pytest

import universalmotif as um

from conftest import args, load_reference, motif_from_record, nums


def _ref(key):
    return load_reference("proximity.json", key)


def _chroms():
    return dict(zip(_ref("chromosome_names"), _ref("chromosomes")))


def _ranges(t):
    # R counts positions from 1, with the end included.
    return pl.DataFrame({
        "chrom": t["seqnames"],
        "start": [int(x) - 1 for x in nums(t["start"])],
        "end": [int(x) for x in nums(t["end"])],
        "strand": t["strand"] if isinstance(t["strand"], list) else [t["strand"]] * len(t["seqnames"]),
    })


def _hits():
    h = _ref("hits")
    return pl.DataFrame({
        "motif": h["motif"],
        "sequence": h["seqnames"],
        "start": [int(x) - 1 for x in nums(h["start"])],
        "end": [int(x) for x in nums(h["end"])],
        "strand": h["strand"],
        "score": nums(h["score"]),
    })


def _close(got, want, ctx):
    got = np.array([np.nan if v is None else v for v in got], dtype=float)
    np.testing.assert_allclose(got, np.array(nums(want), dtype=float), rtol=1e-12, err_msg=ctx)


def test_proximity_reference():
    chroms = _chroms()
    lengths = {k: len(v) for k, v in chroms.items()}
    whole = pl.DataFrame({"chrom": list(lengths), "start": [0] * 3, "end": list(lengths.values())})
    hits, anchors = _hits(), _ranges(_ref("anchors"))
    for case in _ref("proximity"):
        a = args(case["args"])
        count = a.get("count", "hits")
        if case["universe"]:
            universe = _ranges(_ref("universe"))
        elif count == "anchors":
            # Without lengths zones are not clipped at chromosome ends, which
            # is what the R package ends up doing.
            universe = whole
        else:
            universe = lengths
        got = um.motif_proximity(
            hits,
            anchors,
            universe=universe,
            method=a.get("method", "binomial"),
            count=count,
            orientation=a.get("orientation", "undirected"),
            dist_thresholds=a.get("dist.thresholds"),
            ignore_strand=a.get("ignore.strand", True),
            qvalue=1,
        )
        t = case["table"]
        ctx = (a, case["universe"])
        # Ties in P-value are ordered by motif index here and by name in the
        # R package, so rows are matched by motif.
        assert got["pvalue"].is_sorted(), ctx
        assert sorted(got["motif"].to_list()) == sorted(t["motif"]), ctx
        got = pl.concat([got.filter(pl.col("motif") == m) for m in t["motif"]])
        assert got["nhits"].to_list() == nums(t["nhits"]), ctx
        for col in ("best_dist", "hits_near", "hits_far", "frac_near", "expected_near", "enrichment",
                    "log2_enrichment", "pvalue", "qvalue", "ks_d"):
            _close(got[col].to_list(), t[col.replace("_", ".").replace("ks.d", "ks.D")], f"{ctx} {col}")
        for g, w in zip(got["dist_to_nearest"].to_list(), t["dist.to.nearest"]):
            _close(g, w, f"{ctx} distances")


def test_proximity_scans_internally():
    motifs = [motif_from_record(r) for r in _ref("proximity_motifs")]
    anchors = _ranges(_ref("anchors"))
    scanned = um.motif_proximity(anchors=anchors, motifs=motifs, sequences=_chroms(),
                                 scan_pvalue=1e-3, qvalue=1)
    lengths = {k: len(v) for k, v in _chroms().items()}
    given = um.motif_proximity(_hits(), anchors, universe=lengths, qvalue=1)
    assert scanned.equals(given)
    assert scanned["motif"][0] == "creb" and scanned["pvalue"][0] < 1e-6


def test_proximity_options():
    anchors = _ranges(_ref("anchors"))
    lengths = {k: len(v) for k, v in _chroms().items()}
    perm = um.motif_proximity(_hits(), anchors, universe=lengths, method="permutation", n_perm=99,
                              seed=1, qvalue=1)
    assert perm["pvalue"][0] == pytest.approx(0.01)
    again = um.motif_proximity(_hits(), anchors, universe=lengths, method="permutation", n_perm=99,
                               seed=1, qvalue=1, nthreads=2)
    assert perm.equals(again)
    tuples = [(r["chrom"], r["start"], r["end"], r["strand"]) for r in anchors.iter_rows(named=True)]
    assert um.motif_proximity(_hits(), tuples, universe=lengths, qvalue=1).equals(
        um.motif_proximity(_hits(), anchors, universe=lengths, qvalue=1))
    with pytest.raises(ValueError, match="universe"):
        um.motif_proximity(_hits(), anchors)
    with pytest.raises(ValueError):
        um.motif_proximity(_hits(), anchors, universe=lengths, method="ks", count="anchors")
    with pytest.raises(ValueError, match="stranded"):
        um.motif_proximity(_hits(), anchors.with_columns(pl.lit("*").alias("strand")),
                           universe=lengths, orientation="upstream")
    strict = um.motif_proximity(_hits(), anchors, universe=lengths)
    assert (strict["qvalue"] <= 0.1).all()


def test_match_bkg():
    targets = _ref("match_targets")
    universe = _ref("match_universe")
    named = {f"u{i}": s for i, s in enumerate(universe)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = um.match_bkg(targets, universe, n_per_target=2, seed=1)
        assert isinstance(out, list) and len(out) == 24
        out = um.match_bkg(targets, named, n_per_target=2, seed=1)
        assert isinstance(out, dict) and set(out.values()) <= set(universe)
        idx = um.match_bkg(targets, named, n_per_target=2, seed=1, return_indices=True)
    assert idx.columns[:6] == ["target_i", "target_gc", "target_length", "universe_i", "universe_gc",
                               "universe_length"]
    assert idx["universe_name"].to_list() == [f"u{i}" for i in idx["universe_i"]]
    coarse = um.match_bkg(targets, universe, n_per_target=2, n_bins_gc=4, n_bins_length=3, seed=2,
                          return_indices=True)
    assert coarse["universe_i"].n_unique() == 24
    fine = um.match_bkg(targets, universe, seed=1, return_indices=True)
    assert fine["universe_i"].n_unique() == 12
    gap = (fine["target_gc"] - fine["universe_gc"]).abs().mean()
    gc_t = np.array(um._core._gc_fraction(targets))
    gc_u = np.array(um._core._gc_fraction(universe))
    unmatched = np.abs(gc_t[:, None] - gc_u[None, :]).mean()
    assert gap < 0.6 * unmatched

    cov = um.match_bkg(targets, universe, covariates={"signal": list(range(12))},
                       universe_covariates=pl.DataFrame({"signal": [i % 12 for i in range(80)]}),
                       n_bins_gc=2, n_bins_length=2, n_bins_covariates=12,
                       bin_type_covariates="equalwidth", seed=3, return_indices=True)
    diff = (cov["target_signal"] - cov["universe_signal"]).abs()
    # Exact where the target's cell has a sequence, else a neighbouring bin.
    assert (diff == 0).mean() >= 0.75 and diff.max() <= 3
    with pytest.warns(UserWarning) as record:
        um.match_bkg(targets, universe[:5], seed=1)
    messages = [str(w.message) for w in record]
    assert any("smaller" in m for m in messages) and any("ran out" in m for m in messages)
    with pytest.raises(ValueError):
        um.match_bkg(targets, universe, covariates=[1.0] * 12)
    with pytest.raises(ValueError):
        um.match_bkg(["MKVL"], universe)


def test_match_bkg_genome():
    chroms = _chroms()
    targets = _ref("match_targets")
    exclude = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [6000]})
    out = um.match_bkg(targets, genome=chroms, exclude=exclude, n_candidates=300, seed=4,
                       n_per_target=2)
    assert len(out) == 24
    for name, seq in out.items():
        chrom, span = name.split(".")[0].split(":")
        s, e = map(int, span.split("-"))
        assert chrom != "chr1" and chroms[chrom][s:e] == seq and len(seq) == 150
    only = um.match_bkg(targets[:1], genome=chroms, chromosomes=["chr3"], seed=1, return_indices=True)
    assert only["universe_name"][0].startswith("chr3:") and only["universe_length"][0] == 150


def test_genomic_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = {k: len(v) for k, v in _chroms().items()}
    prox = um.motif_proximity(_hits(), _ranges(_ref("anchors")), universe=lengths, qvalue=1)
    fig = um.plot_motif_proximity(prox)
    assert fig.axes[0].get_title().startswith("creb  (p = ")
    fig = um.plot_motif_proximity(prox, motifs=["creb"], log_x=True)
    assert fig.axes[0].get_xscale() == "log"
    idx = um.match_bkg(_ref("match_targets"), _ref("match_universe"), n_bins_gc=4, n_bins_length=3,
                       seed=1, return_indices=True)
    fig = um.plot_match_bkg(indices=idx)
    assert [a.get_title() for a in fig.axes] == ["GC fraction", "Sequence length"]
    fig = um.plot_match_bkg(_ref("match_targets"), _ref("match_universe")[:12], by=["gc"])
    assert len(fig.axes) == 1
    plt.close("all")


def test_packed_genomes():
    # Packed genomes are scanned and cut into windows without decoding whole
    # chromosomes, with the same results.
    chroms = {k: v.upper() for k, v in _chroms().items()}
    packed = um.pack_sequences(chroms)
    motifs = [motif_from_record(r) for r in _ref("proximity_motifs")]
    anchors = _ranges(_ref("anchors"))
    a = um.motif_proximity(anchors=anchors, motifs=motifs, sequences=chroms, scan_pvalue=1e-3, qvalue=1)
    b = um.motif_proximity(anchors=anchors, motifs=motifs, sequences=packed, scan_pvalue=1e-3, qvalue=1)
    assert a.equals(b)
    targets = _ref("match_targets")
    exclude = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [6000]})
    opts = dict(exclude=exclude, n_candidates=300, seed=4, n_per_target=2)
    assert um.match_bkg(targets, genome=packed, **opts) == um.match_bkg(targets, genome=chroms, **opts)
    assert packed.subseq("chr2", 4, 9) == chroms["chr2"][4:9]
    with pytest.raises(IndexError):
        packed.subseq("chr2", 1, len(chroms["chr2"]) + 1)


# Small synthetic cases, with expected values worked out by hand.

def _hit_table(rows):
    return pl.DataFrame(rows, schema=["motif", "sequence", "start", "end", "strand", "score"],
                        orient="row")


def _prox(hits, anchors, **kw):
    kw.setdefault("universe", {"c1": 1000})
    kw.setdefault("dist_thresholds", [10])
    return um.motif_proximity(hits, anchors, qvalue=1, **kw)


def test_range_table_column_variants():
    from universalmotif._genomic import _range_table

    want = [("c1", 2, 7, "*")]
    for chrom in ("chrom", "seqnames", "sequence", "chr", "chromosome"):
        df = pl.DataFrame({chrom: ["c1"], "start": [2], "end": [7]})
        assert _range_table(df, "r").rows() == want, chrom
    assert _range_table(pl.DataFrame({"chr": ["c1"], "start": [2], "stop": [7]}), "r").rows() == want
    assert _range_table(pl.DataFrame({"chr": ["c1"], "start": [2], "width": [5]}), "r").rows() == want
    # Chromosome names that look like numbers come back as strings.
    num = pl.DataFrame({"chrom": [1], "start": [2], "end": [7]})
    assert _range_table(num, "r")["chrom"].to_list() == ["1"]
    # BED-style and empty strands are all unstranded.
    df = pl.DataFrame({"chrom": ["c1"] * 6, "start": [0] * 6, "end": [1] * 6,
                       "strand": ["+", "-", "*", ".", "", None]})
    assert _range_table(df, "r")["strand"].to_list() == ["+", "-", "*", "*", "*", "*"]
    assert _range_table([("c1", 0, 1, "+-")], "r")["strand"].to_list() == ["*"]
    assert _range_table([], "r").schema == {"chrom": pl.String, "start": pl.Int64, "end": pl.Int64,
                                            "strand": pl.String}


def test_range_table_errors():
    from universalmotif._genomic import _range_table

    with pytest.raises(ValueError, match="peaks needs a chromosome column"):
        _range_table(pl.DataFrame({"contig": ["c1"], "start": [0], "end": [1]}), "peaks")
    with pytest.raises(ValueError, match="peaks needs a chromosome column"):
        _range_table(pl.DataFrame({"chrom": ["c1"], "begin": [0], "end": [1]}), "peaks")
    with pytest.raises(ValueError, match="needs an end, stop or width column"):
        _range_table(pl.DataFrame({"chrom": ["c1"], "start": [0]}), "peaks")
    with pytest.raises(ValueError, match="tuples"):
        _range_table([("c1", 0)], "peaks")
    with pytest.raises(ValueError, match="tuples"):
        _range_table([("c1", 0, 1, "+", "extra")], "peaks")
    with pytest.raises(ValueError, match="missing coordinates"):
        _range_table(pl.DataFrame({"chrom": ["c1"], "start": [None], "end": [1]},
                                  schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64}),
                     "peaks")


def test_range_table_from_pandas_and_old_pyranges():
    pd = pytest.importorskip("pandas")
    # Converting these frames to polars needs pyarrow.
    pytest.importorskip("pyarrow")
    from universalmotif._genomic import _range_table

    # A pandas index is dropped, and categorical chromosomes are fine.
    df = pd.DataFrame({"Chromosome": pd.Categorical(["c1", "c2"]), "Start": [3, 4], "End": [9, 8],
                       "Strand": ["-", "+"]}, index=[10, 20])
    assert _range_table(df, "r").rows() == [("c1", 3, 9, "-"), ("c2", 4, 8, "+")]

    class PyRanges:
        """Stands in for a PyRanges object from before version 1, which
        holds its table in ``df`` rather than being a table itself."""

        def __init__(self, df):
            self.df = df

    genome = {"c1": "AAAACCGTTT", "c2": "TTTTGGGGCC"}
    assert um.extract_ranges(genome, PyRanges(df)) == {"c1:3-9(-)": "AACGGT", "c2:4-8": "GGGG"}


def test_polars_bio_metadata_errors_mean_zero_based(monkeypatch):
    pb = pytest.importorskip("polars_bio")
    from universalmotif._genomic import _ranges_frame

    df = pl.DataFrame({"chrom": ["c1"], "start": [5], "end": [9]})

    def broken(_):
        raise KeyError("no metadata")

    monkeypatch.setattr(pb, "get_metadata", broken)
    assert _ranges_frame(df)["start"].to_list() == [5]


def test_proximity_hit_middles_and_distances():
    # Hits count at their middle, rounded down for even widths, and the
    # distance runs to the nearest base of the anchor. Each hit is its own
    # motif so that none are merged for overlapping.
    spans = {"a": (490, 500), "b": (491, 500), "c": (500, 501), "d": (501, 510), "e": (600, 610),
             "f": (10, 20)}
    hits = _hit_table([(m, "c1", s, e, "+", 1.0) for m, (s, e) in spans.items()])
    res = _prox(hits, [("c1", 500, 501)])
    dist = dict(res.select("motif", "dist_to_nearest").iter_rows())
    assert dist == {"a": [-6], "b": [-5], "c": [0], "d": [5], "e": [104], "f": [-486]}
    near = dict(res.select("motif", "hits_near").iter_rows())
    assert near == {"a": 1, "b": 1, "c": 1, "d": 1, "e": 0, "f": 0}
    # The zone of 10 around a 1 bp anchor holds 21 bases out of 1000.
    row = res.filter(pl.col("motif") == "a").row(0, named=True)
    assert (row["nhits"], row["hits_far"]) == (1, 0)
    assert row["expected_near"] == pytest.approx(21 / 1000)
    assert row["frac_near"] == 1
    assert row["enrichment"] == pytest.approx(1000 / 21)
    assert row["log2_enrichment"] == pytest.approx(np.log2(1000 / 21))


def test_proximity_fractions_over_several_hits():
    hits = _hit_table([("a", "c1", s, s + 10, "+", 1.0) for s in (490, 505, 600, 10)])
    row = _prox(hits, [("c1", 500, 501)]).row(0, named=True)
    assert row["dist_to_nearest"] == [-6, 9, 104, -486]
    assert (row["nhits"], row["hits_near"], row["hits_far"]) == (4, 2, 2)
    assert row["expected_near"] == pytest.approx(4 * 21 / 1000)
    assert row["enrichment"] == pytest.approx(0.5 / (21 / 1000))


def test_proximity_binomial_pvalue_and_bonferroni():
    stats = pytest.importorskip("scipy.stats")
    hits = _hit_table([("a", "c1", 490, 500, "+", 1.0), ("a", "c1", 600, 610, "+", 1.0),
                       ("a", "c1", 10, 20, "+", 1.0)])
    one = _prox(hits, [("c1", 500, 501)], dist_thresholds=[10]).row(0, named=True)
    assert one["best_dist"] == 10 and one["hits_near"] == 1
    assert one["pvalue"] == pytest.approx(stats.binom.sf(0, 3, 21 / 1000))
    # Trying a second distance doubles the best P-value.
    two = _prox(hits, [("c1", 500, 501)], dist_thresholds=[10, 100]).row(0, named=True)
    assert two["best_dist"] == 10
    assert two["pvalue"] == pytest.approx(2 * one["pvalue"])


def test_proximity_orientation_follows_anchor_strand():
    # A middle at 494 is upstream of a + anchor at 500 and downstream of a
    # - anchor there.
    hits = _hit_table([("a", "c1", 490, 500, "+", 1.0)])
    plus, minus = [("c1", 500, 501, "+")], [("c1", 500, 501, "-")]
    near = {}
    for anchors, label in ((plus, "+"), (minus, "-")):
        for o in ("upstream", "downstream", "undirected"):
            res = _prox(hits, anchors, orientation=o)
            assert res["orientation"].to_list() == [o]
            near[label, o] = res["hits_near"][0]
            if o != "undirected":
                # Directed zones leave the anchor out: 10 bases on one side.
                assert res["expected_near"][0] == pytest.approx(10 / 1000)
        assert near[label, "undirected"] == 1
    assert (near["+", "upstream"], near["+", "downstream"]) == (1, 0)
    assert (near["-", "upstream"], near["-", "downstream"]) == (0, 1)
    assert _prox(hits, minus)["dist_to_nearest"][0].to_list() == [6]
    assert _prox(hits, plus)["dist_to_nearest"][0].to_list() == [-6]


def test_proximity_zones_clipped_at_chromosome_ends():
    hits = _hit_table([("a", "c1", 0, 4, "+", 1.0)])
    anchors = [("c1", 0, 1)]
    # Counting hits, the zone is cut to the universe either way: bases 0 to
    # 10 of the chromosome.
    for universe in ({"c1": 1000}, [("c1", 0, 1000)]):
        row = _prox(hits, anchors, universe=universe).row(0, named=True)
        assert row["expected_near"] == pytest.approx(11 / 1000)
        assert row["dist_to_nearest"] == [1]
    # Counting anchors, zones are clipped only when lengths are known.
    clipped = _prox(hits, anchors, count="anchors")["expected_near"][0]
    unclipped = _prox(hits, anchors, count="anchors", universe=[("c1", 0, 1000)])["expected_near"][0]
    assert clipped == pytest.approx(1 - np.exp(-11 / 1000))
    assert unclipped == pytest.approx(1 - np.exp(-21 / 1000))


def test_proximity_counts_anchors():
    hits = _hit_table([("a", "c1", 100, 110, "+", 1.0), ("a", "c1", 102, 112, "+", 1.0),
                       ("a", "c1", 700, 710, "+", 1.0), ("b", "c1", 900, 910, "+", 1.0)])
    anchors = [("c1", 105, 106), ("c1", 500, 501), ("c1", 703, 704)]
    res = _prox(hits, anchors, count="anchors", universe={"c1": 1000}).sort("motif")
    assert res["count"].to_list() == ["anchors", "anchors"]
    # nhits is the number of anchors here, split into those with a hit
    # nearby and those without.
    assert res["nhits"].to_list() == [3, 3]
    assert res["hits_near"].to_list() == [2, 0]
    assert res["hits_far"].to_list() == [1, 3]
    assert res["frac_near"].to_list() == pytest.approx([2 / 3, 0])


def test_proximity_overlapping_hits():
    # Overlapping hits of a motif count once with the best score, across
    # strands unless ignore_strand is false.
    hits = _hit_table([("a", "c1", 495, 505, "+", 1.0), ("a", "c1", 497, 507, "-", 3.0),
                       ("a", "c1", 800, 810, "+", 1.0)])
    merged = _prox(hits, [("c1", 500, 501)])
    assert merged["nhits"].to_list() == [2]
    assert merged["dist_to_nearest"][0].to_list() == [1, 304]
    apart = _prox(hits, [("c1", 500, 501)], ignore_strand=False)
    assert apart["nhits"].to_list() == [3]
    assert apart["dist_to_nearest"][0].to_list() == [-1, 1, 304]


def test_proximity_uses_only_anchored_chromosomes():
    hits = _hit_table([("a", "c1", 10, 20, "+", 1.0), ("a", "c2", 10, 20, "+", 1.0)])
    res = _prox(hits, [("c1", 5, 6)], universe={"c1": 100, "c2": 100000}, dist_thresholds=[5])
    assert res["nhits"].to_list() == [1]
    assert res["expected_near"][0] == pytest.approx(11 / 100)
    # Hits only away from the anchors leave nothing to test.
    assert _prox(hits.filter(pl.col("sequence") == "c2"), [("c1", 5, 6)],
                 universe={"c1": 100, "c2": 100}).height == 0


def test_proximity_motif_numbers():
    hits = _hit_table([("z", "c1", 490, 500, "+", 1.0), ("y", "c1", 100, 110, "+", 1.0)])
    # In order of first appearance without motif_i.
    res = _prox(hits, [("c1", 500, 501)]).sort("motif")
    assert dict(res.select("motif", "motif_i").iter_rows()) == {"y": 1, "z": 0}
    # With motif_i, the numbers given are kept.
    res = _prox(hits.with_columns(pl.Series("motif_i", [7, 3])), [("c1", 500, 501)]).sort("motif")
    assert dict(res.select("motif", "motif_i").iter_rows()) == {"y": 3, "z": 7}
    assert res.columns == ["motif", "motif_i", "method", "count", "orientation", "nhits",
                           "best_dist", "hits_near", "hits_far", "frac_near", "expected_near",
                           "enrichment", "log2_enrichment", "pvalue", "qvalue", "ks_d",
                           "dist_to_nearest"]


def test_proximity_ks():
    anchors = [("c1", a, a + 1) for a in range(1000, 10000, 1000)]
    close = _hit_table([("a", "c1", a + d, a + d + 4, "+", 1.0)
                        for a in range(1000, 10000, 1000) for d in (-20, 10)])
    spread = _hit_table([("a", "c1", s, s + 4, "+", 1.0) for s in range(250, 10000, 550)])
    kw = {"universe": {"c1": 10000}, "method": "ks", "dist_thresholds": None}
    near = _prox(close, anchors, **kw)
    assert near["method"].to_list() == ["ks"] and near["nhits"].to_list() == [18]
    assert 0.8 < near["ks_d"][0] <= 1 and near["pvalue"][0] < 1e-6
    far = _prox(spread, anchors, **kw)
    assert far["ks_d"][0] < near["ks_d"][0] and far["pvalue"][0] > 0.05


def test_proximity_hits_from_pandas():
    pd = pytest.importorskip("pandas")
    # Converting these frames to polars needs pyarrow.
    pytest.importorskip("pyarrow")
    hits = _hit_table([("a", "c1", 490, 500, "+", 1.0), ("a", "c1", 10, 20, "+", 1.0)])
    got = _prox(hits.to_pandas(), [("c1", 500, 501)])
    assert isinstance(hits.to_pandas(), pd.DataFrame)
    assert got.equals(_prox(hits, [("c1", 500, 501)]))


def test_proximity_argument_errors():
    hits = _hit_table([("a", "c1", 10, 20, "+", 1.0)])
    anchors = [("c1", 5, 6)]
    seqs = {"c1": "ACGT" * 25}
    motif = um.create_motif("TGACTCA")
    cases = [
        ({"hits": hits, "anchors": anchors, "method": "exact"}, "method must be"),
        ({"hits": hits, "anchors": anchors, "count": "bases"}, "count must be"),
        ({"hits": hits, "anchors": anchors, "orientation": "left"}, "orientation must be"),
        ({"hits": hits, "anchors": anchors, "qvalue": 0}, r"qvalue must be in \(0, 1\]"),
        ({"hits": hits, "anchors": anchors, "qvalue": 1.5}, r"qvalue must be in \(0, 1\]"),
        ({"hits": hits}, "anchors are required"),
        ({"hits": hits, "anchors": anchors, "motifs": motif, "sequences": seqs}, "either hits"),
        ({"anchors": anchors}, "either hits"),
        ({"anchors": anchors, "motifs": motif}, "needs both motifs and sequences"),
        ({"anchors": anchors, "sequences": seqs}, "needs both motifs and sequences"),
        ({"anchors": anchors, "motifs": motif, "sequences": ["ACGT"]}, "must map chromosome names"),
        ({"anchors": [("c9", 5, 6)], "motifs": motif, "sequences": seqs}, "none of the anchor chromosomes"),
        ({"anchors": [("c9", 5, 6)], "motifs": motif, "sequences": um.pack_sequences(seqs)},
         "none of the anchor chromosomes"),
        ({"hits": hits.drop("score"), "anchors": anchors, "universe": {"c1": 100}}, "'score' column"),
        ({"hits": hits.drop("motif"), "anchors": anchors, "universe": {"c1": 100}}, "'motif' column"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            um.motif_proximity(**kw)


def test_proximity_genome_without_anchored_chromosomes(tmp_path):
    path = tmp_path / "g.fa"
    path.write_text(">c1\nACGTACGTAC\n")
    with pytest.raises(ValueError, match="none of the anchor chromosomes"):
        um.motif_proximity(anchors=[("c9", 1, 2)], motifs=um.create_motif("TGACTCA"),
                           sequences=um.Genome(path))


def test_proximity_empty_dist_thresholds():
    hits = _hit_table([("a", "c1", 10, 20, "+", 1.0)])
    with pytest.raises((ValueError, UserWarning)):
        _prox(hits, [("c1", 5, 6)], dist_thresholds=[])


def _gc_targets():
    return ["ACGTACGTGC" * 3, "ATATATATGC" * 3, "GCGCGCGCAT" * 3]


def test_match_bkg_array_covariates():
    targets = _gc_targets()
    universe = [s for s in targets for _ in range(10)]
    # One column per array column, named covariate1, covariate2 and so on.
    one = um.match_bkg(targets, universe, covariates=[0.0, 1.0, 2.0],
                       universe_covariates=[i // 10 for i in range(30)], n_bins_covariates=3,
                       seed=1, return_indices=True)
    assert (one["target_covariate1"] == one["universe_covariate1"]).all()
    two = um.match_bkg(targets, universe, covariates=np.array([[0, 5], [1, 5], [2, 5]]),
                       universe_covariates=np.array([[i // 10, 5] for i in range(30)]),
                       n_bins_covariates={"covariate1": 3, "covariate2": 2},
                       bin_type_covariates={"covariate1": "equalwidth", "covariate2": "quantile"},
                       seed=1, return_indices=True)
    assert {"target_covariate1", "universe_covariate2"} <= set(two.columns)
    assert (two["target_covariate1"] == two["universe_covariate1"]).all()
    assert (two["universe_i"] // 10 == two["target_i"]).all()


def test_match_bkg_covariate_errors():
    targets = _gc_targets()
    universe = targets * 4
    good_u = {"x": [1.0] * 12}
    cases = [
        ({"covariates": np.zeros((3, 1, 1)), "universe_covariates": np.zeros((12, 1, 1))},
         "1-D or 2-D array"),
        ({"covariates": {}, "universe_covariates": {}}, "at least one column"),
        ({"covariates": {"x": [1.0, 2.0]}, "universe_covariates": good_u},
         r"covariates must have one row per sequence \(3\)"),
        ({"covariates": {"x": [1.0] * 3}, "universe_covariates": {"x": [1.0] * 5}},
         r"universe_covariates must have one row per sequence \(12\)"),
        ({"covariates": {"x": [1.0] * 3}, "universe_covariates": {"y": [1.0] * 12}}, "same columns"),
        ({"covariates": {"gc": [1.0] * 3}, "universe_covariates": {"gc": [1.0] * 12}}, "reserved"),
        ({"covariates": {"x": [1.0] * 3}, "universe_covariates": good_u, "n_bins_covariates": {"y": 2}},
         "n_bins_covariates must name every covariate"),
        ({"covariates": {"x": [1.0] * 3}, "universe_covariates": good_u,
              "bin_type_covariates": {"x": "quantile", "y": "quantile"}},
         "bin_type_covariates must name every covariate"),
        ({"covariates": {"x": [1.0] * 3}}, "go together"),
    ]
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            um.match_bkg(targets, universe, seed=1, **kw)


def test_match_bkg_argument_errors():
    targets = _gc_targets()
    genome = {"c1": "ACGT" * 100}
    with pytest.raises(ValueError, match="no target sequences"):
        um.match_bkg([], targets)
    with pytest.raises(ValueError, match="exactly one of universe and genome"):
        um.match_bkg(targets)
    with pytest.raises(ValueError, match="exactly one of universe and genome"):
        um.match_bkg(targets, targets, genome=genome)
    with pytest.raises(ValueError, match="covariates need an explicit universe"):
        um.match_bkg(targets, genome=genome, covariates=[1.0] * 3)
    with pytest.raises(ValueError, match="chromosomes not in the genome: c9"):
        um.match_bkg(targets, genome=genome, chromosomes=["c1", "c9"])
    with pytest.raises(ValueError, match="the universe is empty"):
        um.match_bkg(targets, [])
    with pytest.raises(ValueError, match="the universe is empty"):
        um.match_bkg(targets, {})


def test_match_bkg_genome_windows():
    rng = np.random.default_rng(3)
    genome = {c: "".join(rng.choice(list("ACGT"), n)) for c, n in {"c1": 3000, "c2": 2000}.items()}
    targets = [genome["c1"][2600:2640], genome["c2"][100:125], genome["c1"][2700:2760]]
    exclude = [("c1", 0, 2500), ("c2", 1000, 2000)]
    idx = um.match_bkg(targets, genome=genome, exclude=exclude, n_candidates=200, n_per_target=3,
                       seed=5, return_indices=True)
    assert idx["target_i"].to_list() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # Windows take the widths of the targets, lie within chromosomes and
    # avoid the excluded ranges.
    assert set(idx["universe_length"]) <= {40, 25, 60}
    for name, length in idx.select("universe_name", "universe_length").iter_rows():
        chrom, span = name.split(":")
        s, e = map(int, span.split("-"))
        assert e - s == length and 0 <= s and e <= len(genome[chrom])
        for c, xs, xe in exclude:
            assert c != chrom or e <= xs or s >= xe, name
    # Each window is drawn once.
    assert idx["universe_i"].n_unique() == idx.height
    # The same seed gives the same windows.
    again = um.match_bkg(targets, genome=genome, exclude=exclude, n_candidates=200, n_per_target=3,
                         seed=5, return_indices=True)
    assert again.equals(idx)


def test_match_bkg_named_repeats():
    # A universe too small for unique draws repeats sequences, and their
    # names get .1, .2 and so on.
    with pytest.warns(UserWarning) as record:
        out = um.match_bkg(["ACGTACGTAC"], {"u": "ACGTACGTAC"}, n_per_target=3, seed=1)
    assert any("smaller than n_per_target" in str(w.message) for w in record)
    assert out == {"u": "ACGTACGTAC", "u.1": "ACGTACGTAC", "u.2": "ACGTACGTAC"}


def test_match_bkg_genome_windows_keep_each_targets_width():
    chroms = {"c1": "ACGT" * 500, "c2": "AATTGGCC" * 250}
    targets = {"a": "ACGTACGTAC" * 4, "b": "GGCCAATT" * 5 + "ACGTACGTAC" * 2}
    res = um.match_bkg(targets, genome=chroms, n_per_target=5, n_candidates=200, seed=1,
                       return_indices=True)
    assert (res["target_length"] == res["universe_length"]).all()
    with pytest.raises(ValueError, match="longer than every chromosome"):
        um.match_bkg({"t": "A" * 3000}, genome=chroms, seed=1)


def test_range_positions_must_be_whole_numbers():
    with pytest.raises(ValueError, match="whole numbers"):
        um.extract_ranges({"c1": "ACGTACGT"}, [("c1", 1.5, 4)])
    frame = pl.DataFrame({"chrom": ["c1"], "start": [1.5], "end": [4.0]})
    with pytest.raises(ValueError, match="whole numbers"):
        um.extract_ranges({"c1": "ACGTACGT"}, frame)
    # Floats holding whole numbers are fine.
    frame = pl.DataFrame({"chrom": ["c1"], "start": [1.0], "end": [4.0]})
    assert um.extract_ranges({"c1": "ACGTACGT"}, frame) == {"c1:1-4": "CGT"}
