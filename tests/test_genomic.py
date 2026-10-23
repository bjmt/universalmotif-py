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
