import sys
import warnings

import numpy as np
import polars as pl
import pytest

import universalmotif as um


@pytest.fixture
def ramp(tmp_path):
    """chr1 of 200 bases where base i has the value i + 1, except for the
    last 50, which have no value, and chr2 of 50 bases with value 2."""
    pybigtools = pytest.importorskip("pybigtools")
    path = tmp_path / "ramp.bw"
    bw = pybigtools.open(str(path), "w")
    bw.write({"chr1": 200, "chr2": 50},
             [("chr1", i, i + 1, float(i + 1)) for i in range(150)] + [("chr2", 0, 50, 2.0)])
    return path


def test_windows_follow_the_origin_and_the_strand(ramp):
    sites = pl.DataFrame({"motif": ["a", "a", "b"], "chrom": "chr1", "start": [20, 20, 40],
                          "end": [25, 25, 46], "strand": ["+", "-", "."], "score": [1.0, 2.0, 3.0]})
    sm = um.signal_matrix(ramp, sites, flank=3, bin_size=1)
    assert sm.track == "ramp" and sm.origin == "centre"
    assert sm.sites.columns == sites.columns
    assert sm.positions.tolist() == [-3, -2, -1, 0, 1, 2]
    # Base 22 is the middle of 20-25 on either strand, and the minus strand
    # reads backwards. Of 40-46, base 43 is the second of the middle two.
    assert sm.values[0].tolist() == [20, 21, 22, 23, 24, 25]
    assert sm.values[1].tolist() == [26, 25, 24, 23, 22, 21]
    assert sm.values[2].tolist() == [41, 42, 43, 44, 45, 46]
    start = um.signal_matrix(ramp, sites[:2], flank=2, bin_size=1, origin="start")
    assert start.values.tolist() == [[19, 20, 21, 22], [27, 26, 25, 24]]
    end = um.signal_matrix(ramp, sites[:2], flank=2, bin_size=1, origin="end")
    assert end.values.tolist() == [[24, 25, 26, 27], [22, 21, 20, 19]]
    assert um.signal_matrix(ramp, sites[:1], flank=2, bin_size=1, origin="center").origin == "centre"


def test_bins_and_their_summaries(ramp):
    sites = [("chr1", 100, 100, "+"), ("chr1", 100, 100, "-")]
    sm = um.signal_matrix(ramp, sites, flank=4, bin_size=2)
    assert sm.positions.tolist() == [-3.5, -1.5, 0.5, 2.5]
    assert sm.values[0].tolist() == [97.5, 99.5, 101.5, 103.5]
    assert sm.values[1].tolist() == [103.5, 101.5, 99.5, 97.5]
    assert um.signal_matrix(ramp, sites[:1], flank=4, bin_size=2, stat="max").values[0].tolist() == [98, 100, 102, 104]
    assert um.signal_matrix(ramp, sites[:1], flank=4, bin_size=2, stat="min").values[0].tolist() == [97, 99, 101, 103]
    assert um.signal_matrix(ramp, sites[:1], flank=4, bin_size=4, stat="sum").values[0].tolist() == [394, 410]
    assert sm.values.dtype == np.float32


def test_uncovered_bases_and_chromosome_ends(ramp):
    # 146-154 runs past the last covered base, and 194-202 past the end of
    # chr1.
    covered = um.signal_matrix(ramp, [("chr1", 150, 150)], flank=4, bin_size=2)
    assert covered.values[0].tolist() == [147.5, 149.5, 0, 0]
    left_out = um.signal_matrix(ramp, [("chr1", 150, 150)], flank=4, bin_size=2, uncovered=None)
    assert left_out.values[0][:2].tolist() == [147.5, 149.5] and np.isnan(left_out.values[0][2:]).all()
    edge = um.signal_matrix(ramp, [("chr1", 198, 198), ("chr1", 1, 1)], flank=4, bin_size=2)
    assert edge.values[0][:3].tolist() == [0, 0, 0] and np.isnan(edge.values[0][3])
    assert np.isnan(edge.values[1][0]) and edge.values[1][1:].tolist() == [1, 2.5, 4.5]
    partly = um.signal_matrix(ramp, [("chr1", 1, 1)], flank=4, bin_size=4, stat="sum")
    assert partly.values[0].tolist() == [1, 14]
    with pytest.warns(UserWarning, match="1 chromosome of the sites, such as 'chrX'"):
        missing = um.signal_matrix(ramp, [("chrX", 10, 10), ("chr2", 10, 10)], flank=2, bin_size=1)
    assert missing.values.tolist() == [[0, 0, 0, 0], [2, 2, 2, 2]]


def test_sites_and_sampling(ramp):
    hits = pl.DataFrame({"motif": ["a"] * 6 + ["b"] * 2, "chrom": "chr1", "start": np.arange(8) * 10 + 20,
                         "end": np.arange(8) * 10 + 26, "strand": "+"})
    sm = um.signal_matrix(ramp, hits, flank=2, bin_size=1, max_sites=3, seed=1)
    assert sm.sites["motif"].to_list() == ["a"] * 3 + ["b"] * 2
    # Rows stay in their order and match their sites.
    assert (np.diff(sm.sites["start"].to_numpy()) > 0).all()
    assert (sm.values[:, 2] == sm.sites["start"].to_numpy() + 4).all()
    again = um.signal_matrix(ramp, hits, flank=2, bin_size=1, max_sites=3, seed=1)
    assert again.sites.equals(sm.sites)
    ranges = um.signal_matrix(ramp, hits.drop("motif"), flank=2, bin_size=1, max_sites=4, seed=2)
    assert len(ranges) == 4
    assert um.signal_matrix(ramp, hits, flank=2, bin_size=1, name="atac").track == "atac"
    # Hits not yet on the genome are refused.
    genome = {"chr1": "ACGT" * 50}
    seqs = um.extract_ranges(genome, [("chr1", 10, 60)])
    found = um.scan_sequences(um.create_motif("ACGTACGT"), seqs, pvalue=0.01, rc=False)
    with pytest.raises(ValueError, match="genomic_hits"):
        um.signal_matrix(ramp, found)
    on_genome = um.signal_matrix(ramp, um.genomic_hits(found), flank=2, bin_size=1)
    assert len(on_genome) == found.height and "sequence" in on_genome.sites.columns


def test_signal_matrix_checks(ramp, monkeypatch):
    site = [("chr1", 10, 16)]
    for kw, msg in [({"bin_size": 0}, "bin_size"), ({"flank": 5, "bin_size": 2}, "multiple of bin_size"),
                    ({"flank": 0}, "flank"), ({"origin": "middle"}, "origin"),
                    ({"stat": "median"}, "stat"), ({"max_sites": 0}, "max_sites")]:
        with pytest.raises(ValueError, match=msg):
            um.signal_matrix(ramp, site, **kw)
    with pytest.raises(ValueError, match="start <= end"):
        um.signal_matrix(ramp, [("chr1", 16, 10)])
    monkeypatch.setitem(sys.modules, "pybigtools", None)
    with pytest.raises(ImportError, match=r"universalmotif\[bigwig\]"):
        um.signal_matrix(ramp, site)


def test_profile_and_indexing():
    sites = pl.DataFrame({"motif": ["a", "b", "a"], "chrom": "chr1", "start": [1, 2, 3], "end": [4, 5, 6]})
    values = np.array([[1.0, 2.0], [5.0, 5.0], [3.0, np.nan]])
    sm = um.SignalMatrix(sites, values, "x", 1, 1, "start")
    prof = sm.profile()
    assert prof.columns == ["motif", "position", "mean", "se", "n"]
    assert prof.filter(pl.col("motif") == "a").select("mean", "n").rows() == [(2.0, 2), (2.0, 1)]
    assert prof.filter(pl.col("motif") == "a")["se"][0] == pytest.approx(1.0)
    assert np.isnan(prof.filter(pl.col("motif") == "a")["se"][1])
    pooled = sm.profile(by=None)
    assert pooled.columns == ["position", "mean", "se", "n"] and pooled["mean"][0] == 3.0
    with pytest.raises(ValueError, match="no 'cell' column"):
        sm.profile(by="cell")
    assert um.SignalMatrix(sites.drop("motif"), values, "x", 1, 1, "start").profile().height == 2
    assert um.SignalMatrix(sites[:0], values[:0], "x", 1, 1, "start").profile().height == 0
    assert len(sm["a"]) == 2 and len(sm[["a", "b"]]) == 3
    assert sm[np.array([True, False, True])].values.tolist()[0] == [1.0, 2.0]
    assert sm[[2]].sites["start"].to_list() == [3]
    assert len(sm[[]]) == 0
    assert "3 sites of 2 motifs, x in 2 bins of 1 around the start" in repr(sm)
    with pytest.raises(KeyError, match="'c'"):
        sm["c"]
    with pytest.raises(IndexError):
        sm[np.array([True])]
    with pytest.raises(TypeError):
        sm[[0.5]]


@pytest.fixture
def profile_data():
    rng = np.random.default_rng(3)
    n = 60
    sites = pl.DataFrame({"motif": ["GATA", "AP1"] * (n // 2), "chrom": "chr1",
                          "start": np.arange(n) * 1000, "end": np.arange(n) * 1000 + 6,
                          "score": rng.normal(size=n)})
    peak = np.exp(-np.linspace(-3, 3, 40) ** 2)
    sm = um.SignalMatrix(sites, rng.poisson(6 * peak + 1, (n, 40)).astype(np.float32), "chip", 200, 10, "centre")
    control = um.SignalMatrix(sites[:20], rng.poisson(1, (20, 40)).astype(np.float32), "chip", 200, 10, "centre")
    return sm, control


def test_plot_signal_profile(profile_data):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm, control = profile_data
    fig = um.plot_signal_profile(sm)
    assert [a.get_title() for a in fig.axes[:4:2]] == ["GATA", "AP1"]
    # No axis or colour bar titles unless asked for.
    assert [a.get_ylabel() for a in fig.axes] == [""] * len(fig.axes)
    plt.close(fig)
    # A control split by motif, and one shared by every panel.
    fig = um.plot_signal_profile(sm, control=control, groups=["AP1"], max_rows=10, sort_by="score",
                                 title="ChIP", ylabel="Mean ChIP", heatmap_ylabel="Site, by score",
                                 legend_title="ChIP")
    assert fig.axes[0].get_title() == "ChIP"
    assert [a.get_ylabel() for a in fig.axes] == ["Mean ChIP", "Site, by score", "ChIP"]
    assert fig.axes[1].get_xlabel() == "Position from centre"
    assert fig.axes[1].get_images()[0].get_array().shape == (10, 40)
    # The rows average every site, so the axis runs over all of them.
    assert fig.axes[1].get_ylim() == (30, 0)
    assert any(line.get_label() == "Control" for line in fig.axes[0].get_lines())
    plt.close(fig)
    fig = um.plot_signal_profile(sm, control=control.__class__(control.sites.drop("motif"), control.values,
                                                                 "chip", 200, 10, "centre"), by=None, heatmap=False)
    assert len(fig.axes) == 1
    plt.close(fig)
    # Ratios below 0 get a scale centred on 0.
    ratios = um.SignalMatrix(sm.sites, sm.values - 3, "log2", 200, 10, "centre")
    fig = um.plot_signal_profile(ratios, by=None)
    assert um.plot_signal_profile(ratios, by=None, heatmap=False, xlabel="Distance").axes[0].get_xlabel() == "Distance"
    lo, hi = fig.axes[1].get_images()[0].get_clim()
    assert lo == pytest.approx(-hi)
    plt.close(fig)
    fig, ax = plt.subplots()
    assert um.plot_signal_profile(sm, control=control, ax=ax) is ax
    assert len(ax.get_lines()) >= 2
    plt.close(fig)


def test_plot_signal_profile_checks(profile_data):
    pytest.importorskip("matplotlib")
    sm, control = profile_data
    with pytest.raises(TypeError):
        um.plot_signal_profile(sm.values)
    with pytest.raises(ValueError, match="band"):
        um.plot_signal_profile(sm, band="sd")
    with pytest.raises(ValueError, match="max_rows"):
        um.plot_signal_profile(sm, max_rows=0)
    with pytest.raises(ValueError, match="no sites to draw"):
        um.plot_signal_profile(sm[[]])
    with pytest.raises(ValueError, match="same flank"):
        um.plot_signal_profile(sm, control=um.SignalMatrix(control.sites, control.values, "chip", 200, 10, "start"))
    with pytest.raises(ValueError, match="no 'rank' column"):
        um.plot_signal_profile(sm, sort_by="rank")
    with pytest.raises(ValueError, match="numeric"):
        um.plot_signal_profile(sm, sort_by="motif")
    with pytest.raises(ValueError, match="no sites of 'CTCF'"):
        um.plot_signal_profile(sm, groups="CTCF")
    with pytest.raises(ValueError, match="no 'cell' column"):
        um.plot_signal_profile(sm, by="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import matplotlib.pyplot as plt

        plt.close(um.plot_signal_profile(sm, by=None))


def test_clip():
    sites = pl.DataFrame({"motif": ["a", "b"], "chrom": "chr1", "start": [1, 2], "end": [4, 5]})
    values = np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 100.0]], dtype=np.float32)
    sm = um.SignalMatrix(sites, values, "x", 1, 1, "start")
    capped = sm.clip(upper=10, lower=0.5, quantile=False)
    assert capped.values[0].tolist() == [0.5, 1, 2] and capped.values[1, 2] == 10
    assert np.isnan(capped.values[1, 1]) and capped.values.dtype == np.float32
    # The original is untouched, and the sites and settings carry over.
    assert sm.values[1, 2] == 100 and capped.sites.equals(sites) and capped.track == "x"
    top = sm.clip(upper=0.99)
    assert top.values[1, 2] == pytest.approx(np.quantile([0, 1, 2, 3, 100], 0.99))
    assert top.values[0].tolist() == [0, 1, 2]
    assert sm.clip(upper=None, lower=0.5).values[0, 0] == 2
    np.testing.assert_array_equal(sm.clip(upper=1).values, sm.values)
    empty = um.SignalMatrix(sites, np.full((2, 3), np.nan), "x", 1, 1, "start")
    assert np.isnan(empty.clip(upper=0.99).values).all()
    for kw, msg in [({}, "upper, lower or both"), ({"upper": 1.5}, "quantile from 0 to 1"),
                    ({"upper": float("nan"), "quantile": False}, "NaN"),
                    ({"upper": 0.2, "lower": 0.8}, "lower cannot be above upper")]:
        with pytest.raises(ValueError, match=msg):
            sm.clip(**kw)


def test_heatmap_rows_average_runs_of_sites():
    from universalmotif._signal import _block_means

    x = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0], [7.0, 4.0], [9.0, 6.0]])
    np.testing.assert_allclose(_block_means(x, 2), [[2.0, np.nan], [7.0, 4.0]])
    assert _block_means(x, 5) is x and _block_means(x, None) is x
    assert _block_means(x, 1).tolist() == [[5.0, 4.0]]


def _brute_sites(hits, anchors, flank, bin_size, count="centre", strand="both", value=None):
    """site_matrix() for one motif, one anchor and hit at a time."""
    nbins = 2 * flank // bin_size
    out = np.zeros((len(anchors), nbins))
    for i, (c, s, e, st) in enumerate(anchors):
        minus = st == "-"
        a = (s + e + 1) // 2 if minus else (s + e) // 2
        lo = a - flank
        for row in hits.iter_rows(named=True):
            if row["chrom"] != c:
                continue
            hminus = row["strand"] == "-"
            if strand == "same" and hminus != minus or strand == "opposite" and hminus == minus:
                continue
            w = 1.0 if value is None else row[value]
            if count == "centre":
                bases = [(row["start"] + row["end"] + 1) // 2 - 1 if hminus else (row["start"] + row["end"]) // 2]
            else:
                bases = range(row["start"], row["end"])
            for p in bases:
                k = lo + 2 * flank - 1 - p if minus else p - lo
                if 0 <= k < 2 * flank:
                    out[i, k // bin_size] += w / (bin_size if count == "coverage" else 1)
    return out


def test_site_matrix_matches_counting_by_hand():
    rng = np.random.default_rng(4)
    n = 300
    start = rng.integers(0, 5000, n)
    hits = pl.DataFrame({"chrom": rng.choice(["chr1", "chr2"], n), "start": start,
                         "end": start + rng.integers(4, 12, n), "strand": rng.choice(["+", "-"], n),
                         "score": rng.normal(size=n)})
    anchors = [(c, int(s), int(s) + int(w), st) for c, s, w, st in
               zip(rng.choice(["chr1", "chr2", "chr3"], 25), rng.integers(200, 4800, 25),
                   rng.integers(1, 30, 25), rng.choice(["+", "-", "*"], 25))]
    for kw in [{}, {"strand": "same"}, {"strand": "opposite"}, {"count": "coverage"}, {"value": "score"},
               {"count": "coverage", "strand": "same", "value": "score"}]:
        sm = um.site_matrix(hits, anchors, flank=150, bin_size=10, **kw)
        np.testing.assert_allclose(sm.values, _brute_sites(hits, anchors, 150, 10, **kw), atol=1e-5)
    single = um.site_matrix(hits, anchors, flank=20, bin_size=1)
    np.testing.assert_allclose(single.values, _brute_sites(hits, anchors, 20, 1))


def test_site_matrix_motifs_and_edges():
    hits = pl.DataFrame({"motif": ["b", "a", "b"], "motif_i": [1, 0, 1], "chrom": "chr1",
                         "start": [10, 12, 30], "end": [14, 16, 34], "strand": "+"})
    anchors = pl.DataFrame({"chrom": ["chr1", "chr1"], "start": [12, 50], "end": [13, 51], "peak": ["p1", "p2"]})
    sm = um.site_matrix(hits, anchors, flank=20, bin_size=20)
    assert sm.sites.columns == ["motif", "motif_i", "chrom", "start", "end", "peak"]
    assert sm.sites["motif"].to_list() == ["a", "a", "b", "b"]
    # Windows are [-8, 32) and [30, 70); the hit centred on 32 is in the second only.
    assert sm.values.tolist() == [[0, 1], [0, 0], [0, 1], [1, 0]]
    assert [g for g in sm.profile()["motif"].unique(maintain_order=True)] == ["a", "b"]
    plain = um.site_matrix(hits.drop("motif", "motif_i"), anchors, flank=20, bin_size=20)
    assert plain.sites.columns == anchors.columns and plain.values.tolist() == [[0, 2], [1, 0]]
    # Only bins wholly off the chromosome are NaN.
    edge = um.site_matrix(hits.drop("motif"), [("chr1", 5, 6), ("chr1", 50, 51)], flank=20, bin_size=10,
                          chrom_lengths={"chr1": 60})
    assert np.isnan(edge.values[0, 0]) and edge.values[0, 1:].tolist() == [0, 2, 0]
    assert np.isnan(edge.values[1, 3]) and edge.values[1, :3].tolist() == [1, 0, 0]
    assert um.site_matrix(hits, [("chr9", 10, 11)], flank=20, bin_size=20).values.tolist() == [[0, 0], [0, 0]]
    assert um.site_matrix(hits[:0], anchors, flank=20, bin_size=20).values.tolist() == [[0, 0], [0, 0]]
    with pytest.raises(ValueError, match="chrom_lengths has no length for 'chr1'"):
        um.site_matrix(hits, anchors, flank=20, bin_size=20, chrom_lengths={"chr2": 5})
    for kw, msg in [({"count": "middle"}, "count"), ({"strand": "plus"}, "strand"),
                    ({"value": "score"}, "no 'score' column"), ({"value": "motif"}, "numeric"),
                    ({"flank": 15, "bin_size": 10}, "multiple of bin_size")]:
        with pytest.raises(ValueError, match=msg):
            um.site_matrix(hits, anchors, **kw)
    with pytest.raises(TypeError):
        um.site_matrix([("chr1", 1, 2)], anchors)
    with pytest.raises(ValueError, match="rename it"):
        um.site_matrix(hits, anchors.with_columns(motif=pl.lit("x")))
    scanned = pl.DataFrame({"motif": "a", "sequence": "chr1:0-100", "start": [5], "end": [9], "strand": "+"})
    with pytest.raises(ValueError, match="genomic_hits"):
        um.site_matrix(scanned, anchors)
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    control = um.site_matrix(hits, anchors, flank=20, bin_size=20)
    plt.close(um.plot_signal_profile(sm, control=control))


def test_plot_signal_profile_drop_empty():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    sites = pl.DataFrame({"motif": ["a"] * 6 + ["b"] * 2, "chrom": "chr1", "start": np.arange(8) * 100,
                          "end": np.arange(8) * 100 + 1})
    values = np.zeros((8, 4))
    values[0, 1] = values[1, 2] = 2.0
    values[2, 0] = np.nan
    sm = um.SignalMatrix(sites, values, "x", 20, 10, "centre")
    fig = um.plot_signal_profile(sm, groups=["a"], drop_empty=True)
    image = fig.axes[1].get_images()[0]
    # Only the two rows with signal are drawn, but the mean is over all six.
    assert image.get_array().shape == (2, 4) and fig.axes[1].get_ylim() == (2, 0)
    np.testing.assert_allclose(fig.axes[0].get_lines()[0].get_ydata(), [0, 2 / 6, 2 / 6, 0])
    plt.close(fig)
    kept = um.plot_signal_profile(sm, groups=["a"])
    assert kept.axes[1].get_images()[0].get_array().shape == (6, 4)
    plt.close(kept)
    # A group with no signal at all leaves its heatmap empty.
    fig = um.plot_signal_profile(sm, drop_empty=True)
    assert fig.axes[3].get_images() == []
    plt.close(fig)
