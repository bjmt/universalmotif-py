import numpy as np
import polars as pl
import pytest
import universalmotif as um
from universalmotif._hit_contrib import (
    HitContributions,
    hit_contributions,
    plot_hit_contributions,
    plot_hit_profile,
)

_COMP = str.maketrans("ACGT", "TGCA")


def _rc(seq):
    return seq.translate(_COMP)[::-1]


def _planted(n=60, length=80, seed=3):
    """Random sequences with GATAAG planted once in each, on either strand,
    and hypothetical scores rewarding the site's letters with 1 and the
    three letters starting two after its end, read along the site, with
    0.5, whatever they are."""
    rng = np.random.default_rng(seed)
    seqs, strands, starts = [], [], []
    hyp = rng.normal(0, 0.01, (n, 4, length)).astype(np.float32)
    for i in range(n):
        letters = list(rng.choice(list("ACGT"), length))
        pos = int(rng.integers(12, length - 18))
        minus = i % 2 == 1
        site = _rc("GATAAG") if minus else "GATAAG"
        letters[pos:pos + 6] = site
        seq = "".join(letters)
        for k in range(pos, pos + 6):
            hyp[i, "ACGT".index(seq[k]), k] = 1.0
        flank = range(pos - 5, pos - 2) if minus else range(pos + 8, pos + 11)
        for k in flank:
            hyp[i, "ACGT".index(seq[k]), k] = 0.5
        seqs.append(seq)
        strands.append("-" if minus else "+")
        starts.append(pos)
    return seqs, hyp, strands, starts


def _site_hits(seqs, strands, starts, motif="GATA"):
    return pl.DataFrame({
        "motif": [motif] * len(seqs),
        "motif_i": pl.Series([0] * len(seqs), dtype=pl.UInt32),
        "sequence": [f"seq{i + 1}" for i in range(len(seqs))],
        "sequence_i": pl.Series(range(len(seqs)), dtype=pl.UInt32),
        "start": starts,
        "end": [s + 6 for s in starts],
        "strand": strands,
        "score": np.linspace(5, 10, len(seqs)),
    })


def _brute(seq, scores, start, end, strand, flank, span):
    """One hit's observed and hypothetical window, the slow way."""
    rows = []
    for k in range(span):
        p = end - 1 + flank - k if strand == "-" else start - flank + k
        if 0 <= p < len(seq):
            col = scores[:, p]
            col = col[::-1] if strand == "-" else col
            letter = seq[p] if strand == "+" else seq[p].translate(_COMP)
            obs = col["ACGT".index(letter)] if letter in "ACGT" else 0.0
            rows.append((obs, col))
        else:
            rows.append((np.nan, np.full(4, np.nan)))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]).T


def test_planted_sites_read_along_their_strand():
    seqs, hyp, strands, starts = _planted()
    hits = _site_hits(seqs, strands, starts)
    hc = hit_contributions(hits, hyp, seqs, flank=12, hypothetical=True)
    assert isinstance(hc, HitContributions)
    assert hc.observed.shape == (60, 30) and hc.hypothetical.shape == (60, 4, 30)
    assert hc.width == 6 and hc.positions[0] == -12 and hc.positions[-1] == 17
    np.testing.assert_allclose(hc.hits["contribution"].to_numpy(), 6.0, atol=0.1)
    mean = np.nanmean(hc.observed, axis=0)
    # The site, then the letters two to four after it, whichever strand.
    np.testing.assert_allclose(mean[12:18], 1.0, atol=0.01)
    np.testing.assert_allclose(mean[20:23], 0.5, atol=0.01)
    assert np.abs(mean[:12]).max() < 0.01 and np.abs(mean[23:]).max() < 0.01
    plus, minus = hc["GATA"][hits["strand"] == "+"], hc["GATA"][hits["strand"] == "-"]
    np.testing.assert_allclose(np.nanmean(plus.observed, 0), np.nanmean(minus.observed, 0), atol=0.01)
    # The hypothetical scores of the letters of GATAAG, read along the site.
    idx = np.array(["ACGT".index(c) for c in "GATAAG"])
    core = minus.hypothetical[:, :, 12:18]
    np.testing.assert_allclose(core[:, idx, np.arange(6)], 1.0)


def test_windows_match_a_slow_reading():
    rng = np.random.default_rng(11)
    n, length, h = 7, 40, 300
    seqs = ["".join(rng.choice(list("ACGTN"), length, p=[0.24] * 4 + [0.04])) for _ in range(n)]
    scores = rng.normal(0, 1, (n, 4, length)).astype(np.float32)
    width = np.where(np.arange(h) % 3 == 0, 5, 8)
    start = rng.integers(0, length - 7, h)
    hits = pl.DataFrame({"motif": np.where(width == 5, "short", "long"), "sequence_i": rng.integers(0, n, h),
                         "start": start, "end": start + width, "strand": rng.choice(["+", "-"], h)})
    flank = 6
    hc = hit_contributions(hits, scores, seqs, flank=flank, hypothetical=True)
    assert hc.observed.shape == (h, 8 + 2 * flank)
    for i, row in enumerate(hits.iter_rows(named=True)):
        span = (5 if row["motif"] == "short" else 8) + 2 * flank
        obs, every = _brute(seqs[row["sequence_i"]], scores[row["sequence_i"]], row["start"], row["end"],
                            row["strand"], flank, span)
        np.testing.assert_allclose(hc.observed[i, :span], obs, rtol=1e-6)
        np.testing.assert_allclose(hc.hypothetical[i, :, :span], every, rtol=1e-6)
        assert np.isnan(hc.observed[i, span:]).all()
        assert hc.hits["contribution"][i] == pytest.approx(obs[flank:flank + row["end"] - row["start"]].sum(),
                                                           rel=1e-5, abs=1e-5)
    short = hc["short"]
    assert short.observed.shape == (len(short), 5 + 2 * flank) and short.width == 5
    assert set(short.hits["motif"]) == {"short"}


def test_input_forms_agree():
    seqs, hyp, strands, starts = _planted(n=12)
    hits = _site_hits(seqs, strands, starts)
    onehot = um.one_hot(seqs)
    projected = hyp * onehot
    names = [f"seq{i + 1}" for i in range(len(seqs))]
    ref = hit_contributions(hits, hyp, seqs, flank=5)
    forms = [
        hit_contributions(hits, um.ContributionScores(hyp, seqs, names, None), flank=5),
        hit_contributions(hits, hyp, onehot, flank=5),
        hit_contributions(hits, projected, flank=5),
        hit_contributions(hits, projected.sum(axis=1), flank=5),
        hit_contributions(hits, list(projected.sum(axis=1)), flank=5),
        hit_contributions(hits, list(hyp.transpose(0, 2, 1)), seqs, flank=5),
        hit_contributions(hits, dict(zip(names, hyp)), dict(zip(names, seqs)), flank=5),
    ]
    for hc in forms:
        np.testing.assert_allclose(hc.observed, ref.observed, rtol=1e-6)
        np.testing.assert_allclose(hc.hits["contribution"], ref.hits["contribution"], rtol=1e-6)


def test_ragged_sequences_and_ends():
    scores = {"a": np.arange(10, dtype=np.float32), "b": np.arange(4, dtype=np.float32) + 100}
    hits = pl.DataFrame({"sequence": ["b", "a", "a"], "start": [1, 0, 7], "end": [3, 2, 9],
                         "strand": ["+", "+", "-"]})
    hc = hit_contributions(hits, scores, flank=2)
    expected = [[np.nan, 100, 101, 102, 103, np.nan], [np.nan, np.nan, 0, 1, 2, 3],
                [np.nan, 9, 8, 7, 6, 5]]
    np.testing.assert_array_equal(hc.observed, np.array(expected, dtype=np.float32))
    assert hc.hits["contribution"].to_list() == [203.0, 1.0, 15.0]


def test_genome_positions():
    seqs, hyp, strands, starts = _planted(n=4)
    ranges = pl.DataFrame({"chrom": ["1", "1", "2", "2"], "start": [1000, 5000, 200, 900],
                           "end": [1080, 5080, 280, 980]})
    names = [f"{c}:{s}-{e}" for c, s, e in ranges.iter_rows()]
    local = _site_hits(seqs, strands, starts)
    genomic = local.with_columns(chrom=ranges["chrom"], start=pl.col("start") + ranges["start"],
                                 end=pl.col("end") + ranges["start"], sequence=pl.lit("peak"))
    ref = hit_contributions(local, hyp, seqs, flank=4)
    contrib = um.ContributionScores(hyp, seqs, names, ranges)
    hc = hit_contributions(genomic, contrib, flank=4)
    np.testing.assert_allclose(hc.observed, ref.observed)
    hc = hit_contributions(genomic, hyp, seqs, flank=4, ranges=ranges)
    np.testing.assert_allclose(hc.observed, ref.observed)
    with pytest.raises(ValueError, match="genomic ranges"):
        hit_contributions(genomic, hyp, seqs, flank=4)
    with pytest.raises(ValueError, match="another chromosome"):
        hit_contributions(genomic.with_columns(chrom=pl.lit("3")), contrib)


def test_errors():
    seqs, hyp, strands, starts = _planted(n=4)
    hits = _site_hits(seqs, strands, starts)
    with pytest.raises(ValueError, match="give the sequences"):
        hit_contributions(hits, hyp)
    with pytest.raises(ValueError, match="hypothetical needs"):
        hit_contributions(hits, (hyp * um.one_hot(seqs)).sum(axis=1), hypothetical=True)
    with pytest.raises(ValueError, match="only 4"):
        hit_contributions(hits.with_columns(sequence_i=pl.lit(9, dtype=pl.UInt32)).drop("sequence"), hyp, seqs)
    with pytest.raises(ValueError, match="no contributions for the sequences"):
        hit_contributions(hits, {"x": hyp[0]}, {"x": seqs[0]})
    with pytest.raises(ValueError, match="flank"):
        hit_contributions(hits, hyp, seqs, flank=-1)
    with pytest.raises(ValueError, match="start <= end"):
        hit_contributions(hits.with_columns(end=pl.col("start") - 1), hyp, seqs)
    with pytest.raises(ValueError, match="hold their sequences"):
        hit_contributions(hits, um.ContributionScores(hyp, seqs, ["a"] * 4, None), seqs)
    hc = hit_contributions(hits, hyp, seqs)
    with pytest.raises(KeyError):
        hc["CTCF"]
    with pytest.raises(IndexError):
        hc[np.array([True, False])]


def test_empty_and_selection():
    seqs, hyp, strands, starts = _planted(n=6)
    hits = _site_hits(seqs, strands, starts)
    hc = hit_contributions(hits.head(0), hyp, seqs, flank=3)
    assert len(hc) == 0 and hc.observed.shape == (0, 6)
    hc = hit_contributions(hits, hyp, seqs, flank=3)
    assert repr(hc) == "HitContributions(6 hits of 1 motif, 6 wide with flanks of 3)"
    picked = hc[[4, 1]]
    assert picked.hits["sequence_i"].to_list() == [4, 1]
    np.testing.assert_array_equal(picked.observed, hc.observed[[4, 1]])
    assert len(hc[hc.hits["strand"] == "+"]) == 3


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _two_motifs():
    seqs, hyp, strands, starts = _planted(n=40)
    hits = _site_hits(seqs, strands, starts)
    other = _site_hits(seqs, strands, [s + 30 if s < 40 else s - 30 for s in starts], motif="other")
    other = other.with_columns(motif_i=pl.lit(1, dtype=pl.UInt32), end=pl.col("start") + 9)
    return hit_contributions(pl.concat([hits, other]), hyp, seqs, flank=10, hypothetical=True)


def test_plot_hit_profile_panels():
    hc = _two_motifs()
    fig = plot_hit_profile(hc["GATA"], max_hits=25)
    labels = [a.get_ylabel() for a in fig.axes]
    assert labels == ["Hypothetical", "Mean contribution", "Hit, by contribution", "Contribution"]
    heat = fig.axes[2]
    assert heat.get_xlabel() == "Position from hit start"
    assert heat.images[0].get_array().shape == (25, 26)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "40 hits, 25 drawn" in texts
    # Sorted by contribution from the largest down.
    rows = heat.images[0].get_array()
    assert np.nansum(rows[0, 10:16]) >= np.nansum(rows[-1, 10:16])

    fig = plot_hit_profile(hc, heatmap=False, origin="centre", band="sd")
    assert [a.get_title(loc="left") for a in fig.axes[::2]] == ["GATA", "other"]
    assert fig.axes[1].get_ylim() == fig.axes[3].get_ylim()
    ticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert "0" in ticks and fig.axes[1].get_xlabel() == "Position from hit centre"


def test_plot_hit_profile_into_axes():
    hc = _two_motifs()
    _, ax = plt.subplots()
    assert plot_hit_profile(hc, ax=ax, band=None, colours={"GATA": "red", "other": "black"}) is ax
    assert [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")] == ["GATA", "other"]
    with pytest.raises(ValueError, match="hypothetical"):
        plot_hit_profile(hit_contributions(hc.hits.drop("contribution"), _planted(n=40)[1],
                                           _planted(n=40)[0]), logo=True)
    with pytest.raises(ValueError, match="band"):
        plot_hit_profile(hc, band="iqr")
    with pytest.raises(ValueError, match="no hits of"):
        plot_hit_profile(hc, motifs="CTCF")
    with pytest.raises(TypeError):
        plot_hit_profile(hc.hits)


def test_plot_hit_contributions():
    hc = _two_motifs()
    fig = plot_hit_contributions(hc)
    main = fig.axes[0]
    assert (main.get_xlabel(), main.get_ylabel()) == ("Motif score", "Total contribution")
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts[0].startswith("GATA, ρ = ") and texts[1].startswith("other, ρ = ")
    assert len(fig.axes) == 3

    hits = hc.hits.with_columns(pvalue=10 ** -pl.col("score"))
    fig = plot_hit_contributions(hits, x="pvalue", marginals=False, correlation=False, motifs="GATA",
                                 trend=True)
    assert fig.axes[0].get_xlabel() == "-log10(P-value)"
    x = fig.axes[0].collections[0].get_offsets()[:, 0]
    np.testing.assert_allclose(np.sort(x), np.sort(hits.filter(pl.col("motif") == "GATA")["score"]))

    fig, ax = plt.subplots()
    assert plot_hit_contributions(hc, hexbin=True, ax=ax) is ax
    assert len(fig.axes) == 1
    with pytest.raises(ValueError, match="no 'contribution' column"):
        plot_hit_contributions(hc.hits.drop("contribution"))
    with pytest.raises(ValueError, match="colours has no colours"):
        plot_hit_contributions(hc, colours={"GATA": "red"})


def test_many_hits_are_quick():
    rng = np.random.default_rng(0)
    n, length, h = 200, 1000, 100_000
    obs = rng.normal(0, 1, (n, length)).astype(np.float32)
    start = rng.integers(0, length - 10, h)
    hits = pl.DataFrame({"motif": "m", "sequence_i": rng.integers(0, n, h), "start": start,
                         "end": start + 10, "strand": rng.choice(["+", "-"], h)})
    hc = hit_contributions(hits, obs, flank=40)
    assert hc.observed.shape == (h, 90) and hc.observed.dtype == np.float32
    i = 12345
    s, e = int(hits["start"][i]), int(hits["end"][i])
    total = obs[int(hits["sequence_i"][i]), s:e].sum()
    assert hc.hits["contribution"][i] == pytest.approx(float(total), rel=1e-5)
