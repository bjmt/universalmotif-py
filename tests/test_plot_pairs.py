import itertools

import numpy as np
import polars as pl
import pytest
import universalmotif as um
from universalmotif import _plot_pairs
from universalmotif._plot_pairs import pair_spacings, plot_pair_spacing

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HETERO = ["tandem, first upstream", "tandem, second upstream", "convergent", "divergent"]
HOMO = ["tandem", "convergent", "divergent"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def random_hits(seed, nseq=40, length=200, counts=(6, 5), widths=(8, 6)):
    """A hit table of two motifs placed at random, with motif_i and
    sequence_i as from a scan."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(nseq):
        for i, (name, w) in enumerate(zip(("A", "B"), widths)):
            for _ in range(rng.integers(0, counts[i] + 1)):
                start = int(rng.integers(0, length - w + 1))
                rows.append((name, i, f"seq{s + 1}", s, start, start + w, "+" if rng.random() < 0.6 else "-"))
    return pl.DataFrame(rows, orient="row", schema={
        "motif": pl.String, "motif_i": pl.UInt32, "sequence": pl.String, "sequence_i": pl.UInt32,
        "start": pl.Int64, "end": pl.Int64, "strand": pl.String})


def centre(s, e):
    return (s + e) // 2


def naive_arrangement(a, b, homotypic):
    """The arrangement of two hits, worked out case by case."""
    (sa, ea, ra), (sb, eb, rb) = a, b
    ca, cb = centre(sa, ea), centre(sb, eb)
    if homotypic:
        if cb < ca or (cb == ca and ra == "-" and rb == "+"):
            (ca, ra), (cb, rb) = (cb, rb), (ca, ra)
        if ra == rb:
            return "tandem"
        return "convergent" if ra == "+" else "divergent"
    downstream = cb >= ca if ra == "+" else cb <= ca
    if ra == rb:
        return "tandem, first upstream" if downstream else "tandem, second upstream"
    return "convergent" if downstream else "divergent"


def naive_counts(hits, first, second, max_distance, how, overlaps, split):
    homotypic = second is None
    out = {}
    for _, g in hits.group_by("sequence_i"):
        a = g.filter(pl.col("motif") == first).select("start", "end", "strand").rows()
        if homotypic:
            pairs = itertools.combinations(a, 2)
        else:
            b = g.filter(pl.col("motif") == second).select("start", "end", "strand").rows()
            pairs = itertools.product(a, b)
        for x, y in pairs:
            if not overlaps and x[0] < y[1] and y[0] < x[1]:
                continue
            if how == "centre":
                d = abs(centre(y[0], y[1]) - centre(x[0], x[1]))
            else:
                d = max(y[0] - x[1], x[0] - y[1])
            if d > max_distance:
                continue
            kind = naive_arrangement(x, y, homotypic) if split else "all"
            out[kind, d] = out.get((kind, d), 0) + 1
    return out


def as_counts(res):
    return {(k, d): n for k, d, n in res.select("arrangement", "distance", "observed").rows() if n}


@pytest.mark.parametrize("how", ["centre", "gap"])
@pytest.mark.parametrize("overlaps", [False, True])
@pytest.mark.parametrize("homotypic", [False, True])
def test_observed_counts_match_a_naive_count(how, overlaps, homotypic):
    hits = random_hits(1)
    second = None if homotypic else "B"
    res = pair_spacings(hits, "A", second, max_distance=60, distance=how, overlaps=overlaps)
    assert as_counts(res) == naive_counts(hits, "A", second, 60, how, overlaps, True)
    pooled = pair_spacings(hits, "A", second, max_distance=60, distance=how, overlaps=overlaps,
                           arrangements=False)
    assert as_counts(pooled) == naive_counts(hits, "A", second, 60, how, overlaps, False)


def test_the_table_layout():
    hits = random_hits(2)
    res = pair_spacings(hits, "A", "B", max_distance=40)
    assert res.columns == ["arrangement", "distance", "observed", "expected", "enrichment"]
    assert res.height == 4 * 41
    assert res["arrangement"].unique(maintain_order=True).to_list() == HETERO
    assert res["distance"].min() == 0 and res["distance"].max() == 40
    homo = pair_spacings(hits, "A", max_distance=40)
    assert homo["arrangement"].unique(maintain_order=True).to_list() == HOMO
    gap = pair_spacings(hits, "A", "B", max_distance=40, distance="gap", overlaps=True)
    assert gap["distance"].min() == -8
    zero = res.filter(pl.col("expected") == 0)
    assert zero["enrichment"].null_count() == zero.height


def test_motifs_by_name_index_or_motif():
    hits = random_hits(3)
    by_name = pair_spacings(hits, "A", "B")
    assert by_name.equals(pair_spacings(hits, 0, 1))
    motif = um.create_motif("AAAAAA", name="B")
    assert by_name.equals(pair_spacings(hits, "A", motif))
    assert pair_spacings(hits, "A").equals(pair_spacings(hits, 0, "A"))


def test_chunks_give_the_same_counts(monkeypatch):
    hits = random_hits(4, counts=(12, 12))
    whole = pair_spacings(hits, "A", "B")
    homo = pair_spacings(hits, "A")
    monkeypatch.setattr(_plot_pairs, "_CHUNK", 7)
    assert pair_spacings(hits, "A", "B").equals(whole)
    assert pair_spacings(hits, "A").equals(homo)


def test_sequences_named_without_numbers():
    hits = random_hits(5)
    res = pair_spacings(hits, "A", "B")
    assert pair_spacings(hits.drop("sequence_i"), "A", "B").equals(res)


def naive_positions_null(hits, first, second, max_distance, overlaps):
    """The positions null by brute force: every pooled pair of positions,
    each strand with its share, weighted by the pairs per sequence."""
    homotypic = second is None
    second = first if homotypic else second
    a = hits.filter(pl.col("motif") == first)
    b = hits.filter(pl.col("motif") == second)
    wa, wb = int(np.median(a["end"] - a["start"])), int(np.median(b["end"] - b["start"]))
    pa, pb = (a["strand"] == "+").mean(), (b["strand"] == "+").mean()
    na = a.group_by("sequence_i").len()
    nb = b.group_by("sequence_i").len()
    both = na.join(nb, on="sequence_i")
    weight = float((both["len"] * (both["len_right"] - (1 if homotypic else 0))).sum())
    out = {}
    for x in a["start"].to_list():
        for y in b["start"].to_list():
            if not overlaps and x < y + wb and y < x + wa:
                continue
            d = abs(centre(y, y + wb) - centre(x, x + wa))
            if d > max_distance:
                continue
            for ra, rb in itertools.product("+-", repeat=2):
                p = (pa if ra == "+" else 1 - pa) * (pb if rb == "+" else 1 - pb)
                kind = naive_arrangement((x, x + wa, ra), (y, y + wb, rb), homotypic)
                share = p * weight / (a.height * b.height) / (2 if homotypic else 1)
                out[kind, d] = out.get((kind, d), 0) + share
    return out


@pytest.mark.parametrize("homotypic", [False, True])
def test_positions_null_matches_brute_force(homotypic):
    hits = random_hits(6, nseq=12)
    second = None if homotypic else "B"
    res = pair_spacings(hits, "A", second, max_distance=50)
    want = naive_positions_null(hits, "A", second, 50, False)
    got = {(k, d): e for k, d, e in res.select("arrangement", "distance", "expected").rows()}
    assert set(want) <= set(got)
    for key, e in got.items():
        assert e == pytest.approx(want.get(key, 0.0), abs=1e-9)


def test_uniform_null_matches_enumeration():
    # One hit of each motif in one sequence: the expected counts are the
    # share of all placements of the two at each distance and arrangement.
    hits = pl.DataFrame({"motif": ["A", "B"], "sequence": ["s", "s"], "start": [3, 20], "end": [9, 25],
                         "strand": ["+", "+"]})
    res = pair_spacings(hits, "A", "B", max_distance=30, null="uniform", sequences={"s": 40})
    want = {}
    places = [(x, y) for x in range(40 - 6 + 1) for y in range(40 - 5 + 1)]
    for x, y in places:
        if x < y + 5 and y < x + 6:
            continue
        d = abs(centre(y, y + 5) - centre(x, x + 6))
        if d <= 30:
            kind = naive_arrangement((x, x + 6, "+"), (y, y + 5, "+"), False)
            want[kind, d] = want.get((kind, d), 0) + 1 / len(places)
    for k, d, e in res.select("arrangement", "distance", "expected").rows():
        assert e == pytest.approx(want.get((k, d), 0.0), abs=1e-12)


@pytest.mark.parametrize("null", ["positions", "uniform"])
def test_the_null_fits_hits_placed_at_random(null):
    hits = random_hits(7, nseq=3000, length=150, counts=(3, 3))
    lengths = {f"seq{i + 1}": 150 for i in range(3000)}
    res = pair_spacings(hits, "A", "B", max_distance=60, null=null,
                        sequences=lengths if null == "uniform" else None)
    per = res.group_by("arrangement").agg(pl.col("observed", "expected").sum())
    for obs, exp in per.select("observed", "expected").rows():
        assert abs(obs - exp) < 4 * np.sqrt(exp)
    # Across distances, observed minus expected looks like noise.
    z = (res["observed"] - res["expected"]) / np.sqrt(res["expected"].clip(1e-9))
    assert abs(float(z.filter(res["expected"] > 5).mean())) < 0.3


def plant(seqs, sites, gap, seed):
    starts = np.random.default_rng(seed).integers(0, len(seqs[0]) - 40, len(seqs))
    return [um.insert_sites(s, sites, positions=[int(p), int(p) + gap]) for s, p in zip(seqs, starts)]


@pytest.mark.parametrize(("sites", "kind"), [
    (["TTGACC", "CCAGATAA"], "tandem, second upstream"),
    (["TTATCTGG", "GGTCAA"], "tandem, second upstream"),
    (["CCAGATAA", "TTGACC"], "tandem, first upstream"),
    (["CCAGATAA", "GGTCAA"], "convergent"),
    (["GGTCAA", "CCAGATAA"], "divergent"),
])
def test_planted_arrangements_are_found(sites, kind):
    seqs = plant(um.create_sequences(seqnum=200, seqlen=120, seed=3), sites, 18, 1)
    motifs = [um.example_motifs()[i] for i in (6, 5)]
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-3)
    res = pair_spacings(hits, "GATA", "W-box", max_distance=40)
    top = res.sort("observed", descending=True).row(0, named=True)
    assert top["arrangement"] == kind
    assert top["observed"] > 20 * top["expected"]


@pytest.mark.parametrize(("sites", "kind"), [
    (["CAGATAAT", "CAGATAAT"], "tandem"),
    (["CAGATAAT", "ATTATCTG"], "convergent"),
    (["ATTATCTG", "CAGATAAT"], "divergent"),
])
def test_planted_homotypic_arrangements_are_found(sites, kind):
    seqs = plant(um.create_sequences(seqnum=200, seqlen=120, seed=4), sites, 21, 2)
    hits = um.scan_sequences(um.example_motifs()[6], seqs, pvalue=1e-3)
    res = pair_spacings(hits, "GATA", max_distance=40)
    top = res.sort("observed", descending=True).row(0, named=True)
    assert (top["arrangement"], top["distance"]) == (kind, 21)


def seqlet_table():
    """Hits laid out as from read_seqlets(), with two patterns 40 letters
    apart, centre to centre, in a third of the inputs."""
    rng = np.random.default_rng(8)
    rows = []
    for s in range(600):
        if s % 3 == 0:
            at = int(rng.integers(100, 300))
            rows += [(0, s, at, "+"), (1, s, at + 40, "-")]
        else:
            rows.append((int(rng.integers(0, 2)), s, int(rng.integers(50, 400)), "+-"[int(rng.integers(0, 2))]))
    names = ["pos_patterns.pattern_0", "pos_patterns.pattern_3"]
    return pl.DataFrame({
        "motif": [names[r[0]] for r in rows], "motif_i": [r[0] for r in rows],
        "sequence": [f"seq{r[1] + 1}" for r in rows], "sequence_i": [r[1] for r in rows],
        "start": [r[2] for r in rows], "end": [r[2] + 30 for r in rows], "strand": [r[3] for r in rows],
        "score": rng.random(len(rows)),
    }, schema_overrides={"motif_i": pl.UInt32, "sequence_i": pl.UInt32})


def test_seqlets():
    res = pair_spacings(seqlet_table(), 0, 1, max_distance=150)
    top = res.sort("observed", descending=True).row(0, named=True)
    assert (top["arrangement"], top["distance"], top["observed"]) == ("convergent", 40, 200)


def test_errors():
    hits = random_hits(9)
    with pytest.raises(ValueError, match="no hits of a motif named 'C'"):
        pair_spacings(hits, "C")
    with pytest.raises(ValueError, match="no hits of motif 7"):
        pair_spacings(hits, 7)
    with pytest.raises(ValueError, match="no motif_i column"):
        pair_spacings(hits.drop("motif_i"), 0)
    with pytest.raises(TypeError, match="name or index"):
        pair_spacings(hits, 1.5)
    with pytest.raises(ValueError, match="distance must be"):
        pair_spacings(hits, "A", distance="edge")
    with pytest.raises(ValueError, match="null must be"):
        pair_spacings(hits, "A", null="shuffle")
    with pytest.raises(ValueError, match="max_distance"):
        pair_spacings(hits, "A", max_distance=0)
    with pytest.raises(ValueError, match="give their lengths"):
        pair_spacings(hits, "A", null="uniform")
    with pytest.raises(ValueError, match="only used with null='uniform'"):
        pair_spacings(hits, "A", sequences={"seq1": 200})
    with pytest.raises(ValueError, match="no lengths for: seq"):
        pair_spacings(hits, "A", null="uniform", sequences={"seq1": 200})
    with pytest.raises(ValueError, match="run past the end"):
        pair_spacings(hits, "A", null="uniform", sequences={f"seq{i}": 50 for i in range(1, 41)})
    with pytest.raises(ValueError, match="no 'strand' column"):
        pair_spacings(hits.drop("strand"), "A")
    with pytest.raises(ValueError, match="'sequence' or 'sequence_i'"):
        pair_spacings(hits.drop("sequence", "sequence_i"), "A")
    starless = hits.with_columns(pl.lit("*").alias("strand"))
    with pytest.raises(ValueError, match="arrangements=False"):
        pair_spacings(starless, "A")
    assert pair_spacings(starless, "A", arrangements=False).equals(
        pair_spacings(hits.drop("strand"), "A", arrangements=False))
    far = hits.with_columns(pl.col("start") + pl.col("sequence_i").cast(pl.Int64) * 10_000_000,
                            pl.col("end") + pl.col("sequence_i").cast(pl.Int64) * 10_000_000)
    with pytest.raises(ValueError, match="too many for null='positions'"):
        pair_spacings(far, "A")


def test_plot_draws_a_panel_per_arrangement():
    hits = random_hits(10)
    fig = plot_pair_spacing(hits, "A", "B", max_distance=50, names=["GATA", "NF-E2"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert [ax.texts[0].get_text() for ax in fig.axes] == [
        "GATA→ NF-E2→", "NF-E2→ GATA→", "GATA→ ←NF-E2", "←NF-E2 GATA→"]
    res = pair_spacings(hits, "A", "B", max_distance=50)
    for ax, kind in zip(fig.axes, HETERO):
        heights = [p.get_height() for p in ax.patches]
        assert heights == res.filter(arrangement=kind)["observed"].to_list()
        assert ax.texts[1].get_text().endswith("pairs") or ax.texts[1].get_text() == "1 pair"
    assert fig.axes[-1].get_xlabel() == "Distance between centres (bp)"
    homo = plot_pair_spacing(hits, "A", names="GATA")
    assert [ax.texts[0].get_text() for ax in homo.axes] == ["GATA→ GATA→", "GATA→ ←GATA", "←GATA GATA→"]


def test_plot_helix_and_gap():
    hits = random_hits(11)
    fig = plot_pair_spacing(hits, "A", "B", max_distance=60, helix=21, distance="gap", overlaps=True)
    guides = [ln.get_xdata()[0] for ln in fig.axes[0].lines if ln.get_linestyle() == "--"]
    assert guides == pytest.approx([10.5 * k for k in range(6)])
    assert fig.axes[-1].get_xlabel() == "Gap between hits (bp)"
    assert fig.axes[-1].patches[0].get_center()[0] == pytest.approx(-8)


def test_plot_into_axes():
    hits = random_hits(12)
    _, axes = plt.subplots(1, 2)
    ax = plot_pair_spacing(hits, "A", "B", arrangements=False, ax=axes[0])
    assert isinstance(ax, Axes) and ax is axes[0]
    assert ax.get_ylabel() == "Pairs"
    with pytest.raises(ValueError, match="arrangements=False"):
        plot_pair_spacing(hits, "A", "B", ax=axes[1])


def test_long_names_are_cut():
    hits = random_hits(13).with_columns(
        pl.col("motif").replace({"A": "pos_patterns.ENCSR868FGK_ATAC_counts_pattern_0"}))
    fig = plot_pair_spacing(hits, 0)
    label = fig.axes[0].texts[0].get_text()
    assert label.split()[0] == "…K_ATAC_counts_pattern_0→"
    assert len(label.split()[0]) == 25
    with pytest.raises(ValueError, match="one name"):
        plot_pair_spacing(hits, 0, names=["a", "b"])
    with pytest.raises(ValueError, match="both motifs"):
        plot_pair_spacing(hits, 0, 1, names="a")
