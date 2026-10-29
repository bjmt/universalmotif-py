import io

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text
from universalmotif._deep import _site_letters
from universalmotif._plot_variants import plot_variant_effects, variant_motif_effects

START = 5_248_100


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rc(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


@pytest.fixture(scope="module")
def data():
    """A 300 bp sequence with planted AP-1, E-box and GATA sites, a stand-in
    model scoring it by its motif matches, and every substitution's effect."""
    rng = np.random.default_rng(7)
    motifs = [um.example_motifs()[i] for i in (1, 4, 6)]
    seq = list(rng.choice(list("ACGT"), 300))
    sites = [_site_letters(m) for m in motifs]
    for site, at in ((sites[0], 60), (sites[1], 140), (sites[2], 200), (_rc(sites[2]), 228)):
        seq[at:at + len(site)] = site
    seq = "".join(seq)
    filters = um.to_filters(motifs, rc=True)

    def model(batch):
        x = np.asarray(batch, dtype=float)
        win = np.lib.stride_tricks.sliding_window_view(x, filters.shape[2], axis=2)
        s = np.einsum("nalw,faw->nfl", win, filters)
        return np.clip(s - 6, 0, None).sum(axis=(1, 2)) / 10

    ism = um.mutagenesis_effects(model(um.mutagenesis(seq)), seq)
    pos = np.unique(np.concatenate([rng.choice(300, 40, replace=False), [62, 64, 143, 203, 205]]))
    alts = [str(rng.choice([c for c in "ACGT" if c != seq[p]])) for p in pos]
    effect = np.array([ism["ACGT".index(a), p] for p, a in zip(pos, alts)])
    vcf = pl.DataFrame({
        "position": pos + START, "ref": [seq[p] for p in pos], "alt": alts, "effect": effect,
        "af": rng.uniform(0.01, 0.5, len(pos)), "id": [f"rs{1000 + i}" for i in range(len(pos))],
    })
    return {"seq": seq, "motifs": motifs, "ism": ism, "vcf": vcf}


def _heads(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _n_heads(ax):
    return sum(len(c.get_offsets()) for c in _heads(ax))


def _labels(ax):
    return [t.get_text() for t in ax.texts]


# variant_motif_effects


def test_motif_effects_match_scans_of_both_sequences(data):
    seq, motifs = data["seq"], data["motifs"]
    pvalue = 1e-4
    table = variant_motif_effects(None, seq, motifs, pvalue=pvalue)
    ref_hits = um.scan_sequences(motifs, seq, pvalue=pvalue)
    every = table.select("position", "ref", "alt").unique()
    checked = 0
    for p, r, a in list(every.iter_rows())[:: max(1, every.height // 25)] + [(62, seq[62], "A")]:
        if r == a:
            continue
        alt_seq = seq[:p] + a + seq[p + 1:]
        alt_hits = um.scan_sequences(motifs, alt_seq, pvalue=pvalue).filter(
            (pl.col("start") <= p) & (pl.col("end") > p))
        here_ref = ref_hits.filter((pl.col("start") <= p) & (pl.col("end") > p))
        rows = table.filter((pl.col("position") == p) & (pl.col("alt") == a))
        want = {(m, s, t) for m, s, t in here_ref.select("motif", "start", "strand").iter_rows()}
        want |= {(m, s, t) for m, s, t in alt_hits.select("motif", "start", "strand").iter_rows()}
        assert set(rows.select("motif", "start", "strand").iter_rows()) == want
        for m, s, t, score in alt_hits.select("motif", "start", "strand", "score").iter_rows():
            got = rows.filter((pl.col("motif") == m) & (pl.col("start") == s) & (pl.col("strand") == t))
            assert got["alt_score"][0] == pytest.approx(score, abs=1e-9)
        for m, s, t, score in here_ref.select("motif", "start", "strand", "score").iter_rows():
            got = rows.filter((pl.col("motif") == m) & (pl.col("start") == s) & (pl.col("strand") == t))
            assert got["ref_score"][0] == pytest.approx(score, abs=1e-9)
        checked += 1
    assert checked > 10


def test_motif_effects_columns_and_pvalues(data):
    seq, motifs = data["seq"], data["motifs"]
    table = variant_motif_effects(None, seq, motifs)
    assert table.columns == ["variant", "position", "ref", "alt", "motif", "start", "end", "strand",
                             "ref_score", "alt_score", "delta", "ref_pvalue", "alt_pvalue"]
    assert set(table["motif"]) == {m.name for m in motifs}
    np.testing.assert_allclose(table["delta"], (table["alt_score"] - table["ref_score"]).round(3))
    assert ((table["ref_pvalue"] <= 1e-4) | (table["alt_pvalue"] <= 1e-4)).all()
    widths = {m.name: m.width for m in motifs}
    assert (table["end"] - table["start"]).to_list() == [widths[m] for m in table["motif"]]
    assert ((table["start"] <= table["position"]) & (table["end"] > table["position"])).all()
    gata = motifs[2]
    one = table.filter((pl.col("motif") == "GATA") & (pl.col("ref_pvalue") <= 1e-4)).row(0, named=True)
    assert one["ref_pvalue"] == pytest.approx(um.motif_pvalue(gata, score=one["ref_score"]))


def test_motif_effects_find_gained_sites(data):
    seq, motifs = data["seq"], data["motifs"]
    table = variant_motif_effects(None, seq, motifs)
    gained = table.filter(pl.col("ref_pvalue") > 1e-4)
    assert gained.height > 0
    assert (gained["delta"] > 0).all()


def test_motif_effects_with_start_and_hits(data):
    seq, motifs = data["seq"], data["motifs"]
    plain = variant_motif_effects(data["vcf"].with_columns(pl.col("position") - START), seq, motifs)
    shifted = variant_motif_effects(data["vcf"], seq, motifs, start=START)
    assert shifted["position"].to_list() == [p + START for p in plain["position"]]
    assert shifted["start"].to_list() == [p + START for p in plain["start"]]
    assert shifted.drop("position", "start", "end").equals(plain.drop("position", "start", "end"))
    hits = um.scan_sequences(motifs, seq).filter(pl.col("motif") == "AP-1")
    only = variant_motif_effects(data["vcf"], seq, motifs, start=START, hits=hits)
    assert set(only["motif"]) == {"AP-1"}
    assert set(only.select("start", "strand").iter_rows()) <= {
        (s + START, t) for s, t in hits.select("start", "strand").iter_rows()}


def test_motif_effects_variant_rows_and_inputs(data):
    seq, gata = data["seq"], data["motifs"][2]
    rows = [(10, seq[10], "A" if seq[10] != "A" else "C"), (203, seq[203], "C"), (204, seq[204:206], "G"),
            (205, seq[205], "T")]
    rows = [r for r in rows if r[1] != r[2]]
    table = variant_motif_effects(rows, seq, gata)
    # The deletion is left out, and rows keep their numbers in the input.
    assert 2 not in table["variant"].to_list()
    assert set(table["variant"].to_list()) <= {1, 3}
    assert variant_motif_effects(rows, seq.lower(), gata).equals(table)
    assert variant_motif_effects(rows, seq, gata, rc=False)["strand"].unique().to_list() in ([], ["+"])
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(rows, columns=["POSITION", "REF", "ALT"])
    assert variant_motif_effects(frame, seq, gata).equals(table)


def test_motif_effects_empty_and_errors(data):
    seq, gata = data["seq"], data["motifs"][2]
    empty = variant_motif_effects([(1, seq[1], "N" if seq[1] != "N" else "A")], seq, gata)
    assert empty.height == 0 and "delta" in empty.columns
    with pytest.raises(ValueError, match="do not match the sequence"):
        variant_motif_effects([(5, "N", "A")], seq, gata)
    with pytest.raises(ValueError, match="outside the sequence"):
        variant_motif_effects([(300, "A", "C")], seq, gata)
    with pytest.raises(ValueError, match="pvalue"):
        variant_motif_effects(None, seq, gata, pvalue=0)
    with pytest.raises(ValueError, match="'alt' column"):
        variant_motif_effects(pl.DataFrame({"position": [1], "ref": [seq[1]]}), seq, gata)
    with pytest.raises(TypeError):
        variant_motif_effects(None, ["ACGT"], gata)


# plot_variant_effects


def test_every_substitution_from_an_array(data):
    seq, ism = data["seq"], data["ism"]
    fig = plot_variant_effects(ism, sequence=seq)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert _n_heads(ax) == 3 * len(seq)
    stems = next(c for c in ax.collections if isinstance(c, LineCollection))
    assert len(stems.get_segments()) == 3 * len(seq)
    assert ax.get_xlim() == (0, 300)
    # Transposed arrays are read the same way.
    assert _n_heads(plot_variant_effects(ism.T, sequence=seq).axes[0]) == 3 * len(seq)


@pytest.mark.parametrize("summary", ["mean", "min", "max", "strongest"])
def test_summaries_draw_one_variant_per_position(data, summary):
    fig = plot_variant_effects(data["ism"], sequence=data["seq"], summary=summary)
    ax = fig.axes[0]
    assert _n_heads(ax) == 300
    ys = np.concatenate([c.get_offsets()[:, 1] for c in _heads(ax)])
    ism = data["ism"]
    seq = data["seq"]
    sub = np.array([[ism[k, j] for k in range(4) if "ACGT"[k] != seq[j]] for j in range(300)])
    want = {"mean": sub.mean(1), "min": sub.min(1), "max": sub.max(1),
            "strongest": sub[np.arange(300), np.abs(sub).argmax(1)]}[summary]
    np.testing.assert_allclose(np.sort(ys), np.sort(want))


def test_mean_summary_with_motifs(data):
    seq, motifs = data["seq"], data["motifs"]
    fig = plot_variant_effects(data["ism"], sequence=seq, motifs=motifs, summary="mean", labels=3)
    ax = fig.axes[0]
    assert _n_heads(ax) == 300
    # Means have no alternative letter to score motifs with, so only rings
    # mark the variants in hits.
    titles = [a.get_title().get_text() for a in ax.artists if hasattr(a, "get_title")]
    assert "Motif sites" in titles
    assert all(">" not in t for t in _labels(ax))


def test_colour_by_sign_and_alt(data):
    vcf = data["vcf"]
    ax = plot_variant_effects(vcf, start=START, labels=0).axes[0]
    stems = next(c for c in ax.collections if isinstance(c, LineCollection))
    colours = {to_hex(c) for c in stems.get_colors()}
    assert colours <= {"#c0504d", "#4f81bd", "#9e9e9e"}
    ax = plot_variant_effects(vcf, start=START, colour_by="alt", labels=0).axes[0]
    stems = next(c for c in ax.collections if isinstance(c, LineCollection))
    letters = {"#109648", "#255c99", "#f7b32b", "#d62839", "#9e9e9e"}
    assert {to_hex(c) for c in stems.get_colors()} <= letters
    legend_titles = [a.get_title().get_text() for a in ax.artists if hasattr(a, "get_title")]
    assert "Alternative" in legend_titles
    with pytest.raises(ValueError, match="alternative letter"):
        plot_variant_effects(data["ism"], sequence=data["seq"], summary="mean", colour_by="alt")


def test_hits_track_and_ticks_honour_start(data):
    seq, motifs, vcf = data["seq"], data["motifs"], data["vcf"]
    fig = plot_variant_effects(vcf, sequence=seq, motifs=motifs, start=START)
    eff, hit = fig.axes[0], fig.axes[-1]
    assert hit.get_ylabel() == "Hits"
    assert hit.get_xlim() == (START, START + 300)
    ticks = [t.get_text() for t in hit.get_xticklabels()]
    assert "5,248,200" in ticks
    centre = hit.get_xticks()[ticks.index("5,248,200")]
    assert centre == START + 100.5
    # Every hit found by a scan is drawn, at its position on the axis.
    from matplotlib.collections import PolyCollection

    hits = um.scan_sequences(motifs, seq)
    arrows = next(c for c in hit.collections if isinstance(c, PolyCollection))
    assert len(arrows.get_paths()) == hits.height
    lefts = sorted(round(p.vertices[:, 0].min()) for p in arrows.get_paths())
    assert lefts[0] >= START + int(hits["start"].min()) - 1
    titles = [a.get_title().get_text() for a in eff.artists if hasattr(a, "get_title")]
    assert "Motif sites" in titles


def test_direction_and_rings(data):
    seq, motifs = data["seq"], data["motifs"]
    ax = plot_variant_effects(data["ism"], sequence=seq, motifs=motifs, labels=0).axes[0]
    table = variant_motif_effects(None, seq, motifs)
    best = (table.with_columns(pl.col("delta").abs().alias("size"))
            .sort("variant", "size", descending=[False, True]).unique("variant", keep="first"))
    ups = int((best["delta"] > 0).sum())
    downs = int((best["delta"] < 0).sum())
    from matplotlib.markers import MarkerStyle

    shapes = {kind: MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform()).vertices
              for kind, m in (("up", "^"), ("down", "v"))}
    by_marker = {}
    for c in _heads(ax):
        verts = c.get_paths()[0].vertices
        kind = next((k for k, v in shapes.items() if v.shape == verts.shape and np.allclose(v, verts)),
                    "circle")
        by_marker[kind] = by_marker.get(kind, 0) + len(c.get_offsets())
    assert by_marker["up"] == ups
    assert by_marker["down"] == downs
    assert sum(by_marker.values()) == 900
    hits = um.scan_sequences(motifs, seq)
    covered = np.zeros(300, dtype=bool)
    for s, e in hits.select("start", "end").iter_rows():
        covered[s:e] = True
    ringed = sum(int((np.array([to_hex(e) for e in c.get_edgecolors()]) == "#1a1a1a").sum())
                 for c in _heads(ax))
    assert ringed == 3 * int(covered.sum())
    ax = plot_variant_effects(data["ism"], sequence=seq, motifs=motifs, labels=0, highlight=False).axes[0]
    assert not any((np.array([to_hex(e) for e in c.get_edgecolors()]) == "#1a1a1a").any() for c in _heads(ax))


def test_given_hits_without_motifs(data):
    seq, vcf = data["seq"], data["vcf"]
    hits = um.scan_sequences(data["motifs"], seq)
    fig = plot_variant_effects(vcf, sequence=seq, hits=hits, start=START, labels=0)
    assert len(fig.axes) == 2
    eff = fig.axes[0]
    titles = [a.get_title().get_text() for a in eff.artists if hasattr(a, "get_title")]
    assert "Motif sites" in titles
    with pytest.raises(ValueError, match="several sequences"):
        plot_variant_effects(vcf, sequence=seq, hits=um.scan_sequences(data["motifs"], [seq, seq]),
                             start=START)


def test_labels(data):
    vcf = data["vcf"]
    ax = plot_variant_effects(vcf, start=START, labels=4).axes[0]
    labels = _labels(ax)
    assert len(labels) == 4
    top = vcf.sort(pl.col("effect").abs(), descending=True).head(4)
    want = {f"{p:,} {r}>{a}" for p, r, a in top.select("position", "ref", "alt").iter_rows()}
    assert set(labels) == want
    ax = plot_variant_effects(vcf, sequence=data["seq"], motifs=data["motifs"], start=START,
                              label_by="id", labels=3).axes[0]
    labels = _labels(ax)
    assert len(labels) == 3 and all(t.startswith("rs") for t in labels)
    assert any("\n" in t for t in labels)
    assert _labels(plot_variant_effects(vcf, start=START, labels=0).axes[0]) == []
    # Only one variant per position is labelled.
    labels = _labels(plot_variant_effects(data["ism"], sequence=data["seq"], labels=10).axes[0])
    positions = [t.split(" ")[0] for t in labels]
    assert len(positions) == len(set(positions)) == 10


def test_region_and_sequence_letters(data):
    seq, motifs = data["seq"], data["motifs"]
    fig = plot_variant_effects(data["ism"], sequence=seq, motifs=motifs, start=START,
                               region=(START + 190, START + 250))
    eff, letters = fig.axes[0], fig.axes[1]
    assert _n_heads(eff) == 3 * 60
    assert "".join(t.get_text() for t in letters.texts) == seq[190:250]
    assert fig.axes[-1].get_xlim() == (START + 190, START + 250)
    fig = plot_variant_effects(data["ism"], sequence=seq, region=(190, 250), show_sequence=False)
    assert len(fig.axes) == 1
    # The whole sequence is too long for its letters by default.
    assert len(plot_variant_effects(data["ism"], sequence=seq).axes) == 1
    assert len(plot_variant_effects(data["ism"], sequence=seq, show_sequence=True).axes) == 2
    with pytest.raises(ValueError, match="region must lie within"):
        plot_variant_effects(data["ism"], sequence=seq, region=(250, 310))
    with pytest.raises(ValueError, match="no variants in the region"):
        plot_variant_effects(data["vcf"].filter(pl.col("position") > START + 20), sequence=seq, start=START,
                             region=(START, START + 10))


def test_size_by(data):
    vcf = data["vcf"]
    ax = plot_variant_effects(vcf, start=START, size_by="af", labels=0).axes[0]
    sizes = np.concatenate([c.get_sizes() for c in _heads(ax)])
    assert sizes.min() == pytest.approx(10) and sizes.max() == pytest.approx(80)
    titles = [a.get_title().get_text() for a in ax.artists if hasattr(a, "get_title")]
    assert "af" in titles
    ax = plot_variant_effects(vcf.with_columns(pvalue=10 ** -(pl.col("af") * 10)), start=START,
                              size_by="pvalue", labels=0).axes[0]
    titles = [a.get_title().get_text() for a in ax.artists if hasattr(a, "get_title")]
    assert "-log10(pvalue)" in titles
    with pytest.raises(ValueError, match="numeric"):
        plot_variant_effects(vcf, start=START, size_by="id")


def test_inputs(data):
    vcf, seq = data["vcf"], data["seq"]
    # pandas tables are read with pyarrow, which the pandas extra installs.
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = vcf.rename({"ref": "REF", "alt": "ALT"}).to_pandas()
    ax = plot_variant_effects(frame, sequence=seq, start=START, labels=0).axes[0]
    assert _n_heads(ax) == vcf.height
    rows = list(vcf.select("position", "ref", "alt", "effect").iter_rows())
    assert _n_heads(plot_variant_effects(rows, start=START).axes[0]) == vcf.height
    effects = vcf["effect"].to_numpy() * 2
    ax = plot_variant_effects(vcf.drop("effect"), effects, start=START, labels=0).axes[0]
    assert np.abs(np.concatenate([c.get_offsets()[:, 1] for c in _heads(ax)])).max() == pytest.approx(
        2 * np.abs(vcf["effect"]).max())
    renamed = vcf.rename({"effect": "pred"})
    assert _n_heads(plot_variant_effects(renamed, "pred", start=START).axes[0]) == vcf.height


def test_input_errors(data):
    vcf, seq, ism = data["vcf"], data["seq"], data["ism"]
    with pytest.raises(ValueError, match="do not match the sequence: position 5248"):
        plot_variant_effects(vcf.with_columns(pl.lit("N").alias("ref")), sequence=seq, start=START)
    with pytest.raises(ValueError, match="outside the sequence"):
        plot_variant_effects(vcf, sequence=seq)
    with pytest.raises(ValueError, match="needs the sequence"):
        plot_variant_effects(ism)
    with pytest.raises(ValueError, match="several outputs"):
        plot_variant_effects(np.stack([ism, ism]), sequence=seq)
    with pytest.raises(ValueError, match="letters for"):
        plot_variant_effects(ism[:, :100], sequence=seq)
    with pytest.raises(ValueError, match="'effect' column"):
        plot_variant_effects(vcf.drop("effect"), start=START)
    with pytest.raises(ValueError, match="no 'pred' column"):
        plot_variant_effects(vcf, "pred", start=START)
    with pytest.raises(ValueError, match="effects for"):
        plot_variant_effects(vcf.drop("effect"), [1.0, 2.0], start=START)
    with pytest.raises(ValueError, match="missing or infinite"):
        plot_variant_effects(vcf.with_columns(pl.lit(None, dtype=pl.Float64).alias("effect")), start=START)
    with pytest.raises(ValueError, match="motifs need the sequence"):
        plot_variant_effects(vcf, motifs=data["motifs"], start=START)
    with pytest.raises(ValueError, match="colour_by"):
        plot_variant_effects(vcf, start=START, colour_by="effect")
    with pytest.raises(ValueError, match="summary"):
        plot_variant_effects(vcf, start=START, summary="median")
    with pytest.raises(ValueError, match="labels"):
        plot_variant_effects(vcf, start=START, labels=-1)
    with pytest.raises(ValueError, match="no 'name' column"):
        plot_variant_effects(vcf, start=START, label_by="name")
    with pytest.raises(ValueError, match="show_sequence needs"):
        plot_variant_effects(vcf, start=START, show_sequence=True)


def test_drawing_into_axes(data):
    fig, (top, _other) = plt.subplots(2, 1, figsize=(9, 6))
    out = plot_variant_effects(data["vcf"], sequence=data["seq"], motifs=data["motifs"], start=START, ax=top)
    assert isinstance(out, Axes) and out is top
    assert len(fig.axes) == 3
    hit = next(a for a in fig.axes if a.get_ylabel() == "Hits")
    assert hit.get_position().x0 == pytest.approx(top.get_position().x0)
    assert hit.get_position().width == pytest.approx(top.get_position().width)
    assert hit.get_position().y1 <= top.get_position().y0
    assert fig.get_size_inches().tolist() == [9, 6]


def test_saves_tightly_with_legends_inside(data):
    fig = plot_variant_effects(data["ism"], sequence=data["seq"], motifs=data["motifs"], start=START)
    renderer = fig.canvas.get_renderer()
    for leg in fig.axes[0].artists:
        box = leg.get_window_extent(renderer)
        assert box.x1 <= fig.bbox.x1 + 1 and box.y0 >= -1
    axes = fig.axes
    assert len({round(a.get_position().x0, 6) for a in axes}) == 1
    assert len({round(a.get_position().width, 6) for a in axes}) == 1
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    assert buf.getbuffer().nbytes > 0


def _clustered():
    gata = um.example_motifs()[6]
    seq = um.create_sequences(seqnum=1, seqlen=120, seed=3)[0]
    seq = seq[:50] + "TTAGATAAGG" + seq[60:]

    def model(batch):
        words = [s[i:i + 8] for s in um.from_one_hot(batch) for i in range(len(s) - 7)]
        return um.score_match(gata, words).reshape(len(batch), -1).max(axis=1)

    return um.mutagenesis_effects(model(um.mutagenesis(seq)), seq), seq, gata


def _overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


@pytest.mark.parametrize("kw", [{}, {"region": (30, 80)}, {"labels": 8}])
def test_bunched_labels_are_set_apart(kw):
    # Every substitution in one site: the top variants are bunched, so their
    # labels go in a row below the stems, joined to them by lines.
    ism, seq, gata = _clustered()
    fig = plot_variant_effects(ism, sequence=seq, motifs=gata, pvalue=1e-3, **kw)
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    labels = [t for t in ax.texts if t.get_text()]
    assert len(labels) == kw.get("labels", 5)
    boxes = []
    for t in labels:
        # The text alone, without its leader line.
        t.update_positions(renderer)
        boxes.append(Text.get_window_extent(t, renderer))
    assert not any(_overlap(a, b) for k, a in enumerate(boxes) for b in boxes[k + 1:])
    assert all(t.arrow_patch is not None for t in labels)
    # No label covers a stem or a head.
    stems = next(c for c in ax.collections if isinstance(c, LineCollection))
    lowest = min(ax.transData.transform(s)[:, 1].min() for s in stems.get_segments())
    heads = [c for c in ax.collections if isinstance(c, PathCollection)]
    lowest = min([lowest] + [c.get_window_extent(renderer).y0 for c in heads])
    assert all(b.y1 < lowest for b in boxes)
    # Labels stay in the axes, and the axis spans the effects only.
    frame = ax.get_window_extent(renderer)
    assert all(frame.x0 <= b.x0 and b.x1 <= frame.x1 and frame.y0 <= b.y0 for b in boxes)
    effects = np.array([seg[1][1] for seg in stems.get_segments()])
    lo, hi = ax.spines["left"].get_bounds()
    assert lo <= effects.min() and hi >= effects.max() and lo > ax.get_ylim()[0]
    assert all(lo - 1e-9 <= t <= hi + 1e-9 for t in ax.get_yticks())


def test_spread_keeps_boxes_apart_and_near():
    from universalmotif._plot_variants import _spread

    # Boxes apart from each other stay where they are wanted.
    np.testing.assert_allclose(_spread(np.array([10.0, 50.0]), np.array([10.0, 10.0]), 0, 100, 2),
                               [10.0, 50.0])
    # Boxes wanted at one place are spread evenly around it.
    np.testing.assert_allclose(_spread(np.array([50.0, 50.0, 50.0]), np.full(3, 10.0), 0, 100, 2),
                               [38.0, 50.0, 62.0])
    # A block running off the line is pushed back onto it, in input order.
    np.testing.assert_allclose(_spread(np.array([99.0, 98.0]), np.full(2, 10.0), 0, 100, 0),
                               [95.0, 85.0])
    assert _spread(np.array([50.0, 50.0]), np.full(2, 60.0), 0, 100, 0) is None
