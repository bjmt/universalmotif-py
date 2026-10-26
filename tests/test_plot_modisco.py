import io

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import universalmotif as um
from universalmotif._plot_modisco import plot_modisco


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _modisco_file(path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)

    def pattern(grp, name, n, core):
        g = grp.create_group(name)
        contrib = rng.normal(0, 0.01, (14, 4))
        seq = np.full((14, 4), 0.25)
        for i, letter in enumerate(core):
            j = "ACGT".index(letter)
            contrib[3 + i] = 0
            contrib[3 + i, j] = 1.0
            seq[3 + i] = 0.02
            seq[3 + i, j] = 0.94
        g.create_dataset("contrib_scores", data=contrib)
        g.create_dataset("hypothetical_contribs", data=contrib * 2)
        g.create_dataset("sequence", data=seq)
        g.create_group("seqlets").create_dataset("n_seqlets", data=np.array([n]))
        return g

    with h5py.File(path, "w") as f:
        pos = f.create_group("pos_patterns")
        pattern(pos, "run_pattern_0", 1200, "TTGACGTCAT")
        pattern(pos, "run_pattern_1", 400, "CCTTTTTATAG")
        pattern(pos, "run_pattern_2", 15, "CACGTG")
        pattern(f.create_group("neg_patterns"), "run_pattern_0", 30, "GGATAAGG")
    return path


@pytest.fixture
def modisco(tmp_path):
    return _modisco_file(tmp_path / "modisco.h5")


@pytest.fixture(scope="module")
def known():
    return um.example_motifs()


def _matches(path, known, trim_threshold=0.2, rename=True):
    ppms = um.read_modisco(path, type="PPM", trim_threshold=trim_threshold)
    if rename:
        ppms = [m.replace(name=f"{m.name[:3]}_{m.name.rsplit('_', 1)[1]}") for m in ppms]
    return um.compare_motifs(ppms + known, compare_to=list(range(len(ppms))), qvalue=0.5)


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def _logos(fig):
    return [c for c in fig.axes[0].collections if len(c.get_paths())]


def test_file_report(modisco, known):
    cmp = _matches(modisco, known)
    fig = plot_modisco(modisco, matches=cmp, database=known)
    texts = _texts(fig)
    assert texts[:5] == ["Pattern", "Seqlets", "CWM", "PPM", "Best match"]
    labels = [t for t in texts if t[:4] in ("pos_", "neg_")]
    assert labels == ["pos_0", "pos_1", "pos_2", "neg_0"]
    assert "Positive patterns" in texts and "Negative patterns" in texts
    assert texts.index("Positive patterns") < texts.index("pos_0") < texts.index("Negative patterns")
    assert "1,200" in texts and "30" in texts
    # The best matches are labelled with the example motifs' names.
    best = cmp.filter(pl.col("target_i") >= 4).sort("qvalue").group_by("subject", maintain_order=True).first()
    for target in best["target"]:
        assert target in texts
    # Two logos per pattern plus one per match.
    assert len(_logos(fig)) == 2 * 4 + best.height
    assert fig.axes[0].get_label() == "modisco"


def test_patterns_are_found_by_full_name(modisco, known):
    renamed = _matches(modisco, known)
    full = _matches(modisco, known, rename=False)
    a = _texts(plot_modisco(modisco, matches=renamed, database=known))
    b = _texts(plot_modisco(modisco, matches=full, database=known))
    assert a == b


def test_matches_skip_patterns_and_respect_n_matches(modisco, known):
    cmp = _matches(modisco, known)
    fig = plot_modisco(modisco, matches=cmp, database=known, n_matches=3)
    texts = _texts(fig)
    assert texts[4:7] == ["Match 1", "Match 2", "Match 3"]
    # A pattern is never its own match, nor that of another pattern.
    names = {"pos_0", "pos_1", "pos_2", "neg_0"}
    assert [t for t in texts if t in names] == ["pos_0", "pos_1", "pos_2", "neg_0"]
    per = cmp.filter(pl.col("target_i") >= 4).group_by("subject").len()
    drawn = sum(min(n, 3) for n in per["len"])
    assert len(_logos(fig)) == 2 * 4 + drawn


def test_matches_are_oriented(known):
    wbox = known[5]
    cwm = um.create_motif(um.motif_rc(wbox).matrix / wbox.nsites, alphabet="DNA", type="CWM",
                          name="pos_patterns.pattern_0", nsites=10)
    ppm = um.convert_type(um.motif_rc(wbox), "PPM").replace(name="pos_patterns.pattern_0")
    cmp = um.compare_motifs([ppm, wbox], compare_to=0, qvalue=1)
    assert cmp.filter(pl.col("target") == "W-box")["strand"].item() == "-"
    fig = plot_modisco({"CWM": [cwm], "PPM": [ppm]}, matches=cmp, database=[wbox], use_type="PPM")
    pattern, match = _logos(fig)[1:]
    # Drawn as probabilities, the reverse-complemented match has the same
    # letters as the pattern, so their shapes line up after a shift.
    a, b = pattern.get_paths(), match.get_paths()
    assert len(a) == len(b)
    shift = b[0].vertices[:, 0].min() - a[0].vertices[:, 0].min()
    for p, q in zip(a, b):
        np.testing.assert_allclose(q.vertices[:, 0] - shift, p.vertices[:, 0], atol=1e-9)
        np.testing.assert_allclose(q.vertices[:, 1], p.vertices[:, 1], atol=1e-9)


def test_align_shares_a_frame(known):
    ap1 = known[1]
    core = um.create_motif("TGACTCA", alphabet="DNA", name="pos_patterns.pattern_0", nsites=5)
    ppm = um.convert_type(core, "PPM")
    cmp = um.compare_motifs([ppm, ap1], compare_to=0, qvalue=1, min_overlap=5)
    row = cmp.filter(pl.col("target") == "AP-1")
    assert row.height == 1
    unaligned = plot_modisco([ppm], matches=cmp, database=[ap1])
    aligned = plot_modisco([ppm], matches=cmp, database=[ap1], align=True)

    def start(fig):
        return min(v.vertices[:, 0].min() for v in _logos(fig)[0].get_paths())

    # AP-1 overhangs the pattern by one position on the left, so aligning
    # moves the pattern a position, a tenth of an inch, to the right.
    assert row["offset"].item() == -1
    assert start(aligned) - start(unaligned) == pytest.approx(0.1)
    assert aligned.get_size_inches()[0] >= unaligned.get_size_inches()[0]


def test_logos_are_centred(modisco):
    fig = plot_modisco(modisco, columns=["CWM"])
    logos = _logos(fig)
    centres = [np.mean([p.vertices[:, 0].mean() for p in c.get_paths()]) for c in logos]
    lefts = [min(p.vertices[:, 0].min() for p in c.get_paths()) for c in logos]
    # The CWMs of the four patterns differ in width but share a centre.
    assert len(logos) == 4
    assert max(centres) - min(centres) < 0.1
    assert len({round(x, 3) for x in lefts}) > 1


def test_pattern_selection(modisco):
    texts = _texts(plot_modisco(modisco, patterns=1))
    assert [t for t in texts if t[:4] in ("pos_", "neg_")] == ["pos_0", "neg_0"]
    texts = _texts(plot_modisco(modisco, patterns=(2, 0)))
    assert [t for t in texts if t[:4] in ("pos_", "neg_")] == ["pos_0", "pos_1"]
    assert "Negative patterns" not in texts
    texts = _texts(plot_modisco(modisco, patterns=["neg_0", "pos_patterns.run_pattern_2"]))
    assert [t for t in texts if t[:4] in ("pos_", "neg_")] == ["pos_2", "neg_0"]
    with pytest.raises(ValueError, match="no pattern named"):
        plot_modisco(modisco, patterns=["pos_9"])
    with pytest.raises(ValueError, match="no patterns to draw"):
        plot_modisco(modisco, patterns=0)


def test_names(modisco):
    fig = plot_modisco(modisco, names={"pos_patterns.run_pattern_0": "GATA"}, patterns=1)
    assert "GATA" in _texts(fig) and "neg_0" in _texts(fig)
    fig = plot_modisco(modisco, names=lambda name: name.upper(), patterns=(1, 0))
    assert "POS_PATTERNS.RUN_PATTERN_0" in _texts(fig)


def test_columns(modisco):
    fig = plot_modisco(modisco, columns=["CWM", "hCWM", "PPM"])
    texts = _texts(fig)
    assert texts[:4] == ["Pattern", "CWM", "hCWM", "PPM"]
    assert "1,200" not in texts
    assert len(_logos(fig)) == 3 * 4
    fig = plot_modisco(modisco, columns="seqlets")
    assert not _logos(fig)
    with pytest.raises(ValueError, match="columns"):
        plot_modisco(modisco, columns=["ICM"])


def test_trim_threshold(modisco):
    trimmed = plot_modisco(modisco, patterns=(1, 0), columns=["CWM"])
    full = plot_modisco(modisco, patterns=(1, 0), columns=["CWM"], trim_threshold=None)
    # The untrimmed CWM has all 14 columns and the trimmed one 10, at a
    # tenth of an inch each.
    width = full.get_size_inches()[0] - trimmed.get_size_inches()[0]
    assert width == pytest.approx(0.4)


def test_matches_without_database(modisco, known):
    cmp = _matches(modisco, known)
    fig = plot_modisco(modisco, matches=cmp)
    assert len(_logos(fig)) == 2 * 4
    assert any(t.startswith("q = ") for t in _texts(fig))


def test_no_match(known):
    cwm = um.create_motif(np.eye(4)[:, [0, 1, 2, 3, 0, 1]], alphabet="DNA", type="CWM",
                          name="neg_patterns.pattern_3")
    cmp = pl.DataFrame({"subject": ["x"], "target": ["AP-1"], "qvalue": [0.01]})
    fig = plot_modisco([cwm], matches=cmp, database=known)
    assert "no match" in _texts(fig)
    assert "Negative patterns" in _texts(fig) and "neg_3" in _texts(fig)


def test_motif_lists(known):
    cwm = um.create_motif(np.eye(4)[:, [3, 0, 3, 0, 3, 0]], alphabet="DNA", type="CWM", name="a",
                          nsites=7)
    fig = plot_modisco([cwm])
    texts = _texts(fig)
    assert texts[:3] == ["Pattern", "Seqlets", "CWM"]
    # Names without a group get no heading.
    assert "a" in texts and "7" in texts and "Positive patterns" not in texts
    with pytest.raises(ValueError, match="dict of lists"):
        plot_modisco([cwm, known[0]])
    with pytest.raises(ValueError, match="one motif per pattern"):
        plot_modisco({"CWM": [cwm], "PPM": known[:2]})
    with pytest.raises(ValueError, match="no PPM motifs"):
        plot_modisco({"CWM": [cwm]}, columns=["PPM"])
    with pytest.raises(ValueError, match="must be CWMs"):
        plot_modisco({"CWM": known[:1]})
    with pytest.raises(ValueError, match="keyed"):
        plot_modisco({"ICM": [cwm]})


def test_argument_checks(modisco, known):
    cmp = _matches(modisco, known)
    with pytest.raises(ValueError, match="database needs matches"):
        plot_modisco(modisco, database=known)
    with pytest.raises(ValueError, match="n_matches"):
        plot_modisco(modisco, matches=cmp, n_matches=0)
    with pytest.raises(ValueError, match="use_type"):
        plot_modisco(modisco, use_type="CWM")
    with pytest.raises(TypeError, match="DataFrame"):
        plot_modisco(modisco, matches=cmp.to_dict())
    with pytest.raises(ValueError, match="no 'qvalue' column"):
        plot_modisco(modisco, matches=cmp.drop("qvalue"))
    with pytest.raises(ValueError, match="too small"):
        plot_modisco(modisco, figsize=(1, 1))


def test_figsize_and_tight_bbox(modisco, known):
    cmp = _matches(modisco, known)
    fig = plot_modisco(modisco, matches=cmp, database=known, figsize=(12, 5))
    assert tuple(fig.get_size_inches()) == (12, 5)
    auto = plot_modisco(modisco, matches=cmp, database=known)
    fewer = plot_modisco(modisco, matches=cmp, database=known, patterns=(1, 0))
    assert fewer.get_size_inches()[1] < auto.get_size_inches()[1]
    buf = io.BytesIO()
    auto.savefig(buf, format="png", bbox_inches="tight")
    assert buf.tell() > 0
    # Every text and logo lies inside the figure.
    renderer = auto.canvas.get_renderer()
    box = auto.bbox
    for artist in [*auto.axes[0].texts, *_logos(auto)]:
        ext = artist.get_window_extent(renderer)
        assert ext.x0 >= box.x0 - 1 and ext.x1 <= box.x1 + 1
        assert ext.y0 >= box.y0 - 1 and ext.y1 <= box.y1 + 1
