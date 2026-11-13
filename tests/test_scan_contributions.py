import numpy as np
import polars as pl
import pytest
import universalmotif as um
from test_deep import _modisco_file


def _cwm(name, mat):
    return um.create_motif(np.asarray(mat, dtype=float), type="CWM", name=name, alphabet="DNA")


def _data(seed=1, n=6, length=60):
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]
    weights = rng.normal(0, 0.3, (n, length)).astype(np.float32)
    motifs = [_cwm("a", rng.normal(0, 1, (4, 6))), _cwm("b", rng.normal(0, 1, (4, 9)))]
    return seqs, weights, motifs


def _slow(motifs, projected, rc=True):
    """Every window's similarity, importance and score, written out."""
    rows = []
    for mi, m in enumerate(motifs):
        unit = m.matrix / np.sqrt((m.matrix ** 2).sum())
        w = unit.shape[1]
        filters = [("+", unit)] + ([("-", unit[::-1, ::-1])] if rc else [])
        for si, x in enumerate(projected):
            for p in range(x.shape[1] - w + 1):
                win = x[:, p:p + w].astype(float)
                norm = np.sqrt((win ** 2).sum())
                if norm == 0:
                    continue
                for strand, f in filters:
                    dot = float((win * f).sum())
                    rows.append((mi, si, p, strand, dot, dot / norm, float(np.abs(win).sum())))
    return pl.DataFrame(rows, schema=["motif_i", "sequence_i", "start", "strand", "score", "similarity",
                                      "importance"], orient="row")


def test_every_window_matches_a_slow_reading():
    seqs, weights, motifs = _data()
    projected = um.one_hot(seqs) * weights[:, None, :]
    got = um.scan_contributions(motifs, projected, min_similarity=-1, min_importance=0, no_overlaps=False)
    want = _slow(motifs, projected)
    assert got.height == want.height
    joined = got.join(want, on=["motif_i", "sequence_i", "start", "strand"], suffix="_want")
    assert joined.height == want.height
    for col in ("score", "similarity", "importance"):
        np.testing.assert_allclose(joined[col], joined[f"{col}_want"], rtol=1e-5, atol=1e-6)
    assert (got["end"] - got["start"]).to_list() == [6 if i == 0 else 9 for i in got["motif_i"]]
    # Rows come by motif, sequence and start, + before -.
    assert got.select("motif_i", "sequence_i", "start").equals(got.select("motif_i", "sequence_i", "start")
                                                               .sort("motif_i", "sequence_i", "start"))
    # Letters read along the strand.
    for row in got.head(50).iter_rows(named=True):
        site = seqs[row["sequence_i"]][row["start"]:row["end"]]
        want = site if row["strand"] == "+" else site[::-1].translate(str.maketrans("ACGT", "TGCA"))
        assert row["match"] == want


def test_thresholds_filter_the_slow_reading():
    seqs, weights, motifs = _data(seed=2)
    projected = um.one_hot(seqs) * weights[:, None, :]
    want = _slow(motifs, projected, rc=False)
    l1 = {i: float(np.abs(m.matrix).sum()) for i, m in enumerate(motifs)}
    smin, imin = [0.3, 0.5], [0.06, 0.08]
    keep = want.filter(
        (pl.col("similarity") >= pl.col("motif_i").replace_strict({0: smin[0], 1: smin[1]}))
        & (pl.col("importance") >= pl.col("motif_i").replace_strict({0: imin[0] * l1[0], 1: imin[1] * l1[1]})))
    got = um.scan_contributions(motifs, projected, min_similarity=smin, min_importance=imin, rc=False,
                                no_overlaps=False)
    assert got.height == keep.height > 0
    by_name = um.scan_contributions(motifs, projected, min_similarity={"a": 0.3, "b": 0.5},
                                    min_importance={"a": 0.06, "b": 0.08}, rc=False, no_overlaps=False)
    assert by_name.equals(got)
    table = pl.DataFrame({"motif": ["b", "a"], "min_similarity": [0.5, 0.3], "min_importance": [0.08, 0.06]})
    assert um.scan_contributions(motifs, projected, thresholds=table, rc=False, no_overlaps=False).equals(got)


def test_input_forms_agree():
    seqs, weights, motifs = _data(seed=3)
    onehot = um.one_hot(seqs)
    projected = onehot * weights[:, None, :]
    want = um.scan_contributions(motifs, projected, min_similarity=0.2, min_importance=0.05)
    assert want.height > 0
    hypothetical = projected + (1 - onehot) * 0.7
    forms = [
        {"contributions": weights, "sequences": seqs},
        {"contributions": hypothetical, "sequences": onehot},
        {"contributions": projected.transpose(0, 2, 1)},
        {"contributions": list(projected)},
        {"contributions": um.ContributionScores(hypothetical, seqs, [f"seq{i + 1}" for i in range(6)], None)},
    ]
    for kw in forms:
        got = um.scan_contributions(motifs, min_similarity=0.2, min_importance=0.05, **kw)
        assert got.equals(want), kw.keys()
    named = um.scan_contributions(motifs, dict(zip("uvwxyz", projected)), min_similarity=0.2, min_importance=0.05)
    assert named["sequence"].to_list() == ["uvwxyz"[i] for i in want["sequence_i"]]


def test_ragged_sequences_stay_within_their_ends():
    seqs, weights, motifs = _data(seed=4)
    projected = list(um.one_hot(seqs) * weights[:, None, :])
    projected[1] = projected[1][:, :20]
    got = um.scan_contributions(motifs, projected, min_similarity=-1, min_importance=0, no_overlaps=False)
    assert got.filter(pl.col("sequence_i") == 1)["end"].max() <= 20
    assert got.filter(pl.col("sequence_i") == 0)["end"].max() == 60


def test_overlaps():
    core = np.zeros((4, 6))
    core[[2, 0, 3, 0, 0, 2], range(6)] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # GATAAG
    gat = _cwm("gataag", core)
    half = _cwm("gat", core[:, :3])
    seq = "CCCCGATAAGCCCCCCCC"
    weights = np.full(len(seq), 0.01, dtype=np.float32)
    weights[4:10] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    x = um.one_hot([seq]) * weights[None, None, :]
    raw = um.scan_contributions([gat, half], x, no_overlaps=False)
    assert raw.filter(pl.col("motif") == "gataag").height >= 1
    hits = um.scan_contributions([gat, half], x)
    assert hits.group_by("motif").len().sort("motif")["len"].to_list() == [1, 1]
    assert hits.filter(pl.col("motif") == "gataag")["start"].item() == 4
    best = um.scan_contributions([gat, half], x, no_overlaps_by_motif=True)
    assert best["motif"].to_list() == ["gataag"]
    # Downstream functions take the table.
    assert um.dedup_hits(hits).height == hits.height


def test_modisco_thresholds(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _modisco_file(tmp_path / "modisco.h5")
    thr = um.modisco_thresholds(path, quantile=0.1)
    cwms = um.read_modisco(path)
    assert thr["motif"].to_list() == [m.name for m in cwms]
    assert thr.columns == ["motif", "min_similarity", "min_importance", "seqlets"]
    with h5py.File(path) as f:
        g = f["pos_patterns"]["pattern_2"]
        full = np.asarray(g["contrib_scores"]).T
        total = np.abs(full).sum(0)
        keep = np.flatnonzero(total >= 0.3 * total.max())
        lo, hi = keep[0], keep[-1] + 1
        cwm = full[:, lo:hi]
        x = np.asarray(g["seqlets"]["contrib_scores"])[:, lo:hi, :].transpose(0, 2, 1)
    sim = (x * (cwm / np.sqrt((cwm ** 2).sum()))).sum((1, 2)) / np.sqrt((x ** 2).sum((1, 2)))
    imp = np.abs(x).sum((1, 2)) / np.abs(cwm).sum()
    row = thr.row(1, named=True)
    assert row["motif"] == "pos_patterns.pattern_2"
    assert row["min_similarity"] == pytest.approx(np.quantile(sim, 0.1))
    assert row["min_importance"] == pytest.approx(np.quantile(imp, 0.1))
    assert row["seqlets"] == len(x)
    subs = um.modisco_thresholds(path, subpatterns=True)
    assert subs["motif"].to_list() == [m.name for m in um.read_modisco(path, subpatterns=True)]
    with pytest.raises(ValueError, match="quantile"):
        um.modisco_thresholds(path, quantile=2)


def test_errors():
    seqs, weights, motifs = _data()
    projected = um.one_hot(seqs) * weights[:, None, :]
    with pytest.raises(ValueError, match="CWM"):
        um.scan_contributions(um.create_motif("GATA"), projected)
    with pytest.raises(ValueError, match="need the sequences"):
        um.scan_contributions(motifs, weights)
    with pytest.raises(ValueError, match="2 motifs"):
        um.scan_contributions(motifs, projected, min_similarity=[0.5])
    with pytest.raises(ValueError, match="no value for b"):
        um.scan_contributions(motifs, projected, min_importance={"a": 0.5})
    with pytest.raises(ValueError, match="no value for"):
        um.scan_contributions(motifs, projected, thresholds=pl.DataFrame(
            {"motif": ["a"], "min_similarity": [0.5], "min_importance": [0.5]}))
    with pytest.raises(ValueError, match="NaN"):
        um.scan_contributions(motifs, projected, min_similarity=[0.5, float("nan")])
    with pytest.raises(ValueError, match="no_overlaps_by"):
        um.scan_contributions(motifs, projected, no_overlaps_by="pvalue")
    with pytest.raises(ValueError, match="no motifs"):
        um.scan_contributions([], projected)
    empty = um.scan_contributions(motifs, projected, min_similarity=1.1)
    assert empty.height == 0 and empty.columns[-1] == "match"


def test_normalize_raises_weak_sequences():
    core = np.zeros((4, 6))
    core[[2, 0, 3, 0, 0, 2], range(6)] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # GATAAG
    cwm = _cwm("gataag", core)
    seq = "CCCCGATAAGCCCCCCCCCC"
    site = np.full(len(seq), 0.02, dtype=np.float32)
    site[4:10] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    strengths = [1.0, 1.0, 0.1]  # the last sequence is ten times weaker
    x = um.one_hot([seq] * 3) * (site[None, :] * np.array(strengths, dtype=np.float32)[:, None])[:, None, :]
    plain = um.scan_contributions(cwm, x)
    assert plain["sequence_i"].to_list() == [0, 1]
    scaled = um.scan_contributions(cwm, x, normalize=True)
    assert scaled["sequence_i"].to_list() == [0, 1, 2]
    # Sequences at the median keep their scores.
    np.testing.assert_allclose(scaled["importance"][:2], plain["importance"])
    np.testing.assert_allclose(scaled["importance"][2], plain["importance"][0], rtol=1e-5)
    assert um.scan_contributions(cwm, x[2:], normalize=True).height == 0


def test_gapped_cwms():
    core = np.zeros((4, 10))
    core[[2, 0, 3, 0, 1, 0, 1, 2, 3, 2], range(10)] = 1.0  # GATA, then CACGTG
    gapped = um.add_gap(_cwm("gapped", core), gaploc=4, mingap=2, maxgap=6)
    site = "GATATTTTCACGTG"
    rc = site[::-1].translate(str.maketrans("ACGT", "TGCA"))
    seqs = ["TT" + site + "TT", "CC" + rc + "CC", "TTGATACCACGTGTTTTT"]
    weights = np.full((3, 18), 0.01, dtype=np.float32)
    weights[:2, 2:16] = 0.5
    weights[2, 2:13] = 0.5
    x = um.one_hot(seqs) * weights[:, None, :]
    hits = um.scan_contributions(gapped, x, min_similarity=0.99, min_importance=0)
    # The third sequence's gap of one letter is too short.
    assert hits.select("sequence_i", "start", "end", "strand", "match").rows() == [
        (0, 2, 16, "+", "GATA....CACGTG"), (1, 2, 16, "-", "GATA....CACGTG")]
    np.testing.assert_allclose(hits["similarity"], 1.0, atol=1e-6)
    # The gap's letters count towards neither measure.
    np.testing.assert_allclose(hits["importance"], 5.0, atol=1e-5)
    assert um.scan_contributions(gapped, x, min_similarity=0.99, min_importance=0, use_gaps=False).height == 0
