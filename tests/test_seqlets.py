import numpy as np
import polars as pl
import pytest
import universalmotif as um


@pytest.fixture(scope="module")
def scores():
    """Noisy observed scores with a positive block at 40 to 47 of the even
    sequences and a negative one at 70 to 77 of the odd ones."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, (200, 100))
    x[::2, 40:48] += 0.2
    x[1::2, 70:78] -= 0.2
    return x


def test_blocks_become_seqlets(scores):
    s = um.call_seqlets(scores, flank=0)
    assert s.columns == ["sequence", "sequence_i", "start", "end", "core_start", "core_end", "sign", "score",
                         "pvalue"]
    pos, neg = s.filter(pl.col("sign") == "+"), s.filter(pl.col("sign") == "-")
    on_pos = pos.filter(pl.col("sequence_i") % 2 == 0, pl.col("core_start") >= 36, pl.col("core_end") <= 52)
    on_neg = neg.filter(pl.col("sequence_i") % 2 == 1, pl.col("core_start") >= 66, pl.col("core_end") <= 82)
    assert on_pos.height >= 95 and on_neg.height >= 95
    assert (pos["score"] > 0).all() and (neg["score"] < 0).all()
    assert (s["pvalue"] < 0.01).all()
    assert s["sequence"][0] == f"seq{s['sequence_i'][0] + 1}"
    assert s.select("start", "end").equals(s.select(start="core_start", end="core_end"))
    only = um.call_seqlets(scores, flank=0, sign="positive")
    assert set(only["sign"]) == {"+"} and only.height == pos.height


def test_flanks_fdr_and_names(scores):
    s = um.call_seqlets(scores, flank=3)
    assert ((s["core_start"] - s["start"]) <= 3).all() and ((s["end"] - s["core_end"]) <= 3).all()
    noise = np.random.default_rng(2).normal(0, 0.01, (200, 100))
    assert um.call_seqlets(noise, fdr=None).height > 10 * max(1, um.call_seqlets(noise).height)
    named = um.call_seqlets({f"r{i}": x for i, x in enumerate(scores)}, flank=0)
    assert named["sequence"][0].startswith("r")


def test_letters_and_one_hot_inputs(scores):
    rng = np.random.default_rng(3)
    seqs = ["".join(rng.choice(list("ACGT"), 100)) for _ in range(200)]
    s = um.call_seqlets(scores, sequences=seqs, flank=0)
    row = s.row(0, named=True)
    assert row["match"] == seqs[row["sequence_i"]][row["start"]:row["end"]]
    # Scores of every letter, projected onto the letters present.
    onehot = um.one_hot(seqs)
    projected = onehot * scores[:, None, :]
    by_letter = um.call_seqlets(projected, flank=0)
    assert by_letter.drop("match", strict=False).equals(s.drop("match"))
    with_seqs = um.call_seqlets(projected + (1 - onehot) * 0.5, sequences=seqs, flank=0)
    assert with_seqs.select("sequence_i", "start", "end").equals(s.select("sequence_i", "start", "end"))


def test_seqlets_feed_discovery(scores):
    rng = np.random.default_rng(4)
    seqs = ["".join(rng.choice(list("ACGT"), 100)) for _ in range(200)]
    seqs = [s[:40] + "TGACTCAG" + s[48:] if i % 2 == 0 else s for i, s in enumerate(seqs)]
    s = um.call_seqlets(scores, sequences=seqs, sign="positive")
    found = um.motif_finder(s["match"].to_list(), nmotifs=2, seed=1)
    sel = um.select_motifs(found, seqs, sites=s)
    assert sel["kept"].any()


def test_errors(scores):
    with pytest.raises(ValueError, match="sign"):
        um.call_seqlets(scores, sign="up")
    with pytest.raises(ValueError, match="min_len"):
        um.call_seqlets(scores, min_len=10, max_len=5)
    with pytest.raises(ValueError, match="threshold"):
        um.call_seqlets(scores, threshold=0)
    with pytest.raises(ValueError, match="fdr"):
        um.call_seqlets(scores, fdr=2)
    assert um.call_seqlets(np.zeros((3, 50))).height == 0


def test_batches_against_one_null_give_one_call():
    rng = np.random.default_rng(7)
    scores = rng.normal(0, 0.01, (300, 120))
    scores[::3, 50:58] += 0.15
    scores[1::4, 20:26] -= 0.15
    decoys = rng.permuted(scores, axis=1)
    pilot = scores[:80]
    found, fake = [], []
    for lo in range(0, 300, 70):
        part = um.call_seqlets(scores[lo:lo + 70], null=pilot, fdr=None, flank=1)
        found.append(part.with_columns(pl.col("sequence_i") + lo))
        fake.append(um.call_seqlets(decoys[lo:lo + 70], null=pilot, fdr=None, flank=1))
    kept = um.seqlet_fdr(pl.concat(found), pl.concat(fake))
    whole = um.call_seqlets(scores, null=pilot, decoys=decoys, flank=1)
    cols = ["sequence_i", "start", "end", "sign", "score"]
    assert kept.select(cols).equals(whole.select(cols))
    assert set(kept["sign"]) == {"+", "-"}
    # Decoys of half the positions count double.
    half = um.seqlet_fdr(pl.concat(found), pl.concat(fake[:2]), positions=300 * 120, decoy_positions=140 * 120)
    assert half.height <= kept.height + 5
    with pytest.raises(ValueError, match="fdr"):
        um.seqlet_fdr(kept, kept, fdr=0)
    with pytest.raises(ValueError, match="both positions"):
        um.seqlet_fdr(kept, kept, positions=10)
    with pytest.raises(ValueError, match="score"):
        um.seqlet_fdr(kept.drop("score"), kept)
