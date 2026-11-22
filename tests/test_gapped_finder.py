import numpy as np
import polars as pl
import pytest
import universalmotif as um
from universalmotif._cli import main


@pytest.fixture(scope="module")
def planted():
    """Direct repeats of AGGTCA, spacers of 1 to 5 letters, in half of 400
    random sequences."""
    dr = um.add_gap(um.create_motif("AGGTCAAGGTCA"), gaploc=6, mingap=1, maxgap=5)
    seqs = um.create_sequences(seqnum=400, seqlen=150, seed=1)
    return um.implant_motifs(dr, seqs[:200], seed=1) + seqs[200:]


def test_finds_a_planted_direct_repeat(planted):
    found = um.gapped_motif_finder(planted, k=5, seed=1)
    assert found.height >= 1
    top = found.row(0, named=True)
    assert top["consensus"] in ("AGGTCA[1-5]AGGTCA", "TGACCT[1-5]TGACCT")
    assert (top["left"], top["right"], top["min_gap"], top["max_gap"]) == (6, 6, 1, 5)
    assert top["seqs_pos"] > 150 and top["seqs_neg"] < 10
    # The background is three shuffles of the sequences by default.
    assert (top["n_pos"], top["n_neg"]) == (400, 1200)
    assert top["log10_spacing_pvalue"] < -3 and top["spacing_pvalue"] < 1e-3
    assert 1 <= top["best_spacer"] <= 5 and top["spacer_pairs"] >= 5 and top["spacer_fold"] > 2
    assert top["best_spacer"] in top["strong_spacers"]
    m = top["motif"]
    assert m.isgapped and m.name == "gapped_1" and m.gaploc == [6]
    hits = um.scan_sequences(m, planted, no_overlaps=True)
    assert hits["sequence_i"].n_unique() > 150


def test_columns_and_an_empty_result():
    seqs = um.create_sequences(seqnum=300, seqlen=150, seed=2)
    found = um.gapped_motif_finder(seqs, k=5, seed=1)
    assert found.height == 0
    assert found.columns[:4] == ["name", "consensus", "left", "right"]
    assert found.columns[-1] == "motif"
    assert found.schema["log10_spacing_pvalue"] == pl.Float64
    assert found.schema["strong_spacers"] == pl.List(pl.Int64)


def test_takes_a_background_and_a_mapping(planted):
    named = {f"s{i}": s for i, s in enumerate(planted)}
    bkg = um.shuffle_sequences(planted, seed=3)
    found = um.gapped_motif_finder(named, bkg, k=[5], nmotifs=1, holdout=0, seed=1)
    assert found.height == 1 and found["max_gap"][0] >= 4


def test_checks_its_options(planted):
    with pytest.raises(ValueError, match="k must be from 2 to 6"):
        um.gapped_motif_finder(planted, k=7)
    with pytest.raises(ValueError, match="min_gap <= max_gap"):
        um.gapped_motif_finder(planted, min_gap=5, max_gap=2)
    with pytest.raises(ValueError, match="holdout"):
        um.gapped_motif_finder(planted, holdout=0.95)
    with pytest.raises(ValueError, match="too few sequences"):
        um.gapped_motif_finder(planted[:1])
    with pytest.raises(ValueError, match="mismatch"):
        um.gapped_motif_finder(planted, mismatch="yes")


def test_mismatch_modes(planted):
    # The exact search finds the repeat, so "auto" never runs again.
    auto = um.gapped_motif_finder(planted, k=6, nmotifs=1, seed=1)
    exact = um.gapped_motif_finder(planted, k=6, nmotifs=1, mismatch=False, seed=1)
    assert auto.height == 1 and auto.drop("motif").equals(exact.drop("motif"))
    loose = um.gapped_motif_finder(planted, k=6, nmotifs=1, mismatch=True, seed=1)
    assert loose.height == 1


def test_find_gapped_command(planted, tmp_path):
    fa = tmp_path / "t.fa"
    um.write_fasta(planted, fa)
    code = main(["find-gapped", "-s", str(fa), "-k", "5", "--seed", "1", "--table", str(tmp_path / "t.tsv"),
                 "-o", str(tmp_path / "t.meme")])
    assert code == 0
    table = pl.read_csv(tmp_path / "t.tsv", separator="\t")
    assert table["max_gap"][0] == 5
    motifs = um.read_meme(tmp_path / "t.meme")
    assert len(motifs) == table.height and motifs[0].width == 12


@pytest.fixture(scope="module")
def pair_planted():
    seqs = um.create_sequences(seqnum=1500, seqlen=200, seed=1)
    pair = um.add_gap(um.create_motif("TGTCGGTTGACT"), gaploc=6, mingap=7, maxgap=7)
    seqs[::20] = um.implant_motifs(pair, seqs[::20], seed=1)
    seqs[1::3] = um.implant_motifs(um.create_motif("TGTCGG"), seqs[1::3], seed=2)
    seqs[2::6] = um.implant_motifs(um.create_motif("TTGACT"), seqs[2::6], seed=3)
    return seqs


def test_motif_pair_finder_finds_a_planted_pair(pair_planted):
    arf = um.create_motif("TGTCGG", name="ARF")
    found = um.motif_pair_finder(pair_planted, motifs=arf, seed=1)
    assert found.height >= 1
    top = found.row(0, named=True)
    assert top["name"] == "pair_1" and top["left"] == "ARF" and top["left_strand"] == "+"
    assert top["arrangement"] in ("tandem", "convergent")
    assert top["consensus"].startswith("TGTCGG[") and "TTGACT" in top["consensus"]
    assert top["log10_pvalue"] < -5 and top["fold"] > 5 and not top["repeat"]
    assert np.isclose(top["log10_pvalue"], np.log10(top["pvalue"]))
    m = top["motif"]
    assert m.isgapped and m.gaploc == [6] and (m.mingap[0], m.maxgap[0]) == (top["min_gap"], top["max_gap"])
    assert top["left_motif"].name == "ARF" and top["right_motif"].name == top["right"]
    assert found.columns[-3:] == ["motif", "left_motif", "right_motif"]


def test_motif_pair_finder_without_parts_given_and_on_random_sequences(pair_planted):
    found = um.motif_pair_finder(pair_planted, bkg_sequences=um.shuffle_sequences(pair_planted, k=2, seed=1),
                                 seed=2, nmotifs=4)
    assert found.height >= 1
    seqs = um.create_sequences(seqnum=1500, seqlen=200, seed=3)
    empty = um.motif_pair_finder(seqs, motifs=um.create_motif("TGTCGG"), seed=1, nmotifs=4)
    assert empty.height == 0
    assert empty.schema["repeat"] == pl.Boolean and empty.columns[-1] == "right_motif"


def test_motif_pair_finder_checks_its_options(pair_planted):
    with pytest.raises(ValueError, match="give motifs or set nmotifs"):
        um.motif_pair_finder(pair_planted, nmotifs=0)
    with pytest.raises(ValueError, match="max_gap"):
        um.motif_pair_finder(pair_planted, max_gap=40)
    with pytest.raises(ValueError, match="holdout"):
        um.motif_pair_finder(pair_planted, holdout=0)
    with pytest.raises(ValueError, match="not a DNA motif"):
        um.motif_pair_finder(pair_planted, motifs=um.create_motif("ACDEF", alphabet="AA"))
