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
