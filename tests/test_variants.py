import gzip

import numpy as np
import polars as pl
import pytest
import universalmotif as um
from universalmotif._variants import _trim

PVALUE = 1e-3
_RC = str.maketrans("ACGT", "TGCA")


@pytest.fixture(scope="module")
def data():
    """A 400 bp sequence with planted sites, and substitutions, insertions
    and deletions of every kind along it."""
    rng = np.random.default_rng(11)
    motifs = um.example_motifs()
    seq = "".join(rng.choice(list("ACGT"), 400))
    for site, at in (("TGACTCA", 50), ("AGATAAG", 150), ("CTTATCT", 250), ("CACGTG", 320)):
        seq = seq[:at] + site + seq[at + len(site):]
    rows = []
    for _ in range(150):
        p = int(rng.integers(0, 390))
        kind = int(rng.integers(0, 5))
        word = "".join(rng.choice(list("ACGT"), int(rng.integers(1, 9))))
        if kind == 0:
            rows.append((p, seq[p:p + len(word) + 1], seq[p]))
        elif kind == 1:
            rows.append((p, seq[p], seq[p] + word))
        elif kind == 2 and word[:3] != seq[p:p + 3]:
            rows.append((p, seq[p:p + 3], word[:3]))
        elif kind == 3:
            rows.append((p, "-", word))
        elif word[0] != seq[p]:
            rows.append((p, seq[p], word[0]))
    return {"seq": seq, "motifs": motifs, "rows": rows}


def _window_score(motif, seq, start, strand):
    word = seq[start:start + motif.width]
    if strand == "-":
        word = word[::-1].translate(_RC)
    return um.score_match(motif, word)


# read_vcf


VCF = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    "chr1\t101\trs1\tA\tG,T\t50\tPASS\tAF=0.1,0.2\tGT\t0/1\n"
    "chr1\t205\t.\tCTT\tC\t.\tPASS\t.\tGT\t1/1\n"
    "chr2\t7\trs3\tG\t<DEL>\t9.5\tLowQual\tSVTYPE=DEL\tGT\t0/1\n"
)


def test_read_vcf(tmp_path):
    path = tmp_path / "calls.vcf"
    path.write_text(VCF)
    df = um.read_vcf(path)
    assert df.columns == ["chrom", "position", "id", "ref", "alt", "qual", "filter", "info"]
    assert df.rows() == [
        ("chr1", 100, "rs1", "A", "G", 50.0, "PASS", "AF=0.1,0.2"),
        ("chr1", 100, "rs1", "A", "T", 50.0, "PASS", "AF=0.1,0.2"),
        ("chr1", 204, None, "CTT", "C", None, "PASS", None),
        ("chr2", 6, "rs3", "G", "<DEL>", 9.5, "LowQual", "SVTYPE=DEL"),
    ]
    gz = tmp_path / "calls.vcf.gz"
    gz.write_bytes(gzip.compress(VCF.encode()))
    assert um.read_vcf(gz).equals(df)


def test_read_vcf_empty_and_errors(tmp_path):
    path = tmp_path / "empty.vcf"
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    empty = um.read_vcf(path)
    assert empty.height == 0 and empty["position"].dtype == pl.Int64
    path.write_text("chr1\t100\t200\n")
    with pytest.raises(ValueError, match="#CHROM"):
        um.read_vcf(path)
    path.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\tx\t.\tA\tC\t.\t.\t.\n")
    with pytest.raises(ValueError, match="whole numbers"):
        um.read_vcf(path)


# Insertions and deletions


def test_indels_match_scans_of_both_sequences(data):
    seq, motifs, rows = data["seq"], data["motifs"], data["rows"]
    table = um.variant_motif_effects(rows, seq, motifs, pvalue=PVALUE)
    by_name = {m.name: m for m in motifs}
    thr = {m.name: um.motif_pvalue(m, pvalue=PVALUE) for m in motifs}
    ref_hits = um.scan_sequences(motifs, seq, pvalue=PVALUE)
    unpaired = 0
    for v, (p, r, a) in enumerate(rows):
        r, a = ("" if r == "-" else r), ("" if a == "-" else a)
        tp, tr, ta = _trim(p, r, a)
        alt_seq = seq[:p] + a + seq[p + len(r):]
        alt_hits = um.scan_sequences(motifs, alt_seq, pvalue=PVALUE)

        def overlapping(hits, n, tp=tp):
            rel = pl.col("start") - tp
            return {(m, s, t) for m, s, t in hits.filter((rel > -(pl.col("end") - pl.col("start"))) & (rel < n))
                    .select("motif", "start", "strand").iter_rows()}

        got = table.filter(pl.col("variant") == v)
        strong_ref = {(m, s, t) for m, s, t, x in got.select("motif", "start", "strand", "ref_score").iter_rows()
                      if x is not None and x >= thr[m]}
        strong_alt = {(m, s, t) for m, s, t, x in got.select("motif", "start", "strand", "alt_score").iter_rows()
                      if x is not None and x >= thr[m]}
        assert strong_ref == overlapping(ref_hits, len(tr))
        assert strong_alt == overlapping(alt_hits, len(ta))
        for m, s, t, rs, as_ in got.select("motif", "start", "strand", "ref_score", "alt_score").iter_rows():
            rel = s - tp
            # A window has a score in a sequence exactly when it starts
            # before the end of that sequence's allele.
            assert (rs is not None) == (rel < len(tr))
            assert (as_ is not None) == (rel < len(ta))
            if rs is not None:
                assert rs == pytest.approx(_window_score(by_name[m], seq, s, t), abs=1e-9)
            if as_ is not None:
                assert as_ == pytest.approx(_window_score(by_name[m], alt_seq, s, t), abs=1e-9)
            unpaired += rs is None or as_ is None
    assert unpaired > 0
    lone = table.filter(pl.col("ref_score").is_null() | pl.col("alt_score").is_null())
    assert lone["delta"].null_count() == lone.height
    assert (lone["ref_pvalue"].is_null() == lone["ref_score"].is_null()).all()


def test_trim_places_variants_leftmost():
    assert _trim(10, "A", "AT") == (11, "", "T")
    assert _trim(10, "AT", "A") == (11, "T", "")
    assert _trim(10, "ATT", "AT") == (11, "T", "")
    assert _trim(10, "ACG", "ATG") == (11, "C", "T")
    assert _trim(10, "", "GG") == (10, "", "GG")


def test_unusable_variants_are_left_out_with_a_warning(data):
    seq, gata = data["seq"], data["motifs"][6]
    rows = [(150, seq[150], "<DEL>"), (151, seq[151], "."), (152, seq[152], seq[152]), (153, seq[153], "A,C"),
            (154, seq[154], "C" if seq[154] != "C" else "G")]
    with pytest.warns(UserWarning, match="4 of 5 variants"):
        table = um.variant_motif_effects(rows, seq, gata, pvalue=PVALUE)
    assert set(table["variant"]) <= {4}
    with pytest.warns(UserWarning):
        assert um.variant_motif_effects(rows[:1], seq, gata).height == 0


# Genomes


@pytest.fixture(scope="module")
def genome(data, tmp_path_factory):
    seq = data["seq"]
    path = tmp_path_factory.mktemp("genome") / "genome.fa"
    path.write_text(f">chrA\n{'ACGT' * 30}\n>chrB\n{seq.lower()}\n")
    return path


def _on_genome(rows, chrom="chrB"):
    return (pl.DataFrame(rows, schema=["position", "ref", "alt"], orient="row")
            .with_columns(pl.lit(chrom).alias("chrom"), pl.Series("id", [f"v{i}" for i in range(len(rows))])))


def test_genome_matches_one_sequence(data, genome):
    seq, motifs, rows = data["seq"], data["motifs"], data["rows"]
    one = um.variant_motif_effects(rows, seq, motifs, pvalue=PVALUE)
    variants = _on_genome(rows)
    for g in (um.Genome(genome), genome, {"chrA": "ACGT", "chrB": seq}):
        table = um.variant_motif_effects(variants, g, motifs, pvalue=PVALUE)
        assert table.columns[:6] == ["variant", "chrom", "position", "ref", "alt", "id"]
        assert set(table["chrom"]) == {"chrB"}
        assert table.drop("chrom", "id").equals(one)
        assert table["id"].to_list() == [f"v{v}" for v in table["variant"]]
    # Columns of the variants' own named like those of the table are no bother.
    numbered = variants.with_row_index("variant", offset=100).with_columns(pl.lit(1).alias("_row"))
    assert um.variant_motif_effects(numbered, g, motifs, pvalue=PVALUE).equals(table)
    other = variants.rename({"chrom": "seqnames"})
    assert um.variant_motif_effects(other, um.Genome(genome), motifs, pvalue=PVALUE).drop("chrom", "id").equals(one)


def test_genome_with_hits(data, genome):
    motifs, rows = data["motifs"], data["rows"]
    g = um.Genome(genome)
    hits = um.scan_sequences(motifs, g, pvalue=PVALUE)
    table = um.variant_motif_effects(_on_genome(rows), g, motifs, hits=hits)
    seen = set(hits.filter(pl.col("sequence") == "chrB").select("motif", "start", "strand").iter_rows())
    assert table.height > 0
    assert set(table.select("motif", "start", "strand").iter_rows()) <= seen
    # Sites given as hits are scored whatever their scores.
    weak = um.scan_sequences(motifs, g, pvalue=0.05)
    assert um.variant_motif_effects(_on_genome(rows), g, motifs, hits=weak).height > table.height
    with pytest.raises(ValueError, match="chromosome"):
        um.variant_motif_effects(_on_genome(rows), g, motifs, hits=hits.drop("sequence"))


def test_genome_errors(data, genome):
    seq, gata = data["seq"], data["motifs"][6]
    g = um.Genome(genome)
    with pytest.raises(ValueError, match="'B' is not in the genome; the genome has 'chrB'"):
        um.variant_motif_effects(_on_genome([(5, seq[5], "A")], "B"), g, gata)
    with pytest.raises(ValueError, match="do not match the genome"):
        um.variant_motif_effects(_on_genome([(5, "NN", "A")]), g, gata)
    with pytest.raises(ValueError, match="outside its chromosome"):
        um.variant_motif_effects(_on_genome([(399, "AA", "A")]), g, gata)
    with pytest.raises(ValueError, match="chromosome column"):
        um.variant_motif_effects([(5, seq[5], "A")], g, gata)
    with pytest.raises(ValueError, match="must be a table"):
        um.variant_motif_effects(None, g, gata)
    with pytest.raises(TypeError):
        um.variant_motif_effects(_on_genome([(5, seq[5], "A")]), 5, gata)


def test_variants_at_chromosome_ends(data):
    gata = data["motifs"][6]
    genome = {"c": "AGATAAGGTTCCAGATAAG"}
    rows = [(0, "A", "C"), (18, "G", "GA"), (17, "AG", "A"), (0, "-", "CAGATAAG")]
    table = um.variant_motif_effects(_on_genome(rows, "c"), genome, gata, pvalue=PVALUE)
    assert table.height > 0
    assert (table["start"] >= 0).all()
    made = table.filter(pl.col("variant") == 3)
    assert made.filter(pl.col("ref_score").is_null()).height > 0


# Batches for models


def test_variant_batches(data, genome):
    seq = data["seq"]
    rows = [(100, seq[100], "A" if seq[100] != "A" else "C"), (101, seq[101], seq[101] + "TTT"),
            (102, seq[102:106], seq[102]), (3, seq[3], "G" if seq[3] != "G" else "C"), (398, seq[398:400], "")]
    variants = _on_genome(rows)
    batches = um.variant_batches(um.Genome(genome), variants, width=20, batch_size=2)
    assert len(batches) == 3
    pairs = list(batches)
    assert [r.shape for r, _ in pairs] == [(2, 4, 20), (2, 4, 20), (1, 4, 20)]
    ref = um.from_one_hot(np.concatenate([r for r, _ in pairs]))
    alt = um.from_one_hot(np.concatenate([a for _, a in pairs]))
    padded = "N" * 20 + seq.upper() + "N" * 20

    def window(p):
        return padded[p + 20 - 10:p + 20 + 10]

    for k, (p, r, a) in enumerate(rows):
        assert ref[k] == window(p)
        changed = (padded[:p + 20] + a.upper() + padded[p + 20 + len(r):])[p + 10:p + 30]
        assert alt[k] == changed
    assert ref[3].startswith("NNNNNNN")
    assert alt[2] == window(102)[:10] + seq[102] + seq[106:115].upper()
    # Each pass reads the batches again.
    assert len(list(batches)) == 3


def test_variant_batches_errors(data, genome):
    seq = data["seq"]
    g = um.Genome(genome)
    with pytest.raises(ValueError, match="not letters"):
        um.variant_batches(g, _on_genome([(5, seq[5], "<INS>")]), width=10)
    with pytest.raises(ValueError, match="do not match the genome"):
        um.variant_batches(g, _on_genome([(5, "NNN", "A")]), width=10)
    with pytest.raises(ValueError, match="width"):
        um.variant_batches(g, _on_genome([(5, seq[5], "A")]), width=0)


def test_plot_labels_made_and_lost_sites(data):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq, gata = data["seq"], data["motifs"][6]
    # Sites starting in letters only one allele has: the planted GATA site
    # at 150 lies within the deletion, and one starts within the insertion.
    rows = [(145, seq[145:160], seq[145], -1.0), (300, seq[300], seq[300] + "TTTTTCAGATAAGG", 1.0)]
    variants = pl.DataFrame(rows, schema=["position", "ref", "alt", "effect"], orient="row")
    fig = um.plot_variant_effects(variants, sequence=seq, motifs=gata, pvalue=PVALUE)
    labels = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "GATA lost" in labels and "GATA made" in labels
    plt.close(fig)
