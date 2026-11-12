import random
import sys

import polars as pl
import pytest
import universalmotif as um


def _genome(seed=1):
    rng = random.Random(seed)
    return {c: "".join(rng.choice("ACGT") for _ in range(n))
            for c, n in {"chr1": 5000, "chr2": 3000}.items()}


def _peaks(genome, n=60, seed=2):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        c = rng.choice(list(genome))
        s = rng.randint(0, len(genome[c]) - 300)
        rows.append((c, s, s + rng.randint(50, 299), rng.choice("+-*")))
    return rows


def test_genomic_hits_land_on_the_genome():
    genome = _genome()
    peaks = _peaks(genome)
    motifs = um.example_motifs()[:6]
    seqs = um.extract_ranges(genome, peaks)
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-3)
    gh = um.genomic_hits(hits)
    assert gh.height == hits.height > 0
    assert gh.columns[:5] == ["motif", "motif_i", "sequence", "sequence_i", "chrom"]
    whole = um.scan_sequences(motifs, genome, pvalue=1e-3).rename({"sequence": "chrom"})
    key = ["motif", "chrom", "start", "end", "strand"]
    joined = gh.join(whole.select(key + ["score"]), on=key, how="left", suffix="_genome")
    assert joined["score_genome"].null_count() == 0
    assert (joined["score"] - joined["score_genome"]).abs().max() < 1e-9
    # The ranges can be given instead of read from the names.
    plain = um.scan_sequences(motifs, list(seqs.values()), pvalue=1e-3)
    assert um.genomic_hits(plain, ranges=peaks).drop("sequence").equals(gh.drop("sequence"))


def test_genomic_hits_errors():
    hits = um.scan_sequences(um.example_motifs()[1], {"peak": "CCTGACTCACC"}, pvalue=1e-3)
    with pytest.raises(ValueError, match="chrom:start-end"):
        um.genomic_hits(hits)
    with pytest.raises(ValueError, match="beyond"):
        um.genomic_hits(hits, ranges=[])
    with pytest.raises(ValueError, match="'strand'"):
        um.genomic_hits(hits.drop("strand"))


def test_bed_strands_are_unstranded():
    genome = _genome()
    df = pl.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20], "strand": ["."]})
    assert um.extract_ranges(genome, df) == {"chr1:10-20": genome["chr1"][10:20]}


def test_pandas_ranges():
    pd = pytest.importorskip("pandas")
    genome = _genome()
    plain = pd.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20]})
    assert um.extract_ranges(genome, plain) == {"chr1:10-20": genome["chr1"][10:20]}
    # PyRanges' column names are understood too.
    named = pd.DataFrame({"Chromosome": ["chr1"], "Start": [10], "End": [20], "Strand": ["-"]})
    assert list(um.extract_ranges(genome, named)) == ["chr1:10-20(-)"]


def test_pyranges_round_trip():
    pr = pytest.importorskip("pyranges1")
    genome = _genome()
    peaks = pl.DataFrame(_peaks(genome), schema=["chrom", "start", "end", "strand"], orient="row")
    gr = um.to_pyranges(peaks.with_columns(pl.int_range(pl.len()).alias("id")))
    assert isinstance(gr, pr.PyRanges)
    assert list(gr.columns) == ["Chromosome", "Start", "End", "Strand", "id"]
    assert gr["Start"].tolist() == peaks["start"].to_list()
    assert um.extract_ranges(genome, gr) == um.extract_ranges(genome, peaks)
    hits = um.genomic_hits(um.scan_sequences(um.example_motifs()[:3],
                                             um.extract_ranges(genome, peaks), pvalue=1e-3))
    ghr = um.to_pyranges(hits)
    assert len(ghr) == hits.height
    anchors = [(c, (s + e) // 2, (s + e) // 2 + 1) for c, s, e, _ in peaks.iter_rows()]
    kw = {"anchors": anchors, "universe": {c: len(s) for c, s in genome.items()}, "qvalue": 1}
    assert um.motif_proximity(hits=ghr, **kw).equals(um.motif_proximity(hits=hits, **kw))
    with pytest.raises(ValueError, match="already has"):
        um.to_pyranges(peaks.with_columns(pl.lit(1).alias("Start")))


def test_polars_bio():
    pb = pytest.importorskip("polars_bio")
    genome = _genome()
    peaks = pl.DataFrame(_peaks(genome), schema=["chrom", "start", "end", "strand"], orient="row")
    hits = um.genomic_hits(um.scan_sequences(um.example_motifs()[:3],
                                             um.extract_ranges(genome, peaks), pvalue=1e-3))
    ready = um.to_polars_bio(hits)
    assert ready.columns[:3] == ["chrom", "start", "end"]
    assert pb.get_metadata(ready)["coordinate_system_zero_based"] is True
    overlaps = pb.overlap(ready, um.to_polars_bio(peaks), output_type="polars.DataFrame")
    assert overlaps.select("motif_1", "chrom_1", "start_1").unique().height == \
        hits.select("motif", "chrom", "start").unique().height
    # Frames marked as 1-based, eager or lazy, are moved to 0-based.
    one = peaks.with_columns(pl.col("start") + 1)
    one.config_meta.set(coordinate_system_zero_based=False)
    assert um.extract_ranges(genome, one) == um.extract_ranges(genome, peaks)
    lazy = one.lazy()
    lazy.config_meta.set(coordinate_system_zero_based=False)
    assert um.extract_ranges(genome, lazy) == um.extract_ranges(genome, peaks)


def _hits(rows, cols=("sequence", "start", "end", "strand")):
    return pl.DataFrame(rows, schema=list(cols), orient="row")


def test_genomic_hits_coordinates_by_hand():
    # A hit at 2-5 in a + range from 10 lands at 12-15; in a - range from
    # 5 to 15, read backwards, it lands at 15 - 5 = 10 to 15 - 2 = 13 on
    # the other strand.
    hits = _hits([("chrUn:alt:10-20", 2, 5, "+"), ("chr1:5-15(-)", 2, 5, "+"),
                  ("chr1:5-15(-).1", 0, 3, "-"), ("chr1:5-15.2", 2, 5, "*")])
    got = um.genomic_hits(hits)
    assert got.columns == ["sequence", "chrom", "start", "end", "strand"]
    # Colons in chromosome names and the suffixes of repeated ranges are
    # understood.
    assert got.select("chrom", "start", "end", "strand").rows() == [
        ("chrUn:alt", 12, 15, "+"), ("chr1", 10, 13, "-"), ("chr1", 12, 15, "+"),
        ("chr1", 7, 10, "*"),
    ]
    assert got["sequence"].to_list() == hits["sequence"].to_list()


def test_genomic_hits_matches_extracted_sequence():
    # A hit's genomic span reads, on its genomic strand, the match the scan
    # reported.
    genome = _genome(3)
    peaks = [("chr1", 100, 400, "+"), ("chr1", 100, 400, "-"), ("chr2", 50, 350, "*")]
    hits = um.scan_sequences(um.example_motifs()[:6], um.extract_ranges(genome, peaks), pvalue=1e-3)
    gh = um.genomic_hits(hits)
    assert gh.height > 0
    for c, s, e, st, match in gh.select("chrom", "start", "end", "strand", "match").iter_rows():
        assert um.extract_ranges(genome, [(c, s, e, st)]).popitem()[1] == match


def test_genomic_hits_with_ranges():
    hits = _hits([("s1", 0, 0, 2, "+"), ("s2", 1, 1, 4, "+"), ("s2", 1, 0, 1, "-")],
                 cols=("sequence", "sequence_i", "start", "end", "strand"))
    ranges = pl.DataFrame({"seqnames": ["c", "c"], "start": [100, 200], "width": [10, 10],
                           "strand": ["*", "-"]})
    got = um.genomic_hits(hits, ranges=ranges)
    assert got.select("chrom", "start", "end", "strand").rows() == [
        ("c", 100, 102, "+"), ("c", 206, 209, "-"), ("c", 209, 210, "+"),
    ]
    with pytest.raises(ValueError, match="'sequence_i' column"):
        um.genomic_hits(hits.drop("sequence_i"), ranges=ranges)
    with pytest.raises(ValueError, match="beyond the 2 ranges"):
        um.genomic_hits(hits.with_columns(pl.Series("sequence_i", [0, None, 1])), ranges=ranges)
    with pytest.raises(ValueError, match="beyond the 1 ranges"):
        um.genomic_hits(hits, ranges=[("c", 100, 110)])


def test_genomic_hits_replaces_a_chrom_column():
    hits = _hits([("chr1:10-20", 1, 3, "+")]).with_columns(pl.lit("old").alias("chrom"))
    got = um.genomic_hits(hits)
    assert got.columns == ["sequence", "chrom", "start", "end", "strand"]
    assert got.row(0) == ("chr1:10-20", "chr1", 11, 13, "+")


def test_to_pyranges_conversions(monkeypatch):
    pytest.importorskip("pyranges1")
    gr = um.to_pyranges([("c1", 5, 10, "-"), ("c2", 0, 3)])
    assert list(gr.columns) == ["Chromosome", "Start", "End", "Strand"]
    assert gr["Start"].tolist() == [5, 0] and gr["End"].tolist() == [10, 3]
    assert gr["Strand"].tolist() == ["-", "*"]
    # A table without strands has no Strand column.
    gr = um.to_pyranges(pl.DataFrame({"seqnames": ["c1"], "start": [1], "end": [2], "score": [0.5]}))
    assert list(gr.columns) == ["Chromosome", "Start", "End", "score"]
    with pytest.raises(ValueError, match="needs a chromosome column"):
        um.to_pyranges(pl.DataFrame({"chrom": ["c1"], "start": [1], "width": [2]}))
    monkeypatch.setitem(sys.modules, "pyranges1", None)
    with pytest.raises(ImportError, match=r"universalmotif\[pyranges\]"):
        um.to_pyranges([("c1", 5, 10)])


def test_to_polars_bio_conversions(monkeypatch):
    pb = pytest.importorskip("polars_bio")
    ready = um.to_polars_bio(pl.DataFrame({"score": [1.0], "end": [9], "start": [4],
                                           "seqnames": ["c1"]}))
    assert ready.columns == ["chrom", "start", "end", "score"]
    assert ready.row(0) == ("c1", 4, 9, 1.0)
    assert pb.get_metadata(ready)["coordinate_system_zero_based"] is True
    tuples = um.to_polars_bio([("c1", 4, 9, "-")])
    assert tuples.rows() == [("c1", 4, 9, "-")]
    assert pb.get_metadata(tuples)["coordinate_system_zero_based"] is True
    with pytest.raises(ValueError, match="needs a chromosome column"):
        um.to_polars_bio(pl.DataFrame({"chrom": ["c1"], "start": [1]}))
    monkeypatch.setitem(sys.modules, "polars_bio", None)
    with pytest.raises(ImportError, match=r"universalmotif\[polars-bio\]"):
        um.to_polars_bio([("c1", 5, 10)])


def test_polars_bio_one_based_round_trip():
    pb = pytest.importorskip("polars_bio")
    genome = _genome()
    # 1-based, closed positions 11 to 20 are 0-based, half-open 10 to 20.
    one = pl.DataFrame({"chrom": ["chr1"], "start": [11], "end": [20]})
    one.config_meta.set(coordinate_system_zero_based=False)
    assert um.extract_ranges(genome, one) == {"chr1:10-20": genome["chr1"][10:20]}
    # Widths are kept, so the end moves with the start.
    width = pl.DataFrame({"chrom": ["chr1"], "start": [11], "width": [10]})
    width.config_meta.set(coordinate_system_zero_based=False)
    assert list(um.extract_ranges(genome, width)) == ["chr1:10-20"]
    assert pb.get_metadata(um.to_polars_bio(one))["coordinate_system_zero_based"] is True
    assert um.to_polars_bio(one)["start"].to_list() == [10]


def test_read_bed_formats(tmp_path):
    bed = tmp_path / "x.bed"
    bed.write_text("track name=peaks\n# a comment\nbrowser position chr1:1-100\n\n"
                   "chr1\t10\t20\tp1\t5\t+\n2\t30\t45\t.\t.\t.\n")
    df = um.read_bed(bed)
    assert df.columns == ["chrom", "start", "end", "name", "score", "strand"]
    assert df["chrom"].to_list() == ["chr1", "2"]
    assert df["score"].dtype == pl.Int64 and df["score"].to_list() == [5, None]
    # Three columns, and twelve and more, by position.
    (tmp_path / "x3.bed").write_text("chr1\t0\t5\n")
    assert um.read_bed(tmp_path / "x3.bed").columns == ["chrom", "start", "end"]
    row = "chr1\t0\t100\tg\t960.5\t-\t10\t90\t0,0,255\t2\t10,20,\t0,80,\tx"
    (tmp_path / "x13.bed").write_text(row + "\n")
    df = um.read_bed(tmp_path / "x13.bed")
    assert df.columns[6:] == ["thick_start", "thick_end", "item_rgb", "block_count", "block_sizes",
                              "block_starts", "column_13"]
    assert df["score"].to_list() == [960.5] and df["block_count"].to_list() == [2]
    # narrowPeak and broadPeak, gzipped, with -1 read as missing.
    import gzip

    peaks = tmp_path / "x.narrowPeak.gz"
    peaks.write_bytes(gzip.compress(b"chr1\t100\t300\tp\t500\t.\t8.2\t12.5\t-1\t95\n"
                                    b"chr1\t400\t600\tq\t300\t.\t3.0\t-1\t-1\t-1\n"))
    df = um.read_bed(peaks)
    assert df.columns[6:] == ["signal", "neg_log10_pvalue", "neg_log10_qvalue", "peak"]
    assert df["neg_log10_pvalue"].to_list() == [12.5, None]
    assert df["peak"].to_list() == [95, None]
    (tmp_path / "x.broadPeak").write_text("chr1\t1\t9\tb\t0\t.\t2.5\t3.1\t1.2\n")
    assert um.read_bed(tmp_path / "x.broadPeak").columns[-1] == "neg_log10_qvalue"
    (tmp_path / "x.bg").write_text("chr1\t0\t10\t1.5\nchr1\t10\t20\t0\n")
    assert um.read_bed(tmp_path / "x.bg")["value"].to_list() == [1.5, 0.0]
    # The format can be given whatever the extension.
    (tmp_path / "peaks.txt").write_text("chr1\t100\t300\tp\t500\t.\t8.2\t12.5\t9.1\t95\n")
    assert um.read_bed(tmp_path / "peaks.txt", format="narrowPeak")["peak"].to_list() == [95]
    # Read as BED, columns that do not fit whole numbers are decimals.
    as_bed = um.read_bed(tmp_path / "peaks.txt")
    assert as_bed.columns[-1] == "block_count"
    assert as_bed["thick_start"].to_list() == [8.2]


def test_read_bed_ranges_work_downstream(tmp_path):
    genome = _genome()
    (tmp_path / "r.bed").write_text("chr1\t10\t20\tr1\t0\t-\nchr2\t0\t5\tr2\t0\t.\n")
    ranges = um.read_bed(tmp_path / "r.bed")
    seqs = um.extract_ranges(genome, ranges)
    assert len(seqs) == 2
    assert list(seqs.values())[1] == genome["chr2"][:5]
    masked = um.mask_ranges(genome, ranges)
    assert masked["chr1"][10:20] == "-" * 10


def test_read_bed_errors(tmp_path):
    (tmp_path / "empty.bed").write_text("# nothing\n\n")
    with pytest.raises(ValueError, match="no ranges"):
        um.read_bed(tmp_path / "empty.bed")
    (tmp_path / "two.bed").write_text("chr1\t5\n")
    with pytest.raises(ValueError, match="fewer than 3"):
        um.read_bed(tmp_path / "two.bed")
    (tmp_path / "short.narrowPeak").write_text("chr1\t1\t9\tb\t0\t.\n")
    with pytest.raises(ValueError, match="has 10 columns"):
        um.read_bed(tmp_path / "short.narrowPeak")
    with pytest.raises(ValueError, match="format must be"):
        um.read_bed(tmp_path / "two.bed", format="gff")
