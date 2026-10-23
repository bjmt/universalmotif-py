import random

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
