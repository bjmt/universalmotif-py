import random
import re
import struct
import sys

import numpy as np
import polars as pl
import pytest
import universalmotif as um


def _chroms(seed=1):
    rng = random.Random(seed)
    lens = {"chr1": 6000, "chr2": 2500, "chrM": 70}
    return {c: "".join(rng.choice("ACGTACGTacgtN") for _ in range(n)) for c, n in lens.items()}


def _fasta(chroms, width=60):
    out = []
    for name, seq in chroms.items():
        out.append(f">{name} a description\n")
        out += [seq[i:i + width] + "\n" for i in range(0, len(seq), width)]
    return "".join(out)


@pytest.fixture(params=["plain", "bgzip"])
def genome(request, tmp_path):
    chroms = _chroms()
    text = _fasta(chroms)
    if request.param == "plain":
        path = tmp_path / "g.fa"
        path.write_text(text)
    else:
        bgzf = pytest.importorskip("Bio.bgzf")
        path = tmp_path / "g.fa.gz"
        with bgzf.BgzfWriter(str(path), "wb") as f:
            f.write(text.encode())
    return um.Genome(path), chroms


def test_genome_is_a_mapping(genome):
    g, chroms = genome
    assert list(g) == list(chroms) and len(g) == 3
    assert "chr2" in g and "chr9" not in g and 1 not in g
    assert g.lengths == {c: len(s) for c, s in chroms.items()}
    assert g["chr2"] == chroms["chr2"]
    assert dict(g) == chroms
    assert g.subseq("chr1", 58, 122) == chroms["chr1"][58:122]
    with pytest.raises(KeyError):
        g["chr9"]
    with pytest.raises(IndexError):
        g.subseq("chr1", -1, 10)
    with pytest.raises(ValueError, match="outside"):
        g.subseq("chr1", 10, 6001)
    assert "3 sequences, 8,570 letters" in repr(g)


def test_extract_ranges(genome):
    g, chroms = genome
    rng = random.Random(2)
    ranges = []
    for _ in range(100):
        c = rng.choice(list(chroms))
        s = rng.randint(0, len(chroms[c]) - 1)
        ranges.append((c, s, rng.randint(s + 1, min(len(chroms[c]), s + 400)), rng.choice("+-*")))
    got = um.extract_ranges(g, ranges, nthreads=2)
    assert got == um.extract_ranges(chroms, ranges)
    # Packed sequences lose soft-masking.
    packed = um.extract_ranges(um.pack_sequences(chroms), ranges)
    assert [v.upper() for v in got.values()] == list(packed.values())
    # Minus-strand ranges are reverse complemented, keeping the case.
    seq = chroms["chr1"][99:110]
    rc = seq[::-1].translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))
    assert um.extract_ranges(g, [("chr1", 99, 110, "-")]) == {"chr1:99-110(-)": rc}
    df = pl.DataFrame({"seqnames": ["chr2", "chr2"], "start": [5, 5], "width": [10, 10]})
    assert list(um.extract_ranges(g, df)) == ["chr2:5-15", "chr2:5-15.1"]
    with pytest.raises(ValueError, match="not in the genome"):
        um.extract_ranges(g, [("chr9", 0, 5)])
    with pytest.raises(ValueError, match="does not fit"):
        um.extract_ranges(g, [("chrM", 60, 80)])


def test_genome_scans_like_a_dict(genome):
    g, chroms = genome
    motifs = um.example_motifs()[:4]
    want = um.scan_sequences(motifs, chroms, pvalue=1e-3)
    got = um.scan_sequences(motifs, g, pvalue=1e-3, nthreads=2)
    assert want.height > 0 and got.equals(want)
    anchors = [("chr1", 3000, 3000), ("chr2", 1200, 1200)]
    kw = {"anchors": anchors, "motifs": motifs, "scan_pvalue": 1e-3, "qvalue": 1, "nthreads": 1}
    assert um.motif_proximity(sequences=g, **kw).equals(um.motif_proximity(sequences=chroms, **kw))


def test_match_bkg_reads_only_windows(genome):
    g, chroms = genome
    targets = list(um.extract_ranges(chroms, [("chr1", 99, 199), ("chr2", 999, 1149)]).values())
    exclude = [("chr1", 99, 199)]
    a = um.match_bkg(targets, genome=g, exclude=exclude, n_candidates=200, seed=3)
    b = um.match_bkg(targets, genome=chroms, exclude=exclude, n_candidates=200, seed=3)
    assert a == b
    for name, seq in a.items():
        c, span = name.rsplit(":", 1)
        s, e = map(int, span.split("-"))
        assert chroms[c][s:e] == seq


def _bigwig(path, lengths, entries):
    pybigtools = pytest.importorskip("pybigtools")
    bw = pybigtools.open(str(path), "w")
    bw.write(lengths, entries)
    return path


def test_signal_stats(tmp_path):
    path = _bigwig(tmp_path / "s.bw", {"chr1": 1000, "chr2": 500},
                   [("chr1", 0, 100, 1.0), ("chr1", 100, 200, 3.0), ("chr2", 10, 20, 5.0)])
    ranges = [("chr1", 50, 250), ("chr1", 300, 400), ("chr2", 0, 500), ("chrX", 0, 10)]
    df = um.signal_stats(path, ranges)
    assert df.columns == ["signal"]
    assert np.allclose(df["signal"], [(50 * 1 + 100 * 3) / 200, 0, 50 / 500, 0])
    covered = um.signal_stats({"a": path, "b": path}, ranges, uncovered=None)
    assert covered.columns == ["a", "b"]
    assert np.allclose(covered["a"][:1], [350 / 150])
    assert np.isnan(covered["a"][1]) and np.isnan(covered["a"][3])
    assert um.signal_stats(path, ranges[:1], stat="max")["signal"][0] == 3.0
    assert um.signal_stats(path, ranges[:1], stat="min")["signal"][0] == 0.0
    with pytest.raises(ValueError, match="stat"):
        um.signal_stats(path, ranges, stat="median")
    with pytest.raises(ValueError, match="does not fit"):
        um.signal_stats(path, [("chr2", 400, 600)])


def test_match_bkg_on_signal(tmp_path):
    chroms = _chroms(5)
    lengths = {c: len(s) for c, s in chroms.items()}
    # Signal high over the first half of chr1, low elsewhere.
    entries = [(c, i, min(n, i + 50), 10.0 if c == "chr1" and i < 3000 else 0.5)
               for c, n in lengths.items() for i in range(0, n, 50)]
    path = _bigwig(tmp_path / "s.bw", lengths, entries)
    target_ranges = [("chr1", s, s + 100) for s in range(200, 2200, 200)]
    targets = list(um.extract_ranges(chroms, target_ranges).values())
    plain = um.match_bkg(targets, genome=chroms, n_candidates=500, seed=1, return_indices=True)
    matched = um.match_bkg(targets, genome=chroms, n_candidates=500, seed=1, return_indices=True,
                           bigwig={"atac": path}, target_ranges=target_ranges, n_bins_covariates=2,
                           bin_type_covariates="equalwidth")
    assert set(matched["target_atac"]) == {10.0}
    plain_ranges = [(n.split(":")[0], *map(int, n.split(":")[1].split("-")))
                    for n in plain["universe_name"]]
    assert matched["universe_atac"].mean() > 9
    assert um.signal_stats(path, plain_ranges)["signal"].mean() < matched["universe_atac"].mean() - 2
    with pytest.raises(ValueError, match="go together"):
        um.match_bkg(targets, genome=chroms, bigwig=path)
    with pytest.raises(ValueError, match="needs a genome"):
        um.match_bkg(targets, universe=targets, bigwig=path, target_ranges=target_ranges)
    with pytest.raises(ValueError, match="10 sequences"):
        um.match_bkg(targets, genome=chroms, bigwig=path, target_ranges=target_ranges[:3])


def test_genome_errors(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        um.Genome(tmp_path / "missing.fa")
    ragged = tmp_path / "ragged.fa"
    ragged.write_text(">a\nACGT\nAC\nACGT\n")
    with pytest.raises(ValueError, match="differ in length"):
        um.Genome(ragged)
    import gzip

    plain_gz = tmp_path / "plain.fa.gz"
    plain_gz.write_bytes(gzip.compress(b">a\nACGT\n"))
    with pytest.raises(ValueError, match="bgzip"):
        um.Genome(plain_gz)


def test_signal_values(tmp_path):
    path = _bigwig(tmp_path / "s.bw", {"chr1": 100},
                   [("chr1", 2, 5, 0.5), ("chr1", 5, 7, -1.0)])
    np.testing.assert_allclose(um.signal_values(path, "chr1", 0, 8),
                               [0, 0, 0.5, 0.5, 0.5, -1, -1, 0])
    v = um.signal_values(path, "chr1", 0, 3, uncovered=None)
    assert np.isnan(v[:2]).all() and v[2] == 0.5
    # A chromosome without values is uncovered, as in signal_stats().
    assert np.array_equal(um.signal_values(path, "chrX", 0, 3), [0, 0, 0])
    assert np.isnan(um.signal_values(path, "chrX", 0, 3, uncovered=None)).all()
    with pytest.raises(ValueError):
        um.signal_values(path, "chr1", 90, 110)


def _write_2bit(path, chroms):
    """A 2bit file written by hand from the UCSC format description, with N
    blocks for runs of N and mask blocks for lowercase runs."""
    code = {"T": 0, "C": 1, "A": 2, "G": 3}
    header = struct.pack("<IIII", 0x1A412743, 0, len(chroms), 0)
    offset = len(header) + sum(5 + len(name) for name in chroms)
    index, records = b"", b""
    for name, seq in chroms.items():
        rec = struct.pack("<I", len(seq))
        for pattern in ("[Nn]+", "[a-z]+"):
            blocks = [(m.start(), m.end() - m.start()) for m in re.finditer(pattern, seq)]
            rec += struct.pack(f"<{1 + 2 * len(blocks)}I", len(blocks), *[s for s, _ in blocks],
                               *[n for _, n in blocks])
        rec += struct.pack("<I", 0)
        up = seq.upper().ljust(-(-len(seq) // 4) * 4, "T")
        rec += bytes(
            (code.get(up[i], 0) << 6) | (code.get(up[i + 1], 0) << 4) | (code.get(up[i + 2], 0) << 2)
            | code.get(up[i + 3], 0)
            for i in range(0, len(up), 4)
        )
        index += struct.pack("<B", len(name)) + name.encode() + struct.pack("<I", offset + len(records))
        records += rec
    path.write_bytes(header + index + records)
    return path


def test_genome_2bit(tmp_path):
    chroms = {"chr1": "ACGTNNNNacgtTTGCAn", "chrX": "GATTACA", "chrM": "nnnnACGTacgt"}
    g = um.Genome(_write_2bit(tmp_path / "g.2bit", chroms))
    # Soft-masking and runs of N come back as written.
    assert dict(g) == chroms
    assert g.lengths == {"chr1": 18, "chrX": 7, "chrM": 12}
    assert g.subseq("chr1", 3, 9) == "TNNNNa"
    assert g.subseq("chrX", 0, 0) == ""
    assert um.extract_ranges(g, [("chrX", 1, 5, "-"), ("chrM", 2, 6)]) == {
        "chrX:1-5(-)": "TAAT", "chrM:2-6": "nnAC",
    }
    assert repr(g) == "Genome('g.2bit', 3 sequences, 37 letters)"


def test_genome_with_samtools_index(tmp_path):
    chroms = _chroms(4)
    path = tmp_path / "g.fa"
    path.write_text(_fasta(chroms, width=50))
    # The index samtools faidx writes: name, length, offset of the first
    # base, bases per line and bytes per line.
    rows, offset = [], 0
    for name, seq in chroms.items():
        offset += len(f">{name} a description\n")
        rows.append(f"{name}\t{len(seq)}\t{offset}\t50\t51\n")
        offset += len(seq) + -(-len(seq) // 50)
    (tmp_path / "g.fa.fai").write_text("".join(rows))
    g = um.Genome(path)
    assert dict(g) == chroms
    assert g.subseq("chr2", 49, 151) == chroms["chr2"][49:151]


def test_genome_extract_and_keys(genome):
    g, chroms = genome
    ranges = pl.DataFrame({"chrom": ["chrM", "chr1"], "start": [0, 10], "end": [70, 20],
                           "strand": ["-", "+"]})
    assert g.extract(ranges) == um.extract_ranges(chroms, ranges)
    assert g.extract(ranges, nthreads=2) == um.extract_ranges(g, ranges)
    # Only strings are chromosome names.
    with pytest.raises(KeyError):
        g[0]
    with pytest.raises(KeyError):
        g.subseq("chr9", 0, 1)
    assert g.get("chr9") is None


@pytest.mark.xfail(strict=True, reason="a FASTA whose last line has no newline cannot be opened, "
                                       "although samtools faidx indexes it")
def test_genome_without_final_newline(tmp_path):
    path = tmp_path / "g.fa"
    path.write_text(">a\nACGT\nAC")
    assert dict(um.Genome(path)) == {"a": "ACGTAC"}


def test_genome_unusual_fasta(tmp_path):
    path = tmp_path / "g.fa"
    # Empty sequences, blank lines and Windows line endings.
    path.write_text(">a\n>b\r\nACGT\r\nAC\r\n\n>c\nGG\n")
    g = um.Genome(path)
    assert dict(g) == {"a": "", "b": "ACGTAC", "c": "GG"}
    assert g.subseq("b", 3, 5) == "TA"


def test_extract_ranges_strands_and_bounds():
    genome = {"c": "ACGTRYKMSWBDHVNacgtu"}
    # Every strand but - reads forwards; - is reverse complemented with
    # IUPAC codes and case kept.
    out = um.extract_ranges(genome, [("c", 0, 2, "+"), ("c", 0, 2, "*"), ("c", 0, 2, "."),
                                     ("c", 0, 20, "-")])
    assert out == {"c:0-2": "AC", "c:0-2.1": "AC", "c:0-2.2": "AC",
                   "c:0-20(-)": "aacgtNBDHVWSKMRYACGT"}
    # Positions are 0-based and half-open: 0-1 is the first base, and an
    # empty range is allowed.
    assert um.extract_ranges(genome, [("c", 0, 1), ("c", 19, 20), ("c", 4, 4)]) == {
        "c:0-1": "A", "c:19-20": "u", "c:4-4": "",
    }
    for bad in [("c", -1, 3), ("c", 5, 3), ("c", 18, 21)]:
        with pytest.raises(ValueError, match="does not fit in c"):
            um.extract_ranges(genome, [bad])
    packed = um.pack_sequences({"c": "ACGTTGCA"})
    assert um.extract_ranges(packed, [("c", 1, 4, "-")]) == {"c:1-4(-)": "ACG"}
    with pytest.raises(ValueError, match="does not fit in c"):
        um.extract_ranges(packed, [("c", 6, 9)])


def test_signal_stats_edges(tmp_path, monkeypatch):
    path = _bigwig(tmp_path / "s.bw", {"chr1": 100},
                   [("chr1", 0, 10, 2.0), ("chr1", 10, 20, -4.0)])
    ranges = pl.DataFrame({"chr": ["chr1", "chr1", "chr1"], "start": [5, 30, 0],
                           "end": [15, 30, 20], "strand": ["-", "+", "*"]})
    got = um.signal_stats(path, ranges)
    # The strand does not matter, and an empty range has no signal.
    assert got["signal"][0] == pytest.approx(-1.0)
    assert np.isnan(got["signal"][1])
    assert got["signal"][2] == pytest.approx(-1.0)
    assert um.signal_stats(path, ranges, stat="max")["signal"][0] == 2.0
    assert um.signal_stats(path, ranges, stat="min")["signal"][2] == -4.0
    # Uncovered bases pull the mean towards their value.
    wide = [("chr1", 0, 40)]
    assert um.signal_stats(path, wide)["signal"][0] == pytest.approx(-20 / 40)
    assert um.signal_stats(path, wide, uncovered=1.0)["signal"][0] == pytest.approx(0 / 40)
    assert um.signal_stats(path, wide, uncovered=None)["signal"][0] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="no bigWig files"):
        um.signal_stats({}, wide)
    with pytest.raises(ValueError, match="does not fit"):
        um.signal_stats(path, [("chr1", -5, 10)])
    monkeypatch.setitem(sys.modules, "pybigtools", None)
    with pytest.raises(ImportError, match=r"universalmotif\[bigwig\]"):
        um.signal_stats(path, wide)
    with pytest.raises(ImportError, match=r"universalmotif\[bigwig\]"):
        um.signal_values(path, "chr1", 0, 5)


def test_signal_values_edges(tmp_path):
    path = _bigwig(tmp_path / "s.bw", {"chr1": 10}, [("chr1", 8, 10, 7.0)])
    # The last bases of a chromosome are reachable, and positions are
    # 0-based.
    np.testing.assert_allclose(um.signal_values(path, "chr1", 7, 10), [0, 7, 7])
    np.testing.assert_allclose(um.signal_values(path, "chr1", 7, 10, uncovered=-1), [-1, 7, 7])
    assert um.signal_values(path, "chr1", 8, 9).dtype == np.float64
    for start, end in [(-1, 3), (5, 5), (6, 2)]:
        with pytest.raises(ValueError, match="empty or starts before 0"):
            um.signal_values(path, "chr1", start, end)
    with pytest.raises(ValueError, match=r"does not fit in chr1 \(length 10\)"):
        um.signal_values(path, "chr1", 5, 11)
