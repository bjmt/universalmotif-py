import random

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
