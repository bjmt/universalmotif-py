import gzip
import io
import itertools
import os
import random
import runpy
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import polars as pl
import pytest
import universalmotif as um
from universalmotif._cli import main, read_ranges

DATA = Path(um.__file__).parent / "data"
MOTIFS = str(DATA / "example_motifs.transfac")
TARGETS = str(DATA / "example_targets.fa.gz")
CONTROLS = str(DATA / "example_controls.fa.gz")


def run(*args, ok=True):
    code = main([str(a) for a in args])
    assert (code == 0) == ok, code
    return code


def table(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", null_values="NA")


def names(motifs):
    return [m.name for m in motifs]


@pytest.fixture
def genome(tmp_path):
    rng = random.Random(1)
    chroms = {c: "".join(rng.choice("ACGT") for _ in range(n))
              for c, n in {"chr1": 6000, "chr2": 4000}.items()}
    # Plant an AP-1 site near the middle of each peak.
    peaks = []
    for c, seq in chroms.items():
        for start in range(300, len(seq) - 500, 700):
            mid = start + 100
            seq = seq[:mid] + "TGACTCA" + seq[mid + 7:]
            peaks.append((c, start, start + 200))
        chroms[c] = seq
    fasta = tmp_path / "genome.fa"
    um.write_fasta(chroms, fasta)
    bed = tmp_path / "peaks.bed"
    bed.write_text("".join(f"{c}\t{s}\t{e}\tp{i}\t0\t{'+-'[i % 2]}\n"
                           for i, (c, s, e) in enumerate(peaks)))
    return fasta, bed, chroms, peaks


def test_convert_and_format_detection(tmp_path):
    want = um.read_transfac(MOTIFS)
    out = tmp_path / "m.jaspar"
    run("convert", MOTIFS, "-o", out)
    assert names(um.read_jaspar(out)) == names(want)
    # Contents decide the format when the extension does not.
    for src, reader in [("example.meme", um.read_meme), ("example.jaspar", um.read_jaspar),
                        ("example.motif", um.read_homer), ("example.transfac", um.read_transfac),
                        ("example.cisbp", um.read_cisbp), ("example.uniprobe", um.read_uniprobe)]:
        plain = tmp_path / f"{src}.txt"
        plain.write_text((DATA / src).read_text())
        out = tmp_path / f"{src}.meme"
        run("convert", plain, "-o", out)
        assert names(um.read_meme(out)) == names(reader(DATA / src)), src
    run("convert", MOTIFS, "--type", "PWM", "-o", tmp_path / "pwm.meme")
    # MEME files hold probabilities, whatever the type asked for.
    assert {m.type for m in um.read_meme(tmp_path / "pwm.meme")} == {"PPM"}
    run("convert", MOTIFS, "--rc", "-o", tmp_path / "rc.jaspar")
    rc = um.read_jaspar(tmp_path / "rc.jaspar")
    assert rc[1].consensus == um.motif_rc(want[1]).consensus


def test_motifs_from_standard_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO((DATA / "example.jaspar").read_bytes())))
    run("summary", "-")
    got = pl.read_csv(io.StringIO(capsys.readouterr().out), separator="\t", null_values="NA")
    assert got["name"].to_list() == names(um.read_jaspar(DATA / "example.jaspar"))


def test_summary_filter_trim(tmp_path):
    motifs = um.read_transfac(MOTIFS)
    run("summary", MOTIFS, "-o", tmp_path / "s.tsv")
    want = um.summarize_motifs(motifs)
    got = table(tmp_path / "s.tsv")
    assert got["name"].to_list() == want["name"].to_list()
    assert got["consensus"].to_list() == want["consensus"].to_list()
    run("filter", MOTIFS, "--family", "bZIP", "--family", "ETS", "--icscore", "10",
        "-o", tmp_path / "f.meme")
    assert names(um.read_meme(tmp_path / "f.meme")) == \
        names(um.filter_motifs(motifs, family=["bZIP", "ETS"], icscore=10))
    run("filter", MOTIFS, "--regex", "^G", "-o", tmp_path / "g.meme")
    assert names(um.read_meme(tmp_path / "g.meme")) == ["G-box", "GATA", "GC-box"]
    run("filter", MOTIFS, "--family", "nothing", ok=False)
    run("trim", MOTIFS, "--min-ic", "0.5", "-o", tmp_path / "t.meme")
    assert [m.width for m in um.read_meme(tmp_path / "t.meme")] == \
        [m.width for m in um.trim_motifs(motifs, min_ic=0.5)]


def test_merge_and_tree(tmp_path):
    motifs = um.read_transfac(MOTIFS)
    run("merge", MOTIFS, "-q", "0.5", "--clusters", tmp_path / "c.tsv", "-o", tmp_path / "m.meme")
    assert names(um.read_meme(tmp_path / "m.meme")) == names(um.merge_similar(motifs, qvalue=0.5))
    clusters = table(tmp_path / "c.tsv")
    assert clusters["cluster"].to_list() == \
        um.merge_similar(motifs, qvalue=0.5, return_clusters=True)["cluster"].to_list()
    run("merge", MOTIFS, "--all", "--name", "everything", "-o", tmp_path / "all.meme")
    assert names(um.read_meme(tmp_path / "all.meme")) == ["everything"]
    run("tree", MOTIFS, "-o", tmp_path / "t.nwk")
    newick = (tmp_path / "t.nwk").read_text()
    assert newick.strip().endswith(";") and all(n in newick for n in names(motifs))
    run("tree", MOTIFS, "-k", "3", "-o", tmp_path / "k.tsv")
    assert table(tmp_path / "k.tsv")["cluster"].to_list() == \
        um.motif_tree(motifs).cutree(k=3).tolist()


def test_logos(tmp_path):
    run("logo", MOTIFS, "--each", tmp_path / "svg")
    files = sorted(p.name for p in (tmp_path / "svg").iterdir())
    assert len(files) == 12 and "AP-1.svg" in files
    assert (tmp_path / "svg" / "AP-1.svg").read_text().startswith("<svg")
    run("logo", MOTIFS, ok=False)
    pytest.importorskip("matplotlib")
    run("logo", MOTIFS, "-o", tmp_path / "all.png")
    assert (tmp_path / "all.png").read_bytes()[:4] == b"\x89PNG"


def test_scan(tmp_path, genome):
    motifs = um.read_transfac(MOTIFS)
    run("scan", MOTIFS, "-s", TARGETS, "-o", tmp_path / "h.tsv")
    want = um.scan_sequences(motifs, um.example_sequences("targets"))
    got = table(tmp_path / "h.tsv")
    assert got.select("motif", "sequence", "start", "end", "strand").rows() == \
        want.select("motif", "sequence", "start", "end", "strand").rows()
    fasta, bed, chroms, _peaks = genome
    ap1 = DATA / "example.jaspar"
    run("scan", ap1, "-g", fasta, "-r", bed, "-o", tmp_path / "p.tsv")
    hits = table(tmp_path / "p.tsv").filter(pl.col("motif") == "AP-1")
    # Peaks from the BED file are 0-based; hits come back on the genome.
    for c, s, e in hits.select("chrom", "start", "end").iter_rows():
        assert "TGACTCA" in chroms[c][s - 1:e] or "TGAGTCA" in chroms[c][s - 1:e]
    whole = tmp_path / "w.tsv"
    run("scan", ap1, "-g", fasta, "-o", whole)
    key = ["motif", "chrom", "start", "end", "strand"]
    w = table(whole).rename({"sequence": "chrom"})
    assert table(tmp_path / "p.tsv").join(w, on=key, how="anti").height == 0
    run("scan", ap1, "-g", fasta, "-r", bed, "--local", "-o", tmp_path / "l.tsv")
    assert "chrom" not in table(tmp_path / "l.tsv").columns
    run("scan", MOTIFS, ok=False)
    run("scan", MOTIFS, "-s", tmp_path / "missing.fa", ok=False)


def test_compare(tmp_path):
    motifs = um.read_transfac(MOTIFS)
    run("compare", MOTIFS, "-o", tmp_path / "c.tsv")
    got = table(tmp_path / "c.tsv")
    assert (got["query_i"] != got["target_i"]).all()
    assert got.columns[:4] == ["query", "query_i", "target", "target_i"]
    run("compare", DATA / "example.meme", "--db", MOTIFS, "-o", tmp_path / "d.tsv")
    db = table(tmp_path / "d.tsv")
    assert set(db["target"]) <= set(names(motifs))
    assert db.filter(pl.col("query") == "AP-1")["target"][0] == "AP-1"
    run("compare", MOTIFS, "--matrix", "score", "-o", tmp_path / "m.tsv")
    mat = table(tmp_path / "m.tsv")
    assert mat.shape == (12, 13) and mat["motif"].to_list() == names(motifs)


def test_enrich_find_shuffle(tmp_path):
    motifs = um.read_transfac(MOTIFS)
    run("enrich", MOTIFS, "-s", TARGETS, "-b", CONTROLS, "-o", tmp_path / "e.tsv")
    want = um.enrich_motifs(motifs, um.example_sequences("targets"), um.example_sequences("controls"))
    assert table(tmp_path / "e.tsv")["motif"].to_list() == want["motif"].to_list()
    run("find", "-s", TARGETS, "-b", CONTROLS, "--seed", "1", "--max-width", "8",
        "--table", tmp_path / "f.tsv", "-o", tmp_path / "f.meme")
    found = um.read_meme(tmp_path / "f.meme")
    assert names(found) == table(tmp_path / "f.tsv")["name"].to_list() and found
    run("shuffle", "-s", TARGETS, "-k", "2", "--seed", "3", "-o", tmp_path / "s.fa")
    shuffled = um.read_fasta(tmp_path / "s.fa")
    assert shuffled == um.shuffle_sequences(um.example_sequences("targets"), k=2, seed=3)


def test_extract_and_bkg(tmp_path, genome):
    fasta, bed, chroms, peaks = genome
    run("extract", "-g", fasta, "-r", bed, "-o", tmp_path / "x.fa")
    seqs = um.read_fasta(tmp_path / "x.fa")
    assert list(seqs)[:2] == [f"chr1:{peaks[0][1]}-{peaks[0][2]}",
                              f"chr1:{peaks[1][1]}-{peaks[1][2]}(-)"]
    assert next(iter(seqs.values())) == chroms["chr1"][peaks[0][1]:peaks[0][2]]
    run("bkg", "-g", fasta, "-r", bed, "--seed", "1", "--candidates", "500",
        "--table", tmp_path / "b.tsv", "-o", tmp_path / "b.fa")
    bkg = um.read_fasta(tmp_path / "b.fa")
    assert len(bkg) == len(peaks)
    for name, seq in bkg.items():
        c, span = name.rsplit(":", 1)
        s, e = map(int, span.split("-"))
        assert chroms[c][s:e] == seq
        # Windows keep clear of the peaks.
        assert not any(pc == c and s < pe and e > ps for pc, ps, pe in peaks)
    assert table(tmp_path / "b.tsv").height == len(peaks)
    run("bkg", "-s", TARGETS, "-u", CONTROLS, "--seed", "1", "-o", tmp_path / "u.fa")
    assert list(um.read_fasta(tmp_path / "u.fa")) == \
        list(um.match_bkg(um.example_sequences("targets"), um.example_sequences("controls"), seed=1))
    run("bkg", "-s", TARGETS, ok=False)


def test_matrix(tmp_path, genome):
    run("matrix", MOTIFS, "-s", TARGETS, "-o", tmp_path / "m.tsv")
    want = um.match_motifs(um.read_transfac(MOTIFS), um.example_sequences("targets")).to_polars()
    assert table(tmp_path / "m.tsv").select("sequence", "motif").rows() == \
        want.select("sequence", "motif").rows()
    pytest.importorskip("scipy")
    run("matrix", MOTIFS, "-s", TARGETS, "--value", "counts", "-o", tmp_path / "m.mtx")
    assert (tmp_path / "m.mtx").read_text().startswith("%%MatrixMarket")
    assert len((tmp_path / "m.motifs.txt").read_text().split()) == 12
    assert len((tmp_path / "m.sequences.txt").read_text().split()) == 100
    pytest.importorskip("anndata")
    import anndata

    run("matrix", MOTIFS, "-s", TARGETS, "-o", tmp_path / "m.h5ad")
    assert anndata.read_h5ad(tmp_path / "m.h5ad").shape == (100, 12)


def test_proximity_coocc_posenr(tmp_path, genome):
    fasta, _bed, _chroms, peaks = genome
    summits = tmp_path / "summits.bed"
    summits.write_text("".join(f"{c}\t{s + 103}\t{s + 104}\n" for c, s, _ in peaks))
    ap1 = DATA / "example.jaspar"
    run("proximity", ap1, "-a", summits, "-g", fasta, "-q", "1", "-o", tmp_path / "p.tsv")
    prox = table(tmp_path / "p.tsv")
    assert prox.filter(pl.col("motif") == "AP-1")["hits_near"][0] > 0
    assert "dist_to_nearest" not in prox.columns
    run("scan", ap1, "-g", fasta, "-o", tmp_path / "h.tsv")
    run("proximity", "-a", summits, "--hits", tmp_path / "h.tsv", "-g", fasta, "-q", "1",
        "--nearest", "-o", tmp_path / "p2.tsv")
    p2 = table(tmp_path / "p2.tsv")
    assert p2.drop("dist_to_nearest").equals(prox)
    run("coocc", MOTIFS, "-s", TARGETS, "-o", tmp_path / "c.tsv")
    want = um.motif_coocc(um.read_transfac(MOTIFS), um.example_sequences("targets"))
    assert table(tmp_path / "c.tsv").select("motif_a", "motif_b").rows() == \
        want.select("motif_a", "motif_b").rows()
    run("posenr", MOTIFS, "-s", TARGETS, "-o", tmp_path / "e.tsv")
    pe = table(tmp_path / "e.tsv")
    assert pe["motif"].to_list() == ["W-box"] and pe["motif_i"].to_list() == [5]
    assert "centres" not in pe.columns
    run("posenr", "--hits", tmp_path / "h.tsv", ok=False)


def test_fetch(tmp_path, monkeypatch):
    from universalmotif import databases

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return um.example_motifs()[:2]

    monkeypatch.setattr(databases, "jaspar", fake)
    run("fetch", "jaspar", "--tax-group", "plants", "--release", "2024", "-o", tmp_path / "j.meme")
    assert seen["tax_group"] == "plants" and seen["release"] == 2024
    assert names(um.read_meme(tmp_path / "j.meme")) == ["TATA-box", "AP-1"]


def test_ranges_files(tmp_path):
    bed = tmp_path / "r.txt"
    bed.write_text("track name=x\n#comment\nchr1\t0\t10\nchr2\t5\t8\n")
    df = read_ranges(str(bed))
    assert df.select("chrom", "start", "end").rows() == [("chr1", 0, 10), ("chr2", 5, 8)]
    tsv = tmp_path / "r.tsv"
    tsv.write_text("chrom\tstart\tend\tstrand\nchr1\t1\t10\t-\n")
    assert read_ranges(str(tsv)).row(0) == ("chr1", 1, 10, "-")


def test_errors_are_short(tmp_path, capsys):
    run("convert", tmp_path / "missing.meme", ok=False)
    err = capsys.readouterr().err
    assert "convert: error: no such file" in err
    assert "Traceback" not in err


def test_entry_points():
    out = subprocess.run([sys.executable, "-m", "universalmotif", "--version"],
                         capture_output=True, text=True, check=True).stdout
    assert out.startswith("universalmotif ")
    bindir = Path(sys.executable).parent
    for name in ("universalmotif", "umotif"):
        exe = bindir / name
        if exe.exists():
            res = subprocess.run([str(exe), "summary", MOTIFS], capture_output=True, text=True,
                                 check=True)
            assert res.stdout.startswith("name\t")


# Paths the tests above leave out ---------------------------------------------


def revcomp(seq):
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))


def test_convert_options_and_outputs(tmp_path, capsys):
    motifs = um.read_transfac(MOTIFS)
    # A bare matrix file has nothing to recognize it by but its numbers.
    plain = tmp_path / "plain.txt"
    plain.write_text((DATA / "example.matrix").read_text())
    run("convert", plain, "-o", tmp_path / "plain.jaspar")
    assert names(um.read_jaspar(tmp_path / "plain.jaspar")) == \
        names(um.read_matrix(DATA / "example.matrix"))
    # Output ending in .gz is compressed, and its format comes from the
    # extension before the .gz; reading it back works the same way.
    run("convert", MOTIFS, "-o", tmp_path / "m.jaspar.gz")
    raw = (tmp_path / "m.jaspar.gz").read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert gzip.decompress(raw).decode().startswith(">TATA-box\nA [")
    run("summary", tmp_path / "m.jaspar.gz")
    got = pl.read_csv(io.StringIO(capsys.readouterr().out), separator="\t", null_values="NA")
    assert got["name"].to_list() == names(motifs)
    # -f and -t win over the extensions.
    mislabelled = tmp_path / "jaspar.meme"
    mislabelled.write_text((DATA / "example.jaspar").read_text())
    run("convert", mislabelled, "-f", "jaspar", "-t", "transfac", "-o", tmp_path / "t.meme")
    assert names(um.read_transfac(tmp_path / "t.meme")) == \
        names(um.read_jaspar(DATA / "example.jaspar"))
    # Standard output gets MEME by default.
    run("convert", DATA / "example.jaspar")
    out = capsys.readouterr().out
    assert out.startswith("MEME version")
    # The writers refuse to overwrite files, but the command line does not.
    target = tmp_path / "again.meme"
    target.write_text("old")
    run("convert", MOTIFS, "-o", target)
    assert names(um.read_meme(target)) == names(motifs)
    run("convert", MOTIFS, "--switch-alph", "-o", tmp_path / "rna.meme")
    rna = um.read_meme(tmp_path / "rna.meme")
    assert {m.alphabet for m in rna} == {"RNA"}
    assert [m.consensus for m in rna] == [m.consensus.replace("T", "U") for m in motifs]
    run("convert", MOTIFS, "--type", "PCM", "-o", tmp_path / "pcm.jaspar")
    assert {m.type for m in um.read_jaspar(tmp_path / "pcm.jaspar")} == {"PCM"}


def _cisbp_zip(path, status="D"):
    info = (
        "TF_ID\tFamily_ID\tMotif_ID\tDBID\tTF_Name\tTF_Species\tTF_Status\tFamily_Name\n"
        f"T1\tF1\tM001_3.10\tAT1\tWRKY1\tSome_species\t{status}\tWRKY\n"
        f"T2\tF2\tM002_3.10\tAT2\tERF1\tSome_species\t{status}\tAP2\n"
    )
    pwm = "Pos\tA\tC\tG\tT\n1\t0.1\t0.7\t0.1\t0.1\n2\t0.9\t0.0\t0.05\t0.05\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("TF_Information_all_motifs.txt", info)
        z.writestr("pwms_all_motifs/M001_3.10.txt", pwm)
        z.writestr("pwms_all_motifs/M002_3.10.txt", pwm)
    return path


def test_cisbp_archive_input(tmp_path, capsys):
    archive = _cisbp_zip(tmp_path / "Some_species.ZIP")
    run("convert", archive, DATA / "example.jaspar", "-o", tmp_path / "c.meme")
    got = um.read_meme(tmp_path / "c.meme")
    assert names(got) == ["M001_3.10", "M002_3.10"] + names(um.read_jaspar(DATA / "example.jaspar"))
    assert got[0].consensus == "CA"
    # An archive whose motifs are all inferred holds nothing by default.
    run("summary", _cisbp_zip(tmp_path / "inferred.zip", status="I"), ok=False)
    assert "summary: error: no motifs were read" in capsys.readouterr().err


@pytest.mark.parametrize("args, missing", [
    (["shuffle", "-s", "{tmp}/nope.fa"], "nope.fa"),
    (["extract", "-g", "{tmp}/nope.fa", "-r", "{tmp}/r.bed"], "nope.fa"),
    (["extract", "-g", "{tmp}/g.fa", "-r", "{tmp}/nope.bed"], "nope.bed"),
    (["coocc", MOTIFS, "--hits", "{tmp}/nope.tsv", "--n-sequences", "3"], "nope.tsv"),
    (["proximity", "-a", "{tmp}/r.bed", "--hits", "{tmp}/nope.tsv"], "nope.tsv"),
    (["proximity", "-a", "{tmp}/r.bed", "--hits", "{tmp}/nope.tsv",
      "--universe", "{tmp}/nope.bed"], "nope.bed"),
])
def test_missing_files(tmp_path, capsys, args, missing):
    um.write_fasta({"chr1": "ACGT" * 10}, tmp_path / "g.fa")
    (tmp_path / "r.bed").write_text("chr1\t0\t10\n")
    run(*[a.format(tmp=tmp_path) for a in args], ok=False)
    err = capsys.readouterr().err
    assert err.startswith(f"universalmotif {args[0]}: error: no such file: ")
    assert err.rstrip().endswith(missing) and "Traceback" not in err


def test_empty_ranges(tmp_path, capsys):
    um.write_fasta({"chr1": "ACGT" * 10}, tmp_path / "g.fa")
    bed = tmp_path / "r.bed"
    bed.write_text("track name=x\n# nothing here\n\n")
    run("extract", "-g", tmp_path / "g.fa", "-r", bed, ok=False)
    assert f"error: no ranges in {bed}" in capsys.readouterr().err


def test_sequences_from_standard_input(tmp_path, monkeypatch, capsys):
    raw = (DATA / "example_targets.fa.gz").read_bytes()
    seqs = um.example_sequences("targets")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))
    run("shuffle", "-s", "-", "-k", "2", "--seed", "3", "-o", tmp_path / "s.fa")
    assert um.read_fasta(tmp_path / "s.fa") == um.shuffle_sequences(seqs, k=2, seed=3)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))
    run("scan", MOTIFS, "-s", "-", "-o", tmp_path / "h.tsv")
    want = um.scan_sequences(um.read_transfac(MOTIFS), seqs)
    assert table(tmp_path / "h.tsv").select("motif", "sequence", "start", "strand").rows() == \
        want.select("motif", "sequence", "start", "strand").rows()


def test_scan_options(tmp_path, genome, capsys):
    fasta, _bed, chroms, _peaks = genome
    # Sequences named after their place in the genome, one of them from
    # the minus strand.
    seqs = {"chr1:100-200": chroms["chr1"][100:200],
            "chr2:5-105(-)": revcomp(chroms["chr2"][5:105])}
    um.write_fasta(seqs, tmp_path / "named.fa")
    run("scan", MOTIFS, "-s", tmp_path / "named.fa", "-P", "0.01", "-o", tmp_path / "local.tsv")
    run("scan", MOTIFS, "-s", tmp_path / "named.fa", "-P", "0.01", "--genomic",
        "-o", tmp_path / "genomic.tsv")
    local, genomic = table(tmp_path / "local.tsv"), table(tmp_path / "genomic.tsv")
    assert local.height == genomic.height > 0
    for lo, gen in zip(local.iter_rows(named=True), genomic.iter_rows(named=True)):
        if lo["sequence"].startswith("chr1"):
            assert (gen["chrom"], gen["start"], gen["end"], gen["strand"]) == \
                ("chr1", lo["start"] + 100, lo["end"] + 100, lo["strand"])
        else:
            assert (gen["chrom"], gen["start"], gen["end"]) == \
                ("chr2", 105 - lo["end"], 105 - lo["start"])
            assert gen["strand"] != lo["strand"]
        assert chroms[gen["chrom"]][gen["start"]:gen["end"]] == \
            (gen["match"] if gen["strand"] == "+" else revcomp(gen["match"]))
    run("scan", MOTIFS, "-s", TARGETS, "--no-rc", "-o", tmp_path / "fwd.tsv")
    assert set(table(tmp_path / "fwd.tsv")["strand"]) == {"+"}
    run("scan", MOTIFS, "-s", TARGETS, "-P", "1e-3", "--no-overlaps", "-o", tmp_path / "no.tsv")
    run("scan", MOTIFS, "-s", TARGETS, "-P", "1e-3", "-o", tmp_path / "all.tsv")
    kept = table(tmp_path / "no.tsv")
    assert 0 < kept.height < table(tmp_path / "all.tsv").height
    for _key, grp in kept.group_by("motif", "sequence", "strand"):
        spans = sorted(grp.select("start", "end").rows())
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))
    run("scan", MOTIFS, "-s", TARGETS, "-g", fasta, ok=False)
    assert "either with -s, or with -g" in capsys.readouterr().err


def test_warnings_are_one_line(tmp_path, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        run("scan", MOTIFS, "-s", TARGETS, "-P", "1e-15", "-o", tmp_path / "h.tsv")
    err = capsys.readouterr().err
    assert err.startswith("warning: 12 motifs cannot reach a P-value of 1e-15")
    assert err.count("\n") == 1 and ".py" not in err
    assert table(tmp_path / "h.tsv").height == 0


def test_compare_options(tmp_path, capsys):
    run("compare", MOTIFS, "--all", "-o", tmp_path / "all.tsv")
    every = table(tmp_path / "all.tsv")
    assert every.height == 12 * 11
    assert set(every.select("query_i", "target_i").rows()) == \
        {(i, j) for i in range(12) for j in range(12) if i != j}
    run("compare", MOTIFS, "-q", "0.05", "-o", tmp_path / "q.tsv")
    some = table(tmp_path / "q.tsv")
    assert 0 < some.height < every.height and (some["qvalue"] < 0.05).all()
    run("compare", MOTIFS, "--matrix", "pvalue", "-o", tmp_path / "p.tsv")
    mat = table(tmp_path / "p.tsv")
    assert mat.columns[1:] == mat["motif"].to_list()
    run("compare", MOTIFS, "--matrix", "score", "--db", MOTIFS, ok=False)
    assert "not with --db" in capsys.readouterr().err


def test_targets_from_genome(tmp_path, genome, capsys):
    fasta, bed, _chroms, peaks = genome
    ap1 = DATA / "example.jaspar"
    run("extract", "-g", fasta, "-r", bed, "-o", tmp_path / "peaks.fa")
    run("enrich", ap1, "-g", fasta, "-r", bed, "--seed", "1", "-q", "1", "-o", tmp_path / "g.tsv")
    run("enrich", ap1, "-s", tmp_path / "peaks.fa", "--seed", "1", "-q", "1",
        "-o", tmp_path / "s.tsv")
    got = table(tmp_path / "g.tsv")
    assert got.equals(table(tmp_path / "s.tsv"))
    ap1_row = got.filter(pl.col("motif") == "AP-1")
    assert ap1_row["target_seq_n"][0] == len(peaks) and ap1_row["target_seq_hits"][0] > 0
    run("enrich", ap1, "-s", tmp_path / "peaks.fa", "-g", fasta, ok=False)
    assert "give the target sequences either with -s, or with -g and -r" in capsys.readouterr().err
    run("enrich", ap1, "-g", fasta, ok=False)
    assert "give the target sequences with -s, or with -g and -r" in capsys.readouterr().err


def test_extract_strands_and_width(tmp_path, genome):
    fasta, bed, chroms, peaks = genome
    run("extract", "-g", fasta, "-r", bed, "--width", "50", "-o", tmp_path / "x.fa")
    text = (tmp_path / "x.fa").read_text()
    assert max(len(ln) for ln in text.splitlines() if not ln.startswith(">")) == 50
    seqs = list(um.read_fasta(tmp_path / "x.fa").values())
    for i, (c, s, e) in enumerate(peaks):
        want = chroms[c][s:e]
        assert seqs[i] == (want if i % 2 == 0 else revcomp(want))


def test_find_nothing(tmp_path, capsys):
    rng = random.Random(2)
    um.write_fasta({f"s{i}": "".join(rng.choice("ACGT") for _ in range(100)) for i in range(20)},
                   tmp_path / "r.fa")
    run("find", "-s", tmp_path / "r.fa", "--seed", "1", "--max-width", "6", "--nmotifs", "1",
        "--table", tmp_path / "f.tsv", "-o", tmp_path / "f.meme")
    assert capsys.readouterr().err == "no motifs were found\n"
    assert table(tmp_path / "f.tsv").height == 0
    assert not (tmp_path / "f.meme").exists()


def test_bkg_inputs_and_bigwig(tmp_path, genome, capsys):
    fasta, bed, chroms, peaks = genome
    run("bkg", "-s", TARGETS, "-g", fasta, "-r", bed, ok=False)
    assert "either with -s, or with -g and -r" in capsys.readouterr().err
    run("bkg", "-g", fasta, ok=False)
    assert "give the targets with -s, or with -g and -r" in capsys.readouterr().err
    run("bkg", "-s", TARGETS, "-g", fasta, "--bigwig", "x.bw", ok=False)
    assert "needs the targets as ranges" in capsys.readouterr().err
    run("bkg", "-g", fasta, "-r", bed, "--bigwig", "a/s.bw", "--bigwig", "b/s.bw", ok=False)
    assert "the bigWig name 's' is used twice" in capsys.readouterr().err
    pybigtools = pytest.importorskip("pybigtools")
    lengths = {c: len(s) for c, s in chroms.items()}
    # Signal is high over the first half of chr1 only.
    bw = tmp_path / "atac.signal.bw"
    pybigtools.open(str(bw), "w").write(
        lengths, [(c, i, min(n, i + 50), 10.0 if c == "chr1" and i < 3000 else 0.5)
                  for c, n in lengths.items() for i in range(0, n, 50)])
    run("bkg", "-g", fasta, "-r", bed, "--seed", "1", "--candidates", "500",
        "--bigwig", f"open={bw}", "--bigwig", bw, "--covariate-bin-type", "equalwidth",
        "--covariate-bins", "2", "--table", tmp_path / "b.tsv", "-o", tmp_path / "b.fa")
    matches = table(tmp_path / "b.tsv")
    assert {"target_open", "universe_open", "target_atac", "universe_atac"} <= set(matches.columns)
    assert matches["target_open"].to_list() == matches["target_atac"].to_list()
    # Targets in high signal get backgrounds in high signal.
    high = matches.filter(pl.col("target_open") > 5)
    assert high.height > 0 and (high["universe_open"] > 5).all()
    assert len(um.read_fasta(tmp_path / "b.fa")) == len(peaks)


def test_tree_options(tmp_path):
    motifs = um.read_transfac(MOTIFS)
    run("tree", MOTIFS, "--height", "0.5", "-o", tmp_path / "h.tsv")
    got = table(tmp_path / "h.tsv")
    assert got["motif"].to_list() == names(motifs)
    assert got["motif_i"].to_list() == list(range(12))
    assert got["cluster"].to_list() == um.motif_tree(motifs).cutree(h=0.5).tolist()
    pytest.importorskip("matplotlib")
    run("tree", MOTIFS, "--plot", tmp_path / "t.png", "--layout", "circular",
        "-o", tmp_path / "t.nwk")
    assert (tmp_path / "t.png").read_bytes()[:4] == b"\x89PNG"
    assert (tmp_path / "t.nwk").read_text().strip().endswith(";")


def test_logo_file_names(tmp_path):
    motifs = [m.replace(name=name) for m, name in
              zip(um.read_jaspar(DATA / "example.jaspar"), ["a/b", "a?b", "///"])]
    um.write_meme(motifs, tmp_path / "m.meme")
    run("logo", tmp_path / "m.meme", "--each", tmp_path / "svg")
    # Names unsafe for files are cleaned up, and clashes numbered.
    assert sorted(p.name for p in (tmp_path / "svg").iterdir()) == \
        ["a_b.1.svg", "a_b.svg", "motif.svg"]
    pytest.importorskip("matplotlib")
    run("logo", tmp_path / "m.meme", "--each", tmp_path / "png", "--ext", "png", "--dpi", "50")
    files = sorted((tmp_path / "png").iterdir())
    assert [p.name for p in files] == ["a_b.1.png", "a_b.png", "motif.png"]
    assert all(p.read_bytes()[:4] == b"\x89PNG" for p in files)


def test_fetch_hocomoco_and_cisbp(tmp_path, monkeypatch):
    from universalmotif import databases

    calls = []

    def fake(name):
        def f(*args, **kwargs):
            calls.append((name, args, kwargs))
            return um.example_motifs()[:1]
        return f

    monkeypatch.setattr(databases, "hocomoco", fake("hocomoco"))
    monkeypatch.setattr(databases, "cisbp", fake("cisbp"))
    run("fetch", "hocomoco", "--version", "11", "--species", "MOUSE", "-o", tmp_path / "h.meme")
    run("fetch", "cisbp", "Arabidopsis_thaliana", "--family", "WRKY", "--family", "AP2",
        "--inferred", "--refresh", "-t", "jaspar", "-o", tmp_path / "c.txt")
    assert calls == [
        ("hocomoco", (), {"version": 11, "collection": "CORE", "species": "MOUSE",
                          "refresh": False}),
        ("cisbp", ("Arabidopsis_thaliana",), {"family": ["WRKY", "AP2"], "inferred": True,
                                              "refresh": True}),
    ]
    assert names(um.read_meme(tmp_path / "h.meme")) == ["TATA-box"]
    assert names(um.read_jaspar(tmp_path / "c.txt")) == ["TATA-box"]


def test_proximity_inputs(tmp_path, genome, capsys):
    fasta, _bed, _chroms, peaks = genome
    summits = tmp_path / "summits.bed"
    summits.write_text("".join(f"{c}\t{s + 103}\t{s + 104}\n" for c, s, _ in peaks))
    ap1 = DATA / "example.jaspar"
    base = ["proximity", "-a", summits, "-q", "1"]
    run(*base, ap1, "-g", fasta, "-o", tmp_path / "default.tsv")
    want = table(tmp_path / "default.tsv")
    # A chrom.sizes file, or the genome itself, gives the whole genome as
    # the universe, as does scanning chromosomes from a FASTA file with -s.
    (tmp_path / "chrom.sizes").write_text("chr1\t6000\nchr2\t4000\n")
    run(*base, ap1, "-g", fasta, "--universe", tmp_path / "chrom.sizes", "-o", tmp_path / "s.tsv")
    run(*base, ap1, "-g", fasta, "--universe", fasta, "-o", tmp_path / "g.tsv")
    run(*base, ap1, "-s", fasta, "-o", tmp_path / "f.tsv")
    for name in ("s", "g", "f"):
        assert table(tmp_path / f"{name}.tsv").equals(want), name
    # A BED universe keeps only the hits inside it.
    (tmp_path / "u.bed").write_text("chr1\t0\t3000\n")
    run(*base, ap1, "-g", fasta, "--universe", tmp_path / "u.bed", "-o", tmp_path / "u.tsv")
    near = table(tmp_path / "u.tsv").filter(pl.col("motif") == "AP-1")["nhits"][0]
    assert 0 < near < want.filter(pl.col("motif") == "AP-1")["nhits"][0]
    capsys.readouterr()
    for args, msg in [
        ([ap1, "--hits", tmp_path / "s.tsv"], "either motifs to scan for or --hits"),
        ([], "give motifs to scan for, or a table of hits"),
        ([ap1, "-g", fasta, "-s", fasta], "with -g or -s, not both"),
        ([ap1], "give the genome to scan with -g"),
    ]:
        run(*base, *args, ok=False)
        assert msg in capsys.readouterr().err, args


def test_coocc_from_hits(tmp_path, capsys):
    run("scan", MOTIFS, "-s", TARGETS, "-o", tmp_path / "h.tsv")
    run("coocc", MOTIFS, "-s", TARGETS, "-o", tmp_path / "scanned.tsv")
    run("coocc", MOTIFS, "--hits", tmp_path / "h.tsv", "--n-sequences", "100",
        "-o", tmp_path / "hits.tsv")
    got = table(tmp_path / "hits.tsv")
    assert got.height > 0 and got.equals(table(tmp_path / "scanned.tsv"))
    run("coocc", MOTIFS, "--hits", tmp_path / "h.tsv", ok=False)
    assert "--hits needs --n-sequences" in capsys.readouterr().err


def test_posenr_from_hits(tmp_path, capsys):
    run("scan", MOTIFS, "-s", TARGETS, "-o", tmp_path / "h.tsv")
    length = len(next(iter(um.example_sequences("targets").values())))
    run("posenr", MOTIFS, "-s", TARGETS, "--centres", "-o", tmp_path / "scanned.tsv")
    run("posenr", "--hits", tmp_path / "h.tsv", "--seq-length", length, "--centres",
        "-o", tmp_path / "hits.tsv")
    got = table(tmp_path / "hits.tsv")
    assert got["motif"].to_list() == ["W-box"] and "centres" in got.columns
    assert got.equals(table(tmp_path / "scanned.tsv"))
    capsys.readouterr()
    um.write_fasta({"a": "ACGT" * 10, "b": "ACGT" * 12}, tmp_path / "ragged.fa")
    for args, msg in [
        ([MOTIFS, "--hits", tmp_path / "h.tsv"], "either motifs to scan for or --hits"),
        (["-s", TARGETS], "give motifs to scan for, or a table of hits"),
        ([MOTIFS, "-s", tmp_path / "ragged.fa"], "the sequences differ in length"),
    ]:
        run("posenr", *args, ok=False)
        assert msg in capsys.readouterr().err, args


def test_main_failures(tmp_path, monkeypatch, capsys):
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(um, "summarize_motifs", interrupted)
    assert main(["summary", MOTIFS]) == 130

    def broken(*_args, **_kwargs):
        raise RuntimeError("a bug")

    monkeypatch.setattr(um, "summarize_motifs", broken)
    # Errors the user cannot fix keep their traceback.
    with pytest.raises(RuntimeError, match="a bug"):
        main(["summary", MOTIFS])
    missing = tmp_path / "missing.meme"
    with pytest.raises(Exception, match="no such file"):
        main(["--debug", "convert", str(missing)])
    # Errors name the program as it was called.
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/umotif"])
    assert main(["convert", str(missing)]) == 1
    assert capsys.readouterr().err == f"umotif convert: error: no such file: {missing}\n"


def test_broken_pipe(tmp_path, monkeypatch):
    sink = tmp_path / "stdout"
    fd = os.open(sink, os.O_WRONLY | os.O_CREAT)

    class Closed(io.StringIO):
        def write(self, _text):
            raise BrokenPipeError

        def fileno(self):
            return fd

    try:
        monkeypatch.setattr(sys, "stdout", Closed())
        assert main(["summary", MOTIFS]) == 0
        # Standard output now goes nowhere, so nothing more can fail on it.
        os.write(fd, b"lost")
        assert sink.read_bytes() == b""
    finally:
        os.close(fd)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_as_script(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["umotif", "summary", MOTIFS])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("universalmotif._cli", run_name="__main__")
    assert exit_.value.code == 0
    assert capsys.readouterr().out.startswith("name\tfamily\t")


def test_proximity_bed_universe_named_txt(tmp_path, genome):
    fasta, _bed, _chroms, peaks = genome
    summits = tmp_path / "summits.bed"
    summits.write_text("".join(f"{c}\t{s + 103}\t{s + 104}\n" for c, s, _ in peaks))
    ap1 = DATA / "example.jaspar"
    for ext in ("bed", "txt"):
        (tmp_path / f"u.{ext}").write_text("chr1\t0\t3000\n")
        run("proximity", ap1, "-a", summits, "-g", fasta, "-q", "1", "--universe",
            tmp_path / f"u.{ext}", "-o", tmp_path / f"{ext}.tsv")
    # read_ranges takes BED files whatever their extension, as for -a.
    assert table(tmp_path / "txt.tsv").equals(table(tmp_path / "bed.tsv"))


def test_warning_format_is_restored():
    import warnings

    before = warnings.showwarning
    run("summary", DATA / "example.jaspar")
    assert warnings.showwarning is before
