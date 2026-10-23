import io
import random
import subprocess
import sys
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
