"""Behaviour that differs from the R package on purpose, or that it lacks,
checked directly."""

import gzip

import matplotlib
import pytest

import universalmotif as um

matplotlib.use("Agg")


def test_gzipped_files(tmp_path):
    seqs = {"a b": "ACGT", "c": "TTGA"}
    path = tmp_path / "seqs.fa.gz"
    um.write_fasta(seqs, path)
    assert gzip.decompress(path.read_bytes()).decode().startswith(">a b\n")
    assert um.read_fasta(path, trim_names=False) == seqs
    assert um.read_fasta(path) == {"a": "ACGT", "c": "TTGA"}
    motifs = um.example_motifs()[:2]
    mpath = tmp_path / "motifs.meme.gz"
    um.write_meme(motifs, mpath)
    assert [m.name for m in um.read_meme(mpath)] == [m.name for m in motifs]


def test_fasta_names(tmp_path):
    path = tmp_path / "seqs.fa"
    path.write_text(">chr1 assembled\nACGT\n>chr2\nTT\n>chr1 again\nGG\n")
    # Names are first words by default, everywhere, and may not collide.
    with pytest.raises(ValueError, match="the sequence name 'chr1' appears more than once"):
        um.read_fasta(path)
    with pytest.raises(ValueError, match="the sequence name 'chr1' appears more than once"):
        um.read_fasta(path, packed=True)
    motif = um.create_motif("ACGT")
    with pytest.raises(ValueError, match="the sequence name 'chr1' appears more than once"):
        um.scan_sequences(motif, path, pvalue=0.01)
    # Whole headers stay apart, and repeated ones keep the last, as before.
    assert list(um.read_fasta(path, trim_names=False)) == ["chr1 assembled", "chr2", "chr1 again"]
    hits = um.scan_sequences(motif, path, pvalue=0.01, trim_names=False)
    assert set(hits["sequence"]) == {"chr1 assembled"}
    path.write_text(">chr1 x\nACGT\n>chr1 x\nGG\n")
    with pytest.warns(UserWarning, match="more than once"):
        assert um.read_fasta(path, trim_names=False) == {"chr1 x": "GG"}
    path.write_text(">chr1 assembled from contigs\nACGTAC\n>chr2\tsecond\nTT\n")
    assert um.read_fasta(path) == {"chr1": "ACGTAC", "chr2": "TT"}
    assert list(um.read_fasta(path, packed=True)) == ["chr1", "chr2"]
    hits = um.scan_sequences(motif, path, pvalue=0.01)
    assert set(hits["sequence"]) == {"chr1"}
    assert list(um.Genome(path)) == ["chr1", "chr2"]


def test_packed_size():
    assert repr(um.pack_sequences(um.example_sequences())).endswith("25.2 kB)")
    assert repr(um.pack_sequences({"a": "ACGT"})).endswith(" bytes)")


def test_single_logo_is_named():
    fig = um.view_motifs(um.example_motifs()[0])
    assert fig.axes[0].get_title() == "TATA-box"


def test_tree_groups_and_legend():
    ms = um.example_motifs()
    tree = um.motif_tree(ms)
    ax = tree.plot(groups={m.name: m.family for m in ms})
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    legend = ax.artists[0] if ax.artists else ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()][0] == ms[0].family
    assert legend.get_window_extent(renderer).x1 <= fig.bbox.x1
    with pytest.raises(ValueError, match="no value"):
        tree.plot(groups={"TATA-box": "x"})


def test_proximity_keeps_input_order():
    seqs = um.example_sequences()
    hits = um.scan_sequences([um.example_motifs()[5], um.example_motifs()[3]], seqs)
    anchors = [(name, 500, 500) for name in seqs]
    res = um.motif_proximity(hits, anchors, universe={k: len(v) for k, v in seqs.items()}, qvalue=1)
    assert dict(zip(res["motif"], res["motif_i"])) == {"W-box": 0, "G-box": 1}


def test_complexity_default_alphabet():
    assert um.calc_complexity("AAAAAAAA") == 0
    a, b = um.calc_complexity(["ACGTTGCAAC", "ATATATATAT"])
    assert b == pytest.approx(um.calc_complexity("ATATATATAT", alph="ACGT"))


def test_silent_failures_warn(tmp_path):
    gbox = um.example_motifs()[3]
    seqs = um.example_sequences()
    with pytest.warns(UserWarning, match=r"cannot reach a P-value of 1e-06.*G-box \(1\.53e-05\)"):
        assert um.scan_sequences(gbox, seqs, pvalue=1e-6).height == 0
    with pytest.warns(UserWarning, match="cannot reach"):
        um.enrich_motifs([gbox], seqs, um.example_sequences("controls"), pvalue=1e-6, qvalue=1)
    rna = {k: v.replace("T", "U") for k, v in list(seqs.items())[:5]}
    with pytest.warns(UserWarning, match="contain U but not T"):
        um.scan_sequences(gbox, rna)
    path = tmp_path / "seqs.fa"
    path.write_text(">a\nACGT\n")
    with pytest.warns(UserWarning, match="names a file"):
        um.scan_sequences(gbox, str(path), pvalue=0.5)
    with pytest.warns(UserWarning, match="P-value of 0"):
        assert um.motif_pvalue(gbox, score=100.0) == 0
    short = um.create_motif("CACGTG", name="short")
    with pytest.warns(UserWarning, match="left out: TATA-box"):
        um.merge_motifs([um.example_motifs()[0], short], min_overlap=8)
    with pytest.raises(ValueError, match='type="PWM"'):
        um.create_motif([[0.5, -0.2], [0.1, 0.3], [-0.1, 0.0], [0.0, 0.2]])
    pwm = um.create_motif([[0.5, -0.2], [0.1, 0.3], [-0.1, 0.0], [0.0, 0.2]], "DNA", "PWM")
    assert pwm.type == "PWM"


def _site(site, strand="+-", alphabet=None, name="motif"):
    return um.create_motif(site, alphabet, nsites=20, pseudocount=1, strand=strand, name=name)


def test_gaps_are_corrected_by_default():
    m = um.add_gap(_site("TTGACCAAT"), 3, 0, 3)
    seqs = ["A" * 30 + "TTGACCAAT" + "C" * 30 + "TTGAACCAAT" + "G" * 30]
    kw = {"pvalue": 1e-2, "rc": False}
    corrected = um.scan_sequences(m, seqs, **kw)
    raw = um.scan_sequences(m, seqs, correct_gaps=False, **kw)
    assert corrected.height > 0
    # Four gap sizes, so the P-values are four times as large.
    joined = corrected.join(raw, on=["start", "end"], suffix="_raw")
    assert (joined["pvalue"] == (joined["pvalue_raw"] * 4).clip(upper_bound=1)).all()
    assert raw.height > corrected.height
    full = um.match_motifs(m, seqs, value="counts", **kw).matrix.sum()
    assert full == corrected.height
    assert um.match_motifs(m, seqs, value="counts", correct_gaps=False, **kw).matrix.sum() == raw.height
    # Motifs without gaps are untouched.
    plain = _site("TTGACCAAT")
    assert um.scan_sequences(plain, seqs, **kw).equals(
        um.scan_sequences(plain, seqs, correct_gaps=False, **kw))


def test_palindromes_count_once_without_overlaps(monkeypatch):
    gbox = _site("CACGTG")
    seqs = ["AAAACACGTGAAAA", "TTTCACGTGTTTCACGTGTT"]
    assert um.scan_sequences(gbox, seqs, pvalue=1e-3).height == 6
    once = um.scan_sequences(gbox, seqs, pvalue=1e-3, no_overlaps=True)
    assert once.height == 3 and set(once["strand"]) == {"+"}
    per_strand = um.scan_sequences(gbox, seqs, pvalue=1e-3, no_overlaps=True,
                                   no_overlaps_by_strand=False)
    assert per_strand.height == 6
    # motif_coocc removes overlaps across strands too.
    seen = []
    scan = um._enrich.scan_sequences

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return scan(*args, **kwargs)

    monkeypatch.setattr(um._enrich, "scan_sequences", spy)
    um.motif_coocc([gbox, _site("TTGACC")], seqs, pvalue=1e-3, no_overlaps=True, max_distance=10)
    um.motif_coocc([gbox, _site("TTGACC")], seqs, pvalue=1e-3, no_overlaps=True, max_distance=10,
                   no_overlaps_by_strand=False)
    assert [k["no_overlaps_by_strand"] for k in seen] == [True, False]


def test_one_stranded_motifs_are_scanned_on_their_strand():
    # The site only appears as its reverse complement.
    seqs = {"s": "A" * 20 + "GGGTTTTCC" + "C" * 20}
    kw = {"pvalue": 1e-3}
    strands = {s: um.scan_sequences(_site("GGAAAACCC", s), seqs, **kw)["strand"].to_list()
               for s in ("+-", "+", "-")}
    assert strands == {"+-": ["-"], "+": [], "-": ["-"]}
    for rc in (True, False):
        assert um.scan_sequences(_site("GGAAAACCC", "+"), seqs, rc=rc, **kw).height == 0
        assert um.scan_sequences(_site("GGAAAACCC", "-"), seqs, rc=rc, **kw).height == 1
    assert um.scan_sequences(_site("GGAAAACCC"), seqs, rc=False, **kw).height == 0
    assert um.match_motifs(_site("GGAAAACCC", "+"), seqs, **kw).matrix.sum() == 0
    assert um.match_motifs(_site("GGAAAACCC", "-"), seqs, **kw).matrix.sum() == 1
    effects = um.variant_motif_effects([(21, "G", "A")], seqs["s"], _site("GGAAAACCC", "+"), **kw)
    assert effects.height == 0
    effects = um.variant_motif_effects([(21, "G", "A")], seqs["s"], _site("GGAAAACCC"), **kw)
    assert effects["strand"].to_list() == ["-"]
    targets = [seqs["s"]] * 5
    bkg = ["ACGT" * 12] * 5
    res = {s: um.enrich_motifs(_site("GGAAAACCC", s), targets, bkg, qvalue=1, **kw)
           for s in ("+-", "+")}
    assert res["+-"]["target_seq_hits"].to_list() == [5]
    assert res["+"]["target_seq_hits"].to_list() == [0]


def test_rna_is_scanned_on_one_strand_by_default():
    rna = _site("GGAAAACCC", alphabet="RNA")
    fwd, rev = "A" * 20 + "GGAAAACCC" + "C" * 20, "A" * 20 + "GGGUUUUCC" + "C" * 20
    seqs = {"fwd": fwd, "rev": rev}
    assert um.scan_sequences(rna, seqs, pvalue=1e-3)["sequence"].to_list() == ["fwd"]
    both = um.scan_sequences(rna, seqs, pvalue=1e-3, rc=True)
    assert both["sequence"].to_list() == ["fwd", "rev"]
    assert um.match_motifs(rna, seqs, pvalue=1e-3).matrix.toarray()[:, 0].tolist() == [1, 0]
    res = um.enrich_motifs(rna, [fwd, rev] * 5, ["ACGU" * 12] * 10, qvalue=1, pvalue=1e-3)
    assert res["target_seq_hits"].to_list() == [5]
    # DNA keeps both strands.
    dna = um.scan_sequences(_site("GGAAAACCC"), {k: v.replace("U", "T") for k, v in seqs.items()},
                            pvalue=1e-3)
    assert dna["sequence"].to_list() == ["fwd", "rev"]


def test_motifs_found_on_one_strand_are_for_that_strand():
    seqs = [s[:46] + "TTGACCAT" + s[46:] for s in um.create_sequences(seqnum=20, seqlen=92, seed=1)]
    both = um.motif_finder(seqs, min_width=6, max_width=8, seed=1)
    one = um.motif_finder(seqs, min_width=6, max_width=8, seed=1, rc=False)
    assert both.height and one.height
    assert {m.strand for m in both["motif"]} == {"+-"}
    assert {m.strand for m in one["motif"]} == {"+"}
    rna = um.motif_finder([s.replace("T", "U") for s in seqs], min_width=6, max_width=8, seed=1)
    assert {m.strand for m in rna["motif"]} == {"+"}


def test_shuffles_keep_dinucleotides_by_default():
    seq = "ACGCGTTAGCATCGATCGGGCATTACGCGATATCGCGATCGATCGCGCTAGCTAGG"
    out = um.shuffle_sequences(seq, seed=2)
    assert out != seq
    assert um.count_kmers(out, k=2).equals(um.count_kmers(seq, k=2))
    assert um.count_kmers(um.shuffle_string(seq, seed=2), k=2).equals(um.count_kmers(seq, k=2))
    assert um.shuffle_sequences(seq, k=1, seed=2) != out


def test_merge_similar_linkage():
    sites = ("GATTACAGCATGCA", "CAGCATGCATTGAC", "ATGCATTGACCGTA", "TTGACCGTAGGTCA")
    chain = [_site(s, name=f"c{i}") for i, s in enumerate(sites)]
    motifs = chain + [um.create_motif(10, seed=i, name=f"r{i}") for i in range(8)]
    # Each motif of the chain overlaps the next, so single linkage joins
    # them all, with a warning, while complete linkage keeps the ends apart.
    with pytest.warns(UserWarning, match="cluster 0 holds 4 of the 12 motifs"):
        single = um.merge_similar(motifs, return_clusters=True)["cluster"].to_list()
    assert single[:5] == [0, 0, 0, 0, 1]
    complete = um.merge_similar(motifs, return_clusters=True, linkage="complete")
    assert complete["cluster"].to_list()[:5] == [0, 1, 1, 1, 2]
    merged = um.merge_similar(motifs, linkage="complete")
    assert len(merged) == 10 and merged[0].name == "c0"
    with pytest.raises(ValueError, match="linkage"):
        um.merge_similar(motifs, linkage="average")


def test_meme_strands_default_to_both(tmp_path):
    body = "MOTIF m\nletter-probability matrix: alength= 4 w= 2\n1 0 0 0\n0 1 0 0\n"
    path = tmp_path / "m.meme"
    for head, strand in (("", "+-"), ("strands: +\n", "+"), ("strands: + -\n", "+-")):
        path.write_text(f"MEME version 4\n\nALPHABET= ACGT\n\n{head}\n{body}")
        assert um.read_meme(path)[0].strand == strand
    path.write_text("MEME version 4\n\nALPHABET= ACGU\n\n" + body)
    assert um.read_meme(path)[0].strand == "+-"
