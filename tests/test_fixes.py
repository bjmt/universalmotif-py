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
    assert um.read_fasta(path) == seqs
    motifs = um.example_motifs()[:2]
    mpath = tmp_path / "motifs.meme.gz"
    um.write_meme(motifs, mpath)
    assert [m.name for m in um.read_meme(mpath)] == [m.name for m in motifs]


def test_fasta_names(tmp_path):
    path = tmp_path / "seqs.fa"
    path.write_text(">chr1 assembled\nACGT\n>chr2\nTT\n>chr1 again\nGG\n")
    with pytest.warns(UserWarning, match="more than once"):
        trimmed = um.read_fasta(path, trim_names=True)
    assert trimmed == {"chr1": "GG", "chr2": "TT"}
    assert list(um.read_fasta(path)) == ["chr1 assembled", "chr2", "chr1 again"]
    with pytest.raises(ValueError, match="more than once"):
        um.read_fasta(path, packed=True, trim_names=True)


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
