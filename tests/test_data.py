"""The example data loads, and every sample file reads back."""

import pytest

import universalmotif as um

READERS = {
    "meme": um.read_meme,
    "jaspar": um.read_jaspar,
    "homer": um.read_homer,
    "transfac": um.read_transfac,
    "cisbp": um.read_cisbp,
    "uniprobe": um.read_uniprobe,
    "matrix": um.read_matrix,
}


def test_example_data():
    motifs = um.example_motifs()
    assert len(motifs) == 12 and all(m.family and m.type == "PCM" for m in motifs)
    names = [m.name for m in motifs[:4]]
    for fmt, read in READERS.items():
        got = read(um.example_file(fmt))
        assert len(got) == 4, fmt
        if fmt not in ("matrix", "uniprobe", "cisbp"):
            assert [m.name for m in got] == names, fmt
    targets, controls = um.example_sequences(), um.example_sequences("controls")
    assert len(targets) == len(controls) == 100
    assert um.read_fasta(um.example_file("fasta")) == targets
    with pytest.raises(ValueError):
        um.example_file("yaml")
    with pytest.raises(ValueError):
        um.example_sequences("other")
