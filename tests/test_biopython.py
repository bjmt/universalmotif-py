"""Conversion to and from Biopython's Bio.motifs."""

import numpy as np
import pytest

import universalmotif as um

motifs = pytest.importorskip("Bio.motifs")


def test_round_trip_keeps_counts_and_metadata():
    ms = um.example_motifs()[:3]
    bio = um.to_biopython(ms)
    assert [b.name for b in bio] == [m.name for m in ms]
    back = um.from_biopython(bio)
    for m, b in zip(ms, back):
        assert b.type == "PCM" and b.alphabet == "DNA"
        np.testing.assert_array_equal(b.matrix, m.matrix)
        assert b.nsites == m.nsites and b.pseudocount == m.pseudocount
        assert b.bkg == m.bkg
    single = um.to_biopython(ms[0])
    assert isinstance(single, motifs.Motif)
    assert um.from_biopython(single).name == ms[0].name


def test_probabilities_and_other_alphabets():
    ppm = um.convert_type(um.example_motifs()[0], "PPM")
    bio = um.to_biopython(ppm)
    # Biopython holds counts, so a PPM goes through its nsites.
    assert sum(bio.counts[a][0] for a in "ACGT") == ppm.nsites
    freq = motifs.Motif(alphabet="ACGU", counts={"A": [0.5, 0.1], "C": [0.5, 0.2],
                                                  "G": [0.0, 0.3], "U": [0.0, 0.4]})
    m = um.from_biopython(freq)
    assert m.alphabet == "RNA" and m.type == "PPM" and m.nsites is None
    aa = um.create_motif("MKVLA", alphabet="AA", nsites=10)
    assert um.from_biopython(um.to_biopython(aa)).alphabet == "AA"
    custom = um.create_motif(["XYZ", "XYY"], alphabet="XYZ")
    back = um.from_biopython(um.to_biopython(custom))
    assert back.alphabet == custom.alphabet and back.letters == ["X", "Y", "Z"]


def test_jaspar_motifs():
    from Bio.motifs import jaspar

    jm = jaspar.Motif(
        "MA0001.1", "AGL3", counts={"A": [1, 2], "C": [0, 1], "G": [2, 0], "T": [0, 0]},
        tf_class=["MADS"], tf_family=["MADS box"], species=["3702"], tax_group="plants",
        data_type="SELEX", acc=["P29383"],
    )
    m = um.from_biopython(jm)
    assert (m.name, m.altname, m.family, m.organism) == ("MA0001.1", "AGL3", "MADS box", "3702")
    assert m.extrainfo == {"class": "MADS", "tax_group": "plants", "data_type": "SELEX",
                           "accession": "P29383"}
    out = um.to_biopython(m, jaspar=True)
    assert isinstance(out, jaspar.Motif)
    assert (out.matrix_id, out.name, out.tf_family) == ("MA0001.1", "AGL3", "MADS box")
    text = motifs.write([out], "jaspar")
    assert text.startswith(">MA0001.1 AGL3")


def test_motifs_parsed_by_biopython():
    with open(um.example_file("meme")) as f:
        record = motifs.parse(f, "minimal")
    ms = um.from_biopython(record)
    assert [m.name for m in ms] == [m.name for m in um.read_meme(um.example_file("meme"))]
    assert all(m.alphabet == "DNA" for m in ms)


def test_pseudocounts_are_summed():
    bio = motifs.create(["TACAA", "TACGC"])
    bio.pseudocounts = 0.5
    assert um.from_biopython(bio).pseudocount == 2.0
    bio.pseudocounts = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    assert um.from_biopython(bio).pseudocount == 1.0
