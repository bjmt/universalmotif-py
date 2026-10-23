"""SVG logos, the Jupyter display of motifs, and export to logomaker."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import universalmotif as um

SVG = "{http://www.w3.org/2000/svg}"


def _paths(svg):
    return ET.fromstring(svg).findall(f"{SVG}path")


def test_logo_svg():
    gbox = um.example_motifs()[3]
    svg = um.logo_svg(gbox, height=50, position_width=10)
    root = ET.fromstring(svg)
    assert root.get("width") == "80" and root.get("height") == "50"
    fills = {p.get("fill") for p in _paths(svg)}
    assert fills <= {"#109648", "#255C99", "#F7B32B", "#D62839"}
    green = um.logo_svg(gbox, colours={"A": "green", "C": "green", "G": "green", "T": "green"})
    assert {p.get("fill") for p in _paths(green)} == {"green"}
    assert len(_paths(um.logo_svg(gbox, use_type="PPM"))) >= len(_paths(svg))
    aa = um.create_motif("MKVLA", alphabet="AA", nsites=10)
    assert len(_paths(um.logo_svg(aa))) == 5
    custom = um.create_motif(["XYZ", "XYY"], alphabet="XYZ")
    assert len({p.get("fill") for p in _paths(um.logo_svg(custom))}) == 3


def test_logo_svg_gaps_and_cwms():
    gapped = um.add_gap(um.create_motif("TGACGTCA", nsites=20, pseudocount=1), gaploc=4,
                        mingap=1, maxgap=5)
    root = ET.fromstring(um.logo_svg(gapped, position_width=10))
    assert root.get("width") == "90", "a gap takes the room of one position"
    assert [t.text for t in root.findall(f"{SVG}text")] == ["1–5"]
    cwm = um.create_motif(np.array([[0.5, -0.2, 0.1, 0], [0, 0, 0.3, -0.1]]).T, alphabet="DNA",
                          type="CWM")
    assert _paths(um.logo_svg(cwm))
    with pytest.raises(ValueError, match="CWM"):
        um.logo_svg(um.example_motifs()[0], use_type="CWM")
    bare = um.create_motif(["TGA", "TGA"], type="PCM", pseudocount=0)
    with pytest.raises(ValueError, match="pseudocount"):
        um.logo_svg(bare, use_type="PWM")


def test_repr_html():
    m = um.example_motifs()[3].replace(name="<G-box>", extrainfo={"note": "a & b"})
    out = m._repr_html_()
    assert "<svg" in out and "&lt;G-box&gt;" in out and "a &amp; b" in out
    assert "<details><summary>Matrix</summary>" in out
    bare = um.create_motif(["TGA", "TGA"], type="PCM", pseudocount=0)
    assert "<svg" in bare._repr_html_(), "the logo is drawn from information content"


def test_to_logomaker():
    pytest.importorskip("pandas")
    m = um.example_motifs()[3]
    df = um.to_logomaker(m)
    assert list(df.columns) == m.letters and df.index.tolist() == list(range(m.width))
    icm = um.convert_type(m, "ICM").matrix
    np.testing.assert_allclose(df.to_numpy().T, icm, atol=1e-12)
    ppm = um.to_logomaker(m, use_type="PPM")
    np.testing.assert_allclose(ppm.sum(axis=1), 1)
