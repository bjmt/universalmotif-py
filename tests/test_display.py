"""SVG logos, the Jupyter display of motifs, and export to logomaker."""

import sys
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


def _path_extents(svg):
    """The fill and the x and y ranges of each path of an SVG logo."""
    out = []
    for p in _paths(svg):
        pts = np.array([[float(v) for v in xy.split(",")] for xy in p.get("d")[1:-1].split("L")])
        out.append((p.get("fill"), pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()))
    return out


def test_logo_svg_letter_geometry():
    # Without a pseudocount every column has a single letter of 2 bits, so
    # each letter fills the whole height of its position.
    bare = um.create_motif(["TGA", "TGA"], type="PCM", pseudocount=0, name="<tga>")
    svg = um.logo_svg(bare, height=50, position_width=10)
    root = ET.fromstring(svg)
    assert root.get("aria-label") == "Sequence logo of <tga>"
    assert "&lt;tga&gt;" in svg
    paths = _path_extents(svg)
    assert [p[0] for p in paths] == ["#D62839", "#F7B32B", "#109648"]
    for j, (_, x0, x1, y0, y1) in enumerate(paths):
        assert (y0, y1) == pytest.approx((0.0, 50.0), abs=0.01)
        assert j * 10 < x0 < x1 < (j + 1) * 10
    # A half-informative column is drawn half as high, from the bottom.
    half = um.create_motif(np.array([[0.5, 0.5, 0, 0]]).T, alphabet="DNA", type="PPM")
    tops = [p[3] for p in _path_extents(um.logo_svg(half, height=60))]
    assert min(tops) == pytest.approx(30.0, abs=0.01)
    assert max(p[4] for p in _path_extents(um.logo_svg(half, height=60))) == pytest.approx(60.0, abs=0.01)


def test_logo_svg_colours_and_zero_line():
    gbox = um.example_motifs()[3]
    fills = {p.get("fill") for p in _paths(um.logo_svg(gbox, colours={"A": "green"}))}
    assert fills == {"green", "#7F7F7F"}, "letters without a colour are grey"
    # Logos with values below zero get a line at zero.
    pwm = ET.fromstring(um.logo_svg(gbox, use_type="PWM"))
    lines = pwm.findall(f"{SVG}line")
    assert len(lines) == 1 and lines[0].get("y1") == lines[0].get("y2")
    assert not ET.fromstring(um.logo_svg(gbox)).findall(f"{SVG}line")
    # A CWM of zeros has nothing to draw.
    zero = um.create_motif(np.zeros((4, 3)), alphabet="DNA", type="CWM")
    root = ET.fromstring(um.logo_svg(zero, position_width=10))
    assert root.get("width") == "30" and not _paths(um.logo_svg(zero))


def test_logo_svg_fixed_gap_label():
    gapped = um.add_gap(um.create_motif("TGACGTCA", nsites=20, pseudocount=1), gaploc=2, mingap=3,
                        maxgap=3)
    root = ET.fromstring(um.logo_svg(gapped, position_width=10))
    (line,) = root.findall(f"{SVG}line")
    assert (float(line.get("x1")), float(line.get("x2"))) == (22.0, 28.0)
    assert [t.text for t in root.findall(f"{SVG}text")] == ["3"]
    # Letters after the gap are moved along by one position.
    xs = sorted(p[1] for p in _path_extents(um.logo_svg(gapped, position_width=10)))
    assert not any(20 <= x < 30 for x in xs)


def test_repr_html_rows():
    gapped = um.add_gap(um.create_motif("TGACGTCA", nsites=20, pseudocount=1, name="g"), gaploc=[2, 5],
                        mingap=[1, 3], maxgap=[4, 3])
    out = gapped._repr_html_()
    assert "after 2: 1 to 4, after 5: 3" in out
    assert ">Strands<" in out and ">Target sites<" in out
    m = um.example_motifs()[3].replace(altname="alt", family="bZIP", organism="Arabidopsis", pval=1e-5,
                                       extrainfo={str(i): str(i) for i in range(7)})
    out = m._repr_html_()
    for label in ("Alternate name", "Family", "Organism", "P-value", "Total IC", "Consensus"):
        assert f">{label}<" in out
    assert ">1e-05<" in out
    # Only the first five extra fields are shown.
    assert ">4</th>" in out and ">5</th>" not in out
    aa = um.create_motif("MKVLA", alphabet="AA", nsites=10)
    assert ">Strands<" not in aa._repr_html_()
    cwm = um.create_motif(np.array([[0.5, -0.2, 0.1, 0]]).T, alphabet="DNA", type="CWM")
    assert ">Total IC<" not in cwm._repr_html_()


def test_repr_html_without_a_logo():
    # A count matrix with no counts has no logo to draw, but is still shown.
    empty = um.create_motif(np.zeros((4, 3)), alphabet="DNA", type="PCM")
    with pytest.raises(ValueError):
        um.logo_svg(empty)
    out = empty._repr_html_()
    assert "<svg" not in out and ">Type</th>" in out and "<details>" in out


def test_to_logomaker_types_and_missing_pandas(monkeypatch):
    pytest.importorskip("pandas")
    cwm = um.create_motif(np.array([[0.5, -0.2, 0.1, 0], [0, 0, 0.3, -0.1]]).T, alphabet="DNA",
                          type="CWM")
    df = um.to_logomaker(cwm)
    assert df.index.name == "pos"
    np.testing.assert_array_equal(df.to_numpy(), np.asarray(cwm.matrix).T)
    with pytest.raises(ValueError, match="CWM"):
        um.to_logomaker(um.example_motifs()[3], use_type="CWM")
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"universalmotif\[pandas\]"):
        um.to_logomaker(cwm)
