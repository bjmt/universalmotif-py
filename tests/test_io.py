"""Motif file readers and writers against the R package, plus file
handling."""

import warnings

import pytest

import universalmotif as um

from conftest import (
    args,
    assert_motif_matches,
    load_reference,
    motif_from_record,
    num,
    nums,
    reference_file,
)


def _strs(v):
    # read.table() turns numeric-looking columns into numbers.
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return [x if isinstance(x, str) else str(x) for x in v]


def _read(case):
    a = args(case["args"])
    path = reference_file(case["file"])
    fmt = case["format"]
    if fmt == "meme":
        return um.read_meme(path, cwm=bool(a.get("cwm", False)),
                            readsites=bool(a.get("readsites", False)),
                            readsites_meta=bool(a.get("readsites_meta", False)))
    if fmt == "matrix":
        return um.read_matrix(path, type=a.get("type"), positions=a.get("positions", "columns"),
                              headers=a.get("headers", True), rownames=a.get("rownames", False))
    return getattr(um, f"read_{fmt}")(path)


def test_readers():
    for i, case in enumerate(load_reference("io.json", "read")):
        ctx = f"read case {i}: {case['format']} {case['file']} {case['args']}"
        if case.get("motifs") is None:
            with pytest.raises(ValueError):
                _read(case)
            continue
        got = _read(case)
        motifs = got["motifs"] if isinstance(got, dict) else got
        assert len(motifs) == len(case["motifs"]), ctx
        for g, w in zip(motifs, case["motifs"]):
            assert_motif_matches(g, w, ctx)
        if "sites" in case:
            sites = list(got["sites"].values())
            assert len(sites) == len(case["sites"]), ctx
            for g, w in zip(sites, case["sites"]):
                assert g["sequence"].to_list() == _strs(w["names"]), ctx
                assert g["site"].to_list() == _strs(w["sites"]), ctx
        if "sites_meta" in case:
            meta = got["sites_meta"]
            for g, w in zip(meta.partition_by("motif", maintain_order=True), case["sites_meta"]):
                assert g["sequence"].to_list() == _strs(w["sequence"]), ctx
                assert g["site"].to_list() == _strs(w["site"]), ctx
                # MEME and R count positions from 1.
                assert g["position"].to_list() == [int(x) - 1 for x in nums(w["position"])], ctx
                assert g["pvalue"].to_list() == pytest.approx(nums(w["pvalue"])), ctx
        if "combined" in case:
            comb = got["sites_meta_combined"]
            w = case["combined"]
            assert comb["sequence"].to_list() == _strs(w["sequence"]), ctx
            assert comb["pvalue"].to_list() == pytest.approx(nums(w["pvalue"])), ctx
            # The R package truncates diagrams spanning three or more lines.
            for g, d in zip(comb["diagram"].to_list(), _strs(w["diagram"])):
                assert g == d or (g.startswith(d) and d.endswith("_")), ctx


def _write(case, motifs, path):
    a = args(case["args"])
    fmt = case["format"]
    if fmt == "meme":
        bkg = a.get("bkg")
        um.write_meme(motifs, path, version=int(num(a.get("version", 5))),
                      bkg=None if bkg is None else nums(bkg), strand=a.get("strand"),
                      cwm=bool(a.get("cwm", False)))
    elif fmt == "jaspar":
        um.write_jaspar(motifs, path)
    elif fmt == "homer":
        um.write_homer(motifs, path, threshold=nums(a.get("threshold", 0.8)),
                       threshold_type=a.get("threshold_type", "logodds"))
    elif fmt == "transfac":
        um.write_transfac(motifs, path, name_tag=a.get("name_tag", "ID"),
                          altname_tag=a.get("altname_tag", "NA"))
    elif fmt == "matrix":
        um.write_matrix(motifs, path, positions=a.get("positions", "columns"),
                        rownames=bool(a.get("rownames", False)), type=a.get("type"),
                        sep=None if a.get("sep_none") else a.get("sep", ""),
                        headers=a.get("headers", True), digits=int(num(a.get("digits", 6))))
    else:
        raise AssertionError(fmt)


def test_writers(tmp_path):
    for i, case in enumerate(load_reference("io.json", "write")):
        ctx = f"write case {i}: {case['format']} {case['args']}"
        motifs = [motif_from_record(r) for r in case["motifs"]]
        path = tmp_path / f"case{i}.txt"
        if case.get("text") is None:
            with pytest.raises(ValueError):
                _write(case, motifs, path)
        else:
            _write(case, motifs, path)
            got, want = path.read_text(), case["text"]
            if case["format"] == "homer":
                # Thresholds follow HOMER's own scoring rather than the R
                # package's.
                def strip(t):
                    return [
                        "\t".join(line[1:].split("\t")[:2]) if line.startswith(">") else line
                        for line in t.splitlines()
                    ]

                got, want = strip(got), strip(want)
            if case["format"] == "meme":
                # The R package leaves trailing spaces and writes E= 0 for
                # motifs without an E-value; the port does neither.
                want = "".join(
                    line.rstrip().removesuffix(" E= 0") + "\n" for line in want.splitlines()
                )
            assert got == want, ctx


def test_files_are_not_clobbered(tmp_path):
    motifs = um.read_jaspar(reference_file("jaspar.txt"))
    path = tmp_path / "out.jaspar"
    um.write_jaspar(motifs[:2], path)
    with pytest.raises(FileExistsError):
        um.write_jaspar(motifs, path)
    um.write_jaspar(motifs[2:], path, append=True)
    assert um.read_jaspar(path) == motifs
    um.write_jaspar(motifs[:1], path, overwrite=True)
    assert len(um.read_jaspar(path)) == 1


def test_meme_append_writes_header_once(tmp_path):
    motifs = um.read_meme(reference_file("meme_minimal.txt"))
    path = tmp_path / "out.meme"
    um.write_meme(motifs[:2], path)
    um.write_meme(motifs[2:], path, append=True)
    text = path.read_text()
    assert text.count("MEME version") == 1
    assert [m.name for m in um.read_meme(path)] == [m.name for m in motifs]


def test_skip_lines(tmp_path):
    path = tmp_path / "skip.jaspar"
    path.write_text("some header\nanother line\n" + reference_file("jaspar.txt").read_text())
    assert um.read_jaspar(path, skip=2) == um.read_jaspar(reference_file("jaspar.txt"))


def test_readers_always_return_lists():
    motifs = um.read_matrix(reference_file("hocomoco.txt"), headers=">", positions="rows")
    assert isinstance(motifs, list) and len(motifs) == 1
    assert motifs[0].name == "motif1" and motifs[0].type == "PWM"


def test_meme_sites_structure():
    out = um.read_meme(reference_file("meme_full.txt"), readsites=True, readsites_meta=True)
    assert set(out) == {"motifs", "sites", "sites_meta", "sites_meta_combined"}
    assert list(out["sites"]) == ["1", "2", "3"]
    assert out["sites"]["1"].columns == ["sequence", "site"]
    meta = out["sites_meta"]
    assert meta.columns == ["motif", "sequence", "strand", "position", "pvalue", "site"]
    assert set(meta["strand"].unique().to_list()) <= {"+", "-"}
    sites = out["sites"]["1"]["site"].to_list()
    assert um.create_motif(sites).width == out["motifs"][0].width


def test_unsupported_writes(tmp_path):
    motif = um.read_jaspar(reference_file("jaspar.txt"))[0]
    custom = um.create_motif(["xyz", "xzz"])
    with pytest.raises(ValueError, match="JASPAR"):
        um.write_jaspar(custom, tmp_path / "x.jaspar")
    # Motifs without gaps are written without warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        um.write_matrix(motif, tmp_path / "x.txt")


def test_cisbp_and_uniprobe_writers(tmp_path):
    motifs = um.read_jaspar(reference_file("jaspar.txt"))
    ppms = um.convert_type(motifs, "PPM")
    um.write_cisbp(motifs, tmp_path / "m.cisbp")
    um.write_uniprobe(motifs, tmp_path / "m.uniprobe")
    for back in (um.read_cisbp(tmp_path / "m.cisbp"), um.read_uniprobe(tmp_path / "m.uniprobe")):
        assert [m.name for m in back] == [m.name for m in ppms]
        for b, m in zip(back, ppms):
            assert (b.matrix == m.matrix).all()
    aa = um.create_motif("ACDE")
    with pytest.raises(ValueError, match="CIS-BP"):
        um.write_cisbp(aa, tmp_path / "aa.cisbp")
