"""Fetching motifs from databases, with downloads served from small saved
excerpts, plus live tests run only when UNIVERSALMOTIF_NETWORK_TESTS is set."""

import json
import os
import urllib.parse
import zipfile
from pathlib import Path

import pytest

import universalmotif as um
from universalmotif import databases

DATA = Path(__file__).parent / "data" / "databases"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """A fresh cache, with downloads answered from the saved excerpts and
    recorded in the list returned."""
    monkeypatch.setenv("UNIVERSALMOTIF_CACHE", str(tmp_path / "cache"))
    calls = []
    records = json.loads((DATA / "jaspar_records.json").read_text())
    page = (DATA / "jaspar_page.txt").read_text()

    def fake(url, data=None):
        calls.append((url, data))
        parsed = urllib.parse.urlparse(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if "jaspar" in parsed.netloc and parsed.path.endswith("/matrix/"):
            if query["format"] == "json":
                return json.dumps({"count": 2}).encode()
            return page.encode()
        if "jaspar" in parsed.netloc:
            return json.dumps(records[parsed.path.rstrip("/").rsplit("/", 1)[-1]]).encode()
        if url.endswith("_annotation.jsonl"):
            return (DATA / "h12_annotation.jsonl").read_bytes()
        if url.endswith(".meme"):
            return (DATA / "h11.meme").read_bytes()
        if url.endswith(".tsv"):
            return (DATA / "h11_annotation.tsv").read_bytes()
        if url.endswith("bulk_archive.php"):
            return b'<a href="tmp/Some_species_archive.zip">Download</a>'
        if url.endswith(".zip"):
            return _cisbp_zip(tmp_path / "made.zip").read_bytes()
        raise AssertionError(f"unexpected download {url}")

    monkeypatch.setattr(databases, "_download", fake)
    return calls


def test_jaspar(offline):
    motifs = databases.jaspar(tax_group="plants")
    assert [(m.name, m.altname, m.type) for m in motifs] == [
        ("MA0570.3", "ABF1", "PCM"), ("MA0548.3", "AGL15", "PCM")]
    assert motifs[0].family is None
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(offline[0][0]).query))
    assert query["tax_group"] == "plants" and query["version"] == "latest"
    n = len(offline)
    again = databases.jaspar(tax_group="plants")
    assert again == motifs and len(offline) == n, "a second call is served from the cache"
    databases.jaspar(tax_group="plants", refresh=True)
    assert len(offline) > n
    full = databases.jaspar(tax_group="plants", metadata=True)
    assert full[0].family and full[0].organism == "Arabidopsis thaliana"
    assert full[0].extrainfo["tax_group"] == "plants" and "class" in full[0].extrainfo
    assert full[0].matrix.tolist() == motifs[0].matrix.tolist()


def test_hocomoco(offline):
    h12 = databases.hocomoco()
    assert len(h12) == 2 and all(m.type == "PCM" and ".H12CORE." in m.name for m in h12)
    assert h12[0].altname == "AHR" and h12[0].family == "PAS"
    assert h12[0].extrainfo["quality"] in "ABCD"
    human = databases.hocomoco(species="human")
    assert len(human) == 1 and human[0].organism == "Homo sapiens"
    h11 = databases.hocomoco(11)
    assert [m.altname for m in h11] == ["AHR", "AIRE"]
    assert h11[0].family == "PAS domain factors" and h11[0].organism == "Homo sapiens"
    with pytest.raises(ValueError, match="collection"):
        databases.hocomoco(11, "INVIVO")
    with pytest.raises(ValueError, match="species"):
        databases.hocomoco(species="rat")


def _cisbp_zip(path):
    info = (
        "TF_ID\tFamily_ID\tMotif_ID\tDBID\tTF_Name\tTF_Species\tTF_Status\tFamily_Name\tDBID\t"
        "Motif_Type\tMSource_Identifier\tPMID\n"
        "T1\tF1\tM001_3.10\tAT1\tWRKY1\tSome_species\tD\tWRKY\tx\tPBM\tWeirauch2014\t1\n"
        "T2\tF1\tM001_3.10\tAT2\tWRKY2\tSome_species\tI\tWRKY\tx\tPBM\tWeirauch2014\t1\n"
        "T3\tF2\tM002_3.10\tAT3\tERF1\tSome_species\tD\tAP2\tx\tSELEX\tYin2017\t2\n"
        "T4\tF2\tM003_3.10\tAT4\tERF2\tSome_species\tD\tAP2\tx\tSELEX\tYin2017\t2\n"
        "T5\tF2\t.\tAT5\tERF3\tSome_species\tD\tAP2\tx\t.\t.\t.\n"
    )
    pwm = "Pos\tA\tC\tG\tT\n1\t0.1\t0.7\t0.1\t0.1\n2\t0.9\t0.0\t0.05\t0.05\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("TF_Information_all_motifs.txt", info)
        z.writestr("pwms_all_motifs/M001_3.10.txt", pwm)
        z.writestr("pwms_all_motifs/M002_3.10.txt", pwm)
        z.writestr("pwms_all_motifs/M003_3.10.txt", "Pos\tA\tC\tG\tT\n")
    return path


def test_cisbp_archive(tmp_path):
    path = _cisbp_zip(tmp_path / "archive.zip")
    motifs = databases.read_cisbp_archive(path)
    assert [(m.name, m.altname, m.family, m.organism) for m in motifs] == [
        ("M001_3.10", "WRKY1", "WRKY", "Some species"), ("M002_3.10", "ERF1", "AP2", "Some species")]
    assert motifs[0].type == "PPM" and motifs[0].consensus == "CA"
    assert motifs[0].extrainfo == {"motif_type": "PBM", "source": "Weirauch2014", "pubmed": "1",
                                   "status": "D"}
    assert databases.read_cisbp_archive(path, inferred=True)[0].altname == "WRKY1, WRKY2"
    assert [m.name for m in databases.read_cisbp_archive(path, family="AP2")] == ["M002_3.10"]
    folder = tmp_path / "unpacked"
    zipfile.ZipFile(path).extractall(folder)
    assert databases.read_cisbp_archive(folder) == motifs
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="TF_Information"):
        databases.read_cisbp_archive(tmp_path / "empty")


def test_cisbp_download(offline):
    motifs = databases.cisbp("Some species", family="WRKY")
    assert [m.name for m in motifs] == ["M001_3.10"]
    url, form = offline[0]
    assert url.endswith("bulk_archive.php") and form["selSpec"] == "Some_species"
    n = len(offline)
    databases.cisbp("Some species")
    assert len(offline) == n


def test_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVERSALMOTIF_CACHE", str(tmp_path))
    assert databases.cache_dir() == tmp_path
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / "a.txt").write_text("x")
    assert databases.clear_cache() == 1 and databases.clear_cache() == 0


network = pytest.mark.skipif(
    not os.environ.get("UNIVERSALMOTIF_NETWORK_TESTS"), reason="set UNIVERSALMOTIF_NETWORK_TESTS"
)


@network
def test_live(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVERSALMOTIF_CACHE", str(tmp_path))
    jaspar = databases.jaspar(tax_group="plants", name="ABF1", metadata=True)
    assert jaspar and jaspar[0].altname == "ABF1" and jaspar[0].organism
    assert len(databases.hocomoco(11)) > 300
    assert databases.cisbp("Arabidopsis thaliana", family="WRKY")
    assert um.databases is databases
