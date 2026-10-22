"""Motifs from public databases: JASPAR, HOCOMOCO and CIS-BP.

Each function downloads what it needs the first time it is called and keeps
it in a cache directory, so later calls work offline and return the same
motifs. Pass ``refresh=True`` to download again, for example to pick up a
new JASPAR release when asking for the latest one. Only the standard library
is used for downloading.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import _core
from ._api import create_motif
from ._core import Motif

JASPAR_API = "https://jaspar.elixir.no/api/v1"
HOCOMOCO12_URL = "https://hocomoco12.autosome.org/final_bundle/hocomoco12"
HOCOMOCO11_URL = "https://hocomoco11.autosome.org/final_bundle/hocomoco11"
CISBP_URL = "https://cisbp.ccbr.utoronto.ca"

_TIMEOUT = 120


def cache_dir() -> Path:
    """The directory downloads are kept in.

    It is the ``UNIVERSALMOTIF_CACHE`` environment variable if that is set,
    and otherwise a ``universalmotif`` folder in the usual place for caches:
    ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows, and
    ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere.

    Examples
    --------
    >>> import os
    >>> import universalmotif as um
    >>> os.environ["UNIVERSALMOTIF_CACHE"] = "/tmp/motif-cache"
    >>> um.databases.cache_dir().as_posix()
    '/tmp/motif-cache'
    >>> del os.environ["UNIVERSALMOTIF_CACHE"]
    """
    env = os.environ.get("UNIVERSALMOTIF_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "universalmotif"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "universalmotif" / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "universalmotif"


def clear_cache() -> int:
    """Delete every download in [`cache_dir()`][universalmotif.databases.cache_dir],
    returning the number of files removed.

    Examples
    --------
    >>> import os, tempfile
    >>> import universalmotif as um
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     os.environ["UNIVERSALMOTIF_CACHE"] = tmp
    ...     um.databases.clear_cache()
    0
    >>> del os.environ["UNIVERSALMOTIF_CACHE"]
    """
    root = cache_dir() / "downloads"
    if not root.exists():
        return 0
    n = sum(1 for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root)
    return n


def _user_agent() -> str:
    from . import __version__

    return f"universalmotif/{__version__} (https://github.com/bjmt/universalmotif-py)"


def _download(url: str, data: dict[str, str | list[str]] | None = None) -> bytes:
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data, doseq=True).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _cached(url: str, refresh: bool, data: dict | None = None, suffix: str = "") -> Path:
    """The path of a cached download of ``url``, downloading it if needed."""
    key = url if data is None else url + "?" + json.dumps(data, sort_keys=True)
    path = cache_dir() / "downloads" / (hashlib.sha256(key.encode()).hexdigest()[:24] + suffix)
    if path.exists() and not refresh:
        return path
    content = _download(url, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)
    return path


# JASPAR ----------------------------------------------------------------------


def jaspar(
    collection: str = "CORE",
    tax_group: str | None = None,
    name: str | None = None,
    tax_id: int | None = None,
    tf_class: str | None = None,
    tf_family: str | None = None,
    data_type: str | None = None,
    release: int | None = None,
    all_versions: bool = False,
    metadata: bool = False,
    refresh: bool = False,
) -> list[Motif]:
    """Count matrices from JASPAR, through its REST API.

    The filters are those of the API: ``collection`` (such as ``"CORE"``,
    ``"UNVALIDATED"`` or ``"CNE"``), ``tax_group`` (such as
    ``"vertebrates"``, ``"plants"`` or ``"insects"``), ``name`` for the
    transcription factor, ``tax_id`` for the NCBI taxonomy ID of a species
    (9606 for human), ``tf_class``, ``tf_family`` and ``data_type`` (such
    as ``"ChIP-seq"`` or ``"SELEX"``). Text filters must match JASPAR's
    names exactly, and the API ignores filters it does not know, so a
    misspelt name gives the unfiltered set rather than an error. ``release`` picks a JASPAR release by
    year, the latest by default, and only the latest version of each matrix
    is returned unless ``all_versions``.

    Motifs are PCMs named by their matrix ID, with the transcription factor
    as the alternative name. The list of matrices does not say more than
    that, so with ``metadata`` each matrix's record is also fetched, one
    request per matrix the first time, which fills in the family, the
    species as the organism, and the class, taxonomic group, data type,
    UniProt IDs and PubMed IDs in ``extrainfo``.

    Downloads are cached, so asking for the latest release keeps returning
    the motifs of the first download until ``refresh`` is given.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.databases.jaspar(tax_group="plants", name="ABF1")  # doctest: +SKIP
    >>> motifs[0].name, motifs[0].altname  # doctest: +SKIP
    ('MA0570.3', 'ABF1')
    """
    params = {
        "collection": collection,
        "tax_group": tax_group,
        "name": name,
        "tax_id": tax_id,
        "tf_class": tf_class,
        "tf_family": tf_family,
        "data_type": data_type,
        "release": release,
    }
    params = {k: str(v) for k, v in params.items() if v is not None}
    if not all_versions:
        params["version"] = "latest"
    count_url = _jaspar_url(params, page_size=1, fmt="json")
    count = json.loads(_cached(count_url, refresh).read_text())["count"]
    page_size = 1000
    motifs: list[Motif] = []
    for page in range(1, -(-count // page_size) + 1):
        url = _jaspar_url(params, page_size=page_size, fmt="jaspar", page=page)
        text = _cached(url, refresh, suffix=".txt").read_text()
        if text.strip():
            motifs.extend(_core._read_jaspar(text))
    if metadata:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda m: _jaspar_record(m.name, refresh), motifs))
        motifs = [_with_jaspar_metadata(m, r) for m, r in zip(motifs, records)]
    return motifs


def _jaspar_url(params: dict[str, str], page_size: int, fmt: str, page: int = 1) -> str:
    query = dict(params, page_size=str(page_size), page=str(page), format=fmt)
    return f"{JASPAR_API}/matrix/?{urllib.parse.urlencode(sorted(query.items()))}"


def _jaspar_record(matrix_id: str, refresh: bool) -> dict:
    url = f"{JASPAR_API}/matrix/{urllib.parse.quote(matrix_id)}/?format=json"
    return json.loads(_cached(url, refresh).read_text())


def _join(values) -> str | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        values = "; ".join(str(v) for v in values if v not in (None, ""))
    return str(values) or None


def _with_jaspar_metadata(m: Motif, r: dict) -> Motif:
    species = r.get("species") or []
    organism = _join([s.get("name") for s in species if isinstance(s, dict)])
    extrainfo = dict(m.extrainfo)
    for key, value in (
        ("class", r.get("class")),
        ("tax_group", r.get("tax_group")),
        ("data_type", r.get("type")),
        ("uniprot", r.get("uniprot_ids")),
        ("pubmed", r.get("medline")),
        ("collection", r.get("collection")),
    ):
        text = _join(value)
        if text:
            extrainfo[key] = text
    return m.replace(
        altname=r.get("name") or m.altname,
        family=_join(r.get("family")),
        organism=organism,
        extrainfo=extrainfo,
    )


# HOCOMOCO --------------------------------------------------------------------


def hocomoco(
    version: int = 12,
    collection: str = "CORE",
    species: str | None = None,
    refresh: bool = False,
) -> list[Motif]:
    """Count matrices from HOCOMOCO, for human and mouse transcription factors.

    Version 12 has the collections ``"CORE"``, ``"INVIVO"``, ``"INVITRO"``
    and ``"RSNP"``, each with motifs for human and mouse together, and
    ``species`` keeps only those built from ``"HUMAN"`` or ``"MOUSE"`` data.
    Version 11 has ``"CORE"`` and ``"FULL"``, with separate files for each
    species, so there ``species`` picks one and defaults to ``"HUMAN"``.

    Motifs are PCMs named by their HOCOMOCO ID, such as
    ``AHR.H12CORE.0.P.B``, with the transcription factor as the alternative
    name and its TFClass family as the family. The quality grade, A to D, and
    a few more details go in ``extrainfo``.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.databases.hocomoco()  # doctest: +SKIP
    >>> motifs[0].name, motifs[0].altname, motifs[0].family  # doctest: +SKIP
    ('AHR.H12CORE.0.P.B', 'AHR', 'PAS')
    """
    collection = collection.upper()
    if species is not None:
        species = species.upper()
        if species not in ("HUMAN", "MOUSE"):
            raise ValueError('species must be "HUMAN" or "MOUSE"')
    if version == 12:
        if collection not in ("CORE", "INVIVO", "INVITRO", "RSNP"):
            raise ValueError('collection must be "CORE", "INVIVO", "INVITRO" or "RSNP"')
        name = f"H12{collection}"
        path = _cached(f"{HOCOMOCO12_URL}/{name}/{name}_annotation.jsonl", refresh, suffix=".jsonl")
        out = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    origin = rec.get("original_motif", {}).get("species")
                    if species is None or origin == species:
                        out.append(_hocomoco12_motif(rec))
        return out
    if version == 11:
        if collection not in ("CORE", "FULL"):
            raise ValueError('collection must be "CORE" or "FULL"')
        species = species or "HUMAN"
        c = collection.lower()
        base = f"{HOCOMOCO11_URL}/{c}/{species}/mono"
        meme = _cached(f"{base}/HOCOMOCOv11_{c}_{species}_mono_meme_format.meme", refresh,
                       suffix=".meme")
        annot = _cached(f"{base}/HOCOMOCOv11_{c}_annotation_{species}_mono.tsv", refresh,
                        suffix=".tsv")
        with open(annot) as f:
            rows = {r["Model"]: r for r in csv.DictReader(f, delimiter="\t")}
        motifs = _core._read_meme(meme.read_text(), False)
        return [_hocomoco11_motif(m, rows.get(m.name), species) for m in motifs]
    raise ValueError("version must be 11 or 12")


def _hocomoco12_motif(rec: dict) -> Motif:
    info = rec.get("masterlist_info", {})
    origin = rec.get("original_motif", {})
    extrainfo = {
        "quality": rec.get("quality"),
        "data_type": rec.get("datatype"),
        "class": info.get("tfclass_class"),
        "subfamily": info.get("tfclass_subfamily"),
        "tfclass_id": info.get("tfclass_id"),
        "collection": rec.get("collection"),
    }
    organism = {"HUMAN": "Homo sapiens", "MOUSE": "Mus musculus"}.get(origin.get("species"))
    return create_motif(
        np.array(rec["pcm"], dtype=float).T,
        alphabet="DNA",
        type="PCM",
        name=rec["name"],
        altname=rec.get("tf"),
        family=info.get("tfclass_family"),
        organism=organism,
        extrainfo={k: str(v) for k, v in extrainfo.items() if v not in (None, "")},
    )


def _hocomoco11_motif(m: Motif, row: dict | None, species: str) -> Motif:
    organism = {"HUMAN": "Homo sapiens", "MOUSE": "Mus musculus"}[species]
    if row is None:
        return m.replace(organism=organism)
    family = re.sub(r"\{[^}]*\}$", "", row.get("TF family") or "").strip() or None
    extrainfo = {
        "quality": row.get("Quality"),
        "data_type": row.get("Data source"),
        "subfamily": re.sub(r"\{[^}]*\}$", "", row.get("TF subfamily") or "").strip(),
        "uniprot": row.get("UniProt ID"),
    }
    return m.replace(
        altname=row.get("Transcription factor") or None,
        family=family,
        organism=organism,
        extrainfo={k: v for k, v in extrainfo.items() if v},
    )


# CIS-BP ----------------------------------------------------------------------


def cisbp(
    species: str,
    family: str | Sequence[str] | None = None,
    inferred: bool = False,
    refresh: bool = False,
) -> list[Motif]:
    """Motifs from CIS-BP for one species.

    CIS-BP builds an archive of a species' motifs on request, which is
    downloaded and cached, and then read with
    [`read_cisbp_archive()`][universalmotif.databases.read_cisbp_archive].
    ``species`` is written as in CIS-BP, such as ``"Homo_sapiens"``, though
    spaces may be used instead of underscores. ``family`` keeps only motifs of
    some transcription factor families, such as ``"bZIP"``, and ``inferred``
    adds the motifs CIS-BP assigns to a factor from similar factors, which by
    default are left out.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.databases.cisbp("Arabidopsis thaliana", family="WRKY")  # doctest: +SKIP
    >>> motifs[0].family, motifs[0].organism  # doctest: +SKIP
    ('WRKY', 'Arabidopsis thaliana')
    """
    sp = species.strip().replace(" ", "_")
    form = {
        "selSpec": sp,
        "Spec[]": ["TF_Information", "PWMs"],
        "submit": "Download Species Archive!",
    }
    key_url = f"{CISBP_URL}/bulk_archive.php#{sp}"
    zip_path = cache_dir() / "downloads" / (hashlib.sha256(key_url.encode()).hexdigest()[:24] + ".zip")
    if refresh or not zip_path.exists():
        page = _download(f"{CISBP_URL}/bulk_archive.php", form).decode(errors="replace")
        link = re.search(r'href="(tmp/[^"]+\.zip)"', page)
        if link is None:
            raise ValueError(f"CIS-BP did not offer an archive for the species {sp!r}")
        content = _download(f"{CISBP_URL}/{urllib.parse.quote(link.group(1))}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_name(zip_path.name + ".part")
        tmp.write_bytes(content)
        tmp.replace(zip_path)
    return read_cisbp_archive(zip_path, family=family, inferred=inferred)


def read_cisbp_archive(
    path: str | os.PathLike[str],
    family: str | Sequence[str] | None = None,
    inferred: bool = False,
) -> list[Motif]:
    """Read a CIS-BP download, either the zip archive from its bulk download
    page or the folder it unpacks to.

    The archive needs ``TF_Information_all_motifs.txt``, or
    ``TF_Information.txt``, and the ``pwms_all_motifs`` folder of
    probability matrices. Motifs are PPMs named by their CIS-BP motif ID,
    with the transcription factors they belong to as the alternative name,
    the family and the species, and the experiment type and source in
    ``extrainfo``. Motifs CIS-BP infers from similar factors are left out
    unless ``inferred``, ``family`` keeps only some families, and motifs
    whose matrix file is empty are skipped.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.databases.read_cisbp_archive("Arabidopsis_thaliana.zip")  # doctest: +SKIP
    """
    files = _ArchiveFiles(Path(path))
    info_name = next(
        (n for n in ("TF_Information_all_motifs.txt", "TF_Information.txt") if files.has(n)), None
    )
    if info_name is None:
        raise ValueError(f"{path} has no TF_Information_all_motifs.txt or TF_Information.txt")
    families = None if family is None else {family} if isinstance(family, str) else set(family)
    reader = csv.reader(io.StringIO(files.read(info_name)), delimiter="\t")
    header = next(reader)
    col = {}
    for i, h in enumerate(header):
        col.setdefault(h, i)
    by_motif: dict[str, list[list[str]]] = {}
    for row in reader:
        if len(row) < len(header):
            continue
        motif_id = row[col["Motif_ID"]]
        if motif_id in (".", ""):
            continue
        if not inferred and row[col["TF_Status"]] != "D":
            continue
        if families is not None and row[col["Family_Name"]] not in families:
            continue
        by_motif.setdefault(motif_id, []).append(row)
    def uniq(rows: list[list[str]], name: str) -> str | None:
        vals = list(dict.fromkeys(r[col[name]] for r in rows if name in col and r[col[name]]))
        return ", ".join(vals) or None

    out = []
    for motif_id, rows in by_motif.items():
        text = files.read_pwm(motif_id)
        if text is None:
            continue
        lines = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        letters = lines[0][1:]
        matrix = np.array([[float(v) for v in ln[1:]] for ln in lines[1:]], dtype=float).T


        extrainfo = {
            "motif_type": uniq(rows, "Motif_Type"),
            "source": uniq(rows, "MSource_Identifier"),
            "pubmed": uniq(rows, "PMID"),
            "status": uniq(rows, "TF_Status"),
        }
        organism = uniq(rows, "TF_Species")
        out.append(
            create_motif(
                matrix,
                letters=letters,
                type="PPM",
                name=motif_id,
                altname=uniq(rows, "TF_Name"),
                family=uniq(rows, "Family_Name"),
                organism=organism.replace("_", " ") if organism else None,
                extrainfo={k: v for k, v in extrainfo.items() if v and v != "NULL"},
            )
        )
    return out


class _ArchiveFiles:
    """Files of a CIS-BP download, from a zip archive or a folder."""

    def __init__(self, path: Path):
        self.path = path
        if path.is_dir():
            self.zip = None
            self.names = {p.relative_to(path).as_posix(): p for p in path.rglob("*") if p.is_file()}
        else:
            self.zip = zipfile.ZipFile(path)
            self.names = {n: n for n in self.zip.namelist() if not n.endswith("/")}
        self.by_base = {}
        self.pwms = {}
        for n in self.names:
            base = n.rsplit("/", 1)[-1]
            self.by_base.setdefault(base, n)
            if n.rsplit("/", 2)[-2:-1] == ["pwms_all_motifs"]:
                self.pwms[base] = n

    def has(self, base: str) -> bool:
        return base in self.by_base

    def read(self, base: str) -> str:
        n = self.names[self.by_base[base]]
        data = self.zip.read(n) if self.zip is not None else Path(n).read_bytes()
        return data.decode(errors="replace")

    def read_pwm(self, motif_id: str) -> str | None:
        full = self.pwms.get(f"{motif_id}.txt")
        if full is None:
            return None
        n = self.names[full]
        data = self.zip.read(n) if self.zip is not None else Path(n).read_bytes()
        return data.decode(errors="replace")


__all__ = ["cache_dir", "cisbp", "clear_cache", "hocomoco", "jaspar", "read_cisbp_archive"]
