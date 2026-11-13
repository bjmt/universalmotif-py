"""Run the command line examples marked for it and show their outputs.

A shell block on the command-line page is run when a marker follows it:

- ``<!-- output -->`` shows what the commands print,
- ``<!-- output: FILE LINES [COLUMNS] -->`` shows the first lines of a file
  the commands wrote, tables aligned and cut to their first columns,
- ``<!-- image: FILE -->`` shows a figure they drew.

Only marked blocks are run, in order, in one folder prepared with small
made-up inputs: the package's example motifs and sequences, and a random
genome with AP-1 sites planted in its peaks, and G-box and CRE sites
together in half of them. Nothing is
downloaded. The outputs are kept in .cache/cli_examples until the page or
the package changes.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

from mkdocs.structure.files import File

_ROOT = Path(__file__).resolve().parent.parent
_PAGE = _ROOT / "docs" / "command-line.md"
_CACHE = _ROOT / ".cache" / "cli_examples"
_DEST = "cli"
_BLOCK = re.compile(r"```sh\n([^`]*)```\n((?:\n?<!-- (?:output|image)[^>]*-->[ \t]*\n)+)")
_MARKER = re.compile(r"<!-- (output|image)(?::\s*([^>]*?))?\s*-->")
_shown: dict[int, str] = {}
_images: list[str] = []


def _key() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes() + _PAGE.read_bytes())
    for p in sorted((_ROOT / "python" / "universalmotif").glob("*.py")):
        h.update(p.read_bytes())
    for so in (_ROOT / "python" / "universalmotif").glob("_core*.so"):
        h.update(str(so.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]


def _bgzf(data: bytes) -> bytes:
    """Data compressed as bgzip does, in blocks of at most 64 KB that can be
    read one at a time."""
    out = bytearray()
    for i in range(0, len(data), 65280):
        chunk = data[i:i + 65280]
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        body = c.compress(chunk) + c.flush()
        out += b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00"
        out += struct.pack("<H", len(body) + 25) + body
        out += struct.pack("<II", zlib.crc32(chunk), len(chunk))
    return bytes(out) + bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def _inputs(work: Path) -> None:
    """The files the examples read, made the same way every time."""
    import universalmotif as um

    motifs = um.example_motifs()
    um.write_meme(motifs, work / "motifs.meme")
    um.write_fasta(um.example_sequences("targets"), work / "seqs.fa")
    (work / "sites.fa").write_text(">s1\nTGACTCA\n>s2\nTGAGTCA\n>s3\nTGACTCT\n>s4\nAGACTCA\n")
    rng = random.Random(1)
    genome, peaks, tss = {}, [], []
    for chrom, size in (("chr1", 60000), ("chr2", 40000)):
        seq = "".join(rng.choice("ACGT") for _ in range(size))
        for start in range(1000, size - 2000, 1500):
            mid = start + 250
            seq = seq[:mid] + "TGACTCA" + seq[mid + 7:]
            if rng.random() < 0.5:
                # A G-box and a CRE together, so that the pair co-occurs.
                seq = seq[:mid + 30] + "CACGTG" + seq[mid + 36:]
                seq = seq[:mid - 60] + "TGACGTCA" + seq[mid - 52:]
            peaks.append((chrom, start, start + 500, f"peak{len(peaks) + 1}", rng.randint(0, 1000), "."))
            tss.append((chrom, mid + 60, mid + 61, f"gene{len(tss) + 1}", 0, "+-"[len(tss) % 2]))
        genome[chrom] = seq
    text = "".join(f">{c}\n" + "\n".join(s[i:i + 80] for i in range(0, len(s), 80)) + "\n"
                   for c, s in genome.items())
    (work / "genome.fa.gz").write_bytes(_bgzf(text.encode()))
    for name, rows in (("peaks.bed", peaks), ("tss.bed", tss)):
        (work / name).write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    um.write_fasta(um.extract_ranges(genome, [r[:3] for r in peaks]), work / "peaks.fa")


def _shell(block: str) -> str:
    """A block's commands, run with this Python and two threads at most."""
    umotif = f"{shlex.quote(sys.executable)} -m universalmotif"
    block = block.replace("\\\n", " ").replace("-p 4", "-p 2")
    lines = [re.sub(r"(^|\|\s*)umotif\b", lambda m: m.group(1) + umotif, ln.strip())
             for ln in block.splitlines() if ln.strip()]
    return "set -euo pipefail\n" + "\n".join(lines)


def _table(lines: list[str], ncol: int | None) -> list[str]:
    """Tab-separated lines as aligned columns, cut to the first ncol."""
    rows = [ln.split("\t") for ln in lines]
    cut = ncol is not None and any(len(r) > ncol for r in rows)
    if ncol is not None:
        rows = [r[:ncol] for r in rows]
    widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(max(map(len, rows)))]
    out = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return [ln + ("  ..." if cut else "") for ln in out]


def _render(marker: str, arg: str | None, work: Path, stdout: str, out: Path) -> str:
    if marker == "image":
        src = work / arg.strip()
        name = f"{_PAGE.stem}-{src.name}"
        shutil.copy(src, out / name)
        _images.append(name)
        return f"![The figure written to {src.name}]({_DEST}/{name}){{ loading=lazy }}"
    if not arg:
        lines, title = stdout.rstrip("\n").splitlines(), "Output"
        more = False
        if lines and all("\t" in ln for ln in lines):
            lines = _table(lines, None)
    else:
        parts = arg.split()
        path, nline = work / parts[0], int(parts[1])
        ncol = int(parts[2]) if len(parts) > 2 else None
        text = path.read_text().rstrip("\n").splitlines()
        lines, more, title = text[:nline], len(text) > nline, parts[0]
        if path.suffix in (".tsv", ".bed"):
            lines = _table(lines, ncol)
    if more:
        lines.append("...")
    return f'```text title="{title}"\n' + "\n".join(lines) + "\n```"


def _run(out: Path) -> dict[int, str]:
    markdown = _PAGE.read_text()
    shown: dict[int, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _inputs(work)
        for i, m in enumerate(_BLOCK.finditer(markdown)):
            done = subprocess.run(["bash", "-c", _shell(m.group(1))], cwd=work, capture_output=True,
                                  text=True, timeout=600)
            if done.returncode != 0:
                raise RuntimeError(f"command line example {i + 1} failed:\n{m.group(1)}\n{done.stderr}")
            rendered = [_render(k, a, work, done.stdout, out) for k, a in _MARKER.findall(m.group(2))]
            shown[i] = "\n\n".join(rendered)
    return shown


def on_config(config):
    global _shown, _images
    out = _CACHE / _key()
    saved = out / "outputs.json"
    if saved.exists():
        data = json.loads(saved.read_text())
        _shown = {int(k): v for k, v in data["shown"].items()}
        _images = data["images"]
        return config
    if _CACHE.exists():
        shutil.rmtree(_CACHE)
    out.mkdir(parents=True)
    _images = []
    _shown = _run(out)
    saved.write_text(json.dumps({"shown": _shown, "images": _images}))
    return config


def on_files(files, config):
    out = _CACHE / _key()
    for name in _images:
        files.append(File.generated(config, f"{_DEST}/{name}", abs_src_path=str(out / name)))
    return files


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != _PAGE.name:
        return markdown
    count = iter(range(10_000))

    def replace(m: re.Match) -> str:
        i = next(count)
        return f"```sh\n{m.group(1)}```\n\n{_shown[i]}\n" if i in _shown else m.group(0)

    return _BLOCK.sub(replace, markdown)
