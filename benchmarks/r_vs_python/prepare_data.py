"""Write the input files shared by the R and Python benchmarks.

    python prepare_data.py [--cache DIR] [--out DIR]

Every input is written once to a plain file that both languages read, so the
two sides work on identical data. Only the standard library is used, so the
inputs do not depend on either package. The random draws use a fixed seed.

These downloads are kept in the cache directory, which is the one the
guides use (~/.cache/universalmotif-docs) by default, and fetched if they are
missing:

- the JASPAR 2024 vertebrate and plant collections in MEME format,
- the TAIR10 genome of Arabidopsis thaliana from Ensembl Plants,
- the ChIP-seq peaks of VAL1 (Yuan et al. 2021) from the R package.

Files written to the output directory:

- ``plants.meme``: a copy of the plant JASPAR file.
- ``motifs_<n>.meme``: the vertebrate JASPAR motifs, repeated until there
  are n with copies renamed, for reading larger files. The first of these
  is the JASPAR file itself.
- ``chr1.fa``: chromosome 1 of TAIR10.
- ``windows.fa``: 500 bp windows drawn from chromosomes 1 to 5, holding
  only A, C, G and T.
- ``peaks.fa``: the sequences of the VAL1 peaks.
- ``background.fa``: one random window per peak, of the same length and
  from the same chromosome, holding only A, C, G and T.
- ``sites.txt``: 20 bp sites drawn from chromosomes 1 to 5, one per line.
"""

from __future__ import annotations

import argparse
import gzip
import random
import re
import shutil
import urllib.request
from pathlib import Path

URLS = {
    "vertebrates": "https://jaspar.elixir.no/download/data/2024/CORE/"
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
    "plants": "https://jaspar.elixir.no/download/data/2024/CORE/"
    "JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt",
    "genome": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/"
    "arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz",
    "peaks": "https://raw.githubusercontent.com/bjmt/universalmotif/master/"
    "inst/extdata/VAL1-GFP_val1_calledPeaks.bed",
}

CHROMOSOMES = ["1", "2", "3", "4", "5"]
MEME_SIZES = [879, 8790]
N_WINDOWS = 100_000
WINDOW_WIDTH = 500
N_SITES = 2_000_000
SITE_WIDTH = 20
SEED = 1
NOT_ACGT = re.compile("[^ACGT]")


def fetch(url: str, cache: Path) -> Path:
    """Download a file into the cache unless it is there already."""
    path = cache / url.rsplit("/", 1)[-1]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def read_genome(path: Path) -> dict[str, str]:
    """Read a gzipped FASTA file, naming each sequence by its first word."""
    seqs: dict[str, list[str]] = {}
    name = None
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def write_fasta(path: Path, records) -> None:
    with open(path, "w") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def draw_window(rng: random.Random, seq: str, width: int) -> str:
    """Draw a window holding only A, C, G and T from a sequence."""
    while True:
        start = rng.randrange(len(seq) - width + 1)
        window = seq[start : start + width]
        if not NOT_ACGT.search(window):
            return window


def repeat_meme(source: Path, n: int, out: Path) -> None:
    """Write the first n motifs of the source file repeated as needed, giving
    each copy after the first a suffix so that names stay unique."""
    text = source.read_text()
    first = text.index("\nMOTIF ") + 1
    header, body = text[:first], text[first:]
    blocks = ["MOTIF " + b for b in body.split("MOTIF ") if b.strip()]
    with open(out, "w") as handle:
        handle.write(header)
        for i in range(n):
            block = blocks[i % len(blocks)]
            copy = i // len(blocks)
            if copy:
                line, rest = block.split("\n", 1)
                words = line.split()
                words[1] = f"{words[1]}_{copy}"
                block = " ".join(words) + "\n" + rest
            handle.write(block if block.endswith("\n\n") else block.rstrip("\n") + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "universalmotif-docs")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = {key: fetch(url, args.cache) for key, url in URLS.items()}

    shutil.copyfile(files["plants"], args.out / "plants.meme")
    for n in MEME_SIZES:
        repeat_meme(files["vertebrates"], n, args.out / f"motifs_{n}.meme")

    genome = read_genome(files["genome"])
    write_fasta(args.out / "chr1.fa", [("1", genome["1"])])

    rng = random.Random(SEED)
    lengths = [len(genome[c]) for c in CHROMOSOMES]

    def random_chromosome() -> str:
        return rng.choices(CHROMOSOMES, weights=lengths)[0]

    write_fasta(
        args.out / "windows.fa",
        (
            (f"window{i + 1}", draw_window(rng, genome[random_chromosome()], WINDOW_WIDTH))
            for i in range(N_WINDOWS)
        ),
    )

    with open(args.out / "sites.txt", "w") as handle:
        for _ in range(N_SITES):
            handle.write(draw_window(rng, genome[random_chromosome()], SITE_WIDTH) + "\n")

    peaks = []
    with open(files["peaks"]) as handle:
        for line in handle:
            chrom, start, end = line.split("\t")[:3]
            peaks.append((chrom, int(start), int(end)))
    write_fasta(
        args.out / "peaks.fa",
        (
            (f"{c}:{s}-{e}", genome[c][s:e])
            for c, s, e in peaks
        ),
    )
    write_fasta(
        args.out / "background.fa",
        (
            (f"background{i + 1}", draw_window(rng, genome[c], e - s))
            for i, (c, s, e) in enumerate(peaks)
        ),
    )


if __name__ == "__main__":
    main()
