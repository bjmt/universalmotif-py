"""Time one task of the Python package universalmotif.

    python bench.py TASK SIZE NTHREADS [--noop] [--out FILE] [--data DIR]

Loads the inputs of TASK at SIZE, then times the task alone and prints a
tab-separated line: language, task, size, threads, seconds and a short
summary of the result. With ``--noop`` the inputs are loaded but the task is
not run, which gives the memory used before the task. With ``--out`` the
result is written to FILE for check_outputs.py. The inputs come from
prepare_data.py, and each task mirrors the one of the same name in bench.R.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

import universalmotif as um

# Motifs that cannot reach the P-value cutoff are reported with a warning,
# which the R package does not give.
warnings.simplefilter("ignore")

PVALUES = list(10 ** -np.linspace(2, 8, 100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("size", type=int)
    parser.add_argument("nthreads", type=int)
    parser.add_argument("--noop", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    run, summarize, write = TASKS[args.task](args.data, args.size, args.nthreads)
    gc.collect()

    seconds, summary = float("nan"), "noop"
    if not args.noop:
        start = time.perf_counter()
        result = run()
        seconds = time.perf_counter() - start
        summary = summarize(result)
        if args.out is not None:
            write(result, args.out)

    print("\t".join(["Python", args.task, str(args.size), str(args.nthreads), f"{seconds:.6f}", summary]))


# Inputs ----------------------------------------------------------------------


def motifs_n(file: Path, n: int) -> list[um.Motif]:
    motifs = um.read_meme(file)
    return list(itertools.islice(itertools.cycle(motifs), n))


def read_seqs(file: Path, n: int) -> dict[str, str]:
    return dict(itertools.islice(um.read_fasta(file).items(), n))


# Written results, read by check_outputs.py -----------------------------------


def write_matrices(motifs, file: Path) -> None:
    pl.DataFrame(
        {
            "name": [m.name for m in motifs],
            "values": [",".join(f"{v:.10g}" for v in m.matrix.flatten(order="F")) for m in motifs],
        }
    ).write_csv(file, separator="\t")


def write_hits(hits: pl.DataFrame, file: Path) -> None:
    hits.select("motif", "sequence", "start", "end", "strand", "score", "pvalue").write_csv(
        file, separator="\t"
    )


# Tasks -----------------------------------------------------------------------
#
# Each returns the call to time, a summary of its result and a writer.


def scan_task(motif_file: Path, n_motifs: int, seq_file: Path, size: int, nthreads: int):
    motifs = um.read_meme(motif_file)[:n_motifs]
    seqs = read_seqs(seq_file, size)
    return (
        lambda: um.scan_sequences(motifs, seqs, pvalue=1e-4, rc=True, nthreads=nthreads),
        lambda x: f"hits={x.height}",
        write_hits,
    )


def read_meme(data, size, nthreads):
    file = data / f"motifs_{size}.meme"
    return (lambda: um.read_meme(file), lambda x: f"motifs={len(x)}", write_matrices)


def write_meme(data, size, nthreads):
    motifs = um.read_meme(data / f"motifs_{size}.meme")
    file = Path(tempfile.mkdtemp()) / "motifs.meme"
    return (
        lambda: um.write_meme(motifs, file, overwrite=True),
        lambda x: f"bytes={file.stat().st_size}",
        lambda x, out: shutil.copyfile(file, out),
    )


def convert_type(data, size, nthreads):
    motifs = motifs_n(data / "motifs_879.meme", size)
    return (
        lambda: um.convert_type(motifs, "PWM"),
        lambda x: f"motifs={len(x)}",
        lambda x, file: write_matrices(x[:879], file),
    )


def create_motif(data, size, nthreads):
    with open(data / "sites.txt") as handle:
        sites = [line.rstrip("\n") for line in itertools.islice(handle, size)]
    return (
        lambda: um.create_motif(sites, type="PCM"),
        lambda x: f"nsites={int(x.nsites)}",
        lambda x, file: write_matrices([x], file),
    )


def scan_50(data, size, nthreads):
    return scan_task(data / "motifs_879.meme", 50, data / "windows.fa", size, nthreads)


def scan_400(data, size, nthreads):
    return scan_task(data / "motifs_879.meme", 400, data / "windows.fa", size, nthreads)


def scan_chromosome(data, size, nthreads):
    motifs = um.read_meme(data / "plants.meme")[:10]
    chrom = {name: seq[:size] for name, seq in um.read_fasta(data / "chr1.fa").items()}
    return (
        lambda: um.scan_sequences(motifs, chrom, pvalue=1e-4, rc=True, nthreads=nthreads),
        lambda x: f"hits={x.height}",
        write_hits,
    )


def enrich_motifs(data, size, nthreads):
    motifs = um.read_meme(data / "plants.meme")[:100]
    targets = read_seqs(data / "peaks.fa", size)
    bkg = read_seqs(data / "background.fa", size)
    return (
        lambda: um.enrich_motifs(motifs, targets, bkg, pvalue=1e-4, qvalue=1, nthreads=nthreads),
        lambda x: f"significant={(x['qvalue'] < 0.05).sum()}",
        lambda x, file: x.select("motif", "target_seq_hits", "bkg_seq_hits", "pvalue", "qvalue").write_csv(
            file, separator="\t"
        ),
    )


def motif_pvalue(data, size, nthreads):
    motifs = motifs_n(data / "motifs_879.meme", size)
    pvalues = [PVALUES] * len(motifs)

    def write(x, file):
        pl.DataFrame(
            {
                "motif": np.repeat(np.arange(len(x)), len(PVALUES)),
                "pvalue": np.tile(PVALUES, len(x)),
                "score": np.concatenate(x),
            }
        ).write_csv(file, separator="\t")

    return (
        lambda: um.motif_pvalue(motifs, pvalue=pvalues, nthreads=nthreads),
        lambda x: f"scores={sum(len(s) for s in x)}",
        write,
    )


def compare_motifs(data, size, nthreads):
    motifs = motifs_n(data / "motifs_879.meme", size)
    return (
        lambda: um.compare_motifs(motifs, matrix_out="pvalue", nthreads=nthreads),
        lambda x: f"cells={x.size}",
        lambda x, file: np.savetxt(file, x, delimiter="\t", fmt="%.10g"),
    )


def merge_similar(data, size, nthreads):
    motifs = motifs_n(data / "motifs_879.meme", size)
    return (
        lambda: um.merge_similar(motifs, nthreads=nthreads),
        lambda x: f"motifs={len(x)}",
        write_matrices,
    )


def shuffle_task(k):
    def task(data, size, nthreads):
        seqs = read_seqs(data / "windows.fa", size)
        return (
            lambda: um.shuffle_sequences(seqs, k=k, seed=1, nthreads=nthreads),
            lambda x: f"bases={sum(len(s) for s in x.values())}",
            lambda x, file: um.write_fasta(x, file, overwrite=True),
        )

    return task


def get_bkg(data, size, nthreads):
    seqs = read_seqs(data / "windows.fa", size)
    return (
        lambda: um.get_bkg(seqs, k=[1, 2, 3], nthreads=nthreads),
        lambda x: f"kmers={x.height}",
        lambda x, file: x.select("kmer", "count").write_csv(file, separator="\t"),
    )


def sequence_complexity(data, size, nthreads):
    seqs = read_seqs(data / "windows.fa", size)
    return (
        lambda: um.sequence_complexity(seqs, window_size=20, method="WoottonFederhen", nthreads=nthreads),
        lambda x: f"windows={x.height}",
        lambda x, file: x.write_csv(file, separator="\t"),
    )


TASKS = {
    "read_meme": read_meme,
    "write_meme": write_meme,
    "convert_type": convert_type,
    "create_motif": create_motif,
    "scan_50": scan_50,
    "scan_400": scan_400,
    "scan_chromosome": scan_chromosome,
    "enrich_motifs": enrich_motifs,
    "motif_pvalue": motif_pvalue,
    "compare_motifs": compare_motifs,
    "merge_similar": merge_similar,
    "shuffle_k1": shuffle_task(1),
    "shuffle_k2": shuffle_task(2),
    "get_bkg": get_bkg,
    "sequence_complexity": sequence_complexity,
}


if __name__ == "__main__":
    main()
