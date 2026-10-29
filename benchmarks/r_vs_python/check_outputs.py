"""Check that the R and Python packages agree on each task's result.

    python check_outputs.py DIR

DIR holds the files written by ``bench.R --out DIR/R_<task>.tsv`` and
``bench.py --out DIR/Python_<task>.tsv``, as run.zsh writes them with
``--check``. Prints one line per task found in both languages, and exits with
an error if any disagree.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

import universalmotif as um

RTOL = 1e-6
ATOL = 1e-6


def read_tsv(file: Path) -> pl.DataFrame:
    return pl.read_csv(file, separator="\t", infer_schema_length=None)


def matrices(file: Path) -> dict[str, np.ndarray]:
    table = read_tsv(file)
    return {
        name: np.array([float(v) for v in values.split(",")])
        for name, values in zip(table["name"], table["values"])
    }


def check_matrices(r: Path, py: Path) -> str:
    a, b = matrices(r), matrices(py)
    if len(a) != len(b):
        return f"FAIL {len(a)} motifs in R, {len(b)} in Python"
    if list(a) != list(b):
        return "FAIL motif names differ"
    worst = max(float(np.max(np.abs(a[k] - b[k]))) if a[k].shape == b[k].shape else np.inf for k in a)
    ok = all(a[k].shape == b[k].shape and np.allclose(a[k], b[k], rtol=RTOL, atol=ATOL) for k in a)
    return f"{'ok  ' if ok else 'FAIL'} {len(a)} motifs, largest difference {worst:.2g}"


def check_merged(r: Path, py: Path) -> str:
    # Merged motifs are named after their members, so compare those as sets.
    a, b = matrices(r), matrices(py)
    key = lambda name: frozenset(name.split("/"))  # noqa: E731
    a = {key(k): v for k, v in a.items()}
    b = {key(k): v for k, v in b.items()}
    if set(a) != set(b):
        return f"FAIL clusters differ ({len(a)} motifs in R, {len(b)} in Python)"
    worst = max(float(np.max(np.abs(a[k] - b[k]))) if a[k].shape == b[k].shape else np.inf for k in a)
    ok = worst <= 1e-4
    return f"{'ok  ' if ok else 'FAIL'} {len(a)} motifs after merging, largest difference {worst:.2g}"


def check_meme_files(r: Path, py: Path) -> str:
    a, b = um.read_meme(r), um.read_meme(py)
    same = len(a) == len(b) and all(
        x.name == y.name and x.matrix.shape == y.matrix.shape and np.allclose(x.matrix, y.matrix, atol=ATOL)
        for x, y in zip(a, b)
    )
    lines_a, lines_b = r.read_text().splitlines(), py.read_text().splitlines()
    differ = sum(x != y for x, y in zip(lines_a, lines_b)) + abs(len(lines_a) - len(lines_b))
    return f"{'ok  ' if same else 'FAIL'} {len(a)} motifs read back alike, {differ} of {len(lines_a)} lines differ"


def check_hits(r: Path, py: Path) -> str:
    key = ["motif", "sequence", "start", "end", "strand"]
    a, b = read_tsv(r).sort(key), read_tsv(py).sort(key)
    if a.height != b.height:
        return f"FAIL {a.height} hits in R, {b.height} in Python"
    if not a.select(key).cast(pl.String).equals(b.select(key).cast(pl.String)):
        return f"FAIL {a.height} hits each, but at different places"
    ok = np.allclose(a["score"], b["score"], atol=ATOL) and np.allclose(a["pvalue"], b["pvalue"], rtol=RTOL)
    return f"{'ok  ' if ok else 'FAIL'} {a.height} hits, same places, scores and P-values"


def check_enrichment(r: Path, py: Path) -> str:
    a, b = read_tsv(r), read_tsv(py)
    if set(a["motif"]) != set(b["motif"]):
        return "FAIL different motifs reported"
    j = a.join(b, on="motif", suffix="_py")
    counts = (j["target_seq_hits"] == j["target_seq_hits_py"]).all() and (
        j["bkg_seq_hits"] == j["bkg_seq_hits_py"]
    ).all()
    pvals = np.allclose(j["pvalue"], j["pvalue_py"], rtol=RTOL, atol=0) and np.allclose(
        j["qvalue"], j["qvalue_py"], rtol=RTOL, atol=0
    )
    return f"{'ok  ' if counts and pvals else 'FAIL'} {j.height} motifs, same hit counts, P- and q-values"


def check_thresholds(r: Path, py: Path) -> str:
    a, b = read_tsv(r), read_tsv(py)
    x = a["score"].cast(pl.Float64).to_numpy()
    y = b["score"].cast(pl.Float64).to_numpy()
    if x.shape != y.shape:
        return f"FAIL {x.size} scores in R, {y.size} in Python"
    finite = np.isfinite(x) & np.isfinite(y)
    same_inf = np.array_equal(np.isfinite(x), np.isfinite(y))
    worst = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
    ok = same_inf and worst <= 1e-3
    return f"{'ok  ' if ok else 'FAIL'} {x.size} scores ({(~finite).sum()} infinite), largest difference {worst:.2g}"


def check_matrix(r: Path, py: Path) -> str:
    a, b = np.loadtxt(r, delimiter="\t", ndmin=2), np.loadtxt(py, delimiter="\t", ndmin=2)
    if a.shape != b.shape:
        return f"FAIL shapes {a.shape} and {b.shape}"
    worst = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), np.finfo(float).tiny)))
    ok = np.allclose(a, b, rtol=RTOL, atol=0)
    return f"{'ok  ' if ok else 'FAIL'} {a.shape[0]} x {a.shape[1]} P-values, largest relative difference {worst:.2g}"


def kmer_counts(file: Path, k: int) -> dict[str, Counter]:
    seqs = um.read_fasta(file)
    return {name: Counter(s[i : i + k] for i in range(len(s) - k + 1)) for name, s in seqs.items()}


def check_shuffle(k: int):
    def check(r: Path, py: Path) -> str:
        # Shuffles are random, so check that both keep each sequence's k-mer
        # counts, which must then be the same in both.
        a, b = kmer_counts(r, k), kmer_counts(py, k)
        ok = a == b
        return f"{'ok  ' if ok else 'FAIL'} {len(a)} sequences with the same {k}-mer counts"

    return check


def check_kmers(r: Path, py: Path) -> str:
    a, b = read_tsv(r), read_tsv(py)
    j = a.join(b, on="kmer", suffix="_py", how="full", coalesce=True)
    ok = a.height == b.height == j.height and (j["count"] == j["count_py"]).all()
    return f"{'ok  ' if ok else 'FAIL'} {a.height} k-mer counts"


def check_complexity(r: Path, py: Path) -> str:
    a, b = read_tsv(r), read_tsv(py)
    key = ["sequence", "start", "end"]
    if a.height != b.height or not a.select(key).equals(b.select(key)):
        return f"FAIL {a.height} windows in R, {b.height} in Python"
    worst = float(np.max(np.abs(a["complexity"].to_numpy() - b["complexity"].to_numpy())))
    ok = worst <= 1e-6
    return f"{'ok  ' if ok else 'FAIL'} {a.height} windows, largest difference {worst:.2g}"


CHECKS = {
    "read_meme": check_matrices,
    "write_meme": check_meme_files,
    "convert_type": check_matrices,
    "create_motif": check_matrices,
    "scan_50": check_hits,
    "scan_400": check_hits,
    "scan_chromosome": check_hits,
    "enrich_motifs": check_enrichment,
    "motif_pvalue": check_thresholds,
    "compare_motifs": check_matrix,
    "merge_similar": check_merged,
    "shuffle_k1": check_shuffle(1),
    "shuffle_k2": check_shuffle(2),
    "get_bkg": check_kmers,
    "sequence_complexity": check_complexity,
}


def main() -> None:
    folder = Path(sys.argv[1])
    failed = False
    for task, check in CHECKS.items():
        r, py = folder / f"R_{task}.tsv", folder / f"Python_{task}.tsv"
        if not (r.exists() and py.exists()):
            continue
        result = check(r, py)
        failed |= result.startswith("FAIL")
        print(f"{task:<20} {result}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
