"""Summarize the raw timings written by run.zsh as a Markdown table.

    python summarize.py results/raw.tsv

Times are medians over the runs. Memory is the median peak resident set size
of the runs minus that of the run with ``--noop``, which loads the same
packages and inputs without running the task, so it is the memory the task
itself added at its peak.
"""

from __future__ import annotations

import sys

import polars as pl

LABELS = {
    "read_meme": ("Read MEME motifs", "motifs"),
    "write_meme": ("Write MEME motifs", "motifs"),
    "convert_type": ("Convert motifs to PWMs", "motifs"),
    "create_motif": ("Create a motif from sites", "sites"),
    "scan_50": ("Scan for 50 motifs", "500 bp sequences"),
    "scan_400": ("Scan for 400 motifs", "500 bp sequences"),
    "scan_chromosome": ("Scan a chromosome for 10 motifs", "bp"),
    "enrich_motifs": ("Enrichment of 100 motifs", "peaks"),
    "motif_pvalue": ("Score thresholds for 100 P-values", "motifs"),
    "compare_motifs": ("Compare all against all, with P-values", "motifs"),
    "merge_similar": ("Merge similar motifs", "motifs"),
    "shuffle_k1": ("Shuffle sequences, k = 1", "500 bp sequences"),
    "shuffle_k2": ("Shuffle sequences, k = 2", "500 bp sequences"),
    "get_bkg": ("Count 1-, 2- and 3-mers", "500 bp sequences"),
    "sequence_complexity": ("Sequence complexity", "500 bp sequences"),
}


def seconds(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.3g} s" if x >= 0.01 else f"{x * 1000:.2g} ms"


def megabytes(x: float | None) -> str:
    if x is None:
        return ""
    return f"{max(x, 0) / 2**20:,.0f} MB"


def main() -> None:
    raw = pl.read_csv(sys.argv[1], separator="\t", infer_schema_length=None)
    keys = ["task", "size", "threads", "language"]
    base = raw.filter(pl.col("run") == "noop").group_by(keys).agg(pl.col("peak_rss").median().alias("base_rss"))
    runs = (
        raw.filter(pl.col("run").str.contains(r"^\d+$"))
        .with_columns(pl.col("seconds").cast(pl.Float64))
        .group_by(keys)
        .agg(pl.col("seconds").median(), pl.col("peak_rss").median(), pl.len().alias("runs"))
        .join(base, on=keys, how="left")
        .with_columns((pl.col("peak_rss") - pl.col("base_rss")).alias("memory"))
    )
    wide = runs.pivot(on="language", index=["task", "size", "threads"], values=["seconds", "memory"])

    print("| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    order = {task: i for i, task in enumerate(LABELS)}
    rows = sorted(wide.iter_rows(named=True), key=lambda r: (order.get(r["task"], 99), r["size"], r["threads"]))
    for r in rows:
        label, unit = LABELS.get(r["task"], (r["task"], ""))
        r_s, py_s = r.get("seconds_R"), r.get("seconds_Python")
        speedup = f"{r_s / py_s:,.0f}x" if r_s and py_s else ""
        print(
            f"| {label} | {r['size']:,} {unit} | {r['threads']} | {seconds(r_s)} | {seconds(py_s)} | "
            f"{speedup} | {megabytes(r.get('memory_R'))} | {megabytes(r.get('memory_Python'))} |"
        )


if __name__ == "__main__":
    main()
