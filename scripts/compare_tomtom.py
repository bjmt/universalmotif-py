"""Compare compare_motifs() with TOMTOM from the MEME suite.

Both score an alignment by the sum of the Pearson correlations of its
columns and take P-values from the distribution of that sum under a null of
database columns, so on the same motifs they should largely agree. This
script runs both on a set of query motifs against a motif database and
reports how often the best match agrees, how well the P-values of all pairs
rank alike, and whether the best alignments are the same. By default both
score the whole query, TOMTOM's default and compare_motifs(complete=True);
with --incomplete-scores both score only the overlap.

    uv run --no-sync python scripts/compare_tomtom.py DATABASE.meme QUERIES [...]

QUERIES are motif files in any supported format (by extension) or CIS-BP
archives (.zip). TOMTOM must be on the PATH or given with --tomtom.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import universalmotif as um
from scipy.stats import spearmanr


def read_any(path: Path) -> list[um.Motif]:
    if path.suffix == ".zip":
        return um.databases.read_cisbp_archive(path)
    readers = {".meme": um.read_meme, ".txt": um.read_meme, ".jaspar": um.read_jaspar,
               ".transfac": um.read_transfac, ".motif": um.read_homer}
    return readers.get(path.suffix, um.read_meme)(path)


def run_tomtom(tomtom: str, queries: Path, targets: Path, min_overlap: int,
               incomplete: bool) -> pl.DataFrame:
    extra = ["-incomplete-scores"] if incomplete else []
    out = subprocess.run(
        [tomtom, "-no-ssc", "-text", "-dist", "pearson", "-min-overlap", str(min_overlap),
         "-thresh", "1e9", "-evalue", "-verbosity", "1", *extra, str(queries), str(targets)],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = [line.split("\t") for line in out.splitlines() if line and not line.startswith("#")]
    df = pl.DataFrame(rows[1:], schema=rows[0], orient="row")
    return df.select(
        pl.col("Query_ID").alias("query"),
        pl.col("Target_ID").alias("target"),
        pl.col("Optimal_offset").cast(pl.Int64).alias("tt_offset"),
        pl.col("p-value").cast(pl.Float64).alias("tt_pvalue"),
        pl.col("Overlap").cast(pl.Int64).alias("tt_overlap"),
        pl.col("Orientation").alias("tt_strand"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("database", type=Path)
    ap.add_argument("queries", type=Path, nargs="+")
    ap.add_argument("--sample", type=int, default=60, help="queries sampled from each file")
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--tomtom", default=shutil.which("tomtom") or str(Path.home() / "meme/bin/tomtom"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--incomplete-scores", action="store_true",
                    help="score only the overlap in both, rather than the whole query")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    targets = [m.replace(name=f"T{i}") for i, m in enumerate(read_any(args.database))]
    queries = []
    for path in args.queries:
        ms = read_any(path)
        pick = rng.choice(len(ms), size=min(args.sample, len(ms)), replace=False)
        queries += [ms[i] for i in sorted(pick)]
    queries = [m.replace(name=f"Q{i}") for i, m in enumerate(queries)]
    print(f"{len(queries)} queries against {len(targets)} database motifs")

    with tempfile.TemporaryDirectory() as tmp:
        qfile, tfile = Path(tmp) / "q.meme", Path(tmp) / "t.meme"
        um.write_meme(queries, qfile)
        um.write_meme(targets, tfile)
        tt = run_tomtom(args.tomtom, qfile, tfile, args.min_overlap, args.incomplete_scores)

    nq = len(queries)
    ours = um.compare_motifs(queries + targets, compare_to=list(range(nq)), qvalue=1,
                             min_overlap=args.min_overlap, nthreads=2,
                             complete=not args.incomplete_scores)
    ours = ours.filter(pl.col("target_i") >= nq).select(
        pl.col("subject").alias("query"),
        pl.col("target"),
        pl.col("offset").alias("um_offset"),
        pl.col("pvalue").alias("um_pvalue"),
        pl.col("overlap").alias("um_overlap"),
        pl.col("strand").alias("um_strand"),
        pl.col("score").alias("um_score"),
    )
    both = ours.join(tt, on=["query", "target"], how="inner")
    print(f"{both.height} query and target pairs reported by both "
          f"(TOMTOM {tt.height}, compare_motifs {ours.height})")

    def best(df: pl.DataFrame, col: str) -> pl.DataFrame:
        return df.sort(col).group_by("query", maintain_order=True).first()

    top = best(ours, "um_pvalue").join(best(tt, "tt_pvalue"), on="query", suffix="_tt")
    same = top["target"] == top["target_tt"]
    print(f"same best match: {same.mean():.1%} of queries")
    clear = top["tt_pvalue_tt"] < 1e-5 if "tt_pvalue_tt" in top.columns else top["tt_pvalue"] < 1e-5
    print(f"  {same.filter(clear).mean():.1%} of the {int(clear.sum())} whose best TOMTOM match "
          f"has P < 1e-5")
    # Where the best matches differ, how much worse does each tool rate the
    # other's choice?
    gaps = []
    for row in top.filter(~same).iter_rows(named=True):
        q = row["query"]
        ours_alt = ours.filter((pl.col("query") == q) & (pl.col("target") == row["target_tt"]))
        tt_alt = tt.filter((pl.col("query") == q) & (pl.col("target") == row["target"]))
        if ours_alt.height and tt_alt.height:
            gaps.append((np.log10(ours_alt["um_pvalue"][0] / row["um_pvalue"]),
                         np.log10(tt_alt["tt_pvalue"][0] / row["tt_pvalue"])))
    if gaps:
        g = np.array(gaps)
        print(f"  where they differ, each rates the other's choice a median of "
              f"{np.median(g[:, 0]):.2f} (compare_motifs) and {np.median(g[:, 1]):.2f} (TOMTOM) "
              f"orders of magnitude less significant")
    top5 = []
    for q in range(nq):
        a = set(ours.filter(pl.col("query") == f"Q{q}").sort("um_pvalue")["target"][:5])
        b = set(tt.filter(pl.col("query") == f"Q{q}").sort("tt_pvalue")["target"][:5])
        top5.append(len(a & b) / 5)
    print(f"top five matches shared: {np.mean(top5):.1%} on average")

    rhos = []
    for q in both["query"].unique():
        sub = both.filter(pl.col("query") == q)
        if sub.height > 10:
            rhos.append(spearmanr(sub["um_pvalue"], sub["tt_pvalue"]).statistic)
    print(f"rank correlation of P-values within queries: median {np.median(rhos):.3f}, "
          f"lowest {np.min(rhos):.3f}")
    lp = np.log10(both["um_pvalue"].to_numpy()), np.log10(both["tt_pvalue"].to_numpy())
    print(f"log10 P-values over all pairs: correlation {np.corrcoef(*lp)[0, 1]:.3f}, "
          f"median ratio {np.median(10 ** (lp[0] - lp[1])):.2f}")

    strong = both.filter(pl.col("tt_pvalue") < 1e-4)
    strand = strong["um_strand"] == strong["tt_strand"]
    print(f"of {strong.height} pairs with TOMTOM P < 1e-4: same strand {strand.mean():.1%}, "
          f"same overlap {(strong['um_overlap'] == strong['tt_overlap']).mean():.1%}")


if __name__ == "__main__":
    main()
