#!/usr/bin/env python
"""The insertion test pooled over two seeds, with a smoothed bar, and how
steady its verdicts are.

insertion.py tests each motif on 128 backgrounds and sets its bar from 200
random motifs, all drawn from --seed. With a second work directory made the
same way but with prep --seed 2 (sharing data, files, models and results),

    python pooled.py --works work work-seed2

pools the two. Each motif's effect and Wilcoxon test take all 256 of its
backgrounds, and its bar is the 99th percentile of the effects of the random
motifs of both seeds (about 400 per head), by a quantile regression of them
on width, in place of checks.py's window of widths; on a counts head, a motif
that lowers the counts is held to the 1st percentile instead. A second
verdict regresses on total information content. The backgrounds of each
seed and the random motifs are then bootstrapped BOOTSTRAPS times to see how
often each verdict changes, and how far each cell of the tables moves. For
comparison, the same bar is set on each seed alone, and the verdicts of the
two seeds compared. Everything goes in <first work>/pooled.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import checks  # noqa: E402
import insertion  # noqa: E402

BOOTSTRAPS = 200
UNSTEADY = 0.05            # a verdict that changes in more than this share of bootstraps
KEY = ["model", "head", "method", "motif"]


def _load(works: list[pathlib.Path]) -> list[dict]:
    """Per head, its motifs (as in the prep files) and each background's
    effect of every motif and random motif, from each work directory."""
    heads = []
    for f in sorted((works[0] / "insertion").glob("*.npz")):
        model, head = f.name[:-4].rsplit(".", 1)
        h = {"model": model, "head": head, "real": [], "random": []}
        for w in works:
            for part, d in [("real", w / "insertion"), ("random", w / "insertion" / "random")]:
                z = np.load(d / f.name)
                span, ic = insertion._geometry(z)
                h[part].append({"method": z["method"], "motif": z["name"], "width": span.astype(float), "ic": ic,
                                "d": np.load(d / f"{model}.{head}.diffs.npy").astype(np.float64)})
        a, b = h["real"]
        if not (np.array_equal(a["method"], b["method"]) and np.array_equal(a["motif"], b["motif"])):
            raise ValueError(f"{model} {head}: the two seeds' prep files hold different motifs")
        heads.append(h)
    return heads


def _wilcoxon(d: np.ndarray) -> np.ndarray:
    """The Wilcoxon signed-rank P-value of each row, as insertion.py."""
    from scipy.stats import wilcoxon

    p = np.ones(len(d))
    ok = (d != 0).any(1)
    if ok.any():
        p[ok] = wilcoxon(d[ok], axis=1).pvalue
    return p


def _quantile_line(x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Intercept and slope of the tau quantile of y on x (linear programming)."""
    from scipy.optimize import linprog

    n = len(x)
    X = np.column_stack([np.ones(n), x])
    c = np.concatenate([[0, 0], np.full(n, tau), np.full(n, 1 - tau)])
    r = linprog(c, A_eq=np.hstack([X, np.eye(n), -np.eye(n)]), b_eq=y,
                bounds=[(None, None)] * 2 + [(0, None)] * (2 * n), method="highs")
    return r.x[:2]


def _bars(x: np.ndarray, rx: np.ndarray, re: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The upper (USED_QUANTILE) and lower bars at each x, from the random
    motifs' rx and effects re, x clipped to the random motifs' range."""
    x = np.clip(x, rx.min(), rx.max())
    up, lo = _quantile_line(rx, re, checks.USED_QUANTILE), _quantile_line(rx, re, 1 - checks.USED_QUANTILE)
    return up[0] + up[1] * x, lo[0] + lo[1] * x


def _test(heads: list[dict], pick, by: str = "width") -> pl.DataFrame:
    """Every motif's effect, P-value and bars. pick(h) gives a head's real
    backgrounds' effects (motifs x backgrounds) and its random motifs' rx
    and effects."""
    parts = []
    for h in heads:
        d, rx, re = pick(h, by)
        x = h["real"][0][by]
        up, lo = _bars(x, rx, re)
        parts.append(pl.DataFrame({"model": h["model"], "head": h["head"], "method": h["real"][0]["method"],
                                   "motif": h["real"][0]["motif"], "effect": d.mean(1), "pvalue": _wilcoxon(d),
                                   "bar_up": up, "bar_low": lo}))
    df = pl.concat(parts).unique(KEY, keep="first", maintain_order=True)
    df = df.with_columns(q=pl.Series(checks._bh(df["pvalue"].to_numpy())))
    low = (pl.col("head") == "counts") & (pl.col("effect") < pl.col("bar_low"))
    return df.with_columns(used=(pl.col("q") < checks.USED_Q) & ((pl.col("effect") > pl.col("bar_up")) | low))


def _pooled(rng=None):
    """pick() for the pooled test, on the backgrounds and random motifs drawn
    by rng (all of them when None)."""
    def pick(h, by):
        idx = [np.arange(r["d"].shape[1]) if rng is None else rng.integers(0, r["d"].shape[1], r["d"].shape[1])
               for r in h["real"]]
        d = np.concatenate([r["d"][:, i] for r, i in zip(h["real"], idx)], 1)
        rx = np.concatenate([r[by] for r in h["random"]])
        re = np.concatenate([r["d"][:, i].mean(1) for r, i in zip(h["random"], idx)])
        if rng is not None:
            k = rng.integers(0, len(rx), len(rx))
            rx, re = rx[k], re[k]
        return d, rx, re
    return pick


def _seed(s: int):
    """pick() for seed s alone."""
    return lambda h, by: (h["real"][s]["d"], h["random"][s][by], h["random"][s]["d"].mean(1))


def _interval(v: list[float]) -> tuple[float, float]:
    return float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--works", nargs=2, required=True, help="the work directories of the two seeds")
    ap.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    works = [pathlib.Path(w) for w in args.works]
    out = works[0] / "pooled"
    out.mkdir(exist_ok=True)
    support = checks.support_flags(pl.read_csv(works[0] / "checks" / "support.tsv", separator="\t",
                                               infer_schema_length=None))
    heads = _load(works)
    kind = support.select(*KEY, "kind", "supported")

    def runs(test):
        df = support.join(test.select(*KEY, "used"), on=KEY, how="left")
        return checks._runs(works[0], df, "used", args.jobs)

    # The pooled test.
    point = _test(heads, _pooled())
    point_ic = _test(heads, _pooled(), "ic")
    runs_w, runs_ic = runs(point), runs(point_ic)
    runs_w.write_csv(out / "evidence_runs.tsv", separator="\t")
    runs_ic.write_csv(out / "evidence_runs_ic.tsv", separator="\t")
    cells = checks._cells(runs_w)

    # Each seed alone, under the same bar.
    lines = []
    seeds = [_test(heads, _seed(s)) for s in range(2)]
    j = seeds[0].select(*KEY, a="used").join(seeds[1].select(*KEY, b="used"), on=KEY).join(kind, on=KEY)
    flips = j.with_columns(flip=pl.col("a") != pl.col("b"))
    lines.append(f"Each seed alone, smoothed bar: {flips['flip'].mean():.1%} of {flips.height} verdicts differ "
                 f"between the seeds (" + ", ".join(f"{k} {g['flip'].mean():.1%}" for (k,), g in
                                                    flips.group_by("kind", maintain_order=True)) + ")")

    # Bootstraps.
    boot_used, boot_cells = [], []
    for b in range(args.bootstraps):
        t = _test(heads, _pooled(np.random.default_rng(1000 + b)))
        boot_used.append(point.select(KEY).join(t.select(*KEY, "used"), on=KEY, how="left")["used"].to_numpy())
        for (fam, method), r in checks._cells(runs(t)).items():
            boot_cells.append({"bootstrap": b, "fam": fam, "method": method,
                               **{k: r[k] for k in ["distinct_evidenced", "neither", "share_of_union", "outside",
                                                    "model_only"]}})
        print(f"  bootstrap {b + 1}/{args.bootstraps}", file=sys.stderr, flush=True)
    boot = pl.DataFrame(boot_cells)
    boot.write_csv(out / "bootstrap_cells.tsv", separator="\t")
    boot_used = np.array(boot_used, dtype=float)
    changed = (boot_used != point["used"].to_numpy()[None]).mean(0)
    per_motif = point.with_columns(passes=pl.Series(boot_used.mean(0)), changed=pl.Series(changed)).join(
        point_ic.select(*KEY, bar_up_ic="bar_up", bar_low_ic="bar_low", used_ic="used"), on=KEY)
    per_motif = support.join(per_motif, on=KEY, how="left")
    per_motif.write_csv(out / "evidence.tsv", separator="\t")
    m = per_motif.filter(pl.col("used").is_not_null()).with_columns(unsteady=pl.col("changed") > UNSTEADY)
    lines.append(f"Pooled, smoothed bar: {m['unsteady'].mean():.1%} of {m.height} verdicts change in more than "
                 f"{UNSTEADY:.0%} of {args.bootstraps} bootstraps (" +
                 ", ".join(f"{k} {g['unsteady'].mean():.1%}" for (k,), g in m.group_by("kind", maintain_order=True)) +
                 f"); {m.filter(~pl.col('supported'))['unsteady'].mean():.1%} of those without outside support")
    lines.append(f"Motifs with the model's response: {int(m['used'].sum())}, of which "
                 f"{int((m['used'] & (m['effect'] < 0)).sum())} lower the counts")
    (out / "stability.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    (out / "evidence_tables.md").write_text(_tables(cells, checks._cells(runs_ic), boot) + "\n")
    print((out / "evidence_tables.md").read_text())


def _tables(cells: dict, cells_ic: dict, boot: pl.DataFrame) -> str:
    """The three tables, each cell with its mean over the bootstraps (the
    expected count) and their 5th to 95th percentiles, how far those sit from
    the cells, the differences of each row from contribution_motifs()'s
    defaults over the same bootstraps, and the first table under the
    IC-matched rule."""
    fams = [f for f in checks.COLUMNS if any(k[0] == f for k in cells)]
    methods = [m for m in checks.LABELS if any(k[1] == m for k in cells)]
    methods += sorted({k[1] for k in cells} - set(methods))
    by = {k: g for k, g in boot.group_by("fam", "method")}

    def value(f, m, p, digits):
        r = cells.get((f, m))
        if r is None or r[p] is None:
            return ""
        v = by[(f, m)][p].drop_nulls().to_list()
        if not v:
            return f"{r[p]:.{digits}f}"
        lo, hi = _interval(v)
        return (f"{r[p]:.{digits}f} ({np.mean(v):.{digits}f}) [{lo:.{digits}f}, {hi:.{digits}f}]" if hi > lo
                else f"{r[p]:.{digits}f}")

    def table(title, parts, digits, source=None):
        out = [f"{title}:\n", "| Run | " + " | ".join(checks.COLUMNS[f] for f in fams) + " |",
               "| --- |" + " ---: |" * len(fams)]
        for m in methods:
            if source is None:
                vals = [" / ".join(value(f, m, p, digits) for p in parts) if (f, m) in cells else "" for f in fams]
            else:
                vals = [" / ".join(f"{source[(f, m)][p]:.{digits}f}" for p in parts)
                        if (f, m) in source and all(source[(f, m)][p] is not None for p in parts) else "" for f in fams]
            out.append(f"| {checks.LABELS.get(m, m)} | " + " | ".join(vals) + " |")
        return out + [""]

    def differences(p, digits):
        out = [f"Each run's {p} less that of `contribution_motifs()`, with its bootstrap interval "
               "(* where the interval holds 0):\n",
               "| Run | " + " | ".join(checks.COLUMNS[f] for f in fams) + " |", "| --- |" + " ---: |" * len(fams)]
        for m in methods:
            if m == methods[0]:
                continue
            vals = []
            for f in fams:
                a, b = cells.get((f, methods[0])), cells.get((f, m))
                if a is None or b is None or a[p] is None or b[p] is None:
                    vals.append("")
                    continue
                d = (by[(f, m)].sort("bootstrap").select("bootstrap", p).join(
                    by[(f, methods[0])].sort("bootstrap").select("bootstrap", base=p), on="bootstrap")
                     .drop_nulls().select(pl.col(p) - pl.col("base"))[:, 0].to_list())
                lo, hi = _interval(d)
                vals.append(f"{b[p] - a[p]:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]" +
                            ("*" if lo <= 0 <= hi else ""))
            out.append(f"| {checks.LABELS.get(m, m)} | " + " | ".join(vals) + " |")
        return out + [""]

    def shift():
        rows = []
        for (f, m), r in cells.items():
            for p in ["distinct_evidenced", "neither", "model_only", "share_of_union"]:
                v = by[(f, m)][p].drop_nulls().to_numpy() if (f, m) in by else np.array([])
                if r[p] is None or not len(v):
                    continue
                lo, hi = _interval(v)
                rows.append((p, float(np.mean(v)) - r[p], not lo <= r[p] <= hi))
        out = ["Each cell is given with its mean over the bootstraps (in parentheses) and their 5th to 95th "
               "percentiles. Mean less cell, median and largest:"]
        for p in ["distinct_evidenced", "neither", "model_only", "share_of_union"]:
            d = [x for q, x, _ in rows if q == p]
            out.append(f"{p} {np.median(d):+.2f}, {max(d, key=abs):+.2f}, with "
                       f"{sum(o for q, _, o in rows if q == p)} of {len(d)} cells outside their interval")
        return [";\n".join(out[1:]).join([out[0] + "\n\n", ".\n"])]

    out = shift()
    out += table("Distinct motifs with evidence / motifs with none, per model", ["distinct_evidenced", "neither"], 1)
    out += differences("distinct_evidenced", 1)
    out += table("Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8", ["share_of_union"], 2)
    out += differences("share_of_union", 2)
    out += table("Motifs with outside support / with the model's response only, per model",
                 ["outside", "model_only"], 1)
    out += table("Distinct motifs with evidence / motifs with none, per model, under the IC-matched rule "
                 "(without intervals)", ["distinct_evidenced", "neither"], 1, cells_ic)
    return "\n".join(out)


if __name__ == "__main__":
    main()
