"""Evidence for each motif from outside the models and TF-MoDISco, and the
tables summarizing it.

    python checks.py support  [--models ...]   # phyloP and ChIP-seq, for every motif of every run
    python checks.py evidence                  # with insertion.py's results: per motif, per run, tables

support checks every motif of every method on each model and head, and
ENCODE's own TF-MoDISco patterns as the method "published", against two
kinds of evidence: conservation at its hits (phyloP), and more ChIP-seq
peaks of the factor it is named after, in the same cell line, than
expected. Its hits are matches to its PPM (P < 1e-4) overlapping a positive
seqlet by 5 letters or more, in all of a model's windows. evidence adds the
model's response from insertion.py, and writes work/checks/evidence.tsv (per
motif), evidence_runs.tsv (per run) and evidence_tables.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import multiprocessing
import pathlib
import pickle
import sys

import numpy as np
import polars as pl

import config
import harness

HERE = pathlib.Path(__file__).resolve().parent

# The rows of the tables, in order, and their labels.
LABELS = {
    "cm": "`contribution_motifs()`",
    "cm_uncapped": "`contribution_motifs()`, `max_seqlets=None`",
    "cm_one_pass": "`contribution_motifs()`, one seqlet pass",
    "cm_observed": "`contribution_motifs()`, observed scores only",
    "modisco": "TF-MoDISco, same regions",
    "published": "TF-MoDISco, ENCODE's results",
}
# The columns of the tables: kind of model and head.
COLUMNS = {"bpnet_counts": "BPNet", "chrombpnet_counts": "chromBPNet", "procapnet_counts": "ProCapNet counts",
           "procapnet_profile": "ProCapNet profile"}


# ---------------------------------------------------------------------------
# Motifs

def _pool(fn, tasks, jobs: int) -> list:
    # Spawned rather than forked, since a fork after polars has started its threads can hang.
    with cf.ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        return [r for part in ex.map(fn, tasks) for r in part]


def _motifs(um, work: pathlib.Path, row: dict) -> dict[str, list]:
    """The PPMs of every run on one model and head, in the method's own
    order, and ENCODE's TF-MoDISco patterns as "published"."""
    out = {}
    for f in sorted((work / "results" / row["model"] / row["head"]).glob("*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        if r.get("ppms"):
            out[f.stem] = list(r["ppms"])
    pub = next((p for p in (work / "files" / row["model"] / "modisco").glob("*.h*5") if row["head"] in p.name), None)
    if pub is not None:
        out["published"] = harness._modisco_patterns(um, pub, "PPM")
    return out


def _published(um, work, row) -> list:
    """The CWMs of ENCODE's TF-MoDISco patterns of a model's head."""
    d = work / "files" / row["model"] / "modisco"
    f = next((p for p in d.glob("*.h*5") if row["head"] in p.name), None)
    return harness._modisco_patterns(um, f, "CWM") if f is not None else []


def _similarity(um, a: list, b: list) -> np.ndarray:
    """The CWM similarity of each motif of a (rows) to each of b (columns)."""
    out = np.zeros((len(a), len(b)))
    if not a or not b:
        return out
    allm = [m.replace(name=f"a{i}") for i, m in enumerate(a)] + [m.replace(name=f"b{j}") for j, m in enumerate(b)]
    cmp = um.compare_motifs(allm, compare_to=list(range(len(a))), qvalue=1, nthreads=config.THREADS)
    cmp = cmp.filter(pl.col("subject").str.starts_with("a") & pl.col("target").str.starts_with("b"))
    i = cmp["subject"].str.slice(1).cast(pl.Int64).to_numpy()
    j = cmp["target"].str.slice(1).cast(pl.Int64).to_numpy()
    np.maximum.at(out, (i, j), cmp["score"].to_numpy())
    return out


def _names(um, jaspar, ppms: list) -> list[list[str]]:
    """The factors each motif is named after: those of its three best JASPAR
    matches (q < 0.05) in order, heterodimers split into their parts."""
    out = [[] for _ in ppms]
    if not ppms:
        return out
    nm = um.name_motifs(ppms, jaspar, top=3, nthreads=config.THREADS)
    for i, alt in nm.sort("motif_i", "rank").select("motif_i", "altname").iter_rows():
        for f in (alt or "").upper().split("::"):
            if f and f not in out[i]:
                out[i].append(f)
    return out


def _seqlets(um, work: pathlib.Path, row: dict, strs: list[str], obs: np.ndarray) -> pl.DataFrame:
    """The positive seqlets of all of a model's windows, kept once called."""
    f = work / "checks" / "seqlets" / f"{row['model']}.{row['head']}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    sl = um.call_seqlets(obs, strs, seed=1, nthreads=config.THREADS).filter(pl.col("sign") == "+")
    sl = sl.select("sequence_i", "start", "end")
    sl.write_parquet(f)
    return sl


def _peaks(path: pathlib.Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merged peaks by chromosome, as sorted starts and ends."""
    rows = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            c, s, e = line.split("\t", 3)[:3]
            rows.setdefault(c, []).append((int(s), int(e)))
    out = {}
    for c, iv in rows.items():
        iv.sort()
        starts, ends = [iv[0][0]], [iv[0][1]]
        for s, e in iv[1:]:
            if s <= ends[-1]:
                ends[-1] = max(ends[-1], e)
            else:
                starts.append(s)
                ends.append(e)
        out[c] = (np.array(starts), np.array(ends))
    return out


def _inside(peaks, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.zeros(len(pos), dtype=bool)
    for c in np.unique(chrom):
        if c not in peaks:
            continue
        k = chrom == c
        starts, ends = peaks[c]
        j = np.searchsorted(starts, pos[k], side="right") - 1
        out[k] = (j >= 0) & (pos[k] < ends[np.maximum(j, 0)])
    return out


# ---------------------------------------------------------------------------
# Outside support: phyloP and ChIP-seq

PHYLOP = "phylop/hg38.phyloP100way.bw"
# Hits used for the conservation test, and placements of the same width for its background. phyloP
# depends on composition (CpGs mutate fast, so GC-rich sites look less conserved), so the background of a
# motif is drawn to match its sites' GC content and CpG count.
PHYLOP_HITS, PHYLOP_BACKGROUND = 2000, 20000
# Flanks of a hit for the conservation test: from 10 to 30 letters either side.
FLANK_NEAR, FLANK_FAR = 10, 30


def support(args):
    """Every motif of every run on each model and head, against phyloP and
    the ChIP-seq peaks of the factor it is named after."""
    work = pathlib.Path(args.work)
    rows = harness.manifest(args.models)
    rows = rows.filter(pl.Series([(work / "results" / r["model"] / r["head"]).is_dir()
                                  for r in rows.iter_rows(named=True)]))
    catalogue = {c: json.loads((work / "files" / "chip" / f"{c}.json").read_text())
                 for c in config.CHIP_CELLS if (work / "files" / "chip" / f"{c}.json").exists()}
    tasks = [(str(work), row, catalogue.get(row["cell"], {})) for row in rows.iter_rows(named=True)]
    df = pl.DataFrame(_pool(_support_one, tasks, args.jobs), infer_schema_length=None)
    (work / "checks").mkdir(parents=True, exist_ok=True)
    df.write_csv(work / "checks" / "support.tsv", separator="\t")
    print(f"{df.height} rows in {work / 'checks' / 'support.tsv'}", file=sys.stderr)


def _seqlet_hits(um, ppms: list, strs: list[str], seqlets: pl.DataFrame, pad: int) -> pl.DataFrame:
    """Matches of the motifs (P < 1e-4) overlapping a seqlet by 5 letters or
    more, one row per match, with motif_i the motif's place in ppms. Only
    the seqlets and pad letters either side are scanned, which holds every
    such match for motifs up to pad wide."""
    si = seqlets["sequence_i"].to_numpy()
    s0, e0 = seqlets["start"].to_numpy(), seqlets["end"].to_numpy()
    lo = np.maximum(s0 - pad, 0)
    hi = np.minimum(e0 + pad, np.array([len(strs[i]) for i in si]))
    pieces = [strs[i][a:b] for i, a, b in zip(si, lo, hi)]
    # Motifs whose best site cannot reach P < 1e-4, such as ones of four or six letters, hit
    # where they match as well as they can instead, as in contribution_motifs().
    best = _best_pvalues(ppms)
    parts = []
    usual = [k for k, b in enumerate(best) if b <= 1e-4]
    if usual:
        h = um.scan_sequences([ppms[k] for k in usual], pieces, nthreads=config.THREADS)
        parts.append(h.with_columns(motif_i=pl.Series(np.array(usual, dtype=np.uint32)[h["motif_i"].to_numpy()])))
    for k, b in enumerate(best):
        if b > 1e-4:
            h = um.scan_sequences(ppms[k], pieces, pvalue=b * (1 + 1e-6), nthreads=config.THREADS)
            parts.append(h.with_columns(motif_i=pl.lit(k, dtype=pl.UInt32)))
    hits = pl.concat(parts) if parts else pl.DataFrame()
    if not hits.height:
        return hits
    k = hits["sequence_i"].to_numpy()
    hits = hits.with_columns(sequence_i=pl.Series(si[k]), start=pl.Series(hits["start"].to_numpy() + lo[k]),
                             end=pl.Series(hits["end"].to_numpy() + lo[k]), s0=pl.Series(s0[k]), e0=pl.Series(e0[k]))
    hits = hits.filter(pl.min_horizontal("end", "e0") - pl.max_horizontal("start", "s0") >= 5)
    return hits.unique(["motif_i", "sequence_i", "start", "strand"]).drop("s0", "e0", "sequence")


def _best_pvalues(ppms: list) -> list[float]:
    """The P-value of each motif's best possible site (0 for gapped motifs,
    which always reach the cutoff)."""
    from universalmotif._analysis import motif_pvalue
    from universalmotif._utils import motif_score

    return [0.0 if m.isgapped else float(np.ravel(motif_pvalue(m, score=float(motif_score(m, 1.0))))[0])
            for m in ppms]


def _phylop(work: pathlib.Path, windows: pl.DataFrame, width: int) -> np.ndarray:
    """phyloP for every letter of every window; NaN where there is none."""
    import pyBigWig

    bw = pyBigWig.open(str(work / "files" / PHYLOP))
    sizes = bw.chroms()
    out = np.full((windows.height, width), np.nan, dtype=np.float32)
    for k, (c, s) in enumerate(windows.select("chrom", "start").iter_rows()):
        if c in sizes and s >= 0 and s + width <= sizes[c]:
            out[k] = bw.values(c, s, s + width, numpy=True)
    bw.close()
    return out


def _conservation(cons: np.ndarray, si: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Mean phyloP over each site less that of its flanks; NaN where a site
    or its flanks run off the window or have no phyloP."""
    width = cons.shape[1]
    out = np.full(len(si), np.nan)
    ok = (start - FLANK_FAR >= 0) & (end + FLANK_FAR <= width)
    for k in np.flatnonzero(ok):
        row, a, b = cons[si[k]], start[k], end[k]
        core = row[a:b]
        flank = np.concatenate([row[a - FLANK_FAR:a - FLANK_NEAR], row[b + FLANK_NEAR:b + FLANK_FAR]])
        if np.isnan(core).any() or np.isnan(flank).any():
            continue
        out[k] = core.mean() - flank.mean()
    return out


def _support_one(task) -> list[dict]:
    import universalmotif as um
    from scipy.stats import binomtest, mannwhitneyu

    work, row, files = pathlib.Path(task[0]), task[1], task[2]
    jaspar = um.read_meme(work / "files" / "jaspar.meme")
    strs, obs, _ = harness._load(work, row["model"], row["head"], False)
    width = len(strs[0])
    windows = pl.read_parquet(work / "data" / row["model"] / row["head"] / "windows.parquet")
    wchrom, wstart = windows["chrom"].to_numpy(), windows["start"].to_numpy()
    seqlets = _seqlets(um, work, row, strs, obs)
    sl_si = seqlets["sequence_i"].to_numpy()
    sl_s, sl_e = seqlets["start"].to_numpy(), seqlets["end"].to_numpy()
    sl_mid = (sl_s + sl_e) // 2
    cons = _phylop(work, windows, width)
    rng = np.random.default_rng(1)
    codes = np.load(work / "data" / row["model"] / row["head"] / "seqs.npy", mmap_mode="r")
    background = {}

    def strata(si, a, w):
        # Composition of sites of width w: GC content in sixths, and none, one, or more CpGs.
        pos = np.clip(a[:, None] + np.arange(w), 0, width - 1)
        c = np.asarray(codes[si[:, None], pos]) if len(si) else np.zeros((0, w), dtype=np.uint8)
        gc = ((c == 1) | (c == 2)).mean(1)
        cpg = ((c[:, :-1] == 1) & (c[:, 1:] == 2)).sum(1)
        return np.minimum((gc * 6).astype(int), 5) * 3 + np.minimum(cpg, 2)

    def bg(w):
        # Sites of width w placed at random over seqlets, overlapping them by 5 letters or more.
        if w not in background:
            j = rng.choice(len(sl_si), PHYLOP_BACKGROUND)
            a = rng.integers(sl_s[j] - w + 5, sl_e[j] - 5 + 1)
            d = _conservation(cons, sl_si[j], a, a + w)
            ok = ~np.isnan(d)
            background[w] = (d[ok], strata(sl_si[j][ok], a[ok], w))
        return background[w]

    def matched(d, st, w):
        # Background sites drawn stratum by stratum, as many as the motif has sites in each; sites in
        # strata the background lacks are left out.
        bd, bst = bg(w)
        keep, draw = [], []
        for k in np.unique(st):
            pool = np.flatnonzero(bst == k)
            mine = np.flatnonzero(st == k)
            if len(pool) < 10:
                continue
            keep.append(mine)
            draw.append(bd[rng.choice(pool, len(mine) * 3)])
        if not keep:
            return d[:0], d[:0]
        return d[np.concatenate(keep)], np.concatenate(draw)

    peaks_cache, base_cache, out = {}, {}, []
    for method, ppms in _motifs(um, work, row).items():
        names = _names(um, jaspar, ppms)
        pad = max(m.width + (sum(m.maxgap) if m.isgapped else 0) for m in ppms) + 5
        hits = _seqlet_hits(um, ppms, strs, seqlets, pad)
        for i, m in enumerate(ppms):
            h = hits.filter(pl.col("motif_i") == i) if hits.height else hits
            rec = {"model": row["model"], "kind": row["kind"], "cell": row["cell"], "head": row["head"],
                   "method": method, "motif": m.name, "rank": i + 1, "width": m.width,
                   "consensus": m.consensus or "", "names": ",".join(names[i]), "hits": h.height,
                   "chip_factor": None, "chip_hits": None, "chip_precision": None, "chip_baseline": None,
                   "chip_log2": None, "chip_pvalue": None, "phylop_sites": None, "phylop_diff": None,
                   "phylop_background": None, "phylop_pvalue": None}
            if not h.height:
                out.append(rec)
                continue
            si, st, en = h["sequence_i"].to_numpy(), h["start"].to_numpy(), h["end"].to_numpy()
            # ChIP-seq of the first factor the motif is named after that has any in this cell line.
            factor = next((f for f in names[i] if f in files), None)
            if factor is not None:
                if factor not in peaks_cache:
                    peaks_cache[factor] = _peaks(work / "files" / "chip" / row["cell"] / f"{factor}.bed.gz")
                    base_cache[factor] = float(_inside(peaks_cache[factor], wchrom[sl_si], wstart[sl_si] + sl_mid).mean())
                inside = _inside(peaks_cache[factor], wchrom[si], wstart[si] + (st + en) // 2)
                base = base_cache[factor]
                prec = float(inside.mean())
                rec.update(chip_factor=factor, chip_hits=len(inside), chip_precision=round(prec, 4),
                           chip_baseline=round(base, 4),
                           chip_log2=round(float(np.log2((prec + 1e-4) / (base + 1e-4))), 3),
                           chip_pvalue=float(binomtest(int(inside.sum()), len(inside), min(max(base, 1e-6), 1 - 1e-6),
                                                       alternative="greater").pvalue))
            take = rng.choice(len(si), PHYLOP_HITS, replace=False) if len(si) > PHYLOP_HITS else np.arange(len(si))
            d = _conservation(cons, si[take], st[take], en[take])
            ok = ~np.isnan(d)
            w = int(np.median(en - st))
            d, b = matched(d[ok], strata(si[take][ok], st[take][ok], w), w)
            if len(d) >= 10 and len(b) >= 10:
                rec.update(phylop_sites=len(d), phylop_diff=round(float(d.mean()), 4),
                           phylop_background=round(float(b.mean()), 4),
                           phylop_pvalue=float(mannwhitneyu(d, b, alternative="greater").pvalue))
            out.append(rec)
    print(f"  {row['model']} {row['head']}: {len(out)} motifs", file=sys.stderr, flush=True)
    return out


# What counts as evidence. Each test's P-values are adjusted (Benjamini-Hochberg) over every motif it
# tested, so a motif's verdict can shift a little with the other runs checked beside it.
SUPPORT_Q = 0.01
PHYLOP_MIN_DIFF = 0.1      # phyloP over the flanks, beyond that of random sites in seqlets
CHIP_MIN_LOG2 = np.log2(1.5)
CHIP_MIN_HITS = 20
USED_Q = 0.01              # insertion.py's test of the model's response
# CWM similarity at which two motifs of one run are one motif, and at which motifs of different runs
# are the same motif.
SAME_IN_RUN, SAME_ACROSS = 0.9, 0.8


def _bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted P-values, leaving out missing ones."""
    from scipy.stats import false_discovery_control

    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = false_discovery_control(p[ok])
    return out


def support_flags(df: pl.DataFrame) -> pl.DataFrame:
    """The support table with each test's verdict and the overall one."""
    chip = df.select(pl.when(pl.col("chip_hits") >= CHIP_MIN_HITS).then(pl.col("chip_pvalue")))[:, 0]
    df = df.with_columns(
        phylop_q=pl.Series(_bh(df["phylop_pvalue"].cast(pl.Float64).fill_null(np.nan))).fill_nan(None),
        chip_q=pl.Series(_bh(chip.cast(pl.Float64).fill_null(np.nan))).fill_nan(None))
    df = df.with_columns(
        phylop_ok=(pl.col("phylop_q") < SUPPORT_Q) & (pl.col("phylop_diff") - pl.col("phylop_background") >= PHYLOP_MIN_DIFF),
        chip_ok=(pl.col("chip_q") < SUPPORT_Q) & (pl.col("chip_log2") >= CHIP_MIN_LOG2),
        tested=pl.col("phylop_q").is_not_null() | pl.col("chip_q").is_not_null())
    df = df.with_columns(pl.col("phylop_ok", "chip_ok").fill_null(False))
    return df.with_columns(supported=pl.col("phylop_ok") | pl.col("chip_ok"),
                           named=pl.col("names").fill_null("") != "")


# ---------------------------------------------------------------------------
# Distinct motifs with evidence

def _summary_one(task) -> list[dict]:
    import universalmotif as um

    work, model, head, df = pathlib.Path(task[0]), task[1], task[2], task[3]
    row = {"model": model, "head": head}
    cwms = {}
    d = work / "results" / model / head
    for f in sorted(d.glob("*.pkl")):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cwms[f.stem] = {m.name: m for m in r.get("cwms", []) if m is not None}
    cwms["published"] = {m.name: m for m in _published(um, work, row)}
    sup = df.filter("supported")
    keys = [(r["method"], r["motif"]) for r in sup.iter_rows(named=True) if r["motif"] in cwms.get(r["method"], {})]
    sim = _similarity(um, [cwms[k[0]][k[1]] for k in keys], [cwms[k[0]][k[1]] for k in keys]) if keys else np.zeros((0, 0))
    # Distinct supported motifs across all runs: greedy clusters, the motif with most hits first.
    hits = {(r["method"], r["motif"]): r["hits"] for r in sup.iter_rows(named=True)}
    order = sorted(range(len(keys)), key=lambda i: -hits[keys[i]])
    cluster = np.full(len(keys), -1)
    n_clusters = 0
    for i in order:
        if cluster[i] >= 0:
            continue
        members = (sim[i] >= SAME_ACROSS) & (cluster < 0)
        members[i] = True
        cluster[members] = n_clusters
        n_clusters += 1
    out = []
    kind = df["kind"][0]
    for method in df["method"].unique().sort().to_list():
        m = df.filter(pl.col("method") == method)
        mine = [i for i, k in enumerate(keys) if k[0] == method]
        # Supported motifs of this run, merged when 0.9 alike.
        distinct = 0
        seen = np.zeros(len(mine), dtype=bool)
        for a in range(len(mine)):
            if seen[a]:
                continue
            distinct += 1
            seen |= np.array([sim[mine[a], mine[b]] >= SAME_IN_RUN for b in range(len(mine))])
        out.append({"model": model, "kind": kind, "head": head, "method": method, "motifs": m.height,
                    "tested": int(m["tested"].sum()), "supported": int(m["supported"].sum()),
                    "unsupported": int((m["tested"] & ~m["supported"]).sum()),
                    "untested": int((~m["tested"]).sum()),
                    "phylop_ok": int(m["phylop_ok"].sum()), "chip_ok": int(m["chip_ok"].sum()),
                    "supported_unnamed": int((m["supported"] & ~m["named"]).sum()),
                    "distinct_supported": distinct,
                    "found": len(set(cluster[mine].tolist())), "union": n_clusters})
    print(f"  {model} {head}: {n_clusters} distinct supported motifs", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------------------
# Evidence: outside support and the model's response together

def evidence(args):
    """Per motif and per run: outside support (support) and the model's
    response (insertion.py). A motif with neither is the closest these checks
    come to calling it noise. Also writes the three tables of the benchmarks
    page, as Markdown."""
    work = pathlib.Path(args.work)
    df = support_flags(pl.read_csv(work / "checks" / "support.tsv", separator="\t", infer_schema_length=None))
    key = ["model", "head", "method", "motif"]
    ins = pl.read_csv(work / "checks" / "insertion.tsv", separator="\t", infer_schema_length=None)
    # A run can name two motifs alike (TF-MoDISco's patterns of both signs, say); the first is kept.
    ins = ins.unique(key, keep="first", maintain_order=True)
    ins = ins.with_columns(insertion_q=pl.Series(_bh(ins["pvalue"].cast(pl.Float64).fill_null(np.nan))).fill_nan(None))
    ins = ins.select(*key, insertion_effect="effect", used=(pl.col("insertion_q") < USED_Q) & (pl.col("effect") > 0))
    df = df.join(ins, on=key, how="left")
    df.write_csv(work / "checks" / "evidence.tsv", separator="\t")
    runs = (df.with_columns(fam=pl.col("kind") + "_" + pl.col("head"))
            .group_by("fam", "model", "head", "method")
            .agg(pl.len().alias("motifs"), pl.col("used").is_null().sum().alias("no_insertion"),
                 (pl.col("used") & pl.col("supported")).sum().alias("used_supported"),
                 (pl.col("used") & ~pl.col("supported")).sum().alias("used_only"),
                 (~pl.col("used") & pl.col("supported")).sum().alias("supported_only"),
                 (~pl.col("used") & ~pl.col("supported")).sum().alias("neither")))
    # Distinct motifs with any evidence (used, or supported), merged within each run at SAME_IN_RUN,
    # and across the runs of each model and head at SAME_ACROSS.
    either = df.with_columns(supported=pl.col("supported") | pl.col("used").fill_null(False))
    heads = either.select("model", "head").unique().sort("model", "head").rows()
    tasks = [(str(work), m, h, either.filter((pl.col("model") == m) & (pl.col("head") == h))) for m, h in heads]
    distinct = pl.DataFrame(_pool(_summary_one, tasks, args.jobs), infer_schema_length=None).select(
        "model", "head", "method", distinct_evidenced="distinct_supported", found="found", union="union")
    runs = runs.join(distinct, on=["model", "head", "method"], how="left").with_columns(
        share_of_union=pl.col("found") / pl.col("union")).sort("fam", "model", "method")
    runs.write_csv(work / "checks" / "evidence_runs.tsv", separator="\t")
    tables = _tables(runs)
    (work / "checks" / "evidence_tables.md").write_text(tables)
    print(tables)


def _tables(runs: pl.DataFrame) -> str:
    """The means per model of each kind of model and head, as three Markdown
    tables: distinct motifs with evidence and with none, the share of all
    runs' distinct motifs with evidence each found, and the motifs with
    outside support and with the model's response only."""
    agg = runs.group_by("fam", "method").agg(
        pl.col("distinct_evidenced").mean(), pl.col("neither").mean(), pl.col("share_of_union").mean(),
        (pl.col("used_supported") + pl.col("supported_only")).mean().alias("outside"),
        pl.col("used_only").mean().alias("model_only"))
    cell = {(r["fam"], r["method"]): r for r in agg.iter_rows(named=True)}
    fams = [f for f in COLUMNS if any(k[0] == f for k in cell)]
    methods = [m for m in LABELS if any(k[1] == m for k in cell)]
    methods += sorted({k[1] for k in cell} - set(methods))
    out = []
    for title, fmt in [
        ("Distinct motifs with evidence / motifs with none, per model",
         lambda r: f"{r['distinct_evidenced']:.1f} / {r['neither']:.1f}"),
        ("Share of all runs' distinct motifs with evidence, merged at a cosine of 0.8",
         lambda r: f"{r['share_of_union']:.2f}"),
        ("Motifs with outside support / with the model's response only, per model",
         lambda r: f"{r['outside']:.1f} / {r['model_only']:.1f}"),
    ]:
        out.append(f"{title}:\n")
        out.append("| Run | " + " | ".join(COLUMNS[f] for f in fams) + " |")
        out.append("| --- |" + " ---: |" * len(fams))
        for m in methods:
            vals = [fmt(cell[(f, m)]) if (f, m) in cell else "" for f in fams]
            out.append(f"| {LABELS.get(m, m)} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["support", "evidence"])
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--models", nargs="*", help="regular expressions matched against model names")
    ap.add_argument("--jobs", type=int, default=8, help="models and heads checked side by side")
    args = ap.parse_args()
    {"support": support, "evidence": evidence}[args.check](args)


if __name__ == "__main__":
    main()
