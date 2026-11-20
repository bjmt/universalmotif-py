"""Run the benchmark in stages, each safe to rerun.

    python harness.py fetch [--models ...]
    python harness.py prep  [--models ...]
    python harness.py run   [--models ...] [--methods ...] [--jobs N]

Everything goes under --work (default ./work): downloads in work/files, the
trained models in work/models, prepared windows in work/data/<model>/<head>,
results in work/results/<model>/<head>/<method>.pkl, and one line per run in
work/runs.jsonl with its time and peak memory. The models come from
manifest.tsv and models.tsv, and the methods from config.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import pathlib
import pickle
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np
import polars as pl

import config

HERE = pathlib.Path(__file__).resolve().parent
LETTERS = np.array(list("ACGTN"))
ENCODE = "https://www.encodeproject.org"


# ---------------------------------------------------------------------------
# Downloads

def _request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if start is not None and r.status != 206:
                    raise RuntimeError(f"{url} does not serve byte ranges")
                return r.read() if start is None else r.read(end - start)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """A whole file, kept once downloaded."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 24)
    part.rename(dest)
    return dest


def _ranged(url: str, offset: int, size: int, dest: pathlib.Path, piece: int = 1 << 28, workers: int = 4):
    """Bytes offset..offset+size of url into dest, in pieces side by side."""
    part = dest.with_name(dest.name + ".part")
    with open(part, "wb") as fh:
        fh.truncate(size)

    def get(k):
        a = k * piece
        b = min(a + piece, size)
        data = _request(url, offset + a, offset + b)
        with open(part, "r+b") as fh:
            fh.seek(a)
            fh.write(data)

    n = (size + piece - 1) // piece
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(get, range(n)), 1):
            print(f"  {dest.name}: {k}/{n} pieces", file=sys.stderr, flush=True)
    part.rename(dest)


def _octal(field: bytes) -> int:
    if field[0] & 0x80:  # base-256, for members of 8 GB or more
        return int.from_bytes(field[1:], "big")
    text = field.rstrip(b"\0 ").decode() or "0"
    return int(text, 8)


def tar_extract(url: str, outdir: pathlib.Path, keep) -> list[pathlib.Path]:
    """The members of a tar archive at url for which keep(name) is true,
    reading its headers with range requests so that other members are not
    downloaded. Gzipped archives are streamed whole instead."""
    outdir.mkdir(parents=True, exist_ok=True)
    head = _request(url, 0, 512)
    got = []
    if head[:2] == b"\x1f\x8b":
        # A gzipped archive cannot be read out of order, so it is streamed through once.
        with urllib.request.urlopen(url, timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
            for m in tf:
                print(f"  {pathlib.Path(m.name).name}: {m.size / 1e9:.1f} GB "
                      f"{'kept' if m.isfile() and keep(m.name) else 'passed over'}", file=sys.stderr, flush=True)
                if m.isfile() and keep(m.name):
                    dest = outdir / pathlib.Path(m.name).name
                    if not dest.exists():
                        part = dest.with_name(dest.name + ".part")
                        with tf.extractfile(m) as src, open(part, "wb") as fh:
                            shutil.copyfileobj(src, fh, 1 << 24)
                        part.rename(dest)
                    got.append(dest)
        return got
    offset, long_name, pax = 0, None, {}
    while True:
        hdr = _request(url, offset, offset + 512)
        if len(hdr) < 512 or hdr == b"\0" * 512:
            break
        name = hdr[0:100].rstrip(b"\0").decode()
        prefix = hdr[345:500].rstrip(b"\0").decode() if hdr[257:262] == b"ustar" else ""
        name = f"{prefix}/{name}" if prefix else name
        size = _octal(hdr[124:136])
        kind = hdr[156:157]
        data = offset + 512
        if kind in (b"L", b"x", b"g"):
            body = _request(url, data, data + size)
            if kind == b"L":
                long_name = body.rstrip(b"\0").decode()
            elif kind == b"x":
                for rec in re.findall(rb"\d+ ([^=]+)=([^\n]*)\n", body):
                    pax[rec[0].decode()] = rec[1].decode()
        else:
            name = pax.get("path", long_name or name)
            size = int(pax.get("size", size))
            if kind in (b"0", b"\0") and keep(name):
                dest = outdir / pathlib.Path(name).name
                if not dest.exists():
                    _ranged(url, data, size, dest)
                got.append(dest)
            long_name, pax = None, {}
        offset = data + (size + 511) // 512 * 512
    return got


# ---------------------------------------------------------------------------
# Models

def manifest(models: list[str] | None = None) -> pl.DataFrame:
    """One row per model and head of config.MODELS, or of those also matching
    one of models, from manifest.tsv."""
    df = pl.read_csv(HERE / "manifest.tsv", separator="\t", infer_schema_length=0)
    df = df.with_columns(pl.col("window").cast(pl.Int64))
    df = df.filter(pl.col("model").str.contains("|".join(config.MODELS)))
    if models:
        df = df.filter(pl.col("model").str.contains("|".join(models)))
    return df


def fetch(args):
    """Every download: the genome, JASPAR, phyloP, and for each model its
    regions, scores, TF-MoDISco results and trained folds, then the ChIP-seq
    peaks of each cell line in config.CHIP_CELLS."""
    work = pathlib.Path(args.work)
    files = work / "files"
    download(config.GENOME_2BIT, files / "hg38.2bit")
    download(config.JASPAR, files / "jaspar.meme")
    download(config.PHYLOP, files / "phylop" / "hg38.phyloP100way.bw")
    rows = manifest(args.models)
    for row in rows.iter_rows(named=True):
        d = files / row["model"]
        print(f"{row['model']} {row['head']}", file=sys.stderr, flush=True)
        download(row["regions_url"], d / "regions.bed.gz")
        # The TF-MoDISco results of both heads.
        tar_extract(row["motifs_tar_url"], d / "modisco", lambda n: n.endswith((".h5", ".hd5")))
        if row["kind"] == "procapnet":
            # ProCapNet's archives hold numpy arrays of the whole inputs, hypothetical scores included.
            tar_extract(row["scores_tar_url"], d / row["head"], lambda n: n.endswith((".npy", ".bed.gz")))
            continue
        download(row["bigwig_url"], d / f"{row['head']}.bigWig")
        if row["head"] in config.HYPOTHETICAL_HEADS and not any((d / row["head"]).glob("*fold_mean*.h5")):
            # Only the mean over folds is kept, about 8 GB a head for chromBPNet, but the archive is
            # gzipped and has to be streamed whole: about 60 GB for chromBPNet and 3 GB for BPNet.
            tar_extract(row["scores_tar_url"], d / row["head"], lambda n: "fold_mean" in n and n.endswith(".h5"))
    _fetch_models(work, rows["model"].unique().to_list())
    _fetch_chip(work, [c for c in config.CHIP_CELLS if c in rows["cell"].to_list()])


def _fetch_models(work: pathlib.Path, names: list[str]):
    """Each model's trained folds, for insertion.py, in work/models/<model>.
    BPNet's folds are SavedModels inside a tar of their own, as their .h5
    files cannot be loaded by Python 3.11."""
    table = pl.read_csv(HERE / "models.tsv", separator="\t")
    for row in table.filter(pl.col("model").is_in(names)).iter_rows(named=True):
        d = work / "models" / row["model"]
        if (d / ".done").exists():
            continue
        archive = download(row["url"], d / f"{row['accession']}.tar.gz")
        with tarfile.open(archive) as tf:
            tf.extractall(d, filter="data")
        if row["model"].startswith("bpnet_"):
            for t in d.glob("fold_*/model.fold_*.tar"):
                with tarfile.open(t) as tf:
                    tf.extractall(t.parent / "saved", filter="data")
        (d / ".done").touch()
        print(f"  {row['model']}: model folds", file=sys.stderr, flush=True)


def _fetch_chip(work: pathlib.Path, cells: list[str]):
    """One conservative IDR peak file (GRCh38) per ChIP-seq target in each
    cell line, the one with the most replicates and then the largest, with a
    catalogue of them in work/files/chip/<cell>.json."""
    d = work / "files" / "chip"
    d.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        cat = d / f"{cell}.json"
        if not cat.exists():
            url = (f"{ENCODE}/search/?type=File&output_type=conservative+IDR+thresholded+peaks&assembly=GRCh38"
                   f"&file_format=bed&status=released&assay_title=TF+ChIP-seq&biosample_ontology.term_name={cell}"
                   "&field=href&field=target.label&field=file_size&field=biological_replicates&limit=all&format=json")
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                found = json.load(r)["@graph"]
            best = {}
            for f in found:
                label = ((f.get("target") or {}).get("label") or "").upper()
                key = (len(f.get("biological_replicates") or []), f.get("file_size", 0))
                if label and (label not in best or key > best[label][0]):
                    best[label] = (key, ENCODE + f["href"])
            cat.write_text(json.dumps({k: v[1] for k, v in best.items()}, indent=1))
        peaks = json.loads(cat.read_text())
        with cf.ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda kv, cell=cell: download(kv[1], d / cell / f"{kv[0]}.bed.gz"), peaks.items()))
        print(f"  {cell}: ChIP-seq of {len(peaks)} factors", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Windows

def _codes(seqs: list[str]) -> np.ndarray:
    lut = np.full(256, 4, dtype=np.uint8)
    for k, b in enumerate("ACGT"):
        lut[ord(b)] = lut[ord(b.lower())] = k
    return np.stack([lut[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])


def prep(args):
    """Each model's windows, as letter codes (seqs.npy), observed scores
    (obs.npy) and hypothetical scores (hyp.npy) where there are some."""
    import universalmotif as um

    work = pathlib.Path(args.work)
    for row in manifest(args.models).iter_rows(named=True):
        f = work / "files" / row["model"]
        out = work / "data" / row["model"] / row["head"]
        if (out / "obs.npy").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        w = config.WINDOW[row["kind"]]
        t0 = time.time()
        if row["kind"] == "procapnet":
            hyp = np.load(next((f / row["head"]).glob("*hyp.contrib.scores.npy")), mmap_mode="r")
            onehot = np.load(next((f / row["head"]).glob("*input.seqs.npy")), mmap_mode="r")
            # ProCapNet keeps the letters last.
            hyp = np.ascontiguousarray(np.asarray(hyp, dtype=np.float32).transpose(0, 2, 1))
            onehot = np.asarray(onehot).transpose(0, 2, 1)
            mid, half = hyp.shape[2] // 2, w // 2
            hyp, onehot = hyp[:, :, mid - half:mid + half], onehot[:, :, mid - half:mid + half]
            seqs = np.where(onehot.sum(1) > 0, onehot.argmax(1), 4).astype(np.uint8)
            obs = (hyp * onehot).sum(1).astype(np.float32)
            np.save(out / "hyp.npy", hyp.astype(np.float16))
            # The peaks file is in the same order as the scores, each window centred on its peak.
            with gzip.open(next((f / row["head"]).glob("*.peaks.all.bed.gz")), "rt") as fh:
                peaks = [line.split("\t")[:3] for line in fh]
            mid = np.array([(int(p[1]) + int(p[2])) // 2 for p in peaks]) - w // 2
            windows = pl.DataFrame({"window_i": np.arange(len(seqs)), "chrom": [p[0] for p in peaks],
                                    "start": mid, "end": mid + w})
        else:
            # chromBPNet archives can hold fold means of all regions and of TF-MoDISco's input only;
            # the first covers every window.
            h5s = sorted((f / row["head"]).glob("*fold_mean*.h5"), key=lambda p: "all_regions" not in p.name)
            h5 = h5s[0] if h5s else None
            regions = um.read_bed(f / "regions.bed.gz")
            anchor = "summit" if "peak" in regions.columns else "centre"
            windows = um.drop_overlaps(um.resize_ranges(regions.sort("chrom", "start"), w, anchor=anchor))
            windows = windows.filter(pl.col("start") >= 0).sort("chrom", "start")
            genome = um.Genome(work / "files" / "hg38.2bit")
            seqs = _codes(list(um.extract_ranges(genome, windows.select("chrom", "start", "end")).values()))
            import pybigtools

            bw = pybigtools.open(str(f / f"{row['head']}.bigWig"))
            obs = np.zeros((windows.height, w), dtype=np.float32)
            for k, (c, s, e) in enumerate(windows.select("chrom", "start", "end").iter_rows()):
                obs[k] = np.nan_to_num(bw.values(c, s, e))
            if h5 is not None:
                np.save(out / "hyp.npy", _hyp_for(um, h5, windows, w, genome))
        np.save(out / "seqs.npy", seqs)
        np.save(out / "obs.npy", obs)
        windows.write_parquet(out / "windows.parquet")
        print(f"{row['model']} {row['head']}: {len(seqs)} windows in {time.time() - t0:.0f} s", file=sys.stderr)


def _hyp_for(um, h5, windows: pl.DataFrame, w: int, genome, chunk: int = 5000) -> np.ndarray:
    """Hypothetical scores of the windows, from an ENCODE fold-mean file of
    whole model inputs, read a part at a time. Each window takes the scores of
    the first input that holds it."""
    out = np.zeros((windows.height, 4, w), dtype=np.float16)
    done = np.zeros(windows.height, dtype=bool)
    wc = windows["chrom"].to_numpy()
    ws = windows["start"].to_numpy()
    by_chrom = {c: np.flatnonzero(wc == c) for c in np.unique(wc)}
    for c, s, scores in _inputs(um, h5, genome, chunk):
        rows = by_chrom.get(c)
        if rows is None:
            continue
        starts = ws[rows]
        a, b = np.searchsorted(starts, s), np.searchsorted(starts, s + scores.shape[1] - w, side="right")
        for k in rows[a:b]:
            if not done[k]:
                out[k] = scores[:, ws[k] - s:ws[k] - s + w]
                done[k] = True
    if not done.all():
        print(f"  {int((~done).sum())} windows have no hypothetical scores", file=sys.stderr)
    return out


def _inputs(um, h5, genome, chunk: int):
    """The chromosome, start and hypothetical scores (letters by positions) of
    each input in an ENCODE fold-mean file, read a part at a time. The
    chromBPNet files of all regions hold no sequences, so theirs come from the
    genome."""
    import h5py

    with h5py.File(h5) as fh:
        n = next(fh[k].shape[0] for k in ("hyp_scores", "attributions/shap", "raw/seq") if k in fh)
        seqs = genome if "attributions" in fh else None
    for lo in range(0, n, chunk):
        part = um.read_contributions(h5, sequences=seqs, indices=range(lo, min(lo + chunk, n)))
        if part.ranges is None:
            raise ValueError(f"{h5} gives no coordinates for its scores")
        for (c, s, _), x in zip(part.ranges.iter_rows(), part.scores):
            yield c, s, x


# ---------------------------------------------------------------------------
# Runs

def _load(work: pathlib.Path, model: str, head: str, hypothetical: bool):
    """The letters, observed scores and (if asked for and there are some)
    hypothetical scores of every window of a model's head."""
    d = work / "data" / model / head
    seqs = np.load(d / "seqs.npy")
    strs = ["".join(x) for x in LETTERS[seqs]]
    hyp = None
    if hypothetical and (d / "hyp.npy").exists():
        hyp = np.load(d / "hyp.npy").astype(np.float32)
    return strs, np.load(d / "obs.npy").astype(np.float32), hyp


def run_one(spec: dict) -> dict:
    """One method on one model and head, in this process."""
    import universalmotif as um

    out = pathlib.Path(spec["out"])
    m = config.METHODS[spec["method"]]
    strs, obs, hyp = _load(pathlib.Path(spec["work"]), spec["model"], spec["head"], m.get("hypothetical", True))
    t0 = time.time()
    if m["kind"] == "cm":
        found = um.contribution_motifs(obs, strs, hypothetical=hyp, seed=1, nthreads=config.THREADS, hits=False,
                                       **m["kwargs"])
        result = {"cwms": list(found.cwms), "ppms": list(found.ppms), "motifs": found.motifs.drop("motif", "cwm"),
                  "seqlets": found.seqlets.height}
    else:
        result = _modisco(spec, strs, hyp, m)
    seconds = time.time() - t0
    # TF-MoDISco runs as a command, so its memory is that of the largest child.
    rss = max(resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    gb = rss / (2**30 if sys.platform == "darwin" else 2**20)
    result.update(seconds=seconds, peak_gb=gb, windows=len(strs), hypothetical=hyp is not None)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(result, fh)
    return {k: spec[k] for k in ("model", "head", "method")} | {
        "seconds": round(seconds, 1), "peak_gb": round(gb, 2), "windows": len(strs),
        "hypothetical": hyp is not None, "motifs": len(result["cwms"])}


def _modisco(spec, strs, hyp, m) -> dict:
    """TF-MoDISco (modisco-lite) through its command, found as MODISCO or on
    the PATH, on the same windows and hypothetical scores."""
    if hyp is None:
        raise ValueError("TF-MoDISco needs hypothetical scores")
    tmp = pathlib.Path(spec["out"]).with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    onehot = np.zeros(hyp.shape, dtype=np.float32)
    codes = _codes(strs)
    r, c = np.nonzero(codes < 4)
    onehot[r, codes[r, c], c] = 1
    np.save(tmp / "ohe.npy", onehot)
    np.save(tmp / "hyp.npy", hyp)
    exe = os.environ.get("MODISCO") or shutil.which("modisco")
    if exe is None:
        raise RuntimeError("set MODISCO to the modisco command of a modisco-lite install")
    h5 = tmp / "modisco.h5"
    cmd = [exe, "motifs", "-s", str(tmp / "ohe.npy"), "-a", str(tmp / "hyp.npy"), "-n", str(m["max_seqlets"]),
           "-w", str(hyp.shape[2]), "-o", str(h5)]
    env = {**os.environ, "NUMBA_NUM_THREADS": str(config.THREADS), "OMP_NUM_THREADS": str(config.THREADS)}
    subprocess.run(cmd, check=True, env=env)
    import universalmotif as um

    return {"cwms": _modisco_patterns(um, h5, "CWM"), "ppms": _modisco_patterns(um, h5, "PPM"), "h5": str(h5)}


def _modisco_patterns(um, h5, type: str) -> list:
    """The positive patterns of a TF-MoDISco results file, trimmed as its own reports are."""
    return [p for p in um.read_modisco(h5, type=type, trim_threshold=0.3) if p.name.startswith("pos")]


def run(args):
    """Every method on every model and head, args.jobs at a time, each in a
    process of its own. Only heads with hypothetical scores are run, which
    TF-MoDISco needs, so that every method is compared on the same heads."""
    work = pathlib.Path(args.work)
    specs = []
    for row in manifest(args.models).iter_rows(named=True):
        has_hyp = (work / "data" / row["model"] / row["head"] / "hyp.npy").exists()
        for method in args.methods or list(config.METHODS):
            out = work / "results" / row["model"] / row["head"] / f"{method}.pkl"
            if out.exists() or not has_hyp:
                continue
            specs.append({"work": str(work), "model": row["model"], "head": row["head"], "method": method,
                          "out": str(out)})
    print(f"{len(specs)} runs to do", file=sys.stderr)

    def launch(spec):
        p = subprocess.run([sys.executable, __file__, "_one", json.dumps(spec)], capture_output=True, text=True,
                           check=False)
        if p.returncode:
            return {**{k: spec[k] for k in ("model", "head", "method")}, "error": p.stderr[-2000:]}
        return json.loads(p.stdout.strip().splitlines()[-1])

    with cf.ThreadPoolExecutor(args.jobs) as ex, open(work / "runs.jsonl", "a") as fh:
        for res in ex.map(launch, specs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(json.dumps(res)[:300], file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "prep", "run", "_one"])
    ap.add_argument("spec", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--models", nargs="*", help="regular expressions matched against model names")
    ap.add_argument("--methods", nargs="*", help="methods from config.py (default: all of them)")
    ap.add_argument("--jobs", type=int, default=config.JOBS, help="runs side by side, each with config.THREADS threads")
    args = ap.parse_args()
    if args.stage == "_one":
        print(json.dumps(run_one(json.loads(args.spec))))
        return
    {"fetch": fetch, "prep": prep, "run": run}[args.stage](args)


if __name__ == "__main__":
    main()
