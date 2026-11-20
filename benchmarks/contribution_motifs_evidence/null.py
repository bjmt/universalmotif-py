#!/usr/bin/env python
"""contribution_motifs() on scores with nothing to find.

prep takes the hypothetical scores of every window of seven heads of six
models (HEADS), gives each window those of another window chosen at random,
turned by a random offset (so that a score at the end of a window wraps
round to its start), and takes the observed scores from them at the
window's own letters. The letters are left as they are, so any motif found
is one that the scores alone cannot support. The shuffled scores go in
work/null/data, beside links to work/files, and are run with

    python harness.py run --work work/null --methods cm --models <HEADS' models>

summary then writes work/null/null.tsv, with the motifs found on each head.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

HEADS = [
    ("bpnet_CEBPB_HepG2", "counts"),
    ("bpnet_CTCF_K562", "counts"),
    ("bpnet_GATA1_K562", "counts"),
    ("chrombpnet_HepG2_ATAC-seq", "counts"),
    ("chrombpnet_K562_ATAC-seq", "counts"),
    ("procapnet_K562", "counts"),
    ("procapnet_K562", "profile"),
]


def prep(args):
    work = pathlib.Path(args.work)
    null = work / "null"
    (null / "files").mkdir(parents=True, exist_ok=True)
    for f in (work / "files").iterdir():
        if not (null / "files" / f.name).exists():
            (null / "files" / f.name).symlink_to(f.resolve())
    rng = np.random.default_rng(args.seed)
    for model, head in HEADS:
        src, out = work / "data" / model / head, null / "data" / model / head
        out.mkdir(parents=True, exist_ok=True)
        seqs = np.load(src / "seqs.npy")
        hyp = np.load(src / "hyp.npy").astype(np.float32)
        n, _, width = hyp.shape
        other = rng.permutation(n)
        shift = rng.integers(0, width, n)
        cols = (np.arange(width)[None, :] - shift[:, None]) % width
        hyp = hyp[other][np.arange(n)[:, None], :, cols].transpose(0, 2, 1)
        onehot = np.zeros((n, 4, width), dtype=np.float32)
        r, c = np.nonzero(seqs < 4)
        onehot[r, seqs[r, c], c] = 1
        np.save(out / "seqs.npy", seqs)
        np.save(out / "hyp.npy", hyp.astype(np.float16))
        np.save(out / "obs.npy", (hyp * onehot).sum(1).astype(np.float32))
        print(f"  {model} {head}: {n} windows", file=sys.stderr, flush=True)


def summary(args):
    null = pathlib.Path(args.work) / "null"
    runs = [json.loads(line) for line in open(null / "runs.jsonl")]
    lines = ["model\thead\tmethod\twindows\tseconds\tpeak_gb\tmotifs\tconsensus"]
    for model, head in HEADS:
        for r in runs:
            if (r["model"], r["head"]) != (model, head) or "error" in r:
                continue
            with open(null / "results" / model / head / f"{r['method']}.pkl", "rb") as fh:
                found = pickle.load(fh)
            cons = ",".join(m.consensus or "" for m in found["ppms"])
            lines.append(f"{model}\t{head}\t{r['method']}\t{r['windows']}\t{r['seconds']}\t{r['peak_gb']}\t"
                         f"{r['motifs']}\t{cons}")
    (null / "null.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "summary"])
    ap.add_argument("--work", default=str(HERE / "work"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    {"prep": prep, "summary": summary}[args.stage](args)


if __name__ == "__main__":
    main()
