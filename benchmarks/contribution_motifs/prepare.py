"""Inputs for the contribution_motifs() benchmarks, from ENCODE's files for
the chromBPNet model of ATAC-seq in K562 cells (ENCSR868FGK), cached in
~/.cache/universalmotif-docs.

    python prepare.py sample OUTDIR       # 30,000 regions, with hypothetical scores
    python prepare.py all BIGWIG BED OUTDIR   # every region of a model

Regions are 500 bp windows around the middles of the peaks of ENCFF834JBL,
none overlapping another. "sample" draws 30,000 of them at random (seed 1)
and writes, as the modisco command takes them, the one-hot sequences and
hypothetical scores as (regions, 4, 500) arrays, the observed scores those
imply, and the sequences as letter codes (A, C, G, T, N as 0 to 4). Its
hypothetical scores come from ENCODE's fold-mean score file (ENCFF945WIK,
which needs h5py and hdf5plugin). "all" takes every region of the
peaks in BED, with the observed scores of a bigWig track (for this model,
ENCFF834JBL and ENCFF407GCO for its counts head), and writes the letter
codes and those scores.
"""

import pathlib
import sys

import numpy as np
import polars as pl
import pybigtools
import universalmotif as um

CACHE = pathlib.Path.home() / ".cache" / "universalmotif-docs"
REGIONS = CACHE / "ENCFF834JBL.bed.gz"
SCORES = CACHE / "encff945wik" / "seq_contrib.counts.fold_mean.all_regions.ENCSR868FGK.h5"
W = 500
N = 30_000


def windows(bed=REGIONS) -> pl.DataFrame:
    regions = um.read_bed(bed).with_row_index("region_i")
    return regions.with_columns(wstart=(pl.col("start") + pl.col("end")) // 2 - W // 2)


def sample_windows() -> pl.DataFrame:
    """N windows drawn at random, each clear of those drawn before it."""
    regions = windows()
    order = np.random.default_rng(1).permutation(regions.height)
    chrom, wstart = regions["chrom"].to_numpy(), regions["wstart"].to_numpy()
    taken: dict[str, list[int]] = {}
    keep = []
    for i in order:
        c, s = chrom[i], wstart[i]
        if any(abs(s - t) < W for t in taken.get(c, ())):
            continue
        taken.setdefault(c, []).append(s)
        keep.append(i)
        if len(keep) == N:
            break
    return regions[np.array(keep)].sort("region_i")


def all_windows(bed) -> pl.DataFrame:
    """Every window, kept along each chromosome when it clears the last."""
    regions = windows(bed)
    keep, last = [], {}
    for i, c, s in regions.sort("chrom", "wstart").select("region_i", "chrom", "wstart").iter_rows():
        if s >= 0 and s - last.get(c, -W) >= W:
            keep.append(i)
            last[c] = s
    return regions.filter(pl.col("region_i").is_in(keep)).sort("region_i")


def codes(sel: pl.DataFrame) -> np.ndarray:
    genome = um.Genome(CACHE / "hg38.2bit")
    table = np.full(256, 4, dtype=np.uint8)
    for k, b in enumerate("ACGT"):
        table[ord(b)] = table[ord(b.lower())] = k
    out = np.empty((sel.height, W), dtype=np.uint8)
    for j, (c, s) in enumerate(zip(sel["chrom"], sel["wstart"])):
        out[j] = table[np.frombuffer(genome.subseq(c, s, s + W).encode(), dtype=np.uint8)]
    return out


def hypothetical(sel: pl.DataFrame) -> np.ndarray:
    """The central 500 bases of the windows' rows of the fold-mean scores, as
    (regions, 4, 500)."""
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(SCORES, "r") as f:
        d = f["shap"]["seq"]
        chrom = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["coords_chrom"][:]])
        centre = f["coords_start"][:] + (f["coords_end"][:] - f["coords_start"][:]) // 2
        lookup = {(c, int(x)): i for i, (c, x) in enumerate(zip(chrom, centre))}
        want = [(c, int(w) + W // 2) for c, w in zip(sel["chrom"], sel["wstart"])]
        idx = np.array([lookup.get(k, lookup.get((k[0], k[1] - 1), lookup.get((k[0], k[1] + 1), -1)))
                        for k in want])
        if (idx < 0).any():
            raise ValueError(f"{int((idx < 0).sum())} windows are missing from the score file")
        off = (d.shape[2] - W) // 2
        return np.asarray(d[:, :, off:off + W][idx], dtype=np.float32)


if __name__ == "__main__":
    kind, out = sys.argv[1], pathlib.Path(sys.argv[-1])
    out.mkdir(parents=True, exist_ok=True)
    if kind == "sample":
        sel = sample_windows()
        seqs = codes(sel)
        hyp = hypothetical(sel)
        onehot = (seqs[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32)
        np.save(out / "codes.npy", seqs)
        np.save(out / "ohe.npy", onehot)
        np.save(out / "hyp.npy", hyp)
        np.save(out / "observed.npy", (hyp * onehot).sum(axis=1))
    else:
        sel = all_windows(sys.argv[3])
        seqs = codes(sel)
        bw = pybigtools.open(sys.argv[2])
        attr = np.stack([np.nan_to_num(bw.values(c, s, s + W, missing=0.0))
                         for c, s in zip(sel["chrom"], sel["wstart"])]).astype(np.float32)
        np.save(out / "codes.npy", seqs)
        np.save(out / "observed.npy", attr)
    sel.select("region_i", "chrom", "wstart").write_parquet(out / "regions.parquet")
    print(f"{sel.height} windows written to {out}")
