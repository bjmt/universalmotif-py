"""Examples for the profiles and heatmaps of signal and of motif sites
around genomic sites."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import universalmotif as um

from . import Example

_COMP = str.maketrans("ACGT", "TGCA")
_SIZES = {"chr1": 400_000, "chr2": 350_000, "chr3": 300_000}


def _rc(site: str) -> str:
    return site.translate(_COMP)[::-1]


@cache
def _motifs() -> dict[str, um.Motif]:
    return {m.name: m for m in um.example_motifs()}


def _sites(name: str, n: int, seed: int) -> list[str]:
    """Sites sampled from a motif, keeping those that a scan at a P-value
    of 1e-4 finds, so that most planted sites are seen again."""
    motif = _motifs()[name]
    pool = um.sample_sites(motif, 12 * n, seed=seed)
    found = um.scan_sequences(motif, pool, pvalue=1e-4, rc=False)["sequence_i"].unique().sort()
    kept = [pool[i] for i in found.to_list()]
    return (kept * (n // len(kept) + 1))[:n]


@cache
def _genome() -> tuple[dict[str, str], pl.DataFrame]:
    """Three random chromosomes with 240 stranded anchors, such as TSSs,
    3,500 to 5,000 bp apart. AP-1 sites are planted within 30 bp of most
    anchors, and GATA sites 150 to 250 bp upstream of many of them."""
    rng = np.random.default_rng(3)
    genome = {c: "".join(rng.choice(list("ACGT"), size=n)) for c, n in _SIZES.items()}
    rows = []
    for chrom in _SIZES:
        n = {"chr1": 90, "chr2": 80, "chr3": 70}[chrom]
        gaps = rng.integers(3_500, 5_000, n)
        for pos in 1_000 + np.cumsum(gaps):
            rows.append((chrom, int(pos), int(pos) + 1, "+" if rng.random() < 0.5 else "-"))
    anchors = pl.DataFrame(rows, schema=["chrom", "start", "end", "strand"], orient="row")
    ap1 = iter(_sites("AP-1", len(rows), seed=21))
    gata = iter(_sites("GATA", len(rows), seed=22))

    def put(chrom: str, site: str, pos: int) -> None:
        if rng.random() < 0.5:
            site = _rc(site)
        s = genome[chrom]
        genome[chrom] = s[:pos] + site + s[pos + len(site):]

    for chrom, start, _, strand in rows:
        if rng.random() < 0.7:
            site = next(ap1)
            put(chrom, site, start + int(rng.integers(-30, 31)) - len(site) // 2)
        if rng.random() < 0.6:
            site = next(gata)
            offset = int(rng.integers(150, 251))
            put(chrom, site, start - offset - len(site) // 2 if strand == "+" else start + offset - len(site) // 2)
    return genome, anchors


@cache
def _hits() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hits of AP-1 and GATA in the genome, and of shuffled copies of them
    named as the originals, at a P-value of 1e-4. The shuffles are ones that
    no longer resemble the motifs, which a short palindrome such as AP-1
    can still do after some shuffles."""
    genome, _ = _genome()
    motifs = [_motifs()[n] for n in ("AP-1", "GATA")]
    shuffled = [um.shuffle_motifs(m, seed=s).replace(name=m.name) for m, s in zip(motifs, (2, 5))]
    hits = um.scan_sequences(motifs, genome, pvalue=1e-4, nthreads=2)
    control = um.scan_sequences(shuffled, genome, pvalue=1e-4, nthreads=2)
    return hits, control


@cache
def _bigwig() -> Path:
    """A ChIP track with noise everywhere and a peak of signal at every
    anchor."""
    import pybigtools

    _, anchors = _genome()
    rng = np.random.default_rng(4)
    path = Path(tempfile.mkdtemp()) / "chip.bw"
    step = 20

    def rows(chrom: str, size: int):
        starts = np.arange(0, size, step)
        signal = rng.gamma(2.0, 0.5, len(starts))
        for summit in anchors.filter(pl.col("chrom") == chrom)["start"].to_numpy():
            signal += 6 * np.exp(-0.5 * ((starts + step / 2 - summit) / 80) ** 2)
        for s, v in zip(starts, signal):
            yield chrom, int(s), int(min(s + step, size)), float(v)

    pybigtools.open(str(path), "w").write(_SIZES, (r for c, n in _SIZES.items() for r in rows(c, n)))
    return path


def _chip():
    hits, _ = _hits()
    _, anchors = _genome()
    ap1 = hits.filter(pl.col("motif") == "AP-1").rename({"sequence": "chrom"})
    middle = ((ap1["start"] + ap1["end"]) // 2).to_numpy()
    dist = np.zeros(ap1.height, dtype=np.int64)
    for chrom in _SIZES:
        on = (ap1["chrom"] == chrom).to_numpy()
        summits = np.sort(anchors.filter(pl.col("chrom") == chrom)["start"].to_numpy())
        i = np.clip(np.searchsorted(summits, middle[on]), 1, len(summits) - 1)
        dist[on] = np.minimum(np.abs(middle[on] - summits[i - 1]), np.abs(middle[on] - summits[i]))
    sites = ap1.with_columns(dist=pl.Series(dist))
    peak = um.signal_matrix(_bigwig(), sites.filter(pl.col("dist") < 500), flank=1000, bin_size=20,
                            name="ChIP")
    rest = um.signal_matrix(_bigwig(), sites.filter(pl.col("dist") > 1500), flank=1000, bin_size=20,
                            name="ChIP")
    return um.plot_signal_profile(peak, control=rest, ylabel="Mean ChIP", legend_title="ChIP")


@cache
def _density() -> tuple[um.SignalMatrix, um.SignalMatrix]:
    hits, control = _hits()
    _, anchors = _genome()
    return (um.site_matrix(hits, anchors, flank=500, bin_size=10),
            um.site_matrix(control, anchors, flank=500, bin_size=10))


def _sites_heatmap():
    density, control = _density()
    return um.plot_signal_profile(density, control=control, ylabel="Sites per anchor", drop_empty=True)


def _sites_profile():
    density, control = _density()
    return um.plot_signal_profile(density, control=control, heatmap=False, ylabel="Sites per anchor")


EXAMPLES = {
    "plot_signal_profile": [
        Example("A made-up ChIP track with a peak at each of 240 anchors on three random chromosomes, "
                "read with signal_matrix() around the AP-1 hits within 500 bp of an anchor, against "
                "the AP-1 hits more than 1,500 bp from any anchor in grey.", _chip),
        Example("AP-1 and GATA sites counted with site_matrix() around the same stranded anchors, "
                "where AP-1 sites were planted within 30 bp of most anchors and GATA sites 150 to "
                "250 bp upstream of many, with hits of shuffled copies of the motifs in grey and "
                "anchors without a site left out of the heatmaps with drop_empty=True.", _sites_heatmap),
        Example("The same site counts as profiles only, with heatmap=False.", _sites_profile),
    ],
}
