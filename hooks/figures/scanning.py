"""Examples for the plots of hits, spacings, enrichment, co-occurrence,
positional peaks, matched backgrounds and proximity to anchors."""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl
import universalmotif as um

from . import Example

_COMP = str.maketrans("ACGT", "TGCA")


def _rc(site: str) -> str:
    return site.translate(_COMP)[::-1]


def _random_dna(rng: np.random.Generator, n: int, gc: float = 0.5) -> str:
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=n, p=p))


def _put(seq: str, site: str, pos: int) -> str:
    return seq[:pos] + site + seq[pos + len(site):]


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


# --- plot_hits ----------------------------------------------------------------


@cache
def _hit_seqs() -> dict[str, str]:
    """Six sequences of 400 bp with G-box, W-box and GATA sites planted at
    random places and on random strands."""
    rng = np.random.default_rng(11)
    seqs = um.create_sequences(seqnum=6, seqlen=400, seed=11)
    planted = {"G-box": 2, "W-box": 2, "GATA": 3}
    out = {}
    for i, s in enumerate(seqs):
        for k, (name, n) in enumerate(planted.items()):
            for j, site in enumerate(_sites(name, n, seed=100 * i + 10 * k)):
                if rng.random() < 0.5:
                    site = _rc(site)
                s = _put(s, site, int(rng.integers(0, 400 - len(site))))
        out[f"seq{i + 1}"] = s
    return out


def _hits_sequences():
    motifs = [_motifs()[n] for n in ("G-box", "W-box", "GATA")]
    seqs = _hit_seqs()
    hits = um.scan_sequences(motifs, seqs, pvalue=1e-4)
    return um.plot_hits(hits, seqs, motifs=motifs)


_PEAKS = {
    "chrom": ["chr1", "chr1", "chr2", "chr2"],
    "start": [12_000, 31_500, 8_200, 25_000],
    "end": [12_600, 32_100, 8_800, 25_600],
    "name": ["peak1", "peak2", "peak3", "peak4"],
}


@cache
def _genome() -> dict[str, str]:
    """Two random chromosomes with AP-1, ETS and CArG-box sites planted at
    random places in four peaks."""
    rng = np.random.default_rng(5)
    genome = {"chr1": _random_dna(rng, 60_000), "chr2": _random_dna(rng, 40_000)}
    planted = {"AP-1": 3, "ETS": 2, "CArG-box": 2}
    pools = {n: iter(_sites(n, 4 * k, seed=90 + i)) for i, (n, k) in enumerate(planted.items())}
    for chrom, start, end in zip(_PEAKS["chrom"], _PEAKS["start"], _PEAKS["end"]):
        for name, k in planted.items():
            for _ in range(k):
                site = next(pools[name])
                site = site if rng.random() < 0.5 else _rc(site)
                genome[chrom] = _put(genome[chrom], site, int(rng.integers(start, end - len(site))))
    return genome


def _hits_ranges():
    peaks = pl.DataFrame(_PEAKS)
    seqs = um.extract_ranges(_genome(), peaks)
    motifs = [_motifs()[n] for n in ("AP-1", "ETS", "CArG-box")]
    hits = um.genomic_hits(um.scan_sequences(motifs, seqs, pvalue=1e-4))
    return um.plot_hits(hits, ranges=peaks, motifs=motifs, shade_by="pvalue")


def _hits_many():
    seqs = dict(list(_hit_seqs().items())[:3])
    hits = um.scan_sequences(um.example_motifs(), seqs, pvalue=1e-3)
    return um.plot_hits(hits, seqs, size_by="score", labels=False)


# --- plot_pair_spacing ----------------------------------------------------------


def _scatter(rng: np.random.Generator, seq: str, sites: list[str], chance: float) -> str:
    """Write each site, with the given chance, at a random place on a random
    strand."""
    for site in sites:
        if rng.random() < chance:
            site = site if rng.random() < 0.5 else _rc(site)
            seq = _put(seq, site, int(rng.integers(0, len(seq) - len(site))))
    return seq


@cache
def _paired_hits() -> pl.DataFrame:
    """400 random sequences of 200 bp, each with GATA and W-box sites at
    random places, and an eighth of them also with a GATA site and a W-box
    on the opposite strand 21 bp downstream, written at a random place."""
    rng = np.random.default_rng(3)
    seqs = um.create_sequences(seqnum=400, seqlen=200, seed=3)
    gata = iter(_sites("GATA", 1200, seed=1))
    wbox = iter(_sites("W-box", 1200, seed=2))
    out = []
    for i, s in enumerate(seqs):
        s = _scatter(rng, s, [next(gata), next(wbox), next(gata), next(wbox)], 0.6)
        if i % 8 == 0:
            # The centres of GATA (8 bp) and W-box (9 bp) sites are their
            # fifth letters, so these lie 21 bp apart.
            p = int(rng.integers(0, 200 - 30))
            s = _put(_put(s, next(gata), p), _rc(next(wbox)), p + 21)
        out.append(s)
    motifs = [_motifs()[n] for n in ("GATA", "W-box")]
    return um.scan_sequences(motifs, out)


def _pairs_hetero():
    return um.plot_pair_spacing(_paired_hits(), "GATA", "W-box", max_distance=60, helix=True)


@cache
def _gata_tandem_hits() -> pl.DataFrame:
    """400 random sequences of 200 bp with GATA sites at random places, and a
    third of them also with a pair of GATA sites in tandem 11, 21 or 32 bp
    apart, about one, two or three turns of the helix, at a random place."""
    rng = np.random.default_rng(8)
    seqs = um.create_sequences(seqnum=400, seqlen=200, seed=8)
    gata = iter(_sites("GATA", 2000, seed=4))
    out = []
    for i, s in enumerate(seqs):
        s = _scatter(rng, s, [next(gata) for _ in range(3)], 0.6)
        if i % 3 == 0:
            gap = int(rng.choice([11, 21, 32]))
            p = int(rng.integers(0, 200 - gap - 8))
            a, b = next(gata), next(gata)
            if rng.random() < 0.5:
                a, b = _rc(b), _rc(a)
            s = _put(_put(s, a, p), b, p + gap)
        out.append(s)
    return um.scan_sequences(_motifs()["GATA"], out)


def _pairs_homo():
    return um.plot_pair_spacing(_gata_tandem_hits(), "GATA", max_distance=50, figsize=(6.4, 4.0))


def _pairs_pooled():
    return um.plot_pair_spacing(_gata_tandem_hits(), "GATA", max_distance=50, arrangements=False,
                                helix=21, figsize=(6.4, 2.6))


# --- plot_enrichment ----------------------------------------------------------


@cache
def _enrichment() -> pl.DataFrame:
    """Targets with sites of four motifs planted at different rates, against
    random controls."""
    targets = um.create_sequences(seqnum=300, seqlen=200, seed=21)
    controls = um.create_sequences(seqnum=300, seqlen=200, seed=22)
    rng = np.random.default_rng(21)
    rates = {"W-box": 0.4, "G-box": 0.25, "GATA": 0.15, "ETS": 0.08, "CArG-box": 0.05}
    pools = {name: iter(_sites(name, 300, seed=30 + k)) for k, name in enumerate(rates)}
    out = []
    for s in targets:
        for name, rate in rates.items():
            if rng.random() < rate:
                site = next(pools[name])
                if rng.random() < 0.5:
                    site = _rc(site)
                s = _put(s, site, int(rng.integers(0, 200 - len(site))))
        out.append(s)
    return um.enrich_motifs(um.example_motifs(), out, controls, qvalue=1)


def _enrich_logos():
    return um.plot_enrichment(_enrichment(), motifs=um.example_motifs(), labels=4)


def _enrich_pvalue():
    return um.plot_enrichment(_enrichment(), y="pvalue", cutoff=0.001, labels=["W-box", "GATA", "CArG-box", "CRE"],
                              size_by=None, figsize=(5.5, 4.2))


# --- plot_coocc ----------------------------------------------------------------


@cache
def _coocc() -> pl.DataFrame:
    """Sequences where W-box and GATA sites come together, G-box and ETS
    sites come together, and G-box and W-box sites never do."""
    seqs = um.create_sequences(seqnum=500, seqlen=200, seed=41)
    rng = np.random.default_rng(41)
    groups = [("W-box", "GATA")] * 100 + [("G-box", "ETS")] * 100 + [("CArG-box",)] * 80
    groups += [()] * (500 - len(groups))
    order = rng.permutation(500)
    names = ["W-box", "GATA", "G-box", "ETS", "CArG-box", "GC-box", "Telo-box"]
    pools = {n: iter(_sites(n, 200, seed=50 + k)) for k, n in enumerate(names)}
    out = []
    for i, s in zip(order, seqs):
        for name in groups[i]:
            site = next(pools[name])
            s = _put(s, site if rng.random() < 0.5 else _rc(site), int(rng.integers(0, 200 - len(site))))
        out.append(s)
    motifs = [_motifs()[n] for n in names]
    return um.motif_coocc(motifs, out, pvalue=1e-4)


def _coocc_odds():
    return um.plot_coocc(_coocc())


def _coocc_qvalue():
    return um.plot_coocc(_coocc(), value="qvalue", cutoff=0.01)


def _coocc_pvalue():
    return um.plot_coocc(_coocc(), value="pvalue", order=["W-box", "GATA", "G-box", "ETS", "CArG-box"])


# --- plot_motif_peaks ----------------------------------------------------------


@cache
def _peak_hits() -> pl.DataFrame:
    """Sequences of 500 bp with W-box sites near their middles, GATA sites
    about 120 bp from their starts and G-box sites anywhere."""
    seqs = um.create_sequences(seqnum=300, seqlen=500, seed=61)
    rng = np.random.default_rng(61)
    wbox, gata, gbox = _sites("W-box", 300, 62), _sites("GATA", 300, 63), _sites("G-box", 300, 64)
    out = []
    for i, s in enumerate(seqs):
        if rng.random() < 0.5:
            s = _put(s, wbox[i], int(np.clip(rng.normal(246, 25), 0, 491)))
        if rng.random() < 0.4:
            s = _put(s, _rc(gata[i]), int(np.clip(rng.normal(116, 15), 0, 492)))
        if rng.random() < 0.5:
            s = _put(s, gbox[i], int(rng.integers(0, 492)))
        out.append(s)
    motifs = [_motifs()[n] for n in ("W-box", "GATA", "G-box", "ETS")]
    return um.scan_sequences(motifs, out, pvalue=1e-4)


def _peaks_central():
    peaks = um.motif_peaks(_peak_hits(), seq_length=500, qvalue=1)
    return um.plot_motif_peaks(peaks, bins=40)


def _peaks_local():
    peaks = um.motif_peaks(_peak_hits(), seq_length=500, mode="local", qvalue=1)
    return um.plot_motif_peaks(peaks, motifs=["W-box", "GATA"], bins=40, figsize=(8, 2.8))


# --- plot_match_bkg ----------------------------------------------------------


@cache
def _match_sets():
    """GC-rich targets of about 300 bp, and a universe of 3,000 sequences of
    every GC content from 30 to 70% and lengths from 100 to 700 bp."""
    rng = np.random.default_rng(71)
    targets = [_random_dna(rng, int(np.clip(rng.normal(300, 50), 120, 600)),
                           float(np.clip(rng.normal(0.6, 0.04), 0.3, 0.7))) for _ in range(200)]
    universe = [_random_dna(rng, int(rng.integers(100, 700)), float(rng.uniform(0.3, 0.7)))
                for _ in range(3000)]
    return targets, universe


def _match_matched():
    targets, universe = _match_sets()
    bkg = um.match_bkg(targets, universe, seed=1)
    return um.plot_match_bkg(targets, bkg, figsize=(8, 3))


def _match_covariate():
    targets, universe = _match_sets()
    rng = np.random.default_rng(73)
    signal = {"signal": rng.gamma(6.0, 2.5, len(targets))}
    universe_signal = {"signal": rng.gamma(2.0, 5.0, len(universe))}
    idx = um.match_bkg(targets, universe, covariates=signal, universe_covariates=universe_signal,
                       n_bins_covariates=4, seed=1, return_indices=True)
    return um.plot_match_bkg(indices=idx, figsize=(11, 3))


# --- plot_motif_proximity ----------------------------------------------------


@cache
def _proximity_setup():
    """Three chromosomes of 145 kb with 100 stranded anchors each, about
    1.5 kb apart. W-box sites lie within 60 bp of the anchors, GATA sites
    150 to 300 bp upstream of them, and G-box sites anywhere."""
    rng = np.random.default_rng(81)
    genome, anchors = {}, []
    wbox, gata, gbox = _sites("W-box", 300, 82), _sites("GATA", 300, 83), _sites("G-box", 240, 84)
    k = 0
    for c in ("chr1", "chr2", "chr3"):
        seq = list(_random_dna(rng, 145_000))
        pos = 725 + 1_450 * np.arange(100) + rng.integers(-200, 200, 100)
        for p in pos:
            strand = "+" if rng.random() < 0.5 else "-"
            anchors.append((c, int(p), int(p) + 1, strand))
            w = int(p + rng.integers(-60, 52))
            seq[w:w + 9] = wbox[k] if rng.random() < 0.5 else _rc(wbox[k])
            off = int(rng.integers(150, 300))
            g = int(p - off - 4) if strand == "+" else int(p + off - 4)
            site = gata[k] if strand == "+" else _rc(gata[k])
            seq[g:g + 8] = site
            k += 1
        for p in rng.integers(0, 145_000 - 8, 80):
            seq[p:p + 8] = gbox[int(rng.integers(0, 240))]
        genome[c] = "".join(seq)
    motifs = [_motifs()[n] for n in ("W-box", "GATA", "G-box")]
    return genome, anchors, motifs


@cache
def _proximity() -> pl.DataFrame:
    genome, anchors, motifs = _proximity_setup()
    return um.motif_proximity(anchors=anchors, motifs=motifs, sequences=genome, qvalue=1)


def _proximity_signed():
    return um.plot_motif_proximity(_proximity(), ncol=3, figsize=(11, 2.8))


def _proximity_log():
    return um.plot_motif_proximity(_proximity(), ncol=3, log_x=True, figsize=(11, 2.8))


def _proximity_upstream():
    genome, anchors, motifs = _proximity_setup()
    up = um.motif_proximity(anchors=anchors, motifs=motifs, sequences=genome, orientation="upstream",
                            dist_thresholds=[100, 200, 300, 400], qvalue=1)
    return um.plot_motif_proximity(up, motifs=["GATA", "W-box"], ncol=2, figsize=(8, 2.8))


EXAMPLES = {
    "plot_hits": [
        Example("Hits of the G-box, W-box and GATA example motifs in six random sequences with their sites "
                "planted at random places, with the logos of the motifs in the legend.", _hits_sequences),
        Example("Hits in four peaks of a random genome with AP-1, ETS and CArG-box sites planted in them, "
                "moved to genomic coordinates with genomic_hits() and drawn over the peaks given as named "
                "ranges, shaded by P-value.", _hits_ranges),
        Example("Hits of the twelve example motifs in three of those sequences at a P-value of 1e-3, "
                "sized by score, with overlapping hits stacked in lanes.", _hits_many),
    ],
    "plot_pair_spacing": [
        Example("GATA sites paired with a W-box on the opposite strand 21 bp downstream, written at random "
                "places into 50 of 400 random sequences with other sites of both scattered among them, "
                "show up as a peak in the convergent panel; helix=True marks the helical period through "
                "it.", _pairs_hetero),
        Example("A motif paired with itself: GATA sites written in tandem 11, 21 or 32 bp apart into a "
                "third of 400 random sequences with other GATA sites scattered among them, split into "
                "the three arrangements of a homotypic pair.", _pairs_homo),
        Example("The same GATA pairs with arrangements=False, every pair in one panel, and the helical "
                "period marked through a distance of 21 bp.", _pairs_pooled),
    ],
    "plot_enrichment": [
        Example("Enrichment of the twelve example motifs in targets with W-box, G-box, GATA, ETS and "
                "CArG-box sites planted at falling rates, against random controls, with logos beside the four most "
                "significant.", _enrich_logos),
        Example("The same results by P-value with a cutoff of 0.001, points of one size and labels for "
                "motifs chosen by name, significant or not.", _enrich_pvalue),
    ],
    "plot_coocc": [
        Example("Log2 odds ratios of co-occurrence of seven motifs in 500 sequences, where W-box and GATA "
                "sites were planted together, G-box and ETS sites together, and CArG-box sites alone, "
                "with the motifs clustered.", _coocc_odds),
        Example("The same pairs coloured by q-value, with a dot on those at or below a cutoff of 0.01.",
                _coocc_qvalue),
        Example("The same pairs coloured by P-value, with the motifs in a given order that leaves two "
                "out.", _coocc_pvalue),
    ],
    "plot_motif_peaks": [
        Example("Positions of the best hit per sequence of four motifs in 300 sequences of 500 bp, with "
                "W-box sites planted near the middle and GATA and G-box sites elsewhere, tested in "
                "windows centred on the middle.", _peaks_central),
        Example("The same hits tested with mode=\"local\", which finds the GATA sites planted about "
                "120 bp from the start.", _peaks_local),
    ],
    "plot_match_bkg": [
        Example("GC-rich targets of about 300 bp against a background drawn with match_bkg() from a "
                "universe of 3,000 sequences of every GC content from 30 to 70% and every length from "
                "100 to 700 bp.", _match_matched),
        Example("The same targets matched on a signal covariate as well, in four bins, "
                "drawn from the table of match_bkg(return_indices=True) to show the covariate too.",
                _match_covariate),
    ],
    "plot_motif_proximity": [
        Example("Signed distances from motif hits to the nearest of 300 stranded anchors on three "
                "random chromosomes, with W-box sites planted next to the anchors, GATA sites 150 to "
                "300 bp upstream and G-box sites anywhere.", _proximity_signed),
        Example("The same distances as absolute values on a log scale with log_x=True.", _proximity_log),
        Example("GATA and W-box sites tested upstream of the anchors only, with "
                "orientation=\"upstream\", which finds the GATA sites in a zone on that side.",
                _proximity_upstream),
    ],
}
