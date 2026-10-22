"""Generate the example data shipped in python/universalmotif/data.

Everything here is synthetic. The motifs are built from well-known
consensus sites, with probabilities drawn at random around each site and
counts from a random number of sampled sites, and the sequences are random
promoter-like sequences with some of the motifs planted in them. Run from
the repository root:

    uv run --no-sync python scripts/make_example_data.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

import universalmotif as um

OUT = Path("python/universalmotif/data")
RNG = np.random.default_rng(2026)

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "S": "CG",
    "W": "AT", "K": "GT", "M": "AC", "N": "ACGT",
}

# Name, family, consensus. Lower-case letters are weak flanking positions.
MOTIFS = [
    ("TATA-box", "TBP", "nTATAWAWRn"),
    ("AP-1", "bZIP", "nTGASTCAn"),
    ("CRE", "bZIP", "TGACGTCA"),
    ("G-box", "bZIP", "nCACGTGn"),
    ("E-box", "bHLH", "nCAGCTGn"),
    ("W-box", "WRKY", "nnTTGACYn"),
    ("GATA", "GATA", "nWGATARn"),
    ("GC-box", "C2H2 zinc finger", "GGGGCGGGG"),
    ("ETS", "ETS", "nCCGGAAGTn"),
    ("CArG-box", "MADS", "CCWWWWWWGG"),
    ("Telo-box", "Myb-related", "AAACCCTAA"),
    ("CCAAT-box", "NF-Y", "nnCCAATnn"),
]


def column(letter: str) -> np.ndarray:
    """Probabilities for one position, drawn around its consensus letter."""
    weak = letter.islower()
    allowed = IUPAC[letter.upper()]
    base = np.full(4, 0.04)
    for b in allowed:
        base["ACGT".index(b)] += (0.6 if weak else 0.88) / len(allowed)
    if weak:
        base = 0.5 * base + 0.5 * np.full(4, 0.25)
    base /= base.sum()
    return RNG.dirichlet(base * (12 if weak else 40))


def make_motif(name: str, family: str, consensus: str) -> um.Motif:
    ppm = np.column_stack([column(c) for c in consensus])
    nsites = int(RNG.integers(20, 80))
    # Counts from sampled sites, so that the motif looks like real data.
    sites = ["".join(RNG.choice(list("ACGT"), p=ppm[:, j]) for j in range(ppm.shape[1]))
             for _ in range(nsites)]
    return um.create_motif(sites, type="PCM", name=name, family=family, organism="synthetic",
                           pseudocount=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    motifs = [make_motif(*m) for m in MOTIFS]
    # TRANSFAC keeps each motif's family and organism.
    um.write_transfac(motifs, OUT / "example_motifs.transfac", overwrite=True)

    # AT-rich promoter-like sequences. W-box and G-box sites are planted in
    # the targets, towards the middle; the controls have no planted sites.
    freqs = {"A": 0.33, "C": 0.17, "G": 0.17, "T": 0.33}
    targets = um.create_sequences(seqnum=100, seqlen=1000, freqs=freqs, seed=11)
    controls = um.create_sequences(seqnum=100, seqlen=1000, freqs=freqs, seed=12)
    by_name = {m.name: m for m in motifs}
    for motif, fraction, seed in ((by_name["W-box"], 0.6, 13), (by_name["G-box"], 0.4, 14)):
        chosen = sorted(RNG.choice(len(targets), size=int(fraction * len(targets)), replace=False))
        planted = um.implant_motifs(motif, [targets[i] for i in chosen], centre_bias=3, seed=seed)
        for i, seq in zip(chosen, planted):
            targets[i] = seq
    for name, seqs in (("targets", targets), ("controls", controls)):
        records = {f"{name[:-1]}_{i + 1:03d}": s for i, s in enumerate(seqs)}
        text_path = OUT / f"example_{name}.fa"
        um.write_fasta(records, text_path, width=80, overwrite=True)
        with open(text_path, "rb") as f, gzip.GzipFile(str(text_path) + ".gz", "wb", mtime=0) as g:
            g.write(f.read())
        text_path.unlink()

    # One small file in each motif format.
    four = motifs[:4]
    um.write_meme(four, OUT / "example.meme", overwrite=True)
    um.write_jaspar(four, OUT / "example.jaspar", overwrite=True)
    um.write_homer(four, OUT / "example.motif", overwrite=True)
    um.write_transfac(four, OUT / "example.transfac", overwrite=True)
    um.write_cisbp(four, OUT / "example.cisbp", overwrite=True)
    um.write_uniprobe(four, OUT / "example.uniprobe", overwrite=True)
    um.write_matrix(four, OUT / "example.matrix", overwrite=True)


if __name__ == "__main__":
    main()
