"""Examples of the maps of interactions between the positions of a sequence.

The interactions come from in silico double mutagenesis of a toy model whose
filters are example motifs. GATA and E-box sites earn a bonus together, so
breaking either site loses it, while an AP-1 site adds to the prediction on
its own.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl
import universalmotif as um

from . import Example

_START = 12_408_350
_NAMES = ("GATA", "E-box", "AP-1")


@cache
def _motifs() -> list[um.Motif]:
    found = {m.name: m for m in um.normalize(um.example_motifs())}
    return [found[n] for n in _NAMES]


class _Model:
    """Each motif's PWM is a filter scanned along both strands, and the
    strength of its site is a sigmoid of the best window's score above a
    cutoff. The prediction adds up the strengths, plus a joint term for GATA
    and E-box sites together, the product of their strengths."""

    def __init__(self, joint: float = 3.0, margin: float = 3.0):
        f = um.to_filters(_motifs()).astype(np.float32)
        self.f = np.concatenate([f, f[:, ::-1, ::-1]])
        self.t = np.tile(f.max(axis=1).sum(axis=1) - margin, 2)
        self.joint = joint

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        w = self.f.shape[2]
        p = x.shape[2] - w + 1
        z = np.zeros((x.shape[0], p, self.f.shape[0]), dtype=np.float32)
        xt = x.transpose(0, 2, 1)
        for j in range(w):
            z += xt[:, j:j + p] @ self.f[:, :, j].T
        best = (z.max(axis=1) - self.t).reshape(len(x), 2, -1).max(axis=1)
        s = 1 / (1 + np.exp(-best))
        return s.sum(axis=1) + self.joint * s[:, 0] * s[:, 1]


@cache
def _locus() -> tuple[str, pl.DataFrame]:
    """A 100-letter sequence with a GATA site, an E-box 26 letters on and an
    AP-1 site near the end, and the hits of the three motifs in it."""
    seq = um.create_sequences(seqnum=1, seqlen=100, seed=8)[0]
    seq = um.insert_sites(seq, list(_motifs()), positions=[18, 52, 80])
    hits = um.scan_sequences(_motifs(), seq, pvalue=1e-4)
    hits = hits.unique(["sequence_i", "motif", "start"], keep="first", maintain_order=True)
    return seq, hits


@cache
def _epistasis() -> tuple[np.ndarray, np.ndarray]:
    seq, _ = _locus()
    return um.interaction_effects(_Model()(um.pair_mutagenesis(seq)), seq)


_REGION = (12, 64)


@cache
def _every_pair() -> tuple[np.ndarray, np.ndarray]:
    seq, _ = _locus()
    batch = um.pair_mutagenesis(seq, region=_REGION, alt="all")
    return um.interaction_effects(_Model()(batch), seq, region=_REGION, alt="all")


def _square():
    seq, hits = _locus()
    epistasis, _ = _epistasis()
    return um.plot_interactions(epistasis, seq, start=_START, hits=hits, label="Epistasis")


def _triangle():
    seq, hits = _locus()
    epistasis, single = _epistasis()
    return um.plot_interactions(epistasis, seq, scores=-single, start=_START, hits=hits, triangle=True,
                                max_distance=60, label="Epistasis")


def _all_letters():
    seq, hits = _locus()
    epistasis, effects = _every_pair()
    region = (_START + _REGION[0], _START + _REGION[1])
    return um.plot_interactions(epistasis, seq, scores=-effects.mean(axis=0) * 4 / 3, start=_START,
                                region=region, hits=hits, label="Mean epistasis")


EXAMPLES = {
    "plot_interactions": [
        Example("Epistasis between every pair of positions of a toy sequence, from pair_mutagenesis() "
                "and interaction_effects() with a toy model whose GATA and E-box filters earn a "
                "bonus together while its AP-1 filter acts alone. Each position is replaced by its "
                "complement, and each pair of hits is outlined.", _square),
        Example("The same epistasis with triangle=True, cut off at a distance of 60 with "
                "max_distance, above the drop in the prediction when each position is changed, "
                "drawn as a logo of the sequence.", _triangle),
        Example('Every substitution and every pair of them with alt="all", over the 52 positions '
                "from the GATA site to the E-box, given as region, with the mean epistasis of each "
                "pair of positions and the mean drop at each position as the logo.", _all_letters),
    ],
}
