"""Examples of the plots for sequence models and their attribution scores.

No model is downloaded: the scores all come from a toy convolutional model
whose filters are example motifs, so that its gradients, in silico
mutagenesis and responses to inserted sites behave like those of a real
model without any of its weights.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl
import universalmotif as um
from numpy.lib.stride_tricks import sliding_window_view

from . import Example

_START = 48_213_600
_NAMES = ("AP-1", "GATA", "E-box")


@cache
def _motifs() -> dict[str, um.Motif]:
    return {m.name: m for m in um.normalize(um.example_motifs())}


def _motif_list(names=_NAMES) -> list[um.Motif]:
    return [_motifs()[n] for n in names]


def _rc(site: str) -> str:
    return site[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _onehot(x) -> np.ndarray:
    if isinstance(x, (str, list, tuple)):
        return um.one_hot(x, dtype=float)
    return np.asarray(x, dtype=float)


class _Model:
    """A toy convolutional model. Each motif's PWM is a filter scanned along
    both strands, and every window adds the softplus of its better strand's
    score above a cutoff, times the motif's weight, so that strong sites
    count for more than weak ones and random windows for almost nothing."""

    def __init__(self, names=_NAMES, weights=None, margin: float = 5.0):
        f = um.to_filters(_motif_list(names))
        self.f = np.concatenate([f, f[:, ::-1, ::-1]])
        self.m = len(names)
        top = f.max(axis=1).sum(axis=1)
        self.t = np.tile(top - margin, 2)
        self.w = np.ones(self.m) if weights is None else np.asarray(weights, dtype=float)

    def _z(self, x: np.ndarray) -> np.ndarray:
        """Window scores above the cutoffs, of shape (sequences, 2 * motifs,
        windows), the forward filters first. Sequences are padded with
        zeros, so that sites at either end are seen too."""
        w = self.f.shape[2]
        x = np.pad(x, ((0, 0), (0, 0), (w - 1, w - 1)))
        win = sliding_window_view(x, w, axis=2)
        return np.einsum("nlpw,mlw->nmp", win, self.f) - self.t[None, :, None]

    def __call__(self, x) -> np.ndarray:
        z = self._z(_onehot(x))
        best = np.maximum(z[:, :self.m], z[:, self.m:])
        return (self.w[None, :, None] * np.logaddexp(0, best)).sum(axis=(1, 2))

    def hypothetical(self, x) -> np.ndarray:
        """The gradient of the prediction for every letter, centred at each
        position, as hypothetical contributions."""
        x = _onehot(x)
        z = self._z(x)
        forward = z[:, :self.m] >= z[:, self.m:]
        best = np.where(forward, z[:, :self.m], z[:, self.m:])
        g = self.w[None, :, None] / (1 + np.exp(-best))
        g = np.concatenate([g * forward, g * ~forward], axis=1)
        w = self.f.shape[2]
        out = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * (w - 1)))
        p = z.shape[2]
        for j in range(w):
            out[:, :, j:j + p] += np.einsum("nmp,ml->nlp", g, self.f[:, :, j])
        out = out[:, :, w - 1:w - 1 + x.shape[2]]
        return out - out.mean(axis=1, keepdims=True)


@cache
def _model() -> _Model:
    return _Model(weights=[1.0, 1.2, 0.7])


def _plant(seq: str, sites: list[str], rng: np.random.Generator) -> tuple[str, list[int]]:
    """Write sites into a sequence at random positions, one per equal slot,
    some of them reverse complemented."""
    slot = len(seq) // len(sites)
    starts = []
    for k, site in enumerate(sites):
        if rng.random() < 0.5:
            site = _rc(site)
        p = k * slot + int(rng.integers(1, slot - len(site)))
        seq = seq[:p] + site + seq[p + len(site):]
        starts.append(p)
    return seq, starts


def _hits(seq, pvalue: float = 1e-4, names=_NAMES) -> pl.DataFrame:
    """Hits of the motifs, keeping one of each pair a palindrome gives on
    both strands."""
    hits = um.scan_sequences(_motif_list(names), seq, pvalue=pvalue)
    return hits.unique(["sequence_i", "motif", "start"], keep="first", maintain_order=True)


# One sequence with four sites, scored by the toy model.


@cache
def _locus():
    rng = np.random.default_rng(4)
    bkg = um.create_sequences(seqnum=1, seqlen=120, seed=12)[0]
    plan = [("AP-1", 4), ("GATA", 8), ("E-box", 4), ("GATA", 5)]
    sites = [um.sample_sites(_motifs()[n], n=1, seed=s)[0] for n, s in plan]
    seq, starts = _plant(bkg, sites, rng)
    hyp = _model().hypothetical(seq)[0]
    hits = _hits(seq)
    return seq, hyp, hits, dict(zip(["AP-1", "GATA", "E-box", "GATA 2"], starts))


@cache
def _locus_ism() -> np.ndarray:
    seq = _locus()[0]
    return um.mutagenesis_effects(_model()(um.mutagenesis(seq)), seq)


def _around(site: str, before: int = 6, after: int = 6) -> tuple[int, int]:
    seq, _, _, starts = _locus()
    p = starts[site]
    w = _motifs()[site.split()[0]].width
    return _START + max(0, p - before), _START + min(len(seq), p + w + after)


def _attr_observed():
    seq, hyp, hits, _ = _locus()
    return um.view_attributions(hyp, seq, start=_START, hits=hits, figsize=(10, 2.6))


def _attr_all():
    seq, hyp, _, _ = _locus()
    return um.view_attributions(hyp, seq, kind="all", start=_START, region=_around("AP-1"),
                                ylabel="Hypothetical")


def _attr_heatmap():
    seq, _, hits, _ = _locus()
    return um.view_attributions(_locus_ism(), seq, kind="heatmap", start=_START,
                                region=_around("GATA", 10, 10), hits=hits, ylabel="Effect")


# Variants of the sequence.


def _variant(site: str, sign: int) -> tuple[int, str]:
    """The substitution in a site with the largest effect of the sign given."""
    starts = _locus()[3]
    ism = _locus_ism()
    p = starts[site]
    w = _motifs()[site.split()[0]].width
    part = sign * ism[:, p:p + w]
    letter, pos = np.unravel_index(np.argmax(part), part.shape)
    return p + int(pos), "ACGT"[letter]


def _variant_contributions():
    seq, hyp, _, _ = _locus()
    i, alt = _variant("AP-1", -1)
    alt_hyp = _model().hypothetical(seq[:i] + alt + seq[i + 1:])[0]
    return um.view_variant(hyp, alt_hyp, seq, position=_START + i, alt=alt, start=_START,
                           region=_around("AP-1", 8, 8))


def _variant_mutagenesis():
    seq = _locus()[0]
    i, alt = _variant("GATA 2", 1)
    alt_seq = seq[:i] + alt + seq[i + 1:]
    alt_ism = um.mutagenesis_effects(_model()(um.mutagenesis(alt_seq)), alt_seq)
    return um.view_variant(_locus_ism(), alt_ism, seq, position=_START + i, alt=alt, kind="all",
                           start=_START, region=_around("GATA 2", 8, 8))


# Tracks of one sequence.


def _smooth(x: np.ndarray, width: float) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


def _tracks_signal():
    seq, hyp, hits, _ = _locus()
    observed = (hyp * um.one_hot(seq, dtype=float)[0]).sum(axis=0)
    rng = np.random.default_rng(9)
    profile = _smooth(np.abs(observed), 6) * 120 + 4
    coverage = rng.poisson(profile * rng.uniform(0.7, 1.3)).astype(float)
    return um.view_attribution_tracks(
        {"Coverage": coverage, "Predicted": (profile, "#4C72B0"), "Contribution": hyp},
        seq, kind={"Coverage": "signal", "Predicted": "signal"}, start=_START, hits=hits,
        figsize=(10, 4.8))


def _tracks_kinds():
    seq, hyp, hits, _ = _locus()
    return um.view_attribution_tracks(
        {"Observed": hyp, "Hypothetical": hyp, "Mutagenesis": _locus_ism()}, seq,
        kind={"Hypothetical": "all", "Mutagenesis": "heatmap"}, start=_START,
        region=_around("E-box", 10, 10), hits=hits)


def _tracks_sharey():
    seq, _, hits, _ = _locus()
    tracks = {
        "Cell type 1": _Model(weights=[1.0, 0.15, 0.4]).hypothetical(seq)[0],
        "Cell type 2": _Model(weights=[0.15, 1.2, 0.4]).hypothetical(seq)[0],
    }
    return um.view_attribution_tracks(tracks, seq, start=_START, hits=hits, sharey=True, figsize=(10, 4))


# Several sequences with genomic ranges.


@cache
def _contribution_scores():
    rng = np.random.default_rng(31)
    bkg = um.create_sequences(seqnum=3, seqlen=120, seed=31)
    plans = [[("AP-1", 1), ("E-box", 7), ("GATA", 7)], [("GATA", 8), ("GATA", 10), ("AP-1", 12)],
             [("E-box", 22), ("AP-1", 9), ("GATA", 16)]]
    seqs, starts = [], []
    for s, plan in zip(bkg, plans):
        sites = [um.sample_sites(_motifs()[n], n=1, seed=k)[0] for n, k in plan]
        s, p = _plant(s, sites, rng)
        seqs.append(s)
        starts.append(p)
    genome = [1_520_340, 9_871_205, 33_040_118]
    ranges = pl.DataFrame({"chrom": ["chr2", "chr5", "chr11"], "start": genome,
                           "end": [g + 120 for g in genome]})
    scores = um.ContributionScores(_model().hypothetical(seqs), seqs, ["peak1", "peak2", "peak3"], ranges)
    return scores, starts


def _cs_observed():
    c, _ = _contribution_scores()
    return c.view(0, hits=_hits(c.sequences[0]), figsize=(10, 2.6))


def _cs_all():
    c, starts = _contribution_scores()
    p = int(c.ranges["start"][2]) + starts[2][0]
    return c.view(2, kind="all", region=(p - 8, p + 16), ylabel="Hypothetical")


# Variant effects along part of the sequence.


def _variant_part() -> tuple[str, int]:
    """The 50 letters of the sequence around its GATA and E-box sites, and
    the position of the first."""
    seq = _locus()[0]
    return seq[34:84], _START + 34


@cache
def _part_ism() -> np.ndarray:
    seq, _ = _variant_part()
    return um.mutagenesis_effects(_model()(um.mutagenesis(seq)), seq)


def _effects_all():
    seq, start = _variant_part()
    return um.plot_variant_effects(_part_ism(), sequence=seq, motifs=_motif_list(("GATA", "E-box")),
                                   hits=_hits(seq, names=("GATA", "E-box")), start=start, labels=5)


def _effects_strongest():
    seq, start = _variant_part()
    return um.plot_variant_effects(_part_ism(), sequence=seq, motifs=_motif_list(("GATA", "E-box")),
                                   hits=_hits(seq, names=("GATA", "E-box")), start=start,
                                   summary="strongest", colour_by="alt", labels=0)


def _effects_table():
    seq, _, hits, _ = _locus()
    rng = np.random.default_rng(17)
    pos = np.sort(rng.choice(len(seq), 14, replace=False))
    pos[[3, 9]] = [44, 73]
    ref = [seq[p] for p in pos]
    alt = [rng.choice([b for b in "ACGT" if b != r]) for r in ref]
    alts = [seq[:p] + a + seq[p + 1:] for p, a in zip(pos, alt)]
    model = _model()
    table = pl.DataFrame({
        "position": pos + _START, "REF": ref, "ALT": alt,
        "effect": model(alts) - model(seq)[0],
        "AF": np.round(rng.beta(0.6, 3, len(pos)), 3),
        "ID": [f"rs{n}" for n in rng.integers(10**6, 10**8, len(pos))],
    })
    return um.plot_variant_effects(table, sequence=seq, hits=hits, start=_START, size_by="AF",
                                   label_by="ID", labels=4)


def _effects_motifs():
    seq, start = _variant_part()
    motifs = _motif_list(("GATA", "E-box"))
    changes = um.variant_motif_effects(None, seq, motifs, start=start, pvalue=1e-3)
    strongest = (changes.sort(pl.col("delta").abs(), descending=True)
                 .unique("variant", keep="first", maintain_order=True))
    return um.plot_variant_effects(strongest.select("position", "ref", "alt", pl.col("delta").alias("effect")),
                                   sequence=seq, motifs=motifs, hits=_hits(seq, names=("GATA", "E-box")),
                                   start=start, pvalue=1e-3, ylabel="Change in motif score", labels=3)


# TF-MoDISco patterns and their seqlets, from contributions of the toy model
# around sites drawn from the motifs.


@cache
def _modisco_model() -> _Model:
    """The toy model with a GC-box filter of negative weight, as a repressor."""
    return _Model(("AP-1", "GATA", "E-box", "GC-box"), weights=[1.0, 1.6, 1.0, -0.6])


_PATTERNS = (
    ("pos_patterns.pattern_0", ("AP-1",), 420),
    ("pos_patterns.pattern_1", ("GATA",), 260),
    ("pos_patterns.pattern_2", ("E-box",), 130),
    ("pos_patterns.pattern_3", ("E-box", "GATA"), 70),
    ("neg_patterns.pattern_0", ("GC-box",), 90),
)


@cache
def _seqlets() -> pl.DataFrame:
    """Seqlets of each pattern as read_seqlets() gives them, with their
    hypothetical contributions read along the genome."""
    rng = np.random.default_rng(23)
    frames = []
    for k, (name, parts, n) in enumerate(_PATTERNS):
        drawn = [um.sample_sites(_motifs()[m], n=n, seed=100 + 10 * k + j) for j, m in enumerate(parts)]
        if len(parts) == 2:
            spacers = ["".join(rng.choice(list("ACGT"), 7)) for _ in range(n)]
            sites = [a + sp + b for a, sp, b in zip(drawn[0], spacers, drawn[1])]
        else:
            sites = drawn[0]
        width = len(sites[0]) + 6
        context = um.create_sequences(seqnum=n, seqlen=width + 30, seed=200 + k)
        # Most seqlets are centred on their site, and a few are off by a letter or two.
        shift = rng.choice([0, 0, 0, 0, 0, 0, 0, -1, 1, 2], n)
        seqs = [c[:18 + d] + site + c[18 + d + len(site):] for c, site, d in zip(context, sites, shift)]
        hyp = _modisco_model().hypothetical(seqs)[:, :, 15:15 + width]
        match = [s[15:15 + width] for s in seqs]
        minus = rng.random(n) < 0.5
        genome = np.where(minus[:, None, None], hyp[:, ::-1, ::-1], hyp)
        onehot = um.one_hot(match, dtype=float)
        frames.append(pl.DataFrame({
            "motif": [name] * n, "strand": np.where(minus, "-", "+"), "match": match,
            "score": (hyp * onehot).sum(axis=(1, 2)),
            # Patterns differ in width, so the scores are lists rather than arrays.
            "scores": pl.Series(genome.astype(np.float32)).cast(pl.List(pl.List(pl.Float32))),
        }))
    return pl.concat(frames)


@cache
def _patterns() -> dict[str, list[um.Motif]]:
    seqlets = _seqlets()
    out = {"CWM": [], "hCWM": [], "PPM": []}
    for name, _, n in _PATTERNS:
        rows = seqlets.filter(pl.col("motif") == name)
        hyp = np.array(rows["scores"].to_list(), dtype=float)
        minus = (rows["strand"] == "-").to_numpy()
        hyp[minus] = hyp[minus, ::-1, ::-1]
        onehot = um.one_hot(rows["match"].to_list(), dtype=float)
        for kind, matrix in (("CWM", (hyp * onehot).mean(axis=0)), ("hCWM", hyp.mean(axis=0))):
            out[kind].append(um.create_motif(matrix, alphabet="DNA", type="CWM", name=name, nsites=n))
        out["PPM"].append(um.create_motif(onehot.mean(axis=0), alphabet="DNA", type="PPM", name=name,
                                          nsites=n))
    return out


@cache
def _pattern_matches() -> pl.DataFrame:
    ppms = _patterns()["PPM"]
    return um.compare_motifs(ppms + um.example_motifs(), compare_to=list(range(len(ppms))), qvalue=0.5)


def _modisco_default():
    pats = _patterns()
    return um.plot_modisco({"CWM": pats["CWM"], "PPM": pats["PPM"]}, matches=_pattern_matches(),
                           database=um.example_motifs(), patterns=(3, 1))


def _modisco_aligned():
    names = {"pos_patterns.pattern_3": "E-box/GATA", "neg_patterns.pattern_0": "Repressor"}
    return um.plot_modisco(_patterns(), matches=_pattern_matches(), database=um.example_motifs(),
                           names=names, patterns=["E-box/GATA", "Repressor"], n_matches=2, align=True)


def _seqlets_gata():
    return um.plot_seqlets(_seqlets(), "pos_1", title="GATA (pos_1)")


def _seqlets_composite():
    return um.plot_seqlets(_seqlets(), "pos_3", max_seqlets=40, logo=False, title="E-box/GATA (pos_3)")


# Hits in many sequences and the contributions around them.


@cache
def _hit_set():
    rng = np.random.default_rng(57)
    n, length = 1500, 100
    bkg = um.create_sequences(seqnum=n, seqlen=length, seed=57)
    pool = {m: um.sample_sites(_motifs()[m], n=n, seed=60 + k) for k, m in enumerate(_NAMES)}
    seqs = []
    for i, s in enumerate(bkg):
        chosen = rng.choice(_NAMES, 2)
        seqs.append(_plant(s, [pool[m][i if j == 0 else n - 1 - i] for j, m in enumerate(chosen)], rng)[0])
    hits = _hits(seqs, pvalue=1e-3)
    return um.hit_contributions(hits, _model().hypothetical(seqs), seqs, flank=10, hypothetical=True)


def _profile_one():
    return um.plot_hit_profile(_hit_set(), motifs="GATA")


def _profile_all():
    return um.plot_hit_profile(_hit_set(), origin="centre", band="sd", max_hits=150)


def _contrib_default():
    return um.plot_hit_contributions(_hit_set())


def _contrib_pvalue():
    return um.plot_hit_contributions(_hit_set(), x="pvalue", motifs=["GATA", "E-box"], trend=True,
                                     marginals=False)


# Responses of a toy model to sites written into background sequences.


@cache
def _pair_model() -> _Model:
    return _Model(("GATA", "E-box"))


def _response(seqs) -> np.ndarray:
    """Two outputs of a toy model: GC content plus GATA and E-box sites,
    worth more in GC-rich sequences, and a bonus for the two together that
    follows the helix in the first output and falls off with distance in
    the second."""
    x = _onehot(seqs)
    model = _pair_model()
    z = model._z(x)
    gc = x[:, 1:3].sum(axis=(1, 2)) / x.shape[2]
    sites = model(x) * (0.5 + gc)
    best = z.reshape(len(x), 2, 2, -1).max(axis=1)
    strength = 1 / (1 + np.exp(-best.max(axis=2)))
    centre = best.argmax(axis=2) - model.f.shape[2] / 2
    d = np.abs(centre[:, 1] - centre[:, 0])
    both = strength[:, 0] * strength[:, 1] * 6
    helical = both * (1 + np.cos(2 * np.pi * d / 10.5)) * np.exp(-d / 60)
    close = both * 2 * np.exp(-d / 12)
    base = 4 * gc + 0.15 * sites
    return np.column_stack([base + helical, base + close])


@cache
def _background() -> list[str]:
    return um.create_sequences(seqnum=60, seqlen=100, seed=71)


@cache
def _spacing_predictions():
    bkg = _background()
    gata, ebox = _motif_list(("GATA", "E-box"))
    pairs = um.site_spacings(bkg, gata, ebox, spacings=range(41))
    return {d: _response(seqs) for d, seqs in pairs.items()}, _response(bkg)


def _spacing_helix():
    preds, base = _spacing_predictions()
    return um.plot_spacing({d: p[:, 0] for d, p in preds.items()}, baseline=base[:, 0], helix=True)


def _spacing_outputs():
    preds, base = _spacing_predictions()
    return um.plot_spacing(preds, baseline=base, names=["Cell type 1", "Cell type 2"], band="sd")


def _marg_single():
    bkg = _background()[:40]
    return um.plot_marginalization(_response(bkg)[:, 0], _response(um.insert_sites(bkg, _motifs()["GATA"]))[:, 0])


def _marg_several():
    bkg = _background()
    gata, ebox = _motif_list(("GATA", "E-box"))
    after = {
        "GATA": _response(um.insert_sites(bkg, gata)),
        "E-box": _response(um.insert_sites(bkg, ebox)),
        "Both": _response(um.insert_sites(bkg, [gata, ebox], positions=[38, 59])),
    }
    return um.plot_marginalization(_response(bkg), after, output="Cell type 1",
                                   names=["Cell type 1", "Cell type 2"])


def _marg_box():
    bkg = _background()
    after = {site: _response(um.insert_sites(bkg, site))[:, 0] for site in ("CAGATAAT", "TTGATAAG", "CAGAAAAG")}
    return um.plot_marginalization(_response(bkg)[:, 0], after, kind="box")


EXAMPLES = {
    "view_attributions": [
        Example("Gradient times input from a toy convolutional model, whose filters are the AP-1, "
                "GATA and E-box example motifs, along a sequence with four sites drawn from those "
                "motifs. Positions start at a genomic coordinate and the motif hits are shaded and "
                "named.", _attr_observed),
        Example('The hypothetical contributions of every letter around the AP-1 site with '
                'kind="all", zoomed in with region.', _attr_all),
        Example('In silico mutagenesis effects of the same toy model around a GATA site drawn with '
                'kind="heatmap", with the letters of the sequence marked by dots.', _attr_heatmap),
    ],
    "view_variant": [
        Example("The toy model's contributions around its AP-1 site before and after the "
                "substitution that weakens the site most, with the change at each position "
                "below.", _variant_contributions),
        Example('In silico mutagenesis before and after a substitution that strengthens a weak '
                'GATA site, drawn with kind="all" so that every letter\'s effect is shown.',
                _variant_mutagenesis),
    ],
    "view_attribution_tracks": [
        Example('Simulated coverage and a predicted profile drawn as kind="signal" tracks above the '
                "toy model's contributions, with each motif hit shaded through every track.",
                _tracks_signal),
        Example("The observed and hypothetical contributions around an E-box site above a heatmap "
                "of in silico mutagenesis, all from the same toy model.", _tracks_kinds),
        Example("Contributions from two toy models that weigh the AP-1 and GATA filters "
                "differently, standing in for two cell types, on one y-axis with sharey=True.",
                _tracks_sharey),
    ],
    "ContributionScores": [
        Example("The view() method draws the contributions of one of three toy sequences, with "
                "positions taken from its genomic range and motif hits passed on to "
                "view_attributions().", _cs_observed),
        Example('The hypothetical contributions of another sequence with kind="all", for a region '
                "given in genome positions.", _cs_all),
    ],
    "plot_variant_effects": [
        Example("The effect on a toy model's prediction of every substitution along 50 letters "
                "with a GATA and an E-box site, from mutagenesis() and mutagenesis_effects(). "
                "Variants that weaken a motif site point down and those inside a hit are ringed.",
                _effects_all),
        Example('The same substitutions with summary="strongest", keeping the one with the largest '
                'effect at each position, coloured by its alternative letter with colour_by="alt".',
                _effects_strongest),
        Example("Fourteen random variants in a table with REF, ALT, AF and ID columns, like one "
                "made from a VCF file, with effects from the toy model, heads sized by allele "
                "frequency and the four largest labelled by ID.", _effects_table),
        Example("The change in motif score from variant_motif_effects() as the effect, with no "
                "model at all, keeping the motif site each variant changes most.", _effects_motifs),
    ],
    "plot_modisco": [
        Example("Toy TF-MoDISco patterns built from sites drawn from example motifs and scored by "
                "a toy model, given as lists of CWMs and PPMs, with their best matches among the "
                "example motifs from compare_motifs(). patterns=(3, 1) keeps three positive "
                "patterns and one negative.", _modisco_default),
        Example("Two of the patterns, renamed with names and picked with patterns, with their "
                "hypothetical CWMs and two matches each placed in a shared frame with "
                "align=True.", _modisco_aligned),
    ],
    "plot_seqlets": [
        Example("The seqlets of a toy GATA pattern, with hypothetical contributions from a toy "
                "model, sorted by their total contribution. A few seqlets are shifted by a letter "
                "or two.", _seqlets_gata),
        Example("Forty seqlets of a toy composite pattern of an E-box and a GATA site, without the "
                "logos.", _seqlets_composite),
    ],
    "plot_hit_profile": [
        Example("The mean contribution in and around GATA hits found with scan_sequences() in 1,500 "
                "toy sequences, with contributions from a toy model and flanks of 10, the "
                "hypothetical logo above and every hit below.", _profile_one),
        Example('Hits of three motifs side by side, numbered from their centres with '
                'origin="centre" and with a band of one standard deviation.', _profile_all),
    ],
    "plot_hit_contributions": [
        Example("The total contribution of each motif hit in 1,500 toy sequences against its motif "
                "score, with contributions from a toy model whose filters are the same motifs.",
                _contrib_default),
        Example('The GATA and E-box hits against their P-values with x="pvalue", without the '
                "histograms and with a line through the median contribution of each motif along "
                "the axis.", _contrib_pvalue),
    ],
    "plot_spacing": [
        Example("The mean change in a toy model's prediction for 60 background sequences with a "
                "GATA and an E-box site written in by site_spacings() at spacings of 0 to 40, "
                "with helix=True. The model rewards site centres a whole number of helical turns "
                "apart.", _spacing_helix),
        Example('Both outputs of the toy model, the second of which rewards sites close together '
                'whatever their phase, with a band of one standard deviation.', _spacing_outputs),
    ],
    "plot_marginalization": [
        Example("A toy model's predictions for 40 background sequences before and after "
                "insert_sites() writes a GATA site into each, with lines joining each sequence.",
                _marg_single),
        Example("One output of the toy model for 60 sequences after inserting a GATA site, an "
                "E-box or both two helical turns apart.", _marg_several),
        Example('Three GATA sites of decreasing strength, each written into the same 60 sequences, '
                'compared with kind="box".', _marg_box),
    ],
}
