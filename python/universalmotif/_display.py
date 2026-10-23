"""Logos as SVG, the display of motifs in Jupyter, and export to logomaker."""

from __future__ import annotations

import html
from collections.abc import Mapping

import numpy as np

from . import _core
from ._api import _check_motif
from ._core import Motif
from ._logo import motif_glyphs
from ._plot import _MISSING_COLOUR, AA_COLOURS, DNA_COLOURS, RNA_COLOURS, _y_axis

# matplotlib's tab20, for the letters of custom alphabets.
_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]


def _colours(motif: Motif) -> dict[str, str]:
    known = {"DNA": DNA_COLOURS, "RNA": RNA_COLOURS, "AA": AA_COLOURS}.get(motif.alphabet)
    if known is not None:
        return known
    return {letter: _PALETTE[i % len(_PALETTE)] for i, letter in enumerate(motif.letters)}


def _logo_type(motif: Motif) -> str:
    return "CWM" if motif.type == "CWM" else "ICM"


def _heights(motif: Motif, use_type: str, relative_entropy: bool) -> np.ndarray:
    """Letter heights of a motif drawn as ``use_type``, letters by positions."""
    if use_type == "CWM":
        if motif.type != "CWM":
            raise ValueError('use_type="CWM" needs a motif that is a CWM')
        return np.asarray(motif.matrix, dtype=float)
    return np.array(_core._display_matrix(motif, use_type, relative_entropy), dtype=float).T


def logo_svg(
    motif: Motif,
    use_type: str | None = None,
    height: float = 60,
    position_width: float = 16,
    colours: Mapping[str, str] | None = None,
    relative_entropy: bool = False,
) -> str:
    """A sequence logo of a motif as an SVG image, without matplotlib.

    The letters are those [`view_motifs()`][universalmotif.view_motifs] draws,
    with heights from ``use_type``: information content by default, or
    contribution weights for a CWM. The logo is ``height`` pixels high and
    ``position_width`` pixels wide per position, with no axes, which suits
    reports and web pages. ``colours`` maps letters to colours, by default
    those of [`view_motifs()`][universalmotif.view_motifs]. The gaps of a
    gapped motif are drawn as dotted spaces labelled with their sizes.

    Examples
    --------
    >>> import universalmotif as um
    >>> svg = um.logo_svg(um.example_motifs()[3])
    >>> svg.startswith("<svg") and svg.count("<path") >= 8
    True
    """
    motif = _check_motif(motif)
    use_type = use_type or _logo_type(motif)
    matrix = _heights(motif, use_type, relative_entropy)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "the motif has non-finite values in this type; give it a pseudocount, for example "
            "with normalize(), first"
        )
    rows = motif.letters
    glyphs = motif_glyphs(matrix, rows, use_type)
    _, (lo, hi) = _y_axis([matrix], use_type)
    if hi <= lo:
        hi = lo + 1
    n = matrix.shape[1]
    gaps = list(zip(motif.gaploc, motif.mingap, motif.maxgap)) if motif.isgapped else []
    width = (n + len(gaps)) * position_width
    colours = _colours(motif) if colours is None else dict(colours)

    def px(x, y):
        # Each gap takes the room of one position after its column.
        shift = np.zeros_like(np.asarray(x, dtype=float))
        for loc, _, _ in gaps:
            shift = shift + (np.asarray(x) > loc + 0.5)
        return (x + shift - 0.5) * position_width, (hi - y) / (hi - lo) * height

    paths = []
    for loc, gmin, gmax in gaps:
        gx = (loc + sum(1 for other in gaps if other[0] < loc)) * position_width
        mid = height * 0.55
        paths.append(f'<line x1="{gx + 2:.2f}" y1="{mid:.2f}" x2="{gx + position_width - 2:.2f}" '
                     f'y2="{mid:.2f}" stroke="#7F7F7F" stroke-width="1.5" stroke-dasharray="1.5,2"/>')
        label = str(gmin) if gmin == gmax else f"{gmin}\u2013{gmax}"
        paths.append(f'<text x="{gx + position_width / 2:.2f}" y="{mid - 4:.2f}" font-size="9" '
                     f'text-anchor="middle" fill="#7F7F7F" font-family="sans-serif">{label}</text>')
    for g in glyphs:
        xs, ys = px(g.x, g.y)
        d = "M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in zip(xs, ys)) + "Z"
        fill = colours.get(g.group, _MISSING_COLOUR)
        paths.append(f'<path d="{d}" fill="{html.escape(fill)}"/>')
    if use_type in ("PWM", "CWM") and lo < 0 < hi:
        _, y0 = px(0.5, 0.0)
        paths.append(f'<line x1="0" y1="{y0:.2f}" x2="{width:.2f}" y2="{y0:.2f}" '
                     'stroke="#BFBFBF" stroke-width="0.5"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.2f} {height:.2f}" role="img" '
        f'aria-label="Sequence logo of {html.escape(motif.name)}">{"".join(paths)}</svg>'
    )


def _fmt(x: float) -> str:
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def _repr_html_(motif: Motif) -> str:
    """The display of a motif in Jupyter: a logo next to its metadata, with
    the matrix underneath for those who want it."""
    rows = [("Name", motif.name)]
    for label, value in (("Alternate name", motif.altname), ("Family", motif.family),
                         ("Organism", motif.organism)):
        if value:
            rows.append((label, value))
    rows += [("Alphabet", motif.alphabet), ("Type", motif.type)]
    if motif.alphabet in ("DNA", "RNA"):
        rows.append(("Strands", motif.strand))
    if not np.isnan(motif.icscore):
        rows.append(("Total IC", _fmt(round(motif.icscore, 2))))
    rows.append(("Pseudocount", _fmt(motif.pseudocount)))
    if motif.consensus:
        rows.append(("Consensus", f"<code>{html.escape(motif.consensus)}</code>"))
    for label, value in (("Target sites", motif.nsites), ("Background sites", motif.bkgsites),
                         ("P-value", motif.pval), ("Q-value", motif.qval), ("E-value", motif.eval)):
        if value is not None:
            rows.append((label, _fmt(value)))
    if motif.isgapped:
        gaps = ", ".join(f"after {loc}: {lo}" if lo == hi else f"after {loc}: {lo} to {hi}"
                         for loc, lo, hi in zip(motif.gaploc, motif.mingap, motif.maxgap))
        rows.append(("Gaps", gaps))
    for key, value in list(motif.extrainfo.items())[:5]:
        rows.append((key or "extra", value))
    table = "".join(
        f'<tr><th style="text-align:right;padding:1px 8px;font-weight:normal;opacity:0.7">'
        f"{html.escape(str(k))}</th><td style=\"text-align:left;padding:1px 8px\">"
        f"{v if k == 'Consensus' else html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    try:
        logo = logo_svg(motif, height=90, position_width=20)
    except Exception:
        logo = ""
    matrix = str(motif).split("\n\n", 1)[-1]
    return (
        '<div class="universalmotif-motif">'
        '<div style="display:flex;flex-wrap:wrap;gap:1em;align-items:center">'
        f"<div>{logo}</div><table style=\"border-collapse:collapse\">{table}</table></div>"
        f"<details><summary>Matrix</summary><pre>{html.escape(matrix)}</pre></details></div>"
    )


def to_logomaker(motif: Motif, use_type: str | None = None, relative_entropy: bool = False):
    """A motif as a pandas DataFrame for logomaker.

    logomaker draws logos from a DataFrame with one row per position, indexed
    from 0, and one column per letter. The values are the letter heights of
    ``use_type``, information content by default or contribution weights for
    a CWM, as [`view_motifs()`][universalmotif.view_motifs] draws them. Pass
    ``use_type="PPM"`` for probabilities. Needs the ``pandas`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> df = um.to_logomaker(um.example_motifs()[3])
    >>> list(df.columns), df.shape
    (['A', 'C', 'G', 'T'], (8, 4))
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "to_logomaker needs pandas; install it with `pip install universalmotif[pandas]`"
        ) from e
    motif = _check_motif(motif)
    use_type = use_type or _logo_type(motif)
    matrix = _heights(motif, use_type, relative_entropy).T
    df = pd.DataFrame(matrix, columns=motif.letters)
    df.index.name = "pos"
    return df
