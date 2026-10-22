"""Letter polygons for sequence logos.

Letters are drawn from outlines of the Roboto Medium font, stacked by
height within each position, with negative values stacked below zero.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np


@dataclass
class Glyph:
    """One letter's polygon in plot coordinates."""

    letter_id: str
    group: str
    x: np.ndarray
    y: np.ndarray


@cache
def _font() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    text = resources.files("universalmotif").joinpath("data", "roboto.json").read_text()
    return {k: (np.array(v["x"], dtype=float), np.array(v["y"], dtype=float)) for k, v in json.loads(text).items()}


def _letter(letter: str, height: float, fit_to_width: float, x_spacer: float, letter_id: str,
            centre: bool = True) -> list[Glyph]:
    font = _font()
    if letter not in font:
        warnings.warn(f'letter "{letter}" has no outline and is not drawn', stacklevel=4)
        return []
    gx, gy = font[letter]
    width = fit_to_width - x_spacer
    x = gx * width
    if centre:
        x = x + (1 - width) / 2
    return [Glyph(letter_id, letter, x, gy * height)]


def _multi_letter(letters: str, height: float, fit_to_width: float, x_spacer: float,
                  letter_id: str) -> list[Glyph]:
    n = len(letters)
    out: list[Glyph] = []
    for i, ch in enumerate(letters):
        for g in _letter(ch, height, fit_to_width / n, x_spacer / 10, f"{letter_id}.{i + 1}", centre=False):
            g.x = g.x + (fit_to_width / n) * i + (1 - fit_to_width) / 2
            out.append(g)
    for g in out:
        g.x = g.x * (fit_to_width - x_spacer) + (1 - fit_to_width + x_spacer) / 2
    return out


def _stable_sort(names: list[str], values: np.ndarray, decreasing: bool) -> tuple[list[str], np.ndarray]:
    order = np.argsort(-values if decreasing else values, kind="stable")
    return [names[i] for i in order], values[order]


def _flip_negative(glyphs: list[Glyph], sort_positions: bool, decreasing: bool, flip: bool) -> list[Glyph]:
    """Turn a stack of positive letters into one hanging below zero."""
    key = [re.sub(r"[.]\d+$", "", g.letter_id) for g in glyphs]
    lets = list(dict.fromkeys(key))
    for name in lets:
        members = [g for g, k in zip(glyphs, key) if k == name]
        ys = np.concatenate([g.y for g in members])
        for g in members:
            g.y = -(g.y - ys.min()) if flip else -(np.abs(g.y) - np.abs(ys).min())
    mins = np.array([min(g.y.min() for g, k in zip(glyphs, key) if k == name) for name in lets])
    if sort_positions:
        lets, mins = _stable_sort(lets, mins, decreasing)
    offsets = np.concatenate([[0.0], np.cumsum(mins)])
    for name, off in zip(lets, offsets):
        for g, k in zip(glyphs, key):
            if k == name:
                g.y = g.y + off
    return glyphs


def position_glyphs(
    names: list[str],
    values: np.ndarray,
    position_id: int,
    min_height: float = 0.03,
    x_spacer: float = 0.04,
    y_spacer: float = 0.01,
    sort_positions: bool = False,
    sort_positions_decreasing: bool = True,
    fit_to_height: float | None = None,
    fit_to_width: float = 1.0,
    flip_neg: bool = False,
) -> list[Glyph]:
    """The letters of one position, with x running from 0.5 to 1.5."""
    values = np.asarray(values, dtype=float)
    neg = values < 0
    any_neg = bool(neg.any() and (np.abs(values[neg]) >= 0.03).any())
    negout: list[Glyph] = []
    if any_neg:
        neg_names = [n for n, b in zip(names, neg) if b]
        negout = position_glyphs(neg_names, np.abs(values[neg]), position_id, min_height, x_spacer,
                                 y_spacer, False, not sort_positions_decreasing, None, fit_to_width)
        if negout:
            top = max(g.y.max() for g in negout)
            for g in negout:
                g.y = g.y - top
            negout = _flip_negative(negout, sort_positions, not sort_positions_decreasing, flip_neg)
        keep = values >= 0
        names = [n for n, b in zip(names, keep) if b]
        values = values[keep]
    else:
        values = np.where(values < 0, 0.0, values)

    if fit_to_height is None:
        fit_to_height = float(values.sum())
    keep = values >= min_height
    names = [n for n, b in zip(names, keep) if b]
    values = values[keep]
    if np.all(values == 0):
        return negout

    values = values / values.sum() * fit_to_height
    if sort_positions:
        names, values = _stable_sort(names, values, sort_positions_decreasing)

    n = len(values)
    spacer = y_spacer * ((n - 1) / n) if n > 1 else 0.0
    above = np.concatenate([np.cumsum(values[::-1])[::-1] + spacer / 2, [0.0]])
    out: list[Glyph] = []
    for i, (name, v) in enumerate(zip(names, values)):
        sp = spacer
        if (i == 0 or i == n - 1) and n > 1:
            sp -= spacer / 2
        sp = min(sp, v / 2)
        height = v - sp
        letter_id = f"{position_id}.{name}"
        if len(name) > 1:
            glyphs = _multi_letter(name, height, fit_to_width, x_spacer, letter_id)
        else:
            glyphs = _letter(name, height, fit_to_width, x_spacer, f"{letter_id}.1")
        for g in glyphs:
            g.x = g.x + 0.5
            g.y = g.y + above[i + 1]
        out.extend(glyphs)
    return out + negout


def matrix_glyphs(
    matrix: np.ndarray,
    rows: list[str],
    min_height: float = 0.03,
    x_spacer: float = 0.04,
    y_spacer: float = 0.01,
    sort_positions: bool = False,
    sort_positions_decreasing: bool = True,
    fit_to_height: float | None = None,
    fit_to_width: float = 1.0,
    flip_neg: bool = False,
) -> list[Glyph]:
    """Letters for every position of a (letters, positions) matrix, with
    position ``i``, counted from 1, centred on x = i."""
    out: list[Glyph] = []
    matrix = np.asarray(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        glyphs = position_glyphs(rows, matrix[:, j], j + 1, min_height, x_spacer, y_spacer,
                                 sort_positions, sort_positions_decreasing, fit_to_height,
                                 fit_to_width, flip_neg)
        for g in glyphs:
            g.x = g.x + j
        out.extend(glyphs)
    return out


def motif_glyphs(
    matrix: np.ndarray,
    rows: list[str],
    use_type: str,
    min_height: float = 0.03,
    x_spacer: float = 0.04,
    y_spacer: float = 0.01,
    sort_positions: bool = False,
    sort_positions_decreasing: bool = True,
    fit_to_height: float | None = None,
    flip_neg: bool = False,
) -> list[Glyph]:
    """Letters for a motif matrix of the given type, with the minimum height
    and vertical spacing scaled to the type's range."""
    m = np.array(matrix, dtype=float)
    nrow = m.shape[0]
    if use_type == "PPM":
        m[m > 1] = 1
        return matrix_glyphs(m, rows, min_height, x_spacer, y_spacer, sort_positions,
                             sort_positions_decreasing, fit_to_height, 1.0, flip_neg)
    if use_type == "PCM":
        top = m.sum(axis=0).max()
        return matrix_glyphs(m, rows, min_height * top, x_spacer, y_spacer * top, sort_positions,
                             sort_positions_decreasing, fit_to_height, 1.0, flip_neg)
    if use_type == "PWM":
        cap = np.log2(nrow)
        m[m > cap] = cap
        top = cap - m.min() if (m < 0).any() else cap
        return matrix_glyphs(m, rows, min_height * cap, x_spacer, y_spacer * top, sort_positions,
                             sort_positions_decreasing, None, 1.0, flip_neg)
    if use_type == "ICM":
        cap = np.log2(nrow)
        m[m > cap] = cap
        return matrix_glyphs(m, rows, min_height * cap, x_spacer, y_spacer * cap, sort_positions,
                             sort_positions_decreasing, None, 1.0, flip_neg)
    if use_type == "CWM":
        top = np.where(m > 0, m, 0).sum(axis=0).max() + abs(np.where(m < 0, m, 0).sum(axis=0).min())
        if top == 0:
            top = 1.0
        return matrix_glyphs(m, rows, min_height * top, x_spacer, y_spacer * top, sort_positions,
                             sort_positions_decreasing, None, 1.0, flip_neg)
    raise ValueError("use_type must be one of PCM, PPM, PWM, ICM and CWM")
