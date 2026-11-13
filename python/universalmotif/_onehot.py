"""One-hot arrays given where sequences are expected, and results returned
in the same form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deep import _to_numpy, from_one_hot, one_hot


@dataclass(frozen=True)
class _ArrayForm:
    """How one-hot sequences were given: letters on the last axis rather
    than the middle one, a single sequence without the first axis, the
    dtype, and whether a tensor."""

    letters_last: bool
    single: bool
    dtype: Any
    tensor: bool


def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or (hasattr(x, "detach") and hasattr(x, "shape"))


def _letters_middle(x: Any, alphabet: str) -> tuple[np.ndarray, _ArrayForm]:
    """One-hot sequences as an array of shape (sequences, letters, length),
    from either layout or a single sequence, and the form they came in."""
    n = len(alphabet)
    arr = _to_numpy(x, dtype=None)
    single = arr.ndim == 2
    a = arr[None] if single else arr
    if a.ndim != 3:
        raise ValueError(f"one-hot sequences must have shape (sequences, {n}, length) or "
                         f"(sequences, length, {n}), or ({n}, length) for one sequence, not {arr.shape}")
    last = a.shape[1] != n and a.shape[2] == n
    if last:
        a = a.transpose(0, 2, 1)
    if a.shape[1] != n:
        raise ValueError(f"one-hot sequences must have {n} letters, those of {alphabet}, not shape {arr.shape}")
    return a, _ArrayForm(last, single, arr.dtype, hasattr(x, "detach"))


def _check_one_hot(a: np.ndarray) -> None:
    if (np.count_nonzero(a, axis=1) > 1).any():
        raise ValueError("the sequences must be one-hot, with at most one letter at each position")


def _array_strings(x: Any, alphabet: str) -> tuple[list[str], _ArrayForm]:
    """One-hot sequences as strings, with N for positions without a letter."""
    a, form = _letters_middle(x, alphabet)
    _check_one_hot(a)
    return from_one_hot(a, alphabet=alphabet), form


def _restore_array(a: np.ndarray, form: _ArrayForm, lead: tuple[int, ...] = ()) -> Any:
    """An array of shape (..., letters, length) in the form the input came
    in, with ``lead`` axes, such as copies of each sequence, after the first."""
    if lead:
        a = a.reshape((a.shape[0] // int(np.prod(lead)),) + lead + a.shape[1:])
    if form.letters_last:
        a = np.ascontiguousarray(np.swapaxes(a, -1, -2))
    if form.single:
        a = a[0]
    if form.tensor:
        import torch

        return torch.from_numpy(np.ascontiguousarray(a))
    return a


def _strings_array(seqs: list[str], form: _ArrayForm, alphabet: str, lead: tuple[int, ...] = ()) -> Any:
    """Strings back as one-hot sequences in the form the input came in."""
    a = one_hot(seqs, alphabet=alphabet, dtype=form.dtype) if seqs else np.zeros((0, len(alphabet), 0), form.dtype)
    return _restore_array(a, form, lead)
