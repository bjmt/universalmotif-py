"""Motifs as convolution filters, sequences as one-hot arrays, and
TF-MoDISco results, for working with deep-learning models."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ._analysis import _sequence_pairs
from ._api import _check_motif, convert_type, create_motif
from ._core import Motif

_ALPHABETS = {"ACGT": "DNA", "ACGU": "RNA", "ACDEFGHIKLMNPQRSTVWY": "AA"}


def to_filters(
    motifs: Motif | Iterable[Motif],
    use_type: str = "PWM",
    align: str = "left",
    rc: bool = False,
) -> np.ndarray:
    """Motifs as convolution filters, an array of shape (motifs, letters,
    width).

    This is the layout of the weights of a one-dimensional convolution, such
    as PyTorch's ``Conv1d`` with the letters as input channels, so that
    ``torch.from_numpy(um.to_filters(motifs))`` can be set as a layer's
    weights, or used to scan one-hot sequences from
    [`one_hot()`][universalmotif.one_hot]. Filters hold the motifs as
    ``use_type``, log2 odds by default, and motifs narrower than the widest
    are padded with zeros, on the right with ``align="left"`` or on both
    sides with ``"centre"``. With ``rc``, each motif's reverse complement
    follows it, for DNA and RNA motifs. All motifs must share an alphabet.

    Examples
    --------
    >>> import universalmotif as um
    >>> filters = um.to_filters(um.example_motifs()[:3])
    >>> filters.shape, filters.dtype
    ((3, 4, 10), dtype('float32'))
    >>> bool((filters[2, :, 8:] == 0).all())
    True
    """
    ms = [motifs] if isinstance(motifs, Motif) else [_check_motif(m) for m in motifs]
    if not ms:
        raise ValueError("no motifs given")
    if len({m.alphabet for m in ms}) > 1:
        raise ValueError("all motifs must share an alphabet")
    if align not in ("left", "centre", "center"):
        raise ValueError('align must be "left" or "centre"')
    mats = [np.asarray(convert_type(m, use_type).matrix, dtype=float) for m in ms]
    if rc:
        if ms[0].alphabet not in ("DNA", "RNA"):
            raise ValueError("only DNA and RNA motifs have reverse complements")
        mats = [x for m in mats for x in (m, m[::-1, ::-1])]
    width = max(m.shape[1] for m in mats)
    out = np.zeros((len(mats), mats[0].shape[0], width), dtype=np.float32)
    for i, m in enumerate(mats):
        left = 0 if align == "left" else (width - m.shape[1]) // 2
        out[i, :, left:left + m.shape[1]] = m
    if not np.isfinite(out).all():
        raise ValueError(
            "the filters have infinite values; give the motifs a pseudocount, for example with "
            "normalize(), first"
        )
    return out


def from_filters(
    filters: Any,
    names: Sequence[str] | None = None,
    type: str = "PWM",
    alphabet: str = "DNA",
    bkg: Sequence[float] | None = None,
    trim: bool = True,
) -> list[Motif]:
    """Motifs from convolution filters or matrices from other tools.

    ``filters`` is an array of shape (motifs, letters, width), such as the
    weights of a convolution layer, or a mapping from names to (letters,
    width) matrices, as tangermeme and memelite keep motifs. Tensors are
    turned into numpy arrays first, so PyTorch tensors can be passed as they
    are, once on the CPU. ``type`` says what the values are, and ``alphabet``
    gives the letters in row order. With ``trim``, all-zero columns at
    either end, such as the padding of [`to_filters()`][universalmotif.to_filters],
    are removed.

    Examples
    --------
    >>> import universalmotif as um
    >>> motifs = um.example_motifs()[:3]
    >>> back = um.from_filters(um.to_filters(motifs), names=[m.name for m in motifs])
    >>> [(m.name, m.width, m.type) for m in back]
    [('TATA-box', 10, 'PWM'), ('AP-1', 9, 'PWM'), ('CRE', 8, 'PWM')]
    """
    if isinstance(filters, Mapping):
        names = [str(k) for k in filters]
        arrays = [_to_numpy(v) for v in filters.values()]
    else:
        stack = _to_numpy(filters)
        if stack.ndim == 2:
            stack = stack[None]
        if stack.ndim != 3:
            raise ValueError("filters must have shape (motifs, letters, width)")
        arrays = list(stack)
        names = list(names) if names is not None else [f"filter_{i + 1}" for i in range(len(arrays))]
    if len(names) != len(arrays):
        raise ValueError(f"{len(names)} names for {len(arrays)} filters")
    out = []
    for name, m in zip(names, arrays):
        m = np.asarray(m, dtype=float)
        if m.ndim != 2:
            raise ValueError(f"the matrix of {name!r} is not two-dimensional")
        if trim:
            nonzero = np.flatnonzero(np.abs(m).sum(axis=0) > 0)
            if nonzero.size:
                m = m[:, nonzero[0]:nonzero[-1] + 1]
        out.append(create_motif(m, alphabet=alphabet, type=type, name=name, bkg=bkg))
    return out


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=float)


def one_hot(
    sequences: str | Sequence[str] | Mapping[str, str],
    alphabet: str = "ACGT",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Sequences one-hot encoded, as an array of shape (sequences, letters,
    length).

    This is the layout most sequence models take, with one channel per
    letter of ``alphabet`` in the order given. Letters are matched in either
    case, and letters outside the alphabet, such as N, become columns of
    zeros. The sequences must all have the same length.

    Examples
    --------
    >>> import universalmotif as um
    >>> x = um.one_hot(["ACGTN", "TTTTT"])
    >>> x.shape
    (2, 4, 5)
    >>> x[0].astype(int).tolist()
    [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
    """
    _, seqs = _sequence_pairs(sequences)
    if not seqs:
        return np.zeros((0, len(alphabet), 0), dtype=dtype)
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError("the sequences must all have the same length")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("the alphabet has repeated letters")
    table = np.full(256, -1, dtype=np.int16)
    for i, letter in enumerate(alphabet):
        table[ord(letter.upper())] = i
        table[ord(letter.lower())] = i
    codes = table[np.frombuffer("".join(seqs).encode("latin-1"), dtype=np.uint8)]
    codes = codes.reshape(len(seqs), lengths.pop())
    out = np.zeros((codes.shape[0], len(alphabet), codes.shape[1]), dtype=dtype)
    n, pos = np.nonzero(codes >= 0)
    out[n, codes[n, pos], pos] = 1
    return out


def from_one_hot(x: Any, alphabet: str = "ACGT", unknown: str = "N") -> list[str]:
    """Sequences from one-hot arrays of shape (sequences, letters, length), as
    made by [`one_hot()`][universalmotif.one_hot]. Each position takes the
    letter of its largest value, so soft encodings work too, and positions
    with no positive value become ``unknown``.

    Examples
    --------
    >>> import universalmotif as um
    >>> um.from_one_hot(um.one_hot(["ACGTN", "TTTTT"]))
    ['ACGTN', 'TTTTT']
    """
    arr = _to_numpy(x)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1] != len(alphabet):
        raise ValueError(f"the array must have shape (sequences, {len(alphabet)}, length)")
    letters = np.array(list(alphabet) + [unknown])
    idx = np.where(arr.max(axis=1) > 0, arr.argmax(axis=1), len(alphabet))
    return ["".join(row) for row in letters[idx]]


def read_modisco(
    path: str | os.PathLike[str],
    type: str = "CWM",
    trim_threshold: float | None = 0.3,
    subpatterns: bool = False,
) -> list[Motif]:
    """Read the patterns of a TF-MoDISco results file, as written by
    modisco-lite's ``modisco motifs``.

    ``type`` picks which matrix of each pattern to read: ``"CWM"`` for its
    contribution scores, ``"hCWM"`` for its hypothetical contribution scores,
    also as a CWM, or ``"PPM"`` for the letter frequencies of its seqlets.
    Motifs are named after the pattern, such as ``pos_patterns.pattern_0``,
    with the number of seqlets as ``nsites``. Patterns span a fixed window
    with weak flanks, so by default they are trimmed as modisco's own reports
    do: the columns kept run from the first to the last whose absolute
    contribution reaches ``trim_threshold`` times the largest, and the other
    matrix types are trimmed to the same columns. With ``subpatterns``, the
    subpatterns of each pattern are read as well.

    Needs the ``h5py`` extra.

    Examples
    --------
    >>> import universalmotif as um
    >>> cwms = um.read_modisco("modisco_results.h5")  # doctest: +SKIP
    >>> ppms = um.read_modisco("modisco_results.h5", type="PPM")  # doctest: +SKIP
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "reading TF-MoDISco results needs h5py; install it with `pip install "
            "universalmotif[h5py]`"
        ) from e
    datasets = {"CWM": "contrib_scores", "hCWM": "hypothetical_contribs", "PPM": "sequence"}
    if type not in datasets:
        raise ValueError('type must be "CWM", "hCWM" or "PPM"')
    out = []
    with h5py.File(path, "r") as f:
        for group in ("pos_patterns", "neg_patterns"):
            if group not in f:
                continue
            for name in sorted(f[group], key=_pattern_order):
                pattern = f[group][name]
                out.append(_modisco_motif(pattern, f"{group}.{name}", datasets[type], type,
                                          trim_threshold))
                if subpatterns:
                    for sub in sorted((k for k in pattern if k.startswith("subpattern")),
                                      key=_pattern_order):
                        out.append(_modisco_motif(pattern[sub], f"{group}.{name}.{sub}",
                                                  datasets[type], type, trim_threshold))
    return out


def _pattern_order(name: str) -> tuple[str, int]:
    stem, _, num = name.rpartition("_")
    return (stem, int(num)) if num.isdigit() else (name, -1)


def _modisco_motif(pattern, name: str, dataset: str, kind: str,
                   trim_threshold: float | None) -> Motif:
    # Matrices are stored with one row per position.
    matrix = np.asarray(pattern[dataset][()], dtype=float).T
    if trim_threshold is not None:
        cwm = np.abs(np.asarray(pattern["contrib_scores"][()], dtype=float).T).sum(axis=0)
        keep = np.flatnonzero(cwm >= trim_threshold * cwm.max())
        if keep.size:
            matrix = matrix[:, keep[0]:keep[-1] + 1]
    nsites = None
    if "seqlets" in pattern and "n_seqlets" in pattern["seqlets"]:
        nsites = int(np.asarray(pattern["seqlets"]["n_seqlets"][()]).ravel()[0])
    return create_motif(
        matrix,
        alphabet="DNA",
        type="PPM" if kind == "PPM" else "CWM",
        name=name,
        nsites=nsites,
    )


__all__ = ["from_filters", "from_one_hot", "one_hot", "read_modisco", "to_filters"]
