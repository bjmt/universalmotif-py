"""Motifs as convolution filters, sequences as one-hot arrays, and
TF-MoDISco results, for working with deep-learning models."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core
from ._analysis import _sequence_pairs
from ._api import _check_motif, convert_type, create_motif
from ._core import Motif
from ._threads import _nthreads

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
        # The values are of the type given, so it is not guessed from them;
        # a PWM can have no negative values, and filters of zeros are PWMs.
        out.append(_core._create_from_matrix(np.ascontiguousarray(m), None, alphabet, matrix_type=type,
                                             type=type, name=name, bkg=bkg))
    return out


def _to_numpy(x: Any, dtype: Any = float) -> np.ndarray:
    """An array from a numpy array or tensor, as ``dtype``, or as it is with
    ``dtype=None``."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=dtype)


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
    # One channel at a time, which needs no index arrays.
    for i in range(len(alphabet)):
        out[:, i, :] = codes == i
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
    arr = _to_numpy(x, dtype=None)
    # Numbers are compared as they are; anything else is read as floats.
    if arr.dtype.kind not in "biuf":
        arr = arr.astype(float)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1] != len(alphabet):
        raise ValueError(f"the array must have shape (sequences, {len(alphabet)}, length)")
    idx = np.where(arr.max(axis=1) > 0, arr.argmax(axis=1), len(alphabet))
    symbols = list(alphabet) + [unknown]
    if all(len(s) == 1 and s.isascii() for s in symbols):
        # Each row's letters as bytes, decoded at once.
        rows = np.frombuffer("".join(symbols).encode("ascii"), dtype=np.uint8)[idx]
        return [row.tobytes().decode("ascii") for row in rows]
    letters = np.array(symbols)
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


_SEQLET_SCORES = {"projected": "contrib_scores", "hypothetical": "hypothetical_contribs"}
_SEQLET_DATASETS = ("example_idx", "start", "end", "is_revcomp", "sequence", "contrib_scores")


def read_seqlets(
    path: str | os.PathLike[str],
    patterns: str | Sequence[str] | None = None,
    regions: Any = None,
    window: int | None = None,
    scores: str | None = None,
):
    """Read the seqlets of a TF-MoDISco results file, the sites each
    pattern was built from, as a table of hits.

    The table has one row per seqlet and the columns of a
    [`scan_sequences()`][universalmotif.scan_sequences] result, so that it
    can be drawn with [`plot_hits()`][universalmotif.plot_hits] or tested
    for a peak in position with [`motif_peaks()`][universalmotif.motif_peaks].
    ``motif`` names the
    pattern as [`read_modisco()`][universalmotif.read_modisco] does and
    ``motif_i`` is its place in that list, ``sequence_i`` is the input
    sequence the seqlet was found in, counted from 0, and ``start`` and
    ``end`` are positions counted from 0 with the end left out. ``strand``
    is ``-`` for seqlets that match their pattern on the reverse strand,
    ``score`` is the total contribution of the seqlet's letters, and
    ``match`` holds its letters read along its strand, as in scan results,
    so that the seqlets of a pattern all read the same way. Rows come in
    the order of the patterns, then of the sequences, then of the starts,
    and ``patterns`` reads only the patterns with these names.

    Positions are as modisco gives them, counted from the start of the
    window it looked at in each input, and sequences are named ``seq1``,
    ``seq2`` and so on. Given the ``regions`` the inputs came from, as a BED
    file or a range table in the same order as the inputs, positions are
    moved to the genome instead, a ``chrom`` column is added and sequences
    are named ``chrom:start-end`` after their regions. modisco is often run
    on the central part of each input, such as 500 of 1,000 letters, and
    ``window`` gives its width so that the seqlets are placed within it;
    by default it spans the whole region. Regions on the ``-`` strand are
    taken to have been reverse complemented before going into the model.

    ``scores`` adds a ``scores`` column with the contribution scores of
    each seqlet as an array of shape (4, width), letters in the order ACGT,
    either the hypothetical scores of every letter (``"hypothetical"``) or
    those times the sequence (``"projected"``). They are turned to run
    along the genome, or along the input sequence without regions, so that
    a row's scores can go straight to
    [`view_attributions()`][universalmotif.view_attributions] with the
    sequence at the seqlet's position. When every seqlet has the same width
    the column holds fixed-size arrays, and ``df["scores"].to_numpy()``
    gives them all as one array of shape (seqlets, 4, width); otherwise it
    holds nested lists, and ``np.asarray(row["scores"])`` gives the array
    of one row either way.

    Needs the ``h5py`` extra.

    Examples
    --------
    >>> import numpy as np
    >>> import universalmotif as um
    >>> seqlets = um.read_seqlets("modisco_results.h5")  # doctest: +SKIP
    >>> seqlets = um.read_seqlets("modisco_results.h5", regions="peaks.bed", window=500,
    ...                           scores="hypothetical")  # doctest: +SKIP
    >>> row = seqlets.row(0, named=True)  # doctest: +SKIP
    >>> np.asarray(row["scores"]).shape  # doctest: +SKIP
    (4, 30)
    """
    import polars as pl

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "reading TF-MoDISco results needs h5py; install it with `pip install "
            "universalmotif[h5py]`"
        ) from e
    if scores is not None and scores not in _SEQLET_SCORES:
        raise ValueError('scores must be None, "projected" or "hypothetical"')
    if window is not None:
        if regions is None:
            raise ValueError("window places seqlets within the regions; give regions too")
        window = int(window)
        if window < 1:
            raise ValueError("window must be 1 or more")
    table = None if regions is None else _seqlet_regions(regions)
    wanted = None if patterns is None else [patterns] if isinstance(patterns, str) else list(patterns)

    with h5py.File(path, "r") as f:
        found = [(group, name) for group in ("pos_patterns", "neg_patterns") if group in f
                 for name in sorted(f[group], key=_pattern_order)]
        labels = [f"{group}.{name}" for group, name in found]
        if wanted is not None:
            unknown = [p for p in wanted if p not in labels]
            if unknown:
                shown = ", ".join(labels[:5]) + (", ..." if len(labels) > 5 else "")
                raise ValueError(f"no patterns named {', '.join(map(str, unknown))}; "
                                 f"the file has {shown or 'none'}")
        chosen = [i for i, label in enumerate(labels) if wanted is None or label in wanted]
        groups = []
        for i in chosen:
            seqlets = f[found[i][0]][found[i][1]].get("seqlets")
            need = _SEQLET_DATASETS + (("hypothetical_contribs",) if scores == "hypothetical" else ())
            missing = [k for k in need if seqlets is None or k not in seqlets]
            if missing:
                raise ValueError(f"{labels[i]} has no seqlets in {os.fspath(path)}; "
                                 f"it lacks {', '.join(missing)}")
            groups.append(seqlets)
        widths = [g["sequence"].shape[1] for g in groups]
        counts = [g["example_idx"].shape[0] for g in groups]
        total = sum(counts)
        same = len(set(widths)) == 1
        # Scores of one width are written straight into one array, which
        # polars then takes without a copy.
        values = np.empty((total, 4, widths[0]), dtype=np.float32) if scores is not None and same else None
        ragged = []
        cols: dict[str, list[np.ndarray]] = {k: [] for k in ("motif_i", "example", "start", "end", "rc",
                                                             "score")}
        letters = []
        reversed_input = None if table is None else (table["strand"] == "-").to_numpy()
        at = 0
        for i, g, n in zip(chosen, groups, counts):
            example = np.asarray(g["example_idx"][()], dtype=np.int64)
            if table is not None and n and int(example.max()) >= table.height:
                raise ValueError(f"seqlets come from input {int(example.max())}, counted from 0, but "
                                 f"there are only {table.height} regions; give the regions modisco "
                                 "was run on, in the same order")
            start = np.asarray(g["start"][()], dtype=np.int64)
            order = np.lexsort((start, example))
            rc = np.asarray(g["is_revcomp"][()], dtype=bool)[order]
            cols["motif_i"].append(np.full(n, i, dtype=np.uint32))
            cols["example"].append(example[order])
            cols["start"].append(start[order])
            cols["end"].append(np.asarray(g["end"][()], dtype=np.int64)[order])
            cols["rc"].append(rc)
            contrib = np.asarray(g["contrib_scores"][()], dtype=np.float32)[order]
            cols["score"].append(contrib.sum(axis=(1, 2), dtype=np.float64))
            letters.append(_seqlet_letters(np.asarray(g["sequence"][()])[order]))
            if scores is not None:
                x = contrib if scores == "projected" else np.asarray(g[_SEQLET_SCORES[scores]][()],
                                                                     dtype=np.float32)[order]
                del contrib
                flip = rc if reversed_input is None else rc ^ reversed_input[example[order]]
                x[flip] = x[flip, ::-1, ::-1]
                x = x.transpose(0, 2, 1)
                if values is not None:
                    values[at:at + n] = x
                else:
                    ragged.append(pl.Series("scores", np.ascontiguousarray(x)).cast(
                        pl.List(pl.List(pl.Float32))))
            at += n

    def joined(key, dtype):
        return np.concatenate(cols[key]) if cols[key] else np.zeros(0, dtype=dtype)

    example = joined("example", np.int64)
    start, end, rc = joined("start", np.int64), joined("end", np.int64), joined("rc", bool)
    out = {
        "motif": pl.Series([labels[i] for i in chosen], dtype=pl.String).gather(
            np.repeat(np.arange(len(chosen)), counts)),
        "motif_i": pl.Series(joined("motif_i", np.uint32), dtype=pl.UInt32),
    }
    if table is None:
        out["sequence"] = "seq" + pl.Series(example + 1, dtype=pl.Int64).cast(pl.String)
        out["sequence_i"] = pl.Series(example, dtype=pl.UInt32)
        strand = rc
    else:
        reg = table[example]
        rs, re_ = reg["start"].to_numpy(), reg["end"].to_numpy()
        minus = (reg["strand"] == "-").to_numpy()
        width = re_ - rs
        win = width if window is None else np.full(total, window)
        if (win > width).any():
            raise ValueError(f"window is {window} wide but some regions are only {int(width.min())}")
        if (end > win).any():
            raise ValueError(f"seqlets run up to {int(end.max())}, past the window of {int(win.min())}; "
                             "check window")
        offset = width // 2 - win // 2
        span = end - start
        start = np.where(minus, re_ - offset - end, rs + offset + start)
        end = start + span
        out["sequence"] = reg.select(pl.format("{}:{}-{}", "chrom", "start", "end").alias("name"))["name"]
        out["sequence_i"] = pl.Series(example, dtype=pl.UInt32)
        out["chrom"] = reg["chrom"]
        strand = rc ^ minus
    out["start"] = pl.Series(start, dtype=pl.Int64)
    out["end"] = pl.Series(end, dtype=pl.Int64)
    out["strand"] = pl.Series(np.where(strand, "-", "+"), dtype=pl.String)
    out["score"] = pl.Series(joined("score", np.float64), dtype=pl.Float64)
    out["match"] = pl.concat(letters) if letters else pl.Series([], dtype=pl.String)
    if scores is not None:
        if values is not None:
            out["scores"] = pl.Series(values)
        else:
            out["scores"] = (pl.concat(ragged) if ragged
                             else pl.Series([], dtype=pl.List(pl.List(pl.Float32))))
    return pl.DataFrame(out)


def _seqlet_letters(onehot: np.ndarray) -> Any:
    """The letters of one-hot seqlets of shape (seqlets, width, 4) as a
    polars string column, with N where no letter is set."""
    import polars as pl

    n, w = onehot.shape[:2]
    if w == 0:
        return pl.Series([""] * n, dtype=pl.String)
    idx = np.where(onehot.max(axis=2) > 0, onehot.argmax(axis=2), 4)
    rows = np.ascontiguousarray(np.frombuffer(b"ACGTN", dtype=np.uint8)[idx]).view(f"S{w}").ravel()
    return pl.Series(rows, dtype=pl.Binary).cast(pl.String)


def _seqlet_regions(regions: Any):
    """The regions modisco's inputs came from, as a range table, from a BED
    file or anything a range table is made from."""
    import polars as pl

    from ._genomic import _range_table

    if isinstance(regions, (str, os.PathLike)):
        # Every column is read as text and picked by position, since the
        # names polars gives headerless columns differ between versions.
        bed = pl.read_csv(regions, separator="\t", has_header=False, infer_schema_length=0,
                          comment_prefix="#")
        if bed.width < 3:
            raise ValueError(f"{os.fspath(regions)} needs at least the three columns of a BED file")
        cols = {"chrom": pl.nth(0), "start": pl.nth(1), "end": pl.nth(2)}
        if bed.width >= 6:
            cols["strand"] = pl.nth(5)
        regions = bed.select(**cols)
    table = _range_table(regions, "regions")
    if (table["end"] <= table["start"]).any():
        raise ValueError("regions must have start < end")
    return table


@dataclass(frozen=True, eq=False)
class ContributionScores:
    """Contribution scores of a set of sequences, from
    [`read_contributions()`][universalmotif.read_contributions].

    ``scores`` is an array of shape (sequences, letters, length) holding a
    score for every letter at every position, ``sequences`` the sequences
    themselves, ``names`` their names, and ``ranges`` a DataFrame of the
    genomic range of each sequence, with positions counted from 0 and the
    end left out, when the file gives them. Indexing gives the scores and
    sequence of one sequence, and [`view()`][universalmotif.ContributionScores.view]
    draws them.

    Examples
    --------
    >>> import numpy as np
    >>> import universalmotif as um
    >>> c = um.ContributionScores(np.zeros((2, 4, 5)), ["ACGTA", "TTGAC"], ["a", "b"], None)
    >>> c
    ContributionScores(2 sequences of 5 letters)
    >>> scores, seq = c[1]
    >>> seq
    'TTGAC'
    """

    scores: np.ndarray
    sequences: list[str]
    names: list[str]
    ranges: Any

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> tuple[np.ndarray, str]:
        return self.scores[i], self.sequences[i]

    def __repr__(self) -> str:
        return f"ContributionScores({len(self)} sequences of {self.scores.shape[-1]} letters)"

    def view(self, i: int, kind: str = "observed", **kwargs: Any):
        """Draw the scores of sequence ``i`` with
        [`view_attributions()`][universalmotif.view_attributions], with
        positions on the genome when the ranges are known, so that
        ``region`` is given in genome positions too. Other arguments are
        passed on."""
        from ._plot import view_attributions

        if self.ranges is not None and "start" not in kwargs:
            kwargs["start"] = int(self.ranges["start"][i])
        return view_attributions(self.scores[i], self.sequences[i], kind=kind, **kwargs)


def _letters_first(x: np.ndarray, n: int, what: str) -> np.ndarray:
    """An array of shape (sequences, letters, length), from that or from
    (sequences, length, letters)."""
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"{what} must have shape (sequences, {n}, length)")
    if x.shape[1] != n and x.shape[2] == n:
        x = x.transpose(0, 2, 1)
    if x.shape[1] != n:
        raise ValueError(f"{what} must have shape (sequences, {n}, length), not {x.shape}")
    return x


def _load_array(source: Any, what: str) -> np.ndarray:
    """An array as it is stored, from memory or a file. ``.npy`` files are
    memory-mapped, so that only the parts used are read."""
    if not isinstance(source, (str, os.PathLike)):
        return _to_numpy(source, dtype=None)
    data = np.load(source, mmap_mode="r")
    if isinstance(data, np.ndarray):
        return data
    keys = list(data.keys())
    if len(keys) != 1 and "arr_0" not in keys:
        raise ValueError(f"{source} holds several arrays ({', '.join(keys)}); pass the {what} itself")
    return data[keys[0] if len(keys) == 1 else "arr_0"]


def _chrombpnet_regions(path: str, regions: Any, length: int):
    """The genomic window of each sequence in a chromBPNet scores file: the
    peaks in ``<prefix>.interpreted_regions.bed``, or given as ``regions``,
    each a window of ``length`` centred on its summit."""
    import polars as pl

    if regions is None:
        prefix = path
        for suffix in (".counts_scores.h5", ".profile_scores.h5"):
            if path.endswith(suffix):
                prefix = path[: -len(suffix)]
        bed = prefix + ".interpreted_regions.bed"
        if not os.path.exists(bed):
            return None
        regions = bed
    if isinstance(regions, (str, os.PathLike)):
        # Every column is read as text and picked by position, since the
        # names polars gives headerless columns differ between versions.
        regions = pl.read_csv(regions, separator="\t", has_header=False, infer_schema_length=0)
        if regions.width < 10:
            raise ValueError("the regions need the ten columns of a narrowPeak file")
        regions = regions.select(chrom=pl.nth(0), start=pl.nth(1), summit=pl.nth(9))
    centre = regions["start"].cast(pl.Int64) + regions["summit"].cast(pl.Int64)
    return pl.DataFrame({
        "chrom": regions["chrom"].cast(pl.String),
        "start": centre - length // 2,
        "end": centre - length // 2 + length,
    })


def read_contributions(
    path: str | os.PathLike[str],
    sequences: Any = None,
    scores: str = "hypothetical",
    regions: Any = None,
    indices: Sequence[int] | None = None,
    alphabet: str = "ACGT",
) -> ContributionScores:
    """Read the contribution scores a sequence model gives its inputs.

    Two layouts are read:

    - The HDF5 files chromBPNet writes, ``<prefix>.counts_scores.h5`` and
      ``<prefix>.profile_scores.h5``, which hold the one-hot sequences and
      their scores. ``scores`` picks the hypothetical scores of every letter
      at every position (``"hypothetical"``), or those times the one-hot
      sequences (``"projected"``). The genomic range of each sequence comes
      from ``<prefix>.interpreted_regions.bed`` next to the file, or from
      ``regions``, a narrowPeak file or a table with ``chrom``, ``start``
      and ``summit`` columns.
    - numpy ``.npy`` or ``.npz`` files of scores, as TF-MoDISco takes, with
      the one-hot sequences given as ``sequences``, a file or an array, or
      the sequences themselves as strings.

    Arrays may have shape (sequences, letters, length) or (sequences,
    length, letters), with the letters in the order of ``alphabet``.
    Scores are read as 32-bit floats, and ``indices`` reads just some of the
    sequences of a large HDF5 or ``.npy`` file, which is read a part at a
    time; a ``.npz`` file is compressed, so it is read whole.

    Needs the ``h5py`` extra for HDF5 files.

    Examples
    --------
    >>> import universalmotif as um
    >>> c = um.read_contributions("model.counts_scores.h5")  # doctest: +SKIP
    >>> c.view(0, kind="all")  # doctest: +SKIP
    >>> c = um.read_contributions("shap.npz", sequences="ohe.npz")  # doctest: +SKIP
    """
    if scores not in ("hypothetical", "projected"):
        raise ValueError('scores must be "hypothetical" or "projected"')
    path = os.fspath(path)
    import warnings
    n = len(alphabet)
    pick = None if indices is None else [int(i) for i in indices]
    ranges = None
    if path.endswith((".h5", ".hdf5")):
        if sequences is not None:
            warnings.warn("sequences is ignored for HDF5 files, which hold the sequences", stacklevel=2)
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "reading HDF5 contribution scores needs h5py; install it with `pip install "
                "universalmotif[h5py]`"
            ) from e
        with h5py.File(path, "r") as f:
            if not ("raw" in f and "shap" in f):
                raise ValueError(f"{path} is not a chromBPNet scores file; it holds {', '.join(f)}")
            key = "shap" if scores == "hypothetical" else "projected_shap"

            def read(name):
                ds = f[name]["seq"]
                if pick is None:
                    return ds[()]
                order = np.argsort(pick)
                got = ds[np.asarray(pick)[order]]
                out = np.empty_like(got)
                out[order] = got
                return out

            if pick is not None and len(set(pick)) != len(pick):
                raise ValueError("indices must not repeat")
            onehot = _letters_first(np.asarray(read("raw")), n, "the sequences")
            values = _letters_first(np.asarray(read(key), dtype=np.float32), n, "the scores")
        ranges = _chrombpnet_regions(path, regions, values.shape[-1])
        if ranges is not None and pick is not None:
            ranges = ranges[pick]
    else:
        if regions is not None:
            warnings.warn("regions is only used with chromBPNet HDF5 files and is ignored", stacklevel=2)
        if sequences is None:
            raise ValueError("give the sequences, or a file of them one-hot encoded, with sequences")

        # Sequences are picked before the scores are converted, so that only
        # those are read from memory-mapped files.
        def picked(x):
            x = x[None] if x.ndim == 2 else x
            return x if pick is None else x[pick]

        values = _letters_first(np.asarray(picked(_load_array(path, "scores")), dtype=np.float32),
                                n, "the scores")
        if isinstance(sequences, list) and sequences and isinstance(sequences[0], str):
            onehot = one_hot(sequences, alphabet=alphabet, dtype=np.int8)
            if pick is not None:
                onehot = onehot[pick]
        else:
            onehot = _letters_first(np.asarray(picked(_load_array(sequences, "sequences"))),
                                    n, "the sequences")
        if scores == "projected":
            values = values * onehot
    if onehot.shape[0] != values.shape[0] or onehot.shape[-1] != values.shape[-1]:
        raise ValueError(f"{onehot.shape[0]} sequences of length {onehot.shape[-1]} for "
                         f"{values.shape[0]} of length {values.shape[-1]} in the scores")
    seqs = from_one_hot(onehot, alphabet=alphabet)
    if ranges is not None:
        if ranges.height != len(seqs):
            raise ValueError(f"{ranges.height} regions for {len(seqs)} sequences")
        names = [f"{c}:{s}-{e}" for c, s, e in ranges.iter_rows()]
    else:
        names = [f"seq{i + 1}" for i in (range(len(seqs)) if pick is None else pick)]
    return ContributionScores(values, seqs, names, ranges)


def _substitutions(ref: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and letter indices of every single-letter substitution of a
    one-hot sequence between ``start`` and ``end``, in order."""
    pos, let = np.nonzero(ref[:, start:end].T == 0)
    return pos + start, let


def _mutagenesis_region(sequence: str, region: tuple[int, int] | None) -> tuple[int, int]:
    if not isinstance(sequence, str):
        raise TypeError("give a single sequence as a string")
    a, b = (0, len(sequence)) if region is None else (int(region[0]), int(region[1]))
    if not 0 <= a < b <= len(sequence):
        raise ValueError(f"region must lie within 0 to {len(sequence)}")
    return a, b


def mutagenesis(
    sequence: str,
    alphabet: str = "ACGT",
    region: tuple[int, int] | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Every single-letter substitution of a sequence, one-hot encoded for a
    model.

    Returns an array of shape (variants, letters, length) whose first entry
    is the sequence itself. It is followed, for each position of ``region``
    in turn, by the sequence with each other letter of ``alphabet`` at that
    position, in alphabet order. ``region`` is a (start, end) pair counted
    from 0 with the end left out, and is the whole sequence by default.
    Positions holding a letter outside the alphabet, such as N, get every
    letter. A model's predictions for the batch, in the same order, are
    turned into the effect of each substitution by
    [`mutagenesis_effects()`][universalmotif.mutagenesis_effects]. The batch
    has three variants per position of a DNA sequence, so for long
    sequences, ``region`` keeps it to the part of interest.

    Examples
    --------
    >>> import universalmotif as um
    >>> batch = um.mutagenesis("ACGT")
    >>> batch.shape
    (13, 4, 4)
    >>> um.from_one_hot(batch[:4])
    ['ACGT', 'CCGT', 'GCGT', 'TCGT']
    """
    a, b = _mutagenesis_region(sequence, region)
    ref = one_hot(sequence, alphabet=alphabet, dtype=dtype)
    pos, let = _substitutions(ref[0], a, b)
    out = np.repeat(ref, len(pos) + 1, axis=0)
    rows = np.arange(1, len(pos) + 1)
    out[rows, :, pos] = 0
    out[rows, let, pos] = 1
    return out


def mutagenesis_effects(
    predictions: Any,
    sequence: str,
    alphabet: str = "ACGT",
    region: tuple[int, int] | None = None,
    centre: bool = False,
) -> np.ndarray:
    """The effect of each single-letter substitution, from a model's
    predictions for the batch made by
    [`mutagenesis()`][universalmotif.mutagenesis].

    ``predictions`` holds one prediction per entry of the batch, in order,
    with the sequence's own first; ``sequence``, ``alphabet`` and ``region``
    must be those the batch was made with. The effect of a substitution is
    its prediction minus the sequence's, and the letters of the sequence
    get 0. Returns an array of shape (letters, positions of ``region``), or
    with several outputs per prediction, given as an array of shape
    (variants, outputs), one of shape (outputs, letters, positions). With
    ``centre``, the mean effect at each position is subtracted, as is common
    before drawing the effects as attributions with
    [`view_attributions()`][universalmotif.view_attributions].

    Examples
    --------
    >>> import numpy as np
    >>> import universalmotif as um
    >>> batch = um.mutagenesis("ACGT")
    >>> preds = batch[:, 0, :].sum(axis=1)  # a model that counts As
    >>> um.mutagenesis_effects(preds, "ACGT").astype(int)
    array([[ 0,  1,  1,  1],
           [-1,  0,  0,  0],
           [-1,  0,  0,  0],
           [-1,  0,  0,  0]])
    """
    a, b = _mutagenesis_region(sequence, region)
    ref = one_hot(sequence, alphabet=alphabet)
    pos, let = _substitutions(ref[0], a, b)
    pred = _to_numpy(predictions)
    if pred.ndim == 0 or pred.shape[0] != len(pos) + 1:
        raise ValueError(f"expected {len(pos) + 1} predictions, one for the sequence and one per "
                         f"substitution, not {pred.shape[0] if pred.ndim else 0}")
    diff = pred[1:] - pred[0]
    out = np.zeros((len(alphabet), b - a) + pred.shape[1:])
    out[let, pos - a] = diff
    if centre:
        out = out - out.mean(axis=0, keepdims=True)
    if pred.ndim > 1:
        out = np.moveaxis(out, (0, 1), (-2, -1))
    return out


def _site_letters(site: Any) -> str:
    """A site as letters: a string as it is, or a motif's most probable
    letter at each position."""
    if isinstance(site, Motif):
        ppm = convert_type(site, "PPM").matrix
        letters = np.array(list(site.letters))
        return "".join(letters[np.argmax(ppm, axis=0)])
    if not isinstance(site, str) or not site:
        raise TypeError("a site must be a non-empty string or a motif")
    return site


def insert_sites(
    sequences: str | Sequence[str] | Mapping[str, str],
    sites: Any,
    positions: int | Sequence[int] | None = None,
):
    """Write the same sites into every sequence, for measuring a model's
    response to them.

    ``sites`` is a site given as a string, a motif, whose most probable
    letter at each position is used, or a list of these to insert together.
    ``positions`` gives the start of each site, counted from 0; by default a
    single site is centred in each sequence, rounding down. Sites overwrite
    the letters where they land, later ones over earlier ones, and must fit
    within every sequence.

    Unlike [`implant_motifs()`][universalmotif.implant_motifs], which draws a
    new site from a motif for every placement, the same letters go into
    every sequence, as when comparing a model's predictions for background
    sequences, such as from
    [`shuffle_sequences()`][universalmotif.shuffle_sequences], with and
    without a motif. Returns the new sequences in the same form as the
    input.

    Examples
    --------
    >>> import universalmotif as um
    >>> bkg = ["AAAAAAAAAA", "CCCCCCCCCC"]
    >>> um.insert_sites(bkg, "TGACTCA")
    ['ATGACTCAAA', 'CTGACTCACC']
    >>> um.insert_sites(bkg, ["GG", "TT"], positions=[0, 8])
    ['GGAAAAAATT', 'GGCCCCCCTT']
    """
    names, seqs = _sequence_pairs(sequences)
    several = isinstance(sites, (list, tuple))
    letters = [_site_letters(x) for x in sites] if several else [_site_letters(sites)]
    if positions is None:
        if len(letters) > 1:
            raise ValueError("give the positions of several sites")
        starts = None
    else:
        starts = [int(positions)] if np.ndim(positions) == 0 else [int(p) for p in positions]
        if len(starts) != len(letters):
            raise ValueError(f"{len(starts)} positions for {len(letters)} sites")
    out = []
    for seq in seqs:
        new = list(seq)
        for k, site in enumerate(letters):
            p = (len(seq) - len(site)) // 2 if starts is None else starts[k]
            if p < 0 or p + len(site) > len(seq):
                raise ValueError(f"the site {site} at {p} does not fit in a sequence of length {len(seq)}")
            new[p:p + len(site)] = site
        out.append("".join(new))
    from ._sequences import _restore

    return _restore(sequences, names, out)


def site_spacings(
    sequences: str | Sequence[str] | Mapping[str, str],
    first: Any,
    second: Any,
    spacings: Iterable[int],
    start: int | None = None,
) -> dict[int, Any]:
    """Two sites at a range of spacings, for measuring how a model responds
    to the distance between them.

    For each spacing, ``first`` is written into every sequence at ``start``
    and ``second`` begins that many letters after ``first`` ends. Sites are
    given as for [`insert_sites()`][universalmotif.insert_sites]. By default
    ``start`` places the pair in the middle of each sequence at the widest
    spacing, so that ``first`` stays put as ``second`` moves. Returns a dict
    from each spacing to the new sequences, in the same form as the input.

    Examples
    --------
    >>> import universalmotif as um
    >>> res = um.site_spacings("AAAAAAAAAAAA", "GG", "TT", spacings=[0, 2, 4])
    >>> res
    {0: 'AAGGTTAAAAAA', 2: 'AAGGAATTAAAA', 4: 'AAGGAAAATTAA'}
    """
    spacings = [int(d) for d in spacings]
    if not spacings or min(spacings) < 0:
        raise ValueError("spacings must be one or more whole numbers, 0 or greater")
    a, b = _site_letters(first), _site_letters(second)
    names, seqs = _sequence_pairs(sequences)
    out = {}
    for d in spacings:
        new = []
        for seq in seqs:
            s0 = (len(seq) - len(a) - max(spacings) - len(b)) // 2 if start is None else int(start)
            new.append(insert_sites(seq, [a, b], positions=[s0, s0 + len(a) + d]))
        out[d] = new
    from ._sequences import _restore

    return {d: _restore(sequences, names, v) for d, v in out.items()}


class _OneHotBatches:
    """Batches of one-hot sequences read from a genome as they are needed."""

    def __init__(self, genome, table, batch_size, alphabet, dtype, nthreads):
        self._genome, self._table = genome, table
        self._batch_size, self._alphabet, self._dtype, self._nthreads = batch_size, alphabet, dtype, nthreads

    def __len__(self) -> int:
        return (self._table.height + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        from ._genome import _fetch_ranges

        for i in range(0, self._table.height, self._batch_size):
            seqs = _fetch_ranges(self._genome, self._table.slice(i, self._batch_size), self._nthreads)
            yield one_hot(seqs, alphabet=self._alphabet, dtype=self._dtype)

    def __repr__(self) -> str:
        return f"<{len(self)} batches of up to {self._batch_size} sequences>"


def one_hot_batches(
    genome: Any,
    ranges: Any,
    batch_size: int = 256,
    width: int | None = None,
    alphabet: str = "ACGT",
    dtype: Any = np.float32,
    nthreads: int | None = None,
):
    """Genomic ranges as one-hot batches for a model, read a batch at a time.

    ``genome`` and ``ranges`` are as for
    [`extract_ranges()`][universalmotif.extract_ranges]: a
    [`Genome`][universalmotif.Genome] read from disk,
    [`PackedSequences`][universalmotif.PackedSequences] or a mapping from
    chromosome names to sequences, and a DataFrame or tuples of ranges
    counted from 0 with the end left out. Ranges on the ``-`` strand are
    reverse complemented. Models take sequences of one length, so the ranges
    must all have the same width, or with ``width`` each is replaced by a
    window of that width around its middle, the lower one for even widths.
    Every range is checked before any is read, and must fit within its
    chromosome.

    Returns an iterable of arrays of shape (sequences, letters, width), up
    to ``batch_size`` sequences each and in the order of the ranges, with a
    length so that progress can be shown. Only one batch is held in memory
    at a time, so large sets of regions can be fed to a model, for example
    as ``model(torch.from_numpy(x))`` for each batch ``x``.

    ``nthreads`` defaults to the package setting from
    [`set_nthreads()`][universalmotif.set_nthreads].

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = {"chr1": "AACCGGTTACGTACGTAAAA"}
    >>> batches = um.one_hot_batches(genome, [("chr1", 0, 9), ("chr1", 4, 20, "-")],
    ...                              width=4, batch_size=1)
    >>> len(batches)
    2
    >>> [um.from_one_hot(x) for x in batches]
    [['CCGG'], ['GTAC']]
    """
    nthreads = _nthreads(nthreads)
    from ._genome import _check_ranges
    from ._genomic import _range_table

    if int(batch_size) < 1:
        raise ValueError("batch_size must be 1 or more")
    table = _range_table(ranges, "ranges")
    if width is not None:
        width = int(width)
        if width < 1:
            raise ValueError("width must be 1 or more")
        first = (table["start"] + table["end"]) // 2 - width // 2
        table = table.with_columns(start=first, end=first + width)
    elif table.height and (table["end"] - table["start"]).n_unique() > 1:
        raise ValueError("the ranges have different widths; give width to take windows of one width")
    _check_ranges(genome, table)
    return _OneHotBatches(genome, table, int(batch_size), alphabet, dtype, nthreads)


__all__ = [
    "ContributionScores",
    "from_filters",
    "from_one_hot",
    "insert_sites",
    "mutagenesis",
    "mutagenesis_effects",
    "one_hot",
    "one_hot_batches",
    "read_contributions",
    "read_modisco",
    "read_seqlets",
    "site_spacings",
    "to_filters",
]
