"""Sequences packed for scanning."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from . import _core


class PackedSequences(Mapping[str, str]):
    """Named sequences packed for scanning, from :func:`read_fasta` with
    ``packed=True`` or from :func:`pack_sequences`.

    DNA and RNA take two bits per letter, so a human genome needs about
    0.8 GB instead of 3.1 GB as text; other alphabets take a byte per
    letter. :func:`scan_sequences` scans them as they are, which makes them
    the way to scan one large set of sequences many times.

    They are also a read-only mapping from names to sequences, so they can
    go wherever a dict of sequences can. Each sequence is then decoded when
    it is looked up, with the alphabet's letters in upper case (custom
    alphabets as they were) and any other letters, such as ``N``, as they
    were; soft-masking (lower case) is lost.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _core._PackedSequences):
        self._inner = inner

    def __getitem__(self, name: str) -> str:
        i = self._inner.index(name) if isinstance(name, str) else None
        if i is None:
            raise KeyError(name)
        return self._inner.decode(i)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner.names())

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._inner.index(name) is not None

    def subseq(self, name: str, start: int, end: int) -> str:
        """Positions ``start`` to ``end`` (1-based, inclusive) of the
        sequence called ``name``, decoding only those."""
        i = self._inner.index(name)
        if i is None:
            raise KeyError(name)
        if start < 1:
            raise IndexError("start must be at least 1")
        return self._inner.decode_range(i, start - 1, end)

    @property
    def alphabet(self) -> str:
        """The alphabet the sequences were packed for."""
        return self._inner.alphabet

    @property
    def lengths(self) -> dict[str, int]:
        """The length of each sequence."""
        return dict(zip(self._inner.names(), self._inner.lengths()))

    @property
    def nbytes(self) -> int:
        """Memory held by the packed letters, in bytes."""
        return self._inner.nbytes

    def __repr__(self) -> str:
        total = sum(self._inner.lengths())
        return (
            f"PackedSequences({len(self)} {self.alphabet} sequences, {total:,} letters, "
            f"{self.nbytes / 1e6:,.1f} MB)"
        )


def pack_sequences(
    sequences: str | Sequence[str] | Mapping[str, str],
    alphabet: str = "DNA",
    nthreads: int = 1,
) -> PackedSequences:
    """Pack sequences for scanning (see :class:`PackedSequences`).

    ``alphabet`` is ``"DNA"``, ``"RNA"``, ``"AA"`` or the letters of a custom
    alphabet, and must match the motifs the sequences will be scanned with.
    Unnamed sequences are called ``seq1``, ``seq2`` and so on; names must be
    unique.
    """
    from ._analysis import _sequence_pairs

    names, seqs = _sequence_pairs(sequences)
    return PackedSequences(_core._pack_sequences(names, seqs, alphabet, nthreads))
