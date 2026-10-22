"""Sequences packed for scanning."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from . import _core


class PackedSequences(Mapping[str, str]):
    """Named sequences packed for scanning, from [`read_fasta()`][universalmotif.read_fasta] with
    ``packed=True`` or from [`pack_sequences()`][universalmotif.pack_sequences].

    DNA and RNA take two bits per letter, so a human genome needs about
    0.8 GB instead of 3.1 GB as text; other alphabets take a byte per
    letter. [`scan_sequences()`][universalmotif.scan_sequences] scans them as they are, which makes them
    the way to scan one large set of sequences many times.

    They are also a read-only mapping from names to sequences, so they can
    go wherever a dict of sequences can. Each sequence is then decoded when
    it is looked up. DNA, RNA and protein letters come back in upper case,
    so soft-masking is lost, while the letters of custom alphabets and any
    other letters, such as ``N``, come back as they were.

    Examples
    --------
    >>> import universalmotif as um
    >>> packed = um.pack_sequences({"chr1": "ACGTNNacgt", "chr2": "TTGACC"})
    >>> list(packed), packed.lengths
    (['chr1', 'chr2'], {'chr1': 10, 'chr2': 6})
    >>> packed["chr1"]
    'ACGTNNACGT'
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
        """Positions ``start`` to ``end`` of the sequence called ``name``,
        counted from 1 and inclusive, decoding only those.

        Examples
        --------
        >>> import universalmotif as um
        >>> packed = um.pack_sequences({"chr1": "ACGTNNACGT"})
        >>> packed.subseq("chr1", 3, 8)
        'GTNNAC'
        """
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
            f"{_size(self.nbytes)})"
        )


def _size(nbytes: int) -> str:
    """A number of bytes in the largest unit that keeps it at 1 or more."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:,.1f} {unit}"
    return f"{nbytes} bytes"


def pack_sequences(
    sequences: str | Sequence[str] | Mapping[str, str],
    alphabet: str = "DNA",
    nthreads: int = 1,
) -> PackedSequences:
    """Pack sequences for scanning as [`PackedSequences`][universalmotif.PackedSequences].

    ``alphabet`` is ``"DNA"``, ``"RNA"``, ``"AA"`` or the letters of a custom
    alphabet, and must match the motifs the sequences will be scanned with.
    Unnamed sequences are called ``seq1``, ``seq2`` and so on; names must be
    unique.

    Examples
    --------
    >>> import universalmotif as um
    >>> seqs = um.example_sequences()
    >>> packed = um.pack_sequences(seqs)
    >>> len(packed), packed.alphabet
    (100, 'DNA')
    >>> wbox = um.example_motifs()[5]
    >>> um.scan_sequences(wbox, packed).equals(um.scan_sequences(wbox, seqs))
    True
    """
    from ._analysis import _sequence_pairs

    names, seqs = _sequence_pairs(sequences)
    return PackedSequences(_core._pack_sequences(names, seqs, alphabet, nthreads))
