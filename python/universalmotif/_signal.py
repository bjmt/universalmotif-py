"""bigWig signal around motif hits and other genomic sites, as a matrix of
one row per site, and a plot of its mean profile and heatmap."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_ORIGINS = ("centre", "start", "end")
_STATS = ("mean", "max", "min", "sum")
_CONTROL = "#7A7A7A"


@dataclass(frozen=True, eq=False)
class SignalMatrix:
    """bigWig signal around genomic sites, from
    [`signal_matrix()`][universalmotif.signal_matrix].

    ``sites`` is the table of sites, one row each, in the order of the rows
    of ``values``. ``values`` is an array of shape (sites, bins) with the
    signal in each bin of a window around each site, read along the site's
    strand. Bins beyond the ends of a chromosome are NaN. ``track`` names
    the signal, and ``flank``, ``bin_size`` and ``origin`` are as given to
    [`signal_matrix()`][universalmotif.signal_matrix].

    Indexing with a motif name, or a list of names, gives the sites of
    those motifs. Indexing with a boolean mask or row numbers gives those
    rows.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> sites = pl.DataFrame({"motif": ["a", "b", "a"], "chrom": "chr1", "start": [100, 200, 300],
    ...                       "end": [106, 206, 306], "strand": "+"})
    >>> sm = um.SignalMatrix(sites, np.zeros((3, 4)), "atac", 20, 10, "centre")
    >>> sm
    SignalMatrix(3 sites of 2 motifs, atac in 4 bins of 10 around the centre)
    >>> sm.positions.tolist()
    [-15.5, -5.5, 4.5, 14.5]
    >>> len(sm["a"])
    2
    """

    sites: pl.DataFrame
    values: np.ndarray
    track: str
    flank: int
    bin_size: int
    origin: str

    @property
    def positions(self) -> np.ndarray:
        """The position of the middle of each bin, counting the base at the
        origin as 0 and the bases before it as -1, -2 and so on."""
        nbins = self.values.shape[-1]
        return -self.flank + np.arange(nbins) * self.bin_size + (self.bin_size - 1) / 2

    def __len__(self) -> int:
        return self.sites.height

    def __repr__(self) -> str:
        n = len(self)
        what = f"{n:,} site{'s' if n != 1 else ''}"
        if "motif" in self.sites.columns and n:
            k = self.sites["motif"].n_unique()
            what += f" of {k:,} motif{'s' if k != 1 else ''}"
        nbins = self.values.shape[-1]
        where = {"centre": "the centre", "start": "the start", "end": "the end"}[self.origin]
        return (f"SignalMatrix({what}, {self.track} in {nbins:,} bin{'s' if nbins != 1 else ''} "
                f"of {self.bin_size:,} around {where})")

    def __getitem__(self, key: Any) -> SignalMatrix:
        if isinstance(key, pl.Series) and key.dtype in (pl.String, pl.Categorical, pl.Enum):
            key = key.cast(pl.String).to_list()
        if isinstance(key, str) or (isinstance(key, Sequence) and key and all(isinstance(k, str) for k in key)):
            names = [key] if isinstance(key, str) else list(key)
            if "motif" not in self.sites.columns:
                raise KeyError("the sites have no motif column to pick motifs by")
            motif = self.sites["motif"].cast(pl.String)
            missing = [m for m in names if not (motif == m).any()]
            if missing:
                raise KeyError(f"no sites of {', '.join(map(repr, missing))}")
            rows = np.flatnonzero(motif.is_in(names).to_numpy())
        else:
            if isinstance(key, pl.Series):
                key = key.to_numpy()
            rows = np.asarray(key) if len(key) else np.zeros(0, dtype=np.int64)
            if rows.dtype == bool:
                if rows.shape != (len(self),):
                    raise IndexError(f"a mask must have one value for each of the {len(self)} sites")
                rows = np.flatnonzero(rows)
            elif rows.dtype.kind not in "iu":
                raise TypeError("index sites with motif names, a boolean mask or row numbers")
        return SignalMatrix(self.sites[rows], self.values[rows], self.track, self.flank, self.bin_size,
                            self.origin)

    def clip(self, upper: float | None = None, lower: float | None = None, quantile: bool = True) -> SignalMatrix:
        """The same sites with their extreme values capped, so that a few
        sites with far more signal than the rest, such as artefacts of
        repeats, do not pull the mean around.

        ``upper`` and ``lower`` are quantiles of every value in the matrix,
        such as 0.99 and 0.01, and values above the ``upper`` quantile or
        below the ``lower`` one are set to it. With ``quantile=False`` they
        are the limits themselves. At least one must be given, and an end
        left as ``None`` is not capped. NaN stays NaN. The matrix itself is
        left as it is.

        Examples
        --------
        >>> import numpy as np
        >>> import polars as pl
        >>> import universalmotif as um
        >>> sites = pl.DataFrame({"chrom": "chr1", "start": [100, 200], "end": [106, 206]})
        >>> sm = um.SignalMatrix(sites, np.array([[1.0, 2.0], [3.0, 50.0]]), "atac", 1, 1, "start")
        >>> sm.clip(upper=5, quantile=False).values.tolist()
        [[1.0, 2.0], [3.0, 5.0]]
        >>> sm.clip(upper=0.5, lower=0.25).values.tolist()
        [[1.75, 2.0], [2.5, 2.5]]
        """
        if upper is None and lower is None:
            raise ValueError("give upper, lower or both")
        for what, v in (("upper", upper), ("lower", lower)):
            if v is not None and (np.isnan(v) or (quantile and not 0 <= v <= 1)):
                raise ValueError(f"{what} must be a quantile from 0 to 1" if quantile else f"{what} cannot be NaN")
        if upper is not None and lower is not None and lower > upper:
            raise ValueError("lower cannot be above upper")
        values = self.values
        hi, lo = upper, lower
        if quantile:
            finite = values[np.isfinite(values)].astype(np.float64)
            if finite.size == 0:
                return SignalMatrix(self.sites, values.copy(), self.track, self.flank, self.bin_size, self.origin)
            hi = None if upper is None else float(np.quantile(finite, upper))
            lo = None if lower is None else float(np.quantile(finite, lower))
        clipped = np.clip(values, lo, hi).astype(values.dtype, copy=False)
        return SignalMatrix(self.sites, clipped, self.track, self.flank, self.bin_size, self.origin)

    def profile(self, by: str | None = "motif") -> pl.DataFrame:
        """The mean signal in each bin, with its standard error and the
        number of sites with a value there.

        With ``by``, a column of ``sites`` such as ``"motif"``, there is one
        profile for each of its values, in the order they first appear. The
        default ``"motif"`` is ignored when the sites have no such column.

        Examples
        --------
        >>> import numpy as np
        >>> import polars as pl
        >>> import universalmotif as um
        >>> sites = pl.DataFrame({"chrom": "chr1", "start": [100, 200], "end": [106, 206]})
        >>> sm = um.SignalMatrix(sites, np.array([[1.0, 3.0], [3.0, np.nan]]), "atac", 1, 1, "start")
        >>> sm.profile().select("position", "mean", "n").rows()
        [(-1.0, 2.0, 2), (0.0, 3.0, 1)]
        """
        groups = _groups(self.sites, by, by == "motif")
        frames = []
        for name, rows in groups:
            mean, se, n = _summary(self.values[rows].astype(np.float64))
            cols = {"position": self.positions, "mean": mean, "se": se, "n": n}
            if by is not None and name is not None:
                cols = {by: [name] * len(mean), **cols}
            frames.append(pl.DataFrame(cols))
        if not frames:
            return pl.DataFrame(schema={"position": pl.Float64, "mean": pl.Float64, "se": pl.Float64,
                                        "n": pl.Int64})
        return pl.concat(frames)


def _summary(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mean of each column ignoring NaN, its standard error, and the
    number of values in it."""
    n = np.sum(~np.isnan(x), axis=0)
    mean = np.full(x.shape[1], np.nan)
    se = np.full(x.shape[1], np.nan)
    some, two = n > 0, n > 1
    mean[some] = np.nansum(x[:, some], axis=0) / n[some]
    sd = np.sqrt(np.nansum((x[:, two] - mean[two]) ** 2, axis=0) / (n[two] - 1))
    se[two] = sd / np.sqrt(n[two])
    return mean, se, n


def _groups(sites: pl.DataFrame, by: str | None, optional: bool) -> list[tuple[Any, np.ndarray]]:
    """The rows of each value of a column, in the order the values first
    appear, or every row as one group."""
    if by is None or (optional and by not in sites.columns):
        return [(None, np.arange(sites.height))] if sites.height else []
    if by not in sites.columns:
        raise ValueError(f"the sites have no {by!r} column")
    col = sites[by]
    if col.dtype in (pl.Categorical, pl.Enum):
        col = col.cast(pl.String)
    if "motif_i" in sites.columns and by == "motif":
        order = (sites.select(pl.col("motif").cast(pl.String), pl.col("motif_i"))
                 .unique("motif", keep="first", maintain_order=True)
                 .sort("motif_i", maintain_order=True)["motif"].to_list())
    else:
        order = col.unique(maintain_order=True).to_list()
    values = col.to_list()
    index: dict[Any, list[int]] = {v: [] for v in order}
    for i, v in enumerate(values):
        index[v].append(i)
    return [(v, np.asarray(index[v], dtype=np.int64)) for v in order]


def _check_window(flank: int, bin_size: int, origin: str) -> tuple[int, int, str]:
    flank, bin_size = int(flank), int(bin_size)
    if bin_size < 1:
        raise ValueError("bin_size must be 1 or more")
    if flank < 1 or flank % bin_size:
        raise ValueError("flank must be 1 or more and a multiple of bin_size")
    if origin == "center":
        origin = "centre"
    if origin not in _ORIGINS:
        raise ValueError('origin must be "centre", "start" or "end"')
    return flank, bin_size, origin


def _genomic_table(x, what: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ranges in genomic positions as given, with their columns, and as a
    range table, refusing hits still in positions within the sequences
    scanned."""
    from ._genomic import _range_table, _ranges_frame
    from ._intervals import _REGION

    frame = _ranges_frame(x)
    if isinstance(frame, pl.DataFrame) and frame.height and "sequence" in frame.columns \
            and not any(c in frame.columns for c in ("chrom", "seqnames", "chr", "chromosome")) \
            and _REGION.match(str(frame["sequence"][0])):
        raise ValueError(f"the {what} are in positions within the sequences scanned; move them to the "
                         "genome with genomic_hits() first")
    table = _range_table(frame, what)
    if (table["end"] < table["start"]).any():
        raise ValueError(f"all {what} must have start <= end")
    return (frame if isinstance(frame, pl.DataFrame) else table), table


def _anchors(table: pl.DataFrame, origin: str) -> tuple[np.ndarray, np.ndarray]:
    """The boundary each window is centred on, in genomic positions, and
    whether it is read backwards from it, on the minus strand."""
    start = table["start"].to_numpy()
    end = table["end"].to_numpy()
    minus = (table["strand"] == "-").to_numpy()
    total = start + end
    if origin == "centre":
        anchor = np.where(minus, (total + 1) // 2, total // 2)
    elif origin == "start":
        anchor = np.where(minus, end, start)
    else:
        anchor = np.where(minus, start, end)
    return anchor.astype(np.int64), minus


def signal_matrix(
    bigwig: str | os.PathLike[str],
    sites,
    flank: int = 1000,
    bin_size: int = 10,
    origin: str = "centre",
    stat: str = "mean",
    uncovered: float | None = 0.0,
    max_sites: int | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> SignalMatrix:
    """bigWig signal, such as ATAC, ChIP or conservation, in bins around
    motif hits or other genomic sites.

    Each site gets a window of ``flank`` bases on either side of its
    ``origin``, which is split into bins of ``bin_size`` bases. The signal
    in each bin is summarized by ``stat``. Windows are read along the
    strand of their site, so that every row of the matrix faces the same
    way as its motif, and sites without a strand are read as on the plus
    strand. The rows can be averaged into a profile with
    [`SignalMatrix.profile()`][universalmotif.SignalMatrix.profile] and
    drawn with [`plot_signal_profile()`][universalmotif.plot_signal_profile].

    The base at the origin is position 0. For ``"start"`` this is the first
    letter of the site, for ``"end"`` the first base past its last letter,
    and for ``"centre"`` its middle letter, or the second of the two middle
    letters of a site of even width.

    Bases the file has no value for count as ``uncovered``, which is zero by
    default as is usual for coverage, or are left out with ``None``. Bins
    beyond the ends of a chromosome are NaN.

    Needs the ``bigwig`` extra.

    Parameters
    ----------
    bigwig
        The path of a bigWig file.
    sites
        Sites in genomic positions: hits from
        [`genomic_hits()`][universalmotif.genomic_hits] or from
        ``scan_sequences(ranges=)``, or any range table or (chrom, start,
        end[, strand]) tuples, as for
        [`extract_ranges()`][universalmotif.extract_ranges]. Positions count
        from 0 and leave the end out. The columns of a table are kept in
        [`SignalMatrix.sites`][universalmotif.SignalMatrix], so that a
        ``motif`` or ``score`` column can be used to group or sort the rows.
    flank
        The number of bases on either side of the origin. It must be a
        multiple of ``bin_size``.
    bin_size
        The width of each bin, from 1 for every base up.
    origin
        Centre the windows on the middle of each site, ``"centre"``, or on
        its ``"start"`` or ``"end"``, both taken along its strand.
    stat
        Summarize each bin by its ``"mean"``, ``"max"``, ``"min"`` or
        ``"sum"``.
    uncovered
        The value of bases the file has no value for, or ``None`` to leave
        them out of each bin.
    max_sites
        Keep at most this many sites of each motif, picked at random with
        ``seed``, or of all sites when there is no ``motif`` column. With
        ``None`` every site is kept.
    seed
        The seed for picking sites with ``max_sites``.
    name
        The name of the signal, by default the name of the file without its
        extension.

    Returns
    -------
    SignalMatrix
        The sites and a matrix of their signal.

    References
    ----------
    - Gu Z, Eils R, Schlesner M, Ishaque N (2018). EnrichedHeatmap: an
      R/Bioconductor package for comprehensive visualization of genomic
      signal associations. *BMC Genomics* 19:234.
      <https://doi.org/10.1186/s12864-018-4625-x>
    - Kent WJ, Zweig AS, Barber G, Hinrichs AS, Karolchik D (2010). BigWig and
      BigBed: enabling browsing of large distributed datasets. *Bioinformatics*
      26(17):2204–2207. <https://doi.org/10.1093/bioinformatics/btq351>

    Examples
    --------
    >>> import universalmotif as um
    >>> genome = um.Genome("genome.fa.gz")  # doctest: +SKIP
    >>> peaks = um.extract_ranges(genome, "peaks.bed")  # doctest: +SKIP
    >>> hits = um.genomic_hits(um.scan_sequences(motifs, peaks))  # doctest: +SKIP
    >>> atac = um.signal_matrix("atac.bw", hits, flank=200, bin_size=1)  # doctest: +SKIP
    >>> um.plot_signal_profile(atac)  # doctest: +SKIP
    """
    try:
        import pybigtools
    except ImportError as e:
        raise ImportError(
            "reading bigWig files needs pybigtools; install it with `pip install "
            "'universalmotif[bigwig]'`"
        ) from e
    flank, bin_size, origin = _check_window(flank, bin_size, origin)
    if stat not in _STATS:
        raise ValueError('stat must be "mean", "max", "min" or "sum"')
    if max_sites is not None and max_sites < 1:
        raise ValueError("max_sites must be 1 or more, or None for every site")
    kept, table = _genomic_table(sites, "sites")
    if max_sites is not None:
        rows = _sample(kept, max_sites, seed)
        kept, table = kept[rows], table[rows]
    anchor, minus = _anchors(table, origin)
    width = 2 * flank
    nbins = width // bin_size
    path = os.fspath(bigwig)
    reader = pybigtools.open(path)
    sizes = reader.chroms()
    fill = np.nan if uncovered is None else float(uncovered)
    values = np.full((table.height, nbins), np.nan, dtype=np.float32)
    chroms = table["chrom"].to_list()
    # Windows are read into a block of rows and binned together, a block of
    # about 32 MB at a time.
    block = max(1, (1 << 22) // width)
    buffer = np.empty((min(block, table.height), width), dtype=np.float64)
    for first in range(0, table.height, block):
        last = min(first + block, table.height)
        windows = buffer[: last - first]
        windows.fill(np.nan)
        for k, i in enumerate(range(first, last)):
            c = chroms[i]
            lo = int(anchor[i]) - flank
            size = sizes.get(c)
            if size is None:
                windows[k, max(0, -lo):] = fill
                continue
            a, b = max(lo, 0), min(lo + width, size)
            if a < b:
                windows[k, a - lo:b - lo] = reader.values(c, a, b, fillna=fill)
        flip = minus[first:last]
        windows[flip] = windows[flip, ::-1]
        values[first:last] = _bin(windows, nbins, bin_size, stat)
    if size_missing := sorted({c for c in chroms if c not in sizes}):
        warnings.warn(f"{len(size_missing):,} chromosome{'s' if len(size_missing) != 1 else ''} of the "
                      f"sites, such as {size_missing[0]!r}, are not in {path}; their bases count as "
                      "uncovered", stacklevel=2)
    if name is None:
        name = os.path.basename(path)
        for ext in (".bigwig", ".bigWig", ".bw", ".BW"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
    return SignalMatrix(kept, values, str(name), flank, bin_size, origin)


def site_matrix(
    hits: pl.DataFrame,
    anchors,
    flank: int = 1000,
    bin_size: int = 10,
    origin: str = "centre",
    count: str = "centre",
    value: str | None = None,
    strand: str = "both",
    chrom_lengths: Mapping[str, int] | None = None,
    name: str = "sites",
) -> SignalMatrix:
    """Motif hits in bins around anchors, such as peak summits or TSSs.

    This is the counterpart of
    [`signal_matrix()`][universalmotif.signal_matrix] with motif hits as the
    signal. Each anchor gets a window of ``flank`` bases on either side of
    its ``origin``, split into bins of ``bin_size`` bases, and each bin
    counts the hits in it. The result is a
    [`SignalMatrix`][universalmotif.SignalMatrix] like any other, so it can
    be drawn with
    [`plot_signal_profile()`][universalmotif.plot_signal_profile], whose
    mean is then the number of hits per anchor in each bin. Hits of a
    shuffled motif counted the same way make a control.

    With a ``motif`` column in the hits, there is one row for each anchor
    and motif, the anchors of the first motif first, and a ``motif`` column
    is added to the anchors so that each motif is drawn in its own panel.
    Windows are read along the strand of their anchor, so that upstream of
    a TSS is always on the left, and anchors without a strand are read as
    on the plus strand.

    Only the hits given are counted. When they come from scanning regions
    such as peaks, rather than whole chromosomes, the windows should lie
    within those regions, since a window reaching beyond them counts no
    hits there whether or not the genome has sites.

    Parameters
    ----------
    hits
        Hits in genomic positions, from
        [`genomic_hits()`][universalmotif.genomic_hits], from
        ``scan_sequences(ranges=)`` or from scanning whole chromosomes.
    anchors
        The anchors, as a range table or (chrom, start, end[, strand])
        tuples, as for [`extract_ranges()`][universalmotif.extract_ranges].
        The columns of a table are kept.
    flank, bin_size, origin
        As for [`signal_matrix()`][universalmotif.signal_matrix].
    count
        Count each hit once, in the bin of its middle letter, with
        ``"centre"``, or count every base it covers with ``"coverage"``,
        which gives the mean number of hits over each base of a bin.
    value
        A numeric column of the hits, such as ``"score"``, to add up in
        place of counting each hit as 1.
    strand
        Count every hit with ``"both"``, or only those on the same strand
        as their anchor with ``"same"``, or on the other with
        ``"opposite"``. Hits without a strand count as on the plus strand.
    chrom_lengths
        The length of each chromosome, so that bins beyond the end of one
        are NaN rather than 0, such as ``genome.lengths`` of a
        [`Genome`][universalmotif.Genome].
    name
        The name of the matrix's track.

    Returns
    -------
    SignalMatrix
        The anchors and the hits in each bin around them.

    Examples
    --------
    >>> import polars as pl
    >>> import universalmotif as um
    >>> hits = pl.DataFrame({"motif": "TATA", "chrom": "chr1", "start": [95, 110, 290],
    ...                      "end": [101, 116, 296], "strand": ["+", "+", "-"]})
    >>> tss = [("chr1", 120, 121, "+"), ("chr1", 280, 281, "-")]
    >>> sm = um.site_matrix(hits, tss, flank=30, bin_size=10)
    >>> sm.positions.tolist()
    [-25.5, -15.5, -5.5, 4.5, 14.5, 24.5]
    >>> sm.values.tolist()
    [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    >>> sm.profile().select("position", "mean").rows()[:3]
    [(-25.5, 0.5), (-15.5, 0.5), (-5.5, 0.5)]
    """
    flank, bin_size, origin = _check_window(flank, bin_size, origin)
    if count not in ("centre", "center", "coverage"):
        raise ValueError('count must be "centre" or "coverage"')
    if strand not in ("both", "same", "opposite"):
        raise ValueError('strand must be "both", "same" or "opposite"')
    if not isinstance(hits, pl.DataFrame):
        raise TypeError("hits must be a polars DataFrame of hits in genomic positions")
    if value is not None:
        if value not in hits.columns:
            raise ValueError(f"the hits have no {value!r} column")
        if not hits[value].dtype.is_numeric():
            raise ValueError(f"value needs a numeric column, and {value!r} is not")
    kept, table = _genomic_table(anchors, "anchors")
    if "motif" in kept.columns and "motif" in hits.columns:
        raise ValueError("the anchors have a motif column, which would clash with the motif of the "
                         "hits; rename it first")
    _, hit_table = _genomic_table(hits, "hits")
    anchor, minus = _anchors(table, origin)
    width = 2 * flank
    nbins = width // bin_size
    lo = anchor - flank

    hit_start = hit_table["start"].to_numpy()
    hit_end = hit_table["end"].to_numpy()
    hit_minus = (hit_table["strand"] == "-").to_numpy()
    weight = (np.ones(hits.height) if value is None
              else hits[value].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy())
    groups = _groups(hits, "motif", True)
    names = [g for g, _ in groups] or [None]
    anchor_chroms = table["chrom"].to_list()
    known, anchor_code = np.unique(np.asarray(anchor_chroms, dtype=object), return_inverse=True)
    code_of = {c: i for i, c in enumerate(known.tolist())}
    hit_code = np.fromiter((code_of.get(c, -1) for c in hit_table["chrom"].to_list()), dtype=np.int64,
                           count=hit_table.height)
    chrom_rows = [np.flatnonzero(anchor_code == i) for i in range(len(known))]

    values = np.zeros((len(names) * table.height, nbins), dtype=np.float64)
    for g, (_, rows) in enumerate(groups):
        for c, arows in enumerate(chrom_rows):
            h = rows[hit_code[rows] == c]
            if len(h) == 0:
                continue
            if count == "coverage":
                lens = hit_end[h] - hit_start[h]
                hit_of = np.repeat(h, lens)
                pos = hit_start[hit_of] + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
            else:
                hit_of = h
                pos = np.where(hit_minus[h], (hit_start[h] + hit_end[h] + 1) // 2 - 1,
                               (hit_start[h] + hit_end[h]) // 2)
            order = np.argsort(pos, kind="stable")
            pos, hit_of = pos[order], hit_of[order]
            a_lo = lo[arows]
            first = np.searchsorted(pos, a_lo, side="left")
            last = np.searchsorted(pos, a_lo + width, side="left")
            n = last - first
            if n.sum() == 0:
                continue
            row = np.repeat(arows, n)
            idx = np.repeat(first, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
            p, hh = pos[idx], hit_of[idx]
            flip = minus[row]
            k = np.where(flip, lo[row] + width - 1 - p, p - lo[row])
            if strand != "both":
                same = hit_minus[hh] == flip
                keep = same if strand == "same" else ~same
                row, k, hh = row[keep], k[keep], hh[keep]
            np.add.at(values, (g * table.height + row, k // bin_size), weight[hh])
    if count == "coverage":
        values /= bin_size
    if chrom_lengths is not None:
        size = np.array([chrom_lengths.get(c, -1) for c in anchor_chroms], dtype=np.int64)
        if (size < 0).any():
            missing = anchor_chroms[int(np.flatnonzero(size < 0)[0])]
            raise ValueError(f"chrom_lengths has no length for {missing!r}")
        # The bases of each bin in genomic positions, and whether any lies on
        # the chromosome.
        k = np.arange(nbins)
        base_lo = np.where(minus[:, None], lo[:, None] + width - (k + 1) * bin_size, lo[:, None] + k * bin_size)
        off = (base_lo + bin_size <= 0) | (base_lo >= size[:, None])
        values[np.tile(off, (len(names), 1))] = np.nan
    if names == [None]:
        sites = kept
    else:
        order_i = None
        if "motif_i" in hits.columns:
            pairs = hits.select(pl.col("motif").cast(pl.String), "motif_i").unique("motif", keep="first")
            order_i = dict(pairs.iter_rows())
        parts = []
        for nm in names:
            extra = [pl.lit(nm, dtype=pl.String).alias("motif")]
            if order_i is not None:
                extra.append(pl.lit(order_i[nm]).alias("motif_i"))
            parts.append(kept.select(*extra, pl.all()))
        sites = pl.concat(parts)
    return SignalMatrix(sites, values.astype(np.float32), str(name), flank, bin_size, origin)


def _bin(windows: np.ndarray, nbins: int, bin_size: int, stat: str) -> np.ndarray:
    """Summarize the bins of a block of windows, with NaN for bins with no
    values at all."""
    if bin_size == 1:
        return windows
    chunks = windows.reshape(windows.shape[0], nbins, bin_size)
    empty = np.isnan(chunks).all(axis=2)
    reduce = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin, "sum": np.nansum}[stat]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = reduce(chunks, axis=2)
    out[empty] = np.nan
    return out


def _sample(sites: pl.DataFrame, max_sites: int, seed: int | None) -> np.ndarray:
    """Rows to keep, at most ``max_sites`` of each motif, in their order."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, rows in _groups(sites, "motif", True):
        if len(rows) > max_sites:
            rows = np.sort(rng.choice(rows, max_sites, replace=False))
        keep.append(rows)
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=np.int64)


def _block_means(x: np.ndarray, max_rows: int | None) -> np.ndarray:
    """Rows averaged in runs of consecutive rows, into at most
    ``max_rows`` rows, ignoring NaN."""
    n = x.shape[0]
    if max_rows is None or n <= max_rows:
        return x
    starts = np.round(np.linspace(0, n, max_rows + 1)).astype(np.int64)[:-1]
    sums = np.add.reduceat(np.nan_to_num(x, nan=0.0).astype(np.float64), starts, axis=0)
    counts = np.add.reduceat((~np.isnan(x)).astype(np.int64), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _check_control(matrix: SignalMatrix, control: SignalMatrix) -> None:
    if not isinstance(control, SignalMatrix):
        raise TypeError("control must come from signal_matrix()")
    same = (control.flank, control.bin_size, control.origin) == (matrix.flank, matrix.bin_size, matrix.origin)
    if not same:
        raise ValueError("control must be read with the same flank, bin_size and origin as the sites")
    if len(control) == 0:
        raise ValueError("control has no sites")


def plot_signal_profile(
    matrix: SignalMatrix,
    control: SignalMatrix | None = None,
    by: str | None = "motif",
    groups: Any | Sequence[Any] | None = None,
    band: str | None = "ci",
    heatmap: bool = True,
    sort_by: str | None = None,
    max_rows: int | None = 500,
    vmin: float | None = None,
    vmax: float | None = None,
    colours: str | Mapping[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    heatmap_ylabel: str | None = None,
    legend_title: str | None = None,
    text_size: float = 9,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Figure | Axes:
    """Draw the mean signal around sites, with a heatmap of the sites
    themselves below it.

    The line is the mean of each bin of a
    [`signal_matrix()`][universalmotif.signal_matrix], with a band around
    it, and the dotted line marks the origin. Below it, a heatmap shows the
    sites, one row each, sorted from the most signal down. Each motif, or
    each value of ``by``, gets its own panels side by side, sharing their
    scales.

    A ``control``, read from the same file around other sites such as
    shuffled motifs' hits or matched background regions, is drawn as a grey
    line. A profile alone is easy to over-read. ATAC signal, for example,
    dips over many short sequences because of the cutting preference of the
    Tn5 enzyme, which can look like the footprint of a bound factor. When
    the control's sites have the ``by`` column too, each panel gets the
    control of its own group, and otherwise every panel gets all of it.

    The line and band use the values as they are, so a few sites with far
    more signal than the rest can pull them around. Capping the values
    first with [`SignalMatrix.clip()`][universalmotif.SignalMatrix.clip]
    keeps them in check, while ``vmin`` and ``vmax`` only set the colours
    of the heatmap.

    Needs the ``plot`` extra.

    Parameters
    ----------
    matrix
        Signal around sites from
        [`signal_matrix()`][universalmotif.signal_matrix].
    control
        Signal around control sites, read with the same ``flank``,
        ``bin_size`` and ``origin``.
    by
        The column of the sites to split them by, by default ``"motif"``
        when there is one, or ``None`` for one panel of every site.
    groups
        The values of ``by`` to draw, in this order, by default all of
        them.
    band
        The band around the mean: ``"ci"`` for its 95% confidence interval,
        ``"se"`` for one standard error, or ``None`` for none.
    heatmap
        Draw the heatmap of the sites below the mean.
    sort_by
        A numeric column of the sites to sort the heatmap's rows by, from
        the largest down, such as ``"score"``. By default they are sorted by
        their mean signal.
    max_rows
        Draw at most this many rows in each heatmap, or a row for every
        site with ``None``. With more sites than this, the sorted sites are
        split into this many groups of about equal size, in order, and each
        group is drawn as one row of its average. 2,000 sites drawn in 500
        rows, for example, give rows of 4 sites each. Every site is then
        shown, and sparse signal, such as ATAC read base by base in which
        most bases have no reads, is smoothed.
    vmin, vmax
        The ends of the heatmap's colour scale. By default the scale runs
        from 0 to the 99th percentile of the values drawn, or is centred on
        0 when there are clearly negative values, such as in a track of
        ratios.
    colours
        One colour for the lines, or a mapping from groups to colours. By
        default a single group is drawn in blue, and motifs in the colours
        [`plot_hits()`][universalmotif.plot_hits] gives them.
    title
        The title above the plot. With several groups each panel is also
        named after its group.
    xlabel
        The title of the x-axis, by default ``"Position from centre"``, or
        from the start or end, after the origin of the matrix.
    ylabel, heatmap_ylabel
        Titles for the y-axes of the mean and of the heatmap, such as
        ``"Mean ATAC"`` and ``"Sites"``. There are none by default.
    legend_title
        A title for the colour bar of the heatmap, such as ``"ATAC"``. There
        is none by default.
    text_size
        The size of the text, in points.
    figsize
        The size of the figure in inches, by default sized to the groups.
    ax
        Axes to draw the means into, one line per group, without the
        heatmap. A control split by group is drawn as a faint line in the
        colour of its group. The axes are returned, and otherwise a new
        figure is.

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure, or the axes when ``ax`` is given.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import polars as pl
    >>> import universalmotif as um
    >>> rng = np.random.default_rng(1)
    >>> sites = pl.DataFrame({"motif": ["GATA"] * 50, "chrom": "chr1",
    ...                       "start": np.arange(50) * 1000, "end": np.arange(50) * 1000 + 6})
    >>> peak = np.exp(-np.linspace(-3, 3, 40) ** 2)
    >>> sm = um.SignalMatrix(sites, rng.poisson(5 * peak + 1, (50, 40)).astype(float),
    ...                      "chip", 200, 10, "centre")
    >>> fig = um.plot_signal_profile(sm, ylabel="Mean ChIP")
    >>> [a.get_ylabel() for a in fig.axes[:2]]
    ['Mean ChIP', '']
    >>> plt.close(fig)
    """
    from ._hit_contrib import _palette, _relative_ticks
    from ._plot import _pyplot

    if not isinstance(matrix, SignalMatrix):
        raise TypeError("matrix must come from signal_matrix()")
    if band not in ("ci", "se", None):
        raise ValueError('band must be "ci", "se" or None')
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be 1 or more, or None for every site")
    if len(matrix) == 0:
        raise ValueError("there are no sites to draw")
    if control is not None:
        _check_control(matrix, control)
    if sort_by is not None:
        if sort_by not in matrix.sites.columns:
            raise ValueError(f"the sites have no {sort_by!r} column to sort by")
        if not matrix.sites[sort_by].dtype.is_numeric():
            raise ValueError(f"sort_by needs a numeric column, and {sort_by!r} is not")
    found = _groups(matrix.sites, by, by == "motif")
    names = [g for g, _ in found]
    if groups is not None:
        if names == [None]:
            raise ValueError("groups needs a by column to pick groups from")
        wanted = [groups] if isinstance(groups, str) or not isinstance(groups, Sequence) else list(groups)
        missing = [g for g in wanted if g not in names]
        if missing:
            raise ValueError(f"no sites of {', '.join(map(repr, missing[:5]))}")
        lookup = dict(found)
        found = [(g, lookup[g]) for g in dict.fromkeys(wanted)]
    labels = ["" if g is None else str(g) for g, _ in found]
    if colours is None and (len(found) == 1 or by != "motif"):
        from ._hit_contrib import _LINE
        from ._plot_hits import _motif_colours

        palette = {labels[0]: _LINE} if len(found) == 1 else _motif_colours(labels, list(range(len(labels))))
    else:
        all_labels = [str(g) for g, _ in _groups(matrix.sites, by, by == "motif")]
        palette = _palette(labels, [all_labels.index(x) if x in all_labels else 0 for x in labels], colours)

    split_control = control is not None and by is not None and by in control.sites.columns
    control_rows = dict(_groups(control.sites, by, False)) if split_control else None

    def control_of(group) -> np.ndarray | None:
        if control is None:
            return None
        if control_rows is None:
            return control.values
        rows = control_rows.get(group)
        return None if rows is None else control.values[rows]

    def summary(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        mean, se, _ = _summary(x.astype(np.float64))
        if band is None:
            return mean, None
        return mean, se if band == "se" else 1.96 * se

    plt = _pyplot()
    import matplotlib
    from matplotlib.ticker import MaxNLocator

    x = matrix.positions
    half = matrix.bin_size / 2
    xlim = (x[0] - half, x[-1] + half)
    if xlabel is None:
        xlabel = f"Position from {matrix.origin}"

    def draw_mean(target, rows, group, label) -> None:
        c = control_of(group)
        if c is not None and len(c):
            cm, ch = summary(c)
            if ch is not None:
                target.fill_between(x, cm - ch, cm + ch, color=_CONTROL, alpha=0.18, linewidth=0, zorder=2)
            target.plot(x, cm, color=_CONTROL, linewidth=1.2, zorder=3,
                        label="Control")
        mean, h = summary(matrix.values[rows])
        colour = palette[label]
        if h is not None:
            target.fill_between(x, mean - h, mean + h, color=colour, alpha=0.22, linewidth=0, zorder=4)
        target.plot(x, mean, color=colour, linewidth=1.4, zorder=5, label=label or "Sites")

    def style(target) -> None:
        target.tick_params(labelsize=text_size - 1)
        for side in ("top", "right"):
            target.spines[side].set_visible(False)

    if ax is not None:
        fig = ax.figure
        for (group, rows), label in zip(found, labels):
            mean, h = summary(matrix.values[rows])
            colour = palette[label]
            if h is not None:
                ax.fill_between(x, mean - h, mean + h, color=colour, alpha=0.2, linewidth=0)
            ax.plot(x, mean, color=colour, linewidth=1.4, label=label or None)
        if control is not None and control_rows is None:
            cm, ch = summary(control.values)
            if ch is not None:
                ax.fill_between(x, cm - ch, cm + ch, color=_CONTROL, alpha=0.18, linewidth=0)
            ax.plot(x, cm, color=_CONTROL, linewidth=1.2, label="Control")
        elif control is not None:
            for (group, _), label in zip(found, labels):
                c = control_of(group)
                if c is not None and len(c):
                    ax.plot(x, summary(c)[0], color=palette[label], linewidth=1.2, alpha=0.4,
                            label=f"{label} control")
        ax.axvline(0, color="0.55", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=text_size)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=text_size)
        style(ax)
        if len(found) > 1 or control is not None:
            ax.legend(frameon=False, fontsize=text_size)
        if title is not None:
            ax.set_title(title, fontsize=text_size + 1)
        return ax

    shown = []
    for _, rows in found:
        part = matrix.values[rows]
        if sort_by is not None:
            key = matrix.sites[sort_by][rows].cast(pl.Float64).fill_nan(None).fill_null(-np.inf).to_numpy()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                key = np.nan_to_num(np.nanmean(part, axis=1), nan=-np.inf)
        order = np.argsort(-key, kind="stable")
        shown.append(_block_means(part[order], max_rows))
    if heatmap:
        drawn = np.concatenate([s.ravel() for s in shown])
        drawn = drawn[~np.isnan(drawn)]
        lo_q, hi_q = (np.percentile(drawn, [1, 99]) if drawn.size else (0.0, 1.0))
        diverging = lo_q < 0 < hi_q and -lo_q > 0.1 * hi_q
        if diverging:
            top = max(abs(lo_q), abs(hi_q)) if vmax is None else vmax
            lo_v = -top if vmin is None else vmin
            hi_v = top
            cmap = matplotlib.colormaps["RdBu_r"]
        else:
            lo_v = (min(0.0, float(lo_q)) if vmin is None else vmin)
            hi_v = (float(hi_q) if vmax is None else vmax)
            cmap = matplotlib.colormaps["Blues"]
        if not hi_v > lo_v:
            hi_v = lo_v + 1.0
        cmap = cmap.with_extremes(bad="#E6E6E6")

    rows_of = ["mean", "heat"] if heatmap else ["mean"]
    heights = {"mean": 1.6, "heat": 3.2}
    if figsize is None:
        panel = 2.6
        figsize = (len(found) * (panel + 0.4) + 0.6 + (0.6 if heatmap else 0),
                   sum(heights[r] for r in rows_of) + 1.1)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(len(rows_of), len(found), height_ratios=[heights[r] for r in rows_of],
                            wspace=0.25, hspace=0.1)
    means = []
    heat = im = None
    for j, ((group, rows), label) in enumerate(zip(found, labels)):
        mean_ax = fig.add_subplot(grid[0, j])
        draw_mean(mean_ax, rows, group, label)
        mean_ax.axvline(0, color="0.55", linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
        if ylabel is not None and j == 0:
            mean_ax.set_ylabel(ylabel, fontsize=text_size)
        mean_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        mean_ax.set_xlim(*xlim)
        style(mean_ax)
        means.append(mean_ax)
        bottom = mean_ax
        if heatmap:
            heat = fig.add_subplot(grid[1, j], sharex=mean_ax)
            rows_drawn = shown[j]
            im = heat.imshow(rows_drawn, aspect="auto", cmap=cmap, vmin=lo_v, vmax=hi_v,
                             interpolation="antialiased", interpolation_stage="rgba",
                             extent=(xlim[0], xlim[1], len(rows), 0))
            heat.axvline(0, color="0.25", linewidth=0.6, linestyle=(0, (1, 2)))
            if heatmap_ylabel is not None and j == 0:
                heat.set_ylabel(heatmap_ylabel, fontsize=text_size)
            heat.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            heat.set_ylim(len(rows), 0)
            style(heat)
            mean_ax.tick_params(axis="x", labelbottom=False)
            bottom = heat
        ticks = _relative_ticks(xlim[0], xlim[1])
        bottom.set_xticks(ticks)
        bottom.set_xlim(*xlim)
        bottom.set_xlabel(xlabel, fontsize=text_size)
        heading = label if len(found) > 1 or title is None else title
        mean_ax.set_title(heading, fontsize=text_size + 1)
    if len(means) > 1:
        lo = min(a.get_ylim()[0] for a in means)
        hi = max(a.get_ylim()[1] for a in means)
        for a in means:
            a.set_ylim(lo, hi)
    if control is not None:
        means[-1].legend(frameon=False, fontsize=text_size - 1, loc="upper right")
    if len(found) > 1 and title is not None:
        fig.suptitle(title, fontsize=text_size + 2)
    fig.subplots_adjust(left=0.75 / figsize[0], right=1 - (0.85 if heatmap else 0.25) / figsize[0],
                        top=1 - (0.6 if len(found) > 1 and title else 0.45) / figsize[1],
                        bottom=0.5 / figsize[1])
    if heatmap:
        pos = heat.get_position()
        cax = fig.add_axes((pos.x1 + 0.15 / figsize[0], pos.y0, 0.12 / figsize[0],
                            min(pos.height, 1.4 / figsize[1])))
        bar = fig.colorbar(im, cax=cax)
        if legend_title is not None:
            bar.set_label(legend_title, fontsize=text_size - 1)
        bar.ax.tick_params(labelsize=text_size - 2)
        bar.outline.set_visible(False)
    return fig


__all__ = ["SignalMatrix", "plot_signal_profile", "signal_matrix", "site_matrix"]
