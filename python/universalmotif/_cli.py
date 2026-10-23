"""The ``universalmotif`` (or ``umotif``) command line.

Each subcommand reads motifs, sequences, genomes and ranges from files,
calls the function of the same purpose in the package, and writes motifs,
FASTA or tab-separated tables to a file or to standard output.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import polars as pl

_MOTIF_FORMATS = ("meme", "jaspar", "homer", "transfac", "cisbp", "uniprobe", "matrix")
_EXTENSIONS = {
    ".meme": "meme",
    ".jaspar": "jaspar",
    ".pfm": "jaspar",
    ".motif": "homer",
    ".motifs": "homer",
    ".homer": "homer",
    ".transfac": "transfac",
    ".dat": "transfac",
    ".cisbp": "cisbp",
    ".uniprobe": "uniprobe",
    ".matrix": "matrix",
}
_FORMAT_LIST = ", ".join(_MOTIF_FORMATS)
_BED_SUFFIXES = (".bed", ".narrowpeak", ".broadpeak", ".bedgraph")
_BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand"]


class CliError(Exception):
    """An error to report as a message, without a traceback."""


# Reading and writing --------------------------------------------------------


def _stem_suffix(path: str) -> str:
    """The extension of a path, looking past a final .gz."""
    p = path.lower()
    p = p.removesuffix(".gz")
    return os.path.splitext(p)[1]


def _read_bytes(path: str) -> bytes:
    raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def _sniff_format(text: str) -> str:
    """The motif format of a file, judged from its contents."""
    if re.search(r"^MEME version", text, re.MULTILINE):
        return "meme"
    if re.search(r"^P[0O]\s", text, re.MULTILINE):
        return "transfac"
    if re.search(r"^Pos\t", text, re.MULTILINE):
        return "cisbp"
    if re.search(r"^[ACGTU]:\s", text, re.MULTILINE):
        return "uniprobe"
    if re.search(r"^[ACGTU]\s*\[", text, re.MULTILINE):
        return "jaspar"
    header = re.search(r"^>.*$", text, re.MULTILINE)
    if header:
        return "homer" if "\t" in header.group(0) else "jaspar"
    return "matrix"


def read_motifs(paths: list[str], fmt: str | None = None) -> list:
    """Motifs from files in any supported format, or CIS-BP archives."""
    import universalmotif as um
    from universalmotif import databases

    readers = {
        "meme": um.read_meme,
        "jaspar": um.read_jaspar,
        "homer": um.read_homer,
        "transfac": um.read_transfac,
        "cisbp": um.read_cisbp,
        "uniprobe": um.read_uniprobe,
        "matrix": um.read_matrix,
    }
    motifs = []
    for path in paths:
        if path != "-" and path.lower().endswith(".zip"):
            motifs += databases.read_cisbp_archive(path)
            continue
        if path != "-" and not Path(path).exists():
            raise CliError(f"no such file: {path}")
        text = _read_bytes(path).decode()
        kind = fmt or (None if path == "-" else _EXTENSIONS.get(_stem_suffix(path))) or _sniff_format(text)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "motifs"
            file.write_text(text)
            motifs += readers[kind](file)
    if not motifs:
        raise CliError("no motifs were read")
    return motifs


def _open_output(path: str | None, binary: bool = False):
    if path in (None, "-"):
        return sys.stdout.buffer if binary else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, "wb" if binary else "wt")
    return open(path, "wb" if binary else "w")


def write_motifs(motifs: list, path: str | None, fmt: str | None) -> None:
    import universalmotif as um

    writers = {
        "meme": um.write_meme,
        "jaspar": um.write_jaspar,
        "homer": um.write_homer,
        "transfac": um.write_transfac,
        "cisbp": um.write_cisbp,
        "uniprobe": um.write_uniprobe,
        "matrix": um.write_matrix,
    }
    kind = fmt or (None if path in (None, "-") else _EXTENSIONS.get(_stem_suffix(path))) or "meme"
    if not motifs:
        raise CliError("no motifs left to write")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "motifs"
        writers[kind](motifs, file)
        text = file.read_text()
    out = _open_output(path)
    try:
        out.write(text)
    finally:
        if out is not sys.stdout:
            out.close()


def read_sequences(path: str) -> dict[str, str]:
    import universalmotif as um

    if path == "-":
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "seqs.fa"
            file.write_bytes(_read_bytes(path))
            return um.read_fasta(file)
    if not Path(path).exists():
        raise CliError(f"no such file: {path}")
    return um.read_fasta(path)


def write_sequences(seqs: dict[str, str], path: str | None, width: int = 80) -> None:
    out = _open_output(path)
    try:
        for name, seq in seqs.items():
            out.write(f">{name}\n")
            for i in range(0, len(seq), width):
                out.write(seq[i:i + width] + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


def open_genome(path: str):
    import universalmotif as um

    if not Path(path).exists():
        raise CliError(f"no such file: {path}")
    return um.Genome(path)


def read_ranges(path: str) -> pl.DataFrame:
    """Ranges from a BED file, or from a tab-separated table with a header
    naming chrom, start and end (or width) columns, with positions counted
    from 0 and the end left out, as in BED files."""
    if path != "-" and not Path(path).exists():
        raise CliError(f"no such file: {path}")
    lines = [ln for ln in _read_bytes(path).decode().splitlines()
             if ln.strip() and not ln.startswith(("#", "track", "browser"))]
    if not lines:
        raise CliError(f"no ranges in {path}")
    first = lines[0].split("\t")
    bed = _stem_suffix(path) in _BED_SUFFIXES or (
        len(first) >= 3 and first[1].isdigit() and first[2].isdigit()
    )
    if bed:
        width = max(len(ln.split("\t")) for ln in lines)
        names = _BED_COLUMNS[:width] + [f"V{i + 1}" for i in range(len(_BED_COLUMNS), width)]
        df = pl.read_csv(io.StringIO("\n".join(lines)), separator="\t", has_header=False,
                         new_columns=names, schema_overrides={"chrom": pl.String},
                         infer_schema_length=10000)
        return df
    return pl.read_csv(io.StringIO("\n".join(lines)), separator="\t", null_values="NA",
                       infer_schema_length=10000)


def read_table(path: str) -> pl.DataFrame:
    if path != "-" and not Path(path).exists():
        raise CliError(f"no such file: {path}")
    text = _read_bytes(path).decode()
    return pl.read_csv(io.StringIO(text), separator="\t", null_values="NA",
                       infer_schema_length=10000, comment_prefix="#")


def write_table(df: pl.DataFrame, path: str | None) -> None:
    """A table as tab-separated text, with list columns joined by commas
    and missing values written as NA."""
    cols = []
    for name, dtype in df.schema.items():
        if dtype == pl.Object:
            continue
        if isinstance(dtype, (pl.List, pl.Array)):
            cols.append(pl.col(name).cast(pl.List(pl.String)).list.join(","))
        else:
            cols.append(pl.col(name))
    text = df.select(cols).write_csv(separator="\t", null_value="NA")
    out = _open_output(path)
    try:
        out.write(text)
    finally:
        if out is not sys.stdout:
            out.close()


def _two_columns(path: str) -> bool:
    """Whether the first data line of a file has two tab-separated columns,
    as chromosome sizes do."""
    for line in _read_bytes(path).decode().splitlines():
        if line.strip() and not line.startswith(("#", "track", "browser")):
            return len(line.rstrip("\r").split("\t")) == 2
    return False


def _chrom_sizes(path: str) -> dict[str, int]:
    """Chromosome lengths from a chrom.sizes or .fai file, or a genome."""
    text = _read_bytes(path).decode()
    first = text.split("\n", 1)[0].split("\t")
    if len(first) >= 2 and first[1].strip().isdigit():
        return {ln.split("\t")[0]: int(ln.split("\t")[1]) for ln in text.splitlines() if ln.strip()}
    return open_genome(path).lengths


# Shared options --------------------------------------------------------------


def _motif_args(p: argparse.ArgumentParser, nargs: str = "+") -> None:
    p.add_argument("motifs", nargs=nargs, metavar="MOTIFS",
                   help="motif files in any supported format, recognized from the extension or "
                        "contents, or CIS-BP archives (.zip); - reads standard input")
    p.add_argument("-f", "--format", choices=_MOTIF_FORMATS, metavar="FORMAT",
                   help="the format of the motif files, if it cannot be recognized: "
                        + _FORMAT_LIST)


def _motif_out_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default="-",
                   help="where to write the motifs (default: standard output)")
    p.add_argument("-t", "--to", choices=_MOTIF_FORMATS, metavar="FORMAT",
                   help=f"output format, one of {_FORMAT_LIST} (default: from the output "
                        "file's extension, or meme)")


def _out_arg(p: argparse.ArgumentParser, what: str = "the table") -> None:
    p.add_argument("-o", "--output", default="-",
                   help=f"where to write {what} (default: standard output)")


def _threads_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-p", "--threads", type=int, default=1,
                   help="threads to use; 0 uses all available (default: 1)")


def _seed_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, help="random seed, for reproducible results")


def _scan_args(p: argparse.ArgumentParser, pvalue: float = 1e-4) -> None:
    p.add_argument("-P", "--pvalue", type=float, default=pvalue,
                   help=f"P-value cutoff for motif hits (default: {pvalue:g})")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="scan only the forward strand")


# Commands -------------------------------------------------------------------


def cmd_convert(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    if a.switch_alph:
        motifs = um.switch_alph(motifs)
    if a.rc:
        motifs = um.motif_rc(motifs)
    if a.type:
        motifs = um.convert_type(motifs, a.type, pseudocount=a.pseudocount,
                                 relative_entropy=a.relative_entropy)
    write_motifs(motifs, a.output, a.to)


def cmd_fetch(a):
    from universalmotif import databases

    if a.database == "jaspar":
        motifs = databases.jaspar(collection=a.collection, tax_group=a.tax_group, name=a.name,
                                  tax_id=a.tax_id, tf_class=a.tf_class, tf_family=a.tf_family,
                                  data_type=a.data_type, release=a.release,
                                  all_versions=a.all_versions, refresh=a.refresh)
    elif a.database == "hocomoco":
        motifs = databases.hocomoco(version=a.version, collection=a.collection,
                                    species=a.species, refresh=a.refresh)
    else:
        motifs = databases.cisbp(a.species, family=a.family, inferred=a.inferred,
                                 refresh=a.refresh)
    write_motifs(motifs, a.output, a.to)


def cmd_summary(a):
    import universalmotif as um

    write_table(um.summarize_motifs(read_motifs(a.motifs, a.format)), a.output)


def cmd_filter(a):
    import universalmotif as um

    motifs = um.filter_motifs(
        read_motifs(a.motifs, a.format), name=a.name, altname=a.altname, family=a.family,
        organism=a.organism, width=a.width, alphabet=a.alphabet, type=a.type,
        icscore=a.icscore, nsites=a.nsites, strand=a.strand, pval=a.pval, qval=a.qval,
        eval=a.eval,
    )
    if a.regex:
        pattern = re.compile(a.regex)
        motifs = [m for m in motifs if pattern.search(m.name) or pattern.search(m.altname or "")]
    write_motifs(motifs, a.output, a.to)


def cmd_trim(a):
    import universalmotif as um

    motifs = um.trim_motifs(read_motifs(a.motifs, a.format), min_ic=a.min_ic,
                            trim_from=a.trim_from)
    write_motifs(motifs, a.output, a.to)


def cmd_merge(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    if a.all:
        merged = [um.merge_motifs(motifs, min_overlap=a.min_overlap, rc=a.rc, new_name=a.name,
                                  weighted=a.weighted, nthreads=a.threads)]
    else:
        kw = {"qvalue": a.qvalue, "null": a.null, "min_overlap": a.min_overlap, "rc": a.rc,
                  "weighted": a.weighted, "nthreads": a.threads}
        merged = um.merge_similar(motifs, **kw)
        if a.clusters:
            write_table(um.merge_similar(motifs, return_clusters=True, **kw), a.clusters)
    write_motifs(merged, a.output, a.to)


def cmd_tree(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    tree = um.motif_tree(motifs, min_overlap=a.min_overlap, rc=a.rc, method=a.method,
                         nthreads=a.threads)
    if a.k is not None or a.height is not None:
        groups = tree.cutree(k=a.k, h=a.height)
        df = pl.DataFrame({"motif": [m.name for m in motifs],
                           "motif_i": list(range(len(motifs))),
                           "cluster": [int(g) for g in groups]})
        write_table(df, a.output)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "tree.nwk"
            tree.write_newick(file)
            text = file.read_text()
        out = _open_output(a.output)
        try:
            out.write(text if text.endswith("\n") else text + "\n")
        finally:
            if out is not sys.stdout:
                out.close()
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        ax = tree.plot(layout=a.layout, tip_logos=a.logos)
        ax.figure.savefig(a.plot, bbox_inches="tight")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_") or "motif"


def cmd_logo(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    if a.each:
        out_dir = Path(a.each)
        out_dir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for m in motifs:
            stem = _safe_name(m.name)
            k, base = 1, stem
            while stem in used:
                stem, k = f"{base}.{k}", k + 1
            used.add(stem)
            path = out_dir / f"{stem}.{a.ext}"
            if a.ext == "svg":
                path.write_text(um.logo_svg(m, use_type=a.type, relative_entropy=a.relative_entropy))
            else:
                fig = um.view_motifs(m, use_type=a.type, relative_entropy=a.relative_entropy)
                fig.savefig(path, bbox_inches="tight", dpi=a.dpi)
                _close_figure(fig)
        return
    if a.output in (None, "-"):
        raise CliError("give a file to write the logos to with -o, or a folder with --each")
    fig = um.view_motifs(motifs, use_type=a.type, rc=a.rc, relative_entropy=a.relative_entropy,
                         sort_by=a.sort_by, nthreads=a.threads)
    fig.savefig(a.output, bbox_inches="tight", dpi=a.dpi)
    _close_figure(fig)


def _close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _target_sequences(a, what: str = "sequences"):
    """Sequences from -s, or from -g and -r as the ranges' sequences."""
    import universalmotif as um

    seqs = getattr(a, "sequences", None)
    genome = getattr(a, "genome", None)
    ranges = getattr(a, "ranges", None)
    if seqs and (genome or ranges):
        raise CliError(f"give the {what} either with -s, or with -g and -r")
    if seqs:
        return read_sequences(seqs)
    if genome and ranges:
        return um.extract_ranges(open_genome(genome), read_ranges(ranges), nthreads=a.threads)
    raise CliError(f"give the {what} with -s, or with -g and -r")


def cmd_scan(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    kw = {"pvalue": a.pvalue, "rc": a.rc, "nthreads": a.threads, "no_overlaps": a.no_overlaps,
              "no_overlaps_by": a.overlaps_by, "no_overlaps_by_strand": a.across_strands,
              "no_overlaps_by_motif": a.across_motifs, "use_gaps": a.gaps, "correct_gaps": a.correct_gaps}
    if a.sequences and (a.genome or a.ranges):
        raise CliError("give the sequences either with -s, or with -g (and optionally -r)")
    if a.sequences:
        if a.sequences == "-":
            hits = um.scan_sequences(motifs, read_sequences("-"), **kw)
        else:
            if not Path(a.sequences).exists():
                raise CliError(f"no such file: {a.sequences}")
            # Read and scanned a batch at a time.
            hits = um.scan_sequences(motifs, Path(a.sequences), **kw)
        if a.genomic:
            hits = um.genomic_hits(hits)
    elif a.genome:
        genome = open_genome(a.genome)
        if a.ranges:
            hits = um.scan_sequences(motifs, um.extract_ranges(genome, read_ranges(a.ranges),
                                                               nthreads=a.threads), **kw)
            if not a.local:
                hits = um.genomic_hits(hits)
        else:
            hits = um.scan_sequences(motifs, genome, **kw)
    else:
        raise CliError("give the sequences to scan with -s or -g")
    write_table(hits, a.output)


def cmd_compare(a):
    import numpy as np

    import universalmotif as um

    queries = read_motifs(a.motifs, a.format)
    kw = {"min_overlap": a.min_overlap, "rc": a.rc, "null": a.null, "nthreads": a.threads,
              "complete": a.complete}
    qvalue = 1.0 if a.all else a.qvalue
    if a.matrix:
        if a.db:
            raise CliError("--matrix compares the motifs given with each other, not with --db")
        mat = np.asarray(um.compare_motifs(queries, matrix_out=a.matrix, **kw))
        names = [m.name for m in queries]
        df = pl.DataFrame({"motif": names, **{n: mat[:, j] for j, n in
                                              enumerate(_unique(names))}})
        write_table(df, a.output)
        return
    nq = len(queries)
    if a.db:
        targets = read_motifs(a.db, a.db_format)
        res = um.compare_motifs(queries + targets, compare_to=list(range(nq)), qvalue=qvalue, **kw)
        res = res.filter(pl.col("target_i") >= nq).with_columns(pl.col("target_i") - nq)
        # Names made unique across queries and database go back to their own.
        res = res.with_columns(
            pl.col("subject_i").map_elements(lambda i: queries[i].name, return_dtype=pl.String)
            .alias("subject"),
            pl.col("target_i").map_elements(lambda i: targets[i].name, return_dtype=pl.String)
            .alias("target"),
        )
    else:
        res = um.compare_motifs(queries, compare_to=list(range(nq)), qvalue=qvalue, **kw)
        res = res.filter(pl.col("subject_i") != pl.col("target_i"))
    write_table(res.rename({"subject": "query", "subject_i": "query_i",
                            "subject_consensus": "query_consensus"}), a.output)


def _unique(names: list[str]) -> list[str]:
    from universalmotif._analysis import _make_unique

    return _make_unique(names)


def cmd_enrich(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    targets = _target_sequences(a, "target sequences")
    bkg = read_sequences(a.bkg) if a.bkg else None
    res = um.enrich_motifs(motifs, targets, bkg, pvalue=a.pvalue, qvalue=a.qvalue, test=a.test,
                           rc=a.rc, shuffle_k=a.shuffle_k, seed=a.seed, nthreads=a.threads)
    write_table(res, a.output)


def cmd_find(a):
    import universalmotif as um

    targets = _target_sequences(a, "target sequences")
    bkg = read_sequences(a.bkg) if a.bkg else None
    res = um.motif_finder(targets, bkg, min_width=a.min_width, max_width=a.max_width,
                          nmotifs=a.nmotifs, hit_pvalue=a.hit_pvalue, stop_pvalue=a.stop_pvalue,
                          qvalue=a.qvalue, rc=a.rc, shuffle_k=a.shuffle_k, seed=a.seed,
                          nthreads=a.threads)
    if a.table:
        write_table(res, a.table)
    if res.height == 0:
        print("no motifs were found", file=sys.stderr)
        return
    write_motifs(res["motif"].to_list(), a.output, a.to)


def cmd_shuffle(a):
    import universalmotif as um

    seqs = read_sequences(a.sequences)
    out = um.shuffle_sequences(seqs, k=a.k, method=a.method, seed=a.seed, nthreads=a.threads)
    write_sequences(out, a.output)


def cmd_extract(a):
    import universalmotif as um

    seqs = um.extract_ranges(open_genome(a.genome), read_ranges(a.ranges), nthreads=a.threads)
    write_sequences(seqs, a.output, a.width)


def _bigwig_map(values: list[str]) -> dict[str, str]:
    out = {}
    for i, v in enumerate(values):
        name, sep, path = v.partition("=")
        if not sep:
            name, path = (Path(v).name.split(".")[0] or f"signal{i + 1}"), v
        if name in out:
            raise CliError(f"the bigWig name {name!r} is used twice; name them with NAME=PATH")
        out[name] = path
    return out


def cmd_bkg(a):
    import universalmotif as um

    genome = open_genome(a.genome) if a.genome else None
    ranges = read_ranges(a.ranges) if a.ranges else None
    if a.sequences and ranges is not None:
        raise CliError("give the targets either with -s, or with -g and -r")
    if a.sequences:
        targets = read_sequences(a.sequences)
    elif ranges is not None and genome is not None:
        targets = um.extract_ranges(genome, ranges, nthreads=a.threads)
    else:
        raise CliError("give the targets with -s, or with -g and -r")
    if (a.universe is None) == (genome is None):
        raise CliError("give exactly one of -u (a universe of sequences) and -g (a genome)")
    exclude = None
    if genome is not None:
        parts = [r.select("chrom", "start", "end") for r in
                 ([ranges] if ranges is not None else []) +
                 ([read_ranges(x) for x in a.exclude] if a.exclude else [])]
        exclude = pl.concat(parts) if parts else None
    bigwig = _bigwig_map(a.bigwig) if a.bigwig else None
    if bigwig and ranges is None:
        raise CliError("matching on bigWig signal needs the targets as ranges, with -g and -r")
    kw = {"n_per_target": a.n, "n_bins_gc": a.gc_bins, "n_bins_length": a.length_bins,
              "unique": not a.repeats, "seed": a.seed}
    if genome is not None:
        kw.update(genome=genome, exclude=exclude, n_candidates=a.candidates,
                  chromosomes=a.chromosomes, bigwig=bigwig,
                  target_ranges=ranges if bigwig else None,
                  n_bins_covariates=a.covariate_bins, bin_type_covariates=a.covariate_bin_type)
    else:
        kw.update(universe=read_sequences(a.universe))
    if a.table:
        write_table(um.match_bkg(targets, return_indices=True, **kw), a.table)
    write_sequences(um.match_bkg(targets, **kw), a.output)


def cmd_matrix(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    seqs = _target_sequences(a, "peak sequences")
    m = um.match_motifs(motifs, seqs, pvalue=a.pvalue, value=a.value, rc=a.rc,
                        no_overlaps=a.no_overlaps, nthreads=a.threads)
    out = a.output
    suffix = _stem_suffix(out) if out not in (None, "-") else ""
    if suffix == ".h5ad":
        m.to_anndata().write_h5ad(out)
    elif suffix == ".mtx":
        import scipy.io

        stem = out[:-4] if out.endswith(".mtx") else out[:-7]
        scipy.io.mmwrite(out, m.matrix)
        Path(stem + ".sequences.txt").write_text("".join(f"{s}\n" for s in m.sequences))
        Path(stem + ".motifs.txt").write_text("".join(f"{s}\n" for s in m.motifs))
    else:
        write_table(m.to_polars(), out)


def cmd_proximity(a):
    import universalmotif as um

    anchors = read_ranges(a.anchors)
    kw = {"method": a.method, "count": a.count, "orientation": a.orientation, "rc": a.rc,
              "scan_pvalue": a.pvalue, "dist_thresholds": a.distances, "n_perm": a.n_perm,
              "perm_per_chrom": not a.global_perm, "ignore_strand": not a.stranded,
              "qvalue": a.qvalue, "seed": a.seed, "nthreads": a.threads}
    universe = None
    if a.universe:
        u = a.universe
        # A .txt file can hold chromosome sizes or ranges, so it is told
        # apart by its number of columns.
        sizes_like = (_stem_suffix(u) in (".fai", ".sizes") or u.endswith("chrom.sizes")
                      or (_stem_suffix(u) == ".txt" and _two_columns(u)))
        universe = _chrom_sizes(u) if sizes_like or _is_genome(u) else read_ranges(u)
    if a.hits:
        if a.motifs:
            raise CliError("give either motifs to scan for or --hits, not both")
        hits = read_table(a.hits)
        if universe is None and a.genome:
            universe = open_genome(a.genome).lengths
        res = um.motif_proximity(hits=hits, anchors=anchors, universe=universe, **kw)
    else:
        if not a.motifs:
            raise CliError("give motifs to scan for, or a table of hits with --hits")
        if a.genome and a.sequences:
            raise CliError("give the genome with -g or -s, not both")
        if a.genome:
            seqs = open_genome(a.genome)
        elif a.sequences:
            seqs = read_sequences(a.sequences)
        else:
            raise CliError("give the genome to scan with -g (or -s for a FASTA file of chromosomes)")
        res = um.motif_proximity(anchors=anchors, motifs=read_motifs(a.motifs, a.format),
                                 sequences=seqs, universe=universe, **kw)
    if not a.nearest:
        res = res.drop("dist_to_nearest")
    write_table(res, a.output)


def _is_genome(path: str) -> bool:
    try:
        raw = Path(path).read_bytes()[:4]
    except OSError:
        return False
    return raw[:1] == b">" or raw[:2] == b"\x1f\x8b" or raw in (b"\x43\x27\x41\x1a", b"\x1a\x41\x27\x43")


def cmd_coocc(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    kw = {"pvalue": a.pvalue, "rc": a.rc, "no_overlaps": a.no_overlaps, "max_distance": a.max_distance,
              "min_coocc": a.min_coocc, "pseudocount": a.pseudocount, "self_pairs": a.self_pairs,
              "nthreads": a.threads}
    if a.hits:
        if a.n_sequences is None:
            raise CliError("--hits needs --n-sequences, the number of sequences scanned")
        res = um.motif_coocc(motifs, hits=read_table(a.hits), n_sequences=a.n_sequences, **kw)
    else:
        res = um.motif_coocc(motifs, _target_sequences(a), **kw)
    write_table(res, a.output)


def cmd_posenr(a):
    import universalmotif as um

    if a.hits:
        if a.motifs:
            raise CliError("give either motifs to scan for or --hits, not both")
        hits = read_table(a.hits)
        length = a.seq_length
        if length is None:
            raise CliError("--hits needs --seq-length, the length of the sequences scanned")
    else:
        if not a.motifs:
            raise CliError("give motifs to scan for, or a table of hits with --hits")
        seqs = _target_sequences(a)
        lengths = {len(s) for s in seqs.values()}
        length = a.seq_length
        if length is None:
            if len(lengths) != 1:
                raise CliError("the sequences differ in length; give the length to use with "
                               "--seq-length")
            length = lengths.pop()
        hits = um.scan_sequences(read_motifs(a.motifs, a.format), seqs, pvalue=a.pvalue, rc=a.rc,
                                 nthreads=a.threads)
    res = um.motif_peaks(hits, length, mode=a.mode, qvalue=a.qvalue, min_window=a.min_window,
                         max_window=a.max_window, window_step=a.window_step,
                         position_step=a.position_step, nthreads=a.threads)
    if not a.centres:
        res = res.drop("centres")
    write_table(res, a.output)


# The parser -----------------------------------------------------------------


def _add_target_args(p: argparse.ArgumentParser, what: str, genome_help: str | None = None) -> None:
    p.add_argument("-s", "--sequences", help=f"FASTA file of {what}")
    p.add_argument("-g", "--genome", help=genome_help or "genome (FASTA, bgzipped FASTA or 2bit) "
                                                         f"to take the {what} from, with -r")
    p.add_argument("-r", "--ranges", help=f"BED file or table of the ranges of the {what}, "
                                          "with -g")


class _formatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32)
        # Leave room for the longest subcommand name, which is not measured
        # when the subcommands are added.
        self._action_max_length = 15


def build_parser(prog: str = "universalmotif") -> argparse.ArgumentParser:
    from universalmotif import __version__

    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=_formatter,
        description="Work with sequence motifs from the command line: convert, fetch and curate "
                    "motifs, scan and test sequences, and draw logos.",
    )
    parser.add_argument("--version", action="version", version=f"universalmotif {__version__}")
    parser.add_argument("--debug", action="store_true",
                        help="show the full traceback when something fails")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name, fn, help_text, description=None):
        description = description or help_text[0].upper() + help_text[1:] + "."
        p = sub.add_parser(name, help=help_text, description=description,
                           formatter_class=_formatter)
        p.set_defaults(fn=fn)
        return p

    # Motifs
    p = command("convert", cmd_convert, "convert motifs between formats and types")
    _motif_args(p)
    _motif_out_args(p)
    p.add_argument("--type", choices=("PCM", "PPM", "PWM", "ICM"),
                   help="convert the motifs to this type")
    p.add_argument("--pseudocount", type=float, help="pseudocount to use for the conversion")
    p.add_argument("--relative-entropy", action="store_true",
                   help="with --type ICM, use relative entropy against the background")
    p.add_argument("--rc", action="store_true", help="reverse complement the motifs")
    p.add_argument("--switch-alph", action="store_true",
                   help="switch DNA motifs to RNA and RNA motifs to DNA")

    p = command("fetch", cmd_fetch, "download motifs from JASPAR, HOCOMOCO or CIS-BP",
                "Download motifs from a database, keeping a local cache.")
    dbs = p.add_subparsers(dest="database", metavar="DATABASE", required=True)
    j = dbs.add_parser("jaspar", help="motifs from the JASPAR REST API",
                       description="Download motifs from the JASPAR REST API.",
                       formatter_class=_formatter)
    j.add_argument("--collection", default="CORE", help="collection (default: CORE)")
    j.add_argument("--tax-group", help="taxonomic group, such as plants or vertebrates")
    j.add_argument("--name", help="transcription factor name")
    j.add_argument("--tax-id", type=int, help="NCBI taxonomy ID of the species")
    j.add_argument("--tf-class", help="transcription factor class")
    j.add_argument("--tf-family", help="transcription factor family")
    j.add_argument("--data-type", help="experiment type, such as ChIP-seq")
    j.add_argument("--release", type=int, help="JASPAR release, such as 2024")
    j.add_argument("--all-versions", action="store_true",
                   help="keep every version of each matrix, not only the latest")
    h = dbs.add_parser("hocomoco", help="motifs from HOCOMOCO",
                       description="Download motifs from HOCOMOCO, for human and mouse.",
                       formatter_class=_formatter)
    h.add_argument("--version", type=int, default=12, choices=(11, 12),
                   help="HOCOMOCO version (default: 12)")
    h.add_argument("--collection", default="CORE",
                   help="collection: CORE, INVIVO, INVITRO or RSNP for version 12, and CORE or "
                        "FULL for version 11 (default: CORE)")
    h.add_argument("--species", choices=("HUMAN", "MOUSE"),
                   help="keep only motifs from human or mouse data")
    c = dbs.add_parser("cisbp", help="motifs from CIS-BP",
                       description="Download the motifs of one species from CIS-BP.",
                       formatter_class=_formatter)
    c.add_argument("species", help="species, as named by CIS-BP, such as Arabidopsis_thaliana")
    c.add_argument("--family", action="append", help="keep this TF family (repeatable)")
    c.add_argument("--inferred", action="store_true",
                   help="include motifs inferred from similar proteins")
    for d in (j, h, c):
        d.add_argument("--refresh", action="store_true",
                       help="download again rather than using the cache")
        _motif_out_args(d)

    p = command("summary", cmd_summary, "summarize motifs as a table",
                "Summarize motifs as a table, one row per motif.")
    _motif_args(p)
    _out_arg(p)

    p = command("filter", cmd_filter, "keep the motifs that pass every filter given")
    _motif_args(p)
    _motif_out_args(p)
    for opt in ("name", "altname", "family", "organism", "alphabet", "type", "strand"):
        p.add_argument(f"--{opt}", action="append",
                       help=f"keep motifs with this {opt} (repeatable)")
    p.add_argument("--regex", help="keep motifs whose name or altname matches this pattern")
    p.add_argument("--width", type=int, help="minimum width")
    p.add_argument("--icscore", type=float, help="minimum total information content")
    p.add_argument("--nsites", type=float, help="minimum number of sites")
    p.add_argument("--pval", type=float, help="maximum P-value")
    p.add_argument("--qval", type=float, help="maximum q-value")
    p.add_argument("--eval", type=float, help="maximum E-value")

    p = command("trim", cmd_trim, "trim low-information positions from motif edges")
    _motif_args(p)
    _motif_out_args(p)
    p.add_argument("--min-ic", type=float, default=0.25,
                   help="positions below this information content are trimmed (default: 0.25)")
    p.add_argument("--trim-from", choices=("both", "left", "right"), default="both",
                   help="which edges to trim (default: both)")

    p = command("merge", cmd_merge, "merge similar motifs",
                "Merge motifs that are significantly similar into one motif per cluster, or "
                "merge all of them into one with --all.")
    _motif_args(p)
    _motif_out_args(p)
    p.add_argument("-q", "--qvalue", type=float, default=0.01,
                   help="q-value below which motifs count as similar (default: 0.01)")
    p.add_argument("--null", choices=("empirical", "parametric"), default="empirical",
                   help="null model for the similarity P-values (default: empirical)")
    p.add_argument("--min-overlap", type=int, default=5,
                   help="minimum overlap of aligned motifs (default: 5)")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not align motifs as reverse complements")
    p.add_argument("--weighted", action="store_true", help="weight motifs by their nsites")
    p.add_argument("--all", action="store_true", help="merge all the motifs into one")
    p.add_argument("--name", help="name of the merged motif, with --all")
    p.add_argument("--clusters", help="also write a table of which motifs were merged")
    _threads_arg(p)

    p = command("tree", cmd_tree, "cluster motifs into a tree",
                "Cluster motifs by similarity and write the tree in Newick format, or with -k "
                "or --height a table of clusters.")
    _motif_args(p)
    _out_arg(p, "the tree or clusters")
    p.add_argument("--method", default="complete",
                   choices=("complete", "average", "single", "ward.D", "ward.D2", "mcquitty",
                            "median", "centroid"),
                   help="linkage method (default: complete)")
    p.add_argument("--min-overlap", type=int, default=6,
                   help="minimum overlap of aligned motifs (default: 6)")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not align motifs as reverse complements")
    p.add_argument("-k", type=int, help="cut the tree into this many clusters")
    p.add_argument("--height", type=float, help="cut the tree at this height")
    p.add_argument("--plot", help="also draw the tree to this file (PNG, PDF or SVG)")
    p.add_argument("--layout", choices=("rectangular", "slanted", "circular", "fan"),
                   default="rectangular",
                   help="layout of the drawn tree (default: rectangular)")
    p.add_argument("--logos", action="store_true", help="draw logos at the tips")
    _threads_arg(p)

    p = command("logo", cmd_logo, "draw sequence logos",
                "Draw motifs as sequence logos, aligned in one figure written with -o, or one "
                "file per motif in a folder with --each.")
    _motif_args(p)
    p.add_argument("-o", "--output", help="file for one figure of all the motifs (PNG, PDF, SVG)")
    p.add_argument("--each", metavar="FOLDER", help="write one logo per motif into this folder")
    p.add_argument("--ext", choices=("svg", "png", "pdf"), default="svg",
                   help="file type for --each (default: svg, which needs no extra packages)")
    p.add_argument("--type", choices=("ICM", "PPM", "PWM", "PCM"), default="ICM",
                   help="what the letter heights show (default: ICM)")
    p.add_argument("--relative-entropy", action="store_true",
                   help="information content relative to the background")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not reverse complement motifs to align them")
    p.add_argument("--sort-by", choices=("none", "ic", "similarity"), default="none",
                   help="order of the motifs in the figure: as given, by decreasing information "
                        "content, or clustered by similarity (default: none)")
    p.add_argument("--dpi", type=int, default=150, help="resolution of PNG files (default: 150)")
    _threads_arg(p)

    # Sequences
    p = command("scan", cmd_scan, "scan sequences for motif hits",
                "Scan sequences for motif hits. Sequences come from a FASTA file with -s, which "
                "is read a batch at a time, from a whole genome with -g, or from ranges of a "
                "genome with -g and -r, whose hits are reported in genomic coordinates.")
    _motif_args(p)
    _add_target_args(p, "sequences", genome_help="genome (FASTA, bgzipped FASTA or 2bit) to scan "
                                                 "whole, or in parts with -r")
    _out_arg(p, "the hits")
    _scan_args(p)
    p.add_argument("--no-overlaps", action="store_true",
                   help="remove overlapping hits of each motif, keeping the best")
    p.add_argument("--overlaps-by", choices=("pvalue", "score"), default="pvalue",
                   help="what makes the best hit with --no-overlaps (default: pvalue)")
    p.add_argument("--across-strands", action="store_true",
                   help="with --no-overlaps, compare hits on both strands")
    p.add_argument("--across-motifs", action="store_true",
                   help="with --no-overlaps, compare hits of different motifs")
    p.add_argument("--no-gaps", dest="gaps", action="store_false",
                   help="ignore the gaps of gapped motifs")
    p.add_argument("--correct-gaps", action="store_true",
                   help="correct gapped motifs' P-values for the gap sizes tried")
    p.add_argument("--local", action="store_true",
                   help="with -r, report positions within each range, not on the genome")
    p.add_argument("--genomic", action="store_true",
                   help="with -s, read each sequence's place in the genome from names like "
                        "chrom:start-end and report hits in genomic coordinates")
    _threads_arg(p)

    p = command("compare", cmd_compare, "compare motifs with each other or with a database",
                "Compare motifs with each other, or with a database of motifs given with --db, "
                "reporting the matches that pass a q-value cutoff.")
    _motif_args(p)
    _out_arg(p)
    p.add_argument("--db", nargs="+", help="motif files to compare the motifs with")
    p.add_argument("--db-format", choices=_MOTIF_FORMATS, metavar="FORMAT",
                   help="format of the --db files")
    p.add_argument("-q", "--qvalue", type=float, default=0.1,
                   help="q-value cutoff for reported matches (default: 0.1)")
    p.add_argument("--all", action="store_true", help="report every pair, whatever its q-value")
    p.add_argument("--min-overlap", type=int, default=5,
                   help="minimum overlap of aligned motifs (default: 5)")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not align motifs as reverse complements")
    p.add_argument("--null", choices=("empirical", "parametric"), default="empirical",
                   help="null model for the P-values (default: empirical)")
    p.add_argument("--complete", action="store_true",
                   help="score whole queries rather than only the overlap")
    p.add_argument("--matrix", choices=("score", "pvalue", "qvalue"),
                   help="write an all-against-all matrix of this value instead")
    _threads_arg(p)

    p = command("enrich", cmd_enrich, "test motifs for enrichment in target sequences",
                "Test motifs for enrichment in target sequences against background sequences, "
                "or against shuffled targets when no background is given.")
    _motif_args(p)
    _add_target_args(p, "target sequences")
    p.add_argument("-b", "--bkg", help="FASTA file of background sequences")
    _out_arg(p)
    _scan_args(p)
    p.add_argument("-q", "--qvalue", type=float, default=0.1,
                   help="q-value cutoff for reported motifs (default: 0.1)")
    p.add_argument("--test", choices=("seqs", "hits"), default="seqs",
                   help="count sequences with hits, or hits (default: seqs)")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    _seed_arg(p)
    _threads_arg(p)

    p = command("find", cmd_find, "discover motifs de novo",
                "Discover motifs enriched in target sequences, writing the motifs with -o and "
                "a table of results with --table.")
    _add_target_args(p, "target sequences")
    p.add_argument("-b", "--bkg", help="FASTA file of background sequences")
    _motif_out_args(p)
    p.add_argument("--table", help="also write a table of the motifs found")
    p.add_argument("--min-width", type=int, default=6, help="smallest width (default: 6)")
    p.add_argument("--max-width", type=int, default=15, help="largest width (default: 15)")
    p.add_argument("--nmotifs", type=int, default=10,
                   help="most motifs to find per width (default: 10)")
    p.add_argument("--hit-pvalue", type=float, default=1e-4,
                   help="P-value of a site counting as a hit (default: 1e-4)")
    p.add_argument("--stop-pvalue", type=float, default=1e-3,
                   help="enrichment P-value at which to stop (default: 1e-3)")
    p.add_argument("-q", "--qvalue", type=float, default=0.05,
                   help="q-value cutoff for reported motifs (default: 0.05)")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="search only the forward strand")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    _seed_arg(p)
    _threads_arg(p)

    p = command("shuffle", cmd_shuffle, "shuffle sequences, keeping k-mer counts")
    p.add_argument("-s", "--sequences", required=True, help="FASTA file of sequences")
    _out_arg(p, "the shuffled sequences")
    p.add_argument("-k", type=int, default=1, help="size of the k-mers kept (default: 1)")
    p.add_argument("--method", choices=("euler", "linear", "markov"), default="euler",
                   help="shuffling method (default: euler)")
    _seed_arg(p)
    _threads_arg(p)

    p = command("extract", cmd_extract, "extract the sequences of genomic ranges",
                "Write the sequences of genomic ranges as FASTA, reverse complementing those "
                "on the minus strand.")
    p.add_argument("-g", "--genome", required=True,
                   help="genome (FASTA, bgzipped FASTA or 2bit)")
    p.add_argument("-r", "--ranges", required=True, help="BED file or table of ranges")
    _out_arg(p, "the sequences")
    p.add_argument("--width", type=int, default=80, help="FASTA line width (default: 80)")
    _threads_arg(p)

    p = command("bkg", cmd_bkg, "draw background sequences matched to targets",
                "Draw background sequences matched to the targets on GC content and length, "
                "and optionally on bigWig signal, from a universe of sequences or from random "
                "windows of a genome.")
    p.add_argument("-s", "--sequences", help="FASTA file of target sequences")
    p.add_argument("-r", "--ranges", help="BED file or table of the targets' ranges, with -g; "
                                          "windows overlapping them are not drawn")
    p.add_argument("-g", "--genome", help="genome to draw windows from")
    p.add_argument("-u", "--universe", help="FASTA file of candidate background sequences")
    _out_arg(p, "the background sequences")
    p.add_argument("--table", help="also write a table of the matches")
    p.add_argument("-n", type=int, default=1, help="background sequences per target (default: 1)")
    p.add_argument("--exclude", action="append", help="BED file of ranges to avoid (repeatable)")
    p.add_argument("--chromosomes", nargs="+", help="chromosomes to draw windows from")
    p.add_argument("--candidates", type=int,
                   help="genome windows to draw candidates from (default: max(10000, 10 x targets))")
    p.add_argument("--gc-bins", type=int, default=20, help="GC content bins (default: 20)")
    p.add_argument("--length-bins", type=int, default=10, help="length bins (default: 10)")
    p.add_argument("--bigwig", action="append", metavar="[NAME=]PATH",
                   help="also match on the mean signal of this bigWig file (repeatable)")
    p.add_argument("--covariate-bins", type=int, default=10,
                   help="bins for each bigWig signal (default: 10)")
    p.add_argument("--covariate-bin-type", choices=("quantile", "equalwidth"),
                   default="quantile", help="how bigWig signal is binned (default: quantile)")
    p.add_argument("--repeats", action="store_true",
                   help="allow a background sequence to be drawn more than once")
    _seed_arg(p)
    _threads_arg(p)

    p = command("matrix", cmd_matrix, "make a peak by motif matrix of matches",
                "Make a sparse peak by motif matrix of motif matches, hit counts or best "
                "scores, written as .h5ad, as Matrix Market (.mtx, with the row and column "
                "names alongside), or otherwise as a long table.")
    _motif_args(p)
    _add_target_args(p, "peaks")
    _out_arg(p, "the matrix")
    _scan_args(p, 5e-5)
    p.add_argument("--value", choices=("matches", "counts", "scores"), default="matches",
                   help="what the matrix holds (default: matches)")
    p.add_argument("--no-overlaps", action="store_true",
                   help="count overlapping hits of a motif once")
    _threads_arg(p)

    # Statistics on hits
    p = command("proximity", cmd_proximity, "test whether motif hits sit near anchors",
                "Test whether motif hits sit closer to genomic anchors than chance, scanning "
                "a genome for the motifs or using a table of hits.")
    _motif_args(p, nargs="*")
    p.add_argument("-a", "--anchors", required=True, help="BED file or table of anchors")
    p.add_argument("-g", "--genome", help="genome to scan, or whose lengths make the universe")
    p.add_argument("-s", "--sequences", help="FASTA file of chromosomes to scan instead")
    p.add_argument("--hits", help="table of hits, such as from scan -g, instead of scanning")
    p.add_argument("--universe",
                   help="regions hits could fall in: a BED file, chrom.sizes or .fai file, or a "
                        "genome (default: the whole genome)")
    _out_arg(p)
    _scan_args(p)
    p.add_argument("--method", choices=("binomial", "permutation", "ks"), default="binomial",
                   help="null model (default: binomial)")
    p.add_argument("--count", choices=("hits", "anchors"), default="hits",
                   help="count hits near anchors, or anchors with hits (default: hits)")
    p.add_argument("--orientation", choices=("undirected", "upstream", "downstream"),
                   default="undirected", help="zone around stranded anchors (default: undirected)")
    p.add_argument("--distances", type=lambda s: [int(x) for x in s.split(",")],
                   help="comma-separated distances to test")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="permutations with --method permutation (default: 1000)")
    p.add_argument("--global-perm", action="store_true",
                   help="move anchors anywhere in the universe, not within their chromosome")
    p.add_argument("--stranded", action="store_true",
                   help="count overlapping hits on the two strands separately")
    p.add_argument("-q", "--qvalue", type=float, default=0.1,
                   help="q-value cutoff for reported motifs (default: 0.1)")
    p.add_argument("--nearest", action="store_true",
                   help="include each hit's distance to its nearest anchor")
    _seed_arg(p)
    _threads_arg(p)

    p = command("coocc", cmd_coocc, "test pairs of motifs for co-occurrence",
                "Test pairs of motifs for co-occurring in the same sequences more often than "
                "chance.")
    _motif_args(p)
    _add_target_args(p, "sequences")
    p.add_argument("--hits", help="table of hits instead of scanning, with --n-sequences")
    p.add_argument("--n-sequences", type=int, help="number of sequences the hits came from")
    _out_arg(p)
    _scan_args(p)
    p.add_argument("--no-overlaps", action="store_true",
                   help="remove overlapping hits of each motif before counting")
    p.add_argument("--max-distance", type=int,
                   help="also report how often pairs of hits are within this distance")
    p.add_argument("--min-coocc", type=int, default=1,
                   help="test only pairs sharing at least this many sequences (default: 1)")
    p.add_argument("--pseudocount", type=int, default=0,
                   help="pseudocount added to every cell of the 2x2 table (default: 0)")
    p.add_argument("--self-pairs", action="store_true",
                   help="also count repeated sites of a single motif")
    _threads_arg(p)

    p = command("posenr", cmd_posenr, "test motifs for positional enrichment",
                "Test whether motif hits gather at the centre of the sequences, or anywhere "
                "along them with --mode local, as around peak summits.")
    _motif_args(p, nargs="*")
    _add_target_args(p, "sequences")
    p.add_argument("--hits", help="table of hits instead of scanning, with --seq-length")
    p.add_argument("--seq-length", type=int,
                   help="length of the sequences (default: their length, if all the same)")
    _out_arg(p)
    _scan_args(p)
    p.add_argument("--mode", choices=("central", "local"), default="central",
                   help="windows centred on the sequences, or anywhere along them "
                        "(default: central)")
    p.add_argument("-q", "--qvalue", type=float, default=0.1,
                   help="q-value cutoff for reported motifs (default: 0.1)")
    p.add_argument("--min-window", type=int, default=10, help="smallest window (default: 10)")
    p.add_argument("--max-window", type=int,
                   help="largest window (default: half the sequence length)")
    p.add_argument("--window-step", type=int, default=10, help="window size step (default: 10)")
    p.add_argument("--position-step", type=int, default=5,
                   help="window position step with --mode local (default: 5)")
    p.add_argument("--centres", action="store_true",
                   help="include the centres of the hits used for each motif")
    _threads_arg(p)

    return parser


def _show_warning(message, category, filename, lineno, file=None, line=None):
    print(f"warning: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "universalmotif"
    if prog not in ("universalmotif", "umotif"):
        prog = "universalmotif"
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    shown = warnings.showwarning
    warnings.showwarning = _show_warning
    try:
        args.fn(args)
    except BrokenPipeError:
        # The reader went away, as with `| head`; stop quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        if args.debug:
            raise
        if not isinstance(e, (CliError, ValueError, OSError, ImportError, KeyError)):
            raise
        msg = str(e).strip("'\"") or type(e).__name__
        print(f"{prog} {args.command}: error: {msg}", file=sys.stderr)
        return 1
    finally:
        warnings.showwarning = shown
    return 0


if __name__ == "__main__":
    sys.exit(main())
