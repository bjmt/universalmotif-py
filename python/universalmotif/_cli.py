"""The ``universalmotif`` (or ``umotif``) command line.

Each subcommand reads motifs, sequences, genomes and ranges from files and
calls the package function of the same purpose. It writes motifs, FASTA or
tab-separated tables to a file or to standard output.
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

_MOTIF_FORMATS = ("meme", "jaspar", "homer", "transfac", "cisbp", "uniprobe", "matrix", "universalmotif")
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
    ".yml": "universalmotif",
    ".yaml": "universalmotif",
}
_FORMAT_LIST = ", ".join(_MOTIF_FORMATS)
_FORMAT_NAMES = {"meme": "MEME", "jaspar": "JASPAR", "homer": "HOMER", "transfac": "TRANSFAC",
                 "cisbp": "CIS-BP", "uniprobe": "UniPROBE", "matrix": "a plain matrix",
                 "universalmotif": "universalmotif (the R package's format)"}
_BED_SUFFIXES = (".bed", ".narrowpeak", ".broadpeak", ".bedgraph")


class CliError(Exception):
    """An error to report as a message, without a traceback."""


# Reading and writing --------------------------------------------------------


def _stem_suffix(path: str) -> str:
    """The extension of a path, looking past a final .gz."""
    p = path.lower()
    p = p.removesuffix(".gz")
    return os.path.splitext(p)[1]


def _read_bytes(path: str) -> bytes:
    from ._io import _gunzip

    raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    return _gunzip(raw) if raw[:2] == b"\x1f\x8b" else raw


def _sniff_format(text: str) -> str:
    """The motif format of a file, judged from its contents."""
    if re.search(r"^MEME version", text, re.MULTILINE):
        return "meme"
    if re.search(r"^# universalmotif version", text, re.MULTILINE):
        return "universalmotif"
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
        "universalmotif": um.read_universalmotif,
    }
    motifs = []
    for path in paths:
        if path != "-" and path.lower().endswith(".zip"):
            motifs += databases.read_cisbp_archive(path)
            continue
        if path != "-" and not Path(path).exists():
            raise CliError(f"no such file: {path}")
        text = _read_bytes(path).decode()
        kind, how = fmt, None
        if kind is None and path != "-":
            kind, how = _EXTENSIONS.get(_stem_suffix(path)), "its extension"
        if kind is None:
            kind, how = _sniff_format(text), "its contents"
        # A guessed format is named in errors. Otherwise a wrong guess shows
        # only as a puzzling failure to read.
        hint = "" if how is None else f" (read as {_FORMAT_NAMES[kind]} from {how}; give -f if that is wrong)"
        name = "standard input" if path == "-" else path
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "motifs"
            file.write_text(text)
            try:
                found = readers[kind](file)
            except (ValueError, KeyError) as e:
                if not hint:
                    raise
                raise CliError(f"{name}: {_unquote(str(e))}{hint}") from None
        if not found and hint:
            raise CliError(f"no motifs were read from {name}{hint}")
        motifs += found
    if not motifs:
        raise CliError("no motifs were read")
    return motifs


def _open_output(path: str | None, binary: bool = False):
    if path in (None, "-"):
        return sys.stdout.buffer if binary else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, "wb" if binary else "wt", compresslevel=6)
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
        "universalmotif": um.write_universalmotif,
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
    """Ranges from a BED file or a tab-separated table.

    A table needs a header naming chrom, start and end (or width) columns.
    Positions count from 0 with the end left out, as in BED files."""
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
        from ._intervals import _bed_format, _bed_frame

        fmt = "bed" if path == "-" else _bed_format(path, None)
        try:
            return _bed_frame("\n".join(lines), fmt, path)
        except ValueError as e:
            raise CliError(str(e)) from None
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
                   help="motif files in any supported format, or CIS-BP archives (.zip); the "
                        "format is recognized from the extension or contents, and - reads "
                        "standard input")
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
    p.add_argument("-p", "--threads", type=int,
                   help="threads to use; 0 uses all available (default: 1, or "
                        "UNIVERSALMOTIF_NTHREADS if set)")


def _progress_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None,
                   help="show a progress bar on stderr for work that takes over a second "
                        "(default: on when stderr is a terminal, or UNIVERSALMOTIF_PROGRESS if set)")


def _seed_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, help="random seed, for reproducible results")


def _scan_args(p: argparse.ArgumentParser, pvalue: float = 1e-4) -> None:
    p.add_argument("-P", "--pvalue", type=float, default=pvalue,
                   help=f"P-value cutoff for motif hits (default: {pvalue:g})")
    p.add_argument("--no-rc", dest="rc", action="store_false", default=None,
                   help="scan only the forward strand, as is done by default for RNA")


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
        from ._compare import _clusters_table, _merge_similar_run

        # One run gives both the merged motifs and the clusters.
        ms, clusters, merged = _merge_similar_run(motifs, a.qvalue, a.null, a.min_overlap, a.rc, None,
                                                  a.weighted, a.linkage, a.threads, None, "match", None)
        if a.clusters:
            write_table(_clusters_table(ms, clusters), a.clusters)
    write_motifs(merged, a.output, a.to)


def cmd_tree(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    tree = um.motif_tree(motifs, min_overlap=a.min_overlap, rc=a.rc, linkage=a.linkage,
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
    """A motif name as a file stem. It is kept well within file name
    limits, since merged motifs are named after all their members."""
    stem = re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_") or "motif"
    return stem[:100].rstrip("._+-") or "motif"


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


def _read_sites(path: str) -> dict[str, list[str]]:
    """Aligned sites by motif name.

    The file is FASTA, a plain list with one site per line, or a table with
    a match column such as umotif scan writes. A table's sites are grouped
    by its motif column if it has one."""
    if path != "-" and not Path(path).exists():
        raise CliError(f"no such file: {path}")
    text = _read_bytes(path).decode()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise CliError(f"no sites in {path}")
    if lines[0].startswith(">"):
        return {"": [ln for ln in lines if not ln.startswith(">")]}
    header = lines[0].split("\t")
    if "match" in header:
        df = pl.read_csv(io.StringIO("\n".join(lines)), separator="\t", null_values="NA",
                         infer_schema_length=10000)
        if df["match"].drop_nulls().is_empty():
            raise CliError(f"no sites in {path}")
        if "motif" not in df.columns:
            return {"": df["match"].drop_nulls().to_list()}
        return {name: g["match"].drop_nulls().to_list()
                for (name,), g in df.group_by("motif", maintain_order=True)}
    return {"": lines}


def cmd_create(a):
    import universalmotif as um

    if bool(a.consensus) == bool(a.sites):
        raise CliError("give consensus sequences, or a file of sites with --sites")
    kw = {"alphabet": a.alphabet, "type": a.type, "pseudocount": a.pseudocount,
          "family": a.family, "organism": a.organism}
    names = a.name or []
    motifs = []
    if a.consensus:
        if len(names) not in (0, len(a.consensus)):
            raise CliError("give one --name per consensus sequence, or none")
        for i, cons in enumerate(a.consensus):
            motifs.append(um.create_motif(cons, name=names[i] if names else cons,
                                          nsites=a.nsites, **kw))
    else:
        groups = _read_sites(a.sites)
        if len(names) not in (0, len(groups)):
            raise CliError(f"give one --name per motif ({len(groups)}), or none")
        for i, (group, sites) in enumerate(groups.items()):
            widths = {len(s) for s in sites}
            if len(widths) > 1:
                raise CliError(f"the sites{' of ' + group if group else ''} differ in length "
                               f"({min(widths)} to {max(widths)}); they must be aligned")
            stem = Path(a.sites).name.split(".")[0] if a.sites != "-" else "motif"
            name = names[i] if names else (group or stem or "motif")
            motifs.append(um.create_motif(sites, name=name, **kw))
    write_motifs(motifs, a.output, a.to)


def cmd_heatmap(a):
    import matplotlib

    matplotlib.use("Agg")
    import universalmotif as um

    if a.output in (None, "-"):
        raise CliError("give a file to draw the heatmap to with -o (PNG, PDF or SVG)")
    motifs = read_motifs(a.motifs, a.format)
    if len(motifs) < 2:
        raise CliError("at least two motifs are needed for a heatmap")
    fig = um.plot_comparison(
        motifs, labels=a.labels, value=a.value, min_overlap=a.min_overlap, rc=a.rc,
        cluster=a.cluster, linkage=a.linkage, dendrogram=a.dendrogram, cut=a.cut,
        cut_height=a.cut_height, cmap=a.cmap, annotate=a.annotate, logos=a.logos,
        nthreads=a.threads)
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
              "no_overlaps_by": a.overlaps_by, "no_overlaps_across_strands": a.across_strands,
              "no_overlaps_across_motifs": a.across_motifs, "use_gaps": a.gaps, "correct_gaps": a.correct_gaps,
              "first_hit": a.first_hit}
    if a.first_hit and a.no_overlaps:
        raise CliError("--first-hit keeps one hit per motif and sequence, so it cannot be used with "
                       "--no-overlaps")
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
    if a.bed:
        write_hits_bed(hits, a.output)
    else:
        write_table(hits, a.output)


def cmd_variants(a):
    import universalmotif as um

    motifs = read_motifs(a.motifs, a.format)
    if a.variants != "-" and not Path(a.variants).exists():
        raise CliError(f"no such file: {a.variants}")
    raw = _read_bytes(a.variants)
    header = next((ln for ln in raw.lstrip().splitlines() if not ln.startswith(b"##")), b"")
    if raw.lstrip().startswith(b"##fileformat=VCF") or header.upper().startswith(b"#CHROM\tPOS\tID\tREF\tALT"):
        from ._variants import _vcf_frame

        try:
            variants = _vcf_frame(raw, a.variants)
        except ValueError as e:
            raise CliError(str(e)) from None
    else:
        # A header line may start with #, as in "#chrom position ref alt", while ## lines are comments.
        lines = [ln for ln in raw.splitlines(keepends=True) if not ln.startswith(b"##")]
        if lines and lines[0].startswith(b"#"):
            lines[0] = lines[0][1:]
        variants = pl.read_csv(io.BytesIO(b"".join(lines)), separator="\t", null_values="NA",
                               infer_schema_length=10000, comment_prefix="#")
    hits = read_table(a.hits) if a.hits else None
    table = um.variant_motif_effects(variants, open_genome(a.genome), motifs, pvalue=a.pvalue, hits=hits,
                                     rc=a.rc, nthreads=a.threads, progress=a.progress)
    write_table(table, a.output)


def write_hits_bed(hits: pl.DataFrame, path: str | None) -> None:
    """Hits as BED6, named by motif. They are placed on their chromosome
    where they have one, and otherwise on the sequence scanned."""
    import universalmotif as um

    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "hits.bed"
        if hits.height:
            um.write_bed(hits, file, name="motif", score="")
            text = file.read_text()
        else:
            text = ""
    out = _open_output(path)
    try:
        out.write(text)
    finally:
        if out is not sys.stdout:
            out.close()


def cmd_compare(a):
    import numpy as np

    import universalmotif as um

    queries = read_motifs(a.motifs, a.format)
    kw = {"min_overlap": a.min_overlap, "rc": a.rc, "null": a.null, "nthreads": a.threads,
              "complete": a.complete, "method": a.method, "overhang": a.overhang}
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
        res = um.compare_motifs(queries + targets, compare_to=list(range(nq)), qvalue=1.0, **kw)
        res = res.filter(pl.col("target_i") >= nq).with_columns(pl.col("target_i") - nq)
        # The q-values count only each query's tests against the database,
        # not those against itself and the other queries.
        res = _bh_by_query(res).filter(pl.col("qvalue") <= qvalue)
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


def _bh_by_query(res: pl.DataFrame) -> pl.DataFrame:
    """Benjamini-Hochberg q-values recomputed within each query."""
    import numpy as np

    q = np.full(res.height, np.nan)
    p = res["pvalue"].to_numpy()
    for idx in res.with_row_index().group_by("subject_i").agg("index")["index"]:
        rows = np.asarray(idx)
        ps = p[rows]
        order = np.argsort(ps)
        m = len(ps)
        adj = ps[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        q[rows[order]] = np.minimum(adj, 1.0)
    return res.with_columns(pl.Series("qvalue", q))


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
                          qvalue=a.qvalue, rc=a.rc, shuffle_k=a.shuffle_k, bkg_shuffles=a.bkg_shuffles,
                          stability=a.stability, stability_fraction=a.stability_fraction, seed=a.seed,
                          nthreads=a.threads)
    if a.table:
        write_table(res, a.table)
    if res.height == 0:
        print("no motifs were found", file=sys.stderr)
        return
    write_motifs(res["motif"].to_list(), a.output, a.to)


def cmd_find_gapped(a):
    import universalmotif as um

    targets = _target_sequences(a, "target sequences")
    bkg = read_sequences(a.bkg) if a.bkg else None
    res = um.gapped_motif_finder(targets, bkg, k=a.k, min_gap=a.min_gap, max_gap=a.max_gap,
                                 nmotifs=a.nmotifs, hit_pvalue=a.hit_pvalue, stop_pvalue=a.stop_pvalue,
                                 n_seed_pairs=a.seed_pairs, min_fraction=a.min_fraction, holdout=a.holdout,
                                 mismatch={"off": False, "on": True}.get(a.mismatch, "auto"), mask=a.mask, shuffle_k=a.shuffle_k,
                                 bkg_shuffles=a.bkg_shuffles, rc=a.rc, seed=a.seed, nthreads=a.threads)
    if a.table:
        write_table(res, a.table)
    if res.height == 0:
        print("no gapped motifs were found", file=sys.stderr)
        return
    # Formats without gap notation write each motif as its two parts
    # joined, and the table gives each gap.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".* format cannot keep the gaps")
        write_motifs(res["motif"].to_list(), a.output, a.to)


def cmd_find_pairs(a):
    import universalmotif as um

    targets = _target_sequences(a, "target sequences")
    bkg = read_sequences(a.bkg) if a.bkg else None
    motifs = read_motifs(a.motifs, a.format) if a.motifs else None
    res = um.motif_pair_finder(targets, bkg, motifs=motifs, nmotifs=a.nmotifs, max_gap=a.max_gap,
                               hit_pvalue=a.hit_pvalue, train_pvalue=a.train_pvalue,
                               stop_pvalue=a.stop_pvalue, min_pairs=a.min_pairs, min_fold=a.min_fold,
                               max_window=a.max_window, holdout=a.holdout, merge_overlap=a.merge_overlap,
                               mask=a.mask, shuffle_k=a.shuffle_k, bkg_shuffles=a.bkg_shuffles, seed=a.seed,
                               nthreads=a.threads)
    if a.table:
        write_table(res, a.table)
    if res.height == 0:
        print("no pairs of motifs were found", file=sys.stderr)
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".* format cannot keep the gaps")
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
    from ._sequences import _seed

    # One seed for both calls below, so the table describes the very
    # sequences written.
    kw = {"n_per_target": a.n, "n_bins_gc": a.gc_bins, "n_bins_length": a.length_bins,
          "unique": not a.repeats, "seed": _seed(a.seed), "nthreads": a.threads}
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
              "perm_per_chrom": not a.global_perm, "across_strands": a.across_strands,
              "qvalue": a.qvalue, "seed": a.seed, "nthreads": a.threads}
    universe = None
    if a.universe:
        u = a.universe
        # A .txt file can hold chromosome sizes or ranges. Its number of
        # columns tells them apart.
        sizes_like = (_stem_suffix(u) in (".fai", ".sizes") or u.endswith("chrom.sizes")
                      or (_stem_suffix(u) == ".txt" and _two_columns(u)))
        if _stem_suffix(u) in _BED_SUFFIXES:
            universe = read_ranges(u)
        else:
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
    """Whether a file is a FASTA or 2bit genome, looking inside gzip."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(4)
            if raw[:2] == b"\x1f\x8b":
                fh.seek(0)
                with gzip.open(fh) as gz:
                    return gz.read(1) == b">"
    except OSError:
        return False
    return raw[:1] == b">" or raw in (b"\x43\x27\x41\x1a", b"\x1a\x41\x27\x43")


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


# Contribution scores ---------------------------------------------------------

_MODISCO_SUFFIXES = (".h5", ".hdf5", ".hd5")
_BIGWIG_SUFFIXES = (".bw", ".bigwig")


class _Scores:
    """Contribution scores read for a command: the observed score of each
    letter present, the hypothetical scores of every letter when the file
    has them, and the sequences by name."""

    def __init__(self, observed, hypothetical, sequences):
        self.observed = observed
        self.hypothetical = hypothetical
        self.sequences = sequences


def _read_scores(a, path: str, sequences: str | None = None) -> _Scores:
    """Scores from a chromBPNet or ENCODE HDF5 file, numpy files, or a
    bigWig track read over ranges of a genome."""
    import numpy as np

    import universalmotif as um

    if not Path(path).exists():
        raise CliError(f"no such file: {path}")
    if _stem_suffix(path) in _BIGWIG_SUFFIXES:
        if not (a.genome and a.ranges):
            raise CliError("scores from a bigWig track need the genome and ranges to read, with -g and -r")
        ranges = read_ranges(a.ranges)
        seqs = um.extract_ranges(open_genome(a.genome), ranges, nthreads=a.threads)
        rows = list(ranges.select("chrom", "start", "end").iter_rows())
        values = [um.signal_values(path, c, s, e) for c, s, e in rows]
        if len({len(v) for v in values}) != 1:
            raise CliError("the ranges differ in width; resize them to one width first")
        # Ranges on the - strand are read as their reverse complements, so their scores run backwards too.
        if "strand" in ranges.columns:
            for i, st in enumerate(ranges["strand"].cast(pl.String).to_list()):
                if st == "-":
                    values[i] = values[i][::-1]
        observed = np.asarray(values, dtype=np.float32)
        return _Scores(_centre(a, observed), None, _centre_seqs(a, seqs))
    seq_arg, fasta_names = None, None
    if sequences:
        if not Path(sequences).exists():
            raise CliError(f"no such file: {sequences}")
        if _stem_suffix(sequences) in (".fa", ".fasta", ".fna"):
            fasta = read_sequences(sequences)
            seq_arg, fasta_names = list(fasta.values()), list(fasta)
        else:
            seq_arg = sequences
    elif getattr(a, "genome", None) and _stem_suffix(path) in (".h5", ".hdf5"):
        # ENCODE's files of all regions hold no sequences; they are read
        # from the genome at the file's coordinates.
        seq_arg = open_genome(a.genome)
    # Large files are read a chunk of sequences at a time, each cut to
    # --centre before the next is read, so that the whole file is never held.
    n = _n_regions(path)
    chunks = [None] if n is None else [range(lo, min(lo + _READ_CHUNK, n)) for lo in range(0, n, _READ_CHUNK)]
    hyps, names, letters = [], [], []
    for idx in chunks:
        try:
            c = um.read_contributions(path, sequences=seq_arg, regions=getattr(a, "regions", None), indices=idx)
        except IndexError as e:
            raise CliError(f"the scores in {path} and the sequences do not match: {e}") from None
        hyp = np.asarray(c.scores, dtype=np.float32)
        if hyp.ndim != 3:
            raise CliError(f"the scores in {path} must have one per letter at each position, with shape "
                           f"(sequences, 4, length) or (sequences, length, 4), not {hyp.shape}")
        if hyp.shape[1] != 4:
            hyp = hyp.transpose(0, 2, 1)
        hyps.append(_centre(a, hyp))
        names += list(c.names)
        letters += list(c.sequences)
    if fasta_names is not None:
        if len(fasta_names) != len(letters):
            raise CliError(f"{len(fasta_names)} sequences in {sequences} for the scores of {len(letters)}")
        if not _stem_suffix(path) in (".h5", ".hdf5"):
            names = fasta_names
    # Regions repeated in a file, such as duplicate peaks, keep apart.
    seqs = _centre_seqs(a, dict(zip(_unique_names(names), letters)))
    hyp = np.concatenate(hyps) if len(hyps) > 1 else hyps[0]
    observed = (hyp * um.one_hot(list(seqs.values()))).sum(axis=1)
    return _Scores(observed, hyp, seqs)


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n}_{k + 1}")
    return out


_READ_CHUNK = 20000


def _n_regions(path: str) -> int | None:
    """The number of sequences in an HDF5 or .npy scores file, read a chunk
    at a time, or None for a file read whole."""
    import numpy as np

    if _stem_suffix(path) in (".h5", ".hdf5"):
        import h5py

        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        with h5py.File(path, "r") as f:
            for key in ("hyp_scores", "shap/seq", "attributions/shap"):
                if key in f:
                    return int(f[key].shape[0])
        return None
    if _stem_suffix(path) == ".npy":
        return int(np.load(path, mmap_mode="r").shape[0])
    return None


def _centre(a, x):
    width = getattr(a, "centre", None)
    if not width or width >= x.shape[-1]:
        return x
    lo = (x.shape[-1] - width) // 2
    return x[..., lo:lo + width]


def _centre_seqs(a, seqs: dict[str, str]) -> dict[str, str]:
    """The middles of the sequences, as _centre() cuts their scores. A
    sequence named by its range, as chrom:start-end, is renamed to the range
    of its middle, so that positions counted along it stay right."""
    width = getattr(a, "centre", None)
    if not width:
        return seqs
    out = {}
    for name, s in seqs.items():
        if width >= len(s):
            out[name] = s
            continue
        lo = (len(s) - width) // 2
        m = re.match(r"^(.+):(\d+)-(\d+)$", name)
        if m and int(m[3]) - int(m[2]) == len(s):
            name = f"{m[1]}:{int(m[2]) + lo}-{int(m[2]) + lo + width}"
        out[name] = s[lo:lo + width]
    return out


def _named_paths(values: list[str], what: str) -> dict[str, str]:
    """NAME=PATH arguments, or paths named by their file names."""
    out = {}
    for i, v in enumerate(values):
        name, sep, path = v.partition("=")
        if not sep:
            name, path = (Path(v).name.split(".")[0] or f"{what}{i + 1}"), v
        if name in out:
            raise CliError(f"the {what} name {name!r} is used twice; name them with NAME=PATH")
        if not Path(path).exists():
            raise CliError(f"no such file: {path}")
        out[name] = path
    return out


def read_cwms(paths: list[str], trim_threshold: float | None = 0.3) -> list:
    """CWMs from TF-MoDISco results files or plain matrix files."""
    import universalmotif as um

    out = []
    for path in paths:
        if not Path(path).exists():
            raise CliError(f"no such file: {path}")
        if _stem_suffix(path) in _MODISCO_SUFFIXES:
            out += um.read_modisco(path, type="CWM", trim_threshold=trim_threshold)
        else:
            out += um.read_matrix(path, type="CWM")
    if not out:
        raise CliError("no CWMs were read")
    return out


def write_cwms(cwms: list, path: str) -> None:
    """CWMs as plain matrices, the one format that keeps their values, with
    a first line saying they are CWMs so that every command reads them so."""
    write_motifs(cwms, path, "matrix")
    if path not in (None, "-"):
        body = Path(path).read_text()
        Path(path).write_text("# type: CWM\n" + body)


def _database(a) -> list | None:
    return read_motifs(a.db, a.db_format) if getattr(a, "db", None) else None


def cmd_seqlets(a):
    import universalmotif as um

    sc = _read_scores(a, a.scores, a.sequences)
    null = _read_scores(a, a.null_scores, a.null_sequences).observed if a.null_scores else None
    res = um.call_seqlets(sc.observed, sc.sequences, null_scores=null, threshold=a.threshold,
                          min_len=a.min_len, max_len=a.max_len, flank=a.flank, fdr=a.fdr, sign=a.sign,
                          passes=a.passes, seed=a.seed, nthreads=a.threads)
    write_table(res, a.output)


def cmd_contrib_motifs(a):
    import universalmotif as um

    sc = _read_scores(a, a.scores, a.sequences)
    null = _read_scores(a, a.null_scores, a.null_sequences).observed if a.null_scores else None
    seqlets = read_table(a.seqlets) if a.seqlets else None
    hyp = sc.hypothetical if a.idealize else None
    found = um.contribution_motifs(
        sc.observed, sc.sequences, hypothetical=hyp, seqlets=seqlets, null_scores=null, sign=a.sign,
        seqlet_passes=a.seqlet_passes, max_seqlets=a.max_seqlets or None, later_min_gain=a.later_min_gain,
        contribution_test=a.contribution_test or None, min_width=a.min_width, max_width=a.max_width, nmotifs=a.nmotifs, refine=a.refine, min_gain=a.min_gain,
        max_similarity=a.max_similarity, stability=a.stability, rounds=a.rounds, gapped=a.gapped,
        max_gap=a.max_gap, hits=a.hits_out is not None, hit_quantile=a.hit_quantile,
        pattern_similarity=a.pattern_similarity, seed=a.seed, nthreads=a.threads)
    if not found.motifs.height:
        signs = found.seqlets["sign"].value_counts() if "sign" in found.seqlets.columns else None
        hint = ""
        if a.sign == "positive" and signs is not None and (signs["sign"] == "-").any():
            hint = "; the scores also hold negative seqlets, searched with --sign negative or both"
        print(f"no motifs were found in {found.seqlets.height} seqlets{hint}", file=sys.stderr)
        # Empty outputs, so that pipelines expecting them still run.
        for path in (a.table, a.patterns, a.seqlets_out, a.hits_out, a.thresholds, a.cwms, a.html,
                     a.output if a.output not in (None, "-") else None):
            if path:
                Path(path).write_text("")
        return
    # Composites lose their gaps in motif files, so the table records them.
    gaps = [f"{','.join(map(str, m.gaploc))}:{','.join(map(str, m.mingap))}-{','.join(map(str, m.maxgap))}"
            if m.isgapped else None for m in found.ppms]
    if any(gaps):
        names = [m.name for m, g in zip(found.ppms, gaps) if g]
        print(f"composite motifs {', '.join(names)} are written without their gaps; the gaps column of "
              "--table gives them as positions:smallest-largest", file=sys.stderr)
    if a.table:
        write_table(found.motifs.with_columns(pl.Series("gaps", gaps, dtype=pl.String)), a.table)
    if a.patterns:
        write_table(found.patterns, a.patterns)
    if a.seqlets_out:
        write_table(found.seqlets, a.seqlets_out)
    if a.hits_out:
        write_table(found.hits, a.hits_out)
    if a.thresholds:
        write_table(found.thresholds, a.thresholds)
    if a.cwms:
        write_cwms(found.cwms, a.cwms)
    if a.html:
        found.report(a.html, database=_database(a), nthreads=a.threads)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".* format cannot preserve gap definitions")
        write_motifs(found.ppms, a.output, a.to)


def cmd_scan_contrib(a):
    import universalmotif as um

    cwms = read_cwms(a.cwms, a.trim_threshold)
    sc = _read_scores(a, a.scores, a.sequences)
    kw = {}
    if a.thresholds:
        if _stem_suffix(a.thresholds) in _MODISCO_SUFFIXES:
            kw["thresholds"] = um.modisco_thresholds(a.thresholds, trim_threshold=a.trim_threshold)
        else:
            kw["thresholds"] = read_table(a.thresholds)
    else:
        kw.update(min_similarity=a.min_similarity, min_importance=a.min_importance)
    hits = um.scan_contributions(cwms, sc.observed, sequences=sc.sequences, rc=a.rc, normalize=a.normalize,
                                 no_overlaps=a.no_overlaps, no_overlaps_across_motifs=a.across_motifs,
                                 nthreads=a.threads, **kw)
    write_table(hits, a.output)


def cmd_modisco(a):
    import universalmotif as um

    if not Path(a.modisco).exists():
        raise CliError(f"no such file: {a.modisco}")
    if _stem_suffix(a.modisco) not in _MODISCO_SUFFIXES:
        raise CliError(f"{a.modisco} is not a TF-MoDISco results file (.h5); 'modisco' draws those, and "
                       "'contrib-motifs --html' draws the motifs it finds")
    if not (a.output or a.html):
        raise CliError("give a file for the figure with -o, or for the HTML page with --html")
    patterns = None
    if a.patterns:
        if len(a.patterns) > 2:
            raise CliError("--patterns takes one number, or two for positive and negative patterns")
        patterns = a.patterns[0] if len(a.patterns) == 1 else (a.patterns[0], a.patterns[1])
    fig = um.plot_modisco(a.modisco, database=_database(a), patterns=patterns, n_matches=a.n_matches,
                          qvalue=a.qvalue, trim_threshold=a.trim_threshold, html=a.html, title=a.title,
                          nthreads=a.threads)
    if a.output:
        fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight")


def cmd_combine(a):
    import universalmotif as um

    sources = _named_paths(a.sources, "source")
    for path in sources.values():
        if _stem_suffix(path) not in _MODISCO_SUFFIXES:
            raise CliError(f"{path} is not a TF-MoDISco results file (.h5); 'combine' takes those, and "
                           "in Python, combine_patterns() also takes contribution_motifs() results")
    combined = um.combine_patterns(sources, similarity=a.similarity, linkage=a.linkage,
                                   trim_threshold=a.trim_threshold, database=_database(a), qvalue=a.qvalue,
                                   nthreads=a.threads)
    write_table(combined.clusters, a.output)
    if a.patterns:
        write_table(combined.patterns, a.patterns)
    if a.cwms:
        write_cwms(combined.cwms, a.cwms)
    if a.ppms:
        ppms = [m for m in combined.ppms if m is not None]
        if len(ppms) < len(combined.ppms):
            print("warning: some clusters have no PPM, as their patterns came without one", file=sys.stderr)
        write_motifs(ppms, a.ppms, None)
    if a.html:
        combined.report(a.html, database=_database(a), qvalue=a.qvalue, nthreads=a.threads)


def cmd_name(a):
    import universalmotif as um

    if not a.db:
        raise CliError("give the database motifs to name the motifs by with --db")
    for path in a.motifs if isinstance(a.motifs, list) else [a.motifs]:
        if _stem_suffix(path) in _MODISCO_SUFFIXES:
            raise CliError(f"{path} is a TF-MoDISco results file; name its patterns with "
                           "'umotif modisco FILE --db ... --html report.html'")
    motifs = read_motifs(a.motifs, a.format)
    db = read_motifs(a.db, a.db_format)
    res = um.name_motifs(motifs, db, top=a.top, qvalue=a.qvalue, nthreads=a.threads)
    write_table(res, a.output)


# The parser -----------------------------------------------------------------


def _add_target_args(p: argparse.ArgumentParser, what: str, genome_help: str | None = None) -> None:
    p.add_argument("-s", "--sequences", help=f"FASTA file of {what}")
    p.add_argument("-g", "--genome", help=genome_help or "genome (FASTA, bgzipped FASTA or 2bit) "
                                                         f"to take the {what} from, with -r")
    p.add_argument("-r", "--ranges", help=f"BED file or table of the ranges of the {what}, "
                                          "with -g")


# The commands by what they work on, in the order --help and the
# documentation list them.
COMMAND_GROUPS = (
    ("Motifs", ("convert", "fetch", "create", "summary", "filter", "trim", "merge", "tree", "compare", "name")),
    ("Drawing", ("logo", "heatmap")),
    ("Sequences", ("extract", "shuffle", "bkg")),
    ("Scanning and testing", ("scan", "enrich", "posenr", "coocc", "proximity", "matrix", "variants")),
    ("Discovery", ("find", "find-gapped", "find-pairs")),
    ("Contribution scores", ("seqlets", "contrib-motifs", "scan-contrib", "modisco", "combine")),
)


class _formatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32)
        # Leave room for the longest subcommand name. It is not measured
        # when the subcommands are added.
        self._action_max_length = 19


def build_parser(prog: str = "universalmotif") -> argparse.ArgumentParser:
    from universalmotif import __version__

    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=_formatter,
        description="Work with sequence motifs from the command line: convert, fetch and curate "
                    "motifs, scan and test sequences, discover motifs, find the motifs of sequence "
                    "models, and draw logos.",
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

    p = command("create", cmd_create, "make motifs from consensus sequences or sites",
                "Make motifs from consensus sequences, one per sequence, which may use IUPAC "
                "codes. Or make them from a file of aligned sites with --sites. The file can "
                "be FASTA, a list of one site per line, or a table with a match column such "
                "as umotif scan writes. A table gives one motif per motif named in it.")
    p.add_argument("consensus", nargs="*", metavar="CONSENSUS",
                   help="consensus sequences, such as TGASTCA")
    p.add_argument("--sites", help="file of aligned sites; - reads standard input")
    _motif_out_args(p)
    p.add_argument("--name", action="append",
                   help="motif name, one per motif in order (repeatable; default: the consensus, "
                        "the motif names of a table, or the name of the sites file)")
    p.add_argument("--alphabet", help="DNA, RNA, AA or a string of letters (default: guessed)")
    p.add_argument("--type", choices=("PCM", "PPM", "PWM", "ICM"),
                   help="type of the motifs (default: PPM)")
    p.add_argument("--nsites", type=int,
                   help="number of sites a consensus motif stands for (default: unset, and counted "
                        "as 100 where counts are needed)")
    p.add_argument("--pseudocount", type=float, default=0.0,
                   help="pseudocount for the conversion to --type (default: 0)")
    p.add_argument("--family", help="family of the motifs")
    p.add_argument("--organism", help="organism of the motifs")

    p = command("heatmap", cmd_heatmap, "draw motif similarities as a clustered heatmap",
                "Compare motifs all against all and draw the scores, P-values or q-values as a "
                "clustered heatmap with a dendrogram. The tree can also be shown cut at a "
                "level of similarity.")
    _motif_args(p)
    p.add_argument("-o", "--output", help="file for the heatmap (PNG, PDF or SVG)")
    p.add_argument("--value", choices=("score", "pvalue", "qvalue"), default="score",
                   help="what the cells show; P- and q-values as -log10 (default: score)")
    p.add_argument("--labels", default="name",
                   help="motif field to label rows and columns with, such as altname "
                        "(default: name)")
    p.add_argument("--min-overlap", type=int, default=6,
                   help="minimum overlap of aligned motifs (default: 6)")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not align motifs as reverse complements")
    p.add_argument("--no-cluster", dest="cluster", action="store_false",
                   help="keep the motifs in the order given")
    p.add_argument("--linkage", default="complete",
                   choices=("complete", "average", "single", "ward.D", "ward.D2", "mcquitty",
                            "median", "centroid"),
                   help="agglomeration method (default: complete)")
    p.add_argument("--method", dest="linkage", help=argparse.SUPPRESS)
    p.add_argument("--dendrogram", choices=("left", "top", "both", "none"), default="left",
                   help="where to draw the tree (default: left)")
    p.add_argument("--cut", type=float,
                   help="show where the tree is cut at this score, such as 0.8, boxing the "
                        "groups of motifs")
    p.add_argument("--cut-height", type=float, help="show where the tree is cut at this height")
    p.add_argument("--cmap", default="Blues", help="matplotlib colour map (default: Blues)")
    p.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=None,
                   help="write each value in its cell (default: for up to 20 motifs)")
    p.add_argument("--logos", action=argparse.BooleanOptionalAction, default=None,
                   help="draw logos beside the rows (default: for up to 60 motifs)")
    p.add_argument("--dpi", type=int, default=150, help="resolution of PNG files (default: 150)")
    _threads_arg(p)

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
    p.add_argument("--linkage", choices=("single", "complete"), default="single",
                   help="join a motif to a cluster when it is similar to any member (single), "
                        "or only to every member (complete) (default: single)")
    p.add_argument("--all", action="store_true", help="merge all the motifs into one")
    p.add_argument("--name", help="name of the merged motif, with --all")
    p.add_argument("--clusters", help="also write a table of which motifs were merged")
    _threads_arg(p)

    p = command("tree", cmd_tree, "cluster motifs into a tree",
                "Cluster motifs by similarity and write the tree in Newick format. With -k "
                "or --height, write a table of clusters instead.")
    _motif_args(p)
    _out_arg(p, "the tree or clusters")
    p.add_argument("--linkage", default="complete",
                   choices=("complete", "average", "single", "ward.D", "ward.D2", "mcquitty",
                            "median", "centroid"),
                   help="agglomeration method (default: complete)")
    p.add_argument("--method", dest="linkage", help=argparse.SUPPRESS)
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
                "Draw motifs as sequence logos. They are aligned in one figure written with "
                "-o, or written one file per motif to a folder with --each.")
    _motif_args(p)
    p.add_argument("-o", "--output", help="file for one figure of all the motifs (PNG, PDF, SVG)")
    p.add_argument("--each", metavar="FOLDER", help="write one logo per motif into this folder")
    p.add_argument("--ext", choices=("svg", "png", "pdf"), default="svg",
                   help="file type for --each (default: svg, which needs no extra packages)")
    p.add_argument("--type", choices=("ICM", "PPM", "PWM", "PCM"),
                   help="what the letter heights show (default: contributions for CWMs, and information "
                        "content for other motifs)")
    p.add_argument("--relative-entropy", action="store_true",
                   help="information content relative to the background")
    p.add_argument("--no-rc", dest="rc", action="store_false",
                   help="do not reverse complement motifs to align them")
    p.add_argument("--sort-by", choices=("none", "ic", "similarity"), default="none",
                   help="order of the motifs in the figure, either as given, by decreasing "
                        "information content, or clustered by similarity (default: none)")
    p.add_argument("--dpi", type=int, default=150, help="resolution of PNG files (default: 150)")
    _threads_arg(p)

    # Sequences
    p = command("scan", cmd_scan, "scan sequences for motif hits",
                "Scan sequences for motif hits. The sequences come from one of three places. "
                "With -s, a FASTA file is read a batch at a time. With -g, a whole genome is "
                "scanned. With -g and -r, ranges of a genome are scanned and their hits are "
                "reported in genomic coordinates.")
    _motif_args(p)
    _add_target_args(p, "sequences", genome_help="genome (FASTA, bgzipped FASTA or 2bit) to scan "
                                                 "whole, or in parts with -r")
    _out_arg(p, "the hits")
    _scan_args(p)
    p.add_argument("--no-overlaps", action="store_true",
                   help="remove overlapping hits of each motif, keeping the best")
    p.add_argument("--overlaps-by", choices=("pvalue", "score"), default="pvalue",
                   help="what makes the best hit with --no-overlaps (default: pvalue)")
    p.add_argument("--across-strands", action=argparse.BooleanOptionalAction, default=True,
                   help="with --no-overlaps, compare hits on opposite strands so that a "
                        "palindromic site is kept once (default: on)")
    p.add_argument("--across-motifs", action="store_true",
                   help="with --no-overlaps, compare hits of different motifs")
    p.add_argument("--no-gaps", dest="gaps", action="store_false",
                   help="ignore the gaps of gapped motifs")
    p.add_argument("--correct-gaps", action=argparse.BooleanOptionalAction, default=True,
                   help="correct gapped motifs' P-values for the gap sizes tried (default: on)")
    p.add_argument("--first-hit", action="store_true",
                   help="keep only the leftmost hit of each motif in each sequence; this is "
                        "quicker when most sequences have one")
    p.add_argument("--local", action="store_true",
                   help="with -r, report positions within each range, not on the genome")
    p.add_argument("--bed", action="store_true",
                   help="write the hits as BED, named by motif, rather than as a table")
    p.add_argument("--genomic", action="store_true",
                   help="with -s, read each sequence's place in the genome from names like "
                        "chrom:start-end and report hits in genomic coordinates")
    _progress_arg(p)
    _threads_arg(p)

    p = command("variants", cmd_variants, "score how variants change motif sites",
                "Score how variants change the motif sites they fall in, for substitutions, "
                "insertions and deletions alike. Variants come from a VCF file or a "
                "tab-separated table. A table needs chrom, position, ref and alt columns, "
                "with positions counted from 0. Variants are applied to the genome given "
                "with -g one at a time.")
    _motif_args(p)
    p.add_argument("-v", "--variants", required=True,
                   help="VCF file (plain, gzipped or bgzipped) or table of variants; - reads "
                        "standard input")
    p.add_argument("-g", "--genome", required=True,
                   help="genome the variants are in (FASTA, bgzipped FASTA or 2bit)")
    p.add_argument("--hits", help="table of hits from scan -g; only those sites are scored, "
                                  "whatever their scores")
    _out_arg(p)
    _scan_args(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("compare", cmd_compare, "compare motifs with each other or with a database",
                "Compare motifs with each other, or with a database of motifs given with --db. "
                "The matches that pass a q-value cutoff are reported.")
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
    p.add_argument("--method", choices=("PCC", "EUCL"), default="PCC",
                   help="compare columns by Pearson correlation or Euclidean distance "
                        "(default: PCC)")
    p.add_argument("--overhang", choices=("ignore", "background"), default="ignore",
                   help="compare columns hanging over the other motif's ends with the "
                        "background (default: ignore)")
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
    p.add_argument("-q", "--qvalue", type=float,
                   help="q-value cutoff for reported motifs (default: report every motif)")
    # "hits" is an older name for "sites", still accepted.
    p.add_argument("--test", choices=("seqs", "sites"), default="seqs",
                   type=lambda s: "sites" if s == "hits" else s,
                   help="count sequences with hits, or hits among all possible sites "
                        "(default: seqs)")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("find", cmd_find, "discover motifs de novo",
                "Discover motifs enriched in target sequences. The motifs are written with -o "
                "and a table of results with --table.")
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
    p.add_argument("--no-rc", dest="rc", action="store_false", default=None,
                   help="search only the forward strand, as is done by default for RNA")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    p.add_argument("--bkg-shuffles", type=int, default=3,
                   help="shuffles of the targets making up the background, without --bkg (default: 3)")
    p.add_argument("--stability", type=int, default=0,
                   help="rerun the search this many times on random subsets of the targets, and report "
                        "in the table how often each motif is found again (default: 0, no reruns)")
    p.add_argument("--stability-fraction", type=float, default=0.5,
                   help="share of the targets in each rerun (default: 0.5)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("find-pairs", cmd_find_pairs, "discover pairs of motifs that favour a spacing (experimental)",
                "Discover pairs of motifs whose sites keep a fixed or nearly fixed spacing more "
                "often than in the background. The parts are the motifs given with -m, followed by "
                "motifs found in the targets as by the find command. Each spacing is chosen in part "
                "of the targets and tested in the rest. The pairs are written with -o, each as one "
                "gapped motif, and a table of results with --table. The command is experimental "
                "and may change in later releases.")
    _add_target_args(p, "target sequences")
    p.add_argument("-b", "--bkg", help="FASTA file of background sequences")
    p.add_argument("-m", "--motifs", action="append", metavar="FILE",
                   help="motif file whose motifs are parts, before those found; can be repeated")
    p.add_argument("-f", "--format", choices=_MOTIF_FORMATS, metavar="FORMAT",
                   help="the format of the -m files, if it cannot be recognized: " + _FORMAT_LIST)
    _motif_out_args(p)
    p.add_argument("--table", help="also write a table of the pairs found")
    p.add_argument("--nmotifs", type=int, default=8,
                   help="most motifs found in the targets to add to the parts (default: 8)")
    p.add_argument("--max-gap", type=int, default=31, help="longest spacer, at most 31 (default: 31)")
    p.add_argument("--max-window", type=int, default=3,
                   help="most spacers a spacing may span (default: 3)")
    p.add_argument("--hit-pvalue", type=float, default=1e-4,
                   help="P-value of a site counting as a hit; shorter parts are scanned at up to "
                        "1e-3 (default: 1e-4)")
    p.add_argument("--train-pvalue", type=float, default=1e-3,
                   help="P-value a spacing must reach where it is chosen (default: 1e-3)")
    p.add_argument("--stop-pvalue", type=float, default=1e-3,
                   help="corrected P-value a spacing must reach in the targets held out "
                        "(default: 1e-3)")
    p.add_argument("--min-pairs", type=int, default=5,
                   help="fewest pairs of sites a spacing needs where it is chosen (default: 5)")
    p.add_argument("--min-fold", type=float, default=2.0,
                   help="fewest times more pairs than expected in the targets held out (default: 2)")
    p.add_argument("--holdout", type=float, default=0.3,
                   help="share of the targets and background held out (default: 0.3)")
    p.add_argument("--merge-overlap", type=float, default=0.3,
                   help="share of shared sites at which two pairs are merged; above 1 keeps them all "
                        "(default: 0.3)")
    p.add_argument("--no-mask", dest="mask", action="store_false",
                   help="leave low-complexity regions unmasked")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    p.add_argument("--bkg-shuffles", type=int, default=3,
                   help="shuffles of the targets making up the background, without --bkg (default: 3)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("find-gapped", cmd_find_gapped, "discover gapped motifs de novo (experimental)",
                "Discover motifs made of two parts with a spacer of varying length. The motifs "
                "are written with -o and a table of results with --table. Each motif is tested "
                "in a share of the targets and background held out from the search. MEME, "
                "JASPAR, TRANSFAC and universalmotif files keep the gaps; other formats write "
                "each motif as its two parts joined, and the table gives each gap. The command "
                "is experimental and may change in later releases.")
    _add_target_args(p, "target sequences")
    p.add_argument("-b", "--bkg", help="FASTA file of background sequences")
    _motif_out_args(p)
    p.add_argument("--table", help="also write a table of the motifs found")
    p.add_argument("-k", type=int, nargs="+", default=[4, 5, 6],
                   help="k-mer sizes to seed motifs from, each from 2 to 6 (default: 4 5 6)")
    p.add_argument("--min-gap", type=int, default=0, help="shortest spacer (default: 0)")
    p.add_argument("--max-gap", type=int, default=31, help="longest spacer, at most 31 (default: 31)")
    p.add_argument("--nmotifs", type=int, default=5, help="most motifs to find (default: 5)")
    p.add_argument("--hit-pvalue", type=float, default=1e-4,
                   help="P-value of a site counting as a hit (default: 1e-4)")
    p.add_argument("--stop-pvalue", type=float, default=1e-3,
                   help="spacing P-value at which to stop (default: 1e-3)")
    p.add_argument("--seed-pairs", type=int, default=6,
                   help="most enriched pairs of k-mers built into motifs for each k-mer size "
                        "(default: 6)")
    p.add_argument("--min-fraction", type=float, default=0.02,
                   help="smallest share of targets a pair of k-mers must be in (default: 0.02)")
    p.add_argument("--holdout", type=float, default=0.3,
                   help="share of the targets and background held out to test the motifs; "
                        "0 tests them where they were found (default: 0.3)")
    p.add_argument("--mismatch", choices=["off", "on", "auto"], default="auto",
                   help="count pairs of 6-mers with up to one mismatch, for rarer motifs; auto "
                        "does so only when the exact search finds nothing, and neither does anything "
                        "without 6 among -k (default: auto)")
    p.add_argument("--no-mask", dest="mask", action="store_false",
                   help="leave low-complexity regions unmasked")
    p.add_argument("--no-rc", dest="rc", action="store_false", help="search only the forward strand")
    p.add_argument("--shuffle-k", type=int, default=2,
                   help="k-mer size kept when shuffling targets for a background (default: 2)")
    p.add_argument("--bkg-shuffles", type=int, default=3,
                   help="shuffles of the targets making up the background, without --bkg (default: 3)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("shuffle", cmd_shuffle, "shuffle sequences, keeping k-mer counts")
    p.add_argument("-s", "--sequences", required=True, help="FASTA file of sequences")
    _out_arg(p, "the shuffled sequences")
    p.add_argument("-k", type=int, default=2, help="size of the k-mers kept (default: 2)")
    p.add_argument("--method", choices=("euler", "linear", "markov"), default="euler",
                   help="shuffling method (default: euler)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("extract", cmd_extract, "extract the sequences of genomic ranges",
                "Write the sequences of genomic ranges as FASTA. Ranges on the minus strand "
                "are reverse complemented.")
    p.add_argument("-g", "--genome", required=True,
                   help="genome (FASTA, bgzipped FASTA or 2bit)")
    p.add_argument("-r", "--ranges", required=True, help="BED file or table of ranges")
    _out_arg(p, "the sequences")
    p.add_argument("--width", type=int, default=80, help="FASTA line width (default: 80)")
    _threads_arg(p)

    p = command("bkg", cmd_bkg, "draw background sequences matched to targets",
                "Draw background sequences matched to the targets on GC content and length, "
                "and optionally on bigWig signal. They are drawn from a universe of sequences "
                "or from random windows of a genome.")
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
                "scores. The output file's extension sets its format. A .h5ad file is "
                "written as AnnData, and a .mtx file as Matrix Market with the row and "
                "column names alongside. Anything else is written as a long table.")
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
                "Test whether motif hits sit closer to genomic anchors than chance. The hits "
                "come from scanning a genome for the motifs, or from a table of hits.")
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
    p.add_argument("--across-strands", action=argparse.BooleanOptionalAction, default=True,
                   help="count overlapping hits on the two strands once, as by default; "
                        "--no-across-strands counts each strand on its own")
    p.add_argument("--stranded", dest="across_strands", action="store_false", help=argparse.SUPPRESS)
    p.add_argument("-q", "--qvalue", type=float,
                   help="q-value cutoff for reported motifs (default: report every motif)")
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
    p.add_argument("-q", "--qvalue", type=float,
                   help="q-value cutoff for reported motifs (default: report every motif)")
    p.add_argument("--min-window", type=int, default=10, help="smallest window (default: 10)")
    p.add_argument("--max-window", type=int,
                   help="largest window (default: half the sequence length)")
    p.add_argument("--window-step", type=int, default=10, help="window size step (default: 10)")
    p.add_argument("--position-step", type=int, default=5,
                   help="window position step with --mode local (default: 5)")
    p.add_argument("--centres", action="store_true",
                   help="include the centres of the hits used for each motif")
    _threads_arg(p)

    # Contribution scores
    def score_args(p, what="scores"):
        p.add_argument("scores", help="contribution scores: an HDF5 file from chromBPNet or ENCODE, numpy "
                                      ".npy or .npz files, or a bigWig track of observed scores read over "
                                      "the ranges given with -g and -r")
        p.add_argument("-s", "--sequences",
                       help="the one-hot sequences of numpy scores (.npy or .npz), or a FASTA file of them")
        p.add_argument("-g", "--genome", help="genome to read the ranges of a bigWig track from, with -r, or "
                                              "the sequences of an HDF5 file of all regions, which holds none")
        p.add_argument("-r", "--ranges", help="BED file or table of ranges, all of one width, for a "
                                              "bigWig track")
        p.add_argument("--regions", help="narrowPeak file of the regions of a chromBPNet scores file, if "
                                         "it has no interpreted_regions.bed beside it")
        p.add_argument("--centre", type=int,
                       help="keep only this many positions in the middle of each sequence, such as 500 "
                            "of a model's 2,114; sequences named chrom:start-end are renamed to their "
                            "middles, so positions stay genomic once added to the start")

    def null_args(p):
        p.add_argument("--null-scores",
                       help="scores of sequences with nothing to find, such as the model's for shuffled "
                            "sequences, in the same form as the scores (default: the scores shuffled)")
        p.add_argument("--null-sequences", help="the one-hot sequences of numpy --null-scores")

    def db_args(p, why):
        # One file per --db, so that the option cannot swallow the input file after it.
        p.add_argument("--db", action="append", metavar="FILE",
                       help=f"a motif file, such as JASPAR's, {why}; repeat for several")
        p.add_argument("--db-format", choices=_MOTIF_FORMATS, metavar="FORMAT", help="format of the --db files")

    p = command("seqlets", cmd_seqlets, "call seqlets in a model's contribution scores",
                "Call seqlets, the short stretches of high contribution, in a sequence model's contribution "
                "scores, and write them as a table.")
    score_args(p)
    null_args(p)
    _out_arg(p, "the seqlets")
    p.add_argument("--sign", choices=("both", "positive", "negative"), default="both",
                   help="seqlets of positive or negative contribution, or both (default: both)")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="P-value every stretch inside a seqlet must reach (default: 0.01)")
    p.add_argument("--min-len", type=int, default=4, help="shortest seqlet, before flanks (default: 4)")
    p.add_argument("--max-len", type=int, default=25, help="longest seqlet, before flanks (default: 25)")
    p.add_argument("--flank", type=int, default=3, help="positions added on either side (default: 3)")
    p.add_argument("--fdr", type=float, default=0.2,
                   help="largest share of the null scores' seqlets among those kept (default: 0.2)")
    p.add_argument("--passes", type=int, default=1,
                   help="times to call seqlets, each after the first with those found so far set to 0 "
                        "(default: 1)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("contrib-motifs", cmd_contrib_motifs, "find the motifs a model relies on (experimental)",
                "Find the motifs a sequence model relies on from its contribution scores: call seqlets, "
                "search them for motifs, refine and choose the motifs, and match their CWMs against the "
                "scores. The motifs are written as PPMs with -o, and the other results with the options "
                "naming files for them. CWMs are written as plain matrices, the one format that keeps "
                "their values. The command is experimental and may change in later releases.")
    score_args(p)
    null_args(p)
    _motif_out_args(p)
    p.add_argument("--cwms", help="also write the motifs' CWMs, as plain matrices")
    p.add_argument("--table", help="also write a table of the motifs")
    p.add_argument("--patterns", help="also write a table of the patterns")
    p.add_argument("--seqlets-out", metavar="FILE", help="also write the seqlets")
    p.add_argument("--hits", dest="hits_out", metavar="FILE",
                   help="also scan all the scores for hits of the CWMs, and write them")
    p.add_argument("--thresholds", help="also write each motif's hit thresholds")
    p.add_argument("--html", help="also write the patterns as an HTML report")
    db_args(p, "to name the patterns by in the report")
    p.add_argument("--seqlets", help="table of seqlets to use instead of calling them, as from the "
                                     "seqlets command")
    p.add_argument("--sign", choices=("positive", "negative", "both"), default="positive",
                   help="seqlets to search (default: positive)")
    p.add_argument("--seqlet-passes", type=int, default=2,
                   help="times to call seqlets, the second with those of the first set to 0, which finds "
                        "weaker sites beside strong ones (default: 2)")
    p.add_argument("--max-seqlets", type=int, default=50_000,
                   help="most seqlets of each sign to search, as a random sample, or 0 for all of them "
                        "(default: 50000)")
    p.add_argument("--no-idealize", dest="idealize", action="store_false",
                   help="leave out the search on seqlets rewritten in the letters the model prefers, "
                        "which needs hypothetical scores")
    p.add_argument("--min-width", type=int, default=6, help="smallest motif width (default: 6)")
    p.add_argument("--max-width", type=int, default=15, help="largest motif width (default: 15)")
    p.add_argument("--nmotifs", type=int, default=60, help="most motifs per search (default: 60)")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="keep the motifs as found, without rebuilding them from their hits")
    p.add_argument("--min-gain", type=float, default=0.0025,
                   help="least share of the seqlets a motif must add to be kept (default: 0.0025)")
    p.add_argument("--later-min-gain", type=float,
                   help="least share for motifs of later searches (default: four fifths of --min-gain)")
    p.add_argument("--contribution-test", type=float, default=0.01,
                   help="adjusted P-value cutoff of the test against shuffled copies of each motif, or 0 to "
                        "skip it (default: 0.01)")
    p.add_argument("--max-similarity", type=float, default=0.9,
                   help="most a motif's CWM may resemble one kept before it, by cosine (default: 0.9)")
    p.add_argument("--stability", type=int, default=0,
                   help="rerun each search this many times on halves of the seqlets, keeping motifs "
                        "found in most (default: 0, no reruns)")
    p.add_argument("--rounds", type=int, default=2,
                   help="searches, each of the seqlets the motifs so far leave unreached (default: 2)")
    p.add_argument("--no-gapped", dest="gapped", action="store_false",
                   help="leave out the search for composite motifs of two parts")
    p.add_argument("--max-gap", type=int, default=12, help="longest spacer of a composite (default: 12)")
    p.add_argument("--hit-quantile", type=float, default=0.05,
                   help="quantile of each motif's hits in the seqlets that sets its hit thresholds "
                        "(default: 0.05)")
    p.add_argument("--pattern-similarity", type=float, default=0.8,
                   help="CWM cosine at which motifs are gathered into one pattern (default: 0.8)")
    _seed_arg(p)
    _progress_arg(p)
    _threads_arg(p)

    p = command("scan-contrib", cmd_scan_contrib, "find where a model used each motif",
                "Match CWMs against a model's contribution scores, and write the windows that pass "
                "the thresholds as hits. CWMs come from TF-MoDISco results files or plain matrix "
                "files, such as those of contrib-motifs --cwms, each given with -m.")
    p.add_argument("-m", "--cwms", action="append", required=True, metavar="FILE",
                   help="a TF-MoDISco results file (.h5) or plain matrix file of CWMs; give it once "
                        "for each file")
    score_args(p)
    _out_arg(p, "the hits")
    p.add_argument("--thresholds",
                   help="a TF-MoDISco results file to set each pattern's thresholds from its seqlets, or "
                        "a table with motif, min_similarity and min_importance columns, as from "
                        "contrib-motifs --thresholds")
    p.add_argument("--min-similarity", type=float, default=0.7,
                   help="least cosine between a window's scores and the CWM, without --thresholds "
                        "(default: 0.7)")
    p.add_argument("--min-importance", type=float, default=0.5,
                   help="least importance of a window, as a share of the CWM's, without --thresholds "
                        "(default: 0.5)")
    p.add_argument("--trim-threshold", type=float, default=0.3,
                   help="trim TF-MoDISco patterns to the columns holding this share of the largest "
                        "(default: 0.3)")
    p.add_argument("--normalize", action="store_true",
                   help="scale up the scores of weakly scored sequences to the median before matching")
    p.add_argument("--keep-overlaps", dest="no_overlaps", action="store_false",
                   help="keep overlapping hits of each motif, rather than only the best of them")
    p.add_argument("--across-motifs", action="store_true",
                   help="keep only the best of overlapping hits of different motifs too")
    p.add_argument("--no-rc", dest="rc", action="store_false", help="match only the forward strand")
    _threads_arg(p)

    p = command("modisco", cmd_modisco, "draw TF-MoDISco results as a report",
                "Draw the patterns of a TF-MoDISco results file as a report, one row per pattern, with "
                "their seqlets, CWM, letters and closest database motifs. It is written as a figure "
                "with -o, as an HTML page with --html, or both.")
    p.add_argument("modisco", help="TF-MoDISco results file (.h5)")
    p.add_argument("-o", "--output", help="file for the figure (PNG, PDF or SVG)")
    p.add_argument("--html", help="file for the report as an HTML page")
    db_args(p, "to name the patterns by")
    p.add_argument("--patterns", type=int, nargs="+", metavar="N",
                   help="draw only the first N patterns of each sign, or N_POS N_NEG")
    p.add_argument("--n-matches", type=int, default=3, help="database matches per pattern (default: 3)")
    p.add_argument("-q", "--qvalue", type=float, default=0.05,
                   help="largest q-value of a database match (default: 0.05)")
    p.add_argument("--trim-threshold", type=float, default=0.3,
                   help="trim patterns to the columns holding this share of the largest (default: 0.3)")
    p.add_argument("--title", help="title of the HTML page")
    p.add_argument("--dpi", type=int, default=150, help="resolution of PNG files (default: 150)")
    _threads_arg(p)

    p = command("combine", cmd_combine, "combine the patterns of many TF-MoDISco runs",
                "Combine the patterns of several TF-MoDISco results files into one set, clustering "
                "their CWMs and averaging each cluster. A table of the clusters is written with -o, "
                "and their motifs, patterns and report with the options naming files for them.")
    p.add_argument("sources", nargs="+", metavar="SOURCE",
                   help="TF-MoDISco results files, each as NAME=PATH or a path named by its file name")
    _out_arg(p, "the table of clusters")
    p.add_argument("--patterns", help="also write a table of the patterns and their clusters")
    p.add_argument("--cwms", help="also write the clusters' CWMs, as plain matrices")
    p.add_argument("--ppms", help="also write the clusters' PPMs, in the format of the file's extension")
    p.add_argument("--html", help="also write the clusters as an HTML report")
    db_args(p, "to name the clusters by")
    p.add_argument("--similarity", type=float, default=0.9,
                   help="least CWM cosine for two patterns to share a cluster (default: 0.9)")
    p.add_argument("--linkage", choices=("complete", "single"), default="complete",
                   help="cluster linkage (default: complete)")
    p.add_argument("--trim-threshold", type=float, default=0.2,
                   help="trim patterns and averages to the columns holding this share of the largest "
                        "(default: 0.2)")
    p.add_argument("-q", "--qvalue", type=float, default=0.05,
                   help="largest q-value of a database match (default: 0.05)")
    _threads_arg(p)

    p = command("name", cmd_name, "name motifs by the database motifs they resemble most",
                "Name motifs by the database motifs they resemble most, scored over the whole of each "
                "motif as TOMTOM does, with q-values corrected for the size of the database.")
    _motif_args(p)
    db_args(p, "to name the motifs by")
    _out_arg(p)
    p.add_argument("--top", type=int, default=3, help="matches per motif (default: 3)")
    p.add_argument("-q", "--qvalue", type=float, default=0.05,
                   help="largest q-value of a match (default: 0.05)")
    _threads_arg(p)

    # List the commands by group.
    order = [name for _, names in COMMAND_GROUPS for name in names]
    if set(order) != set(sub.choices) or len(order) != len(sub.choices):
        raise RuntimeError("COMMAND_GROUPS must list every command once")
    sub._choices_actions.sort(key=lambda action: order.index(action.dest))
    sub.choices = {name: sub.choices[name] for name in order}
    return parser


def _show_warning(message, category, filename, lineno, file=None, line=None):
    print(f"warning: {message}", file=sys.stderr)


def _unquote(msg: str) -> str:
    """The message without the quotes KeyError puts around it. They are
    removed only if the whole message is wrapped in one matching pair."""
    if len(msg) >= 2 and msg[0] == msg[-1] and msg[0] in "'\"" and msg[0] not in msg[1:-1]:
        return msg[1:-1]
    return msg


def main(argv: list[str] | None = None) -> int:
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "universalmotif"
    if prog not in ("universalmotif", "umotif"):
        prog = "universalmotif"
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    shown = warnings.showwarning
    warnings.showwarning = _show_warning
    setting = None
    if hasattr(args, "progress"):
        from . import _progress

        # The package's setting is put back afterwards, since main() also
        # runs within Python.
        setting = (_progress, _progress._setting)
        choice = args.progress
        if choice is None:
            # UNIVERSALMOTIF_PROGRESS, when set, decides; otherwise bars show on a terminal.
            if os.environ.get("UNIVERSALMOTIF_PROGRESS") is not None:
                choice = _progress.get_progress()
            else:
                choice = _progress.get_progress() or sys.stderr.isatty()
        _progress.set_progress(choice)
    try:
        args.fn(args)
    except BrokenPipeError:
        # The reader went away, as with `| head`, so stop quietly.
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
        msg = _unquote(str(e)) or type(e).__name__
        print(f"{prog} {args.command}: error: {msg}", file=sys.stderr)
        return 1
    finally:
        warnings.showwarning = shown
        if setting is not None:
            setting[0]._setting = setting[1]
    return 0


if __name__ == "__main__":
    sys.exit(main())
