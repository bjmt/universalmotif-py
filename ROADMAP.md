# Possible extensions

Ideas for connecting the package to the rest of the Python ecosystem. The
first, motif databases and Biopython conversion, is done; the others are
planned in the order below.

## Plan

In order:

1. Motif databases and Biopython conversion (done: `universalmotif.databases`
   with JASPAR, HOCOMOCO and CIS-BP, and `to_biopython()` and
   `from_biopython()`).
2. Jupyter display with an inline logo, and logomaker export.
3. A sparse motif-by-peak matrix for single-cell ATAC analyses.
4. Deep-learning interoperability, and a check of `compare_motifs()`
   against TOMTOM.
5. Indexed genomes and bigWig covariates.
6. Genomic interval libraries.

A command line is left out, since yamtk already covers that ground.

Every extension stays optional, so the core dependencies remain numpy and
polars. Anything that needs another package becomes an extra, imported only
inside the functions that use it and raising a clear error when it is
missing, as plotting does with matplotlib. Several need no new package at
all: databases can be fetched with the standard library, logos can be drawn
as SVG from the glyph outlines the package already computes, filters can be
exchanged as numpy arrays, and FASTA indexes and 2bit files are simple
enough to read in the Rust crate. Biopython, scipy (with AnnData as a
further option) and pyBigWig would be extras. Each optional function says
so in its docstring, its tests skip when the extra is missing, and one CI
job installs every extra.

## Motif databases and Biopython

- Convert motifs to and from `Bio.motifs` (the counterpart of the R
  package's `convert_motifs()`). This also gives access to
  [pyJASPAR](https://github.com/asntech/pyjaspar), which queries JASPAR
  releases 2014 to 2024 offline and returns Biopython motifs.
- A small `universalmotif.databases` module for fetching and caching motifs
  from the JASPAR REST API, HOCOMOCO and CIS-BP, as MotifDb does in R.

## Genomic intervals

- Accept and return [polars-bio](https://biodatageeks.org/polars-bio/)
  frames (Rust and Arrow, close to our own stack) and
  [PyRanges v1](https://pyranges.readthedocs.io/) objects for anchors,
  universes, exclusions and hits. Both use 0-based, end-exclusive
  coordinates, so conversions need care.
- Report scan hits in genomic coordinates when the sequences are windows
  taken from a genome.

## Indexed genomes and signal tracks

- Read genome windows on demand with `pyfaidx` or `pysam.FastaFile` (and
  `py2bit` for 2bit files), so `match_bkg(genome=...)` and
  `motif_proximity()` do not need the whole genome in memory.
- Take covariates for `match_bkg()` straight from bigWig signal with
  `pyBigWig`.

## Deep-learning tools

- Interoperate with [tangermeme](https://tangermeme.readthedocs.io/) and the
  PyTorch-free memesuite-lite (FIMO and TOMTOM): export motifs as numpy or
  torch convolution filters, import their PWMs, and accept tangermeme's
  `PackedSequence` as sequence input.
- Read TF-MoDISco (modisco-lite) results as CWMs.
- Check `compare_motifs()` against TOMTOM on shared examples.

## Single-cell ATAC

- A sparse motif-by-peak matrix (scipy or AnnData) from scan results, as
  the input to chromVAR-style analyses in the scverse ecosystem.

## Notebooks and pipelines

- An HTML representation of `Motif` in Jupyter with a small inline logo.
- Export to `logomaker` DataFrames.
- A `universalmotif` command line for converting, scanning and enrichment
  steps in Snakemake and Nextflow workflows.
