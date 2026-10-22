# Possible extensions

Ideas for connecting the package to the rest of the Python ecosystem, to
revisit once the performance work is done. None of these are started.

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
