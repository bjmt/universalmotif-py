# Changelog

## 0.1.0 (unreleased)

The first version of the Python port of universalmotif, covering most of
the R package:

- Motif creation, type conversion and manipulation, including trimming,
  rounding, merging, reverse complementing, gaps, alphabet switching and
  filtering.
- Readers and writers for MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE
  and plain matrix files, and for gzipped or plain FASTA files.
- Exact P-values and score thresholds, and scoring helpers for single
  matches.
- Sequence scanning on both strands for any alphabet, including gapped
  motifs, packed sequences and FASTA files scanned a batch at a time.
- Motif comparison, merging, clustering and trees.
- Enrichment, co-occurrence, positional enrichment, proximity to genomic
  anchors, composition-matched backgrounds and *de novo* motif discovery.
- Sequence utilities: shuffling, random sequences, k-mers, complexity,
  masking and sliding windows.
- Sequence logos, motif trees and result plots drawn with matplotlib.
- Links to the wider ecosystem: motifs fetched from JASPAR, HOCOMOCO and
  CIS-BP, conversion to and from Biopython, logos in Jupyter and for
  logomaker, motif match matrices for AnnData and pychromVAR, convolution
  filters, one-hot sequences and TF-MoDISco results, genomes read a piece
  at a time from FASTA, bgzip and 2bit files, bigWig signal for matching
  backgrounds, hits in genomic coordinates, and ranges exchanged with
  PyRanges and polars-bio.
- A command line, `universalmotif` or `umotif`, with subcommands for
  converting, fetching, curating and drawing motifs, scanning, comparison,
  enrichment, discovery, backgrounds, match matrices, proximity,
  co-occurrence and positional enrichment.
- k-mers in place of k-lets, so `count_klets()` and `get_klets()` are now
  `count_kmers()` and `get_kmers()`, and `get_bkg()` has a `kmer` column.
- Canadian spelling throughout, so `summarise_motifs()` is now
  `summarize_motifs()`, and the `best_center` and `centers` columns of
  `motif_peaks()` are `best_centre` and `centres`.
- `motif_peaks()` keeps the motif numbers of `scan_sequences()` in
  `motif_i`, as `motif_proximity()` does.
