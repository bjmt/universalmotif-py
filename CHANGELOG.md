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
- Sequence utilities: shuffling, random sequences, k-lets, complexity,
  masking and sliding windows.
- Sequence logos, motif trees and result plots drawn with matplotlib.
