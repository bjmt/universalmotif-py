# Changelog

## Unreleased

- Motifs displayed in Jupyter, and `logo_svg()`, stack the letters of
  each position by height and turn negative letters upside down by
  default, as `view_motifs()` does; `sort_positions` and `flip_neg` turn
  this off.
- Motif and sequence index columns (`motif_i`, `sequence_i`, `subject_i`,
  `target_i` and `universe_i`) are `UInt32` in every table rather than
  `Int64`, which saves memory on large scans; tables still join on them.
- Scanning and enrichment no longer spend about a second per 400 motifs
  checking which motifs can reach the P-value cutoff, and a `Genome` is
  scanned a batch of chromosomes at a time, which makes genome scans about
  twice as fast and lighter on memory.
- `dedup_hits()`, `genomic_hits()`, `motif_peaks()`, `motif_coocc()` and
  `motif_proximity()` work on whole columns rather than row by row in
  Python, which makes them 1.3 to 5 times faster on large hit tables and
  cuts their peak memory by a quarter to a half. `genomic_hits()` and
  `motif_peaks()` now raise a clear error for hits with missing positions.
- `calc_windows()`, `read_contributions()`/`from_one_hot()`, `one_hot()`,
  `read_meme()` and `count_kmers()` on long sequences are 1.7 to 20 times
  faster and several use far less memory.
- Large scans use about a quarter less memory at their peak, and hits in
  sequences longer than 4 Gb keep their positions.
- `normalize()` gives a PCM without a pseudocount a pseudocount of 1.
- Fixes from the Rust crate: `get_matches()` no longer returns words
  scoring just below the score, `score_match()` always agrees with
  scanning, converting with a new pseudocount updates the consensus,
  uniform columns convert to and from IC, merging and `shuffle_motifs()`
  keep one-letter columns as probabilities, `get_bkg(rc=True)` counts
  letters found only on the reverse strands, complexity and FASTA output
  work by letter, TRANSFAC single-site counts are kept, and comparing
  thousands of motifs with P-values is several times faster.

## 0.2.0 (2026-10-24)

- `view_attributions()` draws attribution or contribution scores along a
  sequence, for the bases in the sequence or for every nucleotide, as a
  logo or a heatmap, with motif hits shaded.
- `read_contributions()` reads contribution scores and their sequences
  from chromBPNet HDF5 files, with the genomic range of each sequence, or
  from numpy files as TF-MoDISco takes them.
- `signal_values()` reads the value of a bigWig track at every position
  of a range.
- `mutagenesis()` and `mutagenesis_effects()` make every single-letter
  substitution of a sequence for a model and turn its predictions into
  effects that can be drawn as attributions.
- `insert_sites()` writes the same sites into background sequences, and
  `site_spacings()` places two sites at a range of spacings, for probing
  a model.
- `view_variant()` draws the attribution scores of a sequence and a
  variant of it with their difference.
- `one_hot_batches()` reads genomic ranges as one-hot batches for a model,
  one batch at a time, optionally as windows of one width around their
  middles.
- Logos draw letters below zero upside down by default, so that they read
  outward from zero, and stack them as a mirror image of the letters above
  zero, with the largest furthest out; `flip_neg=False` draws them
  upright.
- `view_motifs()` takes `ax` to draw a single motif into a panel of an
  existing figure.
- A new guide, motifs and deep learning, works through the TF-MoDISco
  patterns and contribution scores of a published chromBPNet model.
- Fixes found by new tests:
  - `score_match()` scores words exactly as `scan_sequences()` does,
    instead of up to 0.001 apart.
  - `Genome` opens FASTA files whose last line has no line ending.
  - `create_sequences()` rejects frequencies that are all zero,
    `motif_proximity()` an empty `dist_thresholds`, `motif_peaks()` hits
    centred outside the sequences, and range tables positions that are
    not whole numbers.
  - `from_filters()` reads filters as the type given rather than guessing
    it, so log-odds filters with no negative values stay as they are.
  - `match_bkg()` matches each target to genome windows of its own width,
    and explains when a target is longer than every chromosome.
  - `umotif proximity --universe` reads a `.txt` file of ranges as ranges
    rather than as chromosome sizes.
  - `view_motifs()` warns that `fit_to_height` is ignored for CWMs, as for
    ICMs and PWMs.
  - Count matrices with an empty column are rejected, and matrices read as
    counts with uneven columns say why.
  - Merging a motif with its reverse complement keeps the first as the
    anchor, and the command line restores the warning format after a run.
- Logos keep their top y-axis tick when the axis ends at a value that
  rounds up, as for information content logos of six-letter alphabets.
- Logos draw negative letters smaller than 0.03, so contribution weight
  matrices of negative TF-MoDISco patterns are no longer blank, and CWM
  logos get round axis ticks and less empty space below zero.
- Letters below zero in logos are separated by the same gaps as letters
  above zero.

## 0.1.0 (2026-10-23)

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
- Positions count from 0 and leave the end out everywhere, as in BED files
  and Python slices, so a hit covers `sequence[start:end]`. Range tables,
  `Genome.subseq()`, `PackedSequences.subseq()`, windows, implant
  positions and masked ranges all follow this, BED files and PyRanges pass
  through unchanged, and window tables have an `end` column in place of
  `stop`.
- k-mers in place of k-lets, so `count_klets()` and `get_klets()` are now
  `count_kmers()` and `get_kmers()`, and `get_bkg()` has a `kmer` column.
- Canadian spelling throughout, so `summarise_motifs()` is now
  `summarize_motifs()`, and the `best_center` and `centers` columns of
  `motif_peaks()` are `best_centre` and `centres`.
- `motif_peaks()` keeps the motif numbers of `scan_sequences()` in
  `motif_i`, as `motif_proximity()` does.
