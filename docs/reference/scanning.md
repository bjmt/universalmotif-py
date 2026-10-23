# Scanning

Scanning sequences for motif hits. Sequences can be strings, lists or dicts of strings, [`PackedSequences`][universalmotif.PackedSequences], or the path to a FASTA file, as described under [`scan_sequences()`][universalmotif.scan_sequences].

::: universalmotif.scan_sequences

::: universalmotif.dedup_hits

## Motif match matrices

Which motifs match which sequences, as a sparse matrix for single-cell ATAC
analyses, without keeping the hits themselves. These need the `scipy` extra,
and AnnData output the `anndata` extra.

::: universalmotif.match_motifs

::: universalmotif.MotifMatches

::: universalmotif.add_motif_matches
