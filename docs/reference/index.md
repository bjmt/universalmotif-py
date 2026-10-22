# API reference

Everything is imported from the top-level package:

```python
import universalmotif as um
```

The functions are grouped by what they do:

- [Motifs](motifs.md): the [`Motif`][universalmotif.Motif] class, and creating, converting and
  modifying motifs.
- [Reading and writing](io.md): motif files, and FASTA files of sequences.
- [P-values and scores](scoring.md): thresholds, and the scores of
  individual matches.
- [Scanning](scanning.md): finding motif hits in sequences.
- [Comparison and clustering](comparison.md): similarity, merging and trees.
- [Enrichment and discovery](enrichment.md): enrichment tests and *de novo*
  motif discovery.
- [Sequences](sequences.md): random and shuffled sequences, backgrounds,
  complexity and masking.
- [Genomic intervals](genomic.md): matched backgrounds and proximity to
  anchors.
- [Plotting](plotting.md): logos, trees and plots of results.
- [Example data](data.md): synthetic motifs, sequences and files for trying things out.

Most functions that do a lot of work take an `nthreads` argument, which is
1 by default or 0 to use every core. Functions that use random numbers take a
`seed`, and the same seed gives the same results again; without one they
draw a fresh seed each time.
