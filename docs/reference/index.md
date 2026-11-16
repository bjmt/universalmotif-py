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
- [Genomes and signal](genomes.md): genomes read from disk a piece at a
  time, the sequences of genomic ranges, and bigWig signal over them.
- [Genomic intervals](genomic.md): matched backgrounds and proximity to
  anchors.
- [Plotting](plotting.md): logos, trees and plots of results.
- [Databases and Biopython](databases.md): motifs from JASPAR, HOCOMOCO
  and CIS-BP, and conversion to and from Biopython.
- [Contribution scores](contributions.md): a model's contribution scores
  and TF-MoDISco results, and where the model used each motif.
- [Motifs from contribution scores](contribution-motifs.md): finding a
  model's motifs from its scores.
- [Model inputs and experiments](model-inputs.md): one-hot sequences,
  convolution filters, in silico experiments and variants for a model.
- [Example data](data.md): synthetic motifs, sequences and files for trying things out.

Most functions that do a lot of work take an `nthreads` argument, where 0
uses every core. Left out, it takes the package setting. That is 1 unless
changed with [`set_nthreads()`][universalmotif.set_nthreads], or with the
`UNIVERSALMOTIF_NTHREADS` environment variable set before the package is
imported. Functions that use random numbers take a `seed`. The same seed gives
the same results again, and without one they draw a fresh seed each time.

::: universalmotif.set_nthreads

::: universalmotif.get_nthreads

Functions that can run for a long time, such as scanning a genome or
searching for motifs, take a `progress` argument. With it, a bar appears on
stderr once the call has run for a second, and Ctrl-C stops the call. Left
out, it takes the package setting. That is off unless changed with
[`set_progress()`][universalmotif.set_progress], or with the
`UNIVERSALMOTIF_PROGRESS` environment variable.

::: universalmotif.set_progress

::: universalmotif.get_progress
