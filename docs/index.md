# universalmotif

`universalmotif` is a Python package for working with sequence motifs:
creating them, converting between their different representations, reading
and writing them in the common file formats, and using them to scan,
compare and test sequences. It is a port of the
[universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package, built on a Rust crate of the same name that does the
heavy lifting, so most of what it does is quite fast and can use several
threads.

Motifs are stored as [`Motif`][universalmotif.Motif] objects, numeric matrices come back as numpy
arrays, and tables come back as [polars](https://pola.rs/) DataFrames.

## A quick look

```python
import universalmotif as um

# A motif from a few aligned sites, then as a log-odds matrix.
ap1 = um.create_motif(["TGACTCA", "TGAGTCA", "TGACTAA"], pseudocount=1)
print(um.convert_type(ap1, "PWM").matrix)

# Read some more from a MEME file, and scan sequences with all of them.
motifs = [ap1] + um.read_meme("motifs.meme")
sequences = um.read_fasta("promoters.fa")
hits = um.scan_sequences(motifs, sequences, pvalue=1e-4)

# How similar are they, and what do they look like?
similarity = um.compare_motifs(motifs)
um.view_motifs(motifs[:4]).savefig("logos.png")
```

## What it can do

- **Motifs:** create them from sites, consensus strings or matrices;
  convert between count, probability, weight, information content and
  contribution matrices; trim, round, merge, reverse complement and filter
  them.
- **Files:** MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE and plain
  matrices, plus gzipped or plain FASTA for sequences.
- **Databases:** motifs fetched from JASPAR, HOCOMOCO and CIS-BP and kept
  in a local cache, and conversion to and from Biopython.
- **Scanning:** exact P-value thresholds, both strands, any alphabet, gapped
  motifs, and whole genomes scanned straight from a FASTA file in a few
  hundred MB.
- **Genomes:** FASTA, bgzip and 2bit genomes read a piece at a time,
  sequences of peaks, and bigWig signal over them to match backgrounds on.
- **Genomic intervals:** hits moved back to genomic coordinates, and ranges
  exchanged with PyRanges and polars-bio.
- **Single-cell ATAC:** sparse peak by motif match matrices, ready for
  AnnData and chromVAR-style analyses.
- **Deep learning:** motifs as convolution filters and back, one-hot
  sequences, TF-MoDISco results read as CWMs, attribution scores drawn
  along sequences, and batches for in-silico mutagenesis and motif
  insertion experiments.
- **Comparison:** motif similarity with P-values, merging of similar motifs,
  and trees.
- **Statistics:** enrichment, co-occurrence, positional enrichment,
  proximity to genomic anchors, composition-matched backgrounds and *de
  novo* motif discovery.
- **Sequences:** shuffling that keeps k-mer counts, random sequences, k-mer
  backgrounds, complexity, masking and sliding windows.
- **Command line:** `umotif` runs the main functions on files, for shell
  scripts and workflow managers.
- **Plots:** sequence logos, motif trees, and plots of the results above,
  motifs shown with their logos in Jupyter, SVG logos for reports, and
  export to logomaker.

## Where to go next

The [installation](installation.md) page explains how to get the package
set up, and the [getting started](guides/getting-started.ipynb) guide walks
through the basics. The guides that follow it go from what the different
motif types mean in [sequence motifs](guides/motif-basics.ipynb) through to
a full analysis of real peaks in [a ChIP-seq
workflow](guides/chipseq-workflow.ipynb), and [building a motif
library](guides/building-a-library.ipynb) shows how to combine motifs from
several databases, while [motifs and deep
learning](guides/deep-learning.ipynb) reads the patterns and contribution
scores of a sequence model. Every function is described in the
[API reference](reference/index.md).

If you already know the R package, the function names will be familiar,
with dotted argument names written with underscores (`min.ic` becomes
`min_ic`) and the `_lite` versions of functions provided under their plain
names. The few places where the results differ on purpose are listed in
[differences from the R package](differences.md).
