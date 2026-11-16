# universalmotif

`universalmotif` is a Python package for creating sequence motifs,
converting between their representations, and reading and writing them in
the common file formats. It also uses motifs to scan, compare and test
sequences. It is a port of the
[universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package. The heavy lifting is done by a Rust crate of the
same name, so most of it is quite fast and can use several threads.

Motifs are stored as [`Motif`][universalmotif.Motif] objects, numeric
matrices come back as numpy arrays, and tables as
[polars](https://pola.rs/) DataFrames.

## A quick look

```python
import universalmotif as um

# A motif from a few aligned sites, then as a log-odds matrix.
ap1 = um.create_motif(["TGACTCA", "TGAGTCA", "TGACTAA"], pseudocount=1)
print(um.convert_type(ap1, "PWM").matrix)

# Add the example motifs, and scan the example sequences with all of them.
motifs = [ap1] + um.example_motifs()
sequences = um.example_sequences("targets")
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
- **Scanning:** exact P-value thresholds, both strands, any alphabet and
  gapped motifs. Whole genomes are scanned straight from a FASTA file in a
  few hundred MB.
- **Genomes:** FASTA, bgzip and 2bit genomes read a piece at a time,
  sequences of peaks, and bigWig signal over them to match backgrounds on.
- **Genomic intervals:** hits moved back to genomic coordinates, and ranges
  exchanged with PyRanges and polars-bio.
- **Single-cell ATAC:** sparse peak by motif match matrices, ready for
  AnnData and chromVAR-style analyses.
- **Deep learning:**
    - motifs as convolution filters and back, and one-hot sequences
    - TF-MoDISco results read as CWMs
    - seqlets and motifs found directly in a model's contribution scores
    - attribution scores drawn along sequences
    - batches for in-silico mutagenesis and motif insertion experiments
- **Comparison:** motif similarity with P-values, merging of similar motifs,
  and trees.
- **Statistics:** enrichment, co-occurrence, positional enrichment,
  proximity to genomic anchors and composition-matched backgrounds.
  *De novo* motif discovery, of gapped motifs too, with refinement and
  selection of the motifs found.
- **Sequences:** shuffling that keeps k-mer counts, random sequences, k-mer
  backgrounds, complexity, masking and sliding windows.
- **Command line:** `umotif` runs the main functions on files, for shell
  scripts and workflow managers.
- **Plots:** sequence logos, motif trees, and plots of the results above.
  Motifs are shown with their logos in Jupyter, SVG logos can go in
  reports, and logos can be exported to logomaker.

## Where to go next

The [installation](installation.md) page explains how to set up the
package, and the [getting started](guides/getting-started.ipynb) guide
walks through the basics. The guides that follow it include:

- [Sequence motifs](guides/motif-basics.ipynb) explains what the different
  motif types mean.
- [A ChIP-seq workflow](guides/chipseq-workflow.ipynb) runs a full analysis
  of real peaks.
- [Building a motif library](guides/building-a-library.ipynb) combines
  motifs from several databases.
- [Motifs and deep learning](guides/deep-learning.ipynb) introduces the
  contribution scores of a sequence model.
- [TF-MoDISco results](guides/tfmodisco.ipynb) reads, names and draws a
  model's TF-MoDISco patterns.
- [Motifs from contribution scores](guides/contribution-motifs.ipynb) finds
  a model's motifs directly from its scores.
- [Experiments on a model](guides/model-experiments.ipynb) makes inputs for a
  model and questions it with changed sequences.
- [Variant analysis](guides/variants.ipynb) scores how the variants of a
  VCF file change the motif sites they fall in.

Every function is described in the [API reference](reference/index.md).

If you already know the R package, the function names will be familiar.
Dotted argument names are written with underscores, so `min.ic` becomes
`min_ic`, and the `_lite` versions of functions are provided under their
plain names. The few places where the results differ on purpose are listed
in [differences from the R package](differences.md).
