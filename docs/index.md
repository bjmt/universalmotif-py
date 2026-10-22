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
- **Scanning:** exact P-value thresholds, both strands, any alphabet, gapped
  motifs, and whole genomes scanned straight from a FASTA file in a few
  hundred MB.
- **Comparison:** motif similarity with P-values, merging of similar motifs,
  and trees.
- **Statistics:** enrichment, co-occurrence, positional enrichment,
  proximity to genomic anchors, composition-matched backgrounds and *de
  novo* motif discovery.
- **Sequences:** shuffling that keeps k-let counts, random sequences, k-let
  backgrounds, complexity, masking and sliding windows.
- **Plots:** sequence logos, motif trees, and plots of the results above.

## Where to go next

The [installation](installation.md) page explains how to get the package
set up, and the [getting started](guides/getting-started.ipynb) guide walks
through the basics. Every function is described in the
[API reference](reference/index.md).

If you already know the R package, the function names will be familiar,
with dotted argument names written with underscores (`min.ic` becomes
`min_ic`) and the `_lite` versions of functions provided under their plain
names. The few places where the results differ on purpose are listed in
[differences from the R package](differences.md).
