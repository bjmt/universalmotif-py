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
- **Files:** MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE, plain
  matrices and the R package's own format, plus gzipped or plain FASTA for
  sequences.
- **Databases:** motifs fetched from JASPAR, HOCOMOCO and CIS-BP and kept
  in a local cache, and conversion to and from Biopython.
- **Scanning:** exact P-value thresholds, both strands, any alphabet and
  gapped motifs. Whole genomes are scanned straight from a FASTA file in a
  few hundred MB.
- **Genomes:** FASTA, bgzip and 2bit genomes read a piece at a time,
  sequences of peaks, and bigWig signal over them to match backgrounds on
  or to average around motif sites.
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
  *De novo* motif discovery, of gapped motifs and of pairs of motifs that
  favour a spacing too, with refinement and selection of the motifs found.
- **Sequences:** shuffling that keeps k-mer counts, random sequences, k-mer
  backgrounds, complexity, masking and sliding windows.
- **Command line:** `umotif` runs the main functions on files, for shell
  scripts and workflow managers.
- **Plots:** sequence logos, motif trees, and plots of the results above.
  Motifs are shown with their logos in Jupyter, SVG logos can go in
  reports, and logos can be exported to logomaker.

## Which function for which task

The functions most tasks start from, and a guide that covers each task:

### Motifs

| Task | Functions | Guide |
|---|---|---|
| Make a motif from sites, a consensus or a matrix | [`create_motif()`][universalmotif.create_motif] | [Working with motifs](guides/working-with-motifs.ipynb) |
| Read or write motif files | [`read_meme()`][universalmotif.read_meme], [`write_meme()`][universalmotif.write_meme] and the other readers and writers | [Working with motifs](guides/working-with-motifs.ipynb) |
| Get motifs from JASPAR, HOCOMOCO or CIS-BP | [`jaspar()`][universalmotif.databases.jaspar], [`hocomoco()`][universalmotif.databases.hocomoco], [`cisbp()`][universalmotif.databases.cisbp] | [Building a motif library](guides/building-a-library.ipynb) |
| Convert between counts, probabilities and weights | [`convert_type()`][universalmotif.convert_type] | [Sequence motifs](guides/motif-basics.ipynb) |
| Draw logos | [`view_motifs()`][universalmotif.view_motifs], [`view_logo()`][universalmotif.view_logo] | [Getting started](guides/getting-started.ipynb) |
| Compare motifs, name them after a database, or merge similar ones | [`compare_motifs()`][universalmotif.compare_motifs], [`name_motifs()`][universalmotif.name_motifs], [`merge_similar()`][universalmotif.merge_similar], [`motif_tree()`][universalmotif.motif_tree] | [Comparing motifs and P-values](guides/comparison-and-pvalues.ipynb) |

### Sequences

| Task | Functions | Guide |
|---|---|---|
| Read FASTA files and genomes | [`read_fasta()`][universalmotif.read_fasta], [`Genome`][universalmotif.Genome] | [Working with sequences](guides/sequences.ipynb) |
| Take the sequences of peaks from a genome | [`extract_ranges()`][universalmotif.extract_ranges], [`resize_ranges()`][universalmotif.resize_ranges] | [A ChIP-seq workflow](guides/chipseq-workflow.ipynb) |
| Make a background matched for composition | [`match_bkg()`][universalmotif.match_bkg], [`shuffle_sequences()`][universalmotif.shuffle_sequences] | [Choosing a background](guides/background-models.ipynb) |

### Scanning and testing

| Task | Functions | Guide |
|---|---|---|
| Find the sites of motifs in sequences or a genome | [`scan_sequences()`][universalmotif.scan_sequences] | [Scanning, enrichment and discovery](guides/scanning-and-enrichment.ipynb), [Scanning large genomes](guides/large-genomes.ipynb) |
| Choose a score or P-value cutoff | [`motif_pvalue()`][universalmotif.motif_pvalue] | [Comparing motifs and P-values](guides/comparison-and-pvalues.ipynb) |
| Test which motifs are enriched in a set of sequences | [`enrich_motifs()`][universalmotif.enrich_motifs] | [Scanning, enrichment and discovery](guides/scanning-and-enrichment.ipynb) |
| Test where in peaks motifs sit, which occur together, or how near anchors such as promoters | [`motif_peaks()`][universalmotif.motif_peaks], [`motif_coocc()`][universalmotif.motif_coocc], [`motif_proximity()`][universalmotif.motif_proximity] | [A ChIP-seq workflow](guides/chipseq-workflow.ipynb), [Scanning large genomes](guides/large-genomes.ipynb) |
| Look at the spacing of two motifs | [`pair_spacings()`][universalmotif.pair_spacings], [`spacing_periods()`][universalmotif.spacing_periods] | [Gapped motifs](guides/gapped-motifs.ipynb) |
| Average bigWig signal, such as ChIP or ATAC, around motif sites, or motif sites around anchors such as summits | [`signal_matrix()`][universalmotif.signal_matrix], [`site_matrix()`][universalmotif.site_matrix], [`plot_signal_profile()`][universalmotif.plot_signal_profile] | [Scanning large genomes](guides/large-genomes.ipynb), [Motifs and deep learning](guides/deep-learning.ipynb) |
| Score how variants change motif sites | [`variant_motif_effects()`][universalmotif.variant_motif_effects] | [Variant analysis](guides/variants.ipynb) |
| Make a sparse matrix of motif matches for single-cell ATAC | [`match_motifs()`][universalmotif.match_motifs] | [Scanning, enrichment and discovery](guides/scanning-and-enrichment.ipynb) |

### Finding new motifs

| Task | Functions | Guide |
|---|---|---|
| Find motifs enriched in sequences | [`motif_finder()`][universalmotif.motif_finder] | [Scanning, enrichment and discovery](guides/scanning-and-enrichment.ipynb) |
| Find motifs of two parts with a spacer of varying length (experimental) | [`gapped_motif_finder()`][universalmotif.gapped_motif_finder] | [Gapped motifs](guides/gapped-motifs.ipynb) |
| Find pairs of motifs that favour a spacing (experimental) | [`motif_pair_finder()`][universalmotif.motif_pair_finder] | [Gapped motifs](guides/gapped-motifs.ipynb) |
| Choose and refine the motifs found | [`select_motifs()`][universalmotif.select_motifs], [`refine_motifs()`][universalmotif.refine_motifs] | [A ChIP-seq workflow](guides/chipseq-workflow.ipynb), [Motifs from contribution scores](guides/contribution-motifs.ipynb) |

### Sequence models

| Task | Functions | Guide |
|---|---|---|
| Read a model's contribution scores | [`read_contributions()`][universalmotif.read_contributions] | [Motifs and deep learning](guides/deep-learning.ipynb) |
| Find the motifs a model relies on | [`contribution_motifs()`][universalmotif.contribution_motifs] | [Motifs from contribution scores](guides/contribution-motifs.ipynb) |
| Read and draw TF-MoDISco results | [`read_modisco()`][universalmotif.read_modisco], [`plot_modisco()`][universalmotif.plot_modisco] | [TF-MoDISco results](guides/tfmodisco.ipynb) |
| Find where a model used each motif | [`scan_contributions()`][universalmotif.scan_contributions], [`hit_contributions()`][universalmotif.hit_contributions] | [TF-MoDISco results](guides/tfmodisco.ipynb), [Motifs from contribution scores](guides/contribution-motifs.ipynb) |
| Make inputs for a model, and insert sites or shuffle regions to question it | [`one_hot()`][universalmotif.one_hot], [`insert_sites()`][universalmotif.insert_sites], [`shuffle_region()`][universalmotif.shuffle_region], [`mutagenesis()`][universalmotif.mutagenesis] | [Experiments on a model](guides/model-experiments.ipynb) |

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
