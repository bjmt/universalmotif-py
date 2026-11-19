---
hide:
  - toc
---

# Benchmarks

This page holds two benchmarks: the speed and memory use of this package
against the R package, and the motifs of
[`contribution_motifs()`](#motifs-from-contribution-scores) checked against
outside evidence.

The speed and memory use of this package compared with the
[universalmotif](https://bioconductor.org/packages/universalmotif/) R
package, on the same inputs with matching parameters. The R side uses the
`_lite` functions where they exist, which run the same algorithms as this
package, and every result was checked against the R package's. All agree
but the clusters of merged motifs: this package leaves each motif's
comparison with itself out of the correction of its q-values, which the R
package includes, and so links a few fewer motifs (288 clusters to 268).

## Results

This package was 5 to 100 times faster, and usually needed less memory.
Times are medians of five runs, and memory is the peak a task added beyond
its loaded inputs.

<div class="bench-table" markdown>
| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.37 s | 0.0352 s | 39x | 27 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.23 s | 0.0364 s | 34x | 99 MB | 20 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 6.05 s | 0.125 s | 49x | 211 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.61 s | 0.356 s | 5x | 1,059 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 36.3 s | 0.384 s | 94x | 22 MB | 11 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.4 s | 0.208 s | 88x | 22 MB | 13 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.8 s | 0.555 s | 64x | 22 MB | 47 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18 s | 0.332 s | 54x | 22 MB | 108 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24.5 s | 0.272 s | 90x | 138 MB | 27 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.4 s | 0.148 s | 84x | 132 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 35.6 s | 0.356 s | 100x | 77 MB | 37 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 18 s | 0.197 s | 91x | 83 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.9 s | 0.585 s | 24x | 31 MB | 38 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.92 s | 0.328 s | 21x | 35 MB | 47 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 48.4 s | 2.07 s | 23x | 391 MB | 291 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25.3 s | 1.22 s | 21x | 383 MB | 293 MB |
| Merge similar motifs | 879 motifs | 1 | 48.6 s | 2.06 s | 24x | 398 MB | 273 MB |
| Merge similar motifs | 879 motifs | 2 | 25.5 s | 1.17 s | 22x | 393 MB | 274 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.63 s | 0.196 s | 19x | 200 MB | 115 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 1.99 s | 0.116 s | 17x | 209 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 10.6 s | 0.678 s | 16x | 229 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 5.69 s | 0.45 s | 13x | 219 MB | 117 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 14.7 s | 0.639 s | 23x | 814 MB | 103 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 12 s | 0.404 s | 30x | 800 MB | 104 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 24.6 s | 0.738 s | 33x | 2,616 MB | 649 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 21.7 s | 0.55 s | 39x | 2,708 MB | 591 MB |
</div>

## Tasks and inputs

| Task | R | Python |
| --- | --- | --- |
| Read and write MEME motifs | `read_meme()`, `write_meme()` | [`read_meme()`][universalmotif.read_meme], [`write_meme()`][universalmotif.write_meme] |
| Convert motifs | `convert_type()` | [`convert_type()`][universalmotif.convert_type] |
| Create a motif | `create_motif()` | [`create_motif()`][universalmotif.create_motif] |
| Scan | `scan_sequences_lite()` | [`scan_sequences()`][universalmotif.scan_sequences] |
| Enrichment | `enrich_motifs_lite()` | [`enrich_motifs()`][universalmotif.enrich_motifs] |
| Score thresholds | `motif_pvalue()` | [`motif_pvalue()`][universalmotif.motif_pvalue] |
| Compare | `compare_motifs_lite()` | [`compare_motifs()`][universalmotif.compare_motifs] |
| Merge | `merge_similar_lite()` | [`merge_similar()`][universalmotif.merge_similar] |
| Shuffle | `shuffle_sequences()` | [`shuffle_sequences()`][universalmotif.shuffle_sequences] |
| Count k-mers | `get_bkg()` | [`get_bkg()`][universalmotif.get_bkg] |
| Complexity | `sequence_complexity()` | [`sequence_complexity()`][universalmotif.sequence_complexity] |

Motifs are from JASPAR 2024 (vertebrates, and plants for the chromosome
scan and enrichment), sequences are random 500 bp windows and chromosome 1
of the *Arabidopsis thaliana* TAIR10 genome, and the peaks are those of
[the ChIP-seq guide](guides/chipseq-workflow.ipynb), tested against
length-matched genomic windows.

## Setup

| | |
| --- | --- |
| Machine | Apple M5 Pro, 48 GB, macOS 26.4.1 |
| R | 4.4.1, universalmotif 1.31.66 |
| Python | 3.12.4, universalmotif 0.6.0, polars 1.44.2 |
| Date | 18 November 2026 |

The scripts are in
[`benchmarks/r_vs_python`](https://github.com/bjmt/universalmotif-py/tree/main/benchmarks/r_vs_python),
along with the raw timings of this run, and its README describes how to run
them.

## Motifs from contribution scores

[`contribution_motifs()`][universalmotif.contribution_motifs] was run with
its defaults on the contribution scores of 25 of ENCODE's models: BPNet
models of 16 single factors, chromBPNet models of ATAC-seq in K562, HepG2
and GM12878 cells, and ProCapNet models of PRO-cap in six cell lines. TF-MoDISco
(modisco-lite 3.0.0, default settings) was run on the same regions, and
ENCODE's own TF-MoDISco results for each model were added beside them.

No set of motifs is known to be complete for any of these models, so each
motif was checked for evidence of its own rather than against another
method's:

- outside support, either conservation at its sites in the regions (mean
  phyloP of its sites against their flanks, compared with sites of the same
  GC and CpG content) or more ChIP-seq peaks of the factor it is named after,
  in the same cell line, than expected;
- the model's response, a change in the prediction when the motif is planted
  in 128 shuffled sequences, against copies of it with its columns shuffled.

Each threshold is a Benjamini-Hochberg q-value of 0.01. Motifs alike in
their CWMs (a cosine of 0.9) were counted once.

<div class="bench-table" markdown>
| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 28.4 / 3.8 | 50.7 / 13.7 | 60.2 / 8.2 | 41.7 / 17.8 |
| `contribution_motifs()`, `seqlet_passes=1` | 8.4 / 0.7 | 33.0 / 0.7 | 41.3 / 2.8 | 40.7 / 10.5 |
| TF-MoDISco, same regions | 12.4 / 1.4 | 26.7 / 1.0 | 21.8 / 2.2 | 26.3 / 12.0 |
| TF-MoDISco, ENCODE's results | 10.9 / 0.4 | 46.7 / 2.7 | 28.2 / 2.3 | 24.8 / 13.7 |
</div>

Means per model of the distinct motifs with evidence of either kind, and of
the motifs with none. The single-pass row also holds motifs from the second
search to `min_gain` itself.

The motifs without evidence are an upper bound on noise, since motifs of
factors without ChIP-seq data, or whose sites are not conserved, land there
too. Split by the kind of evidence, much of what the second seqlet pass adds
on chromBPNet and ProCapNet is backed by the model's response alone:

<div class="bench-table" markdown>
| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 12.1 / 17.1 | 27.0 / 23.7 | 31.0 / 30.0 | 28.8 / 13.0 |
| `contribution_motifs()`, `seqlet_passes=1` | 5.0 / 3.9 | 21.7 / 11.7 | 23.7 / 18.3 | 28.5 / 12.2 |
| TF-MoDISco, same regions | 6.8 / 5.8 | 19.3 / 7.7 | 14.3 / 7.5 | 17.3 / 9.0 |
| TF-MoDISco, ENCODE's results | 5.8 / 5.5 | 30.3 / 18.0 | 19.2 / 9.7 | 19.8 / 5.2 |
</div>

Motifs with outside support, with or without a response from the model,
and motifs with the model's response only, per model and not merged.

On scores shuffled across and within regions, with nothing to find, the
defaults gave 2 motifs over 7 heads of 6 models, where 0.6.0 gave up to 48
per head.
