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

This package was 5 to 115 times faster, and usually needed less memory.
Times are medians of five runs, and memory is the peak a task added beyond
its loaded inputs.

<div class="bench-table" markdown>
| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.41 s | 0.0351 s | 40x | 26 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.25 s | 0.039 s | 32x | 103 MB | 21 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 5.84 s | 0.113 s | 52x | 203 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.6 s | 0.35 s | 5x | 1,055 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 35.5 s | 0.308 s | 115x | 23 MB | 12 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.2 s | 0.166 s | 110x | 21 MB | 13 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.4 s | 0.494 s | 72x | 29 MB | 105 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18 s | 0.297 s | 61x | 20 MB | 106 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24 s | 0.215 s | 112x | 142 MB | 27 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.3 s | 0.12 s | 103x | 140 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 34.8 s | 0.303 s | 115x | 85 MB | 39 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 17.9 s | 0.168 s | 107x | 81 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.4 s | 0.553 s | 24x | 32 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.88 s | 0.326 s | 21x | 28 MB | 46 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 47.2 s | 2.05 s | 23x | 391 MB | 291 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25 s | 1.18 s | 21x | 367 MB | 292 MB |
| Merge similar motifs | 879 motifs | 1 | 47.2 s | 2.02 s | 23x | 392 MB | 272 MB |
| Merge similar motifs | 879 motifs | 2 | 25.2 s | 1.14 s | 22x | 345 MB | 274 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.37 s | 0.182 s | 18x | 202 MB | 172 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 2.01 s | 0.113 s | 18x | 208 MB | 118 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 10.3 s | 0.641 s | 16x | 215 MB | 116 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 5.57 s | 0.444 s | 13x | 215 MB | 119 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 14.3 s | 0.626 s | 23x | 801 MB | 102 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 11.7 s | 0.389 s | 30x | 820 MB | 102 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 24 s | 0.715 s | 34x | 2,623 MB | 590 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 21.3 s | 0.504 s | 42x | 2,697 MB | 590 MB |
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
| Date | 19 November 2026 |

The scripts are in
[`benchmarks/r_vs_python`](https://github.com/bjmt/universalmotif-py/tree/main/benchmarks/r_vs_python),
along with the raw timings of this run, and its README describes how to run
them.

## Motifs from contribution scores

[`contribution_motifs()`][universalmotif.contribution_motifs] was run with
its defaults on the contribution scores of 25 of ENCODE's models: BPNet
models of 16 single factors, chromBPNet models of ATAC-seq in K562, HepG2
and GM12878 cells, and ProCapNet models of PRO-cap in six cell lines.
TF-MoDISco (modisco-lite 3.0.0, default settings, with at most 50,000
seqlets of each sign, as `contribution_motifs()` has by default) was run on
the same regions, and ENCODE's own TF-MoDISco results for each model were
added beside them.

No set of motifs is known to be complete for any of these models, so each
motif was checked for evidence of its own rather than against another
method's:

- outside support, either conservation at its sites in the regions (mean
  phyloP of its sites against their flanks, compared with sites of the same
  GC and CpG content) or more ChIP-seq peaks of the factor it is named after,
  in the same cell line, than expected;
- the model's response, a change in the prediction when the motif is planted
  in 128 shuffled sequences, against copies of it with its columns shuffled.

Each threshold is a Benjamini-Hochberg q-value of 0.01.

<div class="bench-table" markdown>
| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 25.1 / 3.7 | 47.3 / 14.0 | 59.5 / 8.5 | 44.0 / 16.3 |
| `contribution_motifs()`, `max_seqlets=None` | 28.1 / 4.1 | 52.0 / 12.3 | 60.2 / 8.2 | 40.8 / 18.7 |
| `contribution_motifs()`, one seqlet pass | 8.4 / 0.7 | 33.0 / 0.7 | 41.3 / 2.8 | 40.7 / 10.5 |
| TF-MoDISco, same regions | 12.4 / 1.4 | 26.7 / 1.0 | 22.2 / 1.8 | 25.5 / 12.8 |
| TF-MoDISco, ENCODE's results | 10.9 / 0.4 | 46.7 / 2.7 | 28.3 / 2.2 | 24.8 / 13.7 |
</div>

Each cell holds two means per model, separated by a slash: first the
distinct motifs with evidence of either kind (motifs with CWMs at a cosine
of 0.9 or more counted once), then the motifs with none. The one-pass row searched every seqlet, and held motifs from the
second search to `min_gain` itself.

`contribution_motifs()` tends to give several spellings of one site, which
this count favours. Merging the motifs with evidence of every run at a
cosine of 0.8 instead, the share of them each run found is:

<div class="bench-table" markdown>
| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 0.57 | 0.55 | 0.55 | 0.46 |
| `contribution_motifs()`, `max_seqlets=None` | 0.61 | 0.58 | 0.53 | 0.48 |
| `contribution_motifs()`, one seqlet pass | 0.25 | 0.36 | 0.40 | 0.41 |
| TF-MoDISco, same regions | 0.43 | 0.40 | 0.35 | 0.36 |
| TF-MoDISco, ENCODE's results | 0.40 | 0.59 | 0.42 | 0.37 |
</div>

The motifs without evidence are an upper bound on noise, since motifs of
factors without ChIP-seq data, or whose sites are not conserved, land there
too. Split by the kind of evidence, much of what the second seqlet pass adds
on chromBPNet and ProCapNet is backed by the model's response alone:

<div class="bench-table" markdown>
| Run | BPNet | chromBPNet | ProCapNet counts | ProCapNet profile |
| --- | ---: | ---: | ---: | ---: |
| `contribution_motifs()` | 10.8 / 15.3 | 27.3 / 21.0 | 32.5 / 27.8 | 30.3 / 14.0 |
| `contribution_motifs()`, `max_seqlets=None` | 11.7 / 17.1 | 26.7 / 25.3 | 30.8 / 30.2 | 28.8 / 12.2 |
| `contribution_motifs()`, one seqlet pass | 5.0 / 3.9 | 21.7 / 11.7 | 23.7 / 18.3 | 28.5 / 12.2 |
| TF-MoDISco, same regions | 6.4 / 6.1 | 20.0 / 7.0 | 14.3 / 7.8 | 17.2 / 8.3 |
| TF-MoDISco, ENCODE's results | 5.6 / 5.7 | 29.7 / 18.7 | 19.5 / 9.5 | 20.5 / 4.5 |
</div>

Each cell holds two means per model, separated by a slash: first the
motifs with outside support, with or without a response from the model,
then the motifs with the model's response only. These are not merged.

On scores shuffled across and within regions, with nothing to find,
`contribution_motifs()` with two seqlet passes gave 2 motifs over 7 heads of
6 models, where 0.6.0 gave up to 48 per head.

By default `contribution_motifs()` searches at most 50,000 seqlets of each
sign, a random sample of them when there are more. On the machine above,
with 2 threads, the sample made the search 2.7 to 9 times faster on the
chromBPNet models, whose full sets of regions hold about 200,000 to 700,000
seqlets, and it used less memory:

<div class="bench-table" markdown>
| Model | Regions | Seqlets searched | Time, sample | Time, every seqlet | Memory, sample | Memory, every seqlet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chromBPNet K562, counts | 186,881 | 50,000 of 688,674 | 34.9 s | 302 s | 6.8 GB | 9.4 GB |
| chromBPNet K562, profile | 186,881 | 50,000 of 193,411 | 30.0 s | 79.6 s | 3.2 GB | 3.7 GB |
| chromBPNet HepG2, counts | 172,191 | 50,000 of 653,337 | 38.9 s | 287 s | 4.7 GB | 8.3 GB |
| BPNet NFIC, counts | 13,723 | 50,000 of 63,317 | 26.1 s | 28.8 s | 2.0 GB | 1.9 GB |
</div>

Times are of single runs, and memory is the peak of the whole process,
the scores included. On the large sets the peak varies by about a gigabyte
from one run to the next. The other BPNet heads tried, of GATA1 and of NFIC's
profile, had fewer than 50,000 seqlets and so were searched whole either
way.
