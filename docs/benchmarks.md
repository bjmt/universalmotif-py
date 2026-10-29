# Benchmarks

This page compares the speed and memory use of this package with those of
the [universalmotif](https://bioconductor.org/packages/universalmotif/) R
package it was ported from, on common tasks given identical inputs and
matching parameters. Both sides were checked to give the same results, so
the comparison is between two implementations of the same calculations
rather than two different methods.

The R package has a full and a `_lite` version of several functions, and
the lite ones are the counterparts of the functions here: they scan,
compare, merge and test enrichment with the same algorithms and the same
P-values. So the R side of these benchmarks uses `scan_sequences_lite()`,
`enrich_motifs_lite()`, `compare_motifs_lite()` and `merge_similar_lite()`,
which are also the faster choices in R.

## Results

This package was faster at every task, by 4 to 98 times, and most often by
20 to 90 times. Scanning and enrichment gained the most, at 54 to 98 times,
then reading, writing and converting motifs at 33 to 51 times, and
comparing and merging motifs, shuffling sequences, counting k-mers and
sequence complexity at 14 to 40 times. The two smallest gains were for
creating a motif from sites and finding score thresholds, where the R
package already does almost all of the work in compiled code, in Biostrings
for the first and in C++ for the second. Both packages ran about twice as
fast with two threads, apart from k-mer counting and sequence complexity in
R, which gained little.

This package also needed less memory for most tasks, and much less for
some: sequence complexity took 0.6 GB against 2.7 GB in R, counting k-mers
0.1 to 0.2 GB against 0.8 GB, and creating a motif from two million sites
0.3 GB against 1.1 GB. The exceptions were the scans for 400 motifs, the
scans with two threads and the score thresholds, which took a few tens of
MB more than in R, and at most 110 MB.

Times are medians of five runs, and memory is the most the task added at
its peak to what the loaded packages and inputs already used.

| Task | Size | Threads | R | Python | Speed-up | R memory | Python memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Read MEME motifs | 8,790 motifs | 1 | 1.4 s | 0.0339 s | 41x | 26 MB | 22 MB |
| Write MEME motifs | 8,790 motifs | 1 | 1.26 s | 0.0379 s | 33x | 100 MB | 20 MB |
| Convert motifs to PWMs | 87,900 motifs | 1 | 5.92 s | 0.115 s | 51x | 203 MB | 81 MB |
| Create a motif from sites | 2,000,000 sites | 1 | 1.67 s | 0.367 s | 5x | 1,071 MB | 306 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 1 | 36.1 s | 0.385 s | 94x | 20 MB | 10 MB |
| Scan for 50 motifs | 10,000 500 bp sequences | 2 | 18.2 s | 0.21 s | 87x | 24 MB | 74 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 1 | 35.9 s | 0.572 s | 63x | 25 MB | 47 MB |
| Scan for 400 motifs | 1,000 500 bp sequences | 2 | 18.1 s | 0.337 s | 54x | 29 MB | 110 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 1 | 24.5 s | 0.295 s | 83x | 138 MB | 26 MB |
| Scan a chromosome for 10 motifs | 30,427,671 bp | 2 | 12.5 s | 0.154 s | 81x | 146 MB | 24 MB |
| Enrichment of 100 motifs | 1,000 peaks | 1 | 35.6 s | 0.362 s | 98x | 82 MB | 36 MB |
| Enrichment of 100 motifs | 1,000 peaks | 2 | 18 s | 0.2 s | 90x | 83 MB | 36 MB |
| Score thresholds for 100 P-values | 879 motifs | 1 | 13.7 s | 3.5 s | 4x | 20 MB | 38 MB |
| Score thresholds for 100 P-values | 879 motifs | 2 | 6.94 s | 1.95 s | 4x | 28 MB | 46 MB |
| Compare all against all, with P-values | 879 motifs | 1 | 49.2 s | 2.1 s | 23x | 323 MB | 256 MB |
| Compare all against all, with P-values | 879 motifs | 2 | 25.2 s | 1.22 s | 21x | 317 MB | 257 MB |
| Merge similar motifs | 879 motifs | 1 | 48.8 s | 2.03 s | 24x | 330 MB | 237 MB |
| Merge similar motifs | 879 motifs | 2 | 25.3 s | 1.19 s | 21x | 318 MB | 240 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 1 | 3.57 s | 0.199 s | 18x | 206 MB | 118 MB |
| Shuffle sequences, k = 1 | 100,000 500 bp sequences | 2 | 2.08 s | 0.122 s | 17x | 215 MB | 119 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 1 | 11 s | 0.679 s | 16x | 219 MB | 118 MB |
| Shuffle sequences, k = 2 | 100,000 500 bp sequences | 2 | 6.08 s | 0.442 s | 14x | 223 MB | 120 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 1 | 15.2 s | 0.702 s | 22x | 806 MB | 104 MB |
| Count 1-, 2- and 3-mers | 100,000 500 bp sequences | 2 | 12.3 s | 0.425 s | 29x | 805 MB | 168 MB |
| Sequence complexity | 100,000 500 bp sequences | 1 | 25 s | 0.769 s | 32x | 2,653 MB | 591 MB |
| Sequence complexity | 100,000 500 bp sequences | 2 | 22.4 s | 0.561 s | 40x | 2,782 MB | 654 MB |

## The tasks

Each task calls one function on each side, with the same arguments wherever
the two have them and the defaults otherwise.

| Task | R | Python |
| --- | --- | --- |
| Read MEME motifs | `read_meme()` | [`read_meme()`][universalmotif.read_meme] |
| Write MEME motifs | `write_meme()` | [`write_meme()`][universalmotif.write_meme] |
| Convert motifs to PWMs | `convert_type(motifs, "PWM")` | [`convert_type(motifs, "PWM")`][universalmotif.convert_type] |
| Create a motif from sites | `create_motif(sites, type = "PCM")` | [`create_motif(sites, type="PCM")`][universalmotif.create_motif] |
| Scan for motifs | `scan_sequences_lite(pvalue = 1e-4, return.granges = FALSE)` | [`scan_sequences(pvalue=1e-4)`][universalmotif.scan_sequences] |
| Enrichment | `enrich_motifs_lite(motifs, peaks, background, qvalue = 1)` | [`enrich_motifs(motifs, peaks, background, qvalue=1)`][universalmotif.enrich_motifs] |
| Score thresholds | `motif_pvalue(motifs, pvalue = ...)` | [`motif_pvalue(motifs, pvalue=...)`][universalmotif.motif_pvalue] |
| Compare motifs | `compare_motifs_lite(motifs, matrix.out = "pvalue")` | [`compare_motifs(motifs, matrix_out="pvalue")`][universalmotif.compare_motifs] |
| Merge similar motifs | `merge_similar_lite(motifs)` | [`merge_similar(motifs)`][universalmotif.merge_similar] |
| Shuffle sequences | `shuffle_sequences(k = 1)` and `k = 2` | [`shuffle_sequences(k=1)`][universalmotif.shuffle_sequences] and `k=2` |
| Count k-mers | `get_bkg(k = 1:3)` | [`get_bkg(k=[1, 2, 3])`][universalmotif.get_bkg] |
| Sequence complexity | `sequence_complexity(window.size = 20)` | [`sequence_complexity(window_size=20)`][universalmotif.sequence_complexity] |

The inputs come from public data, written once to plain files by a script
using only the Python standard library, so that both languages read the
same sequences and motifs:

- **Motifs** are the 879 vertebrate motifs of JASPAR 2024 (CORE,
  non-redundant), repeated to make the larger sets, and for the chromosome
  scan and the enrichment test the first 10 and 100 of its 805 plant motifs.
  The motif tasks other than reading and writing take the first motifs of
  the vertebrate set, as many as the size says.
- **Sequences** are 500 bp windows drawn at random from the five nuclear
  chromosomes of the *Arabidopsis thaliana* TAIR10 genome, keeping only those
  without N or ambiguity letters, and chromosome 1 in full (30.4 Mb) for the
  long-sequence scan.
- **Sites** for creating a motif are 20 bp windows drawn in the same way.
- **Peaks** are the ChIP-seq peaks of VAL1 from Yuan et al. (2021), used in
  [A ChIP-seq workflow](guides/chipseq-workflow.ipynb), about 1 kb each. The
  background for the enrichment test is one random genomic window per peak,
  of the same length and from the same chromosome. It is given to both
  sides, since a shuffled background would differ between them.
- **P-values** for the score thresholds are 100 values spaced evenly on a log
  scale from 10<sup>-2</sup> to 10<sup>-8</sup>, for each motif.

## How it was measured

Every run of a task is a separate process, started under `/usr/bin/time -l`
for its peak memory. The process loads the package and the task's inputs,
then times the call alone, so starting the interpreter, loading the package
and reading the inputs are left out of the times, while building the
result, a data frame in R and a polars DataFrame here, is included. The
memory of a task is its peak resident set size minus that of a
run which loads the same package and inputs but stops before the call, so
it is the memory the task needed beyond its inputs.

Each task ran five times in each language, alternating between them, and
the table gives the medians. The tasks that can use several threads ran with
one and with two, where `nthreads` was passed to both packages and polars
was limited to the same number of threads.

### Hardware and versions

| | |
| --- | --- |
| Machine | Apple M5 Pro, 15 cores, 48 GB |
| System | macOS 26.4.1 |
| R | 4.4.1, with universalmotif 1.31.60 (commit `aa72ba5`) and Biostrings 2.72.1 |
| Python | 3.12.4, with universalmotif 0.3.0 (commit `cc82e49`), polars 1.44.2 and numpy 2.5.4 |
| Date | 29 October 2026 |

## Do the results agree?

Every task was also run once more in both languages with its result saved,
and the two compared:

- The motifs read, written, converted, created and merged are the same,
  and the MEME files written are identical.
- The hits of every scan are the same, at the same places with the same
  scores and P-values, once their positions are converted, since the R
  package counts from 1 and includes the end while this package counts from
  0 and leaves the end out (see
  [Differences from the R package](differences.md)).
- The enrichment tests give the same hit counts for every motif, and
  P-values and q-values that agree to within 10<sup>-6</sup> of their size.
- The comparison P-values agree to within 10<sup>-6</sup> of their size,
  the score thresholds and k-mer counts are the same, and so are the
  complexities, up to rounding error.
- Shuffled sequences cannot be compared directly, since the two packages
  draw different random numbers, so instead both were checked to keep the
  k-mer counts of every sequence they shuffle.

## Things to keep in mind

- **Fixed costs in R.** Scanning in R builds the score distribution of
  every motif once for its threshold and again for the P-values of its
  hits, which took about 25 ms per motif on this machine however few
  sequences there were. With many motifs and few sequences this dominates
  the time in R, which is why the enrichment test and the chromosome scan
  use 100 and 10 motifs rather than all 805 plant motifs: with all of them,
  enrichment in only 100 peaks took 49 s in R. The speed-up therefore
  depends on the shape of a task as well as its size.
- **Startup is not counted.** Loading the R package and Biostrings takes
  about a second, and importing this package a fraction of one, but neither
  is in the times above, nor is reading the FASTA files, which Biostrings
  does in R.
- **Memory.** Peak resident set size depends on the memory allocator and,
  in R, on when the garbage collector runs, so small differences in memory
  mean little; for a few of the sequence tasks, the peak of this package
  varied by 60 to 90 MB from run to run. A task whose peak stayed below
  that of loading its inputs would be shown as using 0 MB.
- **Threads.** Only one and two threads were tried. The machine was
  otherwise idle, and the times of each task varied by less than a fifth
  between runs, mostly by far less, but other work on a machine can change
  them, especially those of tasks lasting well under a second.
- **One machine.** All runs were on a single Mac, so the numbers will
  differ on other hardware and operating systems, and so may the gaps
  between the two packages.
- **Tasks left out.** Plots and trees are drawn with ggplot2 in R and
  matplotlib here, which do different amounts of work, so `motif_tree()`
  and the logo functions are not compared. Random sequences and motifs,
  `motif_finder()`, `match_bkg()` and the other functions built on random
  draws were left out for the same reason as shuffles: their results cannot
  be checked against each other. The full versions of the R functions that
  have a lite counterpart do more, such as gapped and higher-order scanning
  or other comparison metrics, and so are not compared either.

## Running them yourself

The scripts are in the
[`benchmarks/r_vs_python`](https://github.com/bjmt/universalmotif-py/tree/main/benchmarks/r_vs_python)
directory of the repository, whose README lists what they need. From that
directory, these commands download and write the inputs, then run every task
with checks of the results:

```sh
python prepare_data.py
./run.zsh --check --out results/full
```

The table of medians is written to `results/full/summary.md`, and the
versions and hardware to `results/full/versions.txt`; those of the run on
this page, with the time of every run, are kept in the `2026-10-29`
directory next to the scripts. The whole run took 52 minutes, and
`./run.zsh --small` runs every task at a fraction of its size in a few
minutes.
