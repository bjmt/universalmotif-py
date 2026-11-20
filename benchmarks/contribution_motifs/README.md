# Time and memory of contribution_motifs()

These scripts time `contribution_motifs()` against TF-MoDISco on the same
regions and scores, and `contribution_motifs()` with and without its cap on
the seqlets it searches. The results are on the
[benchmarks page](https://bjmt.github.io/universalmotif-py/benchmarks/#motifs-from-contribution-scores)
of the documentation, which also says how the motifs found were checked.

| File | What it does |
| --- | --- |
| `prepare.py` | Writes the inputs from ENCODE's files for the chromBPNet model of ATAC-seq in K562 cells: 30,000 regions with hypothetical scores, or every region with observed scores. |
| `run_cm.py` | Runs `contribution_motifs()` once on prepared inputs. |
| `run.zsh` | Prepares the inputs if needed and runs every comparison, each in its own process. |
| `scaling.zsh` | Runs both methods on nested samples of the regions, to see how each grows with them. |
| `treemem.py` | Runs a command and samples the memory of all its processes, since TF-MoDISco forks workers. |
| `plot_scaling.py` | Draws the scaling figure of the benchmarks page from `results/scaling.tsv`. |
| `results/scaling.tsv` | The time, peak memory and seqlets of every scaling run. |

## How each method is run

| | `contribution_motifs()` | TF-MoDISco |
| --- | --- | --- |
| Version | this package | modisco-lite 3.0.0 |
| Run as | `contribution_motifs(observed, sequences, seed=1, nthreads=2)`, and with `hypothetical` | `modisco motifs -s ohe.npy -a hyp.npy -n 50000 -w 500` |
| Settings | its defaults | the command's defaults: 2 Leiden runs per clustering, a seqlet FDR of 0.05, seqlet cores of 20 and flanks of 5 |
| Seqlets searched | at most 50,000 of each sign | at most 50,000 of each sign |
| Scores | observed, plus hypothetical where stated | hypothetical |
| Threads | 2 | 2 (`NUMBA_NUM_THREADS` and `OMP_NUM_THREADS`) |

TF-MoDISco clusters the seqlets in two rounds and then subclusters each
pattern, running each Leiden clustering from several random seeds and
keeping the best. The modisco command uses 2 seeds, where its library's
`TFMoDISco()` uses 50 by default and also a seqlet FDR of 0.2. Run
through the library with those two defaults on 21,000 of the
30,000 regions below, TF-MoDISco took 40 to 43 minutes, about ten times as
long as the command. The command is how ENCODE and most users run
TF-MoDISco, so it is the one compared here.

## Requirements

- universalmotif with its `bigwig` and `h5py` extras.
- modisco-lite 3.0.0, in an environment of its own if need be; set
  `MODISCO` to its `modisco` command.
- ENCODE's files in `~/.cache/universalmotif-docs`: the peaks
  (ENCFF834JBL.bed.gz), the bigWig of the counts head's observed scores
  (ENCFF407GCO.bigWig), the fold-mean score file
  (encff945wik/seq_contrib.counts.fold_mean.all_regions.ENCSR868FGK.h5)
  and the hg38 genome (hg38.2bit).
- macOS, whose `/usr/bin/time -l` reports peak memory.

## Running

```sh
PYTHON=path/to/python MODISCO=path/to/modisco ./run.zsh work
```

The other rows of the seqlet cap table on the benchmarks page use the same
steps with other models' files: `prepare.py all BIGWIG BED OUTDIR` and then
`run_cm.py OUTDIR` with and without `max_seqlets=None`.

## Results

On an Apple M5 Pro with 48 GB, macOS 26.4.1, single runs. Times and peak
memory are those of each whole process from `/usr/bin/time -l`, inputs
included, with memory in units of 10^9 bytes.

30,000 K562 regions of 500 bp, 20 November 2026:

| Run | Time | Memory |
| --- | ---: | ---: |
| TF-MoDISco | 254 s | 3.6 GB |
| `contribution_motifs()` | 27.1 s | 3.2 GB |
| `contribution_motifs()`, with `hypothetical` | 38.0 s | 2.7 GB |

Every region, with and without the cap of 50,000 seqlets, 19 November 2026:

| Model | Regions | Seqlets searched | Time, sample | Time, every seqlet | Memory, sample | Memory, every seqlet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chromBPNet K562, counts | 186,881 | 50,000 of 688,674 | 40.2 s | 308 s | 7.3 GB | 10.1 GB |
| chromBPNet K562, profile | 186,881 | 50,000 of 193,411 | 35.5 s | 85.2 s | 3.4 GB | 4.0 GB |
| chromBPNet HepG2, counts | 172,191 | 50,000 of 653,337 | 44.1 s | 292 s | 5.1 GB | 8.9 GB |
| BPNet NFIC, counts | 13,723 | 50,000 of 63,317 | 26.7 s | 29.3 s | 2.1 GB | 2.0 GB |

On the large sets the peak memory varies by about a gigabyte from one run to
the next.

Nested samples of 10,000 to 160,000 regions, 20 November 2026, with
TF-MoDISco given `-n 1000000` as ENCODE ran it and `contribution_motifs()`
given hypothetical scores (`./scaling.zsh work 10000 20000 40000 80000
160000`). TF-MoDISco's memory is the peak over all its processes, which was
that of its main process at every size. The times, memory and seqlets of
every run are in `results/scaling.tsv`:

| Regions | TF-MoDISco | `contribution_motifs()`, every seqlet | `contribution_motifs()`, default cap |
| ---: | ---: | ---: | ---: |
| 10,000 | 72 s, 1.5 GB, 9,192 seqlets | 25 s, 1.5 GB, 36,725 seqlets | 25 s, 1.5 GB |
| 20,000 | 154 s, 2.4 GB, 18,564 seqlets | 44 s, 2.4 GB, 73,074 seqlets | 38 s, 2.7 GB |
| 40,000 | 362 s, 4.5 GB, 37,078 seqlets | 89 s, 4.8 GB, 145,841 seqlets | 44 s, 3.4 GB |
| 80,000 | 791 s, 7.6 GB, 72,501 seqlets | 161 s, 5.8 GB, 291,020 seqlets | 47 s, 4.8 GB |
| 160,000 | 2,176 s, 10.5 GB, 148,788 seqlets | 346 s, 7.1 GB, 588,412 seqlets | 58 s, 7.4 GB |
