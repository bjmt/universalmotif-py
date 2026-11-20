# Motifs from contribution scores, checked against outside evidence

These scripts run `contribution_motifs()` and TF-MoDISco on the
contribution scores of 25 of ENCODE's models, and check every motif they
find for evidence of its own. The results are the tables of the
[benchmarks page](https://bjmt.github.io/universalmotif-py/benchmarks/) of
the documentation.

No set of motifs is known to be complete for any of these models, so rather
than scoring one method by how many of another's motifs it finds, each motif
is checked for two kinds of evidence:

- Outside support: conservation at its sites (the mean phyloP of its sites
  less that of their flanks, against sites of the same GC content and CpG
  count placed at random over seqlets), or more ChIP-seq peaks of the factor
  it is named after, in the same cell line, than expected.
- The model's response: a change in the prediction when the motif is
  planted in 128 shuffled sequences, against copies of it with its columns
  shuffled.

A motif with neither is the closest these checks come to calling it noise,
though motifs of factors without ChIP-seq data, or whose sites are not
conserved, land there too.

| File | What it does |
| --- | --- |
| `config.py` | The models, windows, threads and methods. |
| `manifest.tsv` | Each model and head, with the ENCODE files it needs. |
| `models.tsv` | The archives of the models' trained folds. |
| `harness.py` | Downloads everything, prepares the windows and runs every method. |
| `checks.py` | The conservation and ChIP-seq checks, and the evidence tables. |
| `insertion.py` | The model's response to each motif planted in shuffled sequences. |
| `run.sh` | Runs every stage in order. |
| `results/` | The per-model results the tables were summarized from. |

## The models and methods

The models are BPNet models of 16 single factors, chromBPNet models of
ATAC-seq in K562, HepG2 and GM12878 cells, and ProCapNet models of PRO-cap in
six cell lines. Only heads with hypothetical scores are run, since TF-MoDISco
needs them: the counts heads of the BPNet and chromBPNet models, and both
heads of the ProCapNet models (31 heads in all).

Every run uses every window of a model, not a sample of them:

| Models | Window | Windows |
| --- | --- | --- |
| BPNet | 500 letters about each region's summit (or centre), leaving out windows that overlap one kept before | 4,235 to 52,261 |
| chromBPNet | the same | 151,529 to 172,191 |
| ProCapNet | the middle 1,000 letters of each peak's input, for every peak | 29,140 to 59,221 |

Each run has 2 threads, with 10 runs side by side.

| Row of the tables | Method here | What it runs |
| --- | --- | --- |
| `contribution_motifs()` | `cm` | the defaults, with hypothetical scores |
| `contribution_motifs()`, `max_seqlets=None` | `cm_uncapped` | every seqlet rather than a sample of 50,000 of each sign |
| `contribution_motifs()`, one seqlet pass | `cm_one_pass` | one seqlet pass, every seqlet, later rounds held to `min_gain` itself |
| `contribution_motifs()`, observed scores only | `cm_observed` | the defaults, without hypothetical scores |
| TF-MoDISco, same regions | `modisco` | modisco-lite 3.0.0 (from PyPI) on the same windows and hypothetical scores |
| TF-MoDISco, ENCODE's results | `published` | ENCODE's own TF-MoDISco results for each model |

TF-MoDISco is run as `modisco motifs -s ohe.npy -a hyp.npy -n 50000 -w 500`
(1000 for ProCapNet), with `NUMBA_NUM_THREADS` and `OMP_NUM_THREADS` at 2.
Its patterns, and ENCODE's, are read with
`read_modisco(trim_threshold=0.3)`, keeping the positive ones.

## How the published results were made

The results in `results/` come from an earlier form of these scripts, run
between 17 and 20 November 2026 in five batches with other names for the
methods. The scripts here are the same code with the experiments the tables
do not use taken out, and give the same motifs (checked on one model). The
packages behind each row were:

| Row | Run as | universalmotif-py | universalmotif-rs |
| --- | --- | --- | --- |
| `contribution_motifs()` | `max50k`, 19 Nov | `809b051`, with `max_seqlets=50000` given | `30c4c60` |
| `max_seqlets=None` | `passes2_later_0.002`, 18 Nov | uncommitted build on `58284e4` | uncommitted build on `30c5e11` |
| one seqlet pass | `cm`, 18 Nov | the same build | the same build |
| observed scores only | `observed`, 20 Nov | `314bf4f` | `b260179` |
| TF-MoDISco, same regions | `modisco`, 17 and 18 Nov | (not used) | (not used) |

The two rows of 18 November came from work in progress that was later
committed as `54a56b6` to `e5f9e5f` (and `30c4c60` in the crate), after
being rebased on newer commits to `gapped_motif_finder()`. The method names
here give those rows the same settings with the current options. (The
`max_seqlets=None` row, for instance, set `seqlet_passes=2` and
`later_min_gain=0.002` by hand, which are now the defaults.)

The checks themselves ran with the universalmotif of the time: `58284e4`
for the one-pass row, `e5f9e5f` for the defaults, `max_seqlets=None` and
both TF-MoDISco rows, and `314bf4f` for observed scores only. The three
tables were made from one pool of these six runs. The share of
motifs found depends on the runs in the pool, since motifs alike at a
cosine of 0.8 are merged across all of them, so adding or leaving out a run
changes every row's share.

## Requirements

- A Python with universalmotif (with the `h5py` extra), pybigtools,
  pyBigWig and scipy.
- modisco-lite, with its `modisco` command given as `MODISCO` or on the
  PATH.
- TensorFlow 2.15, for the chromBPNet and BPNet models, and PyTorch, for the
  ProCapNet models. These can be other Pythons than the first (see below),
  and a GPU makes the insertion test much faster.
- Disk: well over 100 GB. The hypothetical scores of the chromBPNet models
  are the bulk of it, and ENCODE's archives of them are gzipped, so each is
  streamed whole (about 60 GB a model) to keep its mean over folds.

## Running

From this directory:

```sh
PYTHON=python TF_PYTHON=/path/to/tf/python TORCH_PYTHON=/path/to/torch/python ./run.sh
```

Each stage skips work already done, so the script can be run again after a
failure. `--models` limits every stage to the models matching any of the
regular expressions given, such as `./run.sh --models '^bpnet_NFYA_K562$'`
for a quick test on one small model (a few minutes once its files are
downloaded). Everything goes in `work/`, or in `WORK`, and the tables end up
in `work/checks/evidence_tables.md`.

Do keep in mind that the evidence will not come out exactly as in
`results/`. Each test's P-values are adjusted over every motif checked
together, so checking fewer models, or other methods beside them, shifts
some verdicts, and the sites drawn at random for the conservation and
insertion tests depend on the order of the runs. The motifs themselves come
out the same.

## Results

| File | What it holds |
| --- | --- |
| `evidence_tables.md` | The three tables, as on the benchmarks page. |
| `evidence_runs.tsv` | One row per run (model, head and method): motifs, those with each kind of evidence, distinct motifs with evidence and the share of the pool found. |
| `evidence.tsv.gz` | One row per motif, with each test's statistics and verdicts. |
| `support.tsv.gz` | The conservation and ChIP-seq checks of every motif. |
| `insertion.tsv.gz` | The insertion test of every motif. |
| `runs.tsv` | The windows, time, peak memory and motifs of every run. The times come from four batches on a shared machine, so they are only a rough guide. |
