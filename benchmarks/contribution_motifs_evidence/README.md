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
  shuffled, that is both significant and larger than that of nearly all
  random motifs of about the same width (see below).

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
| `null.py` | Scores with nothing to find, for `contribution_motifs()`. |
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

## The model's response

Any two sequences differ a little in what a model makes of them, so with 128
backgrounds nearly any motif changes the prediction significantly, random
ones included (about a fifth to a third of random motifs passed the test of
significance on its own, in a trial on two heads). The test is
calibrated on random motifs for this reason. For each head, 200 random
motifs are planted on the same backgrounds as its real ones, each made of
columns drawn at random from all of JASPAR's motifs, so that they are as
informative as real motifs but no model could have learned their sites. Half
take their widths from the head's own motifs (between 6 and 50 letters), and
half are spread evenly over 6 to 50 letters.

A random motif's effect grows with its width (the 99th percentile of their
effects roughly doubles from 6 to 12 letters to 31 to 50), so each motif is
held to the random motifs of about its own width. A motif counts as used by
the model when its test is significant (q < 0.01) and its effect is larger
than the 99th percentile of the effects of the random motifs within 3
letters of its width, the window being widened until it holds at least 30 of
them. A composite's width is its span, with the gap at the middle of its
range.

The effects also grow with a motif's total information content, and the
choice between the two is a judgement call. As a check, each motif is also
held to the 99th percentile of the 50 random motifs nearest to it in total
information content, and `evidence_tables.md` gives the first table again
under that rule, with the cells where the two rules differ by more than 2
motifs (or 0.02 in a share) listed under each table. The two rules agree
closely for the motifs found with positive scores.

## How the published results were made

Every row and every check comes from one build of the packages, tagged
`evidence-1`: universalmotif-py `05cb7a6` and universalmotif-rs `b260179`
(with Python 3.12.16, NumPy 2.5.5, Polars 2.1.0 and SciPy 1.18.1). The four
`contribution_motifs()` rows were run with these scripts on 20 November
2026, as `config.py` gives them, through

```python
um.contribution_motifs(obs, strs, hypothetical=hyp, seed=1, nthreads=2, hits=False, **kwargs)
```

with `hyp` left out for observed scores only. Every run found the same
motifs, with the same hits, as the earlier runs of the same settings behind
the tables first published on the benchmarks page (some of which were made
with work in progress).

TF-MoDISco itself was not rerun. Its results for the same regions come from
runs of modisco-lite 3.0.0 on 17 and 18 November 2026, on the same windows and
hypothetical scores as here (checked window by window), and those and
ENCODE's results were read again with the `evidence-1` build. Every check,
including the insertion test (TensorFlow 2.15 and PyTorch 2.5.1, on a GPU),
was then run on all six rows together on 20 November with these scripts.

To check that the runs and checks can be reproduced, the defaults were run
a second time on three models (the counts head of the BPNet model of GATA1 in
K562, that of chromBPNet in K562, and both heads of ProCapNet in K562) and
checked again. The motifs and their hits came out identical, and so did
the conservation and ChIP-seq checks. The insertion effects agreed to about
four significant figures (the GPU's arithmetic is not quite deterministic),
which changed no motif's verdict under either rule. (Note that this second
run drew the same backgrounds, sites and random motifs as the first, so it
only shows that the same draws give the same verdicts.)

To see how much the insertion test depends on its draws, it was then run
again on every row with a new seed (`insertion.py prep --seed 2`), which
draws new backgrounds, sites and random motifs. The effects agreed closely
(a Pearson correlation of 0.99), but the bars much less so (0.69, with over
half moving by more than 25%, since each rests on only 34 to 94 random
motifs), and the P-values moved by a median of about 50-fold. Of 6,529
motifs, 15% changed verdict under the width-matched rule (17% under the
IC-matched one), mostly ones near the bar, and about as many of these
changes come from the new effects as from the new bars. The conservation and
ChIP-seq checks stay the same, so only the motifs with the model's response
alone move: by up to 4.2 a model (`contribution_motifs()` on ProCapNet's
counts head went from 41.5 to 37.8 distinct motifs with evidence), and the
shares by up to 0.04. Differences of a few motifs between rows (such as
those between the `contribution_motifs()` rows on ProCapNet) are therefore
within the noise of the test. The comparisons with TF-MoDISco all hold,
apart from the share on ProCapNet's counts head, where
`contribution_motifs()` goes from 0.52 against 0.49 for ENCODE's results to
0.52 against 0.52. The tables from the new seed are in
`results/evidence_tables_seed2.md`.

The share of motifs found depends on the runs in the pool, since motifs
alike at a cosine of 0.8 are merged across all of them, so adding or leaving
out a run changes every row's share.

## Scores with nothing to find

`null.py` gives each window of seven heads of six models (the counts heads
of the BPNet models of CEBPB in HepG2, and CTCF and GATA1 in K562, of
chromBPNet in HepG2 and K562, and both heads of ProCapNet in K562) the
hypothetical scores of another window chosen at random, turned by a random
offset, and takes the observed scores from them at the window's own
letters. With the defaults and hypothetical scores, at the `evidence-1`
tags, `contribution_motifs()` found 4 motifs over all seven heads on every
window (one on CEBPB, a CEBP-like site, likely since its letters are those of
CEBPB peaks, and three on chromBPNet in HepG2, one of them part of an Alu
repeat), and 2 motifs (both on chromBPNet in HepG2) on the files of the
earlier check of this kind, which used the same shuffle on a sample of up
to 20,000 windows per head. On every window, universalmotif 0.6.0
(installed from PyPI, with its defaults) found 169 motifs over six of the
heads, from 2 on CEBPB to 53 on ProCapNet's counts head, and failed on
ProCapNet's profile head with an invalid PCM (likely the bug with letters
outside the alphabet fixed in universalmotif-rs `30c5e11`, since 3 of its
windows hold an N). All three are in `results/null.tsv`.

## Earlier results

The tables first published on the benchmarks page came from earlier forms
of these scripts, run between 17 and 20 November 2026 in five batches, with
the model's response counted whenever its test was significant. Under the
calibrated rule the motifs with the model's response only fall for every
method, by between a sixth (TF-MoDISco on BPNet) and two thirds
(`contribution_motifs()` on ProCapNet counts), and the motifs with none rise
to match. The rest changed little. The motifs and their hits are the same as
before, but the conservation test's background sites and the insertion
test's backgrounds were drawn anew: about 4% of the conservation verdicts
changed (none of the ChIP-seq ones), and the insertion effects agree closely
with the earlier ones (a Pearson correlation of 0.98), though enough motifs
sat near the old threshold that some cells moved by up to 3 motifs under the
old rule.

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
| `evidence_tables.md` | The three tables, and the first again under the IC-matched rule. |
| `evidence_tables_seed2.md` | The same, with the insertion test drawn again from a new seed. |
| `evidence_runs.tsv` | One row per run (model, head and method): motifs, those with each kind of evidence, distinct motifs with evidence and the share of the pool found. |
| `evidence_runs_ic.tsv` | The same, under the IC-matched rule. |
| `evidence.tsv.gz` | One row per motif, with each test's statistics, the two bars and the verdicts under both rules. |
| `support.tsv.gz` | The conservation and ChIP-seq checks of every motif. |
| `insertion.tsv.gz` | The insertion test of every motif, with its width and total information content. |
| `insertion_random.tsv.gz` | The insertion test of every random motif. |
| `null.tsv` | The motifs found on scores with nothing to find: at `evidence-1` on every window (`all`) and on the earlier check's sample, and with 0.6.0 on every window. |
| `runs.tsv` | The windows, time, peak memory and motifs of every run. The times of the `contribution_motifs()` rows come from one batch of 10 runs side by side on a shared machine, and TF-MoDISco's from the earlier batches, so they are only a rough guide. |
