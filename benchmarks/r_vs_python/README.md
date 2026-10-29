# Benchmarks against the R package

These scripts time common tasks in the Python package and in the R package
universalmotif on identical inputs, and check that both give the same
results. The results are discussed on the
[benchmarks page](https://bjmt.github.io/universalmotif-py/benchmarks/) of the
documentation.

| File | What it does |
| --- | --- |
| `prepare_data.py` | Downloads the inputs if needed and writes the files both languages read to `data/`. |
| `bench.R` | Times one task with the R package. |
| `bench.py` | Times the same task with the Python package. |
| `run.zsh` | Runs every task in both languages and summarizes the results. |
| `check_outputs.py` | Checks that the two languages agree on each task's result. |
| `summarize.py` | Turns the raw timings into the Markdown table of the documentation. |

## Requirements

- R with Biostrings and universalmotif 1.31.60 or later. This is the
  development version of the R package, whose lite functions are not in a
  Bioconductor release yet.
- Python with universalmotif, polars and numpy.
- macOS or another system whose `/usr/bin/time -l` reports the maximum
  resident set size in bytes. On Linux, `/usr/bin/time -v` reports it in
  kilobytes instead, so `run.zsh` would need a small change.
- zsh, for the driver.

## Running

From this directory, first write the inputs (about 140 MB, and a 36 MB
genome download on the first run):

```sh
python prepare_data.py
```

Then a quick run at small sizes, which also checks that both languages agree:

```sh
./run.zsh --small --runs 3 --check --out results/small
```

And the full benchmark, five runs of every task with one and two threads:

```sh
./run.zsh --check --out results/full
```

`PYTHON` and `RSCRIPT` select the interpreters, `BENCH_R_LIB` names an R
library to load universalmotif from ahead of the others (useful when the
installed version is older), and `UM_R_SOURCE` points to a checkout of the R
package so that its commit is recorded. For example:

```sh
R CMD INSTALL --library=/tmp/Rlib path/to/universalmotif
PYTHON=.venv/bin/python BENCH_R_LIB=/tmp/Rlib UM_R_SOURCE=path/to/universalmotif \
  ./run.zsh --check --out results/full
```

`--tasks "scan_50 scan_400"` runs only the tasks named. A single task can also
be run by hand, which prints its time and a summary of its result:

```sh
Rscript bench.R scan_50 500 1
python bench.py scan_50 500 1
```

## Output

The output directory gets `versions.txt` (the hardware and the versions of
R, Python and the packages), `raw.tsv` (one line per process, with its time
and peak memory), `summary.md` (the medians as a Markdown table) and, with
`--check`, the results of both languages in `check/`.

Each task runs in a separate process. The time is that of the call alone,
without starting the interpreter, loading the packages or reading the inputs.
The memory is the peak resident set size reported by `/usr/bin/time -l`
minus that of a run with `--noop`, which loads the same packages and inputs
but stops before the call.

The results of the run shown in the documentation, made with
`./run.zsh --check --out results/full`, are kept in `2026-10-29/`: its
`versions.txt`, `raw.tsv` and `summary.md`. Every check passed in that run.
