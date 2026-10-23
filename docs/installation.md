# Installation

The package is not on PyPI yet. Until it is, it has to be built from source,
which needs a Rust toolchain as well as Python 3.10 or later. Once it is on
PyPI, `pip install universalmotif` will fetch a prebuilt wheel and none of
this will be needed.

## From source

1. Install Rust with [rustup](https://rustup.rs/), if you don't have it
   already.
2. Clone the Python package and the Rust crate next to each other, since
   the package builds against the crate in the neighbouring folder:

    ```sh
    git clone git@github.com:bjmt/universalmotif-rs.git
    git clone git@github.com:bjmt/universalmotif-py.git
    ```

3. Install the package with pip, adding the `plot` extra to bring in
   matplotlib for the plotting functions:

    ```sh
    pip install "./universalmotif-py[plot]"
    ```

The build takes a minute or two, as the Rust code is compiled with
optimizations. It also installs the `universalmotif` and `umotif`
commands described in [Command line](command-line.md).

## Optional dependencies

- `plot`: matplotlib, for logos and the other plotting functions.
- `scipy`: SciPy, for the sparse matrices of
  [`match_motifs()`](reference/scanning.md#motif-match-matrices).
- `anndata`: AnnData and SciPy, for storing motif matches with single-cell
  ATAC data.
- `h5py`: h5py, for reading TF-MoDISco results with `read_modisco()`.
- `bigwig`: pybigtools, for reading bigWig signal with `signal_stats()`
  and matching backgrounds on it with `match_bkg()`.
- `biopython`: Biopython, for converting motifs to and from `Bio.motifs`
  with `to_biopython()` and `from_biopython()`. The database functions need
  no extra, since they download with the standard library.
- `pyranges`: PyRanges, for `to_pyranges()`. PyRanges objects can be
  given as ranges without it.
- `polars-bio`: polars-bio, for `to_polars_bio()`, which marks tables as
  1-based for polars-bio's range operations.
- `pandas`: pandas and pyarrow, for anyone who prefers pandas DataFrames,
  so that any table the package returns can be converted with
  `.to_pandas()`, and for `to_logomaker()`.

## For development

The repository uses [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/):

```sh
cd universalmotif-py
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
```

Many of the tests compare the package against reference values that live
with the Rust crate, so they also need it checked out next door. To build this documentation, use the `docs` group
instead and run `uv run --no-sync mkdocs serve`.
