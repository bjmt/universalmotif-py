# Installation

The package needs Python 3.10 or later and is installed from PyPI:

```sh
pip install universalmotif
```

Wheels are built for Linux (x86_64 and aarch64), macOS (Intel and Apple
Silicon) and Windows (x86_64), so on those there is nothing to compile.
Installing also adds the `universalmotif` and `umotif` commands described in
[Command line](command-line.md).

## Optional dependencies

The only required dependencies are numpy and polars. Everything else is an
optional extra, needed only by the functions that use it, and installed by
naming it in brackets, such as `pip install "universalmotif[plot]"` or
`pip install "universalmotif[plot,scipy]"` for several at once:

- `plot`: matplotlib, for logos and the other plotting functions.
- `scipy`: SciPy, for the sparse matrices of
  [`match_motifs()`](reference/scanning.md#motif-match-matrices).
- `anndata`: AnnData and SciPy, for storing motif matches with single-cell
  ATAC data.
- `h5py`: h5py, for reading TF-MoDISco results with `read_modisco()` and
  chromBPNet contribution scores with `read_contributions()`.
- `bigwig`: pybigtools, for reading bigWig signal with `signal_stats()` and
  `signal_values()` and matching backgrounds on it with `match_bkg()`.
- `biopython`: Biopython, for converting motifs to and from `Bio.motifs`
  with `to_biopython()` and `from_biopython()`. The database functions need
  no extra, since they download with the standard library.
- `pyranges`: PyRanges, for `to_pyranges()`. PyRanges objects can be
  given as ranges without it.
- `polars-bio`: polars-bio, for `to_polars_bio()`, which marks tables as
  0-based for polars-bio's range operations.
- `pandas`: pandas and pyarrow, for anyone who prefers pandas DataFrames,
  so that any table the package returns can be converted with
  `.to_pandas()`, and for `to_logomaker()`.

## From source

On other platforms pip builds the package from its source distribution,
which needs a Rust toolchain, installed with [rustup](https://rustup.rs/).
The build takes a minute or two, as the Rust code is compiled with
optimizations.

To install the latest code from GitHub instead of a release, clone the
package and the Rust crate next to each other, since the package builds
against the crate in the neighbouring folder:

```sh
git clone https://github.com/bjmt/universalmotif-rs.git
git clone https://github.com/bjmt/universalmotif-py.git
pip install ./universalmotif-py
```

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
with the Rust crate, so they also need it checked out next door. To build
this documentation, use the `docs` group instead and run
`uv run --no-sync mkdocs serve`.
