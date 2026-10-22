# Installation

The package is not on PyPI yet. Until it is, it has to be built from source,
which needs a Rust toolchain as well as Python 3.10 or later. (Once it is on
PyPI, `pip install universalmotif` will fetch a prebuilt wheel and none of
this will be needed.)

## From source

1. Install Rust with [rustup](https://rustup.rs/), if you don't have it
   already.
2. Clone the Python package and the Rust crate next to each other, since
   the package builds against the crate in the neighbouring folder:

    ```sh
    git clone git@github.com:bjmt/universalmotif-rs.git
    git clone git@github.com:bjmt/universalmotif-py.git
    ```

3. Install the package with pip (the plotting functions also need
   matplotlib, which the `plot` extra brings in):

    ```sh
    pip install "./universalmotif-py[plot]"
    ```

The build takes a minute or two, as the Rust code is compiled with
optimisations.

## Optional dependencies

- `plot`: matplotlib, for logos and the other plotting functions.
- `pandas`: pandas and pyarrow, for anyone who prefers pandas DataFrames
  (call `.to_pandas()` on any table the package returns).

## For development

The repository uses [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/):

```sh
cd universalmotif-py
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
```

Many of the tests compare the package against reference values exported
from the R package, which live with the Rust crate (and so also need it
checked out next door). To build this documentation, use the `docs` group
instead and run `uv run --no-sync mkdocs serve`.
