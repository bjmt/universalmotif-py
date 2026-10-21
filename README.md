# pyumotif

pyumotif is a Python package for working with sequence motifs. It is a port
of the [universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package, built on the
[umotif](../motifkit) Rust library, and it uses numpy arrays for motif
matrices and polars DataFrames for tables.

The package is at an early stage. Creating motifs, converting between motif
types, reverse complementing, trimming and summarising are available; reading
and writing motif files, P-values, scanning, comparison, shuffling and motif
discovery will follow.

## Example

```python
import pyumotif as um

motif = um.create_motif("TATAWAW", nsites=50)
pwm = um.convert_type(motif, "PWM", pseudocount=1)
print(pwm)
print(pwm.matrix)

sites = um.create_motif(["AAGT", "ACGT", "ACCT", "TCGT"], type="PCM")
print(um.summarise_motifs([motif, sites]))
```

Function names follow universalmotif, with dotted R argument names written
with underscores (`min.ic` becomes `min_ic`). Where universalmotif has both a
full and a `_lite` version of a function, pyumotif will provide the lite
version under the plain name.

## Development

pyumotif needs a Rust toolchain and [uv](https://docs.astral.sh/uv/). It
expects a checkout of umotif next to it, currently at `../motifkit`.

```sh
uv sync
uv run maturin develop
uv run pytest
```

## License

MIT
