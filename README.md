# universalmotif (Python)

This is the Python version of universalmotif, a package for working with
sequence motifs. It is a port of the
[universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package, built on the Rust version
([universalmotif-rs](../universalmotif-rs)), and it uses numpy arrays for motif
matrices and polars DataFrames for tables.

The package is at an early stage. Creating motifs, converting between motif
types, reverse complementing, trimming and summarising are available, as are
reading and writing motif files (MEME, JASPAR, HOMER, TRANSFAC, CIS-BP,
UniPROBE, plain matrices and FASTA), exact P-values, scanning sequences for
hits, and comparing, merging and clustering motifs. Shuffling, enrichment,
motif discovery and plotting will follow.

## Example

```python
import universalmotif as um

motif = um.create_motif("TATAWAW", nsites=50)
pwm = um.convert_type(motif, "PWM", pseudocount=1)
print(pwm)
print(pwm.matrix)

sites = um.create_motif(["AAGT", "ACGT", "ACCT", "TCGT"], type="PCM")
print(um.summarise_motifs([motif, sites]))

um.write_meme([motif, sites], "motifs.meme")
motifs = um.read_meme("motifs.meme")

sequences = um.read_fasta("promoters.fa")
hits = um.scan_sequences(motifs, sequences, pvalue=1e-4)

similarity = um.compare_motifs(motifs)        # all-against-all matrix
tree = um.motif_tree(motifs)                  # tree.newick, tree.linkage
merged = um.merge_similar(motifs, qvalue=0.01)
```

Function names follow the R package, with dotted R argument names written
with underscores (`min.ic` becomes `min_ic`). Where the R package has both a
full and a `_lite` version of a function, this package will provide the lite
version under the plain name. Readers always return a list of motifs, and
MEME site tables come back as polars DataFrames.

## Development

Building needs a Rust toolchain and [uv](https://docs.astral.sh/uv/), and a
checkout of [universalmotif-rs](../universalmotif-rs) next to this one.

```sh
uv sync
uv run maturin develop
uv run pytest
```

## License

MIT
