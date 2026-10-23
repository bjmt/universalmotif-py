# universalmotif (Python)

A Python package for working with sequence motifs: creating them, converting
between their representations, reading and writing them in the common file
formats, and using them to scan, compare and test sequences. It is a port of
the [universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package, built on its Rust version,
[universalmotif-rs](https://github.com/bjmt/universalmotif-rs), which does
the heavy lifting, so most of it is quick and can use several threads.
Motifs are `Motif` objects, matrices come back as numpy arrays, and tables
come back as [polars](https://pola.rs/) DataFrames.

The [documentation](https://bjmt.github.io/universalmotif-py/) has guides covering everything from motif
basics to a full ChIP-seq analysis, and a reference for every function. The
package is not on PyPI yet, and its interface may still change before the
first release.

## Installation

Until there are prebuilt wheels, the package is built from source, which
needs Python 3.10 or later and a [Rust toolchain](https://rustup.rs/). The
Rust crate has to be checked out next to it:

```sh
git clone https://github.com/bjmt/universalmotif-rs.git
git clone https://github.com/bjmt/universalmotif-py.git
pip install "./universalmotif-py[plot]"
```

The only required dependencies are numpy and polars. Everything else is an
optional extra, needed only by the functions that use it:

| Extra | Brings in | For |
| --- | --- | --- |
| `plot` | matplotlib | logos, trees and the other plots |
| `scipy` | SciPy | sparse motif match matrices |
| `anndata` | AnnData, SciPy | match matrices stored with single-cell ATAC data |
| `biopython` | Biopython | converting to and from `Bio.motifs` |
| `h5py` | h5py | reading TF-MoDISco results |
| `bigwig` | pybigtools | bigWig signal, and backgrounds matched on it |
| `pyranges` | PyRanges | converting tables to PyRanges objects |
| `polars-bio` | polars-bio | preparing tables for polars-bio |
| `pandas` | pandas, pyarrow | pandas DataFrames, and logomaker export |

## A quick look

Using the example motifs and sequences that come with the package:

```python
import universalmotif as um

# A motif from a few aligned sites, and its log-odds matrix.
ap1 = um.create_motif(["TGACTCA", "TGAGTCA", "TGACTAA"], name="AP-1", pseudocount=1)
print(um.convert_type(ap1, "PWM").matrix)

# Scan sequences for the example motifs, and test them for enrichment
# against a set of control sequences.
motifs = um.example_motifs()
targets = um.example_sequences("targets")
controls = um.example_sequences("controls")
hits = um.scan_sequences(motifs, targets, pvalue=1e-4)
enriched = um.enrich_motifs(motifs, targets, controls)

# How similar are the motifs, and what do they look like?
similar = um.compare_motifs(motifs, compare_to="AP-1")
um.view_motifs(motifs[:4]).savefig("logos.png")

# Look for new motifs in the targets.
found = um.motif_finder(targets, controls, seed=1)
```

## What it can do

- **Motifs:** create them from sites, consensus strings or matrices, convert
  between counts, probabilities, log-odds, information content and
  contribution scores, and trim, round, merge, reverse complement and filter
  them.
- **Files:** MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE and plain
  matrices, and FASTA for sequences, gzipped or not.
- **Databases:** motifs fetched from JASPAR, HOCOMOCO and CIS-BP and kept in
  a local cache, and conversion to and from Biopython.
- **Scanning:** exact P-value thresholds, both strands, any alphabet and
  gapped motifs, with whole genomes scanned from a FASTA file in a few
  hundred MB.
- **Comparison:** similarity with P-values, scored either on the overlap or
  on the whole query as TOMTOM does, merging of similar motifs, and trees.
- **Statistics:** enrichment, co-occurrence, positional enrichment,
  proximity to genomic anchors, composition-matched backgrounds and *de
  novo* motif discovery.
- **Genomes:** FASTA, bgzip and 2bit genomes read a piece at a time, the
  sequences of peaks, bigWig signal over them, and hits moved back to
  genomic coordinates, with ranges exchanged with PyRanges and polars-bio.
- **Single-cell ATAC and deep learning:** sparse peak by motif matrices for
  AnnData and pychromVAR, motifs as convolution filters and back, one-hot
  sequences, and TF-MoDISco results.
- **Sequences:** shuffling that keeps k-let counts, random sequences, k-let
  backgrounds, complexity, masking and sliding windows.
- **Plots:** sequence logos, motif trees and plots of the results above,
  with motifs shown as logos in Jupyter and SVG logos for reports.

Scanning, comparison, merging, clustering, enrichment and logos work for
amino acid and custom alphabets as well as DNA and RNA.

## Command line

The package also installs a command line, as both `universalmotif` and
`umotif`, which runs the main functions on files for shell scripts and
workflow managers:

```sh
umotif fetch jaspar --tax-group plants --release 2024 -o jaspar.meme
umotif extract -g genome.fa.gz -r peaks.narrowPeak -o peaks.fa
umotif bkg -g genome.fa.gz -r peaks.narrowPeak --seed 1 -o bkg.fa
umotif enrich jaspar.meme -s peaks.fa -b bkg.fa -p 4 -o enriched.tsv
```

`umotif --help` lists all 19 commands.

## Differences from the R package

Function names follow the R package, with underscores in place of dots, so
`min.ic` becomes `min_ic`, and with Canadian spelling throughout, so that
`summarise_motifs()` is `summarize_motifs()`. Where the R package has
both a full and a `_lite` version of a function, this package provides the
lite one under the plain name. Readers always return a list of motifs, and
MEME site tables come back as polars DataFrames. Functions that use random
numbers take a `seed`, but seeds do not reproduce the R package's results,
since the two use different random number generators.

A number of bugs found in the R package while porting it are fixed here
rather than reproduced, and a few features were left out on purpose.
[DIVERGENCES.md](https://github.com/bjmt/universalmotif-rs/blob/main/DIVERGENCES.md)
in the Rust crate lists every difference.

## Documentation

The documentation lives at <https://bjmt.github.io/universalmotif-py/>. It is built with MkDocs from `docs/`,
with the API reference generated from the docstrings and guides written as
notebooks. To build it and preview it locally:

```sh
uv sync --group docs --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync mkdocs serve
```

## Development

Development uses [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/), with the Rust crate checked out next to
this one, since the package builds against it and the tests read the R
reference values kept there:

```sh
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
```

## Licence

MIT. The letter shapes used in logos are outlines of the Roboto font,
released under the Apache License 2.0, and were exported from the R
package's `fontDFroboto` with `scripts/make_font.R`.
