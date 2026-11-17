# universalmotif (Python)

[![PyPI](https://img.shields.io/pypi/v/universalmotif)](https://pypi.org/project/universalmotif/) [![Downloads](https://img.shields.io/pypi/dm/universalmotif)](https://pypistats.org/packages/universalmotif) [![DOI](https://joss.theoj.org/papers/10.21105/joss.07012/status.svg)](https://doi.org/10.21105/joss.07012)

A Python package for creating sequence motifs, converting between their
representations, and reading and writing them in the common file formats.
It also uses motifs to scan, compare and test sequences. It is a port of the
[universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package. The heavy lifting is done by its Rust version,
[universalmotif-rs](https://github.com/bjmt/universalmotif-rs), so most of
it is quick and can use several threads. Motifs are `Motif` objects,
matrices come back as numpy arrays, and tables as
[polars](https://pola.rs/) DataFrames.

The [documentation](https://bjmt.github.io/universalmotif-py/) has guides
from motif basics to a full ChIP-seq analysis, and a reference for every
function. The package is at an early version, and its interface may still
change.

## Installation

The package needs Python 3.10 or later and can be installed from PyPI:

```sh
pip install universalmotif
```

Wheels are built for Linux (x86_64 and aarch64), macOS (Intel and Apple
Silicon) and Windows (x86_64). Elsewhere pip builds the package from source,
which needs a [Rust toolchain](https://rustup.rs/).

The only required dependencies are numpy and polars. Everything else is an
optional extra, needed only by the functions that use it. For example,
`pip install "universalmotif[plot]"` adds matplotlib for drawing logos.

| Extra | Brings in | For |
| --- | --- | --- |
| `plot` | matplotlib | logos, trees and the other plots |
| `scipy` | SciPy | sparse motif match matrices |
| `anndata` | AnnData, SciPy | match matrices stored with single-cell ATAC data |
| `biopython` | Biopython | converting to and from `Bio.motifs` |
| `h5py` | h5py, hdf5plugin | reading TF-MoDISco results and contribution scores from chromBPNet and ENCODE |
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

- **Motifs:** create them from sites, consensus strings or matrices.
  Convert between counts, probabilities, log-odds, information content and
  contribution scores. Trim, round, merge, reverse complement and filter
  them.
- **Files:** MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE and plain
  matrices, and FASTA for sequences, gzipped or not.
- **Databases:** motifs fetched from JASPAR, HOCOMOCO and CIS-BP and kept in
  a local cache, and conversion to and from Biopython.
- **Scanning:** exact P-value thresholds, both strands, any alphabet and
  gapped motifs. Whole genomes are scanned from a FASTA file in a few
  hundred MB, and variants from VCF files, insertions and deletions
  included, are scored by how they change motif sites.
- **Comparison:** similarity with P-values, scored on the overlap or on the
  whole query, merging of similar motifs, and trees.
- **Statistics:** enrichment, co-occurrence, positional enrichment,
  proximity to genomic anchors and composition-matched backgrounds.
  *De novo* motif discovery, of gapped motifs too, with refinement and
  selection of the motifs found.
- **Genomes:** FASTA, bgzip and 2bit genomes read a piece at a time, the
  sequences of peaks, and bigWig signal over them. Hits can be moved back
  to genomic coordinates, and ranges exchanged with PyRanges and polars-bio.
- **Single-cell ATAC and deep learning:**
    - sparse peak by motif matrices for AnnData and pychromVAR
    - motifs as convolution filters and back, and one-hot sequences
    - TF-MoDISco results, and seqlets and motifs found directly in a
      model's contribution scores
    - plots of attribution scores along sequences
    - batches for in silico mutagenesis and motif insertion experiments
- **Sequences:** shuffling that keeps k-mer counts, random sequences, k-mer
  backgrounds, complexity, masking and sliding windows.
- **Plots:** sequence logos, motif trees and plots of the results above.
  Motifs are shown as logos in Jupyter, and SVG logos can go in reports.

Scanning, comparison, merging, clustering, enrichment and logos work for
amino acid and custom alphabets as well as DNA and RNA.

## Command line

The package also installs a command line, as both `universalmotif` and
`umotif`. It runs the main functions on files, for use in shell scripts and
workflow managers:

```sh
umotif fetch jaspar --tax-group plants --release 2024 -o jaspar.meme
umotif extract -g genome.fa.gz -r peaks.narrowPeak -o peaks.fa
umotif bkg -g genome.fa.gz -r peaks.narrowPeak --seed 1 -o bkg.fa
umotif enrich jaspar.meme -s peaks.fa -b bkg.fa -p 4 -o enriched.tsv
```

`umotif --help` lists all 29 commands, in groups.

## Differences from the R package

Function names follow the R package, with underscores in place of dots, so
`min.ic` becomes `min_ic`. Where the R package has both a full and a `_lite`
version of a function, this package provides the lite one under the plain
name.

Readers return a list of motifs. The exception is `read_meme()` with
`readsites` or `readsites_meta`, which returns a dict of the motifs and
their site tables as polars DataFrames. Functions that use random numbers
take a `seed`, but seeds do not reproduce the R package's results because
the two use different random number generators.

Bugs found in the R package while porting it are fixed here rather than
reproduced.
[DIVERGENCES.md](https://github.com/bjmt/universalmotif-rs/blob/main/DIVERGENCES.md)
in the Rust crate lists the other differences, including the few features
left out on purpose.

## Documentation

The documentation lives at <https://bjmt.github.io/universalmotif-py/>.
It is built with MkDocs from `docs/`. The API reference is generated from
the docstrings, and the guides are written as notebooks. To build and
preview it locally:

```sh
uv sync --group docs --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync mkdocs serve
```

## Development

Development uses [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/). The Rust crate must be checked out next
to this one, because the package builds against it and the tests read the
R reference values kept there:

```sh
git clone https://github.com/bjmt/universalmotif-rs.git
git clone https://github.com/bjmt/universalmotif-py.git
cd universalmotif-py
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
```

## Citation

If this package has been useful for your research, please cite the article
describing universalmotif:

Tremblay BJM (2024). universalmotif: An R package for biological motif
analysis. *Journal of Open Source Software* **9**, 7012.
DOI: [10.21105/joss.07012](https://doi.org/10.21105/joss.07012)

## About this port

This package, and the Rust crate it is built on, are AI-assisted rewrites
of the R/Bioconductor package universalmotif. They were made by Benjamin
Jean-Marie Tremblay, the author of the R package. They are meant to be
faithful to the original, and the test suites check their results against
reference values computed with the R package.

The rewrite was also a chance to make improvements that the R package has
held back on, since changes there are weighed against breaking the
behaviour its existing users rely on.
[DIVERGENCES.md](https://github.com/bjmt/universalmotif-rs/blob/main/DIVERGENCES.md)
lists the places where the ports deliberately differ, including these
improvements, the bugs fixed along the way and the features left out. The
rewrite follows the principles set out at
[rewrites.bio](https://rewrites.bio).

## Licence

MIT. The letter shapes used in logos are outlines of the Roboto font,
released under the Apache License 2.0. They were exported from the R
package's `fontDFroboto` with `scripts/make_font.R`.
