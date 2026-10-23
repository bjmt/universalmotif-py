# universalmotif (Python)

This is the Python version of universalmotif, a package for working with
sequence motifs. It is a port of the
[universalmotif](https://bioconductor.org/packages/universalmotif/)
R/Bioconductor package, built on the Rust version,
[universalmotif-rs](../universalmotif-rs), and it uses numpy arrays for motif
matrices and polars DataFrames for tables.

The package is at an early stage. Creating motifs, converting between motif
types, reverse complementing, trimming and summarising are available, as are
reading and writing MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE, plain
matrix and FASTA files, exact P-values, scanning sequences for
hits, comparing, merging and clustering motifs, k-let backgrounds,
sequence complexity, shuffling, random sequences, implanting sites and
*de novo* motif discovery, enrichment testing, co-occurrence, positional
enrichment, proximity to genomic anchors and composition-matched
backgrounds, along with sequence logos, motif trees and motif
peak plots drawn with matplotlib, which is installed with `pip install
universalmotif[plot]`. Scanning, comparison, merging, clustering,
enrichment and logos work for amino acid and custom alphabets as well as
DNA and RNA, and scanning handles gapped motifs. For genome-sized inputs, `scan_sequences` can scan a gzipped or
plain FASTA file given as a path a batch at a time, and `read_fasta(..., packed=True)` keeps
sequences packed at two bits per base for repeated use. Motifs can be
fetched from JASPAR, HOCOMOCO and CIS-BP with `universalmotif.databases`,
which caches its downloads, and converted to and from Biopython's
`Bio.motifs` with `pip install universalmotif[biopython]`. In Jupyter,
motifs are shown as a logo next to their metadata, `logo_svg()` draws the
same logo as an SVG image, and `to_logomaker()` exports letter heights for
logomaker. For single-cell ATAC data, `match_motifs()` gives a sparse peak by motif
matrix of matches, counts or best scores without keeping every hit, and
`add_motif_matches()` stores it in an AnnData object as pychromVAR expects. For deep-learning models, `to_filters()` and `from_filters()` turn motifs
into convolution filters and back, also taking the motif dictionaries of
tangermeme and memelite, `one_hot()` and `from_one_hot()` encode sequences,
and `read_modisco()` reads TF-MoDISco results as CWMs or PPMs, with
`pip install universalmotif[h5py]`.

## Example

```python
from pathlib import Path

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

background = um.shuffle_sequences(sequences, k=2, seed=1)
found = um.motif_finder(sequences, background, qvalue=0.05)
new_motifs = found["motif"].to_list()

enriched = um.enrich_motifs(motifs, sequences, background, qvalue=0.05)
pairs = um.motif_coocc(motifs, sequences, max_distance=50)
peaks = um.motif_peaks(hits, seq_length=500)  # equal-length, centred sequences
groups = tree.cutree(k=5)

um.view_motifs(motifs[:4]).savefig("logos.png")  # aligned logos
tree.plot(groups="family", tip_logos=True)        # matplotlib axes
um.plot_motif_peaks(peaks)

genome_hits = um.scan_sequences(motifs, Path("genome.fa"))  # streamed
genome = um.read_fasta("genome.fa", packed=True)            # about 0.8 GB
background = um.match_bkg(sequences, genome=genome, exclude=peak_table)
near = um.motif_proximity(anchors=summits, motifs=motifs, sequences=genome)
```

Functions that use random numbers take a `seed`; without one they draw a
fresh seed each time. Seeds do not reproduce the R package's results, since
the two use different random number generators.

Function names follow the R package, with dotted R argument names written
with underscores (`min.ic` becomes `min_ic`). Where the R package has both a
full and a `_lite` version of a function, this package will provide the lite
version under the plain name. Readers always return a list of motifs, and
MEME site tables come back as polars DataFrames.

## Comparison with TOMTOM

`compare_motifs()` scores an alignment by the sum of the Pearson
correlations of its columns and takes P-values from the distribution of
that sum under a null of database columns, much as TOMTOM does with its
Pearson distance. With `complete=True` it also scores whole queries as
TOMTOM does by default: each query column is scored relative to the median
of its correlations with the null columns, so that overhanging columns
count as typical ones, and the P-value is that of the best of all the
alignments tried. `scripts/compare_tomtom.py` runs both on the same motifs.

With 72 motifs, 60 from CIS-BP and the 12 example motifs, against the 805
plant motifs of JASPAR 2024:

| | `complete=True`, TOMTOM default | default, TOMTOM `-incomplete-scores` |
| --- | --- | --- |
| log10 P-values, correlation over all 57,960 pairs | 0.999 | 0.988 |
| P-value rank correlation within queries, median | 0.999 | 0.927 |
| Each query's five best matches shared | 95% | 88% |
| Same best match | 94% | 78% |
| Strong matches on the same strand | 99% | 99% |

Where the best match differs, the two choices are near ties, a tenth to a
fifth of an order of magnitude apart.

## Development

Building needs a Rust toolchain and [uv](https://docs.astral.sh/uv/), and a
checkout of [universalmotif-rs](https://github.com/bjmt/universalmotif-rs)
next to this one, since the package builds against it and the tests read
the R reference values kept there.

```sh
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
```

## Documentation

The documentation is built with MkDocs from `docs/`, with the API reference
generated from the docstrings and the guides written as notebooks that are
run on every build. To build it and preview it locally:

```sh
uv sync --group docs --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync mkdocs serve
```

## License

MIT. The letter shapes used in logos are outlines of the Roboto font,
released under the Apache License 2.0, and were exported from the R
package's `fontDFroboto` with `scripts/make_font.R`.
