# Genomic intervals

Functions for genomic ranges:

- reading and writing BED and peak files, and reading VCF files;
- resizing ranges and dropping overlapping ones;
- composition-matched backgrounds;
- motif proximity to genomic anchors;
- hits in genomic coordinates;
- ranges passed to and from PyRanges and polars-bio.

Positions count from 0 and leave the end out, as in BED files, PyRanges and
Python slices. Ranges can be given as polars DataFrames or tuples, and also as
PyRanges objects, polars-bio frames or pandas DataFrames, including those
with PyRanges' `Chromosome`, `Start` and `End` columns. polars-bio frames
that count from 1, as polars-bio makes them by default, are converted.

::: universalmotif.read_bed

::: universalmotif.write_bed

::: universalmotif.read_vcf

::: universalmotif.resize_ranges

::: universalmotif.drop_overlaps

::: universalmotif.match_bkg

::: universalmotif.motif_proximity

::: universalmotif.genomic_hits

::: universalmotif.to_pyranges

::: universalmotif.to_polars_bio
