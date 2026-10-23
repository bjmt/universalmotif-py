# Genomic intervals

Composition-matched backgrounds, motif proximity to genomic anchors, hits in genomic coordinates, and ranges passed to and from PyRanges and polars-bio. Positions count from 0 and leave the end out, as in BED files, PyRanges and Python slices, and ranges can be given as DataFrames or tuples. PyRanges objects, pandas DataFrames, including those with PyRanges' `Chromosome`, `Start` and `End` columns, and polars-bio frames can be given wherever ranges are taken; polars-bio frames marked as counting from 1, as polars-bio makes them by default, are converted.

::: universalmotif.match_bkg

::: universalmotif.motif_proximity

::: universalmotif.genomic_hits

::: universalmotif.to_pyranges

::: universalmotif.to_polars_bio
