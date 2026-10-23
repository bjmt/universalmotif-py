# Genomic intervals

Composition-matched backgrounds, motif proximity to genomic anchors, hits in genomic coordinates, and ranges passed to and from PyRanges and polars-bio. Ranges are 1-based and include both ends, and can be given as DataFrames or tuples. PyRanges objects, pandas DataFrames with PyRanges' `Chromosome`, `Start` and `End` columns, and polars-bio frames marked as 0-based are taken to be 0-based and end-exclusive and are converted, so they can be given wherever ranges are taken.

::: universalmotif.match_bkg

::: universalmotif.motif_proximity

::: universalmotif.genomic_hits

::: universalmotif.to_pyranges

::: universalmotif.to_polars_bio
