# Genomes and signal

Genomes read from disk a piece at a time, the sequences of genomic ranges, and bigWig signal over them. Positions count from 0 and leave the end out, and ranges can be given as DataFrames or tuples, as for [genomic intervals](genomic.md).

::: universalmotif.Genome
    options:
      members: [lengths, subseq, extract]

::: universalmotif.extract_ranges

::: universalmotif.signal_stats
