# Genomes and signal

Genomes read from disk a piece at a time, the sequences of genomic ranges, and bigWig signal over them. Ranges are 1-based and include both ends, and can be given as DataFrames or tuples, as for [genomic intervals](genomic.md).

::: universalmotif.Genome
    options:
      members: [lengths, subseq, extract]

::: universalmotif.extract_ranges

::: universalmotif.signal_stats
