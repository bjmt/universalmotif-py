# Genomes and signal

Genomes read from disk a piece at a time, the sequences of genomic ranges,
and bigWig signal over them and around motif sites. Positions count from 0 and leave the end out.
Ranges can be given as DataFrames or tuples, as for
[genomic intervals](genomic.md).

::: universalmotif.Genome
    options:
      members: [lengths, subseq, extract]

::: universalmotif.extract_ranges

::: universalmotif.signal_stats

::: universalmotif.signal_values

::: universalmotif.signal_matrix

::: universalmotif.SignalMatrix
    options:
      members: [positions, clip, profile]
