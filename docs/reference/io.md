# Reading and writing

Readers and writers for motif files, and for FASTA files of sequences.
Readers return a list of motifs, even when the file holds just one. The
exception is [`read_meme()`][universalmotif.read_meme] with `readsites` or
`readsites_meta`, which returns a dict of the motifs and their sites. Gzipped
FASTA files, bgzip included, are read as they are.

::: universalmotif.read_meme

::: universalmotif.read_jaspar

::: universalmotif.read_homer

::: universalmotif.read_transfac

::: universalmotif.read_cisbp

::: universalmotif.read_uniprobe

::: universalmotif.read_matrix

::: universalmotif.read_universalmotif

::: universalmotif.write_meme

::: universalmotif.write_jaspar

::: universalmotif.write_homer

::: universalmotif.write_transfac

::: universalmotif.write_cisbp

::: universalmotif.write_uniprobe

::: universalmotif.write_matrix

::: universalmotif.write_universalmotif

::: universalmotif.meme_alph

::: universalmotif.read_fasta

::: universalmotif.write_fasta

::: universalmotif.PackedSequences

::: universalmotif.pack_sequences
