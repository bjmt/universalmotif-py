# Command line

Installing the package also installs a command line, called both
`universalmotif` and `umotif`, which runs the main functions of the package
on files. It suits shell scripts and workflow managers such as Snakemake and
Nextflow, where each step reads files and writes new ones. It can also be
run as `python -m universalmotif`.

```sh
umotif --help
umotif scan --help
```

## Commands

| Command | What it does |
| --- | --- |
| `convert` | Convert motifs between formats and types, or reverse complement them. |
| `fetch` | Download motifs from JASPAR, HOCOMOCO or CIS-BP, keeping a local cache. |
| `summary` | Summarize motifs as a table. |
| `filter` | Keep the motifs that pass every filter given. |
| `trim` | Trim low-information positions from motif edges. |
| `merge` | Merge similar motifs, or all of them into one. |
| `tree` | Cluster motifs into a tree, written as Newick, or cut it into clusters. |
| `logo` | Draw logos, in one aligned figure or one file per motif. |
| `scan` | Scan sequences, whole genomes or genomic ranges for motif hits. |
| `compare` | Compare motifs with each other or with a database. |
| `enrich` | Test motifs for enrichment in target sequences. |
| `find` | Discover motifs *de novo*. |
| `shuffle` | Shuffle sequences, keeping their k-mer counts. |
| `extract` | Write the sequences of genomic ranges as FASTA. |
| `bkg` | Draw background sequences matched to targets on GC content, length and bigWig signal. |
| `matrix` | Make a peak by motif matrix of matches for single-cell ATAC analyses. |
| `proximity` | Test whether motif hits sit closer to genomic anchors than chance. |
| `coocc` | Test pairs of motifs for co-occurrence. |
| `posenr` | Test motifs for positional enrichment within sequences. |

## Inputs and outputs

A few conventions hold across the commands.

- **Motifs** can be in any of the supported formats: MEME, JASPAR, HOMER,
  TRANSFAC, CIS-BP, UniPROBE or plain matrices, gzipped or not, and CIS-BP
  archives downloaded as `.zip` files. The format is recognized from the
  extension, or else from the contents, and can be given with `-f`. Several
  motif files can be given at once, and `-` reads from standard input.
  Motifs are written in the format given with `-t`, or the one matching the
  output file's extension, and MEME otherwise.
- **Sequences** are FASTA files, gzipped or not, given with `-s`.
- **Genomes** are given with `-g`, as plain FASTA, FASTA compressed with
  `bgzip` or 2bit files, and only the parts needed are read. Ordinary gzip
  compression cannot be read in pieces, so a gzipped genome must first be
  recompressed with `bgzip`, as described in [Scanning large
  genomes](guides/large-genomes.ipynb#leaving-the-genome-on-disk).
- **Ranges**, such as peaks, are given with `-r` as BED files, including
  narrowPeak and broadPeak files, or as tab-separated tables with a header
  naming `chrom`, `start` and `end` or `width` columns. Positions count from
  0 and leave the end out, as in BED files. A strand column, or the sixth
  column of a BED file, marks ranges on the minus strand, which are reverse
  complemented.
- **Tables** are written as tab-separated text with a header row, with
  missing values written as `NA` and lists joined by commas. Positions count
  from 0 and leave the end out, so the first three columns of a table of
  hits on a genome can be read as BED.
- **Output** goes to standard output unless a file is given with `-o`, so
  commands can be chained with pipes. Files ending in `.gz` are written
  gzipped.
- **Threads** are set with `-p` and default to 1, and `--seed` makes the
  commands that draw random numbers reproducible.

Warnings, such as for motifs that cannot reach a P-value cutoff, are
printed to standard error, and errors end the command with a short message
and a non-zero exit status. `--debug`, given before the command, shows the
full traceback instead.

## An example

Let us go from a set of ChIP-seq peaks to the motifs enriched in them, in
much the same way as the [ChIP-seq workflow](guides/chipseq-workflow.ipynb)
guide does in Python. Say the peaks are in `peaks.narrowPeak` and the genome
in `genome.fa.gz`, compressed with `bgzip`.

First, the plant motifs of JASPAR 2024, and the sequences of the peaks:

```sh
umotif fetch jaspar --tax-group plants --release 2024 -o jaspar.meme
umotif extract -g genome.fa.gz -r peaks.narrowPeak -o peaks.fa
```

Then background sequences matched to the peaks on GC content and length,
drawn from windows of the genome that keep clear of the peaks. If there is
a bigWig file of signal, such as ATAC-seq coverage, adding
`--bigwig atac.bw` matches on it as well.

```sh
umotif bkg -g genome.fa.gz -r peaks.narrowPeak --seed 1 -o bkg.fa
```

Now the enrichment test, which lists the enriched motifs in order of
significance. Broad peaks can have hundreds of enriched motifs, so here we
keep the twenty most significant, and then find their hits on the genome and
draw their logos:

```sh
umotif enrich jaspar.meme -s peaks.fa -b bkg.fa -p 4 -o enriched.tsv
cut -f1 enriched.tsv | sed -n '2,21p' > top.txt
umotif filter jaspar.meme $(sed 's/^/--name /' top.txt) -o enriched.meme
umotif scan enriched.meme -g genome.fa.gz -r peaks.narrowPeak -p 4 -o hits.tsv
umotif logo enriched.meme -o enriched.pdf
```

The hits come back in genomic coordinates, ready to be loaded alongside the
peaks in a genome browser or a table. Finally, the motifs found can be
checked against the whole of JASPAR, and *de novo* motifs looked for in the
peaks:

```sh
umotif compare enriched.meme --db jaspar.meme --complete -o similar.tsv
umotif find -s peaks.fa -b bkg.fa --seed 1 --table found.tsv -o found.meme
```

## Reference

The help of every command follows, as printed by `umotif COMMAND --help`.

<!-- command help -->
