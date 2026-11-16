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

| Command | What it does | Python function |
| --- | --- | --- |
| `convert` | Convert motifs between formats and types, or reverse complement them. | [`convert_type()`][universalmotif.convert_type], [`motif_rc()`][universalmotif.motif_rc], [`switch_alph()`][universalmotif.switch_alph] |
| `fetch` | Download motifs from JASPAR, HOCOMOCO or CIS-BP, keeping a local cache. | [`databases.jaspar()`][universalmotif.databases.jaspar], [`databases.hocomoco()`][universalmotif.databases.hocomoco], [`databases.cisbp()`][universalmotif.databases.cisbp] |
| `create` | Make motifs from consensus sequences, or from aligned sites in a FASTA file, a list or the hits of `umotif scan`. | [`create_motif()`][universalmotif.create_motif] |
| `summary` | Summarize motifs as a table. | [`summarize_motifs()`][universalmotif.summarize_motifs] |
| `filter` | Keep the motifs that pass every filter given. | [`filter_motifs()`][universalmotif.filter_motifs] |
| `trim` | Trim low-information positions from motif edges. | [`trim_motifs()`][universalmotif.trim_motifs] |
| `merge` | Merge similar motifs, or all of them into one. | [`merge_similar()`][universalmotif.merge_similar], [`merge_motifs()`][universalmotif.merge_motifs] |
| `tree` | Cluster motifs into a tree, written as Newick, or cut it into clusters. | [`motif_tree()`][universalmotif.motif_tree] |
| `logo` | Draw logos, in one aligned figure or one file per motif. | [`view_motifs()`][universalmotif.view_motifs], [`logo_svg()`][universalmotif.logo_svg] |
| `heatmap` | Draw motif similarities as a clustered heatmap, optionally with the tree cut at a score. | [`plot_comparison()`][universalmotif.plot_comparison] |
| `scan` | Scan sequences, whole genomes or genomic ranges for motif hits. | [`scan_sequences()`][universalmotif.scan_sequences] |
| `variants` | Score how the variants of a VCF file change motif sites in a genome. | [`variant_motif_effects()`][universalmotif.variant_motif_effects] |
| `compare` | Compare motifs with each other or with a database. | [`compare_motifs()`][universalmotif.compare_motifs] |
| `enrich` | Test motifs for enrichment in target sequences. | [`enrich_motifs()`][universalmotif.enrich_motifs] |
| `find` | Discover motifs *de novo*. | [`motif_finder()`][universalmotif.motif_finder] |
| `find-gapped` | Discover gapped motifs *de novo*: two parts with a spacer of varying length (experimental). | [`gapped_motif_finder()`][universalmotif.gapped_motif_finder] |
| `shuffle` | Shuffle sequences, keeping their k-mer counts. | [`shuffle_sequences()`][universalmotif.shuffle_sequences] |
| `extract` | Write the sequences of genomic ranges as FASTA. | [`extract_ranges()`][universalmotif.extract_ranges] |
| `bkg` | Draw background sequences matched to targets on GC content, length and bigWig signal. | [`match_bkg()`][universalmotif.match_bkg] |
| `matrix` | Make a peak by motif matrix of matches for single-cell ATAC analyses. | [`match_motifs()`][universalmotif.match_motifs] |
| `proximity` | Test whether motif hits sit closer to genomic anchors than chance. | [`motif_proximity()`][universalmotif.motif_proximity] |
| `coocc` | Test pairs of motifs for co-occurrence. | [`motif_coocc()`][universalmotif.motif_coocc] |
| `posenr` | Test motifs for positional enrichment within sequences. | [`motif_peaks()`][universalmotif.motif_peaks] |

## Inputs and outputs

A few conventions hold across the commands.

- **Motifs** can be MEME, JASPAR, HOMER, TRANSFAC, CIS-BP, UniPROBE or
  plain matrix files, gzipped or not. CIS-BP archives downloaded as `.zip`
  files are read too. The format is recognized
  from the extension, or else from the contents, and can be given with `-f`.
  When a file cannot be read, the error names the format it was taken to be.
  Several motif files can be given at once, and `-` reads from standard
  input. Motifs are written in the format given with `-t`, or the one
  matching the output file's extension, and MEME otherwise.
- **Sequences** are FASTA files, gzipped or not, given with `-s`.
- **Genomes** are given with `-g`, as plain FASTA, FASTA compressed with
  `bgzip` or 2bit files. Only the parts needed are read. Ordinary gzip
  compression cannot be read in pieces, so a gzipped genome must first be
  recompressed with `bgzip`, as described in [Scanning large
  genomes](guides/large-genomes.ipynb#leaving-the-genome-on-disk).
- **Ranges**, such as peaks, are given with `-r`. They can be BED files,
  including narrowPeak and broadPeak, or tab-separated tables with a header
  naming `chrom`, `start` and `end` or `width` columns. Positions count from
  0 and leave the end out, as in BED files. A strand column, or the sixth
  column of a BED file, marks ranges on the minus strand, which are reverse
  complemented.
- **Tables** are written as tab-separated text with a header row. Missing
  values are written as `NA` and lists are joined by commas. Positions count
  from 0 and leave the end out, as in BED files, and `scan --bed` writes
  hits as BED directly.
- **Output** goes to standard output unless a file is given with `-o`, so
  commands can be chained with pipes. Files ending in `.gz` are written
  gzipped.
- **Threads** are set with `-p`. They default to 1, or to the
  `UNIVERSALMOTIF_NTHREADS` environment variable if it is set.
- **Random numbers** drawn by a command are made reproducible with
  `--seed`.
- **Progress** of `scan`, `enrich`, `find`, `find-gapped` and `shuffle` is
  shown as a bar on standard error once they have run for a second, if
  standard error is a terminal. `--no-progress` turns it off, and
  `--progress` turns it on when writing to a log. Ctrl-C stops the command.

Warnings, such as for motifs that cannot reach a P-value cutoff, are
printed to standard error. Errors end the command with a short message and
a non-zero exit status, and `--debug`, given before the command, shows the
full traceback instead.

## An example

Let us go from a set of ChIP-seq peaks to the motifs enriched in them, much
as the [ChIP-seq workflow](guides/chipseq-workflow.ipynb) guide does in
Python. Say the peaks are in `peaks.narrowPeak` and the genome in
`genome.fa.gz`, compressed with `bgzip`.

First, the plant motifs of JASPAR 2024, and the sequences of the peaks:

```sh
umotif fetch jaspar --tax-group plants --release 2024 -o jaspar.meme
umotif extract -g genome.fa.gz -r peaks.narrowPeak -o peaks.fa
```

Then background sequences matched to the peaks on GC content and length,
drawn from windows of the genome that keep clear of the peaks. With a bigWig
file of signal, such as ATAC-seq coverage, adding `--bigwig atac.bw`
matches on it as well.

```sh
umotif bkg -g genome.fa.gz -r peaks.narrowPeak --seed 1 -o bkg.fa
```

Now the enrichment test, which lists every motif in order of significance.
With `-q` it lists only those passing a q-value cutoff. Broad peaks can have
hundreds of enriched motifs, so here we keep the twenty most significant.
Then we find their hits on the genome and draw their logos:

```sh
umotif enrich jaspar.meme -s peaks.fa -b bkg.fa -p 4 -o enriched.tsv
cut -f1 enriched.tsv | sed -n '2,21p' > top.txt
umotif filter jaspar.meme $(sed 's/^/--name /' top.txt) -o enriched.meme
umotif scan enriched.meme -g genome.fa.gz -r peaks.narrowPeak -p 4 -o hits.tsv
umotif logo enriched.meme -o enriched.pdf
```

The hits come back in genomic coordinates, ready to be loaded alongside the
peaks in a genome browser or a table. Finally, we check the motifs found
against the whole of JASPAR, and look for *de novo* motifs in the peaks:

```sh
umotif compare enriched.meme --db jaspar.meme --complete -o similar.tsv
umotif find -s peaks.fa -b bkg.fa --seed 1 --table found.tsv -o found.meme
```

## More examples

The recipes below show the other commands on their own. The file names are
placeholders. `motifs.meme` stands for any motif file, `peaks.bed` for a
BED file of ranges and `genome.fa.gz` for a genome compressed with `bgzip`.

Where an output is shown, it comes from running the commands on small
made-up files when this page is built. These are the package's example
motifs and sequences, and a 100 kb random genome. The genome has AP-1 sites
planted in its peaks, and G-box and CRE sites together in half of them.

### Converting and inspecting motifs

Formats are recognized when reading and chosen from the output file's
extension when writing. Converting a file therefore only takes naming the
output, and several files can be combined into one:

```sh
umotif convert motifs.jaspar -o motifs.meme
umotif convert motifs.meme other.transfac -o combined.meme
```

`--type` changes what the matrices hold, here counts for a JASPAR file, and
`--rc` reverse complements every motif. `summary` lists the motifs with
their consensus, information content and number of sites:

```sh
umotif convert motifs.meme --type PCM -o counts.jaspar
umotif convert motifs.meme --rc -o motifs_rc.meme
umotif summary motifs.meme
```

<!-- output -->

### Curating a collection

Databases hold many closely related motifs. Take the human bZIP motifs of
HOCOMOCO as an example:

- `filter` keeps those of the FOS, JUN, ATF and CREB factors with enough
  information content to be useful.
- `trim` cuts their uninformative edges.
- `heatmap` shows how they group, with the tree cut at a mean column
  correlation of 0.8.

```sh
umotif fetch hocomoco --species HUMAN -o hocomoco.meme
umotif filter hocomoco.meme --regex '^(FOS|JUN|ATF|CREB)' --icscore 8 \
    | umotif trim - -o bzip.meme
umotif heatmap bzip.meme --cut 0.8 -o bzip.pdf
```

`merge` then combines similar motifs, and `--clusters` records which were
merged. Complete linkage requires every pair in a cluster to be similar.
`logo --each` draws every result to a file of its own:

```sh
umotif merge bzip.meme --linkage complete --clusters clusters.tsv -o bzip_merged.meme
umotif logo bzip_merged.meme --each logos
```

### Making motifs

`create` makes motifs from consensus sequences, which may use IUPAC codes,
or from a file of aligned sites. A motif made from a consensus stands for
100 sites, which `--nsites` changes. It gives other letters no weight unless
`--pseudocount` gives them some:

```sh
umotif create TGASTCA CACGTG --name AP-1 --name G-box --pseudocount 1 -o seeds.meme
umotif create --sites sites.fa -o from_sites.meme
```

<!-- output: seeds.meme 16 -->

It also reads the table of hits that `scan` writes, making one motif for
each motif scanned from the sites it matched. This way a rough consensus
motif can be refined from its matches in a set of sequences. The pseudocount
above lets near matches score. Short motifs cannot reach very small
P-values, so the cutoff is loosened. The refined motifs come from the sites
found, and show how much they vary:

```sh
umotif scan seeds.meme -s peaks.fa -P 1e-3 | umotif create --sites - -o refined.meme
umotif summary seeds.meme refined.meme
```

<!-- output -->

### Comparing and clustering

`compare` with `--db` finds the motifs of a database most similar to each
query, such as the motifs found by `find`. `--complete` scores the whole of
each query rather than only where the two motifs overlap. Here only matches
with q-values below 0.05 are kept. Without `--db`, the motifs are compared
with each other, and `--matrix` writes every score at once:

```sh
umotif compare found.meme --db jaspar.meme --complete -q 0.05 -o matches.tsv
```

```sh
umotif compare motifs.meme --matrix score -o scores.tsv
```

<!-- output: scores.tsv 5 7 -->

`tree` clusters motifs and writes the tree in Newick format. `--plot` draws
it with logos at its tips, and `-k` writes the cluster of each motif
instead:

```sh
umotif tree motifs.meme --plot tree.png --logos -o tree.nwk
umotif tree motifs.meme -k 5 -o clusters.tsv
```

<!-- image: tree.png -->

<!-- output: clusters.tsv 13 -->

### Scanning

The sequences of a FASTA file are read and scanned a batch at a time.
`--first-hit` keeps the leftmost hit of each motif in each sequence, which
is quicker when only the presence of a motif matters. `--no-overlaps` keeps
the best of overlapping hits, comparing those on opposite strands too:

```sh
umotif scan motifs.meme -s seqs.fa -o hits.tsv
umotif scan motifs.meme -s seqs.fa --first-hit -o first_hits.tsv
umotif scan motifs.meme -s seqs.fa --no-overlaps -o best_hits.tsv
```

<!-- output: hits.tsv 5 -->

A genome can be scanned whole or only in ranges. Hits in ranges are
reported on the genome, or within each range with `--local`. `--bed` writes
the hits as BED, named by motif, ready for a genome browser:

```sh
umotif scan motifs.meme -g genome.fa.gz -P 1e-5 -p 4 -o genome_hits.tsv
umotif scan motifs.meme -g genome.fa.gz -r peaks.bed --bed -o peak_hits.bed
umotif scan motifs.meme -g genome.fa.gz -r peaks.bed --local -o peak_hits.tsv
```

<!-- output: peak_hits.bed 4 -->

`variants` scores how the variants of a VCF file change the motif sites
they fall in, whether substitutions, insertions or deletions. Each row is a
variant and site, with its score and P-value for each allele. With
`--hits`, only the sites of an earlier genome scan are scored:

```sh
umotif variants motifs.meme -v calls.vcf.gz -g genome.fa.gz -o variant_sites.tsv
umotif variants motifs.meme -v calls.vcf.gz -g genome.fa.gz --hits genome_hits.tsv -o variant_hits.tsv
```

### Sequences and backgrounds

`shuffle` keeps the k-mer counts of sequences, dinucleotides by default.
`extract` writes the sequences of ranges, reverse complementing those on
the minus strand:

```sh
umotif shuffle -s peaks.fa --seed 1 -o shuffled.fa
umotif extract -g genome.fa.gz -r peaks.bed -o peaks.fa
```

`bkg` draws background sequences from the genome, matched to the targets on
GC content and length. `-n` draws several for each target, and `--table`
records which matched which:

```sh
umotif bkg -g genome.fa.gz -r peaks.bed -n 2 --seed 1 --table matched.tsv -o bkg.fa
```

<!-- output: matched.tsv 5 -->

`enrich` and `find` compare the targets with background sequences given
with `-b`. Without them they use shuffled targets, here keeping
trinucleotides with `--shuffle-k 3`. `find` and `find-gapped` shuffle the
targets three times by default, which `--bkg-shuffles` changes. This lets
them find motifs in only a few percent of the targets whatever the seed:

```sh
umotif enrich motifs.meme -s peaks.fa --shuffle-k 3 --seed 1 -o enriched.tsv
umotif find -s peaks.fa --shuffle-k 3 --seed 1 -o found.meme
```

<!-- output: enriched.tsv 5 9 -->

`find-gapped`, which is experimental for now, looks for motifs made of two
parts with a spacer of varying length, such as direct repeats of a half
site. Each motif is tested in a share of the targets and background held
out from the search, 0.3 by default. For peaks from a genome, a background
drawn from the genome with `bkg` gives far fewer false motifs than shuffled
targets.

Here `find-gapped` finds the G-box planted 23 letters from the AP-1 site, read on the
other strand. No motif format keeps gaps, so the motifs are written as
their two parts joined, and the table gives each gap:

```sh
umotif bkg -s peaks.fa -g genome.fa.gz --exclude peaks.bed --seed 1 -o bkg.fa
umotif find-gapped -s peaks.fa -b bkg.fa --max-gap 30 --seed 1 --table gapped.tsv -o gapped.meme
```

<!-- output: gapped.tsv 2 8 -->

### Peaks and their surroundings

`proximity` tests whether the hits of each motif sit closer to anchors,
such as transcription start sites, than expected by chance. It scans the
genome given with `-g`:

```sh
umotif proximity motifs.meme -a tss.bed -g genome.fa.gz -o near_tss.tsv
```

<!-- output: near_tss.tsv 4 9 -->

Three more commands work on the motifs in a set of peaks:

- `coocc` tests which pairs of motifs share peaks more often than chance.
- `posenr` tests which motifs gather at particular positions within the
  peaks, by default around their centres.
- `matrix` writes a peak by motif table of matches for single-cell ATAC
  analyses.

```sh
umotif coocc motifs.meme -g genome.fa.gz -r peaks.bed -o pairs.tsv
umotif posenr motifs.meme -g genome.fa.gz -r peaks.bed -o positions.tsv
umotif matrix motifs.meme -g genome.fa.gz -r peaks.bed -o peak_by_motif.tsv
```

<!-- output: pairs.tsv 4 9 -->

<!-- output: positions.tsv 4 8 -->

<!-- output: peak_by_motif.tsv 5 -->

## Reference

The help of every command follows, as printed by `umotif COMMAND --help`.

<!-- command help -->
