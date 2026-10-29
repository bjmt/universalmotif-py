# Time one task of the R package universalmotif.
#
#   Rscript bench.R TASK SIZE NTHREADS [--noop] [--out FILE] [--data DIR]
#
# Loads the inputs of TASK at SIZE, then times the task alone and prints a
# tab-separated line: language, task, size, threads, seconds and a short
# summary of the result. With --noop the inputs are loaded but the task is
# not run, which gives the memory used before the task. With --out the
# result is written to FILE for check_outputs.py. The inputs come from
# prepare_data.py. If BENCH_R_LIB is set, universalmotif is loaded from that
# library first.

lib <- Sys.getenv("BENCH_R_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))

suppressPackageStartupMessages({
  library(universalmotif)
  library(Biostrings)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1L]
}
script.dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(), value = TRUE)[1])))
task <- args[1]
size <- as.integer(args[2])
nthreads <- as.integer(args[3])
noop <- "--noop" %in% args
out <- opt("--out")
data.dir <- opt("--data", file.path(script.dir, "data"))
data <- function(...) file.path(data.dir, ...)

pvals <- 10^-seq(2, 8, length.out = 100)

motifs_n <- function(file, n) {
  m <- read_meme(file)
  rep(m, length.out = n)
}

read_seqs <- function(file, n) {
  seqs <- readDNAStringSet(file)
  seqs[seq_len(min(n, length(seqs)))]
}

# Written results, read by check_outputs.py.
write_tsv <- function(x, file) {
  write.table(x, file, sep = "\t", quote = FALSE, row.names = FALSE)
}

write_matrices <- function(motifs, file) {
  write_tsv(data.frame(
    name = vapply(motifs, function(m) m@name, ""),
    values = vapply(motifs, function(m)
      paste(sprintf("%.10g", as.numeric(m@motif)), collapse = ","), "")
  ), file)
}

write_hits <- function(hits, file) {
  # Positions as in BED files, as the Python package reports them.
  write_tsv(data.frame(
    motif = hits$motif, sequence = hits$sequence,
    start = hits$start - 1L, end = hits$end, strand = hits$strand,
    score = hits$score, pvalue = hits$pvalue
  ), file)
}

scan_task <- function(motif.file, n.motifs, seq.file) {
  motifs <- read_meme(motif.file)[seq_len(n.motifs)]
  seqs <- read_seqs(seq.file, size)
  list(
    run = function() scan_sequences_lite(motifs, seqs, pvalue = 1e-4,
      RC = TRUE, nthreads = nthreads, return.granges = FALSE),
    summary = function(x) sprintf("hits=%d", nrow(x)),
    write = write_hits
  )
}

chrom_task <- function() {
  motifs <- read_meme(data("plants.meme"))[1:10]
  chrom <- readDNAStringSet(data("chr1.fa"))
  chrom <- subseq(chrom, 1, min(size, width(chrom)))
  list(
    run = function() scan_sequences_lite(motifs, chrom, pvalue = 1e-4,
      RC = TRUE, nthreads = nthreads, return.granges = FALSE),
    summary = function(x) sprintf("hits=%d", nrow(x)),
    write = write_hits
  )
}

shuffle_task <- function(k) {
  seqs <- read_seqs(data("windows.fa"), size)
  list(
    run = function() shuffle_sequences(seqs, k = k, nthreads = nthreads,
      rng.seed = 1),
    summary = function(x) sprintf("bases=%d", sum(width(x))),
    write = function(x, file) writeXStringSet(x, file)
  )
}

tasks <- list(

  read_meme = function() {
    file <- data(sprintf("motifs_%d.meme", size))
    list(
      run = function() read_meme(file),
      summary = function(x) sprintf("motifs=%d", length(x)),
      write = write_matrices
    )
  },

  write_meme = function() {
    motifs <- read_meme(data(sprintf("motifs_%d.meme", size)))
    file <- tempfile(fileext = ".meme")
    list(
      run = function() write_meme(motifs, file, overwrite = TRUE),
      summary = function(x) sprintf("bytes=%d", file.size(file)),
      write = function(x, out) file.copy(file, out, overwrite = TRUE)
    )
  },

  convert_type = function() {
    motifs <- motifs_n(data("motifs_879.meme"), size)
    list(
      run = function() convert_type(motifs, "PWM"),
      summary = function(x) sprintf("motifs=%d", length(x)),
      write = function(x, file) write_matrices(x[1:min(879, length(x))], file)
    )
  },

  create_motif = function() {
    sites <- readLines(data("sites.txt"), n = size)
    list(
      run = function() create_motif(sites, type = "PCM"),
      summary = function(x) sprintf("nsites=%d", as.integer(x@nsites)),
      write = function(x, file) write_matrices(list(x), file)
    )
  },

  scan_50 = function() scan_task(data("motifs_879.meme"), 50, data("windows.fa")),

  scan_400 = function() scan_task(data("motifs_879.meme"), 400, data("windows.fa")),

  scan_chromosome = chrom_task,

  enrich_motifs = function() {
    motifs <- read_meme(data("plants.meme"))[1:100]
    targets <- read_seqs(data("peaks.fa"), size)
    bkg <- read_seqs(data("background.fa"), size)
    list(
      run = function() enrich_motifs_lite(motifs, targets, bkg, pvalue = 1e-4,
        qvalue = 1, nthreads = nthreads),
      summary = function(x) sprintf("significant=%d", sum(x$qvalue < 0.05)),
      write = function(x, file) write_tsv(data.frame(
        motif = x$motif, target_seq_hits = x$target.seq.hits,
        bkg_seq_hits = x$bkg.seq.hits, pvalue = x$pvalue, qvalue = x$qvalue
      ), file)
    )
  },

  motif_pvalue = function() {
    motifs <- motifs_n(data("motifs_879.meme"), size)
    p <- rep(list(pvals), length(motifs))
    list(
      run = function() motif_pvalue(motifs, pvalue = p, nthreads = nthreads),
      summary = function(x) sprintf("scores=%d", length(unlist(x))),
      write = function(x, file) write_tsv(data.frame(
        motif = rep(seq_along(x) - 1L, lengths(x)),
        pvalue = rep(pvals, length(x)), score = unlist(x)
      ), file)
    )
  },

  compare_motifs = function() {
    motifs <- motifs_n(data("motifs_879.meme"), size)
    list(
      run = function() compare_motifs_lite(motifs, matrix.out = "pvalue",
        nthreads = nthreads),
      summary = function(x) sprintf("cells=%d", length(x)),
      write = function(x, file)
        write.table(unname(x), file, sep = "\t", row.names = FALSE,
          col.names = FALSE)
    )
  },

  merge_similar = function() {
    motifs <- motifs_n(data("motifs_879.meme"), size)
    list(
      run = function() merge_similar_lite(motifs, nthreads = nthreads),
      summary = function(x) sprintf("motifs=%d", length(x)),
      write = write_matrices
    )
  },

  shuffle_k1 = function() shuffle_task(1),

  shuffle_k2 = function() shuffle_task(2),

  get_bkg = function() {
    seqs <- read_seqs(data("windows.fa"), size)
    list(
      run = function() get_bkg(seqs, k = 1:3, nthreads = nthreads),
      summary = function(x) sprintf("kmers=%d", nrow(x)),
      write = function(x, file) write_tsv(data.frame(
        kmer = x$klet, count = x$count
      ), file)
    )
  },

  sequence_complexity = function() {
    seqs <- read_seqs(data("windows.fa"), size)
    list(
      run = function() sequence_complexity(seqs, window.size = 20,
        method = "WoottonFederhen", nthreads = nthreads),
      summary = function(x) sprintf("windows=%d", nrow(x)),
      write = function(x, file) write_tsv(data.frame(
        sequence = x$sequence, start = x$start - 1L, end = x$stop,
        complexity = x$complexity
      ), file)
    )
  }

)

if (!task %in% names(tasks))
  stop("unknown task `", task, "`; one of: ",
       paste(names(tasks), collapse = ", "), call. = FALSE)

t <- tasks[[task]]()
invisible(gc())

seconds <- NA_real_
summary <- "noop"
if (!noop) {
  start <- Sys.time()
  result <- t$run()
  seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  summary <- t$summary(result)
  if (!is.null(out)) invisible(t$write(result, out))
}

cat(paste("R", task, size, nthreads, sprintf("%.6f", seconds), summary,
          sep = "\t"), "\n", sep = "")
