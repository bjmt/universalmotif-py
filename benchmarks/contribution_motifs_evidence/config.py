"""What the benchmark runs: the models, how their regions become windows,
and the methods. harness.py, checks.py and insertion.py read it."""

# The models, as regular expressions matched against the names in
# manifest.tsv: 16 single-factor BPNet models, three chromBPNet models of
# ATAC-seq and every ProCapNet model ENCODE hosts.
MODELS = [r"^bpnet_", r"^chrombpnet_(K562|HepG2|GM12878)_ATAC-seq$", r"^procapnet_"]

# Heads of BPNet and chromBPNet models with hypothetical scores, which
# TF-MoDISco needs. ProCapNet's come with every head.
HYPOTHETICAL_HEADS = ["counts"]

# Width of the window taken from the middle of each region. BPNet and
# chromBPNet regions are taken about their summits, and windows that overlap
# one kept before are dropped. ProCapNet's scores cover the whole 1 kb output
# window of each peak, and every peak is kept.
WINDOW = {"chrombpnet": 500, "bpnet": 500, "procapnet": 1000}

# Threads per run, and runs side by side (harness.py run --jobs).
THREADS = 2
JOBS = 10

# The methods. "cm" entries are keyword arguments for contribution_motifs(),
# with hypothetical scores unless they say otherwise; "modisco" runs
# modisco-lite on the same windows and hypothetical scores, with at most
# max_seqlets seqlets of each sign.
METHODS = {
    # The defaults: two seqlet passes, the contribution test, later rounds held
    # to four fifths of min_gain and at most 50,000 seqlets of each sign.
    "cm": {"kind": "cm", "kwargs": {}},
    "cm_observed": {"kind": "cm", "kwargs": {}, "hypothetical": False},
    "cm_uncapped": {"kind": "cm", "kwargs": {"max_seqlets": None}},
    # One seqlet pass, every seqlet, and later rounds held to min_gain itself.
    "cm_one_pass": {"kind": "cm", "kwargs": {"seqlet_passes": 1, "max_seqlets": None, "later_min_gain": 0.0025}},
    "modisco": {"kind": "modisco", "max_seqlets": 50_000},
}

# Files shared by every model.
GENOME_2BIT = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"
JASPAR = ("https://jaspar.elixir.no/download/data/2024/CORE/"
          "JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt")
PHYLOP = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"

# Cell lines whose ChIP-seq peaks are checked against the motifs of every
# model of that cell line.
CHIP_CELLS = ["K562", "HepG2", "GM12878"]
