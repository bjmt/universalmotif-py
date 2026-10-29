#!/usr/bin/env zsh
# Run the R and Python benchmarks and summarize them.
#
#   ./run.zsh [--small] [--runs N] [--check] [--out DIR] [--tasks "a b ..."]
#
# Every task runs in its own process under /usr/bin/time -l, once with
# --noop for the memory used by the loaded packages and inputs, then N times
# (5 by default), alternating R and Python. --small runs each task at about a
# twentieth of the full size, for testing. --check also writes each result
# once and compares the two languages with check_outputs.py. The raw timings
# go to DIR/raw.tsv, the versions to DIR/versions.txt and the medians to
# DIR/summary.md.
#
# Environment: PYTHON and RSCRIPT name the interpreters (python3 and Rscript
# by default), BENCH_R_LIB a library to load the R package from first, and
# UM_R_SOURCE the R package's source checkout, for recording its commit.
# The inputs must first be made with prepare_data.py.

setopt err_exit no_unset pipe_fail

here=${0:A:h}
out=$here/results
runs=5
small=0
check=0
only=""
while (( $# )); do
  case $1 in
    --small) small=1 ;;
    --runs) runs=$2; shift ;;
    --check) check=1 ;;
    --out) out=${2:A}; shift ;;
    --tasks) only=$2; shift ;;
    *) print -u2 "unknown option $1"; exit 1 ;;
  esac
  shift
done

python=${PYTHON:-python3}
rscript=${RSCRIPT:-Rscript}

# Task, full size, small size and thread counts. Sizes count motifs for the
# motif tasks, sites for create_motif, bases for scan_chromosome and
# sequences for the rest.
tasks=(
  "read_meme           8790     879      1"
  "write_meme          8790     879      1"
  "convert_type        87900    4395     1"
  "create_motif        2000000  100000   1"
  "scan_50             10000    500      1,2"
  "scan_400            1000     50       1,2"
  "scan_chromosome     30427671 1500000  1,2"
  "enrich_motifs       1000     100      1,2"
  "motif_pvalue        879      44       1,2"
  "compare_motifs      879      200      1,2"
  "merge_similar       879      200      1,2"
  "shuffle_k1          100000   5000     1,2"
  "shuffle_k2          100000   5000     1,2"
  "get_bkg             100000   5000     1,2"
  "sequence_complexity 100000   5000     1,2"
)

mkdir -p $out
export OMP_NUM_THREADS=1

# Versions and hardware.
{
  print "date\t$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  print "os\tmacOS $(sw_vers -productVersion) ($(uname -m))"
  print "cpu\t$(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores"
  print "memory\t$(( $(sysctl -n hw.memsize) / 1024 ** 3 )) GB"
  $rscript -e '
    lib <- Sys.getenv("BENCH_R_LIB"); if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
    cat("R\t", R.version$major, ".", R.version$minor, "\n", sep = "")
    for (p in c("universalmotif", "Biostrings"))
      cat(p, " (R)\t", as.character(packageVersion(p)), "\n", sep = "")'
  if [[ -n ${UM_R_SOURCE:-} ]]; then
    print "universalmotif (R) commit\t$(git -C $UM_R_SOURCE describe --always --dirty)"
  fi
  $python -c '
import platform, numpy, polars, universalmotif
print(f"Python\t{platform.python_version()}")
print(f"universalmotif (Python)\t{universalmotif.__version__}")
print(f"polars\t{polars.__version__}")
print(f"numpy\t{numpy.__version__}")'
  print "universalmotif (Python) commit\t$(git -C $here describe --always --dirty)"
} > $out/versions.txt
cat $out/versions.txt

# Runs one task in one language and appends a line to raw.tsv:
# language, task, size, threads, run, seconds, peak RSS in bytes, summary.
run_one() {
  local lang=$1 task=$2 size=$3 threads=$4 run=$5; shift 5
  local cmd
  if [[ $lang == R ]]; then
    cmd=($rscript $here/bench.R $task $size $threads "$@")
  else
    cmd=($python $here/bench.py $task $size $threads "$@")
  fi
  local log=$out/time.log
  local line
  line=$(POLARS_MAX_THREADS=$threads /usr/bin/time -l $cmd 2> $log | tail -1) || {
    print -u2 "failed: $cmd"; cat $log >&2; return 1
  }
  local rss=$(awk '/maximum resident set size/ {print $1}' $log)
  local fields=(${(ps:\t:)line})
  print -- "$lang\t$task\t$size\t$threads\t$run\t$fields[5]\t$rss\t$fields[6]" >> $out/raw.tsv
  print -- "$lang\t$task\t$size\t$threads\t$run\t$fields[5]\t$rss\t$fields[6]"
}

print "language\ttask\tsize\tthreads\trun\tseconds\tpeak_rss\tsummary" > $out/raw.tsv

for spec in $tasks; do
  local parts=(${=spec})
  local task=$parts[1]
  local size=$parts[$(( small ? 3 : 2 ))]
  [[ -n $only && " $only " != *" $task "* ]] && continue
  for threads in ${(s:,:)parts[4]}; do
    for lang in R Python; do
      run_one $lang $task $size $threads noop --noop
    done
    for (( i = 1; i <= runs; i++ )); do
      for lang in R Python; do
        run_one $lang $task $size $threads $i
      done
    done
  done
  if (( check )); then
    mkdir -p $out/check
    for lang in R Python; do
      run_one $lang $task $size 1 check --out $out/check/${lang}_$task.tsv
    done
  fi
done

rm -f $out/time.log
(( check )) && $python $here/check_outputs.py $out/check
$python $here/summarize.py $out/raw.tsv | tee $out/summary.md
