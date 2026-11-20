#!/bin/zsh
# How TF-MoDISco and contribution_motifs() scale with the number of seqlets:
# nested random samples of the K562 regions, each method in its own process
# with 2 threads under /usr/bin/time -l. TF-MoDISco runs as ENCODE ran it,
# with up to 1,000,000 seqlets of each sign; contribution_motifs() runs with
# hypothetical scores, with every seqlet and with its default cap.
#
#   ./scaling.zsh WORKDIR N ...
#
# PYTHON is the Python with universalmotif, MODISCO the modisco command of a
# modisco-lite 3.0.0 install, and ONLY set to modisco or cm runs one method.
set -e
here=${0:A:h}
work=${1:A}; shift
python=${PYTHON:-python3}
modisco=${MODISCO:-modisco}
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2

timed() {
  /usr/bin/time -l "$@" 2>&1 | grep -E "^ContributionMotifs|positive seqlets|negative seqlets|treemem|real|maximum resident|rror"
}

only=${ONLY:-all}
for n in "$@"; do
  d=$work/n$n
  [[ -f $d/observed.npy ]] || $python $here/prepare.py sample $d $n
  if [[ $only != cm ]]; then
  echo "== $n regions: TF-MoDISco"
  # TF-MoDISco forks workers, so the memory of its whole process tree is
  # sampled too.
  timed $python $here/treemem.py $modisco motifs -s $d/ohe.npy -a $d/hyp.npy -n 1000000 -w 500 -o $d/modisco.h5 -v
  fi
  [[ $only == modisco ]] && continue
  echo "== $n regions: contribution_motifs(), every seqlet"
  timed $python $here/run_cm.py $d hypothetical max_seqlets=None
  echo "== $n regions: contribution_motifs(), default cap"
  timed $python $here/run_cm.py $d hypothetical
done
echo finished
