#!/bin/zsh
# Times TF-MoDISco and contribution_motifs() on the same 30,000 regions, and
# contribution_motifs() with and without its seqlet cap on every region,
# each run in its own process with 2 threads under /usr/bin/time -l.
#
#   ./run.zsh WORKDIR
#
# PYTHON is the Python with universalmotif, and MODISCO the modisco command
# of a modisco-lite 3.0.0 install.
set -e
here=${0:A:h}
work=${1:A}
python=${PYTHON:-python3}
modisco=${MODISCO:-modisco}
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2

timed() {
  /usr/bin/time -l "$@" 2>&1 | grep -E "^ContributionMotifs|real|maximum resident|rror"
}

[[ -f $work/sample/observed.npy ]] || $python $here/prepare.py sample $work/sample
[[ -f $work/all/observed.npy ]] || $python $here/prepare.py all ~/.cache/universalmotif-docs/ENCFF407GCO.bigWig \
  ~/.cache/universalmotif-docs/ENCFF834JBL.bed.gz $work/all

echo "== TF-MoDISco, 30,000 regions"
timed $modisco motifs -s $work/sample/ohe.npy -a $work/sample/hyp.npy -n 50000 -w 500 -o $work/sample/modisco.h5
echo "== contribution_motifs(), 30,000 regions"
timed $python $here/run_cm.py $work/sample
echo "== contribution_motifs() with hypothetical scores, 30,000 regions"
timed $python $here/run_cm.py $work/sample hypothetical
echo "== contribution_motifs(), every region"
timed $python $here/run_cm.py $work/all
echo "== contribution_motifs(), every region, every seqlet"
timed $python $here/run_cm.py $work/all max_seqlets=None
