#!/usr/bin/env bash
# Run the whole benchmark, stage by stage. Each stage skips work already done,
# so the script can be run again after a failure.
#
#   ./run.sh [--models REGEX ...]
#
# Environment: PYTHON is a Python with universalmotif (with the h5py extra),
# pybigtools, pyBigWig and scipy; TF_PYTHON one with TensorFlow 2.15, for the
# chromBPNet and BPNet models; TORCH_PYTHON one with PyTorch, for the
# ProCapNet models; MODISCO the modisco command of modisco-lite; WORK the
# directory everything goes in (./work by default).

set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
TF_PYTHON=${TF_PYTHON:-$PYTHON}
TORCH_PYTHON=${TORCH_PYTHON:-$PYTHON}
WORK=${WORK:-work}
export MODISCO=${MODISCO:-modisco}
models=()
[[ $# -gt 0 && $1 == --models ]] && models=("$@")

$PYTHON harness.py fetch --work "$WORK" "${models[@]}"
$PYTHON harness.py prep --work "$WORK" "${models[@]}"
$PYTHON harness.py run --work "$WORK" "${models[@]}"
$PYTHON checks.py support --work "$WORK" "${models[@]}"
$PYTHON insertion.py prep --work "$WORK" "${models[@]}"
$TF_PYTHON insertion.py run --work "$WORK" --model-kind tf "${models[@]}"
$TORCH_PYTHON insertion.py run --work "$WORK" --model-kind torch "${models[@]}"
$PYTHON insertion.py collect --work "$WORK"
$PYTHON checks.py evidence --work "$WORK"
