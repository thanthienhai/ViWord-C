#!/usr/bin/env bash
# Direction B of ViWord-C (question-aware compression for Vietnamese legal RAG, ViBidLQA) for the GPU ops team:
# one file, three parts. Needs 1x H100 80 GB, ~60 GB of disk, internet (GitHub, Hugging Face, pip) and a Hugging
# Face token that can read the private repo thanthienhai/viword-c-legal-data (export HF_TOKEN=hf_xxx).
#
#   bash ops_b_run.sh part1   # ~30-40 min: environment, data check, smoke run of the whole chain on a few questions
#   bash ops_b_run.sh part2   # ~5-6 h: utility labels, 7 unit scorers, dev evaluation and its report
#   bash ops_b_run.sh part3   # ~1-1.5 h: the registered methods on the test set, once
#
# Each part packs a .tgz and its .sha256 (vibid_<part>_<date>.tgz). With HF_RESULTS_TOKEN (write access to the
# private results repo HF_RESULTS_REPO) both are uploaded there; otherwise send back the two files printed on the
# last line. Start the next part ONLY when the project owner says so. Do not edit code on the machine; on an error
# the .tgz is packed anyway (it holds the logs). After an error, rerun the same command with the same WORKDIR:
# finished steps are skipped, labels and trained scorers are kept, reader answers are reused.
# Settings via environment, e.g.: GPU=1 WORKDIR=/mnt/hps/anhm-paper/viword-b bash ops_b_run.sh part1
# Out of GPU memory in vLLM: add VIWORD_VLLM_GPU_UTIL=0.7. Steps and errors: docs/RUN_OPS_B.md
set -euo pipefail

SELF=$(realpath "$0")
export WORKDIR=${WORKDIR:-$PWD/viword-b}   # fresh folder, kept between the parts (PVC/NFS)
CODE_REPO=${CODE_REPO:-https://github.com/thanthienhai/ViWord-C.git}
CODE_REF=${CODE_REF:-rag-legal}
export HF_DATA_REPO_B=${HF_DATA_REPO_B:-thanthienhai/viword-c-legal-data}
export HF_DATA_REVISION_B=${HF_DATA_REVISION_B:-main}   # checked file by file against data/SHA256SUMS_B in git
export GPU=${GPU:-0}
export HF_RESULTS_REPO=${HF_RESULTS_REPO:-thanthienhai/viword-c-results}   # packages back to the project owner
OUT=results/b
PART=${1:-}

log() { echo -e "\n=== $* ($(date '+%F %T'))"; }
die() { echo -e "\n!!! $*" >&2; exit 1; }

pack() {  # tarball + checksum next to WORKDIR, uploaded if possible; the last line says what to send
  local tgz py=python3
  tgz="$(dirname "$WORKDIR")/vibid_$1_$(date +%Y%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$WORKDIR" "$OUT"
  (cd "$(dirname "$tgz")" && sha256sum "$(basename "$tgz")" > "$(basename "$tgz").sha256")
  cat "$tgz.sha256"
  if [ -n "${HF_RESULTS_TOKEN:-}" ]; then
    [ -x "$WORKDIR/.venv/bin/python" ] && py="$WORKDIR/.venv/bin/python"
    if HF_TOKEN=$HF_RESULTS_TOKEN "$py" "$WORKDIR/scripts/hf_results.py" push "$tgz"; then
      echo -e "\nUploaded: $(basename "$tgz") (+ .sha256) to $HF_RESULTS_REPO; nothing to send by hand"
      return 0
    fi
    echo "upload to $HF_RESULTS_REPO failed: send the two files below by hand"
  fi
  echo -e "\nSend back: $tgz ($(du -h "$tgz" | cut -f1)) and $tgz.sha256"
}

step() {  # run one step with its own log in $OUT/ops; on failure print the tail and return 1
  local name=$1; shift
  if [ -f "$OUT/ops/.done_$name" ]; then log "$name: done in an earlier run, skipped"; return 0; fi
  log "$name"
  # in a child bash, so that `set -e` and `die` apply inside the step (bash ignores set -e under `if` and `||`)
  if ! bash "$SELF" _call "$@" > "$OUT/ops/$name.log" 2>&1; then
    tail -n 40 "$OUT/ops/$name.log"
    log "$name FAILED, see $OUT/ops/$name.log"
    return 1
  fi
  touch "$OUT/ops/.done_$name"
  log "$name done"
}
run() { step "$@" || { pack "$PART-FAILED"; exit 1; }; }

revisions() {  # code and data versions of this part, for the report
  echo "# $PART $(date '+%F %T')"
  echo "code $(git rev-parse HEAD) ($CODE_REF)"
  .venv/bin/python - <<EOF
from huggingface_hub import HfApi
info = HfApi().dataset_info("$HF_DATA_REPO_B", revision="$HF_DATA_REVISION_B")
print(f"data {info.id} @ {info.sha}")
EOF
}

pull_code_only() {  # fast-forward to origin/$CODE_REF (a fix pushed by the project owner)
  local before after
  before=$(git rev-parse HEAD)
  git fetch -q origin "$CODE_REF"
  git merge -q --ff-only FETCH_HEAD || die "cannot fast-forward the code to origin/$CODE_REF: ask the project owner"
  after=$(git rev-parse HEAD)
  echo "code $before -> $after"
  git diff --stat "$before" "$after"
}

pull_code() {  # the same, then refresh the dependencies in case they changed
  pull_code_only
  .venv/bin/pip install -q -e ".[model,eval,baselines,dev]"
}

enter() {  # the WORKDIR of part1, its log, the environment
  [ -d "$WORKDIR/$OUT/ops" ] || die "$WORKDIR/$OUT/ops not found: run part1 first, in the same WORKDIR"
  cd "$WORKDIR"
  exec > >(tee -a "$OUT/ops/ops_$PART.log") 2>&1
  log "$PART: code $(git rev-parse --short HEAD), GPU $GPU"
}

part1() {
  [ -n "${HF_TOKEN:-}" ] || [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ] \
    || die "no Hugging Face token: export HF_TOKEN=hf_xxx (read access to $HF_DATA_REPO_B)"
  local rerun=0
  if [ -e "$WORKDIR" ]; then  # rerun after an error: same environment, finished steps are skipped
    [ -f "$WORKDIR/$OUT/ops/ops_part1.log" ] \
      || die "$WORKDIR exists but is not a part1 folder of direction B: delete it or set WORKDIR=<new folder>"
    cd "$WORKDIR"
    rerun=1
  else
    git clone -q "$CODE_REPO" "$WORKDIR"
    cd "$WORKDIR"
    git checkout -q "$CODE_REF"
    [ -f data/SHA256SUMS_B ] || { rm -rf "$WORKDIR"; die "code at $CODE_REF has no data/SHA256SUMS_B: ask the project owner"; }
  fi
  mkdir -p "$OUT/ops"
  exec > >(tee -a "$OUT/ops/ops_part1.log") 2>&1
  log "part1: code $(git rev-parse --short HEAD), data $HF_DATA_REPO_B @ $HF_DATA_REVISION_B, GPU $GPU"
  df -h "$WORKDIR" | tail -1
  if [ $rerun = 1 ]; then  # a fix pushed by the project owner after a failed smoke run must reach this rerun
    rm -f "$OUT/ops/.done_update_code_part1"
    if [ -x .venv/bin/pip ]; then run update_code_part1 pull_code; else run update_code_part1 pull_code_only; fi
    log "part1: code now $(git rev-parse --short HEAD)"
  fi
  run setup bash scripts/b_h100.sh setup        # venv, unit tests, GPU check
  grep -E "passed|failed|error|GPU:" "$OUT/ops/setup.log" || true
  .venv/bin/pip freeze > "$OUT/ops/pip_freeze.txt"
  nvidia-smi > "$OUT/ops/nvidia_smi.txt" 2>&1 || true
  run data bash scripts/b_h100.sh data          # pull from Hugging Face, check against data/SHA256SUMS_B
  revisions | tee -a "$OUT/ops/revisions.txt"
  run smoke bash scripts/b_h100.sh smoke        # labels with vLLM, one training epoch, 30 dev questions
  grep -A6 "smoke: 30 dev" "$OUT/b.log" | grep -E "rows|F1" || true
  pack part1
}

part2() {
  enter
  [ -f "$OUT/ops/.done_smoke" ] || die "part1 did not finish: rerun part1"
  # the latest code of the branch: the project owner may have pushed a fix after part1
  rm -f "$OUT/ops/.done_update_code_part2" "$OUT/ops/.done_data_part2"
  run update_code_part2 pull_code
  run data_part2 bash scripts/b_h100.sh data
  revisions | tee -a "$OUT/ops/revisions.txt"
  run labels bash scripts/b_h100.sh labels      # ~45-60 min
  run train bash scripts/b_h100.sh train        # ~1.5-2 h, 7 scorers
  run dev bash scripts/b_h100.sh dev            # ~2-3 h, writes results/b/dev.md
  pack part2
}

part3() {
  enter
  [ -f "$OUT/ops/.done_dev" ] || die "part2 did not finish: rerun part2"
  rm -f "$OUT/ops/.done_update_code_part3" "$OUT/ops/.done_data_part3"
  run update_code_part3 pull_code
  git ls-files --error-unmatch config/b_registration.json > /dev/null 2>&1 \
    || die "config/b_registration.json is not committed at origin/$CODE_REF: the project owner has not registered yet"
  run data_part3 bash scripts/b_h100.sh data
  revisions | tee -a "$OUT/ops/revisions.txt"
  run test bash scripts/b_h100.sh test          # refuses unless the registration is committed and unchanged
  pack part3
}

case $PART in
  part1|part2|part3) "$PART" ;;
  _call) shift; "$@" ;;
  *) sed -n '2,17p' "$SELF"; exit 1 ;;
esac
