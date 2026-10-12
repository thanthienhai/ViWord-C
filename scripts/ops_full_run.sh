#!/usr/bin/env bash
# ViWord-C full run for the GPU ops team: one file, two parts. Needs 1x H100 80 GB, ~100 GB of disk,
# internet (GitHub, Hugging Face, pip) and read access to the private HF repo thanthienhai/viword-c-data
# (export HF_TOKEN=hf_xxx, or `huggingface-cli login`).
#
#   bash ops_full_run.sh part1   # ~5-6 h: environment, data check, pre-check of the new tasks,
#                                #         diagnostics (gates G1/G2/G5), phase 1 (gates on dev, labels, 6 encoders)
#   bash ops_full_run.sh part2   # ~8-12 h: phase 2 (evaluation on 7 test sets + analysis)
#
# After each part, send back the .tgz printed on the last line. Start part2 ONLY when the project
# owner says so. Do not edit code on the machine; on an error, send the .tgz anyway (it holds the logs).
# Settings via environment, e.g.: GPU=1 WORKDIR=/mnt/hps/anhm-paper/viword-c-full bash ops_full_run.sh part1
# After an error, rerun the same command with the same WORKDIR: finished steps are skipped (phase 1 itself
# restarts from its first stage). Out of GPU memory in vLLM: add VIWORD_VLLM_GPU_UTIL=0.7. Steps: docs/RUN_OPS.md
set -euo pipefail

SELF=$(realpath "$0")
export WORKDIR=${WORKDIR:-$PWD/viword-c-full} # fresh folder, kept between part1 and part2 (PVC/NFS)
CODE_REPO=${CODE_REPO:-https://github.com/thanthienhai/ViWord-C.git}
CODE_REF=${CODE_REF:-main}
export HF_DATA_REPO=${HF_DATA_REPO:-thanthienhai/viword-c-data}
export HF_DATA_REVISION=${HF_DATA_REVISION:-main}   # checked file by file against data/SHA256SUMS in git
export GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
READER=${READER:-Qwen/Qwen2.5-7B-Instruct}
LL2=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
OUT=results/full

log() { echo -e "\n=== $* ($(date '+%F %T'))"; }
die() { echo -e "\n!!! $*" >&2; exit 1; }

pack() {  # tarball next to WORKDIR, printed last so the ops team knows what to send
  local tgz
  tgz="$(dirname "$WORKDIR")/viword_$1_$(date +%Y%m%d_%H%M).tgz"
  tar czf "$tgz" -C "$WORKDIR" "$OUT"
  echo -e "\nSend back: $tgz ($(du -h "$tgz" | cut -f1))"
}

step() {  # run one step with its own log in $OUT; on failure print the tail and return 1
  local name=$1; shift
  if [ -f "$OUT/.done_$name" ]; then log "$name: done in an earlier run, skipped"; return 0; fi
  log "$name"
  # in a child bash, so that `set -e` applies inside the step (bash ignores it under `if` and `||`)
  if ! bash "$SELF" _call "$@" > "$OUT/$name.log" 2>&1; then
    tail -n 40 "$OUT/$name.log"
    log "$name FAILED, see $OUT/$name.log"
    return 1
  fi
  touch "$OUT/.done_$name"
  log "$name done"
}
run() { step "$@" || { pack "$PART-FAILED"; exit 1; }; }           # later steps need this one
try_run() { step "$@" || echo "$1" >> "$OUT/FAILED_STEPS"; }       # phase 1 does not need this one

check_data() {  # every data file must match the checksums committed with the code
  sha256sum --quiet -c data/SHA256SUMS || die "data differs from data/SHA256SUMS: wrong HF revision"
  python scripts/check_segmented.py --directory data/segmented --segmenter vncorenlp \
    || die "segmented data is missing VnCoreNLP provenance or has been modified"
  python - <<EOF | tee "$OUT/data_revision.txt"
from huggingface_hub import HfApi
info = HfApi().dataset_info("$HF_DATA_REPO", revision="$HF_DATA_REVISION")
print(f"data {info.id} @ {info.sha}")
EOF
  echo "code $(git rev-parse HEAD)" | tee -a "$OUT/data_revision.txt"
}

precheck() {  # 20 dev examples of each task type that never ran on a GPU, with baselines only
  mkdir -p "$OUT/precheck"
  for task in viquad gsm8k oneruler; do
    python scripts/evaluate.py --task "$task" --data "data/tasks/${task}_dev.jsonl" \
      --segmented "data/segmented/${task}_dev.jsonl" --reader "$READER" --limit 20 \
      --methods none no_context truncation tfidf_sent "scored:model=$LL2,unit=syllable,name=llmlingua2_syl" \
      --out "$OUT/precheck/${task}.jsonl"
  done
  # translate-then-compress (llmlingua package) runs only in phase 2, on Belebele
  python scripts/evaluate.py --task belebele --data data/tasks/belebele.jsonl \
    --segmented data/segmented/belebele.jsonl --reader "$READER" --limit 20 \
    --methods none "translate:inner=llmlingua2" --out "$OUT/precheck/belebele_translate.jsonl"
  python - <<'EOF'
import glob
from collections import defaultdict
from viword.data import read_jsonl
from viword.eval import MAIN_METRIC

groups = defaultdict(list)
for path in sorted(glob.glob("results/full/precheck/*.jsonl")):
    if not path.endswith(".compressed.jsonl"):
        for r in read_jsonl(path):
            groups[r["task"], r["method"], r["ratio"]].append(r)
bad = []
print("| task | method | ratio | n | score | empty answers |\n|---|---|---|---|---|---|")
for (task, method, ratio), rows in sorted(groups.items()):
    score = sum(r[MAIN_METRIC[task]] for r in rows) / len(rows)
    empty = sum(not str(r["prediction"]).strip() for r in rows) / len(rows)
    print(f"| {task} | {method} | {ratio} | {len(rows)} | {score:.3f} | {empty:.0%} |")
    if (method == "none" and score == 0) or empty > 0.5:
        bad.append(f"{task}/{method}@{ratio}")
if bad:
    raise SystemExit(f"pre-check FAILED (score 0 or empty answers): {', '.join(bad)}")
print("pre-check OK")
EOF
}

diagnose() {  # G1 (CBR), G2 (negated ViNLI pairs), G5 on the current segmentation, with the 1.5B LM
  bash scripts/run_pipeline.sh protect_eval diagnose
  cp results/diagnose/belebele.json results/diagnose/vinli.json results/diagnose/gate_report.md \
    results/probe_negation_eval.md results/probe_negation_eval.json "$OUT"/
  for report in results/diagnose/llmlingua2vi_*.json; do
    [ -e "$report" ] && cp "$report" "$OUT"/
  done
}

part1() {
  [ -n "${HF_TOKEN:-}" ] || [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ] \
    || die "no Hugging Face token: export HF_TOKEN=hf_xxx (read access to $HF_DATA_REPO)"
  if [ -e "$WORKDIR" ]; then  # rerun after an error: same code and environment, finished steps are skipped
    [ -f "$WORKDIR/$OUT/ops_part1.log" ] \
      || die "$WORKDIR exists but is not a part1 folder: delete it or set WORKDIR=<new folder>"
    cd "$WORKDIR"
  else
    git clone -q "$CODE_REPO" "$WORKDIR"
    cd "$WORKDIR"
    git checkout -q "$CODE_REF"
    [ -f data/SHA256SUMS ] || { rm -rf "$WORKDIR"; die "code at $CODE_REF has no data/SHA256SUMS: ask the project owner"; }
  fi
  mkdir -p "$OUT"
  rm -f "$OUT/FAILED_STEPS"
  exec > >(tee -a "$OUT/ops_part1.log") 2>&1
  log "part1: code $(git rev-parse --short HEAD), data $HF_DATA_REPO @ $HF_DATA_REVISION, GPU $GPU"
  df -h "$WORKDIR" | tail -1

  run setup bash scripts/smoke_h100.sh setup check_data   # venv, 48 unit tests, GPU check, data pull
  grep -E "passed|failed|error|GPU:" "$OUT/setup.log"
  if ! grep -qE "^[0-9]+ passed in" "$OUT/setup.log" || grep -qE "[0-9]+ (failed|error)" "$OUT/setup.log"; then
    pack "$PART-FAILED"; die "unit tests did not all pass"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip freeze > "$OUT/pip_freeze.txt"
  nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || true
  run check_data check_data
  # a failure in these two does not block phase 1; it is listed in $OUT/FAILED_STEPS
  try_run precheck precheck
  try_run diagnose diagnose
  # phase 1 writes its own stage logs and full.log into $OUT and stops on the first error
  bash scripts/full_h100.sh phase1 || { pack "$PART-FAILED"; die "phase 1 failed, see $OUT/full.log"; }
  [ ! -s "$OUT/FAILED_STEPS" ] || echo "Phase 1 done, but these steps failed: $(xargs < "$OUT/FAILED_STEPS")"
  pack part1
}

part2() {
  [ -d "$WORKDIR/$OUT" ] || die "$WORKDIR/$OUT not found: run part1 first, in the same WORKDIR"
  cd "$WORKDIR"
  grep -q "Phase 1 done" "$OUT/full.log" || die "phase 1 did not finish: rerun part1"
  exec > >(tee -a "$OUT/ops_part2.log") 2>&1
  log "part2: code $(git rev-parse --short HEAD), GPU $GPU"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  run check_data_part2 check_data
  bash scripts/full_h100.sh phase2 || { pack "$PART-FAILED"; die "phase 2 failed, see $OUT/full.log"; }
  pack part2
}

PART=${1:-}
if [ "$PART" = _call ]; then shift; "$@"; exit; fi                 # one step, called by step()
[[ $PART =~ ^part[12]$ ]] || { grep -E '^#( |$)' "$0" | head -16; exit 1; }
"$PART"
