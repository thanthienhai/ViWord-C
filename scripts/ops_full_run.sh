#!/usr/bin/env bash
# ViWord-C full run for the GPU ops team: one file, four parts. Needs 1x H100 80 GB, ~100 GB of disk,
# internet (GitHub, Hugging Face, pip) and a Hugging Face token that can read the private repo
# thanthienhai/viword-c-data (export HF_TOKEN=hf_xxx). No Java: BERTScore (VnCoreNLP + PhoBERT) runs on the
# project owner's machine after part3 (scripts/bertscore_local.sh).
#
#   bash ops_full_run.sh part1   # ~5-6 h: environment, data check, pre-check of the new tasks,
#                                #         diagnostics, phase 1 (gates on dev, labels, 6 encoders, ckpt_check)
#   bash ops_full_run.sh part2   # ~2-4 h: phase 2 (dev decisions: diagnostics, select π, MDE)
#   bash ops_full_run.sh part3   # ~12-18 h: update the code (committed π), phase 3 (7 test sets + analysis)
#   bash ops_full_run.sh part4   # ~12-18 h: the second reader (SeaLLM-7B-v2) on the 7 test sets, same code
#
# Each part packs a .tgz and its .sha256. With HF_RESULTS_TOKEN (write access to the private results repo
# HF_RESULTS_REPO) both are uploaded there; otherwise send back the two files printed on the last line.
# Start the next part ONLY when the project owner says so. Do not edit code on the machine; on an error,
# the .tgz is packed anyway (it holds the logs). Settings via environment, e.g.: GPU=1 WORKDIR=/mnt/hps/anhm-paper/viword-c-full bash ops_full_run.sh part1
# After an error, rerun the same command with the same WORKDIR: finished steps are skipped (a phase
# itself restarts from its first stage). Out of GPU memory in vLLM: add VIWORD_VLLM_GPU_UTIL=0.7.
# Steps: docs/RUN_OPS.md
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
READER2=${READER2:-SeaLLMs/SeaLLM-7B-v2}
export HF_RESULTS_REPO=${HF_RESULTS_REPO:-thanthienhai/viword-c-results}   # packages back to the project owner

log() { echo -e "\n=== $* ($(date '+%F %T'))"; }
die() { echo -e "\n!!! $*" >&2; exit 1; }

pack() {  # tarball + checksum next to WORKDIR, uploaded if possible; the last line says what to send
  local tgz py=python3
  tgz="$(dirname "$WORKDIR")/viword_$1_$(date +%Y%m%d_%H%M).tgz"
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

pull_data() {  # data files added by the project owner after part1 (e.g. results/teacher_test)
  python scripts/hf_data.py pull --repo "$HF_DATA_REPO" --revision "$HF_DATA_REVISION"
}

check_data() {  # every data file must match the checksums committed with the code
  sha256sum --quiet -c data/SHA256SUMS || die "data differs from data/SHA256SUMS: wrong HF revision"
  # Belebele has no dev split: belebele_dev holds the questions readers scored during development
  python scripts/test_exclusions.py belebele-dev || die "cannot derive data/tasks/belebele_dev.jsonl"
  python scripts/check_segmented.py --directory data/segmented --segmenter vncorenlp \
    || die "segmented data is missing VnCoreNLP provenance or has been modified"
  echo "# $(date '+%F %T')" >> "$OUT/data_revision.txt"   # one block per part: the code may move in part3
  python - <<EOF | tee -a "$OUT/data_revision.txt"
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
  # translate-then-compress (llmlingua package) runs only in phase 3, on Belebele; it is checked on
  # belebele_dev (Belebele has no dev split; a reader never sees an unseen test row before phase 3)
  python scripts/evaluate.py --task belebele --data data/tasks/belebele_dev.jsonl \
    --segmented data/segmented/belebele_dev.jsonl --reader "$READER" --limit 20 \
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

diagnose() {  # G1 (CBR), G2 (negated ViNLI pairs), G5; needs the filled blind sheets in data/annot
  bash scripts/run_pipeline.sh protect_eval diagnose
  cp results/diagnose/belebele.json results/diagnose/vinli.json results/diagnose/gate_report.md \
    results/protect_eval.md results/protect_eval.json \
    results/probe_negation_eval.md results/probe_negation_eval.json "$OUT"/
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

  run setup bash scripts/smoke_h100.sh setup check_data   # venv, unit tests, GPU check, data pull
  grep -E "passed|failed|error|GPU:" "$OUT/setup.log"
  # tests that need the prepared data are skipped here: setup runs them before the data pull
  if ! grep -qE "^[0-9]+ passed(, [0-9]+ skipped)?(, [0-9]+ warnings?)? in" "$OUT/setup.log" \
     || grep -qE "[0-9]+ (failed|errors?)" "$OUT/setup.log"; then
    pack "$PART-FAILED"; die "unit tests did not all pass"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip freeze > "$OUT/pip_freeze.txt"
  nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || true
  run check_data check_data
  # a failure in these two does not block phase 1; it is listed in $OUT/FAILED_STEPS
  # (diagnose runs again in phase 2, after training, and must pass there)
  try_run precheck precheck
  try_run diagnose diagnose
  # phase 1 writes its own stage logs and full.log into $OUT and stops on the first error
  bash scripts/full_h100.sh phase1 || { pack "$PART-FAILED"; die "phase 1 failed, see $OUT/full.log"; }
  [ ! -s "$OUT/FAILED_STEPS" ] || echo "Phase 1 done, but these steps failed: $(xargs < "$OUT/FAILED_STEPS")"
  pack part1
}

phase1_ended_at_ckpt_check_gate() {  # a part1 on code before 2026-10-23 stopped at a ckpt_check FAIL;
  # with its report written that is a gate result (the dev decisions of part2 may follow), not a broken run
  grep -q "train done" "$OUT/full.log" && grep -q "ckpt_check FAILED" "$OUT/full.log" \
    && [ -s results/ckpt_check/report.json ]
}

part2() {
  [ -d "$WORKDIR/$OUT" ] || die "$WORKDIR/$OUT not found: run part1 first, in the same WORKDIR"
  cd "$WORKDIR"
  grep -q "Phase 1 done" "$OUT/full.log" || phase1_ended_at_ckpt_check_gate \
    || die "phase 1 did not finish: rerun part1"
  exec > >(tee -a "$OUT/ops_part2.log") 2>&1
  log "part2: code $(git rev-parse --short HEAD), GPU $GPU"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # the dev decisions run on the latest code: a rule fixed after part1 started must reach them; the data
  # follows the code, since data/SHA256SUMS may have moved with it
  rm -f "$OUT/.done_update_code_part2" "$OUT/.done_pull_data_part2" "$OUT/.done_check_data_part2"
  run update_code_part2 pull_code
  log "part2: code now $(git rev-parse --short HEAD)"
  run pull_data_part2 pull_data
  run check_data_part2 check_data
  bash scripts/full_h100.sh phase2 || { pack "$PART-FAILED"; die "phase 2 failed, see $OUT/full.log"; }
  pack part2
}

pull_code() {  # fast-forward to origin/$CODE_REF, log what changed, refresh the dependencies
  local before after
  before=$(git rev-parse HEAD)
  git fetch -q origin "$CODE_REF"
  git merge -q --ff-only FETCH_HEAD || die "cannot fast-forward the code to origin/$CODE_REF: ask the project owner"
  after=$(git rev-parse HEAD)
  echo "code $before -> $after"
  git diff --stat "$before" "$after"
  pip install -q -e ".[seg,model,eval,baselines,dev]"   # in case the dependencies changed
}

update_code() {  # fast-forward to the commit with the π chosen in part2; log what else changed
  # select_pi wrote config/pi_selected.* here; keep that copy, the committed one replaces it
  mkdir -p "$OUT/pi_selected_part2"
  for f in config/pi_selected.json config/pi_selected.env; do
    if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      mv "$f" "$OUT/pi_selected_part2/"
    fi
  done
  pull_code
  git ls-files --error-unmatch config/pi_selected.json config/pi_selected.env >/dev/null 2>&1 \
    || die "config/pi_selected.* is not committed at origin/$CODE_REF: the project owner has not pushed π yet"
  for f in pi_selected.json pi_selected.env; do
    [ -e "$OUT/pi_selected_part2/$f" ] && ! diff -u --strip-trailing-cr "$OUT/pi_selected_part2/$f" "config/$f" \
      && echo "NOTE: committed config/$f differs from the part2 output above"
  done
}

part3() {
  [ -d "$WORKDIR/$OUT" ] || die "$WORKDIR/$OUT not found: run part1 first, in the same WORKDIR"
  cd "$WORKDIR"
  grep -q "Phase 2 done" "$OUT/full.log" || die "phase 2 did not finish: rerun part2"
  exec > >(tee -a "$OUT/ops_part3.log") 2>&1
  log "part3: code $(git rev-parse --short HEAD), GPU $GPU"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # always fetch the latest code and data: the project owner may have pushed a fix after a failed run
  rm -f "$OUT/.done_update_code" "$OUT/.done_pull_data_part3" "$OUT/.done_check_data_part3"
  run update_code update_code
  run pull_data_part3 pull_data
  run check_data_part3 check_data
  bash scripts/full_h100.sh phase3 || { pack "$PART-FAILED"; die "phase 3 failed, see $OUT/full.log"; }
  pack part3
}

part4() {  # second reader on the test sets, with the code and π of part3 (dev decisions are the primary reader's)
  [ -d "$WORKDIR/$OUT" ] || die "$WORKDIR/$OUT not found: run part1 first, in the same WORKDIR"
  cd "$WORKDIR"
  grep -q "Phase 3 done" "$OUT/full.log" || die "phase 3 did not finish: rerun part3"
  exec > >(tee -a "$OUT/ops_part4.log") 2>&1
  log "part4: code $(git rev-parse --short HEAD), reader $READER2, GPU $GPU"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  run reader2_template python scripts/check_reader_template.py --reader "$READER2" \
    --data data/tasks/vimmrc_dev.jsonl --task vimmrc
  cat "$OUT/reader2_template.log"
  run reader2_eval env READER="$READER2" bash scripts/run_pipeline.sh eval
  mkdir -p "$OUT/eval" && cp results/eval/*_"$(basename "$READER2")"*.jsonl "$OUT/eval"/
  pack part4
}

PART=${1:-}
if [ "$PART" = _call ]; then shift; "$@"; exit; fi                 # one step, called by step()
[[ $PART =~ ^part[1234]$ ]] || { grep -E '^#( |$)' "$0" | head -20; exit 1; }
"$PART"
