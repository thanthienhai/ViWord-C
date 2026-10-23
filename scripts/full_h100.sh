#!/usr/bin/env bash
# Full ViWord-C run on one Ubuntu GPU machine (H100 80 GB), in three phases (docs/RUN_FULL_H100.md).
#
#   bash scripts/full_h100.sh phase1   # gates G1/G2/G3 on dev, labels, 3 seeds x 2 encoders, ckpt_check (~4-5 h)
#   bash scripts/full_h100.sh phase2   # dev decisions: diagnose (G1/G2/G5, D4/G6 audit), select π, MDE (~2-4 h)
#   bash scripts/full_h100.sh phase3   # test: evaluation on 7 test sets, analysis (~12-18 h)
#
# Send results/full/ back after each phase and wait for the project owner before the next one:
# a failed gate G3 changes the evaluation design, and the π selected in phase 2 must be committed
# to git by the project owner before phase 3 (stage eval refuses an uncommitted π).
# Needs the environment and data of the smoke test: run `bash scripts/smoke_h100.sh setup check_data`
# first on a fresh machine. Teacher, segmentation and teacher_test stages are NOT run here: their
# outputs come with the HF data (teacher_test needs the internal teacher endpoint). Nor is bertscore: it
# needs Java (VnCoreNLP) and runs on the project owner's machine on $OUT/eval (scripts/bertscore_local.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
VENV=${VENV:-.venv}
OUT=results/full
export CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
log() { echo -e "\n=== $* ($(date '+%F %T'))" | tee -a "$OUT/full.log"; }

# shellcheck disable=SC1091
source "$VENV/bin/activate"

collect() {  # copy reports into $OUT; missing files are skipped (also called after a failed stage)
  local f
  for f in "$@"; do [ -e "$f" ] && cp -r "$f" "$OUT"/; done
  return 0
}

# collect_<stage>: reports worth sending back even when the stage stops on a FAIL gate
collect_gate_eval() { collect results/gate/g123_*.md; }
collect_distill() { collect data/distill/distill_summary.json; }
collect_train() {
  local run
  for run in runs/viword_s* runs/llmlingua2vi_s*; do
    [ -f "$run/train_meta.json" ] && cp "$run/train_meta.json" "$OUT/$(basename "$run")_train_meta.json"
  done
  return 0
}
collect_as() {  # copy one report under a distinct name (ckpt_check and mde both write report.*)
  [ -e "$1" ] && cp "$1" "$OUT/$2"
  return 0
}
collect_ckpt_check() {
  collect_as results/ckpt_check/report.md ckpt_check_report.md
  collect_as results/ckpt_check/report.json ckpt_check_report.json
}
collect_diagnose() {
  collect results/protect_eval.md results/protect_eval.json results/probe_negation_eval.md \
    results/probe_negation_eval.json results/diagnose/belebele.json results/diagnose/vinli.json \
    results/diagnose/llmlingua2vi_*.json results/diagnose/llmlingua2_compare_*.json \
    results/diagnose/gate_report.md
}
collect_select_pi() {
  collect results/pi_selection.md results/pi_selection.stdout.md config/pi_selected.json config/pi_selected.env
}
collect_mde() {
  collect_as results/mde_dev/report.md mde_report.md
  collect_as results/mde_dev/report.json mde_report.json
}
collect_analyze() { collect results/analysis_3seed.md; }

run_stages() {  # run_pipeline.sh stages, full log in $OUT/<name>.log; stop on the first error
  local name=$1; shift
  log "$name: run_pipeline.sh $*"
  if ! bash scripts/run_pipeline.sh "$@" > "$OUT/$name.log" 2>&1; then
    log "$name FAILED, see $OUT/$name.log"
    tail -n 30 "$OUT/$name.log"
    if declare -F "collect_$name" >/dev/null; then "collect_$name"; fi
    exit 1
  fi
  if declare -F "collect_$name" >/dev/null; then "collect_$name"; fi
  log "$name done"
}

run_gate() {  # like run_stages for a gate whose FAIL is a research outcome: $1 = name, $2 = its JSON report
  # The stage deletes its old report first, so a report means the gate decided; a FAIL then blocks only the
  # test evaluation (stage_eval checks the report) and the phase goes on. No report means the stage broke: stop.
  local name=$1 report=$2; shift 2
  log "$name: run_pipeline.sh $*"
  if bash scripts/run_pipeline.sh "$@" > "$OUT/$name.log" 2>&1; then
    log "$name done"
  elif [ -s "$report" ]; then
    log "$name FAIL (gate result, report in $OUT; the test evaluation stays blocked, the phase goes on)"
    echo "$name (gate FAIL)" >> "$OUT/FAILED_STEPS"
  else
    log "$name FAILED, see $OUT/$name.log"
    tail -n 30 "$OUT/$name.log"
    if declare -F "collect_$name" >/dev/null; then "collect_$name"; fi
    exit 1
  fi
  if declare -F "collect_$name" >/dev/null; then "collect_$name"; fi
}

require_runs() {
  local s name
  for s in 0 1 2; do
    for name in viword llmlingua2vi; do
      [ -d "runs/${name}_s$s" ] || { log "missing runs/${name}_s$s: run phase1 first"; exit 1; }
    done
  done
}

phase1() {
  run_stages gate_eval gate_eval
  run_stages distill distill
  run_stages train train
  # a FAIL here is a research outcome: it blocks eval (phase 3), not the dev decisions of phase 2
  run_gate ckpt_check results/ckpt_check/report.json ckpt_check
  log "Phase 1 done. Send back $OUT and wait for the go-ahead before phase 2"
}

phase2() {
  require_runs
  # protect_eval needs the blind sheets data/annot/*_sheet.jsonl filled in and committed
  run_stages diagnose protect_eval diagnose
  run_stages select_pi select_pi
  run_stages mde mde  # FAIL here blocks test evaluation: the report is in $OUT
  log "Phase 2 done. Send back $OUT; the project owner commits config/pi_selected.* before phase 3"
}

phase3() {
  require_runs
  run_stages eval eval  # refuses to start without PASS ckpt_check/MDE, committed π and teacher_test files
  run_stages analyze analyze
  mkdir -p "$OUT/eval" && cp results/eval/*.jsonl "$OUT/eval"/
  log "Phase 3 done. Send back $OUT"
}

[ $# -eq 1 ] && [[ $1 =~ ^phase[123]$ ]] || { grep -E '^#( |$)' "$0" | head -12; exit 1; }
"$1"
