#!/usr/bin/env bash
# Full ViWord-C run on one Ubuntu GPU machine (H100 80 GB), in two phases (docs/RUN_FULL_H100.md).
#
#   bash scripts/full_h100.sh phase1   # gates G1/G2/G3 on dev, labels, 3 seeds x 2 encoders (~4-5 h)
#   bash scripts/full_h100.sh phase2   # select π, MDE, evaluation on 4 test sets + analysis (~4-6 h)
#
# Send results/full/ back after each phase; start phase 2 only when the project owner says so
# (a failed gate G3 changes the evaluation design). Needs the environment and data of the
# smoke test: run `bash scripts/smoke_h100.sh setup check_data` first on a fresh machine.
# The teacher and segmentation stages are NOT run: their outputs come with the HF data.
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

run_stages() {  # run_pipeline.sh stages, full log in $OUT/<name>.log; stop on the first error
  local name=$1; shift
  log "$name: run_pipeline.sh $*"
  if ! bash scripts/run_pipeline.sh "$@" > "$OUT/$name.log" 2>&1; then
    log "$name FAILED, see $OUT/$name.log"
    tail -n 30 "$OUT/$name.log"
    exit 1
  fi
  log "$name done"
}

phase1() {
  run_stages gate_eval gate_eval
  cp results/gate/g123_*.md "$OUT"/
  run_stages distill distill
  cp data/distill/distill_summary.json "$OUT"/
  run_stages train train
  run_stages ckpt_check ckpt_check
  for run in runs/viword_s* runs/llmlingua2vi_s*; do
    cp "$run/train_meta.json" "$OUT/$(basename "$run")_train_meta.json"
  done
  cp results/ckpt_check/report.md results/ckpt_check/report.json "$OUT"/
  log "Phase 1 done. Send back $OUT and wait for the go-ahead before phase 2"
}

phase2() {
  for s in 0 1 2; do
    for name in viword llmlingua2vi; do
      [ -d "runs/${name}_s$s" ] || { log "missing runs/${name}_s$s: run phase1 first"; exit 1; }
    done
  done
  run_stages diagnose protect_eval diagnose
  cp results/diagnose/llmlingua2vi_*.json results/diagnose/gate_report.md "$OUT"/
  run_stages select_pi select_pi
  cp results/pi_selection.md results/pi_selection.stdout.md config/pi_selected.json config/pi_selected.env "$OUT"/
  run_stages mde mde
  cp results/mde_dev/report.md results/mde_dev/report.json "$OUT"/
  run_stages teacher_test teacher_test  # test-set teacher upper bound; needs the teacher endpoint
  run_stages eval eval
  run_stages bertscore bertscore
  run_stages analyze analyze
  cp results/analysis_3seed.md "$OUT"/
  mkdir -p "$OUT/eval" && cp results/eval/*.jsonl "$OUT/eval"/
  log "Phase 2 done. Send back $OUT"
}

[ $# -eq 1 ] && [[ $1 =~ ^phase[12]$ ]] || { grep -E '^#( |$)' "$0" | head -9; exit 1; }
"$1"
