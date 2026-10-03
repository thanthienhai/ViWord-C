#!/usr/bin/env bash
# Smoke test of training + evaluation on one Ubuntu GPU machine (H100 or any GPU >= 40 GB).
#
# Goal: check every code path that could not run on the Windows/GTX 1060 machine
# (XLM-R-large training, vLLM reader, causal-LM baselines) and get a first look at the
# trends. Small on purpose: 1 seed, 1 epoch, a subset of the distilled data, 200 Belebele
# questions and 300 ViNLI pairs. The numbers are NOT results for the paper.
#
# Data (not in git) comes from the private Hugging Face dataset repo HF_DATA_REPO, pinned to the
# commit HF_DATA_REVISION so every run uses the same data (pushed from the data-preparation
# machine with `python scripts/hf_data.py push --repo <user>/viword-c-data`).
# Needs `huggingface-cli login` (or HF_TOKEN) on this machine.
#
# Usage (from the repository root):
#   bash scripts/smoke_h100.sh            # everything
#   bash scripts/smoke_h100.sh setup      # only create the environment
#   bash scripts/smoke_h100.sh train eval # selected steps
#   bash scripts/smoke_h100.sh teacher_gate   # only gate G4 (teacher vs truncation on dev)
# Settings via environment variables, e.g. GPU=1 N_PARAGRAPHS=2000 bash scripts/smoke_h100.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
HF_DATA_REPO=${HF_DATA_REPO:-thanthienhai/viword-c-data}
HF_DATA_REVISION=${HF_DATA_REVISION:-ee91d9e2ecfc96735d8e51a9a400e512b621f279}  # data of 2026-10-01
VENV=${VENV:-.venv}
ENCODER=${ENCODER:-xlm-roberta-large}
READER=${READER:-Qwen/Qwen2.5-7B-Instruct}
LM=${LM:-Qwen/Qwen2.5-1.5B-Instruct}
N_PARAGRAPHS=${N_PARAGRAPHS:-1500}      # paragraphs used for the smoke training subset
N_BELEBELE=${N_BELEBELE:-200}
N_VINLI=${N_VINLI:-300}
RATIOS="0.5 0.333 0.2"
OUT=results/smoke
LL2=microsoft/llmlingua-2-xlm-roberta-large-meetingbank

export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
log() { echo -e "\n=== $* ($(date '+%F %T'))" | tee -a "$OUT/smoke.log"; }

step_setup() {
  log "setup: python venv in $VENV"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -q --upgrade pip
  pip install -q -e ".[model,eval,dev]"     # torch, transformers, vllm, datasets, pytest
  python -m pytest -q tests | tail -1
  python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB, bf16: {torch.cuda.is_bf16_supported()}")
EOF
}

activate() {
  # shellcheck disable=SC1091
  [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
}

step_check_data() {
  if [ -n "$HF_DATA_REPO" ]; then
    log "pull data from $HF_DATA_REPO @ $HF_DATA_REVISION"
    python scripts/hf_data.py pull --repo "$HF_DATA_REPO" --revision "$HF_DATA_REVISION"
  fi
  log "check data"
  for f in data/tasks/belebele.jsonl data/tasks/vinli_test.jsonl data/segmented/belebele.jsonl \
           data/segmented/vinli_test.jsonl data/distill/paragraphs_clean.jsonl \
           data/segmented/paragraphs_clean.jsonl data/distill/teacher.jsonl \
           data/tasks/{vinli,vimmrc,vietnews}_dev.jsonl data/segmented/{vinli,vimmrc,vietnews}_dev.jsonl \
           results/teacher_dev/{vinli,vimmrc,vietnews}_dev.jsonl; do
    [ -s "$f" ] || { echo "missing $f (set HF_DATA_REPO, see the top of this script)"; exit 1; }
    echo "$(wc -l < "$f") $f"
  done
}

step_teacher_gate() {
  log "gate G4: teacher vs truncation on 200 dev examples per task ($READER)"
  READER="$READER" READER_BACKEND=vllm bash scripts/run_pipeline.sh teacher_gate \
    2>&1 | tee -a "$OUT/teacher_gate.log" | grep -E "^(- |->|  teacher|## )|Error|Traceback" || true
  cp results/gate/g4_*.md results/gate/teacher_review_*.md "$OUT"/ 2>/dev/null || true
}

step_distill() {
  log "build labels from the teacher outputs"
  python scripts/distill.py --paragraphs data/distill/paragraphs_clean.jsonl \
    --segmented data/segmented/paragraphs_clean.jsonl --teacher data/distill/teacher.jsonl \
    --output data/distill/distilled.jsonl | tee "$OUT/distill_summary.json"
  log "subset of $N_PARAGRAPHS paragraphs for the smoke training"
  python - <<EOF
import random
from viword.data import read_jsonl, write_jsonl
records = read_jsonl("data/distill/distilled.jsonl")
ids = sorted({r["id"] for r in records})
random.Random(0).shuffle(ids)
keep = set(ids[:$N_PARAGRAPHS])
subset = [r for r in records if r["id"] in keep]
write_jsonl("data/distill/smoke_subset.jsonl", subset)
print(f"{len(subset)} records from {len(keep)} paragraphs")
EOF
}

step_train() {
  for unit in word syllable; do
    name=$([ "$unit" = word ] && echo viword || echo llmlingua2vi)
    log "train $name ($ENCODER, 1 epoch)"
    python scripts/train.py --distilled data/distill/smoke_subset.jsonl --label-unit "$unit" \
      --out "runs/smoke_${name}" --model-name "$ENCODER" --dev-paragraphs 150 --epochs 1 \
      2>&1 | tee -a "$OUT/train_${name}.log" | grep -E "records|epoch"
  done
}

step_eval() {
  log "small task files"
  head -n "$N_BELEBELE" data/tasks/belebele.jsonl > "$OUT/belebele.jsonl"
  head -n "$N_VINLI" data/tasks/vinli_test.jsonl > "$OUT/vinli.jsonl"
  methods=(none no_context lead truncation "selective_context:lm=$LM"
           "scored:model=$LL2,unit=syllable,name=llmlingua2_syl"
           "scored:model=$LL2,unit=word,name=llmlingua2_wordpool"
           "scored:model=runs/smoke_llmlingua2vi,unit=syllable,name=llmlingua2vi"
           "scored:model=runs/smoke_llmlingua2vi,unit=word,name=llmlingua2vi_wordpool"
           "scored:model=runs/smoke_viword,unit=syllable,name=viword_syl"
           "scored:model=runs/smoke_viword,unit=word,name=viword"
           "scored:model=runs/smoke_viword,unit=word,protect=soft,delta=0.2,name=viword_pi"
           "lexprior:train=data/distill/smoke_subset.jsonl,name=lexprior")
  for task in belebele vinli; do
    stem=$([ "$task" = vinli ] && echo vinli_test || echo belebele)
    extra=()
    [ "$task" = vinli ] && extra=(probe_negation)
    log "evaluate $task with $READER (vLLM)"
    rm -f "$OUT/${task}.compressed.jsonl"  # no stale compressions from an earlier smoke run
    python scripts/evaluate.py --task "$task" --data "$OUT/$task.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend vllm \
      --ratios $RATIOS --methods "${methods[@]}" "${extra[@]}" --out "$OUT/${task}_rows.jsonl" \
      2>&1 | tee -a "$OUT/eval_${task}.log" | grep -E "wrote|Error|Traceback" || true
  done
}

step_analyze() {
  log "analysis"
  python scripts/analyze.py --rows "$OUT"/belebele_rows.jsonl "$OUT"/vinli_rows.jsonl \
    --target viword --reference llmlingua2vi --n-boot 2000 | tee "$OUT/analysis.md"
  echo -e "\nSmoke test done. Send back the folder $OUT (logs, analysis.md, *_rows.jsonl)."
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(setup check_data teacher_gate distill train eval analyze)
for s in "${steps[@]}"; do
  [ "$s" = setup ] || activate
  "step_$s"
done
