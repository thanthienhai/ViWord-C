#!/usr/bin/env bash
# Full ViWord-C pipeline, stage by stage (DATN §4–§5).
#
#   bash scripts/run_pipeline.sh data segment diagnose     # week-1 gate, no reader
#   bash scripts/run_pipeline.sh teacher_check             # gate G4 inputs (teacher on dev sets)
#   bash scripts/run_pipeline.sh teacher distill train     # distillation + 3 seeds x 2 label units
#   bash scripts/run_pipeline.sh eval analyze              # needs a reader (vLLM, API or hf)
#
# Every stage reads the previous stage's files, so stages can be run on different machines.
# Settings can be overridden from the environment, e.g. SEGMENTER=underthesea bash ...
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONIOENCODING=utf-8

# ---------------------------------------------------------------------------- settings
TEACHER_URL=${TEACHER_URL:-http://172.16.9.11:30048/v1}
TEACHER_MODEL=${TEACHER_MODEL:-icmodel/icom-model-llm-ic-v3.8-27b}
TEACHER_WORKERS=${TEACHER_WORKERS:-16}
SEGMENTER=${SEGMENTER:-vncorenlp}          # underthesea if the VnCoreNLP download is too slow
SEG_WORKERS=${SEG_WORKERS:-16}              # parallel segmenter processes
READER=${READER:-Qwen/Qwen2.5-7B-Instruct}
READER_BACKEND=${READER_BACKEND:-vllm}     # vllm | api | hf
READER_URL=${READER_URL:-}                 # for READER_BACKEND=api
READER_TOKENIZER=${READER_TOKENIZER:-$READER}
LM=${LM:-Qwen/Qwen2.5-1.5B-Instruct}       # small causal LM for ppl_sent / selective_context
LIMIT=${LIMIT:-}                           # e.g. LIMIT=500 for the gate evaluation
SEEDS=${SEEDS:-"0 1 2"}
RATIOS="0.5 0.333 0.2"
LL2=microsoft/llmlingua-2-xlm-roberta-large-meetingbank

# task name : file stem : field that is compressed
TEST_SETS="belebele:belebele:context vinli:vinli_test:premise vimmrc:vimmrc_test:context vietnews:vietnews_test:context"
DEV_SETS="vinli:vinli_dev:premise vimmrc:vimmrc_dev:context vietnews:vietnews_dev:context"
COST_TOKENIZERS="Qwen/Qwen2.5-7B-Instruct SeaLLMs/SeaLLMs-v3-7B-Chat unsloth/Llama-3.1-8B-Instruct unsloth/gemma-2-9b-it unsloth/gemma-3-12b-it"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

log() { echo -e "\n=== $* ($(date '+%F %T'))"; }

# ---------------------------------------------------------------------------- stages
stage_data() {
  for d in belebele vinli vimmrc vietnews paragraphs; do
    log "prepare $d"; python scripts/prepare_data.py "$d"
  done
  log "leak filter"
  eval_files=()  # evaluation and dev sets only (not vinli_train)
  for spec in $TEST_SETS $DEV_SETS; do
    IFS=: read -r _ stem _ <<< "$spec"; eval_files+=("data/tasks/$stem.jsonl")
  done
  python scripts/leak_filter.py --input data/distill/paragraphs.jsonl \
    --eval-files "${eval_files[@]}" --output data/distill/paragraphs_clean.jsonl
}

stage_segment() {
  for spec in $TEST_SETS $DEV_SETS; do
    IFS=: read -r _ stem field <<< "$spec"
    log "segment $stem"
    python scripts/segment_data.py --input "data/tasks/$stem.jsonl" --field "$field" \
      --output "data/segmented/$stem.jsonl" --segmenter "$SEGMENTER" --workers "$SEG_WORKERS"
  done
  log "segment distillation paragraphs"
  python scripts/segment_data.py --input data/distill/paragraphs_clean.jsonl --field text \
    --output data/segmented/paragraphs_clean.jsonl --segmenter "$SEGMENTER" --workers "$SEG_WORKERS"
}

stage_diagnose() {  # DATN §4.0, gates G1 (CBR), G2 (count), G5
  for task in belebele vinli; do
    stem=$([ "$task" = vinli ] && echo vinli_test || echo belebele)
    extra=$([ "$task" = vinli ] && echo probe_negation || true)
    log "diagnose $task"
    python scripts/diagnose.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --unique-clusters \
      --budget-tokenizer "$READER_TOKENIZER" --cost-tokenizers $COST_TOKENIZERS \
      --methods random lead truncation stopword lexical tfidf_sent \
        "ppl_sent:lm=$LM" "selective_context:lm=$LM" \
        "scored:model=$LL2,unit=syllable,name=llmlingua2_syl" \
        "scored:model=$LL2,unit=word,name=llmlingua2_wordpool" $extra \
      --rr-model xlm-roberta-base --out "results/diagnose/$task.json"
  done
  python scripts/gate_report.py --belebele results/diagnose/belebele.json \
    --vinli results/diagnose/vinli.json --vinli-pairs data/tasks/vinli_test.jsonl \
    --vinli-segmented data/segmented/vinli_test.jsonl --vinli-premises data/tasks/vinli_test.jsonl \
    | tee results/diagnose/gate_report.md
}

stage_teacher_check() {  # gate G4: teacher compressions of dev contexts -> `precomputed:` method
  for spec in $DEV_SETS; do
    IFS=: read -r _ stem field <<< "$spec"
    log "teacher on $stem (200)"
    python scripts/teacher_compress.py --input "data/tasks/$stem.jsonl" --field "$field" --limit 200 \
      --rates 0.5 0.333 0.2 --output "results/teacher_dev/$stem.jsonl" \
      --base-url "$TEACHER_URL" --model "$TEACHER_MODEL" --workers "$TEACHER_WORKERS"
  done
}

stage_teacher() {
  log "teacher on distillation paragraphs"
  python scripts/teacher_compress.py --input data/distill/paragraphs_clean.jsonl --field text \
    --rates 0.5 0.25 --output data/distill/teacher.jsonl \
    --base-url "$TEACHER_URL" --model "$TEACHER_MODEL" --workers "$TEACHER_WORKERS"
}

stage_distill() {
  log "build labels"
  python scripts/distill.py --paragraphs data/distill/paragraphs_clean.jsonl \
    --segmented data/segmented/paragraphs_clean.jsonl --teacher data/distill/teacher.jsonl \
    --output data/distill/distilled.jsonl | tee data/distill/distill_summary.json
}

stage_train() {
  for s in $SEEDS; do
    for unit in word syllable; do
      name=$([ "$unit" = word ] && echo viword || echo llmlingua2vi)
      log "train $name seed $s"
      python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit "$unit" \
        --out "runs/${name}_s$s" --seed "$s"
    done
  done
}

stage_eval() {
  methods=(none no_context random lead truncation tfidf_sent "ppl_sent:lm=$LM" "selective_context:lm=$LM"
           "scored:model=$LL2,unit=syllable,name=llmlingua2_syl"
           "scored:model=$LL2,unit=word,name=llmlingua2_wordpool"
           "lexprior:train=data/distill/distilled.jsonl,name=lexprior")
  for s in $SEEDS; do
    methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=word,name=llmlingua2vi_wordpool_s$s"
              "scored:model=runs/viword_s$s,unit=syllable,name=viword_syl_s$s"
              "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=hard,name=llmlingua2vi_force_s$s"
              "scored:model=runs/viword_s$s,unit=word,protect=soft,delta=${DELTA:-0.2},tiers=${TIERS:-T1+T2+T3},name=viword_pi_s$s")
  done
  reader_tag=$(basename "$READER")
  for spec in $TEST_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    task_methods=("${methods[@]}")
    [ "$task" = belebele ] && task_methods+=("translate:inner=llmlingua2")
    [ "$task" = vinli ] && task_methods+=(probe_negation)
    log "evaluate $task with $READER"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS $LIMIT_ARG \
      --methods "${task_methods[@]}" --out "results/eval/${task}_${reader_tag}.jsonl"
  done
}

stage_analyze() {
  for s in $SEEDS; do
    log "analyze seed $s"
    python scripts/analyze.py --rows results/eval/*.jsonl --target "viword_s$s" --reference "llmlingua2vi_s$s" \
      | tee "results/analysis_seed$s.md"
  done
}

[ $# -gt 0 ] || { grep -E '^#( |$)' "$0" | head -8; exit 1; }
for stage in "$@"; do "stage_$stage"; done
