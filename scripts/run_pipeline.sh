#!/usr/bin/env bash
# Full ViWord-C pipeline, stage by stage (DATN §4–§5).
#
#   bash scripts/run_pipeline.sh data segment annot_sheets  # then annotate data/annot/*_sheet.jsonl by hand
#   bash scripts/run_pipeline.sh protect_eval diagnose     # week-1 gate, no reader (needs the filled sheets)
#   bash scripts/run_pipeline.sh teacher_check             # gate G4 inputs (teacher on dev sets)
#   bash scripts/run_pipeline.sh teacher_gate gate_eval    # needs a reader: G4, then G1/G2/G3 on dev
#   bash scripts/run_pipeline.sh teacher distill train ckpt_check select_pi mde  # dev power check before eval
#   bash scripts/run_pipeline.sh protect_eval diagnose     # after train, also measures CBR for llmlingua2vi_s*
#   bash scripts/run_pipeline.sh teacher_test eval bertscore analyze # teacher upper bound, evaluation
#
# Every stage reads the previous stage's files, so stages can be run on different machines.
# Settings can be overridden from the environment, except SEGMENTER: the main
# pipeline is intentionally fixed to VnCoreNLP.  Use scripts/segment_data.py
# directly for the §4.4 segmenter-sensitivity analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONIOENCODING=utf-8

# ---------------------------------------------------------------------------- settings
# TEACHER_URL/TEACHER_MODEL from .env unless already set; read, not sourced, so .env runs nothing.
# The API keys stay in .env and are read by scripts/teacher_compress.py, not passed on the command line.
if [ -f .env ]; then
  while IFS='=' read -r name value; do
    case "$name" in TEACHER_URL|TEACHER_MODEL|TEACHER_WORKERS|TEACHER_MAX_TOKENS)
      [ -z "${!name:-}" ] && printf -v "$name" '%s' "${value%$'\r'}" ;;
    esac
  done < .env
fi
TEACHER_URL=${TEACHER_URL:-http://172.16.9.11:30049/v1}
TEACHER_MODEL=${TEACHER_MODEL:-icmodel/icom-model-llm-ic-v3.8-27b}
TEACHER_WORKERS=${TEACHER_WORKERS:-16}
# output cap per teacher call; a call that reaches it fails instead of keeping a cut-off text
# (the OneRuler cwe word lists come back almost uncompressed, ~4.8k tokens)
TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-8192}
SEGMENTER=${SEGMENTER:-vncorenlp}
SEG_WORKERS=${SEG_WORKERS:-16}              # parallel segmenter processes
READER=${READER:-Qwen/Qwen2.5-7B-Instruct}
# DATN §3.5/§4.1: gates, ckpt_check, π and α selection and the MDE use the primary reader only;
# their choices are then fixed for every reader. Run the second reader with e.g.
#   READER=SeaLLMs/SeaLLM-7B-v2 bash scripts/run_pipeline.sh eval
# (check its chat template first with scripts/check_reader_template.py).
PRIMARY_READER=${PRIMARY_READER:-Qwen/Qwen2.5-7B-Instruct}
export READER_SYSTEM_PROMPT=${READER_SYSTEM_PROMPT:-}
READER_BACKEND=${READER_BACKEND:-vllm}     # vllm | api | hf
READER_URL=${READER_URL:-}                 # for READER_BACKEND=api
READER_TOKENIZER=${READER_TOKENIZER:-$READER}
# blind sheets from `annot_sheets`, filled in by a human annotator (DATN §3.4, G6)
PROTECT_LABELS=${PROTECT_LABELS:-data/annot/protect_sheet.jsonl}
PROBE_LABELS=${PROBE_LABELS:-data/annot/probe_sheet.jsonl}
PROTECT_REPORT=${PROTECT_REPORT:-results/protect_eval.json}
PROBE_REPORT=${PROBE_REPORT:-results/probe_negation_eval.json}
PI_SELECTED_ENV=${PI_SELECTED_ENV:-config/pi_selected.env}
PI_SELECTED_JSON=${PI_SELECTED_JSON:-config/pi_selected.json}
PI_DEV_LIMIT=${PI_DEV_LIMIT:-500}
ALPHA_SELECTED_ENV=${ALPHA_SELECTED_ENV:-config/alpha_selected.env}   # H3, 8-page version only
ALPHA_SELECTED_JSON=${ALPHA_SELECTED_JSON:-config/alpha_selected.json}
ALPHA_GRID=${ALPHA_GRID:-"0.25 0.5 0.75 1"}
MDE_LIMIT=${MDE_LIMIT:-500}
MDE_BOOT=${MDE_BOOT:-2000}
MDE_MARGIN=${MDE_MARGIN:-0.5}
MDE_JSON=${MDE_JSON:-results/mde_dev/report.json}
MDE_REPORT=${MDE_REPORT:-results/mde_dev/report.md}
CKPT_CHECK_LIMIT=${CKPT_CHECK_LIMIT:-200}
CKPT_CHECK_RATIO=${CKPT_CHECK_RATIO:-0.333}
CKPT_CHECK_REPORT=${CKPT_CHECK_REPORT:-results/ckpt_check/report.md}
CKPT_CHECK_JSON=${CKPT_CHECK_JSON:-results/ckpt_check/report.json}
PI_GRID_TIERS=${PI_GRID_TIERS:-"T1 T1+T2 T1+T2+T3"}
PI_GRID_DELTAS=${PI_GRID_DELTAS:-"0.1 0.2 0.3 0.5"}
TEACHER_TEST_LIMIT=${TEACHER_TEST_LIMIT:-300}
# test rows readers scored during development (DATN §4.3, amendment of 2026-10-22): out of the primary
# analysis and of the teacher upper bound; the Belebele ones form belebele_dev (scripts/test_exclusions.py)
TEST_EXCLUSIONS=${TEST_EXCLUSIONS:-config/test_exclusions.json}
LL2_COMPARE_LIMIT=${LL2_COMPARE_LIMIT:-100}
LL2_COMPARE_RATIO=${LL2_COMPARE_RATIO:-0.333}
LM=${LM:-Qwen/Qwen2.5-1.5B-Instruct}       # small causal LM for ppl_sent / selective_context
LIMIT=${LIMIT:-}                           # cap on test examples per task (gates use GATE_LIMIT on dev)
SEEDS=${SEEDS:-"0 1 2"}
PI_SEEDS=${PI_SEEDS:-$SEEDS}
RATIOS="0.5 0.333 0.2"
LL2=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
RR_MODEL=${RR_MODEL:-xlm-roberta-large}        # DATN §2.2/§4.0 RR MLM

# task name : file stem : field that is compressed
TEST_SETS="belebele:belebele:context vinli:vinli_test:premise vimmrc:vimmrc_test:context vietnews:vietnews_test:context
           viquad:viquad_test:context gsm8k:gsm8k_test:context oneruler:oneruler_test:context"
# the decision gates (G1-G4) use only these three dev sets, as preregistered in DATN §2.3
DEV_SETS="vinli:vinli_dev:premise vimmrc:vimmrc_dev:context vietnews:vietnews_dev:context"
# dev sets of the tasks added on 2026-10-04: segmented and leak-checked, not used by the gates
EXTRA_DEV_SETS="viquad:viquad_dev:context gsm8k:gsm8k_dev:context oneruler:oneruler_dev:context"
# DATN §2.2: Qwen2.5, Llama-3.1, Gemma-2, SeaLLM-v3 and SeaLLM-7B-v2 (the second reader); Gemma-3 is the 8-page reader
COST_TOKENIZERS="Qwen/Qwen2.5-7B-Instruct unsloth/Llama-3.1-8B-Instruct unsloth/gemma-2-9b-it SeaLLMs/SeaLLMs-v3-7B-Chat SeaLLMs/SeaLLM-7B-v2 unsloth/gemma-3-12b-it"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

if [ "$SEGMENTER" != "vncorenlp" ]; then
  echo "SEGMENTER=$SEGMENTER is not allowed for the main pipeline; use vncorenlp." >&2
  echo "Run scripts/segment_data.py directly for underthesea/pyvi sensitivity experiments." >&2
  exit 2
fi

log() { echo -e "\n=== $* ($(date '+%F %T'))"; }

require_segmented() {  # every stage that reads data/segmented checks provenance first (A2)
  python scripts/check_segmented.py --directory data/segmented --segmenter vncorenlp
}

require_annotations() {  # P detector (D4) and negation probe (G6) audited on blind sheets
  python scripts/evaluate_protect.py --verify "$PROTECT_REPORT"
  [ -s "$PROBE_REPORT" ] || {
    echo "missing $PROBE_REPORT; run: bash scripts/run_pipeline.sh protect_eval" >&2
    exit 1
  }
  python -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if r.get('sheet')=='blind-v1' else 'probe audit was not computed from a blind sheet: '+sys.argv[1])" "$PROBE_REPORT"
}

require_primary_reader() {  # dev decisions are made once, with the primary reader
  [ "$READER" = "$PRIMARY_READER" ] || {
    echo "this stage selects or gates on dev and must run with READER=$PRIMARY_READER (got $READER)" >&2
    exit 1
  }
}

lexprior_spec() {  # "đối chứng tra từ", fit on the encoder's train split only (dev ids excluded)
  echo "lexprior:train=data/distill/distilled.jsonl,exclude=runs/viword_s0/dev_ids.jsonl,name=lexprior"
}

require_committed_pi() {
  git ls-files --error-unmatch "$PI_SELECTED_JSON" "$PI_SELECTED_ENV" >/dev/null 2>&1 || {
    echo "π config must be tracked before eval: $PI_SELECTED_JSON and $PI_SELECTED_ENV" >&2
    exit 1
  }
  git diff --quiet -- "$PI_SELECTED_JSON" "$PI_SELECTED_ENV" || {
    echo "π config has uncommitted changes; commit it before eval" >&2
    exit 1
  }
  git diff --cached --quiet -- "$PI_SELECTED_JSON" "$PI_SELECTED_ENV" || {
    echo "π config is staged but not committed; commit it before eval" >&2
    exit 1
  }
}

require_committed_exclusions() {  # the rows left out of the primary analysis are fixed before the test run
  git ls-files --error-unmatch "$TEST_EXCLUSIONS" >/dev/null 2>&1 && git diff --quiet HEAD -- "$TEST_EXCLUSIONS" || {
    echo "$TEST_EXCLUSIONS must be committed, unchanged, before eval" >&2
    exit 1
  }
  python scripts/test_exclusions.py check --config "$TEST_EXCLUSIONS"
}

# ---------------------------------------------------------------------------- stages
stage_data() {
  for d in belebele vinli vimmrc vietnews viquad gsm8k oneruler paragraphs; do
    log "prepare $d"; python scripts/prepare_data.py "$d"
  done
  log "leak filter"
  eval_files=()  # evaluation and dev sets only (not vinli_train)
  for spec in $TEST_SETS $DEV_SETS $EXTRA_DEV_SETS; do
    IFS=: read -r _ stem _ <<< "$spec"; eval_files+=("data/tasks/$stem.jsonl")
  done
  for path in "${eval_files[@]}"; do
    [ -s "$path" ] || { echo "missing evaluation file for leak index: $path" >&2; exit 1; }
  done
  # leak_filter.py also validates the required ViNLI/VietNews/ViMMRC/Belebele
  # coverage and removes stable VietNews train/eval id collisions.
  python scripts/leak_filter.py --input data/distill/paragraphs.jsonl \
    --eval-files "${eval_files[@]}" --output data/distill/paragraphs_clean.jsonl
}

stage_segment() {
  for spec in $TEST_SETS $DEV_SETS $EXTRA_DEV_SETS; do
    IFS=: read -r _ stem field <<< "$spec"
    log "segment $stem"
    python scripts/segment_data.py --input "data/tasks/$stem.jsonl" --field "$field" \
      --output "data/segmented/$stem.jsonl" --segmenter "$SEGMENTER" --workers "$SEG_WORKERS"
  done
  log "segment distillation paragraphs"
  python scripts/segment_data.py --input data/distill/paragraphs_clean.jsonl --field text \
    --output data/segmented/paragraphs_clean.jsonl --segmenter "$SEGMENTER" --workers "$SEG_WORKERS"
  stage_belebele_dev
  python scripts/check_segmented.py --directory data/segmented --segmenter vncorenlp
}

stage_belebele_dev() {  # Belebele has no dev split: the questions readers scored during development
  python scripts/test_exclusions.py belebele-dev
}

stage_annot_sheets() {  # blind annotation sheets for D4 and G6; a human fills them in
  require_segmented
  python scripts/make_annotation_sheets.py --segmented-dir data/segmented --tasks-dir data/tasks \
    --protect-out "$PROTECT_LABELS" --probe-out "$PROBE_LABELS"
}

stage_protect_eval() {  # DATN §3.4: measure P before selecting π
  require_segmented
  log "evaluate linguistic protection detector"
  python scripts/evaluate_protect.py --labels "$PROTECT_LABELS" --segmented-dir data/segmented \
    --out results/protect_eval.md --json-out "$PROTECT_REPORT"
  log "audit negation probe"
  python scripts/evaluate_probe.py --labels "$PROBE_LABELS" --segmented-dir data/segmented \
    --out results/probe_negation_eval.md --json-out "$PROBE_REPORT"
}

stage_diagnose() {  # DATN §4.0, gates G1 (CBR), G2 (count), G5
  require_segmented
  require_annotations  # checked first: the probe report is needed by gate_report after the GPU runs
  for task in belebele vinli; do
    stem=$([ "$task" = vinli ] && echo vinli_test || echo belebele)
    extra=$([ "$task" = vinli ] && echo probe_negation || true)
    log "diagnose $task"
    python scripts/diagnose.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --unique-clusters \
      --budget-tokenizer "$READER_TOKENIZER" --cost-tokenizers $COST_TOKENIZERS \
      --methods random lead truncation stopword lexical tfidf_sent \
        "ppl_sent:lm=$LM" "llmlingua:lm=$LM" "selective_context:lm=$LM" llmlingua2 \
        "scored:model=$LL2,unit=syllable,name=llmlingua2_syl" \
        "scored:model=$LL2,unit=word,name=llmlingua2_wordpool" $extra \
      --rr-model "$RR_MODEL" --out "results/diagnose/$task.json"
  done
  compare_rows=()
  for task in belebele vinli; do
    stem=$([ "$task" = vinli ] && echo vinli_test || echo belebele)
    python scripts/compare_llmlingua2.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --budget-tokenizer "$READER_TOKENIZER" \
      --limit "$LL2_COMPARE_LIMIT" --ratios "$LL2_COMPARE_RATIO" \
      --out "results/diagnose/llmlingua2_compare_$task.json"
    compare_rows+=("results/diagnose/llmlingua2_compare_$task.json")
  done
  vi_methods=()
  vi_found=0
  vi_missing=0
  for s in $SEEDS; do
    if [ -d "runs/llmlingua2vi_s$s" ]; then
      vi_found=$((vi_found + 1))
      vi_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s")
    else
      vi_missing=$((vi_missing + 1))
    fi
  done
  if [ "$vi_found" -gt 0 ] && [ "$vi_missing" -gt 0 ]; then
    echo "incomplete LLMLingua-2-vi seed set; found $vi_found, missing $vi_missing" >&2
    exit 1
  fi
  vi_report_args=()
  if [ "${#vi_methods[@]}" -gt 0 ]; then
    vi_reports=()
    for task in belebele vinli; do
      stem=$([ "$task" = vinli ] && echo vinli_test || echo belebele)
      log "diagnose post-training LLMLingua-2-vi $task"
      out="results/diagnose/llmlingua2vi_$task.json"
      python scripts/diagnose.py --task "$task" --data "data/tasks/$stem.jsonl" \
        --segmented "data/segmented/$stem.jsonl" --unique-clusters \
        --budget-tokenizer "$READER_TOKENIZER" --cost-tokenizers $COST_TOKENIZERS \
        --methods "${vi_methods[@]}" --rr-model "$RR_MODEL" --out "$out"
      vi_reports+=("$out")
    done
    vi_report_args=(--llmlingua2vi-reports "${vi_reports[@]}")
  fi
  python scripts/gate_report.py --belebele results/diagnose/belebele.json \
    --vinli results/diagnose/vinli.json --vinli-pairs data/tasks/vinli_test.jsonl \
    --vinli-segmented data/segmented/vinli_test.jsonl --vinli-premises data/tasks/vinli_test.jsonl \
    --probe-report "$PROBE_REPORT" \
    --llmlingua2-compare "${compare_rows[@]}" \
    ${vi_report_args[@]+"${vi_report_args[@]}"} \
    | tee results/diagnose/gate_report.md
}

stage_teacher_check() {  # gate G4: teacher compressions of dev contexts -> `precomputed:` method
  for spec in $DEV_SETS; do
    IFS=: read -r _ stem field <<< "$spec"
    log "teacher on $stem (200)"
    python scripts/teacher_compress.py --input "data/tasks/$stem.jsonl" --field "$field" --limit 200 \
      --rates 0.5 0.333 0.2 --output "results/teacher_dev/$stem.jsonl" \
      --base-url "$TEACHER_URL" --model "$TEACHER_MODEL" --workers "$TEACHER_WORKERS" --max-tokens "$TEACHER_MAX_TOKENS"
  done
}

stage_teacher_gate() {  # gate G4 decision: teacher vs truncation on the same 200 dev examples
  require_segmented
  reader_tag=$(basename "$READER")
  mkdir -p results/gate
  rows=()
  for spec in $DEV_SETS; do
    IFS=: read -r task stem field <<< "$spec"
    log "review teacher on $stem"
    python scripts/review_teacher.py --teacher "results/teacher_dev/$stem.jsonl" --data "data/tasks/$stem.jsonl" \
      --field "$field" --segmented "data/segmented/$stem.jsonl" --tokenizer "$READER_TOKENIZER" \
      | tee "results/gate/teacher_review_$stem.md"
    log "G4: evaluate teacher on $stem with $READER"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS --limit 200 \
      --methods none no_context truncation lead \
        "precomputed:path=results/teacher_dev/$stem.jsonl,trim=1,name=teacher" \
        "precomputed:path=results/teacher_dev/$stem.jsonl,name=teacher_raw" \
      --out "results/gate/teacher_${stem}_${reader_tag}.jsonl"
    rows+=("results/gate/teacher_${stem}_${reader_tag}.jsonl")
  done
  python scripts/gate_eval.py --split dev --rows "${rows[@]}" --teacher-model "$TEACHER_MODEL" \
    | tee "results/gate/g4_${reader_tag}.md"
}

stage_gate_eval() {  # reader parts of G1, G2, G3 on the DEV sets (no gate decision looks at test)
  require_segmented
  reader_tag=$(basename "$READER")
  mkdir -p results/gate
  rows=()
  for spec in $DEV_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    extra=$([ "$task" = vinli ] && echo probe_negation || true)
    log "gate evaluation on $stem with $READER"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --limit "${GATE_LIMIT:-500}" \
      --methods none no_context lead truncation tfidf_sent "ppl_sent:lm=$LM" "selective_context:lm=$LM" \
        "llmlingua:lm=$LM" lexical stopword \
        "scored:model=$LL2,unit=syllable,name=llmlingua2_syl" \
        "scored:model=$LL2,unit=word,name=llmlingua2_wordpool" llmlingua2 $extra \
      --out "results/gate/dev_${stem}_${reader_tag}.jsonl"
    rows+=("results/gate/dev_${stem}_${reader_tag}.jsonl")
  done
  python scripts/gate_eval.py --split dev --rows "${rows[@]}" | tee "results/gate/g123_${reader_tag}.md"
}

stage_teacher() {
  log "teacher on distillation paragraphs"
  python scripts/teacher_compress.py --input data/distill/paragraphs_clean.jsonl --field text \
    --rates 0.5 0.25 --output data/distill/teacher.jsonl \
    --base-url "$TEACHER_URL" --model "$TEACHER_MODEL" --workers "$TEACHER_WORKERS" --max-tokens "$TEACHER_MAX_TOKENS"
}

stage_distill() {
  require_segmented
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

stage_ckpt_check() {  # DATN §3.3/C3: reader sanity gate on 200 DEV examples
  require_primary_reader
  require_segmented
  mkdir -p results/ckpt_check
  # Never let a PASS report from an older checkpoint set unlock eval.
  rm -f "$CKPT_CHECK_REPORT" "$CKPT_CHECK_JSON"
  checkpoint_methods=()
  for s in $SEEDS; do
    for name in viword llmlingua2vi; do
      checkpoint="runs/${name}_s$s"
      [ -d "$checkpoint" ] || {
        echo "missing $checkpoint; run: bash scripts/run_pipeline.sh train before ckpt_check" >&2
        exit 1
      }
      unit=$([ "$name" = viword ] && echo word || echo syllable)
      checkpoint_methods+=("scored:model=$checkpoint,unit=$unit,name=${name}_s$s")
    done
  done
  reader_tag=$(basename "$READER")
  rows=()
  for spec in $DEV_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    log "checkpoint sanity check on $stem with $READER"
    out="results/ckpt_check/${stem}_${reader_tag}.jsonl"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" \
      --ratios "$CKPT_CHECK_RATIO" --limit "$CKPT_CHECK_LIMIT" \
      --methods truncation "$(lexprior_spec)" "${checkpoint_methods[@]}" --out "$out"
    rows+=("$out")
  done
  python scripts/checkpoint_check.py --rows "${rows[@]}" --ratio "$CKPT_CHECK_RATIO" \
    --expected-per-task "$CKPT_CHECK_LIMIT" --reader "$READER" \
    --out "$CKPT_CHECK_REPORT" --json-out "$CKPT_CHECK_JSON" \
    | tee results/ckpt_check/stdout.md
}

stage_select_pi() {  # DATN §2.3/H2: select π on dev, never on test
  require_primary_reader
  require_segmented
  require_annotations
  reader_tag=$(basename "$READER")
  mkdir -p results/pi_dev config
  pi_methods=(none)
  for s in $PI_SEEDS; do
    pi_methods+=("scored:model=runs/viword_s$s,unit=word,name=viword_s$s")
    for tier in $PI_GRID_TIERS; do
      tier_tag=${tier//+/_}
      for mode in soft hard; do
        deltas="$PI_GRID_DELTAS"
        [ "$mode" = hard ] && deltas="0"
        for delta in $deltas; do
          delta_tag=${delta//./p}
          name="viword_pi_${mode}_${tier_tag}_d${delta_tag}_s${s}"
          pi_methods+=("scored:model=runs/viword_s$s,unit=word,protect=$mode,delta=$delta,tiers=$tier,name=$name")
        done
      done
    done
  done
  pi_rows=()
  for spec in $DEV_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    log "select π on $stem with $READER"
    out="results/pi_dev/${stem}_${reader_tag}.jsonl"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --limit "$PI_DEV_LIMIT" --methods "${pi_methods[@]}" --out "$out"
    pi_rows+=("$out")
  done
  python scripts/select_pi.py --rows "${pi_rows[@]}" --out "$PI_SELECTED_JSON" \
    --report results/pi_selection.md --env "$PI_SELECTED_ENV" | tee results/pi_selection.stdout.md
}

stage_mde() {  # DATN §4.3/J4: estimate power on DEV before any test evaluation
  require_primary_reader
  [ -s "$PI_SELECTED_ENV" ] || {
    echo "missing $PI_SELECTED_ENV; run: bash scripts/run_pipeline.sh select_pi before mde" >&2
    exit 1
  }
  # shellcheck disable=SC1090
  . "$PI_SELECTED_ENV"
  case "${PI_MODE:-}" in soft|hard) ;; *) echo "invalid PI_MODE in $PI_SELECTED_ENV" >&2; exit 1 ;; esac
  : "${PI_TIERS:?missing PI_TIERS in $PI_SELECTED_ENV}"
  : "${PI_DELTA:?missing PI_DELTA in $PI_SELECTED_ENV}"
  require_segmented
  reader_tag=$(basename "$READER")
  mkdir -p results/mde_dev
  mde_methods=(none)
  for s in $SEEDS; do
    mde_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                  "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
                  "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
  done
  rows=()
  for spec in $DEV_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    log "MDE dev pilot on $stem with $READER"
    out="results/mde_dev/${stem}_${reader_tag}.jsonl"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --limit "$MDE_LIMIT" --methods "${mde_methods[@]}" --out "$out"
    rows+=("$out")
  done
  python scripts/mde.py --rows "${rows[@]}" --reader "$READER" --seeds $SEEDS \
    --n-boot "$MDE_BOOT" --margin-points "$MDE_MARGIN" --out "$MDE_JSON" --report "$MDE_REPORT" \
    --test-data vimmrc=data/tasks/vimmrc_test.jsonl vietnews=data/tasks/vietnews_test.jsonl \
      vinli=data/tasks/vinli_test.jsonl --exclude "$TEST_EXCLUSIONS" \
    | tee results/mde_dev/stdout.md
}

stage_select_alpha() {  # DATN §3.5/H3 (8-page): select α on VietNews + ViNLI dev, after select_pi
  require_primary_reader
  require_segmented
  [ -s "$PI_SELECTED_ENV" ] || { echo "run select_pi first" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$PI_SELECTED_ENV"
  reader_tag=$(basename "$READER")
  mkdir -p results/alpha_dev config
  alpha_methods=(none)
  for s in $SEEDS; do
    base="scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS"
    alpha_methods+=("$base,name=viword_pi_s$s")
    for cost in tokens syllables; do
      for a in $ALPHA_GRID; do
        alpha_methods+=("$base,alpha=$a,cost=$cost,name=viword_h3_a${a//./p}_c${cost}_s$s")
      done
    done
  done
  rows=()
  for spec in vinli:vinli_dev vietnews:vietnews_dev; do
    IFS=: read -r task stem <<< "$spec"
    log "select α on $stem with $READER"
    out="results/alpha_dev/${stem}_${reader_tag}.jsonl"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --limit "$PI_DEV_LIMIT" --methods "${alpha_methods[@]}" --out "$out"
    rows+=("$out")
  done
  python scripts/select_alpha.py --rows "${rows[@]}" --out "$ALPHA_SELECTED_JSON" --env "$ALPHA_SELECTED_ENV" \
    --report results/alpha_selection.md
}

stage_teacher_test() {  # DATN §4.2: teacher upper bound on a held-out test subset; needs TEACHER_URL,
  # so it runs on the owner side and ships with the HF data (scripts/hf_data.py push).
  # The subset is the first TEACHER_TEST_LIMIT rows that no reader scored during development; rows of an
  # earlier subset stay in the file (resume), stage_eval reads only the current one
  mkdir -p results/teacher_test
  for spec in $TEST_SETS; do
    IFS=: read -r _ stem field <<< "$spec"
    log "teacher upper bound on $stem ($TEACHER_TEST_LIMIT)"
    python scripts/teacher_compress.py --input "data/tasks/$stem.jsonl" --field "$field" \
      --exclude "$TEST_EXCLUSIONS" --limit "$TEACHER_TEST_LIMIT" --rates $RATIOS \
      --output "results/teacher_test/$stem.jsonl" \
      --base-url "$TEACHER_URL" --model "$TEACHER_MODEL" --workers "$TEACHER_WORKERS" --max-tokens "$TEACHER_MAX_TOKENS"
  done
}

stage_eval() {
  [ -s "$CKPT_CHECK_JSON" ] || {
    echo "missing $CKPT_CHECK_JSON; run: bash scripts/run_pipeline.sh ckpt_check after train" >&2
    exit 1
  }
  # dev-side checks were made with the primary reader; a second reader reuses them unchanged
  python scripts/checkpoint_check.py --verify "$CKPT_CHECK_JSON" --reader "$PRIMARY_READER"
  [ -s "$MDE_JSON" ] || {
    echo "missing $MDE_JSON; run: bash scripts/run_pipeline.sh mde after select_pi" >&2
    exit 1
  }
  require_committed_exclusions
  python scripts/mde.py --verify "$MDE_JSON" --margin-points "$MDE_MARGIN" --reader "$PRIMARY_READER" \
    --exclude "$TEST_EXCLUSIONS"
  require_segmented
  require_annotations
  [ -s "$PI_SELECTED_JSON" ] || {
    echo "missing $PI_SELECTED_JSON; run: bash scripts/run_pipeline.sh select_pi after train" >&2
    exit 1
  }
  [ -s "$PI_SELECTED_ENV" ] || {
    echo "missing $PI_SELECTED_ENV; run: bash scripts/run_pipeline.sh select_pi after train" >&2
    exit 1
  }
  # shellcheck disable=SC1090
  . "$PI_SELECTED_ENV"
  case "${PI_MODE:-}" in soft|hard) ;; *) echo "invalid PI_MODE in $PI_SELECTED_ENV" >&2; exit 1 ;; esac
  : "${PI_TIERS:?missing PI_TIERS in $PI_SELECTED_ENV}"
  : "${PI_DELTA:?missing PI_DELTA in $PI_SELECTED_ENV}"
  require_committed_pi
  methods=(none no_context "random:seed=0" "random:seed=1" "random:seed=2" lead truncation stopword lexical tfidf_sent
           "ppl_sent:lm=$LM" "llmlingua:lm=$LM" "selective_context:lm=$LM" llmlingua2
           "scored:model=$LL2,unit=syllable,name=llmlingua2_syl"
           "scored:model=$LL2,unit=word,name=llmlingua2_wordpool"
           "$(lexprior_spec)")
  for s in $SEEDS; do
    methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=word,name=llmlingua2vi_wordpool_s$s"
              "scored:model=runs/viword_s$s,unit=syllable,name=viword_syl_s$s"
              "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force,tiers=$PI_TIERS,name=llmlingua2vi_force_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force_ctx,tiers=$PI_TIERS,name=llmlingua2vi_forcectx_s$s"
              "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
  done
  # H3 rows (8-page version) only when α was selected on dev and committed
  if [ -s "$ALPHA_SELECTED_ENV" ] && git ls-files --error-unmatch "$ALPHA_SELECTED_ENV" >/dev/null 2>&1 \
     && git diff --quiet -- "$ALPHA_SELECTED_ENV"; then
    # shellcheck disable=SC1090
    . "$ALPHA_SELECTED_ENV"
    for s in $SEEDS; do
      base="scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS"
      methods+=("$base,alpha=$ALPHA,cost=tokens,name=viword_h3_s$s"
                "$base,alpha=$ALPHA_SYL,cost=syllables,name=viword_h3syl_s$s")
    done
  fi
  reader_tag=$(basename "$READER")
  for spec in $TEST_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    teacher_path="results/teacher_test/$stem.jsonl"
    [ -s "$teacher_path" ] || {
      echo "missing $teacher_path; run: bash scripts/run_pipeline.sh teacher_test before eval" >&2
      exit 1
    }
    task_methods=("${methods[@]}")
    if [ "$task" = vimmrc ] || [ "$task" = belebele ]; then
      task_methods+=("longllmlingua:lm=$LM")
    fi
    [ "$task" = belebele ] && task_methods+=("translate:inner=llmlingua2")
    [ "$task" = vinli ] && task_methods+=(probe_negation)
    # the only reader calls on unseen test rows (viword/heldout.py); every check above has passed.
    # The teacher upper bound uses the same rows as stage_teacher_test: the first unseen ones
    log "evaluate teacher upper bound on $task ($TEACHER_TEST_LIMIT samples)"
    VIWORD_TEST_EVAL=1 python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --exclude "$TEST_EXCLUSIONS" --limit "$TEACHER_TEST_LIMIT" \
      --methods "precomputed:path=$teacher_path,name=teacher_upper" \
      --out "results/eval/teacher_upper_${task}_${reader_tag}.jsonl"
    log "evaluate $task with $READER"
    VIWORD_TEST_EVAL=1 python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS $LIMIT_ARG \
      --methods "${task_methods[@]}" --out "results/eval/${task}_${reader_tag}.jsonl"
  done
}

stage_bertscore() {  # DATN §4.1: BERTScore (PhoBERT) for VietNews, after eval; separate dir, so analyze ignores it
  mkdir -p results/bertscore
  for rows in results/eval/vietnews_*.jsonl; do
    case "$rows" in *.compressed.jsonl) continue ;; esac
    out="results/bertscore/$(basename "$rows")"
    log "BERTScore for $rows"
    python scripts/bertscore.py --rows "$rows" --data data/tasks/vietnews_test.jsonl --out "$out" \
      | tee "${out%.jsonl}.md"
  done
}

analyze_hypotheses() {  # the H1 and H2 runs; arguments are passed on to analyze.py
  echo "## H1 — ViWord-C and the 2×2 design"
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword \
    --reference llmlingua2vi truncation lead tfidf_sent \
    --seed-mode aggregate --seeds $SEEDS \
    --primary-reader "$PRIMARY_READER" --hypothesis H1 \
    --holm-family "H1:tasks=vimmrc+belebele,ratios=0.5+0.333+0.2,reader=primary,test=metric" \
    --factorial llmlingua2vi llmlingua2vi_wordpool viword_syl viword "$@"
  echo
  echo "## H2 — linguistic protection π"
  # aggregate_seed_rows strips _s0/_s1/_s2, so these base names select the
  # logical viword_pi_s*, viword_s*, and llmlingua2vi_force_s* rows in results/eval.
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword_pi \
    --reference viword llmlingua2vi_force llmlingua2vi_forcectx \
    --seed-mode aggregate --seeds $SEEDS \
    --primary-reader "$PRIMARY_READER" --hypothesis H2 \
    --holm-family "H2:tasks=vinli,ratios=0.5+0.333+0.2,reader=primary,test=nfr" "$@"
}

stage_analyze() {  # DATN §4.3: primary analysis without the rows readers scored before the test run
  require_committed_exclusions
  log "analyze all seeds with seed-stratified bootstrap"
  {
    echo "# Primary analysis: test rows no reader scored before the test run ($TEST_EXCLUSIONS)"
    echo
    analyze_hypotheses --exclude "$TEST_EXCLUSIONS"
    echo
    echo "# Sensitivity analysis: every test row"
    echo
    analyze_hypotheses
  } | tee results/analysis_3seed.md
}

[ $# -gt 0 ] || { grep -E '^#( |$)' "$0" | head -9; exit 1; }
for stage in "$@"; do "stage_$stage"; done
