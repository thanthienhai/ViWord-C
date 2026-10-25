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
# Added after the review of 2026-10-24 (none changes a registered hypothesis or family):
#   annot_sheets_v2        blind-v2 sheets for P detector v2, a larger probe sheet, a segmentation sheet
#   select_cost_2x2        one cost-aware ranking rule (α or exact DP) for all four 2x2 cells, on dev
#   teacher_variants       neutral-prompt / open-teacher labels on VARIANT_PARAGRAPHS paragraphs + small checkpoints
#   seg_alt long_ctx minpairs   inputs of the EVAL_EXTRAS descriptive evaluations inside `eval`
#   gold_cbr               CBR of every system on human word boundaries (after eval)
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
# P detector v2 (viword/protect.py) is audited on new blind-v2 sheets, never on the blind-v1 ones used to find
# the v1 errors; VIWORD_P_DETECTOR=v1 keeps the v1 rules and their audit
export VIWORD_P_DETECTOR=${VIWORD_P_DETECTOR:-v2}
PROTECT_LABELS_V1=data/annot/protect_sheet.jsonl
PROBE_LABELS_V1=data/annot/probe_sheet.jsonl
if [ "$VIWORD_P_DETECTOR" = v1 ]; then
  PROTECT_LABELS=${PROTECT_LABELS:-$PROTECT_LABELS_V1}
  PROBE_LABELS=${PROBE_LABELS:-$PROBE_LABELS_V1}
else
  PROTECT_LABELS=${PROTECT_LABELS:-data/annot/protect_sheet_v2.jsonl}
  PROBE_LABELS=${PROBE_LABELS:-data/annot/probe_sheet_v2.jsonl}
fi
GOLD_SEGMENTATION=${GOLD_SEGMENTATION:-data/annot/segmentation_sheet.jsonl}   # human word boundaries (item a)
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
# DATN §2.3: the registered ROUGE-L non-inferiority margin (points), used by the MDE gate and by the H2 test.
# To widen it (allowed before the test run when it is below the seed SD or the test MDE), change it here and
# commit the reason; an environment override only serves dev exploration and is refused by eval and analyze.
NI_MARGIN_REGISTERED=0.5
MDE_MARGIN=${MDE_MARGIN:-$NI_MARGIN_REGISTERED}
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

# cost cuts that change no result: encoder logits once per checkpoint × data set (every π/α/budget/reader variant
# then selects on CPU), and one reader call per distinct prompt, reused across stages (greedy decoding)
export VIWORD_LOGIT_CACHE=${VIWORD_LOGIT_CACHE:-cache/logits}
export VIWORD_READER_CACHE=${VIWORD_READER_CACHE:-cache/reader_outputs.jsonl}
SECOND_READER_FULL=${SECOND_READER_FULL:-0}   # 0: a non-primary reader gets (i), (iv), LLMLingua-2, truncation at 1/3
VIETNEWS_FULL=${VIETNEWS_FULL:-0}             # 0: VietNews gets the Table 5 (H2) systems and a few baselines
LONGLLMLINGUA=${LONGLLMLINGUA:-1}             # reference only; 0 skips it when resources are short
# added descriptive analyses (outside the registered H1/H2 families)
H1_CONTROLS=${H1_CONTROLS:-1}                 # cells iii-rand (runs/softrand_s*) and iv-wordloss (runs/wordloss_s*)
SCOPE_BONUS=${SCOPE_BONUS:-0.2}               # operator-scope bonus on top of π (ViNLI); 0 = off
QREL=${QREL:-0}                               # question-relevance bonus (MC/QA), e.g. 0.3; 0 = off
HARMFUL_CUTS=${HARMFUL_CUTS:-1}               # MLM harmful-cut counts of (i) for the H1 mechanism regression
COST2X2_SELECTED_ENV=${COST2X2_SELECTED_ENV:-config/cost2x2_selected.env}
COST2X2_SELECTED_JSON=${COST2X2_SELECTED_JSON:-config/cost2x2_selected.json}
EVAL_EXTRAS=${EVAL_EXTRAS:-}                  # seg_sensitivity variants long_ctx minpairs (see stage_eval)
VARIANT_PARAGRAPHS=${VARIANT_PARAGRAPHS:-1000}
OPEN_TEACHER_URL=${OPEN_TEACHER_URL:-}        # an open-weights teacher (e.g. `vllm serve Qwen/Qwen2.5-32B-Instruct`)
OPEN_TEACHER_MODEL=${OPEN_TEACHER_MODEL:-}
LONG_TARGET_SYLLABLES=${LONG_TARGET_SYLLABLES:-2500}
LONG_LIMIT=${LONG_LIMIT:-300}

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
  python -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if r.get('sheet') in ('blind-v1','blind-v2') else 'probe audit was not computed from a blind sheet: '+sys.argv[1])" "$PROBE_REPORT"
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

require_registered_margin() {  # the margin of the test run is the committed one, never an environment value
  [ "$MDE_MARGIN" = "$NI_MARGIN_REGISTERED" ] || {
    echo "MDE_MARGIN=$MDE_MARGIN differs from the registered margin $NI_MARGIN_REGISTERED in run_pipeline.sh;" \
         "change NI_MARGIN_REGISTERED there and commit it before the test run" >&2
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
  units="word syllable"
  [ "$H1_CONTROLS" = 1 ] && units="$units syllable_softrand word_pooled"  # cells iii-rand, iv-wordloss
  for s in $SEEDS; do
    for unit in $units; do
      case "$unit" in
        word) name=viword ;; syllable) name=llmlingua2vi ;;
        syllable_softrand) name=softrand ;; word_pooled) name=wordloss ;;
      esac
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
    # reported, never gating: the published LLMLingua-2, and the teacher cut to the budget as in G4
    # (results/teacher_dev covers the first 200 dev examples, so only up to that limit)
    reference_methods=(llmlingua2)
    teacher_dev="results/teacher_dev/$stem.jsonl"
    if [ -s "$teacher_dev" ] && [ "$CKPT_CHECK_LIMIT" -le 200 ]; then
      reference_methods+=("precomputed:path=$teacher_dev,trim=1,name=teacher")
    fi
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" \
      --ratios "$CKPT_CHECK_RATIO" --limit "$CKPT_CHECK_LIMIT" \
      --methods truncation "$(lexprior_spec)" "${reference_methods[@]}" "${checkpoint_methods[@]}" --out "$out"
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
  require_registered_margin
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
  # cost cut (p.8): a reader other than the primary one is descriptive; by default it gets (i), (iv),
  # the published LLMLingua-2 and truncation at 1/3 only (SECOND_READER_FULL=1 for the whole table)
  core_methods=(none no_context truncation llmlingua2)
  # VietNews only serves the H2 non-inferiority test (Table 5); the rest of it is descriptive (p.8)
  vietnews_methods=(none no_context lead truncation llmlingua2)
  scope_methods=() qrel_methods=()
  for s in $SEEDS; do
    methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=word,name=llmlingua2vi_wordpool_s$s"
              "scored:model=runs/viword_s$s,unit=syllable,name=viword_syl_s$s"
              "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force,tiers=$PI_TIERS,name=llmlingua2vi_force_s$s"
              "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force_ctx,tiers=$PI_TIERS,name=llmlingua2vi_forcectx_s$s"
              "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
    core_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                   "scored:model=runs/viword_s$s,unit=word,name=viword_s$s")
    vietnews_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                       "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
                       "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force,tiers=$PI_TIERS,name=llmlingua2vi_force_s$s"
                       "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force_ctx,tiers=$PI_TIERS,name=llmlingua2vi_forcectx_s$s"
                       "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
    if [ "$H1_CONTROLS" = 1 ]; then  # cells iii-rand and iv-wordloss (descriptive, outside the H1 family)
      for control in softrand wordloss; do
        [ -d "runs/${control}_s$s" ] || { echo "missing runs/${control}_s$s (H1_CONTROLS=1); run train" >&2; exit 1; }
      done
      methods+=("scored:model=runs/softrand_s$s,unit=syllable,name=softrand_syl_s$s"
                "scored:model=runs/wordloss_s$s,unit=word,name=viword_wl_s$s")
    fi
    if [ "$SCOPE_BONUS" != 0 ]; then  # operator scope on top of the selected π (ViNLI, descriptive)
      scope_methods+=("scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,scope=$SCOPE_BONUS,name=viword_pi_scope_s$s")
    fi
    if [ "$QREL" != 0 ]; then  # question-aware selection (reading comprehension and QA, descriptive)
      qrel_methods+=("scored:model=runs/viword_s$s,unit=word,qrel=$QREL,name=viword_qrel_s$s")
    fi
  done
  # cost-aware 2x2 (item d): the dev-selected rule for all four cells, outside the registered H1 family
  if [ -s "$COST2X2_SELECTED_ENV" ] && git ls-files --error-unmatch "$COST2X2_SELECTED_ENV" >/dev/null 2>&1 \
     && git diff --quiet -- "$COST2X2_SELECTED_ENV"; then
    # shellcheck disable=SC1090
    . "$COST2X2_SELECTED_ENV"
    if [ "${COST2X2_TAG:-a0}" != a0 ]; then
      for s in $SEEDS; do
        methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,$COST2X2_SPEC,name=llmlingua2vi_ca${COST2X2_TAG}_s$s"
                  "scored:model=runs/llmlingua2vi_s$s,unit=word,$COST2X2_SPEC,name=llmlingua2vi_wordpool_ca${COST2X2_TAG}_s$s"
                  "scored:model=runs/viword_s$s,unit=syllable,$COST2X2_SPEC,name=viword_syl_ca${COST2X2_TAG}_s$s"
                  "scored:model=runs/viword_s$s,unit=word,$COST2X2_SPEC,name=viword_ca${COST2X2_TAG}_s$s")
      done
    fi
  fi
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
  subset_reader=0
  [ "$READER" != "$PRIMARY_READER" ] && [ "$SECOND_READER_FULL" != 1 ] && subset_reader=1
  rr_args=()  # harmful cuts of (i) for the H1 mechanism regression (MLM in the compression phase)
  [ "$HARMFUL_CUTS" = 1 ] && rr_args=(--rr-model "$RR_MODEL" --rr-methods llmlingua2vi_s)
  for spec in $TEST_SETS; do
    IFS=: read -r task stem _ <<< "$spec"
    teacher_path="results/teacher_test/$stem.jsonl"
    [ -s "$teacher_path" ] || {
      echo "missing $teacher_path; run: bash scripts/run_pipeline.sh teacher_test before eval" >&2
      exit 1
    }
    task_ratios=$RATIOS
    if [ "$subset_reader" = 1 ]; then
      task_methods=("${core_methods[@]}")
      task_ratios=0.333
    elif [ "$task" = vietnews ] && [ "$VIETNEWS_FULL" != 1 ]; then
      task_methods=("${vietnews_methods[@]}")
    else
      task_methods=("${methods[@]}")
      if [ "$LONGLLMLINGUA" = 1 ] && { [ "$task" = vimmrc ] || [ "$task" = belebele ]; }; then
        task_methods+=("longllmlingua:lm=$LM")
      fi
      [ "$task" = belebele ] && task_methods+=("translate:inner=llmlingua2")
      [ "$task" = vinli ] && task_methods+=(probe_negation ${scope_methods[@]+"${scope_methods[@]}"})
      case "$task" in vimmrc|belebele|viquad) task_methods+=(${qrel_methods[@]+"${qrel_methods[@]}"}) ;; esac
    fi
    # the only reader calls on unseen test rows (viword/heldout.py); every check above has passed.
    # The teacher upper bound uses the same rows as stage_teacher_test: the first unseen ones
    if [ "$subset_reader" = 0 ]; then
    log "evaluate teacher upper bound on $task ($TEACHER_TEST_LIMIT samples)"
    VIWORD_TEST_EVAL=1 python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --exclude "$TEST_EXCLUSIONS" --limit "$TEACHER_TEST_LIMIT" \
      --methods "precomputed:path=$teacher_path,name=teacher_upper" \
      --out "results/eval/teacher_upper_${task}_${reader_tag}.jsonl"
    fi
    log "evaluate $task with $READER"
    VIWORD_TEST_EVAL=1 python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $task_ratios $LIMIT_ARG \
      ${rr_args[@]+"${rr_args[@]}"} --methods "${task_methods[@]}" --out "results/eval/${task}_${reader_tag}.jsonl"
  done
  # descriptive extensions on the same unseen rows, after the same checks (EVAL_EXTRAS, space-separated):
  #   seg_sensitivity  (i)/(iv) at 1/3 with underthesea/pyvi segmentations (stage_seg_alt makes them)
  #   variants         small checkpoints of the neutral prompt / open teacher (stage_teacher_variants)
  #   long_ctx         passages among distractors, a few thousand tokens (stage_long_ctx makes the data)
  #   minpairs         operator-only ViNLI pairs (dev/train premises; reviewed file)
  extra_eval() {  # extra_eval TASK DATA SEGMENTED OUT RATIOS METHOD...
    local task=$1 data=$2 segmented=$3 out=$4 ratios=$5
    shift 5
    VIWORD_TEST_EVAL=1 python scripts/evaluate.py --task "$task" --data "$data" --segmented "$segmented" \
      --reader "$READER" --backend "$READER_BACKEND" ${READER_URL:+--base-url $READER_URL} \
      --tokenizer "$READER_TOKENIZER" --ratios $ratios --methods "$@" --out "$out"
  }
  for extra in $EVAL_EXTRAS; do
    case "$extra" in
      seg_sensitivity)
        mkdir -p results/seg_sensitivity
        for seg in underthesea pyvi; do
          for spec in vinli:vinli_test belebele:belebele vimmrc:vimmrc_test; do
            IFS=: read -r task stem <<< "$spec"
            [ -s "data/segmented_$seg/$stem.jsonl" ] || { echo "missing data/segmented_$seg/$stem.jsonl; run seg_alt" >&2; exit 1; }
            log "segmenter sensitivity: $task with $seg"
            seg_methods=(none)
            for s in $SEEDS; do
              seg_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                            "scored:model=runs/viword_s$s,unit=word,name=viword_s$s")
            done
            extra_eval "$task" "data/tasks/$stem.jsonl" "data/segmented_$seg/$stem.jsonl" \
              "results/seg_sensitivity/${task}_${seg}_${reader_tag}.jsonl" 0.333 "${seg_methods[@]}"
          done
        done ;;
      variants)
        mkdir -p results/variants
        variant_methods=(none)
        for name in viword_small llmlingua2vi_small viword_neutral viword_open llmlingua2vi_open; do
          [ -d "runs/${name}_s0" ] || continue
          unit=$([ "${name#viword}" != "$name" ] && echo word || echo syllable)
          variant_methods+=("scored:model=runs/${name}_s0,unit=$unit,name=${name}_s0")
          [ "$unit" = word ] && variant_methods+=("scored:model=runs/${name}_s0,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=${name}_pi_s0")
        done
        for spec in vinli:vinli_test vimmrc:vimmrc_test belebele:belebele; do
          IFS=: read -r task stem <<< "$spec"
          log "teacher variants on $task"
          extra_eval "$task" "data/tasks/$stem.jsonl" "data/segmented/$stem.jsonl" \
            "results/variants/${task}_${reader_tag}.jsonl" "$RATIOS" "${variant_methods[@]}"
        done ;;
      long_ctx)
        mkdir -p results/long_ctx
        for spec in belebele:belebele_long vimmrc:vimmrc_test_long; do
          IFS=: read -r task stem <<< "$spec"
          [ -s "data/segmented/$stem.jsonl" ] || { echo "missing data/segmented/$stem.jsonl; run long_ctx" >&2; exit 1; }
          long_methods=(none no_context truncation llmlingua2)
          for s in $SEEDS; do
            long_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                           "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
                           "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
          done
          log "long context: $stem"
          extra_eval "$task" "data/tasks/$stem.jsonl" "data/segmented/$stem.jsonl" \
            "results/long_ctx/${stem}_${reader_tag}.jsonl" "0.333 0.2" "${long_methods[@]}"
        done ;;
      minpairs)
        [ -s data/segmented/vinli_minpairs.jsonl ] || { echo "missing data/segmented/vinli_minpairs.jsonl" >&2; exit 1; }
        mp_methods=(none "random:seed=0" "random:seed=1" "random:seed=2")
        for s in $SEEDS; do
          mp_methods+=("scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s"
                       "scored:model=runs/viword_s$s,unit=word,name=viword_s$s"
                       "scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force,tiers=$PI_TIERS,name=llmlingua2vi_force_s$s"
                       "scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s")
        done
        mkdir -p results/minpairs
        extra_eval vinli data/tasks/vinli_minpairs.jsonl data/segmented/vinli_minpairs.jsonl \
          "results/minpairs/vinli_minpairs_${reader_tag}.jsonl" "$RATIOS" "${mp_methods[@]}" ;;
      *) echo "unknown EVAL_EXTRAS entry: $extra" >&2; exit 1 ;;
    esac
  done
}

stage_seg_alt() {  # §4.4 / item a: underthesea and pyvi segmentations of three test sets (no reader)
  for seg in underthesea pyvi; do
    mkdir -p "data/segmented_$seg"
    for spec in vinli:vinli_test:premise belebele:belebele:context vimmrc:vimmrc_test:context; do
      IFS=: read -r _ stem field <<< "$spec"
      log "segment $stem with $seg"
      python scripts/segment_data.py --input "data/tasks/$stem.jsonl" --field "$field" \
        --output "data/segmented_$seg/$stem.jsonl" --segmenter "$seg" --workers "$SEG_WORKERS"
    done
  done
}

stage_long_ctx() {  # item 10: build and segment the long-context sets (no reader; eval reads them via EVAL_EXTRAS)
  require_committed_exclusions
  for spec in belebele:belebele_long vimmrc_test:vimmrc_test_long; do
    IFS=: read -r src stem <<< "$spec"
    python scripts/make_long_context.py --input "data/tasks/$src.jsonl" --exclude "$TEST_EXCLUSIONS" \
      --target-syllables "$LONG_TARGET_SYLLABLES" --limit "$LONG_LIMIT" --out "data/tasks/$stem.jsonl"
    python scripts/segment_data.py --input "data/tasks/$stem.jsonl" --field context \
      --output "data/segmented/$stem.jsonl" --segmenter "$SEGMENTER" --workers "$SEG_WORKERS"
  done
}

stage_minpairs() {  # item 9: candidate minimal pairs from dev/train premises, for human review
  require_segmented
  python scripts/make_minimal_pairs.py --sources vinli_dev vinli_train --n "${MINPAIRS_N:-300}" \
    --out data/annot/vinli_minpairs_review.jsonl
  echo "review data/annot/vinli_minpairs_review.jsonl (set needs_review false), then:" >&2
  echo "  python scripts/make_minimal_pairs.py --reviewed-only data/annot/vinli_minpairs_review.jsonl --out data/tasks/vinli_minpairs.jsonl" >&2
  echo "  python scripts/segment_data.py --input data/tasks/vinli_minpairs.jsonl --field premise --output data/segmented/vinli_minpairs.jsonl" >&2
}

stage_teacher_variants() {  # items g and 11: neutral prompt and open teacher on VARIANT_PARAGRAPHS paragraphs
  require_segmented
  variants=(neutral)
  [ -n "$OPEN_TEACHER_MODEL" ] && variants+=(open)
  for variant in "${variants[@]}"; do
    if [ "$variant" = neutral ]; then
      url=$TEACHER_URL model=$TEACHER_MODEL prompt=neutral
    else
      url=${OPEN_TEACHER_URL:?set OPEN_TEACHER_URL for the open teacher} model=$OPEN_TEACHER_MODEL prompt=default
    fi
    log "teacher variant $variant ($model, prompt $prompt) on $VARIANT_PARAGRAPHS paragraphs"
    python scripts/teacher_compress.py --input data/distill/paragraphs_clean.jsonl --field text \
      --limit "$VARIANT_PARAGRAPHS" --rates 0.5 0.25 --prompt "$prompt" \
      --output "data/distill/teacher_$variant.jsonl" \
      --base-url "$url" --model "$model" --workers "$TEACHER_WORKERS" --max-tokens "$TEACHER_MAX_TOKENS"
    python scripts/distill.py --paragraphs data/distill/paragraphs_clean.jsonl \
      --segmented data/segmented/paragraphs_clean.jsonl --teacher "data/distill/teacher_$variant.jsonl" \
      --output "data/distill/distilled_$variant.jsonl" | tee "data/distill/distill_summary_$variant.json"
    python scripts/teacher_agreement.py --a data/distill/distilled.jsonl --b "data/distill/distilled_$variant.jsonl" \
      --out "results/teacher_agreement_$variant.md"
  done
  # matched checkpoints: the main teacher on the same paragraphs, then each variant (one seed, descriptive)
  small_args=(--ids-from data/distill/distilled_neutral.jsonl --dev-paragraphs "${VARIANT_DEV_PARAGRAPHS:-100}" --seed 0)
  python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit word --out runs/viword_small_s0 "${small_args[@]}"
  python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit syllable --out runs/llmlingua2vi_small_s0 "${small_args[@]}"
  python scripts/train.py --distilled data/distill/distilled_neutral.jsonl --label-unit word --out runs/viword_neutral_s0 "${small_args[@]}"
  if [ -n "$OPEN_TEACHER_MODEL" ]; then
    python scripts/train.py --distilled data/distill/distilled_open.jsonl --label-unit word --out runs/viword_open_s0 "${small_args[@]}"
    python scripts/train.py --distilled data/distill/distilled_open.jsonl --label-unit syllable --out runs/llmlingua2vi_open_s0 "${small_args[@]}"
  fi
}

stage_gold_cbr() {  # item a: CBR of every system (teacher included) on human word boundaries, after eval
  [ -s "$GOLD_SEGMENTATION" ] || { echo "missing $GOLD_SEGMENTATION (annot_sheets_v2, then annotate)" >&2; exit 1; }
  reader_tag=$(basename "$PRIMARY_READER")
  mkdir -p results/gold_cbr
  python scripts/gold_cbr.py --gold "$GOLD_SEGMENTATION" --segmented data/segmented/vinli_test.jsonl \
    --compressed "results/eval/vinli_${reader_tag}.compressed.jsonl" \
                 "results/eval/teacher_upper_vinli_${reader_tag}.compressed.jsonl" \
    --out results/gold_cbr/vinli.md --json-out results/gold_cbr/vinli.json
}

stage_annot_sheets_v2() {  # items f and 12: new blind sheets for detector v2, a larger probe sheet, segmentation
  require_segmented
  python scripts/make_annotation_sheets.py --segmented-dir data/segmented --tasks-dir data/tasks \
    --sheet blind-v2 --exclude-sheets "$PROTECT_LABELS_V1" "$PROBE_LABELS_V1" --seed 1 --per-group 100 \
    --protect-out "$PROTECT_LABELS" --probe-out "$PROBE_LABELS" --only protect probe
  python scripts/make_annotation_sheets.py --tasks-dir data/tasks --only segmentation \
    --segmentation-out "$GOLD_SEGMENTATION" --segmentation-n "${GOLD_SEGMENTATION_N:-300}"
}

stage_select_cost_2x2() {  # item d: one cost-aware ranking rule for all four 2x2 cells, selected on dev
  require_primary_reader
  require_segmented
  reader_tag=$(basename "$READER")
  mkdir -p results/cost2x2_dev config
  cost_methods=(none)
  for s in $SEEDS; do
    for cell in "llmlingua2vi:llmlingua2vi:syllable" "llmlingua2vi_wordpool:llmlingua2vi:word" \
                "viword_syl:viword:syllable" "viword:viword:word"; do
      IFS=: read -r name ckpt unit <<< "$cell"
      cost_methods+=("scored:model=runs/${ckpt}_s$s,unit=$unit,name=${name}_s$s")
      for rule in a0p5:alpha=0.5 a1:alpha=1 dp:exact=1; do
        IFS=: read -r tag rule_spec <<< "$rule"
        cost_methods+=("scored:model=runs/${ckpt}_s$s,unit=$unit,$rule_spec,name=${name}_ca${tag}_s$s")
      done
    done
  done
  rows=()
  for spec in vinli:vinli_dev vimmrc:vimmrc_dev; do
    IFS=: read -r task stem <<< "$spec"
    log "select the 2x2 cost rule on $stem with $READER"
    out="results/cost2x2_dev/${stem}_${reader_tag}.jsonl"
    python scripts/evaluate.py --task "$task" --data "data/tasks/$stem.jsonl" \
      --segmented "data/segmented/$stem.jsonl" --reader "$READER" --backend "$READER_BACKEND" \
      ${READER_URL:+--base-url $READER_URL} --tokenizer "$READER_TOKENIZER" --ratios $RATIOS \
      --limit "$PI_DEV_LIMIT" --methods "${cost_methods[@]}" --out "$out"
    rows+=("$out")
  done
  python scripts/select_cost_2x2.py --rows "${rows[@]}" --out "$COST2X2_SELECTED_JSON" --env "$COST2X2_SELECTED_ENV" \
    --report results/cost2x2_selection.md
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
    --factorial llmlingua2vi llmlingua2vi_wordpool viword_syl viword --margin "$MDE_MARGIN" \
    --cbr-regression llmlingua2vi "$@"
  echo
  if [ "$H1_CONTROLS" = 1 ]; then
    echo "## H1 controls (descriptive): soft labels vs word boundaries"
    echo "iii vs iii-rand: word labels vs randomly softened syllable labels, both decided per syllable."
    python scripts/analyze.py --rows results/eval/*.jsonl --target viword_syl --reference softrand_syl \
      --seed-mode aggregate --seeds $SEEDS --primary-reader "$PRIMARY_READER" --hypothesis none --no-descriptive "$@"
    echo "iv-wordloss vs (i) and vs (iv): the loss taken on the word."
    python scripts/analyze.py --rows results/eval/*.jsonl --target viword_wl --reference llmlingua2vi viword \
      --seed-mode aggregate --seeds $SEEDS --primary-reader "$PRIMARY_READER" --hypothesis none --no-descriptive \
      --margin "$MDE_MARGIN" "$@"
    echo
  fi
  if [ -s "$COST2X2_SELECTED_ENV" ]; then
    # shellcheck disable=SC1090
    . "$COST2X2_SELECTED_ENV"
    if [ "${COST2X2_TAG:-a0}" != a0 ]; then
      echo "## 2x2 with the dev-selected cost-aware ranking ($COST2X2_SPEC, descriptive)"
      t=$COST2X2_TAG
      python scripts/analyze.py --rows results/eval/*.jsonl --target "viword_ca$t" --reference "llmlingua2vi_ca$t" \
        --seed-mode aggregate --seeds $SEEDS --primary-reader "$PRIMARY_READER" --hypothesis none --no-descriptive \
        --factorial "llmlingua2vi_ca$t" "llmlingua2vi_wordpool_ca$t" "viword_syl_ca$t" "viword_ca$t" "$@"
      echo
    fi
  fi
  echo "## H2 — linguistic protection π"
  # aggregate_seed_rows strips _s0/_s1/_s2, so these base names select the
  # logical viword_pi_s*, viword_s*, and llmlingua2vi_force_s* rows in results/eval.
  python scripts/analyze.py --rows results/eval/*.jsonl --target viword_pi \
    --reference viword llmlingua2vi_force llmlingua2vi_forcectx \
    --seed-mode aggregate --seeds $SEEDS \
    --primary-reader "$PRIMARY_READER" --hypothesis H2 \
    --holm-family "H2:tasks=vinli,ratios=0.5+0.333+0.2,reader=primary,test=nfr" --margin "$MDE_MARGIN" "$@"
}

stage_analyze() {  # DATN §4.3: primary analysis without the rows readers scored before the test run
  require_committed_exclusions
  require_registered_margin
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
