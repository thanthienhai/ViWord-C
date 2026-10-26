#!/usr/bin/env bash
# Direction B (question-aware compression for Vietnamese legal RAG, ViBidLQA) on one H100 machine
# (docs/RUN_B_H100.md). Stages, run in this order; send results/b/ back after each call:
#
#   bash scripts/b_h100.sh setup data      # python venv, dependencies, data from the private HF repo (~10 min)
#   bash scripts/b_h100.sh smoke           # the whole chain on a few questions (~15 min)
#   bash scripts/b_h100.sh labels train    # utility labels (~45-60 min), 7 unit scorers (~1.5-2 h)
#   bash scripts/b_h100.sh dev             # dev evaluation of every method and its report (~2-3 h)
#   bash scripts/b_h100.sh test            # only after config/b_registration.json is committed (gate G2)
#
# Reader: Qwen2.5-7B-Instruct in bf16 (not the 8-bit reader of the Kaggle runs), so the dev numbers are measured
# again here for every method, baselines included. Models are downloaded from the Hugging Face hub (none gated).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
VENV=${VENV:-.venv}
HF_DATA_REPO_B=${HF_DATA_REPO_B:-thanthienhai/viword-c-legal-data}
HF_DATA_REVISION_B=${HF_DATA_REVISION_B:-main}
READER=${READER:-Qwen/Qwen2.5-7B-Instruct}
RERANKER=${RERANKER:-BAAI/bge-reranker-v2-m3}
ENCODER=${ENCODER:-$RERANKER}  # the unit scorer starts from the reranker (viword/unit_scorer.py)
LM=${LM:-Qwen/Qwen2.5-1.5B}
SEEDS=${SEEDS:-"0 1 2"}
RATIOS="0.25 0.125 0.0625"
LL2=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
OUT=results/b
LABELS=data/labels/vibid_train_utility.jsonl

export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
export VIWORD_LOGIT_CACHE=${VIWORD_LOGIT_CACHE:-$PWD/.cache/logits_b}
mkdir -p "$OUT/eval" "$OUT/logs"
log() { echo -e "\n=== $* ($(date '+%F %T'))" | tee -a "$OUT/b.log"; }
activate() {
  # shellcheck disable=SC1091
  [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
}

# the fine compressor inside the selected clause: published LLMLingua-2 weights with the question-aware sentence
# bonus (best token compressor on legal text in the Kaggle G0 runs)
FINE="fine=$LL2,unit=syllable,qrel=5"

methods() {  # every method of the dev evaluation (and, once registered, of the test run)
  local s
  printf '%s\n' none no_context lead qsent qunit llmlingua2 "longllmlingua:lm=$LM,units=1"
  echo "scored:model=$LL2,unit=syllable,name=llmlingua2w"
  echo "scored:model=$LL2,unit=syllable,qrel=5,qrel_unit=sentence,name=llmlingua2w_q5"
  echo "unitsel:scorer=$RERANKER,fine=none,name=rerank"                     # reranker baseline: rank and cut
  echo "unitsel:scorer=bm25,$FINE,name=unitsel_bm25"                        # ablation: lexical units + fine
  echo "unitsel:scorer=$RERANKER,$FINE,name=unitsel_rerank"                 # ablation: zero-shot units + fine
  for s in $SEEDS; do
    echo "unitsel:scorer=runs/unit_utility_s$s,$FINE,name=unitsel_util_s$s"  # proposed (H1)
    echo "unitsel:scorer=runs/unit_gold_s$s,$FINE,name=unitsel_gold_s$s"     # gold-unit labels (H4)
  done
  echo "unitsel:scorer=runs/unit_utility_s0,fine=none,name=unitsel_util_nofine_s0"          # no fine compression
  echo "unitsel:scorer=runs/unit_utility_s0,$FINE,lam=0.3,name=unitsel_util_mig_s0"         # MIG allocation
  echo "unitsel:scorer=runs/unit_utility_s0,$FINE,level=sentence,name=unitsel_util_sent_s0" # sentences (H2)
  echo "unitsel:scorer=runs/unit_utility_unseen_s0,$FINE,name=unitsel_util_unseen_s0"       # unseen articles
}

evaluate() {  # $1 data stem, $2 output name, then methods
  local stem=$1 name=$2; shift 2
  python scripts/evaluate.py --task vibid --data "data/tasks/$stem.jsonl" --segmented "data/segmented/$stem.jsonl" \
    --reader "$READER" --tokenizer "$READER" --ratios $RATIOS --max-model-len 4096 --methods "$@" \
    --out "$OUT/eval/$name.jsonl" > "$OUT/logs/eval_$name.log" 2>&1
  grep -h "wrote [0-9]* rows\|reader calls" "$OUT/logs/eval_$name.log" | tee -a "$OUT/b.log"
}

step_setup() {
  log "setup: python venv in $VENV"
  python3 -m venv "$VENV"
  activate
  pip install -q --upgrade pip
  pip install -q -e ".[model,eval,baselines,dev]"
  python -m pytest -q tests | tail -1 | tee -a "$OUT/b.log"
  python - <<'EOF' | tee -a "$OUT/b.log"
import torch, transformers, vllm
assert torch.cuda.is_available(), "no GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB, bf16: {torch.cuda.is_bf16_supported()}; "
      f"torch {torch.__version__}, transformers {transformers.__version__}, vllm {vllm.__version__}")
EOF
}

step_data() {
  activate
  log "data from $HF_DATA_REPO_B @ $HF_DATA_REVISION_B"
  python scripts/hf_data.py pull --repo "$HF_DATA_REPO_B" --revision "$HF_DATA_REVISION_B"
  python scripts/hf_data.py check --sums data/SHA256SUMS_B | tee -a "$OUT/b.log"
}

step_smoke() {  # every piece on a few questions: vLLM labels, scorer training, unitsel through the reader
  activate
  log "smoke: labels on 16 train questions"
  python scripts/vibid_labels.py --contexts data/tasks/vibid_rag_random_train.jsonl --reader "$READER" \
    --limit 16 --out "$OUT/smoke/labels.jsonl" 2>&1 | tee "$OUT/logs/smoke_labels.log" | grep "wrote" | tee -a "$OUT/b.log"
  log "smoke: one epoch of the unit scorer"
  python scripts/train_unit_scorer.py --labels "$OUT/smoke/labels.jsonl" --model-name "$ENCODER" --epochs 1 \
    --dev-share 0.25 --out runs/unit_smoke 2>&1 | tee "$OUT/logs/smoke_train.log" | tail -2 | tee -a "$OUT/b.log"
  log "smoke: 30 dev questions through the reader"
  python scripts/evaluate.py --task vibid --data data/tasks/vibid_rag_random_dev.jsonl \
    --segmented data/segmented/vibid_rag_random_dev.jsonl --reader "$READER" --tokenizer "$READER" \
    --ratios 0.0625 --max-model-len 4096 --limit 30 --out "$OUT/smoke/eval.jsonl" --methods none qunit \
    "unitsel:scorer=runs/unit_smoke,$FINE,name=unitsel_smoke" "unitsel:scorer=$RERANKER,fine=none,name=rerank" \
    > "$OUT/logs/smoke_eval.log" 2>&1
  grep -h "wrote [0-9]* rows" "$OUT/logs/smoke_eval.log" | tee -a "$OUT/b.log"
  python - <<'EOF' | tee -a "$OUT/b.log"
import json, collections
rows = [json.loads(l) for l in open("results/b/smoke/eval.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for r in rows:
    by[r["method"]].append(r)
for m, rs in by.items():
    print(f"{m}: {len(rs)} rows, F1 {sum(r['f1'] for r in rs) / len(rs):.3f}, recall "
          f"{sum(r['recall'] for r in rs) / len(rs):.3f}, max tokens/budget "
          f"{max((r['tokens_per_budget'] or 0) for r in rs):.3f}")
EOF
  rm -rf runs/unit_smoke
  log "smoke done"
}

step_labels() {
  activate
  log "labels: utility of every candidate clause, ViBidLQA train"
  python scripts/vibid_labels.py --contexts data/tasks/vibid_rag_random_train.jsonl --reader "$READER" \
    --out "$LABELS" 2>&1 | tee "$OUT/logs/labels.log" | grep "wrote" | tee -a "$OUT/b.log"
}

step_train() {
  activate
  [ -s "$LABELS" ] || { echo "missing $LABELS: run the labels stage first"; exit 1; }
  local s label
  for s in $SEEDS; do
    for label in utility gold; do
      log "train: unit scorer, $label labels, seed $s"
      python scripts/train_unit_scorer.py --labels "$LABELS" --label "$label" --model-name "$ENCODER" --seed "$s" \
        --out "runs/unit_${label}_s$s" > "$OUT/logs/train_${label}_s$s.log" 2>&1
      tail -1 "$OUT/logs/train_${label}_s$s.log" | tee -a "$OUT/b.log"
      cp "runs/unit_${label}_s$s/train_meta.json" "$OUT/unit_${label}_s${s}_train_meta.json"
    done
  done
  log "train: unit scorer without the train questions on dev/test gold articles, seed 0"
  python scripts/train_unit_scorer.py --labels "$LABELS" --label utility --model-name "$ENCODER" --seed 0 \
    --exclude-heldout-articles --out runs/unit_utility_unseen_s0 > "$OUT/logs/train_utility_unseen_s0.log" 2>&1
  tail -1 "$OUT/logs/train_utility_unseen_s0.log" | tee -a "$OUT/b.log"
  cp runs/unit_utility_unseen_s0/train_meta.json "$OUT/unit_utility_unseen_s0_train_meta.json"
}

step_dev() {
  activate
  local m
  mapfile -t M < <(methods)
  for m in "${M[@]}"; do
    case $m in *runs/unit_*) d=${m#*scorer=}; d=${d%%,*}; [ -d "$d" ] || { echo "missing $d: train first"; exit 1; };; esac
  done
  log "dev: random-position gold contexts, ${#M[@]} methods"
  evaluate vibid_rag_random_dev vibid_rag_random_dev "${M[@]}"
  log "dev: retrieved contexts"
  evaluate vibid_rag_dev vibid_rag_dev "${M[@]}"
  log "dev: gold units alone"
  evaluate vibid_rag_oracle_dev vibid_rag_oracle_dev none
  python scripts/analyze_vibid.py --eval "$OUT/eval" --split dev > "$OUT/dev.md"
  log "dev done: $OUT/dev.md"
}

step_test() {  # the registered configuration only, once (DATN §4.3 and the direction-B proposal, gate G2)
  activate
  local reg=config/b_registration.json
  git ls-files --error-unmatch "$reg" > /dev/null 2>&1 && git diff --quiet HEAD -- "$reg" \
    || { echo "refusing the test run: $reg must be committed and unchanged (gate G2)"; exit 1; }
  mapfile -t M < <(python -c "import json; print('\n'.join(json.load(open('$reg'))['methods']))")
  log "test: ${#M[@]} registered methods ($(git rev-parse --short HEAD))"
  export VIWORD_TEST_EVAL=1
  evaluate vibid_rag_random_test vibid_rag_random_test "${M[@]}"
  evaluate vibid_rag_test vibid_rag_test "${M[@]}"
  evaluate vibid_rag_oracle_test vibid_rag_oracle_test none
  python scripts/analyze_vibid.py --eval "$OUT/eval" --split test \
    --proposed "$(python -c "import json; print(json.load(open('$reg'))['proposed'])")" \
    --baselines "$(python -c "import json; print(','.join(json.load(open('$reg'))['baselines']))")" > "$OUT/test.md"
  log "test done: $OUT/test.md"
}

[ $# -gt 0 ] || { sed -n '2,12p' "$0"; exit 1; }
for stage in "$@"; do
  case $stage in
    setup|data|smoke|labels|train|dev|test) "step_$stage" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
