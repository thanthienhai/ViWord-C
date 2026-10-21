#!/usr/bin/env bash
# BERTScore (PhoBERT) for the VietNews test rows of part 3, on the project owner's machine (DATN §4.1).
# The GPU machine does not run it, so it needs no Java: VnCoreNLP segments the summaries here.
#
#   python scripts/hf_results.py pull --match part4              # or part3; into results/from_gpu/
#   bash scripts/bertscore_local.sh results/from_gpu/viword_part4_<date>.tgz   # or an unpacked folder
#
# Needs Java >= 8, the [seg,eval] extras and data/tasks/vietnews_test.jsonl (VIETNEWS) matching data/SHA256SUMS;
# a GPU is optional (PhoBERT-base). A .tgz must come with its .sha256. Writes
# results/bertscore/<rows>.{jsonl,md} and env.txt (what scored them). The part4 package holds the rows
# of both readers (part3 only those of the primary reader).
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:?usage: bash scripts/bertscore_local.sh <viword_part3_*.tgz | unpacked folder>}
PYTHON=${PYTHON:-python}
OUT=results/bertscore
VIETNEWS=${VIETNEWS:-data/tasks/vietnews_test.jsonl}   # references; data/tasks/vietnews_dev.jsonl for a dev smoke
export PYTHONPATH=. PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false

command -v java >/dev/null || { echo "no Java: VnCoreNLP needs a JRE >= 8" >&2; exit 1; }
grep "  $VIETNEWS$" data/SHA256SUMS | sha256sum --quiet -c \
  || { echo "$VIETNEWS differs from data/SHA256SUMS: pull the HF data" >&2; exit 1; }

if [ -d "$SRC" ]; then
  root=$SRC
else  # only the VietNews rows and the version records, into a folder next to the archive
  "$PYTHON" scripts/hf_results.py verify "$SRC"   # needs the .sha256 packed with it on the GPU machine
  root="${SRC%.tgz}"
  "$PYTHON" - "$SRC" "$root" <<'EOF'
import re, sys, tarfile
with tarfile.open(sys.argv[1]) as tar:
    keep = [m for m in tar.getmembers() if m.isfile()
            and re.search(r"(^|/)(eval/vietnews_[^/]*\.jsonl|data_revision\.txt|pip_freeze\.txt)$", m.name)]
    tar.extractall(sys.argv[2], members=keep, filter="data")
print(f"unpacked {len(keep)} files into {sys.argv[2]}")
EOF
fi
eval_dir=$(find "$root" -type d -name eval -path "*full/eval" | head -1)
[ -n "$eval_dir" ] || eval_dir=$(find "$root" -type d -name eval | head -1)
[ -n "$eval_dir" ] || { echo "no eval/ folder in $SRC" >&2; exit 1; }

mkdir -p "$OUT"
{
  echo "# $(date '+%F %T') BERTScore of $SRC"
  echo "code $(git rev-parse HEAD)"
  "$PYTHON" -c "import sys, torch, transformers, bert_score; print(f'python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, bert_score {bert_score.__version__}, GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
  java -version 2>&1 | head -1
  find "$root" -name data_revision.txt -exec cat {} + 2>/dev/null || true
} | tee -a "$OUT/env.txt"

n=0
for rows in "$eval_dir"/vietnews_*.jsonl; do
  case "$rows" in *.compressed.jsonl) continue ;; esac
  [ -e "$rows" ] || continue
  out="$OUT/$(basename "$rows")"
  echo -e "\n=== BERTScore for $rows ($(date '+%F %T'))"
  "$PYTHON" scripts/bertscore.py --rows "$rows" --data "$VIETNEWS" --out "$out" \
    | tee "${out%.jsonl}.md"
  n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "no vietnews_*.jsonl rows in $eval_dir" >&2; exit 1; }
echo -e "\nBERTScore for $n row files in $OUT"
