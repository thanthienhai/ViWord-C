"""Utility labels for the unit scorer of direction B (viword.unit_labels), on the ViBidLQA train contexts.

  python scripts/vibid_labels.py --contexts data/tasks/vibid_rag_random_train.jsonl \
      --reader Qwen/Qwen2.5-7B-Instruct --out data/labels/vibid_train_utility.jsonl            # H100, vLLM
  python scripts/vibid_labels.py ... --reader Qwen/Qwen2.5-0.5B-Instruct --backend hf --limit 3   # CPU smoke

The contexts are the `random` mode of scripts/legal_rag.py (BM25 units plus the gold units, ~13 units per
question), so every question has its gold units among the candidates. Each row of the output holds the units, their
texts, the gold flags and, per unit, the answer's log-probability with that unit alone and its gain over no context
(`utility`, nats per answer token). Rows built from test passages are refused (viword.heldout).
"""
import argparse
import os
import time

from viword.data import read_jsonl, write_jsonl
from viword.heldout import refuse_unseen_test_rows
from viword.unit_labels import HFAnswerScorer, VLLMAnswerScorer, label_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--system-prompt", default=os.environ.get("READER_SYSTEM_PROMPT") or None)
    args = ap.parse_args()

    rows = read_jsonl(args.contexts)[:args.limit]
    refuse_unseen_test_rows(rows)
    scorer = (VLLMAnswerScorer(args.reader, args.max_model_len) if args.backend == "vllm"
              else HFAnswerScorer(args.reader))
    start = time.perf_counter()
    labels = label_rows(rows, scorer, system_prompt=args.system_prompt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, labels)
    n_units = sum(len(r["units"]) for r in labels)
    gold = [u for r in labels for u, g in zip(r["utility"], r["gold"]) if g]
    other = [u for r in labels for u, g in zip(r["utility"], r["gold"]) if not g]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    top_is_gold = mean([float(r["gold"][max(range(len(r["utility"])), key=r["utility"].__getitem__)])
                        for r in labels])
    print(f"wrote {len(labels)} label rows ({n_units} units) to {args.out} in {time.perf_counter() - start:.0f} s; "
          f"mean utility: gold units {mean(gold):.3f}, others {mean(other):.3f}; "
          f"the best unit is a gold unit for {top_is_gold:.3f} of the questions")


if __name__ == "__main__":
    main()
