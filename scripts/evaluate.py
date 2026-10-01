"""Compress contexts, query one reader, score (DATN §4).

Two phases so that compressors and the reader never share the GPU:
  1. compress every (method, ratio, example) with the budget counted in the reader's
     tokenizer, and cache the results in <out>.compressed.jsonl;
  2. load the reader with vLLM, generate, score, and write one row per example.
`none` and `no_context` are run once (ratio 1.0); analysis pairs them with every ratio.

Usage:
  python scripts/evaluate.py --task belebele --data data/tasks/belebele.jsonl \
      --segmented data/segmented/belebele.jsonl --reader Qwen/Qwen2.5-7B-Instruct \
      --methods none no_context lead truncation llmlingua2 \
          scored:model=runs/viword_s0,unit=word,name=viword_s0 \
      --limit 500 --out results/eval/belebele_qwen7b.jsonl
"""
import argparse
import gc
import os

from tqdm import tqdm

from viword.data import load_contexts, load_task, read_jsonl, write_jsonl
from viword.eval import MAX_NEW_TOKENS, APIReader, HFReader, VLLMReader, build_prompt, parse_prediction, score
from viword.methods import build_compressor
from viword.select import TokenCounter

UNCOMPRESSED = {"none", "no_context"}


def compress_all(args, examples, contexts) -> list[dict]:
    from transformers import AutoTokenizer

    counter = TokenCounter(AutoTokenizer.from_pretrained(args.tokenizer or args.reader))
    rows, cache = [], {}
    for spec in args.methods:
        compressor = build_compressor(spec, cache)
        ratios = [1.0] if compressor.name in UNCOMPRESSED else args.ratios
        for ratio in ratios:
            for ex, ctx in tqdm(list(zip(examples, contexts)), desc=f"{compressor.name} @ {ratio:.2f}"):
                ctx.ratio = ratio
                n_original = counter.count(ctx.text)
                budget = round(ratio * n_original)
                compressed = compressor.compress(ctx, budget, counter)
                rows.append({"id": ex.id, "cluster": ex.cluster, "method": compressor.name,
                             "ratio": round(ratio, 3), "budget": budget,
                             "n_tokens_original": n_original,
                             "n_tokens_compressed": counter.count(compressed),
                             "compressed": compressed})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["vietnews", "vinli", "vimmrc", "belebele"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1 / 3, 0.2])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "api"],
                    help="hf = plain transformers (smoke tests, no vLLM); api = OpenAI-compatible server")
    ap.add_argument("--base-url", default=None, help="for --backend api")
    ap.add_argument("--tokenizer", default=None,
                    help="HF id of the reader's tokenizer, if --reader is not a HF id (api backend)")
    ap.add_argument("--workers", type=int, default=16, help="parallel requests for --backend api")
    args = ap.parse_args()

    examples = load_task(args.task, args.data, args.limit)
    contexts = load_contexts(examples, args.segmented)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cache_path = args.out.replace(".jsonl", "") + ".compressed.jsonl"
    if os.path.exists(cache_path):
        compressed_rows = read_jsonl(cache_path)
        print(f"reusing compressions from {cache_path}")
    else:
        compressed_rows = compress_all(args, examples, contexts)
        write_jsonl(cache_path, compressed_rows)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    by_id = {ex.id: ex for ex in examples}
    if args.backend == "vllm":
        reader = VLLMReader(args.reader, max_model_len=args.max_model_len)
    elif args.backend == "api":
        reader = APIReader(args.reader, args.base_url, args.tokenizer or args.reader, workers=args.workers)
    else:
        reader = HFReader(args.reader)
    task = examples[0].task
    prompts = [build_prompt(by_id[r["id"]], r["compressed"]) for r in compressed_rows]
    outputs = reader.generate(prompts, MAX_NEW_TOKENS[task])

    rows = []
    for r, raw in zip(compressed_rows, outputs):
        ex = by_id[r["id"]]
        prediction = parse_prediction(task, raw)
        rows.append({"task": args.task, "reader": args.reader, **{k: v for k, v in r.items() if k != "compressed"},
                     "prediction": prediction, "raw_output": raw,
                     "n_output_tokens": len(reader.tokenizer.encode(raw, add_special_tokens=False)),
                     **score(ex, prediction)})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
