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
import subprocess
import sys
import time

from tqdm import tqdm

from viword.data import TASKS, examples_from_rows, load_contexts, read_jsonl, write_jsonl
from viword.heldout import excluded_ids, load_exclusions, refuse_unseen_test_rows
from viword.eval import (MAX_NEW_TOKENS, APIReader, HFReader, VLLMReader, budget_deviates,
                         build_prompt, context_budget, parse_prediction, score, tokens_per_budget)
from viword.methods import build_compressor
from viword.diagnose import cbr_counts, chance_cbr_counts, recovery_counts, retention_counts, scope_counts
from viword.segment import align_compressed, flatten
from viword.select import TokenCounter

UNCOMPRESSED = {"none", "no_context"}


def row_cbr(ctx, compressed: str, mlm=None) -> dict:
    """Per-example compound-break counts (DATN §2.2), for the H1 mechanism regression, with
    the expected counts of random syllable deletion at the same keep rate (CBR vs chance),
    retention per protection tier (item g: T1 retention without π) and operator-scope
    retention (item e). With `mlm`, broken words the MLM cannot restore are counted as
    harmful cuts (RR, DATN §2.2), the regressor that can carry the H1 mechanism."""
    if ctx.doc is None:
        return {}
    words = flatten(ctx.doc)
    mask, _ = align_compressed(words, compressed)
    counts = cbr_counts(words, mask)
    partial, full = counts["all", "partial"], counts["all", "full"]
    chance = chance_cbr_counts(words, mask)
    retention = retention_counts(words, mask)
    scope = scope_counts(words, mask)
    out = {"cbr_partial": partial, "cbr_full": full,
           "cbr": partial / (partial + full) if partial + full else None,
           "cbr_chance_partial": round(chance["partial"], 4), "cbr_chance_survived": round(chance["survived"], 4),
           "tier_kept": {g: [retention[g, "kept"], retention[g, "total"]]
                         for g in ("all", "T1", "T2", "T3") if retention[g, "total"]},
           "scope_ops_kept": scope["ops_kept"], "scope_heads_kept": scope["ops_kept_with_head"]}
    if mlm is not None and partial:
        recovered, checked = recovery_counts(words, mask, mlm)
        out.update({"cbr_rr_checked": checked, "cbr_harmful": checked - recovered})
    return out


def compress_all(args, examples, contexts) -> list[dict]:
    from transformers import AutoTokenizer

    counter = TokenCounter(AutoTokenizer.from_pretrained(args.tokenizer or args.reader))
    rows, cache = [], {}
    mlm, rr_methods = None, getattr(args, "rr_methods", None) or []
    if getattr(args, "rr_model", None):
        from transformers import pipeline
        mlm = pipeline("fill-mask", model=args.rr_model)
    for spec in args.methods:
        compressor = build_compressor(spec, cache)
        ratios = [1.0] if compressor.name in UNCOMPRESSED else args.ratios
        for ratio in ratios:
            for ex, ctx in tqdm(list(zip(examples, contexts)), desc=f"{compressor.name} @ {ratio:.2f}"):
                ctx.ratio = ratio
                n_original, budget = context_budget(ctx.text, ratio, counter)
                if hasattr(compressor, "last_stats"):
                    compressor.last_stats = {}
                start = time.perf_counter()
                compressed = compressor.compress(ctx, budget, counter)
                compress_ms = 1000 * (time.perf_counter() - start)
                n_compressed = counter.count(compressed)
                actual_ratio = tokens_per_budget(n_compressed, budget)
                rows.append({"id": ex.id, "cluster": ex.cluster, "method": compressor.name,
                             "budget_tokenizer": args.tokenizer or args.reader,
                             "ratio": round(ratio, 3), "budget": budget,
                             "n_tokens_original": n_original,
                             "n_tokens_compressed": n_compressed,
                             "tokens_per_budget": actual_ratio,
                             "budget_deviation_5pct": budget_deviates(actual_ratio),
                             "compress_ms": round(compress_ms, 2),  # DATN §4.4 latency
                             **({"selection": dict(compressor.last_stats)}
                                if getattr(compressor, "last_stats", None) else {}),
                             **(row_cbr(ctx, compressed,
                                        mlm if any(compressor.name.startswith(p) for p in rr_methods) else None)
                                if compressor.name not in UNCOMPRESSED else {}),
                             "compressed": compressed})
    return rows


def reader_key(reader, prompt: str, max_new_tokens: int) -> str:
    """Greedy decoding is deterministic given (reader, system prompt, token cap, prompt)."""
    import hashlib
    import json

    blob = json.dumps([reader.name, reader.system_prompt, max_new_tokens, prompt], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def generate_cached(reader, prompts: list[str], max_new_tokens: int, cache_path: str | None = None) -> list[str]:
    """Call the reader once per distinct prompt. Identical (compressed context, question) pairs are
    common: the nested tier grid T1 ⊂ T1+T2 ⊂ T1+T2+T3 gives the same compression whenever a
    context has no T2/T3 word, and none/no_context recur in every dev stage. With `cache_path`
    (JSONL), outputs are also reused across runs; the key includes the reader, its system
    prompt and the token cap, so a change to any of them is a miss."""
    import json

    keys = [reader_key(reader, p, max_new_tokens) for p in prompts]
    known = {}
    if cache_path and os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                entry = json.loads(line)
                known[entry["key"]] = entry["output"]
    todo = list(dict.fromkeys(k for k in keys if k not in known))
    if todo:
        first = {}
        for k, p in zip(keys, prompts):
            first.setdefault(k, p)
        fresh = reader.generate([first[k] for k in todo], max_new_tokens)
        known.update(zip(todo, fresh))
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "a", encoding="utf-8") as f:
                for k, out in zip(todo, fresh):
                    f.write(json.dumps({"key": k, "output": out}, ensure_ascii=False) + "\n")
    print(f"reader calls: {len(todo)} for {len(prompts)} prompts "
          f"({len(set(keys))} distinct, {len(set(keys)) - len(todo)} from the cache)")
    return [known[k] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1 / 3, 0.2])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default=None,
                    help="config/test_exclusions.json: drop the test rows readers scored during development "
                         "before --limit (the teacher upper bound uses the first unseen rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "api"],
                    help="hf = plain transformers (smoke tests, no vLLM); api = OpenAI-compatible server")
    ap.add_argument("--base-url", default=None, help="for --backend api")
    ap.add_argument("--tokenizer", default=None,
                    help="HF id of the reader's tokenizer, if --reader is not a HF id (api backend)")
    ap.add_argument("--workers", type=int, default=16, help="parallel requests for --backend api")
    ap.add_argument("--compress-only", action="store_true",
                    help="write <out>.compressed.jsonl and exit (phase 1; run by the vllm backend itself)")
    ap.add_argument("--reader-cache", default=os.environ.get("VIWORD_READER_CACHE") or None,
                    help="JSONL cache of reader outputs keyed by (reader, system prompt, token cap, prompt), "
                         "shared across runs; identical prompts are always sent once per run")
    ap.add_argument("--rr-model", default=None,
                    help="MLM for harmful cuts (broken words it cannot restore), e.g. xlm-roberta-large")
    ap.add_argument("--rr-methods", nargs="*", default=[],
                    help="method-name prefixes that get harmful-cut counts (the syllable systems of the "
                         "H1 mechanism regression, e.g. llmlingua2vi)")
    ap.add_argument("--system-prompt", default=os.environ.get("READER_SYSTEM_PROMPT") or None,
                    help="optional system message for readers whose chat template expects one")
    args = ap.parse_args()

    data_rows = read_jsonl(args.data)
    if args.exclude:
        skip = excluded_ids(load_exclusions(args.exclude), args.data)
        data_rows = [r for r in data_rows if r["id"] not in skip]
    data_rows = data_rows[:args.limit]
    refuse_unseen_test_rows(data_rows)  # DATN §4.3: unseen test rows only in the eval stage
    examples = examples_from_rows(args.task, data_rows)
    contexts = load_contexts(examples, args.segmented)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cache_path = args.out.replace(".jsonl", "") + ".compressed.jsonl"
    if os.path.exists(cache_path):
        compressed_rows = read_jsonl(cache_path)
        print(f"reusing compressions from {cache_path}")
    elif args.backend == "vllm" and not args.compress_only:
        # Phase 1 in a child process: the compressors' models are not reliably freed in this
        # process, and vLLM needs nearly the whole GPU (on a 16 GB T4 it cannot start otherwise).
        subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--compress-only"],
                       check=True)
        compressed_rows = read_jsonl(cache_path)
    else:
        compressed_rows = compress_all(args, examples, contexts)
        write_jsonl(cache_path, compressed_rows)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        if args.compress_only:
            print(f"wrote compressions to {cache_path}")
            return

    # Caches created before the explicit adherence fields were added remain usable.
    # Recompute these fields so every output row has the same budget report whether
    # compression was just run or reused.
    for row in compressed_rows:
        actual_ratio = tokens_per_budget(row["n_tokens_compressed"], row["budget"])
        row.setdefault("budget_tokenizer", args.tokenizer or args.reader)
        row["tokens_per_budget"] = actual_ratio
        row["budget_deviation_5pct"] = budget_deviates(actual_ratio)

    by_id = {ex.id: ex for ex in examples}
    if args.backend == "vllm":
        reader = VLLMReader(args.reader, max_model_len=args.max_model_len)
    elif args.backend == "api":
        reader = APIReader(args.reader, args.base_url, args.tokenizer or args.reader, workers=args.workers)
    else:
        reader = HFReader(args.reader)
    reader.system_prompt = args.system_prompt
    task = examples[0].task
    prompts = [build_prompt(by_id[r["id"]], r["compressed"]) for r in compressed_rows]
    outputs = generate_cached(reader, prompts, MAX_NEW_TOKENS[task], args.reader_cache)

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
