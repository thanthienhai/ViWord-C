"""Query the teacher LLM once per (text, rate), in parallel, with resume.

Used twice:
  * distillation inputs:  --input data/distill/paragraphs.jsonl --field text
  * teacher upper bound / gate G4 on task dev sets:
                          --input data/tasks/vinli_dev.jsonl --field premise --limit 200
    (the output is read by the `precomputed:` evaluation method)

Output lines: {"id", "ratio", "compressed"}. Rows already in the output file are skipped,
so an interrupted run can simply be restarted with the same command.

Usage:
  python scripts/teacher_compress.py --input data/distill/paragraphs.jsonl --field text \
      --output data/distill/teacher.jsonl --rates 0.5 0.25 \
      --base-url http://172.16.9.11:30048/v1 --model icmodel/icom-model-llm-ic-v3.8-27b
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from viword.data import read_jsonl
from viword.distill import teacher_compress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--field", default="text", help="text | context | premise")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 0.25])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5, timeout=600)

    done = set()
    if os.path.exists(args.output):
        done = {(r["id"], r["ratio"]) for r in read_jsonl(args.output)}
    # identical texts (e.g. one ViMMRC passage with several questions) are compressed once
    ids_of_text = {}
    for r in read_jsonl(args.input)[:args.limit]:
        ids_of_text.setdefault(r[args.field], []).append(r["id"])
    jobs = [(ids, text, rate) for text, ids in ids_of_text.items() for rate in args.rates
            if any((i, rate) not in done for i in ids)]
    print(f"{len(done)} rows already done, {len(jobs)} teacher calls to go")

    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(teacher_compress, client, args.model, text, rate): (ids, rate)
                   for ids, text, rate in jobs}
        failed = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            ids, rate = futures[future]
            try:
                compressed = future.result()
            except Exception as e:  # keep going; failed rows are retried on the next run
                failed += 1
                print(f"failed {ids[0]} @ {rate}: {type(e).__name__}: {e}")
                continue
            with lock:
                for i in ids:
                    if (i, rate) not in done:
                        row = {"id": i, "ratio": rate, "compressed": compressed}
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print(f"done; {failed} failed (rerun the same command to retry them)")


if __name__ == "__main__":
    main()
