"""Query the teacher LLM once per (text, rate), in parallel, with resume.

Used twice:
  * distillation inputs:  --input data/distill/paragraphs.jsonl --field text
  * teacher upper bound / gate G4 on task dev/test subsets:
                          --input data/tasks/vinli_dev.jsonl --field premise --limit 200
    (the output is read by the `precomputed:` evaluation method)

Output lines: {"id", "ratio", "compressed", "model"}. Rows already in the output file are
skipped, so an interrupted run can simply be restarted with the same command. A file with
rows from another teacher, or rows without a teacher name (written before this field
existed), is refused: resuming into it would mix two teachers' labels.

The endpoint, model and API keys come from the environment or from `.env` (see .env.example):
TEACHER_URL, TEACHER_MODEL, and TEACHER_API_KEY or TEACHER_API_KEY_1 ... _N. With several keys,
calls are spread over them in turn, each key with its own rate budget. Keys in .env are only
sent to the TEACHER_URL written in .env, never to another endpoint given by the shell or --base-url.

Usage:
  python scripts/teacher_compress.py --input data/distill/paragraphs.jsonl --field text \
      --output data/distill/teacher.jsonl --rates 0.5 0.25
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from viword.data import read_jsonl
from viword.distill import TEACHER_PROMPTS, teacher_compress
from viword.heldout import excluded_ids, load_exclusions


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Set `KEY=value` lines of `path` that are not already in the environment, and return all
    lines read. Comments, `export ` and quotes are understood; nothing is expanded or executed."""
    loaded = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.removeprefix("export ").partition("=")
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            loaded[name] = value
            os.environ.setdefault(name, value)
    return loaded


def api_keys(env=os.environ) -> list[str]:
    """TEACHER_API_KEY, TEACHER_API_KEY_1 ... _32, then OPENAI_API_KEY; placeholders like
    `<key-1>` and duplicates (one rate budget under two names) are dropped. A self-hosted
    vLLM needs no key: ["EMPTY"]."""
    names = ["TEACHER_API_KEY"] + [f"TEACHER_API_KEY_{i}" for i in range(1, 33)] + ["OPENAI_API_KEY"]
    keys = []
    for name in names:
        value = (env.get(name) or "").strip()
        if value and not value.startswith("<") and value not in keys:
            keys.append(value)
    return keys or ["EMPTY"]


def done_rows(path: str, model: str, prompt: str = "default") -> set[tuple[str, float]]:
    """(id, ratio) already in `path`; raises SystemExit if any row is not from `model` and `prompt`."""
    if not os.path.exists(path):
        return set()
    rows = read_jsonl(path)
    prompts = sorted({r.get("prompt", "default") for r in rows} - {prompt})
    if prompts:
        raise SystemExit(f"{path} has rows made with another teacher prompt ({', '.join(prompts)}), "
                         f"not only {prompt}: pass a new --output")
    others = sorted({str(r.get("model")) for r in rows if r.get("model") != model})
    if others:
        raise SystemExit(f"{path} has rows from another teacher ({', '.join(others)}; None = no teacher "
                         f"name, written before it was recorded), not only {model}: move the file away "
                         "or pass a new --output")
    return {(r["id"], r["ratio"]) for r in rows}


def key_env(dotenv: dict[str, str], base_url: str) -> dict[str, str]:
    """The environment to take API keys from. Keys in .env belong to the TEACHER_URL in .env:
    when another endpoint is used, only keys set outside .env are sent to it."""
    if dotenv.get("TEACHER_URL", base_url) == base_url:
        return dict(os.environ)
    return {k: v for k, v in os.environ.items() if k not in dotenv}


def main():
    dotenv = load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--field", default="text", help="text | context | premise")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 0.25])
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_URL"))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default=None,
                    help="config/test_exclusions.json: skip the test rows readers scored during development "
                         "before --limit, as evaluate.py --exclude does for the teacher upper bound")
    ap.add_argument("--prompt", default="default", choices=sorted(TEACHER_PROMPTS),
                    help="teacher instruction; 'neutral' omits the keep-facts/numbers/names/negation line")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="output cap per call; a call that reaches it fails instead of keeping a cut-off text")
    args = ap.parse_args()
    if not args.base_url or not args.model:
        ap.error("set TEACHER_URL and TEACHER_MODEL (in .env or the environment) or pass --base-url/--model")

    done = done_rows(args.output, args.model, args.prompt)
    from openai import OpenAI
    # the SDK retries 429 and 5xx itself, with backoff and honouring Retry-After
    clients = [OpenAI(base_url=args.base_url, api_key=key, max_retries=8, timeout=600) for key in api_keys(key_env(dotenv, args.base_url))]
    print(f"teacher {args.model} at {args.base_url}, {len(clients)} API key(s)")

    # identical texts (e.g. one ViMMRC passage with several questions) are compressed once
    ids_of_text = {}
    rows = read_jsonl(args.input)
    if args.exclude:
        skip = excluded_ids(load_exclusions(args.exclude), args.input)
        rows = [r for r in rows if r["id"] not in skip]
    for r in rows[:args.limit]:
        ids_of_text.setdefault(r[args.field], []).append(r["id"])
    jobs = [(ids, text, rate) for text, ids in ids_of_text.items() for rate in args.rates
            if any((i, rate) not in done for i in ids)]
    print(f"{len(done)} rows already done, {len(jobs)} teacher calls to go")

    lock = threading.Lock()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(teacher_compress, clients[n % len(clients)], args.model, text, rate,
                               args.max_tokens, args.prompt): (ids, rate)
                   for n, (ids, text, rate) in enumerate(jobs)}
        failed = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            ids, rate = futures[future]
            try:
                compressed = future.result()
            except Exception as e:  # keep going; failed rows are retried on the next run
                failed += 1
                print(f"failed {ids[0]} @ {rate}: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                for i in ids:
                    if (i, rate) not in done:
                        row = {"id": i, "ratio": rate, "compressed": compressed, "model": args.model}
                        if args.prompt != "default":
                            row["prompt"] = args.prompt
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print(f"done; {failed} failed (rerun the same command to retry them)")


if __name__ == "__main__":
    main()
