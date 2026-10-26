"""How many VLegal-Bench citations the parsed vbpl corpus can serve (direction B, gate G0 input).

  python scripts/check_legal_citations.py

For task 3.1 (gold option and the three distractors) and task 3.3 (the articles listed in `content`), each
citation is parsed, its document resolved through the vbpl metadata, and looked up in
data/legal/vbpl_articles.jsonl: document parsed into articles → cited article → cited clause → cited point.
The document's own text is searched first; a circular or decision that issues a regulation (Quy chế, Quy định)
is cited by the regulation's article numbers, so its first attached text is the fallback. A missing level is
reported with an example.
"""
import argparse
import json
import re
from collections import Counter, defaultdict


def load_index(path: str) -> dict:
    """(doc_id, attachment, article) → {clause number: [point letters]}, plus None → points of the article."""
    index = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["attachment"] > 1:
                continue
            entry = index[r["doc_id"], r["attachment"], r["article"]]
            for c in r["clauses"]:
                entry[c["n"]] = [p["l"] for p in c["points"]]
            entry[None] = [p["l"] for p in r["points"]]
    return index


def citations(task_dir: str):
    """(task, role, citation text) for 3.1 options and 3.3 content articles."""
    for line in open(f"{task_dir}/3.1/3_1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for opt in r["answers"].split("\n"):
            letter, _, text = opt.partition(": ")
            yield "3.1", "gold" if letter.strip() == r["ground_truth"].strip() else "distractor", text.strip()
    for line in open(f"{task_dir}/3.3/3_3.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for m in re.finditer(r"^- (Điều \d+[a-zđ]? văn bản [^:]+):", r.get("content") or "", re.M):
            yield "3.3", "content", m.group(1)


def main() -> None:
    import pyarrow.parquet as pq

    from viword.legal import Resolver, parse_citation

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/raw/vlegal_bench")
    ap.add_argument("--metadata", default="data/raw/vbpl/data/metadata.parquet")
    ap.add_argument("--articles", default="data/legal/vbpl_articles.jsonl")
    args = ap.parse_args()
    resolver = Resolver(pq.read_table(args.metadata).to_pylist())
    index = load_index(args.articles)
    parsed_docs = {d for d, _, _ in index}
    stages = ("parsed", "resolved", "doc_parsed", "article", "clause", "point")
    counts, examples = defaultdict(Counter), defaultdict(dict)
    for task, role, text in citations(args.tasks):
        key = (task, role)
        counts[key]["total"] += 1
        cit = parse_citation(text)
        reached = []
        if cit:
            reached.append("parsed")
            meta = resolver.resolve(cit)
            if meta:
                reached.append("resolved")
                if meta["id"] in parsed_docs:
                    reached.append("doc_parsed")
                    entry = index.get((meta["id"], 0, cit.article))
                    if entry is None or (cit.clause is not None and cit.clause not in entry):
                        entry = index.get((meta["id"], 1, cit.article), entry)
                    if entry is not None:
                        reached.append("article")
                        if cit.clause is None or cit.clause in entry:
                            reached.append("clause")
                            points = entry.get(cit.clause, [])
                            if cit.point is None or cit.point in points:
                                reached.append("point")
        for s in reached:
            counts[key][s] += 1
        missing = next((s for s in stages if s not in reached), None)
        if missing:
            examples[key].setdefault(missing, text)
    print("| task | role | citations | " + " | ".join(stages) + " |\n|---|---|---|" + "---|" * len(stages))
    for key in sorted(counts):
        c = counts[key]
        print(f"| {key[0]} | {key[1]} | {c['total']} | " + " | ".join(f"{c[s]} ({c[s] / c['total']:.0%})" for s in stages)
              + " |")
    print("\nFirst citation lost at each stage:")
    for key in sorted(examples):
        for stage, text in examples[key].items():
            print(f"- {key[0]} {key[1]}, not {stage}: {text}")


if __name__ == "__main__":
    main()
