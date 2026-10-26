"""VLegal-Bench questions as direction-B tasks, split by gold document (no context yet; see legal_rag.py).

  python scripts/make_legal_tasks.py            # data/tasks/vlegal_dev.jsonl and vlegal_test.jsonl

Kept: task 3.1 (which clause supports the answer; the gold option is a clause citation) and task 3.3 (a
scenario; the benchmark lists the articles it rests on). A question is kept only if its gold provisions are in
the parsed corpus (scripts/check_legal_citations.py): 3.1 needs the cited clause (or article), 3.3 at least one
listed article. The provisions that 3.3 pastes into its question are cut off: the context comes from retrieval.
Rows: id, cluster (= document of the first gold provision, so that dev and test share no gold document), question,
choices (4), answer (A–D), source_task, gold [{doc_id, attachment, article, clause}], gold_cite.
The split is by cluster with a fixed seed; vlegal_test is held out (viword.heldout.TEST_FILES).
"""
import argparse
import json
import random
import re
from collections import defaultdict


def options(text: str) -> tuple[list[str], list[str]]:
    """'A: …\\nB: …' → letters, texts."""
    letters, texts = [], []
    for line in text.split("\n"):
        m = re.match(r"^\s*([A-D])\s*[:.]\s*(.*)$", line)
        if m:
            letters.append(m.group(1))
            texts.append(m.group(2).strip())
        elif texts:
            texts[-1] += " " + line.strip()
    return letters, texts


def load_articles(path: str, doc_ids: set) -> dict:
    """(doc_id, attachment, article) → set of clause numbers (0 = intro) for the needed documents."""
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["doc_id"] not in doc_ids or r["attachment"] > 1:
                continue
            index[r["doc_id"], r["attachment"], r["article"]] = {c["n"] for c in r["clauses"]} | {0}
    return index


def locate(cit, meta, index) -> dict | None:
    """The gold provision in the corpus: the document's own text first, then its first attached text."""
    for att in (0, 1):
        clauses = index.get((meta["id"], att, cit.article))
        if clauses is not None and (cit.clause is None or cit.clause in clauses):
            return {"doc_id": meta["id"], "attachment": att, "article": cit.article, "clause": cit.clause}
    return None


def main() -> None:
    import pyarrow.parquet as pq

    from viword.legal import Resolver, parse_citation

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/raw/vlegal_bench")
    ap.add_argument("--metadata", default="data/raw/vbpl/data/metadata.parquet")
    ap.add_argument("--articles", default="data/legal/vbpl_articles.jsonl")
    ap.add_argument("--out", default="data/tasks")
    ap.add_argument("--dev-share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    resolver = Resolver(pq.read_table(args.metadata).to_pylist())
    raw = []
    for line in open(f"{args.bench}/3.1/3_1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        letters, texts = options(r["answers"])
        gold_text = texts[letters.index(r["ground_truth"].strip())]
        raw.append(("3.1", len(raw), r["question"].strip(), letters, texts, r["ground_truth"].strip(), [gold_text]))
    for line in open(f"{args.bench}/3.3/3_3.jsonl", encoding="utf-8"):
        r = json.loads(line)
        letters, texts = options(r["answers"])
        scenario = r["question"].split("-------")[-1]
        scenario = re.sub(r"^\s*Hãy trả lời câu hỏi sau dựa trên nội dung điều luật đề cập ở trên:\s*", "", scenario)
        cites = re.findall(r"^- (Điều \d+[a-zđ]? văn bản [^:]+):", r.get("content") or "", re.M)
        raw.append(("3.3", r["id"], scenario.strip(), letters, texts, r["ground_truth"].strip(), cites))

    resolved = []
    for task, qid, question, letters, texts, answer, cites in raw:
        pairs = [(c, resolver.resolve(c)) for c in map(parse_citation, cites) if c is not None]
        resolved.append((task, qid, question, letters, texts, answer, cites, [(c, m) for c, m in pairs if m]))
    index = load_articles(args.articles, {m["id"] for *_, pairs in resolved for _, m in pairs})

    rows, dropped = [], defaultdict(int)
    for task, qid, question, letters, texts, answer, cites, pairs in resolved:
        gold = [g for g in (locate(c, m, index) for c, m in pairs) if g]
        if letters != ["A", "B", "C", "D"] or answer not in letters:
            dropped[f"{task} options"] += 1
            continue
        if not gold:
            dropped[f"{task} gold not in corpus"] += 1
            continue
        rows.append({"id": f"vlb{task}-{qid}", "cluster": gold[0]["doc_id"], "question": question, "choices": texts,
                     "answer": answer, "source_task": task, "gold": gold, "gold_cite": cites})

    clusters = sorted({r["cluster"] for r in rows})
    random.Random(args.seed).shuffle(clusters)
    dev_clusters, n = set(), 0
    for c in clusters:  # take whole clusters until the dev share of questions is reached
        if n >= args.dev_share * len(rows):
            break
        dev_clusters.add(c)
        n += sum(r["cluster"] == c for r in rows)
    for split, keep in (("dev", True), ("test", False)):
        part = [r for r in rows if (r["cluster"] in dev_clusters) == keep]
        with open(f"{args.out}/vlegal_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_task = defaultdict(int)
        for r in part:
            by_task[r["source_task"]] += 1
        print(f"vlegal_{split}: {len(part)} questions {dict(by_task)}, {len({r['cluster'] for r in part})} documents")
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()
