"""Direction B retrieval: clause units, a BM25 index, gold recall, and RAG contexts for the legal QA tasks.

  python scripts/legal_rag.py units      # data/legal/units.jsonl from data/legal/vbpl_articles.jsonl
  python scripts/legal_rag.py index [--dataset vibid]   # data/legal/bm25.npz + bm25_vocab.json (bm25_vibid.*)
  python scripts/legal_rag.py recall --split dev [--dataset vibid]
  python scripts/legal_rag.py contexts --split dev [--dataset vibid] [--mode retrieved|gold|random|oracle]
      [--budget 3000]

Datasets (`DATASETS`): `vlegal` (VLegal-Bench 3.1/3.3, scripts/make_legal_tasks.py) retrieves from the whole
corpus; `vibid` (ViBidLQA, scripts/make_vibid_tasks.py) only from the units of the Bidding Law 22/2023/QH15 and
Decree 24/2024/NĐ-CP, so that every distractor comes from the same two documents as the gold, and its gold units
come with the task (the units of the gold article that contain the answer).

A unit is one clause (with its points), an article's lead-in, or an article without clauses, headed by the
document label and the article heading so that a reader can cite it. Retrieval uses the question alone (for
3.3, the scenario): the options are citations or answers, and searching them would hand the retriever the gold.
`contexts` fills each question's context with the best units, best first, up to `budget` reader tokens; mode
`gold` puts the missing gold units in as one block at a seeded random position (the last retrieved units leave
to keep the budget), which separates retrieval failures from compression; mode `random` does the same but first
takes the retrieved gold units out of their ranks too (with ViBidLQA, BM25 ranks the gold first for 75% of the
questions, so in `gold` mode a selection by position alone would look strong); mode `oracle` keeps the gold units
alone. A cited article counts with its first 8 units, in clause order; the evidence keeps the first unit of each
gold provision and adds further ones while they fit in 80% of the budget. Rows from the test split carry
derived_from = "vlegal_test", so only the eval stage may let a reader score them (viword.heldout); vibid rows carry
"vibid_test" the same way.
"""
import argparse
import json
import random
from collections import defaultdict

UNIT_CHARS = 4000  # longer units (tables, annex lists) are cut; such a cut is flagged
MAX_GOLD_UNITS = 8  # a cited article brings at most its first 8 units into the gold and oracle contexts
# docs: the documents retrieval may draw from (None = all); fields: what a context row copies from its question
# (when the question has it: heldout_article exists only in vibid_train)
DATASETS = {
    "vlegal": {"docs": None, "index": "data/legal/bm25", "fields": ("choices", "answer", "source_task")},
    "vibid": {"docs": ("166581", "166583"), "index": "data/legal/bm25_vibid",
              "fields": ("answers", "abstractive_answer", "source_task", "heldout_article")},
}


def uid(doc_id: str, attachment: int, article: str, clause) -> str:
    return f"{doc_id}|{attachment}|{article}|{'' if clause is None else clause}"


def in_order(unit_ids) -> list[str]:
    """Units of one article in clause order (lead-in 0 first; an article without clauses has one unit)."""
    return sorted(unit_ids, key=lambda u: int(u.rsplit("|", 1)[1] or 0))


def cmd_units(args) -> None:
    from viword.legal import article_from_row, article_units, doc_label

    n = cut = 0
    with open(args.articles, encoding="utf-8") as fi, open(args.units, "w", encoding="utf-8") as fo:
        for line in fi:
            r = json.loads(line)
            label = doc_label(r["loai_van_ban"], r["title"] or "", r["so_ky_hieu"] or "")
            for u in article_units(article_from_row(r), label, r.get("attachment_title", "")):
                text = u.text if len(u.text) <= UNIT_CHARS else u.text[:UNIT_CHARS]
                cut += len(text) < len(u.text)
                fo.write(json.dumps({"uid": uid(r["doc_id"], u.attachment, u.article, u.clause), "doc_id": r["doc_id"],
                                     "label": label, "in_force": r["tinh_trang_hieu_luc"], "text": text,
                                     "cut": len(text) < len(u.text)}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} units ({cut} cut at {UNIT_CHARS} characters) in {args.units}")


def read_units(path: str, docs=None) -> tuple[list[str], list[str]]:
    """Unit ids and texts, of the documents in `docs` only if it is given."""
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            if docs is None or u["doc_id"] in docs:
                ids.append(u["uid"])
                texts.append(u["text"])
    return ids, texts


def cmd_index(args) -> None:
    from viword.legal_index import BM25

    ids, texts = read_units(args.units, DATASETS[args.dataset]["docs"])
    bm25 = BM25.fit(texts)
    bm25.save(args.index)
    print(f"BM25 over {len(ids)} units, {len(bm25.vocab)} terms, {bm25.weights.nnz} weights → {args.index}.npz")


def gold_units(question: dict, by_article: dict) -> list[set[str]]:
    """One set of unit ids per gold provision: the cited clause, or every unit of a cited article (or the sets
    the task gives in `gold_units`)."""
    if "gold_units" in question:
        return [set(g) for g in question["gold_units"] if g]
    out = []
    for g in question["gold"]:
        article = by_article.get((g["doc_id"], g["attachment"], g["article"]), [])
        if g["clause"] is None:
            out.append(set(article))
        else:
            out.append({u for u in article if u.endswith(f"|{g['clause']}")})
    return [s for s in out if s]


def load_split(args) -> tuple[list[dict], list[str], dict, object]:
    from viword.legal_index import BM25

    questions = [json.loads(line) for line in open(f"{args.tasks}/{args.dataset}_{args.split}.jsonl",
                                                     encoding="utf-8")]
    ids, _ = read_units(args.units, DATASETS[args.dataset]["docs"])
    by_article = defaultdict(list)
    for u in ids:
        doc, att, art, _ = u.split("|")
        by_article[doc, int(att), art].append(u)
    return questions, ids, by_article, BM25.load(args.index)


def cmd_recall(args) -> None:
    questions, ids, by_article, bm25 = load_split(args)
    ks = (1, 5, 10, 20, 50, 100)
    hits = defaultdict(lambda: defaultdict(list))
    for q in questions:
        golds = gold_units(q, by_article)
        top, _ = bm25.search(q["question"], max(ks))
        ranked = [ids[i] for i in top]
        for k in ks:
            found = [bool(g & set(ranked[:k])) for g in golds]
            hits[q.get("source_task", args.dataset)][f"any@{k}"].append(any(found))
            hits[q.get("source_task", args.dataset)][f"all@{k}"].append(all(found))
    print(f"| task | questions | " + " | ".join(f"any@{k}" for k in ks) + " | all@10 | all@100 |")
    print("|---|---|" + "---|" * (len(ks) + 2))
    for task, h in sorted(hits.items()):
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        print(f"| {task} | {len(h['any@1'])} | " + " | ".join(f"{mean(h[f'any@{k}']):.3f}" for k in ks)
              + f" | {mean(h['all@10']):.3f} | {mean(h['all@100']):.3f} |")


def cmd_contexts(args) -> None:
    from transformers import AutoTokenizer

    questions, ids, by_article, bm25 = load_split(args)
    text_of = dict(zip(*read_units(args.units, DATASETS[args.dataset]["docs"])))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    rng = random.Random(args.seed)
    suffix = "" if args.mode == "retrieved" else f"_{args.mode}"
    out_path = f"{args.tasks}/{args.dataset}_rag{suffix}_{args.split}.jsonl"
    stats = defaultdict(list)
    with open(out_path, "w", encoding="utf-8") as fo:
        for q in questions:
            golds = gold_units(q, by_article)
            top, _ = bm25.search(q["question"], 200)
            ranked = [ids[i] for i in top]
            chosen, used = [], 0
            for u in ranked:
                n = count(text_of[u]) + 2
                if used + n > args.budget and chosen:
                    break
                chosen.append(u)
                used += n
            # the first unit of every gold provision, then their further units while they fit in 80% of the budget
            groups = [in_order(g)[:MAX_GOLD_UNITS] for g in golds]
            evidence = [g[0] for g in groups]
            room = 0.8 * args.budget - sum(count(text_of[u]) + 2 for u in evidence)
            for g in groups:
                for u in g[1:]:
                    n = count(text_of[u]) + 2
                    if n > room:
                        break
                    evidence.insert(evidence.index(g[0]) + g.index(u), u)
                    room -= n
            if args.mode == "oracle":  # the gold units alone: the ceiling of any selection at this retriever
                chosen = list(dict.fromkeys(evidence))
                used = sum(count(text_of[u]) + 2 for u in chosen)
            if args.mode == "random":  # every gold unit leaves its retrieved rank, so its position tells nothing
                for u in [u for u in chosen if any(u in g for g in golds)]:
                    chosen.remove(u)
                    used -= count(text_of[u]) + 2
            if args.mode in ("gold", "random"):
                # the missing gold units, in their own order, as one block at a seeded random position; retrieved
                # units leave from the end to keep the budget
                missing = [u for u in dict.fromkeys(evidence) if u not in chosen]
                need = sum(count(text_of[u]) + 2 for u in missing)
                while chosen and used + need > args.budget and chosen[-1] not in evidence:
                    used -= count(text_of[chosen.pop()]) + 2
                at = rng.randrange(len(chosen) + 1)
                chosen[at:at] = missing
                used += need
            gold_pos = [i for i, u in enumerate(chosen) if any(u in g for g in golds)]
            stats["units"].append(len(chosen))
            stats["tokens"].append(used)
            stats["gold_in"].append(bool(gold_pos))
            row = {"id": q["id"], "cluster": q["cluster"], "context": "\n\n".join(text_of[u] for u in chosen),
                   "question": q["question"], **{k: q[k] for k in DATASETS[args.dataset]["fields"] if k in q},
                   "units": chosen, "gold_positions": gold_pos, "derived_from": f"{args.dataset}_{args.split}"}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"{out_path}: {len(questions)} rows, {mean(stats['units']):.1f} units and {mean(stats['tokens']):.0f} "
          f"reader tokens per context, gold in context for {mean(stats['gold_in']):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["units", "index", "recall", "contexts"])
    ap.add_argument("--articles", default="data/legal/vbpl_articles.jsonl")
    ap.add_argument("--units", default="data/legal/units.jsonl")
    ap.add_argument("--dataset", default="vlegal", choices=sorted(DATASETS))
    ap.add_argument("--index", default=None, help="index prefix (default: the dataset's)")
    ap.add_argument("--tasks", default="data/tasks")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--mode", default="retrieved", choices=["retrieved", "gold", "random", "oracle"])
    ap.add_argument("--budget", type=int, default=3000, help="reader tokens per context")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.index = args.index or DATASETS[args.dataset]["index"]
    {"units": cmd_units, "index": cmd_index, "recall": cmd_recall, "contexts": cmd_contexts}[args.command](args)


if __name__ == "__main__":
    main()
