"""Parse vbpl documents into Điều / Khoản / Điểm (direction B corpus).

  python scripts/build_legal_corpus.py                       # central documents, no translations
  python scripts/build_legal_corpus.py --scope all --limit 2000

Reads data/raw/vbpl/data/{metadata,content}.parquet (scripts/fetch_legal.py) and writes
  data/legal/vbpl_articles.jsonl   one row per article: document fields + the clause/point tree
  data/legal/vbpl_docs.jsonl       one row per document: parse counts and warnings
Documents with no article heading (decisions that are a single text, official letters, scans) keep a docs row
with n_articles = 0 and no article rows.
"""
import argparse
import json
import os
from collections import Counter

DOC_FIELDS = ("id", "so_ky_hieu", "loai_van_ban", "title", "ngay_ban_hanh", "ngay_co_hieu_luc",
              "tinh_trang_hieu_luc", "co_quan_ban_hanh")


def article_row(meta: dict, doc, a) -> dict:
    return {"doc_id": meta["id"], **{k: meta.get(k) for k in DOC_FIELDS if k != "id"},
            "attachment": a.attachment, "attachment_title": doc.attachments[a.attachment - 1].title if a.attachment else "",
            "article": a.number, "article_title": a.title, "part": a.part, "intro": a.intro,
            "clauses": [{"n": c.number, "text": c.text, "points": [{"l": p.letter, "text": p.text} for p in c.points]}
                        for c in a.clauses],
            "points": [{"l": p.letter, "text": p.text} for p in a.points]}


def main() -> None:
    import pyarrow.parquet as pq
    from tqdm import tqdm

    from viword.legal import html_lines, parse_lines

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/vbpl/data")
    ap.add_argument("--out", default="data/legal")
    ap.add_argument("--scope", choices=["central", "all"], default="central",
                    help="central = pham_vi 'Trung ương' or 'Toàn quốc' (laws, decrees, circulars, …; the 2024–2025 "
                         "laws are labelled 'Toàn quốc'); all adds local documents")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = {m["id"]: m for m in pq.read_table(os.path.join(args.raw, "metadata.parquet")).to_pylist()
            if m["loai_van_ban"] != "Bản dịch văn bản"
            and (args.scope == "all" or m["pham_vi"] in ("Trung ương", "Toàn quốc"))}
    os.makedirs(args.out, exist_ok=True)
    pf = pq.ParquetFile(os.path.join(args.raw, "content.parquet"))
    totals, warn_kinds, done = Counter(), Counter(), 0
    with open(os.path.join(args.out, "vbpl_articles.jsonl"), "w", encoding="utf-8") as fa, \
            open(os.path.join(args.out, "vbpl_docs.jsonl"), "w", encoding="utf-8") as fd, \
            tqdm(total=args.limit or len(meta), desc="documents") as bar:
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g)
            for doc_id, html in zip(t.column("id").to_pylist(), t.column("content_html").to_pylist()):
                m = meta.get(doc_id)
                if m is None or not html:
                    continue
                lines = html_lines(html)
                doc = parse_lines(lines)
                n_clauses = sum(len(a.clauses) for a in doc.articles)
                n_points = sum(len(c.points) for a in doc.articles for c in a.clauses) + \
                    sum(len(a.points) for a in doc.articles)
                fd.write(json.dumps({"doc_id": doc_id, **{k: m.get(k) for k in DOC_FIELDS if k != "id"},
                                     "n_lines": len(lines), "n_articles": len(doc.articles), "n_clauses": n_clauses,
                                     "n_points": n_points, "n_attachments": len(doc.attachments),
                                     "n_preamble": len(doc.preamble), "n_closing": len(doc.closing),
                                     "warnings": doc.warnings}, ensure_ascii=False) + "\n")
                for a in doc.articles:
                    fa.write(json.dumps(article_row(m, doc, a), ensure_ascii=False) + "\n")
                totals.update(docs=1, with_articles=bool(doc.articles), articles=len(doc.articles),
                              clauses=n_clauses, points=n_points, with_warnings=bool(doc.warnings),
                              with_attachments=bool(doc.attachments))
                warn_kinds.update(w.split(" ")[0] for w in doc.warnings)
                done += 1
                bar.update()
                if args.limit and done >= args.limit:
                    break
            if args.limit and done >= args.limit:
                break
    print(json.dumps({**totals, "warning_kinds": dict(warn_kinds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
