"""Query Semantic Scholar for the DATN.md literature search; writes results.json (resumable)."""
import json, os, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "results.json")
FIELDS = "title,year,venue,citationCount,externalIds,abstract,tldr"
API = "https://api.semanticscholar.org/graph/v1"

QUERIES = {
    "A": ["prompt compression large language models",
          "context compression LLM token pruning",
          "task-agnostic prompt compression",
          "extractive prompt compression token classification",
          "long context compression small language model perplexity"],
    "B": ["prompt compression word boundary incomplete words",
          "linguistically informed prompt compression",
          "syntax-aware parse tree prompt compression",
          "phrase-level word-level prompt compression"],
    "C": ["negation robustness large language models natural language inference",
          "negation natural language inference benchmark analysis",
          "function words importance language models",
          "information preservation prompt compression evaluation"],
    "D": ["tokenizer fertility multilingual language models",
          "tokenization disparity across languages cost",
          "token cost non-English languages LLM API",
          "vocabulary expansion Vietnamese LLM tokenizer"],
    "E": ["Vietnamese prompt compression",
          "Vietnamese context compression large language model",
          "Vietnamese large language model inference efficiency"],
    "F": ["Chinese prompt compression word segmentation",
          "multilingual prompt compression"],
    "G": ["VnCoreNLP Vietnamese NLP toolkit",
          "Vietnamese word segmentation",
          "ViNLI Vietnamese natural language inference",
          "Vietnamese multiple-choice reading comprehension",
          "Vietnamese abstractive summarization dataset news",
          "PhoBERT pre-trained language models Vietnamese"],
}
# seed papers: fetch citations (who built on them) filtered by keywords
SEEDS = {"LLMLingua-2": "arXiv:2403.12968", "LLMLingua": "arXiv:2310.05736"}


def get(url, tries=12):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "datn-lit"}), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                print("HTTP", e.code, url, file=sys.stderr); return None
        except Exception as e:
            print("ERR", e, file=sys.stderr)
        time.sleep(min(3 * (i + 1), 20))
    return None


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for group, qs in QUERIES.items():
        for q in qs:
            key = f"{group}|{q}"
            if key in res: continue
            d = get(f"{API}/paper/search?query={urllib.parse.quote(q)}&limit=15&fields={FIELDS}")
            if d is None: continue
            res[key] = d.get("data", [])
            print(key, len(res[key]), flush=True)
            json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
            time.sleep(1.5)
    for name, pid in SEEDS.items():
        key = f"CITES|{name}"
        if key in res: continue
        papers, off = [], 0
        while off < 1000:
            d = get(f"{API}/paper/{pid}/citations?fields=title,year,venue,citationCount,externalIds,abstract&limit=100&offset={off}")
            if not d or not d.get("data"): break
            papers += [x["citingPaper"] for x in d["data"]]
            off += 100
            time.sleep(1.5)
            if "next" not in d: break
        res[key] = papers
        print(key, len(papers), flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
