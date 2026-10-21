"""BERTScore (PhoBERT) for the VietNews summaries (DATN §4.1), computed after evaluate.py.

PhoBERT was pre-trained on word-segmented text (RDRSegmenter, multi-syllable words joined
by "_"), so both the reader's summary and the reference are segmented with VnCoreNLP
before scoring. The score uses `bert_score` with `vinai/phobert-base` and layer 10
(`--num-layers`; bert_score has no default layer for PhoBERT, and 10 is its default for
roberta-base, PhoBERT's architecture), without baseline rescaling. PhoBERT reads at most
256 subwords, so longer texts are truncated; the share of truncated rows is reported.

The script adds `bertscore_f1` to every VietNews row and writes a new JSONL; analyze.py
reads the main metric (ROUGE-L) as before, BERTScore is reported next to it.

Usage:
  python scripts/bertscore.py --rows results/eval/vietnews_Qwen2.5-7B-Instruct.jsonl \
      --data data/tasks/vietnews_test.jsonl --out results/eval/vietnews_Qwen2.5-7B-Instruct.bertscore.jsonl
"""
import argparse

from viword.data import read_jsonl, write_jsonl

MAX_SUBWORDS = 256


def cap_phobert_length(tokenizer):
    """bert_score cuts texts at `tokenizer.model_max_length`, which transformers 5 no longer sets to 256 for PhoBERT
    (it is 1e30): a longer summary then overruns the 258 position embeddings and kills the whole file."""
    tokenizer.model_max_length = MAX_SUBWORDS
    return tokenizer


def phobert_input(segmenter, text: str) -> str:
    """Word-segmented form expected by PhoBERT: syllables of a word joined by "_"."""
    return " ".join(w.text.replace(" ", "_") for sentence in segmenter.segment(text) for w in sentence)


def add_bertscore(rows: list[dict], references: dict[str, str], segmenter, score_fn) -> list[dict]:
    """`score_fn(candidates, references) -> list[float]` (F1); injectable for tests."""
    news = [r for r in rows if r["task"] == "vietnews"]
    cache: dict[str, str] = {}

    def seg(text):
        if text not in cache:
            cache[text] = phobert_input(segmenter, text) if text.strip() else ""
        return cache[text]

    candidates = [seg(r.get("raw_output") or r.get("prediction") or "") for r in news]
    refs = [seg(references[r["id"]]) for r in news]
    for r, f1 in zip(news, score_fn(candidates, refs)):
        r["bertscore_f1"] = float(f1)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--data", required=True, help="VietNews task file with the reference summaries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--num-layers", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from bert_score import BERTScorer

    from viword.segment import get_segmenter

    scorer = BERTScorer(model_type=args.model, num_layers=args.num_layers, batch_size=args.batch_size)
    tokenizer = cap_phobert_length(scorer._tokenizer)  # the tokenizer bert_score encodes with
    truncated = 0

    def score_fn(candidates, refs):
        nonlocal truncated
        truncated = sum(len(tokenizer.encode(t)) > MAX_SUBWORDS for t in candidates + refs)
        return scorer.score(candidates, refs)[2].tolist()

    references = {r["id"]: r["reference"] for r in read_jsonl(args.data)}
    rows = add_bertscore(read_jsonl(args.rows), references, get_segmenter("vncorenlp"), score_fn)
    write_jsonl(args.out, rows)
    n = sum(r["task"] == "vietnews" for r in rows)
    print(f"wrote {args.out}: BERTScore for {n} rows; {truncated} of {2 * n} texts longer than "
          f"{MAX_SUBWORDS} PhoBERT subwords (truncated)\n")
    print(summary_table(rows))


def summary_table(rows: list[dict]) -> str:
    """Mean BERTScore F1 per reader × method × ratio, with a cluster-bootstrap 95% CI."""
    import numpy as np

    from viword.stats import cluster_bootstrap

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if "bertscore_f1" in r:
            groups.setdefault((r["reader"], r["method"], r["ratio"]), []).append(r)
    lines = ["| reader | method | ratio | n | BERTScore F1 | 95% CI |", "|---|---|---|---|---|---|"]
    for (reader, method, ratio), group in sorted(groups.items()):
        values = [r["bertscore_f1"] for r in group]
        boot = cluster_bootstrap(values, [r["cluster"] for r in group], 2000)
        lines.append(f"| {reader} | {method} | {ratio} | {len(group)} | {np.mean(values):.4f} | "
                     f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}] |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
