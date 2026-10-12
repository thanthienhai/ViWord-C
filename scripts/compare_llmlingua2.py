"""Compare the scored and package LLMLingua-2 masks at one reader budget.

The scored arm uses the published LLMLingua-2 checkpoint through ``SyllableScorer``;
the official arm uses the ``llmlingua`` package. Both receive the same segmented
documents and the same integer budget from the reader tokenizer. Package text is
aligned back to the source syllables before comparing masks (DATN F2).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

from tqdm import tqdm

from viword import diagnose as D
from viword.data import load_contexts, load_task
from viword.eval import context_budget, tokens_per_budget
from viword.segment import align_compressed, flatten
from viword.select import TokenCounter


DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
MASK_MATCH_THRESHOLD = 0.95
CBR_DIFF_THRESHOLD_POINTS = 1.0


def _unique_documents(examples):
    seen = set()
    return [ex for ex in examples if not (ex.cluster in seen or seen.add(ex.cluster))]


def _cbr_all(counts):
    value = D.cbr(counts)["all"]
    return None if math.isnan(value) else value


def _cbr_diff_points(left, right):
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    return abs(left - right) * 100.0


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def passes_thresholds(result: dict) -> bool:
    """Return whether one ratio satisfies the preregistered F2 thresholds."""
    return (result["mask_exact_rate"] >= MASK_MATCH_THRESHOLD
            and result["cbr_diff_points"] < CBR_DIFF_THRESHOLD_POINTS)


def report_passes(report: dict) -> bool:
    """Return the global decision used by ``gate_report.py``."""
    return bool(report.get("per_ratio")) and all(
        passes_thresholds(result) for result in report["per_ratio"].values()
    )


def compare(args) -> dict:
    from transformers import AutoTokenizer
    from viword.baselines import LLMLinguaOfficial
    from viword.compressor import SelectionConfig, ScoredCompressor
    from viword.model import SyllableScorer

    examples = _unique_documents(load_task(args.task, args.data))[:args.limit]
    contexts = load_contexts(examples, args.segmented)
    budget_tokenizer = args.budget_tokenizer or args.reader
    counter = TokenCounter(AutoTokenizer.from_pretrained(budget_tokenizer))
    scorer = SyllableScorer(args.model, device=args.device)
    scored = ScoredCompressor(scorer, SelectionConfig(unit="syllable"), "llmlingua2_syl")
    official = LLMLinguaOfficial("llmlingua2", device=args.device)

    per_ratio = {}
    for ratio in args.ratios:
        scored_cbr, official_cbr = Counter(), Counter()
        exact, matching, syllable_total = 0, 0, 0
        scored_budget, official_budget = [], []
        for ctx in tqdm(contexts, desc=f"LLMLingua-2 comparison @ {ratio:.3f}"):
            ctx.ratio = ratio
            _, budget = context_budget(ctx.text, ratio, counter)
            scored_text = scored.compress(ctx, budget, counter)
            official_text = official.compress(ctx, budget, counter)
            scored_mask, _ = align_compressed(flatten(ctx.doc), scored_text)
            official_mask, _ = align_compressed(flatten(ctx.doc), official_text)
            exact += scored_mask == official_mask
            matching += sum(a == b for a, b in zip(scored_mask, official_mask))
            syllable_total += len(scored_mask)
            scored_cbr += D.cbr_counts(flatten(ctx.doc), scored_mask)
            official_cbr += D.cbr_counts(flatten(ctx.doc), official_mask)
            scored_budget.append(tokens_per_budget(counter.count(scored_text), budget))
            official_budget.append(tokens_per_budget(counter.count(official_text), budget))

        scored_cbr_value = _cbr_all(scored_cbr)
        official_cbr_value = _cbr_all(official_cbr)
        result = {
            "n_documents": len(contexts),
            "mask_exact_rate": exact / len(contexts) if contexts else 0.0,
            "syllable_agreement": matching / syllable_total if syllable_total else 0.0,
            "scored_cbr": scored_cbr_value,
            "package_cbr": official_cbr_value,
            "cbr_diff_points": _cbr_diff_points(scored_cbr_value, official_cbr_value),
            "mean_scored_tokens_per_budget": _mean(scored_budget),
            "mean_package_tokens_per_budget": _mean(official_budget),
        }
        result["passed"] = passes_thresholds(result)
        per_ratio[f"{ratio:.3f}"] = result

    report = {
        "task": args.task,
        "n_documents": len(contexts),
        "model": args.model,
        "budget_tokenizer": budget_tokenizer,
        "thresholds": {
            "mask_exact_rate": MASK_MATCH_THRESHOLD,
            "cbr_diff_points_strictly_less_than": CBR_DIFF_THRESHOLD_POINTS,
        },
        "per_ratio": per_ratio,
    }
    report["passed"] = report_passes(report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--budget-tokenizer", required=True)
    ap.add_argument("--reader", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ratios", type=float, nargs="+", default=[1 / 3])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = compare(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"task": args.task, "n_documents": report["n_documents"],
                      "passed": report["passed"], "per_ratio": report["per_ratio"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
