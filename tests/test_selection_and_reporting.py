import json

import numpy as np
import pytest
from conftest import W

from viword.compressor import Context, ScoredCompressor, SelectionConfig, select_units
from viword.select import TokenCounter, trim_to_budget
from viword.stats import cluster_bootstrap_slope


class CharTokenizer:
    """One token per character (spaces included), to make costs easy to reason about."""

    def encode(self, text, add_special_tokens=False):
        return list(text)


class FixedScorer:
    def __init__(self, logits):
        self.logits = np.array(logits, dtype=float)

    def syllable_logits(self, doc):
        return self.logits


def test_unit_cost_counts_leading_space_except_for_punctuation():
    counter = TokenCounter(CharTokenizer())
    assert counter.unit_cost("học sinh", False) == len(" học sinh")
    assert counter.unit_cost(",", True) == 1
    assert counter.unit_cost("", True) == 1  # never zero


def test_trim_drops_unprotected_units_before_protected_ones():
    keep = [True, True, True]
    scores = [0.1, 0.9, 0.5]  # unit 0 is protected but has the lowest score
    count = lambda k: 10 * sum(k)  # noqa: E731  each kept unit costs 10 real tokens
    # one unit must go: with priority, the lowest-scoring *unprotected* unit (index 2)
    assert trim_to_budget(keep, scores, 20, count, priority=[1, 0, 0]) == [True, True, False]
    # without priority, the protected low-score unit would be dropped first
    assert trim_to_budget(keep, scores, 20, count) == [False, True, True]


def test_selection_stats_report_cost_deviation_trim_and_protected_tokens():
    words = [W("không", "R"), W("đi", "V"), W("học", "V")]
    stats = {}
    text = select_units(words, [0.1, 0.9, 0.8], 7, TokenCounter(CharTokenizer()), unit="word",
                        priority=[1, 0, 0], protected=[True, False, False], stats=stats)
    assert text.startswith("không")
    assert stats["n_protected"] == 1 and stats["n_protected_kept"] == 1
    assert stats["p_unit_cost"] == len(" không")
    assert stats["tokens_before_trim"] - stats["sum_unit_cost"] == stats["cost_deviation"]


def test_scored_compressor_exposes_last_stats():
    words = [W("không", "R"), W("đi", "V"), W(".", "CH")]
    comp = ScoredCompressor(FixedScorer([-3.0, 3.0, 0.0]), SelectionConfig(unit="word", protect="hard",
                                                                          tiers=("T1",)), "x")
    comp.compress(Context(text="không đi .", doc=[words]), 6, TokenCounter(CharTokenizer()))
    assert comp.last_stats["n_protected"] == 1 and comp.last_stats["n_protected_kept"] == 1


def test_row_cbr_counts_broken_multisyllable_words():
    from scripts.evaluate import row_cbr
    ctx = Context(text="học sinh đi học", doc=[[W("học sinh", "N"), W("đi", "V"), W("học", "V")]])
    assert row_cbr(ctx, "học đi học") == {"cbr_partial": 1, "cbr_full": 0, "cbr": 1.0}
    assert row_cbr(ctx, "học sinh") == {"cbr_partial": 0, "cbr_full": 1, "cbr": 0.0}
    assert row_cbr(Context(text="x"), "x") == {}


def test_cluster_bootstrap_slope_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 400)
    y = 0.3 * x + rng.normal(0, 0.05, 400)
    result = cluster_bootstrap_slope(x, y, [str(i // 4) for i in range(400)], n_boot=500)
    assert result["slope"] == pytest.approx(0.3, abs=0.03)
    assert result["ci_low"] < 0.3 < result["ci_high"]


def test_analyze_reports_empty_answers_and_cbr_regression(tmp_path, monkeypatch, capsys):
    from scripts.analyze import main
    rows = []
    for i in range(20):
        cbr = (i % 5) / 4
        base = {"task": "vimmrc", "reader": "r", "ratio": 0.2, "id": str(i), "cluster": f"c{i}",
                "tokens_per_budget": 1.0}
        rows.append({**base, "method": "llmlingua2vi", "accuracy": 0.0, "cbr": cbr,
                     "prediction": "" if i == 0 else "A"})
        rows.append({**base, "method": "viword", "accuracy": float(cbr > 0.5), "prediction": "A"})
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze.py", "--rows", str(path), "--target", "viword",
                                     "--reference", "llmlingua2vi", "--cbr-regression", "llmlingua2vi",
                                     "--n-boot", "200"])
    main()
    out = capsys.readouterr().out
    assert "5.0% ⚠" in out  # 1 of 20 llmlingua2vi answers could not be parsed
    assert "## H1 mechanism" in out
    line = next(l for l in out.splitlines() if l.startswith("| vimmrc | r | 0.2 | +"))
    assert float(line.split("|")[4].split()[0]) > 0


def test_package_rate_search_never_exceeds_the_budget():
    from viword.baselines import _search_rate
    counter = TokenCounter(CharTokenizer())
    # output length grows with rate; 102 chars would be inside the old ±3% window
    compress_at = lambda rate: "x" * int(round(200 * rate))  # noqa: E731
    text = _search_rate(compress_at, 100, counter)
    assert 97 <= len(text) <= 100


def test_package_rate_search_cuts_to_the_budget_when_the_method_cannot_compress():
    from viword.baselines import _search_rate
    counter = TokenCounter(CharTokenizer())
    # LLMLingua returns a short context unchanged at every rate: no rate fits the budget
    text = _search_rate(lambda rate: "một hai ba bốn năm sáu", 12, counter)
    assert text == "một hai ba"  # leading whole words, 10 chars <= 12 < 15
    assert _search_rate(lambda rate: "dài", 1, counter) == ""  # not even one word fits


def test_greedy_gap_is_at_most_one_and_exact_on_easy_cases():
    from scripts.greedy_gap import objective_gap
    greedy, exact = objective_gap([1.0, 1.0, 1.0], [1, 1, 1], 2, alpha=0.0)
    assert greedy == exact == 2.0
    # score-only greedy takes the big item and stops; DP takes the two small ones
    greedy, exact = objective_gap([0.9, 0.6, 0.6], [3, 2, 2], 4, alpha=0.0)
    assert greedy == 0.9 and exact == 1.2 and greedy <= exact


def test_bertscore_feeds_phobert_word_segmented_text():
    from scripts.bertscore import add_bertscore

    class Seg:
        def segment(self, text):
            return [[W("học sinh"), W("đi", "V")]] if text else []

    seen = {}

    def score_fn(candidates, refs):
        seen["c"], seen["r"] = candidates, refs
        return [0.7] * len(candidates)

    rows = [{"task": "vietnews", "id": "a", "raw_output": "học sinh đi"}, {"task": "vinli", "id": "b"}]
    out = add_bertscore(rows, {"a": "học sinh đi"}, Seg(), score_fn)
    assert seen["c"] == ["học_sinh đi"] and seen["r"] == ["học_sinh đi"]
    assert out[0]["bertscore_f1"] == 0.7 and "bertscore_f1" not in out[1]
