"""Changes of the 2026-10-24 review: label-unit controls, cost-aware selection, scope, detector v2,
per-row diagnostics, caches and the new descriptive scripts."""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import W

from viword.compressor import Context, ScoredCompressor, SelectionConfig
from viword.segment import doc_to_json
from viword.select import TokenCounter, exact_select

ROOT = Path(__file__).parents[1]


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class FixedScorer:
    def __init__(self, logits):
        self.logits = np.array(logits, dtype=float)

    def syllable_logits(self, doc):
        return self.logits


# ---------------------------------------------------------------------------- item 1: iii-rand, iv-wordloss

def _record():
    words = [W("học sinh"), W("đi"), W("học"), W("trường học"), W(".", "CH")]
    # teacher keeps "học" of "học sinh" (cut), "đi", both of "trường học"
    return {"id": "p1", "rate": 0.5, "doc": doc_to_json([words]),
            "syllable_labels": [1, 0, 1, 0, 1, 1, 1], "word_labels": [0.5, 1.0, 0.0, 1.0, 1.0]}


def test_softrand_matches_the_number_and_class_of_softened_syllables():
    from viword.train import softrand_targets, syllable_targets
    record = _record()
    word_view = syllable_targets(record, "word")
    assert word_view == [0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0]
    rand = softrand_targets(record, seed=0)
    hard = record["syllable_labels"]
    softened = [i for i, t in enumerate(rand) if t == 0.5]
    assert len(softened) == 2  # as many as the word view softens
    assert sorted(hard[i] for i in softened) == [0, 1]  # one kept, one dropped, as in the cut word
    assert rand[-1] == 1.0  # punctuation is never softened
    assert rand == softrand_targets(record, seed=0)  # reproducible
    assert all(t in (0.0, 1.0) for i, t in enumerate(rand) if i not in softened)


def test_word_pooled_loss_has_one_term_per_word():
    torch = pytest.importorskip("torch")
    from viword.train import pool_by_word

    logits = torch.tensor([2.0, 0.0, 1.0, 3.0, 5.0])
    targets = torch.tensor([0.5, 0.5, 1.0, 0.0, 0.0])
    groups = torch.tensor([0, 0, 1, 2, 2])
    w_logits, w_targets = pool_by_word(logits, targets, groups, torch)
    assert w_logits.tolist() == [1.0, 1.0, 4.0]
    assert w_targets.tolist() == [0.5, 1.0, 0.0]


def test_build_items_records_window_local_word_groups():
    from conftest import WhitespaceTokenizer

    from viword.train import build_items, loss_unit_targets

    class Tok(WhitespaceTokenizer):
        def __call__(self, pieces, is_split_into_words=True, add_special_tokens=False, **kwargs):
            class Enc(dict):
                def word_ids(self_inner):
                    return [None] + list(range(len(pieces))) + [None]
            return Enc(input_ids=list(range(len(pieces) + 2)))

    items = build_items([_record()], Tok(), "word_pooled", max_length=64)
    assert items[0]["groups"] == [0, 0, 1, 2, 3, 3, 4]
    assert loss_unit_targets(items) == [0.5, 1.0, 0.0, 1.0, 1.0]


# ---------------------------------------------------------------------------- item 2: exact DP, priority

def test_exact_select_is_optimal_and_respects_priority():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 7
        scores = list(rng.random(n))
        costs = list(rng.integers(1, 6, n))
        budget = int(rng.integers(3, 15))
        keep = exact_select(scores, costs, budget)
        assert sum(c for c, k in zip(costs, keep) if k) <= budget
        best = max(sum(s for s, x in zip(scores, xs) if x)
                   for xs in itertools.product([0, 1], repeat=n)
                   if sum(c for c, x in zip(costs, xs) if x) <= budget)
        assert sum(s for s, k in zip(scores, keep) if k) == pytest.approx(best)
    keep = exact_select([0.1, 0.9, 0.8], [3, 3, 3], 4, priority=[1, 0, 0])
    assert keep == [True, False, False]  # the protected unit first, even though it scores lowest


def test_cost_aware_selection_runs_for_word_and_syllable_cells():
    words = [W("không"), W("học sinh"), W("đi"), W(".", "CH")]
    counter = TokenCounter(CharTokenizer())
    ctx = Context(text="không học sinh đi .", doc=[words])
    for unit, logits in (("word", [0.0, 3.0, 3.0, 1.0, 0.0]), ("syllable", [0.0, 3.0, 3.0, 1.0, 0.0])):
        for cfg in (SelectionConfig(unit=unit, alpha=1.0), SelectionConfig(unit=unit, exact=True)):
            comp = ScoredCompressor(FixedScorer(logits), cfg, "x")
            out = comp.compress(ctx, 12, counter)
            assert counter.count(out) <= 12 and comp.last_stats["n_trimmed"] >= 0


# ---------------------------------------------------------------------------- item 4: scope

def test_scope_heads_and_bonus():
    from viword.diagnose import scope_counts
    from viword.protect import scope_heads

    words = [W("Tôi", "P"), W("không", "R"), W("hề", "R"), W("được phép", "V"), W("vào", "V"), W(".", "CH")]
    assert scope_heads(words) == [(1, 3), (3, 4)]  # không -> được phép -> vào
    keep_op_only = [False, True, False, False, False, False]
    assert scope_counts(words, keep_op_only) == {"ops": 2, "ops_kept": 1, "ops_kept_with_head": 0}
    # with the bonus, the head of the protected operator outranks a neutral word
    logits = [0.0, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0]  # per syllable ("được phép" has two)
    cfg = SelectionConfig(unit="word", protect="hard", tiers=("T1",), scope=0.3)
    comp = ScoredCompressor(FixedScorer(logits), cfg, "x")
    words_, scores, _, _ = comp.unit_scores(Context(text="", doc=[words]))
    assert scores[3] > scores[4]


def test_question_relevance_bonus():
    words = [W("Hà Nội"), W("đẹp"), W("lắm")]
    cfg = SelectionConfig(unit="word", qrel=0.5)
    comp = ScoredCompressor(FixedScorer([0.0, 0.0, 0.0, 0.0]), cfg, "x")
    _, scores, _, _ = comp.unit_scores(Context(text="", doc=[words], question="Hà Nội ở đâu?"))
    assert scores[0] > scores[1] == scores[2]


# ---------------------------------------------------------------------------- item 5: detector v2

def test_detector_v2_catches_untagged_names_and_skips_article_mot():
    from viword.protect import DETECTOR_VERSION, protection_tiers
    if DETECTOR_VERSION != "v2":
        pytest.skip("VIWORD_P_DETECTOR=v1")
    words = [W("Hôm nay", "N"), W("anh", "N"), W("Minh", "N"), W("mua", "V"), W("một", "M"), W("con", "Nc"),
             W("mèo", "N"), W("và", "C"), W("một", "M"), W("kg", "Nu"), W(".", "CH"),
             W("Sau", "E"), W("đó", "P"), W("Thanh Hóa", "N"), W("mưa", "V")]
    tiers = protection_tiers(words)
    assert tiers[0] is None  # sentence-initial, only its first syllable capitalised
    assert tiers[2] == "T1"  # "Minh": capitalised mid-sentence, NER missed it
    assert tiers[4] is None  # "một con" = article
    assert tiers[8] == "T1"  # "một kg" = the number one
    assert tiers[11] is None  # "Sau" after "." starts a sentence
    assert tiers[13] == "T1"  # every syllable capitalised


def test_detector_v1_is_still_available():
    code = ("from conftest import W; from viword.protect import protection_tiers;"
            "print(protection_tiers([W('anh'), W('Minh'), W('một', 'M'), W('con', 'Nc')]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__("os").environ, "VIWORD_P_DETECTOR": "v1",
                              "PYTHONPATH": f"{ROOT}{__import__('os').pathsep}{ROOT / 'tests'}"})
    assert out.stdout.strip() == "[None, None, 'T1', None]"


# ---------------------------------------------------------------------------- caches

def test_reader_is_called_once_per_distinct_prompt_and_reuses_the_disk_cache(tmp_path):
    from scripts.evaluate import generate_cached

    class Reader:
        name, system_prompt, calls = "r", None, []

        def generate(self, prompts, max_new_tokens):
            self.calls.append(list(prompts))
            return [p.upper() for p in prompts]

    reader, cache = Reader(), str(tmp_path / "c.jsonl")
    assert generate_cached(reader, ["a", "b", "a"], 8, cache) == ["A", "B", "A"]
    assert reader.calls == [["a", "b"]]
    assert generate_cached(reader, ["b", "c"], 8, cache) == ["B", "C"]
    assert reader.calls[-1] == ["c"]
    generate_cached(reader, ["b"], 16, cache)  # another token cap is another key
    assert reader.calls[-1] == ["b"]


def test_logit_disk_cache_skips_the_encoder(tmp_path):
    from viword.model import SyllableScorer

    class Counting(SyllableScorer):
        n = 0

        def _syllable_logits(self, doc):
            Counting.n += 1
            return np.arange(3, dtype=float)

    doc = [[W("học sinh"), W("đi")]]
    first = Counting("some/model", cache_dir=str(tmp_path))
    assert first.syllable_logits(doc).tolist() == [0.0, 1.0, 2.0]
    second = Counting("some/model", cache_dir=str(tmp_path))
    assert second.syllable_logits(doc).tolist() == [0.0, 1.0, 2.0]
    assert Counting.n == 1 and second.model is None  # the encoder was never loaded


def test_forward_cache_runs_the_classifier_once_per_input():
    torch = pytest.importorskip("torch")
    from viword.baselines import _ForwardCache

    class Model:
        calls, config = 0, "cfg"

        def __call__(self, input_ids=None, attention_mask=None):
            Model.calls += 1
            return input_ids.sum()

    cached = _ForwardCache(Model())
    ids = torch.tensor([[1, 2, 3]])
    for _ in range(4):
        cached(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert Model.calls == 1 and cached.config == "cfg"
    cached(input_ids=ids + 1, attention_mask=torch.ones_like(ids))
    assert Model.calls == 2


# ---------------------------------------------------------------------------- analysis

def _rows():
    rows = []
    for i in range(8):
        base = {"task": "vinli", "reader": "r", "id": f"x{i}", "cluster": f"c{i}", "ratio": 0.333, "budget": 10,
                "n_tokens_original": 30, "n_tokens_compressed": 10, "tokens_per_budget": 1.0}
        gold = "entailment"
        rows.append({**base, "method": "none", "ratio": 1.0, "prediction": gold, "accuracy": 1.0})
        rows.append({**base, "method": "random", "prediction": "contradiction" if i < 4 else gold,
                     "accuracy": float(i >= 4)})
        rows.append({**base, "method": "viword", "prediction": "contradiction" if i < 1 else gold,
                     "accuracy": float(i >= 1), "cbr_partial": 0, "cbr_full": 2, "cbr_chance_partial": 0.8,
                     "cbr_chance_survived": 1.9, "tier_kept": {"all": [5, 6], "T1": [1, 1]},
                     "scope_ops_kept": 1, "scope_heads_kept": 1,
                     "selection": {"n_trimmed": 1, "cost_deviation": 2}})
        rows.append({**base, "method": "probe_negation", "n_tokens_compressed": 29 if i < 2 else 30,
                     "prediction": gold, "accuracy": 1.0})
    return rows


def test_analyze_prints_descriptive_tables_and_excess_nfr(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    out = subprocess.run([sys.executable, "scripts/analyze.py", "--rows", str(path), "--target", "viword",
                          "--reference", "random", "--hypothesis", "none", "--n-boot", "200"],
                         capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
                         encoding="utf-8").stdout
    assert "## CBR vs random syllable deletion" in out and "| -0.4211 |" in out  # 0 − 0.8/1.9
    assert "## Table 4 bookkeeping" in out and "| 100.0% |" in out
    assert "## Retention by protection tier" in out and "1.000 (8)" in out
    assert "| r | viword | 0.333 | 0.1250 | -0.3750 |" in out  # excess NFR = 1/8 − 4/8
    assert "Flip direction on negated premises" in out
    assert "p Holm" in out and "| — |" in out  # descriptive run: nothing Holm-adjusted


# ---------------------------------------------------------------------------- new scripts

def test_gold_cbr_measures_on_human_boundaries():
    from scripts.gold_cbr import measure

    auto = [W("học sinh"), W("đi"), W("học")]
    gold_rows = [{"id": "a", "gold_words": ["học", "sinh_đi", "học"]}]  # a (made-up) different boundary
    rows = [{"id": "a", "method": "viword", "ratio": 0.5, "compressed": "học sinh"}]
    result = measure(gold_rows, {"a": auto}, rows)
    s = result["systems"][0]
    assert s["cbr_vncorenlp"] == 0.0  # whole VnCoreNLP word kept
    assert s["cbr_gold"] == 1.0  # but the human word "sinh đi" is cut
    b = result["vncorenlp_boundary"]
    assert (b["tp"], b["fp"], b["fn"]) == (2, 1, 1)  # gold ends {1, 3, 4}, VnCoreNLP ends {2, 3, 4}


def test_cohen_kappa():
    from viword.stats import cohen_kappa
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_teacher_agreement_compares_label_views():
    from scripts.teacher_agreement import compare
    a = _record()
    b = {**a, "syllable_labels": [1, 1, 1, 0, 1, 1, 1], "word_labels": [1.0, 1.0, 0.0, 1.0, 1.0],
         "keep_rate": 6 / 7, "teacher_cbr": 0.0}
    a = {**a, "keep_rate": 5 / 7, "teacher_cbr": 0.5}
    result = compare([a], [b])
    assert result["pairs"] == 1 and result["syllable_agreement"] == pytest.approx(6 / 7)
    assert result["word_label_mae"] == pytest.approx(0.1)


def test_minimal_pairs_flip_only_the_operator():
    from scripts.make_minimal_pairs import minimal_pairs
    doc = doc_to_json([[W("Anh"), W("ấy"), W("không", "R"), W("đến", "V"), W(".", "CH")]])
    rows = minimal_pairs([{"id": "p", "cluster": "c", "premise": "Anh ấy không đến."}], {"p": doc}, n=5)
    assert [r["label"] for r in rows] == ["entailment", "contradiction"]
    assert rows[1]["hypothesis"] == "Anh ấy đến." and rows[0]["hypothesis"] == rows[0]["premise"]
    assert all(r["needs_review"] for r in rows)


def test_long_context_rows_are_refused_outside_the_eval_stage(tmp_path):
    from scripts.make_long_context import long_rows
    from viword.heldout import refuse_unseen_test_rows
    rows = [{"id": f"q{i}", "cluster": f"l{i}", "context": " ".join(["x"] * 50), "question": "?",
             "choices": list("abcd"), "answer": "A"} for i in range(5)]
    out = long_rows(rows, target_syllables=120, limit=2)
    assert len(out) == 2 and all(r["n_passages"] == 3 for r in out)
    assert all(r["id"].endswith("#long") for r in out)
    out[0]["derived_from"] = "belebele"
    with pytest.raises(SystemExit):
        refuse_unseen_test_rows(out, tasks_dir=tmp_path, exclusions_path=tmp_path / "none.json", env={})


def test_select_cost_2x2_takes_the_best_shared_rule():
    from scripts.select_cost_2x2 import select
    rows = []
    for cell in ("llmlingua2vi", "llmlingua2vi_wordpool", "viword_syl", "viword"):
        for tag, value in (("", 0.50), ("_caa0p5", 0.52), ("_caa1", 0.55), ("_cadp", 0.55)):
            for ratio in (0.5, 0.333, 0.2):
                rows.append({"method": f"{cell}{tag}_s0", "task": "vinli", "ratio": ratio, "accuracy": value})
    decision = select(rows)
    assert decision["selected"] == "a1" and decision["spec"] == "alpha=1"  # tie with dp: the simpler rule


def test_neutral_teacher_prompt_drops_only_the_keep_line():
    from viword.distill import TEACHER_PROMPTS, teacher_prompt
    assert "phủ định" in TEACHER_PROMPTS["default"] and "phủ định" not in TEACHER_PROMPTS["neutral"]
    assert len(TEACHER_PROMPTS["default"].splitlines()) == len(TEACHER_PROMPTS["neutral"].splitlines()) + 1
    assert "âm tiết" in teacher_prompt("một hai ba", 0.5, "neutral")
