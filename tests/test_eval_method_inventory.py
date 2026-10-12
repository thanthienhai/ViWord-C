from pathlib import Path


PIPELINE = Path(__file__).parents[1] / "scripts" / "run_pipeline.sh"
README = Path(__file__).parents[1] / "README.md"


def test_stage_eval_covers_section_42_baselines():
    text = PIPELINE.read_text(encoding="utf-8")
    for spec in (
        "stopword",
        "lexical",
        '"llmlingua:lm=$LM"',
        "llmlingua2",
        '"random:seed=0"',
        '"random:seed=1"',
        '"random:seed=2"',
        '"longllmlingua:lm=$LM"',
        'name=teacher_upper',
    ):
        assert spec in text


def test_teacher_upper_bound_is_a_required_pre_eval_stage():
    text = PIPELINE.read_text(encoding="utf-8")
    assert "stage_teacher_test()" in text
    assert "bash scripts/run_pipeline.sh teacher_test before eval" in text
    assert '--limit "$TEACHER_TEST_LIMIT" --methods "precomputed:path=$teacher_path,name=teacher_upper"' in text


def test_every_method_spec_in_the_pipeline_has_a_known_name():
    # a misspelt spec (e.g. "longllingua") only fails after every earlier method has been
    # compressed on GPU, so check the names statically
    import re
    known = {"none", "no_context", "lead", "truncation", "stopword", "lexical", "tfidf_sent",
             "probe_negation", "random", "ppl_sent", "selective_context", "llmlingua2",
             "llmlingua2_force", "llmlingua", "longllmlingua", "translate", "precomputed",
             "scored", "lexprior"}
    text = PIPELINE.read_text(encoding="utf-8")
    quoted = re.findall(r'"([a-z_0-9]+):[a-z_]+=[^"]*"', text)
    assert quoted, "no method specs found"
    unknown = sorted(set(quoted) - known)
    assert not unknown, unknown


def test_dev_decisions_are_tied_to_the_primary_reader():
    text = PIPELINE.read_text(encoding="utf-8")
    for stage in ("stage_ckpt_check() {", "stage_select_pi() {", "stage_mde() {", "stage_select_alpha() {"):
        body = text[text.index(stage):text.index("\n}\n", text.index(stage))]
        assert "require_primary_reader" in body, stage
    assert '--reader "$PRIMARY_READER"' in text and '--primary-reader "$PRIMARY_READER"' in text


def test_cost_tokenizers_cover_the_datn_list_and_g5_counts_them_all():
    text = PIPELINE.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("COST_TOKENIZERS="))
    for name in ("Qwen2.5", "Llama-3.1", "gemma-2", "SeaLLMs", "Vistral"):
        assert name in line, name
    gate_report = (PIPELINE.parent / "gate_report.py").read_text(encoding="utf-8")
    assert "xlm-roberta-base" not in gate_report


def test_g3_compares_every_token_level_compressor():
    gate_eval = (PIPELINE.parent / "gate_eval.py").read_text(encoding="utf-8")
    for method in ("llmlingua2_syl", "llmlingua2_wordpool", "llmlingua2", "llmlingua", "selective_context",
                   "lexical", "stopword"):
        assert f'"{method}"' in gate_eval, method


def test_reader_messages_carry_an_optional_system_prompt():
    from viword.eval import chat_messages
    assert chat_messages("hỏi") == [{"role": "user", "content": "hỏi"}]
    assert chat_messages("hỏi", "Bạn là trợ lí.")[0] == {"role": "system", "content": "Bạn là trợ lí."}
