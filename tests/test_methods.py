import sys
from types import SimpleNamespace

from conftest import W

from viword.compressor import Context
from viword.methods import build_compressor
from viword.protect import force_tokens_for_context
from viword.select import TokenCounter


def test_llmlingua2_force_builds_contextual_force_tokens(monkeypatch, tokenizer):
    calls = []

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            pass

        def compress_prompt(self, text, **kwargs):
            calls.append(kwargs)
            return {"compressed_prompt": text}

    monkeypatch.setitem(sys.modules, "llmlingua", SimpleNamespace(PromptCompressor=FakePromptCompressor))
    words = [W("không hề", "R"), W("Hà Nội", "Np", "B-LOC"), W("42", "M")]
    context = Context(text="không hề Hà Nội 42", doc=[words])
    compressor = build_compressor("llmlingua2_force")
    compressor.compress(context, 100, TokenCounter(tokenizer))

    assert calls
    assert calls[-1]["force_tokens"] == force_tokens_for_context(words)
    assert calls[-1]["force_reserve_digit"] is True
