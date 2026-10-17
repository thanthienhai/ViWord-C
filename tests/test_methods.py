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


def test_llmlingua_gets_a_legacy_kv_cache_it_can_index_and_slice():
    # llmlingua 0.2.2 indexes kv[0][0] and slices [k, v] lists; transformers 5 only takes Cache objects
    import pytest

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from viword.baselines import _LegacyKVModel

    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = _LegacyKVModel(transformers.LlamaForCausalLM(config).eval())
    ids = torch.randint(0, 50, (1, 8))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        full = model(ids, attention_mask=mask, use_cache=True)
        kv = full.past_key_values
        assert kv[0][0].shape[2] == 8
        kv = [[k[:, :, :5], v[:, :, :5]] for k, v in kv]  # what llmlingua does between windows
        rest = model(ids[:, 5:], attention_mask=mask, past_key_values=kv, use_cache=True)
    assert torch.allclose(rest.logits, full.logits[:, 5:], atol=1e-5)
    assert rest.past_key_values[0][0].shape[2] == 8
    assert model.config is config
