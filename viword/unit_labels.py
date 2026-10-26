"""Utility labels of direction B: how much one unit of a legal RAG context helps the reader give the gold answer.

For a question q with gold answer a and a candidate unit u (a clause, scripts/legal_rag.py), the label is the
gain in the reader's log-probability of the answer when u is the whole context,

    U(u) = log p(a | prompt(q, u)) − log p(a | prompt(q, ∅)),

averaged over the answer's tokens. prompt() is the evaluation prompt of the task (`viword.eval.build_prompt`, the
lqa prompt for vibid) in the reader's chat template, so the labels ask the same reader the same question as the
evaluation does. The proposal's H4 compares these labels with the gold-unit labels (the units holding the answer).
"""
from __future__ import annotations

import os

from .data import Example
from .eval import build_prompt, chat_messages


def answer_ids(tokenizer, prompt: str, answer: str, system_prompt: str | None = None) -> tuple[list[int], int]:
    """Token ids of the chat turn for `prompt` followed by `answer` as the reply, and where the answer starts.
    The two parts are tokenized apart: the generation prompt ends with a newline, so nothing merges across."""
    messages = chat_messages(prompt, system_prompt)
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    head = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    tail = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return head + tail, len(head)


class HFAnswerScorer:
    """Sum of the answer tokens' log-probabilities with plain transformers, one sequence at a time (small readers,
    CPU smoke tests)."""

    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).to(self.device).eval()

    def logprobs(self, sequences: list[tuple[list[int], int]]) -> list[float]:
        out = []
        for ids, start in sequences:
            x = self.torch.tensor([ids], device=self.device)
            with self.torch.no_grad():
                logp = self.torch.log_softmax(self.model(x).logits[0].float(), dim=-1)
            out.append(float(sum(logp[k - 1, ids[k]] for k in range(start, len(ids)))))
        return out


class VLLMAnswerScorer:
    """The same sum with vLLM's prompt log-probabilities, in one batch (the H100 runs). Prefix caching is off:
    vLLM does not return prompt log-probabilities for tokens served from the cache."""

    def __init__(self, model: str, max_model_len: int = 4096):
        from vllm import LLM

        kwargs = {"gpu_memory_utilization": float(os.environ.get("VIWORD_VLLM_GPU_UTIL", 0.9)),
                  "dtype": os.environ.get("VIWORD_VLLM_DTYPE", "auto"),
                  "tensor_parallel_size": int(os.environ.get("VIWORD_VLLM_TP", 1)),
                  "enable_prefix_caching": False}
        if os.environ.get("VIWORD_VLLM_QUANT"):
            kwargs["quantization"] = os.environ["VIWORD_VLLM_QUANT"]
        self.llm = LLM(model=model, max_model_len=max_model_len, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def logprobs(self, sequences: list[tuple[list[int], int]]) -> list[float]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids, _ in sequences], params)
        out = []
        for (ids, start), o in zip(sequences, outputs):
            # prompt_logprobs[k] maps token ids to Logprob for position k (None at k = 0); the prompt's own token
            # is always among them
            out.append(float(sum(o.prompt_logprobs[k][ids[k]].logprob for k in range(start, len(ids)))))
        return out


def label_rows(rows: list[dict], scorer, task: str = "lqa", system_prompt: str | None = None) -> list[dict]:
    """One label row per context row (a vibid RAG row: question, answers, context of units joined by blank lines,
    units, gold_positions): the answer's log-probability with no context and with each unit alone."""
    sequences, spans = [], []
    for r in rows:
        units = r["context"].split("\n\n")
        if len(units) != len(r["units"]):
            raise ValueError(f"{r['id']}: {len(units)} blank-line blocks for {len(r['units'])} units")
        ex = Example(r["id"], r["cluster"], task, "", question=r["question"], extra={"answers": r["answers"]})
        start = len(sequences)
        for text in [""] + units:
            sequences.append(answer_ids(scorer.tokenizer, build_prompt(ex, text), r["answers"][0], system_prompt))
        spans.append((start, len(units)))
    logp = scorer.logprobs(sequences)
    out = []
    for r, (start, n) in zip(rows, spans):
        n_answer = len(sequences[start][0]) - sequences[start][1]
        empty, per_unit = logp[start], logp[start + 1:start + 1 + n]
        out.append({"id": r["id"], "cluster": r["cluster"], "question": r["question"], "answer": r["answers"][0],
                    "units": r["units"], "texts": r["context"].split("\n\n"),
                    "gold": [i in set(r["gold_positions"]) for i in range(n)],
                    "n_answer_tokens": n_answer, "logp_empty": empty, "logp": per_unit,
                    "utility": [(lp - empty) / max(1, n_answer) for lp in per_unit],
                    **({"heldout_article": r["heldout_article"]} if "heldout_article" in r else {})})
    return out
