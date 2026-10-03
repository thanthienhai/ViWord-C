"""Readers, prompts and task metrics (DATN §4.1).

Only the context is compressed; instructions, questions, hypotheses and answer choices are
kept verbatim. Readers are zero-shot with greedy decoding.

ROUGE is computed here on lowercased syllables: the `rouge_score` package drops every
non-ASCII character, which silently breaks Vietnamese.
"""
from __future__ import annotations

import re
from collections import Counter

from .data import Example

NLI_LABELS = ["entailment", "contradiction", "neutral", "other"]
MAIN_METRIC = {"vietnews": "rougeL", "vinli": "accuracy", "vimmrc": "accuracy", "belebele": "accuracy"}


def build_prompt(example: Example, context: str) -> str:
    if example.task == "summ":
        return ("Tóm tắt văn bản sau thành vài câu ngắn bằng tiếng Việt.\n\n"
                f"Văn bản: {context}\n\nTóm tắt:")
    if example.task == "nli":
        return ("Cho một tiền đề và một giả thuyết. Xác định quan hệ giữa chúng. Trả lời đúng một từ: "
                "entailment, contradiction, neutral hoặc other.\n\n"
                f"Tiền đề: {context}\nGiả thuyết: {example.question}\n\nNhãn:")
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", example.choices))
    return ("Đọc đoạn văn và chọn đáp án đúng. Chỉ trả lời một chữ cái (A, B, C hoặc D).\n\n"
            f"Đoạn văn: {context}\n\nCâu hỏi: {example.question}\n{options}\n\nĐáp án:")


MAX_NEW_TOKENS = {"summ": 256, "nli": 8, "mc": 8}


def parse_prediction(task: str, text: str) -> str:
    text = text.strip()
    if task == "nli":
        lowered = text.lower()
        hits = [(lowered.find(label), label) for label in NLI_LABELS if label in lowered]
        return min(hits)[1] if hits else ""
    if task == "mc":
        match = re.search(r"\b([ABCD])\b", text)
        return match.group(1) if match else ""
    return text


# ---------------------------------------------------------------------------- metrics

def _syllables(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_n(prediction: str, reference: str, n: int) -> float:
    def grams(tokens):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    p, r = grams(_syllables(prediction)), grams(_syllables(reference))
    overlap = sum((p & r).values())
    if not p or not r or not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    a, b = _syllables(prediction), _syllables(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def score(example: Example, prediction: str) -> dict[str, float]:
    """Per-example metrics. The first key is the task's main metric."""
    if example.task == "summ":
        return {"rougeL": rouge_l(prediction, example.answer),
                "rouge1": rouge_n(prediction, example.answer, 1),
                "rouge2": rouge_n(prediction, example.answer, 2)}
    return {"accuracy": float(prediction == example.answer)}


# ---------------------------------------------------------------------------- reader

class HFReader:
    """Fallback reader with plain transformers, one prompt at a time (slow; for small models,
    smoke tests, and machines where vLLM is unavailable, e.g. Windows)."""

    def __init__(self, model: str, device: str | None = None, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(self.device).eval()

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        outputs = []
        for p in prompts:
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                                     enable_thinking=False, return_tensors="pt",
                                                     return_dict=True).to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            outputs.append(self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
        return outputs


class APIReader:
    """A reader served behind an OpenAI-compatible chat API (e.g. `vllm serve`), queried in
    parallel. `tokenizer` is the Hugging Face id of the reader's tokenizer, used for budgets.
    Do not use the teacher model as a reader: the teacher would grade its own compressions."""

    def __init__(self, model: str, base_url: str, tokenizer: str, api_key: str = "EMPTY", workers: int = 16):
        from openai import OpenAI
        from transformers import AutoTokenizer

        self.name = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5, timeout=600)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.workers = workers

    def _one(self, prompt: str, max_new_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.name, messages=[{"role": "user", "content": prompt}], temperature=0.0,
            max_tokens=max_new_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.S)

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        from tqdm import tqdm

        with ThreadPoolExecutor(self.workers) as pool:
            return list(tqdm(pool.map(lambda p: self._one(p, max_new_tokens), prompts), total=len(prompts)))


class VLLMReader:
    """A target LLM served with vLLM (offline). Thinking mode is always disabled."""

    def __init__(self, model: str, max_model_len: int = 8192, **kwargs):
        from vllm import LLM

        self.name = model
        self.llm = LLM(model=model, max_model_len=max_model_len, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        from vllm import SamplingParams

        messages = [[{"role": "user", "content": p}] for p in prompts]
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self.llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outputs]
