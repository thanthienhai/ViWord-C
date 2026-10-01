"""Distilled labels: one teacher output, two label views (DATN §3.2).

1. The teacher LLM compresses the *unsegmented* text by deletion only.
2. The output is aligned back to the original syllables (`segment.align_compressed`);
   samples where the teacher rewrote too much (high unmatched ratio) are dropped.
3. Syllable labels: 1 if the syllable is kept (used by LLMLingua-2-vi).
   Word labels: soft label = share of the word's syllables that are kept (used by
   ViWord-C). Soft labels replace the "keep if ≥ 50%" rule, which turns every 1-of-2 tie
   into "keep" and shifts the class balance away from the syllable labels.
"""
from __future__ import annotations

import re

from .segment import Document, align_compressed, flatten, simple_tokens

# The target length is given as a syllable count: with a percentage the teacher produced the
# same length for 1/3 and 1/5 (pilot on 30 dev texts, see README "Ghi chú cài đặt").
TEACHER_PROMPT = """Bạn là một bộ nén văn bản. Hãy rút gọn văn bản dưới đây bằng cách CHỈ XOÁ bớt từ.
Quy tắc:
- Mỗi từ trong kết quả phải có trong văn bản gốc, giữ nguyên thứ tự và giữ nguyên cách viết.
- Không thêm từ mới, không thêm dấu câu mới, không viết lại, không viết tắt, không đổi từ đồng nghĩa, không dịch.
- Giữ lại những thông tin quan trọng nhất (sự kiện, con số, tên riêng, phủ định).
- Văn bản gốc có {n} âm tiết. Kết quả phải có khoảng {target} âm tiết (không quá {upper}).
Chỉ trả về văn bản đã rút gọn, không giải thích.

Văn bản:
{text}"""


def teacher_prompt(text: str, rate: float) -> str:
    n = len(text.split())
    target = max(1, round(n * rate))
    return TEACHER_PROMPT.format(n=n, target=target, upper=round(target * 1.1), text=text)


def teacher_compress(client, model: str, text: str, rate: float, max_tokens: int = 4096) -> str:
    """One deletion-only compression by the teacher through an OpenAI-compatible chat API
    (a `vllm serve` endpoint or OpenAI). Only the output text is used, no logits.
    Greedy decoding; thinking mode is disabled (vLLM `chat_template_kwargs`, ignored by
    servers that do not support it) and any leftover <think> block is removed."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": teacher_prompt(text, rate)}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def make_labels(doc: Document, compressed: str) -> dict:
    """Both label views from one teacher output, plus the teacher's own statistics."""
    words = flatten(doc)
    syllable_mask, unmatched_ratio = align_compressed(words, compressed)
    syllable_labels = [int(k) for k in syllable_mask]

    word_labels, partial, multi, i = [], 0, 0, 0
    for w in words:
        n = len(w.syllables)
        kept = sum(syllable_labels[i:i + n])
        i += n
        word_labels.append(kept / n)
        if n > 1 and kept > 0:
            multi += 1
            partial += kept < n
    return {
        "syllable_labels": syllable_labels,
        "word_labels": word_labels,
        "unmatched_ratio": unmatched_ratio,
        "keep_rate": sum(syllable_labels) / max(len(syllable_labels), 1),
        "teacher_cbr": partial / multi if multi else 0.0,
    }


# ---------------------------------------------------------------------------- leak check

def shingles(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Lowercased word/punctuation n-grams, independent of how the text was segmented."""
    tokens = simple_tokens(text)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class LeakIndex:
    """n-gram (default 13 tokens) index of all evaluation contexts.

    Training paragraphs sharing any n-gram with an evaluation context are dropped. The
    index refuses to be empty: a leak check that silently compares against nothing is
    worse than no check (a lesson from the LACC pilot).
    """

    def __init__(self, texts: list[str], n: int = 13):
        self.n = n
        self.grams: set = set()
        for t in texts:
            self.grams |= shingles(t, n)
        if not self.grams:
            raise ValueError("leak index is empty: evaluation files not found or too short")

    def leaks(self, text: str) -> bool:
        return not shingles(text, self.n).isdisjoint(self.grams)
