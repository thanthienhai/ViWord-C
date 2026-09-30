"""Evaluation data (DATN §4.1).

Every task is read from a normalized JSONL file (one example per line); `data/README.md`
describes how to build these files from the original releases. Belebele can also be
loaded directly from the Hugging Face hub.

`cluster` is the source document (article, premise, passage). Several examples can share
one source, so the bootstrap resamples clusters, not examples (DATN §4.3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

TASKS = {
    "vietnews": "summ",  # summarization
    "vinli": "nli",  # natural language inference, compress the premise
    "vimmrc": "mc",  # multiple-choice reading comprehension
    "belebele": "mc",
}


@dataclass
class Example:
    id: str
    cluster: str
    task: str  # "summ", "nli" or "mc"
    context: str  # the part that is compressed
    question: str = ""  # mc question, or nli hypothesis (never compressed)
    choices: list[str] = field(default_factory=list)
    answer: str = ""  # mc: "A".."D"; nli: label; summ: reference summary
    extra: dict = field(default_factory=dict)  # e.g. {"context_en": ...} for Belebele


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task(name: str, path: str, limit: int | None = None) -> list[Example]:
    """Normalized formats:
    vietnews: {id, cluster, context, reference}
    vinli:    {id, cluster, premise, hypothesis, label}
    vimmrc:   {id, cluster, context, question, choices, answer}   (answer = "A".."D")
    belebele: same as vimmrc, optional context_en
    """
    task = TASKS[name]
    examples = []
    for r in read_jsonl(path)[:limit]:
        if task == "summ":
            ex = Example(r["id"], r["cluster"], task, r["context"], answer=r["reference"])
        elif task == "nli":
            ex = Example(r["id"], r["cluster"], task, r["premise"], question=r["hypothesis"],
                         answer=r["label"])
        else:
            ex = Example(r["id"], r["cluster"], task, r["context"], question=r["question"],
                         choices=r["choices"], answer=r["answer"],
                         extra={"context_en": r["context_en"]} if "context_en" in r else {})
        examples.append(ex)
    return examples


def load_contexts(examples: list[Example], segmented_path: str):
    """Pair each example with its cached segmentation (scripts/segment_data.py)."""
    from .compressor import Context
    from .segment import doc_from_json, normalize

    docs = {r["id"]: r["doc"] for r in read_jsonl(segmented_path)}
    missing = [ex.id for ex in examples if ex.id not in docs]
    if missing:
        raise KeyError(f"{len(missing)} examples are not segmented, e.g. {missing[:3]}")
    return [Context(text=normalize(ex.context), doc=doc_from_json(docs[ex.id]),
                    question=ex.question or None, text_en=ex.extra.get("context_en", ""),
                    example_id=ex.id) for ex in examples]


def belebele_from_hub(out_path: str) -> None:
    """Write Belebele (vie_Latn, with the parallel eng_Latn passage) in the normalized format."""
    from datasets import load_dataset

    vi = load_dataset("facebook/belebele", "vie_Latn", split="test")
    en = load_dataset("facebook/belebele", "eng_Latn", split="test")
    # the two configs are not stored in the same row order: join on (link, question_number)
    english = {(b["link"], b["question_number"]): b for b in en}
    rows = []
    for a in vi:
        b = english[a["link"], a["question_number"]]
        rows.append({
            "id": f"{a['link']}#{a['question_number']}",
            "cluster": a["link"],
            "context": a["flores_passage"],
            "context_en": b["flores_passage"],
            "question": a["question"],
            "choices": [a[f"mc_answer{k}"] for k in range(1, 5)],
            "answer": "ABCD"[int(a["correct_answer_num"]) - 1],
        })
    write_jsonl(out_path, rows)
