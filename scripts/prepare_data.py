"""Download public releases and write them in the normalized format of data/README.md.

Usage:
  python scripts/prepare_data.py belebele     # data/tasks/belebele.jsonl (900 questions, no dev split)
  python scripts/prepare_data.py vinli        # data/tasks/vinli_{train,dev,test}.jsonl
  python scripts/prepare_data.py vimmrc       # data/tasks/vimmrc_{dev,test}.jsonl
  python scripts/prepare_data.py vietnews     # data/tasks/vietnews_{dev,test}.jsonl (1000 sampled each)
  python scripts/prepare_data.py viquad       # data/tasks/viquad_{dev,test}.jsonl (answerable questions)
  python scripts/prepare_data.py gsm8k        # data/tasks/gsm8k_{dev,test}.jsonl (8-shot demonstrations)
  python scripts/prepare_data.py oneruler     # data/tasks/oneruler_{dev,test}.jsonl (4096 tokens, 7 subtasks)
  python scripts/prepare_data.py paragraphs   # data/distill/paragraphs.jsonl (VietNews train + Wikipedia)

Sources (record the exact revision in data/README.md when reporting results):
  belebele   facebook/belebele (vie_Latn + parallel eng_Latn)
  vinli      presencesw/vinli_4_label: unofficial mirror of ViNLI (Huynh et al., COLING 2022);
             split sizes match the paper
  vimmrc     ura-hcmut/ViMMRC: mirror of ViMMRC (the official uitnlp/vimmrc2.0 is gated)
  vietnews   nam194/vietnews: word-segmented release ("Cơ_quan", "27/3 ,"), detokenized here
  viquad     taidng/UIT-ViQuAD2.0 (UIT-ViQuAD 2.0, arXiv 2203.11400); validation split = our test
  gsm8k      hllj/vi_gsm8k (GSM8K translated with GPT-3.5 and checked by the uploader; MIT)
  oneruler   starbix/oneruler, config vi_4096 (OneRuler, arXiv 2503.01996)
  paragraphs VietNews train articles + Wikipedia vi (wikimedia/wikipedia 20231101.vi)
"""
import argparse
import ast
import hashlib
import os
import random
import re

from viword.data import write_jsonl


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def detokenize_vietnews(text: str) -> str:
    """'Cơ_quan điều_tra , TP. Hưng_Yên' -> 'Cơ quan điều tra, TP. Hưng Yên'."""
    text = text.replace("_", " ")
    text = re.sub(r" ([,.;:!?)\]}%…])", r"\1", text)
    text = re.sub(r"([(\[{]) ", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def take_sentences(text: str, max_syllables: int) -> str:
    """Longest prefix of whole sentences with at most `max_syllables` syllables."""
    out, n = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        k = len(sentence.split())
        if n + k > max_syllables:
            break
        out.append(sentence)
        n += k
    return " ".join(out)


def is_poem(text: str) -> bool:
    """ViMMRC contains poems from primary-school textbooks: many short lines."""
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines) >= 4 and sorted(len(l.split()) for l in lines)[len(lines) // 2] <= 10


# ---------------------------------------------------------------------------- tasks

def prepare_belebele(out_dir):
    # belebele_from_hub stores the source passage's stable `link` as cluster;
    # question_number remains part of id so questions on one passage stay grouped.
    from viword.data import belebele_from_hub
    belebele_from_hub(os.path.join(out_dir, "belebele.jsonl"))


def prepare_vinli(out_dir):
    from datasets import load_dataset

    data = load_dataset("presencesw/vinli_4_label")
    for split in ("train", "dev", "test"):
        # The mirror has no premise id: a stable hash of premise text defines the cluster.
        rows = [{"id": f"vinli-{split}-{i}", "cluster": short_hash(r["sentence1"]), "premise": r["sentence1"],
                 "hypothesis": r["sentence2"], "label": r["gold_label"]} for i, r in enumerate(data[split])]
        write_jsonl(os.path.join(out_dir, f"vinli_{split}.jsonl"), rows)
        print(f"vinli {split}: {len(rows)} pairs, {len({r['cluster'] for r in rows})} premises")


def prepare_vimmrc(out_dir):
    from datasets import load_dataset

    data = load_dataset("ura-hcmut/ViMMRC")
    for split, name in (("validation", "dev"), ("test", "test")):
        # All questions over one article share its article-text hash as cluster.
        rows, poems, ambiguous = [], 0, 0
        for i, r in enumerate(data[split]):
            if is_poem(r["article"]):
                poems += 1
                continue
            choices = ast.literal_eval(r["options"])
            gold = choices["ABCD".index(r["answer"].strip())]
            if choices.count(gold) > 1:  # the gold text is also a "wrong" option: two right answers
                ambiguous += 1
                continue
            rows.append({"id": f"vimmrc-{name}-{i}", "cluster": short_hash(r["article"]),
                         "context": r["article"], "question": r["question"],
                         "choices": choices, "answer": r["answer"].strip()})
        write_jsonl(os.path.join(out_dir, f"vimmrc_{name}.jsonl"), rows)
        print(f"vimmrc {name}: {len(rows)} questions, {len({r['cluster'] for r in rows})} passages, "
              f"{poems} poem questions and {ambiguous} questions with a duplicated gold choice dropped")


def prepare_vietnews(out_dir, n: int = 1000, seed: int = 0):
    from datasets import load_dataset

    data = load_dataset("nam194/vietnews")
    for split, name in (("validation", "dev"), ("test", "test")):
        # One article/guid is one cluster; sampling changes rows, not article identity.
        rows = [{"id": f"vietnews-{name}-{r['guid']}", "cluster": f"vietnews-{name}-{r['guid']}",
                 "context": detokenize_vietnews(r["article"]), "reference": detokenize_vietnews(r["abstract"])}
                for r in data[split] if r["article"] and r["abstract"]]
        rows = random.Random(seed).sample(rows, min(n, len(rows)))
        write_jsonl(os.path.join(out_dir, f"vietnews_{name}.jsonl"), rows)
        print(f"vietnews {name}: {len(rows)} articles sampled")


def prepare_viquad(out_dir, n_dev_passages: int = 250, seed: int = 0):
    """UIT-ViQuAD 2.0 (taidng/UIT-ViQuAD2.0). The public test split has no answers, so the
    validation split is our test set and dev is sampled by passage from train. Unanswerable
    questions are dropped: compression cannot be scored on them with span F1."""
    from datasets import load_dataset

    data = load_dataset("taidng/UIT-ViQuAD2.0")

    def rows_of(split):
        rows = []
        for r in split:
            answers = list(dict.fromkeys(r["answers"]["text"])) if r["answers"] else []
            if r["is_impossible"] or not answers:
                continue
            rows.append({"id": f"viquad-{r['id']}", "cluster": short_hash(r["context"]),
                         "context": r["context"], "question": r["question"], "answers": answers})
        return rows

    test = rows_of(data["validation"])
    train = rows_of(data["train"])
    passages = sorted({r["cluster"] for r in train})
    dev_passages = set(random.Random(seed).sample(passages, n_dev_passages))
    dev = [r for r in train if r["cluster"] in dev_passages]
    for name, rows in (("dev", dev), ("test", test)):
        write_jsonl(os.path.join(out_dir, f"viquad_{name}.jsonl"), rows)
        print(f"viquad {name}: {len(rows)} answerable questions, {len({r['cluster'] for r in rows})} passages")


def prepare_gsm8k(out_dir, k: int = 8, n_dev: int = 500, seed: int = 0):
    """GSM8K in Vietnamese (hllj/vi_gsm8k, MIT). As in LLMLingua, the compressed context is the
    k-shot chain-of-thought demonstrations. Every question gets its own k demonstrations drawn
    from the train pool (seeded), so contexts differ and the cluster bootstrap has clusters."""
    import json

    from huggingface_hub import hf_hub_download

    def load(name):
        with open(hf_hub_download("hllj/vi_gsm8k", name, repo_type="dataset"), encoding="utf-8") as f:
            return json.load(f)

    # Machine-translation artifacts, dropped everywhere: questions that contain the worked
    # solution ("... là 21 + 17 = 38 thẻ"; no English GSM8K question contains "="), and
    # degenerate repetition loops (a 3,514-syllable test question, a 4,005-syllable explanation).
    def clean(r, max_syllables):
        return "=" not in r["question"] and len((r["question"] + " " + r["explanation"]).split()) <= max_syllables

    train, test = load("vi_train.json"), [r for r in load("vi_test.json") if "=" not in r["question"]
                                          and len(r["question"].split()) <= 250]
    rng = random.Random(seed)
    order = rng.sample(range(len(train)), len(train))
    dev_items = [train[i] for i in order[:n_dev] if clean(train[i], 400)]
    pool = [train[i] for i in order[n_dev:] if clean(train[i], 400)]  # p99 of a demonstration: 257 syllables

    def demo(r):
        return f"Câu hỏi: {r['question']}\nLời giải: {r['explanation']}\nĐáp án: {r['answer']}"

    for name, items in (("dev", dev_items), ("test", test)):
        rows = []
        for r in items:
            demos = random.Random(f"{seed}-{name}-{r['index']}").sample(pool, k)
            rows.append({"id": f"gsm8k-{name}-{r['index']}", "cluster": f"gsm8k-{name}-{r['index']}",
                         "context": "\n\n".join(demo(d) for d in demos), "question": r["question"],
                         "answer": str(r["answer"])})
        write_jsonl(os.path.join(out_dir, f"gsm8k_{name}.jsonl"), rows)
        print(f"gsm8k {name}: {len(rows)} questions, {k} demonstrations each")


def prepare_oneruler(out_dir, length: int = 4096, n_test: int = 100, n_dev: int = 30, seed: int = 0):
    """OneRuler, Vietnamese (mirror starbix/oneruler of the generated sets; generator: OneRuler,
    arXiv 2503.01996, MIT). 7 subtasks (needle-in-a-haystack variants and common-word
    extraction). Only the haystack / word list is compressed; the instruction, the question and
    the answer format stay verbatim in `template`."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("starbix/oneruler", f"vi_{length}/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    rows = {"dev": [], "test": []}
    for subtask, group in df.groupby("task"):
        idx = random.Random(f"{seed}-{subtask}").sample(list(group.index), n_test + n_dev)
        for split, chosen in (("test", idx[:n_test]), ("dev", idx[n_test:])):
            for i in chosen:
                r = df.loc[i]
                if "<Văn bản>" in r.context:  # niah: instruction + <Văn bản> haystack </Văn bản>
                    head, rest = r.context.split("<Văn bản>", 1)
                    context = rest.split("</Văn bản>", 1)[0].strip()
                    template = f"{head}<Văn bản>\n{{context}}\n</Văn bản>\n\n{r.question.strip()}\n\n{r.answer_prefix}"
                else:  # cwe: the word list goes into the empty <Danh sách></Danh sách> slot
                    context = r.context.strip()
                    template = re.sub(r"<Danh sách>\s*</Danh sách>", "<Danh sách>\n{context}\n</Danh sách>",
                                      r.question) + r.answer_prefix
                if template.count("{context}") != 1:
                    raise ValueError(f"no context slot in the OneRuler template of row {i} ({subtask})")
                rows[split].append({"id": f"oneruler{length}-{subtask}-{i}", "cluster": f"oneruler{length}-{i}",
                                    "context": context, "template": template, "answers": list(r.answer),
                                    "subtask": subtask})
    for split, items in rows.items():
        write_jsonl(os.path.join(out_dir, f"oneruler_{split}.jsonl"), items)
        print(f"oneruler {split}: {len(items)} examples ({length} tokens, 7 subtasks)")


def prepare_paragraphs(out_path, n_per_source: int = 5000, min_syl: int = 300, max_syl: int = 1500,
                       seed: int = 0):
    """Distillation inputs (DATN §3.2): 300–1500 syllables, whole sentences only.
    Overlap with evaluation sets is removed later by scripts/leak_filter.py, which checks
    both 13-token n-grams and stable VietNews article IDs."""
    from datasets import load_dataset

    rows, rng = [], random.Random(seed)
    news = load_dataset("nam194/vietnews", split="train")
    for i in rng.sample(range(len(news)), len(news)):
        text = take_sentences(detokenize_vietnews(news[i]["article"]), max_syl)
        if len(text.split()) >= min_syl:
            rows.append({"id": f"news-{news[i]['guid']}", "source": "vietnews", "text": text})
        if len(rows) >= n_per_source:
            break

    wiki = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    n_news = len(rows)
    for article in wiki.shuffle(seed=seed, buffer_size=10_000):
        # one chunk per article: consecutive paragraphs, up to max_syl syllables
        paragraphs = [p for p in article["text"].split("\n") if len(p.split()) >= 20]
        text = take_sentences(" ".join(paragraphs), max_syl)
        if len(text.split()) >= min_syl:
            rows.append({"id": f"wiki-{article['id']}", "source": "wikipedia", "text": text})
        if len(rows) - n_news >= n_per_source:
            break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_jsonl(out_path, rows)
    print(f"paragraphs: {n_news} VietNews + {len(rows) - n_news} Wikipedia -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["belebele", "vinli", "vimmrc", "vietnews", "viquad", "gsm8k", "oneruler",
                                        "paragraphs"])
    ap.add_argument("--out-dir", default="data/tasks")
    ap.add_argument("--paragraphs-out", default="data/distill/paragraphs.jsonl")
    ap.add_argument("--n-per-source", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "paragraphs":
        prepare_paragraphs(args.paragraphs_out, args.n_per_source)
    else:
        {"belebele": prepare_belebele, "vinli": prepare_vinli, "vimmrc": prepare_vimmrc,
         "vietnews": prepare_vietnews, "viquad": prepare_viquad, "gsm8k": prepare_gsm8k,
         "oneruler": prepare_oneruler}[args.dataset](args.out_dir)


if __name__ == "__main__":
    main()
