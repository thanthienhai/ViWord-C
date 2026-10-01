"""Download public releases and write them in the normalized format of data/README.md.

Usage:
  python scripts/prepare_data.py belebele     # data/tasks/belebele.jsonl (900 questions, no dev split)
  python scripts/prepare_data.py vinli        # data/tasks/vinli_{train,dev,test}.jsonl
  python scripts/prepare_data.py vimmrc       # data/tasks/vimmrc_{dev,test}.jsonl
  python scripts/prepare_data.py vietnews     # data/tasks/vietnews_{dev,test}.jsonl (1000 sampled each)
  python scripts/prepare_data.py paragraphs   # data/distill/paragraphs.jsonl (VietNews train + Wikipedia)

Sources (record the exact revision in data/README.md when reporting results):
  belebele   facebook/belebele (vie_Latn + parallel eng_Latn)
  vinli      presencesw/vinli_4_label: unofficial mirror of ViNLI (Huynh et al., COLING 2022);
             split sizes match the paper
  vimmrc     ura-hcmut/ViMMRC: mirror of ViMMRC (the official uitnlp/vimmrc2.0 is gated)
  vietnews   nam194/vietnews: word-segmented release ("Cơ_quan", "27/3 ,"), detokenized here
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
    from viword.data import belebele_from_hub
    belebele_from_hub(os.path.join(out_dir, "belebele.jsonl"))


def prepare_vinli(out_dir):
    from datasets import load_dataset

    data = load_dataset("presencesw/vinli_4_label")
    for split in ("train", "dev", "test"):
        # the mirror has no premise id: the premise text defines the bootstrap cluster
        rows = [{"id": f"vinli-{split}-{i}", "cluster": short_hash(r["sentence1"]), "premise": r["sentence1"],
                 "hypothesis": r["sentence2"], "label": r["gold_label"]} for i, r in enumerate(data[split])]
        write_jsonl(os.path.join(out_dir, f"vinli_{split}.jsonl"), rows)
        print(f"vinli {split}: {len(rows)} pairs, {len({r['cluster'] for r in rows})} premises")


def prepare_vimmrc(out_dir):
    from datasets import load_dataset

    data = load_dataset("ura-hcmut/ViMMRC")
    for split, name in (("validation", "dev"), ("test", "test")):
        rows, poems = [], 0
        for i, r in enumerate(data[split]):
            if is_poem(r["article"]):
                poems += 1
                continue
            choices = ast.literal_eval(r["options"])
            rows.append({"id": f"vimmrc-{name}-{i}", "cluster": short_hash(r["article"]),
                         "context": r["article"], "question": r["question"],
                         "choices": choices, "answer": r["answer"].strip()})
        write_jsonl(os.path.join(out_dir, f"vimmrc_{name}.jsonl"), rows)
        print(f"vimmrc {name}: {len(rows)} questions, {len({r['cluster'] for r in rows})} passages, "
              f"{poems} poem questions dropped")


def prepare_vietnews(out_dir, n: int = 1000, seed: int = 0):
    from datasets import load_dataset

    data = load_dataset("nam194/vietnews")
    for split, name in (("validation", "dev"), ("test", "test")):
        rows = [{"id": f"vietnews-{name}-{r['guid']}", "cluster": f"vietnews-{name}-{r['guid']}",
                 "context": detokenize_vietnews(r["article"]), "reference": detokenize_vietnews(r["abstract"])}
                for r in data[split] if r["article"] and r["abstract"]]
        rows = random.Random(seed).sample(rows, min(n, len(rows)))
        write_jsonl(os.path.join(out_dir, f"vietnews_{name}.jsonl"), rows)
        print(f"vietnews {name}: {len(rows)} articles sampled")


def prepare_paragraphs(out_path, n_per_source: int = 5000, min_syl: int = 300, max_syl: int = 1500,
                       seed: int = 0):
    """Distillation inputs (DATN §3.2): 300–1500 syllables, whole sentences only.
    Overlap with evaluation sets is removed later by the leak check in scripts/distill.py."""
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
    ap.add_argument("dataset", choices=["belebele", "vinli", "vimmrc", "vietnews", "paragraphs"])
    ap.add_argument("--out-dir", default="data/tasks")
    ap.add_argument("--paragraphs-out", default="data/distill/paragraphs.jsonl")
    ap.add_argument("--n-per-source", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "paragraphs":
        prepare_paragraphs(args.paragraphs_out, args.n_per_source)
    else:
        {"belebele": prepare_belebele, "vinli": prepare_vinli, "vimmrc": prepare_vimmrc,
         "vietnews": prepare_vietnews}[args.dataset](args.out_dir)


if __name__ == "__main__":
    main()
