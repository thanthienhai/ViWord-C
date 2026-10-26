"""Unit scorers of direction B: how relevant each unit (clause) of a legal RAG context is to the question.

`CrossEncoderUnitScorer` reads any sequence-classification checkpoint with one output: a multilingual reranker used
as it is (the proposal's "reranker" baseline, e.g. BAAI/bge-reranker-v2-m3) or the scorer trained here on utility or
gold labels (`train`). `BM25UnitScorer` is the lexical control (the units of the context ranked by BM25, as `qunit`).

The scorer starts from a multilingual reranker (BAAI/bge-reranker-v2-m3 on the H100 runs), so that "trained −
reranker" isolates what the labels add. On 40 smoke questions, xlm-roberta-base (an MLM checkpoint) did not learn
steadily (train loss 2.38 → 1.98 → 2.23), while a small mMARCO reranker did (dev top-1 gold 0.70 → 0.80).

Training is listwise: the units of one question form a group, the scores go through a softmax over the group, and
the loss is the cross-entropy against a target distribution over the same units, softmax(utility / tau) for utility
labels (viword.unit_labels) and uniform over the gold units for gold labels (H4).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

LABELS = ("utility", "gold")


class BM25UnitScorer:
    name = "bm25"

    def scores(self, question: str, units: list[str]) -> list[float]:
        from .legal_index import BM25
        return [float(s) for s in BM25.fit(units).scores(question or "")]


class CrossEncoderUnitScorer:
    def __init__(self, model: str, max_length: int = 512, batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name = torch, model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_length, self.batch_size = max_length, batch_size
        self._cache: dict = {}  # the same context is scored once for all ratios

    def scores(self, question: str, units: list[str]) -> list[float]:
        key = (question, tuple(units))
        if key not in self._cache:
            out = []
            for k in range(0, len(units), self.batch_size):
                batch = units[k:k + self.batch_size]
                enc = self.tokenizer([question] * len(batch), batch, truncation="only_second",
                                     max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    out += self.model(**enc).logits[:, 0].float().tolist()
            self._cache[key] = out
        return list(self._cache[key])


def targets(row: dict, label: str, tau: float) -> list[float]:
    """Target distribution over the units of one label row."""
    if label == "gold":
        gold = [float(g) for g in row["gold"]]
        return [g / sum(gold) for g in gold] if sum(gold) else []
    u = np.asarray(row["utility"], dtype=float) / tau
    e = np.exp(u - u.max())
    return (e / e.sum()).tolist()


def groups(rows: list[dict], label: str, tau: float) -> list[dict]:
    """One training group per question: (question, unit texts, target distribution). Rows without a gold unit give
    no group for gold labels."""
    out = []
    for r in rows:
        t = targets(r, label, tau)
        if t:
            out.append({"id": r["id"], "question": r["question"], "texts": r["texts"], "target": t,
                        "gold": r["gold"]})
    return out


def listwise_loss(scores, target, torch):
    """Cross-entropy between the target distribution and the softmax of the scores over one group."""
    return -(target * torch.log_softmax(scores, dim=-1)).sum()


def split_by_cluster(rows: list[dict], dev_share: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Train/dev rows of the label file, split by gold article with a fixed seed (independent of the training
    seed), so that the questions of one article fall on one side."""
    clusters = sorted({r["cluster"] for r in rows})
    random.Random(0).shuffle(clusters)
    dev, n = set(), 0
    for c in clusters:
        if n >= dev_share * len(rows):
            break
        dev.add(c)
        n += sum(r["cluster"] == c for r in rows)
    return [r for r in rows if r["cluster"] not in dev], [r for r in rows if r["cluster"] in dev]


def _group_scores(model, tokenizer, group, max_length, device):
    enc = tokenizer([group["question"]] * len(group["texts"]), group["texts"], truncation="only_second",
                    max_length=max_length, padding=True, return_tensors="pt").to(device)
    return model(**enc).logits[:, 0].float()


def evaluate(model, tokenizer, dev_groups, max_length, device, torch) -> dict:
    """Dev loss, and how often the top-scored unit is a gold unit (top1_gold) and the reciprocal rank of the first
    gold unit (mrr_gold)."""
    model.eval()
    losses, top1, mrr = [], [], []
    with torch.no_grad():
        for g in dev_groups:
            s = _group_scores(model, tokenizer, g, max_length, device)
            losses.append(listwise_loss(s, torch.tensor(g["target"], device=device), torch).item())
            order = torch.argsort(s, descending=True).tolist()
            if any(g["gold"]):
                top1.append(float(g["gold"][order[0]]))
                mrr.append(1.0 / (1 + next(k for k, i in enumerate(order) if g["gold"][i])))
    model.train()
    return {"loss": float(np.mean(losses)), "top1_gold": float(np.mean(top1)), "mrr_gold": float(np.mean(mrr))}


def train(rows: list[dict], out_dir: str, label: str = "utility", model_name: str = "BAAI/bge-reranker-v2-m3",
          epochs: int = 2, lr: float = 1e-5, groups_per_step: int = 4, max_length: int = 384, tau: float = 0.5,
          seed: int = 0, dev_share: float = 0.1) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .train import _amp_dtype

    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                               torch_dtype=torch.float32).to(device)
    train_rows, dev_rows = split_by_cluster(rows, dev_share)
    train_groups, dev_groups = groups(train_rows, label, tau), groups(dev_rows, label, tau)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(train_groups) + groups_per_step - 1) // groups_per_step)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))
    amp_dtype = _amp_dtype(torch, device)
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype == torch.float16)
    rng, history, best = random.Random(seed), [], float("inf")
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        order, train_losses = list(range(len(train_groups))), []
        rng.shuffle(order)
        for k in range(0, len(order), groups_per_step):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                loss = sum(listwise_loss(_group_scores(model, tokenizer, train_groups[i], max_length, device),
                                         torch.tensor(train_groups[i]["target"], device=device), torch)
                           for i in order[k:k + groups_per_step]) / len(order[k:k + groups_per_step])
            train_losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        dev = evaluate(model, tokenizer, dev_groups, max_length, device, torch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **dev})
        print(f"epoch {epoch}: train loss {np.mean(train_losses):.4f}  dev loss {dev['loss']:.4f}  "
              f"top-1 gold {dev['top1_gold']:.3f}  MRR gold {dev['mrr_gold']:.3f}")
        if dev["loss"] < best:
            best = dev["loss"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    meta = {"label": label, "model_name": model_name, "epochs": epochs, "lr": lr, "groups_per_step": groups_per_step,
            "max_length": max_length, "tau": tau, "seed": seed, "dev_share": dev_share,
            "amp_dtype": str(amp_dtype).removeprefix("torch.") if device == "cuda" else "float32",
            "n_train_groups": len(train_groups), "n_dev_groups": len(dev_groups),
            "n_train_units": sum(len(g["texts"]) for g in train_groups), "history": history}
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
