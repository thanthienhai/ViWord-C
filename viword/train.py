"""Training the keep/drop encoder (DATN §3.3).

One model type serves every cell of the 2×2 design (DATN §2.3): the encoder always
outputs one logit per syllable (mean over its subwords). What changes is the target:

  label_unit="syllable": hard syllable labels            -> LLMLingua-2-vi
  label_unit="word":     soft word label of the syllable's word, copied to each of its
                         syllables                         -> ViWord-C

The decision unit (syllable or word) is chosen later, at compression time
(`compressor.SelectionConfig.unit`). Loss: BCE with logits and a positive-class weight;
checkpoint selection by dev loss (works with soft labels), without any reader.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from .model import make_windows
from .segment import doc_from_json, flatten, syllables_of


def syllable_targets(record: dict, label_unit: str) -> list[float]:
    if label_unit == "syllable":
        return [float(x) for x in record["syllable_labels"]]
    words = flatten(doc_from_json(record["doc"]))
    return [label for w, label in zip(words, record["word_labels"]) for _ in w.syllables]


def build_items(records: list[dict], tokenizer, label_unit: str, max_length: int) -> list[dict]:
    """Split every document into sentence windows; one training item per window."""
    items = []
    for r in records:
        doc = doc_from_json(r["doc"])
        sylls = syllables_of(flatten(doc))
        targets = syllable_targets(r, label_unit)
        for start, end in make_windows(doc, tokenizer, max_length):
            enc = tokenizer(sylls[start:end], is_split_into_words=True, truncation=True,
                            max_length=max_length)
            items.append({
                "input_ids": enc["input_ids"],
                "word_ids": [-1 if i is None else i for i in enc.word_ids()],
                "targets": targets[start:end],
            })
    return items


def _batches(items: list[dict], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for k in range(0, len(order), batch_size):
        yield [items[i] for i in order[k:k + batch_size]]


def _batch_loss(model, batch, pad_id, pos_weight, device, torch):
    """Mean BCE over syllables; syllable logit = mean of its subword logits."""
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for b, x in enumerate(batch):
        ids[b, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
        mask[b, :len(x["input_ids"])] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[..., 0].float()

    unit_logits, unit_targets = [], []
    for b, x in enumerate(batch):
        wid = torch.tensor(x["word_ids"], device=device)
        valid = wid >= 0
        n, sub_logits = len(x["targets"]), logits[b, :len(wid)][valid]
        sums = torch.zeros(n, device=device).index_add(0, wid[valid], sub_logits)
        counts = torch.zeros(n, device=device).index_add(0, wid[valid], torch.ones_like(sub_logits))
        seen = counts > 0  # syllables cut off by truncation have no subwords
        unit_logits.append((sums / counts.clamp(min=1))[seen])
        unit_targets.append(torch.tensor(x["targets"], device=device)[seen])
    logit, target = torch.cat(unit_logits), torch.cat(unit_targets)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, target, pos_weight=torch.tensor(pos_weight, device=device))
    return loss, logit.detach(), target


def evaluate(model, items, pad_id, pos_weight, device, batch_size, torch) -> dict:
    model.eval()
    losses, tp, fp, fn = [], 0, 0, 0
    with torch.no_grad():
        for batch in _batches(items, batch_size, False, random.Random(0)):
            loss, logit, target = _batch_loss(model, batch, pad_id, pos_weight, device, torch)
            losses.append(loss.item())
            hard = (target == 0) | (target == 1)  # F1 only on units with a hard label
            pred, gold = (logit > 0)[hard], (target == 1)[hard]
            tp += int((pred & gold).sum())
            fp += int((pred & ~gold).sum())
            fn += int((~pred & gold).sum())
    model.train()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"loss": float(np.mean(losses)), "f1": f1}


def train(train_records: list[dict], dev_records: list[dict], out_dir: str, label_unit: str,
          model_name: str = "xlm-roberta-large", epochs: int = 3, lr: float = 1e-5,
          batch_size: int = 8, max_length: int = 512, seed: int = 0) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=1).to(device)
    train_items = build_items(train_records, tokenizer, label_unit, max_length)
    dev_items = build_items(dev_records, tokenizer, label_unit, max_length)

    keep_rate = float(np.mean([t for x in train_items for t in x["targets"]]))
    pos_weight = (1 - keep_rate) / keep_rate  # class weighting (DATN §3.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(train_items) + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 - s / steps)

    rng, history, best = random.Random(seed), [], float("inf")
    for epoch in range(1, epochs + 1):
        for batch in _batches(train_items, batch_size, True, rng):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, _, _ = _batch_loss(model, batch, tokenizer.pad_token_id, pos_weight, device, torch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        dev = evaluate(model, dev_items, tokenizer.pad_token_id, pos_weight, device, batch_size, torch)
        history.append({"epoch": epoch, **dev})
        print(f"epoch {epoch}: dev loss {dev['loss']:.4f}  dev F1 {dev['f1']:.4f}")
        if dev["loss"] < best:
            best = dev["loss"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    meta = {"label_unit": label_unit, "model_name": model_name, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "max_length": max_length, "seed": seed,
            "train_keep_rate": keep_rate, "n_train_windows": len(train_items),
            "n_dev_windows": len(dev_items), "history": history}
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
