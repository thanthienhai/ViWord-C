"""Training the keep/drop encoder (DATN §3.3).

One model type serves every cell of the 2×2 design (DATN §2.3): the encoder always
outputs one logit per syllable (mean over its subwords). What changes is the target:

  label_unit="syllable": hard syllable labels            -> LLMLingua-2-vi
  label_unit="word":     soft word label of the syllable's word, copied to each of its
                         syllables                         -> ViWord-C

Two controls separate "soft labels" from "word boundaries" (cells iii-rand and iv-wordloss):

  label_unit="syllable_softrand": the hard syllable labels, with as many syllables set to 0.5
                         as the word view softens (the syllables of the words the teacher cut),
                         but at random positions: the same number of kept and of dropped
                         syllables, drawn among the non-punctuation syllables of the record.
                         If this cell matches word labels, the "label unit" effect is label
                         smoothing, not word boundaries.
  label_unit="word_pooled": the word view, with the loss on the word: syllable logits are
                         pooled to one mean logit per word (as `model.pool_to_words` does at
                         inference) and BCE is taken against the soft word label, one term
                         per word. This is the full "word as the unit" variant.

The decision unit (syllable or word) is chosen later, at compression time
(`compressor.SelectionConfig.unit`). Loss is BCE with logits and a positive-class weight,
computed per visible syllable (per visible word for `word_pooled`). For `label_unit="word"`,
the soft word target is copied to each syllable before this loss; the word-pooled inference
logit is not used in training. Checkpoint selection is by dev loss, without any reader.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from .model import make_windows
from .segment import doc_from_json, flatten, syllables_of


LABEL_UNITS = ("syllable", "word", "syllable_softrand", "word_pooled")


def syllable_targets(record: dict, label_unit: str, seed: int = 0) -> list[float]:
    if label_unit == "syllable":
        return [float(x) for x in record["syllable_labels"]]
    if label_unit == "syllable_softrand":
        return softrand_targets(record, seed)
    words = flatten(doc_from_json(record["doc"]))
    return [label for w, label in zip(words, record["word_labels"]) for _ in w.syllables]


def softrand_targets(record: dict, seed: int = 0) -> list[float]:
    """Hard syllable labels with random 0.5s, matched to the word view (cell iii-rand).

    The word view softens every syllable of a word the teacher cut (0 < word label < 1):
    k1 of those syllables were kept and k0 dropped. Here k1 random kept and k0 random
    dropped non-punctuation syllables of the same record are set to 0.5 instead, so the
    number of soft targets and the classes they come from match, but they no longer sit on
    word boundaries. Drawn per record from (seed, id, rate), so it is reproducible."""
    words = flatten(doc_from_json(record["doc"]))
    labels = [float(x) for x in record["syllable_labels"]]
    k = {0: 0, 1: 0}
    pool = {0: [], 1: []}
    i = 0
    for w, word_label in zip(words, record["word_labels"]):
        n = len(w.syllables)
        if 0 < word_label < 1:
            for j in range(i, i + n):
                k[int(labels[j])] += 1
        if not w.is_punct:
            for j in range(i, i + n):
                pool[int(labels[j])].append(j)
        i += n
    rng = random.Random(f"softrand:{seed}:{record.get('id', '')}:{record.get('rate', '')}")
    for cls in (0, 1):
        for j in rng.sample(pool[cls], min(k[cls], len(pool[cls]))):
            labels[j] = 0.5
    return labels


def word_groups(record: dict) -> list[int]:
    """Index of the word each syllable belongs to (for `word_pooled`)."""
    words = flatten(doc_from_json(record["doc"]))
    return [wi for wi, w in enumerate(words) for _ in w.syllables]


def build_items(records: list[dict], tokenizer, label_unit: str, max_length: int,
                seed: int = 0) -> list[dict]:
    """Split every document into sentence windows; one training item per window.
    For `word_pooled`, `groups` maps each syllable of the window to a window-local word
    index (a word cut by a window edge is pooled over its visible syllables)."""
    items = []
    for r in records:
        doc = doc_from_json(r["doc"])
        sylls = syllables_of(flatten(doc))
        targets = syllable_targets(r, label_unit, seed)
        groups = word_groups(r) if label_unit == "word_pooled" else None
        for start, end in make_windows(doc, tokenizer, max_length):
            enc = tokenizer(sylls[start:end], is_split_into_words=True, truncation=True,
                            max_length=max_length)
            item = {
                "input_ids": enc["input_ids"],
                "word_ids": [-1 if i is None else i for i in enc.word_ids()],
                "targets": targets[start:end],
            }
            if groups is not None:
                first = groups[start]
                item["groups"] = [g - first for g in groups[start:end]]
            items.append(item)
    return items


def loss_unit_targets(items: list[dict]) -> list[float]:
    """Targets as the loss sees them: per syllable, or per word for `word_pooled` items."""
    out = []
    for x in items:
        if "groups" in x:
            seen = {}
            for g, t in zip(x["groups"], x["targets"]):
                seen.setdefault(g, t)
            out += list(seen.values())
        else:
            out += x["targets"]
    return out


def _batches(items: list[dict], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for k in range(0, len(order), batch_size):
        yield [items[i] for i in order[k:k + batch_size]]


def _batch_loss(model, batch, pad_id, pos_weight, device, torch):
    """Mean BCE over visible syllables; word targets are already copied to syllables.
    Items with `groups` (word_pooled) pool their syllable logits by word first, so the loss
    has one term per visible word, against the soft word label."""
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
        syl_logits = (sums / counts.clamp(min=1))[seen]
        syl_targets = torch.tensor(x["targets"], device=device)[seen]
        if "groups" in x:  # word_pooled: mean of the word's syllable logits, as pool_to_words
            syl_logits, syl_targets = pool_by_word(syl_logits, syl_targets,
                                                   torch.tensor(x["groups"], device=device)[seen], torch)
        unit_logits.append(syl_logits)
        unit_targets.append(syl_targets)
    logit, target = torch.cat(unit_logits), torch.cat(unit_targets)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, target, pos_weight=torch.tensor(pos_weight, device=device))
    return loss, logit.detach(), target


def pool_by_word(logits, targets, groups, torch):
    """Mean logit and mean target per word present in `groups`. Every syllable of a word
    carries the same soft word label, so the mean target is that label."""
    if len(groups) == 0:
        return logits, targets
    n = int(groups.max()) + 1
    counts = torch.zeros(n, device=logits.device).index_add(0, groups, torch.ones_like(logits))
    present = counts > 0
    w_logits = torch.zeros(n, device=logits.device).index_add(0, groups, logits) / counts.clamp(min=1)
    w_targets = torch.zeros(n, device=logits.device).index_add(0, groups, targets.float()) / counts.clamp(min=1)
    return w_logits[present], w_targets[present]


def _amp_dtype(torch, device: str):
    """bf16 where the GPU has it natively (compute capability ≥ 8, e.g. A100/H100), else fp16."""
    if device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


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
    # fp32 master weights whatever the checkpoint's dtype (some are saved in fp16, and recent
    # transformers load that dtype by default); autocast handles the low precision
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=1,
                                                            torch_dtype=torch.float32).to(device)
    if label_unit not in LABEL_UNITS:
        raise ValueError(f"label_unit must be one of {LABEL_UNITS}, got {label_unit!r}")
    train_items = build_items(train_records, tokenizer, label_unit, max_length, seed)
    dev_items = build_items(dev_records, tokenizer, label_unit, max_length, seed)

    # class weighting (DATN §3.3), over the units the loss is taken on
    unit_targets = loss_unit_targets(train_items)
    keep_rate = float(np.mean(unit_targets))
    pos_weight = (1 - keep_rate) / keep_rate
    soft_share = float(np.mean([0 < t < 1 for t in unit_targets]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(train_items) + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 - s / steps)

    amp_dtype = _amp_dtype(torch, device)
    # fp16 (GPUs without native bf16, e.g. T4) needs loss scaling; bf16 does not
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype == torch.float16)
    rng, history, best = random.Random(seed), [], float("inf")
    for epoch in range(1, epochs + 1):
        for batch in _batches(train_items, batch_size, True, rng):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                loss, _, _ = _batch_loss(model, batch, tokenizer.pad_token_id, pos_weight, device, torch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
            "amp_dtype": str(amp_dtype).removeprefix("torch.") if device == "cuda" else "float32",
            "train_keep_rate": keep_rate, "train_soft_target_share": soft_share,
            "n_train_windows": len(train_items),
            "n_dev_windows": len(dev_items), "history": history}
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
