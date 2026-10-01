"""Word segmentation and syllable/word bookkeeping.

A document is a list of sentences and a sentence is a list of `Word`s. Every compressor
decision is expressed as a boolean mask over syllables: syllable-level systems decide per
syllable, word-level systems decide per word and expand the decision to its syllables.
All systems render their output through the same `render()`, so punctuation and spacing
are handled identically everywhere (DATN §3.1).
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Word:
    text: str  # surface form, syllables separated by a space, e.g. "học sinh"
    pos: str = ""  # POS tag (VLSP tag set, e.g. N, V, A, R, M, CH)
    ner: str = "O"  # NER label, "O" outside entities

    @property
    def syllables(self) -> list[str]:
        return self.text.split(" ")

    @property
    def is_punct(self) -> bool:
        return all(unicodedata.category(c).startswith("P") for c in self.text)


Document = list[list[Word]]  # list of sentences


def normalize(text: str) -> str:
    """NFC normalization. Vietnamese text in NFD breaks both segmentation and token counts."""
    return unicodedata.normalize("NFC", text)


def flatten(doc: Document) -> list[Word]:
    return [w for sentence in doc for w in sentence]


def syllables_of(words: list[Word]) -> list[str]:
    return [s for w in words for s in w.syllables]


def word_index_of_syllables(words: list[Word]) -> list[int]:
    """For every syllable, the index of the word it belongs to."""
    return [i for i, w in enumerate(words) for _ in w.syllables]


def expand_word_mask(words: list[Word], word_mask: list[bool]) -> list[bool]:
    return [keep for w, keep in zip(words, word_mask) for _ in w.syllables]


# ---------------------------------------------------------------------------- rendering

NO_SPACE_BEFORE = set(",.;:!?)]}…%")
NO_SPACE_AFTER = set("([{")


def render(words: list[Word], syllable_mask: list[bool]) -> str:
    """Join the kept syllables back into text, in the original order."""
    pieces, i = [], 0
    for w in words:
        n = len(w.syllables)
        kept = [s for s, keep in zip(w.syllables, syllable_mask[i:i + n]) if keep]
        i += n
        if kept:
            pieces.append(" ".join(kept))
    out = ""
    for piece in pieces:
        if not out:
            out = piece
        elif piece[0] in NO_SPACE_BEFORE or out[-1] in NO_SPACE_AFTER:
            out += piece
        else:
            out += " " + piece
    return out


def render_all(words: list[Word]) -> str:
    return render(words, [True] * len(syllables_of(words)))


# ---------------------------------------------------------------------------- alignment

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def simple_tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(normalize(text))]


def align_compressed(words: list[Word], compressed_text: str) -> tuple[list[bool], float]:
    """Map an extractive compression back onto the original syllables.

    Both sides are split into the same letter/punctuation tokens and matched with
    difflib (longest matching blocks). A syllable counts as kept if any of its tokens is
    matched. Returns (syllable_mask, unmatched_ratio), where unmatched_ratio is the share of
    output *words* not found in the source; a high value means the compressor rewrote text
    instead of only deleting it (used to filter teacher outputs).
    """
    source, owner = [], []
    for si, syllable in enumerate(syllables_of(words)):
        for t in simple_tokens(syllable):
            source.append(t)
            owner.append(si)
    target = simple_tokens(compressed_text)

    mask = [False] * len(syllables_of(words))
    matched_target = set()
    for block in SequenceMatcher(a=source, b=target, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            mask[owner[block.a + k]] = True
            matched_target.add(block.b + k)
    # rewriting is measured on words only: inserted punctuation does not change the labels
    words = [k for k, t in enumerate(target) if t[0].isalnum()]
    unmatched_ratio = sum(k not in matched_target for k in words) / len(words) if words else 0.0
    return mask, unmatched_ratio


# ---------------------------------------------------------------------------- serialization

def doc_to_json(doc: Document) -> list[list[list[str]]]:
    return [[[w.text, w.pos, w.ner] for w in sentence] for sentence in doc]


def doc_from_json(data: list[list[list[str]]]) -> Document:
    return [[Word(*item) for item in sentence] for sentence in data]


# ---------------------------------------------------------------------------- segmenters

VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
VNCORENLP_FILES = ["VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab",
                   "models/wordsegmenter/wordsegmenter.rdr", "models/postagger/vi-tagger",
                   "models/ner/vi-500brownclusters.xz", "models/ner/vi-ner.xz",
                   "models/ner/vi-pretrainedembeddings.xz", "models/dep/vi-dep.xz"]


def download_vncorenlp(save_dir: str) -> None:
    """Same files as py_vncorenlp.download_model, fetched with urllib (that function
    shells out to `wget`, which is missing on Windows)."""
    import urllib.request

    for name in VNCORENLP_FILES:
        path = os.path.join(save_dir, name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(VNCORENLP_URL + name, path + ".part")
            os.replace(path + ".part", path)  # never leave a truncated file behind


class VnCoreNLPSegmenter:
    """RDRSegmenter + POS + NER from VnCoreNLP (main segmenter, needs Java)."""

    def __init__(self, save_dir: str = "models/vncorenlp"):
        import py_vncorenlp

        save_dir = os.path.abspath(save_dir)
        download_vncorenlp(save_dir)
        cwd = os.getcwd()
        self.model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner"], save_dir=save_dir)
        os.chdir(cwd)  # py_vncorenlp changes the working directory on load

    def segment(self, text: str) -> Document:
        annotated = self.model.annotate_text(normalize(text))
        return [
            [Word(t["wordForm"].replace("_", " "), t["posTag"], t["nerLabel"]) for t in sentence]
            for _, sentence in sorted(annotated.items())
        ]


class UndertheseaSegmenter:
    """underthesea word segmentation + POS + NER (segmenter-sensitivity analysis)."""

    def segment(self, text: str) -> Document:
        from underthesea import ner, sent_tokenize

        doc = []
        for sentence in sent_tokenize(normalize(text)):
            doc.append([Word(word, pos, label or "O") for word, pos, _chunk, label in ner(sentence)])
        return doc


class PyviSegmenter:
    """pyvi word segmentation + POS, no NER (segmenter-sensitivity analysis)."""

    def segment(self, text: str) -> Document:
        from pyvi import ViPosTagger, ViTokenizer

        doc = []
        for sentence in re.split(r"(?<=[.!?])\s+", normalize(text).strip()):
            if sentence:
                words, tags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
                doc.append([Word(w.replace("_", " "), tag) for w, tag in zip(words, tags)])
        return doc


def get_segmenter(name: str, **kwargs):
    return {"vncorenlp": VnCoreNLPSegmenter, "underthesea": UndertheseaSegmenter,
            "pyvi": PyviSegmenter}[name](**kwargs)
