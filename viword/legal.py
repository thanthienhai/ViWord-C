"""Vietnamese legal documents (direction B): HTML → lines → Điều / Khoản / Điểm, and citations.

A normative document is a preamble, then articles ("Điều 5. Tiêu đề"), each with an optional lead-in, numbered
clauses ("1.", "2.", …) and lettered points ("a)", "b)", …, in the Vietnamese order a b c d đ e g h …), then a
closing part (adoption sentence, "Nơi nhận:", signatures) and appendices. The parser keeps that tree.

Three traps shape it:
- Amending documents quote the new text ("1. Sửa đổi khoản 2 Điều 5 như sau: “2. … a) …”"), and a quote may span
  many lines and hold its own "Điều", clauses and points. A line that starts inside a quote, or with a quote mark,
  is never structure: it continues the current unit.
- A numbered or lettered line is a new clause or point only if it continues the sequence (clause n after n − 1,
  point after the previous letter); anything else (a list inside a point, a table row) is continuation.
- Citations name a document by number ("Nghị định 88/2023/NĐ-CP") or by name and year ("Luật Đất đai 2024");
  `resolve` maps both to a vbpl id through the metadata.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser

POINT_LETTERS = "a b c d đ e g h i k l m n o p q r s t u v x y".split()
_NEXT_LETTER = {a: b for a, b in zip(POINT_LETTERS, POINT_LETTERS[1:])}
DOC_TYPES = ("Hiến pháp", "Bộ luật", "Luật", "Pháp lệnh", "Nghị quyết liên tịch", "Nghị quyết", "Nghị định",
             "Thông tư liên tịch", "Thông tư", "Quyết định", "Chỉ thị", "Lệnh")
NUMBER = r"\d+[a-zA-Z]?/\d{4}/[A-ZĐa-zđ0-9\-]+"

_ARTICLE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*[\.:]\s*(.*)$")
_CLAUSE = re.compile(r"^(\d+)\.\s+(.*)$")
_POINT = re.compile(r"^([a-zđ])\)\s+(.*)$")
_PART = re.compile(r"^(Phần\s+(thứ\s+\S+|[IVXLC\d]+)|Chương\s+[IVXLC\d]+|Mục\s+\d+|Tiểu mục\s+\d+)\b", re.I)
_ATTACHED = re.compile(r"^\(?\s*[Bb]an hành kèm theo\b")
# The closing part: adoption sentence, recipients, signature blocks (appendices come after them)
_CLOSING = re.compile(r"^(Nơi nhận\s*:|(Luật|Bộ luật|Pháp lệnh|Nghị quyết|Hiến pháp) này (đã )?được .{0,120}"
                      r"(thông qua|ban hành)|(TM|KT|TL|TUQ)\.\s)")
_OPEN, _CLOSE = "“", "”"


def normalize(text: str) -> str:
    return re.sub(r"[ \t ​]+", " ", unicodedata.normalize("NFC", text)).strip()


class _Lines(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "title"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "title"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_lines(html: str) -> list[str]:
    """Visible text of a document, one line per block element, whitespace collapsed, NFC."""
    p = _Lines()
    p.feed(html)
    p.close()
    return [line for line in (normalize(x) for x in "".join(p.parts).split("\n")) if line]


@dataclass
class Point:
    letter: str
    text: str


@dataclass
class Clause:
    number: int
    text: str  # the lead-in (and any continuation before the first point)
    points: list[Point] = field(default_factory=list)


@dataclass
class Article:
    number: str
    title: str
    part: str = ""  # the innermost Phần / Chương / Mục heading above it
    intro: str = ""  # lead-in before the first clause (or the whole body of an article without clauses)
    clauses: list[Clause] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)  # points directly under the article (no clauses)
    attachment: int = 0  # 0 = the document itself, k = its k-th attached text (Quy chế, Quy định, …)


@dataclass
class Attachment:
    title: str  # its upper-case name, e.g. "QUY CHẾ BẢO VỆ BÍ MẬT NHÀ NƯỚC TRONG NGÀNH THANH TRA"
    preamble: list[str] = field(default_factory=list)  # "(Ban hành kèm theo …)" and the lines up to its first article


@dataclass
class LegalDocument:
    preamble: list[str]
    articles: list[Article]
    closing: list[str]
    warnings: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)


def _quote_delta(line: str, depth: int) -> int:
    """Quote depth after `line`: “ opens, ” closes, a straight " toggles."""
    for ch in line:
        if ch == _OPEN:
            depth += 1
        elif ch == _CLOSE:
            depth = max(0, depth - 1)
        elif ch == '"':
            depth = depth - 1 if depth else 1
    return depth


def _join(a: str, b: str) -> str:
    return f"{a}\n{b}" if a else b


def _is_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.9 * len(letters)


_SIGNER = re.compile(r"^(TM|KT|TL|TUQ)\.\s|^(PHÓ |Q\. )?(BỘ TRƯỞNG|THỨ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|CHÁNH ÁN|"
                     r"VIỆN TRƯỞNG|TỔNG KIỂM TOÁN|TỔNG THANH TRA|THỐNG ĐỐC|CHỦ NHIỆM|TỔNG GIÁM ĐỐC|CỤC TRƯỞNG)\b")


def _pop_caps(lines: list[str]) -> list[str]:
    """Remove and return the upper-case lines at the end of `lines` (the name of an attached text), stopping
    at a signature line."""
    tail = []
    while lines and _is_caps(lines[-1]) and not _SIGNER.match(lines[-1]):
        tail.insert(0, lines.pop())
    return tail


def _drop_repeated_body(lines: list[str]) -> tuple[list[str], bool]:
    """Some crawled pages hold the body twice: cut the copy that repeats the first article heading and the
    lines after it."""
    first = next((i for i, line in enumerate(lines) if _ARTICLE.match(line)), None)
    if first is None:
        return lines, False
    for j in range(first + 1, len(lines)):
        k = min(8, j - first, len(lines) - j)
        if lines[j] == lines[first] and k > 1 and lines[j:j + k] == lines[first:first + k]:
            return lines[:j], True
    return lines, False


def parse_lines(lines: list[str]) -> LegalDocument:
    lines, repeated = _drop_repeated_body(lines)
    doc = LegalDocument([], [], [], ["repeated body dropped"] if repeated else [])
    part, depth, closed = "", 0, False
    art: Article | None = None
    clause: Clause | None = None

    def slot() -> tuple[object, str]:
        """The deepest open unit and its text attribute (after the points, the last point)."""
        if clause is not None:
            return (clause.points[-1], "text") if clause.points else (clause, "text")
        return (art.points[-1], "text") if art.points else (art, "intro")

    def append(text: str) -> None:
        obj, attr = slot()
        setattr(obj, attr, _join(getattr(obj, attr), text))

    def caps_tail() -> list[str]:
        """The upper-case lines that end the current unit or the closing part: an attached text's name."""
        if closed:
            return _pop_caps(doc.closing)
        if art is None:
            return []
        obj, attr = slot()
        kept = getattr(obj, attr).split("\n")
        tail = _pop_caps(kept)
        setattr(obj, attr, "\n".join(kept))
        return tail

    def attach(first_line: str | None) -> None:
        """Start an attached text. After a marker line its chapter headings are still to come; on a bare
        numbering restart the "Chương I" just read already belongs to it, so the part is kept."""
        nonlocal art, clause, closed, part
        doc.attachments.append(Attachment(" ".join(caps_tail()), [first_line] if first_line else []))
        art, clause, closed = None, None, False
        if first_line:
            part = ""

    previous = ""
    for line in lines:
        previous, before = line, previous
        quoted = depth > 0 or line[0] in (_OPEN, '"')
        m = None if quoted else _ARTICLE.match(line)
        if depth > 0 and art is not None:
            # a lost closing quote would swallow the rest of the document: the next article heading in
            # sequence, not itself quoted, ends the quote
            m2 = _ARTICLE.match(line)
            if m2 and line[0] not in (_OPEN, '"') and _next_number(art.number) == m2.group(1):
                doc.warnings.append(f"quote reset before Điều {m2.group(1)}")
                depth, quoted, m = 0, False, m2
        depth = _quote_delta(line, depth)
        if not quoted and _ATTACHED.match(line) and (art is not None or closed) \
                and (line[0] == "(" or _is_caps(before)):
            attach(line)  # "(Ban hành kèm theo Quyết định số …)" under a name: a text issued with the document
            continue
        if closed and not (m and m.group(1) == "1"):
            doc.closing.append(line)
            continue
        if m:
            if closed or (art is not None and m.group(1) == "1" and art.number != "1"):
                attach(None)  # numbering restarts after the signatures or without a marker: an attached text
            elif art is not None and _next_number(art.number) != m.group(1):
                doc.warnings.append(f"Điều {m.group(1)} after Điều {art.number}")
            art = Article(m.group(1), m.group(2), part, attachment=len(doc.attachments))
            doc.articles.append(art)
            clause = None
        elif art is None:
            if not quoted and _PART.match(line):
                part = line
            (doc.attachments[-1].preamble if doc.attachments else doc.preamble).append(line)
        elif not quoted and _CLOSING.match(line):
            closed = True
            doc.closing.append(line)
        elif not quoted and _PART.match(line) and len(line) < 200:
            part = line
        elif not quoted and (mc := _CLAUSE.match(line)) and _clause_follows(art, int(mc.group(1))):
            clause = Clause(int(mc.group(1)), mc.group(2))
            art.clauses.append(clause)
        elif not quoted and (mp := _POINT.match(line)) and _point_follows(art, clause, mp.group(1)):
            (clause.points if clause is not None else art.points).append(Point(mp.group(1), mp.group(2)))
        else:
            append(line)
    if depth:
        doc.warnings.append("unclosed quote at the end")
    return doc


def _next_number(number: str) -> str:
    digits = re.match(r"\d+", number).group(0)
    return str(int(digits) + 1)


def _clause_follows(art: Article, n: int) -> bool:
    if art.points:  # points directly under the article: a number now is a list inside the last point
        return False
    return (not art.clauses) or n == art.clauses[-1].number + 1


def _point_follows(art: Article, clause: Clause | None, letter: str) -> bool:
    points = clause.points if clause is not None else art.points
    if not points:
        return letter == "a"
    return _NEXT_LETTER.get(points[-1].letter) == letter


def parse_html(html: str) -> LegalDocument:
    return parse_lines(html_lines(html))


# ---------------------------------------------------------------------------------------------- units

@dataclass
class Unit:
    """A selection unit: one clause (with its points), or one article without clauses."""
    article: str
    clause: int | None
    text: str  # what the reader sees, headed by its identifier ("Khoản 2 Điều 5. …")
    points: list[str] = field(default_factory=list)  # letters, for point-level citations
    attachment: int = 0


def is_placeholder(text: str) -> bool:
    """A form to fill in ("Điều 1. ………………", "Địa chỉ: ……"): mostly dots, ellipses or underscores."""
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c in ".…_" for c in chars) >= 0.3 * len(chars)


def clause_text(c: Clause) -> str:
    return "\n".join([f"{c.number}. {c.text}".rstrip()] + [f"{p.letter}) {p.text}" for p in c.points])


def article_units(a: Article, label: str = "", attachment_title: str = "") -> list[Unit]:
    """The units of one article: its lead-in (clause 0) and one unit per clause, or the whole article when it
    has no clauses. A form in an appendix gives none."""
    if is_placeholder(a.title) or (not a.clauses and not a.points and is_placeholder(a.intro)):
        return []
    owner = label
    if a.attachment:
        name = attachment_title.capitalize() or "Văn bản"
        owner = f"{name} (kèm theo {label})" if label else name
    head = (f"{owner} " if owner else "") + f"Điều {a.number}. {a.title}".rstrip(". ") + "."
    if not a.clauses:
        body = "\n".join([a.intro] * bool(a.intro) + [f"{p.letter}) {p.text}" for p in a.points])
        return [Unit(a.number, None, f"{head}\n{body}".rstrip(), [p.letter for p in a.points], a.attachment)]
    out = [Unit(a.number, 0, f"{head}\n{a.intro}", attachment=a.attachment)] if a.intro else []
    for c in a.clauses:
        out.append(Unit(a.number, c.number, f"{head}\n{clause_text(c)}", [p.letter for p in c.points], a.attachment))
    return out


def units(doc: LegalDocument, label: str = "") -> list[Unit]:
    """Clause-level units; `label` (e.g. "Nghị định 88/2023/NĐ-CP") heads every unit's text so that a reader
    can cite it. Articles of an attached text are headed by its name: "Quy chế … (kèm theo <label>)"."""
    return [u for a in doc.articles
            for u in article_units(a, label, doc.attachments[a.attachment - 1].title if a.attachment else "")]


def article_from_row(row: dict) -> Article:
    """An article back from a data/legal/vbpl_articles.jsonl row (scripts/build_legal_corpus.py)."""
    return Article(row["article"], row["article_title"], row["part"], row["intro"],
                   [Clause(c["n"], c["text"], [Point(p["l"], p["text"]) for p in c["points"]]) for c in row["clauses"]],
                   [Point(p["l"], p["text"]) for p in row["points"]], row["attachment"])


_UNIT_ID = re.compile(r"^(.*?\bĐiều\s+\d+[a-zđ]?)\.")
_CLAUSE_ID = re.compile(r"^\d+\.")


def identifier_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the identifiers in a RAG context of units separated by blank lines
    (scripts/legal_rag.py): the document label and "Điều N" that open each unit's first line (not the article
    title), and the clause number that opens its second line. Task 3.1 is answered with exactly these."""
    spans, pos = [], 0
    for block in text.split("\n\n"):
        first, newline, rest = block.partition("\n")
        if (m := _UNIT_ID.match(first)):
            spans.append((pos, pos + m.end(1)))
        if newline and (c := _CLAUSE_ID.match(rest)):
            start = pos + len(first) + 1
            spans.append((start, start + c.end()))
        pos += len(block) + 2
    return spans


def word_starts(words: list, text: str) -> list[int | None]:
    """Character offset in `text` of each segmented word of it (in order), or None. Words are aligned to the
    text by finding their syllables one after another; a syllable not found within 200 characters is skipped,
    so a segmenter's change to a token costs that word's offset, not the alignment."""
    from .segment import normalize as seg_normalize

    starts, pos = [], 0
    for w in words:
        start = None
        for s in w.syllables:
            i = text.find(seg_normalize(s), pos, pos + 200 + len(s))
            if i >= 0:
                start = i if start is None else start
                pos = i + len(s)
        starts.append(start)
    return starts


def identifier_flags(words: list, text: str) -> list[bool]:
    """For each segmented word of `text` (in order), whether it starts inside an identifier span."""
    spans = identifier_spans(text)
    return [s is not None and any(a <= s < b for a, b in spans) for s in word_starts(words, text)]


def word_blocks(words: list, text: str) -> list[int]:
    """For each segmented word of `text` (in order), the index of its unit (blocks separated by blank lines);
    a word that could not be aligned stays in the unit of the word before it."""
    import bisect

    starts, pos = [0], 0
    for block in text.split("\n\n")[:-1]:
        pos += len(block) + 2
        starts.append(pos)
    out, last = [], 0
    for s in word_starts(words, text):
        last = last if s is None else bisect.bisect_right(starts, s) - 1
        out.append(last)
    return out


_NAMED_TYPES = ("Hiến pháp", "Bộ luật", "Luật", "Pháp lệnh")


def doc_label(doc_type: str, title: str, number: str) -> str:
    """How a document is cited: laws by name and number ("Luật Lưu trữ số 33/2024/QH15"), the rest by type and
    number ("Nghị định 77/2025/NĐ-CP")."""
    if doc_type not in _NAMED_TYPES:
        return f"{doc_type} {number}".strip()
    name = normalize(title)
    name = re.sub(r"\s*\b[Ss][ốỐ]\s*" + NUMBER, "", name, flags=re.I).strip()
    if name.lower().startswith(doc_type.lower() + " "):
        name = name[len(doc_type) + 1:]
    if _is_caps(name):
        name = name.capitalize()
    return f"{doc_type} {name} số {number}".strip()


# ---------------------------------------------------------------------------------------------- citations

@dataclass
class Citation:
    article: str
    clause: int | None = None
    point: str | None = None
    doc_type: str = ""
    number: str = ""  # "88/2023/NĐ-CP"
    name: str = ""  # "Đất đai" for "Luật Đất đai 2024"
    year: int | None = None


_CITE = re.compile(r"^\s*(?:Điểm\s+([a-zđ])\s+)?(?:(?:Khoản|khoản)\s+(\d+)\s+)?(?:Điều|điều)\s+(\d+[a-zđ]?)\s+"
                   r"(?:của\s+)?(.+?)\s*\.?$")


def parse_citation(text: str) -> Citation | None:
    """'Điểm a Khoản 1 Điều 1 Nghị định số 113/2007/NĐ-CP', 'Khoản 8 Điều 81 Luật Đất đai 2024', …"""
    m = _CITE.match(normalize(text))
    if not m:
        return None
    point, clause, article, rest = m.groups()
    cit = Citation(article, int(clause) if clause else None, point)
    for t in DOC_TYPES:
        if rest.startswith(t + " ") or rest == t:
            cit.doc_type, rest = t, rest[len(t):].strip()
            break
    if (n := re.search(NUMBER, rest)):
        cit.number = n.group(0).upper()
        return cit
    if (y := re.search(r"\b(?:năm\s+)?((?:19|20)\d{2})$", rest)):
        cit.year, rest = int(y.group(1)), rest[:y.start()].strip()
    cit.name = re.sub(r"^số\s+", "", rest).strip()
    return cit


def name_key(title: str) -> str:
    """A document name without its type, number and year, lower case: "LUẬT TỔ CHỨC TÒA ÁN NHÂN DÂN SỐ
    34/2024/QH15", "Luật khám bệnh, chữa bệnh 2023 số 15/2023/QH15" and "Đất đai" all reduce to the bare name."""
    t = normalize(title).lower()
    t = re.sub(r"\s*\bsố\s*" + NUMBER.lower(), " ", t)
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)
    for kind in DOC_TYPES:
        if t.startswith(kind.lower() + " "):
            t = t[len(kind) + 1:]
            break
    return re.sub(r"\s+", " ", t).strip(" ,.;")


class Resolver:
    """Document lookup over the vbpl metadata rows (id, title, so_ky_hieu, loai_van_ban, ngay_ban_hanh)."""

    def __init__(self, metadata: list[dict]):
        self.by_number, self.by_name = {}, {}
        for m in metadata:
            if m.get("loai_van_ban") == "Bản dịch văn bản":
                continue
            if m.get("so_ky_hieu"):
                self.by_number.setdefault(str(m["so_ky_hieu"]).upper().strip(), []).append(m)
            year = str(m.get("ngay_ban_hanh") or "")[-4:]
            if year.isdigit() and m.get("title"):
                self.by_name.setdefault((m.get("loai_van_ban"), name_key(m["title"]), int(year)), []).append(m)

    def resolve(self, cit: Citation) -> dict | None:
        """The metadata row of the cited document, or None. A name matches on (type, `name_key`, year of
        issue); consolidated texts (Văn bản hợp nhất) only when nothing else matches."""
        if cit.number:
            rows = self.by_number.get(cit.number, [])
        elif cit.name and cit.year is not None:
            rows = self.by_name.get((cit.doc_type, name_key(cit.name), cit.year), [])
        else:
            return None
        rows = [r for r in rows if r.get("loai_van_ban") != "Văn bản hợp nhất"] or rows
        return rows[0] if rows else None
