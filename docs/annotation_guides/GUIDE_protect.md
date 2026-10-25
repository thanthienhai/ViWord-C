# Guide: protected set P (sheet blind-v2)

Each input row is one Vietnamese sentence, already word-segmented: `words` = list of `[index, word, POS, NER]`
(VnCoreNLP; a word with several syllables has them space-separated, e.g. "học sinh"). Label every word that
belongs to P **by its meaning in this sentence**, with its tier. Do not try to guess what an automatic
detector would say: label what is true linguistically. POS/NER are hints from a tool and can be wrong
(NER often misses names).

## Tiers (DATN §3.4)

- **T1 — polarity, condition and facts**
  - real negation: không, chưa, chẳng, chả, đừng, chớ, không hề, chưa từng — only when it negates a
    proposition;
  - condition/concession: nếu, trừ khi, giá mà, dù;
  - **numbers**: digits, dates, times, scores, codes with digits ("19/5", "15-40", "4/2019.", "CT3", "1987"),
    number words when they count or order ("ba người", "tầng ba", "thứ nhất" → "nhất" is T1 there, "hai bên",
    "cùng một địa điểm"); "không" read as zero ("không độ") is T1 as a number;
  - **named entities**: every word of a real name, including names the NER missed, and the head nouns that are
    part of the official name: "Bộ" + "Công Thương", "huyện" + "Cẩm Xuyên", "thành phố" + "Huế", "báo" +
    "Tuổi Trẻ", "Sở" + "Kế hoạch" + "và" + "đầu tư" + "Đà Nẵng", "Trường" + "ĐH" + "Bách khoa", "TP." + "HCM".
    Products, brands, events with proper names, laws/decrees with their number count too. Do NOT include
    titles/roles in front of a name ("ông", "bà", "HLV", "Phó thủ tướng", "bác sĩ") and do not include
    punctuation inside a name ("-", "(", ")", ",").
- **T2 — tense/aspect and modality**: đã, đang, sẽ, vừa, mới (= just, adverb), sắp, từng; phải (= must),
  nên (= should), cần, có thể (only when it is ONE word "có thể" = can/may), được phép (only as one word).
- **T3 — quantity and comparison**: mọi, mỗi, tất cả, chỉ (= only), hơn, nhất (superlative), kém (with "hơn").

## Not in P (disambiguation)

- "không"/"chưa" as a question particle at the end of a question ("có ... không?", "đã ... chưa?", also with
  a final particle between: "không ạ?"); "không những / không chỉ ... mà còn" (additive); "chẳng hạn" (= for
  example); "không" inside a compound word segmented as one word ("không khí", "hàng không", "không gian"); a
  product name like "nồi chiên không dầu" (not a negation).
- "một" used as an article ("một con mèo", "một cái", "một bức ảnh", "một chút", "một mình", "một số") — only
  counting "một" is T1 ("một người trong ba người", "cùng một địa điểm" = the same one place, "một kg").
- "mới" = new (adjective, "nhà mới"), "phải" = right side / "phải không?" tag / "phải" as "correct", "nên" =
  therefore (conjunction "vì vậy nên"), "chỉ" = thread / to point ("chỉ đường"), "kém" = poor ("học kém"),
  "hơn" inside a fixed non-comparative expression if any; "được" alone; "các", "những", "cái", "con".
- "nhất" as ordinal ("thứ nhất") is a NUMBER → T1, not T3; superlative "nhất" ("lớn nhất") → T3.

## Examples (human-reviewed labels from the earlier blind-v1 sheet)

See `in/protect_v1_examples.jsonl` (format `index:word ...` and `gold_tiers`). Follow the same conventions.

## Output

One JSON line per input row, same order: `{"id": ..., "sentence": ..., "gold_tiers": {"<index>": "T1"|"T2"|"T3"}}`
(`{}` if the sentence has no word in P). Indices are strings of the given word indices. Optionally add
`"note": "..."` for genuinely ambiguous decisions (keep it short; no note needed for clear cases).
