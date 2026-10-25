# Guides: negation probe, word segmentation, minimal pairs

## A. Probe sheet (gold_negation)

Each row is a Vietnamese premise (ViNLI). `gold_negation` = `true` if the premise contains at least one real
negation that reverses the polarity of a proposition (không, chưa, chẳng, chả, đừng, chớ, không hề, chưa từng,
...), `false` otherwise. Not a negation: question particle "có ... không?", "không những/không chỉ ... mà còn",
"chẳng hạn", "không" inside a compound ("không khí", "hàng không", "không gian", "không kích" = air strike,
"không lâu" = soon), "không" = zero, product names ("nồi chiên không dầu"). Negative-meaning words without a
negation morpheme ("thiếu", "từ chối", "bác bỏ") are NOT negation for this sheet. A negated subordinate clause
counts ("anh nói rằng sẽ không đến" → true).
Output per row: `{"id": ..., "gold_negation": true|false}` plus optional `"note"` when ambiguous.

## B. Segmentation sheet (gold_words)

Each row gives the syllables of a premise (punctuation already split off) in order. Group them into words
following the VLSP word-segmentation conventions: join the syllables of one word with "_".
- Compounds and Sino-Vietnamese words are one word: "học_sinh", "cộng_đồng", "ghi_nhận", "thủ_tướng",
  "bệnh_viện", "đa_khoa", "khu_vực".
- Reduplicatives are one word: "sạch_sẽ", "lung_linh".
- A proper name is one word: "Hà_Nội", "Nguyễn_Văn_An", "Lê_Minh_Khái", "Thừa_Thiên"; but a title/common noun
  before it is separate ("ông Nguyễn_Văn_An", "tỉnh Thừa_Thiên"), and multi-part organisation names are split
  into their words ("Bộ Công_Thương", "Bệnh_viện Đa_khoa khu_vực Nghĩa_Lộ").
- Function words, classifiers, numbers, punctuation are their own tokens ("của", "các", "đã", "con", "154",
  ",", "."). Tokens like "14h", "27/5" stay as given (never split a given syllable).
- Fixed expressions that are lexicalised as one word are joined ("tuy_nhiên", "vì_vậy", "có_thể" when modal,
  "không_khí" = air); free phrases are not ("ăn cơm", "đi học").
You may never change, add, drop or reorder syllables: `gold_words` with "_" replaced by " " and split on
spaces must equal the given `syllables` exactly. Output per row: `{"id": ..., "gold_words": [...]}` plus optional
`"note"`.

## C. Minimal pairs (review)

Each item is a premise with exactly one detected negation (`operator`) and two hypotheses: the premise itself
(→ entailment, always fine) and `hypothesis_contradiction` = the premise with the negation word removed
(→ should be a contradiction of the premise). Decide for each item:
- `keep`: true only if (1) the operator really negates a proposition in the premise, (2) the contradiction
  hypothesis is fluent, grammatical Vietnamese, and (3) premise vs that hypothesis is clearly a
  contradiction (an NLI annotator would say contradiction, not neutral). Imperatives ("đừng ..."), negations
  inside a quotation/reported belief where removal gives a neutral pair, idioms, or cases where removal
  leaves a broken sentence → fix or drop.
- You may minimally fix `hypothesis_contradiction` so it is the fluent positive counterpart (e.g. "chưa" →
  "đã" when needed: "Anh ấy chưa đến" → "Anh ấy đã đến"; "không hề" → remove both syllables; fix spacing
  around punctuation). Keep every other word of the premise unchanged.
Output per item: `{"base": ..., "keep": true|false, "hypothesis_contradiction": "<final text>", "note": "..."}`
(`note` required when keep is false or when you edited the hypothesis; say what and why in a few words).
