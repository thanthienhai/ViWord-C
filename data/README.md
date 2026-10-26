# Dữ liệu

Thư mục này không được đưa lên git, trừ file README này. Bố cục dự kiến:

```
data/
  tasks/          một file JSONL chuẩn hoá cho mỗi tác vụ và mỗi split
  segmented/      kết quả của scripts/segment_data.py: {"id", "doc"},
                  kèm một sidecar *.jsonl.meta.json cho từng file
  distill/        paragraphs.jsonl (đầu vào cho thầy) và distilled.jsonl (nhãn)
```

## Định dạng chuẩn hoá (`data/tasks/*.jsonl`)

`cluster` là tài liệu nguồn: bài báo, premise hoặc đoạn văn. Các ví dụ dùng chung một nguồn phải có cùng `cluster`,
vì bootstrap lấy mẫu lại theo cụm (DATN §4.3).

| Tác vụ | Trường |
|---|---|
| VietNews | `id`, `cluster` (= id bài), `context` (bài báo), `reference` (tóm tắt) |
| ViNLI | `id`, `cluster` (= premise hoặc id premise), `premise`, `hypothesis`, `label` ∈ {entailment, contradiction, neutral, other} |
| ViMMRC | `id`, `cluster` (= id đoạn văn), `context`, `question`, `choices` (4 lựa chọn), `answer` ∈ {A, B, C, D} |
| Belebele | như ViMMRC, thêm `context_en` (đoạn song song eng_Latn); tạo bằng `viword.data.belebele_from_hub` |
| UIT-ViQuAD 2.0 | `id`, `cluster` (= đoạn văn), `context`, `question`, `answers` (danh sách); độ đo F1 và EM trên âm tiết |
| GSM8K-vi | `id`, `cluster`, `context` (8 ví dụ có lời giải, **phần được nén**), `question`, `answer` (số) |
| OneRuler-vi | `id`, `cluster`, `context` (văn bản hoặc danh sách từ, phần được nén), `template` (chỉ dẫn và câu hỏi, có chỗ `{context}`), `answers`, `subtask`; độ đo: tỉ lệ đáp án xuất hiện trong đầu ra (RULER) |

`data/distill/paragraphs.jsonl`: `id`, `text` (đoạn VietNews/Wikipedia 300–1500 âm tiết, không trùng test).

`data/tasks/belebele_dev.jsonl` và `data/segmented/belebele_dev.jsonl` (+ sidecar `provenance: subset`) không có
trên Hugging Face: `scripts/test_exclusions.py belebele-dev` chép nguyên dòng từ các file `belebele` đã kiểm checksum.
Đó là 200 câu Belebele (111 đoạn) mà reader đã chấm trong lúc phát triển (`config/test_exclusions.json`, DATN §4.3);
phân tích test bỏ chúng, smoke và precheck dùng chúng thay cho dev. `check_segmented.py` chấp nhận sidecar `subset`
khi file nguồn còn nguyên checksum, sidecar của nguồn hợp lệ và mọi dòng có trong nguồn.

## Nguồn

- VietNews: Nguyen et al. (2019), bộ dữ liệu tóm tắt tin tức tiếng Việt.
- ViNLI: Huynh et al. (COLING 2022).
- ViMMRC / ViMMRC 2.0: bộ đọc hiểu trắc nghiệm tiếng Việt (UIT).
- Belebele: Bandarkar et al. (2023), Hugging Face `facebook/belebele`.

Ghi lại phiên bản, split và mọi bước lọc vào file này khi chuẩn hoá, để kết quả tái tạo được.

## Provenance tách từ

Mọi file trong `data/segmented/` dùng cho pipeline chính phải được sinh bằng **VnCoreNLP 1.2** với
**RDRSegmenter + POS + NER**. Sidecar cùng tên có đuôi `.meta.json` ghi `segmenter`, các annotator,
đầu vào/trường nguồn và `output_sha256`; chạy `python scripts/check_segmented.py` để kiểm tra tất cả file.
Pipeline chính sẽ dừng nếu thiếu sidecar, sidecar chỉ bộ tách khác, SHA-256 không khớp, hoặc sidecar viết hồi tố
chưa được `scripts/verify_segmentation.py` xác minh (tách lại một mẫu ngẫu nhiên bằng VnCoreNLP, phải khớp hoàn toàn).

**Sự cố 12/10/2026.** Các cache `belebele`, `vinli_{dev,test}`, `vimmrc_{dev,test}`, `vietnews_{dev,test}` và
`paragraphs_clean` có sidecar ghi VnCoreNLP nhưng khi tách lại thì khác VnCoreNLP ở mọi mẫu thử (30/30) và **khớp từng
byte với underthesea**. Chúng đã được tách lại bằng VnCoreNLP; bản cũ (sidecar đã sửa thành `underthesea`) nằm ở
`data/segmented_legacy_underthesea/`, chỉ dùng cho phân tích độ nhạy §4.4. Nhãn chưng cất được dựng lại từ đúng các đầu
ra thầy cũ (không gọi lại thầy); bản nhãn cũ: `data/distill/distilled_underthesea.jsonl`. Cache `gsm8k`, `oneruler`,
`viquad` vốn đúng là VnCoreNLP (đã xác minh). `vinli_train` được tách thêm để lấy đủ premise cho phiếu G6.

**Vị trí dấu thanh.** VnCoreNLP đổi kiểu bỏ dấu **mới sang cũ** trong `wordForm` ("hòa" → "hoà", "khỏe" → "khoẻ").
`VnCoreNLPSegmenter` giờ chuẩn hoá lại `wordForm` bằng `normalize()`, nên từ trong cache khớp văn bản nguồn đã chuẩn
hoá (trước đó khoảng 0,6% từ khác cách viết, làm văn bản nén trộn hai kiểu dấu). `underthesea`
và `pyvi` chỉ được chạy trực tiếp từ `scripts/segment_data.py` cho phân tích độ nhạy §4.4, với output tách
riêng, không dùng làm dữ liệu chính.

## Các bước lọc đã áp dụng

| Bộ | Bước lọc | Số mẫu bị loại |
|---|---|---|
| ViMMRC | câu hỏi về bài thơ (`is_poem`) | dev 63, test 87 |
| ViMMRC | đáp án đúng trùng nguyên văn với một lựa chọn khác (hai đáp án đúng) | dev 1 (`vimmrc-dev-270`), test 1 (`vimmrc-test-5`) |
| ViNLI | không lọc; giữ cả nhãn `other` (giả thuyết không liên quan tới premise), 4 nhãn cân bằng | 0 |
| UIT-ViQuAD 2.0 | câu hỏi không có đáp án (`is_impossible`); tập test công khai không có đáp án nên **validation làm test**, dev lấy 250 đoạn từ train | test 2.653 câu / 557 đoạn; dev 1.175 / 250 |
| GSM8K-vi | lỗi dịch máy: câu hỏi chứa sẵn lời giải (có dấu "=", trong khi bản tiếng Anh gốc không câu nào có) và vòng lặp lặp câu (một câu hỏi 3.514 âm tiết, một lời giải 4.005 âm tiết); ví dụ dài hơn 400 âm tiết | test 12 (còn 1.307), dev 6 (còn 494), kho ví dụ tương tự |
| OneRuler-vi | lấy mẫu 100 (test) và 30 (dev) mỗi tác vụ con từ cấu hình `vi_4096` | test 700, dev 210 |
| Đoạn chưng cất | trùng n-gram 13 token với tập đánh giá/dev (`leak_filter.py`), kể cả các bộ mới; thêm một bộ lọc bảo thủ theo GUID VietNews | 768 VietNews (664 n-gram + 104 GUID, trong đó 91 chỉ trùng GUID), 70 Wikipedia (lần chạy 12/10; còn 9.162 đoạn: 4.232 VietNews + 4.930 Wikipedia). Bộ lọc bắt buộc có ViNLI/VietNews/ViMMRC/Belebele và các split dev/test tương ứng. GUID của `nam194/vietnews` là số thứ tự trong từng split, **không** phải mã bài: 91 đoạn chỉ trùng GUID là bài khác, bị loại thừa nhưng vô hại |

Nguồn của các bộ thêm ngày 04/10/2026:

| Bộ | Nguồn | Giấy phép | Ghi chú |
|---|---|---|---|
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` (bản sao trên HF), bài gốc arXiv 2203.11400 | thẻ HF không ghi | đoạn văn lấy từ Wikipedia tiếng Việt, nên 39 đoạn chưng cất Wikipedia thêm bị bộ lọc rò rỉ loại |
| GSM8K-vi | `hllj/vi_gsm8k` (dịch bằng GPT-3.5, có kiểm tra), gốc `openai/gsm8k` | MIT | là **dữ liệu dịch máy**, báo cáo riêng thành nhóm "MT" |
| OneRuler-vi | `starbix/oneruler` (bản sao các tập đã sinh), bộ sinh gốc OneRuler arXiv 2503.01996 | thẻ bản sao không ghi; repo gốc MIT | văn bản nền là một cuốn tiểu thuyết công cộng (Tố Tâm), chỉ dẫn do người dịch |
| VlogQA | `uitnlp/vlogqa` (chính thức, gated) | CC BY-SA 4.0 | **chưa tích hợp**: cần bấm đồng ý điều khoản trên Hugging Face rồi mới xem được định dạng |

Các file mới được tách từ sau khi `normalize()` đã thống nhất kiểu bỏ dấu, nên ranh giới từ tốt hơn các file cũ (xem
mục "Còn tồn tại"). Mô hình VnCoreNLP tải từ bản sao HF `lewisnguyn/bonix-model-vncorenlp`, từng file khớp git blob
SHA với repo gốc `vncorenlp/VnCoreNLP`; provenance của từng file được giữ trong sidecar và được kiểm tra trước
khi chạy gate/đánh giá.

## Rà soát dữ liệu (12/10/2026)

Kiểm tra rò rỉ dựa trên n-gram 13 token và văn bản trùng. GUID VietNews là số thứ tự trong từng split, không phải mã
bài: 101 GUID xuất hiện ở cả dev và test nhưng không cặp nào cùng văn bản (kiểm tra 12/10/2026). Bộ lọc GUID vì vậy
chỉ là bộ lọc bảo thủ (loại thừa ~100 đoạn), không phải bằng chứng về bài trùng.

**Đã sửa:**

- **Bộ đánh giá P và probe phủ định: gán nhãn lại.** Bộ nhãn cũ `protect_eval.jsonl` trùng 200/200 tài liệu với
  đầu ra của chính `protection_tiers()` và không có câu nào không chứa P; bộ probe cũ có 50 dòng nhưng chỉ 30 premise,
  mọi nhãn `true`. Cả hai đã chuyển sang `data/annot/invalid_detector_prefilled/` và không được dùng.

### Gán nhãn bộ phát hiện P và probe phủ định

`python scripts/make_annotation_sheets.py` sinh hai phiếu mù (seed 0); không ghi đè phiếu đã có.

- `data/annot/protect_sheet.jsonl`: mỗi dòng là một premise ViNLI dev (`sentence` = null) hoặc một câu của bài
  VietNews dev (`sentence` = chỉ số câu). Trường `words` liệt kê `[chỉ số, từ, POS, NER]`. Người gán điền
  `gold_tiers` = `{"chỉ số": "T1"|"T2"|"T3"}` cho mọi từ thuộc P theo bảng DATN §3.4, **xét nghĩa trong câu** (ví
  dụ *không* nghi vấn cuối câu hỏi, *không khí*, *sợi chỉ* không thuộc P; số và thực thể tên thuộc T1), `{}` nếu
  câu không có từ nào thuộc P, và điền `annotator`. Không chạy bộ phát hiện trước khi gán.
- `data/annot/probe_sheet.jsonl`: 100 premise ViNLI (dev + train) khác nhau. Điền `gold_negation` = `true` nếu
  premise có ít nhất một phủ định thật (đảo cực tính mệnh đề), `false` nếu không, và `annotator`.
- Sau khi điền: `bash scripts/run_pipeline.sh protect_eval` (ghi `results/protect_eval.{md,json}` và
  `results/probe_negation_eval.{md,json}`). Hai script đánh giá từ chối phiếu chưa điền, phiếu không phải `blind-v1`,
  và phiếu có từ không khớp bản tách từ hiện hành.

- **Vị trí dấu thanh (căn chỉnh).** Bản tách từ cũ và đầu ra thầy dùng hai kiểu bỏ dấu khác nhau.
  Đầu ra của thầy giữ cách viết của nguồn, nên bước căn chỉnh không khớp những âm tiết này và gán nhãn *bỏ*:
  với nguồn viết kiểu cũ, tỉ lệ giữ của nhóm âm tiết này chỉ là 0,151 (nguồn viết kiểu mới: 0,432). Cách viết cũ có
  trong 51% đoạn chưng cất (77% đoạn VietNews). `normalize()` giờ đưa mọi văn bản về một kiểu bỏ dấu; tỉ lệ giữ sau
  khi sửa là 0,410. Số token nguồn không khớp văn bản tách từ giảm từ 0,23% xuống 0,006%.
- **Số và đơn vị.** VnCoreNLP tách "6h30" thành "6" + "h30"; bước căn chỉnh giờ tách chữ và số thành token riêng.
- Sau các sửa trên, lọc GUID bảo thủ và tách lại bằng VnCoreNLP: 16.274 nhãn hợp lệ từ 9.162 đoạn sạch; CBR gộp
  của chính thầy là 4,90% (52.370/1.068.838 từ nhiều âm tiết còn giữ), CBR trung bình theo mẫu 6,29%; hai thống kê
  được giữ riêng trong `data/distill/distill_summary.json`. (Theo từ của underthesea trước đây: 6,58% và 8,14%.)
- Độ dài đoạn chưng cất đo theo âm tiết của văn bản gốc (tách theo khoảng trắng) đều nằm trong 300–1.500. Đếm theo
  bản tách từ VnCoreNLP có thể vượt nhẹ (tối đa 1.560) vì dấu câu và số dính chữ được tách thành đơn vị riêng.

**Còn tồn tại, chưa sửa:**

- **264 đoạn Wikipedia dài không có đầu ra của thầy** (trung vị 1.376 âm tiết, so với 1.159 ở các đoạn có đầu ra);
  `wiki-983160` thiếu tỉ lệ 0,25. Kho chưng cất vì vậy thiếu đúng các đoạn dài nhất. Chạy lại
  `teacher_compress.py` (có resume) để bù.
- **Hiển thị số:** văn bản sau tách từ giữ khoảng trắng do VnCoreNLP chèn ("8 h", "1 m72"); số token thay đổi trung
  bình dưới 0,3%, tối đa +10% ở một premise ViNLI.

**Đặc điểm cần biết khi phân tích:**

- **ViNLI:** premise ngắn (trung vị 38 token Qwen; 115/377 premise test ≤ 30 token). Ở tỉ lệ 1/5 ngân sách chỉ còn
  khoảng 6–8 token, nên điểm của mọi phương pháp tiến về mức `no_context`.
- **ViMMRC:** test chỉ có 69 đoạn (dev 33); đáp án lệch (test: A 31%, B 29%, C 27%, D 13%), nên mốc "luôn chọn A" là
  31%, không phải 25%.
- **Kho chưng cất:** sau lọc rò rỉ còn 4.232 đoạn VietNews và 4.930 đoạn Wikipedia; đoạn Wikipedia dài gấp khoảng
  2,3 lần (trung vị 1.166 so với 502 âm tiết), nên chiếm phần lớn số token huấn luyện.

## Hướng B (nhánh `rag-legal`): dữ liệu pháp luật

Nguồn thô nằm ở `data/raw/<tên>/`, mỗi thư mục có một `SOURCE.json` (repo, revision đã ghim, giấy phép, sha256 của
từng file). Tải lại: `python scripts/fetch_legal.py`. Giấy phép được kiểm ngày 26/10/2026:

| Tên | Nguồn | Giấy phép | Dùng cho |
|---|---|---|---|
| `vlegal_bench` | `CMC-OPENAI/VLegal-Bench` @215f273 | CC BY-NC-ND 4.0 | Chỉ đánh giá; không phát hành ngữ cảnh đã nén hay dịch, chỉ công bố điểm |
| `vbpl` | `th1nhng0/vietnamese-legal-documents` @8977887, chỉ `data/` (810 MB) | CC BY 4.0 (bản biên soạn) | Kho văn bản chính |
| `phapdien` | `tmquan/phapdien-moj-gov-vn` @a1b6860, không lấy embedding | CC BY 4.0 | Dự phòng; xem lưu ý dưới |
| `vibidlqa` | `ntphuc149/ViBidLQA` @95dc5d6 (gated: bấm đồng ý trên Hugging Face trước) | apache-2.0 | Hỏi đáp trích đoạn; văn bản gốc lấy từ thuvienphapluat.vn |

`python scripts/build_legal_corpus.py` tách các văn bản trung ương ("Trung ương" và "Toàn quốc"; các luật 2024–2025
mang nhãn "Toàn quốc"), không lấy bản dịch, bằng `viword.legal`, rồi ghi:

- `data/legal/vbpl_articles.jsonl`: mỗi dòng một Điều, gồm thông tin văn bản, `attachment` (0 = văn bản chính,
  k = văn bản kèm theo thứ k), `article`, `article_title`, `part`, `intro`, `clauses[{n, text, points[{l, text}]}]`
  và `points` (các Điểm nằm thẳng dưới Điều).
- `data/legal/vbpl_docs.jsonl`: mỗi dòng một văn bản, gồm số Điều/Khoản/Điểm, số văn bản kèm theo và cảnh báo.

Kết quả lần dựng ngày 26/10/2026:

- 46.119 văn bản. 31.627 văn bản có Điều; tổng cộng 389.553 Điều, 994.511 Khoản và 767.557 Điểm.
- 3.796 văn bản có văn bản kèm theo. 860 Điều là biểu mẫu và không thành đơn vị chọn.
- 1.910 văn bản có cảnh báo:
  - 1.339 lần số Điều không liền mạch;
  - 838 lần đặt lại ngoặc kép bị mất;
  - 462 văn bản còn ngoặc kép chưa đóng ở cuối;
  - 27 văn bản có phần thân bị lặp và đã được cắt.
- Văn bản không có Điều chủ yếu thuộc hai nhóm. Một là thông tư và văn bản cũ chia theo mục "I.", "1/", "a.". Hai là
  trang không có nội dung: 4.196 văn bản có ≤ 5 dòng, kể cả Luật Thuế giá trị gia tăng 2024.

`python scripts/check_legal_citations.py` đo phần trích dẫn VLegal-Bench mà kho phục vụ được, ở mức Khoản:

| Tác vụ | Trích dẫn | Tra được văn bản | Có Điều | Có Khoản |
|---|---|---|---|---|
| 3.1, đáp án đúng | 600 | 589 (98%) | 541 (90%) | 541 (90%) |
| 3.1, phương án nhiễu | 1.800 | 1.030 (57%) | 541 (30%) | 463 (26%) |
| 3.3, Điều trong `content` | 715 | 697 (97%) | 622 (87%) | 622 (87%) |

Phương án nhiễu của 3.1 phần lớn là trích dẫn không có thật (ví dụ Khoản 4 Điều 10 của một nghị định chỉ có 3
Điều). phapdien gần như không chứa các văn bản 3.1 trích dẫn: chỉ 6/299 cặp (văn bản, Điều) có số hiệu, vì bộ pháp
điển cập nhật chậm và đánh số lại Điều. Vì vậy vbpl là kho chính.

### ViBidLQA (26/10/2026)

Chỉ dùng file test (1.000 câu do người viết, đáp án là đoạn trích); file train (5.298 câu do LLM sinh) để dành làm
nhãn huấn luyện. `python scripts/make_vibid_tasks.py` gắn đoạn văn của mỗi câu với Điều của Luật Đấu thầu
22/2023/QH15 (vbpl 166581) hoặc Nghị định 24/2024/NĐ-CP (166583) có văn bản phủ ≥ 90% âm tiết của đoạn, rồi lấy
Khoản vàng là các Khoản của Điều đó chứa đáp án. Ngữ cảnh vì vậy chỉ gồm đơn vị vbpl, Khoản vàng trông như mọi Khoản
khác.

- Giữ 945 câu; bỏ 22 câu có đoạn văn không khớp Điều nào và 33 câu có đáp án không nằm trong Khoản nào (khác phiên
  bản văn bản).
- Chia theo Điều vàng (seed 0): `vibid_dev` 390 câu (38 Điều), `vibid_test` 555 câu (52 Điều). `vibid_test` được
  chặn như `vlegal_test` (`viword.heldout.LEGAL_TEST_FILES`).
- Tác vụ `vibid` (loại `lqa`): prompt "trả lời bằng đoạn trích nguyên văn", tối đa 64 token sinh (32 sẽ cắt 12% đáp
  án), đọc mọi dòng của câu trả lời, chấm F1 trên âm tiết.

`python scripts/legal_rag.py index|recall|contexts --dataset vibid` truy xuất chỉ trong 1.040 Khoản của hai văn bản
(chỉ mục `data/legal/bm25_vibid.*`), nên mọi đơn vị nhiễu cùng luật với Khoản vàng. Trên dev, BM25 chỉ với câu hỏi
xếp một Khoản vàng lên đầu ở 75% câu và đưa vào top-10 ở 99%. Ngữ cảnh khoảng 2.750 token, 10,4 Khoản:

| Chế độ | File | Khoản vàng |
|---|---|---|
| retrieved: thứ tự BM25 | `vibid_rag_dev.jsonl` | có ở 98,7%, thường đứng đầu |
| random: Khoản vàng rút khỏi thứ hạng, đặt ở vị trí ngẫu nhiên | `vibid_rag_random_dev.jsonl` | 100%, vị trí đều |
| oracle: chỉ Khoản vàng | `vibid_rag_oracle_dev.jsonl` | 305 token trung bình |

Chế độ `gold` của VLegal không dùng cho ViBidLQA: Khoản vàng đã được truy xuất thì giữ nguyên thứ hạng, nên vẫn
đứng đầu ở 76% câu và lead thắng chỉ nhờ vị trí.
