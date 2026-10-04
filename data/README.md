# Dữ liệu

Thư mục này không được đưa lên git, trừ file README này. Bố cục dự kiến:

```
data/
  tasks/          một file JSONL chuẩn hoá cho mỗi tác vụ và mỗi split
  segmented/      kết quả của scripts/segment_data.py: {"id", "doc"}
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

## Nguồn

- VietNews: Nguyen et al. (2019), bộ dữ liệu tóm tắt tin tức tiếng Việt.
- ViNLI: Huynh et al. (COLING 2022).
- ViMMRC / ViMMRC 2.0: bộ đọc hiểu trắc nghiệm tiếng Việt (UIT).
- Belebele: Bandarkar et al. (2023), Hugging Face `facebook/belebele`.

Ghi lại phiên bản, split và mọi bước lọc vào file này khi chuẩn hoá, để kết quả tái tạo được.

## Các bước lọc đã áp dụng

| Bộ | Bước lọc | Số mẫu bị loại |
|---|---|---|
| ViMMRC | câu hỏi về bài thơ (`is_poem`) | dev 63, test 87 |
| ViMMRC | đáp án đúng trùng nguyên văn với một lựa chọn khác (hai đáp án đúng) | dev 1 (`vimmrc-dev-270`), test 1 (`vimmrc-test-5`) |
| ViNLI | không lọc; giữ cả nhãn `other` (giả thuyết không liên quan tới premise), 4 nhãn cân bằng | 0 |
| UIT-ViQuAD 2.0 | câu hỏi không có đáp án (`is_impossible`); tập test công khai không có đáp án nên **validation làm test**, dev lấy 250 đoạn từ train | test 2.653 câu / 557 đoạn; dev 1.175 / 250 |
| GSM8K-vi | lỗi dịch máy: câu hỏi chứa sẵn lời giải (có dấu "=", trong khi bản tiếng Anh gốc không câu nào có) và vòng lặp lặp câu (một câu hỏi 3.514 âm tiết, một lời giải 4.005 âm tiết); ví dụ dài hơn 400 âm tiết | test 12 (còn 1.307), dev 6 (còn 494), kho ví dụ tương tự |
| OneRuler-vi | lấy mẫu 100 (test) và 30 (dev) mỗi tác vụ con từ cấu hình `vi_4096` | test 700, dev 210 |
| Đoạn chưng cất | trùng n-gram 13 token với tập đánh giá/dev (`leak_filter.py`), kể cả các bộ mới | 677 VietNews, 70 Wikipedia (lần chạy 04/10; còn 9.253 đoạn). So với lần 30/09: thêm 42 đoạn bị loại (82 đầu ra của thầy bỏ khỏi nhãn); 15 đoạn VietNews trước bị loại nay không còn khớp n-gram nào (không truy được nguyên nhân); chúng không có đầu ra của thầy nên không vào dữ liệu huấn luyện |

Nguồn của các bộ thêm ngày 04/10/2026:

| Bộ | Nguồn | Giấy phép | Ghi chú |
|---|---|---|---|
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` (bản sao trên HF), bài gốc arXiv 2203.11400 | thẻ HF không ghi | đoạn văn lấy từ Wikipedia tiếng Việt, nên 39 đoạn chưng cất Wikipedia thêm bị bộ lọc rò rỉ loại |
| GSM8K-vi | `hllj/vi_gsm8k` (dịch bằng GPT-3.5, có kiểm tra), gốc `openai/gsm8k` | MIT | là **dữ liệu dịch máy**, báo cáo riêng thành nhóm "MT" |
| OneRuler-vi | `starbix/oneruler` (bản sao các tập đã sinh), bộ sinh gốc OneRuler arXiv 2503.01996 | thẻ bản sao không ghi; repo gốc MIT | văn bản nền là một cuốn tiểu thuyết công cộng (Tố Tâm), chỉ dẫn do người dịch |
| VlogQA | `uitnlp/vlogqa` (chính thức, gated) | CC BY-SA 4.0 | **chưa tích hợp**: cần bấm đồng ý điều khoản trên Hugging Face rồi mới xem được định dạng |

Các file mới được tách từ sau khi `normalize()` đã thống nhất kiểu bỏ dấu, nên ranh giới từ tốt hơn các file cũ (xem
mục "Còn tồn tại"). Mô hình VnCoreNLP tải từ bản sao HF `lewisnguyn/bonix-model-vncorenlp`, từng file khớp git blob
SHA với repo gốc `vncorenlp/VnCoreNLP`.

## Rà soát dữ liệu (04/10/2026)

Không có id trùng, trường rỗng, hay văn bản trùng giữa dev và test ở mọi bộ (VietNews kiểm tra cả 200 ký tự đầu).

**Đã sửa:**

- **Vị trí dấu thanh.** VnCoreNLP tự đổi kiểu bỏ dấu cũ sang mới ("hoà" → "hòa", "khoẻ" → "khỏe", "uỷ" → "ủy").
  Đầu ra của thầy giữ cách viết của nguồn, nên bước căn chỉnh không khớp những âm tiết này và gán nhãn *bỏ*:
  với nguồn viết kiểu cũ, tỉ lệ giữ của nhóm âm tiết này chỉ là 0,151 (nguồn viết kiểu mới: 0,432). Cách viết cũ có
  trong 51% đoạn chưng cất (77% đoạn VietNews). `normalize()` giờ đưa mọi văn bản về một kiểu bỏ dấu; tỉ lệ giữ sau
  khi sửa là 0,410. Số token nguồn không khớp văn bản tách từ giảm từ 0,23% xuống 0,006%.
- **Số và đơn vị.** VnCoreNLP tách "6h30" thành "6" + "h30"; bước căn chỉnh giờ tách chữ và số thành token riêng.
- Sau hai sửa trên: 14.221 nhãn hợp lệ (trước: 13.988; còn 14.155 sau khi lọc thêm theo các bộ đánh giá mới), CBR của chính thầy 6,0% (trước: 6,7%, bị thổi phồng vì
  từ ghép bị khớp một nửa).

**Còn tồn tại, chưa sửa:**

- **264 đoạn Wikipedia dài không có đầu ra của thầy** (trung vị 1.376 âm tiết, so với 1.159 ở các đoạn có đầu ra);
  `wiki-983160` thiếu tỉ lệ 0,25. Kho chưng cất vì vậy thiếu đúng các đoạn dài nhất. Chạy lại
  `teacher_compress.py` (có resume) để bù.
- **Ranh giới từ khi tách trên văn bản viết dấu kiểu cũ**, ví dụ "Việt Nam Dân chủ Cộng" + "hòa" (316 lần). Tỉ lệ
  nhóm âm tiết nêu trên nằm trong từ nhiều âm tiết: 67,7% (nguồn kiểu cũ) so với 70,5% (kiểu mới). Cần tách từ lại
  sau khi chuẩn hoá đầu vào (cần VnCoreNLP và Java).
- **Hiển thị số:** văn bản sau tách từ giữ khoảng trắng do VnCoreNLP chèn ("8 h", "1 m72"); số token thay đổi trung
  bình dưới 0,3%, tối đa +10% ở một premise ViNLI.

**Đặc điểm cần biết khi phân tích:**

- **ViNLI:** premise ngắn (trung vị 38 token Qwen; 115/377 premise test ≤ 30 token). Ở tỉ lệ 1/5 ngân sách chỉ còn
  khoảng 6–8 token, nên điểm của mọi phương pháp tiến về mức `no_context`.
- **ViMMRC:** test chỉ có 69 đoạn (dev 33); đáp án lệch (test: A 31%, B 29%, C 27%, D 13%), nên mốc "luôn chọn A" là
  31%, không phải 25%.
- **Kho chưng cất:** sau lọc rò rỉ còn 4.323 đoạn VietNews và 4.930 đoạn Wikipedia; đoạn Wikipedia dài gấp khoảng
  2,3 lần (trung vị 1.166 so với 502 âm tiết), nên chiếm phần lớn số token huấn luyện.
