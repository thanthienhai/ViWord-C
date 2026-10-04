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
| Đoạn chưng cất | trùng n-gram 13 token với tập đánh giá/dev (`leak_filter.py`) | 689 VietNews, 31 Wikipedia |

## Rà soát dữ liệu (04/10/2026)

Không có id trùng, trường rỗng, hay văn bản trùng giữa dev và test ở mọi bộ (VietNews kiểm tra cả 200 ký tự đầu).

**Đã sửa:**

- **Vị trí dấu thanh.** VnCoreNLP tự đổi kiểu bỏ dấu cũ sang mới ("hoà" → "hòa", "khoẻ" → "khỏe", "uỷ" → "ủy").
  Đầu ra của thầy giữ cách viết của nguồn, nên bước căn chỉnh không khớp những âm tiết này và gán nhãn *bỏ*:
  với nguồn viết kiểu cũ, tỉ lệ giữ của nhóm âm tiết này chỉ là 0,151 (nguồn viết kiểu mới: 0,432). Cách viết cũ có
  trong 51% đoạn chưng cất (77% đoạn VietNews). `normalize()` giờ đưa mọi văn bản về một kiểu bỏ dấu; tỉ lệ giữ sau
  khi sửa là 0,410. Số token nguồn không khớp văn bản tách từ giảm từ 0,23% xuống 0,006%.
- **Số và đơn vị.** VnCoreNLP tách "6h30" thành "6" + "h30"; bước căn chỉnh giờ tách chữ và số thành token riêng.
- Sau hai sửa trên: 14.221 nhãn hợp lệ (trước: 13.988), CBR của chính thầy 6,0% (trước: 6,7%, bị thổi phồng vì
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
- **Kho chưng cất:** sau lọc rò rỉ còn 4.311 đoạn VietNews và 4.969 đoạn Wikipedia; đoạn Wikipedia dài gấp khoảng
  2,3 lần (trung vị 1.166 so với 502 âm tiết), nên chiếm phần lớn số token huấn luyện.
