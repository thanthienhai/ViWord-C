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

Ghi lại phiên bản, split và mọi bước lọc (ví dụ bỏ bài thơ trong ViMMRC, bỏ nhãn "other" trong ViNLI) vào file này
khi chuẩn hoá, để kết quả tái tạo được.
