# Bản nháp gán nhãn ngày 2026-10-25 (chờ Than Ngoc Thien duyệt)

Các phiếu dưới đây do LLM (Claude Opus 5.5) gán **mù**: 10 subagent độc lập, mỗi agent một phần, chỉ thấy phiếu và
hướng dẫn (`docs/annotation_guides/`), không được mở hay chạy bộ phát hiện P, probe, hay bộ tách từ nào. Mỗi phần
được kiểm bằng `docs/annotation_guides/validate.py`. Mọi dòng mang `annotator`/`reviewer` =
`claude-opus-5-5 (LLM draft, pending review by Than Ngoc Thien)`; sau khi duyệt, đổi thành
`claude-opus-5-5 + Than Ngoc Thien` như phiếu v1. Trong bài phải nêu: nhãn do LLM gợi ý, người duyệt thấy gợi ý
(có thể bị neo theo). Người gán thứ hai cho κ (mục 12) phải làm trên bản sao **trống**, không xem bản này.

## Cần duyệt gì

| Phiếu | Nội dung | Ưu tiên xem |
|---|---|---|
| `data/annot/protect_sheet_v2.jsonl` (blind-v2) | 200 câu (100 premise ViNLI dev + 100 câu VietNews dev, không trùng v1); 1.062 từ thuộc P (T1 927, T2 86, T3 49); 12 câu không có P | 60 dòng có `note` |
| `data/annot/probe_sheet_v2.jsonl` | 200 premise ViNLI dev+train; 99 có phủ định, 101 không | 9 dòng có `note` |
| `data/annot/segmentation_sheet.jsonl` | 300 premise ViNLI test, 7.121 từ (VLSP) | 5 dòng có `note`; quy ước bên dưới |
| `data/annot/vinli_minpairs_review.jsonl` | 300 premise → giữ 79 (`needs_review: false`), 20 có sửa câu (`chưa`→`đã`, `không phải`→`là`); loại 221 (`needs_review: "dropped"`, lý do ở `review_note`) | mục giữ có sửa; một nửa lý do loại |

## Quy ước đã áp dụng (khi hướng dẫn chưa nói rõ)

- **P.** Thực thể tên gồm mọi từ của tên chính thức, kể cả *và/của/tại* bên trong và danh từ đầu tên (*Bộ, huyện,
  Sở, báo, Trường*); không gồm chức danh (*ông, HLV, Trưởng, Phó thủ tướng*) và dấu câu (kể cả "&" trong *GD&ĐT*).
  *Chính phủ/Quốc phòng* trong "Thủ tướng Chính phủ", "Bộ trưởng Quốc phòng Mỹ" được gán T1 theo tiền lệ *Quốc hội*
  của v1 (hai agent làm khác nhau ở ca này). Số gồm số thứ tự (*thứ nhất, lần thứ ba, quý III*) và *một* khi đếm; *đầu
  tiên* không tính. *không thể* (một từ) là phủ định T1. *ít nhất* và *duy nhất* được gán T3.
- **Tách từ.** Theo VLSP; sau khi ghép ba phần, 17 cụm bị tách/gộp không nhất quán được chuẩn hóa (29 chỗ):
  gộp *sau_khi, diễn_ra, chuyến_bay, năm_nay, xuyên_qua, đưa_tin, ví_như, cực_nam, tìm_thấy, còn_lại, Shop_VnExpress,
  Dải_Gaza*; tách *sau đó, trước đó, trong đó, dưới đây, cách đây*.
- **Cặp tối thiểu.** Chặt: loại khi bỏ phủ định làm vế lý do/tương phản vô nghĩa, câu cầu khiến (*đừng*), phủ định
  trong điều kiện/câu hỏi/trích dẫn, thành ngữ (*không ít, không ngừng, chưa kể*), hoặc từ không phải phủ định
  (*hàng không, không người lái, nồi chiên không dầu, trên không*). Hai agent chặt khác nhau (giữ 52/150 và 27/150);
  agent thứ hai ước khoảng 15–20 mục loại vì "câu tự mâu thuẫn với vế lý do" có thể giữ nếu nới tiêu chí.

## Kết quả kiểm tra trên các phiếu nháp

- **Bộ phát hiện P v2 trên blind-v2** (`results/protect_eval.md`): F1 T1/T2/T3 = 0,909 / 0,971 / 0,899 (precision
  0,926 / 0,955 / 1,000; recall 0,893 / 0,988 / 0,816). So với v1 trên blind-v1: recall T1 0,852 → 0,893,
  precision T1 0,957 → 0,926.
  - Nhận nhầm T1 (66): chủ yếu chữ hoa không phải tên (36: viết tắt *USD, HLV, MV, DJ, BOT, HĐXX*, chức danh
    *Trưởng, Chánh*, từ đầu đoạn) và số (20: *mỗi* POS M bị xếp T1 thay vì T3, *một* làm mạo từ không đứng trước
    danh từ đơn vị).
  - Bỏ sót T1 (99): phần không viết hoa của tên tổ chức dài (*và, của, phòng, huyện*…), *không thể*, tên giải/tác
    phẩm.
  - **Không chỉnh luật theo các lỗi này**: chỉnh trên blind-v2 sẽ lặp lại vòng tròn của mục 5; mọi sửa tiếp theo
    cần phiếu blind-v3. Riêng *mỗi* (POS M → T1) là lỗi thứ tự kiểm tra trong `protection_tier` (số được xét trước
    bảng từ), có từ v1.
- **Probe phủ định (G6)** (`results/probe_negation_eval.md`): precision 0,970 (97/100), recall 0,980 → đạt, biên ba
  câu thay vì một.
- **VnCoreNLP so với phiếu tách từ:** khớp 100% dãy âm tiết (0 câu bị loại); ranh giới P 0,967, R 0,992, F1 0,980.
  Khác biệt chủ yếu là quy ước: phiếu gộp *cho_biết, sau_khi, diễn_ra, tạm_dừng, gần_đây* mà VnCoreNLP tách. CBR
  "theo ranh giới người" vì vậy phản ánh một phần quy ước VLSP của phiếu; nên báo cáo kèm con số này.

## Sửa code đi kèm

- `make_annotation_sheets.py`: phiếu tách từ lấy dãy âm tiết của bản tách hiện hành (dấu câu tách riêng), không còn
  tách theo khoảng trắng (khi đó `gold_cbr.py` loại gần như mọi câu).
- `make_minimal_pairs.py`: giả thuyết mâu thuẫn được dựng bằng cách xóa đúng lần xuất hiện của từ phủ định trong
  premise gốc (đếm theo âm tiết, nên *không* trong *Không quân* không bị xóa nhầm), không dựng lại từ bản tách từ.
  Với 300 premise, cách mới cho đúng câu mà người duyệt dựng lại tay ở mọi mục giữ không sửa từ.

## Sau khi duyệt

```
bash scripts/run_pipeline.sh protect_eval        # nếu đổi nhãn protect/probe
python scripts/make_minimal_pairs.py --reviewed-only data/annot/vinli_minpairs_review.jsonl --out data/tasks/vinli_minpairs.jsonl
python scripts/segment_data.py --input data/tasks/vinli_minpairs.jsonl --field premise --output data/segmented/vinli_minpairs.jsonl
```
rồi thêm `data/tasks/vinli_minpairs.jsonl`, `data/segmented/vinli_minpairs.jsonl{,.meta.json}` vào
`data/SHA256SUMS` và `hf_data.py push` (chưa đẩy, chờ duyệt).
