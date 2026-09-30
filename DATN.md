# DATN — Nén ngữ cảnh có nhận thức ngôn ngữ cho LLM tiếng Việt

> **Đề tài:** Nghiên cứu và phát triển phương pháp nén ngữ cảnh có nhận thức ngôn ngữ cho mô hình ngôn ngữ lớn
> tiếng Việt.
> **Ý tưởng đề xuất (tên làm việc):** *"Đừng cắt đôi từ"* — nén prompt theo **đơn vị từ tiếng Việt** và theo
> **chi phí token của LLM đích** (Word- and Tokenizer-Aware Prompt Compression for Vietnamese, gọi tắt **ViWord-C**).
> **Quy mô:** ~10 tuần, nhắm hội nghị nhỏ/khu vực hoặc tạp chí tầm trung.
> **Tài nguyên:** cụm H100 80GB (đã dùng cho LACC). LLM thầy 72B cần 2×H100 (bf16) hoặc 1×H100 với AWQ 4-bit;
> encoder và reader 7B chỉ cần 1 GPU.
> **Quan hệ với pilot LACC** trước đó: xem §6.

---

## 1. Bối cảnh nén prompt cho LLM

| Công trình | Nội dung | Liên quan |
|---|---|---|
| Selective Context (Li et al., EMNLP 2023) | bỏ token/cụm có self-information thấp theo một LM nhỏ | baseline |
| LLMLingua (EMNLP 2023, arXiv 2310.05736) / LongLLMLingua (ACL 2024, arXiv 2310.06839) | bỏ token theo perplexity của LM nhỏ, coarse-to-fine | baseline |
| **LLMLingua-2** (Findings ACL 2024, arXiv 2403.12968) | GPT-4 tạo dữ liệu nén trích xuất → token classification bằng XLM-RoBERTa-large; có thử chuyển sang tiếng Trung (LongBench-Zh) | **baseline chính** và khung huấn luyện mà ta sửa |
| PartPrompt (arXiv 2409.15395) | dùng cây cú pháp để nén; chỉ ra LLMLingua giữ **từ không trọn vẹn** ("8" trong "1988", "elling" trong "selling") do quyết định theo tokenizer | bằng chứng trực tiếp cho vấn đề "cắt đôi từ" — ở tiếng Anh; ta đo mức độ của vấn đề này ở tiếng Việt (§4.0) |
| Prompt Compression Survey (NAACL 2025, arXiv 2410.12388) | tổng quan hard/soft prompt compression | phần related work |
| CAVEWOMAN (arXiv 2606.24083) | nén đầu vào kiểu "ngôn ngữ rút gọn" có thể làm model trả lời dài hơn → tổng chi phí tăng | ta nên báo cáo cả **độ dài đầu ra**, không chỉ token đầu vào |
| **Lost in Compression** (arXiv 2608.26175, 07/2026) | kiểm tra bộ nén trích xuất trên 10 ngôn ngữ (không có tiếng Việt), ngân sách khớp theo tokenizer đích. Tiếng Trung (đơn lập) sụp gần về 0; phần Thảo luận quy cho việc xoá ký tự trong từ nhiều chữ và bỏ trợ từ ngữ pháp (*ba, bei, le, de*). Khoảng cách do **dữ liệu giám sát**; không đề xuất cách sửa | **công trình gần nhất**. Gợi ý (chưa đo) P1/P2 trên một ngôn ngữ đơn lập khác. ViWord-C = cách sửa kỹ thuật, trên tiếng Việt |
| Every Time I Hire a Linguist… (arXiv 2607.25335) | bộ nén chỉ dùng luật ngôn ngữ (tìm bằng tiến hoá), ngang bộ nén học được ở mức nén vừa | baseline "chỉ luật"; tiếng Anh |
| Lexical Prompt Compression (arXiv 2609.13154) | pipeline từ vựng tất định (stopword, POS, giữ thực thể tên) | baseline tất định; tiếng Anh |
| Fundamental Limits of Prompt Compression (arXiv 2407.15504) | khung rate–distortion cho nén token | trích dẫn cho §3.5 |

Danh sách đầy đủ (~50 bài, theo nhóm, kèm mức độ ưu tiên): `docs/LITERATURE_DATN.md`.

**Khoảng trống:** tìm trên Semantic Scholar/arXiv/ACL Anthology (09/2026) không thấy bài nén prompt nào cho
tiếng Việt. *Lost in Compression* chỉ ra vấn đề cho ngôn ngữ đơn lập (tiếng Trung) nhưng không có phương pháp.
Ta định vị bài là: **phương pháp đầu tiên xử lý ranh giới từ và hư từ khi nén prompt cho ngôn ngữ đơn lập,
thực hiện trên tiếng Việt**. *Vẫn phải tra lại trên Google Scholar và kỷ yếu VLSP/RIVF/KSE/SoICT trước khi viết
"đầu tiên".*

**Pilot trước đó: LACC (`../vncompress`, 06–09/2026, không công bố).** LACC chấm điểm token bằng perplexity +
thanh điệu + hình thái (hạ trọng số toàn bộ hư từ) và không chứng minh được các mệnh đề đề ra. Các bài học ảnh
hưởng trực tiếp tới thiết kế ViWord-C (số liệu: `../vncompress/results/report/WAVE4_REPORT.md`):
- **Arm encoder E6 (kiểu LLMLingua-2) sụp về mức ngẫu nhiên** (needle @8x: 0.026, random 0.017) dù PR-AUC trên
  nhãn thầy đạt 0.84. Nhãn lấy từ perplexity của Qwen2.5-0.5B, PhoBERT nhận văn bản chưa tách từ, bước chọn chạy
  trên token BPE của reader. → ViWord-C: thầy nén trích xuất 72B, tách từ trước, đơn vị = từ, kiểm tra thầy và
  checkpoint trên dev trước khi đánh giá đầy đủ (§3.2, §3.3).
- **Hạ trọng số toàn bộ hư từ làm kết quả tệ hơn** (`ppl_only` thắng LACC đầy đủ). → ViWord-C chỉ bảo vệ một tập
  hẹp, có khử nhập nhằng (§3.4).
- **Baseline đơn giản rất mạnh.** Truncation (giữ đầu + cuối) ngang hoặc thắng bộ nén học được; chọn theo câu
  vượt xa chọn theo token ở 8x (0.760 so với ≤ 0.28). → thêm Lead-k và baseline cấp câu (§4.2), cổng G3 (§2.3).
- **Tiếng Việt không bị hại nặng hơn** các ngôn ngữ khác với LLMLingua (τ lệch ±0.02 trên 17 ngôn ngữ). → không
  khẳng định "tiếng Việt đặc biệt"; chỉ khẳng định các hiện tượng đo được trực tiếp (CBR, tỉ lệ giữ hư từ).
- **Lỗi quy trình:** bootstrap gom mọi dòng vào một cụm, TOST thiếu lực, rò rỉ holdout, scorer 124M quá yếu. →
  bootstrap theo cụm, tính MDE trước, kiểm tra rò rỉ tự động, baseline dùng LM đủ mạnh (§4.3, §3.2, §4.2).

**Hoà giải *Lost in Compression* với LACC.** *Lost in Compression* thấy khoảng cách ở các bộ nén học nhãn tiếng
Anh, nhưng không thấy ở phương pháp tất định và bộ nén huấn luyện đa ngữ. LACC không thấy khoảng cách với LLMLingua
(perplexity). Hai kết quả khớp nhau: vấn đề nằm ở **nguồn nhãn giám sát**, không nằm ở bản thân tiếng Việt. Câu
hỏi còn mở, cũng là câu hỏi của H1: khi đã có nhãn bản ngữ thì **đơn vị quyết định** có còn quan trọng không. Vì
vậy §4.0 báo cáo tách theo loại bộ nén.

## 2. Vấn đề và giả thuyết

### 2.1 Bối cảnh ngôn ngữ

Tiếng Việt là ngôn ngữ đơn lập: **dấu cách ngăn âm tiết, không ngăn từ**, và từ không biến hình. Nhiều từ là
từ ghép/láy nhiều âm tiết: *học sinh, doanh nghiệp, Hà Nội, có thể*. Quan hệ ngữ pháp (phủ định, thì/thể, số,
tình thái) được mã hoá **bằng hư từ và trật tự từ**, không bằng hình thái.

Các bộ nén trích xuất hiện có không tính đến điều này:
- **LLMLingua-2** gộp subword thành "từ" theo tiền tố `▁` (tức là theo dấu cách), rồi lấy trung bình xác suất
  giữ (`__merge_token_to_word`, `is_begin_of_new_word` trong `llmlingua`). Với tiếng Việt, đơn vị quyết định
  thực tế là **âm tiết**. Bộ phân loại lại được huấn luyện bằng nhãn **tiếng Anh** (MeetingBank).
- **LLMLingua / Selective Context** bỏ token có perplexity/self-information thấp. Hư từ tần suất cao bị bỏ trước.
- *Lost in Compression* (2608.26175) thấy tiếng Trung, cũng là ngôn ngữ đơn lập, có utility gần về 0 ở keep-rate
  0.33. Phần Thảo luận của họ **đề xuất** (chưa đo) đúng hai cơ chế này: xoá ký tự trong từ nhiều chữ tạo ra từ
  khác hoặc không phải từ, và bộ phân loại học nhãn tiếng Anh bỏ các trợ từ ngữ pháp. Họ không đề xuất cách sửa,
  và không có tiếng Việt.
- Khác tiếng Trung: tiếng Việt có dấu cách giữa các âm tiết nên XLM-R gần như không cắt bên trong âm tiết. Mức
  độ lỗi ở tiếng Việt có thể nhẹ hơn tiếng Trung, nên phải đo trực tiếp (§4.0), không suy ra từ số liệu tiếng
  Trung.

### 2.2 Ba vấn đề (mỗi vấn đề có độ đo chẩn đoán riêng, đo ở §4.0)

- **P1 — cắt đôi từ.** Bộ nén giữ một phần âm tiết của một từ nhiều âm tiết: *học sinh → học*,
  *doanh nghiệp → nghiệp*. Khác với tiếng Anh ("selling" → "elling" là mảnh vô nghĩa), mảnh tiếng Việt thường
  **là một từ hợp lệ mang nghĩa khác**.
  - *Độ đo:*
    - **CBR** (Compound Break Rate), tính trên các từ nhiều âm tiết (theo RDRSegmenter) **còn giữ ít nhất một
      âm tiết**: CBR = số từ bị giữ một phần / (số từ bị giữ một phần + số từ được giữ trọn). Định nghĩa này ít
      phụ thuộc vào tỉ lệ nén hơn so với chia cho mọi từ nhiều âm tiết.
    - **Mốc ngẫu nhiên:** nếu bỏ âm tiết độc lập với tỉ lệ giữ r, từ 2 âm tiết còn sót bị cắt với xác suất
      2(1−r)/(2−r) (≈ 67% ở r = 1/2, 80% ở 1/3, 89% ở 1/5). CBR của mọi bộ nén báo cáo cạnh mốc này.
    - **RR** (Recovery Rate): với mỗi từ bị cắt, cho một mô hình ngôn ngữ (XLM-R MLM, hoặc chính reader qua
      prompt) điền lại âm tiết bị mất từ ngữ cảnh đã nén; RR = % điền đúng. Vết cắt không khôi phục được là
      vết cắt có hại.
    - CBR và RR tách theo loại từ: tên riêng; từ ghép chính phụ (*máy bay, học sinh*); từ láy và từ ghép đẳng
      lập (*sạch sẽ, quần áo*). Cắt nhóm cuối thường ít hại.
  - *Giả thuyết phụ:* vì mảnh còn lại thường là một từ hợp lệ, phần lớn vết cắt khó khôi phục (RR thấp).
- **P2 — mất hư từ mang nghĩa quyết định.** Bỏ *không/chưa/chẳng/đừng* làm đảo cực tính. Bỏ *đã/đang/sẽ* làm
  mất thì. Bỏ *nếu/trừ khi/dù* biến giả định hoặc nhượng bộ thành khẳng định. Bỏ số hoặc thực thể tên làm mất
  dữ kiện. *Giả thuyết:* các bộ nén dựa trên LM và các bộ phân loại học từ nhãn tiếng Anh (nơi mạo từ và trợ từ
  vốn bị bỏ) có xu hướng bỏ nhóm này (đo ở §4.0).
  - *Độ đo:* **tỉ lệ giữ** của từng nhóm hư từ, so với tỉ lệ giữ trung bình ở cùng mức nén.
    **NFR** (NLI Flip Rate) = % cặp ViNLI mà nhãn do reader dự đoán thay đổi khi premise bị nén. NFR đo độ ổn
    định, không đo độ đúng, nên luôn báo cáo kèm accuracy, NFR trên các cặp reader trả lời đúng khi chưa nén, và
    chiều lật nhãn.
  - *Thí nghiệm dò (probe) cận trên:* chỉ xoá các từ phủ định khỏi premise, không đụng gì khác, rồi đo NFR.
    Nếu NFR của probe này thấp, reader không nhạy với việc mất phủ định và H2 không đáng làm.
  - *Đếm trước:* tỉ lệ premise ViNLI chứa từ phủ định (sau khử nhập nhằng). Nhiều cặp mâu thuẫn có thể được tạo
    bằng cách thêm phủ định vào *hypothesis*, nên số premise có phủ định có thể ít (xem cổng G2).
- **P3 — chi phí token không đồng đều.** Chi phí thật tính bằng token của **LLM đích T**. Số token của một từ
  tiếng Việt, c_T(w), dao động mạnh giữa các từ: âm tiết phổ biến chỉ tốn 1 token, âm tiết có dấu hiếm bị tách
  thành nhiều mảnh byte. Mức dao động này cũng khác nhau giữa các tokenizer (Llama-3.1, Qwen2.5, Gemma-2,
  SeaLLM, Vistral).
  - *Độ đo:* fertility (token/âm tiết), **CV_T** = hệ số biến thiên của c_T(w) trên kho văn bản, và tương quan
    giữa c_T(w) và số âm tiết của w.
  - *Lưu ý logic:* nếu mọi từ đắt như nhau (c_T tỉ lệ đều với độ dài), chọn theo điểm/chi phí **không** khác
    chọn theo điểm. Lợi ích của H3 phụ thuộc vào **độ phân tán** của chi phí (CV_T), không phụ thuộc fertility
    trung bình. Nếu c_T(w) gần tỉ lệ với số âm tiết ở mọi tokenizer, "nhận thức tokenizer" không khác "chuẩn hoá
    theo độ dài từ" (xem đối chứng H3 và cổng G5).

### 2.3 Giả thuyết

Mọi so sánh đều ở **cùng ngân sách, đo bằng token của LLM đích** (§4.2), dùng cùng một reader và báo cáo riêng
từng tác vụ × reader × tỉ lệ nén {1/2, 1/3, 1/5}.

- **H1 (đơn vị từ).** *Câu hỏi:* khi đã có dữ liệu giám sát bản ngữ (điều *Lost in Compression* cho là cần),
  đơn vị quyết định có còn quan trọng không? *Giả thuyết:* với *cùng* đầu ra của LLM thầy, *cùng* encoder
  XLM-R-large và *cùng* bước chọn, quyết định ở **cấp từ ngôn ngữ** (sau RDRSegmenter) cho điểm tác vụ cao hơn
  quyết định ở **cấp âm tiết** (= LLMLingua-2-vi). Kết quả theo chiều nào cũng là một phát hiện.
  - *Thiết kế 2×2:* {đơn vị nhãn: âm tiết, từ} × {đơn vị quyết định: âm tiết, từ} (§4.3). Ô (nhãn âm tiết,
    quyết định từ) là LLMLingua-2-vi + gộp xác suất theo từ lúc suy luận, không huấn luyện lại. Nếu ô này ngang
    ViWord-C thì đóng góp thật là "gộp theo từ lúc suy luận", một plug-in dùng được cho mọi bộ nén (thử cả trên
    LLMLingua-2 gốc).
  - *Dự đoán phụ (cơ chế):* lợi ích tăng theo CBR của bản cấp âm tiết. Nó lớn nhất ở tỉ lệ 1/5 và gần 0 ở 1/2.
    Kiểm tra thêm ở mức từng ví dụ: hồi quy chênh lệch điểm theo CBR của chính ví dụ đó.
  - *Kiểm định:* bootstrap ghép cặp theo cụm, một phía, Holm trên {tác vụ × tỉ lệ} với reader chính. Tác vụ
    chính của H1 là ViMMRC và Belebele; VietNews (thiên lệch phần mở đầu, §4.1) và ViNLI chỉ báo cáo mô tả.
- **H2 (bảo vệ ngôn ngữ).** Thêm ưu tiên bảo vệ π làm **giảm NFR** so với cùng hệ thống không có π, **và** so
  với LLMLingua-2-vi + `force_tokens` (bật `force_reserve_digit`) dùng cùng danh sách từ. Đồng thời ROUGE-L trên
  VietNews **không kém quá 0.5 điểm** (kiểm định non-inferiority).
  - *Cấu hình chính (chỉ định trước khi chạy test):* trong các cấu hình π (tầng × soft/hard × δ) mà ROUGE-L trên
    dev không giảm quá 0.5 điểm so với `none`, chọn cấu hình có NFR trên dev thấp nhất; hoà thì chọn cấu hình
    bảo vệ ít từ hơn. Chỉ cấu hình này được kiểm định trên test; các cấu hình khác là khám phá (§4.4).
  - *Lề non-inferiority:* lề 0.5 ROUGE-L được cố định trước khi chạy test. Lề phải lớn hơn độ lệch chuẩn giữa 3
    seed trên dev và lớn hơn MDE (§4.3); nếu không, nới lề và ghi rõ lý do trước khi chạy test.
  - *Dự đoán phụ:* lợi ích tập trung ở các cặp có premise chứa từ phủ định. Báo cáo tách theo tầng (§3.4).
- **H3 (nhận thức tokenizer; chỉ trong bản 8 trang).** Chọn theo s(w)/c_T(w)^α* cho điểm cao hơn chọn với
  α = 0, ở cùng ngân sách token đích.
  - *Đối chứng bắt buộc:* thay c_T(w) bằng số âm tiết n_syl(w) (không phụ thuộc tokenizer). Vì s(w) là trung bình
    (§3.1), α > 0 có thể chỉ đang phạt từ dài. Chỉ khi c_T thắng n_syl mới được gọi là "nhận thức tokenizer".
  - *Dự đoán phụ:* lợi ích tăng theo CV_T của tokenizer. Kiểm tra bằng tương quan hạng trên 4–5 tokenizer, chỉ
    báo cáo mô tả vì n nhỏ.

**Cổng quyết định** (ghi trước khi chạy §4.0; ngưỡng đặt trước, không sửa sau khi thấy số liệu):

| Cổng | Đi tiếp nếu | Nếu không |
|---|---|---|
| G1 — H1 | Tuần 1: CBR của LLMLingua-2 gốc ≥ 5% ở tỉ lệ 1/3, **và** gộp theo từ lúc suy luận cải thiện điểm ở ≥ 1 tác vụ × tỉ lệ. Sau tuần 4: CBR của LLMLingua-2-vi ≥ 5% ở 1/3 | hạ H1 xuống phân tích; trọng tâm sang H2 và phần chẩn đoán |
| G2 — H2 | NFR của probe phủ định ≥ 5 điểm %, **và** có ≥ 300 cặp ViNLI có phủ định trong premise | probe < 5 điểm %: bỏ H2. Thiếu cặp: dựng bộ kiểm tra phủ định riêng (§4.1) |
| G3 — phạm vi | ở tỉ lệ 1/5, bộ nén cấp token tốt nhất không thua baseline cấp câu quá 5 điểm (accuracy hoặc ROUGE-L) trên đa số tác vụ | thiết kế 2 tầng (chọn câu rồi chọn từ trong các câu đã chọn), hoặc giới hạn bài ở 1/2–1/3 |
| G4 — thầy | trên 200 mẫu dev, đầu ra nén của thầy thắng truncation ở 1/3 trên ≥ 2 tác vụ | đổi thầy hoặc prompt trước khi chưng cất 10k mẫu |
| G5 — H3 | tương quan Spearman giữa c_T(w) và số âm tiết < 0.9 ở ít nhất 2 tokenizer | bỏ H3 |

## 3. Phương pháp ViWord-C

```
văn bản ─► chuẩn hoá Unicode (NFC) ─► tách từ (RDRSegmenter) + NER ─► encoder ─► p_keep(w) cho mỗi TỪ
        ─► điểm s(w) = p_keep(w) + π(w) ─► chọn từ theo chi phí c_T(w) và ngân sách B_T
        ─► ghép lại theo thứ tự gốc (bỏ "_", khôi phục khoảng trắng) ─► kiểm tra độ dài thật, cắt bớt nếu vượt
```

### 3.1 Đơn vị quyết định = từ ngôn ngữ ("anchor")
- Chuẩn hoá NFC. Văn bản tiếng Việt có thể ở dạng dấu tổ hợp (NFD), làm sai cả tách từ lẫn số token.
- Tách từ + NER bằng **VnCoreNLP (RDRSegmenter)**. Độ nhạy với công cụ tách từ đo bằng pyvi/underthesea (§4.4).
- Mỗi từ w được ánh xạ tới dãy subword của encoder. Logit của từ = trung bình logit các subword (thử cả lấy
  subword đầu). Giữ/bỏ **cả từ**, nên CBR = 0 theo cách xây dựng khi đo bằng RDRSegmenter; vì vậy §4.4 báo cáo
  thêm CBR của ViWord-C khi đo bằng một bộ tách từ khác. Dấu câu là đơn vị riêng.
- **Gộp điểm cấp từ:** thử trung bình (mặc định) và p_keep × n_syl(w) (điểm tăng theo độ dài từ); chọn trên dev.
  Lựa chọn này ảnh hưởng trực tiếp tới H3 (§2.3).
- **Chính sách dấu câu** giống hệt nhau giữa ViWord-C và LLMLingua-2-vi; nếu không, H1 không còn cô lập được biến.
- Văn bản dài: chia cửa sổ trượt theo ranh giới câu (XLM-R ≤ 512 subword), trùng lặp 1 câu;
  logit của từ nằm trong vùng trùng lấy trung bình.

### 3.2 Nhãn chưng cất: một đầu ra thầy, hai cách nhìn nhãn
- Nguồn: VietNews + Wikipedia tiếng Việt, 5k–10k đoạn (300–1500 âm tiết). **Loại khỏi tập train mọi bài có
  trong test VietNews.**
- **Kiểm tra rò rỉ tự động:** loại khỏi tập train mọi đoạn trùng n-gram với premise ViNLI (premise lấy từ báo,
  có thể trùng VietNews), test VietNews, ViMMRC và Belebele; ghi số đoạn bị loại. Script dừng với lỗi nếu không
  tìm thấy tập test để so, không được bỏ qua trong im lặng (bài học holdout của LACC).
- LLM thầy (Qwen2.5-72B-Instruct; GPT-4o-mini làm phương án phụ nếu còn được cung cấp) nén trích xuất: *chỉ
  xoá, không đổi thứ tự, không thêm, không viết lại*. Mỗi đoạn được sinh ở 2 mức nén (≈1/2 và ≈1/4) để nhãn đủ
  đa dạng.
- Đầu vào cho thầy là **văn bản gốc chưa tách từ**. Không đưa dạng `học_sinh` để thầy không bị "ép" giữ nguyên
  từ; nhờ vậy ta đo được CBR của chính thầy.
- **Kiểm tra thầy trước khi chưng cất (cổng G4):** chạy thầy trên 200 mẫu dev và đo (a) điểm tác vụ của đầu ra
  thầy so với truncation, (b) CBR của thầy, (c) tỉ lệ đầu ra bị loại vì thầy viết lại thay vì chỉ xoá. Bài học
  E6 của LACC: encoder có thể học rất giỏi một nhãn tồi.
- Căn chỉnh: tìm dãy con chung dài nhất (LCS) giữa âm tiết gốc và âm tiết đầu ra; bỏ đoạn có alignment gap lớn
  (theo tiêu chí lọc của LLMLingua-2).
- Từ cùng một căn chỉnh sinh **hai cách nhìn nhãn**:
  - *cấp âm tiết*, dùng cho LLMLingua-2-vi;
  - *cấp từ*: **nhãn mềm** = tỉ lệ âm tiết của từ được thầy giữ (từ 2 âm tiết nhận 0, 0.5 hoặc 1), huấn luyện
    bằng BCE với target mềm. Nhãn mềm thay cho quy tắc "giữ nếu ≥ 50% âm tiết được giữ": quy tắc đó biến mọi
    trường hợp hoà 1/2 thành "giữ", làm lệch phân phối lớp so với nhãn cấp âm tiết. Báo cáo tỉ lệ từ bị thầy
    giữ một phần.

  Hai hệ thống chỉ khác ở đơn vị nhãn và đơn vị quyết định. Đây là điều kiện để H1 cô lập được biến.

### 3.3 Encoder và huấn luyện
- **XLM-R-large** (560M, cùng backbone LLMLingua-2): dùng cho **mọi kiểm định H1–H3**.
- PhoBERT-base: không thuộc phần chính; chỉ đưa vào phụ lục nếu còn thời gian.
- Loss BCE cấp từ với target mềm, có trọng số lớp (tỉ lệ giữ ≈ 30–50%), AdamW, 3 epoch, chọn checkpoint theo
  loss BCE trên dev (dùng được với nhãn mềm; F1 trên các nhãn cứng báo cáo kèm), không dùng reader. Chạy 3 seed;
  báo cáo trung bình ± độ lệch chuẩn.
- **Kiểm tra nhanh checkpoint:** F1 trên nhãn dev chưa đủ (E6 của LACC: PR-AUC 0.84 nhưng ngang ngẫu nhiên ở tác
  vụ). Sau khi chọn checkpoint, chạy 200 mẫu dev qua reader: checkpoint phải thắng truncation và đối chứng tra từ
  (§4.2) trước khi đánh giá đầy đủ. Đây là kiểm tra đạt/không đạt trên dev, không dùng để chọn checkpoint.

### 3.4 Ưu tiên bảo vệ ngôn ngữ π(w)
Tập bảo vệ **P** gồm các cặp **(từ, nhãn POS)** theo VnCoreNLP, không phải chuỗi ký tự. P chia tầng để phân tích:

| Tầng | Nội dung | Lý do |
|---|---|---|
| T1 — cực tính, điều kiện & dữ kiện | phủ định (*không, chưa, chẳng, chả, đừng, chớ, không hề, chưa từng*), điều kiện/nhượng bộ (*nếu, trừ khi, giá mà, dù*), **số**, **thực thể tên** (NER) | bỏ là đổi/mất sự thật; bỏ *nếu* biến giả định thành khẳng định |
| T2 — thời, thể, tình thái | *đã, đang, sẽ, vừa, mới, sắp, từng*; *phải, nên, cần, có thể, được phép* | bỏ là đổi thời hoặc mức cam kết |
| T3 — lượng & so sánh | *mọi, mỗi, tất cả, chỉ, hơn, nhất, kém* | bỏ là đổi phạm vi |

- **Không** bảo vệ *các, những, cái, con*: mang ít nghĩa, bảo vệ chỉ tốn ngân sách.
- **Khử nhập nhằng** bằng luật ngữ cảnh + POS (VnCoreNLP):
  - *không* đứng cuối câu hỏi (*có … không?*) là trợ từ nghi vấn, không phải phủ định; tương tự *chưa* cuối câu
    hỏi (*đã … chưa?*);
  - *không* trước lượng từ/danh từ số (*không độ*) là số 0;
  - *không những / không chỉ … mà còn* là cấu trúc tăng tiến, không phải phủ định; *chẳng hạn* nghĩa là "ví dụ";
  - từ đa nghĩa, phân biệt bằng POS: *mới* (vừa mới / mới = new), *phải* (phải = must / bên phải / phải không),
    *nên* (nên = should / vì vậy nên), *chỉ* (chỉ = only / sợi chỉ / chỉ đường), *kém* (kém hơn / học kém),
    *được* (bị động / có thể / nhận);
  - *có* chỉ là tình thái khi đi trong *có thể*.
  - Ở cấp từ, *không khí, hàng không, không gian* là một đơn vị nên không bị nhầm với phủ định. Ở cấp âm tiết,
    `force_tokens=["không"]` bảo vệ nhầm cả các chữ *không* này, vừa tốn ngân sách vừa tạo mảnh từ. Dùng làm
    Hình 1.
- **Đánh giá bộ phát hiện P:** gán nhãn tay ~200 câu (lấy từ VietNews và ViNLI), báo cáo precision/recall theo
  tầng. Danh sách + luật + tập gán nhãn công bố kèm bài.
- Biến thể:
  - `none`;
  - `soft`: s(w) = p_keep(w) + δ·1[w ∈ P], δ ∈ {0.1, 0.2, 0.3, 0.5} (chỉ dò trên dev);
  - `hard`: xếp P lên trước, tương đương `force_tokens` nhưng ở cấp từ, có khử nhập nhằng.

  Trên test chỉ báo cáo `none` và **một** cấu hình π, chọn theo quy tắc ở §2.3 (H2). Các tầng và soft/hard để ở
  phân tích (§4.4).
- Đối chứng bắt buộc: LLMLingua-2-vi + `force_tokens` = cùng danh sách, bật `force_reserve_digit` (không khử
  nhập nhằng, cấp âm tiết).

### 3.5 Chọn từ theo chi phí token đích (H3, chỉ trong bản 8 trang)
Cho tokenizer đích T và ngân sách B_T (token của T, chỉ tính **phần ngữ cảnh**, không tính chỉ dẫn/câu hỏi):

  max Σ s(w)·x_w  s.t.  Σ c_T(w)·x_w ≤ B_T,  x_w ∈ {0,1}

1. c_T(w) = số token T của `" " + w` (w đã khôi phục khoảng trắng). Dấu câu tính không có khoảng trắng đứng trước.
2. Xếp theo s(w) / c_T(w)^α và chọn tham lam cho tới khi hết ngân sách. Với `hard`: xếp P trước (trong P cũng
   theo tỉ số này), rồi đến phần còn lại.
3. **Bước lấp:** sau khi dừng, duyệt các từ chưa chọn theo s(w) giảm dần và thêm từ nào còn vừa ngân sách.
4. **Kiểm tra độ dài thật:** ghép lại và đếm bằng T. BPE phụ thuộc ngữ cảnh nên Σ c_T có thể lệch; nếu vượt thì
   bỏ dần các từ có s thấp nhất cho đến khi khớp. Báo cáo độ lệch trung bình.

- α ∈ {0, 0.25, 0.5, 0.75, 1}, chỉ dò trên dev. α = 0 là chọn theo điểm thuần, tương đương LLMLingua-2; α = 1
  là knapsack tham lam cổ điển. Trên test chỉ báo cáo α = 0 và α*.
- **Mốc chính xác:** với n ~ 1000 từ và B_T ~ 1000 token, knapsack 0/1 giải đúng bằng quy hoạch động
  (O(n·B_T) ≈ 10⁶ phép tính). Báo cáo khoảng cách giữa tham lam + bước lấp và lời giải DP.
- **Đối chứng độ dài:** chạy lại với c_T(w) thay bằng n_syl(w) (§2.3, H3).
- **Tương tác với số:** tokenizer của Llama-3/Qwen tách số theo chữ số nên số đắt. Với α > 0, báo cáo tỉ lệ giữ
  số (thuộc T1) để thấy α có làm bỏ số không.
- **Chọn siêu tham số không phụ thuộc reader đánh giá:** α và δ chọn **một lần** trên dev (500 mẫu VietNews +
  500 ViNLI) với *một* reader (Qwen2.5-7B), rồi **cố định** cho mọi reader, tác vụ và tokenizer. Báo cáo riêng
  α* chuyển sang reader khác tốt đến đâu.
- Đổi T chỉ đổi c_T, không cần huấn luyện lại.
- Novelty: *Lost in Compression* **đánh giá** với ngân sách khớp tokenizer đích; ta **đưa chi phí đó vào bước
  chọn**.

### 3.6 Ngoài phạm vi (nêu trong Limitations)
Nén trừu tượng/viết lại; nén theo câu hỏi (query-aware); soft prompt/KV cache; bảo đảm mạch lạc cú pháp của
đầu ra (các từ được chọn độc lập, như LLMLingua-2).

## 4. Thực nghiệm

### 4.0 Thí nghiệm chẩn đoán — tuần cổng (làm TRƯỚC, ~1 tuần, không huấn luyện gì)
Một reader (Qwen2.5-7B), 300–500 mẫu mỗi tác vụ, 3 tỉ lệ 1/2, 1/3, 1/5. Tái dùng khung đánh giá của
`../vncompress` (bootstrap theo cụm đã sửa lỗi, các arm `truncation`/`no_context`/`random`, kiểm tra rò rỉ);
không tái dùng phương pháp của LACC.

Arm:
- mốc: không nén, `no_context`, bỏ từ ngẫu nhiên, Lead-k, truncation, chọn theo câu (TF-IDF và bộ chọn câu theo
  perplexity, §4.2);
- bộ nén có sẵn: LLMLingua-2 (bản công bố), LLMLingua (LM đủ mạnh), Selective Context;
- **LLMLingua-2 + gộp xác suất theo từ lúc suy luận** (không huấn luyện lại): ô rẻ nhất của thiết kế 2×2, cho tín
  hiệu sớm về H1.

Độ đo (§2.2):
- **CBR** có điều kiện, cạnh mốc ngẫu nhiên; **RR**; cả hai tách theo loại từ.
- **Tỉ lệ giữ** từng tầng hư từ T1/T2/T3, so với tỉ lệ giữ trung bình.
- **Fertility**, **CV_T** và tương quan c_T(w)–số âm tiết của 5 tokenizer đích.
- **Probe phủ định:** chỉ xoá từ phủ định khỏi premise ViNLI, đo NFR với 1 reader; **đếm số premise có phủ định**.
- Kết quả tách theo loại bộ nén (học nhãn tiếng Anh / perplexity / tất định), theo phần hoà giải ở §1.

Thử thầy trên 200 mẫu dev (§3.2, cổng G4).

Sau khi huấn luyện xong (tuần 2–4), lặp lại CBR cho **LLMLingua-2-vi** (đã huấn luyện trên dữ liệu tiếng Việt,
xem rủi ro 6).

→ Áp dụng các cổng G1–G5 (§2.3). Kết quả là Bảng 1 (và Hình 1) cùng một trang ghi quyết định đi tiếp/đổi hướng.
Kể cả khi mọi cổng đều đóng, bảng chẩn đoán này vẫn đủ cho một bài workshop.

### 4.1 Tác vụ đánh giá
| Tác vụ | Dữ liệu | Cái gì bị nén | Độ đo |
|---|---|---|---|
| Tóm tắt | **VietNews** (test, lấy mẫu 1000) | bài báo | ROUGE-1/2/L, BERTScore (PhoBERT) |
| Suy luận ngôn ngữ (nhạy phủ định) | **ViNLI** (Huynh et al., COLING 2022) | premise (giữ nguyên hypothesis) | accuracy, **NFR** (tỉ lệ lật nhãn so với premise gốc) |
| Đọc hiểu trắc nghiệm | **ViMMRC / ViMMRC 2.0** | đoạn văn (giữ nguyên câu hỏi + đáp án) | accuracy |
| Đọc hiểu (song song đa ngữ) | **Belebele** (vie_Latn, + eng_Latn) | đoạn văn | accuracy; so sánh trực tiếp với *Lost in Compression* và đo khoảng cách Việt–Anh |
| (tuỳ chọn, bản 8 trang) Tóm tắt đa văn bản, đầu vào dài | **VLSP 2022 AbMuSu** | cụm văn bản | ROUGE-1/2/L |

Với mọi tác vụ: chỉ nén **phần ngữ cảnh**; chỉ dẫn, câu hỏi và các lựa chọn giữ nguyên. Dev (chọn α, δ) lấy
từ tập train/validation, không trùng test.

- **VietNews** có thiên lệch phần mở đầu (Lead-3 là baseline tóm tắt tin tức khó thắng), nên Lead-k/truncation
  có thể rất mạnh ở 1/3 và 1/5. VietNews dùng cho kiểm định non-inferiority của H2 và báo cáo mô tả, không làm
  tác vụ chính của H1.
- **ViNLI** có **4 nhãn** (entailment / neutral / contradiction / other). Có thể bỏ nhãn "other" để thành NLI 3 lớp
  chuẩn. Premise chỉ dài một câu, nén còn 1/5 thì chỉ còn vài token. Nếu cổng G2 thiếu cặp có phủ định, dựng thêm
  bộ kiểm tra phủ định 300–500 mẫu (câu có phủ định lấy từ VietNews + câu hỏi có/không, kiểm tay), hoặc nhúng
  premise vào đoạn văn nguồn để ngữ cảnh dài hơn.
- **ViMMRC:** loại các bài thơ, hoặc báo cáo riêng.

LLM đích (reader), zero-shot, decoding greedy: **Qwen2.5-7B-Instruct** (đa ngữ, reader chính) và
**Vistral-7B-Chat** (chuyên tiếng Việt). Bản 8 trang thêm Llama-3.1-8B-Instruct, SeaLLMs-v3-7B và một reader thế
hệ mới (Gemma 3, vocab rất lớn, hợp để kiểm tra H3; nếu dùng Qwen3 thì tắt chế độ thinking).

### 4.2 Phương pháp so sánh
| Nhóm | Phương pháp |
|---|---|
| Mốc | không nén; `no_context` (không có ngữ cảnh) |
| Tầm thường / tất định | truncation (giữ đầu + cuối), **Lead-k** (giữ phần đầu), bỏ từ ngẫu nhiên (cấp từ, 3 seed), bỏ stopword tiếng Việt |
| Cấp câu | trích câu TF-IDF (theo *Lost in Compression*); bộ chọn câu theo perplexity (như `lacc_sentence`, arm tốt nhất của LACC) |
| Chỉ luật ngôn ngữ | pipeline từ vựng (stopword + POS + giữ thực thể tên, theo 2609.13154) chuyển sang tiếng Việt |
| Dịch rồi nén | chỉ trên Belebele: dùng bản eng_Latn (người dịch) làm bản dịch lý tưởng → LLMLingua-2, không chạy dịch máy (*Lost in Compression* thấy cách này thắng nén bản ngữ ở 3/5 ngôn ngữ) |
| Đã công bố | Selective Context và LLMLingua (dùng LM đủ mạnh, ví dụ Qwen2.5-1.5B trở lên, để tránh baseline "người rơm"), **LLMLingua-2** (xlm-roberta-large, bản gốc) |
| Tham chiếu (có câu hỏi) | LongLLMLingua trên ViMMRC/Belebele; dùng câu hỏi nên không đưa vào so sánh chính |
| **Đối chứng quan trọng** | **LLMLingua-2-vi**: *cùng* đầu ra thầy §3.2, *cùng* XLM-R-large, *cùng* bước chọn, nhưng nhãn và quyết định ở cấp **âm tiết** (như LLMLingua-2 gốc). Chỉ khác đơn vị, nên cô lập được H1 |
| Đối chứng H1 (2×2) | LLMLingua-2-vi + gộp theo từ lúc suy luận (nhãn âm tiết, quyết định từ); mô hình học nhãn mềm cấp từ nhưng quyết định ở cấp âm tiết (nhãn từ, quyết định âm tiết); thêm LLMLingua-2 gốc + gộp theo từ |
| **Đối chứng tra từ** | p_keep(w) = tỉ lệ thầy giữ loại từ w trên tập train, không dùng ngữ cảnh; cùng bước chọn. Nếu ViWord-C chỉ ngang đối chứng này thì encoder không đóng góp gì |
| Đối chứng H2 | LLMLingua-2-vi + `force_tokens` = danh sách P, bật `force_reserve_digit` (cơ chế bảo vệ cứng có sẵn trong LLMLingua-2) |
| Cận trên | đầu ra nén của chính LLM thầy, trên tập con 200–300 mẫu mỗi tác vụ (thầy không khớp ngân sách chính xác, nên báo cáo tỉ lệ nén thực tế) |
| Của ta | ViWord-C (XLM-R-large) × {π: none / cấu hình chọn trên dev} × {α = 0 / α* (bản 8 trang)} |

Mức nén: ngân sách **tính theo token của LLM đích**, bằng 1/2, 1/3, 1/5 độ dài ngữ cảnh gốc. Với các phương
pháp chỉ nhận tham số `rate` theo tokenizer của chính chúng (LLMLingua, LLMLingua-2), **tìm nhị phân** giá trị
`rate` để độ dài thật theo tokenizer đích nằm trong ±3% ngân sách. Báo cáo tỉ lệ nén **thực tế** cho mọi phương
pháp, và đánh dấu các cặp so sánh lệch ngân sách > 5%.

### 4.3 Ablation (bảng chính)
| Hàng | Nhãn | Quyết định | π | α | kiểm định |
|---|---|---|---|---|---|
| LLMLingua-2-vi | âm tiết | âm tiết | none | 0 | — |
| LLMLingua-2-vi + gộp theo từ | âm tiết | từ | none | 0 | H1 (2×2) |
| nhãn từ, quyết định âm tiết | từ (mềm) | âm tiết | none | 0 | H1 (2×2) |
| ViWord-C, π = none | từ (mềm) | từ | none | 0 | **H1** |
| LLMLingua-2-vi + `force_tokens` | âm tiết | âm tiết | hard (không khử nhập nhằng) | 0 | đối chứng H2 |
| + bảo vệ π (cấu hình chính) | từ (mềm) | từ | theo quy tắc §2.3 | 0 | **H2** |
| + tokenizer-aware (bản 8 trang) | từ (mềm) | từ | như hàng H2 | α* | **H3** |
| + chi phí = số âm tiết (bản 8 trang) | từ (mềm) | từ | như hàng H2 | α* với n_syl | đối chứng H3 |

Kiểm định:
- Bootstrap ghép cặp **theo cụm** (10k lần), cụm = tài liệu nguồn: premise ViNLI (một premise có nhiều hypothesis),
  đoạn ViMMRC (một đoạn có nhiều câu hỏi), đoạn Belebele (900 câu hỏi trên 488 đoạn), bài VietNews. **Báo cáo
  riêng từng tác vụ × reader × tỉ lệ**, không gộp.
- Test cho phần phân tích dùng đúng định dạng output thật (lỗi gom cụm của LACC lọt qua vì test fixture khác dữ
  liệu thật).
- **Tính MDE trước khi chạy test** cho H1 và cho kiểm định non-inferiority của H2. Nếu MDE lớn hơn hiệu ứng kỳ
  vọng, tăng số mẫu trước khi chạy, không phải sau.
- Holm trong từng họ giả thuyết H1/H2/H3, với reader chính (Qwen2.5-7B); các reader khác là phân tích mô tả.
- NFR dùng McNemar. Non-inferiority ROUGE-L với lề −0.5 (lý do chọn lề ở §2.3).
- Seed: một kết luận chỉ được coi là có ý nghĩa khi đúng ở cả 3 seed (hoặc dùng bootstrap phân tầng: lấy mẫu seed
  rồi lấy mẫu cụm). Bảng chính báo cáo trung bình 3 seed.

### 4.4 Phân tích phụ (lấp chỗ trống cho bài dài)
- CBR, RR và tỉ lệ giữ phủ định của mọi phương pháp, kể cả LLM thầy.
- CBR của ViWord-C khi đo bằng bộ tách từ khác (underthesea/pyvi), vì với RDRSegmenter CBR = 0 theo cách xây dựng.
- H2 tách theo tầng (T1 / T1+T2 / T1+T2+T3) và theo soft/hard.
- Ví dụ định tính "cắt đôi từ làm đổi nghĩa" (*học sinh → học*, *không đồng ý → đồng ý*).
- Ảnh hưởng của lỗi tách từ: thay RDRSegmenter bằng pyvi và xem độ nhạy.
- Độ dài đầu ra của reader (theo CAVEWOMAN): nén có làm model trả lời dài hơn không.
- Độ trễ nén (ms/văn bản) so với LLMLingua (cần LM 7B) và LLMLingua-2.

## 5. Kế hoạch & tài nguyên

| Tuần | Việc | Sản phẩm |
|---|---|---|
| 1 | Tuần cổng §4.0: dựng loader VietNews/ViNLI/ViMMRC/Belebele + khung đánh giá (tái dùng `../vncompress`); chạy chẩn đoán với 1 reader; thử thầy trên 200 mẫu dev | Bảng 1 (CBR, RR, retention, fertility); quyết định theo G1–G5 |
| 2–4 | Kiểm tra rò rỉ; sinh dữ liệu chưng cất (§3.2) + căn chỉnh nhãn cấp từ và cấp âm tiết; huấn luyện ViWord-C (XLM-R) + LLMLingua-2-vi + ô 2×2; chọn δ, α trên dev; kiểm tra nhanh checkpoint trên 200 mẫu dev; gán nhãn 200 câu cho bộ phát hiện P | ~8k mẫu train, 500 dev; checkpoint; precision/recall của P |
| 5–7 | Tính MDE; đánh giá đầy đủ: 2 reader × 4 tác vụ × 3 tỉ lệ (vLLM) | bảng chính + ablation; bản 4 trang có thể nộp từ khoảng tuần 6 |
| 8–10 | Phân tích, viết bài 4–8 trang, công bố code + danh sách hư từ + dữ liệu nén | bản nộp |

**Tính toán:**
- Encoder < 1 GPU-giờ mỗi lần chạy.
- LLM thầy 72B: 10k đoạn × 2 mức nén, cỡ 10M token sinh → vài giờ với vLLM.
- Đánh giá: ~35 biến thể (tính cả 3 seed cho các mô hình huấn luyện và các ô 2×2) × 4 tác vụ × 3 tỉ lệ × ~1000
  mẫu × 2 reader ≈ 0.8–1 triệu lượt sinh (bản 8 trang: nhân thêm theo số reader). LLMLingua và Selective
  Context cần thêm một LM để nén.
- Tổng dự trù **3–5 GPU-ngày** trên H100, tính cả gỡ lỗi.

**Cấu trúc code** (đã cài đặt; cách chạy ở `README.md`). Các cơ chế của khung đánh giá LACC (bootstrap theo
cụm, kiểm tra rò rỉ, các arm mốc) được viết lại gọn trong repo này, không import từ `../vncompress`:
```
viword/
  segment.py      # tách từ (VnCoreNLP/underthesea/pyvi), mặt nạ âm tiết, render, căn chỉnh
  protect.py      # tập P theo (từ, POS, ngữ cảnh) + luật khử nhập nhằng
  select.py       # tham lam s/c_T^α + bước lấp, knapsack DP chính xác, cắt theo độ dài thật
  model.py        # encoder chấm điểm theo âm tiết, cửa sổ trượt, đối chứng tra từ
  compressor.py   # giao diện chung; ViWord-C, LLMLingua-2-vi và các ô 2×2
  baselines.py    # Lead-k, truncation, random, stopword, luật từ vựng, chọn câu, Selective Context,
                  # LLMLingua(-2), LongLLMLingua, probe phủ định, precomputed (cận trên thầy)
  distill.py      # prompt thầy, nhãn âm tiết + nhãn mềm cấp từ, kiểm tra rò rỉ
  train.py        # huấn luyện XLM-R
  diagnose.py     # CBR (+ mốc ngẫu nhiên), RR, retention, fertility, CV_T (§4.0)
  data.py, eval.py, stats.py   # dữ liệu, reader vLLM + metric, bootstrap theo cụm / Holm / McNemar / MDE
scripts/          # segment_data, distill, train, diagnose, evaluate, analyze
tests/            # test CPU
```

## 6. Quan hệ với pilot LACC

| | LACC (pilot, không công bố) | ViWord-C (DATN) |
|---|---|---|
| Bài toán | nén cấp token, chủ yếu theo perplexity (có biến thể dùng câu hỏi) | nén prompt **không phụ thuộc câu hỏi** (task-agnostic) |
| Đơn vị | token BPE của reader | **từ** tiếng Việt |
| Tín hiệu giám sát | perplexity + thanh điệu + hình thái (hạ trọng số mọi hư từ) | chưng cất nén trích xuất từ LLM thầy + tri thức ngôn ngữ |
| Mô hình | GPT-2-vi 124M / Qwen3-4B LoRA; PhoBERT (E6) | word classifier (XLM-R) |
| Đóng góp | không chứng minh được; kết quả âm tính | ranh giới từ + hư từ + chi phí tokenizer đích |
| Dữ liệu | VCC-Bench (tự xây, 5 tác vụ), XQuAD/TyDiQA | VietNews, ViNLI, ViMMRC, Belebele |
| Reader | Qwen2.5-7B → Qwen3-8B | Qwen2.5 / Vistral (+ 3 reader ở bản 8 trang) |

ViWord-C không dùng lại dữ liệu, nhãn hay mô hình của LACC; chỉ tái dùng khung đánh giá (bootstrap theo cụm,
kiểm tra rò rỉ, các arm mốc). Mục này chỉ dùng trong luận văn (chương "Hướng tiếp cận ban đầu và bài học"),
không đưa vào bài báo vì LACC chưa công bố.

## 7. Rủi ro và quy tắc quyết định

1. **LLM chịu được từ bị cắt đôi (H1 hiệu ứng nhỏ).** Khi đó đẩy trọng tâm sang H2 (phủ định/NLI, hiệu ứng
   ngữ nghĩa rõ, dễ minh hoạ) và bài chẩn đoán §4.0. Riêng phần "benchmark + phân tích lỗi nén prompt tiếng
   Việt" đã đủ cho một bài ngắn/workshop.
2. **Lỗi tách từ** làm nhiễu đơn vị: báo cáo độ nhạy theo công cụ tách từ (§4.4). Nếu cần có thể dùng gộp
   theo từ điển âm tiết làm phương án đơn giản.
3. **Bảo vệ cứng ăn hết ngân sách ở tỉ lệ 1/5:** đã có biến thể `soft`, và báo cáo tỉ lệ token dùng cho P.
4. **Reviewer cho rằng "chỉ là LLMLingua-2 + tách từ":** đối chứng LLMLingua-2-vi (cùng dữ liệu, cùng encoder),
   thiết kế 2×2 và H3 (chọn theo chi phí tokenizer đích, bản 8 trang) là các điểm để phản biện. Cần nhấn mạnh
   tính tổng quát cho các ngôn ngữ đơn lập/không tách từ bằng dấu cách (Thái, Lào, Khmer, Trung) ở phần thảo luận.
5. **Novelty:** kiểm tra lại tài liệu (§1) trước khi viết "đầu tiên".
6. **"Chỉ cần dữ liệu bản ngữ là đủ"** (phát hiện của *Lost in Compression*). Nếu LLMLingua-2-vi, huấn luyện
   trên dữ liệu tiếng Việt, đã tốt ngang ViWord-C thì H1 thất bại. Khi đó đóng góp là: (a) benchmark + chẩn
   đoán tiếng Việt, (b) H2 và H3, vốn độc lập với dữ liệu huấn luyện. Vì thế cần đo CBR của **cả**
   LLMLingua-2-vi, không chỉ bản gốc (§4.0, cổng G1).
7. **Thiên lệch phần mở đầu ở tin tức:** Lead-k/truncation có thể thắng mọi bộ nén cấp token trên VietNews ở 1/3
   và 1/5. → VietNews không làm tác vụ chính của H1; Lead-k là baseline chính thức (§4.1, §4.2).
8. **Cấp câu áp đảo cấp token ở tỉ lệ nén cao** (LACC: 0.760 so với ≤ 0.28 ở 8x). → cổng G3; thiết kế 2 tầng
   (chọn câu rồi chọn từ) nếu cần.
9. **Chất lượng thầy:** thầy tồi thì encoder học giỏi một nhãn tồi (E6 của LACC). → cổng G4, cận trên là đầu ra
   thầy, kiểm tra nhanh checkpoint trên dev (§3.2, §3.3).

## 8. Nơi nộp gợi ý (hội nghị nhỏ / khu vực, tạp chí tầm trung)

- **Hội nghị:** PACLIC, RIVF, KSE, SoICT, MAPR, NICS, FAIR (hội nghị quốc gia); workshop VLSP; các workshop về
  ngôn ngữ ít tài nguyên / Đông Nam Á gắn với ACL/EMNLP/COLING; ACL/EMNLP Student Research Workshop (nếu là sinh
  viên).
- **Tạp chí tầm trung:** ACM TALLIP (chuyên xử lý ngôn ngữ châu Á, rất hợp đề tài), Vietnam Journal of Computer
  Science, Tạp chí Tin học và Điều khiển học (JCSC).

**Deadline và xếp hạng mỗi năm khác nhau — cần kiểm tra trên trang từng nơi.** Bản 4 trang (chẩn đoán + H1/H2)
hợp với workshop, FAIR, SoICT, RIVF. Bản 8 trang (thêm H3, 3 reader và tác vụ đầu vào dài) hợp với PACLIC/KSE
hoặc tạp chí.

## 9. Đóng góp dự kiến (để viết abstract)

1. Phân tích định lượng đầu tiên (cần xác minh) về việc các bộ nén prompt hiện có **cắt đôi từ ghép** và **bỏ
   hư từ mang nghĩa** trong tiếng Việt, kèm độ đo CBR (có mốc ngẫu nhiên), tỉ lệ khôi phục RR và tỉ lệ lật nhãn
   NLI. Đây là phần mở rộng tiếng Việt cho kết quả của *Lost in Compression* (vốn không có tiếng Việt), tách theo
   nguồn nhãn giám sát.
2. Câu trả lời có kiểm soát (thiết kế 2×2) cho câu hỏi: khi đã có nhãn bản ngữ, đơn vị quyết định có còn quan
   trọng không. Kết quả theo chiều nào cũng là phát hiện.
3. **ViWord-C**: bộ nén cấp từ, có ưu tiên bảo vệ ngôn ngữ và chọn từ theo chi phí tokenizer đích; đổi LLM
   đích không cần huấn luyện lại.
4. Dữ liệu nén trích xuất tiếng Việt (chưng cất), danh sách hư từ bảo vệ (cặp từ–POS, kèm tập 200 câu gán nhãn),
   và code mở.
