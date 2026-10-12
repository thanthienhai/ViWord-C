# DATN — Nén ngữ cảnh có nhận thức ngôn ngữ cho LLM tiếng Việt

> **Đề tài:** Nghiên cứu và phát triển phương pháp nén ngữ cảnh có nhận thức ngôn ngữ cho mô hình ngôn ngữ lớn
> tiếng Việt.
> **Ý tưởng đề xuất (tên làm việc):** *"Đừng cắt đôi từ"* — nén prompt theo **đơn vị từ tiếng Việt** và theo
> **chi phí token của LLM đích** (Word- and Tokenizer-Aware Prompt Compression for Vietnamese, gọi tắt **ViWord-C**).
> **Quy mô:** ~10 tuần, nhắm hội nghị nhỏ/khu vực hoặc tạp chí tầm trung.
> **Tài nguyên:** cụm H100 80GB (đã dùng cho LACC). Encoder và reader 7B chỉ cần 1 GPU; LLM thầy thực dùng là
> **iCOMM `icmodel/icom-model-llm-ic-v3.8-27b` (27B)**, được gọi qua endpoint vLLM tương thích OpenAI
> (`http://172.16.9.11:30048/v1`). Đề cương ban đầu dự kiến Qwen2.5-72B-Instruct, nhưng pipeline thực tế đổi sang
> endpoint iCOMM 27B đang được cung cấp để sinh dữ liệu; đây là thay đổi về mô hình/hạ tầng, không phải thay đổi
> phương pháp. G4 và bước chưng cất phải dùng cùng model này.
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
  trên token BPE của reader. → ViWord-C: thầy nén trích xuất iCOMM 27B, tách từ trước, đơn vị = từ, kiểm tra thầy và
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
- **LLMLingua-2** bản công bố gộp subword thành "từ" theo tiền tố `▁` (tức là theo dấu cách), rồi lấy trung bình
  xác suất giữ (`__merge_token_to_word`, `is_begin_of_new_word` trong `llmlingua`). Đây là baseline nguyên bản;
  với tiếng Việt, đơn vị quyết định thực tế là **âm tiết**. Bộ phân loại lại được huấn luyện bằng nhãn **tiếng Anh**
  (MeetingBank). Nhánh kiểm soát **LLMLingua-2 + gộp theo từ** trong thiết kế này dùng quy ước logit cố định ở
  §2.3 (`sigmoid(mean(logit))`) để nhất quán với mọi ô 2×2, không tuyên bố tái hiện phép gộp xác suất nội bộ của
  package nguyên bản.
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
    - **Mốc ngẫu nhiên:** nếu bỏ âm tiết độc lập với tỉ lệ giữ r, với từ n âm tiết thì xác suất bị cắt có điều kiện
      trên việc còn ít nhất một âm tiết là `[1 − r^n − (1−r)^n] / [1 − (1−r)^n]`. Mốc báo cáo được cộng theo
      phân phối n âm tiết thực tế của corpus (không mặc định n = 2); mốc 2 âm tiết vẫn ghi riêng để đối chiếu.
      CBR của mọi bộ nén báo cáo cạnh mốc theo phân phối này.
    - **RR** (Recovery Rate): với mỗi từ bị cắt, cho XLM-R MLM (mặc định
      `xlm-roberta-large`, hoặc chính reader qua prompt) điền lại âm tiết bị mất từ ngữ cảnh đã nén; RR = %
      điền đúng. Với âm tiết bị mất có từ hai subword trở lên, chèn đúng số vị trí `<mask>` theo tokenizer rồi
      giải mã chuỗi token top-1 trước khi so với âm tiết đích; không coi những trường hợp này là thất bại mặc định.
      Vết cắt không khôi phục được là vết cắt có hại.
    - CBR và RR tách theo loại từ: **tên riêng** (NER khác `O` hoặc POS `Np`); **từ láy** (heuristic: từ hai âm
      tiết cùng phụ âm đầu, *sạch sẽ, lung linh*); **khác** (gồm từ ghép chính phụ *máy bay, học sinh* và từ ghép
      đẳng lập *quần áo*). Repo không có từ điển từ ghép đẳng lập nên không tách được nhóm này khỏi ghép chính phụ;
      heuristic láy cũng bỏ sót láy vần (*lôi thôi*) và chưa đo độ chính xác, nên chỉ báo cáo mô tả.
  - *Giả thuyết phụ:* vì mảnh còn lại thường là một từ hợp lệ, phần lớn vết cắt khó khôi phục (RR thấp).
- **P2 — mất hư từ mang nghĩa quyết định.** Bỏ *không/chưa/chẳng/đừng* làm đảo cực tính. Bỏ *đã/đang/sẽ* làm
  mất thì. Bỏ *nếu/trừ khi/dù* biến giả định hoặc nhượng bộ thành khẳng định. Bỏ số hoặc thực thể tên làm mất
  dữ kiện. *Giả thuyết:* các bộ nén dựa trên LM và các bộ phân loại học từ nhãn tiếng Anh (nơi mạo từ và trợ từ
  vốn bị bỏ) có xu hướng bỏ nhóm này (đo ở §4.0).
  - *Độ đo:* **tỉ lệ giữ** của từng nhóm hư từ, so với tỉ lệ giữ trung bình ở cùng mức nén. Một từ được tính là
    *giữ* khi còn ít nhất một âm tiết của nó trong bản nén (từ bị cắt đôi vẫn tính là giữ; phần đó đã do CBR đo);
    mẫu số "tất cả" là mọi từ không phải dấu câu.
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

**Quy ước điểm cố định:** encoder sinh logit ℓ_s cho mỗi âm tiết. Nhánh quyết định cấp âm tiết dùng
`σ(ℓ_s)`; nhánh quyết định cấp từ dùng `σ(mean_{s∈w} ℓ_s)`, tức **gộp logit trước rồi mới áp dụng sigmoid**.
Không dùng `mean_{s∈w} σ(ℓ_s)`. Quy ước này áp dụng cho mọi ô 2×2 và cho LLMLingua-2 gốc + gộp theo từ.

- **H1 (đơn vị từ).** *Câu hỏi:* khi đã có dữ liệu giám sát bản ngữ (điều *Lost in Compression* cho là cần),
  đơn vị quyết định có còn quan trọng không? *Giả thuyết:* với *cùng* đầu ra của LLM thầy, *cùng* encoder
  XLM-R-large và *cùng* bước chọn, quyết định ở **cấp từ ngôn ngữ** (sau RDRSegmenter) cho điểm tác vụ cao hơn
  quyết định ở **cấp âm tiết** (= LLMLingua-2-vi). Kết quả theo chiều nào cũng là một phát hiện.
  - *Thiết kế 2×2:* {đơn vị nhãn: âm tiết, từ} × {đơn vị quyết định: âm tiết, từ} (§4.3). Ô (nhãn âm tiết,
    quyết định từ) là LLMLingua-2-vi + gộp **logit theo từ rồi sigmoid** lúc suy luận, không huấn luyện lại. Nếu ô này ngang
    ViWord-C thì đóng góp thật là "gộp theo từ lúc suy luận", một plug-in dùng được cho mọi bộ nén (thử cả trên
    LLMLingua-2 gốc).
  - *Dự đoán phụ (cơ chế):* lợi ích tăng theo CBR của bản cấp âm tiết. Nó lớn nhất ở tỉ lệ 1/5 và gần 0 ở 1/2.
    Kiểm tra thêm ở mức từng ví dụ: hồi quy chênh lệch điểm theo CBR của chính ví dụ đó.
  - *Kiểm định:* bootstrap ghép cặp theo cụm, một phía. Họ H1 đăng ký đúng là
    `{ViMMRC, Belebele} × {1/2, 1/3, 1/5}` với reader chính; Holm chỉ hiệu chỉnh các ô trong họ này.
    VietNews (thiên lệch phần mở đầu, §4.1) và ViNLI chỉ báo cáo mô tả.
- **H2 (bảo vệ ngôn ngữ).** Thêm ưu tiên bảo vệ π làm **giảm NFR** so với cùng hệ thống không có π, **và** so
  với đối chứng chính `llmlingua2vi_force` (LLMLingua-2-vi + ngữ nghĩa `force_tokens`/`force_reserve_digit`: khớp
  chuỗi danh sách P ở cấp âm tiết, không khử nhập nhằng, không NER, cùng tầng với cấu hình π được chọn). Đồng thời
  ROUGE-L trên
  VietNews **không kém quá 0.5 điểm** (kiểm định non-inferiority).
  Vì prompt thầy đã yêu cầu giữ phủ định, con số và tên riêng (§3.2), H2 được diễn giải là **lợi ích gia tăng của
  π ở bước suy luận trên một encoder đã học nhãn có prior này**, không phải phép đo độc lập toàn bộ đóng góp của
  việc bảo vệ. Kết quả phải báo cáo tách theo các premise có phủ định; không được khẳng định prompt thầy không
  ảnh hưởng đến đối chứng π.
  - *Cấu hình chính (chỉ định trước khi chạy test):* trong các cấu hình π (tầng × soft/hard × δ) mà ROUGE-L trên
    dev không giảm quá 0.5 điểm so với `none` **ở mọi tỉ lệ nén** (xét tỉ lệ tệ nhất, vì kiểm định
    non-inferiority trên test chạy riêng từng tỉ lệ), chọn cấu hình có NFR trên dev thấp nhất; hoà thì chọn cấu hình
    bảo vệ ít từ hơn (lưới tầng lồng nhau T1 ⊂ T1+T2 ⊂ T1+T2+T3, nên ít tầng hơn đúng là ít từ hơn). Chỉ cấu hình này được kiểm định trên test; các cấu hình khác là khám phá (§4.4).
  - *Lề non-inferiority:* lề 0.5 ROUGE-L được cố định trước khi chạy test. Lề phải lớn hơn độ lệch chuẩn giữa 3
    seed trên dev và lớn hơn MDE **quy về cỡ tập test** (§4.3); nếu không, nới lề và ghi rõ lý do trước khi chạy
    test.
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
| G1 — H1 | Tuần 1: CBR của LLMLingua-2 gốc ≥ 5% ở tỉ lệ 1/3, **và** gộp logit theo từ rồi sigmoid lúc suy luận cải thiện điểm ở ≥ 1 tác vụ × tỉ lệ. Sau tuần 4: CBR của LLMLingua-2-vi ≥ 5% ở 1/3 | hạ H1 xuống phân tích; trọng tâm sang H2 và phần chẩn đoán |
| G2 — H2 | NFR của probe phủ định ≥ 5 điểm % trên **ViNLI dev**, **và** có ≥ 300 cặp **ViNLI test** có phủ định trong premise; vế đếm chỉ dùng văn bản/ID, không dùng nhãn NLI | probe < 5 điểm %: bỏ H2. Thiếu cặp: dựng bộ kiểm tra phủ định riêng (§4.1) |
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
- Tách từ + POS + NER bằng **VnCoreNLP (RDRSegmenter)**. Mỗi file cache `data/segmented/*.jsonl` phải có
  sidecar provenance ghi `vncorenlp` và SHA-256; sidecar viết hồi tố chỉ được chấp nhận sau khi
  `scripts/verify_segmentation.py` tách lại một mẫu ngẫu nhiên bằng VnCoreNLP và khớp hoàn toàn. Mọi stage đọc
  `data/segmented` đều kiểm tra điều này trước. Độ nhạy với công cụ tách từ đo riêng bằng pyvi/underthesea (§4.4),
  không trộn vào các ô chính.
- *Sự cố đã sửa (12/10/2026):* các cache ViNLI, ViMMRC, VietNews, Belebele và kho chưng cất trước đây ghi là
  VnCoreNLP nhưng thực ra do **underthesea** sinh (khớp từng byte với underthesea trên mẫu thử, khác VnCoreNLP 1.2 ở
  ranh giới từ, POS và NER). Các file này đã được tách lại bằng VnCoreNLP và nhãn chưng cất được dựng lại (không gọi
  lại thầy); bản cũ giữ ở `data/segmented_legacy_underthesea/` chỉ cho phân tích độ nhạy §4.4. Mọi số liệu chẩn
  đoán/smoke tính trước ngày này dùng đơn vị từ của underthesea và phải tính lại.
- Mỗi âm tiết được ánh xạ tới dãy subword của encoder và nhận **logit bằng trung bình số học logit các subword**.
  Sau đó, logit của từ là **trung bình số học logit các âm tiết của từ**. Đây là phép gộp hai tầng cố định
  (subword → âm tiết → từ): các âm tiết có trọng số như nhau trong một từ, không phụ thuộc số subword; không
  thử biến thể lấy subword đầu hay nhân p_keep × n_syl trên dev. Giữ/bỏ **cả từ**, nên CBR = 0 theo cách xây dựng
  khi đo bằng RDRSegmenter; vì vậy §4.4 báo cáo thêm CBR của ViWord-C khi đo bằng một bộ tách từ khác. Dấu câu
  là đơn vị riêng.
- **Chính sách dấu câu** giống hệt nhau giữa ViWord-C và LLMLingua-2-vi; nếu không, H1 không còn cô lập được biến.
- **Ghép văn bản trong H1:** cả bốn ô 2×2, gồm LLMLingua-2-vi và ViWord-C ở `unit=syllable|word`, đều chọn
  cùng kiểu mặt nạ âm tiết rồi đi qua `viword.segment.render()`. Vì vậy cùng một mặt nạ luôn cho cùng một chuỗi,
  kể cả khoảng trắng trước/sau dấu câu. LLMLingua-2 bản package, `translate` và `precomputed` không đi qua
  `render()`; chúng là baseline/cận trên ngoài phép so sánh H1 (lần lượt dùng đầu ra text của package, ngữ cảnh
  English của Belebele, và đầu ra thầy đã lưu).
- Văn bản dài: chia cửa sổ trượt theo ranh giới câu (XLM-R ≤ 512 subword), trùng lặp 1 câu;
  logit của từ nằm trong vùng trùng lấy trung bình. Câu dài hơn 200 âm tiết được cắt thành các đoạn 200 âm tiết
  trước khi ghép cửa sổ, để mỗi đoạn vừa một cửa sổ; khi đó ranh giới cửa sổ có thể rơi giữa câu. Mọi âm tiết
  đều được phủ (có test).

### 3.2 Nhãn chưng cất: một đầu ra thầy, hai cách nhìn nhãn
- Nguồn: VietNews + Wikipedia tiếng Việt, 5k–10k đoạn (300–1500 âm tiết). **Loại khỏi tập train mọi bài có
  trong test VietNews.**
- **Kiểm tra rò rỉ tự động:** index bắt buộc phải có ViNLI, VietNews, ViMMRC, Belebele và các split dev/test tương
  ứng; file thiếu/rỗng làm pipeline dừng. Loại khỏi tập train mọi đoạn trùng n-gram 13 token với các context/premise
  đó. Ngoài ra có một bộ lọc bảo thủ theo GUID VietNews (`news-{guid}` với `vietnews-{dev|test}-{guid}`), nhưng
  GUID của `nam194/vietnews` chỉ là số thứ tự trong từng split, không phải mã bài (101 GUID xuất hiện ở cả dev
  và test với hai bài khác nhau). Vì vậy khớp GUID **không** có nghĩa là trùng bài; bộ lọc chỉ loại thêm khoảng 100
  đoạn không liên quan và được giữ lại để bản dữ liệu đã đóng băng tái lập được. Kiểm tra rò rỉ thật sự là n-gram.
  Log ghi coverage và số đoạn bị loại theo nguồn/lý do; không được bỏ qua lỗi holdout trong im lặng (bài học
  holdout của LACC).
- **LLM thầy thực dùng:** iCOMM `icmodel/icom-model-llm-ic-v3.8-27b` (27B), gọi qua endpoint vLLM tương thích
  OpenAI. Bản đề cương ban đầu ghi Qwen2.5-72B-Instruct, nhưng đã đổi vì pipeline thực tế dùng endpoint iCOMM 27B
  đang được cung cấp để sinh các đầu ra thầy; các file `teacher*.jsonl` hiện tại không phải đầu ra của Qwen 72B.
  Model này nén trích xuất: *chỉ xoá, không đổi thứ tự, không thêm, không viết lại*. Mỗi đoạn được sinh ở 2 mức nén
  (≈1/2 và ≈1/4) để nhãn đủ đa dạng. Không dùng GPT-4o-mini làm fallback trong bộ dữ liệu hiện tại; nếu đổi
  `TEACHER_MODEL`, phải chạy lại G4 trước khi chưng cất.
- Đầu vào cho thầy là **văn bản gốc chưa tách từ**. Không đưa dạng `học_sinh` để thầy không bị "ép" giữ nguyên
  từ; nhờ vậy ta đo được CBR của chính thầy.
- Prompt thực tế (xem `viword/distill.py`) cụ thể hoá các ràng buộc trên: mọi từ đầu ra phải có trong nguồn,
  đúng thứ tự/cách viết, không thêm dấu câu, không viết lại, không viết tắt, không dùng từ đồng nghĩa hay dịch.
  Lệnh pipeline truyền trường `text` chưa tách từ cho thầy và sinh hai mức `--rates 0.5 0.25` trước khi căn chỉnh
  với file đã tách từ.
- Prompt cũng dặn thầy giữ **sự kiện, con số, tên riêng và phủ định**. Vì vậy nhãn chưng cất đã chứa một prior
  bảo vệ một phần các thành phần mà π hướng tới, đặc biệt là phủ định; đây là một giới hạn diễn giải chứ không phải
  một phép thử prompt-ablation đã chạy. Khi đổi câu lệnh prompt, phải sinh lại nhãn và chạy lại G4.
- **Kiểm tra thầy trước khi chưng cất (cổng G4):** chạy thầy trên 200 mẫu dev và đo (a) điểm tác vụ của đầu ra
  thầy so với truncation, (b) CBR của thầy, (c) tỉ lệ đầu ra bị loại vì thầy viết lại thay vì chỉ xoá. Bài học
  E6 của LACC: encoder có thể học rất giỏi một nhãn tồi.
- Căn chỉnh: tách văn bản nguồn và đầu ra thành token chữ/số/dấu câu sau `normalize()`, rồi tìm **LCS chính xác**
  bằng Hunt–Szymanski trên chuỗi token. Mỗi token khớp đánh dấu âm tiết gốc sở hữu nó; token lặp chỉ được ghép
  theo thứ tự nguồn. Tỉ lệ `unmatched_ratio` là phần token chữ/số của đầu ra không nằm trong LCS, nên đầu ra đổi
  thứ tự hoặc viết lại bị phát hiện thay vì dùng `difflib.SequenceMatcher`.
- Lọc thực dùng của ViWord-C là ngưỡng cố định: bỏ nếu `unmatched_ratio > 0.10` hoặc `keep_rate > 0.90` (`scripts/distill.py`).
  Đây **không phải** tiêu chí gốc của LLMLingua-2: implementation công bố ([`label_word.py`](https://github.com/microsoft/LLMLingua/blob/main/experiments/llmlingua2/data_collection/label_word.py), [`filter.py`](https://github.com/microsoft/LLMLingua/blob/main/experiments/llmlingua2/data_collection/filter.py)) tính
  `variation_rate` theo việc token đầu ra có xuất hiện ở đâu đó trong nguồn và `alignment_gap` giữa hitting/matching,
  rồi bỏ hai nhóm ở trên phân vị 90% (xấp xỉ 20% mẫu), không dùng ngưỡng 10%/90% cố định. Vì vậy repo phải gọi
  đây là chính sách lọc riêng, không ghi “theo tiêu chí LLMLingua-2”.
- Từ cùng một căn chỉnh sinh **hai cách nhìn nhãn**:
  - *cấp âm tiết*, dùng cho LLMLingua-2-vi;
  - *cấp từ*: **nhãn mềm** = tỉ lệ âm tiết của từ được thầy giữ (từ 2 âm tiết nhận 0, 0.5 hoặc 1). Khi huấn luyện,
    target mềm của từ được chép cho từng âm tiết của từ và đưa vào BCE cấp âm tiết. Nhãn mềm thay cho quy tắc
    "giữ nếu ≥ 50% âm tiết được giữ": quy tắc đó biến mọi
    trường hợp hoà 1/2 thành "giữ", làm lệch phân phối lớp so với nhãn cấp âm tiết. Báo cáo tỉ lệ từ bị thầy
    giữ một phần. Cụ thể, `word_labels[w]` là trung bình nhãn 0/1 của đúng các âm tiết thuộc `w`; nếu lặp nhãn
    mềm này về từng âm tiết thì trung bình có trọng số theo số âm tiết **bằng đúng `keep_rate`**. `teacher_cbr`
    được tính trên các từ nhiều âm tiết còn giữ ít nhất một âm tiết: số từ bị giữ một phần chia cho tổng số từ bị
    giữ một phần và giữ trọn. `scripts/distill.py` báo cáo cả trung bình theo mẫu, CBR gộp toàn corpus và hai số đếm.

  Hai hệ thống chỉ khác ở đơn vị nhãn và đơn vị quyết định. Đây là điều kiện để H1 cô lập được biến.

### 3.3 Encoder và huấn luyện
- **XLM-R-large** (560M, cùng backbone LLMLingua-2): dùng cho **mọi kiểm định H1–H3**.
- PhoBERT-base: không thuộc phần chính; chỉ đưa vào phụ lục nếu còn thời gian.
- **Loss BCE cấp âm tiết:** logit của mỗi âm tiết (trung bình logit subword) được so với target; với
  `label_unit=word`, target mềm của từ được chép cho từng âm tiết rồi tính BCE trên từng âm tiết. Vì vậy từ dài
  đóng góp nhiều hạng loss hơn và logit đã gộp ở cấp từ chỉ được dùng lúc suy luận (§2.3, §3.1), không dùng trong
  objective huấn luyện. Loss có trọng số lớp (tỉ lệ giữ ≈ 30–50%), AdamW, 3 epoch, chọn checkpoint theo loss BCE
  trên dev (dùng được với nhãn mềm; F1 trên các nhãn cứng báo cáo kèm), không dùng reader. Chạy 3 seed; báo cáo
  trung bình ± độ lệch chuẩn.
- **Kiểm tra nhanh checkpoint:** F1 trên nhãn dev chưa đủ (E6 của LACC: PR-AUC 0.84 nhưng ngang ngẫu nhiên ở tác
  vụ). Sau khi chọn checkpoint, chạy 200 mẫu dev qua reader: checkpoint phải thắng truncation và đối chứng tra từ
  (§4.2) trước khi đánh giá đầy đủ. Pipeline thực hiện bằng stage `ckpt_check` ở tỉ lệ 1/3 trên 200 mẫu của từng
  dev set cổng (ViNLI, ViMMRC, VietNews): mọi checkpoint ViWord-C (`viword_s*`) phải thắng `truncation` **và**
  `lexprior` (đối chứng tra từ, fit trên phần train của encoder) trên **đa số** dev set, cùng quy tắc đa số với G4,
  vì VietNews có thiên lệch phần mở đầu mạnh tới mức chính thầy thua truncation. Checkpoint LLMLingua-2-vi là đối
  chứng nên chỉ được báo cáo, không chặn pipeline. Đây là kiểm tra đạt/không đạt trên dev, không dùng để chọn
  checkpoint; `eval` bị chặn nếu báo cáo JSON không ghi `PASS`.

### 3.4 Ưu tiên bảo vệ ngôn ngữ π(w)
Tập bảo vệ **P** được xác định theo **từ + ngữ cảnh**, không chỉ theo chuỗi ký tự. Cụ thể trong code: thực thể
tên lấy từ nhãn NER, số lấy từ POS `M` hoặc chữ số, các từ trong bảng dưới khớp theo mặt chữ (đã tách từ), và năm
từ đa nghĩa *mới, phải, nên, chỉ, kém* chỉ được bảo vệ khi luật POS/ngữ cảnh chấp nhận; phủ định có thêm luật ngữ
cảnh riêng. Các từ còn lại của bảng không có ràng buộc POS. P chia tầng để phân tích:

| Tầng | Nội dung | Lý do |
|---|---|---|
| T1 — cực tính, điều kiện & dữ kiện | phủ định (*không, chưa, chẳng, chả, đừng, chớ, không hề, chưa từng*), điều kiện/nhượng bộ (*nếu, trừ khi, giá mà, dù*), **số**, **thực thể tên** (NER) | bỏ là đổi/mất sự thật; bỏ *nếu* biến giả định thành khẳng định |
| T2 — thời, thể, tình thái | *đã, đang, sẽ, vừa, mới, sắp, từng*; *phải, nên, cần, có thể, được phép* | bỏ là đổi thời hoặc mức cam kết |
| T3 — lượng & so sánh | *mọi, mỗi, tất cả, chỉ, hơn, nhất, kém* | bỏ là đổi phạm vi |

- **Không** bảo vệ *các, những, cái, con*: mang ít nghĩa, bảo vệ chỉ tốn ngân sách.
- **Khử nhập nhằng** bằng luật ngữ cảnh + POS (VnCoreNLP):
  - *không* đứng cuối câu hỏi (*có … không?*) là trợ từ nghi vấn, không phải phủ định; tương tự *chưa* cuối câu
    hỏi (*đã … chưa?*). Luật cho phép trợ từ cuối câu xen giữa (*không ạ?, chưa nhỉ?, … hả?*). **Giới hạn:** câu
    hỏi không có dấu `?` không được nhận ra, và *không ạ.* (lời đáp "không") vẫn là phủ định;
  - *không* đọc như số 0 (POS `M`, hoặc trước *độ, giờ, phẩy, chấm*: *không độ, không giờ, không phẩy năm*)
    không phải phủ định; nó vẫn thuộc T1 với tư cách số. *không* trước một chữ số (*không 3 người*) vẫn tính là
    phủ định;
  - *không những / không chỉ … mà còn* là cấu trúc tăng tiến, không phải phủ định; *chẳng hạn* nghĩa là "ví dụ";
  - từ đa nghĩa, phân biệt bằng POS: *mới* (vừa mới / mới = new), *phải* (phải = must / bên phải / phải không),
    *nên* (nên = should / vì vậy nên), *chỉ* (chỉ = only / sợi chỉ / chỉ đường), *kém* (kém hơn / học kém);
  - *được* chưa có luật khử nhập nhằng riêng: từ đơn *được* không nằm trong P, còn cụm *được phép* chỉ được
    bảo vệ khi bộ tách từ trả về đúng một `Word` như trong lexicon;
  - *có thể* chỉ là tình thái khi bộ tách từ trả về đúng một `Word` nhiều âm tiết; nếu tách thành *có* + *thể*,
    code hiện tại không gộp hai token để áp dụng luật này.
  - Ở cấp từ, *không khí, hàng không, không gian* là một đơn vị nên không bị nhầm với phủ định. Bản legacy
    `force_tokens=["không"]` ở cấp chuỗi âm tiết bảo vệ nhầm cả các chữ *không* này, vừa tốn ngân sách vừa tạo
    mảnh từ; đây là sai khác của baseline package, không phải đường H2 chính. Dùng làm Hình 1.
- **Đánh giá bộ phát hiện P:** gán nhãn **mù** trên phiếu `data/annot/protect_sheet.jsonl` do
  `scripts/make_annotation_sheets.py` sinh: 100 premise ViNLI dev và 100 câu lấy ngẫu nhiên từ bài VietNews dev,
  lấy mẫu đều (có cả câu không chứa từ nào thuộc P, để đo được recall). Phiếu chỉ cho thấy từ, POS, NER, **không**
  cho thấy dự đoán của bộ phát hiện. `scripts/evaluate_protect.py` chỉ nhận phiếu `blind-v1` đã điền đủ và từ trên
  phiếu phải khớp bản tách từ hiện hành; kết quả P/R/F1 theo T1–T3 ghi vào `results/protect_eval.{md,json}`. Chạy
  **trước** khi chọn cấu hình π; thay đổi luật hoặc bộ tách từ phải gán lại. *Sự cố đã sửa:* bộ nhãn trước đây
  (`data/annot/invalid_detector_prefilled/`) trùng 200/200 với đầu ra của chính bộ phát hiện (P = R = 1.000 theo
  cách xây dựng), nên bị loại.
- **Kiểm tra riêng probe phủ định (G6):** phiếu mù `data/annot/probe_sheet.jsonl` gồm 50 premise **khác nhau** mà
  `probe_negation` sửa và 50 premise nó không sửa, trộn lẫn, không cho biết nhóm. ViNLI dev chỉ có 42 premise chứa
  phủ định nên phiếu lấy premise từ cả ViNLI dev và train (không dùng test). `scripts/evaluate_probe.py` yêu cầu ≥ 50
  premise khác nhau bị probe sửa; G6 đạt khi precision ≥ 95% trên nhóm này; recall tính từ nhóm không bị sửa. Hai mẫu
  *không kích* và *không lâu* được kiểm thử như ngoại lệ không phải phủ định.
- Biến thể:
  - `none`;
  - `soft`: s(w) = p_keep(w) + δ·1[w ∈ P], δ ∈ {0.1, 0.2, 0.3, 0.5} (chỉ dò trên dev);
  - `hard`: xếp P lên trước ở cấp từ, có khử nhập nhằng;
  - `force`: ngữ nghĩa `force_tokens` cho đối chứng H2: khớp chuỗi danh sách P (mục nhiều âm tiết khớp theo dãy âm
    tiết) cộng chữ số, ở cấp âm tiết, không khử nhập nhằng, không NER;
  - `force_ctx` (ablation): P đã khử nhập nhằng và có NER theo từng `Word`, mở rộng xuống âm tiết.

  Trên test chỉ báo cáo `none` và **một** cấu hình π, chọn theo quy tắc ở §2.3 (H2). Các tầng và soft/hard để ở
  phân tích (§4.4).
- **Đối chứng H2 chính thức:** `llmlingua2vi_force_s*` dùng checkpoint LLMLingua-2-vi, quyết định cấp âm tiết và
  `protect=force` với **cùng tầng** như cấu hình π được chọn (`tiers=$PI_TIERS`). Đây đúng là thiết kế ban đầu:
  cùng danh sách từ, bật `force_reserve_digit`, không khử nhập nhằng, cấp âm tiết; nhờ đó H2 đo được giá trị của
  việc bảo vệ ở cấp từ **có** khử nhập nhằng so với `force_tokens` thông thường. Một bản trung gian từng dùng P đã
  khử nhập nhằng cho đối chứng này; bản đó nay là ablation `llmlingua2vi_forcectx_s*` (`protect=force_ctx`), báo
  cáo mô tả để tách "đơn vị từ" khỏi "khử nhập nhằng + NER".
- `llmlingua2_force` là baseline package LLMLingua-2 gốc, cũng nhận cùng P động qua `force_tokens` và bật
  `force_reserve_digit`; dùng để chẩn đoán tác động của package/model, **không** thay thế đối chứng H2 chính vì
  checkpoint và cơ chế chọn khác. Cách cũ `naive_syllable_protection` chỉ khớp chuỗi âm tiết đơn và chữ số,
  không có cụm/NER/ngữ cảnh, nên không còn được dùng cho hàng H2. Vì API package nhận danh sách chuỗi chứ không
  nhận mask theo vị trí, một mặt chữ lặp ở hai ngữ cảnh POS khác nhau có thể bị ép ở cả hai vị trí; đây là sai khác
  đã ghi nhận của baseline package, còn `llmlingua2vi_force_s*` giữ khử nhập nhằng theo từng `Word`.

### 3.5 Chọn từ theo chi phí token đích (H3, chỉ trong bản 8 trang)
Cho tokenizer đích T và ngân sách B_T (token của T, chỉ tính **phần ngữ cảnh**, không tính chỉ dẫn/câu hỏi):

  max Σ s(w)·x_w  s.t.  Σ c_T(w)·x_w ≤ B_T,  x_w ∈ {0,1}

1. c_T(w) = số token T của `" " + w` (w đã khôi phục khoảng trắng). Dấu câu tính không có khoảng trắng đứng trước.
2. Xếp theo s(w) / c_T(w)^α và chọn tham lam cho tới khi hết ngân sách. Với `hard`: xếp P trước (trong P cũng
   theo tỉ số này), rồi đến phần còn lại.
3. **Bước lấp:** sau khi dừng, duyệt các từ chưa chọn theo s(w) giảm dần và thêm từ nào còn vừa ngân sách.
4. **Kiểm tra độ dài thật:** ghép lại và đếm bằng T. BPE phụ thuộc ngữ cảnh nên Σ c_T có thể lệch; nếu vượt thì
   bỏ dần các từ có s thấp nhất cho đến khi khớp. Với `hard`, các từ ngoài P bị bỏ trước mọi từ thuộc P. Mỗi dòng
   kết quả ghi Σ c_T, độ dài thật trước bước cắt, độ lệch, số đơn vị bị cắt và số token dành cho P
   (trường `selection`), để báo cáo độ lệch trung bình và phần ngân sách P chiếm (rủi ro 3, §7).

- α ∈ {0, 0.25, 0.5, 0.75, 1}, chỉ dò trên dev. α = 0 là chọn theo điểm thuần, tương đương LLMLingua-2; α = 1
  là knapsack tham lam cổ điển. Trên test chỉ báo cáo α = 0 và α*.
- **Mốc chính xác:** với n ~ 1000 từ và B_T ~ 1000 token, knapsack 0/1 giải đúng bằng quy hoạch động
  (O(n·B_T) ≈ 10⁶ phép tính). Báo cáo khoảng cách giữa tham lam + bước lấp và lời giải DP bằng
  `scripts/greedy_gap.py` (tỉ số mục tiêu Σs tham lam / Σs DP, trung bình và tệ nhất, trước bước cắt).
- **Đối chứng độ dài:** chạy lại với c_T(w) thay bằng n_syl(w) (§2.3, H3).
- **Tương tác với số:** tokenizer của Llama-3/Qwen tách số theo chữ số nên số đắt. Với α > 0, báo cáo tỉ lệ giữ
  số (thuộc T1) để thấy α có làm bỏ số không.
- **Chọn siêu tham số không phụ thuộc reader đánh giá:** α và δ chọn **một lần** trên dev (500 mẫu VietNews +
  500 ViNLI) với *một* reader (Qwen2.5-7B), rồi **cố định** cho mọi reader, tác vụ và tokenizer. Báo cáo riêng
  α* chuyển sang reader khác tốt đến đâu.
- **Chọn α (stage `select_alpha`, `scripts/select_alpha.py`):** chạy trên cấu hình π đã chọn, lưới α ∈ {0.25, 0.5,
  0.75, 1} cho c_T và, riêng, cho đối chứng n_syl. Quy tắc (cố định trước khi chạy test): α* làm cực đại mức tăng
  trung bình so với α = 0 của độ đo chính (điểm ROUGE-L hoặc accuracy), trung bình đều trên 2 tác vụ dev × 3 tỉ lệ
  × 3 seed; α = 0 cũng là ứng viên, hoà thì chọn α nhỏ hơn. Kết quả ghi `config/alpha_selected.{json,env}`; hàng H3
  chỉ được thêm vào `stage_eval` khi file này đã commit.
- **Chọn cấu hình π:** `scripts/select_pi.py` quét `T1`, `T1+T2`, `T1+T2+T3` × `soft|hard` × δ (`0.1, 0.2,
  0.3, 0.5`; `hard` dùng δ=0) trên 500 mẫu ViNLI + 500 VietNews dev cho từng seed đã chạy. ROUGE-L VietNews
  được so với cùng checkpoint `viword_s*` ở π=`none`; NFR ViNLI là tỉ lệ lật dự đoán so với `none` không nén.
  Loại cấu hình mất quá 0.5 điểm ROUGE-L, rồi chọn NFR thấp nhất; hoà thì chọn ít tầng bảo vệ hơn. Quyết định
  được ghi vào `config/pi_selected.json`/`.env` và phải được commit trước `stage_eval`; test chỉ dùng cấu hình đó.
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
- **LLMLingua-2 + gộp logit theo từ rồi sigmoid lúc suy luận** (không huấn luyện lại): ô rẻ nhất của thiết kế 2×2, cho tín
  hiệu sớm về H1.

Độ đo (§2.2):
- **CBR** có điều kiện, cạnh mốc ngẫu nhiên; **RR**; cả hai tách theo loại từ.
- **RR** dùng `xlm-roberta-large` trong pipeline (`RR_MODEL` có thể ghi đè). Mỗi âm tiết bị xoá được biểu diễn
  bằng số `<mask>` bằng số subword mà tokenizer dùng; đầu ra multi-mask được gom và giải mã theo từng âm tiết.
  Vì vậy mô hình RR và quy tắc đo khớp với định nghĩa ở §2.2, kể cả âm tiết tách thành từ hai subword.
- **Tỉ lệ giữ** từng tầng hư từ T1/T2/T3, so với tỉ lệ giữ trung bình.
- **Fertility**, **CV_T** và tương quan c_T(w)–số âm tiết của 5 tokenizer đích.
- **Probe phủ định:** chỉ xoá từ phủ định khỏi premise ViNLI, đo NFR với 1 reader; **đếm số premise có phủ định**.
- **Tách nguồn cho G2:** `scripts/gate_eval.py --split dev` tính NFR trên các cặp của `vinli_dev` mà probe thực sự làm thay đổi
  premise; lệnh này bị chặn nếu truyền `--split test`. Phần đếm độc lập của `scripts/gate_report.py` chạy trên
  `vinli_test` và chỉ chiếu các trường `id`, `cluster`, `doc` trước khi tính, nên không sử dụng trường nhãn NLI. Vì vậy
  test chỉ cung cấp văn bản cho vế đếm đã định trước, không cung cấp nhãn hoặc điểm reader để chọn cấu hình/đánh giá NFR.
- Kết quả tách theo loại bộ nén (học nhãn tiếng Anh / perplexity / tất định), theo phần hoà giải ở §1.
- **Hoà giải F2:** `scripts/compare_llmlingua2.py` so mặt nạ của `scored:...unit=syllable` với package gốc trên
  100 tài liệu nguồn ở ngân sách 1/3. Nếu mặt nạ không trùng ít nhất 95% hoặc CBR lệch từ 1 điểm phần trăm trở lên,
  `gate_report.py` dùng method package `llmlingua2` cho G1; nếu đạt, giữ `llmlingua2_syl`.

Thử thầy trên 200 mẫu dev (§3.2, cổng G4).

Sau khi huấn luyện xong (tuần 2–4), lặp lại CBR cho **LLMLingua-2-vi** (đã huấn luyện trên dữ liệu tiếng Việt,
xem rủi ro 6). Lệnh `bash scripts/run_pipeline.sh diagnose` tự phát hiện các checkpoint
`runs/llmlingua2vi_s*`, chạy `scripts/diagnose.py` trên cùng hai corpus chẩn đoán ở tỉ lệ 1/3, rồi
`gate_report.py` báo cáo từng seed và CBR gộp; vế sau của G1 dùng CBR gộp và ngưỡng ≥ 5%.

→ Áp dụng các cổng G1–G5 (§2.3). Kết quả là Bảng 1 (và Hình 1) cùng một trang ghi quyết định đi tiếp/đổi hướng.
Kể cả khi mọi cổng đều đóng, bảng chẩn đoán này vẫn đủ cho một bài workshop.

### 4.1 Tác vụ đánh giá
| Tác vụ | Dữ liệu | Cái gì bị nén | Độ đo |
|---|---|---|---|
| Tóm tắt | **VietNews** (test, lấy mẫu 1000) | bài báo | ROUGE-1/2/L (tự cài trên âm tiết), BERTScore (PhoBERT-base, lớp 10, đầu vào tách từ bằng VnCoreNLP; stage `bertscore`) |
| Suy luận ngôn ngữ (nhạy phủ định) | **ViNLI** (Huynh et al., COLING 2022) | premise (giữ nguyên hypothesis) | accuracy, **NFR** (tỉ lệ lật nhãn so với premise gốc) |
| Đọc hiểu trắc nghiệm | **ViMMRC / ViMMRC 2.0** | đoạn văn (giữ nguyên câu hỏi + đáp án) | accuracy |
| Đọc hiểu (song song đa ngữ) | **Belebele** (vie_Latn, + eng_Latn) | đoạn văn | accuracy; so sánh trực tiếp với *Lost in Compression* và đo khoảng cách Việt–Anh |
| (tuỳ chọn, bản 8 trang) Tóm tắt đa văn bản, đầu vào dài | **VLSP 2022 AbMuSu** | cụm văn bản | ROUGE-1/2/L |
| *Khám phá* — QA trích xuất | **UIT-ViQuAD 2.0** (validation làm test) | đoạn văn | F1, EM |
| *Khám phá* — nén ví dụ minh hoạ | **GSM8K-vi** (dịch máy) | 8 ví dụ có lời giải | accuracy; báo cáo riêng nhóm "MT" |
| *Khám phá* — ngữ cảnh dài | **OneRuler-vi** (`vi_4096`) | văn bản nền ~4.300 token | recall chuỗi |

Ba tác vụ *khám phá* được thêm ngày 04/10/2026 để kiểm tra độ rộng; chúng chỉ báo cáo mô tả, **không** thuộc họ
Holm của H1/H2 và không tham gia các cổng.

Với mọi tác vụ: chỉ nén **phần ngữ cảnh**; chỉ dẫn, câu hỏi và các lựa chọn giữ nguyên. Dev (chọn α, δ) lấy
từ tập train/validation, không trùng test.

- **VietNews** có thiên lệch phần mở đầu (Lead-3 là baseline tóm tắt tin tức khó thắng), nên Lead-k/truncation
  có thể rất mạnh ở 1/3 và 1/5. VietNews dùng cho kiểm định non-inferiority của H2 và báo cáo mô tả, không làm
  tác vụ chính của H1.
- **ViNLI** có **4 nhãn** (entailment / neutral / contradiction / other). **Quyết định:** giữ đủ 4 nhãn như bộ dữ
  liệu gốc; prompt liệt kê cả bốn nhãn và accuracy tính trên 4 lớp. Không bỏ "other" vì NFR đo việc lật nhãn so với
  premise gốc, và bỏ một lớp sẽ đổi tập cặp được đánh giá giữa các phương pháp. Premise chỉ dài một câu, nén còn 1/5 thì chỉ còn vài token. Nếu cổng G2 thiếu cặp có phủ định, dựng thêm
  bộ kiểm tra phủ định 300–500 mẫu (câu có phủ định lấy từ VietNews + câu hỏi có/không, kiểm tay), hoặc nhúng
  premise vào đoạn văn nguồn để ngữ cảnh dài hơn.
- **ViMMRC:** loại các bài thơ, hoặc báo cáo riêng.

LLM đích (reader), zero-shot, decoding greedy: **Qwen2.5-7B-Instruct** (đa ngữ, reader chính) và
**Vistral-7B-Chat** (chuyên tiếng Việt). Mọi quyết định trên dev (cổng, `ckpt_check`, π, α, MDE) chỉ dùng reader
chính (`PRIMARY_READER`) rồi cố định; reader thứ hai chỉ chạy `eval` với `READER=...`. Vistral là repo gated và
thẻ mô hình dùng system prompt: chạy với `READER_SYSTEM_PROMPT` và kiểm tra template đã render bằng
`scripts/check_reader_template.py` trước khi đánh giá. Mỗi dòng kết quả báo cáo tỉ lệ câu trả lời không đọc được
(phải < 2%, `analyze.py` đánh dấu ⚠). Bản 8 trang thêm Llama-3.1-8B-Instruct, SeaLLMs-v3-7B và một reader thế
hệ mới (Gemma 3, vocab rất lớn, hợp để kiểm tra H3; nếu dùng Qwen3 thì tắt chế độ thinking).

### 4.2 Phương pháp so sánh
| Nhóm | Phương pháp |
|---|---|
| Mốc | không nén; `no_context` (không có ngữ cảnh) |
| Tầm thường / tất định | truncation (giữ đầu + cuối), **Lead-k** (giữ phần đầu), bỏ từ ngẫu nhiên (cấp từ, 3 seed), bỏ stopword tiếng Việt |
| Cấp câu | trích câu TF-IDF (theo *Lost in Compression*); bộ chọn câu theo perplexity `ppl_sent` (điểm câu = surprisal trung bình theo token, giữ câu điểm cao trước) |
| Chỉ luật ngôn ngữ | pipeline từ vựng (stopword + POS + giữ thực thể tên, theo 2609.13154) chuyển sang tiếng Việt |
| Dịch rồi nén | chỉ trên Belebele: dùng bản eng_Latn (người dịch) làm bản dịch lý tưởng → LLMLingua-2, không chạy dịch máy (*Lost in Compression* thấy cách này thắng nén bản ngữ ở 3/5 ngôn ngữ) |
| Đã công bố | Selective Context (cài lại ở cấp từ: self-information của từ = tổng surprisal các token của từ, chọn theo ngân sách; bản gốc lọc theo ngưỡng phân vị trên cụm từ và cần spaCy tiếng Anh) và LLMLingua (dùng LM đủ mạnh, ví dụ Qwen2.5-1.5B trở lên, để tránh baseline "người rơm"), **LLMLingua-2** (xlm-roberta-large, bản gốc) |
| Tham chiếu (có câu hỏi) | LongLLMLingua trên ViMMRC/Belebele; dùng câu hỏi nên không đưa vào so sánh chính |
| **Đối chứng quan trọng** | **LLMLingua-2-vi**: *cùng* đầu ra thầy §3.2, *cùng* XLM-R-large, *cùng* bước chọn, nhưng nhãn và quyết định ở cấp **âm tiết** (như LLMLingua-2 gốc). Chỉ khác đơn vị, nên cô lập được H1 |
| Đối chứng H1 (2×2) | LLMLingua-2-vi + gộp logit theo từ rồi sigmoid lúc suy luận (nhãn âm tiết, quyết định từ); mô hình học nhãn mềm cấp từ nhưng quyết định ở cấp âm tiết (nhãn từ, quyết định âm tiết); thêm LLMLingua-2 gốc + cùng phép gộp logit |
| **Đối chứng tra từ** | p_keep(w) = tỉ lệ thầy giữ loại từ w trên tập train, không dùng ngữ cảnh; cùng bước chọn. Nếu ViWord-C chỉ ngang đối chứng này thì encoder không đóng góp gì |
| Đối chứng H2 | `llmlingua2vi_force_s*`: LLMLingua-2-vi + `protect=force` (ngữ nghĩa `force_tokens`, cùng tầng với π); ablation `llmlingua2vi_forcectx_s*` dùng P đã khử nhập nhằng; `llmlingua2_force` chỉ là baseline package chẩn đoán |
| Cận trên | đầu ra nén của chính LLM thầy, trên tập con 200–300 mẫu mỗi tác vụ (thầy không khớp ngân sách chính xác, nên báo cáo tỉ lệ nén thực tế). Ở cổng G4, `trim=1` cắt đầu ra thầy **từ cuối** theo khoảng trắng cho tới khi vừa ngân sách (không theo điểm, vì thầy không có điểm từng từ); cận trên trên test dùng đầu ra thầy không cắt và báo cáo tỉ lệ nén thực tế |
| Của ta | ViWord-C (XLM-R-large) × {π: none / cấu hình chọn trên dev} × {α = 0 / α* (bản 8 trang)} |

`stage_eval` hiện thực từng dòng bằng spec cụ thể: `stopword`, `lexical`, `ppl_sent`, LLMLingua theo perplexity
(`llmlingua:lm=$LM`), LLMLingua-2 package gốc (`llmlingua2`), LLMLingua-2-vi/ViWord-C và
`random:seed=0|1|2`. `longllmlingua:lm=$LM` chỉ được thêm cho ViMMRC và Belebele vì đây là baseline có câu hỏi.
Chạy `bash scripts/run_pipeline.sh teacher_test` trước `eval` để sinh cận trên thầy trên 300 mẫu mỗi tác vụ vào
`results/teacher_test/<stem>.jsonl`; `stage_eval` đọc chúng bằng `precomputed:...,name=teacher_upper` và báo cáo tỉ lệ
nén thực tế trong một lượt evaluate riêng 300 mẫu/tác vụ, không dùng cận trên này để chọn π.

Mức nén: với mọi phương pháp, `B = round(ratio × n_tokens(context))`, trong đó `n_tokens(context)` được đếm bằng
tokenizer của reader và chỉ tính phần ngữ cảnh gốc; chỉ dẫn, câu hỏi và lựa chọn không nằm trong B. `translate` cũng
dùng đúng B này: bản `context_en` chỉ thay đổi đầu vào của compressor, không thay đổi ngân sách chung. Báo cáo
`n_tokens_compressed / B` cho mọi phương pháp; các cặp so sánh có chênh lệch thực tế > 5% phải được đánh dấu. Với
các phương pháp chỉ nhận tham số `rate` theo tokenizer của chính chúng (LLMLingua, LLMLingua-2), **tìm nhị phân**
giá trị `rate` để độ dài thật theo tokenizer đích nằm trong [−3%, 0] của ngân sách: không bao giờ vượt B, cùng giới
hạn cứng mà `trim_to_budget` áp cho mọi phương pháp dựa trên mặt nạ.

**`ppl_sent` so với `lacc_sentence` của LACC.** Hai bộ chọn câu cùng xếp câu theo điểm trung bình của token (cao
giữ trước), nhưng khác ba điểm: (1) `lacc_sentence` dùng câu hỏi (perplexity có điều kiện theo câu hỏi kiểu
LongLLMLingua và trọng số liên quan câu hỏi), còn `ppl_sent` không dùng câu hỏi vì so sánh chính là
task-agnostic; (2) `lacc_sentence` luôn giữ câu đầu và câu cuối; (3) `lacc_sentence` thêm trọn câu cho tới khi
**vượt** ngân sách, còn `ppl_sent` không vượt B và lấp phần còn lại bằng phần đầu của câu tốt nhất chưa chọn (nếu
không, premise ViNLI một câu bị nén thành rỗng). Kết quả 0.760 của LACC ở 8x vì vậy là của một bộ chọn **có câu
hỏi**; tham chiếu có câu hỏi trong bài này là LongLLMLingua.

### 4.3 Ablation (bảng chính)
| Hàng | Nhãn | Quyết định | π | α | kiểm định |
|---|---|---|---|---|---|
| LLMLingua-2-vi | âm tiết | âm tiết | none | 0 | — |
| LLMLingua-2-vi + gộp logit theo từ | âm tiết | từ | none | 0 | H1 (2×2) |
| nhãn từ, quyết định âm tiết | từ (mềm) | âm tiết | none | 0 | H1 (2×2) |
| ViWord-C, π = none | từ (mềm) | từ | none | 0 | **H1** |
| LLMLingua-2-vi + `force_tokens` | âm tiết | âm tiết | force (khớp chuỗi P + chữ số, không khử nhập nhằng) | 0 | đối chứng H2 |
| + bảo vệ π (cấu hình chính) | từ (mềm) | từ | theo quy tắc §2.3 | 0 | **H2** |
| + tokenizer-aware (bản 8 trang) | từ (mềm) | từ | như hàng H2 | α* | **H3** |
| + chi phí = số âm tiết (bản 8 trang) | từ (mềm) | từ | như hàng H2 | α* với n_syl | đối chứng H3 |

Kiểm định:
- Bootstrap ghép cặp **theo cụm** (10k lần), cụm = tài liệu nguồn: premise ViNLI (một premise có nhiều hypothesis),
  đoạn ViMMRC (một đoạn có nhiều câu hỏi), đoạn Belebele (900 câu hỏi trên 488 đoạn), bài VietNews. Mỗi lần lấy mẫu
  lại chọn ID cụm có hoàn lại; toàn bộ dòng trong cụm được giữ và mẫu số mang trọng số bằng kích thước cụm, nên
  ước lượng vẫn là trung bình theo ví dụ nhưng không giả định các hypothesis/câu hỏi độc lập. **Báo cáo riêng từng
  tác vụ × reader × tỉ lệ**, không gộp.
- Test cho phần phân tích dùng đúng định dạng output thật (lỗi gom cụm của LACC lọt qua vì test fixture khác dữ
  liệu thật).
- **Tính MDE trước khi chạy test:** `scripts/mde.py` chạy trên các dòng dev sau khi chọn π, tính H1 theo **đúng
  tương phản đã đăng ký** `viword − llmlingua2vi` trên ViMMRC dev khả dụng, H2 theo NFR ViNLI và ROUGE-L VietNews.
  Bootstrap dùng thiết kế chéo seed × cụm như phân tích chính. MDE dev được quy về cỡ test theo
  MDE_test = MDE_dev · √(số cụm dev / số cụm test); report ghi cả hai, hiệu ứng pilot và SD giữa ba seed vào
  `results/mde_dev/report.{md,json}`. Lề non-inferiority 0.5 điểm ROUGE-L phải lớn hơn MDE **test** và SD seed;
  `stage_eval` bị chặn nếu report không PASS, nếu report tính cho lề khác, hoặc tính theo tương phản H1 cũ.
  Belebele hiện chỉ có split test trong dữ liệu chuẩn hoá, nên không được dùng thay cho dev và report phải đánh dấu
  task này thiếu coverage MDE.
- Họ Holm được khai báo bằng `--holm-family NAME:tasks=…,ratios=…,reader=primary,test=metric|nfr`; mỗi lượt
  `analyze.py` kiểm định **một** giả thuyết (`--hypothesis H1|H2`) và chỉ hiệu chỉnh trong họ của giả thuyết đó;
  mọi p-value khác in thô. H1: `{ViMMRC, Belebele} × {1/2, 1/3, 1/5}`, test=metric. H2:
  `{ViNLI} × {1/2, 1/3, 1/5}`, test=nfr. Các reader khác là phân tích mô tả.
- NFR: kiểm định chính là bootstrap ghép cặp **một phía** trên hiệu cờ lật nhãn (target − đối chứng), lấy mẫu lại
  theo premise (và theo seed khi gộp seed), vì nhiều hypothesis dùng chung một premise và ba seed đánh giá cùng một
  ví dụ. McNemar chính xác giả định các cặp độc lập nên chỉ được in **mô tả** theo từng seed (một phía). Đây là
  thay đổi so với "NFR dùng McNemar" ban đầu. Non-inferiority ROUGE-L với lề −0.5 (lý do chọn lề ở §2.3).
- p-value một phía của mọi phép so ghép cặp là p-value bootstrap percentile: tỉ lệ trung bình bootstrap của hiệu
  ≤ 0 (với NFR: ≥ 0, vì giả thuyết là target lật **ít** hơn). Khoảng tin cậy là phân vị 2.5–97.5%.
- H1, dự đoán cơ chế: `analyze.py --cbr-regression llmlingua2vi` hồi quy (OLS) mức tăng theo từng ví dụ
  `viword − llmlingua2vi` lên CBR của chính bản nén LLMLingua-2-vi ở ví dụ đó (trường `cbr` do `evaluate.py` ghi),
  CI bằng bootstrap theo cụm; báo cáo mô tả, không vào họ Holm.
- `stage_analyze` chạy một lượt H2 riêng với `--target viword_pi --reference viword llmlingua2vi_force
  llmlingua2vi_forcectx --hypothesis H2` sau khi chuẩn hoá các hậu tố seed; lượt này xuất kiểm định NFR (Holm trong
  họ H2) và non-inferiority ROUGE-L cho từng đối chứng.
- Seed: bảng chính gộp đúng ba seed bằng trung bình và độ lệch chuẩn giữa các seed. Kiểm định dùng bootstrap theo
  **thiết kế chéo seed × cụm**: mỗi lần lấy lại seed có hoàn lại và **một** bộ ID cụm có hoàn lại, áp chung cho mọi seed
  được chọn (cùng một cụm xuất hiện ở cả ba seed). Lấy cụm độc lập trong từng seed làm SE thấp đi khoảng 40% trên
  dữ liệu mô phỏng; lỗi này đã được sửa ngày 12/10/2026. Thiếu một seed thì phân tích dừng.

### 4.4 Phân tích phụ (lấp chỗ trống cho bài dài)
- CBR, RR và tỉ lệ giữ phủ định của mọi phương pháp, kể cả LLM thầy.
- CBR của ViWord-C khi đo bằng bộ tách từ khác (underthesea/pyvi), vì với RDRSegmenter CBR = 0 theo cách xây dựng.
- H2 tách theo tầng (T1 / T1+T2 / T1+T2+T3) và theo soft/hard.
- Ví dụ định tính "cắt đôi từ làm đổi nghĩa" (*học sinh → học*, *không đồng ý → đồng ý*).
- Ảnh hưởng của lỗi tách từ: thay RDRSegmenter bằng pyvi và xem độ nhạy.
- Độ dài đầu ra của reader (theo CAVEWOMAN): nén có làm model trả lời dài hơn không.
- Độ trễ nén (ms/văn bản) so với LLMLingua (cần LM 7B) và LLMLingua-2: mỗi dòng nén ghi `compress_ms`
  (thời gian của riêng lời gọi `compress`, không gồm tải mô hình).

## 5. Kế hoạch & tài nguyên

| Tuần | Việc | Sản phẩm |
|---|---|---|
| 1 | Tuần cổng §4.0: dựng loader VietNews/ViNLI/ViMMRC/Belebele + khung đánh giá (tái dùng `../vncompress`); chạy chẩn đoán với 1 reader; thử thầy trên 200 mẫu dev | Bảng 1 (CBR, RR, retention, fertility); quyết định theo G1–G5 |
| 2–4 | Kiểm tra rò rỉ; sinh dữ liệu chưng cất (§3.2) + căn chỉnh nhãn cấp từ và cấp âm tiết; huấn luyện ViWord-C (XLM-R) + LLMLingua-2-vi + ô 2×2; chọn δ, α trên dev; kiểm tra nhanh checkpoint trên 200 mẫu dev; gán nhãn 200 câu cho bộ phát hiện P | ~8k mẫu train, 500 dev; checkpoint; precision/recall của P |
| 5–7 | Tính MDE; đánh giá đầy đủ: 2 reader × 4 tác vụ × 3 tỉ lệ (vLLM) | bảng chính + ablation; bản 4 trang có thể nộp từ khoảng tuần 6 |
| 8–10 | Phân tích, viết bài 4–8 trang, công bố code + danh sách hư từ + dữ liệu nén | bản nộp |

**Tính toán:**
- Encoder < 1 GPU-giờ mỗi lần chạy.
- LLM thầy iCOMM 27B qua endpoint vLLM: 10k đoạn × 2 mức nén, cỡ 10M token sinh → vài giờ, tuỳ tải server.
- Đánh giá: ~35 biến thể (tính cả 3 seed cho các mô hình huấn luyện và các ô 2×2) × 4 tác vụ × 3 tỉ lệ × ~1000
  mẫu × 2 reader ≈ 0.8–1 triệu lượt sinh (bản 8 trang: nhân thêm theo số reader). LLMLingua và Selective
  Context cần thêm một LM để nén.
- Tổng dự trù **3–5 GPU-ngày** trên H100, tính cả gỡ lỗi.

**Cấu trúc code** (đã cài đặt; cách chạy ở `README.md`). Các cơ chế của khung đánh giá LACC (bootstrap theo
cụm, kiểm tra rò rỉ, các arm mốc) được viết lại gọn trong repo này, không import từ `../vncompress`:
```
viword/
  segment.py      # tách từ VnCoreNLP; underthesea/pyvi chỉ cho độ nhạy, mặt nạ âm tiết, render, căn chỉnh
  segment_meta.py # sidecar provenance của cache tách từ (bộ tách, SHA-256, xác minh lại bằng VnCoreNLP)
  protect.py      # tập P theo (từ, POS, ngữ cảnh) + luật khử nhập nhằng
  select.py       # tham lam s/c_T^α + bước lấp, knapsack DP chính xác, cắt theo độ dài thật
  model.py        # encoder chấm điểm theo âm tiết, cửa sổ trượt, đối chứng tra từ
  compressor.py   # giao diện chung; ViWord-C, LLMLingua-2-vi và các ô 2×2
  baselines.py    # Lead-k, truncation, random, stopword, luật từ vựng, chọn câu, Selective Context,
                  # LLMLingua(-2), LongLLMLingua, probe phủ định, precomputed (cận trên thầy)
  distill.py      # prompt thầy, nhãn âm tiết + nhãn mềm cấp từ, kiểm tra rò rỉ
  train.py        # huấn luyện XLM-R
  diagnose.py     # CBR (+ mốc ngẫu nhiên), RR, retention, fertility, CV_T (§4.0)
  data.py, eval.py, stats.py   # dữ liệu, reader vLLM + metric, bootstrap seed × cụm / Holm / kiểm định NFR / MDE
scripts/          # prepare_data, leak_filter, segment_data, check_segmented, verify_segmentation,
                  # teacher_compress, review_teacher, distill, train, checkpoint_check, diagnose, compare_llmlingua2,
                  # gate_report, gate_eval, make_annotation_sheets, evaluate_protect, evaluate_probe, select_pi,
                  # select_alpha, mde, greedy_gap, check_reader_template, evaluate, bertscore, analyze;
                  # run_pipeline.sh nối tất cả
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
