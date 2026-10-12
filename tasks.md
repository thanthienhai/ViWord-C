# Kế hoạch rà soát: mã nguồn so với phương pháp đề xuất (DATN.md)

Mục tiêu: xác nhận code cài đúng phương pháp ViWord-C, các đối chứng và quy trình thống kê trong `DATN.md`
**trước** khi chạy full test. Mỗi chỗ lệch được xử lý theo một trong ba cách: sửa code, sửa DATN cho khớp code
(kèm lý do), hoặc ghi vào Limitations.

## Cách dùng

- Rà theo thứ tự **P0 → P1 → P2**. Trong P0, làm theo đợt 1 → 5. Các đợt đầu có thể buộc phải chưng cất hoặc
  huấn luyện lại, nên làm trước để tránh chạy lại các bước tốn GPU.

  | Mức | Ý nghĩa | Số mục |
  |---|---|---|
  | **P0** | Ảnh hưởng trực tiếp tới tính hợp lệ của H1/H2 hoặc của các cổng. Phải xong trước khi chạy test. | 29 |
  | **P1** | Ảnh hưởng tới baseline, độ đo phụ hoặc khả năng tái lập. Xong trước khi viết bài. | 24 |
  | **P2** | Tài liệu và chi tiết nhỏ. | 10 |

- Mã mục (A1, B3…) giữ theo mục của DATN để dễ tra: A = §3.1 tiền xử lý, B = §3.2 chưng cất, C = §3.3 huấn luyện,
  D = §3.4 bảo vệ π, E = §3.5 chọn theo ngân sách, F = §4.2 baseline, G = §2.2/§4.0 chẩn đoán, H = cổng G1–G5,
  I = §4.1 đánh giá, J = §4.3 thống kê, K = bảng ablation, L = tài liệu.
- "Dấu hiệu" là điều đã thấy khi đọc lướt code ngày 12/10/2026, cần xác nhận khi rà.
- Bước 0: `pytest -q` phải pass. Mỗi chỗ lệch đã sửa phải có thêm test tương ứng.
- Ghi kết luận từng mục vào bảng ở cuối file.

---

## P0: phải xong trước khi chạy test

### Đợt 1: quyết định phương pháp (nếu đổi thì phải chưng cất hoặc huấn luyện lại)

#### B1. Mô hình thầy
- **Rà soát:** DATN §3.2 ghi thầy là Qwen2.5-72B-Instruct. Pipeline dùng `icmodel/icom-model-llm-ic-v3.8-27b`.
- **Ở đâu:** `scripts/run_pipeline.sh:17-18`, `README.md:73`, `DATN.md:8`, `DATN.md:189`.
- **Kỳ vọng:** DATN ghi đúng thầy thực dùng (tên, kích thước, lý do đổi), hoặc đổi lại thầy. G4 đo trên đúng thầy
  dùng để chưng cất.

#### B2. Prompt thầy
- **Rà soát:**
  - Có đúng ràng buộc chỉ xoá, không đổi thứ tự, không thêm, không viết lại không.
  - Đầu vào có phải văn bản chưa tách từ không, và có sinh ở 2 mức 0.5 / 0.25 không.
  - Dấu hiệu: prompt dặn thầy giữ "phủ định, con số, tên riêng". Nhãn vì vậy đã chứa sẵn một phần π, có thể
    làm nhiễu phép so sánh H2 (π so với không π).
- **Ở đâu:** `viword/distill.py:17-50`, `scripts/run_pipeline.sh:150-155`.
- **Kỳ vọng:** ba điểm đầu đạt. Câu "giữ phủ định" được ghi vào DATN §3.2 và phần diễn giải H2, hoặc thử prompt
  không có câu này trên 200 mẫu dev để đo ảnh hưởng.

#### A5. Gộp điểm từ subword lên từ
- **Rà soát:** DATN §3.1 viết logit của từ = trung bình logit các subword. Code lấy trung bình hai tầng (subword
  lên âm tiết, rồi âm tiết lên từ), nên trọng số khác. Hai biến thể "subword đầu" và "p_keep × n_syl" mà §3.1
  yêu cầu thử chưa có tuỳ chọn trong `SelectionConfig`.
- **Ở đâu:** `viword/model.py:20-26`, `viword/model.py:89-95`, `viword/compressor.py:32-40`,
  `viword/compressor.py:81-82`.
- **Kỳ vọng:** chốt một định nghĩa, viết giống nhau ở DATN, README và code. Nếu giữ yêu cầu chọn trên dev thì thêm
  tuỳ chọn `pool=` và ghi lại kết quả chọn.

#### A6. Gộp xác suất hay gộp logit
- **Rà soát:** DATN §2.3 và §4.0 viết "gộp xác suất theo từ" (như LLMLingua-2 gốc). Code tính
  `sigmoid(mean(logit))`. Điều này ảnh hưởng tới ô 2×2 rẻ nhất và tới cổng G1.
- **Ở đâu:** `viword/compressor.py:81-85`.
- **Kỳ vọng:** chốt một cách, dùng cho mọi ô 2×2 và cho LLMLingua-2 gốc + gộp, và mô tả đúng trong DATN.

#### C1. Loss "BCE cấp từ"
- **Rà soát:** với nhãn từ, code chép nhãn mềm của từ cho từng âm tiết rồi tính BCE trên **từng âm tiết**. Hệ quả:
  từ nhiều âm tiết nặng hơn trong loss, và mô hình không được huấn luyện trên logit cấp từ dùng lúc suy luận.
- **Ở đâu:** `viword/train.py:26-30`, `viword/train.py:59-82`.
- **Kỳ vọng:** chọn một trong hai hướng.
  - Gộp logit lên từ trong loss khi nhãn là từ, với điều kiện hai ô 2×2 vẫn chỉ khác nhau ở nhãn.
  - Giữ cách hiện tại và sửa DATN thành "BCE cấp âm tiết, target mềm cấp từ".

### Đợt 2: dữ liệu, nhãn và ngân sách đúng

#### A2. Bộ tách từ là VnCoreNLP (RDRSegmenter) kèm POS và NER
- **Rà soát:** mọi file `data/segmented/*.jsonl` có được sinh bằng VnCoreNLP không. README cho phép dùng
  `underthesea` khi chạy thử.
- **Ở đâu:** `viword/segment.py:174-191`, `scripts/segment_data.py`, `scripts/run_pipeline.sh:20`.
- **Kỳ vọng:** mỗi file ghi rõ bộ tách từ đã dùng và đều là `vncorenlp`. File nào sinh bằng bộ khác thì sinh lại.

#### A3. Dấu câu và cách ghép văn bản giống nhau giữa các hệ thống
- **Rà soát:** mọi ô 2×2 có đi qua cùng `render()` không. Liệt kê các phương pháp không đi qua `render()`:
  LLMLingua gốc, `translate`, `precomputed`.
- **Ở đâu:** `viword/segment.py:76-97`, `viword/compressor.py:43-65`, `viword/baselines.py:226-300`.
- **Kỳ vọng:** cùng một mặt nạ cho ra cùng một chuỗi ở mọi ô 2×2. Không có ngoại lệ nào thuộc phép so sánh H1.

#### B3. Căn chỉnh đầu ra thầy về âm tiết gốc
- **Rà soát:**
  - Căn chỉnh dùng `difflib`, không phải LCS chính xác như DATN viết. Cần thử trường hợp từ lặp lại (*của*, *và*,
    số), vì có thể khớp nhầm vị trí.
  - Tiêu chí lọc (>10% chữ không có trong nguồn, giữ >90%) cần so với tiêu chí của LLMLingua-2 mà DATN dẫn.
- **Ở đâu:** `viword/segment.py:104-139`, `scripts/distill.py:28-50`.
- **Kỳ vọng:** có test cho từ lặp và cho đầu ra đổi thứ tự. Kiểm tay 50 mẫu: nhãn sai < 2%. DATN mô tả đúng thuật
  toán căn chỉnh và tiêu chí lọc thực dùng.

#### B4. Hai cách nhìn nhãn từ cùng một căn chỉnh
- **Rà soát:** nhãn âm tiết (0/1) và nhãn từ mềm có sinh từ cùng một mặt nạ không.
- **Ở đâu:** `viword/distill.py:53-74`.
- **Kỳ vọng:** trung bình theo âm tiết của nhãn từ đúng bằng `keep_rate`. Tỉ lệ từ bị thầy giữ một phần được báo
  cáo.

#### B5. Kiểm tra rò rỉ
- **Rà soát:**
  - Index n-gram có đủ ViNLI, VietNews test, ViMMRC, Belebele và các tập dev không.
  - Thiếu file đánh giá thì script có báo lỗi không.
  - Bài VietNews test có bị loại theo id, không chỉ theo n-gram, không.
- **Ở đâu:** `viword/distill.py:77-102`, `scripts/leak_filter.py`, `scripts/run_pipeline.sh:44-56`,
  `scripts/prepare_data.py`.
- **Kỳ vọng:** thiếu file thì báo lỗi, log ghi số đoạn bị loại theo nguồn, và id VietNews của kho chưng cất không
  giao với id VietNews test.

#### E6. Định nghĩa ngân sách
- **Rà soát:** B = round(tỉ lệ × số token ngữ cảnh), đếm bằng tokenizer của reader, không tính chỉ dẫn hay câu
  hỏi. Áp dụng như nhau cho mọi phương pháp, kể cả `translate`.
- **Ở đâu:** `scripts/evaluate.py:30-49`, `scripts/diagnose.py:75`.
- **Kỳ vọng:** khớp §4.2. Tỉ lệ `n_tokens_compressed / budget` được báo cáo cho mọi phương pháp, và các cặp lệch
  > 5% được đánh dấu.

### Đợt 3: đối chứng và ưu tiên bảo vệ π

#### D2. Luật khử nhập nhằng
- **Rà soát:** đối chiếu từng luật trong §3.4 với code.
  - *không* / *chưa* cuối câu hỏi chỉ được bắt khi token kế tiếp là `?`.
  - *không* trước số được coi là phủ định trong `has_negation` và `probe_negation`.
  - *được* chưa có luật.
  - *có thể* phụ thuộc bộ tách từ có gộp thành một từ hay không.
- **Ở đâu:** `viword/protect.py:29-72`, `tests/test_protect.py`.
- **Kỳ vọng:** mỗi luật có một test dương và một test âm. Luật nào không cài thì ghi vào DATN.

#### D4. Đánh giá bộ phát hiện P trên ~200 câu gán nhãn tay
- **Rà soát:** tập gán nhãn và script tính precision/recall theo tầng chưa có trong repo.
- **Ở đâu:** chưa có. README:170-172 cũng ghi "phải đo, không được giả định".
- **Kỳ vọng:** có file nhãn, script đánh giá và bảng P/R theo tầng, có **trước** khi chọn cấu hình π.
- **Đã làm:** `data/protect_eval.jsonl` có 200 mẫu dev (100 ViNLI + 100 VietNews); `scripts/evaluate_protect.py`
  sinh P/R/F1 theo T1–T3; `stage_eval` yêu cầu `results/protect_eval.md` trước khi chạy các cấu hình π.

#### D6. Đối chứng H2: LLMLingua-2-vi + `force_tokens`
- **Rà soát:** DATN yêu cầu dùng cùng danh sách P. Code chỉ lấy từ một âm tiết và chữ số; thiếu *không hề, chưa
  từng, trừ khi, giá mà, có thể, được phép, tất cả* và NER. Có hai cài đặt (`llmlingua2_force` và
  `llmlingua2vi_force`) nhưng chưa rõ cài nào là đối chứng chính thức.
- **Ở đâu:** `viword/protect.py:83-87`, `viword/methods.py:65-67`, `scripts/run_pipeline.sh:175`.
- **Kỳ vọng:** đối chứng được ghi rõ trong DATN. Khác biệt với P được liệt kê kèm lý do.
- **Đã làm:** chốt `llmlingua2vi_force_s*` là đối chứng H2 chính thức với `protect=force`, dùng P động gồm cụm,
  số và NER; `llmlingua2_force` là baseline package chẩn đoán, không phải hàng H2 chính. `naive_syllable_protection`
  được giữ riêng cho legacy ablation và không còn dùng trong pipeline H2. DATN/README đã ghi thêm sai khác occurrence-
  level của API `force_tokens` khi mặt chữ lặp ở các ngữ cảnh POS khác nhau.

#### D7. Chọn cấu hình π trên dev
- **Rà soát:** §2.3 quy định quét (tầng × soft/hard × δ) trên dev, giữ cấu hình có ROUGE-L giảm ≤ 0.5, rồi chọn
  cấu hình có NFR thấp nhất. Hiện cấu hình lấy từ biến môi trường mặc định (`DELTA=0.2`, `TIERS=T1+T2+T3`) và chưa
  có script quét.
- **Ở đâu:** `scripts/run_pipeline.sh:176`.
- **Kỳ vọng:** có script quét trên dev. Cấu hình được chọn ghi ra file và commit trước khi stage `eval` chạy.
- **Đã làm:** `scripts/run_pipeline.sh select_pi` chạy grid T1/T1+T2/T1+T2+T3 × soft/hard × δ trên ViNLI +
  VietNews dev; `scripts/select_pi.py` lọc theo ROUGE-L ≤ 0.5 điểm rồi chọn NFR thấp nhất, ghi
  `config/pi_selected.json`, `config/pi_selected.env` và `results/pi_selection.md`. `stage_eval` bắt buộc đọc
  config này, không còn dùng mặc định `DELTA`/`TIERS`.

#### F1. Danh sách phương pháp ở stage `eval` so với §4.2
- **Rà soát:** `stage_eval` còn thiếu:
  - `stopword`, `lexical`;
  - LLMLingua theo perplexity;
  - LLMLingua-2 qua package gốc;
  - LongLLMLingua;
  - random 3 seed (hiện chỉ có seed 0);
  - cận trên của thầy trên tập test.
- **Ở đâu:** `scripts/run_pipeline.sh:166-189`, `viword/methods.py`.
- **Kỳ vọng:** mỗi dòng §4.2 có một spec trong `stage_eval`, hoặc được ghi rõ là bỏ và vì sao.
- **Đã làm:** `stage_eval` bổ sung `stopword`, `lexical`, LLMLingua theo perplexity, LLMLingua-2 package gốc,
  random với seed 0/1/2; LongLLMLingua được thêm có điều kiện cho ViMMRC/Belebele vì cần câu hỏi. Thêm stage
  `teacher_test` (mặc định 300 mẫu/tác vụ), rồi đưa `precomputed:...,name=teacher_upper` vào từng task test; nếu
  thiếu file cận trên, `stage_eval` dừng và báo lệnh cần chạy.

#### F2. "LLMLingua-2 gốc" trong chẩn đoán và cổng G1
- **Rà soát:** chẩn đoán và G1 dùng `scored:` với trọng số LLMLingua-2: nạp sẵn âm tiết, lấy trung bình logit.
  Package gốc gộp theo `▁` và lấy trung bình xác suất. Chưa chắc hai cách cho cùng kết quả.
- **Ở đâu:** `viword/model.py:60-98`, `viword/baselines.py:271-300`, `scripts/gate_report.py:33-41`.
- **Kỳ vọng:** trên 100 văn bản ở cùng ngân sách, mặt nạ giữ trùng ≥ 95% và CBR lệch < 1 điểm %. Nếu không đạt,
  G1 dùng bản package gốc.
- **Đã làm:** `scripts/compare_llmlingua2.py` so sánh 100 tài liệu ở ngân sách 1/3, báo cáo tỷ lệ mask trùng,
  độ đồng thuận theo âm tiết và chênh lệch CBR. `stage_diagnose` chạy thêm package `llmlingua2`; nếu báo cáo F2
  không đạt, `gate_report.py` tự chọn `llmlingua2` thay cho `llmlingua2_syl` khi quyết định G1.

#### K1. Mỗi hàng ablation §4.3 có đúng một spec
- **Rà soát:** ánh xạ hàng → spec trong `stage_eval`.

  | Hàng §4.3 | Spec hiện có |
  |---|---|
  | LLMLingua-2-vi | `llmlingua2vi_s*` |
  | + gộp theo từ | `llmlingua2vi_wordpool_s*` |
  | nhãn từ, quyết định âm tiết | `viword_syl_s*` |
  | ViWord-C, π = none | `viword_s*` |
  | + `force_tokens` | `llmlingua2vi_force_s*` |
  | + π (cấu hình chính) | `viword_pi_s*` (đọc từ `config/pi_selected.env`, chọn theo D7) |
  | + tokenizer-aware / + n_syl | chưa có (bản 8 trang) |
- **Ở đâu:** `DATN.md:358-368`, `scripts/run_pipeline.sh:166-189`.
- **Kỳ vọng:** bảng ánh xạ đầy đủ được ghi vào README.
- **Đã làm:** README có bảng ánh xạ đầy đủ cho 6 hàng chính, theo từng seed 0/1/2; hai hàng
  `tokenizer-aware` và `cost = số âm tiết` được ghi rõ là chưa chạy trong bản chính, dành cho bản 8 trang.

### Đợt 4: độ đo chẩn đoán và cổng

#### G1. CBR và mốc ngẫu nhiên
- **Rà soát:** mốc ngẫu nhiên chỉ tính cho từ 2 âm tiết, trong khi CBR gộp cả từ 3–4 âm tiết.
- **Ở đâu:** `viword/diagnose.py:43-80`, `scripts/diagnose.py:84-92`.
- **Kỳ vọng:** mốc tính theo phân phối số âm tiết thực tế (hoặc CBR báo cáo riêng cho từ 2 âm tiết). Có test với
  mặt nạ ngẫu nhiên: CBR đo được hội tụ về mốc.
- **Đã làm:** thêm `chance_cbr_distribution()` theo độ dài từ thực tế; `diagnose.py` ghi `chance_cbr` và phân phối
  độ dài, vẫn giữ `chance_cbr_2syl` để đối chiếu. Test Monte Carlo với từ 2/3/4 âm tiết xác nhận CBR hội tụ về mốc.

#### G3. Recovery Rate (RR)
- **Rà soát:** mỗi âm tiết bị mất được thay bằng một mask duy nhất. Âm tiết mà XLM-R tách thành ≥ 2 subword vì vậy
  không bao giờ khôi phục được, kéo RR thấp. Pipeline dùng `xlm-roberta-base`, khác với DATN.
- **Ở đâu:** `viword/diagnose.py:83-112`, `scripts/run_pipeline.sh:86`.
- **Kỳ vọng:** RR xử lý được âm tiết nhiều subword, hoặc dùng RR qua reader. Mô hình RR khớp DATN. Có test với âm
  tiết 2 subword.
- **Đã làm:** RR đếm subword bằng tokenizer của MLM và chèn đúng số `<mask>` cho từng âm tiết bị xoá; kết quả
  multi-mask được gom theo âm tiết rồi giải mã trước khi so khớp. Mặc định pipeline đổi sang `xlm-roberta-large`
  (có thể ghi đè bằng `RR_MODEL`), và thêm test hồi quy với một âm tiết hai subword.

#### G6. Probe phủ định và đếm premise có phủ định
- **Rà soát:** probe chỉ xoá phủ định (sau khử nhập nhằng). Phụ thuộc kết quả D2.
- **Ở đâu:** `viword/baselines.py:216-223`, `viword/diagnose.py:141-142`, `scripts/gate_report.py:43-50`.
- **Kỳ vọng:** kiểm tay 50 premise bị probe sửa: ≥ 95% là phủ định thật.
- **Đã làm:** bổ sung ngoại lệ `không kích` và `không lâu`, thêm 50 mẫu ViNLI dev gán nhãn tay trong
  `data/probe_negation_eval.jsonl`, script `scripts/evaluate_probe.py`, báo cáo JSON/Markdown và nối G6 vào
  `stage_protect_eval`/`gate_report`. Audit hiện đạt precision 100% (50/50), recall 100%.

#### H1. Cổng G1
- **Rà soát:** ngưỡng 5% ở tỉ lệ 1/3 và phương pháp dùng (liên quan F2). Vế "sau tuần 4: CBR của LLMLingua-2-vi
  ≥ 5%" chưa có lệnh đo.
- **Ở đâu:** `scripts/gate_report.py:36-41`, `scripts/gate_eval.py:89-97`.
- **Kỳ vọng:** cả hai vế có lệnh đo. Vế sau chạy `diagnose` với checkpoint `llmlingua2vi_s*`.
- **Đã làm:** `stage_diagnose` giữ phép đo tuần đầu với `llmlingua2_syl` và, khi thấy
  `runs/llmlingua2vi_s*`, chạy thêm `diagnose.py` cho mọi seed. `gate_report.py` nhận
  `--llmlingua2vi-reports`, báo cáo từng seed và CBR gộp tại 1/3, đạt khi CBR gộp ≥ 5%; thêm test pooling.

#### H2. Cổng G2
- **Rà soát:** NFR tính trên dev, số cặp đếm trên test (chỉ dùng văn bản, không dùng nhãn).
- **Ở đâu:** `scripts/gate_eval.py:99-107`, `scripts/gate_report.py:43-50`.
- **Kỳ vọng:** ngưỡng khớp DATN. Ghi rõ tập dùng cho từng vế và xác nhận không vi phạm nguyên tắc "không nhìn test".
- **Đã làm:** `gate_eval.py --split dev` là reader gate dev-only và dùng ngưỡng NFR `0.05`; truyền `--split test`
  sẽ dừng. `gate_report.py` dùng ngưỡng 300 cho vế đếm trên `vinli_test`, chiếu chỉ `id`/`cluster`/`doc` và bỏ
  nhãn NLI. DATN/README đã ghi rõ test chỉ cung cấp văn bản cho vế đếm cấu trúc, không cung cấp nhãn hay kết quả
  reader để chọn cấu hình hoặc tính NFR.

### Đợt 5: các bước bắt buộc trước test và thống kê

#### C3. Kiểm tra nhanh checkpoint bằng reader
- **Rà soát:** §3.3 yêu cầu chạy 200 mẫu dev; checkpoint phải thắng truncation và đối chứng tra từ (bài học E6 của
  LACC). Pipeline hiện đi thẳng từ `train` sang `eval`.
- **Ở đâu:** `scripts/run_pipeline.sh`.
- **Kỳ vọng:** có stage `ckpt_check` trên dev in PASS/FAIL. `eval` chỉ chạy khi PASS.
- **Đã làm:** thêm `stage_ckpt_check`, chạy 200 mẫu mỗi dev set ở tỉ lệ 1/3 cho toàn bộ `viword_s*` và
  `llmlingua2vi_s*`, so với `truncation` và `lexical`, ghi `results/ckpt_check/report.{md,json}`. `stage_eval`
  gọi `--verify` và dừng nếu JSON không phải PASS; `full_h100.sh` cũng chạy cổng này ngay sau `train`.

#### J1. Bootstrap theo cụm và định nghĩa cụm
- **Rà soát:** lấy mẫu theo cụm có trọng số theo kích thước cụm. Cụm: ViNLI = premise, ViMMRC = đoạn, Belebele =
  link, VietNews = bài.
- **Ở đâu:** `viword/stats.py:15-51`, `scripts/prepare_data.py:76-111`, `viword/data.py:122`.
- **Kỳ vọng:** số cụm khớp dữ liệu (Belebele 488, ViNLI test 377). Có test chạy trên đúng định dạng row thật của
  `evaluate.py`.
- **Đã làm:** `cluster_bootstrap` ghi rõ và kiểm tra việc lấy lại ID cụm có hoàn lại, giữ trọng số theo số dòng trong
  cụm; test mới đọc row chuẩn hóa thật để kiểm tra 488 cụm Belebele, 377 premise ViNLI test và dạng row đầu ra của
  `evaluate.py` (kèm kiểm tra lệch độ dài).

#### J2. Họ giả thuyết cho Holm
- **Rà soát:** DATN quy định họ H1 = {ViMMRC, Belebele} × 3 tỉ lệ với reader chính. `analyze.py` hiện áp Holm trên
  mọi tác vụ có trong rows.
- **Ở đâu:** `scripts/analyze.py:80-98`.
- **Kỳ vọng:** có tham số khai báo họ cho từng giả thuyết; bảng H1 chỉ hiệu chỉnh trong họ đã đăng ký.
- **Đã làm:** thêm `--holm-family` (có thể lặp, mỗi spec có tên họ, task, tỉ lệ và reader) cùng
  `--primary-reader`. Mặc định H1 là {ViMMRC, Belebele} × {1/2, 1/3, 1/5}; chỉ các ô thuộc họ H1
  được Holm, còn p-value ngoài họ vẫn báo dạng thô và cột hiệu chỉnh là `—`. Pipeline truyền rõ spec này.

#### J3. Gộp 3 seed
- **Rà soát:** DATN yêu cầu kết luận đúng ở cả 3 seed, hoặc dùng bootstrap phân tầng. `stage_analyze` hiện chạy
  từng seed riêng, không gộp.
- **Ở đâu:** `scripts/run_pipeline.sh:192-201`, `scripts/analyze.py`.
- **Kỳ vọng:** có chế độ gộp seed (trung bình ± độ lệch chuẩn) và kiểm định theo đúng quy tắc DATN.
- **Đã làm:** thêm `--seed-mode aggregate --seeds 0 1 2`; các method hậu tố `_s0/_s1/_s2` được ghép theo cùng
  task/reader/tỉ lệ, bảng chính báo cáo trung bình ± SD giữa seed, và target–reference dùng bootstrap phân tầng
  seed → cụm. Thiếu seed làm phân tích lỗi. `stage_analyze` giờ chạy một bảng gộp vào `results/analysis_3seed.md`.

#### J4. MDE trước khi chạy test
- **Rà soát:** hàm `minimum_detectable_effect` có sẵn nhưng không được gọi ở đâu.
- **Ở đâu:** `viword/stats.py:64-70`.
- **Kỳ vọng:** có script tính MDE cho H1 và H2 trên dev, ghi ra file trước khi chạy test. Lề 0.5 ROUGE-L được đối
  chiếu với MDE và với độ lệch chuẩn giữa 3 seed.
- **Đã làm:** thêm `scripts/mde.py` và stage `mde` sau `select_pi`, dùng bootstrap phân tầng seed → cụm trên dev.
  Report `results/mde_dev/report.{md,json}` ghi MDE/hiệu ứng/SD cho H1, H2 NFR và H2 ROUGE-L; `stage_eval` gọi
  `mde.py --verify` và dừng nếu 0.5 điểm không lớn hơn MDE hoặc SD seed. Belebele thiếu split dev nên được báo rõ,
  không lấy mẫu test thay thế.

#### J5. Kiểm định H2
- **Rà soát:** `stage_analyze` chỉ đặt target là `viword_s*`, nên H2 (`viword_pi` so với `viword` và
  `llmlingua2vi_force`: NFR theo McNemar, non-inferiority ROUGE-L) không được kiểm định.
- **Ở đâu:** `scripts/run_pipeline.sh:192-201`, `scripts/analyze.py:125-168`.
- **Kỳ vọng:** có lời gọi `analyze.py --target viword_pi_s* --reference viword_s* llmlingua2vi_force_s*`.
- **Đã làm:** `stage_analyze` giữ lượt H1 và thêm lượt H2 với target `viword_pi`, reference `viword` và
  `llmlingua2vi_force`; `aggregate_seed_rows` chuẩn hoá tương ứng các method `_s0/_s1/_s2`. Report chung ghi
  McNemar trên NFR cho cả hai đối chứng và non-inferiority ROUGE-L trên VietNews; thêm test hồi quy cho cả hai phép.

---

## P1: xong trước khi viết bài

| Mã | Rà soát gì | Ở đâu | Kết quả kỳ vọng |
|---|---|---|---|
| A1 | NFC và vị trí dấu thanh giống nhau ở mọi điểm vào: tách từ, ngân sách, thầy, căn chỉnh | `viword/segment.py:36-55`, `viword/data.py:94` | Văn bản NFD hoặc dấu kiểu cũ cho cùng `doc`, cùng ngân sách, cùng nhãn |
| A4 | Cửa sổ trượt: theo ranh giới câu, trùng 1 câu; câu > 200 âm tiết bị cắt nhỏ | `viword/model.py:29-98` | Test: mọi âm tiết được phủ, không âm tiết nào logit 0; việc cắt câu được ghi vào README |
| C2 | `pos_weight`, AdamW, lr 1e-5, 3 epoch, chọn theo loss dev, chia dev theo id đoạn | `viword/train.py:85-148`, `scripts/train.py` | Khớp §3.3; hai loại nhãn dùng cùng `dev_ids.jsonl` |
| D1 | Danh sách từ các tầng T1/T2/T3; loại *các, những, cái, con* | `viword/protect.py:14-26` | Khớp từng mục bảng §3.4, có test |
| D5 | `soft` / `hard`; bước `trim` có thể bỏ cả từ trong P; tỉ lệ ngân sách dùng cho P | `viword/compressor.py:88-94`, `viword/select.py:71-79` | Có test cho `hard`; quyết định hành vi `trim`; log token dùng cho P |
| E1 | c_T(w) = số token của `" " + w`; dấu câu không có khoảng trắng | `viword/select.py:14-29` | Khớp §3.5 bước 1, có test |
| E2 | Tham lam theo s/c^α, bước lấp theo s, ưu tiên `hard` | `viword/select.py:32-50` | Khớp §3.5 bước 2–3; test với α = 0 và α = 1 |
| E3 | Trim theo độ dài thật; báo cáo độ lệch giữa Σc_T và độ dài thật | `viword/select.py:71-79` | Log Σc_T trước khi trim hoặc số từ bị trim |
| E5 | Quét α trên dev và đối chứng `cost=syllables` (bản 8 trang) | `viword/compressor.py:39` | Có script quét; α* ghi ra file trước test |
| F3 | Tìm nhị phân `rate` cho phép baseline vượt tới +3%; đầu ra không qua `render()` | `viword/baselines.py:251-300` | Quyết định có cắt về đúng ngân sách không; báo cáo tỉ lệ nén thực tế |
| F4 | `tfidf_sent`, `ppl_sent` so với `lacc_sentence`; bước lấp câu dở | `viword/baselines.py:107-188` | Cùng hướng xếp hạng với LACC; bước lấp ghi vào DATN |
| F6 | Đối chứng tra từ đang fit trên cả 500 đoạn dev của encoder | `viword/model.py:101-132`, `scripts/run_pipeline.sh:171` | Fit chỉ trên phần train |
| G2 | Phân loại từ: chưa tách được từ ghép đẳng lập; heuristic láy theo phụ âm đầu | `viword/diagnose.py:28-40` | Có từ điển hoặc sửa tên nhóm; kiểm tay 100 từ |
| G4 | Tỉ lệ giữ theo tầng; "giữ" nghĩa là còn ít nhất 1 âm tiết | `viword/diagnose.py:117-138` | Khớp §2.2; định nghĩa ghi rõ trong bài |
| G5 | Danh sách tokenizer chi phí dùng Gemma-3, thiếu Vistral | `scripts/run_pipeline.sh:37` | Có đủ hai reader chính, khớp DATN |
| H3 | G3: danh sách bộ nén cấp token thiếu LLMLingua theo perplexity và `lexical` | `scripts/gate_eval.py:54-57` | Gồm mọi bộ nén cấp token có trong §4.0 |
| H4 | G4: 200 mẫu dev, thầy dùng `trim=1`, đúng thầy dùng để chưng cất | `scripts/gate_eval.py:71-87` | Kết quả G4 ghi kèm tên thầy |
| H6 | Ngưỡng các cổng không bị sửa sau khi đã có số liệu | `git log -p` các file cổng và `DATN.md` | Không có commit đổi ngưỡng sau lần chạy đầu, hoặc có lý do |
| I1 | Tác vụ ngoài DATN: ViQuAD, GSM8K-vi, OneRuler-vi | `viword/data.py:15-23`, `DATN.md:311-330` | DATN ghi vai trò (chính hay khám phá); tác vụ khám phá không vào họ Holm |
| I2 | Chỉ nén ngữ cảnh; ViNLI giữ hay bỏ nhãn `other` | `viword/eval.py:17-44` | Khớp §4.1; quyết định về `other` được ghi rõ |
| I3 | ROUGE tự cài trên âm tiết; BERTScore chưa cài; loại bài thơ ViMMRC | `viword/eval.py:80-138` | ROUGE đối chiếu được với bản chuẩn; cài BERTScore hoặc bỏ khỏi DATN |
| I4 | Hai reader (Qwen2.5, Vistral), greedy, tắt thinking; template chat của Vistral | `viword/eval.py:143-216` | Có lệnh chạy cả hai reader; template được kiểm tra |
| I5 | Parser NLI/MC khi ngữ cảnh bị nén mạnh | `viword/eval.py:47-75` | Tỉ lệ dự đoán rỗng < 2% ở mọi phương pháp × tỉ lệ nén |
| J6 | Tương phản 2×2; hồi quy chênh lệch điểm theo CBR chưa có | `scripts/analyze.py:100-123` | Test với hiệu ứng biết trước; có phân tích hồi quy hoặc bỏ dự đoán này |

---

## P2: tài liệu và chi tiết nhỏ

| Mã | Rà soát gì | Ở đâu | Kết quả kỳ vọng |
|---|---|---|---|
| B6 | Kho chưng cất 5k–10k đoạn, 300–1500 âm tiết | `scripts/prepare_data.py`, `data/README.md` | Số liệu khớp DATN; đủ ~8k train + 500 dev |
| D3 | P là cặp (từ, POS), nhưng code chỉ dùng POS cho 5 từ | `viword/protect.py:33-40` | DATN mô tả đúng, hoặc bổ sung ràng buộc POS |
| E4 | Knapsack DP bỏ qua `priority`/`alpha`; chưa có lệnh đo khoảng cách so với tham lam | `viword/select.py:53-68` | Test so với vét cạn; có lệnh đo khoảng cách |
| F5 | Selective Context cài lại ở cấp từ | `viword/baselines.py:147-213` | Test gán token cho từ; khác biệt với bản gốc ghi vào DATN |
| F7 | `translate` (Belebele); cắt đầu ra thầy từ cuối khi `trim=1` | `viword/methods.py:30-37`, `viword/baselines.py:226-246` | Khớp §4.2; báo cáo tỉ lệ nén thực tế của thầy |
| G7 | Bảng chẩn đoán tách theo loại bộ nén | `scripts/diagnose.py:105-112` | Có nhóm "loại bộ nén"; mỗi nhóm có ít nhất một đại diện |
| H5 | G5 loại trừ cứng `xlm-roberta-base` | `scripts/gate_report.py:52-59` | Chỉ đếm tokenizer đích có trong DATN |
| I6 | Độ trễ nén chưa được ghi | `scripts/evaluate.py:30-49` | Thời gian nén ghi trong `*.compressed.jsonl` |
| J7 | p-value bootstrap; McNemar hai phía cho giả thuyết một phía | `viword/stats.py:36-89` | Cách tính ghi rõ trong bài; chiều kiểm định khớp H2 |
| L1 | README, DATN và code đồng bộ: "42 test", cấu trúc code ở §5, các tham chiếu "DATN §x" | `README.md:52`, `README.md:158-172`, `DATN.md:409-426` | Không còn mâu thuẫn giữa ba nguồn |

---

## Bảng ghi kết quả

| Mục | Mức | Đợt | Kết luận (Đạt / Sửa code / Sửa DATN / Limitations) | Người rà | Ngày | Commit / ghi chú |
|---|---|---|---|---|---|---|
| B1 | P0 | 1 | Sửa DATN | Codex | 12/10/2026 | Ghi iCOMM `icmodel/icom-model-llm-ic-v3.8-27b` (27B), lý do đổi từ Qwen2.5-72B-Instruct; G4 và chưng cất dùng chung `TEACHER_MODEL` |
| B2 | P0 | 1 | Sửa DATN | Codex | 12/10/2026 | Xác nhận prompt chỉ xoá/raw text/2 mức 0.5–0.25; ghi prior giữ phủ định, số, tên riêng và giới hạn diễn giải H2 |
| A5 | P0 | 1 | Sửa DATN + code docs | Codex | 12/10/2026 | Chốt gộp hai tầng cố định subword→âm tiết→từ; bỏ yêu cầu thử `subword đầu`/`p_keep × n_syl` trên dev |
| A6 | P0 | 1 | Sửa DATN + code docs | Codex | 12/10/2026 | Chốt `sigmoid(mean(logit))` cho mọi nhánh `unit=word`, gồm 2×2 và LLMLingua-2 + gộp từ |
| C1 | P0 | 1 | Sửa DATN + README + code docs | Codex | 12/10/2026 | Giữ implementation hiện tại: BCE cấp âm tiết, target mềm cấp từ; xác nhận từ nhiều âm tiết đóng góp nhiều hạng loss hơn |
| A2 | P0 | 2 | Sửa code + README + DATN | Codex | 12/10/2026 | Pipeline chính khóa `vncorenlp`; `segment_data.py` ghi sidecar `*.jsonl.meta.json` (segmenter, POS/NER, SHA-256), checker bắt buộc mọi file; underthesea/pyvi chỉ còn cho §4.4 |
| A3 | P0 | 2 | Đạt + sửa docs/test | Codex | 12/10/2026 | Bốn ô 2×2 đều qua `ScoredCompressor → select_units → render`; LLMLingua package, `translate`, `precomputed` (và control `none/no_context`) được ghi rõ là ngoài H1; thêm test/định nghĩa chung về dấu câu và ghép chuỗi |
| B3 | P0 | 2 | Sửa code + DATN + test | Codex | 12/10/2026 | Đạt: thay `SequenceMatcher` bằng LCS chính xác Hunt–Szymanski; test từ lặp/số và đổi thứ tự; manual audit 50 mẫu đầu không có lỗi căn chỉnh vị trí (0/50, <2%), 5 mẫu bị loại đúng bởi `unmatched_ratio > 0.10`; tái sinh `data/distill/distilled.jsonl` (16.435 mẫu). Ngưỡng thực dùng `unmatched_ratio > 0.10` hoặc `keep_rate > 0.90`, phân biệt với LLMLingua-2 lọc theo phân vị 90% |
| B4 | P0 | 2 | Sửa code + DATN + README + test | Codex | 12/10/2026 | Đạt: `word_labels` lấy từ cùng `syllable_labels`; kiểm tra trung bình nhãn từ mở rộng theo âm tiết bằng `keep_rate`; xác minh 16.435/16.435 record không lỗi bất biến; báo cáo `teacher_cbr` theo mẫu và CBR gộp toàn corpus cùng số đếm partial/full; tái sinh `data/distill/distilled.jsonl` |
| B5 | P0 | 2 | Sửa leak_filter + pipeline + data + test | Codex | 12/10/2026 | Đạt: bắt buộc coverage ViNLI/VietNews/ViMMRC/Belebele và dev/test, file thiếu/rỗng báo lỗi; log coverage và loại theo source/reason; thêm lọc stable VietNews GUID. Chạy lại kho: 16.095 context index, loại 734 n-gram + 104 ID, còn 9.162 đoạn; giao ID VietNews train/distill với test = 0; segmentation/teacher/distill tái sinh đồng bộ |
| E6 | P0 | 2 | | | | |
| D2 | P0 | 3 | | | | |
| D4 | P0 | 3 | | | | |
| D6 | P0 | 3 | | | | |
| D7 | P0 | 3 | | | | |
| F1 | P0 | 3 | | | | |
| F2 | P0 | 3 | | | | |
| K1 | P0 | 3 | | | | |
| G1 | P0 | 4 | | | | |
| G3 | P0 | 4 | | | | |
| G6 | P0 | 4 | | | | |
| H1 | P0 | 4 | | | | |
| H2 | P0 | 4 | | | | |
| C3 | P0 | 5 | | | | |
| J1 | P0 | 5 | | | | |
| J2 | P0 | 5 | | | | |
| J3 | P0 | 5 | | | | |
| J4 | P0 | 5 | | | | |
| J5 | P0 | 5 | | | | |
| A1 | P1 | | | | | |
| A4 | P1 | | | | | |
| C2 | P1 | | | | | |
| D1 | P1 | | | | | |
| D5 | P1 | | | | | |
| E1 | P1 | | | | | |
| E2 | P1 | | | | | |
| E3 | P1 | | | | | |
| E5 | P1 | | | | | |
| F3 | P1 | | | | | |
| F4 | P1 | | | | | |
| F6 | P1 | | | | | |
| G2 | P1 | | | | | |
| G4 | P1 | | | | | |
| G5 | P1 | | | | | |
| H3 | P1 | | | | | |
| H4 | P1 | | | | | |
| H6 | P1 | | | | | |
| I1 | P1 | | | | | |
| I2 | P1 | | | | | |
| I3 | P1 | | | | | |
| I4 | P1 | | | | | |
| I5 | P1 | | | | | |
| J6 | P1 | | | | | |
| B6 | P2 | | | | | |
| D3 | P2 | | | | | |
| E4 | P2 | | | | | |
| F5 | P2 | | | | | |
| F7 | P2 | | | | | |
| G7 | P2 | | | | | |
| H5 | P2 | | | | | |
| I6 | P2 | | | | | |
| J7 | P2 | | | | | |
| L1 | P2 | | | | | |

---

## Rà soát lại lần 2 (12/10/2026)

Phạm vi: working tree chưa commit (40 file sửa, ~25 file mới). `pytest`: 110/110 pass (cần
`--basetemp` trỏ tới thư mục ghi được trên máy Windows này).

Tổng hợp: P0 đạt 14/29, đạt một phần 12, chưa đạt 3. P1 đạt 3/24, đạt một phần 4, chưa làm 17. P2 đạt 0/10,
đạt một phần 5, chưa làm 5.

### Lỗi phải sửa trước khi chạy test

| # | Lỗi | Ở đâu | Hệ quả |
|---|---|---|---|
| 1 | Bootstrap gộp seed lấy mẫu cụm **độc lập trong từng seed**, trong khi cụm là chung giữa các seed | `viword/stats.py:47-86` | SE thấp ~40% (mô phỏng 0,063 so với 0,103): CI hẹp, p nhỏ, MDE nhỏ. Lan sang J3, J4, J5, tương phản 2×2. Sửa: mỗi lần lặp rút một bộ cụm, dùng chung cho mọi seed được chọn |
| 2 | Spec gõ sai `longllingua`; test khoá luôn chuỗi sai | `scripts/run_pipeline.sh:454`, `tests/test_eval_method_inventory.py:18`, `viword/methods.py:8` | `eval` ViMMRC/Belebele báo `unknown method spec` sau khi đã nén các phương pháp trước, cache chưa ghi, `set -e` dừng stage |
| 3 | Nhãn D4 trùng 200/200 tài liệu với đầu ra `protection_tiers()`; không có tài liệu nào không chứa P | `data/protect_eval.jsonl`, `results/protect_eval.md` | P = R = 1.000 không chứng minh gì. Cần gán nhãn độc lập (không xem gợi ý của bộ phát hiện), có cả câu không chứa P |
| 4 | Nhãn G6: 50 dòng chỉ 30 premise, mọi nhãn `true`, mẫu lấy từ chỗ probe đã sửa | `data/probe_negation_eval.jsonl` | Recall 100% đúng theo cách chọn mẫu, không đo được gì. Cần 50 premise khác nhau, gồm cả premise probe không sửa |
| 5 | Kiểm tra checkpoint so với `lexical` thay vì `lexprior`; bắt cả `llmlingua2vi_s*` phải PASS; phải thắng truncation trên VietNews (chính thầy đã thua ở đây) | `scripts/checkpoint_check.py:22`, `:76-84` | So sai đối chứng của §3.3, và nhiều khả năng chặn `eval` vĩnh viễn |
| 6 | McNemar khi gộp seed cộng cặp bất đồng của 3 seed và mọi hypothesis cùng premise | `scripts/analyze.py:362-370` | n nhân 3, bỏ qua cụm, p quá nhỏ; kiểm định hai phía cho giả thuyết một phía |
| 7 | MDE của H1 đo `viword − viword_syl`, trong khi H1 kiểm định `viword − llmlingua2vi`; MDE ở cỡ dev, không quy về cỡ test; `--verify` bỏ qua `--margin-points` | `scripts/mde.py:128-143` | PASS/FAIL của MDE không nói gì về lực kiểm định của H1 |
| 8 | Đối chứng H2 `llmlingua2vi_force` không nhận `tiers=$PI_TIERS` | `scripts/run_pipeline.sh:441` | Nếu D7 chọn khác `T1+T2+T3` thì hai bên không còn cùng P |
| 9 | Họ Holm gán cứng `"H1"`; lượt H2 bị áp họ H1 | `scripts/analyze.py:289` | `--holm-family H2:...` không có tác dụng |

### Lỗi nhỏ và việc còn dở

- Bộ lọc theo GUID VietNews nhiều khả năng khớp giả: GUID là chỉ số trong từng split, 104 đoạn bị loại có trùng từ vựng
  trung vị 27%. Không gây rò rỉ, nhưng mô tả trong DATN §3.2 và `data/README.md` sai (`scripts/leak_filter.py:41-49`).
- `check_segmented.py` chỉ chạy ở `stage_segment`. 14/15 sidecar là hồi tố ("legacy snapshot"), không phải bằng chứng.
- `stage_diagnose` chỉ kiểm tra `$PROBE_REPORT` sau vòng chẩn đoán tốn GPU (`scripts/run_pipeline.sh:144-148`).
- D2: câu hỏi không có `?`, *không ạ?*, *chưa nhỉ?* vẫn tính là phủ định, DATN chưa ghi giới hạn này.
- D6: đối chứng `force` nay dùng P đã khử nhập nhằng, khác DATN gốc. Cần ghi rõ là H2 không còn đo giá trị của khử
  nhập nhằng.
- D7: mức sụt ROUGE lấy trung bình 3 tỉ lệ (sụt riêng ở 1/5 có thể lọt); hoà chọn "ít tầng", chưa phải "ít từ".
- F2: vế điểm của G1 (`scripts/gate_eval.py:60-61`) vẫn dùng bản `scored` khi F2 trượt.
- B3: không có artefact cho đợt kiểm tay 50 mẫu. Ghi chú B3/B4 ghi 16.435 mẫu, dữ liệu thật 16.274.
- B6: đoạn chưng cất dài nhất 2.862 âm tiết, vượt khoảng 300–1.500.
- F6: `lexprior` vẫn fit trên cả dev của encoder (`scripts/run_pipeline.sh:435`).
- L1: README vẫn ghi "42 test"; DATN §5 thiếu 7 script mới và `viword/segment_meta.py`.

### Kết luận từng mục

| Mức | Đạt | Đạt một phần | Chưa đạt |
|---|---|---|---|
| P0 | B1, B2, A5, A6, C1, A3, B4, E6, K1, D7, G1, H1, H2, J1 | A2, B3, B5, D2, D6, F2, G3, G6, J2, J3, J4, J5 | **D4, F1, C3** |
| P1 | C2, E2, H6 | D1, E1, F3, H4 | A1, A4, D5, E3, E5, F4, F6, G2, G4, G5, H3, I1, I2, I3, I4, I5, J6 |
| P2 | — | B6, E4, F5, F7, L1 | D3, G7, H5, I6, J7 |

---

## Sửa lỗi sau rà soát lần 2 (12/10/2026)

`pytest`: 135/135 pass. Chưa commit. Mỗi lỗi đã sửa đều có test mới.

### Lỗi chặn

| # | Lỗi | Đã sửa | Test |
|---|---|---|---|
| 1 | Bootstrap gộp seed lấy cụm độc lập theo seed | `viword/stats.py`: thiết kế chéo seed × cụm, một bộ cụm dùng chung cho mọi seed | `test_seed_bootstrap_shares_cluster_draws_across_seeds`, `..._matches_cluster_bootstrap_when_seeds_agree` |
| 2 | Spec `longllingua` | `run_pipeline.sh`, `methods.py`; test kiểm tra tĩnh mọi tên spec trong pipeline | `test_every_method_spec_in_the_pipeline_has_a_known_name` |
| 3–4 | Nhãn D4/G6 vòng tròn | Bộ cũ chuyển sang `data/annot/invalid_detector_prefilled/`. `scripts/make_annotation_sheets.py` sinh phiếu mù `data/annot/{protect,probe}_sheet.jsonl`. `evaluate_protect.py` / `evaluate_probe.py` / `gate_report.py` chỉ nhận phiếu `blind-v1` đã điền, từ khớp bản tách từ, ≥ 50 premise khác nhau bị probe sửa. `diagnose`, `select_pi`, `eval` dừng nếu chưa có | `test_evaluate_protect.py`, `test_evaluate_probe.py`, `test_annotation_sheets.py` |
| 5 | `ckpt_check` so sai đối chứng, chặn sai | So với `truncation` + `lexprior`; chỉ chặn `viword_s*`; luật đa số dev set như G4; LLMLingua-2-vi chỉ báo cáo | `test_checkpoint_check.py` |
| 6 | McNemar cộng dồn seed và cặp cùng premise | `flip_rate_test`: bootstrap một phía theo premise (+ seed), Holm trong họ H2; McNemar một phía chỉ in mô tả theo từng seed | `test_flip_rate_test_counts_premises_not_pairs`, `test_analyze_seeds.py` |
| 7 | MDE đo sai tương phản H1 | `scripts/mde.py`: `viword − llmlingua2vi`; quy MDE về cỡ test; `--verify` kiểm tra lề và tương phản; stage `mde` chạy `llmlingua2vi_s*` | `test_mde.py` |
| 8 | Đối chứng H2 không nhận tầng π | `tiers=$PI_TIERS`; `protect=force` nay đúng ngữ nghĩa `force_tokens` của DATN (khớp chuỗi, không khử nhập nhằng); bản P theo ngữ cảnh thành ablation `force_ctx` | `test_force_protection_is_plain_string_matching` |
| 9 | Họ Holm gán cứng H1 | `--hypothesis H1/H2`, họ có trường `test` (metric hoặc nfr) | `test_h2_run_does_not_reuse_the_h1_family` |

### Lỗi mới phát hiện khi sửa

- **A2: bản tách từ chính là underthesea, không phải VnCoreNLP.** `scripts/verify_segmentation.py` tách lại mẫu
  ngẫu nhiên: ViNLI, ViMMRC, VietNews, Belebele và `paragraphs_clean` khác VnCoreNLP 30/30. Chúng khớp từng byte với
  underthesea. Đã tách lại bằng VnCoreNLP; bản cũ ở `data/segmented_legacy_underthesea/`. Nhãn chưng cất dựng lại
  từ đầu ra thầy cũ (không gọi lại thầy): vẫn 16.274 mẫu, CBR của thầy 4,90% (trước: 6,58% theo từ underthesea).
  `check_segmented.py` nay từ chối sidecar hồi tố chưa xác minh, và mọi stage đọc `data/segmented` đều kiểm tra.
- **VnCoreNLP đổi dấu mới → cũ** ("hòa" → "hoà"), ngược với điều `data/README.md` ghi. `VnCoreNLPSegmenter` nay
  chuẩn hoá lại `wordForm`; cache đã được cập nhật và khớp đầu ra của bộ tách đã sửa (12/12 mẫu mỗi file).

### Lỗi nhỏ đã sửa

D2 (trợ từ cuối câu hỏi, *không* = số 0); D7 (điều kiện ROUGE xét từng tỉ lệ); F2 (G1 in thêm tham chiếu
package); F6 (`lexprior` loại `dev_ids`); G3 (RR so khớp sau `normalize`); G1 (`keep_rate` không tính dấu câu);
mô tả bộ lọc GUID VietNews (GUID là chỉ số theo split); `PROBE_REPORT` kiểm tra trước vòng GPU; `set -u` với mảng
rỗng; `full_h100.sh` chạy `teacher_test` trước `eval`; README (số test, ghi chú cài đặt), DATN §3–§5,
data/README, `data/SHA256SUMS`.

### Còn lại, cần người làm

1. **Gán nhãn hai phiếu mù** trong `data/annot/`, rồi chạy `protect_eval`. Pipeline dừng ở `diagnose` cho tới khi có.
2. **Đẩy dữ liệu mới lên HF** (`thanthienhai/viword-c-data`): `data/segmented/*`, `data/SHA256SUMS`, và hai phiếu sau
   khi điền. `ops_full_run.sh` so dữ liệu HF với `SHA256SUMS`, nên sẽ báo lỗi cho tới khi HF được cập nhật.
3. **Chạy lại mọi kết quả cũ** trong `results/diagnose`, `results/smoke`, `results/gate`. Chúng dùng đơn vị từ của
   underthesea và bootstrap cũ.
4. Các mục P1/P2 "chưa đạt" ở lần rà 2 (trừ những mục đã sửa ở trên) vẫn mở.

---

## Sửa các mục P1/P2 còn mở (12/10/2026)

`pytest`: 161/161 pass, pyflakes sạch. Chưa commit.

| Mục | Đã làm | Ở đâu | Test |
|---|---|---|---|
| A1 | Test cho từng điểm vào: `load_contexts`, ngân sách NFD/NFC, căn chỉnh khi thầy dùng kiểu dấu khác, chuẩn hoá `wordForm` của VnCoreNLP | `tests/test_inputs_and_windows.py` | 4 test |
| A4 | Test câu 450 âm tiết: mọi âm tiết được phủ và chấm điểm; ghi việc cắt câu > 200 âm tiết vào DATN §3.1 và README | như trên | 1 test |
| D1 | Test so khớp nguyên văn bảng §3.4, mỗi mục được bảo vệ ở ngữ cảnh thường, loại *các/những/cái/con* | như trên | 3 test |
| D3 | DATN §3.4 mô tả đúng: khớp mặt chữ + luật POS cho 5 từ + NER/POS số | `DATN.md` | — |
| D5 | Với `hard`, bước cắt bỏ từ ngoài P trước; ghi số token dành cho P | `viword/select.py`, `viword/compressor.py` | `test_trim_drops_unprotected_units_before_protected_ones`, `test_selection_stats_*` |
| E1 | Test `unit_cost` (khoảng trắng đầu, dấu câu, không bao giờ bằng 0) | `tests/test_selection_and_reporting.py` | 1 test |
| E3 | Mỗi dòng ghi `selection`: Σc_T, độ dài thật trước bước cắt, độ lệch, số đơn vị bị cắt | `compressor.py`, `scripts/evaluate.py` | 2 test |
| E4 | `scripts/greedy_gap.py`: tỉ số mục tiêu tham lam/DP | — | 1 test |
| E5 | `scripts/select_alpha.py` + stage `select_alpha`; quy tắc chọn α* ghi trong DATN §3.5; hàng H3 chỉ vào `eval` khi `config/alpha_selected.env` đã commit | `run_pipeline.sh` | `tests/test_select_alpha.py` |
| F3 | Tìm `rate` trong [−3%, 0], không bao giờ vượt B | `viword/baselines.py` | 1 test |
| F4 | Đối chiếu với `../vncompress`: `lacc_sentence` dùng câu hỏi, luôn giữ câu đầu/cuối, được vượt ngân sách; `ppl_sent` thì không. Ghi vào DATN §4.2 và README | — | — |
| F5 | Test gán surprisal token → từ; DATN ghi khác biệt với bản gốc | `tests/test_inputs_and_windows.py` | 1 test |
| F7 | DATN ghi cách cắt đầu ra thầy (`trim=1`, cắt từ cuối) | `DATN.md` | — |
| G2 | DATN đổi tên nhóm đúng với code (tên riêng / láy heuristic / khác) và ghi giới hạn | `DATN.md` | — |
| G4 | DATN định nghĩa "giữ" = còn ít nhất 1 âm tiết | `DATN.md` | — |
| G5 | Thêm Vistral vào danh sách tokenizer; `diagnose` nạp mọi tokenizer trước vòng GPU | `run_pipeline.sh`, `scripts/diagnose.py` | test tĩnh |
| G7 | Bảng chẩn đoán có cột và thứ tự theo nhóm bộ nén | `viword/diagnose.py`, `scripts/diagnose.py` | — |
| H3 | G3 xét mọi bộ nén cấp token (thêm LLMLingua, LLMLingua-2 package, `lexical`, `stopword`); `gate_eval` stage chạy đủ | `scripts/gate_eval.py`, `run_pipeline.sh` | test tĩnh |
| H4 | Báo cáo G4 ghi tên thầy | `scripts/gate_eval.py` | — |
| H5 | Bỏ loại trừ cứng `xlm-roberta-base` | `scripts/gate_report.py` | test tĩnh |
| I1 | DATN §4.1 thêm ViQuAD, GSM8K-vi, OneRuler-vi là tác vụ khám phá, ngoài họ Holm và cổng | `DATN.md` | — |
| I2 | Quyết định: giữ đủ 4 nhãn ViNLI | `DATN.md`, README | — |
| I3 | BERTScore PhoBERT (`scripts/bertscore.py`, stage `bertscore`); đã chạy thử trên CPU | `pyproject.toml` (`bert_score`) | 1 test |
| I4 | `PRIMARY_READER` tách khỏi `READER`; các stage quyết định trên dev chỉ chạy với reader chính; `READER_SYSTEM_PROMPT`; `scripts/check_reader_template.py` | `run_pipeline.sh`, `viword/eval.py` | 2 test |
| I5 | `analyze.py` thêm cột tỉ lệ câu trả lời rỗng, ⚠ khi > 2% | `scripts/analyze.py` | 1 test |
| I6 | `compress_ms` cho mọi dòng nén | `scripts/evaluate.py` | — |
| J6 | `evaluate.py` ghi CBR từng ví dụ; `analyze.py --cbr-regression` hồi quy mức tăng theo CBR, CI bootstrap cụm | `viword/stats.py` | 3 test |
| J7 | DATN ghi cách tính p-value bootstrap và chiều kiểm định NFR | `DATN.md` | — |
| L1 | README: số test, cấu trúc, cách chạy; DATN §5: danh sách script; các tham chiếu "DATN §x" trong code đều trỏ tới mục có thật | — | — |

### Còn cần người làm hoặc cần quyền truy cập

- **I4:** Vistral là repo gated. Máy này không có quyền nên chưa kiểm tra được chat template. Trên máy có token
  HF được cấp quyền, chạy `scripts/check_reader_template.py` rồi đặt `READER_SYSTEM_PROMPT`. Thiếu quyền thì
  `diagnose` (tokenizer G5) cũng dừng ngay từ đầu.
- **G2:** chưa kiểm tay 100 từ để đo độ chính xác của heuristic từ láy, nên CBR theo loại từ chỉ báo cáo mô tả.
- **I5:** ngưỡng < 2% câu trả lời rỗng chỉ kiểm tra được sau khi có kết quả reader.
