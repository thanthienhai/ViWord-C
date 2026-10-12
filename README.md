# ViWord-C

Mã nguồn nghiên cứu cho đề tài *Nén ngữ cảnh có nhận thức ngôn ngữ cho LLM tiếng Việt*
(Word- and Tokenizer-Aware Prompt Compression for Vietnamese). Bộ nén trích xuất quyết định giữ/bỏ theo
**từ tiếng Việt** thay vì âm tiết, có **ưu tiên bảo vệ hư từ mang nghĩa** (phủ định, điều kiện, thời/thể, lượng
từ), và tính ngân sách bằng **token của LLM đích**.

- [`data/README.md`](data/README.md): nguồn, các bước lọc và kết quả rà soát dữ liệu (lỗi đã sửa, vấn đề còn tồn tại).
- [`DATN.md`](DATN.md): đề xuất nghiên cứu (vấn đề P1–P3, giả thuyết H1–H3, cổng quyết định G1–G5, thiết kế thực
  nghiệm). Các chú thích "DATN §x" trong mã nguồn trỏ tới tài liệu này.
- [`docs/LITERATURE_DATN.md`](docs/LITERATURE_DATN.md): tài liệu liên quan, theo nhóm và mức ưu tiên.
- [`docs/s2_search/`](docs/s2_search/): script tìm tài liệu trên Semantic Scholar (`search.py`, chạy lại được và
  tiếp tục từ chỗ dừng) cùng kết quả thô (`results.json`, `arxiv.xml`).

Cấu trúc repo theo [Good Research Code Handbook](https://goodresearch.dev) (một package cài được, cùng các
thư mục `scripts/`, `tests/`, `data/`); README theo [mẫu của Papers with Code](https://github.com/paperswithcode/releasing-research-code).

> Trạng thái: mã nguồn đã sẵn sàng cho tuần cổng §4.0, **chưa có kết quả thực nghiệm**. Bảng kết quả ở cuối file
> sẽ được điền sau khi chạy.

## Cấu trúc

```
viword/                  thư viện
  segment.py             tách từ (VnCoreNLP; underthesea/pyvi chỉ cho phân tích độ nhạy), mặt nạ âm tiết, render, căn chỉnh
  protect.py             tập bảo vệ P theo (từ, POS, ngữ cảnh), 3 tầng, luật khử nhập nhằng       §3.4
  select.py              chọn theo ngân sách: tham lam s/c^α + bước lấp, knapsack DP chính xác     §3.5
  model.py               encoder chấm điểm theo âm tiết, cửa sổ trượt theo câu, đối chứng tra từ    §3.1, §4.2
  compressor.py          giao diện chung compress(context, budget, counter); ViWord-C / 2×2          §2.3
  baselines.py           Lead-k, truncation, random, stopword, luật từ vựng, chọn câu, Selective
                         Context, LLMLingua(-2), LongLLMLingua, probe phủ định, precomputed          §4.2
  methods.py             tạo bộ nén từ chuỗi cấu hình ("scored:model=...,unit=word")
  distill.py             prompt LLM thầy, nhãn âm tiết + nhãn mềm cấp từ, kiểm tra rò rỉ           §3.2
  train.py               huấn luyện encoder (XLM-R), chọn checkpoint theo loss dev                  §3.3
  diagnose.py            CBR (+ mốc ngẫu nhiên), RR, tỉ lệ giữ theo tầng, fertility/CV_T            §2.2
  data.py, eval.py       dữ liệu chuẩn hoá, prompt reader, ROUGE cho tiếng Việt, reader vLLM        §4.1
  stats.py               bootstrap seed × cụm, Holm, kiểm định NFR, non-inferiority, MDE, hồi quy    §4.3
  segment_meta.py        sidecar provenance của cache tách từ                                         §3.1
scripts/                 prepare_data, leak_filter, segment_data, check_segmented, verify_segmentation,
                         teacher_compress, review_teacher, distill, train, checkpoint_check, diagnose,
                         compare_llmlingua2, gate_report, gate_eval, make_annotation_sheets,
                         evaluate_protect, evaluate_probe, select_pi, select_alpha, mde, greedy_gap,
                         check_reader_template, evaluate, bertscore, analyze; run_pipeline.sh nối tất cả
tests/                   test CPU, không cần GPU, mô hình hay Java
data/                    dữ liệu (không đưa lên git), xem data/README.md
```

## Requirements

Python ≥ 3.10. Mỗi nhóm phụ thuộc là một extra để chỉ cài phần cần dùng:

```bash
pip install -e ".[dev]"                      # lõi + test
pip install -e ".[seg,model,baselines]"      # tách từ, encoder, LLMLingua
pip install -e ".[eval,teacher]"             # reader vLLM (Linux + GPU), LLM thầy qua API
pytest -q                                    # 161 test, chạy trên CPU
```

VnCoreNLP cần Java ≥ 1.8; mô hình được tải tự động vào `models/vncorenlp/` ở lần chạy đầu. Nguồn tải là
raw.githubusercontent.com và có thể rất chậm. Pipeline chính chỉ chấp nhận VnCoreNLP (RDRSegmenter + POS + NER);
`scripts/segment_data.py --segmenter underthesea|pyvi` chỉ dành cho phân tích độ nhạy §4.4 và không được trộn
vào `data/segmented/` của pipeline chính. Mỗi lệnh tách từ ghi sidecar `<file>.jsonl.meta.json` với bộ tách từ,
annotator và SHA-256 của file JSONL. Trên Windows,
đặt `PYTHONIOENCODING=utf-8` để in được tiếng Việt ra console. vLLM không chạy trên Windows: khi đó
`scripts/evaluate.py --backend hf` dùng `transformers` (chậm hơn, chỉ nên dùng cho mô hình nhỏ và smoke test).

## Chạy toàn bộ pipeline

`scripts/run_pipeline.sh` chạy từng giai đoạn; mỗi giai đoạn chỉ đọc file do giai đoạn trước ghi, nên có thể chạy
các giai đoạn trên các máy khác nhau. Cấu hình (LLM thầy, reader…) đặt qua biến môi trường ở đầu file; bộ tách từ
của pipeline chính được cố định là VnCoreNLP.

```bash
bash scripts/run_pipeline.sh data segment annot_sheets   # rồi gán nhãn hai phiếu trong data/annot/ (người làm)
bash scripts/run_pipeline.sh protect_eval diagnose        # đo P/R trước khi chọn π; tuần cổng không cần reader
bash scripts/run_pipeline.sh train ckpt_check select_pi mde    # kiểm tra checkpoint, chọn π và tính MDE trên dev
bash scripts/run_pipeline.sh select_alpha                        # chỉ bản 8 trang: chọn α (H3) trên dev
bash scripts/run_pipeline.sh teacher_test                        # cận trên thầy trên 300 mẫu test/tác vụ
bash scripts/run_pipeline.sh eval bertscore analyze              # sau khi config + teacher upper bound đã sẵn sàng
READER=Viet-Mistral/Vistral-7B-Chat READER_SYSTEM_PROMPT="..." bash scripts/run_pipeline.sh eval  # reader thứ hai
bash scripts/run_pipeline.sh teacher_check           # thầy nén 200 mẫu dev mỗi tác vụ -> cận trên / cổng G4
bash scripts/run_pipeline.sh teacher_gate gate_eval  # cần reader: quyết định G4, rồi G1/G2/G3 trên tập dev
bash scripts/run_pipeline.sh teacher distill train   # chưng cất + 3 seed × 2 loại nhãn (cần GPU lớn)
READER_BACKEND=vllm bash scripts/run_pipeline.sh teacher_test ckpt_check eval analyze
```

`ckpt_check` là cổng bắt buộc giữa `train` và `eval`: trên đúng 200 mẫu của mỗi dev set cấu hình, ở tỉ lệ 1/3,
stage chạy reader cho mọi `viword_s*` (quyết định cấp từ) và `llmlingua2vi_s*` (cấp âm tiết), rồi so điểm chính với
`truncation` và baseline tra từ `lexical`. Mọi checkpoint phải thắng cả hai baseline; báo cáo ở
`results/ckpt_check/report.{md,json}` phải là `PASS`, nếu không `stage_eval` dừng.

`mde` chạy trước `eval` trên các split dev, dùng cùng ba seed và bootstrap phân tầng seed → cụm. Nó ghi
`results/mde_dev/report.{md,json}`: MDE và độ lệch chuẩn seed cho H1, NFR H2 trên ViNLI, cùng ROUGE-L H2 trên
VietNews. Lề non-inferiority 0,5 điểm ROUGE-L phải lớn hơn cả MDE và SD giữa seed; `stage_eval` gọi `--verify` và
dừng nếu report không PASS. Belebele hiện không có split dev trong nguồn chuẩn hoá, nên report ghi rõ H1 dev coverage
chỉ gồm ViMMRC và không thay bằng mẫu test.

`stage_analyze` không còn xuất kết quả từng seed riêng. Nó gọi
`analyze.py --seed-mode aggregate --seeds 0 1 2`, ghép các method hậu tố `_s0/_s1/_s2` theo cùng task, reader,
tỉ lệ và báo cáo trung bình cùng độ lệch chuẩn giữa ba seed trong `results/analysis_3seed.md`. Lượt H1 dùng
`viword` làm target cho các đối chứng chính và thiết kế 2×2. Lượt H2 gọi riêng
`--target viword_pi --reference viword llmlingua2vi`; trong đó `analyze.py` thực hiện McNemar trên NFR cho cả
`viword` và `llmlingua2vi_force`, cùng non-inferiority ROUGE-L trên VietNews. Kiểm định target–reference dùng
bootstrap phân tầng: lấy lại seed rồi lấy lại cụm trong seed đó; baseline không huấn luyện được dùng như tầng chung
cho cả ba seed. Nếu thiếu một seed, phân tích dừng thay vì kết luận từ tập chưa đủ.

LLM thầy mặc định là `icmodel/icom-model-llm-ic-v3.8-27b` tại `http://172.16.9.11:30048/v1` (vLLM, API tương thích
OpenAI). Thầy chỉ cần trả về văn bản; không dùng logit.

## Dữ liệu

`scripts/prepare_data.py` tải và chuẩn hoá mọi bộ dữ liệu về định dạng ở [`data/README.md`](data/README.md):

| Tác vụ | Nguồn | Ghi chú |
|---|---|---|
| Belebele | `facebook/belebele` (vie_Latn + eng_Latn) | 900 câu hỏi / 488 đoạn; ghép Việt–Anh theo (link, số câu hỏi) |
| ViNLI | `presencesw/vinli_4_label` (bản sao không chính thức) | test: 2.991 cặp nhưng chỉ 377 premise |
| ViMMRC | `ura-hcmut/ViMMRC` (bản chính thức `uitnlp/vimmrc2.0` bị gated) | đã loại câu hỏi về bài thơ; test chỉ 69 đoạn |
| VietNews | `nam194/vietnews` | bản gốc đã tách từ (`Cơ_quan`), được trả về văn bản thường; lấy mẫu 1.000 bài |
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` | QA trích xuất; validation làm test (2.653 câu có đáp án), F1 |
| GSM8K-vi | `hllj/vi_gsm8k` (dịch máy) | nén 8 ví dụ có lời giải; đã loại 12 câu test lỗi dịch; báo cáo riêng nhóm "MT" |
| OneRuler-vi | `starbix/oneruler`, `vi_4096` | ngữ cảnh dài khoảng 4.300 token, 7 tác vụ con × 100 mẫu |
| Đoạn chưng cất | VietNews train + Wikipedia vi (20231101) | 300–1.500 âm tiết, trọn câu; 5.000 đoạn mỗi nguồn |

Mỗi file được tách từ **một lần** (`scripts/segment_data.py`); các văn bản trùng nhau (premise ViNLI) chỉ tách một lần.

## Tuần cổng: chẩn đoán (DATN §4.0)

`diagnose.py --unique-clusters` đếm mỗi tài liệu nguồn một lần. Kết quả ghi vào `results/diagnose/*.json`;
`scripts/gate_report.py` đối chiếu với ngưỡng các cổng G1 (phần CBR), G2 (phần đếm) và G5. Phần cần reader của các cổng
chạy trên **tập dev**, không dùng tập test: `teacher_gate` (G4, thầy so với truncation trên đúng 200 mẫu dev thầy
đã nén) và `gate_eval` (G1 phần điểm, G2 phần NFR, G3; `GATE_LIMIT=500`). Cả hai gọi `scripts/gate_eval.py` và ghi
`results/gate/*.md`. Trong `diagnose`, `scripts/compare_llmlingua2.py` đối chiếu package gốc với arm `scored:` trên
100 tài liệu ở ngân sách 1/3; `gate_report.py` tự chuyển G1 sang package nếu mặt nạ trùng dưới 95% hoặc CBR lệch
ít nhất 1 điểm %. Chỉ chưng cất khi G4 đạt.

Mốc CBR ngẫu nhiên trong `results/diagnose/*.json` dùng đúng phân phối độ dài các từ nhiều âm tiết của corpus:
`chance_cbr` là tỉ số kỳ vọng sau khi cộng xác suất partial/survived của từng độ dài; `chance_cbr_2syl` chỉ giữ
lại để đối chiếu lịch sử.

G1 có hai phép đo CBR ở tỉ lệ 1/3: vế đầu dùng `llmlingua2_syl`/package gốc trong chẩn đoán tuần đầu; sau khi
đã huấn luyện, chạy lại `bash scripts/run_pipeline.sh diagnose`. Stage này tự phát hiện mọi
`runs/llmlingua2vi_s*`, đo CBR từng seed và ghi CBR gộp của `llmlingua2vi_s*` vào `results/diagnose/gate_report.md`;
vế sau đạt khi CBR gộp ≥ 5%.

G2 cố ý dùng hai nguồn khác nhau: NFR của `probe_negation` được tính bởi `gate_eval.py --split dev` trên
`data/tasks/vinli_dev.jsonl`; script từ chối `--split test` để không dùng kết quả reader trên test. Vế đếm ≥300 cặp
được `gate_report.py` tính trên `vinli_test`, nhưng chỉ dùng văn bản premise, segmentation và `id`/`cluster` (chiếu bỏ
trường `label`). Do đó không có nhãn test hay dự đoán reader test nào đi vào NFR hoặc vào việc chọn cấu hình.

RR dùng `xlm-roberta-large` (hoặc giá trị của `RR_MODEL`) và không còn ép mỗi âm tiết bị xoá vào một mask duy
nhất: `diagnose.py` chèn số `<mask>` bằng số subword của tokenizer, rồi ghép/giải mã các dự đoán top-1 theo từng
âm tiết. Do đó âm tiết tách thành hai hay nhiều subword vẫn được đánh giá đúng khả năng khôi phục; có test hồi quy
cho trường hợp hai subword trong `tests/test_diagnose.py`.

## Training

1. **Chưng cất nhãn từ LLM thầy** (§3.2), gồm 3 bước:
   - `leak_filter.py`: bắt buộc index đủ ViNLI/VietNews/ViMMRC/Belebele cùng các split dev/test cần thiết;
     bỏ đoạn trùng n-gram 13 token với tập đánh giá, **trước** khi gọi API (thêm một bộ lọc bảo thủ theo GUID
     VietNews; GUID chỉ là số thứ tự trong split nên không phải kiểm tra rò rỉ). File
     thiếu hoặc rỗng làm pipeline dừng; log ghi coverage và số đoạn bị loại theo nguồn/lý do;
   - `teacher_compress.py`: gọi thầy song song, chạy lại được khi bị ngắt;
   - `distill.py`: căn chỉnh bằng LCS chính xác trên token chữ/số/dấu câu; bỏ đầu ra viết lại/đổi thứ tự
     (`unmatched_ratio > 10%`) hoặc không nén (`keep_rate > 90%`). Đây là ngưỡng cố định riêng của ViWord-C,
     khác bộ lọc theo phân vị 90% của `variation_rate` và `alignment_gap` trong LLMLingua-2 gốc.

    Trên 17.767 đầu ra còn thuộc kho sạch: 16.274 mẫu được giữ (91,6%), 703 bị loại vì viết lại/đổi thứ tự và 790
    vì không nén. Đo theo từ của VnCoreNLP (tách lại ngày 12/10/2026), CBR trung bình theo mẫu là 6,29%; CBR gộp
    toàn corpus là 4,90% (52.370/1.068.838 từ nhiều âm tiết còn giữ). Các con số 8,14% / 6,58% công bố trước đó đo
    theo từ của underthesea (xem data/README.md, "Provenance tách từ"). `distill.py` báo cáo cả hai; mẫu giữ một phần được giữ dưới dạng nhãn mềm theo âm tiết
    (nên dùng nhãn mềm).
    Cổng G4 đạt ở 2/3 tác vụ (smoke test 04/10), nhưng thầy **thua truncation trên VietNews**
    (−1,25 điểm ROUGE-L), là miền chiếm một nửa kho chưng cất.

2. **Huấn luyện encoder** (§3.3), 3 seed cho mỗi loại nhãn (giai đoạn `train`). Siêu tham số mặc định: XLM-R-large,
   AdamW lr 1e-5, 3 epoch, batch 8, cửa sổ 512 subword, BCE cấp âm tiết có trọng số lớp dương (1−p)/p. Với
   `label_unit=word`, target mềm của từ được chép cho từng âm tiết; checkpoint chọn theo loss trên dev (500 đoạn
   tách theo id, cố định). Cần GPU ≥ 24 GB.

## Evaluation

Một lần chạy = một tác vụ × một reader. Bước nén và bước đọc chạy tách nhau để không tranh GPU, và bản nén được
lưu lại (`*.compressed.jsonl`) để tái sử dụng. Reader có ba backend: `vllm` (mặc định), `api` (server tương thích
OpenAI, thêm `--base-url` và `--tokenizer`), `hf` (transformers, cho smoke test). Không dùng LLM thầy làm reader.

```bash
python scripts/evaluate.py --task belebele --data data/tasks/belebele.jsonl \
    --segmented data/segmented/belebele.jsonl --reader Qwen/Qwen2.5-7B-Instruct \
    --methods none no_context lead truncation llmlingua2 translate:inner=llmlingua2 \
        scored:model=runs/llmlingua2vi_s0,unit=syllable,name=llmlingua2vi_s0 \
        scored:model=runs/llmlingua2vi_s0,unit=word,name=llmlingua2vi_wordpool_s0 \
        scored:model=runs/viword_s0,unit=word,name=viword_s0 \
        scored:model=runs/viword_s0,unit=word,protect=soft,delta=0.2,name=viword_pi_s0 \
    --out results/eval/belebele_qwen7b.jsonl

python scripts/analyze.py --rows results/eval/*.jsonl --target viword \
    --reference llmlingua2vi truncation lead tfidf_sent \
    --seed-mode aggregate --seeds 0 1 2 \
    --primary-reader Qwen/Qwen2.5-7B-Instruct \
    --holm-family 'H1:tasks=vimmrc+belebele,ratios=0.5+0.333+0.2,reader=primary' \
    --factorial llmlingua2vi llmlingua2vi_wordpool viword_syl viword
```

`--holm-family` khai báo họ giả thuyết theo cú pháp
`NAME:tasks=task+task,ratios=r+r,reader=primary|all|READER`; có thể lặp tham số này cho H1/H2/H3.
Mặc định H1 chỉ gồm ViMMRC/Belebele × `{1/2, 1/3, 1/5}` trên reader khai báo bằng `--primary-reader`.
P-value thô của các ô ngoài họ vẫn được in, nhưng cột Holm để `—` và không đưa các ô đó vào hiệu chỉnh.
Luôn đưa cả truncation, lead và baseline cấp câu vào làm reference, vì ở LACC các baseline đơn giản này hoà hoặc
thắng bộ nén đã học. `--factorial` tính hiệu ứng chính (đơn vị quyết định, loại nhãn) và tương tác của thiết kế 2×2
kèm khoảng tin cậy, thay vì đọc bằng mắt bốn giá trị trung bình. Với ViNLI, NFR được báo thêm trên riêng các premise
có phủ định: tính trên mọi cặp thì tác động của phủ định bị pha loãng, vì chỉ khoảng 14% số cặp có phủ định.

Bootstrap trong `stats.py` lấy lại ID cụm có hoàn lại, giữ toàn bộ các dòng của cụm và chia cho tổng số dòng được
chọn; vì vậy cụm được lấy làm đơn vị phụ thuộc nhưng vẫn có trọng số theo kích thước cụm. Khóa cụm của dữ liệu chuẩn
hóa là: ViNLI = premise, ViMMRC = đoạn, Belebele = `link`, VietNews = bài (`guid`). Test hồi quy đọc đúng các
trường `task/id/cluster/method/ratio/accuracy` mà `evaluate.py` ghi ra và kiểm tra Belebele có 488 cụm, ViNLI test có
377 premise khi dữ liệu đã chuẩn bị.

Chạy full trên H100: [`docs/RUN_FULL_H100.md`](docs/RUN_FULL_H100.md) (`scripts/full_h100.sh`, hai pha).

Các ô của thiết kế 2×2 (§2.3) chỉ khác nhau ở checkpoint và `unit`:

| | quyết định cấp âm tiết | quyết định cấp từ |
|---|---|---|
| nhãn âm tiết | `runs/llmlingua2vi_s*`, `unit=syllable` | `runs/llmlingua2vi_s*`, `unit=word` |
| nhãn từ (mềm) | `runs/viword_s*`, `unit=syllable` | `runs/viword_s*`, `unit=word` (ViWord-C) |

Ánh xạ đầy đủ từng hàng ablation §4.3 sang spec trong `stage_eval` (mỗi `s` chạy với seed 0, 1, 2):

| Hàng §4.3 | Spec trong `stage_eval` |
|---|---|
| LLMLingua-2-vi | `scored:model=runs/llmlingua2vi_s$s,unit=syllable,name=llmlingua2vi_s$s` |
| LLMLingua-2-vi + gộp logit theo từ | `scored:model=runs/llmlingua2vi_s$s,unit=word,name=llmlingua2vi_wordpool_s$s` |
| Nhãn từ, quyết định âm tiết | `scored:model=runs/viword_s$s,unit=syllable,name=viword_syl_s$s` |
| ViWord-C, π = none | `scored:model=runs/viword_s$s,unit=word,name=viword_s$s` |
| LLMLingua-2-vi + `force_tokens` | `scored:model=runs/llmlingua2vi_s$s,unit=syllable,protect=force,tiers=$PI_TIERS,name=llmlingua2vi_force_s$s`; ablation `protect=force_ctx`, `name=llmlingua2vi_forcectx_s$s` |
| ViWord-C + π (cấu hình chính) | `scored:model=runs/viword_s$s,unit=word,protect=$PI_MODE,delta=$PI_DELTA,tiers=$PI_TIERS,name=viword_pi_s$s`; các biến lấy từ `config/pi_selected.env` do D7 chọn trên dev |
| + tokenizer-aware | Chưa có spec trong `stage_eval`; để dành cho bản 8 trang (`alpha=α*`) |
| + chi phí = số âm tiết | Chưa có spec trong `stage_eval`; để dành cho bản 8 trang (`cost=syllables`) |

Các biến thể khác, cũng qua `scored:`: `protect=soft|hard` và `tiers=T1+T2`; `alpha=0.5` và `cost=syllables`
(H3 và đối chứng độ dài); `exact=1` (knapsack DP). Đối chứng tra từ:
`lexprior:train=data/distill/distilled.jsonl,exclude=runs/viword_s0/dev_ids.jsonl` (fit chỉ trên phần train của
encoder; dev split cố định nên mọi seed dùng chung `dev_ids.jsonl`).
Cận trên G4/dev: `precomputed:path=results/teacher_dev/vinli_dev.jsonl,trim=1,name=teacher` (`trim=1` cắt đầu ra
thầy về đúng ngân sách như mọi phương pháp khác). Cận trên test được sinh riêng bởi stage `teacher_test` bên dưới
để giữ đúng báo cáo tỉ lệ nén thực tế.

`stage_eval` cũng chạy đủ các baseline §4.2: `stopword`, `lexical`, `ppl_sent`, LLMLingua theo perplexity
(`llmlingua:lm=$LM`), LLMLingua-2 package gốc (`llmlingua2`), LLMLingua-2-vi/ViWord-C và random với seed 0, 1,
2. LongLLMLingua chỉ chạy trên ViMMRC/Belebele vì cần câu hỏi. Cận trên thầy trên tập test được sinh bởi
`teacher_test` (mặc định 300 mẫu/tác vụ), đưa vào eval qua `precomputed:path=results/teacher_test/.../name=teacher_upper`;
stage này chạy cận trên trong một lượt riêng 300 mẫu/tác vụ để không yêu cầu file teacher chứa toàn bộ test. Đây là
tham chiếu mô tả, không dùng để chọn π.

## Ghi chú cài đặt (so với DATN.md)

- **Ngân sách:** với mọi phương pháp, `B = round(ratio * n_tokens_context)`, trong đó `n_tokens_context` là số
  token của phần ngữ cảnh gốc theo tokenizer của reader; chỉ dẫn, câu hỏi và lựa chọn không được đếm. `translate`
  cũng dùng cùng B của context gốc (bản `context_en` chỉ là đầu vào nén). Mỗi dòng lưu
  `n_tokens_compressed / budget`; `analyze.py` báo cáo theo phương pháp và đánh dấu các cặp lệch quá 5%. Các phương
  pháp mask-based trong ma trận H1 đi qua cùng một hàm `render()` và bước `trim_to_budget`. LLMLingua/LLMLingua-2
  được tìm nhị phân tham số `rate` để độ dài nằm trong [−3%, 0] của ngân sách (không bao giờ vượt B).
- **Thống kê của bước chọn:** mỗi dòng của `ScoredCompressor` ghi trường `selection` (Σ c_T, độ dài thật trước bước
  cắt và độ lệch, số đơn vị bị cắt, số token dành cho P), `compress_ms` (độ trễ nén) và `cbr` (CBR của chính ví dụ,
  dùng cho `analyze.py --cbr-regression`). Với `hard`, bước cắt bỏ các từ ngoài P trước.
- **Gộp điểm cố định:** logit của âm tiết là trung bình số học logit các subword của nó; logit của từ là trung bình
  số học logit các âm tiết của nó. Đây là gộp hai tầng subword → âm tiết → từ, nên các âm tiết có trọng số như nhau
  trong một từ; không có biến thể `pool=` hoặc lựa chọn pooling trên dev.
- **Ghép đầu ra H1:** bốn ô 2×2 đều chọn mặt nạ âm tiết và gọi chung `viword.segment.render()`, nên dấu câu và
  khoảng trắng không phụ thuộc ô. LLMLingua-2 package, `translate` và `precomputed` không dùng renderer vì là
  đầu ra text/external ngoài H1; `none` và `no_context` cũng là control text-only.
- Với mọi phương pháp `scored:...,unit=word` (gồm các ô 2×2 và LLMLingua-2 + word pooling), code gộp logit trước
  rồi mới áp dụng sigmoid: `sigmoid(mean(logit))`, không phải trung bình các xác suất âm tiết.
- **Nhãn cấp từ:** mỗi âm tiết nhận nhãn mềm của từ chứa nó.
- **Cửa sổ trượt:** câu dài hơn 200 âm tiết được cắt thành đoạn 200 âm tiết trước khi ghép cửa sổ 512 subword,
  nên ranh giới cửa sổ có thể rơi giữa câu; mọi âm tiết vẫn được chấm điểm.
- **Selective Context:** tái hiện ở cấp từ (tổng surprisal các token của từ), không dùng package gốc vì package đó
  phụ thuộc spaCy tiếng Anh.
- **Baseline cấp câu** (`tfidf_sent`, `ppl_sent`): chọn trọn câu theo điểm, phần ngân sách còn lại lấp bằng phần đầu
  của câu tốt nhất chưa chọn. Nếu không lấp, premise ViNLI (thường một câu dài hơn ngân sách) bị nén thành rỗng.
  `ppl_sent` **không** giống hệt `lacc_sentence` của LACC: LACC dùng câu hỏi, luôn giữ câu đầu và cuối, và được vượt
  ngân sách (chi tiết ở DATN §4.2).
- **ROUGE:** tự cài trên âm tiết, vì `rouge_score` bỏ mọi ký tự ngoài ASCII. **BERTScore** (PhoBERT-base, lớp 10,
  văn bản tách từ bằng VnCoreNLP) chạy sau `eval` bằng stage `bertscore`, ghi vào `results/bertscore/`.
- **Reader thứ hai:** cổng, `ckpt_check`, π, α và MDE chỉ dùng `PRIMARY_READER` (mặc định Qwen2.5-7B); các stage đó
  dừng nếu `READER` khác. Vistral là repo gated; kiểm tra template bằng `scripts/check_reader_template.py` và đặt
  `READER_SYSTEM_PROMPT` theo thẻ mô hình. `analyze.py` đánh dấu ⚠ khi > 2% câu trả lời không đọc được.
- **ViNLI:** giữ đủ 4 nhãn (gồm `other`) như bộ dữ liệu gốc.
- **Tác vụ khám phá** (ViQuAD, GSM8K-vi, OneRuler-vi): chỉ báo cáo mô tả, không vào họ Holm hay cổng.
- **Danh sách stopword** (`viword/resources/stopwords_vi.txt`) là danh sách ngắn tự soạn. Cần thay bằng danh sách
  đã công bố trước khi báo cáo kết quả.
- **Bộ phát hiện P và probe phủ định (G6): cần người gán nhãn.** `bash scripts/run_pipeline.sh annot_sheets` sinh
  hai phiếu mù `data/annot/protect_sheet.jsonl` (200 câu dev lấy mẫu đều, chỉ có từ/POS/NER, không có dự đoán của bộ
  phát hiện) và `data/annot/probe_sheet.jsonl` (50 premise ViNLI probe sửa + 50 premise không sửa, trộn lẫn). Người
  gán nhãn điền `gold_tiers` / `gold_negation` và `annotator` (hướng dẫn ở data/README.md), rồi chạy stage
  `protect_eval`. `diagnose`, `select_pi` và `eval` dừng cho tới khi có báo cáo từ phiếu mù. Bộ nhãn cũ trùng
  200/200 với đầu ra của chính bộ phát hiện, đã chuyển sang `data/annot/invalid_detector_prefilled/`.
- **Đối chứng H2:** hàng chính thức là `llmlingua2vi_force_s*`: checkpoint LLMLingua-2-vi ở cấp âm tiết với
  `protect=force`, tức ngữ nghĩa `force_tokens`: khớp chuỗi danh sách P (kể cả mục nhiều âm tiết) cộng chữ số,
  không khử nhập nhằng, không NER, cùng tầng với cấu hình π. Ablation `llmlingua2vi_forcectx_s*` dùng P đã khử
  nhập nhằng và NER ở cấp âm tiết.
  `llmlingua2_force` là baseline package LLMLingua-2 riêng để chẩn đoán; nó cũng nhận P động qua `force_tokens`,
  nhưng không phải đối chứng H2 chính vì dùng checkpoint và cơ chế chọn khác. Do API package nhận chuỗi thay vì
  mask theo vị trí, mặt chữ lặp với POS khác nhau có thể bị ép ở cả hai lần xuất hiện; đây là sai khác đã biết.
- **Chọn π trên dev:** `bash scripts/run_pipeline.sh select_pi` quét `T1`, `T1+T2`, `T1+T2+T3` ×
  `soft|hard` × δ (`0.1, 0.2, 0.3, 0.5`; `hard` dùng δ=0), so ROUGE-L VietNews với `viword_s*` (π=`none`)
  và NFR ViNLI với `none` không nén. Chỉ cấu hình mất không quá 0.5 điểm ROUGE-L được xét; cấu hình có NFR thấp
  nhất được ghi vào `config/pi_selected.json` và `config/pi_selected.env`, cùng bảng tại `results/pi_selection.md`.
  `stage_eval` dừng nếu các file chọn này chưa tồn tại.

## Results

Chưa có. Bảng 1 (chẩn đoán, §4.0) và bảng chính (ablation §4.3) sẽ được điền kèm lệnh tái tạo chính xác.

## Citation & licence

Chưa có bài công bố. Licence: chưa chọn.
