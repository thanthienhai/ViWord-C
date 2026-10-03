# ViWord-C

Mã nguồn nghiên cứu cho đề tài *Nén ngữ cảnh có nhận thức ngôn ngữ cho LLM tiếng Việt*
(Word- and Tokenizer-Aware Prompt Compression for Vietnamese). Bộ nén trích xuất quyết định giữ/bỏ theo
**từ tiếng Việt** thay vì âm tiết, có **ưu tiên bảo vệ hư từ mang nghĩa** (phủ định, điều kiện, thời/thể, lượng
từ), và tính ngân sách bằng **token của LLM đích**.

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
  segment.py             tách từ (VnCoreNLP / underthesea / pyvi), mặt nạ âm tiết, render, căn chỉnh
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
  stats.py               bootstrap theo cụm, Holm, McNemar, non-inferiority, MDE                    §4.3
scripts/                 prepare_data, leak_filter, segment_data, teacher_compress, distill, train,
                         diagnose, gate_report, evaluate, analyze; run_pipeline.sh nối tất cả
tests/                   test CPU, không cần GPU, mô hình hay Java
data/                    dữ liệu (không đưa lên git), xem data/README.md
```

## Requirements

Python ≥ 3.10. Mỗi nhóm phụ thuộc là một extra để chỉ cài phần cần dùng:

```bash
pip install -e ".[dev]"                      # lõi + test
pip install -e ".[seg,model,baselines]"      # tách từ, encoder, LLMLingua
pip install -e ".[eval,teacher]"             # reader vLLM (Linux + GPU), LLM thầy qua API
pytest -q                                    # 42 test, chạy trên CPU
```

VnCoreNLP cần Java ≥ 1.8; mô hình được tải tự động vào `models/vncorenlp/` ở lần chạy đầu. Nguồn tải là
raw.githubusercontent.com và có thể rất chậm; nếu vậy, dùng `--segmenter underthesea` để chạy thử. Trên Windows,
đặt `PYTHONIOENCODING=utf-8` để in được tiếng Việt ra console. vLLM không chạy trên Windows: khi đó
`scripts/evaluate.py --backend hf` dùng `transformers` (chậm hơn, chỉ nên dùng cho mô hình nhỏ và smoke test).

## Chạy toàn bộ pipeline

`scripts/run_pipeline.sh` chạy từng giai đoạn; mỗi giai đoạn chỉ đọc file do giai đoạn trước ghi, nên có thể chạy
các giai đoạn trên các máy khác nhau. Cấu hình (LLM thầy, bộ tách từ, reader…) đặt qua biến môi trường ở đầu file.

```bash
bash scripts/run_pipeline.sh data segment diagnose   # tuần cổng, không cần reader (G1 phần CBR, G2 phần đếm, G5)
bash scripts/run_pipeline.sh teacher_check           # thầy nén 200 mẫu dev mỗi tác vụ -> cận trên / cổng G4
bash scripts/run_pipeline.sh teacher_gate gate_eval  # cần reader: quyết định G4, rồi G1/G2/G3 trên tập dev
bash scripts/run_pipeline.sh teacher distill train   # chưng cất + 3 seed × 2 loại nhãn (cần GPU lớn)
READER_BACKEND=vllm bash scripts/run_pipeline.sh eval analyze
```

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
| Đoạn chưng cất | VietNews train + Wikipedia vi (20231101) | 300–1.500 âm tiết, trọn câu; 5.000 đoạn mỗi nguồn |

Mỗi file được tách từ **một lần** (`scripts/segment_data.py`); các văn bản trùng nhau (premise ViNLI) chỉ tách một lần.

## Tuần cổng: chẩn đoán (DATN §4.0)

`diagnose.py --unique-clusters` đếm mỗi tài liệu nguồn một lần. Kết quả ghi vào `results/diagnose/*.json`;
`scripts/gate_report.py` đối chiếu với ngưỡng các cổng G1 (phần CBR), G2 (phần đếm) và G5. Phần cần reader của các cổng
chạy trên **tập dev**, không dùng tập test: `teacher_gate` (G4, thầy so với truncation trên đúng 200 mẫu dev thầy
đã nén) và `gate_eval` (G1 phần điểm, G2 phần NFR, G3; `GATE_LIMIT=500`). Cả hai gọi `scripts/gate_eval.py` và ghi
`results/gate/*.md`. Chỉ chưng cất khi G4 đạt.

## Training

1. **Chưng cất nhãn từ LLM thầy** (§3.2), gồm 3 bước:
   - `leak_filter.py`: bỏ đoạn trùng n-gram với tập đánh giá, **trước** khi gọi API;
   - `teacher_compress.py`: gọi thầy song song, chạy lại được khi bị ngắt;
   - `distill.py`: căn chỉnh và dựng nhãn; bỏ đầu ra viết lại (> 10% chữ không có trong nguồn) hoặc không nén
     (giữ > 90%).

   Smoke test 19 đoạn: 79% đầu ra hợp lệ; chính thầy cắt đôi ~9% từ ghép (nên dùng nhãn mềm).

2. **Huấn luyện encoder** (§3.3), 3 seed cho mỗi loại nhãn (giai đoạn `train`). Siêu tham số mặc định: XLM-R-large,
   AdamW lr 1e-5, 3 epoch, batch 8, cửa sổ 512 subword, BCE có trọng số lớp dương (1−p)/p, chọn checkpoint theo loss
   trên dev (500 đoạn tách theo id, cố định). Cần GPU ≥ 24 GB.

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

python scripts/analyze.py --rows results/eval/*.jsonl --target viword_s0 --reference llmlingua2vi_s0
```

Các ô của thiết kế 2×2 (§2.3) chỉ khác nhau ở checkpoint và `unit`:

| | quyết định cấp âm tiết | quyết định cấp từ |
|---|---|---|
| nhãn âm tiết | `runs/llmlingua2vi_s*`, `unit=syllable` | `runs/llmlingua2vi_s*`, `unit=word` |
| nhãn từ (mềm) | `runs/viword_s*`, `unit=syllable` | `runs/viword_s*`, `unit=word` (ViWord-C) |

Các biến thể khác, cũng qua `scored:`: `protect=soft|hard` và `tiers=T1+T2`; `alpha=0.5` và `cost=syllables`
(H3 và đối chứng độ dài); `exact=1` (knapsack DP). Đối chứng tra từ: `lexprior:train=data/distill/distilled.jsonl`.
Cận trên từ thầy: `precomputed:path=results/teacher_dev/vinli_dev.jsonl,trim=1,name=teacher` (`trim=1` cắt đầu ra
thầy về đúng ngân sách như mọi phương pháp khác).

## Ghi chú cài đặt (so với DATN.md)

- **Ngân sách:** tính bằng token của reader, chỉ trên phần ngữ cảnh. Mọi phương pháp đi qua cùng một hàm `render()`
  và bước `trim_to_budget`. LLMLingua/LLMLingua-2 được tìm nhị phân tham số `rate` để lọt ±3% ngân sách.
- **Nhãn cấp từ:** mỗi âm tiết nhận nhãn mềm của từ chứa nó. Logit của từ là trung bình logit các âm tiết; logit của
  âm tiết là trung bình logit các subword.
- **Selective Context:** tái hiện ở cấp từ (tổng surprisal các token của từ), không dùng package gốc vì package đó
  phụ thuộc spaCy tiếng Anh.
- **Baseline cấp câu** (`tfidf_sent`, `ppl_sent`): chọn trọn câu theo điểm, phần ngân sách còn lại lấp bằng phần đầu
  của câu tốt nhất chưa chọn. Nếu không lấp, premise ViNLI (thường một câu dài hơn ngân sách) bị nén thành rỗng.
- **ROUGE:** tự cài trên âm tiết, vì `rouge_score` bỏ mọi ký tự ngoài ASCII. BERTScore **chưa** được cài đặt.
- **Danh sách stopword** (`viword/resources/stopwords_vi.txt`) là danh sách ngắn tự soạn. Cần thay bằng danh sách
  đã công bố trước khi báo cáo kết quả.
- **Bộ phát hiện P** chỉ dùng luật đơn giản. Độ chính xác của nó phải được đo trên ~200 câu gán nhãn tay (§3.4),
  không được giả định.

## Results

Chưa có. Bảng 1 (chẩn đoán, §4.0) và bảng chính (ablation §4.3) sẽ được điền kèm lệnh tái tạo chính xác.

## Citation & licence

Chưa có bài công bố. Licence: chưa chọn.
