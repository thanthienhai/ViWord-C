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
scripts/                 các bước chạy (xem bên dưới)
tests/                   test CPU, không cần GPU, mô hình hay Java
data/                    dữ liệu (không đưa lên git), xem data/README.md
```

## Requirements

Python ≥ 3.10. Mỗi nhóm phụ thuộc là một extra để chỉ cài phần cần dùng:

```bash
pip install -e ".[dev]"                      # lõi + test
pip install -e ".[seg,model,baselines]"      # tách từ, encoder, LLMLingua
pip install -e ".[eval,teacher]"             # reader vLLM (Linux + GPU), LLM thầy qua API
pytest -q                                    # 39 test, chạy trên CPU
```

VnCoreNLP cần Java ≥ 1.8; mô hình được tải tự động vào `models/vncorenlp/` ở lần chạy đầu. Nguồn tải là
raw.githubusercontent.com và có thể rất chậm; nếu vậy, dùng `--segmenter underthesea` để chạy thử. Trên Windows,
đặt `PYTHONIOENCODING=utf-8` để in được tiếng Việt ra console. vLLM không chạy trên Windows: khi đó
`scripts/evaluate.py --backend hf` dùng `transformers` (chậm hơn, chỉ nên dùng cho mô hình nhỏ và smoke test).

## Dữ liệu

Mỗi tác vụ được chuẩn hoá về một file JSONL (định dạng ở [`data/README.md`](data/README.md)), rồi tách từ **một
lần** và lưu lại:

```bash
python -c "from viword.data import belebele_from_hub; belebele_from_hub('data/tasks/belebele.jsonl')"
python scripts/segment_data.py --input data/tasks/belebele.jsonl --field context \
    --output data/segmented/belebele.jsonl
python scripts/segment_data.py --input data/tasks/vinli_test.jsonl --field premise \
    --output data/segmented/vinli_test.jsonl
```

## Tuần cổng: chẩn đoán (DATN §4.0)

Không cần huấn luyện. Đo CBR, tỉ lệ giữ hư từ, chi phí token và số premise có phủ định:

```bash
python scripts/diagnose.py --task vinli --data data/tasks/vinli_test.jsonl \
    --segmented data/segmented/vinli_test.jsonl --limit 500 \
    --budget-tokenizer Qwen/Qwen2.5-7B-Instruct \
    --cost-tokenizers Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct Viet-Mistral/Vistral-7B-Chat \
    --methods random lead truncation llmlingua2 \
        scored:model=microsoft/llmlingua-2-xlm-roberta-large-meetingbank,unit=syllable,name=llmlingua2_syl \
        scored:model=microsoft/llmlingua-2-xlm-roberta-large-meetingbank,unit=word,name=llmlingua2_wordpool \
    --rr-model xlm-roberta-large --out results/diagnose/vinli.json
```

Điểm tác vụ cho các cổng G1, G3, G4 lấy từ `scripts/evaluate.py` với `--limit 500` (xem phần Evaluation).

## Training

1. **Chưng cất nhãn từ LLM thầy** (§3.2). Thầy được gọi qua API tương thích OpenAI, ví dụ
   `vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --port 8000`. Kiểm tra thầy trước (cổng G4) bằng `--limit 200
   --save-compressions results/teacher_dev.jsonl`, rồi mới chạy toàn bộ:

   ```bash
   python scripts/segment_data.py --input data/distill/paragraphs.jsonl --field text \
       --output data/segmented/paragraphs.jsonl
   python scripts/distill.py --paragraphs data/distill/paragraphs.jsonl \
       --segmented data/segmented/paragraphs.jsonl --eval-files data/tasks/*.jsonl \
       --model Qwen/Qwen2.5-72B-Instruct-AWQ --output data/distill/distilled.jsonl
   ```

2. **Huấn luyện encoder** (§3.3), 3 seed cho mỗi loại nhãn:

   ```bash
   for s in 0 1 2; do
     python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit word     --out runs/viword_s$s          --seed $s
     python scripts/train.py --distilled data/distill/distilled.jsonl --label-unit syllable --out runs/llmlingua2vi_s$s --seed $s
   done
   ```

   Siêu tham số mặc định: XLM-R-large, AdamW lr 1e-5, 3 epoch, batch 8, cửa sổ 512 subword, BCE có trọng số lớp
   dương (1−p)/p, chọn checkpoint theo loss trên dev (500 đoạn tách theo id, cố định).

## Evaluation

Một lần chạy = một tác vụ × một reader. Bước nén và bước đọc chạy tách nhau để không tranh GPU, và bản nén được
lưu lại (`*.compressed.jsonl`) để tái sử dụng.

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
Cận trên từ thầy: `precomputed:path=results/teacher_dev.jsonl,name=teacher`.

## Ghi chú cài đặt (so với DATN.md)

- **Ngân sách:** tính bằng token của reader, chỉ trên phần ngữ cảnh. Mọi phương pháp đi qua cùng một hàm `render()`
  và bước `trim_to_budget`. LLMLingua/LLMLingua-2 được tìm nhị phân tham số `rate` để lọt ±3% ngân sách.
- **Nhãn cấp từ:** mỗi âm tiết nhận nhãn mềm của từ chứa nó. Logit của từ là trung bình logit các âm tiết; logit của
  âm tiết là trung bình logit các subword.
- **Selective Context:** tái hiện ở cấp từ (tổng surprisal các token của từ), không dùng package gốc vì package đó
  phụ thuộc spaCy tiếng Anh.
- **ROUGE:** tự cài trên âm tiết, vì `rouge_score` bỏ mọi ký tự ngoài ASCII. BERTScore **chưa** được cài đặt.
- **Danh sách stopword** (`viword/resources/stopwords_vi.txt`) là danh sách ngắn tự soạn. Cần thay bằng danh sách
  đã công bố trước khi báo cáo kết quả.
- **Bộ phát hiện P** chỉ dùng luật đơn giản. Độ chính xác của nó phải được đo trên ~200 câu gán nhãn tay (§3.4),
  không được giả định.

## Results

Chưa có. Bảng 1 (chẩn đoán, §4.0) và bảng chính (ablation §4.3) sẽ được điền kèm lệnh tái tạo chính xác.

## Citation & licence

Chưa có bài công bố. Licence: chưa chọn.
