# Chạy smoke test ViWord-C trên máy GPU (Ubuntu)

Mục đích: kiểm tra toàn bộ đường chạy huấn luyện + đánh giá trên một GPU lớn và xem trước xu hướng. Quy mô nhỏ
(1 seed, 1 epoch, ~1.500 đoạn, 200 câu Belebele, 300 cặp ViNLI). **Số liệu không dùng cho bài báo.**

## Yêu cầu

- Ubuntu, Python ≥ 3.10, 1 GPU ≥ 40 GB (H100/A100), có hỗ trợ bf16.
- Tài khoản Hugging Face có quyền đọc repo **private** `thanthienhai/viword-c-data` (nhờ chủ repo mời vào).
- Khoảng 60 GB đĩa trống (môi trường, mô hình XLM-R-large, Qwen2.5-7B, Qwen2.5-1.5B, dữ liệu).

## 1. Lấy code và đăng nhập Hugging Face

```bash
git clone https://github.com/thanthienhai/ViWord-C.git
cd ViWord-C
huggingface-cli login        # hoặc: export HF_TOKEN=hf_xxx
```

## 2. Chạy smoke test

```bash
HF_DATA_REPO=thanthienhai/viword-c-data bash scripts/smoke_h100.sh
```

Lệnh trên tự làm lần lượt các bước:

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| `setup` | tạo `.venv`, cài torch / transformers / vLLM, chạy test, in thông tin GPU | 5–10 phút |
| `check_data` | tải dữ liệu từ HF về `data/` và `results/teacher_dev/`, kiểm tra đủ file | 1–3 phút |
| `distill` | dựng nhãn từ đầu ra của thầy, lấy tập con 1.500 đoạn | vài phút |
| `train` | huấn luyện 2 encoder XLM-R-large (nhãn từ, nhãn âm tiết), 1 epoch | 15–20 phút |
| `eval` | nén + đọc bằng Qwen2.5-7B (vLLM) trên Belebele và ViNLI | 20–30 phút |
| `analyze` | bảng kết quả và kiểm định | < 1 phút |

Chạy lại từng bước riêng nếu cần (môi trường `.venv` đã có):

```bash
HF_DATA_REPO=thanthienhai/viword-c-data bash scripts/smoke_h100.sh check_data
bash scripts/smoke_h100.sh train eval analyze
```

Tuỳ chỉnh qua biến môi trường: `GPU=1` (chọn GPU), `N_PARAGRAPHS=3000`, `READER=<model HF>`, `VENV=<đường dẫn>`.

## 3. Gửi lại kết quả

Nén và gửi lại toàn bộ thư mục `results/smoke/`:

```bash
tar czf smoke_results.tgz results/smoke
```

Thư mục gồm `smoke.log`, `train_*.log`, `eval_*.log`, `distill_summary.json`, `analysis.md` và `*_rows.jsonl`.

## Khi gặp lỗi

- **Thiếu file dữ liệu / 401 khi tải**: kiểm tra `huggingface-cli whoami` và quyền đọc repo `viword-c-data`.
- **Hết bộ nhớ GPU ở bước `eval`**: thêm `--gpu-memory-utilization` thấp hơn trong `viword/eval.py` (lớp `VLLMReader`)
  hoặc dùng GPU khác bằng `GPU=<id>`.
- **Lỗi khác**: dừng lại, gửi `results/smoke/smoke.log` cùng file log của bước bị lỗi; không sửa code tại chỗ.
