# Hướng dẫn chạy smoke test ViWord-C trên máy GPU (dành cho đội vận hành)

**Mục đích:** chạy thử một lần toàn bộ đường chạy (kiểm tra thầy → dựng nhãn → huấn luyện → đánh giá) trên một GPU
lớn, để chắc mã nguồn chạy được và xem trước xu hướng. Quy mô nhỏ: 1 seed, 1 epoch, khoảng 1.500 đoạn huấn luyện,
200 câu Belebele, 300 cặp ViNLI. **Số liệu không dùng cho bài báo.**

**Tóm tắt:** clone code → đăng nhập Hugging Face → chạy **một lệnh** trong `tmux` (khoảng 1–1,5 giờ) → nén thư mục
`results/smoke/` và gửi lại. Không cần sửa code hay cấu hình gì.

## Yêu cầu

| Hạng mục | Yêu cầu |
|---|---|
| Hệ điều hành | Ubuntu, Python ≥ 3.10 (có `python3 -m venv`), `git`, `tmux` |
| GPU | 1 GPU ≥ 40 GB, hỗ trợ bf16 (H100/A100); driver NVIDIA hỗ trợ CUDA 12 (`nvidia-smi` chạy được) |
| Đĩa | khoảng 60 GB trống (môi trường Python, các mô hình XLM-R-large, Qwen2.5-7B, Qwen2.5-1.5B, dữ liệu) |
| Mạng | truy cập được github.com, huggingface.co và pypi.org |
| Tài khoản | tài khoản Hugging Face có quyền đọc repo **private** `thanthienhai/viword-c-data` (nhờ chủ repo mời vào) |

Dữ liệu được tải tự động và cố định ở một phiên bản (revision `ee91d9e`), nên mọi lần chạy dùng cùng một bộ dữ liệu.

## 1. Lấy code và đăng nhập Hugging Face

```bash
git clone https://github.com/thanthienhai/ViWord-C.git
cd ViWord-C
huggingface-cli login        # dán token có quyền read; hoặc: export HF_TOKEN=hf_xxx
huggingface-cli whoami       # phải in ra tên tài khoản
nvidia-smi                   # xem GPU nào đang rảnh
```

Nếu máy chưa có `huggingface-cli`, cài bằng `pip install -U huggingface_hub`, hoặc bỏ qua bước login và dùng
`export HF_TOKEN=hf_xxx`.

## 2. Chạy smoke test

Chạy trong `tmux` để không bị dừng khi mất kết nối SSH:

```bash
tmux new -s viword
bash scripts/smoke_h100.sh            # dùng GPU 0; GPU khác: GPU=1 bash scripts/smoke_h100.sh
# thoát tmux mà vẫn để chạy: Ctrl+B rồi D;  vào lại: tmux attach -t viword
```

Script tự làm lần lượt các bước sau:

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| `setup` | tạo `.venv`, cài torch / transformers / vLLM, chạy test, in thông tin GPU | 5–10 phút |
| `check_data` | tải dữ liệu từ Hugging Face về `data/` và `results/teacher_dev/`, kiểm tra đủ file | 1–3 phút |
| `teacher_gate` | cổng G4: Qwen2.5-7B đọc đầu ra của thầy và của truncation trên 200 mẫu dev × 3 tác vụ | 10–15 phút |
| `distill` | dựng nhãn từ đầu ra của thầy, lấy tập con 1.500 đoạn | vài phút |
| `train` | huấn luyện 2 encoder XLM-R-large (nhãn từ, nhãn âm tiết), 1 epoch mỗi encoder | 15–20 phút |
| `eval` | nén và đọc bằng Qwen2.5-7B (vLLM) trên Belebele và ViNLI | 20–30 phút |
| `analyze` | bảng kết quả và kiểm định | < 1 phút |

Theo dõi tiến độ từ một cửa sổ khác:

```bash
tail -f results/smoke/smoke.log      # mỗi bước bắt đầu bằng dòng "=== <bước> (<thời gian>)"
```

Lần chạy **thành công** khi `smoke.log` kết thúc bằng dòng `Smoke test done. Send back the folder results/smoke ...`
và trong `results/smoke/` có các file `g4_*.md` và `analysis.md`.

### Chạy lại một phần

Nếu một bước bị lỗi, sau khi xử lý có thể chạy lại từ bước đó (môi trường `.venv` đã có, không cần `setup` lại):

```bash
bash scripts/smoke_h100.sh check_data                     # chỉ tải lại dữ liệu
bash scripts/smoke_h100.sh teacher_gate                   # chỉ cổng G4
bash scripts/smoke_h100.sh distill train eval analyze     # các bước sau G4
```

Các biến môi trường có thể đổi: `GPU=<id>`, `VIWORD_VLLM_GPU_UTIL=0.7` (phần bộ nhớ GPU vLLM được dùng, mặc định
0.9), `VENV=<đường dẫn môi trường>`. Các biến khác (`READER`, `N_PARAGRAPHS`…) **giữ mặc định**, trừ khi chủ đề tài yêu cầu.

## 3. Gửi lại kết quả

```bash
tar czf smoke_results.tgz results/smoke
```

Gửi file `smoke_results.tgz` (vài chục MB) cho chủ đề tài. Thư mục gồm:

| File | Nội dung |
|---|---|
| `g4_*.md` | kết quả cổng G4 (chủ đề tài xem file này trước tiên) |
| `teacher_review_*.md` | thống kê chất lượng đầu ra của thầy trên từng tác vụ |
| `smoke.log`, `teacher_gate.log`, `train_*.log`, `eval_*.log` | log của từng bước |
| `distill_summary.json`, `analysis.md`, `*_rows.jsonl` | thống kê nhãn, bảng kết quả, kết quả từng mẫu |

Cũng gửi kèm kết quả `nvidia-smi` và `pip freeze > results/smoke/pip_freeze.txt` (chạy sau `source .venv/bin/activate`)
để tái lập được môi trường.

## Khi gặp lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| `missing data/...` hoặc lỗi 401/403 khi tải dữ liệu | `huggingface-cli whoami`; kiểm tra tài khoản đã được mời vào repo `thanthienhai/viword-c-data`, rồi chạy lại `check_data` |
| `no GPU visible` ở bước `setup` | kiểm tra `nvidia-smi` và biến `GPU=<id>` |
| Lỗi cài `vllm` / `torch` (xung đột CUDA) | gửi lại phần log cài đặt và kết quả `nvidia-smi`; không tự đổi phiên bản |
| Hết bộ nhớ GPU (`CUDA out of memory`) ở `teacher_gate` hoặc `eval` | chọn GPU rảnh bằng `GPU=<id>`, hoặc chạy lại bước đó với `VIWORD_VLLM_GPU_UTIL=0.7` |
| Hết bộ nhớ GPU ở `train` | chọn GPU rảnh khác (bước này cần gần trọn một GPU 40 GB) |
| `smoke.log` có dòng `teacher_gate FAILED` | script vẫn chạy tiếp các bước sau; gửi kèm `results/smoke/teacher_gate.log` |
| Lỗi khác (`Traceback` trong log) | dừng lại, gửi `smoke.log` cùng log của bước bị lỗi. **Không sửa code tại chỗ.** |
