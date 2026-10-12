# Hướng dẫn chạy full ViWord-C trên máy GPU (dành cho đội vận hành)

**Mục đích:** chạy lần chính thức: 3 seed × 2 encoder XLM-R-large × 3 epoch trên toàn bộ dữ liệu chưng cất, rồi
đánh giá trên 7 tập test. Lần này thay cho `docs/RUN_FULL_H100.md`.

**Tóm tắt:** tải **một file** `ops_full_run.sh` → chạy `part1` (khoảng 5–6 giờ) → gửi lại file `.tgz` → **chờ chủ đề
tài đồng ý** → chạy `part2` (khoảng 8–12 giờ) → gửi lại file `.tgz`. Script tự clone code, tạo môi trường, tải và
kiểm tra dữ liệu, ghi log và đóng gói kết quả. Không cần sửa code hay cấu hình gì.

> **Không dùng lại thư mục của lần smoke test 04/10.** Lần đó chạy code và dữ liệu cũ. Script chạy trong một
> thư mục mới (`WORKDIR`), giữ nguyên thư mục đó giữa `part1` và `part2`.

## Yêu cầu

| Hạng mục | Yêu cầu |
|---|---|
| Hệ điều hành | Ubuntu, Python ≥ 3.10 (có `python3 -m venv`), `git`, `tar`, `sha256sum`, `tmux` (nếu chạy qua SSH) |
| GPU | 1× H100 80 GB (bf16); `nvidia-smi` chạy được. Lần smoke dùng image `dynamo-vllm-runtime:1.5.0-cu130-vllm0290`: dùng lại được |
| Đĩa | khoảng **100 GB** trống ở `WORKDIR` (môi trường Python, mô hình, 6 checkpoint ~13 GB, các bản nén và kết quả) |
| Mạng | truy cập được github.com, huggingface.co và pypi.org |
| Tài khoản | token Hugging Face có quyền đọc repo **private** `thanthienhai/viword-c-data` (dùng lại secret của lần smoke) |
| Thư mục | `WORKDIR` nằm trên ổ được giữ lại giữa hai phần (PVC/NFS), vì `part2` đọc checkpoint do `part1` tạo ra |

Không cần Java hay mô hình thầy: dữ liệu đã tách từ và đầu ra của thầy đều có sẵn trên Hugging Face.

## 1. Chuẩn bị

```bash
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/main/scripts/ops_full_run.sh
export HF_TOKEN=hf_xxx                                      # token có quyền read
export WORKDIR=/mnt/hps/anhm-paper/viword-c-full            # thư mục CHƯA tồn tại, trên PVC/NFS
nvidia-smi                                                  # xem GPU nào đang rảnh
```

## 2. Phần 1 (`part1`, khoảng 5–6 giờ)

Chạy trong `tmux` để không bị dừng khi mất kết nối SSH:

```bash
tmux new -s viword-full
bash ops_full_run.sh part1             # dùng GPU 0; GPU khác: GPU=1 bash ops_full_run.sh part1
# thoát tmux mà vẫn để chạy: Ctrl+B rồi D;  vào lại: tmux attach -t viword-full
```

Nếu chạy bằng Pod K8s (như lần smoke): đặt lệnh của container là `bash ops_full_run.sh part1`, truyền `HF_TOKEN`
từ secret, `WORKDIR` trỏ vào PVC, `restartPolicy: Never`.

Script tự làm lần lượt các bước sau:

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| clone | clone code vào `WORKDIR` | < 1 phút |
| `setup` | tạo `.venv`, cài torch / transformers / vLLM / llmlingua, chạy unit test, kiểm tra GPU, tải dữ liệu từ Hugging Face | 10–15 phút |
| `check_data` | so từng file dữ liệu với `data/SHA256SUMS` trong code; ghi phiên bản code và dữ liệu | 1 phút |
| `precheck` | chạy thử 20 mẫu cho mỗi tác vụ mới (ViQuAD, GSM8K, OneRuler) và translate-then-compress trên Belebele | 15–20 phút |
| `diagnose` | chẩn đoán cho các cổng G1, G2, G5 trên Belebele và ViNLI | 30–45 phút |
| phase 1 | cổng G1/G2/G3 trên tập dev; dựng nhãn; huấn luyện 6 encoder (3 seed × 2 loại nhãn) | 4–5 giờ |
| đóng gói | nén `results/full` thành `viword_part1_<ngày>_<giờ>.tgz`, đặt cạnh `WORKDIR` | < 1 phút |

`precheck` và `diagnose` không chặn phase 1. Nếu một trong hai bị lỗi, script vẫn chạy tiếp, ghi tên bước vào
`results/full/FAILED_STEPS` và in ra ở cuối. Các bước còn lại lỗi là dừng ngay.

**Theo dõi** từ một cửa sổ khác:

```bash
tail -f $WORKDIR/results/full/ops_part1.log    # mỗi bước bắt đầu bằng "=== <bước> (<thời gian>)"
tail -f $WORKDIR/results/full/full.log         # các giai đoạn bên trong phase 1
```

**Thành công** khi dòng cuối của `ops_part1.log` là `Send back: .../viword_part1_<ngày>_<giờ>.tgz (<cỡ>)` và
`full.log` có dòng `Phase 1 done`.

**Gửi lại** file `viword_part1_*.tgz` (vài MB) cho chủ đề tài. **Dừng ở đây, chờ trả lời.** Không chạy `part2` khi
chưa được đồng ý: nếu cổng G3 không đạt thì cách đánh giá ở phần 2 phải đổi.

## 3. Phần 2 (`part2`, khoảng 8–12 giờ, chỉ khi được đồng ý)

Dùng **đúng** `WORKDIR` của phần 1:

```bash
tmux attach -t viword-full             # hoặc: tmux new -s viword-full
export HF_TOKEN=hf_xxx WORKDIR=/mnt/hps/anhm-paper/viword-c-full
bash ops_full_run.sh part2
```

Script kiểm tra lại dữ liệu, nén và đọc trên 7 tập test (Belebele, ViNLI, ViMMRC, VietNews, ViQuAD, GSM8K, OneRuler),
chạy phân tích, rồi đóng gói thành `viword_part2_<ngày>_<giờ>.tgz`. Thời gian là ước tính thô: 3 tập mới gần gấp
đôi số mẫu, và OneRuler có ngữ cảnh dài 4.096 token.

**Thành công** khi `full.log` có dòng `Phase 2 done` và dòng cuối của `ops_part2.log` là `Send back: ...`.
**Gửi lại** file `viword_part2_*.tgz` (khoảng vài trăm MB, gồm kết quả từng mẫu).

## 4. Khi gặp lỗi

Khi một bước lỗi, script in 40 dòng cuối của log bước đó, **vẫn đóng gói** `viword_<phần>-FAILED_<ngày>_<giờ>.tgz`
rồi dừng. Gửi file này cho chủ đề tài. **Không sửa code tại chỗ.**

Để chạy lại sau khi đã xử lý nguyên nhân (ví dụ đổi GPU): chạy **đúng lệnh cũ với cùng `WORKDIR`**. Các bước đã xong
được bỏ qua (dòng `done in an earlier run, skipped`). Riêng phase 1 chạy lại từ đầu; phase 2 dùng lại các bản nén
đã xong.

| Hiện tượng (trong log) | Cách xử lý |
|---|---|
| `no Hugging Face token` | `export HF_TOKEN=hf_xxx` rồi chạy lại |
| `... exists but is not a part1 folder` | `WORKDIR` trỏ vào một thư mục có sẵn (ví dụ thư mục smoke). Chọn `WORKDIR` mới |
| `code at main has no data/SHA256SUMS` | chủ đề tài chưa đẩy code mới. Báo chủ đề tài, chờ, rồi chạy lại |
| `data differs from data/SHA256SUMS` | dữ liệu trên Hugging Face chưa khớp với code. Báo chủ đề tài; **không** chạy tiếp bằng cách khác |
| lỗi 401/403 khi tải dữ liệu | token không có quyền đọc repo `thanthienhai/viword-c-data` |
| `unit tests did not all pass` | gửi file `-FAILED.tgz` (có `setup.log`) |
| `no GPU visible` | kiểm tra `nvidia-smi` và biến `GPU=<id>` |
| lỗi cài `vllm` / `torch` (xung đột CUDA) | gửi file `-FAILED.tgz` và kết quả `nvidia-smi`; không tự đổi phiên bản |
| `CUDA out of memory` ở `precheck`, `diagnose`, `gate_eval` hoặc `eval` | chọn GPU rảnh (`GPU=<id>`) hoặc thêm `VIWORD_VLLM_GPU_UTIL=0.7`, rồi chạy lại |
| `CUDA out of memory` ở `train` | GPU đang bị chia sẻ; chọn GPU rảnh khác rồi chạy lại |
| Pod/máy bị dừng giữa chừng | chạy lại đúng phần đó với cùng `WORKDIR` |
| `phase 1 did not finish: rerun part1` khi chạy `part2` | phần 1 chưa xong; chạy lại `part1` với cùng `WORKDIR` |
| Lỗi khác (`Traceback`) | gửi file `-FAILED.tgz`. Không sửa code tại chỗ |

Ví dụ chạy lại phần 1 trên GPU 1 với ít bộ nhớ vLLM hơn:

```bash
GPU=1 VIWORD_VLLM_GPU_UTIL=0.7 bash ops_full_run.sh part1
```

## 5. Nội dung các file gửi lại

| File trong `results/full/` | Nội dung |
|---|---|
| `ops_part1.log`, `ops_part2.log` | toàn bộ những gì script in ra |
| `data_revision.txt` | phiên bản code (commit git) và dữ liệu (commit Hugging Face) đã dùng |
| `pip_freeze.txt`, `nvidia_smi.txt` | môi trường, để tái lập |
| `setup.log`, `check_data.log`, `precheck.log`, `diagnose.log` | log của từng bước |
| `precheck/` | kết quả chạy thử các tác vụ mới (`precheck.log` có bảng điểm) |
| `gate_report.md`, `belebele.json`, `vinli.json` | kết quả chẩn đoán, các cổng G1/G2/G5 |
| `full.log`, `<giai đoạn>.log` | log của phase 1 và phase 2 |
| `g123_*.md` | kết quả các cổng G1/G2/G3 trên dev (chủ đề tài xem file này trước tiên) |
| `distill_summary.json`, `*_train_meta.json` | thống kê nhãn và thông tin huấn luyện 6 encoder |
| `mde_dev/report.{md,json}` | MDE H1/H2 trên dev, kiểm tra lề 0,5 ROUGE-L trước test |
| `FAILED_STEPS` | chỉ có khi `precheck` hoặc `diagnose` bị lỗi |
| `analysis_3seed.md`, `eval/` | (phần 2) bảng kết quả gộp 3 seed, kiểm định và kết quả từng mẫu |
