# Hướng dẫn chạy full ViWord-C trên máy GPU (dành cho đội vận hành)

> **Đã thay bằng [`RUN_OPS.md`](RUN_OPS.md)** (một file `scripts/ops_full_run.sh`, có kiểm tra dữ liệu và chạy thử
> các tác vụ mới). Giữ lại file này để tham khảo cách chạy từng pha bằng `full_h100.sh`.

Chạy sau khi smoke test đã thành công (`docs/RUN_SMOKE_H100.md`). Lần chạy này cho **số liệu chính thức**: 3 seed ×
2 encoder XLM-R-large × 3 epoch trên toàn bộ dữ liệu chưng cất, đánh giá trên đủ 4 tác vụ.

Chia làm **hai pha**, giữa hai pha phải chờ chủ đề tài đồng ý:

| Pha | Lệnh | Việc | Thời gian ước tính (1× H100) |
|---|---|---|---|
| 1 | `bash scripts/full_h100.sh phase1` | cổng G1/G2/G3 trên tập dev; dựng nhãn; huấn luyện 6 encoder; `ckpt_check` reader trên 200 mẫu dev | 4–5 giờ |
| 2 | `bash scripts/full_h100.sh phase2` | chọn π trên dev, tính MDE, nén + đọc trên Belebele, ViNLI, ViMMRC, VietNews; phân tích | 4–6 giờ |

Lý do tách pha: nếu cổng G3 không đạt, cách đánh giá ở pha 2 phải đổi. Không chạy pha 2 khi chưa được đồng ý.

## Yêu cầu

Giống smoke test (1 GPU ≥ 80 GB khuyến nghị, quyền đọc repo HF `thanthienhai/viword-c-data`), thêm:

- Khoảng **40 GB** đĩa trống ngoài phần của smoke (6 checkpoint XLM-R-large ~13 GB, các bản nén và kết quả).
- Thư mục repo phải **giữ nguyên giữa hai pha** (PVC/NFS), vì pha 2 đọc checkpoint trong `runs/` do pha 1 tạo ra.

## 0. Chuẩn bị (bỏ qua nếu dùng lại đúng thư mục đã chạy smoke)

```bash
git clone https://github.com/thanthienhai/ViWord-C.git && cd ViWord-C
huggingface-cli login                          # hoặc: export HF_TOKEN=hf_xxx
bash scripts/smoke_h100.sh setup check_data    # tạo .venv và tải dữ liệu (~10 phút)
```

Nếu dùng lại thư mục smoke cũ, cập nhật code và cài lại phụ thuộc (đã thêm gói `llmlingua`):

```bash
git pull
bash scripts/smoke_h100.sh setup check_data
```

## 1. Pha 1

```bash
tmux new -s viword-full
bash scripts/full_h100.sh phase1               # GPU khác: GPU=1 bash scripts/full_h100.sh phase1
```

Theo dõi: `tail -f results/full/full.log` (mỗi giai đoạn có dòng bắt đầu và `done`); log chi tiết của từng giai đoạn
nằm trong `results/full/<giai đoạn>.log`.

Pha 1 **thành công** khi `full.log` kết thúc bằng `Phase 1 done`. Khi đó gửi lại:

```bash
tar czf full_phase1.tgz results/full
```

Gồm `g123_*.md` (kết quả cổng), `distill_summary.json`, `*_train_meta.json` (6 encoder), báo cáo `ckpt_check` và các log.
**Dừng ở đây, chờ chủ đề tài trả lời.**

## 2. Pha 2 (chỉ khi được đồng ý)

```bash
tmux attach -t viword-full                     # hoặc tmux new -s viword-full
bash scripts/full_h100.sh phase2
tar czf full_phase2.tgz results/full
```

Thành công khi `full.log` kết thúc bằng `Phase 2 done`. Phase 2 phải chạy MDE dev và có
`results/full/report.{md,json}` trước khi test evaluation; file tar có `analysis_3seed.md`, báo cáo MDE và thư mục `eval/`
(kết quả từng mẫu, khoảng vài trăm MB).

## Khi gặp lỗi

Script dừng ngay ở giai đoạn lỗi, in 30 dòng cuối của log và ghi `<giai đoạn> FAILED` vào `full.log`.

| Hiện tượng | Cách xử lý |
|---|---|
| `CUDA out of memory` ở `gate_eval` hoặc `eval` | chạy lại pha đó với `VIWORD_VLLM_GPU_UTIL=0.7`, hoặc chọn GPU rảnh bằng `GPU=<id>` |
| `CUDA out of memory` ở `train` | GPU đang bị chia sẻ; chọn GPU rảnh khác |
| `missing runs/...: run phase1 first` | pha 1 chưa xong hoặc chạy pha 2 ở thư mục khác |
| `No module named 'llmlingua'` | chạy lại `bash scripts/smoke_h100.sh setup` |
| Pod bị dừng giữa chừng | chạy lại đúng pha đó. Các bản nén đã xong được dùng lại; huấn luyện chạy lại từ đầu |
| Lỗi khác | gửi `results/full/full.log` và log của giai đoạn lỗi. **Không sửa code tại chỗ.** |

Lưu ý khi chạy lại `eval`: bản nén được lưu trong `results/eval/*.compressed.jsonl` và được **dùng lại** ở lần sau.
Nếu code đã được cập nhật giữa hai lần chạy, xoá các file đó trước (`rm results/eval/*.compressed.jsonl`).
