# Hướng dẫn chạy hướng B (nén ngữ cảnh RAG pháp luật, ViBidLQA) trên máy H100

Một script, `scripts/b_h100.sh`, chia theo giai đoạn. Sau mỗi lần gọi, gửi lại thư mục `results/b/` và chờ chủ đề tài
trả lời trước giai đoạn kế tiếp.

| Lệnh | Việc | Thời gian ước tính (1× H100 80 GB) |
|---|---|---|
| `bash scripts/b_h100.sh setup data` | tạo `.venv`, cài phụ thuộc, chạy test; tải dữ liệu hướng B từ repo HF riêng tư | ~10 phút |
| `bash scripts/b_h100.sh smoke` | chạy thử cả chuỗi trên vài câu: nhãn bằng vLLM, huấn luyện scorer 1 epoch, nén và đọc 30 câu dev | ~15 phút |
| `bash scripts/b_h100.sh labels train` | nhãn hữu ích trên 3.654 câu train (~47 nghìn Khoản); 7 scorer tinh chỉnh từ `BAAI/bge-reranker-v2-m3` (3 seed × nhãn hữu ích, 3 seed × nhãn Khoản vàng, 1 bản không thấy Điều của dev/test) | ~45–60 phút + ~1,5–2 giờ |
| `bash scripts/b_h100.sh dev` | 22 phương pháp × 3 tỉ lệ (1/4, 1/8, 1/16) × 2 chế độ ngữ cảnh trên 390 câu dev, cộng oracle; báo cáo `results/b/dev.md` | ~2–3 giờ |
| `bash scripts/b_h100.sh test` | chỉ chạy khi `config/b_registration.json` đã được commit (gate G2); chạy đúng một lần | ~1–1,5 giờ |

Tổng cộng khoảng 6–7 giờ GPU (nhãn khoảng 30 triệu token prompt, không dùng prefix caching vì cần log-probability của prompt; dev khoảng 38 triệu token). Thời gian là ước tính (reader 7B bf16 khoảng 10.000 token/giây trên H100; trên T4 đo
được 550–760), chưa đo trên H100.

## Yêu cầu

- 1 GPU ≥ 40 GB (khuyến nghị H100 80 GB), CUDA, Python ≥ 3.10, khoảng 40 GB đĩa trống (mô hình tải từ Hugging Face:
  Qwen2.5-7B-Instruct, Qwen2.5-1.5B, BAAI/bge-reranker-v2-m3, LLMLingua-2 large; checkpoint 7 scorer ~16 GB). Không cần Java: phần tách từ đã làm sẵn và nằm trong dữ liệu.
- Quyền đọc repo HF riêng tư `thanthienhai/viword-c-legal-data` (`huggingface-cli login` hoặc `export HF_TOKEN=...`).
  Mọi mô hình đều không gated.
- Thư mục repo giữ nguyên giữa các giai đoạn (`runs/` chứa scorer, `data/labels/` chứa nhãn).

## Các bước

```bash
git clone https://github.com/thanthienhai/ViWord-C.git && cd ViWord-C && git checkout rag-legal
huggingface-cli login
tmux new -s viword-b
bash scripts/b_h100.sh setup data smoke
tar czf b_smoke.tgz results/b          # gửi lại, chờ trả lời
bash scripts/b_h100.sh labels train dev
tar czf b_dev.tgz results/b            # gửi lại, chờ trả lời
```

GPU khác: `GPU=1 bash scripts/b_h100.sh ...`. Theo dõi: `tail -f results/b/b.log`; log chi tiết từng bước ở
`results/b/logs/`.

**smoke thành công** khi `b.log` có dòng `smoke done`, ba dòng `wrote … rows` và bốn dòng tóm tắt phương pháp với
`max tokens/budget` ≤ 1,05. Nếu vLLM báo lỗi ở bước nhãn (`prompt_logprobs`), gửi `results/b/logs/smoke_labels.log`.

## Giai đoạn test (chỉ khi được báo)

Chủ đề tài commit `config/b_registration.json` (phương pháp đề xuất, các baseline, danh sách phương pháp chạy trên
test) sau khi xem kết quả dev. Sau đó:

```bash
git pull
bash scripts/b_h100.sh test
tar czf b_test.tgz results/b
```

Script từ chối chạy nếu file đăng ký chưa được commit hoặc đã bị sửa so với commit.

## Gửi lại gì

Toàn bộ `results/b/`: `b.log`, `logs/`, `eval/` (điểm từng câu và bản nén), `*_train_meta.json` của các scorer,
`dev.md` hoặc `test.md`. Không cần gửi `runs/` hay `data/labels/` (có thể giữ trên máy để chạy lại).

## Ghi chú cho chủ đề tài

- Dữ liệu: `data/SHA256SUMS_B` liệt kê 22 file (148 MB); đẩy lên repo HF bằng
  `python scripts/hf_data.py push --repo thanthienhai/viword-c-legal-data --sums data/SHA256SUMS_B`.
- Reader H100 là bf16, khác reader 8 bit của các lượt Kaggle, nên mọi số liệu dev được đo lại tại đây.
- Bản nén của VLegal-Bench không được phát hành (CC BY-NC-ND); bộ dữ liệu này chỉ có ViBidLQA.
