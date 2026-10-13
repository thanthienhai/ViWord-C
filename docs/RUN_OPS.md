# Hướng dẫn chạy full ViWord-C trên máy GPU (dành cho đội vận hành)

**Mục đích:** chạy lần chính thức: 3 seed × 2 encoder XLM-R-large × 3 epoch trên toàn bộ dữ liệu chưng cất, chọn π
và kiểm tra lực thống kê trên dev, rồi đánh giá trên 7 tập test. Lần này thay cho `docs/RUN_FULL_H100.md`.

**Tóm tắt:** tải **một file** `ops_full_run.sh` rồi chạy **ba phần**. Sau mỗi phần gửi lại file `.tgz` và **chờ chủ đề
tài đồng ý** rồi mới chạy phần tiếp:

| Phần | Lệnh | Việc | Thời gian ước tính (1× H100) |
|---|---|---|---|
| 1 | `bash ops_full_run.sh part1` | môi trường, kiểm tra dữ liệu, chạy thử, cổng G1/G2/G3 trên dev, dựng nhãn, huấn luyện 6 encoder, `ckpt_check` | 5–6 giờ |
| 2 | `bash ops_full_run.sh part2` | các quyết định trên dev: chẩn đoán G1/G2/G5, chọn π, tính MDE | 2–4 giờ |
| 3 | `bash ops_full_run.sh part3` | lấy code có π đã chốt, nén + đọc trên 7 tập test, BERTScore, phân tích | 12–18 giờ |

Script tự clone code, tạo môi trường, tải và kiểm tra dữ liệu, ghi log và đóng gói kết quả. Không cần sửa code hay
cấu hình gì.

Lý do phải dừng giữa các phần:
- Sau phần 1: nếu cổng G3 không đạt thì cách đánh giá phải đổi.
- Sau phần 2: π chọn trên dev phải được chủ đề tài **commit vào git** trước khi nhìn tập test. Bước `eval` từ chối
  chạy nếu π chưa được commit.

**Khác với bản hướng dẫn trước (hai phần):**
- Có thêm **phần 3**. Phần 2 cũ (nén + đọc trên test) nay là phần 3, chạy sau khi chủ đề tài commit π.
- Phần 2 mới chỉ chạy trên dev: chẩn đoán, chọn π, tính MDE.
- Token Hugging Face phải được duyệt quyền mô hình gated `Viet-Mistral/Vistral-7B-Chat` (mục 1).
- **Tải lại `ops_full_run.sh` trước mỗi phần.** Bản cũ không có `part3` và không kiểm tra dữ liệu mới.
- Hai cổng `ckpt_check` (phần 1) và `mde` (phần 2) có thể dừng với kết quả `FAIL`. Đó là kết quả nghiên cứu, không phải
  lỗi máy (mục 5).

> **Không dùng lại thư mục của các lần chạy trước** (smoke 04/10 hay lần full nào trước ngày 13/10). Dữ liệu tách từ
> đã được làm lại bằng VnCoreNLP, nên mọi kết quả và checkpoint cũ không còn dùng được. Script chạy trong một thư mục
> mới (`WORKDIR`), giữ nguyên thư mục đó qua cả ba phần.

## 0. Trước khi bắt đầu: chờ chủ đề tài báo sẵn sàng

Đội vận hành không làm các việc dưới đây, chỉ cần chờ chủ đề tài báo đã xong. Nếu chạy khi chưa xong, script sẽ dừng
với thông báo ở mục 5.

| Trước phần | Chủ đề tài đã làm | Nếu chưa làm |
|---|---|---|
| 1 | đẩy code lên GitHub `main`, đẩy dữ liệu mới lên Hugging Face (khớp `data/SHA256SUMS`) | `check_data` dừng |
| 1 | gán nhãn và commit hai phiếu mù `data/annot/protect_sheet.jsonl`, `probe_sheet.jsonl` | phần 1: `diagnose` lỗi (không chặn); phần 2: dừng |
| 3 | commit `config/pi_selected.{json,env}` (lấy từ file `.tgz` của phần 2) | `update_code` dừng |
| 3 | sinh cận trên của thầy trên test (`results/teacher_test`), đẩy lên Hugging Face, cập nhật `data/SHA256SUMS` | `check_data_part3` hoặc `eval` dừng |

## 1. Yêu cầu

| Hạng mục | Yêu cầu |
|---|---|
| Hệ điều hành | Ubuntu, Python ≥ 3.10 (có `python3 -m venv`), `git`, `tar`, `sha256sum`, `tmux` (nếu chạy qua SSH) |
| GPU | 1× H100 80 GB (bf16); `nvidia-smi` chạy được. Lần smoke dùng image `dynamo-vllm-runtime:1.5.0-cu130-vllm0290`: dùng lại được |
| Đĩa | khoảng **100 GB** trống ở `WORKDIR` (môi trường Python, mô hình, 6 checkpoint ~13 GB, các bản nén và kết quả) |
| Mạng | truy cập được github.com, huggingface.co và pypi.org |
| Tài khoản Hugging Face | một token có quyền đọc repo **private** `thanthienhai/viword-c-data` (dùng lại secret của lần smoke), **và** tài khoản của token đã được duyệt quyền truy cập mô hình **gated** [`Viet-Mistral/Vistral-7B-Chat`](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) (vào trang mô hình, bấm đồng ý điều khoản, chờ duyệt). Bước `diagnose` cần tokenizer của Vistral |
| Thư mục | `WORKDIR` nằm trên ổ được giữ lại qua cả ba phần (PVC/NFS), vì phần sau đọc checkpoint và báo cáo của phần trước |

Không cần Java hay mô hình thầy: dữ liệu đã tách từ và mọi đầu ra của thầy đều có sẵn trên Hugging Face.

Kiểm tra nhanh token trước khi chạy (cả hai lệnh phải in `ok`; `401`/`403` là chưa có quyền):

```bash
export HF_TOKEN=hf_xxx
curl -fsS -o /dev/null -H "Authorization: Bearer $HF_TOKEN"   https://huggingface.co/api/datasets/thanthienhai/viword-c-data && echo ok
curl -fsSL -o /dev/null -H "Authorization: Bearer $HF_TOKEN"   https://huggingface.co/Viet-Mistral/Vistral-7B-Chat/resolve/main/tokenizer_config.json && echo ok
```

## 2. Phần 1 (`part1`, khoảng 5–6 giờ)

```bash
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/main/scripts/ops_full_run.sh
export HF_TOKEN=hf_xxx                                      # token có quyền read (xem mục 1)
export WORKDIR=/mnt/hps/anhm-paper/viword-c-full            # thư mục CHƯA tồn tại, trên PVC/NFS
nvidia-smi                                                  # xem GPU nào đang rảnh

tmux new -s viword-full                # để không bị dừng khi mất kết nối SSH
bash ops_full_run.sh part1             # dùng GPU 0; GPU khác: GPU=1 bash ops_full_run.sh part1
# thoát tmux mà vẫn để chạy: Ctrl+B rồi D;  vào lại: tmux attach -t viword-full
```

Nếu chạy bằng Pod K8s (như lần smoke): đặt lệnh của container là `bash ops_full_run.sh part1` (phần sau thì
`part2`, `part3`), truyền `HF_TOKEN` từ secret, `WORKDIR` trỏ vào PVC, `restartPolicy: Never`.

Script tự làm lần lượt các bước sau:

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| clone | clone code vào `WORKDIR` | < 1 phút |
| `setup` | tạo `.venv`, cài torch / transformers / vLLM / llmlingua / bert_score, chạy unit test (vài test cần dữ liệu được bỏ qua, `skipped` là bình thường), kiểm tra GPU, tải dữ liệu từ Hugging Face | 10–15 phút |
| `check_data` | so từng file dữ liệu với `data/SHA256SUMS` trong code, kiểm tra nguồn gốc tách từ (VnCoreNLP); ghi phiên bản code và dữ liệu | 1 phút |
| `precheck` | chạy thử 20 mẫu cho mỗi tác vụ mới (ViQuAD, GSM8K, OneRuler) và translate-then-compress trên Belebele | 15–20 phút |
| `diagnose` | đánh giá bộ phát hiện P trên phiếu đã gán nhãn; chẩn đoán G1, G2, G5 trên Belebele và ViNLI | 30–45 phút |
| `gate_eval` | cổng G1/G2/G3 trên 3 tập dev (13 bộ nén, thêm probe trên ViNLI, × 3 tỉ lệ) | 30–60 phút |
| `distill`, `train` | dựng nhãn; huấn luyện 6 encoder (3 seed × 2 loại nhãn, 3 epoch) | 3,5–4 giờ |
| `ckpt_check` | reader kiểm tra 6 checkpoint trên 200 mẫu mỗi tập dev | 5–10 phút |
| đóng gói | nén `results/full` thành `viword_part1_<ngày>_<giờ>.tgz`, đặt cạnh `WORKDIR` | < 1 phút |

`precheck` và `diagnose` không chặn các bước sau. Nếu một trong hai bị lỗi, script vẫn chạy tiếp, ghi tên bước vào
`results/full/FAILED_STEPS` và in ra ở cuối (`diagnose` sẽ chạy lại ở phần 2). Các bước còn lại lỗi là dừng ngay.

**Theo dõi** từ một cửa sổ khác:

```bash
tail -f $WORKDIR/results/full/ops_part1.log    # mỗi bước bắt đầu bằng "=== <bước> (<thời gian>)"
tail -f $WORKDIR/results/full/full.log         # các giai đoạn gate_eval, distill, train, ckpt_check
```

**Thành công** khi dòng cuối của `ops_part1.log` là `Send back: .../viword_part1_<ngày>_<giờ>.tgz (<cỡ>)` và
`full.log` có dòng `Phase 1 done`.

**Gửi lại** file `viword_part1_*.tgz` (vài MB) cho chủ đề tài. **Dừng ở đây, chờ trả lời.**

## 3. Phần 2 (`part2`, khoảng 2–4 giờ, chỉ khi được đồng ý)

Dùng **đúng** `WORKDIR` của phần 1:

```bash
tmux attach -t viword-full             # hoặc: tmux new -s viword-full
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/main/scripts/ops_full_run.sh   # bản mới nhất
export HF_TOKEN=hf_xxx WORKDIR=/mnt/hps/anhm-paper/viword-c-full
bash ops_full_run.sh part2
```

| Bước (trong `full.log`) | Việc | Thời gian ước tính |
|---|---|---|
| `check_data_part2` | kiểm tra lại dữ liệu | 1 phút |
| `diagnose` | đánh giá P và probe trên phiếu mù; chẩn đoán G1/G2/G5, thêm 3 checkpoint LLMLingua-2-vi | 45–60 phút |
| `select_pi` | chọn π trên dev: 49 cấu hình × 3 tỉ lệ × ~1.200 mẫu dev | 1–2,5 giờ |
| `mde` | ước lượng hiệu ứng nhỏ nhất phát hiện được (MDE) trên dev | 15–30 phút |

**Thành công** khi `full.log` có dòng `Phase 2 done` và dòng cuối của `ops_part2.log` là `Send back: ...`.
**Gửi lại** file `viword_part2_*.tgz` (vài MB). **Dừng ở đây, chờ chủ đề tài commit π rồi báo chạy phần 3.**

## 4. Phần 3 (`part3`, khoảng 12–18 giờ, chỉ khi được báo)

Dùng **đúng** `WORKDIR` của hai phần trước:

```bash
tmux attach -t viword-full             # hoặc: tmux new -s viword-full
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/main/scripts/ops_full_run.sh   # bản mới nhất
export HF_TOKEN=hf_xxx WORKDIR=/mnt/hps/anhm-paper/viword-c-full
bash ops_full_run.sh part3
```

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| `update_code` | cập nhật code lên commit mới nhất của `main` (chỉ fast-forward), kiểm tra π đã được commit, cài lại gói nếu cần | 1–3 phút |
| `pull_data_part3`, `check_data_part3` | tải phần dữ liệu mới (cận trên của thầy trên test) và kiểm tra lại toàn bộ | 2–5 phút |
| `eval` | nén + đọc trên 7 tập test (Belebele, ViNLI, ViMMRC, VietNews, ViQuAD, GSM8K, OneRuler): ~10.000 mẫu × 3 tỉ lệ × ~40 phương pháp | 12–18 giờ |
| `bertscore` | BERTScore (PhoBERT) cho VietNews | 10–20 phút |
| `analyze` | bảng kết quả gộp 3 seed và kiểm định H1/H2 | < 5 phút |

Thời gian `eval` là ước tính thô từ tốc độ của lần smoke (khoảng 25–75 dòng/giây). OneRuler có ngữ cảnh dài 4.096
token và GSM8K sinh lời giải dài, nên hai tập này chậm hơn.

**Thành công** khi `full.log` có dòng `Phase 3 done` và dòng cuối của `ops_part3.log` là `Send back: ...`.
**Gửi lại** file `viword_part3_*.tgz` (khoảng vài trăm MB, gồm kết quả từng mẫu).

## 5. Khi gặp lỗi

Khi một bước lỗi, script in 40 dòng cuối của log bước đó, **vẫn đóng gói** `viword_<phần>-FAILED_<ngày>_<giờ>.tgz`
rồi dừng. Gửi file này cho chủ đề tài. **Không sửa code tại chỗ.**

Để chạy lại sau khi đã xử lý nguyên nhân (ví dụ đổi GPU): chạy **đúng lệnh cũ với cùng `WORKDIR`**. Các bước đã xong
được bỏ qua (dòng `done in an earlier run, skipped`). Riêng các giai đoạn trong `full.log` chạy lại từ đầu pha;
`eval` dùng lại các bản nén đã xong. Phần 3 luôn lấy lại code và dữ liệu mới nhất trước khi chạy.

**Cổng không đạt không phải lỗi máy.** Hai bước dưới đây dừng có chủ đích khi kết quả nghiên cứu không đạt. Báo cáo
vẫn nằm trong file `-FAILED.tgz`. Gửi file đó, **không chạy lại**, chờ chủ đề tài quyết định.

| Bước | Dòng cuối log | Ý nghĩa |
|---|---|---|
| `ckpt_check` (phần 1) | `## FAIL` trong bảng | checkpoint ViWord-C không thắng truncation hoặc đối chứng tra từ trên đa số tập dev |
| `mde` (phần 2) | `mde: FAIL; increase the DEV pilot ...` | lực thống kê không đủ cho lề 0,5 ROUGE-L đã đăng ký |

Các lỗi khác:

| Hiện tượng (trong log) | Cách xử lý |
|---|---|
| `no Hugging Face token` | `export HF_TOKEN=hf_xxx` rồi chạy lại |
| `... exists but is not a part1 folder` | `WORKDIR` trỏ vào một thư mục có sẵn (ví dụ thư mục smoke). Chọn `WORKDIR` mới |
| `code at main has no data/SHA256SUMS` | chủ đề tài chưa đẩy code mới. Báo chủ đề tài, chờ, rồi chạy lại |
| `data differs from data/SHA256SUMS` | dữ liệu trên Hugging Face chưa khớp với code. Báo chủ đề tài; **không** chạy tiếp bằng cách khác |
| `segmented data is missing VnCoreNLP provenance` | như trên: dữ liệu tách từ trên Hugging Face chưa phải bản mới. Báo chủ đề tài |
| lỗi 401/403 khi tải dữ liệu | token không có quyền đọc repo `thanthienhai/viword-c-data` |
| `gated repo` / `Access to model Viet-Mistral/Vistral-7B-Chat is restricted` (ở `diagnose`) | tài khoản của token chưa được duyệt Vistral (mục 1). Xin quyền rồi chạy lại |
| `missing annotator for`, `gold_tiers not filled in`, `labels must come from a blind sheet`, `missing results/protect_eval.json` | chủ đề tài chưa điền xong phiếu mù. Ở phần 1 chỉ ghi vào `FAILED_STEPS`; ở phần 2 thì báo chủ đề tài, chờ, rồi chạy lại `part2` |
| `config/pi_selected.* is not committed at origin/main` | chủ đề tài chưa commit π. Báo chủ đề tài, chờ, rồi chạy lại `part3` |
| `cannot fast-forward the code` | code trên máy đã bị sửa hoặc lịch sử `main` bị viết lại. Gửi file `-FAILED.tgz`; không tự `git reset` |
| `missing results/teacher_test/...` | chủ đề tài chưa đẩy cận trên của thầy lên Hugging Face. Báo chủ đề tài, chờ, rồi chạy lại `part3` |
| `unit tests did not all pass` | gửi file `-FAILED.tgz` (có `setup.log`) |
| `no GPU visible` | kiểm tra `nvidia-smi` và biến `GPU=<id>` |
| lỗi cài `vllm` / `torch` (xung đột CUDA) | gửi file `-FAILED.tgz` và kết quả `nvidia-smi`; không tự đổi phiên bản |
| `CUDA out of memory` ở `precheck`, `diagnose`, `gate_eval`, `ckpt_check`, `select_pi`, `mde` hoặc `eval` | chọn GPU rảnh (`GPU=<id>`) hoặc thêm `VIWORD_VLLM_GPU_UTIL=0.7`, rồi chạy lại |
| `CUDA out of memory` ở `train` | GPU đang bị chia sẻ; chọn GPU rảnh khác rồi chạy lại |
| Pod/máy bị dừng giữa chừng | chạy lại đúng phần đó với cùng `WORKDIR` |
| `phase 1 did not finish: rerun part1` / `phase 2 did not finish: rerun part2` | phần trước chưa xong; chạy lại phần đó với cùng `WORKDIR` |
| `part3` in ra hướng dẫn sử dụng thay vì chạy | đang dùng `ops_full_run.sh` cũ (hai phần). Tải lại bằng lệnh `curl` ở mục 4 |
| Lỗi khác (`Traceback`) | gửi file `-FAILED.tgz`. Không sửa code tại chỗ |

Ví dụ chạy lại phần 1 trên GPU 1 với ít bộ nhớ vLLM hơn:

```bash
GPU=1 VIWORD_VLLM_GPU_UTIL=0.7 bash ops_full_run.sh part1
```

## 6. Nội dung các file gửi lại

| File trong `results/full/` | Phần | Nội dung |
|---|---|---|
| `ops_part{1,2,3}.log` | 1–3 | toàn bộ những gì script in ra |
| `data_revision.txt` | 1–3 | phiên bản code (commit git) và dữ liệu (commit Hugging Face) của từng phần |
| `pip_freeze.txt`, `nvidia_smi.txt` | 1 | môi trường, để tái lập |
| `<bước>.log`, `full.log` | 1–3 | log của từng bước và từng giai đoạn |
| `precheck/` | 1 | kết quả chạy thử các tác vụ mới (`precheck.log` có bảng điểm) |
| `g123_*.md` | 1 | kết quả các cổng G1/G2/G3 trên dev (chủ đề tài xem file này trước tiên) |
| `distill_summary.json`, `*_train_meta.json` | 1 | thống kê nhãn và thông tin huấn luyện 6 encoder |
| `ckpt_check_report.{md,json}` | 1 | kiểm tra reader cho 6 checkpoint |
| `FAILED_STEPS` | 1 | chỉ có khi `precheck` hoặc `diagnose` bị lỗi |
| `protect_eval.{md,json}`, `probe_negation_eval.{md,json}` | 1, 2 | độ chính xác của bộ phát hiện P và của probe phủ định trên phiếu mù |
| `gate_report.md`, `belebele.json`, `vinli.json`, `llmlingua2*_*.json` | 1, 2 | kết quả chẩn đoán, các cổng G1/G2/G5 |
| `pi_selection.md`, `pi_selected.{json,env}` | 2 | π được chọn trên dev (chủ đề tài commit hai file `pi_selected.*`) |
| `mde_report.{md,json}` | 2 | MDE H1/H2 trên dev, kiểm tra lề 0,5 ROUGE-L trước test |
| `pi_selected_part2/` | 3 | bản π do phần 2 sinh ra; log `update_code.log` ghi chú nếu khác bản đã commit |
| `analysis_3seed.md`, `bertscore/`, `eval/` | 3 | bảng kết quả gộp 3 seed, kiểm định, BERTScore VietNews và kết quả từng mẫu |

## Phụ lục: việc của chủ đề tài giữa các phần

Đội vận hành không cần đọc phần này.

**Trước phần 1.** Đẩy code (`git push`) và dữ liệu (`python scripts/hf_data.py push --repo thanthienhai/viword-c-data`),
sao cho `sha256sum -c data/SHA256SUMS` đúng trên bản Hugging Face. Gán nhãn hai phiếu trong `data/annot/`, chạy
`bash scripts/run_pipeline.sh protect_eval` để kiểm tra phiếu hợp lệ, rồi commit.

**Sau phần 2, trước phần 3.**

```bash
# 1. π: lấy hai file từ viword_part2_*.tgz, xem pi_selection.md, rồi commit đúng nguyên văn
mkdir -p config && cp <giải nén>/results/full/pi_selected.{json,env} config/
git add config/pi_selected.json config/pi_selected.env && git commit -m "Select π on dev" && git push
# 2. cận trên của thầy trên test (cần gọi được TEACHER_URL và gói openai: pip install -e ".[teacher]"; không cần GPU)
bash scripts/run_pipeline.sh teacher_test
sha256sum results/teacher_test/*.jsonl >> data/SHA256SUMS      # chỉ một lần; chạy lại thì xoá các dòng cũ trước
git add data/SHA256SUMS && git commit -m "Checksums for teacher_test" && git push
python scripts/hf_data.py push --repo thanthienhai/viword-c-data --message "teacher_test"
```

Có thể làm bước 2 ngay từ trước phần 1, vì nó không phụ thuộc kết quả nào trên GPU. Bản 8 trang (H3) còn cần
`select_alpha` trên dev và commit `config/alpha_selected.*` trước phần 3; không có hai file này thì `eval` bỏ qua các
dòng H3. Reader thứ hai (Vistral) không nằm trong ba phần này: chạy riêng `eval` với `READER=Viet-Mistral/Vistral-7B-Chat`
sau khi đã kiểm tra chat template (`scripts/check_reader_template.py`).
