# Hướng dẫn chạy hướng B (nén ngữ cảnh RAG pháp luật, ViBidLQA) trên máy GPU (dành cho đội vận hành)

**Mục đích:** huấn luyện bộ chấm điểm Khoản (scorer) của hướng B và đánh giá bộ nén đề xuất cùng các baseline trên
tập dev của ViBidLQA; sau khi chủ đề tài đăng ký cấu hình, chạy một lần trên tập test.

**Tóm tắt:** tải **một file** `ops_b_run.sh` (nhánh `rag-legal`) rồi chạy **ba phần**. Sau mỗi phần, gói kết quả
`.tgz` và checksum `.sha256` được đẩy lên Hugging Face (hoặc gửi tay, mục 5), rồi **chờ chủ đề tài đồng ý** mới chạy
phần tiếp:

| Phần | Lệnh | Việc | Thời gian ước tính (1× H100) |
|---|---|---|---|
| 1 | `bash ops_b_run.sh part1` | môi trường, tải và kiểm tra dữ liệu, chạy thử cả chuỗi trên vài câu | 30–40 phút |
| 2 | `bash ops_b_run.sh part2` | dựng nhãn, huấn luyện 7 scorer, đánh giá 22 phương pháp trên dev | 5–6 giờ |
| 3 | `bash ops_b_run.sh part3` | chạy đúng cấu hình đã đăng ký trên tập test, một lần | 1–1,5 giờ |

Thời gian là ước tính từ tốc độ đo trên T4, **chưa đo trên H100**; phần 1 cho số đo đầu tiên.

Script tự clone code, tạo môi trường, tải và kiểm tra dữ liệu, ghi log và đóng gói kết quả. Không cần sửa code hay
cấu hình gì.

Lý do phải dừng giữa các phần:
- Sau phần 1: phần chạy thử dùng vLLM theo một cách chưa từng chạy trên GPU (lấy log-probability của prompt). Nếu lỗi,
  chủ đề tài sửa code trước khi tốn 5–6 giờ của phần 2.
- Sau phần 2: chủ đề tài phải chọn và **commit cấu hình** (`config/b_registration.json`) trước khi nhìn tập test.
  Phần 3 từ chối chạy nếu cấu hình chưa được commit.

**Khác với lần chạy ViWord-C (`ops_full_run.sh`, `docs/RUN_OPS.md`):**
- Đây là một lần chạy riêng: **file khác** (`ops_b_run.sh`), **nhánh code khác** (`rag-legal`), **repo dữ liệu khác**
  (`thanthienhai/viword-c-legal-data`), **thư mục làm việc khác**. Không dùng lại `WORKDIR` của ViWord-C.
- Có thể chạy song song với ViWord-C nếu mỗi bên dùng một GPU riêng (`GPU=<id>`).
- Repo nhận kết quả vẫn là `thanthienhai/viword-c-results`, cùng `HF_RESULTS_TOKEN`; gói có tên `vibid_<phần>_...`.
- **Không cho reader đọc tập test trước phần 3**, kể cả khi chạy thử. `evaluate.py` từ chối các dòng test
  (`refusing to let a reader score ... test row(s)`); không tìm cách vượt qua.

## 0. Trước khi bắt đầu: chờ chủ đề tài báo sẵn sàng

Đội vận hành không làm các việc dưới đây, chỉ cần chờ chủ đề tài báo đã xong.

| Trước phần | Chủ đề tài đã làm | Nếu chưa làm | Trạng thái 26/10 |
|---|---|---|---|
| 1 | đẩy code nhánh `rag-legal` lên GitHub | clone hoặc `data` dừng | **xong** |
| 1 | đẩy dữ liệu lên repo HF private `thanthienhai/viword-c-legal-data` (khớp `data/SHA256SUMS_B`) | `data` dừng | **xong** (revision `e40f36f`, 22/22 file khớp) |
| 1 | cấp cho `HF_TOKEN` của đội vận hành quyền **đọc** repo dữ liệu mới | lỗi 401/403 ở bước `data` | **cần làm**: repo tạo 26/10; token fine-grained của ViWord-C chỉ đọc `viword-c-data` |
| 1 | `HF_RESULTS_TOKEN` ghi được `thanthienhai/viword-c-results` | gói kết quả phải gửi tay (mục 5) | **xong** (dùng lại token của ViWord-C) |
| 3 | commit `config/b_registration.json` (sau khi xem kết quả dev của phần 2) | phần 3 dừng | chờ phần 2 |

## 1. Yêu cầu

| Hạng mục | Yêu cầu |
|---|---|
| Hệ điều hành | Ubuntu, Python ≥ 3.10 (có `python3 -m venv`), `git`, `tar`, `sha256sum`, `curl`, `tmux` (nếu chạy qua SSH) |
| GPU | 1× H100 80 GB (bf16); tối thiểu 40 GB. `nvidia-smi` chạy được. Image `dynamo-vllm-runtime:1.5.0-cu130-vllm0290` của lần smoke dùng lại được |
| Đĩa | khoảng **60 GB** trống ở `WORKDIR`: môi trường Python, các mô hình tải về (Qwen2.5-7B-Instruct, Qwen2.5-1.5B, BAAI/bge-reranker-v2-m3, LLMLingua-2 large), 7 scorer (~16 GB), các bản nén và kết quả |
| Mạng | truy cập được github.com, raw.githubusercontent.com, huggingface.co và pypi.org |
| `HF_TOKEN` | token có quyền **đọc** repo private `thanthienhai/viword-c-legal-data`. Mọi mô hình đều không gated |
| `HF_RESULTS_TOKEN` | token chỉ có quyền **ghi** repo private `thanthienhai/viword-c-results` (như ViWord-C). Không có thì gửi tay (mục 5) |
| Thư mục | `WORKDIR` **mới**, nằm trên ổ được giữ lại qua cả ba phần (PVC/NFS): phần sau dùng nhãn và scorer của phần trước |

Không cần Java: dữ liệu đã tách từ sẵn.

Kiểm tra quyền trước khi chạy (lệnh `curl` phải in `ok`; `401`/`403` là token chưa có quyền, báo chủ đề tài):

```bash
export HF_TOKEN=hf_xxx HF_RESULTS_TOKEN=hf_yyy
curl -fsS -o /dev/null -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/datasets/thanthienhai/viword-c-legal-data && echo ok
```

## 2. Phần 1 (`part1`, khoảng 30–40 phút)

```bash
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/rag-legal/scripts/ops_b_run.sh
export HF_TOKEN=hf_xxx                                      # token đọc repo dữ liệu (mục 1)
export HF_RESULTS_TOKEN=hf_yyy                              # token ghi repo kết quả (mục 1)
export WORKDIR=/mnt/hps/anhm-paper/viword-b                 # thư mục CHƯA tồn tại, trên PVC/NFS
nvidia-smi                                                  # xem GPU nào đang rảnh

tmux new -s viword-b                   # để không bị dừng khi mất kết nối SSH
bash ops_b_run.sh part1                # dùng GPU 0; GPU khác: GPU=1 bash ops_b_run.sh part1
# thoát tmux mà vẫn để chạy: Ctrl+B rồi D;  vào lại: tmux attach -t viword-b
```

Nếu chạy bằng Pod K8s: đặt lệnh của container là `bash ops_b_run.sh part1` (phần sau thì `part2`, `part3`), truyền
`HF_TOKEN` và `HF_RESULTS_TOKEN` từ secret, `WORKDIR` trỏ vào PVC, `restartPolicy: Never`.

Script tự làm lần lượt các bước sau:

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| clone | clone nhánh `rag-legal` vào `WORKDIR` | < 1 phút |
| `setup` | tạo `.venv`, cài torch / transformers / vLLM / llmlingua, chạy unit test, kiểm tra GPU | 10–15 phút |
| `data` | tải dữ liệu từ Hugging Face, so từng file với `data/SHA256SUMS_B` trong code; ghi phiên bản code và dữ liệu | 1–2 phút |
| `smoke` | chạy thử cả chuỗi: nhãn cho 16 câu train bằng vLLM, huấn luyện scorer 1 epoch, nén và đọc 30 câu dev bằng 4 phương pháp | 15–20 phút (tải mô hình lần đầu chiếm phần lớn) |
| đóng gói | nén `results/b` thành `vibid_part1_<ngày>_<giờ>.tgz`, đặt cạnh `WORKDIR`, đẩy lên repo kết quả | < 1 phút |

**Theo dõi** từ một cửa sổ khác:

```bash
tail -f $WORKDIR/results/b/ops/ops_part1.log   # mỗi bước bắt đầu bằng "=== <bước> (<thời gian>)"
tail -f $WORKDIR/results/b/b.log               # các giai đoạn bên trong (labels, train, dev, ...)
```

**Thành công** khi `ops_part1.log` có dòng `=== smoke done`, bốn dòng tóm tắt dạng
`<phương pháp>: 30 rows, F1 ..., recall ..., max tokens/budget ...` (cột cuối ≤ 1,05), và dòng cuối là
`Uploaded: vibid_part1_<ngày>_<giờ>.tgz (+ .sha256) to thanthienhai/viword-c-results; nothing to send by hand`.

**Báo** chủ đề tài là đã xong (nếu dòng cuối là `Send back: ...` thì gửi tay hai file, mục 5). **Dừng ở đây, chờ trả lời.**

## 3. Phần 2 (`part2`, khoảng 5–6 giờ, chỉ khi được đồng ý)

Dùng **đúng** `WORKDIR` của phần 1:

```bash
tmux attach -t viword-b                # hoặc: tmux new -s viword-b
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/rag-legal/scripts/ops_b_run.sh   # bản mới nhất
export HF_TOKEN=hf_xxx HF_RESULTS_TOKEN=hf_yyy WORKDIR=/mnt/hps/anhm-paper/viword-b
bash ops_b_run.sh part2
```

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| `update_code_part2` | cập nhật code lên commit mới nhất của `rag-legal` (chỉ fast-forward), cài lại gói nếu cần | 1–3 phút |
| `data_part2` | tải lại và kiểm tra dữ liệu (`data/SHA256SUMS_B` có thể đã đổi cùng code) | 1–2 phút |
| `labels` | reader 7B chấm mức hữu ích của từng Khoản cho 3.654 câu train (~47 nghìn Khoản) | 45–60 phút |
| `train` | huấn luyện 7 scorer từ `BAAI/bge-reranker-v2-m3` (3 seed × 2 loại nhãn, thêm 1 bản kiểm soát) | 1,5–2 giờ |
| `dev` | nén + đọc 390 câu dev: 22 phương pháp × 3 tỉ lệ × 2 kiểu ngữ cảnh, cộng một mốc; viết `results/b/dev.md` | 2–3 giờ |
| đóng gói | `vibid_part2_<ngày>_<giờ>.tgz` (khoảng vài trăm MB, có kết quả từng câu) | 1–2 phút |

**Thành công** khi `ops_part2.log` có `=== dev done`, file `results/b/dev.md` tồn tại, và dòng cuối là `Uploaded: ...`
(hoặc `Send back: ...`, mục 5). **Dừng ở đây, chờ chủ đề tài đăng ký cấu hình rồi báo chạy phần 3.**

## 4. Phần 3 (`part3`, khoảng 1–1,5 giờ, chỉ khi được báo)

Dùng **đúng** `WORKDIR` của hai phần trước:

```bash
tmux attach -t viword-b                # hoặc: tmux new -s viword-b
curl -fsSLO https://raw.githubusercontent.com/thanthienhai/ViWord-C/rag-legal/scripts/ops_b_run.sh   # bản mới nhất
export HF_TOKEN=hf_xxx HF_RESULTS_TOKEN=hf_yyy WORKDIR=/mnt/hps/anhm-paper/viword-b
bash ops_b_run.sh part3
```

| Bước | Việc | Thời gian ước tính |
|---|---|---|
| `update_code_part3` | cập nhật code lên `rag-legal` mới nhất; kiểm tra `config/b_registration.json` đã được commit | 1–3 phút |
| `data_part3` | tải lại và kiểm tra dữ liệu | 1–2 phút |
| `test` | nén + đọc 555 câu test bằng đúng các phương pháp đã đăng ký; viết `results/b/test.md` | 1–1,5 giờ |
| đóng gói | `vibid_part3_<ngày>_<giờ>.tgz` | 1–2 phút |

**Thành công** khi `ops_part3.log` có `=== test done` và dòng cuối là `Uploaded: ...` (hoặc `Send back: ...`).
Phần 3 chỉ chạy **một lần**; không chạy lại khi đã thành công. Việc của đội vận hành dừng ở đây.

## 5. Khi không đẩy được lên Hugging Face

Không có `HF_RESULTS_TOKEN`, hoặc upload lỗi (dòng `upload to ... failed`), thì dòng cuối là
`Send back: <file>.tgz (<cỡ>) and <file>.tgz.sha256`. Gửi **cả hai file** cho chủ đề tài: thiếu `.sha256` thì phía
chủ đề tài không kiểm tra được bản chép. Có thể đẩy lại sau từ máy bất kỳ có code và token:

```bash
HF_TOKEN=$HF_RESULTS_TOKEN python scripts/hf_results.py push <file>.tgz
```

## 6. Khi gặp lỗi

Khi một bước lỗi, script in 40 dòng cuối của log bước đó, **vẫn đóng gói** `vibid_<phần>-FAILED_<ngày>_<giờ>.tgz`
(đẩy lên như gói thường) rồi dừng. Báo chủ đề tài. **Không sửa code tại chỗ.**

Để chạy lại sau khi đã xử lý nguyên nhân (ví dụ đổi GPU) hoặc sau khi chủ đề tài báo đã đẩy bản sửa: tải lại
`ops_b_run.sh` và chạy **đúng lệnh cũ với cùng `WORKDIR`**.
- Mỗi lần chạy lại, code được cập nhật lên commit mới nhất của `rag-legal` (chỉ fast-forward), kể cả ở phần 1.
- Các bước đã xong được bỏ qua (dòng `done in an earlier run, skipped`).
- Trong phần 2: nhãn đã dựng xong và các scorer đã huấn luyện xong được giữ lại; câu trả lời của reader đã có được
  dùng lại, nên chạy lại `dev` nhanh hơn lần đầu.

| Hiện tượng (trong log) | Cách xử lý |
|---|---|
| `no Hugging Face token` | `export HF_TOKEN=hf_xxx` rồi chạy lại |
| `401` / `403` / `Repository Not Found` ở bước `data` | token chưa có quyền đọc `thanthienhai/viword-c-legal-data`. Báo chủ đề tài |
| `... exists but is not a part1 folder of direction B` | `WORKDIR` trỏ vào thư mục có sẵn (ví dụ của ViWord-C). Chọn `WORKDIR` mới |
| `code at rag-legal has no data/SHA256SUMS_B` | chủ đề tài chưa đẩy code. Báo chủ đề tài, chờ, rồi chạy lại |
| `not pushed:` kèm `differs from data/SHA256SUMS_B` hoặc `missing` ở bước `data` | dữ liệu trên Hugging Face chưa khớp với code. Báo chủ đề tài; **không** chạy tiếp bằng cách khác |
| `unit tests` lỗi ở bước `setup` (dòng `failed` trong `setup.log`) | gửi gói `-FAILED` |
| `no GPU visible` | kiểm tra `nvidia-smi` và biến `GPU=<id>` |
| lỗi cài `vllm` / `torch` (xung đột CUDA) | gửi gói `-FAILED` và kết quả `nvidia-smi`; không tự đổi phiên bản |
| lỗi ở `smoke` trong `results/b/logs/smoke_labels.log` (ví dụ `prompt_logprobs`, `TokensPrompt`) | lỗi code của bước dựng nhãn bằng vLLM, đã dự kiến có thể xảy ra. Gửi gói `-FAILED`, chờ chủ đề tài đẩy bản sửa, rồi chạy lại `part1` |
| `CUDA out of memory` ở `smoke`, `labels`, `dev` hoặc `test` | chọn GPU rảnh (`GPU=<id>`) hoặc thêm `VIWORD_VLLM_GPU_UTIL=0.7`, rồi chạy lại |
| `CUDA out of memory` ở `train` | GPU đang bị chia sẻ; chọn GPU rảnh khác rồi chạy lại |
| `missing runs/unit_...: train first` | bước `train` chưa xong; chạy lại `part2` |
| `part1 did not finish: rerun part1` / `part2 did not finish: rerun part2` | phần trước chưa xong; chạy lại phần đó với cùng `WORKDIR` |
| `config/b_registration.json is not committed` hoặc `refusing the test run` | chủ đề tài chưa đăng ký cấu hình. Báo chủ đề tài, chờ, rồi chạy lại `part3` |
| `refusing to let a reader score ... test row(s)` | một bước ngoài phần 3 định cho reader đọc tập test. Không chạy tiếp bằng cách khác; gửi gói `-FAILED` |
| `cannot fast-forward the code` | code trên máy đã bị sửa hoặc lịch sử nhánh bị viết lại. Gửi gói `-FAILED`; không tự `git reset` |
| lỗi tải mô hình từ Hugging Face (timeout, `ConnectionError`) | mạng chập chờn; chạy lại đúng lệnh |
| Pod/máy bị dừng giữa chừng | chạy lại đúng phần đó với cùng `WORKDIR` |
| Lỗi khác (`Traceback`) | gửi gói `-FAILED`. Không sửa code tại chỗ |

Ví dụ chạy lại phần 2 trên GPU 1 với ít bộ nhớ vLLM hơn:

```bash
GPU=1 VIWORD_VLLM_GPU_UTIL=0.7 bash ops_b_run.sh part2
```

## 7. Nội dung các file gửi lại

| File trong `results/b/` | Phần | Nội dung |
|---|---|---|
| `ops/ops_part{1,2,3}.log` | 1–3 | toàn bộ những gì script in ra |
| `ops/<bước>.log` | 1–3 | log đầy đủ của từng bước (`setup.log` có kết quả unit test) |
| `ops/revisions.txt` | 1–3 | phiên bản code (commit git) và dữ liệu (commit Hugging Face) của từng phần |
| `ops/pip_freeze.txt`, `ops/nvidia_smi.txt` | 1 | môi trường, để tái lập |
| `b.log`, `logs/` | 1–3 | log các giai đoạn bên trong (dựng nhãn, huấn luyện, từng lượt đánh giá) |
| `smoke/` | 1 | nhãn và kết quả của lượt chạy thử |
| `unit_*_train_meta.json` | 2 | thông tin huấn luyện 7 scorer (loss, độ chính xác chọn Khoản theo từng epoch) |
| `eval/` | 2, 3 | kết quả từng câu và bản nén của từng phương pháp |
| `dev.md` | 2 | bảng F1 / recall / EM trên dev và các phép so sánh (chủ đề tài xem file này trước tiên) |
| `test.md` | 3 | bảng tương ứng trên test |

Nhãn (`data/labels/`) và các scorer (`runs/`) không nằm trong gói; giữ nguyên trong `WORKDIR` cho tới khi chủ đề tài
báo xong.

## Phụ lục: việc của chủ đề tài giữa các phần

Đội vận hành không cần đọc phần này.

**Trước phần 1.** Cấp cho token đọc của đội vận hành quyền đọc `thanthienhai/viword-c-legal-data` (token
fine-grained: thêm repo này vào danh sách). Code và dữ liệu đã đẩy ngày 26/10 (`rag-legal`, dữ liệu `e40f36f`).
Nếu đổi dữ liệu: cập nhật `data/SHA256SUMS_B`, đẩy dữ liệu **trước**
(`python scripts/hf_data.py push --repo thanthienhai/viword-c-legal-data --sums data/SHA256SUMS_B`), rồi mới push git.

**Nhận gói của mỗi phần** (kiểm tra checksum khi tải):

```bash
python scripts/hf_results.py list                        # các gói đã về
python scripts/hf_results.py pull --match vibid_part1    # vào results/from_gpu/, dừng nếu checksum sai
```

**Sau phần 1.** Xem `ops_part1.log` và `results/b/b.log`: bốn dòng tóm tắt của lượt chạy thử, tốc độ thực tế trên
H100. Nếu `smoke` lỗi, sửa code, push `rag-legal`, rồi báo đội vận hành chạy lại `part1`.

**Sau phần 2, trước phần 3.** Đọc `dev.md` (và đối chiếu gate G1 của đề cương). Chọn phương pháp đề xuất, các
baseline, tỉ lệ chính và biên H1b; chép `config/b_registration.example.json` thành `config/b_registration.json`, sửa
theo kết quả dev, commit và push `rag-legal`. Sau khi phần 3 chạy, không sửa file này nữa.
