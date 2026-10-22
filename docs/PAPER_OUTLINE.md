# Khung bài báo ViWord-C (bản phác 22/10/2026)

Bản phác này viết **trước** lần chạy chính thức, để cách kể của bài không phụ thuộc vào việc ViWord-C thắng.
Smoke 21/10 cho thấy H1 có thể gần 0, H2 chưa rõ, và LLMLingua-2 bản công bố có lúc ngang bản chưng cất
(DATN §7, rủi ro 1 và 6). Vì vậy bài đặt trọng tâm vào **chẩn đoán** và **câu trả lời có kiểm soát** của thiết kế
2×2. ViWord-C là công cụ của thí nghiệm đó, không phải lời hứa vượt mọi baseline. Số liệu chỉ điền sau part3;
các ô `[…]` dưới đây là chỗ trống, không phải kết quả.

## Luận điểm

Các bộ nén prompt hiện có cắt đôi từ ghép và bỏ hư từ mang nghĩa trong tiếng Việt ở mức đo được (CBR, tỉ lệ giữ
hư từ, NFR). Với **cùng** nhãn chưng cất bản ngữ, **cùng** encoder và **cùng** bước chọn, chuyển đơn vị quyết định
từ âm tiết sang từ [có / không] cải thiện điểm tác vụ. Đây là câu hỏi *Lost in Compression* để ngỏ: khi đã có nhãn
bản ngữ, đơn vị quyết định còn quan trọng không.

## Bốn kịch bản kết quả và cách viết

Kết quả nào cũng là một phát hiện (DATN §9, đóng góp 2). Chọn kịch bản theo phân tích chính
(`results/analysis_3seed.md`, phần "Primary analysis"); phân tích độ nhạy trên toàn bộ tập test báo cáo cạnh đó.

| H1 (đơn vị từ) | H2 (bảo vệ π) | Bài nói gì | Độ dài, nơi nộp (DATN §8) |
|---|---|---|---|
| có hiệu ứng | có hiệu ứng | ViWord-C sửa được cả hai vấn đề; đóng góp phương pháp | 8 trang (thêm H3): PACLIC, KSE, ACM TALLIP |
| không | có hiệu ứng | Khi có nhãn bản ngữ, đơn vị không còn quan trọng; bảo vệ hư từ có khử nhập nhằng thì có | 4–6 trang: RIVF, SoICT, FAIR |
| có hiệu ứng | không | Ranh giới từ quan trọng; NFR không nhạy với π trên premise ngắn | 4–6 trang |
| không | không | Benchmark chẩn đoán đầu tiên cho tiếng Việt và một kết quả âm tính có lực thống kê (MDE báo trước) | 4 trang: workshop VLSP, workshop ngôn ngữ ít tài nguyên, SRW |

Hai biến thể cắt ngang cả bốn hàng, đọc từ hàng tham chiếu của `ckpt_check` (bổ sung 22/10):

- **Bản chưng cất không hơn LLMLingua-2 bản công bố:** nói thẳng rằng nhãn thầy là nút thắt. Nếu thầy (cắt về
  ngân sách) cũng không hơn bản công bố thì học trò khó vượt; khi đó đóng góp nằm ở chẩn đoán và 2×2, không ở bộ nén.
- **Truncation hoặc Lead-k thắng trên VietNews:** đã lường trước (thiên lệch phần mở đầu, DATN §4.1); VietNews chỉ
  dùng cho non-inferiority của H2.

## Cấu trúc bài

1. **Mở đầu.** Nén prompt và chi phí token; tiếng Việt là ngôn ngữ đơn lập, mảnh của một từ thường là một từ hợp
   lệ khác (*học sinh → học*); khoảng trống: *Lost in Compression* không có tiếng Việt và không đề xuất cách sửa;
   đóng góp theo kịch bản ở trên.
2. **Công trình liên quan.** LLMLingua, LongLLMLingua, LLMLingua-2, Selective Context, PartPrompt (từ không trọn vẹn
   ở tiếng Anh), *Lost in Compression*, các baseline chỉ luật và từ vựng (DATN §1). Trước khi viết "đầu tiên":
   tra lại Google Scholar và kỷ yếu VLSP, RIVF, KSE, SoICT (skill `lit-review`, không bịa nguồn).
3. **Chẩn đoán (DATN §4.0).** CBR có mốc ngẫu nhiên theo phân phối số âm tiết, RR, tỉ lệ giữ T1/T2/T3, fertility
   và CV_T của sáu tokenizer, probe phủ định. Tách theo loại bộ nén (học nhãn tiếng Anh, perplexity, tất định).
4. **Phương pháp.** Thiết kế 2×2 {nhãn âm tiết, từ} × {quyết định âm tiết, từ} trên cùng đầu ra thầy và XLM-R-large;
   ViWord-C: đơn vị từ (RDRSegmenter), ưu tiên π có khử nhập nhằng; H3 (chi phí tokenizer đích) chỉ ở bản 8 trang.
5. **Thiết lập thực nghiệm.** Bốn tác vụ chính và ba tác vụ khám phá, hai reader, ngân sách đếm bằng tokenizer
   của reader, baseline §4.2, bootstrap theo cụm và seed, Holm theo họ, MDE tính trước trên dev, sửa đổi 22/10
   (phân tích chính bỏ các dòng test reader đã chấm trong lúc phát triển).
6. **Kết quả.** H1 trên họ {ViMMRC, Belebele} × {1/2, 1/3, 1/5}; hiệu ứng chính và tương tác của 2×2; hồi quy mức
   tăng theo CBR (dự đoán cơ chế). H2: NFR trên ViNLI, tách premise có phủ định; non-inferiority ROUGE-L VietNews.
7. **Phân tích.** H2 theo tầng; độ nhạy theo bộ tách từ; độ dài đầu ra của reader; reader thứ hai; độ trễ nén.
8. **Hạn chế.** Mục dưới.

## Bảng và hình, lấy từ đâu

| Trong bài | File của pipeline | Có sau |
|---|---|---|
| Bảng 1: chẩn đoán CBR, RR, tỉ lệ giữ hư từ, fertility | `results/diagnose/{belebele,vinli}.json`, `gate_report.md` | part2 (diagnose sau train) |
| Hình 1: `force_tokens=["không"]` cấp âm tiết so với bảo vệ cấp từ (*không khí*, *hàng không*) | ví dụ định tính từ `results/eval/vinli_*.jsonl` | part3 |
| Bảng 2: H1, tương phản `viword − llmlingua2vi`, Holm | `results/analysis_3seed.md`, phần H1 | part3 |
| Bảng 3: 2×2, hiệu ứng đơn vị, nhãn, tương tác | như trên, mục "2x2 design" | part3 |
| Bảng 4: H2, NFR và non-inferiority | `results/analysis_3seed.md`, phần H2 | part3 |
| Mốc tham chiếu: LLMLingua-2 bản công bố và thầy trên dev | `results/ckpt_check/report.md` | part1 |
| Phụ lục: MDE, lề non-inferiority, SD seed | `results/mde_dev/report.md` | part2 |
| Phụ lục: BERTScore VietNews | `results/bertscore/*.md` (máy chủ đề tài) | sau part3/part4 |
| Phụ lục: reader thứ hai | `results/analysis_3seed.md` (dòng SeaLLM-7B-v2, mô tả) | part4 |

## Hạn chế cần viết (đã biết từ bây giờ)

- Prompt thầy dặn giữ phủ định, số và tên riêng, nên H2 đo lợi ích **gia tăng** của π trên một encoder đã học prior
  này (DATN §2.3).
- Hai phiếu mù do LLM gợi ý nhãn, người duyệt chấp nhận: có thể nghiêng theo nhãn gợi ý; G6 đạt với biên một câu.
- Reader đã chấm một số dòng test trong lúc phát triển (smoke 04/10 và các lần sau); phân tích chính bỏ chúng,
  phân tích độ nhạy giữ (DATN §4.3, sửa đổi 22/10/2026).
- Premise ViNLI chỉ dài một câu; ở 1/3 và 1/5 chỉ còn vài token, nên NFR đo cả việc mất thông tin nói chung.
- Hai reader 7B; bản 8 trang thêm reader khác.
- CBR của ViWord-C bằng 0 theo cách xây dựng khi đo bằng RDRSegmenter; báo cáo thêm khi đo bằng bộ tách từ khác.

## Viết được ngay, và viết sau part3

- **Ngay:** mục 1, 2, 4, 5 và Hạn chế; bản nháp Bảng 1 từ part1 (G1, G2, G5), hoàn chỉnh sau part2.
- **Sau part3:** chọn kịch bản theo bảng ở trên, rồi mục 6, 7 và tóm tắt.
