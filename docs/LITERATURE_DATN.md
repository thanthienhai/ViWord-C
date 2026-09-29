# Tài liệu liên quan cho DATN (ViWord-C)

Tìm ngày 2026-09-29, theo các nhóm từ khóa A–G. Nguồn: Semantic Scholar Graph API (nhóm A; kết quả thô trong
`docs/s2_search/results.json`, script `docs/s2_search/search.py`); web search giới hạn trong
semanticscholar.org / arxiv.org / aclanthology.org (nhóm B–G, vì API không có key bị giới hạn 429 liên tục);
abstract lấy từ arXiv API. Mức độ: ★★★ = phải đọc kỹ / ảnh hưởng thiết kế, ★★ = trích dẫn + so sánh,
★ = trích dẫn bối cảnh.

## ★★★ Sát nhất — đọc trước

| Bài | Nội dung | Ảnh hưởng tới DATN |
|---|---|---|
| **Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors** (arXiv 2608.26175, 07/2026) | Kiểm tra 4 bộ nén học được (LLMLingua-2 XLM-R/mBERT, Kompress-v2, XProvence) và 4 baseline tất định trên 10 ngôn ngữ (EN, PL, LV, FI, ET, LT, UK, **ZH**, AR, HI; **không có tiếng Việt**), ngân sách khớp theo tokenizer đích, 11 LLM đích, Belebele. Ở keep-rate 0.33, tiếng Anh giữ 57–62% utility, **tiếng Trung gần như 0**. Nguyên nhân: không có khoảng trắng ngăn từ, WordPiece cắt chữ Hán, và **bỏ hư từ ngữ pháp** mà tiếng Trung (đơn lập) phụ thuộc. Khoảng cách này do dữ liệu giám sát, không do kiến trúc. **Không đề xuất cách sửa kỹ thuật**, chỉ đưa khuyến nghị triển khai | Xác nhận P1 + P2 trên một ngôn ngữ đơn lập khác → phần motivation mạnh. Là **công trình gần nhất**: ViWord-C trở thành "cách sửa kỹ thuật cho ngôn ngữ đơn lập mà bài này chỉ ra, trên tiếng Việt". Mượn giao thức ngân sách khớp theo tokenizer đích, baseline tất định, translate-then-compress và Belebele-vi. Do "huấn luyện bằng dữ liệu bản ngữ là đủ" là phát hiện của họ, đối chứng **LLMLingua-2-vi** trở thành bắt buộc |
| **LLMLingua-2** (Findings ACL 2024, arXiv 2403.12968; 416 trích dẫn) | Chưng cất nén trích xuất từ GPT-4 → token classification bằng XLM-R | khung huấn luyện gốc, baseline chính |
| **Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors** (arXiv 2607.25335) | Tìm tổ hợp luật ngôn ngữ (từ vựng, cú pháp, ngữ nghĩa, diễn ngôn) bằng tìm kiếm tiến hoá. Chạy trên CPU, không cần LM, ngang các bộ nén học được ở mức nén nhẹ–vừa | Ủng hộ việc dùng tri thức ngôn ngữ; là baseline "chỉ luật" để so với ưu tiên bảo vệ π của ta. Tiếng Anh |
| **Parse Trees Guided LLM Prompt Compression** (PartPrompt, arXiv 2409.15395) | Nén theo cây cú pháp; chỉ ra LLMLingua giữ **từ không trọn vẹn** | bằng chứng P1 cho tiếng Anh |

## ★★ Nén prompt — baseline và so sánh

| Bài | Ghi chú |
|---|---|
| LLMLingua (EMNLP 2023, 2310.05736) / LongLLMLingua (ACL 2024, 2310.06839) | nén theo perplexity |
| Selective Context — *Compressing Context to Enhance Inference Efficiency of LLMs* (EMNLP 2023) | self-information |
| **Lexical Prompt Compression … Training-Free, Deterministic Pipeline** (arXiv 2609.13154) | 11 phép biến đổi từ vựng (bỏ stopword, lọc theo POS, **giữ thực thể tên**, …); chỉ bỏ stopword đã giảm 29.6% token mà BERTScore vẫn 0.913. Baseline tất định mạnh; tiếng Anh |
| Selection-p (EMNLP 2024, 2410.11786) | tự giám sát, không cần GPT-4 → có thể làm nguồn nhãn khác |
| DAC: Dynamic Attention-aware Task-Agnostic Prompt Compression (ACL 2025, 2507.11942) | entropy + attention |
| Task-agnostic Prompt Compression with Context-aware Sentence Embedding (TMLR 2025, 2502.13374) | cấp câu |
| EFPC (2503.07956), PROMPT-SAW (2404.00489), TACO-RL (ACL 2024, 2409.13035) | biến thể nén |
| Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor (2511.12281) | nén trừu tượng bằng LLM nhỏ (Qwen3-4B + GRPO) — khác hướng (viết lại) |
| **Fundamental Limits of Prompt Compression: A Rate-Distortion Framework** (2407.15504) | khung rate–distortion cho nén token → trích dẫn cho §3.5 |
| Understanding and Improving Information Preservation in Prompt Compression (2503.19114) | độ đo bảo toàn thông tin / thực thể → thêm "entity retention" |
| Prompt Compression for LLMs: A Survey (NAACL 2025, 2410.12388) | tổng quan |
| An Empirical Study on Prompt Compression for LLMs (2505.00019) + PCToolkit (2403.17411) | toolkit chạy sẵn 6 phương pháp, có phân tích **bỏ từ** (word omission) → dùng để chạy baseline nhanh |
| Compression Method Matters: Benchmark-Dependent Output Dynamics (2603.23527), CAVEWOMAN (2606.24083) | nén đầu vào có thể làm **đầu ra dài ra** → báo cáo độ dài đầu ra |
| Prompt Compression in the Wild: Latency, Rate Adherence, Quality (2604.02985) | đo độ trễ và mức bám tỉ lệ nén |
| Prompt Compression in Diffusion LLMs: LLMLingua-2 on LLaDA (2605.17932) | lỗi chủ yếu do **bỏ sót thông tin**, không do lệch ngữ nghĩa |

## ★★ Phủ định / hư từ (H2)

| Bài | Ghi chú |
|---|---|
| An Analysis of NLI Benchmarks through the Lens of Negation (Hossain et al., ACL 2020) | phủ định quan trọng trong NLI |
| This is not a Dataset: A Large Negation Benchmark to Challenge LLMs (EMNLP 2023, 2310.15941) | LLM kém nhạy với phủ định |
| Language models are not naysayers (Truong et al., *SEM 2023) | phân tích LM trên benchmark phủ định |
| ScoNe: Benchmarking Negation Reasoning (2305.19426) | benchmark |
| Strong hallucinations from negation and how to fix them (2402.10543); Making LMs Robust Against Negation (2502.07717) | bối cảnh |

## ★★ Tokenizer / fertility (H3)

| Bài | Ghi chú |
|---|---|
| Language Model Tokenizers Introduce Unfairness Between Languages (Petrov et al., NeurIPS 2023) | chênh lệch độ dài tới 15× |
| Do All Languages Cost the Same? (Ahia et al., EMNLP 2023) | chi phí API theo ngôn ngữ |
| The Token Tax: Systematic Bias in Multilingual Tokenization (2509.05486) | fertility dự đoán độ chính xác |
| How does a Language-Specific Tokenizer affect LLMs? (2502.12560) | tokenizer riêng theo ngôn ngữ |
| SeaLLMs (2312.00738) | mở rộng vocab cho ngôn ngữ Đông Nam Á |

## ★ Tài nguyên tiếng Việt

| Bài | Dùng cho |
|---|---|
| VnCoreNLP (NAACL 2018 demo, 1801.01331); A Fast and Accurate Vietnamese Word Segmenter — RDRSegmenter (LREC 2018, 1709.06307) | tách từ, NER |
| Is word segmentation necessary for Vietnamese sentiment classification? (2301.00418) | rủi ro 2 (tách từ có cần không) |
| PhoBERT (Findings EMNLP 2020) | encoder |
| ViNLI (COLING 2022) — >30k cặp, **4 nhãn** (entailment / neutral / contradiction / other) | tác vụ H2 |
| ViMMRC 2.0 (2303.18162) — 699 đoạn, 5,273 câu hỏi | đọc hiểu |
| VietNews (Nguyen et al. 2019; dùng trong ViT5, NAACL-SRW 2022) — ~150k bài | tóm tắt |
| Crossing Linguistic Horizons: Finetuning and Evaluation of Vietnamese LLMs (Findings NAACL 2024, 2403.02715) | danh sách LLM tiếng Việt |
| Belebele (vie_Latn) | cầu nối trực tiếp với *Lost in Compression* |

## Kiểm tra độ mới

Không tìm thấy bài nén prompt/ngữ cảnh nào cho **tiếng Việt** (các truy vấn nhóm E chỉ trả về bài tổng quát;
DeepEdu-v1 (2609.31568) là sparse attention cho giáo dục, không phải nén prompt). *Lost in Compression* là bài
đa ngôn ngữ gần nhất: không có tiếng Việt và không đề xuất phương pháp. → Vẫn cần tra thêm trên Google Scholar
và kỷ yếu VLSP/RIVF/KSE/SoICT (Semantic Scholar không phủ hết các hội nghị trong nước).
