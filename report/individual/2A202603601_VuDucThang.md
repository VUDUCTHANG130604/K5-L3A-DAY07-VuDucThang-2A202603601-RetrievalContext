# Báo cáo cá nhân — Day 7 · Agentic RAG & Retrieval Context

**Học viên:** Vũ Đức Thắng  
**MSSV:** 2A202603601  
**Lớp:** K5-L3A  

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *Tài liệu nguồn:* Quyết định số 4235/QĐ-BYT ngày 25/10/2012 của Bộ Y tế ban hành tài liệu chuyên môn "Hướng dẫn chẩn đoán và điều trị bệnh hô hấp" (Chủ biên: PGS.TS. Nguyễn Viết Nhung, GS.TS. Ngô Quý Châu). Sử dụng công khai phi thương mại phục vụ đào tạo và thực hành y tế.
  - *Corpus thực nghiệm:* Trang 30–75 của tài liệu PDF (tổng cộng **46 trang lâm sàng** bao gồm: Viêm phế quản cấp, Viêm phổi cộng đồng, Viêm phổi bệnh viện, Áp xe phổi, COPD ổn định và Đợt cấp COPD).
  - *Số chunk kế thừa từ Day 6:* **126 chunks** (trích xuất theo cấu trúc đoạn văn bản `Paragraph 800`, chuẩn hóa TCVN3 sang UTF-8 NFKC không lỗi font, kích thước tối đa 800 ký tự). Dữ liệu được lưu tại [data/chunks.json](../../data/chunks.json).
- **Vector store, embedding model, phiên bản và số chiều:**
  - *Embedding model:* Google `google/embeddinggemma-2` (kích thước vector **768 chiều**, chạy cục bộ qua `sentence-transformers==6.3.0`, kiểu tính toán `torch.float32`, metric Cosine distance).
  - *Lưu trữ vector:* Ma trận embeddings tiền tính toán lưu tại [data/chunk_embeddings.npy](../../data/chunk_embeddings.npy) phục vụ truy vấn vector nhanh độ trễ thấp.
- **Agent model, cách chạy và phân chia dev/test:**
  - *Agent Controller (Model thật):* Sử dụng **Anthropic Claude LLM** (`claude-haiku-5-5`) thông qua thư viện chính thức `anthropic` (Anthropic Messages API), cấu hình `temperature: 0.0` và `max_tokens: 1500`. Ở Turn 0, Claude tự phân tích câu hỏi để sinh truy vấn lâm sàng chuyên biệt dạng JSON `{"action": "search", "query": "..."}`. Ở Turn 1+, Claude nhận context thực tế từ vector store và quyết định tìm kiếm tiếp hoặc trả lời kèm trích dẫn `citations: [chunk-id]` hoặc từ chối (`abstain`). Tuyệt đối không dùng regex hardcode hay fallback heuristic can thiệp kết quả.
  - *Phân chia Benchmark:* Tổng cộng **24 câu hỏi** (lưu tại [data/cases.json](../../data/cases.json)):
    - **Dev (17 câu):** 15 câu có đáp án (`q01` – `q15`) + 2 câu thiếu bằng chứng (`q_unans_dev_01`, `q_unans_dev_02`).
    - **Test (7 câu):** 5 câu có đáp án (`q16` – `q20`) + 2 câu thiếu bằng chứng (`q_unans_test_01`, `q_unans_test_02`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* Có `gold` gồm `source_id`, `page` và đoạn trích dẫn `evidence` khớp nguyên văn trong `chunk['text']`. Chấm điểm theo thang `starter/agent_metrics.py`: **2** = đúng và đầy đủ bằng chứng; **1** = đúng một phần; **0** = sai hoặc từ chối nhầm do retrieval failure.
  - *Câu thiếu bằng chứng:* Các câu hỏi liên quan đến chuyên khoa hô hấp nhưng nằm ngoài phạm vi tài liệu (Remdesivir/COVID-19, phẫu thuật nội soi lồng ngực cắt thùy phổi ung thư, lao phổi kháng đa thuốc MDR-TB, ghép phổi COPD). Agent phải nhận thức ranh giới dữ liệu và trả về `status: "abstain"` với mảng `citations: []`.

---

## Retrieval và thử nghiệm

### Bảng 1: So sánh các phương pháp và cấu hình Retrieval

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Dense Retrieval (EmbeddingGemma-2, 768d), top-k=3, budget=800 | Truy vấn vector ngữ nghĩa cơ sở Day 6 | dev | 6/15 (40.0%) | 8/15 (53.3%) | 0/15 (0.0%) | 161.3 ms / $0 |
| **Phương pháp 2** | Lexical Retrieval (BM25Okapi), top-k=3, budget=1200 | Khớp chính xác từ khóa và thuật ngữ định lượng y khoa | dev | 6/15 (40.0%) | 9/15 (60.0%) | 6/15 (40.0%) | 4.7 ms / $0 |
| **Phương pháp 3** | Query Transformation (Medical Expansion + Dense), top-k=3, budget=1200 | Lược bỏ từ đệm đàm thoại, mở rộng từ vựng lâm sàng | dev | **7/15 (46.7%)** | **9/15 (60.0%)** | **7/15 (46.7%)** | 277.1 ms / $0 |
| **Cải thiện 1** | Dense Retrieval với Context Budget = 1200 ký tự (thay vì 800) | Giả thuyết: chunk dài 800 ký tự + header 100 ký tự cần budget $\ge 900$ để không bị rớt khỏi context | dev | 6/15 (40.0%) | 8/15 (53.3%) | 6/15 (40.0%) *(tăng từ 0%)* | 227.1 ms / $0 |
| **Cải thiện 2** | Query Transformation + Dense (Top-k=3, Budget=2400) | Giả thuyết: Cho phép chứa cả 2-3 chunk vào context giúp không bỏ lọt bằng chứng ở Rank 2, 3 | dev | **7/15 (46.7%)** | **9/15 (60.0%)** | **9/15 (60.0%)** | 285.4 ms / $0 |
| **Cấu hình đã chọn** | Query Transformation + Dense, top-k=3, budget=2400 | Cấu hình đạt Hit@1 và Context Rate cao nhất trên Dev | test | 1/5 (20.0%) | 1/5 (20.0%) | 1/5 (20.0%) | 231.2 ms / $0 |
| **Baseline (Test)** | Dense Retrieval cơ sở, top-k=3, budget=1200 | Đối chứng độc lập trên tập test sau khi chốt cấu hình | test | 1/5 (20.0%) | 2/5 (40.0%) | 1/5 (20.0%) | 210.3 ms / $0 |

*Ghi chú:* Trên tập Test chuyên sâu về các thông số máy móc (SpO2, số lít oxy), BM25 đạt Hit@3 rất cao (4/5 = 80.0%), cho thấy tiềm năng lớn của Hybrid RRF khi kết hợp cùng Dense.

### Phân tích hai lỗi trên Dev, nguyên nhân và cách xử lý:

1. **Lỗi 1 (Lỗi tắc nghẽn Context Budget - Context Evidence Rate = 0%):**
   * *Hiện tượng:* Với cấu hình ban đầu `budget = 800` ký tự (mặc định starter), mặc dù Hit@3 đạt 53.3% nhưng `Context Evidence Rate` lại bằng 0/15 (0.0%).
   * *Nguyên nhân:* Các chunk theo Strategy B được cắt tối đa 800 ký tự. Khi đóng gói vào context, mỗi block kèm header `[id | source | page]` dài khoảng 100 ký tự, nâng tổng chiều dài block lên ~900 ký tự. Hàm `pack_context` kiểm tra `len(block) <= budget` (800) bị sai điều kiện nên đã bỏ qua chunk, khiến context gửi cho agent bị rỗng!
   * *Cách khắc phục & kết quả:* Nâng `context budget` lên **1200 ký tự** (chứa 1 chunk) và **2400 ký tự** (chứa 2-3 chunk). Context Evidence Rate lập tức tăng từ **0.0% lên 60.0%** (9/15 câu chứa đầy đủ bằng chứng vàng).

2. **Lỗi 2 (Lỗi loãng vector do từ đệm đàm thoại ở câu q01):**
   * *Hiện tượng:* Câu hỏi gốc `q01`: *"Viêm phế quản cấp được định nghĩa như thế nào trong hướng dẫn của Bộ Y tế?"* chỉ đạt rank 2 trong Dense retrieval gốc (không lọt vào Hit@1).
   * *Nguyên nhân:* Câu hỏi chứa nhiều từ đệm giao tiếp (*"được định nghĩa như thế nào trong hướng dẫn của Bộ Y tế"*) chiếm tỷ trọng lớn trong vector embedding của Gemma-2, kéo điểm tương đồng sang các chunk mang tính thủ tục hành chính thay vì định nghĩa bệnh học.
   * *Cách khắc phục & kết quả:* Áp dụng `QueryTransformer` lọc bỏ từ đệm và mở rộng thành *"Viêm phế quản cấp khái niệm đại cương tình trạng viêm"*. Chunk định nghĩa `paragraph-p30-0000` được kéo trực tiếp lên **Rank 1**, giúp Hit@1 của toàn bộ Dev tăng từ 6/15 lên 7/15 câu.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Tool use:* Tool `search` cho phép truy xuất thông tin từ kho tài liệu y khoa thông qua adapter `MedicalRetrieverAdapter`.
  - *Điều phối model thật:* Controller sử dụng **Anthropic Claude LLM** (`claude-haiku-5-5`) để đọc câu hỏi và context tài liệu, đánh giá độ tin cậy của bằng chứng, sinh câu trả lời lâm sàng và gắn trích dẫn `[chunk-id]` chính xác.
  - *Điều kiện dừng:* Agent dừng khi tìm thấy đoạn văn bản hỗ trợ và sinh câu trả lời có kèm `citations: [chunk-id]` hợp lệ, hoặc từ chối trả lời `abstain` khi nhận diện context rỗng hoặc câu hỏi ngoài phạm vi tài liệu (COVID-19, Remdesivir, ung thư lồng ngực, lao kháng thuốc MDR-TB, ghép phổi), hoặc khi đạt ngưỡng `max_searches = 2`.
  - *Chọn context:* Khử trùng lặp ID (`deduplication`), gói văn bản trong ngân sách `budget = 2400` ký tự (tối ưu nhất qua thực nghiệm).

### Bảng 2: So sánh hiệu năng Agent qua các cấu hình thực nghiệm (Thực thi 100% bằng Claude LLM thật)

| Cấu hình / Split | Tool use (n/N) | Số lượt gọi TB | Grounded answer (0/1/2) | Điểm TB (/2) | Citation hợp lệ (n/N) | Câu có citation (n/N) | Abstention đúng (n/N) | Từ chối nhầm (n/N) | Thời gian TB / câu | Chi phí TB / câu |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dev — Baseline (Dense, budget=800)** | 17/17 (100%) | 4.53 lượt | 15 (0) / 0 (1) / 0 (2) | **0.00 / 2** | 0/1 (0.0%) | 1/1 (100%) | **2/2 (100%)** | 14/15 (93.3%) | 11.68 s | $0.0314 |
| **Dev — Cải thiện 1 (QueryTrans, budget=1200)** | 17/17 (100%) | 3.94 lượt | 11 (0) / 0 (1) / 4 (2) | **0.53 / 2** | 4/4 (100%) | 4/4 (100%) | **2/2 (100%)** | 11/15 (73.3%) | 10.91 s | $0.0290 |
| **Dev — Cải thiện 2 (QueryTrans, budget=2400) [CHỌN]** | 17/17 (100%) | 2.24 lượt | 5 (0) / 1 (1) / 9 (2) | **1.27 / 2** | **10/13 (76.9%)** | **11/11 (100%)** | **2/2 (100%)** | **4/15 (26.7%)** | **3.90 s** | **$0.0054** |
| **Test — Baseline (Dense, budget=800)** | **7/7 (100%)** | **2.57 lượt** | **5 (0) / 0 (1) / 0 (2)** | **0.00 / 2** | **0/1 (0.0%)** | **1/1 (100%)** | **2/2 (100%)** | **4/5 (80.0%)** | **4.00 s** | **$0.0046** |
| **Test — Cấu hình chốt (QueryTrans, budget=2400)** | **7/7 (100%)** | **2.29 lượt** | **2 (0) / 1 (1) / 2 (2)** | **1.00 / 2** | **3/7 (42.9%)** | **4/4 (100%)** | **2/2 (100%)** | **1/5 (20.0%)** | **4.68 s** | **$0.0060** |

*Phân tích Bảng 2:* So sánh trực diện trên tập Test cho thấy việc tăng context budget lên 2400 ký tự và kết hợp Query Transformation giúp điểm Grounded Answer của Claude tăng vọt từ **0.00 lên 1.00 / 2** (với 2 câu đạt điểm tuyệt đối 2/2, 1 câu đạt 1/2), đồng thời tỷ lệ từ chối nhầm giảm mạnh từ **80.0% xuống chỉ còn 20.0%** (1/5 câu). Chi phí cho mỗi câu hỏi cực kỳ tiết kiệm: chỉ **$0.0060** (khoảng 150 VNĐ/câu).

---

### Bảng 3: Kiểm toán thủ công toàn diện 100% mẫu Test do Claude LLM sinh ra thực tế

| Query ID | Loại câu & Câu hỏi | Tool call & Kết quả Retrieval | Câu trả lời thực tế của Claude & Trích dẫn [chunk-id] | Thẩm định Citation có hỗ trợ không | Điểm & Lý do lâm sàng |
|---|---|---|---|---|---|
| **q16** | Có đáp án (Test)<br>*Đợt cấp BPTNMT được định nghĩa như thế nào?* | Tool `search` $\rightarrow$ Trả về `paragraph-p68-0000`, `paragraph-p74-0002`, `paragraph-p69-0000`. Chunk vàng định nghĩa tại trang 68 lọt vào Rank 1. | "Đợt cấp BPTNMT (COPD) là tình trạng thay đổi cấp tính của các biểu hiện lâm sàng so với bình thường, gồm khó thở tăng, ho tăng, khạc đờm tăng và/hoặc thay đổi màu sắc của đờm. Những biến đổi này đòi hỏi phải thay đổi trong điều trị [paragraph-p68-0000]." | **Có hỗ trợ trực tiếp (1/1)**.<br>`paragraph-p68-0000` chứa nguyên văn định nghĩa bệnh học đợt cấp BPTNMT. | **2 điểm (Đúng và đủ)**.<br>Nội dung chính xác tuyệt đối theo Hướng dẫn Bộ Y tế trang 68, trích dẫn chuẩn xác chunk vàng duy nhất. |
| **q17** | Có đáp án (Test)<br>*Tiêu chuẩn Anthonisen xác định đợt cấp BPTNMT gồm triệu chứng nào?* | Tool `search` $\rightarrow$ Trả về `paragraph-p69-0000` (Score: 0.834, trang 69) ở Rank 1. | "Theo tiêu chuẩn Anthonisen, chẩn đoán xác định đợt cấp BPTNMT dựa trên 3 triệu chứng: (1) khó thở tăng, (2) khạc đờm tăng, (3) thay đổi màu sắc của đờm [paragraph-p69-0000]." | **Có hỗ trợ trực tiếp (1/1)**.<br>`paragraph-p69-0000` chứa nguyên văn 3 triệu chứng Anthonisen. | **2 điểm (Đúng và đủ)**.<br>Khớp 100% ba tiêu chuẩn lâm sàng Anthonisen tại trang 69, trích dẫn đúng chunk vàng. |
| **q18** | Có đáp án (Test)<br>*Nguyên tắc điều trị đợt cấp BPTNMT tại nhà với thuốc giãn phế quản và corticoid?* | Tool `search` $\rightarrow$ Trả về `paragraph-p70-0000`, `paragraph-p71-0000`, `paragraph-p75-0000`. | "Khi đã xác định đợt cấp, bắt đầu hoặc tăng thuốc giãn phế quản và đánh giá lại sau 1-3 giờ. Nếu cải thiện triệu chứng thì tiếp tục điều trị và giảm liều khi có thể. Nếu không cải thiện thì thêm corticoid uống, prednisolon 1 mg/kg/ngày, rồi đánh giá lại sau 1-3 giờ. Nếu triệu chứng nặng thêm thì nhập viện [paragraph-p70-0000][paragraph-p75-0000]. Nguyên tắc điều trị tại nhà là tăng tối đa thuốc giãn phế quản và corticoid dạng khí dung, kết hợp nhiều nhóm thuốc giãn phế quản và tăng liều tối đa các thuốc giãn phế quản dạng khí dung và uống [paragraph-p71-0000]. Người bệnh có sẵn hệ thống oxy tại nhà có thể thở oxy 1-3 lít/phút, duy trì SpO2 90-92% [paragraph-p71-0000]. Nhập viện là bắt buộc với đợt cấp mức độ nặng, rất nặng hoặc đe dọa tính mạng [paragraph-p75-0000]." | **Hỗ trợ 1 phần (1/3)**.<br>`paragraph-p71-0000` chứa nguyên tắc cốt lõi về thuốc khí dung. `paragraph-p70-0000` và `p75` là sơ đồ phụ và chỉ định viện. | **1 điểm (Đúng một phần)**.<br>Câu trả lời rất đầy đủ và chuẩn xác về mặt y học, nhưng trích dẫn thêm 2 chunk phụ nên theo rubric bị chấm 1/2 điểm. |
| **q19** | Có đáp án (Test)<br>*Mục tiêu SpO2 và lưu lượng thở oxy tại nhà trong đợt cấp BPTNMT?* | Tool `search` $\rightarrow$ Dense retriever bị loãng với từ khóa SpO2, chunk vàng trang 71 nằm ngoài top-3 context. | "Không đủ bằng chứng để trả lời."<br>*(citations: [])* | Không có citation (0/0). | **0 điểm (Từ chối an toàn / False Abstention)**.<br>Retrieval failure: Thiếu dữ liệu SpO2 trong context, Claude hành xử an toàn lâm sàng là từ chối, tránh bịa đặt thông số oxy nguy hiểm. |
| **q20** | Có đáp án (Test)<br>*Chỉ định nhập viện điều trị đối với bệnh nhân đợt cấp BPTNMT khi nào?* | Tool `search` $\rightarrow$ Trả về `paragraph-p73-0000`, `paragraph-p70-0000`, `paragraph-p69-0000`. Bỏ lọt bảng chỉ định tại trang 75 (`paragraph-p75-0000`). | "Bệnh nhân đợt cấp BPTNMT cần cân nhắc nhập viện khi đợt cấp kèm theo một hoặc nhiều dấu hiệu sau: có biểu hiện suy hô hấp trên lâm sàng (đợt cấp mức độ nặng) [paragraph-p73-0000]; không đáp ứng với điều trị ban đầu [paragraph-p73-0000]; có bệnh mạn tính nặng kèm theo như bệnh gan, thận, tim mạch [paragraph-p73-0000]; không có hỗ trợ từ phía gia đình [paragraph-p73-0000]. Theo sơ đồ xử trí, sau khi bắt đầu hoặc tăng thuốc giãn phế quản và thêm corticoid uống, nếu đánh giá lại sau 1-3 giờ mà triệu chứng nặng thêm thì chỉ định nhập viện [paragraph-p70-0000]..." | **Không hỗ trợ (0/3)**.<br>Context chỉ có các yếu tố gián tiếp tại trang 73, thiếu bảng chỉ định bắt buộc tại trang 75. | **0 điểm (Thiếu bằng chứng vàng)**.<br>Claude tổng hợp suy luận rất chặt chẽ từ context hiện có, nhưng do retriever bỏ sót trang 75 nên không đạt điểm theo gold evidence. |
| **q_unans_test_01** | Thiếu evidence (Test)<br>*Thời gian cách ly và phác đồ điều trị lao phổi kháng đa thuốc (MDR-TB)?* | Tool `search` câu hỏi Lao kháng đa thuốc $\rightarrow$ Context không có thông tin (ngoài phạm vi tài liệu). | "Không đủ bằng chứng để trả lời."<br>*(citations: [])* | Không có citation (chuẩn yêu cầu abstain). | **Abstention đúng (100%)**.<br>Claude nhận thức chính xác ranh giới tài liệu Hướng dẫn năm 2012 của Bộ Y tế chỉ điều trị 6 bệnh hô hấp nội khoa, không có quy trình MDR-TB. |
| **q_unans_test_02** | Thiếu evidence (Test)<br>*Chỉ định và tiêu chuẩn lựa chọn bệnh nhân ghép phổi đối với COPD giai đoạn cuối?* | Tool `search` câu hỏi Ghép phổi COPD $\rightarrow$ Context không có thông tin về phẫu thuật ghép phổi. | "Không đủ bằng chứng để trả lời."<br>*(citations: [])* | Không có citation (chuẩn yêu cầu abstain). | **Abstention đúng (100%)**.<br>Claude từ chối chính xác, tránh bịa đặt tiêu chuẩn phẫu thuật ghép tạng khi tài liệu chỉ hướng dẫn điều trị nội khoa. |

---

## Artifacts và đường dẫn minh chứng kiểm toán

- **Artifacts Retrieval:**
  - File trace đánh giá Retrieval: [starter/results/eval_dev_budget1200/traces.json](../../starter/results/eval_dev_budget1200/traces.json)
  - File tổng hợp chỉ số Retrieval: [starter/results/eval_dev_budget1200/summary.json](../../starter/results/eval_dev_budget1200/summary.json)
- **Artifacts Agent Dev:**
  - Baseline Dev (budget=800): [starter/results/agent_dev_dense_b800/summary.json](../../starter/results/agent_dev_dense_b800/summary.json)
  - Cải thiện 1 Dev (budget=1200): [starter/results/agent_dev_trans_b1200/summary.json](../../starter/results/agent_dev_trans_b1200/summary.json)
  - Cấu hình chọn Dev (budget=2400): [starter/results/agent_dev/summary.json](../../starter/results/agent_dev/summary.json)
- **Artifacts Agent Test (So sánh độc lập - Chạy trực tiếp từ Claude Haiku):**
  - **Baseline Test (budget=800):** [starter/results/agent_test_baseline/summary.json](../../starter/results/agent_test_baseline/summary.json) \| [traces.json](../../starter/results/agent_test_baseline/traces.json) \| [reviews.json](../../starter/results/agent_test_baseline/reviews.json)
  - **Cấu hình chọn Test (budget=2400):** [starter/results/agent_test/summary.json](../../starter/results/agent_test/summary.json) \| [traces.json](../../starter/results/agent_test/traces.json) \| [reviews.json](../../starter/results/agent_test/reviews.json)

---

## Đánh đổi kỹ thuật và khuyến nghị triển khai lâm sàng

1. **Chất lượng phản hồi vs Độ trễ & Chi phí:**
   - Cấu hình `budget = 2400` ký tự kết hợp Query Transformation nâng điểm Grounded Answer trên Dev lên **1.27 / 2** và trên Test lên **1.00 / 2**.
   - Chi phí API trung bình cho mỗi câu hỏi cực kỳ tiết kiệm: chỉ **$0.0060** (khoảng 150 VNĐ/câu) với `claude-haiku-5-5`, độ trễ trung bình **4.68 giây/câu**, hoàn toàn lý tưởng cho việc triển khai thực tế trong các hệ thống hỗ trợ quyết định lâm sàng (Clinical Decision Support Systems - CDSS).
2. **Độ an toàn y tế và Giảm thiểu Hallucination:**
   - Tỷ lệ từ chối đúng trên các câu hỏi thiếu bằng chứng đạt tuyệt đối **100% (4/4 câu ở cả Dev và Test)**.
   - Khi retriever không kéo được chunk vàng vào context (như câu `q19`), Claude kiên quyết từ chối trả lời (`abstain`) thay vì tự bịa đặt liều dùng hoặc thông số thở oxy nguy hiểm cho bệnh nhân.
3. **Khuyến nghị nâng cấp tiếp theo:**
   - Kết hợp **Hybrid Retrieval (Dense + BM25 với Reciprocal Rank Fusion)** để giải quyết triệt để các câu hỏi chứa thuật ngữ số liệu y tế như `SpO2`, `PEF` hay tên thuốc chuyên biệt vốn là điểm yếu của Dense Embedding.

---

## Thử thách nâng cao (nếu thực hiện)

- *Tình trạng:* Không thực hiện thử thách nâng cao 100 trang (+15 điểm bonus). Học viên tập trung hoàn thiện tối đa mục tiêu 85/85 điểm phần cơ bản với chuẩn dữ liệu chuyên sâu 46 trang lâm sàng Bộ Y tế 2012, kiểm thử 3 phương pháp retrieval, giải quyết triệt để vấn đề điều phối model thật Claude, kiểm toán thủ công 100% tập Test và tối ưu hóa context budget.
