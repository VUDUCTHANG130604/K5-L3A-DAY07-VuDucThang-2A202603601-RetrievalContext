  # Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

## Giao diện benchmark

Mở dashboard local để xem kết quả retrieval, metrics agent và trace theo từng câu hỏi:

```bash
python scripts/serve_dashboard.py
```

Sau đó truy cập `http://127.0.0.1:8765`. Dashboard chỉ đọc dữ liệu trong `data/` và `starter/results/`.


---

## Kết quả và Hướng dẫn chạy nhanh (Bản nộp của Vũ Đức Thắng - 2A202603601)

### 1. Báo cáo cá nhân
Xem báo cáo đầy đủ số liệu tại: [report/individual/2A202603601_VuDucThang.md](report/individual/2A202603601_VuDucThang.md)

### 2. Các lệnh chạy tái lập thực nghiệm

* **Chạy kiểm tra toàn bộ 15 Unit Tests (chạy offline):**
  ```bash
  python -m unittest discover -s tests -v
  ```

* **Đánh giá 3 phương pháp Retrieval (Dense, BM25, Query Transformation - chạy offline):**
  ```bash
  python src/retrieval/evaluator.py dev 1200
  python src/retrieval/evaluator.py test 1200
  ```

* **Tái lập Artifacts Baseline Test (Dense top-k=3, budget=800 - chạy offline):**
  ```bash
  python scripts/generate_test_baseline.py
  ```

* **Demo so sánh trực diện 3 phương pháp retrieval trên 1 câu hỏi y khoa (chạy offline):**
  ```bash
  python scripts/demo_quick_test.py
  ```

* **Chạy Agent Benchmark với Claude Model thật (Yêu cầu API Key):**
  Tạo file `starter/.env` từ `starter/.env.example`, điền `ANTHROPIC_API_KEY` của bạn rồi chạy:
  ```bash
  python scripts/run_agent_benchmark.py
  ```
