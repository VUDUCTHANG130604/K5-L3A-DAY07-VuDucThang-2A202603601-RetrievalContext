"""Build 24-question benchmark for Day 7:
- 20 answerable questions from Day 6 (15 dev + 5 test) with exact substring evidence
- 4 unanswerable questions (2 dev + 2 test) with verifiable absence reasons
"""
import json
import re
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"
chunks_file = data_dir / "chunks.json"
day6_q_file = Path(r"d:\AI Thực chiến\Lad 6\K5-L3A-Day06-VuDucThang-03601-DataIngestionEngineering\benchmark\questions.json")

def build_benchmark():
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    day6_questions = json.loads(day6_q_file.read_text(encoding="utf-8"))

    def find_exact_evidence(chunk_text, quote_str):
        words = [w.lower() for w in re.findall(r'\w+', quote_str)]
        if not words:
            return quote_str
        pattern = r'[\s\W]+'.join(re.escape(w) for w in words)
        match = re.search(pattern, chunk_text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    # 1. Process 20 answerable questions
    benchmark_questions = []
    cases = []

    for q in day6_questions:
        page = q["gold_page"]
        page_chunks = [c for c in chunks if c["page"] == page]
        
        exact_ev = None
        for c in page_chunks:
            sub = find_exact_evidence(c["text"], q["evidence_quote"])
            if sub and sub in c["text"]:
                exact_ev = sub
                break

        if not exact_ev:
            raise ValueError(f"Could not find exact evidence in chunks for {q['id']}")

        gold_entry = {
            "source_id": q["gold_source_id"],
            "page": page,
            "evidence": exact_ev
        }

        q_item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": [gold_entry],
            "expected_answer": q["expected_answer"]
        }
        if "filter" in q:
            q_item["filter"] = q["filter"]

        benchmark_questions.append(q_item)

        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })

    # 2. Add 4 Unanswerable questions (2 dev + 2 test)
    unanswerable = [
        {
            "id": "q_unans_dev_01",
            "split": "dev",
            "question": "Liều dùng và phác đồ sử dụng thuốc kháng virus Remdesivir trong điều trị viêm phổi do SARS-CoV-2 (COVID-19) được hướng dẫn như thế nào?",
            "expected_answer": "Không đủ bằng chứng để trả lời. Tài liệu ban hành năm 2012 nên không có hướng dẫn về SARS-CoV-2 hay thuốc Remdesivir.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu ban hành năm 2012 theo QĐ 4235/QĐ-BYT, trước khi đại dịch COVID-19 xuất hiện, nên hoàn toàn không chứa thông tin về SARS-CoV-2 hay Remdesivir."
        },
        {
            "id": "q_unans_dev_02",
            "split": "dev",
            "question": "Quy trình và kỹ thuật phẫu thuật nội soi lồng ngực cắt thùy phổi trong điều trị ung thư phổi không tế bào nhỏ được thực hiện ra sao?",
            "expected_answer": "Không đủ bằng chứng để trả lời. Tài liệu trong phạm vi khảo sát (trang 30-75) không có quy trình phẫu thuật cắt thùy phổi điều trị ung thư.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Phạm vi tài liệu trích xuất (trang 30-75) chỉ bao gồm các bệnh hô hấp nhiễm khuẩn và COPD, không chứa nội dung phẫu thuật lồng ngực hay điều trị ung thư phổi."
        },
        {
            "id": "q_unans_test_01",
            "split": "test",
            "question": "Thời gian cách ly và phác đồ điều trị bệnh lao phổi kháng đa thuốc (MDR-TB) theo chương trình chống lao quốc gia là gì?",
            "expected_answer": "Không đủ bằng chứng để trả lời. Tài liệu không có hướng dẫn về phác đồ điều trị bệnh lao phổi kháng đa thuốc MDR-TB.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Các chương từ trang 30-75 không bao gồm bệnh lao kháng đa thuốc; bệnh này thuộc hướng dẫn chuyên biệt của Chương trình Chống lao Quốc gia."
        },
        {
            "id": "q_unans_test_02",
            "split": "test",
            "question": "Chỉ định và tiêu chuẩn lựa chọn bệnh nhân ghép phổi đối với bệnh nhân COPD giai đoạn cuối theo hướng dẫn Bộ Y tế năm 2012 là gì?",
            "expected_answer": "Không đủ bằng chứng để trả lời. Tài liệu không có thông tin chỉ định phẫu thuật ghép phổi cho bệnh nhân COPD.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Trong nội dung COPD (trang 58-75), tài liệu chỉ đề cập đến điều trị nội khoa, thở oxy, thông khí hỗ trợ và phục hồi chức năng, hoàn toàn không có hướng dẫn ghép phổi."
        }
    ]

    for unans in unanswerable:
        cases.append({
            "id": unans["id"],
            "split": unans["split"],
            "question": unans["question"],
            "answerable": False,
            "expected_behavior": unans["expected_behavior"],
            "absence_reason": unans["absence_reason"]
        })

    # Save questions.json (used by retrieval benchmarks)
    q_out = data_dir / "questions.json"
    with open(q_out, "w", encoding="utf-8") as f:
        json.dump(benchmark_questions, f, ensure_ascii=False, indent=2)

    # Save cases.json (used by agent benchmarks & agent_metrics.py)
    c_out = data_dir / "cases.json"
    with open(c_out, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Created {q_out} with {len(benchmark_questions)} answerable questions.")
    print(f"Created {c_out} with {len(cases)} total cases (20 answerable + 4 unanswerable).")
    print(f"  - Dev: {sum(1 for c in cases if c['split'] == 'dev')} ({sum(1 for c in cases if c['split'] == 'dev' and c['answerable'])} answerable, {sum(1 for c in cases if c['split'] == 'dev' and not c['answerable'])} unanswerable)")
    print(f"  - Test: {sum(1 for c in cases if c['split'] == 'test')} ({sum(1 for c in cases if c['split'] == 'test' and c['answerable'])} answerable, {sum(1 for c in cases if c['split'] == 'test' and not c['answerable'])} unanswerable)")

if __name__ == "__main__":
    build_benchmark()
