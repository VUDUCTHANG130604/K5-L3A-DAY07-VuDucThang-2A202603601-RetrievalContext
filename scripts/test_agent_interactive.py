"""Interactive test script for MedicalAgent without PowerShell encoding issues."""
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / "starter" / ".env", override=True)

from src.agent.medical_agent import MedicalController, MedicalRetrieverAdapter
from starter.agentic import run_agent

def test_single_question():
    question = "Viêm phế quản cấp được định nghĩa như thế nào?"
    print("==================================================")
    print("KIỂM TRA AGENTIC RAG VỚI CLAUDE HAIKU THẬT")
    print("==================================================")
    print(f"Câu hỏi: {question}\n")

    controller = MedicalController()
    retriever = MedicalRetrieverAdapter(mode="query_transformation", budget=2400)

    print(">> Đang gọi Claude điều phối Turn 0 và truy xuất văn bản...")
    result = run_agent(question, controller, retriever, max_searches=2, top_k=3, budget=2400)

    print("\n>> KẾT QUẢ TRẢ VỀ:")
    print(f"Trạng thái (status): {result['status']}")
    print(f"Câu trả lời (answer):\n{result['answer']}")
    print(f"Trích dẫn (citations): {result['citations']}")
    print(f"\nSố events ghi nhận: {len(result['events'])}")
    for i, ev in enumerate(result['events'], 1):
        print(f"  Turn {i} ({ev['event']}): {ev.get('query') or ev.get('note', '')}")

    stats = controller.get_query_stats()
    print(f"\n>> THỐNG KÊ CHI PHÍ & ĐỘ TRỄ:")
    print(f"  - Model: {stats.get('model')}")
    print(f"  - Thời gian: {stats.get('latency_seconds')} giây")
    print(f"  - Input tokens: {stats.get('input_tokens')}")
    print(f"  - Output tokens: {stats.get('output_tokens')}")
    print(f"  - Chi phí ước tính: ${stats.get('cost_usd')} USD (~{stats.get('cost_usd') * 25400:.1f} VNĐ)")

if __name__ == "__main__":
    test_single_question()
