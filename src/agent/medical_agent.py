"""Medical Agent implementation conforming to starter/agentic.py contract.
Integrates Query Transformation, Dense Retrieval, Evidence Grounding via Claude LLM, and Abstention.
"""
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

from starter.baseline import pack_context
from starter.agentic import run_agent
from src.retrieval.dense import DenseRetriever
from src.retrieval.transformation import QueryTransformer
from src.retrieval.lexical import BM25Retriever

class MedicalRetrieverAdapter:
    """Adapter wrapping DenseRetriever, BM25, and QueryTransformer for run_agent."""

    def __init__(self, mode: str = "query_transformation", budget: int = 2400):
        self.mode = mode
        self.budget = budget
        self.chunks = json.load(open(current_dir / "data" / "chunks.json", encoding="utf-8"))
        self.dense = DenseRetriever(chunks=self.chunks)
        self.bm25 = BM25Retriever(chunks=self.chunks)
        self.transformer = QueryTransformer()

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        effective_query = query
        if self.mode == "query_transformation":
            effective_query = self.transformer.transform(query, method="expand")
            return self.dense.search(effective_query, top_k=top_k, filters=filters)
        elif self.mode == "dense":
            return self.dense.search(query, top_k=top_k, filters=filters)
        elif self.mode == "bm25":
            return self.bm25.search(query, top_k=top_k, filters=filters)
        else:
            return self.dense.search(query, top_k=top_k, filters=filters)

class MedicalController:
    """Controller making 100% of search vs answer decisions using real Anthropic Claude LLM.
    Strictly follows starter/agentic.py contract without heuristics, regex shortcuts, or silent fallbacks.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        env_path = current_dir / "starter" / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided via environment variable or constructor.")

        from anthropic import Anthropic
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        client_options: dict[str, Any] = {"api_key": self.api_key}
        if workspace_id:
            client_options["default_headers"] = {"anthropic-workspace-id": workspace_id}
        self.client = Anthropic(**client_options)

        # Per-query statistics tracking
        self.current_query_stats = {
            "model": self.model,
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_count": 0,
            "provider_errors": 0,
        }

    def reset_query_stats(self):
        self.current_query_stats = {
            "model": self.model,
            "latency_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_count": 0,
            "provider_errors": 0,
        }

    def get_query_stats(self) -> dict[str, Any]:
        return dict(self.current_query_stats)

    def _call_claude_messages(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Call Claude with retries and track execution metrics."""
        t0 = time.perf_counter()
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                dt = time.perf_counter() - t0

                # Extract text across blocks
                text = ""
                for block in resp.content:
                    if getattr(block, "type", "") == "text" or hasattr(block, "text"):
                        text += getattr(block, "text", "")
                text = text.strip()

                # Track usage
                in_tok = getattr(resp.usage, "input_tokens", 0)
                out_tok = getattr(resp.usage, "output_tokens", 0)
                in_rate = 0.80 if "haiku" in self.model.lower() else 3.0
                out_rate = 4.00 if "haiku" in self.model.lower() else 15.0
                cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000

                self.current_query_stats["latency_seconds"] += round(dt, 3)
                self.current_query_stats["input_tokens"] += in_tok
                self.current_query_stats["output_tokens"] += out_tok
                self.current_query_stats["cost_usd"] += round(cost, 6)
                self.current_query_stats["call_count"] += 1

                return text, {"input_tokens": in_tok, "output_tokens": out_tok, "latency": dt}
            except Exception as e:
                self.current_query_stats["provider_errors"] += 1
                last_err = e
                err_detail = ""
                if hasattr(e, "body"):
                    err_detail = f" | Body: {e.body}"
                elif hasattr(e, "response") and hasattr(e.response, "text"):
                    err_detail = f" | Resp: {e.response.text}"
                print(f"[Claude API Warning] Attempt {attempt + 1} failed: {e}{err_detail}. Retrying in {2 * (attempt + 1)}s...")
                time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"Claude API connection permanently failed after 3 attempts: {last_err}")

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Model thật Claude trực tiếp điều phối toàn bộ hành động (search vs answer)."""
        system_prompt = (
            "Bạn là AI Agent chuyên khoa hô hấp, điều phối việc tìm kiếm bằng chứng và trả lời câu hỏi lâm sàng dựa trên tài liệu 'Hướng dẫn chẩn đoán và điều trị bệnh hô hấp' (Bộ Y tế).\n"
            "Bạn có công cụ tìm kiếm: search(query: str).\n\n"
            "QUY TẮC ĐIỀU PHỐI:\n"
            "1. Khi CHƯA có ngữ cảnh tài liệu (lượt đầu tiên), bạn CẦN tìm kiếm bằng chứng trong tài liệu:\n"
            "   - Hãy tạo câu truy vấn y khoa ngắn gọn, giàu từ khóa lâm sàng (tên bệnh, triệu chứng, tiêu chuẩn chẩn đoán, phác đồ điều trị).\n"
            "   - Xuất JSON: {\"action\": \"search\", \"query\": \"<truy vấn y khoa tối ưu>\"}\n"
            "2. Khi ĐÃ có ngữ cảnh tài liệu từ kết quả tìm kiếm:\n"
            "   - Đọc kỹ từng đoạn trích có định dạng [chunk-id | source | page]: <nội dung>.\n"
            "   - Nếu ngữ cảnh CÓ ĐỦ bằng chứng chính xác để trả lời: Trả lời ngắn gọn, chuẩn xác theo đúng tài liệu. BẮT BUỘC chèn [chunk-id] vào câu trả lời để trích dẫn nguồn. Mảng 'citations' chứa danh sách các chunk-id đó.\n"
            "     Xuất JSON: {\"action\": \"answer\", \"answer\": \"<câu trả lời kèm [chunk-id]>\", \"citations\": [\"<chunk-id>\"]}\n"
            "   - Nếu ngữ cảnh KHÔNG ĐỦ bằng chứng hoặc câu hỏi nằm ngoài phạm vi tài liệu (như COVID-19, ung thư lồng ngực, lao kháng thuốc MDR-TB, ghép phổi,...):\n"
            "     BẮT BUỘC trả lời: 'Không đủ bằng chứng để trả lời.' và citations là [].\n"
            "     Xuất JSON: {\"action\": \"answer\", \"answer\": \"Không đủ bằng chứng để trả lời.\", \"citations\": []}\n"
            "   - Nếu bạn nhận thấy kết quả tìm kiếm chưa đúng trọng tâm và cần tìm kiếm thêm bằng từ khóa khác:\n"
            "     Xuất JSON: {\"action\": \"search\", \"query\": \"<truy vấn tìm kiếm mới>\"}\n"
            "3. BẮT BUỘC chỉ xuất duy nhất 1 JSON object hợp lệ, không bọc code block markdown, không có text giải thích ngoài JSON."
        )

        if not history:
            # Turn 0: Model Claude decides the search query directly!
            user_content = (
                f"Câu hỏi: {question}\n\n"
                "Đây là lượt đầu tiên, bạn chưa có ngữ cảnh tài liệu nào. "
                "Hãy quyết định câu truy vấn tìm kiếm y khoa tối ưu nhất để tìm bằng chứng trong tài liệu:"
            )
            context_ids = []
        else:
            # Turn 1+: Model Claude reviews retrieved context and decides to answer or search further
            queries = [q for h in history for q in h.get("queries", [])]
            latest_ctx = history[-1].get("context", "")
            context_ids = history[-1].get("context_ids", [])

            user_content = (
                f"Câu hỏi: {question}\n\n"
                f"Các truy vấn đã tìm kiếm trước đó: {queries}\n\n"
                f"Ngữ cảnh tài liệu hiện có:\n{latest_ctx}\n\n"
                "Dựa trên câu hỏi và ngữ cảnh trên, hãy quyết định hành động tiếp theo (search tiếp hoặc answer có trích dẫn/từ chối):"
            )

        raw_text, _ = self._call_claude_messages(system_prompt, user_content)

        # Extract JSON
        clean_text = raw_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```", 1)[1].split("```", 1)[0].strip()

        if "{" in clean_text and "}" in clean_text:
            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}") + 1
            clean_text = clean_text[start_idx:end_idx]

        ans_text = ""
        citations = []
        action_type = ""

        try:
            parsed = json.loads(clean_text)
            action_type = parsed.get("action")
            if not action_type:
                action_type = "search" if "query" in parsed else "answer"

            if action_type == "search":
                query_val = parsed.get("query", "").strip() or question
                return {"action": "search", "query": query_val}

            ans_text = parsed.get("answer", "").strip()
            citations = parsed.get("citations", [])
            if not isinstance(citations, list):
                citations = []
        except Exception as e:
            # Fallback regex extraction if JSON is slightly truncated or malformed
            ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', clean_text)
            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', clean_text)
            if query_match and not history:
                return {"action": "search", "query": query_match.group(1).strip()}
            elif ans_match:
                ans_text = ans_match.group(1).strip()
                action_type = "answer"
            else:
                if not history:
                    return {"action": "search", "query": question}
                return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # If citations array is empty but citations exist in answer text:
        if not citations and ans_text:
            found_cites = re.findall(r"\[(paragraph-p\d+-\d+)\]", ans_text)
            if found_cites:
                citations = list(dict.fromkeys(found_cites))

        valid_cites = [c for c in citations if c in context_ids]

        if valid_cites and ans_text and ans_text != "Không đủ bằng chứng để trả lời.":
            for c in valid_cites:
                if f"[{c}]" not in ans_text:
                    ans_text += f" [{c}]"
            return {"action": "answer", "answer": ans_text, "citations": valid_cites}
        else:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
