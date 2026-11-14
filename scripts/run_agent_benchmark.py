"""Run Agent Benchmark on Dev (17 cases) and Test (7 cases) using Claude LLM.
Compares configurations and audits traces with starter/agent_metrics.py.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "starter"))

from starter.agentic import run_agent
from src.agent.medical_agent import MedicalRetrieverAdapter, MedicalController

def run_agent_evaluation(split_name: str, mode: str = "query_transformation", budget: int = 2400, output_subfolder: str | None = None):
    cases_file = current_dir / "data" / "cases.json"
    questions_file = current_dir / "data" / "questions.json"
    
    all_cases = json.loads(cases_file.read_text(encoding="utf-8"))
    q_map = {q["id"]: q for q in json.loads(questions_file.read_text(encoding="utf-8"))}
    
    cases = [c for c in all_cases if c.get("split") == split_name]
    subfolder = output_subfolder or f"agent_{split_name}"
    print(f"\n================================================================================")
    print(f"RUNNING AGENT BENCHMARK: {split_name.upper()} ({len(cases)} cases | mode={mode} | budget={budget})")
    print(f"================================================================================")

    controller = MedicalController()
    retriever = MedicalRetrieverAdapter(mode=mode, budget=budget)

    traces = []
    reviews = []

    for case in cases:
        cid = case["id"]
        question = case["question"]
        is_answerable = case["answerable"]

        filters = q_map.get(cid, {}).get("filter")
        controller.reset_query_stats()
        result = run_agent(question, controller, retriever, filters=filters, max_searches=2, top_k=3, budget=budget)
        q_stats = controller.get_query_stats()
        
        status = result["status"]
        answer = result["answer"]
        citations = result["citations"]
        events = result["events"]

        trace_entry = {
            "query_id": cid,
            "split": split_name,
            "status": status,
            "answer": answer,
            "citations": citations,
            "model": q_stats.get("model", "claude-sonnet-5-5"),
            "latency_seconds": q_stats.get("latency_seconds", 0.0),
            "input_tokens": q_stats.get("input_tokens", 0),
            "output_tokens": q_stats.get("output_tokens", 0),
            "cost_usd": q_stats.get("cost_usd", 0.0),
            "call_count": q_stats.get("call_count", 0),
            "provider_errors": q_stats.get("provider_errors", 0),
            "events": events
        }
        traces.append(trace_entry)

        # Audit review
        if is_answerable:
            gold = q_map.get(cid, {}).get("gold", [])
            gold_ev = gold[0]["evidence"] if gold else ""
            gold_page = gold[0]["page"] if gold else None

            citation_support = []
            for cite in citations:
                chunk_obj = next((c for c in retriever.dense.chunks if c["id"] == cite), None)
                if chunk_obj and chunk_obj["page"] == gold_page and gold_ev in chunk_obj["text"]:
                    citation_support.append(True)
                else:
                    citation_support.append(False)

            if status == "abstain":
                grade = 0
            else:
                if all(citation_support) and citation_support:
                    grade = 2
                elif any(citation_support):
                    grade = 1
                else:
                    grade = 0

            review_entry = {
                "query_id": cid,
                "answer_grade": grade,
                "citation_support": citation_support
            }
        else:
            is_correct_abstention = (status == "abstain")
            review_entry = {
                "query_id": cid,
                "abstention_correct": is_correct_abstention,
                "citation_support": []
            }

        reviews.append(review_entry)
        cites_str = ", ".join(citations) if citations else "none"
        lat = q_stats.get("latency_seconds", 0.0)
        tok = q_stats.get("input_tokens", 0) + q_stats.get("output_tokens", 0)
        cost = q_stats.get("cost_usd", 0.0)
        print(f"[{cid}] status: {status:8s} | answerable: {str(is_answerable):5s} | citations: {cites_str:30s} | lat: {lat:.2f}s | tok: {tok} | ${cost:.5f}")
        sys.stdout.flush()
        time.sleep(1.5)  # Gentle spacing for API rate limits

    out_dir = current_dir / "starter" / "results" / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    
    cmd = [
        sys.executable,
        str(current_dir / "starter" / "agent_metrics.py"),
        "--cases", str(cases_file),
        "--traces", str(out_dir / "traces.json"),
        "--reviews", str(out_dir / "reviews.json"),
        "--split", split_name,
        "--output", str(out_dir / "summary.json")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    summary = json.loads(res.stdout)

    # Augment summary with cost & latency metrics
    total_latency = sum(t.get("latency_seconds", 0.0) for t in traces)
    total_in_tok = sum(t.get("input_tokens", 0) for t in traces)
    total_out_tok = sum(t.get("output_tokens", 0) for t in traces)
    total_cost = sum(t.get("cost_usd", 0.0) for t in traces)
    total_calls = sum(t.get("call_count", 0) for t in traces)
    provider_errs = sum(t.get("provider_errors", 0) for t in traces)
    n_cases = len(traces)

    summary["execution_metadata"] = {
        "model": controller.model,
        "mode": mode,
        "context_budget": budget,
        "total_calls": total_calls,
        "mean_calls_per_case": round(total_calls / n_cases, 2) if n_cases else 0,
        "total_latency_seconds": round(total_latency, 3),
        "mean_latency_seconds": round(total_latency / n_cases, 3) if n_cases else 0,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "mean_tokens_per_case": round((total_in_tok + total_out_tok) / n_cases, 1) if n_cases else 0,
        "total_cost_usd": round(total_cost, 6),
        "mean_cost_usd": round(total_cost / n_cases, 6) if n_cases else 0,
        "provider_errors": provider_errs
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nMetrics Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    return summary

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "exp1", "dev_baseline"):
        print("\n>>> EXPERIMENT 1: Dev Baseline (Dense, budget=800)")
        run_agent_evaluation("dev", mode="dense", budget=800, output_subfolder="agent_dev_dense_b800")

    if target in ("all", "exp2", "dev_trans_b1200"):
        print("\n>>> EXPERIMENT 2: Dev Improved 1 (Query Transformation, budget=1200)")
        run_agent_evaluation("dev", mode="query_transformation", budget=1200, output_subfolder="agent_dev_trans_b1200")

    if target in ("all", "exp3", "dev_trans_b2400", "agent_dev"):
        print("\n>>> EXPERIMENT 3: Dev Improved 2 (Query Transformation, budget=2400) - FINAL SELECTED")
        run_agent_evaluation("dev", mode="query_transformation", budget=2400, output_subfolder="agent_dev")

    if target in ("all", "exp4", "test_baseline", "agent_test_baseline"):
        print("\n>>> EXPERIMENT 4: Test Baseline (Dense, budget=800)")
        run_agent_evaluation("test", mode="dense", budget=800, output_subfolder="agent_test_baseline")

    if target in ("all", "exp5", "test_chosen", "agent_test"):
        print("\n>>> EXPERIMENT 5: Test Chosen (Query Transformation, budget=2400)")
        run_agent_evaluation("test", mode="query_transformation", budget=2400, output_subfolder="agent_test")

    print("\n================================================================================")
    print("BENCHMARK COMPLETED FOR TARGET:", target)
    print("================================================================================")

if __name__ == "__main__":
    main()
