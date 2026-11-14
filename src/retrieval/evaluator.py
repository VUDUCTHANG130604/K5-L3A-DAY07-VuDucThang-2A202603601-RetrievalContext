"""Comprehensive Retrieval Evaluator for 3 methods:
1. Dense Retrieval (EmbeddingGemma-2)
2. Lexical Retrieval (BM25Okapi)
3. Query Transformation (Medical Expansion + Dense Retrieval)
"""
import json
import statistics
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

from starter.baseline import hit, pack_context
from src.retrieval.dense import DenseRetriever
from src.retrieval.lexical import BM25Retriever
from src.retrieval.transformation import QueryTransformer

def run_evaluation(split: str = "dev", budget: int = 1200, top_k: int = 3):
    chunks_path = current_dir / "data" / "chunks.json"
    questions_path = current_dir / "data" / "questions.json"
    
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("split") == split]

    if not questions:
        raise ValueError(f"No questions found for split: {split}")

    print(f"\n========================================================")
    print(f"EVALUATION ON SPLIT: {split.upper()} (N = {len(questions)}, budget = {budget})")
    print(f"========================================================")

    dense = DenseRetriever(chunks=chunks)
    bm25 = BM25Retriever(chunks=chunks)
    transformer = QueryTransformer()

    methods = [
        ("dense", "Dense Retrieval (EmbeddingGemma-2)"),
        ("bm25", "Lexical Retrieval (BM25)"),
        ("query_transformation", "Query Transformation (Medical Expansion + Dense)")
    ]

    all_results = {}
    all_traces = {}

    for method_key, method_name in methods:
        traces = []
        latencies = []
        hit1_count = 0
        hit3_count = 0
        context_hit_count = 0
        total_context_chars = 0

        for q in questions:
            q_id = q["id"]
            raw_query = q["question"]
            gold = q["gold"]

            effective_query = raw_query
            transformation_trace = None
            if method_key == "query_transformation":
                effective_query = transformer.transform(raw_query, method="expand")
                transformation_trace = {
                    "raw_query": raw_query,
                    "transformed_query": effective_query
                }

            start_t = time.perf_counter()
            if method_key == "bm25":
                results = bm25.search(effective_query, top_k=top_k)
            elif method_key in ("dense", "query_transformation"):
                results = dense.search(effective_query, top_k=top_k)
            else:
                raise ValueError(f"Unknown method {method_key}")

            elapsed_ms = (time.perf_counter() - start_t) * 1000
            latencies.append(elapsed_ms)

            # Context packing
            context, context_ids = pack_context(results, budget=budget)
            total_context_chars += len(context)

            # Hit checks
            marked = [{**c, "hit": hit(c, gold)} for c in results]
            is_hit1 = bool(marked and marked[0]["hit"])
            is_hit3 = any(c["hit"] for c in marked[:top_k])
            is_context_hit = any(g["evidence"] in context for g in gold)

            if is_hit1: hit1_count += 1
            if is_hit3: hit3_count += 1
            if is_context_hit: context_hit_count += 1

            trace_item = {
                "query_id": q_id,
                "split": split,
                "question": raw_query,
                "method": method_key,
                "top_k": top_k,
                "budget": budget,
                "gold": gold,
                "results": marked,
                "context": context,
                "context_ids": context_ids,
                "hit1": is_hit1,
                "hit3": is_hit3,
                "context_hit": is_context_hit,
                "latency_ms": elapsed_ms
            }
            if transformation_trace:
                trace_item["transformation"] = transformation_trace
            traces.append(trace_item)

        N = len(questions)
        summary = {
            "method": method_key,
            "method_name": method_name,
            "split": split,
            "N": N,
            "budget": budget,
            "hit1": hit1_count,
            "hit1_rate": round(hit1_count / N, 4),
            "hit3": hit3_count,
            "hit3_rate": round(hit3_count / N, 4),
            "context_hit": context_hit_count,
            "context_evidence_rate": round(context_hit_count / N, 4),
            "avg_context_chars": round(total_context_chars / N, 2),
            "median_ms": round(statistics.median(latencies), 3),
            "service_cost_usd": 0.0
        }

        all_results[method_key] = summary
        all_traces[method_key] = traces

        print(f"\n[{method_name}]")
        print(f"  Hit@1: {hit1_count}/{N} ({summary['hit1_rate']*100:.1f}%)")
        print(f"  Evidence Hit@3: {hit3_count}/{N} ({summary['hit3_rate']*100:.1f}%)")
        print(f"  Context Evidence Rate (budget {budget}): {context_hit_count}/{N} ({summary['context_evidence_rate']*100:.1f}%)")
        print(f"  Median Latency: {summary['median_ms']} ms")

    # Save results
    output_dir = current_dir / "starter" / "results" / f"eval_{split}_budget{budget}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(output_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(all_traces, f, ensure_ascii=False, indent=2)

    return all_results

if __name__ == "__main__":
    split_arg = "dev"
    b_arg = 1200
    if len(sys.argv) > 1:
        split_arg = sys.argv[1]
    if len(sys.argv) > 2:
        b_arg = int(sys.argv[2])
    run_evaluation(split=split_arg, budget=b_arg)
